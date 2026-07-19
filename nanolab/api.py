"""The nanolab web app: a read-only JSON API over the lab db + the UI shell.

`nanolab ui` serves this with uvicorn. The API computes nothing the CLI
doesn't — every number comes from the same SQLite rows, so UI-vs-CLI
equality is trivially true. The UI lives in nanolab/ui/ (no build step).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from . import db

UI_DIR = Path(__file__).parent / "ui"

# ── background jobs (UI-triggered actions) ──────────────────────────────────
JOBS: list[dict] = []
_job_ids = itertools.count(1)


def _start_job(kind: str, label: str, target) -> dict:
    job = {
        "id": next(_job_ids),
        "kind": kind,
        "label": label,
        "status": "running",
        "error": None,
        "started_at": db.utcnow(),
    }
    JOBS.insert(0, job)
    del JOBS[20:]

    def run():
        try:
            target()
            job["status"] = "done"
        except BaseException as exc:  # surfaced in the UI, not swallowed
            job["status"] = "failed"
            job["error"] = str(exc)[:400]

    threading.Thread(target=run, daemon=True).start()
    return job


def _run_subprocess(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(" · ".join(tail) or f"exit code {proc.returncode}")


def _defaults() -> dict:
    from . import config as config_mod

    config_mod.load_dotenv()
    key_var = os.environ.get("NANOLAB_API_KEY_VAR", "")
    return {
        "model": os.environ.get("NANOLAB_DEFAULT_MODEL", ""),
        "base_url": os.environ.get("NANOLAB_API_BASE_URL", ""),
        "key_var": key_var,
        "key_present": bool(key_var and os.environ.get(key_var)),
    }


def _rows(cursor) -> list[dict]:
    return [dict(r) for r in cursor.fetchall()]


def _overview() -> dict:
    conn = db.connect()
    try:
        evals = conn.execute(
            "SELECT status, COUNT(*) AS n FROM eval_runs GROUP BY status"
        ).fetchall()
        eval_counts = {r["status"]: r["n"] for r in evals}
        trains = conn.execute(
            "SELECT status, COUNT(*) AS n FROM train_runs GROUP BY status"
        ).fetchall()
        train_counts = {r["status"]: r["n"] for r in trains}
        best = conn.execute(
            """
            SELECT e.mean_reward, e.model, v.slug FROM eval_runs e
            JOIN environments v ON v.id = e.env_id
            WHERE e.status='done' AND e.mean_reward IS NOT NULL
            ORDER BY e.mean_reward DESC LIMIT 1
            """
        ).fetchone()
        tokens = conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens+completion_tokens),0) AS t FROM ledger"
        ).fetchone()["t"]
        recent_evals = _rows(
            conn.execute(
                """
                SELECT e.id, e.model, e.status, e.mean_reward, e.started_at, v.slug
                FROM eval_runs e JOIN environments v ON v.id = e.env_id
                ORDER BY e.id DESC LIMIT 8
                """
            )
        )
        recent_trains = _rows(
            conn.execute(
                """
                SELECT t.id, t.model, t.status, t.steps_completed,
                       t.reward_curve_json, v.slug
                FROM train_runs t LEFT JOIN environments v ON v.id = t.env_id
                ORDER BY t.id DESC LIMIT 8
                """
            )
        )
        n_envs = conn.execute("SELECT COUNT(*) AS n FROM environments").fetchone()["n"]
        n_adapters = conn.execute("SELECT COUNT(*) AS n FROM adapters").fetchone()["n"]
    finally:
        conn.close()
    for t in recent_trains:
        curve = json.loads(t.pop("reward_curve_json") or "[]")
        t["rewards"] = [p["reward"] for p in curve]
    return {
        "evals": {
            "active": eval_counts.get("running", 0) + eval_counts.get("pending", 0),
            "done": eval_counts.get("done", 0),
            "total": sum(eval_counts.values()),
        },
        "training": {
            "active": train_counts.get("running", 0),
            "done": train_counts.get("done", 0),
            "total": sum(train_counts.values()),
        },
        "best": dict(best) if best else None,
        "tokens": tokens,
        "environments": n_envs,
        "adapters": n_adapters,
        "recent_evals": recent_evals,
        "recent_trains": recent_trains,
    }


def _environments() -> list[dict]:
    from . import envs as envs_mod

    installed = envs_mod.list_installed()
    conn = db.connect()
    try:
        for env in installed:
            row = conn.execute(
                """
                SELECT e.mean_reward, e.model FROM eval_runs e
                JOIN environments v ON v.id = e.env_id
                WHERE v.slug = ? AND e.status='done' AND e.mean_reward IS NOT NULL
                ORDER BY e.mean_reward DESC LIMIT 1
                """,
                (env["slug"],),
            ).fetchone()
            env["best_reward"] = row["mean_reward"] if row else None
            env["best_model"] = row["model"] if row else None
    finally:
        conn.close()
    return installed


def _environment_detail(env_id: str) -> dict | None:
    """Everything the repo-style page needs: metadata, README, deps, source
    files, and this environment's leaderboard — all from the local install."""
    from importlib import metadata as im
    from importlib.util import find_spec

    conn = db.connect()
    try:
        row = db.get_environment(conn, env_id)
        if row is None:
            return None
        leaderboard = _rows(
            conn.execute(
                """
                SELECT e.id, e.model, e.status, e.mean_reward, e.num_examples,
                       e.rollouts_per_example, e.finished_at
                FROM eval_runs e JOIN environments v ON v.id = e.env_id
                WHERE v.slug = ? AND e.status = 'done'
                ORDER BY e.mean_reward DESC, e.id DESC
                """,
                (row["slug"],),
            )
        )
    finally:
        conn.close()

    out: dict = {
        "slug": row["slug"],
        "env_id": row["env_id"],
        "version": row["version"],
        "installed_at": row["installed_at"],
        "leaderboard": leaderboard,
        "summary": None,
        "readme": None,
        "requires": [],
        "requires_python": None,
        "files": [],
    }
    try:
        dist = im.distribution(row["env_id"])
        meta = dist.metadata
        out["summary"] = meta.get("Summary")
        out["requires"] = meta.get_all("Requires-Dist") or []
        out["requires_python"] = meta.get("Requires-Python")
        body = meta.get_payload()
        out["readme"] = body.strip() if body and body.strip() else meta.get("Description")
    except Exception:
        pass
    try:
        spec = find_spec(row["env_id"].replace("-", "_"))
        if spec and spec.origin:
            origin = Path(spec.origin)
            paths = (
                sorted(origin.parent.glob("*.py"))
                if origin.name == "__init__.py"
                else [origin]
            )
            for p in paths[:8]:
                text = p.read_text(errors="replace")
                out["files"].append({"name": p.name, "content": text[:40_000]})
    except Exception:
        pass
    return out


def _evals() -> list[dict]:
    conn = db.connect()
    try:
        runs = _rows(
            conn.execute(
                """
                SELECT e.id, e.model, e.status, e.mean_reward, e.num_examples,
                       e.rollouts_per_example, e.seed, e.started_at, e.finished_at,
                       v.slug AS env
                FROM eval_runs e JOIN environments v ON v.id = e.env_id
                ORDER BY e.id DESC
                """
            )
        )
        for run in runs:
            stats = conn.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN metrics_json LIKE '%\"_error\"%'"
                " THEN 1 ELSE 0 END) AS errs FROM samples WHERE eval_run_id = ?",
                (run["id"],),
            ).fetchone()
            run["samples"] = stats["n"] or 0
            run["errors"] = stats["errs"] or 0
    finally:
        conn.close()
    return runs


def _eval_detail(run_id: int) -> dict | None:
    conn = db.connect()
    try:
        run = conn.execute(
            """
            SELECT e.*, v.slug AS env FROM eval_runs e
            JOIN environments v ON v.id = e.env_id WHERE e.id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            return None
        samples = _rows(
            conn.execute(
                "SELECT example_index, rollout_index, prompt_json, completion_json,"
                " reward, metrics_json FROM samples WHERE eval_run_id = ?"
                " ORDER BY example_index, rollout_index",
                (run_id,),
            )
        )
    finally:
        conn.close()
    out = dict(run)
    out["meta"] = json.loads(out.pop("metrics_json") or "{}")
    out.pop("params_json", None)
    out["env"] = run["env"]
    parsed = []
    for s in samples:
        parsed.append(
            {
                "example": s["example_index"],
                "rollout": s["rollout_index"],
                "reward": s["reward"],
                "metrics": json.loads(s["metrics_json"] or "{}"),
                "prompt": json.loads(s["prompt_json"] or "null"),
                "completion": json.loads(s["completion_json"] or "null"),
            }
        )
    out["rollouts"] = parsed
    return out


def _training() -> list[dict]:
    conn = db.connect()
    try:
        runs = _rows(
            conn.execute(
                """
                SELECT t.id, t.model, t.status, t.steps_completed,
                       t.reward_curve_json, t.started_at, t.finished_at,
                       v.slug AS env
                FROM train_runs t LEFT JOIN environments v ON v.id = t.env_id
                ORDER BY t.id DESC
                """
            )
        )
    finally:
        conn.close()
    for run in runs:
        curve = json.loads(run.pop("reward_curve_json") or "[]")
        run["rewards"] = [p["reward"] for p in curve]
    return runs


def _training_detail(run_id: int) -> dict | None:
    conn = db.connect()
    try:
        run = conn.execute(
            """
            SELECT t.*, v.slug AS env FROM train_runs t
            LEFT JOIN environments v ON v.id = t.env_id WHERE t.id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            return None
        adapters = _rows(
            conn.execute(
                "SELECT id, base_model, step, path, created_at FROM adapters"
                " WHERE train_run_id = ? ORDER BY step",
                (run_id,),
            )
        )
    finally:
        conn.close()
    out = dict(run)
    curve = json.loads(out.pop("reward_curve_json") or "[]")
    out["curve"] = curve
    out["adapters"] = adapters
    return out


def _deployments() -> list[dict]:
    from . import serve as serve_mod

    return [vars(d) for d in serve_mod.list_deployments()]


def _adapters() -> list[dict]:
    from . import serve as serve_mod

    live = {
        d.adapter_id: d for d in serve_mod.list_deployments() if d.status == "running"
    }
    conn = db.connect()
    try:
        rows = _rows(
            conn.execute(
                """
                SELECT a.*, t.status AS run_status, v.slug AS env
                FROM adapters a
                LEFT JOIN train_runs t ON t.id = a.train_run_id
                LEFT JOIN environments v ON v.id = t.env_id
                ORDER BY a.id DESC
                """
            )
        )
    finally:
        conn.close()
    for row in rows:
        dep = live.get(row["id"])
        row["deployed"] = bool(dep)
        row["endpoint"] = dep.endpoint if dep else None
        row["exists"] = Path(row["path"]).exists()
    return rows


def build_app():
    from starlette.applications import Starlette
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    def endpoint(fn):
        async def handler(request):
            kwargs = dict(request.path_params)
            data = fn(**kwargs)
            if data is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            return JSONResponse(data)

        return handler

    async def index(request):
        return FileResponse(UI_DIR / "index.html")

    async def jobs(request):
        return JSONResponse(JOBS)

    async def action_eval(request):
        body = await request.json()
        env = (body.get("env") or "").strip()
        defaults = _defaults()
        model = (body.get("model") or "").strip() or defaults["model"]
        base_url = defaults["base_url"]
        key_var = defaults["key_var"] or "OPENAI_API_KEY"
        if not env:
            return JSONResponse({"error": "pick an environment"}, status_code=400)
        if not model:
            return JSONResponse(
                {"error": "no model — set NANOLAB_DEFAULT_MODEL in .env or type one"},
                status_code=400,
            )
        if not base_url or not os.environ.get(key_var):
            return JSONResponse(
                {"error": "no API endpoint/key configured — set NANOLAB_API_BASE_URL "
                 "and NANOLAB_API_KEY_VAR (+ the key) in the repo's .env"},
                status_code=400,
            )
        n = int(body.get("n") or 5)
        r = int(body.get("r") or 1)
        temperature = body.get("temperature")

        # evals run as a subprocess of the CLI, not in a thread: environment
        # rubrics may install signal handlers (main-thread-only), and this
        # keeps exactly one code path for evals however they're triggered
        cmd = [
            sys.executable, "-m", "nanolab.cli", "eval", "run", env,
            "-m", model, "-b", base_url, "-k", key_var,
            "-n", str(n), "-r", str(r), "-c", "2", "--force",
        ]
        if temperature not in (None, ""):
            cmd += ["-T", str(temperature)]

        def run_eval():
            _run_subprocess(cmd)

        return JSONResponse({"job": _start_job("eval", f"eval · {env} · {model}", run_eval)})

    async def action_deploy(request):
        body = await request.json()
        adapter_id = body.get("adapter_id")
        if not adapter_id:
            return JSONResponse({"error": "adapter_id required"}, status_code=400)

        def run_deploy():
            from .policy_server import _free_port
            from . import serve as serve_mod

            serve_mod.create_deployment(
                int(adapter_id), port=_free_port(), ready_timeout=600, local=True
            )

        return JSONResponse(
            {"job": _start_job("deploy", f"deploy · adapter #{adapter_id} · local", run_deploy)}
        )

    async def action_stop_deployment(request):
        body = await request.json()
        dep_id = body.get("deployment_id")
        if not dep_id:
            return JSONResponse({"error": "deployment_id required"}, status_code=400)
        from . import serve as serve_mod

        try:
            dep = serve_mod.stop_deployment(int(dep_id))
        except serve_mod.ServeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"stopped": dep.id})

    async def action_install(request):
        body = await request.json()
        slug = (body.get("slug") or "").strip()
        if not slug:
            return JSONResponse({"error": "environment name required"}, status_code=400)

        def run_install():
            from . import envs

            envs.install(slug)

        return JSONResponse({"job": _start_job("install", f"install · {slug}", run_install)})

    return Starlette(
        routes=[
            Route("/api/overview", endpoint(_overview)),
            Route("/api/environments", endpoint(_environments)),
            Route("/api/environments/{env_id:str}", endpoint(_environment_detail)),
            Route("/api/evals", endpoint(_evals)),
            Route("/api/evals/{run_id:int}", endpoint(_eval_detail)),
            Route("/api/training", endpoint(_training)),
            Route("/api/training/{run_id:int}", endpoint(_training_detail)),
            Route("/api/deployments", endpoint(_deployments)),
            Route("/api/adapters", endpoint(_adapters)),
            Route("/api/defaults", endpoint(_defaults)),
            Route("/api/jobs", jobs),
            Route("/api/actions/eval", action_eval, methods=["POST"]),
            Route("/api/actions/install", action_install, methods=["POST"]),
            Route("/api/actions/deploy", action_deploy, methods=["POST"]),
            Route("/api/actions/stop-deployment", action_stop_deployment, methods=["POST"]),
            Mount("/assets", StaticFiles(directory=UI_DIR), name="assets"),
            Route("/", index),
            Route("/{path:path}", index),  # hash-router fallback
        ]
    )


def serve_ui(port: int = 3456, open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    print(f"nanolab ui → {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=port, log_level="warning")
