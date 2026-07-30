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


def ui_version() -> str:
    """Newest mtime across UI files — changes whenever the UI code changes."""
    newest = max(
        (p.stat().st_mtime for p in UI_DIR.rglob("*") if p.is_file()), default=0
    )
    return str(int(newest))

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


def _hub(search: str = "", sort: str = "stars", page: int = 1) -> dict:
    """Browse the Prime Hub via the prime CLI (real listing, live)."""
    installed = set()
    conn = db.connect()
    try:
        for row in conn.execute("SELECT slug FROM environments").fetchall():
            installed.add(row["slug"])
    finally:
        conn.close()

    cmd = ["prime", "env", "list", "--output", "json", "--sort", sort,
           "--num", "48", "--page", str(page)]
    if search:
        cmd += ["--search", search]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return {"error": "hub unreachable — is the prime CLI installed?", "environments": []}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "could not parse hub response", "environments": []}
    envs = data.get("environments", [])
    for e in envs:
        e["installed"] = e.get("environment") in installed
    return {"environments": envs, "page": page}


# hub detail pages hit the network through the prime CLI (seconds per call) —
# cache aggressively; hub content changes rarely
_HUB_CACHE: dict[str, object] = {}


def _prime_inspect(slug: str, path: str = "") -> str:
    """Raw `prime env inspect` output for a hub env (file listing or one file)."""
    key = f"inspect:{slug}:{path}"
    if key in _HUB_CACHE:
        return _HUB_CACHE[key]  # type: ignore[return-value]
    cmd = ["prime", "env", "inspect", slug] + ([path] if path else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "inspect failed").strip()[-200:])
    out = proc.stdout
    _HUB_CACHE[key] = out
    return out


def _hub_files(slug: str) -> list[dict]:
    """Parse the inspect table into a plain file list (dirs excluded)."""
    import re

    files = []
    for line in _prime_inspect(slug).splitlines():
        m = re.match(r"\s*│\s*(file|dir)\s*│\s*(\S+)\s*│\s*([\d.]+\s*\w+|-)\s*│", line)
        if m and m.group(1) == "file":
            files.append({"name": m.group(2), "size": m.group(3)})
    return files


def _hub_file_content(slug: str, path: str) -> str:
    """One file's contents; inspect prints a short header before the body."""
    out = _prime_inspect(slug, path)
    lines = out.splitlines()
    # skip the echo header (blank / slug@version / path) the CLI prepends
    start = 0
    for i, line in enumerate(lines[:6]):
        if line.strip() == path or line.strip().endswith(f"@latest"):
            start = i + 1
    return "\n".join(lines[start:]).strip()


def _hub_detail(owner: str, name: str) -> dict:
    """GitHub-style detail for a hub env, no install required."""
    slug = f"{owner}/{name}"
    conn = db.connect()
    try:
        installed = conn.execute(
            "SELECT 1 FROM environments WHERE slug=?", (slug,)
        ).fetchone() is not None
    finally:
        conn.close()
    listing = _hub(search=name)  # carries description/stars/tags/version
    meta = next(
        (e for e in listing.get("environments", []) if e.get("environment") == slug),
        {},
    )
    files = _hub_files(slug)
    readme = ""
    if any(f["name"].lower() == "readme.md" for f in files):
        try:
            readme = _hub_file_content(slug, "README.md")
        except RuntimeError:
            readme = ""
    return {
        "slug": slug,
        "installed": installed,
        "description": meta.get("description", ""),
        "stars": meta.get("stars"),
        "tags": meta.get("tags", []),
        "version": meta.get("version", ""),
        "files": files,
        "readme": readme,
    }


def _configs() -> list[dict]:
    """Training configs available to launch, with a parsed summary."""
    import tomllib

    out = []
    for path in sorted(Path("configs").glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text())
            envs_ = data.get("env", [{}])
            out.append({
                "path": str(path),
                "name": path.stem,
                "model": data.get("model", "?"),
                "env": envs_[0].get("id", "?") if envs_ else "?",
                "max_steps": data.get("max_steps"),
                "learning_rate": data.get("learning_rate"),
            })
        except Exception:
            continue
    return out


def _cloud_runs() -> list[dict]:
    from . import cloud as cloud_mod

    return cloud_mod.list_runs()


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


def _episode_turns(run_id: int, example: int = 0, rollout: int = 0) -> dict:
    """One multi-turn episode as a timeline of notebook rewrites.

    For memory/scribe environments the assistant turns ARE the notebook, so
    consecutive turns diff line-by-line into kept / added / dropped. That
    makes the trained model's judgment visible instead of asserted.
    """
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT s.prompt_json, s.completion_json, s.reward, e.model,"
            " v.slug AS env FROM samples s JOIN eval_runs e ON e.id = s.eval_run_id"
            " JOIN environments v ON v.id = e.env_id WHERE s.eval_run_id = ?"
            " AND s.example_index = ? AND s.rollout_index = ?",
            (run_id, example, rollout)).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": "no such rollout"}

    def text(m):
        c = m.get("content") if isinstance(m, dict) else None
        return c if isinstance(c, str) else json.dumps(c)

    prompt = json.loads(row["prompt_json"] or "[]")
    completion = json.loads(row["completion_json"] or "[]")
    msgs = [m for m in (prompt + completion) if isinstance(m, dict)]

    turns, prev_lines = [], []
    pending_input = ""
    for m in msgs:
        role = m.get("role")
        if role in ("user", "system"):
            if role == "user":
                pending_input = text(m)
            continue
        if role != "assistant":
            continue
        nb = text(m) or ""
        lines = [ln.strip() for ln in nb.splitlines() if ln.strip()]
        prev_set, cur_set = set(prev_lines), set(lines)
        turns.append({
            "turn": len(turns) + 1,
            "input": pending_input,
            "notebook": nb,
            "chars": len(nb),
            "kept": [ln for ln in lines if ln in prev_set],
            "added": [ln for ln in lines if ln not in prev_set],
            "dropped": [ln for ln in prev_lines if ln not in cur_set],
        })
        prev_lines = lines
        pending_input = ""
    return {"run_id": run_id, "example": example, "rollout": rollout,
            "model": row["model"], "env": row["env"], "reward": row["reward"],
            "turns": turns}


def _eval_compare(run_a: int, run_b: int) -> dict:
    """Two eval runs side by side, paired per example — the A/B view.

    Pairing is by example_index (same dataset order = same task), so a diff
    row is genuinely the same question answered by two models. Nothing is
    recomputed: both sides are read from stored sample rows.
    """
    conn = db.connect()
    try:
        runs = {}
        for rid in (run_a, run_b):
            row = conn.execute(
                "SELECT e.*, v.slug AS env FROM eval_runs e JOIN environments v"
                " ON v.id = e.env_id WHERE e.id = ?", (rid,)).fetchone()
            if row is None:
                return {"error": f"no eval run #{rid}"}
            runs[rid] = dict(row)
        samples = {}
        for rid in (run_a, run_b):
            samples[rid] = {
                (s["example_index"], s["rollout_index"]): s["reward"]
                for s in conn.execute(
                    "SELECT example_index, rollout_index, reward FROM samples"
                    " WHERE eval_run_id = ?", (rid,)).fetchall()
            }
    finally:
        conn.close()

    keys = sorted(set(samples[run_a]) | set(samples[run_b]))
    rows, better, worse, same = [], 0, 0, 0
    for ex, ro in keys:
        a = samples[run_a].get((ex, ro))
        b = samples[run_b].get((ex, ro))
        delta = None if a is None or b is None else b - a
        if delta is not None:
            if delta > 1e-9:
                better += 1
            elif delta < -1e-9:
                worse += 1
            else:
                same += 1
        rows.append({"example": ex, "rollout": ro, "a": a, "b": b, "delta": delta})

    def head(rid):
        r = runs[rid]
        return {"id": r["id"], "model": r["model"], "env": r["env"],
                "mean_reward": r["mean_reward"], "num_examples": r["num_examples"],
                "rollouts_per_example": r["rollouts_per_example"],
                "status": r["status"], "started_at": r["started_at"]}

    ma, mb = runs[run_a]["mean_reward"], runs[run_b]["mean_reward"]
    return {
        "a": head(run_a), "b": head(run_b),
        "same_env": runs[run_a]["env"] == runs[run_b]["env"],
        "delta": None if ma is None or mb is None else mb - ma,
        "counts": {"better": better, "worse": worse, "same": same},
        "rows": rows,
    }


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

    def endpoint(fn):
        async def handler(request):
            kwargs = dict(request.path_params)
            data = fn(**kwargs)
            if data is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            return JSONResponse(data)

        return handler

    NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}

    async def index(request):
        return FileResponse(UI_DIR / "index.html", headers=NO_CACHE)

    async def asset(request):
        name = request.path_params["name"]
        target = (UI_DIR / name).resolve()
        # never serve outside the ui dir
        if UI_DIR.resolve() not in target.parents or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(target, headers=NO_CACHE)

    async def hub(request):
        q = request.query_params
        try:
            data = _hub(
                search=q.get("search", ""),
                sort=q.get("sort", "stars"),
                page=int(q.get("page", 1)),
            )
        except Exception as exc:
            data = {"error": str(exc)[:200], "environments": []}
        return JSONResponse(data)

    async def hub_detail(request):
        p = request.path_params
        try:
            return JSONResponse(_hub_detail(p["owner"], p["name"]))
        except Exception as exc:
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)

    async def hub_file(request):
        p = request.path_params
        path = request.query_params.get("path", "")
        if not path or "/" in path or path.startswith("."):
            return JSONResponse({"error": "bad path"}, status_code=400)
        try:
            content = _hub_file_content(f'{p["owner"]}/{p["name"]}', path)
            return JSONResponse({"path": path, "content": content[:200_000]})
        except Exception as exc:
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)

    async def jobs(request):
        return JSONResponse(JOBS)

    async def version(request):
        return JSONResponse({"ui": ui_version()})

    async def action_cli(request):
        """Run a nanolab CLI command from the UI console. Never a shell —
        args are split and passed to our own CLI module only."""
        import shlex

        body = await request.json()
        raw = (body.get("command") or "").strip()
        if not raw:
            return JSONResponse({"error": "type a command"}, status_code=400)
        try:
            args = shlex.split(raw)
        except ValueError as exc:
            return JSONResponse({"error": f"can't parse: {exc}"}, status_code=400)
        # tolerate people typing the full incantation
        while args and args[0] in ("uv", "run", "nanolab", "$"):
            args.pop(0)
        if not args:
            return JSONResponse({"error": "type a command, e.g. `eval list`"}, status_code=400)
        if args[0] == "ui":
            return JSONResponse(
                {"error": "`ui` starts this very server — it can't run inside itself"},
                status_code=400,
            )

        cmd = [sys.executable, "-m", "nanolab.cli", *args]

        def run_cli(job=None):
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            out = ((proc.stdout or "") + (proc.stderr or "")).strip()
            run_cli.output = out[-8000:] or "(no output)"
            if proc.returncode != 0:
                raise RuntimeError(run_cli.output.splitlines()[-1] if run_cli.output else "failed")

        job = _start_job("cli", f"$ nanolab {' '.join(args)}", run_cli)
        job["command"] = raw

        # attach output to the job once the thread finishes
        def attach():
            import time as _t

            for _ in range(3600):
                if job["status"] != "running":
                    job["output"] = getattr(run_cli, "output", None)
                    return
                _t.sleep(0.5)

        threading.Thread(target=attach, daemon=True).start()
        return JSONResponse({"job": job})

    # read-only nanolab commands the Console page may run
    CONSOLE_ALLOWED = {
        ("env", "list"), ("eval", "list"), ("eval", "show"), ("eval", "compare"),
        ("training", "list"), ("training", "show"), ("deployments", "list"),
        ("cloud", "list"), ("cloud", "status"), ("instrument",), ("version",),
        ("report",),
    }

    async def action_cli(request):
        body = await request.json()
        raw = (body.get("command") or "").strip()
        parts = raw.split()
        # tolerate the prefixes people naturally type
        while parts and parts[0] in ("uv", "run", "nanolab", "$"):
            parts.pop(0)
        allowed = parts and any(
            tuple(parts[: len(sig)]) == sig for sig in CONSOLE_ALLOWED
        )
        if not allowed:
            cmds = sorted({" ".join(s) for s in CONSOLE_ALLOWED})
            return JSONResponse(
                {"error": "the console runs read-only nanolab commands: "
                 + " · ".join(cmds)},
                status_code=400,
            )
        job = {
            "id": next(_job_ids), "kind": "cli",
            "label": f"$ nanolab {' '.join(parts)}",
            "status": "running", "error": None, "output": None,
            "started_at": db.utcnow(),
        }
        JOBS.insert(0, job)
        del JOBS[20:]

        def run_cli():
            proc = subprocess.run(
                [sys.executable, "-m", "nanolab.cli", *parts],
                capture_output=True, text=True, timeout=300,
            )
            job["output"] = (proc.stdout + proc.stderr).strip() or "(no output)"
            if proc.returncode != 0:
                raise RuntimeError(job["output"][-300:])

        def wrapper():
            try:
                run_cli()
                job["status"] = "done"
            except BaseException as exc:
                job["status"] = "failed"
                job["error"] = str(exc)[:400]

        threading.Thread(target=wrapper, daemon=True).start()
        return JSONResponse({"job": job})

    async def action_train_cloud(request):
        body = await request.json()
        config = (body.get("config") or "").strip()
        target = Path(config)
        if (
            not config
            or target.suffix != ".toml"
            or not target.is_file()
            or Path("configs").resolve() not in target.resolve().parents
        ):
            return JSONResponse(
                {"error": "pick a training config from configs/"}, status_code=400
            )

        def run_push():
            from . import cloud as cloud_mod

            cloud_mod.push(config)

        return JSONResponse(
            {"job": _start_job("train", f"training → Kaggle · {target.stem}", run_push)}
        )

    async def action_chat(request):
        """Proxy a chat turn to a running local deployment (avoids CORS)."""
        import httpx

        body = await request.json()
        dep_id = body.get("deployment_id")
        messages = body.get("messages") or []
        if not dep_id or not messages:
            return JSONResponse({"error": "deployment_id and messages required"}, status_code=400)
        from . import serve as serve_mod

        dep = next((d for d in serve_mod.list_deployments()
                    if d.id == int(dep_id) and d.status == "running"), None)
        if dep is None:
            return JSONResponse({"error": "that deployment is not running"}, status_code=400)
        payload = {
            "model": dep.served_name,
            "messages": messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 400),
        }
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{dep.endpoint}/chat/completions", json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"model did not answer: {exc}"}, status_code=502)
        return JSONResponse({"content": content})

    async def action_dismiss_job(request):
        body = await request.json()
        jid = body.get("id")
        JOBS[:] = [j for j in JOBS if j["id"] != jid]
        return JSONResponse({"dismissed": jid})

    # ---- the Memory Agent (v0.3): amnesiac chat + scribe-maintained notebook

    async def agent_state(request):
        from starlette.concurrency import run_in_threadpool

        from . import agent as agent_mod

        return JSONResponse(await run_in_threadpool(agent_mod.current_session))

    async def action_agent_chat(request):
        from starlette.concurrency import run_in_threadpool

        from . import agent as agent_mod

        body = await request.json()
        try:
            session = await run_in_threadpool(
                agent_mod.chat_turn, body.get("message") or ""
            )
        except agent_mod.AgentError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"turn failed: {exc}"}, status_code=502)
        return JSONResponse(session)

    async def action_agent_writer(request):
        from . import agent as agent_mod

        body = await request.json()
        try:
            return JSONResponse(agent_mod.set_writer(body.get("writer") or ""))
        except agent_mod.AgentError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def action_agent_reset(request):
        from . import agent as agent_mod

        return JSONResponse(agent_mod.reset_session())

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
            Route("/api/hub", hub),
            Route("/api/hub/{owner:str}/{name:str}", hub_detail),
            Route("/api/hub/{owner:str}/{name:str}/file", hub_file),
            # :path, not :str — installed slugs contain slashes (owner/name),
            # and uvicorn decodes %2F back to / before routing
            Route("/api/environments/{env_id:path}", endpoint(_environment_detail)),
            Route("/api/evals", endpoint(_evals)),
            Route("/api/evals/compare/{run_a:int}/{run_b:int}",
                  endpoint(_eval_compare)),
            Route("/api/evals/{run_id:int}/episode/{example:int}/{rollout:int}",
                  endpoint(_episode_turns)),
            Route("/api/evals/{run_id:int}", endpoint(_eval_detail)),
            Route("/api/training", endpoint(_training)),
            Route("/api/training/{run_id:int}", endpoint(_training_detail)),
            Route("/api/deployments", endpoint(_deployments)),
            Route("/api/adapters", endpoint(_adapters)),
            Route("/api/defaults", endpoint(_defaults)),
            Route("/api/configs", endpoint(_configs)),
            Route("/api/cloud", endpoint(_cloud_runs)),
            Route("/api/actions/train-cloud", action_train_cloud, methods=["POST"]),
            Route("/api/actions/cli", action_cli, methods=["POST"]),
            Route("/api/jobs", jobs),
            Route("/api/version", version),
            Route("/api/actions/cli", action_cli, methods=["POST"]),
            Route("/api/actions/chat", action_chat, methods=["POST"]),
            Route("/api/actions/dismiss-job", action_dismiss_job, methods=["POST"]),
            Route("/api/agent", agent_state),
            Route("/api/actions/agent-chat", action_agent_chat, methods=["POST"]),
            Route("/api/actions/agent-writer", action_agent_writer, methods=["POST"]),
            Route("/api/actions/agent-reset", action_agent_reset, methods=["POST"]),
            Route("/api/actions/eval", action_eval, methods=["POST"]),
            Route("/api/actions/install", action_install, methods=["POST"]),
            Route("/api/actions/deploy", action_deploy, methods=["POST"]),
            Route("/api/actions/stop-deployment", action_stop_deployment, methods=["POST"]),
            Route("/assets/{name:path}", asset),
            Route("/", index),
            Route("/{path:path}", index),  # hash-router fallback
        ]
    )


def _cloud_poller(interval: float = 120.0) -> None:
    """Background: advance cloud runs; auto-merge finished ones into the lab."""
    import time

    while True:
        try:
            from . import cloud as cloud_mod

            for event in cloud_mod.poll_once():
                JOBS.insert(0, {
                    "id": next(_job_ids), "kind": "cloud", "label": event,
                    "status": "failed" if "failed" in event else "done",
                    "error": None, "started_at": db.utcnow(),
                })
                del JOBS[20:]
        except Exception:
            pass  # no kaggle token / offline — poller stays quiet
        time.sleep(interval)


def serve_ui(port: int = 3456, open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    print(f"nanolab ui → {url}  (Ctrl+C to stop)")
    threading.Thread(target=_cloud_poller, daemon=True).start()
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=port, log_level="warning")
