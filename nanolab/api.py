"""The nanolab web app: a read-only JSON API over the lab db + the UI shell.

`nanolab ui` serves this with uvicorn. The API computes nothing the CLI
doesn't — every number comes from the same SQLite rows, so UI-vs-CLI
equality is trivially true. The UI lives in nanolab/ui/ (no build step).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import db

UI_DIR = Path(__file__).parent / "ui"


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

    return Starlette(
        routes=[
            Route("/api/overview", endpoint(_overview)),
            Route("/api/environments", endpoint(_environments)),
            Route("/api/evals", endpoint(_evals)),
            Route("/api/evals/{run_id:int}", endpoint(_eval_detail)),
            Route("/api/training", endpoint(_training)),
            Route("/api/training/{run_id:int}", endpoint(_training_detail)),
            Route("/api/deployments", endpoint(_deployments)),
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
