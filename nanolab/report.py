"""Static leaderboard — one self-contained HTML file from the lab db.

No server, no JS build step. `nanolab report` writes results/leaderboard.html.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import db

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nanolab leaderboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
         max-width: 60rem; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .4rem .8rem; border-bottom: 1px solid #8884; }}
  th {{ font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  .best {{ font-weight: 700; }}
  footer {{ margin-top: 2rem; font-size: .85rem; opacity: .7; }}
</style>
</head>
<body>
<h1>nanolab leaderboard</h1>
{sections}
<footer>generated {generated} · {n_runs} completed eval runs · one CLI, one SQLite file</footer>
</body>
</html>
"""

_SECTION = """<h2>{env}</h2>
<table>
<tr><th>model</th><th>reward</th><th>±std</th><th>n×r</th><th>errors</th><th>run</th><th>date</th></tr>
{rows}
</table>
"""

_ROW = (
    "<tr><td{best}>{model}</td><td class=num{best}>{reward}</td>"
    "<td class=num>{std}</td><td class=num>{nr}</td><td class=num>{errors}</td>"
    "<td class=num>#{run_id}</td><td>{date}</td></tr>"
)

_TRAIN_SECTION = """<h2>training runs</h2>
<table>
<tr><th>run</th><th>model</th><th>env</th><th>status</th><th>steps</th><th>reward curve</th><th>first → last</th></tr>
{rows}
</table>
"""

_TRAIN_ROW = (
    "<tr><td class=num>#{run_id}</td><td>{model}</td><td>{env}</td>"
    "<td>{status}</td><td class=num>{steps}</td><td>{svg}</td>"
    "<td class=num>{first_last}</td></tr>"
)


def _curve_svg(rewards: list[float], width: int = 220, height: int = 40) -> str:
    """Inline SVG polyline of a reward curve; self-contained, no JS."""
    if len(rewards) < 2:
        return "—"
    lo, hi = min(rewards), max(rewards)
    span = (hi - lo) or 1.0
    step_x = width / (len(rewards) - 1)
    points = " ".join(
        f"{i * step_x:.1f},{height - 3 - (r - lo) / span * (height - 6):.1f}"
        for i, r in enumerate(rewards)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg"><polyline points="{points}" '
        f'fill="none" stroke="#4a9eff" stroke-width="2"/></svg>'
    )


def render(output_path: str | Path | None = None) -> Path:
    conn = db.connect()
    try:
        runs = conn.execute(
            """
            SELECT e.*, v.slug AS env_slug
            FROM eval_runs e JOIN environments v ON v.id = e.env_id
            WHERE e.status = 'done'
            ORDER BY v.slug, e.mean_reward DESC, e.id DESC
            """
        ).fetchall()
        stats = {
            row["id"]: conn.execute(
                """
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN metrics_json LIKE '%"_error"%' THEN 1 ELSE 0 END) AS errs
                FROM samples WHERE eval_run_id = ?
                """,
                (row["id"],),
            ).fetchone()
            for row in runs
        }
        train_runs = conn.execute(
            """
            SELECT t.*, v.slug AS env_slug FROM train_runs t
            LEFT JOIN environments v ON v.id = t.env_id
            ORDER BY t.id DESC
            """
        ).fetchall()
    finally:
        conn.close()

    sections = []
    by_env: dict[str, list] = {}
    for row in runs:
        by_env.setdefault(row["env_slug"], []).append(row)

    for env_slug, env_runs in by_env.items():
        rows = []
        for i, row in enumerate(env_runs):
            meta = json.loads(row["metrics_json"] or "{}")
            reward_std = meta.get("avg_metrics", {}).get("reward_std")
            counts = stats[row["id"]]
            rows.append(
                _ROW.format(
                    best=' class="best"' if i == 0 else "",
                    model=html.escape(row["model"]),
                    reward=f"{row['mean_reward']:.3f}" if row["mean_reward"] is not None else "—",
                    std=f"{reward_std:.3f}" if isinstance(reward_std, float) else "—",
                    nr=f"{row['num_examples']}×{row['rollouts_per_example']}",
                    errors=counts["errs"] or 0,
                    run_id=row["id"],
                    date=(row["finished_at"] or "")[:10],
                )
            )
        sections.append(_SECTION.format(env=html.escape(env_slug), rows="\n".join(rows)))

    if train_runs:
        trows = []
        for t in train_runs:
            curve = json.loads(t["reward_curve_json"] or "[]")
            rewards = [p["reward"] for p in curve]
            first_last = (
                f"{rewards[0]:.3f} → {rewards[-1]:.3f}" if rewards else "—"
            )
            trows.append(
                _TRAIN_ROW.format(
                    run_id=t["id"],
                    model=html.escape(t["model"]),
                    env=html.escape(t["env_slug"] or "?"),
                    status=t["status"],
                    steps=t["steps_completed"],
                    svg=_curve_svg(rewards),
                    first_last=first_last,
                )
            )
        sections.append(_TRAIN_SECTION.format(rows="\n".join(trows)))

    page = _PAGE.format(
        sections="\n".join(sections) if sections else "<p>No completed eval runs yet.</p>",
        generated=db.utcnow(),
        n_runs=len(runs),
    )
    target = Path(output_path) if output_path else Path("results") / "leaderboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    return target
