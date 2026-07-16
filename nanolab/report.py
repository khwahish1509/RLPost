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

    page = _PAGE.format(
        sections="\n".join(sections) if sections else "<p>No completed eval runs yet.</p>",
        generated=db.utcnow(),
        n_runs=len(runs),
    )
    target = Path(output_path) if output_path else Path("results") / "leaderboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    return target
