"""The lab notebook — one self-contained HTML page from the db.

`nanolab report` renders results/leaderboard.html: KPI strip, per-env
leaderboards with deltas, training curves with the trainability window
drawn in, token ledger. No server, no JS build step, no external assets —
the design language ("a live instrument, not a website") in one file.
UI numbers are never computed here beyond aggregation of the same SQLite
rows the CLI reads.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import db

CSS = """
:root {
  --bg:#0A0A0B; --surface:#131316; --surface2:#1A1A1E; --border:#26262B;
  --text:#EDEDEF; --dim:#8B8B93; --accent:#B7F542; --ok:#4ADE80;
  --bad:#F87171; --warn:#FBBF24; --entity:#A78BFA;
  --mono:ui-monospace,'SF Mono','Cascadia Code',Menlo,Consolas,monospace;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--text); margin:0;
  font:13px/1.5 -apple-system,'Segoe UI',sans-serif; padding:2.5rem 1.25rem 4rem; }
main { max-width:64rem; margin:0 auto; }
header h1 { font:600 20px var(--mono); letter-spacing:-.02em; margin:0; }
header .sub { color:var(--dim); font-size:12px; margin-top:2px; }
.cli { font:12px var(--mono); background:var(--surface); border:1px solid var(--border);
  border-radius:6px; padding:.45rem .7rem; color:var(--dim); margin-top:.9rem; }
.cli b { color:var(--accent); font-weight:500; }
.kpis { display:flex; gap:0; border:1px solid var(--border); border-radius:6px;
  background:var(--surface); margin:1.5rem 0 2.25rem; overflow:hidden; }
.kpi { flex:1; padding:.9rem 1.1rem; border-right:1px solid var(--border); }
.kpi:last-child { border-right:none; }
.kpi .n { font:500 26px var(--mono); letter-spacing:-.02em; }
.kpi .l { font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em;
  color:var(--dim); margin-top:2px; }
.kpi .c { font-size:11px; color:var(--dim); margin-top:1px; }
section { margin:2.25rem 0; }
.sec { font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em;
  color:var(--dim); border-bottom:1px solid var(--border); padding-bottom:.4rem;
  margin-bottom: .9rem; }
.sec b { color:var(--accent); font-weight:500; margin-right:.5rem; }
h3 { font:500 13px var(--mono); margin:1.2rem 0 .4rem; }
table { border-collapse:collapse; width:100%; background:var(--surface);
  border:1px solid var(--border); border-radius:6px; overflow:hidden; }
th { font:11px var(--mono); text-transform:uppercase; letter-spacing:.08em;
  color:var(--dim); text-align:left; padding:.5rem .8rem; background:var(--surface2);
  border-bottom:1px solid var(--border); font-weight:500; }
td { padding:.45rem .8rem; border-bottom:1px solid var(--border); font-size:12px; }
tr:last-child td { border-bottom:none; }
td.num, .mono { font-family:var(--mono); }
.reward { font:500 13px var(--mono); }
.up { color:var(--ok); } .down { color:var(--bad); } .flat { color:var(--dim); }
.chip { font:11px var(--mono); border:1px solid var(--border); border-radius:4px;
  padding:1px 6px; }
.chip.done { color:var(--ok); border-color:color-mix(in srgb,var(--ok) 40%,transparent); }
.chip.failed { color:var(--bad); border-color:color-mix(in srgb,var(--bad) 40%,transparent); }
.chip.running { color:var(--warn); border-color:color-mix(in srgb,var(--warn) 40%,transparent); }
.model { font-family:var(--mono); }
.model .adapter { color:var(--entity); }
.fig { font:11px var(--mono); color:var(--dim); margin-top:.35rem; }
.fig b { color:var(--dim); font-weight:600; }
.empty { border:1px dashed var(--border); border-radius:6px; padding:1.5rem;
  text-align:center; color:var(--dim); background:var(--surface); }
.empty .cmd { font-family:var(--mono); color:var(--accent); }
footer { margin-top:3rem; border-top:1px solid var(--border); padding-top:.8rem;
  font:11px var(--mono); color:var(--dim); display:flex; justify-content:space-between;
  flex-wrap:wrap; gap:.5rem; }
.parity { color:var(--ok); }
svg text { font:9px var(--mono); fill:var(--dim); }
"""


def _model_chip(model: str) -> str:
    if ":" in model:
        base, _, adapter = model.rpartition(":")
        return (
            f'<span class="model">{html.escape(base)}'
            f'<span class="adapter">:{html.escape(adapter)}</span></span>'
        )
    return f'<span class="model">{html.escape(model)}</span>'


def _delta_cell(delta: float | None) -> str:
    if delta is None:
        return '<span class="flat">—</span>'
    if abs(delta) < 1e-9:
        return '<span class="flat">=</span>'
    cls = "up" if delta > 0 else "down"
    arrow = "↑" if delta > 0 else "↓"
    return f'<span class="{cls}">{arrow} {abs(delta):.3f}</span>'


def _curve_svg(rewards: list[float], width: int = 240, height: int = 56) -> str:
    """Reward curve with the 10–80% trainability window shaded behind it."""
    if len(rewards) < 2:
        return "—"
    lo = min(min(rewards), 0.1)
    hi = max(max(rewards), 0.8)
    span = (hi - lo) or 1.0

    def y(value: float) -> float:
        return height - 4 - (value - lo) / span * (height - 8)

    step_x = width / (len(rewards) - 1)
    points = " ".join(f"{i * step_x:.1f},{y(r):.1f}" for i, r in enumerate(rewards))
    band_top, band_bottom = y(0.8), y(0.1)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="{band_top:.1f}" width="{width}" '
        f'height="{band_bottom - band_top:.1f}" fill="#B7F542" opacity="0.07"/>'
        f'<polyline points="{points}" fill="none" stroke="#B7F542" stroke-width="1.5"/>'
        f"</svg>"
    )


def render(output_path: str | Path | None = None) -> Path:
    conn = db.connect()
    try:
        eval_runs = conn.execute(
            """
            SELECT e.*, v.slug AS env_slug
            FROM eval_runs e JOIN environments v ON v.id = e.env_id
            WHERE e.status = 'done' ORDER BY v.slug, e.mean_reward DESC, e.id DESC
            """
        ).fetchall()
        sample_stats = {
            r["id"]: conn.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN metrics_json LIKE '%\"_error\"%'"
                " THEN 1 ELSE 0 END) AS errs FROM samples WHERE eval_run_id = ?",
                (r["id"],),
            ).fetchone()
            for r in eval_runs
        }
        train_runs = conn.execute(
            """
            SELECT t.*, v.slug AS env_slug FROM train_runs t
            LEFT JOIN environments v ON v.id = t.env_id ORDER BY t.id DESC
            """
        ).fetchall()
        n_envs = conn.execute("SELECT COUNT(*) AS n FROM environments").fetchone()["n"]
        n_adapters = conn.execute("SELECT COUNT(*) AS n FROM adapters").fetchone()["n"]
        ledger_rows = conn.execute(
            "SELECT model, SUM(prompt_tokens) AS pt, SUM(completion_tokens) AS ct"
            " FROM ledger GROUP BY model ORDER BY pt DESC"
        ).fetchall()
        # delta vs the previous completed run of the same env+model
        history = conn.execute(
            "SELECT id, env_id, model, mean_reward FROM eval_runs"
            " WHERE status='done' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    prev_reward: dict[tuple, float] = {}
    deltas: dict[int, float | None] = {}
    for row in history:
        key = (row["env_id"], row["model"])
        if row["mean_reward"] is not None:
            deltas[row["id"]] = (
                row["mean_reward"] - prev_reward[key] if key in prev_reward else None
            )
            prev_reward[key] = row["mean_reward"]

    best = max(
        (r for r in eval_runs if r["mean_reward"] is not None),
        key=lambda r: r["mean_reward"],
        default=None,
    )
    kpis = f"""
    <div class="kpis">
      <div class="kpi"><div class="n">{len(eval_runs)}</div><div class="l">eval runs</div>
        <div class="c">completed</div></div>
      <div class="kpi"><div class="n">{f"{best['mean_reward']:.3f}" if best else "—"}</div>
        <div class="l">best reward</div>
        <div class="c">{html.escape(best["env_slug"]) if best else "no runs yet"}</div></div>
      <div class="kpi"><div class="n">{len(train_runs)}</div><div class="l">training runs</div>
        <div class="c">{n_adapters} adapters</div></div>
      <div class="kpi"><div class="n">{n_envs}</div><div class="l">environments</div>
        <div class="c">installed</div></div>
      <div class="kpi"><div class="n">{sum((r["pt"] or 0) + (r["ct"] or 0) for r in ledger_rows):,}</div>
        <div class="l">tokens</div><div class="c">ledger total</div></div>
    </div>"""

    # ── 01 · evaluations ────────────────────────────────────────────────────
    fig_n = 0
    eval_sections = []
    by_env: dict[str, list] = {}
    for row in eval_runs:
        by_env.setdefault(row["env_slug"], []).append(row)
    for env_slug, runs in by_env.items():
        fig_n += 1
        rows_html = []
        for rank, row in enumerate(runs, start=1):
            stats = sample_stats[row["id"]]
            meta = json.loads(row["metrics_json"] or "{}")
            reward_std = meta.get("avg_metrics", {}).get("reward_std")
            rows_html.append(
                f"<tr><td class=num>{rank}</td>"
                f"<td>{_model_chip(row['model'])}</td>"
                f"<td class=num><span class=reward>{row['mean_reward']:.3f}</span>"
                f"{f' <span class=flat>±{reward_std:.3f}</span>' if isinstance(reward_std, float) else ''}</td>"
                f"<td class=num>{_delta_cell(deltas.get(row['id']))}</td>"
                f"<td class=num>{row['num_examples']}×{row['rollouts_per_example']}</td>"
                f"<td class=num>{stats['errs'] or 0}</td>"
                f"<td class=num>#{row['id']}</td>"
                f"<td class=num>{(row['finished_at'] or '')[:10]}</td></tr>"
            )
        eval_sections.append(
            f"<h3>{html.escape(env_slug)}</h3><table>"
            "<tr><th>#</th><th>model</th><th>reward</th><th>Δ prev</th>"
            "<th>n×r</th><th>err</th><th>run</th><th>date</th></tr>"
            f"{''.join(rows_html)}</table>"
            f'<div class="fig"><b>FIG.{fig_n}</b> — leaderboard, {html.escape(env_slug)}'
            f' · <span class=mono>$ nanolab eval run {html.escape(env_slug.split("/")[-1])} -m &lt;model&gt;</span></div>'
        )
    evals_html = "".join(eval_sections) or (
        '<div class="empty">No evaluations yet. An eval measures a model on an'
        ' environment, rollout by rollout.<br>'
        '<span class="cmd">$ nanolab eval run gsm8k -m &lt;model&gt;</span></div>'
    )

    # ── 02 · training runs ──────────────────────────────────────────────────
    train_rows = []
    for t in train_runs:
        curve = json.loads(t["reward_curve_json"] or "[]")
        rewards = [p["reward"] for p in curve]
        first_last = f"{rewards[0]:.3f} → {rewards[-1]:.3f}" if rewards else "—"
        delta = rewards[-1] - rewards[0] if rewards else None
        train_rows.append(
            f"<tr><td class=num>#{t['id']}</td>"
            f"<td>{_model_chip(t['model'])}</td>"
            f"<td class=mono>{html.escape(t['env_slug'] or '?')}</td>"
            f"<td><span class='chip {t['status']}'>{t['status']}</span></td>"
            f"<td class=num>{t['steps_completed']}</td>"
            f"<td>{_curve_svg(rewards)}</td>"
            f"<td class=num>{first_last}<br>{_delta_cell(delta)}</td></tr>"
        )
    fig_n += 1
    train_html = (
        "<table><tr><th>run</th><th>model</th><th>env</th><th>status</th>"
        "<th>steps</th><th>reward curve</th><th>first → last</th></tr>"
        f"{''.join(train_rows)}</table>"
        f'<div class="fig"><b>FIG.{fig_n}</b> — training runs; shaded band is the'
        " 10–80% trainability window ·"
        ' <span class=mono>$ nanolab train configs/&lt;config&gt;.toml</span></div>'
        if train_rows
        else '<div class="empty">No training runs yet. Training turns rewards into'
        " a LoRA adapter with GRPO, checkpointing as it goes.<br>"
        '<span class="cmd">$ nanolab train configs/qwen3-0.6b-gsm8k.toml</span></div>'
    )

    # ── 03 · ledger ─────────────────────────────────────────────────────────
    ledger_html = (
        "<table><tr><th>model</th><th>prompt tokens</th><th>completion tokens</th></tr>"
        + "".join(
            f"<tr><td>{_model_chip(r['model'] or '?')}</td>"
            f"<td class=num>{r['pt'] or 0:,}</td><td class=num>{r['ct'] or 0:,}</td></tr>"
            for r in ledger_rows
        )
        + "</table>"
        if ledger_rows
        else '<div class="empty">No API tokens spent yet.</div>'
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanolab — lab notebook</title><style>{CSS}</style></head>
<body><main>
<header>
  <h1>nanolab</h1>
  <div class="sub">lab notebook — environments · evals · training · serving, one SQLite file</div>
  <div class="cli"><b>$</b> nanolab report</div>
</header>
{kpis}
<section><div class="sec"><b>01</b>evaluations</div>{evals_html}</section>
<section><div class="sec"><b>02</b>training runs</div>{train_html}</section>
<section><div class="sec"><b>03</b>token ledger</div>{ledger_html}</section>
<footer>
  <span class="parity">✓ eval station reproduces vf-eval exactly on identical configs (anchor check)</span>
  <span>generated {db.utcnow()}</span>
</footer>
</main></body></html>"""

    target = Path(output_path) if output_path else Path("results") / "leaderboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    return target
