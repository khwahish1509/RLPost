"""The four-column instrument: where does improvement actually live?

For the same stream tasks, four numbers:

  1. base      — the frozen Player alone (empty notebook)
  2. +context  — the Player reading a Scribe-maintained notebook
  3. +weights  — a trained (LoRA) Player alone
  4. +both     — the trained Player reading the notebook

The gaps carry the diagnosis:
  - +context ≈ +weights → the failure was MISSING KNOWLEDGE: text closes it,
    on any model, including ones whose weights you can't touch.
  - +weights ≫ +context → the failure was MISSING SKILL: only weight
    training closes it.

A stream-environment eval run measures columns 1 and 2 in one shot
(baseline_score / player_score in its metrics). Columns 3 and 4 come from
the same eval with the Player pointed at a served adapter. This module just
reads those stored runs and renders the comparison — the instrument never
computes a number the eval station didn't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import db

# below this gap size two columns are considered equivalent
EQUIVALENCE = 0.05


class InstrumentError(RuntimeError):
    pass


@dataclass
class FourColumns:
    env: str
    player: str
    adapter_player: str | None
    base: float
    context: float
    weights: float | None
    both: float | None
    verdict: str


def _stream_scores(conn, run_id: int) -> tuple[str, str, float, float]:
    row = conn.execute(
        """
        SELECT e.*, v.slug AS env_slug FROM eval_runs e
        JOIN environments v ON v.id = e.env_id WHERE e.id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise InstrumentError(f"No eval run with id {run_id}")
    meta = json.loads(row["metrics_json"] or "{}").get("avg_metrics", {})
    if "baseline_score" not in meta or "player_score" not in meta:
        raise InstrumentError(
            f"Eval run #{run_id} has no baseline_score/player_score metrics — "
            "the instrument needs stream-environment runs (e.g. scribe-stream), "
            "whose rubric measures the Player with and without the notebook."
        )
    return (
        row["env_slug"],
        row["model"],
        float(meta["baseline_score"]),
        float(meta["player_score"]),
    )


def _verdict(base, context, weights, both) -> str:
    knowledge = context - base
    if weights is None:
        return (
            f"+context lifts the Player by {knowledge:+.3f}. Columns 3–4 pending: "
            "run the same stream eval with the Player served as base:adapter "
            "to separate missing-knowledge from missing-skill."
        )
    skill = weights - base
    if abs(knowledge - skill) <= EQUIVALENCE:
        head = (
            "MISSING KNOWLEDGE: notes match weight-training "
            f"({knowledge:+.3f} vs {skill:+.3f}) — text closes this gap on any "
            "model, no training required."
        )
    elif skill > knowledge:
        head = (
            "MISSING SKILL: weight-training beats notes "
            f"({skill:+.3f} vs {knowledge:+.3f}) — this gap only closes by "
            "changing the model."
        )
    else:
        head = (
            "KNOWLEDGE-DOMINANT: notes beat weight-training "
            f"({knowledge:+.3f} vs {skill:+.3f}) — the cheap, portable path wins "
            "here."
        )
    if both is not None:
        best_single = max(context, weights)
        if both > best_single + EQUIVALENCE:
            head += f" +both adds synergy ({both - best_single:+.3f} over the best single)."
        elif both < best_single - EQUIVALENCE:
            head += f" +both underperforms the best single column ({both - best_single:+.3f}) — the two paths interfere."
    return head


def four_columns(base_run_id: int, adapter_run_id: int | None = None) -> FourColumns:
    conn = db.connect()
    try:
        env, player, base, context = _stream_scores(conn, base_run_id)
        weights = both = None
        adapter_player = None
        if adapter_run_id is not None:
            env2, adapter_player, weights, both = _stream_scores(conn, adapter_run_id)
            if env2 != env:
                raise InstrumentError(
                    f"Runs are from different environments ({env} vs {env2}) — "
                    "the four columns must share the same tasks."
                )
    finally:
        conn.close()
    return FourColumns(
        env=env,
        player=player,
        adapter_player=adapter_player,
        base=base,
        context=context,
        weights=weights,
        both=both,
        verdict=_verdict(base, context, weights, both),
    )


def render_text(cols: FourColumns) -> str:
    def cell(v):
        return f"{v:.3f}" if v is not None else "  —  "

    lines = [
        f"env: {cols.env}",
        f"player: {cols.player}"
        + (f"  ·  adapter player: {cols.adapter_player}" if cols.adapter_player else ""),
        "",
        "  column        score   Δ vs base",
        f"  1  base       {cell(cols.base)}     —",
        f"  2  +context   {cell(cols.context)}   {cols.context - cols.base:+.3f}",
        f"  3  +weights   {cell(cols.weights)}"
        + (f"   {cols.weights - cols.base:+.3f}" if cols.weights is not None else "   pending"),
        f"  4  +both      {cell(cols.both)}"
        + (f"   {cols.both - cols.base:+.3f}" if cols.both is not None else "   pending"),
        "",
        f"verdict: {cols.verdict}",
    ]
    return "\n".join(lines)
