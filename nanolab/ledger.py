"""Token accounting. Every API call lands one row in the ledger table."""

from __future__ import annotations

import sqlite3

from . import db


def record(
    conn: sqlite3.Connection,
    run_kind: str,
    run_id: int | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    conn.execute(
        """
        INSERT INTO ledger (run_kind, run_id, model, prompt_tokens, completion_tokens, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_kind, run_id, model, prompt_tokens, completion_tokens, db.utcnow()),
    )
    conn.commit()


def totals_by_model(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT model,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens
            FROM ledger GROUP BY model ORDER BY 2 DESC
            """
        ).fetchall()
    )
