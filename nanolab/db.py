"""SQLite schema + helpers. One file holds the whole lab.

Tables: environments, eval_runs, samples, train_runs, adapters, ledger.
Default path is results/nanolab.db; override with the NANOLAB_DB env var
(tests point it at a tmp file so smoke tests stay hermetic).
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("results") / "nanolab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS environments (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,   -- as installed: owner/name, owner/name@ver, or local name
    env_id       TEXT NOT NULL,          -- verifiers load_environment id, e.g. alphabet-sort
    version      TEXT,
    installed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id                   INTEGER PRIMARY KEY,
    env_id               INTEGER NOT NULL REFERENCES environments(id),
    model                TEXT NOT NULL,
    endpoint             TEXT,
    num_examples         INTEGER,
    rollouts_per_example INTEGER,
    seed                 INTEGER,
    params_json          TEXT,
    status               TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    mean_reward          REAL,
    metrics_json         TEXT,
    started_at           TEXT,
    finished_at          TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    id            INTEGER PRIMARY KEY,
    eval_run_id   INTEGER NOT NULL REFERENCES eval_runs(id),
    example_index INTEGER NOT NULL,
    rollout_index INTEGER NOT NULL,
    prompt_json   TEXT,
    completion_json TEXT,
    reward        REAL,
    metrics_json  TEXT,
    from_cache    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    UNIQUE (eval_run_id, example_index, rollout_index)  -- resume key: skip what exists
);

CREATE TABLE IF NOT EXISTS train_runs (
    id                INTEGER PRIMARY KEY,
    env_id            INTEGER REFERENCES environments(id),
    model             TEXT NOT NULL,
    config_toml       TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    steps_completed   INTEGER NOT NULL DEFAULT 0,
    reward_curve_json TEXT,
    started_at        TEXT,
    finished_at       TEXT
);

CREATE TABLE IF NOT EXISTS adapters (
    id           INTEGER PRIMARY KEY,
    train_run_id INTEGER REFERENCES train_runs(id),
    base_model   TEXT NOT NULL,
    step         INTEGER,
    path         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger (
    id                INTEGER PRIMARY KEY,
    run_kind          TEXT NOT NULL,  -- eval|train|serve
    run_id            INTEGER,
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deployments (
    id          INTEGER PRIMARY KEY,
    adapter_id  INTEGER REFERENCES adapters(id),
    base_model  TEXT NOT NULL,
    served_name TEXT NOT NULL,      -- model name requests should use
    endpoint    TEXT NOT NULL,      -- e.g. http://localhost:8000/v1
    pid         INTEGER,
    status      TEXT NOT NULL DEFAULT 'running',  -- running|stopped|dead
    created_at  TEXT NOT NULL
);
"""


def db_path() -> Path:
    return Path(os.environ.get("NANOLAB_DB", DEFAULT_DB))


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the lab database with the schema applied."""
    target = Path(path) if path is not None else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def register_environment(
    conn: sqlite3.Connection, slug: str, env_id: str, version: str | None
) -> int:
    """Insert or refresh an installed environment; returns its row id."""
    conn.execute(
        """
        INSERT INTO environments (slug, env_id, version, installed_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            env_id = excluded.env_id,
            version = excluded.version,
            installed_at = excluded.installed_at
        """,
        (slug, env_id, version, utcnow()),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM environments WHERE slug = ?", (slug,)).fetchone()
    return int(row["id"])


def list_environments(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute("SELECT * FROM environments ORDER BY installed_at DESC").fetchall()
    )


def get_environment(conn: sqlite3.Connection, ref: str) -> sqlite3.Row | None:
    """Look up an environment by slug or by bare env id."""
    return conn.execute(
        "SELECT * FROM environments WHERE slug = ? OR env_id = ?", (ref, ref)
    ).fetchone()
