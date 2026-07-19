"""Merge training artifacts (adapters + db records) into the local lab.

Used when a training run happens elsewhere — a Kaggle kernel, a Colab
session — and its output comes home as a directory or zip containing
`adapters/…` and `results/nanolab.db`. Train runs and their adapter rows
are inserted as NEW local rows (ids remapped); adapter files are copied
into the local `adapters/` tree under fresh run directories.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

from . import db


class MergeError(RuntimeError):
    pass


def _extract(source: Path, workdir: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix == ".zip":
        target = workdir / "artifacts"
        with zipfile.ZipFile(source) as zf:
            zf.extractall(target)
        return target
    raise MergeError(f"Not a directory or zip: {source}")


def merge_artifacts(source: str | Path, workdir: str | Path = ".cache/merge") -> list[int]:
    """Merge remote artifacts into the local lab; returns new train_run ids."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    root = _extract(Path(source), workdir)

    remote_db = root / "results" / "nanolab.db"
    if not remote_db.exists():
        found = list(root.rglob("nanolab.db"))
        if not found:
            raise MergeError(f"No nanolab.db found under {root}")
        remote_db = found[0]
        root = remote_db.parent.parent

    remote = sqlite3.connect(remote_db)
    remote.row_factory = sqlite3.Row
    local = db.connect()
    new_ids: list[int] = []
    try:
        for run in remote.execute("SELECT * FROM train_runs ORDER BY id").fetchall():
            env_row = None
            env = remote.execute(
                "SELECT slug, env_id, version FROM environments WHERE id = ?",
                (run["env_id"],),
            ).fetchone()
            if env is not None:
                env_row = db.register_environment(
                    local, env["slug"], env["env_id"], env["version"]
                )
            cur = local.execute(
                "INSERT INTO train_runs (env_id, model, config_toml, status,"
                " steps_completed, reward_curve_json, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    env_row, run["model"], run["config_toml"], run["status"],
                    run["steps_completed"], run["reward_curve_json"],
                    run["started_at"], run["finished_at"],
                ),
            )
            new_run_id = int(cur.lastrowid)
            new_ids.append(new_run_id)

            run_dir = Path("adapters") / f"run{new_run_id}"
            for adapter in remote.execute(
                "SELECT * FROM adapters WHERE train_run_id = ? ORDER BY step",
                (run["id"],),
            ).fetchall():
                old_path = Path(adapter["path"])
                src = root / old_path
                new_path = run_dir / old_path.name
                if src.exists():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    if new_path.exists():
                        shutil.rmtree(new_path)
                    shutil.copytree(src, new_path)
                local.execute(
                    "INSERT INTO adapters (train_run_id, base_model, step, path,"
                    " created_at) VALUES (?,?,?,?,?)",
                    (
                        new_run_id, adapter["base_model"], adapter["step"],
                        str(new_path), adapter["created_at"],
                    ),
                )
        local.commit()
    finally:
        remote.close()
        local.close()
    return new_ids
