"""Smoke tests — hermetic, network-free. CI runs exactly this."""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from nanolab import db, envs
from nanolab.cli import app

runner = CliRunner()

TABLES = {"environments", "eval_runs", "samples", "train_runs", "adapters", "ledger"}


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "lab.db"
    monkeypatch.setenv("NANOLAB_DB", str(path))
    return path


def test_schema_creates_all_tables(tmp_db):
    conn = db.connect()
    try:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert TABLES <= names


def test_register_and_list_environment(tmp_db):
    conn = db.connect()
    try:
        env_id = db.register_environment(
            conn, "primeintellect/alphabet-sort", "alphabet-sort", "0.1.0"
        )
        assert env_id == 1
        # re-registering the same slug updates, not duplicates
        again = db.register_environment(
            conn, "primeintellect/alphabet-sort", "alphabet-sort", "0.2.0"
        )
        assert again == env_id
        rows = db.list_environments(conn)
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["version"] == "0.2.0"


def test_samples_resume_key_is_unique(tmp_db):
    conn = db.connect()
    try:
        env_row = db.register_environment(conn, "x/y", "y", None)
        conn.execute(
            "INSERT INTO eval_runs (env_id, model) VALUES (?, ?)", (env_row, "m")
        )
        conn.execute(
            "INSERT INTO samples (eval_run_id, example_index, rollout_index, created_at)"
            " VALUES (1, 0, 0, ?)",
            (db.utcnow(),),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO samples (eval_run_id, example_index, rollout_index, created_at)"
                " VALUES (1, 0, 0, ?)",
                (db.utcnow(),),
            )
    finally:
        conn.close()


def test_parse_slug():
    assert envs.parse_slug("primeintellect/alphabet-sort") == "alphabet-sort"
    assert envs.parse_slug("owner/env@0.2.3") == "env"
    assert envs.parse_slug("gsm8k") == "gsm8k"
    with pytest.raises(envs.EnvInstallError):
        envs.parse_slug("owner/bad env name")


def test_cli_help_and_version():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_env_list_empty(tmp_db):
    result = runner.invoke(app, ["env", "list"])
    assert result.exit_code == 0
    assert "no environments installed" in result.output


def test_eval_list_empty(tmp_db):
    result = runner.invoke(app, ["eval", "list"])
    assert result.exit_code == 0
    assert "no eval runs" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "run", "alphabet-sort", "-m", "some-model"],
        ["eval", "show", "1"],
        ["train", "configs/example.toml"],
        ["deployments", "create", "1"],
        ["deployments", "list"],
        ["report"],
    ],
)
def test_stubs_say_not_implemented(tmp_db, argv):
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "not implemented" in result.output


def test_ledger_roundtrip(tmp_db):
    from nanolab import ledger

    conn = db.connect()
    try:
        ledger.record(conn, "eval", None, "test-model", 100, 20)
        totals = ledger.totals_by_model(conn)
    finally:
        conn.close()
    assert totals[0]["model"] == "test-model"
    assert totals[0]["prompt_tokens"] == 100
