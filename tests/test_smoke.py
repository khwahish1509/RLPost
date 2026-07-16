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
    # isolate from the repo's .env and results/ so tests stay hermetic
    monkeypatch.chdir(tmp_path)
    for var in ("NANOLAB_DEFAULT_MODEL", "NANOLAB_API_BASE_URL", "NANOLAB_API_KEY_VAR"):
        monkeypatch.delenv(var, raising=False)
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
        ["train", "configs/example.toml"],
        ["deployments", "create", "1"],
        ["deployments", "list"],
    ],
)
def test_stubs_say_not_implemented(tmp_db, argv):
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "not implemented" in result.output


# ── evaluation station (network-free) ────────────────────────────────────────


def _fake_outputs(model="fake-model"):
    """A synthetic GenerateOutputs: 2 examples × 2 rollouts, one errored."""
    def rollout(example_id, reward, error=None):
        return {
            "example_id": example_id,
            "prompt": [{"role": "user", "content": f"q{example_id}"}],
            "completion": [{"role": "assistant", "content": "a"}],
            "reward": reward,
            "metrics": {"accuracy": reward},
            "stop_condition": "max_turns_for_example" if error is None else "has_error",
            "error": error,
            "token_usage": {"input_tokens": 10, "output_tokens": 5},
        }

    return {
        "outputs": [
            rollout(0, 1.0),
            rollout(0, 0.5),
            rollout(1, 0.0, error={"error": "RateLimitError", "message": "429"}),
            rollout(1, 1.0),
        ],
        "metadata": {"model": model, "avg_reward": 0.625, "avg_metrics": {"accuracy": 0.625}},
    }


def _seed_run(params="{}", status="done", mean_reward=0.625):
    conn = db.connect()
    env_row_id = db.register_environment(conn, "test/fake-env", "fake-env", "0.0.1")
    cur = conn.execute(
        "INSERT INTO eval_runs (env_id, model, num_examples, rollouts_per_example,"
        " params_json, status, mean_reward, metrics_json, started_at) "
        "VALUES (?, 'fake-model', 2, 2, ?, ?, ?, '{}', ?)",
        (env_row_id, params, status, mean_reward, db.utcnow()),
    )
    conn.commit()
    run_id = cur.lastrowid
    return conn, run_id


def test_persist_outputs_and_show(tmp_db):
    from nanolab import evaluate, ledger

    conn, run_id = _seed_run()
    n_samples, n_errors = evaluate.persist_outputs(conn, run_id, _fake_outputs())
    assert (n_samples, n_errors) == (4, 1)
    totals = ledger.totals_by_model(conn)
    assert totals[0]["prompt_tokens"] == 40
    conn.close()

    detail = evaluate.show(run_id)
    assert detail["reward"]["n"] == 4
    assert detail["reward"]["avg"] == pytest.approx(0.625)
    assert detail["metrics"]["accuracy"]["avg"] == pytest.approx(0.625)
    assert detail["errors"] == {"RateLimitError": 1}
    assert detail["stop_conditions"]["has_error"] == 1


def test_persist_outputs_is_idempotent(tmp_db):
    from nanolab import evaluate

    conn, run_id = _seed_run()
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    evaluate.persist_outputs(conn, run_id, _fake_outputs())  # resume re-write
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM samples WHERE eval_run_id = ?", (run_id,)
    ).fetchone()["n"]
    conn.close()
    assert n == 4  # INSERT OR REPLACE on the resume key, no duplicates


def test_run_level_cache_hit_needs_no_network(tmp_db, monkeypatch):
    from nanolab import evaluate

    params = evaluate._params_json(
        base_url="http://x", num_examples=2, rollouts_per_example=2,
        shuffle_seed=None, temperature=None, max_tokens=None,
    )
    conn, run_id = _seed_run(params=params)
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    conn.close()

    monkeypatch.setenv("FAKE_KEY", "k")
    summary = evaluate.run(
        "fake-env", "fake-model", api_base_url="http://x", api_key_var="FAKE_KEY",
        num_examples=2, rollouts_per_example=2,
    )
    assert summary.cached is True
    assert summary.run_id == run_id
    assert summary.mean_reward == pytest.approx(0.625)


def test_eval_run_cli_requires_model_and_endpoint(tmp_db):
    result = runner.invoke(app, ["eval", "run", "fake-env"])
    assert result.exit_code == 1
    assert "No model" in result.output


def test_report_renders_leaderboard(tmp_db, tmp_path):
    from nanolab import evaluate, report

    conn, run_id = _seed_run()
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    conn.close()
    out = report.render(tmp_path / "board.html")
    text = out.read_text()
    assert "fake-model" in text and "fake-env" in text and "0.625" in text


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
