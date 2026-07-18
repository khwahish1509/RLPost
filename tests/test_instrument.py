"""Tests for the four-column instrument (base · +context · +weights · +both)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from nanolab import db, instrument
from nanolab.cli import app

runner = CliRunner()


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOLAB_DB", str(tmp_path / "lab.db"))
    monkeypatch.chdir(tmp_path)


def _seed_stream_run(conn, env_id, model, baseline, player) -> int:
    cur = conn.execute(
        "INSERT INTO eval_runs (env_id, model, params_json, status, mean_reward,"
        " metrics_json, started_at) VALUES (?, ?, '{}', 'done', ?, ?, ?)",
        (
            env_id,
            model,
            player - baseline,
            json.dumps(
                {"avg_metrics": {"baseline_score": baseline, "player_score": player}}
            ),
            db.utcnow(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def test_two_columns_pending(tmp_db):
    conn = db.connect()
    env_id = db.register_environment(conn, "scribe-stream", "scribe-stream", "0.1.0")
    run = _seed_stream_run(conn, env_id, "big-model", 0.0, 0.857)
    conn.close()

    cols = instrument.four_columns(run)
    assert cols.base == 0.0 and cols.context == pytest.approx(0.857)
    assert cols.weights is None and cols.both is None
    assert "pending" in cols.verdict.lower() or "Columns 3–4 pending" in cols.verdict

    result = runner.invoke(app, ["instrument", str(run)])
    assert result.exit_code == 0
    assert "+context" in result.output and "+0.857" in result.output


def test_four_columns_missing_knowledge_verdict(tmp_db):
    conn = db.connect()
    env_id = db.register_environment(conn, "scribe-stream", "scribe-stream", "0.1.0")
    base_run = _seed_stream_run(conn, env_id, "player", 0.10, 0.80)      # knowledge +0.70
    adapter_run = _seed_stream_run(conn, env_id, "player:3", 0.78, 0.85)  # skill +0.68
    conn.close()

    cols = instrument.four_columns(base_run, adapter_run)
    assert cols.weights == pytest.approx(0.78)
    assert "MISSING KNOWLEDGE" in cols.verdict


def test_four_columns_missing_skill_verdict(tmp_db):
    conn = db.connect()
    env_id = db.register_environment(conn, "scribe-stream", "scribe-stream", "0.1.0")
    base_run = _seed_stream_run(conn, env_id, "player", 0.10, 0.20)      # knowledge +0.10
    adapter_run = _seed_stream_run(conn, env_id, "player:3", 0.60, 0.65)  # skill +0.50
    conn.close()

    assert "MISSING SKILL" in instrument.four_columns(base_run, adapter_run).verdict


def test_instrument_rejects_non_stream_runs(tmp_db):
    conn = db.connect()
    env_id = db.register_environment(conn, "primeintellect/gsm8k", "gsm8k", "0.1.3")
    cur = conn.execute(
        "INSERT INTO eval_runs (env_id, model, params_json, status, metrics_json,"
        " started_at) VALUES (?, 'm', '{}', 'done', '{}', ?)",
        (env_id, db.utcnow()),
    )
    conn.commit()
    run = cur.lastrowid
    conn.close()
    with pytest.raises(instrument.InstrumentError, match="stream-environment"):
        instrument.four_columns(run)


def test_instrument_rejects_mixed_environments(tmp_db):
    conn = db.connect()
    a = db.register_environment(conn, "scribe-stream", "scribe-stream", "0.1.0")
    b = db.register_environment(conn, "other-stream", "other-stream", "0.1.0")
    run_a = _seed_stream_run(conn, a, "player", 0.0, 0.5)
    run_b = _seed_stream_run(conn, b, "player:3", 0.1, 0.6)
    conn.close()
    with pytest.raises(instrument.InstrumentError, match="different environments"):
        instrument.four_columns(run_a, run_b)
