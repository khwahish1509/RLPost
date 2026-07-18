"""API tests for the web app — served from the same db the CLI reads."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from nanolab import db
from nanolab.api import build_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOLAB_DB", str(tmp_path / "lab.db"))
    monkeypatch.chdir(tmp_path)
    return TestClient(build_app())


def _seed(conn):
    env_id = db.register_environment(conn, "primeintellect/gsm8k", "gsm8k", "0.1.3")
    conn.execute(
        "INSERT INTO eval_runs (env_id, model, num_examples, rollouts_per_example,"
        " params_json, status, mean_reward, metrics_json, started_at, finished_at)"
        " VALUES (?, 'test-model', 2, 1, '{}', 'done', 0.75,"
        " '{\"avg_metrics\": {\"accuracy\": 0.75}}', ?, ?)",
        (env_id, db.utcnow(), db.utcnow()),
    )
    conn.execute(
        "INSERT INTO samples (eval_run_id, example_index, rollout_index,"
        " prompt_json, completion_json, reward, metrics_json, created_at)"
        " VALUES (1, 0, 0, '[{\"role\":\"user\",\"content\":\"q\"}]',"
        " '[{\"role\":\"assistant\",\"content\":\"a\"}]', 1.0, '{}', ?)",
        (db.utcnow(),),
    )
    conn.execute(
        "INSERT INTO train_runs (env_id, model, config_toml, status,"
        " steps_completed, reward_curve_json, started_at) VALUES"
        " (?, 'Qwen/Qwen3-0.6B', 'max_steps = 2', 'done', 2,"
        " '[{\"step\":0,\"reward\":0.3,\"loss\":1.0},{\"step\":1,\"reward\":0.5,\"loss\":0.9}]', ?)",
        (env_id, db.utcnow()),
    )
    conn.execute(
        "INSERT INTO adapters (train_run_id, base_model, step, path, created_at)"
        " VALUES (1, 'Qwen/Qwen3-0.6B', 1, 'adapters/run1/step1', ?)",
        (db.utcnow(),),
    )
    conn.commit()


def test_overview_empty_lab(client):
    data = client.get("/api/overview").json()
    assert data["evals"]["total"] == 0
    assert data["best"] is None


def test_full_api_surface(client):
    conn = db.connect()
    _seed(conn)
    conn.close()

    overview = client.get("/api/overview").json()
    assert overview["evals"]["done"] == 1
    assert overview["best"]["mean_reward"] == 0.75
    assert overview["recent_trains"][0]["rewards"] == [0.3, 0.5]

    evals = client.get("/api/evals").json()
    assert evals[0]["env"] == "primeintellect/gsm8k"
    assert evals[0]["samples"] == 1

    detail = client.get("/api/evals/1").json()
    assert detail["rollouts"][0]["completion"][0]["content"] == "a"
    assert detail["meta"]["avg_metrics"]["accuracy"] == 0.75
    assert client.get("/api/evals/99").status_code == 404

    training = client.get("/api/training").json()
    assert training[0]["rewards"] == [0.3, 0.5]
    tdetail = client.get("/api/training/1").json()
    assert tdetail["adapters"][0]["base_model"] == "Qwen/Qwen3-0.6B"
    assert tdetail["config_toml"] == "max_steps = 2"

    assert client.get("/api/deployments").json() == []


def test_ui_shell_served(client):
    for path in ("/", "/evals", "/anything"):
        response = client.get(path)
        assert response.status_code == 200
        assert "nanolab" in response.text
    assert "sidebar" in client.get("/assets/app.css").text
    assert "router" in client.get("/assets/app.js").text
