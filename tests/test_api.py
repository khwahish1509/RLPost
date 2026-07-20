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


def test_eval_action_runs_cli_subprocess(client, monkeypatch):
    import time

    from nanolab import api as api_mod

    api_mod.JOBS.clear()
    captured = {}
    monkeypatch.setattr(api_mod, "_run_subprocess", lambda cmd: captured.setdefault("cmd", cmd))
    monkeypatch.setenv("NANOLAB_API_BASE_URL", "http://endpoint")
    monkeypatch.setenv("NANOLAB_API_KEY_VAR", "FAKE_KEY")
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setenv("NANOLAB_DEFAULT_MODEL", "default-model")

    resp = client.post("/api/actions/eval", json={"env": "gsm8k", "n": 3})
    assert resp.status_code == 200
    for _ in range(100):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs[0]["status"] == "done"
    cmd = captured["cmd"]
    # the exact same CLI a terminal would use, main-thread safe
    assert cmd[1:6] == ["-m", "nanolab.cli", "eval", "run", "gsm8k"]
    assert "default-model" in cmd and "--force" in cmd and "3" in cmd


def test_eval_action_validates_configuration(client, monkeypatch):
    for var in ("NANOLAB_API_BASE_URL", "NANOLAB_API_KEY_VAR", "NANOLAB_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert client.post("/api/actions/eval", json={}).status_code == 400
    assert client.post("/api/actions/eval", json={"env": "x", "model": "m"}).status_code == 400


def test_install_action_runs_as_background_job(client, monkeypatch):
    import time

    from nanolab import api as api_mod
    from nanolab import envs

    api_mod.JOBS.clear()
    called = {}
    monkeypatch.setattr(envs, "install", lambda slug: called.setdefault("slug", slug))
    resp = client.post("/api/actions/install", json={"slug": "owner/thing"})
    assert resp.status_code == 200
    for _ in range(100):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs[0]["status"] == "done" and called["slug"] == "owner/thing"


def test_failed_job_carries_error(client, monkeypatch):
    import time

    from nanolab import api as api_mod
    from nanolab import envs

    api_mod.JOBS.clear()

    def boom(slug):
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(envs, "install", boom)
    client.post("/api/actions/install", json={"slug": "x/y"})
    for _ in range(100):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs[0]["status"] == "failed"
    assert "hub exploded" in jobs[0]["error"]


def test_serve_command_local_vs_vllm():
    from pathlib import Path

    from nanolab.serve import _serve_command

    local = _serve_command("Qwen/Qwen3-0.6B", Path("adapters/x"), "adapter-5", 8765, True)
    assert local[1:3] == ["-m", "nanolab.serve_local"]
    assert "--adapter" in local and "adapters/x" in local
    vllm = _serve_command("Qwen/Qwen3-0.6B", Path("adapters/x"), "adapter-5", 8000, False)
    assert vllm[0] == "vllm" and "--enable-lora" in vllm


def test_policy_server_sampling_passthrough():
    import httpx

    from nanolab.policy_server import PolicyServer

    seen = {}

    def gen(messages, temperature=None, max_tokens=None):
        seen.update(temperature=temperature, max_tokens=max_tokens)
        return "ok"

    with PolicyServer(gen, pass_sampling=True) as server:
        resp = httpx.post(
            f"{server.base_url}/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}],
                  "temperature": 0.0, "max_tokens": 32},
            timeout=10.0,
        )
    assert resp.status_code == 200
    assert seen == {"temperature": 0.0, "max_tokens": 32}


def test_pick_device_returns_valid_pair():
    pytest.importorskip("torch")
    from nanolab.serve_local import pick_device

    device, dtype = pick_device()
    assert device in ("cuda", "mps", "cpu")


def test_adapters_endpoint_and_deploy_action(client, monkeypatch):
    import time

    from nanolab import api as api_mod
    from nanolab import serve as serve_mod

    conn = db.connect()
    _seed(conn)
    conn.close()

    adapters = client.get("/api/adapters").json()
    assert adapters[0]["base_model"] == "Qwen/Qwen3-0.6B"
    assert adapters[0]["deployed"] is False

    api_mod.JOBS.clear()
    called = {}
    monkeypatch.setattr(
        serve_mod, "create_deployment",
        lambda adapter_id, **kw: called.update(adapter_id=adapter_id, local=kw.get("local")),
    )
    resp = client.post("/api/actions/deploy", json={"adapter_id": 1})
    assert resp.status_code == 200
    for _ in range(100):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs[0]["status"] == "done"
    assert called == {"adapter_id": 1, "local": True}


def test_hub_browse_marks_installed(client, monkeypatch):
    import json
    import subprocess

    from nanolab import api as api_mod

    conn = db.connect()
    db.register_environment(conn, "primeintellect/gsm8k", "gsm8k", "0.1.3")
    conn.close()

    fake = json.dumps({"environments": [
        {"environment": "hud/hud-text-2048", "stars": 33, "tags": ["game"],
         "description": "2048", "version": "0.1.0"},
        {"environment": "primeintellect/gsm8k", "stars": 5, "tags": ["math"],
         "description": "grade school math", "version": "0.1.3"},
    ]})

    class FakeProc:
        returncode = 0
        stdout = fake

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    resp = client.get("/api/hub?search=math&sort=stars")
    assert resp.status_code == 200
    envs = resp.json()["environments"]
    by_name = {e["environment"]: e for e in envs}
    assert by_name["hud/hud-text-2048"]["installed"] is False
    assert by_name["primeintellect/gsm8k"]["installed"] is True


def test_hub_handles_cli_failure(client, monkeypatch):
    import subprocess

    class FailProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FailProc())
    data = client.get("/api/hub").json()
    assert data["environments"] == []
    assert "error" in data


def test_console_runs_nanolab_commands(client):
    import time

    from nanolab import api as api_mod

    api_mod.JOBS.clear()
    resp = client.post("/api/actions/cli", json={"command": "uv run nanolab version"})
    assert resp.status_code == 200
    for _ in range(200):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running" and jobs[0].get("output"):
            break
        time.sleep(0.05)
    assert jobs[0]["status"] == "done"
    assert "0.1.0" in jobs[0]["output"]


def test_console_blocks_ui_command(client):
    resp = client.post("/api/actions/cli", json={"command": "ui"})
    assert resp.status_code == 400
    assert client.post("/api/actions/cli", json={"command": ""}).status_code == 400


def test_configs_and_train_cloud_action(client, monkeypatch, tmp_path):
    import time

    from nanolab import api as api_mod
    from nanolab import cloud

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "demo.toml").write_text(
        'model = "m"\nmax_steps = 5\nlearning_rate = 1e-5\n[[env]]\nid = "gsm8k"\n'
    )
    configs = client.get("/api/configs").json()
    assert configs[0]["name"] == "demo" and configs[0]["env"] == "gsm8k"

    api_mod.JOBS.clear()
    pushed = {}
    monkeypatch.setattr(cloud, "push", lambda c: pushed.setdefault("config", c))
    resp = client.post("/api/actions/train-cloud", json={"config": "configs/demo.toml"})
    assert resp.status_code == 200
    for _ in range(100):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs[0]["status"] == "done" and pushed["config"] == "configs/demo.toml"
    # path traversal / junk rejected
    assert client.post("/api/actions/train-cloud", json={"config": "../evil.toml"}).status_code == 400
    assert client.post("/api/actions/train-cloud", json={}).status_code == 400


def test_console_rejects_arbitrary_commands(client):
    resp = client.post("/api/actions/cli", json={"command": "rm -rf /"})
    assert resp.status_code == 400
    assert "read-only" in resp.json()["error"]


def test_chat_action_requires_running_deployment(client):
    resp = client.post("/api/actions/chat", json={})
    assert resp.status_code == 400
    resp = client.post(
        "/api/actions/chat",
        json={"deployment_id": 99, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert "not running" in resp.json()["error"]


def test_ui_shell_served(client):
    for path in ("/", "/evals", "/anything"):
        response = client.get(path)
        assert response.status_code == 200
        assert "nanolab" in response.text
    assert "sidebar" in client.get("/assets/app.css").text
    assert "router" in client.get("/assets/app.js").text
