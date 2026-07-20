"""Tests for cloud training (Kaggle client mocked) and artifact merging."""

from __future__ import annotations

import json
import sqlite3

import pytest

from nanolab import artifacts, cloud, db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOLAB_DB", str(tmp_path / "lab.db"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_kernel_slug():
    assert cloud.kernel_slug("configs/qwen3-0.6b-gsm8k.toml") == "nanolab-train-qwen3-0-6b-gsm8k"


def test_kernel_script_and_metadata(monkeypatch):
    monkeypatch.setattr(cloud, "_username", lambda: "someone")
    script = cloud.build_kernel_script("configs/x.toml")
    assert "git clone" in script and "configs/x.toml" in script
    assert "compare_adapter.py" in script and "nanolab-artifacts.zip" in script
    meta = cloud.build_kernel_metadata("configs/x.toml")
    assert meta["id"] == "someone/nanolab-train-x"
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"


def test_push_records_cloud_run(tmp_db, monkeypatch):
    (tmp_db / "configs").mkdir()
    (tmp_db / "configs" / "x.toml").write_text("max_steps = 1")
    monkeypatch.setattr(cloud, "_username", lambda: "someone")
    monkeypatch.setattr(cloud, "_kaggle", lambda args: "ok")
    ref = cloud.push("configs/x.toml")
    assert ref == "someone/nanolab-train-x"
    runs = cloud.list_runs()
    assert runs[0]["status"] == "pushed"
    assert runs[0]["kernel_ref"] == ref


def test_status_parsing(monkeypatch):
    monkeypatch.setattr(
        cloud, "_kaggle",
        lambda args: 'someone/nanolab-train-x has status "KernelWorkerStatus.RUNNING"',
    )
    assert cloud.status("someone/nanolab-train-x") == "running"


def test_merge_artifacts_roundtrip(tmp_db):
    # fabricate a "remote" artifacts directory: db + adapter files
    remote_root = tmp_db / "incoming"
    (remote_root / "results").mkdir(parents=True)
    adapter_dir = remote_root / "adapters" / "run1" / "step00009"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")

    remote = sqlite3.connect(remote_root / "results" / "nanolab.db")
    remote.executescript(db.SCHEMA)
    remote.execute(
        "INSERT INTO environments (slug, env_id, version, installed_at)"
        " VALUES ('primeintellect/gsm8k','gsm8k','0.1.3',?)", (db.utcnow(),),
    )
    remote.execute(
        "INSERT INTO train_runs (env_id, model, config_toml, status,"
        " steps_completed, reward_curve_json, started_at)"
        " VALUES (1,'Qwen/Qwen3-0.6B','max_steps=10','done',10,"
        " '[{\"step\":0,\"reward\":0.4,\"loss\":1.0}]',?)", (db.utcnow(),),
    )
    remote.execute(
        "INSERT INTO adapters (train_run_id, base_model, step, path, created_at)"
        " VALUES (1,'Qwen/Qwen3-0.6B',9,'adapters/run1/step00009',?)", (db.utcnow(),),
    )
    remote.commit()
    remote.close()

    new_ids = artifacts.merge_artifacts(remote_root)
    assert len(new_ids) == 1
    conn = db.connect()
    run = conn.execute("SELECT * FROM train_runs WHERE id=?", (new_ids[0],)).fetchone()
    adapter = conn.execute(
        "SELECT * FROM adapters WHERE train_run_id=?", (new_ids[0],)
    ).fetchone()
    conn.close()
    assert run["steps_completed"] == 10
    assert json.loads(run["reward_curve_json"])[0]["reward"] == 0.4
    from pathlib import Path

    assert Path(adapter["path"]).joinpath("adapter_model.safetensors").exists()


def test_poll_once_transitions(tmp_db, monkeypatch):
    (tmp_db / "configs").mkdir()
    (tmp_db / "configs" / "x.toml").write_text("max_steps = 1")
    monkeypatch.setattr(cloud, "_username", lambda: "someone")
    monkeypatch.setattr(cloud, "_kaggle", lambda args: "ok")
    ref = cloud.push("configs/x.toml")

    # pushed → running
    monkeypatch.setattr(cloud, "status", lambda r: "running")
    events = cloud.poll_once()
    assert any("running" in e for e in events)
    assert cloud.list_runs()[0]["status"] == "running"

    # running → complete → auto-merge (pull mocked)
    monkeypatch.setattr(cloud, "status", lambda r: "complete")
    monkeypatch.setattr(cloud, "pull", lambda r: [7])
    events = cloud.poll_once()
    assert any("merged" in e for e in events)

    # a re-push supersedes older open rows for the same kernel
    cloud.push("configs/x.toml")
    statuses = [r["status"] for r in cloud.list_runs() if r["kernel_ref"] == ref]
    assert statuses.count("pushed") == 1


def test_merge_rejects_junk(tmp_db):
    junk = tmp_db / "junk"
    junk.mkdir()
    with pytest.raises(artifacts.MergeError):
        artifacts.merge_artifacts(junk)
