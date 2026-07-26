"""Memory Agent tests — network-free: every model call is monkeypatched.

The contracts that matter: sessions persist in the db, the assistant is
amnesiac by construction (sees notebook + newest message only), the notebook
cap is enforced in code, and writer=none really is the ablation.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from nanolab import agent, db
from nanolab.api import build_app


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOLAB_DB", str(tmp_path / "lab.db"))
    monkeypatch.chdir(tmp_path)
    db.connect().close()  # create schema
    monkeypatch.setattr(
        agent, "_default_endpoint", lambda: ("http://fake/v1", "k", "fake-model")
    )
    return tmp_path


def test_session_persists_and_resets(lab):
    s1 = agent.current_session()
    assert s1["writer"] == "trained" and s1["notebook"] == ""
    assert agent.current_session()["id"] == s1["id"]  # stable across calls
    s2 = agent.reset_session()
    assert s2["id"] > s1["id"]


def test_writer_switch_validates(lab):
    assert agent.set_writer("prompted")["writer"] == "prompted"
    with pytest.raises(agent.AgentError):
        agent.set_writer("clever-nonsense")


def test_chat_turn_is_amnesiac_and_records_control(lab, monkeypatch):
    seen = []

    def fake_chat(base, key, model, messages, **kw):
        seen.append(messages)
        return "reply-text"

    monkeypatch.setattr(agent, "_chat", fake_chat)
    agent.set_writer("none")  # no rewrite thread in this test
    s = agent.chat_turn("my name is Khwahish")

    # two calls: the agent (with notebook) and the control (empty notebook)
    assert len(seen) == 2
    for messages in seen:
        # amnesia contract: system + newest user message ONLY — no transcript
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "my name is Khwahish"
    assert "(empty)" in seen[0][0]["content"]  # notebook rendered into system
    assert s["turns"] == 1
    assert s["messages"][-1]["content"] == "reply-text"
    assert s["messages"][-1]["control"] == "reply-text"
    # writer=none: the ablation — nothing rewrites, nothing pending
    assert s["scribe_state"] == "idle" and s["notebook"] == ""


def test_rewrite_enforces_cap_in_code(lab, monkeypatch):
    monkeypatch.setattr(agent, "_chat", lambda *a, **k: "x" * 5000)
    s = agent.current_session()
    agent._rewrite_notebook(s["id"], "prompted", "", "hi", "hello", 1)
    after = agent.current_session()
    assert len(after["notebook"]) == agent.NOTEBOOK_CAP  # truncation, not trust
    assert after["scribe_state"] == "idle"


def test_rewrite_failure_is_surfaced_not_hidden(lab, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(agent, "_chat", boom)
    s = agent.current_session()
    agent._update(s["id"], notebook="old notes")
    agent._rewrite_notebook(s["id"], "prompted", "old notes", "hi", "hello", 1)
    after = agent.current_session()
    assert after["scribe_state"].startswith("error:")
    assert after["notebook"] == "old notes"  # a failed rewrite never eats memory


def test_api_routes(lab):
    client = TestClient(build_app())
    state = client.get("/api/agent").json()
    assert state["writer"] == "trained" and "notebook" in state

    bad = client.post("/api/actions/agent-writer", json={"writer": "nope"})
    assert bad.status_code == 400

    ok = client.post("/api/actions/agent-writer", json={"writer": "none"})
    assert ok.json()["writer"] == "none"

    fresh = client.post("/api/actions/agent-reset", json={})
    assert fresh.json()["id"] > state["id"]

    empty = client.post("/api/actions/agent-chat", json={"message": "  "})
    assert empty.status_code == 400
