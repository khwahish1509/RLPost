"""Offline mechanics tests for the memory-stream environment (the S0 gate).

Everything runs with the fake reader — no API, no network. The Lift machinery
must behave correctly on the extremes before any GPU time is spent:
perfect scribe → high lift · lazy scribe → zero · sloppy scribe → punished.
"""

from __future__ import annotations

import asyncio

import pytest

memory_stream = pytest.importorskip("memory_stream")

from memory_stream import MemoryStreamEnv, load_environment  # noqa: E402
from memory_stream.dialogues import EVAL_SEED_BASE, generate_stream  # noqa: E402
from memory_stream.reader import FakeReader  # noqa: E402


def test_streams_are_deterministic_and_well_formed():
    a = generate_stream(7)
    assert a.to_info() == generate_stream(7).to_info()
    assert a.to_info() != generate_stream(8).to_info()
    assert len(a.sessions) == 4
    kinds = {q.kind for q in a.questions}
    assert "abstention" in kinds and "update" in kinds
    # every questioned fact value was actually said in some session
    text = "\n".join(a.sessions)
    for q in a.questions:
        if q.kind != "abstention":
            assert q.answer in text, f"answer {q.answer!r} never said"
        else:
            assert q.trap in text and q.answer == "unknown"
    # updates: stale value said earlier, new value said later
    for q in a.questions:
        if q.kind == "update":
            assert q.stale in text and q.answer in text


def test_fake_reader_semantics():
    r = FakeReader()

    async def check(q, nb):
        return (await r.read(q, nb)).correct

    fact = {"kind": "fact", "answer": "Pune"}
    upd = {"kind": "update", "answer": "Pune", "stale": "Berlin"}
    abst = {"kind": "abstention", "answer": "unknown", "trap": "Rex"}
    assert asyncio.run(check(fact, "city: Pune")) is True
    assert asyncio.run(check(fact, "nothing")) is False
    assert asyncio.run(check(upd, "city: Pune")) is True
    assert asyncio.run(check(upd, "city: Berlin -> Pune")) is False  # append-only fails
    assert asyncio.run(check(abst, "clean notes")) is True
    assert asyncio.run(check(abst, "dog Rex barking")) is False      # sloppy attribution fails


def _run_episode(notebook_strategy) -> tuple:
    from verifiers.types import State

    env = load_environment(reader_model="fake", num_train_streams=2)
    ds = env.get_dataset()
    row = ds[0]

    async def episode():
        state = State.for_task(row)
        state["trajectory"] = []
        await env.setup_state(state)
        messages = list(row["prompt"])
        while state.get("final_env_response") is None:
            notebook = notebook_strategy(state)
            messages = messages + [{"role": "assistant", "content": notebook}]
            response = await env.env_response(messages, state)
            messages = messages + response
        return env, state

    return asyncio.run(episode())


def _score(env, state) -> float:
    async def s():
        await env.rubric.score_rollout(state)
    asyncio.run(s())
    return state["reward"]


def test_perfect_scribe_gets_high_lift():
    def perfect(state):
        # keep the LATEST value of every user fact; nothing else
        qs = state["mem_questions"]
        lines = [f"{q['text']} -> {q['answer']}" for q in qs if q["kind"] != "abstention"]
        return "\n".join(lines)

    env, state = _run_episode(perfect)
    lift = _score(env, state)
    assert lift == pytest.approx(1.0 - state["baseline_score"])
    assert state["with_notes_score"] == pytest.approx(1.0)


def test_lazy_scribe_gets_zero_lift():
    env, state = _run_episode(lambda s: "nothing useful recorded")
    assert _score(env, state) == pytest.approx(0.0)
    # baseline: facts fail, abstention passes — same with or without the notes
    assert state["with_notes_score"] == pytest.approx(state["baseline_score"])


def test_sloppy_scribe_is_punished():
    def sloppy(state):
        # dumps every session verbatim — junk, traps and stale values included
        i = state["next_session"]
        return "\n".join(state["mem_sessions"][:max(1, i)])[:600]

    env, state = _run_episode(sloppy)
    _score(env, state)
    # dumping must not beat a careful notebook: traps + stale values cost it
    assert state["with_notes_score"] < 1.0


def test_notebook_cap_enforced():
    env = load_environment(reader_model="fake", notebook_char_cap=100,
                           num_train_streams=2)
    assert isinstance(env, MemoryStreamEnv)
    from verifiers.types import State

    ds = env.get_dataset()
    state = State.for_task(ds[0])
    state["trajectory"] = []

    async def one_turn():
        await env.setup_state(state)
        messages = list(ds[0]["prompt"]) + [
            {"role": "assistant", "content": "x" * 10_000}]
        await env.env_response(messages, state)

    asyncio.run(one_turn())
    assert len(state["notebook"]) == 100


def test_eval_streams_are_held_out():
    env = load_environment(reader_model="fake", num_train_streams=3, num_eval_streams=3)
    train_seeds = {r["info"]["seed"] for r in env.get_dataset()}
    eval_seeds = {r["info"]["seed"] for r in env.get_eval_dataset()}
    assert train_seeds.isdisjoint(eval_seeds)
    assert min(eval_seeds) >= EVAL_SEED_BASE
