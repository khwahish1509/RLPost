"""Offline mechanics tests for the scribe-stream environment.

Everything runs with the fake Player — no API, no network. These tests are
the S0 gate: the Lift machinery must behave correctly on the extremes
(perfect scribe → high lift, lazy scribe → zero lift) before any API money
or training time is spent.
"""

from __future__ import annotations

import asyncio

import pytest

scribe_stream = pytest.importorskip("scribe_stream")

from scribe_stream import ScribeStreamEnv, load_environment  # noqa: E402
from scribe_stream.player import FakePlayer, parse_int  # noqa: E402
from scribe_stream.streams import EVAL_SEED_BASE, N_TASKS, generate_stream  # noqa: E402


def test_streams_are_deterministic_and_solvable():
    a = generate_stream(5)
    b = generate_stream(5)
    assert a.to_info() == b.to_info()
    assert a.to_info() != generate_stream(6).to_info()
    assert len(a.tasks) == N_TASKS

    revealed: list[int] = []
    for task in a.tasks:
        # every foreign figure must have been revealed by an earlier task
        for v in task.foreign_values:
            assert v in revealed, f"task {task.idx} needs unrevealed figure {v}"
        assert isinstance(task.answer, int)
        assert str(task.answer) in task.reveal
        revealed.append(task.answer)
    # task 1 self-contained; all later tasks depend on notes
    assert a.tasks[0].foreign_values == []
    assert all(t.foreign_values for t in a.tasks[1:])


def test_longer_horizon_keeps_unique_referenceable_labels():
    # beyond the 10 base item names, labels must stay unique so every figure
    # is still referenceable by name — otherwise the notebook is ambiguous
    stream = generate_stream(EVAL_SEED_BASE, num_tasks=16)
    assert len(stream.tasks) == 16
    labels = [t.reveal.split(" = ")[0] for t in stream.tasks]
    assert len(set(labels)) == len(labels)
    revealed: list[int] = []
    for task in stream.tasks:
        for v in task.foreign_values:
            assert v in revealed
        revealed.append(task.answer)


def test_num_tasks_flows_through_env():
    env = load_environment(player_model="fake", num_tasks=12, num_train_streams=2)
    assert env.max_turns == 12 + 2
    assert len(env.get_dataset()[0]["info"]["tasks"]) == 12


def test_hard_mode_adds_tagged_distractors_without_breaking_solvability():
    plain = generate_stream(10_000, num_tasks=8)
    hard = generate_stream(10_000, num_tasks=8, distractors_per_task=3, mark_reuse=True)

    # tasks, answers and dependencies are identical — only the RECORD changes
    assert [t.answer for t in hard.tasks] == [t.answer for t in plain.tasks]
    assert [t.foreign_values for t in hard.tasks] == [t.foreign_values for t in plain.tasks]

    real_values = {t.answer for t in hard.tasks}
    for i, task in enumerate(hard.tasks):
        lines = task.reveal.splitlines()
        assert len(lines) == 1 + 3  # the real figure + 3 distractors
        # the real figure is still present and still solvable from the record
        assert f"= {task.answer}" in task.reveal
        assert ("(needed later)" in task.reveal) or ("(one-off)" in task.reveal)
        # distractor values never coincide with a real figure (can't fake Lift)
        distractor_lines = [ln for ln in lines if "sample)" in ln]
        assert len(distractor_lines) == 3
        for ln in distractor_lines:
            dval = int(ln.split("= ")[1].split(" ")[0])
            assert dval not in real_values


def test_hard_mode_flows_through_env_arg():
    env = load_environment(
        player_model="fake", num_train_streams=2,
        distractors_per_task=2, mark_reuse=True,
    )
    record = env.get_dataset()[0]["info"]["tasks"][0]["reveal"]
    assert record.count("\n") == 2  # 1 real + 2 distractors → 2 newlines


def test_parse_int():
    assert parse_int("the answer is \\boxed{42}") == 42
    assert parse_int("= 1,234 total") == 1234
    assert parse_int("no numbers") is None


def _run_episode(notebook_strategy) -> dict:
    """Drive a full episode by hand: setup_state + env_response per turn."""
    from verifiers.types import State

    env = load_environment(player_model="fake", num_train_streams=2)
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


def test_perfect_scribe_gets_full_lift():
    def perfect(state):
        # write down every figure revealed so far
        revealed = [t["reveal"] for t in state["stream_tasks"][: state["next_task"]]]
        return "\n".join(revealed)

    env, state = _run_episode(perfect)
    results = state["player_results"]
    assert len(results) == N_TASKS - 1
    assert all(r["correct"] for r in results)

    async def score():
        await env.rubric.score_rollout(state)

    asyncio.run(score())
    assert state["reward"] == pytest.approx(1.0)  # lift: 1.0 with − 0.0 without
    assert state["metrics"]["player_score"] == pytest.approx(1.0)
    assert state["metrics"]["baseline_score"] == pytest.approx(0.0)


def test_lazy_scribe_gets_zero_lift():
    env, state = _run_episode(lambda state: "nothing useful")
    assert not any(r["correct"] for r in state["player_results"])

    async def score():
        await env.rubric.score_rollout(state)

    asyncio.run(score())
    assert state["reward"] == pytest.approx(0.0)


def test_notebook_cap_enforced():
    env = load_environment(player_model="fake", notebook_char_cap=100)
    assert isinstance(env, ScribeStreamEnv)

    from verifiers.types import State

    ds = env.get_dataset()
    state = State.for_task(ds[0])
    state["trajectory"] = []

    async def one_turn():
        await env.setup_state(state)
        messages = list(ds[0]["prompt"]) + [
            {"role": "assistant", "content": "x" * 10_000}
        ]
        await env.env_response(messages, state)

    asyncio.run(one_turn())
    assert len(state["notebook"]) == 100  # log-dumping is physically impossible


def test_eval_streams_are_held_out():
    env = load_environment(player_model="fake", num_train_streams=3, num_eval_streams=3)
    train_seeds = {r["info"]["seed"] for r in env.get_dataset()}
    eval_seeds = {r["info"]["seed"] for r in env.get_eval_dataset()}
    assert train_seeds.isdisjoint(eval_seeds)
    assert min(eval_seeds) >= EVAL_SEED_BASE
