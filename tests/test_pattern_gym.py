"""Offline tests for pattern-gym: determinism, solvability, splits, scoring."""

from __future__ import annotations

import asyncio

import pytest

pattern_gym = pytest.importorskip("pattern_gym")

from pattern_gym import correct, load_environment, parse_answer  # noqa: E402
from pattern_gym.rules import EVAL_SEED_BASE, generate_task  # noqa: E402


def test_tasks_deterministic_and_consistent():
    a = generate_task(7)
    b = generate_task(7)
    assert (a.demos, a.test_input, a.answer) == (b.demos, b.test_input, b.answer)
    assert generate_task(8).answer != a.answer or generate_task(8).demos != a.demos
    # demos and test use 4 distinct inputs
    inputs = [x for x, _ in a.demos] + [a.test_input]
    assert len(set(inputs)) == 4


@pytest.mark.parametrize("difficulty,tiers", [
    ("easy", {1}), ("medium", {2}), ("hard", {3}), ("mixed", {1, 2, 3}),
])
def test_difficulty_controls_tiers(difficulty, tiers):
    seen = {generate_task(s, difficulty).tier for s in range(60)}
    assert seen <= tiers
    if difficulty == "mixed":
        assert seen == tiers  # all tiers appear in a mixed sample


def test_parse_and_score():
    assert parse_answer("thinking...\n\\boxed{HELLO}") == "HELLO"
    assert parse_answer("no box") is None
    completion = [{"role": "assistant", "content": "\\boxed{abc}"}]
    assert correct(completion, "abc") == 1.0
    assert correct(completion, "abd") == 0.0


def test_env_loads_and_splits_are_disjoint():
    env = load_environment(num_train_examples=5, num_eval_examples=5)
    train = env.get_dataset()
    ev = env.get_eval_dataset()
    train_seeds = {r["info"]["seed"] for r in train}
    eval_seeds = {r["info"]["seed"] for r in ev}
    assert train_seeds.isdisjoint(eval_seeds)
    assert min(eval_seeds) >= EVAL_SEED_BASE
    assert "boxed" in train[0]["prompt"][0]["content"]  # system prompt present


def test_rubric_scores_real_rollout_offline():
    from verifiers.types import State

    env = load_environment(num_train_examples=2)
    ds = env.get_dataset()
    row = ds[0]
    good = State.for_task(row)
    good["completion"] = [
        {"role": "assistant", "content": f"the rule is X\n\\boxed{{{row['answer']}}}"}
    ]
    good["trajectory"] = []
    bad = State.for_task(row)
    bad["completion"] = [{"role": "assistant", "content": "\\boxed{wrong}"}]
    bad["trajectory"] = []

    async def score():
        await env.rubric.score_rollout(good)
        await env.rubric.score_rollout(bad)

    asyncio.run(score())
    assert good["reward"] == 1.0
    assert bad["reward"] == 0.0
