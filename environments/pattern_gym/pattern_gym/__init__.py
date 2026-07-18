"""pattern-gym: rule-induction micro-tasks with tunable difficulty.

Three demonstrations of a hidden transformation, one test input, exact-match
scoring. Single-turn — trainable by any GRPO loop that handles one question →
one answer → one grade, on a free T4.

Design goals (in order):
1. Verifiable by code, impossible to sweet-talk (exact string match).
2. Difficulty as a first-class knob (easy/medium/hard/mixed) so a target
   model's baseline can be *placed* inside the 10–80% trainability window.
3. Seeded and split: train seeds and eval seeds never overlap.
4. Rule *families*, so streams of related tasks can later be built on top —
   notes like "this family shifts letters by 3" transfer within a family,
   which is exactly what memory-training needs.
"""

from __future__ import annotations

import re

import verifiers as vf

from .rules import EVAL_SEED_BASE, generate_task, render_prompt

SYSTEM_PROMPT = (
    "You infer hidden transformation rules from examples. Think briefly if "
    "needed, then reply with the transformed result inside \\boxed{...}. "
    "The answer must match exactly."
)


def _text(completion) -> str:
    """Last assistant message content — live rollouts carry message OBJECTS,
    stored rollouts carry dicts; handle both."""
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict):
                role, content = message.get("role"), message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            if role == "assistant":
                return str(content or "")
        return ""
    return str(completion)


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1].strip() if matches else None


def correct(completion, answer) -> float:
    return 1.0 if parse_answer(_text(completion)) == str(answer).strip() else 0.0


def has_boxed(completion) -> float:
    return 1.0 if parse_answer(_text(completion)) is not None else 0.0


def _build(num: int, seed_base: int, difficulty: str):
    from datasets import Dataset

    rows = []
    for seed in range(seed_base, seed_base + num):
        task = generate_task(seed, difficulty)
        rows.append(
            {
                "question": render_prompt(task),
                "answer": task.answer,
                "info": {"family": task.family, "tier": task.tier, "seed": task.seed},
            }
        )
    return Dataset.from_list(rows)


def load_environment(
    difficulty: str = "mixed",
    num_train_examples: int = 2000,
    num_eval_examples: int = 200,
):
    if difficulty not in ("easy", "medium", "hard", "mixed"):
        raise ValueError("difficulty must be easy | medium | hard | mixed")
    rubric = vf.Rubric(funcs=[correct, has_boxed], weights=[1.0, 0.0])
    return vf.SingleTurnEnv(
        dataset=lambda: _build(num_train_examples, 0, difficulty),
        eval_dataset=lambda: _build(num_eval_examples, EVAL_SEED_BASE, difficulty),
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )
