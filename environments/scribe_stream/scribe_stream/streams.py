"""Seeded stream generator: chains of tasks where later tasks need numbers
revealed by earlier ones.

Each stream is N_TASKS tasks. Task 1 is self-contained. Every later task
requires exactly one "figure" — a value revealed only after an earlier task —
plus local numbers stated in the task itself. A Player seeing tasks one at a
time, statelessly, cannot know foreign figures unless the Scribe's notebook
carries them. That dependency is what makes Lift measurable.

Determinism: everything derives from `random.Random(f"stream:{seed}")`, so a
stream id is a complete description of the episode. Train streams use seeds
[0, num_train); eval streams use [EVAL_SEED_BASE, ...) — held out by
construction.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

N_TASKS = 8
EVAL_SEED_BASE = 10_000

ITEMS = [
    "widgets", "gears", "bolts", "valves", "pumps",
    "coils", "rotors", "filters", "springs", "bearings",
]


@dataclass
class Task:
    idx: int                 # 0-based
    text: str
    answer: int
    reveal: str              # the RECORD line shown after the task is played
    foreign_values: list[int] = field(default_factory=list)  # needed from notes


@dataclass
class Stream:
    seed: int
    tasks: list[Task]

    def to_info(self) -> dict:
        return {"seed": self.seed, "tasks": [asdict(t) for t in self.tasks]}


def _figure_name(idx: int, item: str) -> str:
    return f"figure #{idx + 1} ({item})"


def generate_stream(seed: int) -> Stream:
    rng = random.Random(f"stream:{seed}")
    items = rng.sample(ITEMS, N_TASKS)
    tasks: list[Task] = []
    figures: list[int] = []  # figures[i] = revealed answer of task i

    # task 1: self-contained
    q, u = rng.randint(12, 48), rng.randint(3, 9)
    total = q * u
    tasks.append(
        Task(
            idx=0,
            text=(
                f"A shipment of {items[0]} arrived: {q} boxes with {u} units "
                f"each. How many units in total?"
            ),
            answer=total,
            reveal=f"{_figure_name(0, items[0])} = {total}",
        )
    )
    figures.append(total)

    for i in range(1, N_TASKS):
        j = rng.randrange(len(figures))  # which earlier figure this task needs
        v = figures[j]
        ref = _figure_name(j, items[j])
        kind = rng.choice(["add", "sub", "mul", "combine"] if i >= 2 else ["add", "sub", "mul"])
        if kind == "add":
            a, b = rng.randint(15, 60), rng.randint(2, 8)
            local = a * b
            answer = v + local
            text = (
                f"A new shipment of {items[i]} arrived: {a} boxes with {b} "
                f"units each. Add {ref}, which you learned earlier, to this "
                f"shipment's total units. What is the sum?"
            )
            foreign = [v]
        elif kind == "sub":
            d = rng.randint(2, max(3, v // 3))
            answer = v - d
            text = (
                f"{d} units were removed from {ref}, which you learned "
                f"earlier. How many units remain?"
            )
            foreign = [v]
        elif kind == "mul":
            p = rng.randint(2, 5)
            answer = v * p
            text = (
                f"Each unit counted in {ref}, which you learned earlier, is "
                f"worth ${p}. What is the total value in dollars?"
            )
            foreign = [v]
        else:  # combine two earlier figures
            k = rng.randrange(len(figures))
            if k == j:
                k = (j + 1) % len(figures)
            v2 = figures[k]
            ref2 = _figure_name(k, items[k])
            answer = v + v2
            text = (
                f"Add {ref} and {ref2}, both of which you learned earlier. "
                f"What is their sum?"
            )
            foreign = sorted({v, v2})
        tasks.append(
            Task(
                idx=i,
                text=text,
                answer=answer,
                reveal=f"{_figure_name(i, items[i])} = {answer}",
                foreign_values=foreign,
            )
        )
        figures.append(answer)

    return Stream(seed=seed, tasks=tasks)
