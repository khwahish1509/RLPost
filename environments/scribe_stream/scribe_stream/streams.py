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


def generate_stream(
    seed: int,
    num_tasks: int = N_TASKS,
    distractors_per_task: int = 0,
    mark_reuse: bool = False,
) -> Stream:
    """Build one stream.

    Defaults reproduce the original transcription task exactly. The two extra
    knobs create a *selection* task — a curriculum where note-taking stops
    being trivial:

    - ``distractors_per_task``: each RECORD reveals this many extra one-off
      figures that are never referenced again. Copied indiscriminately they
      pad the notebook until — under a binding ``notebook_char_cap`` — the
      figures that ARE needed get truncated away. The learnable skill is
      dropping them.
    - ``mark_reuse``: tag each revealed figure ``(needed later)`` or
      ``(one-off)``, giving the Scribe the signal it must learn to act on.
    """
    rng = random.Random(f"stream:{seed}")
    # more tasks than distinct items → cycle names with a shipment suffix so
    # every figure still has a unique, referenceable label at any horizon
    if num_tasks <= len(ITEMS):
        items = rng.sample(ITEMS, num_tasks)
    else:
        items = [
            f"{ITEMS[i % len(ITEMS)]} batch-{i // len(ITEMS) + 1}"
            for i in range(num_tasks)
        ]
        rng.shuffle(items)
    tasks: list[Task] = []
    figures: list[int] = []  # figures[i] = revealed answer of task i
    referenced: set[int] = set()  # task indices some later task depends on

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

    for i in range(1, num_tasks):
        j = rng.randrange(len(figures))  # which earlier figure this task needs
        referenced.add(j)
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
            referenced.add(k)
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

    if mark_reuse or distractors_per_task:
        _decorate_records(
            rng, tasks, items, figures, referenced,
            distractors_per_task, mark_reuse,
        )
    return Stream(seed=seed, tasks=tasks)


def _decorate_records(
    rng, tasks, items, figures, referenced, distractors_per_task, mark_reuse
):
    """Rewrite each task's RECORD with reuse tags and distractor one-offs.

    Runs only when a hard-mode knob is set, so the default stream (and every
    test that pins it) is untouched. Distractor values never collide with a
    real figure, so they can't accidentally satisfy the fake Player.
    """
    real_values = set(figures)
    for i, task in enumerate(tasks):
        base = f"{_figure_name(i, items[i])} = {figures[i]}"
        if mark_reuse:
            base += " (needed later)" if i in referenced else " (one-off)"
        lines = [base]
        for d in range(distractors_per_task):
            while True:
                dval = rng.randint(100, 9999)
                if dval not in real_values:
                    break
            ditem = rng.choice(ITEMS)
            dline = f"figure #{i + 1}.{d + 1} ({ditem} sample) = {dval}"
            if mark_reuse:
                dline += " (one-off)"
            lines.append(dline)
        if distractors_per_task:
            rng.shuffle(lines)  # hide the needed figure among the noise
        task.reveal = "\n".join(lines)
