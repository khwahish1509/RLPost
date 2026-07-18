"""Seeded rule generators for pattern-gym.

A task shows three input→output demonstrations of a hidden transformation
and one test input; the answer is the transformed test input, checked
exactly. Rules are grouped in tiers so the environment's difficulty can be
tuned to land a model's baseline inside the 10–80% trainability window —
GRPO's precondition, designed in rather than hoped for.

Determinism: everything derives from random.Random(f"pattern:{seed}").
Train seeds start at 0; eval seeds at EVAL_SEED_BASE — held out by
construction.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

EVAL_SEED_BASE = 100_000

VOWELS = "aeiou"


def _word(rng: random.Random, lo: int = 4, hi: int = 8) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def _num(rng: random.Random) -> int:
    return rng.randint(12, 97)


@dataclass
class Rule:
    name: str
    tier: int                # 1 easy · 2 medium · 3 used in compositions
    kind: str                # "str" or "int"
    describe: str            # internal note (never shown to the model)
    apply: callable
    gen_input: callable


def _shift_char(c: str, k: int) -> str:
    return chr((ord(c) - 97 + k) % 26 + 97)


def _make_rules(rng: random.Random) -> list[Rule]:
    """Instantiate the rule pool; parameterized rules draw their constant
    from the stream's rng so every seed sees different constants."""
    k_add = rng.randint(3, 19)
    k_mul = rng.choice([2, 3, 4, 6, 7])
    k_shift = rng.randint(1, 5)
    return [
        Rule("reverse", 1, "str", "reverse the string",
             lambda s: s[::-1], _word),
        Rule("double_chars", 1, "str", "double every character",
             lambda s: "".join(c + c for c in s), _word),
        Rule("swap_ends", 1, "str", "swap first and last characters",
             lambda s: s[-1] + s[1:-1] + s[0], _word),
        Rule("upper", 1, "str", "uppercase everything",
             lambda s: s.upper(), _word),
        Rule(f"add_{k_add}", 1, "int", f"add {k_add}",
             lambda n: n + k_add, _num),
        Rule(f"caesar_{k_shift}", 2, "str", f"shift letters by {k_shift}",
             lambda s: "".join(_shift_char(c, k_shift) for c in s), _word),
        Rule("sort_letters", 2, "str", "sort the letters",
             lambda s: "".join(sorted(s)), _word),
        Rule("upper_vowels", 2, "str", "uppercase only the vowels",
             lambda s: "".join(c.upper() if c in VOWELS else c for c in s), _word),
        Rule(f"mul_{k_mul}", 2, "int", f"multiply by {k_mul}",
             lambda n: n * k_mul, _num),
        Rule("digit_sum", 2, "int", "sum of digits",
             lambda n: sum(int(d) for d in str(n)), _num),
    ]


def _compose(rng: random.Random, rules: list[Rule]) -> Rule:
    """Tier-3: two lowercase-preserving string rules applied in sequence
    (case-changing rules break letter-shift semantics when chained)."""
    pool = [
        r for r in rules
        if r.kind == "str" and r.name not in ("upper", "upper_vowels")
    ]
    a, b = rng.sample(pool, 2)
    return Rule(
        name=f"{a.name}+{b.name}",
        tier=3,
        kind="str",
        describe=f"{a.describe}, then {b.describe}",
        apply=lambda s: b.apply(a.apply(s)),
        gen_input=_word,
    )


@dataclass
class Task:
    seed: int
    family: str
    tier: int
    demos: list[tuple[str, str]]
    test_input: str
    answer: str


def generate_task(seed: int, difficulty: str = "mixed") -> Task:
    rng = random.Random(f"pattern:{seed}")
    rules = _make_rules(rng)
    if difficulty == "easy":
        pool = [r for r in rules if r.tier == 1]
        rule = rng.choice(pool)
    elif difficulty == "medium":
        pool = [r for r in rules if r.tier == 2]
        rule = rng.choice(pool)
    elif difficulty == "hard":
        rule = _compose(rng, rules)
    else:  # mixed: 40% easy, 40% medium, 20% composed
        roll = rng.random()
        if roll < 0.4:
            rule = rng.choice([r for r in rules if r.tier == 1])
        elif roll < 0.8:
            rule = rng.choice([r for r in rules if r.tier == 2])
        else:
            rule = _compose(rng, rules)

    inputs: set = set()
    while len(inputs) < 4:
        inputs.add(rule.gen_input(rng))
    demo_in = sorted(inputs, key=str)[:3]
    test_input = sorted(inputs, key=str)[3]
    demos = [(str(x), str(rule.apply(x))) for x in demo_in]
    return Task(
        seed=seed,
        family=rule.name,
        tier=rule.tier,
        demos=demos,
        test_input=str(test_input),
        answer=str(rule.apply(test_input if rule.kind == "str" else int(test_input))),
    )


def render_prompt(task: Task) -> str:
    lines = ["A hidden rule transforms inputs into outputs:"]
    for i, (x, y) in enumerate(task.demos, 1):
        lines.append(f"  {x} -> {y}")
    lines.append("")
    lines.append(f"Apply the same rule to: {task.test_input}")
    return "\n".join(lines)
