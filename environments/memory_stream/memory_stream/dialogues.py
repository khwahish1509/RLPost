"""Seeded multi-session dialogue generator for conversational memory training.

Each stream = one user talking to an assistant across SESSIONS. Woven into the
chat are:
  - STABLE facts   ("my cat is called Miso")            → asked later
  - UPDATED facts  ("actually I moved to Pune now")     → only the LATEST value counts
  - DISTRACTORS    one-off chatter, including *other people's* attributes
                   ("my neighbor's dog is called Rex")  → keeping them sloppily is
                   punished by abstention questions (see reader.py)

After the last session come QUESTIONS:
  - fact:       "What is my cat's name?"        → correct = latest value in notes
  - update:     asked about an updated fact      → correct = new value, NOT the old
  - abstention: about something the USER never said ("What is my dog's name?"
                when only the neighbor's dog was mentioned) → correct = unknown

Determinism: everything derives from random.Random(f"mem:{seed}"). Train seeds
[0, N); eval seeds [EVAL_SEED_BASE, ...) — held out by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

EVAL_SEED_BASE = 10_000

FIRST_NAMES = ["Asha", "Rohan", "Mia", "Kabir", "Lena", "Tara", "Dev", "Ira",
               "Noah", "Zara", "Vik", "Meera", "Sam", "Anya", "Ravi", "Nina"]
CITIES = ["Pune", "Austin", "Lisbon", "Osaka", "Nairobi", "Toronto", "Berlin",
          "Jaipur", "Seoul", "Denver", "Porto", "Hanoi"]
PET_KINDS = ["cat", "parrot", "rabbit", "turtle", "hamster"]
JOBS = ["nurse", "architect", "teacher", "chef", "electrician", "pharmacist",
        "translator", "photographer"]
DRINKS = ["chai", "matcha", "espresso", "lassi", "kombucha", "cocoa"]
HOBBIES = ["chess", "pottery", "bouldering", "birdwatching", "origami", "salsa"]
SIBLINGS = ["sister", "brother"]
CARS = ["Swift", "Corolla", "Model 3", "i20", "Civic", "Duster"]
ALLERGIES = ["peanuts", "shellfish", "gluten", "lactose", "soy"]
MOVIES = ["a heist movie", "an old western", "a space documentary",
          "a cooking show", "a tennis match", "a nature series"]
WEATHER = ["raining all day", "way too hot", "windy and grey", "surprisingly sunny"]

# fact key → (question, how a value is phrased into chat, value pool)
FACT_SPECS = {
    "city":    ("Which city do I live in?",        "I live in {v} these days.", CITIES),
    "pet":     ("What is my {kind}'s name?",       "we adopted a {kind} — named him {v}.", FIRST_NAMES),
    "job":     ("What is my job?",                 "work's been busy — being a {v} is no joke.", JOBS),
    "drink":   ("What is my favourite drink?",     "unwinding with my usual {v} right now.", DRINKS),
    "sibling": ("What is my {kind}'s name?",       "my {kind} {v} visited over the weekend.", FIRST_NAMES),
    "hobby":   ("What hobby did I pick up?",       "started doing {v} on Thursdays, loving it.", HOBBIES),
    "car":     ("Which car do I drive?",           "finally got the {v} serviced.", CARS),
    "allergy": ("What am I allergic to?",          "had a scare — turns out I'm allergic to {v}.", ALLERGIES),
}
UPDATABLE = {"city", "drink", "job", "car"}  # facts that can change mid-stream


@dataclass
class Fact:
    key: str            # e.g. "pet"
    kind: str           # sub-kind for pet/sibling (e.g. "cat", "sister")
    question: str
    values: list[str]   # value history; the LAST one is the truth
    said_in: list[int] = field(default_factory=list)  # session index per value


@dataclass
class Question:
    text: str
    answer: str          # expected answer; "unknown" for abstention
    kind: str            # fact | update | abstention
    stale: str = ""      # for updates: the outdated value (must NOT be the answer)
    trap: str = ""       # for abstention: a distractor value that a sloppy notebook suggests


@dataclass
class Stream:
    seed: int
    sessions: list[str]              # rendered user messages, one blob per session
    questions: list[Question]
    facts: list[Fact]

    def to_info(self) -> dict:
        return {
            "seed": self.seed,
            "sessions": self.sessions,
            "questions": [vars(q) for q in self.questions],
        }


def _chatter(rng: random.Random) -> str:
    return rng.choice([
        f"watched {rng.choice(MOVIES)} last night, decent.",
        f"weather here has been {rng.choice(WEATHER)}.",
        "long day, mostly meetings.",
        "thinking about repainting the kitchen.",
        "slept badly, don't ask.",
    ])


def _neighbor_line(rng: random.Random, trap_kind: str, trap_val: str) -> str:
    """A distractor carrying someone ELSE'S attribute — the abstention trap."""
    who = rng.choice(["my neighbor", "my coworker", "my cousin"])
    if trap_kind == "dog":
        return f"{who}'s dog {trap_val} kept barking all night."
    if trap_kind == "city":
        return f"{who} just moved to {trap_val}, lucky them."
    return f"{who} won't stop talking about their {trap_kind} {trap_val}."


def generate_stream(
    seed: int,
    num_sessions: int = 4,
    facts_per_stream: int = 5,
    updates_per_stream: int = 1,
    distractors_per_session: int = 3,
    num_questions: int = 6,
) -> Stream:
    rng = random.Random(f"mem:{seed}")

    # --- choose the user's facts
    keys = rng.sample(list(FACT_SPECS), facts_per_stream)
    facts: list[Fact] = []
    for k in keys:
        q_tmpl, say_tmpl, pool = FACT_SPECS[k]
        kind = rng.choice(PET_KINDS) if k == "pet" else rng.choice(SIBLINGS) if k == "sibling" else ""
        facts.append(Fact(key=k, kind=kind, question=q_tmpl.format(kind=kind),
                          values=[rng.choice(pool)]))

    # --- schedule updates (a later session replaces the value)
    updatable = [f for f in facts if f.key in UPDATABLE]
    rng.shuffle(updatable)
    for f in updatable[:updates_per_stream]:
        pool = FACT_SPECS[f.key][2]
        new = rng.choice([v for v in pool if v != f.values[0]])
        f.values.append(new)

    # --- the abstention trap: an attribute the USER never has, worn by a neighbor
    trap_kind = "dog"
    used = {v for f in facts for v in f.values}
    trap_val = rng.choice([n for n in FIRST_NAMES if n not in used])

    # --- lay facts into sessions
    for f in facts:
        first = rng.randrange(0, max(1, num_sessions - 1))
        f.said_in = [first]
        if len(f.values) > 1:  # update lands strictly later
            f.said_in.append(rng.randrange(first + 1, num_sessions))

    sessions: list[str] = []
    for s in range(num_sessions):
        lines: list[str] = []
        for f in facts:
            for vi, sess in enumerate(f.said_in):
                if sess != s:
                    continue
                _, say_tmpl, _ = FACT_SPECS[f.key]
                line = say_tmpl.format(v=f.values[vi], kind=f.kind)
                if vi > 0:
                    line = f"update — {line.rstrip('.')} (not {f.values[vi-1]} anymore)."
                lines.append(line)
        lines.append(_neighbor_line(rng, trap_kind, trap_val))
        for _ in range(max(0, distractors_per_session - 1)):
            lines.append(_chatter(rng))
        rng.shuffle(lines)
        sessions.append("\n".join(f"- {ln}" for ln in lines))

    # --- questions
    questions: list[Question] = []
    updated = [f for f in facts if len(f.values) > 1]
    plain = [f for f in facts if len(f.values) == 1]
    for f in updated:
        questions.append(Question(text=f.question, answer=f.values[-1],
                                  kind="update", stale=f.values[0]))
    rng.shuffle(plain)
    for f in plain:
        if len(questions) >= num_questions - 1:
            break
        questions.append(Question(text=f.question, answer=f.values[-1], kind="fact"))
    questions.append(Question(text="What is my dog's name?", answer="unknown",
                              kind="abstention", trap=trap_val))
    rng.shuffle(questions)
    return Stream(seed=seed, sessions=sessions, questions=questions, facts=facts)
