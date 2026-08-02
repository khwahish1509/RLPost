"""Cross-domain generator: WORKPLACE streams (the transfer test).

Same game, different world. The Scribe trained only on personal-life chat
(pets, cities, allergies — dialogues.py). These streams are standup-style
work talk: projects, deadlines, clients, budgets — different vocabulary,
different sentence shapes, different trap flavours. A model that learned the
*skill* (keep the speaker's facts, refuse other teams', replace what
changed) should transfer; a model that memorized templates should not.

Deliberately a standalone copy of the dialogues.py structure rather than a
refactor: the personal generator's seeded random call order must stay
byte-identical, or stored training/eval comparisons break.
"""

from __future__ import annotations

import random

from .dialogues import EVAL_SEED_BASE, Fact, Question, Stream  # noqa: F401

FIRST_NAMES = ["Asha", "Rohan", "Mia", "Kabir", "Lena", "Tara", "Dev", "Ira",
               "Noah", "Zara", "Vik", "Meera", "Sam", "Anya", "Ravi", "Nina"]
PROJECTS = ["Atlas", "Nimbus", "Falcon", "Beacon", "Quartz", "Ember", "Delta", "Onyx"]
CLIENTS = ["Northwind", "Acme", "Globex", "Initech", "Vertex", "Halcyon"]
MONTHS = ["January", "March", "April", "June", "August", "October", "December"]
TOOLS = ["Django", "FastAPI", "Rails", "Spring", "Laravel", "Phoenix"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ROOMS = ["4B", "7A", "2C", "9F", "3D", "6E"]
BUDGETS = ["120", "250", "80", "400", "175", "310"]

# fact key → (question, how it's said in a standup, value pool)
WORK_FACTS = {
    "project":  ("What is my project called?",
                 "kickoff done — our project is officially called {v}.", PROJECTS),
    "deadline": ("Which month is my deadline?",
                 "deadline landed: we ship in {v}.", MONTHS),
    "client":   ("Who is my client?",
                 "spent an hour on a call with {v}, our client.", CLIENTS),
    "teammate": ("Who joined my team?",
                 "good news, {v} joined my team this sprint.", FIRST_NAMES),
    "tool":     ("Which framework is my team using?",
                 "we settled the stack argument — going with {v}.", TOOLS),
    "sync":     ("Which day is my weekly sync?",
                 "weekly sync is on {v} now.", DAYS),
    "room":     ("Which room does my team sit in?",
                 "facilities moved us to room {v}.", ROOMS),
    "budget":   ("What is my project budget in thousands?",
                 "finance approved the budget: {v}k.", BUDGETS),
}
WORK_UPDATABLE = {"deadline", "tool", "sync", "budget"}

# traps: attributes of OTHER teams/people, never the speaker's
WORK_TRAPS = {
    "manager": ("What is my manager's name?",
                "{who} won't stop praising their manager {v}.", FIRST_NAMES),
    "vendor":  ("Which vendor does my team use?",
                "{who} is stuck in procurement talks with vendor {v}.", CLIENTS),
    "release": ("Which month is my team's off-site?",
                "{who} booked their off-site for {v}.", MONTHS),
}
WHO = ["the platform team", "the data team", "the mobile crew"]

CHATTER = [
    "CI was red the entire morning, of course.",
    "retro ran forty minutes over.",
    "someone broke the build right before lunch.",
    "expense reports are due friday, apparently.",
    "the coffee machine on our floor died again.",
]


def _work_trap_line(rng: random.Random, kind: str, val: str) -> str:
    return WORK_TRAPS[kind][1].format(who=rng.choice(WHO), v=val)


def generate_work_stream(
    seed: int,
    num_sessions: int = 4,
    facts_per_stream: int = 5,
    updates_per_stream: int = 1,
    distractors_per_session: int = 3,
    num_questions: int = 6,
    abstentions_per_stream: int = 1,
) -> Stream:
    rng = random.Random(f"work:{seed}")

    need_upd = min(updates_per_stream, len(WORK_UPDATABLE), facts_per_stream)
    upd_keys = rng.sample(sorted(WORK_UPDATABLE), need_upd)
    rest = [k for k in WORK_FACTS if k not in upd_keys]
    keys = upd_keys + rng.sample(rest, facts_per_stream - len(upd_keys))
    rng.shuffle(keys)

    facts: list[Fact] = []
    for k in keys:
        q_tmpl, _say, pool = WORK_FACTS[k]
        facts.append(Fact(key=k, kind="", question=q_tmpl, values=[rng.choice(pool)]))

    for f in [f for f in facts if f.key in WORK_UPDATABLE][:updates_per_stream]:
        pool = WORK_FACTS[f.key][2]
        f.values.append(rng.choice([v for v in pool if v != f.values[0]]))

    used = {v for f in facts for v in f.values}
    n_traps = max(1, min(abstentions_per_stream, len(WORK_TRAPS)))
    trap_kinds = rng.sample(sorted(WORK_TRAPS), n_traps)
    traps: list[tuple[str, str]] = []
    for tk in trap_kinds:
        pool = [v for v in WORK_TRAPS[tk][2] if v not in used]
        tv = rng.choice(pool)
        used.add(tv)
        traps.append((tk, tv))

    for f in facts:
        first = rng.randrange(0, max(1, num_sessions - 1))
        f.said_in = [first]
        if len(f.values) > 1:
            f.said_in.append(rng.randrange(first + 1, num_sessions))

    sessions: list[str] = []
    for s in range(num_sessions):
        lines: list[str] = []
        for f in facts:
            for vi, sess in enumerate(f.said_in):
                if sess != s:
                    continue
                _q, say_tmpl, _p = WORK_FACTS[f.key]
                line = say_tmpl.format(v=f.values[vi])
                if vi > 0:
                    line = f"change of plan — {line.rstrip('.')} (scrap {f.values[vi-1]})."
                lines.append(line)
        tk, tv = traps[s % len(traps)]
        lines.append(_work_trap_line(rng, tk, tv))
        for _ in range(max(0, distractors_per_session - 1)):
            lines.append(rng.choice(CHATTER))
        rng.shuffle(lines)
        sessions.append("\n".join(f"- {ln}" for ln in lines))

    questions: list[Question] = []
    for f in [f for f in facts if len(f.values) > 1]:
        questions.append(Question(text=f.question, answer=f.values[-1],
                                  kind="update", stale=f.values[0]))
    plain = [f for f in facts if len(f.values) == 1]
    rng.shuffle(plain)
    for f in plain:
        if len(questions) >= num_questions - len(traps):
            break
        questions.append(Question(text=f.question, answer=f.values[-1], kind="fact"))
    for tk, tv in traps:
        questions.append(Question(text=WORK_TRAPS[tk][0], answer="unknown",
                                  kind="abstention", trap=tv))
    rng.shuffle(questions)
    return Stream(seed=seed, sessions=sessions, questions=questions, facts=facts)
