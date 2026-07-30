"""memory-stream: conversational memory as a verifiers MultiTurnEnv.

One episode = one user's multi-session dialogue. After each session the model
under test — the Scribe — rewrites a hard-capped notebook (its ONLY output).
When the sessions end, a frozen reader answers questions about earlier
sessions using only that notebook.

Reward (Lift) = reader accuracy with the notebook − accuracy with an empty
one. The question mix makes lazy strategies lose: updates punish append-only
notes, abstention traps punish sloppy attribution, and the cap punishes
copying everything.

Anti-cheat trio (inherited from scribe-stream): the cap · held-out eval
seeds · a frozen reader.
"""

from __future__ import annotations

import os

import verifiers as vf

from .dialogues import EVAL_SEED_BASE, generate_stream
from .reader import build_reader

NOTEBOOK_CHAR_CAP = 600

SCRIBE_SYSTEM = (
    "You are a memory assistant. A user chats with you across sessions. "
    "Later, a separate reader will answer questions about the user using ONLY "
    "your notebook — it never sees the conversation. After each session, "
    "reply with ONLY the full new notebook contents (it fully replaces the "
    "old one; anything beyond ~{cap} characters is cut off). Keep facts about "
    "THE USER that will matter later; when a fact changes, replace the old "
    "value; drop chit-chat and other people's details."
)


def _message_text(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


class MemoryStreamEnv(vf.MultiTurnEnv):
    def __init__(self, reader, notebook_char_cap: int = NOTEBOOK_CHAR_CAP,
                 num_sessions: int = 4, **kwargs):
        super().__init__(max_turns=num_sessions + 2, **kwargs)
        self.reader = reader
        self.notebook_char_cap = notebook_char_cap

    async def setup_state(self, state):
        info = state.get("info", {})
        state["mem_sessions"] = info["sessions"]
        state["mem_questions"] = info["questions"]
        state["next_session"] = 1  # session 0 is in the prompt
        state["notebook"] = ""
        return state

    async def env_response(self, messages, state, **kwargs):
        notebook = _message_text(messages[-1])[: self.notebook_char_cap]
        state["notebook"] = notebook

        sessions = state["mem_sessions"]
        i = state["next_session"]
        state["next_session"] = i + 1
        if i < len(sessions):
            body = (f"Session {i + 1} of {len(sessions)} — the user said:\n"
                    f"{sessions[i]}\n\nWrite the full new notebook now.")
            if i + 1 >= len(sessions):
                body += "\nThis is the last session."
            return [{"role": "user", "content": body}]
        # the scribe has now written once after every session — the notebook
        # captured above is the final one the reader will use
        message = {"role": "user", "content": "Stream complete."}
        state["final_env_response"] = [message]
        return [message]


def _build_dataset(num_streams: int, seed_base: int, num_sessions: int,
                   facts_per_stream: int, updates_per_stream: int,
                   distractors_per_session: int, num_questions: int, cap: int,
                   abstentions_per_stream: int = 1):
    from datasets import Dataset

    rows = []
    for seed in range(seed_base, seed_base + num_streams):
        stream = generate_stream(
            seed, num_sessions=num_sessions, facts_per_stream=facts_per_stream,
            updates_per_stream=updates_per_stream,
            distractors_per_session=distractors_per_session,
            num_questions=num_questions,
            abstentions_per_stream=abstentions_per_stream,
        )
        rows.append({
            "prompt": [
                {"role": "system", "content": SCRIBE_SYSTEM.format(cap=cap)},
                {"role": "user", "content": (
                    f"Session 1 of {num_sessions} — the user said:\n"
                    f"{stream.sessions[0]}\n\nWrite the full notebook now.")},
            ],
            "answer": "",
            "info": stream.to_info(),
        })
    return Dataset.from_list(rows)


def load_environment(
    reader_model: str | None = None,
    reader_base_url: str | None = None,
    reader_key_var: str | None = None,
    num_train_streams: int = 200,
    num_eval_streams: int = 50,
    notebook_char_cap: int = NOTEBOOK_CHAR_CAP,
    num_sessions: int = 4,
    facts_per_stream: int = 5,
    updates_per_stream: int = 1,
    distractors_per_session: int = 3,
    num_questions: int = 6,
    abstentions_per_stream: int = 1,
    question_weights: dict | None = None,
):
    reader_model = reader_model or os.environ.get("NANOLAB_READER_MODEL", "fake")
    reader_base_url = reader_base_url or os.environ.get("NANOLAB_API_BASE_URL", "")
    key_var = reader_key_var or os.environ.get("NANOLAB_API_KEY_VAR", "")
    api_key = os.environ.get(key_var, "") if key_var else ""
    reader = build_reader(reader_model, reader_base_url, api_key)

    # question-type weights let a curriculum make selection failures
    # (kept traps, stale values) expensive relative to easy fact recall
    weights = {"fact": 1.0, "update": 1.0, "abstention": 1.0}
    if question_weights:
        weights.update({k: float(v) for k, v in question_weights.items()})

    async def lift(state) -> float:
        """Weighted reader accuracy with the notebook minus with an empty one."""
        questions = state.get("mem_questions", [])
        if not questions:
            return 0.0
        notebook = state.get("notebook", "")
        num_w = num_b = denom = 0.0
        for q in questions:
            w = weights.get(q.get("kind", "fact"), 1.0)
            denom += w
            r = await reader.read(q, notebook)
            num_w += w * bool(r.correct)
            b = await reader.read(q, "")
            num_b += w * bool(b.correct)
        acc = num_w / denom
        base = num_b / denom
        state["with_notes_score"] = acc
        state["baseline_score"] = base
        return acc - base

    def reader_score(state) -> float:
        return float(state.get("with_notes_score", 0.0))

    def baseline_score(state) -> float:
        return float(state.get("baseline_score", 0.0))

    rubric = vf.Rubric(funcs=[lift, reader_score, baseline_score],
                       weights=[1.0, 0.0, 0.0])

    return MemoryStreamEnv(
        reader=reader,
        notebook_char_cap=notebook_char_cap,
        num_sessions=num_sessions,
        dataset=lambda: _build_dataset(
            num_train_streams, 0, num_sessions, facts_per_stream,
            updates_per_stream, distractors_per_session, num_questions,
            notebook_char_cap, abstentions_per_stream),
        eval_dataset=lambda: _build_dataset(
            num_eval_streams, EVAL_SEED_BASE, num_sessions, facts_per_stream,
            updates_per_stream, distractors_per_session, num_questions,
            notebook_char_cap, abstentions_per_stream),
        rubric=rubric,
    )
