"""scribe-stream: a verifiers MultiTurnEnv where the model under test is a
note-taking Scribe and the reward is Lift.

One episode = one stream of 8 dependent tasks. A frozen Player attempts each
task statelessly — its only memory is the Scribe's notebook. After each task,
the outcome and the task's revealed figure are shown to the Scribe, whose
ONLY output is the next version of the notebook (hard-capped in code).

Reward (Lift) = Player's mean correctness on tasks 2..8 with the notebook,
minus the same Player's mean correctness with an empty notebook (cached).

Anti-cheat trio:
- notebook cap kills log-dumping;
- eval streams come from a disjoint seed range (held out by construction);
- the Player is frozen — the only thing training can improve is the notes.
"""

from __future__ import annotations

import os

import verifiers as vf

from .player import build_player
from .streams import EVAL_SEED_BASE, N_TASKS, generate_stream

# ~1,500 tokens at ~4 chars/token — enforced by truncation, not by trust
NOTEBOOK_CHAR_CAP = 6000

SCRIBE_SYSTEM = (
    "You are the Scribe. A separate, frozen Player solves small arithmetic "
    "tasks one at a time, with NO memory between tasks — the ONLY thing it "
    "carries forward is your notebook, exactly as you write it. After each "
    "task you will see the Player's result and a RECORD line revealing that "
    "task's figure. Future tasks will require earlier figures. Reply with "
    "ONLY the full new contents of the notebook (it fully replaces the old "
    f"one; anything beyond ~{NOTEBOOK_CHAR_CAP} characters is cut off)."
)


def _message_text(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


class ScribeStreamEnv(vf.MultiTurnEnv):
    def __init__(self, player, notebook_char_cap: int = NOTEBOOK_CHAR_CAP, **kwargs):
        super().__init__(max_turns=N_TASKS + 2, **kwargs)
        self.player = player
        self.notebook_char_cap = notebook_char_cap

    async def setup_state(self, state):
        info = state.get("info", {})
        state["stream_tasks"] = info["tasks"]
        state["next_task"] = 1  # task 0 is presented in the prompt
        state["notebook"] = ""
        state["player_results"] = []
        return state

    async def env_response(self, messages, state, **kwargs):
        notebook = _message_text(messages[-1])[: self.notebook_char_cap]
        state["notebook"] = notebook

        tasks = state["stream_tasks"]
        i = state["next_task"]
        task = tasks[i]
        result = await self.player.play(task, notebook)
        state["player_results"].append(
            {"task": i, "correct": bool(result.correct), "parsed": result.parsed}
        )
        state["next_task"] = i + 1

        verdict = (
            "CORRECT"
            if result.correct
            else f"WRONG (correct answer: {task['answer']})"
        )
        body = (
            f"Player attempted task {i + 1}/{len(tasks)}:\n"
            f"  task: {task['text']}\n"
            f"  Player answered: {result.parsed} — {verdict}\n\n"
            f"RECORD — {task['reveal']}\n\n"
        )
        if state["next_task"] >= len(tasks):
            message = {"role": "user", "content": body + "Stream complete."}
            state["final_env_response"] = [message]
            return [message]
        message = {
            "role": "user",
            "content": body + "Write the full new notebook now.",
        }
        return [message]


def _build_dataset(num_streams: int, seed_base: int):
    from datasets import Dataset

    rows = []
    for seed in range(seed_base, seed_base + num_streams):
        stream = generate_stream(seed)
        task0 = stream.tasks[0]
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SCRIBE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task 1/{N_TASKS} was:\n  {task0.text}\n\n"
                            f"RECORD — {task0.reveal}\n\n"
                            "Write the full notebook now."
                        ),
                    },
                ],
                "answer": "",
                "info": stream.to_info(),
            }
        )
    return Dataset.from_list(rows)


def load_environment(
    player_model: str | None = None,
    player_base_url: str | None = None,
    player_key_var: str | None = None,
    num_train_streams: int = 200,
    num_eval_streams: int = 50,
    notebook_char_cap: int = NOTEBOOK_CHAR_CAP,
):
    player_model = player_model or os.environ.get("NANOLAB_PLAYER_MODEL", "fake")
    player_base_url = player_base_url or os.environ.get("NANOLAB_API_BASE_URL", "")
    key_var = player_key_var or os.environ.get("NANOLAB_API_KEY_VAR", "")
    api_key = os.environ.get(key_var, "") if key_var else ""
    player = build_player(player_model, player_base_url, api_key)

    async def lift(state) -> float:
        """Mean Player correctness on tasks 2..8 with notes minus without."""
        with_notes = [r["correct"] for r in state.get("player_results", [])]
        if not with_notes:
            return 0.0
        tasks = state["stream_tasks"]
        baseline = []
        for i in range(1, len(tasks)):
            result = await player.play(tasks[i], "")
            baseline.append(bool(result.correct))
        state["baseline_results"] = baseline
        score = sum(with_notes) / len(with_notes)
        base = sum(baseline) / len(baseline)
        state["with_notes_score"] = score
        state["baseline_score"] = base
        return score - base

    def player_score(state) -> float:
        return float(state.get("with_notes_score", 0.0))

    def baseline_score(state) -> float:
        return float(state.get("baseline_score", 0.0))

    rubric = vf.Rubric(
        funcs=[lift, player_score, baseline_score], weights=[1.0, 0.0, 0.0]
    )

    return ScribeStreamEnv(
        player=player,
        notebook_char_cap=notebook_char_cap,
        dataset=lambda: _build_dataset(num_train_streams, 0),
        eval_dataset=lambda: _build_dataset(num_eval_streams, EVAL_SEED_BASE),
        rubric=rubric,
    )
