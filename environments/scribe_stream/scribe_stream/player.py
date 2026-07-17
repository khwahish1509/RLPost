"""The frozen Player: attempts each task statelessly, with only the notebook.

Every real call is disk-cached (streams re-run constantly during Scribe
development) and made at temperature 0 so the cache is honest. `model="fake"`
gives a deterministic offline Player: it answers correctly iff every foreign
figure the task needs appears in the notebook — the mechanics of Lift without
an API.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(".cache") / "player"

PLAYER_SYSTEM = (
    "You solve one small arithmetic task. You may be given a NOTEBOOK with "
    "facts recorded from earlier tasks; trust it. If the task refers to a "
    "figure you cannot find in the notebook or the task text, make your best "
    "guess. Always end with the final integer answer in \\boxed{}."
)


@dataclass
class PlayResult:
    correct: bool
    parsed: int | None
    text: str


def parse_int(text: str) -> int | None:
    boxed = re.findall(r"\\boxed\{(-?\d+)", text.replace(",", ""))
    if boxed:
        return int(boxed[-1])
    plain = re.findall(r"-?\d+", text.replace(",", ""))
    return int(plain[-1]) if plain else None


class FakePlayer:
    """Correct iff the notebook contains every foreign figure the task needs."""

    async def play(self, task: dict, notebook: str) -> PlayResult:
        needed = task.get("foreign_values", [])
        knows = all(str(v) in notebook for v in needed)
        if knows:
            return PlayResult(True, task["answer"], "(fake: notes sufficient)")
        return PlayResult(False, -1, "(fake: missing figures, guessed)")


class ApiPlayer:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _messages(self, task: dict, notebook: str) -> list[dict]:
        nb = notebook.strip() or "(empty)"
        return [
            {"role": "system", "content": PLAYER_SYSTEM},
            {
                "role": "user",
                "content": f"NOTEBOOK:\n{nb}\n\nTASK:\n{task['text']}",
            },
        ]

    def _cache_path(self, messages: list[dict]) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "base_url": self.base_url, "messages": messages,
                 "temperature": self.temperature},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return CACHE_DIR / f"{key}.json"

    async def play(self, task: dict, notebook: str) -> PlayResult:
        import httpx

        messages = self._messages(task, notebook)
        cache_file = self._cache_path(messages)
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"] or ""
            data = {"text": text}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        parsed = parse_int(data["text"])
        return PlayResult(parsed == task["answer"], parsed, data["text"])


def build_player(model: str, base_url: str = "", api_key: str = ""):
    if model == "fake":
        return FakePlayer()
    if not base_url or not api_key:
        raise ValueError(
            "scribe-stream: a real player needs player_base_url and an API key "
            "(or use player_model='fake' for offline mechanics)"
        )
    return ApiPlayer(model=model, base_url=base_url, api_key=api_key)
