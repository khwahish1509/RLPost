"""The frozen reader: answers questions using ONLY the notebook.

FakeReader (offline, $0, deterministic) mirrors how a literal-minded reader
would use notes — and is deliberately strict so the reward teaches precision:

  - fact:       correct iff the answer value appears in the notebook
  - update:     correct iff the NEW value appears and the STALE one does not
                (append-only notebooks fail; replacement is the skill)
  - abstention: the trap value is someone else's attribute; correct iff the
                trap does NOT appear in the notebook (sloppy attribution fails)

ApiReader asks a real model (local Qwen or hosted), disk-cached at T=0 so
training and eval stay reproducible and cheap. Grading is string-match on the
expected answer — no LLM judge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(".cache") / "reader"

READER_SYSTEM = (
    "You answer ONE question about a user, using ONLY the NOTES provided. "
    "The notes were written by an assistant during earlier conversations. "
    "Answer with the single word or name only. If the notes do not contain "
    "the answer, reply exactly: unknown"
)


@dataclass
class ReadResult:
    correct: bool
    answered: str


def _contains(notebook: str, value: str) -> bool:
    return value.lower() in notebook.lower()


class FakeReader:
    async def read(self, question: dict, notebook: str) -> ReadResult:
        kind = question.get("kind", "fact")
        answer = question["answer"]
        if kind == "abstention":
            trap = question.get("trap", "")
            if trap and _contains(notebook, trap):
                return ReadResult(False, trap)          # mis-attributed junk
            return ReadResult(True, "unknown")
        if kind == "update":
            stale = question.get("stale", "")
            ok = _contains(notebook, answer) and not (stale and _contains(notebook, stale))
            return ReadResult(ok, answer if ok else (stale or "unknown"))
        ok = _contains(notebook, answer)
        return ReadResult(ok, answer if ok else "unknown")


class ApiReader:
    def __init__(self, model: str, base_url: str, api_key: str,
                 max_tokens: int = 40, timeout: float = 300.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _messages(self, question: dict, notebook: str) -> list[dict]:
        nb = notebook.strip() or "(empty)"
        return [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": f"NOTES:\n{nb}\n\nQUESTION: {question['text']}"},
        ]

    def _cache_path(self, messages: list[dict]) -> Path:
        key = hashlib.sha256(json.dumps(
            {"model": self.model, "base_url": self.base_url, "messages": messages},
            sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / f"{key}.json"

    async def read(self, question: dict, notebook: str) -> ReadResult:
        import httpx

        messages = self._messages(question, notebook)
        cache_file = self._cache_path(messages)
        if cache_file.exists():
            text = json.loads(cache_file.read_text())["text"]
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "temperature": 0.0, "max_tokens": self.max_tokens},
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"] or ""
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"text": text}))
        answered = text.strip().splitlines()[0].strip() if text.strip() else "unknown"
        expected = question["answer"].lower()
        got = answered.lower()
        correct = expected in got if expected != "unknown" else ("unknown" in got or "know" in got)
        return ReadResult(correct, answered)


def build_reader(model: str, base_url: str = "", api_key: str = ""):
    if model == "fake":
        return FakeReader()
    if not base_url or not api_key:
        raise ValueError(
            "memory-stream: a real reader needs reader_base_url and an API key "
            "(or use reader_model='fake' for offline mechanics)"
        )
    return ApiReader(model=model, base_url=base_url, api_key=api_key)
