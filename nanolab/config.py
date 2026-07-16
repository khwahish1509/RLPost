"""Tiny .env loader — no dependency, never overrides an already-set variable.

Recognized variables:
  NANOLAB_DB            path to the lab SQLite file (default results/nanolab.db)
  NANOLAB_API_BASE_URL  default OpenAI-compatible endpoint for evals
  NANOLAB_API_KEY_VAR   name of the env var holding the API key (not the key!)
  NANOLAB_DEFAULT_MODEL default model for `nanolab eval run`
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)
