"""The Memory Agent: a chat whose ONLY memory is a capped notebook.

The design mirrors the scribe-stream experiment exactly, so the room is the
experiment made interactive:

- The ASSISTANT is amnesiac by construction: each turn it sees only its
  system prompt, the current notebook, and the newest user message. No
  transcript. If the notebook doesn't carry a fact, the assistant has
  genuinely lost it.
- A NO-MEMORY CONTROL answers every turn alongside it (same model, empty
  notebook), so the notebook's contribution is visible per turn instead of
  asserted.
- After each exchange the WRITER rewrites the notebook (full replacement,
  hard 400-char cap enforced in code — the same binding budget the Scribe
  was trained under):
    trained  — the S2 checkpoint served locally (lazily spawned)
    prompted — the same instructions to the lab's default API model
    none     — the notebook never changes (the ablation)

The rewrite runs in a background thread; the UI polls the session and picks
up the new notebook when it lands. State lives in the agent_sessions table —
memory that survives a restart is the whole point.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import db

NOTEBOOK_CAP = 400
SCRIBE_ADAPTER = Path("adapters") / "scribe_s2" / "step00029"
SCRIBE_BASE = "Qwen/Qwen3-0.6B"

ASSISTANT_SYSTEM = (
    "You are a helpful assistant with severe amnesia: you remember NOTHING "
    "about this conversation except the NOTEBOOK below, which your scribe "
    "maintains. Trust it as your only memory. If the user refers to something "
    "that is not in the notebook and not in their current message, admit you "
    "don't have it. Keep replies short and concrete.\n\nNOTEBOOK:\n{notebook}"
)

WRITER_SYSTEM = (
    "You are the Scribe. A separate assistant with no memory relies on a "
    "notebook you maintain — it is the ONLY thing it carries between turns. "
    "You will see the current notebook and the newest exchange. Reply with "
    "ONLY the full new contents of the notebook as short 'name = value' "
    f"lines (it fully replaces the old one; anything beyond {NOTEBOOK_CAP} "
    "characters is cut off). Keep facts that will matter later; drop "
    "chit-chat."
)


class AgentError(RuntimeError):
    pass


# ---------------------------------------------------------------- sessions

def _now() -> str:
    return db.utcnow()


def current_session() -> dict:
    """The newest session, created on demand."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM agent_sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO agent_sessions (created_at, updated_at) VALUES (?, ?)",
                (_now(), _now()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        out = dict(row)
    finally:
        conn.close()
    out["messages"] = json.loads(out.pop("messages_json") or "[]")
    out["scribe_ready"] = _scribe_ready()
    return out


def reset_session() -> dict:
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO agent_sessions (created_at, updated_at) VALUES (?, ?)",
            (_now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return current_session()


def set_writer(writer: str) -> dict:
    if writer not in ("trained", "prompted", "none"):
        raise AgentError("writer must be trained, prompted, or none")
    session = current_session()
    _update(session["id"], writer=writer)
    return current_session()


def _update(session_id: int, **fields) -> None:
    fields["updated_at"] = _now()
    keys = ", ".join(f"{k} = ?" for k in fields)
    conn = db.connect()
    try:
        conn.execute(
            f"UPDATE agent_sessions SET {keys} WHERE id = ?",
            (*fields.values(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- endpoints

def _default_endpoint() -> tuple[str, str, str]:
    """(base_url, api_key, model) for the lab's default API endpoint."""
    from . import config as config_mod

    config_mod.load_dotenv()
    base = os.environ.get("NANOLAB_API_BASE_URL", "")
    key_var = os.environ.get("NANOLAB_API_KEY_VAR", "")
    model = os.environ.get("NANOLAB_DEFAULT_MODEL", "")
    key = os.environ.get(key_var, "") if key_var else ""
    if not (base and key and model):
        raise AgentError(
            "the agent needs the default endpoint configured — set "
            "NANOLAB_API_BASE_URL, NANOLAB_API_KEY_VAR and NANOLAB_DEFAULT_MODEL "
            "in .env"
        )
    return base, key, model


def _chat(base_url: str, api_key: str, model: str, messages: list[dict],
          max_tokens: int = 400, temperature: float = 0.3,
          timeout: float = 120.0) -> str:
    import httpx

    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


# ------------------------------------------------- the local scribe server

_SCRIBE = {"port": None, "proc": None, "starting": False}
_SCRIBE_LOCK = threading.Lock()


def _scribe_ready() -> bool:
    port = _SCRIBE["port"]
    if not port:
        return False
    try:
        import httpx

        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2).status_code == 200
    except Exception:
        return False


def _ensure_scribe_server() -> str:
    """Start the trained-scribe server if needed; return its base_url.

    Loading takes ~15s on Apple silicon; callers run in background threads so
    blocking here is fine.
    """
    with _SCRIBE_LOCK:
        if _scribe_ready():
            return f"http://127.0.0.1:{_SCRIBE['port']}/v1"
        if not SCRIBE_ADAPTER.exists():
            raise AgentError(
                f"trained scribe adapter not found at {SCRIBE_ADAPTER} — "
                "train one (configs/qwen3-0.6b-scribe.toml) or switch the "
                "writer to 'prompted'"
            )
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ)
        env.setdefault("NANOLAB_LOCAL_API_KEY", "local")
        proc = subprocess.Popen(
            [sys.executable, "-m", "nanolab.serve_local",
             "--base", SCRIBE_BASE, "--adapter", str(SCRIBE_ADAPTER),
             "--served-name", "scribe-s2", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        _SCRIBE.update(port=port, proc=proc)
    for _ in range(60):  # up to ~3 min for a cold model download
        if _scribe_ready():
            return f"http://127.0.0.1:{_SCRIBE['port']}/v1"
        if proc.poll() is not None:
            _SCRIBE.update(port=None, proc=None)
            raise AgentError("the scribe server exited during startup")
        time.sleep(3)
    raise AgentError("the scribe server did not become ready in time")


def stop_scribe_server() -> None:
    proc = _SCRIBE.get("proc")
    if proc and proc.poll() is None:
        proc.kill()
    _SCRIBE.update(port=None, proc=None)


# ---------------------------------------------------------------- the turn

def chat_turn(user_message: str) -> dict:
    """One exchange: amnesiac reply + no-memory control, then a background
    notebook rewrite. Returns the session as of the reply (rewrite pending)."""
    user_message = (user_message or "").strip()
    if not user_message:
        raise AgentError("say something")
    session = current_session()
    base, key, model = _default_endpoint()

    def ask(notebook: str) -> str:
        return _chat(base, key, model, [
            {"role": "system", "content": ASSISTANT_SYSTEM.format(notebook=notebook.strip() or "(empty)")},
            {"role": "user", "content": user_message},
        ])

    reply = ask(session["notebook"])
    try:
        control = ask("")  # the no-memory control: same model, empty notebook
    except Exception:
        control = ""       # the control is illustration, never a blocker

    messages = session["messages"] + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply, "control": control,
         "writer": session["writer"], "at": _now()},
    ]
    turns = session["turns"] + 1
    _update(session["id"], messages_json=json.dumps(messages), turns=turns,
            scribe_state="rewriting" if session["writer"] != "none" else "idle")

    if session["writer"] != "none":
        threading.Thread(
            target=_rewrite_notebook,
            args=(session["id"], session["writer"], session["notebook"],
                  user_message, reply, turns),
            daemon=True,
        ).start()
    return current_session()


def _rewrite_notebook(session_id: int, writer: str, notebook: str,
                      user_message: str, reply: str, turn: int) -> None:
    # cold-start subtlety, learned live: the trained scribe carries notebook
    # CONTENT forward, so a "(empty)" placeholder gets copied as if it were a
    # fact. A blank section is in-distribution; a placeholder is not.
    prompt = [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content":
            f"CURRENT NOTEBOOK:\n{notebook.strip()}\n\n"
            f"NEW EXCHANGE (turn {turn}):\n"
            f"user: {user_message}\nassistant: {reply}\n\n"
            "Write the full new notebook now."},
    ]
    try:
        if writer == "trained":
            url = _ensure_scribe_server()
            text = _chat(url, os.environ.get("NANOLAB_LOCAL_API_KEY", "local"),
                         "scribe-s2", prompt, max_tokens=512,
                         temperature=0.0, timeout=300.0)
        else:  # prompted
            base, key, model = _default_endpoint()
            text = _chat(base, key, model, prompt, temperature=0.0)
        new_notebook = text.strip()[:NOTEBOOK_CAP]  # the cap is code, not trust
        _update(session_id, notebook=new_notebook, scribe_state="idle")
    except Exception as exc:  # surface the failure in the room, don't hide it
        _update(session_id, scribe_state=f"error: {str(exc)[:160]}")
