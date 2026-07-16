"""The inference station: serve trained adapters, resolve base:adapter strings.

`nanolab deployments create <adapter-id>` launches vLLM with `--enable-lora`,
serving the adapter as an OpenAI-compatible endpoint; the deployment is
registered in the db and `nanolab eval run -m <base>:<adapter-id>` resolves to
it automatically — the trained adapter measured through our own endpoint,
inside our own eval station. That's the loop closing.

vLLM needs a CUDA box (it's not a project dependency). Laptop alternative,
documented in docs/serving.md: merge the LoRA into the base model, convert to
GGUF, and serve with llama.cpp's llama-server — the resulting endpoint is
OpenAI-compatible too, so `--api-base-url` pointing at it works the same.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import db

# vLLM serves without auth by default; verifiers' client still wants a key
# env var, so we point it at a dummy that we guarantee exists.
LOCAL_KEY_VAR = "NANOLAB_LOCAL_API_KEY"


class ServeError(RuntimeError):
    pass


@dataclass
class Deployment:
    id: int
    adapter_id: int | None
    base_model: str
    served_name: str
    endpoint: str
    pid: int | None
    status: str


def _row_to_deployment(row) -> Deployment:
    return Deployment(
        id=row["id"],
        adapter_id=row["adapter_id"],
        base_model=row["base_model"],
        served_name=row["served_name"],
        endpoint=row["endpoint"],
        pid=row["pid"],
        status=row["status"],
    )


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def parse_model_string(model: str) -> tuple[str, str] | None:
    """`base:adapter-ref` → (base, adapter_ref); plain model names → None.

    HF model ids never contain ':', so a colon unambiguously marks an
    adapter reference (an adapter row id, e.g. "Qwen/Qwen3-0.6B:3").
    """
    if ":" not in model:
        return None
    base, _, ref = model.rpartition(":")
    if not base or not ref:
        raise ServeError(f"Malformed base:adapter model string: {model!r}")
    return base, ref


def resolve_model(model: str) -> tuple[str, str, str] | None:
    """Resolve a base:adapter string to (endpoint, served_name, api_key_var).

    Returns None for plain model names. Raises if the adapter exists but no
    live deployment serves it.
    """
    parsed = parse_model_string(model)
    if parsed is None:
        return None
    base, ref = parsed
    conn = db.connect()
    try:
        adapter = conn.execute(
            "SELECT * FROM adapters WHERE id = ?", (ref,)
        ).fetchone()
        if adapter is None:
            raise ServeError(
                f"No adapter with id {ref!r} — see the adapters table "
                "(training registers checkpoints there)."
            )
        if adapter["base_model"] != base:
            raise ServeError(
                f"Adapter {ref} was trained on {adapter['base_model']!r}, "
                f"not {base!r} — model string must be "
                f"{adapter['base_model']}:{ref}"
            )
        dep_row = conn.execute(
            "SELECT * FROM deployments WHERE adapter_id = ? AND status = 'running' "
            "ORDER BY id DESC LIMIT 1",
            (adapter["id"],),
        ).fetchone()
        if dep_row is None or not _pid_alive(dep_row["pid"]):
            raise ServeError(
                f"No live deployment serves adapter {ref}. "
                f"Start one with: nanolab deployments create {ref}"
            )
        os.environ.setdefault(LOCAL_KEY_VAR, "local")
        return dep_row["endpoint"], dep_row["served_name"], LOCAL_KEY_VAR
    finally:
        conn.close()


def create_deployment(
    adapter_id: int,
    port: int = 8000,
    ready_timeout: float = 900.0,
) -> Deployment:
    """Launch vLLM serving an adapter's base model with the LoRA attached."""
    if shutil.which("vllm") is None:
        raise ServeError(
            "vLLM is not installed (it needs a CUDA box): pip install vllm\n"
            "Laptop path: docs/serving.md (merge LoRA → GGUF → llama.cpp)."
        )
    conn = db.connect()
    try:
        adapter = conn.execute(
            "SELECT * FROM adapters WHERE id = ?", (adapter_id,)
        ).fetchone()
        if adapter is None:
            raise ServeError(f"No adapter with id {adapter_id}")
        adapter_path = Path(adapter["path"])
        if not adapter_path.exists():
            raise ServeError(f"Adapter files missing: {adapter_path}")

        served_name = f"adapter-{adapter_id}"
        log_dir = Path("results") / "deployments"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"deployment-a{adapter_id}-p{port}.log"
        cmd = [
            "vllm", "serve", adapter["base_model"],
            "--enable-lora",
            "--lora-modules", f"{served_name}={adapter_path}",
            "--port", str(port),
        ]
        with open(log_file, "ab") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        endpoint = f"http://localhost:{port}/v1"
        _wait_ready(endpoint, proc, ready_timeout, log_file)

        cur = conn.execute(
            "INSERT INTO deployments (adapter_id, base_model, served_name, "
            "endpoint, pid, status, created_at) VALUES (?, ?, ?, ?, ?, 'running', ?)",
            (adapter_id, adapter["base_model"], served_name, endpoint, proc.pid, db.utcnow()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM deployments WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _row_to_deployment(row)
    finally:
        conn.close()


def _wait_ready(endpoint: str, proc, timeout: float, log_file: Path) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServeError(
                f"vLLM exited with code {proc.returncode} — see {log_file}"
            )
        try:
            if httpx.get(f"{endpoint}/models", timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3.0)
    proc.terminate()
    raise ServeError(f"vLLM not ready after {timeout:.0f}s — see {log_file}")


def list_deployments() -> list[Deployment]:
    """All deployments, with liveness re-checked (dead pids get marked)."""
    conn = db.connect()
    try:
        rows = conn.execute("SELECT * FROM deployments ORDER BY id DESC").fetchall()
        for row in rows:
            if row["status"] == "running" and not _pid_alive(row["pid"]):
                conn.execute(
                    "UPDATE deployments SET status = 'dead' WHERE id = ?", (row["id"],)
                )
        conn.commit()
        return [
            _row_to_deployment(r)
            for r in conn.execute("SELECT * FROM deployments ORDER BY id DESC")
        ]
    finally:
        conn.close()


def stop_deployment(deployment_id: int) -> Deployment:
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
        ).fetchone()
        if row is None:
            raise ServeError(f"No deployment with id {deployment_id}")
        if _pid_alive(row["pid"]):
            os.killpg(os.getpgid(row["pid"]), signal.SIGTERM)
        conn.execute(
            "UPDATE deployments SET status = 'stopped' WHERE id = ?", (deployment_id,)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
        ).fetchone()
        return _row_to_deployment(row)
    finally:
        conn.close()
