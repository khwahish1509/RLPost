"""In-process OpenAI-compatible endpoint for the training policy.

The bridge that lets the trainer handle multi-turn environments (the Scribe):
during rollout collection the current model is exposed at
http://127.0.0.1:<port>/v1/chat/completions, so verifiers' own rollout engine
— the exact machinery the eval station already trusts — can drive full
episodes against the policy being trained, Player calls and all.

Deliberately minimal: one POST route, one GET route, starlette + uvicorn in a
background thread (both already ship with verifiers). `generate_fn` receives
the conversation messages and returns the assistant's reply text; it is
serialized with a lock because the GPU generates one request at a time.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable

Messages = list[dict]


class PolicyServerError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PolicyServer:
    def __init__(self, generate_fn: Callable[[Messages], str], model_name: str = "policy"):
        self.generate_fn = generate_fn
        self.model_name = model_name
        self.port = _free_port()
        self._lock = threading.Lock()
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _build_app(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        def models(request):
            return JSONResponse({"data": [{"id": self.model_name, "object": "model"}]})

        async def chat(request):
            body = await request.json()
            messages = body.get("messages", [])
            import anyio

            def run():
                with self._lock:
                    return self.generate_fn(messages)

            text = await anyio.to_thread.run_sync(run)
            return JSONResponse(
                {
                    "id": "policy-completion",
                    "object": "chat.completion",
                    "model": body.get("model", self.model_name),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

        return Starlette(
            routes=[
                Route("/v1/models", models, methods=["GET"]),
                Route("/v1/chat/completions", chat, methods=["POST"]),
            ]
        )

    def start(self, ready_timeout: float = 15.0) -> None:
        import httpx
        import uvicorn

        config = uvicorn.Config(
            self._build_app(), host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.base_url}/models", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.05)
        raise PolicyServerError("policy server did not become ready")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def __enter__(self) -> "PolicyServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
