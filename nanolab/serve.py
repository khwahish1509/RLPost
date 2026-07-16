"""Phase 5 — the inference station.

Contract (from PLAN.md):
- launch vLLM with --enable-lora and runtime adapter loading;
- register served adapters (deployments create/list);
- resolve `base:adapter` model strings for evaluate.py;
- documented alternative path: merge LoRA → GGUF → llama.cpp on a laptop.
"""

from __future__ import annotations


def create_deployment(*args, **kwargs):
    raise NotImplementedError("Phase 5 — see PLAN.md")
