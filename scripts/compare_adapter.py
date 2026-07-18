"""The exam: base model vs trained checkpoint(s) on held-out questions.

Usage (GPU box):
    uv run python scripts/compare_adapter.py                 # newest checkpoint
    uv run python scripts/compare_adapter.py adapters/run1/step00049 [more...]

Same questions (held-out eval split), same greedy decoding, same rubric —
whatever delta appears is the training, nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

N_QUESTIONS = 64
BATCH = 16
MODEL = "Qwen/Qwen3-0.6B"
ENV = "gsm8k"


def newest_checkpoint() -> str:
    steps = sorted(Path("adapters").glob("run*/step*"))
    if not steps:
        raise SystemExit("no checkpoints found under adapters/ — train first")
    return str(steps[-1])


def exam(model, tok, env, rows, tag):
    import torch

    from nanolab.train import score_completions

    model.eval()
    rewards: list[float] = []
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        prompts = [
            tok.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            seqs = model.generate(
                **enc, do_sample=False, max_new_tokens=256,
                pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(
            seqs[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        rewards += score_completions(env, batch, texts)
    score = sum(rewards) / len(rewards)
    print(f"{tag:<28} {score:.3f}  ({sum(rewards):.0f}/{len(rewards)} correct)")
    return score


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanolab.train import load_single_turn_env

    checkpoints = sys.argv[1:] or [newest_checkpoint()]

    env = load_single_turn_env(ENV)
    ds = env.get_eval_dataset()
    rows = [ds[i] for i in range(N_QUESTIONS)]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=device)
    base_score = exam(base, tok, env, rows, "base")

    wrapped = None
    for ckpt in checkpoints:
        if wrapped is None:
            wrapped = PeftModel.from_pretrained(base, ckpt, adapter_name="a")
        else:
            wrapped.load_adapter(ckpt, adapter_name="a" + ckpt[-5:])
            wrapped.set_adapter("a" + ckpt[-5:])
        score = exam(wrapped, tok, env, rows, Path(ckpt).name)
        print(f"{'  delta vs base':<28} {score - base_score:+.3f}\n")


if __name__ == "__main__":
    main()
