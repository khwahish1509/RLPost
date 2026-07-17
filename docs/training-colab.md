# Training on Colab (T4 is enough for Qwen3-0.6B)

The trainer is a synchronous GRPO+LoRA loop (`nanolab/train.py`). Sessions
die; the loop assumes it — checkpoints carry adapter + optimizer + step, and
batches are derived from `(seed, step)`, so `--resume` continues exactly
where the run stopped.

## Cells

```bash
# 1. get the code (private repo — use a GitHub token, or upload a zip)
!git clone https://<TOKEN>@github.com/khwahish1509/RLPost.git nanolab
%cd nanolab

# 2. deps: project + GPU stack
!pip install -q uv && uv sync
!uv add torch transformers peft

# 3. the environment being trained on
!uv tool install prime && uv run nanolab env install primeintellect/gsm8k

# 4. train (re-run this same cell with --resume after any disconnect)
!uv run nanolab train configs/qwen3-0.6b-gsm8k.toml
# ... after a disconnect:
!uv run nanolab train configs/qwen3-0.6b-gsm8k.toml --resume
```

## What you should see

- A pre-flight line: `pre-flight baseline reward: 0.xxx` — the run aborts
  unless it lands inside the 10–80% trainability window.
- One line per step: `step N reward 0.xxx loss y.yyyy` — the reward curve is
  also stored in the db (`train_runs.reward_curve_json`).
- `adapters/run<id>/step<k>/` directories every 10 steps, each registered in
  the `adapters` table.

## Done-when for Phase 3

A 50-step run completes (surviving at least one restart via `--resume`), the
stored curve is non-flat, and an adapter directory exists. Then Phase 4:
evaluate the adapter against its own base model with `nanolab eval`.

## Bringing results home

Download `results/nanolab.db` and `adapters/` from Colab and drop them into
the local repo — the db is the lab; `nanolab eval list` / `report` read it.
