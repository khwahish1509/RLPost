"""Phase 3 — the training station.

Contract (from PLAN.md):
- TOML config mirroring prime-rl's schema: model, max_steps, batch_size,
  rollouts_per_example, [[env]] (see configs/);
- synchronous loop: env dataset → rollouts → rewards → GRPO step
  (TRL GRPOTrainer; Unsloth variant for Colab) → LoRA adapter;
- pre-flight trainability check, hard-coded: baseline reward must fall in
  the 10–80% window, else abort with a clear message;
- reward curve logged per step into train_runs;
- checkpoint every 10 steps; --resume restores model+optimizer+step.

Heavy deps (torch/trl/unsloth) are NOT project dependencies — training runs
in a GPU session (Colab/Kaggle) where they're installed separately.
"""

from __future__ import annotations


def train(*args, **kwargs):
    raise NotImplementedError("Phase 3 — see PLAN.md")
