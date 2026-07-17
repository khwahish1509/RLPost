"""The training station: TOML config → GRPO + LoRA synchronous loop → adapter.

One for-loop, no orchestrator. Each step:
  sample batch → generate rollouts_per_example completions per example →
  score with the env's own rubric (offline, no API) → group-normalized
  advantages → policy-gradient step on completion tokens → log curve →
  checkpoint every `checkpoint_every` steps.

The GRPO objective here is the on-policy special case: one gradient step per
batch means the importance ratio is exactly 1 and clipping is inert, so the
loss reduces to advantage-weighted NLL over completion tokens with per-group
(mean/std) advantage normalization — the same regime as the well-known small
-scale GRPO notebook recipes. No KL term in v0.1.

Scope, v0.1: single-turn environments only (the completion is the whole
trajectory, so the rubric can score it directly). Multi-turn training needs
served-endpoint rollouts and arrives with the inference station.

Heavy deps (torch, transformers, peft) are NOT project dependencies — they
load lazily inside train(). On the GPU box (Colab T4 works for Qwen3-0.6B):
  pip install torch transformers peft
Resume is a first-class citizen: checkpoints carry adapter + optimizer +
step + RNG, and batch sampling is derived from (seed, step), so a resumed
run draws the same batches it would have.
"""

from __future__ import annotations

import json
import os
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db

TRAINABILITY_WINDOW = (0.10, 0.80)  # baseline reward must land inside


class TrainError(RuntimeError):
    pass


# ── config ───────────────────────────────────────────────────────────────────


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


@dataclass
class TrainConfig:
    model: str
    env_id: str
    max_steps: int
    batch_size: int = 8
    rollouts_per_example: int = 8
    learning_rate: float = 1e-5
    checkpoint_every: int = 10
    seed: int = 0
    max_new_tokens: int = 256
    temperature: float = 1.0
    micro_batch_size: int = 8
    max_grad_norm: float = 1.0
    # Qwen3-style models burn the whole token budget inside <think> blocks;
    # small-model RL wants direct answers, so thinking is off by default.
    enable_thinking: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    raw_toml: str = ""


def load_config(path: str | Path) -> TrainConfig:
    p = Path(path)
    if not p.is_file():
        raise TrainError(f"Config file not found: {p}")
    text = p.read_text()
    try:
        data: dict[str, Any] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TrainError(f"Invalid TOML in {p}: {exc}") from exc

    for key in ("model", "max_steps"):
        if key not in data:
            raise TrainError(f"{p}: missing required key {key!r}")
    envs_ = data.get("env", [])
    if not isinstance(envs_, list) or len(envs_) != 1 or "id" not in envs_[0]:
        raise TrainError(
            f"{p}: exactly one [[env]] block with an 'id' is required in v0.1"
        )
    if int(data.get("rollouts_per_example", 8)) < 2:
        raise TrainError(
            f"{p}: rollouts_per_example must be >= 2 — GRPO advantages are "
            "computed within each example's group of rollouts"
        )
    if int(data["max_steps"]) < 1:
        raise TrainError(f"{p}: max_steps must be >= 1")

    lora_raw = data.get("lora", {})
    config = TrainConfig(
        model=str(data["model"]),
        env_id=str(envs_[0]["id"]),
        max_steps=int(data["max_steps"]),
        batch_size=int(data.get("batch_size", 8)),
        rollouts_per_example=int(data.get("rollouts_per_example", 8)),
        learning_rate=float(data.get("learning_rate", 1e-5)),
        checkpoint_every=int(data.get("checkpoint_every", 10)),
        seed=int(data.get("seed", 0)),
        max_new_tokens=int(data.get("max_new_tokens", 256)),
        temperature=float(data.get("temperature", 1.0)),
        micro_batch_size=int(data.get("micro_batch_size", 8)),
        max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        enable_thinking=bool(data.get("enable_thinking", False)),
        lora=LoraConfig(
            r=int(lora_raw.get("r", 16)),
            alpha=int(lora_raw.get("alpha", 32)),
            dropout=float(lora_raw.get("dropout", 0.0)),
        ),
        raw_toml=text,
    )
    return config


# ── GRPO math (pure, torch-free, unit-tested) ────────────────────────────────


def group_advantages(rewards: list[float], group_size: int) -> list[float]:
    """Per-group (mean/std)-normalized advantages.

    `rewards` is flat, laid out as consecutive groups of `group_size` rollouts
    of the same example. A zero-variance group (all rollouts equally good/bad)
    carries no learning signal and gets zero advantages.
    """
    import numpy as np

    r = np.asarray(rewards, dtype=np.float64)
    if len(r) % group_size != 0:
        raise TrainError(
            f"{len(r)} rewards do not divide into groups of {group_size}"
        )
    groups = r.reshape(-1, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    adv = np.where(std > 1e-8, (groups - mean) / (std + 1e-8), 0.0)
    return adv.reshape(-1).tolist()


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Terminal-friendly curve: one block char per step."""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(
        SPARK_CHARS[round((v - lo) / span * (len(SPARK_CHARS) - 1))] for v in values
    )


def check_trainability(mean_reward: float) -> None:
    """PI's product rule: outside the 10–80% window there is nothing to train."""
    lo, hi = TRAINABILITY_WINDOW
    if mean_reward < lo:
        raise TrainError(
            f"Pre-flight failed: baseline reward {mean_reward:.3f} < {lo} — the "
            "model can barely ever succeed, so GRPO gets no positive signal. "
            "Use an easier env, a stronger base model, or better prompts."
        )
    if mean_reward > hi:
        raise TrainError(
            f"Pre-flight failed: baseline reward {mean_reward:.3f} > {hi} — the "
            "task is already (nearly) solved, so there is nothing to learn. "
            "Use a harder env or a smaller base model."
        )


def batch_indices(seed: int, step: int, dataset_size: int, batch_size: int) -> list[int]:
    """Deterministic batch for (seed, step): resume redraws identical batches."""
    rng = random.Random(f"{seed}:{step}")  # str seeding is stable across runs
    return rng.sample(range(dataset_size), min(batch_size, dataset_size))


def grpo_backward(
    model,
    seqs,
    completion_ids,
    advantages: list[float],
    pad_token_id: int,
    micro_batch_size: int,
) -> float:
    """Advantage-weighted NLL over completion tokens; backpropagates gradients.

    The caller owns zero_grad / clip / optimizer.step. Micro-batched so long
    batches fit in memory; gradients accumulate across micro-batches.
    """
    import torch

    device = seqs.device
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    total_loss = 0.0
    n_micro = 0
    for i in range(0, seqs.shape[0], micro_batch_size):
        s = seqs[i : i + micro_batch_size]
        comp = completion_ids[i : i + micro_batch_size]
        a = adv[i : i + micro_batch_size]
        attn = (s != pad_token_id).long()
        logits = model(input_ids=s, attention_mask=attn).logits
        comp_start = s.shape[1] - comp.shape[1]
        # logits at position t predict token t+1
        logits = logits[:, comp_start - 1 : -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = torch.gather(logprobs, 2, comp.unsqueeze(-1)).squeeze(-1)
        mask = (comp != pad_token_id).float()
        seq_lp = (token_lp * mask).sum(dim=1)
        denom = mask.sum().clamp(min=1.0)
        loss = -(a * seq_lp).sum() / denom
        loss.backward()
        total_loss += float(loss.detach())
        n_micro += 1
    return total_loss / max(n_micro, 1)


# ── multi-turn rollouts → training pairs ────────────────────────────────────
# A multi-turn rollout (e.g. a Scribe episode) trains as one pair per
# assistant turn: (everything before the turn, the turn itself), all pairs
# sharing the episode's advantage. The pairs feed the SAME grpo_backward as
# single-turn batches — the loss never needs to know about conversations.


def turn_pairs(prompt_msgs: list[dict], completion_msgs: list[dict]) -> list[tuple[list[dict], str]]:
    """Explode a rollout into (context_messages, assistant_text) pairs."""
    context = [dict(m) for m in prompt_msgs]
    pairs: list[tuple[list[dict], str]] = []
    for message in completion_msgs:
        if message.get("role") == "assistant":
            pairs.append(([dict(m) for m in context], str(message.get("content", ""))))
        context.append(dict(message))
    return pairs


def collate_pairs(tokenizer, pairs, device, enable_thinking: bool = False):
    """Tokenize pairs into (seqs, completion_ids) shaped for grpo_backward.

    Each row is [left-pad | prompt | completion | right-pad→no] — completions
    are left-padded too so the completion block sits at the row's right edge,
    which is the alignment grpo_backward assumes.
    """
    import torch

    pad_id = tokenizer.pad_token_id
    rows = []
    for context, assistant_text in pairs:
        prefix = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            assistant_text + (tokenizer.eos_token or ""), add_special_tokens=False
        )["input_ids"]
        rows.append((prompt_ids, completion_ids))

    max_comp = max(len(c) for _, c in rows)
    max_total = max(len(p) + max_comp for p, _ in rows)
    seqs, comps = [], []
    for prompt_ids, completion_ids in rows:
        # completion block occupies the last max_comp columns: text first,
        # padding after, so positions stay contiguous with the prompt
        comp_padded = completion_ids + [pad_id] * (max_comp - len(completion_ids))
        full = prompt_ids + comp_padded
        seqs.append([pad_id] * (max_total - len(full)) + full)
        comps.append(comp_padded)
    return (
        torch.tensor(seqs, dtype=torch.long, device=device),
        torch.tensor(comps, dtype=torch.long, device=device),
    )


# ── environment scoring (offline; single-turn only) ─────────────────────────


def load_single_turn_env(env_id: str):
    from nanolab import envs as nanolab_envs

    env = nanolab_envs.load(env_id)
    if type(env).__name__ != "SingleTurnEnv":
        raise TrainError(
            f"{env_id} is a {type(env).__name__}; the v0.1 trainer supports "
            "single-turn environments only (the rubric must be able to score a "
            "bare completion). Multi-turn training arrives with the inference "
            "station."
        )
    return env


def score_completions(env, rows: list[dict], completions: list[str]) -> list[float]:
    """Score plain-text completions with the env's own rubric, no API."""
    import asyncio

    from verifiers.types import State

    async def _score_all() -> list[float]:
        rewards = []
        for row, text in zip(rows, completions):
            state = State.for_task(row)
            state["completion"] = [{"role": "assistant", "content": text}]
            state["trajectory"] = []
            await env.rubric.score_rollout(state)
            rewards.append(float(state.get("reward") or 0.0))
        return rewards

    return asyncio.run(_score_all())


# ── db bookkeeping ───────────────────────────────────────────────────────────


def start_run(conn, config: TrainConfig, env_row_id: int | None) -> int:
    cur = conn.execute(
        "INSERT INTO train_runs (env_id, model, config_toml, status, started_at)"
        " VALUES (?, ?, ?, 'running', ?)",
        (env_row_id, config.model, config.raw_toml, db.utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def find_resumable_run(conn, config: TrainConfig) -> int | None:
    row = conn.execute(
        "SELECT id FROM train_runs WHERE config_toml = ? AND status IN "
        "('running', 'failed') ORDER BY id DESC LIMIT 1",
        (config.raw_toml,),
    ).fetchone()
    return int(row["id"]) if row else None


def log_step(conn, run_id: int, step: int, mean_reward: float, loss: float) -> None:
    row = conn.execute(
        "SELECT reward_curve_json FROM train_runs WHERE id = ?", (run_id,)
    ).fetchone()
    curve = json.loads(row["reward_curve_json"] or "[]")
    curve = [p for p in curve if p["step"] != step]  # resume may rewrite a step
    curve.append({"step": step, "reward": mean_reward, "loss": loss})
    curve.sort(key=lambda p: p["step"])
    conn.execute(
        "UPDATE train_runs SET reward_curve_json = ?, steps_completed = ? WHERE id = ?",
        (json.dumps(curve), step + 1, run_id),
    )
    conn.commit()


def register_adapter(conn, run_id: int, base_model: str, step: int, path: str) -> int:
    cur = conn.execute(
        "INSERT INTO adapters (train_run_id, base_model, step, path, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (run_id, base_model, step, path, db.utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(conn, run_id: int, status: str) -> None:
    conn.execute(
        "UPDATE train_runs SET status = ?, finished_at = ? WHERE id = ?",
        (status, db.utcnow(), run_id),
    )
    conn.commit()


# ── the loop (GPU box; lazy heavy imports) ───────────────────────────────────


def train(config_path: str | Path, resume: bool = False) -> int:
    """Run (or resume) a training run; returns the train_run id."""
    config = load_config(config_path)

    # reduce CUDA fragmentation (must be set before torch touches the GPU)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import torch  # noqa: F401
        from peft import LoraConfig as PeftLoraConfig
        from peft import PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TrainError(
            "Training needs torch + transformers + peft (not installed by "
            "default — they're GPU-box dependencies). Install with: "
            "pip install torch transformers peft"
        ) from exc

    import torch

    env = load_single_turn_env(config.env_id)
    dataset = env.get_dataset()

    conn = db.connect()
    env_row = db.get_environment(conn, config.env_id)
    env_row_id = env_row["id"] if env_row else None

    run_id: int | None = None
    start_step = 0
    checkpoint_dir: Path | None = None
    if resume:
        run_id = find_resumable_run(conn, config)
        if run_id is None:
            raise TrainError("--resume: no interrupted run found for this config")
        row = conn.execute(
            "SELECT steps_completed FROM train_runs WHERE id = ?", (run_id,)
        ).fetchone()
        start_step = int(row["steps_completed"])
        ckpt = conn.execute(
            "SELECT path FROM adapters WHERE train_run_id = ? ORDER BY step DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        checkpoint_dir = Path(ckpt["path"]) if ckpt else None
        conn.execute(
            "UPDATE train_runs SET status = 'running' WHERE id = ?", (run_id,)
        )
        conn.commit()
        print(f"resuming run #{run_id} from step {start_step}")
    else:
        run_id = start_run(conn, config, env_row_id)
        print(f"train run #{run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=dtype, device_map=device
    )
    if checkpoint_dir is not None:
        model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            PeftLoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
    # needed so gradients reach LoRA params once checkpointing is on;
    # checkpointing itself is toggled per phase: ON for the loss pass only,
    # OFF for generation (it forces use_cache=False and breaks sampling)
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )
    if checkpoint_dir is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
        )

    def generate_batch(rows: list[dict]) -> tuple[list[str], "torch.Tensor", "torch.Tensor"]:
        model.gradient_checkpointing_disable()
        base.config.use_cache = True
        model.eval()
        try:
            prompts = [
                tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=config.enable_thinking,
                )
                for row in rows
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                seqs = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                    num_return_sequences=config.rollouts_per_example,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            completion_ids = seqs[:, prompt_len:]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            return texts, seqs, completion_ids
        finally:
            model.train()

    def grpo_step(seqs, completion_ids, advantages) -> float:
        # recompute-not-store during backward: the T4 memory saver
        model.gradient_checkpointing_enable()
        optimizer.zero_grad(set_to_none=True)
        loss = grpo_backward(
            model, seqs, completion_ids, advantages,
            tokenizer.pad_token_id, config.micro_batch_size,
        )
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), config.max_grad_norm
        )
        optimizer.step()
        return loss

    def save_checkpoint(step: int) -> str:
        path = Path("adapters") / f"run{run_id}" / f"step{step:05d}"
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
        (path / "trainer_state.json").write_text(
            json.dumps({"run_id": run_id, "step": step, "model": config.model})
        )
        register_adapter(conn, run_id, config.model, step, str(path))
        return str(path)

    try:
        # pre-flight: the trainability window, on the untouched policy
        if start_step == 0:
            rows = [dataset[i] for i in batch_indices(config.seed, -1, len(dataset), config.batch_size)]
            texts, _, _ = generate_batch(rows)
            expanded = [r for r in rows for _ in range(config.rollouts_per_example)]
            baseline = score_completions(env, expanded, texts)
            mean_baseline = sum(baseline) / len(baseline)
            print(f"pre-flight baseline reward: {mean_baseline:.3f}")
            print(f"pre-flight sample completion: {texts[0][:300]!r}")
            check_trainability(mean_baseline)

        for step in range(start_step, config.max_steps):
            rows = [dataset[i] for i in batch_indices(config.seed, step, len(dataset), config.batch_size)]
            texts, seqs, completion_ids = generate_batch(rows)
            expanded = [r for r in rows for _ in range(config.rollouts_per_example)]
            rewards = score_completions(env, expanded, texts)
            advantages = group_advantages(rewards, config.rollouts_per_example)
            if device == "cuda":
                torch.cuda.empty_cache()  # release generation buffers before the loss pass
            loss = grpo_step(seqs, completion_ids, advantages)
            mean_reward = sum(rewards) / len(rewards)
            log_step(conn, run_id, step, mean_reward, loss)
            print(f"step {step:4d}  reward {mean_reward:.3f}  loss {loss:.4f}")
            if (step + 1) % config.checkpoint_every == 0 or step == config.max_steps - 1:
                path = save_checkpoint(step)
                print(f"  checkpoint → {path}")
        finish_run(conn, run_id, "done")
        print(f"run #{run_id} complete")
        return run_id
    except BaseException:
        finish_run(conn, run_id, "failed")
        raise
    finally:
        conn.close()
