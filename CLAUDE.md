# nanolab — context for Claude sessions

(Stands in for `nanolab-context.md`. Read `PLAN.md` next — it is the working plan and progress tracker.)

## What this project is

A self-hosted, single-user RL product loop:

**Environments** (verifiers-format tasks) → **Evaluations** (measure a model over rollouts) → **Training** (GRPO + LoRA) → **Inference** (serve the adapter via vLLM/llama.cpp) → measure again. One CLI (`nanolab`), one SQLite file, one closed loop.

The capstone (Phase 7) is **the Scribe**, a memory experiment: a stream environment where a persistent notebook is the only thing a small "Scribe" model can edit, a frozen Player model plays the tasks, and the Scribe's reward is **Lift** — how much its notes improve the Player on future held-out tasks. It is trained with nanolab's own trainer.

## How the pieces talk

- `nanolab/cli.py` — five verbs: `env`, `eval`, `train`, `deployments`, `report` (typer).
- `nanolab/envs.py` — install/load/list verifiers environments (Hub-compatible via the `prime` CLI); registers installs in the db.
- `nanolab/evaluate.py` — async rollout runner against any OpenAI-compatible endpoint; pacing, caching, resume, rubric scoring. (Phase 2)
- `nanolab/train.py` — TOML config → GRPO+LoRA synchronous loop → adapter + reward curve. (Phase 3)
- `nanolab/serve.py` — vLLM `--enable-lora` lifecycle; `base:adapter` model strings. (Phase 5)
- `nanolab/ledger.py` — token accounting.
- `nanolab/report.py` — static `leaderboard.html` from the db.
- `nanolab/db.py` — SQLite schema: `environments`, `eval_runs`, `samples`, `train_runs`, `adapters`, `ledger`. Default path `results/nanolab.db`, override with `NANOLAB_DB`.

## Reference code (in `reference/`, gitignored — clone if missing)

- `verifiers` (MIT) — our actual dependency; the environment format.
- `prime-rl` (Apache 2.0) — GRPO loss + training TOML schema reference only.
- `prime-cli` — CLI grammar reference.

## Engineering rules (permanent)

1. Synchronous training loop only — no orchestrators, no async trainer.
2. Every environment reduces to one scalar reward.
3. Cache and resume everything; assume every long process gets killed.
4. The Phase-2 anchor (nanolab eval matches `vf-eval` on identical config) is re-run after any rollout-path refactor.
5. One task > two evenings → cut scope.
6. Every phase ends in an artifact.
7. Out of scope for v0.1: website, multi-tenant anything, new environment formats, async trainer.

## Dev basics

- Python 3.11+ managed by uv; `uv sync` then `uv run nanolab ...`.
- Tests: `uv run pytest` (smoke tests must stay network-free).
- CI: GitHub Actions runs the smoke tests on every push.
