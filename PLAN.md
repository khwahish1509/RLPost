# NANOLAB — THE BUILD APPROACH

One document. What to build, in what order. Start here.

Companion context: `CLAUDE.md` (full background). This file is the working plan **and the progress tracker** — check boxes as tasks complete.

## WHAT WE ARE BUILDING

Two things, one on top of the other:

**Part 1 — nanolab**: a working, self-hosted RL product loop: Environments (tasks) → Evaluations (measure a model) → Training (GRPO + LoRA) → Inference (serve the trained adapter) → measure again. One CLI, one SQLite file, one closed loop on our own machines.

**Part 2 — the Scribe** (the capstone): once the loop works, point it at memory. A stream environment where a notebook persists across a sequence of tasks; a frozen Player plays the tasks; the Scribe model only writes/edits the notebook; its reward is **Lift** — how much its notes improve the Player's score on future, unseen tasks. Train the Scribe with the same nanolab trainer. The lab is the machine; the Scribe is its most interesting tenant.

## ARCHITECTURE

```
              ┌────────────────────────────────────────────┐
              │                 nanolab CLI                 │
              │  env · eval · train · deployments · report  │
              └──────┬──────────┬──────────┬────────┬──────┘
                     │          │          │        │
               envs.py    evaluate.py   train.py  serve.py
              (verifiers   (rollouts +  (GRPO +   (vLLM /
               wrapper)     scoring)     LoRA)     llama.cpp)
                     └──────────┴─────┬────┴────────┘
                                    db.py  (SQLite: envs, eval_runs,
                                    samples, train_runs, adapters, ledger)
                                      │
                                  report.py → leaderboard.html
```

**Stack**: Python 3.11+ + uv · verifiers (environment format — full Hub compatibility) · TRL GRPOTrainer / Unsloth (training) · vLLM with `--enable-lora` (serving; llama.cpp as the laptop path) · SQLite · typer · Docker for sandboxed envs · Colab/Kaggle for GPU work.

**Reference code** (cloned into `reference/`, gitignored): `verifiers` (MIT — our dependency), `prime-rl` (Apache 2.0 — GRPO loss + TOML schema reference), Unsloth GRPO notebooks (training recipe).

## THE BUILD, PHASE BY PHASE

### Phase 0 — First contact (1 evening)

Goal: run PI's actual environment stack locally, once, before writing any code.

- [x] Install uv, then `uv tool install prime`
- [x] `prime env install primeintellect/alphabet-sort`
- [x] Get an API key for a Player model (Gemini AI Studio or Groq)
- [x] Run `vf-eval alphabet-sort -m <model> -n 10` locally — mean reward 0.759, zero errors
- [x] Save the output table to `results/phase0-notes.md`
- [x] Clone verifiers, prime-rl, prime-cli into a `reference/` folder; skim verifiers' AGENTS.md

Done when: the first eval table exists and you've seen an environment run end to end.

### Phase 1 — Skeleton (2–3 evenings)

- [x] `git init nanolab`, pyproject with typer + verifiers + httpx deps
- [x] `cli.py` with all verbs stubbed (each prints "not implemented")
- [x] `db.py`: create tables — environments, eval_runs, samples, train_runs, adapters, ledger
- [x] `envs.py`: wrap install/load/list (shell out to prime/uv, import verifiers directly); register installs in db
- [x] `tests/test_smoke.py` scaffold + GitHub Actions running it

Done when: `nanolab env install primeintellect/alphabet-sort && nanolab env list` works from a clean clone.

### Phase 2 — Evaluation station (1 week)

- [ ] `evaluate.py`: async rollout runner using verifiers' env interface against any OpenAI-compatible endpoint
- [ ] Rate-limit pacing + retries (free tiers throttle; treat throttling as normal, not an error)
- [ ] On-disk response cache keyed by hash(env, task, model, params) — repeat runs are instant
- [ ] Resume: an interrupted eval continues where it stopped
- [ ] Rubric scoring → samples + eval_runs rows + ledger tokens
- [ ] `nanolab eval run <env> -m <model> -n 20 -r 3`, `eval list`, `eval show <id>` with per-metric breakdown
- [ ] `report.py` v1: leaderboard.html from db

Done when (**THE ANCHOR**): on an identical config (env, model, seed, n), `nanolab eval run` matches `vf-eval`'s numbers. Every later refactor must re-pass this check.

### Phase 3 — Training station bring-up (1 week)

- [ ] TOML config parser mirroring PI's schema: `model`, `max_steps`, `batch_size`, `rollouts_per_example`, `[[env]]`
- [ ] `train.py`: env dataset → rollouts → rewards → GRPO step (TRL GRPOTrainer; Unsloth variant for Colab) → LoRA adapter
- [ ] Pre-flight check hard-coded: baseline reward must be in the 10–80% window, else abort with a clear message (the trainability rule)
- [ ] Reward curve logged per step into train_runs
- [ ] Checkpoint every 10 steps; `--resume` flag restores model+optimizer+step (sessions die; assume it)
- [ ] First run: Qwen3-0.6B, tiny env, 50 steps, on Colab T4

Done when: a 50-step run completes (surviving at least one restart via resume), the curve is non-flat and stored, and an adapter file lands in `adapters/`.

### Phase 4 — Full training pass (1 week)

- [ ] Qwen3-1.7B on alphabet-sort or a gsm8k-class env, 100–200 steps (across sessions via resume)
- [ ] Adapter checkpoints registered in the adapters table with run + step
- [ ] Post-run: `nanolab eval` the adapter vs the base model on the same env

Done when: the trained adapter beats its own pre-training baseline inside nanolab's own eval.

### Phase 5 — Inference station + loop closure (3–5 evenings)

- [ ] `serve.py`: launch vLLM with `--enable-lora` and runtime adapter loading; register served adapters
- [ ] `nanolab deployments create <adapter-id>` / `list`
- [ ] `base:adapter` model strings resolved inside evaluate.py
- [ ] Alternative path documented: merge LoRA → GGUF → llama.cpp for laptop serving

Done when (**LOOP CLOSED**): `nanolab eval run <env> -m Qwen3-1.7B:<adapter>` — the trained adapter, measured through our own endpoint, inside our own eval station.

### Phase 6 — Ship v0.1.0 (2–4 evenings)

- [ ] README: loop diagram, 3-command quickstart, honest scope section
- [ ] A gif: eval → train → re-eval, the number going up
- [ ] Short writeup: "The full RL product loop, self-hosted — how it actually works end to end"
- [ ] Tag v0.1.0, publish

Done when: a stranger reproduces the Phase-2 anchor check from the README in under 15 minutes.

### Phase 7 — THE SCRIBE (2–3 weeks; the reason all of the above exists)

- [ ] Stream environment (a verifiers MultiTurnEnv): one episode = N=8 related tasks in sequence; a markdown notebook persists across them, hard-capped at 1,500 tokens in code; `env_response` runs a frozen Player model on the next task and returns the outcome; the model under test is the Scribe, whose only output is notebook edits
- [ ] Lift metric in the rubric: average score on tasks 2–8 with notes minus without, computed on held-out task sets
- [ ] Anti-cheat trio: token cap (kills log-dumping) · held-out tasks (kills answer-memorizing) · frozen Player (kills "the model just got better")
- [ ] S1 check — before any training: a prompted Scribe (a strong API model) must produce clearly positive Lift on held-out streams. No lift = stop and investigate the environment, not the trainer
- [ ] Cache all Player calls (streams re-run constantly during Scribe development)
- [ ] GRPO-train a small Scribe (Qwen3 0.6B–1.7B) on Lift reward using nanolab's own trainer
- [ ] S2 check: trained small Scribe ≥ prompted same-size Scribe on Lift

Done when: a small model, trained in our own lab, measurably out-teaches its untrained self — running on our own machinery.

## ENGINEERING RULES (short, permanent)

1. Synchronous training loop only — no orchestrators, no async. At our scale a for-loop is correct.
2. Every environment reduces to one scalar reward. If it can't, it's not ready.
3. Cache and resume everything: API responses, rollouts, checkpoints. Assume every long process gets killed.
4. The Phase-2 anchor (nanolab matches vf-eval) gets re-run after any refactor of the rollout path.
5. One task > two evenings → cut its scope, don't extend it.
6. Every phase ends in an artifact (table, curve, adapter, page) — not a feeling.
7. Out of scope, permanently for v0.1: website, multi-tenant anything, new environment formats, the async trainer.
