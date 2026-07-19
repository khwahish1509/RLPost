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

- [x] `evaluate.py`: rollout runner against any OpenAI-compatible endpoint — builds the same EvalConfig vf-eval builds and executes verifiers' own `run_evaluation`, so the code path is shared by construction
- [x] Rate-limit pacing + retries (defaults: max_concurrent 4, max_retries 10; throttled rollouts are recorded, not crashed on)
- [x] On-disk cache keyed by hash(env, model, params, n, r, seed) — an identical completed config is served from the db in <100ms; `--force` re-runs
- [x] Resume: `--resume` finds the newest incomplete results dir for the config and continues where it stopped (verifiers-native)
- [x] Rubric scoring → samples + eval_runs rows + ledger tokens
- [x] `nanolab eval run <env> -m <model> -n 20 -r 3`, `eval list`, `eval show <id>` with per-metric breakdown
- [x] `report.py` v1: leaderboard.html from db
- [x] **THE ANCHOR — PASSED 2026-07-16**: identical config (alphabet-sort, grok-4.20-0309-non-reasoning, n=10, r=1, T=0.0, c=1) gives identical results on both sides — avg 0.875, std 0.216, per-example rewards equal to every decimal. Re-pass command pair:
  `vf-eval alphabet-sort -k XAI_API_KEY -b https://api.x.ai/v1 -m grok-4.20-0309-non-reasoning -n 10 -r 1 -c 1 -T 0.0 --disable-tui` then
  `nanolab eval run alphabet-sort -m grok-4.20-0309-non-reasoning -n 10 -r 1 -c 1 -T 0.0 --force`

Done when (**THE ANCHOR**): on an identical config (env, model, seed, n), `nanolab eval run` matches `vf-eval`'s numbers. Every later refactor must re-pass this check. ✅

### Phase 3 — Training station bring-up (1 week)

- [x] TOML config parser mirroring the standard RL-training schema: `model`, `max_steps`, `batch_size`, `rollouts_per_example`, `[[env]]` — validated with clear errors
- [x] `train.py`: env dataset → rollouts (model.generate) → rewards (env's own rubric, scored offline — proven against gsm8k's real verifier) → GRPO step (on-policy advantage-weighted NLL, group-normalized) → LoRA adapter. v0.1 scope: single-turn envs
- [x] Pre-flight check hard-coded: baseline reward must be in the 10–80% window, else abort with a clear message (the trainability rule)
- [x] Reward curve logged per step into train_runs (resume-safe step rewrite)
- [x] Checkpoint every 10 steps; `--resume` restores adapter+optimizer+step, and batches derive from (seed, step) so a resumed run redraws identical batches
- [x] First run: Qwen3-0.6B on gsm8k, 50 steps, Colab T4 — **completed 2026-07-17**: zero crashes, 5 checkpoints, curve stored (`nanolab training show 1`). Post-mortem: lora_B grew 1.19→2.67 (learning real, no gradient bug) but lr 1e-5 capped total movement at ~5e-4 — held-out exam delta exactly +0.000. Run 2 config: lr 1e-4, 100 steps

Done when: a 50-step run completes (surviving at least one restart via resume), the curve is non-flat and stored, and an adapter file lands in `adapters/`.

### Phase 4 — Full training pass (1 week)

- [ ] Qwen3-1.7B on alphabet-sort or a gsm8k-class env, 100–200 steps (across sessions via resume)
- [ ] Adapter checkpoints registered in the adapters table with run + step
- [ ] Post-run: `nanolab eval` the adapter vs the base model on the same env

Done when: the trained adapter beats its own pre-training baseline inside nanolab's own eval.

### Phase 5 — Inference station + loop closure (3–5 evenings)

- [x] `serve.py`: launch vLLM with `--enable-lora`; deployments registered in the db with pid-liveness tracking (`create` / `list` / `stop`)
- [x] `nanolab deployments create <adapter-id>` / `list` / `stop`
- [x] `base:adapter` model strings resolved inside evaluate.py (adapter looked up, base sanity-checked, routed to the live local endpoint) — anchor re-passed after the change
- [x] Alternative path documented: merge LoRA → GGUF → llama.cpp for laptop serving (docs/serving.md)
- [ ] Live vLLM run on a CUDA box with a real adapter (needs Phase 3/4's Colab output)

Done when (**LOOP CLOSED**): `nanolab eval run <env> -m Qwen3-0.6B:<adapter>` — the trained adapter, measured through our own endpoint, inside our own eval station.

### Phase 6 — Ship v0.1.0 (2–4 evenings)

- [ ] README: loop diagram, 3-command quickstart, honest scope section
- [ ] A gif: eval → train → re-eval, the number going up
- [ ] Short writeup: "The full RL product loop, self-hosted — how it actually works end to end"
- [ ] Tag v0.1.0, publish

Done when: a stranger reproduces the Phase-2 anchor check from the README in under 15 minutes.

### Phase 7 — THE SCRIBE (2–3 weeks; the reason all of the above exists)

- [x] Stream environment (`environments/scribe_stream/`, a verifiers MultiTurnEnv): one episode = 8 dependent tasks; each later task needs a "figure" revealed only by an earlier task; `env_response` runs the frozen Player statelessly with only the notebook; the Scribe's sole output is the notebook (full rewrite per turn)
- [x] Lift metric in the rubric: Player's mean score on tasks 2–8 with notes minus without, on held-out streams (disjoint seed ranges)
- [x] Anti-cheat trio: ~1,500-token cap enforced by truncation in code · held-out eval seeds · frozen Player at temperature 0
- [x] **S1 PASSED (2026-07-17)**: prompted Grok Scribe on 10 held-out streams → **Lift 0.857** (Player: 0.0% without notes → 85.7% with), zero errors. There is signal to train on.
- [x] Cache all Player calls (disk cache in .cache/player, T=0 so the cache is honest; `player_model="fake"` for free offline mechanics — 6 tests cover the extremes)
- [ ] GRPO-train a small Scribe (Qwen3 0.6B–1.7B) on Lift reward using nanolab's own trainer
- [ ] S2 check: trained small Scribe ≥ prompted same-size Scribe on Lift

Done when: a small model, trained in our own lab, measurably out-teaches its untrained self — running on our own machinery.

### THE INSTRUMENT (rung one of the frontier ladder — no training required)

- [x] `nanolab instrument <run> [<run>]`: the four-column comparison — base · +context · +weights · +both — read from stored stream-eval runs, with the missing-knowledge vs missing-skill verdict computed from the gaps
- [x] Columns 1–2 live with real data: scribe-stream, frozen Player → base 0.000, +context +0.857
- [ ] Columns 3–4: rerun the stream eval with the Player served as `base:adapter` (needs the first score-moving adapter + a serving session)
- [ ] The north-star experiment: does a *trained* Scribe's lift transfer to less-similar tasks better than a prompted one's? (After S2.)

## THE MOLDING (v0.2 core): match the hosted-product experience, $0, single-user

Goal: a user should complete the whole loop — install env → eval → train → deploy → re-eval — **without leaving nanolab's UI**. Training rides Kaggle's free T4 via their API; serving rides the user's own machine.

### Phase A — Cloud training via the Kaggle API (the centerpiece)

- [ ] `nanolab/cloud.py`: Kaggle client wrapper (auth via `~/.kaggle/kaggle.json`; friendly setup errors). **Prerequisite: phone-verified Kaggle account + API token — user-side, one time.**
- [ ] Kernel builder: generate a script kernel (GPU+internet enabled) that clones the repo at the current commit, installs, runs `nanolab train <config> --resume`, runs the held-out exam, and leaves adapters+db+exam-output as kernel output
- [ ] `nanolab train --cloud <config>`: push the kernel, record the cloud run, return immediately
- [ ] Poller (CLI `nanolab cloud pull` + background thread in the API server): watch kernel status; on completion download output, unzip, **auto-merge** db records + adapters (formalize the merge into `nanolab/artifacts.py` — currently ad-hoc)
- [ ] UI: Training page gets **＋ New training run** (config dropdown from `configs/`); cloud runs show live status (queued / running on Kaggle · elapsed / merging / done) in the activity strip and Training table
- [ ] Tests with the Kaggle client mocked; honest-limits note in the UI (logs arrive at completion — Kaggle API constraint)

Done when: user clicks ＋ New training run, closes the laptop, reopens later, and the finished curve + adapter + exam delta are sitting in the Training tab, having never touched a notebook.

### Phase B — Local inference station (closes the loop with zero external deps)

- [ ] `nanolab/serve_local.py`: load base+adapter via transformers on **MPS** (Apple GPU) / CPU, wrap in the existing PolicyServer → OpenAI-compatible endpoint on localhost; register in `deployments` (kind: local)
- [ ] `nanolab deployments create <adapter-id> --local` + a **Deploy** button on training-run/adapter pages
- [ ] `eval run -m base:adapter` against the local endpoint — **LOOP CLOSED on the user's own machine** (slow ≠ untrue)
- [ ] Instrument columns 3–4 via a scribe-stream eval with the Player pointed at the local endpoint
- [ ] Playground page (v0.2 design doc): side-by-side base vs base:adapter chat on local serving

Done when: `nanolab instrument <run> <run>` prints all four columns produced entirely on one laptop.

### Phase C — Hub browsing polish (small)

- [ ] Surface hub environment discovery in the Environments page if the prime CLI exposes a listing/search; else keep install-by-name + link out

## AFTER v0.1 (the v0.2 flagship)

- [ ] **The instrument panel**: a local single-tenant web UI over the same SQLite file — design direction, tokens, IA, and stack are settled in `docs/frontend-direction.md`. v0.1 already ships the design language as the static lab notebook (`nanolab report`). Prerequisite: a thin read-only HTTP API over the db.

## ENGINEERING RULES (short, permanent)

1. Synchronous training loop only — no orchestrators, no async. At our scale a for-loop is correct.
2. Every environment reduces to one scalar reward. If it can't, it's not ready.
3. Cache and resume everything: API responses, rollouts, checkpoints. Assume every long process gets killed.
4. The Phase-2 anchor (nanolab matches vf-eval) gets re-run after any refactor of the rollout path.
5. One task > two evenings → cut its scope, don't extend it.
6. Every phase ends in an artifact (table, curve, adapter, page) — not a feeling.
7. Out of scope, permanently for v0.1: website, multi-tenant anything, new environment formats, the async trainer.
