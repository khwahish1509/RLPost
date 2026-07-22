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

- [x] Run 2 on Kaggle (via the cloud API): Qwen3-0.6B on gsm8k, lr 1e-4, 100 steps, 10 checkpoints registered as run #2 (adapters #6–#15)
- [x] Post-run exam on the final checkpoint: base 0.422 → step99 0.000 (delta −0.422) — but the curve shows a **policy collapse**: climbed to 0.875 by step ~17, then decayed to 0 by step 99. Final ≠ best. The per-decade checkpoints let us evaluate the peak instead.
- [x] **Evaluate the peak checkpoint (step 19) vs base — 2026-07-20: base 0.375 → step19 0.500, delta +0.125** (12→16 correct on 32 held-out gsm8k, greedy, our own rubric). Run 3 config (lr 5e-5, 40 steps) set for a clean full-64 run.

- [x] **Two-instrument addendum (2026-07-20)**: the official station head-to-head at n=32 with a 512-token budget reads base 0.625 vs trained 0.562 (Δ −0.062, within noise), while the 256-token exam reads base 0.375 vs trained 0.500 (Δ +0.125). Diagnosis: the base loses heavily to truncation at 256; training (run at 256) taught budget-fit answers. The improvement is real **within the training regime** and evaporates outside it — run 2's collapse-scarred checkpoint is not a robust win. Run 3 (clean run) is the proper test; future comparisons must pin max_tokens.

- [x] **RUN 3 — the clean pass (2026-07-20)**: lr 5e-5 × 40 steps, no collapse (max 0.891, healthy to the end). Kernel exam on 64 held-out questions: **base 0.422 → final checkpoint 0.562, Δ +0.141 (≈2.3σ)**. No checkpoint-hunting: the *last* checkpoint wins. Merged as train run #3 (adapters #16–19).

Done when: a trained checkpoint beats its own pre-training baseline inside nanolab's own eval. **✅ DONE — cleanly, at n=64, on the final checkpoint.**

### Phase 5 — Inference station + loop closure (3–5 evenings)

- [x] `serve.py`: launch vLLM with `--enable-lora`; deployments registered in the db with pid-liveness tracking (`create` / `list` / `stop`)
- [x] `nanolab deployments create <adapter-id>` / `list` / `stop`
- [x] `base:adapter` model strings resolved inside evaluate.py (adapter looked up, base sanity-checked, routed to the live local endpoint) — anchor re-passed after the change
- [x] Alternative path documented: merge LoRA → GGUF → llama.cpp for laptop serving (docs/serving.md)
- [ ] Live vLLM run on a CUDA box with a real adapter (the fast path; local path already proven)
- [x] **LOOP CLOSED — 2026-07-19, locally**: `nanolab deployments create 5 --local` served run-1's adapter on Apple-GPU via the policy server; `nanolab eval run gsm8k -m Qwen/Qwen3-0.6B:5` scored 1.000 (2/2) through the lab's own endpoint. All five stations, one laptop, no CUDA, $0.

Done when (**LOOP CLOSED**): `nanolab eval run <env> -m Qwen3-0.6B:<adapter>` — the trained adapter, measured through our own endpoint, inside our own eval station. ✅

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
- [x] Multi-turn trainer wired: PolicyServer + `env.generate` + per-turn pairs → unchanged `grpo_backward`; env-aware cloud kernel (`nanolab train --cloud configs/qwen3-0.6b-scribe.toml` installs `environments/scribe_stream` as an editable package, skips the gsm8k exam); `num_tasks` is now a first-class env arg (horizon knob)
- [x] **S2 BASELINE MEASURED (2026-07-21) — the trainability gate did its job.** Prompted *untrained* Qwen3-0.6B as the Scribe already scores **Lift 0.905** against the fake Player (eval #16, n=6, 8 tasks) *and* **0.905** against a real grok Player (eval #17) — essentially matching the frontier grok Scribe's own **0.857** (eval #5). Its clean `figure #N (item) = value` ledger transfers to a real reader unchanged; its only failures are occasional notebook *collapse*. Doubling the horizon to 16 tasks did not open a gap (≈0.95, eval #18). **Diagnosis: on these streams note-taking reduces to transcription, which the base model has already mastered — there is no trainable gap, and the 0.8 trainability ceiling correctly refuses to train.** A good RL engineer does not fake a win on a solved task.
- [x] **HARDER CURRICULUM BUILT & VALIDATED (2026-07-21) — the task is now trainable.** `scribe_stream` gained `distractors_per_task` + `mark_reuse`: each RECORD buries the needed figure among one-off distractors (tagged `needed later` / `one-off`), and a tight `notebook_char_cap` makes copying everything overflow. Measured: base Qwen drops from **0.905 → Lift 0.548** (eval #22, n=6, 3 distractors/task, cap 400) — inside the 10–80% window. Mechanism confirmed from the rollout: the base model transcribes all 4 lines/record, ignores the tags, overflows the 400-char cap (notebook 649 chars), and truncation drops needed figures. A model that keeps only `needed later` lines fits and scores ~1.0 → ~0.45 of learnable headroom. `configs/qwen3-0.6b-scribe.toml` now trains on this curriculum (fake Player → $0 on a free Kaggle T4).
- [x] **S2 PASSED (2026-07-22) — THE LAB TRAINED A MEMORY SKILL INTO A SMALL MODEL.** `nanolab train --cloud` on the hard curriculum: reward climbed from pre-flight **0.411 → ~1.0 by step ~13 and held steady** (no collapse). The Kaggle kernel hit the ~12h GPU limit at ~step 35, but Kaggle committed the working dir on timeout, so checkpoints step 9/19/29 survived (recovered to `adapters/scribe_s2/`). **Held-out verdict (eval #23, step 29, n=6): Lift 1.000 vs the untrained baseline's 0.548 (eval #22)** — a +0.452 jump on streams it never trained on. Mechanism verified from the rollout: the trained Scribe writes a 189-char notebook (fits the 400 cap) with **zero distractor lines**, versus the base model's 649-char overflow that kept 11 distractors and dropped needed figures. It learned to select what's worth keeping.
- [x] S2 check: trained small Scribe (1.000) ≫ prompted same-size Scribe (0.548) on held-out Lift ✓

Done when: a small model, trained in our own lab, measurably out-teaches its untrained self — running on our own machinery. **✓ MET 2026-07-22: 0.548 (untrained) → 1.000 (trained) on held-out streams.**

### THE INSTRUMENT (rung one of the frontier ladder — no training required)

- [x] `nanolab instrument <run> [<run>]`: the four-column comparison — base · +context · +weights · +both — read from stored stream-eval runs, with the missing-knowledge vs missing-skill verdict computed from the gaps
- [x] **ALL FOUR COLUMNS LIVE ON ONE LAPTOP (2026-07-21)** — a self-consistent reading with the *same* Player family, notebook written by a grok Scribe, Player served locally on MPS (base on :58001, run-3 adapter #19 on :58002). Eval #19 (base Player) + eval #21 (adapter Player): **base 0.000 · +context +0.393 · +weights +0.000 · +both +0.429**. Verdict **KNOWLEDGE-DOMINANT**: the notebook lifts the Player by +0.393 while a gsm8k-trained adapter *with no notes* still scores 0.000 — no arithmetic skill can invent a figure it was never shown. (The frontier-reader ceiling is higher: with a grok Player the same notebook lifts +0.857, eval #5 — the 0.6B reader simply can't always do the arithmetic even once it has the figures.)
- [ ] The north-star experiment: does a *trained* Scribe's lift transfer to less-similar tasks better than a prompted one's? (Gated on a trainable Scribe — see the harder-curriculum item above.)

## THE MOLDING (v0.2 core): match the hosted-product experience, $0, single-user

Goal: a user should complete the whole loop — install env → eval → train → deploy → re-eval — **without leaving nanolab's UI**. Training rides Kaggle's free T4 via their API; serving rides the user's own machine.

### Phase A — Cloud training via the Kaggle API (the centerpiece)

- [x] `nanolab/cloud.py`: Kaggle client wrapper (auth via `~/.kaggle/kaggle.json`; friendly setup errors) — **user completed phone-verify + token 2026-07-19**
- [x] Kernel builder: script kernel (GPU+internet) that clones the repo, installs, `nanolab train <config> --resume`, runs the held-out exam, zips adapters+db as output
- [x] `nanolab train --cloud <config>` pushes and records in the new `cloud_runs` table — **first live push 2026-07-19: `KernelWorkerStatus.RUNNING` on Kaggle within seconds**
- [x] `nanolab cloud list/status/pull` — pull downloads output and auto-merges via `nanolab/artifacts.py` (formalized, tested round-trip)
- [ ] UI: Training page **＋ New training run** button + cloud-run status in the activity strip (API server background poller)
- [x] Tests with the Kaggle client mocked (74 total)

Done when: user clicks ＋ New training run, closes the laptop, reopens later, and the finished curve + adapter + exam delta are sitting in the Training tab, having never touched a notebook.

### Phase B — Local inference station (closes the loop with zero external deps)

- [x] `nanolab/serve_local.py`: base+adapter on cuda→mps→cpu (auto-picked), served through the PolicyServer with request sampling passthrough; plain-load+move (device_map hangs on MPS — learned live)
- [x] `nanolab deployments create <adapter-id> --local` + Deploy/Stop buttons and an adapters registry on the Inference page
- [x] `eval run -m base:adapter` against the local endpoint — **LOOP CLOSED 2026-07-19** (eval #10, reward 1.000)
- [ ] Instrument columns 3–4 via a scribe-stream eval with the Player pointed at the local endpoint
- [x] Playground shipped (single-model chat with any deployment; side-by-side A/B compare deferred to v0.2)

Done when: `nanolab instrument <run> <run>` prints all four columns produced entirely on one laptop.

### Phase C — Hub browsing polish (small)

- [x] Hub browser shipped: search + star-ranked grid over the 1,388 hub environments with one-click install, inside the Environments tab

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
