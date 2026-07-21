# nanolab

**A self-hosted reinforcement-learning lab: measure a model, train it, serve it, measure again — on hardware you already have.**

```
Environments ──▶ Evaluations ──▶ Training (GRPO + LoRA) ──▶ Inference ──▶ re-eval
     ▲                                                                      │
     └────────────────────── one CLI · one SQLite file ◀────────────────────┘
```

Everything lands in one SQLite file; a local web app (`nanolab ui`) and a static
lab notebook (`nanolab report`) render it. No cloud, no accounts, no build steps.

## Receipts, not claims

- **The anchor check.** nanolab's eval station builds the same configuration the
  reference `vf-eval` tool builds and executes it through the same library code
  path. Verified live: identical config → **0.875 vs 0.875**, matching to every
  decimal on every example — and re-verified after each change to the rollout
  path.
- **Training produces a measurably better model.** The clean run (Qwen3-0.6B,
  gsm8k, GRPO+LoRA, lr 5e-5 × 40 steps on a free Kaggle T4, launched by API):
  **base 0.422 → trained 0.562 on 64 held-out questions (+0.141, ≈2.3σ)** —
  the *final* checkpoint, no cherry-picking, no collapse. An earlier hotter
  run also taught the honest footnote: gains are measured within the
  training-time token budget (256), and the lab's own instruments caught and
  quantified that regime-dependence before it could be over-claimed. Every
  answer behind every number is in the db.
- **Trainability gate, hard-coded.** Training refuses to start unless the
  baseline reward sits in the 10–80% window (GRPO learns from mixed groups;
  all-failures or all-successes teach nothing). This gate caught two real bugs
  before they could waste GPU-hours.
- **Memory lift is real.** In the `scribe-stream` environment, a frozen Player
  scores **0%** on dependent tasks without notes and **85.7%** with a
  Scribe-maintained notebook (10 held-out streams, zero errors). The reward —
  **Lift** — is exactly that difference.
- **Training moves weights in the right direction.** One GRPO step on a live
  model: 3/3 rewarded completions became more likely, 5/5 punished completions
  less likely, base weights untouched (LoRA). It's a permanent regression test.

## Quickstart

```bash
uv sync
uv run nanolab env install primeintellect/gsm8k   # any hub environment works
uv run nanolab eval run gsm8k -m <model> -n 10    # any OpenAI-compatible endpoint
uv run nanolab ui                                  # the web app
```

Evals are cached (identical config returns in milliseconds), resumable, and
every API token is ledgered.

## The stations

| Command | What it does |
|---|---|
| `nanolab env install/list` | install verifiers-format environments (hub-compatible) |
| `nanolab eval run/show/list/compare` | rollouts + rubric scoring; rollout-level inspection; A/B deltas |
| `nanolab train <config.toml> --resume` | GRPO+LoRA, synchronous loop, checkpoint/resume, deterministic batch replay |
| `nanolab training list/show` | reward curves (terminal sparklines) + checkpoint registry |
| `nanolab deployments create/list/stop` | serve adapters via vLLM `--enable-lora`; `base:adapter` model strings |
| `nanolab instrument <run> [<run>]` | the four-column instrument (below) |
| `nanolab ui` / `nanolab report` | local web app / self-contained HTML notebook |

Training runs free on a Colab/Kaggle T4: `notebooks/train_gsm8k_colab.ipynb`
is four idempotent cells — on Kaggle, *Save & Run All* trains in the background
with no tab open. Multi-turn environments train through a built-in policy
server, so conversation-based rewards (like Lift) use the same trainer.

## The four-column instrument

Where does improvement actually live? For the same tasks:

| | column | measures |
|---|---|---|
| 1 | `base` | the frozen Player alone |
| 2 | `+context` | the Player reading a trained/prompted notebook |
| 3 | `+weights` | a LoRA-trained Player alone |
| 4 | `+both` | trained Player + notebook |

If +context ≈ +weights, the failure was **missing knowledge** — text closes it,
on any model, including closed ones. If +weights ≫ +context, it was **missing
skill** — only training closes it. Current live reading on `scribe-stream`:
`base 0.000 · +context +0.857 · +weights/+both pending the next adapter`.

## The Scribe (the destination)

`environments/scribe_stream/` is a stream environment: 8 chained tasks where
each later task needs a figure revealed only by an earlier one. A frozen Player
attempts each task statelessly; the model under test — the **Scribe** — can do
exactly one thing: rewrite a notebook capped at ~1,500 tokens. Reward = Lift.
Anti-cheat trio: the cap (kills log-dumping), held-out stream seeds (kills
memorizing), the frozen Player (kills "the model just got better"). The next
milestone is GRPO-training a small Scribe on Lift with nanolab's own trainer.

## Status

v0.1.0-dev. Working and verified: environments, evals (+anchor), the trainer
(mechanically proven; first score-moving run in progress), serving code, the
Scribe environment (S1 passed), the web app, 52 tests, CI. Remaining for
v0.1.0: a training run that beats its baseline on the held-out exam, and the
live served-adapter loop closure.

## Layout

```
├── nanolab/            # cli, api+ui, db, envs, evaluate, train, serve, instrument, …
├── environments/       # scribe_stream (verifiers MultiTurnEnv)
├── configs/            # training TOMLs
├── notebooks/          # one-click GPU training
├── scripts/            # the held-out exam
├── tests/              # network-free (CI runs these)
└── results/            # gitignored: the db, eval outputs, leaderboard
```

MIT license.
