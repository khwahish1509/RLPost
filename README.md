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
  path. The per-example receipt is committed at
  [`evidence/anchor/`](evidence/anchor/); re-run it yourself with
  `scripts/verify_anchor.sh`.
- **Training produces a measurably better model.** The clean run (Qwen3-0.6B,
  gsm8k, GRPO+LoRA, lr 5e-5 × 40 steps on a free Kaggle T4, launched by API):
  **base 0.422 → trained 0.562 on 64 held-out questions (27/64 → 36/64,
  +0.141)** — the *final* checkpoint, no cherry-picking, no collapse. Stated
  precisely: a two-proportion test gives **z ≈ 1.6, p ≈ 0.11** — suggestive
  at this sample size, not significant, and we say so rather than rounding it
  up. An earlier hotter run also taught the honest footnote: gains are
  measured within the training-time token budget (256), and the lab's own
  instruments caught and quantified that regime-dependence before it could be
  over-claimed. Every answer behind every number is in the db.
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
skill** — only training closes it. Full live reading on `scribe-stream`, all
four columns produced on one laptop (Player served locally on MPS, notebook
written by a frozen Scribe):

```
1 base       0.000     Qwen-0.6B reader, empty notebook
2 +context   0.393     ... reading the notebook
3 +weights   0.000     a gsm8k-trained adapter, empty notebook
4 +both      0.429     the adapter + the notebook
→ KNOWLEDGE-DOMINANT: notes lift +0.393; weight-training lifts +0.000 —
  no arithmetic skill can invent a figure the model was never shown.
```

## The Scribe (the destination)

`environments/scribe_stream/` is a stream environment: 8 chained tasks where
each later task needs a figure revealed only by an earlier one. A frozen Player
attempts each task statelessly; the model under test — the **Scribe** — can do
exactly one thing: rewrite a notebook capped at ~1,500 tokens. Reward = Lift.
Anti-cheat trio: the cap (kills log-dumping), held-out stream seeds (kills
memorizing), the frozen Player (kills "the model just got better").

**What the baseline measurement found (and why it matters).** Before training a
Scribe, the trainability gate demands the prompted baseline sit in the 10–80%
window. It doesn't: a prompted *untrained* Qwen-0.6B already scores **Lift
0.905** — it writes a clean `figure #N = value` ledger that a real reader uses
just as well (0.905 with a grok Player), matching a frontier Scribe's own 0.857.
On these streams, note-taking is *transcription*, which the base model has
already mastered; doubling the horizon to 16 tasks doesn't change it. So the lab
refuses to train — the honest outcome.

**So the task was rebuilt to demand *judgment*.** Each RECORD now buries the
needed figure among one-off distractors (tagged `needed later` / `one-off`), and
a tight notebook cap means copying everything overflows and truncates away what
matters. That single change moves the prompted base model from **0.905 → Lift
0.548** — into the trainable window: it blindly transcribes all the noise,
overflows the cap, and drops needed figures.

**And then training closed the gap.** GRPO+LoRA on this curriculum (free Kaggle
T4) moved the reward from a 0.411 pre-flight to ~1.0 within ~13 steps and held
it there. On **held-out** streams the trained Scribe scores **Lift 1.000 —
on all 12 of 12 streams, zero errors — vs the untrained ~0.55** (both sides
measured twice). The rollouts show *why*: it writes a 189-char notebook
(under the 400 cap) with **zero distractor lines**, where the untrained model
overflowed at 649 chars keeping 11 distractors and losing what mattered. A
0.6B model, trained in this lab, learned to *select what's worth remembering* —
and it generalizes to streams it never saw.

**And the skill now trains on real language.** The `memory-stream` environment
generates multi-session user dialogues — stable facts, updated facts (only the
latest counts), chatter, and attribution traps — and rewards the Scribe by how
much its notebook lifts a frozen reader on questions about earlier sessions.
Untrained, the model's notes score **−0.03** (worse than no notes at all);
after 30 GRPO steps on a free T4, **+0.49 on held-out conversations** — the
reader goes from 17% to 65% accuracy using its notebook. The lab's own
inspection kept it honest: that model learned *verbatim retention*, not
selection — it fit only because the cap barely bound.

**So the cap was tightened until copying was impossible.** Six sessions of
chatter against a 350-character notebook — a dump overflows 3.6×. Three
models, same held-out conversations, same reader:

```
untrained    -0.133   notebook 272 chars, keeps the chatter, loses the facts
S3 copier    +0.100   notebook 764 chars → truncated in 10/10 streams
S3b trained  +0.367   notebook 221 chars → fits in 10/10 streams
```

The copier's collapse (+0.486 → +0.100 once its cap actually bound) is the
diagnosis confirmed. What the new model learned, read off its own notebooks:
**compression** — prose rewritten as dense `Key: Value` pairs, updates
sometimes encoded outright ("Matcha: Not. Cocoa: Present."). Facts recalled
0.87 vs the copier's 0.52. What it had *not* yet learned: dropping other
people's details or deleting stale values.

**So those failures were priced into the reward — and the model learned to
choose.** Two recurring traps per stream, trap and update questions counting
double, training warm-started from the compression checkpoint. Held-out
verdict: **+0.375** vs the compression model's −0.023 under the same rules
(reader accuracy 74% vs 34%). The mechanism, counted across every held-out
notebook: traps written down fell **13/16 → 3/16**; stale values kept fell
**9/16 → 2/16**. Three training generations, each shortcut caught by reading
the notebooks and each fix forced by curriculum design: retention →
compression → **selection**. A 0.6B model, trained on free GPUs, that keeps
your facts, refuses your neighbor's, and erases what's out of date.

**And the skill transfers.** Tested across three drift rungs it never trained
on (reuse hints removed · 5 distractors instead of 3 · 12 tasks instead of 8),
the prompted base model degrades — 0.536 / 0.518 / 0.443 — while the trained
Scribe holds **1.000 on every rung**, so the gap *widens* with distance
(+0.46 → +0.56). The hints-removed rung is the decisive ablation: the Scribe
filters junk by content, not by copying the training labels. Scope stated
honestly: drift within the same task family, checker Player, n=8 per rung —
every rollout in the db.

## Status

**v0.2.0 — the research arc is complete.** Every question the lab was built
to answer is answered, with db rows behind each number: the loop improves a
model's weights (gsm8k +0.141 at n=64, final checkpoint); memory carries
signal (Lift 0.857); the note-taking skill is trainable (0.548 → 1.000 on
12/12 held-out streams); the skill transfers under drift (perfect across
three rungs while the prompted baseline decays); and the four-column
instrument attributes the gain (KNOWLEDGE-DOMINANT). The web app renders it
all as a living paper — every number opens its raw rollouts. 99 tests, CI.
Full story: [docs/writeup.md](docs/writeup.md). Next (v0.3): the interactive
Memory Agent — chat where the trained Scribe maintains the notebook live.

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
