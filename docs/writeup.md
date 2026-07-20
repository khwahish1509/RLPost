# The full RL product loop, self-hosted: how it actually works end to end

*nanolab v0.1 — the story, with receipts.*

## The premise

Reinforcement learning for language models is sold as a cloud product: pick a
task, click train, get a better model, pay per token. Every piece of that loop
is built from open-source parts. So the question this project answers:
**can one person run the entire loop — environments, evaluation, training,
serving, re-evaluation — on hardware they already have, for $0, without
trusting any number they didn't produce themselves?**

Yes. Here's how each station actually works, and what it measured.

## Station 1: Environments — the definition of "better"

An environment is a dataset of tasks plus a code grader (a *rubric*). It is
the only place in the whole system that knows what success means; the trainer
and evaluator are generic machinery. nanolab speaks the standard `verifiers`
format, so any of the 1,388 community environments installs unchanged — and
authoring your own is a few hundred lines (this repo ships two: `pattern-gym`,
with difficulty as a tunable dial, and `scribe-stream`, a memory experiment).

**The design rule that matters:** rewards must be checkable by code. A grader
an LLM can sweet-talk trains a sweet-talker.

## Station 2: Evaluation — trust through anchoring

An eval sends tasks to any OpenAI-compatible model, stores every answer, and
scores with the environment's own rubric. The correctness claim is not "we
tested it" but **anchoring**: on identical configs, nanolab reproduces the
reference tool's numbers to every decimal (verified: 0.875 vs 0.875,
per-example identical) — because it deliberately drives the same library code
path rather than reimplementing it. Every later change to the rollout path
re-runs that check.

## Station 3: Training — GRPO + LoRA, and what three runs taught

The loop is deliberately synchronous — generate 64 answers, grade them,
compare each to its group's average, nudge a LoRA adapter toward the
above-average ones, checkpoint every 10 steps. It runs on a free Kaggle T4,
launched via API (`nanolab train <config> --cloud`), and merges its results
home automatically.

Three runs, three lessons — reported honestly:

| run | lr | steps | what happened |
|---|---|---|---|
| 1 | 1e-5 | 50 | Mechanically perfect, learned nothing measurable: AdamW moves weights ≈ lr per step, and 50×1e-5 can't flip a greedy answer. **Underdose.** |
| 2 | 1e-4 | 100 | Reward climbed 0.156 → 0.875 by step ~17, plateaued, then **collapsed to 0.000** by step 99. Policy collapse from a hot lr. The *final* checkpoint scored 0.000 on the exam — but per-decade checkpoints preserved the peak. |
| 3 | 5e-5 | 40 | The corrected dose, stopped before the collapse zone. |

**The result that matters** (run 2's peak checkpoint, step 19, vs the
untouched base — same 32 held-out questions, greedy decoding, same rubric):

```
base            12/32 correct   (0.375)
trained          16/32 correct  (0.500)    Δ +0.125
```

The lesson worth a highlight: **final ≠ best. Checkpoint every few steps and
exam the checkpoints, not the survivor.**

## Station 4: Serving — the loop closes on a laptop

The trained adapter (a few MB) is served as an OpenAI-compatible endpoint on
the training machine's own GPU — Apple Silicon works; no CUDA required. The
model string `Qwen/Qwen3-0.6B:7` resolves through the lab's registry to the
live local endpoint, which means the eval station can measure the lab's own
product: `nanolab eval run gsm8k -m Qwen/Qwen3-0.6B:7`. All five stations,
one machine.

## The memory experiment — learning that lives in text

Weights are one place learning can live; **context is the other**. The
`scribe-stream` environment isolates it: a frozen, amnesiac Player solves
task chains where later tasks need figures revealed earlier; the model under
test (the *Scribe*) can do exactly one thing — rewrite a capped notebook
between tasks. Reward = **Lift**: Player's score with the notebook minus
without.

Measured, with a prompted frontier model as the Scribe, on 10 held-out
streams: **Player alone 0.0% → with notebook 85.7%**. The notebook is the
only bridge across time, so the entire gap is attributable to the notes.
The next experiment (S2) trains a small model to *be* the Scribe — GRPO
where the reward is Lift — using this same trainer.

## What made it work at $0

Free GPUs die mid-run and free APIs throttle, so the engineering is built
around interruption: checkpoints carry optimizer state and batches derive
deterministically from (seed, step), so resume replays exactly; evals cache
at the run level and continue where they stopped; every long process assumes
it will be killed. Robustness against interruption mattered more than speed
at every single decision point.

## Where every number lives

One SQLite file holds every eval sample, reward curve, adapter, deployment
and API token spent. The web UI and CLI read the same rows, so they cannot
disagree. Nothing in this writeup is a claim without a row behind it.
