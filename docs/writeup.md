# The full RL product loop, self-hosted: how it actually works end to end

*nanolab v1.0 — the story, with receipts. Part I is the loop; Parts II and
III are what it was built to reach: a small model trained, over three
generations, to choose what to remember.*

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
| 3 | 5e-5 | 40 | The corrected dose, stopped before the collapse zone. **Clean:** no collapse, and the *final* checkpoint beats base. |

**The headline** (run 3, final checkpoint, 64 held-out questions, greedy,
measured inside the training kernel's own exam):

```
base            0.422
trained         0.562     Δ +0.141  (≈2.3σ)
```

Two lessons worth highlighting. First, from run 2: **final ≠ best** — its
final checkpoint scored 0.000 after collapse while its step-19 peak scored
0.500 vs base 0.375; checkpoint every few steps and exam the checkpoints, not
the survivor. Second, from the lab's own instruments: the gain is strongest
inside the training-time token budget — with a doubled budget the base reads
0.625 at the station. The improvement is real *within its regime*, and the
instruments caught the regime before it could be over-claimed. Pin your
budgets across comparisons.

## Station 4: Serving — the loop closes on a laptop

The trained adapter (a few MB) is served as an OpenAI-compatible endpoint on
the training machine's own GPU — Apple Silicon works; no CUDA required. The
model string `Qwen/Qwen3-0.6B:7` resolves through the lab's registry to the
live local endpoint, which means the eval station can measure the lab's own
product: `nanolab eval run gsm8k -m Qwen/Qwen3-0.6B:7`. All five stations,
one machine.

## Part II — the memory experiment: learning that lives in text

Weights are one place learning can live; **context is the other**. The
`scribe-stream` environment isolates it: a frozen, amnesiac Player solves
task chains where later tasks need figures revealed earlier; the model under
test (the *Scribe*) can do exactly one thing — rewrite a capped notebook
between tasks. Reward = **Lift**: Player's score with the notebook minus
without. Anti-cheat trio: the cap kills log-dumping, held-out stream seeds
kill memorizing, the frozen Player kills "the model just got better."

**Signal exists (S1).** With a prompted frontier model as the Scribe, on 10
held-out streams: Player alone 0.0% → with notebook 85.7% (eval #5). The
notebook is the only bridge across time, so the whole gap is the notes.

**An honest failure, caught by the gate.** Training a small Scribe was next —
and the baseline measurement stopped it. An *untrained* Qwen3-0.6B already
scored Lift 0.905 against the checker (eval #16) **and 0.905 against a real
grok reader** (eval #17), matching the frontier Scribe. On those streams
note-taking reduces to transcription, which the base model had already
mastered; doubling the horizon to 16 tasks changed nothing. The trainability
gate refused to train. Correctly. There was no skill to teach.

**So the task was rebuilt to demand judgment.** Each record now buries the
needed figure among one-off distractor figures, under a binding 400-character
notebook cap: copy everything and you overflow, and truncation eats the
figures you needed. That single change dropped the untrained baseline to
**0.548** (eval #22) — inside the 10–80% window where a reward signal exists.
The rollout shows the mechanism: the base model transcribes all the noise
(649 chars, 11 junk lines) and loses what mattered.

**Training closed the gap (S2).** GRPO on the multi-turn path — the policy
served to the same anchored rollout engine, per-turn pairs through the same
loss — on a free Kaggle T4. Pre-flight 0.411; the curve saturates near 1.0 by
step 13 and holds, no collapse (the kernel hit Kaggle's 12-hour wall near
step 35; its per-decade checkpoints were recovered from the committed working
directory — checkpoint discipline paying out again). On held-out streams the
trained Scribe scores **Lift 1.000 — 12 of 12 streams, zero errors (eval
#24)** vs the untrained 0.548. Its final notebook: 189 characters, zero
distractor lines.

**The skill transfers.** Three drift rungs it never trained on, prompted vs
trained, server identity verified before every eval:

| rung | prompted | trained |
|---|---|---|
| hints removed | 0.536 (#28) | **1.000** (#25) |
| 5 distractors, trained on 3 | 0.518 (#29) | **1.000** (#26) |
| 12 tasks, trained on 8 | 0.443 (#30) | **1.000** (#27) |

The prompted model degrades with distance; the trained one holds, so the gap
*widens* (+0.46 → +0.56). The hints-removed rung is the decisive ablation:
without the labels it trained with, the Scribe still filters junk by content
— it learned selection, not label-copying. Scope, stated honestly: drift
within one task family, checker Player, n=8 per rung; a different task family
entirely is the untested fourth rung.

**Where the improvement lives.** The four-column instrument, all four columns
produced on one laptop (Player served locally): base 0.000 · +context +0.393
· +weights +0.000 · +both +0.429 (evals #19, #21). A gsm8k-trained adapter
with an empty notebook still scores zero — no arithmetic skill invents a
figure it was never shown. **KNOWLEDGE-DOMINANT: the notebook carries almost
everything; the weights alone carry nothing.** That is, in one table, why
memory matters.

One more receipt from this phase: a first "prompted" transfer ladder scored
an impossible 1.000 everywhere — traced to a supposedly-stopped server still
answering on the old port, so the "base" evals had silently measured the
trained model. The rows were deleted and re-run with an identity check before
every eval. The discipline of distrusting your own good news is the product.

## Part III — conversational memory: three generations of a skill

Arithmetic chains prove mechanics; the real target is language. The
`memory-stream` environment generates multi-session user dialogues with four
ingredient types: **stable facts** ("we adopted a cat — named him Ravi"),
**updated facts** ("I live in Pune now" — only the latest value counts),
**chatter** (one-off noise), and **attribution traps** — *other people's*
details ("my neighbor's dog Anya…") that poison a later question if the
notebook keeps them. After each session the Scribe rewrites a capped
notebook; a frozen reader then answers held-out questions from the notebook
alone. Question types are graded strictly: update questions fail append-only
notes (old value present = wrong); abstention questions fail sloppy
attribution (trap present = wrong). This design follows where the 2025–26
literature converged — fixed-budget memory rewritten per segment (MemAgent),
downstream-QA accuracy as the reward (Mem-α) — at a thousandth of the compute.

**The starting point was negative.** An untrained 0.6B Scribe's notebooks
made the reader *worse than no notes at all* (Lift −0.028, eval #32): it kept
the trap lines and the chatter and dropped the facts.

**Three training generations followed, each shortcut caught by reading
notebooks, each fix a redesigned curriculum — never a plea:**

| generation | curriculum | held-out result | what it actually learned |
|---|---|---|---|
| S3 | 4 sessions, loose 600-char cap | −0.028 → **+0.486** | *verbatim retention* — copied everything; the cap barely bound |
| S3b | 6 sessions vs a 350-char cap (dump overflows 3.6×) | **+0.367**, while the S3 copier collapses to +0.100 here | *compression* — dense `Key: Value` notes (221 chars, 0/10 over cap) — but still wrote down every trap |
| S3c | 2 recurring traps/stream; trap & update questions weighted 2×; warm-started from S3b | **+0.375** vs S3b's −0.023 under the same rules | *selection* |

The S3c mechanism, counted across every held-out notebook (cap applied as
the reader sees it): **traps written down 13/16 → 3/16; stale values kept
9/16 → 2/16**; abstention accuracy 0.19 → 0.81; update accuracy 0.19 → 0.75;
plain-fact recall paid a choosiness tax, 0.75 → 0.62. Retention →
compression → selection: the reward-hacking law, run three times on purpose.

**A real reader confirms it.** The same frozen notebooks, re-graded by grok
instead of the deterministic checker: S3c **+0.352** · S3b +0.011 ·
untrained +0.091 (vs +0.375 / −0.023 / −0.182 under the checker). The
trained lift survives a real reader essentially unchanged. The interesting
nuance: a smart reader partially *rescues* bad notebooks — grok reasons
about attribution and refuses traps the checker falls for — which makes the
trained margin the conservative number.

**And the transfer question gets a precise answer.** A second generator the
Scribe never trained on — standup-style work streams (projects, deadlines,
clients, budgets; trap flavours are other teams' managers and vendors) —
zero retraining: S3c +0.159 · S3b +0.182 · untrained −0.159. Read with the
mechanism counts, that decomposes cleanly: **structured retention and
update-replacement transfer** (S3c still deletes stale values out-of-domain,
3/16 kept); **fine-grained attribution filtering does not** (traps written
10/16 out-of-domain vs 3/16 at home — the filter keyed on personal-domain
surface cues). At 0.6B with single-domain training, part of "what to
remember" is a general skill and part is domain habit. The obvious next
experiment is mixed-domain training; it is left as exactly that.

**Scope, stated plainly:** n=8–12 per verdict, synthetic dialogues from
seeded generators, a deterministic checker as the training-time reader, one
model size, no public benchmark run yet. These are the claims' boundaries;
inside them, every number has a row.

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
disagree. As of v0.2 the UI renders this story as a living paper — every
number above wears a footnote mark that opens the raw rollout rows behind it.
Nothing in this writeup is a claim without a row behind it.

## What v1.0 is, in one sentence

A self-hosted laboratory that trains AI models and proves the training
worked — and its flagship experiment taught a tiny model, over three
curriculum generations, the skill of choosing what to remember: verified by
a real reader, decomposed under domain transfer, and receipted at every
step, including its own false starts.
