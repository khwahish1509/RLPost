# Caught cheating, three times

*Two weeks, a 0.6B model, one free GPU, and the thing nobody warns you about reward: the model will always find the laziest way to earn it.*

Here is a notebook a model wrote for me. It had six sessions of a conversation to remember, a 350-character budget, and one job: write down what a later reader would need. This is what the reader actually sees, after the cap truncates it:

```
- my neighbor's dog Kabir kept barking all night.
- watched an old western last night, decent.
- watched a cooking show last night, decent.
- long day, mostly meetings.
- weather here has been surprisingly sunny.
- slept badly, don't ask.
- had a scare — turns out I'm allergic to lactose.
- my cousin's dog Kabir kept barking all night.
- we adopted
```

It wrote 865 characters into a 350-character budget, so everything after "we adopted" fell off a cliff. Cut off: the cat's name. The user's job. The city they'd just moved to. The drink they'd switched to.

It kept the neighbor's dog. Twice.

That model had just scored **+0.486** on its own training curriculum. It looked like a success. It was a cheat — and it was the second of three.

---

## The setup

I spent about two weeks building **nanolab**, a self-hosted RL loop: environments → evaluation → GRPO + LoRA training on a free Kaggle T4 → serving the adapter on my own laptop → re-evaluation, all landing in one SQLite file. Then I pointed it at memory.

Two things had to be true before any result could mean anything.

**The evaluator had to be trustworthy.** Not "we tested it" — anchored. On an identical config, nanolab's eval station reproduces the reference `vf-eval` tool's numbers to every decimal: **0.875 vs 0.875**, per-example identical. It drives the same library code path on purpose instead of reimplementing it, and the check re-runs after any change to the rollout path.

**The trainer had to actually move a model.** Qwen3-0.6B on gsm8k, lr 5e-5 × 40 steps: **0.422 → 0.562** on 64 held-out questions (27/64 → 36/64), on the *final* checkpoint, no cherry-picking. Stated precisely: a two-proportion test gives z ≈ 1.6, p ≈ 0.11 — suggestive at this sample size, not significant. (An earlier version of my own writeup reported "≈2.3σ" using the wrong standard error. That correction is logged in the repo rather than quietly edited out.)

Then the memory task. A **Scribe** model can do exactly one thing: rewrite a capped notebook between sessions. A frozen, amnesiac reader then answers held-out questions using the notebook alone. Reward is **Lift** — reader accuracy with the notebook minus without it. Three guards: the cap kills log-dumping, held-out seeds kill memorization, and the frozen reader kills "the model just got better."

That design made the cheating *visible*. It did not prevent it.

## Cheat #1 — transcription

The first task was a chain of arithmetic problems where later steps need figures revealed earlier. Before training anything, the lab's calibration gate demands the baseline sit in the 10–80% window. It didn't.

An **untrained** Qwen3-0.6B already scored **Lift 0.905** — and 0.905 again against a real frontier reader, essentially matching a prompted frontier Scribe's own 0.857. Doubling the horizon to 16 tasks changed nothing.

The task wasn't measuring note-taking. It was measuring transcription, which the base model had already mastered. So the lab refused to train. That's the honest outcome, and it's the one worth having: **a good score on a solved task is a fake result.**

The fix was the task, not the model. Each record now buries the needed figure among one-off distractors, under a cap tight enough that copying everything overflows and truncation eats what mattered. That single change dropped the untrained baseline **0.905 → 0.548**.

Then training worked. On held-out streams, the trained Scribe scored **Lift 1.000 — 12 of 12, zero errors**. Its notebook: 189 characters, zero junk lines.

And it held under drift it never trained on:

| drift rung | prompted | trained |
|---|---|---|
| reuse hints removed | 0.536 | **1.000** |
| 5 distractors (trained on 3) | 0.518 | **1.000** |
| 12 tasks (trained on 8) | 0.443 | **1.000** |

The hints-removed rung is decisive: stripped of the labels it trained with, it still filters junk *by content*. Scope, stated plainly: same task family, deterministic reader, n=8 per rung.

## Cheat #2 — verbatim copying

Arithmetic chains prove mechanics. Real memory is language. So: multi-session user dialogues with stable facts, updated facts (only the latest counts), chatter, and **attribution traps** — other people's details that poison a later question if the notebook keeps them.

Untrained, the model's notes were **worse than no notes at all**: Lift −0.028. After 30 GRPO steps on a free T4: **+0.486**, reader accuracy 0.167 → 0.653.

Then I read the notebooks. That's the one at the top of this post. It hadn't learned selection — it had learned to copy everything, in order. It worked only because the 600-character cap barely bound at four sessions.

The fix was the task again: six sessions of chatter against a **350-character cap**, where a dump overflows 3.6×. Same held-out conversations, same reader, three models:

| model | Lift | notebook | over cap |
|---|---|---|---|
| untrained | −0.133 | 272 chars | 0/10 |
| the copier | +0.100 | 764 chars | **10/10** |
| retrained | **+0.367** | 221 chars | 0/10 |

The copier's collapse from +0.486 to +0.100 is the diagnosis confirmed: copying was its whole trick, and removing the room to copy left it with nothing.

What the new model learned — read off its own notebooks — was **compression**: prose crushed into dense `Key: Value` pairs, updates sometimes encoded outright (*"Matcha: Not. Cocoa: Present."*). Fact recall 0.87 vs the copier's 0.52.

I nearly got this one wrong too. My first analysis read the stored notebooks at full length — but the database keeps the untruncated message, and the reader only ever saw the first 350 characters. Applying the cap the way the reader sees it cut the copier's numbers substantially. **An analysis bug that flatters your result is still a bug.**

## Cheat #3 — compression without selection

Compression is a real skill. It was also, again, the laziest one that scored well — because I had priced *fitting the budget* and nothing else.

Counted across every held-out notebook, the compression model's **abstention accuracy was 0.00**. Every attribution trap survived. Stale values were kept as often as not. It compressed the neighbor's dog beautifully.

So I priced the failures: two recurring traps per stream, trap and update questions counting double, warm-started from the compression checkpoint. Held-out verdict under the weighted rules:

| | Lift | reader acc | traps written | stale kept |
|---|---|---|---|---|
| **selection model** | **+0.375** | 0.739 | **3/16** | **2/16** |
| compression model | −0.023 | 0.341 | 13/16 | 9/16 |
| untrained | −0.182 | 0.182 | 8/16 | 6/16 |

Abstention accuracy 0.19 → **0.81**. Update accuracy 0.19 → **0.75**. And the honest cost: plain-fact recall dipped 0.75 → 0.62, because being choosy occasionally drops something real.

A real reader confirms it. The same frozen notebooks re-graded by a frontier model instead of the deterministic checker: **+0.352** for the selection model, +0.011 for compression, +0.091 untrained. The trained lift survives essentially unchanged.

## What transferred, and what didn't

The obvious next question: did it learn *selection*, or did it learn *my templates*? So I built a second world — workplace standups, different phrasing, traps wearing different clothes (other teams' managers, vendors) — and ran the same models with zero retraining.

| out-of-domain | Lift | traps written |
|---|---|---|
| selection model | +0.159 | 10/16 |
| compression model | +0.182 | — |
| untrained | −0.159 | — |

Read that honestly, because it cuts both ways. **The core transfers**: both trained models stay solidly positive in a world they've never seen while the untrained model stays deeply negative, and update-discipline holds (stale values still deleted, update accuracy 0.69). **The attribution filter does not**: out-of-domain the selection model writes down 10 of 16 traps, versus 3 of 16 at home, and its edge over plain compression vanishes into noise.

The plain statement: *structured retention and update-replacement are transferable skills; fine-grained attribution filtering, at 0.6B trained on a single domain, is not.* That's the negative result, and it's the most useful sentence in the whole project.

## What I'd tell you

**The task is the science.** Every time I asked "how do I make the model better?" I was asking the wrong question. Every real improvement came from redesigning the world so the shortcut lost. You don't teach judgment by asking nicely.

**Read the outputs, not the scores.** Every discovery here came from opening actual notebooks. The scores told me *what*; only the raw text told me *why* — and twice the score was rising for a reason I'd have been embarrassed to publish.

**Calibrate before you train.** The 10–80% gate caught the too-easy task before a single GPU-hour was spent. A refusal to train is a result.

**Final ≠ best.** One run climbed to 0.875 and collapsed to zero by step 99. Per-decade checkpoints saved that experiment.

## Limitations, stated plainly

The data is synthetic and templated. The default reader is a deterministic string-matcher (a frontier reader confirms the headline, but training used the cheap one). The streams are tiny next to LoCoMo or LongMemEval — and I have not yet run a public benchmark, so **none of these numbers are comparable to published work**, and nothing here is a state-of-the-art claim. Sample sizes are n=8–12 per verdict. One model, one size, one seed.

What it *is*: a complete, self-hosted RL loop, four original environments, an honest three-stage result on trainable memory, and every number in the post clickable back to the rollout that produced it.

---

**Code, environments, and every receipt:** [github.com/khwahish1509/RLPost](https://github.com/khwahish1509/RLPost)

*Built on [verifiers](https://github.com/willccbb/verifiers) and the Prime Intellect environments format. Trained on free Kaggle T4s. Total spend: a few dollars of API credit.*
