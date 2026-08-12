# Two weeks, one laptop: I built an AI training lab, then taught a tiny model what's worth remembering

*And found that only half the skill transferred.*

---

There's a kind of software that sounds like it needs a company behind it: the machine that makes AI models better. You give it a task, it measures your model, trains it, serves the improved version, and measures again. Companies sell this as a cloud product, priced per token.

Every piece of it is built from open parts. So I wanted to know:

**Can one person run that entire loop on hardware they already own, for nothing — and then use it to answer a real research question?**

Two weeks later: yes, and the research question gave an answer I didn't expect.

This is the whole story — the lab first, then the experiment it was built for.

---

# Part 1: The machine

## The rule I set before writing any code

Don't trust a number you didn't produce yourself.

That sounds obvious. It isn't. Most AI results are a screenshot of a number with no way to check it. I decided every number in this project would be clickable — stored with the raw answers behind it, in one database file.

That single rule shaped everything, and it caught me lying to myself four separate times. More on that later.

## Station 1 and 2: tasks, and an honest ruler

A **task** here isn't a dataset — it's problems plus a grader that scores answers automatically. No human marking. The grader is the only part of the system that knows what "better" means.

Then the measuring station. And here's where I did the single most important thing in the project: **I checked the ruler before measuring anything with it.**

There's a standard open-source tool for running these evaluations. I built mine to drive that same underlying code — not to reimplement it — and then ran both on an identical configuration.

```
reference tool : 0.875
my lab         : 0.875
```

Identical to every decimal, on every individual question. That check gets re-run any time I change how answers are generated. It's the foundation: if the ruler is wrong, every result after it is decoration.

## Station 3: training, and three runs that taught me the trade

Training here is reinforcement learning: the model tries a task many times, each attempt gets scored, and attempts that beat the group average get reinforced. Only a small add-on file changes (a LoRA adapter) — the original model is frozen.

I built one guard rail before running anything: **the lab refuses to train if the starting score is above 80% or below 10%.** If a model already aces a task, there's nothing to teach; if it fails everything, there's no signal. This gate later saved me an entire wasted experiment.

Then three runs on a free cloud GPU, and each one taught me something:

| Run | Setting | What happened |
|---|---|---|
| 1 | gentle learning rate, 50 steps | Ran perfectly. Learned **nothing** measurable. Underdosed. |
| 2 | hot learning rate, 100 steps | Climbed to a great score by step 17… then **collapsed to zero** by step 99. |
| 3 | in between, stop early | Clean. No collapse. |

Run 2 is the one I think about. If I'd only kept the *final* model, I would have concluded training destroyed it. Because I saved a snapshot every 10 steps, I could go back and find the peak. **The last checkpoint is not the best checkpoint** — that lesson cost me nothing only because I'd built for it in advance.

Run 3, the clean one, on 64 questions the model had never seen:

```
before training : 42.2%
after training  : 56.2%
```

**An honest note on that number.** I originally described it as a strong result. When I went back and did the statistics properly, it's **p ≈ 0.11** — suggestive, not statistically significant at this sample size. The direction is right; the sample is small. I'm reporting the corrected version because the first version was wrong and lives in my git history either way.

There was a second correction, too. My own instruments caught that the improvement was strongest *inside the answer-length budget the model trained under* — give both models a longer budget and the gap shrinks into noise. Real gain, narrower claim than I first wanted to make.

## Station 4: it runs on the laptop

The trained add-on file gets served on my own machine — an Apple laptop, no special hardware — as a normal API endpoint. Which means the measuring station can then measure the lab's own product.

That closed the loop. Tasks → measure → train → serve → measure again. Five stations, one laptop, one free cloud GPU, $0.

Building the machine was the point of week one. But the machine was only ever a means. Here's what it was for.

---

# Part 2: Can you *train* memory?

## The problem

AI models forget everything. Close the chat and it's gone. So every agent has the same problem: **what do you write down, and what do you throw away?**

The industry answer is engineering — save everything to a database and search it later. The research bet is different: maybe *deciding what to keep* is a skill a model can learn.

I wanted to test that bet at small scale. So I built a memory game.

## The game

A fake user chats across several sessions and mentions things: *"we adopted a cat — named him Ravi."* Some facts change: *"I live in Pune now, not Berlin."* There's small talk. And there are **traps** — other people's details, like *"my neighbor's dog Anya kept barking."*

Two models play:

- **The Scribe** — a tiny 0.6B model. Its only job is to rewrite a notebook after each session. The notebook is capped — deliberately too small to hold everything.
- **The Reader** — a frozen model that never learns and never sees the conversation. At the end it answers questions using **only the notebook**.

The score is **Lift**: how much better the Reader does with the notebook than with an empty one. Useless notes score zero. Misleading notes score *negative*.

Three question types, graded strictly:

- **Facts** — needs the detail in the notes.
- **Updates** — wrong if the *old* value is still there. Adding without deleting fails.
- **Traps** — "What's my dog's name?" The right answer is *"unknown."* If the notebook copied that line down, the Reader confidently gets it wrong.

So the notebook must **keep**, **replace**, and **refuse**.

## First: the gate refuses to train

My first version of this game used arithmetic instead of chat. Before training anything, I measured the untrained model — and it already scored **0.905**.

The task was too easy. It only required copying numbers down, which the base model already did perfectly. There was no skill to teach, and the trainability gate refused to run.

I could have trained anyway and reported a nice-looking number. Instead I rebuilt the task to require judgment — burying needed values among junk, under a notebook too small for both. The untrained score dropped to **0.548**, inside the trainable window. *Then* training worked: **0.548 → 1.000** on unseen tasks.

That was arithmetic, though. The real target is language.

## Generation 1: it learned to copy

On real conversations, the untrained model scored **−0.028**. Negative — its notes made the Reader *worse than no notes at all*. It wrote down the small talk and the neighbor's dog, and dropped the actual facts.

After training: **+0.486**. Big jump.

Then I read its notebooks. It had learned to **copy everything, word for word**, and scored well only because the notebook happened to be just big enough to fit. It hadn't learned to choose anything.

**A model learns the laziest strategy that still gets rewarded.** Not the elegant one you imagined. This would happen twice more.

## Generation 2: it learned to compress

So I made copying impossible: more sessions, more noise, and a notebook that the full conversation overflows **3.6 times over**.

The old copy-everything model collapsed from +0.486 to **+0.100** — its notebooks now got cut off mid-sentence. Diagnosis confirmed.

The newly trained model hit **+0.367**, and its notebooks showed a genuinely new skill — **compression**:

```
Ravi: Cat. Berlin: Living. Lactose Allergy: Present.
Matcha: Not. Cocoa: Present.
```

221 characters. Fit every time.

But it still wrote down the neighbor's dog. Every single time. It had learned to fit more in — not to *choose*.

## Generation 3: it learned to choose

So I made junk expensive: two traps per conversation, and trap/update questions counting **double** in the reward. I also warm-started from the compression model so it refined that skill instead of relearning it.

Same unseen conversations, same grader:

| Model | Lift |
|---|---|
| **Gen 3 (selection)** | **+0.375** |
| Gen 2 (compression) | −0.023 |
| Untrained | −0.182 |

The compression champion scores *below zero* once junk is priced properly.

And the behaviour changed, not just the score — counted across every notebook it wrote:

| Behaviour | Gen 2 | **Gen 3** |
|---|---|---|
| Wrote down the trap | 13 / 16 | **3 / 16** |
| Kept the outdated value | 9 / 16 | **2 / 16** |
| Trap questions right | 19% | **81%** |
| Update questions right | 19% | **75%** |

It paid a price too: plain fact recall slipped from 75% to 62%. Being choosy sometimes means dropping something you needed.

**Retention → compression → selection.** Every shortcut was caught by *reading the notebooks*, never by looking at the score. Every fix was a redesigned task, not a better prompt. You don't ask a model for judgment — you build a world where every shortcut loses.

## But did it just please my grader?

Fair challenge. My grader was a simple checker program.

So I took the exact same notebooks — already written, frozen — and had a real AI (Grok) answer the questions instead.

| Model | My checker | **Real AI reader** |
|---|---|---|
| Gen 3 | +0.375 | **+0.352** |
| Gen 2 | −0.023 | +0.011 |
| Untrained | −0.182 | +0.091 |

Barely moved. With those notebooks, the real reader answered questions about the user at **72% accuracy versus 36% with no notes**.

One detail I liked: the *untrained* model looks better under a real reader. Why? A smart reader partially rescues bad notes — it reads "my cousin's dog Kabir" and reasons *that's not the user's dog*, refusing a trap my checker fell for. Which makes my headline number the **conservative** one.

---

# The finding: half the skill moved

Here's the question I actually cared about. All of this was learned on **personal chat** — pets, cities, allergies, "my neighbor's dog."

What happens somewhere it's never been?

I built a second world: **workplace standup talk.** Projects, deadlines, clients, budgets. Different words, different rhythms, different traps — now it's *"the platform team's manager Lena."* No retraining. Just drop the model in.

| Model | At home | **At work (never seen)** |
|---|---|---|
| Gen 3 (selection) | +0.375 | **+0.159** |
| Gen 2 (compression) | −0.023 | **+0.182** |
| Untrained | −0.182 | **−0.159** |

A big drop, and the two trained models basically tie. But the notebook counts show exactly *what* broke.

**What transferred:**

- **Structured note-taking.** Both trained models stay clearly positive in a world they never saw; the untrained one stays hopeless.
- **Replacing outdated values.** At home it kept stale values 2 times in 16. At work — brand new domain — still just **3 in 16**. Clean transfer.

**What didn't:**

- **Refusing other people's details.** At home: 3 traps written out of 16. At work: **10 out of 16.**

Its filter never learned the *idea* of "this belongs to someone else." It learned the *sound* of it — *"my neighbor's…"*, *"my cousin's…"*. Rephrase the same trap as *"the platform team's manager"* and it walks right past.

So a learned memory skill isn't one thing. It's a bundle, and the parts have different strength:

> **Structure and updating are portable skills. Attribution filtering — at this size, trained on one domain — is a domain habit wearing a skill's clothes.**

The obvious fix is training across mixed domains. I haven't run it. It's the next experiment, and I'd rather say that than imply I have.

---

## What I'm not claiming

This matters more than any result above.

- **Small scale.** 8–12 conversations per test. Directional, not statistically heavy.
- **My own tasks.** Synthetic conversations from generators I wrote. Real chat is messier.
- **One model size.** 0.6B. Published work in this area runs at 3B–14B, and my transfer failure may partly be a capacity limit.
- **No public benchmark yet.** There are standard tests for this (LongMemEval, LoCoMo) and production systems report ~92–94 on them. **My numbers are not comparable**, because I haven't run them. That's a real gap and it's next.
- **The math result is suggestive, not significant** (p ≈ 0.11), as noted above.

What I *do* claim: inside those limits, every number has raw data behind it — every conversation, every notebook, every answer, stored and inspectable.

---

## Four things I'd tell anyone doing this

**1. Check your ruler first.** Anchoring my evaluation against the reference tool (0.875 = 0.875) is why I could trust anything afterwards.

**2. Measure before you train.** The gate that refuses to train on a solved task saved me from a fake win.

**3. Read the outputs, not the scores.** Every real discovery here came from opening notebooks. Scores told me *that* something changed; only the notebooks told me *what* — and twice the answer was "it's cheating."

**4. Distrust your own good news.** At one point an *untrained* model scored suspiciously well. The cause: a leftover server on an old port was quietly serving the trained model instead. I deleted those results and now every test asks the server "who are you?" before running. The result you love is the one to check hardest.

That last one is the theme. This lab caught me being wrong four times — a task that was too easy, a gain that only held in one regime, a statistic I computed with the wrong formula, and a measurement that wasn't measuring what I thought. Each catch made the project smaller and truer.

---

## Where it leaves me

The machine works: five stations, one laptop, free GPUs, every number receipted.

The experiment answered its question: yes, a tiny model can be *taught* to decide what's worth remembering — in three generations, each one blocking the last one's shortcut.

And the unexpected part, the one I'd actually want another researcher to take: **when you train a memory skill, you're training several skills at once, and they don't travel equally.** Keeping and updating facts came along for the ride. Knowing *whose* facts they were stayed home.

If you're putting memory into an agent, that distinction is worth knowing before you assume a fine-tuned memory model behaves the same way outside its training data.

Code, raw data, and the full technical write-up: **github.com/khwahish1509/RLPost**

*Everything ran on a laptop and free cloud GPUs. Total spend: about $5 of API credit for verification.*
