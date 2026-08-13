# Two weeks, one laptop: I built an AI training lab, then taught a tiny model what's worth remembering

*And found that only half the skill transferred.*

---

AI models forget everything. Close the chat, and it's gone. Everyone building AI agents fights this the same way — save everything to a database, search it later. Engineering.

I wanted to test a different idea: **what if deciding what to remember is a skill — something a model can be *taught*?**

To find out, I first had to build the machine that could teach it. This is the story of both: the lab, and the experiment it was built for. Everything ran on my laptop and free cloud GPUs. Total spend: about $5.

---

# Part 1: The machine

## One rule before any code

**Don't trust a number you didn't produce yourself.**

Most AI results are a screenshot with no way to check it. I decided every number in this project would live in one database, with the raw model answers stored behind it. That rule shaped everything — and it caught me being wrong four separate times before anyone else could.

## Check the ruler first

The lab's first station measures models: give a model a task with an automatic grader, run it many times, average the scores.

Before measuring anything, I checked the measuring stick itself. There's a standard open-source tool for these evaluations; I built my lab to drive that same code — not to reimplement it — and ran both on an identical setup:

```
reference tool : 0.875
my lab         : 0.875
```

Identical to every decimal, on every single question. If the ruler is wrong, everything after it is decoration. This check re-runs after any change to how answers are generated.

## Training, and the run that collapsed

Training here is reinforcement learning: the model tries a task many times, good attempts get reinforced, and only a small add-on file changes (a LoRA adapter) — the base model stays frozen.

One guard rail first: **the lab refuses to train if the model's starting score is above 80% or below 10%.** Ace the task and there's nothing to teach; fail everything and there's no signal. This gate later saved an entire experiment.

Then three runs on a free cloud GPU, on school math:

| Run | Setting | What happened |
|---|---|---|
| 1 | gentle, 50 steps | Ran perfectly. Learned nothing. Underdosed. |
| 2 | aggressive, 100 steps | Great score by step 17 — then **collapsed to zero** by step 99. |
| 3 | middle, stop early | Clean. |

Run 2 is the one I think about. Judged by its *final* version, training destroyed the model. But I'd saved a snapshot every 10 steps, so I could go back and find the peak. **The last checkpoint is not the best checkpoint.**

The clean run, on 64 questions the model had never seen: **42.2% → 56.2%**.

Two honest footnotes, because they're the point of the project. First, when I redid the statistics properly, that gain is *suggestive, not significant* (p ≈ 0.11) — the sample is small, and my first write-up overstated it. Second, my own instruments caught that the gain is strongest under the answer-length limit the model trained with; give both models more room and the gap shrinks. Real improvement, narrower claim.

## The loop closes

The trained adapter serves on my Mac as a normal API endpoint — which means the lab can measure its own product. Tasks → measure → train → serve → measure again. Five stations, one laptop, $0.

That was week one. The machine was never the point, though. Here's what it was for.

---

# Part 2: Teaching memory

## The game

A fake user chats with an assistant across several sessions. They mention facts: *"we adopted a cat — named him Ravi."* Facts change: *"I live in Pune now, not Berlin."* There's small talk. And there are **traps** — other people's details, like *"my neighbor's dog Anya kept barking all night."*

Two models play:

- **The Scribe** — a tiny 0.6B model whose only job is rewriting a notebook after each session. The notebook is capped — deliberately too small to hold everything.
- **The Reader** — a frozen model that never learns and never sees the conversation. At the end, it answers questions using **only the notebook**.

The score is **Lift**: how much the notebook improves the Reader over having no notes. Useless notes score zero. Misleading notes score *negative*.

The questions are graded without mercy:

- **Facts** — "What's my cat's name?" The detail must be in the notes.
- **Updates** — wrong if the *old* value is still there. Adding without deleting fails.
- **Traps** — "What's my dog's name?" Correct answer: *"unknown"* — the dog belongs to the neighbor. If the notebook copied that line, the Reader confidently gets it wrong.

A good notebook has to **keep**, **replace**, and **refuse**.

## The experiment that refused to run

My first version used arithmetic chains instead of chat. Before training, I measured the untrained model — and it already scored **0.905**. The task only required copying numbers down, which the base model did perfectly. Nothing to teach. The trainability gate refused to run, and it was right.

I rebuilt the task to require judgment — needed values buried in junk, notebook too small for both. The untrained score dropped to 0.548. *Then* training worked: **0.548 → 1.000** on unseen chains. Proof the machinery could teach selection — on arithmetic. Language turned out to be a three-act story.

## Act 1: it learned to copy

On real conversations, the untrained model scored **−0.028** — negative. Its notes made the Reader *worse than no notes*: it wrote down the small talk and the neighbor's dog, and dropped the facts.

After training: **+0.486**. Big jump — until I read the notebooks. It had learned to **copy everything word for word**, and scored well only because the notebook happened to be just big enough.

**A model learns the laziest strategy that still gets rewarded.** Never the elegant one you imagined.

## Act 2: it learned to compress

I made copying impossible: more sessions, more noise, and a notebook the conversation overflows **3.6 times over**.

The copier collapsed from +0.486 to **+0.100** — its notebooks got cut off, losing the facts at the end. And the newly trained model, at **+0.367**, showed a genuinely new skill. Its notebooks stopped being sentences:

```
Ravi: Cat. Berlin: Living. Lactose Allergy: Present.
Matcha: Not. Cocoa: Present.
```

221 characters. Fit every time. Real compression.

But it still wrote down the neighbor's dog. Every single time.

## Act 3: it learned to choose

I made junk expensive: two traps per conversation, trap and update questions counting **double** in the reward, training warm-started from the compression model.

Same unseen conversations, same grader:

| Model | Lift |
|---|---|
| **Act 3 (selection)** | **+0.375** |
| Act 2 (compression) | −0.023 |
| Untrained | −0.182 |

The compression champion drops *below zero* once junk is priced properly. And this time the *behaviour* changed, counted across every notebook:

| Behaviour | Act 2 | **Act 3** |
|---|---|---|
| Wrote down the trap | 13 / 16 | **3 / 16** |
| Kept the outdated value | 9 / 16 | **2 / 16** |
| Trap questions right | 19% | **81%** |
| Update questions right | 19% | **75%** |

There was a cost: plain fact recall slipped from 75% to 62%. Choosy sometimes means dropping something you needed. Worth it here.

**Retention → compression → selection.** Every shortcut caught by *reading the notebooks*, never the scores. Every fix a redesigned world, never a nicer prompt.

## Did it just learn to please my grader?

Fair question — my grader was a simple checker program. So I took the exact same notebooks, frozen, and had a real AI (Grok) answer the questions instead:

| Model | My checker | **Real AI reader** |
|---|---|---|
| Act 3 | +0.375 | **+0.352** |
| Act 2 | −0.023 | +0.011 |
| Untrained | −0.182 | +0.091 |

Barely moved. With the trained notebooks, the real reader answered at **72% versus 36% with no notes**.

One lovely detail: the *untrained* model improves under a real reader. A smart reader rescues bad notes — it sees "my cousin's dog Kabir" and reasons *that's not the user's dog*, refusing a trap my checker fell for. Which makes the trained margin the **conservative** number.

---

# The finding: half the skill moved

Everything above was learned on **personal chat** — pets, cities, allergies, neighbors.

So I built a world the model had never seen: **workplace standup talk.** Projects, deadlines, clients, budgets. Different words, different rhythms, different traps — now it's *"the platform team's manager Lena."* No retraining. Just drop it in.

| Model | At home | **At work (never seen)** |
|---|---|---|
| Act 3 (selection) | +0.375 | **+0.159** |
| Act 2 (compression) | −0.023 | **+0.182** |
| Untrained | −0.182 | **−0.159** |

A big drop — and the two trained models tie. But the notebooks show exactly *what* broke:

**Transferred:**
- **Structured note-taking.** Both trained models stay clearly positive in a world they never saw; untrained stays hopeless.
- **Replacing outdated values.** At home: stale values kept 2/16. At work: still just **3/16**. Clean transfer.

**Did not transfer:**
- **Refusing other people's details.** At home: 3 traps written out of 16. At work: **10 out of 16.**

The filter never learned the *idea* of "this belongs to someone else." It learned the *sound* of it — "my neighbor's…", "my cousin's…". Say the same trap in office language and it walks right past.

So here's the sentence I'd want another builder to take away:

> **A learned memory skill is a bundle, and the parts travel differently. Structure and updating are portable skills. Attribution filtering — at this size, trained on one domain — is a domain habit wearing a skill's clothes.**

The obvious fix is training on mixed domains. I haven't run it yet. I'd rather tell you that than imply I have.

---

## What I'm not claiming

- **Small scale.** 8–12 test conversations per verdict, one seed. Directional, not statistically heavy.
- **My own tasks.** Synthetic conversations from generators I wrote. Real chat is messier.
- **One model size.** 0.6B. Published RL-memory work (Memory-R1, Mem-α) runs at 3B–14B, and my transfer failure may partly be a capacity limit — I can't rule that out without a bigger model.
- **No public benchmark yet.** The field has standard tests (LongMemEval, LoCoMo); production systems report ~92–94 on them. My numbers aren't comparable, because I haven't run those tests. That's next.
- **The raw data lives in my lab's database** — every conversation, notebook, and answer, inspectable in the app. The public repo carries the code, the environments, and the technical write-up to regenerate everything; it does not yet ship the database itself. Making the headline numbers one-command reproducible for strangers is on the list, and I'm saying so rather than pretending it's already true.

## Four rules I'd keep

1. **Check your ruler first.** The 0.875 = 0.875 anchor is why anything after it was trustworthy.
2. **Measure before you train.** The gate that refused the too-easy task saved me from publishing a fake win.
3. **Read the outputs, not the scores.** Scores said *something* changed. Only the notebooks said *what* — and twice the answer was "it's cheating."
4. **Distrust your own good news.** An untrained model once scored suspiciously well; a leftover server on an old port was secretly serving the trained one. Those rows got deleted, and every test now asks the server "who are you?" first. The result you love is the one to check hardest.

That's the real theme. This lab caught me being wrong four times — a too-easy task, a one-regime gain, a wrong statistic, a wrong server. Each catch made the project smaller and truer.

---

## Where this goes

Next, in order: a public benchmark number (LongMemEval), more repeats with error bars, a bigger model to separate "skill doesn't transfer" from "model too small," and the mixed-domain training my own results are begging for.

If you're building memory into an agent, the one thing worth taking today: **don't assume a fine-tuned memory model behaves the same outside its training data — ask which parts of the skill are actually portable.** Mine kept the facts and updated them faithfully in a world it had never seen. It just forgot whose facts they were.

Code, environments, and the full technical write-up: **github.com/khwahish1509/RLPost**
