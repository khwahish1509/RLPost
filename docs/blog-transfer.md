# I taught a tiny AI what to remember. It cheated. Then it got good.

*Two weeks. One laptop. Free GPUs. About $5. Every number below is from a real run.*

---

Here is a notebook a tiny model wrote for me.

It had six chats to remember and a 350-character budget. This is what the next model actually saw — the rest got cut off:

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

It kept the neighbor's dog. Twice.

It lost the cat's name. The city. The job. The drink they switched to. Cut off at *"we adopted"*.

That model had just scored **+0.486**. It looked like a win. It was cheating — copying everything, in order, until the page ran out.

I caught it by reading the notebook. Not the score.

Then I made copying impossible. Then I made junk expensive. Then the same tiny model learned to choose. And then I put it in a world it had never seen, and only half the skill came with it.

This is that story. No screenshots of a demo. No "vibes." Real runs.

---

## The scoreboard (real runs only)

I built a small lab on my laptop called **nanolab**. It can measure a model, train it, serve it, and measure it again. Everything lands in one database, with the raw answers behind every number.

Here is what the real runs showed.

**The ruler works.** A standard eval tool scored **0.875**. My lab scored **0.875**. Same setup. Same questions. Same answers, to every decimal. If the ruler is wrong, nothing after it matters.

**Training can move a small model.** Qwen3-0.6B — a model you can run on a laptop — on school math it had never seen: **27 / 64 right → 36 / 64 right** (42% → 56%). That's the *last* save of the run, not a cherry-picked peak. Free cloud GPU.

**A memory skill can be taught.** I gave the model one job: rewrite a tiny notebook. A second model, frozen, later answers questions using *only* that notebook. Score = how much the notes help vs no notes at all.

On number-puzzles it had never seen: **untrained 0.55 → trained 1.00**. **12 / 12 streams. Zero errors.**

On chat: a real AI reading those notes went from **36% right with no notes → 72% with the trained notes**.

**And then half of it failed in a new world.** Same model. No extra training. Workplace chat instead of personal chat. It still kept facts and updated old ones. It forgot whose facts they were.

Those five things are the project. Everything else is how I got there.

---

## The game

A fake person chats over a few days.

- *"We adopted a cat — named him Ravi."*
- Later: *"I live in Pune now, not Berlin."*
- Small talk. Weather. TV.
- Traps: *"My neighbor's dog Anya kept barking all night."*

Two models play.

**The Scribe** is tiny (0.6B). After each chat it rewrites a notebook. The notebook is too small to hold everything. That is the point.

**The Reader** never learns, and never sees the chat. At the end it answers from the notebook alone.

Questions are mean on purpose:

- *What's my cat's name?* — the fact must be there.
- *Where do I live?* — the old city still in the notes is a fail.
- *What's my dog's name?* — the right answer is *"I don't know."* The dog is the neighbor's. If the notebook copied that line, the Reader says Anya with confidence.

A good notebook has to **keep**, **replace**, and **refuse**.

---

## First, the lab said no

I started with number puzzles, not chat. Later puzzles need numbers from earlier ones. The Scribe's job: write them down.

Before training, I measured the untrained model. It already scored **0.905**. It was just copying numbers. Nothing to teach.

So the lab **refused to train**. Starting score too high. That refusal was the right result. A perfect score on an easy task is a fake win.

I made the puzzles harder: needed numbers buried in junk, notebook too small for both. Untrained score fell to **0.55**.

Then training worked. Held-out puzzles, never seen:

**0.55 → 1.00. 12 out of 12. Zero errors.**

The trained notebook was 189 characters. Zero junk lines.

I also tried three harder versions it never trained on — no hint labels, more junk, longer chains. The untrained model got worse (0.54 → 0.52 → 0.44). The trained one stayed at **1.00** every time.

So the machine can teach "keep what matters." On numbers. Chat was harder.

---

## Chat, act 1 — it copied

Untrained, on chat, the notes made the Reader *worse than no notes*: **−0.028**. It wrote the small talk and the neighbor's dog. It dropped the facts.

I trained it. Score jumped to **+0.486**. Reader went from 17% to 65%.

Then I opened the notebooks. That's the one at the top of this post. Copy. Everything. In order.

**A model learns the laziest trick that still gets paid.** Not the clever one you wanted.

---

## Chat, act 2 — it compressed

I made copying lose. More chats. More noise. A notebook so small the full dump overflows **3.6 times**.

The copier fell from +0.486 to **+0.100**. Its notes got cut off. The facts were at the end. Gone.

A new run, same hard setup: **+0.367**. Notebooks stopped looking like sentences:

```
Ravi: Cat. Berlin: Living. Lactose Allergy: Present.
Matcha: Not. Cocoa: Present.
```

221 characters. Fit every time. That's real. That's compression.

It still wrote down the neighbor's dog. Every time.

---

## Chat, act 3 — it chose

I made junk expensive. Two traps per chat. Trap and update questions counted **double**. I started from the compression model, not from scratch.

Same new chats. Same grader.

| Model | Lift | Traps written | Old values kept |
|---|---|---|---|
| **Trained to choose** | **+0.375** | **3 / 16** | **2 / 16** |
| Compression model | −0.023 | 13 / 16 | 9 / 16 |
| Never trained | −0.182 | 8 / 16 | 6 / 16 |

Trap questions: **19% → 81%**. Update questions: **19% → 75%**.

Plain facts slipped a bit (75% → 62%). Choosy means you sometimes drop a real thing. Worth it here.

**Copy → compress → choose.** I did not get there by asking nicer. I changed the game until the cheat stopped working. I knew it was a cheat because I read the notebooks.

---

## "Maybe it just learned to fool your grader?"

My grader is a simple checker. Fair question.

I froze the notebooks — same text, no retraining — and let a real AI (Grok) answer instead.

| Model | My checker | Real AI |
|---|---|---|
| **Choose** | **+0.375** | **+0.352** |
| Compress | −0.023 | +0.011 |
| Never trained | −0.182 | +0.091 |

Barely moved. With the trained notes, the real AI got **72% right vs 36% with an empty notebook**.

Funny bit: bad notes look *better* to a smart reader. It sees *"my cousin's dog"* and thinks *that's not your dog*. My checker falls for it. So the trained model's lead is the *smaller*, safer number.

---

## The test that mattered

All of that was personal chat. Pets. Cities. Neighbors.

I built a second world the model had never seen: **work standups**. Projects, deadlines, clients. Traps like *"the platform team's manager Lena."* No extra training. Drop it in.

| Model | Home | Work (never seen) |
|---|---|---|
| Choose | +0.375 | **+0.159** |
| Compress | −0.023 | **+0.182** |
| Never trained | −0.182 | **−0.159** |

Both trained models still help. The untrained one still hurts. So *something* transferred.

What, exactly?

**Came with it**

- Writing short structured notes
- Replacing old facts (stale values kept: 2/16 at home, **3/16 at work**)

**Did not**

- Refusing other people's details (traps written: 3/16 at home, **10/16 at work**)

It never learned the idea *"this belongs to someone else."* It learned the *sound* of it — *"my neighbor's…"*, *"my cousin's…"*. Say the same trap in office English and it writes it down.

That's the sentence I want you to take:

**A memory skill is a bundle. The parts don't travel together.** Keeping a fact and updating a fact can be real skills. Knowing whose fact it is — at this size, trained on one kind of chat — can just be a habit.

I have not trained on mixed worlds yet. I'm saying that so I don't pretend I have.

---

## What this is not

Small tests: 8–12 chats per result. Tasks I wrote, not a public benchmark. One model size. I have not run LongMemEval or LoCoMo yet, so **don't compare these numbers to papers**. That's next.

What it *is*: real training runs on a free GPU, real notebooks I read by hand, a second grader that didn't change the story, and a new world where half the skill stayed and half fell off.

---

If you build memory into an agent: **don't trust a fine-tune in a new domain until you check which part moved.** Mine kept the facts. It updated them. It forgot who they belonged to.

Code and the full write-up: [github.com/khwahish1509/RLPost](https://github.com/khwahish1509/RLPost)
