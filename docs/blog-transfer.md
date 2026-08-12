# I trained a tiny AI to take notes. Only half the skill transferred.

*What happens when you teach a 0.6B model to decide what's worth remembering — and then move it to a world it has never seen.*

---

AI models forget everything. Close the chat, and it's gone. So everyone building AI agents is solving the same problem: **what do you write down, and what do you throw away?**

Most people solve it with engineering — save everything to a database, search it later. I wanted to know something different:

**Can a model be *taught* the skill of choosing what to remember? And if it learns that skill in one setting, does it still work somewhere else?**

I spent two weeks finding out, on a laptop and free cloud GPUs. The answer turned out to be *half yes* — and the half that failed is the interesting part.

---

## The setup

I built a small memory game.

A fake user chats with an assistant across six sessions. They mention things: *"we adopted a cat — named him Ravi."* Some facts change later: *"I live in Pune now, not Berlin."* There's small talk. And there are **traps** — other people's details, like *"my neighbor's dog Anya kept barking."*

Two models play:

- **The Scribe** — a tiny 0.6B model. Its *only* job is to rewrite a notebook after each session. The notebook is capped at 350 characters — deliberately too small to hold everything.
- **The Reader** — a frozen model that never learns. At the end, it answers questions using **only the notebook**. It never sees the conversation.

The score is called **Lift**: how much better the Reader does *with* the notebook than with an empty one. If the notes are useless, Lift is zero. If they mislead, Lift goes negative.

The questions are graded strictly, which matters:

- **Fact questions** — "What's my cat's name?" Needs the fact in the notes.
- **Update questions** — wrong if the *old* value is still in the notebook. Adding without deleting fails.
- **Trap questions** — "What's my dog's name?" The correct answer is *"unknown"* (the dog belongs to the neighbor). If the notebook copied that line down, the Reader confidently gets it wrong.

So the notebook has to do three different things: **keep**, **replace**, and **refuse**.

---

## Generation 1: it learned to copy

Untrained, the small model scored **−0.028**. Negative. Its notes made the Reader *worse than having no notes at all* — it wrote down the small talk and the neighbor's dog, and dropped the actual facts.

After training with reinforcement learning (reward = Lift), it jumped to **+0.486**. Big win!

Except I read its notebooks. It had learned to **copy everything, word for word**. It scored well only because the notebook happened to be *just* big enough to fit. It hadn't learned to choose anything.

This is the first lesson of the whole project: **a model learns the laziest strategy that still gets rewarded.** Not the elegant one you imagined.

## Generation 2: it learned to compress

So I made copying impossible. Six sessions instead of four, more noise, and a notebook so small the full conversation overflows it **3.6×**.

The old copy-everything model collapsed from +0.486 to **+0.100** — its notebooks got cut off mid-sentence, losing the facts at the end. Diagnosis confirmed.

The newly trained model scored **+0.367**, and reading its notes showed a genuinely new skill: **compression**. It stopped writing sentences and started writing dense pairs:

```
Ravi: Cat. Berlin: Living. Lactose Allergy: Present.
Matcha: Not. Cocoa: Present.
```

221 characters. Fit every time. Real skill.

But it still wrote down the neighbor's dog. Every single time. It fit more into the budget — it still wasn't *choosing*.

## Generation 3: it learned to choose

So I made junk expensive. Two traps per conversation instead of one, and trap/update questions now counted **double** in the reward. I also warm-started training from the compression model, so it refined that skill instead of relearning it.

Same unseen conversations, same grader:

| Model | Lift |
|---|---|
| **Generation 3 (selection)** | **+0.375** |
| Generation 2 (compression) | −0.023 |
| Untrained | −0.182 |

The compression champion scores *below zero* once junk is priced properly. The new model wins clearly.

And this time the behavior changed, not just the score. Counted across every notebook it wrote:

| Behaviour | Gen 2 | **Gen 3** |
|---|---|---|
| Wrote down the trap (other people's details) | 13 / 16 | **3 / 16** |
| Kept the outdated value after a change | 9 / 16 | **2 / 16** |
| Trap questions answered right | 19% | **81%** |
| Update questions answered right | 19% | **75%** |

It also paid a price: plain fact recall slipped from 75% to 62%. Being choosy sometimes means dropping something you needed. An honest trade, and worth it here.

Three generations: **retention → compression → selection.** Every shortcut was caught by *reading the notebooks*, never by looking at the score. And every fix was a redesigned task, not a better prompt. You don't ask a model for judgment. You build a world where every shortcut loses.

---

## But did it actually work — or did it just please my grader?

Fair question. My grader was a simple checker program. Maybe the model learned to satisfy *that*, not to write genuinely useful notes.

So I took the exact same notebooks — frozen, already written — and handed them to a real AI (Grok) to answer the questions instead.

| Model | My checker | **Real AI reader** |
|---|---|---|
| Gen 3 (selection) | +0.375 | **+0.352** |
| Gen 2 (compression) | −0.023 | +0.011 |
| Untrained | −0.182 | +0.091 |

The trained result barely moved. With those notebooks, the real reader answered questions about the user at **72% accuracy versus 36% with no notes**.

One detail I found genuinely interesting: the *untrained* model looks better under a real reader (−0.18 → +0.09). Why? A smart reader partially rescues bad notes — it reads "my cousin's dog Kabir" and reasons *that's not the user's dog*, refusing a trap my checker fell for. Which means my headline number is the **conservative** one.

---

## The real experiment: does the skill move?

Here's the question I actually cared about. The model learned all this on **personal chat** — pets, cities, allergies, "my neighbor's dog."

What happens in a world it has never seen?

I built a second generator: **workplace standup talk.** Projects, deadlines, clients, budgets. Different words, different sentence shapes, different traps — now it's *"the platform team's manager Lena"* and *"procurement's vendor Acme."* No retraining. Just drop the model in.

| Model | At home (personal) | **At work (never seen)** |
|---|---|---|
| Gen 3 (selection) | +0.375 | **+0.159** |
| Gen 2 (compression) | −0.023 | **+0.182** |
| Untrained | −0.182 | **−0.159** |

At first glance: a big drop, and the two trained models basically tie. But the notebook counts tell you exactly *what* broke.

**What transferred:**

- **Structured note-taking.** Both trained models stay clearly positive in a world they never saw; the untrained one stays hopeless. Training taught something general.
- **Replacing outdated values.** At home, Gen 3 kept stale values only 2 times in 16. At work — a brand-new domain — still just **3 in 16**. This skill moved cleanly.

**What did not transfer:**

- **Refusing other people's details.** At home: 3 traps written out of 16. At work: **10 out of 16.**

That's the finding. Its filter hadn't learned the *idea* of "this belongs to someone else." It had learned the *sound* of it — "my neighbor's…", "my cousin's…". Rephrase the same trap as *"the platform team's manager"* and it walks straight past.

So a learned memory skill isn't one thing. It's a bundle, and the parts have different strength:

> **Structure and updating are portable skills. Attribution filtering — at this size, trained on one domain — is a domain habit wearing a skill's clothes.**

The obvious fix is training on mixed domains. I haven't run it. It's the next experiment, and I'd rather say that than pretend.

---

## What I'm not claiming

This matters more than any result above.

- **Small scale.** 8–12 conversations per test. Directional, not statistically heavyweight.
- **My own tasks.** Synthetic conversations from generators I wrote. Real chat is messier.
- **One model size.** 0.6B. Published RL-memory work runs at 3B–14B, and my transfer failure may partly be a capacity limit.
- **No public benchmark yet.** The field has standard tests (LongMemEval, LoCoMo) and production systems report ~92–94 on them. **My numbers are not comparable to those**, because I haven't run them. That's a real gap, and it's next on my list.

What I *do* claim: inside those limits, every number has raw data behind it — every conversation, every notebook, every answer is stored and clickable.

---

## Four things I'd tell anyone doing this

**1. Measure before you train.** My lab refuses to train if the starting score is already too high — and that gate saved me. My first memory task looked great until I measured: the untrained model already scored 0.905, because that task only required copying. There was nothing to teach. I rebuilt it.

**2. Read the outputs, not the scores.** Every real discovery here came from opening notebooks. Scores told me *that* something changed. Only the notebooks told me *what* — and twice, the answer was "it's cheating."

**3. The final checkpoint isn't always the best.** One run climbed to a great score, then collapsed to zero. Saving snapshots every few steps let me rescue the peak.

**4. Distrust your own good news.** At one point an *untrained* model scored a suspiciously perfect result. The cause: a leftover server on an old port was quietly serving the trained model instead. I deleted those rows and now every test asks the server "who are you?" before running. The result you love is the one to check hardest.

---

## Where this leaves me

I set out to answer whether note-taking can be trained. Yes — measurably, on free hardware, in three generations of a curriculum where each generation blocked the previous one's shortcut.

But the more useful answer is the one I didn't expect: **when you train a memory skill, you're training several skills at once, and they don't travel equally.** Keeping and updating facts came along for the ride. Knowing whose facts they were stayed home.

If you're building memory into an agent, that distinction is worth knowing before you assume a fine-tuned memory model will behave the same way outside its training data.

Code, raw data, and the full technical write-up: **github.com/khwahish1509/RLPost**

*Everything ran on a laptop and free cloud GPUs. Total cost: about $5 of API credit for verification.*
