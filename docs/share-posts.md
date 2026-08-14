# Share posts — paste these as-is

Replace `PASTE_DEVTO_URL` with your Dev.to article URL.

This is two projects, not one:
1. A training lab — Prime Intellect's loop, self-hosted
2. A memory experiment — Letta's idea, trained in that lab

---

## LinkedIn (one post)

Copy everything between the lines.

---

I didn't just train a memory model. I built the training loop first — then used it.

Two weeks. One laptop. Free GPUs. About $5. Every number is from a real run.

**Part 1 — the training lab**

I rebuilt Prime Intellect's product loop on my own machines: environments → eval → train → serve → eval again. One CLI. One database. No cloud product.

Receipts, not slides:
• The measuring stick matches the standard tool: 0.875 = 0.875, every question, every decimal
• A 0.6B model on school math it had never seen: 27/64 → 36/64 (last save of the run, not a picked peak)
• The trained adapter serves on my Mac as a normal API. The lab can measure its own product.

That was the machine. It was never the point. Here's what it was for.

**Part 2 — Letta-style memory, trained**

Letta's bet: deciding what to remember is a skill you can teach, not a database you bolt on.

So I trained a tiny 0.6B "Scribe." Its only job: rewrite a small notebook after each chat. A frozen reader later answers from the notebook alone. If the notes help, the Scribe gets paid.

It cheated twice. I caught both by reading notebooks, not scores.

Round 1: score jumped to +0.49. It had copied the whole conversation word for word.
Round 2: I made the notebook too small to copy. It learned real compression. It still wrote down the neighbor's dog every time (not the user's dog).
Round 3: I made junk expensive. Traps written 13/16 → 3/16. Old values kept 9/16 → 2/16.

Then I dropped it into workplace chat it had never seen. No retraining.

What transferred: structured notes, and updating old facts.
What did not: whose facts they were (3/16 traps at home → 10/16 at work).

Two different things, one laptop: a working RL loop, and a memory skill that only half-survived a new world.

Full story: PASTE_DEVTO_URL
Code: https://github.com/khwahish1509/RLPost

---

## X — post this first (single tweet)

---

I built two things on a laptop (~$5):

1. Prime Intellect's RL loop, self-hosted: env → eval → train → serve → eval again.

2. Letta-style memory: a 0.6B model that learned what to remember. It cheated first.

PASTE_DEVTO_URL

---

## X — full thread (reply to that tweet, one box at a time)

---

2/
Part 1 is a training project, not a memory demo.

Prime Intellect sells: pick a task, measure a model, train, serve, measure again.

I ran that whole loop on my machines. One CLI. One database. No hosted product.

---

3/
Receipts from real runs:

Ruler check: standard tool 0.875, my lab 0.875. Same questions. Every decimal.

School math, 64 new questions: 27 right → 36 right. Last save, not a picked peak.

Trained adapter served on my Mac. Lab measured its own product.

---

4/
That was week one. The loop was the machine.

Part 2 is Letta's idea: memory is a skill you train, not a database you search.

I pointed the same trainer at a tiny Scribe. It can only rewrite a capped notebook. A frozen reader scores the notes.

---

5/
It cheated.

Round 1: +0.49. I read the notebooks. Copy-paste of the whole chat.

Round 2: notebook too small to copy. Real compression. Still wrote the neighbor's dog every time.

---

6/
Round 3: I priced junk. Traps 13/16 → 3/16. Stale facts 9/16 → 2/16.

Then a world it never saw (work chat, no retraining).

Notes + updates transferred. "Whose fact is this?" did not (3/16 traps → 10/16).

---

7/
Two projects:

A self-hosted PI training loop that actually moves a model.

A Letta-style memory skill that only half-transfers out of domain.

Story: PASTE_DEVTO_URL
Code: github.com/khwahish1509/RLPost

---
