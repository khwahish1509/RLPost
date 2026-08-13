# Share posts for the blog

## LinkedIn

---

I spent two weeks teaching a tiny AI model what's worth remembering.

It cheated on me twice.

The setup: a 0.6B model (the "Scribe") rewrites a small notebook after each chat session. A second, frozen model answers questions later using ONLY that notebook. If the notes help, the Scribe gets rewarded. That's it — reinforcement learning on note-taking.

Generation 1: the score jumped from negative to +0.49. Success? I read its notebooks. It had learned to copy the entire conversation word for word. Not memory — photocopying.

Generation 2: I shrank the notebook until copying couldn't fit. It learned real compression — dense "Key: Value" notes. But it still wrote down the neighbor's dog's name every single time (a planted trap — the dog isn't the user's).

Generation 3: I made junk expensive in the reward. It finally learned to choose: traps written fell from 13/16 to 3/16, outdated values kept fell from 9/16 to 2/16.

Then the real experiment: I dropped it into a world it had never seen — workplace chat instead of personal chat. No retraining.

Result: half the skill transferred.
✅ Structured notes and updating old facts — traveled perfectly.
❌ Refusing other people's details — collapsed. It had learned the *sound* of personal-life junk ("my neighbor's..."), not the concept.

The lesson for anyone building AI agent memory: a trained memory skill is a bundle, and the parts travel differently. Test which parts before you trust it in a new domain.

Everything ran on my laptop + free cloud GPUs. ~$5 total. Every shortcut was caught by reading actual outputs, never by trusting scores.

Full story with all the numbers (including my own mistakes — there were four): [BLOG LINK]

Code: github.com/khwahish1509/RLPost

---

## X / Twitter thread

---

1/
I trained a 0.6B model to decide what's worth remembering.

It cheated twice. I caught it both times by reading its notebooks.

And when I moved it to a new domain, exactly half the skill transferred.

The whole story, on a laptop + free GPUs (~$5 total): 🧵

2/
The game: a tiny "Scribe" model rewrites a capped notebook after each chat session.

A frozen Reader later answers questions using ONLY the notebook.

Reward = how much the notes help the Reader vs no notes at all.

Untrained score: −0.03. Its notes were worse than nothing.

3/
Training round 1: score jumps to +0.49 🎉

Then I read the notebooks.

It had learned to copy the ENTIRE conversation verbatim. It scored well only because the notebook barely fit.

Models learn the laziest strategy that still gets rewarded. Always.

4/
Round 2: I shrank the notebook until copying overflows 3.6×.

The copier collapsed to +0.10.

The retrained model hit +0.37 with a genuinely new skill — compression:

"Ravi: Cat. Berlin: Living. Matcha: Not. Cocoa: Present."

But it still wrote down the neighbor's dog. Every time.

5/
Round 3: I planted 2 traps per conversation and made trap/update questions count double in the reward.

It finally learned to CHOOSE:
• traps written: 13/16 → 3/16
• stale values kept: 9/16 → 2/16
• trap questions correct: 19% → 81%

Retention → compression → selection.

6/
"But maybe it just learned to please your grader?"

Fair. So I took the same frozen notebooks and had a real LLM (Grok) answer instead of my checker:

+0.375 → +0.352. Barely moved.

Reader accuracy with its notes: 72% vs 36% without.

7/
Now the real experiment. Everything was learned on PERSONAL chat (pets, cities, neighbors).

I dropped it into workplace standup talk — projects, deadlines, clients. Zero retraining.

Score fell from +0.375 to +0.159. But the notebooks show exactly WHAT broke:

8/
✅ Transferred: structured notes, and replacing outdated values (stale kept: 2/16 home, 3/16 at work — clean).

❌ Did NOT transfer: refusing other people's details (3/16 → 10/16 traps written).

It learned the SOUND of junk ("my neighbor's..."), not the concept.

9/
The takeaway for anyone building agent memory:

A learned memory skill is a bundle. The parts travel differently.

Structure & updating = portable skills.
Attribution filtering (at 0.6B, one domain) = a domain habit wearing a skill's clothes.

10/
Honest limits: n=8–12 per test, my own synthetic tasks, one model size, no public benchmark yet (that's next).

And the lab caught ME being wrong 4 times — too-easy task, one-regime gain, wrong statistic, wrong server answering.

Full story: [BLOG LINK]
Code: github.com/khwahish1509/RLPost

---
