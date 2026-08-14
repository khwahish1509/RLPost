# scribe-stream

**The minimal trainable-memory environment.** A frozen, amnesiac Player
solves chains of arithmetic tasks where later tasks need figures revealed
only by earlier ones. The policy — the *Scribe* — can do exactly one thing:
rewrite a hard-capped notebook between tasks. Reward = **Lift**: the
Player's score with the notebook minus without it.

Deterministic, offline, $0: the default Player is a code checker, so the
whole reward loop runs with no API calls — ideal for RL bring-up and for
studying memory mechanics in isolation before moving to language
(see the companion environment `memory-stream`).

## Mechanics

- One episode = one stream of N dependent tasks (seeded, fully
  deterministic; eval seeds disjoint from train by construction)
- After each task the Scribe sees the outcome + a RECORD line revealing the
  task's figure, and must reply with the full new notebook (truncated in
  code at the cap)
- The Player attempts each task statelessly with ONLY the notebook

Anti-cheat trio: the cap kills log-dumping · held-out seeds kill
memorization · the frozen Player kills "the model just got better."

## Hard mode: selection under budget

The default stream is transcription-easy (an untrained 0.6B already scores
0.905 — measured, and the reason the knobs below exist). Hard mode buries
each needed figure among one-off distractors under a binding cap, so
copying everything overflows and loses what mattered:

| arg | default | effect |
|---|---|---|
| `num_tasks` | 8 | horizon (labels stay unique at any length) |
| `distractors_per_task` | 0 | junk figures per RECORD |
| `mark_reuse` | false | tag lines `(needed later)` / `(one-off)` |
| `notebook_char_cap` | 6000 | the budget — set ~400 to make it bind |
| `player_model` | `"fake"` | code checker, or any OpenAI-compatible endpoint |

## Results achieved with this environment

- Untrained Qwen3-0.6B on hard mode (3 distractors, cap 400): **0.548**
- After GRPO + LoRA on a free Kaggle T4: **1.000 on 12/12 held-out
  streams** — final notebook 189 chars, zero distractor lines
- Ablations it never trained on: hints removed **1.000**, 5 distractors
  **1.000**, 12 tasks **1.000** — while the untrained baseline degrades
  (0.54 → 0.52 → 0.44). It learned selection by content, not label-copying.

Full training story and receipts: **github.com/khwahish1509/RLPost**

## Quickstart

```python
from verifiers import load_environment
env = load_environment("scribe-stream")  # easy mode
env = load_environment("scribe-stream",  # trainable hard mode
    distractors_per_task=3, mark_reuse=True, notebook_char_cap=400)
```
