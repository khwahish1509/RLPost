# memory-stream

**Train a model to decide what's worth remembering.** Multi-session user
dialogues → a hard-capped notebook the policy rewrites after each session →
a frozen reader answers held-out questions from the notebook alone.

Reward = **Lift**: reader accuracy with the notebook minus without it.
Useless notes score zero; misleading notes score negative.

## Why this environment is hard (and cheat-resistant)

Each stream weaves four ingredient types into natural chat:

- **Stable facts** — *"we adopted a cat — named him Ravi"* → asked later
- **Updated facts** — *"I live in Pune now, not Berlin"* → only the latest
  value counts; an append-only notebook **fails** update questions
- **Chatter** — one-off noise that wastes budget
- **Attribution traps** — *other people's* details (*"my neighbor's dog
  Anya…"*); the matching question's correct answer is *unknown*, so a
  notebook that keeps the trap makes the reader confidently wrong

Anti-cheat trio: a character cap enforced in code (copying everything
overflows and truncates), held-out eval seeds (disjoint generator ranges),
and a frozen reader (improvement can only come from the notes).

## Knobs (env args)

| arg | default | effect |
|---|---|---|
| `num_sessions` | 4 | stream length |
| `facts_per_stream` / `updates_per_stream` | 5 / 1 | fact & update load |
| `distractors_per_session` | 3 | noise level |
| `abstentions_per_stream` | 1 | recurring traps (dog/boss/bike) |
| `notebook_char_cap` | 600 | the budget — make it bind |
| `question_weights` | all 1.0 | e.g. `{abstention: 2, update: 2}` prices selection failures |
| `domain` | `"personal"` | `"work"` = held-out standup-style world for transfer tests |
| `reader_model` | `"fake"` | deterministic offline reader ($0 training) or any OpenAI-compatible endpoint |

## Results achieved with this environment

Trained Qwen3-0.6B (GRPO + LoRA, free Kaggle T4), three curriculum
generations — each blocking the previous run's shortcut:

| generation | curriculum | held-out Lift | learned strategy |
|---|---|---|---|
| 1 | loose cap | −0.03 → **+0.49** | verbatim copying (caught by reading notebooks) |
| 2 | cap binds 3.6× | **+0.37** (copier collapses to +0.10) | compression — dense `Key: Value` notes |
| 3 | traps recur, selection priced 2× | **+0.38** (gen-2 scores −0.02 here) | **selection**: traps written 13/16 → 3/16, stale values kept 9/16 → 2/16 |

Verified by re-grading frozen notebooks with a real LLM reader (+0.35), and
probed for transfer with `domain="work"`: structured retention and
update-replacement transfer; attribution filtering does not — it keys on
surface cues at this scale. Full story, receipts and blog:
**github.com/khwahish1509/RLPost**

## Quickstart

```python
from verifiers import load_environment
env = load_environment("memory-stream")           # offline, $0
# or with selection pricing:
env = load_environment("memory-stream",
    num_sessions=6, distractors_per_session=4, updates_per_stream=2,
    abstentions_per_stream=2, notebook_char_cap=350,
    question_weights={"abstention": 2.0, "update": 2.0})
```

Open problems this environment is built to study: ownership/attribution
tracking (add "whose X?" questions), mixed-domain generalization, and the
memory-quality-per-token frontier.
