# pattern-gym

**Rule-induction micro-tasks with difficulty as a first-class knob.**

Each task shows three demonstrations of a hidden transformation and one test
input; the model answers with the transformed result in `\boxed{}`, checked by
exact match — verifiable by code, impossible to sweet-talk.

```
A hidden rule transforms inputs into outputs:
  fkzqe -> fkzqee...      (demo 1)
  ...
Apply the same rule to: qmzvb
```

## Why this environment exists

1. **Trainability by design.** GRPO only learns when a model's baseline sits
   in the 10–80% window (mixed groups carry the signal). Most environments
   *hope* to land there; pattern-gym makes it a dial — `easy` (reverse,
   double, +k), `medium` (caesar shifts, letter sorts, digit sums), `hard`
   (two rules composed), `mixed` (40/40/20 blend).
2. **Honest splits.** Train tasks use seeds `0…`, eval tasks `100000…` — held
   out by construction, reproducible from the seed alone.
3. **Rule families for memory research.** Every task belongs to a family
   (`caesar_3`, `sort_letters+reverse`, …). Streams of same-family tasks make
   notes genuinely transferable — the property memory-training rewards.

## Usage

```
nanolab env install pattern-gym                    # local install
nanolab eval run pattern-gym -m <model> -n 20
```

Environment args (`-a`): `difficulty` (easy|medium|hard|mixed),
`num_train_examples`, `num_eval_examples`.

## Metrics

- `correct` — exact match (the reward)
- `has_boxed` — answer format compliance (diagnostic, weight 0)
