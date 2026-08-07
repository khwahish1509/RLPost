# THE ANCHOR — committed evidence

The anchor is nanolab's central correctness claim: **on an identical config, nanolab's
eval station reproduces the reference `vf-eval` tool's numbers to every decimal on every
example.** It holds by construction — `evaluate.py` builds the same `EvalConfig` and
executes verifiers' own `run_evaluation` rather than reimplementing the rollout loop —
and it is re-run after any change to the rollout path (engineering rule 4).

Everything else in this repo is downstream of this check. So it lives in git, not in the
gitignored `results/`.

## The config

| | |
|---|---|
| environment | `alphabet-sort` v0.1.12 (Prime Intellect hub) |
| model | `grok-4.20-0309-non-reasoning` @ `https://api.x.ai/v1` |
| n / r | 10 examples, 1 rollout each |
| temperature | 0.0 |
| max concurrent | 1 |
| verifiers | 0.2.0 (`31dc0a6`) |
| date | 2026-07-16 |

## The result

```
mean reward   0.8745119703540165
stdev         0.2162
errors        0.0
per-example   [0.6095794231167655, 1.0, 1.0, 1.0, 1.0,
               0.7900575643740181, 1.0, 1.0, 0.3454827160493828, 1.0]
```

Full per-example records — reward, weighted_reward, turn count, and SHA-256 digests of
the prompt and completion — are in [`nanolab-side.json`](nanolab-side.json).

Two independent nanolab runs of this config (`results/evals/…/7adacfa1` and `…/3fed93ee`,
eval_runs #2 and #3 in the db) produced byte-identical per-example rewards.

## Provenance — read this before citing the anchor

**What is committed here is the nanolab side only.** The `vf-eval` side of the original
2026-07-16 comparison was observed live in the terminal and matched to every decimal, but
its `results.jsonl` was never written to disk: the run at `outputs/evals/alphabet-sort--
grok-4.20-0309-non-reasoning/78478c0d/` shows its env worker terminating (*"Death pipe
closed — parent is gone"*) before the results file was saved. Only its logs survive.

So the honest statement is: **the nanolab side is receipted here; the `vf-eval` side is
reproducible on demand but was not retained.** Rather than restate a number whose
artifact we don't hold, `scripts/verify_anchor.sh` regenerates both sides and diffs them.

This note exists because the project's whole claim is that unverifiable numbers get
labelled as such.

## Re-running it

```bash
scripts/verify_anchor.sh          # runs both sides, diffs per-example rewards
```

Or by hand:

```bash
# reference side
vf-eval alphabet-sort -k XAI_API_KEY -b https://api.x.ai/v1 \
  -m grok-4.20-0309-non-reasoning -n 10 -r 1 -c 1 -T 0.0 --disable-tui

# nanolab side
nanolab eval run alphabet-sort -m grok-4.20-0309-non-reasoning \
  -n 10 -r 1 -c 1 -T 0.0 --force
```

Pass condition: per-example rewards equal on both sides, to every decimal.

Note the anchor requires a live xAI key and spends real tokens on someone else's API — it
is deliberately not part of the network-free CI suite.
