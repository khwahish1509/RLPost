# nanolab

**Train language models with reinforcement learning — on your own machine, end to end.**

nanolab is a self-hosted RL lab that closes the whole loop, small enough to understand in one sitting:

```
Environments ──▶ Evaluations ──▶ Training (GRPO + LoRA) ──▶ Inference ──▶ re-eval
     ▲                                                                      │
     └────────────────────── one CLI · one SQLite file ◀────────────────────┘
```

- **Environments** are standard [verifiers](https://github.com/PrimeIntellect-ai/verifiers) modules — any environment on the public Environments Hub runs here unmodified.
- **Evaluations** run rollouts against any OpenAI-compatible endpoint, with response caching, rate-limit pacing, and resume built in.
- **Training** is a deliberately synchronous GRPO + LoRA loop — a for-loop you can read, checkpointed every 10 steps.
- **Inference** serves trained adapters (vLLM `--enable-lora`; llama.cpp as the laptop path) so the lab can re-measure its own output.
- Everything lands in **one SQLite file**: runs, samples, reward curves, adapters, token ledger.

## Quickstart

```bash
uv sync
uv run nanolab env install primeintellect/alphabet-sort
uv run nanolab env list
```

## The CLI

| Verb | What it does | Status |
|---|---|---|
| `nanolab env` | install / list environments | ✅ working |
| `nanolab eval` | run rollouts, score with rubrics, inspect runs | Phase 2 |
| `nanolab train` | GRPO+LoRA from a TOML config, resumable | Phase 3 |
| `nanolab deployments` | serve adapters, `base:adapter` model strings | Phase 5 |
| `nanolab report` | static leaderboard.html from the db | Phase 2 |

The full build plan — phase by phase, with explicit "done when" checks — is in [PLAN.md](PLAN.md).

## Design principles

1. **Synchronous over clever.** No orchestrators. At single-user scale a for-loop is correct.
2. **One scalar reward.** Every environment reduces to one number, or it isn't ready.
3. **Cache and resume everything.** API responses, rollouts, checkpoints — assume every long process gets killed.
4. **Anchored numbers.** The eval runner must reproduce `vf-eval`'s results on identical configs; that check re-runs after every refactor of the rollout path.
5. **Every phase ends in an artifact** — a table, a curve, an adapter, a page. Not a feeling.

## The capstone: the Scribe

Once the loop closes, it trains its first interesting tenant: a small model whose only ability is writing notes. A frozen Player model works through a stream of related tasks; the **Scribe** maintains a token-capped notebook between tasks; the Scribe's reward is **Lift** — how much its notes improve the Player on future, held-out tasks. If it works, a small model trained in this lab measurably out-teaches its untrained self. See Phase 7 in [PLAN.md](PLAN.md).

## Layout

```
├── nanolab/          # cli, db (SQLite), envs, evaluate, train, serve, ledger, report
├── configs/          # training TOMLs
├── tests/            # network-free smoke tests (CI runs these)
└── results/          # gitignored artifacts: the db, eval tables, leaderboard.html
```

## Status

Phase 1 of 7 complete — the environment station works against the live Hub; eval, training, and serving land next. [PLAN.md](PLAN.md) doubles as the progress tracker.

## License

MIT
