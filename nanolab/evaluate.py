"""Phase 2 — the evaluation station.

Contract (from PLAN.md):
- async rollout runner over verifiers' env interface, against any
  OpenAI-compatible endpoint (model strings may be `base:adapter`, resolved
  via serve.py from Phase 5 on);
- rate-limit pacing + retries — throttling is normal, not an error;
- on-disk response cache keyed by hash(env, task, model, params);
- resume — an interrupted eval continues where it stopped (the samples
  table's UNIQUE(eval_run_id, example_index, rollout_index) is the resume key);
- rubric scoring writes samples + eval_runs rows and ledger tokens.

THE ANCHOR: on identical config (env, model, seed, n), results must match
`vf-eval`. Re-run that check after any refactor of this file.
"""

from __future__ import annotations


def run_eval(*args, **kwargs):
    raise NotImplementedError("Phase 2 — see PLAN.md")
