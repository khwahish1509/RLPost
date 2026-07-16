"""The evaluation station: rollouts + scoring, persisted to the lab db.

Design rule (THE ANCHOR): nanolab must reproduce `vf-eval`'s numbers on an
identical config. We guarantee that by construction — this module builds the
same EvalConfig that vf-eval's own CLI builds (using verifiers' helpers for
env defaults, sampling-arg merging, and resume detection) and executes it with
verifiers' own `run_evaluation`. nanolab adds the layers around that call:

- a run-level cache: an identical completed config is served from the db
  instantly (`--force` bypasses);
- resume: verifiers-native — `--resume` finds the newest incomplete results
  dir for the config and continues where it stopped;
- persistence: every rollout lands in `samples`, aggregates in `eval_runs`,
  token usage in `ledger`;
- pacing defaults tuned for free-tier endpoints (low concurrency, patient
  retries) — throttling is normal, not an error. Pass explicit values when
  running anchor comparisons so both sides match.

Re-run the anchor check after any change to this file.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from . import db, envs, ledger

# nanolab's own pacing defaults (deliberately gentler than vf-eval's 32/3).
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_MAX_RETRIES = 10


class EvalError(RuntimeError):
    pass


@dataclass
class EvalSummary:
    run_id: int
    env: str
    model: str
    status: str
    mean_reward: float | None
    num_samples: int
    num_errors: int
    avg_metrics: dict[str, float]
    results_path: str | None
    cached: bool


def _params_json(
    *,
    base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    shuffle_seed: int | None,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    """Canonical config fingerprint used by the run-level cache."""
    return json.dumps(
        {
            "base_url": base_url,
            "num_examples": num_examples,
            "rollouts_per_example": rollouts_per_example,
            "shuffle_seed": shuffle_seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    )


def _build_config(
    *,
    env_id: str,
    model: str,
    api_base_url: str,
    api_key_var: str,
    num_examples: int,
    rollouts_per_example: int,
    shuffle_seed: int | None,
    temperature: float | None,
    max_tokens: int | None,
    max_concurrent: int,
    max_retries: int,
    resume: bool,
):
    """Mirror vf-eval's build_eval_config for the direct-endpoint path."""
    from verifiers.scripts.eval import (
        DEFAULT_ENV_DIR_PATH,
        merge_sampling_args,
    )
    from verifiers.types import ClientConfig, EvalConfig
    from verifiers.utils.path_utils import find_latest_incomplete_eval_results_path

    shuffle = shuffle_seed is not None
    sampling_args = merge_sampling_args(
        None,
        max_tokens=max_tokens,
        temperature=temperature,
        include_none_max_tokens=True,
    )
    client_config = ClientConfig(
        client_type="openai_chat_completions",
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        # vf-eval's default: sticky session header per trajectory
        extra_headers_from_state={"X-Session-ID": "trajectory_id"},
    )
    resume_path = None
    if resume:
        resume_path = find_latest_incomplete_eval_results_path(
            env_id=env_id,
            model=model,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            env_dir_path=DEFAULT_ENV_DIR_PATH,
            output_dir="results",
        )
    return EvalConfig(
        env_id=env_id,
        env_args={},
        env_dir_path=DEFAULT_ENV_DIR_PATH,
        output_dir="results",
        model=model,
        client_config=client_config,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        disable_tui=True,
        save_results=True,
        resume_path=resume_path,
    )


def resolve_counts(env_id: str, num_examples: int | None, rollouts: int | None) -> tuple[int, int]:
    """vf-eval semantics: unset counts fall back to the env's own defaults."""
    if num_examples is not None and rollouts is not None:
        return int(num_examples), int(rollouts)

    from verifiers.scripts.eval import (
        DEFAULT_NUM_EXAMPLES,
        DEFAULT_ROLLOUTS_PER_EXAMPLE,
        get_env_eval_defaults,
    )

    defaults = get_env_eval_defaults(env_id)
    n = num_examples if num_examples is not None else defaults.get(
        "num_examples", DEFAULT_NUM_EXAMPLES
    )
    r = rollouts if rollouts is not None else defaults.get(
        "rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE
    )
    return int(n), int(r)


def persist_outputs(conn, run_id: int, outputs: dict[str, Any]) -> tuple[int, int]:
    """Write RolloutOutputs into samples + ledger; returns (n_samples, n_errors)."""
    rollout_counter: dict[Any, int] = {}
    n_errors = 0
    total_in = 0
    total_out = 0
    metadata = outputs.get("metadata", {})
    model = metadata.get("model", "?")

    for rollout in outputs.get("outputs", []):
        example_id = rollout.get("example_id")
        idx = rollout_counter.get(example_id, 0)
        rollout_counter[example_id] = idx + 1
        error = rollout.get("error")
        if error:
            n_errors += 1
        usage = rollout.get("token_usage") or {}
        total_in += int(usage.get("input_tokens") or 0)
        total_out += int(usage.get("output_tokens") or 0)
        metrics = dict(rollout.get("metrics") or {})
        if rollout.get("stop_condition"):
            metrics["_stop_condition"] = rollout["stop_condition"]
        if error:
            metrics["_error"] = error.get("error", "error")
        conn.execute(
            """
            INSERT OR REPLACE INTO samples
                (eval_run_id, example_index, rollout_index, prompt_json,
                 completion_json, reward, metrics_json, from_cache, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                run_id,
                example_id,
                idx,
                json.dumps(rollout.get("prompt")),
                json.dumps(rollout.get("completion")),
                rollout.get("reward"),
                json.dumps(metrics),
                db.utcnow(),
            ),
        )
    conn.commit()
    n_samples = sum(rollout_counter.values())
    if total_in or total_out:
        ledger.record(conn, "eval", run_id, model, total_in, total_out)
    return n_samples, n_errors


def _find_cached_run(conn, env_row_id: int, model: str, params: str):
    return conn.execute(
        """
        SELECT * FROM eval_runs
        WHERE env_id = ? AND model = ? AND params_json = ? AND status = 'done'
        ORDER BY id DESC LIMIT 1
        """,
        (env_row_id, model, params),
    ).fetchone()


def _summary_from_row(conn, row, cached: bool) -> EvalSummary:
    counts = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN metrics_json LIKE '%"_error"%' THEN 1 ELSE 0 END) AS errs
        FROM samples WHERE eval_run_id = ?
        """,
        (row["id"],),
    ).fetchone()
    env_slug = conn.execute(
        "SELECT slug FROM environments WHERE id = ?", (row["env_id"],)
    ).fetchone()["slug"]
    meta = json.loads(row["metrics_json"] or "{}")
    return EvalSummary(
        run_id=row["id"],
        env=env_slug,
        model=row["model"],
        status=row["status"],
        mean_reward=row["mean_reward"],
        num_samples=counts["n"] or 0,
        num_errors=counts["errs"] or 0,
        avg_metrics=meta.get("avg_metrics", {}),
        results_path=meta.get("results_path"),
        cached=cached,
    )


def run(
    env_ref: str,
    model: str,
    *,
    api_base_url: str,
    api_key_var: str,
    num_examples: int | None = None,
    rollouts_per_example: int | None = None,
    shuffle_seed: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    resume: bool = False,
    force: bool = False,
) -> EvalSummary:
    # base:adapter model strings resolve to a live local deployment
    from . import serve

    try:
        resolved = serve.resolve_model(model)
    except serve.ServeError as exc:
        raise EvalError(str(exc)) from exc
    if resolved is not None:
        api_base_url, served_name, api_key_var = resolved
        # the run is recorded under the base:adapter name; requests use the
        # deployment's served model name
        served_model = served_name
    else:
        served_model = model

    conn = db.connect()
    try:
        env_row = db.get_environment(conn, env_ref)
        if env_row is None:
            raise EvalError(
                f"Environment {env_ref!r} is not installed. "
                f"Try: nanolab env install {env_ref}"
            )
        env_id = env_row["env_id"]
        n, r = resolve_counts(env_id, num_examples, rollouts_per_example)
        params = _params_json(
            base_url=api_base_url,
            num_examples=n,
            rollouts_per_example=r,
            shuffle_seed=shuffle_seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not force and not resume:
            cached_row = _find_cached_run(conn, env_row["id"], model, params)
            if cached_row is not None:
                return _summary_from_row(conn, cached_row, cached=True)

        config = _build_config(
            env_id=env_id,
            model=served_model,
            api_base_url=api_base_url,
            api_key_var=api_key_var,
            num_examples=n,
            rollouts_per_example=r,
            shuffle_seed=shuffle_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            resume=resume,
        )

        cur = conn.execute(
            """
            INSERT INTO eval_runs
                (env_id, model, endpoint, num_examples, rollouts_per_example,
                 seed, params_json, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (env_row["id"], model, api_base_url, n, r, shuffle_seed, params, db.utcnow()),
        )
        conn.commit()
        run_id = int(cur.lastrowid)

        from verifiers.utils.eval_utils import run_evaluation

        try:
            outputs = asyncio.run(run_evaluation(config))
        except BaseException:
            conn.execute(
                "UPDATE eval_runs SET status = 'failed', finished_at = ? WHERE id = ?",
                (db.utcnow(), run_id),
            )
            conn.commit()
            raise

        metadata = outputs.get("metadata", {})
        n_samples, n_errors = persist_outputs(conn, run_id, outputs)
        meta_blob = {
            "avg_metrics": metadata.get("avg_metrics", {}),
            "avg_error": metadata.get("avg_error"),
            "pass_at_k": metadata.get("pass_at_k", {}),
            "time": metadata.get("time"),
            "usage": metadata.get("usage"),
            "results_path": str(metadata.get("path_to_save", "")) or None,
        }
        conn.execute(
            """
            UPDATE eval_runs
            SET status = 'done', mean_reward = ?, metrics_json = ?, finished_at = ?
            WHERE id = ?
            """,
            (metadata.get("avg_reward"), json.dumps(meta_blob), db.utcnow(), run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return _summary_from_row(conn, row, cached=False)
    finally:
        conn.close()


def show(run_id: int) -> dict[str, Any]:
    """Full per-metric breakdown for one run, computed from stored samples."""
    import statistics

    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise EvalError(f"No eval run with id {run_id}")
        env_slug = conn.execute(
            "SELECT slug FROM environments WHERE id = ?", (row["env_id"],)
        ).fetchone()["slug"]
        samples = conn.execute(
            "SELECT reward, metrics_json FROM samples WHERE eval_run_id = ? "
            "ORDER BY example_index, rollout_index",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    rewards = [s["reward"] for s in samples if s["reward"] is not None]
    per_metric: dict[str, list[float]] = {}
    stop_conditions: dict[str, int] = {}
    errors: dict[str, int] = {}
    for s in samples:
        metrics = json.loads(s["metrics_json"] or "{}")
        for key, value in metrics.items():
            if key == "_stop_condition":
                stop_conditions[value] = stop_conditions.get(value, 0) + 1
            elif key == "_error":
                errors[value] = errors.get(value, 0) + 1
            elif isinstance(value, (int, float)):
                per_metric.setdefault(key, []).append(float(value))

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "avg": statistics.mean(values) if values else 0.0,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    return {
        "run_id": run_id,
        "env": env_slug,
        "model": row["model"],
        "status": row["status"],
        "num_examples": row["num_examples"],
        "rollouts_per_example": row["rollouts_per_example"],
        "seed": row["seed"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "mean_reward": row["mean_reward"],
        "reward": stats(rewards),
        "rewards": rewards,
        "metrics": {k: stats(v) for k, v in sorted(per_metric.items())},
        "stop_conditions": stop_conditions,
        "errors": errors,
        "meta": json.loads(row["metrics_json"] or "{}"),
    }
