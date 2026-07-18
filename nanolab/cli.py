"""The nanolab CLI: env · eval · train · deployments · report.

Phase 1: `env` is real; the other verbs are stubs that name the phase that
implements them, so the CLI surface is stable from day one.
"""

from __future__ import annotations

import os

import typer

from . import __version__, config, db, envs

app = typer.Typer(
    name="nanolab",
    help="A self-hosted RL lab: env → eval → train → serve → re-eval.",
    no_args_is_help=True,
)

env_app = typer.Typer(help="Install and list verifiers environments.", no_args_is_help=True)
eval_app = typer.Typer(help="Run and inspect evaluations.", no_args_is_help=True)
training_app = typer.Typer(help="Inspect training runs and reward curves.", no_args_is_help=True)
deploy_app = typer.Typer(help="Serve trained adapters.", no_args_is_help=True)

app.add_typer(env_app, name="env")
app.add_typer(eval_app, name="eval")
app.add_typer(training_app, name="training")
app.add_typer(deploy_app, name="deployments")

@app.callback()
def _main() -> None:
    """nanolab — one CLI, one SQLite file, one closed RL loop."""
    config.load_dotenv()


@app.command()
def version() -> None:
    """Print the nanolab version."""
    typer.echo(__version__)


# ── env ──────────────────────────────────────────────────────────────────────


@env_app.command("install")
def env_install(slug: str = typer.Argument(help="owner/name[@version] or local env name")) -> None:
    """Install a verifiers environment (Prime Hub compatible) and register it."""
    try:
        installed = envs.install(slug)
    except envs.EnvInstallError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"installed {installed.slug} (env id: {installed.env_id}, "
        f"version: {installed.version or '?'})",
        fg="green",
    )


@env_app.command("list")
def env_list() -> None:
    """List environments registered in the lab db."""
    rows = envs.list_installed()
    if not rows:
        typer.echo("no environments installed — try: nanolab env install primeintellect/alphabet-sort")
        return
    width = max(len(r["slug"]) for r in rows)
    for r in rows:
        marker = "" if r["importable"] else "  [missing from venv!]"
        typer.echo(
            f"{r['slug']:<{width}}  v{r['version']}  installed {r['installed_at']}{marker}"
        )


# ── eval ─────────────────────────────────────────────────────────────────────


@eval_app.command("run")
def eval_run(
    env: str = typer.Argument(help="Environment id or slug"),
    model: str = typer.Option(
        None, "--model", "-m", help="Model name (default: $NANOLAB_DEFAULT_MODEL)"
    ),
    num_examples: int = typer.Option(
        None, "--num-examples", "-n", help="Examples (default: env's own default)"
    ),
    rollouts: int = typer.Option(
        None, "--rollouts", "-r", help="Rollouts per example (default: env's own default)"
    ),
    seed: int = typer.Option(
        None, "--seed", help="Shuffle-seed; enables dataset shuffling when set"
    ),
    api_base_url: str = typer.Option(
        None, "--api-base-url", "-b", help="OpenAI-compatible endpoint (default: $NANOLAB_API_BASE_URL)"
    ),
    api_key_var: str = typer.Option(
        None, "--api-key-var", "-k", help="Env var holding the API key (default: $NANOLAB_API_KEY_VAR)"
    ),
    max_concurrent: int = typer.Option(None, "--max-concurrent", "-c"),
    max_retries: int = typer.Option(None, "--max-retries"),
    temperature: float = typer.Option(None, "--temperature", "-T"),
    max_tokens: int = typer.Option(None, "--max-tokens", "-t"),
    resume: bool = typer.Option(False, "--resume", help="Continue the newest incomplete run of this config"),
    force: bool = typer.Option(False, "--force", help="Re-run even if an identical config already completed"),
    env_args: str = typer.Option(
        None, "--env-args", "-a", help='Environment arguments as JSON, e.g. \'{"player_model": "fake"}\''
    ),
) -> None:
    """Evaluate a model on an environment (rollouts + rubric scoring)."""
    import json as json_mod

    from . import evaluate

    parsed_env_args = None
    if env_args:
        try:
            parsed_env_args = json_mod.loads(env_args)
        except json_mod.JSONDecodeError as exc:
            typer.secho(f"--env-args is not valid JSON: {exc}", fg="red", err=True)
            raise typer.Exit(1) from exc

    model = model or os.environ.get("NANOLAB_DEFAULT_MODEL")
    api_base_url = api_base_url or os.environ.get("NANOLAB_API_BASE_URL")
    api_key_var = api_key_var or os.environ.get("NANOLAB_API_KEY_VAR", "OPENAI_API_KEY")
    if not model:
        typer.secho("No model: pass -m or set NANOLAB_DEFAULT_MODEL in .env", fg="red", err=True)
        raise typer.Exit(1)
    if not api_base_url:
        typer.secho("No endpoint: pass -b or set NANOLAB_API_BASE_URL in .env", fg="red", err=True)
        raise typer.Exit(1)
    if not os.environ.get(api_key_var):
        typer.secho(f"API key env var {api_key_var} is empty (set it in .env)", fg="red", err=True)
        raise typer.Exit(1)

    try:
        summary = evaluate.run(
            env,
            model,
            api_base_url=api_base_url,
            api_key_var=api_key_var,
            num_examples=num_examples,
            rollouts_per_example=rollouts,
            shuffle_seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent if max_concurrent is not None else evaluate.DEFAULT_MAX_CONCURRENT,
            max_retries=max_retries if max_retries is not None else evaluate.DEFAULT_MAX_RETRIES,
            resume=resume,
            force=force,
            env_args=parsed_env_args,
        )
    except evaluate.EvalError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc

    tag = " (cached — identical config already completed; --force to re-run)" if summary.cached else ""
    reward = f"{summary.mean_reward:.3f}" if summary.mean_reward is not None else "—"
    typer.secho(f"eval run #{summary.run_id}{tag}", fg="green")
    typer.echo(f"  env:     {summary.env}")
    typer.echo(f"  model:   {summary.model}")
    typer.echo(f"  reward:  {reward}  ({summary.num_samples} rollouts, {summary.num_errors} errors)")
    for key, value in sorted(summary.avg_metrics.items()):
        typer.echo(f"  {key}: {value:.3f}")
    if summary.results_path:
        typer.echo(f"  results: {summary.results_path}")
    typer.echo(f"  details: nanolab eval show {summary.run_id}")


@eval_app.command("list")
def eval_list() -> None:
    """List eval runs."""
    conn = db.connect()
    try:
        rows = conn.execute(
            """
            SELECT e.id, v.slug, e.model, e.status, e.mean_reward, e.started_at
            FROM eval_runs e JOIN environments v ON v.id = e.env_id
            ORDER BY e.id DESC
            """
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        typer.echo("no eval runs yet")
        return
    for r in rows:
        reward = f"{r['mean_reward']:.3f}" if r["mean_reward"] is not None else "—"
        typer.echo(f"#{r['id']}  {r['slug']}  {r['model']}  {r['status']}  reward={reward}")


@eval_app.command("show")
def eval_show(run_id: int = typer.Argument(help="Eval run id")) -> None:
    """Per-metric breakdown for one eval run."""
    from . import evaluate

    try:
        detail = evaluate.show(run_id)
    except evaluate.EvalError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc

    typer.secho(f"eval run #{detail['run_id']}  [{detail['status']}]", fg="green")
    typer.echo(f"  env:    {detail['env']}")
    typer.echo(f"  model:  {detail['model']}")
    typer.echo(
        f"  config: n={detail['num_examples']} r={detail['rollouts_per_example']}"
        f" seed={detail['seed']}"
    )
    typer.echo(f"  window: {detail['started_at']} → {detail['finished_at']}")
    r = detail["reward"]
    typer.echo(f"  reward: avg {r['avg']:.3f}  std {r['std']:.3f}  (n={r['n']})")
    if detail["rewards"]:
        typer.echo(f"  values: {[round(x, 3) for x in detail['rewards']]}")
    if detail["metrics"]:
        typer.echo("  metrics:")
        for key, s in detail["metrics"].items():
            typer.echo(f"    {key}: avg {s['avg']:.3f}  std {s['std']:.3f}")
    if detail["stop_conditions"]:
        typer.echo(f"  stop conditions: {detail['stop_conditions']}")
    if detail["errors"]:
        typer.secho(f"  errors: {detail['errors']}", fg="yellow")
    if detail["meta"].get("results_path"):
        typer.echo(f"  results dir: {detail['meta']['results_path']}")


@eval_app.command("compare")
def eval_compare(
    run_a: int = typer.Argument(help="Baseline eval run id (e.g. the base model)"),
    run_b: int = typer.Argument(help="Challenger eval run id (e.g. the adapter)"),
) -> None:
    """Side-by-side comparison of two eval runs (Phase 4: adapter vs base)."""
    from . import evaluate

    try:
        a = evaluate.show(run_a)
        b = evaluate.show(run_b)
    except evaluate.EvalError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc

    if a["env"] != b["env"]:
        typer.secho(
            f"warning: different environments ({a['env']} vs {b['env']}) — "
            "comparison is not apples-to-apples",
            fg="yellow",
        )
    typer.echo(f"env: {a['env']}")
    typer.echo(f"  A  #{a['run_id']}  {a['model']}  n={a['num_examples']} r={a['rollouts_per_example']}")
    typer.echo(f"  B  #{b['run_id']}  {b['model']}  n={b['num_examples']} r={b['rollouts_per_example']}")
    ra, rb = a["reward"], b["reward"]
    delta = rb["avg"] - ra["avg"]
    typer.echo(f"\n  reward A: {ra['avg']:.3f} ± {ra['std']:.3f}   (n={ra['n']})")
    typer.echo(f"  reward B: {rb['avg']:.3f} ± {rb['std']:.3f}   (n={rb['n']})")
    color = "green" if delta > 0 else ("red" if delta < 0 else "yellow")
    typer.secho(f"  Δ (B − A): {delta:+.3f}", fg=color, bold=True)
    shared = sorted(set(a["metrics"]) & set(b["metrics"]))
    if shared:
        typer.echo("\n  per-metric Δ:")
        for key in shared:
            d = b["metrics"][key]["avg"] - a["metrics"][key]["avg"]
            typer.echo(f"    {key}: {a['metrics'][key]['avg']:.3f} → {b['metrics'][key]['avg']:.3f}  ({d:+.3f})")


# ── train ────────────────────────────────────────────────────────────────────


@app.command()
def train(
    config: str = typer.Argument(help="Path to a training TOML (see configs/)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from the last checkpoint"),
) -> None:
    """GRPO+LoRA training from a TOML config (synchronous loop, GPU box)."""
    from . import train as train_mod

    try:
        run_id = train_mod.train(config, resume=resume)
    except train_mod.TrainError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"train run #{run_id} finished — adapters/ has the checkpoints", fg="green")


# ── training runs ────────────────────────────────────────────────────────────


@training_app.command("list")
def training_list() -> None:
    """List training runs (start one with: nanolab train <config.toml>)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            """
            SELECT t.*, v.slug AS env_slug FROM train_runs t
            LEFT JOIN environments v ON v.id = t.env_id
            ORDER BY t.id DESC
            """
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        typer.echo("no training runs yet — start one with: nanolab train configs/<config>.toml")
        return
    import json as json_mod

    for r in rows:
        curve = json_mod.loads(r["reward_curve_json"] or "[]")
        last = f"{curve[-1]['reward']:.3f}" if curve else "—"
        typer.echo(
            f"#{r['id']}  {r['model']}  on {r['env_slug'] or '?'}  "
            f"[{r['status']}]  steps={r['steps_completed']}  last-reward={last}"
        )


@training_app.command("show")
def training_show(run_id: int = typer.Argument(help="Training run id")) -> None:
    """Reward curve + checkpoints for one training run."""
    import json as json_mod

    from .train import sparkline

    conn = db.connect()
    try:
        r = conn.execute(
            """
            SELECT t.*, v.slug AS env_slug FROM train_runs t
            LEFT JOIN environments v ON v.id = t.env_id WHERE t.id = ?
            """,
            (run_id,),
        ).fetchone()
        if r is None:
            typer.secho(f"No training run with id {run_id}", fg="red", err=True)
            raise typer.Exit(1)
        adapters = conn.execute(
            "SELECT * FROM adapters WHERE train_run_id = ? ORDER BY step", (run_id,)
        ).fetchall()
    finally:
        conn.close()

    curve = json_mod.loads(r["reward_curve_json"] or "[]")
    rewards = [p["reward"] for p in curve]
    typer.secho(f"train run #{r['id']}  [{r['status']}]", fg="green")
    typer.echo(f"  model:  {r['model']}")
    typer.echo(f"  env:    {r['env_slug'] or '?'}")
    typer.echo(f"  steps:  {r['steps_completed']}")
    typer.echo(f"  window: {r['started_at']} → {r['finished_at'] or '…'}")
    if rewards:
        typer.echo(f"  curve:  {sparkline(rewards)}")
        typer.echo(
            f"          first {rewards[0]:.3f} → last {rewards[-1]:.3f}"
            f"  (min {min(rewards):.3f}, max {max(rewards):.3f})"
        )
        delta = rewards[-1] - rewards[0]
        color = "green" if delta > 0 else "yellow"
        typer.secho(f"          Δ over run: {delta:+.3f}", fg=color)
    else:
        typer.echo("  curve:  (no steps logged)")
    if adapters:
        typer.echo("  checkpoints:")
        for a in adapters:
            typer.echo(f"    adapter #{a['id']}  step {a['step']}  {a['path']}")
        last = adapters[-1]
        typer.echo(
            f"  evaluate it: nanolab eval run <env> -m {last['base_model']}:{last['id']}"
        )


# ── deployments ──────────────────────────────────────────────────────────────


@deploy_app.command("create")
def deployments_create(
    adapter_id: int = typer.Argument(help="Adapter id from the adapters table"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Serve an adapter through vLLM --enable-lora (CUDA box)."""
    from . import serve

    try:
        dep = serve.create_deployment(adapter_id, port=port)
    except serve.ServeError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"deployment #{dep.id} running (pid {dep.pid})", fg="green")
    typer.echo(f"  endpoint: {dep.endpoint}")
    typer.echo(f"  eval it:  nanolab eval run <env> -m {dep.base_model}:{adapter_id}")


@deploy_app.command("list")
def deployments_list() -> None:
    """List deployments (liveness re-checked)."""
    from . import serve

    deps = serve.list_deployments()
    if not deps:
        typer.echo("no deployments — create one with: nanolab deployments create <adapter-id>")
        return
    for d in deps:
        typer.echo(
            f"#{d.id}  {d.base_model}:{d.adapter_id}  {d.endpoint}  "
            f"pid={d.pid}  [{d.status}]"
        )


@deploy_app.command("stop")
def deployments_stop(deployment_id: int = typer.Argument(help="Deployment id")) -> None:
    """Stop a running deployment."""
    from . import serve

    try:
        dep = serve.stop_deployment(deployment_id)
    except serve.ServeError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"deployment #{dep.id} stopped", fg="green")


# ── report ───────────────────────────────────────────────────────────────────


@app.command()
def ui(
    port: int = typer.Option(3456, "--port", "-p"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser"),
) -> None:
    """Serve the nanolab web app (read-only view over the lab db)."""
    from . import api

    api.serve_ui(port=port, open_browser=not no_open)


@app.command()
def report() -> None:
    """Render the static leaderboard.html from the db."""
    from . import report as report_mod

    path = report_mod.render()
    typer.secho(f"wrote {path}", fg="green")


if __name__ == "__main__":
    app()
