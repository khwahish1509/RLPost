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
deploy_app = typer.Typer(help="Serve trained adapters.", no_args_is_help=True)

app.add_typer(env_app, name="env")
app.add_typer(eval_app, name="eval")
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
def report() -> None:
    """Render the static leaderboard.html from the db."""
    from . import report as report_mod

    path = report_mod.render()
    typer.secho(f"wrote {path}", fg="green")


if __name__ == "__main__":
    app()
