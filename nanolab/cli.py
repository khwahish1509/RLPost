"""The nanolab CLI: env · eval · train · deployments · report.

Phase 1: `env` is real; the other verbs are stubs that name the phase that
implements them, so the CLI surface is stable from day one.
"""

from __future__ import annotations

import typer

from . import __version__, db, envs

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

NOT_IMPLEMENTED_EXIT = 2


def _stub(what: str, phase: int) -> None:
    typer.secho(f"{what}: not implemented yet (Phase {phase} in PLAN.md).", fg="yellow")
    raise typer.Exit(NOT_IMPLEMENTED_EXIT)


@app.callback()
def _main() -> None:
    """nanolab — one CLI, one SQLite file, one closed RL loop."""


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
    model: str = typer.Option(..., "--model", "-m", help="Model name, or base:adapter"),
    num_examples: int = typer.Option(20, "--num-examples", "-n"),
    rollouts: int = typer.Option(3, "--rollouts", "-r", help="Rollouts per example"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Evaluate a model on an environment (rollouts + rubric scoring)."""
    _stub(f"eval run {env} -m {model} -n {num_examples} -r {rollouts} --seed {seed}", 2)


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
    _stub(f"eval show {run_id}", 2)


# ── train ────────────────────────────────────────────────────────────────────


@app.command()
def train(
    config: str = typer.Argument(help="Path to a training TOML (see configs/)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from the last checkpoint"),
) -> None:
    """GRPO+LoRA training from a TOML config (synchronous loop)."""
    _stub(f"train {config}{' --resume' if resume else ''}", 3)


# ── deployments ──────────────────────────────────────────────────────────────


@deploy_app.command("create")
def deployments_create(adapter_id: int = typer.Argument(help="Adapter id from the adapters table")) -> None:
    """Serve an adapter through vLLM --enable-lora."""
    _stub(f"deployments create {adapter_id}", 5)


@deploy_app.command("list")
def deployments_list() -> None:
    """List deployments."""
    _stub("deployments list", 5)


# ── report ───────────────────────────────────────────────────────────────────


@app.command()
def report() -> None:
    """Render the static leaderboard.html from the db."""
    _stub("report", 2)


if __name__ == "__main__":
    app()
