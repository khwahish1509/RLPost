"""Install / load / list verifiers environments, Hub-compatible.

Installation shells out to the `prime` CLI (`prime env install owner/name`),
which resolves the package from the Prime Hub and installs it into the active
virtualenv with uv. We then verify the module is importable and register it in
the lab db. Loading goes straight through verifiers.load_environment, so
anything from the Hub — or any local verifiers env — is just another env here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from importlib import metadata

from . import db


class EnvInstallError(RuntimeError):
    pass


@dataclass
class InstalledEnv:
    slug: str      # what the user asked for, e.g. primeintellect/alphabet-sort
    env_id: str    # what verifiers loads, e.g. alphabet-sort
    version: str | None


def parse_slug(slug: str) -> str:
    """owner/name[@version] | name[@version] -> bare env id (name)."""
    name = slug.split("/")[-1]
    name = name.split("@")[0]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise EnvInstallError(f"Not a valid environment id: {slug!r}")
    return name


def installed_version(env_id: str) -> str | None:
    """Distribution version for an installed env package, if present."""
    try:
        return metadata.version(env_id)
    except metadata.PackageNotFoundError:
        return None


def install(slug: str) -> InstalledEnv:
    """Install an environment via the prime CLI and register it in the db."""
    if shutil.which("prime") is None:
        raise EnvInstallError(
            "The `prime` CLI is not on PATH. Install it with: uv tool install prime"
        )
    env_id = parse_slug(slug)
    proc = subprocess.run(
        ["prime", "env", "install", slug],
        capture_output=True,
        text=True,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    # prime returns exit 0 even when a dependency build fails, so returncode
    # alone can't be trusted — the importability check below is the real gate
    if proc.returncode != 0:
        raise EnvInstallError(f"`prime env install {slug}` failed:\n{combined}")
    version = installed_version(env_id)
    if version is None and not _importable(env_id):
        detail = ""
        if "Installation failed" in combined or "Build failure" in combined or "build" in combined.lower():
            detail = (
                " Its build failed — usually a heavy or system-level dependency "
                "that won't compile here (some hub environments need a specific "
                "OS or extra libraries). This one can't run on this machine."
            )
        raise EnvInstallError(
            f"Couldn't install {slug!r}.{detail} "
            "Try a different environment — most install cleanly."
        )
    conn = db.connect()
    try:
        db.register_environment(conn, slug, env_id, version)
    finally:
        conn.close()
    return InstalledEnv(slug=slug, env_id=env_id, version=version)


def _importable(env_id: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(env_id.replace("-", "_")) is not None


def load(env_id: str, **env_args):
    """Load a verifiers environment object by id (heavy import kept local)."""
    import verifiers

    return verifiers.load_environment(parse_slug(env_id), **env_args)


def list_installed() -> list[dict]:
    """Registered environments, with a live check that each still imports."""
    conn = db.connect()
    try:
        rows = db.list_environments(conn)
    finally:
        conn.close()
    return [
        {
            "slug": r["slug"],
            "env_id": r["env_id"],
            "version": r["version"] or installed_version(r["env_id"]) or "?",
            "installed_at": r["installed_at"],
            "importable": _importable(r["env_id"]),
        }
        for r in rows
    ]
