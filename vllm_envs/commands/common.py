"""Helpers shared by several subcommands."""

from pathlib import Path

from .. import drift
from ..config import MARKER_NAME, Config
from ..log import die, warn
from ..registry import live_envs, read_marker
from ..util import git_toplevel


def repo_root(path: Path) -> Path:
    try:
        return git_toplevel(path)
    except Exception:
        die(f"not inside a git repository: {path}")
        raise


def env_root_from_cwd() -> Path:
    root = repo_root(Path.cwd())
    if not (root / MARKER_NAME).exists():
        die(f"not inside a ve-managed env (no {MARKER_NAME} in {root})")
    return root


def resolve_env(cfg: Config, key: str) -> tuple[str, Path]:
    envs = live_envs(cfg)
    if key in envs:
        return key, envs[key]
    root = repo_root(Path(key).expanduser())
    if not (root / MARKER_NAME).exists():
        die(f"not a ve-managed env: {root}")
    return read_marker(root).get("name", root.name), root


def record_drift_before_removal(env_root: Path) -> None:
    if not (env_root / MARKER_NAME).exists():
        return
    try:
        drift.record_if_new(env_root, drift.compute(env_root), "found by ve reap/rm")
    except Exception as e:
        warn(f"drift check skipped ({e})", log=False)
