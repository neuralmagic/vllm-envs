"""post-checkout hook: auto re-resolve layers on commit hops (incl. bisect)."""

import os
import shutil
import sys
import time
from pathlib import Path

from .config import BUILD_LAYER_PATHS, MARKER_NAME, Config
from .layers import sync
from .log import say, warn
from .registry import live_envs, register_env, write_marker
from .util import run

HOOK_SENTINEL = "# installed by ve (vllm-envs)"

HOOK_TEMPLATE = """#!/bin/sh
{sentinel}
hook_dir="$(dirname "$0")"
if [ -x "$hook_dir/post-checkout.pre-ve" ]; then
    "$hook_dir/post-checkout.pre-ve" "$@" || exit $?
fi
exec {ve} hook post-checkout "$1" "$2" "$3"
"""

REQ_WATCH_PATHS = ("requirements",) + BUILD_LAYER_PATHS


def _hooks_dir(repo: Path) -> Path:
    common = run(["git", "rev-parse", "--git-common-dir"], cwd=repo).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = repo / common_path
    return common_path / "hooks"


def install_hook(repo: Path) -> None:
    hooks = _hooks_dir(repo)
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "post-checkout"
    ve_bin = shutil.which("ve") or sys.argv[0]
    content = HOOK_TEMPLATE.format(sentinel=HOOK_SENTINEL, ve=ve_bin)
    if hook.exists():
        existing = hook.read_text()
        if HOOK_SENTINEL in existing:
            if existing != content:
                hook.write_text(content)
            return
        # chain-load pre-existing hook
        say("preserving existing post-checkout hook as post-checkout.pre-ve")
        hook.rename(hooks / "post-checkout.pre-ve")
    hook.write_text(content)
    hook.chmod(0o755)
    _exclude_marker(repo)


def _exclude_marker(repo: Path) -> None:
    common = run(["git", "rev-parse", "--git-common-dir"], cwd=repo).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = repo / common_path
    exclude = common_path / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = exclude.read_text().splitlines() if exclude.exists() else []
    wanted = [MARKER_NAME, ".ve/", ".venv/"]
    missing = [w for w in wanted if w not in lines]
    if missing:
        exclude.write_text("\n".join(lines + missing) + "\n")


def _primary_repo(root: Path) -> Path:
    common = run(["git", "rev-parse", "--git-common-dir"], cwd=root).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = root / common_path
    return common_path.resolve().parent


def init_env(cfg: Config, root: Path, name: str | None = None) -> str:
    """Manage an existing checkout/worktree in place. Returns the env name."""
    repo = _primary_repo(root)
    base = name or root.name
    envs = live_envs(cfg)
    resolved = base
    i = 2
    while resolved in envs and envs[resolved].resolve() != root.resolve():
        resolved = f"{base}-{i}"
        i += 1
    install_hook(repo)
    write_marker(
        root, {"name": resolved, "repo": str(repo), "created_at": time.time()}
    )
    register_env(cfg, resolved, root, repo)
    sync(cfg, root)
    return resolved


def _maybe_auto_init(cfg: Config, root: Path) -> int:
    if not (root / "setup.py").exists():
        return 0
    git_dir = run(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=root
    ).stdout.strip()
    common = run(["git", "rev-parse", "--git-common-dir"], cwd=root).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = root / common_path
    if Path(git_dir).resolve() == common_path.resolve():
        return 0  # primary clone checkout; init explicitly with `ve init`
    say("new worktree detected → auto-initializing env (VE_NO_AUTO_INIT=1 to skip)")
    try:
        name = init_env(cfg, root)
    except Exception as e:
        warn(f"auto-init failed: {e}")
        warn("env NOT initialized — run `ve init` in the worktree")
        return 1
    say(f"env '{name}' ready — activate with: ve activate")
    return 0


def handle_post_checkout(cfg: Config, old: str, new: str, flag: str) -> int:
    if flag == "0":  # file checkout, not a branch/commit switch
        return 0
    root = Path(
        run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    )
    managed = (root / MARKER_NAME).exists()
    zeros = "0" * 40
    if old == zeros or new == zeros:
        # initial checkout (worktree add / clone)
        if managed or os.environ.get("VE_NO_AUTO_INIT") == "1":
            return 0
        return _maybe_auto_init(cfg, root)
    if not managed:
        return 0  # unmanaged worktree
    if old == new:
        return 0
    changed = run(
        ["git", "diff", "--name-only", old, new, "--", *REQ_WATCH_PATHS],
        cwd=root,
    ).stdout.strip()
    if not changed:
        return 0
    n = len(changed.splitlines())
    say(f"{n} build/deps-relevant file(s) changed across checkout → syncing env")
    try:
        sync(cfg, root)
    except Exception as e:
        warn(f"sync failed: {e}")
        warn("env may be INCONSISTENT — fix and run `ve sync`")
        return 1
    return 0
