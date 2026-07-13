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
if [ -x "$hook_dir/{name}.pre-ve" ]; then
    "$hook_dir/{name}.pre-ve" "$@" || exit $?
fi
exec {ve} hook {name} {args}
"""

# Hooks ve installs, with the positional args each forwards to `ve hook <name>`.
# post-checkout resyncs on commit hops; post-commit catches relevant edits made
# in-place; post-rewrite resyncs once a rebase finishes (the other hooks defer
# while a sequencer operation is in progress).
HOOK_ARGS = {
    "post-checkout": '"$1" "$2" "$3"',
    "post-commit": "",
    "post-rewrite": '"$1"',
}

# git state files signalling a sequencer op with a transient HEAD.
SEQUENCER_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)

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
    ve_bin = shutil.which("ve") or sys.argv[0]
    for name, args in HOOK_ARGS.items():
        _install_one(hooks, name, ve_bin, args)
    _exclude_marker(repo)


def _install_one(hooks: Path, name: str, ve_bin: str, args: str) -> None:
    hook = hooks / name
    content = HOOK_TEMPLATE.format(
        sentinel=HOOK_SENTINEL, ve=ve_bin, name=name, args=args
    )
    if hook.exists():
        existing = hook.read_text()
        if HOOK_SENTINEL in existing:
            if existing != content:
                hook.write_text(content)
            return
        # chain-load pre-existing hook
        say(f"preserving existing {name} hook as {name}.pre-ve")
        hook.rename(hooks / f"{name}.pre-ve")
    hook.write_text(content)
    hook.chmod(0o755)


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
    say(f"env '{name}' ready — activate with: source {root}/.venv/bin/activate")
    return 0


def _sequencer_in_progress(root: Path) -> bool:
    """True while a rebase/merge/cherry-pick/revert/bisect is underway.

    HEAD is transient during these ops, so syncing the env to it wastes work
    (rebase resyncs at every `onto` checkout) and races the operation. The env
    is resynced once the op finishes: rebase/amend via the post-rewrite hook,
    bisect via the post-checkout on `git bisect reset`.
    """
    for name in SEQUENCER_MARKERS:
        path = run(
            ["git", "rev-parse", "--git-path", name], cwd=root, check=False
        ).stdout.strip()
        if not path:
            continue
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return True
    return False


def _resync(cfg: Config, root: Path) -> int:
    try:
        sync(cfg, root)
    except Exception as e:
        warn(f"sync failed: {e}")
        warn("previous .venv retained; its hashes may be STALE for this checkout")
        return 1
    return 0


def _changed_watch_paths(root: Path, old: str, new: str) -> list[str]:
    return run(
        ["git", "diff", "--name-only", old, new, "--", *REQ_WATCH_PATHS],
        cwd=root,
    ).stdout.splitlines()


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
    if _sequencer_in_progress(root):
        # e.g. rebase checking out its `onto`; resync once the op completes.
        say("git rebase/bisect in progress → deferring env sync")
        return 0
    changed = _changed_watch_paths(root, old, new)
    if not changed:
        return 0
    n = len(changed)
    say(f"{n} build/deps-relevant file(s) changed across checkout → syncing env")
    return _resync(cfg, root)


def handle_post_commit(cfg: Config) -> int:
    """Resync a managed worktree after committing relevant in-place edits."""
    root = Path(
        run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    )
    if not (root / MARKER_NAME).exists() or _sequencer_in_progress(root):
        return 0
    parent = run(
        ["git", "rev-parse", "--verify", "--quiet", "HEAD^"],
        cwd=root,
        check=False,
    ).stdout.strip()
    changed = _changed_watch_paths(root, parent, "HEAD") if parent else ["initial"]
    if not changed:
        return 0
    say(f"{len(changed)} build/deps-relevant file(s) committed → syncing env")
    return _resync(cfg, root)


def _rewrite_range(root: Path) -> tuple[str, str] | None:
    """(before, after) shas for the rewrite, else None.

    git feeds post-rewrite `<old> <new>` pairs on stdin, oldest-first; the last
    line's old is the pre-op tip (works for both rebase and amend). Falls back
    to ORIG_HEAD..HEAD (set by rebase; stale for amend) when stdin is empty.
    """
    try:
        pairs = [ln.split() for ln in sys.stdin.read().splitlines() if ln.strip()]
    except Exception:
        pairs = []
    if pairs and len(pairs[-1]) >= 2:
        return pairs[-1][0], pairs[-1][1]
    orig = run(
        ["git", "rev-parse", "--verify", "--quiet", "ORIG_HEAD"],
        cwd=root,
        check=False,
    ).stdout.strip()
    head = run(
        ["git", "rev-parse", "--verify", "--quiet", "HEAD"], cwd=root, check=False
    ).stdout.strip()
    return (orig, head) if orig and head else None


def handle_post_rewrite(cfg: Config, kind: str) -> int:
    """Resync after a rebase/amend rewrites HEAD (paired with the mid-op skip)."""
    root = Path(
        run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    )
    if not (root / MARKER_NAME).exists():
        return 0  # unmanaged worktree
    rng = _rewrite_range(root)
    if rng and rng[0] != rng[1]:
        changed = _changed_watch_paths(root, rng[0], rng[1])
        if not changed:
            return 0
        n = len(changed)
        say(f"{n} build/deps-relevant file(s) changed across {kind} → syncing env")
    else:
        say(f"{kind} complete → syncing env")
    return _resync(cfg, root)
