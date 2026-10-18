"""post-checkout hook: auto re-resolve layers on commit hops (incl. bisect)."""

import os
import shutil
import sys
import time
from pathlib import Path

from . import rules
from .config import MARKER_NAME, SCRATCH_DIR_NAME, Config
from .envsync import report_sync_failure, sync
from .log import error, say, warn
from .registry import live_envs, register_env, write_marker
from .util import atomic_write_text, git, git_path, git_toplevel, run

HOOK_SENTINEL = "# installed by ve (vllm-envs)"

HOOK_TEMPLATE = """#!/bin/sh
{sentinel}
hook_dir="$(dirname "$0")"
if [ -x "$hook_dir/{name}.pre-ve" ]; then
    "$hook_dir/{name}.pre-ve" "$@" || exit $?
fi
exec {ve} hook {name} {args}
"""

# For hooks git feeds on stdin: both the chained hook and ve must see it.
HOOK_TEMPLATE_STDIN = """#!/bin/sh
{sentinel}
hook_dir="$(dirname "$0")"
input="$(mktemp)" || exit 1
trap 'rm -f "$input"' EXIT
cat > "$input"
if [ -x "$hook_dir/{name}.pre-ve" ]; then
    "$hook_dir/{name}.pre-ve" "$@" < "$input" || exit $?
fi
{ve} hook {name} {args} < "$input"
"""

# Hooks ve installs, with the positional args each forwards to `ve hook <name>`.
# post-checkout resyncs on commit hops; post-commit catches relevant edits made
# in-place; post-merge catches `git pull`/merges; post-rewrite resyncs once a
# rebase finishes (the other hooks defer while a sequencer op is underway).
HOOK_ARGS = {
    "post-checkout": '"$1" "$2" "$3"',
    "post-commit": "",
    "post-merge": '"$1"',
    "post-rewrite": '"$1"',
}
HOOKS_WITH_STDIN = {"post-rewrite"}

# git state files signalling a sequencer op with a transient HEAD.
SEQUENCER_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
)

# Requirements, build inputs and extras inputs, from the rule table.
REQ_WATCH_PATHS = rules.watch_paths()


def _common_dir(path: Path) -> Path:
    return git_path(path, "--git-common-dir")


def _hooks_dir(repo: Path) -> Path:
    """Where git runs hooks from, honouring core.hooksPath."""
    return git_path(repo, "--git-path", "hooks")


def install_hook(repo: Path) -> None:
    hooks = _hooks_dir(repo)
    hooks.mkdir(parents=True, exist_ok=True)
    ve_bin = shutil.which("ve") or sys.argv[0]
    for name, args in HOOK_ARGS.items():
        _install_one(hooks, name, ve_bin, args)
    _exclude_marker(repo)


def _install_one(hooks: Path, name: str, ve_bin: str, args: str) -> None:
    hook = hooks / name
    template = HOOK_TEMPLATE_STDIN if name in HOOKS_WITH_STDIN else HOOK_TEMPLATE
    content = template.format(sentinel=HOOK_SENTINEL, ve=ve_bin, name=name, args=args)
    if hook.exists():
        existing = hook.read_text()
        if HOOK_SENTINEL in existing:
            if existing == content:
                return
        else:
            # chain-load pre-existing hook
            say(f"preserving existing {name} hook as {name}.pre-ve")
            hook.rename(hooks / f"{name}.pre-ve")
    # Replace by rename: a shell running the old hook reads it incrementally.
    atomic_write_text(hook, content, mode=0o755)


def _exclude_marker(repo: Path) -> None:
    exclude = _common_dir(repo) / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = exclude.read_text().splitlines() if exclude.exists() else []
    wanted = [MARKER_NAME, f"{SCRATCH_DIR_NAME}/", ".venv/"]
    missing = [w for w in wanted if w not in lines]
    if missing:
        exclude.write_text("\n".join(lines + missing) + "\n")


def _primary_repo(root: Path) -> Path:
    return _common_dir(root).resolve().parent


def init_env(
    cfg: Config, root: Path, name: str | None = None, origin: str = "init"
) -> str:
    """Manage an existing checkout/worktree in place. Returns the env name.
    ``origin`` (init | auto) marks the checkout as the user's: `ve rm` only
    unmanages it, never deletes it."""
    repo = _primary_repo(root)
    base = name or root.name
    envs = live_envs(cfg)
    resolved = base
    i = 2
    while resolved in envs and envs[resolved].resolve() != root.resolve():
        resolved = f"{base}-{i}"
        i += 1
    install_hook(repo)
    write_marker(root, {
        "name": resolved, "repo": str(repo), "origin": origin,
        "created_at": time.time(),
    })
    register_env(cfg, resolved, root, repo)
    sync(cfg, root)
    return resolved


def _maybe_auto_init(cfg: Config, root: Path) -> int:
    if not (root / "setup.py").exists():
        return 0
    git_dir = git_path(root, "--absolute-git-dir")
    if git_dir.resolve() == _common_dir(root).resolve():
        return 0  # primary clone checkout; init explicitly with `ve init`
    say("new worktree detected → auto-initializing env (VE_NO_AUTO_INIT=1 to skip)")
    try:
        name = init_env(cfg, root, origin="auto")
    except Exception as e:
        error(f"auto-init failed: {e}")
        if (root / MARKER_NAME).exists():
            warn("env only partly set up — fix the error above, then run "
                 "`ve sync` in the worktree", log=False)
        else:
            warn("env NOT initialized — run `ve init` in the worktree", log=False)
        return 1
    say(f"env '{name}' ready — activate with: source {root}/.venv/bin/activate")
    return 0


def _sequencer_in_progress(root: Path) -> bool:
    """True while a rebase/merge/cherry-pick/revert is underway.

    HEAD is transient during these ops, so syncing the env to it wastes work
    (rebase resyncs at every `onto` checkout) and races the operation. The env
    is resynced once the op finishes, via the post-rewrite hook. Bisect is
    deliberately absent: each bisect checkout is a commit the user is about to
    test, and BISECT_LOG exists for every one of them, including the
    checkout done by `git bisect reset`.
    """
    for name in SEQUENCER_MARKERS:
        path = git_path(root, "--git-path", name, check=False)
        if path is not None and path.exists():
            return True
    return False


def _resync(cfg: Config, root: Path) -> int:
    try:
        sync(cfg, root)
    except Exception as e:
        return report_sync_failure(e)
    return 0


def _changed_watch_paths(root: Path, old: str, new: str) -> list[str]:
    return run(
        ["git", "diff", "--name-only", old, new, "--", *REQ_WATCH_PATHS],
        cwd=root,
    ).stdout.splitlines()


def handle_post_checkout(cfg: Config, old: str, new: str, flag: str) -> int:
    if flag == "0":  # file checkout, not a branch/commit switch
        return 0
    root = git_toplevel(Path.cwd())
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
        say("git rebase/merge in progress → deferring env sync")
        return 0
    changed = _changed_watch_paths(root, old, new)
    if not changed:
        return 0
    n = len(changed)
    say(f"{n} build/deps-relevant file(s) changed across checkout → syncing env")
    return _resync(cfg, root)


def handle_post_commit(cfg: Config) -> int:
    """Resync a managed worktree after committing relevant in-place edits."""
    root = git_toplevel(Path.cwd())
    if not (root / MARKER_NAME).exists() or _sequencer_in_progress(root):
        return 0
    parent = git(["rev-parse", "--verify", "--quiet", "HEAD^"], cwd=root, check=False)
    changed = _changed_watch_paths(root, parent, "HEAD") if parent else ["initial"]
    if not changed:
        return 0
    say(f"{len(changed)} build/deps-relevant file(s) committed → syncing env")
    return _resync(cfg, root)


def handle_post_merge(cfg: Config) -> int:
    """Resync after a merge or `git pull` moved HEAD across relevant files."""
    root = git_toplevel(Path.cwd())
    if not (root / MARKER_NAME).exists():
        return 0
    orig = git(["rev-parse", "--verify", "--quiet", "ORIG_HEAD"], cwd=root, check=False)
    changed = _changed_watch_paths(root, orig, "HEAD") if orig else ["unknown"]
    if not changed:
        return 0
    say(f"{len(changed)} build/deps-relevant file(s) changed by merge → syncing env")
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
    orig = git(["rev-parse", "--verify", "--quiet", "ORIG_HEAD"], cwd=root, check=False)
    head = git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=root, check=False)
    return (orig, head) if orig and head else None


def handle_post_rewrite(cfg: Config, kind: str) -> int:
    """Resync after a rebase/amend rewrites HEAD (paired with the mid-op skip)."""
    root = git_toplevel(Path.cwd())
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
