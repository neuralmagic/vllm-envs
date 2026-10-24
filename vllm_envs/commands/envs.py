"""Env lifecycle: new, init, activate/shellenv, sync, rm, list, status."""

import re
import shutil
import time
from pathlib import Path

from .. import drift, precompiled, rules
from ..buildlayer import find_wheel
from ..config import MARKER_NAME, Config
from ..envsync import report_sync_failure, sync
from ..extprojects import user_overrides
from ..extras import resolve as resolve_vllm_extras
from ..hashing import build_key, build_paths_dirty, detect_platform, venv_keys
from ..hooks import init_env, install_hook
from ..log import die, say, warn
from ..registry import (
    live_envs,
    read_marker,
    register_env,
    unregister_env,
    write_marker,
)
from ..store import read_meta
from ..util import git, run
from ..worktrees import removal_safety
from .common import env_root_from_cwd, record_drift_before_removal, repo_root
from .drift import drift_state


def cmd_new(cfg: Config, args) -> int:
    repo = repo_root(Path(args.repo) if args.repo else Path.cwd())
    if not (repo / "setup.py").exists():
        warn(f"{repo} does not look like a vLLM checkout (no setup.py)")
    name = re.sub(r"[^A-Za-z0-9._\-]", "-", args.name or args.ref).lstrip(".-")
    if not name:
        die(f"cannot derive an env name from {args.name or args.ref!r}; pass --name")
    dest = cfg.envs_root / name
    if dest.exists():
        die(f"env '{name}' already exists at {dest} (use --name or `ve rm {name}`)")
    cfg.envs_root.mkdir(parents=True, exist_ok=True)

    say(f"creating worktree {dest} at {args.ref}")
    run(
        ["git", "worktree", "add", "--detach", str(dest), args.ref],
        cwd=repo,
        env={"VE_NO_AUTO_INIT": "1"},  # ve new does its own init below
        stream_prefix="[ve]   [git] ",
    )
    install_hook(repo)
    write_marker(dest, {"name": name, "repo": str(repo), "origin": "new",
                        "created_at": time.time()})
    register_env(cfg, name, dest, repo)
    t0 = time.time()
    sync(cfg, dest, fresh_venv=False)
    say(f"env '{name}' ready in {time.time() - t0:.0f}s: {dest}")
    say(f"activate with: source {dest}/.venv/bin/activate")
    return 0


def cmd_init(cfg: Config, args) -> int:
    root = repo_root(Path.cwd())
    if (root / MARKER_NAME).exists():
        die(f"already a ve-managed env: {root} (use `ve sync`)")
    if not (root / "setup.py").exists():
        warn(f"{root} does not look like a vLLM checkout (no setup.py)")
    t0 = time.time()
    name = init_env(cfg, root, args.name)
    say(f"env '{name}' ready in {time.time() - t0:.0f}s: {root}")
    say(f"activate with: source {root}/.venv/bin/activate")
    say("(or `ve activate` after adding to your shell rc: "
        'eval "$(ve shellenv)")')
    return 0


SHELL_FUNC = """\
ve() {
    if [ "$1" = "activate" ]; then
        local _ve_root _ve_act
        _ve_root=$(command git rev-parse --show-toplevel 2>/dev/null) || {
            echo "[ve] not inside a git repo" >&2; return 1; }
        _ve_act="$_ve_root/.venv/bin/activate"
        [ -f "$_ve_act" ] || {
            echo "[ve] no venv at $_ve_root/.venv — run 've sync'" >&2; return 1; }
        . "$_ve_act"
    else
        command ve "$@"
    fi
}
"""


def cmd_shellenv(cfg: Config, args) -> int:
    print(SHELL_FUNC, end="")
    return 0


def cmd_activate(cfg: Config, args) -> int:
    # only reached without the shellenv function; the function sources directly
    root = env_root_from_cwd()
    activate = root / ".venv" / "bin" / "activate"
    if not activate.exists():
        die(f"no venv at {root / '.venv'} — run `ve sync`")
    print(f"source {activate}")
    warn("run the line above (or just `source .venv/bin/activate`); to make "
         "`ve activate` work directly, add to your shell rc: "
         'eval "$(ve shellenv)"', log=False)
    return 0


def cmd_sync(cfg: Config, args) -> int:
    env_root = env_root_from_cwd()
    install_hook(Path(read_marker(env_root).get("repo") or env_root))  # upgrade hooks
    try:
        sync(cfg, env_root, fresh_venv=args.fresh_venv)
    except Exception as e:
        return report_sync_failure(e)
    return 0


def cmd_rm(cfg: Config, args) -> int:
    envs = live_envs(cfg)
    if args.name not in envs:
        die(f"unknown env '{args.name}' (known: {', '.join(sorted(envs)) or 'none'})")
    dest = envs[args.name]
    marker = read_marker(dest)
    repo = Path(marker["repo"]) if marker.get("repo") else None
    if _env_origin(cfg, dest, marker) != "new":
        say(f"'{args.name}' is a checkout ve manages in place (ve init / "
            "auto-init): unmanaging, leaving the checkout intact")
        (dest / MARKER_NAME).unlink(missing_ok=True)
        unregister_env(cfg, args.name)
        return 0
    if not args.force:
        safe, reason = removal_safety(dest)
        if not safe:
            die(f"refusing to remove '{args.name}': {reason}; "
                "commit/push or use --force")
    record_drift_before_removal(dest)
    say(f"removing env '{args.name}' at {dest}")
    if repo is not None and repo.is_dir():
        run(
            ["git", "worktree", "remove", "--force", str(dest)],
            cwd=repo,
            check=False,
        )
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    unregister_env(cfg, args.name)
    say("removed (shared caches untouched)")
    return 0


def _env_origin(cfg: Config, dest: Path, marker: dict) -> str:
    """new (ve-created, deletable) | init | auto. Markers written before the
    origin field: only worktrees under envs_root came from `ve new`."""
    if marker.get("origin"):
        return marker["origin"]
    try:
        dest.resolve().relative_to(cfg.envs_root.resolve())
    except ValueError:
        return "init"
    return "new"


def cmd_list(cfg: Config, args) -> int:
    envs = live_envs(cfg)
    if not envs:
        say("no envs")
        return 0
    for name, path in sorted(envs.items()):
        m = read_marker(path)
        head = git(["rev-parse", "--short", "HEAD"], cwd=path) if path.exists() else "?"
        print(f"{name:24s} {head:12s} {path}  (build={m.get('build_hash') or 'private'})")
    return 0


def cmd_status(cfg: Config, args) -> int:
    root = env_root_from_cwd()
    m = read_marker(root)
    platform = cfg.platform or detect_platform()
    keys = venv_keys(root, platform, cfg.python,
                     with_test=cfg.with_test, cap=cfg.cap)
    bhash = build_key(root, platform, cfg.python, root / ".venv")
    head = git(["rev-parse", "--short", "HEAD"], cwd=root)
    overrides = user_overrides()
    dirty = build_paths_dirty(root)

    print(f"env:        {m.get('name', '?')} @ {head}  ({root})")
    print(f"platform:   {platform} (python {cfg.python})")

    def state(current: str | None, wanted: str) -> str:
        return "OK" if current == wanted else f"STALE (have {current or 'none'})"

    local_venv = root / ".venv"
    local_state = state(m.get("venv_full_hash"), keys.full_hash)
    if not local_venv.is_dir():
        local_state = "MISSING (run `ve sync`)"
    print(f"local venv: {local_state}")

    def template_state(store: str, key: str) -> str:
        entry = cfg.store(store) / key
        if (entry / ".complete").exists():
            return "HIT"
        if entry.is_dir():
            meta = read_meta(entry)
            detail = meta.get("stage") or "previous construction did not finish"
            return f"INCOMPLETE ({detail})"
        return "MISS"

    print(
        f"venv base:  {keys.base_hash}  "
        f"{template_state('venvs-base', keys.base_hash)}"
    )
    print(
        f"venv full:  {keys.full_hash}  "
        f"{template_state('venvs', keys.full_hash)}"
    )
    if cfg.with_test and keys.layout.test_files:
        rel = keys.layout.test_files[0].relative_to(root)
        print(f"test deps:  included ({rel})")
    else:
        print(f"test deps:  excluded ({'no test reqs found' if cfg.with_test else 'disabled'})")
    fired = [f"{v['id']}@{v['rev']}" for v in rules.explain(keys.facts, cfg.with_test)
             if v["fired"]]
    print(f"rules:      {', '.join(fired) or 'none'}  (`ve rules` explains)")
    build_state = state(m.get("build_hash") or None, bhash)
    if m.get("attach_mode") == "local-build":
        build_state = "private (dirty/override build)"
    elif m.get("attach_mode") == "precompiled-fetch" and m.get("build_hash") == bhash:
        build_state = (
            "OK (cached precompiled)" if find_wheel(cfg.store("builds") / bhash)
            else "OK (env-local precompiled fallback)"
        )
    print(f"build:      {bhash}  {build_state}")
    origin = read_meta(cfg.store("builds") / bhash).get("origin", "")
    if origin.startswith("precompiled:"):
        commit = origin.split(":", 1)[1]
        print(f"kernels:    precompiled from {commit[:12]} "
              f"({precompiled.describe(precompiled.relation(root, commit))})")
    if cfg.with_vllm_extras and platform == "cuda":
        try:
            wanted_extras = resolve_vllm_extras(root, keys.base_hash, keys.full_hash).key
            print(
                f"vLLM extras:{wanted_extras}  "
                f"{state(m.get('vllm_extras_hash'), wanted_extras)}"
            )
        except RuntimeError:
            print("vLLM extras: unsupported by this commit")
    else:
        print("vLLM extras: disabled")
    if not keys.layout.recognized:
        print("note:       unrecognized requirements layout — coarse hashing in effect")
    if dirty:
        print("dirty:      csrc/cmake working tree is dirty → private builds, no publish")
    for var, val in overrides.items():
        print(f"override:   build-layer caching DISABLED: {var}={val}")
    d = drift.compute(root)
    if d is not None:
        print(f"drift:      {drift_state(d)}  (`ve drift .` for details)")
    return 0
