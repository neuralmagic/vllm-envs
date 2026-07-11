import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import MARKER_NAME, STORE_NAMES, Config, load_config
from .extprojects import user_overrides
from .fiemap import reflink_usage
from .gc import collect_candidates, run_gc, total_size
from .hashing import build_key, build_paths_dirty, detect_platform, venv_keys
from .hooks import handle_post_checkout, init_env, install_hook
from .layers import sync
from .log import die, say, warn
from .registry import (
    live_envs,
    read_marker,
    register_env,
    unregister_env,
    write_marker,
)
from .store import read_meta, write_meta
from .util import git, human_size, run


def _repo_root(path: Path) -> Path:
    try:
        return Path(git(["rev-parse", "--show-toplevel"], cwd=path))
    except Exception:
        die(f"not inside a git repository: {path}")
        raise


def _env_root_from_cwd() -> Path:
    root = _repo_root(Path.cwd())
    if not (root / MARKER_NAME).exists():
        die(f"not inside a ve-managed env (no {MARKER_NAME} in {root})")
    return root


def cmd_new(cfg: Config, args) -> int:
    repo = _repo_root(Path(args.repo) if args.repo else Path.cwd())
    if not (repo / "setup.py").exists():
        warn(f"{repo} does not look like a vLLM checkout (no setup.py)")
    name = args.name or re.sub(r"[^A-Za-z0-9._\-]", "-", args.ref)
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
    write_marker(dest, {"name": name, "repo": str(repo), "created_at": time.time()})
    register_env(cfg, name, dest, repo)
    t0 = time.time()
    sync(cfg, dest, fresh_venv=False)
    say(f"env '{name}' ready in {time.time() - t0:.0f}s: {dest}")
    say(f"activate with: source {dest}/.venv/bin/activate")
    return 0


def cmd_init(cfg: Config, args) -> int:
    root = _repo_root(Path.cwd())
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
    root = _env_root_from_cwd()
    activate = root / ".venv" / "bin" / "activate"
    if not activate.exists():
        die(f"no venv at {root / '.venv'} — run `ve sync`")
    print(f"source {activate}")
    warn("run the line above (or just `source .venv/bin/activate`); to make "
         "`ve activate` work directly, add to your shell rc: "
         'eval "$(ve shellenv)"')
    return 0


def cmd_sync(cfg: Config, args) -> int:
    env_root = _env_root_from_cwd()
    sync(cfg, env_root, fresh_venv=args.fresh_venv)
    return 0


def cmd_rm(cfg: Config, args) -> int:
    envs = live_envs(cfg)
    if args.name not in envs:
        die(f"unknown env '{args.name}' (known: {', '.join(sorted(envs)) or 'none'})")
    dest = envs[args.name]
    repo = Path(read_marker(dest).get("repo", ""))
    if str(repo) and repo.resolve() == dest.resolve():
        say(f"'{args.name}' is an in-place env (ve init): unmanaging, "
            "leaving the checkout intact")
        (dest / MARKER_NAME).unlink(missing_ok=True)
        unregister_env(cfg, args.name)
        return 0
    say(f"removing env '{args.name}' at {dest}")
    if repo.is_dir():
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
    root = _env_root_from_cwd()
    m = read_marker(root)
    platform = cfg.platform or detect_platform()
    keys = venv_keys(root, platform, cfg.python, cap=cfg.cap)
    bhash = build_key(root, platform, cfg.python)
    head = git(["rev-parse", "--short", "HEAD"], cwd=root)
    overrides = user_overrides()
    dirty = build_paths_dirty(root)

    print(f"env:        {m.get('name', '?')} @ {head}  ({root})")
    print(f"platform:   {platform} (python {cfg.python})")

    def state(current: str | None, wanted: str) -> str:
        return "OK" if current == wanted else f"STALE (have {current or 'none'})"

    print(f"venv base:  {keys.base_hash}  {state(m.get('venv_base_hash'), keys.base_hash)}")
    print(f"venv full:  {keys.full_hash}  {state(m.get('venv_full_hash'), keys.full_hash)}")
    build_state = state(m.get("build_hash") or None, bhash)
    if m.get("attach_mode") == "local-build":
        build_state = "private (dirty/override build)"
    print(f"build:      {bhash}  {build_state}")
    if not keys.layout.recognized:
        print("note:       unrecognized requirements layout — coarse hashing in effect")
    if dirty:
        print("dirty:      csrc/cmake working tree is dirty → private builds, no publish")
    for var, val in overrides.items():
        print(f"override:   build-layer caching DISABLED: {var}={val}")
    return 0


def cmd_gc(cfg: Config, args) -> int:
    run_gc(cfg, dry_run=args.dry_run, free_gb=args.free)
    return 0


def _du_many(paths: list[Path]) -> dict[str, int]:
    """Parallel `du -sB1` per path (hardlinks deduped within each path)."""
    procs = {
        str(p): subprocess.Popen(
            ["du", "-sB1", str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for p in paths if p.is_dir()
    }
    out: dict[str, int] = {}
    for key, proc in procs.items():
        line, _ = proc.communicate()
        try:
            out[key] = int(line.split()[0])
        except (IndexError, ValueError):
            out[key] = 0
    return out


def _du_combined(paths: list[Path]) -> int:
    """One du over all paths: hardlinks deduped across them (true footprint)."""
    existing = [str(p) for p in paths if p.is_dir()]
    if not existing:
        return 0
    proc = run(["du", "-scB1", *existing], check=False)
    for line in proc.stdout.splitlines():
        if line.endswith("\ttotal"):
            return int(line.split()[0])
    return 0


def _age(ts: float) -> str:
    h = (time.time() - ts) / 3600
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def cmd_du(cfg: Config, args) -> int:
    candidates = collect_candidates(cfg)
    store_paths = [cfg.store(s) for s in STORE_NAMES]
    uv_dir = ccache_dir = None
    p = run(["uv", "cache", "dir"], check=False)
    if p.returncode == 0 and p.stdout.strip():
        uv_dir = Path(p.stdout.strip())
    p = run(["ccache", "--get-config", "cache_dir"], check=False)
    if p.returncode == 0 and p.stdout.strip():
        ccache_dir = Path(p.stdout.strip())
    related = [d for d in (uv_dir, ccache_dir) if d is not None]

    envs = live_envs(cfg)
    say("measuring disk usage (du + reflink extent scan — may take a minute)...")
    phys = _du_many(store_paths + related)
    combined = _du_combined(store_paths + related)

    groups: dict[str, list[Path]] = {s: [cfg.store(s)] for s in STORE_NAMES}
    for name, path in envs.items():
        groups[f"env:{name}"] = [path / ".venv"]
    usage = reflink_usage(groups)

    print(f"{'store':14s} {'entries':>7s} {'capped':>10s} {'du':>10s}")
    capped_total = phys_total = 0
    for store in STORE_NAMES:
        cs = [c for c in candidates if c.store == store]
        capped = sum(c.size for c in cs)
        physical = phys.get(str(cfg.store(store)), 0)
        capped_total += capped
        phys_total += physical
        print(f"{store:14s} {len(cs):>7d} {human_size(capped):>10s} "
              f"{human_size(physical):>10s}")
    print(f"{'ve total':14s} {'':>7s} {human_size(capped_total):>10s} "
          f"{human_size(phys_total):>10s}  (cap {cfg.max_size_gb:.0f}GB)")

    print()
    for label, d in (("uv cache", uv_dir), ("ccache", ccache_dir)):
        if d is None:
            print(f"{label:14s} {'—':>18s}  (not found)")
        else:
            print(f"{label:14s} {human_size(phys.get(str(d), 0)):>18s}  {d}")
    all_individual = phys_total + sum(phys.get(str(d), 0) for d in related)
    shared = max(all_individual - combined, 0)
    print(f"{'combined':14s} {human_size(combined):>18s}  "
          f"(hardlink-shared across the above: {human_size(shared)})")
    if usage.supported:
        print(f"{'physical':14s} {human_size(usage.unique_total):>18s}  "
              "(reflink-aware: unique on-disk blocks, ve stores + live envs)")

    if envs:
        print()
        for name, path in sorted(envs.items()):
            sz = usage.apparent.get(f"env:{name}", 0)
            excl = usage.exclusive.get(f"env:{name}", 0)
            excl_col = f" excl {human_size(excl):>9s}" if usage.supported else ""
            print(f"env {name:20s} {human_size(sz):>10s}{excl_col}  {path}/.venv")

    p = run(["df", "-B1", "--output=avail", str(cfg.cache_dir)], check=False)
    try:
        avail = int(p.stdout.splitlines()[-1])
        print(f"\ndisk free: {human_size(avail)} on {cfg.cache_dir}")
    except (IndexError, ValueError):
        pass
    print("note: capped = physical blocks apportioned by hardlink count "
          "(what gc enforces); du = standalone per-store blocks (reflink-shared "
          "extents counted per-file); physical/excl = reflink-aware unique "
          "blocks (excl = bytes reclaimed if that env alone is deleted)")

    if args.entries:
        for store in STORE_NAMES:
            cs = sorted(
                (c for c in candidates if c.store == store),
                key=lambda c: -c.size,
            )
            if not cs:
                continue
            print(f"\n== {store} ==")
            for c in cs:
                meta = read_meta(c.entry)
                origin = meta.get("origin") or meta.get("base") or meta.get("kind", "")
                prot = f"  [{c.protected}]" if c.protected else ""
                print(f"{human_size(c.size):>10s}  {c.entry.name}  "
                      f"last-used={_age(c.last_used)}  {origin}{prot}")
    return 0


def cmd_pin(cfg: Config, args) -> int:
    store, _, name = args.entry.partition("/")
    if store not in STORE_NAMES or not name:
        die(f"expected <store>/<hash>, e.g. builds/abc123 (stores: {', '.join(STORE_NAMES)})")
    entry = cfg.store(store) / name
    if not entry.is_dir():
        die(f"no such entry: {entry}")
    meta = read_meta(entry)
    meta["pinned"] = not args.unpin
    write_meta(entry, meta)
    say(f"{'unpinned' if args.unpin else 'pinned'} {store}/{name}")
    return 0


def cmd_hook(cfg: Config, args) -> int:
    if args.event != "post-checkout":
        return 0
    return handle_post_checkout(cfg, args.old, args.new, args.flag)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ve", description="Fast disposable vLLM dev environments"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="create env from a ref (worktree + cached layers)")
    sp.add_argument("ref")
    sp.add_argument("--name", "-n")
    sp.add_argument("--repo", help="path to the vLLM clone (default: cwd)")
    sp.set_defaults(func=cmd_new)

    sp = sub.add_parser(
        "init", help="manage the current checkout/worktree in place (build + venv)"
    )
    sp.add_argument("--name", "-n", help="env name (default: directory name)")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser(
        "activate",
        help="activate the env venv in the current shell "
             "(needs eval \"$(ve shellenv)\" in your rc)",
    )
    sp.set_defaults(func=cmd_activate)

    sp = sub.add_parser(
        "shellenv",
        help="print shell function enabling `ve activate`; "
             "add to your rc: eval \"$(ve shellenv)\"",
    )
    sp.set_defaults(func=cmd_shellenv)

    sp = sub.add_parser("sync", help="re-resolve layers for the current worktree HEAD")
    sp.add_argument("--fresh-venv", action="store_true",
                    help="discard env venv divergence, re-clone from template")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("rm", help="remove an env (never shared caches)")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("list", help="list envs")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("status", help="show env layer hashes and cache state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("gc", help="prune caches (LRU, 50GB default cap)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--free", type=float, metavar="GB",
                    help="evict until this many GB are reclaimed")
    sp.set_defaults(func=cmd_gc)

    sp = sub.add_parser(
        "du", help="cache usage audit: per store, uv cache/ccache, live envs"
    )
    sp.add_argument("--entries", "-e", action="store_true",
                    help="also list per-entry sizes with age and origin")
    sp.set_defaults(func=cmd_du)

    sp = sub.add_parser("pin", help="pin/unpin a store entry (exempt from gc)")
    sp.add_argument("entry", help="<store>/<hash>")
    sp.add_argument("--unpin", action="store_true")
    sp.set_defaults(func=cmd_pin)

    sp = sub.add_parser("hook", help=argparse.SUPPRESS)
    sp.add_argument("event")
    sp.add_argument("old")
    sp.add_argument("new")
    sp.add_argument("flag")
    sp.set_defaults(func=cmd_hook)

    args = p.parse_args(argv)
    cfg = load_config()
    for store in STORE_NAMES:
        cfg.store(store).mkdir(parents=True, exist_ok=True)
    try:
        return args.func(cfg, args)
    except KeyboardInterrupt:
        warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
