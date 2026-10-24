"""Shared-cache commands: du, gc, pin."""

import subprocess
from pathlib import Path

from ..config import STORE_NAMES, Config
from ..fiemap import reflink_usage
from ..gc import collect_candidates, run_gc
from ..log import die, say
from ..registry import live_envs
from ..store import read_meta, write_meta
from ..util import age_str, human_size, run


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


def cmd_du(cfg: Config, args) -> int:
    candidates = collect_candidates(cfg)
    uv_dir = None
    p = run(["uv", "cache", "dir"], check=False)
    if p.returncode == 0 and p.stdout.strip():
        uv_dir = Path(p.stdout.strip())
    ccache_dir = cfg.ccache_dir if cfg.ccache_dir.is_dir() else None

    envs = live_envs(cfg)
    say("measuring disk usage (reflink extent scan — may take a minute)...")
    groups: dict[str, list[Path]] = {s: [cfg.store(s)] for s in STORE_NAMES}
    scopes = {s: "stores" for s in STORE_NAMES}
    for name, path in envs.items():
        groups[f"env:{name}"] = [path / ".venv"]
        scopes[f"env:{name}"] = "venvs"
    usage = reflink_usage(groups, scopes)

    if usage.supported:
        logical = dict(usage.apparent)
    else:  # non-reflink filesystem: fall back to plain du for apparent sizes
        du = _du_many(
            [cfg.store(s) for s in STORE_NAMES]
            + [path / ".venv" for path in envs.values()]
        )
        logical = {s: du.get(str(cfg.store(s)), 0) for s in STORE_NAMES}
        for name, path in envs.items():
            logical[f"env:{name}"] = du.get(str(path / ".venv"), 0)

    def row(name: str, entries: str, key: str) -> str:
        line = f"  {name:18s} {entries:>7s} {human_size(logical.get(key, 0)):>10s}"
        if usage.supported:
            excl = usage.exclusive.get(key, 0)
            line += f" {human_size(excl):>10s}"
        return line

    head = f"  {'':18s} {'entries':>7s} {'logical':>10s}"
    if usage.supported:
        head += f" {'delete frees':>10s}"

    def subtotal(logical_sum: int, scope: str) -> None:
        line = f"  {'subtotal':18s} {'':>7s} {human_size(logical_sum):>10s}"
        if usage.supported:
            freed = usage.exclusive_by_scope.get(scope, 0)
            line += f" {human_size(freed):>10s}"
        print(line)

    print("stores")
    print(head)
    stores = sorted(STORE_NAMES, key=lambda s: -logical.get(s, 0))
    store_total = 0
    for store in stores:
        n = sum(1 for c in candidates if c.store == store)
        store_total += logical.get(store, 0)
        note = (
            "  (package data shared with uv cache)"
            if store == "venvs-base"
            else ""
        )
        print(row(store, str(n), store) + note)
    subtotal(store_total, "stores")

    if envs:
        print("\nvenvs")
        print(head)
        env_total = 0
        for name in sorted(envs):
            env_total += logical.get(f"env:{name}", 0)
            print(row(name, "", f"env:{name}"))
        subtotal(env_total, "venvs")

    if usage.supported:
        # Assign store↔venv reflinks to stores so these two figures are
        # non-overlapping and add exactly to the true on-disk total.
        stores_phys = usage.unique_by_scope.get("stores", 0)
        venvs_phys = usage.exclusive_by_scope.get("venvs", 0)
        print("\nphysical (non-overlapping)")
        print(f"  stores                         {human_size(stores_phys):>10s}")
        print(f"  venvs                          {human_size(venvs_phys):>10s}")
        print(f"  total                          {human_size(usage.unique_total):>10s}")
        print(f"  store gc cap                   {cfg.max_size_gb:.1f}GB")

    print()
    cache_dirs = [d for d in (uv_dir, ccache_dir) if d is not None]
    cache_du = _du_many(cache_dirs)
    for label, d in (("uv cache", uv_dir), ("ccache", ccache_dir)):
        if d is None:
            print(f"{label:10s} {'—':>10s}  (not found)")
        else:
            note = "  (separate; reflink source)" if label == "uv cache" else ""
            print(f"{label:10s} {human_size(cache_du.get(str(d), 0)):>10s}  {d}{note}")

    p = run(["df", "-B1", "--output=avail", str(cfg.cache_dir)], check=False)
    try:
        avail = int(p.stdout.splitlines()[-1])
        print(f"\ndisk free: {human_size(avail)} on {cfg.cache_dir}")
    except (IndexError, ValueError):
        pass
    print("note: logical counts reflinked files in every venv; delete frees is the "
          "space reclaimed by deleting that row/subtotal; physical assigns "
          "store↔venv shared blocks to stores so stores + venvs = total")

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
                      f"last-used={age_str(c.last_used, digits=1)}  {origin}{prot}")
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
