"""LRU garbage collection across the content-addressed stores."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import EVICTION_ORDER, Config
from .fiemap import reflink_usage
from .locks import release, try_entry_lock
from .log import say, warn
from .registry import referenced_build_hashes
from .store import META_NAME, entry_size, is_pinned, last_used
from .util import human_size, run


@dataclass
class Candidate:
    store: str
    entry: Path
    size: int
    last_used: float
    protected: str  # "" = evictable, else reason


def collect_candidates(cfg: Config) -> list[Candidate]:
    now = time.time()
    min_age_s = cfg.min_age_hours * 3600
    refs = referenced_build_hashes(cfg)
    out: list[Candidate] = []
    for store in EVICTION_ORDER:
        base = cfg.store(store)
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            lu = last_used(entry)
            protected = ""
            if is_pinned(entry):
                protected = "pinned"
            elif store == "builds" and entry.name in refs:
                protected = "referenced by live env"
            elif now - lu < min_age_s:
                protected = f"used <{cfg.min_age_hours:.0f}h ago"
            out.append(Candidate(store, entry, entry_size(entry), lu, protected))
    return out


def total_size(candidates: list[Candidate]) -> int:
    return sum(c.size for c in candidates)


def _evict(c: Candidate, dry_run: bool) -> bool:
    age_d = (time.time() - c.last_used) / 86400
    label = f"{c.store}/{c.entry.name} ({human_size(c.size)}, last used {age_d:.1f}d ago)"
    if dry_run:
        say(f"would evict {label}")
        return True
    fd = try_entry_lock(c.entry)
    if fd is None:
        warn(f"skipping {label}: locked by another process")
        return False
    try:
        say(f"evicting {label}")
        shutil.rmtree(c.entry, ignore_errors=True)
        lock_file = c.entry.parent / (c.entry.name + ".lock")
        lock_file.unlink(missing_ok=True)
        return True
    finally:
        release(fd)


def _prune_uv_cache(dry_run: bool) -> None:
    if shutil.which("uv") is None:
        return
    if dry_run:
        say("would run `uv cache prune`")
        return
    # safe: venv hardlinks keep pruned files' data alive; only future
    # installs pay a re-download
    say("pruning uv cache")
    run(["uv", "cache", "prune"], check=False, stream_prefix="[ve]   [uv] ")


def run_gc(
    cfg: Config,
    dry_run: bool = False,
    free_gb: float | None = None,
) -> None:
    candidates = collect_candidates(cfg)

    # Enforce the cap on real on-disk size: stores are heavily reflinked, so the
    # logical sum wildly overcounts. `phys[key]` is the physical bytes an entry's
    # eviction actually frees (extents not shared with any other entry).
    def key(c: Candidate) -> str:
        return f"{c.store}/{c.entry.name}"

    usage = reflink_usage({key(c): [c.entry] for c in candidates})
    if usage.supported:
        total = usage.unique_total
        phys = usage.exclusive
    else:  # non-reflink filesystem: physical == logical
        total = total_size(candidates)
        phys = {key(c): c.size for c in candidates}

    cap = cfg.max_size_gb * 1024**3
    say(f"cache size: {human_size(total)} physical / cap {human_size(cap)} "
        f"({len(candidates)} entries)")

    if free_gb is not None:
        target_reclaim = free_gb * 1024**3
    elif total > cap:
        target_reclaim = total - cap
    else:
        say("under cap; nothing to do")
        _prune_uv_cache(dry_run)
        return

    # Eviction order: store priority (EVICTION_ORDER), then LRU within store.
    order = {s: i for i, s in enumerate(EVICTION_ORDER)}
    evictable = sorted(
        (c for c in candidates if not c.protected),
        key=lambda c: (order[c.store], c.last_used),
    )
    skipped = [c for c in candidates if c.protected]
    for c in skipped:
        say(f"protected: {c.store}/{c.entry.name} ({c.protected})")

    reclaimed = 0
    for c in evictable:
        if reclaimed >= target_reclaim:
            break
        if _evict(c, dry_run):
            reclaimed += phys.get(key(c), 0)
    verb = "would reclaim" if dry_run else "reclaimed"
    say(f"{verb} {human_size(reclaimed)} (target {human_size(target_reclaim)})")
    if reclaimed < target_reclaim:
        warn("could not reach target: remaining entries are protected "
             "(refcounted, pinned, or inside the min-age window)")
    _prune_uv_cache(dry_run)
