"""`ve drift`: hand-made venv changes per env."""

import json
import shutil
import sys
from pathlib import Path

from .. import drift
from ..config import Config
from ..log import die, say
from ..registry import live_envs
from ..util import age_str
from .common import resolve_env


def drift_state(d) -> str:
    if d.empty:
        return "none" + (" (since last sync; pre-tracking env)"
                         if d.mode == "legacy" else "")
    return d.summary() + ("  [pre-tracking env: approximate]"
                          if d.mode == "legacy" else "")


def _drift_detail(name: str, path: Path, d) -> None:
    mode = ("vs. packages ve installed at last sync" if d.mode == "tracked"
            else "approximate: env predates drift tracking; changes made after "
                 "its last sync")
    print(f"env:      {name}  ({path})")
    print(f"baseline: {mode}, {age_str(d.since)} ago")
    if d.empty:
        print("drift:    none")
        return
    for n, spec in sorted(d.added.items()):
        print(f"  added    {spec}")
    for n, (old, new) in sorted(d.changed.items()):
        if old == new:
            print(f"  reinst.  {new}")
        else:
            print(f"  changed  {old}  →  {new}")
    for n, spec in sorted(d.removed.items()):
        print(f"  removed  {spec}")
    if d.n_files:
        print(f"  patched  {d.n_files} file(s) in site-packages:")
        for f in d.files:
            print(f"           {f}")
        if d.n_files > len(d.files):
            print(f"           … {d.n_files - len(d.files)} more")


def cmd_drift(cfg: Config, args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    if args.env:
        name, path = resolve_env(cfg, args.env)
        d = drift.compute(path)
        if d is None:
            die(f"no usable venv at {path / '.venv'}")
        if args.record:
            drift.record_if_new(path, d, "found by ve drift --record")
        if args.json:
            print(json.dumps({"env": name, "path": str(path), **_drift_json(d)},
                             indent=2))
        else:
            _drift_detail(name, path, d)
        return 0

    envs = sorted(live_envs(cfg).items())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda kv: drift.compute(kv[1]), envs))
    rows = []
    for (name, path), d in zip(envs, results):
        if d is None or (d.empty and not args.all_envs):
            continue
        if args.record:
            drift.record_if_new(path, d, "found by ve drift --record")
        rows.append((name, path, d))
    if args.json:
        print(json.dumps([{"env": n, "path": str(p), **_drift_json(d)}
                          for n, p, d in rows], indent=2))
        return 0
    if not rows:
        say(f"no drift in {len(envs)} env(s)")
        return 0
    width = min(max(len(n) for n, _, _ in rows), 40)
    print(f"{'env':{width}s}  {'pkgs':>5s}  {'files':>5s}  {'since':>5s}  changes")
    for name, _, d in sorted(rows, key=lambda r: r[2].since or 0, reverse=True):
        label = name if len(name) <= width else name[: width - 1] + "…"
        pkgs = len(d.added) + len(d.removed) + len(d.changed)
        since = age_str(d.since) + ("*" if d.mode == "legacy" else "")
        line = (f"{label:{width}s}  {pkgs:5d}  {d.n_files:5d}  {since:>5s}  "
                f"{d.summary()}")
        if sys.stdout.isatty():
            cols = shutil.get_terminal_size().columns
            if len(line) > cols:
                line = line[: cols - 1] + "…"
        print(line)
    print(f"\n{len(rows)} of {len(envs)} env(s) drifted from what ve installed. "
          "details: `ve drift <env>`"
          + ("  |  * = env predates drift tracking (changes since its last sync)"
             if any(d.mode == "legacy" for _, _, d in rows) else ""))
    return 0


def _drift_json(d) -> dict:
    return {
        "mode": d.mode, "since": d.since, "added": d.added, "removed": d.removed,
        "changed": {k: list(v) for k, v in d.changed.items()},
        "n_files": d.n_files, "files": d.files,
    }
