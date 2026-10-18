"""Read-only diagnostics: `ve precompiled` and `ve rules`."""

import json
from pathlib import Path

from .. import precompiled, rules, treefacts
from ..config import Config
from ..hashing import detect_platform
from ..util import git
from .common import repo_root


def cmd_precompiled(cfg: Config, args) -> int:
    root = repo_root(Path(args.path or Path.cwd()))
    look = precompiled.lookup(cfg, root, use_index=False, record=False)
    if args.json:
        print(json.dumps(look.to_json(), indent=2))
        return 0
    on_main = [m.distance for m in look.matches if m.distance is not None]
    print(f"head:        {look.head[:12]}")
    print(f"paths:       {', '.join(look.paths)}")
    print(f"identity:    {look.identity}")
    print(f"merge-base:  {(look.merge_base or 'none')[:12]} "
          f"on {look.upstream or '?'} ({look.scanned} commits scanned)")
    if on_main:
        print(f"main run:    {len(on_main)} matching commit(s), "
              f"{-min(on_main)} before .. {max(on_main)} after merge-base")
    else:
        print("main run:    no main commit has these compiled sources")
    for commit, result in look.probes:
        print(f"probe:       {commit[:12]}  {result}")
    if look.chosen is None:
        print(f"wheel:       none — {look.reason}")
        return 1
    stale = (f"; stale warning paths: {', '.join(look.chosen.stale)}"
             if look.chosen.stale else "")
    print(f"wheel:       {look.chosen.commit[:12]} "
          f"({precompiled.describe(look.chosen.distance)}{stale})")
    print(f"url:         {look.url}")
    return 0


def _fact_value(root: Path, value) -> str:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(root))
        except ValueError:
            return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fact_value(root, v) for v in value) or "-"
    return "-" if value in (None, "") else str(value)


def cmd_rules(cfg: Config, args) -> int:
    root = repo_root(Path(args.path or Path.cwd()))
    platform = cfg.platform or detect_platform()
    facts = treefacts.TreeFacts(root, platform)
    verdicts = rules.explain(facts, with_test=cfg.with_test)
    if args.json:
        print(json.dumps({
            "root": str(root), "platform": platform, "machine": facts.machine,
            "facts": {
                name: {"value": json.loads(json.dumps(
                           getattr(facts, name),
                           default=lambda v: _fact_value(root, v))),
                       "detects": spec.detects, "since": spec.since,
                       "gates": spec.gates}
                for name, spec in treefacts.FACTS.items()
            },
            "rules": verdicts,
        }, indent=2))
        return 0
    head = git(["rev-parse", "--short", "HEAD"], cwd=root, check=False) or "?"
    print(f"tree:      {root} @ {head}  ({platform}, {facts.machine})")
    print("facts:")
    width = max(map(len, treefacts.FACTS))
    for name, spec in treefacts.FACTS.items():
        since = f"  [{spec.since}]" if spec.since not in ("-", "") else ""
        print(f"  {name:<{width}}  {_fact_value(root, getattr(facts, name))}{since}")
    print("rules fired:")
    for v in verdicts:
        if not v["fired"] or v["reactive"]:
            continue
        if v["slot"]:
            detail = f"[{v['slot']}] {', '.join(v['selected'])}"
        elif v["keys"]:
            detail = f"keys: {', '.join(v['keys'])}"
        else:
            detail = "unkeyed: " + "; ".join(v["effects"])
        print(f"  {v['id']}@{v['rev']}  {detail}")
        print(f"      {v['reason']}")
    idle = [v["id"] for v in verdicts if not v["fired"]]
    print(f"not firing: {', '.join(idle) or 'none'}")
    print("reactive fallbacks (armed, in code):")
    for v in verdicts:
        if v["reactive"]:
            print(f"  {v['id']}@{v['rev']}  {'; '.join(v['effects'])}")
    return 0
