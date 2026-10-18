"""`ve errors`: the local incident log and agent notes."""

import gzip
import json
import shutil
import sys
import time
from pathlib import Path

from .. import errorlog
from ..config import Config
from ..log import die, say
from ..util import git


def _errors_since(cfg: Config, days: float) -> list[dict]:
    cutoff = time.time() - days * 86400
    out = []
    for rec in errorlog.read(cfg.cache_dir):
        try:
            ts = time.mktime(time.strptime(rec["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            out.append(rec)
    return out


def _one_line(text: str, width: int) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


def cmd_errors_summary(cfg: Config, args) -> int:
    recs = _errors_since(cfg, args.days)
    notes: dict[str, list[dict]] = {}
    groups: dict[str, list[dict]] = {}
    for rec in recs:
        if rec.get("level") == "note":
            notes.setdefault(rec.get("ref_fingerprint", ""), []).append(rec)
        elif args.level in ("all", rec.get("level")):
            groups.setdefault(rec["fingerprint"], []).append(rec)
    if args.json:
        print(json.dumps(
            [{"fingerprint": fp, "count": len(g), "level": g[-1]["level"],
              "first_seen": g[0]["ts"], "last_seen": g[-1]["ts"],
              "message": g[-1]["message"], "latest_id": g[-1]["id"],
              "notes": [n["message"] for n in notes.get(fp, [])]}
             for fp, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))],
            indent=2,
        ))
        return 0
    if not groups:
        say(f"no incidents in the last {args.days:g} days "
            f"({errorlog.log_path(cfg.cache_dir)})")
        return 0
    print(f"{'count':>5s}  {'level':7s} {'last seen':19s} {'notes':>5s}  "
          f"{'fingerprint':11s} message")
    for fp, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][-1]["ts"])):
        last = g[-1]
        print(f"{len(g):5d}  {last['level']:7s} {last['ts'][:19]:19s} "
              f"{len(notes.get(fp, [])):5d}  {fp:11s} {_one_line(last['message'], 90)}")
    print(f"\ndetails: `ve errors show <fingerprint|id>`  |  "
          f"log: {errorlog.log_path(cfg.cache_dir)}")
    return 0


def cmd_errors_list(cfg: Config, args) -> int:
    recs = _errors_since(cfg, args.days)[-args.limit:]
    if args.json:
        print(json.dumps(recs, indent=2))
        return 0
    for rec in recs:
        who = rec.get("agent") or "human"
        print(f"{rec['ts'][:19]}  {rec['id']}  {rec['level']:7s} {who:11s} "
              f"{' '.join(rec.get('argv', []))[:28]:28s} "
              f"{_one_line(rec['message'], 70)}")
    return 0


def cmd_errors_show(cfg: Config, args) -> int:
    recs = errorlog.read(cfg.cache_dir)
    match = [r for r in recs if args.key in (r.get("id"), r.get("fingerprint"))]
    if not match:
        die(f"no incident with id or fingerprint '{args.key}'")
    fps = {r["fingerprint"] for r in match if r.get("level") != "note"}
    ids = {r["id"] for r in match}
    related = [
        r for r in recs
        if r.get("level") == "note"
        and (r.get("ref") in ids or r.get("ref_fingerprint") in fps)
    ]
    shown = match[-args.limit:]
    if args.json:
        print(json.dumps({"incidents": shown, "notes": related}, indent=2))
        return 0
    for rec in shown:
        print(f"== {rec['id']}  {rec['level']}  {rec['ts']}  fingerprint={rec['fingerprint']}")
        print(f"message:  {rec['message']}")
        print(f"command:  {' '.join(rec.get('argv', []))}  (cwd {rec.get('cwd')})")
        if rec.get("repo"):
            print(f"repo:     {rec['repo']} @ {rec.get('head')}")
        print(f"ve:       {rec.get('ve_version')}  host={rec.get('host')}  "
              f"agent={rec.get('agent') or 'human'} {rec.get('agent_session', '')}")
        sub = rec.get("subprocess")
        if sub:
            print(f"failed:   {' '.join(sub['cmd'])}  (exit {sub['returncode']})")
            full_log = sub.get("full_log")
            if args.full and full_log and Path(full_log).exists():
                with gzip.open(full_log, "rt", encoding="utf-8") as f:
                    shutil.copyfileobj(f, sys.stdout)
            else:
                if sub.get("error_excerpt"):
                    print("errors:")
                    print("\n".join("  " + ln
                                    for ln in sub["error_excerpt"].splitlines()))
                if sub.get("output_tail"):
                    print("output tail:")
                    print("\n".join("  " + ln for ln in
                                    sub["output_tail"].splitlines()[-args.tail:]))
            if full_log:
                state = "" if Path(full_log).exists() else "  (pruned)"
                print(f"full log: {full_log}{state}  (`--full` to print)")
        if rec.get("traceback") and args.traceback:
            print(rec["traceback"].rstrip())
        print()
    if len(match) > len(shown):
        print(f"({len(match) - len(shown)} older occurrence(s) hidden; --limit to show)")
    for note in related:
        print(f"note {note['ts'][:19]} ({note.get('agent') or 'human'}): {note['message']}")
    return 0


def cmd_errors_note(cfg: Config, args) -> int:
    recs = [r for r in errorlog.read(cfg.cache_dir) if r.get("level") != "note"]
    if args.ref:
        target = next((r for r in reversed(recs)
                       if args.ref in (r.get("id"), r.get("fingerprint"))), None)
        if target is None:
            die(f"no incident with id or fingerprint '{args.ref}'")
    else:
        top = git(["rev-parse", "--show-toplevel"], check=False)
        here = [r for r in recs if not top or r.get("repo") == top]
        target = next((r for r in reversed(here) if r.get("level") == "error"),
                      here[-1] if here else None)
    errorlog.configure(cfg.cache_dir, ["ve", "errors", "note"])
    if not errorlog.enabled():
        die("incident log disabled (VE_NO_ERROR_LOG is set)")
    note_id = errorlog.append_note(args.message, target)
    if target:
        say(f"note {note_id} attached to {target['id']} "
            f"(fingerprint {target['fingerprint']}): {_one_line(target['message'], 70)}")
    else:
        say(f"note {note_id} recorded (no matching incident; pass --ref to link one)")
    return 0
