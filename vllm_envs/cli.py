"""`ve` argument parsing and entry point; commands live in vllm_envs.commands."""

import argparse
import sys

from . import errorlog
from .commands import cache, diagnostics, envs, errors, hook, reap
from .commands import drift as drift_cmd
from .config import STORE_NAMES, load_config
from .log import warn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ve", description="Fast disposable vLLM dev environments"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="create env from a ref (worktree + cached layers)")
    sp.add_argument("ref")
    sp.add_argument("--name", "-n")
    sp.add_argument("--repo", help="path to the vLLM clone (default: cwd)")
    sp.set_defaults(func=envs.cmd_new)

    sp = sub.add_parser(
        "init", help="manage the current checkout/worktree in place (build + venv)"
    )
    sp.add_argument("--name", "-n", help="env name (default: directory name)")
    sp.set_defaults(func=envs.cmd_init)

    sp = sub.add_parser(
        "activate",
        help="activate the env venv in the current shell "
             "(needs eval \"$(ve shellenv)\" in your rc)",
    )
    sp.set_defaults(func=envs.cmd_activate)

    sp = sub.add_parser(
        "shellenv",
        help="print shell function enabling `ve activate`; "
             "add to your rc: eval \"$(ve shellenv)\"",
    )
    sp.set_defaults(func=envs.cmd_shellenv)

    sp = sub.add_parser("sync", help="re-resolve layers for the current worktree HEAD")
    sp.add_argument("--fresh-venv", action="store_true",
                    help="discard env venv divergence, re-clone from template")
    sp.set_defaults(func=envs.cmd_sync)

    sp = sub.add_parser("rm", help="remove an env (never shared caches)")
    sp.add_argument("name")
    sp.add_argument("--force", action="store_true",
                    help="remove a `ve new` env even with uncommitted/unpushed work")
    sp.set_defaults(func=envs.cmd_rm)

    sp = sub.add_parser("list", help="list envs")
    sp.set_defaults(func=envs.cmd_list)

    sp = sub.add_parser("status", help="show env layer hashes and cache state")
    sp.set_defaults(func=envs.cmd_status)

    sp = sub.add_parser(
        "reap",
        help="list worktrees; reap ones inactive for longer than a cutoff",
    )
    sp.add_argument("targets", nargs="*", help="worktree name(s) or path(s) to reap")
    sp.add_argument("--stale", action="store_true",
                    help="reap all stale worktrees that pass the safety gate")
    sp.add_argument("--older-than", type=float, metavar="DAYS",
                    help="staleness cutoff in days (default: [reap] max_age_days "
                         "in config.toml, VE_REAP_MAX_AGE_DAYS, or 30)")
    sp.add_argument("--force", action="store_true",
                    help="bypass the clean+pushed safety gate")
    sp.add_argument("--dry-run", action="store_true", help="show what would be reaped")
    sp.set_defaults(func=reap.cmd_reap)

    sp = sub.add_parser("gc", help="prune caches (LRU, 100GB default cap)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--free", type=float, metavar="GB",
                    help="evict until this many GB are reclaimed")
    sp.set_defaults(func=cache.cmd_gc)

    sp = sub.add_parser(
        "du", help="cache usage audit: per store, uv cache/ccache, live envs"
    )
    sp.add_argument("--entries", "-e", action="store_true",
                    help="also list per-entry sizes with age and origin")
    sp.set_defaults(func=cache.cmd_du)

    sp = sub.add_parser(
        "drift",
        help="packages/files changed by hand in env venvs since ve set them up",
    )
    sp.add_argument("env", nargs="?",
                    help="env name or path for details (default: table of all envs)")
    sp.add_argument("--all-envs", action="store_true",
                    help="include envs without drift in the table")
    sp.add_argument("--record", action="store_true",
                    help="also record drift in the incident log (`ve errors`)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=drift_cmd.cmd_drift)

    sp = sub.add_parser(
        "precompiled",
        help="show which upstream wheel matches this worktree's compiled sources "
             "(read-only, no download)",
    )
    sp.add_argument("path", nargs="?", help="worktree (default: cwd)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=diagnostics.cmd_precompiled)

    sp = sub.add_parser(
        "rules",
        help="explain this worktree's vLLM-version facts and which rules fire "
             "(read-only)",
    )
    sp.add_argument("path", nargs="?", help="worktree (default: cwd)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=diagnostics.cmd_rules)

    sp = sub.add_parser("pin", help="pin/unpin a store entry (exempt from gc)")
    sp.add_argument("entry", help="<store>/<hash>")
    sp.add_argument("--unpin", action="store_true")
    sp.set_defaults(func=cache.cmd_pin)

    sp = sub.add_parser(
        "errors",
        help="local incident log: recurring ve failures/fallbacks and agent notes",
    )
    esub = sp.add_subparsers(dest="errors_cmd")
    sp.set_defaults(func=errors.cmd_errors_summary, days=30.0, level="all", json=False)
    sp.add_argument("--days", type=float, default=30.0)
    sp.add_argument("--level", choices=("all", "error", "warning", "drift"), default="all")
    sp.add_argument("--json", action="store_true")

    ep = esub.add_parser("list", help="individual incidents, newest last")
    ep.add_argument("--days", type=float, default=30.0)
    ep.add_argument("--limit", type=int, default=50)
    ep.add_argument("--json", action="store_true")
    ep.set_defaults(func=errors.cmd_errors_list)

    ep = esub.add_parser("show", help="details for an incident id or fingerprint")
    ep.add_argument("key")
    ep.add_argument("--limit", type=int, default=3)
    ep.add_argument("--traceback", "-t", action="store_true")
    ep.add_argument("--tail", type=int, default=40, metavar="N",
                    help="lines of the stored output tail to print")
    ep.add_argument("--full", action="store_true",
                    help="print the complete saved output of the failed command")
    ep.add_argument("--json", action="store_true")
    ep.set_defaults(func=errors.cmd_errors_show)

    ep = esub.add_parser(
        "note",
        help="record what you did about an incident (default: latest error "
             "in this repo, else latest warning)",
    )
    ep.add_argument("message")
    ep.add_argument("--ref", help="incident id or fingerprint to attach to")
    ep.set_defaults(func=errors.cmd_errors_note)

    sp = sub.add_parser("hook", help=argparse.SUPPRESS)
    sp.add_argument("event")
    sp.add_argument("rest", nargs="*")
    sp.set_defaults(func=hook.cmd_hook)

    args = p.parse_args(argv)
    cfg = load_config()
    errorlog.configure(
        None if args.cmd == "errors" else cfg.cache_dir,
        ["ve", *(sys.argv[1:] if argv is None else argv)],
    )
    for store in STORE_NAMES:
        cfg.store(store).mkdir(parents=True, exist_ok=True)
    try:
        return args.func(cfg, args)
    except KeyboardInterrupt:
        warn("interrupted", log=False)
        return 130
    except Exception as e:
        errorlog.record("error", f"unhandled {type(e).__name__}: {e}", e)
        raise


if __name__ == "__main__":
    sys.exit(main())
