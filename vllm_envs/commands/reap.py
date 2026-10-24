"""`ve reap`: the worktree table and age-gated, safety-gated removal."""

import shutil
import sys
from pathlib import Path

from ..config import Config
from ..log import die, say, warn
from ..registry import live_envs, read_marker, unregister_env
from ..util import age_str, git, run
from ..worktrees import collect
from .common import record_drift_before_removal


def _reap_repo(cfg: Config) -> Path:
    if top := git(["rev-parse", "--show-toplevel"], check=False):
        return Path(top)
    for path in live_envs(cfg).values():
        repo = Path(read_marker(path).get("repo") or path)
        if (repo / ".git").exists():
            return repo
    die("not inside a git repo; run `ve reap` from a repo checkout")
    raise


def _worktree_labels(wts: list) -> dict[Path, str]:
    """Directory name, widened to parent/name (then the full path) when
    several worktrees share a directory name."""
    labels = {w.path: w.path.name for w in wts}
    for widen in (lambda p: f"{p.parent.name}/{p.name}", str):
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        for path, label in labels.items():
            if counts[label] > 1:
                labels[path] = widen(path)
    return labels


_WORKTREE_NAME_MAX = 40


def _print_worktrees(wts: list, max_age_days: float) -> None:
    # stale first (oldest at the top), then active ones, primary last
    wts = sorted(wts, key=lambda w: (
        w.state == "primary", not w.is_candidate,
        (w.activity or 0) if w.is_candidate else -(w.activity or 0),
    ))
    labels = _worktree_labels(wts)
    rows = []
    n_safe = n_cand = 0
    for w in wts:
        if w.is_candidate:
            n_cand += 1
            safe, reason = w.safety()
            n_safe += safe
            reap = "SAFE" if safe else f"blocked:{reason.split()[0]}"
        else:
            reap = "-"
        rows.append((
            labels[w.path], w.state, age_str(w.activity),
            "clean" if not w.dirty else f"{w.dirty} dirty",
            w.sync, reap, w.branch,
        ))

    header = ("worktree", "state", "age", "tree", "sync", "reap", "branch")
    right = {2, 3}
    widths = [max(len(r[i]) for r in [header, *rows])
              for i in range(len(header) - 1)]
    branch_room = None
    if sys.stdout.isatty():
        widths[0] = min(widths[0], _WORKTREE_NAME_MAX)
        used = sum(widths) + 2 * len(widths)
        branch_room = max(shutil.get_terminal_size().columns - used, 20)

    def fmt(row: tuple) -> str:
        name = row[0]
        if len(name) > widths[0]:
            name = name[: widths[0] - 1] + "…"
        cells = [
            cell.rjust(widths[i]) if i in right else cell.ljust(widths[i])
            for i, cell in enumerate((name, *row[1:-1]))
        ]
        branch = row[-1]
        if branch_room is not None and len(branch) > branch_room:
            branch = branch[: branch_room - 1] + "…"
        return "  ".join([*cells, branch])

    print(fmt(header))
    for row in rows:
        print(fmt(row))
    print(f"\n{n_cand} stale (inactive > {max_age_days:g}d; {n_safe} safe to reap). "
          "reap all safe: `ve reap --stale`  |  one: `ve reap <name>`  |  "
          "override gate: --force")


def _reap_one(cfg: Config, repo: Path, w, envs: dict, dry_run: bool) -> bool:
    label = f"{w.path.name} [{w.state}] {w.branch} (last active {age_str(w.activity)})"
    if dry_run:
        say(f"would reap {label}")
        return True
    say(f"reaping {label}")
    if w.path.exists():
        record_drift_before_removal(w.path)
        rm = run(["git", "worktree", "remove", "--force", str(w.path)],
                 cwd=repo, check=False)
        if rm.returncode != 0:
            # git refusing is a verdict (e.g. locked), not a cue to rmtree
            warn(f"git refused to remove {w.path}: {rm.stderr.strip()}", log=False)
            return False
    for name, path in envs.items():
        if path.resolve() == w.path.resolve():
            unregister_env(cfg, name)
    run(["git", "worktree", "prune"], cwd=repo, check=False)
    return True


def cmd_reap(cfg: Config, args) -> int:
    repo = _reap_repo(cfg)
    max_age = (args.older_than if args.older_than is not None
               else cfg.reap_max_age_days)
    wts = collect(repo, max_age_days=max_age)
    if not args.targets and not args.stale:
        _print_worktrees(wts, max_age)
        return 0

    if args.stale:
        selected = [w for w in wts if w.is_candidate]
    else:
        labels = _worktree_labels(wts)
        index: dict[str, object] = {}
        for w in wts:
            index[labels[w.path]] = w
            index[str(w.path)] = w
        selected = []
        for t in args.targets:
            w = index.get(t) or index.get(str(Path(t).expanduser().resolve()))
            if w is None:
                prefix = t.rstrip("…")
                matches = {id(v): v for k, v in index.items()
                           if k.startswith(prefix)}
                if len(matches) == 1:
                    w = next(iter(matches.values()))
                elif len(matches) > 1:
                    warn(f"'{t}' matches {len(matches)} worktrees; "
                         "use a longer name or the full path", log=False)
                    continue
            if w is None:
                warn(f"no worktree matching '{t}'", log=False)
            else:
                selected.append(w)

    envs = live_envs(cfg)
    reaped = 0
    for w in selected:
        if not w.is_candidate and not args.force:
            why = ("the primary checkout" if w.state == "primary"
                   else f"active {age_str(w.activity)} ago (< {max_age:g}d)")
            warn(f"skip {w.path.name} — {why}; use --force to override", log=False)
            continue
        if w.locked:
            warn(f"skip {w.path.name} — locked; `git worktree unlock` it first",
                 log=False)
            continue
        safe, reason = w.safety()
        if not safe and not args.force:
            warn(f"skip {w.path.name} — {reason}; commit/push or use --force",
                 log=False)
            continue
        reaped += _reap_one(cfg, repo, w, envs, dry_run=args.dry_run)
    verb = "would reap" if args.dry_run else "reaped"
    say(f"{verb} {reaped} worktree(s)")
    return 0
