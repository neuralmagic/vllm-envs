"""Worktree inventory and safe, age-based reaping.

Every linked worktree of a repo is a reap candidate once it has been inactive
for longer than the configured age. `ve reap` refuses to delete anything with
un-pushed or uncommitted work unless forced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .util import git, git_path, run

# Listing must not write to the worktrees: plain `git status` refreshes the
# index of every worktree it inspects.
_NO_LOCKS = {"GIT_OPTIONAL_LOCKS": "0"}


@dataclass
class Worktree:
    path: Path
    branch: str
    head: str
    state: str  # primary | worktree | missing
    activity: float | None  # epoch of the most recent git activity
    dirty: int  # count of `git status --porcelain` lines
    sync: str  # pushed | merged | ahead:N | local | unknown
    max_age_s: float = 30 * 86400
    locked: bool = False  # `git worktree lock`: never reaped, even with --force

    @property
    def is_candidate(self) -> bool:
        """Eligible for reaping: a linked worktree inactive past the cutoff."""
        if self.state == "missing":
            return True
        if self.state != "worktree":
            return False
        return self.activity is None or time.time() - self.activity > self.max_age_s

    def safety(self) -> tuple[bool, str]:
        """Conservative gate: clean tree AND pushed-or-merged."""
        if self.locked:
            return False, "locked (git worktree unlock to allow)"
        if self.state == "missing":
            return True, "directory gone"
        if self.dirty:
            return False, f"dirty ({self.dirty} uncommitted)"
        if self.sync in ("pushed", "merged"):
            return True, self.sync
        if self.sync.startswith("ahead"):
            return False, f"unpushed ({self.sync})"
        return False, "unpushed commits"


def _porcelain_worktrees(repo: Path) -> list[dict]:
    proc = run(["git", "worktree", "list", "--porcelain"], cwd=repo, check=False)
    entries: list[dict] = []
    cur: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            if cur:
                entries.append(cur)
            cur = {"path": line[len("worktree "):], "branch": "", "head": ""}
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):][:12]
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):].replace("refs/heads/", "")
        elif line == "detached":
            cur["branch"] = "(detached)"
        elif line == "bare":
            cur["branch"] = "(bare)"
        elif line == "locked" or line.startswith("locked "):
            cur["locked"] = True
    if cur:
        entries.append(cur)
    return entries


def _main_ref(repo: Path) -> str:
    for ref in ("origin/main", "origin/master", "main", "master"):
        if run(["git", "rev-parse", "--verify", "--quiet", ref],
               cwd=repo, check=False).returncode == 0:
            return ref
    return "HEAD"


def _sync_state(wt: Path, head: str, main_ref: str) -> str:
    if head in ("", "(bare)"):
        return "unknown"
    # commits reachable from HEAD but from no remote-tracking ref == truly unpushed
    unpushed = run(["git", "rev-list", "--count", "HEAD", "--not", "--remotes"],
                   cwd=wt, check=False).stdout.strip()
    if unpushed == "0":
        return "pushed"
    if run(["git", "merge-base", "--is-ancestor", "HEAD", main_ref],
           cwd=wt, check=False).returncode == 0:
        return "merged"
    return f"ahead:{unpushed}" if unpushed.isdigit() else "local"


def removal_safety(path: Path) -> tuple[bool, str]:
    """Same gate as `ve reap` for one worktree: clean and pushed-or-merged."""
    if not path.is_dir():
        return True, "directory gone"
    porc = run(["git", "status", "--porcelain"], cwd=path, check=False, env=_NO_LOCKS)
    dirty = len([ln for ln in porc.stdout.splitlines() if ln.strip()])
    head = git(["rev-parse", "HEAD"], cwd=path, check=False)
    wt = Worktree(path=path, branch="", head=head, state="worktree", activity=None,
                  dirty=dirty, sync=_sync_state(path, head, _main_ref(path)))
    return wt.safety()


def _last_reflog_time(log: Path) -> float | None:
    """Timestamp of the newest reflog entry. The file's mtime is no good:
    `git gc` expires reflogs and rewrites every worktree's log at once."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - 4096, 0))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        # "<old> <new> Name <email> <epoch> <tz>\t<message>"
        fields = line.split("\t", 1)[0].rsplit(" ", 2)
        if len(fields) == 3 and fields[1].isdigit():
            return float(fields[1])
    return None


def _last_activity(wt: Path) -> float | None:
    """Latest of the HEAD commit time, the newest HEAD reflog entry, and the
    worktree's HEAD file mtime, so checking out an old commit still counts
    as recent use."""
    times: list[float] = []
    ct = git(["log", "-1", "--format=%ct"], cwd=wt, check=False)
    if ct.isdigit():
        times.append(float(ct))
    git_dir = git_path(wt, "--absolute-git-dir", check=False)
    if git_dir is not None:
        reflog = _last_reflog_time(git_dir / "logs" / "HEAD")
        if reflog is not None:
            times.append(reflog)
        try:
            times.append((git_dir / "HEAD").stat().st_mtime)
        except OSError:
            pass
    return max(times, default=None)


def collect(repo: Path, max_age_days: float = 30.0) -> list[Worktree]:
    """Inventory every worktree of `repo` with its git state and age."""
    main_ref = _main_ref(repo)
    entries = _porcelain_worktrees(repo)
    primary = Path(entries[0]["path"]) if entries else repo
    out: list[Worktree] = []
    for e in entries:
        path = Path(e["path"])
        if path == primary:
            state = "primary"
        elif not path.is_dir():
            state = "missing"
        else:
            state = "worktree"
        dirty = 0
        sync = "unknown"
        activity = None
        if state == "worktree":
            porc = run(["git", "status", "--porcelain"], cwd=path, check=False,
                       env=_NO_LOCKS)
            dirty = len([ln for ln in porc.stdout.splitlines() if ln.strip()])
            sync = _sync_state(path, e["head"], main_ref)
            activity = _last_activity(path)
        out.append(Worktree(
            path=path,
            branch=e["branch"],
            head=e["head"],
            state=state,
            activity=activity,
            dirty=dirty,
            sync=sync,
            max_age_s=max_age_days * 86400,
            locked=e.get("locked", False),
        ))
    return out
