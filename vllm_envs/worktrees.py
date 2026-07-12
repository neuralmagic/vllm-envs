"""Worktree inventory and safe reaping.

t3code creates a git worktree per session under ~/.t3/worktrees/<repo>/ and
records each in its state DB (projection_threads: worktree_path, archived_at,
deleted_at). When a session is archived the worktree is left on disk. This
module cross-references git, that DB, and the ve registry so `ve reap` can list
worktrees with enough context to prune them by hand — and refuses to delete
anything with un-pushed or uncommitted work unless forced.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .util import run

T3_DB = Path.home() / ".t3" / "userdata" / "state.sqlite"
T3_WORKTREES = Path.home() / ".t3" / "worktrees"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _t3_states(db: Path = T3_DB) -> dict[str, dict]:
    """Map worktree_path -> {state, activity, branch, title} from t3's DB.

    A path may back several threads; it counts as active if any live thread
    uses it, else archived if any archived, else deleted.
    """
    if not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = con.execute(
            "select worktree_path, branch, title, updated_at, archived_at, "
            "deleted_at from projection_threads where worktree_path is not null"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()

    out: dict[str, dict] = {}
    rank = {"active": 2, "archived": 1, "deleted": 0}
    for path, branch, title, updated, archived, deleted in rows:
        state = "deleted" if deleted else ("archived" if archived else "active")
        activity = _iso_to_epoch(updated) or 0.0
        cur = out.get(path)
        if cur is None or activity > cur["activity"]:
            out[path] = {"state": state, "activity": activity,
                         "branch": branch or "", "title": title or ""}
        # a live thread wins the state regardless of which row is newest
        if rank[state] > rank[out[path]["state"]]:
            out[path]["state"] = state
    return out


@dataclass
class Worktree:
    path: Path
    branch: str
    head: str
    state: str  # active | archived | deleted | orphan | external | primary
    activity: float | None  # epoch of most recent git/t3 touch
    title: str
    dirty: int  # count of `git status --porcelain` lines
    sync: str  # pushed | merged | ahead:N | local | unknown

    @property
    def is_candidate(self) -> bool:
        """A t3 worktree whose session is gone — eligible for reaping."""
        return self.state in ("archived", "deleted", "orphan")

    def safety(self) -> tuple[bool, str]:
        """Conservative gate: clean tree AND pushed-or-merged."""
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


def _classify(path: Path, primary: Path, t3: dict) -> tuple[str, dict]:
    if path == primary:
        return "primary", {}
    key = str(path)
    if key in t3:
        return t3[key]["state"], t3[key]
    try:
        path.relative_to(T3_WORKTREES)
        return "orphan", {}
    except ValueError:
        return "external", {}


def collect(repo: Path) -> list[Worktree]:
    """Inventory every worktree of `repo`, enriched with t3 + git state."""
    t3 = _t3_states()
    main_ref = _main_ref(repo)
    entries = _porcelain_worktrees(repo)
    primary = Path(entries[0]["path"]) if entries else repo
    out: list[Worktree] = []
    for e in entries:
        path = Path(e["path"])
        state, meta = _classify(path, primary, t3)
        exists = path.is_dir()
        dirty = 0
        sync = "unknown"
        git_activity = None
        if exists and state != "primary":
            porc = run(["git", "status", "--porcelain"], cwd=path, check=False)
            dirty = len([ln for ln in porc.stdout.splitlines() if ln.strip()])
            sync = _sync_state(path, e["head"], main_ref)
            ct = run(["git", "log", "-1", "--format=%ct"], cwd=path,
                     check=False).stdout.strip()
            git_activity = float(ct) if ct.isdigit() else None
        activity = max(
            [t for t in (meta.get("activity"), git_activity) if t], default=None
        )
        out.append(Worktree(
            path=path,
            branch=e["branch"] or meta.get("branch", ""),
            head=e["head"],
            state=state,
            activity=activity,
            title=meta.get("title", ""),
            dirty=dirty,
            sync=sync,
        ))
    return out


def age_str(epoch: float | None) -> str:
    if not epoch:
        return "?"
    h = (time.time() - epoch) / 3600
    if h < 48:
        return f"{h:.0f}h"
    return f"{h / 24:.0f}d"
