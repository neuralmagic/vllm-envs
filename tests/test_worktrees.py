import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from vllm_envs.config import load_config
from vllm_envs.worktrees import collect


def git(cwd: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
        env={**os.environ, **(env or {})},
    ).stdout.strip()


class AgeBasedReapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t",
            "commit", "-q", "--allow-empty", "-m", "init")

    def add_worktree(self, name: str, age_days: float) -> Path:
        path = self.tmp / name
        git(self.repo, "worktree", "add", "-q", "--detach", str(path))
        old = time.time() - age_days * 86400
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(old))
        git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
            "--allow-empty", "-m", name,
            env={"GIT_COMMITTER_DATE": stamp, "GIT_AUTHOR_DATE": stamp})
        git_dir = Path(git(path, "rev-parse", "--absolute-git-dir"))
        log = git_dir / "logs" / "HEAD"
        lines = log.read_text().splitlines()
        log.write_text("".join(
            re.sub(r" \d+ ([+-]\d{4})\t", f" {int(old)} \\1\t", ln) + "\n"
            for ln in lines
        ))
        for f in ("HEAD", "index"):
            os.utime(git_dir / f, (old, old))
        return path

    def states(self, max_age_days: float) -> dict[str, bool]:
        return {w.path.name: w.is_candidate
                for w in collect(self.repo, max_age_days=max_age_days)}

    def test_stale_is_purely_inactivity_past_the_cutoff(self):
        self.add_worktree("old", 45)
        self.add_worktree("recent", 3)

        self.assertEqual(self.states(30), {"repo": False, "old": True, "recent": False})
        self.assertEqual(self.states(2), {"repo": False, "old": True, "recent": True})

    def test_checkout_of_old_commit_counts_as_recent_activity(self):
        path = self.add_worktree("old", 45)
        git(path, "checkout", "-q", "--detach", "HEAD~1")
        self.assertFalse(self.states(30)["old"])

    def test_reflog_rewrite_by_gc_does_not_count_as_activity(self):
        path = self.add_worktree("old", 45)
        git_dir = Path(git(path, "rev-parse", "--absolute-git-dir"))
        os.utime(git_dir / "logs" / "HEAD")
        os.utime(git_dir / "index")
        self.assertTrue(self.states(30)["old"])

    def test_listing_does_not_refresh_activity(self):
        path = self.add_worktree("old", 45)
        (path / "untracked").write_text("x")
        collect(self.repo, max_age_days=30)
        self.assertTrue(self.states(30)["old"])

    def test_missing_worktree_dir_is_stale_and_safe(self):
        path = self.add_worktree("gone", 1)
        subprocess.run(["rm", "-rf", str(path)], check=True)
        (w,) = [w for w in collect(self.repo) if w.path.name == "gone"]
        self.assertEqual(w.state, "missing")
        self.assertTrue(w.is_candidate)
        self.assertEqual(w.safety(), (True, "directory gone"))

    def test_cutoff_is_configurable(self):
        cache = self.tmp / "cache"
        cache.mkdir()
        old = os.environ.get("VE_CACHE_DIR")
        self.addCleanup(lambda: os.environ.__setitem__("VE_CACHE_DIR", old)
                        if old else os.environ.pop("VE_CACHE_DIR", None))
        os.environ["VE_CACHE_DIR"] = str(cache)
        os.environ.pop("VE_REAP_MAX_AGE_DAYS", None)
        self.assertEqual(load_config().reap_max_age_days, 30.0)
        (cache / "config.toml").write_text("[reap]\nmax_age_days = 14\n")
        self.assertEqual(load_config().reap_max_age_days, 14.0)
        os.environ["VE_REAP_MAX_AGE_DAYS"] = "7"
        self.addCleanup(os.environ.pop, "VE_REAP_MAX_AGE_DAYS", None)
        self.assertEqual(load_config().reap_max_age_days, 7.0)


if __name__ == "__main__":
    unittest.main()
