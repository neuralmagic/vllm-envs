import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.config import MARKER_NAME
from vllm_envs.hooks import install_hook


class GitLifecycleIntegrationTest(unittest.TestCase):
    """Exercise the installed hooks through Git, replacing only costly sync."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.root / "repo"
        self.worktree = self.root / "feature"
        self.cache = self.root / "cache"
        self.sync_log = self.root / "sync.jsonl"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.repo.mkdir()

        package_root = Path(__file__).resolve().parents[1]
        ve = self.bin / "ve"
        ve.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                import json
                import os
                import subprocess
                from pathlib import Path

                import vllm_envs.hooks
                from vllm_envs.cli import main
                from vllm_envs.registry import update_marker

                def record_sync(cfg, root, *args, **kwargs):
                    root = Path(root)
                    head = subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=root,
                        check=True, capture_output=True, text=True,
                    ).stdout.strip()
                    with Path(os.environ["VE_TEST_SYNC_LOG"]).open("a") as f:
                        f.write(json.dumps({"root": str(root), "head": head}) + "\\n")
                    update_marker(root, synced_head=head)

                vllm_envs.hooks.sync = record_sync
                raise SystemExit(main())
                """
            )
        )
        ve.chmod(0o755)

        env = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": f"{package_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "VE_CACHE_DIR": str(self.cache),
            "VE_NO_AUTO_INIT": "",
            "VE_TEST_SYNC_LOG": str(self.sync_log),
        }
        self.enterContext(patch.dict(os.environ, env))

        self.git(self.repo, "init", "-b", "main")
        self.git(self.repo, "config", "user.name", "Test User")
        self.git(self.repo, "config", "user.email", "test@example.com")
        (self.repo / "setup.py").write_text("# initial build input\n")
        requirements = self.repo / "requirements"
        requirements.mkdir()
        (requirements / "common.txt").write_text("example==1\n")
        (self.repo / "vllm.py").write_text("VERSION = 1\n")
        self.git(self.repo, "add", ".")
        self.git(self.repo, "commit", "-m", "initial")
        self.initial_head = self.head(self.repo)
        install_hook(self.repo)

    def git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, env=os.environ.copy(), check=False,
            capture_output=True, text=True,
        )
        if proc.returncode:
            self.fail(
                f"git {' '.join(args)} failed ({proc.returncode})\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc

    def head(self, cwd: Path) -> str:
        return self.git(cwd, "rev-parse", "HEAD").stdout.strip()

    def syncs(self) -> list[dict[str, str]]:
        if not self.sync_log.exists():
            return []
        return [json.loads(line) for line in self.sync_log.read_text().splitlines()]

    def test_worktree_edits_commits_and_checkouts_stay_in_sync(self):
        added = self.git(
            self.repo, "worktree", "add", "-b", "feature", str(self.worktree)
        )

        self.assertTrue(
            (self.worktree / MARKER_NAME).exists(),
            f"worktree hook did not initialize env; stderr:\n{added.stderr}",
        )
        self.assertEqual(self.syncs(), [{"root": str(self.worktree), "head": self.initial_head}])

        # Ordinary source edits do not invalidate cached dependency/build layers.
        (self.worktree / "vllm.py").write_text("VERSION = 2\n")
        self.git(self.worktree, "add", "vllm.py")
        self.git(self.worktree, "commit", "-m", "edit source")
        self.assertEqual(len(self.syncs()), 1)

        # A committed requirements edit must immediately converge this worktree.
        (self.worktree / "requirements" / "common.txt").write_text("example==2\n")
        self.git(self.worktree, "add", "requirements/common.txt")
        self.git(self.worktree, "commit", "-m", "edit requirements")
        requirements_head = self.head(self.worktree)
        self.assertEqual(self.syncs()[-1], {
            "root": str(self.worktree), "head": requirements_head,
        })

        # Hopping back across that commit must converge to the checked-out HEAD.
        self.git(self.worktree, "checkout", "--detach", self.initial_head)
        self.assertEqual(self.syncs()[-1], {
            "root": str(self.worktree), "head": self.initial_head,
        })
        marker = (self.worktree / MARKER_NAME).read_text()
        self.assertIn(f'synced_head = "{self.initial_head}"', marker)

    def commit_requirements(self, cwd: Path, version: int) -> str:
        (cwd / "requirements" / "common.txt").write_text(f"example=={version}\n")
        self.git(cwd, "commit", "-qam", f"requirements v{version}")
        return self.head(cwd)

    def test_every_bisect_checkout_syncs_the_env(self):
        self.git(self.repo, "worktree", "add", "-b", "feature", str(self.worktree))
        middle = self.commit_requirements(self.worktree, 2)
        tip = self.commit_requirements(self.worktree, 3)

        self.git(self.worktree, "bisect", "start", tip, self.initial_head)
        self.assertEqual(self.syncs()[-1]["head"], middle)
        self.git(self.worktree, "bisect", "reset")
        self.assertEqual(self.syncs()[-1]["head"], tip)

    def test_merge_of_requirements_change_syncs_the_env(self):
        self.git(self.repo, "worktree", "add", "-b", "feature", str(self.worktree))
        self.git(self.repo, "checkout", "-q", "-b", "upstream")
        merged = self.commit_requirements(self.repo, 5)
        self.git(self.repo, "checkout", "-q", "main")
        before = len(self.syncs())

        self.git(self.worktree, "merge", "-q", "--ff-only", "upstream")

        self.assertEqual(len(self.syncs()), before + 1)
        self.assertEqual(self.syncs()[-1]["head"], merged)

    def test_chained_post_rewrite_hook_and_ve_both_get_stdin(self):
        hooks = Path(self.git(self.repo, "rev-parse", "--git-path", "hooks").stdout.strip())
        hooks = hooks if hooks.is_absolute() else self.repo / hooks
        seen = self.root / "chained-stdin"
        chained = hooks / "post-rewrite.pre-ve"
        chained.write_text(f"#!/bin/sh\ncat > {seen}\n")
        chained.chmod(0o755)
        self.git(self.repo, "worktree", "add", "-b", "feature", str(self.worktree))
        self.commit_requirements(self.worktree, 2)
        before = len(self.syncs())

        (self.worktree / "requirements" / "common.txt").write_text("example==9\n")
        self.git(self.worktree, "commit", "-qa", "--amend", "--no-edit")

        self.assertTrue(seen.read_text().strip())
        self.assertGreater(len(self.syncs()), before)

    def test_hooks_install_into_core_hooks_path(self):
        custom = self.root / "custom-hooks"
        self.git(self.repo, "config", "core.hooksPath", str(custom))
        install_hook(self.repo)
        self.assertTrue((custom / "post-checkout").exists())
        self.assertTrue((custom / "post-merge").exists())

    def test_rm_never_deletes_an_auto_initialized_worktree(self):
        from vllm_envs.cli import main
        from vllm_envs.registry import read_marker

        self.git(self.repo, "worktree", "add", "-b", "feature", str(self.worktree))
        (self.worktree / "wip.py").write_text("unsaved = True\n")
        name = read_marker(self.worktree)["name"]
        self.assertEqual(read_marker(self.worktree)["origin"], "auto")

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["rm", name]), 0)

        self.assertTrue((self.worktree / "wip.py").exists())
        self.assertFalse((self.worktree / MARKER_NAME).exists())

    def test_rm_of_ve_new_env_requires_force_when_dirty(self):
        from vllm_envs.cli import main
        from vllm_envs.config import load_config
        from vllm_envs.registry import register_env, write_marker

        envs_root = self.root / "envs"
        envs_root.mkdir()
        dest = envs_root / "scratch"
        self.git(self.repo, "worktree", "add", "--detach", str(dest))
        write_marker(dest, {"name": "scratch", "repo": str(self.repo), "origin": "new"})
        (dest / "wip.py").write_text("unsaved = True\n")
        with patch.dict(os.environ, {"VE_ENVS_ROOT": str(envs_root)}):
            register_env(load_config(), "scratch", dest, self.repo)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["rm", "scratch"])
                self.assertTrue((dest / "wip.py").exists())
                self.assertEqual(main(["rm", "scratch", "--force"]), 0)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
