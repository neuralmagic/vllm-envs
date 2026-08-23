"""Opt-in tests that perform real vLLM, CUDA, and DeepEP builds."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from vllm_envs.config import MARKER_NAME
from vllm_envs.registry import read_marker


class RealVllmLifecycleTest(unittest.TestCase):
    def setUp(self):
        if os.environ.get("VE_REAL_E2E") != "1":
            self.skipTest("set VE_REAL_E2E=1 to run real build/install tests")
        source = os.environ.get("VE_REAL_E2E_REPO")
        if not source:
            self.skipTest("VE_REAL_E2E_REPO must point to a vLLM Git checkout")
        if shutil.which("nvidia-smi") is None:
            self.skipTest("real CUDA/DeepEP E2E requires nvidia-smi")

        self.source = Path(source).expanduser().resolve()
        if not (self.source / "setup.py").is_file():
            self.fail(f"not a vLLM checkout: {self.source}")

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.root / "repo"
        self.worktree = self.root / "worktree"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        cache_override = os.environ.get("VE_REAL_E2E_CACHE_DIR")
        self.cache = (
            Path(cache_override).expanduser().resolve()
            if cache_override
            else self.root / "cache"
        )

        package_root = Path(__file__).resolve().parents[1]
        ve = self.bin / "ve"
        ve.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                from vllm_envs.cli import main
                raise SystemExit(main())
                """
            )
        )
        ve.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": f"{self.bin}{os.pathsep}{self.env['PATH']}",
                "PYTHONPATH": (
                    f"{package_root}{os.pathsep}{self.env.get('PYTHONPATH', '')}"
                ),
                "VE_CACHE_DIR": str(self.cache),
                "VE_ENVS_ROOT": str(self.root / "envs"),
                "VE_NO_AUTO_INIT": "",
                "VE_NO_SYNC": "",
                "VE_WITH_VLLM_EXTRAS": "1",
                "VE_WITH_TEST": os.environ.get("VE_REAL_E2E_WITH_TEST", "1"),
            }
        )

        self.execute("git", "clone", "--shared", str(self.source), str(self.repo))
        ref = os.environ.get("VE_REAL_E2E_REF", "HEAD")
        self.execute("git", "checkout", "--detach", ref, cwd=self.repo)
        self.execute("git", "config", "user.name", "ve real E2E", cwd=self.repo)
        self.execute(
            "git", "config", "user.email", "ve-e2e@example.invalid", cwd=self.repo
        )

    def execute(self, *cmd: str, cwd: Path | None = None, stream: bool = False):
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=self.env,
            check=True,
            text=True,
            capture_output=not stream,
        )

    def assert_imports(self, worktree: Path) -> None:
        python = worktree / ".venv" / "bin" / "python"
        self.execute(
            str(python),
            "-c",
            "import vllm; import deep_ep",
            cwd=worktree,
            stream=True,
        )

    def test_real_build_deepep_worktree_commit_and_checkout(self):
        # Cold mode performs every package install and build. With a persistent
        # VE_REAL_E2E_CACHE_DIR this also exercises the genuine cache-hit path.
        self.execute("ve", "init", cwd=self.repo, stream=True)
        initial_head = self.execute(
            "git", "rev-parse", "HEAD", cwd=self.repo
        ).stdout.strip()

        self.execute(
            "git",
            "worktree",
            "add",
            "-b",
            "ve-real-e2e",
            str(self.worktree),
            cwd=self.repo,
            stream=True,
        )
        self.assertTrue((self.worktree / MARKER_NAME).exists())
        initial = read_marker(self.worktree)
        self.assertTrue(initial.get("venv_full_hash"))
        self.assertTrue(initial.get("build_hash"))
        self.assertTrue(initial.get("deepep_hash"))
        deep_entry = self.cache / "ep-kernels" / initial["deepep_hash"]
        self.assertTrue((deep_entry / ".complete").exists())
        self.assert_imports(self.worktree)

        # setup.py is a watched build input. Committing this harmless edit must
        # run a real extension build through post-commit.
        setup = self.worktree / "setup.py"
        setup.write_text(setup.read_text() + "\n# ve real E2E rebuild\n")
        self.execute("git", "add", "setup.py", cwd=self.worktree)
        self.execute(
            "git", "commit", "-m", "ve real E2E rebuild", cwd=self.worktree,
            stream=True,
        )
        rebuilt = read_marker(self.worktree)
        self.assertNotEqual(rebuilt.get("build_hash"), initial["build_hash"])
        self.assertEqual(rebuilt.get("venv_full_hash"), initial["venv_full_hash"])
        self.assertEqual(rebuilt.get("deepep_hash"), initial["deepep_hash"])
        self.assert_imports(self.worktree)

        # Checking out the original commit must restore the original cached
        # build and leave the real venv/DeepEP installation usable.
        self.execute(
            "git", "checkout", "--detach", initial_head, cwd=self.worktree,
            stream=True,
        )
        restored = read_marker(self.worktree)
        self.assertEqual(restored.get("build_hash"), initial["build_hash"])
        self.assertEqual(restored.get("venv_full_hash"), initial["venv_full_hash"])
        self.assertEqual(restored.get("deepep_hash"), initial["deepep_hash"])
        self.assert_imports(self.worktree)


if __name__ == "__main__":
    unittest.main()
