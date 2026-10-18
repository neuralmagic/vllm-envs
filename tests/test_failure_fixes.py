import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs import attach
from vllm_envs.buildlayer import BuildResolution, local_build_env
from vllm_envs.config import Config
from vllm_envs.util import run_retrying_transient


class BuildEnvTest(unittest.TestCase):
    def test_builds_share_ccache_across_worktrees(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / "wt").mkdir()
        cfg = Config(cache_dir=tmp / "cache", ccache_max_size_gb=50)
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("CCACHE_DIR", None)
            built = local_build_env(tmp / "wt", tmp / "wt" / ".venv", cfg)
        self.assertEqual(built["CCACHE_BASEDIR"], str(tmp / "wt"))
        self.assertEqual(built["CCACHE_DIR"], str(tmp / "cache" / "ccache"))
        self.assertEqual(built["CCACHE_MAXSIZE"], "50G")


class RefusedFallbackTest(unittest.TestCase):
    def test_error_leads_with_the_build_failure(self):
        build_error = subprocess.CalledProcessError(
            1, ["cmake"], output="fatal error: elfutils/libdwfl.h: No such file"
        )
        refused = subprocess.CalledProcessError(
            1, ["uv"], output=(
                "        ValueError: No precompiled wheel is published yet for\n"
                "        0c25849, and the nearest one was built from different\n"
                "        compiled sources: csrc/foo.cu\n"
                "        Wait for wheel publication, rebase onto ...\n"
            ),
        )
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        res = BuildResolution("precompiled-fetch", None, "abc", shared=False,
                              build_error=build_error)
        with (
            patch.object(attach, "uv_pip_install", side_effect=refused),
            patch.object(attach, "read_marker", return_value={}),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(RuntimeError) as ctx,
        ):
            attach.attach(SimpleConfig(root), root, root / ".venv", res)

        msg = str(ctx.exception)
        self.assertIn("local build failed", msg)
        self.assertIn("No precompiled wheel is published yet for 0c25849", msg)
        self.assertNotIn("Wait for", msg)
        self.assertIs(ctx.exception.__cause__, build_error)


class SimpleConfig:
    def __init__(self, root: Path):
        self.root = root

    def store(self, name: str) -> Path:
        return self.root / name


class TransientRetryTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(patch("vllm_envs.util.time.sleep"))

    def test_network_flake_is_retried(self):
        flake = subprocess.CalledProcessError(
            2, ["bash"], output="error: Request failed after 3 retries in 7.2s\n"
        )
        ok = subprocess.CompletedProcess(["bash"], 0, "", "")
        with patch("vllm_envs.util.run", side_effect=[flake, ok]) as run:
            self.assertIs(run_retrying_transient(["bash"], "install"), ok)
        self.assertEqual(run.call_count, 2)

    def test_real_failure_is_not_retried(self):
        fail = subprocess.CalledProcessError(1, ["uv"], output="No solution found\n")
        with patch("vllm_envs.util.run", side_effect=[fail]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_retrying_transient(["uv"], "install")
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
