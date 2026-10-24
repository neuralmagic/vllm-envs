"""Regression tests for bugs found in the October 2026 codebase review."""

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.extras import ExtrasResolution, _sync_kv_connectors
from vllm_envs.hashing import requirements_layout
from vllm_envs.registry import MarkerError, read_marker, write_marker
from vllm_envs.worktrees import collect


class MarkerTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_strings_with_quotes_and_backslashes_round_trip(self):
        data = {"name": 'we"ird\\name', "repo": "/a b/c", "n": 3, "ok": True}
        write_marker(self.root, data)
        self.assertEqual(read_marker(self.root), data)

    def test_corrupt_marker_raises_instead_of_reading_as_unmanaged(self):
        (self.root / ".vllm-env.toml").write_text('[ve]\nname = "trunc')
        with self.assertRaises(MarkerError):
            read_marker(self.root)

    def test_missing_marker_is_empty(self):
        self.assertEqual(read_marker(self.root), {})


class LockedWorktreeTest(unittest.TestCase):
    def test_locked_worktree_is_never_safe_to_reap(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        repo = tmp / "repo"
        repo.mkdir()
        git = lambda *a, cwd=repo: subprocess.run(
            ["git", *a], cwd=cwd, check=True, capture_output=True)
        git("init", "-q", "-b", "main")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
            "--allow-empty", "-m", "init")
        git("worktree", "add", "-q", "--detach", str(tmp / "wt"))
        git("worktree", "lock", str(tmp / "wt"))

        (wt,) = [w for w in collect(repo) if w.path.name == "wt"]

        self.assertTrue(wt.locked)
        self.assertEqual(wt.safety()[0], False)


class KvConnectorKeyTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.installer = self.root / "install-kv-connectors.sh"
        self.installer.write_text("uv pip install -r x\n")
        self.reqs = self.root / "kv_connectors.txt"
        self.reqs.write_text("lmcache==0.5\n")
        self.resolution = ExtrasResolution("key", self.installer, self.reqs)
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(
            patch("vllm_envs.extras.dists_installed", return_value=True))

    def sync(self) -> int:
        with patch("vllm_envs.util.run") as run:
            _sync_kv_connectors(self.root, self.root / ".venv", self.resolution, "v1")
        return run.call_count

    def test_requirement_bump_reinstalls_even_when_nixl_imports(self):
        self.assertEqual(self.sync(), 1)
        self.assertEqual(self.sync(), 0)
        self.reqs.write_text("lmcache==0.6\n")
        self.assertEqual(self.sync(), 1)


class TestLockSelectionTest(unittest.TestCase):
    def layout(self, files: list[str], machine: str = "x86_64"):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for f in files:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text("x==1\n")
        with patch("vllm_envs.treefacts.host_platform.machine", return_value=machine):
            lay = requirements_layout(root, "cuda", with_test=True)
        return [str(p.relative_to(root)) for p in lay.test_files]

    mid_era = ["requirements/build.txt", "requirements/common.txt",
               "requirements/cuda.txt", "requirements/test.in",
               "requirements/test.txt"]

    def test_mid_era_cuda_lock_is_used(self):
        self.assertEqual(self.layout(self.mid_era), ["requirements/test.txt"])

    def test_non_x86_hosts_use_the_unpinned_input(self):
        self.assertEqual(self.layout(self.mid_era, "aarch64"),
                         ["requirements/test.in"])


if __name__ == "__main__":
    unittest.main()


class PrecompiledVariantAndVerifyTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_variant_follows_venv_torch_not_driver(self):
        from vllm_envs import precompiled

        sp = self.root / ".venv/lib/python3.12/site-packages"
        (sp / "torch-2.9.0+cu129.dist-info").mkdir(parents=True)
        with patch("vllm_envs.precompiled.cuda_version", return_value="13.0"):
            self.assertEqual(precompiled._variant("cuda", self.root / ".venv"), "cu129")
            self.assertEqual(precompiled._variant("cuda", self.root / "none"), "cu130")

    def test_corrupt_download_is_rejected(self):
        import zipfile

        from vllm_envs.precompiled import _verify_wheel

        good = self.root / "good.whl"
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("vllm-1.dist-info/METADATA", "Name: vllm\n")
        _verify_wheel(good)
        truncated = self.root / "bad.whl"
        truncated.write_bytes(good.read_bytes()[:-30])
        with self.assertRaises((zipfile.BadZipFile, ValueError)):
            _verify_wheel(truncated)


class BuildKeyEnvTest(unittest.TestCase):
    def test_runtime_knobs_and_gpu_reservations_do_not_split_the_build_key(self):
        from vllm_envs.hashing import _build_environment_identity

        base = _build_environment_identity()
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3",
                                       "VLLM_LOGGING_LEVEL": "DEBUG",
                                       "TORCH_LOGS": "+dynamo"}):
            self.assertEqual(_build_environment_identity(), base)
        with patch.dict("os.environ", {"CMAKE_ARGS": "-DFOO=1"}):
            self.assertNotEqual(_build_environment_identity(), base)


class CmakePinParsingTest(unittest.TestCase):
    def pins(self, cmake: str, platform: str):
        from vllm_envs.extprojects import parse_pins

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "CMakeLists.txt").write_text(cmake)
        return parse_pins(root, platform)

    BRANCHED = """
if(FOO_SRC_DIR)
  FetchContent_Declare(foo SOURCE_DIR ${FOO_SRC_DIR})
else()
  if (VLLM_TARGET_DEVICE STREQUAL "rocm")
    set(FOO_TAG "rocm-sha")
  else()
    set(FOO_TAG "v1.0")
  endif()
  FetchContent_Declare(foo
    GIT_REPOSITORY https://x/foo.git
    GIT_TAG ${FOO_TAG})
endif()
"""

    def test_per_platform_branch_pins_the_taken_branch(self):
        self.assertEqual(self.pins(self.BRANCHED, "cuda")["foo"].tag, "v1.0")
        self.assertEqual(self.pins(self.BRANCHED, "rocm")["foo"].tag, "rocm-sha")

    def test_undefined_or_ambiguous_variables_are_not_pinned(self):
        undefined = "FetchContent_Declare(foo\n GIT_REPOSITORY https://x\n GIT_TAG v${VER})"
        ambiguous = """
if(SOME_OPTION)
  set(T "a")
else()
  set(T "b")
endif()
FetchContent_Declare(foo
  GIT_REPOSITORY https://x
  GIT_TAG ${T})
"""
        self.assertNotIn("foo", self.pins(undefined, "cuda"))
        self.assertNotIn("foo", self.pins(ambiguous, "cuda"))

    def test_cache_sets_are_understood(self):
        cmake = ('set(T "v2" CACHE STRING "tag")\n'
                 "FetchContent_Declare(foo\n GIT_REPOSITORY https://x\n GIT_TAG ${T})")
        self.assertEqual(self.pins(cmake, "cuda")["foo"].tag, "v2")
