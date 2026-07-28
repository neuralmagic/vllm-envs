import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.hashing import build_key, build_paths_dirty
from vllm_envs.layers import _build_wheel, local_build_env, resolve_build


class LocalBuildCacheIntegrationTest(unittest.TestCase):
    def git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def make_venv(self, root: Path, torch_record: str = "torch files v1") -> Path:
        venv = root / ".venv"
        dist = venv / "lib/python3.12/site-packages/torch-2.9.1.dist-info"
        dist.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            "implementation = CPython\nversion = 3.12.13\n"
        )
        (dist / "METADATA").write_text("Name: torch\nVersion: 2.9.1\n")
        (dist / "WHEEL").write_text("Tag: cp312-cp312-linux_x86_64\n")
        (dist / "RECORD").write_text(torch_record)
        return venv

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.git(self.repo, "init", "-b", "main")
        self.git(self.repo, "config", "user.name", "Test User")
        self.git(self.repo, "config", "user.email", "test@example.com")
        (self.repo / "setup.py").write_text("# build input\n")
        self.git(self.repo, "add", "setup.py")
        self.git(self.repo, "commit", "-m", "initial")
        self.other = self.base / "other"
        self.git(self.repo, "worktree", "add", "--detach", str(self.other), "HEAD")
        self.cache = self.base / "cache"
        self.cfg = Config(cache_dir=self.cache, platform="cpu", python="3.12")

    def test_clean_worktrees_reuse_one_successful_local_wheel(self):
        first_venv = self.make_venv(self.repo)
        second_venv = self.make_venv(self.other)
        builds = 0

        def fake_build(cfg, env_root, venv, build_temp, dist_dir, use_pinned_ext):
            nonlocal builds
            builds += 1
            dist_dir.mkdir(parents=True, exist_ok=True)
            wheel = dist_dir / "vllm-1.0-cp312-cp312-linux_x86_64.whl"
            wheel.write_bytes(b"compiled wheel")
            return wheel

        with (
            patch("vllm_envs.layers.try_fetch_precompiled", return_value=None),
            patch("vllm_envs.layers._build_wheel", side_effect=fake_build),
        ):
            first = resolve_build(self.cfg, self.repo, first_venv)
            second = resolve_build(self.cfg, self.other, second_venv)

        self.assertEqual(builds, 1)
        self.assertTrue(first.shared)
        self.assertEqual(first.build_hash, second.build_hash)
        self.assertEqual(second.mode, "store-wheel")
        self.assertEqual(second.wheel, first.wheel)

    def test_torch_and_build_flags_separate_local_binary_keys(self):
        venv = self.make_venv(self.repo)
        original = build_key(self.repo, "cpu", "3.12", venv)

        record = next(venv.glob("lib*/python*/site-packages/torch-*.dist-info/RECORD"))
        record.write_text("torch files v2")
        changed_torch = build_key(self.repo, "cpu", "3.12", venv)

        with patch.dict("os.environ", {"CXXFLAGS": "-march=native"}):
            changed_flags = build_key(self.repo, "cpu", "3.12", venv)

        self.assertNotEqual(original, changed_torch)
        self.assertNotEqual(changed_torch, changed_flags)

    def test_worktree_venv_paths_do_not_separate_build_keys(self):
        first_venv = self.make_venv(self.repo)
        second_venv = self.make_venv(self.other)
        for venv in (first_venv, second_venv):
            bindir = venv / "bin"
            bindir.mkdir()
            for name in ("cmake", "ninja"):
                tool = bindir / name
                tool.write_text("#!/bin/sh\necho fake-tool 1.0\n")
                tool.chmod(0o755)

        self.assertEqual(
            build_key(self.repo, "cpu", "3.12", first_venv),
            build_key(self.other, "cpu", "3.12", second_venv),
        )

    @unittest.skipUnless(shutil.which("ccache") and shutil.which("c++"),
                         "ccache and c++ are required")
    def test_ccache_hits_across_content_addressed_build_directories(self):
        source = self.repo / "unchanged.cc"
        source.write_text("int unchanged() { return 42; }\n")
        build_a = self.base / "cmake-build" / "hash-a"
        build_b = self.base / "cmake-build" / "hash-b"
        build_a.mkdir(parents=True)
        build_b.mkdir(parents=True)
        cache = self.base / "ccache"
        env = {
            **os.environ,
            **local_build_env(self.repo),
            "CCACHE_DIR": str(cache),
        }

        def compile_in(build: Path) -> Path:
            output = build / "unchanged.o"
            subprocess.run(
                ["ccache", "c++", "-g", "-c", str(source), "-o", str(output)],
                cwd=build, env=env, check=True, capture_output=True, text=True,
            )
            return output

        first = compile_in(build_a)
        second = compile_in(build_b)
        stats = json.loads(subprocess.run(
            ["ccache", "--print-stats", "--format=json"],
            env=env, check=True, capture_output=True, text=True,
        ).stdout)

        self.assertEqual(stats["cache_miss"], 1)
        self.assertEqual(
            stats["direct_cache_hit"] + stats["preprocessed_cache_hit"], 1
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_local_wheel_is_built_once_then_packaged_without_rebuild(self):
        venv = self.make_venv(self.repo)
        build_temp = self.base / "build-temp"
        dist = self.base / "dist"
        dist.mkdir()
        (dist / "vllm-stale.whl").write_bytes(b"wrong branch")
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "vllm-current.whl").write_bytes(b"current branch")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("vllm_envs.layers.run", side_effect=fake_run):
            wheel = _build_wheel(
                self.cfg, self.repo, venv, build_temp, dist,
                use_pinned_ext=False,
            )

        self.assertEqual(len(commands), 1)
        self.assertIn("bdist_wheel", commands[0])
        self.assertIn("--skip-build", commands[0])
        self.assertEqual(wheel.name, "vllm-current.whl")
        self.assertFalse((dist / "vllm-stale.whl").exists())

    def test_local_build_tools_come_from_target_venv(self):
        venv = self.make_venv(self.repo)
        bindir = venv / "bin"
        bindir.mkdir(parents=True)
        ninja = bindir / "ninja"
        ninja.write_text("#!/bin/sh\necho target-venv-ninja\n")
        ninja.chmod(0o755)

        with patch.dict("os.environ", {"PATH": "/usr/bin"}):
            env = {**os.environ, **local_build_env(self.repo, venv)}
            found = subprocess.run(
                ["ninja"], env=env, check=True, capture_output=True, text=True,
            )

        self.assertEqual(env["VIRTUAL_ENV"], str(venv))
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(bindir))
        self.assertEqual(found.stdout.strip(), "target-venv-ninja")

    def test_checkout_leftover_generated_marlin_sources_are_not_user_edits(self):
        venv = self.make_venv(self.repo)
        before = build_key(self.repo, "cpu", "3.12", venv)
        generated = (
            self.repo / "csrc/quantization/marlin/"
            "sm80_kernel_float16_u4_float16.cu"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("// generated by the release CMake configure\n")

        self.assertFalse(build_paths_dirty(self.repo))
        self.assertEqual(before, build_key(self.repo, "cpu", "3.12", venv))

        user_source = self.repo / "csrc/user_kernel.cu"
        user_source.write_text("// intentional source edit\n")
        self.assertTrue(build_paths_dirty(self.repo))
        self.assertNotEqual(before, build_key(self.repo, "cpu", "3.12", venv))


if __name__ == "__main__":
    unittest.main()
