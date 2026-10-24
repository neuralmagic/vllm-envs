import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.layers import (
    _install_flashinfer_kernels,
    _install_req_groups,
    _uv_pip,
    ensure_full_template,
    sync,
)
from vllm_envs.store import write_meta


class FlashInferKernelsTest(unittest.TestCase):
    def test_required_flashinfer_downloads_kernel_wheels(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text("flashinfer-python==0.6.13\n")
        venv = root / ".venv"
        flashinfer = venv / "bin" / "flashinfer"
        flashinfer.parent.mkdir(parents=True)
        flashinfer.touch()

        with (
            patch("vllm_envs.layers.run") as run,
        ):
            _install_flashinfer_kernels(venv, "cuda", [requirements])

        run.assert_called_once_with(
            [str(flashinfer), "download-kernels"],
            stream_prefix="[ve]   [flashinfer] ",
        )

    def test_uv_install_retries_index_rate_limits(self):
        limited = subprocess.CalledProcessError(
            2, ["uv"], output="429 Too Many Requests"
        )
        with (
            patch("vllm_envs.layers.run", side_effect=[limited, None]) as run,
            patch("vllm_envs.layers.time.sleep") as sleep,
        ):
            _uv_pip(Path("/venv"), ["example"])

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(5)


class InstallReqGroupsTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.venv = root / ".venv"
        self.cap = root / "constraints.txt"
        self.test_files = [root / "test-cuda.txt"]
        self.runtime_files = [root / "cuda.txt"]
        self.groups = [self.test_files, self.runtime_files]

    def test_resolves_all_groups_in_one_pass(self):
        with patch("vllm_envs.layers._install_reqs") as install:
            _install_req_groups(
                "cfg", "keys", self.venv, self.groups, "cuda", self.cap
            )

        install.assert_called_once_with(
            "cfg", "keys", self.venv,
            [*self.test_files, *self.runtime_files], "cuda", self.cap,
        )

    def test_falls_back_to_one_pass_per_group(self):
        # A stale test lock (cuda-pathfinder==1.3.3) against a runtime pin that
        # needs a newer one: unsatisfiable together, fine in separate passes.
        passes = []

        def install(cfg, keys, venv, files, platform, cap_dest):
            passes.append(list(files))
            if len(passes) == 1:  # the combined resolve
                raise subprocess.CalledProcessError(1, ["uv"])

        with patch("vllm_envs.layers._install_reqs", side_effect=install):
            _install_req_groups(
                "cfg", "keys", self.venv, self.groups, "cuda", self.cap
            )

        self.assertEqual(
            passes,
            [
                [*self.test_files, *self.runtime_files],
                self.test_files,
                self.runtime_files,
            ],
        )

    def test_single_group_failure_propagates(self):
        with patch(
            "vllm_envs.layers._install_reqs",
            side_effect=subprocess.CalledProcessError(1, ["uv"]),
        ) as install:
            with self.assertRaises(subprocess.CalledProcessError):
                _install_req_groups(
                    "cfg", "keys", self.venv, [self.runtime_files, []],
                    "cuda", self.cap,
                )

        install.assert_called_once()

    def test_no_requirement_files_is_a_noop(self):
        with patch("vllm_envs.layers._install_reqs") as install:
            _install_req_groups("cfg", "keys", self.venv, [[], []], "cuda", self.cap)

        install.assert_not_called()


class FullTemplateRecoveryTest(unittest.TestCase):
    def test_resumes_matching_incomplete_template(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cfg = Config(cache_dir=root / "cache")
        keys = SimpleNamespace(
            base_hash="base", full_hash="full",
            layout=SimpleNamespace(test_files=[], runtime_files=[]),
        )
        base = cfg.store("venvs-base") / "base"
        (base / "bin").mkdir(parents=True)
        (base / ".complete").touch()
        entry = cfg.store("venvs") / "full"
        (entry / "bin").mkdir(parents=True)
        (entry / "bin" / "python").touch()
        write_meta(
            entry,
            {
                "kind": "venv-full", "hash": "full", "base": "base",
                "state": "incomplete", "stage": "installing dependencies",
            },
        )

        with (
            patch("vllm_envs.layers._install_req_groups") as install,
            patch("vllm_envs.layers._venv_freeze", return_value=[]),
            patch("vllm_envs.layers.reflink_clone") as clone,
        ):
            self.assertEqual(ensure_full_template(cfg, keys, "cuda"), entry)

        install.assert_called_once()
        clone.assert_not_called()
        self.assertTrue((entry / ".complete").exists())


class SyncFailureLogTest(unittest.TestCase):
    def test_retains_log_only_after_failure(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cfg = Config(cache_dir=root / "cache")
        with (
            patch("vllm_envs.layers.run", return_value=SimpleNamespace(stdout="abc\n")),
            patch("vllm_envs.layers.resolve_venv", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            sync(cfg, root)

        failure_log = root / ".ve" / "last-sync-failure.log"
        self.assertTrue(failure_log.exists())

        with (
            patch("vllm_envs.layers.run", return_value=SimpleNamespace(stdout="abc\n")),
            patch("vllm_envs.layers.resolve_venv", return_value=(root / ".venv", "keys")),
            patch("vllm_envs.layers.resolve_build", return_value="build"),
            patch("vllm_envs.layers.attach"),
            patch.object(cfg, "platform", "cpu"),
        ):
            sync(cfg, root)

        self.assertFalse(failure_log.exists())


if __name__ == "__main__":
    unittest.main()
