import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from vllm_envs.config import Config
from vllm_envs.layers import (
    _filtered_req_files,
    _install_flashinfer_jit_cache,
    _install_narrow_index_packages,
    _install_req_groups,
    _uv_pip,
    ensure_full_template,
    sync,
)
from vllm_envs.store import write_meta


class NarrowIndexTest(unittest.TestCase):
    def test_filtering_drops_index_directives_and_inlines_nested_files(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "common.txt").write_text("numpy==2.0.0\n")
        (root / "cuda.txt").write_text(
            "-r common.txt\n"
            "--extra-index-url https://flashinfer.ai/whl/\n"
            "flashinfer-python==0.6.16.post3\n"
            "flashinfer-cubin==0.6.16.post3\n"
        )
        dest = root / "out"
        dest.mkdir()

        filtered = _filtered_req_files([root / "cuda.txt"], dest)

        self.assertEqual(
            filtered[0].read_text().split(),
            ["numpy==2.0.0", "flashinfer-python==0.6.16.post3"],
        )

    def test_narrow_index_package_installed_scoped_to_its_index(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text("flashinfer-cubin==0.6.16.post3\n")
        venv = root / ".venv"

        with patch("vllm_envs.layers._uv_pip") as uv_pip:
            _install_narrow_index_packages(venv, [requirements])

        uv_pip.assert_called_once_with(
            venv,
            [
                "flashinfer-cubin==0.6.16.post3",
                "--no-deps",
                "--index-url",
                "https://flashinfer.ai/whl/",
            ],
        )


class FlashInferJitCacheTest(unittest.TestCase):
    def test_required_flashinfer_installs_matching_cuda_jit_cache(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text("flashinfer-python==0.6.13\n")
        venv = root / ".venv"

        with (
            patch(
                "vllm_envs.layers.run",
                side_effect=[
                    SimpleNamespace(stdout="Version: 0.6.13\n"),
                    SimpleNamespace(stdout="13.0\n"),
                ],
            ) as run,
            patch("vllm_envs.layers._uv_pip") as uv_pip,
        ):
            _install_flashinfer_jit_cache(venv, "cuda", [requirements])

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        "uv",
                        "pip",
                        "show",
                        "--python",
                        str(venv / "bin" / "python"),
                        "flashinfer-python",
                    ]
                ),
                call(
                    [
                        str(venv / "bin" / "python"),
                        "-c",
                        "import torch; print(torch.version.cuda or '')",
                    ]
                ),
            ],
        )
        uv_pip.assert_called_once_with(
            venv,
            [
                "flashinfer-jit-cache==0.6.13",
                "--no-deps",
                "--index-url",
                "https://flashinfer.ai/whl/cu130",
            ],
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
