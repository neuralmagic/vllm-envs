import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from vllm_envs.layers import _install_flashinfer_jit_cache, _install_req_groups


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
                "--index-url",
                "https://flashinfer.ai/whl/cu130",
            ],
        )


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


if __name__ == "__main__":
    unittest.main()
