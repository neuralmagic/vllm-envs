import subprocess
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from vllm_envs.layers import _install_flashinfer_jit_cache, _install_req_groups
from vllm_envs.layers import fixup_fa_cute_imports


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


class FixupFaCuteImportsTests(unittest.TestCase):
    """A precompiled + editable attach copies vllm/vllm_flash_attn/cute/ without
    running cmake's rewrite, and the runtime shim only covers the symlink case,
    so ve has to apply the rewrite itself."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env_root = Path(self.tmp.name)
        self.cute = self.env_root / "vllm" / "vllm_flash_attn" / "cute"
        self.cute.mkdir(parents=True)

    def test_rewrites_bare_imports_in_a_copied_tree(self):
        src = self.cute / "flash_fwd.py"
        src.write_text("from flash_attn.cute import utils\nimport flash_attn.cute.pack\n")

        fixup_fa_cute_imports(self.env_root)

        self.assertEqual(
            src.read_text(),
            "from vllm.vllm_flash_attn.cute import utils\n"
            "import vllm.vllm_flash_attn.cute.pack\n",
        )

    def test_is_idempotent(self):
        src = self.cute / "flash_fwd.py"
        src.write_text("from flash_attn.cute import utils\n")

        fixup_fa_cute_imports(self.env_root)
        once = src.read_text()
        fixup_fa_cute_imports(self.env_root)

        self.assertEqual(src.read_text(), once)

    def test_leaves_a_symlinked_tree_alone(self):
        # VLLM_FLASH_ATTN_SRC_DIR builds symlink cute/ and rely on the runtime
        # shim to register a virtual flash_attn package; rewriting the source
        # checkout in place would be wrong.
        real = self.env_root / "fa-src" / "cute"
        real.mkdir(parents=True)
        (real / "flash_fwd.py").write_text("from flash_attn.cute import utils\n")
        shutil.rmtree(self.cute)
        self.cute.symlink_to(real, target_is_directory=True)

        fixup_fa_cute_imports(self.env_root)

        self.assertEqual(
            (real / "flash_fwd.py").read_text(), "from flash_attn.cute import utils\n"
        )

    def test_missing_tree_is_a_noop(self):
        shutil.rmtree(self.cute)
        fixup_fa_cute_imports(self.env_root)  # must not raise


if __name__ == "__main__":
    unittest.main()
