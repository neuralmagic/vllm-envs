import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.deepep import resolve, sync_deepep


INSTALLER = """\
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"script-ref"}
NVSHMEM_VER=${NVSHMEM_VER:-"3.3.24"}
"""


class DeepEPResolutionTest(unittest.TestCase):
    def setUp(self):
        self.run_mock = self.enterContext(
            patch(
                "vllm_envs.deepep.run",
                return_value=SimpleNamespace(stdout=""),
            )
        )

    def make_root(self) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        installer = root / "tools" / "ep_kernels" / "install_python_libraries.sh"
        installer.parent.mkdir(parents=True)
        installer.write_text(INSTALLER)
        return root

    def test_uses_centralized_docker_pin_and_installer_nvshmem_default(self):
        root = self.make_root()
        versions = root / "docker" / "versions.json"
        versions.parent.mkdir()
        versions.write_text(
            json.dumps({"variable": {"DEEPEP_COMMIT_HASH": {"default": "docker-ref"}}})
        )

        resolution = resolve(root, "venv-hash")

        self.assertEqual(resolution.ref, "docker-ref")
        self.assertEqual(resolution.nvshmem_version, "3.3.24")

    def test_uses_docker_deepep_architectures(self):
        root = self.make_root()
        dockerfile = root / "docker" / "Dockerfile"
        dockerfile.parent.mkdir()
        dockerfile.write_text("export TORCH_CUDA_ARCH_LIST='9.0a 10.0a'\n")

        resolution = resolve(root, "venv-hash")

        self.assertEqual(resolution.cuda_arch_list, "9.0a 10.0a")

    def test_explicit_architecture_override_wins(self):
        root = self.make_root()

        with patch.dict("os.environ", {"TORCH_CUDA_ARCH_LIST": "10.0"}):
            resolution = resolve(root, "venv-hash")

        self.assertEqual(resolution.cuda_arch_list, "10.0")

    def test_adds_local_cuda_architecture_to_docker_targets(self):
        root = self.make_root()
        dockerfile = root / "docker" / "Dockerfile"
        dockerfile.parent.mkdir()
        dockerfile.write_text("export TORCH_CUDA_ARCH_LIST='9.0a 10.0a'\n")
        self.run_mock.return_value = SimpleNamespace(stdout="10.3\n10.3\n")

        resolution = resolve(root, "venv-hash")

        self.assertEqual(resolution.cuda_arch_list, "9.0a 10.0a 10.3a")

    def test_falls_back_to_installer_pin(self):
        root = self.make_root()

        resolution = resolve(root, "venv-hash")

        self.assertEqual(resolution.ref, "script-ref")

    def test_venv_hash_changes_layer_key(self):
        root = self.make_root()

        first = resolve(root, "first")
        second = resolve(root, "second")

        self.assertNotEqual(first.key, second.key)

    def test_commit_before_ep_installer_is_skipped(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))

        with patch("vllm_envs.deepep.resolve") as resolve_mock:
            sync_deepep(Config(), root, root / ".venv", "venv-hash")

        resolve_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
