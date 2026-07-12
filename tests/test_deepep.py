import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.deepep import resolve, sync_deepep


INSTALLER = """\
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"script-ref"}
NVSHMEM_VER=${NVSHMEM_VER:-"3.3.24"}
"""


class DeepEPResolutionTest(unittest.TestCase):
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
