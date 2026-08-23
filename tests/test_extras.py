import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.extras import ExtrasResolution, _sync_kv_connectors, resolve


class ExtrasResolutionTest(unittest.TestCase):
    def test_bundle_key_tracks_vllm_installers_and_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            installer = root / ".buildkite/scripts/install-kv-connectors.sh"
            requirements = root / "requirements/kv_connectors.txt"
            installer.parent.mkdir(parents=True)
            requirements.parent.mkdir()
            installer.write_text("uv pip install --system -r requirements.txt\n")
            requirements.write_text("nixl==1\n")

            with patch("vllm_envs.extras.resolve_deepep", side_effect=RuntimeError):
                first = resolve(root, "venv-key")
                requirements.write_text("nixl==2\n")
                second = resolve(root, "venv-key")

            self.assertNotEqual(first.key, second.key)
            self.assertEqual(second.kv_installer, installer)
            self.assertEqual(second.kv_requirements, requirements)

    def test_kv_helper_targets_managed_virtual_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / ".venv"
            installer = root / "install-kv-connectors.sh"
            requirements = root / "kv_connectors.txt"
            installer.write_text(
                'uv pip install --system -r "$KV_CONNECTORS_REQUIREMENTS"\n'
            )
            requirements.write_text("nixl==1\n")
            resolution = ExtrasResolution("key", installer, requirements)
            observed = {}

            def fake_run(command, **kwargs):
                observed["script"] = Path(command[1]).read_text()
                observed["env"] = kwargs["env"]

            with (
                patch("vllm_envs.extras._has_import", return_value=False),
                patch("vllm_envs.extras.run", side_effect=fake_run),
            ):
                _sync_kv_connectors(venv, resolution)

            self.assertNotIn("--system", observed["script"])
            self.assertEqual(observed["env"]["VIRTUAL_ENV"], str(venv))
            self.assertEqual(
                observed["env"]["KV_CONNECTORS_REQUIREMENTS"], str(requirements)
            )


if __name__ == "__main__":
    unittest.main()
