import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.layers import BuildResolution, attach, resolve_build
from vllm_envs.registry import read_marker, write_marker


class PrecompiledFallbackIntegrationTest(unittest.TestCase):
    def test_successful_env_local_fallback_is_reused(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        venv = root / ".venv"
        (venv / "lib").mkdir(parents=True)
        (root / "vllm").mkdir()
        (root / "vllm" / "_C.abi3.so").touch()
        write_marker(
            root,
            {"build_hash": "build-key", "attach_mode": "precompiled-fetch"},
        )

        with (
            patch("vllm_envs.layers.build_key", return_value="build-key"),
            patch("vllm_envs.layers.build_paths_dirty", return_value=False),
            patch("vllm_envs.layers.user_overrides", return_value={}),
            patch("vllm_envs.layers.try_fetch_precompiled", return_value=None),
            patch("vllm_envs.layers.editable.editable_present", return_value=True),
        ):
            resolution = resolve_build(Config(cache_dir=root / "cache"), root, venv)

        self.assertEqual(
            resolution,
            BuildResolution("precompiled-fetch", None, "build-key", shared=False),
        )

    def test_fallback_records_the_content_hash_after_attach(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        venv = root / ".venv"
        venv.mkdir()
        write_marker(root, {"name": "test"})
        resolution = BuildResolution(
            "precompiled-fetch", None, "build-key", shared=False
        )

        with patch("vllm_envs.layers._uv_pip"):
            attach(Config(), root, venv, resolution)

        marker = read_marker(root)
        self.assertEqual(marker["build_hash"], "build-key")
        self.assertEqual(marker["attach_mode"], "precompiled-fetch")


if __name__ == "__main__":
    unittest.main()
