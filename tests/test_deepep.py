import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.deepep import DeepEPResolution, _prepare_sources, resolve, sync_deepep
from vllm_envs.registry import read_marker, write_marker
from vllm_envs.store import read_meta


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


class DeepEPLayerIntegrationTest(unittest.TestCase):
    def test_historical_short_refs_are_prefetched_before_running_installer(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        installer = root / "install.sh"
        installer.write_text(
            'PPLX_COMMIT_HASH=${PPLX_COMMIT_HASH:-"12cecfd"}\n'
        )
        resolution = DeepEPResolution(
            "key", "73b6ea4", "3.3.24", "10.3a", installer
        )

        def fake_fetch(repo, ref, dest, label):
            dest.mkdir(parents=True)
            if label == "pplx-kernels":
                (dest / "setup.py").write_text(
                    'cmake_args = ["-WITH_TESTS=OFF"]\n'
                )

        with patch("vllm_envs.deepep.fetch_ref", side_effect=fake_fetch) as fetch:
            _prepare_sources(resolution, root / "workspace")

        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args_list[0].args[1], "12cecfd")
        self.assertEqual(fetch.call_args_list[1].args[1], "73b6ea4")
        setup = root / "workspace" / "pplx-kernels" / "setup.py"
        self.assertIn('"-DWITH_TESTS=OFF"', setup.read_text())

    def test_sync_reuses_complete_layer_and_rebuilds_when_installer_changes(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        checkout = root / "checkout"
        venv = checkout / ".venv"
        installer = checkout / "tools" / "ep_kernels" / "install_python_libraries.sh"
        installer.parent.mkdir(parents=True)
        installer.write_text(INSTALLER)
        (venv / "bin").mkdir(parents=True)
        write_marker(checkout, {"name": "test"})
        cfg = Config(cache_dir=root / "cache")

        state = {
            "installed": False,
            "builds": 0,
            "installs": 0,
            "install_command": [],
        }

        def fake_run(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd[0] == "bash":
                workspace = Path(cmd[cmd.index("--workspace") + 1])
                dist = workspace / "dist"
                dist.mkdir(parents=True)
                (dist / f"deep_ep-{state['builds']}.whl").touch()
                (dist / f"pplx_kernels-{state['builds']}.whl").touch()
                state["builds"] += 1
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd[:2] == [str(venv / "bin" / "python"), "-c"]:
                return subprocess.CompletedProcess(
                    cmd, 0 if state["installed"] else 1, "", ""
                )
            if cmd[:3] == ["uv", "pip", "install"]:
                state["installed"] = True
                state["installs"] += 1
                state["install_command"] = cmd
                return subprocess.CompletedProcess(cmd, 0, "", "")
            self.fail(f"unexpected command: {cmd}")

        with (
            patch("vllm_envs.deepep.run", side_effect=fake_run),
            patch("vllm_envs.deepep.cuda_version", return_value="12.9"),
            patch("vllm_envs.deepep._prepare_sources"),
        ):
            sync_deepep(cfg, checkout, venv, "venv-hash")
            first_hash = read_marker(checkout)["deepep_hash"]
            first_entry = cfg.store("ep-kernels") / first_hash

            self.assertTrue((first_entry / ".complete").exists())
            self.assertEqual(read_meta(first_entry)["deepep_ref"], "script-ref")
            self.assertEqual((state["builds"], state["installs"]), (1, 1))
            self.assertTrue(
                any("deep_ep-0.whl" in arg for arg in state["install_command"])
            )
            self.assertTrue(
                any("pplx_kernels-0.whl" in arg for arg in state["install_command"])
            )

            sync_deepep(cfg, checkout, venv, "venv-hash")
            self.assertEqual((state["builds"], state["installs"]), (1, 1))

            installer.write_text(INSTALLER + "# changed installer behavior\n")
            sync_deepep(cfg, checkout, venv, "venv-hash")

        second_hash = read_marker(checkout)["deepep_hash"]
        self.assertNotEqual(first_hash, second_hash)
        self.assertTrue((cfg.store("ep-kernels") / second_hash / ".complete").exists())
        self.assertEqual((state["builds"], state["installs"]), (2, 2))


if __name__ == "__main__":
    unittest.main()
