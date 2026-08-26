import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.util import _reflink_support, reflink_clone


class ReflinkCloneTest(unittest.TestCase):
    def setUp(self):
        _reflink_support.clear()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = self.root / "src"
        self.src.mkdir()

    def test_failed_probe_skips_recursive_reflink(self):
        dst = self.root / "dst"
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=1 if "--reflink=always" in command else 0)

        with patch("vllm_envs.util.run", side_effect=fake_run):
            reflink_clone(self.src, dst)

        self.assertEqual(len(calls), 2)
        self.assertNotIn("-a", calls[0])
        self.assertEqual(calls[1], ["cp", "-a", "--", str(self.src), str(dst)])

    def test_successful_probe_uses_recursive_reflink(self):
        dst = self.root / "dst"

        with patch(
            "vllm_envs.util.run", return_value=SimpleNamespace(returncode=0)
        ) as run:
            reflink_clone(self.src, dst)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "cp",
                "-a",
                "--reflink=always",
                "--",
                str(self.src),
                str(dst),
            ],
        )

    def test_probe_result_is_cached_by_device_pair(self):
        with patch(
            "vllm_envs.util.run", return_value=SimpleNamespace(returncode=1)
        ) as run:
            reflink_clone(self.src, self.root / "dst-1")
            reflink_clone(self.src, self.root / "dst-2")

        probe_calls = [
            call for call in run.call_args_list
            if "--reflink=always" in call.args[0]
        ]
        self.assertEqual(len(probe_calls), 1)

    def test_recursive_reflink_failure_cleans_up_and_falls_back(self):
        dst = self.root / "dst"
        calls = 0

        def fake_run(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(returncode=0)
            if calls == 2:
                dst.mkdir()
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(returncode=0)

        with patch("vllm_envs.util.run", side_effect=fake_run):
            reflink_clone(self.src, dst)

        self.assertEqual(calls, 3)
        self.assertFalse(dst.exists())


if __name__ == "__main__":
    unittest.main()
