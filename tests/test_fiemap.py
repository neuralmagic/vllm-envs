import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.fiemap import reflink_usage


class ReflinkScopeAccountingTest(unittest.TestCase):
    def test_store_shared_blocks_and_incremental_venv_blocks_are_additive(self):
        roots = {
            "/store": (1, [(100, 10), (200, 10)]),
            "/venv-a": (2, [(100, 10), (300, 10), (400, 10)]),
            "/venv-b": (3, [(100, 10), (400, 10), (500, 10)]),
        }
        fd_extents = {}
        next_fd = iter(range(10, 20))

        def walk(root):
            yield str(root), [], ["data"]

        def lstat(path):
            inode, _ = roots[str(Path(path).parent)]
            return SimpleNamespace(st_dev=1, st_ino=inode, st_mode=stat.S_IFREG)

        def open_file(path, flags):
            fd = next(next_fd)
            fd_extents[fd] = roots[str(Path(path).parent)][1]
            return fd

        with (
            patch("vllm_envs.fiemap.os.walk", side_effect=walk),
            patch("vllm_envs.fiemap.os.lstat", side_effect=lstat),
            patch("vllm_envs.fiemap.os.open", side_effect=open_file),
            patch("vllm_envs.fiemap.os.close"),
            patch(
                "vllm_envs.fiemap._extents", side_effect=lambda fd: fd_extents[fd]
            ),
        ):
            usage = reflink_usage(
                {
                    "store": [Path("/store")],
                    "venv-a": [Path("/venv-a")],
                    "venv-b": [Path("/venv-b")],
                },
                {"store": "stores", "venv-a": "venvs", "venv-b": "venvs"},
            )

        self.assertEqual(usage.unique_total, 50)
        self.assertEqual(usage.unique_by_scope, {"stores": 20, "venvs": 40})
        self.assertEqual(usage.exclusive_by_scope, {"stores": 10, "venvs": 30})
        self.assertEqual(
            usage.unique_by_scope["stores"] + usage.exclusive_by_scope["venvs"],
            usage.unique_total,
        )


if __name__ == "__main__":
    unittest.main()
