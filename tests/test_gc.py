import contextlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.config import Config
from vllm_envs.gc import TOMBSTONE, collect_candidates, run_gc
from vllm_envs.locks import entry_lock


class GcTest(unittest.TestCase):
    def setUp(self):
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(cache_dir=self.cache, min_age_hours=0)
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(patch("vllm_envs.gc._prune_uv_cache"))
        self.enterContext(patch("vllm_envs.gc.referenced_build_hashes", return_value=set()))
        self.enterContext(patch("vllm_envs.gc.referenced_deepep_hashes", return_value=set()))

    def entry(self, store: str, name: str, age_s: float = 3600) -> Path:
        e = self.cfg.store(store) / name
        e.mkdir(parents=True)
        (e / "payload").write_bytes(b"x" * 4096)
        (e / ".complete").touch()
        old = time.time() - age_s
        os.utime(e, (old, old))
        return e

    def test_eviction_keeps_the_lock_file_and_removes_the_entry(self):
        e = self.entry("venvs", "abc")
        with entry_lock(e):
            pass
        run_gc(self.cfg, free_gb=1)
        self.assertFalse(e.exists())
        self.assertTrue((e.parent / "abc.lock").exists())

    def test_interrupted_delete_never_leaves_entry_under_its_name(self):
        e = self.entry("venvs", "abc")
        with patch("vllm_envs.gc.shutil.rmtree"):  # killed mid-delete
            run_gc(self.cfg, free_gb=1)
        self.assertFalse(e.exists())
        (tomb,) = e.parent.glob(f".abc{TOMBSTONE}*")
        self.assertTrue((tomb / ".complete").exists())
        self.assertEqual(collect_candidates(self.cfg), [])

        run_gc(self.cfg, free_gb=1)
        self.assertFalse(tomb.exists())

    def test_locked_entry_is_skipped(self):
        e = self.entry("venvs", "abc")
        with entry_lock(e):
            run_gc(self.cfg, free_gb=1)
        self.assertTrue(e.exists())


if __name__ == "__main__":
    unittest.main()
