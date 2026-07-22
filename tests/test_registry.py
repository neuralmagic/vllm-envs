import multiprocessing
import tempfile
import unittest
from pathlib import Path

from vllm_envs.config import Config
from vllm_envs.registry import load_registry, register_env


def _register_env(cache_dir: str, index: int, start) -> None:
    start.wait()
    cfg = Config(cache_dir=Path(cache_dir))
    register_env(
        cfg,
        f"env-{index}",
        Path(cache_dir) / f"env-{index}",
        Path(cache_dir) / "repo",
    )


class RegistryConcurrencyTest(unittest.TestCase):
    def test_parallel_registrations_are_not_lost(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = Config(cache_dir=Path(temp_dir))
            context = multiprocessing.get_context("fork")
            start = context.Event()
            count = 32
            processes = [
                context.Process(target=_register_env, args=(temp_dir, i, start))
                for i in range(count)
            ]

            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(
                set(load_registry(cfg)),
                {f"env-{i}" for i in range(count)},
            )


if __name__ == "__main__":
    unittest.main()
