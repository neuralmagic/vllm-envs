import io
import json
import platform
import subprocess
import tempfile
import unittest
import zipfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

from vllm_envs import precompiled
from vllm_envs.config import Config
from vllm_envs.precompiled import (
    MAX_PROBES,
    MISSING,
    compiled_paths,
    find_matches,
    lookup,
    object_ids,
    try_fetch_precompiled,
)

ARCH = platform.machine()


class FakeWheels:
    """wheels.vllm.ai stand-in: metadata for `published` commits, 404 else."""

    def __init__(self, published=()):
        self.published = set(published)
        self.urls: list[str] = []

    def __call__(self, url: str) -> list:
        self.urls.append(url)
        commit = url.removeprefix(precompiled.WHEELS_BASE).split("/", 1)[0]
        if commit not in self.published:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return [{
            "package_name": "vllm",
            "platform_tag": f"manylinux_2_28_{ARCH}",
            "path": f"../../../{commit}/vllm-1.0-cp38-abi3-manylinux_2_28_{ARCH}.whl",
        }]

    def probed(self) -> list[str]:
        return [u.removeprefix(precompiled.WHEELS_BASE).split("/", 1)[0]
                for u in self.urls]


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, msg: str, **files: str | None) -> str:
        """Write (or delete, with None) files keyed by path with '__' for '/'."""
        for key, content in files.items():
            path = self.root / key.replace("__", "/").replace("_DOT_", ".")
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.git("add", "-A")
        self.git("commit", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(cache_dir=self.cache, platform="cuda")
        self.enterContext(patch.object(precompiled, "cuda_version",
                                       return_value="13.0"))
        self.enterContext(patch.object(precompiled, "user_overrides",
                                       return_value={}))
        self.repo = r = Repo(self.root)
        # main: c0 csrc=A | c1 csrc=B | c2 py | c3 py | c4 py | c5 csrc=C
        # branch forks at c2 with a Python-only commit.
        self.c0 = r.commit("c0", csrc__k_DOT_cu="A", vllm___custom_ops_DOT_py="ops",
                           pyproject_DOT_toml="[p]", setup_DOT_py="# s\n",
                           vllm__x_DOT_py="0")
        self.c1 = r.commit("c1", csrc__k_DOT_cu="B")
        self.c2 = r.commit("c2", vllm__x_DOT_py="2")
        r.git("branch", "feature")
        self.c3 = r.commit("c3", vllm__x_DOT_py="3")
        self.c4 = r.commit("c4", vllm__x_DOT_py="4")
        self.c5 = r.commit("c5", csrc__k_DOT_cu="C")
        r.git("switch", "feature")
        self.head = r.commit("b1", vllm__y_DOT_py="feature")

    def run_lookup(self, wheels: FakeWheels, **kw):
        with patch.object(precompiled, "_fetch_metadata", wheels):
            return lookup(self.cfg, self.root, **kw)


class PrecompiledLookupTest(_Fixture):
    def test_matching_run_ordered_nearest_to_merge_base(self):
        look = find_matches(self.root)
        self.assertEqual(look.merge_base, self.c2)
        self.assertEqual(
            [(m.commit, m.distance) for m in look.matches],
            [(self.head, None), (self.c2, 0), (self.c1, -1), (self.c3, 1),
             (self.c4, 2)],
        )

    def test_forward_match_after_merge_base(self):
        look = self.run_lookup(FakeWheels({self.c3, self.c4}))
        self.assertEqual((look.chosen.commit, look.chosen.distance), (self.c3, 1))
        self.assertEqual(
            look.url,
            f"{precompiled.WHEELS_BASE}{self.c3}/vllm-1.0-cp38-abi3-"
            f"manylinux_2_28_{ARCH}.whl",
        )

    def test_backward_match_and_nearest_wins(self):
        look = self.run_lookup(FakeWheels({self.c1, self.c4}))
        self.assertEqual((look.chosen.commit, look.chosen.distance), (self.c1, -1))

    def test_exact_head_wheel_preferred(self):
        look = self.run_lookup(FakeWheels({self.head, self.c2}))
        self.assertEqual(look.chosen.commit, self.head)

    def test_unpublished_skipped_and_root_variant_never_used(self):
        wheels = FakeWheels({self.c4})
        look = self.run_lookup(wheels)
        self.assertEqual(look.chosen.commit, self.c4)
        self.assertEqual(wheels.probed(),
                         [self.head, self.c2, self.c1, self.c3, self.c4])
        self.assertTrue(all("/cu130/vllm/metadata.json" in u for u in wheels.urls))

    def test_branch_csrc_change_has_no_match(self):
        self.repo.commit("b2", csrc__k_DOT_cu="PR kernel")
        wheels = FakeWheels({self.c1, self.c2, self.c3, self.c4, self.c5})
        look = self.run_lookup(wheels)
        self.assertIsNone(look.url)
        self.assertEqual(wheels.probed(), [look.head])

    def test_branch_custom_ops_change_has_no_match(self):
        self.repo.commit("b2", vllm___custom_ops_DOT_py="new op")
        look = self.run_lookup(FakeWheels({self.c2, self.c3}))
        self.assertIsNone(look.url)

    def test_setup_py_difference_matches_with_warning(self):
        self.repo.commit("b2", setup_DOT_py="# branch tweak\n")
        look = self.run_lookup(FakeWheels({self.c2}))
        self.assertEqual(look.chosen.commit, self.c2)
        self.assertEqual(look.chosen.stale, ["setup.py"])

    def test_path_set_read_from_setup_py(self):
        self.assertEqual(compiled_paths(self.root)[0],
                         precompiled.DEFAULT_BLOCKING_PATHS)
        self.repo.commit("b2", setup_DOT_py=(
            "class precompiled_wheel_utils:\n"
            "    WHEEL_BLOCKING_PATHS = (\n        'csrc',\n    )\n"
            "    WHEEL_WARNING_PATHS = ('rust',)\n"
        ), pyproject_DOT_toml="[changed]")
        self.assertEqual(compiled_paths(self.root), (("csrc",), ("rust",)))
        # pyproject.toml is no longer blocking for this tree
        look = self.run_lookup(FakeWheels({self.c2}))
        self.assertEqual(look.chosen.commit, self.c2)

    def test_missing_path_is_part_of_identity(self):
        oids = object_ids(self.root, [self.c0], ("csrc", "nope.txt"))[self.c0]
        self.assertNotEqual(oids[0], MISSING)
        self.assertEqual(oids[1], MISSING)
        self.repo.commit("b2", vllm___custom_ops_DOT_py=None)
        look = self.run_lookup(FakeWheels({self.c2, self.c3}))
        self.assertIsNone(look.url)

    def test_probe_cap(self):
        r = self.repo
        r.git("switch", "main")
        for i in range(MAX_PROBES + 4):
            r.commit(f"py{i}", vllm__x_DOT_py=f"m{i}")
        wheels = FakeWheels()
        with patch.object(precompiled, "_fetch_metadata", wheels):
            look = lookup(self.cfg, self.root)
        self.assertIsNone(look.url)
        self.assertEqual(len(wheels.urls), MAX_PROBES)

    def test_index_short_circuits_second_lookup(self):
        first = self.run_lookup(FakeWheels({self.c3}))
        self.assertEqual(first.source, "probe")
        wheels = FakeWheels()
        second = self.run_lookup(wheels)
        self.assertEqual(wheels.urls, [])
        self.assertEqual((second.source, second.url), ("index", first.url))
        self.assertEqual(second.chosen.distance, 1)
        index = json.loads((self.cache / precompiled.INDEX_NAME).read_text())
        self.assertEqual(len(index["hits"]), 1)

    def test_negative_memo_expires(self):
        now = 1_000_000.0
        with patch.object(precompiled.time, "time", return_value=now):
            self.assertIsNone(self.run_lookup(FakeWheels()).url)
            wheels = FakeWheels({self.c2})
            look = self.run_lookup(wheels)
        self.assertEqual(wheels.urls, [])  # all memoized as unpublished
        self.assertIsNone(look.url)
        later = now + precompiled.NEGATIVE_TTL_S + 1
        with patch.object(precompiled.time, "time", return_value=later):
            look = self.run_lookup(wheels)
        self.assertEqual(look.chosen.commit, self.c2)

    def test_readonly_lookup_writes_nothing(self):
        self.run_lookup(FakeWheels({self.c2}), use_index=False, record=False)
        self.assertFalse((self.cache / precompiled.INDEX_NAME).exists())


def _wheel_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("vllm-0.1.dist-info/METADATA", "Name: vllm\n")
    return buf.getvalue()


WHEEL_BYTES = _wheel_bytes()


class TryFetchPrecompiledTest(_Fixture):
    def fetch(self, wheels: FakeWheels, payload: bytes = WHEEL_BYTES):
        body = io.BytesIO(payload)
        with patch.object(precompiled, "_fetch_metadata", wheels), \
                patch.object(precompiled.urllib.request, "urlopen",
                             return_value=body):
            return try_fetch_precompiled(self.cfg, self.root, "bh")

    def test_downloads_and_records_origin(self):
        wheel = self.fetch(FakeWheels({self.c3}))
        self.assertEqual(wheel.read_bytes(), WHEEL_BYTES)
        meta = json.loads((wheel.parent / "meta.json").read_text())
        self.assertEqual(meta["origin"], f"precompiled:{self.c3}")
        self.assertEqual(meta["distance"], 1)
        self.assertEqual(meta["merge_base"], self.c2)

    def test_corrupt_download_is_not_published(self):
        self.assertIsNone(self.fetch(FakeWheels({self.c3}), b"<html>503</html>"))
        self.assertEqual(list((self.cache / "builds" / "bh").glob("*.whl*")), [])

    def test_dirty_blocking_paths_disable_precompiled(self):
        for rel in ("pyproject.toml", "vllm/_custom_ops.py"):
            path = self.root / rel
            original = path.read_text()
            path.write_text(original + "# dirty\n")
            wheels = FakeWheels({self.c2})
            self.assertIsNone(self.fetch(wheels), rel)
            self.assertEqual(wheels.urls, [], rel)
            path.write_text(original)

    def test_release_head_off_main_tried_first(self):
        r = self.repo
        r.git("switch", "-c", "release", self.c4)
        head = r.commit("release", setup_DOT_py="# release build\n")
        wheels = FakeWheels({head, self.c4})
        wheel = self.fetch(wheels)
        self.assertEqual(wheels.probed(), [head])
        meta = json.loads((wheel.parent / "meta.json").read_text())
        self.assertEqual(meta["origin"], f"precompiled:{head}")


if __name__ == "__main__":
    unittest.main()
