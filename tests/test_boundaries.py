"""Facts and rules on both sides of the upstream commits that introduced them.

Each case `git archive`s the relevant paths at an introducing commit and its
parent from a local vLLM clone (VE_VLLM_REPO, default ~/local/vllm) and
asserts the detected value on each side. Skipped without the clone/commit.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from vllm_envs import rules
from vllm_envs.treefacts import EP_INSTALLER, TreeFacts

VLLM_REPO = Path(os.environ.get("VE_VLLM_REPO", Path.home() / "local" / "vllm"))
TREE_PATHS = (
    "setup.py", EP_INSTALLER, "docker/versions.json", "docker/Dockerfile",
    ".buildkite/scripts/install-kv-connectors.sh", "rust-toolchain.toml",
    "rust/Cargo.toml", "tools/build_rust.py",
)


def files(name):
    def probe(facts: TreeFacts):
        return [p.relative_to(facts.root).as_posix() for p in getattr(facts, name)]
    return probe


def fact(name):
    def probe(facts: TreeFacts):
        value = getattr(facts, name)
        return value.relative_to(facts.root).as_posix() if isinstance(value, Path) else value
    return probe


def fires(rule_id, files="none"):
    """Whether a rule fires, with Req predicates evaluated against the full
    venv layer's requirement files when files="full"."""
    def probe(facts: TreeFacts):
        layout = facts.layout(with_test=True)
        chosen = [*layout.runtime_files, *layout.test_files] if files == "full" else []
        return rule_id in {r.id for r in rules.fired(facts.context(chosen))}
    return probe


# (introducing commit, what, probe, value at parent, value at commit)
BOUNDARIES = (
    ("206e2577fa9", "requirements/ dir: build layout", files("build_requirements"),
     ["requirements-build.txt"], ["requirements/build.txt"]),
    ("206e2577fa9", "requirements/test.txt CUDA lock era", files("test_requirements"),
     ["requirements-test.txt"], ["requirements/test.txt"]),
    ("fc645f1acc0", "per-platform build requirements", files("build_requirements"),
     ["requirements/build.txt"], ["requirements/build/cuda.txt"]),
    ("fc645f1acc0", "per-platform test locks", files("test_requirements"),
     ["requirements/test.txt"], ["requirements/test/cuda.txt"]),
    ("d45fe333fb8", "EP installer", fact("ep_installer"), None, EP_INSTALLER),
    ("d45fe333fb8", "installer alone is not enough", fact("ep_supported"),
     False, False),
    ("c4c0354eec8", "installer DeepEP ref default", fact("deepep_ref"),
     None, "73b6ea4"),
    ("c4c0354eec8", "installer pplx ref default", fact("pplx_ref"),
     None, "12cecfd"),
    ("c4c0354eec8", "DeepEP layer supported", fact("ep_supported"), False, True),
    ("c4c0354eec8", "pplx CMake flag patch", fires("pplx-with-tests-flag"),
     False, True),
    ("9e078d05825", "docker/versions.json DeepEP pin", fact("deepep_ref_source"),
     EP_INSTALLER, "docker/versions.json"),
    ("eb19955c370", "pplx removed", fires("pplx-with-tests-flag"), True, False),
    ("5d8e90a9661", "NCCL pin", fact("nccl_version"), None, "2.30.7"),
    ("5d8e90a9661", "NCCL override", fires("deepep-nccl-override"), False, True),
    ("f2ace1d57d2", "Rust frontend", fact("rust_frontend"), False, True),
    ("816cc73a9be", "KV installer", fact("kv_supported"), False, True),
    ("816cc73a9be", "KV --system strip", fires("kv-installer-venv"), False, True),
    ("c9d33c60dcd", "FlashInfer default dependency",
     fires("flashinfer-jit-cache", files="full"), False, True),
    ("29e4d2d42e1", "jit-cache still derived after download-kernels",
     fires("flashinfer-jit-cache", files="full"), True, True),
    ("ba47bb5be1b", "flashinfer-cubin derived once off PyPI",
     fires("flashinfer-cubin", files="full"), False, True),
    ("29e4d2d42e1", "flashinfer-cubin derived after leaving cuda.txt",
     fires("flashinfer-cubin", files="full"), True, True),
    ("90ba2f34b5a", "setup.py wheel path constants", fact("wheel_paths_declared"),
     False, True),
)


def _git(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(VLLM_REPO), *args],
                          capture_output=True, check=False, **kwargs)


def _extract(rev: str, dest: Path) -> None:
    top = _git("ls-tree", "--name-only", rev, text=True).stdout.split()
    wanted = [n for n in top if n.startswith("requirements")] + list(TREE_PATHS)
    files = _git("ls-tree", "-r", "--name-only", rev, "--", *wanted,
                 text=True).stdout.split()
    archive = _git("archive", rev, "--", *files)
    archive.check_returncode()
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


@unittest.skipUnless((VLLM_REPO / ".git").exists(), f"no vLLM clone at {VLLM_REPO}")
class BoundaryCommitTest(unittest.TestCase):
    trees: dict[str, TreeFacts] = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.trees = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def facts_at(self, rev: str) -> TreeFacts:
        if rev not in self.trees:
            dest = Path(self.tmp.name) / rev.replace("^", "_parent")
            dest.mkdir()
            _extract(rev, dest)
            self.trees[rev] = TreeFacts(dest, "cuda", "x86_64")
        return self.trees[rev]

    def test_boundaries(self):
        for commit, what, probe, before, after in BOUNDARIES:
            with self.subTest(commit=commit, what=what):
                if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
                    self.skipTest(f"{commit} not in {VLLM_REPO}")
                self.assertEqual(probe(self.facts_at(f"{commit}^")), before)
                self.assertEqual(probe(self.facts_at(commit)), after)

    def test_extras_and_deepep_agree_across_ep_eras(self):
        from vllm_envs.extras import resolve as resolve_extras
        from vllm_envs.deepep import resolve as resolve_deepep
        from unittest.mock import patch

        for commit in ("d45fe333fb8", "c4c0354eec8", "3c98c2d21b3"):
            for rev in (f"{commit}^", commit):
                with self.subTest(rev=rev), \
                        patch("vllm_envs.deepep.run") as run, \
                        patch("vllm_envs.deepep.cuda_version", return_value="13.0"):
                    run.return_value.stdout = ""
                    facts = self.facts_at(rev)
                    bundle = resolve_extras(facts.root, "base", "venv")
                    self.assertEqual(bundle.deepep, facts.ep_supported)
                    if facts.ep_supported:
                        resolve_deepep(facts.root, "venv")
                    else:
                        with self.assertRaises(RuntimeError):
                            resolve_deepep(facts.root, "venv")


if __name__ == "__main__":
    unittest.main()
