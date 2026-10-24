import tempfile
import unittest
from pathlib import Path

from vllm_envs.reqs import (
    canonical_name,
    declared_floors,
    freeze_map,
    freeze_names,
    requirement_names,
)
from vllm_envs.sitepkgs import dist_info_map, dists_installed, site_packages


class RequirementParsingTest(unittest.TestCase):
    def test_names_are_pep503_normalized(self):
        self.assertEqual(canonical_name("Zope.Interface__x"), "zope-interface-x")

    def test_files_skip_comments_options_and_urls(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        req = root / "r.txt"
        req.write_text(
            "-r common.txt\n# comment\nTorch_Audio==2.9  # pinned\n"
            "pkg @ https://example.com/pkg.whl\nnumpy>=1.26; python_version>'3'\n"
        )
        self.assertEqual(requirement_names([req]), {"torch-audio", "numpy"})
        self.assertEqual(declared_floors([req]),
                         {"torch-audio": (2, 9), "numpy": (1, 26)})

    def test_freeze_parsing_shares_one_name_form(self):
        lines = ["-e file:///src/vllm", "Foo_Bar==1.0", "baz @ file:///w/baz.whl"]
        self.assertEqual(set(freeze_map(lines)), {"vllm", "foo-bar", "baz"})
        self.assertEqual(freeze_names(lines), {"foo-bar"})


class SitePackagesTest(unittest.TestCase):
    def test_dist_info_lookup(self):
        venv = Path(self.enterContext(tempfile.TemporaryDirectory()))
        sp = venv / "lib" / "python3.12" / "site-packages"
        (sp / "deep_ep-1.2.dist-info").mkdir(parents=True)
        self.assertEqual(site_packages(venv), sp)
        self.assertTrue(dists_installed(venv, ["deep_ep"]))
        self.assertFalse(dists_installed(venv, ["deep_ep", "nixl"]))
        self.assertEqual(dist_info_map(sp), {"deep-ep": sp / "deep_ep-1.2.dist-info"})


if __name__ == "__main__":
    unittest.main()
