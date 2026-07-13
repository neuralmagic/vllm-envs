import subprocess
import tempfile
import unittest
from pathlib import Path

from vllm_envs.precompiled import _candidate_commits


class PrecompiledCandidateIntegrationTest(unittest.TestCase):
    def git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def test_release_head_is_tried_before_main_ancestors(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.git(root, "init", "-b", "main")
        self.git(root, "config", "user.name", "Test User")
        self.git(root, "config", "user.email", "test@example.com")
        (root / "setup.py").write_text("# main build\n")
        self.git(root, "add", "setup.py")
        self.git(root, "commit", "-m", "main")
        self.git(root, "switch", "-c", "release")
        (root / "setup.py").write_text("# release-specific build\n")
        self.git(root, "commit", "-am", "release")
        release_head = self.git(root, "rev-parse", "HEAD")

        candidates = _candidate_commits(root)

        self.assertEqual(candidates, [release_head])


if __name__ == "__main__":
    unittest.main()
