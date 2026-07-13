import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs.extprojects import fetch_ref


class ExactGitRefIntegrationTest(unittest.TestCase):
    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    def test_fetches_tag_not_reachable_from_default_branch(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        origin = root / "origin"
        child = root / "child"
        checkout = root / "checkout"
        child.mkdir()
        self.git(child, "init", "-b", "main")
        self.git(child, "config", "user.name", "Test User")
        self.git(child, "config", "user.email", "test@example.com")
        (child / "header.h").write_text("submodule content\n")
        self.git(child, "add", "header.h")
        self.git(child, "commit", "-m", "child")
        origin.mkdir()
        self.git(origin, "init", "-b", "main")
        self.git(origin, "config", "user.name", "Test User")
        self.git(origin, "config", "user.email", "test@example.com")
        (origin / "data").write_text("main\n")
        self.git(origin, "add", "data")
        self.git(origin, "commit", "-m", "main")
        main = self.git(origin, "rev-parse", "HEAD")
        (origin / "data").write_text("archived\n")
        self.git(
            origin,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(child),
            "deps/child",
        )
        self.git(origin, "commit", "-am", "archived with submodule")
        self.git(origin, "tag", "archived")
        self.git(origin, "reset", "--hard", main)

        with patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            fetch_ref(str(origin), "archived", checkout, "test")

        self.assertEqual((checkout / "data").read_text(), "archived\n")
        self.assertEqual(
            (checkout / "deps" / "child" / "header.h").read_text(),
            "submodule content\n",
        )
        self.assertEqual(
            self.git(checkout, "rev-parse", "HEAD"),
            self.git(origin, "rev-parse", "archived"),
        )


if __name__ == "__main__":
    unittest.main()
