import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.extprojects import _github_full_ref, fetch_ref


class GitHubRefResolutionTest(unittest.TestCase):
    def test_uses_authenticated_cli_when_anonymous_api_fails(self):
        sha = "73b6ea4a439ba03a695563f9fd242c8e4b02b37c"
        with (
            patch(
                "vllm_envs.extprojects.urllib.request.urlopen",
                side_effect=OSError,
            ),
            patch("vllm_envs.extprojects.shutil.which", return_value="/usr/bin/gh"),
            patch(
                "vllm_envs.extprojects.run",
                return_value=SimpleNamespace(returncode=0, stdout=f"{sha}\n"),
            ) as run,
        ):
            resolved = _github_full_ref(
                "https://github.com/deepseek-ai/DeepEP", "73b6ea4"
            )

        self.assertEqual(resolved, sha)
        self.assertEqual(run.call_args.args[0][:2], ["gh", "api"])


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

        with patch.dict(
            os.environ,
            {
                "GIT_ALLOW_PROTOCOL": "file",
                "GIT_DIR": str(origin / ".git"),
                "GIT_WORK_TREE": str(origin),
                "GIT_PREFIX": "leaked/from/hook/",
            },
        ):
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

    def test_fetches_short_commit_from_default_branch_without_api(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        origin = root / "origin"
        checkout = root / "checkout"
        origin.mkdir()
        self.git(origin, "init", "-b", "main")
        self.git(origin, "config", "user.name", "Test User")
        self.git(origin, "config", "user.email", "test@example.com")
        (origin / "data").write_text("pinned\n")
        self.git(origin, "add", "data")
        self.git(origin, "commit", "-m", "pinned")
        pinned = self.git(origin, "rev-parse", "HEAD")
        for index in range(3):
            (origin / "data").write_text(f"head-{index}\n")
            self.git(origin, "commit", "-am", f"head {index}")

        with patch(
            "vllm_envs.extprojects._github_full_ref",
            side_effect=AssertionError("hosting API must not be used"),
        ):
            fetch_ref(str(origin), pinned[:10], checkout, "test")

        self.assertEqual((checkout / "data").read_text(), "pinned\n")
        self.assertEqual(self.git(checkout, "rev-parse", "HEAD"), pinned)


if __name__ == "__main__":
    unittest.main()
