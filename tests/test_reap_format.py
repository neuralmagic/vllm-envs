import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.commands.reap import _print_worktrees, _worktree_labels


def _wt(path: str, branch: str = "main", dirty: int = 0, state: str = "worktree"):
    return SimpleNamespace(
        path=Path(path), branch=branch, state=state, activity=None,
        dirty=dirty, sync="pushed", is_candidate=False,
    )


class WorktreeLabelsTest(unittest.TestCase):
    def test_unique_names_stay_short(self):
        wts = [_wt("/a/one"), _wt("/b/two")]
        self.assertEqual(
            _worktree_labels(wts), {Path("/a/one"): "one", Path("/b/two"): "two"}
        )

    def test_duplicate_names_widen_to_parent_then_full_path(self):
        wts = [_wt("/x/a/vllm"), _wt("/x/b/vllm"), _wt("/y/b/vllm")]
        self.assertEqual(
            _worktree_labels(wts),
            {
                Path("/x/a/vllm"): "a/vllm",
                Path("/x/b/vllm"): "/x/b/vllm",
                Path("/y/b/vllm"): "/y/b/vllm",
            },
        )


class PrintWorktreesTest(unittest.TestCase):
    def render(self, wts, tty: bool, columns: int = 120) -> list[str]:
        out = io.StringIO()
        out.isatty = lambda: tty
        with (
            redirect_stdout(out),
            patch("shutil.get_terminal_size",
                  return_value=SimpleNamespace(columns=columns, lines=40)),
        ):
            _print_worktrees(wts, 30)
        return out.getvalue().splitlines()

    def test_columns_align_regardless_of_name_and_dirty_width(self):
        wts = [
            _wt("/r/short", dirty=7157),
            _wt("/r/" + "long-name-" * 3, branch="feature/x"),
        ]
        lines = self.render(wts, tty=False)
        state_cols = {line.index("worktree", 10) for line in lines[1:3]}
        self.assertEqual(len(state_cols), 1)
        self.assertIn("7157 dirty", lines[1] + lines[2])

    def test_tty_truncates_long_names_and_branches(self):
        wts = [_wt("/r/" + "n" * 60, branch="b" * 200)]
        row = self.render(wts, tty=True, columns=120)[1]
        self.assertIn("n" * 39 + "…", row)
        self.assertLessEqual(len(row), 120)
        self.assertTrue(row.endswith("…"))


if __name__ == "__main__":
    unittest.main()
