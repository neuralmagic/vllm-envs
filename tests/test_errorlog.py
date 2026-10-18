import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs import cli, errorlog
from vllm_envs.commands import envs
from vllm_envs.log import error, warn


class ErrorLogTest(unittest.TestCase):
    def setUp(self):
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"VE_CACHE_DIR": str(self.cache)}))
        os.environ.pop("VE_NO_ERROR_LOG", None)
        self.addCleanup(setattr, errorlog, "_log_path", None)
        errorlog.configure(self.cache, ["ve", "sync"])
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def records(self) -> list[dict]:
        return errorlog.read(self.cache)

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(list(argv))
        return rc, out.getvalue()

    def test_unconfigured_logging_is_a_noop(self):
        errorlog._log_path = None
        warn("not recorded")
        self.assertEqual(self.records(), [])

    def test_warning_inside_except_keeps_subprocess_output(self):
        try:
            raise subprocess.CalledProcessError(
                2, ["cmake", "--build"], output="ninja: error: elfutils missing\n"
            )
        except subprocess.CalledProcessError:
            warn("local build failed; falling back to precompiled")

        (rec,) = self.records()
        self.assertEqual(rec["level"], "warning")
        self.assertEqual(rec["argv"], ["ve", "sync"])
        self.assertEqual(rec["exc_type"], "CalledProcessError")
        self.assertEqual(rec["subprocess"]["cmd"], ["cmake", "--build"])
        self.assertEqual(rec["subprocess"]["returncode"], 2)
        self.assertIn("elfutils missing", rec["subprocess"]["output_tail"])

    def test_streamed_subprocess_failure_keeps_output_tail(self):
        from vllm_envs.util import run

        try:
            run(["sh", "-c", "echo building; echo 'fatal: no elfutils' >&2; exit 3"],
                stream_prefix="")
        except subprocess.CalledProcessError:
            warn("local build failed")

        (rec,) = self.records()
        self.assertEqual(rec["subprocess"]["returncode"], 3)
        self.assertIn("fatal: no elfutils", rec["subprocess"]["output_tail"])

    def test_buried_ninja_failure_is_excerpted_and_full_log_saved(self):
        from vllm_envs.util import run

        warning_spam = (
            "platform.h:627:33: warning: 'double4' is deprecated "
            "[-Wdeprecated-declarations]\n" * 3000
        )
        script = self.cache / "build.sh"
        script.write_text(
            "cat <<'EOF'\n"
            "[536/545] Building CUDA object a.cu.o\n"
            "FAILED: CMakeFiles/_C.dir/csrc/hisparse/residency.cu.o\n"
            "/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DFOO\n"
            f"{warning_spam}"
            "csrc/hisparse/residency.cu(88): error: identifier \"kSlots\" "
            "is undefined\n"
            "1 error detected in the compilation of \"residency.cu\".\n"
            "[537/545] Building CUDA object b.cu.o\n"
            f"{warning_spam}"
            "ninja: build stopped: subcommand failed.\n"
            "subprocess.CalledProcessError: Command '['cmake']' returned "
            "non-zero exit status 1.\n"
            "EOF\nexit 1\n"
        )
        try:
            run(["sh", str(script)], stream_prefix="")
        except subprocess.CalledProcessError:
            with contextlib.redirect_stderr(io.StringIO()):
                warn("local build failed")

        (rec,) = self.records()
        sub = rec["subprocess"]
        self.assertNotIn("kSlots", sub["output_tail"])
        self.assertIn("FAILED: CMakeFiles/_C.dir/csrc/hisparse/residency.cu.o",
                      sub["error_excerpt"])
        self.assertIn('error: identifier "kSlots" is undefined', sub["error_excerpt"])
        self.assertNotIn("CalledProcessError", sub["error_excerpt"])
        self.assertLess(len(sub["error_excerpt"].splitlines()), 30)

        rc, out = self.run_cli("errors", "show", rec["id"])
        self.assertEqual(rc, 0)
        self.assertLess(out.index("kSlots"), out.index("output tail:"))
        self.assertIn(sub["full_log"], out)
        rc, out = self.run_cli("errors", "show", rec["id"], "--full")
        self.assertEqual(out.count("warning: 'double4' is deprecated"), 6000)

    def test_streamed_success_leaves_no_spool(self):
        from vllm_envs.util import run

        with patch("tempfile.tempdir", str(self.cache)):
            run(["sh", "-c", "echo ok"], stream_prefix="")
        self.assertEqual(list(self.cache.glob("ve-output-*")), [])

    def test_log_rotates_past_size_limit(self):
        with patch.object(errorlog, "MAX_BYTES", 1):
            error("first")
            error("second")
            error("third")

        self.assertTrue((self.cache / "errors.jsonl.1").exists())
        self.assertEqual(
            [r["message"] for r in self.records()], ["second", "third"]
        )

    def test_advisory_warnings_can_skip_the_log(self):
        warn("informational", log=False)
        self.assertEqual(self.records(), [])

    def test_fingerprint_ignores_paths_hashes_and_numbers(self):
        a = errorlog.fingerprint("error", "sync failed at /a/b/wt1 (abc1234def) exit 1")
        b = errorlog.fingerprint("error", "sync failed at /x/wt-2 (0123456789ab) exit 7")
        c = errorlog.fingerprint("error", "auto-init failed at /a/b/wt1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_unhandled_exception_is_recorded_and_reraised(self):
        with (
            patch.object(envs, "cmd_list", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            self.run_cli("list")

        (rec,) = self.records()
        self.assertEqual(rec["level"], "error")
        self.assertEqual(rec["message"], "unhandled RuntimeError: boom")
        self.assertEqual(rec["argv"], ["ve", "list"])
        self.assertIn("RuntimeError: boom", rec["traceback"])

    def test_opt_out_disables_recording(self):
        with patch.dict(os.environ, {"VE_NO_ERROR_LOG": "1"}):
            errorlog.configure(self.cache, ["ve", "sync"])
        error("ignored")
        self.assertEqual(self.records(), [])

    def test_summary_groups_by_fingerprint_and_counts_notes(self):
        error("sync failed at /wt/a")
        error("sync failed at /wt/b")
        warn("local build failed")
        target = self.records()[0]

        rc, _ = self.run_cli("errors", "note", "--ref", target["id"],
                             "re-ran ve sync --fresh-venv")
        self.assertEqual(rc, 0)
        rc, out = self.run_cli("errors", "--json")
        self.assertEqual(rc, 0)

        summary = json.loads(out)
        self.assertEqual([g["count"] for g in summary], [2, 1])
        self.assertEqual(summary[0]["fingerprint"], target["fingerprint"])
        self.assertEqual(summary[0]["notes"], ["re-ran ve sync --fresh-venv"])

    def test_note_defaults_to_latest_error_in_this_repo(self):
        error("first failure")
        warn("later fallback")
        latest_error = self.records()[0]

        with patch.object(errorlog, "_repo_context", return_value={}):
            self.run_cli("errors", "note", "worked around it")

        note = self.records()[-1]
        self.assertEqual(note["level"], "note")
        self.assertEqual(note["ref"], latest_error["id"])

    def test_show_includes_attached_notes(self):
        error("sync failed: uv resolution")
        rec = self.records()[0]
        self.run_cli("errors", "note", "--ref", rec["fingerprint"], "pinned numpy")

        rc, out = self.run_cli("errors", "show", rec["fingerprint"])
        self.assertEqual(rc, 0)
        self.assertIn("sync failed: uv resolution", out)
        self.assertIn("pinned numpy", out)

    def test_errors_command_itself_is_not_logged(self):
        with self.assertRaises(SystemExit):
            self.run_cli("errors", "show", "does-not-exist")
        self.assertEqual(self.records(), [])


if __name__ == "__main__":
    unittest.main()
