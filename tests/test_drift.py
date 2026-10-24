import contextlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_envs import drift, errorlog
from vllm_envs.config import MARKER_NAME
from vllm_envs.envsync import _tracking_drift


class DriftTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / MARKER_NAME).write_text('name = "env"\n')
        self.sp = self.root / ".venv/lib/python3.12/site-packages"
        self.sp.mkdir(parents=True)
        self.freeze = ["numpy==2.1.0", "torch==2.9.0"]
        self.enterContext(
            patch.object(drift, "venv_freeze", side_effect=lambda _: list(self.freeze))
        )
        self.cache = self.root / "cache"
        self.addCleanup(errorlog.configure, None, [])
        errorlog.configure(self.cache, ["ve", "sync"])

    def set_baseline(self, lines: list[str], age_s: float = 100) -> None:
        path = drift.baseline_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        old = time.time() - age_s
        os.utime(path, (old, old))

    def install(self, name: str, version: str, files: tuple[str, ...] = ()) -> None:
        dist = self.sp / f"{name}-{version}.dist-info"
        dist.mkdir()
        (dist / "RECORD").write_text(
            "".join(f"{f},,\n" for f in files) + f"{dist.name}/RECORD,,\n"
        )
        for f in files:
            (self.sp / f).parent.mkdir(parents=True, exist_ok=True)
            (self.sp / f).write_text("x")

    def test_tracked_diff_reports_added_removed_changed(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0", "einops==0.8.0"])
        self.freeze = ["numpy==2.2.0", "torch==2.9.0", "ruff==0.14.0"]

        d = drift.compute(self.root, scan_files=False)

        self.assertEqual(d.mode, "tracked")
        self.assertEqual(d.added, {"ruff": "ruff==0.14.0"})
        self.assertEqual(d.removed, {"einops": "einops==0.8.0"})
        self.assertEqual(d.changed, {"numpy": ("numpy==2.1.0", "numpy==2.2.0")})

    def test_patched_files_found_and_new_package_files_attributed(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0"])
        self.install("ruff", "0.14.0", ("ruff/__init__.py",))
        self.freeze.append("ruff==0.14.0")
        patched = self.sp / "torch/nn/functional.py"
        patched.parent.mkdir(parents=True)
        patched.write_text("hack")
        (self.sp / "torch/__pycache__").mkdir()
        (self.sp / "torch/__pycache__/x.cpython-312.pyc").write_text("")

        d = drift.compute(self.root)

        self.assertEqual(d.files, ["torch/nn/functional.py"])
        self.assertIn("1 patched file(s) in torch", d.summary())

    def test_ve_changes_fold_into_baseline_but_agent_changes_stay(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0"])
        pre = ["numpy==2.1.0", "ruff==0.14.0", "torch==2.9.0"]  # agent added ruff
        post = ["numpy==2.2.0", "ruff==0.14.0", "torch==2.9.0"]  # ve bumped numpy

        drift.fold_into_baseline(self.root, pre, post, recloned=False)
        self.freeze = post
        d = drift.compute(self.root, scan_files=False)

        self.assertEqual(d.added, {"ruff": "ruff==0.14.0"})
        self.assertEqual(d.changed, {})

    def test_ve_reverting_an_agent_change_clears_it(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0"])
        pre = ["numpy==2.3.0", "torch==2.9.0"]
        post = ["numpy==2.1.0", "torch==2.9.0"]

        drift.fold_into_baseline(self.root, pre, post, recloned=False)
        self.freeze = post

        self.assertTrue(drift.compute(self.root, scan_files=False).empty)

    def test_reclone_resets_baseline(self):
        self.set_baseline(["numpy==2.1.0"])
        drift.fold_into_baseline(
            self.root, ["numpy==2.1.0", "ruff==1"], ["torch==2.9.0"], recloned=True
        )
        self.freeze = ["torch==2.9.0"]
        self.assertTrue(drift.compute(self.root, scan_files=False).empty)

    def test_noop_sync_keeps_baseline_time_so_patches_stay_visible(self):
        self.set_baseline(self.freeze)
        before = drift.baseline_path(self.root).stat().st_mtime
        with contextlib.redirect_stderr(io.StringIO()), _tracking_drift(self.root):
            pass
        self.assertEqual(drift.baseline_path(self.root).stat().st_mtime, before)

    def test_ve_same_version_reinstall_is_not_blamed_on_agent(self):
        self.set_baseline(self.freeze)
        with contextlib.redirect_stderr(io.StringIO()), _tracking_drift(self.root):
            self.install("torch", "2.9.0", ("torch/__init__.py",))
        self.assertTrue(drift.compute(self.root).empty)
        self.assertEqual(errorlog.read(self.cache), [])

    def test_agent_reinstall_of_editable_is_named(self):
        self.set_baseline(["-e file:///src/vllm", "torch==2.9.0"])
        self.freeze = ["-e file:///src/vllm", "torch==2.9.0"]
        self.install("vllm", "0.30.0", ("__editable__.vllm-0.30.0.pth",))
        (self.sp / "vllm-0.30.0.dist-info/direct_url.json").write_text(
            '{"url": "file:///src/vllm", "dir_info": {"editable": true}}'
        )

        d = drift.compute(self.root)

        self.assertEqual(list(d.changed), ["vllm"])
        self.assertEqual(d.n_files, 0)
        self.assertIn("(reinstalled)", d.summary())

    def test_legacy_env_flags_packages_installed_after_last_sync(self):
        old = time.time() - 1000
        os.utime(self.root / MARKER_NAME, (old, old))
        tmpl = self.root / ".ve/template-freeze.txt"
        tmpl.parent.mkdir(parents=True, exist_ok=True)
        tmpl.write_text("numpy==2.1.0\ntorch==2.9.0\neinops==0.8.0\n")
        self.install("ruff", "0.14.0")
        self.install("numpy", "2.1.0")
        self.freeze = ["numpy==2.1.0", "ruff==0.14.0", "torch==2.9.0"]

        d = drift.compute(self.root, scan_files=False)

        self.assertEqual(d.mode, "legacy")
        self.assertEqual(d.added, {"ruff": "ruff==0.14.0"})
        self.assertEqual(d.changed, {"numpy": ("numpy==2.1.0", "numpy==2.1.0")})
        self.assertEqual(d.removed, {"einops": "einops==0.8.0"})
        self.assertIn("numpy 2.1.0 (reinstalled)", d.summary())

    def test_drift_recorded_once_per_state_and_grouped_by_package(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0"])
        self.freeze = ["numpy==2.1.0", "ruff==0.14.0", "torch==2.9.0"]

        for _ in range(2):
            drift.record_if_new(self.root, drift.compute(self.root), "test")
        self.freeze[1] = "ruff==0.15.0"
        drift.record_if_new(self.root, drift.compute(self.root), "test")

        recs = errorlog.read(self.cache)
        self.assertEqual([r["level"] for r in recs], ["drift", "drift"])
        self.assertEqual(recs[0]["fingerprint"], recs[1]["fingerprint"])
        self.assertEqual(recs[0]["drift"]["added"], {"ruff": "ruff==0.14.0"})
        self.assertEqual(recs[0]["env"], str(self.root))

    def test_sync_logs_prior_drift_and_folds_its_own_changes(self):
        self.set_baseline(["numpy==2.1.0", "torch==2.9.0"])
        self.freeze = ["numpy==2.1.0", "ruff==0.14.0", "torch==2.9.0"]

        with contextlib.redirect_stderr(io.StringIO()), _tracking_drift(self.root):
            self.freeze = ["numpy==2.2.0", "ruff==0.14.0", "torch==2.9.0"]

        (rec,) = errorlog.read(self.cache)
        self.assertEqual(rec["drift"]["added"], {"ruff": "ruff==0.14.0"})
        d = drift.compute(self.root, scan_files=False)
        self.assertEqual(d.added, {"ruff": "ruff==0.14.0"})
        self.assertEqual(d.changed, {})

    def test_reclone_with_template_mtimes_is_not_drift(self):
        self.set_baseline(["mistral-common==1.11.6", "torch==2.9.0"], age_s=1000)
        self.freeze = ["mistral-common @ git+https://x/mc@abc", "torch==2.9.0"]
        venv = self.root / ".venv"

        with contextlib.redirect_stderr(io.StringIO()), _tracking_drift(self.root):
            # a fresh clone of a template built after the old baseline
            os.rename(venv, self.root / "old-venv")
            self.sp.mkdir(parents=True)
            self.install("mistral-common", "1.11.6")
            self.install("torch", "2.9.0")
            built = time.time() - 500
            for d in self.sp.iterdir():
                os.utime(d, (built, built))
            self.freeze = ["mistral-common==1.11.6", "torch==2.9.0"]

        (rec,) = errorlog.read(self.cache)
        self.assertEqual(list(rec["drift"]["changed"]), ["mistral-common"])
        self.assertTrue(drift.compute(self.root).empty)

    def test_first_sync_of_legacy_env_starts_tracking(self):
        with contextlib.redirect_stderr(io.StringIO()), _tracking_drift(self.root):
            pass
        self.assertEqual(
            drift.baseline_path(self.root).read_text().split(), sorted(self.freeze)
        )


if __name__ == "__main__":
    unittest.main()
