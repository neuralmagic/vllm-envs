import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs.editable import capture, editable_present, replay


class EditableConsoleScriptTest(unittest.TestCase):
    def test_capture_and_replay_console_script(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        old_root = root / "old"
        new_root = root / "new"
        store = root / "store"
        old_venv = old_root / ".venv"
        new_venv = new_root / ".venv"
        old_sp = old_venv / "lib" / "python3.12" / "site-packages"
        new_sp = new_venv / "lib" / "python3.12" / "site-packages"
        dist = old_sp / "vllm-1.dist-info"
        dist.mkdir(parents=True)
        (old_sp / "__editable__.vllm-1.pth").write_text(str(old_root))
        (old_sp / "__editable___vllm_1_finder.py").write_text(str(old_root))
        (dist / "entry_points.txt").write_text(
            "[console_scripts]\nvllm = vllm.entrypoints.cli.main:main\n"
        )
        (dist / "RECORD").write_text("vllm-1.dist-info/RECORD,,\n")
        script = old_venv / "bin" / "vllm"
        script.parent.mkdir(parents=True)
        script.write_text(f"#!{old_venv}/bin/python\n")
        wheel = root / "vllm.whl"
        with zipfile.ZipFile(wheel, "w"):
            pass

        with (
            patch("vllm_envs.editable._head", return_value="head"),
            patch("vllm_envs.editable.run", return_value=SimpleNamespace(stdout="")),
        ):
            capture(old_root, old_venv, wheel, store)
            new_sp.mkdir(parents=True)
            self.assertTrue(replay(new_root, new_venv, store))

        replayed = new_venv / "bin" / "vllm"
        self.assertTrue(editable_present(new_venv))
        self.assertIn(str(new_venv), replayed.read_text())
        self.assertTrue(replayed.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
