import contextlib
import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_envs import sysdeps


def _tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for headers in sysdeps.ELFUTILS_HEADERS.values():
            for src in headers:
                data = f"/* {src} */\n".encode()
                info = tarfile.TarInfo(f"elfutils-{sysdeps.ELFUTILS_VERSION}/{src}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class ElfutilsHeadersTest(unittest.TestCase):
    def setUp(self):
        cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = SimpleNamespace(store=lambda name: cache / name)
        self.data = _tarball()
        self.enterContext(patch.object(
            sysdeps, "ELFUTILS_SHA256", hashlib.sha256(self.data).hexdigest()))
        self.enterContext(patch.dict(os.environ, {"CPATH": ""}))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.fetches = 0

    def urlopen(self, url, timeout):
        self.fetches += 1
        return contextlib.nullcontext(io.BytesIO(self.data))

    def env(self, present: set[str]) -> dict[str, str]:
        with patch.object(sysdeps, "_compiler_finds", lambda h: h in present), \
             patch.object(sysdeps.urllib.request, "urlopen", self.urlopen):
            return sysdeps.include_env(self.cfg)

    def test_nothing_when_the_host_has_elfutils_devel(self):
        self.assertEqual(self.env({"elfutils/libdwfl.h", "libelf.h"}), {})
        self.assertEqual(self.fetches, 0)

    def test_fetches_once_and_never_shadows_host_libelf(self):
        cpath = self.env({"libelf.h"})["CPATH"].split(os.pathsep)
        self.assertEqual(len(cpath), 1)
        self.assertTrue((Path(cpath[0]) / "elfutils/libdwfl.h").is_file())
        self.assertTrue((Path(cpath[0]) / "dwarf.h").is_file())
        self.env({"libelf.h"})
        self.assertEqual(self.fetches, 1)

    def test_adds_libelf_headers_only_when_missing(self):
        cpath = self.env(set())["CPATH"].split(os.pathsep)
        self.assertEqual([Path(p).name for p in cpath], ["include", "include-libelf"])
        self.assertTrue((Path(cpath[1]) / "gelf.h").is_file())

    def test_checksum_mismatch_leaves_the_build_unchanged(self):
        self.data = b"tampered"
        self.assertEqual(self.env(set()), {})
        self.data = _tarball()
        self.assertIn("CPATH", self.env(set()))


if __name__ == "__main__":
    unittest.main()
