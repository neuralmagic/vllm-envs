"""Build-time system headers vLLM's image installs but a host may lack."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .config import Config
from .locks import entry_lock
from .log import say, warn
from .store import touch_last_used

# 8263ea12bd (#56876) moved DeepGEMM to a deep_jit that includes
# <elfutils/libdwfl.h>; vLLM's image installs elfutils-devel. Only the headers
# are needed: libdw.so.1 is dlopen'ed at runtime.
ELFUTILS_VERSION = "0.194"
ELFUTILS_URL = (
    f"https://sourceware.org/elfutils/ftp/{ELFUTILS_VERSION}/"
    f"elfutils-{ELFUTILS_VERSION}.tar.bz2"
)
ELFUTILS_SHA256 = "09e2ff033d39baa8b388a2d7fbc5390bfde99ae3b7c67c7daaf7433fbcf0f01e"
# include dir -> {tarball path: header}; libelf headers (elfutils-libelf-devel)
# get their own dir so they never shadow a host's own copies.
ELFUTILS_HEADERS = {
    "include": {
        "libdwfl/libdwfl.h": "elfutils/libdwfl.h",
        "libdw/libdw.h": "elfutils/libdw.h",
        "version.h": "elfutils/version.h",
        "libdw/dwarf.h": "dwarf.h",
    },
    "include-libelf": {
        "libelf/libelf.h": "libelf.h",
        "libelf/gelf.h": "gelf.h",
    },
}


def _compiler_finds(header: str) -> bool:
    cxx = os.environ.get("CXX") or shutil.which("c++") or "g++"
    try:
        proc = subprocess.run(
            [cxx, "-E", "-x", "c++", "-", "-o", os.devnull],
            input=f"#include <{header}>\n", capture_output=True, text=True,
        )
    except OSError:
        return True  # no compiler to probe: leave the build to report it
    return proc.returncode == 0


def _fetch_elfutils_headers(dest: Path) -> None:
    with urllib.request.urlopen(ELFUTILS_URL, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ELFUTILS_SHA256:
        raise RuntimeError(f"{ELFUTILS_URL}: sha256 {digest} != {ELFUTILS_SHA256}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
        for subdir, headers in ELFUTILS_HEADERS.items():
            for src, rel in headers.items():
                member = tar.extractfile(f"elfutils-{ELFUTILS_VERSION}/{src}")
                (dest / subdir / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / subdir / rel).write_bytes(member.read())


def elfutils_includes(cfg: Config) -> list[Path]:
    """Include dirs with the elfutils headers the compiler lacks."""
    if _compiler_finds("elfutils/libdwfl.h"):
        return []
    entry = cfg.store("ext-src") / f"elfutils-headers-{ELFUTILS_VERSION}"
    with entry_lock(entry):
        if not (entry / ".complete").exists():
            say(f"elfutils-devel headers missing; fetching elfutils {ELFUTILS_VERSION} headers")
            shutil.rmtree(entry, ignore_errors=True)
            try:
                _fetch_elfutils_headers(entry)
            except Exception as e:
                warn(f"cannot fetch elfutils headers ({e}); DeepGEMM builds "
                     "need elfutils-devel installed")
                return []
            (entry / ".complete").touch()
        touch_last_used(entry)
    dirs = [entry / "include"]
    if not _compiler_finds("libelf.h"):
        dirs.append(entry / "include-libelf")
    return dirs


def include_env(cfg: Config) -> dict[str, str]:
    """CPATH additions for the headers this host lacks."""
    dirs = [str(d) for d in elfutils_includes(cfg)]
    if not dirs:
        return {}
    if existing := os.environ.get("CPATH"):
        dirs.append(existing)
    return {"CPATH": os.pathsep.join(dirs)}
