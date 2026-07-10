"""Content-addressed layer key computation.

All hashes are computed from *working-tree contents* (tracked + modified +
untracked files), never from the commit object, so dirty trees always get
their own keys and can never collide with clean-commit artifacts.
"""

import hashlib
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from shutil import which

from .config import BUILD_LAYER_PATHS
from .util import run

TORCH_PIN_RE = re.compile(r"^(torch|torchaudio|torchvision|nvidia-|triton)", re.I)


@lru_cache(maxsize=1)
def detect_platform() -> str:
    if which("nvidia-smi"):
        return "cuda"
    if which("rocminfo") or Path("/opt/rocm").is_dir():
        return "rocm"
    return "cpu"


@lru_cache(maxsize=8)
def cuda_version() -> str:
    try:
        out = run(["nvidia-smi"], check=False).stdout
        if m := re.search(r"CUDA Version:\s*(\d+\.\d+)", out):
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def _sha(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ReqLayout:
    """Platform-relevant requirements files for a given worktree."""

    build_files: list[Path]  # heavy layer (torch + build deps)
    runtime_files: list[Path]  # top-up layer
    recognized: bool


def requirements_layout(root: Path, platform: str, with_test: bool = False) -> ReqLayout:
    req = root / "requirements"

    def existing(*paths: Path) -> list[Path]:
        return [p for p in paths if p.is_file()]

    # Modern layout: requirements/build/<platform>.txt
    build = existing(req / "build" / f"{platform}.txt")
    if not build:
        # Mid-era layout: requirements/build.txt
        build = existing(req / "build.txt")
    runtime = existing(
        req / "common.txt",
        req / f"{platform}.txt",
        req / "lint.txt",
    )
    if not runtime:
        # Old layout: root-level requirements-*.txt
        runtime = existing(
            root / "requirements-common.txt",
            root / f"requirements-{platform}.txt",
            root / "requirements-lint.txt",
        )
        build = build or existing(root / "requirements-build.txt")
    if with_test:
        runtime += existing(
            req / "test" / f"{platform}.in",
            req / "test.in",
            root / "requirements-test.txt",
        )

    if build and runtime:
        return ReqLayout(build, runtime, recognized=True)

    # Unrecognized layout: hash everything requirements-ish (coarse but correct).
    all_reqs = sorted(
        set(root.glob("requirements*")) | set(req.rglob("*")) if req.is_dir() else set(root.glob("requirements*"))
    )
    all_reqs = [p for p in all_reqs if p.is_file()]
    return ReqLayout(all_reqs, all_reqs, recognized=False)


def torch_pin_lines(files: list[Path]) -> list[str]:
    lines = []
    for f in files:
        for line in f.read_text(errors="replace").splitlines():
            if TORCH_PIN_RE.match(line.strip()):
                lines.append(line.strip())
    return sorted(lines)


@dataclass
class VenvKeys:
    base_hash: str
    full_hash: str
    layout: ReqLayout


def venv_keys(
    root: Path, platform: str, python: str, with_test: bool = False
) -> VenvKeys:
    layout = requirements_layout(root, platform, with_test)
    cuda = cuda_version() if platform == "cuda" else "n/a"
    base_parts = [
        "v1",
        platform,
        python,
        cuda,
        *(f"{p.name}:{_hash_file(p)}" for p in layout.build_files),
        # torch pins from runtime files fold into the base key so a torch bump
        # rebuilds the base template instead of a heavy top-up in the full layer
        *torch_pin_lines(layout.runtime_files),
    ]
    base_hash = _sha(base_parts)
    full_parts = [
        base_hash,
        *(f"{p.relative_to(root)}:{_hash_file(p)}" for p in layout.runtime_files),
    ]
    return VenvKeys(base_hash, _sha(full_parts), layout)


def worktree_content_hash(root: Path, paths: tuple[str, ...]) -> str:
    """Hash working-tree contents of the given paths (tracked+modified+untracked)."""
    listed = run(
        ["git", "ls-files", "-cmo", "--exclude-standard", "-z", "--", *paths],
        cwd=root,
    ).stdout
    files = sorted({f for f in listed.split("\0") if f})
    present = [f for f in files if (root / f).is_file()]
    deleted = [f for f in files if not (root / f).exists()]
    blob_hashes: list[str] = []
    if present:
        proc = subprocess.run(
            ["git", "hash-object", "--stdin-paths"],
            cwd=root,
            input="\n".join(present),
            capture_output=True,
            text=True,
            check=True,
        )
        blob_hashes = proc.stdout.split()
    parts = [f"{p}={h}" for p, h in zip(present, blob_hashes)]
    parts += [f"{p}=DELETED" for p in deleted]
    return _sha(["v1", *parts])


def build_key(root: Path, platform: str, python: str) -> str:
    cuda = cuda_version() if platform == "cuda" else "n/a"
    tree = worktree_content_hash(root, BUILD_LAYER_PATHS)
    return _sha(["v1", platform, python, cuda, tree])


def build_paths_dirty(root: Path) -> bool:
    """True if the working tree is dirty for build-layer paths."""
    out = run(
        ["git", "status", "--porcelain", "--", *BUILD_LAYER_PATHS], cwd=root
    ).stdout.strip()
    return bool(out)
