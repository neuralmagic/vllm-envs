"""Content-addressed layer key computation.

All hashes are computed from *working-tree contents* (tracked + modified +
untracked files), never from the commit object, so dirty trees always get
their own keys and can never collide with clean-commit artifacts.
"""

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from shutil import which

from .config import BUILD_LAYER_PATHS, DEFAULT_CACHE_DIR
from .util import run

TORCH_PIN_RE = re.compile(r"^(torch|torchaudio|torchvision|nvidia-|triton)", re.I)


@lru_cache(maxsize=1)
def detect_platform() -> str:
    if which("nvidia-smi"):
        return "cuda"
    if which("rocminfo") or Path("/opt/rocm").is_dir():
        return "rocm"
    return "cpu"


def _cuda_cache_file() -> Path:
    root = Path(os.environ.get("VE_CACHE_DIR") or DEFAULT_CACHE_DIR)
    return root / "cuda-version.json"


@lru_cache(maxsize=1)
def cuda_version() -> str:
    # nvidia-smi costs ~0.8s per run — cache on disk keyed by the driver line
    try:
        driver = Path("/proc/driver/nvidia/version").read_text().splitlines()[0]
    except (OSError, IndexError):
        driver = ""
    cache = _cuda_cache_file()
    if driver:
        try:
            data = json.loads(cache.read_text())
            if data.get("driver") == driver and data.get("cuda"):
                return data["cuda"]
        except (OSError, json.JSONDecodeError):
            pass
    ver = "unknown"
    try:
        out = run(["nvidia-smi"], check=False).stdout
        if m := re.search(r"CUDA Version:\s*(\d+\.\d+)", out):
            ver = m.group(1)
    except Exception:
        pass
    if driver and ver != "unknown":
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"driver": driver, "cuda": ver}))
        except OSError:
            pass
    return ver


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


_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_FLOOR_RE = re.compile(r">=?\s*v?(\d+(?:\.\d+)*)")
_CEILING_RE = re.compile(r"<|==|~=")


def cap_constraints(files: list[Path], mode: str) -> str:
    """Constraint lines capping requirements that have a version floor but no
    ceiling: `pkg>=X.Y.Z` → `pkg<X.(Y+1)` (minor) or `pkg<(X+1)` (major), so
    unpinned deps resolve near the floor's era instead of to latest. The
    highest floor wins when a package appears in several files."""
    if mode == "none":
        return ""
    floors: dict[str, tuple[tuple[int, ...], str]] = {}
    for f in files:
        for raw in f.read_text(errors="replace").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-") or "://" in line:
                continue
            spec = line.split(";", 1)[0].strip()
            m = _REQ_NAME_RE.match(spec)
            if not m:
                continue
            name, rest = m.group(1), spec[len(m.group(1)):]
            if _CEILING_RE.search(rest):
                continue
            fm = _FLOOR_RE.search(rest)
            if not fm:
                continue
            floor = tuple(int(x) for x in fm.group(1).split("."))
            key = name.lower().replace("_", "-")
            if key not in floors or floor > floors[key][0]:
                floors[key] = (floor, name)
    lines = []
    for _, (floor, name) in sorted(floors.items()):
        parts = [*floor, 0, 0]
        bound = f"{parts[0] + 1}" if mode == "major" else f"{parts[0]}.{parts[1] + 1}"
        lines.append(f"{name}<{bound}\n")
    return "".join(lines)


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
    root: Path, platform: str, python: str, with_test: bool = False,
    cap: str = "minor",
) -> VenvKeys:
    layout = requirements_layout(root, platform, with_test)
    cuda = cuda_version() if platform == "cuda" else "n/a"
    base_parts = [
        "v1",
        platform,
        python,
        cuda,
        # cap="none" stays un-folded so pre-capping templates keep their keys
        *([f"cap:{cap}"] if cap != "none" else []),
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
