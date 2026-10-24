"""Content-addressed layer key computation.

All hashes are computed from *working-tree contents* (tracked + modified +
untracked files), never from the commit object, so dirty trees always get
their own keys and can never collide with clean-commit artifacts.
"""

import hashlib
import json
import os
import platform as host_platform
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from shutil import which

from . import rules
from .config import DEFAULT_CACHE_DIR
from .sitepkgs import dist_infos
from .treefacts import ReqLayout, TreeFacts, facts_for
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
        # Newer drivers label this "CUDA UMD Version" while older releases
        # use "CUDA Version".
        if m := re.search(r"CUDA(?: UMD)? Version:\s*(\d+\.\d+)", out):
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requirements_layout(
    root: Path, platform: str, with_test: bool = False,
    facts: TreeFacts | None = None,
) -> ReqLayout:
    return (facts or facts_for(root, platform)).layout(with_test)


def _derived_requirement_parts(files: list[Path], facts: TreeFacts) -> list[str]:
    """Venv-key parts of the rules that fire for these requirement files."""
    return rules.key_parts(facts.context(files), "venv")


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
    facts: TreeFacts | None = None


def venv_keys(
    root: Path, platform: str, python: str, with_test: bool = False,
    cap: str = "minor", facts: TreeFacts | None = None,
) -> VenvKeys:
    facts = facts or facts_for(root, platform)
    layout = facts.layout(with_test)
    cuda = cuda_version() if platform == "cuda" else "n/a"
    base_parts = [
        "v1",
        platform,
        python,
        cuda,
        # cap="none" stays un-folded so pre-capping templates keep their keys
        *([f"cap:{cap}"] if cap != "none" else []),
        *(f"{p.name}:{file_sha256(p)}" for p in layout.build_files),
        *_derived_requirement_parts(layout.build_files, facts),
        # torch pins from runtime files fold into the base key so a torch bump
        # rebuilds the base template instead of a heavy top-up in the full layer
        *torch_pin_lines(layout.runtime_files),
    ]
    base_hash = _sha(base_parts)
    full_parts = [
        base_hash,
        *(f"{p.relative_to(root)}:{file_sha256(p)}" for p in layout.runtime_files),
        *(f"{p.relative_to(root)}:{file_sha256(p)}" for p in layout.test_files),
        *_derived_requirement_parts(
            [*layout.runtime_files, *layout.test_files], facts
        ),
    ]
    return VenvKeys(base_hash, _sha(full_parts), layout, facts)


# Several vLLM releases generate Marlin instantiations into csrc; they survive
# checkouts as untracked build outputs, not source edits.
_GENERATED_OUTPUT = re.compile(
    r"^csrc/(?:libtorch_stable/)?(?:quantization/marlin|moe/marlin_moe_wna16)/"
    r"(?:sm\d+_kernel_.*\.cu|kernel_selector\.h)$"
)


def _generated_output(path: str) -> bool:
    return bool(_GENERATED_OUTPUT.match(path))


def worktree_content_hash(root: Path, paths: tuple[str, ...]) -> str:
    """Hash working-tree contents of the given paths (tracked+modified+untracked)."""
    tracked = run(
        ["git", "ls-files", "-cm", "-z", "--", *paths],
        cwd=root,
    ).stdout.split("\0")
    untracked = run(
        ["git", "ls-files", "-o", "--exclude-standard", "-z", "--", *paths],
        cwd=root,
    ).stdout.split("\0")
    # Never ignore a tracked path, even if a release starts committing one.
    files = sorted(
        {f for f in tracked if f}
        | {f for f in untracked if f and not _generated_output(f)}
    )
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


# Variables vLLM's setup.py/CMake (or the compilers they drive) read when
# building extensions. An allowlist: runtime knobs (VLLM_LOGGING_LEVEL,
# CUDA_LAUNCH_BLOCKING, ...) and GPU reservations must not split the cache.
# *_SRC_DIR overrides are handled separately (they disable build caching).
_BUILD_ENV_NAMES = {
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CMAKE_ARGS",
    "CMAKE_BUILD_TYPE", "CUDA_HOME", "CUDA_PATH", "NVCC_APPEND_FLAGS",
    "NVCC_PREPEND_FLAGS", "NVCC_THREADS", "TORCH_CUDA_ARCH_LIST",
    "PYTORCH_ROCM_ARCH", "ROCM_PATH", "ACL_ROOT_DIR", "TORCH_NIGHTLY",
    "VLLM_TARGET_DEVICE", "VLLM_MAIN_CUDA_VERSION", "VLLM_CPU_X86",
    "VLLM_CPU_ARM_BF16", "VLLM_CPU_ARM_I8MM", "VLLM_CPU_RVV_BF16",
    "VLLM_REQUIRE_RUST_FRONTEND", "VLLM_VERSION_OVERRIDE",
    "VLLM_SKIP_VERSION_SUFFIX", "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX",
}


def _command_identity(
    command: str, *args: str, normalize_root: Path | None = None
) -> str:
    """Return a stable tool identity, optionally relative to a movable root."""
    executable = which(command)
    if executable is None:
        return f"{command}:missing"
    try:
        proc = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=10
        )
        output = (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        output = f"error:{type(exc).__name__}"
    # Resolve symlinks such as /usr/local/cuda -> cuda-13.0.
    identity = Path(executable).resolve()
    if normalize_root is not None:
        try:
            identity = Path("<venv>") / identity.relative_to(
                normalize_root.resolve()
            )
        except ValueError:
            pass
    return f"{identity}\n{output}"


def _distribution_identity(venv: Path, package: str) -> str:
    """Fingerprint an installed wheel without importing its package."""
    candidates = dist_infos(venv, package)
    if not candidates:
        return f"{package}:missing"
    dist = candidates[-1]
    parts = [dist.name]
    # RECORD captures the wheel's file hashes. METADATA and WHEEL cover
    # incomplete/nonstandard RECORD files.
    for name in ("METADATA", "WHEEL", "RECORD"):
        path = dist / name
        parts.append(f"{name}:{file_sha256(path) if path.is_file() else 'missing'}")
    return "\n".join(parts)


def _build_distributions_identity(venv: Path) -> str:
    # setup.py executes against this venv, so these packages can affect the
    # generated wheel even when the source and system compiler are unchanged.
    return "\n".join(
        _distribution_identity(venv, package)
        for package in ("torch", "setuptools", "wheel", "cmake", "ninja")
    )


def _python_abi_identity(venv: Path, configured_python: str) -> str:
    cfg = venv / "pyvenv.cfg"
    values: dict[str, str] = {}
    if cfg.is_file():
        for line in cfg.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip().lower()] = value.strip()
    return ":".join([
        configured_python,
        values.get("implementation", "cpython"),
        values.get("version", values.get("version_info", "unknown")),
    ])


def _cuda_toolkit_identity() -> str:
    for variable in ("CMAKE_CUDA_COMPILER", "NVCC"):
        if value := os.environ.get(variable):
            path = Path(value).expanduser()
            if path.is_file():
                return _command_identity(str(path), "--version")
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if value := os.environ.get(variable):
            path = Path(value).expanduser() / "bin" / "nvcc"
            if path.is_file():
                return _command_identity(str(path), "--version")
    default = Path("/usr/local/cuda/bin/nvcc")
    if default.is_file():
        return _command_identity(str(default), "--version")
    return _command_identity("nvcc", "--version")


def _gpu_target_identity() -> str:
    if arches := os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return f"TORCH_CUDA_ARCH_LIST={arches}"
    # Reading the driver's inventory is effectively free, unlike spawning
    # nvidia-smi (~100-500ms). Auto-detected targets depend on the *set* of
    # visible GPU models: on a homogeneous host that set is the same whatever
    # CUDA_VISIBLE_DEVICES says, so only mixed hosts key on visibility.
    inventory = Path("/proc/driver/nvidia/gpus")
    models: list[str] = []
    if inventory.is_dir():
        for info in sorted(inventory.glob("*/information")):
            try:
                model = next(
                    line.partition(":")[2].strip()
                    for line in info.read_text(errors="replace").splitlines()
                    if line.startswith("Model:")
                )
            except (OSError, StopIteration):
                continue
            models.append(model)
    if models:
        distinct = sorted(set(models))
        identity = "driver-models:" + "|".join(distinct)
        if len(distinct) > 1:
            identity += ":visible=" + os.environ.get("CUDA_VISIBLE_DEVICES", "*")
        return identity
    executable = which("nvidia-smi")
    if executable is None:
        return "gpu-target:unavailable"
    # PyTorch auto-detects the visible GPUs when no target list is supplied.
    # Model names conservatively distinguish targets on older drivers that do
    # not expose the compute_cap query field.
    for fields in ("compute_cap,name", "name"):
        try:
            proc = subprocess.run(
                [executable, f"--query-gpu={fields}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            rows = sorted({line.strip() for line in proc.stdout.splitlines()
                           if line.strip()})
            if proc.returncode == 0 and rows:
                return f"{fields}:" + "|".join(rows)
        except (OSError, subprocess.SubprocessError):
            break
    return "gpu-target:unknown"


def _build_environment_identity() -> str:
    values = {
        name: value for name, value in os.environ.items()
        if name in _BUILD_ENV_NAMES
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def build_key(root: Path, platform: str, python: str, venv: Path) -> str:
    """Hash every input that can affect a reusable compiled vLLM wheel."""
    cuda = cuda_version() if platform == "cuda" else "n/a"
    tree = worktree_content_hash(root, facts_for(root, platform).build_layer_paths)
    def venv_tool(name: str) -> str:
        candidate = venv / "bin" / name
        return str(candidate) if candidate.is_file() else name

    compiler = "\n".join([
        _command_identity(os.environ.get("CC", "cc"), "--version"),
        _command_identity(os.environ.get("CXX", "c++"), "--version"),
        _command_identity(venv_tool("cmake"), "--version", normalize_root=venv),
        _command_identity(venv_tool("ninja"), "--version", normalize_root=venv),
    ])
    libc_name, libc_version = host_platform.libc_ver()
    parts = [
        # v3: local compilation no longer hashes its build directory in
        # ccache. Keep directory-sensitive local wheels in their old keys.
        "v3", platform,
        _python_abi_identity(venv, python),
        f"host:{host_platform.system()}:{host_platform.machine()}:{libc_name}:{libc_version}",
        f"cuda-driver:{cuda}", f"source:{tree}",
        f"build-distributions:{_build_distributions_identity(venv)}",
        f"compiler:{compiler}",
        f"env:{_build_environment_identity()}",
    ]
    if platform == "cuda":
        parts.extend([
            f"cuda-toolkit:{_cuda_toolkit_identity()}",
            f"gpu-target:{_gpu_target_identity()}",
        ])
    return _sha(parts)


def build_paths_dirty(root: Path, paths: tuple[str, ...] | None = None) -> bool:
    """True if the working tree is dirty for the given (build-layer) paths."""
    if paths is None:
        paths = facts_for(root, detect_platform()).build_layer_paths
    lines = run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=root,
    ).stdout.splitlines()
    for line in lines:
        if line.startswith("?? ") and _generated_output(line[3:]):
            continue
        return True
    return False
