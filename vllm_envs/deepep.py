"""Content-addressed DeepEP installation using vLLM's supported installer."""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .extprojects import fetch_ref
from .hashing import cuda_version
from .locks import entry_lock
from .log import say
from .registry import read_marker, update_marker
from .store import touch_last_used, write_meta
from .util import run


@dataclass(frozen=True)
class DeepEPResolution:
    key: str
    ref: str
    nvshmem_version: str
    cuda_arch_list: str
    installer: Path
    nccl_version: str = ""


def _shell_default(script: str, name: str) -> str:
    pattern = rf'^{name}=\$\{{{name}:-["\']?([^"\'}} ]+)["\']?\}}'
    match = re.search(pattern, script, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"vLLM EP installer has no {name} default")
    return match.group(1)


def _docker_variable(root: Path, name: str) -> str | None:
    versions = root / "docker" / "versions.json"
    try:
        value = json.loads(versions.read_text())["variable"][name]["default"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return str(value)


def _docker_cuda_arch_list(root: Path) -> str:
    try:
        dockerfile = (root / "docker" / "Dockerfile").read_text()
    except OSError:
        return ""
    match = re.search(r"export TORCH_CUDA_ARCH_LIST=['\"]([^'\"]+)", dockerfile)
    return match.group(1) if match else ""


def _cuda_arch_list(root: Path) -> str:
    if override := os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return override
    archs = _docker_cuda_arch_list(root).split()
    proc = run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=False,
    )
    for capability in proc.stdout.splitlines():
        capability = capability.strip()
        if capability and int(capability.split(".", 1)[0]) >= 9:
            local_arch = f"{capability}a"
            if local_arch not in archs:
                archs.append(local_arch)
    return " ".join(archs)


def resolve(root: Path, venv_hash: str) -> DeepEPResolution:
    installer = root / "tools" / "ep_kernels" / "install_python_libraries.sh"
    if not installer.is_file():
        raise RuntimeError(f"vLLM checkout has no EP installer: {installer}")
    script = installer.read_text()
    ref = (
        _docker_variable(root, "DEEPEP_COMMIT_HASH")
        or _shell_default(script, "DEEPEP_COMMIT_HASH")
    )
    nvshmem = _shell_default(script, "NVSHMEM_VER")
    nccl = _docker_variable(root, "NCCL_VERSION") or ""
    archs = _cuda_arch_list(root)
    payload = json.dumps(
        {
            "installer": hashlib.sha256(installer.read_bytes()).hexdigest(),
            "ref": ref,
            "nvshmem": nvshmem,
            "nccl": nccl,
            "venv": venv_hash,
            "cuda": cuda_version(),
            "archs": archs,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return DeepEPResolution(key, ref, nvshmem, archs, installer, nccl)


def _wheels(entry: Path) -> list[Path]:
    return sorted((entry / "workspace" / "dist").glob("*.whl"))


def _prepare_sources(resolution: DeepEPResolution, workspace: Path) -> None:
    """Pre-fetch exact refs that historical installer scripts clone loosely."""
    script = resolution.installer.read_text()
    try:
        pplx_ref = _shell_default(script, "PPLX_COMMIT_HASH")
    except RuntimeError:
        pplx_ref = ""
    if pplx_ref:
        pplx_dir = workspace / "pplx-kernels"
        fetch_ref(
            "https://github.com/ppl-ai/pplx-kernels",
            pplx_ref,
            pplx_dir,
            "pplx-kernels",
        )
        # Historical pplx-kernels passed this as a malformed CMake warning
        # option. CMake 4 rejects it; the intended cache variable needs -D.
        setup = pplx_dir / "setup.py"
        if setup.is_file():
            text = setup.read_text()
            fixed = text.replace('"-WITH_TESTS=OFF"', '"-DWITH_TESTS=OFF"')
            if fixed != text:
                setup.write_text(fixed)
    fetch_ref(
        "https://github.com/deepseek-ai/DeepEP",
        resolution.ref,
        workspace / "DeepEP",
        "DeepEP",
    )


def _ensure_wheels(
    cfg: Config, venv: Path, resolution: DeepEPResolution
) -> list[Path]:
    entry = cfg.store("ep-kernels") / resolution.key
    with entry_lock(entry):
        wheels = _wheels(entry)
        if wheels and (entry / ".complete").exists():
            say(f"DeepEP layer: cache HIT ({resolution.key})")
            touch_last_used(entry)
            return wheels
        say(
            f"DeepEP layer: cache MISS ({resolution.key}) — building with vLLM installer"
        )
        if entry.exists():
            shutil.rmtree(entry)
        workspace = entry / "workspace"
        workspace.mkdir(parents=True)
        _prepare_sources(resolution, workspace)
        build_env = {"VIRTUAL_ENV": str(venv)}
        if resolution.cuda_arch_list:
            build_env["TORCH_CUDA_ARCH_LIST"] = resolution.cuda_arch_list
        if override := _nccl_override(venv, resolution, entry / "nccl-override.txt"):
            build_env["UV_OVERRIDE"] = override
        if nccl_lib := _nccl_lib_dir(venv):
            for var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                current = os.environ.get(var, "")
                build_env[var] = f"{nccl_lib}:{current}" if current else nccl_lib
        run(
            [
                "bash",
                str(resolution.installer),
                "--workspace",
                str(workspace),
                "--mode",
                "wheel",
                "--deepep-ref",
                resolution.ref,
                "--nvshmem-ver",
                resolution.nvshmem_version,
            ],
            env=build_env,
            stream_prefix="[ve]   [DeepEP] ",
        )
        wheels = _wheels(entry)
        if not any(w.name.lower().startswith("deep_ep-") for w in wheels):
            raise RuntimeError("vLLM EP installer produced no deep_ep wheel")
        (entry / ".complete").touch()
        write_meta(
            entry,
            {
                "kind": "ep-kernels",
                "hash": resolution.key,
                "deepep_ref": resolution.ref,
                "nvshmem_version": resolution.nvshmem_version,
                "cuda_arch_list": resolution.cuda_arch_list,
                "wheels": [wheel.name for wheel in wheels],
            },
        )
        touch_last_used(entry)
        return wheels


_NCCL_LIB_DIR_SNIPPET = (
    "import importlib.util as u, pathlib;"
    "s = u.find_spec('nvidia.nccl');"
    "print(pathlib.Path(list(s.submodule_search_locations)[0], 'lib') if s else '')"
)


def _nccl_lib_dir(venv: Path) -> str:
    """Where the venv's NCCL wheel keeps libnccl.so.2.

    DeepEP links NCCL with `-l:libnccl.so.2` but — unlike NVSHMEM — never adds
    that wheel's lib dir to library_dirs, so the link only works when NCCL is on
    the default search path (as in vLLM's image, not in a plain venv)."""
    out = run(
        [str(venv / "bin" / "python"), "-c", _NCCL_LIB_DIR_SNIPPET], check=False
    )
    path = out.stdout.strip() if out.returncode == 0 else ""
    return path if path and Path(path).is_dir() else ""


def _nccl_override(venv: Path, resolution: DeepEPResolution, dest: Path) -> str | None:
    """Write the uv override that keeps NCCL at the version vLLM's image builds
    DeepEP against, and return its path.

    DeepEP's Gin backend needs NCCL >= 2.30.4 (ncclGinRequest_t and friends) and
    resolves NCCL from the Python environment, but torch requires an exact older
    nvidia-nccl wheel — so any resolve inside the EP installer (it runs
    `uv pip install cmake torch ninja`) drags NCCL back down and the build dies
    on missing GIN types.  docker/Dockerfile handles this with UV_OVERRIDE for
    CUDA 13; mirror it, and only when that wheel is in the venv at all (CUDA 12
    images do not pin NCCL either)."""
    if not resolution.nccl_version:
        return None
    package = "nvidia-nccl-cu13"
    shown = run(
        ["uv", "pip", "show", "--python", str(venv / "bin" / "python"), package],
        check=False,
    )
    if shown.returncode != 0:
        return None
    installed = re.search(r"^Version:\s*(\S+)", shown.stdout, re.MULTILINE)
    if not installed or installed.group(1) != resolution.nccl_version:
        say(
            f"DeepEP layer: overriding {package}=={resolution.nccl_version} for "
            f"the NCCL Gin backend (venv has "
            f"{installed.group(1) if installed else 'none'})"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"{package}=={resolution.nccl_version}\n")
    return str(dest)


def sync_deepep(cfg: Config, root: Path, venv: Path, venv_hash: str) -> None:
    """Install the vLLM-pinned DeepEP wheel into a CUDA environment."""
    installer = root / "tools" / "ep_kernels" / "install_python_libraries.sh"
    if not installer.is_file():
        say("DeepEP layer: unsupported by this vLLM commit — skipping")
        return
    resolution = resolve(root, venv_hash)
    marker = read_marker(root)
    entry = cfg.store("ep-kernels") / resolution.key
    cached_wheels = _wheels(entry)
    imports = ["deep_ep"]
    if any(w.name.lower().startswith("pplx_kernels-") for w in cached_wheels):
        imports.append("pplx_kernels")
    installed = (
        run(
            [
                str(venv / "bin" / "python"),
                "-c",
                f"import {', '.join(imports)}",
            ],
            check=False,
        ).returncode
        == 0
    )
    if (
        marker.get("deepep_hash") == resolution.key
        and (entry / ".complete").exists()
        and cached_wheels
        and installed
    ):
        say(f"DeepEP layer: up to date ({resolution.key})")
        touch_last_used(entry)
        return
    wheels = _ensure_wheels(cfg, venv, resolution)
    # The deep_ep wheel requires the Gin-capable NCCL that torch pins away from,
    # so this install needs the same override the build used.
    install_env = {}
    if override := _nccl_override(venv, resolution, entry / "nccl-override.txt"):
        install_env["UV_OVERRIDE"] = override
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "--reinstall",
            *(str(wheel) for wheel in wheels),
        ],
        env=install_env or None,
        stream_prefix="[ve]   [uv] ",
    )
    update_marker(root, deepep_hash=resolution.key)
