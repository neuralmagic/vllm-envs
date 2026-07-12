"""Content-addressed DeepEP installation using vLLM's supported installer."""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
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


def _shell_default(script: str, name: str) -> str:
    pattern = rf'^{name}=\$\{{{name}:-["\']?([^"\'}} ]+)["\']?\}}'
    match = re.search(pattern, script, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"vLLM EP installer has no {name} default")
    return match.group(1)


def _docker_deepep_ref(root: Path) -> str | None:
    versions = root / "docker" / "versions.json"
    try:
        value = json.loads(versions.read_text())["variable"]["DEEPEP_COMMIT_HASH"][
            "default"
        ]
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


def resolve(root: Path, venv_hash: str) -> DeepEPResolution:
    installer = root / "tools" / "ep_kernels" / "install_python_libraries.sh"
    if not installer.is_file():
        raise RuntimeError(f"vLLM checkout has no EP installer: {installer}")
    script = installer.read_text()
    ref = _docker_deepep_ref(root) or _shell_default(script, "DEEPEP_COMMIT_HASH")
    nvshmem = _shell_default(script, "NVSHMEM_VER")
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST") or _docker_cuda_arch_list(root)
    payload = json.dumps(
        {
            "installer": hashlib.sha256(installer.read_bytes()).hexdigest(),
            "ref": ref,
            "nvshmem": nvshmem,
            "venv": venv_hash,
            "cuda": cuda_version(),
            "archs": archs,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return DeepEPResolution(key, ref, nvshmem, archs, installer)


def _wheel(entry: Path) -> Path | None:
    wheels = sorted((entry / "workspace" / "dist").glob("*.whl"))
    return wheels[-1] if wheels else None


def _ensure_wheel(cfg: Config, venv: Path, resolution: DeepEPResolution) -> Path:
    entry = cfg.store("ep-kernels") / resolution.key
    with entry_lock(entry):
        wheel = _wheel(entry)
        if wheel is not None and (entry / ".complete").exists():
            say(f"DeepEP layer: cache HIT ({resolution.key})")
            touch_last_used(entry)
            return wheel
        say(
            f"DeepEP layer: cache MISS ({resolution.key}) — building with vLLM installer"
        )
        if entry.exists():
            shutil.rmtree(entry)
        workspace = entry / "workspace"
        workspace.mkdir(parents=True)
        build_env = {"VIRTUAL_ENV": str(venv)}
        if resolution.cuda_arch_list:
            build_env["TORCH_CUDA_ARCH_LIST"] = resolution.cuda_arch_list
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
        wheel = _wheel(entry)
        if wheel is None:
            raise RuntimeError("vLLM EP installer produced no DeepEP wheel")
        (entry / ".complete").touch()
        write_meta(
            entry,
            {
                "kind": "ep-kernels",
                "hash": resolution.key,
                "deepep_ref": resolution.ref,
                "nvshmem_version": resolution.nvshmem_version,
                "cuda_arch_list": resolution.cuda_arch_list,
                "wheel": wheel.name,
            },
        )
        touch_last_used(entry)
        return wheel


def sync_deepep(cfg: Config, root: Path, venv: Path, venv_hash: str) -> None:
    """Install the vLLM-pinned DeepEP wheel into a CUDA environment."""
    installer = root / "tools" / "ep_kernels" / "install_python_libraries.sh"
    if not installer.is_file():
        say("DeepEP layer: unsupported by this vLLM commit — skipping")
        return
    resolution = resolve(root, venv_hash)
    marker = read_marker(root)
    entry = cfg.store("ep-kernels") / resolution.key
    installed = (
        run(
            [str(venv / "bin" / "python"), "-c", "import deep_ep"],
            check=False,
        ).returncode
        == 0
    )
    if (
        marker.get("deepep_hash") == resolution.key
        and (entry / ".complete").exists()
        and installed
    ):
        say(f"DeepEP layer: up to date ({resolution.key})")
        touch_last_used(entry)
        return
    wheel = _ensure_wheel(cfg, venv, resolution)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "--reinstall",
            str(wheel),
        ],
        stream_prefix="[ve]   [uv] ",
    )
    update_marker(root, deepep_hash=resolution.key)
