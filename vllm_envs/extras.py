"""Install vLLM's supported optional runtime bundle."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .deepep import resolve as resolve_deepep
from .deepep import sync_deepep
from .log import say
from .registry import read_marker, update_marker
from .util import run


@dataclass(frozen=True)
class ExtrasResolution:
    key: str
    kv_installer: Path | None
    kv_requirements: Path | None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, venv_hash: str) -> ExtrasResolution:
    parts = [f"venv={venv_hash}"]
    try:
        parts.append(f"ep={resolve_deepep(root, venv_hash).key}")
    except RuntimeError:
        parts.append("ep=unsupported")

    installer = root / ".buildkite" / "scripts" / "install-kv-connectors.sh"
    requirements = root / "requirements" / "kv_connectors.txt"
    if installer.is_file() and requirements.is_file():
        parts.extend(
            (f"kv-installer={_digest(installer)}", f"kv-req={_digest(requirements)}")
        )
        kv_installer: Path | None = installer
        kv_requirements: Path | None = requirements
    else:
        parts.append("kv=unsupported")
        kv_installer = None
        kv_requirements = None

    key = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
    return ExtrasResolution(key, kv_installer, kv_requirements)


def _has_import(venv: Path, module: str) -> bool:
    return (
        run(
            [str(venv / "bin" / "python"), "-c", f"import {module}"],
            check=False,
        ).returncode
        == 0
    )


def _sync_kv_connectors(venv: Path, resolution: ExtrasResolution) -> None:
    if resolution.kv_installer is None or resolution.kv_requirements is None:
        say("vLLM extras: KV connector installer unsupported by this commit — skipping")
        return
    if _has_import(venv, "nixl"):
        say("vLLM extras: KV connectors already installed")
        return

    # The upstream CI helper deliberately targets a container's system Python.
    # Run the same maintained logic against this managed virtual environment.
    script = resolution.kv_installer.read_text().replace(" --system", "")
    with tempfile.TemporaryDirectory(prefix="ve-vllm-extras-") as tmp:
        installer = Path(tmp) / "install-kv-connectors.sh"
        installer.write_text(script)
        env = {
            "VIRTUAL_ENV": str(venv),
            "PATH": f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            "KV_CONNECTORS_REQUIREMENTS": str(resolution.kv_requirements),
        }
        run(
            ["bash", str(installer)],
            env=env,
            stream_prefix="[ve]   [vLLM extras] ",
        )


def sync_vllm_extras(cfg: Config, root: Path, venv: Path, venv_hash: str) -> None:
    """Install vLLM's EP kernels and KV connectors as one supported bundle."""
    resolution = resolve(root, venv_hash)
    marker = read_marker(root)
    kv_ready = resolution.kv_installer is None or _has_import(venv, "nixl")
    if marker.get("vllm_extras_hash") == resolution.key and kv_ready:
        try:
            deep_ep_supported = resolve_deepep(root, venv_hash)
        except RuntimeError:
            deep_ep_supported = None
        if deep_ep_supported is None or _has_import(venv, "deep_ep"):
            say(f"vLLM extras: up to date ({resolution.key})")
            return

    sync_deepep(cfg, root, venv, venv_hash)
    _sync_kv_connectors(venv, resolution)
    update_marker(root, vllm_extras_hash=resolution.key)
