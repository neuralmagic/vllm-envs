"""Install vLLM's supported optional runtime bundle."""

import dataclasses
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import rules
from .config import Config
from .deepep import resolve as resolve_deepep
from .deepep import sync_deepep
from .hashing import file_sha256
from .log import say
from .registry import read_marker, update_marker
from .sitepkgs import dists_installed
from .treefacts import KV_INSTALLER, facts_for
from .util import run_retrying_transient


@dataclass(frozen=True)
class ExtrasResolution:
    key: str
    kv_installer: Path | None
    kv_requirements: Path | None
    # Fired rules' kv-layer source patches and their key parts.
    kv_patches: tuple = ()
    kv_rules: tuple[str, ...] = ()
    deepep: bool = False


def resolve(root: Path, base_hash: str, full_hash: str) -> ExtrasResolution:
    """DeepEP builds only against torch/NCCL, which the base venv fixes, so it
    keys on base_hash: a requirements or test-lock change must not force a
    DeepEP/NVSHMEM rebuild. KV connectors install against the full venv."""
    facts = facts_for(root, "cuda")
    parts = [f"venv={full_hash}"]
    if facts.ep_supported:
        parts.append(f"ep={resolve_deepep(root, base_hash, facts).key}")
    else:
        parts.append("ep=unsupported")

    if not facts.kv_supported:
        parts.append("kv=unsupported")
        key = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
        return ExtrasResolution(key, None, None, deepep=facts.ep_supported)
    ctx = facts.context()
    resolution = ExtrasResolution(
        "", facts.kv_installer, facts.kv_requirements,
        kv_patches=tuple(
            e for _, e in rules.effects(ctx, rules.PatchSource, layer="kv")
        ),
        kv_rules=tuple(rules.key_parts(ctx, "kv")),
        deepep=facts.ep_supported,
    )
    parts.extend((
        f"kv-installer={file_sha256(facts.kv_installer)}",
        f"kv-req={file_sha256(facts.kv_requirements)}",
        *resolution.kv_rules,
    ))
    key = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
    return dataclasses.replace(resolution, key=key)


def _kv_key(resolution: ExtrasResolution, venv_hash: str) -> str:
    assert resolution.kv_installer and resolution.kv_requirements
    parts = (venv_hash, file_sha256(resolution.kv_installer),
             file_sha256(resolution.kv_requirements), *resolution.kv_rules)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def _sync_kv_connectors(
    root: Path, venv: Path, resolution: ExtrasResolution, venv_hash: str
) -> None:
    if resolution.kv_installer is None or resolution.kv_requirements is None:
        say("vLLM extras: KV connector installer unsupported by this commit — skipping")
        return
    key = _kv_key(resolution, venv_hash)
    if (read_marker(root).get("kv_connectors_hash") == key
            and dists_installed(venv, ["nixl"])):
        say("vLLM extras: KV connectors up to date")
        return

    # The upstream CI helper targets a container's system Python; the rules'
    # patches (kv-installer-venv) retarget it to this managed venv.
    script = rules.apply_patches(
        resolution.kv_patches, KV_INSTALLER, resolution.kv_installer.read_text()
    )
    with tempfile.TemporaryDirectory(prefix="ve-vllm-extras-") as tmp:
        installer = Path(tmp) / "install-kv-connectors.sh"
        installer.write_text(script)
        env = {
            "VIRTUAL_ENV": str(venv),
            "PATH": f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            "KV_CONNECTORS_REQUIREMENTS": str(resolution.kv_requirements),
        }
        # The installer uninstalls packages before downloading their
        # replacements, so a network flake mid-run leaves them missing; it is
        # idempotent, so rerun it.
        run_retrying_transient(
            ["bash", str(installer)],
            "KV connector install",
            env=env,
            stream_prefix="[ve]   [vLLM extras] ",
        )
    update_marker(root, kv_connectors_hash=key)


def sync_vllm_extras(
    cfg: Config, root: Path, venv: Path, base_hash: str, full_hash: str
) -> None:
    """Install vLLM's EP kernels and KV connectors as one supported bundle."""
    resolution = resolve(root, base_hash, full_hash)
    marker = read_marker(root)
    if marker.get("vllm_extras_hash") == resolution.key:
        wanted = []
        if resolution.kv_installer is not None:
            wanted.append("nixl")
        if resolution.deepep:
            wanted.append("deep_ep")
        if dists_installed(venv, wanted):
            say(f"vLLM extras: up to date ({resolution.key})")
            return

    sync_deepep(cfg, root, venv, base_hash)
    _sync_kv_connectors(root, venv, resolution, full_hash)
    update_marker(root, vllm_extras_hash=resolution.key)
