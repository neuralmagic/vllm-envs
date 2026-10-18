"""Content-addressed DeepEP installation using vLLM's supported installer."""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import rules
from .config import Config
from .extprojects import fetch_ref
from .hashing import cuda_version, file_sha256
from .locks import entry_lock
from .log import say
from .registry import read_marker, update_marker
from .sitepkgs import dist_infos, dists_installed, torch_cuda_major
from .store import touch_last_used, write_meta
from .treefacts import TreeFacts, facts_for
from .util import run


@dataclass(frozen=True)
class DeepEPResolution:
    key: str
    ref: str
    cuda_arch_list: str
    installer: Path
    nccl_version: str = ""
    pplx_ref: str = ""
    # Fired rules' effects (rules.OverrideReq / rules.PatchSource, layer "ep").
    overrides: tuple[str, ...] = ()
    patches: tuple = ()


def _cuda_arch_list(facts: TreeFacts) -> str:
    if override := os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return override
    archs = facts.docker_cuda_arch_list.split()
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


def resolve(
    root: Path, venv_hash: str, facts: TreeFacts | None = None
) -> DeepEPResolution:
    facts = facts or facts_for(root, "cuda")
    if not facts.ep_supported:
        raise RuntimeError(facts.ep_unsupported_reason())
    installer = facts.ep_installer
    archs = _cuda_arch_list(facts)
    ctx = facts.context()
    fields = {
        "installer": file_sha256(installer),
        "ref": facts.deepep_ref,
        "nccl": facts.nccl_version or "",
        "venv": venv_hash,
        "cuda": cuda_version(),
        "archs": archs,
    }
    if parts := rules.key_parts(ctx, "ep"):
        fields["rules"] = parts
    key = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]
    return DeepEPResolution(
        key, facts.deepep_ref, archs, installer,
        facts.nccl_version or "", facts.pplx_ref or "",
        overrides=tuple(
            e.spec for _, e in rules.effects(ctx, rules.OverrideReq, layer="ep")
        ),
        patches=tuple(
            e for _, e in rules.effects(ctx, rules.PatchSource, layer="ep")
        ),
    )


def _wheels(entry: Path) -> list[Path]:
    return sorted((entry / "workspace" / "dist").glob("*.whl"))


def _prepare_sources(resolution: DeepEPResolution, workspace: Path) -> None:
    """Pre-fetch exact refs that historical installer scripts clone loosely,
    then apply the rules' source patches to them."""
    if resolution.pplx_ref:
        fetch_ref(
            "https://github.com/ppl-ai/pplx-kernels",
            resolution.pplx_ref,
            workspace / "pplx-kernels",
            "pplx-kernels",
        )
    fetch_ref(
        "https://github.com/deepseek-ai/DeepEP",
        resolution.ref,
        workspace / "DeepEP",
        "DeepEP",
    )
    for patch in resolution.patches:
        target = workspace / patch.file
        if target.is_file():
            text = target.read_text()
            if (fixed := patch.apply(text)) != text:
                target.write_text(fixed)


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


def _installed_nccl(venv: Path) -> tuple[str, str] | None:
    """(CUDA major, version) of the venv's nvidia-nccl-cuNN wheel."""
    for dist in dist_infos(venv, "nvidia_nccl_cu*"):
        if m := re.fullmatch(r"nvidia_nccl_cu(\d+)-([^-]+)\.dist-info", dist.name):
            return m.group(1), m.group(2)
    return None


def _nccl_override(venv: Path, resolution: DeepEPResolution, dest: Path) -> str | None:
    """Write the uv override that keeps NCCL at the version vLLM's image builds
    DeepEP against, and return its path.

    DeepEP's Gin backend needs NCCL >= 2.30.4 (ncclGinRequest_t and friends) and
    resolves NCCL from the Python environment, but torch requires an exact older
    nvidia-nccl wheel — so any resolve inside the EP installer (it runs
    `uv pip install cmake torch ninja`) drags NCCL back down and the build dies
    on missing GIN types.  docker/Dockerfile handles this with UV_OVERRIDE of
    nvidia-nccl-cu${CUDA_MAJOR} (rule deepep-nccl-override); mirror it for the
    NCCL wheel the venv's torch actually uses, and only when there is one."""
    installed = _installed_nccl(venv)
    if not resolution.overrides or installed is None:
        return None
    cuda_major, version = installed
    lines = [
        spec.format(cuda_major=cuda_major, nccl_version=resolution.nccl_version)
        for spec in resolution.overrides
    ]
    if version != resolution.nccl_version:
        say(
            f"DeepEP layer: overriding {', '.join(lines)} for the NCCL Gin "
            f"backend (venv has {version})"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(f"{line}\n" for line in lines))
    return str(dest)


def _toolkit_cuda_major() -> tuple[Path, str | None]:
    """The CUDA toolkit the EP installer compiles with (it uses
    ${CUDA_HOME:-/usr/local/cuda}/bin/nvcc) and its major version."""
    home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    nvcc = home / "bin" / "nvcc"
    if not nvcc.is_file():
        return home, None
    m = re.search(r"release (\d+)\.", run([str(nvcc), "--version"], check=False).stdout)
    return home, m.group(1) if m else None


def sync_deepep(cfg: Config, root: Path, venv: Path, venv_hash: str) -> None:
    """Install the vLLM-pinned DeepEP wheel into a CUDA environment."""
    facts = facts_for(root, "cuda")
    if not facts.ep_supported:
        say(f"DeepEP layer: unsupported by this vLLM commit "
            f"({facts.ep_unsupported_reason()}) — skipping")
        return
    cuda_home, toolkit = _toolkit_cuda_major()
    torch_major = torch_cuda_major(venv)
    if toolkit and torch_major and toolkit != torch_major:
        say(f"DeepEP layer: CUDA toolkit {cuda_home} is CUDA {toolkit} but torch "
            f"is built for CUDA {torch_major} — skipping (set CUDA_HOME to a "
            f"CUDA {torch_major} toolkit to build DeepEP)")
        return
    resolution = resolve(root, venv_hash, facts)
    marker = read_marker(root)
    entry = cfg.store("ep-kernels") / resolution.key
    cached_wheels = _wheels(entry)
    dists = ["deep_ep"]
    if any(w.name.lower().startswith("pplx_kernels-") for w in cached_wheels):
        dists.append("pplx_kernels")
    installed = dists_installed(venv, dists)
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
