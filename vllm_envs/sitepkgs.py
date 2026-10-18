"""What a venv has installed, read from site-packages without importing it."""

import re
from pathlib import Path

from .reqs import canonical_name
from .util import run


def site_packages_dirs(venv: Path) -> list[Path]:
    return sorted(venv.glob("lib*/python3*/site-packages"))


def site_packages(venv: Path) -> Path | None:
    dirs = site_packages_dirs(venv)
    return dirs[0] if dirs else None


def dist_infos(venv: Path, dist_glob: str) -> list[Path]:
    """dist-info dirs of distributions whose (on-disk) name matches
    `dist_glob`, across every site-packages dir, sorted."""
    return sorted(
        p for sp in site_packages_dirs(venv)
        for p in sp.glob(f"{dist_glob}-*.dist-info") if p.is_dir()
    )


def dists_installed(venv: Path, dists: list[str]) -> bool:
    """Check dist-info presence rather than importing: e.g. `import deep_ep`
    loads torch, which cost ~3s of wall time on every no-op sync."""
    return all(dist_infos(venv, dist) for dist in dists)


def dist_info_map(sp: Path) -> dict[str, Path]:
    """Canonical distribution name -> its dist-info dir in `sp`."""
    out = {}
    for d in sp.glob("*.dist-info"):
        name = d.name[: -len(".dist-info")].rsplit("-", 1)[0]
        out[canonical_name(name)] = d
    return out


def venv_freeze(venv: Path, strict: bool = False) -> list[str] | None:
    """Sorted `uv pip freeze` lines. Non-strict: None when the venv has no
    interpreter or the freeze fails; strict: the failure raises."""
    python = venv / "bin" / "python"
    if not strict and not python.exists():
        return None
    p = run(["uv", "pip", "freeze", "--python", str(python)], check=strict)
    if p.returncode != 0:
        return None
    return sorted(line.strip() for line in p.stdout.splitlines() if line.strip())


def torch_cuda_major(venv: Path) -> str | None:
    """CUDA major of the venv's torch from its dist-info name (torch-X+cu130),
    without importing torch. That, not the driver, fixes the wheel's ABI."""
    for dist in dist_infos(venv, "torch"):
        if m := re.search(r"\+cu(\d+?)(\d)\.dist-info$", dist.name):
            return m.group(1)
    return None
