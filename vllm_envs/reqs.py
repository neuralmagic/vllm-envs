"""Requirement-file and `pip freeze` line parsing, with PEP 503 names."""

import re
from collections.abc import Iterable
from pathlib import Path

NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_FLOOR_RE = re.compile(r">=?\s*v?(\d+(?:\.\d+)*)")
_CEILING_RE = re.compile(r"<|==|~=")
_DECLARED_MIN_RE = re.compile(r"(?:===|==|~=|>=)\s*v?(\d+(?:\.\d+)*)")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def at_least(actual: tuple[int, ...], required: tuple[int, ...]) -> bool:
    length = max(len(actual), len(required))
    return actual + (0,) * (length - len(actual)) >= required + (0,) * (
        length - len(required)
    )


def requirement_lines(raw_lines: Iterable[str]) -> list[str]:
    """Requirement specs, without comments, options (-r/-c/-e ...) or URLs."""
    lines = []
    for raw in raw_lines:
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-") and "://" not in line:
            lines.append(line)
    return lines


def file_requirement_lines(files) -> list[str]:
    return requirement_lines(
        raw for f in files
        for raw in Path(f).read_text(errors="replace").splitlines()
    )


def requirement_names(files) -> set[str]:
    return {
        canonical_name(m.group(1))
        for line in file_requirement_lines(files)
        if (m := NAME_RE.match(line))
    }


def declared_floors(files) -> dict[str, tuple[int, ...]]:
    """Highest minimum version (==, >=, ~=, ===) declared per package."""
    floors: dict[str, tuple[int, ...]] = {}
    for line in file_requirement_lines(files):
        m = NAME_RE.match(line)
        version = _DECLARED_MIN_RE.search(line, m.end()) if m else None
        if version is None:
            continue
        name, floor = canonical_name(m.group(1)), version_tuple(version.group(1))
        if name not in floors or at_least(floor, floors[name]):
            floors[name] = floor
    return floors


def cap_constraints(
    files: list[Path], mode: str, extra_requirements: list[str] | None = None
) -> str:
    """Constraint lines capping requirements that have a version floor but no
    ceiling: `pkg>=X.Y.Z` → `pkg<X.(Y+1)` (minor) or `pkg<(X+1)` (major), so
    unpinned deps resolve near the floor's era instead of to latest. The
    highest floor wins when a package appears in several files."""
    if mode == "none":
        return ""
    floors: dict[str, tuple[tuple[int, ...], str]] = {}
    # A lock file commonly pins a package while an input requirement keeps its
    # lower bound.  Do not generate a synthetic cap from the lower bound in
    # that case: ``pyzmq>=25`` must not become ``pyzmq<26`` when the selected
    # test lock explicitly requires ``pyzmq==27.1.0``.
    explicitly_bounded: set[str] = set()
    lines = file_requirement_lines(files)
    lines += requirement_lines(extra_requirements or [])
    for line in lines:
        spec = line.split(";", 1)[0].strip()
        m = NAME_RE.match(spec)
        if not m:
            continue
        name, rest = m.group(1), spec[len(m.group(1)):]
        key = canonical_name(name)
        if _CEILING_RE.search(rest):
            explicitly_bounded.add(key)
            continue
        fm = _FLOOR_RE.search(rest)
        if not fm:
            continue
        floor = version_tuple(fm.group(1))
        if key not in floors or floor > floors[key][0]:
            floors[key] = (floor, name)
    out = []
    for key, (floor, name) in sorted(floors.items()):
        if key in explicitly_bounded:
            continue
        parts = [*floor, 0, 0]
        bound = f"{parts[0] + 1}" if mode == "major" else f"{parts[0]}.{parts[1] + 1}"
        out.append(f"{name}<{bound}\n")
    return "".join(out)


def freeze_map(lines: list[str]) -> dict[str, str]:
    """name -> pin/spec line, for `uv pip freeze` output."""
    out: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            m = re.search(r"#egg=([A-Za-z0-9._-]+)", line)
            name = m.group(1) if m else Path(line[3:].strip()).name
        else:
            m = NAME_RE.match(line)
            if not m:
                continue
            name = m.group(1)
        out[canonical_name(name)] = line
    return out


def freeze_names(lines: list[str]) -> set[str]:
    """Names of the index-installed packages in freeze output (editables and
    direct URL installs excluded)."""
    return {
        canonical_name(m.group(1))
        for line in lines
        if not line.startswith(("-", "#")) and "://" not in line
        and (m := NAME_RE.match(line.strip()))
    }
