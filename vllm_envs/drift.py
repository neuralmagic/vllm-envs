"""Detect hand-made changes to a ve-managed venv ("drift").

Agents that hit a broken env often recover by `pip install`-ing, upgrading or
uninstalling packages, or by patching files in site-packages. None of that is
an error, so it never reaches the incident log, yet it is exactly the signal
that ve set the env up wrong. Drift is measured against what ve itself left:

* tracked envs: ``.ve/synced-freeze.txt`` is the package set at the end of the
  last sync (ve's own installs are folded in, so agent changes stay visible
  across syncs); files modified after that baseline was written count as
  patches.
* envs synced before drift tracking: packages/files modified after the last
  sync wrote the env marker, plus template packages that went missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import MARKER_NAME, scratch_dir
from .reqs import freeze_map
from .sitepkgs import dist_info_map, site_packages, venv_freeze
from .util import atomic_write_text, run

BASELINE_NAME = "synced-freeze.txt"
RECORDED_NAME = "drift-recorded.txt"
TEMPLATE_FREEZE_NAME = "template-freeze.txt"
MTIME_SLACK_S = 2.0
# file timestamps come from a coarse kernel clock that can lag time.time()
CLOCK_SKEW_S = 1.0
MAX_FILES_LISTED = 200

def _version(spec: str) -> str:
    if "==" in spec:
        return spec.split("==", 1)[1]
    if " @ " in spec:
        return "@ " + spec.split(" @ ", 1)[1]
    return spec


@dataclass
class Drift:
    mode: str  # tracked | legacy
    added: dict[str, str] = field(default_factory=dict)
    removed: dict[str, str] = field(default_factory=dict)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    n_files: int = 0
    since: float | None = None

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.n_files)

    def packages_summary(self) -> str:
        parts = [f"+{n}=={_version(s)}" for n, s in sorted(self.added.items())]
        parts += [f"-{n}" for n in sorted(self.removed)]
        parts += [f"{n} {_version(b)} (reinstalled)" if a == b
                  else f"{n} {_version(a)}→{_version(b)}"
                  for n, (a, b) in sorted(self.changed.items())]
        return ", ".join(parts)

    def summary(self) -> str:
        bits = []
        if self.added or self.removed or self.changed:
            bits.append(self.packages_summary())
        if self.n_files:
            tops = sorted({f.split("/", 1)[0] for f in self.files})
            bits.append(f"{self.n_files} patched file(s) in "
                        f"{', '.join(tops[:6])}{' …' if len(tops) > 6 else ''}")
        return "; ".join(bits) or "none"

    def signature(self) -> str:
        return f"{self.packages_summary()}|{self.n_files}"


def baseline_path(env_root: Path) -> Path:
    return scratch_dir(env_root) / BASELINE_NAME


def _owned_roots(dist_info: Path) -> set[str]:
    """Top-level site-packages entries installed by a distribution."""
    roots = {dist_info.name}
    try:
        for line in (dist_info / "RECORD").read_text(errors="replace").splitlines():
            path = line.split(",", 1)[0]
            if path and not path.startswith(".."):
                roots.add(path.split("/", 1)[0])
    except OSError:
        pass
    return roots


def _modified_files(
    sp: Path, after: float, before: float | None, skip_roots: set[str]
) -> list[str]:
    """site-packages files modified in (after, before), via find(1): a venv
    holds ~150k files, too many to stat quickly from Python."""
    cmd = ["find", str(sp), "-mindepth", "1"]
    prune = [r for r in sorted(skip_roots) if "/" not in r and r not in (".", "..")]
    if prune:
        cmd += ["("]
        for i, root in enumerate(prune):
            cmd += (["-o"] if i else []) + ["-path", str(sp / root)]
        cmd += [")", "-prune", "-o"]
    cmd += ["-name", "__pycache__", "-prune", "-o", "-type", "f",
            "!", "-name", "*.pyc", "-newermt", f"@{after + MTIME_SLACK_S:.3f}"]
    if before is not None:
        cmd += ["!", "-newermt", f"@{before - CLOCK_SKEW_S:.3f}"]
    cmd += ["-printf", "%P\\n"]
    p = run(cmd, check=False)
    return sorted(line for line in p.stdout.splitlines() if line)


def _rekey_editables(entries: dict[str, str], dists: dict[str, Path]) -> dict[str, str]:
    """`uv pip freeze` lists editables by URL only; name them via the
    direct_url.json of the matching dist-info."""
    if not any(spec.startswith("-e ") for spec in entries.values()):
        return entries
    by_url: dict[str, str] = {}
    for name, d in dists.items():
        try:
            url = json.loads((d / "direct_url.json").read_text()).get("url")
        except (OSError, ValueError):
            continue
        if url:
            by_url[url.rstrip("/")] = name
    out = {}
    for key, spec in entries.items():
        if spec.startswith("-e "):
            key = by_url.get(spec[3:].strip().rstrip("/"), key)
        out[key] = spec
    return out


def _touched(
    cur: dict[str, str], dists: dict[str, Path], since: float,
    before: float | None,
) -> list[str]:
    """Installed packages whose metadata was (re)written in (since, before)."""
    out = []
    for name in cur:
        d = dists.get(name)
        try:
            mtime = d.stat().st_mtime if d is not None else None
            if mtime is not None and mtime > since + MTIME_SLACK_S and (
                before is None or mtime < before - CLOCK_SKEW_S
            ):
                out.append(name)
        except OSError:
            pass
    return out


def dists_written_since(venv: Path, since: float) -> bool:
    sp = site_packages(venv)
    if sp is None:
        return False
    since -= CLOCK_SKEW_S
    for d in sp.glob("*.dist-info"):
        try:
            if d.stat().st_mtime >= since:
                return True
        except OSError:
            pass
    return False


def _diff(base: dict[str, str], cur: dict[str, str], drift: Drift) -> None:
    for name, spec in cur.items():
        if name not in base:
            drift.added[name] = spec
        elif base[name] != spec:
            drift.changed[name] = (base[name], spec)
    for name, spec in base.items():
        if name not in cur:
            drift.removed[name] = spec


def compute(
    env_root: Path,
    current: list[str] | None = None,
    scan_files: bool = True,
    before: float | None = None,
) -> Drift | None:
    """Drift of env_root/.venv from what ve installed (None: no usable venv)."""
    venv = env_root / ".venv"
    if current is None:
        current = venv_freeze(venv)
    if current is None:
        return None
    cur = freeze_map(current)
    sp = site_packages(venv)
    dists = dist_info_map(sp) if sp else {}
    cur = _rekey_editables(cur, dists)
    baseline = baseline_path(env_root)

    if baseline.exists():
        drift = Drift("tracked", since=baseline.stat().st_mtime)
        base = _rekey_editables(
            freeze_map(baseline.read_text().splitlines()), dists
        )
        _diff(base, cur, drift)
        for name in _touched(cur, dists, drift.since, before):
            if name not in drift.added and name not in drift.changed:
                drift.changed[name] = (cur[name], cur[name])
    else:
        marker = env_root / MARKER_NAME
        if not marker.exists():
            return None
        drift = Drift("legacy", since=marker.stat().st_mtime)
        template = scratch_dir(env_root) / TEMPLATE_FREEZE_NAME
        tmpl = (freeze_map(template.read_text().splitlines())
                if template.exists() else {})
        for name in _touched(cur, dists, drift.since, before):
            if name in tmpl or cur[name].startswith("-e "):
                drift.changed[name] = (tmpl.get(name, cur[name]), cur[name])
            else:
                drift.added[name] = cur[name]
        for name, spec in tmpl.items():
            if name not in cur:
                drift.removed[name] = spec

    if scan_files and sp is not None and drift.since is not None:
        skip: set[str] = set()
        for name in (*drift.added, *drift.changed):
            if name in dists:
                skip |= _owned_roots(dists[name])
        files = _modified_files(sp, drift.since, before, skip)
        drift.n_files = len(files)
        drift.files = files[:MAX_FILES_LISTED]
    return drift


def fold_into_baseline(
    env_root: Path, pre: list[str] | None, post: list[str] | None, recloned: bool
) -> None:
    """After a sync that changed the venv, make ve's own package changes part
    of the baseline while leaving earlier hand-made changes visible as drift.
    Rewriting also moves the baseline time past ve's own file writes."""
    if post is None:
        return
    path = baseline_path(env_root)
    if recloned or pre is None or not path.exists():
        new = freeze_map(post)
    else:
        new = freeze_map(path.read_text().splitlines())
        before, after = freeze_map(pre), freeze_map(post)
        for name in before.keys() | after.keys():
            if before.get(name) == after.get(name):
                continue
            if name in after:
                new[name] = after[name]
            else:
                new.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(sorted(new.values())) + "\n")


def record_if_new(env_root: Path, drift: Drift | None, context: str) -> None:
    """Log drift to the incident log once per distinct drift state."""
    from .errorlog import record_drift

    if drift is None or drift.empty:
        return
    marker = scratch_dir(env_root) / RECORDED_NAME
    sig = drift.signature()
    try:
        if marker.exists() and marker.read_text() == sig:
            return
    except OSError:
        pass
    if record_drift(env_root, drift, context) is not None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(sig)
        except OSError:
            pass
