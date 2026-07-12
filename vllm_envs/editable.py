"""Editable-install capture/replay: skip setup.py/uv on exact repeat attaches.

A PEP 660 editable install of vLLM leaves three site-packages artifacts (a
.pth, a finder module, a dist-info dir) plus files extracted from the wheel
into the worktree. All are replayable: after the first shared attach at a
build hash we capture them into builds/<hash>/editable/ (site artifacts +
the extracted-member list; the file bytes live in the extracted/ mirror).
A later attach at the *same HEAD commit* reflinks the extracted files and
copies the site artifacts with path rewriting — no setup.py runs.

Safety: replay is opportunistic. The HEAD sha must match capture (the
version string embeds the commit), every mirrored file must exist, and any
failure falls back to the full uv attach.
"""

import base64
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from .log import say
from .util import run

META_NAME = "editable.json"
SCHEMA_VERSION = 2


def site_packages(venv: Path) -> Path | None:
    for p in sorted((venv / "lib").glob("python3.*")):
        sp = p / "site-packages"
        if sp.is_dir():
            return sp
    return None


def _artifacts(sp: Path) -> tuple[Path, Path, Path] | None:
    pth = sorted(sp.glob("__editable__.vllm-*.pth"))
    finder = sorted(sp.glob("__editable___vllm_*_finder.py"))
    dist = sorted(sp.glob("vllm-*.dist-info"))
    if len(pth) == 1 and len(finder) == 1 and len(dist) == 1:
        return pth[0], finder[0], dist[0]
    return None


def editable_present(venv: Path) -> bool:
    sp = site_packages(venv)
    arts = _artifacts(sp) if sp else None
    if arts is None:
        return False
    return all((venv / "bin" / name).is_file() for name in _console_scripts(arts[2]))


def _console_scripts(dist: Path) -> list[str]:
    entry_points = dist / "entry_points.txt"
    if not entry_points.is_file():
        return []
    parser = configparser.ConfigParser()
    parser.read(entry_points)
    return (
        sorted(parser["console_scripts"])
        if parser.has_section("console_scripts")
        else []
    )


def _head(env_root: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=env_root).stdout.strip()


def _store_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    shutil.copy2(src, tmp)
    tmp.chmod(tmp.stat().st_mode & ~0o222)
    os.replace(tmp, dst)


def capture(env_root: Path, venv: Path, wheel: Path, store_entry: Path) -> None:
    """Record the editable install at builds/<hash>/editable/ (idempotent)."""
    ed = store_entry / "editable"
    try:
        existing = json.loads((ed / META_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("schema") == SCHEMA_VERSION:
        return
    if ed.exists():
        shutil.rmtree(ed)
    sp = site_packages(venv)
    arts = _artifacts(sp) if sp else None
    if arts is None:
        return
    tracked = set(run(["git", "ls-files"], cwd=env_root).stdout.splitlines())
    with zipfile.ZipFile(wheel) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
    extracted = [
        m
        for m in members
        if m.startswith("vllm/") and m not in tracked and (env_root / m).is_file()
    ]
    mirror = store_entry / "extracted"
    for m in extracted:  # top up the .so mirror with non-.so extracted files
        if not (mirror / m).is_file():
            _store_file(env_root / m, mirror / m)
    site_dir = ed / "site"
    if site_dir.exists():
        shutil.rmtree(site_dir)
    site_dir.mkdir(parents=True)
    for a in arts:
        if a.is_dir():
            shutil.copytree(a, site_dir / a.name)
        else:
            shutil.copy2(a, site_dir / a.name)
    scripts = _console_scripts(arts[2])
    scripts_dir = ed / "scripts"
    scripts_dir.mkdir()
    for name in scripts:
        script = venv / "bin" / name
        if not script.is_file():
            shutil.rmtree(ed)
            return
        shutil.copy2(script, scripts_dir / name)
    (ed / META_NAME).write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "head": _head(env_root),
                "worktree": str(env_root),
                "extracted": extracted,
                "scripts": scripts,
            },
            indent=2,
        )
    )
    say(f"attach: captured editable install → builds/{store_entry.name}/editable")


def worktree_complete(env_root: Path, store_entry: Path) -> bool:
    """All captured extracted files still present (self-heal check for no-op skip)."""
    try:
        meta = json.loads((store_entry / "editable" / META_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return True  # nothing captured; nothing to verify
    return all((env_root / m).is_file() for m in meta.get("extracted", []))


def _record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _fix_record(sp: Path, dist: Path) -> None:
    record = dist / "RECORD"
    lines = []
    for line in record.read_text().splitlines():
        rel, _, rest = line.rpartition(",")
        rel, _, _ = rel.rpartition(",")
        f = sp / rel
        if rel and not rel.endswith("RECORD") and f.is_file():
            lines.append(f"{rel},{_record_hash(f)},{f.stat().st_size}")
        else:
            lines.append(line)
    record.write_text("\n".join(lines) + "\n")


def replay(env_root: Path, venv: Path, store_entry: Path) -> bool:
    """Reproduce a captured editable install; False → caller runs full attach."""
    ed = store_entry / "editable"
    try:
        meta = json.loads((ed / META_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("schema") != SCHEMA_VERSION or meta.get("head") != _head(env_root):
        return False  # version string embeds the commit; do not fake it
    sp = site_packages(venv)
    if sp is None:
        return False
    mirror = store_entry / "extracted"
    extracted = meta.get("extracted", [])
    if not all((mirror / m).is_file() for m in extracted):
        return False

    for m in extracted:
        src, dst = mirror / m, env_root / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".ve-tmp")
        proc = subprocess.run(
            ["cp", "--reflink=auto", str(src), str(tmp)], capture_output=True
        )
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.chmod((src.stat().st_mode & 0o777) | 0o600)
        os.replace(tmp, dst)

    old_s, new_s = meta["worktree"], str(env_root)
    copied: list[Path] = []
    for item in (ed / "site").iterdir():
        dst = sp / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
            copied += [p for p in dst.rglob("*") if p.is_file()]
        else:
            shutil.copy2(item, dst)
            copied.append(dst)
    for p in copied:
        p.chmod(0o644)
        if old_s != new_s:
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if old_s in text:
                p.write_text(text.replace(old_s, new_s))
    for name in meta.get("scripts", []):
        src = ed / "scripts" / name
        if not src.is_file():
            return False
        dst = venv / "bin" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        text = dst.read_text()
        if old_s != new_s and old_s in text:
            dst.write_text(text.replace(old_s, new_s))
        dst.chmod(0o755)
    dists = sorted(sp.glob("vllm-*.dist-info"))
    if len(dists) == 1 and (dists[0] / "RECORD").is_file():
        _fix_record(sp, dists[0])
    say(
        f"attach: replayed editable install from builds/{store_entry.name} "
        f"({len(extracted)} extracted files, no setup.py)"
    )
    return True
