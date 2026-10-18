"""Attach: editable install of the worktree, with the wheel as .so donor."""

import os
import re
import subprocess
import zipfile
from pathlib import Path

from . import editable
from .buildlayer import BuildResolution, local_fallback_current
from .config import Config
from .log import say, warn
from .registry import read_marker, update_marker
from .util import atomic_copy, human_size
from .venvlayer import uv_pip_install


def _dedupe_extracted_sos(env_root: Path, wheel: Path, store_entry: Path) -> None:
    """Replace wheel-extracted .so copies in the worktree with reflinks to a
    shared mirror under builds/<hash>/extracted/.

    XFS CoW: N envs at the same hash share one physical copy until a file is
    rewritten (e.g. a private rebuild), and envs stay self-contained if the
    store entry is later evicted. Best-effort: plain copies are kept whenever
    reflinks are unavailable.
    """
    mirror = store_entry / "extracted"
    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist() if n.endswith(".so") or ".so." in n]
    shared_bytes = 0
    count = 0
    for name in names:
        wt = env_root / name
        if not wt.is_file() or wt.is_symlink():
            continue
        m = mirror / name
        if not m.is_file():
            atomic_copy(wt, m, 0o444)
        tmp = wt.with_name(wt.name + ".ve-reflink")
        proc = subprocess.run(
            ["cp", "--reflink=always", str(m), str(tmp)], capture_output=True
        )
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            return  # non-reflink filesystem: keep plain copies
        tmp.chmod(0o644)
        size = wt.stat().st_size
        os.replace(tmp, wt)
        shared_bytes += size
        count += 1
    if count:
        say(f"deduped {count} extracted .so files → reflinks of "
            f"builds/{store_entry.name}/extracted ({human_size(shared_bytes)} shared)")


def _fallback_reason(exc: subprocess.CalledProcessError) -> str:
    text = " ".join((exc.output or "").split())
    m = re.search(r"\w+Error: (.{0,300}?)(?: Wait for| hint:|$)", text)
    return m.group(1).strip() if m else f"exit {exc.returncode}"


def attach(cfg: Config, env_root: Path, venv: Path, res: BuildResolution) -> None:
    store_entry = cfg.store("builds") / res.build_hash
    marker = read_marker(env_root)
    if (
        res.mode == "precompiled-fetch"
        and res.wheel is None
        and local_fallback_current(env_root, venv, marker, res.build_hash)
    ):
        say(f"attach: env-local precompiled fallback up to date ({res.build_hash})")
        return
    if res.shared:
        if (marker.get("build_hash") == res.build_hash
                and marker.get("attach_mode") == res.mode
                and editable.editable_present(venv)
                and editable.worktree_complete(env_root, store_entry)):
            say(f"attach: up to date (builds/{res.build_hash})")
            if res.wheel is not None:
                try:  # seed the replay cache from an already-attached env
                    editable.capture(env_root, venv, res.wheel, store_entry)
                except Exception as e:
                    warn(f"attach capture skipped ({e})")
            return
        if res.wheel is not None:
            try:
                if editable.replay(env_root, venv, store_entry):
                    update_marker(
                        env_root, build_hash=res.build_hash, attach_mode=res.mode
                    )
                    return
            except Exception as e:
                warn(f"attach replay failed ({e}); running full attach")

    env = {"VLLM_USE_PRECOMPILED": "1"}
    if res.wheel is not None:
        env["VLLM_PRECOMPILED_WHEEL_LOCATION"] = str(res.wheel)
        say(f"attaching extensions from {res.wheel.name} + editable install")
    else:
        say("attaching via upstream precompiled wheel fetch + editable install")
    try:
        uv_pip_install(venv, ["-e", str(env_root), "--no-build-isolation", "--no-deps"],
                env=env)
    except subprocess.CalledProcessError as fallback:
        if res.build_error is None:
            raise
        raise RuntimeError(
            f"local build failed and the upstream precompiled fallback was "
            f"refused ({_fallback_reason(fallback)}); fix the build error above "
            f"(`ve errors show` has the excerpt)"
        ) from res.build_error
    if res.shared and res.wheel is not None:
        try:
            _dedupe_extracted_sos(env_root, res.wheel, store_entry)
        except Exception as e:
            warn(f".so dedupe skipped ({e}); worktree keeps plain copies")
        try:
            editable.capture(env_root, venv, res.wheel, store_entry)
        except Exception as e:
            warn(f"attach capture skipped ({e})")
    update_marker(
        env_root,
        build_hash=(
            res.build_hash
            if res.shared or res.mode == "precompiled-fetch"
            else ""
        ),
        attach_mode=res.mode,
    )
