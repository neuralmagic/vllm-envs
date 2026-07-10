"""Layer resolution: venv templates, compiled extensions, attach."""

import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import editable
from .config import SCRATCH_DIR_NAME, Config
from .extprojects import src_dir_env, user_overrides
from .hashing import VenvKeys, build_key, build_paths_dirty, detect_platform, venv_keys
from .locks import entry_lock
from .log import say, warn
from .precompiled import try_fetch_precompiled
from .registry import read_marker, update_marker
from .store import touch_last_used, write_meta
from .util import human_size, reflink_clone, run


def _rewrite_venv_paths(old: Path, new: Path) -> None:
    """Fix activate scripts / console-script shebangs after cloning a venv."""
    bin_dir = new / "bin"
    old_s, new_s = str(old), str(new)
    if old_s == new_s or not bin_dir.is_dir():
        return
    for f in bin_dir.iterdir():
        if f.is_symlink() or not f.is_file():
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if old_s in text:
            mode = f.stat().st_mode
            f.write_text(text.replace(old_s, new_s))
            f.chmod(mode)


def _uv_pip(venv: Path, args: list[str], env: dict | None = None) -> None:
    run(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *args],
        env=env,
        stream_prefix="[ve]   [uv] ",
    )


def _torch_backend_args(platform: str) -> list[str]:
    return ["--torch-backend=auto"] if platform == "cuda" else []


def _venv_freeze(venv: Path) -> list[str]:
    out = run(["uv", "pip", "freeze", "--python", str(venv / "bin" / "python")]).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def _freeze_names(lines: list[str]) -> set[str]:
    names = set()
    for line in lines:
        if line.startswith(("-", "#")) or "://" in line:
            continue
        name = re.split(r"[=<>!\[@; ]", line, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def template_freeze(entry: Path) -> list[str]:
    f = entry / "freeze.txt"
    if not f.exists():  # templates predating freeze tracking
        f.write_text("\n".join(_venv_freeze(entry)) + "\n")
    return f.read_text().splitlines()


# --------------------------------------------------------------------------
# Layer 1: venv templates
# --------------------------------------------------------------------------


def ensure_base_template(cfg: Config, keys: VenvKeys, platform: str) -> Path:
    entry = cfg.store("venvs-base") / keys.base_hash
    with entry_lock(entry):
        if (entry / ".complete").exists():
            say(f"venv base layer: cache HIT ({keys.base_hash})")
            touch_last_used(entry)
            return entry
        say(f"venv base layer: cache MISS ({keys.base_hash}) — installing torch/build deps")
        if entry.exists():
            shutil.rmtree(entry)
        run(["uv", "venv", "--python", cfg.python, str(entry)])
        req_args: list[str] = []
        for f in keys.layout.build_files:
            req_args += ["-r", str(f)]
        if req_args:
            _uv_pip(entry, [*req_args, *_torch_backend_args(platform)])
        (entry / ".complete").touch()
        write_meta(entry, {"kind": "venv-base", "hash": keys.base_hash})
        touch_last_used(entry)
    return entry


def ensure_full_template(cfg: Config, keys: VenvKeys, platform: str) -> Path:
    entry = cfg.store("venvs") / keys.full_hash
    with entry_lock(entry):
        if (entry / ".complete").exists():
            say(f"venv full layer: cache HIT ({keys.full_hash})")
            touch_last_used(entry)
            return entry
        base = ensure_base_template(cfg, keys, platform)
        say(f"venv full layer: cache MISS ({keys.full_hash}) — deriving from base")
        if entry.exists():
            shutil.rmtree(entry)
        reflink_clone(base, entry)
        _rewrite_venv_paths(base, entry)
        (entry / ".complete").unlink(missing_ok=True)
        req_args: list[str] = []
        for f in keys.layout.runtime_files:
            req_args += ["-r", str(f)]
        if req_args:
            _uv_pip(entry, [*req_args, *_torch_backend_args(platform)])
        (entry / "freeze.txt").write_text("\n".join(_venv_freeze(entry)) + "\n")
        (entry / ".complete").touch()
        write_meta(entry, {"kind": "venv-full", "hash": keys.full_hash, "base": keys.base_hash})
        touch_last_used(entry)
    return entry


def resolve_venv(cfg: Config, env_root: Path, fresh: bool = False) -> tuple[Path, VenvKeys]:
    """Ensure env_root/.venv matches the worktree's requirements."""
    platform = cfg.platform or detect_platform()
    keys = venv_keys(env_root, platform, cfg.python)
    venv = env_root / ".venv"
    marker = read_marker(env_root)
    current = marker.get("venv_full_hash")

    if venv.exists() and current == keys.full_hash and not fresh:
        say(f"venv layer: up to date ({keys.full_hash})")
        return venv, keys

    if venv.exists() and not fresh:
        if current and marker.get("venv_base_hash") == keys.base_hash:
            # Base unchanged: converge the private venv in place (top-up),
            # keeping agent divergence (debug prints, extra packages).
            say(f"venv layer: requirements changed → top-up install into env venv "
                f"({current} → {keys.full_hash})")
            template = ensure_full_template(cfg, keys, platform)
            req_args: list[str] = []
            for f in keys.layout.runtime_files:
                req_args += ["-r", str(f)]
            if req_args:
                _uv_pip(venv, [*req_args, *_torch_backend_args(platform)])
            _uninstall_removed_deps(env_root, venv, template)
            update_marker(env_root, venv_full_hash=keys.full_hash, venv_base_hash=keys.base_hash)
            return venv, keys
        if current:
            warn("torch/build deps changed across hop → replacing env venv with "
                 "a fresh clone (local site-packages edits are discarded)")
        else:
            warn(f"replacing unmanaged venv at {venv} with a ve-managed clone")

    template = ensure_full_template(cfg, keys, platform)
    if venv.exists():
        shutil.rmtree(venv)
    t0 = time.time()
    reflink_clone(template, venv)
    _rewrite_venv_paths(template, venv)
    # heal templates derived before path rewriting existed
    _rewrite_venv_paths(cfg.store("venvs-base") / keys.base_hash, venv)
    (venv / ".complete").unlink(missing_ok=True)
    say(f"venv layer: reflink-cloned template {keys.full_hash} ({time.time() - t0:.1f}s)")
    _record_template_freeze(env_root, template_freeze(template))
    update_marker(env_root, venv_full_hash=keys.full_hash, venv_base_hash=keys.base_hash)
    return venv, keys


def _freeze_record(env_root: Path) -> Path:
    return _scratch(env_root) / "template-freeze.txt"


def _record_template_freeze(env_root: Path, lines: list[str]) -> None:
    _freeze_record(env_root).write_text("\n".join(lines) + "\n")


def _uninstall_removed_deps(env_root: Path, venv: Path, template: Path) -> None:
    """Uninstall deps that dropped out of the new template's resolution.

    removed = old template freeze − new template freeze, so user-added
    packages (in neither freeze) always survive.
    """
    record = _freeze_record(env_root)
    new_lines = template_freeze(template)
    if record.exists():
        removed = _freeze_names(record.read_text().splitlines()) - _freeze_names(new_lines)
        installed = _freeze_names(_venv_freeze(venv))
        removed &= installed
        if removed:
            say(f"venv layer: uninstalling {len(removed)} removed dep(s): "
                f"{', '.join(sorted(removed))}")
            run(
                ["uv", "pip", "uninstall", "--python", str(venv / "bin" / "python"),
                 *sorted(removed)],
                check=False,
                stream_prefix="[ve]   [uv] ",
            )
    else:
        warn("no template freeze record for this env (created pre-freeze-tracking); "
             "removed deps not uninstalled this hop")
    _record_template_freeze(env_root, new_lines)


# --------------------------------------------------------------------------
# Build layer: compiled extensions (wheel donor per build-hash)
# --------------------------------------------------------------------------


@dataclass
class BuildResolution:
    mode: str  # "store-wheel" | "precompiled-fetch" | "local-build"
    wheel: Path | None
    build_hash: str
    shared: bool  # attached to the shared store (refcounted)


def _find_wheel(dirpath: Path) -> Path | None:
    wheels = sorted(dirpath.glob("vllm-*.whl")) if dirpath.is_dir() else []
    return wheels[-1] if wheels else None


def _scratch(env_root: Path) -> Path:
    p = env_root / SCRATCH_DIR_NAME
    p.mkdir(exist_ok=True)
    return p


def _build_wheel(
    cfg: Config,
    env_root: Path,
    venv: Path,
    build_temp: Path,
    dist_dir: Path,
    use_pinned_ext: bool,
) -> Path:
    env = {
        "VLLM_DISABLE_SCCACHE": "1",  # local ccache policy
        "CMAKE_C_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
        # per-env fetchcontent dir: no cross-env build-dir races for any
        # project we couldn't pin
        "FETCHCONTENT_BASE_DIR": str(_scratch(env_root) / "fetchcontent"),
    }
    if use_pinned_ext:
        env.update(src_dir_env(cfg, env_root))
    build_temp.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    say(f"building extensions (ccache, build dir {build_temp})...")
    t0 = time.time()
    run(
        [
            str(venv / "bin" / "python"),
            "setup.py",
            "build_ext",
            "--build-temp",
            str(build_temp),
            "bdist_wheel",
            "--dist-dir",
            str(dist_dir),
        ],
        cwd=env_root,
        env=env,
        stream_prefix="[ve]   ",
    )
    say(f"build finished in {time.time() - t0:.0f}s")
    wheel = _find_wheel(dist_dir)
    if wheel is None:
        raise RuntimeError(f"build produced no wheel in {dist_dir}")
    return wheel


def resolve_build(cfg: Config, env_root: Path, venv: Path) -> BuildResolution:
    platform = cfg.platform or detect_platform()
    bhash = build_key(env_root, platform, cfg.python)
    overrides = user_overrides()
    dirty = build_paths_dirty(env_root)

    if overrides:
        say(f"build-layer caching disabled: local overrides active "
            f"({', '.join(overrides)}) — building privately")
    elif dirty:
        say(f"build layer: dirty csrc/cmake tree → private build ({bhash})")

    if not overrides and not dirty:
        entry = cfg.store("builds") / bhash
        wheel = _find_wheel(entry)
        if wheel is not None:
            say(f"build layer: cache HIT ({bhash})")
            touch_last_used(entry)
            return BuildResolution("store-wheel", wheel, bhash, shared=True)
        say(f"build layer: cache MISS ({bhash})")
        with entry_lock(entry):
            wheel = _find_wheel(entry)  # re-check after lock (lost race = win)
            if wheel is not None:
                say(f"build layer: cache HIT after wait ({bhash})")
                touch_last_used(entry)
                return BuildResolution("store-wheel", wheel, bhash, shared=True)
            # Fast path: build inputs match the merge-base commit on main →
            # fetch the upstream precompiled wheel into the store.
            fetched = try_fetch_precompiled(cfg, env_root, bhash)
            if fetched is not None:
                return BuildResolution("precompiled-fetch", fetched, bhash, shared=True)
            build_temp = cfg.store("cmake-build") / bhash
            dist_dir = _scratch(env_root) / "dist"
            try:
                built = _build_wheel(
                    cfg, env_root, venv, build_temp, dist_dir, use_pinned_ext=True
                )
            except Exception as e:
                warn(f"local build failed ({e}); falling back to upstream "
                     f"VLLM_USE_PRECOMPILED attach (not cached)")
                return BuildResolution("precompiled-fetch", None, bhash, shared=False)
            entry.mkdir(parents=True, exist_ok=True)
            target = entry / built.name
            shutil.copy2(built, target)
            target.chmod(0o444)
            write_meta(entry, {"kind": "build", "hash": bhash, "wheel": built.name})
            touch_last_used(entry)
            if build_temp.exists():
                write_meta(build_temp, {"kind": "cmake-build", "hash": bhash})
                touch_last_used(build_temp)
            say(f"build layer published → builds/{bhash}")
            return BuildResolution("store-wheel", target, bhash, shared=True)

    # dirty or override → private build, never published
    build_temp = _scratch(env_root) / "cmake-build"
    dist_dir = _scratch(env_root) / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    built = _build_wheel(
        cfg, env_root, venv, build_temp, dist_dir, use_pinned_ext=not overrides
    )
    return BuildResolution("local-build", built, bhash, shared=False)


# --------------------------------------------------------------------------
# Attach: editable install using the wheel as .so donor
# --------------------------------------------------------------------------


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
            m.parent.mkdir(parents=True, exist_ok=True)
            tmp = m.with_name(m.name + f".tmp{os.getpid()}")
            shutil.copy2(wt, tmp)
            tmp.chmod(0o444)
            os.replace(tmp, m)  # atomic; concurrent attachers write identical bytes
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


def attach(cfg: Config, env_root: Path, venv: Path, res: BuildResolution) -> None:
    store_entry = cfg.store("builds") / res.build_hash
    if res.shared:
        marker = read_marker(env_root)
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
    _uv_pip(venv, ["-e", str(env_root), "--no-build-isolation", "--no-deps"], env=env)
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
        build_hash=res.build_hash if res.shared else "",
        attach_mode=res.mode,
    )


# --------------------------------------------------------------------------
# Sync: full orchestration for the current worktree state
# --------------------------------------------------------------------------


def sync(cfg: Config, env_root: Path, fresh_venv: bool = False) -> None:
    if os.environ.get("VE_NO_SYNC") == "1":
        warn("VE_NO_SYNC=1 — env is STALE; run `ve sync` when ready")
        return
    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=env_root).stdout.strip()
    venv, _keys = resolve_venv(cfg, env_root, fresh=fresh_venv)
    res = resolve_build(cfg, env_root, venv)
    attach(cfg, env_root, venv, res)
    say(f"env consistent at {head}")
