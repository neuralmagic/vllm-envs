"""Layer resolution: venv templates (1a/1b), compiled extensions (3), attach."""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import SCRATCH_DIR_NAME, Config
from .extprojects import src_dir_env, user_overrides
from .hashing import VenvKeys, build_key, build_paths_dirty, detect_platform, venv_keys
from .locks import entry_lock
from .log import say, warn
from .precompiled import try_fetch_precompiled
from .registry import read_marker, update_marker
from .store import touch_last_used, write_meta
from .util import reflink_clone, run


def _uv_pip(venv: Path, args: list[str], env: dict | None = None) -> None:
    run(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *args],
        env=env,
        stream_prefix="[ve]   [uv] ",
    )


def _torch_backend_args(platform: str) -> list[str]:
    return ["--torch-backend=auto"] if platform == "cuda" else []


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
        (entry / ".complete").unlink(missing_ok=True)
        req_args: list[str] = []
        for f in keys.layout.runtime_files:
            req_args += ["-r", str(f)]
        if req_args:
            _uv_pip(entry, [*req_args, *_torch_backend_args(platform)])
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
            # Base unchanged: converge the private venv in place (top-up).
            # Keeps agent divergence (debug prints); removed deps linger.
            say(f"venv layer: requirements changed → top-up install into env venv "
                f"({current} → {keys.full_hash})")
            req_args: list[str] = []
            for f in keys.layout.runtime_files:
                req_args += ["-r", str(f)]
            if req_args:
                _uv_pip(venv, [*req_args, *_torch_backend_args(platform)])
            update_marker(env_root, venv_full_hash=keys.full_hash, venv_base_hash=keys.base_hash)
            return venv, keys
        warn("torch/build deps changed across hop → replacing env venv with a "
             "fresh clone (local site-packages edits are discarded)")

    template = ensure_full_template(cfg, keys, platform)
    if venv.exists():
        shutil.rmtree(venv)
    t0 = time.time()
    reflink_clone(template, venv)
    (venv / ".complete").unlink(missing_ok=True)
    say(f"venv layer: reflink-cloned template {keys.full_hash} ({time.time() - t0:.1f}s)")
    update_marker(env_root, venv_full_hash=keys.full_hash, venv_base_hash=keys.base_hash)
    return venv, keys


# --------------------------------------------------------------------------
# Layer 3: compiled extensions (wheel donor per build-hash)
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
        say(f"layer-3 caching disabled: local overrides active "
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


def attach(cfg: Config, env_root: Path, venv: Path, res: BuildResolution) -> None:
    env = {"VLLM_USE_PRECOMPILED": "1"}
    if res.wheel is not None:
        env["VLLM_PRECOMPILED_WHEEL_LOCATION"] = str(res.wheel)
        say(f"attaching extensions from {res.wheel.name} + editable install")
    else:
        say("attaching via upstream precompiled wheel fetch + editable install")
    _uv_pip(venv, ["-e", str(env_root), "--no-build-isolation", "--no-deps"], env=env)
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
