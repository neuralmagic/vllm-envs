"""Build layer: compiled extensions, one shared wheel per build hash."""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import editable
from .config import Config, scratch_dir
from .extprojects import src_dir_env, user_overrides
from .hashing import build_key, build_paths_dirty, detect_platform
from .locks import entry_lock
from .log import say, warn
from .precompiled import try_fetch_precompiled
from .registry import read_marker
from .store import touch_last_used, write_meta
from .util import atomic_copy, run


@dataclass
class BuildResolution:
    mode: str  # "store-wheel" | "precompiled-fetch" | "local-build"
    wheel: Path | None
    build_hash: str
    shared: bool  # attached to the shared store (refcounted)
    build_error: BaseException | None = None  # local build failure behind a fallback


def find_wheel(dirpath: Path) -> Path | None:
    wheels = sorted(dirpath.glob("vllm-*.whl")) if dirpath.is_dir() else []
    return wheels[-1] if wheels else None


def local_fallback_current(
    env_root: Path, venv: Path, marker: dict, build_hash: str
) -> bool:
    """The env already carries an uncached upstream-precompiled attach for
    `build_hash` (a local build failed earlier)."""
    return (
        marker.get("build_hash") == build_hash
        and marker.get("attach_mode") == "precompiled-fetch"
        and editable.editable_present(venv)
        and any((env_root / "vllm").glob("*.so"))
    )


def local_build_env(
    env_root: Path, venv: Path | None = None, cfg: Config | None = None
) -> dict[str, str]:
    """Environment shared by every local C/C++/CUDA compilation."""
    env = {
        "VLLM_DISABLE_SCCACHE": "1",  # local ccache policy
        # Every worktree builds in <worktree>/.ve/cmake-build, so with paths
        # under the worktree rewritten relative, an unchanged translation
        # unit hashes the same in every worktree (measured: 29/30 hits for a
        # second worktree's _moe_C build). NOHASHDIR keeps the differing
        # compilation directory (RelWithDebInfo debug info) out of the hash.
        "CCACHE_BASEDIR": str(env_root),
        "CCACHE_NOHASHDIR": "true",
        "CMAKE_C_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
        # Per-env fetchcontent dir: no cross-env build-dir races for any
        # project we couldn't pin.
        "FETCHCONTENT_BASE_DIR": str(
            scratch_dir(env_root, create=True) / "fetchcontent"
        ),
    }
    if cfg is not None:
        env["CCACHE_DIR"] = str(cfg.ccache_dir)
        env["CCACHE_MAXSIZE"] = f"{cfg.ccache_max_size_gb:g}G"
    if venv is not None:
        # Running the venv's Python does not update PATH for subprocesses that
        # setup.py/CMake launch. Make its approved cmake/ninja executables
        # authoritative instead of depending on the invoking shell.
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', os.defpath)}"
    return env


def _build_wheel(
    cfg: Config,
    env_root: Path,
    venv: Path,
    build_temp: Path,
    dist_dir: Path,
    use_pinned_ext: bool,
) -> Path:
    env = local_build_env(env_root, venv, cfg)
    if use_pinned_ext:
        env.update(src_dir_env(cfg, env_root))
    build_temp.mkdir(parents=True, exist_ok=True)
    # This scratch directory survives commit hops. Never let find_wheel pick
    # a successful wheel left by a different branch/build hash.
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
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
            # The explicit build_ext above already compiled and installed all
            # extension outputs into build/lib. Without this, bdist_wheel runs
            # build_ext a second time in the same Python process; modern vLLM
            # skips CMake configure via its did_config guard and can then fail
            # with "Error: could not load cache".
            "--skip-build",
            "--dist-dir",
            str(dist_dir),
        ],
        cwd=env_root,
        env=env,
        stream_prefix="[ve]   ",
    )
    say(f"build finished in {time.time() - t0:.0f}s")
    wheel = find_wheel(dist_dir)
    if wheel is None:
        raise RuntimeError(f"build produced no wheel in {dist_dir}")
    return wheel


def resolve_build(cfg: Config, env_root: Path, venv: Path) -> BuildResolution:
    platform = cfg.platform or detect_platform()
    bhash = build_key(env_root, platform, cfg.python, venv)
    overrides = user_overrides()
    dirty = build_paths_dirty(env_root)
    marker = read_marker(env_root)

    if overrides:
        say(f"build-layer caching disabled: local overrides active "
            f"({', '.join(overrides)}) — building privately")
    elif dirty:
        say(f"build layer: dirty csrc/cmake tree → private build ({bhash})")

    if not overrides and not dirty:
        entry = cfg.store("builds") / bhash
        wheel = find_wheel(entry)
        if wheel is not None:
            say(f"build layer: cache HIT ({bhash})")
            touch_last_used(entry)
            return BuildResolution("store-wheel", wheel, bhash, shared=True)
        say(f"build layer: cache MISS ({bhash})")
        with entry_lock(entry):
            wheel = find_wheel(entry)  # re-check after lock (lost race = win)
            if wheel is not None:
                say(f"build layer: cache HIT after wait ({bhash})")
                touch_last_used(entry)
                return BuildResolution("store-wheel", wheel, bhash, shared=True)
            # Fast path: build inputs match the merge-base commit on main →
            # fetch the upstream precompiled wheel into the store.
            fetched = try_fetch_precompiled(cfg, env_root, bhash)
            if fetched is not None:
                return BuildResolution("precompiled-fetch", fetched, bhash, shared=True)
            if local_fallback_current(env_root, venv, marker, bhash):
                say(
                    f"build layer: env-local precompiled fallback up to date ({bhash})"
                )
                return BuildResolution(
                    "precompiled-fetch", None, bhash, shared=False
                )
            if not cfg.allow_local_build:
                raise RuntimeError("local builds are disabled")
            build_temp = scratch_dir(env_root, create=True) / "cmake-build"
            dist_dir = scratch_dir(env_root, create=True) / "dist"
            try:
                built = _build_wheel(
                    cfg, env_root, venv, build_temp, dist_dir, use_pinned_ext=True
                )
            except Exception as e:
                warn(f"local build failed ({e}); falling back to upstream "
                     f"VLLM_USE_PRECOMPILED attach (not cached)")
                return BuildResolution(
                    "precompiled-fetch", None, bhash, shared=False, build_error=e
                )
            target = entry / built.name
            atomic_copy(built, target, 0o444)
            write_meta(entry, {"kind": "build", "hash": bhash, "wheel": built.name})
            touch_last_used(entry)
            say(f"build layer published → builds/{bhash}")
            return BuildResolution("store-wheel", target, bhash, shared=True)

    # dirty or override → private build, never published
    if not cfg.allow_local_build:
        raise RuntimeError("local builds are disabled")
    build_temp = scratch_dir(env_root, create=True) / "cmake-build"
    dist_dir = scratch_dir(env_root, create=True) / "dist"
    built = _build_wheel(
        cfg, env_root, venv, build_temp, dist_dir, use_pinned_ext=not overrides
    )
    return BuildResolution("local-build", built, bhash, shared=False)
