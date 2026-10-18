"""Venv layers: base/full templates keyed on requirements, cloned per env."""

import re
import shutil
import subprocess
import time
from pathlib import Path

from . import rules, treefacts
from .config import Config, scratch_dir
from .drift import TEMPLATE_FREEZE_NAME
from .hashing import VenvKeys, detect_platform, venv_keys
from .locks import entry_lock
from .log import say, warn
from .registry import read_marker, update_marker
from .reqs import cap_constraints, freeze_names
from .sitepkgs import venv_freeze
from .store import read_meta, touch_last_used, write_meta
from .util import is_transient, reflink_clone, run, run_retrying_transient


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


def uv_pip_install(venv: Path, args: list[str], env: dict | None = None) -> None:
    run_retrying_transient(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *args],
        "uv pip install",
        env=env,
        stream_prefix="[ve]   [uv] ",
    )


_CAP_FALLBACK = {"minor": "major", "major": "none"}  # rule cap-fallback


def _install_context(
    keys: VenvKeys, platform: str, requirement_files: list[Path]
) -> rules.Context:
    facts = getattr(keys, "facts", None)
    if facts is not None:
        return facts.context(requirement_files)
    return rules.Context(platform, treefacts.host_platform.machine(),
                         files=tuple(requirement_files))


def _torch_backend_args(ctx: rules.Context) -> list[str]:
    if ctx.platform != "cuda":
        return []
    # uv's --torch-backend selects an index for torch packages only; rules'
    # ExtraIndex effects cover auxiliary wheels (e.g. torchcodec+cu130).
    args = ["--torch-backend=auto"]
    found = rules.effects(ctx, rules.ExtraIndex)
    strategies = sorted({e.strategy for _, e in found if e.strategy})
    for strategy in strategies:
        args += ["--index-strategy", strategy]
    for _, effect in found:
        for url in effect.urls(ctx):
            args += ["--extra-index-url", url]
    return args


def _installed_version(venv: Path, package: str) -> str:
    shown = run(
        ["uv", "pip", "show", "--python", str(venv / "bin" / "python"), package]
    ).stdout
    match = re.search(r"^Version:\s*(\S+)", shown, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{package} is required but was not installed")
    return match.group(1).split("+", 1)[0]


def _torch_cuda_variant(venv: Path) -> str:
    torch_cuda = run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import torch; print(torch.version.cuda or '')",
        ]
    ).stdout.strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", torch_cuda)
    if match is None:
        raise RuntimeError("cannot select a CUDA wheel index without CUDA version")
    return "".join(match.groups())


def _derived_installs(venv: Path, ctx: rules.Context) -> None:
    """Install packages rules derive from installed versions (DerivedInstall),
    e.g. flashinfer-jit-cache matching flashinfer-python and torch's CUDA."""
    variant = None
    for _, effect in rules.effects(ctx, rules.DerivedInstall):
        version = _installed_version(venv, effect.version_of)
        variant = variant or _torch_cuda_variant(venv)
        uv_pip_install(
            venv,
            [
                f"{effect.package}=={version}",
                "--no-deps",
                "--index-url",
                effect.index.format(cuda_variant=variant),
            ],
        )


class ResolutionError(subprocess.CalledProcessError):
    """uv could not resolve the requirement set (as opposed to failing to
    download or build it). Only this triggers ve's install fallbacks."""


def _install_reqs(
    cfg: Config, keys: VenvKeys, venv: Path, requirement_files: list[Path],
    platform: str, cap_dest: Path,
) -> str:
    """Install requirements with cap constraints, walking down the fallback
    chain (minor → major → none) when caps make resolution unsatisfiable —
    stale floors (e.g. tokenizers>=0.21 vs transformers needing >=0.22) can
    conflict under tight caps (rule cap-fallback). Returns the cap mode the
    install succeeded with."""
    ctx = _install_context(keys, platform, requirement_files)
    derived_requirements = sorted(
        {e.spec for _, e in rules.effects(ctx, rules.AddReq, layer="venv")}
    )
    override_requirements = sorted(
        {e.spec for _, e in rules.effects(ctx, rules.OverrideReq, layer="venv")}
    )
    req_args = [arg for f in requirement_files for arg in ("-r", str(f))]
    req_args.extend(derived_requirements)
    override_dest = cap_dest.with_name("overrides.txt")
    if override_requirements:
        override_dest.write_text("\n".join(override_requirements) + "\n")
        override_args = ["--overrides", str(override_dest)]
    else:
        override_dest.unlink(missing_ok=True)
        override_args = []
    mode = cfg.cap
    while True:
        content = cap_constraints(
            [*keys.layout.build_files, *keys.layout.runtime_files,
             *keys.layout.test_files], mode,
            extra_requirements=[*derived_requirements, *override_requirements],
        )
        cap_args = []
        if content:
            cap_dest.write_text(content)
            cap_args = ["-c", str(cap_dest)]
        else:
            cap_dest.unlink(missing_ok=True)
        install_args = [
            *req_args,
            *cap_args,
            *override_args,
            # Only the files being installed: they are what this layer's key
            # hashes (the base venv must not depend on the test lock).
            *_torch_backend_args(_install_context(keys, platform, requirement_files)),
        ]
        # Resolve first so only an unsatisfiable set reaches the split-pass
        # fallback in _install_req_groups. Caps are a heuristic, so any
        # non-transient failure under them loosens them: a capped floor can
        # select an old sdist with no wheel for this Python that no longer
        # builds (pyzmq<25.1 on 3.12 / g++ 14).
        try:
            run_retrying_transient(
                ["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
                 "--dry-run", *install_args],
                "uv resolve",
            )
        except subprocess.CalledProcessError as probe:
            if is_transient(probe):
                raise
            nxt = _CAP_FALLBACK.get(mode) if cap_args else None
            if nxt is None:
                raise ResolutionError(probe.returncode, probe.cmd,
                                      output=probe.output, stderr=probe.stderr) from probe
            say(f"venv layer: cap={mode} incompatible → trying cap={nxt}")
            mode = nxt
            continue
        try:
            uv_pip_install(venv, install_args)
        except subprocess.CalledProcessError as e:
            nxt = _CAP_FALLBACK.get(mode) if cap_args else None
            if nxt is None or is_transient(e):
                raise
            warn(f"venv layer: install failed under cap={mode} → trying cap={nxt}")
            mode = nxt
            continue
        _derived_installs(venv, ctx)
        return mode


def _install_req_groups(
    cfg: Config, keys: VenvKeys, venv: Path, groups: list[list[Path]],
    platform: str, cap_dest: Path,
) -> dict:
    """Install requirement groups in one resolve, falling back to one pass per
    group when their pins cannot hold simultaneously.

    vLLM's compiled test lock (requirements/test/<platform>.txt) periodically
    goes stale against requirements/<platform>.txt — e.g. the lock pinning
    cuda-pathfinder==1.3.3 while the pinned flashinfer-python needs >=1.5.4.
    Resolving every file at once is stricter than vLLM itself: upstream installs
    those files in separate passes, so such pairs never have to agree.  Mirror
    that rather than failing the layer; groups are installed in order, so the
    last group's pins win wherever they overlap (rule split-install-passes).
    Returns how the install went, for the template's meta."""
    files = [f for group in groups for f in group]
    if not files:
        return {}
    try:
        return {"cap": _install_reqs(cfg, keys, venv, files, platform, cap_dest),
                "split_passes": False}
    except ResolutionError:
        groups = [group for group in groups if group]
        if len(groups) < 2:
            raise
        warn("requirement files cannot be resolved together → installing them "
             "in separate passes (later files win where they overlap)")
    caps = [_install_reqs(cfg, keys, venv, group, platform, cap_dest)
            for group in groups]
    return {"cap": caps, "split_passes": True}


def template_freeze(entry: Path) -> list[str]:
    f = entry / "freeze.txt"
    if not f.exists():  # templates predating freeze tracking
        f.write_text("\n".join(venv_freeze(entry, strict=True)) + "\n")
    return f.read_text().splitlines()


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
        requirement_files = keys.layout.build_files
        cap_used = None
        if requirement_files:
            cap_used = _install_reqs(cfg, keys, entry, requirement_files,
                                     platform, entry / "constraints.txt")
        (entry / ".complete").touch()
        write_meta(entry, {"kind": "venv-base", "hash": keys.base_hash,
                           "cap": cap_used})
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
        meta = read_meta(entry)
        resumable = (
            entry.is_dir()
            and (entry / "bin" / "python").exists()
            and meta.get("kind") == "venv-full"
            and meta.get("hash") == keys.full_hash
            and meta.get("base") == keys.base_hash
        )
        if resumable:
            say(
                f"venv full layer: cache INCOMPLETE ({keys.full_hash}) — "
                f"resuming {meta.get('stage', 'dependency install')}"
            )
        else:
            say(
                f"venv full layer: cache MISS ({keys.full_hash}) — "
                "deriving from base"
            )
            if entry.exists():
                shutil.rmtree(entry)
            reflink_clone(base, entry)
            _rewrite_venv_paths(base, entry)
            (entry / ".complete").unlink(missing_ok=True)
        attempt = {
            "kind": "venv-full",
            "hash": keys.full_hash,
            "base": keys.base_hash,
            "state": "building",
            "stage": "installing test/runtime dependencies",
            "size_bytes": meta.get("size_bytes", 0),
        }
        write_meta(entry, attempt)
        try:
            # Test lock first so runtime pins win if they must be split.
            install = _install_req_groups(
                cfg, keys, entry,
                [keys.layout.test_files, keys.layout.runtime_files],
                platform, entry / "constraints.txt",
            )
            attempt["stage"] = "recording installed packages"
            write_meta(entry, attempt)
            (entry / "freeze.txt").write_text(
                "\n".join(venv_freeze(entry, strict=True)) + "\n"
            )
        except BaseException as exc:
            attempt.update(
                state="incomplete",
                error=type(exc).__name__,
                failed_at=time.time(),
            )
            write_meta(entry, attempt)
            raise
        (entry / ".complete").touch()
        write_meta(
            entry,
            {"kind": "venv-full", "hash": keys.full_hash,
             "base": keys.base_hash, "state": "complete", **install},
        )
        touch_last_used(entry)
    return entry


def resolve_venv(cfg: Config, env_root: Path, fresh: bool = False) -> tuple[Path, VenvKeys]:
    """Ensure env_root/.venv matches the worktree's requirements."""
    platform = cfg.platform or detect_platform()
    keys = venv_keys(env_root, platform, cfg.python,
                     with_test=cfg.with_test, cap=cfg.cap)
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
            _install_req_groups(
                cfg, keys, venv,
                [keys.layout.test_files, keys.layout.runtime_files],
                platform, scratch_dir(env_root, create=True) / "constraints.txt",
            )
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
    return scratch_dir(env_root, create=True) / TEMPLATE_FREEZE_NAME


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
        removed = freeze_names(record.read_text().splitlines()) - freeze_names(new_lines)
        installed = freeze_names(venv_freeze(venv, strict=True))
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
