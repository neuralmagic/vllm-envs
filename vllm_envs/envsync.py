"""Sync: bring a worktree's venv, build and extras layers to its HEAD."""

import os
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

from . import drift, treefacts
from .attach import attach
from .buildlayer import resolve_build
from .config import Config, scratch_dir
from .extras import sync_vllm_extras
from .hashing import detect_platform
from .locks import entry_lock
from .log import error, say, warn
from .util import git
from .venvlayer import resolve_venv


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def _failure_only_sync_log(env_root: Path):
    scratch = scratch_dir(env_root, create=True)
    failed = scratch / "last-sync-failure.log"
    cutoff = time.time() - 86400
    for stale in scratch.glob(".sync.*.pending"):  # left by killed syncs
        try:
            if stale.stat().st_mtime < cutoff:
                stale.unlink()
        except OSError:
            pass
    original_stderr = sys.stderr
    with tempfile.NamedTemporaryFile(
        mode="w", dir=scratch, prefix=".sync.", suffix=".pending", delete=False
    ) as log, redirect_stderr(_Tee(original_stderr, log)):
        pending = Path(log.name)
        try:
            yield
        except BaseException:
            log.flush()
            pending.replace(failed)
            print(
                f"[ve] failure log retained at {failed}",
                file=original_stderr,
                flush=True,
            )
            raise
        else:
            pending.unlink(missing_ok=True)
            failed.unlink(missing_ok=True)


def sync(cfg: Config, env_root: Path, fresh_venv: bool = False) -> None:
    if os.environ.get("VE_NO_SYNC") == "1":
        warn("VE_NO_SYNC=1 — env is STALE; run `ve sync` when ready", log=False)
        return
    with entry_lock(scratch_dir(env_root, create=True) / "sync"), treefacts.scope():
        with _failure_only_sync_log(env_root), _tracking_drift(env_root):
            head = git(["rev-parse", "--short", "HEAD"], cwd=env_root)
            venv, keys = resolve_venv(cfg, env_root, fresh=fresh_venv)
            res = resolve_build(cfg, env_root, venv)
            attach(cfg, env_root, venv, res)
            platform = cfg.platform or detect_platform()
            if cfg.with_vllm_extras and platform == "cuda":
                sync_vllm_extras(cfg, env_root, venv, keys.base_hash, keys.full_hash)
            say(f"env consistent at {head}")


def report_sync_failure(exc: Exception) -> int:
    """Shared by `ve sync` and the git hooks; returns the exit code."""
    error(f"sync failed: {exc}")
    warn("previous .venv retained; its hashes may be STALE for this checkout",
         log=False)
    return 1


def _venv_identity(venv: Path) -> int | None:
    try:
        return venv.stat().st_ino
    except OSError:
        return None


@contextmanager
def _tracking_drift(env_root: Path):
    """Log hand-made venv changes found before this sync, then fold the
    sync's own package changes into the drift baseline."""
    venv = env_root / ".venv"
    pre = identity = None
    started = time.time()
    try:
        pre = drift.venv_freeze(venv)
        identity = _venv_identity(venv)
        if pre is not None:
            drift.record_if_new(
                env_root, drift.compute(env_root, pre, scan_files=False),
                "found by ve sync",
            )
    except Exception as e:
        warn(f"drift check skipped ({e})", log=False)
    try:
        yield
    finally:
        try:
            post = drift.venv_freeze(venv)
            recloned = identity is not None and identity != _venv_identity(venv)
            if (recloned or post != pre
                    or not drift.baseline_path(env_root).exists()
                    or drift.dists_written_since(venv, started)):
                if pre is not None:
                    # the baseline is about to move past these; log them first
                    drift.record_if_new(
                        env_root, drift.compute(env_root, pre, before=started),
                        "found by ve sync",
                    )
                drift.fold_into_baseline(env_root, pre, post, recloned)
        except Exception as e:
            warn(f"drift baseline not updated ({e})", log=False)
