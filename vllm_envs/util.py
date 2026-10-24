import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from .log import say, warn

_leftover_spools: list[str] = []


@atexit.register
def _remove_leftover_spools() -> None:
    for name in _leftover_spools:
        Path(name).unlink(missing_ok=True)


def run(
    cmd: list[str],
    cwd: Path | str | None = None,
    env: dict[str, str | None] | None = None,
    capture: bool = True,
    check: bool = True,
    stream_prefix: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command. stream_prefix streams output to stderr line by line."""
    full_env = dict(os.environ)
    if env:
        for name, value in env.items():
            if value is None:
                full_env.pop(name, None)
            else:
                full_env[name] = value
    if stream_prefix is not None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        recent = deque(maxlen=200)
        # The decisive error (e.g. ninja's FAILED block) is often thousands of
        # lines before the end, so keep the full output for the incident log.
        spool = tempfile.NamedTemporaryFile(
            "w", prefix="ve-output-", suffix=".log", delete=False
        )
        _leftover_spools.append(spool.name)
        try:
            with spool:
                for line in proc.stdout:
                    recent.append(line)
                    spool.write(line)
                    print(
                        f"{stream_prefix}{line.rstrip()}", file=sys.stderr,
                        flush=True,
                    )
        finally:
            proc.stdout.close()
            proc.wait()
            if not (check and proc.returncode != 0):
                Path(spool.name).unlink(missing_ok=True)
        output = "".join(recent)
        if check and proc.returncode != 0:
            err = subprocess.CalledProcessError(proc.returncode, cmd, output=output)
            err.output_log = spool.name
            raise err
        return subprocess.CompletedProcess(cmd, proc.returncode, output, "")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        capture_output=capture,
        text=True,
        check=check,
    )


# Package-index/network failures that a later attempt can succeed past.
_TRANSIENT = re.compile(
    r"429 Too Many Requests|50[234] [A-Z]|Request failed after \d+ retr|"
    r"error sending request|operation timed out|Connection reset|"
    r"Temporary failure in name resolution"
)


def is_transient(exc: subprocess.CalledProcessError) -> bool:
    return bool(_TRANSIENT.search(f"{exc.output or ''}\n{exc.stderr or ''}"))


def run_retrying_transient(
    cmd: list[str], what: str, delays: tuple[int, ...] = (5, 15, 30), **kwargs
) -> subprocess.CompletedProcess:
    """`run(..., stream_prefix=...)` that retries network flakes."""
    for attempt in range(len(delays) + 1):
        try:
            return run(cmd, **kwargs)
        except subprocess.CalledProcessError as exc:
            if not is_transient(exc) or attempt == len(delays):
                raise
            delay = delays[attempt]
            warn(f"{what} hit a transient network error; retrying in {delay}s "
                 f"({attempt + 1}/{len(delays)})", log=False)
            time.sleep(delay)
    raise AssertionError("unreachable")


def atomic_write_text(path: Path, text: str, mode: int | None = None) -> None:
    """Write via a sibling temp file + rename, so readers never see a
    truncated file (e.g. after a full disk or a killed process). `mode` is
    applied before the rename."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text)
        if mode is not None:
            tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_copy(src: Path, dst: Path, mode: int) -> None:
    """copy2 + chmod to a sibling temp file, then rename: concurrent writers
    of identical bytes race harmlessly and readers never see a partial file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    shutil.copy2(src, tmp)
    tmp.chmod(mode)
    os.replace(tmp, dst)


def git(args: list[str], cwd: Path | str | None = None, check: bool = True) -> str:
    return run(["git", *args], cwd=cwd, check=check).stdout.strip()


def git_path(cwd: Path, *args: str, check: bool = True) -> Path | None:
    """A `git rev-parse` path query (e.g. --git-common-dir, --git-path X),
    made absolute; None when it fails with check=False."""
    out = git(["rev-parse", *args], cwd=cwd, check=check)
    if not out:
        return None
    path = Path(out)
    return path if path.is_absolute() else Path(cwd) / path


def git_toplevel(cwd: Path | str) -> Path:
    return Path(git(["rev-parse", "--show-toplevel"], cwd=cwd))


_reflink_support: dict[tuple[int, int], bool] = {}


def _supports_reflink(src: Path, dst_parent: Path) -> bool:
    """Probe and cache reflink support for a source/destination device pair."""
    devices = (src.stat().st_dev, dst_parent.stat().st_dev)
    if devices in _reflink_support:
        return _reflink_support[devices]
    if devices[0] != devices[1]:
        _reflink_support[devices] = False
        return False

    with tempfile.TemporaryDirectory(
        dir=dst_parent, prefix=".ve-reflink-probe-"
    ) as probe_dir:
        probe_src = Path(probe_dir) / "src"
        probe_dst = Path(probe_dir) / "dst"
        probe_src.write_bytes(b"\0")
        result = run(
            [
                "cp",
                "--reflink=always",
                "--",
                str(probe_src),
                str(probe_dst),
            ],
            check=False,
        )
    supported = result.returncode == 0
    _reflink_support[devices] = supported
    return supported


def reflink_clone(src: Path, dst: Path) -> None:
    """Clone a directory tree with reflinks, falling back to a plain copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _supports_reflink(src, dst.parent):
        try:
            run(
                [
                    "cp",
                    "-a",
                    "--reflink=always",
                    "--",
                    str(src),
                    str(dst),
                ]
            )
            return
        except subprocess.CalledProcessError:
            if dst.exists():
                shutil.rmtree(dst)

    say(f"reflink unsupported, falling back to plain copy: {src} -> {dst}")
    run(["cp", "-a", "--", str(src), str(dst)])


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def apportioned_size_bytes(path: Path) -> int:
    """Physical bytes attributed to this tree: st_blocks/st_nlink per file, so
    hardlink-shared files (uv cache, sibling templates) count a fair share.
    Reflink-shared extents are still counted once per file (invisible to stat).
    """
    total = 0.0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            total += st.st_blocks * 512 / max(st.st_nlink, 1)
    return int(total)


def age_str(epoch: float | None, digits: int = 0) -> str:
    """Time since `epoch` in hours, or days from 48h on."""
    if not epoch:
        return "?"
    h = (time.time() - epoch) / 3600
    if h < 48:
        return f"{h:.{digits}f}h"
    return f"{h / 24:.{digits}f}d"


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
