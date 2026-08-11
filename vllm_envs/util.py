import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from .log import say


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
        try:
            for line in proc.stdout:
                recent.append(line)
                print(
                    f"{stream_prefix}{line.rstrip()}", file=sys.stderr, flush=True
                )
        finally:
            proc.stdout.close()
        proc.wait()
        output = "".join(recent)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=output
            )
        return subprocess.CompletedProcess(cmd, proc.returncode, output, "")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        capture_output=capture,
        text=True,
        check=check,
    )


def git(args: list[str], cwd: Path | str, check: bool = True) -> str:
    return run(["git", *args], cwd=cwd, check=check).stdout.strip()


def reflink_clone(src: Path, dst: Path) -> None:
    """Clone a directory tree with XFS/btrfs reflinks, falling back to a copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(["cp", "-a", "--reflink=always", str(src), str(dst)])
    except subprocess.CalledProcessError:
        say(f"reflink unsupported, falling back to plain copy: {src} -> {dst}")
        if dst.exists():
            shutil.rmtree(dst)
        run(["cp", "-a", "--reflink=auto", str(src), str(dst)])


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


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
