import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def entry_lock(entry_dir: Path):
    """Exclusive flock scoped to a store entry (sidecar .lock next to it)."""
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = entry_dir.parent / (entry_dir.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def try_entry_lock(entry_dir: Path) -> int | None:
    """Non-blocking acquire; returns fd (caller closes) or None if held."""
    lock_path = entry_dir.parent / (entry_dir.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None


def release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
