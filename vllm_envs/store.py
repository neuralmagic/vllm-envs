import json
import time
from pathlib import Path

from .util import apportioned_size_bytes, dir_size_bytes

META_NAME = "meta.json"
LAST_USED_NAME = ".last-used"


def write_meta(entry: Path, data: dict) -> None:
    data = dict(data)
    data.setdefault("created_at", time.time())
    if "size_bytes" not in data:
        data["size_bytes"] = dir_size_bytes(entry)
    (entry / META_NAME).write_text(json.dumps(data, indent=2))


def read_meta(entry: Path) -> dict:
    try:
        return json.loads((entry / META_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def touch_last_used(entry: Path) -> None:
    try:
        (entry / LAST_USED_NAME).touch()
    except OSError:
        pass


def last_used(entry: Path) -> float:
    for candidate in (entry / LAST_USED_NAME, entry / META_NAME, entry):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def entry_size(entry: Path) -> int:
    """Physical apportioned size, computed live: recorded size_bytes goes stale
    (e.g. the builds/ extracted mirror lands after publish) and logical sizes
    overcount hardlink sharing — the gc cap enforces physical usage."""
    return apportioned_size_bytes(entry)


def is_pinned(entry: Path) -> bool:
    return bool(read_meta(entry).get("pinned"))
