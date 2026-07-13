"""Reflink-aware disk accounting via the FIEMAP ioctl.

`du` deduplicates hardlinks (same inode) but is blind to reflinks: distinct
inodes that share physical extents (CoW clones, as produced by
`cp --reflink`). On XFS/btrfs the whole vllm-envs store is reflinked, so `du`
reports each venv at full size and massively overcounts the real footprint.

This walks files, reads their physical extent maps, and counts each physical
extent once — giving the true on-disk size and, per group, the bytes that
would actually be reclaimed if that group alone were deleted.
"""

from __future__ import annotations

import fcntl
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 0x00000001

FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
_SKIP_FLAGS = FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_DATA_INLINE

_HDR = struct.Struct("=QQIIII")  # start, length, flags, mapped, count, reserved
_EXT = struct.Struct("=QQQ QQ I 3I")  # logical, physical, length, res64[2], flags, res[3]
_BATCH = 512
_SHARED = -1  # owner sentinel: extent referenced by more than one group


def _extents(fd: int) -> Iterable[tuple[int, int]]:
    """Yield (physical_offset, length) for each real extent of an open file."""
    start = 0
    while True:
        buf = bytearray(_HDR.size + _EXT.size * _BATCH)
        _HDR.pack_into(buf, 0, start, 1 << 63, FIEMAP_FLAG_SYNC, 0, _BATCH, 0)
        try:
            fcntl.ioctl(fd, FS_IOC_FIEMAP, buf)
        except OSError:
            return
        mapped = _HDR.unpack_from(buf, 0)[3]
        if mapped == 0:
            return
        end = start
        for i in range(mapped):
            logical, physical, length, _r0, _r1, flags, *_ = _EXT.unpack_from(
                buf, _HDR.size + _EXT.size * i
            )
            end = logical + length
            if physical and not (flags & _SKIP_FLAGS):
                yield physical, length
            if flags & FIEMAP_EXTENT_LAST:
                return
        start = end


def _merge_len(intervals: list[tuple[int, int]]) -> int:
    """Total bytes covered by the union of [start, end) intervals."""
    intervals.sort()
    total = cur_s = cur_e = 0
    have = False
    for s, e in intervals:
        if not have or s > cur_e:
            if have:
                total += cur_e - cur_s
            cur_s, cur_e, have = s, e, True
        elif e > cur_e:
            cur_e = e
    if have:
        total += cur_e - cur_s
    return total


@dataclass
class ReflinkUsage:
    supported: bool
    unique_total: int = 0  # union of all extents across every group (true footprint)
    apparent: dict[str, int] = field(default_factory=dict)  # sum of extents, ~ du
    exclusive: dict[str, int] = field(default_factory=dict)  # reclaimed if group deleted
    unique_by_scope: dict[str, int] = field(default_factory=dict)  # phys per scope
    exclusive_by_scope: dict[str, int] = field(default_factory=dict)  # scope-only phys


def reflink_usage(
    groups: dict[str, Iterable[Path]],
    scopes: dict[str, str] | None = None,
) -> ReflinkUsage:
    """Physical-extent accounting for named groups of directory roots.

    A physical extent is "exclusive" to a group when every file referencing it
    lives in that group; deleting the group frees those bytes. Extents shared
    across groups belong to none exclusively.

    `scopes` maps group name -> scope label. `unique_by_scope` reports all
    physical bytes referenced by a scope, while `exclusive_by_scope` reports
    bytes referenced by that scope and no other scope.
    """
    # key: (st_dev, physical_offset) -> owning group index, or _SHARED
    owner: dict[tuple[int, int], int] = {}
    length: dict[tuple[int, int], int] = {}
    apparent = {name: 0 for name in groups}
    supported = False

    scope_bit = {sc: 1 << i for i, sc in enumerate(sorted(set(scopes.values())))} \
        if scopes else {}
    group_bit = {n: scope_bit[scopes[n]] for n in groups} if scopes else {}
    mask: dict[tuple[int, int], int] = {}

    for gid, (name, roots) in enumerate(groups.items()):
        seen_inodes: set[tuple[int, int]] = set()  # dedupe hardlinks (like du)
        for root in roots:
            for dirpath, _dirs, files in os.walk(root):
                for fname in files:
                    path = os.path.join(dirpath, fname)
                    try:
                        st = os.lstat(path)
                    except OSError:
                        continue
                    if not (st.st_mode & 0o170000) == 0o100000:
                        continue
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    except OSError:
                        continue
                    ino = (st.st_dev, st.st_ino)
                    new_inode = ino not in seen_inodes
                    seen_inodes.add(ino)
                    try:
                        for physical, ext_len in _extents(fd):
                            supported = True
                            key = (st.st_dev, physical)
                            length[key] = ext_len
                            if new_inode:  # apparent == du: reflinks counted per file
                                apparent[name] += ext_len
                            if scopes:
                                mask[key] = mask.get(key, 0) | group_bit[name]
                            prev = owner.get(key)
                            if prev is None:
                                owner[key] = gid
                            elif prev != gid:
                                owner[key] = _SHARED
                    finally:
                        os.close(fd)

    if not supported:
        return ReflinkUsage(supported=False)

    per_dev: dict[int, list[tuple[int, int]]] = {}
    per_scope: dict[str, dict[int, list[tuple[int, int]]]] = {
        sc: {} for sc in scope_bit
    }
    exclusive_scope: dict[str, dict[int, list[tuple[int, int]]]] = {
        sc: {} for sc in scope_bit
    }
    exclusive = {name: 0 for name in groups}
    names = list(groups)
    for (dev, phys), ext_len in length.items():
        iv = (phys, phys + ext_len)
        per_dev.setdefault(dev, []).append(iv)
        m = mask.get((dev, phys), 0)
        for sc, bit in scope_bit.items():
            if m & bit:
                per_scope[sc].setdefault(dev, []).append(iv)
            if m == bit:
                exclusive_scope[sc].setdefault(dev, []).append(iv)
        gid = owner[(dev, phys)]
        if gid != _SHARED:
            exclusive[names[gid]] += ext_len

    def merge(devs: dict[int, list[tuple[int, int]]]) -> int:
        return sum(_merge_len(ivs) for ivs in devs.values())

    return ReflinkUsage(
        supported=True,
        unique_total=merge(per_dev),
        apparent=apparent,
        exclusive=exclusive,
        unique_by_scope={sc: merge(devs) for sc, devs in per_scope.items()},
        exclusive_by_scope={
            sc: merge(devs) for sc, devs in exclusive_scope.items()
        },
    )
