"""Find and fetch an upstream precompiled wheel (wheels.vllm.ai) whose compiled
sources are identical to this worktree's.

"Compiled sources" follows vLLM's own definition: the paths its setup.py lists
in WHEEL_BLOCKING_PATHS. A commit's identity is the git object id of each of
those paths, so any main commit with HEAD's identity — before or after the
merge-base — has a wheel with exactly the extensions HEAD would build.
Identities form contiguous runs on main, and publication lags the tip by a few
commits, so the nearest match to the merge-base usually has a wheel.
"""

import hashlib
import json
import platform as _platform
import re
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote, urljoin

from .config import Config
from .extprojects import user_overrides
from .hashing import build_paths_dirty, cuda_version
from .log import say
from .sitepkgs import torch_cuda_major
from .store import touch_last_used, write_meta
from .treefacts import (  # noqa: F401 (defaults re-exported)
    DEFAULT_BLOCKING_PATHS,
    DEFAULT_WARNING_PATHS,
    wheel_paths,
)
from .util import atomic_write_text, git, run

WHEELS_BASE = "https://wheels.vllm.ai/"
CUDA_VARIANTS = {"12": "cu129", "13": "cu130"}
UPSTREAMS = ("origin/main", "upstream/main", "main")
SEARCH_BEHIND = 300  # main commits older than the merge-base to consider
SEARCH_AHEAD = 2000  # bounds the scan for long-lived branches
MAX_PROBES = 8
PROBE_TIMEOUT_S = 10
DOWNLOAD_TIMEOUT_S = 30
NEGATIVE_TTL_S = 3600
INDEX_NAME = "precompiled-index.json"
MISSING = "missing"


@dataclass
class Candidate:
    commit: str
    # Signed first-parent distance from the merge-base on main (+ newer,
    # - older); None for a HEAD that is not on main (e.g. a release branch).
    distance: int | None
    # Warning paths (setup.py, rust) that differ from HEAD: vLLM only warns.
    stale: list[str] = field(default_factory=list)


@dataclass
class Lookup:
    head: str
    paths: tuple[str, ...]
    warning_paths: tuple[str, ...]
    identity: str
    upstream: str | None = None
    merge_base: str | None = None
    scanned: int = 0
    matches: list[Candidate] = field(default_factory=list)
    variant: str | None = None
    arch: str = ""
    probes: list[tuple[str, str]] = field(default_factory=list)  # commit, result
    chosen: Candidate | None = None
    url: str | None = None
    source: str = ""  # "index" | "probe"
    reason: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def describe(distance: int | None) -> str:
    if distance is None:
        return "exact HEAD (off main)"
    if distance == 0:
        return "at merge-base"
    side = "after" if distance > 0 else "before"
    return f"{abs(distance)} commit(s) {side} merge-base"


# --------------------------------------------------------------------------
# Compiled-source identity
# --------------------------------------------------------------------------


def compiled_paths(root: Path, rev: str = "HEAD") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(blocking, warning) path sets as defined by `rev`'s own setup.py.

    Read from the commit, not the working tree (TreeFacts.wheel_*_paths):
    wheel identity is a property of commits; uncommitted changes under the
    path set disable the lookup instead."""
    try:
        source = git(["cat-file", "blob", f"{rev}:setup.py"], cwd=root)
    except Exception:
        source = ""
    return wheel_paths(source)


def object_ids(
    root: Path, commits: list[str], paths: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Per commit, the object id of each path (MISSING if absent), in one
    `git cat-file` call."""
    if not commits:
        return {}
    query = "".join(f"{c}:{p}\n" for c in commits for p in paths)
    out = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname)"],
        cwd=root, input=query, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    oids = [MISSING if line.endswith(" missing") else line for line in out]
    n = len(paths)
    return {c: tuple(oids[i * n:(i + 1) * n]) for i, c in enumerate(commits)}


def identity_of(paths: tuple[str, ...], oids: tuple[str, ...]) -> str:
    lines = [f"{p}={o}" for p, o in zip(paths, oids)]
    return hashlib.sha256("\n".join(["v1", *lines]).encode()).hexdigest()[:16]


def _upstream(root: Path) -> tuple[str | None, str | None]:
    for ref in UPSTREAMS:
        try:
            return ref, git(["merge-base", "HEAD", ref], cwd=root)
        except Exception:
            continue
    return None, None


def _main_window(root: Path, upstream: str, base: str) -> tuple[list[str], int]:
    """First-parent main commits, newest first, from the tip (or SEARCH_AHEAD
    past the merge-base) to SEARCH_BEHIND before it; plus the base's index."""
    ahead = int(git(
        ["rev-list", "--first-parent", "--count", f"{base}..{upstream}"], cwd=root
    ) or 0)
    skip = max(0, ahead - SEARCH_AHEAD)
    commits = git(
        ["rev-list", "--first-parent", f"--skip={skip}",
         f"--max-count={ahead - skip + SEARCH_BEHIND + 1}", upstream],
        cwd=root,
    ).splitlines()
    try:
        return commits, commits.index(base)
    except ValueError:
        return commits, ahead - skip


def find_matches(root: Path, paths: tuple[str, ...] | None = None) -> Lookup:
    """HEAD's identity and every nearby main commit sharing it, nearest to
    the merge-base first (HEAD itself always first)."""
    blocking, warning = compiled_paths(root)
    paths = paths or blocking
    head = git(["rev-parse", "HEAD"], cwd=root)
    upstream, base = _upstream(root)
    window, base_idx = _main_window(root, upstream, base) if base else ([], 0)
    commits = list(dict.fromkeys([head, *window]))
    all_paths = paths + tuple(p for p in warning if p not in paths)
    oids = object_ids(root, commits, all_paths)
    n = len(paths)
    target = identity_of(paths, oids[head][:n])
    look = Lookup(head, paths, warning, target, upstream, base, len(window))
    position = {c: i for i, c in enumerate(window)}

    def candidate(c: str) -> Candidate:
        d = base_idx - position[c] if c in position else None
        stale = [p for p, a, b in zip(all_paths[n:], oids[c][n:], oids[head][n:])
                 if a != b]
        return Candidate(c, d, stale)

    on_main = [c for c in window if identity_of(paths, oids[c][:n]) == target]
    ranked = sorted((candidate(c) for c in on_main if c != head),
                    key=lambda m: (abs(m.distance), m.distance > 0))
    look.matches = [candidate(head), *ranked]
    return look


# --------------------------------------------------------------------------
# wheels.vllm.ai probing + result index
# --------------------------------------------------------------------------


def _variant(platform: str, venv: Path | None = None) -> str | None:
    if platform != "cuda":
        return None
    major = (torch_cuda_major(venv) if venv else None) or cuda_version().split(".")[0]
    return CUDA_VARIANTS.get(major)


def _verify_wheel(path: Path) -> None:
    """A truncated or corrupt download must not be published permanently."""
    with zipfile.ZipFile(path) as zf:
        if bad := zf.testzip():
            raise ValueError(f"corrupt member {bad}")
        if not any(n.endswith(".dist-info/METADATA") for n in zf.namelist()):
            raise ValueError("no .dist-info/METADATA (not a wheel)")


def _fetch_metadata(url: str) -> list:
    with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_S) as r:
        return json.loads(r.read().decode())


def _probe(commit: str, variant: str, arch: str) -> tuple[str, str | None]:
    """("found", url) | ("unpublished", None) | ("error: ...", None).

    Only the variant directory is used: like vLLM, never fall back to the
    root wheel, whose CUDA compatibility cannot be verified."""
    repo_url = f"{WHEELS_BASE}{commit}/{variant}/vllm/"
    try:
        meta = _fetch_metadata(f"{repo_url}metadata.json")
    except urllib.error.HTTPError as e:
        return ("unpublished", None) if e.code == 404 else (f"error: HTTP {e.code}", None)
    except Exception as e:
        return f"error: {e}", None
    for wheel in meta if isinstance(meta, list) else []:
        if wheel.get("package_name") == "vllm" and arch in wheel.get(
            "platform_tag", ""
        ):
            return "found", urljoin(repo_url, wheel["path"])
    return "unpublished", None


def _index_path(cfg: Config) -> Path:
    return cfg.cache_dir / INDEX_NAME


def _load_index(cfg: Config) -> dict:
    try:
        data = json.loads(_index_path(cfg).read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    return {"hits": data.get("hits", {}), "misses": data.get("misses", {})}


def _save_index(cfg: Config, index: dict) -> None:
    now = time.time()
    index["misses"] = {k: t for k, t in index["misses"].items()
                       if now - t < NEGATIVE_TTL_S}
    path = _index_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(index, indent=1, sort_keys=True))


def lookup(cfg: Config, root: Path, *, use_index: bool = True,
           record: bool = True, paths: tuple[str, ...] | None = None) -> Lookup:
    """Resolve a wheel URL for HEAD's compiled sources without downloading.

    The index maps identity+variant+arch to a known wheel (valid forever:
    identical sources) and memoizes unpublished commits for NEGATIVE_TTL_S,
    since publication lags main and those commits gain wheels later."""
    look = find_matches(root, paths)
    look.variant = _variant(cfg.platform or "cuda", root / ".venv")
    look.arch = _platform.machine()
    if look.variant is None:
        look.reason = "no published wheel variant for this platform/CUDA"
        return look
    index = _load_index(cfg) if use_index or record else {"hits": {}, "misses": {}}
    key = f"{look.identity}/{look.variant}/{look.arch}"
    by_commit = {m.commit: m for m in look.matches}
    if use_index and (hit := index["hits"].get(key)):
        look.chosen = by_commit.get(hit["commit"]) or Candidate(
            hit["commit"], relation(root, hit["commit"]))
        look.url, look.source = hit["url"], "index"
        return look
    now = time.time()
    dirty = False
    for m in look.matches[:MAX_PROBES]:
        miss_key = f"{m.commit}/{look.variant}/{look.arch}"
        if use_index and now - index["misses"].get(miss_key, 0) < NEGATIVE_TTL_S:
            look.probes.append((m.commit, "unpublished (memo)"))
            continue
        result, url = _probe(m.commit, look.variant, look.arch)
        look.probes.append((m.commit, result))
        if result == "unpublished":
            index["misses"][miss_key] = now
            dirty = True
        if url:
            look.chosen, look.url, look.source = m, url, "probe"
            index["hits"][key] = {"commit": m.commit, "url": url, "at": now}
            dirty = True
            break
    if record and dirty:
        try:
            _save_index(cfg, index)
        except OSError:
            pass
    if look.url is None:
        main_matches = len(look.matches) - 1
        look.reason = (
            f"no published {look.variant} wheel among {len(look.probes)} probed "
            f"commit(s) with identical compiled sources "
            f"({main_matches} match(es) on {look.upstream or 'main'})"
        )
    return look


def relation(root: Path, commit: str) -> int | None:
    """Signed first-parent distance of `commit` from HEAD's merge-base."""
    _, base = _upstream(root)
    if not base:
        return None
    for lo, hi, sign in ((base, commit, 1), (commit, base, -1)):
        if run(["git", "merge-base", "--is-ancestor", lo, hi], cwd=root,
               check=False).returncode == 0:
            count = git(["rev-list", "--first-parent", "--count", f"{lo}..{hi}"],
                        cwd=root)
            return sign * int(count)
    return None


# --------------------------------------------------------------------------
# Build-layer entry point
# --------------------------------------------------------------------------


def try_fetch_precompiled(cfg: Config, env_root: Path, bhash: str) -> Path | None:
    """Download the upstream wheel for this tree into builds/<bhash>/.

    Returns the wheel path on success, None when unavailable/unsafe (build
    locally). Caller holds the entry lock and has verified the tree is clean
    for the build-layer paths.
    """
    if user_overrides():
        # *_SRC_DIR overrides mean the upstream wheel was not built from
        # this env's inputs — never substitute it
        return None
    paths, _ = compiled_paths(env_root)
    if build_paths_dirty(env_root, paths):
        say(f"precompiled: uncommitted changes under {', '.join(paths)} "
            f"→ HEAD identity untrusted, building locally")
        return None
    look = lookup(cfg, env_root, paths=paths)
    if look.url is None or look.chosen is None:
        say(f"precompiled: {look.reason}; building locally")
        return None
    url, chosen = look.url, look.chosen

    entry = cfg.store("builds") / bhash
    entry.mkdir(parents=True, exist_ok=True)
    name = unquote(re.sub(r"[?#].*$", "", url.rsplit("/", 1)[-1])) or "vllm-precompiled.whl"
    tmp = entry / (name + ".part")
    stale = f"; differs in warning paths {', '.join(chosen.stale)}" if chosen.stale else ""
    say(f"build layer: fetching precompiled wheel from {chosen.commit[:12]} "
        f"({describe(chosen.distance)}, identity {look.identity}{stale})")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as r, \
                open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        _verify_wheel(tmp)
        target = entry / name
        tmp.rename(target)
        target.chmod(0o444)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        action = (
            "will build locally"
            if cfg.allow_local_build
            else "local builds are disabled"
        )
        say(f"precompiled fetch failed ({e}); {action}")
        return None
    write_meta(entry, {
        "kind": "build", "hash": bhash, "wheel": name,
        "origin": f"precompiled:{chosen.commit}",
        "source_identity": look.identity,
        "source_paths": list(look.paths),
        "merge_base": look.merge_base,
        "distance": chosen.distance,
        "variant": look.variant,
    })
    touch_last_used(entry)
    say(f"build layer published (precompiled) → builds/{bhash}")
    return target
