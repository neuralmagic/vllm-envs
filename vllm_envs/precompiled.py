"""Fetch upstream precompiled wheels (wheels.vllm.ai) into the build store.

Safe only when the worktree's build inputs (csrc/ cmake/ CMakeLists.txt
setup.py) are byte-identical to the merge-base commit on origin/main that
produced the wheel; the caller guarantees the tree is clean for those paths.
"""

import json
import platform as _platform
import re
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urljoin

from .config import BUILD_LAYER_PATHS, Config
from .hashing import cuda_version
from .log import say
from .store import touch_last_used, write_meta
from .util import git

WHEELS_BASE = "https://wheels.vllm.ai/"
CUDA_VARIANTS = {"12": "cu129", "13": "cu130"}
TIMEOUT_S = 30


def _base_main_commit(root: Path) -> str | None:
    for upstream in ("origin/main", "upstream/main", "main"):
        try:
            return git(["merge-base", "HEAD", upstream], cwd=root)
        except Exception:
            continue
    return None


def _build_inputs_match_base(root: Path, base: str) -> bool:
    try:
        diff = git(
            ["diff", "--name-only", base, "HEAD", "--", *BUILD_LAYER_PATHS],
            cwd=root,
        )
    except Exception:
        return False
    return not diff.strip()


def _variant(platform: str) -> str | None:
    if platform != "cuda":
        return None
    major = cuda_version().split(".")[0]
    return CUDA_VARIANTS.get(major)


def _fetch_json(url: str) -> list | None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _select_wheel_url(commit: str, variant: str | None) -> str | None:
    arch = _platform.machine()
    dirs = [f"{variant}/", ""] if variant else [""]
    for vdir in dirs:
        repo_url = f"{WHEELS_BASE}{commit}/{vdir}"
        meta = _fetch_json(f"{repo_url}vllm/metadata.json")
        if not isinstance(meta, list):
            continue
        for wheel in meta:
            if wheel.get("package_name") == "vllm" and arch in wheel.get(
                "platform_tag", ""
            ):
                return urljoin(repo_url, wheel["path"])
    return None


def try_fetch_precompiled(cfg: Config, env_root: Path, bhash: str) -> Path | None:
    """Download the upstream wheel for this tree into builds/<bhash>/.

    Returns the wheel path on success, None when unavailable/unsafe. Caller
    holds the entry lock and has verified the tree is clean for build paths.
    """
    base = _base_main_commit(env_root)
    if not base or not _build_inputs_match_base(env_root, base):
        return None
    platform = cfg.platform or "cuda"
    url = _select_wheel_url(base, _variant(platform))
    if url is None:
        return None

    entry = cfg.store("builds") / bhash
    entry.mkdir(parents=True, exist_ok=True)
    name = unquote(re.sub(r"[?#].*$", "", url.rsplit("/", 1)[-1])) or "vllm-precompiled.whl"
    tmp = entry / (name + ".part")
    say(f"build layer: fetching precompiled wheel for base {base[:12]} "
        f"({url.rsplit('/', 1)[-1]})")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        target = entry / name
        tmp.rename(target)
        target.chmod(0o444)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        say(f"precompiled fetch failed ({e}); will build locally")
        return None
    write_meta(
        entry,
        {"kind": "build", "hash": bhash, "wheel": name, "origin": f"precompiled:{base}"},
    )
    touch_last_used(entry)
    say(f"build layer published (precompiled) → builds/{bhash}")
    return target
