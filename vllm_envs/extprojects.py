"""Pinned external-project sources (cutlass, flash-attn, ...) shared read-only.

Pins are parsed from the *worktree's* cmake files so every commit resolves its
own historically-correct versions. Parsed pins are cloned once into
ext-src/<project>-<pin>/ and injected into builds via the *_SRC_DIR env vars,
which avoids FetchContent re-clones and .deps build-dir races across
concurrent envs. Projects whose pins can't be parsed are simply left to
FetchContent (using a per-env FETCHCONTENT_BASE_DIR).
"""

import os
import json
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import EXT_PROJECT_ENV_VARS, Config
from .locks import entry_lock
from .log import say, warn
from .store import touch_last_used, write_meta
from .util import run

SET_RE = re.compile(
    r'set\s*\(\s*([A-Za-z0-9_]+)\s+"?([^")\s]+)"?(?:\s+CACHE\b[^)]*)?\s*\)', re.I
)
COMMENT_RE = re.compile(r"#[^\n]*")
DECLARE_RE = re.compile(
    r"FetchContent_(?:Declare|Populate)\s*\(\s*([A-Za-z0-9_\-]+)(.*?)\)", re.I | re.S
)
FIELD_RE = re.compile(
    r'(GIT_REPOSITORY|GIT_TAG|SOURCE_SUBDIR)\s+"?([^"\n]+?)"?\s*$', re.M
)

# Git exports these to hooks so nested Git commands operate on the checkout
# that triggered the hook. External-source repositories must not inherit them.
FOREIGN_GIT_ENV = {
    name: None
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    )
}


@dataclass
class ExtPin:
    name: str
    repo: str
    tag: str
    submodules: bool
    # SOURCE_SUBDIR from the declare: the *_SRC_DIR env var must point at the
    # same subdir the unpinned FetchContent path would use (e.g. triton_kernels
    # lives under python/triton_kernels/triton_kernels of the triton repo)
    subdir: str = ""


def _resolve(value: str, variables: dict[str, set[str]]) -> str | None:
    """Substitute ${VAR}s; None when a variable is undefined or set to
    different values (e.g. in if/else branches): we cannot know which one
    CMake will use, so leave that project to vLLM's own FetchContent."""
    unknown = False

    def repl(m: re.Match) -> str:
        nonlocal unknown
        values = variables.get(m.group(1), set())
        if len(values) != 1:
            unknown = True
            return ""
        return next(iter(values))

    resolved = re.sub(r"\$\{([A-Za-z0-9_]+)\}", repl, value).strip()
    return None if unknown or not resolved else resolved


_BRANCH_RE = re.compile(r"\b(if|elseif|else|endif)\s*\(([^)]*)\)", re.I)
_PLATFORM_COND_RE = re.compile(
    r'^\s*(VLLM_TARGET_DEVICE|VLLM_GPU_LANG)\s+STREQUAL\s+"?(\w+)"?\s*$', re.I
)
_GPU_LANG = {"cuda": "CUDA", "rocm": "HIP"}


def _platform_cond(cond: str, platform: str) -> bool | None:
    """Truth of a platform test (the common way vLLM's cmake picks per-device
    pins); None for anything else."""
    m = _PLATFORM_COND_RE.match(cond)
    if not m:
        return None
    actual = platform if m.group(1).upper() == "VLLM_TARGET_DEVICE" else _GPU_LANG.get(platform)
    return (actual or "").lower() == m.group(2).lower()


def _possible_sets(text: str, platform: str) -> dict[str, set[str]]:
    """Values each variable may hold, dropping set()s inside branches that are
    certainly not taken on this platform."""
    events = sorted(
        [(m.start(), "branch", m) for m in _BRANCH_RE.finditer(text)]
        + [(m.start(), "set", m) for m in SET_RE.finditer(text)],
        key=lambda e: e[0],
    )
    # per open if(): [truth of current branch, any earlier branch certainly taken]
    stack: list[list] = []
    out: dict[str, set[str]] = {}
    for _, kind, m in events:
        if kind == "branch":
            word, cond = m.group(1).lower(), m.group(2)
            if word == "if":
                stack.append([_platform_cond(cond, platform), False])
            elif word in ("elseif", "else") and stack:
                frame = stack[-1]
                frame[1] = frame[1] or frame[0] is True
                frame[0] = False if frame[1] else (
                    _platform_cond(cond, platform) if word == "elseif" else
                    (None if frame[0] is None else True))
            elif word == "endif" and stack:
                stack.pop()
            continue
        if any(frame[0] is False for frame in stack):
            continue
        out.setdefault(m.group(1), set()).add(m.group(2))
    return out


def parse_pins(root: Path, platform: str = "cuda") -> dict[str, ExtPin]:
    """Parse external project pins from the worktree's cmake files."""
    files = [root / "CMakeLists.txt"]
    ext_dir = root / "cmake" / "external_projects"
    if ext_dir.is_dir():
        files += sorted(ext_dir.glob("*.cmake"))

    pins: dict[str, ExtPin] = {}
    for f in files:
        if not f.is_file():
            continue
        text = COMMENT_RE.sub("", f.read_text(errors="replace"))
        variables = _possible_sets(text, platform)
        for m in DECLARE_RE.finditer(text):
            name, body = m.group(1).lower(), m.group(2)
            fields = {k: v for k, v in FIELD_RE.findall(body)}
            repo = fields.get("GIT_REPOSITORY")
            tag = fields.get("GIT_TAG")
            if not repo or not tag:
                continue
            repo = _resolve(repo, variables)
            tag = _resolve(tag, variables)
            if not repo or not tag or "$" in (repo + tag):
                continue
            pins[name] = ExtPin(
                name=name,
                repo=repo,
                tag=tag,
                submodules="GIT_SUBMODULES" in body.upper(),
                subdir=_resolve(fields.get("SOURCE_SUBDIR", ""), variables) or "",
            )
    return pins


def _entry_dir(cfg: Config, pin: ExtPin) -> Path:
    safe_tag = re.sub(r"[^A-Za-z0-9._\-]", "_", pin.tag)[:48]
    return cfg.store("ext-src") / f"{pin.name}-{safe_tag}"


def _github_full_ref(repo: str, ref: str) -> str | None:
    """Expand a short GitHub SHA, including commits no longer on a branch."""
    if not re.fullmatch(r"[0-9a-fA-F]{7,39}", ref):
        return None
    parsed = urllib.parse.urlparse(repo.removesuffix(".git"))
    if parsed.hostname != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        return None
    url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}/commits/{ref}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "vllm-envs",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            sha = json.load(response).get("sha", "")
    except (OSError, ValueError):
        sha = ""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha) and shutil.which("gh"):
        response = run(
            [
                "gh",
                "api",
                f"repos/{parts[0]}/{parts[1]}/commits/{ref}",
                "--jq",
                ".sha",
            ],
            check=False,
        )
        sha = response.stdout.strip() if response.returncode == 0 else ""
    return sha if re.fullmatch(r"[0-9a-fA-F]{40}", sha) else None


def _fetch_short_ref_from_default_branch(dest: Path, ref: str) -> bool:
    """Resolve a short commit from bounded default-branch history.

    Git servers generally do not advertise short object IDs as fetchable refs.
    Fetching a small amount of default-branch history handles the common case
    without relying on a hosting-provider API or cloning the full repository.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{7,39}", ref):
        return False
    for depth in (64, 256, 1024):
        fetch = run(
            ["git", "fetch", f"--depth={depth}", "origin", "HEAD"],
            cwd=dest,
            env=FOREIGN_GIT_ENV,
            check=False,
        )
        if fetch.returncode != 0:
            return False
        resolved = run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=dest,
            env=FOREIGN_GIT_ENV,
            check=False,
        )
        if resolved.returncode == 0:
            return True
    return False


def fetch_ref(repo: str, ref: str, dest: Path, label: str) -> None:
    """Create a checkout by fetching only the requested tag/commit."""
    if dest.exists():
        run(["rm", "-rf", str(dest)])
    dest.mkdir(parents=True)
    run(["git", "init", "--quiet"], cwd=dest, env=FOREIGN_GIT_ENV)
    run(
        ["git", "remote", "add", "origin", repo],
        cwd=dest,
        env=FOREIGN_GIT_ENV,
    )
    fetch = run(
        ["git", "fetch", "--depth=1", "origin", ref],
        cwd=dest,
        env=FOREIGN_GIT_ENV,
        check=False,
    )
    checkout_ref = "FETCH_HEAD"
    if fetch.returncode != 0:
        if _fetch_short_ref_from_default_branch(dest, ref):
            checkout_ref = ref
        else:
            expanded = _github_full_ref(repo, ref)
            if expanded is None:
                raise RuntimeError(f"cannot fetch {repo} at {ref}")
            run(
                ["git", "fetch", "--depth=1", "origin", expanded],
                cwd=dest,
                env=FOREIGN_GIT_ENV,
                stream_prefix=f"[ve]   [{label}] ",
            )
    run(
        ["git", "checkout", "--quiet", "--detach", checkout_ref],
        cwd=dest,
        env=FOREIGN_GIT_ENV,
    )
    _ensure_submodules(dest, label)


def _ensure_submodules(dest: Path, label: str) -> None:
    if (dest / ".gitmodules").is_file():
        status = run(
            ["git", "submodule", "status", "--recursive"],
            cwd=dest,
            env=FOREIGN_GIT_ENV,
            check=False,
        )
        if status.returncode != 0 or any(
            line.startswith(("-", "+")) for line in status.stdout.splitlines()
        ):
            run(
                [
                    "git", "submodule", "update", "--init", "--recursive",
                    "--depth=1",
                ],
                cwd=dest,
                env=FOREIGN_GIT_ENV,
                stream_prefix=f"[ve]   [{label}] ",
            )


def ensure_ext_src(cfg: Config, pin: ExtPin) -> Path:
    entry = _entry_dir(cfg, pin)
    with entry_lock(entry):
        if entry.is_dir() and (entry / ".complete").exists():
            _ensure_submodules(entry, pin.name)
            touch_last_used(entry)
            return entry
        say(f"ext-src: cloning {pin.name} @ {pin.tag}")
        fetch_ref(pin.repo, pin.tag, entry, pin.name)
        (entry / ".complete").touch()
        write_meta(entry, {"project": pin.name, "tag": pin.tag, "repo": pin.repo})
        touch_last_used(entry)
    return entry


def user_overrides() -> dict[str, str]:
    """*_SRC_DIR env vars the user set themselves (unpinned local inputs)."""
    return {
        var: os.environ[var]
        for var in EXT_PROJECT_ENV_VARS.values()
        if os.environ.get(var)
    }


def src_dir_env(cfg: Config, root: Path) -> dict[str, str]:
    """Env vars mapping parsed pins to shared read-only source checkouts."""
    env: dict[str, str] = {}
    pins = parse_pins(root, cfg.platform or "cuda")
    for name, var in EXT_PROJECT_ENV_VARS.items():
        if os.environ.get(var):
            continue  # user override wins (handled as unpinned elsewhere)
        # cmake declare names vary slightly (e.g. vllm-flash-attn)
        pin = pins.get(name) or pins.get(name.replace("-", "_"))
        if pin is None:
            continue
        try:
            src = ensure_ext_src(cfg, pin)
            if pin.subdir:
                src = src / pin.subdir
                if not src.is_dir():
                    warn(f"ext-src {name}: subdir {pin.subdir} missing in "
                         f"{pin.tag}; falling back to FetchContent")
                    continue
            env[var] = str(src)
        except Exception as e:  # clone failure → let FetchContent handle it
            warn(f"ext-src {name} unavailable ({e}); falling back to FetchContent")
    return env
