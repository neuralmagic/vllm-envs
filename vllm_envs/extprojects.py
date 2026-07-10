"""Pinned external-project sources (cutlass, flash-attn, ...) shared read-only.

Pins are parsed from the *worktree's* cmake files so every commit resolves its
own historically-correct versions. Parsed pins are cloned once into
ext-src/<project>-<pin>/ and injected into builds via the *_SRC_DIR env vars,
which avoids FetchContent re-clones and .deps build-dir races across
concurrent envs. Projects whose pins can't be parsed are simply left to
FetchContent (using a per-env FETCHCONTENT_BASE_DIR).
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import EXT_PROJECT_ENV_VARS, Config
from .locks import entry_lock
from .log import say, warn
from .store import touch_last_used, write_meta
from .util import run

SET_RE = re.compile(r'set\s*\(\s*([A-Za-z0-9_]+)\s+"?([^")\s]+)"?\s*\)', re.I)
COMMENT_RE = re.compile(r"#[^\n]*")
DECLARE_RE = re.compile(
    r"FetchContent_(?:Declare|Populate)\s*\(\s*([A-Za-z0-9_\-]+)(.*?)\)", re.I | re.S
)
FIELD_RE = re.compile(
    r'(GIT_REPOSITORY|GIT_TAG|SOURCE_SUBDIR)\s+"?([^"\n]+?)"?\s*$', re.M
)


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


def _resolve(value: str, variables: dict[str, str]) -> str | None:
    def repl(m: re.Match) -> str:
        return variables.get(m.group(1), "")

    resolved = re.sub(r"\$\{([A-Za-z0-9_]+)\}", repl, value).strip()
    return resolved or None


def parse_pins(root: Path) -> dict[str, ExtPin]:
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
        variables = {m.group(1): m.group(2) for m in SET_RE.finditer(text)}
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


def ensure_ext_src(cfg: Config, pin: ExtPin) -> Path:
    entry = _entry_dir(cfg, pin)
    with entry_lock(entry):
        if entry.is_dir() and (entry / ".complete").exists():
            touch_last_used(entry)
            return entry
        say(f"ext-src: cloning {pin.name} @ {pin.tag}")
        if entry.exists():
            run(["rm", "-rf", str(entry)])
        clone = ["git", "clone", "--filter=blob:none", pin.repo, str(entry)]
        run(clone, stream_prefix=f"[ve]   [{pin.name}] ")
        run(["git", "checkout", "--quiet", pin.tag], cwd=entry)
        if pin.submodules:
            run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=entry,
                stream_prefix=f"[ve]   [{pin.name}] ",
            )
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
    pins = parse_pins(root)
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
