"""Env registry: envs.json (name -> workspace) + per-env .vllm-env.toml marker."""

import json
import time
from pathlib import Path

from .config import MARKER_NAME, Config
from .locks import entry_lock
from .util import atomic_write_text


def _load_registry(cfg: Config) -> dict:
    try:
        return json.loads(cfg.registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_registry(cfg: Config) -> dict:
    with entry_lock(cfg.registry_path):
        return _load_registry(cfg)


def _save_registry(cfg: Config, reg: dict) -> None:
    cfg.registry_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.registry_path, json.dumps(reg, indent=2))


def register_env(cfg: Config, name: str, path: Path, repo: Path) -> None:
    with entry_lock(cfg.registry_path):
        reg = _load_registry(cfg)
        reg[name] = {
            "path": str(path),
            "repo": str(repo),
            "created_at": time.time(),
        }
        _save_registry(cfg, reg)


def unregister_env(cfg: Config, name: str) -> None:
    with entry_lock(cfg.registry_path):
        reg = _load_registry(cfg)
        reg.pop(name, None)
        _save_registry(cfg, reg)


def live_envs(cfg: Config) -> dict[str, Path]:
    """Registered envs whose workspaces still exist (stale entries pruned)."""
    with entry_lock(cfg.registry_path):
        reg = _load_registry(cfg)
        alive = {}
        stale = []
        for name, info in reg.items():
            p = Path(info["path"])
            if (p / MARKER_NAME).exists():
                alive[name] = p
            else:
                stale.append(name)
        if stale:
            for name in stale:
                reg.pop(name, None)
            _save_registry(cfg, reg)
        return alive


# --- per-env marker (minimal toml writer/reader; stdlib has no toml writer) ---


class MarkerError(RuntimeError):
    pass


def write_marker(env_root: Path, data: dict) -> None:
    lines = ["[ve]"]
    for k, v in data.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            # JSON string escapes are valid TOML basic-string escapes
            lines.append(f"{k} = {json.dumps(str(v))}")
    atomic_write_text(env_root / MARKER_NAME, "\n".join(lines) + "\n")


def read_marker(env_root: Path) -> dict:
    """The env marker ({} when absent). A corrupt marker raises rather than
    reading as {}: callers treat {} as "unmanaged" and would replace the
    env's venv."""
    import tomllib

    path = env_root / MARKER_NAME
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    try:
        return tomllib.loads(text).get("ve", {})
    except tomllib.TOMLDecodeError as e:
        raise MarkerError(f"corrupt env marker {path} ({e}); fix or delete it") from e


def update_marker(env_root: Path, **updates) -> None:
    data = read_marker(env_root)
    data.update(updates)
    write_marker(env_root, data)


def referenced_build_hashes(cfg: Config) -> set[str]:
    """Build hashes attached by any live env (GC refcount source)."""
    hashes = set()
    for path in live_envs(cfg).values():
        m = read_marker(path)
        if h := m.get("build_hash"):
            hashes.add(h)
    return hashes


def referenced_deepep_hashes(cfg: Config) -> set[str]:
    """DeepEP layers whose NVSHMEM runtime is used by a live environment."""
    hashes = set()
    for path in live_envs(cfg).values():
        if h := read_marker(path).get("deepep_hash"):
            hashes.add(h)
    return hashes
