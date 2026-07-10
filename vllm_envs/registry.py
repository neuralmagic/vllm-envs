"""Env registry: envs.json (name -> workspace) + per-env .vllm-env.toml marker."""

import json
import time
from pathlib import Path

from .config import MARKER_NAME, Config


def load_registry(cfg: Config) -> dict:
    try:
        return json.loads(cfg.registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_registry(cfg: Config, reg: dict) -> None:
    cfg.registry_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.registry_path.write_text(json.dumps(reg, indent=2))


def register_env(cfg: Config, name: str, path: Path, repo: Path) -> None:
    reg = load_registry(cfg)
    reg[name] = {"path": str(path), "repo": str(repo), "created_at": time.time()}
    save_registry(cfg, reg)


def unregister_env(cfg: Config, name: str) -> None:
    reg = load_registry(cfg)
    reg.pop(name, None)
    save_registry(cfg, reg)


def live_envs(cfg: Config) -> dict[str, Path]:
    """Registered envs whose workspaces still exist (stale entries pruned)."""
    reg = load_registry(cfg)
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
        save_registry(cfg, reg)
    return alive


# --- per-env marker (minimal toml writer/reader; stdlib has no toml writer) ---


def write_marker(env_root: Path, data: dict) -> None:
    lines = ["[ve]"]
    for k, v in data.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            lines.append(f'{k} = "{v}"')
    (env_root / MARKER_NAME).write_text("\n".join(lines) + "\n")


def read_marker(env_root: Path) -> dict:
    import tomllib

    try:
        return tomllib.loads((env_root / MARKER_NAME).read_text()).get("ve", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


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
