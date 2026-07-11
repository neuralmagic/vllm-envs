import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vllm-envs"
DEFAULT_ENVS_ROOT = Path.home() / "vllm-envs"

STORE_NAMES = ("cmake-build", "ext-src", "venvs", "venvs-base", "builds")
# Cross-store eviction priority (first evicted first).
EVICTION_ORDER = ("cmake-build", "ext-src", "venvs", "venvs-base", "builds")

MARKER_NAME = ".vllm-env.toml"
SCRATCH_DIR_NAME = ".ve"

# Paths (relative to repo root) whose changes invalidate the build layer.
BUILD_LAYER_PATHS = ("csrc", "cmake", "CMakeLists.txt", "setup.py")

EXT_PROJECT_ENV_VARS = {
    "cutlass": "VLLM_CUTLASS_SRC_DIR",
    "vllm-flash-attn": "VLLM_FLASH_ATTN_SRC_DIR",
    "flashmla": "FLASH_MLA_SRC_DIR",
    "deepgemm": "DEEPGEMM_SRC_DIR",
    "qutlass": "QUTLASS_SRC_DIR",
    "fmha_sm100": "FMHA_SM100_SRC_DIR",
    "triton_kernels": "TRITON_KERNELS_SRC_DIR",
}


@dataclass
class Config:
    cache_dir: Path = DEFAULT_CACHE_DIR
    envs_root: Path = DEFAULT_ENVS_ROOT
    max_size_gb: float = 50.0
    min_age_hours: float = 72.0
    python: str = "3.12"
    platform: str = ""  # auto-detect when empty
    cap: str = "minor"  # cap unpinned requirement floors: minor | major | none
    raw: dict = field(default_factory=dict)

    @property
    def registry_path(self) -> Path:
        return self.cache_dir / "envs.json"

    def store(self, name: str) -> Path:
        assert name in STORE_NAMES, name
        return self.cache_dir / name


def load_config() -> Config:
    cfg = Config()
    cache_dir = os.environ.get("VE_CACHE_DIR")
    if cache_dir:
        cfg.cache_dir = Path(cache_dir)
    cfg_file = cfg.cache_dir / "config.toml"
    if cfg_file.exists():
        data = tomllib.loads(cfg_file.read_text())
        cfg.raw = data
        cache = data.get("cache", {})
        cfg.max_size_gb = float(cache.get("max_size_gb", cfg.max_size_gb))
        cfg.min_age_hours = float(cache.get("min_age_hours", cfg.min_age_hours))
        core = data.get("core", {})
        cfg.envs_root = Path(core.get("envs_root", cfg.envs_root)).expanduser()
        cfg.python = str(core.get("python", cfg.python))
        cfg.platform = str(core.get("platform", cfg.platform))
        cfg.cap = str(data.get("venv", {}).get("cap", cfg.cap))
    if v := os.environ.get("VE_MAX_SIZE_GB"):
        cfg.max_size_gb = float(v)
    if v := os.environ.get("VE_ENVS_ROOT"):
        cfg.envs_root = Path(v)
    if v := os.environ.get("VE_CAP"):
        cfg.cap = v
    if cfg.cap not in ("minor", "major", "none"):
        raise SystemExit(f"[ve] invalid cap mode {cfg.cap!r} (minor|major|none)")
    return cfg
