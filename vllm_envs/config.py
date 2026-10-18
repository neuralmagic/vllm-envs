import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vllm-envs"
DEFAULT_ENVS_ROOT = Path.home() / "vllm-envs"

STORE_NAMES = (
    "cmake-build", "ext-src", "ep-kernels", "venvs", "venvs-base", "builds"
)
# Cross-store eviction priority (first evicted first).
EVICTION_ORDER = (
    "cmake-build", "ext-src", "venvs", "venvs-base", "ep-kernels", "builds"
)

MARKER_NAME = ".vllm-env.toml"
SCRATCH_DIR_NAME = ".ve"


def scratch_dir(env_root: Path, create: bool = False) -> Path:
    """The env's private working directory (<worktree>/.ve)."""
    path = env_root / SCRATCH_DIR_NAME
    if create:
        path.mkdir(exist_ok=True)
    return path


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
    max_size_gb: float = 100.0
    ccache_max_size_gb: float = 100.0  # ve-managed ccache (one vLLM build ≈ 5GB)
    min_age_hours: float = 72.0
    python: str = "3.12"
    platform: str = ""  # auto-detect when empty
    cap: str = "minor"  # cap unpinned requirement floors: minor | major | none
    with_test: bool = True  # install & cache requirements/test/<platform>.txt
    with_vllm_extras: bool = True  # install vLLM's optional runtime bundle
    allow_local_build: bool = True  # build extensions locally when no wheel is available
    reap_max_age_days: float = 30.0  # `ve reap`: worktrees inactive longer are stale
    raw: dict = field(default_factory=dict)

    @property
    def registry_path(self) -> Path:
        return self.cache_dir / "envs.json"

    @property
    def ccache_dir(self) -> Path:
        """ve's compiler cache, unless the user points ccache elsewhere."""
        if v := os.environ.get("CCACHE_DIR"):
            return Path(v)
        return self.cache_dir / "ccache"

    def store(self, name: str) -> Path:
        assert name in STORE_NAMES, name
        return self.cache_dir / name


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE or text in _FALSE:
        return text in _TRUE
    raise ValueError(f"expected true/false, got {value!r}")


def _as_size(value) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"must be >= 0, got {value!r}")
    return number


def _as_path(value) -> Path:
    return Path(str(value)).expanduser()


def _choice(*allowed: str):
    def parse(value) -> str:
        if str(value) not in allowed:
            raise ValueError(f"expected one of {'|'.join(allowed)}, got {value!r}")
        return str(value)
    return parse


# (section, key, Config attribute, parser, env vars; later keys are aliases)
SCHEMA = (
    ("cache", "max_size_gb", "max_size_gb", _as_size, ("VE_MAX_SIZE_GB",)),
    ("cache", "ccache_max_size_gb", "ccache_max_size_gb", _as_size,
     ("VE_CCACHE_MAX_SIZE_GB",)),
    ("cache", "min_age_hours", "min_age_hours", _as_size, ()),
    ("core", "envs_root", "envs_root", _as_path, ("VE_ENVS_ROOT",)),
    ("core", "python", "python", str, ()),
    ("core", "platform", "platform", _choice("", "cuda", "rocm", "cpu", "xpu"), ()),
    ("venv", "cap", "cap", _choice("minor", "major", "none"), ("VE_CAP",)),
    ("venv", "test", "with_test", _as_bool, ("VE_WITH_TEST",)),
    ("venv", "vllm_extras", "with_vllm_extras", _as_bool,
     ("VE_WITH_VLLM_EXTRAS", "VE_WITH_DEEPEP")),
    ("venv", "deepep", "with_vllm_extras", _as_bool, ()),  # legacy alias
    ("venv", "allow_local_build", "allow_local_build", _as_bool,
     ("VE_ALLOW_LOCAL_BUILD",)),
    ("reap", "max_age_days", "reap_max_age_days", _as_size,
     ("VE_REAP_MAX_AGE_DAYS",)),
)


def load_config() -> Config:
    """Defaults < config.toml < environment. Invalid values stop with a
    one-line error naming the key (this also runs inside git hooks)."""
    cfg = Config()
    if cache_dir := os.environ.get("VE_CACHE_DIR"):
        cfg.cache_dir = Path(cache_dir).expanduser()
    cfg_file = cfg.cache_dir / "config.toml"
    data: dict = {}
    if cfg_file.exists():
        try:
            data = tomllib.loads(cfg_file.read_text())
        except tomllib.TOMLDecodeError as e:
            raise SystemExit(f"[ve] config error: {cfg_file}: {e}")
    cfg.raw = data
    known = {(section, key) for section, key, *_ in SCHEMA}
    for section, table in data.items():
        for key in table if isinstance(table, dict) else ():
            if (section, key) not in known:
                print(f"[ve] WARNING: unknown config key [{section}] {key} "
                      f"in {cfg_file}", file=sys.stderr)
    seen: set[str] = set()
    for section, key, attr, parse, _ in SCHEMA:
        value = data.get(section, {}).get(key)
        if value is None or attr in seen:
            continue
        seen.add(attr)
        try:
            setattr(cfg, attr, parse(value))
        except (TypeError, ValueError) as e:
            raise SystemExit(f"[ve] config error: [{section}] {key} in {cfg_file}: {e}")
    for _, _, attr, parse, env_vars in SCHEMA:
        for name in env_vars:
            if value := os.environ.get(name):
                try:
                    setattr(cfg, attr, parse(value))
                except (TypeError, ValueError) as e:
                    raise SystemExit(f"[ve] config error: {name}: {e}")
                break
    return cfg
