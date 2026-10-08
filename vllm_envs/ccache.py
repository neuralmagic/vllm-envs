"""ccache policy for ve-driven local builds.

Cross-hash build reuse rides on ccache (content-addressed build directories +
CCACHE_NOHASHDIR), which leaves two silent failure modes:

* ccache's built-in 5 GiB cap evicts a vLLM-scale cache after the first full
  build, turning later branch hops into apparent "no caching" rebuilds. ve
  raises the cap to a vLLM-appropriate floor whenever ccache still reports
  the built-in default; any explicit user configuration is left untouched.
* hit rates were invisible. Every ve build now reports them, so a cold or
  evicted cache is diagnosable from the build output alone.
"""

import json
import os
import re
from shutil import which

from .config import Config
from .log import say
from .util import human_size, run

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgt]?)i?b?\s*$", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
# ccache's documented built-in cap. Any other effective value means the user
# configured ccache and ve must not stomp their choice.
_BUILTIN_DEFAULT_BYTES = 5 * 1024**3


def parse_size(text: str) -> int | None:
    """Bytes in a ccache size string ('5.0 GiB', '50G', '1.5M'), else None."""
    match = _SIZE_RE.match(text or "")
    if match is None:
        return None
    try:
        return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()])
    except ValueError:
        return None


def max_size_env(cfg: Config) -> dict[str, str]:
    """CCACHE_MAXSIZE for ve builds, or {} when the user owns the setting.

    ve only raises the cap when ccache still reports its built-in default; a
    CCACHE_MAXSIZE in the environment or any non-default max_size in ccache's
    own configuration always wins.
    """
    if which("ccache") is None or os.environ.get("CCACHE_MAXSIZE"):
        return {}
    proc = run(["ccache", "--get-config", "max_size"], check=False)
    if proc.returncode == 0:
        current = parse_size(proc.stdout)
        if current is None or current != _BUILTIN_DEFAULT_BYTES:
            return {}  # configured (or unlimited): not ve's to change
    # ccache would not say (e.g. 3.x without --get-config): assume the
    # built-in default rather than keep the silent-eviction failure mode.
    floor = f"{cfg.ccache_max_size_gb:g}G"
    say(
        f"ccache cap at built-in default → ve builds use {floor} "
        "(cache.ccache_max_size_gb adjusts; "
        "export CCACHE_MAXSIZE to take over)"
    )
    return {"CCACHE_MAXSIZE": floor}


def stats(env: dict[str, str] | None = None) -> dict | None:
    """Snapshot ccache's counters in the environment ve builds in."""
    if which("ccache") is None:
        return None
    proc = run(
        ["ccache", "--print-stats", "--format=json"], env=env, check=False
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def summarize(before: dict | None, after: dict | None) -> str | None:
    """One-line hit-rate report for the compilations between two snapshots.

    Counters are global to the cache dir, so builds running concurrently in
    other envs can bleed into the deltas; close enough for a diagnostic line.
    """
    if not before or not after:
        return None

    def delta(key: str) -> int:
        try:
            return int(after.get(key, 0)) - int(before.get(key, 0))
        except (TypeError, ValueError):
            return 0

    hits = delta("direct_cache_hit") + delta("preprocessed_cache_hit")
    misses = delta("cache_miss")
    if hits + misses == 0:
        return "ccache: no cacheable compilations"
    rate = 100 * hits / (hits + misses)
    line = f"ccache: {hits} hits / {misses} misses ({rate:.0f}%)"
    if misses > 0 and rate < 25:
        line += " — cache cold or evicted (ccache -s to inspect)"
    size_before = before.get("cache_size_kibibyte")
    size_after = after.get("cache_size_kibibyte")
    if size_before is not None and size_after is not None:
        try:
            if int(size_after) != int(size_before):
                line += (
                    f", cache {human_size(int(size_before) * 1024)} → "
                    f"{human_size(int(size_after) * 1024)}"
                )
        except (TypeError, ValueError):
            pass
    return line
