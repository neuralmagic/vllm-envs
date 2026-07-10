# vllm-envs (`ve`)

Fast disposable vLLM dev environments from any branch/commit, built on
content-addressed layer caches. Spin up an agent workspace in seconds; hop
commits (including `git bisect`) without re-paying dependency installs or CUDA
builds.

## Install

```bash
uv tool install --editable /path/to/vllm-envs   # or: uv pip install -e .
```

Requires: `uv`, `ccache`, git ≥ 2.15. Reflink-capable filesystem (XFS/btrfs)
recommended; falls back to plain copies otherwise.

## Usage

```bash
cd /path/to/vllm            # your vLLM clone
ve new main                 # env from main
ve new v0.13.0 --name bisect1
cd ~/vllm-envs/bisect1
source .venv/bin/activate
git checkout <sha>          # post-checkout hook auto-syncs layers
git bisect start ...        # works transparently
ve status                   # layer hashes + cache state
ve rm bisect1
ve gc --dry-run             # LRU cache pruning (50GB default cap)
```

## How it works

Layers, each a content-addressed cache under `~/.cache/vllm-envs/`:

| Layer | Store | Key |
|---|---|---|
| 1a torch + build deps | `venvs-base/` | platform-scoped build requirements + torch pins + python/CUDA version |
| 1b full python deps | `venvs/` | 1a key + platform-relevant runtime requirements |
| 2 external sources | `ext-src/` | pins parsed from the worktree's cmake files |
| 3 compiled extensions | `builds/` (wheels) + `cmake-build/` (incremental trees) | working-tree content hash of csrc/ cmake/ CMakeLists.txt setup.py |
| 4 python source | the worktree | n/a |

- Envs get **private reflink-cloned venvs** — edit site-packages freely.
- Extensions attach via each commit's own `VLLM_PRECOMPILED_WHEEL_LOCATION`
  extraction logic (historically correct for old releases).
- Dirty csrc/ or user `*_SRC_DIR` overrides → private builds, never published
  to the shared store.
- ccache is forced (`VLLM_DISABLE_SCCACHE=1`); clean builds use a persistent
  per-hash cmake build tree for incremental rebuilds.

## Config

`~/.cache/vllm-envs/config.toml`:

```toml
[cache]
max_size_gb = 50       # VE_MAX_SIZE_GB overrides
min_age_hours = 72

[core]
envs_root = "~/vllm-envs"   # VE_ENVS_ROOT overrides
python = "3.12"
```

Env vars: `VE_CACHE_DIR`, `VE_NO_SYNC=1` (skip hook sync, warn instead).

## Caveats (v1)

- Requirements top-up on commit hop converges the env venv but does not
  uninstall removed deps; use `ve sync --fresh-venv` for a clean clone.
- The upstream precompiled-wheel fallback (`VLLM_USE_PRECOMPILED`) is used
  when a local build fails; those artifacts are not captured into the store.
- Attach extracts `.so` files into each worktree (vLLM's own vendoring logic)
  rather than symlinking — sharing is at the wheel level.
