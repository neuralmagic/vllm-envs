# vllm-envs (`ve`)

Fast disposable vLLM dev environments from any branch/commit, built on content-addressed layer caches. Spin up an agent workspace in seconds; hop commits (including `git bisect`) without re-paying dependency installs or CUDA builds.

## Install

```bash
uv tool install --editable /path/to/vllm-envs   # or: uv pip install -e .
```

Requires: `uv`, `ccache`, git ≥ 2.15. Reflink-capable filesystem (XFS/btrfs) recommended; falls back to plain copies otherwise.

## Usage

Worktree-native workflow — manage your clone in place, and every new worktree becomes an env automatically:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
ve init                     # manage this checkout in place: venv + extensions from cache
ve activate                 # subshell with the venv active (exit to leave)
eval "$(ve activate)"       # ...or source it into the current shell
git worktree add ../my-feature my-branch   # auto-runs ve init in the new worktree
cd ../my-feature && ve activate
```

Or spawn disposable envs by ref:

```bash
cd /path/to/vllm            # your vLLM clone
ve new main                 # env from main
ve new v0.13.0 --name bisect1
cd ~/vllm-envs/bisect1
ve activate
git checkout <sha>          # post-checkout hook auto-syncs layers
git bisect start ...        # works transparently
ve status                   # layer hashes + cache state
ve rm bisect1               # for `ve init` envs: unmanages, never deletes your checkout
ve gc --dry-run             # LRU cache pruning (50GB default cap)
```

Auto-init on `git worktree add` fires only for worktrees of a repo where the hook is installed (`ve init` or `ve new` installs it); set `VE_NO_AUTO_INIT=1` to skip it for one command.

## How it works

Caches, each content-addressed under `~/.cache/vllm-envs/`:

| Cache | Store | Key |
|---|---|---|
| 1a torch + build deps | `venvs-base/` | platform-scoped build requirements + torch pins + python/CUDA version |
| 1b full python deps | `venvs/` | 1a key + platform-relevant runtime requirements |
| 2 external sources | `ext-src/` | pins parsed from the worktree's cmake files |
| 3 compiled extensions | `builds/` (wheels + extracted mirror) + `cmake-build/` (incremental trees) | working-tree content hash of csrc/ cmake/ CMakeLists.txt setup.py |
| 4 python source | the worktree | n/a |

Only 1a→1b is a derivation chain (1b templates are reflink-cloned from 1a and topped up). 1b, 2, and 3 are independent, parallel caches: their keys don't reference each other, and layer 2 is a pure input cache for layer-3 builds whose pins are already covered by the layer-3 key (the cmake files are hashed). By change frequency on main: csrc/cmake (layer 3) changes many times a day, requirements (1b) roughly weekly, external pins (2) roughly monthly — so layer 3 is resolved via the precompiled-wheel fast path whenever the tree matches a main commit.

- Envs get **private reflink-cloned venvs** — edit site-packages freely.
- Extensions attach via each commit's own `VLLM_PRECOMPILED_WHEEL_LOCATION` extraction logic (historically correct for old releases). After extraction, the `.so` files are rewritten as **reflinks of a shared mirror** in `builds/<hash>/extracted/`, so N envs at the same hash share one physical copy (~450MB saved per env). CoW means a private rebuild or store eviction never affects other envs.
- Layer-3 resolution order for clean trees: store hit → precompiled wheel fetched from wheels.vllm.ai (when build inputs match the origin/main merge-base; published into the store) → local ccache build.
- Dirty csrc/ or user `*_SRC_DIR` overrides → private builds, never published to the shared store.
- ccache is forced (`VLLM_DISABLE_SCCACHE=1`); clean builds use a persistent per-hash cmake build tree for incremental rebuilds.

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

- Requirements top-up on commit hop converges the env venv but does not uninstall removed deps; use `ve sync --fresh-venv` for a clean clone.
- The upstream precompiled-wheel fallback (`VLLM_USE_PRECOMPILED`) is used when a local build fails; those artifacts are not captured into the store.
- Old refs resolve historically-correct *pins*, but unpinned transitive deps (e.g. transformers ranges) resolve to current versions, which can break serving on old releases — that's a vLLM requirements property, not an env-tool one.
