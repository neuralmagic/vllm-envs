# vllm-envs (`ve`)

Fast disposable vLLM dev environments from any branch/commit, built on content-addressed layer caches. Spin up an agent workspace in seconds; hop commits (including `git bisect`) without re-paying dependency installs or CUDA builds.

## Install

```bash
uv tool install --editable /path/to/vllm-envs   # or: uv pip install -e .
```

Requires: `uv`, `ccache`, git ≥ 2.15. Reflink-capable filesystem (XFS/btrfs) recommended; falls back to plain copies otherwise.

## Usage

**Manage a clone in place** — build + deps assembled from cache into `./.venv`:

```bash
cd vllm                     # your vLLM clone
ve init
source .venv/bin/activate   # plain venv — always works
```

**New worktrees become envs automatically.** After `ve init` (or `ve new`) has run once in a repo, its worktree hook auto-initializes every new worktree:

```bash
git worktree add -b my-branch ../my-feature
cd ../my-feature && source .venv/bin/activate
```

Omit `-b` only when the branch already exists. If `VE_NO_AUTO_INIT=1`, the hook
skips environment creation and you must run `ve init` in the new worktree:

```bash
cd ../my-feature
ve init
source .venv/bin/activate
```

**Or spawn disposable envs by ref** (created under `~/vllm-envs/`):

```bash
ve new main                 # env from main
ve new v0.13.0 --name bisect1
```

Inside an env you can `git checkout <sha>` / `git bisect` freely — a hook re-syncs the layers on each hop. Optionally add `eval "$(ve shellenv)"` to your shell rc, then `ve activate` sources the venv from anywhere inside an env.

The first few environment setups can be slow while dependency and build caches
populate. Setup gets faster with continued use as more layers become cache hits.

**Everyday commands:**

| Command | What it does |
|---|---|
| `ve status` | layer hashes + cache state for the current env |
| `ve sync` | re-resolve layers for the current worktree HEAD |
| `ve list` | list live envs |
| `ve rm <name>` | remove an env (`ve init` envs are unmanaged, never deleted) |
| `ve reap` | list worktrees; reap stale ones (see below) |
| `ve gc [--dry-run]` | LRU cache pruning (100GB default cap, on physical usage) |
| `ve du` | disk audit: logical vs reflink-aware physical, per store/env |

## Using with t3code

t3code creates a git worktree per session under `~/.t3/worktrees/<repo>/`. Because auto-init fires on `git worktree add`, each session gets a ready `.venv` from cache automatically — just install the hook once in your clone:

```bash
cd ~/local/vllm && ve init   # once; installs the worktree hook
```

Archiving a t3code session leaves its worktree on disk. `ve reap` finds and prunes them:

```bash
ve reap             # list every worktree: state (active/archived/deleted/orphan),
                    #   last activity, dirty count, push status, and reap safety
ve reap --stale     # remove all archived/deleted/orphan worktrees...
                    #   ...but only the clean AND pushed/merged ones
ve reap <name>      # reap one by name or path
```

Reaping is gated: a worktree with uncommitted changes or un-pushed commits is skipped (add `--force` to override, `--dry-run` to preview). Active, primary, and non-t3 worktrees are never touched without `--force`.

Prefix a single command with `VE_NO_AUTO_INIT=1` to skip auto-init. Avoid
exporting it globally; if it is already exported, use `VE_NO_AUTO_INIT=` to
enable auto-init for a worktree command.

## How it works

An env is just a git worktree plus a private `.venv`, assembled from shared content-addressed stores under `~/.cache/vllm-envs/`:

| Store | Contents | Keyed by |
|---|---|---|
| `venvs-base/` | torch + build deps | build requirements + torch pins + python/CUDA version |
| `venvs/` | full deps (derived from `venvs-base`); test deps too when `[venv] test` is on | base key + runtime (+ test) requirements |
| `ep-kernels/` | DeepEP wheel and its NVSHMEM runtime | vLLM installer + Docker pin + venv/CUDA inputs |
| `ext-src/` | pinned external sources (cutlass, flash-attn, ...) | project + pin parsed from the worktree's cmake files |
| `builds/` | compiled-extension wheel + extracted-file mirror + editable-install replay | source + Python/Torch ABI + host/compiler/CUDA/GPU target + build flags |
| `cmake-build/` | persistent cmake trees for incremental local builds | same hash as `builds/` |

```mermaid
flowchart LR
    subgraph store ["~/.cache/vllm-envs (content-addressed, LRU gc)"]
        base["venvs-base/#lt;hash#gt;<br>torch + build deps"]
        full["venvs/#lt;hash#gt;<br>+ runtime deps"]
        builds["builds/#lt;hash#gt;<br>wheel + extracted mirror<br>+ editable replay"]
        ext["ext-src/#lt;project#gt;-#lt;pin#gt;"]
        cmake["cmake-build/#lt;hash#gt;"]
    end
    wheelsvllm["wheels.vllm.ai"]
    subgraph env ["env = git worktree"]
        venv[".venv (private clone)"]
        wt["python source<br>+ extracted .so"]
    end
    base -- "reflink + top-up" --> full
    full -- "venv: reflink clone" --> venv
    builds -- "attach: .so reflinks<br>+ editable install" --> wt
    builds -- "attach: .pth/finder/dist-info" --> venv
    wheelsvllm -. "clean tree on a main commit" .-> builds
    ext -- "*_SRC_DIR" --> cmake
    cmake -. "local ccache build" .-> builds
```

`ve sync` (run by `ve new`/`ve init` and the post-checkout hook) resolves three layers, skipping whatever is already consistent:

1. **venv** — requirements files are hashed; on a hit the env's `.venv` is reflink-cloned from the cached template (CoW: edit site-packages freely, other envs are unaffected). Templates are derived `venvs-base` → `venvs`, so a runtime-requirements change only re-installs the delta. On a commit hop the private venv is converged in place (top-up install + uninstall of dropped deps).
2. **build** — csrc/cmake/setup.py content is hashed together with the effective Python/Torch ABI, host/compiler/CUDA toolchains, GPU target, and build flags. Clean-tree resolution order: store hit → exact-commit or compatible-main precompiled wheel from wheels.vllm.ai (published into the store) → local ccache build (also published), with `ext-src/` pins injected via `*_SRC_DIR` env vars and a persistent per-hash cmake tree for incremental rebuilds. Local builds disable sccache and set `CCACHE_NOHASHDIR=true`, allowing unchanged translation units to hit across content-addressed build directories. Dirty csrc/ or user `*_SRC_DIR` overrides → private builds, never published.
3. **attach** — makes the worktree importable as an editable install: the wheel's `.so` files and bundled third-party py files are placed into the worktree as reflinks of the shared mirror in `builds/<hash>/extracted/` (~450MB physical shared per env), and the editable-install artifacts (`.pth`, finder, dist-info) are written into the venv. The first attach at a build hash runs vLLM's own setup.py once (historically correct extraction for old releases) and captures the result; later attaches at the same commit replay it with no setup.py run, and an already-attached env is a no-op.

Warm-cache timing: fresh `ve new`/`ve init` ~7s, no-op `ve sync` ~1s; a cold build costs one normal vLLM build, then every env at that hash shares it.

## Config

`~/.cache/vllm-envs/config.toml`:

```toml
[cache]
max_size_gb = 100      # VE_MAX_SIZE_GB overrides; enforced on reflink-aware physical usage
min_age_hours = 72

[core]
envs_root = "~/vllm-envs"   # VE_ENVS_ROOT overrides
python = "3.12"

[venv]
cap = "minor"   # cap unpinned requirement floors: minor | major | none (VE_CAP overrides)
test = true     # install & cache requirements/test/<platform>.txt into the full venv
                #   (prefers the pinned .txt, falls back to .in); VE_WITH_TEST overrides
deepep = true   # install DeepEP in every CUDA env; VE_WITH_DEEPEP overrides
```

With `test = true` (the default) the `venvs/` layer also installs vLLM's test
dependencies (`pytest`, `lm-eval`, ...), cached and reflink-shared like the rest;
set `test = false` (or `VE_WITH_TEST=0`) to keep envs lean.

When a CUDA vLLM checkout requires `flashinfer-python`, `ve` also installs the
matching `flashinfer-jit-cache` wheel from FlashInfer's CUDA-specific index.
This derived dependency is required and has no opt-out.

CUDA environments install DeepEP by default using vLLM's
`tools/ep_kernels/install_python_libraries.sh`. The DeepEP commit follows
`docker/versions.json`, falling back to the installer's own default, and the
NVSHMEM version follows the installer default. Set `deepep = false` (or
`VE_WITH_DEEPEP=0`) to opt out. Commits from before that installer was added
are detected and skipped. CUDA architectures include the DeepEP Docker targets
and locally detected GPU targets, unless `TORCH_CUDA_ARCH_LIST` is explicitly
set.

Env vars: `VE_CACHE_DIR`, `VE_NO_SYNC=1` (skip hook sync, warn instead).

## Tests

The default suite is fast and replaces only external builds and installs:

```bash
python -m unittest discover -s tests -v
```

An opt-in E2E mode runs real `uv` installs, vLLM/CUDA compilation, DeepEP
build and import, Git hooks, worktree creation, a committed build-input edit,
and checkout back to the original cached build:

```bash
VE_REAL_E2E=1 \
VE_REAL_E2E_REPO=~/local/vllm \
VE_REAL_E2E_CACHE_DIR=~/.cache/vllm-envs-e2e \
python -m unittest -v tests.test_real_e2e
```

`VE_REAL_E2E_CACHE_DIR` is optional. Without it the test uses and removes an
isolated cold cache; setting it keeps the expensive layers for subsequent
cache-hit runs. Select another commit with `VE_REAL_E2E_REF`; set
`VE_REAL_E2E_WITH_TEST=0` to omit vLLM's test requirements. Real E2E requires
an NVIDIA host and is skipped unless explicitly enabled.

## Caveats (v1)

- Requirements top-up on commit hop converges the env venv (installs the delta, uninstalls deps dropped from the template resolution); user-added packages survive, but for a guaranteed-clean venv use `ve sync --fresh-venv`.
- The upstream precompiled-wheel fallback (`VLLM_USE_PRECOMPILED`) is used when a local build fails; those artifacts are not captured into the store.
- Unpinned requirement floors (e.g. `transformers>=4.5`) are capped near their era at resolve time: by default each floored-but-uncapped requirement gets an upper bound at the next minor (`>=4.5` → `<4.6`); `cap = "major"` bounds at the next major, `"none"` restores resolve-to-latest. When the caps are mutually unsatisfiable (stale floors), ve automatically falls back minor → major → none with a warning. The cap mode folds into the venv template key.
