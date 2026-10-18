"""Declarative vLLM commit/version-specific behaviour: one table of rules.

A Rule reads "trees where every `when` predicate holds need the `then`
effects". Predicates probe content (files, requirement declarations, named
TreeFacts), never commit ranges, except the AncestorOf escape hatch. Rules
sharing a `slot` are alternatives: the highest-priority (then first-listed)
one that holds wins.

A rule that fires and has a keyed effect contributes `rule:<id>@<rev>` to the
cache key of that effect's layer, so bumping `rev` invalidates exactly what the
rule affects. Effects whose output is already hashed (selected requirement
files, hashed paths, untracked-file filters) and effects that only document a
reactive fallback in code are unkeyed.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from .reqs import (
    at_least, canonical_name, declared_floors, requirement_names, version_tuple,
)

if TYPE_CHECKING:
    from .treefacts import TreeFacts


# --------------------------------------------------------------------------
# Evaluation context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What rules are evaluated against: a tree (optional), the host, and the
    requirement files a consumer is about to hash or install."""

    platform: str
    machine: str
    root: Path | None = None
    facts: TreeFacts | None = None
    files: tuple[Path, ...] = ()
    _memo: dict = field(default_factory=dict, compare=False, repr=False)

    def path(self, template: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / template.format(platform=self.platform)

    def names(self) -> set[str]:
        if "names" not in self._memo:
            self._memo["names"] = requirement_names(self.files)
        return self._memo["names"]

    def floors(self) -> dict[str, tuple[int, ...]]:
        if "floors" not in self._memo:
            self._memo["floors"] = declared_floors(self.files)
        return self._memo["floors"]

    def text(self) -> str:
        if "text" not in self._memo:
            self._memo["text"] = "\n".join(
                f.read_text(errors="replace") for f in self.files
            )
        return self._memo["text"]


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------


class Pred:
    def holds(self, ctx: Context) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Exists(Pred):
    path: str  # relative to the tree; may use {platform}

    def holds(self, ctx: Context) -> bool:
        p = ctx.path(self.path)
        return p is not None and p.exists()


@dataclass(frozen=True)
class Absent(Pred):
    path: str

    def holds(self, ctx: Context) -> bool:
        p = ctx.path(self.path)
        return p is not None and not p.exists()


@dataclass(frozen=True)
class Req(Pred):
    """A package declared by the context's requirement files. `spec` is
    empty (declared at all), ">=X" (declared minimum at least X) or "<X"
    (declared minimum below X)."""

    name: str
    spec: str = ""

    def holds(self, ctx: Context) -> bool:
        name = canonical_name(self.name)
        if not self.spec:
            return name in ctx.names()
        m = re.fullmatch(r"(>=|<)\s*(\d+(?:\.\d+)*)", self.spec)
        if m is None:
            raise ValueError(f"unsupported requirement predicate: {self}")
        floor = ctx.floors().get(name)
        if floor is None:
            return False
        meets = at_least(floor, version_tuple(m.group(2)))
        return meets if m.group(1) == ">=" else not meets


@dataclass(frozen=True)
class Platform(Pred):
    name: str  # cuda | rocm | cpu

    def holds(self, ctx: Context) -> bool:
        return ctx.platform == self.name


@dataclass(frozen=True)
class Arch(Pred):
    name: str  # platform.machine(), e.g. x86_64

    def holds(self, ctx: Context) -> bool:
        return ctx.machine == self.name


_TRUTHY = object()


@dataclass(frozen=True)
class Fact(Pred):
    """A named TreeFacts value: truthy by default, else equal to `value`."""

    name: str
    value: Any = _TRUTHY

    def holds(self, ctx: Context) -> bool:
        if ctx.facts is None:
            return False
        actual = getattr(ctx.facts, self.name)
        return bool(actual) if self.value is _TRUTHY else actual == self.value


@dataclass(frozen=True)
class AncestorOf(Pred):
    """Escape hatch for behaviour with no content signal: HEAD contains
    `commit`. Prefer a content predicate whenever one exists."""

    commit: str

    def holds(self, ctx: Context) -> bool:
        if ctx.root is None:
            return False
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", self.commit, "HEAD"],
            cwd=ctx.root, capture_output=True,
        )
        return proc.returncode == 0


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------


class Effect:
    keyed: ClassVar[bool] = False
    layer: str | None = None

    def matches(self, ctx: Context) -> bool:
        """Guard: an effect with nothing to act on keeps its rule from firing."""
        return True


@dataclass(frozen=True)
class Select(Effect):
    """Choose the existing files among `paths` for the rule's slot."""

    paths: tuple[str, ...]

    def existing(self, ctx: Context) -> list[Path]:
        found = (ctx.path(p) for p in self.paths)
        return [p for p in found if p is not None and p.is_file()]

    def matches(self, ctx: Context) -> bool:
        return bool(self.existing(ctx))


@dataclass(frozen=True)
class AddReq(Effect):
    spec: str
    layer: str = "venv"
    keyed: ClassVar[bool] = True


@dataclass(frozen=True)
class OverrideReq(Effect):
    """A resolver override (uv --overrides / UV_OVERRIDE). `spec` may use
    {placeholders} the consumer fills in."""

    spec: str
    layer: str = "venv"
    keyed: ClassVar[bool] = True


@dataclass(frozen=True)
class ExtraIndex(Effect):
    """An extra index per distinct `pattern` capture in the requirement files."""

    pattern: str
    url: str  # formatted with the capture as {0}
    strategy: str = ""
    layer: str = "venv"
    keyed: ClassVar[bool] = True

    def urls(self, ctx: Context) -> list[str]:
        captures = sorted(set(re.findall(self.pattern, ctx.text())))
        return [self.url.format(c) for c in captures]

    def matches(self, ctx: Context) -> bool:
        return bool(re.search(self.pattern, ctx.text()))


@dataclass(frozen=True)
class HashPaths(Effect):
    paths: tuple[str, ...]
    layer: str = "build"


@dataclass(frozen=True)
class WatchPaths(Effect):
    """Paths whose change across a checkout makes the git hooks resync."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class IgnoreUntracked(Effect):
    """Untracked files that are build outputs, not source edits."""

    pattern: str


@dataclass(frozen=True)
class PatchSource(Effect):
    file: str  # relative to the patched layer's working directory
    old: str
    new: str
    layer: str = ""
    keyed: ClassVar[bool] = True

    def apply(self, text: str) -> str:
        return text.replace(self.old, self.new)


@dataclass(frozen=True)
class DerivedInstall(Effect):
    """Install `package` at `version_of`'s installed version, from `index`
    (formatted with {cuda_variant}, e.g. 130)."""

    package: str
    version_of: str
    index: str
    layer: str = "venv"
    keyed: ClassVar[bool] = True


@dataclass(frozen=True)
class Fallback(Effect):
    """A reactive try-and-fallback that lives in code (`handler`); listed here
    so every version workaround is discoverable from one table."""

    kind: str
    handler: str  # dotted path of the implementing function


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    id: str
    rev: int
    reason: str  # why, citing the upstream commit/PR
    when: tuple[Pred, ...] = ()
    then: tuple[Effect, ...] = ()
    slot: str | None = None
    priority: int = 0

    @property
    def key_part(self) -> str:
        return f"rule:{self.id}@{self.rev}"

    def holds(self, ctx: Context) -> bool:
        return all(p.holds(ctx) for p in self.when) and all(
            e.matches(ctx) for e in self.then
        )


_MARLIN_GENERATED = (
    r"^csrc/(?:libtorch_stable/)?(?:quantization/marlin|moe/marlin_moe_wna16)/"
    r"(?:sm\d+_kernel_.*\.cu|kernel_selector\.h)$"
)
_BUILD_SOURCES = ("csrc", "cmake", "CMakeLists.txt", "setup.py")
_RUST_SOURCES = ("rust", "rust-toolchain.toml", "tools/build_rust.py")

RULES: tuple[Rule, ...] = (
    # Requirements layout eras (layout.build / layout.runtime / layout.test).
    Rule("layout-build-per-platform", 1,
         "fc645f1acc0 (#39024, 2026-04-10) moved build deps to "
         "requirements/build/<platform>.txt",
         slot="layout.build",
         then=(Select(("requirements/build/{platform}.txt",)),)),
    Rule("layout-build-platform-dir", 1,
         "206e2577fa9 (#12547, 2025-03-08): ROCm build deps live in "
         "requirements/rocm-build.txt",
         slot="layout.build",
         then=(Select(("requirements/{platform}-build.txt",)),)),
    Rule("layout-build-dir", 1,
         "206e2577fa9 (#12547, 2025-03-08) moved requirements into "
         "requirements/",
         slot="layout.build",
         then=(Select(("requirements/build.txt",)),)),
    Rule("layout-build-root-platform", 1,
         "before 206e2577fa9: root-level requirements-<platform>-build.txt",
         slot="layout.build",
         then=(Select(("requirements-{platform}-build.txt",)),)),
    Rule("layout-build-root", 1,
         "before 206e2577fa9: root-level requirements-build.txt",
         slot="layout.build",
         then=(Select(("requirements-build.txt",)),)),
    Rule("layout-runtime-dir", 1,
         "206e2577fa9 (#12547, 2025-03-08) moved requirements into "
         "requirements/",
         slot="layout.runtime",
         then=(Select(("requirements/common.txt", "requirements/{platform}.txt",
                       "requirements/lint.txt")),)),
    Rule("layout-runtime-root", 1,
         "before 206e2577fa9: root-level requirements-*.txt",
         slot="layout.runtime",
         then=(Select(("requirements-common.txt", "requirements-{platform}.txt",
                       "requirements-lint.txt")),)),
    Rule("test-lock-per-platform", 1,
         "fc645f1acc0 (#39024, 2026-04-10) per-platform test locks; locks "
         "are compiled for x86_64",
         slot="layout.test", when=(Arch("x86_64"),),
         then=(Select(("requirements/test/{platform}.txt",)),)),
    Rule("test-input-per-platform", 1,
         "fc645f1acc0 (#39024): unpinned inputs for hosts the locks were not "
         "compiled for",
         slot="layout.test",
         then=(Select(("requirements/test/{platform}.in",)),)),
    Rule("test-lock-cuda", 1,
         "206e2577fa9 (#12547, 2025-03-08) .. fc645f1acc0: requirements/test.txt "
         "was the CUDA x86_64 lock",
         slot="layout.test", when=(Platform("cuda"), Arch("x86_64")),
         then=(Select(("requirements/test.txt",)),)),
    Rule("test-lock-rocm", 1,
         "206e2577fa9 (#12547) .. fc645f1acc0: requirements/rocm-test.txt held "
         "ROCm's pins (hand-pinned; uv-compiled lock since db01535e2bd #37930)",
         slot="layout.test", when=(Platform("rocm"), Arch("x86_64")),
         then=(Select(("requirements/rocm-test.txt",)),)),
    Rule("test-input", 1,
         "206e2577fa9 (#12547) .. fc645f1acc0: unpinned requirements/test.in",
         slot="layout.test",
         then=(Select(("requirements/test.in",)),)),
    Rule("test-lock-root", 1,
         "before 206e2577fa9: root-level requirements-test.txt lock",
         slot="layout.test", when=(Arch("x86_64"),),
         then=(Select(("requirements-test.txt",)),)),
    Rule("layout-unrecognized", 1,
         "no known layout era matched: hash every requirements file "
         "(coarse but correct)",
         when=(Fact("requirements_recognized", False),),
         then=(Fallback("hash all requirements files",
                        "vllm_envs.treefacts.TreeFacts.layout"),)),

    # Cross-package compatibility (venv layer).
    Rule("cutlass-dsl-4.6-quack", 1,
         "nvidia-cutlass-dsl 4.6 (vLLM FA4/CuTe DSL kernels, 8b5014d3dd3 #32974) "
         "breaks quack-kernels before 0.6.1",
         when=(Req("nvidia-cutlass-dsl", ">=4.6.0"),),
         then=(AddReq("quack-kernels>=0.6.1"),)),
    Rule("flashinfer-0.6.15-cuda-pathfinder", 1,
         "flashinfer-python 0.6.15 needs cuda-pathfinder>=1.5.4 while compiled "
         "test locks still pinned older; override until the lock catches up",
         when=(Req("flashinfer-python", ">=0.6.15"),
               Req("cuda-pathfinder", "<1.5.4")),
         then=(OverrideReq("cuda-pathfinder>=1.5.4"),)),
    Rule("cuda-local-version-index", 1,
         "pinned locks carry +cuNNN local versions (d47b605eca4 #17576; e.g. "
         "torchcodec==0.14.0+cu130) that uv's --torch-backend does not cover",
         when=(Platform("cuda"),),
         then=(ExtraIndex(r"\+cu(\d+)", "https://download.pytorch.org/whl/cu{0}",
                          strategy="unsafe-best-match"),)),
    Rule("flashinfer-jit-cache", 1,
         "c9d33c60dcd (#26443) made FlashInfer a default CUDA dependency; "
         "vLLM's image adds the matching flashinfer-jit-cache (via `flashinfer "
         "download-kernels` since 29e4d2d42e1 #58765)",
         when=(Platform("cuda"), Req("flashinfer-python")),
         then=(DerivedInstall("flashinfer-jit-cache", "flashinfer-python",
                              "https://flashinfer.ai/whl/cu{cuda_variant}"),)),
    Rule("flashinfer-cubin", 1,
         "flashinfer-cubin left PyPI at 0.6.14 (ba47bb5be1b #47669) and "
         "requirements/cuda.txt at 29e4d2d42e1 (#58765); vLLM's image installs "
         "it with `flashinfer download-kernels`. A no-op while still pinned",
         when=(Platform("cuda"), Req("flashinfer-python", ">=0.6.14")),
         then=(DerivedInstall("flashinfer-cubin", "flashinfer-python",
                              "https://flashinfer.ai/whl"),)),
    Rule("cap-fallback", 1,
         "stale floors (e.g. tokenizers>=0.21 vs transformers needing >=0.22) "
         "conflict under tight caps",
         then=(Fallback("cap minor -> major -> none",
                        "vllm_envs.venvlayer._install_reqs"),)),
    Rule("split-install-passes", 1,
         "compiled test locks go stale against requirements/<platform>.txt; "
         "upstream installs them in separate passes",
         then=(Fallback("one install pass per requirement group",
                        "vllm_envs.venvlayer._install_req_groups"),)),

    # Build layer inputs and hook triggers.
    Rule("build-sources", 1,
         "compiled extensions are built from these paths",
         then=(HashPaths(_BUILD_SOURCES),)),
    Rule("rust-frontend-sources", 1,
         "f2ace1d57d2 (#40848, 2026-05-20) Rust frontend; fdfb2566c03 (#44981) "
         "tools/build_rust.py: wheels carry vllm/_rust_*.so",
         when=(Fact("rust_frontend"),),
         then=(HashPaths(_RUST_SOURCES),)),
    Rule("marlin-generated-sources", 1,
         "several vLLM releases generate Marlin instantiations into csrc; they "
         "survive checkouts as untracked build outputs",
         then=(IgnoreUntracked(_MARLIN_GENERATED),)),
    Rule("watch-requirements", 1,
         "requirements feed the venv layers",
         then=(WatchPaths(("requirements",)),)),
    Rule("watch-extras-inputs", 1,
         "inputs of the DeepEP and KV-connector extras layers",
         then=(WatchPaths(("docker/versions.json", "docker/Dockerfile",
                           "tools/ep_kernels",
                           ".buildkite/scripts/install-kv-connectors.sh")),)),
    Rule("ext-src-short-sha", 1,
         "historical cmake/installer pins use short SHAs servers do not "
         "advertise",
         then=(Fallback("bounded default-branch fetch, then GitHub API",
                        "vllm_envs.extprojects.fetch_ref"),)),
    Rule("ext-src-subdir-missing", 1,
         "a pinned tag may lack the SOURCE_SUBDIR newer cmake expects",
         then=(Fallback("leave the project to FetchContent",
                        "vllm_envs.extprojects.src_dir_env"),)),
    Rule("local-build-precompiled-fallback", 1,
         "a failed local build falls back to upstream VLLM_USE_PRECOMPILED",
         then=(Fallback("uncached upstream precompiled attach",
                        "vllm_envs.buildlayer.resolve_build"),)),

    # vLLM extras.
    Rule("deepep-nccl-override", 1,
         "5d8e90a9661 (#45321, 2026-07-24): DeepEP's Gin backend needs "
         "NCCL_VERSION; vLLM's image overrides nvidia-nccl-cu${CUDA_MAJOR}",
         when=(Fact("ep_supported"), Fact("nccl_version")),
         then=(OverrideReq("nvidia-nccl-cu{cuda_major}=={nccl_version}",
                           layer="ep"),)),
    Rule("pplx-with-tests-flag", 1,
         "historical pplx-kernels pass -WITH_TESTS=OFF, which CMake 4 rejects; "
         "pplx was dropped in eb19955c370 (#33724)",
         when=(Fact("ep_supported"), Fact("pplx_ref")),
         then=(PatchSource("pplx-kernels/setup.py", '"-WITH_TESTS=OFF"',
                           '"-DWITH_TESTS=OFF"', layer="ep"),)),
    Rule("kv-installer-venv", 1,
         "816cc73a9be (#44266) KV CI helper targets the container's system "
         "Python; run it against the managed venv",
         when=(Fact("kv_supported"),),
         then=(PatchSource(".buildkite/scripts/install-kv-connectors.sh",
                           " --system", "", layer="kv"),)),
)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def fired(ctx: Context, rules: tuple[Rule, ...] = RULES) -> list[Rule]:
    """Unslotted rules that hold for `ctx`."""
    return [r for r in rules if r.slot is None and r.holds(ctx)]


def select(
    ctx: Context, slot: str, rules: tuple[Rule, ...] = RULES
) -> tuple[Rule | None, list[Path]]:
    """The winning rule of `slot` and the files its Select effects chose."""
    candidates = sorted(
        (r for r in rules if r.slot == slot), key=lambda r: -r.priority
    )
    for rule in candidates:
        if rule.holds(ctx):
            files = [
                f for e in rule.then if isinstance(e, Select)
                for f in e.existing(ctx)
            ]
            return rule, files
    return None, []


def effects(
    ctx: Context, kind: type, layer: str | None = None,
    rules: tuple[Rule, ...] = RULES,
) -> list[tuple[Rule, Any]]:
    return [
        (rule, e)
        for rule in fired(ctx, rules)
        for e in rule.then
        if isinstance(e, kind) and (layer is None or e.layer == layer)
    ]


def key_parts(
    ctx: Context, layer: str, rules: tuple[Rule, ...] = RULES
) -> list[str]:
    """`rule:<id>@<rev>` for every fired rule with a keyed `layer` effect."""
    return [
        rule.key_part
        for rule in fired(ctx, rules)
        if any(e.keyed and e.layer == layer for e in rule.then)
    ]


def _static(kind: type) -> list:
    return [e for rule in RULES for e in rule.then if isinstance(e, kind)]


def watch_paths() -> tuple[str, ...]:
    """Every path any rule may hash or watch, whether or not it fires now: a
    checkout can cross the boundary where a rule starts or stops firing."""
    paths = [p for e in _static(WatchPaths) for p in e.paths]
    paths += [p for e in _static(HashPaths) for p in e.paths]
    return tuple(dict.fromkeys(paths))


def untracked_ignores() -> tuple[re.Pattern, ...]:
    return tuple(re.compile(e.pattern) for e in _static(IgnoreUntracked))


def apply_patches(patches, file: str, text: str) -> str:
    for patch in patches:
        if patch.file == file:
            text = patch.apply(text)
    return text


def explain(facts: TreeFacts, with_test: bool = True) -> list[dict]:
    """Every rule's verdict for a tree, with the cache keys it enters.

    Requirement predicates are evaluated against each venv layer's own files,
    as venv_keys and the installs do."""
    layout = facts.layout(with_test)
    contexts = {
        "venv-base": facts.context(layout.build_files),
        "venv-full": facts.context([*layout.runtime_files, *layout.test_files]),
    }
    tree = facts.context()
    winners = {}
    for slot in dict.fromkeys(r.slot for r in RULES if r.slot):
        rule, files = select(tree, slot)
        if rule is not None:
            winners[slot] = (rule.id, files)
    out = []
    for rule in RULES:
        entry = {"id": rule.id, "rev": rule.rev, "reason": rule.reason,
                 "slot": rule.slot, "fired": False, "keys": [], "selected": [],
                 "effects": [repr(e) for e in rule.then],
                 # Armed try-and-fallback code, not a property of the tree.
                 "reactive": not rule.when and all(
                     isinstance(e, Fallback) for e in rule.then)}
        if rule.slot:
            winner = winners.get(rule.slot)
            if winner and winner[0] == rule.id:
                entry["fired"] = True
                entry["selected"] = [str(f.relative_to(facts.root)) for f in winner[1]]
        else:
            uses_files = any(isinstance(p, Req) for p in rule.when) or any(
                isinstance(e, ExtraIndex) for e in rule.then)
            holds_in = ([name for name, ctx in contexts.items() if rule.holds(ctx)]
                        if uses_files else (["tree"] if rule.holds(tree) else []))
            entry["fired"] = bool(holds_in)
            for e in rule.then:
                if not (e.keyed and holds_in):
                    continue
                layers = holds_in if e.layer == "venv" else [e.layer]
                entry["keys"] += [k for k in layers if k not in entry["keys"]]
        out.append(entry)
    return out
