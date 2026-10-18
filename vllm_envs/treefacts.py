"""Named facts about a vLLM working tree, computed once per sync.

Every probe of "what does this vLLM commit look like" lives here, so all
consumers (venv/build keys, DeepEP, KV connectors, hooks, `ve status`,
`ve rules`) agree. Facts read working-tree files only (syncs operate on the
checkout) and are evaluated lazily. Each fact records its detector, the
upstream change that introduced what it detects, and what it gates; `ve rules`
prints them.
"""

from __future__ import annotations

import ast
import json
import platform as host_platform
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from . import rules
from .reqs import requirement_names


@dataclass(frozen=True)
class FactSpec:
    name: str
    detects: str
    since: str
    gates: str


FACTS: dict[str, FactSpec] = {}


def fact(*, detects: str, since: str, gates: str):
    def wrap(fn):
        FACTS[fn.__name__] = FactSpec(fn.__name__, detects, since, gates)
        return cached_property(fn)
    return wrap


@dataclass
class ReqLayout:
    """Platform-relevant requirements files for a given worktree."""

    build_files: list[Path]  # heavy layer (torch + build deps)
    runtime_files: list[Path]  # top-up layer
    recognized: bool
    test_files: list[Path] = field(default_factory=list)  # optional test deps


EP_INSTALLER = "tools/ep_kernels/install_python_libraries.sh"
KV_INSTALLER = ".buildkite/scripts/install-kv-connectors.sh"
KV_REQUIREMENTS = "requirements/kv_connectors.txt"
# Wheel path sets for commits whose setup.py predates the constants.
DEFAULT_BLOCKING_PATHS = (
    "csrc", "cmake", "CMakeLists.txt", "pyproject.toml", "vllm/_custom_ops.py"
)
DEFAULT_WARNING_PATHS = ("rust", "setup.py")


def shell_default(script: str, name: str) -> str | None:
    """`NAME=${NAME:-value}`: a pin the script also lets callers override."""
    pattern = rf'^{name}=\$\{{{name}:-["\']?([^"\'}} ]+)["\']?\}}'
    match = re.search(pattern, script, re.MULTILINE)
    return match.group(1) if match else None


def setup_constant(source: str, name: str) -> tuple[str, ...] | None:
    """A tuple of strings assigned to `name` anywhere in setup.py (e.g. in a
    class body), without executing it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        try:
            paths = ast.literal_eval(value)
        except ValueError:
            return None
        if (isinstance(paths, (tuple, list)) and paths
                and all(isinstance(p, str) for p in paths)):
            return tuple(paths)
    return None


def wheel_paths(setup_source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(blocking, warning) compiled-source path sets declared by setup.py."""
    return (
        setup_constant(setup_source, "WHEEL_BLOCKING_PATHS") or DEFAULT_BLOCKING_PATHS,
        setup_constant(setup_source, "WHEEL_WARNING_PATHS") or DEFAULT_WARNING_PATHS,
    )


class TreeFacts:
    def __init__(self, root: Path, platform: str, machine: str | None = None):
        self.root = Path(root)
        self.platform = platform
        self.machine = machine or host_platform.machine()

    def context(self, files=()) -> rules.Context:
        return rules.Context(
            self.platform, self.machine, self.root, self, tuple(files)
        )

    def _file(self, rel: str) -> Path | None:
        p = self.root / rel
        return p if p.is_file() else None

    def _read(self, rel: str) -> str:
        p = self._file(rel)
        return p.read_text(errors="replace") if p else ""

    # ---- requirements layout -------------------------------------------

    def _slot(self, slot: str) -> list[Path]:
        return rules.select(self.context(), slot)[1]

    @fact(detects="layout.build slot rules (file presence)",
          since="see layout-build-* rules", gates="venv base key + install")
    def build_requirements(self) -> list[Path]:
        return self._slot("layout.build")

    @fact(detects="layout.runtime slot rules (file presence)",
          since="see layout-runtime-* rules", gates="venv full key + install")
    def runtime_requirements(self) -> list[Path]:
        return self._slot("layout.runtime")

    @fact(detects="layout.test slot rules (file presence + host arch)",
          since="see test-* rules", gates="venv full key + install ([venv] test)")
    def test_requirements(self) -> list[Path]:
        return self._slot("layout.test")

    @fact(detects="both build and runtime layout slots matched",
          since="-", gates="coarse hash-everything fallback (layout-unrecognized)")
    def requirements_recognized(self) -> bool:
        return bool(self.build_requirements and self.runtime_requirements)

    def layout(self, with_test: bool = False) -> ReqLayout:
        if self.requirements_recognized:
            return ReqLayout(
                list(self.build_requirements), list(self.runtime_requirements),
                recognized=True,
                test_files=list(self.test_requirements) if with_test else [],
            )
        req = self.root / "requirements"
        found = set(self.root.glob("requirements*"))
        if req.is_dir():
            found |= set(req.rglob("*"))
        everything = sorted(p for p in found if p.is_file())
        return ReqLayout(everything, everything, recognized=False)

    @fact(detects="flashinfer-python in runtime/test requirements",
          since="c9d33c60dcd (#26443, 2025-10-09)",
          gates="flashinfer-jit-cache derived install")
    def flashinfer_python(self) -> bool:
        files = [*self.runtime_requirements, *self.test_requirements]
        return "flashinfer-python" in requirement_names(files)

    # ---- build layer ---------------------------------------------------

    @fact(detects="rust-toolchain.toml or rust/Cargo.toml exists",
          since="f2ace1d57d2 (#40848, 2026-05-20)",
          gates="Rust sources in the build key (rust-frontend-sources)")
    def rust_frontend(self) -> bool:
        return bool(self._file("rust-toolchain.toml") or self._file("rust/Cargo.toml"))

    @fact(detects="HashPaths effects of fired rules", since="-",
          gates="build key, dirty check")
    def build_layer_paths(self) -> tuple[str, ...]:
        found = rules.effects(self.context(), rules.HashPaths, layer="build")
        return tuple(dict.fromkeys(p for _, e in found for p in e.paths))

    @cached_property
    def _wheel_paths(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return wheel_paths(self._read("setup.py"))

    @fact(detects="setup.py declares WHEEL_BLOCKING_PATHS (AST)",
          since="90ba2f34b5a (#58871, 2026-10-09)",
          gates="whether wheel path sets come from setup.py or ve's defaults")
    def wheel_paths_declared(self) -> bool:
        return setup_constant(self._read("setup.py"), "WHEEL_BLOCKING_PATHS") is not None

    @fact(detects="setup.py WHEEL_BLOCKING_PATHS, else ve default",
          since="90ba2f34b5a (#58871)", gates="precompiled wheel identity")
    def wheel_blocking_paths(self) -> tuple[str, ...]:
        return self._wheel_paths[0]

    @fact(detects="setup.py WHEEL_WARNING_PATHS, else ve default",
          since="90ba2f34b5a (#58871)", gates="precompiled stale-path warning")
    def wheel_warning_paths(self) -> tuple[str, ...]:
        return self._wheel_paths[1]

    # ---- DeepEP --------------------------------------------------------

    @fact(detects=f"{EP_INSTALLER} exists",
          since="d45fe333fb8 (#17964, 2025-05-12)", gates="DeepEP layer")
    def ep_installer(self) -> Path | None:
        return self._file(EP_INSTALLER)

    @cached_property
    def _ep_script(self) -> str:
        return self._read(EP_INSTALLER)

    def _docker_variable(self, name: str) -> str | None:
        try:
            value = json.loads(self._read("docker/versions.json"))
            return str(value["variable"][name]["default"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None

    @fact(detects="docker/versions.json DEEPEP_COMMIT_HASH, else installer "
                  "${DEEPEP_COMMIT_HASH:-...}",
          since="9e078d05825 (#31492, 2026-01-17) / c4c0354eec8 (#29131, "
                "2025-11-26)",
          gates="DeepEP ref and layer key")
    def deepep_ref(self) -> str | None:
        return self._docker_variable("DEEPEP_COMMIT_HASH") or shell_default(
            self._ep_script, "DEEPEP_COMMIT_HASH")

    @fact(detects="which file supplied deepep_ref", since="9e078d05825 (#31492)",
          gates="explanation only")
    def deepep_ref_source(self) -> str | None:
        if self._docker_variable("DEEPEP_COMMIT_HASH"):
            return "docker/versions.json"
        if shell_default(self._ep_script, "DEEPEP_COMMIT_HASH"):
            return EP_INSTALLER
        return None

    @fact(detects="ep_installer and deepep_ref (installer takes --deepep-ref "
                  "and --mode wheel)",
          since="c4c0354eec8 (#29131)",
          gates="DeepEP layer in sync_deepep and the extras bundle key")
    def ep_supported(self) -> bool:
        return bool(self.ep_installer and self.deepep_ref)

    def ep_unsupported_reason(self) -> str:
        if not self.ep_installer:
            return f"no EP installer ({EP_INSTALLER}, added d45fe333fb8)"
        if not self.deepep_ref:
            return "EP installer pins no DeepEP ref (added c4c0354eec8)"
        return ""

    @fact(detects="installer ${PPLX_COMMIT_HASH:-...}",
          since="c4c0354eec8 (#29131); removed eb19955c370 (#33724, 2026-02-26)",
          gates="pplx-kernels prefetch + pplx-with-tests-flag patch")
    def pplx_ref(self) -> str | None:
        return shell_default(self._ep_script, "PPLX_COMMIT_HASH")

    @fact(detects="docker/versions.json NCCL_VERSION",
          since="5d8e90a9661 (#45321, 2026-07-24)",
          gates="deepep-nccl-override")
    def nccl_version(self) -> str | None:
        return self._docker_variable("NCCL_VERSION")

    @fact(detects="docker/Dockerfile `export TORCH_CUDA_ARCH_LIST=` (EP stage)",
          since="eb5352a7707 (#26966, 2025-11-22)",
          gates="DeepEP CUDA architectures")
    def docker_cuda_arch_list(self) -> str:
        dockerfile = self._read("docker/Dockerfile")
        m = re.search(r"export TORCH_CUDA_ARCH_LIST=['\"]([^'\"]+)", dockerfile)
        value = m.group(1) if m else ""
        # 6b24947628 (#61829): the export is "$DEEPEP_ARCH_LIST", built from a
        # literal plus CUDA-version-conditional appends (the local GPU's arch
        # is added by deepep._cuda_arch_list).
        if var := re.fullmatch(r"\$\{?(\w+)\}?", value):
            m = re.search(rf"\b{var.group(1)}=['\"]([^'\"$]+)['\"]", dockerfile)
            value = m.group(1) if m else ""
        return value

    # ---- KV connectors -------------------------------------------------

    @fact(detects=f"{KV_INSTALLER} exists",
          since="816cc73a9be (#44266, 2026-06-01)", gates="KV connectors")
    def kv_installer(self) -> Path | None:
        return self._file(KV_INSTALLER)

    @fact(detects=f"{KV_REQUIREMENTS} exists",
          since="5561681d040 (#18193, 2025-07-04)", gates="KV connectors")
    def kv_requirements(self) -> Path | None:
        return self._file(KV_REQUIREMENTS)

    @fact(detects="kv_installer and kv_requirements", since="816cc73a9be (#44266)",
          gates="KV connectors in the extras bundle")
    def kv_supported(self) -> bool:
        return bool(self.kv_installer and self.kv_requirements)


_SCOPE: ContextVar[dict | None] = ContextVar("treefacts_scope", default=None)


@contextmanager
def scope():
    """Share one TreeFacts per (tree, platform) for the duration of a sync."""
    if _SCOPE.get() is not None:
        yield
        return
    token = _SCOPE.set({})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def facts_for(root: Path, platform: str) -> TreeFacts:
    cache = _SCOPE.get()
    if cache is None:
        return TreeFacts(root, platform)
    key = (str(Path(root).resolve()), platform)
    if key not in cache:
        cache[key] = TreeFacts(root, platform)
    return cache[key]
