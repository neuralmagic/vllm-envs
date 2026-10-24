"""Rule engine for the workaround table in `workarounds.py`.

A Rule reads "trees where every `when` predicate holds need the `then`
effects". Predicates probe content (requirement declarations, the platform,
named TreeFacts), never commit ranges. A fired rule contributes
`rule:<id>@<rev>` to the cache key of each layer its effects change, so
bumping `rev` invalidates exactly what the rule affects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    """What rules are evaluated against: the platform, a tree's facts
    (optional), and the requirement files a consumer is about to hash or
    install."""

    platform: str
    facts: TreeFacts | None = None
    files: tuple[Path, ...] = ()
    _memo: dict = field(default_factory=dict, compare=False, repr=False)

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


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------


class Effect:
    layer: str = ""

    def matches(self, ctx: Context) -> bool:
        """Guard: an effect with nothing to act on keeps its rule from firing."""
        return True


@dataclass(frozen=True)
class AddReq(Effect):
    spec: str
    layer: str = "venv"


@dataclass(frozen=True)
class OverrideReq(Effect):
    """A resolver override (uv --overrides / UV_OVERRIDE). `spec` may use
    {placeholders} the consumer fills in."""

    spec: str
    layer: str = "venv"


@dataclass(frozen=True)
class ExtraIndex(Effect):
    """An extra index per distinct `pattern` capture in the requirement files."""

    pattern: str
    url: str  # formatted with the capture as {0}
    strategy: str = ""
    layer: str = "venv"

    def urls(self, ctx: Context) -> list[str]:
        captures = sorted(set(re.findall(self.pattern, ctx.text())))
        return [self.url.format(c) for c in captures]

    def matches(self, ctx: Context) -> bool:
        return bool(re.search(self.pattern, ctx.text()))


@dataclass(frozen=True)
class PatchSource(Effect):
    file: str  # relative to the patched layer's working directory
    old: str
    new: str
    layer: str = ""

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

    @property
    def key_part(self) -> str:
        return f"rule:{self.id}@{self.rev}"

    def holds(self, ctx: Context) -> bool:
        return all(p.holds(ctx) for p in self.when) and all(
            e.matches(ctx) for e in self.then
        )


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def _table(rules: tuple[Rule, ...] | None) -> tuple[Rule, ...]:
    if rules is None:
        from .workarounds import RULES
        return RULES
    return rules


def fired(ctx: Context, rules: tuple[Rule, ...] | None = None) -> list[Rule]:
    return [r for r in _table(rules) if r.holds(ctx)]


def effects(
    ctx: Context, kind: type, layer: str | None = None,
    rules: tuple[Rule, ...] | None = None,
) -> list[tuple[Rule, Any]]:
    return [
        (rule, e)
        for rule in fired(ctx, rules)
        for e in rule.then
        if isinstance(e, kind) and (layer is None or e.layer == layer)
    ]


def key_parts(
    ctx: Context, layer: str, rules: tuple[Rule, ...] | None = None
) -> list[str]:
    """`rule:<id>@<rev>` for every fired rule with a `layer` effect."""
    return [
        rule.key_part
        for rule in fired(ctx, rules)
        if any(e.layer == layer for e in rule.then)
    ]


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
    out = []
    for rule in _table(None):
        uses_files = any(isinstance(p, Req) for p in rule.when) or any(
            isinstance(e, ExtraIndex) for e in rule.then)
        holds_in = ([name for name, ctx in contexts.items() if rule.holds(ctx)]
                    if uses_files else (["tree"] if rule.holds(facts.context()) else []))
        keys = []
        for e in rule.then if holds_in else ():
            keys += [k for k in (holds_in if e.layer == "venv" else [e.layer])
                     if k not in keys]
        out.append({"id": rule.id, "rev": rule.rev, "reason": rule.reason,
                    "fired": bool(holds_in), "keys": keys,
                    "effects": [repr(e) for e in rule.then]})
    return out
