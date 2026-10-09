"""A family's tuning declaration (RECIPES_AND_TUNING §4.1; MODEL_FAMILY_CONTRACT C6).

These are the declarations only. RECIPES RT-1 builds the search engine that reads them (the plan,
the candidates, the inner folds and the record), and RT-2 gives each family its ``tuning`` member.
The model-family contract adds ``structural`` (MC-1): the settings that are part of what the family
is (C1), such as an architecture, a loss or a booster type, which are stated and never searched.

A declaration is checked when it is made, so a family module whose tuning breaks C6 fails at import
with every problem named: a setting both structural and searched, and a searched dimension without
its plain label, its quiet term, a known scale or its source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, get_args

TuningKind = Literal["none", "path", "search"]
Scale = Literal["log", "linear", "int", "log_int", "choice", "share_of_units"]


@dataclass(frozen=True)
class Dimension:
    """One setting a family's tuning may move (RECIPES §4.1)."""

    name: str  # the estimator's parameter
    label: str  # plain: "how big each correction step is"
    term: str  # quiet: "learning rate"
    low: float
    high: float
    scale: Scale
    choices: tuple = ()
    source: str = ""


@dataclass(frozen=True)
class TuningDecl:
    """How a family is tuned (RECIPES §4.1), with the contract's ``structural`` settings (C6)."""

    kind: TuningKind
    dimensions: tuple[Dimension, ...] = ()  # searched
    by_hand: tuple[Dimension, ...] = ()  # settable by hand, never searched (Huber's threshold)
    standard: Mapping[str, Any] = field(default_factory=dict)  # the standard settings
    standard_source: str = ""  # "scikit-learn's defaults", "ranger's defaults"
    fixed: Mapping[str, Any] = field(default_factory=dict)  # stated, never searched
    early_stopping: Mapping[str, Any] | None = None
    out_of_bag: bool = False
    space_version: str = ""  # "boosted_trees/1"
    reason: str = ""  # ≤ 22 words
    # MODEL_FAMILY_CONTRACT C6: the settings that are identity (C1), never searched: the
    # architecture, the parameterization and output multiplier, the loss, the booster type.
    structural: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = tuning_problems(self)
        if problems:
            raise ValueError("the tuning declaration breaks the model-family contract: "
                             + "; ".join(problems))


def tuning_problems(decl: TuningDecl) -> list[str]:
    """What in ``decl`` breaks MODEL_FAMILY_CONTRACT C6, one plain clause each; empty when
    nothing does."""
    out: list[str] = []
    if decl.kind not in get_args(TuningKind):
        out.append(f"its kind {decl.kind!r} is not among {list(get_args(TuningKind))}")
    both = sorted({d.name for d in decl.dimensions} & set(decl.structural))
    if both:
        out.append(f"{both} are structural, part of what the family is, so they are never "
                   f"searched dimensions (C6)")
    for d in decl.dimensions:
        lacks = [what for what, given in (("a plain label", d.label), ("a quiet term", d.term),
                                          ("a source", d.source)) if not str(given).strip()]
        if d.scale not in get_args(Scale):
            lacks.append(f"a scale among {list(get_args(Scale))}")
        if lacks:
            out.append(f"the dimension {d.name!r} needs {', '.join(lacks)} (C6)")
    return out


__all__ = ["Dimension", "Scale", "TuningDecl", "TuningKind", "tuning_problems"]
