"""The model-family protocol and its registry (M1_CONTRACT §7).

A family is a plug-in: it declares what it needs (``needs_scaling``,
``handles_missing``), what it assumes (``inductive_bias``), and how it judges a
situation (``assess``), so the shelf, the pipeline, the lineage and the
consequence previews all derive from the declaration. A later milestone adds a
family by calling :func:`register_family` — nothing else in the app switches on
family keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import Purpose, Task

Fit = Literal["good", "fair", "poor"]
TASKS: tuple[Task, ...] = ("regression", "binary", "multiclass")
INDUCTIVE_BIAS_WORDS = 20


@dataclass(frozen=True)
class Situation:
    """What the shelf knows about the analysis when it ranks families."""

    task: Task
    purpose: Purpose | None
    n_rows: int
    n_features: int
    n_events: int | None = None  # binary: rows in the rarer class
    n_classes: int | None = None


@dataclass(frozen=True)
class Assessment:
    """A family's own judgment of a situation: order (``score``, higher first) and stated concern."""

    score: float
    fit: Fit
    concerns: tuple[str, ...] = ()


@runtime_checkable
class ModelFamily(Protocol):
    key: str
    label: str
    tasks: tuple[Task, ...]
    inductive_bias: str  # ≤ 20 words
    strengths: tuple[str, ...]
    cautions: tuple[str, ...]
    needs_scaling: bool
    handles_missing: bool

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        """An unfitted sklearn estimator; the pipeline's last step."""

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        """(label, detail) of the model step, in the app's voice."""

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        """Coefficient rows ``{feature, estimate, ci_low, ci_high, p}`` from the pipeline fit on
        ``X``/``y`` (training rows), or None when the family has none."""

    def assess(self, situation: Situation) -> Assessment:
        """How well this family suits the situation, with every concern stated."""


class FamilyInfo(BaseModel):
    """What ``GET /api/models`` says about one family."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: str
    label: str
    tasks: list[Task]
    inductive_bias: str
    strengths: list[str]
    cautions: list[str]
    needs_scaling: bool
    handles_missing: bool


_REGISTRY: dict[str, ModelFamily] = {}


def register_family(family: ModelFamily) -> ModelFamily:
    """Add ``family`` to the shelf. Its order of registration breaks ties in the ranking."""
    if not isinstance(family, ModelFamily):
        raise TypeError(f"{family!r} does not implement ModelFamily")
    words = len(family.inductive_bias.split())
    if words > INDUCTIVE_BIAS_WORDS:
        raise ValueError(f"{family.key}: inductive_bias has {words} words; the budget is "
                         f"{INDUCTIVE_BIAS_WORDS}")
    unknown = set(family.tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"{family.key}: unknown tasks {sorted(unknown)}")
    if family.key in _REGISTRY and _REGISTRY[family.key] is not family:
        raise ValueError(f"a model family {family.key!r} is already registered")
    _REGISTRY[family.key] = family
    return family


def unregister_family(key: str) -> None:
    """Remove a family (for plug-ins that are unloaded, and for tests)."""
    _REGISTRY.pop(key, None)


def families(task: Task | None = None) -> list[ModelFamily]:
    """Registered families (those that can model ``task``, when given), in registration order."""
    return [f for f in _REGISTRY.values() if task is None or task in f.tasks]


def get_family(key: str) -> ModelFamily:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"there is no model family {key!r}; known: {', '.join(_REGISTRY)}") from None


def info(family: ModelFamily) -> FamilyInfo:
    return FamilyInfo(
        key=family.key, label=family.label, tasks=list(family.tasks),
        inductive_bias=family.inductive_bias, strengths=list(family.strengths),
        cautions=list(family.cautions), needs_scaling=family.needs_scaling,
        handles_missing=family.handles_missing,
    )


def rank(situation: Situation) -> list[tuple[ModelFamily, Assessment]]:
    """Every family that can model the task, best first. The shelf is never shortened."""
    order = {f.key: i for i, f in enumerate(families())}
    judged = [(f, f.assess(situation)) for f in families(situation.task)]
    return sorted(judged, key=lambda fa: (-fa[1].score, order[fa[0].key]))


# ── shared helpers for families ──────────────────────────────────────────────


class FamilyBase:
    """Defaults a concrete family can inherit; it still declares everything the protocol asks."""

    key: str = ""
    label: str = ""
    tasks: tuple[Task, ...] = TASKS
    inductive_bias: str = ""
    strengths: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    needs_scaling: bool = False
    handles_missing: bool = False

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        return None

    def __repr__(self) -> str:
        return f"<model family {self.key}>"


def coefficient_rows(features: Sequence[str], estimates: Any, *, intercept: Any = None,
                     classes: Sequence[Any] | None = None, ci_low: Any = None, ci_high: Any = None,
                     p: Any = None) -> list[dict[str, Any]]:
    """Coefficient rows in the fit artifact's shape.

    ``estimates`` is (p,) or (k, p) for k class rows; with class rows each feature is labeled
    ``feature [class]``. ``intercept`` is a scalar or (k,). CIs and p-values align with estimates.
    """
    import numpy as np

    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    lo = None if ci_low is None else np.atleast_2d(np.asarray(ci_low, dtype=float))
    hi = None if ci_high is None else np.atleast_2d(np.asarray(ci_high, dtype=float))
    pv = None if p is None else np.atleast_2d(np.asarray(p, dtype=float))
    icpt = None if intercept is None else np.atleast_1d(np.asarray(intercept, dtype=float))
    rows: list[dict[str, Any]] = []
    for k in range(est.shape[0]):
        suffix = f" [{classes[k]}]" if classes is not None and est.shape[0] > 1 else ""
        names = (["(intercept)"] if icpt is not None else []) + list(features)
        values = ([icpt[k]] if icpt is not None else []) + list(est[k])
        for j, (name, value) in enumerate(zip(names, values)):
            rows.append({
                "feature": f"{name}{suffix}",
                "estimate": _finite(value),
                "ci_low": None if lo is None else _finite(lo[k][j]),
                "ci_high": None if hi is None else _finite(hi[k][j]),
                "p": None if pv is None else _finite(pv[k][j]),
            })
    return rows


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


__all__ = [
    "Assessment", "FamilyBase", "FamilyInfo", "Fit", "ModelFamily", "Situation", "TASKS",
    "coefficient_rows", "families", "get_family", "info", "rank", "register_family", "unregister_family",
]
