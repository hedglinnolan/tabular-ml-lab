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
TASKS: tuple[Task, ...] = ("regression", "binary", "multiclass", "ordinal", "time_to_event")
# What a family models unless it says otherwise: an ordered outcome is modeled as unordered
# classes by a family that does not declare ``ordered_levels = True`` (on that shelf it says so and
# ranks a step lower: audit ME-19, RO-10); a time-to-event outcome is an event with its follow-up,
# which a family takes only by declaring it (``survival.Cox``).
DEFAULT_TASKS: tuple[Task, ...] = ("regression", "binary", "multiclass", "ordinal")
ORDER_BLIND = "Treats the ordered levels as unordered classes, so it ignores their order."
ORDER_BLIND_COST = 1.0
INDUCTIVE_BIAS_WORDS = 20


@dataclass(frozen=True)
class Situation:
    """What the shelf knows about the analysis when it ranks families."""

    task: Task
    purpose: Purpose | None
    n_rows: int
    n_features: int
    n_events: int | None = None  # binary: rows in the rarer class; time to event: rows with the event
    n_classes: int | None = None
    # Candidate predictor parameters (a category with k levels is k − 1 of them), which sample-size
    # criteria count; None: one per predictor. The outcome's mean and SD on the rows ranked
    # (regression), which Riley et al.'s intercept criterion needs.
    n_parameters: int | None = None
    outcome_mean: float | None = None
    outcome_sd: float | None = None
    lenses: tuple[str, ...] = ()  # the declared lenses (an omics lens changes what is sound)
    class_counts: tuple[int, ...] | None = None  # rows per class or level
    # Units when an identifier repeats in these rows (``inference.resolve_clusters``); None when
    # every row is a unit of its own, or nothing says which rows belong together.
    n_units: int | None = None


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
        ``X``/``y``, or None when the family has none. Under prediction those are the training
        rows; under inference, every analyzed row (BLUEPRINT §12 ruling 3)."""

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
    # The purposes the family serves, and whether it predicts (a family that only tests has no
    # cross-validated score). Read with getattr and these defaults: FamilyBase declares them.
    purposes: list[Purpose] = ["prediction", "inference"]
    predicts: bool = True


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
        purposes=list(getattr(family, "purposes", ("prediction", "inference"))),
        predicts=bool(getattr(family, "predicts", True)),
    )


def rank(situation: Situation) -> list[tuple[ModelFamily, Assessment]]:
    """Every family that can model the task, best first. The shelf is never shortened."""
    order = {f.key: i for i, f in enumerate(families())}
    judged = []
    for family in families(situation.task):
        judged_one = family.assess(situation)
        if situation.task == "ordinal" and not getattr(family, "ordered_levels", False):
            judged_one = Assessment(judged_one.score - ORDER_BLIND_COST, judged_one.fit,
                                    (ORDER_BLIND, *judged_one.concerns))
        judged.append((family, judged_one))

    def place(fa: tuple[ModelFamily, Assessment]) -> tuple[float, bool, int]:
        # Under prediction, a family that makes no predictions (WP11's feature-wise tests) goes
        # after every family that does and scores as well: it has nothing to offer the purpose.
        family, judged_one = fa
        silent = situation.purpose == "prediction" and not getattr(family, "predicts", True)
        return (-judged_one.score, silent, order[family.key])

    return sorted(judged, key=place)


# ── shared helpers for families ──────────────────────────────────────────────


class FamilyBase:
    """Defaults a concrete family can inherit; it still declares everything the protocol asks."""

    key: str = ""
    label: str = ""
    tasks: tuple[Task, ...] = DEFAULT_TASKS
    inductive_bias: str = ""
    strengths: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    needs_scaling: bool = False
    handles_missing: bool = False
    purposes: tuple[Purpose, ...] = ("prediction", "inference")  # the purposes it serves
    predicts: bool = True  # False: it tests and makes no predictions (no cross-validated score)
    # Its model is a weighted sum of the values as given, so their scale (raw counts or log) is part
    # of what it assumes; raw omics values wait for a normalization (``methods.omics``).
    linear_in_values: bool = False
    # Whether Harrell's bootstrap optimism correction is sound for it (``models.validation``). A
    # learner that nearly memorizes its rows scores the original rows inside each resample almost
    # perfectly, so the bootstrap understates its optimism (Coley et al. 2023): it declares False,
    # and the fit keeps its cross-validated score as its internal validation.
    bootstrap_optimism: bool = True

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        return None

    def __repr__(self) -> str:
        return f"<model family {self.key}>"


def reports_coefficients(family: Any) -> bool:
    """Whether ``family`` has a coefficient table: it defines ``coefficients`` itself rather than
    inheriting :class:`FamilyBase`'s, which has none."""
    method = getattr(type(family), "coefficients", None)
    return method is not None and method is not FamilyBase.coefficients


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
    "Assessment", "DEFAULT_TASKS", "FamilyBase", "FamilyInfo", "Fit", "ModelFamily", "ORDER_BLIND",
    "Situation", "TASKS", "coefficient_rows", "families", "get_family", "info", "rank",
    "register_family", "reports_coefficients", "unregister_family",
]
