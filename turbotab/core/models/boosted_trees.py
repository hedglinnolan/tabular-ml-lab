"""``boosted_trees``: gradient-boosted decision trees (HistGradientBoosting), missing values native.

Its fitted trees reach the explanations through its ``trees`` member (:func:`hgb_ensemble`), the
one place that reads scikit-learn's private ``_predictors`` and ``_baseline_prediction``
(MODEL_FAMILY_CONTRACT C10; WAVE_C6A_PLAN §2, package RT-1a).

**Tuned (RT-5a; RECIPES §4.1).** Its settings are searched inside every training fold
(:data:`TUNING`): learning rate, leaves per tree, the smallest leaf (capped at a twentieth of the
plan's count, :func:`leaf_cap`), the L2 pull, the share of columns per split, and the number of
trees when the plan does not stop early. scikit-learn's defaults are the standard candidate, and
below an effective size of 300 (units, events or the rarest class's count) the plan keeps it
alone: one fit at scikit-learn's defaults. That fit equals a direct ``HistGradientBoosting*()``
fit at the same thread count while the plan's rows are at most 10,000 (above them the stopping
rows are whole units the plan draws, not scikit-learn's 10% of positions) and the fit's rows at
most 200,000 (above them the plan's seed, not scikit-learn's, picks the binning subsample).
"Try both" for blanks waits for RT-3 (C6b).

**Pinned threads (RECIPES §4.7).** The family builds :class:`PinnedHistGradientBoostingRegressor`
/ :class:`PinnedHistGradientBoostingClassifier`: scikit-learn's estimators with one more
parameter, ``n_threads`` (the plan's ``threads``, 1 without a plan), the OpenMP thread count of
every fit and prediction. Histogram boosting otherwise takes every thread the process allows on
each of a search's small fits, which oversubscribes the machine when fits run side by side.
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Situation,
    Source,
    register_family,
)
from turbotab.core.models.tuning import Dimension, TuningDecl

SMALL_N = 500
TINY_N = 200

STANDARD = "default"  # the standard smallest leaf: scikit-learn's own, never capped
STANDARD_LEAF = 20  # HistGradientBoosting*'s min_samples_leaf default
LEAF_SHARE = 20  # a searched leaf holds at most a twentieth of the plan's units (RECIPES §4.1)
_TUNABILITY = "Probst, Boulesteix & Bischl 2019"

TUNING = TuningDecl(
    "search",
    dimensions=(
        Dimension("learning_rate", "how big each correction step is", "learning rate", 0.01, 0.3,
                  "log", source=_TUNABILITY),
        Dimension("max_leaf_nodes", "how many leaves each tree may grow", "leaves per tree", 4,
                  128, "log_int", source=_TUNABILITY),
        Dimension("min_samples_leaf", "the fewest rows a leaf may hold", "minimum leaf size", 2,
                  200, "log_int", source=_TUNABILITY),
        Dimension("l2_regularization", "how strongly each leaf's value is pulled toward zero",
                  "L2 regularization", 1e-3, 10.0, "log", source=_TUNABILITY),
        Dimension("max_features", "the share of columns each split tries", "feature subsampling",
                  0.3, 1.0, "linear", source=_TUNABILITY),
        Dimension("max_iter", "how many trees", "boosting rounds", 25, 500, "log_int",
                  source=_TUNABILITY, active="without_early_stopping"),
    ),
    standard={"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": STANDARD,
              "l2_regularization": 0.0, "max_features": 1.0, "max_iter": 100},
    standard_source="scikit-learn's defaults",
    # Sobol candidates stop early from an effective size of 1,500 (the plan's); the standard ones
    # by scikit-learn's own rule, above 10,000 of the plan's rows, at its own 100 trees and
    # patience.
    early_stopping={"param": "max_iter", "rounds": 1000, "patience_param": "n_iter_no_change",
                    "patience": 20, "share": 0.1},
    space_version="boosted_trees/1",
    structural=("loss",),
    reason="Its settings are searched inside every training fold, with scikit-learn's defaults "
           "always among the candidates.",
)


def leaf_cap(*, plan: Any = None, n_units: int) -> int:
    """The most rows a searched smallest leaf may hold (RECIPES §4.1): a twentieth of the plan's
    count, at least 1, the same in every fit the plan makes. For a measured outcome that count is
    the plan's units (``n_plan``); for a yes/no or class outcome the plan counts events or the
    rarest class, which is not what a leaf holds, and stores no count of units, so it is the plan's
    rows (``plan_rows``: rows, not units, when rows repeat; open for a ruling). Without a plan, the
    fit's own units."""
    if plan is not None:
        n = int(plan.n_plan) if plan.unit == "units" else int(plan.plan_rows)
    else:
        n = int(n_units)
    return max(1, n // LEAF_SHARE)


_CONTROLLER: Any = None


def _openmp(threads: int) -> Any:
    """A context holding OpenMP to ``threads`` in the calling thread (threadpoolctl's limit; one
    controller, built once, so a prediction pays microseconds for it)."""
    global _CONTROLLER
    if _CONTROLLER is None:
        from threadpoolctl import ThreadpoolController

        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER.limit(limits=int(threads), user_api="openmp")


def _with_threads(base: type) -> Any:
    """``base.__init__`` with one more keyword, ``n_threads`` (default 1), its signature
    scikit-learn's parameter list: ``get_params``, ``set_params`` and ``clone`` see every one of
    ``base``'s parameters, whatever the installed version declares, and ``n_threads``."""
    signature = inspect.signature(base.__init__)

    def __init__(self: Any, *, n_threads: int = 1, **params: Any) -> None:
        base.__init__(self, **params)
        self.n_threads = n_threads

    threads = inspect.Parameter("n_threads", inspect.Parameter.KEYWORD_ONLY, default=1)
    __init__.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*signature.parameters.values(), threads])
    return __init__


def _with_thread_constraint(base: type) -> dict[str, Any]:
    """``base``'s parameter constraints and ``n_threads``': a whole number, at least 1."""
    from numbers import Integral

    from sklearn.utils._param_validation import Interval

    return {**base._parameter_constraints,  # type: ignore[attr-defined]
            "n_threads": [Interval(Integral, 1, None, closed="left")]}


class _Pinned:
    """Every fit and prediction at ``n_threads`` OpenMP threads (RECIPES §4.7)."""

    n_threads: int

    def fit(self, X: Any, y: Any, sample_weight: Any = None, *, X_val: Any = None,
            y_val: Any = None, sample_weight_val: Any = None) -> Any:
        with _openmp(self.n_threads):
            return super().fit(X, y, sample_weight, X_val=X_val,  # type: ignore[misc]
                               y_val=y_val, sample_weight_val=sample_weight_val)

    def predict(self, X: Any) -> Any:
        with _openmp(self.n_threads):
            return super().predict(X)  # type: ignore[misc]


class PinnedHistGradientBoostingRegressor(_Pinned, HistGradientBoostingRegressor):
    """scikit-learn's ``HistGradientBoostingRegressor`` on ``n_threads`` OpenMP threads."""

    __init__ = _with_threads(HistGradientBoostingRegressor)
    _parameter_constraints = _with_thread_constraint(HistGradientBoostingRegressor)


class PinnedHistGradientBoostingClassifier(_Pinned, HistGradientBoostingClassifier):
    """scikit-learn's ``HistGradientBoostingClassifier`` on ``n_threads`` OpenMP threads."""

    __init__ = _with_threads(HistGradientBoostingClassifier)
    _parameter_constraints = _with_thread_constraint(HistGradientBoostingClassifier)

    def predict_proba(self, X: Any) -> Any:
        with _openmp(self.n_threads):
            return super().predict_proba(X)

    def decision_function(self, X: Any) -> Any:
        with _openmp(self.n_threads):
            return super().decision_function(X)


def hgb_ensemble(model: Any) -> Any:
    """A fitted ``HistGradientBoostingRegressor`` / ``…Classifier`` as an ``explain.TreeEnsemble``
    on its raw (margin) scale: the outcome's for a regression, the log-odds for a binary
    classifier, one output per class for several. Its node tables are scikit-learn's own
    (``TreePredictor.nodes``), and a row goes left when its value is ``<=`` the threshold."""
    from turbotab.core.models.explain import TreeEnsemble, leaf_paths

    base = np.atleast_1d(np.asarray(model._baseline_prediction, dtype=float).ravel())
    outputs = len(model._predictors[0])
    tables = [[iteration[k].nodes for iteration in model._predictors] for k in range(outputs)]
    trees = [[leaf_paths(nodes) for nodes in per_output] for per_output in tables]
    return TreeEnsemble(base=base, trees=trees, tables=tables, rule="le", scale="margin")


class BoostedTrees(FamilyBase):
    key = "boosted_trees"
    label = "Boosted trees"
    inductive_bias = ("Effects are step functions that can bend and interact; many shallow trees "
                      "each correct the last.")
    strengths = (
        "Finds curves and interactions without being told.",
        "Uses rows with missing values as they are.",
    )
    cautions = (
        "No coefficients: effects are read from curves.",
        "Overfits easily and needs many rows.",
        "Validated by cross-validation: Harrell's bootstrap overstates a near-interpolating "
        "learner's performance.",
    )
    needs_scaling = False
    handles_missing = True
    # Near-interpolating: its apparent AUC approaches 1, and Harrell's bootstrap leaves about 0.2 of
    # AUC uncorrected (the repair round's replication; Coley et al. 2023), so it is not applied.
    bootstrap_optimism = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it).
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="PinnedHistGradientBoostingRegressor; "
                                  "PinnedHistGradientBoostingClassifier (scikit-learn's "
                                  "histogram gradient boosting on the plan's threads)",
                        seed_policy="random_state from the tuning plan's seed (derive_seed of the "
                                    "split's seed, family and space version), 0 without a plan; "
                                    "it seeds the column subsampling, and the binning subsample "
                                    "above 200,000 rows")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = True
    # Under inference it gives no table: its curves describe the declared exposure.
    inference_decl = InferenceDecl(table="description_only",
                                   words="the gradient-boosted tree model")
    invariances = ("monotone_per_column",)
    bias_terms = (Named(plain="Many shallow trees, each fit to what the trees before it missed.",
                        known_as="gradient boosting", source=Source("friedman2001")),)
    curve_shape = "piecewise_constant"
    # No sourced closed form for any of these (MODEL_FAMILY_CONTRACT C7).
    complexity = (
        Knob(setting="max_iter", more_means="more_flexible"),  # rounds
        Knob(setting="learning_rate", more_means="more_flexible"),
        Knob(setting="max_leaf_nodes", more_means="more_flexible"),
        Knob(setting="min_samples_leaf", more_means="simpler"),
        Knob(setting="l2_regularization", more_means="simpler"),
    )
    output = "margin"
    raw_scale = CLASS_SCALES
    attribution = "trees"
    architecture = ("trees",)
    review_lenses = ("shared",)
    sources = (Source("friedman2001"),)
    consequence = "Many shallow trees: finds curves and interactions; gives no coefficients."
    tuning = TUNING
    defaults_version = "2"  # RT-5a: tuned; "Try both" for blanks joins with RT-3

    def methods_label(self, task: Task | None) -> str:
        return "gradient-boosted trees"

    def trees(self, step: Any) -> Any:
        """Its fitted model step's trees, for TreeSHAP and the tree view (C10)."""
        return hgb_ensemble(step)

    def settings(self, values: Mapping[str, Any], *, task: str, n_units: int, n_rows: int,
                 y: Any = None, Z: Any = None, plan: Any = None) -> dict[str, Any]:
        """A candidate's values as the estimator's parameters on one fit: the standard leaf
        (``"default"``) as scikit-learn's own 20, never capped; any other leaf capped at
        :func:`leaf_cap`; the plan's seed as ``random_state`` (0 without a plan); the plan's
        threads as ``n_threads`` (1 without a plan). It resolves a candidate's values once, and
        its output is held as it is (``TunedPipeline.at``): fed back in, a resolved standard leaf
        of 20 is a number and would be capped like a searched one."""
        out = dict(values)
        leaf = out.get("min_samples_leaf")
        if isinstance(leaf, str):
            if leaf != STANDARD:
                raise ValueError(f"min_samples_leaf {leaf!r} is not a number of rows or "
                                 f"{STANDARD!r}")
            out["min_samples_leaf"] = STANDARD_LEAF
        elif leaf is not None:
            out["min_samples_leaf"] = min(int(leaf), leaf_cap(plan=plan, n_units=n_units))
        out["random_state"] = int(plan.seed) if plan is not None else 0
        out["n_threads"] = int(plan.threads) if plan is not None else 1
        return out

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        if task == "regression":
            return PinnedHistGradientBoostingRegressor(random_state=0)
        return PinnedHistGradientBoostingClassifier(random_state=0)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        return ("Histogram gradient boosting",
                "Its settings are searched inside every training fold, scikit-learn's defaults "
                "(100 trees of at most 31 leaves) always among them and used alone below an "
                "effective size of 300 (units, events or the rarest class's count). Searched "
                "settings stop early from an effective size of 1,500, the defaults above 10,000 "
                "training rows, holding back a tenth of the fold's units, the latest when the "
                "folds follow time.")

    def assess(self, s: Situation) -> Assessment:
        concerns: list[str] = []
        fit = "good"
        score = 3.0 if s.n_rows >= 2000 else 2.0
        if s.n_rows < SMALL_N:
            concerns.append(f"{s.n_rows:,} rows is small for boosted trees: they overfit easily "
                            f"and their cross-validated score is noisy.")
            fit = "poor" if s.n_rows < TINY_N else "fair"
            score = 0.5 if s.n_rows < TINY_N else 1.0
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: most splits will "
                            f"fit noise.")
            fit = "poor"
            score = min(score, 1.0)
        if s.purpose == "inference":
            concerns.append("No coefficients or intervals: effects are read from curves instead.")
            fit = "fair" if fit == "good" else fit
            score = min(score, 1.0)
        return Assessment(score, fit, tuple(concerns))


BOOSTED_TREES = register_family(BoostedTrees())
