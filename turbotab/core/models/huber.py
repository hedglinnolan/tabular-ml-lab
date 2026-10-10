"""``huber``: robust linear regression, Huber's M-estimator (RECIPES_AND_TUNING §2.2;
MODEL_FAMILY_CONTRACT §3.2; WAVE_C6A_PLAN RT-5c).

A straight-line model in which rows far from the line count less. The model step,
:class:`RobustLinearRegression`, is a thin scikit-learn wrapper around statsmodels' ``RLM`` with
Huber's norm at the threshold t = 1.345, which keeps 95% of least squares' efficiency under normal
errors (Huber 1964), fitted by iteratively reweighted least squares with the scale re-estimated at
every step as the median absolute deviation, median|r| / 0.6745 (Holland & Welsch 1977, the constant
MASS::rlm uses), and no penalty.

**Prediction only** in v2 (RECIPES §2.2, §5): it declares ``purposes = ("prediction",)`` and gives no
inference table. Nothing keeps it out of inference by its key (V2X_SEAMS row 8): the shelf reads the
declaration in :meth:`Huber.assess`.

**Not tuned by cross-validation.** Squared error, the primary score, would walk the threshold toward
no robustness at all, so the threshold is the standard 1.345 unless it is set by hand
(``TuningDecl`` kind ``"none"``, by hand ``t`` in [1, 3]).
"""
from __future__ import annotations

import functools
import math
import warnings
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Identity,
    Knob,
    Named,
    Prior,
    Situation,
    Source,
    coefficient_rows,
    register_family,
)
from turbotab.core.models.tuning import Dimension, TuningDecl

T_STANDARD = 1.345  # Huber (1964): 95% of least squares' efficiency under normal errors
# The MAD's divisor: the standard normal's upper quartile as Holland & Welsch (1977) and MASS::rlm
# write it. statsmodels' own ``mad`` divides by the unrounded 0.67449, which moves the fit in its
# sixth significant digit, so the scale here is computed as they write it.
MAD_CONSTANT = 0.6745
SCALE = "MAD"  # the only scale declared (RECIPES §4.1's table: "MAD scale; no penalty")
PENALTY = "none"
# RECIPES §2.2's shelf table: "Robust linear | 1.5 (numeric only; after linear on ties)". Linear
# registers first, so a tie goes to linear; rows that fall short cost a point, as they cost linear.
SCORE = 1.5
SHORT_ROWS_COST = 1.0
T_RANGE = (1, 3)  # the threshold set by hand (RECIPES §4.1)
# A MAD at or below this share of the outcome's size is floating-point dust left by an exact fit
# (most rows on the line), not a scale: such a fit has converged.
EXACT_FIT_SCALE = 1e3 * np.finfo(float).eps
CAUTION = ("Its slopes match least squares' when the errors are spread the same way at every value "
           "of the predictors. With skewed errors its predictions shift from the mean toward the "
           "median, and squared error charges it for that.")


@functools.cache
def _rlm() -> type:
    """statsmodels' ``RLM`` with the scale re-estimated as median|r| / :data:`MAD_CONSTANT`."""
    from statsmodels.robust.robust_linear_model import RLM

    class HuberRLM(RLM):
        def _estimate_scale(self, resid: Any) -> float:
            return float(np.median(np.abs(resid))) / MAD_CONSTANT

    return HuberRLM


class RobustLinearRegression(RegressorMixin, BaseEstimator):
    """Huber's M-estimator of a linear model with an intercept, fitted by statsmodels' ``RLM``.

    ``t``: Huber's threshold, in units of the scale; ``scale`` and ``penalty`` are stated, not
    chosen (only ``"MAD"`` and ``"none"`` are fitted). The fit iterates until no standardized
    residual moves by ``tol``, or ``max_iter`` steps, and warns when it stops short.

    Fitted: ``coef_``, ``intercept_`` (the linear closed forms read them), ``scale_`` (the final MAD
    scale), ``n_iter_`` and ``converged_`` (C8, IRLS convergence)."""

    def __init__(self, t: float = T_STANDARD, scale: str = SCALE, penalty: str = PENALTY,
                 max_iter: int = 100, tol: float = 1e-10) -> None:
        self.t = t
        self.scale = scale
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol

    def _check(self) -> float:
        try:
            t = float(self.t)
        except (TypeError, ValueError):
            t = math.nan
        low, high = T_RANGE
        if not math.isfinite(t) or not low <= t <= high:
            raise ValueError(f"Huber's threshold t is set by hand in [{low}, {high}], "
                             f"not {self.t!r}")
        if self.scale != SCALE:
            raise ValueError(f"the scale is re-estimated as the {SCALE} at every step; "
                             f"{self.scale!r} is not declared")
        if self.penalty != PENALTY:
            raise ValueError(f"robust linear regression is fitted with no penalty; "
                             f"{self.penalty!r} is not declared")
        return t

    def fit(self, X: Any, y: Any) -> "RobustLinearRegression":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.utils.validation import validate_data
        from statsmodels.robust.norms import HuberT

        t = self._check()
        X, y = validate_data(self, X, y, y_numeric=True, ensure_min_samples=2, dtype=float)
        exog = np.column_stack([np.ones(len(y)), X])
        with warnings.catch_warnings():
            # statsmodels warns of a scale of 0 (an exact fit): that fit is the answer, not a fault
            warnings.simplefilter("ignore")
            result = _rlm()(np.asarray(y, dtype=float), exog, M=HuberT(t=t)).fit(
                maxiter=int(self.max_iter), tol=float(self.tol), conv="sresid")
        params = np.asarray(result.params, dtype=float)
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]
        self.scale_ = float(result.scale)
        history = result.fit_history
        self.n_iter_ = int(history["iteration"])
        moves = history["sresid"]
        exact = self.scale_ <= EXACT_FIT_SCALE * float(np.max(np.abs(y)))
        self.converged_ = bool(exact or (
            len(moves) >= 3 and np.all(np.abs(np.asarray(moves[-1]) - np.asarray(moves[-2]))
                                       < float(self.tol))))
        if not self.converged_:
            warnings.warn(f"Robust linear regression did not converge in {self.max_iter} "
                          f"reweighting steps; its coefficients are the last step's.",
                          ConvergenceWarning, stacklevel=2)
        return self

    def predict(self, X: Any) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted, validate_data

        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, dtype=float)
        return X @ self.coef_ + self.intercept_


class Huber(FamilyBase):
    key = "huber"
    label = "Robust linear regression"
    tasks = ("regression",)
    inductive_bias = "Straight-line effects fitted so that far-off rows count less."
    strengths = (
        "A few far-off values pull its line less than they pull least squares'.",
        "One coefficient per predictor, read as a straight-line effect.",
    )
    cautions = (
        CAUTION,
        "Made for prediction: it gives no intervals for an effect.",
    )
    needs_scaling = False  # its fit rescales itself: the scale is re-estimated from the residuals
    handles_missing = False
    linear_in_values = True
    # MODEL_FAMILY_CONTRACT §1 (§3.2's row for it).
    identity = Identity(kind="estimator", library="statsmodels",
                        estimator="RobustLinearRegression (a scikit-learn wrapper around "
                                  "statsmodels' RLM with Huber's norm)")
    purposes = ("prediction",)
    predicts = True
    flexible = False
    bootstrap_optimism = True
    sample_efficiency = (Prior(says="As many rows per predictor as least squares; under normal "
                                    "errors it is 95% as efficient, under heavy tails more.",
                               kind="convention", source=Source("huber1964")),)
    bias_terms = (Named(plain="Rows far from the line count less, so a few wild values cannot "
                              "drag it far.",
                        known_as="Huber loss", source=Source("huber1964")),)
    invariances = ("linear_maps",)
    curve_shape = "straight"
    # A larger threshold lets far-off rows pull the line more, toward least squares.
    complexity = (Knob(setting="t", more_means="more_flexible", source=Source("huber1964")),)
    diagnostics = ("convergence",)
    output = "value"
    raw_scale = MappingProxyType({"regression": "value"})
    attribution = "linear"
    architecture = ("equation",)
    review_lenses = ("shared",)
    sources = (Source("huber1964"), Source("holland1977irls"))
    cost_model = "cross_product"  # every reweighting step solves a weighted least squares
    tuning = TuningDecl(
        "none",
        by_hand=(Dimension("t", "how far a row may sit before it counts less", "Huber threshold",
                           *T_RANGE, "linear", source="Huber 1964; Holland & Welsch 1977"),),
        standard={"t": T_STANDARD}, standard_source="Huber 1964",
        fixed={"scale": SCALE, "penalty": PENALTY}, space_version="huber/1",
        reason="Squared error would walk the threshold toward no robustness at all, so it is set, "
               "not searched.")
    consequence = ("Straight-line effects where rows far from the line count less: steadier "
                   "against outliers; for prediction only.")

    def methods_label(self, task: Task | None) -> str:
        return "robust linear regression (Huber's M-estimator)"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return RobustLinearRegression()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        return ("Robust linear regression",
                f"Fits the straight-line effect of every column so that far-off rows count less "
                f"(Huber's threshold {T_STANDARD} unless set by hand, the scale re-estimated from "
                f"the median absolute deviation at every step, no penalty).")

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        return coefficient_rows([str(f) for f in model.feature_names_in_], model.coef_,
                                intercept=model.intercept_)

    def assess(self, s: Situation) -> Assessment:
        from turbotab.core.models.sample_size import prediction_concern, prediction_minimum

        if s.purpose is not None and s.purpose not in self.purposes:
            served = " and ".join(self.purposes)
            return Assessment(0.0, "poor", (f"Made for {served} only: it gives no intervals for "
                                            f"an effect, so it is not offered for {s.purpose}.",))
        if s.n_features >= s.n_rows:
            return Assessment(0.0, "poor", (f"{s.n_features:,} predictors for {s.n_rows:,} rows: "
                                            f"its line has no unique solution.",))
        parameters = s.n_parameters if s.n_parameters is not None else s.n_features
        minimum = prediction_minimum(s.task, parameters, n_rows=s.n_rows, n_events=s.n_events,
                                     outcome_mean=s.outcome_mean, outcome_sd=s.outcome_sd)
        said = prediction_concern(minimum, s.n_rows) if minimum is not None else None
        if said:
            return Assessment(SCORE - SHORT_ROWS_COST, "fair", (said,))
        return Assessment(SCORE, "good")


HUBER = register_family(Huber())
