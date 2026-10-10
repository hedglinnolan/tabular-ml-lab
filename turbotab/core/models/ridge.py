"""``ridge``: straight-line effects all shrunk toward zero together by one L2 penalty (RT-5b;
RECIPES_AND_TUNING §2.2, §4.1; MODEL_FAMILY_CONTRACT §3.2).

**The penalty, per row.** The family tunes λ, the penalty per row on the scaled columns, along a
path of 50 points from 10² down to 10⁻⁵ (RECIPES §4.1), so one λ means the same pull on any number
of rows and the grid reaches near zero, where wide data's best penalty can sit (Kobak et al. 2020).
Each fit turns λ into its estimator's own parameter on its own rows (:meth:`Ridge.settings`):

* a number: scikit-learn's ``Ridge`` minimizes ‖y − Zβ − b‖² + α‖β‖², so ``alpha`` = nλ;
* a yes/no outcome or classes: ``LogisticRegression`` with ``l1_ratio`` 0 (all L2) minimizes
  C·Σᵢ ℓᵢ + ½‖W‖², so ``C`` = 1/(nλ), the same Σᵢ ℓᵢ + (nλ/2)‖W‖². The intercepts are never
  penalized in either.

**The path** (:meth:`Ridge.path`): one split's whole grid. For a number it is the closed form on
the centered matrix (the intercept unpenalized) through one SVD, Zc = UDVᵀ,
β(λ) = V diag(dⱼ/(dⱼ² + nλ)) Uᵀ(y − ȳ), b = ȳ − z̄ᵀβ (ESL §3.4.1, eq. 3.47). For classes it is
L-BFGS at each point, warm-started from the stronger penalty before it, to the tolerance the
refit uses, so the path's point and the refit at that λ are the same fit.

**Row weights are relative.** Given ``weights``, the path scales them to mean 1, as glmnet scales
its observation weights to sum to n, so the loss is Σᵢ (wᵢ/w̄)ℓᵢ beside the penalty nλ: weights that
sum to a population total move no λ, and the refit at λ is the path's point when it is given
``sample_weight`` w/w̄.

**The shelf** (RECIPES §2.6, conventions): 2.0 from 2,000 training rows, after boosted trees' 3.0
and the elastic net's 2.5; 3.0 when predictors outnumber rows, below the elastic net's 4.0 and the
screened net's 3.8; 1.0 below 2,000 rows, under the 1.5 that least squares and the screened elastic
net hold there, so the two families the reference journeys pick today stay first.

**What waits.** The penalty is chosen only by RT-1b's engine, which sets each fit's ``alpha`` or
``C`` through :meth:`Ridge.settings`; :meth:`Ridge.build` alone carries scikit-learn's default of 1,
so ``describe`` is true of a fit only once the engine runs it, and this family lands with the engine
or after it. Full coding ("every level its own column", RECIPES §2.2) is the recipe's
``onehot_drop``, declared here as ``onehot_drop = None`` (RECIPES §2.4), which the shared one-hot
steps read (RT-5f) until RT-2's recipe holds it. The shrinkage-path view reads the elastic net's
tuning record (RT-5f); ridge declares the equation only.
``updating`` is empty: a uniform calibration-slope shrinkage of a ridge is not sourced
(WAVE_C6A_PLAN §5, ruling 7).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge as RidgeRegression

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Prior,
    Situation,
    Source,
    coefficient_rows,
    register_family,
)
from turbotab.core.models.tuning import Dimension, PathFit, TuningDecl

# L-BFGS's tolerance and reach for logistic ridge, in the path and the refit alike: at 10⁻¹⁰ the
# coefficients sit within 10⁻⁶ of the penalized likelihood's minimum (scikit-learn's 10⁻⁴ does not).
LOGISTIC_TOL = 1e-10
LOGISTIC_MAX_ITER = 10_000
WIDE_SCORE = 3.0  # RECIPES §2.6: at p ≥ n, below the elastic net's 4.0 (a convention)
BASE_SCORE = 2.0  # RECIPES §2.6, from FROM_ROWS training rows (a convention)
FROM_ROWS = 2_000  # RECIPES §2.6: the order it states holds "at 2,000 training rows or more"
SMALL_SCORE = 1.0  # below FROM_ROWS: under the journeys' second picks there, 1.5 (a convention)
FEW_ROWS = 50  # as the elastic net: below this the inner folds choose λ from very few rows

LAMBDA = Dimension(
    "lambda", "how strongly coefficients are pulled toward zero", "penalty (λ)", 1e-5, 1e2,
    "log", source="Probst, Boulesteix & Bischl 2019; Kobak et al. 2020", points=50)
TUNING = TuningDecl(
    "path", dimensions=(LAMBDA,), space_version="ridge/1",
    reason="The pull is chosen inside the training rows, so the score includes that choice.")

METHODS_LABELS = {
    "regression": "ridge regression",
    "binary": "logistic ridge regression",
    "multiclass": "multinomial logistic ridge regression",
    "ordinal": "multinomial logistic ridge regression, which ignores the levels' order",
}


def _logistic(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", tol=LOGISTIC_TOL,
                              max_iter=LOGISTIC_MAX_ITER)


class Ridge(FamilyBase):
    key = "ridge"
    label = "Ridge"
    inductive_bias = ("Straight-line effects all shrunk toward zero together; correlated predictors "
                      "share weight; none is dropped.")
    strengths = (
        "Stable when predictors are correlated: they share the weight.",
        "Keeps every predictor, so nothing is dropped by chance.",
    )
    cautions = (
        "Shrunk coefficients have no confidence intervals.",
        "It keeps every predictor, so it does not say which ones matter.",
    )
    needs_scaling = True
    handles_missing = False
    linear_in_values = True
    # MODEL_FAMILY_CONTRACT §1 (§3.2's column for it).
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="Ridge (Cholesky); LogisticRegression (L-BFGS, l1_ratio 0)",
                        seed_policy="none: deterministic (a closed form; L-BFGS from zero)")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True  # path families are bootstrapped (RECIPES §4.3)
    inference_decl = InferenceDecl(table="shrunk_no_intervals")
    sample_efficiency = (
        Prior(says="In the worst case, the rows it needs grow at least in step with the number of "
                   "irrelevant columns.",
              kind="bound", source=Source("ng2004l1l2", "the lower bound for rotationally "
                                                         "invariant learners")),
    )
    bias_terms = (
        Named(plain="Ridge barely shrinks the main patterns in your columns and shrinks the minor "
                    "ones hard.",
              known_as="L2 shrinkage", source=Source("hastie2009", "§3.4.1")),
    )
    invariances = ("rotation_after_scaling",)
    curve_shape = "straight"
    complexity = (
        Knob(setting="lambda", more_means="simpler", formula="ridge",
             source=Source("hastie2009", "§3.4.1, eqs. 3.47 and 3.50, with α = nλ")),
    )
    output = "margin"
    updating = ()
    raw_scale = CLASS_SCALES
    attribution = "linear"
    architecture = ("equation",)
    review_lenses = ("shared",)
    sources = (
        Source("hastie2009", "§3.4.1"),
        Source("kobak2020ridge", "the best penalty on wide data can be near zero"),
        Source("probst2019tunability", "glmnet gained most from tuning"),
    )
    cost_model = "cross_product"  # Cholesky factors the p × p (or n × n) cross-product
    tuning = TUNING
    onehot_drop = None  # RECIPES §2.2, §2.4: every level its own column (full coding)
    consequence = ("A penalized straight-line model that shrinks every effect and keeps every "
                   "predictor.")

    def methods_label(self, task: Task | None) -> str:
        return METHODS_LABELS.get(task, "ridge regression")

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        if task == "regression":
            return RidgeRegression(alpha=1.0, solver="cholesky")
        return _logistic()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        label = "Ridge regression" if task == "regression" else "Logistic ridge regression"
        return label, ("Pulls every coefficient toward zero by one penalty, chosen along a path of "
                       "50 strengths within the training rows it is given; keeps every predictor.")

    def settings(self, values: Mapping[str, Any], *, task: str, n_units: int, n_rows: int,
                 y: Any = None, Z: Any = None, plan: Any = None) -> dict[str, Any]:
        """λ per row as the estimator's own parameter on this fit's ``n_rows``: ``alpha`` = nλ, or
        ``C`` = 1/(nλ) (the module docstring)."""
        rest = {k: v for k, v in values.items() if k != "lambda"}
        penalty = float(n_rows) * float(values["lambda"])
        if task == "regression":
            return {**rest, "alpha": penalty}
        return {**rest, "C": 1.0 / penalty}

    def path(self, Z: Any, y: Any, grid: Mapping[str, Any], *, task: str,
             weights: Any = None) -> PathFit:
        """Every λ of ``grid`` on these rows (the module docstring), as a ``tuning.PathFit`` with one
        outer setting. ``weights``, when given, weight each row's loss as ``sample_weight`` does
        once scaled to mean 1 (the module docstring); the penalty stays nλ, n the rows."""
        Z = np.asarray(Z, dtype=float)
        lams = np.asarray(grid["lambda"], dtype=float)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.mean()
        if task == "regression":
            coefs, intercepts = _least_squares_path(Z, np.asarray(y, dtype=float), lams, weights)
        else:
            coefs, intercepts = _logistic_path(Z, np.asarray(y), lams, weights)
        return PathFit(values=lams[None, :], coefs=coefs[None], intercepts=intercepts[None])

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        """Coefficients per unit of each matrix column (the scaling undone); no intervals."""
        model = pipeline[-1]
        features = [str(f) for f in model.feature_names_in_]
        coef = np.atleast_2d(np.asarray(model.coef_, dtype=float))
        intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=float))
        scaler = pipeline.named_steps.get("scale")
        if scaler is not None:
            coef = coef / scaler.scale_
            intercept = intercept - coef @ scaler.mean_
        classes = list(getattr(model, "classes_", [])) or None
        return coefficient_rows(features, coef, intercept=intercept,
                                classes=classes if classes and len(classes) > 2 else None)

    def assess(self, s: Situation) -> Assessment:
        """RECIPES §2.6's conventions (the module docstring): 2.0 from 2,000 training rows, 1.0
        below, and 3.0 when predictors outnumber rows; under inference "fair" at most 2.0, as the
        elastic net."""
        concerns: list[str] = []
        fit = "good"
        score = BASE_SCORE if s.n_rows >= FROM_ROWS else SMALL_SCORE
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: it keeps every one, "
                            f"each shrunk.")
            score = WIDE_SCORE
        if s.purpose == "inference":
            concerns.append("Penalized coefficients are shrunk and carry no confidence intervals.")
            fit, score = "fair", min(score, BASE_SCORE)
        if s.n_rows < FEW_ROWS:
            concerns.append(f"With {s.n_rows:,} rows the inner cross-validation picks the penalty "
                            f"from very few rows.")
            fit = "fair"
        return Assessment(score, fit, tuple(concerns))


def _least_squares_path(Z: np.ndarray, y: np.ndarray, lams: np.ndarray,
                        weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """(G, 1, p) coefficients and (G, 1) intercepts by one SVD of the (weighted) centered matrix."""
    n = Z.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    zbar = w @ Z / w.sum()
    ybar = float(w @ y / w.sum())
    root = np.sqrt(w)
    U, d, Vt = np.linalg.svd(root[:, None] * (Z - zbar), full_matrices=False)
    Uty = U.T @ (root * (y - ybar))
    shrink = d[None, :] / (d[None, :] ** 2 + n * lams[:, None])  # (G, r)
    coefs = (shrink * Uty[None, :]) @ Vt  # (G, p)
    intercepts = ybar - coefs @ zbar
    return coefs[:, None, :], intercepts[:, None]


def _logistic_path(Z: np.ndarray, y: np.ndarray, lams: np.ndarray,
                   weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """(G, K, p) coefficients and (G, K) intercepts: L-BFGS at C = 1/(nλ), strongest first, each
    started from the point before it."""
    n = Z.shape[0]
    model = _logistic()
    model.set_params(warm_start=True)
    coefs, intercepts = [], []
    for lam in lams:
        model.set_params(C=1.0 / (n * float(lam)))
        model.fit(Z, y, sample_weight=weights)
        coefs.append(np.array(model.coef_, dtype=float))
        intercepts.append(np.array(model.intercept_, dtype=float))
    return np.asarray(coefs), np.asarray(intercepts)


RIDGE = register_family(Ridge())

__all__ = ["FROM_ROWS", "LOGISTIC_MAX_ITER", "LOGISTIC_TOL", "METHODS_LABELS", "RIDGE", "Ridge",
           "TUNING"]
