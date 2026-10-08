"""``elastic_net``: a penalized linear model whose penalty is tuned by an inner cross-validation.

The inner cross-validation is the estimator's own (:class:`PooledElasticNetCV` /
``LogisticRegressionCV``), so it only ever splits the rows the estimator is fit on: inside an
outer fold, that is the training part of the fold and nothing else.

**The same penalty on every platform** (RECIPES_AND_TUNING §4.2, "Score" and "Choice"; §4.7). On
the WP7 prediction fixture macOS (Accelerate) and Linux (OpenBLAS) chose different penalties for
the same rows, and the intercept moved by up to 6.4. The inner folds differed: they were drawn
from row keys hashed from the values' exact bits, which differ in the last bit between the two
(``inner_cv.row_keys`` now hashes single precision). Two weaker links are closed here, so that the
same folds give the same penalty. scikit-learn's ``ElasticNetCV`` computes each point of the curve
by coordinate descent to a tolerance of 10⁻⁴, 3 × 10⁻⁵ to 8 × 10⁻⁴ off the exact loss on that
fixture and on NHANES, while the runner-up penalty can sit 2 × 10⁻⁶ above the best: a summation
order that stops the solver a sweep earlier or later could move the choice. And it takes the first
minimum of the folds' unweighted mean, unrounded. :class:`PooledElasticNetCV` converges the paths
to :data:`SOLVER_TOL` and chooses by :func:`lowest_rounded` over the pooled inner loss.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import Assessment, FamilyBase, Situation, coefficient_rows, register_family

L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
# saga is slow on the full default grid (≈ 150 s per multiclass fit on 17,000 NHANES rows); this
# grid stops short of the nearly unpenalized end, which the linear family already covers.
LOGISTIC_L1_RATIOS = (0.2, 0.6, 1.0)
LOGISTIC_CS = tuple(float(c) for c in np.logspace(-3, 1, 8))
# Coordinate descent's tolerance on every path fit and the refit (scikit-learn's is 10⁻⁴): the
# largest coefficient update relative to the largest coefficient, then the duality gap relative to
# ‖y‖². At 10⁻¹² the pooled inner losses are within 2.2 × 10⁻¹² of the exact solution's (the WP7
# fixture; NHANES at 400, 2,000 and 13,600 rows with squared nutrients beside them), far inside
# the 10⁻⁹ the choice rounds to, for two to three times the solver's time (0.1 s to 0.3 s a fit).
# Collinear columns need more sweeps to get there: at 5,000 up to 55 path points stopped short on
# those NHANES fits, and the fit stage would have reported it as a concern; 100,000 never bound.
SOLVER_TOL = 1e-12
SOLVER_MAX_ITER = 100_000
LOSS_PRECISION = 1e-9  # pooled losses are rounded to this share of the smallest before the argmin


def inner_folds(n_rows: int) -> int:
    return 5 if 100 <= n_rows <= 5000 else 3


def lowest_rounded(losses: Any, precision: float = LOSS_PRECISION) -> int:
    """The flat index of the lowest loss, each rounded first to a multiple of ``precision`` times
    the smallest; equal rounded losses go to the lowest index (RECIPES_AND_TUNING §4.2, "Choice").

    Rounding makes a near-tie a tie, and a tie the earlier candidate's: on a penalty path, the
    larger penalty. Without it, two losses closer than floating-point noise trade places with the
    order of a sum."""
    values = np.asarray(losses, dtype=float).ravel()
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    scale = float(np.min(np.abs(values[finite]))) or 1.0
    rounded = np.where(finite, np.round(values / (precision * scale)), np.inf)
    return int(np.argmin(rounded))  # the first of equal minima


class PooledElasticNetCV(ElasticNetCV):
    """``ElasticNetCV`` choosing its penalty and mix on the pooled inner loss, rounded.

    scikit-learn's paths, grid and refit are kept; only the choice changes. scikit-learn averages
    the folds' mean squared errors and takes the first minimum unrounded. Here each fold's error
    counts by its rows (or its rows' weights): the squared error summed over every inner
    validation row, divided by their number. :func:`lowest_rounded` then chooses over the grid,
    mixes in their order and penalties largest first, so a tie goes to the earlier mix and the
    larger penalty. When that differs from scikit-learn's choice the model is refit at it, as
    scikit-learn refits. ``pooled_loss_`` holds the curve, one row per mix.

    The folds are drawn once, so the paths and the weights see the same splits.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any) -> "PooledElasticNetCV":
        from sklearn.model_selection import check_cv

        given = self.cv
        folds = list(check_cv(given).split(X, y, **params))
        self.cv = folds
        try:
            super().fit(X, y, sample_weight=sample_weight, **params)
        finally:
            self.cv = given
        mixes = np.atleast_1d(np.asarray(self.l1_ratio, dtype=float))
        if sample_weight is None:
            weights = np.array([len(test) for _, test in folds], dtype=float)
        else:
            w = np.broadcast_to(np.asarray(sample_weight, dtype=float), (len(np.asarray(y)),))
            weights = np.array([w[test].sum() for _, test in folds], dtype=float)
        errors = np.asarray(self.mse_path_, dtype=float).reshape(len(mixes), -1, len(folds))
        self.pooled_loss_ = (errors * weights).sum(axis=2) / weights.sum()
        grid = np.atleast_2d(np.asarray(self.alphas_, dtype=float))
        if grid.shape[0] != len(mixes):  # explicit alphas: one grid for every mix
            grid = np.repeat(grid, len(mixes), axis=0)
        mix, step = np.unravel_index(lowest_rounded(self.pooled_loss_), self.pooled_loss_.shape)
        alpha, l1_ratio = float(grid[mix, step]), float(mixes[mix])
        if alpha == float(self.alpha_) and l1_ratio == float(self.l1_ratio_):
            return self
        model = self._get_estimator()
        model.set_params(**{k: v for k, v in self.get_params().items() if k in model.get_params()})
        model.set_params(alpha=alpha, l1_ratio=l1_ratio)
        if isinstance(self.precompute, str) and self.precompute == "auto":
            model.precompute = False  # as scikit-learn refits
        model.fit(X, y, sample_weight=sample_weight)
        self.alpha_, self.l1_ratio_ = alpha, l1_ratio
        self.coef_, self.intercept_ = model.coef_, model.intercept_
        self.dual_gap_, self.n_iter_ = model.dual_gap_, model.n_iter_
        return self


class ElasticNet(FamilyBase):
    key = "elastic_net"
    label = "Elastic net"
    inductive_bias = ("Straight-line effects shrunk toward zero; few predictors matter, correlated "
                      "ones share weight.")
    strengths = (
        "Works with more predictors than rows.",
        "Tunes its own penalty by cross-validation on training rows.",
    )
    cautions = (
        "Shrunk coefficients have no confidence intervals.",
        "Which of several correlated predictors it keeps is arbitrary.",
    )
    needs_scaling = True
    handles_missing = False
    linear_in_values = True

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        cv = inner_folds(n_rows)
        if task == "regression":
            from turbotab.core.models.wide import for_wide  # p > n: float32 and threads

            return for_wide(PooledElasticNetCV(l1_ratio=list(L1_RATIOS), cv=cv,
                                               max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL),
                            n_rows, n_features)
        from sklearn.linear_model import LogisticRegressionCV

        return LogisticRegressionCV(
            Cs=list(LOGISTIC_CS), l1_ratios=LOGISTIC_L1_RATIOS, solver="saga", cv=cv,
            scoring="neg_log_loss", max_iter=1000, tol=1e-3, use_legacy_attributes=False,
            random_state=0,
        )

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        label = "Elastic net regression" if task == "regression" else "Penalized logistic regression"
        return label, ("Chooses the penalty's strength and its lasso-ridge mix by an inner "
                       "cross-validation within the training rows it is given.")

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
        concerns: list[str] = []
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: the predictors it "
                            f"keeps will change from sample to sample.")
            if s.purpose == "inference":  # AUDIT_REPORT ME-18: the caveat holds at p ≥ n too
                concerns.append("Penalized coefficients are shrunk and carry no confidence intervals.")
                return Assessment(2.0, "fair", tuple(concerns))
            return Assessment(4.0, "good", tuple(concerns))
        fit = "good"
        score = 2.5
        if s.purpose == "inference":
            concerns.append("Penalized coefficients are shrunk and carry no confidence intervals.")
            fit, score = "fair", 2.0
        if s.n_rows < 50:
            concerns.append(f"With {s.n_rows:,} rows the inner cross-validation picks the penalty "
                            f"from very few rows.")
            fit = "fair"
        return Assessment(score, fit, tuple(concerns))


ELASTIC_NET = register_family(ElasticNet())

__all__ = ["ELASTIC_NET", "ElasticNet", "L1_RATIOS", "LOSS_PRECISION", "PooledElasticNetCV",
           "SOLVER_MAX_ITER", "SOLVER_TOL", "inner_folds", "lowest_rounded"]
