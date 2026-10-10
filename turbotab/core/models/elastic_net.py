"""``elastic_net``: a penalized linear model whose penalty is tuned by an inner cross-validation.

The inner cross-validation is the estimator's own (:class:`PooledElasticNetCV` /
:class:`PooledLogisticRegressionCV`), so it only ever splits the rows the estimator is fit on:
inside an outer fold, that is the training part of the fold and nothing else.

**The same penalty on every platform** (RECIPES_AND_TUNING §4.2, "Score" and "Choice"; §4.7). On
the WP7 prediction fixture macOS (Accelerate) and Linux (OpenBLAS) chose different penalties for
the same rows, and the intercept moved by up to 6.4. The inner folds differed: they were drawn
from row keys hashed from the values' exact bits, which differ in the last bit between the two
(``inner_cv.row_keys`` now hashes single precision). Two weaker links are closed here, so that the
same folds give the same penalty, for a number, a yes/no and classes alike:

* **Each point of the inner curve is exact** (:mod:`turbotab.core.models.exact_path`).
  scikit-learn's ``ElasticNetCV`` computes it by coordinate descent to a tolerance of 10⁻⁴,
  3 × 10⁻⁵ to 8 × 10⁻⁴ off the exact loss on that fixture and on NHANES, while the runner-up
  penalty can sit 2 × 10⁻⁶ above the best: a summation order that stops the solver a sweep earlier
  or later could move the choice. Its ``LogisticRegressionCV`` (``saga``, its only elastic-net
  solver) was 2 × 10⁻³ off at the family's tolerance, and no tighter tolerance fixed it. Both
  paths are now solved by an active-set method that ends at the exact solution, warm-started from
  the strongest penalty.
* **The choice is the pooled inner loss, rounded.** scikit-learn takes the first minimum of the
  folds' unweighted mean, unrounded; :func:`lowest_rounded` chooses over the pooled loss (squared
  error, or log loss) rounded to 10⁻⁹ of the smallest, a tie going to the earlier mix and the
  larger penalty.

A matrix too wide for the exact path (more than :data:`EXACT_MAX_COLUMNS` columns, or
:data:`EXACT_MAX_COEFFICIENTS` logistic coefficients) keeps scikit-learn's solvers under the same
choice; its penalty is not promised to be the same on every platform (as ``wide`` says of p > n).
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import Assessment, FamilyBase, Situation, coefficient_rows, register_family
from turbotab.core.models.base import CLASS_SCALES, Identity, InferenceDecl, Knob, Source
# The choice moved to the tuning plan (RT-1a), the one rule every tuned family chooses by; the names
# stay here for the readers that import them from the elastic net.
from turbotab.core.models.tuning import LOSS_PRECISION  # noqa: F401 - re-exported
from turbotab.core.models.tuning import choose as lowest_rounded

L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
# saga was slow on the full default grid (≈ 150 s per multiclass fit on 17,000 NHANES rows); this
# grid stops short of the nearly unpenalized end, which the linear family already covers.
LOGISTIC_L1_RATIOS = (0.2, 0.6, 1.0)
LOGISTIC_CS = tuple(float(c) for c in np.logspace(-3, 1, 8))
# Coordinate descent's tolerance where it still runs (more than EXACT_MAX_COLUMNS columns, and the
# wide fit's float64 twin; scikit-learn's is 10⁻⁴): at 10⁻¹² the pooled inner losses are within
# 2.2 × 10⁻¹² of the exact solution's on the WP7 fixture and on NHANES, far inside the 10⁻⁹ the
# choice rounds to; collinear columns need up to 100,000 sweeps to get there.
SOLVER_TOL = 1e-12
SOLVER_MAX_ITER = 100_000
# The exact path's reach, timed at one thread for a whole inner cross-validation: least squares on
# 2,000 rows × 500 columns, 12 s for six mixes × 100 penalties × five folds (coordinate descent at
# 10⁻⁴: 18 s); logistic on 1,000 × 200, 0.4 s (saga at 10⁻³: 4.3 s to 28 s by table), on four
# classes × 150 columns 19 s (saga: 41 s), eight classes × 70, 9 s to 30 s (saga: 44 s to 61 s;
# 17 s on the verifier's table). Beyond these a linear solve per step costs more than a sweep.
EXACT_MAX_COLUMNS = 500
EXACT_MAX_COEFFICIENTS = 600


def inner_folds(n_rows: int) -> int:
    return 5 if 100 <= n_rows <= 5000 else 3


def _fold_weights(folds: list[tuple[Any, Any]], sample_weight: Any, n: int) -> np.ndarray:
    """Each inner validation fold's rows, or their weights' sum: what pools the folds' losses."""
    if sample_weight is None:
        return np.array([len(test) for _, test in folds], dtype=float)
    w = np.broadcast_to(np.asarray(sample_weight, dtype=float), (n,))
    return np.array([w[test].sum() for _, test in folds], dtype=float)


class PooledElasticNetCV(ElasticNetCV):
    """``ElasticNetCV`` over an exact path, choosing its penalty and mix on the pooled inner loss,
    rounded.

    scikit-learn's grid is kept (each mix's own, from the penalty that zeros every coefficient down
    to a thousandth of it). Each inner fold's path is solved exactly by feature-sign search
    (``exact_path.gaussian_path``), from the largest penalty down, each point started from the
    last; the refit at the chosen penalty likewise. Each fold's error counts by its rows (or its
    rows' weights): the squared error summed over every inner validation row, divided by their
    number. :func:`lowest_rounded` then chooses over the grid, mixes in their order and penalties
    largest first, so a tie goes to the earlier mix and the larger penalty. ``pooled_loss_`` holds
    the curve, one row per mix.

    Where the exact path does not apply (weights, positive coefficients, more than
    :data:`EXACT_MAX_COLUMNS` columns, or a subclass that sets ``exact = False``) scikit-learn's
    coordinate descent computes the paths at the given tolerance and only the choice changes: the
    model is refit at it where it differs from scikit-learn's. The folds are drawn once, so the
    paths and the weights see the same splits.
    """

    exact = True

    def _exact_applies(self, X: Any, sample_weight: Any) -> bool:
        from scipy import sparse

        return bool(self.exact and sample_weight is None and not self.positive
                    and not sparse.issparse(X) and np.ndim(X) == 2
                    and np.shape(X)[1] <= EXACT_MAX_COLUMNS
                    and isinstance(self.precompute, (bool, str)))

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any) -> "PooledElasticNetCV":
        from sklearn.model_selection import check_cv

        if self._exact_applies(X, sample_weight):
            return self._fit_exact(X, y, **params)
        given = self.cv
        folds = list(check_cv(given).split(X, y, **params))
        self.cv = folds
        try:
            super().fit(X, y, sample_weight=sample_weight, **params)
        finally:
            self.cv = given
        mixes = np.atleast_1d(np.asarray(self.l1_ratio, dtype=float))
        weights = _fold_weights(folds, sample_weight, len(np.asarray(y)))
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

    def _fit_exact(self, X: Any, y: Any, **params: Any) -> "PooledElasticNetCV":
        from sklearn.model_selection import check_cv
        from sklearn.utils.validation import column_or_1d, validate_data

        from turbotab.core.models.exact_path import alpha_grid, gaussian_path

        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        y = column_or_1d(y, warn=True).astype(np.float64)
        folds = list(check_cv(self.cv).split(X, y, **params))
        mixes = np.atleast_1d(np.asarray(self.l1_ratio, dtype=float))
        if isinstance(self.alphas, Integral):
            grids = np.array([alpha_grid(X, y, float(mix), n_alphas=int(self.alphas), eps=self.eps,
                                         fit_intercept=self.fit_intercept) for mix in mixes])
        else:
            grids = np.tile(np.sort(np.asarray(self.alphas, dtype=float))[::-1], (len(mixes), 1))
        errors = np.empty((len(mixes), grids.shape[1], len(folds)))
        for m, mix in enumerate(mixes):
            for f, (train, test) in enumerate(folds):
                coefs, intercepts, _ = gaussian_path(X[train], y[train], grids[m], float(mix),
                                                     fit_intercept=self.fit_intercept)
                residuals = X[test] @ coefs + intercepts - y[test][:, None]
                errors[m, :, f] = (residuals ** 2).mean(axis=0)
        weights = _fold_weights(folds, None, len(y))
        self.mse_path_ = np.squeeze(errors)
        self.pooled_loss_ = (errors * weights).sum(axis=2) / weights.sum()
        mix, step = np.unravel_index(lowest_rounded(self.pooled_loss_), self.pooled_loss_.shape)
        self.alpha_, self.l1_ratio_ = grids[mix, step], mixes[mix]
        coefs, intercepts, _ = gaussian_path(X, y, [self.alpha_], float(self.l1_ratio_),
                                             fit_intercept=self.fit_intercept)
        self.coef_, self.intercept_ = coefs[:, 0], float(intercepts[0])
        # scikit-learn's layout: one grid per mix from its own computation, one row for one mix
        self.alphas_ = (grids if len(mixes) > 1 else grids[0]) if isinstance(self.alphas, Integral) \
            else grids[0]
        self.dual_gap_, self.n_iter_ = 0.0, 0
        return self


class PooledLogisticRegressionCV(LogisticRegressionCV):
    """``LogisticRegressionCV`` over an exact path, choosing its ``C`` and mix on the pooled inner
    log loss, rounded.

    Each inner fold's path runs from the strongest penalty (the smallest ``C``) to the weakest,
    each fit started from the last (``exact_path.logistic_path``: proximal Newton, its quadratic
    solved exactly); the refit at the chosen values likewise. Each fold's log loss counts by its
    rows (or its rows' weights): the log loss summed over every inner validation row, divided by
    their number. :func:`lowest_rounded` chooses over the mixes in their order and the ``C`` values
    smallest first, so a tie goes to the earlier mix and the larger penalty. The score is always
    the log loss. ``pooled_loss_`` holds the curve, one row per mix; ``scores_``, ``coefs_paths_``
    and ``n_iter_`` are laid out as scikit-learn's are with ``use_legacy_attributes=False``.

    Where the exact path does not apply (class weights, a sparse matrix, more than
    :data:`EXACT_MAX_COEFFICIENTS` coefficients) scikit-learn computes the paths with its solver
    (``saga``) at the given tolerance, and only the choice changes, as above.

    More than two classes under a pure lasso (mix 1.0) have no unique coefficients: one number
    added to a feature's coefficient in every class changes no probability, and no penalty while
    it stays between that feature's two middle coefficients. Whichever solver ran, each feature is
    reported at the middle of that interval (``exact_path.middle_of_ties``), so the reported
    coefficients do not follow the last bit of the data.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any
            ) -> "PooledLogisticRegressionCV":
        from scipy import sparse
        from sklearn.model_selection import check_cv
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import check_classification_targets
        from sklearn.utils.validation import validate_data

        from turbotab.core.models.exact_path import log_loss_rows, logistic_path

        Cs = np.sort(np.logspace(-4, 4, int(self.Cs)) if isinstance(self.Cs, Integral)
                     else np.asarray(self.Cs, dtype=float))  # the strongest penalty first
        mixes = np.atleast_1d(np.asarray(
            (0.0,) if self.l1_ratios is None or isinstance(self.l1_ratios, str) else self.l1_ratios,
            dtype=float))
        if sparse.issparse(X) or self.class_weight is not None:
            return self._fit_given(X, y, sample_weight, Cs, **params)
        X, y = validate_data(self, X, y, dtype=np.float64, order="C")
        check_classification_targets(y)
        encoder = LabelEncoder().fit(y)
        self.classes_ = encoder.classes_
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes in the data, but the "
                             f"data contains only one class: {self.classes_[0]}.")
        rows = 1 if n_classes == 2 else n_classes
        if rows * X.shape[1] > EXACT_MAX_COEFFICIENTS:
            return self._fit_given(X, y, sample_weight, Cs, **params)
        codes = encoder.transform(y)
        weights = (np.ones(len(codes)) if sample_weight is None
                   else np.broadcast_to(np.asarray(sample_weight, dtype=float), (len(codes),)))
        folds = list(check_cv(self.cv, y, classifier=True).split(X, y, **params))
        p = X.shape[1]
        paths = np.zeros((len(folds), len(mixes), len(Cs), rows, p + int(self.fit_intercept)))
        losses = np.zeros((len(folds), len(mixes), len(Cs)))
        steps = np.zeros((len(folds), len(mixes), len(Cs)), dtype=np.int64)
        for f, (train, test) in enumerate(folds):
            for m, mix in enumerate(mixes):
                out, steps[f, m], _ = logistic_path(
                    X[train], codes[train], n_classes, Cs, float(mix),
                    sample_weight=None if sample_weight is None else weights[train],
                    fit_intercept=self.fit_intercept)
                for c, (W, b) in enumerate(out):
                    paths[f, m, c, :, :p] = W
                    if self.fit_intercept:
                        paths[f, m, c, :, p] = b
                    losses[f, m, c] = float(weights[test] @ log_loss_rows(X[test], codes[test], W, b))
        fold_weight = _fold_weights(folds, weights, len(codes))
        self.scores_ = -losses / fold_weight[:, None, None]
        self.coefs_paths_, self.n_iter_ = paths, steps
        self.pooled_loss_ = losses.sum(axis=0) / fold_weight.sum()
        m, c = np.unravel_index(lowest_rounded(self.pooled_loss_), self.pooled_loss_.shape)
        (W, b), = logistic_path(X, codes, n_classes, [Cs[c]], float(mixes[m]),
                                sample_weight=sample_weight, fit_intercept=self.fit_intercept)[0]
        self.Cs_, self.l1_ratios_ = Cs, mixes
        self.C_, self.l1_ratio_ = float(Cs[c]), float(mixes[m])
        self.coef_, self.intercept_ = W, (b if self.fit_intercept else np.zeros(rows))
        return self

    def _fit_given(self, X: Any, y: Any, sample_weight: Any, Cs: np.ndarray, **params: Any
                   ) -> "PooledLogisticRegressionCV":
        """scikit-learn's paths and solver; the choice as above, refit where it differs."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import check_cv

        folds = list(check_cv(self.cv, y, classifier=True).split(X, y, **params))
        given = {k: getattr(self, k) for k in ("cv", "Cs", "scoring", "use_legacy_attributes")}
        self.set_params(cv=folds, Cs=list(Cs), scoring="neg_log_loss", use_legacy_attributes=False)
        try:
            super().fit(X, y, sample_weight=sample_weight, **params)
        finally:
            self.set_params(**given)
        scores = np.asarray(self.scores_, dtype=float)  # (folds, mixes, Cs): each fold's mean
        weight = _fold_weights(folds, sample_weight, len(np.asarray(y)))
        self.pooled_loss_ = -(scores * weight[:, None, None]).sum(axis=0) / weight.sum()
        m, c = np.unravel_index(lowest_rounded(self.pooled_loss_), self.pooled_loss_.shape)
        C, mix = float(np.asarray(self.Cs_)[c]), float(np.asarray(self.l1_ratios_)[m])
        if C != float(self.C_) or mix != float(self.l1_ratio_):
            model = LogisticRegression(C=C, l1_ratio=mix, solver=self.solver, tol=self.tol,
                                       max_iter=self.max_iter, random_state=self.random_state,
                                       fit_intercept=self.fit_intercept,
                                       class_weight=self.class_weight)
            model.fit(X, y, sample_weight=sample_weight)
            self.C_, self.l1_ratio_ = C, mix
            self.coef_, self.intercept_ = model.coef_, model.intercept_
        if len(self.classes_) > 2 and mix == 1.0:
            from turbotab.core.models.exact_path import middle_of_ties

            self.coef_ = middle_of_ties(self.coef_)  # as the exact path reports a pure lasso
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
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it).
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="PooledElasticNetCV (Float32ElasticNetCV past the exact path's "
                                  "columns when wide); PooledLogisticRegressionCV",
                        seed_policy="random_state 0, fixed")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="shrunk_no_intervals")
    invariances = ("column_scale",)
    curve_shape = "straight"
    # The penalty's strength; its lasso-ridge mix moves sparsity and grouping, not one direction of
    # complexity, so it is not a knob here. Each knob is the estimator's grid (``alphas``, the
    # logistic fit's inverse penalties ``Cs``), from which its inner cross-validation chooses the
    # fitted ``alpha_`` or ``C_``; the formula, the ridge part's, takes ``alpha_`` and
    # ``l1_ratio_`` (C7).
    complexity = (
        Knob(setting="alphas", more_means="simpler", formula="elastic_net_ridge_part",
             source=Source("hastie2009", "§3.4.1, eqs. 3.47 and 3.50, for the ridge part")),
        Knob(setting="Cs", more_means="more_flexible"),
    )
    output = "margin"
    raw_scale = CLASS_SCALES
    attribution = "linear"
    architecture = ("equation", "shrinkage")
    review_lenses = ("shared",)
    consequence = ("A penalized linear model: shrinks correlated nutrients together, tuned inside "
                   "training folds.")

    def methods_label(self, task: Task | None) -> str:
        return "elastic net"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        cv = inner_folds(n_rows)
        if task == "regression":
            from turbotab.core.models.wide import for_wide  # p > n: float32 and threads

            return for_wide(PooledElasticNetCV(l1_ratio=list(L1_RATIOS), cv=cv,
                                               max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL),
                            n_rows, n_features)
        return PooledLogisticRegressionCV(
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

__all__ = ["ELASTIC_NET", "EXACT_MAX_COEFFICIENTS", "EXACT_MAX_COLUMNS", "ElasticNet", "L1_RATIOS",
           "LOSS_PRECISION", "PooledElasticNetCV", "PooledLogisticRegressionCV", "SOLVER_MAX_ITER",
           "SOLVER_TOL", "inner_folds", "lowest_rounded"]
