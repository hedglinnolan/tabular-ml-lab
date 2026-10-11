"""``elastic_net``: a penalized linear model whose penalty and lasso-ridge mix are chosen by the
shared path search, nested in every fit (RT-5f; RECIPES_AND_TUNING §2.2, §4.1, §4.3; F4 and F5).

**The path search.** The family is a path family, one declaration per task (:data:`TUNING`): a
number searches the mix ρ over (0.1, 0.5, 0.7, 0.9, 0.95, 1) and, for each, 100 ratios r from 1 down
to 10⁻³ (log-spaced) of each split's own λ_max; a yes/no outcome or classes, ρ over (0.2, 0.6, 1)
and 8 ratios. The engine (``tuning.TunedPipeline``) draws the inner splits from each fit's own rows,
refits the steps before the model on each split's training rows (the screened family's screen among
them, so no screen sees the rows that score its penalty: F4), runs :meth:`ElasticNet.path` on them,
scores every grid point on the pooled inner loss (survey-weighted under the population answer, the
models fit unweighted), chooses by ``tuning.choose`` (a tie to the earlier mix and the larger
penalty) and refits the plain estimator at r·λ_max of the fit's own rows
(:meth:`ElasticNet.settings`).
The chosen values are on the fit's ``tuning_`` record, not on the estimator: it has no ``alpha_``.

**λ_max** (:func:`lambda_max`), on the rows at hand: the smallest penalty per row at which every
coefficient is zero, read from the optimality conditions at zero with the intercepts fit (glmnet's
start, Friedman, Hastie & Tibshirani 2010): max_j |Σᵢ wᵢ(zᵢⱼ − z̄ⱼ)(yᵢ − ȳ)| / (ρ Σᵢ wᵢ) for a number
(scikit-learn's ``alpha_max``); for classes the largest over features and classes of
|Σᵢ wᵢ zᵢⱼ(yᵢₖ − ȳₖ)| / (ρ Σᵢ wᵢ), yᵢₖ the indicator of class k (a yes/no outcome: the second class's
alone, the one its coefficients are the log-odds of).

**The penalty per row** (F5). scikit-learn's ``ElasticNet`` minimizes (1/2n)‖y − Zβ − b‖² +
αρ‖β‖₁ + α(1 − ρ)/2‖β‖², already per row, so α = r·λ_max. ``LogisticRegression`` minimizes
C Σᵢ ℓᵢ + ρ‖W‖₁ + (1 − ρ)/2‖W‖², which is n·C times Σᵢ ℓᵢ/n + λ(ρ‖W‖₁ + (1 − ρ)/2‖W‖²) at
C = 1/(nλ), so C = 1/(n·r·λ_max): the grid follows each fit's rows, where the fixed C grid it had
meant a weaker penalty per row on more rows.

**Exact** (:mod:`turbotab.core.models.exact_path`). The path and the refit are solved exactly
wherever the exact path reaches: the refit's estimators, :class:`ExactElasticNet` and
:class:`ExactLogisticRegression`, are scikit-learn's own fit by the exact path at their one penalty,
so the refit at a grid point is that point of the path. Past :data:`EXACT_MAX_COLUMNS` columns (or
:data:`EXACT_MAX_COEFFICIENTS` logistic coefficients) the path and the refit run the same solver
instead: coordinate descent to :data:`SOLVER_TOL`, in single precision at ``wide.WIDE_TOL`` when
the matrix has more columns than rows (``wide.coordinate_path``), and ``saga`` for the logistic
loss; the refit then agrees with the warm-started path to that tolerance, and the penalty is not
promised to be the same on every platform. Both read the matrix they are handed, so a screen that
leaves the screened net within the exact path's reach is fit exactly in the inner splits and in
the refit alike, whatever width the table was built for.

**Full coding** (RECIPES §2.2, §2.4): ``onehot_drop = None``, read by the shared one-hot steps
(``pipeline.shared_steps``), gives every level its own column, so the penalty and the predictions do
not depend on which level sorts first. Under a pure lasso a category's coefficients are tied: its
indicators mark every row once, so they can move together along one direction that changes no
prediction and no penalty, and a solver stops at whichever end its column order reaches first.
They are reported at the middle of that set (:func:`middle_of_category_ties`): for two levels,
plus and minus half the contrast, the elastic net's limit as its mix nears 1. More than two classes
under a pure lasso are also reported at the middle of the classes' tied set
(``exact_path.middle_of_ties``), by either solver, in the path and in the refit.

**The selection step's elastic net** (``variable_selection``) keeps its own estimators,
:class:`PooledElasticNetCV` and :class:`PooledLogisticRegressionCV`: ``ElasticNetCV`` and
``LogisticRegressionCV`` over the same exact path, choosing on the pooled inner loss rounded to 10⁻⁹
of the smallest (:func:`lowest_rounded`), on the CV estimators' own grids. The same penalty on every
platform was won for them first (RECIPES §4.2, "Score" and "Choice"; §4.7): on the WP7 prediction
fixture macOS (Accelerate) and Linux (OpenBLAS) chose different penalties for the same rows when
each point of the curve came from coordinate descent at 10⁻⁴, ``saga`` at 10⁻³, and the folds'
unrounded mean.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import ElasticNet as ElasticNetRegression
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import Assessment, FamilyBase, Situation, coefficient_rows, register_family
from turbotab.core.models.base import CLASS_SCALES, Identity, InferenceDecl, Knob, Source
# The choice moved to the tuning plan (RT-1a), the one rule every tuned family chooses by; the names
# stay here for the readers that import them from the elastic net.
from turbotab.core.models.tuning import LOSS_PRECISION  # noqa: F401 - re-exported
from turbotab.core.models.tuning import Dimension, PathFit, TuningDecl
from turbotab.core.models.tuning import choose as lowest_rounded

L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
# saga was slow on the full default grid (≈ 150 s per multiclass fit on 17,000 NHANES rows); this
# grid stops short of the nearly unpenalized end, which the linear family already covers.
LOGISTIC_L1_RATIOS = (0.2, 0.6, 1.0)
# The ratios r of each split's own λ_max (RECIPES §4.1): from 1, where every coefficient is zero,
# down to RATIO_LOW, log-spaced; 100 points for a number, 8 for classes (glmnet's 100 down to 10⁻³
# of λ_max; the logistic grid kept to 8 for saga's cost past the exact path).
RATIO_LOW = 1e-3
RATIO_POINTS = 100
LOGISTIC_RATIO_POINTS = 8
# saga past the exact path's reach, as the family has always run it there (path and refit alike)
SAGA_TOL = 1e-3
SAGA_MAX_ITER = 1000
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


# ── the family's path search (RT-5f) ─────────────────────────────────────────

_TINY = float(np.finfo(np.float64).resolution)  # λ_max with nothing to explain (as alpha_grid)
_RATIO_SOURCE = "Probst, Boulesteix & Bischl 2019"


def _relative(weights: Any, n: int) -> np.ndarray:
    """Row weights scaled to mean 1 (glmnet scales its observation weights to sum to n), so the
    loss per row and the penalty per row keep their meaning; every row 1 without weights."""
    if weights is None:
        return np.ones(n)
    w = np.broadcast_to(np.asarray(weights, dtype=float), (n,))
    return w / w.mean()


def _indicators(task: str, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(classes in sorted order, each row's class code, the indicator columns the coefficients'
    rows model): the second class alone for two classes, one column per class otherwise."""
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).ravel()
    if len(classes) < 2:
        raise ValueError("A penalized logistic fit needs rows of at least two classes.")
    Y = np.zeros((len(codes), len(classes)))
    Y[np.arange(len(codes)), codes] = 1.0
    return classes, codes, (Y[:, 1:] if len(classes) == 2 else Y)


def lambda_max(Z: Any, y: Any, l1_ratio: float, *, task: str, weights: Any = None) -> float:
    """The smallest penalty per row at which every coefficient is zero on these rows, the
    intercepts fit (the module docstring): the largest |Σᵢ wᵢ zᵢⱼ rᵢₖ| / (ρ Σᵢ wᵢ), rᵢₖ the residual
    of the intercept-only fit (y − ȳ, or each class's indicator less its share). ``weights`` as the
    path takes them (scaled to mean 1 here); at least floating point's resolution."""
    Z = np.asarray(Z, dtype=float)
    w = _relative(weights, Z.shape[0])
    if task == "regression":
        R = np.asarray(y, dtype=float).reshape(-1, 1)
    else:
        R = _indicators(task, y)[2]
    R = R - (w @ R) / w.sum()
    Zc = Z - (w @ Z) / w.sum()
    top = float(np.max(np.abs(Zc.T @ (w[:, None] * R)), initial=0.0)) / (w.sum() * float(l1_ratio))
    return max(top, _TINY)


def _exact_gaussian(Z: np.ndarray, y: np.ndarray, alphas: np.ndarray, mix: float,
                    w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(G, p) coefficients and (G,) intercepts by the exact path, the strongest penalty first;
    weighted rows as the unweighted problem on √w·(Z − z̄_w), √w·(y − ȳ_w)."""
    from turbotab.core.models.exact_path import gaussian_path

    if w is None:
        coefs, intercepts, _ = gaussian_path(Z, y, alphas, mix)
        return coefs.T, np.asarray(intercepts, dtype=float)
    zbar, ybar = w @ Z / w.sum(), float(w @ y / w.sum())
    root = np.sqrt(w)
    coefs, _, _ = gaussian_path(root[:, None] * (Z - zbar), root * (y - ybar), alphas, mix,
                                fit_intercept=False)
    return coefs.T, ybar - zbar @ coefs


def _gaussian(Z: np.ndarray, y: np.ndarray, alphas: np.ndarray, mix: float,
              w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """One mix's least-squares path: exact up to :data:`EXACT_MAX_COLUMNS` columns, otherwise
    coordinate descent (``wide.coordinate_path``)."""
    if Z.shape[1] <= EXACT_MAX_COLUMNS:
        return _exact_gaussian(Z, y, alphas, mix, w)
    from turbotab.core.models.wide import coordinate_path

    return coordinate_path(Z, y, alphas, mix, weights=w)


# ── a category's tied coefficients under a pure lasso (full coding) ──────────

TIE_PASSES = 50  # the most passes the class and category middles alternate for (two suffice)


def category_groups(Z: Any) -> list[np.ndarray]:
    """The runs of adjacent columns of ``Z`` that code one category with every level its own
    column (full coding, RECIPES §2.4): each column takes two values, its higher one marking its
    level (scaling keeps the order), no two mark the same row, and together they mark every row.
    The one-hot steps write a category's levels side by side, so a run is found by reading the
    columns in order; a partition of the rows the steps did not write side by side is not found,
    and its coefficients are reported as the solver left them."""
    Z = np.asarray(Z)
    if Z.ndim != 2 or not Z.shape[0] or not Z.shape[1]:
        return []
    low, high = Z.min(axis=0), Z.max(axis=0)
    two = (high > low) & np.all((Z == low) | (Z == high), axis=0)
    groups: list[np.ndarray] = []
    j, p = 0, Z.shape[1]
    while j < p:
        if not two[j]:
            j += 1
            continue
        marked = (Z[:, j] == high[j]).astype(np.int64)
        k, end = j + 1, None
        while k < p and two[k]:
            marked += Z[:, k] == high[k]
            k += 1
            if marked.max() > 1:
                break
            if marked.min() == 1:
                end = k
                break
        if end is None:
            j += 1
        else:
            groups.append(np.arange(j, end))
            j = end
    return groups


def _tied_middle(beta: np.ndarray, u: np.ndarray) -> float:
    """The step t to the middle of the pure lasso's tied set along ``u``: Σₗ |βₗ + t·uₗ| =
    Σₗ uₗ·|βₗ/uₗ + t| is lowest on the weighted median of −βₗ/uₗ (weights uₗ), an interval when
    the weights below one point make exactly half (two levels always do), else one point."""
    at = -beta / u
    order = np.argsort(at, kind="stable")
    at, w = at[order], u[order]
    total = float(w.sum())
    below = np.cumsum(w)
    j = int(np.searchsorted(below, total / 2 - 1e-9 * total))
    j = min(j, len(at) - 1)
    if abs(float(below[j]) - total / 2) <= 1e-9 * total and j + 1 < len(at):
        return float(at[j] + at[j + 1]) / 2
    return float(at[j])


def middle_of_category_ties(Z: Any, coef: Any, intercept: Any, *, multinomial: bool,
                            groups: list[np.ndarray] | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """A pure lasso's coefficients (``coef``: one row per class, or one row) at the middle of the
    set they are tied on when ``Z`` codes a category with every level its own column (RECIPES
    §2.4; :func:`category_groups`).

    A category's indicators mark every row once, so with uₗ = 1/(its column's two values' gap)
    Σₗ uₗ·zₗ is one constant c on every row: moving its coefficients by t·u (and the intercept by
    −t·c) changes no prediction, and under a pure lasso no penalty while t stays in the tied set
    (:func:`_tied_middle`). A solver stops at whichever end its column order reaches first, so
    the category's coefficients would follow which level sorts first (one level at 0, the other
    carrying the whole contrast). Each is moved to the middle of its set: for two levels, ±half
    the contrast, the elastic net's limit as its mix nears 1; the same point from any solution.
    With more than two classes the classes' own tie (``exact_path.middle_of_ties``: a number
    added to a column in every class) alternates with the categories' until neither moves. A
    row whose level the fit never saw (every indicator 0) is predicted at that middle too."""
    W = np.array(np.atleast_2d(coef), dtype=float)
    b = np.array(np.atleast_1d(intercept), dtype=float)
    Z = np.asarray(Z)
    groups = category_groups(Z) if groups is None else groups
    if not groups and not multinomial:
        return W, b
    parts = []
    for G in groups:
        low, high = Z[:, G].min(axis=0).astype(float), Z[:, G].max(axis=0).astype(float)
        u = 1.0 / (high - low)
        parts.append((G, u, 1.0 + float(u @ low)))
    for _ in range(TIE_PASSES):
        before = W.copy()
        for G, u, c in parts:
            for k in range(W.shape[0]):
                t = _tied_middle(W[k, G], u)
                W[k, G] += t * u
                b[k] -= t * c
        if multinomial:
            W = W - np.median(W, axis=0)
        if np.max(np.abs(W - before), initial=0.0) <= 1e-14 * max(1.0, float(np.abs(W).max(initial=0.0))):
            break
    return W, b


def _saga(C: float = 1.0, l1_ratio: float = 0.5) -> LogisticRegression:
    return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", tol=SAGA_TOL,
                              max_iter=SAGA_MAX_ITER, random_state=0)


def _logistic(Z: np.ndarray, codes: np.ndarray, n_classes: int, Cs: np.ndarray, mix: float,
              w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One mix's logistic path at ``Cs`` (the strongest penalty first): (G, K, p) coefficients and
    (G, K) intercepts, K = 1 for two classes; exact within :data:`EXACT_MAX_COEFFICIENTS`,
    otherwise ``saga`` warm-started along the path."""
    from turbotab.core.models.exact_path import logistic_path

    rows = 1 if n_classes == 2 else n_classes
    if rows * Z.shape[1] <= EXACT_MAX_COEFFICIENTS:
        out, _, _ = logistic_path(Z, codes, n_classes, Cs, mix, sample_weight=w)
        return (np.asarray([np.atleast_2d(W) for W, _ in out]),
                np.asarray([np.atleast_1d(b) for _, b in out]))
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    model = _saga(l1_ratio=mix).set_params(warm_start=True)
    coefs, intercepts = [], []
    for C in Cs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.set_params(C=float(C)).fit(Z, codes, sample_weight=w)
        coefs.append(np.atleast_2d(np.array(model.coef_, dtype=float)))
        intercepts.append(np.atleast_1d(np.array(model.intercept_, dtype=float)))
    return np.asarray(coefs), np.asarray(intercepts)


def _middle_of_path(Z: np.ndarray, coefs: np.ndarray, intercepts: np.ndarray, *,
                    multinomial: bool) -> tuple[np.ndarray, np.ndarray]:
    """A pure lasso's path, (G, K, p) and (G, K), each point at the middle of its tied set
    (:func:`middle_of_category_ties`), as the refit reports it."""
    groups = category_groups(Z)
    if not groups and not multinomial:
        return coefs, intercepts
    coefs, intercepts = np.array(coefs, dtype=float), np.array(intercepts, dtype=float)
    for g in range(coefs.shape[0]):
        coefs[g], intercepts[g] = middle_of_category_ties(Z, coefs[g], intercepts[g],
                                                          multinomial=multinomial, groups=groups)
    return coefs, intercepts


class ExactElasticNet(ElasticNetRegression):
    """scikit-learn's ``ElasticNet`` at its one ``alpha`` and ``l1_ratio``, fit as the family's
    path computes that point on the matrix it is handed, so the refit at a grid point is that
    point of the path: the exact path (``exact_path.gaussian_path``) up to
    :data:`EXACT_MAX_COLUMNS` columns, past them the path's own coordinate descent
    (``wide.coordinate_path``: single precision at ``wide.WIDE_TOL`` when the matrix has more
    columns than rows, else double precision at :data:`SOLVER_TOL`), which agrees with the
    warm-started path to the solver's tolerance. The choice reads the matrix the fit receives,
    not the width the pipeline was built for, so a screen or a full coding that changes the
    columns changes nothing here. Weights, positive coefficients, a sparse matrix, or a subclass
    that sets ``exact = False``, keep scikit-learn's own coordinate descent. A pure lasso is
    reported at the middle of a category's tied set (:func:`middle_of_category_ties`)."""

    exact = True

    def fit(self, X: Any, y: Any, sample_weight: Any = None, check_input: bool = True
            ) -> "ExactElasticNet":
        from scipy import sparse
        from sklearn.utils.validation import column_or_1d, validate_data

        from turbotab.core.models.exact_path import gaussian_path

        dense = not sparse.issparse(X) and np.ndim(X) == 2
        if not (self.exact and sample_weight is None and not self.positive and dense):
            super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        else:
            X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
            y = column_or_1d(y, warn=True).astype(np.float64)
            if X.shape[1] <= EXACT_MAX_COLUMNS:
                coefs, intercepts, _ = gaussian_path(X, y, [float(self.alpha)],
                                                     float(self.l1_ratio),
                                                     fit_intercept=self.fit_intercept)
                self.coef_ = coefs[:, 0]
                self.intercept_ = float(intercepts[0]) if self.fit_intercept else 0.0
            elif self.fit_intercept:
                from turbotab.core.models.wide import coordinate_path

                coefs, intercepts = coordinate_path(X, y, np.array([float(self.alpha)]),
                                                    float(self.l1_ratio))
                self.coef_, self.intercept_ = coefs[0], float(intercepts[0])
            else:
                return super().fit(X, y, check_input=check_input)
            self.dual_gap_, self.n_iter_ = 0.0, 0
        if float(self.l1_ratio) == 1.0 and self.fit_intercept and dense and not self.positive:
            Z = X if isinstance(X, np.ndarray) else np.asarray(X)
            W, b = middle_of_category_ties(Z, self.coef_, self.intercept_, multinomial=False)
            self.coef_, self.intercept_ = W[0].astype(np.asarray(self.coef_).dtype), float(b[0])
        return self


class ExactLogisticRegression(LogisticRegression):
    """scikit-learn's ``LogisticRegression`` at its one ``C`` and ``l1_ratio``, solved by the exact
    path (``exact_path.logistic_path``: proximal Newton), so the family's refit at a point of its
    path is that point. Class weights, a sparse matrix or more than :data:`EXACT_MAX_COEFFICIENTS`
    coefficients keep scikit-learn's solver (``saga``). Under a pure lasso the coefficients are
    reported at the middle of their tied set either way: more than two classes' (a number added
    to a column in every class, ``exact_path.middle_of_ties``) and a category's with every level
    its own column (:func:`middle_of_category_ties`)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "ExactLogisticRegression":
        from scipy import sparse
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import check_classification_targets
        from sklearn.utils.validation import validate_data

        from turbotab.core.models.exact_path import logistic_path

        mix = float(self.l1_ratio)
        if sparse.issparse(X) or self.class_weight is not None or np.ndim(X) != 2:
            return self._given(X, y, sample_weight)
        X, y = validate_data(self, X, y, dtype=np.float64, order="C")
        check_classification_targets(y)
        encoder = LabelEncoder().fit(y)
        n_classes = len(encoder.classes_)
        rows = 1 if n_classes == 2 else n_classes
        if n_classes < 2 or rows * X.shape[1] > EXACT_MAX_COEFFICIENTS:
            return self._given(X, y, sample_weight)
        out, steps, _ = logistic_path(X, encoder.transform(y), n_classes, [float(self.C)], mix,
                                      sample_weight=sample_weight,
                                      fit_intercept=self.fit_intercept)
        W, b = out[0]
        self.classes_ = encoder.classes_
        self.coef_ = np.atleast_2d(W)
        self.intercept_ = np.atleast_1d(b) if self.fit_intercept else np.zeros(rows)
        self.n_iter_ = np.asarray(steps, dtype=np.int32)
        return self._middle(X)

    def _given(self, X: Any, y: Any, sample_weight: Any) -> "ExactLogisticRegression":
        super().fit(X, y, sample_weight=sample_weight)
        from scipy import sparse

        return self if sparse.issparse(X) else self._middle(np.asarray(X))

    def _middle(self, X: np.ndarray) -> "ExactLogisticRegression":
        if float(self.l1_ratio) != 1.0:
            return self
        multinomial = len(self.classes_) > 2
        if not self.fit_intercept:  # a category's move needs the intercept to absorb it
            if multinomial:
                from turbotab.core.models.exact_path import middle_of_ties

                self.coef_ = middle_of_ties(self.coef_)
            return self
        self.coef_, self.intercept_ = middle_of_category_ties(X, self.coef_, self.intercept_,
                                                              multinomial=multinomial)
        return self


def _decl(mixes: tuple[float, ...], points: int) -> TuningDecl:
    return TuningDecl(
        "path",
        dimensions=(
            Dimension("l1_ratio", "how much of the pull drops predictors rather than shrinking "
                      "them", "lasso-ridge mix (ρ)", min(mixes), max(mixes), "choice",
                      choices=mixes, source=_RATIO_SOURCE),
            Dimension("ratio", "how strongly coefficients are pulled toward zero",
                      "penalty (a share of λ_max)", RATIO_LOW, 1.0, "log", source=_RATIO_SOURCE,
                      points=points),
        ),
        space_version="elastic_net/1",
        reason="The pull and its mix are chosen inside the training rows, so the score includes "
               "that choice.")


# RECIPES §4.1: one declaration per task; the pure-ridge end belongs to ridge.
TUNING_REGRESSION = _decl(L1_RATIOS, RATIO_POINTS)
TUNING_CLASSES = _decl(LOGISTIC_L1_RATIOS, LOGISTIC_RATIO_POINTS)
TUNING: Mapping[str, TuningDecl] = {"regression": TUNING_REGRESSION, "binary": TUNING_CLASSES,
                                    "multiclass": TUNING_CLASSES, "ordinal": TUNING_CLASSES}


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
                        estimator="ExactElasticNet (WideElasticNet when built wide: the "
                                  "path's coordinate descent past the exact path's columns); "
                                  "ExactLogisticRegression (saga past the exact path's "
                                  "coefficients)",
                        seed_policy="none: deterministic (the exact path); saga's random_state 0 "
                                    "past it")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="shrunk_no_intervals", words="the elastic net")
    invariances = ("column_scale",)
    curve_shape = "straight"
    # The penalty's strength, the path's own dimension: a ratio r of each fit's λ_max, which
    # ``settings`` turns into ``alpha`` (or ``C``); its lasso-ridge mix moves sparsity and
    # grouping, not one direction of complexity, so it is not a knob here. The formula, the ridge
    # part's, takes the fitted estimator's ``alpha`` and ``l1_ratio`` (C7).
    complexity = (
        Knob(setting="ratio", more_means="simpler", formula="elastic_net_ridge_part",
             source=Source("hastie2009", "§3.4.1, eqs. 3.47 and 3.50, for the ridge part")),
    )
    output = "margin"
    raw_scale = CLASS_SCALES
    attribution = "linear"
    architecture = ("equation", "shrinkage")
    review_lenses = ("shared",)
    tuning = TUNING
    # RECIPES §2.2 and §2.4: every level its own column (full coding), read by the shared one-hot
    # steps (``pipeline.onehot_drop``); RT-2's recipe moves it into the encoding slot.
    onehot_drop = None
    consequence = ("A penalized linear model: shrinks correlated nutrients together, tuned inside "
                   "training folds.")

    def methods_label(self, task: Task | None) -> str:
        return "elastic net"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        """The plain estimator the path search refits at its chosen values (``settings``); built
        at scikit-learn's own defaults, which no fit keeps."""
        if task == "regression":
            from turbotab.core.models.wide import for_wide  # built wide: WideElasticNet

            return for_wide(ExactElasticNet(max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL),
                            n_rows, n_features)
        return ExactLogisticRegression(C=1.0, l1_ratio=0.5, solver="saga", tol=SAGA_TOL,
                                       max_iter=SAGA_MAX_ITER, random_state=0)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        label = "Elastic net regression" if task == "regression" else "Penalized logistic regression"
        return label, ("Chooses the penalty's strength and its lasso-ridge mix along a path, by an "
                       "inner cross-validation within the training rows it is given.")

    def settings(self, values: Mapping[str, Any], *, task: str, n_units: int, n_rows: int,
                 y: Any = None, Z: Any = None, plan: Any = None) -> dict[str, Any]:
        """A grid point as the estimator's own parameters on this fit's rows (the module
        docstring): ``alpha`` = r·λ_max, or ``C`` = 1/(n·r·λ_max), λ_max read from the fit's own
        model matrix ``Z`` and outcome ``y``, with ``l1_ratio`` ρ."""
        if Z is None or y is None:
            raise ValueError("The elastic net's penalty is a share of its λ_max on the rows it is "
                             "fit on, so its settings need those rows' matrix and outcome.")
        rest = {k: v for k, v in values.items() if k != "ratio"}
        mix = float(values["l1_ratio"])
        penalty = float(values["ratio"]) * lambda_max(Z, y, mix, task=task)
        if task == "regression":
            return {**rest, "l1_ratio": mix, "alpha": penalty}
        return {**rest, "l1_ratio": mix, "C": 1.0 / (float(n_rows) * penalty)}

    def path(self, Z: Any, y: Any, grid: Mapping[str, Any], *, task: str,
             weights: Any = None) -> PathFit:
        """Every grid point on these rows, as a ``tuning.PathFit``: for each mix ρ (the outer
        settings, in order) the ratios r (strongest first) times these rows' own λ_max(ρ); least
        squares at α = r·λ_max, the logistic loss at C = 1/(n·r·λ_max), each mix's path warm-started
        from its strongest penalty. ``weights``, when given, weight each row's loss as
        ``sample_weight`` does once scaled to mean 1; the penalty stays per row. A pure lasso's
        points are at the middle of their tied sets (:func:`middle_of_category_ties`), as the
        refit reports them, so a row whose level these rows lack is scored at the same point
        whichever level sorts first."""
        Z = np.asarray(Z, dtype=float)
        n = Z.shape[0]
        mixes = np.atleast_1d(np.asarray(grid["l1_ratio"], dtype=float))
        ratios = np.atleast_1d(np.asarray(grid["ratio"], dtype=float))
        w = None if weights is None else _relative(weights, n)
        values = np.array([ratios * lambda_max(Z, y, float(mix), task=task, weights=w)
                           for mix in mixes])
        if task == "regression":
            y = np.asarray(y, dtype=float)

            def one(m: int) -> tuple[np.ndarray, np.ndarray]:
                return _gaussian(Z, y, values[m], float(mixes[m]), w)

            if Z.shape[1] > EXACT_MAX_COLUMNS and len(mixes) > 1:  # wide: a thread per mix
                from concurrent.futures import ThreadPoolExecutor

                from turbotab.core.models.wide import fit_threads

                with ThreadPoolExecutor(max_workers=min(fit_threads(), len(mixes))) as pool:
                    fits = list(pool.map(one, range(len(mixes))))
            else:
                fits = [one(m) for m in range(len(mixes))]
            coefs = np.asarray([c for c, _ in fits])[:, :, None, :]
            intercepts = np.asarray([b for _, b in fits])[:, :, None]
        else:
            classes, codes, _ = _indicators(task, y)
            weight = np.ones(n) if w is None else w
            fits = [_logistic(Z, codes, len(classes), 1.0 / (weight.sum() * values[m]),
                              float(mix), weight) for m, mix in enumerate(mixes)]
            coefs = np.asarray([c for c, _ in fits])
            intercepts = np.asarray([b for _, b in fits])
        multinomial = task != "regression" and coefs.shape[2] > 1
        for m in np.flatnonzero(mixes == 1.0):  # a pure lasso: its ties' middles, as the refit's
            coefs[m], intercepts[m] = _middle_of_path(Z, coefs[m], intercepts[m],
                                                      multinomial=multinomial)
        return PathFit(values=values, coefs=coefs, intercepts=intercepts)

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

__all__ = ["ELASTIC_NET", "EXACT_MAX_COEFFICIENTS", "EXACT_MAX_COLUMNS", "ElasticNet",
           "ExactElasticNet", "ExactLogisticRegression", "L1_RATIOS", "LOGISTIC_L1_RATIOS",
           "LOGISTIC_RATIO_POINTS", "LOSS_PRECISION", "PooledElasticNetCV",
           "PooledLogisticRegressionCV", "RATIO_LOW", "RATIO_POINTS", "SAGA_MAX_ITER", "SAGA_TOL",
           "SOLVER_MAX_ITER", "SOLVER_TOL", "TIE_PASSES", "TUNING", "category_groups",
           "inner_folds", "lambda_max", "lowest_rounded", "middle_of_category_ties"]
