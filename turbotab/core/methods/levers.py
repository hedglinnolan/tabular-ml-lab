"""Explore's levers as in-fold rules the resampling repeats (MODELING_SEQUENCE §0 ruling 3; §1 rows
1, 5 and 10; ``set_levers``).

The prediction review's critical finding: below 20,000 units the app's validation plan holds no rows
out, so "training rows" are every row, and "any lever pulled from an outcome-relationship view (a
spline from a bending plot, dropping a weak predictor) then sits outside the bootstrap/CV, yet the
score is labeled optimism-corrected". Harrell (*Regression Modeling Strategies*, Validation): strong
internal validation repeats "_all_ analysis steps involving Y afresh at each re-sample". Moscovich &
Rosset (JRSS B 2022;84:1474): "unsupervised preprocessing can, in fact, introduce a substantial bias
into cross-validation estimates". So each Explore lever is offered first as a rule that is a pipeline
step, fitted on the rows each fit sees, and the resampling repeats it:

* :class:`RuleSplines` — **spline df by a stated rule.** Every continuous predictor enters as a
  restricted cubic spline whose number of knots k follows Harrell's rule (RMS 2nd ed. §2.4.6: "If the
  sample size is small (n < 30, say) k = 3 is a good starting point. If the sample is large (n > 100),
  k = 5 is a good choice", with k = 4 "a good compromise" between) on the fitting rows' effective
  size (Harrell §4.4: the number of rows for a numeric outcome, the rarer class's count for a
  categorical one), the knots at Harrell's percentiles of the fitting rows. The rule never reads how
  the outcome moves with the predictor.
* :class:`InnerCVForms` — **nonlinearity by inner cross-validation.** Within the fitting rows, each
  continuous predictor is linear or that spline as an inner K-fold cross-validation of a regression
  on every input says (mean squared error, or log loss for a yes/no outcome; the spline wins only
  where its inner loss is lower), its knots placed on each inner training fold. The outer resampling
  repeats the choice, so its optimism is in the corrected score.
* :class:`VarianceFilter` — **variance filter in-fold.** Near-zero-variance predictors (caret's
  ``nearZeroVar`` rule, Kuhn 2008, *J Stat Softw* 28(5): the most common value over the second most
  common above 95/5 and at most 10% distinct values, or one value), or all but the ``keep`` most
  variable continuous predictors, dropped on each training fold's own values.
* :class:`ImbalanceCorrected` — **an imbalance correction in-fold, followed by recalibration**
  (TRIPOD+AI 13: "If class imbalance methods were used, state why and how this was done, and any
  subsequent methods to recalibrate the model"). Van den Goorbergh et al. (JAMIA 2022;29:1525): "The
  use of random undersampling, random oversampling, or SMOTE yielded poorly calibrated models: the
  probability to belong to the minority class was strongly overestimated", so the corrected model's
  log-odds are recalibrated (an intercept and a slope, logistic recalibration) on inner
  cross-validated predictions within the fitting rows, never on the rows it scores.

A lever applied by hand after an outcome view is not one of these: it is recorded and disclosed as
outside the corrected score (``stages/explore.py``).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from turbotab.core.methods.exposure_form import rcs_basis, rcs_knots, spline_names

HARRELL_SMALL = 30  # below this effective size, 3 knots
HARRELL_LARGE = 100  # above it, 5 knots; 4 between
SPLINE_RULE = ("Harrell's rule (RMS 2nd ed. §2.4.6): 3 knots below an effective size of 30, 5 above "
               "100, else 4")
MIN_DISTINCT = 10  # a convention: fewer distinct values read as a code or a short count, not a curve
INNER_FOLDS = 5
NZV_FREQ = 95 / 5  # caret::nearZeroVar's freqCut
NZV_UNIQUE = 10.0  # … and its uniqueCut, in percent
RECAL_FOLDS = 5


def infer_task(task: str | None, y: Any) -> str:
    """The task a step reads: the one it was built for, else read from the outcome it is fitted
    with (a preview builds pipelines before the task is answered): two values a yes/no outcome,
    more than ten numbers a numeric one, else classes."""
    if task:
        return str(task)
    if y is None:
        return "regression"
    values = pd.Series(np.asarray(y, dtype=object)).dropna()
    k = int(values.nunique())
    if k <= 2:
        return "binary"
    numbers = pd.to_numeric(values, errors="coerce")
    return "regression" if numbers.notna().all() and k > 10 else "multiclass"


def effective_size(task: str | None, y: Any, n_rows: int) -> int:
    """Harrell's effective sample size (RMS §4.4): the rows for a numeric outcome; the rarer class's
    count for a categorical one (``y`` None: the rows)."""
    task = infer_task(task, y)
    if y is None or task in ("regression", "time_to_event"):
        return int(n_rows)
    counts = pd.Series(np.asarray(y, dtype=object)).value_counts()
    return int(counts.min()) if len(counts) else int(n_rows)


def knots_by_rule(n_effective: int) -> int:
    """Harrell's number of knots for this effective size (module docstring)."""
    n = int(n_effective)
    if n < HARRELL_SMALL:
        return 3
    if n > HARRELL_LARGE:
        return 5
    return 4


def _numbers(X: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)


def curve_candidates(X: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """The columns a form rule may bend: the listed ones present (every column when None), numeric,
    with at least :data:`MIN_DISTINCT` distinct values."""
    names = [str(c) for c in X.columns] if columns is None else [c for c in columns if c in X.columns]
    out = []
    for c in names:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            continue
        values = _numbers(X, c)
        if len(np.unique(values[np.isfinite(values)])) >= MIN_DISTINCT:
            out.append(c)
    return out


def _spline_frame(X: pd.DataFrame, knots: Mapping[str, np.ndarray]) -> pd.DataFrame:
    parts: dict[str, Any] = {}
    for name in X.columns:
        column = str(name)
        if column not in knots:
            parts[column] = X[name]
            continue
        values = _numbers(X, column)
        names = spline_names(column, len(knots[column]))
        parts[names[0]] = values
        basis = rcs_basis(values, knots[column])
        for j, out in enumerate(names[1:]):
            parts[out] = basis[:, j]
    return pd.DataFrame(parts, index=X.index)


def _names_out(columns: Sequence[str], knots: Mapping[str, np.ndarray]) -> np.ndarray:
    names: list[str] = []
    for column in columns:
        names.extend(spline_names(column, len(knots[column])) if column in knots else [column])
    return np.asarray(names, dtype=object)


class RuleSplines(TransformerMixin, BaseEstimator):
    """Every candidate continuous input as a restricted cubic spline, k by Harrell's rule on the
    fitting rows (module docstring). ``columns``: the candidates (None: every column)."""

    def __init__(self, columns: Sequence[str] | None = None, task: str = "regression"):
        self.columns = columns
        self.task = task

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RuleSplines":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_effective_ = effective_size(self.task, y, len(X))
        self.k_ = knots_by_rule(self.n_effective_)
        self.knots_: dict[str, np.ndarray] = {}
        self.skipped_: dict[str, str] = {}
        for c in curve_candidates(X, self.columns):
            try:
                self.knots_[c], _ = rcs_knots(_numbers(X, c), self.k_)
            except ValueError as exc:
                self.skipped_[c] = str(exc)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "knots_"):
            raise ValueError("RuleSplines is not fitted yet.")
        return _spline_frame(X, self.knots_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out([str(c) for c in self.feature_names_in_], self.knots_)

    def lineage(self) -> list[dict[str, Any]]:
        out = []
        for column in (str(c) for c in self.feature_names_in_):
            if column not in self.knots_:
                out.append({"output": column, "inputs": [column], "operation": "kept",
                            "formula": None})
                continue
            k = len(self.knots_[column])
            for name in spline_names(column, k):
                out.append({"output": name, "inputs": [column],
                            "operation": f"restricted cubic spline, {k} knots by rule",
                            "formula": f"{SPLINE_RULE}; effective size {self.n_effective_}"})
        return out


# ── the inner cross-validated choice of form ──────────────────────────────────


def _design(frame: pd.DataFrame) -> np.ndarray:
    """An intercept beside every column, a blank filled by its column's mean on these rows (the
    inner comparison's own design: the pipeline fills blanks before this step)."""
    values = frame.to_numpy(dtype=float)
    if np.isnan(values).any():
        means = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        values = np.where(np.isfinite(values), values, means)
    return np.column_stack([np.ones(len(values)), values])


def _ols_loss(Z_fit: np.ndarray, y_fit: np.ndarray, Z_test: np.ndarray, y_test: np.ndarray) -> float:
    beta = np.linalg.lstsq(Z_fit, y_fit, rcond=None)[0]
    return float(np.mean((y_test - Z_test @ beta) ** 2))


def logistic_coefficients(Z: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None
                          ) -> np.ndarray | None:
    """The logistic maximum-likelihood coefficients of ``y`` on the columns of ``Z`` (R's
    ``glm(family = binomial)``): Newton–Raphson with step halving to a score below 10⁻¹² of the
    rows (``models.causal.logistic_fit``). None only when it does not converge. The size of a
    coefficient is no failure: a spline basis on a 0–1 share or a predictor in grams converges to
    coefficients far from 0, as R's ``glm`` does."""
    from turbotab.core.models.causal import logistic_fit

    try:
        beta = logistic_fit(Z, y, offset=offset)
    except ValueError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


def _logistic_loss(Z_fit: np.ndarray, y_fit: np.ndarray, Z_test: np.ndarray,
                   y_test: np.ndarray) -> float:
    beta = logistic_coefficients(Z_fit, y_fit)
    if beta is None:
        return float("inf")
    eta = Z_test @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p)))


def inner_folds(n: int, cv: Any, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The inner splits: those the fit handed this step (``cv`` as a list of (train, test)
    positions: whole units, or time order; ``models.inner_cv``), else ``cv`` shuffled folds."""
    if isinstance(cv, (list, tuple)) and cv and isinstance(cv[0], (list, tuple)):
        return [(np.asarray(a), np.asarray(b)) for a, b in cv]
    k = max(2, min(int(cv or INNER_FOLDS), n))
    order = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[order] = np.arange(n) % k
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def form_losses(X: pd.DataFrame, y01: np.ndarray, column: str, k: int, *, task: str,
                splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """(linear, spline) inner cross-validated loss of a regression on every input of ``X``, with
    ``column`` linear or a restricted cubic spline of ``k`` knots placed on each inner training
    fold's values. Inf where a fold cannot place the knots or fit."""
    loss = _logistic_loss if task == "binary" else _ols_loss
    lin, spl = [], []
    weights = []
    for train, test in splits:
        X_fit, X_test = X.iloc[train], X.iloc[test]
        lin.append(loss(_design(X_fit), y01[train], _design(X_test), y01[test]))
        try:
            knots, _ = rcs_knots(_numbers(X_fit, column), k)
        except ValueError:
            spl.append(float("inf"))
        else:
            spl.append(loss(_design(_spline_frame(X_fit, {column: knots})), y01[train],
                            _design(_spline_frame(X_test, {column: knots})), y01[test]))
        weights.append(len(test))
    w = np.asarray(weights, dtype=float)
    return float(np.dot(w, lin) / w.sum()), float(np.dot(w, spl) / w.sum())


def _outcome01(task: str, y: Any) -> np.ndarray:
    values = np.asarray(y)
    if task == "binary":
        levels = sorted(pd.unique(values).tolist(), key=lambda v: str(v))
        if len(levels) != 2:
            raise ValueError("The inner comparison of forms needs a yes/no outcome with both levels.")
        return (values == levels[1]).astype(float)
    return values.astype(float)


class InnerCVForms(TransformerMixin, BaseEstimator):
    """Each candidate continuous input linear or a restricted cubic spline (k by Harrell's rule),
    chosen within the fitting rows by inner cross-validation (module docstring). A numeric or yes/no
    outcome; other outcomes keep every input linear and say so (``note_``)."""

    def __init__(self, columns: Sequence[str] | None = None, task: str = "regression",
                 cv: Any = INNER_FOLDS, seed: int = 0):
        self.columns = columns
        self.task = task
        self.cv = cv
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InnerCVForms":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_effective_ = effective_size(self.task, y, len(X))
        self.k_ = knots_by_rule(self.n_effective_)
        self.knots_: dict[str, np.ndarray] = {}
        self.losses_: dict[str, tuple[float, float]] = {}
        self.note_: str | None = None
        task = infer_task(self.task, y)
        if y is None or task not in ("regression", "binary"):
            self.note_ = ("The inner comparison of forms covers a numeric or yes/no outcome; every "
                          "input stays linear.")
            return self
        y01 = _outcome01(task, y)
        numeric = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]]
        splits = inner_folds(len(X), self.cv, int(self.seed))
        for c in curve_candidates(X, self.columns):
            lin, spl = form_losses(numeric, y01, c, self.k_, task=task, splits=splits)
            self.losses_[c] = (lin, spl)
            if spl < lin:
                try:
                    self.knots_[c], _ = rcs_knots(_numbers(X, c), self.k_)
                except ValueError:
                    continue
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "knots_"):
            raise ValueError("InnerCVForms is not fitted yet.")
        return _spline_frame(X, self.knots_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out([str(c) for c in self.feature_names_in_], self.knots_)

    def lineage(self) -> list[dict[str, Any]]:
        out = []
        for column in (str(c) for c in self.feature_names_in_):
            if column not in self.knots_:
                op = ("kept linear by inner cross-validation" if column in self.losses_ else "kept")
                out.append({"output": column, "inputs": [column], "operation": op, "formula": None})
                continue
            k = len(self.knots_[column])
            lin, spl = self.losses_[column]
            for name in spline_names(column, k):
                out.append({"output": name, "inputs": [column],
                            "operation": f"restricted cubic spline, {k} knots, chosen by inner "
                                         f"cross-validation",
                            "formula": f"inner loss {spl:.4g} against {lin:.4g} linear"})
        return out


# ── the variance filter ──────────────────────────────────────────────────────


def near_zero_variance(values: Any) -> bool:
    """caret's ``nearZeroVar`` with its defaults (module docstring): one distinct value, or a
    frequency ratio above 95/5 together with at most 10% distinct values. Blanks are left out."""
    s = pd.Series(np.asarray(values, dtype=object)).dropna()
    if len(s) == 0:
        return True
    counts = s.value_counts()
    if len(counts) == 1:
        return True
    ratio = float(counts.iloc[0]) / float(counts.iloc[1])
    unique_pct = 100.0 * len(counts) / len(s)
    return bool(ratio > NZV_FREQ and unique_pct <= NZV_UNIQUE)


class VarianceFilter(TransformerMixin, BaseEstimator):
    """Drop near-zero-variance candidates (``method = "near_zero"``), or keep only the ``keep`` most
    variable continuous candidates (``"top"``), learned on the fitting rows. Outcome-free."""

    def __init__(self, method: str = "near_zero", keep: int | None = None,
                 columns: Sequence[str] | None = None):
        self.method = method
        self.keep = keep
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VarianceFilter":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        names = ([str(c) for c in X.columns] if self.columns is None
                 else [c for c in self.columns if c in X.columns])
        dropped: list[str] = []
        if self.method == "near_zero":
            dropped = [c for c in names if near_zero_variance(X[c].to_numpy())]
        elif self.method == "top":
            curves = curve_candidates(X, names)
            variance = {c: float(np.nanvar(_numbers(X, c), ddof=1)) for c in curves}
            ranked = sorted(curves, key=lambda c: (-variance[c], c))
            dropped = ranked[int(self.keep or len(ranked)):]
        else:
            raise ValueError(f"Unknown variance filter {self.method!r}.")
        self.dropped_ = dropped
        self.kept_ = [str(c) for c in X.columns if str(c) not in set(dropped)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "kept_"):
            raise ValueError("VarianceFilter is not fitted yet.")
        return X.loc[:, [c for c in X.columns if str(c) not in set(self.dropped_)]]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.kept_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        return [{"output": c, "inputs": [c], "operation": "kept by the in-fold variance filter",
                 "formula": None} for c in self.kept_]


# ── an imbalance correction, in-fold, then recalibration ──────────────────────


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def resampled_rows(y: np.ndarray, method: str, rng: np.random.Generator) -> np.ndarray:
    """The rows an undersampling or oversampling draw fits on: every minority row with the majority
    drawn down to its count without replacement, or every majority row with the minority drawn up
    to its count with replacement."""
    levels, counts = np.unique(y, return_counts=True)
    minority, majority = levels[np.argmin(counts)], levels[np.argmax(counts)]
    mino, majo = np.flatnonzero(y == minority), np.flatnonzero(y == majority)
    if method == "undersample":
        return np.sort(np.concatenate([mino, rng.choice(majo, size=len(mino), replace=False)]))
    if method == "oversample":
        return np.sort(np.concatenate([majo, mino, rng.choice(mino, size=len(majo) - len(mino),
                                                                  replace=True)]))
    raise ValueError(f"Unknown resampling {method!r}.")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """scikit-learn's ``class_weight="balanced"``: n / (classes × the row's class count)."""
    levels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return len(y) / (len(levels) * counts[inverse].astype(float))


class ImbalanceCorrected(ClassifierMixin, BaseEstimator):
    """A yes/no model fitted with an imbalance correction (``weights``: balanced class weights;
    ``undersample``, ``oversample``: a seeded draw), then recalibrated by logistic recalibration of
    its log-odds on inner cross-validated predictions within the fitting rows (module docstring).

    ``cv`` is set by the fit (``models.inner_cv``) to whole-unit or time-ordered inner splits when the
    outer folds are drawn so. Attributes the wrapped model has (``coef_``) are read through it.

    **Early stopping** (RECIPES F12, §4.3). ``early_stopping`` and ``validation_fraction`` are the
    wrapped model's (``wrap_model`` copies them; None reads the wrapped model's own at fit), so
    ``inner_cv.fit_pipeline`` passes a fit that stops early its stopping rows, ``X_val``:
    whole units drawn before any step was fit. Fit without them, above 10,000 received rows under
    ``"auto"``, the stopping units are drawn here first (``inner_cv.validation_rows``, by the
    ``groups`` or ``order`` handed to ``fit``; handed inner splits without either, it refuses,
    rather than draw rows that split a unit). Either way:
    only the other rows are resampled; the stopping rows reach every model this fit makes (the
    deployed one and each recalibration fit) as they are, and no recalibration fold holds them; the
    recalibration splits cover the rows this fit trains on. The threshold reads the rows the fit
    receives, never the resampled count: the wrapped model's own ``"auto"`` is overridden, so it
    never draws a stopping split by position from resampled copies.

    **Tuning** (RECIPES §4.3): ``inner_param`` names the parameter holding the wrapped model, so a
    search's candidate settings reach it (``estimator__learning_rate``) while the early-stopping
    switch and share stay the wrapper's own. A candidate is the wrapped, recalibrated model, so
    the search scores exactly what is deployed."""

    inner_param = "estimator"

    def __init__(self, estimator: Any = None, method: str = "weights", recalibrate: bool = True,
                 cv: Any = RECAL_FOLDS, seed: int = 0, early_stopping: Any = None,
                 validation_fraction: float | None = None):
        self.estimator = estimator
        self.method = method
        self.recalibrate = recalibrate
        self.cv = cv
        self.seed = seed
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction

    def stopping_setting(self) -> tuple[Any, float | None]:
        """(flag, share): this wrapper's, else the wrapped model's; (False, None) for a wrapped
        model that does not stop early. ``inner_cv`` reads the setting through this, so a wrapper
        built with the defaults takes the stopping-rows path as one built by ``wrap_model`` does."""
        params = getattr(self.estimator, "get_params", lambda deep=False: {})(deep=False)
        if "early_stopping" not in params or "validation_fraction" not in params:
            return False, None
        flag = params["early_stopping"] if self.early_stopping is None else self.early_stopping
        share = (params["validation_fraction"] if self.validation_fraction is None
                 else self.validation_fraction)
        return flag, share

    def _fit_one(self, X: Any, y: np.ndarray, rng: np.random.Generator,
                 val: tuple[Any, np.ndarray] | None = None) -> Any:
        from sklearn.base import clone

        model = clone(self.estimator)
        stop: dict[str, Any] = {}
        if "early_stopping" in model.get_params(deep=False):
            model.set_params(early_stopping=val is not None)
            if val is not None:
                stop = {"X_val": val[0], "y_val": val[1]}
        if self.method == "weights":
            weighted = dict(stop)
            if stop:  # the stopping loss weighs each class as the training loss does
                weighted["sample_weight_val"] = _class_weights(y, val[1])
            try:
                return model.fit(X, y, sample_weight=balanced_weights(y), **weighted)
            except TypeError:  # a model that takes no weights is drawn up instead
                rows = resampled_rows(y, "oversample", rng)
                return model.fit(_take(X, rows), y[rows], **stop)
        rows = resampled_rows(y, self.method, rng)
        return model.fit(_take(X, rows), y[rows], **stop)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None, *, groups: Any = None,
            order: Any = None) -> "ImbalanceCorrected":
        """``X_val``, ``y_val``: the stopping rows (``inner_cv.fit_pipeline``). Without them, a fit
        that stops early draws its stopping units here, by ``groups`` (the unit per row) or
        ``order`` (each row's unit rank in time) when given, as ``inner_cv.validation_rows`` does;
        handed inner splits (``cv``) without either, it refuses, since those splits may hold whole
        units that a draw by rows would split."""
        from turbotab.core.models.inner_cv import EARLY_STOPPING_ROWS, row_keys, validation_rows

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("An imbalance correction here is for a yes/no outcome.")
        flag, share = self.stopping_setting()
        val: tuple[Any, np.ndarray] | None = None
        kept: np.ndarray | None = None  # the rows trained on, if stopping units are drawn here
        n_received = len(y)
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("Stopping rows need both X_val and y_val.")
            if share is None or flag is False:
                raise ValueError("Stopping rows were passed to a fit that does not stop early.")
            val = (X_val, np.asarray(y_val))
        elif share is not None and (flag is True
                                    or (flag == "auto" and n_received > EARLY_STOPPING_ROWS)):
            given = isinstance(self.cv, (list, tuple)) and len(self.cv) > 0
            if given and groups is None and order is None:
                raise ValueError("This fit draws its own stopping units but was handed inner "
                                 "splits without the units: pass groups or order, or fit "
                                 "through inner_cv.fit_pipeline.")
            keys = row_keys(X, y) if groups is None and order is None else None
            held = validation_rows(float(share), groups=groups, keys=keys, order=order, y=y,
                                   seed=int(self.seed))
            val = (_take(X, np.flatnonzero(held)), y[held])
            kept = np.flatnonzero(~held)
            X, y = _take(X, kept), y[kept]
        rng = np.random.default_rng(int(self.seed))
        self.estimator_ = self._fit_one(X, y, rng, val)
        self.stopping_rows_ = 0 if val is None else len(val[1])
        self.calibration_ = (0.0, 1.0)
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        self.recalibration_note_: str | None = None
        if not self.recalibrate:
            return self
        lp = np.full(len(y), np.nan)
        for train, test in _recalibration_folds(self.cv, int(self.seed), len(y), kept, n_received):
            if len(np.unique(y[train])) < 2:
                continue
            inner = self._fit_one(_take(X, train), y[train], rng, val)
            lp[test] = _logit(_positive(inner, _take(X, test), self.classes_[1]))
        ok = np.isfinite(lp)
        event = (y == self.classes_[1]).astype(float)
        fitted = logistic_coefficients(np.column_stack([np.ones(int(ok.sum())), lp[ok]]), event[ok])
        if fitted is not None:
            self.calibration_ = (float(fitted[0]), float(fitted[1]))
        else:
            self.recalibration_note_ = ("The recalibration did not converge on the inner "
                                        "predictions, so the corrected model's log-odds stand "
                                        "as fitted.")
        return self

    def deployed_coefficients(self) -> tuple[float, np.ndarray] | None:
        """The deployed model's intercept and coefficients on the log-odds scale when the wrapped
        model is linear in its inputs: the recalibration's ``a + b·(β₀ + xβ)`` gives ``a + b·β₀``
        and ``b·β``. None for a model with no coefficients."""
        inner = self.__dict__.get("estimator_")
        if inner is None or not hasattr(inner, "coef_"):
            return None
        a, b = self.calibration_
        coef = np.asarray(inner.coef_, dtype=float).ravel()
        intercept = float(np.ravel(getattr(inner, "intercept_", [0.0]))[0])
        return a + b * intercept, b * coef

    def predict_proba(self, X: Any) -> np.ndarray:
        a, b = self.calibration_
        p = 1.0 / (1.0 + np.exp(-(a + b * _logit(_positive(self.estimator_, X, self.classes_[1])))))
        return np.column_stack([1 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("estimator_",):
            raise AttributeError(name)
        inner = self.__dict__.get("estimator_")
        if inner is not None and name.endswith("_") and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(name)


def _take(X: Any, rows: np.ndarray) -> Any:
    return X.iloc[rows] if hasattr(X, "iloc") else np.asarray(X)[rows]


def _class_weights(y: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """The stopping rows' weights: each row its class's balanced weight in the training rows."""
    levels, counts = np.unique(y, return_counts=True)
    weight = dict(zip(levels.tolist(), (len(y) / (len(levels) * counts)).tolist()))
    return np.asarray([weight.get(v, 1.0) for v in np.asarray(y_val).tolist()], dtype=float)


def _recalibration_folds(cv: Any, seed: int, n: int, kept: np.ndarray | None,
                         n_received: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The recalibration's inner splits over the ``n`` rows the fit trains on. When the fit drew
    its own stopping units (``kept``: the positions it trains on among the ``n_received`` rows it
    was handed), splits handed over for the received rows are cut down to the kept rows."""
    given = isinstance(cv, (list, tuple)) and cv and isinstance(cv[0], (list, tuple))
    if kept is None or not given:
        return inner_folds(n, cv, seed)
    position = np.full(n_received, -1)
    position[kept] = np.arange(len(kept))
    out = []
    for train, test in cv:
        a, b = position[np.asarray(train)], position[np.asarray(test)]
        a, b = a[a >= 0], b[b >= 0]
        if len(a) and len(b):
            out.append((a, b))
    return out


def _positive(model: Any, X: Any, level: Any) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    return proba[:, list(model.classes_).index(level)]


# ── the design: which steps a lever answer adds (``models.pipeline``) ─────────


def lever_steps(levers: Mapping[str, Any] | None, task: str, candidates: Sequence[str],
                seed: int = 0) -> list[tuple[str, Any]]:
    """The in-fold steps a ``set_levers`` answer adds after the shared construction steps: the form
    rule, then the variance filter (MODELING_SEQUENCE §1.1: construction, then filters)."""
    if not levers:
        return []
    steps: list[tuple[str, Any]] = []
    forms = levers.get("forms") or "none"
    if forms == "rule":
        steps.append(("lever_forms", RuleSplines(list(candidates), task)))
    elif forms == "inner_cv":
        steps.append(("lever_forms", InnerCVForms(list(candidates), task, INNER_FOLDS, seed)))
    filt = levers.get("variance_filter") or "none"
    if filt != "none":
        steps.append(("lever_filter", VarianceFilter(filt, levers.get("keep"), None)))
    return steps


def describe_step(name: str, spec: Any) -> tuple[str, str]:
    """(label, detail) of an EXPLORE step in the design's step list (``pipeline.describe_steps``)."""
    levers = getattr(spec, "levers", None) or {}
    if name == "lever_forms":
        if levers.get("forms") == "inner_cv":
            return ("Forms by inner cross-validation",
                    "Each continuous predictor linear or a spline, as an inner 5-fold "
                    "cross-validation on the training fold chose; knots by Harrell's rule.")
        return ("Splines by a stated rule",
                f"Every continuous predictor a restricted cubic spline; {SPLINE_RULE}, on each "
                f"training fold's rows.")
    if name == "lever_filter":
        if levers.get("variance_filter") == "top":
            return ("Variance filter, in-fold",
                    f"Keeps the {int(levers.get('keep') or 0):,} most variable predictors of each "
                    f"training fold.")
        return ("Variance filter, in-fold",
                "Drops near-zero-variance predictors on each training fold (caret's nearZeroVar).")
    from turbotab.core.models.variable_selection import LABELS

    method = (getattr(spec, "selection", None) or {}).get("method") or "none"
    return (LABELS.get(method, "Selection"),
            "Selects on each training fold's outcome, so every fold and resample repeats it.")


def wrap_model(model: Any, levers: Mapping[str, Any] | None, task: str, seed: int = 0) -> Any:
    """The model step with the answer's imbalance correction around it (a yes/no outcome). A
    wrapped model that stops early lends the wrapper its ``early_stopping`` and
    ``validation_fraction``, so the fit hands the wrapper its stopping rows (RECIPES F12)."""
    method = (levers or {}).get("imbalance") or "none"
    if method == "none" or task != "binary":
        return model
    params = model.get_params(deep=False)
    stopping = ({"early_stopping": params["early_stopping"],
                 "validation_fraction": params["validation_fraction"]}
                if "early_stopping" in params and "validation_fraction" in params else {})
    return ImbalanceCorrected(model, method, True, RECAL_FOLDS, seed, **stopping)


# ── the contracts (BLUEPRINT §13), the leash and the sentence ─────────────────


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "spline_rule" in CONTRACTS:
        return
    here = "turbotab.core.methods.levers"
    prediction = ("prediction",)
    declared = ("Not offered: under inference each form is declared before the estimates "
                "(set_exposure_form), never chosen by the data")

    def option(key: str, label: str, customary: str, says: str, rung: str) -> ContractOption:
        return ContractOption(key, label, customary, {"prediction": says, "inference": declared},
                              {"prediction": rung, "inference": "not_offered"})

    def contract(key: str, label: str, scope: str, **fields: Any) -> None:
        register_contract(MethodContract(key=key, label=label, slot="in_fold", scope=scope,
                                         package="EXPLORE", decision="set_levers", stage="design",
                                         **fields))

    contract(
        "spline_rule", "Splines for every continuous predictor, knots by a stated rule",
        "training_fold", place="5 · Functional form (prediction)", run_order=7.0,
        scope_note="Its knots are placed on the values of the rows each fit sees; it never reads "
                   "the outcome's relationship with the predictor.",
        needs=("continuous predictors", "the fitting rows"),
        question="Let a stated rule bend every continuous predictor, in each training fold?",
        options=(option("rule", "Every continuous predictor as a spline, k by Harrell's rule",
                        "Splines are chosen per predictor after looking at plots",
                        "Sound: the rule is stated in advance and the resampling repeats it",
                        "recommended"),),
        storyboard=("count the fitting rows", "k by Harrell's rule", "place the knots on each "
                    "predictor's values", "replace each predictor by its spline basis"),
        relations=(Relation("implies", "in_fold_rule",
                            "the spline's knots are placed again in every training fold and "
                            "resample, so its optimism is in the corrected score",
                            purposes=prediction, enforced_by=f"{here}:RuleSplines",
                            id="spline_rule_in_fold"),
                   Relation("implies", "candidate_parameters",
                            "the rule's spline columns count among the candidate predictor "
                            "parameters Riley's minimum sample size reads before the shelf",
                            purposes=prediction,
                            enforced_by="turbotab.core.stages.modeling:rule_spline_terms",
                            id="spline_rule_counted_by_riley"),),
        short="splines with knots by Harrell's rule",
        sources=("Harrell, Regression Modeling Strategies 2nd ed. §2.4.6",),
        sentence=f"{here}:decision_sentence")
    contract(
        "inner_cv_form", "Nonlinearity chosen by inner cross-validation", "model",
        place="5 · Functional form (prediction)", run_order=7.1,
        scope_note="It reads the outcome on the training fold's inner folds, so it is fitted in "
                   "each training fold.",
        needs=("continuous predictors", "a numeric or yes/no outcome"),
        question="Let inner cross-validation choose linear or spline for each predictor?",
        options=(option("inner_cv", "Linear or spline by inner cross-validation",
                        "Nonlinearity is usually tested on all rows, then kept or dropped",
                        "Sound under prediction: the choice is made inside each training fold",
                        "recommended"),),
        storyboard=("split the training fold into inner folds", "score linear and spline on them",
                    "keep the better form", "refit on the whole training fold"),
        relations=(Relation("implies", "in_fold_choice",
                            "each predictor's form is chosen anew in every training fold, so the "
                            "choice's optimism is in the corrected score", purposes=prediction,
                            enforced_by=f"{here}:InnerCVForms", id="form_chosen_in_fold"),
                   Relation("implies", "candidate_parameters",
                            "every spline the inner choice may keep counts among the candidate "
                            "predictor parameters Riley's minimum sample size reads",
                            purposes=prediction,
                            enforced_by="turbotab.core.stages.modeling:rule_spline_terms",
                            id="inner_cv_counted_by_riley"),),
        short="each predictor's form by inner cross-validation",
        sources=("Harrell, Regression Modeling Strategies, Validation",
                 "Varma & Simon, BMC Bioinformatics 2006;7:91"),
        sentence=f"{here}:decision_sentence")
    contract(
        "variance_filter", "A variance filter fitted in each training fold", "training_fold",
        place="8 · Selection (filters)", run_order=7.5,
        scope_note="Outcome-free, but it learns each column's spread from the study rows, so it is "
                   "fitted in-fold (Moscovich & Rosset 2022).",
        needs=("candidate predictors",),
        question="Drop near-constant predictors, or keep the most variable, in each training fold?",
        options=(
            option("near_zero", "Drop near-zero-variance predictors in each fold",
                   "Filters are usually run once on every row before modeling",
                   "Sound in-fold: outcome-free, learned on the fold", "recommended"),
            option("top", "Keep the most variable predictors in each fold",
                   "Variance and IQR filters are customary for omics before modeling",
                   "Sound in-fold at p ≫ n", "available")),
        storyboard=("measure each predictor's spread on the training fold", "drop the near-constant "
                    "ones (or all but the most variable)", "fit on the rest"),
        relations=(Relation("precedes", "variable_selection",
                            "filters run before selection (MODELING_SEQUENCE §1.1)",
                            purposes=prediction, id="filter_before_selection"),),
        short="the variance filter",
        sources=("Kuhn, J Stat Softw 2008;28(5) (nearZeroVar)",
                 "Moscovich & Rosset, JRSS B 2022;84:1474"),
        sentence=f"{here}:decision_sentence")
    contract(
        "imbalance_correction", "An imbalance correction in-fold, then recalibration", "model",
        place="10 · Model-specific preprocessing", run_order=9.0,
        scope_note="It reweights or redraws the training fold by its outcome classes and "
                   "recalibrates on inner predictions, so it is fitted in-fold.",
        needs=("a yes/no outcome",),
        question="Correct the class imbalance in each training fold, then recalibrate?",
        options=(
            option("none", "No correction; a threshold chosen in-fold if a decision rests on it",
                   "Corrections are common for rare outcomes in ML",
                   "Sound: imbalance is not a problem in itself; moving the threshold gives the "
                   "same sensitivity and specificity (van den Goorbergh et al. 2022)",
                   "recommended"),
            option("weights", "Balanced class weights in-fold, then recalibration",
                   "Class weights are customary in ML",
                   "Miscalibrates until recalibrated; no gain in AUC (van den Goorbergh et al. "
                   "2022)", "rank_lower"),
            option("undersample", "Undersampling in-fold, then recalibration",
                   "Random undersampling is customary in ML",
                   "Discards rows and miscalibrates until recalibrated", "rank_lower"),
            option("oversample", "Oversampling in-fold, then recalibration",
                   "Random oversampling (and SMOTE) is customary in ML",
                   "Miscalibrates until recalibrated (van den Goorbergh et al. 2022)",
                   "rank_lower")),
        storyboard=("reweight or redraw the training fold by class", "fit the model",
                    "recalibrate its log-odds on inner cross-validated predictions",
                    "score the held-out fold"),
        relations=(Relation("implies", "recalibration",
                            "every imbalance correction is followed by recalibration within the "
                            "training fold (TRIPOD+AI 13)", purposes=prediction,
                            when=("weights", "undersample", "oversample"),
                            enforced_by=f"{here}:ImbalanceCorrected", id="correction_recalibrated"),),
        short="the imbalance correction with its recalibration",
        sources=("van den Goorbergh et al., JAMIA 2022;29:1525", "TRIPOD+AI item 13"),
        sentence=f"{here}:decision_sentence")


_register_contracts()


def _levers_are_prediction(decision: Any, ctx: Any) -> None:
    """Explore's in-fold rules are prediction's (MODELING_SEQUENCE ruling 3); under inference every
    form is declared before the estimates and nothing is chosen by the data."""
    from turbotab.core.decisions import Refusal, SetPurpose, _state

    state = _state(ctx)
    if state is None or getattr(state, "purpose", None) != "inference":
        return
    if (decision.forms, decision.variance_filter, decision.imbalance) == ("none", "none", "none"):
        return
    raise Refusal(
        "levers_not_inference",
        "Under inference a predictor's form is declared before the estimates, never chosen by the "
        "rows they are reported from, and an imbalance correction would change the comparison you want: "
        "declare the form of each study factor instead.",
        exits=[{"label": "Declare the form of what you study (the functional-form question)",
                "decision": None},
               {"label": "Make the purpose prediction", "decision": SetPurpose(purpose="prediction")}])


def _levers_fit_the_task(decision: Any, ctx: Any) -> None:
    """An imbalance correction is for a yes/no outcome; the variance filter's ``top`` needs how
    many to keep; the form rules need fewer candidate predictors than rows."""
    from turbotab.core.decisions import Refusal, _ctx, _state

    state = _state(ctx)
    task = getattr(state, "task", None) if state is not None else None
    task = task or _ctx(ctx, "detected_task")
    if decision.imbalance != "none" and task is not None and task != "binary":
        raise Refusal("imbalance_task", "An imbalance correction here is for a yes/no outcome.",
                      exits=[{"label": "No correction",
                              "decision": decision.model_copy(update={"imbalance": "none"})}])
    if decision.variance_filter == "top" and not decision.keep:
        raise Refusal("keep_how_many", "Keeping the most variable predictors needs how many to keep.",
                      exits=[{"label": "Keep the 1,000 most variable",
                              "decision": decision.model_copy(update={"keep": 1000})},
                             {"label": "Drop near-zero-variance predictors instead",
                              "decision": decision.model_copy(update={"variance_filter": "near_zero"})}])
    if decision.forms == "none":
        return
    from turbotab.core.sequence import artifact

    cohort = artifact(ctx, "cohort") or {}
    n = cohort.get("n_final")
    predictors = cohort.get("predictors") or []
    if n and predictors and len(predictors) >= int(n):
        raise Refusal(
            "forms_at_p_much_greater_n",
            f"{len(predictors):,} candidate predictors for {int(n):,} rows: a spline for each would "
            f"multiply the parameters; at p ≫ n a filter or selection comes first.",
            exits=[{"label": "Keep every predictor linear",
                    "decision": decision.model_copy(update={"forms": "none"})}])
    selection = getattr(state, "selection", None) if state is not None else None
    if (selection is not None and selection.method == "stepwise" and not selection.sensitivity
            and getattr(state, "purpose", None) != "inference"):
        # Backward elimination starts from every column the rule makes (``variable_selection``):
        # the splines must leave the smallest training fold room to fit them all.
        from turbotab.core.decisions import LeverSpec
        from turbotab.core.models.variable_selection import stepwise_room

        probe = state.model_copy(update={"levers": LeverSpec(
            **decision.model_dump(exclude={"kind"}))})
        room = stepwise_room(probe, ctx, task)
        if room is not None and room[0] >= room[1] - 1:
            raise Refusal(
                "stepwise_needs_rows",
                f"With these splines backward elimination would start from all {room[0]:,} "
                f"candidate columns, and the smallest training fold has {room[1]:,} rows: a model "
                f"with every column and an intercept needs more rows than that.",
                exits=[{"label": "Keep every predictor linear",
                        "decision": decision.model_copy(update={"forms": "none"})},
                       {"label": "Choose the elastic net in the selection question",
                        "decision": None}])


def _register_validators() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("set_levers", _levers_are_prediction)
    register_validator("set_levers", _levers_fit_the_task)


_register_validators()


def decision_sentence(d: Any, state: Any) -> str:
    """The ``set_levers`` record's methods sentence: each rule, said as fitted in each training
    fold."""
    from turbotab.core.voice import tick

    parts = []
    if d.forms == "rule":
        parts.append("every continuous predictor entered as a restricted cubic spline with its "
                     "number of knots set by Harrell's rule (3 below an effective size of 30, 5 "
                     "above 100, else 4) and its knots at Harrell's percentiles")
    elif d.forms == "inner_cv":
        parts.append("each continuous predictor entered linearly or as a restricted cubic spline "
                     "(knots by Harrell's rule), as an inner 5-fold cross-validation chose")
    if d.variance_filter == "near_zero":
        parts.append("near-zero-variance predictors were dropped (caret's nearZeroVar rule)")
    elif d.variance_filter == "top":
        parts.append(f"all but the {tick(f'{d.keep:,}')} most variable predictors were dropped")
    if d.imbalance != "none":
        how = {"weights": "balanced class weights", "undersample": "random undersampling",
               "oversample": "random oversampling"}[d.imbalance]
        parts.append(f"the class imbalance was corrected by {how}, followed by logistic "
                     f"recalibration on inner cross-validated predictions (TRIPOD+AI 13)")
    if not parts:
        return "No Explore lever was applied as an in-fold rule."
    body = "; ".join(parts)
    return (f"Within each training fold, {body}, so the resampling repeated each rule and its "
            f"optimism is in the corrected score.")


__all__ = ["HARRELL_LARGE", "HARRELL_SMALL", "INNER_FOLDS", "ImbalanceCorrected", "InnerCVForms",
           "MIN_DISTINCT", "NZV_FREQ", "NZV_UNIQUE", "RuleSplines", "SPLINE_RULE", "VarianceFilter",
           "balanced_weights", "curve_candidates", "effective_size", "form_losses", "inner_folds",
           "decision_sentence", "describe_step", "infer_task", "knots_by_rule", "lever_steps",
           "logistic_coefficients", "near_zero_variance", "resampled_rows", "wrap_model"]
