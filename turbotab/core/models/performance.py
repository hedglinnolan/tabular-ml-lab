"""How well a model predicts, with its uncertainty: intervals, calibration, and the spread across
clusters (AUDIT_REPORT §5 WP9, closing ME-10 with :mod:`turbotab.core.models.validation`).

TRIPOD+AI (2024) item 23a: "Report model performance estimates with confidence intervals". Van
Calster et al. 2019 (BMC Med 17:230): "estimated risks can be unreliable even when the algorithms
have good discrimination". So every score the fit reports carries a standard error, and every
probability model is checked for calibration as well as discrimination.

**Intervals on a held-out score.** The AUC's is DeLong's (DeLong, DeLong & Clarke-Pearson 1988,
Biometrics 44:837), computed by Sun & Xu's midrank algorithm (IEEE Signal Process Lett 2014;
21:1389), the variance pROC's ``ci.auc(method = "delong")`` reports. Every other score is a mean
of per-row losses, or a smooth function of such means (R² = 1 − SSE/SST, RMSE = √MSE), and its
standard error is the influence-function (delta-method) one. When a unit's rows repeat, the
influence values are summed within each unit first, so the interval is cluster-robust (Obuchowski
1997, Biometrics 53:567, for the AUC).

**Standard errors of cross-validated scores.** LeDell, Petersen & van der Laan (Electron J Stat
2015;9:1583) give the influence-curve variance of a cross-validated AUC: each fold's AUC has its own
influence values, and ``σ² = (1/V) Σ_v mean_{i∈v}(IC_i²)``, ``SE = σ/√n``, which is the square root
of the sum of the folds' variances over V. The same construction gives the SE of every other fold
mean (Brier score, log loss, accuracy), and the delta method gives the SE of the pooled R², RMSE and
MAE over every out-of-fold prediction. **What it describes**: the uncertainty from which rows were
scored, with each fold's model taken as fitted — LeDell's target, the mean of the fold models'
own performance on new rows. It leaves out the variability of the training sets themselves, which
Bates, Hastie & Tibshirani (JASA 2023) show makes intervals for the expected performance of the
fitting procedure too narrow when predictors are many relative to rows. LeDell et al. report
92–93% coverage at n = 1,000 and 94–95% from n = 5,000.

**Calibration** (Van Calster et al. 2016, J Clin Epidemiol 74:167; 2019). For a probability p with
linear predictor ``lp = logit(p)``: the *calibration intercept* fits ``logit P(y) = a + lp`` (slope
fixed at 1, so ``a`` is calibration-in-the-large; target 0, "negative values suggest
overestimation"); the *calibration slope* fits ``logit P(y) = a + b·lp`` (target 1; "A slope < 1
suggests that estimated risks are too extreme"). The *smoothed curve* is ``lowess(p, y, iter = 0)``
(Cleveland's smoother, span 2/3, no robustness iterations, as rms ``val.prob`` draws it), and
Eavg (the integrated calibration index), E90 and Emax summarize ``|p − smooth(p)|`` as ``val.prob``
does. For a numeric outcome the same three are read on the outcome's scale: the intercept is the
mean of ``y − ŷ``, the slope is the least-squares slope of ``y`` on ``ŷ``.

**When calibration is flagged.** When the data show miscalibration beyond a tolerance: a slope
whose whole 95% interval lies outside 0.9–1.1, or an intercept whose whole interval lies beyond
±0.1 (log-odds) or ± a tenth of the outcome's standard deviation. The 0.9 is Riley et al.'s target
for acceptable overfitting (Stat Med 2019;38:1276: "small optimism in predictor effect estimates
as defined by a global shrinkage factor of ≥0.9"), applied symmetrically; the intercept's
tolerance is a convention, said as one. Every number is reported either way; the concern is kept
for what the interval demonstrates. A rule that flagged a point outside the tolerance whose
interval merely excluded the target fired on the true risks of a perfectly calibrated model in 10%
of 400 simulated samples of 2,000 rows (two tests at the 5% level); this one, in none.

**Across clusters** (internal–external validation). Each cluster's score is pooled by
DerSimonian–Laird random-effects meta-analysis, with the between-cluster SD τ and a 95% prediction
interval for a new cluster (Higgins, Thompson & Spiegelhalter 2009, JRSS A 172:137: ``t_{k−2}``).
The AUC is pooled on the logit scale (Snell et al., Stat Methods Med Res 2018;27:3505: "Normality was
vastly improved when using the logit transformation for the C-statistic … and therefore we recommend
these scales to be used for meta-analysis"), every other score on its own.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

LEVEL = 0.95
SLOPE_TOLERANCE = 0.1  # Riley et al. 2019: shrinkage ≥ 0.9, applied either side of 1
INTERCEPT_TOLERANCE = 0.1  # log-odds, or a tenth of the outcome's SD: a convention
CURVE_POINTS = 40
SMOOTHER = "lowess, span 2/3, no robustness iterations (as rms val.prob)"
EPS = 1e-12
FLAT = 1e-8  # predictions spread less than this (relative to the outcome's) have no slope


def z_value(level: float = LEVEL) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2.0)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Interval(_Model):
    """A score with its standard error and its 95% interval."""

    estimate: float | None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    level: float = LEVEL
    method: str = ""  # "DeLong" · "influence function" · "… clustered by <unit>"


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _root(variance: Any) -> float | None:
    """√variance, or None when the variance is not a finite non-negative number (a degenerate fit
    can return a slightly negative one by rounding)."""
    v = _finite(variance)
    return math.sqrt(v) if v is not None and v >= 0 else None


def interval(estimate: float | None, se: float | None, method: str,
             bounds: tuple[float, float] | None = None) -> Interval:
    """``estimate ± z·se``, clipped to ``bounds`` (the metric's own range) when given."""
    est, s = _finite(estimate), _finite(se)
    if est is None or s is None:
        return Interval(estimate=est, se=s, method=method)
    z = z_value()
    lo, hi = est - z * s, est + z * s
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return Interval(estimate=est, se=s, ci_low=lo, ci_high=hi, method=method)


# ── influence values, summed within units ────────────────────────────────────


def unit_codes(groups: Any, n: int) -> np.ndarray | None:
    """Integer unit codes for ``groups`` (None: every row its own unit). A missing label is a
    unit of its own row, as the split maps it."""
    if groups is None:
        return None
    import pandas as pd

    labels = np.asarray(groups, dtype=object)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} unit labels for {n} rows")
    keyed = [f"__missing_{i}" if (g is None or (isinstance(g, float) and math.isnan(g))) else str(g)
             for i, g in enumerate(labels)]
    return pd.factorize(np.asarray(keyed, dtype=object))[0]


def variance_of_mean(influence: np.ndarray, codes: np.ndarray | None) -> float:
    """``Var(mean)`` from per-row influence values: ``Σ_g (Σ_{i∈g} IC_i)² / n²``, with the
    ``G/(G − 1)`` small-sample factor of the cluster-robust estimator (``n/(n − 1)`` per row)."""
    ic = np.asarray(influence, dtype=float)
    n = len(ic)
    if n < 2:
        return float("nan")
    sums = ic if codes is None else np.bincount(codes, weights=ic)
    g = len(sums)
    if g < 2:
        return float("nan")
    return float((sums ** 2).sum() / n ** 2 * g / (g - 1))


# ── the AUC ──────────────────────────────────────────────────────────────────


def _midrank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x, method="average")


def auc_components(positive: Any, score: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC and DeLong's structural components, by Sun & Xu's midranks.

    ``V10_i`` (each positive): the share of negatives it outscores, ties counting one half.
    ``V01_j`` (each negative): the share of positives that outscore it. ``AUC = mean(V10) =
    mean(V01)``. Returns (AUC, V10 over positives, V01 over negatives), NaN when a class is absent.
    """
    pos_mask = np.asarray(positive, dtype=bool)
    s = np.asarray(score, dtype=float)
    pos, neg = s[pos_mask], s[~pos_mask]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    tz = _midrank(np.concatenate([pos, neg]))
    v10 = (tz[:m] - _midrank(pos)) / n
    v01 = 1.0 - (tz[m:] - _midrank(neg)) / m
    return float(v10.mean()), v10, v01


def delong(positive: Any, score: Any) -> tuple[float, float]:
    """AUC and its DeLong variance: ``S10/m + S01/n``, the components' sample variances (n − 1)."""
    auc, v10, v01 = auc_components(positive, score)
    if len(v10) < 2 or len(v01) < 2:
        return auc, float("nan")
    return auc, float(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))


def auc_influence(positive: Any, score: Any) -> tuple[float, np.ndarray]:
    """AUC and each row's influence value (LeDell et al. 2015), aligned with the input rows.

    Positives: ``(V10_i − AUC)·n/m``; negatives: ``(V01_j − AUC)·n/n₀``. ``Σ IC²/n²`` is DeLong's
    variance with n in place of n − 1; summed within units first, it is the clustered one.
    """
    pos_mask = np.asarray(positive, dtype=bool)
    auc, v10, v01 = auc_components(pos_mask, score)
    ic = np.zeros(len(pos_mask))
    if not math.isfinite(auc):
        return auc, ic
    n, m = len(pos_mask), int(pos_mask.sum())
    ic[pos_mask] = (v10 - auc) * n / m
    ic[~pos_mask] = (v01 - auc) * n / (n - m)
    return auc, ic


def auc_interval(positive: Any, score: Any, groups: Any = None, unit: str | None = None) -> Interval:
    """The AUC with DeLong's interval; cluster-robust (summed influence values) when ``groups``."""
    pos = np.asarray(positive, dtype=bool)
    codes = unit_codes(groups, len(pos))
    if codes is None or len(np.unique(codes)) == len(codes):
        auc, var = delong(pos, score)
        return interval(auc, _root(var), "DeLong", (0.0, 1.0))
    auc, ic = auc_influence(pos, score)
    var = variance_of_mean(ic, codes)
    return interval(auc, _root(var),
                    f"DeLong, clustered by {unit or 'unit'}", (0.0, 1.0))


# ── per-row losses ───────────────────────────────────────────────────────────


def log_loss_rows(y: Any, proba: Any, classes: Sequence[Any]) -> np.ndarray:
    """Each row's −log p(observed class), with p clipped as scikit-learn's ``log_loss`` does."""
    P = np.asarray(proba, dtype=float)
    P = np.clip(P, np.finfo(P.dtype).eps, 1 - np.finfo(P.dtype).eps)
    P = P / P.sum(axis=1, keepdims=True)
    index = {c: i for i, c in enumerate(classes)}
    col = np.asarray([index.get(v, -1) for v in np.asarray(y, dtype=object).tolist()])
    if (col < 0).any():
        raise ValueError("an outcome level the model never saw")
    return -np.log(P[np.arange(len(col)), col])


def _mean_interval(losses: np.ndarray, codes: np.ndarray | None, method: str,
                   bounds: tuple[float, float] | None = None) -> Interval:
    losses = np.asarray(losses, dtype=float)
    mean = float(losses.mean()) if len(losses) else float("nan")
    var = variance_of_mean(losses - mean, codes)
    return interval(mean, _root(var), method, bounds)


def regression_parts(y: Any, pred: Any, reference: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(e², d², |e|) per row: the model's squared error, the no-predictor model's (against the
    mean of the rows the model was fit on, ``reference``, a scalar or one per row), and |e|."""
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(pred, dtype=float)
    d = y - np.asarray(reference, dtype=float)
    return e ** 2, d ** 2, np.abs(e)


def regression_intervals(e2: np.ndarray, d2: np.ndarray, ae: np.ndarray,
                         codes: np.ndarray | None, method: str) -> dict[str, Interval]:
    """R² = 1 − A/B (A = mean e², B = mean d²), RMSE = √A and MAE, by the delta method.

    IC(R²)_i = −(e²_i − A)/B + A·(d²_i − B)/B² (Hawinkel et al.'s gradient (−1/MST, MSE/MST²)
    applied row by row); IC(RMSE)_i = (e²_i − A)/(2√A).
    """
    n = len(e2)
    if n == 0:
        return {m: Interval(estimate=None, method=method) for m in ("r2", "rmse", "mae")}
    A, B = float(e2.mean()), float(d2.mean())
    out: dict[str, Interval] = {}
    if B > 0:
        ic = -(e2 - A) / B + A * (d2 - B) / B ** 2
        var = variance_of_mean(ic, codes)
        out["r2"] = interval(1 - A / B, _root(var), method, (-math.inf, 1.0))
    else:
        out["r2"] = Interval(estimate=None, method=method)
    rmse = math.sqrt(A)
    if rmse > 0:
        var = variance_of_mean((e2 - A) / (2 * rmse), codes)
        out["rmse"] = interval(rmse, _root(var), method, (0.0, math.inf))
    else:
        out["rmse"] = Interval(estimate=rmse, se=0.0, ci_low=0.0, ci_high=0.0, method=method)
    out["mae"] = _mean_interval(ae, codes, method, (0.0, math.inf))
    return out


def _method(codes: np.ndarray | None, unit: str | None) -> str:
    if codes is not None and len(np.unique(codes)) < len(codes):
        return f"influence function, clustered by {unit or 'unit'}"
    return "influence function"


def score_intervals(task: str, y: Any, prediction: Any, *, classes: Sequence[Any] | None = None,
                    reference: Any = None, groups: Any = None, unit: str | None = None
                    ) -> dict[str, Interval]:
    """Every metric of ``task`` on these scored rows, each with its standard error and interval.

    ``prediction``: ŷ for regression, the class-probability matrix otherwise (columns in
    ``classes``' order). ``reference``: the training mean R² is measured against. Macro-F1 has no
    influence-function SE here and gets none.
    """
    y = np.asarray(y)
    n = len(y)
    if task == "time_to_event":
        return {}  # no interval for a held-out C-index is computed here (WP12b's outcome)
    codes = unit_codes(groups, n)
    method = _method(codes, unit)
    if task == "regression":
        ref = float(np.mean(y.astype(float))) if reference is None else reference
        return regression_intervals(*regression_parts(y, prediction, ref), codes, method)
    P = np.asarray(prediction, dtype=float)
    classes = list(classes or [])
    out: dict[str, Interval] = {}
    if task == "binary":
        positive = y == classes[1]
        out["auc"] = (auc_interval(positive, P[:, 1], groups, unit) if codes is not None
                      else auc_interval(positive, P[:, 1]))
        out["brier"] = _mean_interval((P[:, 1] - positive) ** 2, codes, method, (0.0, 1.0))
        out["log_loss"] = _mean_interval(log_loss_rows(y, P, classes), codes, method, (0.0, math.inf))
        return out
    pred = np.asarray(classes, dtype=object)[P.argmax(axis=1)]
    out["accuracy"] = _mean_interval((pred == y.astype(object)).astype(float), codes, method,
                                     (0.0, 1.0))
    out["log_loss"] = _mean_interval(log_loss_rows(y, P, classes), codes, method, (0.0, math.inf))
    return out


# ── standard errors of cross-validated scores ────────────────────────────────


def fold_variance(task: str, metric: str, y: Any, prediction: Any, *, classes: Sequence[Any],
                  groups: Any = None) -> float | None:
    """The variance of one fold's score (a fold-mean metric) from its rows' influence values."""
    y = np.asarray(y)
    codes = unit_codes(groups, len(y))
    P = np.asarray(prediction, dtype=float)
    if metric == "auc":
        positive = y == classes[1]
        if positive.sum() < 2 or (~positive).sum() < 2:
            return None
        if codes is None or len(np.unique(codes)) == len(codes):
            var = delong(positive, P[:, 1])[1]  # DeLong's own, as on the held-out rows
        else:
            var = variance_of_mean(auc_influence(positive, P[:, 1])[1], codes)
    elif metric == "brier":
        loss = (P[:, 1] - (y == classes[1])) ** 2
        var = variance_of_mean(loss - loss.mean(), codes)
    elif metric == "log_loss":
        loss = log_loss_rows(y, P, classes)
        var = variance_of_mean(loss - loss.mean(), codes)
    elif metric == "accuracy":
        hit = (np.asarray(classes, dtype=object)[P.argmax(axis=1)] == y.astype(object)).astype(float)
        var = variance_of_mean(hit - hit.mean(), codes)
    else:
        return None
    return var if math.isfinite(var) else None


# ── calibration ──────────────────────────────────────────────────────────────


class CurvePoint(_Model):
    x: float  # predicted (probability, or the predicted outcome)
    y: float  # observed, smoothed


class Calibration(_Model):
    """Calibration of one set of predictions: in the large, its slope, and its smoothed curve."""

    n: int
    observed: float  # mean outcome (event rate, or the outcome's mean)
    expected: float  # mean prediction
    intercept: Interval  # target 0: calibration-in-the-large (log-odds, or outcome units)
    slope: Interval  # target 1: below 1, predictions too extreme
    curve: list[CurvePoint] = []
    eavg: float | None = None  # mean |prediction − smoothed observed| (the ICI)
    e90: float | None = None
    emax: float | None = None
    smoother: str = SMOOTHER
    flagged: bool = False
    concern: str | None = None


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def logistic_fit(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                 codes: np.ndarray | None = None, max_iter: int = 50
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Maximum-likelihood logistic regression by Newton–Raphson: (coefficients, covariance).

    The covariance is the inverse information, or the cluster-robust sandwich when ``codes`` give
    each row's unit. None when the fit does not converge (separation).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = off + X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu)
        info = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(info, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        return None
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 50:
        return None
    eta = off + X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    info = X.T @ (X * (mu * (1 - mu))[:, None])
    try:
        bread = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    if codes is None or len(np.unique(codes)) == len(codes):
        return beta, bread
    scores = X * (y - mu)[:, None]
    sums = np.vstack([np.bincount(codes, weights=scores[:, j]) for j in range(X.shape[1])]).T
    g = len(sums)
    meat = sums.T @ sums * g / (g - 1)
    return beta, bread @ meat @ bread


def _smooth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """lowess(x, y, iter = 0) fitted at every x (R's defaults: f = 2/3, delta = 1% of the range)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    span = float(x.max() - x.min()) if len(x) else 0.0
    return lowess(y, x, frac=2.0 / 3.0, it=0, delta=0.01 * span, return_sorted=False)


def _curve(x: np.ndarray, fitted: np.ndarray, bounds: tuple[float, float] | None) -> list[CurvePoint]:
    """The smoothed curve at up to :data:`CURVE_POINTS` quantiles of the predictions."""
    order = np.argsort(x, kind="stable")
    xs, fs = x[order], fitted[order]
    picks = np.unique(np.round(np.linspace(0, len(xs) - 1, min(CURVE_POINTS, len(xs)))).astype(int))
    out = []
    for i in picks:
        yv = float(fs[i])
        if bounds is not None:
            yv = min(max(yv, bounds[0]), bounds[1])
        out.append(CurvePoint(x=float(xs[i]), y=yv))
    return out


def _signed(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}".replace("-", "−")


def _flag(task: str, intercept: Interval, slope: Interval, observed: float, expected: float,
          scale: float, where: str) -> str | None:
    """The concern a calibration earns (module docstring), or None."""
    parts: list[str] = []
    s = slope
    if s.estimate is not None and s.ci_low is not None and s.ci_high is not None \
            and (s.ci_high < 1.0 - SLOPE_TOLERANCE or s.ci_low > 1.0 + SLOPE_TOLERANCE):
        ci = f"95% interval {_signed(s.ci_low)} to {_signed(s.ci_high)}"
        if s.estimate < 1:
            what = ("high risks run too high and low risks too low" if task == "binary"
                    else "high predictions run too high and low ones too low")
            parts.append(f"predictions are too extreme {where}: calibration slope "
                         f"{_signed(s.estimate)} ({ci}); {what}")
        else:
            parts.append(f"predictions are too timid {where}: calibration slope "
                         f"{_signed(s.estimate)} ({ci}); they spread less than the outcomes do")
    a = intercept
    tol = INTERCEPT_TOLERANCE * (1.0 if task == "binary" else scale)
    if a.estimate is not None and a.ci_low is not None and a.ci_high is not None \
            and (a.ci_high < -tol or a.ci_low > tol):
        high = a.estimate < 0  # observed below predicted
        if task == "binary":
            parts.append(f"predicted risks run too {'high' if high else 'low'} on average {where}: "
                         f"{expected:.1%} predicted against {observed:.1%} observed (calibration "
                         f"intercept {_signed(a.estimate)})")
        else:
            parts.append(f"predictions run too {'high' if high else 'low'} on average {where}: by "
                         f"{abs(a.estimate):.3g} (calibration-in-the-large)")
    if not parts:
        return None
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def calibration(task: str, y: Any, prediction: Any, *, classes: Sequence[Any] | None = None,
                groups: Any = None, where: str = "out of fold") -> Calibration | None:
    """Calibration of these predictions (module docstring); None for multiclass or too few rows.

    ``prediction``: ŷ for regression; the probability matrix for binary (columns in ``classes``'
    order). ``where`` words the concern ("out of fold", "on the held-out rows").
    """
    if task not in ("regression", "binary"):
        return None
    y = np.asarray(y)
    n = len(y)
    if n < 10:
        return None
    codes = unit_codes(groups, n)
    clustered = codes is not None and len(np.unique(codes)) < n
    method = "cluster-robust Wald" if clustered else "Wald"
    if task == "binary":
        P = np.asarray(prediction, dtype=float)
        p = P[:, 1] if P.ndim == 2 else P
        event = (y == list(classes or [])[1]).astype(float) if classes is not None else y.astype(float)
        if event.min() == event.max():
            return None
        lp = _logit(p)
        fit_a = logistic_fit(np.ones((n, 1)), event, offset=lp, codes=codes if clustered else None)
        # Predictions that do not vary have no slope (a penalty that removed every predictor).
        fit_b = (logistic_fit(np.column_stack([np.ones(n), lp]), event,
                              codes=codes if clustered else None)
                 if float(np.std(lp)) > FLAT else None)
        a = (interval(fit_a[0][0], _root(fit_a[1][0, 0]), method) if fit_a
             else Interval(estimate=None, method=method))
        b = (interval(fit_b[0][1], _root(fit_b[1][1, 1]), method) if fit_b
             else Interval(estimate=None, method=method))
        fitted = _smooth(p, event)
        err = np.abs(p - fitted)
        observed, expected = float(event.mean()), float(p.mean())
        cal = Calibration(n=n, observed=observed, expected=expected, intercept=a, slope=b,
                          curve=_curve(p, fitted, (0.0, 1.0)), eavg=float(err.mean()),
                          e90=float(np.quantile(err, 0.9)), emax=float(err.max()))
        scale = 1.0
    else:
        yy = y.astype(float)
        yhat = np.asarray(prediction, dtype=float)
        resid = yy - yhat
        a = _mean_interval(resid, codes if clustered else None, method)
        X = np.column_stack([np.ones(n), yhat])
        try:
            M = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            M = None
        if M is None or float(np.std(yhat)) <= FLAT * max(float(np.std(yy)), EPS):
            b = Interval(estimate=None, method=method)
        else:
            coef = M @ X.T @ yy
            e = yy - X @ coef
            if clustered:
                scores = X * e[:, None]
                sums = np.vstack([np.bincount(codes, weights=scores[:, j]) for j in range(2)]).T
                g = len(sums)
                V = M @ (sums.T @ sums) @ M * g / (g - 1)
            else:
                V = M * float(e @ e) / (n - 2)
            b = interval(coef[1], _root(V[1, 1]), method)
        fitted = _smooth(yhat, yy)
        err = np.abs(yhat - fitted)
        cal = Calibration(n=n, observed=float(yy.mean()), expected=float(yhat.mean()), intercept=a,
                          slope=b, curve=_curve(yhat, fitted, None), eavg=float(err.mean()),
                          e90=float(np.quantile(err, 0.9)), emax=float(err.max()))
        scale = float(np.std(yy, ddof=1)) if n > 1 else 1.0
    concern = _flag(task, cal.intercept, cal.slope, cal.observed, cal.expected, scale, where)
    cal.flagged = concern is not None
    cal.concern = concern
    return cal


# ── across clusters: random-effects summary ─────────────────────────────────


class Pooled(_Model):
    """A DerSimonian–Laird random-effects summary of one score across clusters."""

    metric: str
    scale: Literal["identity", "logit"]
    k: int  # clusters pooled
    estimate: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    tau: float | None = None  # between-cluster SD, on ``scale``
    i2: float | None = None  # share of the spread beyond chance
    pi_low: float | None = None  # 95% prediction interval for a new cluster (k ≥ 3)
    pi_high: float | None = None


def random_effects(metric: str, estimates: Sequence[float], ses: Sequence[float],
                   scale: Literal["identity", "logit"] = "identity") -> Pooled:
    """DerSimonian–Laird pooling of per-cluster ``estimates`` with standard errors ``ses``.

    On the logit scale each estimate θ becomes logit θ with SE ``se/(θ(1 − θ))`` (the delta
    method), and every result is transformed back.
    """
    from scipy import stats

    th = np.asarray(estimates, dtype=float)
    se = np.asarray(ses, dtype=float)
    keep = np.isfinite(th) & np.isfinite(se) & (se > 0)
    if scale == "logit":
        keep &= (th > 0) & (th < 1)
    th, se = th[keep], se[keep]
    k = int(len(th))
    if k == 0:
        return Pooled(metric=metric, scale=scale, k=0, estimate=None)
    if scale == "logit":
        se = se / (th * (1 - th))
        th = np.log(th / (1 - th))
    back = (lambda v: 1 / (1 + math.exp(-v))) if scale == "logit" else (lambda v: v)
    if k == 1:
        return Pooled(metric=metric, scale=scale, k=1, estimate=float(back(th[0])))
    w = 1.0 / se ** 2
    fixed = float((w * th).sum() / w.sum())
    q = float((w * (th - fixed) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    mu = float((ws * th).sum() / ws.sum())
    se_mu = math.sqrt(1.0 / ws.sum())
    z = z_value()
    out = Pooled(metric=metric, scale=scale, k=k, estimate=float(back(mu)),
                 ci_low=float(back(mu - z * se_mu)), ci_high=float(back(mu + z * se_mu)),
                 tau=math.sqrt(tau2), i2=max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0)
    if k >= 3:
        half = float(stats.t.ppf(0.5 + LEVEL / 2, k - 2)) * math.sqrt(tau2 + se_mu ** 2)
        out.pi_low, out.pi_high = float(back(mu - half)), float(back(mu + half))
    return out


__all__ = [
    "Calibration", "CurvePoint", "INTERCEPT_TOLERANCE", "Interval", "LEVEL", "Pooled",
    "SLOPE_TOLERANCE", "auc_components", "auc_influence", "auc_interval", "calibration", "delong",
    "fold_variance", "interval", "log_loss_rows", "logistic_fit", "random_effects",
    "regression_intervals", "regression_parts", "score_intervals", "unit_codes",
    "variance_of_mean", "z_value",
]
