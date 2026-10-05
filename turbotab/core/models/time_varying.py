"""Time-varying exposures by g-methods (V2_DEFINITION_OF_DONE §2, the causal-inference row).

A long table holds one row per unit per time point: the exposure ``A_t``, the time-varying
confounders ``L_t`` measured before it, the baseline covariates ``V`` and the outcome that follows.
When a confounder of a later exposure is itself changed by an earlier exposure, standard regression
cannot adjust for it. Robins, Hernán & Brumback (2000, *Epidemiology* 11:550–560, abstract): "In
observational studies with exposures or treatments that vary over time, standard approaches for
adjustment of confounding are biased when there exist time-dependent confounders that are also
affected by previous treatment." Two g-methods adjust for such a confounder without blocking the
earlier exposure's effect through it. This module computes both, and the diagnostics that come
before their estimates. It reads numbers only; the routing that decides when it runs is
``turbotab/core/time_varying.py``, and the stage that runs it is ``stages/time_varying.py``.

**Inverse-probability weights** (:func:`ipw_weights`). The weights are R ``ipw::ipwtm``'s for
``family = "binomial", link = "logit"`` (van der Wal & Geskus 2011, *J Stat Softw* 43(13)), computed
the way its source does. The rows are sorted by unit and time. Pooled logistic models for the
numerator and the denominator are fit on the rows the ``type`` selects:

* ``"all"``: every row (an exposure that can switch on and off);
* ``"first"``: each unit's rows up to and including its first exposed row (an exposure that, once
  started, stays: initiation);
* ``"cens"``: the same rows for a censoring indicator.

Each row's factor is the fitted probability of what happened: ``p̂`` for an exposed row, ``1 − p̂`` for
an unexposed one, ``1 − p̂`` for the censoring row under ``"cens"``, and 1 after the first event under
``"first"`` or ``"cens"``. The stabilized weight is the ratio of the cumulative products of the
numerator's and the denominator's factors within the unit. Cole & Hernán (2008, *Am J Epidemiol*
168:656–664) say "A necessary condition for correct model specification is that the stabilized
weights have a mean of one" and "Estimated weights with the mean far from one or very extreme values
are indicative of nonpositivity or misspecification of the weight model". So the distribution is
summarized (:func:`summarize`, :func:`by_time`) before any estimate.

**Loss to follow-up** (:func:`censoring_weights`). The indicator is 1 on a unit's last observed time
point when it was lost after it. That row's outcome was seen, so the analyzed rows keep it. The
outcome at time point t is seen when the unit stayed through every earlier time point, so a row's
censoring weight is the inverse probability of having stayed through the previous one. That is
ipwtm's ``"cens"`` product read at the unit's previous row, with the models fit on the rows at risk
of being lost (a row with the event ends follow-up). The product at the row itself would add the
probability of staying after its own outcome was seen. That factor depends on the confounders at that
time point and would unbalance them.

**Truncation** (:func:`truncate`) is ipwtm's ``trunc``: a weight at or below the ``p``-th percentile
is set to it, and one above the ``(1 − p)``-th to that (R's type-7 quantile, NumPy's ``linear``).
Cole & Hernán: "Assuming that the marginal structural model estimate is correct, one can see the
growing bias as the weights are progressively truncated. Simultaneously, one can see the increasing
precision as the weights are progressively truncated."

**Positivity per time point** (:func:`positivity`): at each time, how many rows were exposed and how
many not, and the range of the denominator's fitted probability of exposure. A time with no exposed
(or no unexposed) row, or fitted probabilities at 0 or 1, is where the estimate rests on the model.

**The marginal structural model** (:func:`fit_msm`) is a weighted generalized linear model (logit for
an event in each interval, identity for a repeated measure) whose interval treats each unit as a
cluster: the sandwich ``A⁻¹BA⁻¹`` with ``A = Σ w_i v(μ_i) x_i x_iᵀ`` and ``B = Σ_g U_g U_gᵀ``,
``U_g = Σ_{i∈g} w_i (y_i − μ_i) x_i``. That is the robust variance of R ``geepack::geeglm`` with an
independence working correlation (Liang & Zeger 1986). Hernán, Brumback & Robins (2000, *Epidemiology*
11:561–570) call this interval conservative ("95% conservative confidence interval"): it treats the
weights as known.

**The parametric g-formula** (:func:`gformula`) is R ``gfoRmula``'s algorithm for a survival outcome
(McGrath et al. 2020, *Patterns* 1:100008), read from its source (``simulate``, ``pred_fun_cov``,
``pred_fun_Y``, ``lagged``):

* the covariate models are fit on the rows after the first time point;
* the outcome model is fit on every row (a unit lost to follow-up contributes the time points it
  was seen; the risks are those had no unit been lost, assuming loss depends only on the measured
  history);
* a binary covariate is drawn from its logistic model, and a normal one from its linear model with
  the residual root mean square, cut to the covariate's observed range;
* the time-varying covariates at the first time point are the unit's observed ones, and lagged
  values before it are 0;
* the risk is ``Σ_t h_t Π_{k<t}(1 − h_k)`` from the predicted hazards, averaged over the simulated
  units, and every strategy uses the same random numbers.

The Monte Carlo sample is the units themselves when ``n_sim`` equals their number, else that many
drawn with replacement. Its Monte Carlo error is reported as ``sd(r_i)/√n_sim`` over the simulated
units' risks ``r_i``, which bounds it from above. The natural course (each exposure drawn from its own
model) is compared with the observed risk as a check of the models (:func:`observed_risk`). The
interval is the percentile bootstrap over units (:func:`gformula_bootstrap`), refitting every model.

Every model here enters time as ``t`` and ``t²`` of the time point's index (0, 1, …, K − 1), the
specification of Hernán & Robins's pooled logistic models (*Causal Inference: What If*, ch. 17
programs: "time + timesq"). Nothing here imports R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from turbotab.core.models.effects import RARE_OUTCOME, e_values

Z95 = float(norm.ppf(0.975))
TIME, TIME2 = "__t", "__t2"  # the time point's index and its square, in every model
LAG = "lag1_"  # the previous time point's value of a column (0 at the first time point)
WeightKind = Literal["all", "first", "cens"]
Family = Literal["binomial", "gaussian"]
CovariateKind = Literal["binary", "normal"]
# The truncation options: the percentile below which (and above whose complement) weights are set
# to it; ``none`` keeps them as estimated (Cole & Hernán 2008).
TRUNCATIONS: dict[str, float | None] = {"none": None, "p1_p99": 0.01, "p5_p95": 0.05}
# A fitted probability of exposure this close to 0 or 1 is counted as a near-violation of
# positivity: the app's convention for the count it shows, not a test.
NEAR = 0.01


class NotEstimable(ValueError):
    """A model that cannot be fit as declared: an indicator the covariates separate perfectly (no
    exposed, or no unexposed, row in some stratum: a positivity violation), or no rows at all."""


# ── long tables ─────────────────────────────────────────────────────────────


def time_index(values: Any) -> np.ndarray:
    """Each row's time point as its rank among the distinct times, 0 … K − 1."""
    return np.unique(np.asarray(values), return_inverse=True)[1].astype(np.int64)


def sort_long(data: pd.DataFrame, id: str, time: str) -> tuple[pd.DataFrame, np.ndarray]:
    """``data`` sorted by unit and time (stably, as R's ``order(id, time)``), and the positions
    that put the sorted rows back in the original order."""
    order = np.lexsort((np.asarray(data[time]), pd.factorize(data[id], sort=True)[0]))
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return data.iloc[order].reset_index(drop=True), back


def with_history(data: pd.DataFrame, id: str, time: str, lagged: Iterable[str] = ()) -> pd.DataFrame:
    """``data`` (sorted by unit and time) with ``__t``, ``__t2`` and ``lag1_<c>`` for each ``c``:
    the unit's value at the previous time point, 0 at its first (``gfoRmula``'s ``lagged`` with
    ``baselags = FALSE``)."""
    out = data.copy()
    t = time_index(out[time]).astype(float)
    out[TIME] = t
    out[TIME2] = t * t
    for c in lagged:
        out[LAG + c] = out.groupby(id, sort=False)[c].shift(1).fillna(0.0).astype(float).to_numpy()
    return out


def check_schedule(data: pd.DataFrame, id: str, time: str) -> dict[str, int]:
    """How the units' time points line up: ``late`` units whose first row is not the first time
    point, ``gaps`` units with a time point missing between their first and last, ``repeats``
    units with two rows at one time point."""
    t = time_index(data[time])
    frame = pd.DataFrame({"id": data[id].to_numpy(), "t": t})
    g = frame.groupby("id", sort=False)["t"]
    first, last, n, distinct = g.min(), g.max(), g.size(), g.nunique()
    return {"units": int(len(first)), "late": int((first > 0).sum()),
            "gaps": int(((last - first + 1) != distinct).sum()),
            "repeats": int((n != distinct).sum()), "time_points": int(t.max() + 1) if len(t) else 0}


# ── generalized linear models, fit exactly ──────────────────────────────────


@dataclass(frozen=True)
class GLMFit:
    """A fitted logistic or linear model: coefficients by term (``(Intercept)`` first), the terms
    the design dropped as aliased (R's ``NA`` coefficients), and for a linear model the residual
    root mean square that ``gfoRmula`` draws with."""

    family: str
    names: tuple[str, ...]
    coef: np.ndarray
    kept: tuple[int, ...]
    dropped: tuple[str, ...]
    iterations: int
    rmse: float | None = None

    def coefficients(self) -> dict[str, float]:
        return {self.names[k]: float(b) for k, b in zip(self.kept, self.coef)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        eta = X[:, list(self.kept)] @ self.coef
        return expit(eta) if self.family == "binomial" else eta


def design(data: pd.DataFrame | Mapping[str, Any], terms: Sequence[str], n: int | None = None
           ) -> tuple[np.ndarray, tuple[str, ...]]:
    """The model matrix: an intercept, then each term's column as a float."""
    if n is None:
        n = len(data) if isinstance(data, pd.DataFrame) else len(next(iter(data.values())))
    cols = [np.ones(n)]
    for t in terms:
        v = np.asarray(data[t], dtype=float)
        cols.append(np.broadcast_to(v, (n,)).astype(float) if v.ndim == 0 else v)
    return np.column_stack(cols), ("(Intercept)", *terms)


def _independent(X: np.ndarray, tol: float = 1e-7) -> list[int]:
    """Columns kept left to right while each adds a direction the earlier ones lack (R's ``qr``
    drops a column whose residual is below ``tol`` of its norm, and reports it ``NA``)."""
    kept: list[int] = []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j]
        norm_v = float(np.linalg.norm(v))
        if norm_v == 0.0:
            continue
        r = v - Q @ (Q.T @ v)
        r = r - Q @ (Q.T @ r)
        if float(np.linalg.norm(r)) > tol * norm_v:
            kept.append(j)
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
    return kept


def fit_glm(X: np.ndarray, y: Any, names: Sequence[str], family: Family = "binomial",
            weights: Any = None, *, tol: float = 1e-12, max_iter: int = 100) -> GLMFit:
    """The maximum-likelihood fit, converged far past R ``glm``'s default (relative change in the
    deviance below 1e-8): Newton–Raphson with step halving for the logit, weighted least squares
    by QR for the identity. A logit that does not converge is separated, and is refused."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] == 0:
        raise NotEstimable("there are no rows to fit the model on")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    kept = _independent(X)
    dropped = tuple(names[j] for j in range(X.shape[1]) if j not in kept)
    Z = X[:, kept]
    if family == "gaussian":
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)
        resid = y - Z @ beta
        return GLMFit("gaussian", tuple(names), beta, tuple(kept), dropped, 1,
                      rmse=float(np.sqrt(np.mean(resid ** 2))))
    if np.any((y != 0) & (y != 1)):
        raise NotEstimable("a logistic model needs an outcome of 0 and 1")

    def deviance(b: np.ndarray) -> float:
        mu = np.clip(expit(Z @ b), 1e-300, 1 - 1e-16)
        return float(-2.0 * np.sum(w * (y * np.log(mu) + (1 - y) * np.log1p(-mu))))

    beta = np.zeros(Z.shape[1])
    dev = deviance(beta)
    for it in range(1, max_iter + 1):
        mu = expit(Z @ beta)
        W = w * mu * (1 - mu)
        H = (Z * W[:, None]).T @ Z
        g = Z.T @ (w * (y - mu))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as exc:
            raise NotEstimable("the model's information is singular (an indicator the covariates "
                               "separate perfectly)") from exc
        new = beta + step
        new_dev = deviance(new)
        halvings = 0
        while new_dev > dev + 1e-10 * (abs(dev) + 1) and halvings < 30:
            step = step / 2
            new = beta + step
            new_dev = deviance(new)
            halvings += 1
        beta, old, dev = new, dev, new_dev
        if np.max(np.abs(step)) <= tol * (1.0 + np.max(np.abs(beta))) and abs(old - dev) <= 1e-10 * (abs(dev) + 0.1):
            fitted = expit(Z @ beta)
            if np.max(np.abs(beta)) > 30 and (fitted.min() < 1e-10 or fitted.max() > 1 - 1e-10):
                raise NotEstimable("the covariates separate the indicator perfectly: some rows have "
                                   "a fitted probability of 0 or 1 (a positivity violation)")
            return GLMFit("binomial", tuple(names), beta, tuple(kept), dropped, it)
    raise NotEstimable("the logistic model did not converge: the covariates separate the indicator "
                       "(no exposed, or no unexposed, rows in some stratum: a positivity violation)")


# ── inverse-probability weights (R ``ipw::ipwtm``, binomial, logit) ───────────


@dataclass(frozen=True)
class WeightModel:
    """One set of stabilized weights, in the rows' original order. ``p_event``: the denominator
    model's fitted probability that the indicator is 1, on the rows it was fit on (NaN elsewhere);
    ``modeled``: those rows."""

    kind: str
    weights: np.ndarray
    numerator_product: np.ndarray
    denominator_product: np.ndarray
    p_event: np.ndarray
    modeled: np.ndarray
    numerator: GLMFit | None
    denominator: GLMFit


def _selected(ids: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    """ipwtm's ``selvar``: every row for ``all``; else each unit's rows up to and including its
    first 1 (every row of a unit that never has one)."""
    if kind == "all":
        return np.ones(len(a), dtype=bool)
    hit = (a == 1).astype(int)
    before = pd.Series(hit).groupby(ids, sort=False).cumsum().to_numpy() - hit
    return before == 0


def _factors(fit: GLMFit | None, X: np.ndarray, a: np.ndarray, sel: np.ndarray,
             kind: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.full(len(a), np.nan)
    f = np.ones(len(a))
    if fit is None:
        return f, p
    p[sel] = fit.predict(X[sel])
    one = sel & (a == 1)
    zero = sel & (a == 0)
    f[zero] = 1.0 - p[zero]
    f[one] = (1.0 - p[one]) if kind == "cens" else p[one]
    return f, p


def ipw_weights(data: pd.DataFrame, *, id: str, time: str, indicator: str,
                numerator: Sequence[str] | None, denominator: Sequence[str],
                kind: WeightKind) -> WeightModel:
    """Stabilized weights for ``indicator`` (the exposure, or a censoring indicator under
    ``"cens"``), as ``ipwtm(exposure = indicator, family = "binomial", link = "logit", numerator =
    ~ numerator, denominator = ~ denominator, id, timevar = time, type = kind)``. ``numerator`` None
    is ipwtm's absent numerator: unstabilized weights. Every term is a numeric column of ``data``."""
    if kind not in ("all", "first", "cens"):
        raise ValueError(f"unknown weight type {kind!r}")
    d, back = sort_long(data, id, time)
    a = np.asarray(d[indicator], dtype=float)
    if np.isnan(a).any():
        raise NotEstimable(f"`{indicator}` has missing values")
    ids = d[id].to_numpy()
    sel = _selected(ids, a, kind)
    Xd, nd = design(d, denominator)
    den = fit_glm(Xd[sel], a[sel], nd)
    fd, pd_ = _factors(den, Xd, a, sel, kind)
    num = None
    fn = np.ones(len(a))
    if numerator is not None:
        Xn, nn = design(d, numerator)
        num = fit_glm(Xn[sel], a[sel], nn)
        fn, _ = _factors(num, Xn, a, sel, kind)
    wd = pd.Series(fd).groupby(ids, sort=False).cumprod().to_numpy()
    wn = pd.Series(fn).groupby(ids, sort=False).cumprod().to_numpy()
    return WeightModel(kind, (wn / wd)[back], wn[back], wd[back], pd_[back], sel[back], num, den)


def censoring_weights(data: pd.DataFrame, *, id: str, time: str, indicator: str,
                      numerator: Sequence[str] | None, denominator: Sequence[str],
                      at_risk: Any = None) -> WeightModel:
    """Stabilized weights for loss to follow-up, where ``indicator`` is 1 on a unit's last observed
    time point when it was lost after it (that row's outcome was seen, so the cohort keeps it).

    The outcome at time point t is seen when the unit stayed through every earlier one, so a row's
    weight is the inverse probability of having stayed through the *previous* time point. That is
    ipwtm's ``type = "cens"`` product read at the unit's previous row (1 at its first). The models
    are fit on the rows at risk of being lost (``at_risk``: a row with an event ends follow-up, so it
    is not). ``modeled``, ``p_event`` and the products are ipwtm's on those rows."""
    risk = np.ones(len(data), dtype=bool) if at_risk is None else np.asarray(at_risk, dtype=bool)
    fit = ipw_weights(data.loc[risk], id=id, time=time, indicator=indicator, numerator=numerator,
                      denominator=denominator, kind="cens")
    product = np.full(len(data), np.nan)
    product[risk] = fit.weights
    d = pd.DataFrame({"id": data[id].to_numpy(), "t": np.asarray(data[time]), "w": product,
                      "row": np.arange(len(data))})
    d = d.sort_values(["id", "t"], kind="mergesort")
    lagged = d.groupby("id", sort=False)["w"].shift(1).fillna(1.0).to_numpy()
    weights = np.empty(len(data))
    weights[d["row"].to_numpy()] = lagged

    def spread(values: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(len(data), fill)
        out[risk] = values
        return out

    return WeightModel("cens_lagged", weights, spread(fit.numerator_product, np.nan),
                       spread(fit.denominator_product, np.nan), spread(fit.p_event, np.nan),
                       spread(fit.modeled, False).astype(bool), fit.numerator, fit.denominator)


def truncate(weights: Any, level: float | None) -> np.ndarray:
    """ipwtm's ``trunc``: weights at or below the ``level`` quantile set to it, weights above the
    ``1 − level`` quantile set to that (type-7 quantiles). ``None`` keeps them."""
    w = np.asarray(weights, dtype=float)
    if level is None:
        return w.copy()
    if not 0 <= level <= 0.5:
        raise ValueError("a truncation level is between 0 and 0.5")
    lo, hi = np.quantile(w, level), np.quantile(w, 1 - level)
    out = w.copy()
    out[w <= lo] = lo
    out[w > hi] = hi
    return out


SUMMARY_QUANTILES = (("p1", 0.01), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p99", 0.99))


def summarize(weights: Any) -> dict[str, float | int]:
    """The weights' distribution: n, mean, SD (n − 1), min, the 1st, 25th, 50th, 75th and 99th
    percentiles (type 7) and max."""
    w = np.asarray(weights, dtype=float)
    out: dict[str, float | int] = {"n": int(w.size), "mean": float(w.mean()),
                                   "sd": float(w.std(ddof=1)) if w.size > 1 else 0.0,
                                   "min": float(w.min())}
    for name, q in SUMMARY_QUANTILES:
        out[name] = float(np.quantile(w, q))
    out["max"] = float(w.max())
    return out


def by_time(weights: Any, t: Any) -> list[dict[str, float | int]]:
    """:func:`summarize` at each time point (``t`` its index), in time order."""
    w = np.asarray(weights, dtype=float)
    t = np.asarray(t)
    return [{"time": int(k), **summarize(w[t == k])} for k in np.unique(t)]


def positivity(t: Any, exposed: Any, p_exposed: Any, modeled: Any,
               near: float = NEAR) -> list[dict[str, Any]]:
    """At each time point, over the rows the exposure model was fit on: rows, exposed, unexposed,
    the range of the fitted probability of exposure, and how many rows sit within ``near`` of 0 or
    of 1."""
    t = np.asarray(t)
    a = np.asarray(exposed, dtype=float)
    p = np.asarray(p_exposed, dtype=float)
    m = np.asarray(modeled, dtype=bool)
    out = []
    for k in np.unique(t[m]):
        rows = m & (t == k)
        pk = p[rows]
        out.append({"time": int(k), "rows": int(rows.sum()), "exposed": int((a[rows] == 1).sum()),
                    "unexposed": int((a[rows] == 0).sum()),
                    "p_min": float(pk.min()), "p_max": float(pk.max()),
                    "near_zero": int((pk < near).sum()), "near_one": int((pk > 1 - near).sum())})
    return out


# ── the marginal structural model ────────────────────────────────────────────


@dataclass(frozen=True)
class MSMFit:
    """The weighted model with its unit-clustered sandwich variance (independence GEE)."""

    family: str
    names: tuple[str, ...]
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    n_rows: int
    n_units: int

    def row(self, name: str) -> dict[str, float]:
        j = self.names.index(name)
        b, s = float(self.coef[j]), float(self.se[j])
        z = b / s if s > 0 else math.nan
        return {"estimate": b, "se": s, "ci_low": b - Z95 * s, "ci_high": b + Z95 * s,
                "z": z, "p": float(2 * norm.sf(abs(z))) if s > 0 else math.nan}


def fit_msm(X: np.ndarray, names: Sequence[str], y: Any, weights: Any, units: Any,
            family: Family) -> MSMFit:
    """The marginal structural model's weighted fit and its robust variance, clustered by unit
    (R ``geeglm(..., weights, id, corstr = "independence")``'s ``san.se``)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    fit = fit_glm(X, y, names, family, w)
    if fit.dropped:
        raise NotEstimable(f"the model's terms are aliased: {', '.join(fit.dropped)}")
    mu = fit.predict(X)
    v = mu * (1 - mu) if family == "binomial" else np.ones(len(y))
    A = (X * (w * v)[:, None]).T @ X
    scores = X * (w * (y - mu))[:, None]
    codes = pd.factorize(pd.Series(np.asarray(units)))[0]
    U = np.zeros((int(codes.max()) + 1, X.shape[1]))
    np.add.at(U, codes, scores)
    Ainv = np.linalg.inv(A)
    V = Ainv @ (U.T @ U) @ Ainv
    return MSMFit(family, tuple(names), fit.coef, np.sqrt(np.diag(V)), V, len(y), int(U.shape[0]))


# ── the parametric g-formula (R ``gfoRmula``, survival outcome) ────────────────


@dataclass(frozen=True)
class Covariate:
    """A time-varying covariate the g-formula simulates, in its order: its model's terms may name
    the baseline covariates, earlier covariates' current values, ``lag1_<c>`` and ``__t``/``__t2``."""

    name: str
    kind: CovariateKind
    terms: tuple[str, ...]


@dataclass(frozen=True)
class GFormulaSpec:
    id: str
    time: str
    exposure: str
    outcome: str
    covariates: tuple[Covariate, ...]
    exposure_terms: tuple[str, ...]
    outcome_terms: tuple[str, ...]
    baseline: tuple[str, ...] = ()
    lagged: tuple[str, ...] = ()
    # An exposure that once started stays (initiation): its model is fit on the rows not yet
    # exposed, and the natural course carries it forward.
    absorbing: bool = False


@dataclass(frozen=True)
class GFormulaModels:
    covariates: dict[str, GLMFit]
    exposure: GLMFit
    outcome: GLMFit
    ranges: dict[str, tuple[float, float]]


@dataclass
class GFormulaResult:
    """Risks by the end of each time point, per strategy (``natural`` is the natural course), the
    Monte Carlo sample and its error, and the observed risk for the natural-course check."""

    time_points: int
    n_sim: int
    seed: int
    risks: dict[str, list[float]]
    mc_se: dict[str, float]
    observed: list[float]
    models: GFormulaModels
    unit_risks: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def final(self, strategy: str) -> float:
        return self.risks[strategy][-1]


def _prepared(data: pd.DataFrame, spec: GFormulaSpec) -> pd.DataFrame:
    d, _ = sort_long(data, spec.id, spec.time)
    lagged = tuple(dict.fromkeys(spec.lagged))
    return with_history(d, spec.id, spec.time, lagged)


def fit_gformula(d: pd.DataFrame, spec: GFormulaSpec) -> GFormulaModels:
    """Every model the simulation draws from, fit as ``gfoRmula`` fits them (``d`` from
    :func:`with_history`)."""
    later = d[TIME] > 0
    fits: dict[str, GLMFit] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for c in spec.covariates:
        rows = d.loc[later]
        X, names = design(rows, c.terms)
        family: Family = "binomial" if c.kind == "binary" else "gaussian"
        fits[c.name] = fit_glm(X, rows[c.name].to_numpy(float), names, family)
        if c.kind == "normal":
            ranges[c.name] = (float(d[c.name].min()), float(d[c.name].max()))
    rows = d.loc[later & ((d[LAG + spec.exposure] == 0) if spec.absorbing else True)]
    X, names = design(rows, spec.exposure_terms)
    exposure = fit_glm(X, rows[spec.exposure].to_numpy(float), names)
    X, names = design(d, spec.outcome_terms)
    outcome = fit_glm(X, d[spec.outcome].to_numpy(float), names)
    return GFormulaModels(fits, exposure, outcome, ranges)


def _simulate(base: pd.DataFrame, spec: GFormulaSpec, models: GFormulaModels, K: int,
              value: float | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One strategy's simulation: ``value`` the static exposure at every time point (None: the
    natural course). Returns the mean risk by each time point and each simulated unit's final risk."""
    rng = np.random.default_rng(seed)
    n = len(base)
    cur: dict[str, np.ndarray] = {v: base[v].to_numpy(float) for v in spec.baseline}
    prev = {c: np.zeros(n) for c in spec.lagged}
    risk = np.zeros(n)
    surv = np.ones(n)
    means = np.zeros(K)
    for t in range(K):
        cur[TIME] = np.full(n, float(t))
        cur[TIME2] = np.full(n, float(t * t))
        for c in spec.lagged:
            cur[LAG + c] = prev[c]
        if t == 0:
            for c in spec.covariates:
                cur[c.name] = base[c.name].to_numpy(float)
            a = base[spec.exposure].to_numpy(float)
        else:
            for c in spec.covariates:
                X, _ = design(cur, c.terms, n)
                mu = models.covariates[c.name].predict(X)
                if c.kind == "binary":
                    cur[c.name] = rng.binomial(1, mu).astype(float)
                else:
                    lo, hi = models.ranges[c.name]
                    cur[c.name] = np.clip(rng.normal(mu, models.covariates[c.name].rmse), lo, hi)
            X, _ = design(cur, spec.exposure_terms, n)
            a = rng.binomial(1, models.exposure.predict(X)).astype(float)
            if spec.absorbing:
                a = np.where(cur[LAG + spec.exposure] == 1, 1.0, a)
        if value is not None:
            a = np.full(n, float(value))
        cur[spec.exposure] = a
        X, _ = design(cur, spec.outcome_terms, n)
        h = models.outcome.predict(X)
        risk = risk + h * surv
        surv = surv * (1 - h)
        means[t] = risk.mean()
        prev = {c: cur[c].copy() for c in spec.lagged}
    return means, risk


def observed_risk(d: pd.DataFrame, spec: GFormulaSpec) -> list[float]:
    """The nonparametric risk by each time point: the share of the rows at each time point with
    the event, accumulated as ``Σ_t h_t Π_{k<t}(1 − h_k)`` (``gfoRmula``'s ``obs_calculate``
    without censoring weights, so it assumes loss to follow-up is independent of the outcome)."""
    h = d.groupby(TIME)[spec.outcome].mean().sort_index().to_numpy(float)
    surv = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
    return [float(x) for x in np.cumsum(h * surv)]


def gformula(data: pd.DataFrame, spec: GFormulaSpec, *,
             strategies: Mapping[str, float] | None = None, n_sim: int | None = None,
             seed: int = 2026, natural: bool = True) -> GFormulaResult:
    """The parametric g-formula's risks under each static strategy (``{"never": 0, "always": 1}``
    by default) and the natural course."""
    strategies = dict(strategies if strategies is not None else {"never": 0.0, "always": 1.0})
    d = _prepared(data, spec)
    sched = check_schedule(d, spec.id, spec.time)
    if sched["late"] or sched["gaps"] or sched["repeats"]:
        raise NotEstimable("every unit needs one row at each time point from the first, with none "
                           "missing in between")
    K = sched["time_points"]
    models = fit_gformula(d, spec)
    base = d.loc[d[TIME] == 0].reset_index(drop=True)
    n_units = len(base)
    n_sim = int(n_sim or n_units)
    rng = np.random.default_rng(seed)
    if n_sim != n_units:
        base = base.iloc[rng.integers(0, n_units, n_sim)].reset_index(drop=True)
    sim_seed = int(rng.integers(0, 2 ** 31 - 1))
    plan: dict[str, float | None] = {**({"natural": None} if natural else {}), **strategies}
    risks, se, units = {}, {}, {}
    for name, value in plan.items():
        means, r = _simulate(base, spec, models, K, value, sim_seed)
        risks[name] = [float(x) for x in means]
        se[name] = float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else 0.0
        units[name] = r
    return GFormulaResult(K, n_sim, seed, risks, se, observed_risk(d, spec), models, units)


def gformula_bootstrap(data: pd.DataFrame, spec: GFormulaSpec, *, reps: int, n_sim: int | None,
                       seed: int, strategies: Mapping[str, float] | None = None,
                       contrast: tuple[str, str] = ("always", "never")) -> dict[str, Any]:
    """Percentile intervals over ``reps`` resamples of the units (each resampled unit a new unit),
    every model refit and the simulation rerun: for each strategy's final risk, and the risk
    difference and ratio of ``contrast``. Resamples whose models cannot be fit are counted."""
    rng = np.random.default_rng(seed)
    groups = list(data.groupby(spec.id, sort=False).indices.values())
    sizes = np.array([len(g) for g in groups])
    draws: dict[str, list[float]] = {}
    failed = 0
    for _ in range(int(reps)):
        pick = rng.integers(0, len(groups), len(groups))
        sample = data.iloc[np.concatenate([groups[k] for k in pick])].reset_index(drop=True)
        sample[spec.id] = np.repeat(np.arange(len(pick)), sizes[pick])
        try:
            res = gformula(sample, spec, strategies=strategies, n_sim=n_sim,
                           seed=int(rng.integers(0, 2 ** 31 - 1)), natural=False)
        except NotEstimable:
            failed += 1
            continue
        for name, values in res.risks.items():
            draws.setdefault(name, []).append(values[-1])
        a, b = res.final(contrast[0]), res.final(contrast[1])
        draws.setdefault("difference", []).append(a - b)
        draws.setdefault("ratio", []).append(a / b if b > 0 else math.nan)
    out = {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
           for k, v in draws.items()}
    return {"intervals": out, "reps": int(reps), "failed": failed, "draws": draws}


# ── sensitivity to unmeasured confounding (MODELING_SEQUENCE §0 ruling 10) ─────

# VanderWeele & Ding (2017, Ann Intern Med 167:268–274), computed by the one implementation every
# inference result uses (ESTIMAND's ``models.effects.e_values``, checked against R ``EValue``): the
# E-value of a risk ratio and of the limit of its interval nearer 1 (1 when the interval holds 1); an
# odds ratio read as a risk ratio when the outcome is rare, below 15%, and as √OR otherwise; a
# difference in means d (in outcome SDs) as RR = exp(0.91 d), its interval exp(0.91 d ± 1.78 se).
# The lane's artifact keeps its own shape (``rr``, ``lo``, ``hi``, ``point``, ``ci``).
RARE = RARE_OUTCOME


def _lane(found: dict[str, Any]) -> dict[str, float]:
    return {"rr": float(found["rr"]), "lo": float(found["rr_low"]), "hi": float(found["rr_high"]),
            "point": float(found["point"]), "ci": float(found["limit"])}


def e_value(rr: float, lo: float, hi: float) -> dict[str, float]:
    """The E-value of a risk ratio and of its 95% interval."""
    return _lane(e_values(rr, lo, hi, measure="RR"))


def e_value_or(odds: float, lo: float, hi: float, rare: bool) -> dict[str, float]:
    return _lane(e_values(odds, lo, hi, measure="OR", rare=rare))


def e_value_md(d: float, se: float) -> dict[str, float]:
    """``d`` and ``se`` on the outcome's standard-deviation scale."""
    return _lane(e_values(d, measure="OLS", sd=1.0, se=se))


__all__ = [
    "Covariate", "GFormulaResult", "RARE", "e_value", "e_value_md", "e_value_or", "GFormulaSpec", "GLMFit", "LAG", "MSMFit", "NEAR",
    "NotEstimable", "SUMMARY_QUANTILES", "TIME", "TIME2", "TRUNCATIONS", "WeightModel", "by_time",
    "censoring_weights", "check_schedule", "design", "fit_glm", "fit_gformula", "fit_msm", "gformula",
    "gformula_bootstrap", "ipw_weights", "observed_risk", "positivity", "sort_long", "summarize",
    "time_index", "truncate", "with_history",
]
