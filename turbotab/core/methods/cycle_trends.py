"""Design-based trends across stacked NHANES cycles (D6; engine only until D1 asks it under Describe).

A measure's mean, or the prevalence of a condition, in each cycle of a stack
(:mod:`turbotab.core.methods.cycles`), and whether it changed across them. NCHS's guidelines for
the analysis of trends (Ingram et al. 2018, Vital Health Stat 2(179)) set the approach: use every
cycle, not the first and last (Guideline 2); fit the trend to the record-level data with the survey
design, so the weights, the year-to-year correlation and the design's degrees of freedom are all
honored (Guideline 5); assess nonlinearity first, beginning with the highest-order term, and only
then read the linear trend (Issue 7).

**Time.** Each cycle is placed at its midpoint in years (Guideline 4: intervals of unequal length,
as the 3.2-year 2017–March 2020 file is, are placed at their midpoints; 1999–2000 is 2000). A slope
is per year.

**The cycle estimates.** In cycle c, μ̂_c = Σ w·y ÷ Σ w over its analyzed rows, with its standard
error by linearization over the design (u_i = w_i (y_i − μ̂_c) ÷ Σ w; the PSU totals' spread within
strata, :func:`turbotab.core.models.survey.total_variance`), as R's ``svyby(…, svymean)`` gives it,
and its degrees of freedom the PSUs less the strata that hold the cycle's rows (Analytic Guidelines
§3.2.3.2). A mean's interval is μ̂_c ± t(df_c)·SE. A prevalence's interval is Korn and Graubard's
(1998, Survey Methodology 24:193), as R survey's ``svyciprop(method = "beta")`` computes it: the
effective sample size n* = p̂(1 − p̂) ÷ SE² × (t(n − 1) ÷ t(df))² and the Clopper–Pearson limits
Beta(α/2; n*p̂, n*(1 − p̂) + 1) and Beta(1 − α/2; n*p̂ + 1, n*(1 − p̂)), n the cycle's rows. A
prevalence of 0 or 1 has no design-based spread and is given no interval. The cycles' joint
covariance is the linearization of all their means together over the whole design.

**Orthogonal polynomial contrasts** (Issue 7, as SUDAAN's DESCRIPT POLY does): the contrasts of the
cycle estimates with R's ``contr.poly(T, scores = times)`` coefficients (orthonormal polynomials in
the cycles' times, so unequal spacing is handled), each with its standard error from the joint
covariance and a t test on the design's degrees of freedom; up to the cubic, and no further than
T − 1. On the linear scale only (Issue 8), and without covariates (Issue 9).

**Polynomial regression** (Issue 7). The design-based regression of the record-level measure on
time (least squares for a mean or, on the linear scale, a prevalence; logistic regression for a
prevalence on the log-odds scale), R's ``svyglm``. Backward elimination as the guidelines describe:
fit up to the cubic (no further than T − 1), test the highest power; if it is significant the trend
is nonlinear, else drop it and test the next; the degree-one model's slope is the linear trend.
Covariates may be added (Issue 9). A linear trend in a prevalence is refused when any fitted value
falls outside 0 to 1 (Guideline 8b), with the log-odds scale as its exit.

**Joinpoint, at joins named beforehand** (Issues 11–12 and Appendix IV, Parameterization A). E(y) =
β₀ + β₁·t + δ₁·(t − x₁)₊ + …, each joinpoint xₖ at an observed cycle strictly inside the range: the
first segment's slope β₁, each change δₖ, and each later segment's slope β₁ + δ₁ + … + δₖ, with
design-based standard errors. The guidelines locate joinpoints with NCI's Joinpoint software on the
aggregated estimates and then fit the model to the record-level data with survey software; the
search itself is not done here (it could not be checked against a reference, and choosing the join
from the same data and then testing it there overstates the evidence), so a join must be named in
advance: the exits are a join named from an outside event, or the polynomial tests.

**Degrees of freedom.** Every test is a t test on the design's degrees of freedom, the PSUs less the
strata that hold the analyzed rows (Guideline 5's "number of PSUs minus the number of sampling
strata"; SUDAAN's and Stata's convention), not ``svyglm``'s residual degrees of freedom; fewer than
8 is noticed (NCHS presentation standards). With no survey design (the "these participants"
answer), every row has weight one and is its own sampling unit.

The goal it serves is Describe (a trend across cycles describes the population over time; it is
not an effect, and nothing is predicted). Until Describe is the engine's third purpose (D1), its
labels for Describe are :data:`DESCRIBE_LABELS`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

KINDS: tuple[str, ...] = ("mean", "prevalence")
METHODS: tuple[str, ...] = ("contrasts", "regression", "joinpoint")
SCALES: tuple[str, ...] = ("linear", "logit")
ALPHA = 0.05
CONF = 0.95
MAX_DEGREE = 3
ORDERS = ("linear", "quadratic", "cubic")
DESCRIBE = "describe"

SOURCES = {
    "ingram2018": "Ingram et al. 2018, Vital Health Stat 2(179), Issues 4–12 and Appendix IV",
    "kg1998": "Korn & Graubard 1998, Survey Methodology 24:193",
    "guidelines": "NHANES Analytic Guidelines 2011–2016, §3.2.3",
    "lumley": "Lumley 2010, Complex Surveys: A Guide to Analysis Using R",
    "r_survey": "R survey (svyby, svyciprop, svycontrast, svyglm)",
}

REGRESSION_EXIT = {"label": "Test the trend by polynomial regression instead", "method": "regression"}
LINEAR_EXIT = {"label": "Use the linear scale", "scale": "linear"}
LOGIT_EXIT = {"label": "Use the log-odds scale (logistic regression)", "scale": "logit"}
SAMPLE_EXIT = {"label": "Estimate for these participants instead: record the sample-only "
                        "attestation", "decision": {"kind": "set_survey", "estimand": "sample"}}


class TrendRefused(ValueError):
    """The data or the request cannot support the trend asked for; the message says why in plain
    words, ``exits`` the ways forward."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


@dataclass(frozen=True)
class Eligibility:
    offered: bool
    says: str
    exits: tuple[dict[str, Any], ...] = ()


GOALS: tuple[str, ...] = ("describe", "estimate", "predict")
DESCRIBE_EXIT = {"label": "Ask it under Describe", "goal": "describe"}


def eligible(goal: str) -> Eligibility:
    """Whether ``goal`` (describe · estimate · predict) offers a trend across cycles, in plain
    words."""
    if goal not in GOALS:
        raise ValueError(f"goal {goal!r} is unknown")
    if goal == "describe":
        return Eligibility(True, "Whether the measure changed across the survey cycles.")
    return Eligibility(False, "A trend across survey cycles describes how the population changed "
                              "over time; it is not " + ("an effect" if goal == "estimate" else
                                                         "a prediction") + ".",
                       (dict(DESCRIBE_EXIT),))


# ── results ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleEstimate:
    cycle: str
    time: float
    n: int  # analyzed rows in the cycle
    estimate: float
    se: float
    lower: float | None
    upper: float | None
    df: int
    interval: str  # "t" or "korn_graubard"


@dataclass(frozen=True)
class Term:
    name: str
    estimate: float
    se: float
    t: float
    p: float
    df: int
    lower: float
    upper: float


@dataclass
class Trend:
    kind: str
    method: str
    scale: str
    name: str
    estimates: list[CycleEstimate]
    covariance: list[list[float]]
    df: int
    terms: list[Term]  # the tests: contrasts; each degree's highest power; segments and changes
    models: dict[int, list[Term]] = field(default_factory=dict)  # regression: degree -> its terms
    shape: str = ""  # nonlinear · increasing · decreasing · stable
    degree: int = 1  # the regression degree kept, or the highest contrast tested
    joins: tuple[float, ...] = ()
    weighted: bool = True
    n: int = 0
    covariates: tuple[str, ...] = ()
    estimator: str = ""
    concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def says(self) -> str:
        return _says(self)


# ── the computation ──────────────────────────────────────────────────────────


def cycle_trend(values: Any, cycles: Any, *, method: str = "contrasts", kind: str = "mean",
                scale: str = "linear", design: Any = None, row_ids: Any = None,
                times: Mapping[Any, float] | None = None, degree: int | None = None,
                joins: Sequence[Any] | str = (), covariates: pd.DataFrame | None = None,
                name: str = "the measure") -> Trend:
    """The trend of ``values`` across ``cycles`` (one cycle label per row; the stacked table's
    ``cycle`` column), by ``method`` (contrasts · regression · joinpoint; the module docstring has
    every formula). ``kind`` is mean or prevalence (values 0 or 1); ``scale`` linear or logit (a
    prevalence's regression or joinpoint). ``design`` is the survey design over every stacked row
    (:meth:`~turbotab.core.methods.cycles.Stacked.design`); ``row_ids`` the rows' ids in it (the
    Series' index by default). A row with a missing value is outside the analysis but keeps its
    place in the design. ``times`` maps a cycle to its time (default: its midpoint in years).
    ``degree`` caps the polynomial (default cubic); ``joins`` are the joinpoints, as cycles or
    times. Raises :class:`TrendRefused` with its exits."""
    if method not in METHODS:
        raise ValueError(f"method {method!r} is not one of {METHODS}")
    if kind not in KINDS or scale not in SCALES:
        raise ValueError(f"kind {kind!r} or scale {scale!r} is unknown")
    if scale == "logit" and kind == "mean":
        raise TrendRefused("The log-odds scale is for a prevalence (a yes-or-no measure); a mean is "
                           "tested on its own scale.", (LINEAR_EXIT,))
    if scale == "logit" and method == "contrasts":
        raise TrendRefused("Comparing the cycles' prevalences side by side works on their own "
                           "scale, not on the log-odds scale (Ingram et al. 2018, Issue 8). "
                           "(Orthogonal polynomial contrasts.)",
                           (LINEAR_EXIT, {**REGRESSION_EXIT, "scale": "logit"}))
    if covariates is not None and method == "contrasts":
        raise TrendRefused("Comparing the cycles' estimates side by side cannot take other "
                           "variables into account (Ingram et al. 2018, Issue 9); a regression on "
                           "time can. (Orthogonal polynomial contrasts.)",
                           (REGRESSION_EXIT, {"label": "Leave the adjustment out",
                                              "covariates": None}))
    if isinstance(joins, str):
        raise TrendRefused(
            "Where the trend changes is not searched for here: choosing the joinpoint from these "
            "estimates and then testing it on them overstates the evidence, and the search NCHS "
            "uses (NCI's Joinpoint software) could not be checked against a reference. Name the "
            "cycle where a change is expected, from an event outside these data. (A joinpoint "
            "search.)",
            ({"label": "Name the joinpoint cycle", "joins": None}, REGRESSION_EXIT))
    y, labels, ids, cov = _aligned(values, cycles, row_ids, covariates, design)
    keep = np.isfinite(y) & pd.notna(labels)
    if cov is not None:
        keep &= np.all(np.isfinite(cov), axis=1)
    if kind == "prevalence":
        bad = keep & ~np.isin(y, (0.0, 1.0))
        if bad.any():
            raise TrendRefused(
                f"A prevalence counts who has the condition, so each value must be 0 or 1; "
                f"{int(bad.sum()):,} values here are neither.",
                ({"label": "Say which value means the condition", "event": None},
                 {"label": "Describe the mean instead", "kind": "mean"}))
    if design is None:
        design = _equal_design(ids[keep])
        weighted = False
    else:
        weighted = True
    from turbotab.core.models.survey import domain_of

    domain = domain_of(ids[keep], design)
    rows = np.flatnonzero(keep)[domain.keep]
    y, labels, w, at = y[rows], np.asarray(labels, dtype=object)[rows], domain.weight, domain.at
    x_cov = None if cov is None else cov[rows]
    levels, time = _times(labels, times)
    T = len(levels)
    if T < 2:
        raise TrendRefused(f"A trend needs at least two cycles with analyzed rows; "
                           f"{T} {'has' if T == 1 else 'have'} them.",
                           ({"label": "Stack more cycles", "stage": "ingest"},))
    code = pd.Index(levels).get_indexer(labels)
    t = np.asarray([time[c] for c in code])
    estimates, V, var = _cycle_estimates(y, w, code, levels, time, at, design, kind)
    if var.df < 1:
        raise TrendRefused(
            f"The analyzed rows lie in {var.domain_psu} sampled clusters in {var.domain_strata} "
            "survey layers, one cluster per layer, so nothing is left to show how much the estimates "
            f"would vary from one survey to the next. (Survey terms: {var.domain_psu} PSUs, "
            f"{var.domain_strata} strata, no design degrees of freedom.)", (SAMPLE_EXIT,))
    df = int(var.df)
    concerns = _concerns(design, var, domain, len(rows), estimates, weighted)
    cov_names = tuple(str(c) for c in covariates.columns) if covariates is not None else ()
    if method == "contrasts":
        terms, shape, deg = _contrasts(estimates, V, time, df, degree, T)
        models: dict[int, list[Term]] = {}
        joined: tuple[float, ...] = ()
        estimator = "Orthogonal polynomial contrasts of the cycle estimates"
    elif method == "regression":
        terms, models, shape, deg = _regression(y, t, x_cov, w, at, design, domain, df, degree, T,
                                                kind, scale, cov_names)
        joined = ()
        estimator = ("Survey-weighted logistic regression on time" if scale == "logit" else
                     "Survey-weighted least squares on time")
    else:
        joined = _joins(joins, levels, time)
        terms, shape = _joinpoint(y, t, x_cov, w, at, design, domain, df, joined, kind, scale,
                                  cov_names)
        models, deg = {}, 1
        estimator = ("Joinpoint regression at joins named beforehand, "
                     + ("logistic" if scale == "logit" else "least squares") + ", survey-weighted")
    if not weighted:
        estimator += " (these participants: equal weights, each row its own sampling unit)"
    return Trend(kind=kind, method=method, scale=scale, name=name, estimates=estimates,
                 covariance=V.tolist(), df=df, terms=terms, models=models, shape=shape,
                 degree=deg, joins=joined, weighted=weighted, n=len(rows), covariates=cov_names,
                 estimator=estimator, concerns=concerns)


def _aligned(values: Any, cycles: Any, row_ids: Any, covariates: Any, design: Any
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """The values, their cycles, their row ids and the covariates, paired by the Series' index."""
    if isinstance(values, pd.Series):
        index = values.index
        if isinstance(cycles, pd.Series) and not cycles.index.equals(index):
            if not cycles.index.isin(index).all() or cycles.index.has_duplicates:
                raise ValueError("the cycles and the values are labeled by different rows")
            cycles = cycles.reindex(index)
        if covariates is not None and not covariates.index.equals(index):
            covariates = covariates.reindex(index)
        ids = np.asarray(row_ids if row_ids is not None else index)
    else:
        ids = np.asarray(row_ids) if row_ids is not None else np.arange(len(values))
    y = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(dtype=float)
    labels = np.asarray(pd.Series(cycles).to_numpy(), dtype=object)
    if len(labels) != len(y) or len(ids) != len(y):
        raise ValueError("every value needs its cycle and its row id")
    if design is not None and not np.issubdtype(np.asarray(ids).dtype, np.integer):
        raise ValueError("Under a survey design each row needs its row number in the design; "
                         "pass row_ids.")
    cov = None
    if covariates is not None:
        frame = covariates.apply(pd.to_numeric, errors="coerce")
        if frame.shape[1] == 0:
            cov = None
        else:
            cov = frame.to_numpy(dtype=float)
            if cov.shape[0] != len(y):
                raise ValueError("the covariates must have one row per value")
    return y, labels, np.asarray(ids, dtype=np.int64), cov


def _equal_design(ids: np.ndarray) -> Any:
    from turbotab.core.models.survey import SurveyDesign

    n = len(ids)
    return SurveyDesign(row_ids=np.asarray(ids, dtype=np.int64), weight=np.ones(n),
                        stratum=np.zeros(n, dtype=np.int64), psu=np.arange(n, dtype=np.int64),
                        weight_column=None, strata_column=None, psu_column=None,
                        psu_note="no survey design: each row is its own sampling unit")


def _times(labels: np.ndarray, times: Mapping[Any, float] | None) -> tuple[list[Any], np.ndarray]:
    """The cycles present, in time order, and each one's time."""
    from turbotab.core.methods.cycles import StackRefused, cycle_of

    present = list(pd.unique(pd.Series(labels)))
    found: dict[Any, float] = {}
    unknown = []
    for c in present:
        if times is not None and c in times:
            found[c] = float(times[c])
            continue
        try:
            found[c] = cycle_of(c).midpoint
        except StackRefused:
            unknown.append(c)
    unknown += [c for c, v in found.items() if not math.isfinite(v)]
    if unknown:
        raise TrendRefused(
            f"No time is known for {', '.join(f'`{c}`' for c in unknown)}; a trend places each "
            "cycle in time (its midpoint in years for an NHANES cycle).",
            ({"label": "Give each cycle its time", "times": {str(c): None for c in unknown}},))
    order = sorted(present, key=lambda c: found[c])
    tt = np.array([found[c] for c in order])
    if len(np.unique(tt)) < len(tt):
        raise TrendRefused("Two cycles are placed at the same time, so they cannot be told apart "
                           "in a trend.", ({"label": "Give each cycle its own time",
                                            "times": None},))
    return order, tt


def _t(df: float) -> float:
    return float(stats.t.ppf(1 - (1 - CONF) / 2, df))


def _term(name: str, est: float, se: float, df: int) -> Term:
    t = est / se if se > 0 else math.copysign(math.inf, est) if est else 0.0
    p = float(2 * stats.t.sf(abs(t), df))
    half = _t(df) * se
    return Term(name, float(est), float(se), float(t), p, int(df), float(est - half),
                float(est + half))


def _cycle_estimates(y: np.ndarray, w: np.ndarray, code: np.ndarray, levels: list[Any],
                     time: np.ndarray, at: np.ndarray, design: Any, kind: str
                     ) -> tuple[list[CycleEstimate], np.ndarray, Any]:
    from turbotab.core.models.survey import total_variance

    T = len(levels)
    U = np.zeros((design.n_rows, T))
    means = np.zeros(T)
    out = []
    for c in range(T):
        m = code == c
        W = float(w[m].sum())
        mu = float(w[m] @ y[m]) / W
        means[c] = mu
        U[at[m], c] = w[m] * (y[m] - mu) / W
        mask = np.zeros(design.n_rows, dtype=bool)
        mask[at[m]] = True
        own = total_variance(U[:, [c]], design, mask)
        se = math.sqrt(max(float(own.meat[0, 0]), 0.0))
        n = int(m.sum())
        lo = hi = None
        if own.df >= 1:
            if kind == "prevalence":
                lo, hi = _korn_graubard(mu, se, n, own.df)
            else:
                lo, hi = mu - _t(own.df) * se, mu + _t(own.df) * se
        out.append(CycleEstimate(str(levels[c]), float(time[c]), n, mu, se, lo, hi, int(own.df),
                                 "korn_graubard" if kind == "prevalence" else "t"))
    whole = np.zeros(design.n_rows, dtype=bool)
    whole[at] = True
    var = total_variance(U, design, whole)
    return out, var.meat, var


def _korn_graubard(p: float, se: float, n: int, df: int) -> tuple[float | None, float | None]:
    """Korn & Graubard's (1998) interval, as R survey's ``svyciprop(method = "beta")``."""
    if not (0 < p < 1) or se <= 0 or n < 2:
        return None, None
    alpha = 1 - CONF
    n_eff = p * (1 - p) / (se * se) * (stats.t.ppf(alpha / 2, n - 1) / stats.t.ppf(alpha / 2, df)) ** 2
    return (float(stats.beta.ppf(alpha / 2, n_eff * p, n_eff * (1 - p) + 1)),
            float(stats.beta.ppf(1 - alpha / 2, n_eff * p + 1, n_eff * (1 - p))))


def poly_contrasts(times: Sequence[float], degree: int) -> np.ndarray:
    """R's ``contr.poly(T, scores = times)[, 1:degree]``: orthonormal polynomials in the centered
    times (the Gram–Schmidt basis of 1, t, t², …, each column of unit length, its sign that of the
    QR's diagonal, as R's ``make.poly`` takes it)."""
    s = np.asarray(times, dtype=float)
    X = np.vander(s - s.mean(), N=len(s), increasing=True)
    Q, R = np.linalg.qr(X)
    Z = Q * np.sign(np.diag(R))
    return Z[:, 1:degree + 1]


def _contrasts(estimates: list[CycleEstimate], V: np.ndarray, time: np.ndarray, df: int,
               degree: int | None, T: int) -> tuple[list[Term], str, int]:
    d = min(degree or MAX_DEGREE, MAX_DEGREE, T - 1)
    Z = poly_contrasts(time, d)
    mu = np.array([e.estimate for e in estimates])
    est = Z.T @ mu
    cov = Z.T @ V @ Z
    terms = [_term(ORDERS[j], est[j], math.sqrt(max(cov[j, j], 0.0)), df) for j in range(d)]
    return terms, _shape(terms), d


def _shape(highest_first: Sequence[Term]) -> str:
    """NCHS's reading (Ingram et al. 2018, Issue 7): from the highest order down, a significant
    nonlinear term means a nonlinear trend; else the linear term's sign, or stable."""
    ordered = sorted(highest_first, key=lambda r: ORDERS.index(r.name) if r.name in ORDERS else 0,
                     reverse=True)
    for r in ordered:
        if r.name != "linear" and r.p < ALPHA:
            return "nonlinear"
    lin = next((r for r in ordered if r.name == "linear"), None)
    if lin is None or lin.p >= ALPHA:
        return "stable"
    return "increasing" if lin.estimate > 0 else "decreasing"


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, at: np.ndarray, design: Any, domain: Any,
         scale: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(β, its design-based covariance, the fitted values) of a weighted fit over the design."""
    from turbotab.core.models.survey import (total_variance, weighted_least_squares,
                                             weighted_logistic)

    fit = weighted_logistic(X, y, w) if scale == "logit" else weighted_least_squares(X, y, w)
    if not fit.converged:
        raise TrendRefused("The fit on the log-odds scale did not settle on an answer, so no trend "
                           "is given on that scale. (The logistic fit did not converge.)",
                           (LINEAR_EXIT,))
    u = np.zeros((design.n_rows, X.shape[1]))
    u[at] = fit.scores
    var = total_variance(u, design, domain.mask(design))
    B = np.linalg.pinv(fit.information)
    Vb = B @ var.meat @ B
    eta = X @ fit.estimate
    fitted = 1 / (1 + np.exp(-eta)) if scale == "logit" else eta
    return fit.estimate, (Vb + Vb.T) / 2, fitted


def _check_unit_interval(fitted: np.ndarray, kind: str, scale: str) -> None:
    if kind == "prevalence" and scale == "linear" and (fitted.min() < 0 or fitted.max() > 1):
        raise TrendRefused(
            "The straight-line trend in this prevalence runs below 0 or above 1 within the cycles "
            "studied, which a prevalence cannot do (Ingram et al. 2018, Guideline 8b). (A linear "
            "probability model outside the unit interval.)",
            (LOGIT_EXIT,))


def _regression(y: np.ndarray, t: np.ndarray, cov: np.ndarray | None, w: np.ndarray,
                at: np.ndarray, design: Any, domain: Any, df: int, degree: int | None, T: int,
                kind: str, scale: str, cov_names: tuple[str, ...]
                ) -> tuple[list[Term], dict[int, list[Term]], str, int]:
    top = min(degree or MAX_DEGREE, MAX_DEGREE, T - 1)
    center = float(np.unique(t).mean())
    tc = t - center
    models: dict[int, list[Term]] = {}
    fitted: dict[int, np.ndarray] = {}
    tests: list[Term] = []
    kept = 1
    for d in range(top, 0, -1):  # backward elimination: the highest power first
        X = np.column_stack([np.ones_like(tc), *[tc ** k for k in range(1, d + 1)],
                             *([] if cov is None else [cov])])
        beta, Vb, fitted[d] = _fit(X, y, w, at, design, domain, scale)
        names = ["intercept", *ORDERS[:d], *cov_names]
        models[d] = [_term(nm, beta[j], math.sqrt(max(Vb[j, j], 0.0)), df)
                     for j, nm in enumerate(names)]
        tests.append(models[d][d])
        if d > 1 and models[d][d].p < ALPHA and kept == 1:
            kept = d
    _check_unit_interval(fitted[kept], kind, scale)
    shape = "nonlinear" if kept > 1 else _shape([models[1][1]])
    return tests, models, shape, kept


def _joins(joins: Sequence[Any], levels: list[Any], time: np.ndarray) -> tuple[float, ...]:
    if not joins:
        raise TrendRefused("A trend that bends needs the cycle where it may bend, named in "
                           "advance. (A joinpoint.)",
                           ({"label": "Name the joinpoint cycle", "joins": None}, REGRESSION_EXIT))
    at_cycle = {str(c): float(tt) for c, tt in zip(levels, time)}
    out = []
    for j in joins:
        if str(j) in at_cycle:
            x = at_cycle[str(j)]
        else:
            try:
                x = float(j)
            except (TypeError, ValueError):
                x = math.nan
        if not any(abs(x - tt) < 1e-9 for tt in time):
            raise TrendRefused(
                f"`{j}` is not one of the cycles studied, and the trend may bend only at a cycle, "
                "not between two (Ingram et al. 2018, Guideline 11). (The joinpoint.)",
                ({"label": "Name one of the cycles", "joins": None},))
        if x <= time[0] + 1e-9 or x >= time[-1] - 1e-9:
            raise TrendRefused(
                f"`{j}` is the first or the last cycle, so one side of the bend would have a "
                "single cycle and no slope. (The joinpoint.)", ({"label": "Name an inner cycle",
                                                       "joins": None},))
        out.append(x)
    if len(set(out)) < len(out):
        raise TrendRefused("A cycle is named twice as a bend. (A joinpoint.)", ({"label": "Name each once",
                                                            "joins": None},))
    return tuple(sorted(out))


def _joinpoint(y: np.ndarray, t: np.ndarray, cov: np.ndarray | None, w: np.ndarray,
               at: np.ndarray, design: Any, domain: Any, df: int, joins: tuple[float, ...],
               kind: str, scale: str, cov_names: tuple[str, ...]) -> tuple[list[Term], str]:
    hinges = [np.clip(t - x, 0, None) for x in joins]
    # Time centered at the first joinpoint: the slopes and changes are the same, and the fit stays
    # well conditioned with calendar years.
    X = np.column_stack([np.ones_like(t), t - joins[0], *hinges,
                         *([] if cov is None else [cov])])
    beta, Vb, fitted = _fit(X, y, w, at, design, domain, scale)
    _check_unit_interval(fitted, kind, scale)
    K = len(joins)
    terms = [_term("slope 1", beta[1], math.sqrt(max(Vb[1, 1], 0.0)), df)]
    for k in range(K):
        j = 2 + k
        terms.append(_term(f"change at {_num(joins[k])}", beta[j], math.sqrt(max(Vb[j, j], 0.0)),
                           df))
        c = np.zeros(len(beta))
        c[1:j + 1] = 1.0
        terms.append(_term(f"slope {k + 2}", float(c @ beta), math.sqrt(max(float(c @ Vb @ c), 0)),
                           df))
    changed = any(r.name.startswith("change") and r.p < ALPHA for r in terms)
    if changed:
        shape = "nonlinear"
    else:
        lin = terms[0]
        shape = "stable" if lin.p >= ALPHA else ("increasing" if lin.estimate > 0 else "decreasing")
    return terms, shape


def _concerns(design: Any, var: Any, domain: Any, n: int, estimates: list[CycleEstimate],
              weighted: bool) -> list[str]:
    from turbotab.core.models.survey import FEW_DESIGN_DF, LONELY_RULE

    out = []
    left = domain.left.get("unplaced", 0) + domain.left.get("unweighted", 0)
    if left:
        out.append(f"{left:,} row{'s' if left != 1 else ''} without a place in the design or a "
                   f"weight above zero {'are' if left != 1 else 'is'} left out.")
    if weighted and var.lonely:
        out.append(f"{len(var.lonely)} survey layer{'s have' if len(var.lonely) != 1 else ' has'} a "
                   f"single sampled cluster; its total is {LONELY_RULE}.")
    if weighted and var.df < FEW_DESIGN_DF:
        out.append(f"Only {var.df} design degrees of freedom ({var.domain_psu} PSUs minus "
                   f"{var.domain_strata} strata): NCHS asks that estimates on fewer than "
                   f"{FEW_DESIGN_DF} be reviewed before they are published.")
    for e in estimates:
        if e.lower is None:
            out.append(f"{e.cycle}: " + ("everyone or no one analyzed has the condition, so the "
                                         "survey design gives this prevalence no spread and no "
                                         "interval." if e.se == 0 else
                                         "no design degrees of freedom are left for an interval."))
    return out


# ── words ────────────────────────────────────────────────────────────────────


def _num(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.4g}" if abs(v) < 10000 else f"{v:,.0f}"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _says(r: Trend) -> str:
    first, last = r.estimates[0], r.estimates[-1]
    what = "prevalence" if r.kind == "prevalence" else "mean"
    span = f"from {first.cycle} to {last.cycle}"
    fmt = (lambda v: f"{100 * v:.1f}%") if r.kind == "prevalence" else _num
    head = (f"The {what} of {r.name} was {fmt(first.estimate)} in {first.cycle} and "
            f"{fmt(last.estimate)} in {last.cycle}")
    if r.method == "joinpoint":
        slopes = [t for t in r.terms if t.name.startswith("slope")]
        parts = [f"{t.name}: {_num(t.estimate)} per year ({_p(t.p)})" for t in slopes]
        return head + f"; joinpoint at {', '.join(_num(j) for j in r.joins)}: " + "; ".join(parts) + "."
    if r.shape == "nonlinear":
        bent = next((t for order in reversed(ORDERS[1:]) for t in r.terms
                     if t.name == order and t.p < ALPHA), None)
        return head + (f"; the change {span} was not a straight line (the {bent.name} term, "
                       f"{_p(bent.p)})." if bent else ".")
    linear = next((t for t in (r.models.get(1, []) or r.terms) if t.name == "linear"), None)
    if linear is None:
        return head + "."
    if r.shape == "stable":
        return head + f"; no change {span} was detected (linear trend {_p(linear.p)})."
    word = "rose" if r.shape == "increasing" else "fell"
    if r.method == "regression":
        unit = " on the log-odds scale" if r.scale == "logit" else (
            " percentage points" if r.kind == "prevalence" else "")
        size = 100 * linear.estimate if (r.kind == "prevalence" and r.scale == "linear") else \
            linear.estimate
        return head + (f"; it {word} {span} by {_num(abs(size))}{unit} per year (linear trend, "
                       f"{_p(linear.p)}).")
    return head + f"; it {word} {span} (linear trend, {_p(linear.p)})."


def trend_sentence(result: Trend | None = None) -> str:
    """The methods sentence of a trend across cycles: the estimates, their intervals, the test and
    its degrees of freedom."""
    r = result
    what = ("prevalence" if r and r.kind == "prevalence" else "mean")
    sentence = (f"The {what} in each NHANES cycle was estimated with the survey weights, and its "
                "standard error by Taylor linearization over the strata and PSUs")
    if r and r.kind == "prevalence":
        sentence += ", with Korn–Graubard 95% confidence intervals (Korn & Graubard 1998)"
    sentence += ". "
    method = r.method if r else "contrasts"
    if method == "contrasts":
        sentence += ("The trend across cycles was tested with orthogonal polynomial contrasts of "
                     "the cycle estimates, each cycle placed at its midpoint year, beginning with "
                     "the highest order")
    elif method == "regression":
        sentence += ("The trend was tested by survey-weighted "
                     + ("logistic" if r and r.scale == "logit" else "linear")
                     + " regression of the record-level data on the cycle's midpoint year, "
                     "testing the cubic, then the quadratic, then the linear term")
        if r and r.covariates:
            sentence += f", adjusted for {', '.join(r.covariates)}"
    else:
        joins = ", ".join(_num(j) for j in r.joins) if r else "a cycle named beforehand"
        sentence += (f"A joinpoint regression with the joinpoint at {joins}, named before the "
                     "analysis, was fitted to the record-level data with the survey design")
    sentence += (" (NCHS Guidelines for Analysis of Trends, Ingram et al. 2018); tests used the "
                 "design's degrees of freedom (PSUs minus strata)"
                 + (f", here {r.df}" if r else "") + ".")
    if r and not r.weighted:
        sentence = sentence.replace("with the survey weights, and its standard error by Taylor "
                                    "linearization over the strata and PSUs",
                                    "for these participants, unweighted")
    return sentence


# ── the contract ─────────────────────────────────────────────────────────────

# The labels for Describe, which is not yet one of the engine's purposes (D1 adds it and carries
# these into the contract): option -> (sound, rung).
DESCRIBE_LABELS: dict[str, tuple[str, str]] = {
    "contrasts": ("Sound for a trend in a mean or a prevalence on its own scale with no adjustment; "
                  "one request tests the linear, quadratic and cubic terms", "recommended"),
    "regression": ("Sound always, with or without adjustment; the logistic form keeps a "
                   "prevalence's trend between 0 and 1", "available"),
    "joinpoint": ("Sound only with joinpoints named before looking at the data (an event outside "
                  "them); the location is not searched for", "available"),
}


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "cycle_trends" in CONTRACTS:
        return
    here = "turbotab.core.methods.cycle_trends"
    not_here = ("Not offered under {goal}: a trend across survey cycles describes the population "
                "over time (offered under Describe; see DESCRIBE_LABELS)")

    def option(key: str, label: str, customary: str, order: int) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"prediction": not_here.format(goal="Predict"),
                               "inference": not_here.format(goal="Estimate an effect")},
                              {"prediction": "not_offered", "inference": "not_offered"},
                              {"prediction": order, "inference": order})

    register_contract(MethodContract(
        key="cycle_trends", label="Trends across stacked survey cycles", slot="evaluation",
        scope="descriptive", package="D6", run_order=8.5,
        scope_note=("It reads every analyzed row of every cycle, as an estimate does, and the survey "
                    "design over the stacked table; nothing is fitted per fold, and nothing it "
                    "computes is applied to a row or informs a modeling choice."),
        needs=("a stack of two or more cycles (three or more to look for a bend), with each "
               "cycle's time", "a numeric measure (a mean) or a yes-or-no one (a prevalence)",
               "the survey design over the stacked rows, or the answer that the estimates describe "
               "these participants", "optional: covariates (regression and joinpoint only), and "
               "joinpoints named in advance"),
        question="Did the measure change across the cycles?",
        options=(
            option("contrasts", "Orthogonal polynomial contrasts of the cycle estimates",
                   "NCHS's usual test for NHANES trends (Ingram et al. 2018, Issue 7; SUDAAN "
                   "DESCRIPT POLY; R svycontrast with contr.poly)", 0),
            option("regression", "Polynomial regression on time, survey-weighted",
                   "Ingram et al. 2018, Issues 7–9; R svyglm with the cycle's year as a predictor",
                   1),
            option("joinpoint", "Joinpoint regression at joinpoints named beforehand",
                   "Ingram et al. 2018, Issues 11–12 and Appendix IV (the model fitted with survey "
                   "software; the location found by NCI's Joinpoint, not done here)", 2),
        ),
        storyboard=("place each cycle at its midpoint year", "estimate each cycle's mean or "
                    "prevalence with its interval", "test the highest-order term first",
                    "then the linear trend", "say whether it rose, fell, stayed or bent"),
        relations=(
            Relation("implies", "design_based_trend",
                     "the cycle estimates, their covariance and every test are design-based, on "
                     "the design's degrees of freedom (PSUs minus strata)", purposes=(DESCRIBE,),
                     condition="a survey design answered as the surveyed population",
                     enforced_by=f"{here}:cycle_trend", id="design_based"),
            Relation("implies", "korn_graubard_intervals",
                     "a prevalence's interval in each cycle is Korn and Graubard's",
                     purposes=(DESCRIBE,), condition="a yes-or-no measure",
                     enforced_by=f"{here}:cycle_trend", id="korn_graubard"),
            Relation("implies", "nonlinearity_first",
                     "the highest-order term is tested first; the linear trend is read only when no "
                     "bend is found", purposes=(DESCRIBE,), when=("contrasts", "regression"),
                     condition="three or more cycles", enforced_by=f"{here}:cycle_trend",
                     id="highest_first"),
            Relation("conflicts", "contrasts_take_no_covariates",
                     "contrasts of the cycle estimates cannot be adjusted for covariates",
                     purposes=(DESCRIBE,), when=("contrasts",), rung="refused",
                     exits=("polynomial regression", "leave the adjustment out"),
                     condition="covariates with the contrasts", enforced_by=f"{here}:cycle_trend",
                     id="contrasts_unadjusted"),
            Relation("conflicts", "contrasts_on_their_own_scale",
                     "contrasts on the log-odds scale are refused", purposes=(DESCRIBE,),
                     when=("contrasts",), rung="refused",
                     exits=("the linear scale", "logistic regression"),
                     condition="the logit scale with the contrasts",
                     enforced_by=f"{here}:cycle_trend", id="contrasts_linear"),
            Relation("conflicts", "prevalence_line_leaves_unit_interval",
                     "a straight-line trend in a prevalence that runs below 0 or above 1 is refused",
                     purposes=(DESCRIBE,), when=("regression", "joinpoint"), rung="refused",
                     exits=("the log-odds scale",),
                     condition="a fitted prevalence outside 0 to 1",
                     enforced_by=f"{here}:cycle_trend", id="unit_interval"),
            Relation("conflicts", "joinpoint_not_searched",
                     "the joinpoint's location is not searched for; it must be named in advance at "
                     "an inner cycle", purposes=(DESCRIBE,), when=("joinpoint",), rung="refused",
                     exits=("name the joinpoint cycle", "polynomial regression"),
                     condition="a joinpoint search, or a joinpoint outside the inner cycles",
                     enforced_by=f"{here}:cycle_trend", id="no_search"),
            Relation("conflicts", "no_design_df",
                     "no design degrees of freedom are left for a test", purposes=(DESCRIBE,),
                     rung="refused", exits=("the sample-only attestation",),
                     condition="the analyzed rows lie in as many PSUs as strata",
                     enforced_by=f"{here}:cycle_trend", id="design_df"),
            Relation("conflicts", "not_an_effect",
                     "not offered under Estimate an effect or Predict: a trend across cycles is a "
                     "description", purposes=("inference", "prediction"), rung="refused",
                     exits=("ask it under Describe",), condition="the Estimate or Predict goal",
                     enforced_by=f"{here}:eligible", id="describe_only"),
        ),
        sources=tuple(SOURCES.values()),
        sentence=f"{here}:trend_sentence"))


_register_contract()

__all__ = ["CycleEstimate", "DESCRIBE", "DESCRIBE_LABELS", "Eligibility", "KINDS", "METHODS", "SCALES", "Term",
           "Trend", "TrendRefused", "cycle_trend", "eligible", "poly_contrasts", "trend_sentence"]
