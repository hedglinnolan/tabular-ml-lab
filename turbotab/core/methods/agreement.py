"""Agreement between two measurements by the Bland–Altman method (D5; amendment of 2026-10-07).

Two measurements of one quantity on the same rows: two measurement methods (paired columns, under
Describe), or two models' out-of-fold predictions (under Predict). Correlation answers whether they
are related, not whether they agree; Bland & Altman (1986, *Lancet* i:307) describe agreement by the
differences d = A − B against the pair's mean (A + B)/2.

**Differences** (Bland & Altman 1986, "Measuring agreement" and "Precision of estimated limits of
agreement"; 1999, *Stat Methods Med Res* 8:135). The bias is the mean difference d̄, its interval
d̄ ± t(n − 1)·s/√n; the 95% limits of agreement are d̄ ± 1.96 s, each with the interval of the 1986
paper, ± t(n − 1)·√(3 s²/n) ("the standard error of d̄ − 2s and d̄ + 2s is about √(3s²/n)"), as R's
BlandAltmanLeh and blandr compute it.

**Ratios** (Bland & Altman 1986, "Relation between difference and mean"). When the spread of the
differences grows with the magnitude, the same computation on the natural logs, back-transformed:
the bias becomes a ratio A/B and the limits say between which ratios 95% of the pairs lie (the
paper's 0.80 to 1.27 "times the long axis VCF"). Every value must be positive.

**Regression-based limits** (Bland & Altman 1999, equations 3.1 to 3.3). D = b₀ + b₁·M fitted by least
squares, then the absolute residuals |R| = c₀ + c₁·M; the limits at a mean M are
b₀ + b₁·M ± 1.96·√(π/2)·(c₀ + c₁·M), since E|R| = σ·√(2/π) for a normal residual (the paper's ± 2.46).
The paper gives them no interval, and none is computed.

**Two checks, with every form.** Proportional bias: the slope of the difference on the mean, with its
interval and p-value (the first regression of the 1999 paper's regression approach). A spread that
grows: the slope of the absolute residuals on the mean (its second regression); a positive slope at p < 0.05 is noticed, and with every
value positive the ratio form is offered first. Their t has the residual degrees of freedom R's
``svyglm`` gives a two-coefficient fit: the design's degrees of freedom less one under a design or
repeated pairs (people − 2 for the latter), n − 2 otherwise.

(Section numbers are not cited. The 1986 paper numbers none, and its sections are named above from
its text. The 1999 paper's text could not be read; its regression approach is equations 3.1 to 3.3
as MedCalc's manual cites them, and the "§5.2" once cited for the ratios is removed: the ratios are
cited to the 1986 section that shows them.)

**Repeated pairs per person** (Bland & Altman 2007, *J Biopharm Stat* 17:571, "true value varies").
With k people and m_i pairs each (n = Σ m_i), a one-way analysis of variance of the differences on
the person gives MS_b (k − 1 df) and MS_w (n − k df); with λ = (n² − Σ m_i²)/((k − 1)·n), the
between-person variance is σ²_b = (MS_b − MS_w)/λ (zero when negative), the within-person variance
σ²_w = MS_w, and the variance of a single difference σ²_d = σ²_b + σ²_w. The bias is the mean of all
n differences, its variance (σ²_b·Σ m_i² + σ²_w·n)/n² on t(k − 1); the limits are d̄ ± 1.96·σ_d.
Their intervals are Zou's MOVER (2013, *Stat Methods Med Res* 22:630): σ²_d = MS_b/λ + (1 − 1/λ)·MS_w
is a sum of two independent mean squares, each with its chi-square interval, combined by MOVER and
then combined again with the bias's interval. With equal m_i this is Zou's computation exactly
(MS_b/m is the variance of the person means, λ = m the harmonic mean). When MS_b < MS_w the
between-person variance is taken as zero, and the bias's standard error is then Zou's: the variance
of the person means over k, not MS_w/n. The slope checks take each
person's pairs as one cluster (a sandwich over people, t(k − 1)), as R's ``svyglm`` with ``ids = ~
person`` does.

**Survey weights** (the population answer). The bias and the limits are design-based: μ̂ = Σ w·d / Σ w,
σ̂² = n/(n − 1)·Σ w·(d − μ̂)²/Σ w (R survey's ``svyvar``, the repo's one rule for a weighted SD), the
limits μ̂ ± 1.96·σ̂, and their standard errors by linearization over the design (strata, PSUs, the
lonely-PSU rule of :mod:`turbotab.core.models.survey`), intervals on t(PSUs − strata). The slope
checks are the weighted regressions with the design's sandwich (R's ``svyglm``); regression-based
limits are the weighted lines. Repeated pairs per person under the population answer are **refused**:
the variance-components form has no design-based version in common use, and weighting it as if each
pair were a person would understate the limits' width. The exits: one pair per person, or the
sample-only answer.

**Under Predict** the two models' predictions are their out-of-fold predictions only (a row no fold
scored is left out): every number here is computed from predictions each made by a model that never
saw the row, and nothing fitted here is applied to any row.

The goals it serves: Describe (two measurement methods) and Predict (two models). Estimate an effect
does not offer it: agreement describes two measurements and is not an effect. Until Describe is the
engine's third purpose (D1), its labels for Describe are :data:`DESCRIBE_LABELS`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Z = 1.96  # the 95% limits' multiplier, as Bland & Altman write it
SPREAD_FACTOR = math.sqrt(math.pi / 2)  # E|R| = σ·√(2/π); Bland & Altman 1999's 2.46 is 1.96·this
CONF = 0.95
FORMS: tuple[str, ...] = ("differences", "ratios", "regression")
COMPARISONS: tuple[str, ...] = ("methods", "models")
GOALS: tuple[str, ...] = ("describe", "estimate", "predict")
MIN_PAIRS = 3  # a slope's t needs n − 2 ≥ 1
# Describe is not yet one of the engine's purposes (D1 adds it to contracts.PURPOSES): the relations
# that apply only under Describe are keyed to it now, so they fire under neither offered purpose.
DESCRIBE = "describe"

SOURCES = {
    "ba1986": "Bland & Altman 1986, Lancet i:307 (doi:10.1016/S0140-6736(86)90837-8)",
    "ba1999": "Bland & Altman 1999, Stat Methods Med Res 8:135 (doi:10.1177/096228029900800204)",
    "ba2007": "Bland & Altman 2007, J Biopharm Stat 17:571 (doi:10.1080/10543400701329422)",
    "zou2013": "Zou 2013, Stat Methods Med Res 22:630 (doi:10.1177/0962280211402548)",
    "lumley": "Lumley 2010, Complex Surveys: A Guide to Analysis Using R (Wiley), ch. 2 and 5",
}

DIFFERENCES_EXIT = {"label": "Use the differences instead", "form": "differences"}
RATIOS_EXIT = {"label": "Use the ratios instead", "form": "ratios"}
ONE_PAIR_EXIT = {"label": "Keep one pair per person (the first)", "one_pair_per_unit": True}
DESCRIBE_EXIT = {"label": "Ask it under Describe", "goal": "describe"}


class AgreementRefused(ValueError):
    """The data or the goal cannot support the agreement asked for; the message says why, ``exits``
    the ways forward."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


# ── which goal offers which comparison ──────────────────────────────────────


@dataclass(frozen=True)
class Eligibility:
    offered: bool
    says: str
    exits: tuple[dict[str, Any], ...] = ()


def eligible(goal: str, comparison: str) -> Eligibility:
    """Whether ``goal`` (describe · estimate · predict) offers agreement between two ``comparison``
    (methods · models), in plain words."""
    if goal not in GOALS or comparison not in COMPARISONS:
        raise ValueError(f"goal {goal!r} or comparison {comparison!r} is unknown")
    if goal == "estimate":
        return Eligibility(False, "Agreement describes two measurements; it is not an effect, so "
                                  "Estimate an effect does not offer it.", (dict(DESCRIBE_EXIT),))
    if goal == "describe" and comparison == "models":
        return Eligibility(False, "Describe fits no models, so there are no predictions to compare.")
    if goal == "predict" and comparison == "methods":
        return Eligibility(False, "Comparing two measurement methods is a description, not a "
                                  "prediction.", (dict(DESCRIBE_EXIT),))
    if comparison == "methods":
        return Eligibility(True, "How closely the two measurement methods agree, pair by pair.")
    return Eligibility(True, "How closely the two models' out-of-fold predictions agree, row by row.")


# ── results ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float | None
    upper: float | None
    se: float | None = None


@dataclass(frozen=True)
class Slope:
    """A slope on the pair's mean: of the difference (proportional bias) or of the absolute
    residual (a spread that grows)."""

    intercept: float
    slope: float
    se: float
    lower: float
    upper: float
    p: float
    df: int


@dataclass(frozen=True)
class Agreement:
    form: str  # differences · ratios · regression
    scale: str  # difference · ratio
    first: str
    second: str
    n_pairs: int
    n_units: int  # people (the repeated form) or pairs
    repeated: bool
    weighted: bool
    bias: Interval  # on the reported scale: a difference, or a ratio first/second
    lower_limit: Interval
    upper_limit: Interval
    sd: float  # SD of a single difference (on the log scale for ratios)
    proportional: Slope  # the difference (log ratio) on the mean (of logs)
    spread: Slope  # the absolute residual on the mean
    spread_grows: bool
    lines: dict[str, list[float]] | None = None  # regression-based limits at the min, mean and max
    components: dict[str, float] | None = None  # σ²_b, σ²_w (repeated pairs)
    df: int | None = None  # the intervals' degrees of freedom
    estimator: str = ""
    concerns: list[str] = field(default_factory=list)
    noticing: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def says(self) -> str:
        """The result in plain words."""
        return _says(self)


# ── the computation ──────────────────────────────────────────────────────────


def agreement(first: Any, second: Any, *, form: str = "differences", units: Any = None,
              design: Any = None, row_ids: Any = None,
              names: tuple[str, str] = ("the first", "the second"),
              one_pair_per_unit: bool = False) -> Agreement:
    """Bland–Altman agreement of two paired measurements ``first`` and ``second`` (the module
    docstring has every formula). ``units`` names each pair's person: when a person has more than
    one pair the repeated form is used. ``design`` (a :class:`~turbotab.core.models.survey.
    SurveyDesign`, the population answer) makes it design-based; ``row_ids`` are the pairs' row ids
    in the design (the inputs' index when they are Series). A pair with either value missing leaves;
    ``one_pair_per_unit`` keeps each person's first complete pair (an exit of the refusals).
    Raises :class:`AgreementRefused` with its exits."""
    if form not in FORMS:
        raise ValueError(f"form {form!r} is not one of {FORMS}")
    first, second, units = _paired(first, second, units)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("the two measurements must be paired, one value each per row")
    ids = _row_ids(first, row_ids, len(a)) if design is not None else np.arange(len(a))
    keep = np.isfinite(a) & np.isfinite(b)
    unit = None if units is None else np.asarray(pd.Series(units).to_numpy())
    if unit is not None:
        if len(unit) != len(a):
            raise ValueError("units must name every pair's person")
        keep &= pd.notna(unit)
    if one_pair_per_unit and unit is not None:
        keep &= ~(pd.Series(unit).where(keep).duplicated(keep="first").to_numpy() & keep)
    a, b, ids = a[keep], b[keep], ids[keep]
    unit = None if unit is None else unit[keep]
    if form == "ratios" and (np.any(a <= 0) or np.any(b <= 0)):
        raise AgreementRefused(
            "Ratios need every value above zero; some are zero or below, so their logs do not "
            "exist.", (DIFFERENCES_EXIT, {"label": "Use the regression-based limits instead",
                                          "form": "regression"}))
    if len(a) < MIN_PAIRS:
        raise AgreementRefused(f"Agreement needs at least {MIN_PAIRS} complete pairs; "
                               f"{len(a)} {'is' if len(a) == 1 else 'are'} here.")
    x, y = (np.log(a), np.log(b)) if form == "ratios" else (a, b)
    d, m = x - y, (x + y) / 2
    if np.allclose(d, d[0], rtol=0, atol=1e-12 * max(1.0, float(np.max(np.abs(m))))):
        raise AgreementRefused("The two measurements differ by the same amount on every pair, so "
                               "there is no spread to describe: the bias is the whole story.")
    codes, k = (None, len(d)) if unit is None else _codes(unit)
    repeated = codes is not None and k < len(d)
    if repeated and design is not None:
        raise AgreementRefused(
            "People here have more than one pair, and the repeated-pairs form (Bland & Altman "
            "2007) has no survey-weighted version in common use; weighting each pair as if it were "
            "a person would make the limits too narrow.",
            (ONE_PAIR_EXIT, {"label": "Estimate for these participants instead: record the "
                                      "sample-only attestation",
                             "decision": {"kind": "set_survey", "estimand": "sample"}}))
    if repeated and form == "regression":
        raise AgreementRefused(
            "Regression-based limits (Bland & Altman 1999) assume one pair per person; people here "
            "have more than one.", (DIFFERENCES_EXIT, RATIOS_EXIT, ONE_PAIR_EXIT))
    if repeated and k < 2:
        raise AgreementRefused("Every pair belongs to one person: agreement between methods needs "
                               "at least two people.", (ONE_PAIR_EXIT,))
    if design is not None:
        out = _design_based(d, m, ids, design)
    elif repeated:
        out = _repeated(d, m, codes, k)
    else:
        out = _simple(d, m)
    bias, lo, hi, sd, prop, spread, lines, comps, df, estimator, concerns, used = out
    if form == "ratios":
        bias, lo, hi = _exp(bias), _exp(lo), _exp(hi)
    grows = bool(spread.slope > 0 and spread.p < 0.05)
    noticing = None
    if grows and form == "differences":
        positive = bool(np.all(a > 0) and np.all(b > 0))
        noticing = ("The differences spread wider as the values grow" + (
            ", so a single pair of limits misdescribes both ends: the ratios form fits a spread "
            "that grows in proportion (Bland & Altman 1986)." if positive else
            ", so a single pair of limits misdescribes both ends: the regression-based limits "
            "follow it (Bland & Altman 1999)."))
    return Agreement(form=form, scale="ratio" if form == "ratios" else "difference",
                     first=names[0], second=names[1], n_pairs=int(used),
                     n_units=int(k if repeated else used),
                     repeated=bool(repeated), weighted=design is not None, bias=bias,
                     lower_limit=lo, upper_limit=hi, sd=float(sd), proportional=prop,
                     spread=spread, spread_grows=grows,
                     lines=lines if form == "regression" else None, components=comps, df=df,
                     estimator=estimator, concerns=concerns, noticing=noticing)


def model_agreement(oof: Any, first: str, second: str, *, form: str = "differences") -> Agreement:
    """Agreement of two models' out-of-fold predictions (:class:`~turbotab.core.models.selection.
    OutOfFold`): a number for a numeric outcome, the event's predicted probability for a yes/no one.
    Only rows a fold scored count; when ``oof.units`` repeat, the repeated-pairs form is used."""
    task = getattr(oof, "task", None)
    if task not in ("regression", "binary"):
        raise AgreementRefused(
            "Agreement compares two numbers per row; "
            + ("for this outcome each model predicts a risk that changes over time (a "
               "time-to-event outcome)" if task == "time_to_event" else
               "this outcome has more than two categories, so each model gives a chance for every "
               "category (a multiclass outcome)")
            + ", not one number.", ({"label": "Compare the models by their scores instead",
                                     "stage": "models"},))
    for key in (first, second):
        if key not in oof.predictions:
            raise ValueError(f"no out-of-fold predictions for {key!r}")
    pa, pb = (np.asarray(oof.predictions[k], dtype=float) for k in (first, second))
    if task == "binary":
        pa, pb = pa[:, -1], pb[:, -1]
    scored = np.asarray(oof.scored, dtype=bool)
    pa, pb = np.where(scored, pa, np.nan), np.where(scored, pb, np.nan)
    return agreement(pa, pb, form=form, units=getattr(oof, "units", None), names=(first, second))


def _paired(first: Any, second: Any, units: Any) -> tuple[Any, Any, Any]:
    """Two Series (and a Series of units) are paired by their index, never by position: a
    reordered copy of a column pairs each row with itself. Indexes that do not hold the same rows
    are refused."""
    if not isinstance(first, pd.Series):
        return first, second, units
    rows = first.index

    def lined_up(other: Any, what: str) -> Any:
        if not isinstance(other, pd.Series) or other.index.equals(rows):
            return other
        if (rows.has_duplicates or other.index.has_duplicates or len(other) != len(rows)
                or not other.index.isin(rows).all()):
            raise ValueError(
                f"The {what} and the first measurement are labeled by different rows, so they "
                "cannot be paired row by row. Take both from one table, or pass plain lists in "
                "the same order.")
        return other.reindex(rows)

    return first, lined_up(second, "second measurement"), lined_up(units, "people")


def _row_ids(values: Any, row_ids: Any, n: int) -> np.ndarray:
    """The pairs' row ids in the survey design (asked only when there is one)."""
    if row_ids is not None:
        return np.asarray(row_ids, dtype=np.int64)
    if isinstance(values, pd.Series):
        if not pd.api.types.is_integer_dtype(values.index):
            raise ValueError("Under a survey design each pair needs its row number in the design, "
                             "and these rows are labeled by name: pass row_ids.")
        return np.asarray(values.index, dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _codes(unit: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(unit), sort=True)
    return codes.astype(np.int64), int(len(uniques))


def _exp(i: Interval) -> Interval:
    return Interval(float(np.exp(i.estimate)), None if i.lower is None else float(np.exp(i.lower)),
                    None if i.upper is None else float(np.exp(i.upper)), None)


def _t(df: float) -> float:
    return float(stats.t.ppf(1 - (1 - CONF) / 2, df))


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(β, residuals, (XᵀX)⁻¹) of y on [1, x]."""
    X = np.column_stack([np.ones_like(x), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    return beta, y - X @ beta, bread


def _slope(beta: np.ndarray, se: float, df: int) -> Slope:
    if df < 1:  # no degrees of freedom left for the slope's t: reported without a test
        return Slope(float(beta[0]), float(beta[1]), float(se), math.nan, math.nan, math.nan, 0)
    t = _t(df)
    stat = beta[1] / se if se > 0 else math.inf
    return Slope(float(beta[0]), float(beta[1]), float(se), float(beta[1] - t * se),
                 float(beta[1] + t * se), float(2 * stats.t.sf(abs(stat), df)), int(df))


def _lines(m: np.ndarray, line: np.ndarray, spread: np.ndarray) -> dict[str, list[float]]:
    at = np.array([m.min(), m.mean(), m.max()])
    center = line[0] + line[1] * at
    half = Z * SPREAD_FACTOR * (spread[0] + spread[1] * at)
    return {"at": at.tolist(), "bias": center.tolist(), "lower": (center - half).tolist(),
            "upper": (center + half).tolist()}


def _simple(d: np.ndarray, m: np.ndarray) -> tuple:
    """One pair per person, unweighted (Bland & Altman 1986; 1999)."""
    n = len(d)
    mean, s = float(d.mean()), float(d.std(ddof=1))
    t = _t(n - 1)
    se_bias, se_limit = s / math.sqrt(n), math.sqrt(3 * s * s / n)
    bias = Interval(mean, mean - t * se_bias, mean + t * se_bias, se_bias)
    limits = [Interval(c, c - t * se_limit, c + t * se_limit, se_limit)
              for c in (mean - Z * s, mean + Z * s)]
    beta, resid, bread = _ols(m, d)
    sigma2 = float(resid @ resid) / (n - 2)
    prop = _slope(beta, math.sqrt(sigma2 * bread[1, 1]), n - 2)
    gamma, r2, _ = _ols(m, np.abs(resid))
    spread = _slope(gamma, math.sqrt(float(r2 @ r2) / (n - 2) * bread[1, 1]), n - 2)
    return (bias, limits[0], limits[1], s, prop, spread, _lines(m, beta, gamma), None, n - 1,
            "Bland–Altman limits of agreement", [], n)


def _cluster_slope(x: np.ndarray, y: np.ndarray, codes: np.ndarray, k: int
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """(β, residuals, SE of the slope) of y on [1, x], a sandwich over the clusters with k/(k − 1)."""
    beta, resid, bread = _ols(x, y)
    X = np.column_stack([np.ones_like(x), x])
    totals = np.zeros((k, 2))
    np.add.at(totals, codes, X * resid[:, None])
    totals -= totals.mean(axis=0)
    meat = k / (k - 1) * totals.T @ totals
    V = bread @ meat @ bread
    return beta, resid, float(math.sqrt(max(V[1, 1], 0.0)))


def _repeated(d: np.ndarray, m: np.ndarray, codes: np.ndarray, k: int) -> tuple:
    """Repeated pairs per person (Bland & Altman 2007; Zou 2013's MOVER)."""
    n = len(d)
    counts = np.bincount(codes, minlength=k).astype(float)
    means = np.bincount(codes, weights=d, minlength=k) / counts
    grand = float(d.mean())
    ms_b = float(np.sum(counts * (means - grand) ** 2)) / (k - 1)
    ms_w = float(np.sum((d - means[codes]) ** 2)) / (n - k)
    lam = (n * n - float(np.sum(counts ** 2))) / ((k - 1) * n)
    var_b = max((ms_b - ms_w) / lam, 0.0)
    var_d = var_b + ms_w
    sd = math.sqrt(var_d)
    if ms_b >= ms_w:
        se_bias = math.sqrt((var_b * float(np.sum(counts ** 2)) + ms_w * n) / (n * n))
    else:  # σ²_b truncated to zero: Zou 2013's standard error, from the person means' variance
        se_bias = math.sqrt(float(np.var(means, ddof=1)) / k)
    t = _t(k - 1)
    bias = Interval(grand, grand - t * se_bias, grand + t * se_bias, se_bias)
    alpha = 1 - CONF
    terms = [(1 / lam, ms_b, k - 1), (1 - 1 / lam, ms_w, n - k)] if ms_b >= ms_w else \
        [(1.0, ms_w, n - k)]
    down = math.sqrt(sum((c * ms * (1 - nu / stats.chi2.ppf(1 - alpha / 2, nu))) ** 2
                         for c, ms, nu in terms))
    up = math.sqrt(sum((c * ms * (nu / stats.chi2.ppf(alpha / 2, nu) - 1)) ** 2
                       for c, ms, nu in terms))
    sd_lo, sd_hi = math.sqrt(max(var_d - down, 0.0)), math.sqrt(var_d + up)
    below, above = grand - bias.lower, bias.upper - grand
    lo_c, hi_c = grand - Z * sd, grand + Z * sd
    lower = Interval(lo_c, lo_c - math.hypot(below, Z * (sd_hi - sd)),
                     lo_c + math.hypot(above, Z * (sd - sd_lo)))
    upper = Interval(hi_c, hi_c - math.hypot(below, Z * (sd - sd_lo)),
                     hi_c + math.hypot(above, Z * (sd_hi - sd)))
    beta, resid, se = _cluster_slope(m, d, codes, k)
    prop = _slope(beta, se, k - 2)  # svyglm's residual df: k − 1 clusters, less one for the slope
    gamma, _, se2 = _cluster_slope(m, np.abs(resid), codes, k)
    spread = _slope(gamma, se2, k - 2)
    concerns = []
    if ms_b < ms_w:
        concerns.append("The person-to-person part of the variance came out below zero and is "
                        "taken as zero: the pairs vary as much within a person as between people. "
                        "The bias's interval then comes from how much the people's average "
                        "differences vary (Zou 2013).")
    if k - 2 < 1:
        concerns.append(_NO_SLOPE_TEST)
    return (bias, lower, upper, sd, prop, spread, None, {"between": var_b, "within": ms_w},
            k - 1, "Bland–Altman limits of agreement for repeated pairs per person "
                   "(variance components; MOVER intervals)", concerns, n)


def _design_based(d: np.ndarray, m: np.ndarray, ids: np.ndarray, design: Any) -> tuple:
    """The population answer: design-based bias, limits and slopes (module docstring)."""
    from turbotab.core.models.survey import (FEW_DESIGN_DF, SAMPLE_EXIT, domain_of,
                                             total_variance)

    domain = domain_of(ids, design)
    d, m, w = d[domain.keep], m[domain.keep], domain.weight
    n = domain.n
    if n < MIN_PAIRS:
        raise AgreementRefused(f"Fewer than {MIN_PAIRS} pairs have a survey weight above zero.")
    W = float(w.sum())
    mu = float(w @ d) / W
    v0 = float(w @ (d - mu) ** 2) / W
    c = n / (n - 1)
    sd = math.sqrt(c * v0)
    u_mu = w * (d - mu) / W
    u_v = c * w * ((d - mu) ** 2 - v0) / W
    U = np.column_stack([u_mu, u_mu - Z * u_v / (2 * sd), u_mu + Z * u_v / (2 * sd)])
    beta, resid, bread = _wls(m, d, w)
    X = np.column_stack([np.ones_like(m), m])
    gamma, _, bread2 = _wls(m, np.abs(resid), w)
    r2 = np.abs(resid) - X @ gamma
    U = np.column_stack([U, (X * (w * resid)[:, None]) @ bread[:, 1],
                         (X * (w * r2)[:, None]) @ bread2[:, 1]])
    full = np.zeros((design.n_rows, U.shape[1]))
    full[domain.at] = U
    var = total_variance(full, design, domain.mask(design))
    if var.df < 1:
        groups = f"{var.domain_psu} sampled cluster{'s' if var.domain_psu != 1 else ''}"
        layers = f"{var.domain_strata} survey layer{'s' if var.domain_strata != 1 else ''}"
        raise AgreementRefused(
            f"The pairs come from {groups} in {layers}, one cluster per layer, so nothing is left "
            "to show how much the estimate would vary from one survey to the next: no interval "
            f"can be given. (Survey terms: {var.domain_psu} PSU"
            f"{'s' if var.domain_psu != 1 else ''}, {var.domain_strata} "
            f"strat{'a' if var.domain_strata != 1 else 'um'}, no design degrees of freedom.)",
            ({"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}},))
    se = np.sqrt(np.clip(np.diag(var.meat), 0, None))
    t = _t(var.df)
    centers = (mu, mu - Z * sd, mu + Z * sd)
    bias, lower, upper = (Interval(cc, cc - t * s, cc + t * s, float(s))
                          for cc, s in zip(centers, se[:3]))
    concerns = []
    if var.df < FEW_DESIGN_DF:
        concerns.append(f"The pairs come from few sampled clusters, so the intervals are wide "
                        f"(design degrees of freedom: {var.df}).")
    if var.lonely:
        concerns.append("One survey layer has a single sampled cluster; its spread is measured "
                        "around the overall mean instead (a stratum with one PSU; R survey's "
                        "lonely.psu \"adjust\").")
    if var.df - 1 < 1:
        concerns.append(_NO_SLOPE_TEST)
    left = domain.left.get("unplaced", 0) + domain.left.get("unweighted", 0)
    if left:
        concerns.append(f"{left} pair{'s' if left != 1 else ''} without a place in the design or a "
                        f"weight above zero {'are' if left != 1 else 'is'} left out.")
    # svyglm's residual df for a two-coefficient fit: the design's df less one
    return (bias, lower, upper, sd, _slope(beta, float(se[3]), var.df - 1),
            _slope(gamma, float(se[4]), var.df - 1), _lines(m, beta, gamma), None, var.df,
            "Survey-weighted Bland–Altman limits of agreement (linearization over the design)",
            concerns, n)


_NO_SLOPE_TEST = ("Too few people or sampled clusters are left to test whether the difference or its "
                  "spread changes with the size: the slopes are shown without an interval.")


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = bread @ X.T @ (w * y)
    return beta, y - X @ beta, bread


# ── words ────────────────────────────────────────────────────────────────────


def _num(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def _says(r: Agreement) -> str:
    who = f"{r.first} − {r.second}" if r.scale == "difference" else f"{r.first} / {r.second}"
    if r.form == "regression" and r.lines:
        at = r.lines
        return (f"The difference ({who}) changes with the size of the values: at a mean of "
                f"{_num(at['at'][0])} it is {_num(at['bias'][0])} (95% of pairs between "
                f"{_num(at['lower'][0])} and {_num(at['upper'][0])}); at {_num(at['at'][2])} it is "
                f"{_num(at['bias'][2])} (between {_num(at['lower'][2])} and {_num(at['upper'][2])}).")
    unit = "times" if r.scale == "ratio" else ""
    text = (f"On average {r.first} reads {_num(r.bias.estimate)} {unit} {r.second}".replace("  ", " ")
            if r.scale == "ratio" else
            f"On average {r.first} reads {_num(abs(r.bias.estimate))} "
            f"{'above' if r.bias.estimate >= 0 else 'below'} {r.second}")
    return (f"{text} (the bias; 95% CI {_num(r.bias.lower)} to {_num(r.bias.upper)}). For 95% of "
            f"{'people' if r.repeated else 'pairs'}, {who} lies between {_num(r.lower_limit.estimate)}"
            f" and {_num(r.upper_limit.estimate)} (the limits of agreement).")


def agreement_sentence(result: Agreement | None = None, *, comparison: str = "methods") -> str:
    """The methods sentence of a Bland–Altman agreement analysis, naming its form, its intervals,
    its checks and any weighting or repeated pairs."""
    r = result
    what = ("the two models' out-of-fold predictions" if comparison == "models" else
            f"{r.first} and {r.second}" if r else "the two measurement methods")
    form = r.form if r else "differences"
    sentence = (f"Agreement between {what} was described by the Bland–Altman method (Bland & "
                "Altman 1986, 1999)")
    if form == "ratios":
        sentence += (" on the log scale, back-transformed to ratios, because the differences "
                     "spread wider as the values grew")
    if form == "regression":
        sentence += (", with regression-based limits of agreement: the difference and its absolute "
                     "residual were regressed on the pair's mean (Bland & Altman 1999).")
    else:
        sentence += (": the mean difference (bias) and the 95% limits of agreement (bias ± 1.96 SD "
                     "of the differences), each with a 95% confidence interval.")
    if r and r.repeated:
        sentence += (f" Each of the {r.n_units} people contributed several pairs, so the SD of a "
                     "difference combined the between- and within-person variance components "
                     "(Bland & Altman 2007), with MOVER confidence intervals for the limits "
                     "(Zou 2013).")
    if r and r.weighted:
        sentence += (" Estimates were weighted to the surveyed population, with standard errors by "
                     "linearization over the strata and primary sampling units.")
    sentence += (" Proportional bias was checked by regressing the difference on the pair's mean.")
    return sentence


# ── the contract ─────────────────────────────────────────────────────────────

# The labels for Describe, which is not yet one of the engine's purposes (D1 adds it and carries
# these into the contract): option -> (sound, rung).
DESCRIBE_LABELS: dict[str, tuple[str, str]] = {
    "differences": ("Sound when the spread of the differences is about the same at every size; the "
                    "limits assume the differences are roughly normal", "recommended"),
    "ratios": ("Sound when the spread grows in proportion to the values, all above zero; offered "
               "first when the spread is seen to grow", "available"),
    "regression": ("Sound when the bias or the spread changes with the size but not in proportion; "
                   "no interval for its lines", "available"),
}


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "bland_altman" in CONTRACTS:
        return
    here = "turbotab.core.methods.agreement"
    offered = (DESCRIBE, "prediction")  # the goals that offer it (inference refuses it)
    not_estimate = ("Not offered under Estimate an effect: agreement describes two measurements and "
                    "is not an effect (offered under Describe; see DESCRIBE_LABELS)")

    def option(key: str, label: str, customary: str, predict: str, rung: str, order: int
               ) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"prediction": predict, "inference": not_estimate},
                              {"prediction": rung, "inference": "not_offered"},
                              {"prediction": order, "inference": order})

    register_contract(MethodContract(
        key="bland_altman", label="Agreement between two measurements (Bland–Altman)",
        slot="evaluation", scope="descriptive", package="AGREEMENT", run_order=8.0,
        scope_note=("Nothing is fitted per fold. It reads every analyzed pair (under Describe, every "
                    "eligible row, as an estimate does) and never the outcome model; under Predict "
                    "it reads the two models' out-of-fold predictions only. It reports, and nothing "
                    "it computes is applied to a row or informs a modeling choice."),
        needs=("two numeric measurements of the same quantity on the same rows (two columns), or "
               "two fitted models' out-of-fold predictions of a numeric or yes/no outcome",
               "at least three complete pairs",
               "optional: the person each pair belongs to (repeated pairs), the survey design"),
        question="How closely do the two measurements agree?",
        options=(
            option("differences", "Differences: bias and 95% limits of agreement",
                   "The standard method-comparison analysis (Bland & Altman 1986, 1999); "
                   "BlandAltmanLeh and blandr in R",
                   "Sound for two models' out-of-fold predictions of a number; describes how far "
                   "apart they are, not which is right", "available", 0),
            option("ratios", "Ratios: the analysis on the log scale, back-transformed",
                   "Bland & Altman 1986 for a spread that grows with the size",
                   "Sound when both models' predictions are above zero and their gap grows with the "
                   "size; refused with a value at or below zero", "available", 1),
            option("regression", "Regression-based limits of agreement",
                   "Bland & Altman 1999; less common in published comparisons",
                   "Sound when the gap changes with the size; one pair per row only", "available", 2),
        ),
        storyboard=("pair each row's two values", "take the difference (or the log ratio) and the "
                    "pair's mean", "the bias: the mean difference, with its interval",
                    "the limits: bias ± 1.96 SD, with their intervals",
                    "check the difference and its spread against the mean"),
        relations=(
            Relation("conflicts", "ratios_need_positive_values",
                     "ratios are refused when a value is zero or below", when=("ratios",),
                     rung="refused", exits=("the differences", "the regression-based limits"),
                     condition="a value at or below zero", enforced_by=f"{here}:agreement", purposes=offered,
                     id="ratios_need_positive"),
            Relation("implies", "spread_noticed",
                     "a spread that grows with the size is noticed, and the ratios form (all values "
                     "above zero) or the regression-based limits are offered first",
                     when=("differences",), condition="the absolute residual's slope on the mean "
                     "above zero at p < 0.05", enforced_by=f"{here}:agreement", id="spread_grows",
                     purposes=offered),
            Relation("implies", "proportional_bias_reported",
                     "the slope of the difference on the pair's mean is reported with its interval",
                     condition="always", enforced_by=f"{here}:agreement", id="proportional_bias",
                     purposes=offered),
            Relation("implies", "repeated_pairs_components",
                     "the SD of a difference combines the between- and within-person variance "
                     "(Bland & Altman 2007), with MOVER intervals (Zou 2013), and the slopes are "
                     "clustered by person", when=("differences", "ratios"),
                     condition="a person with more than one pair", enforced_by=f"{here}:agreement",
                     id="repeated_pairs", purposes=offered),
            Relation("conflicts", "regression_limits_need_one_pair",
                     "regression-based limits are refused with repeated pairs per person",
                     when=("regression",), rung="refused",
                     exits=("the differences", "the ratios", "one pair per person"),
                     condition="a person with more than one pair", enforced_by=f"{here}:agreement",
                     id="regression_one_pair", purposes=offered),
            Relation("implies", "population_design",
                     "the bias, the limits and the slopes are weighted, with design-based standard "
                     "errors on t(PSUs − strata)", purposes=(DESCRIBE,),
                     condition="a survey design answered as the surveyed population (Describe)",
                     enforced_by=f"{here}:agreement", id="design_based"),
            Relation("conflicts", "repeated_pairs_under_design",
                     "repeated pairs per person under the population answer are refused: the "
                     "variance-components form has no design-based version in common use",
                     purposes=(DESCRIBE,), rung="refused",
                     exits=("one pair per person", "the sample-only attestation"),
                     condition="the surveyed population and a person with more than one pair",
                     enforced_by=f"{here}:agreement", id="repeats_not_weighted"),
            Relation("conflicts", "no_design_df",
                     "no design degrees of freedom are left for an interval", purposes=(DESCRIBE,),
                     rung="refused", exits=("the sample-only attestation",),
                     condition="the pairs lie in as many PSUs as strata",
                     enforced_by=f"{here}:agreement", id="design_df"),
            Relation("implies", "out_of_fold_only",
                     "two models are compared on their out-of-fold predictions only; a row no fold "
                     "scored is left out", purposes=("prediction",),
                     condition="two fitted models", enforced_by=f"{here}:model_agreement",
                     id="out_of_fold"),
            Relation("conflicts", "one_number_per_row",
                     "refused for a multiclass or time-to-event outcome: their predictions are not "
                     "one number per row", purposes=("prediction",), rung="refused",
                     exits=("compare the models by their scores",),
                     condition="a multiclass or time-to-event outcome",
                     enforced_by=f"{here}:model_agreement", id="one_number"),
            Relation("conflicts", "not_an_effect",
                     "not offered under Estimate an effect: agreement is a description",
                     purposes=("inference",), rung="refused", exits=("ask it under Describe",),
                     condition="the Estimate an effect goal", enforced_by=f"{here}:eligible",
                     id="estimate_refused"),
            Relation("conflicts", "methods_are_described",
                     "two measurement methods are compared under Describe, not Predict",
                     purposes=("prediction",), rung="refused", exits=("ask it under Describe",),
                     condition="two measurement columns under Predict",
                     enforced_by=f"{here}:eligible", id="methods_under_predict"),
        ),
        sources=tuple(SOURCES.values()),
        sentence=f"{here}:agreement_sentence"))


_register_contract()

__all__ = ["AgreementRefused", "Agreement", "DESCRIBE", "DESCRIBE_LABELS", "Eligibility", "FORMS", "Interval",
           "Slope", "agreement", "agreement_sentence", "eligible", "model_agreement"]
