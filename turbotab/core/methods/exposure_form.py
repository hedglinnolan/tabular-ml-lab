"""The form an exposure takes in the model: a straight line, a restricted cubic spline, quintiles,
categories at declared cut points, one data-derived cut point, or, for a food with many
non-consumers, non-consumers as their own category beside a spline among consumers (AUDIT_REPORT
§5 WP12, ME-17: "Exposure–response is straight-line only"; MODELING_SEQUENCE §1 row 5, the package
FORM).

NUTRITION_PACK §07G: "**Restricted cubic spline** of outcome vs energy-adjusted intake: 3–5 knots
at conventional percentiles (**3 knots at 10/50/90**, or 4 at 5/35/65/95) … and a reported **p for
non-linearity** … **[CONVENTION, now near-default; quintiles remain expected alongside]**", and
§08: "**p for trend using the median of each quintile as a continuous score, not the quintile
number**". The form is a pipeline step (:class:`ExposureForms`) that runs after energy
adjustment, so a spline or a quintile is of the energy-adjusted intake, and it is fit inside the
pipeline: knots, cut points and quintile medians are learned on the rows each fit sees (every
training fold under cross-validation), never on a held-out row.

**The spline is Harrell's** (``rms::rcs``, which calls ``Hmisc::rcspline.eval`` with its default
``norm = 2``). Knots: :func:`rcs_knots` follows ``rcspline.eval``'s default placement line by line,
including its two adjustments: with fewer than 100 values the outer knots move to the 5th smallest
and 5th largest values, and a value at either extreme held by at least ``fractied`` (5%) of the
rows (zero alcohol, say), while no interior value is that common, becomes a knot of its own just
inside it. Basis (:func:`rcs_basis`): for knots t₁ < … < t_k and ``kd = (t_k − t₁)^(2/3)``,

    x_j = ((x − t_j)/kd)₊³ + [(t_{k−1} − t_j)((x − t_k)/kd)₊³ − (t_k − t_j)((x − t_{k−1})/kd)₊³] / (t_k − t_{k−1})

for j = 1…k − 2, beside x itself: k − 1 columns, linear beyond the outer knots (Harrell 2015,
*Regression Modeling Strategies*, 2nd ed., §2.4.5). The columns are named as ``rms`` prints them:
``x``, ``x'``, ``x''``.

**k by a declared rule** (MODELING_SEQUENCE §1 row 5; the review: "Under inference, fix k by a
declared rule (k=4 by default; 3 for small n; 5 for large n). AIC chooses k using the outcome, so
under inference it is a recorded data-driven choice"). The rule is Harrell's (RMS §2.4.6): "For
many datasets, k = 4 offers an adequate fit of the model and is a good compromise between
flexibility and loss of precision caused by overfitting a small sample. When the sample size is
large (e.g., n ≥ 100 with a continuous uncensored response variable), k = 5 is a good choice.
Small samples (< 30, say) may require the use of k = 3." The sample size it reads is the effective
one Harrell's §4.4 limits a model by (n for a numeric outcome; the smaller of the events and
non-events; the events of a time to event; n − Σnᵢ³/n² for an ordinal outcome)
(:func:`knots_by_rule`). A declaration without ``knots`` takes the rule's k, recorded with it.

**The tests** (:func:`exposure_tests`), under inference, use the coefficient table's own
covariance (HC3, CR2 or the model's information): for a spline, a Wald test that all of its
columns are zero, *the test of association*, and that its nonlinear columns are zero
(``rms::anova``'s "Nonlinear" row). A non-significant nonlinearity test never refits a straight
line (Grambsch & O'Brien 1991: the test of association's type I error then rises by about half);
a switch the user records once the estimates were seen is marked so (``plan_lock``). For quintiles,
the trend test of the pack: the exposure scored by the median of its quintile (medians of the
fitting rows), one column in place of the four indicators, its coefficient tested on the same
covariance, labeled "p for linear trend (customary)" (the review: "never 'dose–response'"). Beside
a declared exposure's spline the quintile table is produced by default, its boundaries and its
reference stated (MODELING_SEQUENCE §0 ruling 2).

**On the transformed scale.** A form belongs to the exposure's final scale (log, energy model,
scale scoring, omics normalization): the declaration records that scale
(:func:`transform_signature`), and a later transform makes it stale (:func:`current_forms`):
re-asked, never kept, with its knots, its cut points and the estimand's unit
(MODELING_SEQUENCE §2, "A domain transform of the exposure *invalidates* the functional-form
answer, the knots, the cut points and the estimand's unit"). A spline or categories on an energy
residual is the nutrient's curve on the energy-adjusted scale at mean energy, not the substitution
curve (its label says so, and spline(N) + E is offered as that route).

**The leash** (MODELING_SEQUENCE §4): under inference, quintiles rank lower and sit beside the
spline; data-derived "optimal" cut points (Altman & Royston 2006: "seriously biased") and a
continuous confounder cut into three or fewer groups (Brenner & Blettner 1997: "serious residual
confounding") are blocked and recorded. **Continuous confounders get a declared form too**, a
spline by default. **Mass at zero** (STROBE-nut nut-11, nut-14): non-consumers as their own
category beside a spline among consumers with knots at consumers' percentiles, or the
consumers-only domain recorded as an estimand change.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, TransformerMixin

Form = Literal["linear", "spline", "quintiles", "categories", "optimal", "zero_spline"]
FORMS: tuple[str, ...] = ("linear", "spline", "quintiles", "categories", "optimal", "zero_spline")
SPLINE_FORMS = ("spline", "zero_spline")
# Harrell's default percentiles (``rcspline.eval``: ``seq(outer, 1 − outer, length = k)`` with
# outer 0.10 for 3 knots, 0.05 for 4–6), which are the pack's "10/50/90" and "5/35/65/95".
KNOT_PERCENTILES: dict[int, tuple[float, ...]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}
KNOT_CHOICES = tuple(KNOT_PERCENTILES)
DEFAULT_KNOTS = 4  # Harrell (RMS §2.4.6): four knots are "a good compromise between flexibility and loss of precision"
FRACTIED = 0.05  # ``rcspline.eval``'s and ``rms::rcs``'s default
SMALL_SAMPLE = 100  # below this many values, the outer knots are the 5th smallest and largest
QUINTILES = 5
SCORE_SUFFIX = " (quintile median)"
# FORM: Harrell's rule for k (RMS §2.4.6), on the effective sample size (RMS §4.4).
HARRELL_SMALL = 30  # "Small samples (< 30, say) may require the use of k = 3"
HARRELL_LARGE = 100  # "When the sample size is large (e.g., n ≥ 100 …), k = 5 is a good choice"
HARRELL_RMS = "Harrell, Regression Modeling Strategies, 2nd ed., §2.4.6"
# A value at the bottom held by this share of the rows is a mass at zero: ``rcspline.eval``'s own
# ``fractied``, the share at which it stops treating the lowest value as an ordinary one.
MASS_AT_ZERO = FRACTIED
# A confounder with at least this many distinct values is continuous, and its form is declared
# (MODELING_SEQUENCE §1 row 5, "Continuous confounders get a declared form too"); with fewer it
# enters as recorded (a straight line, or indicators when declared codes).
CONTINUOUS = 10
COARSE_GROUPS = 3  # Brenner & Blettner 1997: residual confounding "if the number of categories is small"
OPTIMAL_RANGE = (0.10, 0.90)  # the candidate cut points of a minimum-p search (Altman et al. 1994)
CONSUMER_SUFFIX = "_consumer"
ABOVE_SUFFIX = "_above"
TREND_LABEL = "p for linear trend (customary)"
GRAMBSCH = "Grambsch & O'Brien 1991, Stat Med 10:697"
ALTMAN = "Altman & Royston 2006, BMJ 332:1080"
BRENNER = "Brenner & Blettner 1997, Epidemiology 8:429"
STROBE_NUT = "STROBE-nut (Lachat et al. 2016) nut-14"


# ── the restricted cubic spline ──────────────────────────────────────────────


def _quantile(values: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """R's ``quantile`` (type 7, its default), which is numpy's ``linear`` method."""
    return np.quantile(values, np.asarray(p, dtype=float), method="linear")


# ``Hmisc::rcspline.eval`` stops below this many values: the fewest a spline's knots can be placed on.
SPLINE_MIN_VALUES = 6


def rcs_knots(x: Any, n_knots: int = DEFAULT_KNOTS, *, fractied: float = FRACTIED
              ) -> tuple[np.ndarray, list[str]]:
    """Knot locations as ``Hmisc::rcspline.eval`` places them when only their number is given.

    Returns the knots and plain notes on any departure from the nominal percentiles. Raises
    ``ValueError`` where ``rcspline.eval`` stops: fewer than 6 values, or fewer than 3 distinct
    knots.
    """
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    n = len(xx)
    nk = int(n_knots)
    if n < SPLINE_MIN_VALUES:
        raise ValueError(f"A spline needs at least {SPLINE_MIN_VALUES} recorded values to place "
                         f"its knots; there are {n}.")
    if nk < 3:
        raise ValueError("A restricted cubic spline needs at least 3 knots.")
    notes: list[str] = []
    xu = np.unique(xx)
    nxu = len(xu)
    if nxu - 2 <= nk:
        knots = xu[1:-1]
        notes.append(f"only {nxu} distinct values, so the knots are the {nxu - 2} interior ones")
    else:
        outer = 0.05 if nk > 3 else 0.1
        if nk > 6:
            outer = 0.025
        nke = nk
        first: float | None = None
        last: float | None = None
        override_first = override_last = False
        if 0 < fractied < 1:
            _, counts = np.unique(xx, return_counts=True)
            f = counts / n
            interior = f[1:-1]
            if (interior.max() if interior.size else -np.inf) < fractied:
                if f[0] >= fractied:
                    first = float(xx[xx > xx.min()].min())
                    xx = xx[xx > first]
                    nke -= 1
                    override_first = True
                    notes.append(f"{f[0]:.0%} of the values are the lowest one, so a knot sits at "
                                 f"the next value up")
                if f[-1] >= fractied:
                    last = float(xx[xx < xx.max()].max())
                    xx = xx[xx < last]
                    nke -= 1
                    override_last = True
                    notes.append(f"{f[-1]:.0%} of the values are the highest one, so a knot sits "
                                 f"at the next value down")
        if nke == 1:
            knots = np.array([float(np.median(xx))])
        else:
            if nxu <= nke:
                knots = xu.astype(float)
            else:
                p = (np.linspace(0.5, 1.0 - outer, nke) if nke == 2
                     else np.linspace(outer, 1.0 - outer, nke))
                knots = _quantile(xx, p)
                if len(np.unique(knots)) < min(nke, 3):
                    knots = _quantile(xx, np.linspace(outer, 1.0 - outer, 2 * nke))
                    if first is not None and len(np.unique(knots)) < 3:
                        mid = (first + last) / 2.0 if last is not None else float(np.median(xx))
                        knots = np.sort(np.array([first, mid, last if last is not None
                                                  else float(_quantile(xx, [1.0 - outer])[0])]))
                    if len(np.unique(knots)) < 3:
                        raise ValueError("Fewer than 3 distinct knots can be placed: the values "
                                         "are too heavily tied.")
                    notes.append(f"the usual percentiles gave fewer than {nke} distinct knots, so "
                                 f"they were spread over {2 * nke} percentiles instead")
            knots = np.array(knots, dtype=float)
            if len(xx) < SMALL_SAMPLE:
                ordered = np.sort(xx)
                if not override_first:
                    knots[0] = ordered[4]
                if not override_last:
                    knots[nke - 1] = ordered[len(ordered) - 5]
                notes.append(f"with {len(xx)} values (fewer than {SMALL_SAMPLE}), the outer knots "
                             f"are the 5th smallest and 5th largest")
        knots = np.concatenate([[first] if first is not None else [], knots,
                                [last] if last is not None else []])
    knots = np.unique(np.asarray(knots, dtype=float))
    if len(knots) < 3:
        raise ValueError("Fewer than 3 distinct knots can be placed: the values are too heavily "
                         "tied for a spline.")
    return knots, notes


def rcs_basis(x: Any, knots: Sequence[float]) -> np.ndarray:
    """The k − 2 nonlinear columns of Harrell's restricted cubic spline (norm = 2), n × (k − 2).

    Missing values stay missing. Beyond the outer knots every column is linear in x.
    """
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3:
        raise ValueError("A restricted cubic spline needs at least 3 knots.")
    x = np.asarray(x, dtype=float)
    kd = (t[-1] - t[0]) ** (2.0 / 3.0)
    last, before = t[-1], t[-2]

    def cube(c: float) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.maximum((x - c) / kd, 0.0) ** 3

    tail_last, tail_before = cube(last), cube(before)
    out = np.empty((len(x), k - 2))
    for j in range(k - 2):
        out[:, j] = cube(t[j]) + ((before - t[j]) * tail_last
                                  - (last - t[j]) * tail_before) / (last - before)
    return out


def spline_names(column: str, n_knots: int) -> list[str]:
    """``x``, ``x'``, ``x''``…: the spline's k − 1 columns, as ``rms`` labels them."""
    return [column] + [f"{column}{chr(39) * (j + 1)}" for j in range(n_knots - 2)]


# ── k by a declared rule (Harrell, RMS §2.4.6) ───────────────────────────────


def effective_n(task: str | None, y: Any, event: Any = None) -> float | None:
    """The sample size a model of ``y`` is limited by (Harrell, RMS §4.4, Table 4.1): n for a
    numeric outcome; the smaller of the events and non-events for a yes/no outcome; the events of a
    time to event; n − Σnᵢ³/n² over an ordinal outcome's levels. None when it cannot be read."""
    from turbotab.core.stages.rows import _level_key

    values = pd.Series(np.asarray(y, dtype=object))
    values = values[values.notna()]
    n = len(values)
    if n == 0 or task is None:
        return None
    if task in ("regression", "multiclass"):
        return float(n)
    keys = values.map(_level_key)
    if task in ("binary", "time_to_event"):
        if event is not None:
            events = int((keys == _level_key(event)).sum())
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            events = int((numeric.fillna(0) != 0).sum()) if numeric.notna().all() else None
            if events is None:
                return None
        return float(events if task == "time_to_event" else min(events, n - events))
    if task == "ordinal":
        counts = keys.value_counts().to_numpy(dtype=float)
        return float(n - (counts ** 3).sum() / n ** 2)
    return float(n)


def knots_by_rule(n_effective: float | None) -> int:
    """Harrell's rule: 3 knots below an effective sample size of 30, 5 from 100, else 4."""
    if n_effective is None:
        return DEFAULT_KNOTS
    if n_effective < HARRELL_SMALL:
        return 3
    return 5 if n_effective >= HARRELL_LARGE else 4


EFFECTIVE_WORDS = {"regression": "the number of analyzed rows",
                   "multiclass": "the number of analyzed rows",
                   "binary": "the smaller of the events and the non-events",
                   "time_to_event": "the number of events",
                   "ordinal": "n − Σnᵢ³/n² over the outcome's levels"}


def rule_words(k: int, n_effective: float | None, task: str | None) -> str:
    """The rule stated: ``k = 5 by Harrell's rule (3 knots below an effective sample size of 30,
    5 from 100, else 4; here 1,200, the number of analyzed rows)``."""
    rule = (f"Harrell's rule (3 knots below an effective sample size of {HARRELL_SMALL}, 5 from "
            f"{HARRELL_LARGE}, else 4")
    if n_effective is None:
        # The rule was not applied (no effective sample size was read): its default, said so,
        # never "by the rule" with a result it did not compute (the FORM repair, item 2a).
        return f"k = {k}, the default of {rule}; no effective sample size was read)"
    size = f"{n_effective:,.0f}" if abs(n_effective - round(n_effective)) < 1e-9 else f"{n_effective:,.1f}"
    return f"k = {k} by {rule}; here {size}, {EFFECTIVE_WORDS.get(str(task), 'the analyzed rows')})"


# ── quintiles and categories ─────────────────────────────────────────────────


def quantile_cuts(x: Any, groups: int = QUINTILES) -> np.ndarray:
    """The groups − 1 inner cut points of ``x`` (R type 7 quantiles, as ``pandas.qcut`` takes)."""
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    return _quantile(values, np.arange(1, groups) / groups)


def quantile_group(x: Any, cuts: Sequence[float]) -> np.ndarray:
    """Each value's group, 0 for the lowest: right-closed intervals, the lowest group holding its
    cut point (``pandas.qcut``'s and R ``cut(…, include.lowest = TRUE)``'s rule). A value below
    the lowest or above the highest value seen in the fit falls in the end group; a missing value
    gets −1."""
    x = np.asarray(x, dtype=float)
    out = np.searchsorted(np.asarray(cuts, dtype=float), x, side="left").astype(np.int64)
    out[~np.isfinite(x)] = -1
    return out


def quintile_names(column: str) -> list[str]:
    """``x_Q2`` … ``x_Q5``: the indicators against the lowest quintile, ``x``'s Q1."""
    return [f"{column}_Q{g}" for g in range(2, QUINTILES + 1)]


def category_names(column: str, n_cuts: int) -> list[str]:
    """``x_C2`` … ``x_C{k+1}``: indicators of the groups above the lowest, at declared cut points."""
    return [f"{column}_C{g}" for g in range(2, n_cuts + 2)]


def boundaries_words(column: str, cuts: Sequence[float], what: str = "quintile") -> str:
    """``quintile 1: `x` ≤ 12.1 (the reference); quintile 2: 12.1 < `x` ≤ 15 …``: STROBE 16b's
    category boundaries, the reference named."""
    c = [f"{v:.4g}" for v in cuts]
    parts = [f"{what} 1: `{column}` ≤ {c[0]} (the reference)"]
    for g in range(1, len(c)):
        parts.append(f"{what} {g + 1}: {c[g - 1]} < `{column}` ≤ {c[g]}")
    parts.append(f"{what} {len(c) + 1}: `{column}` > {c[-1]}")
    return "; ".join(parts)


# ── a data-derived ("optimal") cut point ──────────────────────────────────────


def optimal_cut(values: Any, y: Any, task: str) -> tuple[float, float, int]:
    """The minimum-p cut point: among the distinct type-7 quantiles of ``values`` from the 10th to
    the 90th percentile, the one whose two groups' outcomes differ most (Welch's t for a numeric
    outcome, Pearson's χ² for a yes/no one). Returns (cut, its minimum p, candidates tried). Its p
    is not a valid test: the search inflates it (Altman et al. 1994)."""
    from scipy import stats

    x = np.asarray(values, dtype=float)
    yy = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(yy)
    x, yy = x[keep], yy[keep]
    if task not in ("regression", "binary"):
        raise ValueError("A data-derived cut point is searched here for a numeric or a yes/no "
                         "outcome only.")
    candidates = np.unique(_quantile(x, np.linspace(OPTIMAL_RANGE[0], OPTIMAL_RANGE[1], 81)))
    best: tuple[float, float] | None = None
    tried = 0
    for c in candidates:
        high = x > c
        if high.sum() < 2 or (~high).sum() < 2:
            continue
        tried += 1
        if task == "regression":
            p = float(stats.ttest_ind(yy[high], yy[~high], equal_var=False).pvalue)
        else:
            table = np.array([[np.sum(yy[high] == 1), np.sum(yy[high] != 1)],
                              [np.sum(yy[~high] == 1), np.sum(yy[~high] != 1)]], dtype=float)
            if (table.sum(axis=0) == 0).any():
                continue
            p = float(stats.chi2_contingency(table, correction=False)[1])
        if not math.isfinite(p):
            continue
        if best is None or p < best[1]:
            best = (float(c), p)
    if best is None:
        raise ValueError("No cut point leaves two groups to compare.")
    return best[0], best[1], tried


# ── what each form learns, and the pipeline step ─────────────────────────────


def _form_of(spec: Any) -> tuple[str, int | None]:
    if isinstance(spec, str):
        return spec, None
    get = spec.get if isinstance(spec, Mapping) else (lambda k, s=spec: getattr(s, k, None))
    return str(get("form")), (int(get("knots")) if get("knots") is not None else None)


def _field(spec: Any, name: str, default: Any = None) -> Any:
    if isinstance(spec, str) or spec is None:
        return default
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    return getattr(spec, name, default)


def place(spec: Any, values: Any, y: Any = None, task: str | None = None) -> dict[str, Any]:
    """What a form learns from ``values`` (its recorded values on the fitting rows): a spline's
    knots, a mass at zero's knots among consumers, quintiles' cut points, declared categories'
    cut points, the minimum-p cut point (which reads ``y``); and, for a declared exposure's spline,
    the companion quintiles' cut points (``companion``)."""
    form, k = _form_of(spec)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {"form": form, "n_observed": int(len(x))}
    if form == "spline":
        knots, notes = rcs_knots(x, k or DEFAULT_KNOTS)
        out.update(knots=[float(v) for v in knots], notes=list(notes))
        if _field(spec, "companion"):
            out["companion_cuts"] = [float(v) for v in quantile_cuts(x)]
    elif form == "zero_spline":
        consumers = x[x > 0]
        knots, notes = rcs_knots(consumers, k or DEFAULT_KNOTS)
        out.update(knots=[float(v) for v in knots], notes=list(notes),
                   zeros=int((x == 0).sum()))
    elif form == "quintiles":
        out["cuts"] = [float(v) for v in quantile_cuts(x)]
    elif form == "categories":
        cuts = _field(spec, "cuts") or []
        if not cuts:
            raise ValueError("Categories need their cut points declared.")
        out["cuts"] = [float(v) for v in cuts]
    elif form == "optimal":
        if y is None:
            raise ValueError("A data-derived cut point is searched on the outcome, which this step "
                             "was not given.")
        cut, p, tried = optimal_cut(values, y, task or "regression")
        out.update(cuts=[cut], p=p, tried=tried)
    return out


def _task_of(y: Any) -> str:
    values = np.asarray(y)
    if values.dtype.names:  # a time to event
        return "time_to_event"
    finite = pd.to_numeric(pd.Series(values.ravel()), errors="coerce").dropna()
    return "binary" if set(np.unique(finite)) <= {0, 1} else "regression"


class ExposureForms(TransformerMixin, BaseEstimator):
    """Each named column replaced by its form: a spline basis, quintile or category indicators, a
    data-derived cut's indicator, or a consumer indicator beside a spline among consumers.

    ``forms``: column -> ``{"form": "spline", "knots": 4}``, ``{"form": "quintiles"}``,
    ``{"form": "categories", "cuts": [...]}``, ``{"form": "optimal"}`` or
    ``{"form": "zero_spline", "knots": 4}`` (a ``linear`` form, or a column not named, passes
    through). Fit on the rows each fit sees; the knots, the cut points and each quintile's median
    are kept. Row-local once fit.
    """

    def __init__(self, forms: Mapping[str, Any] | None = None):
        self.forms = forms

    def _specs(self) -> dict[str, Any]:
        specs = {}
        for column, spec in (self.forms or {}).items():
            form, _ = _form_of(spec)
            if form not in FORMS:
                raise ValueError(f"Unknown exposure form {form!r} for {column}.")
            if form != "linear":
                specs[str(column)] = spec
        return specs

    def _plan(self) -> dict[str, tuple[str, int | None]]:
        plan = {}
        for column, spec in self._specs().items():
            form, knots = _form_of(spec)
            plan[column] = (form, knots or (DEFAULT_KNOTS if form in SPLINE_FORMS else None))
        return plan

    def _start(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} needs a pandas DataFrame with named columns.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.knots_: dict[str, np.ndarray] = {}
        self.knot_notes_: dict[str, list[str]] = {}
        self.cuts_: dict[str, np.ndarray] = {}
        self.medians_: dict[str, np.ndarray] = {}
        self.counts_: dict[str, list[int]] = {}
        self.optimal_: dict[str, dict[str, Any]] = {}
        self.zeros_: dict[str, int] = {}
        self.companion_cuts_: dict[str, np.ndarray] = {}

    def _values(self, X: pd.DataFrame, column: str) -> np.ndarray:
        if column not in X.columns:
            raise ValueError(f"`{column}` is not among the model's inputs at this step (an "
                             f"energy adjustment may have replaced it), so its form cannot "
                             f"be applied.")
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(f"`{column}` is not numeric, so it has no spline or quintiles.")
        return X[column].to_numpy(dtype=float, na_value=np.nan)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ExposureForms":
        self._start(X)
        for column, spec in self._specs().items():
            values = self._values(X, column)
            form, _ = _form_of(spec)
            yy = None
            if form == "optimal" and y is None:
                # Fit without the outcome (the design's lineage): the median holds the place, and
                # the lineage says so; every model fit searches the outcome.
                finite = values[np.isfinite(values)]
                self._adopt(column, {"form": "optimal", "cuts": [float(np.median(finite))],
                                     "p": None, "tried": None}, values, strict=True)
                continue
            if form == "optimal":
                yy = np.asarray(y)
            self._adopt(column, place(spec, values, yy, _task_of(yy) if yy is not None else None),
                        values, strict=True)
        return self

    def _adopt(self, column: str, params: Mapping[str, Any], values: np.ndarray, *,
               strict: bool) -> None:
        """Hold what ``column``'s form learned (``params``, from :func:`place`); ``strict``: the
        fitting rows must fill every group (a fit on its own rows), else empty groups are kept
        (a completed copy at cut points placed on the observed values)."""
        form = str(params.get("form") or self._plan()[column][0])
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if form in SPLINE_FORMS:
            self.knots_[column] = np.asarray(params["knots"], dtype=float)
            self.knot_notes_[column] = list(params.get("notes") or [])
            if form == "zero_spline":
                self.zeros_[column] = int((v == 0).sum())
            if params.get("companion_cuts") is not None:
                self.companion_cuts_[column] = np.asarray(params["companion_cuts"], dtype=float)
            return
        cuts = np.asarray(params["cuts"], dtype=float)
        groups = len(cuts) + 1
        group = quantile_group(v, cuts)
        counts = np.bincount(group, minlength=groups)[:groups]
        if strict and form == "quintiles" and (len(np.unique(cuts)) < QUINTILES - 1
                                              or (counts == 0).any()):
            raise ValueError(f"`{column}` has too many tied values to be cut into five "
                             f"groups: its quintile cut points are "
                             f"{', '.join(f'{c:g}' for c in cuts)}.")
        if strict and form in ("categories", "optimal") and (counts == 0).any():
            empty = [g + 1 for g in range(groups) if counts[g] == 0]
            raise ValueError(f"`{column}` has no value in group {empty[0]} of its cut points "
                             f"({', '.join(f'{c:g}' for c in cuts)}).")
        self.cuts_[column] = cuts
        self.medians_[column] = np.array([float(np.median(v[group == g])) if counts[g]
                                          else float("nan") for g in range(groups)])
        self.counts_[column] = [int(c) for c in counts]
        if form == "optimal":
            self.optimal_[column] = {"cut": float(cuts[0]), "p": params.get("p"),
                                     "tried": params.get("tried")}

    def _outputs(self, column: str) -> list[str]:
        plan = self._plan()
        if column not in plan:
            return [column]
        form, _ = plan[column]
        if form == "spline":
            return spline_names(column, len(self.knots_[column]))
        if form == "zero_spline":
            return [f"{column}{CONSUMER_SUFFIX}", *spline_names(column, len(self.knots_[column]))]
        if form == "quintiles":
            return quintile_names(column)
        if form == "categories":
            return category_names(column, len(self.cuts_[column]))
        return [f"{column}{ABOVE_SUFFIX}"]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names: list[str] = []
        for column in self.feature_names_in_:
            names.extend(self._outputs(str(column)))
        return np.asarray(names, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "knots_"):
            raise ValueError("ExposureForms is not fitted yet.")
        plan = self._plan()
        parts: dict[str, Any] = {}
        for name in X.columns:
            column = str(name)
            if column not in plan:
                parts[column] = X[name]
                continue
            values = X[name].to_numpy(dtype=float, na_value=np.nan)
            form, _ = plan[column]
            if form in SPLINE_FORMS:
                basis = rcs_basis(values, self.knots_[column])
                names = spline_names(column, len(self.knots_[column]))
                if form == "zero_spline":
                    # Non-consumers are their own category (the reference): the consumer
                    # indicator; every spline column is 0 at zero, the knots all lying above it.
                    parts[f"{column}{CONSUMER_SUFFIX}"] = np.where(np.isfinite(values),
                                                                   (values > 0).astype(float),
                                                                   np.nan)
                parts[names[0]] = values
                for j, out in enumerate(names[1:]):
                    parts[out] = basis[:, j]
                continue
            if form == "optimal":
                cut = float(self.cuts_[column][0])
                parts[f"{column}{ABOVE_SUFFIX}"] = np.where(np.isfinite(values),
                                                            (values > cut).astype(float), np.nan)
                continue
            group = quantile_group(values, self.cuts_[column])
            outputs = self._outputs(column)
            for g, out in enumerate(outputs, start=1):
                parts[out] = np.where(group < 0, np.nan, (group == g).astype(float))
        return pd.DataFrame(parts, index=X.index)

    def scores(self, column: str, values: Any) -> np.ndarray:
        """The trend score: each value's quintile median (from the fitting rows)."""
        group = quantile_group(values, self.cuts_[column])
        medians = self.medians_[column]
        return np.where(group < 0, np.nan, medians[np.clip(group, 0, len(medians) - 1)])

    def lineage(self) -> list[dict[str, Any]]:
        """``{output, inputs, operation, formula}`` per output, as the energy step's."""
        plan = self._plan()
        entries: list[dict[str, Any]] = []
        for column in (str(c) for c in self.feature_names_in_):
            if column not in plan:
                entries.append({"output": column, "inputs": [column], "operation": "kept",
                                "formula": None})
                continue
            form, _ = plan[column]
            if form in SPLINE_FORMS:
                knots = self.knots_[column]
                op = f"restricted cubic spline, {len(knots)} knots"
                formula = knot_sentence(column, knots, self.knot_notes_[column])
                if form == "zero_spline":
                    entries.append({"output": f"{column}{CONSUMER_SUFFIX}", "inputs": [column],
                                    "operation": "consumer indicator",
                                    "formula": f"1 when `{column}` > 0; non-consumers "
                                               f"(`{column}` = 0) are the reference"})
                    op += " among consumers"
                for out in spline_names(column, len(knots)):
                    entries.append({"output": out, "inputs": [column], "operation": op,
                                    "formula": formula})
                continue
            cuts = self.cuts_[column]
            if form == "optimal":
                searched = (self.optimal_.get(column) or {}).get("p") is not None
                where = (f"the cut point with the smallest outcome p-value on the fitting rows"
                         if searched else "the median, holding the place of the cut point each "
                                          "model fit searches on the outcome")
                entries.append({"output": f"{column}{ABOVE_SUFFIX}", "inputs": [column],
                                "operation": "data-derived cut point",
                                "formula": f"1 when `{column}` > {cuts[0]:.4g}, {where}; "
                                           f"`{column}` ≤ {cuts[0]:.4g} is the reference"})
                continue
            word = "quintile" if form == "quintiles" else "category"
            for g, out in enumerate(self._outputs(column), start=1):
                low, high = cuts[g - 1], (cuts[g] if g < len(cuts) else None)
                within = f"{low:.4g} < {column}" + (f" ≤ {high:.4g}" if high is not None else "")
                entries.append({"output": out, "inputs": [column], "operation": f"{word} indicator",
                                "formula": f"1 when {within} ({word} {g + 1}); {word} 1, "
                                           f"{column} ≤ {cuts[0]:.4g}, is the reference"})
        return entries


def nonlinear_outputs(step: ExposureForms, column: str) -> list[str]:
    """The columns a nonlinearity test reads: a spline's nonlinear terms (among consumers for a
    mass at zero's)."""
    outputs = step._outputs(column)
    form = step._plan()[column][0]
    if form == "spline":
        return outputs[1:]
    if form == "zero_spline":
        return outputs[2:]
    return []


def knot_sentence(column: str, knots: Sequence[float], notes: Sequence[str] = ()) -> str:
    """``knots at 12.1, 25, 41.3 and 70.2 (the 5th, 35th, 65th and 95th percentiles)``."""
    values = [f"{k:.4g}" for k in knots]
    listed = values[0] if len(values) == 1 else f"{', '.join(values[:-1])} and {values[-1]}"
    nominal = KNOT_PERCENTILES.get(len(knots))
    where = ""
    if nominal is not None and not notes:
        pct = [f"{100 * p:g}" for p in nominal]
        named = [f"{p}{_ordinal_suffix(p)}" for p in pct]
        where = f" (the {', '.join(named[:-1])} and {named[-1]} percentiles)"
    extra = f"; {'; '.join(notes)}" if notes else ""
    return f"knots of {column} at {listed}{where}{extra}"


def _ordinal_suffix(number: str) -> str:
    if "." in number:
        return "th"
    n = int(number)
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def form_columns(step: ExposureForms) -> dict[str, list[str]]:
    """Formed column -> its output columns in the model matrix."""
    return {column: step._outputs(column) for column in step._plan()}


def formed_meanings(terms: Mapping[str, str], step: ExposureForms | None) -> dict[str, str]:
    """Each formed column's meaning carried to the matrix columns the form made of it.

    ``terms`` maps a matrix column to what its coefficient means (``methods.energy.describe_model``,
    WP6), read as if each formed column were still one straight-line term; the form step turns it
    into a spline basis or indicators (WP12a), whose coefficients mean a part of that curve, not a
    per-unit effect."""
    out = dict(terms)
    if step is None:
        return out
    plan = step._plan()
    for column, outputs in form_columns(step).items():
        base = out.pop(column, None) if column not in outputs else out.get(column)
        if base is None:
            continue
        form = plan[column][0]
        for k, name in enumerate(outputs):
            if form == "spline":
                out[name] = (f"{base}: term {k + 1} of {len(outputs)} of its restricted cubic "
                             f"spline (read the curve and its tests, not one term)")
            elif form == "zero_spline":
                out[name] = (f"{base}: consumers against non-consumers at zero intake" if k == 0
                             else f"{base}: term {k} of {len(outputs) - 1} of its spline among "
                                  f"consumers (read the curve and its tests, not one term)")
            elif form == "quintiles":
                out[name] = f"{base}: quintile {k + 2} against the lowest"
            elif form == "categories":
                out[name] = f"{base}: category {k + 2} against the lowest"
            else:
                out[name] = f"{base}: above the data-derived cut point against below it"
    return out


def formed_name(column: str, adjustment: Any) -> str:
    """The column the form step receives for ``column`` after the energy step: the step that runs
    first may have replaced it (``EnergyAdjuster``'s output names: the residual method's
    ``{column}_adj``, the density methods' ``{column}_per_{energy}``). The form then shapes the
    energy-adjusted intake, and its terms carry that name. Unchanged without an adjustment, under
    the standard model (energy enters beside the nutrient) and for a column it does not adjust."""
    if adjustment is None:
        return column
    get = adjustment.get if isinstance(adjustment, Mapping) else (
        lambda k, a=adjustment: getattr(a, k, None))
    method, energy = get("method"), get("energy_column")
    if column not in (get("nutrients") or []) or method in (None, "none", "standard"):
        return column
    if method in ("residual", "residual_energy_dropped"):
        return f"{column}_adj"
    if method in ("density", "density_multivariate"):
        return f"{column}_per_{energy}"
    return column  # the partition replaces the nutrient by its kcal; the decision refuses a form


def adjusted_forms(forms: Mapping[str, Any], adjustment: Any) -> dict[str, Any]:
    """``forms`` keyed by the columns the form step receives (:func:`formed_name`)."""
    return {formed_name(str(column), adjustment): spec for column, spec in forms.items()}


def model_terms(predictors: Sequence[str], forms: Mapping[str, Any] | None) -> int:
    """How many columns ``predictors`` put in the model matrix once formed: a spline k − 1, a
    quintile exposure 4, categories one per cut point, a cut point 1, a mass at zero's k (its
    consumer indicator and its spline), any other predictor 1 (the count the shelf's
    per-predictor rules read)."""
    total = 0
    for column in predictors:
        spec = (forms or {}).get(column)
        form, knots = _form_of(spec) if spec is not None else ("linear", None)
        if form == "spline":
            total += (knots or DEFAULT_KNOTS) - 1
        elif form == "zero_spline":
            total += (knots or DEFAULT_KNOTS)
        elif form == "quintiles":
            total += QUINTILES - 1
        elif form == "categories":
            total += len(_field(spec, "cuts") or []) or 1
        else:
            total += 1
    return total


# ── the form as the state holds it: on the exposure's final scale ─────────────


def transform_signature(state: Any, column: str) -> str:
    """The domain transforms ``column`` passes through before its form (MODELING_SEQUENCE §1 row
    4): its scale's scoring, the omics normalization and batch adjustment, and the energy model's
    residual or density (with its log, energy column and strata). ``"raw"`` when none does. A form
    declared on one signature is stale on another (:func:`current_forms`)."""
    parts: list[str] = []
    for sc in getattr(state, "scales", None) or []:
        if _field(sc, "name") == column:
            parts.append("score:" + ":".join([str(_field(sc, "scoring")),
                                              ",".join(_field(sc, "items") or []),
                                              ",".join(sorted(_field(sc, "reverse") or [])),
                                              f"{_field(sc, 'low')}-{_field(sc, 'high')}"]))
    try:
        from turbotab.core.methods.omics import normalization_of

        norm = normalization_of(state)
    except Exception:  # noqa: BLE001 - a state with no findings normalizes nothing
        norm = None
    if norm is not None and column in (norm.get("columns") or []):
        parts.append(f"normalize:{norm.get('method')}:{norm.get('kind')}")
    batch = getattr(state, "batch", None)
    roles = getattr(state, "roles", None) or {}
    if (batch is not None and _field(batch, "method") == "reference_combat"
            and roles.get(column) == "exposure"):
        parts.append(f"batch:{_field(batch, 'method')}:{_field(batch, 'column')}")
    adj = getattr(state, "energy_adjustment", None)
    if adj is not None and column in (_field(adj, "nutrients") or []) and \
            _field(adj, "method") not in ("none", "standard"):
        parts.append(f"energy:{_field(adj, 'method')}:{_field(adj, 'energy_column')}:"
                     f"{'log' if _field(adj, 'log_transform') else 'linear'}:"
                     f"{_field(adj, 'strata') or ''}")
    return ";".join(parts) or "raw"


def scale_words(state: Any, column: str) -> str:
    """The unit an estimate of ``column`` is per, on its final scale: ``unit of `fiber_g```,
    ``unit of `fiber_g`'s energy-adjusted residual on `kcal```, ``point of the `pss` score``."""
    tick = f"`{column}`"
    for sc in getattr(state, "scales", None) or []:
        if _field(sc, "name") == column:
            n = len(_field(sc, "items") or [])
            return (f"point of the {tick} score (the {_field(sc, 'scoring')} of {n} items, each "
                    f"{_field(sc, 'low')}–{_field(sc, 'high')})")
    adj = getattr(state, "energy_adjustment", None)
    if adj is not None and column in (_field(adj, "nutrients") or []):
        method, E = _field(adj, "method"), f"`{_field(adj, 'energy_column')}`"
        strata = _field(adj, "strata")
        within = f" within each level of `{strata}`" if strata else ""
        if method in ("residual", "residual_energy_dropped"):
            if _field(adj, "log_transform"):
                return (f"unit of {tick}'s energy-adjusted residual (log {tick} on log {E}"
                        f"{within}, back-transformed at the geometric mean of {E})")
            return f"unit of {tick}'s energy-adjusted residual on {E}{within}, at mean energy"
        if method in ("density", "density_multivariate"):
            return f"unit of {tick} per unit of {E} (a density)"
    try:
        from turbotab.core.methods.omics import normalization_of

        norm = normalization_of(state)
    except Exception:  # noqa: BLE001
        norm = None
    if norm is not None and column in (norm.get("columns") or []):
        return f"unit of {tick} after its {norm.get('method')} normalization"
    return f"unit of {tick}"


def _forms_of(state: Any) -> dict[str, Any]:
    return {str(c): f for c, f in (getattr(state, "exposure_forms", None) or {}).items()
            if f is not None}


def is_current(state: Any, column: str, spec: Any) -> bool:
    """A form stands on the scale it was declared on (a form recorded without one stands), while
    its column is an amount: a column the user later recorded as codes for categories enters as
    one indicator per level and takes no form (BLUEPRINT §14.3: every confirmation is honored)."""
    from turbotab.core.readings import confirmed_codes

    if column in confirmed_codes(state):
        return False
    scale = _field(spec, "scale")
    return scale is None or scale == transform_signature(state, column)


def rule_stale(spec: Any, card: Mapping[str, Any] | None) -> bool:
    """A spline whose k was set by Harrell's rule on an effective sample size the analyzed rows no
    longer have (the card's, read on the rows the fit sees now): an exclusion, the complete cases
    or the consumers-only domain changed them. Its k and the n its record states are re-derived
    by asking again, never kept (the FORM repair, item 2b)."""
    if card is None or _form_of(spec)[0] not in SPLINE_FORMS or \
            _field(spec, "knots_rule") != "harrell":
        return False
    present = card.get("n_effective")
    if present is None:
        return False
    said = _field(spec, "n_effective")
    return said is None or abs(float(said) - float(present)) > 1e-9 * max(1.0, abs(float(present)))


def current_forms(state: Any) -> dict[str, Any]:
    """Each declared form that still stands: declared on the exposure's present scale, of a column
    still read as an amount. A form a later transform left stale is not applied, its knots and cut
    points not kept: the form question asks it again (MODELING_SEQUENCE §2)."""
    return {c: f for c, f in _forms_of(state).items() if is_current(state, c, f)}


def stale_forms(state: Any, card: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Each declared form a later domain transform of its column left stale, and, given the form
    card read on the present rows, each whose k by the rule was read on other rows (re-asked). A
    column since recorded as codes is in neither: it takes no form, so nothing is asked again."""
    codes = _codes(state)
    return {c: f for c, f in _forms_of(state).items()
            if c not in codes and (not is_current(state, c, f) or rule_stale(f, card))}


def _codes(state: Any) -> set[str]:
    from turbotab.core.readings import confirmed_codes

    return set(confirmed_codes(state))


def estimand_unit(state: Any, column: str) -> str:
    """The estimand's unit for ``column``: the unit the standing form declared, else the unit of
    its present scale (never a stale form's)."""
    spec = current_forms(state).get(column)
    unit = _field(spec, "unit") if spec is not None else None
    return str(unit) if unit else scale_words(state, column)


def exposure_of(state: Any) -> str | None:
    """The one declared exposure (an exposure family has none: its members are each linear)."""
    from turbotab.core import estimand as est

    spec = est.current_estimand(state)
    if spec is None or _field(spec, "family"):
        return None
    return str(_field(spec, "exposure"))


def design_forms(state: Any, present: Sequence[str], numeric: Sequence[str]) -> dict[str, Any]:
    """The forms the design applies: the standing ones (:func:`current_forms`) of present numeric
    columns, each as a plain dict; a declared exposure's spline under inference carries the
    companion quintiles (MODELING_SEQUENCE §0 ruling 2: "the quintile table is produced beside it
    by default")."""
    exposure = exposure_of(state) if getattr(state, "purpose", None) == "inference" else None
    out: dict[str, Any] = {}
    have, num = set(present), set(numeric)
    for c, f in current_forms(state).items():
        if c not in have or c not in num or _form_of(f)[0] == "linear":
            continue
        spec = f.model_dump() if hasattr(f, "model_dump") else dict(f)
        if c == exposure and spec.get("form") == "spline":
            spec["companion"] = True
        out[c] = spec
    return out


def domain_columns(state: Any) -> list[str]:
    """The declared exposure, when its form restricts the analysis to consumers (an estimand
    change: who the estimate is about, not the scale it is on, so a transform leaves it standing;
    another exposure does not)."""
    if getattr(state, "purpose", None) != "inference":
        return []
    exposure = exposure_of(state)
    domains = getattr(state, "form_domains", None) or {}
    return [c for c, v in domains.items() if c == exposure and v == "consumers"]


# ── labels the form changes (MODELING_SEQUENCE §2, corrected relations) ───────


def residual_form(state: Any, column: str, spec: Any = None) -> bool:
    """A spline or categories on an energy residual (the identity with the standard model, linear
    in N, does not survive a nonlinear basis)."""
    spec = spec if spec is not None else current_forms(state).get(column)
    if spec is None or _form_of(spec)[0] == "linear":
        return False
    adj = getattr(state, "energy_adjustment", None)
    return (adj is not None and _field(adj, "method") in ("residual", "residual_energy_dropped")
            and column in (_field(adj, "nutrients") or []))


def residual_label(column: str, energy: str | None) -> str:
    """The label of a curve or categories on a residual (MODELING_SEQUENCE §2): "It is the
    nutrient's curve on the energy-adjusted scale at mean energy, not the substitution curve.\""""
    return (f"the curve of `{column}` on the energy-adjusted scale at mean energy (its residual on "
            f"`{energy}`), not the substitution curve at fixed total energy: the residual model "
            f"equals the standard model only for a straight line; a spline of `{column}` with "
            f"`{energy}` beside it (the standard model) is the route to the substitution curve")


def substitution_route(column: str, energy: str | None, nutrients: Sequence[str],
                       knots: int | None = None) -> list[dict[str, Any]]:
    """spline(N) + E: the standard model with the spline on the nutrient itself (the decisions)."""
    return [{"kind": "set_energy_adjustment", "method": "standard", "energy_column": energy,
             "nutrients": list(nutrients)},
            {"kind": "set_exposure_form", "column": column, "form": "spline",
             **({"knots": knots} if knots else {})}]


def substitution_curve_label(state: Any, donor: str, recipient: str) -> str | None:
    """A log or a spline on an energy component: the substitution curve still moves k kcal, but
    the effect now depends on k and on each person's intake, so the curve is an average over the
    stated population at the stated k (MODELING_SEQUENCE §2, corrected; the review: "Draft 1 said
    'ratio rather than difference'; that was wrong"). None for a linear model of both."""
    adj = getattr(state, "energy_adjustment", None)
    nutrients = set(_field(adj, "nutrients") or []) if adj is not None else set()
    logged = [c for c in (donor, recipient) if adj is not None and c in nutrients
              and _field(adj, "log_transform")]
    shaped = [c for c in (donor, recipient)
              if _form_of(current_forms(state).get(c) or "linear")[0] != "linear"]
    if not logged and not shaped:
        return None
    what = []
    if logged:
        what.append(f"a log of {' and '.join(f'`{c}`' for c in logged)}")
    if shaped:
        what.append(f"a nonlinear form of {' and '.join(f'`{c}`' for c in shaped)}")
    return (f"With {' and '.join(what)}, moving k kcal no longer has one effect per kcal: each "
            f"person's effect depends on k and on their own intake, so each point of the curve is "
            f"an average of those effects over the rows the curve reads, at that k (k-specific), "
            f"not a coefficient difference"
            + ("; an all-components contrast is not defined on logged components" if logged
               else ""))


SHARE_REALLOCATION = ("Reallocating shares of a composition (an isometric log-ratio model of the "
                      "diet, Leite 2016) is not in this version: it goes to v2.x (MODELING_SEQUENCE "
                      "§0 ruling 11). The kcal substitution moves energy from one source to another "
                      "at fixed total energy; the share-of-energy swap moves a percentage of each "
                      "person's energy.")


# ── the options, labeled as north star 5 asks ────────────────────────────────

Role = Literal["exposure", "confounder"]


def _option(value: str, label: str, customary: str, sound: str, consequence: str,
            rung: str) -> dict[str, Any]:
    return {"value": value, "label": label, "customary": customary, "sound": sound,
            "consequence": consequence, "rung": rung}


def options(purpose: str | None, role: Role = "exposure", *, mass_at_zero: bool = False,
            residual: bool = False) -> list[dict[str, Any]]:
    """The form options for ``purpose`` and ``role``, ordered by soundness for it, each labeled
    *customary in* (with its source) and *sound for* (with the reason), independently, with its
    leash rung (MODELING_SEQUENCE §4).

    Under inference the spline leads (for a mass at zero, non-consumers beside a spline among
    consumers leads), quintiles rank lower and are produced beside it anyway, and data-derived cut
    points are blocked and recorded; a confounder's categories into three or fewer groups too.
    Under prediction the spline still leads (it nests the line), and the cut forms come last.
    """
    inference = purpose == "inference"
    on_residual = (" On a residual it is the curve on the energy-adjusted scale at mean energy, "
                   "not the substitution curve." if residual else "")
    spline = _option(
        "spline", "Restricted cubic spline",
        "Increasingly customary in nutritional epidemiology: \"now near-default\" "
        "(NUTRITION_PACK §07G; Desquilbet & Mariotti 2010).",
        "Sound: a smooth curve with few parameters, linear in the tails, with a test of "
        "nonlinearity; knots at Harrell's percentiles of the fitting rows." + on_residual,
        "Adds a curve's terms per exposure and a test of whether it bends.", "recommended")
    linear = _option(
        "linear", "Straight line", "Customary: one coefficient per unit of intake.",
        ("Sound when the relation is close to a line; a curve it misses biases the slope."
         if role == "exposure" else
         "Often controls a confounder well (Brenner & Blettner 1997); a curve it misses leaves "
         "residual confounding."),
        "One coefficient per unit; no curvature.", "available")
    quintiles = _option(
        "quintiles", "Quintiles with a trend test",
        "Customary in nutritional epidemiology: \"quintiles remain expected alongside\" the "
        "spline (NUTRITION_PACK §07G).",
        ("Weaker for inference: cutting a continuous intake into fifths discards the variation "
         "within each fifth and assumes a step at each cut; the trend test scores each fifth by "
         "its median." if inference else
         "Weak for prediction: five steps discard the variation within each fifth."),
        "Four indicators against the lowest fifth, and a trend across medians.", "rank_lower")
    categories = _option(
        "categories", "Categories at declared cut points",
        "Customary for clinical bands (BMI classes, age bands) and for confounders.",
        ("Coarser than the variable: the categories assume a step at each cut. A confounder in "
         "three or fewer groups leaves serious residual confounding (Brenner & Blettner 1997)."
         if role == "confounder" else
         "Coarser than the variable; cut points declared from outside these data."),
        "One indicator per group above the lowest, the boundaries stated.", "rank_lower")
    optimal = _option(
        "optimal", "A data-derived cut point",
        "Seen in clinical papers: the cut point with the smallest p-value.",
        ("Unsound for inference: a data-derived \"optimal\" cut point leads to serious bias "
         "(Altman & Royston 2006)." if inference else
         "Searched inside each training fold; still discards the variation on each side."),
        "Two groups split where the outcome differs most on these rows.",
        "block_and_record" if inference else "rank_lower")
    out = [spline, linear, categories, quintiles, optimal] if role == "confounder" else \
        [spline, linear, quintiles, categories, optimal]
    if mass_at_zero and role == "exposure":
        zero = _option(
            "zero_spline", "Non-consumers apart, a spline among consumers",
            "STROBE-nut nut-11 asks how non-consumers were handled.",
            "Sound for a mass at zero: the non-consumers are their own category (the reference) "
            "and the curve among consumers has knots at consumers' percentiles.",
            "A consumer indicator and a spline among consumers.", "recommended")
        out.insert(0, zero)
        if inference:
            out.insert(1, _option(
                "consumers_only", "Consumers only (an estimand change)",
                "Common for episodically consumed foods; STROBE-nut nut-14 asks which population.",
                "Answers a different question: the effect among consumers, not in everyone.",
                "Non-consumers leave the analysis; the spline is among consumers.", "available"))
    if not inference:
        for o in out:
            o["rung"] = {"spline": "recommended", "zero_spline": "recommended",
                         "linear": "available"}.get(o["value"], "rank_lower")
    return out


# ── the form question (MODELING_SEQUENCE §1 row 5) ───────────────────────────


def _info_of(info: Mapping[str, Any] | None, column: str) -> Mapping[str, Any] | None:
    if info is None:
        return None
    found = info.get(column)
    if found is None:
        return None
    if isinstance(found, Mapping):
        return found
    return {"dtype": getattr(found, "dtype", None), "n_unique": getattr(found, "n_unique", None)}


def form_candidates(state: Any) -> list[str]:
    """The columns the form question reads under inference: the declared exposure and each
    adjusted covariate (an exposure family's members and total energy are stated, never read)."""
    from turbotab.core import estimand as est

    if getattr(state, "purpose", None) != "inference":
        return []
    spec = est.current_estimand(state)
    if spec is None:
        return []
    exposure = None if _field(spec, "family") else str(_field(spec, "exposure"))
    roles = est.predictor_roles(state)
    fe = est.fixed_effects_column(state)
    out = [exposure] if exposure is not None else []
    out += [c for c, d in est.derived_roles(state).items()
            if d.adjusted and c != exposure and c != fe and c in roles
            and roles.get(c) != "energy"]
    return out


def code_reading(state: Any, column: str, facts: Mapping[str, Any] | None) -> Any:
    """``column``'s code-or-amount reading while the ledger holds it unsettled (BLUEPRINT §14.1:
    the form of a column is a number-changing consumer of whether its numbers are codes or
    amounts, and only ``readings.py`` decides that kind); None when it is settled or not asked:
    the values settle it (decimals that read as amounts), an answer settled it, or another answer
    reads the column as an amount (the energy model computing with it, a scale summing it, an assay
    lens's exposure)."""
    from turbotab.core import readings as R

    if facts is None:
        return None
    left, amounts = R.energy_plan(state, [column])
    if column in left | amounts | R.scale_items(state):
        return None
    found = R.unsettled_codes(state, [column], {column: facts})
    return found[0] if found else None


def form_plan(state: Any, info: Mapping[str, Any] | None,
              facts: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, list[Any]]:
    """Under inference, the columns whose form is declared before estimates (``needs``: the
    declared exposure and each adjusted continuous confounder, an amount by the readings ledger
    with at least :data:`CONTINUOUS` distinct values), those stated instead, each with why
    (``stated``), and those whose code-or-amount reading the ledger still holds unsettled
    (``waiting``: the question asks the reading first, BLUEPRINT §14.2, never proposing a form on
    a guess). ``info``: column → ``{dtype, n_unique}``; ``facts``: column → the readings ledger's
    whole-number facts (``readings.whole_facts``)."""
    from turbotab.core import estimand as est
    from turbotab.core.readings import confirmed_codes, guess_words

    plan: dict[str, list[Any]] = {"needs": [], "stated": [], "waiting": []}
    if getattr(state, "purpose", None) != "inference":
        return plan
    spec = est.current_estimand(state)
    if spec is None:
        return plan
    needs, stated, waiting = plan["needs"], plan["stated"], plan["waiting"]
    roles = est.predictor_roles(state)
    codes = set(confirmed_codes(state))
    adj = getattr(state, "energy_adjustment", None)
    partitioned = set(_field(adj, "nutrients") or []) if adj is not None and \
        _field(adj, "method") in ("partition", "all_components") else set()
    scales = {_field(sc, "name"): sc for sc in getattr(state, "scales", None) or []}
    fe = est.fixed_effects_column(state)

    def continuous(column: str, role: str) -> tuple[bool, str]:
        if column in scales:
            return True, ""
        if column in codes:
            return False, "declared codes: it enters as indicators"
        found = _info_of(info, column)
        if found is None:
            return False, "not read yet"
        n = int(found.get("n_unique") or 0)
        # The reading is asked only where its answers differ here: with fewer than CONTINUOUS
        # values a column takes no declared form either way (codes: indicators; amounts: as
        # recorded), so the fit asks it, not this question (BLUEPRINT §14.2: ask where it matters).
        reading = (code_reading(state, column, (facts or {}).get(column)) if n >= CONTINUOUS
                   else None)
        if reading is not None:
            waiting.append({"column": column, "role": role,
                            "guess": str(reading.value),
                            "why": (f"its code-or-amount reading is not settled ("
                                    f"{guess_words(reading)}? {reading.evidence}): codes enter "
                                    f"as indicators and take no form; an amount takes one")})
            return False, ""
        if str(found.get("dtype")) not in ("numeric", "integer"):
            return False, "a category: it enters as indicators"
        if n < CONTINUOUS:
            return False, f"{n} distinct values: it enters as recorded"
        return True, ""

    family = bool(_field(spec, "family"))
    exposure = None if family else str(_field(spec, "exposure"))
    if family:
        for c in est.family_exposures(state):
            stated.append({"column": c, "why": "a member of the exposure family: each member "
                                               "enters as a straight line, one test per member"})
    elif exposure is not None:
        if exposure in partitioned:
            stated.append({"column": exposure, "why": "the energy partition replaces it by its "
                                                      "kcal, a straight line"})
        else:
            ok, why = continuous(exposure, "exposure")
            if ok:
                needs.append({"column": exposure, "role": "exposure"})
            elif why:
                stated.append({"column": exposure, "why": why})
    derived = est.derived_roles(state)
    for c, r in roles.items():
        if r == "energy" and c not in derived:
            stated.append({"column": c, "why": "total energy: the energy model sets its term"})
    for c, d in derived.items():
        if not d.adjusted or c == exposure or c == fe or c not in roles:
            continue
        if roles.get(c) == "energy":
            stated.append({"column": c, "why": "total energy: the energy model sets its term"})
            continue
        if c in partitioned:
            stated.append({"column": c, "why": "the energy partition replaces it by its kcal"})
            continue
        ok, why = continuous(c, "confounder")
        if ok:
            needs.append({"column": c, "role": "confounder"})
        elif why:
            stated.append({"column": c, "why": why})
    return plan


def form_needs(state: Any, info: Mapping[str, Any] | None,
               facts: Mapping[str, Mapping[str, Any]] | None = None
               ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """``form_plan``'s columns asked about and those stated instead (a column waiting for its
    reading is in neither)."""
    plan = form_plan(state, info, facts)
    return plan["needs"], plan["stated"]


def form_gate(state: Any, card: Mapping[str, Any] | None) -> tuple[str, str | None] | None:
    """The Router's gate for the form question: stated under prediction (each family's own form
    stands, a spline one tap away); under inference not applicable when no continuous exposure or
    confounder is in the model, else asked."""
    purpose = getattr(state, "purpose", None)
    if purpose == "prediction":
        return ("skipped", "Under prediction each predictor enters as each family takes it; the "
                           "spline benchmark is on the shelf, and a spline can be declared for any "
                           "numeric predictor.")
    if purpose != "inference":
        return None
    from turbotab.core import estimand as est

    if est.current_estimand(state) is None:
        # No exposure to declare (no predictor in the model): no form to declare either; else the
        # question waits behind the estimand's.
        found = est.estimand_gate(state)
        return found if found is not None and found[0] == "not_applicable" else None
    if card is None:
        return None
    if card.get("purpose") != "inference" or not card.get("ready", True):
        return None
    if not card.get("needs") and not card.get("waiting"):
        return ("not_applicable", "No continuous exposure or confounder is in the model, so no "
                                  "form is declared; each enters as recorded.")
    return None


def standing_forms(state: Any, card: Mapping[str, Any] | None) -> dict[str, Any]:
    """The declared forms that stand on the present scale, codes and rows (:func:`current_forms`,
    less those whose k by the rule the card's rows re-derive)."""
    return {c: f for c, f in current_forms(state).items() if not rule_stale(f, card)}


def unanswered_forms(state: Any, card: Mapping[str, Any] | None) -> list[str]:
    """The columns the card asks about with no standing form, and every declared spline whose k
    by the rule was read on other rows than the card's."""
    if card is None:
        return []
    standing = standing_forms(state, card)
    asked = [str(n["column"]) for n in card.get("needs") or [] if str(n["column"]) not in standing]
    return list(dict.fromkeys([*asked, *(c for c, f in current_forms(state).items()
                                         if rule_stale(f, card))]))


def form_answer(state: Any, card: Mapping[str, Any] | None) -> Any:
    """The form question's answer: the standing forms, once every column the card asks about has
    one (a stale form is no answer) and no column waits for its code-or-amount reading; None
    otherwise, and while the card is not read for the present answers (the Router hands it only
    a fresh card: an answer on the card last computed could let a later question be answered
    before this one reopens)."""
    if card is None or getattr(state, "purpose", None) != "inference":
        return None
    if card.get("waiting") or unanswered_forms(state, card):
        return None
    return standing_forms(state, card) or None


def zero_share(values: Any) -> tuple[float, bool]:
    """(the share of recorded values at exactly zero, whether any is negative)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(v):
        return 0.0, False
    return float(np.mean(v == 0)), bool((v < 0).any())


def proposal(role: str, purpose: str | None, k: int, *, mass_at_zero: bool) -> dict[str, Any]:
    """The form the card leads with: a spline with k by the rule; for an exposure with a mass at
    zero, non-consumers apart and a spline among consumers."""
    form = "zero_spline" if (mass_at_zero and role == "exposure") else "spline"
    return {"form": form, "knots": k, "knots_rule": "harrell"}


class _Card(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class FormOptionCard(_Card):
    value: str
    label: str
    customary: str
    sound: str
    consequence: str
    rung: str


class FormNeed(_Card):
    """One column the form question asks about."""

    column: str
    role: Literal["exposure", "confounder"]
    receives: str  # the column the form step receives (an energy-adjusted one, say)
    scale: str  # its transforms (``transform_signature``)
    unit: str  # the estimand's unit on that scale
    zero_share: float
    mass_at_zero: bool
    n_effective: float | None
    rule_knots: int
    rule: str
    options: list[FormOptionCard]
    proposal: dict[str, Any]
    label: str | None = None  # a curve on a residual, labeled
    route: list[dict[str, Any]] | None = None  # spline(N) + E, the substitution curve's route


class FormStated(_Card):
    column: str
    why: str


class FormWaiting(_Card):
    """A column the question would ask about if it holds amounts, while its code-or-amount reading
    is unsettled: the reading is asked first (the question's ask card), and no form is proposed."""

    column: str
    role: Literal["exposure", "confounder"]
    guess: str  # the reading's best guess, "code" or "amount" (never a settlement)
    why: str


class FormsArtifact(_Card):
    """The ``forms`` stage: the functional-form question's card (MODELING_SEQUENCE §1 row 5)."""

    purpose: str | None
    ready: bool
    exposure: str | None
    task: str | None
    n_effective: float | None
    rule_knots: int
    rule: str
    needs: list[FormNeed]
    stated: list[FormStated]
    waiting: list[FormWaiting] = []
    answer: dict[str, Any] | None  # the card's one-tap ``set_forms``: every proposal
    beside: str | None


FORMS_READS = ("purpose", "target", "task", "event", "estimand", "adjustment", "clusters",
               "energy_adjustment", "scales", "findings", "batch", "categorical", "lens",
               "missing", "aggregation", "outcome_order", "roles", "roles_unconfirmed",
               "role_confirmations", "reading_confirmations", "shape_confirmations",
               "column_units", "form_domains")


def forms_card(state: Any, frame: pd.DataFrame | None, info: Mapping[str, Any] | None,
               y: Any = None, task: str | None = None,
               facts: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The form question's card: each column it asks about, the column the form receives, its
    present scale and unit, whether a mass at zero is there, k by the rule on the effective sample
    size, the options labeled for its role, the proposal, and on a residual its label and the
    route to the substitution curve; the columns stated instead, with why; and those waiting for
    their code-or-amount reading (``facts``: the ledger's whole-number facts)."""
    purpose = getattr(state, "purpose", None)
    task = task or getattr(state, "task", None)
    n_eff = effective_n(task, y, getattr(state, "event", None)) if y is not None else None
    k = knots_by_rule(n_eff)
    plan = form_plan(state, info, facts)
    needs, stated = plan["needs"], plan["stated"]
    adj = getattr(state, "energy_adjustment", None)
    entries = []
    for need in needs:
        c, role = need["column"], need["role"]
        share, negative = (zero_share(frame[c]) if frame is not None and c in frame.columns
                           else (0.0, False))
        mass = share >= MASS_AT_ZERO and not negative and transform_signature(state, c) == "raw"
        residual = (adj is not None and _field(adj, "method") in ("residual",
                                                                   "residual_energy_dropped")
                    and c in (_field(adj, "nutrients") or []))
        entry = {
            "column": c, "role": role, "receives": formed_name(c, adj),
            "scale": transform_signature(state, c), "unit": scale_words(state, c),
            "zero_share": round(share, 4), "mass_at_zero": bool(mass),
            "n_effective": n_eff, "rule_knots": k, "rule": rule_words(k, n_eff, task),
            "options": options(purpose, "exposure" if role == "exposure" else "confounder",
                               mass_at_zero=bool(mass), residual=residual),
            "proposal": {**proposal(role, purpose, k, mass_at_zero=bool(mass)),
                         # a consumers-only domain stands in the one-tap answer (never reset)
                         **({"domain": "consumers"} if c in domain_columns(state) else {})},
            "label": (residual_label(c, _field(adj, "energy_column")) if residual else None),
            "route": (substitution_route(c, _field(adj, "energy_column"),
                                         list(_field(adj, "nutrients") or []), k)
                      if residual and role == "exposure" else None),
        }
        entries.append(entry)
    answer = {c["column"]: {k_: v for k_, v in c["proposal"].items()} for c in entries}
    card = {"purpose": purpose, "ready": True, "exposure": exposure_of(state),
            "task": task, "n_effective": n_eff, "rule_knots": k,
            "rule": rule_words(k, n_eff, task), "needs": entries, "stated": stated,
            "waiting": plan["waiting"],
            "answer": ({"kind": "set_forms", "forms": answer} if answer else None),
            "beside": ("Quintiles are produced beside the exposure's spline, their boundaries and "
                       "reference stated, with the p for linear trend (customary)."
                       if purpose == "inference" else None)}
    return FormsArtifact.model_validate(card).model_dump(mode="json")


def forms_stage(ctx: Any) -> dict[str, Any]:
    """The ``forms`` stage: the form question's card (MODELING_SEQUENCE §1 row 5), read on the
    analyzed rows (the cohort's) of the working table."""
    from turbotab.core.graph import Bundle
    from turbotab.core.stages.data import open_store

    state = ctx.state
    if getattr(state, "purpose", None) != "inference":
        return Bundle(data=forms_card(state, None, None))
    cohort = ctx.inputs.get("cohort")
    rows = (cohort.frames["rows"]["row_id"].to_numpy(dtype=np.int64)
            if cohort is not None and "rows" in (cohort.frames or {}) else None)
    from turbotab.core.readings import whole_facts

    with open_store(ctx) as store:
        info = {c.name: {"dtype": c.dtype, "n_unique": c.n_unique} for c in store.info().columns}
        # BLUEPRINT §14.1: whether a candidate's numbers are codes or amounts is the readings
        # ledger's, read from the working table's whole numbers (the values' own facts).
        candidates = [c for c in form_candidates(state) if c in store.columns]
        facts = whole_facts(candidates, info, store)
        wanted, _ = form_needs(state, info, facts)
        columns = [n["column"] for n in wanted if n["column"] in store.columns]
        target = state.target if state.target in store.columns else None
        frame = store.materialize(list(dict.fromkeys([*columns, *([target] if target else [])])),
                                  rows) if (columns or target) else None
    y = frame[target] if frame is not None and target is not None else None
    target_info = ctx.inputs.get("target_info")
    data = getattr(target_info, "data", target_info) or {}
    task = state.task or (data.get("task") if isinstance(data, Mapping) else None)
    return Bundle(data=forms_card(state, frame, info, y, task, facts))


# ── tests under inference ────────────────────────────────────────────────────


def _index(names: Sequence[str]) -> dict[str, int]:
    return {str(n): i for i, n in enumerate(names)}


def _reference(info: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
               idx: Sequence[int]) -> tuple[str, float | None, str] | None:
    """The joint test's reference distribution for this table: (kind, denominator df, words)."""
    covariance = info.get("covariance")
    if covariance == "HC3":
        dfs = [rows[i].get("df") for i in idx if rows[i].get("df") is not None]
        if not dfs:
            return None
        return "F", float(dfs[0]), "HC3 covariance"
    if covariance in ("CR2", "CR1", "CR0"):  # CR0: the Cox model's Lin–Wei sandwich
        g = info.get("n_clusters")
        if not g or g < 2:
            return None
        return "F", float(g - 1), (f"{covariance} cluster-robust covariance by "
                                   f"`{info.get('grouped_by')}` (G = {g:,}), on G − 1 "
                                   f"denominator degrees of freedom as Stata does")
    if covariance == "model":
        return "chi2", None, "the model's information"
    if covariance == "design":
        # MS4: a design-based covariance rests on the design's degrees of freedom; the adjusted
        # Wald F (``models.survey.adjusted_wald``) allows for that.
        d = (info.get("survey") or {}).get("df")
        if not d or d < 1:
            return None
        return "F_design", float(d), (f"the Taylor-linearized covariance over the survey design, "
                                      f"an adjusted Wald F for its {int(d):,} design degrees of "
                                      f"freedom, Korn & Graubard 1990")
    return None


def wald_test(estimates: np.ndarray, cov: np.ndarray, idx: Sequence[int], info: Mapping[str, Any],
              rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The Wald test that the coefficients ``idx`` are all zero, on the table's covariance:
    ``W = bᵀ V⁻¹ b`` over q = rank(V) coefficients; F = W/q on (q, df) for HC3 and cluster-robust
    tables, χ²(q) for model-based ones. None when the table has no covariance to test on."""
    from scipy import stats

    ref = _reference(info, rows, idx)
    if ref is None or cov is None:
        return None
    b = np.asarray(estimates, dtype=float)[list(idx)]
    V = np.asarray(cov, dtype=float)[np.ix_(list(idx), list(idx))]
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
        return None
    q = int(np.linalg.matrix_rank(V))
    if q == 0:
        return None
    W = float(b @ np.linalg.pinv(V) @ b)
    kind, den, words = ref
    if kind == "F_design":
        from turbotab.core.models.survey import adjusted_wald

        adjusted = adjusted_wald(W, q, int(den))
        if adjusted is None:
            return None
        F, den_df, p = adjusted
        return {"statistic": F, "df_num": q, "df_den": den_df, "distribution": "F", "p": p,
                "basis": words}
    if kind == "F":
        F = W / q
        return {"statistic": F, "df_num": q, "df_den": den, "distribution": "F",
                "p": float(stats.f.sf(F, q, den)), "basis": words}
    return {"statistic": W, "df_num": q, "df_den": None, "distribution": "chi2",
            "p": float(stats.chi2.sf(W, q)), "basis": words}


FIRTH_PLR = ("the penalized likelihood-ratio test of the Firth-penalized fit, the restricted fit "
             "keeping the full model's penalty (Heinze & Schemper 2002)")


def firth_lr_test(X: np.ndarray, y: np.ndarray, idx: Sequence[int],
                  full: Any = None) -> dict[str, Any] | None:
    """The penalized likelihood-ratio test that the coefficients ``idx`` are all zero, for a
    logistic model the data separate (a Firth-penalized fit has no Wald covariance to test on,
    its intervals being profile ones): ``2 (l*(β̂) − max_{β: β_idx = 0} l*(β))`` on χ²(q), with
    ``l*`` the penalized log-likelihood ``l(β) + ½ log |I(β)|`` of the full design, as the table's
    own per-coefficient p-values are (``models.inference.firth_profile``) and as logistf's
    nested tests are."""
    from scipy import stats

    from turbotab.core.models.inference import firth_fit

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = [int(j) for j in idx]
    if not idx:
        return None
    fit = full if full is not None else firth_fit(X, y)
    start = np.array(fit.beta, dtype=float)
    start[idx] = 0.0
    restricted = firth_fit(X, y, fixed={j: 0.0 for j in idx}, start=start)
    if not (math.isfinite(fit.loglik) and math.isfinite(restricted.loglik)):
        return None
    statistic = max(0.0, 2.0 * (fit.loglik - restricted.loglik))
    return {"statistic": statistic, "df_num": len(idx), "df_den": None, "distribution": "chi2",
            "p": float(stats.chi2.sf(statistic, len(idx))), "basis": FIRTH_PLR,
            "converged": bool(fit.converged and restricted.converged)}


def pooled_chi_square(statistics: Sequence[float], k: int) -> dict[str, Any] | None:
    """Li, Meng, Raghunathan & Rubin's (1991) D2: χ²(k) statistics, one per imputed copy, pooled
    where the copies give no covariance for D1 (a penalized likelihood-ratio test under separation;
    van Buuren, *Flexible Imputation of Missing Data*, 2nd ed., §5.3.4):
    ``r₂ = (1 + 1/m) var(√dₘ)``, ``D₂ = (d̄/k − (m + 1)/(m − 1) r₂) / (1 + r₂)`` on F(k, ν₂),
    ``ν₂ = k^(−3/m) (m − 1)(1 + 1/r₂)²``. None for fewer than two copies."""
    from scipy import stats

    d = np.asarray(statistics, dtype=float)
    m = len(d)
    if m < 2 or k < 1 or not np.all(np.isfinite(d)):
        return None
    r2 = (1.0 + 1.0 / m) * float(np.var(np.sqrt(np.clip(d, 0.0, None)), ddof=1))
    D2 = max(0.0, (float(d.mean()) / k - (m + 1) / (m - 1) * r2) / (1.0 + r2))
    if r2 <= 1e-12:
        return {"statistic": D2, "df_num": k, "df_den": None,
                "p": float(stats.chi2.sf(D2 * k, k)), "r2": r2}
    nu2 = k ** (-3.0 / m) * (m - 1) * (1.0 + 1.0 / r2) ** 2
    return {"statistic": D2, "df_num": k, "df_den": nu2, "p": float(stats.f.sf(D2, k, nu2)),
            "r2": r2}


def firth_design(matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None
                 ) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """The column names, the design and the 0/1 outcome a Firth table is fit on: the model matrix
    with its intercept first (``models.inference._table``), the event the second class."""
    classes = list(classes or [])
    if len(classes) != 2:
        return None
    names = ["(intercept)", *(str(c) for c in matrix.columns)]
    design = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
    return names, design, (np.asarray(y) == classes[1]).astype(float)


def joint_test_name(result: Mapping[str, Any]) -> str:
    """"Wald test", or the penalized likelihood-ratio test a separated logistic model takes."""
    return ("Penalized likelihood-ratio test" if result.get("basis") == FIRTH_PLR
            else "Wald test")


def _statistic_words(test: Mapping[str, Any]) -> str:
    from turbotab.core.models.inference import format_p

    if test["distribution"] == "F":
        stat = f"F({test['df_num']}, {test['df_den']:,.0f}) = {test['statistic']:.2f}"
    else:
        stat = f"χ²({test['df_num']}) = {test['statistic']:.2f}"
    return f"{stat}, p = {format_p(test['p'])}"


def form_step(pipeline: Any) -> ExposureForms | None:
    named = getattr(pipeline, "named_steps", {}) or {}
    step = named.get("form")
    return step if isinstance(step, ExposureForms) else None


def _before_form(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    """What the form step saw: X through every step before it."""
    names = [name for name, _ in pipeline.steps]
    at = names.index("form")
    return X if at == 0 else pipeline[:at].transform(X)


def _test(column: str, form: str, test: str, result: Mapping[str, Any] | None = None,
          **extra: Any) -> dict[str, Any]:
    base = {"column": column, "form": form, "test": test, "statistic": None, "df_num": None,
            "df_den": None, "distribution": "z", "p": None, "estimate": None, "ci_low": None,
            "ci_high": None, "knots": None, "medians": None, "label": None, "boundaries": None,
            "reference": None}
    if result is not None:
        base.update(statistic=result["statistic"], df_num=result["df_num"],
                    df_den=result["df_den"], distribution=result["distribution"], p=result["p"])
    base.update(extra)
    return base


def exposure_tests(family: Any, pipeline: Any, X: pd.DataFrame, y: Any, *, task: str,
                   clusters: Any, table: Any, outcome: Any = None,
                   survey: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
    """The tests each formed exposure carries under inference, and any concerns about them.

    ``table`` is the family's inference table on ``pipeline`` (rows, ``info``, ``cov``). A spline
    gets two Wald tests: every term (the test of association) and the nonlinear terms; a mass at
    zero's, every term (the consumer indicator with its spline) and the nonlinear terms among
    consumers; a declared exposure's spline also gets its companion quintiles beside it. Quintiles
    get their global test and the trend test, a refit of the family's inference table with the
    exposure scored by its quintile medians; declared categories their global test; a data-derived
    cut point its coefficient. The refits take the table's outcome and, under a surveyed
    population, its design (WP8, WP10), when the family's ``inference_matrix`` accepts them.
    """
    step = form_step(pipeline)
    if step is None or table is None:
        return [], []
    rows = list(table.rows)
    names = [str(r["feature"]) for r in rows]
    where = _index(names)
    estimates = np.array([np.nan if r["estimate"] is None else r["estimate"] for r in rows], float)
    cov = getattr(table, "cov", None)
    info = dict(table.info or {})
    tests: list[dict[str, Any]] = []
    concerns: list[str] = []
    if task == "multiclass" or (task == "ordinal" and not getattr(family, "ordered_levels", False)):
        return [], ["Tests of an exposure's form are not computed for a multinomial model: each "
                    "level has its own curve."]
    if info.get("refused"):
        return [], []
    held: dict[str, Any] = {}

    def joint(chosen: Sequence[int]) -> dict[str, Any] | None:
        """The joint test on the table's covariance; for a logistic model the data separate (a
        Firth table, profile intervals and no Wald covariance), the penalized likelihood-ratio
        test on the same design (the FORM repair: the test of association under separation)."""
        result = wald_test(estimates, cov, chosen, info, rows)
        if result is not None or info.get("covariance") != "profile":
            return result
        from turbotab.core.models.linear import model_matrix

        if "design" not in held:
            classes = list(getattr(pipeline[-1], "classes_", []))
            held["design"] = firth_design(model_matrix(pipeline, X), y, classes)
            held["fit"] = None
        found = held["design"]
        if found is None:
            return None
        design_names, design, event = found
        at = {n: i for i, n in enumerate(design_names)}
        try:
            picked = [at[names[i]] for i in chosen]
        except KeyError:
            return None
        if held["fit"] is None:
            from turbotab.core.models.inference import firth_fit

            held["fit"] = firth_fit(design, event)
        result = firth_lr_test(design, event, picked, held["fit"])
        if result is not None and not result.get("converged", True):
            concerns.append("A penalized fit behind a likelihood-ratio test stopped before "
                            "converging; treat that test with care.")
        return result

    plan = step._plan()
    for column, (form, _) in plan.items():
        outputs = step._outputs(column)
        if not all(o in where for o in outputs):
            continue
        if form in SPLINE_FORMS:
            idx = [where[o] for o in outputs]
            nonlinear = [where[o] for o in nonlinear_outputs(step, column)]
            knots = knot_sentence(column, step.knots_[column], step.knot_notes_[column])
            among = " among consumers" if form == "zero_spline" else ""
            for test_kind, chosen, what in (
                    ("overall", idx, f"all {len(idx)} terms of `{column}` are"),
                    ("nonlinear", nonlinear, f"the {len(nonlinear)} nonlinear "
                                             f"term{'s' if len(nonlinear) > 1 else ''} of "
                                             f"`{column}`{among} "
                                             f"{'are' if len(nonlinear) > 1 else 'is'}")):
                result = joint(chosen)
                if result is None:
                    if test_kind == "overall":
                        concerns.append(f"The spline tests for `{column}` need a Wald covariance, "
                                        f"which this table ({info.get('covariance')}) does not "
                                        f"give.")
                    continue
                lead = ("The test of association: " if test_kind == "overall" else "")
                tail = ("" if test_kind == "overall" else
                        f" A non-significant result does not refit a straight line: dropping the "
                        f"curve after its test inflates the test of association's type I error "
                        f"({GRAMBSCH}).")
                caption = (f"{lead}{joint_test_name(result)} that {what} zero "
                           f"({result['basis']}): {_statistic_words(result)}. "
                           f"{knots[0].upper()}{knots[1:]}.{tail}")
                tests.append(_test(column, form, test_kind, result,
                                   knots=[float(v) for v in step.knots_[column]],
                                   caption=caption))
            if column in step.companion_cuts_:
                found, worry = _companion(family, pipeline, step, column, X, y, task=task,
                                          clusters=clusters, outcome=outcome, survey=survey)
                tests.extend(found)
                concerns.extend(worry)
            continue
        idx = [where[o] for o in outputs]
        if form in ("quintiles", "categories"):
            # MS2: the global test that every indicator is zero, a multi-df Wald test on the
            # table's own covariance (pooled by D1 under multiple imputation).
            result = joint(idx)
            word = "quintile" if form == "quintiles" else "category"
            if result is not None:
                tests.append(_test(
                    column, form, "global", result,
                    medians=[float(v) for v in step.medians_[column]],
                    boundaries=[float(v) for v in step.cuts_[column]],
                    reference=f"{word} 1, `{column}` ≤ {step.cuts_[column][0]:.4g}",
                    caption=(f"{joint_test_name(result)} that all {len(idx)} {word} indicators "
                             f"of `{column}` are zero ({result['basis']}): "
                             f"{_statistic_words(result)}. "
                             f"{boundaries_words(column, step.cuts_[column], word).capitalize()}.")))
            if form == "categories":
                continue
            trend = _trend(family, pipeline, step, column, X, y, task=task, clusters=clusters,
                           outcome=outcome, survey=survey)
            if trend is None:
                concerns.append(f"The trend test for `{column}` could not be computed.")
                continue
            tests.append(trend)
            continue
        # A data-derived cut point: its one coefficient, with the search it came from.
        row = rows[idx[0]]
        found = step.optimal_.get(column) or {}
        tests.append(_test(
            column, form, "cut", None, statistic=(row["estimate"] / row["se"]) if row.get("se")
            else None, df_num=1, df_den=row.get("df"),
            distribution="t" if row.get("df") is not None else "z", p=row.get("p"),
            estimate=row.get("estimate"), ci_low=row.get("ci_low"), ci_high=row.get("ci_high"),
            boundaries=[float(step.cuts_[column][0])],
            reference=f"`{column}` ≤ {step.cuts_[column][0]:.4g}",
            caption=(f"`{column}` above {step.cuts_[column][0]:.4g} against below it: a cut point "
                     f"the data chose (the smallest of {found.get('tried')} p-values on these "
                     f"rows), so its p-value and interval are too small: data-derived cut points "
                     f"lead to serious bias ({ALTMAN}).")))
    return tests, concerns


def _refit(family: Any, matrix: pd.DataFrame, y: Any, *, pipeline: Any, task: str, clusters: Any,
           outcome: Any, survey: Any) -> Any:
    """The family's inference table on ``matrix``, with the table's outcome and design."""
    import inspect

    refit = getattr(family, "inference_matrix", None)
    if refit is None:
        return None
    classes = list(getattr(pipeline[-1], "classes_", [])) or None
    accepts = inspect.signature(refit).parameters
    extra = {name: value for name, value in (("outcome", outcome), ("survey", survey))
             if value is not None and name in accepts}
    if survey is not None and "survey" not in extra:
        return None  # a refit without the design beside a design-based table would mislead
    return refit(matrix, y, task=task, classes=classes, clusters=clusters, **extra)


def _trend(family: Any, pipeline: Any, step: ExposureForms, column: str, X: pd.DataFrame, y: Any,
           *, task: str, clusters: Any, outcome: Any = None,
           survey: Any = None) -> dict[str, Any] | None:
    """The trend test on quintile medians: the model matrix with ``column``'s indicators replaced
    by one column, each row's quintile median, refit by the family's own inference table."""
    from turbotab.core.models.inference import format_p
    from turbotab.core.models.linear import model_matrix

    matrix = model_matrix(pipeline, X)
    indicators = quintile_names(column)
    if not all(c in matrix.columns for c in indicators):
        return None
    raw = _before_form(pipeline, X)[column].to_numpy(dtype=float, na_value=np.nan)
    score_name = f"{column}{SCORE_SUFFIX}"
    at = list(matrix.columns).index(indicators[0])
    kept = matrix.drop(columns=indicators)
    kept.insert(at, score_name, step.scores(column, raw))
    table = _refit(family, kept, y, pipeline=pipeline, task=task, clusters=clusters,
                   outcome=outcome, survey=survey)
    if table is None:
        return None
    row = next((r for r in table.rows if str(r["feature"]) == score_name), None)
    if row is None or row.get("p") is None:
        return None
    medians = ", ".join(f"{m:.4g}" for m in step.medians_[column])
    reference = "t" if row.get("df") is not None else "z"
    caption = (f"The {TREND_LABEL} across quintiles of `{column}`: each row scored by its "
               f"quintile's median ({medians}, from the fitting rows), entered as one continuous "
               f"term in place of the four indicators; its coefficient {row['estimate']:+.4g} per "
               f"unit, p = {format_p(row['p'])} ({table.info.get('caption', '').rstrip('.')}). A "
               f"test of a linear trend, not of a dose–response.")
    if survey is not None:
        # The review of the modeling sequence (§1 step 5, quintiles): "Under survey design, state
        # whether cut points are weighted."
        caption += (" The cut points and medians are the analyzed rows' own, unweighted; the "
                    "trend's coefficient and interval are design-based.")
    return _test(column, "quintiles", "trend", None,
                 statistic=(row["estimate"] / row["se"]) if row.get("se") else None,
                 df_num=1, df_den=row.get("df"), distribution=reference, p=row["p"],
                 estimate=row["estimate"], ci_low=row.get("ci_low"), ci_high=row.get("ci_high"),
                 medians=[float(m) for m in step.medians_[column]],
                 boundaries=[float(v) for v in step.cuts_[column]], label=TREND_LABEL,
                 reference=f"quintile 1, `{column}` ≤ {step.cuts_[column][0]:.4g}",
                 caption=caption)


def _companion(family: Any, pipeline: Any, step: ExposureForms, column: str, X: pd.DataFrame,
               y: Any, *, task: str, clusters: Any, outcome: Any = None,
               survey: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
    """The quintile table beside a declared exposure's spline (MODELING_SEQUENCE §0 ruling 2): the
    model refit with ``column``'s spline terms replaced by quintile indicators (each quintile
    against the lowest, the boundaries and the reference stated), and by each row's quintile
    median (the {TREND_LABEL}). The cut points are the companion's own, placed on the fitting
    rows (on the observed values under multiple imputation, fixed across copies); the medians are
    each fit's."""
    from turbotab.core.models.inference import format_p
    from turbotab.core.models.linear import model_matrix

    matrix = model_matrix(pipeline, X)
    terms = step._outputs(column)
    if not all(c in matrix.columns for c in terms):
        return [], []
    raw = _before_form(pipeline, X)[column].to_numpy(dtype=float, na_value=np.nan)
    cuts = step.companion_cuts_[column]
    group = quantile_group(raw, cuts)
    counts = np.bincount(group[group >= 0], minlength=QUINTILES)[:QUINTILES]
    if (counts == 0).any() or len(np.unique(cuts)) < QUINTILES - 1:
        return [], [f"The quintiles beside `{column}`'s spline could not be formed: its values are "
                    f"too heavily tied."]
    medians = np.array([float(np.median(raw[group == g])) for g in range(QUINTILES)])
    at = list(matrix.columns).index(terms[0])
    kept = matrix.drop(columns=terms)
    indicators = quintile_names(column)
    for j, name in enumerate(indicators, start=1):
        kept.insert(at + j - 1, name, np.where(group < 0, np.nan, (group == j).astype(float)))
    out: list[dict[str, Any]] = []
    reference = f"quintile 1, `{column}` ≤ {cuts[0]:.4g}"
    bounds = boundaries_words(column, cuts)
    # MS4 (the review, §1 step 5, quintiles): "Under survey design, state whether cut points are
    # weighted."
    design = (" The cut points and medians are the analyzed rows' own, unweighted; the coefficients "
              "and intervals are design-based, over the survey design." if survey is not None
              else "")
    table = _refit(family, kept, y, pipeline=pipeline, task=task, clusters=clusters,
                   outcome=outcome, survey=survey)
    if table is None:
        return [], [f"The quintiles beside `{column}`'s spline could not be refit."]
    by = {str(r["feature"]): r for r in table.rows}
    for g, name in enumerate(indicators, start=2):
        row = by.get(name)
        if row is None or row.get("estimate") is None:
            continue
        ratio = (f" (ratio {row['ratio']:.3g}, {row['ratio_low']:.3g} to {row['ratio_high']:.3g})"
                 if row.get("ratio") is not None and row.get("ratio_low") is not None else "")
        out.append(_test(
            column, "quintiles", f"companion_q{g}", None,
            statistic=(row["estimate"] / row["se"]) if row.get("se") else None, df_num=1,
            df_den=row.get("df"), distribution="t" if row.get("df") is not None else "z",
            p=row.get("p"), estimate=row["estimate"], ci_low=row.get("ci_low"),
            ci_high=row.get("ci_high"), medians=[float(m) for m in medians],
            boundaries=[float(c) for c in cuts], reference=reference,
            what=f"quintile {g} of `{column}` against the lowest",
            caption=(f"Beside the spline: quintile {g} of `{column}` against the lowest, "
                     f"{row['estimate']:+.4g}{ratio}, p = {format_p(row.get('p'))}; {bounds}."
                     f"{design}")))
    score_name = f"{column}{SCORE_SUFFIX}"
    scored = matrix.drop(columns=terms)
    scored.insert(at, score_name, np.where(group < 0, np.nan,
                                           medians[np.clip(group, 0, QUINTILES - 1)]))
    trend_table = _refit(family, scored, y, pipeline=pipeline, task=task, clusters=clusters,
                         outcome=outcome, survey=survey)
    row = next((r for r in (trend_table.rows if trend_table is not None else [])
                if str(r["feature"]) == score_name), None)
    if row is not None and row.get("p") is not None:
        listed = ", ".join(f"{m:.4g}" for m in medians)
        out.append(_test(
            column, "quintiles", "companion_trend", None,
            statistic=(row["estimate"] / row["se"]) if row.get("se") else None, df_num=1,
            df_den=row.get("df"), distribution="t" if row.get("df") is not None else "z",
            p=row["p"], estimate=row["estimate"], ci_low=row.get("ci_low"),
            ci_high=row.get("ci_high"), medians=[float(m) for m in medians],
            boundaries=[float(c) for c in cuts], label=TREND_LABEL, reference=reference,
            what=f"the {TREND_LABEL} coefficient per unit of `{column}`",
            caption=(f"Beside the spline, the {TREND_LABEL}: each row scored by its quintile's "
                     f"median ({listed}), one term in place of the spline; its coefficient "
                     f"{row['estimate']:+.4g} per unit, p = {format_p(row['p'])}. The spline's "
                     f"overall test is the test of association; this is a test of a linear trend, "
                     f"not of a dose–response.{design}")))
    return out, []


def describe(forms: Mapping[str, Any]) -> str:
    """The pipeline step's detail line: which column takes which form, and where its knots or cut
    points come from."""
    parts = []
    learned = []
    for column, spec in forms.items():
        form, knots = _form_of(spec)
        if form == "spline":
            parts.append(f"a restricted cubic spline of {column} with {knots or DEFAULT_KNOTS} "
                         f"knots at Harrell's percentiles")
            if "knots" not in learned:
                learned.append("knots")
        elif form == "zero_spline":
            parts.append(f"non-consumers of {column} as their own category beside a spline among "
                         f"consumers with {knots or DEFAULT_KNOTS} knots at consumers' percentiles")
            if "knots" not in learned:
                learned.append("knots")
        elif form == "quintiles":
            parts.append(f"quintile indicators of {column} against its lowest fifth")
            if "cut points" not in learned:
                learned.append("cut points")
        elif form == "categories":
            cuts = ", ".join(f"{c:g}" for c in (_field(spec, "cuts") or []))
            parts.append(f"indicators of {column} at the declared cut points {cuts}, against its "
                         f"lowest group")
        elif form == "optimal":
            parts.append(f"{column} split at the cut point with the smallest outcome p-value")
            if "data-derived cut points" not in learned:
                learned.append("data-derived cut points")
    if not parts:
        return "Every exposure enters as a straight line."
    joined = parts[0] if len(parts) == 1 else f"{'; '.join(parts[:-1])}; and {parts[-1]}"
    tail = (f" The {' and '.join(learned)} are learned on the rows each fit sees, every training "
            f"fold included." if learned else "")
    return f"{joined[0].upper()}{joined[1:]}.{tail}"


# ── the record: the sentence a declaration writes ────────────────────────────


def form_sentence(column: str, spec: Any, state: Any, task: str | None = None) -> str:
    """How one predictor entered the models, as the record says it (the rule for k stated; the
    scale it is on; a mass at zero's two parts; a consumers-only domain as an estimand change;
    quintiles' trend labeled customary; a coarse or data-derived cut recorded as a limitation; a
    curve on a residual labeled as what it is)."""
    from turbotab.core.voice import tick

    form, k = _form_of(spec)
    purpose = getattr(state, "purpose", None)
    tested = purpose == "inference"
    scale = transform_signature(state, column)
    values = "values" if scale == "raw" else (
        "energy-adjusted values" if scale.startswith("energy:") or ";energy:" in scale
        else "values on its transformed scale")
    adj = getattr(state, "energy_adjustment", None)
    previous = _forms_of(state).get(column)
    was = _form_of(previous)[0] if previous is not None else None
    text: str
    if form == "linear":
        text = f"{tick(column)} entered the models as a straight line"
        if was in ("spline", "zero_spline") and tested:
            text += (f", in place of the restricted cubic spline declared before; dropping a "
                     f"curve after its nonlinearity test inflates the test of association's type "
                     f"I error ({GRAMBSCH})")
    elif form in SPLINE_FORMS:
        k = k or DEFAULT_KNOTS
        rule = (f", {rule_words(k, _field(spec, 'n_effective'), task or getattr(state, 'task', None))},"
                if _field(spec, "knots_rule") == "harrell" else "")
        pct = [f"{100 * p:g}" for p in KNOT_PERCENTILES.get(k, ())]
        if form == "zero_spline":
            where = (f" at the {', '.join(pct[:-1])} and {pct[-1]} percentiles of consumers' "
                     f"{values} (`{column}` above 0) in the rows each model was fit on" if pct else "")
            text = (f"Non-consumers of {tick(column)} (`{column}` = 0) entered the models as their "
                    f"own category, the reference, beside a restricted cubic spline among "
                    f"consumers with {tick(k)} knots{rule}{where} (STROBE-nut nut-11)")
        else:
            where = (f" at the {', '.join(pct[:-1])} and {pct[-1]} percentiles of its {values} in "
                     f"the rows each model was fit on (Harrell's placement)" if pct
                     else f" placed on its {values}")
            text = (f"{tick(column)} entered the models as a restricted cubic spline with {tick(k)} "
                    f"knots{rule}{where}")
        if tested:
            text += ("; the test of association is the Wald test that every term is zero, and "
                     "nonlinearity was tested by a Wald test that its nonlinear terms are zero, "
                     "a non-significant result never refitting a straight line")
            if form == "spline" and column == exposure_of(state):
                text += (f"; quintiles were reported beside it, their boundaries and reference "
                         f"stated, with the {TREND_LABEL} across quintile medians")
    elif form == "quintiles":
        trend = (f"; the {TREND_LABEL} scored each quintile by its median, entered as one "
                 f"continuous term" if tested else "")
        text = (f"{tick(column)} entered the models as quintiles of its {values} in the rows each "
                f"model was fit on, the lowest the reference{trend}")
    elif form == "categories":
        cuts = _field(spec, "cuts") or []
        text = (f"{tick(column)} entered the models as {len(cuts) + 1} categories at the declared "
                f"cut points {', '.join(f'{c:g}' for c in cuts)}, the lowest the reference")
    else:
        text = (f"{tick(column)} entered the models split at a data-derived cut point, the one "
                f"with the smallest outcome p-value on the fitting rows")
    if _field(spec, "domain") == "consumers":
        text += (f"; the analysis was restricted to consumers of {tick(column)} (`{column}` above 0): "
                 f"an estimand change, the effect among consumers, not in the whole population "
                 f"({STROBE_NUT})")
    if _field(spec, "acknowledged"):
        if form == "optimal":
            text += (f", recorded as a limitation: data-derived cut points lead to serious bias "
                     f"({ALTMAN})")
        else:
            text += (f", recorded as a limitation: a confounder in {COARSE_GROUPS} or fewer groups "
                     f"leaves serious residual confounding ({BRENNER})")
    if residual_form(state, column, spec):
        text += f"; this is {residual_label(column, _field(adj, 'energy_column'))}"
    return text


# ── the leash: refusals, completions and the contract ─────────────────────────


def _state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state as state_of

    return state_of(ctx)


def _role_of(state: Any, column: str) -> str:
    return "exposure" if column == exposure_of(state) else "confounder"


def _values_of(ctx: Any, column: str) -> np.ndarray | None:
    """``column``'s recorded values on the analyzed rows, when the context can read them."""
    from turbotab.core.decisions import _ctx

    store_fn, analyzed = _ctx(ctx, "store"), _ctx(ctx, "analyzed")
    if not callable(store_fn):
        return None
    try:
        store = store_fn()
        if store is None or column not in store.columns:
            return None
        rows = analyzed() if callable(analyzed) else None
        frame = store.materialize([column], rows)
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    except Exception:  # noqa: BLE001 - a check that cannot read the values checks nothing
        return None


def _form_kind_fits(decision: Any, ctx: Any) -> None:
    """Cut points belong to categories (and categories need them); a mass at zero's form and the
    consumers-only domain need zeros and no negative values; a data-derived cut point searches a
    numeric or yes/no outcome on rows it can see, never across imputed copies; the consumers-only
    domain is an inference estimand, and the declared exposure's."""
    from turbotab.core.decisions import Refusal, SetExposureForm

    column = decision.column
    spline = {"label": "A restricted cubic spline",
              "decision": SetExposureForm(column=column, form="spline")}
    if decision.form == "categories" and not decision.cuts:
        raise Refusal("cuts_needed", f"Categories of `{column}` need their cut points declared, "
                                     f"from outside these data.",
                      exits=[spline, {"label": "Declare the cut points", "decision": None}])
    if decision.cuts and decision.form != "categories":
        raise Refusal("cuts_without_categories",
                      f"Cut points belong to categories; the {decision.form} form has none.",
                      exits=[{"label": f"{decision.form.replace('_', ' ').capitalize()} without "
                                       f"cut points",
                              "decision": decision.model_copy(update={"cuts": None})}])
    state = _state(ctx)
    purpose = getattr(state, "purpose", None) if state is not None else None
    if decision.domain == "consumers":
        if purpose == "prediction":
            raise Refusal("domain_for_inference",
                          "A consumers-only domain changes the population the estimate is about; "
                          "under prediction the model is used on everyone, consumers or not.",
                          exits=[{"label": "Non-consumers apart, a spline among consumers",
                                  "decision": SetExposureForm(column=column, form="zero_spline")}])
        if state is not None and exposure_of(state) not in (None, column):
            raise Refusal("domain_of_the_exposure",
                          f"The consumers-only domain is the declared exposure's "
                          f"(`{exposure_of(state)}`); restricting by `{column}` would change the "
                          f"population of another effect.",
                          exits=[{"label": f"Keep everyone; a form of `{column}`", "decision":
                                  decision.model_copy(update={"domain": "all"})}])
    # A consumers-only domain already in force leaves no zero among the analyzed rows: answering
    # it again (the form question's one-tap answer keeps it) is not a domain without consumers.
    kept = (decision.domain == "consumers" and decision.form != "zero_spline"
            and state is not None and column in domain_columns(state))
    if (decision.form == "zero_spline" or decision.domain == "consumers") and not kept:
        values = _values_of(ctx, column)
        if values is not None:
            share, negative = zero_share(values)
            if negative or share == 0:
                raise Refusal(
                    "no_mass_at_zero",
                    f"`{column}` has {'negative values' if negative else 'no value at zero'}, so "
                    f"there are no non-consumers to set apart.",
                    exits=[spline, {"label": "A straight line",
                                    "decision": SetExposureForm(column=column, form="linear")}])
    if decision.form == "optimal":
        task = getattr(state, "task", None) if state is not None else None
        if task not in (None, "regression", "binary"):
            raise Refusal("optimal_task",
                          f"A data-derived cut point is searched here for a numeric or a yes/no "
                          f"outcome only; this outcome is {task}.",
                          exits=[spline, {"label": "Quintiles",
                                          "decision": SetExposureForm(column=column,
                                                                      form="quintiles")}])
        missing = getattr(state, "missing", None) if state is not None else None
        if purpose == "inference" and missing is not None and \
                getattr(missing, "strategy", None) == "multiple_imputation":
            raise Refusal("optimal_under_imputation",
                          "A data-derived cut point would be searched again in every imputed copy, "
                          "and the copies' estimates would not be of one variable.",
                          exits=[spline, {"label": "Declare the cut points from outside these data",
                                          "decision": None}])


def _form_reads_a_settled_reading(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14.1–§14.2: a form is a number-changing consumer of whether the column's numbers
    are codes or amounts (codes enter as one indicator per level and take no form). A form of a
    column recorded as codes is refused, its way forward the reading's other answer; one whose
    whole numbers the ledger holds unsettled asks the reading first, never recording a form on a
    guess (the FORM repair: a spline recorded for a 1–12 band later confirmed as codes)."""
    from turbotab.core import readings as R
    from turbotab.core.decisions import Refusal, _ctx, _store_of

    state = _state(ctx)
    if state is None:
        return
    column = decision.column
    if column in _codes(state):
        raise Refusal(
            "codes_take_no_form",
            f"`{column}` is recorded as codes for categories: it enters the models as one "
            f"indicator per level, so it takes no form. If its numbers are amounts, say so first.",
            exits=[R.confirm_exit("code_or_count", column, "amount",
                                  f"`{column}` is a count or an amount (it then takes a form)")])
    if column in {sc for sc in (_field(s, "name") for s in getattr(state, "scales", None) or [])}:
        return  # a scale's score: an amount by the scales answer
    info = _ctx(ctx, "column_info")
    store = _store_of(ctx)
    if not info and store is None:
        return  # nothing reads the values here: the fit asks
    facts = R.whole_facts([column], info, store).get(column)
    reading = code_reading(state, column, facts)
    if reading is None:
        return
    readings = R.labeled([reading], state)
    raise Refusal(
        "reading_unsettled",
        f"`{column}`'s numbers may be codes for categories (one indicator per level, no form) "
        f"or amounts (a form); the form waits for the answer. "
        f"{R.ask_text(readings, state)}",
        exits=R.ask_exits(readings, state))


def _cuts_are_declared_or_recorded(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §4 under inference: a data-derived cut point, and a continuous confounder
    cut into three or fewer groups, are blocked and recorded: refused with their exits (the
    spline first), kept only with the acknowledgment the record states."""
    from turbotab.core.decisions import Refusal, SetExposureForm

    state = _state(ctx)
    if decision.acknowledged or state is None or getattr(state, "purpose", None) != "inference":
        return
    column = decision.column
    role = _role_of(state, column)
    spline = {"label": "A restricted cubic spline (recommended)",
              "decision": SetExposureForm(column=column, form="spline")}
    keep = {"label": "Keep it, recorded as a limitation",
            "decision": decision.model_copy(update={"acknowledged": True})}
    if decision.form == "optimal":
        raise Refusal(
            "optimal_cut_point",
            f"A cut point chosen where the outcome differs most on these rows is data-derived: "
            f"its estimate and p-value are biased away from the null (Altman & Royston 2006: "
            f"\"the use of a data-derived 'optimal' cutpoint leads to serious bias\").",
            exits=[spline, {"label": "Quintiles", "decision": SetExposureForm(column=column,
                                                                              form="quintiles")},
                   {"label": "Declare the cut points from outside these data", "decision": None},
                   keep])
    if decision.form == "categories" and role == "confounder" and \
            len(decision.cuts or []) + 1 <= COARSE_GROUPS:
        groups = len(decision.cuts or []) + 1
        raise Refusal(
            "coarse_confounder",
            f"`{column}` is a confounder cut into {groups} groups: within each group it still "
            f"varies with the exposure, so its confounding is only partly removed (Brenner & "
            f"Blettner 1997: \"categorization of the confounder may often lead to serious "
            f"residual confounding if the number of categories is small\").",
            exits=[spline, {"label": "A straight line",
                            "decision": SetExposureForm(column=column, form="linear")}, keep])


def _forms_each_fit(decision: Any, ctx: Any) -> None:
    """The one-tap answer: each column's form checked as its own ``set_exposure_form`` is; a
    refusal's exits keep the other columns' forms."""
    from turbotab.core import decisions as d

    for column, spec in decision.forms.items():
        probe = d.SetExposureForm(column=column, **spec.model_dump())
        try:
            for fn in d._VALIDATORS.get("set_exposure_form", ()):
                fn(probe, ctx)
        except d.Refusal as refused:
            exits = []
            for e in refused.exits:
                taken = e.get("decision")
                if taken and taken.get("kind") == "set_exposure_form" and \
                        taken.get("column") == column:
                    forms = {**{c: s.model_dump() for c, s in decision.forms.items()},
                             column: {k: v for k, v in taken.items() if k not in ("kind", "column")}}
                    exits.append({"label": e["label"],
                                  "decision": {"kind": "set_forms", "forms": forms}})
                else:
                    exits.append(e)
            raise d.Refusal(refused.code, f"`{column}`: {refused.message}", exits=exits) from None


def _n_effective(ctx: Any, state: Any) -> tuple[float | None, bool]:
    """The effective sample size the rule reads, and whether the context reads rows at all (the
    server's does): the fresh form card's (read on the analyzed rows' outcome); else read here
    from the outcome on the analyzed rows, once those rows are known (a fresh cohort or split)
    and the task is (answered, else the target stage's detected task). ``(None, True)`` while the
    rows are being read: never the whole table's rows in their place, nor a task left unread."""
    from turbotab.core.decisions import _ctx
    from turbotab.core.sequence import artifact

    card = artifact(ctx, "forms") or {}
    if card.get("n_effective") is not None:
        return float(card["n_effective"]), True
    target = getattr(state, "target", None)
    store_fn, analyzed = _ctx(ctx, "store"), _ctx(ctx, "analyzed")
    if not callable(store_fn):
        return None, False
    task = getattr(state, "task", None) or _task_of_ctx(ctx)
    rows = None
    try:
        rows = analyzed() if callable(analyzed) else None
    except Exception:  # noqa: BLE001 - rows that cannot be read are not known yet
        rows = None
    if not target or task is None or rows is None:
        return None, True
    try:
        store = store_fn()
        if store is None or target not in store.columns:
            return None, True
        raw = store.materialize([target], rows)[target]
    except Exception:  # noqa: BLE001 - rows that cannot be read are not known yet
        return None, True
    return effective_n(task, raw, getattr(state, "event", None)), True


def _rows_not_read(column: str, spec: Mapping[str, Any]) -> Any:
    """The rule cannot be applied while the analyzed rows (or the outcome's task) are being read:
    refused for now, with the spline at a k the user gives as the way forward."""
    from turbotab.core.decisions import Refusal, SetExposureForm

    base = {k: v for k, v in spec.items() if k not in ("knots", "knots_rule", "n_effective")}
    return Refusal(
        "not_yet",
        f"k by Harrell's rule reads the effective sample size of the analyzed rows, and they are "
        f"being read for the answers just given; declare `{column}`'s spline again in a moment, or "
        f"give its k.",
        exits=[{"label": f"A spline of `{column}` with {k} knots, k given",
                "decision": SetExposureForm(column=column, **{**base, "knots": k})}
               for k in KNOT_CHOICES])


def complete_spec(column: str, spec: Any, ctx: Any) -> dict[str, Any]:
    """A declaration as it is recorded: k by Harrell's rule when a spline gives none (the rule and
    the effective sample size it read kept with it), and the scale and the estimand's unit it was
    declared on. Under inference, while the context reads rows but they are not read yet, the
    rule waits (refused ``not_yet``) rather than state a k it did not compute."""
    state = _state(ctx)
    out = dict(spec)
    if state is None:
        return out
    form = out.get("form")
    if form in SPLINE_FORMS and (out.get("knots") is None or out.get("knots_rule") == "harrell"):
        # The rule's k, read here on the analyzed rows (a client's own k is kept, without the rule).
        n_eff, reads_rows = _n_effective(ctx, state)
        if n_eff is None and reads_rows and getattr(state, "purpose", None) == "inference":
            raise _rows_not_read(column, out)
        out.update(knots=knots_by_rule(n_eff), knots_rule="harrell", n_effective=n_eff)
    elif form not in SPLINE_FORMS:
        out.update(knots_rule=None, n_effective=None)
    out["scale"] = transform_signature(state, column)
    out["unit"] = scale_words(state, column)
    return out


def _form_records_its_scale(decision: Any, ctx: Any) -> Any:
    fields = complete_spec(decision.column, decision.model_dump(exclude={"kind", "column"}), ctx)
    return decision.model_copy(update=fields)


def _forms_record_their_scale(decision: Any, ctx: Any) -> Any:
    from turbotab.core.decisions import ExposureFormSpec, Refusal

    forms = {}
    for c, s in decision.forms.items():
        try:
            forms[c] = ExposureFormSpec(**complete_spec(c, s.model_dump(), ctx))
        except Refusal as refused:
            # The one-tap answer's way forward keeps the other columns' forms (as its validator's).
            exits = []
            for e in refused.exits:
                taken = e.get("decision")
                taken = taken.model_dump(mode="json") if hasattr(taken, "model_dump") else taken
                if taken and taken.get("kind") == "set_exposure_form":
                    every = {k: v.model_dump(mode="json") for k, v in decision.forms.items()}
                    every[c] = {k: v for k, v in taken.items() if k not in ("kind", "column")}
                    exits.append({"label": e["label"],
                                  "decision": {"kind": "set_forms", "forms": every}})
                else:
                    exits.append(e)
            raise Refusal(refused.code, refused.message, exits=exits) from None
    return decision.model_copy(update={"forms": forms})


def _share_reallocation_refused(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §0 ruling 11: a request to reallocate shares (compositional, ilr) is
    refused with that reason, and the kcal substitution is offered instead."""
    from turbotab.core.decisions import Refusal

    if getattr(decision, "scale", None) != "share_reallocation":
        return
    base = decision.model_dump(exclude={"kind"})
    raise Refusal(
        "share_reallocation_v2x", SHARE_REALLOCATION,
        exits=[{"label": "The kcal substitution", "decision": {
                    **base, "kind": "set_substitution", "scale": "kcal"}},
               {"label": "The share-of-energy swap", "decision": {
                    **base, "kind": "set_substitution", "scale": "percent_energy"}}])


def _task_of_ctx(ctx: Any) -> str | None:
    """The answered task, else the detected one (the server's ``DecisionContext.task``, or its
    sentence facts' ``detected_task``)."""
    from turbotab.core.decisions import _ctx

    task = _ctx(ctx, "task") or _ctx(ctx, "detected_task")
    return str(task) if task else None


def _sentence(d: Any, state: Any, ctx: Any) -> str:
    text = form_sentence(d.column, d, state, _task_of_ctx(ctx))
    standing = _forms_standing(d, state, ctx)
    return f"{text}. {standing}" if standing else text


def _forms_sentence(d: Any, state: Any, ctx: Any) -> str:
    parts = [form_sentence(c, s, state, _task_of_ctx(ctx)) for c, s in d.forms.items()]
    text = "; ".join(p[0].lower() + p[1:] if i and p[:1] != "`" else p
                     for i, p in enumerate(parts))
    standing = _forms_standing(d, state, ctx)
    return f"{text}. {standing}" if standing else text


def _forms_standing(d: Any, state: Any, ctx: Any) -> str | None:
    """The declaration's standing clause (``voice.register_standing``): the columns it gave a form
    that the user has since recorded as codes for categories, which enter as one indicator per
    level and take no form. The record keeps the sentence as said; the methods text, which says
    what the analysis is now, ends it with this (BLUEPRINT §14.3: every confirmation is honored)."""
    from turbotab.core.voice import listing

    columns = [d.column] if getattr(d, "column", None) else list(getattr(d, "forms", {}) or {})
    coded = [c for c in columns if c in _codes(state)]
    if not coded:
        return None
    one = len(coded) == 1
    return (f"{listing(coded)} {'was' if one else 'were'} then recorded as codes for categories, "
            f"so {'it' if one else 'each'} entered the models as one indicator per level, not in "
            f"the form declared here")


# ── the method contract (BLUEPRINT §13) ──────────────────────────────────────

PACKAGE = "FORM"
_SENTENCE = "turbotab.core.methods.exposure_form:form_sentence"


def _relation(id: str, kind: str, target: str, condition: str, says: str, *,
              rung: str | None = None, exits: Sequence[str] = (), enforced_by: str = "",
              purposes: tuple[str, ...] = ("prediction", "inference"),
              when: tuple[str, ...] = ()) -> Any:
    from turbotab.core.contracts import Relation

    return Relation(kind, target, says, purposes=purposes, rung=rung, when=when,
                    exits=tuple(exits), enforced_by=enforced_by, condition=condition, id=id)


def _contract_option(key: str, label: str, customary: str, inference: str, prediction: str,
                     rung_i: str, rung_p: str) -> Any:
    from turbotab.core.contracts import ContractOption

    return ContractOption(key, label, customary, sound={"inference": inference,
                                                        "prediction": prediction},
                          rung={"inference": rung_i, "prediction": rung_p})


def _register_contracts() -> None:
    from turbotab.core.contracts import MethodContract, register_contract

    me = "turbotab.core.methods.exposure_form"
    register_contract(MethodContract(
        key="exposure_transform", package=PACKAGE,
        label="A domain transform of the exposure (log, energy model, scale scoring, omics "
              "normalization)",
        slot="in_fold", scope="training_fold",
        scope_note="the energy residual and the omics normalizations learn from study rows; "
                   "scoring by a published key is row-local, but the transform's place in the "
                   "chain is what this contract states",
        needs=("a declared exposure",), question="energy_adjustment",
        place="MODELING_SEQUENCE §1 row 4", run_order=1.0,
        storyboard=("the exposure as recorded", "its transform", "its final scale"),
        sentence=f"{me}:scale_words",
        options=(
            _contract_option("residual", "The energy residual",
                             "customary (Willett's residual method)",
                             "Sound for the energy-adjusted scale; a curve on it is not the "
                             "substitution curve", "Sound: the choice matters little for "
                             "prediction", "available", "available"),
            _contract_option("log", "A log scale", "customary for skewed intakes",
                             "Sound when the effect is multiplicative; the curve is then "
                             "k-specific", "Sound in-fold", "available", "available"),
            _contract_option("score", "A scale score", "customary (a sum or mean of items)",
                             "Sound with its reliability stated", "Sound", "available",
                             "available"),
        ),
        relations=(
            _relation("transform-invalidates-form", "invalidates", "functional_form",
                      "a declared form, then any change of the exposure's transform",
                      "the form, its knots and cut points and the estimand's unit are asked "
                      "again on the new scale, never kept",
                      enforced_by=f"{me}:current_forms"),
            _relation("transform-precedes-form", "precedes", "functional_form", "always",
                      "the form is declared on the exposure's final scale",
                      enforced_by="turbotab.core.interview:QUESTION_KEYS"),
        )))
    register_contract(MethodContract(
        key="functional_form", package=PACKAGE,
        label="The functional form of a continuous exposure or confounder",
        slot="in_fold", scope="training_fold",
        option_scopes={"optimal": "model"},
        scope_note="knots, cut points and quintile medians are placed on the rows each fit sees "
                   "(every analyzed row under inference; the training fold under prediction); a "
                   "data-derived cut point also reads the outcome",
        needs=("a continuous exposure or confounder",), question="form",
        place="MODELING_SEQUENCE §1 row 5", decision="set_exposure_form", run_order=2.0,
        leash={"inference": "recommended", "prediction": "available"},
        storyboard=("the exposure on its final scale", "the knots at Harrell's percentiles",
                    "the curve and its two tests", "the quintiles beside it"),
        sentence=_SENTENCE,
        options=(
            _contract_option("spline", "Restricted cubic spline, k by Harrell's rule",
                             "increasingly customary, \"now near-default\" (NUTRITION_PACK §07G)",
                             "Sound: declared before estimates; the overall test is the test of "
                             "association", "Sound: nests the line", "recommended",
                             "recommended"),
            _contract_option("zero_spline", "Non-consumers apart, a spline among consumers",
                             "STROBE-nut nut-11 asks how non-consumers were handled",
                             "Sound for a mass at zero", "Sound for a mass at zero",
                             "recommended", "recommended"),
            _contract_option("linear", "Straight line", "customary",
                             "Sound when the relation is close to a line",
                             "Sound when close to a line", "available", "available"),
            _contract_option("quintiles", "Quintiles, beside the spline",
                             "customary: the field's primary-analysis convention (Turner 2010)",
                             "A coarser contrast, not a false number: ranked lower, produced "
                             "beside the spline", "Discards the variation within each fifth",
                             "rank_lower", "rank_lower"),
            _contract_option("categories", "Categories at declared cut points",
                             "customary for clinical bands",
                             "Coarser; a confounder in three or fewer groups is blocked and "
                             "recorded", "Coarser", "rank_lower", "rank_lower"),
            _contract_option("optimal", "A data-derived cut point",
                             "seen in clinical papers (the minimum p-value)",
                             "Unsound: serious bias (Altman & Royston 2006)",
                             "In-fold only; discards the variation on each side",
                             "block_and_record", "rank_lower"),
        ),
        relations=(
            _relation("k-by-rule", "implies", "the knots", "a spline declared without k",
                      "k is Harrell's rule on the effective sample size, the rule stated in the "
                      "record", enforced_by=f"{me}:complete_spec"),
            _relation("rows-rederive-k", "invalidates", "the knots",
                      "a spline whose k the rule set, then a change of the analyzed rows (an "
                      "exclusion, the complete cases, the consumers-only domain)",
                      "k and the effective sample size the record states are re-derived on the "
                      "rows the fit sees: the form question asks again, never keeping them",
                      purposes=("inference",), enforced_by=f"{me}:rule_stale"),
            _relation("codes-take-no-form", "conflicts", "a form of a column read as codes",
                      "a whole-numbered column whose code-or-amount reading is unsettled, or "
                      "recorded as codes",
                      "the reading is asked first (the card proposes no form on a guess); a "
                      "column recorded as codes enters as indicators and takes no form",
                      rung="refused", exits=("confirm it holds amounts", "confirm it holds codes"),
                      enforced_by=f"{me}:_form_reads_a_settled_reading"),
            _relation("separation-plr", "implies", "penalized likelihood-ratio tests",
                      "a logistic model the data separate (Firth's penalized likelihood)",
                      "the test of association, the nonlinearity and the global tests are "
                      "penalized likelihood-ratio tests on the full design, as the table's own "
                      "p-values are", purposes=("inference",), enforced_by=f"{me}:firth_lr_test"),
            _relation("no-silent-linear-refit", "disables", "a linear refit after the nonlinearity "
                      "test", "a non-significant nonlinearity test",
                      "the overall test stays the test of association; a switch is recorded "
                      "after the estimates were seen", purposes=("inference",),
                      enforced_by=f"{me}:exposure_tests"),
            _relation("quintiles-beside", "implies", "the quintile table",
                      "the declared exposure's spline under inference",
                      "the quintiles are produced beside it, boundaries and reference stated, the "
                      "p for linear trend (customary) from category medians",
                      purposes=("inference",), enforced_by=f"{me}:design_forms"),
            _relation("form-mi-d1", "implies", "D1 pooling", "multiple imputation",
                      "the overall, nonlinearity and global tests are pooled by D1 at knots fixed "
                      "on the observed values", purposes=("inference",),
                      enforced_by="turbotab.core.stages.modeling:_pool_form_tests"),
            _relation("optimal-cut-blocked", "conflicts", "a data-derived cut point",
                      "inference", "blocked and recorded (serious bias)", rung="block_and_record",
                      exits=("a restricted cubic spline", "quintiles",
                             "cut points declared from outside these data",
                             "keep it, recorded as a limitation"),
                      purposes=("inference",), enforced_by=f"{me}:_cuts_are_declared_or_recorded"),
            _relation("coarse-confounder-blocked", "conflicts", "a confounder in three or fewer "
                      "groups", "inference", "blocked and recorded (residual confounding)",
                      rung="block_and_record",
                      exits=("a restricted cubic spline", "a straight line",
                             "keep it, recorded as a limitation"),
                      purposes=("inference",), enforced_by=f"{me}:_cuts_are_declared_or_recorded"),
            _relation("mass-at-zero", "enables", "non-consumers apart",
                      "an exposure with a mass at zero",
                      "non-consumers as their own category beside a spline among consumers, or "
                      "the consumers-only domain as an estimand change (nut-14)",
                      enforced_by=f"{me}:forms_card"),
            _relation("consumers-only-domain", "implies", "the participant flow",
                      "the consumers-only domain", "the non-consumers leave on a line of their "
                      "own, and the caption names the population", purposes=("inference",),
                      enforced_by=f"{me}:domain_columns"),
            _relation("residual-curve-label", "implies", "the curve's label",
                      "a spline or categories on an energy residual",
                      "the nutrient's curve on the energy-adjusted scale at mean energy, with "
                      "spline(N) + E offered as the substitution route",
                      enforced_by=f"{me}:residual_label"),
            _relation("log-or-spline-substitution", "implies", "the substitution curve's label",
                      "a log or spline on an energy component",
                      "the curve is a k-specific average over the analyzed rows",
                      enforced_by=f"{me}:substitution_curve_label"),
            _relation("share-reallocation-refused", "conflicts", "a share reallocation",
                      "a request to reallocate shares (ilr)",
                      "refused: compositional models go to v2.x", rung="refused",
                      exits=("the kcal substitution", "the share-of-energy swap"),
                      enforced_by=f"{me}:_share_reallocation_refused"),
        ),
        sources=(HARRELL_RMS, GRAMBSCH, ALTMAN, BRENNER, "Desquilbet & Mariotti 2010, Stat Med "
                 "29:1037", "Lachat et al. 2016, PLoS Med 13:e1002036 (STROBE-nut)",
                 "Tomova et al. 2022, Am J Clin Nutr 115:189")))


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator
    from turbotab.core.voice import register_sentence, register_standing

    register_validator("set_exposure_form", _form_reads_a_settled_reading)
    register_validator("set_exposure_form", _form_kind_fits)
    register_validator("set_exposure_form", _cuts_are_declared_or_recorded)
    register_validator("set_forms", _forms_each_fit)
    register_validator("set_substitution", _share_reallocation_refused, first=True)
    register_completion("set_exposure_form", _form_records_its_scale)
    register_completion("set_forms", _forms_record_their_scale)
    register_sentence("set_exposure_form")(_sentence)
    register_sentence("set_forms")(_forms_sentence)
    register_standing("set_exposure_form")(_forms_standing)
    register_standing("set_forms")(_forms_standing)
    _register_ask()
    _register_contracts()


def _form_readings(ctx: Any) -> list[Any]:
    """The form question's ask card (``turbotab.core.ask``): the code-or-amount readings of the
    columns its card waits on, each with its guess, its evidence and its answers."""
    from turbotab.core import readings as R

    card = ctx.artifact("forms") or {}
    waiting = [str(w["column"]) for w in card.get("waiting") or []]
    if not waiting:
        return []
    facts = R.whole_facts(waiting, ctx.info(), ctx.table())
    found = [code_reading(ctx.state, c, facts.get(c)) for c in waiting]
    return [r for r in found if r is not None]


def _register_ask() -> None:
    from turbotab.core.ask import CONSUMERS

    CONSUMERS["form"] = ("the functional form", _form_readings)


_register()

__all__ = [
    "ABOVE_SUFFIX", "CONSUMER_SUFFIX", "CONTINUOUS", "DEFAULT_KNOTS", "ExposureForms", "FIRTH_PLR",
    "FORMS", "code_reading", "firth_design", "firth_lr_test", "form_candidates", "form_plan",
    "joint_test_name", "rule_stale", "standing_forms",
    "FORMS_READS", "FRACTIED", "Form", "HARRELL_LARGE", "HARRELL_SMALL", "KNOT_CHOICES",
    "KNOT_PERCENTILES", "MASS_AT_ZERO", "QUINTILES", "SHARE_REALLOCATION", "SPLINE_MIN_VALUES",
    "TREND_LABEL",
    "adjusted_forms", "boundaries_words", "category_names", "complete_spec", "current_forms",
    "describe", "design_forms", "domain_columns", "effective_n", "estimand_unit",
    "exposure_tests", "form_answer", "form_columns", "form_gate", "form_needs", "form_sentence",
    "form_step", "formed_meanings", "formed_name", "forms_card", "forms_stage", "knot_sentence",
    "knots_by_rule", "model_terms", "nonlinear_outputs", "optimal_cut", "options", "place",
    "quantile_cuts", "quantile_group", "quintile_names", "rcs_basis", "rcs_knots",
    "residual_form", "residual_label", "rule_words", "scale_words", "spline_names",
    "stale_forms", "substitution_curve_label", "substitution_route", "transform_signature",
    "unanswered_forms", "wald_test",
]
