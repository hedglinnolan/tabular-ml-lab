"""Case-control samples and matched sets, as an engine method core (SIZING E4).

A case-control sample draws the people with the outcome (the cases) and the people without it (the
controls) separately, so how many of each the table holds is set by the sampling, not by how common
the outcome is. Three samplings are named here, each analyzed its own way:

* **unmatched**: cases and controls drawn separately, with nothing matched;
* **frequency-matched**: controls drawn so that their distribution of a few matching factors (an
  age band, sex) follows the cases';
* **individually matched**: each case has its own controls, chosen to share its matching factors;
  the table carries a set identifier, and each set holds its case (or cases) and their controls.

**Odds ratios** (:func:`odds_ratios`). For an unmatched or frequency-matched sample, logistic
regression of being a case on the exposures, always adjusted for the matching factors (Breslow &
Day 1980, ch. 6). Its slopes are the population's log odds ratios, because outcome-dependent
sampling moves only the intercept (Prentice & Pyke 1979); the intercept is set by the sampling and
is not reported. The matching factors' own odds ratios are not estimated: the matching fixed their
distribution among the controls by design. Wald intervals on the normal, as R's ``glm`` with
``confint.default`` computes them.

For individually matched sets, conditional logistic regression (Breslow & Day 1980, ch. 7): the
likelihood of each set's cases being the cases, given the set and how many cases it holds, which
removes every set's own intercept. For one case per set it is the stratified Cox partial likelihood
with one event per stratum; with several cases in a set it is the exact conditional likelihood,
summed over every way of choosing that many members, computed by the recursion of Gail, Lubin &
Rubinstein (1981), as R's ``survival::clogit`` computes it (``method = "exact"``). A set with no
case or no control adds nothing and leaves. A column that is the same for every member of every set
(a matching factor) has no odds ratio here and is refused with the exit of leaving it out.

**Survey designs.** An unmatched or frequency-matched sample drawn inside a complex survey takes the
design-based logistic regression of :mod:`turbotab.core.models.survey` (R's ``svyglm`` with a
quasibinomial family: t intervals on the PSUs minus the strata). Matched sets under the population
answer are refused: conditional logistic regression has no design-based form in common use. The
exits: estimate for these participants (the sample-only attestation), or, with every matching
factor named, analyze the sample as frequency-matched (Pearce 2016).

**What is refused, everywhere.** The share of the rows with the outcome is not a prevalence and no
row's fitted probability is a risk: both are set by how many cases and controls were drawn
(Rothman, Greenland & Lash, Modern Epidemiology, ch. 8). Prevalence, absolute risk and a risk
difference are refused under every goal, each with its reason and exits (:func:`eligible`).

**Under Predict** (:class:`PriorCorrected`, :func:`predicted_risks`). A model trained on an
unmatched case-control sample is recalibrated to a stated population prevalence τ by the prior
correction (King & Zeng 2001, eq. 7): its log odds are shifted by
``log[(ȳ/(1 − ȳ)) · ((1 − τ)/τ)]``, ȳ the case share of the rows it was trained on; for a logistic
model this is the intercept β̂₀ − ln[((1 − τ)/τ)(ȳ/(1 − ȳ))]. ȳ is learned from the training fold
alone, so the correction is fitted inside each training fold. A frequency-matched sample needs the
prevalence within each matching stratum (the controls' sampling fraction differs by stratum), and
the matching factor among the model's features. Absolute risk for individually matched sets is
refused for a methods reason, not a missing feature: the conditional likelihood removes every set's
intercept, so the model has no baseline risk to recalibrate, and the controls were chosen to
resemble their case, so no single offset turns the sample's odds into the population's (Janes &
Pepe 2008). Its exit ranks each case against its own controls (:func:`matched_set_scores`), the
conditional model fitted in each training fold. Matched sets stay whole inside folds
(:func:`case_control_folds`); sets that share a person (incidence-density sampling) are kept
together too.

**Detection helpers** (no routing): :func:`prevalence_signal` (the outcome share far above a stated
population rate) and :func:`matched_sets_signal` / :func:`set_column_signals` (a column whose sets
each hold exactly one case) say a table *may* be a case-control sample, for the noticings to use.

The reference tests (``turbotab/core/tests/acceptance/test_case_control.py``) hold every number to
R: ``survival::clogit`` (with its own worked example, the ``infert`` matched study), ``glm``,
``survey::svyglm`` and ``binom.test``, and the prior correction to King & Zeng's equation by hand.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone

SAMPLINGS: tuple[str, ...] = ("unmatched", "frequency_matched", "individually_matched")
GOALS: tuple[str, ...] = ("describe", "estimate", "predict")
ASKS: tuple[str, ...] = ("odds_ratios", "prevalence", "absolute_risk", "predicted_risk", "ranking",
                         "within_set_ranking")
DESCRIBE = "describe"  # not yet one of the engine's purposes (D1 adds it)
LEVEL = 0.95
Z = float(stats.norm.ppf(0.5 + LEVEL / 2))
_TOL = 1e-11
_MAX_ITER = 100
# A log odds ratio this large per standard deviation of its column is an odds ratio no study
# estimates: the fit is running off to infinity (checked by the separation test first).
_HUGE = 30.0

SOURCES = {
    "breslow_day": "Breslow & Day 1980, IARC Sci Publ 32, ch. 6 and 7",
    "prentice_pyke": "Prentice & Pyke 1979, Biometrika 66:403",
    "gail": "Gail, Lubin & Rubinstein 1981, Biometrika 68:703",
    "king_zeng": "King & Zeng 2001, Polit Anal 9:137",
    "pearce": "Pearce 2016, BMJ i969",
    "janes_pepe": "Janes & Pepe 2008, Biometrics 64:1",
    "rothman": "Rothman, Greenland & Lash, Modern Epidemiology, ch. 8",
    "hosmer": "Hosmer, Lemeshow & Sturdivant 2013, ch. 7",
    "lumley": "Lumley 2010, Complex Surveys: A Guide to Analysis Using R (Wiley), ch. 5",
}

_SAMPLED = ("the cases and the controls were sampled separately, so how many of each the table "
            "holds was set by the study, not by how common the outcome is")
ODDS_EXIT = {"label": "Estimate odds ratios instead", "ask": "odds_ratios", "goal": "estimate"}
RANKING_EXIT = {"label": "Rank people only: report how well the model separates cases from "
                         "controls, not risks", "ask": "ranking"}
WITHIN_SET_EXIT = {"label": "Rank each case against its own matched controls",
                   "ask": "within_set_ranking"}
PREVALENCE_EXIT = {"label": "Give the population prevalence of the outcome",
                   "needs": "population_prevalence"}
STRATUM_PREVALENCE_EXIT = {"label": "Give the population prevalence within each matching stratum",
                           "needs": "prevalence_by_stratum"}
SAMPLE_ONLY_EXIT = {"label": "Estimate for these participants instead: record the sample-only "
                             "attestation", "decision": {"kind": "set_survey", "estimand": "sample"}}


class CaseControlRefused(ValueError):
    """What was asked cannot be had from this sample; ``reason`` says why in plain words, ``term``
    is the technical name as a quiet label, and ``exits`` are the ways forward."""

    def __init__(self, reason: str, exits: Sequence[Mapping[str, Any]] = (), term: str = ""):
        super().__init__(reason + (f" (Technical term: {term}.)" if term else ""))
        self.reason = reason
        self.term = term
        self.exits = [dict(e) for e in exits]


# ── which goal offers what ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Eligibility:
    offered: bool
    says: str
    exits: tuple[dict[str, Any], ...] = ()
    term: str = ""

    def require(self) -> None:
        """Raise the refusal when it is not offered."""
        if not self.offered:
            raise CaseControlRefused(self.says, self.exits, self.term)


def _check(goal: str, sampling: str, ask: str) -> None:
    if goal not in GOALS:
        raise ValueError(f"goal {goal!r} is not one of {GOALS}")
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling {sampling!r} is not one of {SAMPLINGS}")
    if ask not in ASKS:
        raise ValueError(f"ask {ask!r} is not one of {ASKS}")


def eligible(goal: str, sampling: str, ask: str, *,
             prevalence: float | Mapping[Any, float] | None = None) -> Eligibility:
    """Whether ``goal`` (describe · estimate · predict) offers ``ask`` for a sample drawn by
    ``sampling``, in plain words, with the exits when it does not. ``prevalence`` is the stated
    population prevalence (a number, or one per matching stratum), which only Predict uses."""
    _check(goal, sampling, ask)
    matched = sampling == "individually_matched"
    if ask == "absolute_risk" and goal == "predict":
        ask = "predicted_risk"
    if ask in ("prevalence", "absolute_risk"):
        what = ("How common the outcome is" if ask == "prevalence" else
                "A risk, or a difference in risk,")
        exits = [ODDS_EXIT] + ([] if matched else [
            {"label": "Predict risks recalibrated to a known population prevalence",
             "ask": "predicted_risk", "goal": "predict"}])
        return Eligibility(False, f"{what} cannot be read from these rows: {_SAMPLED}. The odds "
                                  "ratios are what a case-control sample estimates.",
                           tuple(exits), "outcome-dependent sampling; the odds ratio is the "
                                         "estimable measure")
    if ask == "odds_ratios":
        if goal == "estimate":
            how = ("conditional logistic regression within the matched sets" if matched else
                   "logistic regression adjusted for the matching factors"
                   if sampling == "frequency_matched" else "logistic regression")
            return Eligibility(True, f"Odds ratios, by {how}.")
        return Eligibility(False, "Odds ratios are effects, asked under Estimate an effect.",
                           ({"label": "Ask it under Estimate an effect", "goal": "estimate"},))
    if goal != "predict":
        return Eligibility(False, "Predictions are asked under Predict.",
                           ({"label": "Ask it under Predict", "goal": "predict"},))
    if ask == "within_set_ranking":
        if matched:
            return Eligibility(True, "Each case ranked against its own matched controls, by the "
                                     "conditional model fitted in each training fold.")
        return Eligibility(False, "There are no matched sets to rank within.", (RANKING_EXIT,))
    if ask == "ranking":
        if matched:
            return Eligibility(
                False, "The controls were chosen to resemble their case, so ranking everyone "
                       "together mixes people from different sets; a case is compared with its own "
                       "controls.", (WITHIN_SET_EXIT,), "individually matched case-control design")
        caveat = ("" if sampling == "unmatched" else " The controls were matched to the cases on "
                  "some factors, so how well those factors separate cases from controls here "
                  "understates it in the population (Janes & Pepe 2008).")
        return Eligibility(True, "How well the model separates cases from controls; it gives no "
                                 "risks." + caveat)
    # predicted_risk
    if matched:
        return Eligibility(
            False, "Risks cannot be predicted from individually matched sets. The analysis within "
                   "each set leaves no baseline risk to recalibrate, and the controls were chosen "
                   "to resemble their case, so no single correction turns these odds into anyone's "
                   "risk. This is a limit of the design, not a missing feature.",
            (WITHIN_SET_EXIT, ODDS_EXIT),
            "conditional likelihood without set intercepts; matched controls are not a random "
            "sample of non-cases")
    if prevalence is None:
        return Eligibility(
            False, f"The model's probabilities are not risks: {_SAMPLED}. With the population "
                   "prevalence of the outcome they can be recalibrated into risks.",
            (PREVALENCE_EXIT, RANKING_EXIT), "prior correction needs the population prevalence")
    if sampling == "frequency_matched" and not isinstance(prevalence, Mapping):
        return Eligibility(
            False, "The controls were drawn to match the cases' matching factors, so how often a "
                   "control was sampled differs from one matching stratum to the next: one "
                   "population prevalence cannot correct every stratum.",
            (STRATUM_PREVALENCE_EXIT, RANKING_EXIT),
            "stratum-specific sampling fractions under frequency matching")
    where = (" within each matching stratum" if isinstance(prevalence, Mapping) else "")
    return Eligibility(True, f"Risks recalibrated to the population prevalence{where} (prior "
                             "correction), the correction learned in each training fold.")


SAMPLING_WORDS = {"unmatched": "an unmatched case-control sample",
                  "frequency_matched": "a frequency-matched case-control sample",
                  "individually_matched": "individually matched sets"}


def offered_first(contract: str, goal: str, sampling: str, *,
                  prevalence: float | Mapping[Any, float] | None = None) -> tuple[str, str]:
    """The option of ``contract`` (``case_control_effects`` or ``case_control_risks``) offered
    first for ``goal`` and ``sampling``, and why, by :func:`eligible`'s own rules: the sampling
    decides it, never a static rank."""
    matched = sampling == "individually_matched"
    if contract == "case_control_effects":
        if goal == "estimate":
            return ("conditional" if matched else "unconditional",
                    eligible("estimate", sampling, "odds_ratios").says)
        if matched:
            return "conditional", eligible("predict", sampling, "within_set_ranking").says
        return "unconditional", ("The logistic model, trained in each fold; its probabilities "
                                 "are not risks until recalibrated.")
    if contract != "case_control_risks":
        raise ValueError(f"{contract!r} is not a case-control contract")
    for ask, option in (("predicted_risk", "prior_correction"), ("ranking", "ranking_only"),
                        ("within_set_ranking", "within_set_ranking")):
        e = eligible(goal, sampling, ask, prevalence=prevalence)
        if e.offered:
            return option, e.says
    raise CaseControlRefused(eligible(goal, sampling, "predicted_risk").says)


# ── reading the table ────────────────────────────────────────────────────────


def _cases(values: Any, case: Any = None) -> tuple[np.ndarray, Any]:
    """1.0 for a case, 0.0 for a control, NaN where the outcome is missing; and the case value."""
    s = pd.Series(values)
    seen = pd.unique(s.dropna())
    if len(seen) != 2:
        raise CaseControlRefused(
            f"A case-control analysis needs an outcome with two values, a case and a control; "
            f"this one has {len(seen)}.", term="binary outcome")
    if case is None:
        as_set = set(seen.tolist())
        if as_set <= {0, 1} or as_set <= {False, True}:
            case = 1
        else:
            raise ValueError(f"say which value is a case: the outcome's values are {sorted(map(str, seen))}")
    if case not in set(seen.tolist()):
        raise ValueError(f"the case value {case!r} is not one of the outcome's values")
    out = np.where(s.isna(), np.nan, (s == case).astype(float))
    return out.astype(float), case


@dataclass(frozen=True)
class _Coder:
    """How each column becomes model terms: a number as itself, a category as one indicator per
    level after the first (levels read from the rows it was fitted on)."""

    columns: tuple[str, ...]
    levels: Mapping[str, tuple[Any, ...]]  # categorical column -> its levels, the first the reference

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "_Coder":
        levels = {}
        for c in columns:
            s = frame[c]
            if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
                levels[c] = tuple(sorted(pd.unique(s.dropna()).tolist(), key=str))
        return cls(tuple(columns), levels)

    def terms(self, column: str) -> list[str]:
        if column in self.levels:
            return [f"{column}={v}" for v in self.levels[column][1:]]
        return [column]

    @property
    def names(self) -> list[str]:
        return [t for c in self.columns for t in self.terms(c)]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = {}
        for c in self.columns:
            s = frame[c]
            if c in self.levels:
                for v in self.levels[c][1:]:
                    parts[f"{c}={v}"] = np.where(s.isna(), np.nan, (s == v).astype(float))
            else:
                parts[c] = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
        return pd.DataFrame(parts, index=frame.index, columns=self.names)


def _need(frame: pd.DataFrame, columns: Sequence[str | None]) -> None:
    missing = [c for c in columns if c is not None and c not in frame.columns]
    if missing:
        raise ValueError(f"no column named {', '.join(map(repr, missing))}")


def _listed(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ── conditional logistic regression ──────────────────────────────────────────


@dataclass(frozen=True)
class ConditionalFit:
    """The exact conditional logistic fit: coefficients, their covariance (the inverse observed
    information), the conditional log-likelihood at zero and at the estimate."""

    beta: np.ndarray
    cov: np.ndarray
    loglik_null: float
    loglik: float
    iterations: int
    converged: bool


def _shapes(codes: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """The informative sets, grouped by (size, number of cases): each group's row positions as an
    (S, m) array, cases first in each row, and its number of cases."""
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    by_shape: dict[tuple[int, int], list[np.ndarray]] = {}
    for rows in np.split(order, bounds):
        d = int(y[rows].sum())
        if 0 < d < len(rows):
            rows = rows[np.argsort(-y[rows], kind="stable")]
            by_shape.setdefault((len(rows), d), []).append(rows)
    return [(np.vstack(v), d) for (_, d), v in sorted(by_shape.items())]


def _conditional_terms(X: np.ndarray, groups: list[tuple[np.ndarray, int]], beta: np.ndarray
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """The conditional log-likelihood, its gradient and the observed information at ``beta``.

    Each set's denominator is B(d, m) = Σ over every d-subset of its members of exp(Σ η), built by
    the recursion B(k, i) = B(k, i − 1) + r_i · B(k − 1, i − 1) (Gail, Lubin & Rubinstein 1981),
    with its first and second derivatives carried along; η is centered within each set so no
    exponential overflows."""
    p = X.shape[1]
    ll, grad, info = 0.0, np.zeros(p), np.zeros((p, p))
    for rows, d in groups:
        Xg = X[rows]  # (S, m, p)
        S, m, _ = Xg.shape
        eta = Xg @ beta
        c = eta.max(axis=1)
        r = np.exp(eta - c[:, None])
        B = np.zeros((S, d + 1))
        B[:, 0] = 1.0
        dB = np.zeros((S, d + 1, p))
        d2B = np.zeros((S, d + 1, p, p))
        for i in range(m):
            x = Xg[:, i, :]
            ri = r[:, i]
            xx = x[:, :, None] * x[:, None, :]
            for k in range(min(i + 1, d), 0, -1):
                b1, db1, d2b1 = B[:, k - 1], dB[:, k - 1], d2B[:, k - 1]
                xd = x[:, :, None] * db1[:, None, :]
                d2B[:, k] += ri[:, None, None] * (xx * b1[:, None, None] + xd
                                                  + xd.transpose(0, 2, 1) + d2b1)
                dB[:, k] += ri[:, None] * (x * b1[:, None] + db1)
                B[:, k] += ri * b1
        Bd = B[:, d]
        mean = dB[:, d] / Bd[:, None]
        ll += float(eta[:, :d].sum() - np.sum(d * c + np.log(Bd)))
        grad += Xg[:, :d, :].sum(axis=(0, 1)) - mean.sum(axis=0)
        info += (d2B[:, d] / Bd[:, None, None]).sum(axis=0) - np.einsum("si,sj->ij", mean, mean)
    return ll, grad, (info + info.T) / 2


def conditional_logistic(X: Any, y: Any, sets: Any) -> ConditionalFit:
    """Conditional logistic regression of ``y`` (1 a case, 0 a control) on ``X`` within ``sets``,
    by Newton–Raphson on the exact conditional likelihood (module docstring), steps halved until
    the likelihood does not fall. Sets with no case or no control add nothing. The caller checks
    that the columns vary within sets and that no column separates the cases from their controls
    (:func:`odds_ratios` does both)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = pd.factorize(pd.Series(sets), sort=True)[0]
    groups = _shapes(codes, y)
    if not groups:
        raise CaseControlRefused("No matched set holds both a case and a control, so there is "
                                 "nothing to compare.", term="no informative matched sets")
    beta = np.zeros(X.shape[1])
    ll, grad, info = _conditional_terms(X, groups, beta)
    ll0 = ll
    converged = False
    it = 0
    for it in range(1, _MAX_ITER + 1):
        step = np.linalg.solve(info, grad)
        if float(grad @ step) < _TOL ** 2 * max(1.0, abs(ll)):
            converged = True
            break
        for _ in range(60):
            trial = beta + step
            ll_t, g_t, i_t = _conditional_terms(X, groups, trial)
            if ll_t >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            step /= 2.0
        settled = (abs(ll_t - ll) <= 1e-14 * max(1.0, abs(ll))
                   and float(np.max(np.abs(step))) < 1e-9)
        beta, ll, grad, info = trial, ll_t, g_t, i_t
        if settled:  # rounding keeps the Newton decrement from its floor; the fit has stopped moving
            converged = True
            break
    cov = np.linalg.inv(info)
    return ConditionalFit(beta, (cov + cov.T) / 2, ll0, ll, it, converged)


def _logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Unweighted logistic regression: (β, covariance, log-likelihood, converged)."""
    from turbotab.core.models.survey import weighted_logistic

    fit = weighted_logistic(X, y, np.ones(len(y)))
    eta = X @ fit.estimate
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    cov = np.linalg.inv(fit.information)
    return fit.estimate, (cov + cov.T) / 2, ll, fit.converged


# ── odds ratios ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OddsRatio:
    term: str
    log_or: float
    se: float
    odds_ratio: float
    lower: float
    upper: float
    p: float
    df: float | None = None  # None: the normal (Wald z); else t on the design's df


@dataclass(frozen=True)
class CaseControlOdds:
    sampling: str
    estimator: str
    rows: tuple[OddsRatio, ...]
    n_rows: int
    n_cases: int
    n_controls: int
    matching: tuple[str, ...] = ()
    n_sets: int | None = None
    n_informative_sets: int | None = None
    controls_per_case: dict[str, int] | None = None  # "1:2" -> sets with one case and two controls
    weighted: bool = False
    df: float | None = None
    loglik: float | None = None
    loglik_null: float | None = None
    lr_test: tuple[float, int, float] | None = None  # (statistic, df, p) of every term at once
    withheld: tuple[dict[str, Any], ...] = ()  # prevalence and risk: refused, with reasons and exits
    concerns: tuple[str, ...] = ()

    def row(self, term: str) -> OddsRatio:
        for r in self.rows:
            if r.term == term:
                return r
        raise KeyError(term)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def says(self) -> str:
        return _says(self)


def _withheld(sampling: str) -> tuple[dict[str, Any], ...]:
    out = []
    for ask in ("prevalence", "absolute_risk"):
        e = eligible("estimate", sampling, ask)
        out.append({"what": ask, "says": e.says, "exits": list(e.exits), "term": e.term})
    return tuple(out)


def _z_row(term: str, b: float, se: float) -> OddsRatio:
    p = float(2 * stats.norm.sf(abs(b / se))) if se > 0 else math.nan
    return OddsRatio(term, float(b), float(se), float(math.exp(b)), float(math.exp(b - Z * se)),
                     float(math.exp(b + Z * se)), p)


def odds_ratios(frame: pd.DataFrame, outcome: str, exposures: Sequence[str], *,
                sampling: str = "unmatched", matching: Sequence[str] = (),
                adjust: Sequence[str] = (), sets: str | None = None, case: Any = None,
                design: Any = None, row_ids: Any = None) -> CaseControlOdds:
    """Odds ratios for ``exposures`` (and the ``adjust`` columns) from a case-control sample
    (module docstring). ``matching`` names the matching factors: in the model under frequency
    matching, carried by ``sets`` (the matched-set column) under individual matching. ``design``
    (a :class:`~turbotab.core.models.survey.SurveyDesign`, the population answer) makes an
    unmatched or frequency-matched analysis design-based; ``row_ids`` are the rows' ids in it (the
    frame's index by default). Rows missing a value leave. Raises :class:`CaseControlRefused`."""
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling {sampling!r} is not one of {SAMPLINGS}")
    exposures, matching, adjust = list(exposures), list(matching), list(adjust)
    if not exposures:
        raise ValueError("name at least one exposure")
    _need(frame, [outcome, sets, *exposures, *matching, *adjust])
    matched = sampling == "individually_matched"
    if sampling == "frequency_matched" and not matching:
        raise CaseControlRefused(
            "A frequency-matched sample is analyzed with its matching factors in the model, and "
            "none is named.", ({"label": "Name the matching factors", "needs": "matching"},),
            "adjustment for the matching factors")
    both = [c for c in exposures if c in matching]
    if both:
        raise CaseControlRefused(
            f"{_listed(both)} {'was' if len(both) == 1 else 'were'} matched on, so the controls "
            "were chosen to resemble the cases on it: its odds ratio cannot be estimated here.",
            ({"label": f"Leave {_listed(both)} out of the exposures", "drop": both},),
            "matching factor")
    if matched:
        return _matched(frame, outcome, exposures, adjust, matching, sets, case, design)
    y, _ = _cases(frame[outcome], case)
    columns = exposures + [c for c in adjust if c not in exposures] + \
        [c for c in matching if c not in exposures + adjust]
    coder = _Coder.fit(frame, columns)
    M = coder.transform(frame)
    keep = np.isfinite(y) & M.notna().all(axis=1).to_numpy()
    M, y = M[keep], y[keep]
    names = coder.names
    matching_terms = {t for c in matching for t in coder.terms(c)}
    concerns = _left_concern(int((~keep).sum()))
    n, cases = len(y), int(y.sum())
    if cases == 0 or cases == n:
        raise CaseControlRefused("The rows left hold only cases or only controls.")
    X = np.column_stack([np.ones(n), M.to_numpy(dtype=float)])
    _collinear(X[:, 1:], names, intercept=True)
    from turbotab.core.models.inference import separated_columns

    gone = [names[j - 1] for j in separated_columns(X, y) if j > 0]
    if gone:
        raise CaseControlRefused(
            f"{_listed(gone)} {'separates' if len(gone) == 1 else 'separate'} the cases from the "
            "controls completely, so the odds ratio is infinite and no interval exists.",
            ({"label": f"Leave {_listed(gone)} out, or merge its rare levels", "drop": gone},),
            "complete separation")
    shown = [t for t in names if t not in matching_terms]
    if design is not None:
        ids = _row_ids(frame, row_ids)[keep]
        rows, df, est, more = _design_rows(M, y, design, ids, shown)
        concerns.extend(more)
        ll = ll0 = lr = None
    else:
        beta, cov, ll, ok = _logistic(X, y)
        se = np.sqrt(np.diag(cov))
        rows = tuple(_z_row(t, beta[i + 1], se[i + 1]) for i, t in enumerate(names) if t in shown)
        share = cases / n
        ll0 = float(cases * math.log(share) + (n - cases) * math.log(1 - share))
        stat = 2 * (ll - ll0)
        lr = (float(stat), len(names), float(stats.chi2.sf(stat, len(names))))
        df = None
        est = "logistic regression (maximum likelihood; Wald intervals)"
        if not ok:
            concerns.append("The fit stopped before converging; treat these numbers with care.")
    if matching:
        concerns.append(f"The matching factors ({_listed(matching)}) are in the model; their own "
                        "odds ratios are not estimated, because the matching fixed them by design.")
    return CaseControlOdds(
        sampling=sampling, estimator=est, rows=rows, n_rows=n, n_cases=cases,
        n_controls=n - cases, matching=tuple(matching), weighted=design is not None, df=df,
        loglik=ll, loglik_null=ll0, lr_test=lr, withheld=_withheld(sampling),
        concerns=tuple(concerns))


def _left_concern(left: int) -> list[str]:
    return [f"{left} row{'s' if left != 1 else ''} with a missing value "
            f"{'leave' if left != 1 else 'leaves'}."] if left else []


def _collinear(M: np.ndarray, names: Sequence[str], *, intercept: bool) -> None:
    base = [np.ones(len(M))] if intercept else []
    kept: list[np.ndarray] = list(base)
    dependent = []
    for j, t in enumerate(names):
        trial = np.column_stack(kept + [M[:, j]]) if kept else M[:, [j]]
        if np.linalg.matrix_rank(trial) <= len(kept):
            dependent.append(t)
        else:
            kept.append(M[:, j])
    if dependent:
        raise CaseControlRefused(
            f"{_listed(dependent)} {'is' if len(dependent) == 1 else 'are'} fully determined by the "
            "other columns, so no odds ratio can be told apart for it.",
            ({"label": f"Leave {_listed(dependent)} out", "drop": dependent},), "collinearity")


def _design_rows(M: pd.DataFrame, y: np.ndarray, design: Any, ids: np.ndarray,
                 shown: Sequence[str]) -> tuple[tuple[OddsRatio, ...], float, str, list[str]]:
    """The design-based logistic fit (R's svyglm, quasibinomial) on the kept rows, whose ids in
    the design are ``ids``."""
    from turbotab.core.models.survey import survey_table

    table = survey_table("binary", M.set_axis(pd.Index(ids)), y, [0.0, 1.0], design)
    if table.info.get("refused"):
        raise CaseControlRefused(table.info["refused"], table.info.get("exits") or (SAMPLE_ONLY_EXIT,),
                                 "survey design")
    by = {r["feature"]: r for r in table.rows}
    out = []
    for t in shown:
        r = by[t]
        out.append(OddsRatio(t, r["estimate"], r["se"], math.exp(r["estimate"]),
                             math.exp(r["ci_low"]), math.exp(r["ci_high"]), r["p"], r["df"]))
    return (tuple(out), float(table.info["survey"]["df"]), table.info["estimator"],
            list(table.concerns))


def _row_ids(frame: pd.DataFrame, row_ids: Any) -> np.ndarray:
    """Each frame row's id in the survey design (asked only when there is one)."""
    if row_ids is not None:
        ids = np.asarray(row_ids, dtype=np.int64)
        if len(ids) != len(frame):
            raise ValueError("row_ids must name every row of the frame")
        return ids
    if not pd.api.types.is_integer_dtype(frame.index):
        raise ValueError("Under a survey design each row needs its row number in the design, and "
                         "these rows are labeled by name: pass row_ids.")
    return np.asarray(frame.index, dtype=np.int64)


def _matched(frame: pd.DataFrame, outcome: str, exposures: list[str], adjust: list[str],
             matching: list[str], sets: str | None, case: Any, design: Any) -> CaseControlOdds:
    if sets is None:
        raise CaseControlRefused(
            "Individually matched sets are analyzed within each set, and no column names the sets.",
            ({"label": "Name the matched-set column", "needs": "sets"},), "matched-set identifier")
    if design is not None:
        exits = [SAMPLE_ONLY_EXIT]
        if matching:
            exits.append({"label": "Analyze it as frequency-matched: logistic regression adjusted "
                                   "for the matching factors", "sampling": "frequency_matched"})
        raise CaseControlRefused(
            "The analysis within matched sets has no survey-weighted version in common use, so "
            "these sets cannot be weighted to the surveyed population.", exits,
            "conditional logistic regression under a complex survey design")
    y, _ = _cases(frame[outcome], case)
    columns = exposures + [c for c in adjust if c not in exposures]
    coder = _Coder.fit(frame, columns)
    M = coder.transform(frame)
    keep = np.isfinite(y) & M.notna().all(axis=1).to_numpy() & frame[sets].notna().to_numpy()
    left = int((~keep).sum())
    M, y, set_ids = M[keep], y[keep], frame[sets].to_numpy()[keep]
    names = coder.names
    codes = pd.factorize(pd.Series(set_ids), sort=True)[0]
    n_sets = int(codes.max() + 1) if len(codes) else 0
    cases_per = np.bincount(codes, weights=y, minlength=n_sets).astype(int)
    size = np.bincount(codes, minlength=n_sets)
    informative = (cases_per > 0) & (cases_per < size)
    if not informative.any():
        raise CaseControlRefused("No matched set holds both a case and a control, so there is "
                                 "nothing to compare.", term="no informative matched sets")
    in_inf = informative[codes]
    X = M.to_numpy(dtype=float)
    Xi, yi, ci = X[in_inf], y[in_inf], codes[in_inf]
    means = np.zeros((n_sets, X.shape[1]))
    np.add.at(means, ci, Xi)
    means /= np.maximum(size, 1)[:, None]
    centered = Xi - means[ci]
    constant = [t for j, t in enumerate(names) if np.allclose(centered[:, j], 0.0, atol=1e-12)]
    if constant:
        cols = [c for c in columns if any(t in constant for t in coder.terms(c))]
        raise CaseControlRefused(
            f"{_listed(cols)} {'is' if len(cols) == 1 else 'are'} the same for every member of each "
            "matched set, so the comparison within sets says nothing about it: it was matched on, "
            "and its odds ratio cannot be estimated.",
            ({"label": f"Leave {_listed(cols)} out", "drop": cols},),
            "a matching factor is not estimable in conditional logistic regression")
    _collinear(centered, names, intercept=False)
    from turbotab.core.models.inference import separated_columns

    pairs = _case_control_pairs(ci, yi)
    Zd = Xi[pairs[:, 0]] - Xi[pairs[:, 1]]
    gone = [names[j] for j in separated_columns(Zd, np.ones(len(Zd)))]
    if gone:
        raise CaseControlRefused(
            f"Within every set, {_listed(gone)} puts each case above (or below) its own controls, "
            "so the odds ratio is infinite and no interval exists.",
            ({"label": f"Leave {_listed(gone)} out, or merge its rare levels", "drop": gone},),
            "monotone conditional likelihood (separation within matched sets)")
    fit = conditional_logistic(Xi, yi, ci)
    se = np.sqrt(np.diag(fit.cov))
    sd = Xi.std(axis=0)
    if not fit.converged or np.any(np.abs(fit.beta) * np.where(sd > 0, sd, 1.0) > _HUGE):
        raise CaseControlRefused(
            "The fit did not settle on finite odds ratios.",
            ({"label": "Leave out the exposure with the largest odds ratio, or merge rare levels",
              "drop": [names[int(np.argmax(np.abs(fit.beta) * sd))]]},),
            "non-convergence of the conditional likelihood")
    rows = tuple(_z_row(t, fit.beta[i], se[i]) for i, t in enumerate(names))
    stat = 2 * (fit.loglik - fit.loglik_null)
    ratio: dict[str, int] = {}
    for k in np.flatnonzero(informative):
        key = f"{cases_per[k]}:{size[k] - cases_per[k]}"
        ratio[key] = ratio.get(key, 0) + 1
    concerns = _left_concern(left)
    dropped = int(n_sets - informative.sum())
    if dropped:
        concerns.append(f"{dropped} set{'s' if dropped != 1 else ''} without both a case and a "
                        f"control add{'' if dropped != 1 else 's'} nothing and "
                        f"{'leave' if dropped != 1 else 'leaves'}.")
    if any(int(k.split(':')[0]) > 1 for k in ratio):
        concerns.append("Some sets hold more than one case; their likelihood counts every way of "
                        "choosing that many cases from the set (the exact conditional likelihood, "
                        "Gail, Lubin & Rubinstein 1981).")
    if matching:
        loose = _not_matched_within(frame.loc[keep].iloc[in_inf], matching, sets)
        if loose:
            concerns.append(f"{_listed(loose)} differ{'s' if len(loose) == 1 else ''} within some "
                            "sets (matching within a range): what the matching left is not "
                            "removed by the analysis and may confound a little.")
    return CaseControlOdds(
        sampling="individually_matched",
        estimator="conditional logistic regression (exact conditional likelihood; Wald intervals)",
        rows=rows, n_rows=int(in_inf.sum()), n_cases=int(yi.sum()),
        n_controls=int(len(yi) - yi.sum()), matching=tuple(matching), n_sets=n_sets,
        n_informative_sets=int(informative.sum()), controls_per_case=dict(sorted(ratio.items())),
        loglik=fit.loglik, loglik_null=fit.loglik_null,
        lr_test=(float(stat), len(names), float(stats.chi2.sf(stat, len(names)))),
        withheld=_withheld("individually_matched"), concerns=tuple(concerns))


def _case_control_pairs(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every (case, control) pair of rows within a set."""
    out = []
    frame = pd.DataFrame({"s": codes, "y": y, "i": np.arange(len(y))})
    for _, g in frame.groupby("s", sort=True):
        cs, ct = g["i"][g["y"] == 1].to_numpy(), g["i"][g["y"] == 0].to_numpy()
        out.extend((a, b) for a in cs for b in ct)
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def _not_matched_within(frame: pd.DataFrame, matching: Sequence[str], sets: str) -> list[str]:
    return [c for c in matching if (frame.groupby(sets)[c].nunique(dropna=False) > 1).any()]


# ── Predict: the prior correction, the folds, the within-set ranking ─────────


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def prior_correction_offset(sample_share: float, prevalence: float) -> float:
    """What the prior correction subtracts from the log odds: ln[(ȳ/(1 − ȳ)) · ((1 − τ)/τ)],
    ȳ the sample's case share and τ the population prevalence (King & Zeng 2001, eq. 7)."""
    for name, v in (("sample share", sample_share), ("prevalence", prevalence)):
        if not 0.0 < float(v) < 1.0:
            raise ValueError(f"the {name} must lie strictly between 0 and 1, not {v!r}")
    return float(math.log(sample_share / (1 - sample_share)) - math.log(prevalence / (1 - prevalence)))


def recalibrate(probability: Any, offset: float | np.ndarray) -> np.ndarray:
    """The sample's probabilities with ``offset`` taken off their log odds."""
    return 1.0 / (1.0 + np.exp(-(_logit(probability) - offset)))


def _check_prevalence(prevalence: Any) -> None:
    values = prevalence.values() if isinstance(prevalence, Mapping) else [prevalence]
    for v in values:
        if not (isinstance(v, (int, float)) and 0.0 < float(v) < 1.0):
            raise CaseControlRefused(f"A population prevalence is a share between 0 and 1; "
                                     f"{v!r} is not.", (PREVALENCE_EXIT,))


class PriorCorrected(ClassifierMixin, BaseEstimator):
    """A classifier trained on a case-control sample, its predicted probabilities recalibrated to
    the population prevalence (King & Zeng 2001's prior correction; module docstring).

    ``fit`` fits a clone of ``estimator`` and learns the case share ȳ of the rows it is given (so
    in cross-validation, of the training fold alone); ``predict_proba`` takes
    ``prior_correction_offset(ȳ, prevalence)`` off each prediction's log odds. The case is the
    second of the sorted classes (1, True). With ``stratum`` (a column of X: the frequency-matching
    factor) the share and the prevalence are per stratum, and ``prevalence`` maps each stratum to
    its own."""

    def __init__(self, estimator: Any = None, prevalence: float | Mapping[Any, float] = 0.5,
                 stratum: str | None = None):
        self.estimator = estimator
        self.prevalence = prevalence
        self.stratum = stratum

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "PriorCorrected":
        _check_prevalence(self.prevalence)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise CaseControlRefused("A case-control model needs both cases and controls among the "
                                     "rows it is trained on.")
        event = (y == self.classes_[1]).astype(float)
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        if self.stratum is None:
            if isinstance(self.prevalence, Mapping):
                raise ValueError("a prevalence per stratum needs the stratum column")
            self.sample_share_ = float(event.mean())
            self.offset_ = prior_correction_offset(self.sample_share_, float(self.prevalence))
            return self
        if not isinstance(self.prevalence, Mapping):
            raise CaseControlRefused(eligible("predict", "frequency_matched", "predicted_risk",
                                              prevalence=0.5).says,
                                     (STRATUM_PREVALENCE_EXIT, RANKING_EXIT))
        s = _stratum_values(X, self.stratum)
        shares, offsets = {}, {}
        for level in pd.unique(s):
            here = event[s == level]
            if level not in self.prevalence:
                raise CaseControlRefused(f"No population prevalence is given for the stratum "
                                         f"{level!r}.", (STRATUM_PREVALENCE_EXIT,))
            share = float(here.mean())
            if not 0.0 < share < 1.0:
                raise CaseControlRefused(
                    f"The training rows of the stratum {level!r} hold only "
                    f"{'cases' if share == 1.0 else 'controls'}, so its correction cannot be "
                    "learned.", (RANKING_EXIT,))
            shares[level] = share
            offsets[level] = prior_correction_offset(share, float(self.prevalence[level]))
        self.sample_share_, self.offset_ = shares, offsets
        return self

    def _offsets(self, X: Any) -> float | np.ndarray:
        if self.stratum is None:
            return self.offset_
        s = _stratum_values(X, self.stratum)
        unseen = sorted({str(v) for v in pd.unique(s) if v not in self.offset_})
        if unseen:
            raise CaseControlRefused(f"The stratum {_listed(unseen)} was not among the training "
                                     "rows, so it has no correction.")
        return np.asarray([self.offset_[v] for v in s], dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.asarray(self.estimator_.predict_proba(X), dtype=float)[:, 1]
        risk = recalibrate(p, self._offsets(X))
        return np.column_stack([1 - risk, risk])

    def decision_function(self, X: Any) -> np.ndarray:
        return _logit(self.predict_proba(X)[:, 1])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def _stratum_values(X: Any, stratum: str) -> np.ndarray:
    if not isinstance(X, pd.DataFrame) or stratum not in X.columns:
        raise CaseControlRefused(
            f"The correction per matching stratum reads {stratum!r} among the model's features, "
            "and it is not one of them: the model must condition on the factor it was matched on.",
            ({"label": f"Add {stratum} to the features", "add": [stratum]},))
    return X[stratum].to_numpy()


def _join_sets(sets: np.ndarray | None, persons: np.ndarray | None, n: int) -> np.ndarray | None:
    """One group label per row: its matched set, sets that share a person joined into one."""
    if sets is None and persons is None:
        return None
    from turbotab.core.models.folds import unit_labels

    if sets is None:
        return unit_labels(persons, n)
    s = unit_labels(sets, n)
    if persons is None:
        return s
    p = unit_labels(persons, n)
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first: dict[str, str] = {}
    for si, pi in zip(s, p):
        find(si)
        if pi.startswith("__missing_"):
            continue
        if pi in first:
            a, b = find(first[pi]), find(si)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[pi] = si
    return np.asarray([find(si) for si in s], dtype=object)


def case_control_folds(y: Any, *, sets: Any = None, persons: Any = None, folds: int = 5,
                       seed: int = 0) -> np.ndarray:
    """A fold number per row for cross-validating on a case-control sample: matched sets whole
    (sets sharing a person, as incidence-density sampling allows, joined into one), cases spread
    across the folds. A row with no set is a unit of its own."""
    from turbotab.core.models.folds import kfold_assignment

    y = np.asarray(y)
    n = len(y)
    groups = _join_sets(None if sets is None else np.asarray(sets, dtype=object),
                        None if persons is None else np.asarray(persons, dtype=object), n)
    fold = kfold_assignment(n, strata=y, groups=groups, folds=folds, seed=seed)
    if groups is not None:
        split = pd.Series(fold).groupby(pd.Series(groups)).nunique()
        if (split > 1).any():  # kfold_assignment keeps units whole; this guards the guarantee
            raise RuntimeError("a matched set was split across folds")
    return fold


@dataclass(frozen=True)
class OutOfFoldRisks:
    risk: np.ndarray  # recalibrated risk of each row, from the model its fold did not train
    fold: np.ndarray
    sample_share: tuple[Any, ...]  # each fold's training case share (a number, or one per stratum)
    offset: tuple[Any, ...]
    prevalence: float | Mapping[Any, float]
    sampling: str

    @property
    def says(self) -> str:
        if isinstance(self.prevalence, Mapping):
            return ("Each prediction is a risk, recalibrated to the stated prevalence within its "
                    "matching stratum by the correction its training fold learned.")
        return (f"Each prediction is a risk, recalibrated to the stated population prevalence of "
                f"{self.prevalence:.3g} by the correction its training fold learned (the folds' "
                f"case shares {min(self.sample_share):.3g} to {max(self.sample_share):.3g}).")


def predicted_risks(estimator: Any, X: Any, y: Any, *, sampling: str = "unmatched",
                    prevalence: float | Mapping[Any, float] | None = None,
                    stratum: str | None = None, sets: Any = None, persons: Any = None,
                    folds: int = 5, seed: int = 0, design: Any = None) -> OutOfFoldRisks:
    """Out-of-fold risks from a model trained on a case-control sample: in each training fold a
    :class:`PriorCorrected` copy of ``estimator`` is fitted (the model and the case share both),
    and the held-out rows are scored. Refused for individually matched sets, without the
    prevalence, with one prevalence under frequency matching, and under survey weights."""
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling {sampling!r} is not one of {SAMPLINGS}")
    eligible("predict", sampling, "predicted_risk", prevalence=prevalence).require()
    if design is not None:
        raise CaseControlRefused(
            "The stated prevalence already corrects for how the cases and controls were drawn; "
            "survey weights that also carry it would correct the same thing twice.",
            ({"label": "Recalibrate without the survey weights", "design": None}, RANKING_EXIT),
            "prior correction under a weighted design")
    _check_prevalence(prevalence)
    if sampling == "frequency_matched" and stratum is None:
        raise CaseControlRefused("A prevalence per stratum needs the matching stratum's column.",
                                 ({"label": "Name the matching stratum's column",
                                   "needs": "stratum"},))
    yv = np.asarray(y)
    n = len(yv)
    fold = case_control_folds(yv, sets=sets, persons=persons, folds=folds, seed=seed)
    risk = np.full(n, np.nan)
    shares, offsets = [], []
    take = (lambda A, rows: A.iloc[rows]) if isinstance(X, (pd.DataFrame, pd.Series)) else \
        (lambda A, rows: np.asarray(A)[rows])
    for f in np.unique(fold):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        model = PriorCorrected(clone(estimator), prevalence, stratum).fit(take(X, train), yv[train])
        risk[test] = model.predict_proba(take(X, test))[:, 1]
        shares.append(model.sample_share_)
        offsets.append(model.offset_)
    return OutOfFoldRisks(risk, fold, tuple(shares), tuple(offsets), prevalence, sampling)


@dataclass(frozen=True)
class MatchedScores:
    score: np.ndarray  # each row's linear predictor from the conditional model its fold did not train
    fold: np.ndarray
    concordance: float  # the share of case–control pairs within held-out sets the case ranks above
    n_pairs: int
    coefficients: tuple[np.ndarray, ...]  # each training fold's conditional log odds ratios
    terms: tuple[str, ...]

    @property
    def says(self) -> str:
        return (f"Within its own matched set, a case was ranked above its control in "
                f"{100 * self.concordance:.0f}% of {self.n_pairs:,} case–control pairs, each scored "
                "by a model that never saw its set. The scores rank people within a set only: "
                "they are not risks.")


def matched_set_scores(frame: pd.DataFrame, outcome: str, exposures: Sequence[str], sets: str, *,
                       persons: str | None = None, case: Any = None, folds: int = 5,
                       seed: int = 0) -> MatchedScores:
    """Under Predict, each case ranked against its own matched controls: in each training fold the
    conditional logistic model (and the coding of its categories) is fitted on the training sets,
    and every held-out row is scored by its linear predictor. The concordance counts the
    case–control pairs within held-out sets in which the case scores higher (a tie counts half)."""
    exposures = list(exposures)
    _need(frame, [outcome, sets, persons, *exposures])
    y, _ = _cases(frame[outcome], case)
    keep = np.isfinite(y) & frame[exposures].notna().all(axis=1).to_numpy() & \
        frame[sets].notna().to_numpy()
    data = frame.loc[keep]
    y = y[keep]
    set_ids = data[sets].to_numpy()
    fold = case_control_folds(y, sets=set_ids,
                              persons=None if persons is None else data[persons].to_numpy(),
                              folds=folds, seed=seed)
    score = np.full(len(y), np.nan)
    coefs = []
    terms: list[str] = []
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        coder = _Coder.fit(data[train], exposures)
        terms = coder.names
        fit = odds_ratios(data[train], outcome, exposures, sampling="individually_matched",
                          sets=sets, case=case)
        beta = np.array([fit.row(t).log_or for t in terms])
        coefs.append(beta)
        Xt = coder.transform(data[test]).fillna(0.0).to_numpy(dtype=float)
        score[test] = Xt @ beta
    codes = pd.factorize(pd.Series(set_ids), sort=True)[0]
    pairs = _case_control_pairs(codes, y)
    diff = score[pairs[:, 0]] - score[pairs[:, 1]]
    conc = float(np.mean(np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0)))) if len(pairs) \
        else math.nan
    full = np.full(len(frame), np.nan)
    full[np.flatnonzero(keep)] = score
    fold_full = np.full(len(frame), -1)
    fold_full[np.flatnonzero(keep)] = fold
    return MatchedScores(full, fold_full, conc, int(len(pairs)), tuple(coefs), tuple(terms))


# ── detection helpers (signals for the noticings; no routing) ────────────────


@dataclass(frozen=True)
class Signal:
    """A sign that a table may be a case-control sample. ``says`` is in plain words and never
    settles the design: the design question does."""

    kind: str  # outcome_share_above_population · one_case_per_set
    says: str
    column: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


# The outcome share must be at least this many times the stated rate, and the excess beyond
# chance (one-sided exact binomial) below this p, before the share is a signal.
SHARE_RATIO = 2.0
SHARE_P = 0.001
# A set column is a signal when at least this share of its sets (two or more rows) holds exactly
# one case, over at least this many sets.
ONE_CASE_SHARE = 0.9
MIN_SETS = 5
MAX_SET_SIZE = 11  # 1:10 matching; larger groups are clusters, not matched sets


def prevalence_signal(outcome: Any, population_prevalence: float, *, case: Any = None
                      ) -> Signal | None:
    """A signal when the share of rows with the outcome is far above a stated population rate:
    at least :data:`SHARE_RATIO` times it, and more than chance allows (one-sided exact binomial
    p < :data:`SHARE_P`). None otherwise. The ratio of the sample's odds to the population's is
    how much more often a case was sampled than a control."""
    if not 0.0 < float(population_prevalence) < 1.0:
        raise ValueError("a population prevalence lies strictly between 0 and 1")
    y, _ = _cases(outcome, case)
    y = y[np.isfinite(y)]
    n, k = len(y), int(y.sum())
    if n == 0:
        return None
    share = k / n
    tau = float(population_prevalence)
    p = float(stats.binomtest(k, n, tau, alternative="greater").pvalue)
    if share < SHARE_RATIO * tau or p >= SHARE_P:
        return None
    odds = (share / (1 - share)) / (tau / (1 - tau)) if share < 1 else math.inf
    return Signal("outcome_share_above_population",
                  f"{100 * share:.0f}% of the rows have the outcome, against a stated "
                  f"{100 * tau:.2g}% in the population: the cases may have been sampled separately "
                  "from the controls (a case-control sample).",
                  evidence={"share": share, "population_prevalence": tau, "n": n, "cases": k,
                            "ratio": share / tau, "sampling_odds_ratio": odds, "p": p})


def matched_sets_signal(frame: pd.DataFrame, outcome: str, set_column: str, *, case: Any = None,
                        candidates: Sequence[str] | None = None) -> Signal | None:
    """A signal when ``set_column`` groups the rows into sets (2 to :data:`MAX_SET_SIZE` rows)
    that each hold exactly one case: at least :data:`ONE_CASE_SHARE` of :data:`MIN_SETS` or more
    such sets. The evidence counts the case-to-control ratios and names the columns equal within
    (nearly) every set, the likely matching factors (among ``candidates``, default every other
    column)."""
    _need(frame, [outcome, set_column])
    y, _ = _cases(frame[outcome], case)
    ok = np.isfinite(y) & frame[set_column].notna().to_numpy()
    if not ok.any():
        return None
    data = frame.loc[ok]
    g = pd.DataFrame({"s": data[set_column].to_numpy(), "y": y[ok]})
    size = g.groupby("s")["y"].size()
    cases = g.groupby("s")["y"].sum()
    multi = size[size >= 2].index
    if len(multi) < MIN_SETS or size.median() > MAX_SET_SIZE:
        return None
    one = int((cases[multi] == 1).sum())
    share = one / len(multi)
    if share < ONE_CASE_SHARE:
        return None
    ratios = (cases[multi].astype(int).astype(str) + ":" +
              (size[multi] - cases[multi]).astype(int).astype(str)).value_counts()
    others = [c for c in (candidates if candidates is not None else data.columns)
              if c not in (outcome, set_column)]
    matching = []
    for c in others:
        within = data.groupby(set_column)[c].nunique(dropna=False).loc[multi]
        if data[c].nunique(dropna=True) > 1 and float((within == 1).mean()) >= ONE_CASE_SHARE:
            matching.append(c)
    singletons = int((size == 1).sum())
    words = (f"The column {set_column} groups the rows into {len(multi):,} sets, and "
             f"{'every one' if one == len(multi) else f'{one:,} of them'} holds exactly one case: "
             "the rows may be matched case-control sets, each case with its own controls.")
    if matching:
        words += f" Within each set, {_listed(matching)} {'is' if len(matching) == 1 else 'are'} the same."
    return Signal("one_case_per_set", words, column=set_column,
                  evidence={"sets": int(len(multi)), "one_case_sets": one, "share": share,
                            "ratios": {str(k): int(v) for k, v in ratios.sort_index().items()},
                            "singletons": singletons, "matching_factors": matching})


def set_column_signals(frame: pd.DataFrame, outcome: str, *, case: Any = None,
                       exclude: Sequence[str] = ()) -> list[Signal]:
    """:func:`matched_sets_signal` for every column that could name sets: whole numbers or text,
    repeating, not the outcome; strongest (the most one-case sets) first."""
    out = []
    for c in frame.columns:
        if c == outcome or c in exclude:
            continue
        s = frame[c].dropna()
        if s.empty or s.nunique() >= len(s) or s.nunique() < MIN_SETS:
            continue
        if pd.api.types.is_float_dtype(s) and not np.allclose(s, np.round(s)):
            continue
        found = matched_sets_signal(frame, outcome, c, case=case)
        if found is not None:
            out.append(found)
    return sorted(out, key=lambda s: (-s.evidence["share"], -s.evidence["sets"]))


# ── words ────────────────────────────────────────────────────────────────────


def _num(v: float) -> str:
    return f"{v:.3g}"


def _says(r: CaseControlOdds) -> str:
    first = r.rows[0]
    text = (f"Among these cases and controls, the odds of being a case were {_num(first.odds_ratio)} "
            f"times as high per unit of {first.term}" if "=" not in first.term else
            f"Among these cases and controls, the odds of being a case were "
            f"{_num(first.odds_ratio)} times as high for {first.term.replace('=', ' = ')}")
    text += f" (95% CI {_num(first.lower)} to {_num(first.upper)})"
    if r.sampling == "individually_matched":
        text += ", comparing each case with its own matched controls"
    return (text + ". How common the outcome is, and anyone's risk, cannot be read from these rows: "
            + _SAMPLED + ".")


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _count(k: int, noun: str) -> str:
    return f"{_WORDS[k] if k < len(_WORDS) else k} {noun}{'s' if k != 1 else ''}"


def _ratio_words(key: str, sets: int) -> str:
    cases, controls = (int(v) for v in key.split(":"))
    return (f"{sets:,} set{'s' if sets != 1 else ''} of {_count(cases, 'case')} and "
            f"{_count(controls, 'control')}")


def case_control_sentence(result: CaseControlOdds | None = None, *,
                          sampling: str = "unmatched") -> str:
    """The methods sentence of a case-control analysis: its sampling, its estimator, the matching
    factors and what is withheld."""
    r = result
    sampling = r.sampling if r else sampling
    if sampling == "individually_matched":
        sets = (f" within the {r.n_informative_sets:,} matched sets" if r and r.n_informative_sets
                else " within the matched sets")
        ratio = ""
        if r and r.controls_per_case:
            ratio = " (" + "; ".join(_ratio_words(k, v) for k, v in sorted(
                r.controls_per_case.items(), key=lambda kv: -kv[1])) + ")"
        several = not r or any(int(k.split(":")[0]) > 1 for k in (r.controls_per_case or {}))
        out = (f"Odds ratios were estimated by conditional logistic regression{sets}{ratio} "
               "(Breslow & Day 1980)")
        if several:
            out += (", with the exact conditional likelihood where a set held more than one case "
                    "(Gail, Lubin & Rubinstein 1981)")
        if r and r.matching:
            out += f"; the matching factors ({_listed(list(r.matching))}) were not estimated"
    else:
        adj = (f", adjusted for the matching factors ({_listed(list(r.matching))})"
               if r and r.matching else ", adjusted for the matching factors"
               if sampling == "frequency_matched" else "")
        out = (f"Odds ratios were estimated by logistic regression{adj} (Breslow & Day 1980); "
               "the intercept reflects the sampling of cases and controls and was not interpreted "
               "(Prentice & Pyke 1979)")
        if r and r.weighted:
            out += (", weighted to the surveyed population with standard errors by linearization "
                    "over the strata and primary sampling units")
    return out + (". Prevalence and absolute risks were not estimated, because cases and controls "
                  "were sampled separately.")


def risks_sentence(result: OutOfFoldRisks) -> str:
    """The methods sentence of recalibrated risks under Predict."""
    if isinstance(result.prevalence, Mapping):
        target = "the stated prevalence within each matching stratum"
    else:
        target = f"a population prevalence of {100 * result.prevalence:.3g}%"
    return (f"Because cases and controls were sampled separately, predicted risks were "
            f"recalibrated to {target} by the prior correction (King & Zeng 2001), the case share "
            f"of each training fold setting the offset; matched sets were kept whole within folds.")


def _clause(run: Mapping[str, Any]) -> str | None:
    result = run.get("case_control")
    return case_control_sentence(result) if isinstance(result, CaseControlOdds) else None


# ── the contracts ────────────────────────────────────────────────────────────


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "case_control_effects" in CONTRACTS:
        return
    here = "turbotab.core.methods.case_control"
    risks_refused = ("Not offered under Estimate an effect: a case-control sample gives no "
                     "absolute risk there; its odds ratios are its effects")

    register_contract(MethodContract(
        key="case_control_effects",
        label="Odds ratios from a case-control sample (unmatched, frequency-matched or matched sets)",
        slot="model", scope="model", package="CASECONTROL", run_order=1.5,
        scope_note=("The outcome model itself: it reads the outcome of every analyzed row. Under "
                    "Estimate it is fitted once on the analysis rows; under Predict the model it "
                    "fits is trained inside each training fold, matched sets kept whole."),
        needs=("a yes/no outcome whose cases and controls were sampled separately",
               "the sampling: unmatched, frequency-matched (with its matching factors) or "
               "individually matched (with the matched-set column)",
               "one or more exposures; optional: further adjustment columns, the survey design"),
        question="How are odds ratios estimated from these cases and controls?",
        options=(
            ContractOption(
                "unconditional", "Logistic regression adjusted for the matching factors",
                "The standard analysis of an unmatched or frequency-matched case-control study "
                "(Breslow & Day 1980; Prentice & Pyke 1979; Hosmer, Lemeshow & Sturdivant 2013)",
                {"inference": "Sound for an unmatched or frequency-matched sample: its slopes are "
                              "the population's log odds ratios; the intercept is set by the "
                              "sampling and is not a risk",
                 "prediction": "Sound as the model trained in each fold; its probabilities become "
                               "risks only through the prior correction (case_control_risks)"},
                {"inference": "recommended", "prediction": "available"},
                {"inference": 0, "prediction": 0}),
            ContractOption(
                "conditional", "Conditional logistic regression within the matched sets",
                "The standard analysis of individually matched sets (Breslow & Day 1980; Pearce "
                "2016); survival::clogit in R",
                {"inference": "Sound for individually matched sets, any number of controls per "
                              "case; the matching factors' own odds ratios are not estimated",
                 "prediction": "Sound only to rank a case against its own controls; it gives no "
                               "risk, since every set's intercept is removed"},
                {"inference": "recommended", "prediction": "available"},
                {"inference": 1, "prediction": 1}),
        ),
        storyboard=("read which rows are cases and which controls", "with matched sets: keep the "
                    "sets that hold both a case and a control", "fit the logistic model (adjusted "
                    "for the matching factors), or the conditional model within sets",
                    "report each exposure's odds ratio with its 95% interval",
                    "withhold prevalence and absolute risk, with their reason"),
        relations=(
            Relation("conflicts", "prevalence_not_estimable",
                     "prevalence, absolute risk and a risk difference are refused: the sampling "
                     "set how many cases and controls the table holds", rung="refused",
                     exits=("odds ratios", "risks recalibrated to a known prevalence (Predict, "
                            "unmatched)"),
                     condition="any case-control sampling", enforced_by=f"{here}:eligible",
                     id="no_prevalence_or_risk"),
            Relation("implies", "matching_factors_in_model",
                     "the matching factors are in the logistic model, and their own odds ratios "
                     "are not reported", when=("unconditional",),
                     condition="frequency matching (a frequency-matched sample with none named is "
                               "refused)", enforced_by=f"{here}:odds_ratios",
                     id="frequency_matching_adjusted"),
            Relation("implies", "conditional_within_sets",
                     "individually matched sets are analyzed within each set; a set without both "
                     "a case and a control leaves, and a set with several cases takes the exact "
                     "conditional likelihood", when=("conditional",),
                     condition="individually matched sets", enforced_by=f"{here}:odds_ratios",
                     id="matched_sets_conditional"),
            Relation("conflicts", "matched_on_not_estimable",
                     "a column the same for every member of each set has no odds ratio in the "
                     "conditional analysis", when=("conditional",), rung="refused",
                     exits=("leave it out",), condition="a column constant within every set",
                     enforced_by=f"{here}:odds_ratios", id="matched_on"),
            Relation("conflicts", "separation",
                     "an exposure that separates cases from controls (or each case from its own "
                     "controls) has an infinite odds ratio and is refused", rung="refused",
                     exits=("leave it out, or merge its rare levels",),
                     condition="complete separation", enforced_by=f"{here}:odds_ratios",
                     id="separated"),
            Relation("implies", "population_design",
                     "an unmatched or frequency-matched sample under the population answer takes "
                     "design-based logistic regression, t intervals on the PSUs minus the strata",
                     when=("unconditional",), purposes=("inference",),
                     condition="a survey design answered as the surveyed population",
                     enforced_by=f"{here}:odds_ratios", id="design_based"),
            Relation("conflicts", "matched_sets_not_weighted",
                     "matched sets under the population answer are refused: conditional logistic "
                     "regression has no design-based form in common use", when=("conditional",),
                     purposes=("inference",), rung="refused",
                     exits=("the sample-only attestation", "analyze as frequency-matched with "
                            "every matching factor named"),
                     condition="the surveyed population and matched sets",
                     enforced_by=f"{here}:odds_ratios", id="matched_design_refused"),
            Relation("precedes", "case_control_risks",
                     "the model is fitted before its predictions are recalibrated",
                     purposes=("prediction",), id="before_risks"),
        ),
        sources=(SOURCES["breslow_day"], SOURCES["prentice_pyke"], SOURCES["gail"],
                 SOURCES["pearce"], SOURCES["hosmer"], SOURCES["rothman"], SOURCES["lumley"]),
        clause=_clause, sentence=f"{here}:case_control_sentence"))

    register_contract(MethodContract(
        key="case_control_risks",
        label="Risks from a model trained on a case-control sample",
        slot="model", scope="model", package="CASECONTROL", run_order=1.6,
        scope_note=("Learns the case share of the rows the model is trained on, so under Predict it "
                    "is fitted inside each training fold with the model; the stated population "
                    "prevalence is a number the person gives, not learned from any row."),
        needs=("a classifier trained on an unmatched or frequency-matched case-control sample",
               "the population prevalence of the outcome (one per matching stratum under "
               "frequency matching)"),
        question="How do the model's predictions become risks?",
        options=(
            ContractOption(
                "prior_correction", "Recalibrated to the population prevalence (prior correction)",
                "King & Zeng 2001's prior correction; Prentice & Pyke 1979 for why only the "
                "intercept moves",
                {"prediction": "Sound when cases and controls were drawn by the outcome alone (or "
                               "per matching stratum, with each stratum's prevalence) and the "
                               "prevalence is known",
                 "inference": risks_refused},
                {"prediction": "recommended", "inference": "not_offered"},
                {"prediction": 0, "inference": 0}),
            ContractOption(
                "ranking_only", "No risks: how well the model separates cases from controls",
                "The case-control sample's own probabilities are not risks (Rothman, Greenland & "
                "Lash, Modern Epidemiology, ch. 8)",
                {"prediction": "Sound without the prevalence; under frequency matching the "
                               "matching factors' share of the ranking is understated (Janes & "
                               "Pepe 2008)",
                 "inference": risks_refused},
                {"prediction": "available", "inference": "not_offered"},
                {"prediction": 1, "inference": 1}),
            ContractOption(
                "within_set_ranking", "Rank each case against its own matched controls",
                "The conditional model's linear predictor within each set (Breslow & Day 1980; "
                "Janes & Pepe 2008)",
                {"prediction": "Sound for individually matched sets, the only ranking they support",
                 "inference": risks_refused},
                {"prediction": "available", "inference": "not_offered"},
                {"prediction": 2, "inference": 2}),
        ),
        storyboard=("in each training fold, fit the model", "learn that fold's case share",
                    "shift each held-out prediction's log odds by the sample's odds over the "
                    "population's", "report risks (or, without the prevalence, the ranking only)"),
        relations=(
            Relation("conflicts", "no_risk_from_matched_sets",
                     "risks for individually matched sets are refused: the conditional model has "
                     "no baseline risk, and matched controls are no random sample of non-cases",
                     when=("prior_correction",), purposes=("prediction",), rung="refused",
                     exits=("rank each case against its own matched controls", "odds ratios"),
                     condition="individually matched sets", enforced_by=f"{here}:eligible",
                     id="matched_risk_refused"),
            Relation("conflicts", "prevalence_needed",
                     "risks without the population prevalence are refused", when=("prior_correction",),
                     purposes=("prediction",), rung="refused",
                     exits=("give the population prevalence", "the ranking only"),
                     condition="no population prevalence stated", enforced_by=f"{here}:eligible",
                     id="needs_prevalence"),
            Relation("conflicts", "prevalence_per_stratum",
                     "one prevalence under frequency matching is refused: the controls' sampling "
                     "differs by stratum", when=("prior_correction",), purposes=("prediction",),
                     rung="refused", exits=("the prevalence within each matching stratum",
                                            "the ranking only"),
                     condition="frequency matching with a single prevalence",
                     enforced_by=f"{here}:eligible", id="stratum_prevalence"),
            Relation("implies", "offset_in_fold",
                     "the case share that sets the offset is learned from each training fold",
                     when=("prior_correction",), purposes=("prediction",),
                     condition="always", enforced_by=f"{here}:PriorCorrected", id="fold_share"),
            Relation("implies", "sets_whole_in_folds",
                     "matched sets (and sets sharing a person) stay whole inside folds",
                     purposes=("prediction",), condition="a matched-set column",
                     enforced_by=f"{here}:case_control_folds", id="sets_whole"),
            Relation("conflicts", "weights_correct_twice",
                     "survey weights beside the prior correction are refused: both would correct "
                     "the sampling", when=("prior_correction",), purposes=("prediction",),
                     rung="refused", exits=("recalibrate without the survey weights",
                                            "the ranking only"),
                     condition="a survey design", enforced_by=f"{here}:predicted_risks",
                     id="no_weights"),
            Relation("conflicts", "risks_not_under_estimate",
                     "not offered under Estimate an effect: no absolute risk from a case-control "
                     "sample", purposes=("inference",), rung="refused",
                     exits=("odds ratios",), condition="the Estimate an effect goal",
                     enforced_by=f"{here}:eligible", id="estimate_refused"),
        ),
        sources=(SOURCES["king_zeng"], SOURCES["prentice_pyke"], SOURCES["janes_pepe"],
                 SOURCES["breslow_day"], SOURCES["rothman"]),
        sentence=f"{here}:risks_sentence"))


_register_contracts()

__all__ = ["ASKS", "CaseControlOdds", "CaseControlRefused", "ConditionalFit", "Eligibility",
           "GOALS", "MatchedScores", "OddsRatio", "OutOfFoldRisks", "PriorCorrected", "SAMPLINGS",
           "Signal", "case_control_folds", "case_control_sentence", "conditional_logistic",
           "eligible", "matched_set_scores", "offered_first", "matched_sets_signal", "odds_ratios",
           "predicted_risks", "prevalence_signal", "prior_correction_offset", "recalibrate",
           "risks_sentence", "set_column_signals"]
