"""Leave-one-site-out (internal–external) validation, pooled across sites: an engine method core (T3).

Under Predict, with a declared site, cluster or batch column, each site in turn is held out: the
whole pipeline is fitted on every other site and scored on the held-out one. Riley et al. (BMJ
2016;353:i3140), on internal–external cross-validation: "all but one of the studies are used for
model development, with the remaining study used for external validation. This process is repeated
a further k −1 times, on each occasion omitting a different study"; and, for reporting, "quantify the
between-cluster heterogeneity in performance, for example, via a random-effects meta-analysis and
deriving 95% prediction intervals for calibration and discrimination performance in a new cluster".

**In each held-out site** (:mod:`turbotab.core.models.performance`, the engine's own scorers):

* discrimination, for a yes/no outcome: the C statistic (the AUC) with DeLong's standard error;
* calibration: calibration-in-the-large (the intercept of the logistic recalibration with the
  linear predictor as an offset; for a number, the mean of outcome − prediction) and the
  calibration slope (the coefficient of the linear predictor; for a number, of the prediction), each
  with its Wald standard error, and the observed and expected event rates or means;
* for a number, R² against the mean of the rows the model was fitted on, and the RMSE (reported per
  site, not pooled).

When rows repeat within a unit the site's intervals are clustered by the unit, as the fit's are.

**Pooled across sites** by a random-effects meta-analysis on the scales Snell et al. (Stat Methods
Med Res 2018;27:3505) found nearer between-study normality: "Normality was vastly improved when
using the logit transformation for the C-statistic … and therefore we recommend these scales to be
used for meta-analysis"; "a normal between-study distribution was usually reasonable for the
calibration slope and calibration-in-the-large". The logit C statistic's standard error is
SE(C)/(C(1 − C)) (the delta method), and the summary and its intervals are transformed back.

* **The between-site variance** τ² by restricted maximum likelihood (Fisher scoring, as R's
  metafor ``rma(method = "REML")`` iterates it, to 10⁻¹²; Viechtbauer 2010), the default; or by
  DerSimonian–Laird's moments (``method="dl_wald"``), the fit's own summary.
* **The interval of the summary** by Hartung–Knapp–Sidik–Jonkman, as Snell et al. recommend
  "particularly when the number of studies, k, is small": μ̂ ± t(k − 1)·q·σ̂_μ, with
  q² = Σŵᵢ(yᵢ − μ̂)²/(k − 1), σ̂_μ = 1/√Σŵᵢ, ŵᵢ = 1/(Sᵢ² + τ̂²) (metafor's ``test = "knha"``); with
  ``dl_wald``, μ̂ ± z·σ̂_μ.
* **The prediction interval** for a new site, Higgins, Thompson & Spiegelhalter's (2009):
  μ̂ ± t(k − 2)·√(τ̂² + σ̂_μ²) (metafor's ``predtype = "Riley"``), from three sites.
* **Heterogeneity**: I² = τ̂²/(τ̂² + s²), s² = (k − 1)Σwᵢ/((Σwᵢ)² − Σwᵢ²) the typical within-site
  variance (wᵢ = 1/Sᵢ²), as metafor computes it for any τ² estimator; Cochran's Q on k − 1 degrees of
  freedom beside it. Riley et al.: "Rather than focusing on I², which might be misleading when the
  study sample sizes are large, … the extent of heterogeneity in model performance is better
  quantified by a 95% prediction interval", so the prediction interval is said first.

**Everything learned is fitted inside each training fold**: the pipeline is made fresh for every
held-out site and fitted on the other sites' rows only (:func:`turbotab.core.models.metrics.
cross_validate`, the fit's own loop), so no row of a held-out site informs its own score.

**Refusals, each with its exits.** Fewer than three sites (no prediction interval, and two sites
are two external validations, not a spread); a unit (a person) whose rows lie in more than one site,
since the model trained without a site would still have seen that person; a survey design under
the surveyed-population answer, since a site's design-weighted performance has no design-based
pooling here (the exits: score the sites on the rows as sampled, or the design-based
cross-validation the fit offers); an outcome other than a number or yes/no. A site that cannot be
scored (one outcome class only, too few rows) is listed with its reason and left out of the pooling
of that measure; a site whose rows all lack the site's value is never held out and never fitted on.

The goal that offers it: Predict. Estimate an effect and Describe fit no prediction model to carry
to a new site.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

LEVEL = 0.95
MIN_SITES = 3
METHODS: tuple[str, ...] = ("reml_hksj", "dl_wald")
GOALS: tuple[str, ...] = ("describe", "estimate", "predict")
NOT_RECORDED = "(not recorded)"
REML_TOL = 1e-12
REML_MAX = 1000

SOURCES = {
    "riley2016": "Riley et al. 2016, BMJ 353:i3140 (doi:10.1136/bmj.i3140)",
    "snell2018": "Snell et al. 2018, Stat Methods Med Res 27:3505 (doi:10.1177/0962280217705678)",
    "higgins2009": "Higgins, Thompson & Spiegelhalter 2009, J R Stat Soc A 172:137 "
                   "(doi:10.1111/j.1467-985X.2008.00552.x)",
    "viechtbauer2010": "Viechtbauer 2010, J Stat Softw 36(3) (doi:10.18637/jss.v036.i03)",
    "steyerberg2016": "Steyerberg & Harrell 2016, J Clin Epidemiol 69:245",
}

KFOLD_EXIT = {"label": "Validate by k-fold cross-validation instead", "validation": "kfold"}
SAMPLE_EXIT = {"label": "Score the sites on the rows as sampled: record the sample-only "
                        "attestation", "decision": {"kind": "set_survey", "estimand": "sample"}}
DESIGN_CV_EXIT = {"label": "Use the design-based cross-validation the fit offers", "stage": "fit"}
PREDICT_EXIT = {"label": "Ask it under Predict", "goal": "predict"}


class SiteValidationRefused(ValueError):
    """The data or the goal cannot support the validation asked for; the message says why,
    ``exits`` the ways forward."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


@dataclass(frozen=True)
class Eligibility:
    offered: bool
    says: str
    exits: tuple[dict[str, Any], ...] = ()


def eligible(goal: str) -> Eligibility:
    if goal not in GOALS:
        raise ValueError(f"goal {goal!r} is unknown")
    if goal != "predict":
        return Eligibility(False, "Leave-one-site-out validation checks how a prediction model "
                                  "carries to a site it never saw; this goal fits no prediction "
                                  "model.", (dict(PREDICT_EXIT),))
    return Eligibility(True, "Each site held out in turn, the model fitted on the others, and the "
                             "sites' scores pooled with their spread.")


# ── pooling ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pooled:
    """A random-effects summary of one measure across sites, on its reported scale."""

    measure: str
    scale: str  # identity · logit (the scale pooled on; every number here is transformed back)
    k: int
    estimate: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    pi_low: float | None = None
    pi_high: float | None = None
    tau2: float | None = None  # on the pooled scale
    i2: float | None = None  # a share, 0 to 1
    q: float | None = None
    q_df: int | None = None
    q_p: float | None = None
    se: float | None = None  # σ̂_μ on the pooled scale (the Wald one)
    se_hk: float | None = None  # q·σ̂_μ (Hartung–Knapp), when used
    method: str = ""
    note: str | None = None


def _reml_tau2(y: np.ndarray, v: np.ndarray) -> float:
    """τ² by REML, Fisher scoring with step halving (metafor's iteration), truncated at zero."""
    k = len(y)
    tau2 = max(0.0, float(np.var(y, ddof=1) - np.mean(v))) if k > 1 else 0.0
    for _ in range(REML_MAX):
        w = 1.0 / (v + tau2)
        P = np.diag(w) - np.outer(w, w) / w.sum()
        Py = P @ y
        adj = (float(Py @ Py) - float(np.trace(P))) / float((P * P).sum())
        while tau2 + adj < 0:
            adj /= 2.0
            if abs(adj) < REML_TOL:
                break
        new = max(0.0, tau2 + adj)
        if abs(new - tau2) <= REML_TOL * max(1.0, tau2):
            tau2 = new
            break
        tau2 = new
    return 0.0 if tau2 < REML_TOL else tau2


def _dl_tau2(y: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / v
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    return max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0


def pool(measure: str, estimates: Sequence[float], ses: Sequence[float], *,
         scale: Literal["identity", "logit"] = "identity", method: str = "reml_hksj",
         level: float = LEVEL) -> Pooled:
    """The random-effects summary of per-site ``estimates`` with standard errors ``ses`` (module
    docstring). On the logit scale each estimate θ becomes logit θ with SE se/(θ(1 − θ))."""
    if method not in METHODS:
        raise ValueError(f"method {method!r} is not one of {METHODS}")
    th = np.asarray(estimates, dtype=float)
    se = np.asarray(ses, dtype=float)
    keep = np.isfinite(th) & np.isfinite(se) & (se > 0)
    if scale == "logit":
        keep &= (th > 0) & (th < 1)
    th, se = th[keep], se[keep]
    k = int(len(th))
    label = ("REML, Hartung–Knapp–Sidik–Jonkman interval" if method == "reml_hksj" else
             "DerSimonian–Laird, Wald interval")
    if k < 2:
        return Pooled(measure, scale, k, None, method=label,
                      note=f"{k} site{'s' if k != 1 else ''} scored: nothing to pool.")
    if scale == "logit":
        se = se / (th * (1 - th))
        th = np.log(th / (1 - th))
    back = (lambda x: float(1 / (1 + math.exp(-x)))) if scale == "logit" else (lambda x: float(x))
    v = se ** 2
    tau2 = _reml_tau2(th, v) if method == "reml_hksj" else _dl_tau2(th, v)
    w_fe = 1.0 / v
    mu_fe = float((w_fe * th).sum() / w_fe.sum())
    q = float((w_fe * (th - mu_fe) ** 2).sum())
    s2 = (k - 1) * w_fe.sum() / (w_fe.sum() ** 2 - (w_fe ** 2).sum())
    i2 = tau2 / (tau2 + s2)
    w = 1.0 / (v + tau2)
    mu = float((w * th).sum() / w.sum())
    se_mu = math.sqrt(1.0 / w.sum())
    half_level = 0.5 + level / 2
    se_hk = None
    if method == "reml_hksj":
        q_hk = float((w * (th - mu) ** 2).sum()) / (k - 1)
        se_hk = math.sqrt(q_hk) * se_mu
        half = float(stats.t.ppf(half_level, k - 1)) * se_hk
    else:
        half = float(stats.norm.ppf(half_level)) * se_mu
    out = dict(measure=measure, scale=scale, k=k, estimate=back(mu), ci_low=back(mu - half),
               ci_high=back(mu + half), tau2=float(tau2), i2=float(i2), q=q, q_df=k - 1,
               q_p=float(stats.chi2.sf(q, k - 1)), se=se_mu, se_hk=se_hk, method=label)
    if k >= 3:
        spread = float(stats.t.ppf(half_level, k - 2)) * math.sqrt(tau2 + se_mu ** 2)
        out.update(pi_low=back(mu - spread), pi_high=back(mu + spread))
    else:
        out["note"] = "Two sites scored: no prediction interval (it needs three)."
    return Pooled(**out)


# ── per site ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Measure:
    estimate: float | None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True)
class SiteScore:
    site: str
    n: int
    n_fit: int
    n_events: int | None = None
    observed: float | None = None  # event rate or mean outcome
    expected: float | None = None  # mean predicted risk or prediction
    c_statistic: Measure | None = None
    citl: Measure | None = None  # calibration-in-the-large: log-odds (yes/no) or outcome units
    slope: Measure | None = None
    r2: float | None = None
    rmse: float | None = None
    note: str | None = None


@dataclass
class SiteValidation:
    site_column: str
    task: str
    sites: list[SiteScore]
    pooled: dict[str, Pooled]
    method: str
    left_out: int = 0  # rows with no site value
    clustered_by: str | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def forest(self, measure: str) -> dict[str, Any]:
        """One measure's rows and summary, for the forest view."""
        rows = []
        for s in self.sites:
            m = getattr(s, measure)
            rows.append({"site": s.site, "n": s.n, "estimate": None if m is None else m.estimate,
                         "low": None if m is None else m.ci_low,
                         "high": None if m is None else m.ci_high, "note": s.note})
        return {"measure": measure, "rows": rows, "pooled": asdict(self.pooled[measure])
                if measure in self.pooled else None}

    @property
    def says(self) -> str:
        return _says(self)


def _measure(interval: Any) -> Measure | None:
    if interval is None or interval.estimate is None:
        return None
    return Measure(float(interval.estimate), interval.se, interval.ci_low, interval.ci_high)


def site_validation(task: str, make: Callable[[], Any], X: Any, y: Any, sites: Any, *,
                    site_column: str = "site", units: Any = None, unit_name: str | None = None,
                    design: Any = None, goal: str = "predict", method: str = "reml_hksj",
                    fit: Callable[[Any, Any, Any, np.ndarray], Any] | None = None,
                    before_fold: Callable[[int, int], None] | None = None) -> SiteValidation:
    """Hold each level of ``sites`` out in turn, fit ``make()`` on the others (``fit`` as
    :func:`~turbotab.core.models.metrics.cross_validate` takes it), score the held-out site, and
    pool across sites (module docstring). ``units`` names each row's unit when rows repeat;
    ``design`` is the survey design when the population answer was given (refused). Raises
    :class:`SiteValidationRefused` with its exits."""
    from turbotab.core.models import performance as perf
    from turbotab.core.models.metrics import cross_validate

    e = eligible(goal)
    if not e.offered:
        raise SiteValidationRefused(e.says, e.exits)
    if task not in ("binary", "regression"):
        raise SiteValidationRefused(
            "Pooling a site's discrimination and calibration is defined here for a numeric or a "
            "yes/no outcome; this one is " + {"multiclass": "one of several categories (a multiclass "
                                              "outcome)",
                                              "ordinal": "ordered categories (an ordinal outcome)",
                                              "time_to_event": "a time to an event (a survival "
                                                               "outcome)"}.get(task, task) + ".",
            ({"label": "Use the internal–external validation of the fit, which pools its primary "
                       "score", "validation": "internal_external"},))
    if design is not None:
        raise SiteValidationRefused(
            "Under the surveyed-population answer each site's score would have to be "
            "design-weighted, and pooling design-weighted site scores is not available here.",
            (SAMPLE_EXIT, DESIGN_CV_EXIT))
    if method not in METHODS:
        raise ValueError(f"method {method!r} is not one of {METHODS}")
    yv = np.asarray(y)
    n = len(yv)
    raw = pd.Series(np.asarray(sites, dtype=object))
    if len(raw) != n:
        raise ValueError("sites must name every row's site")
    present = raw.notna().to_numpy()
    labels = raw.where(raw.notna(), NOT_RECORDED).astype(str).to_numpy()
    levels = sorted(set(labels[present].tolist()))
    if len(levels) < MIN_SITES:
        raise SiteValidationRefused(
            f"`{site_column}` has {len(levels)} site{'s' if len(levels) != 1 else ''}; pooling "
            f"across sites with a prediction interval for a new one needs at least {MIN_SITES}.",
            (KFOLD_EXIT,))
    unit = None
    if units is not None:
        unit = pd.Series(np.asarray(units, dtype=object))
        if len(unit) != n:
            raise ValueError("units must name every row's unit")
        spread = pd.DataFrame({"u": unit[present].to_numpy(), "s": labels[present]}).dropna()
        across = spread.groupby("u")["s"].nunique()
        if (across > 1).any():
            k = int((across > 1).sum())
            raise SiteValidationRefused(
                f"{k:,} {unit_name or 'unit'}{'s' if k != 1 else ''} ha{'ve' if k != 1 else 's'} rows "
                f"in more than one site, so the model fitted without a site would still have seen "
                f"them, and the site's score would not be external.",
                ({"label": f"Correct `{site_column}` so each {unit_name or 'unit'} sits in one site"},
                 KFOLD_EXIT))
    if task == "binary":
        observed_classes = sorted(set(yv[present].tolist()), key=str)
        if len(observed_classes) != 2:
            raise SiteValidationRefused("A yes/no outcome needs both of its values among the rows.")
    notes: dict[str, str] = {}
    pairs = []
    for i, level in enumerate(levels):
        test = present & (labels == level)
        train = present & (labels != level)
        if task == "binary" and len(set(yv[train].tolist())) < 2:
            notes[level] = "Not scored: the other sites hold rows of one outcome only."
            continue
        pairs.append((i, train, test))
    cv = cross_validate(task, make, X, yv, pairs, fit=fit, before_fold=before_fold,
                        keep_predictions=True)
    scored = {levels[pair[0]]: (pair, pred, size)
              for pair, pred, size in zip(pairs, cv.predictions, cv.sizes)}
    rows: list[SiteScore] = []
    for level in levels:
        n_site = int((present & (labels == level)).sum())
        if level not in scored:
            rows.append(SiteScore(level, n_site, 0, note=notes.get(level)))
            continue
        (_, train, test), pred, size = scored[level]
        groups = None if unit is None else unit.to_numpy()[test]
        if task == "binary":
            classes = list(pred.classes or [])
            event = pred.y == classes[1]
            p = np.asarray(pred.prediction, dtype=float)[:, 1]
            events = int(event.sum())
            if events in (0, len(event)):
                rows.append(SiteScore(level, n_site, size[0], n_events=events,
                                      observed=float(event.mean()), expected=float(p.mean()),
                                      note="Not scored: this site holds rows of one outcome only."))
                continue
            auc = perf.auc_interval(event, p, groups=groups, unit=unit_name)
            cal = perf.calibration("binary", pred.y, pred.prediction, classes=classes,
                                   groups=groups, where="in this held-out site")
            rows.append(SiteScore(
                level, n_site, size[0], n_events=events, observed=float(event.mean()),
                expected=float(p.mean()), c_statistic=_measure(auc),
                citl=None if cal is None else _measure(cal.intercept),
                slope=None if cal is None else _measure(cal.slope),
                note=None if cal is not None else "Calibration not assessed: fewer than 10 rows."))
        else:
            yy = np.asarray(pred.y, dtype=float)
            yhat = np.asarray(pred.prediction, dtype=float)
            cal = perf.calibration("regression", yy, yhat, groups=groups,
                                   where="in this held-out site")
            sse = float(((yy - yhat) ** 2).sum())
            ref = float(((yy - float(pred.reference)) ** 2).sum())
            rows.append(SiteScore(
                level, n_site, size[0], observed=float(yy.mean()), expected=float(yhat.mean()),
                citl=None if cal is None else _measure(cal.intercept),
                slope=None if cal is None else _measure(cal.slope),
                r2=1 - sse / ref if ref > 0 else None, rmse=math.sqrt(sse / len(yy)),
                note=None if cal is not None else "Calibration not assessed: fewer than 10 rows."))
    pooled: dict[str, Pooled] = {}
    measures = (("c_statistic", "logit"), ("citl", "identity"), ("slope", "identity")) \
        if task == "binary" else (("citl", "identity"), ("slope", "identity"))
    for name, scale in measures:
        est = [getattr(s, name).estimate if getattr(s, name) is not None else math.nan for s in rows]
        se = [(getattr(s, name).se if getattr(s, name) is not None and getattr(s, name).se is not None
               else math.nan) for s in rows]
        pooled[name] = pool(name, est, se, scale=scale, method=method)  # type: ignore[arg-type]
    clustered = None
    if unit is not None and unit[present].duplicated().any():
        clustered = unit_name or "unit"
    return SiteValidation(site_column=site_column, task=task, sites=rows, pooled=pooled,
                          method=method, left_out=int((~present).sum()), clustered_by=clustered)


# ── words ────────────────────────────────────────────────────────────────────

MEASURE_WORDS = {"c_statistic": "the C statistic", "citl": "calibration-in-the-large",
                 "slope": "the calibration slope"}


def _n(x: float | None, places: int = 2) -> str:
    return "—" if x is None else f"{x:.{places}f}".replace("-", "−")


def _says(r: SiteValidation) -> str:
    k = sum(1 for s in r.sites if s.c_statistic is not None or s.slope is not None
            or s.citl is not None)
    head = "c_statistic" if r.task == "binary" else "slope"
    p = r.pooled.get(head)
    text = f"Each of {len(r.sites)} levels of `{r.site_column}` was held out in turn ({k} scored)."
    if p is not None and p.estimate is not None:
        text += (f" Pooled, {MEASURE_WORDS[head]} was {_n(p.estimate)} (95% CI {_n(p.ci_low)} to "
                 f"{_n(p.ci_high)})")
        if p.pi_low is not None:
            text += (f"; in a new site it would lie between {_n(p.pi_low)} and {_n(p.pi_high)} "
                     f"(95% prediction interval)")
        text += f", I² {p.i2:.0%}." if p.i2 is not None else "."
    return text


def site_validation_sentence(result: SiteValidation | None = None) -> str:
    """The methods sentence of leave-one-site-out validation."""
    r = result
    column = f"`{r.site_column}`" if r else "the declared site"
    what = ("discrimination (the C statistic) and calibration (calibration-in-the-large and the "
            "calibration slope)" if r is None or r.task == "binary" else
            "calibration (calibration-in-the-large and the calibration slope)")
    sentence = (f"Internal–external validation held out each level of {column} in turn, fitting "
                f"the whole pipeline on the remaining sites and assessing {what} in the held-out "
                f"site (Riley et al. 2016).")
    if r is None or r.method == "reml_hksj":
        sentence += (" Site estimates were pooled by random-effects meta-analysis (REML, with "
                     "Hartung–Knapp–Sidik–Jonkman confidence intervals), the C statistic on the "
                     "logit scale (Snell et al. 2018), with 95% prediction intervals for a new site "
                     "(Higgins, Thompson & Spiegelhalter 2009) and I².")
    else:
        sentence += (" Site estimates were pooled by DerSimonian–Laird random-effects meta-analysis "
                     "with Wald confidence intervals, the C statistic on the logit scale (Snell et "
                     "al. 2018), with 95% prediction intervals for a new site (Higgins, Thompson & "
                     "Spiegelhalter 2009) and I².")
    return sentence


# ── the contract ─────────────────────────────────────────────────────────────


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "site_validation" in CONTRACTS:
        return
    here = "turbotab.core.methods.site_validation"
    prediction = ("prediction",)
    not_inference = ("Not offered under Estimate an effect: it validates a prediction model at a "
                     "site it never saw")

    register_contract(MethodContract(
        key="site_validation", label="Leave-one-site-out validation, pooled across sites",
        slot="evaluation", scope="training_fold", package="SITEVAL", run_order=6.5,
        decision="set_validation", place="11 · Tuning and comparison",
        scope_note=("Each held-out site is scored by a pipeline made fresh and fitted on the other "
                    "sites' rows only, so everything learned is learned inside the training fold; "
                    "the pooling reads only the sites' scores."),
        needs=("a declared site, cluster or batch column with at least three levels",
               "a numeric or yes/no outcome", "the whole pipeline, refitted per site",
               "optional: the unit when rows repeat (each in one site)"),
        question="How are the sites' scores pooled?",
        options=(
            ContractOption(
                "reml_hksj", "REML random effects, Hartung–Knapp–Sidik–Jonkman interval",
                "Snell et al. 2018 (REML, HKSJ, logit C); metafor's rma (Viechtbauer 2010)",
                {"prediction": "Sound with few sites: the interval allows for τ² being estimated; "
                               "the prediction interval says where a new site would land",
                 "inference": not_inference},
                {"prediction": "recommended", "inference": "not_offered"},
                {"prediction": 0, "inference": 0}),
            ContractOption(
                "dl_wald", "DerSimonian–Laird random effects, Wald interval",
                "The customary random-effects summary; the fit's own internal–external summary",
                {"prediction": "Sound with many sites; with few, its interval is too narrow",
                 "inference": not_inference},
                {"prediction": "available", "inference": "not_offered"},
                {"prediction": 1, "inference": 1}),
        ),
        storyboard=("hold one site out", "fit the whole pipeline on the other sites",
                    "score the held-out site: C statistic, calibration-in-the-large, slope",
                    "repeat for every site", "pool the sites, with the prediction interval and I²"),
        relations=(
            Relation("implies", "fitted_in_fold",
                     "the pipeline is made fresh for each held-out site and fitted on the other "
                     "sites only", purposes=prediction, condition="always",
                     enforced_by=f"{here}:site_validation", id="in_fold"),
            Relation("implies", "logit_c_scale",
                     "the C statistic is pooled on the logit scale and transformed back; the "
                     "calibration measures on their own", purposes=prediction,
                     condition="a yes/no outcome", enforced_by=f"{here}:pool", id="pooling_scales"),
            Relation("implies", "prediction_interval",
                     "each pooled measure carries a 95% prediction interval for a new site, said "
                     "before I²", purposes=prediction, condition="three or more sites scored",
                     enforced_by=f"{here}:pool", id="new_site_interval"),
            Relation("implies", "unscored_site_listed",
                     "a site with one outcome class only, or too few rows, is listed with its "
                     "reason and left out of that measure's pooling", purposes=prediction,
                     condition="a site that cannot be scored", enforced_by=f"{here}:site_validation",
                     id="unscored_site"),
            Relation("implies", "clustered_site_intervals",
                     "each site's intervals are clustered by the unit", purposes=prediction,
                     condition="rows repeat within a unit", enforced_by=f"{here}:site_validation",
                     id="clustered_sites"),
            Relation("conflicts", "too_few_sites",
                     "refused with fewer than three sites", purposes=prediction, rung="refused",
                     exits=("k-fold cross-validation",), condition="fewer than three sites",
                     enforced_by=f"{here}:site_validation", id="min_sites"),
            Relation("conflicts", "unit_in_two_sites",
                     "refused when a unit's rows lie in more than one site", purposes=prediction,
                     rung="refused", exits=("correct the site column", "k-fold cross-validation"),
                     condition="a unit in more than one site", enforced_by=f"{here}:site_validation",
                     id="unit_across_sites"),
            Relation("conflicts", "population_design",
                     "refused under the surveyed-population answer: design-weighted site scores "
                     "are not pooled here", purposes=prediction, rung="refused",
                     exits=("the sample-only attestation", "design-based cross-validation"),
                     condition="a survey design answered as the surveyed population",
                     enforced_by=f"{here}:site_validation", id="survey_refused"),
            Relation("conflicts", "number_or_yes_no",
                     "refused for a multiclass, ordinal or time-to-event outcome",
                     purposes=prediction, rung="refused",
                     exits=("the fit's internal–external validation of its primary score",),
                     condition="another outcome type", enforced_by=f"{here}:site_validation",
                     id="task_refused"),
            Relation("conflicts", "not_an_estimate",
                     "not offered under Estimate an effect", purposes=("inference",),
                     rung="refused", exits=("ask it under Predict",),
                     condition="the Estimate an effect goal", enforced_by=f"{here}:eligible",
                     id="inference_refused"),
        ),
        sources=tuple(SOURCES.values()),
        sentence=f"{here}:site_validation_sentence"))


_register_contract()

__all__ = ["Eligibility", "Measure", "Pooled", "SiteScore", "SiteValidation",
           "SiteValidationRefused", "eligible", "pool", "site_validation",
           "site_validation_sentence"]
