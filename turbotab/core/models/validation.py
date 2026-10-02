"""Internal validation beyond one k-fold run (AUDIT_REPORT §5 WP9: ME-11, E16).

One k-fold run, or one random split, is the weakest internal validation. Steyerberg et al. (J Clin
Epidemiol 2001;54:774): "Internal validity could best be estimated with bootstrapping, which
provided stable estimates with low bias. We conclude that split-sample validation is inefficient,
and recommend bootstrapping for estimation of internal validity". Steyerberg (J Clin Epidemiol
2018;103:131): "In small samples, cross-validation and bootstrapping are more efficient approaches.
In conclusion, random data splitting should be abolished for validation of prediction models."

**Repeated k-fold** draws the folds ``r`` times (the split stage's ``fold_r1`` … columns) and
averages the repeats' estimates (:mod:`turbotab.core.models.metrics`), so the score no longer rests
on one partition; the repeats' spread is reported beside it.

**Bootstrap optimism correction** (Harrell, Lee & Mark, Stat Med 1996;15:361; the "enhanced
bootstrap" of Collins et al., BMJ 2024;384:e074819, Box 2), with the whole pipeline refit in every
resample — imputation, energy adjustment, scaling and any inner tuning:

1. fit on every training row and score those rows: the *apparent* performance;
2. draw a bootstrap resample of whole units (rows when nothing repeats), fit the pipeline on it,
   and score that model on the resample (its apparent performance) and on the original rows;
3. the resample's *optimism* is the difference; repeat ``B`` times and average;
4. the *optimism-corrected* performance is the apparent performance minus the mean optimism.

Resample b draws ``rng.integers(0, U, U)`` from ``numpy.random.default_rng(seed)`` (U units, in
order of first appearance), one draw per resample in turn, so an independent implementation can
replay the same resamples. A resample missing a class cannot be fit; it is skipped (its draw still
made) and counted, and fewer than :data:`MIN_OK_SHARE` of ``B`` succeeding refuses the estimate.
R² on a resample is measured against the resample's mean, which is the mean of the rows that model
was fit on, as everywhere else in the app. Calibration intercept and slope are corrected the same
way (for logistic regression the apparent values are exactly 0 and 1).

**Internal–external validation** (Collins et al. 2024, Box 4: "the performance of this model
(developed on all the data) is then examined using cross validation by cluster, where a cluster is
held out … and the same model building steps … are applied to the remaining clusters"; "The results
can then be presented in a forest plot … and a summary estimate calculated using (random effects)
meta-analysis"): one fold per level of the cluster column, each cluster's score with its interval,
and a random-effects summary with the between-cluster spread
(:func:`turbotab.core.models.performance.random_effects`).

**Which comes first** (:func:`validation_plan`, read by the seal plan). Under prediction, below
:data:`RESAMPLE_BELOW` units the resampling options lead, bootstrap optimism correction first and
repeated k-fold second, and a holdout stays on offer with its tension in one line. The number is
Harrell's ("Split-Sample Model Validation", fharrell.com/post/split-val, 23 January 2017): "Data
splitting is an unstable method for validating models or classifiers, especially when the number
of subjects is less than about 20,000 (fewer if signal:noise ratio is high)"; and "To be as good
as the bootstrap, about 100 repeats of 10-fold cross-validation are required", hence the
bootstrap first. The clinical pack (CLINICAL_SURVEY_PACK §A5.5) gives the same order: "Bootstrap optimism
correction is the recommended default …; repeated k-fold CV is acceptable" (a CONVENTION). At or
above it, one k-fold run leads and internal–external validation follows: "In large samples,
interest should shift to assessment of heterogeneity in model performance across settings"
(Steyerberg 2018). Under inference the scores describe fit, not the estimate, and one k-fold run
leads. When the answers ask for time order the folds forward-chain, and the options that would
ignore time (bootstrap resamples, repeated draws) come last.
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import Validation
from turbotab.core.models import performance as perf
from turbotab.core.models.metrics import METRICS, PRIMARY, CrossValidated, higher_is_better, predict, score

MIN_OK_SHARE = 0.9
CALIBRATION_KEYS = ("calibration_intercept", "calibration_slope")
# Below this many units, under prediction, resampling ranks first (module docstring: Harrell 2017).
RESAMPLE_BELOW = 20_000
RESAMPLE_SOURCE = ("Harrell 2017, Split-sample model validation: data splitting is unstable below "
                   "about 20,000 subjects")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class OptimismEstimate(_Model):
    apparent: float | None
    optimism: float | None  # mean over the successful resamples
    corrected: float | None  # apparent − optimism
    optimism_sd: float | None = None  # spread of the resamples' optimism


class Optimism(_Model):
    """Harrell's bootstrap optimism correction of one family (module docstring)."""

    n_boot: int
    n_ok: int
    failed: int
    seconds: float
    resampled: str  # "rows" or "<unit> units"
    estimates: dict[str, OptimismEstimate]
    refused: str | None = None


def _calibration_points(task: str, y: np.ndarray, prediction: np.ndarray,
                        classes: Sequence[Any] | None) -> dict[str, float | None]:
    """Calibration intercept and slope point estimates (no smoothing, no intervals): cheap enough
    to compute on every resample."""
    if task == "regression":
        yy = y.astype(float)
        yhat = np.asarray(prediction, dtype=float)
        var = float(np.var(yhat))
        slope = float(np.cov(yhat, yy, ddof=0)[0, 1] / var) if var > 0 else None
        return {"calibration_intercept": float(np.mean(yy - yhat)), "calibration_slope": slope}
    if task != "binary":
        return {}
    event = (y == list(classes or [])[1]).astype(float)
    if event.min() == event.max():
        return {"calibration_intercept": None, "calibration_slope": None}
    lp = perf._logit(prediction[:, 1])
    n = len(event)
    a = perf.logistic_fit(np.ones((n, 1)), event, offset=lp)
    b = perf.logistic_fit(np.column_stack([np.ones(n), lp]), event)
    return {"calibration_intercept": float(a[0][0]) if a else None,
            "calibration_slope": float(b[0][1]) if b else None}


def performance_points(task: str, model: Any, X: Any, y: Any,
                       reference: float | None) -> dict[str, float | None]:
    """Every metric of ``task`` and the calibration intercept and slope, as points."""
    y = np.asarray(y)
    out: dict[str, float | None] = dict(score(task, model, X, y, reference=reference))
    if task in ("regression", "binary"):
        classes = None if task == "regression" else list(model.classes_)
        out.update(_calibration_points(task, y, predict(task, model, X), classes))
    return out


def optimism_bootstrap(task: str, make: Callable[[], Any],
                       fit: Callable[[Any, Any, Any, np.ndarray], Any], X: Any, y: Any, *,
                       n_boot: int, seed: int = 0, groups: Any = None, unit: str | None = None,
                       final: Any = None,
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Optimism:
    """Bootstrap optimism correction of the pipeline ``make()`` (module docstring).

    ``fit(model, X_b, y_b, units_b)`` fits a fresh pipeline on a resample; ``units_b`` labels each
    resampled row with its original unit, so every copy of a unit keeps to one side of any inner
    split. ``final`` is the pipeline already fit on every row (else one is fit).
    """
    import pandas as pd

    started = time.perf_counter()
    y = np.asarray(y)
    n = len(y)
    if groups is None:
        unit_of = np.arange(n)
        rows_of = [np.asarray([i]) for i in range(n)]
        labels = np.arange(n).astype(str).astype(object)
        resampled = "rows"
    else:
        labels = np.asarray([f"__missing_{i}" if g is None or (isinstance(g, float) and math.isnan(g))
                             else str(g) for i, g in enumerate(np.asarray(groups, dtype=object))],
                            dtype=object)
        unit_of = pd.factorize(labels)[0]
        order = np.argsort(unit_of, kind="stable")
        bounds = np.searchsorted(unit_of[order], np.arange(unit_of.max() + 2))
        rows_of = [order[bounds[u]:bounds[u + 1]] for u in range(unit_of.max() + 1)]
        resampled = f"{unit or 'unit'} units"
    n_units = len(rows_of)
    reference = float(np.mean(y.astype(float))) if task == "regression" else None
    if final is None:
        final = fit(make(), X, y, labels)
    apparent = performance_points(task, final, X, y, reference)
    keys = list(apparent)
    need = len(np.unique(y)) if task != "regression" else 1
    rng = np.random.default_rng(seed)
    optimism: dict[str, list[float]] = {k: [] for k in keys}
    failed = 0
    for b in range(int(n_boot)):
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        draw = rng.integers(0, n_units, n_units)
        rows = np.concatenate([rows_of[u] for u in draw])
        yb = y[rows]
        if task != "regression" and len(np.unique(yb)) < need:
            failed += 1
            continue
        Xb = X.iloc[rows] if hasattr(X, "iloc") else np.asarray(X)[rows]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fit(make(), Xb, yb, labels[rows])
                ref_b = float(np.mean(yb.astype(float))) if task == "regression" else None
                own = performance_points(task, model, Xb, yb, ref_b)
                test = performance_points(task, model, X, y, ref_b)
        except Exception:  # noqa: BLE001 - a resample that cannot be fit is counted, not fatal
            failed += 1
            continue
        for k in keys:
            a, t = own.get(k), test.get(k)
            if a is not None and t is not None and math.isfinite(a) and math.isfinite(t):
                optimism[k].append(a - t)
        if progress is not None:
            progress(b + 1, int(n_boot))
    n_ok = int(n_boot) - failed
    refused = None
    if n_ok < MIN_OK_SHARE * int(n_boot):
        refused = (f"Only {n_ok:,} of {int(n_boot):,} resamples could be fit (a resample needs every "
                   f"class); the correction needs {MIN_OK_SHARE:.0%} of them.")
    estimates: dict[str, OptimismEstimate] = {}
    for k in keys:
        app = apparent.get(k)
        values = optimism[k]
        mean = float(np.mean(values)) if values and refused is None else None
        estimates[k] = OptimismEstimate(
            apparent=app if app is not None and math.isfinite(app) else None,
            optimism=mean,
            corrected=(app - mean) if mean is not None and app is not None else None,
            optimism_sd=float(np.std(values, ddof=1)) if len(values) > 1 and refused is None else None)
    return Optimism(n_boot=int(n_boot), n_ok=n_ok, failed=failed,
                    seconds=round(time.perf_counter() - started, 3), resampled=resampled,
                    estimates=estimates, refused=refused)


# ── which validation comes first ─────────────────────────────────────────────


class ValidationOption(_Model):
    """One way the training rows can validate the models, as the split question offers it."""

    validation: Validation
    label: str
    measures: str  # what it gives, in one line
    cost: str  # what it costs, in refits
    needs_cluster: bool = False  # internal–external: the user names the column whose levels fold


class ValidationPlan(_Model):
    """The validation options in the order to offer them, and why (module docstring)."""

    options: list[ValidationOption]
    resampling_first: bool  # bootstrap optimism correction and repeated k-fold lead
    reason: str
    holdout_note: str | None  # the holdout's tension in one line, when resampling leads
    below: int = RESAMPLE_BELOW
    source: str = RESAMPLE_SOURCE


def validation_plan(purpose: str | None, n_units: int, *, time_ordered: bool = False,
                    folds: int = 5, repeats: int | None = None, n_boot: int | None = None,
                    unit: str = "rows") -> ValidationPlan:
    """The validation options for this purpose and size, in order (module docstring)."""
    from turbotab.core.decisions import OPTIMISM_BOOT, REPEATS

    r = int(repeats or REPEATS)
    b = int(n_boot or OPTIMISM_BOOT)
    n = f"{int(n_units):,} {unit}"
    by: dict[str, ValidationOption] = {
        "bootstrap": ValidationOption(
            validation="bootstrap", label="Bootstrap optimism correction",
            measures=f"Every row trains and scores; {b} whole-pipeline refits on resamples "
                     f"estimate the optimism.",
            cost=f"about {b} refits of each family, besides one {folds}-fold run"),
        "repeated_kfold": ValidationOption(
            validation="repeated_kfold", label=f"Repeated cross-validation ({r} × {folds})",
            measures=f"The folds are drawn {r} times, so the score no longer rests on one "
                     f"partition.",
            cost=f"{r * folds} refits of each family"),
        "kfold": ValidationOption(
            validation="kfold",
            label="Time-ordered cross-validation" if time_ordered else "Cross-validation, once",
            measures=("Each fold scored by models fit on earlier units." if time_ordered else
                      "One partition into folds: cheap, and noisier at small sizes."),
            cost=f"{folds} refits of each family"),
        "internal_external": ValidationOption(
            validation="internal_external", label="Internal–external, by cluster",
            measures="Each site, study or period scored by models fit on the others, with the "
                     "spread.",
            cost="one refit of each family per cluster", needs_cluster=True),
    }
    holdout_note = None
    if time_ordered:
        order = ["kfold", "internal_external", "repeated_kfold", "bootstrap"]
        first = False
        reason = ("The answers ask for time order, so the folds forward-chain by whole unit; "
                  "bootstrap resamples and repeated draws would ignore time, so they come last.")
    elif purpose == "inference":
        order = ["kfold", "repeated_kfold", "bootstrap", "internal_external"]
        first = False
        reason = ("Under inference the scores describe fit, not the estimate, so one "
                  "cross-validation run comes first.")
    elif n_units < RESAMPLE_BELOW:
        order = ["bootstrap", "repeated_kfold", "kfold", "internal_external"]
        first = True
        reason = (f"With {n}, below about {RESAMPLE_BELOW:,}, one split or one partition gives "
                  f"an unstable score (Harrell); resampling the whole pipeline uses every row, so "
                  f"it comes first.")
        holdout_note = (f"A holdout is a lockbox against analyst overfitting; at {n} it costs "
                        f"precision: its rows neither train the models nor enter the resampled "
                        f"score.")
    else:
        order = ["kfold", "internal_external", "repeated_kfold", "bootstrap"]
        first = False
        reason = (f"With {n}, one cross-validation run is precise; across sites or periods, "
                  f"internal–external validation shows how performance varies (Steyerberg 2018).")
    return ValidationPlan(options=[by[k] for k in order], resampling_first=first, reason=reason,
                          holdout_note=holdout_note)


# ── internal–external validation ─────────────────────────────────────────────


class ClusterScore(_Model):
    """One held-out cluster: its rows, its scores, and the primary metric's interval."""

    cluster: str
    n: int
    n_events: int | None = None  # binary: rows of the event
    scores: dict[str, float | None]
    primary: perf.Interval
    note: str | None = None


class InternalExternal(_Model):
    """Internal–external validation by ``cluster`` (module docstring)."""

    cluster: str
    metric: str
    clusters: list[ClusterScore]
    pooled: perf.Pooled
    spread: str  # one sentence: the range and the between-cluster SD


def internal_external(task: str, cv: CrossValidated, labels: Sequence[str], cluster: str, *,
                      groups: Any = None, unit: str | None = None) -> InternalExternal:
    """Each cluster's performance (``cv``'s folds are the clusters, in ``labels``' order) and a
    random-effects summary of the primary metric."""
    primary = PRIMARY[task]
    units = None if groups is None else np.asarray(groups, dtype=object)
    rows: list[ClusterScore] = []
    estimates, ses = [], []
    for (fold, scores), pred in zip(enumerate(cv.per_fold), cv.predictions):
        name = str(labels[fold]) if fold < len(labels) else str(fold)
        n_events = None
        note = None
        if task == "binary":
            n_events = int((pred.y == (pred.classes or [None, None])[1]).sum())
        found = perf.score_intervals(task, pred.y, pred.prediction, classes=pred.classes,
                                     reference=pred.reference,
                                     groups=None if units is None else units[pred.rows], unit=unit)
        interval = found.get(primary) or perf.Interval(estimate=scores.get(primary))
        clean = {k: (float(v) if v is not None and math.isfinite(float(v)) else None)
                 for k, v in scores.items()}
        if interval.estimate is None or not math.isfinite(interval.estimate):
            note = "Too few rows of one class to score this cluster."
            interval = perf.Interval(estimate=None, method=interval.method)
        else:
            estimates.append(interval.estimate)
            ses.append(interval.se if interval.se is not None else float("nan"))
        rows.append(ClusterScore(cluster=name, n=int(len(pred.y)), n_events=n_events, scores=clean,
                                 primary=interval, note=note))
    scale: Literal["identity", "logit"] = "logit" if primary == "auc" else "identity"
    pooled = perf.random_effects(primary, estimates, ses, scale=scale)
    return InternalExternal(cluster=cluster, metric=primary, clusters=rows, pooled=pooled,
                            spread=_spread_sentence(primary, cluster, estimates, pooled))


def _spread_sentence(metric: str, cluster: str, estimates: Sequence[float], pooled: perf.Pooled) -> str:
    from turbotab.core.models.metrics import LABELS

    label = LABELS.get(metric, metric)
    if not estimates:
        return f"No level of {cluster} could be scored."
    lo, hi = min(estimates), max(estimates)
    text = (f"Across {len(estimates)} levels of {cluster}, {label} ranged from {lo:.3f} to "
            f"{hi:.3f}".replace("-", "−"))
    if pooled.estimate is not None and pooled.k >= 2:
        text += f"; the random-effects summary is {pooled.estimate:.3f}"
        if pooled.ci_low is not None:
            text += f" (95% CI {pooled.ci_low:.3f} to {pooled.ci_high:.3f})"
        if pooled.pi_low is not None:
            text += (f", and a new {cluster} would score between {pooled.pi_low:.3f} and "
                     f"{pooled.pi_high:.3f} (95% prediction interval)")
    return text.replace("-", "−") + "."


# ── how far apart two families are, on these rows ────────────────────────────


class FamilyDifference(_Model):
    """Two families' primary metric, paired over the same folds, with a corrected interval."""

    a: str
    b: str
    metric: str
    difference: float | None  # a − b, signed so that positive favors a
    ci_low: float | None
    ci_high: float | None
    df: int | None
    sentence: str


def family_differences(task: str, results: dict[str, CrossValidated],
                       summaries: dict[str, dict[str, dict[str, Any]]],
                       labels: dict[str, str]) -> list[FamilyDifference]:
    """Every pair of families on the primary metric, with Nadeau & Bengio's corrected interval
    (``baseline.versus_baseline``'s), so whether two scores differ is read off the user's own
    folds rather than a rule of thumb about sample size (audit ME-11, the "below about 50 rows"
    claim)."""
    from turbotab.core.models.baseline import versus_baseline
    from turbotab.core.models.metrics import LABELS

    primary = PRIMARY[task]
    sign = 1.0 if higher_is_better(primary) else -1.0
    keys = list(results)
    out: list[FamilyDifference] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ea, eb = summaries[a][primary]["estimate"], summaries[b][primary]["estimate"]
            gain = sign * (ea - eb) if ea is not None and eb is not None else None
            v = versus_baseline(primary, [f[primary] for f in results[a].per_fold],
                                [f[primary] for f in results[b].per_fold], gain=gain,
                                test_share=results[a].test_share)
            label = LABELS.get(primary, primary)
            if v.gain is None or v.ci_low is None:
                sentence = (f"{labels[a]} and {labels[b]} cannot be compared on {label} with "
                            f"these folds.")
            else:
                inside = v.ci_low <= 0 <= v.ci_high
                ahead = labels[a] if v.gain >= 0 else labels[b]
                verdict = ("not distinguishable on these rows" if inside
                           else f"{ahead} is ahead on these rows")
                sentence = (f"{labels[a]} against {labels[b]}: {label} differs by "
                            f"{abs(v.gain):.3f} (95% interval of the difference {v.ci_low:.3f} to "
                            f"{v.ci_high:.3f}, corrected for shared training rows): "
                            f"{verdict}.").replace("-", "−")
            out.append(FamilyDifference(a=a, b=b, metric=primary, difference=v.gain,
                                        ci_low=v.ci_low, ci_high=v.ci_high, df=v.df,
                                        sentence=sentence))
    return out


__all__ = ["CALIBRATION_KEYS", "ClusterScore", "FamilyDifference", "InternalExternal",
           "MIN_OK_SHARE", "Optimism", "OptimismEstimate", "RESAMPLE_BELOW", "RESAMPLE_SOURCE",
           "ValidationOption", "ValidationPlan", "family_differences", "internal_external",
           "optimism_bootstrap", "performance_points", "validation_plan", "METRICS"]
