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

**Not for a near-interpolating learner.** A family that nearly memorizes its rows (boosted trees)
scores the original rows almost perfectly inside each resample, since about 63% of them were in it,
so the resample's "test" score is inflated and the optimism understated: the corrected score stays
too high. Coley et al., BMC Med Res Methodol 2023;23:33 (read at PMC9890785 on 2026-10-02): "While
previous literature demonstrated the validity of bootstrap optimism correction for parametric
models in small samples, this approach did not accurately validate performance of a rare-event
prediction model estimated with random forests in a large clinical dataset." The repair round's
replication with scikit-learn (``test_wp9_validation_performance.py``): boosted trees' corrected AUC
about 0.91 against a true 0.69–0.71 on fresh rows. Such a family declares
``bootstrap_optimism = False`` (``models/base.py``) and the fit keeps its cross-validated score
(:func:`not_applied`).

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

**At least 500 resamples** (MS6). Collins et al. (BMJ 2024): "Steyerberg and colleagues have shown
that the expected optimism could precisely be estimated with as few as 200 bootstraps with minor
sampling variability; with modern computational power, we generally recommend at least 500
bootstraps." The default is ``decisions.OPTIMISM_BOOT`` = 500; fewer are allowed and say so
(:func:`resample_concern`), and the shelf's measured estimate counts every refit before anything
runs.

**Comparisons** (MODELING_SEQUENCE ruling 4, MS6). Families are compared on the repeated k-fold
substrate (``folds.comparison_folds``), pair by pair, by Nadeau & Bengio's corrected resampled t as
Bouckaert & Frank (PAKDD 2004) apply it to r repeats of K folds (:func:`corrected_t`): over the rK
paired differences ``d`` of the primary, ``t = mean(d) / √((1/(rK) + n₂/n₁)·s²_d)`` on ``rK − 1``
degrees of freedom, ``n₂/n₁`` the rows scored over the rows fit. With K families there are
K(K − 1)/2 such intervals and none is corrected for their number, so they are descriptive; the
choice among the families is corrected by BBC-CV (``selection.py``).

**What a performance interval describes.** Cross-validation and the bootstrap estimate "the average
prediction error of models fit on other unseen training sets drawn from the same population" (Bates,
Hastie & Tibshirani, JASA 2023), so every performance sentence says *the expected performance of
this modeling procedure at this sample size* (:data:`PROCEDURE`), never this fitted model's. With
more candidate predictors than rows (p/n above :data:`P_OVER_N`, a convention) the naive interval
undercovers ("with n = 90 observations of p = 1000 features … intervals with desired miscoverage of
10% give 31% miscoverage"), so it is labeled :data:`TOO_NARROW` unless Bates et al.'s nested
cross-validation interval is run (:func:`nested_cv_interval`), which the fit offers with its compute
estimate.

**The nested cross-validation interval** (Bates, Hastie & Tibshirani 2023, Algorithm 1, as their R
package ``nestedcv`` computes it, MIT-licensed, github.com/stephenbates19/nestedcv ``R/core.R``):
for each of R repetitions, K folds; for each pair of folds a model fit without both scores each of
them, and a model fit without fold k scores fold k. With ``e_in`` the inner errors for fold k (the
other folds, each scored by the model fit without it and k) and ``e_out`` fold k's own,
``a_k = mean(e_in) − mean(e_out)`` and ``b_k = var(e_out)/|I_k|``; the interval's inflation is
``√max(0, mean(a² − b)) / (sd(e)/√n_sub)`` (``e`` every inner error, ``n_sub = ⌊n(K − 1)/K⌋``),
held between 1 and √K; the point estimate is ``mean(e) − bias``, the bias ``(mean(e) −
mean(CV)) · (1 + ((K − 2)/K)^1.5)`` from ``⌈R/5⌉`` plain cross-validation runs; and the interval is
``estimate ± z · sd(e)/√n · inflation``. It needs a per-row loss, so it covers the strictly proper
primaries (squared error, log loss, the ranked probability score, the Brier score at a horizon).
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
RECOMMENDED_BOOT = 500  # Collins et al. 2024: "at least 500 bootstraps"
PROCEDURE = "the expected performance of this modeling procedure at this sample size"
TOO_NARROW = "likely too narrow (Bates, Hastie & Tibshirani 2023)"
P_OVER_N = 1.0  # more candidate predictors than rows: a convention, stated where it is applied
NESTED_REPS = 50  # Bates et al.'s ``nested_cv`` default repetitions
NESTED_SOURCE = "Bates, Hastie & Tibshirani 2023"


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
                       reference: float | None, horizon: float | None = None
                       ) -> dict[str, float | None]:
    """Every metric of ``task`` and the calibration intercept and slope, as points; for a binary
    outcome also Somers' Dxy = 2·AUC − 1, the rank correlation rms ``validate`` reports."""
    y = np.asarray(y)
    out: dict[str, float | None] = dict(score(task, model, X, y, reference=reference,
                                              horizon=horizon))
    if task == "binary" and out.get("auc") is not None and math.isfinite(out["auc"]):
        out["dxy"] = 2.0 * float(out["auc"]) - 1.0
    if task in ("regression", "binary"):
        classes = None if task == "regression" else list(model.classes_)
        out.update(_calibration_points(task, y, predict(task, model, X), classes))
    return out


def resample_concern(n_boot: int) -> str | None:
    """The concern of fewer resamples than Collins et al. recommend (module docstring), or None."""
    if int(n_boot) >= RECOMMENDED_BOOT:
        return None
    return (f"{int(n_boot):,} bootstrap resamples: Collins et al. (2024) recommend at least "
            f"{RECOMMENDED_BOOT}; 200 is the fewest Steyerberg found estimates the optimism with "
            f"minor sampling variability.")


def optimism_bootstrap(task: str, make: Callable[[], Any],
                       fit: Callable[[Any, Any, Any, np.ndarray], Any], X: Any, y: Any, *,
                       n_boot: int, seed: int = 0, groups: Any = None, unit: str | None = None,
                       final: Any = None, horizon: float | None = None,
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Optimism:
    """Bootstrap optimism correction of the pipeline ``make()`` (module docstring).

    ``fit(model, X_b, y_b, units_b)`` fits a fresh pipeline on a resample; ``units_b`` labels each
    resampled row with its original unit, so every copy of a unit keeps to one side of any inner
    split. ``final`` is the pipeline already fit on every row (else one is fit). Each resample
    draws whole units (``folds.unit_rows``, ``folds.draw_units``).
    """
    from turbotab.core.models.folds import draw_units, unit_labels, unit_rows

    started = time.perf_counter()
    y = np.asarray(y)
    n = len(y)
    _, rows_of = unit_rows(groups, n)
    if groups is None:
        labels = np.arange(n).astype(str).astype(object)
        resampled = "rows"
    else:
        labels = unit_labels(groups, n)
        resampled = f"{unit or 'unit'} units"
    n_units = len(rows_of)
    reference = float(np.mean(y.astype(float))) if task == "regression" else None
    if final is None:
        final = fit(make(), X, y, labels)
    apparent = performance_points(task, final, X, y, reference, horizon)
    keys = list(apparent)
    need = len(np.unique(y)) if task not in ("regression", "time_to_event") else 1
    rng = np.random.default_rng(seed)
    optimism: dict[str, list[float]] = {k: [] for k in keys}
    failed = 0
    for b in range(int(n_boot)):
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        draw = draw_units(rng, n_units)
        rows = np.concatenate([rows_of[u] for u in draw])
        yb = y[rows]
        if task not in ("regression", "time_to_event") and len(np.unique(yb)) < need:
            failed += 1
            continue
        Xb = X.iloc[rows] if hasattr(X, "iloc") else np.asarray(X)[rows]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fit(make(), Xb, yb, labels[rows])
                ref_b = float(np.mean(yb.astype(float))) if task == "regression" else None
                own = performance_points(task, model, Xb, yb, ref_b, horizon)
                test = performance_points(task, model, X, y, ref_b, horizon)
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
    caution: str | None = None  # where it is not sound, in one line (north star 5)
    cluster: str | None = None  # WP17: the grouping the cluster question named, folded by here


BOOTSTRAP_CAUTION = ("Sound for regression-type families; a near-interpolating one (boosted trees) "
                     "keeps its cross-validated score, since the bootstrap overstates it (Coley et "
                     "al. 2023).")


def not_applied(family_label: str, n_boot: int, unit: str | None = None) -> Optimism:
    """The optimism record of a family that declares Harrell's bootstrap unsound for it: nothing
    estimated, and the reason, which the fit states as a concern (module docstring)."""
    return Optimism(n_boot=int(n_boot), n_ok=0, failed=0, seconds=0.0,
                    resampled=f"{unit} units" if unit else "rows", estimates={},
                    refused=(f"Harrell's bootstrap optimism correction is not applied to "
                             f"{family_label}: a learner that nearly memorizes its rows scores the "
                             f"original rows inside each resample (about 63% of them) almost "
                             f"perfectly, so the bootstrap understates its optimism and overstates "
                             f"its performance (Coley et al. 2023). Its cross-validated score is its "
                             f"internal validation."))


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
    from turbotab.core.models.folds import COMPARISON_REPEATS

    r = int(repeats or REPEATS)
    b = int(n_boot or OPTIMISM_BOOT)
    n = f"{int(n_units):,} {unit}"
    # Under prediction the comparisons run on repeated k-fold whatever leads (MS6): its refits are
    # counted in every option's cost; the first run of folds is the headline's own.
    compare = purpose != "inference" and not time_ordered
    sub = max(r, COMPARISON_REPEATS)
    by: dict[str, ValidationOption] = {
        "bootstrap": ValidationOption(
            validation="bootstrap", label="Bootstrap optimism correction",
            measures=f"Every row trains and scores; {b} whole-pipeline refits on resamples "
                     f"estimate the optimism.",
            cost=(f"about {b} refits of each family, besides {sub * folds} for the comparisons' "
                  f"{sub} × {folds}-fold repeats" if compare else
                  f"about {b} refits of each family, besides one {folds}-fold run"),
            caution=BOOTSTRAP_CAUTION),
        "repeated_kfold": ValidationOption(
            validation="repeated_kfold", label=f"Repeated cross-validation ({r} × {folds})",
            measures=f"The folds are drawn {r} times, so the score no longer rests on one "
                     f"partition.",
            cost=f"{(sub if compare else r) * folds} refits of each family"),
        "kfold": ValidationOption(
            validation="kfold",
            label="Time-ordered cross-validation" if time_ordered else "Cross-validation, once",
            measures=("Each fold scored by models fit on earlier units." if time_ordered else
                      "One partition into folds: cheap, and noisier at small sizes."),
            cost=(f"{sub * folds} refits of each family: the score's {folds}-fold run is the "
                  f"first of the comparisons' {sub} × {folds}-fold repeats"
                  if compare else f"{folds} refits of each family")),
        "internal_external": ValidationOption(
            validation="internal_external", label="Internal–external, by cluster",
            measures="Each site, study or period scored by models fit on the others, with the "
                     "spread.",
            cost=("one refit of each family per cluster" +
                  (f", besides {sub * folds} for the comparisons' {sub} × {folds}-fold repeats"
                   if compare else "")),
            needs_cluster=True),
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
                      groups: Any = None, unit: str | None = None,
                      metric: str | None = None) -> InternalExternal:
    """Each cluster's performance (``cv``'s folds are the clusters, in ``labels``' order) and a
    random-effects summary of the primary metric (``metric``, default the task's)."""
    primary = metric or PRIMARY[task]
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
                                     groups=None if units is None else units[pred.rows], unit=unit,
                                     horizon=cv.horizon)
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


class PairedTest(_Model):
    """Bouckaert & Frank's corrected repeated k-fold t over paired fold scores (module docstring)."""

    mean: float | None  # mean of the paired differences, signed so that positive favors the first
    variance: float | None  # their sample variance
    se: float | None  # √((1/(rK) + n₂/n₁)·variance)
    t: float | None
    df: int | None  # rK − 1
    p: float | None  # two-sided
    ci_low: float | None
    ci_high: float | None
    folds: int  # rK paired folds
    repeats: int
    test_share: float | None  # n₂/n₁: rows scored over rows fit


def corrected_t(a: Sequence[float | None], b: Sequence[float | None], *,
                test_share: float | None, repeats: int, higher_is_better: bool = True,
                level: float = 0.95) -> PairedTest:
    """The corrected repeated k-fold t of ``a`` against ``b`` (fold scores over the same folds of
    every repeat, in order). ``d = a − b`` (``b − a`` for a score that is better when lower), so a
    positive difference favors ``a``; folds where either is missing are left out."""
    from scipy import stats

    sign = 1.0 if higher_is_better else -1.0
    d = np.asarray([sign * (float(x) - float(y)) for x, y in zip(a, b)
                    if x is not None and y is not None and math.isfinite(float(x))
                    and math.isfinite(float(y))], dtype=float)
    k = int(len(d))
    share = test_share if test_share is not None and test_share > 0 else None
    out = PairedTest(mean=float(d.mean()) if k else None, variance=None, se=None, t=None, df=None,
                     p=None, ci_low=None, ci_high=None, folds=k, repeats=int(repeats),
                     test_share=share)
    if k < 2:
        return out
    if share is None:
        share = 1.0 / max(k / max(int(repeats), 1) - 1.0, 1.0)  # K-fold: 1/(K − 1)
        out.test_share = share
    var = float(d.var(ddof=1))
    se = math.sqrt((1.0 / k + share) * var)
    out.variance, out.se, out.df = var, se, k - 1
    q = float(stats.t.ppf(0.5 + level / 2, k - 1))
    out.ci_low, out.ci_high = out.mean - q * se, out.mean + q * se
    if se > 0:
        out.t = out.mean / se
        out.p = float(2.0 * stats.t.sf(abs(out.t), k - 1))
    return out


class FamilyDifference(_Model):
    """Two families' primary metric, paired over the same folds of the comparison substrate, with
    the corrected repeated k-fold t (module docstring)."""

    a: str
    b: str
    metric: str
    difference: float | None  # mean paired difference, signed so that positive favors a
    ci_low: float | None
    ci_high: float | None
    df: int | None
    sentence: str
    t: float | None = None
    p: float | None = None
    se: float | None = None
    folds: int | None = None  # rK paired folds
    repeats: int | None = None


COMPARISONS_NOTE = ("Each pairwise interval is descriptive: none is corrected for how many pairs "
                    "there are, and the choice among the families is corrected by bootstrap "
                    "bias-corrected cross-validation instead.")


def family_differences(task: str, results: dict[str, CrossValidated],
                       summaries: dict[str, dict[str, dict[str, Any]]] | None,
                       labels: dict[str, str], metric: str | None = None
                       ) -> list[FamilyDifference]:
    """Every pair of families on the primary (``metric``) by :func:`corrected_t` over the folds of
    ``results`` (the comparison substrate: every repeat's folds), so whether two scores differ is
    read off the user's own folds rather than a rule of thumb about sample size (audit ME-11)."""
    primary = metric or PRIMARY[task]
    higher = higher_is_better(primary)
    keys = list(results)
    label = score_words(primary)
    out: list[FamilyDifference] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = results[a], results[b]
            test = corrected_t(ra.folds_of(primary), rb.folds_of(primary),
                               test_share=ra.test_share, repeats=len(ra.repeats()),
                               higher_is_better=higher)
            if test.mean is None or test.ci_low is None or test.ci_high is None:
                sentence = (f"{labels[a]} and {labels[b]} cannot be compared on {label} with "
                            f"these folds.")
            else:
                inside = test.ci_low <= 0 <= test.ci_high
                ahead = labels[a] if test.mean >= 0 else labels[b]
                verdict = ("not distinguishable on these rows" if inside
                           else f"{ahead} is ahead on these rows")
                runs = (f"{test.repeats} × {test.folds // max(test.repeats, 1)} folds"
                        if test.repeats > 1 else f"{test.folds} folds")
                t_said = f"{test.t:.2f}".replace("-", "−") if test.t is not None else None
                stat = (f"; corrected repeated k-fold t {t_said} on {test.df} df, {runs}"
                        if t_said is not None else f", {runs}")
                sentence = (f"{labels[a]} against {labels[b]}: {label} differs by "
                            f"{_fmt(abs(test.mean))} (95% interval of the difference "
                            f"{_fmt(test.ci_low)} to {_fmt(test.ci_high)}{stat}): {verdict}.")
            out.append(FamilyDifference(a=a, b=b, metric=primary, difference=test.mean,
                                        ci_low=test.ci_low, ci_high=test.ci_high, df=test.df,
                                        sentence=sentence, t=test.t, p=test.p, se=test.se,
                                        folds=test.folds, repeats=test.repeats))
    return out


# ── what a performance interval describes (module docstring) ─────────────────


def wide(n_predictors: int, n_rows: int) -> bool:
    """More candidate predictors than rows (p/n above :data:`P_OVER_N`)."""
    return n_rows > 0 and n_predictors / n_rows > P_OVER_N


def wide_clause(n_predictors: int, n_rows: int) -> str:
    """Why an interval is labeled too narrow, in a clause: the ratio and the convention."""
    return (f"{n_predictors:,} candidate predictors for {n_rows:,} rows (p/n "
            f"{n_predictors / n_rows:.1f}, above the {P_OVER_N:g} this app takes as p ≫ n)")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".replace("-", "−")


def performance_sentence(label: str, estimate: float | None, low: float | None,
                         high: float | None, *, how: str, narrow: str | None = None) -> str:
    """One performance sentence: the score, its interval and how it was made, said as the expected
    performance of this modeling procedure at this sample size (Bates et al. 2023); ``narrow``: why
    the interval is labeled :data:`TOO_NARROW`, when it is."""
    interval = (f" (95% interval {_fmt(low)} to {_fmt(high)}"
                + (f", {TOO_NARROW}: {narrow}" if narrow else "") + ")"
                if low is not None and high is not None else "")
    return f"{label} {_fmt(estimate)}{interval} by {how}: {PROCEDURE}."


# ── the nested cross-validation interval (module docstring) ──────────────────


class NestedCV(_Model):
    """Bates, Hastie & Tibshirani's nested cross-validation interval for one family's primary."""

    metric: str
    estimate: float | None  # the debiased point estimate
    ci_low: float | None
    ci_high: float | None
    raw_mean: float | None  # mean of every inner error, before the bias correction
    bias: float | None
    inflation: float | None  # how much wider than the naive interval, between 1 and √K
    sd: float | None  # SD of the inner errors
    reps: int
    bias_reps: int
    folds: int
    fits: int
    seconds: float
    method: str = f"nested cross-validation ({NESTED_SOURCE})"
    refused: str | None = None


PER_ROW_LOSSES = ("mse", "log_loss", "brier", "rps", "brier_t")


def nested_cv_fits(folds: int, reps: int = NESTED_REPS, bias_reps: int | None = None) -> int:
    """The refits nested cross-validation makes: ``K(K − 1)/2 + K`` per repetition, and ``K`` per
    plain cross-validation run of the bias estimate."""
    k = int(folds)
    b = int(math.ceil(reps / 5)) if bias_reps is None else int(bias_reps)
    return int(reps) * (k * (k - 1) // 2 + k) + b * k


def loss_rows(task: str, metric: str, y: Any, prediction: Any, *, classes: Sequence[Any] | None,
              horizon: float | None = None) -> np.ndarray | None:
    """The per-row losses whose mean is ``metric`` (the strictly proper primaries); None for a
    score that is not a mean of per-row losses (R², AUC, C)."""
    y = np.asarray(y)
    P = np.asarray(prediction, dtype=float)
    if metric == "mse":
        return (y.astype(float) - P) ** 2
    if metric == "log_loss":
        return perf.log_loss_rows(y, P, list(classes or []))
    if metric == "brier" and task == "binary":
        return (P[:, 1] - (y == list(classes or [])[1])) ** 2
    if metric == "rps":
        return perf.rps_rows(y, P, list(classes or []))
    if metric == "brier_t":
        return perf.brier_rows(y, P[:, 1], horizon)
    return None


def _plain_folds(n: int, groups: Any, folds: int, rng: np.random.Generator) -> np.ndarray:
    """One plain cross-validation run's folds ``1 … K`` over whole units (every unit used), as
    Bates et al.'s ``naive_cv`` draws them over rows."""
    from turbotab.core.models.folds import unit_rows

    _, rows_of = unit_rows(groups, n)
    unit_fold = (np.arange(len(rows_of)) % int(folds) + 1)[rng.permutation(len(rows_of))]
    ids = np.zeros(n, dtype=np.int64)
    for u, rows in enumerate(rows_of):
        ids[rows] = unit_fold[u]
    return ids


def nested_cv_interval(task: str, metric: str, make: Callable[[], Any],
                       fit: Callable[[Any, Any, Any], Any], X: Any, y: Any, *, folds: int = 5,
                       reps: int = NESTED_REPS, bias_reps: int | None = None, seed: int = 0,
                       groups: Any = None, horizon: float | None = None, level: float = 0.95,
                       draws: dict[str, list[np.ndarray]] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> NestedCV:
    """The nested cross-validation interval for ``metric`` (module docstring).

    ``fit(model, X_fit, y_fit)`` fits a fresh pipeline (``make()``). Folds are whole units
    (``folds.nested_folds``), drawn from ``numpy.random.default_rng(seed)``: every repetition's,
    then every plain run's. ``draws`` (``{"nested": [...], "plain": [...]}``, fold ids per row)
    replaces the draws, so an independent implementation can be given the same folds.
    """
    from statistics import NormalDist

    from turbotab.core.models.folds import nested_folds

    started = time.perf_counter()
    y = np.asarray(y)
    n = len(y)
    K = int(folds)
    B = int(math.ceil(reps / 5)) if bias_reps is None else int(bias_reps)
    fits = 0

    def refused(reason: str, nested_n: int = 0, plain_n: int = 0) -> NestedCV:
        return NestedCV(metric=metric, estimate=None, ci_low=None, ci_high=None, raw_mean=None,
                        bias=None, inflation=None, sd=None, reps=nested_n, bias_reps=plain_n,
                        folds=K, fits=fits, seconds=round(time.perf_counter() - started, 3),
                        refused=reason)

    if metric not in PER_ROW_LOSSES:
        return refused(f"Nested cross-validation needs a mean of per-row losses; {metric} is not "
                       f"one.")
    rng = np.random.default_rng(seed)
    nested = list((draws or {}).get("nested") or [])
    plain = list((draws or {}).get("plain") or [])
    if not nested:
        nested = [nested_folds(n, groups=groups, folds=K, rng=rng) for _ in range(int(reps))]
    if not plain:
        plain = [_plain_folds(n, groups, K, rng) for _ in range(B)]

    def take(data: Any, mask: np.ndarray) -> Any:
        return data.iloc[np.flatnonzero(mask)] if hasattr(data, "iloc") else np.asarray(data)[mask]

    def fitted(train: np.ndarray) -> Any:
        nonlocal fits
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit(make(), take(X, train), y[train])
        fits += 1
        return model

    def scored_by(model: Any, scored: np.ndarray) -> np.ndarray:
        classes = list(getattr(model, "classes_", None) if hasattr(model, "classes_") else [])
        found = loss_rows(task, metric, y[scored],
                          predict(task, model, take(X, scored), horizon=horizon),
                          classes=classes, horizon=horizon)
        if found is None or not np.all(np.isfinite(found)):
            raise ValueError(f"{metric} could not be computed on a fold")
        return found

    def losses(train: np.ndarray, scored: np.ndarray) -> np.ndarray:
        return scored_by(fitted(train), scored)

    pivots: list[tuple[float, float]] = []
    every: list[np.ndarray] = []
    cv_means: list[float] = []
    try:
        for fold_id in nested:
            fold_id = np.asarray(fold_id, dtype=np.int64)
            held: dict[tuple[int, int], np.ndarray] = {}
            for f1 in range(1, K):
                for f2 in range(f1 + 1, K + 1):
                    model = fitted((fold_id != f1) & (fold_id != f2))  # one fit without both
                    held[(f1, f2)] = scored_by(model, fold_id == f1)  # fold f1, fit without f1, f2
                    held[(f2, f1)] = scored_by(model, fold_id == f2)
            for f1 in range(1, K + 1):
                e_out = losses(fold_id != f1, fold_id == f1)
                e_in = np.concatenate([held[(f2, f1)] for f2 in range(1, K + 1) if f2 != f1])
                pivots.append((float(e_in.mean() - e_out.mean()),
                               float(e_out.var(ddof=1) / len(e_out)) if len(e_out) > 1 else 0.0))
            for f1 in range(1, K):
                for f2 in range(f1 + 1, K + 1):
                    every.extend([held[(f1, f2)], held[(f2, f1)]])
        for fold_id in plain:
            fold_id = np.asarray(fold_id, dtype=np.int64)
            errs = [losses(fold_id != f, fold_id == f) for f in range(1, K + 1)]
            cv_means.append(float(np.concatenate(errs).mean()))
    except ValueError as exc:
        return refused(f"Nested cross-validation could not run: {exc}.", len(nested), len(plain))
    e = np.concatenate(every)
    a = np.asarray([p[0] for p in pivots])
    b = np.asarray([p[1] for p in pivots])
    sd = float(e.std(ddof=1))
    n_sub = math.floor(n * (K - 1) / K)
    inflation = (math.sqrt(max(0.0, float(np.mean(a ** 2 - b)))) / (sd / math.sqrt(n_sub))
                 if sd > 0 else 1.0)
    inflation = max(1.0, min(inflation, math.sqrt(K)))
    raw = float(e.mean())
    bias = (raw - float(np.mean(cv_means))) * (1 + ((K - 2) / K) ** 1.5) if cv_means else 0.0
    estimate = raw - bias
    z = NormalDist().inv_cdf(0.5 + level / 2)
    half = z * sd / math.sqrt(n) * inflation
    return NestedCV(metric=metric, estimate=estimate, ci_low=estimate - half,
                    ci_high=estimate + half, raw_mean=raw, bias=bias, inflation=inflation, sd=sd,
                    reps=len(nested), bias_reps=len(plain), folds=K, fits=fits,
                    seconds=round(time.perf_counter() - started, 3))


# ── the method contracts (BLUEPRINT §13) ─────────────────────────────────────
#
# Each method this module brings to the comparison enters through the one registry
# (``turbotab.core.contracts``): slot, data scope, needs, routing (its option labeled customary and
# sound per purpose, with the leash's rung), storyboard, sentence (its numbers as placeholders in
# braces) and relations, each with the stable id the fit's chain (:func:`fired_chain`) records.

VALIDATION_CONTRACTS: tuple[str, ...] = ("proper_primary", "comparison_substrate", "bbc_cv",
                              "bootstrap_optimism", "horizon_calibration", "nested_cv_interval")
_SCOPE = ("Each score reads the out-of-fold or out-of-resample predictions of models fit on the "
          "training rows only; nothing here is learned from a row it scores.")


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS as REGISTRY, ContractOption, MethodContract,
                                        Relation, register_contract)

    if VALIDATION_CONTRACTS[0] in REGISTRY:
        return
    prediction = ("prediction",)
    here = "turbotab.core.models.validation"

    def stated(key: str, label: str, customary: str, prediction_says: str, inference_says: str,
               rungs: tuple[str, str]) -> tuple[ContractOption, ...]:
        return (ContractOption(key, label, customary,
                               {"prediction": prediction_says, "inference": inference_says},
                               {"prediction": rungs[0], "inference": rungs[1]}),)

    def contract(key: str, label: str, **fields: Any) -> None:
        register_contract(MethodContract(key=key, label=label, slot="evaluation",
                                         scope="training_fold", scope_note=_SCOPE, decision="set_split",
                                         stage="fit", place="11 · Tuning and comparison", **fields))

    contract(
        "proper_primary", "A strictly proper primary score", run_order=1.0,
        needs=("the task", "out-of-fold predictions"),
        question="(stated, not asked: the primary score is strictly proper)",
        options=stated(
            "proper", "A strictly proper per-row loss",
            "AUC and the C-index are the field's customary headline (semi-proper)",
            "Stated, not asked: compare, choose and declare on the primary; the customary headline "
            "reported beside it with the tension",
            "Stated: the scores describe fit, not the estimate", ("recommended", "available")),
        storyboard=("score each row's prediction by a per-row proper loss", "average the losses",
                    "report AUC or C beside it as the customary headline"),
        sentence="{headline} is the customary headline: it ranks risks without asking whether they "
                 "are right (semi-proper), so the models were compared, chosen and declared on "
                 "{primary}, a strictly proper score (Van Calster et al., STRATOS TG6).",
        relations=(Relation(
            "implies", "customary_headline",
            "the comparisons, the choice and the declaration read the strictly proper primary; "
            "AUC or C is reported as the customary headline",
            condition="a binary, ordinal or time-to-event task",
            enforced_by="turbotab.core.models.metrics:tension", id="proper_score_primary"),),
        sources=("Van Calster et al., STRATOS TG6, arXiv:2412.10288",
                 "Harrell, Statistically efficient ways to quantify added predictive value"))
    contract(
        "comparison_substrate", "Repeated k-fold comparison substrate", run_order=2.0,
        needs=("training rows", "the unit when rows repeat"),
        question="(stated: whatever validation leads, at least 10 × K folds)",
        options=stated(
            "repeated_kfold", "At least 10 × K folds, whole units together",
            "A single k-fold run or one split is the field's habit",
            "Stated: whatever validation leads, at least 10 × K folds",
            "Not a comparison", ("recommended", "not_offered")),
        storyboard=("draw the folds 10 times, whole units together",
                    "fit every family and the baseline on each fold",
                    "pair their fold scores", "the corrected repeated k-fold t"),
        sentence="models were compared with each other and with the no-predictor baseline on "
                 "{folds}-fold cross-validation repeated {repeats} times, by the corrected repeated "
                 "k-fold t (Nadeau & Bengio 2003; Bouckaert & Frank 2004)",
        relations=(
            Relation("implies", "corrected_repeated_t",
                     "paired comparisons and the baseline verdict on at least 10 × K folds, with "
                     "rK − 1 degrees of freedom", purposes=prediction,
                     condition="any family under prediction",
                     enforced_by=f"{here}:corrected_t", id="families_compared_on_substrate"),
            Relation("implies", "grouped_resampling",
                     "grouped folds in every repeat, a grouped holdout, a bootstrap by unit, BBC-CV "
                     "by unit and nested folds by unit",
                     condition="rows repeat within a unit",
                     enforced_by="turbotab.core.models.folds:kfold_assignment",
                     id="repeated_units_group_resampling")),
        sources=("Nadeau & Bengio, Mach Learn 2003;52:239", "Bouckaert & Frank, PAKDD 2004"))
    contract(
        "bbc_cv", "Bootstrap bias-corrected cross-validation", run_order=3.0,
        needs=("two or more families", "their out-of-fold predictions"),
        question="(stated: the choice among families is corrected)",
        options=stated(
            "bbc_cv", "The choice corrected by BBC-CV",
            "Reporting the best family's own cross-validated score is the field's habit",
            "Stated: the choice among families is corrected; with no holdout the "
            "selection-corrected estimate is the result (refuse the winner's own score as the "
            "result)", "Not applicable", ("recommended", "not_offered")),
        storyboard=("resample the out-of-fold predictions by unit",
                    "choose the best family on the resample", "score it on the units left out",
                    "average"),
        sentence="the choice among {k} families was corrected by bootstrap bias-corrected "
                 "cross-validation (Tsamardinos et al. 2018, {b} resamples)",
        relations=(
            Relation("implies", "selection_corrected_score",
                     "the best family's score is corrected for the choice", purposes=prediction,
                     condition="two or more families under prediction",
                     enforced_by="turbotab.core.models.selection:selection_optimism",
                     id="choice_among_families_bbc"),
            Relation("conflicts", "winner_own_score",
                     "the winner's own score as the result is refused; the selection-corrected "
                     "estimate is the result", purposes=prediction, rung="refused",
                     condition="no rows held out and the families compared",
                     exits=("Keep the compared families",),
                     enforced_by="turbotab.core.models.selection:_compared_families_stay",
                     id="no_holdout_declares_corrected")),
        sources=("Tsamardinos, Greasidou & Borboudakis, Mach Learn 2018;107:1895",))
    contract(
        "bootstrap_optimism", "Bootstrap optimism correction", run_order=4.0,
        needs=("a regression-type family", "the whole pipeline"),
        question="How is performance estimated on these rows?",
        options=stated(
            "bootstrap", "Harrell's bootstrap, the whole pipeline refit on each resample",
            "Steyerberg's floor of 200 resamples is customary",
            "Asked (the split question), ranked first below 20,000 units; at least 500 resamples "
            "(Collins et al. 2024)", "Offered after one cross-validation run",
            ("recommended", "available")),
        storyboard=("fit on every row: the apparent score",
                    "refit on a resample of whole units", "score it on the resample and on every row",
                    "average the gap: the optimism", "subtract it"),
        sentence="Harrell's bootstrap ({b} resamples, the whole pipeline refit on each)",
        relations=(Relation("implies", "resample_concern",
                            "the concern is stated (Collins et al. 2024)",
                            condition="fewer than 500 resamples",
                            enforced_by=f"{here}:resample_concern", id="bootstrap_resamples"),),
        sources=("Harrell, Lee & Mark, Stat Med 1996;15:361", "Collins et al., BMJ 2024;384:e074819"))
    contract(
        "horizon_calibration", "Calibration by a horizon, and by level", run_order=5.0,
        needs=("a time to event with its follow-up and a horizon",
               "or an ordinal or multiclass outcome"),
        question="(stated: the prediction horizon is declared with the follow-up, else the median "
                 "follow-up time)",
        options=stated(
            "by_horizon_and_level", "At the horizon; by level for ordinal and multiclass outcomes",
            "Calibration is often left unreported for these tasks",
            "Stated: the prediction horizon is declared with the follow-up, else the median "
            "follow-up time",
            "Stated beside the fit scores", ("recommended", "available")),
        storyboard=("each row's risk by the horizon", "the Kaplan–Meier observed risk",
                    "observed against expected, by risk group", "the calibration slope"),
        sentence="calibration was assessed at {horizon} by observed (Kaplan–Meier) against "
                 "predicted risk and the calibration slope",
        relations=(
            Relation("implies", "horizon",
                     "the Brier score and calibration are read at the declared prediction horizon, "
                     "or at the median follow-up time, stated", condition="a time-to-event outcome",
                     enforced_by="turbotab.core.models.performance:horizon_calibration",
                     id="time_to_event_horizon"),
            Relation("implies", "not_assessed_note",
                     "the record says calibration not assessed, and why",
                     condition="calibration cannot be computed",
                     enforced_by=f"{here}:calibration_by", id="calibration_not_assessed")),
        sources=("Graf et al., Stat Med 1999;18:2529", "McLernon et al., Ann Intern Med 2023;176:105",
                 "Van Calster et al., BMC Med 2019;17:230"))
    contract(
        "nested_cv_interval", "The nested cross-validation interval", run_order=6.0,
        needs=("a per-row proper loss", "p ≫ n"),
        question="Predictors outnumber the rows: run the nested cross-validation interval?",
        options=stated(
            "nested_cv", "Bates, Hastie & Tibshirani's nested cross-validation interval",
            "The naive cross-validation interval is customary",
            "Offered with its compute estimate where p/n > 1; else the interval is labeled likely "
            "too narrow", "Not applicable", ("available", "not_offered")),
        storyboard=("for each repetition, each pair of folds fit without both",
                    "each fold fit without it", "the spread of the inner against the outer errors",
                    "widen the interval by it"),
        sentence="its interval is the nested cross-validation interval ({reps} repetitions of "
                 "{folds} folds; Bates, Hastie & Tibshirani 2023)",
        relations=(Relation("enables", "nested_cv_interval",
                            "the nested cross-validation interval is offered, and the naive "
                            "interval is labeled likely too narrow until it runs",
                            purposes=prediction, condition="more candidate predictors than rows",
                            enforced_by=f"{here}:nested_cv_interval", id="p_much_greater_n"),),
        sources=("Bates, Hastie & Tibshirani, JASA 2023 (arXiv:2104.00673)",))


_register_contracts()


def contract(key: str) -> Any:
    """One of this module's contracts, as the one registry holds it."""
    from turbotab.core.contracts import contract as registered

    if key not in VALIDATION_CONTRACTS:
        raise KeyError(f"{key} is not a validation contract")
    return registered(key)


def relation_ids() -> list[str]:
    return [r.id for key in VALIDATION_CONTRACTS for r in contract(key).relations]


class ChainLink(_Model):
    """A relation that fired on this fit: because ``because``, ``then`` (BLUEPRINT §13)."""

    relation: str
    because: str
    then: str


# ── what the fit stage reports with these (MS6) ──────────────────────────────


def score_words(metric: str) -> str:
    """A metric's label mid-sentence: an acronym as it is, a word in lower case."""
    from turbotab.core.models.metrics import LABELS

    label = LABELS.get(metric, metric)
    keep = label[:2].isupper() or label.startswith(("R²", "C-", "Brier"))
    return label if keep else label[0].lower() + label[1:]


def headline_how(validation: str, scheme: str, folds: int, repeats: int,
                 cluster: str | None = None) -> str:
    """How the headline cross-validated score was made, in words."""
    if scheme == "time_ordered":
        return f"cross-validation over {folds} time-ordered folds"
    if validation == "internal_external" and cluster:
        return f"internal–external validation by {cluster}"
    if repeats > 1:
        return f"{folds}-fold cross-validation repeated {repeats} times"
    return f"{folds}-fold cross-validation"


def level_words(task: str, level: str) -> str:
    """How a level's calibration concern is introduced."""
    return f"At or above `{level}`" if task == "ordinal" else f"For `{level}`"


def calibration_by(task: str, y: Any, prediction: Any, classes: Sequence[Any] | None,
                   levels: Sequence[str] | None, groups: Any, horizon: float | None,
                   where: str) -> tuple[list[Any] | None, Any, str | None]:
    """(calibration by level, calibration at the horizon, the "not assessed" note): what MS6 adds
    to the binary and numeric calibration (``performance.calibration``) for the other tasks."""
    if task in ("regression", "binary"):
        return None, None, None
    if task in ("ordinal", "multiclass"):
        if len(np.asarray(y)) < 10:
            return None, None, f"{perf.NOT_ASSESSED}: fewer than 10 scored rows."
        found = perf.level_calibration(task, y, prediction, list(classes or []),
                                       names=list(levels) if levels is not None else None,
                                       groups=groups, where=where)
        if not any(item.calibration is not None for item in found):
            return found, None, f"{perf.NOT_ASSESSED}: no level had both outcomes among the scored rows."
        return found, None, None
    if task == "time_to_event":
        P = np.asarray(prediction, dtype=float)
        if horizon is None or P.ndim != 2 or not np.all(np.isfinite(P[:, 1])):
            return None, None, f"{perf.NOT_ASSESSED}: no risk by a horizon could be predicted."
        cal = perf.horizon_calibration(y, P[:, 1], horizon, where=where)
        if cal is None:
            return None, None, (f"{perf.NOT_ASSESSED}: fewer than 10 scored rows, or no event by "
                                f"the horizon.")
        return None, cal, None
    return None, None, f"{perf.NOT_ASSESSED} for this task."


def fired_chain(task: str, primary: str, *, inference: bool, grouped_by: str | None,
                families: Sequence[str], n_holdout: int, comparison: Any, selection: Any,
                result: Any, n_boot: int, horizon_note: str | None, notes: Sequence[str | None],
                narrow: str | None, nested: bool) -> list[ChainLink]:
    """The relations of :data:`CONTRACTS` that fired on this fit, each said as "because …, …"."""
    from turbotab.core.models.metrics import HEADLINE, LABELS, SPOKEN

    links: list[ChainLink] = []
    headline = HEADLINE.get(task)
    if headline is not None and headline != primary:
        links.append(ChainLink(
            relation="proper_score_primary",
            because=(f"the outcome is {'an' if task[0] in 'aeiou' else 'a'} "
                     f"{task.replace('_', '-')} outcome"),
            then=(f"the models were compared, chosen and declared on {SPOKEN.get(primary, primary)}, "
                  f"and {LABELS[headline]} is reported as the customary headline")))
    if not inference and comparison is not None and families:
        rk = comparison.repeats * comparison.folds
        on = (f"{comparison.folds}-fold cross-validation repeated {comparison.repeats} times"
              if comparison.repeats > 1 else
              f"the {comparison.folds} time-ordered folds, run once as time order allows")
        links.append(ChainLink(
            relation="families_compared_on_substrate", because="the purpose is prediction",
            then=(f"every family and the no-predictor baseline were compared on {on} ({rk} paired "
                  f"folds, {rk - 1} degrees of freedom)")))
    if grouped_by:
        links.append(ChainLink(
            relation="repeated_units_group_resampling",
            because=f"rows repeat within `{grouped_by}`",
            then=("every fold of every repeat, the held-out rows, every bootstrap resample, the "
                  "BBC-CV resamples and the nested folds keep each unit's rows together")))
    if not inference and selection:
        links.append(ChainLink(
            relation="choice_among_families_bbc", because=f"{len(families)} families were compared",
            then="the best family's score is corrected for the choice by BBC-CV"))
    if not inference and result is not None and getattr(result, "basis", None) == "selection_corrected":
        links.append(ChainLink(
            relation="no_holdout_declares_corrected",
            because="no rows were held out and the families were compared on these rows",
            then="the result is the selection-corrected estimate, never the winner's own score"))
    if n_boot and n_boot < RECOMMENDED_BOOT:
        links.append(ChainLink(relation="bootstrap_resamples",
                               because=f"{n_boot:,} bootstrap resamples were asked for",
                               then=f"the concern is stated: Collins et al. (2024) recommend at "
                                    f"least {RECOMMENDED_BOOT}"))
    if task == "time_to_event" and horizon_note:
        links.append(ChainLink(relation="time_to_event_horizon",
                               because="the outcome is a time to event",
                               then=horizon_note[0].lower() + horizon_note[1:].rstrip(".")))
    if any(n for n in notes):
        links.append(ChainLink(relation="calibration_not_assessed",
                               because="calibration could not be computed for a family",
                               then="its record says calibration not assessed, and why"))
    if narrow:
        links.append(ChainLink(
            relation="p_much_greater_n", because=narrow,
            then=("the nested cross-validation interval ran" if nested else
                  f"the intervals are labeled {TOO_NARROW}, and the nested cross-validation "
                  f"interval is offered")))
    return links


__all__ = ["BOOTSTRAP_CAUTION", "CALIBRATION_KEYS", "COMPARISONS_NOTE", "ChainLink",
           "ClusterScore", "FamilyDifference", "InternalExternal", "METRICS", "MIN_OK_SHARE",
           "NESTED_REPS", "NestedCV", "Optimism", "OptimismEstimate", "P_OVER_N",
           "PER_ROW_LOSSES", "PROCEDURE", "PairedTest", "RECOMMENDED_BOOT", "RESAMPLE_BELOW",
           "RESAMPLE_SOURCE", "TOO_NARROW", "ValidationOption", "ValidationPlan",
           "contract", "corrected_t", "family_differences", "internal_external", "loss_rows",
           "nested_cv_fits", "nested_cv_interval", "not_applied", "optimism_bootstrap",
           "performance_points", "performance_sentence", "relation_ids", "resample_concern",
           "validation_plan", "wide", "wide_clause", "calibration_by", "fired_chain",
           "headline_how", "level_words", "score_words", "VALIDATION_CONTRACTS"]
