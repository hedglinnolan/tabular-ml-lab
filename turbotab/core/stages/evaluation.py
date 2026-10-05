"""The ``evaluation`` stage: what the fit's comparison does not say by itself (MODELING_SEQUENCE §1
rows 9 and 11, prediction; §0 ruling 13; §5 "New from this spec").

Under prediction, on the fit's own comparison substrate (the same folds, ``fit.objects["comparison"]``):

* **the regression-with-splines benchmark, always fitted** (the review: "Always fit the no-predictor
  baseline and a properly specified regression benchmark with splines"): least squares or logistic
  regression, every continuous predictor a restricted cubic spline with knots by Harrell's rule, on
  the same in-fold preprocessing as every family; its score, its verdict against the baseline and
  its paired difference from each family;
* **what the interpretable model costs or gains** against the best flexible family, BBC-corrected
  over the flexible set (``models.selection.interpretable_cost``);
* **intended use → the decision curve**, the threshold chosen in-fold, **subgroup performance with
  intervals**, **shrinkage offered as model updating**, and **internal–external cross-validation by
  cluster** when a grouping is named (``models.decision_curve``, ``models.validation``);
* **inclusion frequencies** of the in-fold selection across folds (``models.variable_selection``);
* **design-based cross-validation** under the surveyed-population answer (``models.design_cv``);
* the levers set by hand after an outcome view, which the corrected score does not cover
  (``stages.explore``).

Under inference no cross-validated score is shown (ruling 13; §1 row 11): the stage says so, and
runs the selection sensitivity analysis when one is declared (backward elimination by Wald tests
pooled across the imputed copies by Rubin's rules).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.estimand import FIT_SCORES, MODEL_SCORES
from turbotab.core.graph import Bundle, StageContext
from turbotab.core.jobs import Cancelled
from turbotab.core.stages.data import open_store

EVALUATION_READS: tuple[str, ...] = ("target", "purpose", "task", "event", "outcome_order",
                                     "intended_use",
                                     "updating", "selection", "levers", "survey",
                                     "models", "split", "follow_up", "outcome_views",
                                     "exposure_forms", "outcome_scale")
BENCHMARK = "spline_benchmark"
BENCHMARK_LABEL = "Regression with splines (benchmark)"
SUPPORTED = ("regression", "binary")
NO_SCORE_UNDER_INFERENCE = (
    "Under inference no cross-validated score is shown: the declared model is reported by its "
    "estimates, not compared or chosen by how it predicts (MODELING_SEQUENCE §1 row 11).")
INCLUSION_FOLDS = 20  # the training folds the inclusion frequencies refit the selection on
MAX_IECV_CLUSTERS = 50


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Benchmark(_Model):
    key: str = BENCHMARK
    label: str = BENCHMARK_LABEL
    metric: str
    estimate: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    versus_baseline: dict[str, Any] | None = None
    differences: list[dict[str, Any]] = []
    sentence: str


class DesignBased(_Model):
    label: str
    source: str
    metric: str
    folds: int
    repeats: int
    note: str | None = None
    families: list[dict[str, Any]]
    choice: dict[str, Any] | None = None
    sentence: str


class EvaluationArtifact(_Model):
    """The ``evaluation`` artifact (module docstring)."""

    purpose: str | None
    task: str
    scores_shown: bool
    note: str | None = None
    metric: str | None = None
    reported_family: str | None = None
    benchmark: Benchmark | None = None
    interpretable: dict[str, Any] | None = None
    decision_curve: dict[str, Any] | None = None
    subgroups: list[dict[str, Any]] = []
    shrinkage: dict[str, Any] | None = None
    internal_external: list[dict[str, Any]] = []
    inclusion: dict[str, Any] | None = None
    design_based: DesignBased | None = None
    hand_levers: list[str] = []
    estimates: dict[str, Any] | None = None  # inference: the selection sensitivity analysis
    sentences: list[str] = []


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".replace("-", "−")


def _oof(cv: Any, n: int, task: str, classes: Sequence[Any]) -> np.ndarray:
    """The first repeat's out-of-fold predictions in row order (classes in ``classes`` order); NaN
    where a row was not scored."""
    rows_of = cv._repeat()
    if task == "regression":
        out = np.full(n, np.nan)
        for f, r in zip(cv.predictions, rows_of):
            if r == 0:
                out[f.rows] = np.asarray(f.prediction, dtype=float)
        return out
    out = np.full((n, len(classes)), np.nan)
    for f, r in zip(cv.predictions, rows_of):
        if r != 0:
            continue
        block = np.zeros((len(f.rows), len(classes)))
        for j, c in enumerate(f.classes or classes):
            block[:, list(classes).index(c)] = np.asarray(f.prediction)[:, j]
        out[f.rows] = block
    return out


def _reported(fit: Mapping[str, Any]) -> str | None:
    """The family whose score is the result: the declared one, else the best on the substrate."""
    if fit.get("final_model"):
        return str(fit["final_model"])
    result = fit.get("result") or {}
    if result.get("family"):
        return str(result["family"])
    from turbotab.core.models.metrics import LOWER_IS_BETTER

    metric = fit.get("primary_metric")
    scored = [(m["family"], (m.get("compared_on") or (m.get("cv") or {}).get(metric) or {})
               .get("estimate")) for m in fit.get("models") or [] if m.get("cv")]
    scored = [(k, v) for k, v in scored if v is not None]
    if not scored:
        return None
    lower = metric in LOWER_IS_BETTER
    return (min if lower else max)(scored, key=lambda kv: kv[1])[0]


def evaluation_stage(ctx: StageContext) -> Bundle:
    state = ctx.state
    from turbotab.core.stages.modeling import _task

    task = _task(ctx)
    if state.purpose == "inference":
        return Bundle(data=_inference(ctx, task).model_dump(mode="json"))
    fit = ctx.inputs["fit"]
    comparison = (fit.objects or {}).get("comparison")
    if task not in SUPPORTED or comparison is None:
        reason = (f"The benchmark, the decision curve and the design-based scores are computed for a "
                  f"numeric or yes/no outcome; a {task.replace('_', '-')} outcome's are not yet."
                  if task not in SUPPORTED else "The fit made no comparison to evaluate.")
        return Bundle(data=EvaluationArtifact(purpose=state.purpose, task=task, scores_shown=True,
                                              note=reason).model_dump(mode="json"))
    return Bundle(data=_prediction(ctx, task, fit, comparison).model_dump(mode="json"))


# ── under prediction ─────────────────────────────────────────────────────────


def _prediction(ctx: StageContext, task: str, fit: Any, comparison: Mapping[str, Any]
                ) -> EvaluationArtifact:
    from dataclasses import replace

    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.baseline import versus_baseline
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import LABELS, cross_validate, higher_is_better
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, modeling_frame
    from turbotab.core.models.selection import OutOfFold, interpretable_cost, is_flexible
    from turbotab.core.models.validation import family_differences, score_words
    from turbotab.core.stages.explore import hand_levers
    from turbotab.core.stages.modeling import _baseline_cv, coded_outcome

    state = ctx.state
    data = fit.data
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    metric = str(data.get("primary_metric"))
    label = score_words(metric)
    pairs, repeat_of = comparison["pairs"], comparison["repeat_of"]
    substrates = dict(comparison["results"])
    groups = comparison.get("groups")
    ids = np.asarray(comparison["train_ids"], dtype=np.int64)
    target = state.target
    extra = [c for c in (state.intended_use.subgroups if state.intended_use else [])]
    cluster = state.clusters.column if state.clusters is not None else None
    survey = state.survey if state.survey is not None and state.survey.estimand == "population" \
        else None
    design_columns = ([c for c in (survey.weight, survey.strata, survey.psu, survey.cycle,
                                   survey.four_year_weight) if c] if survey is not None else [])
    ctx.progress(0.02, "Reading the training rows")
    with open_store(ctx) as store:
        present = set(store.columns)
        wanted = list(dict.fromkeys([*spec.inputs, target, *extra,
                                     *([cluster] if cluster else []), *design_columns]))
        frame = modeling_frame(store, [c for c in wanted if c in present], ids, outcome=target)
    X = frame[spec.inputs]
    y = np.asarray(coded_outcome(task, frame[target].to_numpy(), state.event))
    n = len(y)
    classes = [] if task == "regression" else sorted(pd.unique(y).tolist(), key=lambda v: str(v))
    families = [k for k in (state.models or []) if k in substrates]
    labels = {m["family"]: m["label"] for m in data.get("models") or []}
    labels[BENCHMARK] = BENCHMARK_LABEL
    reported = _reported(data)

    def fit_rows(model: Any, X_fit: Any, y_fit: Any, rows: Any = None) -> Any:
        take = slice(None) if rows is None else rows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_pipeline(model, X_fit, y_fit,
                                groups=None if groups is None else np.asarray(groups)[take])

    def check() -> None:
        if ctx.cancelled():
            raise Cancelled()

    out = EvaluationArtifact(purpose=state.purpose, task=task, scores_shown=True, metric=metric,
                             reported_family=reported)
    # ── the benchmark, on the comparison substrate ──
    from turbotab.core.models.linear import LINEAR
    from turbotab.core.models.validation import wide

    bench = bench_cv = None
    if wide(len(spec.inputs), n):
        # p ≫ n: least squares has no unique solution, with or without splines, so no regression
        # benchmark is fitted; the baseline still is (the fit's verdicts).
        out.note = (f"{len(spec.inputs):,} candidate predictors for {n:,} rows: a regression with "
                    f"splines on every predictor has no unique fit, so no benchmark is fitted and "
                    f"nothing is weighed against it.")
        out.sentences.append(out.note)
    else:
        ctx.progress(0.1, "Fitting the regression-with-splines benchmark on the same folds")
        bench_spec = replace(spec, levers={"forms": "rule", "variance_filter": "none", "keep": None,
                                           "imbalance": "none"}, selection=None)
        bench = build_pipeline(bench_spec, LINEAR, task, "prediction", n, len(spec.inputs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bench_cv = cross_validate(task, lambda: clone(bench), X, y, pairs, fit=fit_rows,
                                      before_fold=lambda i, k: check(), repeat_of=repeat_of,
                                      keep_predictions=True)
        base_cv = _baseline_cv(task, X, y, pairs, repeat_of)
        versus_metric = "r2" if task == "regression" else metric
        summary = bench_cv.summary(task, groups=groups)
        entry = summary.get(metric) or {}
        versus = versus_baseline(versus_metric, bench_cv.folds_of(versus_metric),
                                 base_cv.folds_of(versus_metric), test_share=bench_cv.test_share)
        diffs = family_differences(task, {BENCHMARK: bench_cv,
                                          **{k: substrates[k] for k in families}},
                                   None, labels, metric=metric)
        diffs = [d.model_dump(mode="json") for d in diffs if d.a == BENCHMARK or d.b == BENCHMARK]
        words = {"better": "better than", "no_better": "not shown better than",
                 "worse": "worse than"}[versus.verdict]
        said = (f"The regression-with-splines benchmark (every continuous predictor a restricted "
                f"cubic spline, knots by Harrell's rule, on the same in-fold preprocessing) scored "
                f"cross-validated {label} {_fmt(entry.get('estimate'))} on the families' "
                f"{len(pairs)} paired folds, {words} the no-predictor baseline.")
        out.benchmark = Benchmark(metric=metric, estimate=entry.get("estimate"),
                                  ci_low=entry.get("ci_low"), ci_high=entry.get("ci_high"),
                                  versus_baseline=versus.model_dump(mode="json"),
                                  differences=diffs, sentence=said)
        out.sentences.append(said)

    # ── what the interpretable model costs or gains ──
    check()
    flexible = [k for k in families if is_flexible(get_family(k))]
    first = [p for p, r in zip(pairs, repeat_of) if r == 0]
    oof = OutOfFold(task, X, y, first, units=groups)
    if bench_cv is not None:
        oof.predictions[BENCHMARK] = _oof(bench_cv, n, task, classes)
    for k in families:
        oof.predictions[k] = _oof(substrates[k], n, task, classes)
    if bench_cv is None:
        out.interpretable = None
    elif flexible:
        ctx.progress(0.3, "Weighing the interpretable model against the flexible ones")
        cost = interpretable_cost(task, metric, oof, BENCHMARK, flexible, labels=labels,
                                  metric_label=label, seed=int(getattr(state.split, "seed", 0) or 0))
        if cost is not None:
            out.interpretable = cost
            out.sentences.append(cost["text"])
    else:
        out.interpretable = {"text": "No flexible family was fitted, so there is nothing for the "
                                     "interpretable model to cost or gain against."}

    # ── intended use: the decision curve and the in-fold threshold ──
    use = state.intended_use
    reported_oof = oof.predictions.get(reported) if reported else None
    if (use is not None and use.use == "decision_support" and task == "binary"
            and reported_oof is not None and reported in pipelines):
        ctx.progress(0.45, "Drawing the decision curve and choosing the threshold in each fold")
        out.decision_curve = _decision(state, X, y, oof, reported, labels, pipelines[reported],
                                       first, fit_rows, check)
        out.sentences.append(out.decision_curve["sentence"])

    # ── subgroup performance ──
    if use is not None and use.subgroups and reported_oof is not None:
        from turbotab.core.models.decision_curve import subgroup_performance
        from turbotab.core.models.metrics import HEADLINE

        ok = np.isfinite(reported_oof if reported_oof.ndim == 1 else reported_oof.sum(axis=1))
        metrics = [m for m in (metric, HEADLINE.get(task), "r2" if task == "regression" else None)
                   if m]
        columns = {c: frame[c].to_numpy()[ok] for c in use.subgroups if c in frame.columns}
        out.subgroups = subgroup_performance(task, y[ok], reported_oof[ok], columns,
                                             classes=classes or None, metrics=metrics,
                                             reference=oof.reference[ok] if task == "regression"
                                             else None)

    # ── shrinkage, offered as model updating ──
    out.shrinkage = _shrinkage(state, task, data, fit, X, y)
    if out.shrinkage is not None and out.shrinkage.get("sentence"):
        out.sentences.append(out.shrinkage["sentence"])

    # ── internal–external cross-validation by cluster ──
    if cluster and cluster in frame.columns and reported in pipelines:
        models = {reported: pipelines[reported], **({BENCHMARK: bench} if bench is not None else {})}
        out.internal_external = _iecv(task, X, y, frame[cluster].to_numpy(), cluster, metric,
                                      models, labels, fit_rows, check)

    # ── the selection's inclusion frequencies ──
    if spec.selection and reported in pipelines:
        from turbotab.core.models.variable_selection import inclusion_frequencies

        ctx.progress(0.7, "Counting how often each predictor was kept across folds")
        found = inclusion_frequencies(pipelines[reported], X, y, pairs[:INCLUSION_FOLDS], fit=fit_rows)
        if found is not None:
            found["sentence"] = (f"Each candidate's inclusion frequency over {found['folds']} "
                                 f"training folds of the substrate (Heinze et al. 2018).")
            out.inclusion = found

    # ── design-based cross-validation under the surveyed population ──
    if survey is not None:
        ctx.progress(0.8, "Design-based cross-validation: folds of whole PSUs within strata")
        out.design_based = _design_based(state, task, metric, X, y, frame, survey,
                                         {**{k: pipelines[k] for k in families if k in pipelines},
                                          **({BENCHMARK: bench} if bench is not None else {})},
                                         labels, fit_rows, check, classes)
        out.sentences.append(out.design_based.sentence)

    # ── the levers set by hand after an outcome view ──
    hands = hand_levers(state)
    out.hand_levers = [h.sentence for h in hands]
    if hands:
        out.sentences.append(
            "The corrected score does not include the optimism of the lever"
            + ("s" if len(hands) > 1 else "") + " set by hand after an outcome view: "
            + " ".join(h.sentence for h in hands))
    ctx.progress(1.0, "Done")
    return out


def _decision(state: Any, X: pd.DataFrame, y: np.ndarray, oof: Any, reported: str,
              labels: Mapping[str, str], pipeline: Any, first: Sequence[Any], fit_rows: Any,
              check: Any) -> dict[str, Any]:
    from sklearn.base import clone

    from turbotab.core.models.decision_curve import (DEFAULT_RANGE, decision_curve, grid,
                                                     threshold_in_fold, useful_range)

    use = state.intended_use
    low = use.threshold_low if use.threshold_low is not None else DEFAULT_RANGE[0]
    high = use.threshold_high if use.threshold_high is not None else DEFAULT_RANGE[1]
    thresholds = grid(low, high)
    event = (y == sorted(pd.unique(y).tolist(), key=lambda v: str(v))[-1]).astype(float)
    risks = {}
    for k in [x for x in (reported, BENCHMARK) if x in oof.predictions]:
        P = oof.predictions[k]
        risks[k] = P[:, -1]
    ok = np.all(np.isfinite(np.column_stack(list(risks.values()))), axis=1)
    rows = decision_curve(event[ok], {k: r[ok] for k, r in risks.items()}, thresholds)
    folds = []
    for _, fit_mask, test_mask in first:
        check()
        model = fit_rows(clone(pipeline), X.iloc[np.flatnonzero(fit_mask)], y[fit_mask], fit_mask)
        level = list(model.classes_).index(sorted(model.classes_, key=lambda v: str(v))[-1])
        folds.append({"train_event": event[fit_mask],
                      "train_risk": model.predict_proba(X.iloc[np.flatnonzero(fit_mask)])[:, level],
                      "test_event": event[test_mask],
                      "test_risk": model.predict_proba(X.iloc[np.flatnonzero(test_mask)])[:, level]})
    chosen = threshold_in_fold(folds, thresholds, declared=use.threshold)
    useful = useful_range(rows, reported)
    name = labels.get(reported, reported)
    span = (f"it beats treating everyone and no one from {useful[0]:g} to {useful[1]:g}"
            if useful else "it beats neither treating everyone nor no one anywhere in the range")
    how = (f"at the declared threshold {use.threshold:g}" if use.threshold is not None else
           f"at the threshold Youden's J chose in each training fold (median "
           f"{_fmt(chosen['median_threshold'])})")
    sentence = (f"Decision curve analysis of {name}'s out-of-fold risks over thresholds from "
                f"{low:g} to {high:g}: {span} (Vickers & Elkin 2006); {how}, the held-out "
                f"sensitivity was {_fmt(chosen['sensitivity'])}, the specificity "
                f"{_fmt(chosen['specificity'])} and the net benefit {_fmt(chosen['net_benefit'])} "
                f"per row.")
    return {"family": reported, "low": low, "high": high, "rows": rows, "useful": useful,
            "in_fold": chosen, "sentence": sentence}


def _shrinkage(state: Any, task: str, data: Mapping[str, Any], fit: Any, X: pd.DataFrame,
               y: np.ndarray) -> dict[str, Any] | None:
    """The offer of uniform shrinkage for the unpenalized regression, and, once answered, the
    shrunk model (``models.decision_curve.shrinkage``)."""
    from turbotab.core.models.decision_curve import shrinkage, slope_of

    entry = next((m for m in data.get("models") or [] if m.get("family") == "linear"), None)
    final = ((fit.objects or {}).get("fitted") or {}).get("linear")
    if entry is None or final is None:
        return None
    slope, how = slope_of(entry)
    if slope is None:
        return None
    offer = {"factor": slope, "how": how,
             "decision": {"kind": "set_updating", "method": "shrinkage"},
             "label": f"Shrink the regression's coefficients by {slope:.3f} (model updating)"}
    if state.updating is None or state.updating.method != "shrinkage":
        offer["sentence"] = None
        return offer
    from turbotab.core.models.linear import model_matrix

    model = final.steps[-1][1]
    coef = np.asarray(model.coef_, dtype=float).ravel()
    matrix = model_matrix(final, X)
    yy = y.astype(float) if task == "regression" else (
        y == sorted(pd.unique(y).tolist(), key=lambda v: str(v))[-1]).astype(float)
    done = shrinkage(task, matrix, yy, coef, slope)
    offer.update(done)
    offer["sentence"] = (f"As model updating, the regression's coefficients were multiplied by "
                         f"{slope:.3f}, {how}, and its intercept re-estimated with the shrunk "
                         f"linear predictor as an offset ({_fmt(done['intercept'])}).")
    return offer


def _iecv(task: str, X: pd.DataFrame, y: np.ndarray, cluster: np.ndarray, column: str,
          metric: str, models: Mapping[str, Any], labels: Mapping[str, str], fit_rows: Any,
          check: Any) -> list[dict[str, Any]]:
    """Leave one cluster out (Collins et al. 2024, Box 4) for the reported family and the benchmark,
    with the random-effects summary (``validation.internal_external``)."""
    from sklearn.base import clone

    from turbotab.core.models.metrics import cross_validate
    from turbotab.core.models.validation import internal_external

    values = pd.Series(np.asarray(cluster, dtype=object)).astype(str).to_numpy()
    levels = sorted(set(values.tolist()))
    if not 2 <= len(levels) <= MAX_IECV_CLUSTERS:
        return []
    pairs = [(i, values != c, values == c) for i, c in enumerate(levels)]
    out = []
    for key, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate(task, lambda _p=pipeline: clone(_p), X, y, pairs, fit=fit_rows,
                                before_fold=lambda i, k: check(), keep_predictions=True)
        found = internal_external(task, cv, levels, column, metric=metric)
        out.append({"family": key, "label": labels.get(key, key),
                    **found.model_dump(mode="json")})
    return out


def _design_based(state: Any, task: str, metric: str, X: pd.DataFrame, y: np.ndarray,
                  frame: pd.DataFrame, survey: Any, pipelines: Mapping[str, Any],
                  labels: Mapping[str, str], fit_rows: Any, check: Any,
                  classes: Sequence[Any]) -> DesignBased:
    """Design-based cross-validation (``models.design_cv``) of each family, the benchmark and the
    baseline."""
    from sklearn.base import clone

    from turbotab.core.methods.survey import analysis_weights
    from turbotab.core.models import design_cv as D
    from turbotab.core.models.metrics import cross_validate
    from turbotab.core.models.validation import loss_rows, score_words
    from turbotab.core.stages.modeling import baseline_model

    weights = analysis_weights(frame, survey.weight, survey.cycle, survey.four_year_weight).weights
    strata = frame[survey.strata].to_numpy() if survey.strata else None
    psu = frame[survey.psu].to_numpy() if survey.psu else None
    k = int(getattr(state.split, "folds", None) or 5)
    seed = int(getattr(state.split, "seed", 0) or 0)
    ok = np.isfinite(weights) & (weights > 0)
    note = None
    if not ok.all():
        note = (f"{int((~ok).sum()):,} rows with no positive weight are outside the population "
                f"the scores describe.")
    Xo, yo, wo = X.iloc[np.flatnonzero(ok)], y[ok], weights[ok]
    so = None if strata is None else strata[ok]
    po = None if psu is None else psu[ok]
    columns, folds_used, fold_note = [], k, None
    for r in range(D.REPEATS):
        fold, folds_used, fold_note = D.design_folds(so, po, len(yo),
                                                     k, seed + r)
        columns.append(fold)
    pairs = []
    repeat_of = []
    for r, fold in enumerate(columns):
        for f, fit_mask, test_mask in D.fold_pairs(fold):
            pairs.append((f, fit_mask, test_mask))
            repeat_of.append(r)
    models = {**pipelines, "baseline": None}
    labels = {**labels, "baseline": "No-predictor baseline"}
    results: dict[str, dict[str, Any]] = {}
    losses_first: dict[str, np.ndarray] = {}
    tests = [t for _, _, t in pairs]
    for key, pipeline in models.items():
        check()
        make = (lambda: baseline_model(task)) if pipeline is None else (lambda _p=pipeline: clone(_p))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate(task, make, Xo, yo, pairs, fit=None if pipeline is None else fit_rows,
                                repeat_of=repeat_of, keep_predictions=True)
        loss = []
        first_loss = np.full(len(yo), np.nan)
        first_pred: Any = None
        for pred, r, t in zip(cv.predictions, repeat_of, tests):
            l = loss_rows(task, metric, pred.y, pred.prediction, classes=pred.classes)
            loss.append(D.weighted_mean(l, wo[t]))
            if r == 0:
                first_loss[pred.rows] = l
        cvp = _oof(cv, len(yo), task, classes)
        cal = D.weighted_calibration(task, yo, cvp, wo, classes or None)
        results[key] = {"family": key, "label": labels.get(key, key), "fold_scores": loss,
                        "estimate": float(np.mean(loss)),
                        "pooled_first": D.weighted_mean(first_loss, wo),
                        "se_first": D.svymean_se(first_loss, wo, so, po),
                        "calibration": cal}
        losses_first[key] = first_loss
    base = results["baseline"]["fold_scores"]
    share = float(np.mean([t.sum() / f.sum() for _, f, t in pairs]))
    for key, entry in results.items():
        if key == "baseline":
            continue
        test = D.paired(entry["fold_scores"], base, test_share=share, repeats=D.REPEATS)
        entry["versus_baseline"] = test.model_dump(mode="json")
    choice = D.design_bbc({k: v for k, v in losses_first.items() if k not in ("baseline", BENCHMARK)},
                          wo, so, po, seed=seed)
    label = score_words(metric)
    best = min((k for k in results if k not in ("baseline",)), key=lambda k: results[k]["estimate"])
    sentence = (f"Under the surveyed-population answer, performance is the population's: "
                f"{D.LABEL} ({D.SOURCE}) with {folds_used} folds of whole PSUs within strata, drawn "
                f"{D.REPEATS} times, every loss, calibration and comparison weighted by "
                f"`{survey.weight}`; {labels.get(best, best)} scored a weighted {label} of "
                f"{_fmt(results[best]['estimate'])} against the baseline's "
                f"{_fmt(results['baseline']['estimate'])}.")
    if choice is not None:
        sentence += (f" Corrected for choosing among the families (PSUs resampled within strata), "
                     f"{_fmt(choice['corrected'])} (95% interval {_fmt(choice['corrected_low'])} "
                     f"to {_fmt(choice['corrected_high'])}).")
    return DesignBased(label=D.LABEL, source=D.SOURCE, metric=metric, folds=folds_used,
                       repeats=D.REPEATS, note=" ".join(x for x in (note, fold_note) if x) or None,
                       families=list(results.values()), choice=choice, sentence=sentence)


# ── under inference ──────────────────────────────────────────────────────────


def _inference(ctx: StageContext, task: str) -> EvaluationArtifact:
    out = EvaluationArtifact(purpose="inference", task=task, scores_shown=False,
                             note=NO_SCORE_UNDER_INFERENCE)
    spec_answer = ctx.state.selection
    if spec_answer is None or not spec_answer.sensitivity or spec_answer.method == "none":
        return out
    if task not in SUPPORTED:
        out.estimates = {"note": f"The selection sensitivity analysis reads a numeric or yes/no "
                                 f"outcome; a {task.replace('_', '-')} outcome's is not run."}
        return out
    out.estimates = selection_sensitivity(ctx, task)
    out.sentences.append(out.estimates.get("sentence") or "")
    return out


def selection_sensitivity(ctx: StageContext, task: str) -> dict[str, Any]:
    """Backward elimination by pooled Wald tests over the declared model's covariates, the exposure
    kept (``models.variable_selection.backward_wald``), on every analyzed row; across the imputed
    copies under multiple imputation."""
    from sklearn.base import clone

    from turbotab.core.estimand import current_estimand, exposures_of
    from turbotab.core.models.linear import LINEAR
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, modeling_frame
    from turbotab.core.models.variable_selection import (INFERENCE_ALPHA, backward_wald,
                                                         term_groups)
    from turbotab.core.stages.modeling import coded_outcome

    state = ctx.state
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    objects = fit.objects or {}
    ids = objects.get("every_row_ids")
    target = state.target
    with open_store(ctx) as store:
        frame = modeling_frame(store, [*spec.inputs, target], ids, outcome=target)
    y = np.asarray(coded_outcome(task, frame[target].to_numpy(), state.event))
    imputations = objects.get("imputations")
    if imputations is not None:
        frames = [f[spec.inputs] for f in imputations["frames"]]
        outcomes = imputations.get("outcomes") or [y] * len(frames)
    else:
        keep = frame[spec.inputs].notna().all(axis=1).to_numpy()
        frames, outcomes = [frame.loc[keep, spec.inputs]], [y[keep]]
    template = build_pipeline(spec, LINEAR, task, "inference", len(y), len(spec.inputs))
    matrices = []
    for X_k, y_k in zip(frames, outcomes):
        head = clone(template)[:-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = head.fit_transform(X_k, np.asarray(y_k))
        matrices.append(pd.DataFrame(np.asarray(m, dtype=float), columns=[str(c) for c in m.columns]))
    terms = term_groups(list(matrices[0].columns), spec.inputs)
    current = current_estimand(state)
    exposures = exposures_of(state, current) if current is not None else []
    forced = [t for t in terms if t in exposures]
    found = backward_wald(task, matrices, outcomes, terms, forced)
    dropped = [s["dropped"] for s in found["path"] if s["dropped"]]
    from turbotab.core.voice import listing

    pooled = ("pooled across the imputed copies by Rubin's rules" if len(frames) > 1
              else "on the complete rows")
    found["sentence"] = (
        f"As a labeled sensitivity analysis (not the reported model), backward elimination at "
        f"α = {INFERENCE_ALPHA} with Wald tests {pooled} (Wood, White & Royston 2008), the exposure "
        f"kept, removed {listing(dropped) if dropped else 'no covariate'}; the declared model is the "
        f"reported one.")
    found["forced"] = forced
    return found


# ── what a client is shown of the fit under inference (ruling 13; §1 row 11) ──

# The model-level fields that hold a score: LEASH's list (``estimand.MODEL_SCORES``), which the
# served gate withholds from a fit whose plan is open, is the one list; ruling 13 withholds them from
# every fit served under inference (wave 2b integration).
SCORE_FIELDS = MODEL_SCORES
SCORE_LINKS = ("proper_score_primary", "calibration_not_assessed", "p_much_greater_n",
               "bootstrap_resamples", "time_to_event_horizon", "families_compared_on_substrate",
               "choice_among_families_bbc", "no_holdout_declares_corrected")


def withhold_scores(artifact: Any, state: Any) -> Any:
    """The served fit under inference with every cross-validated score withheld (each family's
    scores, calibration and verdict against the baseline, the comparisons, the selection and the
    declared result, and the concerns that quote them), and the reason in ``cv_definition``. The
    stage's own artifact keeps them for the stages that read it (the explanations' floor)."""
    if not isinstance(artifact, dict) or getattr(state, "purpose", None) != "inference":
        return artifact
    from turbotab.core.estimand import without_scores

    out = dict(artifact)
    models = []
    for m in out.get("models") or []:
        m = without_scores(m)  # the cross-validated scores, the baseline's, calibration, optimism
        quoted = set(m.get("score_concerns") or [])
        m["concerns"] = [c for c in m.get("concerns") or [] if c not in quoted]
        m["score_concerns"] = []
        models.append(m)
    out["models"] = models
    for key in (*FIT_SCORES, "tension", "headline_metric", "headline_label", "ranking", "wide",
                "se_definition"):
        if key in out:
            out[key] = None
    out["comparisons"] = []
    out["chain"] = [c for c in out.get("chain") or [] if c.get("relation") not in SCORE_LINKS]
    out["cv_definition"] = NO_SCORE_UNDER_INFERENCE
    return out


def design_sentence(d: Any, state: Any) -> str:
    """The design-based cross-validation clause the survey answer implies under prediction."""
    from turbotab.core.models import design_cv as D

    if getattr(d, "estimand", None) == "population":
        return (f"Performance was estimated for the surveyed population by {D.LABEL} ({D.SOURCE}): "
                f"folds of whole PSUs within strata, every loss, calibration and comparison weighted "
                f"by the survey weight.")
    return ("Performance was estimated on these rows, unweighted: the procedure's performance on "
            "these participants, not the surveyed population's.")


__all__ = ["BENCHMARK", "BENCHMARK_LABEL", "EVALUATION_READS", "EvaluationArtifact",
           "NO_SCORE_UNDER_INFERENCE", "design_sentence", "evaluation_stage",
           "selection_sensitivity", "withhold_scores"]
