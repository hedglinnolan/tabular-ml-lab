"""Consequence previews for the domain methods and the causal lanes (V2 definition of done §2;
BLUEPRINT §13, each method's storyboard): what the method would do to the user's own rows, drawn in
the closed vocabulary with the method's own labeled steps.

| Kind                     | Views (primary first)                                                 |
|--------------------------|-----------------------------------------------------------------------|
| ``set_scales``           | table focus, the method's story: items as answered → reverse-keyed    |
|                          | items turned over → the score → the calibrated score · distribution of|
|                          | the score, as summed and calibrated · lineage, items into the score   |
| ``set_usual_intake``     | distribution, the shrinkage story: a single day's recall → each       |
|                          | person's mean of days → usual intake (the NCI method)                 |
| ``set_causal``           | distribution of the estimator's own propensity, story: the unexposed's|
|                          | → the exposed's (the overlap) · row flow of a trim                    |
| ``set_time_varying``     | distribution of the stabilized weights, story: the exposure's weights |
|                          | → times loss to follow-up's → truncated                               |
| ``set_measurement_error``| relationship: each person's mean of recalls against the calibrated    |
|                          | exposure, story: the recalls around each mean                         |
| ``set_batch``            | relationship: the first principal component by batch, before and after|
|                          | the correction · lineage of the batch step                            |

**Preview = what will happen.** Each builder calls the downstream stage's own functions on the
answer's state, over the stage's own upstream artifacts (``consequences.stage_context``), on the
rows the purpose allows. Only the parts that add variance are left out: no bootstrap, no replicate
weights, no outcome model. Every number a preview shows is a point the stage computes the same way
(``tests/acceptance/test_previews_kinds.py``). Where a table is larger than a preview's budget,
the rows are a stated sample (the basis says which); the stage reads every row.

**Outcome-blind.** None of these previews reads the outcome's relation to anything: the propensity,
the weights, the calibration and the batch correction are the methods' design steps, which are
declared before any estimate is seen (Rubin 2008, "design trumps analysis").

Importing this module registers the builders.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    MAX_VIEWS, DistributionFrame, DistributionView, FitLine, FrameRow, HistogramData, LineageView,
    Mark, PreviewContext, RelationshipFrame, RelationshipView, RowFlowView, RowStep, TableFocusView,
    TableFrame, TableRow, after_state, fmt_count, fmt_value, register_consequence, stage_context,
)
from turbotab.core.plan_previews import (
    ask, block_and_record, caption, fit_design, frame_label, histogram, lineage_change, names, num,
    points, pool, shared_edges, specifies_the_model, tick, title, whole,
)

TABLE_ROWS = 6
TABLE_COLUMNS = 12


def _json(value: Any) -> Any:
    from turbotab.core.datastore import json_safe

    return json_safe(value)


# ── set_scales ───────────────────────────────────────────────────────────────


def _repeat(frame: pd.DataFrame, spec: Any) -> tuple[Any, Any]:
    """A repeat administration's score and a calibration substudy's reference, as the scales stage
    reads them (``stages.scales._retest_score``; the reference column as numbers)."""
    from turbotab.core.stages.scales import _retest_score

    retest = _retest_score(frame, spec) if spec.reliability == "test_retest" else None
    reference = (pd.to_numeric(frame[spec.reference], errors="coerce").to_numpy(dtype=float)
                 if spec.reference and spec.reference in frame.columns else None)
    return retest, reference


def scale_numbers(spec: Any, keyed: pd.DataFrame, design: Any, state: Any,
                  corrected: Sequence[Any]) -> dict[str, Any]:
    """The scale's reliability as the scales stage reads it (``stages.scales.reliability_of``: ω,
    a repeat administration's ICC, or a substudy's), and, when its correction is eligible, its
    calibrated score and λ given every other column of the model's matrix
    (``methods.scales.calibrate_jointly``, the stage's own algebra): the point the stage computes
    before its bootstrap. With a blank among the model's inputs the stage calibrates over the
    imputed copies (or refuses a single fill), so no λ is drawn here."""
    from turbotab.core.methods import scales as S
    from turbotab.core.stages.scales import reliability_of

    retest, reference = _repeat(design.frame, spec)
    rel, concerns = reliability_of(spec, [keyed], retest, reference)
    out: dict[str, Any] = {"reliability": rel, "concerns": concerns, "lambda": None,
                           "calibrated": None}
    names_ = [s.name for s in corrected]
    if spec.name not in names_ or spec.name not in design.matrix.columns:
        return out
    if design.frame[list(design.spec.inputs)].isna().to_numpy().any():
        out["blanks"] = True
        return out
    matrix = design.matrix
    js = [list(matrix.columns).index(n) for n in names_ if n in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    W, Z = values[:, js], np.delete(values, js, axis=1)
    errors = []
    for s in corrected:
        found = reliability_of(s, [S.keyed_items(design.frame, s.items, s.reverse, s.low, s.high)],
                               None, None)[0]
        omega = found.get("value")
        if omega is None:
            return out
        score = S.score_items(S.keyed_items(design.frame, s.items, s.reverse, s.low, s.high)
                              .to_numpy(dtype=float), s.scoring)
        errors.append((1.0 - float(omega)) * float(np.var(score, ddof=1)))
    try:
        cal = S.calibrate_jointly(W, Z, errors, [None] * len(js))
    except S.ScaleRefused as refused:
        out["refused"] = str(refused)
        return out
    j = names_.index(spec.name)
    out["lambda"] = float(cal.attenuation[j])
    out["calibrated"] = cal.calibrated[:, j]
    return out


def _corrected(after: Any, ctx: PreviewContext) -> list[Any]:
    """The scales whose correction the scales stage would run (``stages.scales._why_not``): under
    inference, a reflective score from its internal consistency, with no survey population block
    and no blanks filled once."""
    if getattr(after, "purpose", None) != "inference":
        return []
    survey = getattr(after, "survey", None)
    if survey is not None and survey.estimand == "population":
        return []
    return [s for s in after.scales or [] if s.correction == "regression_calibration"
            and s.reliability == "internal_consistency" and s.kind == "reflective"]


def _scale_columns(scales: Sequence[Any]) -> list[str]:
    return [c for s in scales for c in s.columns()]


def scales_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.methods import scales as S

    if not decision.scales:
        return []
    ids = pool(ctx)
    if ids is None:
        ids = ctx.sample_row_ids()
    after = after_state(decision, ctx)
    current = {s.name: s for s in (ctx.state.scales or [])}
    spec = next((s for s in decision.scales if current.get(s.name) != s), decision.scales[0])
    then = fit_design(ctx, after, ids, extra=_scale_columns(decision.scales))
    if then is None:
        return []
    now = fit_design(ctx, ctx.state, ids)
    items = [c for c in spec.items if c in then.frame.columns]
    if len(items) != len(spec.items):
        return []
    raw = then.frame[items]
    keyed = S.keyed_items(raw, spec.items, spec.reverse, spec.low, spec.high)
    score = S.score_items(keyed.to_numpy(dtype=float), spec.scoring)
    found = scale_numbers(spec, keyed, then, after, _corrected(after, ctx))
    rel = found["reliability"]
    lam, calibrated = found["lambda"], found["calibrated"]
    if found.get("blanks"):
        ctx.read["note"] = ("Blanks among the model's inputs: the correction is computed over the "
                            "imputed copies when the scales run, so λ is not drawn here.")
    ctx.read["scales"] = {"name": spec.name, "omega": rel.get("value"), "alpha": rel.get("alpha"),
                          "coefficient": rel.get("coefficient"), "lambda": lam}
    # The story on real rows: the first rows of the sample, the items as answered, turned over,
    # summed, and calibrated.
    shown = [c for c in spec.items][: TABLE_COLUMNS - 2]
    picked = list(range(min(TABLE_ROWS, len(raw))))
    row_ids = [int(r) for r in raw.index[picked]]

    def frame_rows(values: pd.DataFrame, extra: Mapping[str, Any] | None = None) -> list[FrameRow]:
        out = []
        for i, rid in zip(picked, row_ids):
            cells = {c: _json(values.iloc[i][c]) for c in shown}
            for k, col in (extra or {}).items():
                cells[k] = _json(col[i])
            out.append(FrameRow(row_id=rid, values=cells))
        return out

    score_col = spec.name
    story = [TableFrame(label=frame_label("Items as answered"), columns=shown,
                        rows=frame_rows(raw)),
             TableFrame(label=frame_label(f"{len(spec.reverse)} reverse-keyed items turned over"
                                          if spec.reverse else "Every item keyed one way"),
                        columns=shown, rows=frame_rows(keyed))]
    after_cols = [*shown, score_col]
    extra: dict[str, Any] = {score_col: score}
    if calibrated is not None:
        verb = "Summed" if spec.scoring == "sum" else "Averaged"
        story.append(TableFrame(label=frame_label(f"{verb} into {score_col}"),
                                columns=after_cols, rows=frame_rows(keyed, {score_col: score})))
        cal_col = f"{score_col} (calibrated)"
        after_cols = [*after_cols, cal_col]
        extra[cal_col] = calibrated
    rows = []
    for i, rid in zip(picked, row_ids):
        before = {c: _json(raw.iloc[i][c]) for c in shown}
        after_cells = {c: _json(keyed.iloc[i][c]) for c in shown}
        for k, col in extra.items():
            after_cells[k] = _json(col[i])
        rows.append(TableRow(row_id=rid, before=before, after=after_cells))
    changed = [(rid, c) for rid in row_ids for c in [*[r for r in spec.reverse if r in shown],
                                                     *extra]]
    label = rel.get("label") or "ω"
    alpha = rel.get("alpha")
    if rel.get("value") is not None:
        text = (f"{tick(score_col)} {'sums' if spec.scoring == 'sum' else 'averages'} "
                f"{len(spec.items)} items; {label} {fmt_value(round(rel['value'], 2))}"
                + (f", α {fmt_value(round(alpha, 2))} (customary)." if alpha is not None else "."))
    else:
        text = (f"{tick(score_col)} {'sums' if spec.scoring == 'sum' else 'averages'} "
                f"{len(spec.items)} items; no reliability from them: {spec.kind} index.")
    views: list[Any] = [TableFocusView(
        title=title(f"From items to {tick(score_col)}"),
        caption=caption(text),
        emphasis=[score_col],
        columns_before=shown,
        columns_after=after_cols,
        rows=rows,
        changed=changed,
        n_affected_columns=len(spec.items) + len(extra),
        story=story,
    )]
    if calibrated is not None and lam is not None:
        edges = shared_edges(score, calibrated)
        others = [c for c in then.columns if c != score_col]
        views.append(DistributionView(
            title=title(f"{tick(score_col)}, as summed and calibrated"),
            caption=caption(f"Calibrated given {names(others, 2)}: λ = {fmt_value(round(lam, 2))}, "
                            f"so the scores draw toward their mean."),
            emphasis=[score_col],
            column=score_col,
            before=histogram(score, edges=edges),
            after=histogram(calibrated, edges=edges),
            before_label=f"{score_col} as summed",
            after_label=f"{score_col} calibrated, E[X | W, Z]",
        ))
    arrive, leave = lineage_change(now.lineage if now is not None else None, then.lineage)
    if arrive or leave:
        views.append(LineageView(
            title=title("Items into one score"),
            caption=caption(f"{fmt_count(len(spec.items))} items leave the model; "
                            f"{tick(score_col)} enters as one column."),
            emphasis=[score_col, *spec.items][:12],
            before=now.lineage if now is not None else None,
            after=then.lineage,
        ))
    survey = getattr(after, "survey", None)
    if (spec.correction != "none" and getattr(after, "purpose", None) == "inference"
            and getattr(survey, "estimand", None) == "population"):
        # MODELING_SEQUENCE §4: the correction has no design-based estimator, so the scales stage
        # blocks and records it (``stages.scales.population_block``); the score still enters.
        from turbotab.core.stages.scales import population_block

        block_and_record(ctx, *population_block(), decision)
    return views[:MAX_VIEWS]


# ── set_usual_intake ─────────────────────────────────────────────────────────

PERCENTILE_GRID = tuple(range(1, 100))
MAX_PERSONS = 5000  # persons a usual-intake preview fits on; a larger table is sampled


def quantile_histogram(quantiles: Mapping[Any, float], n: int,
                       edges: Sequence[float]) -> HistogramData:
    """The people expected in each bin of ``edges`` under a distribution given by its percentiles
    1 … 99 (linear between them, the outer 1% at each end placed at the 1st and 99th): the curve the
    method reports, drawn on the same axis as the recalls."""
    pairs = sorted((float(p), float(q)) for p, q in quantiles.items())
    ps = np.array([p for p, _ in pairs])
    qs = np.array([q for _, q in pairs])
    e = np.asarray(edges, dtype=float)
    cdf = np.interp(e, qs, ps / 100.0, left=0.0, right=1.0)
    cdf[-1] = 1.0
    cdf[0] = 0.0
    return HistogramData(edges=[float(x) for x in e], counts=whole(np.diff(cdf) * n))


def usual_intake_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.usual_intake import analysis, eligible_rows, gate

    if decision.model == "none":
        ctx.read["note"] = f"No usual-intake distribution is estimated for {tick(decision.nutrient)}."
        return []
    after = after_state(decision, ctx)
    sctx = stage_context(ctx, after, ("oriented", "findings", "structure", "working"))
    if sctx is None:
        return []
    structure = getattr(sctx.inputs.get("structure"), "data", sctx.inputs.get("structure"))
    fmt, reason = gate(after, structure)
    spec = (after.usual_intake or {}).get(decision.nutrient)
    if spec is None:
        return []
    working = getattr(sctx.inputs["working"], "data", sctx.inputs["working"])
    # One working row per person (a wide table, or a long one combined per person): a sample of
    # rows is a sample of people, each with every recall.
    per_person = fmt == "wide" or (fmt == "long" and dict(working).get("aggregation") is not None)
    with open_store(sctx) as store:
        rows = eligible_rows(sctx, store)
        sampled = None
        if per_person and len(rows) > MAX_PERSONS:
            sampled = (len(rows), MAX_PERSONS)
            rows = np.sort(np.random.default_rng(0).choice(rows, size=MAX_PERSONS, replace=False))
        found = analysis(sctx, store, decision.nutrient, spec, fmt, reason, rows,
                         replicate=False, percentiles=PERCENTILE_GRID)
    if not found.applies:
        ctx.read["note"] = found.refused or found.methods
        return []
    n = int(found.n_persons)
    day = {int(k): float(v) for k, v in found.day_one.items()}
    mean = {int(k): float(v) for k, v in found.mean_of_days.items()}
    usual = {int(k): float(v.value) for k, v in found.percentiles.items()}
    edges = shared_edges(np.array([day[1], day[99], mean[1], mean[99], usual[1], usual[99]]))
    span = {name: (q[5], q[95]) for name, q in (("day", day), ("mean", mean), ("usual", usual))}
    marks = ([Mark(value=float(decision.cutoff), label=f"{decision.cutoff_kind or 'cut-off'} "
                                                        f"{num(decision.cutoff)}")]
             if decision.cutoff is not None else [])
    model = "two-part model" if decision.model == "two_part" else "amount-only model"
    words = (f"{tick(decision.nutrient)} 5th–95th percentile: one day {num(span['day'][0])}–"
             f"{num(span['day'][1])}, mean of days {num(span['mean'][0])}–{num(span['mean'][1])}, "
             f"usual {num(span['usual'][0])}–{num(span['usual'][1])}.")
    ctx.read["usual_intake"] = {"day_one": span["day"], "mean_of_days": span["mean"],
                                "usual": span["usual"], "n_persons": n}
    if sampled is not None:
        ctx.read["basis"] = (f"The usual-intake model fitted on a random sample of "
                             f"{sampled[1]:,} of the {sampled[0]:,} eligible participants "
                             f"(seed 0); the analysis fits every one.")
    else:
        ctx.read["basis"] = (f"The usual-intake model fitted on all {n:,} eligible participants, "
                             f"without its variance replicates.")
    return [DistributionView(
        title=title(f"Usual intake of {tick(decision.nutrient)}"),
        caption=caption(words),
        emphasis=[decision.nutrient],
        column=decision.nutrient,
        before=quantile_histogram(day, n, edges),
        after=quantile_histogram(usual, n, edges),
        before_label="a single day's recall",
        after_label=f"usual intake (NCI {model})",
        marks=marks,
        story=[DistributionFrame(label=frame_label("Each person's mean of their days"),
                                 hist=quantile_histogram(mean, n, edges),
                                 x_label=f"{decision.nutrient}, mean of days")],
    )]


# ── set_causal ───────────────────────────────────────────────────────────────

# The preview's budget for the estimator's own nuisance learner (V2 gate 5: a preview answers in
# under a second). Main-terms regression is the estimator's own on every row (a fifth of a second on
# the NHANES export). A flexible learner costs about a second or more however few the rows (200
# trees, a hundred boosting rounds or a cross-validated lasso path, in each of five folds), so its
# preview reads the causal card's main-terms propensity, cross-fitted as ``causal_design`` reads
# it, and the basis says so: the learner's own is read when the lane runs, which withholds the
# estimate on its own positivity (``stages.causal.causal_stage``).
CARD_LEARNERS = ("nuisance_forest", "lasso", "untuned_boosted_trees")


def causal_numbers(sctx: Any, spec: Any, *, budget: bool = True) -> dict[str, Any] | None:
    """The propensity (or, for a numeric exposure, its residual variation) the causal stage reads
    before any estimate (``stages.causal.causal_stage``: the same rows, learner, folds, seed and
    bound), with what the answer's trim would keep. ``budget``: within the preview's budget
    (:data:`CARD_LEARNERS`), where ``source`` says which propensity was read."""
    from turbotab.core.models import causal as est
    from turbotab.core.stages.causal import Withheld, _design, default_learner, prepare

    try:
        prep = prepare(sctx)
        design, _, _ = _design(sctx, prep, sample_only=spec.sample_only)
    except Withheld as exc:
        return {"withheld": exc.reason, "asked": exc}
    n = len(prep.y)
    learner = None if spec.method == "pds_lasso" else (spec.learner or default_learner(n))
    exact = spec.method == "pds_lasso" or (spec.method == "tmle" and learner == "linear")
    source = "card" if budget and not exact and learner in CARD_LEARNERS else "own"
    weights = None if design is None else design.weight
    groups = None if design is None else design.groups
    d = prep.d
    X = prep.X if prep.X.shape[1] else np.zeros((n, 1))
    m = n
    out: dict[str, Any] = {"exposure": prep.exposure, "kind": prep.exposure_kind, "n": n,
                           "learner": learner, "source": source, "level": prep.level}
    if source == "card":
        # ``stages.causal._causal_design``: five folds, one split, the split's seed.
        seed = int(getattr(sctx.state.split, "seed", 0) or 0) if sctx.state.split is not None else 0
        splits = est.sample_splits(m, 5, 1, seed=seed, groups=groups)
        factory, own_seed = est.learner_factory("linear"), seed
        model = "main-terms logistic regression, cross-fitted over 5 folds (the causal card's)"
    else:
        seed = int(spec.seed)
        splits = est.sample_splits(m, spec.folds, spec.repetitions, seed=seed, groups=groups)
        own_seed = seed if spec.method == "dml_irm" else seed + 500
        name = "linear" if spec.method == "pds_lasso" or learner == "linear" else str(learner)
        factory = est.learner_factory(name)
        model = f"{est.LEARNER_WORDS[name]}, cross-fitted over {spec.folds} folds"
    if prep.exposure_kind == "binary":
        if exact:
            G = np.column_stack([np.ones(m), X])
            g = 0.5 * (1.0 + np.tanh(0.5 * (G @ est.logistic_fit(G, d, weights))))
            model = "main-terms logistic regression on every row"
        else:
            [g] = est.cross_fit(factory, True, X, d, splits[0], weights, own_seed)
        target = "ATT" if spec.population == "exposed" else "ATE"
        found = est.overlap(g, d, est.tmle_gbound(m), weights, target=target)
        out.update(g=g, d=d, overlap=found, model=model, target=target)
        if spec.trim is not None:
            out["n_trimmed"] = int((~est.trim_rows(g, spec.trim)).sum())
    else:
        [mu] = est.cross_fit(factory, False, X, d, splits[0], weights, own_seed)
        out.update(variation=est.variation(d, mu, weights), d=d, fitted=mu, model=model)
    return out


def _causal_basis(found: Mapping[str, Any]) -> str:
    from turbotab.core.models.causal import LEARNER_NAMES

    if found["source"] == "card":
        learner = LEARNER_NAMES[found["learner"]]
        return (f"Overlap read by the causal card's main-terms logistic propensity, cross-fitted on "
                f"all {found['n']:,} complete analyzed rows; the {learner}'s own "
                f"propensity is read when the lane runs, before any estimate. No outcome read.")
    return (f"The estimator's own propensity on all {found['n']:,} complete analyzed rows, its "
            f"learner and folds; no outcome read.")


def causal_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    if decision.method == "none":
        ctx.read["note"] = "No causal estimate beside the primary model; nothing else changes."
        return []
    after = after_state(decision, ctx)
    sctx = stage_context(ctx, after, ("working", "split", "target_info"))
    if sctx is None or getattr(after, "purpose", None) != "inference":
        return []
    found = causal_numbers(sctx, decision)
    if found is None:
        return []
    if "withheld" in found:
        # The lane's own reason; where it is an ask (a reading the propensity rests on), the
        # preview offers each confirmation as the lane's card does (BLUEPRINT §14).
        if any(e.get("decision") for e in found["asked"].exits):
            ask(ctx, found["asked"])
        else:
            ctx.read["note"] = found["withheld"]
        return []
    ctx.read["basis"] = _causal_basis(found)
    if found["kind"] != "binary":
        v = found["variation"]
        d, mu = found["d"], found["fitted"]
        edges = shared_edges(d, d - mu)
        ctx.read["causal"] = {"r2": float(v.r2), "source": found["source"]}
        return [DistributionView(
            title=title(f"What is left of {tick(found['exposure'])}"),
            caption=caption(f"The covariates explain {fmt_value(float(v.r2))} of "
                            f"{tick(found['exposure'])}'s variance; the estimate rests on the "
                            f"rest."),
            emphasis=[found["exposure"]],
            column=found["exposure"],
            before=histogram(d, edges=edges),
            after=histogram(d - mu, edges=edges),
            before_label=f"{found['exposure']} as recorded",
            after_label=f"{found['exposure']} beyond the covariates",
        )]
    o, g, d = found["overlap"], found["g"], found["d"]
    edges = [float(x) for x in np.linspace(0.0, 1.0, 26)]
    bound = float(o.bound)
    marks = [Mark(value=bound, label=f"bound {num(bound)}"),
             Mark(value=1.0 - bound, label=f"bound {num(1 - bound)}")]
    if decision.trim is not None:
        marks += [Mark(value=float(decision.trim), label=f"trim {decision.trim:g}"),
                  Mark(value=1.0 - float(decision.trim), label=f"trim {1 - decision.trim:g}")]
    kept_g = g if decision.trim is None else g[(g >= decision.trim) & (g <= 1 - decision.trim)]
    ctx.read["causal"] = {"n_outside": int(o.n_outside), "max_weight": float(o.max_weight),
                          "n_trimmed": found.get("n_trimmed"), "n": found["n"],
                          "source": found["source"]}
    text = (f"{fmt_count(o.n_outside)} of {fmt_count(found['n'])} rows have a propensity beyond "
            f"the {fmt_value(bound)} bound; largest weight {fmt_value(o.max_weight)}.")
    views: list[Any] = [DistributionView(
        title=title(f"Who could have been exposed to {tick(found['exposure'])}"),
        caption=caption(text),
        emphasis=[found["exposure"]],
        column="propensity",
        before=histogram(g, edges=edges),
        after=histogram(kept_g, edges=edges),
        before_label="every row's propensity",
        after_label="the rows the estimator reads",
        marks=marks,
        story=[DistributionFrame(label=frame_label("The unexposed rows' propensity"),
                                 hist=histogram(g[d == 0], edges=edges)),
               DistributionFrame(label=frame_label("The exposed rows' propensity"),
                                 hist=histogram(g[d == 1], edges=edges))],
    )]
    # The rows a trim keeps are counted on the estimator's own propensity only: the card's is the
    # overlap's picture, not the forest's or the lasso's trim.
    if decision.trim is not None and found.get("n_trimmed") is not None and found["source"] != "card":
        n, gone = found["n"], int(found["n_trimmed"])
        start = RowStep(key="analyzed", label="Complete analyzed rows", n=n)
        views.append(RowFlowView(
            title=title("Rows inside the overlap"),
            caption=caption(f"Trimming at {decision.trim:g} keeps {fmt_count(n - gone)} rows: the "
                            f"effect becomes the overlap population's."),
            emphasis=["trimmed"],
            before=[start],
            after=[start, RowStep(key="trimmed", label=f"Propensity in [{decision.trim:g}, "
                                                        f"{1 - decision.trim:g}]",
                                  n=n - gone, dropped=gone, reason="outside the trimmed overlap")],
        ))
    return views


# ── set_time_varying ─────────────────────────────────────────────────────────


def lane_weights(sctx: Any, after: Any) -> dict[str, Any] | None:
    """The time-varying stage's diagnostics before any estimate (``stages.time_varying``): for a
    marginal structural model the stabilized weights (``_msm``, up to its outcome model: the same
    design, histories, weight models and truncation); for the g-formula the exposure model's fitted
    probabilities that its positivity reads (``_gformula``, up to its simulation). None without a
    lane; ``withheld`` with the stage's own reason when it would read none."""
    from turbotab.core.models import time_varying as tv
    from turbotab.core.stages.time_varying import (NotReady, _codes, _complete, _design_columns,
                                                   read_setting, weight_terms)
    from turbotab.core.time_varying import affected_confounders, current_lane, exposure_of

    lane = current_lane(after)
    exposure = exposure_of(after)
    if lane is None or exposure is None:
        return None
    structure = getattr(sctx.inputs.get("structure"), "data", sctx.inputs.get("structure"))
    msm = lane.method == "msm_iptw"
    try:
        frame, setting = read_setting(sctx, exposure, affected_confounders(after), structure)
        _complete(frame, setting, lane)
        codes = _codes(after)
        design, names_ = _design_columns(frame, [*lane.confounders, *lane.baseline], codes)
        if not msm:
            if setting.outcome_kind != "event":
                raise NotReady("The parametric g-formula here needs an event: a yes/no outcome at "
                               "each time point, with no rows after a unit's event.")
            for c in lane.confounders:  # a two-level code's one indicator stands for it
                if names_[c] != [c]:
                    design[c] = design[names_[c][0]] if names_[c] else 0.0
                    names_[c] = [c]
        d = pd.concat([frame[["__unit", "__time", "__y"]], design], axis=1)
        d["__a"] = pd.to_numeric(frame[setting.exposure]).to_numpy(float)
        d = tv.with_history(d, "__unit", "__time", ["__a"])
        terms = weight_terms(lane, names_)
        exposure_w = tv.ipw_weights(d, id="__unit", time="__time", indicator="__a",
                                    numerator=terms["numerator"] if msm else None,
                                    denominator=terms["denominator"],
                                    kind="first" if lane.pattern == "initiation" else "all",
                                    label=f"`{setting.exposure}`")
        censor_w = None
        if lane.censoring and msm:
            d["__c"] = pd.to_numeric(frame[lane.censoring]).to_numpy(float)
            at_risk = (d["__y"] == 0).to_numpy() if setting.outcome_kind == "event" else None
            censor_w = tv.censoring_weights(d, id="__unit", time="__time", indicator="__c",
                                            numerator=terms["censoring_numerator"],
                                            denominator=terms["censoring_denominator"],
                                            at_risk=at_risk,
                                            label=f"loss to follow-up (`{lane.censoring}`)")
    except (NotReady, tv.NotEstimable) as exc:
        return {"withheld": str(exc), "lane": lane}
    w = exposure_w.weights * (censor_w.weights if censor_w is not None else 1.0)
    level = tv.TRUNCATIONS[lane.truncation] if lane.truncation else None
    return {"lane": lane, "setting": setting, "exposure": exposure_w.weights,
            "censoring": None if censor_w is None else censor_w.weights, "weights": w,
            "used": tv.truncate(w, level) if lane.truncation else w, "level": level,
            "p_event": exposure_w.p_event, "modeled": exposure_w.modeled,
            "a": d["__a"].to_numpy()}


def time_varying_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.models import time_varying as tv

    if decision.method == "standard":
        ctx.read["note"] = ("Standard regression reads no weights and no simulation: the estimate "
                            "is the fitted model's.")
        return []
    after = after_state(decision, ctx)
    sctx = stage_context(ctx, after, ("working", "split", "target_info", "structure"))
    if sctx is None:
        return []
    found = lane_weights(sctx, after)
    if found is None:
        return []
    if "withheld" in found:
        ctx.read["note"] = found["withheld"]
        return []
    setting = found["setting"]
    ctx.read["basis"] = (f"The models of what you study fitted on all {setting.rows:,} analyzed rows of "
                         f"{setting.units:,} units; no outcome read.")
    # MODELING_SEQUENCE §4: the g-methods have no design-based estimator, so under the surveyed
    # population the lane blocks and records its estimates (``stages.time_varying.
    # population_block``); its diagnostics, drawn here, describe these rows as they are.
    from turbotab.core.stages.time_varying import population_block

    blocked = population_block(after)
    if blocked is not None:
        block_and_record(ctx, *blocked, decision)
    if decision.method == "gformula":
        p, modeled, a = found["p_event"], found["modeled"], found["a"]
        edges = [float(x) for x in np.linspace(0.0, 1.0, 26)]
        near = int(((p < tv.NEAR) | (p > 1 - tv.NEAR))[modeled].sum())
        ctx.read["time_varying"] = {"near": near}
        return [DistributionView(
            title=title(f"Positivity of {tick(setting.exposure)} over time"),
            caption=caption(f"{fmt_count(near)} rows have a fitted probability of being exposed within "
                            f"{tv.NEAR} of 0 or 1."),
            emphasis=[setting.exposure],
            column="probability of being exposed",
            before=histogram(p[modeled & (a == 0)], edges=edges),
            after=histogram(p[modeled & (a == 1)], edges=edges),
            before_label="unexposed rows", after_label="exposed rows",
        )]
    w, used = found["weights"], found["used"]
    summary, kept = tv.summarize(w), tv.summarize(used)
    ctx.read["time_varying"] = {"weights": summary, "used": kept}
    edges = shared_edges(w)
    marks = []
    if found["level"] is not None:
        lo, hi = float(np.quantile(w, found["level"])), float(np.quantile(w, 1 - found["level"]))
        marks = [Mark(value=lo, label=f"{found['level']:.0%} cut {num(lo)}"),
                 Mark(value=hi, label=f"{1 - found['level']:.0%} cut {num(hi)}")]
    # The method's steps: the exposure's weights, times loss to follow-up's (when declared), then
    # truncated (when declared).
    story = []
    first = found["exposure"] if found["censoring"] is not None else w
    if found["censoring"] is not None and found["level"] is not None:
        story = [DistributionFrame(label=frame_label("Times loss-to-follow-up weights"),
                                   hist=histogram(w, edges=edges))]
    text = (f"Stabilized weights: mean {fmt_value(summary['mean'])}, largest "
            f"{fmt_value(summary['max'])}"
            + (f"; truncated, largest {fmt_value(kept['max'])}." if found["level"]
               else "; kept as they are."))
    return [DistributionView(
        title=title("The weights the outcome model reads"),
        caption=caption(text),
        emphasis=[setting.exposure],
        column="stabilized weight",
        before=histogram(first, edges=edges),
        after=histogram(used, edges=edges),
        before_label=(f"{setting.exposure}'s weights" if found["censoring"] is not None
                      else "stabilized weights"),
        after_label=("truncated weights" if found["level"] else "stabilized weights, untruncated"),
        marks=marks,
        story=story,
    )]


# ── set_measurement_error ────────────────────────────────────────────────────

IMPUTED_LAMBDA = ("Blanks among the model's inputs: the calibration runs inside each imputed copy "
                  "when the stage runs, so λ is not drawn here.")
SINGLE_FILL = ("Blanks among the model's inputs: the missing-values answer decides the rows the "
               "calibration reads, so λ is drawn when the stage runs on them.")


def calibration_numbers(sctx: Any, after: Any) -> dict[str, Any] | None:
    """Each reported error-prone intake's recalls, mean and calibrated value, and its attenuation
    Γ_jj at the most common number of recalls, as the calibration stage computes them (MS5,
    ``stages.calibration``: the cohort's people with a recall day, the linear pipeline refit on
    them, each day through the same steps and centered on the person's value
    (``recall_matrix``), every error-prone column calibrated jointly with every other column of the
    model as a covariate (``methods.calibration.calibrate``)), without the outcome model or its
    bootstrap. None when the stage would not calibrate (its reason is the stage's).

    REPAIR-RC: the design is read as the stage reads it (``stages.calibration.analysis_design``):
    under the surveyed population only the survey domain is calibrated, survey-weighted, as the
    stage does. With a blank among the model's inputs and the blanks not dropped (complete cases),
    the stage calibrates inside each imputed copy, so no λ is drawn here (``{"withheld": why}``),
    as the scales preview does."""
    from sklearn.base import clone

    from turbotab.core.methods.calibration import CalibrationRefused, Recalls, calibrate
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.calibration import (NO_LINEAR, NONE_ERROR_PRONE, _nonlinear,
                                                  analysis_design, calibrated_family,
                                                  combine_rule, day_rows, error_prone,
                                                  recall_matrix)
    from turbotab.core.stages.data import open_store
    from turbotab.core.voice import listing

    spec_me = after.measurement_error
    design = sctx.inputs["design"]
    pipelines = design.objects["pipelines"]
    adj = after.energy_adjustment
    key = calibrated_family(after.models)
    if key is None or key not in pipelines:
        return {"withheld": NO_LINEAR}
    spec = DesignSpec.from_dict(design.objects["spec"])
    energy = adj.energy_column if adj is not None and adj.method != "none" else None
    rows = sctx.inputs["cohort"].frames["rows"]["row_id"].to_numpy(dtype=np.int64)
    with open_store(sctx) as store:
        frame = modeling_frame(store, list(spec.inputs), rows)
    X_all = frame[list(spec.inputs)]
    strategy = (spec.missing or {}).get("strategy")
    complete_case = strategy == "complete_case"
    if X_all.isna().to_numpy().any() and not complete_case:
        # Multiple imputation: the stage calibrates each copy. A single fill: the stage's own
        # missing-data handling for the inference table decides, so nothing is drawn here.
        return {"withheld": IMPUTED_LAMBDA if strategy == "multiple_imputation" else SINGLE_FILL}
    _, survey, in_domain, weights_all = analysis_design(sctx, after, frame.index)
    if survey is not None and survey.refusal:
        # The stage refuses with the design's own words and exits (block and record, §4).
        return {"refused": str(survey.refusal), "exits": list(survey.exits or [])}
    raw_columns = list(dict.fromkeys([*spec.inputs, *([energy] if energy else [])]))
    days = day_rows(sctx, raw_columns, rows)
    day_unit = days.pop("__unit").to_numpy(dtype=np.int64)
    position = {int(r): i for i, r in enumerate(frame.index.to_numpy(dtype=np.int64))}
    day_person = np.array([position.get(int(u), -1) for u in day_unit], dtype=np.int64)

    class _Fitted:  # model_matrix, error_prone and recall_matrix read a fitted pipeline's steps
        def __init__(self, inner: Any):
            self.steps = [*inner.steps, ("model", None)]
            self._inner = inner

        def __getitem__(self, key: Any) -> Any:
            return self._inner

    template = clone(pipelines[key])
    first = _Fitted(clone(template)[:-1].fit(X_all))  # the transform steps only: outcome-free
    columns = [str(c) for c in model_matrix(first, X_all).columns]
    items = error_prone(first, columns, spec.roles, energy)
    if not items:
        return {"withheld": NONE_ERROR_PRONE}
    shaped = [i["feature"] for i in items if _nonlinear(columns, i["feature"])]
    if shaped:  # the stage's refusal, shortened to the preview's one line
        return {"withheld": f"{listing(shaped)} {'has' if len(shaped) == 1 else 'have'} a declared "
                            f"spline or quintiles, and regression calibration here corrects a "
                            f"linear term."}
    working = dict(getattr(sctx.inputs["working"], "data", sctx.inputs["working"]))
    raw = list(dict.fromkeys(c for i in items for c in i["inputs"]))
    other = [c for c in raw if combine_rule(after, working, c) != "mean"]
    if other:  # as the stage says it
        return {"withheld": f"{listing(other)} {'was' if len(other) == 1 else 'were'} not combined "
                            f"by the mean of the recalls, so the error is a single day's: combine "
                            f"by the mean to calibrate."}
    features = [i["feature"] for i in items]
    wanted = set(spec_me.exposures)
    named = [i for i in items if i["source"] in wanted or i["feature"] in wanted]
    exposure_roles = {c for c, r in spec.roles.items() if r == "exposure"}
    reported = named or [i for i in items if i["source"] in exposure_roles] or list(items)
    # the people the stage calibrates: a recall day, and (complete cases) every input recorded
    on_day = day_person >= 0
    day_frame = days.loc[on_day].reset_index(drop=True)
    day_of = day_person[on_day]
    keep = (np.bincount(day_of, minlength=len(frame)) >= 1) & in_domain
    if X_all.isna().to_numpy().any() and complete_case:
        keep &= ~X_all.isna().any(axis=1).to_numpy()
    persons = np.flatnonzero(keep)
    remap = np.full(len(frame), -1)
    remap[persons] = np.arange(len(persons))
    day_mask = remap[day_of] >= 0
    day_frame = day_frame.loc[day_mask].reset_index(drop=True)
    day_of = remap[day_of[day_mask]]
    order = np.argsort(day_of, kind="stable")
    day_frame, day_of = day_frame.iloc[order].reset_index(drop=True), day_of[order]
    X = X_all.iloc[persons]
    fitted = _Fitted(clone(template)[:-1].fit(X))
    matrix = model_matrix(fitted, X)
    if [str(c) for c in matrix.columns] != columns:
        return None
    values, who = recall_matrix(fitted, X, day_frame, day_of, raw, features, matrix)
    rec = Recalls.of(values, who, len(X))
    have = rec.counts() >= 1
    if not have.all():
        rec = rec.subset(have)
    J = [columns.index(f) for f in features]
    others = [c for c in range(len(columns)) if c not in set(J)]
    Xm = matrix.to_numpy(dtype=float)[have]
    w = None if weights_all is None else weights_all[persons][have]
    try:
        cal = calibrate(rec, Xm[:, others] if others else None, w)
    except CalibrationRefused as refused:
        return {"exposures": [{**i, "refused": str(refused)} for i in reported]}
    modal = cal.modal_k
    gamma = cal.slope(modal)
    means = rec.means()
    out = []
    for item in reported:
        j = features.index(item["feature"])
        out.append({**item, "lambda": float(gamma[j, j]), "modal": modal,
                    "means": means[:, j], "calibrated": cal.calibrated[:, j],
                    "values": rec.values[:, j], "person": rec.person, "n": int(cal.n),
                    "n_repeat": int(cal.within.n_repeat)})
    return {"exposures": out}


def measurement_error_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    if decision.method == "none":
        ctx.read["note"] = ("The energy-adjusted study factors stay as each person's mean of recalls, "
                            "uncorrected.")
        return []
    from turbotab.core.consequences import Caution, CautionExit
    from turbotab.core.plan_previews import cannot_draw, inference_first, rows_not_ready
    from turbotab.core.stages.calibration import unread_refusal

    def refuse(reason: str, exits: Sequence[Mapping[str, Any]]) -> list[Any]:
        """The stage's refusal, as it records it: its exits that record a decision as the caution's
        controls, else the reason as the one line."""
        doors = [CautionExit(label=str(e["label"]), decision=dict(e["decision"]))
                 for e in exits if e.get("decision")]
        if doors and ctx.caution is None:
            ctx.caution = Caution(text=reason, exits=doors)
        else:
            ctx.read["note"] = reason
        return []

    after = after_state(decision, ctx)
    if getattr(after, "purpose", None) != "inference":
        return cannot_draw(decision, ctx, inference_first("Regression calibration is declared"))
    # What the calibration stage refuses before it reads a row (no repeated recalls; time points;
    # declared under another adjustment set), said as the stage records it (never an empty canvas).
    working = ctx.artifact("working")
    working = getattr(working, "data", working)
    if isinstance(working, Mapping):
        structure = ctx.artifact("structure")
        unread = unread_refusal(after, working, getattr(structure, "data", structure))
        if unread is not None:
            return refuse(unread["reason"], unread.get("exits") or [])
    sctx = stage_context(ctx, after, ("oriented", "findings", "structure", "working", "cohort",
                                      "design", "target_info"))
    if sctx is None:
        ctx.read["note"] = rows_not_ready(ctx)
        return []
    found = calibration_numbers(sctx, after) or {}
    if found.get("refused"):
        return refuse(found["refused"], found.get("exits") or [])
    if found.get("withheld"):
        ctx.read["note"] = found["withheld"]
        return []
    good = [e for e in found.get("exposures", []) if e.get("lambda") is not None]
    if not good:
        # The calibration itself refused on these recalls (no one with two, a covariance that is
        # not positive): its own words, as the stage records them.
        ctx.read["note"] = next((str(e["refused"]) for e in found.get("exposures", [])
                                 if e.get("refused")),
                                "The model's columns are being read again after the last answer; "
                                "λ is drawn once they are.")
        return []
    e = good[0]
    ctx.read["calibration"] = {x["feature"]: x["lambda"] for x in good}
    ctx.read["basis"] = (f"Calibrated on all {e['n']:,} eligible participants' recalls, without "
                         f"the outcome model or its bootstrap.")
    means, cal = e["means"], e["calibrated"]
    lam = e["lambda"]
    feature = e["feature"]
    text =(f"{tick(feature)}: λ = {fmt_value(round(lam, 2))} at {e['modal']} recalls, from "
            f"{fmt_count(e['n_repeat'])} people with repeats; values shrink toward the mean.")
    return [RelationshipView(
        title=title(f"{tick(feature)}, as measured and calibrated"),
        caption=caption(text),
        emphasis=[e["source"]],
        x_label=f"{feature}, each person's mean of recalls",
        y_label_before=f"{feature} as measured",
        y_label_after=f"{feature} calibrated, E[X | W̄, Z]",
        points_before=points(means, means),
        points_after=points(means, cal),
        r_before=None,
        r_after=None,
        story=[RelationshipFrame(label=frame_label("Each recall around its person's mean"),
                                 points=points(means[e["person"]], e["values"]), r=None,
                                 fit_line=FitLine(slope=1.0, intercept=0.0),
                                 y_label=f"{feature}, one day")],
    )]


# ── set_batch ────────────────────────────────────────────────────────────────


def batch_share(pc: np.ndarray, batch: Sequence[Any]) -> float:
    """The share of a component's variance between batches (η², the one-way ANOVA's)."""
    x = np.asarray(pc, dtype=float)
    labels = pd.Series([str(b) for b in batch])
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    means = pd.Series(x).groupby(labels).transform("mean").to_numpy()
    return float(((means - x.mean()) ** 2).sum() / total)


def _by_batch(pc: np.ndarray, batch: Sequence[Any],
              order: Sequence[str]) -> list[tuple[float, float]]:
    """Each row at its batch's position (with a fixed jitter) against its component score."""
    pos = {b: i for i, b in enumerate(order)}
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(pc))
    x = np.array([pos[str(b)] for b in batch], dtype=float) + jitter
    return points(x, pc)


def batch_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.methods.batch import principal_components

    if decision.method in ("none", "not_a_batch"):
        ctx.read["note"] = (f"{tick(decision.column)} is left as it is: no value changes."
                            if decision.method == "none" else
                            f"{tick(decision.column)} is read as an ordinary column.")
        return []
    ids = pool(ctx)
    if ids is None:
        return []
    after = after_state(decision, ctx)
    then = fit_design(ctx, after, ids, extra=[decision.column])
    now = fit_design(ctx, ctx.state, ids, extra=[decision.column])
    if then is None or now is None or decision.column not in now.frame.columns:
        return []
    step = dict(then.fitted.steps).get("batch")
    columns = list(step.fitted_columns_) if step is not None else [
        c for c, r in then.spec.roles.items() if r == "exposure" and c in now.matrix.columns]
    columns = [c for c in columns if c in now.frame.columns]
    if len(columns) < 2:
        return []
    batch = now.frame[decision.column].astype(str).to_numpy()
    # The features as the batch step receives them (any normalization and log before it), and as
    # it leaves them; batch as a covariate changes no value, so both are what the model sees.
    if step is not None:
        raw = _through(then, "batch", include=False)[columns].to_numpy(dtype=float)
        adjusted = _through(then, "batch")[columns].to_numpy(dtype=float)
    else:
        seen = [c for c in columns if c in then.matrix.columns]
        raw = adjusted = then.matrix[seen].to_numpy(dtype=float)
    keep =np.isfinite(raw).all(axis=1) & np.isfinite(adjusted).all(axis=1)
    if keep.sum() < 3:
        return []
    pc_before, _ = principal_components(raw[keep], 1)
    pc_after, _ = principal_components(adjusted[keep], 1)
    labels = batch[keep]
    order = sorted(set(labels.tolist()))
    share_before = batch_share(pc_before[:, 0], labels)
    share_after = batch_share(pc_after[:, 0], labels)
    ctx.read["batch"] = {"share_before": share_before, "share_after": share_after,
                         "columns": columns, "n": int(keep.sum())}
    if step is not None:
        text = (f"{tick(decision.column)} explains {share_before:.0%} of the first component's "
                f"variance; after reference ComBat, {share_after:.0%}.")
    else:
        text = (f"{tick(decision.column)} explains {share_before:.0%} of the first component's "
                f"variance; as a covariate the model compares rows within batches.")
    views: list[Any] = [RelationshipView(
        title=title(f"The first component, batch by batch"),
        caption=caption(text),
        emphasis=[decision.column],
        x_label=f"{decision.column} ({', '.join(order[:6])}{'…' if len(order) > 6 else ''})",
        y_label_before=f"PC1 of {len(columns)} features",
        y_label_after=(f"PC1 after reference ComBat" if step is not None
                       else f"PC1 of {len(columns)} features"),
        points_before=_by_batch(pc_before[:, 0], labels, order),
        points_after=_by_batch(pc_after[:, 0], labels, order),
        r_before=None,
        r_after=None,
    )]
    arrive, leave = lineage_change(now.lineage, then.lineage)
    views.append(LineageView(
        title=title(f"Where {tick(decision.column)} enters"),
        caption=caption(f"{fmt_count(len(columns))} features batch-adjusted inside each training "
                        f"fold, without the outcome." if step is not None else
                        f"{tick(decision.column)} enters the model as indicators, one per batch "
                        f"after the first."),
        emphasis=[decision.column, *arrive][:12],
        before=now.lineage,
        after=then.lineage,
    ))
    return views[:MAX_VIEWS]


def _through(design: Any, last: str, include: bool = True) -> pd.DataFrame:
    """The raw inputs through every fitted step up to ``last`` (and through it, ``include``)."""
    X = design.frame[design.spec.inputs]
    for name, step in design.fitted.steps:
        if name == last and not include:
            break
        X = step.transform(X)
        if name == last:
            break
    return X


register_consequence("set_scales", scales_views)
register_consequence("set_usual_intake", usual_intake_views)
register_consequence("set_causal", causal_views)
register_consequence("set_time_varying", time_varying_views)
register_consequence("set_measurement_error", measurement_error_views)
register_consequence("set_batch", specifies_the_model(batch_views))

__all__ = ["batch_share", "calibration_numbers", "causal_numbers", "lane_weights",
           "quantile_histogram", "scale_numbers"]
