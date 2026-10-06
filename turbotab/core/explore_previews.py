"""Consequence previews for Explore's levers and the prediction evaluation's answers (MODELING_SEQUENCE
§1 rows 1, 2, 8 and 11; §3, "Selection | lineage: the columns that leave, per fold"): what each
answer does to the model the user is building, drawn on their own rows in the closed vocabulary
(BLUEPRINT §11 rule 2). Wave 2b's EXPLORE kinds under wave 2c's "every choice previewed".

| Kind                 | Views (primary first)                                                   |
|----------------------|-------------------------------------------------------------------------|
| ``set_levers``       | lineage: the in-fold steps the levers add after construction (splines   |
|                      | by the rule or by inner cross-validation, the variance filter), fitted  |
|                      | as the design's pipeline fits them on the preview's training rows; an   |
|                      | imbalance correction: row flow of the rows each class trains on         |
| ``set_selection``    | lineage: the terms the in-fold selection keeps on the preview's training|
|                      | rows (every training fold repeats it); under inference the declared     |
|                      | model unchanged (the selection is a labeled sensitivity analysis)       |
| ``set_intended_use`` | distribution of the fitted model's risks with the thresholds marked     |
|                      | (decision support, a yes/no outcome, once fitted) · rows per level of   |
|                      | the first subgroup                                                      |
| ``set_updating``     | relationship: each training row's prediction before and after uniform   |
|                      | shrinkage by the fit's calibration slope, the intercept re-estimated    |
|                      | (``models.decision_curve.shrinkage``, as the evaluation stage calls it) |

Each is prediction's choice (refused, or a sensitivity analysis, under inference), so a preview
reads the training rows' outcome only where the stage does (an inner choice of form, a selection,
the intercept after shrinkage), never a held-out row (``ctx.training_row_ids``).

Importing this module registers the builders.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    DistributionView, LineageView, Mark, PreviewContext, RelationshipView, RowFlowView, RowStep,
    after_state, fmt_count, register_consequence,
)
from turbotab.core.plan_previews import (
    Design, caption, drawn, fit_design, histogram, lineage_change, moved, num, points, pool,
    rows_word, tick, title,
)


# ── the outcome and the in-fold steps, on the preview's rows ──────────────────


def _outcome(ctx: PreviewContext, state: Any, index: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """The outcome coded as the stages code it (``coded_outcome``) at ``index``, and which rows
    record it; None without a target."""
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.modeling import coded_outcome

    target = getattr(state, "target", None)
    if not target or target not in ctx.datastore.columns:
        return None
    frame = modeling_frame(ctx.datastore, [target], np.asarray(index), outcome=target)
    values = frame[target].reindex(index)
    keep = values.notna().to_numpy()
    y = np.asarray(coded_outcome(state.task, values[keep].to_numpy(), state.event))
    return y, keep


def explored(design: Design, task: str, y: np.ndarray | None,
             keep: np.ndarray | None) -> tuple[Any, list[tuple[str, Any]]]:
    """``(lineage, fitted steps)``: the design's construction followed by the steps Explore's
    answers add (``pipeline.explore_steps``, as ``build_pipeline`` adds them), fitted on the rows
    that record the outcome."""
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import explore_steps

    steps = explore_steps(design.spec, task)
    if not steps:
        return design.lineage, []
    X = design.matrix if keep is None else design.matrix.loc[keep]
    fitted: list[tuple[str, Any]] = []
    for name, step in steps:
        step.fit(X, y)
        X = step.transform(X)
        fitted.append((name, step))
    lineage = trace([*design.fitted.steps, *fitted], design.spec.inputs, design.spec.roles,
                    missing_counts(design.frame[design.spec.inputs]))
    return lineage, fitted


def _pair(decision: Any, ctx: PreviewContext) -> tuple[Any, Design | None, Design, Any] | None:
    """``(after state, current design, answered design, rows)`` on the training rows; None before
    the split or while the state names no predictor."""
    ids = pool(ctx)
    if ids is None:
        return None
    after = after_state(decision, ctx)
    then = fit_design(ctx, after, ids)
    if then is None:
        return None
    return after, fit_design(ctx, ctx.state, ids), then, ids


# ── set_levers ───────────────────────────────────────────────────────────────


def levers_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    found = _pair(decision, ctx)
    if found is None:
        return []
    after, now, then, _ = found
    if getattr(after, "purpose", None) == "inference":
        return []
    task = str(after.task or "regression")
    outcome = _outcome(ctx, after, then.matrix.index)
    y, keep = outcome if outcome is not None else (None, None)
    views: list[Any] = []
    lineage, steps = explored(then, task, y, keep)
    before = explored(now, task, y, keep)[0] if now is not None else None
    if steps:
        arrive, leave = lineage_change(before, lineage)
        parts = []
        forms = dict(steps).get("lever_forms")
        if forms is not None:
            from turbotab.core.methods.levers import InnerCVForms

            bent = list(forms.knots_)
            how = ("as inner cross-validation chose" if isinstance(forms, InnerCVForms) else
                   f"with {forms.k_} knots by Harrell's rule")
            parts.append(f"{fmt_count(len(bent))} continuous "
                         f"{'predictor bends' if len(bent) == 1 else 'predictors bend'} {how}")
        filt = dict(steps).get("lever_filter")
        if filt is not None:
            parts.append(f"{fmt_count(len(filt.dropped_))} "
                         f"{'leaves' if len(filt.dropped_) == 1 else 'leave'} by the variance "
                         f"filter")
        ctx.read["levers"] = {
            "knots": ({c: [float(v) for v in kn] for c, kn in forms.knots_.items()}
                      if forms is not None else {}),
            "k": int(forms.k_) if forms is not None else None,
            "dropped": list(filt.dropped_) if filt is not None else []}
        text = ("; ".join(parts) + ".") if parts else moved(arrive, leave)
        views.append(LineageView(
            title=title("Explore's levers, in each training fold"),
            caption=caption(text[0].upper() + text[1:]),
            emphasis=[*arrive, *leave][:12],
            before=before, after=lineage))
    method = getattr(decision, "imbalance", "none") or "none"
    if method != "none" and task == "binary" and y is not None and len(y):
        from turbotab.core.methods.levers import resampled_rows

        levels, counts = np.unique(y, return_counts=True)
        n = int(len(y))
        start = RowStep(key="training", label=f"{rows_word(ctx).capitalize()} rows", n=n)
        if method == "weights":
            rows = n
            reason = "each class weighted equally, every row kept"
        else:
            rows = int(len(resampled_rows(np.asarray(y), method, np.random.default_rng(0))))
            reason = ("the larger class drawn down to the smaller" if method == "undersample"
                      else "the smaller class drawn up to the larger")
        ctx.read["imbalance"] = {"method": method, "rows": rows,
                                 "counts": {str(lv): int(c) for lv, c in zip(levels, counts)}}
        views.append(RowFlowView(
            title=title("The rows each fold's model trains on"),
            caption=caption(f"{fmt_count(rows)} of {fmt_count(n)} {drawn(ctx)}{rows_word(ctx)} rows: "
                            f"{reason}; the risks are then recalibrated."),
            emphasis=["imbalance"],
            before=[start],
            after=[start, RowStep(key="imbalance", label="Rows the model trains on", n=rows,
                                  dropped=max(0, n - rows), reason=reason)]))
    return views


# ── set_selection ────────────────────────────────────────────────────────────


def selection_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.models.variable_selection import LABELS

    found = _pair(decision, ctx)
    if found is None:
        return []
    after, now, then, _ = found
    label = LABELS.get(decision.method, "Selection")
    if getattr(after, "purpose", None) == "inference" or decision.sensitivity:
        # The declared model is the reported one; the selection runs beside it as a labeled
        # sensitivity analysis whose tests are estimates, withheld until the plan is answered.
        return [LineageView(
            title=title("The declared model stays as declared"),
            caption=caption(f"Every declared term stays; a labeled sensitivity analysis runs "
                            f"beside it: {label}."),
            emphasis=[], before=then.lineage, after=then.lineage)]
    task = str(after.task or "regression")
    outcome = _outcome(ctx, after, then.matrix.index)
    if outcome is None:
        return []
    y, keep = outcome
    lineage, steps = explored(then, task, y, keep)
    before = explored(now, task, y, keep)[0] if now is not None else None
    select = dict(steps).get("select")
    if select is None:
        arrive, leave = lineage_change(before, lineage)
        return [LineageView(
            title=title("No selection: every term enters"),
            caption=caption(moved(arrive, leave)), emphasis=[*arrive, *leave][:12],
            before=before, after=lineage)] if before is not None else []
    kept = [str(c) for c in getattr(select, "kept_", [])]
    terms = [str(c) for c in getattr(select, "feature_names_in_", [])]
    leave = [c for c in terms if c not in set(kept)]
    ctx.read["selection"] = {"method": decision.method, "kept": kept, "dropped": leave}
    return [LineageView(
        title=title(label),
        caption=caption(f"On these {fmt_count(int(keep.sum()))} {drawn(ctx)}{rows_word(ctx)} rows "
                        f"it keeps {fmt_count(len(kept))} of {fmt_count(len(terms))} columns; "
                        f"every training fold repeats it."),
        emphasis=leave[:12], before=before, after=lineage)]


# ── set_intended_use ─────────────────────────────────────────────────────────


def _fitted(ctx: PreviewContext, family: str | None) -> tuple[Any, dict[str, Any]] | None:
    """The fit's final pipeline for ``family`` and its model entry; None while the fit is not
    fresh."""
    fit = ctx.artifact("fit")
    if fit is None or family is None:
        return None
    pipeline = ((getattr(fit, "objects", None) or {}).get("fitted") or {}).get(family)
    entry = next((m for m in (fit.data or {}).get("models") or [] if m.get("family") == family),
                 None)
    if pipeline is None or entry is None:
        return None
    return pipeline, entry


def _inputs(ctx: PreviewContext, pipeline: Any, ids: Any) -> pd.DataFrame | None:
    from turbotab.core.models.pipeline import modeling_frame

    design = ctx.artifact("design")
    if design is None:
        return None
    inputs = list((design.objects or {}).get("spec", {}).get("inputs") or [])
    if not inputs or any(c not in ctx.datastore.columns for c in inputs):
        return None
    return modeling_frame(ctx.datastore, inputs, np.asarray(ids))


def _level_view(values: Any, column: str, text: str) -> DistributionView | None:
    raw = pd.Series(values).dropna().astype(str)
    levels = sorted(raw.unique().tolist())
    if not levels:
        return None
    codes = pd.Categorical(raw, categories=levels).codes.astype(float)
    hist = histogram(codes, edges=[i - 0.5 for i in range(len(levels) + 1)])
    return DistributionView(
        title=title(f"Rows in each level of {tick(column)}"), caption=caption(text),
        emphasis=[column], column=column, before=hist, after=hist,
        before_label="rows per level", after_label="rows per level",
        marks=[Mark(value=float(i), label=str(level)) for i, level in enumerate(levels)])


def intended_use_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.models.decision_curve import DEFAULT_RANGE
    from turbotab.core.stages.evaluation import _reported

    after = after_state(decision, ctx)
    ids = pool(ctx)
    if ids is None or getattr(after, "purpose", None) == "inference":
        return []
    views: list[Any] = []
    fit = ctx.artifact("fit")
    task = str(after.task or "regression")
    if decision.use == "decision_support" and task == "binary" and fit is not None:
        found = _fitted(ctx, _reported(fit.data or {}))
        X = _inputs(ctx, found[0], ids) if found is not None else None
        if found is not None and X is not None:
            pipeline, entry = found
            model = pipeline[-1]
            classes = list(getattr(model, "classes_", []))
            from turbotab.core.stages.modeling import coded_outcome

            event = coded_outcome(task, np.asarray([after.event]), after.event)[0] \
                if after.event is not None else (sorted(classes, key=str)[-1] if classes else 1)
            at = classes.index(event) if event in classes else len(classes) - 1
            risk = np.asarray(pipeline.predict_proba(X))[:, at]
            low = decision.threshold_low if decision.threshold_low is not None else DEFAULT_RANGE[0]
            high = (decision.threshold_high if decision.threshold_high is not None
                    else DEFAULT_RANGE[1])
            marks = [Mark(value=float(low), label="lowest threshold"),
                     Mark(value=float(high), label="highest threshold")]
            if decision.threshold is not None:
                marks.append(Mark(value=float(decision.threshold), label="declared threshold"))
            hist = histogram(risk, edges=[i / 40 for i in range(41)])
            ctx.read["risks"] = {"family": entry.get("family"), "low": float(low),
                                 "high": float(high), "n": int(len(risk))}
            views.append(DistributionView(
                title=title("Where the decision thresholds fall"),
                caption=caption(f"{entry.get('label') or entry.get('family')}'s risks for "
                                f"{fmt_count(len(risk))} {drawn(ctx)}{rows_word(ctx)} rows; the "
                                f"curve runs from {num(low)} to {num(high)}."),
                emphasis=[str(after.target)], column=str(after.target), before=hist, after=hist,
                before_label="risk", after_label="risk", marks=marks))
    for column in decision.subgroups[:1]:
        if column not in ctx.datastore.columns:
            continue
        from turbotab.core.models.decision_curve import grouping_of, value_facts
        from turbotab.core.readings import whole_facts

        values = ctx.datastore.materialize([column], ids)[column]
        # The groups the evaluation stage scores: the column's levels as codes, its thirds as an
        # amount, as the readings ledger holds it (BLUEPRINT §14.3).
        facts = whole_facts([column], None, ctx.datastore)
        how, _ = grouping_of(after, column, facts[column] if column in facts
                             else value_facts(values))
        if how == "thirds":
            x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x):
                cuts = np.quantile(x, [1 / 3, 2 / 3])
                hist = histogram(x)
                views.append(DistributionView(
                    title=title(f"Rows in each third of {tick(column)}"),
                    caption=caption(f"Performance is reported in each third of {tick(column)}, "
                                    f"cut at {num(cuts[0])} and {num(cuts[1])}, with intervals."),
                    emphasis=[column], column=column, before=hist, after=hist,
                    before_label="rows", after_label="rows",
                    marks=[Mark(value=float(c), label=f"cut {i + 1}") for i, c in enumerate(cuts)]))
            continue
        if how is None:
            ctx.read["note"] = (f"Whether {tick(column)}'s numbers are codes or amounts is asked "
                                f"before its groups are drawn.")
            continue
        n_levels = int(values.dropna().astype(str).nunique())
        found_view = _level_view(values, column, (
            f"Performance is reported in each of the {fmt_count(n_levels)} levels of "
            f"{tick(column)}, with intervals."))
        if found_view is not None:
            views.append(found_view)
    if not views:
        ctx.read["note"] = ("The decision curve is drawn on a yes/no outcome's out-of-fold risks "
                            "once the models are fitted.")
    return views


# ── set_updating ─────────────────────────────────────────────────────────────


def updating_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.models.decision_curve import deployed_coefficients, shrinkage, slope_of
    from turbotab.core.models.linear import model_matrix

    after = after_state(decision, ctx)
    ids = pool(ctx)
    if ids is None or getattr(after, "purpose", None) == "inference":
        return []
    found = _fitted(ctx, "linear")
    if found is None:
        ctx.read["note"] = ("Shrinkage multiplies the regression's coefficients by its calibration "
                            "slope once the models are fitted.")
        return []
    pipeline, entry = found
    slope, how = slope_of(entry)
    X = _inputs(ctx, pipeline, ids)
    outcome = _outcome(ctx, after, X.index) if X is not None else None
    if slope is None or X is None or outcome is None:
        return []
    y, keep = outcome
    task = str(after.task or "regression")
    X = X.loc[keep]
    matrix = model_matrix(pipeline, X)
    deployed = deployed_coefficients(pipeline[-1])  # the recalibrated model's, as the stage's
    if deployed is None:
        return []
    intercept, coef = deployed
    lp = matrix.to_numpy(dtype=float) @ coef + intercept
    yy = (np.asarray(y, dtype=float) if task == "regression" else
          (np.asarray(y) == sorted(pd.unique(np.asarray(y)).tolist(), key=str)[-1]).astype(float))
    try:
        done = shrinkage(task, matrix, yy, coef, slope)
    except ValueError as exc:  # the stage says so in the record; the preview says it here
        ctx.read["note"] = f"{exc} The model would not be updated."
        return []
    shrunk = float(done["intercept"]) + slope * (matrix.to_numpy(dtype=float) @ coef)
    if task == "binary":
        lp, shrunk = 1 / (1 + np.exp(-lp)), 1 / (1 + np.exp(-shrunk))
    applied = decision.method == "shrinkage"
    after_values = shrunk if applied else lp
    ctx.read["shrinkage"] = {"factor": float(slope), "intercept": float(done["intercept"]),
                             "applied": applied}
    word = "risk" if task == "binary" else "prediction"
    text = (f"Coefficients × {slope:.3f}, {how.split(' (')[0]}; the intercept re-estimated, so "
            f"each {word} moves toward the mean." if applied else
            f"No updating: each {word} stays as fitted.")
    return [RelationshipView(
        title=title(f"Each {word}, before and after updating"),
        caption=caption(text), emphasis=["linear"],
        x_label=f"{word} as fitted", y_label_before=f"{word} as fitted",
        y_label_after=f"{word} {'after shrinkage' if applied else 'as fitted'}",
        points_before=points(lp, lp), points_after=points(lp, after_values),
        r_before=None, r_after=None)]


register_consequence("set_levers", levers_views)
register_consequence("set_selection", selection_views)
register_consequence("set_intended_use", intended_use_views)
register_consequence("set_updating", updating_views)

__all__ = ["explored", "intended_use_views", "levers_views", "selection_views", "updating_views"]
