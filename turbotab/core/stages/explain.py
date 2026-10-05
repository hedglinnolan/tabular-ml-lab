"""The ``explain`` stage: the fitted families described (``turbotab/core/models/explain.py``).

It reads the rows and fits the estimates read: under prediction the training rows and each family
refit on all of them (the fit's ``fitted``); under inference every analyzed row and the families
refit on them (``every_row``; BLUEPRINT §12 ruling 3), a family with no coefficient table refit on
them here, as the substitution stage does. Nothing it computes reads a held-out row.

The floor reads each family's cross-validated score against the no-predictor baseline's on the
same folds (``versus_baseline``), which is how far the fit stage itself says the model generalizes.
Under inference the stage is an estimate stage (``estimand.ESTIMATE_STAGES``): it is withheld until
the exposure, its effect and the adjustment set are answered, and its first display locks the plan.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.jobs import Cancelled
from turbotab.core.stages.data import open_store

EXPLAIN_READS: tuple[str, ...] = ("explain", "purpose", "task", "event", "outcome_order",
                                  "outcome_unit", "column_units", "estimand", "missing")
SUPPORTED_TASKS = ("regression", "binary")


def explain_stage(ctx: StageContext) -> Bundle:
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models import explain as E
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import LABELS, PRIMARY, higher_is_better
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.readings import settled_roles
    from turbotab.core.stages.modeling import (BASELINE_LABEL, _task, coded_outcome,
                                               pinned_to_full_fit, row_ids_of)

    state = ctx.state
    spec_answer = state.explain
    task = _task(ctx)
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    if task not in SUPPORTED_TASKS:
        artifact = E.ExplainArtifact(
            purpose=state.purpose, task=task, describes=E.DESCRIBES, rows=0, rows_of=0,
            rows_basis="", curve_method=spec_answer.curves, families=[], curves=[], methods="",
            notes=[f"Explanations are built for a numeric or a yes/no outcome; a {task} outcome's "
                   f"are not."])
        return Bundle(data=artifact.model_dump(mode="json"))
    inference = state.purpose == "inference"
    objects = fit.objects or {}
    every_ids = objects.get("every_row_ids")
    on_every_row = inference and every_ids is not None
    if on_every_row:
        ids = np.asarray(every_ids, dtype=np.int64)
        fitted = dict(objects.get("every_row") or {})
    else:
        ids = row_ids_of(design.frames["training"])
        fitted = dict(objects["fitted"])
    target = state.target
    grouped_by = objects.get("grouped_by")
    ctx.progress(0.01, "Reading the rows the models were fit on")
    extra = [target] + ([grouped_by] if grouped_by and grouped_by not in spec.inputs
                        and grouped_by != target else [])
    with open_store(ctx) as store:
        frame = modeling_frame(store, [*spec.inputs, *extra], ids, outcome=target)
    X = frame[spec.inputs]
    y = np.asarray(coded_outcome(task, frame[target].to_numpy(), state.event,
                                 order=state.outcome_order))
    units = frame[grouped_by].to_numpy() if grouped_by else None
    if units is not None and len(pd.unique(units)) == len(units):
        # One row per unit (the rows were combined per unit): no unit repeats, so resampling by it
        # is resampling rows, and saying "whole units" would be false (as the fit stage reads it).
        units, grouped_by = None, None
    pipelines = design.objects["pipelines"]
    by_key = {m["family"]: m for m in fit.data["models"]}
    primary = PRIMARY[task]

    def refit(model: Any, X_b: pd.DataFrame, y_b: Any, units_b: Any) -> Any:
        # Every copy of a resampled row (every row of a resampled unit) keeps to one side of the
        # refit's inner splits (the row-id index travels with the resample).
        inner = units_b if units_b is not None else X_b.index.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_pipeline(model, X_b, y_b, groups=inner)

    families = []
    for key in [k for k in (state.models or []) if k in by_key]:
        if ctx.cancelled():
            raise Cancelled()
        family = get_family(key)
        if not getattr(family, "predicts", True) or key not in pipelines:
            continue
        if key not in fitted:
            # Under inference a family with no coefficient table was fit on the training rows only;
            # it is described as refit on every analyzed row, as its estimates would be.
            ctx.progress(0.03, f"{family.label}: refitting on every analyzed row")
            fitted[key] = refit(clone(pipelines[key]), X, y, units)
        m = by_key[key]
        cv = (m.get("cv") or {}).get(primary) or {}
        families.append(E.FamilyFit(
            key=key, label=family.label, fitted=fitted[key],
            unfitted=pinned_to_full_fit(clone(pipelines[key]), fitted[key]),
            versus=m.get("versus_baseline"), score=cv.get("estimate"),
            baseline=(m.get("baseline") or {}).get("value")))
    roles = settled_roles(state)
    exposures = [c for c in spec.predictors if roles.get(c) == "exposure"]
    declared: list[str] = []
    if inference:
        from turbotab.core.estimand import current_estimand, exposures_of

        current = current_estimand(state)
        declared = exposures_of(state, current) if current is not None else []
    setting = E.Setting(
        task=task, purpose=state.purpose, target=target, event=state.event, X=X, y=y, state=state,
        units=units, unit_name=grouped_by, exposures=list(spec_answer.exposures),
        declared=declared, candidates=exposures, curve_method=spec_answer.curves,
        reseeds=int(spec_answer.reseeds), seed=int(getattr(state.split, "seed", 0) or 0),
        metric_label=LABELS[primary], baseline_label=BASELINE_LABEL[task],
        higher_is_better=higher_is_better(primary),
        multiple_imputation=inference and spec.multiple_imputation())

    def progress(fraction: float, message: str) -> None:
        if ctx.cancelled():
            raise Cancelled()
        ctx.progress(fraction, message)

    artifact = E.explain(families, setting, refit, progress=progress)
    return Bundle(data=artifact.model_dump(mode="json"))


__all__ = ["EXPLAIN_READS", "explain_stage"]
