"""The ``secondary`` stage: the declared "further adjusted for" model beside the primary (WP17).

MODELING_SEQUENCE §1 step 3: a covariate whose timing is unknown (a cross-sectional BMI) is "a
declared with-and-without pair", and §4: a mediator left out of a total-effect set is offered as
"further adjusted for BMI", a labeled secondary. VanderWeele (2019) on BMI measured at the same time
as the exposure: "We cannot adequately distinguish in this setting between confounding and
mediation." The adjustment answers (``turbotab/core/estimand.py``) name the columns; this stage fits
the primary model and the same model further adjusted for them, on the same rows, and reports the
exposure's estimate in each, so a reader sees how far the choice moves the answer.

**The rows.** Under inference every analyzed row (BLUEPRINT §12 ruling 3). Under complete cases the
primary's rows may still be missing a secondary column; both models are then refit on the rows
where every secondary column is recorded, and the artifact says how many that is, so the two
estimates differ by the adjustment alone. Under multiple imputation each model pools its own
imputations (the imputation model holds the secondary columns too).

**The model** is the design's own pipeline, built again from the design spec with the secondary
columns added (``models/pipeline.py``): every step, energy adjustment and imputation included, is the
primary's. Only families with a coefficient table take part.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.models.artifacts import Coefficient, Inference


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class SecondaryFit(_Model):
    label: str  # "Primary" | "Further adjusted for `bmi`"
    adjusted_for: list[str]  # the model's columns besides the exposure
    n_rows: int
    coefficients: list[Coefficient] | None  # the exposure's rows only
    inference: Inference | None = None
    concerns: list[str] = []


class SecondaryFamily(_Model):
    family: str
    label: str
    fits: list[SecondaryFit]


class SecondaryArtifact(_Model):
    purpose: Literal["inference"]
    exposure: str
    further: list[str]  # the columns the secondary model adds
    rows: str  # which rows both models were fit on
    families: list[SecondaryFamily]
    methods: str  # the methods sentence


def secondary_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.estimand import (current_estimand, exposure_key, exposures_of,
                                        primary_features, secondary_columns)
    from turbotab.core.models import get_family
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.inference import Outcome, cluster_columns
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, design_spec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_missing_for_table, _survey, _task, coded_outcome,
                                               outcome_levels, read_assignment)
    from turbotab.core.stages.sensitivity import fit_on_rows
    from turbotab.core.voice import listing

    state = ctx.state
    spec_estimand = current_estimand(state)
    further = secondary_columns(state)
    if state.purpose != "inference" or spec_estimand is None or not further:
        # Nothing to fit is not a failure: no covariate's answers put one beside the primary
        # (unknown timing, or "further adjusted for"), or the purpose reads no effect at all.
        why = ("Under prediction no coefficient is read as an effect, so no secondary adjustment "
               "model is declared." if state.purpose != "inference" else
               "No secondary model is declared: no covariate's answers put one beside the primary.")
        return Bundle(data=SecondaryArtifact(
            purpose="inference", exposure=str(getattr(spec_estimand, "exposure", "") or ""),
            further=[], rows="", families=[], methods=why).model_dump(mode="json"))
    # One exposure, or each exposure of a family (``*``): their rows are the estimates reported.
    exposure = exposure_key(spec_estimand)
    exposures = exposures_of(state, spec_estimand)
    task = _task(ctx)
    target = state.target
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    families = [get_family(k) for k in (state.models or []) if k in pipelines]
    families = [f for f in families if reports_coefficients(f) and hasattr(f, "inference")]
    if not families:
        raise ValueError("None of the chosen model families has a coefficient table, so there is "
                         "no estimate to set beside the primary's; choose the linear model.")
    assignment = read_assignment(ctx.inputs["split"])
    rows = assignment.index.to_numpy()  # every analyzed row (ruling 3)
    extra = [c for c in further if c not in spec.inputs]
    with open_store(ctx) as store:
        unit_columns = cluster_columns(state, store.columns)
        info = {c.name: c for c in store.info().columns}
        columns = list(dict.fromkeys([*spec.inputs, *extra, target, *unit_columns]))
        frame = modeling_frame(store, columns, rows, outcome=target)
    strategy = state.missing.strategy if state.missing is not None else None
    note = "every analyzed row"
    if strategy != "multiple_imputation" and extra:
        recorded = frame[extra].notna().all(axis=1)
        if not recorded.all():
            frame = frame.loc[recorded]
            note = (f"the {int(len(frame)):,} analyzed rows with {listing(extra)} recorded, so the "
                    f"two differ by the adjustment alone")
    spec2 = design_spec(state, frame[[*spec.inputs, *extra]],
                        [*spec.predictors, *[c for c in extra if c not in spec.predictors]],
                        column_info=info, energy_factors=spec.energy_factors)

    outcome = Outcome(name=target, labels=outcome_levels(task, frame[target].to_numpy(), state.event))
    levels = None
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, levels = ordinal_outcome(frame[target].to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    if task == "time_to_event":
        from turbotab.core.models.survival import follow_up_columns, time_to_event_outcome

        with open_store(ctx) as store:
            timing = modeling_frame(store, follow_up_columns(state), frame.index.to_numpy())
        coded = time_to_event_outcome(state, timing, coded)
    y = np.asarray(coded)
    from turbotab.core.models.inference import resolve_clusters

    every_unit = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
    survey, _ = _survey(ctx, every_unit)
    n_cols = len(spec2.inputs) + 1
    keys = [f.key for f in families]
    analyses = [("Primary", spec, {f.key: pipelines[f.key] for f in families}),
                (f"Further adjusted for {listing(further)}", spec2,
                 {f.key: build_pipeline(spec2, f, task, state.purpose, len(frame), n_cols)
                  for f in families})]
    out: list[dict[str, Any]] = []
    total = max(1, len(families) * len(analyses))
    done = 0
    missing_by = {label: _missing_for_table(ctx, s, frame[list(s.inputs)], y, task, keys,
                                            loss={"n_dropped": None})
                  for label, s, _ in analyses}
    for family in families:
        fits = []
        for label, s, built in analyses:
            done += 1
            ctx.progress(0.1 + 0.85 * done / total, f"{family.label}: {label}")
            adjusted = [c for c in s.predictors if c not in exposures]
            try:
                _, (coef, inference), concerns = fit_on_rows(
                    state, family, built[family.key], frame, s.inputs, y, task, unit_columns,
                    outcome=outcome, survey=survey, levels=levels, missing=missing_by[label],
                    spec=s)
            except Exception as exc:  # noqa: BLE001 - a model that cannot be fit says why
                fits.append({"label": label, "adjusted_for": adjusted, "n_rows": int(len(frame)),
                             "coefficients": None, "concerns": [f"It could not be fit: {exc}"]})
                continue
            features = {f for e in exposures
                        for f in primary_features([r["feature"] for r in coef or []], e)}
            fits.append({"label": label, "adjusted_for": adjusted, "n_rows": int(len(frame)),
                         "coefficients": [r for r in coef or [] if r["feature"] in features],
                         "inference": inference, "concerns": concerns})
        out.append({"family": family.key, "label": family.label, "fits": fits})
    methods = (f"Beside the primary model, the same model further adjusted for {listing(further)} "
               f"was declared before the estimates were shown and fit on {note}; the estimate of "
               f"{listing(exposures, limit=3)} is reported from each.")
    artifact = SecondaryArtifact(purpose="inference", exposure=exposure, further=further,
                                 rows=note, families=out, methods=methods)
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


SECONDARY_READS = ("adjustment", "estimand", "clusters", "purpose", "models", "task", "event",
                   "target", "roles", "roles_unconfirmed", "role_confirmations",
                   "reading_confirmations", "shape_confirmations", "missing", "survey",
                   "outcome_order", "follow_up", "categorical", "energy_adjustment", "grain",
                   "exposure_forms", "lens", "findings")

__all__ = ["SECONDARY_READS", "SecondaryArtifact", "SecondaryFamily", "SecondaryFit",
           "secondary_stage"]
