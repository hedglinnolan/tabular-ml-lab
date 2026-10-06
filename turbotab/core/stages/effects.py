"""The ``effects`` stage: the exposure's effect as the declared estimand reports it (the package
ESTIMAND; MODELING_SEQUENCE §0 rulings 9 and 10, §1 rows 2, 11 and 12).

Under inference, once the exposure, its effect and the adjustment set are answered (WP17), this
stage builds what the results show of the exposure:

* **The declared model sequence (Table 2).** MODELING_SEQUENCE §1 row 11: "a crude model, a
  declared adjustment sequence (Model 1: age, sex and energy; Model 2: plus confounders; optional
  Model 3: plus possible mediators, labeled) and the primary model, shown for the exposure only".
  The crude model is always shown (STROBE item 16a: "Give unadjusted estimates and, if applicable,
  confounder-adjusted estimates and their precision"), for every family with a coefficient table,
  the mixed model and GEE included: each is refit on the columns of each model. Model 2 is the
  primary: the full adjustment set the answers derive, the fit stage's own table. Model 1 holds the
  columns the user declared (``set_model_sequence``; never read from a name). Model 3 adds the
  columns the answers set beside the primary (unknown timing, or "further adjusted for"), labeled
  as not a total effect. The exposure's definition is the same in every model: each model is the
  primary pipeline's own model matrix (its energy model, forms and fill fitted once on these rows),
  restricted to the exposure's columns and those of the model's covariates; Model 1's "energy"
  brings every term the energy model made of total energy and the other sources. Only the
  exposure's rows are effects; every other row is listed apart, under "adjustment terms, not
  effect estimates" (Westreich & Greenland 2013).
* **The marginal risk difference and ratio**, when the estimand declares one: standardization over
  the analyzed rows from the logistic model (``models/effects.py``), with percentile intervals from
  a bootstrap that refits the whole chain, by unit when rows repeat. Below the unit floor no
  interval is reported, as for every other interval (``inference.floor_refusal``); a resample
  whose outcome happens to be separated is kept at the likelihood's limit and counted.
* **Diagnostics of the primary model**, reported and never acted on silently: proportional hazards
  by Schoenfeld residuals (a Cox model), leverage and Cook's distance (least squares, logistic). A
  failed check of the exposure carries its exits, each a ``respond_diagnostic`` decision; the
  recorded response is then shown beside the estimate (the hazard ratio before and after the median
  event time; the primary refit without the influential rows), or the estimate stays labeled.
* **Sensitivity to unmeasured confounding** for every reported estimate: the robustness value with
  each adjusted covariate as a named benchmark, ranked first for a least-squares estimate, and the
  E-value for the estimate and the confidence limit nearer the null (``models/effects.py``). A
  categorical exposure's each level, against the reference, is one estimate; a curve has none, so
  its straight-line estimate beside it (the same model, the spline's nonlinear terms left out) is
  the one analyzed, and said so.

**The rows** are every analyzed row (BLUEPRINT §12 ruling 3), so Model 2 is the fit's primary
estimate, number for number. Under complete cases, when Model 3 adds columns that are sometimes
missing, Model 3 alone is fit on the rows where they are recorded, with the primary's adjustment
refit on those same rows beside it, so the two differ by the added columns alone; it says how many.
Under multiple imputation each model is fit in each completed copy and pooled by Rubin's rules, the
crude model, Model 1 and Model 2 in the fit's own copies and Model 3 in copies imputed with its
added columns; the diagnostics and the robustness value read the first copy, and say so.

**The surveyed population** (MODELING_SEQUENCE §0 ruling 6, "binds every family and every
display"): every model of a family with a design-based estimator (least squares, logistic, Cox,
proportional odds) is design-based, and the E-value reads the design-based estimate with the
population's event share or standard deviation; a family without one is blocked and recorded with
the fit's exits.
"""
from __future__ import annotations

import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core import contracts
from turbotab.core.graph import Bundle, StageContext
from turbotab.core.models.artifacts import AdjustmentTerm, Coefficient, Inference, InferenceExit


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


SequenceKey = Literal["crude", "model_1", "model_2", "model_3"]


class SequenceFit(_Model):
    key: SequenceKey
    label: str  # "Unadjusted" | "Model 1" | "Model 2 (primary)" | "Model 3"
    adjusted_for: list[str]  # the columns besides the exposure
    note: str | None = None  # what the model is (Model 3: "possible mediators: not a total effect")
    n_rows: int
    effects: list[Coefficient] | None  # the exposure's rows only
    inference: Inference | None = None
    concerns: list[str] = []
    # Model 3 fit on fewer rows (its added columns sometimes missing, under complete cases): the
    # primary's adjustment refit on Model 3's own rows, the exposure's rows only, so the two differ
    # by the added columns alone.
    comparison: list[Coefficient] | None = None
    comparison_label: str | None = None


class AppendixModel(_Model):
    key: SequenceKey
    label: str
    terms: list[AdjustmentTerm]


class MarginalContrast(_Model):
    setting: str
    risk_low: float
    risk_high: float
    rd: float
    rr: float | None
    rd_low: float | None = None
    rd_high: float | None = None
    rr_low: float | None = None
    rr_high: float | None = None
    # The risk ratio's upper limit is infinite: in more than 2.5% of the resamples the risk at the
    # first setting was 0 (a separated resample's limit); ``rr_high`` is then null.
    rr_unbounded: bool = False
    n_boot: int = 0
    n_limit: int = 0  # resamples whose outcome was separated, kept at the likelihood's limit
    n_failed: int = 0  # resamples that could not be fit at all, left out
    failed_reason: str | None = None
    n_rr_infinite: int = 0
    n_rr_undefined: int = 0
    by_unit: str | None = None


class Marginal(_Model):
    declared: Literal["risk_difference", "risk_ratio"]
    method: str
    contrasts: list[MarginalContrast] = []
    refused: str | None = None  # no marginal risks at all
    interval_refused: str | None = None  # the risks are shown; their interval is not (too few units)
    exits: list[InferenceExit] = []
    concerns: list[str] = []


class DiagnosticTest(_Model):
    term: str
    statistic: float
    df: float
    p: float


class Diagnostic(_Model):
    check: Literal["proportional_hazards", "influence"]
    status: Literal["passed", "failed", "not_assessed"]
    method: str
    reading: str  # one sentence: what the check found, on these rows
    tests: list[DiagnosticTest] = []
    threshold: float | None = None
    reference: str | None = None  # what the threshold is: "the median of F(7, 893)"
    flagged: int | None = None
    largest: float | None = None
    n: int | None = None
    # Rows with leverage 1 (the fit passes through them): no Cook's distance, as R reports NaN.
    leverage_one: int | None = None
    exits: list[InferenceExit] = []
    response: str | None = None  # the recorded action (``respond_diagnostic``)
    change: list[Coefficient] | None = None  # what the response shows beside the estimate
    change_label: str | None = None


class EValueResult(_Model):
    measure: str
    rr: float
    rr_low: float | None = None
    rr_high: float | None = None
    point: float
    limit: float | None = None
    interval_includes_null: bool | None = None
    rare: bool | None = None
    converted: bool
    # A difference's E-value: the standard deviation it was standardized by, and whose it is
    # (MODELING_SEQUENCE §0 ruling 14): the surveyed population's (design-weighted) or the rows'.
    sd: float | None = None
    sd_basis: Literal["design_weighted", "sample"] | None = None


class BenchmarkResult(_Model):
    covariate: str
    r2dxj: float
    r2yxj: float
    r2dz: float
    r2yz: float
    estimate: float
    # The adjusted interval rests on the classical standard error: null when the interval shown is
    # not classical (``RobustnessResult.interval_note``).
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    kd: float


class RobustnessResult(_Model):
    exposure: str
    estimate: float
    se: float  # the classical standard error the robustness value's algebra uses
    t: float
    dof: float
    partial_r2: float
    rv: float
    rv_alpha: float | None = None  # null when the interval shown is not the classical one
    alpha: float
    covariance: str = "classical"  # the covariance of the interval shown
    interval_note: str | None = None
    benchmarks: list[BenchmarkResult] = []


class Sensitivity(_Model):
    feature: str
    methods: list[str]  # in rank order
    e_value: EValueResult | None = None
    robustness: RobustnessResult | None = None
    reading: str
    not_computed: str | None = None
    # What the analysis is of, when it is not the shown row itself: "the straight-line estimate
    # beside the curve" (``companion``, its row), or "the marginal risk ratio".
    of: str | None = None
    companion: Coefficient | None = None


class EffectsFamily(_Model):
    family: str
    label: str
    sequence: list[SequenceFit]
    appendix: list[AppendixModel]
    marginal: Marginal | None = None
    diagnostics: list[Diagnostic] = []
    sensitivity: list[Sensitivity] = []
    concerns: list[str] = []


class ModelOne(_Model):
    declared: list[str] | None
    guess: list[str]
    allowed: list[str]
    decision: dict[str, Any]
    reason: str


class EffectsArtifact(_Model):
    purpose: Literal["inference"]
    exposure: str
    exposures: list[str]
    measure: str | None = None
    measure_label: str | None = None
    scale: str | None = None  # ruling 9: "difference" | "ratio"
    conditioning: str | None = None  # "conditional" | "marginal" | "conditional and marginal"
    collapsible: bool | None = None
    rows: str = ""
    appendix_title: str
    multiplicity: str | None = None
    model_1: ModelOne | None = None
    families: list[EffectsFamily]
    methods: str


# ``multiplicity``: a family's one method (MS7's ``set_multiplicity`` or the estimand card's), which
# the feature-wise table carries in every declared model.
EFFECTS_READS = ("adjustment", "estimand", "multiplicity", "model_sequence", "diagnostic_responses",
                 "clusters", "purpose", "models", "task", "event", "target", "roles", "roles_unconfirmed",
                 "role_confirmations", "reading_confirmations", "shape_confirmations", "missing",
                 "survey", "outcome_order", "follow_up", "categorical", "energy_adjustment",
                 "grain", "exposure_forms", "lens", "findings", "column_units", "split",
                 "repeat_kind", "unit")
# The families whose declared models are refit on the primary's model matrix: every family with a
# coefficient table today, the families that model the unit (a random intercept, a working
# correlation) included, so the unadjusted estimate is shown for each (STROBE 16a).
SEQUENCE_FAMILIES = ("linear", "proportional_odds", "cox", "featurewise", "mixed", "gee")
LABELS = {"crude": "Unadjusted", "model_1": "Model 1", "model_2": "Model 2 (primary)",
          "model_3": "Model 3"}
STRAIGHT_LINE = "the straight-line estimate beside the curve"


# ── the model matrix's columns, by what they came from ───────────────────────


def matrix_sources(fitted: Any, inputs: Sequence[str]) -> dict[str, tuple[set[str], set[str]]]:
    """For each model-matrix column, (its raw source columns, its adjusted-lane sources), read from
    the fitted steps as the lineage reads them (``models/lineage.py``), never from a name."""
    from turbotab.core.models.lineage import _Origin, _run
    from turbotab.core.models.pipeline import ADJUST_STEPS

    steps = list(fitted.steps[:-1])
    adjusted = _run([(n, s) for n, s in steps if n in ADJUST_STEPS], [str(c) for c in inputs], None)
    matrix = _run([(n, s) for n, s in steps if n not in ADJUST_STEPS], list(adjusted), None)
    out: dict[str, tuple[set[str], set[str]]] = {}
    for column, origin in matrix.items():
        raw: set[str] = set()
        for a in origin.sources:
            raw |= set(adjusted.get(a, _Origin([a])).sources)
        out[str(column)] = (raw, set(origin.sources))
    return out


def energy_outputs(fitted: Any) -> set[str]:
    """The adjusted-lane columns the energy model made of total energy and the other energy
    sources (kept energy, ``kcal_from_<source>``, ``kcal_from_other``): the terms that enter with
    "energy" in Model 1."""
    step = getattr(fitted, "named_steps", {}).get("energy")
    if step is None or not hasattr(step, "lineage"):
        return set()
    out = set()
    for entry in step.lineage():
        if entry["operation"] in ("partition", "partition-other") or (
                entry["operation"] in ("kept", "pass-through")
                and entry["output"] == getattr(step, "energy_column", None)):
            out.add(str(entry["output"]))
    return out


def spline_terms(fitted: Any, features: Sequence[str]) -> tuple[str, list[str]] | None:
    """When the exposure enters as a restricted cubic spline: (its straight-line column, the
    spline's nonlinear columns), read from the fitted form step; None otherwise."""
    from turbotab.core.methods.exposure_form import form_step

    step = form_step(fitted)
    if step is None:
        return None
    wanted = {str(f) for f in features}
    for column, (form, _) in step._plan().items():
        if form != "spline":
            continue
        outputs = [str(o) for o in step._outputs(column)]
        if outputs[0] in wanted and set(outputs[1:]) & wanted:
            return outputs[0], outputs[1:]
    return None


# ── a table on a model matrix, for each family ───────────────────────────────


def matrix_table(family: Any, matrix: pd.DataFrame, y: Any, *, task: str, classes: Any,
                 clusters: Any, outcome: Any, survey: Any, levels: Any, event: Any,
                 features: Sequence[str]) -> Any:
    """The family's inference table on ``matrix`` (every analyzed row), as the fit stage computes
    it for the primary: design-based under ``survey`` for the families that have a design-based
    estimator (the stage blocks the others before they get here). None for a family whose table
    is not made from a matrix alone."""
    from turbotab.core.models.inference import _on_rows, _on_scale

    if family.key == "featurewise":
        from turbotab.core.models.featurewise import featurewise_table

        table = featurewise_table(matrix, y, task, [f for f in features if f in matrix.columns],
                                  clusters, event)
        return _on_rows(table, len(matrix), "all")
    if family.key == "cox":
        return family.inference_matrix(matrix, y, task="time_to_event", classes=[0, 1],
                                       clusters=clusters, outcome=outcome, rows="all", survey=survey)
    if family.key == "proportional_odds":
        return family.inference_matrix(matrix, y, task=task, classes=levels, clusters=clusters,
                                       outcome=outcome, rows="all", survey=survey)
    if family.key == "linear":
        return family.inference_matrix(matrix, y, task=task, classes=classes, clusters=clusters,
                                       outcome=outcome, rows="all", survey=survey)
    if family.key == "mixed":
        from turbotab.core.models.repeated import mixed_table

        return _on_rows(_on_scale(mixed_table(matrix, y, clusters), task, None, outcome),
                        len(matrix), "all")
    if family.key == "gee":
        from turbotab.core.models.repeated import gee_table

        return _on_rows(_on_scale(gee_table(matrix, y, clusters, task, classes), task, classes,
                                  outcome), len(matrix), "all")
    return None


# ── the stage ────────────────────────────────────────────────────────────────


def _not_applicable(state: Any, exposure: str, why: str) -> Bundle:
    from turbotab.core.models.effects import APPENDIX_TITLE

    return Bundle(data=EffectsArtifact(purpose="inference", exposure=exposure, exposures=[],
                                       appendix_title=APPENDIX_TITLE, families=[],
                                       methods=why).model_dump(mode="json"))


def supplied_copies_reason(implicate: str) -> str:
    """Why the declared models are withheld when the rows are the data's own imputed copies."""
    return (f"The rows are the data's own imputed copies (numbered by `{implicate}`). The fit's "
            f"table analyzes each copy with its own outcome and pools them by Rubin's rules, so its "
            f"primary model is the estimate. The declared models beside it (the unadjusted model, "
            f"Model 1 and Model 3), the marginal risks, the diagnostics and the sensitivity "
            f"analyses are not built over the copies, so none is shown: fit on the copies stacked, "
            f"each participant would count once per copy and every interval would leave out the "
            f"variation between the copies.")


def effects_stage(ctx: StageContext) -> Bundle:
    from turbotab.core import estimand as est
    from turbotab.core.models import get_family
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.effects import APPENDIX_TITLE
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, design_spec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_missing_for_table, _survey, _task, coded_outcome,
                                               imputed_copies_column, outcome_levels,
                                               read_assignment)

    state = ctx.state
    spec_e = est.current_estimand(state)
    if state.purpose != "inference" or spec_e is None:
        why = ("Under prediction no coefficient is read as an effect, so no effect is reported."
               if state.purpose != "inference" else
               "No effect is reported until the exposure and its effect are declared.")
        return _not_applicable(state, "", why)
    key = est.exposure_key(spec_e)
    copies = imputed_copies_column(state)
    if copies is not None:
        # Relation ``sequence-supplied-copies``: the fit pools its table over the data's own
        # imputed copies (MS3, wave 1b); this stage's models are not built over them, and on the
        # copies stacked each participant would count once per copy.
        return _not_applicable(state, key, supplied_copies_reason(copies))
    exposures = est.exposures_of(state, spec_e)
    task = _task(ctx)
    target = state.target
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    families = [get_family(k) for k in (state.models or []) if k in pipelines]
    families = [f for f in families if reports_coefficients(f) and hasattr(f, "inference")]
    if not families:
        raise ValueError("None of the chosen model families has a coefficient table, so there is "
                         "no effect to report; choose the linear model.")
    further = est.secondary_columns(state)
    declared = est.current_model_sequence(state)
    model_one = list(declared.model_1) if declared is not None else None
    assignment = read_assignment(ctx.inputs["split"])
    extra = [c for c in further if c not in spec.inputs]
    with open_store(ctx) as store:
        unit_columns = cluster_columns(state, store.columns)
        info = {c.name: c for c in store.info().columns}
        follow = []
        if task == "time_to_event":
            from turbotab.core.models.survival import follow_up_columns

            follow = follow_up_columns(state)
        columns = list(dict.fromkeys([*spec.inputs, *extra, target, *unit_columns, *follow]))
        frame = modeling_frame(store, columns, assignment.index.to_numpy(), outcome=target)
    strategy = state.missing.strategy if state.missing is not None else None
    # Every model but Model 3 is fit on every analyzed row, as the fit is (ruling 3). Model 3's
    # added columns may be missing where the primary's are not: under complete cases it alone is
    # fit on the rows where they are recorded; under multiple imputation they are imputed.
    rows3 = None
    if strategy != "multiple_imputation" and extra:
        recorded = frame[extra].notna().all(axis=1).to_numpy()
        if not recorded.all():
            rows3 = recorded
    frame3 = frame.loc[rows3] if rows3 is not None else frame
    spec3 = (design_spec(state, frame3[[*spec.inputs, *extra]],
                         [*spec.predictors, *[c for c in extra if c not in spec.predictors]],
                         column_info=info) if further else None)
    outcome = Outcome(name=target, labels=outcome_levels(task, frame[target].to_numpy(), state.event))
    levels = None
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, levels = ordinal_outcome(frame[target].to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    if task == "time_to_event":
        from turbotab.core.models.survival import time_to_event_outcome

        coded = time_to_event_outcome(state, frame, coded)
    y = np.asarray(coded)
    y3 = y[rows3] if rows3 is not None else y
    clusters = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
    clusters3 = (resolve_clusters(state, frame3[list(unit_columns)])
                 if unit_columns and rows3 is not None else clusters)
    survey, _ = _survey(ctx, clusters)
    keys = [f.key for f in families]
    # The primary's missing values exactly as the fit handles them (the same rows, columns and
    # seed, and since wave 1b's MS2 the same imputation model: the survey design under the
    # population answer, the clustering by unit, parts of totals left to their own models in the
    # energy identity), so under multiple imputation Model 2 pools the fit's own completed copies.
    # Model 3's copies are imputed with its columns; it differs in rows from the rest only under
    # complete cases, where no imputation model reads the design. Every copy is fit as the fit
    # stage fits it (MI repair, ``missing.copy_pipeline``): the knots placed once on the observed
    # values, and an impute step that refuses a blank made inside the copy, never the median.
    from turbotab.core.readings import nesting

    parts = dict((design.objects or {}).get("nested") or {})
    missing = _missing_for_table(ctx, spec, frame[list(spec.inputs)], y, task, keys,
                                 loss={"n_dropped": None}, survey=survey, clusters=clusters,
                                 nested=nesting(state, parts, columns=spec.inputs))
    missing3 = (_missing_for_table(ctx, spec3, frame3[list(spec3.inputs)], y3, task, keys,
                                   loss={"n_dropped": None}, survey=survey, clusters=clusters3,
                                   nested=nesting(state, parts, columns=spec3.inputs))
                if spec3 is not None else None)
    facts = est.measure_facts(str(spec_e.measure))
    run = _Run(ctx=ctx, state=state, spec_e=spec_e, key=key, exposures=exposures, task=task,
               frame=frame, y=y, spec=spec, spec3=spec3, pipelines=pipelines, outcome=outcome,
               levels=levels, clusters=clusters, survey=survey, missing=missing,
               model_one=model_one, further=further, extra=extra, unit_columns=list(unit_columns),
               rows3=rows3, frame3=frame3, y3=y3, clusters3=clusters3, missing3=missing3)
    out = []
    for i, family in enumerate(families):
        ctx.progress(0.05 + 0.9 * i / len(families), f"{family.label}: the declared models")
        out.append(run.family_block(family, build_pipeline))
    card = est.model_sequence_card(state)
    family_spec = bool(spec_e.family)
    artifact = EffectsArtifact(
        purpose="inference", exposure=key, exposures=exposures, measure=str(spec_e.measure),
        measure_label=est.MEASURE_WORDS.get(str(spec_e.measure)), scale=facts["scale"],
        conditioning=facts["conditioning"], collapsible=facts["collapsible"],
        rows=f"all {len(frame):,} analyzed rows", appendix_title=APPENDIX_TITLE,
        multiplicity=est.multiplicity_statement(state, spec_e, len(exposures)) if family_spec else None,
        model_1=ModelOne(**card) if card is not None else None,
        families=out, methods="")
    artifact.methods = methods_sentence(state, artifact)
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


def _first_sentence(text: str) -> str:
    """The first sentence of a refusal, without its period: what a methods sentence quotes."""
    head = re.split(r"(?<=[.!?])\s+", str(text).strip(), maxsplit=1)[0]
    return head.rstrip(".")


class _Run:
    """One stage run's shared inputs, and each family's block."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    # the copies the models are fit in: the completed copies under multiple imputation
    @staticmethod
    def copies(missing: Any, frame: pd.DataFrame) -> list[pd.DataFrame]:
        imputations = getattr(missing, "imputations", None) if missing else None
        if imputations is not None:
            return list(imputations.frames)
        return [frame]

    def _design(self) -> Any:
        return self.survey.design if self.survey is not None else None

    @staticmethod
    def _held(missing: Any) -> Inference:
        """A model whose missing values cannot be handled as the answer says: no table, the
        reason and the answer's exits (as the fit's table carries them)."""
        from turbotab.core.models.survey import blocked

        info = blocked(missing.refusal, missing.exits, estimator="not fitted").info
        return Inference(**{k: v for k, v in info.items() if k in Inference.model_fields})

    def _blocked(self, family: Any, info: Mapping[str, Any]) -> dict[str, Any]:
        """A family with no estimate (block and record, BLUEPRINT §11.3): Model 2 carries the
        reason and the fit's exits; nothing else is shown for it."""
        reason = str(info.get("refused") or info.get("caption") or "")
        inference = Inference(**{k: v for k, v in info.items() if k in Inference.model_fields})
        return EffectsFamily(family=family.key, label=family.label, sequence=[SequenceFit(
            key="model_2", label=LABELS["model_2"], adjusted_for=[], n_rows=len(self.frame),
            effects=None, inference=inference, concerns=[reason])], appendix=[],
            concerns=[reason]).model_dump(mode="json")

    def family_block(self, family: Any, build_pipeline: Any) -> dict[str, Any]:
        from turbotab.core import estimand as est
        from turbotab.core.methods.missing import copy_pipeline
        from turbotab.core.models.effects import split_rows
        from turbotab.core.models.inner_cv import fit_pipeline
        from turbotab.core.models.linear import model_matrix
        from turbotab.core.models.survey import blocked, has_design_estimator, no_design_estimator
        from turbotab.core.stages.modeling import _inference_table, with_units

        concerns: list[str] = []
        design = self._design()
        if self.survey is not None and self.survey.refusal:
            return self._blocked(family, blocked(self.survey.refusal, self.survey.exits).info)
        if self.missing is not None and self.missing.refusal:
            return self._blocked(family, blocked(self.missing.refusal, self.missing.exits,
                                                 estimator="not fitted").info)
        if design is not None and not has_design_estimator(family, self.task):
            # Ruling 6: under the surveyed population a family with no design-based estimator is
            # blocked and recorded, its exits the fit's (the family that has one; the sample-only
            # attestation), never shown unweighted.
            return self._blocked(family, no_design_estimator(family, self.task,
                                                             list(self.state.models or [])).info)
        units = (pd.Series(self.clusters.codes, index=self.frame.index)
                 if self.clusters is not None and self.clusters.clustered else None)
        pipeline = self.pipelines[family.key]
        predictors = list(est.predictor_roles(self.state))
        allowed_one = set(self.model_one or [])
        energy_named = bool(allowed_one & set(est.energy_terms_columns(self.state)))
        per_copy: dict[str, list[Any]] = {}
        failures: dict[str, str] = {}
        adjusted_for: dict[str, list[str]] = {}
        features: list[str] = []
        first: dict[str, Any] = {}
        curve: tuple[str, list[str]] | None = None  # a spline's straight-line and nonlinear columns
        supported = family.key in SEQUENCE_FAMILIES
        imputed = getattr(self.missing, "imputations", None) if self.missing else None
        for k, X_k in enumerate(self.copies(self.missing, self.frame)):
            self._unless_cancelled()
            X = X_k[list(self.spec.inputs)]
            # Under multiple imputation each copy is fit as the fit stage fits it (MI repair):
            # the knots placed once on the observed values, and an impute step that refuses a
            # blank made inside the copy instead of median-filling it (``copy_pipeline``).
            fitted = fit_pipeline(with_units(copy_pipeline(pipeline, imputed), units), X, self.y)
            if self.levels is not None:
                fitted[-1].level_names_ = list(self.levels)
            classes = list(getattr(fitted[-1], "classes_", [])) or None
            if not supported:
                table = self._table(lambda: _inference_table(
                    family, fitted, X, self.y, task=self.task, clusters=self.clusters,
                    outcome=self.outcome, rows="all", survey=design), "model_2", failures)
                per_copy.setdefault("model_2", []).append(table)
                if k == 0 and table is not None:
                    features = sorted({f for e in self.exposures for f in est.primary_features(
                        [r["feature"] for r in table.rows], e, predictors)})
                    adjusted_for["model_2"] = [c for c in self._scored(self.spec.predictors)
                                               if c not in self.exposures]
                continue
            M = model_matrix(fitted, X)
            sources = matrix_sources(fitted, self.spec.inputs)
            feats = [c for c in M.columns
                     if c in {f for e in self.exposures
                              for f in est.primary_features(M.columns, e, predictors)}]
            energy = energy_outputs(fitted)

            def in_model_one(c: str) -> bool:
                raw, adj = sources.get(str(c), ({str(c)}, {str(c)}))
                return bool(raw & allowed_one) or (energy_named and bool(adj & energy))

            subsets = {"crude": feats, "model_2": list(M.columns)}
            if self.model_one is not None:
                subsets["model_1"] = [c for c in M.columns if c in feats or in_model_one(c)]
            curve = spline_terms(fitted, feats) if family.key != "featurewise" else None
            if curve is not None:
                # The curve's straight-line estimate, for its sensitivity analysis: the same model
                # with the spline's nonlinear terms left out (pooled like every other model).
                subsets["straight"] = [c for c in M.columns if c not in set(curve[1])]
            kw = dict(task=self.task, classes=classes, clusters=self.clusters, outcome=self.outcome,
                      survey=design, levels=self.levels, event=self.state.event, features=feats)
            for name, cols in subsets.items():
                per_copy.setdefault(name, []).append(self._table(
                    lambda _c=cols: self._with_relative(
                        family, fitted, M[_c], self.y, matrix_table(family, M[_c], self.y, **kw),
                        kw), name, failures))
            if k == 0:
                features = feats
                first = {"fitted": fitted, "matrix": M, "sources": sources, "classes": classes}
                adjusted_for = {"crude": [], "model_2": self._raw(M.columns, feats, sources)}
                if "model_1" in subsets:
                    adjusted_for["model_1"] = self._raw(subsets["model_1"], feats, sources)
        comparison = None
        if self.spec3 is not None and supported:
            comparison = self._model_three(family, build_pipeline, per_copy, failures, predictors,
                                           design)
            adjusted_for["model_3"] = [*adjusted_for.get("model_2", []),
                                       *[c for c in self.further
                                         if c not in adjusted_for.get("model_2", [])]]
        if not supported:
            concerns.append(f"{family.label} has no model matrix here, so only the primary model is "
                            f"refit: the unadjusted model and Model 1 are refit for least squares, "
                            f"logistic, proportional-odds, Cox, feature-wise, mixed and GEE models.")
        sequence, appendix = [], []
        for name in ("crude", "model_1", "model_2", "model_3"):
            tables = per_copy.get(name)
            if not tables:
                continue
            missing = self.missing3 if name == "model_3" else self.missing
            spec = self.spec3 if name == "model_3" else self.spec
            n_model = len(self.frame3) if name == "model_3" else len(self.frame)
            table = self._pool(tables, missing, spec) if name not in failures else None
            if table is None:
                # Model 3's own imputation refused, or asked what only the user can settle (a
                # column's time-invariance under clustered imputation, MI repair): its exits stand
                # beside the reason, so the question carries its answers (BLUEPRINT §14.2).
                held = (missing is not None and name == "model_3" and bool(missing.refusal))
                sequence.append(SequenceFit(
                    key=name, label=LABELS[name], adjusted_for=adjusted_for.get(name, []),
                    note=self._note(name), n_rows=n_model, effects=None,
                    inference=self._held(missing) if held else None,
                    concerns=[f"It could not be fit: {failures.get(name, 'no table')}"]))
                continue
            if missing is not None:
                missing.record(table)
            table = self._multiplicity(family, table)
            rows = table.rows or []
            mine = {f for e in self.exposures
                    for f in est.primary_features([r["feature"] for r in rows], e, predictors)}
            shown, terms = split_rows(rows, mine)
            fit = SequenceFit(
                key=name, label=LABELS[name], adjusted_for=adjusted_for.get(name, []),
                note=self._note(name), n_rows=int(table.info.get("n_rows") or n_model),
                effects=shown if table.rows else None,
                inference=table.info, concerns=list(table.concerns))
            if name == "model_3" and comparison is not None:
                fit.comparison = [Coefficient(**{k: v for k, v in r.items()
                                                 if k in Coefficient.model_fields})
                                  for r in (comparison.rows or []) if r["feature"] in mine]
                fit.comparison_label = (
                    f"The primary's adjustment refit on Model 3's {len(self.frame3):,} rows, so "
                    f"Model 3 differs from it by {self._listing(self.extra)} alone.")
            sequence.append(fit)
            if terms:
                appendix.append(AppendixModel(key=name, label=LABELS[name], terms=terms))
        out = EffectsFamily(family=family.key, label=family.label, sequence=sequence,
                            appendix=appendix, concerns=concerns)
        primary = next((s for s in sequence if s.key == "model_2"), None)
        if supported and first:
            # Each analysis beside the estimates is reported, or says why it could not be: one
            # that fails never takes the declared models down with it (as the fit stage's tables).
            measure = str(self.spec_e.measure)
            try:
                out.marginal = self.marginal(family, pipeline)
            except Exception as exc:  # noqa: BLE001 - said in the artifact, never silent
                self._unless_cancelled()
                out.marginal = (Marginal(declared=measure, method="",  # type: ignore[arg-type]
                                         refused=f"The marginal risks could not be computed: {exc}")
                                if measure in est.MARGINAL else None)
            try:
                out.diagnostics = self.diagnostics(family, first, features, primary, pipeline,
                                                   units)
            except Exception as exc:  # noqa: BLE001
                self._unless_cancelled()
                check = "proportional_hazards" if family.key == "cox" else "influence"
                out.diagnostics = [Diagnostic(check=check, status="not_assessed", method="",
                                              reading=f"Not assessed: {exc}")]
            straight = None
            if curve is not None and "straight" not in failures:
                straight = self._pool(per_copy.get("straight") or [], self.missing, self.spec)
            try:
                out.sensitivity = self.sensitivity(family, first, features, primary, out.marginal,
                                                   curve, straight)
            except Exception as exc:  # noqa: BLE001
                self._unless_cancelled()
                out.sensitivity = [Sensitivity(feature=", ".join(features), methods=[], reading="",
                                               not_computed=f"Not computed: {exc}")]
        return out.model_dump(mode="json")

    def _model_three(self, family: Any, build_pipeline: Any, per_copy: dict[str, list[Any]],
                     failures: dict[str, str], predictors: Sequence[str], design: Any) -> Any:
        """Model 3 in each of its copies (its own rows; under multiple imputation, copies imputed
        with its added columns), and, when its rows are fewer than the primary's, the primary's
        adjustment refit on those rows (returned). Each copy is fit as the fit stage fits one
        (``copy_pipeline``), on the knots its own imputation placed."""
        from turbotab.core import estimand as est
        from turbotab.core.methods.missing import copy_pipeline
        from turbotab.core.models.inner_cv import fit_pipeline
        from turbotab.core.models.linear import model_matrix
        from turbotab.core.stages.modeling import with_units

        if self.missing3 is not None and self.missing3.refusal:
            # Model 3's own columns cannot be handled as the answer says (an imputation refused):
            # it is listed with that reason, the other models stand.
            per_copy.setdefault("model_3", []).append(None)
            failures.setdefault("model_3", self.missing3.refusal)
            return None
        pipeline3 = build_pipeline(self.spec3, family, self.task, "inference", len(self.frame3),
                                   len(self.spec3.inputs) + 1)
        units3 = (pd.Series(self.clusters3.codes, index=self.frame3.index)
                  if self.clusters3 is not None and self.clusters3.clustered else None)
        comparison = None
        imputed3 = getattr(self.missing3, "imputations", None) if self.missing3 else None
        for k, X_k in enumerate(self.copies(self.missing3, self.frame3)):
            self._unless_cancelled()
            X3 = X_k[list(self.spec3.inputs)]
            fitted3 = fit_pipeline(with_units(copy_pipeline(pipeline3, imputed3), units3), X3,
                                   self.y3)
            if self.levels is not None:
                fitted3[-1].level_names_ = list(self.levels)
            classes3 = list(getattr(fitted3[-1], "classes_", [])) or None
            M3 = model_matrix(fitted3, X3)
            feats3 = [c for c in M3.columns
                      if c in {f for e in self.exposures
                               for f in est.primary_features(M3.columns, e, predictors)}]
            kw3 = dict(task=self.task, classes=classes3, clusters=self.clusters3,
                       outcome=self.outcome, survey=design, levels=self.levels,
                       event=self.state.event, features=feats3)
            per_copy.setdefault("model_3", []).append(self._table(
                lambda: self._with_relative(family, fitted3, M3, self.y3,
                                            matrix_table(family, M3, self.y3, **kw3), kw3),
                "model_3", failures))
            if self.rows3 is not None and k == 0:
                sources3 = matrix_sources(fitted3, self.spec3.inputs)
                added = set(self.extra)
                own = [c for c in M3.columns
                       if not (sources3.get(str(c), ({str(c)}, set()))[0] & added)]
                comparison = self._table(lambda: matrix_table(family, M3[own], self.y3, **kw3),
                                         "model_3_comparison", failures)
                if comparison is not None:
                    comparison = self._multiplicity(family, comparison)
        return comparison

    def _multiplicity(self, family: Any, table: Any) -> Any:
        if family.key == "featurewise" and not table.info.get("refused"):
            # The family's one multiplicity method, as the fit's table carries it (MS7): with
            # unadjusted p-values recorded, no q column; every member is shown either way.
            from turbotab.core.methods.omics import apply_multiplicity, multiplicity_policy

            return apply_multiplicity(table, multiplicity_policy(self.state))
        return table

    @staticmethod
    def _listing(columns: Sequence[str]) -> str:
        from turbotab.core.voice import listing

        return listing(list(columns))

    def _unless_cancelled(self) -> None:
        if self.ctx.cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()

    def _table(self, fn: Any, name: str, failures: dict[str, str]) -> Any:
        """``fn()``, the table of one declared model in one copy; a model that cannot be fit is
        listed with its reason, never dropped silently and never taking the others down."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            self._unless_cancelled()
            failures.setdefault(name, str(exc))
            return None

    def _with_relative(self, family: Any, fitted: Any, matrix: pd.DataFrame, y: Any, table: Any,
                       kw: Mapping[str, Any]) -> Any:
        """Under the all-components model, each nutrient's average relative effect (the
        substitution; Tomova et al. 2022) beside the model's coefficients, from this model's own
        matrix and table (``methods.energy.relative_effect_rows``), wherever the model holds two or
        more energy sources; the unadjusted model holds one, so it has none."""
        adj = self.spec.energy_adjustment()
        if (table is None or not table.rows or adj is None or adj.method != "all_components"
                or family.key != "linear" or self.task == "multiclass"):
            return table
        from turbotab.core.methods.energy import relative_effect_rows

        step = getattr(fitted, "named_steps", {}).get("energy")
        factors = dict(getattr(getattr(step, "pooled_", step), "factors_", {}) or {})
        found = relative_effect_rows(
            matrix, list(adj.nutrients), factors,
            table=lambda m: matrix_table(family, m, y, **kw).rows, coefficients=table.rows)
        table.rows = [*table.rows, *found]
        return table

    def _raw(self, columns: Sequence[str], feats: Sequence[str],
             sources: Mapping[str, tuple[set[str], set[str]]]) -> list[str]:
        """The columns a model's matrix columns come from, the exposure's aside: each raw column,
        and a declared scale by its own name, never its items (MS8: the score is in the model and
        the items are not, so Table 2 and the methods say the model is adjusted for the scale)."""
        found: set[str] = set()
        exposures = set(self.exposures)
        scales = self._scales()
        for c in columns:
            if c in feats:
                continue
            raw, adjusted = sources.get(str(c), ({str(c)}, set()))
            scored = {a for a in adjusted if a in scales}
            items = {i for a in scored for i in scales[a]}
            found |= (scored | (set(raw) - items)) - exposures
        order = [*self.spec.inputs, *sorted(found)]
        return [c for c in dict.fromkeys(order) if c in found]

    def _scales(self) -> dict[str, list[str]]:
        """Each declared scale's name and its items (MS8)."""
        return {str(sc["name"]): [str(i) for i in sc.get("items") or []]
                for sc in getattr(self.spec, "scales", None) or []}

    def _scored(self, columns: Sequence[str]) -> list[str]:
        """``columns`` with each declared scale's items replaced by the scale's name, in order."""
        out: list[str] = []
        for c in columns:
            scale = next((name for name, items in self._scales().items() if c in items), None)
            out.append(scale or c)
        return list(dict.fromkeys(out))

    def _note(self, name: str) -> str | None:
        from turbotab.core.voice import listing

        if name == "crude":
            return "STROBE 16a's unadjusted estimate: no other column in the model."
        if name == "model_1":
            return "the declared first adjustment: " + (listing(self.model_one) if self.model_one
                                                        else "none")
        if name == "model_2":
            return "the full adjustment set the answers derive: the reported estimate."
        what = "a possible mediator" if len(self.further) == 1 else "possible mediators"
        note = f"further adjusted for {listing(self.further)}, {what}: not a total effect."
        if self.rows3 is not None:
            note += (f" Fit on the {len(self.frame3):,} of the {len(self.frame):,} analyzed rows with "
                     f"{listing(self.extra)} recorded; the primary's adjustment refit on those "
                     f"rows is shown beside it.")
        return note

    def _pool(self, tables: Sequence[Any], missing: Any, spec: Any) -> Any:
        from turbotab.core.models.inference import InferenceTable

        tables = [t for t in tables if t is not None]
        if not tables:
            return None
        if len(tables) == 1:
            return tables[0]
        if any(t.info.get("refused") for t in tables):
            return tables[0]
        from turbotab.core.methods.imputation import pool_rows
        from turbotab.core.methods.missing import mi_concerns, multiple_imputation_info

        imputations = missing.imputations
        y = self.y3 if spec is self.spec3 else self.y
        pooled = pool_rows([t.rows for t in tables])
        info = dict(tables[0].info)
        info["caption"] = (f"Multiple imputation, m = {len(tables)}: each completed copy analyzed "
                           f"as follows, then pooled by Rubin's rules. {info.get('caption', '')}"
                           ).strip()
        # MS2 (wave 1b): the record names the design's df as Rubin's complete-data df and the
        # largest Monte Carlo error among the pooled rows, as the fit's record does.
        from turbotab.core.stages.modeling import _design_df

        info["missing"] = multiple_imputation_info(imputations, spec, len(y), rows=pooled,
                                                   df_com=_design_df(tables[0]))
        concerns = list(tables[0].concerns) + mi_concerns(imputations, pooled, len(y))
        return InferenceTable(pooled, info, concerns)

    # ── the marginal risk difference and ratio ──

    def marginal(self, family: Any, pipeline: Any) -> Marginal | None:
        from sklearn.base import clone

        from turbotab.core import estimand as est
        from turbotab.core.models import effects
        from turbotab.core.models.inference import floor_refusal
        from turbotab.core.models.pipeline import transformer

        measure = str(self.spec_e.measure)
        if measure not in est.MARGINAL or family.key != "linear" or self.task != "binary":
            return None
        method = (f"Standardization over the analyzed rows (g-computation) from the logistic "
                  f"model: each row's predicted risk with the exposure set each way, averaged; "
                  f"95% percentile intervals from {effects.BOOT:,} bootstrap resamples that refit "
                  f"the whole chain")
        out = Marginal(declared=measure, method=method)  # type: ignore[arg-type]
        survey = self.survey
        if survey is not None and survey.design is not None:
            out.refused = ("The marginal risks are standardized over these participants; no "
                           "design-based standardization over the surveyed population is built "
                           "here.")
            exits = [{"label": "Describe these participants (the sample-only answer)",
                      "decision": {**self.state.survey.model_dump(mode="json"),
                                   "kind": "set_survey", "estimand": "sample"}},
                     {"label": "Report the conditional odds ratio",
                      "decision": {**self.spec_e.model_dump(mode="json"), "kind": "set_estimand",
                                   "measure": "odds_ratio"}}]
            out.exits = [InferenceExit(**e) for e in exits]
            return out
        if getattr(self.missing, "imputations", None) is not None:
            out.refused = ("The marginal risks are not pooled over multiple imputations here: the "
                           "bootstrap would have to repeat the imputation in every resample.")
            out.exits = [InferenceExit(label="Report the conditional odds ratio", decision={
                **self.spec_e.model_dump(mode="json"), "kind": "set_estimand",
                "measure": "odds_ratio"})]
            return out
        adj = self.spec.energy_adjustment()
        contrast = self.spec_e.contrast
        if contrast == "substitution" and adj is not None and adj.method == "all_components":
            out.refused = ("Under the all-components model a substitution moves several energy "
                           "sources at once (a weighted contrast); its standardization is not "
                           "built here. The conditional odds ratio's average relative effect "
                           "carries it.")
            out.exits = [InferenceExit(label="Report the conditional odds ratio", decision={
                **self.spec_e.model_dump(mode="json"), "kind": "set_estimand",
                "measure": "odds_ratio"})]
            return out
        exposure = str(self.spec_e.exposure)
        X = self.frame[list(self.spec.inputs)]
        energy = None
        if contrast == "addition" and adj is not None and adj.method == "partition":
            fitted = transformer(clone(pipeline).steps[:-1]).fit(X)
            step = fitted.named_steps.get("energy")
            factors = dict(getattr(getattr(step, "pooled_", step), "factors_", {}) or {})
            if exposure in factors:
                energy = (str(adj.energy_column), float(factors[exposure]))

        def fit(X_b: pd.DataFrame, y_b: np.ndarray) -> Any:
            steps = transformer(clone(pipeline).steps[:-1]).fit(X_b)

            def matrix_of(Z: pd.DataFrame) -> np.ndarray:
                M = steps.transform(Z)
                M = M.to_numpy(dtype=float) if hasattr(M, "to_numpy") else np.asarray(M, float)
                return np.column_stack([np.ones(len(Z)), M])
            return matrix_of

        seed = int(getattr(self.state.split, "seed", 0) or 0) if self.state.split is not None else 0
        units = (self.clusters.codes if self.clusters is not None and self.clusters.clustered
                 else None)
        # The unit floor binds this interval as it binds every other (§2: refusal below a floor):
        # a bootstrap of a handful of units is as narrow as the sandwich the fit refuses.
        refusal = floor_refusal(self.clusters, task="binary") if self.clusters is not None else None
        n_boot = 0 if refusal is not None else effects.BOOT
        try:
            settings = effects.settings_of(self.frame[exposure], exposure)
        except effects.Unestimable as exc:
            out.refused = f"The marginal risks cannot be standardized: {exc}."
            return out
        last = [0]

        def progress(b: int, total: int) -> None:
            if b - last[0] >= max(1, total // 20) or b == total:
                last[0] = b
                self.ctx.progress(0.5, f"Standardizing the risks: bootstrap resample {b:,} of "
                                       f"{total:,}")

        for setting in settings:
            try:
                found = effects.g_computation(
                    fit, X, self.y.astype(float), exposure, setting, n_boot=n_boot,
                    seed=seed, units=units,
                    unit_column=self.clusters.column if units is not None and n_boot else None,
                    energy=energy, progress=progress, cancelled=self.ctx.cancelled)
            except effects.Unestimable as exc:
                out.refused = f"The marginal risks cannot be standardized: {exc}."
                return out
            values = found.as_dict()
            unbounded = values["rr_high"] is not None and math.isinf(values["rr_high"])
            if unbounded:
                values["rr_high"] = None
            out.contrasts.append(MarginalContrast(
                **{k: v for k, v in values.items() if k in MarginalContrast.model_fields},
                rr_unbounded=unbounded))
        if refusal is not None:
            out.interval_refused = refusal[0]
            out.exits = [InferenceExit(**e) for e in refusal[1]]
            out.method = (f"Standardization over the analyzed rows (g-computation) from the "
                          f"logistic model: each row's predicted risk with the exposure set each "
                          f"way, averaged; no interval is reported ({_first_sentence(refusal[0])})")
        out.concerns = self._marginal_concerns(out.contrasts)
        return out

    @staticmethod
    def _marginal_concerns(contrasts: Sequence[MarginalContrast]) -> list[str]:
        """What the bootstrap met and how it was handled, each said once per contrast."""
        out = []
        for c in contrasts:
            lead = f"{c.setting}: " if len(contrasts) > 1 else ""
            if c.n_limit:
                out.append(
                    f"{lead}In {c.n_limit:,} of the {c.n_boot:,} bootstrap resamples the covariates "
                    f"separated the outcome (a level or a combination with no event, or only "
                    f"events, in that resample). Each is kept at the likelihood's limit, where the "
                    f"separated rows' risks are 0 or 1, the fit R's glm approaches as its "
                    f"coefficients run away; leaving them out would cut the interval's most "
                    f"extreme resamples.")
            if c.n_failed:
                out.append(
                    f"{lead}{c.n_failed:,} of the {c.n_boot:,} resamples could not be fit "
                    f"({c.failed_reason}) and are left out, so the intervals rest on the other "
                    f"{c.n_boot - c.n_failed:,} and may be too narrow.")
            if c.rr_unbounded:
                out.append(
                    f"{lead}The risk ratio's upper limit is unbounded: in {c.n_rr_infinite:,} of "
                    f"the resamples the risk at the first setting was 0, so its ratio was "
                    f"infinite.")
            if c.n_rr_undefined:
                out.append(
                    f"{lead}In {c.n_rr_undefined:,} resamples both risks were 0, so no risk ratio "
                    f"exists there; its interval rests on the others.")
        return out

    # ── diagnostics ──

    def diagnostics(self, family: Any, first: Mapping[str, Any], features: Sequence[str],
                    primary: SequenceFit | None, pipeline: Any, units: Any) -> list[Diagnostic]:
        from turbotab.core import estimand as est

        responses = est.current_responses(self.state)
        imputed = getattr(self.missing, "imputations", None) is not None
        on = " (on the first of the completed copies)" if imputed else ""
        M = first["matrix"]
        if self.survey is not None and self.survey.design is not None:
            reason = ("Not assessed: the model is design-based, and these checks read an "
                      "unweighted fit.")
            check = "proportional_hazards" if family.key == "cox" else "influence"
            return [Diagnostic(check=check, status="not_assessed", method="", reading=reason)]
        if family.key == "cox":
            return [self._ph(M, features, first["sources"], responses, on)]
        if family.key == "linear" and self.task in ("regression", "binary"):
            return [self._influence(family, M, features, responses, on, units)]
        return [Diagnostic(check="influence", status="not_assessed", method="",
                           reading=f"Not assessed for {family.label}: leverage and Cook's distance "
                                   f"are computed for least-squares and logistic models.")]

    def _exits(self, check: str, single: bool) -> list[InferenceExit]:
        from turbotab.core import estimand as est

        actions = [a for a in est.DIAGNOSTIC_ACTIONS[check]
                   if single or a == "keep_labeled"]
        return [InferenceExit(label=est.ACTION_WORDS[a][0].upper() + est.ACTION_WORDS[a][1:],
                              decision={"kind": "respond_diagnostic", "exposure": self.key,
                                        "check": check, "action": a}) for a in actions]

    def _ph(self, M: pd.DataFrame, features: Sequence[str], sources: Mapping[str, Any],
            responses: Mapping[str, str], on: str) -> Diagnostic:
        from turbotab.core.models import effects
        from turbotab.core.models.inference import format_p
        from turbotab.core.models.survival import cox_fit

        X = M.to_numpy(dtype=float)
        fit = cox_fit(X, self.y)
        method = (f"Proportional hazards by Schoenfeld residuals: the score test of each term's "
                  f"product with a function of time (one minus the Kaplan–Meier estimate), as R's "
                  f"cox.zph computes it ({effects.GRAMBSCH_THERNEAU}){on}")
        terms: dict[str, list[int]] = {}
        for j, c in enumerate(M.columns):
            if c in features:
                name = self.key if self.key != "*" else str(c)
            else:
                raw = sorted(sources.get(str(c), ({str(c)}, set()))[0])
                name = raw[0] if len(raw) == 1 else str(c)
            terms.setdefault(name, []).append(j)
        zph = effects.cox_zph(X, self.y, fit.beta, terms)
        tests = [DiagnosticTest(term=r["term"], statistic=r["chisq"], df=r["df"], p=r["p"])
                 for r in [*zph["terms"], zph["global"]]]
        mine = [r for r in zph["terms"] if any(M.columns[j] in features for j in terms[r["term"]])]
        failed = [r for r in mine if r["p"] < effects.PH_ALPHA]
        g = zph["global"]
        if not failed:
            reading = (f"The exposure's hazard ratio shows no departure from proportional hazards "
                       f"at the customary {effects.PH_ALPHA:g} level (p = "
                       f"{format_p(min(r['p'] for r in mine)) if mine else 'not tested'}; global "
                       f"χ²({g['df']}) = {g['chisq']:.2f}, p = {format_p(g['p'])}).")
            return Diagnostic(check="proportional_hazards", status="passed", method=method,
                              reading=reading, tests=tests)
        worst = min(failed, key=lambda r: r["p"])
        single = len(features) == 1
        reading = (f"The exposure's hazard ratio changes over follow-up (χ²({worst['df']}) = "
                   f"{worst['chisq']:.2f}, p = {format_p(worst['p'])}, below the customary "
                   f"{effects.PH_ALPHA:g}): the reported ratio is an average over follow-up.")
        out = Diagnostic(check="proportional_hazards", status="failed", method=method,
                         reading=reading, tests=tests, exits=self._exits("proportional_hazards", single))
        action = responses.get("proportional_hazards")
        if action is not None:
            out.response, out.exits = action, []
            if action == "period_hazard_ratios" and single:
                j = list(M.columns).index(features[0])
                events = self.y["time"][self.y["event"].astype(bool)]
                cut = float(np.median(events))
                periods = effects.period_hazard_ratios(X, self.y, j, cut)
                out.change = [Coefficient(
                    feature=f"{features[0]} ({p['period']} {cut:.4g})", estimate=p["estimate"],
                    ci_low=p["ci_low"], ci_high=p["ci_high"], p=None, se=p["se"],
                    ratio=p["ratio"], ratio_low=p["ratio_low"], ratio_high=p["ratio_high"])
                    for p in periods["periods"]]
                out.change_label = (f"The hazard ratio before and after {cut:.4g}, the median "
                                    f"event time: follow-up split there, the exposure's product "
                                    f"with the later period added.")
        return out

    def _influence(self, family: Any, M: pd.DataFrame, features: Sequence[str],
                   responses: Mapping[str, str], on: str, units: Any) -> Diagnostic:
        from turbotab.core.models import effects

        X = np.column_stack([np.ones(len(M)), M.to_numpy(dtype=float)])
        y = self.y.astype(float)
        try:
            found = (effects.ols_influence(X, y) if self.task == "regression"
                     else effects.logistic_influence(X, y))
        except (effects.Unestimable, np.linalg.LinAlgError) as exc:
            return Diagnostic(check="influence", status="not_assessed", method="",
                              reading=f"Not assessed: {exc}.")
        p, n = int(found["p"]), int(found["n"])
        threshold = effects.cook_threshold(p, n)
        D = np.asarray(found["cooks"], dtype=float)
        # A NaN distance (leverage 1) is above no reference, as R's comparison leaves it out.
        flagged = np.flatnonzero(np.nan_to_num(D, nan=-np.inf) > threshold)
        aliased = int(found.get("aliased") or 0)
        rank = (f"; p is the model's rank, {aliased:,} aliased column"
                f"{'s' if aliased != 1 else ''} left out as R's lm leaves {'them' if aliased != 1 else 'it'}"
                if aliased else "")
        method = (f"Leverage and Cook's distance of every row, as R's hatvalues and "
                  f"cooks.distance give them, against the median of F({p}, {n - p:,}) = "
                  f"{threshold:.3g} ({effects.COOK}{rank}){on}")
        finite = D[np.isfinite(D)]
        largest = float(finite.max()) if len(finite) else None
        one = int(found.get("leverage_one") or 0)
        base = dict(check="influence", method=method, threshold=threshold,
                    reference=f"the median of F({p}, {n - p:,})", flagged=int(len(flagged)),
                    largest=largest, n=n, leverage_one=one)
        exact = ""
        if one:
            exact = (f" {one:,} row{'s' if one != 1 else ''} with leverage 1 (the fit passes "
                     f"through {'them' if one != 1 else 'it'}, as through the one row of a level "
                     f"held by one row) {'have' if one != 1 else 'has'} no Cook's distance, as R "
                     f"reports it: removing {'one' if one != 1 else 'it'} moves only the "
                     f"coefficient that fits it.")
        top = f"{largest:.3g}" if largest is not None else "none"
        if not len(flagged):
            return Diagnostic(status="passed", tests=[], **base,
                              reading=(f"No row moves the estimates past Cook's reference "
                                       f"(largest distance {top}, against {threshold:.3g})."
                                       + exact))
        reading = (f"{len(flagged):,} row{'s move' if len(flagged) != 1 else ' moves'} the estimates "
                   f"past Cook's reference (largest distance {top}, against {threshold:.3g})."
                   + exact)
        out = Diagnostic(status="failed", reading=reading, exits=self._exits("influence", True),
                         **base)
        action = responses.get("influence")
        if action is not None:
            out.response, out.exits = action, []
            if action == "without_influential":
                keep = np.ones(len(M), dtype=bool)
                keep[flagged] = False
                from turbotab.core.models.inference import resolve_clusters

                clusters = (resolve_clusters(self.state, self.frame.loc[keep, self.unit_columns])
                            if self.unit_columns else None)
                table = matrix_table(family, M.loc[keep], self.y[keep], task=self.task,
                                     classes=[0, 1] if self.task == "binary" else None,
                                     clusters=clusters, outcome=self.outcome, survey=None,
                                     levels=None, event=self.state.event, features=features)
                rows = [Coefficient(**{k: v for k, v in r.items() if k in Coefficient.model_fields})
                        for r in table.rows if r["feature"] in features]
                out.change = rows
                out.change_label = (f"The primary model refit without the {len(flagged):,} "
                                    f"influential row{'s' if len(flagged) != 1 else ''}, on "
                                    f"{int(keep.sum()):,} rows.")
        return out

    # ── sensitivity to unmeasured confounding ──

    def sensitivity(self, family: Any, first: Mapping[str, Any], features: Sequence[str],
                    primary: SequenceFit | None, marginal: Marginal | None,
                    curve: tuple[str, list[str]] | None = None,
                    straight: Any = None) -> list[Sensitivity]:
        """Every reported estimate's sensitivity analysis (ruling 10: "offered for every inference
        result"), or why one is not defined for it."""
        from turbotab.core import estimand as est

        if primary is None or not primary.effects:
            return []
        rows = [r for r in primary.effects if r.estimate is not None]
        if not rows:
            return []
        if self.task in ("ordinal", "multiclass"):
            return [Sensitivity(feature=rows[0].feature, methods=[], reading="",
                                not_computed=f"No E-value or robustness value is defined here for "
                                             f"a {est.MEASURE_WORDS[est.MEASURE_OF_TASK[self.task]]}.")]
        M, sources = first["matrix"], first["sources"]
        imputed = getattr(self.missing, "imputations", None) is not None
        covariance = str(primary.inference.covariance) if primary.inference is not None else "none"
        if curve is not None:
            return [self._curve(family, M, sources, features, curve, straight, marginal, imputed)]
        return [self._one(family, row, M, sources, marginal, imputed, covariance, features)
                for row in rows]

    def _curve(self, family: Any, M: pd.DataFrame, sources: Mapping[str, Any],
               features: Sequence[str], curve: tuple[str, list[str]], table: Any,
               marginal: Marginal | None, imputed: bool) -> Sensitivity:
        """A curve has no one coefficient to bound. Under a declared marginal measure the curve's
        own one-unit standardization is the estimate; otherwise its straight-line estimate beside
        it (``table``: the same model, the spline's nonlinear terms left out) is analyzed, and the
        reading says so."""
        straight, nonlinear = curve
        whole = str(self.spec_e.exposure or straight)
        contrast = self._contrast_for(marginal, straight, features)
        if contrast is not None and contrast.rr is not None:
            return self._marginal_sensitivity(whole, contrast, imputed)
        row = next((r for r in (table.rows or []) if r["feature"] == straight), None) if table else None
        if row is None or row.get("estimate") is None:
            return Sensitivity(feature=whole, methods=[], reading="",
                               not_computed="The exposure enters as a curve, and its straight-line "
                                            "estimate, which the sensitivity analysis would bound, "
                                            "could not be fit.")
        companion = Coefficient(**{k: v for k, v in row.items() if k in Coefficient.model_fields})
        keep = [c for c in M.columns if c not in set(nonlinear)]
        out = self._one(family, companion, M[keep], sources, None, imputed,
                        str(table.info.get("covariance") or "none"), [straight],
                        what=STRAIGHT_LINE)
        if companion.ratio is not None:
            said = f"a ratio of {companion.ratio:.4g} per unit"
            low, high = companion.ratio_low, companion.ratio_high
        else:
            said = f"{companion.estimate:.4g} per unit"
            low, high = companion.ci_low, companion.ci_high
        if low is not None and high is not None:
            said += f", 95% CI {low:.4g} to {high:.4g}"
        lead = ("The exposure enters as a curve (a restricted cubic spline), which no single "
                "coefficient carries; this analysis bounds the straight-line estimate beside it, "
                f"the same model with the spline's nonlinear terms left out ({said}).")
        out.reading = f"{lead} {out.reading}".strip()
        out.of, out.companion = STRAIGHT_LINE, companion
        return out

    def _contrast_for(self, marginal: Marginal | None, feature: str,
                      features: Sequence[str]) -> MarginalContrast | None:
        """The declared marginal contrast a row of the exposure's is read through: the one
        contrast of a single-term or curved exposure, or a categorical exposure's level against
        the first (its indicator, ``<exposure>_<level>``)."""
        if marginal is None or not marginal.contrasts:
            return None
        if len(marginal.contrasts) == 1:
            return marginal.contrasts[0]
        exposure = str(self.spec_e.exposure or "")
        for c in marginal.contrasts:
            found = re.match(r"^`(.+)` against `(.+)`$", c.setting)
            if found and feature == f"{exposure}_{found.group(1)}":
                return c
        return None

    def _event_share(self) -> float:
        """The event's share among the analyzed rows: the population's (weighted) under the
        surveyed-population answer, as the design-based estimate it qualifies."""
        values = (self.y["event"] if self.task == "time_to_event" else self.y).astype(float)
        design = self._design()
        if design is not None:
            from turbotab.core.models.survey import domain_of

            domain = domain_of(self.frame.index, design)
            return float(np.average(values[domain.keep], weights=domain.weight))
        return float(np.mean(values))

    def _outcome_sd(self) -> tuple[float, str]:
        """The outcome's standard deviation and its basis: the population's under the
        surveyed-population answer (R survey's ``svyvar`` over the design's domain, every family
        shown there being design-based), else the rows', by the one rule the causal lane uses too
        (``models.effects.estimand_sd``; MODELING_SEQUENCE §0 ruling 14)."""
        from turbotab.core.models.effects import estimand_sd

        values = np.asarray(self.y.astype(float))
        design = self._design()
        if design is not None:
            from turbotab.core.models.survey import domain_of

            domain = domain_of(self.frame.index, design)
            return estimand_sd(values[domain.keep], domain.raw)
        return estimand_sd(values)

    def _marginal_sensitivity(self, feature: str, c: MarginalContrast,
                              imputed: bool) -> Sensitivity:
        from turbotab.core.models import effects

        high = math.inf if c.rr_unbounded else c.rr_high
        found = effects.unmeasured_confounding(measure="risk_ratio", estimate=float(c.rr),
                                               ci_low=c.rr_low, ci_high=high)
        out = self._sensitivity(feature, found, imputed, None, what="the marginal risk ratio")
        out.of = "the marginal risk ratio"
        return out

    def _one(self, family: Any, row: Coefficient, M: pd.DataFrame, sources: Mapping[str, Any],
             marginal: Marginal | None, imputed: bool, covariance: str, features: Sequence[str],
             what: str = "the estimate") -> Sensitivity:
        from turbotab.core.models import effects

        feature = str(row.feature)
        design = self._design()
        if self.task == "regression":
            # MODELING_SEQUENCE §0 ruling 14: the SD the estimand speaks of. Under the surveyed-
            # population answer every family shown is design-based and reports the population's
            # difference, standardized by the outcome's design-weighted SD over the same domain;
            # otherwise by the analyzed rows' own SD.
            sd, basis = self._outcome_sd()
            matrix = y = None
            benchmarks = None
            reason = None
            if family.key not in ("linear", "featurewise"):
                reason = (f"it is defined for one least-squares coefficient "
                          f"({effects.CINELLI_HAZLETT}), and this family ({family.label}) does "
                          f"not fit by least squares")
            elif design is not None:
                reason = "the robustness value is defined for an unweighted least-squares fit"
            elif feature not in M.columns:
                reason = ("the average relative effect is a contrast of several coefficients, not "
                          "one regressor's")
            elif family.key == "featurewise":
                # Each member's own test: the outcome on that member and the adjustment columns
                # (the other members are separate questions), its covariates the benchmarks.
                others = set(features) - {feature}
                matrix = M[[c for c in M.columns if c not in others]]
                y = self.y.astype(float)
                benchmarks = self._benchmarks(M, sources, features)
            else:
                matrix, y = M, self.y.astype(float)
                benchmarks = self._benchmarks(M, sources, features)
            found = effects.unmeasured_confounding(
                measure="mean_difference", estimate=float(row.estimate), ci_low=row.ci_low,
                ci_high=row.ci_high, se=row.se, outcome_sd=sd, matrix=matrix, y=y,
                exposure_column=feature if matrix is not None else None, benchmarks=benchmarks,
                covariance=covariance, sd_basis=basis)
            return self._sensitivity(feature, found, imputed, reason, what=what)
        if family.key == "featurewise":
            return Sensitivity(feature=feature, methods=[], reading="",
                               not_computed="The feature-wise design models each exposure on the "
                                            "outcome, so the exposure is no regressor whose "
                                            "confounding these analyses bound.")
        contrast = self._contrast_for(marginal, feature, features)
        if contrast is not None and contrast.rr is not None:
            return self._marginal_sensitivity(feature, contrast, imputed)
        share = self._event_share()
        measure = "hazard_ratio" if self.task == "time_to_event" else "odds_ratio"
        ratio = row.ratio if row.ratio is not None else math.exp(float(row.estimate))
        found = effects.unmeasured_confounding(measure=measure, estimate=float(ratio),
                                               ci_low=row.ratio_low, ci_high=row.ratio_high,
                                               outcome_share=share)
        return self._sensitivity(feature, found, imputed, None, what=what)

    def _benchmarks(self, M: pd.DataFrame, sources: Mapping[str, Any],
                    feats: Sequence[str]) -> dict[str, list[str]]:
        """Each adjusted covariate (a raw column) and the matrix columns it is made of."""
        from turbotab.core import estimand as est

        adjusted = {c for c, d in est.derived_roles(self.state).items() if d.adjusted}
        out: dict[str, list[str]] = {}
        for c in M.columns:
            if c in feats:
                continue
            raw = sources.get(str(c), ({str(c)}, set()))[0]
            if len(raw) == 1 and next(iter(raw)) in adjusted:
                out.setdefault(next(iter(raw)), []).append(str(c))
        return out

    def _sensitivity(self, feature: str, found: Mapping[str, Any], imputed: bool,
                     reason: str | None, what: str = "the estimate") -> Sensitivity:
        e = found.get("e_value")
        r = found.get("robustness")
        parts = [sensitivity_reading(found, feature, str(self.state.target), what)]
        if imputed and r is not None:
            parts.append("The robustness value reads the first completed copy.")
        return Sensitivity(
            feature=feature, methods=list(found.get("methods") or []),
            e_value=EValueResult(**e) if e is not None else None,
            robustness=RobustnessResult(**{**r, "benchmarks": [BenchmarkResult(**b) for b in
                                                              r.get("benchmarks") or []]})
            if r is not None else None,
            reading=" ".join(p for p in parts if p),
            not_computed=(f"No robustness value: {reason}." if reason else None))


def sensitivity_reading(found: Mapping[str, Any], feature: str, target: str,
                        what: str = "the estimate") -> str:
    """What one sensitivity analysis says (``models.effects.unmeasured_confounding``'s result), in
    the words every inference result and the causal lane share: the robustness value with the
    benchmark that moves the estimate most, then the E-value, never a pass or a fail. Where the
    interval shown is not the classical one, the robustness value's interval form and the
    benchmarks' intervals are not given, and the reading says why."""
    from turbotab.core.models import effects
    from turbotab.core.voice import tick

    e = found.get("e_value")
    r = found.get("robustness")
    parts = []
    if r is not None:
        bench = r.get("benchmarks") or []
        lead = (f"An unmeasured confounder would need a partial R² of {r['rv']:.1%} with both "
                f"{tick(feature)} and {tick(target)}, beyond the measured covariates, to bring the "
                f"estimate to zero")
        if r.get("rv_alpha") is not None:
            lead += f", and {r['rv_alpha']:.1%} to bring its 95% interval to include zero"
        lead += f" ({effects.CINELLI_HAZLETT})"
        if bench:
            b = max(bench, key=lambda x: abs(r["estimate"] - x["estimate"]))
            lead += (f"; one as strong as {tick(b['covariate'])} would move it to "
                     f"{b['estimate']:.4g}")
            if b.get("ci_low") is not None and b.get("ci_high") is not None:
                lead += f" (95% CI {b['ci_low']:.4g} to {b['ci_high']:.4g})"
        parts.append(lead + ".")
        if r.get("interval_note"):
            parts.append(str(r["interval_note"]))
    if e is not None:
        if e.get("interval_includes_null"):
            limit = "1, as its interval includes the null"
        elif e.get("limit") is not None:
            limit = f"{e['limit']:.2f}"
        elif e.get("rr_low") is None and e.get("rr_high") is None:
            limit = "none, as no interval is reported"
        else:
            limit = "not computed"
        parts.append(f"E-value for {what}: {e['point']:.2f}, and for the confidence limit "
                     f"nearer the null {limit} ({effects.VANDERWEELE_DING}); an E-value is read "
                     f"against the confounders one can name, never as a pass or a fail.")
    return " ".join(parts)


SENSITIVITY_NAMES = {
    "robustness_value": "the Cinelli–Hazlett robustness value (each adjusted covariate a named "
                        "benchmark)",
    "e_value": "the E-value for the estimate and for the confidence limit nearer the null"}


# MODELING_SEQUENCE §0 ruling 14: under the surveyed-population answer a difference's E-value is
# standardized by the population's SD, which the methods text says (the rows' own SD is the
# default reading of VanderWeele & Ding's approximation, and goes unsaid).
POPULATION_SD = ("the difference standardized by the outcome's design-weighted standard deviation "
                 "in the surveyed population")


def sensitivity_clause(methods: Sequence[str], names: Mapping[str, str] | None = None,
                       of: str | None = None, sd_basis: str | None = None) -> str:
    """The methods text's sentence for the analyses that ran, in rank order; ``of`` names what
    they bound when it is not the reported estimate itself, and ``sd_basis`` the E-value's
    standard deviation, named when it is the surveyed population's."""
    words = {**SENSITIVITY_NAMES, **(names or {})}
    if sd_basis == "design_weighted" and "e_value" in words:
        words["e_value"] = f"{words['e_value']}, {POPULATION_SD}"
    subject = f" of {of}" if of else ""
    return (f"Sensitivity to unmeasured confounding{subject} is reported by "
            f"{' and by '.join(words[m] for m in methods)}, never as a pass or a fail.")


# ── the methods sentence ─────────────────────────────────────────────────────


def _robustness_name(lines: Sequence[Sensitivity]) -> str | None:
    """The robustness value's name in the methods text, from what was computed: named benchmarks
    or none, and its interval form when the intervals shown are not classical."""
    found = [s.robustness for s in lines if s.robustness is not None]
    if not found:
        return None
    benchmarked = any(r.benchmarks for r in found)
    words = ("each adjusted covariate a named benchmark" if benchmarked else
             "no adjusted covariate to benchmark it against")
    plain = next((r for r in found if r.rv_alpha is None), None)
    if plain is not None:
        words += (f"; its form for the 95% interval, and the benchmarks' intervals, are not "
                  f"reported, as they assume classical standard errors and the intervals "
                  f"reported are {plain.covariance}")
    return f"the Cinelli–Hazlett robustness value ({words})"


UNFIT_WORDS = {"crude": "The unadjusted model", "model_1": "Model 1"}


def methods_sentence(state: Any, artifact: EffectsArtifact) -> str:
    """The methods text this stage writes, from what it fitted: the sequence, the display rule, the
    marginal standardization, the diagnostics and their responses, and the sensitivity analyses."""
    from turbotab.core.models.effects import COOK, WESTREICH
    from turbotab.core.voice import listing, tick

    if not artifact.families:
        return artifact.methods
    whose = ("each exposure" if artifact.exposure == "*" else tick(artifact.exposure))
    fam = artifact.families[0]
    # The models reported: one declared but not fit (Model 3 held by its own imputation's question,
    # say) is said apart, never listed among them.
    keys = [s.key for s in fam.sequence if s.effects]
    two = next((s for s in fam.sequence if s.key == "model_2"), None)
    if two is None or not two.effects:
        why = ((two.inference.refused if two is not None and two.inference is not None else None)
               or (two.concerns[0] if two is not None and two.concerns else None)
               or "no model could be fit")
        return f"No estimate of {whose} is reported: {why}"
    seq = []
    if "crude" in keys:
        seq.append("unadjusted")
    if "model_1" in keys:
        one = next(s for s in fam.sequence if s.key == "model_1")
        seq.append(f"Model 1, adjusted for {listing(one.adjusted_for) if one.adjusted_for else 'nothing besides it'}")
    seq.append("Model 2, the primary, adjusted for "
               + (listing(two.adjusted_for, limit=8) if two.adjusted_for else "nothing besides it"))
    if "model_3" in keys:
        three = next(s for s in fam.sequence if s.key == "model_3")
        added = [c for c in three.adjusted_for if c not in two.adjusted_for]
        what = "a possible mediator" if len(added) == 1 else "possible mediators"
        part = f"Model 3, further adjusted for {listing(added)}, {what}, so not a total effect"
        if three.n_rows < two.n_rows:
            part += (f", on the {three.n_rows:,} rows with {listing(added)} recorded, beside the "
                     f"primary's adjustment refit on those rows")
        seq.append(part)
    if len(seq) == 1:
        text = f"The estimate of {whose} is reported from the primary model fit on {artifact.rows}: {seq[0]}."
    else:
        text = (f"The estimate of {whose} is reported across a declared sequence of models fit on "
                f"{artifact.rows}: {'; '.join(seq)}.")
    for s in fam.sequence:
        if s.effects or s.key == "model_2":
            continue
        if s.key == "model_3":
            added = [c for c in s.adjusted_for if c not in two.adjusted_for]
            text += (f" Model 3, further adjusted for {listing(added)}, could not be fit and is not "
                     f"reported.")
        else:
            text += f" {UNFIT_WORDS[s.key]} could not be fit and is not reported."
    text += (f" Only the exposure's estimates are shown as effects; every other coefficient is "
             f"listed apart as an adjustment term, not an effect estimate ({WESTREICH}).")
    survey = two.inference.survey if two.inference is not None else None
    if two.inference is not None and two.inference.covariance == "design" and survey is not None:
        weight = f" by {tick(survey.weight)}" if survey.weight else ""
        text += (f" Each model is design-based: weighted{weight}, with Taylor-linearized intervals "
                 f"over the survey's strata and PSUs, so the estimates describe the surveyed "
                 f"population.")
    if fam.marginal is not None and not fam.marginal.refused and fam.marginal.contrasts:
        c = fam.marginal.contrasts[0]
        text += (f" The marginal risk difference and risk ratio ({c.setting}) were estimated by "
                 f"standardization over the analyzed rows (g-computation) from the logistic model")
        if fam.marginal.interval_refused:
            text += f"; no interval is reported ({_first_sentence(fam.marginal.interval_refused)})."
        else:
            text += (f", with 95% percentile intervals from {c.n_boot:,} bootstrap resamples "
                     f"refitting the whole chain" + (f" by `{c.by_unit}`" if c.by_unit else ""))
            limits = {x.n_limit for x in fam.marginal.contrasts}
            fails = {x.n_failed for x in fam.marginal.contrasts}
            if max(limits):
                some = (f"{max(limits):,}" if len(limits) == 1 else
                        f"between {min(limits):,} and {max(limits):,}")
                text += (f"; in {some} of them the covariates separated the outcome, and the model "
                         f"was taken at the likelihood's limit (the separated rows' risks 0 or 1)")
            if max(fails):
                some = (f"{max(fails):,}" if len(fails) == 1 else
                        f"between {min(fails):,} and {max(fails):,}")
                text += f"; {some} could not be fit and are left out of the intervals"
            text += "."
    for d in fam.diagnostics:
        if d.status == "not_assessed":
            continue
        if d.check == "proportional_hazards":
            g = next((t for t in d.tests if t.term == "GLOBAL"), None)
            text += (f" Proportional hazards were checked by Schoenfeld residuals on the Kaplan–Meier "
                     f"time scale (global χ²({g.df:.0f}) = {g.statistic:.2f}, p = {_p(g.p)})" if g
                     else " Proportional hazards were checked by Schoenfeld residuals")
        else:
            text += (f" Influence was checked by Cook's distance against {d.reference} ({COOK}): "
                     f"{d.flagged:,} row{'s' if d.flagged != 1 else ''} above it")
            if d.leverage_one:
                text += (f", and {d.leverage_one:,} row{'s' if d.leverage_one != 1 else ''} with "
                         f"leverage 1, which {'have' if d.leverage_one != 1 else 'has'} no distance")
        if d.status == "failed":
            from turbotab.core.estimand import ACTION_WORDS

            text += ("; the check failed for the exposure, and the response recorded is "
                     + ACTION_WORDS[d.response] + "." if d.response else
                     "; the check failed for the exposure, and no response is recorded yet.")
        else:
            text += "."
    lines = [s for s in fam.sensitivity if s.methods]
    if lines:
        names = {}
        robustness = _robustness_name(lines)
        if robustness is not None:
            names["robustness_value"] = robustness
        of = STRAIGHT_LINE if any(s.of == STRAIGHT_LINE for s in lines) else None
        e = lines[0].e_value
        text += " " + sensitivity_clause(lines[0].methods, names, of=of,
                                         sd_basis=e.sd_basis if e is not None else None)
    if artifact.multiplicity:
        text += " " + artifact.multiplicity
    return text


def _p(p: float) -> str:
    from turbotab.core.models.inference import format_p

    return format_p(p)


# ── the method contracts (BLUEPRINT §13), in the one registry (``turbotab.core.contracts``) ──

PACKAGE = "ESTIMAND"
_SENTENCE = "turbotab.core.stages.effects:methods_sentence"
# Under prediction no effect is estimated: each method is offered under inference only.
_INFERENCE_ONLY = {"inference": "recommended", "prediction": "not_offered"}
_NO_EFFECT = "Not offered: a prediction reports no effect measure."


def _option(key: str, label: str, customary: str, sound: str, rung: str) -> contracts.ContractOption:
    """An option offered under inference (with its verdict and rung) and not under prediction."""
    return contracts.ContractOption(key, label, customary,
                                    sound={"inference": sound, "prediction": _NO_EFFECT},
                                    rung={"inference": rung, "prediction": "not_offered"})


def _relation(id: str, kind: str, target: str, condition: str, says: str,
              rung: str | None = None) -> contracts.Relation:
    return contracts.Relation(kind, target, says, purposes=("inference",), rung=rung,
                              condition=condition, id=id)


def _contract(**fields: Any) -> contracts.MethodContract:
    return contracts.MethodContract(package=PACKAGE, **fields)


CONTRACTS = tuple(contracts.register_contract(c) for c in (
    _contract(
        key="effect_measure",
        label="The effect measure (difference or ratio; conditional or marginal)",
        slot="model", scope="descriptive",
        scope_note="a declaration of what the model reports: it reads the outcome's event share to "
                   "rank the options, and changes no row",
        needs=("a declared exposure", "the outcome's task"), question="estimand",
        place="MODELING_SEQUENCE §1 row 2", decision="set_estimand", leash=_INFERENCE_ONLY,
        storyboard=("the model's conditional measure", "the marginal measure averaged over the rows",
                    "which one the estimand reports"),
        sentence="turbotab.core.voice:_set_estimand",
        options=(
            _option("odds_ratio", "Conditional odds ratio",
                    "logistic regression's adjusted odds ratio, the field's norm "
                    "(MODELING_SEQUENCE review, inference row 2)",
                    "Conditional: non-collapsible, it changes with any covariate that predicts the "
                    "outcome", "available"),
            _option("risk_difference", "Marginal risk difference",
                    "uncommon outside trials (STROBE 16c asks for absolute risk)",
                    "Sound: averaged over the rows' own covariates; ranked first when the event is "
                    "common (Zhang & Yu 1998)", "available"),
            _option("risk_ratio", "Marginal risk ratio", "uncommon in nutrition cohorts",
                    "Sound: collapsible over the covariates it is standardized over", "available"),
        ),
        relations=(
            _relation("measure-labels", "implies", "the estimand card", "always",
                      "each measure is labeled difference or ratio, conditional or marginal; odds "
                      "and hazard ratios non-collapsible"),
            _relation("marginal-first", "enables", "the estimand card's order",
                      "a yes/no outcome whose event share is above 10%",
                      "the marginal risk difference and ratio rank first"),
            _relation("precision-changes-estimand", "implies", "the adjustment answers",
                      "an odds or hazard ratio and a covariate answered a cause of the outcome only",
                      "the card, the record and the caption say the conditional estimand changes"),
            _relation("measure-shown", "enables", "the effect display", "a declared measure",
                      "the caption and the effects stage report it"),
        ),
        sources=("Daniel, Zhang & Farewell 2021, Biom J 63:528", "Zhang & Yu 1998, JAMA 280:1690",
                 "VanderWeele 2019, Eur J Epidemiol 34:211")),
    _contract(
        key="g_computation", label="Marginal standardization (g-computation)",
        slot="model", scope="model",
        scope_note="row i's standardized risk depends on every analyzed row and on the outcome "
                   "through the fitted model, so it is fit on every analyzed row (inference has no "
                   "held-out fold)",
        needs=("a yes/no outcome", "the logistic model (the linear family)", "a declared exposure"),
        question="estimand", place="MODELING_SEQUENCE §1 row 2", decision="set_estimand",
        stage="effects", leash=_INFERENCE_ONLY,
        storyboard=("fit the logistic model", "set every row's exposure to each value",
                    "average the predicted risks", "difference and ratio",
                    "bootstrap the whole chain for the intervals"),
        sentence=_SENTENCE,
        options=(
            _option("standardized", "Standardized over the analyzed rows",
                    "uncommon in nutrition cohorts; the standard for an absolute risk (STROBE 16c)",
                    "Sound: the marginal risks of the declared contrast, the intervals from a "
                    "bootstrap of the whole chain", "recommended"),
        ),
        relations=(
            _relation("gcomp-bootstrap-chain", "implies", "the marginal intervals",
                      "the marginal measure declared",
                      "each resample refits the whole pipeline and the model, whole units when "
                      "rows repeat; a resample whose outcome is separated is kept at the "
                      "likelihood's limit and counted, never dropped silently"),
            _relation("gcomp-unit-floor", "disables", "the marginal intervals",
                      "rows repeat within fewer units than the unit floor",
                      "the risks are shown without an interval, with the tables' reason and "
                      "exits"),
            _relation("gcomp-survey-blocked", "conflicts", "a population survey estimand",
                      "the surveyed-population answer",
                      "blocked and recorded, exits: the sample-only answer or the conditional odds "
                      "ratio", rung="block_and_record"),
            _relation("gcomp-or-beside", "implies", "the conditional odds ratio",
                      "the marginal measure declared",
                      "the model's conditional odds ratio is reported beside it"),
        ),
        sources=("Daniel, Zhang & Farewell 2021, Biom J 63:528",)),
    _contract(
        key="model_sequence", label="The declared model sequence (Table 2)",
        slot="evaluation", scope="model",
        scope_note="each model is fit on every analyzed row, the exposure's definition fitted once",
        needs=("a declared exposure", "the adjustment answers"), question="stated",
        place="MODELING_SEQUENCE §1 row 11", decision="set_model_sequence", stage="effects",
        leash=_INFERENCE_ONLY,
        storyboard=("unadjusted", "Model 1 (age, sex, energy as declared)", "Model 2 (primary)",
                    "Model 3 (possible mediators)", "the adjustment terms set apart"),
        sentence=_SENTENCE,
        options=(
            _option("declared_sequence", "Unadjusted, Model 1, the primary and Model 3",
                    "customary: the nutrition-cohort Table 2 (Westreich & Greenland 2013)",
                    "Sound when only the exposure's rows are read as effects", "recommended"),
        ),
        relations=(
            _relation("exposure-enables-display", "enables", "the effect display",
                      "the exposure, its effect and the adjustment set answered",
                      "the fit and the effects stage show estimates; withheld before"),
            _relation("crude-always", "implies", "the unadjusted model", "always",
                      "the unadjusted estimate is shown (STROBE 16a)"),
            _relation("table2-appendix", "implies", "the adjustment terms", "always",
                      "only the exposure's rows are effects; the rest sit under \"adjustment "
                      "terms, not effect estimates\""),
            _relation("model3-labeled", "implies", "Model 3",
                      "the answers set possible mediators beside the primary",
                      "Model 3 adds them, labeled as not a total effect"),
            _relation("sequence-invalidated", "invalidates", "Model 1",
                      "another exposure, or Model 1's columns leave the primary set",
                      "Model 1 is declared again, never silently kept"),
            _relation("sequence-in-lock", "implies", "the analysis-plan lock",
                      "the plan is locked", "the declared sequence is part of the plan"),
            _relation("model3-own-rows", "implies", "Model 3's rows",
                      "complete cases, and Model 3's added columns sometimes missing",
                      "every other model is fit on every analyzed row (Model 2 is the fit's "
                      "primary); Model 3 alone on the rows where they are recorded, the primary's "
                      "adjustment refit on those rows beside it"),
            _relation("sequence-design-based", "implies", "every declared model",
                      "the surveyed-population answer",
                      "each model is design-based, or the family is blocked and recorded with the "
                      "fit's exits; never an unweighted refit beside a design-based primary"),
            _relation("sequence-supplied-copies", "disables", "the declared models",
                      "the rows are the data's own imputed copies (set_repeat_kind, kept as rows)",
                      "the fit's table pools the primary over the copies; the declared models, "
                      "the marginal risks, the diagnostics and the sensitivity analyses are "
                      "withheld with the reason, never fit on the copies stacked"),
        ),
        sources=("Westreich & Greenland 2013, Am J Epidemiol 177:292", "STROBE item 16a")),
    _contract(
        key="diagnostics", label="Diagnostics of the primary model",
        slot="evaluation", scope="descriptive",
        scope_note="reads every analyzed row to report; changes nothing until a response is "
                   "recorded",
        needs=("a fitted primary model",), question="stated", place="MODELING_SEQUENCE §1 row 11",
        decision="respond_diagnostic", stage="effects",
        leash={"inference": "block_and_record", "prediction": "not_offered"},
        storyboard=("the primary fit", "the check", "the reading", "the recorded response"),
        sentence=_SENTENCE,
        options=(
            _option("keep_labeled", "Keep the estimate, labeled with the failed check",
                    "customary: the check reported beside the estimate",
                    "Sound when the failure is stated with the estimate", "available"),
            _option("period_hazard_ratios", "The hazard ratio before and after the median event time",
                    "customary for a proportional-hazards failure (Grambsch & Therneau 1994)",
                    "Sound: the ratio is shown where it holds", "recommended"),
            _option("without_influential", "The primary refit without the influential rows, beside",
                    "customary for influential rows (Cook 1977)",
                    "Sound as a sensitivity analysis beside the primary, never in its place",
                    "available"),
        ),
        relations=(
            _relation("failed-check-recorded", "implies", "a recorded response",
                      "a failed check of the exposure",
                      "the check carries respond_diagnostic exits; the response recorded is shown "
                      "and stated, never applied silently"),
            _relation("ph-period-ratios", "enables", "the period hazard ratios",
                      "a failed proportional-hazards check, the period response recorded",
                      "the hazard ratio before and after the median event time is shown"),
            _relation("influence-refit", "enables", "the refit without influential rows",
                      "a failed influence check, that response recorded",
                      "the primary refit without the flagged rows is shown beside"),
        ),
        sources=("Grambsch & Therneau 1994, Biometrika 81:515", "Cook 1977, Technometrics 19:15")),
    _contract(
        key="unmeasured_confounding", label="Sensitivity to unmeasured confounding",
        slot="evaluation", scope="descriptive",
        scope_note="reads the reported estimate and, for the robustness value, the primary model's "
                   "rows; informs no modeling choice",
        needs=("a reported estimate",), question="stated", place="MODELING_SEQUENCE §1 row 11",
        stage="effects",
        # Offered for every inference result; the causal lane requires it (MODELING_SEQUENCE §0
        # ruling 10), which the causal contract's own relation states.
        leash={"inference": "available", "prediction": "not_offered"},
        storyboard=("the estimate", "how strong a confounder would have to be",
                    "a named measured covariate for scale"),
        sentence=_SENTENCE,
        options=(
            _option("robustness_value", "Robustness value, benchmarked",
                    "growing; anchored to the study's own covariates (Cinelli & Hazlett 2020)",
                    "Sound: bounded by named measured covariates; ranked first for a linear "
                    "outcome", "recommended"),
            _option("e_value", "E-value", "customary (growing), limited (Blum et al. 2020)",
                    "Conditional: no threshold exists; read against named confounders",
                    "available"),
        ),
        relations=(
            _relation("sensitivity-offered", "implies", "every inference result",
                      "an estimate is reported", "its sensitivity analysis is shown"),
            _relation("rv-first-linear", "implies", "the order of the analyses", "a linear outcome",
                      "the robustness value leads, the E-value follows"),
            _relation("no-threshold", "disables", "a pass/fail verdict", "always",
                      "no E-value or robustness value is called a pass or a fail"),
            _relation("rv-interval-classical", "disables", "the robustness value's interval form",
                      "the interval shown is not the classical one (HC3, cluster-robust, pooled)",
                      "RV for the interval and the benchmarks' intervals are not reported, and the "
                      "reading says why; RV_q and the adjusted estimates stay"),
            _relation("curve-straight-line", "implies", "the straight-line estimate",
                      "the exposure enters as a curve (a spline)",
                      "its straight-line estimate beside it is shown and bounded, and named so"),
        ),
        sources=("VanderWeele & Ding 2017, Ann Intern Med 167:268",
                 "Cinelli & Hazlett 2020, J R Stat Soc B 82:39")),
    _contract(
        key="exposure_family", label="An exposure family and its multiplicity",
        slot="evaluation", scope="descriptive",
        scope_note="a declaration of what is reported; the tests themselves are the feature-wise "
                   "family's, fit on every analyzed row, and their multiplicity is the "
                   "``multiplicity`` contract's",
        needs=("two or more exposures",), question="estimand", place="MODELING_SEQUENCE §1 row 2",
        decision="set_estimand", leash={"inference": "available", "prediction": "not_offered"},
        storyboard=("each exposure in turn", "the multiplicity method", "every member shown"),
        sentence="turbotab.core.voice:_set_estimand",
        options=(
            _option("family", "Each exposure in turn, every member shown",
                    "customary for metabolome-wide and nutrient-wide association studies",
                    "Sound with its multiplicity method stated; this is not selection",
                    "available"),
        ),
        relations=(
            _relation("family-implies-multiplicity", "implies", "the family's table",
                      "an exposure family declared",
                      "BH q-values or the count of tests, stated in the caption and record"),
            _relation("family-every-member", "implies", "the effect display",
                      "an exposure family declared",
                      "every member is shown, significant or not (not selection)"),
            _relation("omics-unadjusted-blocked", "conflicts", "unadjusted p-values",
                      "an omics family", "blocked and recorded; exit: Benjamini–Hochberg",
                      rung="block_and_record"),
        ),
        sources=("Benjamini & Hochberg 1995, J R Stat Soc B 57:289",
                 "Rothman 1990, Epidemiology 1:43")),
    _contract(
        key="plan_export", label="The analysis-plan export",
        slot="evaluation", scope="descriptive", scope_note="a record of the decisions; reads no row",
        needs=("an inference plan",), question="stated", place="MODELING_SEQUENCE §1 row 12",
        decision="lock_plan", leash={"inference": "available", "prediction": "not_offered"},
        storyboard=("the plan as declared", "its canonical JSON", "its SHA-256"),
        sentence="turbotab.core.plan_lock:plan_text",
        options=(
            _option("export", "The plan as declared, with its SHA-256",
                    "customary for external registration (a timestamp and a content hash)",
                    "Sound: it says what was declared in the software and when", "available"),
        ),
        relations=(
            _relation("export-replays", "implies", "the plan export", "the same decision log",
                      "the same bytes and the same hash"),
            _relation("never-preregistered", "disables",
                      "the words prespecified and preregistered", "always",
                      "the text says what was declared in the software and when"),
        ),
        sources=("Gelman & Loken 2013",)),
))
__all__ = ["EFFECTS_READS", "EffectsArtifact", "EffectsFamily", "POPULATION_SD", "SENSITIVITY_NAMES",
           "SequenceFit",
           "effects_stage", "energy_outputs", "matrix_sources", "matrix_table", "methods_sentence",
           "sensitivity_clause", "sensitivity_reading", "supplied_copies_reason"]
