"""The ``effects`` stage: the exposure's effect as the declared estimand reports it (the package
ESTIMAND; MODELING_SEQUENCE §0 rulings 9 and 10, §1 rows 2, 11 and 12).

Under inference, once the exposure, its effect and the adjustment set are answered (WP17), this
stage builds what the results show of the exposure:

* **The declared model sequence (Table 2).** MODELING_SEQUENCE §1 row 11: "a crude model, a
  declared adjustment sequence (Model 1: age, sex and energy; Model 2: plus confounders; optional
  Model 3: plus possible mediators, labeled) and the primary model, shown for the exposure only".
  The crude model is always shown (STROBE item 16a: "Give unadjusted estimates and, if applicable,
  confounder-adjusted estimates and their precision"). Model 2 is the primary: the full adjustment
  set the answers derive. Model 1 holds the columns the user declared (``set_model_sequence``;
  never read from a name). Model 3 adds the columns the answers set beside the primary (unknown
  timing, or "further adjusted for"), labeled as not a total effect. The exposure's definition is
  the same in every model: each model is the primary pipeline's own model matrix (its energy
  model, forms and fill fitted once on these rows), restricted to the exposure's columns and those
  of the model's covariates; Model 1's "energy" brings every term the energy model made of total
  energy and the other sources. Only the exposure's rows are effects; every other row is listed
  apart, under "adjustment terms, not effect estimates" (Westreich & Greenland 2013).
* **The marginal risk difference and ratio**, when the estimand declares one: standardization over
  the analyzed rows from the logistic model (``models/effects.py``), with percentile intervals from
  a bootstrap that refits the whole chain, by unit when rows repeat.
* **Diagnostics of the primary model**, reported and never acted on silently: proportional hazards
  by Schoenfeld residuals (a Cox model), leverage and Cook's distance (least squares, logistic). A
  failed check of the exposure carries its exits, each a ``respond_diagnostic`` decision; the
  recorded response is then shown beside the estimate (the hazard ratio before and after the median
  event time; the primary refit without the influential rows), or the estimate stays labeled.
* **Sensitivity to unmeasured confounding** for the primary estimate: the robustness value with
  each adjusted covariate as a named benchmark, ranked first for a linear outcome, and the E-value
  for the estimate and the confidence limit nearer the null (``models/effects.py``).

**The rows** are every analyzed row (BLUEPRINT §12 ruling 3). Under complete cases, when Model 3
adds columns that are sometimes missing, every model is fit on the rows where they are recorded, so
the models differ by their adjustment alone, and the artifact says how many. Under multiple
imputation each model is fit in each completed copy and pooled by Rubin's rules; the diagnostics
and the robustness value read the first copy, and say so.
"""
from __future__ import annotations

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
    n_boot: int = 0
    n_failed: int = 0
    by_unit: str | None = None


class Marginal(_Model):
    declared: Literal["risk_difference", "risk_ratio"]
    method: str
    contrasts: list[MarginalContrast] = []
    refused: str | None = None
    exits: list[InferenceExit] = []


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


class BenchmarkResult(_Model):
    covariate: str
    r2dxj: float
    r2yxj: float
    r2dz: float
    r2yz: float
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    kd: float


class RobustnessResult(_Model):
    exposure: str
    estimate: float
    se: float
    t: float
    dof: float
    partial_r2: float
    rv: float
    rv_alpha: float
    alpha: float
    benchmarks: list[BenchmarkResult] = []


class Sensitivity(_Model):
    feature: str
    methods: list[str]  # in rank order
    e_value: EValueResult | None = None
    robustness: RobustnessResult | None = None
    reading: str
    not_computed: str | None = None


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
                 "grain", "exposure_forms", "lens", "findings", "column_units", "split")
SEQUENCE_FAMILIES = ("linear", "proportional_odds", "cox", "featurewise")
LABELS = {"crude": "Unadjusted", "model_1": "Model 1", "model_2": "Model 2 (primary)",
          "model_3": "Model 3"}


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


# ── a table on a model matrix, for each family ───────────────────────────────


def matrix_table(family: Any, matrix: pd.DataFrame, y: Any, *, task: str, classes: Any,
                 clusters: Any, outcome: Any, survey: Any, levels: Any, event: Any,
                 features: Sequence[str]) -> Any:
    """The family's inference table on ``matrix`` (every analyzed row), or None for a family whose
    table is not made from a matrix alone (a model of the unit)."""
    from turbotab.core.models.inference import _on_rows, _on_scale

    if family.key == "featurewise":
        from turbotab.core.models.featurewise import featurewise_table

        table = featurewise_table(matrix, y, task, [f for f in features if f in matrix.columns],
                                  clusters, event)
        return _on_rows(table, len(matrix), "all")
    if family.key == "cox":
        from turbotab.core.models.survival import cox_table

        table = cox_table(matrix, y, clusters)
        return _on_rows(_on_scale(table, "time_to_event", [0, 1], outcome), len(matrix), "all")
    if family.key == "proportional_odds":
        table = family.inference_matrix(matrix, y, task=task, classes=levels, clusters=clusters,
                                        outcome=outcome)
        return _on_rows(table, len(matrix), "all")
    if family.key == "linear":
        return family.inference_matrix(matrix, y, task=task, classes=classes, clusters=clusters,
                                       outcome=outcome, rows="all", survey=survey)
    return None


# ── the stage ────────────────────────────────────────────────────────────────


def _not_applicable(state: Any, exposure: str, why: str) -> Bundle:
    from turbotab.core.models.effects import APPENDIX_TITLE

    return Bundle(data=EffectsArtifact(purpose="inference", exposure=exposure, exposures=[],
                                       appendix_title=APPENDIX_TITLE, families=[],
                                       methods=why).model_dump(mode="json"))


def effects_stage(ctx: StageContext) -> Bundle:
    from turbotab.core import estimand as est
    from turbotab.core.models import get_family
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.effects import APPENDIX_TITLE
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, design_spec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_missing_for_table, _survey, _task, coded_outcome,
                                               outcome_levels, read_assignment)
    from turbotab.core.voice import listing

    state = ctx.state
    spec_e = est.current_estimand(state)
    if state.purpose != "inference" or spec_e is None:
        why = ("Under prediction no coefficient is read as an effect, so no effect is reported."
               if state.purpose != "inference" else
               "No effect is reported until the exposure and its effect are declared.")
        return _not_applicable(state, "", why)
    key = est.exposure_key(spec_e)
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
    note = f"all {len(frame):,} analyzed rows"
    if strategy != "multiple_imputation" and extra:
        recorded = frame[extra].notna().all(axis=1)
        if not recorded.all():
            frame = frame.loc[recorded]
            note = (f"the {len(frame):,} analyzed rows with {listing(extra)} recorded, so the "
                    f"models differ by their adjustment alone")
    spec3 = (design_spec(state, frame[[*spec.inputs, *extra]],
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
    clusters = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
    survey, _ = _survey(ctx, clusters)
    largest = spec3 or spec
    missing = _missing_for_table(ctx, largest, frame[list(largest.inputs)], y, task,
                                 [f.key for f in families], loss={"n_dropped": None})
    facts = est.measure_facts(str(spec_e.measure))
    run = _Run(ctx=ctx, state=state, spec_e=spec_e, key=key, exposures=exposures, task=task,
               frame=frame, y=y, spec=spec, spec3=spec3, pipelines=pipelines, outcome=outcome,
               levels=levels, clusters=clusters, survey=survey, missing=missing,
               model_one=model_one, further=further, unit_columns=list(unit_columns))
    out = []
    for i, family in enumerate(families):
        ctx.progress(0.05 + 0.9 * i / len(families), f"{family.label}: the declared models")
        out.append(run.family_block(family, build_pipeline))
    card = est.model_sequence_card(state)
    family_spec = bool(spec_e.family)
    artifact = EffectsArtifact(
        purpose="inference", exposure=key, exposures=exposures, measure=str(spec_e.measure),
        measure_label=est.MEASURE_WORDS.get(str(spec_e.measure)), scale=facts["scale"],
        conditioning=facts["conditioning"], collapsible=facts["collapsible"], rows=note,
        appendix_title=APPENDIX_TITLE,
        multiplicity=est.multiplicity_statement(state, spec_e, len(exposures)) if family_spec else None,
        model_1=ModelOne(**card) if card is not None else None,
        families=out, methods="")
    artifact.methods = methods_sentence(state, artifact)
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


class _Run:
    """One stage run's shared inputs, and each family's block."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    # the copies the models are fit in: the completed copies under multiple imputation
    def copies(self) -> list[pd.DataFrame]:
        imputations = getattr(self.missing, "imputations", None) if self.missing else None
        if imputations is not None:
            return list(imputations.frames)
        return [self.frame]

    def family_block(self, family: Any, build_pipeline: Any) -> dict[str, Any]:
        from sklearn.base import clone

        from turbotab.core import estimand as est
        from turbotab.core.models.effects import split_rows
        from turbotab.core.models.inner_cv import fit_pipeline
        from turbotab.core.models.linear import model_matrix
        from turbotab.core.stages.modeling import _inference_table, with_units

        concerns: list[str] = []
        refusal = None
        if self.survey is not None and self.survey.refusal:
            refusal = (self.survey.refusal, self.survey.exits)
        elif self.missing is not None and self.missing.refusal:
            refusal = (self.missing.refusal, self.missing.exits)
        if refusal is not None:
            return EffectsFamily(family=family.key, label=family.label, sequence=[SequenceFit(
                key="model_2", label=LABELS["model_2"], adjusted_for=[], n_rows=len(self.frame),
                effects=None, concerns=[refusal[0]])], appendix=[],
                concerns=[refusal[0]]).model_dump(mode="json")
        design = self.survey.design if self.survey is not None else None
        if design is not None and family.key != "linear":
            concerns.append("Fit without the survey weights: these estimates describe these "
                            "participants, not the surveyed population.")
            design = None
        units = (pd.Series(self.clusters.codes, index=self.frame.index)
                 if self.clusters is not None and self.clusters.clustered else None)
        pipeline = self.pipelines[family.key]
        pipeline3 = (build_pipeline(self.spec3, family, self.task, "inference", len(self.frame),
                                    len(self.spec3.inputs) + 1) if self.spec3 is not None else None)
        predictors = list(est.predictor_roles(self.state))
        allowed_one = set(self.model_one or [])
        energy_named = bool(allowed_one & set(est.energy_terms_columns(self.state)))
        per_copy: dict[str, list[Any]] = {}
        failures: dict[str, str] = {}
        adjusted_for: dict[str, list[str]] = {}
        features: list[str] = []
        first: dict[str, Any] = {}
        supported = family.key in SEQUENCE_FAMILIES
        for k, X_k in enumerate(self.copies()):
            if self.ctx.cancelled():
                from turbotab.core.jobs import Cancelled

                raise Cancelled()
            X = X_k[list(self.spec.inputs)]
            fitted = fit_pipeline(with_units(clone(pipeline), units), X, self.y)
            if self.levels is not None:
                fitted[-1].level_names_ = list(self.levels)
            classes = list(getattr(fitted[-1], "classes_", [])) or None
            if not supported:
                table = self._table(lambda: _inference_table(
                    family, fitted, X, self.y, task=self.task, clusters=self.clusters,
                    outcome=self.outcome, rows="all"), "model_2", failures)
                per_copy.setdefault("model_2", []).append(table)
                if k == 0 and table is not None:
                    features = sorted({f for e in self.exposures for f in est.primary_features(
                        [r["feature"] for r in table.rows], e, predictors)})
                    adjusted_for["model_2"] = [c for c in self.spec.predictors
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
            kw = dict(task=self.task, classes=classes, clusters=self.clusters, outcome=self.outcome,
                      survey=design, levels=self.levels, event=self.state.event, features=feats)
            for name, cols in subsets.items():
                per_copy.setdefault(name, []).append(self._table(
                    lambda _c=cols: self._with_relative(
                        family, fitted, M[_c], matrix_table(family, M[_c], self.y, **kw), kw),
                    name, failures))
            if pipeline3 is not None:
                X3 = X_k[list(self.spec3.inputs)]
                fitted3 = fit_pipeline(with_units(clone(pipeline3), units), X3, self.y)
                if self.levels is not None:
                    fitted3[-1].level_names_ = list(self.levels)
                M3 = model_matrix(fitted3, X3)
                feats3 = [c for c in M3.columns
                          if c in {f for e in self.exposures
                                   for f in est.primary_features(M3.columns, e, predictors)}]
                kw3 = {**kw, "features": feats3}
                per_copy.setdefault("model_3", []).append(self._table(
                    lambda: self._with_relative(family, fitted3, M3,
                                                matrix_table(family, M3, self.y, **kw3), kw3),
                    "model_3", failures))
            if k == 0:
                features = feats
                first = {"fitted": fitted, "matrix": M, "sources": sources, "classes": classes}
                adjusted_for = {"crude": [], "model_2": self._raw(M.columns, feats, sources)}
                if "model_1" in subsets:
                    adjusted_for["model_1"] = self._raw(subsets["model_1"], feats, sources)
                if pipeline3 is not None:
                    adjusted_for["model_3"] = [*adjusted_for["model_2"],
                                               *[c for c in self.further
                                                 if c not in adjusted_for["model_2"]]]
        if not supported:
            concerns.append(f"{family.label} models the unit itself, so only the primary model is "
                            f"refit here: the unadjusted model and Model 1 are refit for least "
                            f"squares, logistic, proportional-odds, Cox and feature-wise models.")
        sequence, appendix = [], []
        for name in ("crude", "model_1", "model_2", "model_3"):
            tables = per_copy.get(name)
            if not tables:
                continue
            table = self._pool(tables, name) if name not in failures else None
            if table is None:
                sequence.append(SequenceFit(
                    key=name, label=LABELS[name], adjusted_for=adjusted_for.get(name, []),
                    note=self._note(name), n_rows=len(self.frame), effects=None,
                    concerns=[f"It could not be fit: {failures.get(name, 'no table')}"]))
                continue
            if self.missing is not None:
                self.missing.record(table)
            if family.key == "featurewise" and not table.info.get("refused"):
                # The family's one multiplicity method, as the fit's table carries it (MS7): with
                # unadjusted p-values recorded, no q column; every member is shown either way.
                from turbotab.core.methods.omics import apply_multiplicity, multiplicity_policy

                table = apply_multiplicity(table, multiplicity_policy(self.state))
            rows = table.rows or []
            mine = {f for e in self.exposures
                    for f in est.primary_features([r["feature"] for r in rows], e, predictors)}
            shown, terms = split_rows(rows, mine)
            sequence.append(SequenceFit(
                key=name, label=LABELS[name], adjusted_for=adjusted_for.get(name, []),
                note=self._note(name), n_rows=int(table.info.get("n_rows") or len(self.frame)),
                effects=shown if table.rows else None,
                inference=table.info, concerns=list(table.concerns)))
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
            try:
                out.sensitivity = self.sensitivity(family, first, features, primary, out.marginal)
            except Exception as exc:  # noqa: BLE001
                self._unless_cancelled()
                out.sensitivity = [Sensitivity(feature=", ".join(features), methods=[], reading="",
                                               not_computed=f"Not computed: {exc}")]
        return out.model_dump(mode="json")

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

    def _with_relative(self, family: Any, fitted: Any, matrix: pd.DataFrame, table: Any,
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
            table=lambda m: matrix_table(family, m, self.y, **kw).rows, coefficients=table.rows)
        table.rows = [*table.rows, *found]
        return table

    def _raw(self, columns: Sequence[str], feats: Sequence[str],
             sources: Mapping[str, tuple[set[str], set[str]]]) -> list[str]:
        """The raw columns a model's matrix columns come from, the exposure's aside."""
        found: set[str] = set()
        exposures = set(self.exposures)
        for c in columns:
            if c in feats:
                continue
            found |= sources.get(str(c), ({str(c)}, set()))[0] - exposures
        order = [*self.spec.inputs, *sorted(found)]
        return [c for c in dict.fromkeys(order) if c in found]

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
        return f"further adjusted for {listing(self.further)}, {what}: not a total effect."

    def _pool(self, tables: Sequence[Any], name: str) -> Any:
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

        imputations = self.missing.imputations
        pooled = pool_rows([t.rows for t in tables])
        info = dict(tables[0].info)
        info["caption"] = (f"Multiple imputation, m = {len(tables)}: each completed copy analyzed "
                           f"as follows, then pooled by Rubin's rules. {info.get('caption', '')}"
                           ).strip()
        info["missing"] = multiple_imputation_info(imputations, self.spec3 or self.spec, len(self.y))
        concerns = list(tables[0].concerns) + mi_concerns(imputations, pooled, len(self.y))
        return InferenceTable(pooled, info, concerns)

    # ── the marginal risk difference and ratio ──

    def marginal(self, family: Any, pipeline: Any) -> Marginal | None:
        from sklearn.base import clone

        from turbotab.core import estimand as est
        from turbotab.core.models import effects
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
                    fit, X, self.y.astype(float), exposure, setting, n_boot=effects.BOOT,
                    seed=seed, units=units,
                    unit_column=self.clusters.column if units is not None else None,
                    energy=energy, progress=progress, cancelled=self.ctx.cancelled)
            except effects.Unestimable as exc:
                out.refused = f"The marginal risks cannot be standardized: {exc}."
                return out
            out.contrasts.append(MarginalContrast(**{k: v for k, v in found.as_dict().items()}))
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
        flagged = np.flatnonzero(D > threshold)
        method = (f"Leverage and Cook's distance of every row, as R's hatvalues and "
                  f"cooks.distance give them, against the median of F({p}, {n - p:,}) = "
                  f"{threshold:.3g} ({effects.COOK}){on}")
        largest = float(np.nanmax(D)) if len(D) else None
        base = dict(check="influence", method=method, threshold=threshold,
                    reference=f"the median of F({p}, {n - p:,})", flagged=int(len(flagged)),
                    largest=largest, n=n)
        if not len(flagged):
            return Diagnostic(status="passed", tests=[], **base,
                              reading=(f"No row moves the estimates past Cook's reference "
                                       f"(largest distance {largest:.3g}, against {threshold:.3g})."))
        reading = (f"{len(flagged):,} row{'s' if len(flagged) != 1 else ''} move the estimates past "
                   f"Cook's reference (largest distance {largest:.3g}, against {threshold:.3g}).")
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
                    primary: SequenceFit | None, marginal: Marginal | None) -> list[Sensitivity]:
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
        if family.key != "featurewise" and len(features) > 1:
            return [Sensitivity(feature=", ".join(features), methods=[], reading="",
                                not_computed="The exposure enters as several terms (a curve or "
                                             "categories), so no single coefficient carries the "
                                             "effect for an E-value or a robustness value.")]
        out: list[Sensitivity] = []
        M, sources = first["matrix"], first["sources"]
        imputed = getattr(self.missing, "imputations", None) is not None
        for row in rows:
            out.append(self._one(family, row, M, sources, marginal, imputed))
        return out

    def _one(self, family: Any, row: Coefficient, M: pd.DataFrame, sources: Mapping[str, Any],
             marginal: Marginal | None, imputed: bool) -> Sensitivity:
        from turbotab.core.models import effects

        feature = str(row.feature)
        if self.task == "regression":
            sd = float(np.std(self.y.astype(float), ddof=1))
            matrix = y = None
            benchmarks = None
            reason = None
            if family.key == "featurewise":
                reason = ("benchmarks need the covariates' own coefficients, which the feature-wise "
                          "tests project out")
            elif self.survey is not None and self.survey.design is not None:
                reason = "the robustness value is defined for an unweighted least-squares fit"
            elif feature not in M.columns:
                reason = ("the average relative effect is a contrast of several coefficients, not "
                          "one regressor's")
            else:
                matrix, y = M, self.y.astype(float)
                benchmarks = self._benchmarks(M, sources, [feature])
            found = effects.unmeasured_confounding(
                measure="mean_difference", estimate=float(row.estimate), ci_low=row.ci_low,
                ci_high=row.ci_high, se=row.se, outcome_sd=sd, matrix=matrix, y=y,
                exposure_column=feature if matrix is not None else None, benchmarks=benchmarks)
            if family.key == "featurewise" and row.df and row.se:
                t = float(row.estimate) / float(row.se)
                rv = effects.robustness_value(t, float(row.df))
                rva = effects.robustness_value(t, float(row.df), alpha=0.05)
                found["robustness"] = effects.LinearSensitivity(
                    exposure=feature, estimate=float(row.estimate), se=float(row.se), t=t,
                    dof=float(row.df), partial_r2=effects.partial_r2(t, float(row.df)), rv=rv,
                    rv_alpha=rva).as_dict()
                found["methods"] = ["robustness_value", *found["methods"]]
            return self._sensitivity(feature, found, imputed, reason)
        if family.key == "featurewise":
            return Sensitivity(feature=feature, methods=[], reading="",
                               not_computed="The feature-wise design models each exposure on the "
                                            "outcome, so the exposure is no regressor whose "
                                            "confounding these analyses bound.")
        share = (float(np.mean(self.y["event"])) if self.task == "time_to_event"
                 else float(np.mean(self.y.astype(float))))
        measure = "hazard_ratio" if self.task == "time_to_event" else "odds_ratio"
        if marginal is not None and marginal.contrasts and marginal.contrasts[0].rr is not None:
            c = marginal.contrasts[0]
            found = effects.unmeasured_confounding(measure="risk_ratio", estimate=c.rr,
                                                   ci_low=c.rr_low, ci_high=c.rr_high)
            return self._sensitivity(feature, found, imputed, None, what="the marginal risk ratio")
        found = effects.unmeasured_confounding(measure=measure, estimate=float(row.ratio),
                                               ci_low=row.ratio_low, ci_high=row.ratio_high,
                                               outcome_share=share)
        return self._sensitivity(feature, found, imputed, None)

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
            reading=" ".join(parts), not_computed=(f"No robustness value: {reason}." if reason
                                                   else None))


def sensitivity_reading(found: Mapping[str, Any], feature: str, target: str,
                        what: str = "the estimate") -> str:
    """What one sensitivity analysis says (``models.effects.unmeasured_confounding``'s result), in
    the words every inference result and the causal lane share: the robustness value with the
    benchmark that moves the estimate most, then the E-value, never a pass or a fail."""
    from turbotab.core.models import effects
    from turbotab.core.voice import tick

    e = found.get("e_value")
    r = found.get("robustness")
    parts = []
    if r is not None:
        bench = r.get("benchmarks") or []
        lead = (f"An unmeasured confounder would need a partial R² of {r['rv']:.1%} with both "
                f"{tick(feature)} and {tick(target)}, beyond the measured covariates, to bring the "
                f"estimate to zero, and {r['rv_alpha']:.1%} to bring its 95% interval to "
                f"include zero ({effects.CINELLI_HAZLETT})")
        if bench:
            b = max(bench, key=lambda x: abs(r["estimate"] - x["estimate"]))
            lead += (f"; one as strong as {tick(b['covariate'])} would move it to "
                     f"{b['estimate']:.4g} (95% CI {b['ci_low']:.4g} to {b['ci_high']:.4g})")
        parts.append(lead + ".")
    if e is not None:
        limit = ("1, as its interval includes the null" if e.get("interval_includes_null")
                 else f"{e['limit']:.2f}" if e.get("limit") is not None else "not computed")
        parts.append(f"E-value for {what}: {e['point']:.2f}, and for the confidence limit "
                     f"nearer the null {limit} ({effects.VANDERWEELE_DING}); an E-value is read "
                     f"against the confounders one can name, never as a pass or a fail.")
    return " ".join(parts)


SENSITIVITY_NAMES = {
    "robustness_value": "the Cinelli–Hazlett robustness value (each adjusted covariate a named "
                        "benchmark)",
    "e_value": "the E-value for the estimate and for the confidence limit nearer the null"}


def sensitivity_clause(methods: Sequence[str], names: Mapping[str, str] | None = None) -> str:
    """The methods text's sentence for the analyses that ran, in rank order."""
    words = {**SENSITIVITY_NAMES, **(names or {})}
    return (f"Sensitivity to unmeasured confounding is reported by "
            f"{' and by '.join(words[m] for m in methods)}, never as a pass or a fail.")


# ── the methods sentence ─────────────────────────────────────────────────────


def methods_sentence(state: Any, artifact: EffectsArtifact) -> str:
    """The methods text this stage writes, from what it fitted: the sequence, the display rule, the
    marginal standardization, the diagnostics and their responses, and the sensitivity analyses."""
    from turbotab.core.models.effects import COOK, WESTREICH
    from turbotab.core.voice import listing, tick

    if not artifact.families:
        return artifact.methods
    whose = ("each exposure" if artifact.exposure == "*" else tick(artifact.exposure))
    fam = artifact.families[0]
    keys = [s.key for s in fam.sequence]
    seq = ["unadjusted"]
    if "model_1" in keys:
        one = next(s for s in fam.sequence if s.key == "model_1")
        seq.append(f"Model 1, adjusted for {listing(one.adjusted_for) if one.adjusted_for else 'nothing besides it'}")
    two = next((s for s in fam.sequence if s.key == "model_2"), None)
    seq.append("Model 2, the primary, adjusted for "
               + (listing(two.adjusted_for, limit=8) if two is not None and two.adjusted_for
                  else "nothing besides it"))
    if "model_3" in keys:
        three = next(s for s in fam.sequence if s.key == "model_3")
        added = [c for c in three.adjusted_for
                 if c not in next(s for s in fam.sequence if s.key == "model_2").adjusted_for]
        what = "a possible mediator" if len(added) == 1 else "possible mediators"
        seq.append(f"Model 3, further adjusted for {listing(added)}, {what}, so not a total effect")
    text = (f"The estimate of {whose} is reported across a declared sequence of models fit on "
            f"{artifact.rows}: {'; '.join(seq)}. Only the exposure's estimates are shown as "
            f"effects; every other coefficient is listed apart as an adjustment term, not an "
            f"effect estimate ({WESTREICH}).")
    if fam.marginal is not None and not fam.marginal.refused and fam.marginal.contrasts:
        c = fam.marginal.contrasts[0]
        text += (f" The marginal risk difference and risk ratio ({c.setting}) were estimated by "
                 f"standardization over the analyzed rows (g-computation) from the logistic model, "
                 f"with 95% percentile intervals from {c.n_boot:,} bootstrap resamples refitting "
                 f"the whole chain" + (f" by `{c.by_unit}`" if c.by_unit else "") + ".")
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
        if d.status == "failed":
            from turbotab.core.estimand import ACTION_WORDS

            text += ("; the check failed for the exposure, and the response recorded is "
                     + ACTION_WORDS[d.response] + "." if d.response else
                     "; the check failed for the exposure, and no response is recorded yet.")
        else:
            text += "."
    lines = [s for s in fam.sensitivity if s.methods]
    if lines:
        text += " " + sensitivity_clause(lines[0].methods)
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
                      "rows repeat"),
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
__all__ = ["EFFECTS_READS", "EffectsArtifact", "EffectsFamily", "SENSITIVITY_NAMES", "SequenceFit",
           "effects_stage", "energy_outputs", "matrix_sources", "matrix_table", "methods_sentence",
           "sensitivity_clause", "sensitivity_reading"]
