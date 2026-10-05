"""The causal lane's stages (``turbotab/core/causal.py``; V2 definition of done §2).

* ``causal_design`` (outcome-free) is the card the causal question shows before any choice: the
  options ranked for the declared plan, the four assumptions each with its diagnostic, and
  practical positivity read from a cross-fitted main-terms propensity (a yes/no exposure) or the
  share of a numeric exposure's variance the covariates explain. It reads the outcome's
  missingness and level counts, never its relationship to anything (Rubin 2008, *Ann Appl Stat*
  2:808: "design trumps analysis").
* ``causal`` runs the chosen estimator once the answer and the model families are recorded (the
  primary model and the lane declared together before either estimate is shown). The assumptions
  come first in
  its artifact; the estimate is withheld, with the reason and the ways forward, when positivity is
  practically violated by the estimator's own propensities, when the survey answer is unanswered
  or blocks the method, when the clusters are too few, or when incomplete rows wait for their
  answer (block and record).

**The data** are the declared plan's: every analyzed row (BLUEPRINT §12 ruling 3), the exposure and
the adjustment set the answers keep, through the shared steps every family's pipeline starts with
(the energy model, the declared codes, one-hot encoding; ``models/pipeline.py``), fit on the
analyzed rows, outcome-free. The exposure is the matrix column the lineage traces to it; the rest
of the matrix is the adjustment set. Rows missing any of them, or the outcome, are left out with
their count stated (the lane is not pooled over multiple imputations in v2).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.custom_sound import LabeledOption
from turbotab.core.graph import Bundle, StageContext

DEFAULT_FLEXIBLE_AT = 500  # rows from which the default nuisance learner is a random forest


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class OverlapView(_Model):
    model: str  # which propensity model the overlap reads
    bins: list[float]
    exposed: list[float]
    unexposed: list[float]
    bound: float
    n_outside: int
    share_outside: float
    extreme_weight_share: float
    max_weight: float
    ess_exposed: float
    ess_unexposed: float
    n_exposed: int
    n_unexposed: int
    min_exposed: float
    max_unexposed: float
    violated: bool


class VariationView(_Model):
    model: str
    r2: float
    residual_sd: float
    violated: bool


class AssumptionView(_Model):
    key: str
    label: str
    statement: str
    diagnostic: str
    status: Literal["untestable", "checked", "violated", "stated", "pending"]
    source: str


class CausalExit(_Model):
    label: str
    decision: dict[str, Any] | None = None


class PositivityView(_Model):
    violated: bool
    reason: str | None = None


class MissingView(_Model):
    n_rows: int  # every analyzed row
    n_complete: int
    n_incomplete: int
    answer: str | None
    blocked: bool
    reason: str | None = None


class SurveyView(_Model):
    population: bool  # the surveyed-population answer: the weights enter every fit
    weight: str | None = None


class CausalDesignArtifact(_Model):
    purpose: Literal["inference"]
    offered: bool
    reason: str | None = None  # why the lane is not offered on this plan
    stated: str | None = None  # the Router's stated skip (the lane one step away)
    exposure: str
    exposure_kind: Literal["binary", "continuous"] | None = None
    level: str | None = None  # a yes/no exposure's level coded 1
    task: str | None = None
    n: int = 0  # complete rows
    n_limit: int = 0  # the limiting sample size (rows, or the rarer outcome level)
    candidates: list[str] = []  # the adjustment set's model-matrix columns
    many: bool = False
    options: list[LabeledOption] = []
    recommended: str | None = None
    assumptions: list[AssumptionView] = []
    overlap: OverlapView | None = None
    variation: VariationView | None = None
    positivity: PositivityView = PositivityView(violated=False)
    missing: MissingView | None = None
    survey: SurveyView = SurveyView(population=False)
    leash: str = ""


class CausalEstimate(_Model):
    label: str  # "Average effect", "Effect among the exposed", "Effect per unit", …
    measure: str
    estimate: float | None
    se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    df: int | None = None
    scale: Literal["difference", "ratio"] = "difference"


class CausalSelection(_Model):
    """What the post-double-selection lassos chose, by raw column, and the model terms kept."""

    outcome: list[str]
    exposure: list[str]
    union: list[str]
    terms: list[str] = []


class CausalArtifact(_Model):
    purpose: Literal["inference"]
    exposure: str
    method: str
    method_label: str
    learner: str | None = None
    population: str = "all"
    # The assumptions and the diagnostics come first: they are shown before any estimate.
    assumptions: list[AssumptionView] = []
    declared: list[str] = []
    overlap: OverlapView | None = None
    variation: VariationView | None = None
    withheld: str | None = None
    exits: list[CausalExit] = []
    estimates: list[CausalEstimate] = []
    repetitions: list[dict[str, float]] = []
    n: int = 0
    n_trimmed: int = 0
    selected: CausalSelection | None = None
    sensitivity: dict[str, Any] | None = None
    methods: str = ""


# ── the plan's data ──────────────────────────────────────────────────────────


@dataclass
class Prepared:
    """The declared plan's rows and columns, outcome coded, ready for an estimator."""

    exposure: str
    task: str
    row_ids: np.ndarray
    y: np.ndarray
    d: np.ndarray
    X: np.ndarray
    x_names: list[str]
    exposure_kind: str
    level: str | None
    adjusted: list[str]  # the raw covariates the answers adjust for
    timing_unknown: list[str]
    missing: MissingView
    frame: pd.DataFrame  # the complete rows' raw columns (row id index), incl. cluster columns
    unit_columns: list[str] = field(default_factory=list)
    x_sources: dict[str, list[str]] = field(default_factory=dict)  # matrix column -> raw columns

    def raw(self, columns: Sequence[str]) -> list[str]:
        """The raw columns behind some matrix columns, in the adjustment set's order."""
        behind = {s for c in columns for s in self.x_sources.get(c, [c])}
        return [c for c in self.adjusted if c in behind]


class Withheld(Exception):
    """The lane cannot estimate on this plan yet: the reason, and the ways forward."""

    def __init__(self, reason: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(reason)
        self.reason = reason
        self.exits = [_exit(e) for e in exits]


def _exit(e: Mapping[str, Any]) -> dict[str, Any]:
    decision = e.get("decision")
    if decision is not None and hasattr(decision, "model_dump"):
        decision = decision.model_dump(mode="json")
    return {"label": str(e.get("label")), "decision": decision}


def _raw_sources(steps: Sequence[tuple[str, Any]], inputs: Sequence[str]) -> dict[str, list[str]]:
    """Each matrix column's raw source columns, traced through the fitted steps."""
    from turbotab.core.models.lineage import trace_step

    current: dict[str, list[str]] = {c: [c] for c in inputs}
    for _, step in steps:
        nxt: dict[str, list[str]] = {}
        for out, (parents, _op, _formula) in trace_step(step, list(current)).items():
            nxt[out] = list(dict.fromkeys(s for p in parents for s in current.get(p, [p])))
        current = nxt
    return current


def _exposure_column(sources: Mapping[str, Sequence[str]], exposure: str) -> str:
    from turbotab.core.estimand import primary_features

    found = [c for c, src in sources.items() if exposure in src]
    if len(found) > 1:
        own = primary_features(found, exposure)
        found = own if len(own) == 1 else found
    if len(found) != 1:
        raise Withheld(
            f"`{exposure}` enters the model as {len(found)} columns "
            f"({', '.join(f'`{c}`' for c in found[:4])}), and the causal lane estimates one effect "
            f"of one column: a numeric exposure per unit, or a yes/no exposure's two levels.",
            [{"label": "Keep the primary model only", "decision": None}])
    return found[0]


def prepare(ctx: StageContext) -> Prepared:
    """The declared plan's data for the lane (see the module docstring)."""
    from turbotab.core.decisions import left_out
    from turbotab.core.estimand import current_estimand, derived_roles
    from turbotab.core.models.inference import cluster_columns
    from turbotab.core.models.pipeline import (design_spec, input_columns, model_predictors,
                                               modeling_frame, shared_steps, transformer)
    from turbotab.core.readings import Unsettled, predictors_or_ask
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import _task, coded_outcome, read_assignment

    state = ctx.state
    spec = current_estimand(state)
    exposure = str(spec.exposure)
    task = _task(ctx)
    target = state.target
    try:
        with open_store(ctx) as store:
            summaries = {c.name: {"dtype": c.dtype, "n_unique": c.n_unique}
                         for c in store.info().columns}
            predictors_or_ask(state, summaries, drop=left_out(state), store=store)
    except Unsettled as exc:
        raise Withheld(str(exc), exc.exits) from exc
    predictors = model_predictors(state)
    if exposure not in predictors:
        raise Withheld(f"`{exposure}` is not among the model's predictors as the plan stands.")
    assignment = read_assignment(ctx.inputs["split"])
    rows = assignment.index.to_numpy()  # every analyzed row (ruling 3)
    adj = state.energy_adjustment
    inputs = input_columns(predictors, adj)
    with open_store(ctx) as store:
        units = cluster_columns(state, store.columns)
        info = {c.name: c for c in store.info().columns}
        columns = list(dict.fromkeys([*inputs, target, *units]))
        frame = modeling_frame(store, columns, rows, outcome=target)
    dspec = design_spec(state, frame[inputs], predictors, column_info=info)
    dspec = dataclasses.replace(dspec, impute=False, indicators=False, energy_fill=None)
    levels = set(dspec.levels)
    needed = [c for c in dspec.inputs if c not in levels]
    complete = frame[[*needed, target]].notna().all(axis=1).to_numpy()
    strategy = state.missing.strategy if state.missing is not None else None
    n_rows, n_complete = int(len(frame)), int(complete.sum())
    blocked = n_complete < n_rows and strategy in ("impute", "multiple_imputation")
    reason = None
    if n_complete < n_rows:
        lost = n_rows - n_complete
        if blocked:
            reason = (f"{lost:,} of the {n_rows:,} analyzed rows miss the outcome, the exposure or "
                      f"an adjusted covariate. The missing-values answer "
                      f"({str(strategy).replace('_', ' ')}) would need every causal estimate pooled "
                      f"over imputations, which the causal lane does not do in v2; it can run on "
                      f"the {n_complete:,} complete rows, assuming they represent the rest.")
        else:
            reason = (f"{lost:,} of the {n_rows:,} analyzed rows miss the outcome, the exposure or "
                      f"an adjusted covariate, so the lane runs on the {n_complete:,} complete "
                      f"rows, as complete cases do.")
    missing = MissingView(n_rows=n_rows, n_complete=n_complete, n_incomplete=n_rows - n_complete,
                          answer=strategy, blocked=blocked, reason=reason)
    frame = frame.loc[complete]
    if len(frame) < 20:
        raise Withheld(f"Only {len(frame):,} complete rows: too few to cross-fit anything.")
    shared = transformer(shared_steps(dspec))
    matrix = shared.fit_transform(frame[dspec.inputs])
    sources = _raw_sources(shared.steps, dspec.inputs)
    xcol = _exposure_column({c: sources.get(c, [c]) for c in matrix.columns}, exposure)
    d = pd.to_numeric(matrix[xcol], errors="coerce").to_numpy(dtype=float)
    X = matrix.drop(columns=[xcol])
    keep = [c for c in X.columns if np.nanstd(pd.to_numeric(X[c], errors="coerce")) > 0]
    X = X[keep].apply(pd.to_numeric, errors="coerce")
    values = np.unique(d[np.isfinite(d)])
    level = None
    if len(values) == 2:
        kind = "binary"
        lo, hi = float(values[0]), float(values[1])
        if (lo, hi) != (0.0, 1.0):
            d = (d == hi).astype(float)
        level = (xcol[len(exposure) + 1:] if xcol.startswith(f"{exposure}_") and xcol != exposure
                 else _plain(hi))
    elif len(values) < 2:
        raise Withheld(f"`{exposure}` takes one value on the complete rows, so it has no effect to "
                       f"estimate.")
    else:
        kind = "continuous"
    y = np.asarray(coded_outcome(task, frame[target].to_numpy(), state.event), dtype=float)
    # What the lane adjusts for: the raw columns behind the matrix's other columns (the answers'
    # adjusted covariates, total energy as the energy model keeps it, a grouping's fixed effects).
    behind = list(dict.fromkeys(s for col in X.columns for s in sources.get(col, [col])))
    adjusted = ([c for c in predictors if c in behind and c != exposure]
                + [c for c in behind if c not in predictors and c != exposure])
    unknown = [c for c, r in derived_roles(state).items() if r.role == "timing_unknown"]
    return Prepared(exposure=exposure, task=task, row_ids=frame.index.to_numpy(), y=y, d=d,
                    X=X.to_numpy(dtype=float), x_names=list(X.columns), exposure_kind=kind,
                    level=level, adjusted=adjusted, timing_unknown=unknown, missing=missing,
                    x_sources={c: list(sources.get(c, [c])) for c in X.columns},
                    frame=frame, unit_columns=list(units))


def _plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def n_limit(task: str, y: np.ndarray) -> int:
    """Harrell's limiting sample size: the rows (a numeric outcome) or the rarer level's count."""
    if task == "binary":
        ones = int(np.sum(y == 1))
        return min(ones, len(y) - ones)
    return int(len(y))


def _design(ctx: StageContext, prep: Prepared, *,
            sample_only: bool = False) -> tuple[Any, str | None, str | None]:
    """The variance design for the lane's rows: ``(Design | None, weight column, cluster
    column)``; raises :class:`Withheld` when the survey answer or the clusters stop it."""
    from turbotab.core.models.causal import Design
    from turbotab.core.models.inference import floor_refusal, resolve_clusters
    from turbotab.core.stages.modeling import _survey

    state = ctx.state
    clusters = (resolve_clusters(state, prep.frame[prep.unit_columns])
                if prep.unit_columns else None)
    survey, _ = _survey(ctx, clusters)
    if survey is not None and survey.refusal and not sample_only:
        raise Withheld(survey.refusal, survey.exits)
    if survey is not None and survey.answer == "population" and not sample_only:
        design = survey.design
        position = pd.Series(np.arange(len(design.row_ids)), index=design.row_ids)
        placed = np.isin(prep.row_ids, design.row_ids)
        if not placed.all():
            raise Withheld(f"{int((~placed).sum()):,} complete rows have no stratum, PSU or weight, "
                           f"so they are outside the surveyed population's design.")
        at = position.loc[prep.row_ids].to_numpy()
        weight = design.weight[at]
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
            raise Withheld("Some complete rows have no positive survey weight; the population "
                           "estimate needs one on every row it reads.")
        return (Design(weight=weight, stratum=design.stratum[at], psu=design.psu[at],
                       label="survey", full=design, positions=at), design.weight_column, None)
    if clusters is not None:
        refused = floor_refusal(clusters, prep.task)
        if refused is not None:
            raise Withheld(refused[0], refused[1])
        if clusters.clustered:
            return Design(psu=np.asarray(clusters.codes), label="clusters"), None, clusters.column
    return None, None, None


# ── causal_design ────────────────────────────────────────────────────────────


def _not_offered(exposure: str, reason: str) -> Bundle:
    return Bundle(data=CausalDesignArtifact(purpose="inference", offered=False, reason=reason,
                                            exposure=exposure).model_dump(mode="json"))


def causal_design_stage(ctx: StageContext) -> Bundle:
    """The causal question's card. A failure to read the plan's data is the card's own reason (the
    lane not offered, with why), so the Router never holds the primary analysis on the lane."""
    from turbotab.core.jobs import Cancelled

    try:
        return _causal_design(ctx)
    except Cancelled:
        raise
    except Withheld as exc:
        return _not_offered(_exposure_of(ctx.state), exc.reason)
    except Exception as exc:  # noqa: BLE001 - the lane is optional; its failure is stated
        return _not_offered(_exposure_of(ctx.state),
                            f"The causal lane could not read the declared plan's data: {exc}")


def _exposure_of(state: Any) -> str:
    from turbotab.core.estimand import current_estimand

    return str(getattr(current_estimand(state), "exposure", "") or "")


def _causal_design(ctx: StageContext) -> Bundle:
    from turbotab.core import causal
    from turbotab.core.estimand import adjustment_answer, current_estimand
    from turbotab.core.models import causal as est

    state = ctx.state
    spec = current_estimand(state)
    exposure = str(getattr(spec, "exposure", "") or "")
    reason = causal.plan_reason(state)
    if state.purpose != "inference":
        reason = reason or "The causal lane is asked under inference."
    if reason is None and spec is None:
        reason = "Declare the exposure and its effect first; the causal lane estimates that effect."
    if reason is None and adjustment_answer(state) is None:
        reason = "Answer the adjustment set first; the causal lane adjusts for what it keeps."
    if reason is not None:
        return _not_offered(exposure, reason)
    ctx.progress(0.1, "Reading the declared plan's rows")
    try:
        prep = prepare(ctx)
    except Withheld as exc:
        return _not_offered(exposure, exc.reason)
    population = bool(state.survey is not None and state.survey.estimand == "population")
    try:
        design, weight_column, _ = _design(ctx, prep)
    except Withheld:
        design, weight_column = None, None  # the lane's own stage says why; the card still shows
    weights = None if design is None else design.weight
    seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0
    groups = None if design is None else design.groups
    ctx.progress(0.4, "Reading positivity from the exposure and the covariates")
    try:
        splits = est.sample_splits(len(prep.y), 5, 1, seed=seed, groups=groups)
    except ValueError as exc:
        return _not_offered(exposure, str(exc))
    overlap_view = variation_view = None
    positivity = PositivityView(violated=False)
    X = prep.X if prep.X.shape[1] else np.zeros((len(prep.y), 1))
    if prep.exposure_kind == "binary":
        [g] = est.cross_fit(est.learner_factory("linear"), True, X, prep.d, splits[0], weights, seed)
        found = est.overlap(g, prep.d, est.tmle_gbound(len(prep.y)), weights)
        overlap_view = OverlapView(model="main-terms logistic regression, cross-fitted over 5 folds",
                                   **dataclasses.asdict(found))
        if found.violated:
            positivity = PositivityView(violated=True, reason=_positivity_reason(prep, found))
    else:
        [m] = est.cross_fit(est.learner_factory("linear"), False, X, prep.d, splits[0], weights, seed)
        found_v = est.variation(prep.d, m, weights)
        variation_view = VariationView(model="main-terms linear regression, cross-fitted over 5 folds",
                                       **dataclasses.asdict(found_v))
        if found_v.violated:
            positivity = PositivityView(violated=True, reason=_variation_reason(prep, found_v))
    limit = n_limit(prep.task, prep.y)
    many = causal.many_candidates(limit, len(prep.x_names))
    opts = causal.options(prep.exposure_kind, prep.task, many, population)
    assumptions = causal.assumption_card(
        exposure=prep.exposure, exposure_kind=prep.exposure_kind, adjusted=prep.adjusted,
        timing_unknown=prep.timing_unknown,
        overlap=dataclasses.asdict(found) if prep.exposure_kind == "binary" else None,
        variation=dataclasses.asdict(found_v) if prep.exposure_kind != "binary" else None,
        contrast=getattr(spec, "contrast", None), level=prep.level)
    first = opts[0].key if opts else None
    stated = causal.stated_reason(first, many, len(prep.x_names), limit)
    artifact = CausalDesignArtifact(
        purpose="inference", offered=True, stated=stated, exposure=prep.exposure,
        exposure_kind=prep.exposure_kind, level=prep.level, task=prep.task, n=int(len(prep.y)),
        n_limit=limit, candidates=prep.x_names, many=many, options=opts, recommended=first,
        assumptions=[AssumptionView(**a) for a in assumptions], overlap=overlap_view,
        variation=variation_view, positivity=positivity, missing=prep.missing,
        survey=SurveyView(population=population, weight=weight_column),
        leash=("Offered under inference after the exposure, its effect and the adjustment set are "
               "declared; the assumptions are declared before any estimate."))
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


def _positivity_reason(prep: Prepared, found: Any) -> str:
    b = found.bound
    return (f"{found.n_outside:,} of {len(prep.y):,} rows ({found.share_outside:.1%}, at or above "
            f"the lane's line of 1%) have a propensity outside [{b:.4f}, {1 - b:.4f}], the bound "
            f"the estimator caps at: for them one level of `{prep.exposure}` is all but "
            f"impossible given the covariates, so their weights are capped and the estimate leans "
            f"on extrapolation there (practical positivity violation; Hernán & Robins 2020, §3.3). "
            f"Trim to the rows where both levels are plausible, which changes the estimand to "
            f"them, or keep every row and record it.")


def _variation_reason(prep: Prepared, found: Any) -> str:
    return (f"The covariates explain {found.r2:.0%} of `{prep.exposure}`'s variance (at least "
            f"{0.9:.0%}, a variance inflation factor of 10 or more): the effect rests on little of "
            f"the exposure's own variation, and the estimate extrapolates where it does not vary. "
            f"Keep every row and record it, or revisit the adjustment set.")


# ── causal ───────────────────────────────────────────────────────────────────


def _withheld(base: dict[str, Any], reason: str, exits: Sequence[Mapping[str, Any]] = ()) -> Bundle:
    artifact = CausalArtifact(**{**base, "withheld": reason,
                                 "exits": [CausalExit(**_exit(e)) for e in exits], "estimates": []})
    return Bundle(data=artifact.model_dump(mode="json"))


def default_learner(n: int) -> str:
    """The default nuisance learner: a random forest from 500 rows, else the cross-validated
    lasso (a random forest's propensities and predictions are noisy on fewer rows)."""
    return "random_forest" if n >= DEFAULT_FLEXIBLE_AT else "lasso"


def causal_stage(ctx: StageContext) -> Bundle:
    from turbotab.core import causal
    from turbotab.core.decisions import SetCausal
    from turbotab.core.estimand import current_estimand
    from turbotab.core.models import causal as est

    state = ctx.state
    spec = causal.current_causal(state)
    exposure = str(getattr(spec, "exposure", "") or getattr(state.causal, "exposure", "") or "")
    method = str(getattr(spec, "method", "none") or "none")
    base: dict[str, Any] = {"purpose": "inference", "exposure": exposure, "method": method,
                            "method_label": causal.METHOD_LABELS.get(method, method)}
    if spec is None:
        reason = causal.plan_reason(state) or (
            "The causal answer was given for another exposure, or before the plan; it is re-asked, "
            "never kept.")
        return _withheld(base, reason)
    if method == "none":
        return Bundle(data=CausalArtifact(**base).model_dump(mode="json"))
    decision = SetCausal(**spec.model_dump())
    base["declared"] = list(spec.assumptions)
    base["population"] = spec.population
    missing = [a for a in causal.ASSUMPTIONS if a not in spec.assumptions]
    if missing:
        return _withheld(base, "The assumptions are declared before any estimate: "
                               + ", ".join(causal.ASSUMPTION_WORDS[a] for a in missing) + ".",
                         [{"label": "Declare these assumptions", "decision": decision.model_copy(
                             update={"assumptions": list(causal.ASSUMPTIONS)})}])
    ctx.progress(0.05, "Reading the declared plan's rows")
    try:
        prep = prepare(ctx)
    except Withheld as exc:
        return _withheld(base, exc.reason, exc.exits)
    estimand = current_estimand(state)
    binary = prep.exposure_kind == "binary"

    def card(found: Any = None, found_v: Any = None) -> list[AssumptionView]:
        return [AssumptionView(**a) for a in causal.assumption_card(
            exposure=prep.exposure, exposure_kind=prep.exposure_kind, adjusted=prep.adjusted,
            timing_unknown=prep.timing_unknown,
            overlap=dataclasses.asdict(found) if found is not None else None,
            variation=dataclasses.asdict(found_v) if found_v is not None else None,
            contrast=getattr(estimand, "contrast", None), level=prep.level)]

    base["assumptions"] = card()
    if prep.missing.blocked and not spec.complete_rows:
        return _withheld(base, str(prep.missing.reason),
                         [{"label": "Estimate on the complete rows, its assumption stated",
                           "decision": decision.model_copy(update={"complete_rows": True})}])
    if method in ("dml_irm", "tmle") and not binary:
        return _withheld(base, f"{causal.METHOD_LABELS[method]} needs a yes/no exposure; "
                               f"`{prep.exposure}` is numeric.",
                         [{"label": causal.METHOD_LABELS["dml_plr"], "decision": decision.model_copy(
                             update={"method": "dml_plr", "trim": None, "population": "all"})}])
    try:
        design, weight_column, cluster_column = _design(ctx, prep, sample_only=spec.sample_only)
    except Withheld as exc:
        return _withheld(base, exc.reason, exc.exits)
    if method == "pds_lasso" and weight_column is not None:
        return _withheld(base, "The post-double-selection lasso cannot carry the survey weights "
                               "(its plug-in penalty assumes unweighted rows).",
                         [{"label": f"{causal.METHOD_LABELS['dml_plr']}, survey-weighted",
                           "decision": decision.model_copy(update={"method": "dml_plr"})},
                          {"label": "For these participants only, recorded as unweighted",
                           "decision": decision.model_copy(update={"sample_only": True})}])
    n = len(prep.y)
    learner = None if method == "pds_lasso" else (spec.learner or default_learner(n))
    base["learner"] = learner
    seed = int(spec.seed)
    weights = None if design is None else design.weight
    groups = None if design is None else design.groups
    y, d, X = prep.y, prep.d, (prep.X if prep.X.shape[1] else np.zeros((n, 1)))
    ctx.progress(0.15, "Reading positivity from the estimator's own propensities")
    try:
        splits = est.sample_splits(n, spec.folds, spec.repetitions, seed=seed, groups=groups)
    except ValueError as exc:
        return _withheld(base, str(exc))
    words = est.LEARNER_WORDS[learner or "linear"]
    # The estimators' own seeds for their first split's exposure model (``models/causal.py``).
    own_seed = seed if method == "dml_irm" else seed + 500
    bound = est.tmle_gbound(n)
    n_trimmed = 0
    if binary:
        if method == "pds_lasso" or (method == "tmle" and learner == "linear"):
            G = np.column_stack([np.ones(n), X])
            g = 0.5 * (1.0 + np.tanh(0.5 * (G @ est.logistic_fit(G, d, weights))))
            model = "main-terms logistic regression on every row"
        else:
            factory = est.learner_factory("linear" if learner == "linear" else str(learner))
            [g] = est.cross_fit(factory, True, X, d, splits[0], weights, own_seed)
            model = f"{words}, cross-fitted over {spec.folds} folds (the estimator's own)"
        target = "ATT" if spec.population == "exposed" else "ATE"
        found = est.overlap(g, d, bound, weights, target=target)
        base["overlap"] = OverlapView(model=model, **dataclasses.asdict(found))
        base["assumptions"] = card(found=found)
        if found.violated and spec.trim is None and not spec.acknowledged:
            return _withheld(base, _positivity_reason(prep, found),
                             causal.positivity_exits(decision, binary=True))
        if spec.trim is not None:
            keep = est.trim_rows(g, spec.trim)
            n_trimmed = int((~keep).sum())
            if keep.sum() < 20 or len(np.unique(d[keep])) < 2:
                return _withheld(base, f"Trimming at {spec.trim:g} leaves {int(keep.sum()):,} rows "
                                       f"or one exposure level: no overlap population to estimate "
                                       f"on.")
            y, d, X = y[keep], d[keep], X[keep]
            design = None if design is None else design.subset(keep)
            weights = None if design is None else design.weight
            groups = None if design is None else design.groups
            n = len(y)
            bound = est.tmle_gbound(n)
            try:
                splits = est.sample_splits(n, spec.folds, spec.repetitions, seed=seed, groups=groups)
            except ValueError as exc:
                return _withheld(base, str(exc))
    else:
        factory = est.learner_factory("linear" if method == "pds_lasso" else str(learner))
        [m] = est.cross_fit(factory, False, X, d, splits[0], weights, own_seed)
        found_v = est.variation(d, m, weights)
        base["variation"] = VariationView(
            model=(f"{est.LEARNER_WORDS['linear' if method == 'pds_lasso' else str(learner)]}, "
                   f"cross-fitted over {spec.folds} folds"), **dataclasses.asdict(found_v))
        base["assumptions"] = card(found_v=found_v)
        if found_v.violated and not spec.acknowledged:
            return _withheld(base, _variation_reason(prep, found_v),
                             causal.positivity_exits(decision, binary=False))
    ctx.progress(0.3, f"Estimating by {causal.METHOD_LABELS[method].lower()}")
    outcome_binary = prep.task == "binary"
    if method == "dml_plr":
        result = est.dml_plr(y, d, X, learner=str(learner), splits=splits, design=design,
                             outcome_binary=outcome_binary, seed=seed)
    elif method == "dml_irm":
        result = est.dml_irm(y, d, X, learner=str(learner), splits=splits,
                             score="ATT" if spec.population == "exposed" else "ATE", bound=bound,
                             design=design, outcome_binary=outcome_binary, seed=seed)
    elif method == "tmle":
        result = est.tmle(y, d, X, learner=str(learner),
                          family="binomial" if outcome_binary else "gaussian", design=design,
                          splits=None if learner == "linear" else splits, seed=seed)
    else:
        result = est.pds_lasso(y, d, X)
    measure = "risk_difference" if outcome_binary else "mean_difference"
    label = ("Effect per unit" if method in ("dml_plr", "pds_lasso") else
             "Effect among the exposed" if spec.population == "exposed" else "Average effect")
    estimates = [CausalEstimate(label=label, measure=measure, estimate=result.estimate,
                                se=result.se, ci_low=result.ci[0], ci_high=result.ci[1],
                                p_value=result.p_value, df=result.df)]
    for key, words_ in (("risk_ratio", "Marginal risk ratio"), ("odds_ratio", "Marginal odds ratio")):
        extra = result.extra.get(key)
        if extra:
            estimates.append(CausalEstimate(label=words_, measure=key, estimate=extra["estimate"],
                                            se=extra["se_log"], ci_low=extra["ci"][0],
                                            ci_high=extra["ci"][1], p_value=None, df=result.df,
                                            scale="ratio"))
    selection = None
    if method == "pds_lasso":
        names = prep.x_names
        selection = CausalSelection(
            outcome=prep.raw([names[i] for i in result.extra["selected_outcome"]]),
            exposure=prep.raw([names[i] for i in result.extra["selected_exposure"]]),
            union=prep.raw([names[i] for i in result.extra["selected"]]),
            terms=[names[i] for i in result.extra["selected"]])
    methods = causal.methods_sentence(
        method=method, exposure=prep.exposure, outcome=str(state.target),
        effect=str(getattr(estimand, "effect", "total") or "total"), adjusted=prep.adjusted,
        learner_words=est.LEARNER_WORDS.get(str(learner), ""), folds=spec.folds,
        repetitions=spec.repetitions, n=n, population=spec.population, level=prep.level,
        bound=bound, trimmed=n_trimmed, n_trim_kept=n, trim_at=spec.trim or causal.TRIM_AT,
        weight=weight_column,
        cluster=cluster_column, selected=None if selection is None else selection.model_dump(),
        candidates=len(prep.x_names),
        measure_words="difference in risk" if outcome_binary else "difference in the mean outcome")
    # INTEGRATION POINT: sensitivity to unmeasured confounding is required in the causal lane
    # (MODELING_SEQUENCE §0 ruling 10); ``causal.sensitivity_for`` is the one call site.
    sensitivity = causal.sensitivity_for(
        {"estimate": result.estimate, "se": result.se, "ci": list(result.ci)}, measure=measure,
        outcome_sd=None if outcome_binary else float(np.std(y, ddof=1)), dof=result.df)
    artifact = CausalArtifact(
        **base, estimates=estimates, repetitions=list(result.repetitions), n=n,
        n_trimmed=n_trimmed, selected=selection, sensitivity=sensitivity, methods=methods)
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


CAUSAL_READS: tuple[str, ...] = (
    "purpose", "target", "task", "event", "estimand", "adjustment", "clusters", "roles",
    "roles_unconfirmed", "role_confirmations", "reading_confirmations", "shape_confirmations",
    "missing", "survey", "categorical", "energy_adjustment", "exposure_forms", "grain", "lens",
    "findings", "split", "column_units")

__all__ = ["CAUSAL_READS", "CausalArtifact", "CausalDesignArtifact", "CausalEstimate",
           "Prepared", "Withheld", "causal_design_stage", "causal_stage", "default_learner",
           "n_limit", "prepare"]
