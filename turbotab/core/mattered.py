"""Which decisions mattered, under Estimate (SWARM_ENGINE_PLAN E3): the specification curve of
:mod:`turbotab.core.methods.spec_curve` over the plan's recorded choices, computed by the
``mattered`` stage after the lock and served at ``GET /projects/{pid}/mattered``.

**After the lock, as a stage job.** The stage requires the plan's lock (``plan_locked``, which only
a ``lock_plan`` record in force sets), so before it the stage is blocked: nothing is fitted, and the
route refuses with the line saying when the curve opens (``spec_curve.SEALED``). No request fits a
specification. The stage depends on the fit and reads every slot its code reads
(:data:`MATTERED_READS`), so a choice changed after the lock makes the curve stale, and it is
computed again for the plan as it then stands.

**The choices (v0).** Each is a fork the decision log recorded, its recorded option the primary:

* each recorded exclusion rule (``set_exclusions``), applied as recorded or dropped;
* each covariate the adjustment answers put in the primary model (``set_adjustment``), kept or
  dropped;
* each covariate the adjustment card offered and the answers left out, left out or added;
* each alternative the analyst declared as examined (:data:`EXAMINED`, the field ``examined``: a
  list of :class:`MatteredExamined`, each changing the adjustment set). The stage hands an empty
  list until EC's analyst log records them; that log is then a slot the stage reads, so its key
  covers it.

The curve changes one choice at a time (:data:`CROSSING`): the primary and each departure from it.

**The primary stays the reported estimate.** The curve follows the linear family's coefficient of
the declared exposure, the fit stage's own table. Each specification is refit by
``spec_curve.linear_fitter`` on the table's own columns (complete cases on each specification's
columns, HC3 or CR2 by the unit, or design-based under the surveyed population). The primary
specification must reproduce the reported estimate (its estimate and interval within
:data:`AGREEMENT` of its standard error, on the same rows); otherwise no specification is shown and
the curve is refused, since the others would not be fitted as the primary is. Before anything is
fitted, a pipeline step of the plan that this refit does not repeat refuses the curve by name
(:func:`not_refitted`): a fill of missing values that any specification's columns need, the energy
model (other than total energy kept as a covariate), a declared form, blanks kept as a level, a
scale's score, an omics normalization, values below detection, a batch correction, the D-ratio
filter, or the data's own imputed copies.

**Rows.** Every analyzed row (BLUEPRINT §12 ruling 3). The specifications' rows start from the
participant flow without its exclusion rules and without its complete-cases line
(``stages.rows.compute_cohort``); each rule is a choice, and complete cases are judged on each
specification's own columns (the module docstring of ``spec_curve``), which is the flow's own
judgment of the primary under the complete-cases answer. Under the surveyed-population answer the
design is read over every row of the working table, and each specification's rows are a domain of
it (``methods.survey.for_fit``).
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from turbotab.core.graph import StageContext
from turbotab.core.methods import spec_curve as S

STAGE = "mattered"
CROSSING = "one_at_a_time"  # the curve's default (SWARM_ENGINE_PLAN E3)
FAMILY = "linear"  # the family whose coefficient the curve follows (spec_curve.linear_fitter)
EXAMINED = "examined"  # the field that holds the alternatives the analyst declared as examined
# The primary specification reproduces the reported estimate when its estimate and interval ends
# lie within this share of the reported standard error (a float64 tolerance: the same estimator on
# the same columns in another order differs by rounding, a missed plan step by far more).
AGREEMENT = 1e-6
RECORDED = "recorded"  # every choice's recorded option, the primary's
# What the stage hands ``spec_curve`` as the lock: the state's ``plan_locked`` slot, which only a
# ``lock_plan`` record in force sets (the stage's key reads the state, not the record's id; the
# route names the lock in force by its record id).
LOCKED = "plan_locked"
# The pipeline steps (``models.pipeline.shared_steps``) the refit does not repeat, in words.
STEP_WORDS = {
    "d_ratio": "the D-ratio filter",
    "detect": "the fill of values below detection",
    "normalize": "the omics normalization",
    "log": "the omics log transform",
    "impute": "the fill of missing values",
    "batch": "the batch correction",
    "score": "the scores of the declared scales",
    "energy": "the energy model",
    "form": "a declared form other than a straight line",
    "levels": "blanks kept as a level of their own",
}
NOT_OFFERED_YET = ("The curve is offered under Estimate an effect, once the goal is chosen and what "
                   "you study is declared.")
ONE_EXPOSURE = ("The curve follows one exposure's estimate; under an exposure family each exposure "
                "would need a curve of its own, which this curve does not draw yet.")
ONE_EXPOSURE_EXIT = {"label": "Declare one column you study instead of the family",
                     "stage": "question"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class MatteredExamined(_Model):
    """An alternative the analyst declared as examined: the adjustment set with ``add`` added and
    ``drop`` dropped, beside the primary (one departure of its own)."""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    add: list[str] = []
    drop: list[str] = []


class MatteredOption(_Model):
    key: str
    label: str


class MatteredChoice(_Model):
    key: str
    label: str
    primary: str
    decision: str
    options: list[MatteredOption]


class MatteredSpec(_Model):
    key: str
    picks: dict[str, str]
    primary: bool
    estimate: float
    low: float | None
    high: float | None
    n: int
    se: float | None = None
    p: float | None = None
    df: float | None = None
    estimator: str = ""
    covariance: str = ""
    scale_key: str = ""


class MatteredNotEstimated(_Model):
    key: str
    picks: dict[str, str]
    primary: bool
    reason: str


class MatteredSummary(_Model):
    k: int
    median: float
    lowest: float
    highest: float
    above: int
    below: int
    spanning: int
    primary_rank: int | None


class MatteredCurve(_Model):
    """``spec_curve.SpecCurve`` as data (its ``to_dict``)."""

    estimate_label: str
    choices: list[MatteredChoice]
    specs: list[MatteredSpec]
    not_estimated: list[MatteredNotEstimated]
    total: int
    drawn: int
    sampled: bool
    rule: str | None
    crossing: str
    summary: MatteredSummary | None
    level: float | None
    sealed: str | None
    plan_lock: str | None
    weighted: bool
    clustered: bool
    exits: list[dict[str, Any]]


class MatteredReported(_Model):
    """The reported estimate the primary specification reproduces: the fit stage's own row."""

    family: str
    feature: str
    estimate: float
    ci_low: float
    ci_high: float
    se: float | None
    n_rows: int
    covariance: str | None


class MatteredRefusal(_Model):
    message: str
    exits: list[dict[str, Any]]


class MatteredArtifact(_Model):
    """The stage's artifact."""

    crossing: str
    curve: MatteredCurve | None
    says: str | None
    methods: str | None
    reported: MatteredReported | None
    examined: list[MatteredExamined]
    refused: MatteredRefusal | None


class Mattered(_Model):
    """``GET /projects/{pid}/mattered``: the stage's status and, once it is fresh for the current
    answers, its curve. ``plan_lock`` is the lock in force, by its record id."""

    status: str
    fresh: bool
    key: str | None
    plan_lock: str | None
    curve: MatteredCurve | None
    says: str | None
    methods: str | None
    reported: MatteredReported | None
    examined: list[MatteredExamined]
    error: str | None


# ── what the stage reads ─────────────────────────────────────────────────────

# Every slot the stage's code reads, directly or through the engine's helpers it calls (the
# participant flow, the adjustment card, the design's coding, the clustering, the survey design),
# so its key covers each one (turbotab/core/tests/test_mattered.py reads them back).
MATTERED_READS: tuple[str, ...] = (
    "plan_locked", "purpose", "estimand", "adjustment", "clusters", "models", "target", "task",
    "event", "exclusions", "missing", "findings", "roles", "roles_unconfirmed",
    "role_confirmations", "reading_confirmations", "shape_confirmations", "lens", "categorical",
    "energy_adjustment", "exposure_forms", "form_domains", "follow_up", "scales", "batch",
    "column_units", "levers", "selection", "survey", "grain", "outcome_order", "repeat_kind",
    "unit", "aggregation", "numbers_read", "feature_table", "joins", "codebooks", "sex_codings",
)
MATTERED_REQUIRES: tuple[str, ...] = ("plan_locked", "estimand", "models", "purpose")


def examined_of(state: Any) -> list[MatteredExamined]:
    """The alternatives the analyst declared as examined (:data:`EXAMINED`). None is recorded
    anywhere yet: EC's analyst log will record them, and this reads that slot then."""
    return []


# ── the goal ─────────────────────────────────────────────────────────────────


def goal_of(state: Any) -> str | None:
    """``spec_curve``'s goal for ``state``: estimate (inference with what you study declared),
    describe, predict, or None with no goal answered."""
    purpose = getattr(state, "purpose", None)
    if purpose == "prediction":
        return "predict"
    if purpose == "inference":
        return "estimate" if getattr(state, "estimand", None) is not None else "describe"
    return None


def offered(state: Any) -> S.Eligibility:
    """Whether the goal offers the curve, with the line and exits that say why not."""
    goal = goal_of(state)
    if goal is None:
        return S.Eligibility(False, NOT_OFFERED_YET, (dict(S.ESTIMATE_EXIT),))
    return S.eligible(goal)


# ── the choices ──────────────────────────────────────────────────────────────


def exclusion_choices(rules: Sequence[Any], masks: Sequence[pd.Series]) -> list[S.Choice]:
    """Each recorded exclusion rule, applied as recorded (``keep``: the rows it keeps) or dropped."""
    from turbotab.core.stages.rows import rule_label

    return [S.Choice(f"exclusion:{i}", f"The exclusion rule {rule_label(rule)}",
                     (S.Alternative(RECORDED, "Applied, as recorded", keep=mask),
                      S.Alternative("dropped", "Dropped")),
                     RECORDED, decision="set_exclusions")
            for i, (rule, mask) in enumerate(zip(rules, masks))]


def adjustment_choices(adjusted: Sequence[str], left_out: Sequence[str]) -> list[S.Choice]:
    """Each covariate the adjustment answers put in the primary, kept or dropped; then each the
    adjustment card offered and the answers left out, left out or added."""
    out = [S.Choice(f"adjusted:{c}", f"`{c}` in the adjustment set",
                    (S.Alternative(RECORDED, "Adjusted for, as recorded"),
                     S.Alternative("dropped", "Dropped from the adjustment set", drop=(c,))),
                    RECORDED, decision="set_adjustment") for c in adjusted]
    out += [S.Choice(f"left_out:{c}", f"`{c}`, which the adjustment answers left out",
                     (S.Alternative(RECORDED, "Left out, as recorded"),
                      S.Alternative("added", "Added to the adjustment set", add=(c,))),
                     RECORDED, decision="set_adjustment") for c in left_out]
    return out


def examined_choices(examined: Sequence[MatteredExamined]) -> list[S.Choice]:
    """Each alternative the analyst declared as examined, beside the recorded plan."""
    return [S.Choice(f"examined:{e.key}", e.label,
                     (S.Alternative(RECORDED, "As recorded"),
                      S.Alternative("examined", e.label, add=tuple(e.add), drop=tuple(e.drop))),
                     RECORDED, decision=EXAMINED) for e in examined]


# ── what the refit does not repeat ───────────────────────────────────────────


def not_refitted(spec: Any, *, gaps: bool) -> list[str]:
    """The steps of the plan's pipeline (``models.pipeline.shared_steps``) that the curve's refit
    does not repeat, in words. ``gaps``: some column a specification reads has a missing value on
    the specifications' rows, so a fill would change it. Total energy kept as a covariate (the
    standard model, unlogged) leaves every column as it is, so it is no such step."""
    from turbotab.core.models.pipeline import shared_steps

    out: list[str] = []
    for name, _step in shared_steps(spec):
        if name == "onehot" or (name == "impute" and not gaps):
            continue
        if name == "energy":
            adj = spec.energy_adjustment()
            if adj is not None and adj.method == "standard" and not adj.log_transform:
                continue
        words = STEP_WORDS.get(name, f"the step `{name}`")
        if words not in out:
            out.append(words)
    return out


# ── the reported estimate ────────────────────────────────────────────────────


def reported_of(fit: Any, exposure: str) -> MatteredReported:
    """The linear family's row of ``exposure`` in the fit stage's table; refused when there is none
    (the family not chosen, a table refused or withheld by its own answer, the exposure split into
    several columns by a form or an energy model)."""
    data = getattr(fit, "data", fit) or {}
    model = next((m for m in data.get("models") or [] if m.get("family") == FAMILY), None)
    if model is None:
        raise S.SpecCurveRefused(
            "The curve follows the linear model's estimate (least squares, or logistic regression "
            "for a yes/no outcome), and the linear model is not among the chosen families.",
            ({"label": "Choose the linear model among the families", "stage": "models"},))
    info = model.get("inference") or {}
    if info.get("refused"):
        raise S.SpecCurveRefused(
            f"The linear model reports no estimate here, so the curve has no primary to follow: "
            f"{info['refused']}", tuple(info.get("exits") or ()))
    row = next((r for r in model.get("coefficients") or [] if r.get("feature") == exposure), None)
    if row is None or row.get("estimate") is None:
        raise S.SpecCurveRefused(
            f"The linear model reports no single coefficient of `{exposure}`, so the curve has no "
            f"one estimate to follow across the choices.", (dict(S.SCALE_EXIT),))
    if row.get("ci_low") is None or row.get("ci_high") is None:
        raise S.SpecCurveRefused(
            f"The linear model's estimate of `{exposure}` has no interval here, so the curve has "
            f"no primary interval to follow.", ())
    n = info.get("n_rows")
    return MatteredReported(family=FAMILY, feature=exposure, estimate=float(row["estimate"]),
                            ci_low=float(row["ci_low"]), ci_high=float(row["ci_high"]),
                            se=None if row.get("se") is None else float(row["se"]),
                            n_rows=int(n) if n is not None else -1,
                            covariance=info.get("covariance"))


def agrees(primary: S.Spec, reported: MatteredReported) -> bool:
    """Whether the primary specification is the reported estimate (:data:`AGREEMENT`)."""
    scale = reported.se if reported.se is not None and reported.se > 0 else abs(reported.estimate)
    bound = AGREEMENT * max(scale, 1e-300)

    def near(a: float | None, b: float) -> bool:
        return a is not None and math.isfinite(a) and abs(a - b) <= bound

    return (near(primary.estimate, reported.estimate) and near(primary.low, reported.ci_low)
            and near(primary.high, reported.ci_high) and int(primary.n) == reported.n_rows)


def _num(x: float | None) -> str:
    return "none" if x is None else f"{x:.6g}".replace("-", "−")


# ── the stage ────────────────────────────────────────────────────────────────


def _artifact(*, curve: S.SpecCurve | None = None, reported: MatteredReported | None = None,
              examined: Sequence[MatteredExamined] = (),
              refused: S.SpecCurveRefused | None = None) -> dict[str, Any]:
    doc = None
    if curve is not None:
        data = curve.to_dict()
        data["plan_lock"] = None  # the route names the lock in force by its record id
        doc = MatteredCurve.model_validate(data)
    return MatteredArtifact(
        crossing=CROSSING, curve=doc,
        says=curve.says if curve is not None else None,
        methods=(S.spec_curve_sentence(curve) if curve is not None and not curve.sealed else None),
        reported=reported, examined=list(examined),
        refused=(MatteredRefusal(message=str(refused), exits=list(refused.exits))
                 if refused is not None else None),
    ).model_dump(mode="json")


def mattered_stage(ctx: StageContext) -> dict[str, Any]:
    """The specification curve over the recorded choices, one at a time, after the lock; a curve
    that cannot be drawn is the artifact's ``refused``, with why and its exits."""
    examined = examined_of(ctx.state)
    try:
        curve, reported = curve_for(ctx, examined=examined)
    except S.SpecCurveRefused as refusal:
        return _artifact(examined=examined, refused=refusal)
    return _artifact(curve=curve, reported=reported, examined=examined)


def curve_for(ctx: StageContext, *, examined: Sequence[MatteredExamined] = ()
              ) -> tuple[S.SpecCurve, MatteredReported | None]:
    """The curve for the context's state and artifacts (``working``, ``split``, ``target_info``,
    ``design``, ``fit``), with the analyst's ``examined`` alternatives beside the recorded choices.
    Raises :class:`~turbotab.core.methods.spec_curve.SpecCurveRefused` with its exits."""
    from turbotab.core.decisions import as_rule, missing_strategy
    from turbotab.core.estimand import adjustment_card, current_estimand
    from turbotab.core.models.inference import cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import DesignSpec, design_spec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_task, coded_outcome, imputed_copies_column,
                                               outcome_levels)
    from turbotab.core.stages.rows import compute_cohort, rule_keep
    from turbotab.core.stages.working import table_info

    state = ctx.state
    eligibility = offered(state)
    if not eligibility.offered:
        raise S.SpecCurveRefused(eligibility.says, eligibility.exits)
    estimand = current_estimand(state)
    if estimand is None or getattr(estimand, "family", False) or not getattr(estimand, "exposure",
                                                                              None):
        if getattr(estimand, "family", False):
            raise S.SpecCurveRefused(ONE_EXPOSURE, (dict(ONE_EXPOSURE_EXIT),))
        raise S.SpecCurveRefused(NOT_OFFERED_YET, (dict(S.ESTIMATE_EXIT),))
    exposure = str(estimand.exposure)
    target = str(state.target)
    task = _task(ctx)
    spec = DesignSpec.from_dict(ctx.inputs["design"].objects["spec"])
    covariates = [c for c in spec.predictors if c != exposure]
    card = adjustment_card(state) or {}
    adjusted = [c for c in card.get("adjusted") or [] if c in covariates]
    left_out = [c for c in card.get("left_out") or [] if c not in covariates and c != exposure]
    rules = [as_rule(r) for r in (state.exclusions or [])]
    ingest = table_info(ctx)
    grouped_by = (getattr(ctx.inputs.get("split"), "data", None) or {}).get("grouped_by")

    with open_store(ctx) as store:
        missing_in = [c for e in examined for c in (*e.add, *e.drop) if c not in store.columns]
        if missing_in:
            raise S.SpecCurveRefused(
                f"An alternative declared as examined names `{missing_in[0]}`, which the table "
                f"does not hold.", ())
        # The flow without its exclusion rules and its complete-cases line: each rule is a choice,
        # and complete cases are judged on each specification's own columns.
        _, base, _ = compute_cohort(store, state.model_copy(update={"exclusions": None,
                                                                    "missing": None}), ingest)
        base = np.asarray(base, dtype=np.int64)
        masks = []
        for rule in rules:
            read = store.materialize(list(dict.fromkeys(rule.reads())), base)
            masks.append(rule_keep(read, rule))
        unit_columns = cluster_columns(state, store.columns, [grouped_by])
        added = list(dict.fromkeys([*left_out, *(c for e in examined for c in e.add)]))
        columns = list(dict.fromkeys([target, exposure, *covariates, *added, *unit_columns]))
        frame = modeling_frame(store, columns, base, outcome=target)
        from turbotab.core.methods.survey import for_fit

        primary_rows = np.ones(len(frame), dtype=bool)
        for mask in masks:
            primary_rows &= mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
        primary_rows &= frame[[target, exposure, *covariates]].notna().all(axis=1).to_numpy()
        clusters = (resolve_clusters(state, frame.loc[primary_rows, unit_columns], [grouped_by])
                    if unit_columns else None)
        unit = clusters.column if clusters is not None and clusters.clustered else None
        survey = for_fit(state, store, unit)

    choices = [*exclusion_choices(rules, masks), *adjustment_choices(adjusted, left_out),
               *examined_choices(examined)]
    if task == "binary":
        levels = outcome_levels(task, frame[target].to_numpy(), state.event) or {}
        what = (f"Difference in log-odds of `{target}` being `{levels.get(1, 1)}` per unit of "
                f"`{exposure}`")
        scale_key = f"log_odds:{exposure}"
    else:
        what = f"Difference in mean `{target}` per unit of `{exposure}`"
        scale_key = f"difference:{exposure}"
    if not state.plan_locked:
        return S.specification_curve(choices, lambda picks: None, lock=None, estimate_label=what,
                                     crossing=CROSSING), None

    reported = reported_of(ctx.inputs["fit"], exposure)
    read = [target, exposure, *covariates, *added]
    gaps = bool(frame[read].isna().to_numpy().any())
    held = not_refitted(spec, gaps=gaps and missing_strategy(state) != "complete_case")
    if imputed_copies_column(state) is not None:
        held.append("the analysis of the data's own imputed copies")
    if held:
        raise S.SpecCurveRefused(
            "The curve refits the linear model on the table's own columns, and the plan holds "
            + _listing(held) + ", which that refit does not repeat, so no specification would be "
            "fitted as the primary is.", ())
    if survey is not None and survey.refusal:
        raise S.SpecCurveRefused(survey.refusal, survey.exits)
    design = survey.design if survey is not None and survey.answer == "population" else None

    # The outcome as the fit codes it (the declared event as 1), and the columns the design's coding
    # reads as categories (codes the user declared, text) as categories here too.
    coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    frame = frame.copy()
    frame[target] = np.asarray(coded, dtype=float) if task == "regression" else np.asarray(coded)
    classes = None
    if task == "binary":
        values = frame[target].to_numpy()
        classes = sorted(pd.unique(values[pd.notna(values)]).tolist())
    predictors = [c for c in read if c != target]
    try:
        coding = design_spec(state, frame[predictors], predictors)
    except ValueError as exc:  # a column the design cannot code yet says why, as the design does
        raise S.SpecCurveRefused(str(exc), ()) from exc
    for c in coding.categorical:
        frame[c] = frame[c].astype(object).where(frame[c].notna(), np.nan)
    # The unit the primary's intervals cluster by (the fit's own reading of it), each row's unit
    # by it, a blank its own unit as the fit keys it; none when the primary's rows do not cluster.
    units = None
    if unit is not None and design is None:
        keys = frame[unit].astype(object).to_numpy().copy()
        blank = frame[unit].isna().to_numpy()
        keys[blank] = [f"__missing_{rid}" for rid in np.asarray(frame.index)[blank]]
        units = pd.Series(keys, index=frame.index)
    fit = S.linear_fitter(frame, outcome=target, exposure=exposure, covariates=covariates,
                          choices=choices, task=task, classes=classes, units=units,
                          unit_name=unit if units is not None else None, design=design)
    curve = S.specification_curve(choices, fit, lock=LOCKED, estimate_label=what,
                                  scale_key=scale_key, crossing=CROSSING,
                                  weighted=design is not None, clustered=units is not None)
    primary = next(s for s in curve.specs if s.primary)
    if not agrees(primary, reported):
        raise S.SpecCurveRefused(
            f"The curve's refit of the recorded choices gives {_num(primary.estimate)} "
            f"({_num(primary.low)} to {_num(primary.high)}, {primary.n:,} rows) where the "
            f"reported estimate is {_num(reported.estimate)} ({_num(reported.ci_low)} to "
            f"{_num(reported.ci_high)}, {reported.n_rows:,} rows), so its other specifications "
            f"would not be fitted as the primary is, and none is shown.", ())
    return curve, reported


def _listing(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ── serving ──────────────────────────────────────────────────────────────────


def withheld(artifact: Any, reason: str) -> Any:
    """The stage's artifact with every estimate removed and the reason first (``withheld``), as
    ``estimand.withhold`` withholds the other estimate stages'."""
    if not isinstance(artifact, dict):
        return artifact
    out = dict(artifact)
    out["withheld"] = reason
    curve = out.get("curve")
    if isinstance(curve, dict):
        out["curve"] = {**curve, "specs": [], "not_estimated": [], "summary": None, "level": None}
    out["says"] = None
    out["methods"] = None
    out["reported"] = None
    return out


def with_lock(artifact: Any, lock_id: str) -> Any:
    """The stage's artifact with its curve's ``plan_lock``: the lock in force, by its record id."""
    if not isinstance(artifact, dict) or not isinstance(artifact.get("curve"), dict):
        return artifact
    return {**artifact, "curve": {**artifact["curve"], "plan_lock": lock_id}}


def shows_estimates(artifact: Any) -> bool:
    """Whether the artifact, as served, puts an estimate in view (a specification's)."""
    curve = artifact.get("curve") if isinstance(artifact, Mapping) else None
    return isinstance(curve, Mapping) and bool(curve.get("specs"))


def served(artifact: Any, *, status: str, fresh: bool, key: str | None, lock_id: str,
           error: str | None = None) -> Mattered:
    """What the route serves: the curve once the stage is fresh for the current answers (a stale
    curve is of another plan, so none is served while it is computed again), the lock in force by
    its record id; a curve that cannot be drawn is refused (:class:`~spec_curve.SpecCurveRefused`)."""
    data = artifact if fresh and isinstance(artifact, Mapping) else None
    if data is not None and data.get("refused"):
        refused = data["refused"]
        raise S.SpecCurveRefused(refused["message"], refused.get("exits") or ())
    curve = None
    if data is not None and data.get("curve") is not None:
        curve = MatteredCurve.model_validate({**data["curve"], "plan_lock": lock_id})
    return Mattered(
        status=status, fresh=bool(fresh), key=key, plan_lock=lock_id, curve=curve,
        says=data.get("says") if data is not None else None,
        methods=data.get("methods") if data is not None else None,
        reported=(MatteredReported.model_validate(data["reported"])
                  if data is not None and data.get("reported") else None),
        examined=[MatteredExamined.model_validate(e)
                  for e in (data or {}).get("examined") or []],
        error=error)


__all__ = ["AGREEMENT", "CROSSING", "EXAMINED", "FAMILY", "LOCKED", "MATTERED_READS",
           "MATTERED_REQUIRES", "Mattered", "MatteredArtifact", "MatteredCurve",
           "MatteredExamined", "MatteredReported", "RECORDED", "STAGE", "adjustment_choices",
           "agrees", "curve_for", "examined_choices", "examined_of", "exclusion_choices",
           "goal_of", "mattered_stage", "not_refitted", "offered", "reported_of", "served",
           "shows_estimates", "with_lock", "withheld"]
