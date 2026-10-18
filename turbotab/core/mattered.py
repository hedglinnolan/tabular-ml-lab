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
The methods sentence names these alternatives as the engine derived them
(:func:`methods_sentence`); only an examined one is said to be declared.

**The primary stays the reported estimate.** The curve follows the linear family's coefficient of
the declared exposure, the fit stage's own table. Each specification is refit by
``spec_curve.linear_fitter`` on the table's own columns (complete cases on each specification's
columns), as the fit would fit the plan that specification is: its intervals HC3 (for a yes/no
outcome Wald, or Firth's profile interval where a column separates the outcome), or CR2 by the
unit its own rows repeat in, or design-based under the surveyed population (below), so they need
not be the primary's, and the methods sentence names each kind where the fit gives it. The primary
specification must reproduce the reported estimate (its estimate and interval within
:data:`AGREEMENT` of its standard error, on the same rows); otherwise no specification is shown and
the curve is refused, since the others would not be fitted as the primary is. That refusal quotes
no estimate: it is served where no estimate is.

* **Which rows belong together** is read on each specification's own rows, as the fit reads its
  plan's (``inference.resolve_clusters`` over that specification's rows and the candidate unit
  columns): a specification whose rows repeat by a unit is CR2 by it, one whose rows do not repeat
  is HC3, whatever the primary's rows do. Where that reading refuses, the fit reports no interval,
  so the specification is not estimated, with the reason. Under the surveyed population a design
  with no PSU column takes that unit as its sampling unit (``methods.survey.for_fit``), so each
  specification's design is the one its own unit gives.
* **Readings.** A departure that adds a column is judged by the fit's own reading gate
  (``readings.predictors_or_ask``, as the design stage calls it) with that column no longer left
  out: a role or a code-or-amount reading the plan would wait for leaves the departure not
  estimated, its reason which reading is not answered and where it is answered, without the fit's
  question, which the curve cannot take. A column whose role puts it in no model (an examined
  alternative may add one) is asked its code-or-amount reading as a predictor is.
* **What it cannot vary.** The alternatives are derived from the record, so each refusal of what
  cannot be varied says why with no exit to declare more or fewer: no recorded choice at all
  (:data:`NOTHING_TO_VARY`), more one-at-a-time specifications than ``spec_curve.CAP``
  (:data:`CAP_REFUSAL`), or an outcome the refit does not follow (:data:`OUTCOME_REFUSAL`:
  multiclass, ordinal, time to an event), the last before the lock too (:func:`offered`).

* **The measure.** Under a yes/no outcome the coefficient is a log odds ratio. The curve is drawn
  under the odds ratio, and its sentence says the primary is the log of the reported odds ratio,
  quoting both (:func:`says_of`). A marginal risk difference or ratio (``estimand.MARGINAL``) is
  standardized over the analyzed rows from the logistic model, which the refit does not repeat for
  each specification, so the curve is refused by name under it, before the lock too
  (:func:`offered`).
* **What the refit does not repeat.** Before anything is read from the fit or fitted, a pipeline
  step of the plan that this refit does not repeat refuses the curve by name (:func:`not_refitted`):
  a fill of missing values that the primary's columns need, the energy model (other than total
  energy kept as a covariate), a declared form, blanks kept as a level, a scale's score, an omics
  normalization, values below detection, a batch correction, the D-ratio filter, or the data's own
  imputed copies. Each other specification is judged the same way, on the coding the plan would
  give its own columns (``pipeline.design_spec`` over the table's column summaries, as the design
  stage reads them): a column only a departure adds, coded otherwise by the plan's answers (its
  blanks a level, a declared form), leaves that specification not estimated, with the step named.

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
    "levels": "the coding of blanks as a level of their own",
}
NOT_OFFERED_YET = ("The curve is offered under Estimate an effect, once the goal is chosen and what "
                   "you study is declared.")
ONE_EXPOSURE = ("The curve follows one study factor's estimate; under a family of study factors "
                "each would need a curve of its own, which this curve does not draw yet.")
ONE_EXPOSURE_EXIT = {"label": "Declare one column you study instead of the family",
                     "stage": "question"}
# Under a marginal measure (``estimand.MARGINAL``), in its words (``estimand.MEASURE_WORDS``).
MARGINAL_REFUSAL = ("The curve refits the logistic model's coefficient, a conditional log odds "
                    "ratio, and the estimate reported is a {words}, standardized over the analyzed "
                    "rows from that model (g-computation). The refit does not standardize each "
                    "specification, so the curve is not drawn under this measure.")
# What the curve cannot vary, in its own words and with no exit: every alternative is derived from
# the record (no route accepts a declared one yet), so none can be declared to change these.
NOTHING_TO_VARY = ("The plan records no exclusion rule, no covariate the adjustment answers put in "
                   "the model, and none they leave out, so no recorded choice has an alternative "
                   "to refit: the curve would be the primary alone.")
CAP_REFUSAL = ("Changing each recorded choice one at a time gives {n:,} specifications, more than "
               "the {cap:,} the curve fits at most, so it is not drawn.")
# The outcomes whose model the refit does not follow (``spec_curve.linear_fitter`` refits least
# squares or logistic regression), each named in words. ``{outcome}`` is the outcome column, and
# under a time to an event the event column with its follow-up (:func:`outcome_refused`).
OUTCOME_WORDS = {"multiclass": "an outcome of several classes",
                 "ordinal": "an outcome of ordered levels",
                 "time_to_event": "a time to an event"}
OUTCOME_REFUSAL = ("The curve refits one coefficient of what you study per specification, by least "
                   "squares for a numeric outcome or logistic regression for a yes/no one; "
                   "{outcome} is {words}, so the curve is not drawn for it.")
# The alternatives each kind of choice derives (its key's prefix), as the methods sentence says.
KIND_WORDS: tuple[tuple[str, str], ...] = (
    ("exclusion:", "each exclusion rule dropped"),
    ("adjusted:", "each covariate in the adjustment set dropped"),
    ("left_out:", "each covariate the adjustment answers left out added"),
    ("examined:", "each alternative the analyst declared as examined taken instead"),
)
# The estimator every specification is refitted by (``spec_curve.linear_fitter``), by the task.
ESTIMATOR_WORDS = {"regression": "least squares", "binary": "logistic regression"}
# Each interval the refit gives a specification, by its ``covariance`` (``models.inference``), and
# where the fit gives it: on rows that repeat within a unit CR2 by it (a separated logistic model
# there reports none, so that specification is not estimated), else HC3, or for a yes/no outcome
# Wald, or Firth's profile interval where a column separates the outcome. Under a survey design
# every one is design-based instead (``design``), which the design's own sentence says.
INTERVAL_WORDS = {
    "HC3": "heteroskedasticity-robust (HC3) where they do not repeat within a unit",
    "model": ("the logistic model's Wald interval where they do not repeat within a unit and no "
              "column separates the outcome"),
    "profile": ("the profile penalized-likelihood interval of Firth's logistic regression where a "
                "column separates the outcome"),
    "CR2": "cluster-robust (CR2) by the unit where they repeat within one",
}


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
    """The reported estimate the primary specification reproduces: the fit stage's own row.
    ``ratio``: the row's odds ratio exp(β) for a yes/no outcome, the number the fit reports beside
    the log odds; None for a difference."""

    family: str
    feature: str
    estimate: float
    ci_low: float
    ci_high: float
    se: float | None
    n_rows: int
    covariance: str | None
    ratio: float | None = None


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
    """Whether the goal, the measure and the outcome offer the curve, with the line and exits that
    say why not. A marginal measure refuses it (the module docstring), and so does a recorded
    outcome the refit does not follow (:func:`outcome_refused`), each with no exit: the measure and
    the outcome are part of the question, never changed to draw a sensitivity analysis."""
    from turbotab.core.estimand import MARGINAL, MEASURE_WORDS, current_estimand

    goal = goal_of(state)
    if goal is None:
        return S.Eligibility(False, NOT_OFFERED_YET, (dict(S.ESTIMATE_EXIT),))
    eligibility = S.eligible(goal)
    if not eligibility.offered:
        return eligibility
    measure = getattr(current_estimand(state), "measure", None)
    if measure in MARGINAL:
        return S.Eligibility(False, MARGINAL_REFUSAL.format(words=MEASURE_WORDS[measure]), ())
    outcome = outcome_refused(getattr(state, "task", None), getattr(state, "target", None),
                              follow_up_of(state))
    if outcome is not None:
        return S.Eligibility(False, outcome, ())
    return eligibility


def follow_up_of(state: Any) -> str | None:
    """The follow-up time column of a time-to-event outcome (``set_follow_up``), None unanswered."""
    return getattr(getattr(state, "follow_up", None), "time_column", None)


def outcome_refused(task: str | None, target: str | None,
                    follow_up: str | None = None) -> str | None:
    """Why the refit does not follow an outcome of this ``task`` (:data:`OUTCOME_REFUSAL`); None
    for a numeric or yes/no outcome, or one whose task is not known yet. Under a time to an event
    the outcome column is the event, so the outcome is named as the event with its follow-up
    (``follow_up``, its time column, when it is answered)."""
    if task in (None, "regression", "binary"):
        return None
    words = OUTCOME_WORDS.get(str(task), f"a {str(task).replace('_', ' ')} outcome")
    outcome = f"`{target}`"
    if task == "time_to_event":
        outcome += f" with its follow-up `{follow_up}`" if follow_up else " with its follow-up"
    return OUTCOME_REFUSAL.format(outcome=outcome, words=words)


def varied(choices: Sequence[S.Choice]) -> None:
    """Refuses, in this curve's words and with no exit, choices it cannot vary one at a time: none
    (:data:`NOTHING_TO_VARY`), or more specifications than ``spec_curve.CAP``
    (:data:`CAP_REFUSAL`), before ``spec_curve`` would refuse them with an exit to declare more or
    fewer alternatives, which the derived choices cannot take."""
    if not choices:
        raise S.SpecCurveRefused(NOTHING_TO_VARY, ())
    n = 1 + sum(len(c.options) - 1 for c in choices)
    if n > S.CAP:
        raise S.SpecCurveRefused(CAP_REFUSAL.format(n=n, cap=S.CAP), ())


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


def reading_wait(state: Any, added: Sequence[str], summaries: Mapping[str, Any],
                 store: Any) -> str | None:
    """Why the fit would wait before fitting the plan with ``added`` among its predictors: which
    reading is not answered and where it is answered (:func:`unanswered`), never the fit's question,
    which the curve cannot take; None when every reading that plan rests on is settled. The gate is
    the fit's (``readings.predictors_or_ask`` over the column summaries, as the design stage calls
    it) with ``added`` no longer left out. A column the gate does not count as a predictor (its
    role puts it in no model) is asked its code-or-amount reading as the gate asks a predictor's
    (``readings.unsettled_codes``). Two answers that disagree (codes against the energy or scales
    answer) wait on no unanswered reading, and are said in the fit's own words."""
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import (Unsettled, ask_exits, predictors_or_ask, unsettled_codes,
                                        whole_facts)

    wanted = set(added)
    try:
        preds = predictors_or_ask(state, summaries, drop=[c for c in left_out(state)
                                                          if c not in wanted], store=store)
    except Unsettled as exc:  # the fit asks, never guesses
        return unanswered(exc.readings, exc.exits) or str(exc)
    except ValueError as exc:
        return str(exc)
    extra = [c for c in added if c not in preds]
    codes = unsettled_codes(state, extra, whole_facts(extra, summaries, store)) if extra else []
    if not codes:
        return None
    return unanswered(codes, ask_exits(codes, state))


def _reading_words(r: Any, commas: set[str]) -> str:
    """Which reading ``r`` is, in words (``commas``: the columns whose ask is how their commas
    read, a text-number reading the user answered as amounts)."""
    column = f"`{r.column}`"
    if r.kind == "role":
        return f"the role of {column}"
    if r.kind == "code_or_count" and r.column in commas:
        return f"how the commas in {column} read"
    if r.kind == "code_or_count":
        return f"whether {column} holds codes for categories or amounts"
    if r.kind == "detection_limit":
        return f"how the values of {column} below a detection limit read"
    return f"how {column} reads"


def unanswered(readings: Sequence[Any], exits: Sequence[Mapping[str, Any]]) -> str | None:
    """The readings of ``readings`` that wait for an answer, in words, and the quest stage where
    each is answered: where the decisions of the fit's ``exits`` are placed (``quest.kind_place``);
    no question. None when none waits (two answers that disagree)."""
    from turbotab.core.quest import STAGE_NAMES, kind_place

    decided = [e["decision"] for e in exits if isinstance(e.get("decision"), Mapping)]
    commas = {str(x.get("finding_id"))[len("text_numbers__"):] for x in decided
              if x.get("kind") == "apply_repair"
              and str(x.get("finding_id") or "").startswith("text_numbers__")}
    what = list(dict.fromkeys(_reading_words(r, commas) for r in readings if not r.settled))
    if not what:
        return None
    where: list[str] = []
    for x in decided:
        try:
            stage = STAGE_NAMES[kind_place(str(x.get("kind"))).stage]
        except KeyError:
            continue
        if stage not in where:
            where.append(stage)
    one = len(what) == 1
    text = f"{_listing(what)} {'is' if one else 'are'} not answered yet"
    if where:
        text += f"; {'it is' if one else 'they are'} answered in {_listing(where)}"
    return text + "."


# ── the reported estimate ────────────────────────────────────────────────────


def followed(task: str = "regression") -> str | None:
    """The key of the family whose coefficient the curve follows: the ``task``'s default inference
    family by its declaration (``models.base.inference_default``: least squares for a numeric
    outcome, logistic regression for a yes/no one, as ``spec_curve.linear_fitter`` refits), read
    from the declaration, never a switch on a family's key (MODEL_FAMILY_CONTRACT C1)."""
    from turbotab.core.models.base import inference_default

    default = inference_default(task)  # type: ignore[arg-type]
    return default.key if default is not None else None


def reported_of(fit: Any, exposure: str, task: str = "regression") -> MatteredReported:
    """The followed family's row of ``exposure`` in the fit stage's table (:func:`followed`: the
    linear family for a numeric or yes/no outcome); refused when there is none (the family not
    chosen, a table refused or withheld by its own answer, the exposure split into several columns
    by a form or an energy model)."""
    data = getattr(fit, "data", fit) or {}
    key = followed(task)
    model = next((m for m in data.get("models") or [] if key is not None
                  and m.get("family") == key), None)
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
    return MatteredReported(family=str(key), feature=exposure, estimate=float(row["estimate"]),
                            ci_low=float(row["ci_low"]), ci_high=float(row["ci_high"]),
                            se=None if row.get("se") is None else float(row["se"]),
                            n_rows=int(n) if n is not None else -1,
                            covariance=info.get("covariance"),
                            ratio=None if row.get("ratio") is None else float(row["ratio"]))


def agrees(primary: S.Spec, reported: MatteredReported) -> bool:
    """Whether the primary specification is the reported estimate (:data:`AGREEMENT`)."""
    scale = reported.se if reported.se is not None and reported.se > 0 else abs(reported.estimate)
    bound = AGREEMENT * max(scale, 1e-300)

    def near(a: float | None, b: float) -> bool:
        return a is not None and math.isfinite(a) and abs(a - b) <= bound

    return (near(primary.estimate, reported.estimate) and near(primary.low, reported.ci_low)
            and near(primary.high, reported.ci_high) and int(primary.n) == reported.n_rows)


# ── words ────────────────────────────────────────────────────────────────────


def _short(x: float) -> str:
    return f"{x:.3g}".replace("-", "−")


def says_of(curve: S.SpecCurve, reported: MatteredReported | None, task: str) -> str:
    """What the curve says, its numbers the curve's own: the range and median of the estimates,
    the primary (for a yes/no outcome, the log of the reported odds ratio, both quoted), and the
    counts of intervals wholly above and below zero. Descriptive: never a test."""
    if curve.sealed:
        return curve.sealed
    s = curve.summary
    if s is None:
        return "No specification could be estimated."
    primary = next((x for x in curve.specs if x.primary), None)
    text = (f"Across {s.k:,} specifications the estimate ranged from {_short(s.lowest)} to "
            f"{_short(s.highest)} (median {_short(s.median)})")
    if primary is not None and task == "binary":
        ratio = reported.ratio if reported is not None else None
        text += (f"; the primary, {_short(primary.estimate)}, is the log of the reported odds ratio"
                 + (f", {_short(ratio)}" if ratio is not None else ""))
    elif primary is not None:
        text += f"; the primary, {_short(primary.estimate)}, is the reported estimate"
    text += (f". {s.above:,} interval{'s' if s.above != 1 else ''} lay wholly above zero and "
             f"{s.below:,} wholly below.")
    if curve.not_estimated:
        text += f" {len(curve.not_estimated):,} could not be estimated, each with its reason."
    return text


def methods_sentence(curve: S.SpecCurve, *, task: str) -> str:
    """The methods sentence of the stage's curve (one choice at a time, :data:`CROSSING`): the
    alternatives named as the engine derived them from the recorded choices (:data:`KIND_WORDS`,
    the kinds the curve holds), each specification refitted by the primary's estimator on its own
    rows (:data:`ESTIMATOR_WORDS`), the primary the reported estimate. Each specification's interval
    is the one the fit gives its own rows, which need not be the primary's: the sentence names the
    intervals the estimated specifications have, each where the fit gives it
    (:data:`INTERVAL_WORDS`, by each specification's own ``covariance``); under a survey design
    every one is design-based."""
    keys = [str(c["key"]) for c in curve.choices]
    kinds = [words for prefix, words in KIND_WORDS if any(k.startswith(prefix) for k in keys)]
    sentence = ("As a sensitivity analysis after the analysis plan was locked, the exposure's "
                "estimate was refitted with each recorded analytic choice changed, one at a time"
                + (": " + _listing(kinds) if kinds else "") + "; each specification estimated "
                "was refitted by " + ESTIMATOR_WORDS.get(task, "the primary's estimator")
                + (" on its own domain of the design" if curve.weighted else " on its own rows")
                + ", and the primary remains the reported estimate"
                + (" (the log of its odds ratio)" if task == "binary" else "") + ".")
    covariances = {s.covariance for s in curve.specs} - {"design"}
    if covariances and covariances <= set(INTERVAL_WORDS):
        sentence += (" Each specification's interval is the one the fit gives its rows: "
                     + "; ".join(w for k, w in INTERVAL_WORDS.items() if k in covariances) + ".")
    if curve.weighted:
        sentence += (" Every specification was design-based, an exclusion rule defining a domain "
                     "of the full design.")
    return sentence + " No joint test across specifications was computed."


# ── the stage ────────────────────────────────────────────────────────────────


def _artifact(*, curve: S.SpecCurve | None = None, reported: MatteredReported | None = None,
              task: str = "", examined: Sequence[MatteredExamined] = (),
              refused: S.SpecCurveRefused | None = None) -> dict[str, Any]:
    doc = None
    if curve is not None:
        data = curve.to_dict()
        data["plan_lock"] = None  # the route names the lock in force by its record id
        doc = MatteredCurve.model_validate(data)
    return MatteredArtifact(
        crossing=CROSSING, curve=doc,
        says=says_of(curve, reported, task) if curve is not None else None,
        methods=(methods_sentence(curve, task=task) if curve is not None and not curve.sealed
                 else None),
        reported=reported, examined=list(examined),
        refused=(MatteredRefusal(message=str(refused), exits=list(refused.exits))
                 if refused is not None else None),
    ).model_dump(mode="json")


def mattered_stage(ctx: StageContext) -> dict[str, Any]:
    """The specification curve over the recorded choices, one at a time, after the lock; a curve
    that cannot be drawn is the artifact's ``refused``, with why and its exits."""
    from turbotab.core.stages.modeling import _task

    examined = examined_of(ctx.state)
    try:
        curve, reported = curve_for(ctx, examined=examined)
    except S.SpecCurveRefused as refusal:
        return _artifact(examined=examined, refused=refusal)
    return _artifact(curve=curve, reported=reported, task=_task(ctx), examined=examined)


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
    # the task as the fit reads it, recorded or detected
    outcome = outcome_refused(task, target, follow_up_of(state))
    if outcome is not None:
        raise S.SpecCurveRefused(outcome, ())
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
        # Every row's column summaries, as the design stage reads them for the levels.
        column_info = {c.name: c for c in store.info().columns}
        # The fit's own reading gate over each departure that adds a column (:func:`reading_wait`),
        # read once the plan is locked: why each such departure's fit would wait, by its choice.
        waits: dict[str, str] = {}
        if state.plan_locked:
            summaries = {c.name: {"dtype": c.dtype, "n_unique": c.n_unique}
                         for c in column_info.values()}
            adds = {**{f"left_out:{c}": [c] for c in left_out},
                    **{f"examined:{e.key}": [c for c in e.add if c not in covariates
                                             and c != exposure] for e in examined}}
            for choice, cols in adds.items():
                why = reading_wait(state, cols, summaries, store) if cols else None
                if why is not None:
                    waits[choice] = why

    choices = [*exclusion_choices(rules, masks), *adjustment_choices(adjusted, left_out),
               *examined_choices(examined)]
    varied(choices)
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

    # The steps the refit does not repeat, judged before the fit's row is read (a step that renames
    # the exposure, as the energy model does, is named for what it is) and before any fit.
    raw = frame  # the columns as the table holds them, which the design's coding reads
    fills = missing_strategy(state) != "complete_case"
    by_choice = {c.key: c for c in choices}
    recorded = {c.key: c.primary for c in choices}

    def columns_of(picks: Mapping[str, str]) -> list[str]:
        """The specification's columns besides the outcome, as ``linear_fitter`` builds them."""
        cols = list(dict.fromkeys(covariates))
        for ck, ok in picks.items():
            o = by_choice[ck].option(ok)
            cols = [c for c in cols if c not in o.drop] + [c for c in o.add if c not in cols]
        return [exposure, *(c for c in cols if c != exposure)]

    def rule_rows(picks: Mapping[str, str]) -> np.ndarray:
        keep = np.ones(len(raw), dtype=bool)
        for ck, ok in picks.items():
            o = by_choice[ck].option(ok)
            if o.keep is not None:
                keep &= o.keep.reindex(raw.index).fillna(False).to_numpy(dtype=bool)
        return keep

    def held_by(picks: Mapping[str, str], coding: Any = None) -> list[str]:
        """The steps the plan would apply to this specification's columns that the refit does
        not repeat: on ``coding`` (the design stage's own, for the primary), else on the coding
        the plan's answers give these columns over every row's summaries, as the design does."""
        cols = columns_of(picks)
        if coding is None:
            coding = design_spec(state, raw[cols], cols, column_info=column_info)
        gaps = fills and bool(raw.loc[rule_rows(picks), [target, *cols]].isna().to_numpy().any())
        return not_refitted(coding, gaps=gaps)

    try:
        held = held_by(recorded, spec)
        held += [w for w in held_by(recorded) if w not in held]
    except ValueError as exc:  # a column the design cannot code yet says why, as the design does
        raise S.SpecCurveRefused(str(exc), ()) from exc
    if imputed_copies_column(state) is not None:
        held.append("the analysis of the data's own imputed copies")
    if held:
        raise S.SpecCurveRefused(
            "The curve refits the linear model on the table's own columns, and the plan holds "
            + _listing(held) + ", which that refit does not repeat, so no specification would be "
            "fitted as the primary is.", ())
    reported = reported_of(ctx.inputs["fit"], exposure, task)
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
    predictors = [exposure, *covariates, *added]
    try:
        coding = design_spec(state, frame[predictors], predictors)
    except ValueError as exc:  # a column the design cannot code yet says why, as the design does
        raise S.SpecCurveRefused(str(exc), ()) from exc
    for c in coding.categorical:
        frame[c] = frame[c].astype(object).where(frame[c].notna(), np.nan)
    def unit_of(picks: Mapping[str, str]) -> str | None:
        """The unit this specification's rows repeat in, as the fit reads the rows of the plan it
        is (``inference.resolve_clusters`` over its own rows: those its rules keep with every
        column it reads present); None when they do not repeat. Where that reading refuses, the
        fit reports no interval without a design (the design-based fit does not read it)."""
        if not unit_columns:
            return None
        cols = columns_of(picks)
        rows = rule_rows(picks) & frame[[target, *cols]].notna().all(axis=1).to_numpy()
        found = resolve_clusters(state, frame.loc[rows, unit_columns], [grouped_by])
        if found.refusal and design is None:
            raise S.SpecCurveRefused(found.refusal, tuple(found.exits))
        return found.column if found.clustered else None

    # One refit per unit the specifications' rows repeat in: each row's unit by it, a blank its own
    # unit as the fit keys it; under the surveyed population, the design that unit gives (with no
    # PSU column the unit is its sampling unit), the primary's own design for the primary's unit.
    fitters: dict[str | None, Any] = {}

    def fitter_for(u: str | None) -> Any:
        if u in fitters:
            return fitters[u]
        units, by = None, None
        if design is not None:
            found = survey
            if u != unit:
                with open_store(ctx) as store:
                    found = for_fit(state, store, u)
            if found is None or found.refusal or found.design is None:
                fitters[u] = (found.refusal if found is not None and found.refusal else
                              "the survey design for these rows could not be built")
                return fitters[u]
            by = found.design
        elif u is not None:
            keys = frame[u].astype(object).to_numpy().copy()
            blank = frame[u].isna().to_numpy()
            keys[blank] = [f"__missing_{rid}" for rid in np.asarray(frame.index)[blank]]
            units = pd.Series(keys, index=frame.index)
        fitters[u] = S.linear_fitter(frame, outcome=target, exposure=exposure,
                                     covariates=covariates, choices=choices, task=task,
                                     classes=classes, units=units, unit_name=u if units is not None
                                     else None, design=by)
        return fitters[u]

    changed_words = "With this choice changed, "

    def fit(picks: Mapping[str, str]) -> S.SpecFit:
        """A departure is fitted as the fit would fit the plan it is: it waits where that fit
        would wait for a reading (``waits``), and is not estimated where the plan would code its
        columns with a step the refit does not repeat (the primary was judged above) or where its
        rows' unit leaves no interval; its intervals cluster by the unit its own rows repeat in."""
        departure = dict(picks) != recorded
        if departure:
            wait = next((waits[k] for k, o in picks.items() if o != recorded[k] and k in waits),
                        None)
            if wait is not None:
                raise S.SpecCurveRefused(changed_words + "the fit would first wait for an answer, "
                                         "so it is not estimated: " + wait, ())
            try:
                held = held_by(picks)
            except ValueError as exc:
                raise S.SpecCurveRefused(str(exc), ()) from exc
            if held:
                raise S.SpecCurveRefused(
                    changed_words + "the plan would apply " + _listing(held) + ", which the "
                    "curve's refit does not repeat, so it is not fitted as the primary is.", ())
            try:
                u = unit_of(picks)
            except S.SpecCurveRefused as refusal:
                raise S.SpecCurveRefused(changed_words + "the fit would report no interval: "
                                         + str(refusal), ()) from refusal
        else:
            u = unit_of(picks)
        refit = fitter_for(u)
        if isinstance(refit, str):  # the design this specification's unit gives was refused
            raise S.SpecCurveRefused((changed_words + "the fit would report no interval: "
                                      if departure else "") + refit, ())
        return refit(picks)

    curve = S.specification_curve(choices, fit, lock=LOCKED, estimate_label=what,
                                  scale_key=scale_key, crossing=CROSSING,
                                  weighted=design is not None)
    # Clustered: some specification's intervals cluster by the unit its rows repeat in.
    curve.clustered = any(s.covariance == "CR2" for s in curve.specs)
    primary = next(s for s in curve.specs if s.primary)
    if not agrees(primary, reported):
        # No estimate is quoted: the refusal is served where none is (``served``, ``withheld``).
        rows = (f" (it reads {primary.n:,} rows where the fit reads {reported.n_rows:,})"
                if reported.n_rows >= 0 and int(primary.n) != reported.n_rows else "")
        raise S.SpecCurveRefused(
            f"The curve's refit of the recorded choices does not reproduce the reported "
            f"estimate{rows}, so its other specifications would not be fitted as the primary is, "
            f"and none is shown.", ())
    return curve, reported


def _listing(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ── serving ──────────────────────────────────────────────────────────────────


def withheld(artifact: Any, reason: str) -> Any:
    """The stage's artifact with every estimate removed and the reason first (``withheld``), as
    ``estimand.withhold`` withholds the other estimate stages'. A refusal of the curve goes too: it
    was drawn up for answers a question now open may change, and the reason stands first."""
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
    out["refused"] = None
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


__all__ = ["AGREEMENT", "CAP_REFUSAL", "CROSSING", "ESTIMATOR_WORDS", "EXAMINED", "followed",
           "INTERVAL_WORDS", "KIND_WORDS", "LOCKED", "MARGINAL_REFUSAL", "MATTERED_READS",
           "MATTERED_REQUIRES", "Mattered", "MatteredArtifact", "MatteredCurve",
           "MatteredExamined", "MatteredReported", "NOTHING_TO_VARY", "OUTCOME_REFUSAL",
           "OUTCOME_WORDS", "RECORDED", "STAGE", "adjustment_choices", "agrees", "curve_for",
           "examined_choices", "examined_of", "exclusion_choices", "follow_up_of", "goal_of",
           "mattered_stage", "methods_sentence", "not_refitted", "offered", "outcome_refused",
           "reading_wait", "reported_of", "says_of", "served", "shows_estimates", "unanswered",
           "varied", "with_lock", "withheld"]
