"""The stage registry (SIZING P0.4): where every engine object sits among the quest log's seven
stages, and what each stage reports.

The quest log (calm/FOUNDATION §3) has seven fixed stages: Your data, Your question, First look,
Who's in, Models, Results, Write-up. The engine knows only the Router's ``QUESTION_KEYS``, its
decision kinds, its findings and its ~30 compute stages. This module places each of them in one
stage, as the crosswalk does (``docs/turbotab-next/crosswalk/CROSSWALK.md``, "Engine stages and
quest stages" and the audited disagreements); the registry test holds every placement to
``crosswalk.json``.

* **Router questions** (:data:`QUESTIONS`): each with the crosswalk card it is asked on, its label
  and its place in the stage's order.
* **Decision kinds** (:func:`kind_place`): a kind that answers a Router question sits with it; every
  other kind is placed in :data:`OTHER_KINDS`. ``revert`` has no place of its own: it sits where the
  record it undoes sits (:func:`record_stage`).
* **Declarations** (:data:`DECLARATIONS`): the decisions with no Router key that the quest log lists
  as lines of their own (disagreement 11), each with when it applies, the answers its card reads
  ("Waiting for", disagreement 20) and whether it counts toward progress. An offer nothing records
  declining (a join, a codebook, the explanations) is listed and never counted.
* **Findings** (:func:`finding_place`): a finding is decided at the question it routes to, or the
  one it was held for (``defer_finding``); with neither, in Your data, where its repair is chosen.
  Explore's findings sit in :data:`EXPLORE_FINDINGS`.
* **Noticings**: each catalog thread of the understanding layer sits where the crosswalk decides it
  (``quest_noticings.json``, :func:`noticing_stage`), so the thread registry (U1) places them here.
* **Compute stages** (:data:`COMPUTE`): the quest stage that shows each one's result, so a result
  gone stale names the stage that shows it. Each estimate stage declares it serves an estimate
  (``Stage.serves``), and ``estimand.ESTIMATE_STAGES`` is read from those declarations (V2X_SEAMS
  seam guard 7).

**What each stage reports** (:func:`quest_log`): its lines under their labels (a question the
Router asks is a Decide; a default it states is a Confirm, or For the record where no other choice
is offered; FOUNDATION §3; a declaration carries its method contracts' tier, :func:`contract_tier`,
unless the crosswalk rules otherwise, :data:`TIER_RULINGS`), with "Waiting for" on a line whose
earlier answers are missing;
progress as answered over required, each counted Decide one and the stage's Confirm sweep one,
empty (``None``) for a stage not reached; and why it reopened. A stage reopens when an answer
decided elsewhere invalidates its own: a question the Router asks again (the decisions'
invalidation relations: a form left stale, an adjustment set missing a new covariate's answers, a
new outcome's event), or a result computed for answers that have changed since (the stage graph's
freshness). The reason names the stage of the answer that changed: "Your change to Who's in
reopened 2 questions in Models." A change made in the stage itself reopens its own lines, which say
so, but gives the stage no reason line: the person is looking at it.

The Confirm sweep is answered when each of its lines carries an answer recorded by the person;
P0.5 adds the one record that confirms a whole sweep. Progress reads the registry's lines only:
the noticings of the understanding layer join them when they are wired (P0.9).

Regenerating ``quest_noticings.json`` from the crosswalk (the registry test fails while it drifts)::

    venv/bin/python -m turbotab.core.tests.test_stage_registry --write-noticings
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

# Bumped when the shape or the meaning of the quest log changes (GET /projects/{pid}/quest).
QUEST_VERSION = 1

STAGES: tuple[tuple[str, str], ...] = (
    ("data", "Your data"),
    ("question", "Your question"),
    ("first_look", "First look"),
    ("whos_in", "Who's in"),
    ("models", "Models"),
    ("results", "Results"),
    ("writeup", "Write-up"),
)
STAGE_NAMES: dict[str, str] = dict(STAGES)
STAGE_INDEX: dict[str, int] = {key: i for i, (key, _) in enumerate(STAGES)}
StageKey = Literal["data", "question", "first_look", "whos_in", "models", "results", "writeup"]
Label = Literal["Decide", "Confirm", "For the record"]
DECIDE: Label = "Decide"
CONFIRM: Label = "Confirm"
RECORD: Label = "For the record"

NOTICINGS_FILE = Path(__file__).resolve().parent / "quest_noticings.json"


@dataclass(frozen=True)
class Place:
    """Where one engine object is decided: its stage, the crosswalk card (``crosswalk.json`` id),
    its label and its position in the stage's order (the crosswalk's)."""

    stage: str
    item: str
    label: Label = DECIDE
    order: float = 0.0


# ── the Router's questions ───────────────────────────────────────────────────

QUESTIONS: dict[str, Place] = {
    "lens": Place("data", "q:lens", DECIDE, 5),
    "orientation": Place("data", "q:orientation", DECIDE, 7),
    "target": Place("question", "q:target", DECIDE, 1),
    "event": Place("question", "q:event", DECIDE, 2),
    "task": Place("question", "q:task", DECIDE, 3),
    "follow_up": Place("question", "q:follow_up", DECIDE, 7),
    "purpose": Place("question", "q:goal", DECIDE, 18),
    "grain": Place("whos_in", "q:grain", DECIDE, 1),
    "repeat_kind": Place("whos_in", "q:repeat_kind", DECIDE, 2),
    "unit": Place("whos_in", "q:unit", DECIDE, 3),
    "aggregation": Place("whos_in", "q:aggregation", DECIDE, 4),
    "temporal": Place("whos_in", "q:temporal", DECIDE, 5),
    # Asked after Who's in's structural questions today (disagreement 1; P0.6 records it as a
    # completion), so Your data's roles line waits for them and says so.
    "roles": Place("data", "q:roles", DECIDE, 29),
    "clusters": Place("whos_in", "q:clusters", DECIDE, 6),
    "survey": Place("whos_in", "q:survey", DECIDE, 7),
    "exclusions": Place("whos_in", "q:exclusions", DECIDE, 11),
    "missing": Place("whos_in", "q:missing", DECIDE, 13),
    "split": Place("whos_in", "q:split", DECIDE, 23),
    "estimand": Place("models", "q:estimand", DECIDE, 1),
    "adjustment": Place("models", "q:adjustment", DECIDE, 2),
    "time_varying": Place("models", "q:time_varying", DECIDE, 3),
    "energy_adjustment": Place("models", "q:energy_adjustment", DECIDE, 4),
    "form": Place("models", "q:form", DECIDE, 5),
    # Stated by default, so they sit in the Confirm sweep (CROSSWALK §5, Estimate step 7).
    "modification": Place("models", "q:modification", CONFIRM, 133),
    "causal": Place("models", "q:causal", CONFIRM, 134),
    "models": Place("models", "q:models", DECIDE, 9),
    "substitution": Place("models", "decision:substitution-pair", DECIDE, 13),
    "open_seal": Place("results", "q:open_seal", DECIDE, 24),
}
# How a question reads when the Router states its answer instead of asking (``skipped``): a default
# set for the person, a Confirm line; For the record where the data leave no other answer to offer
# (the outcome's kind read at high confidence, ``default:task_settled``; no column that can group).
STATED: dict[str, Label] = {"task": RECORD, "clusters": RECORD}
# Kinds that answer a Router question beside the one its slot names (``sequence.question_of``): the
# follow-up's "the same for everyone", and what the task question still asks after the task.
ALSO_ANSWERS: dict[str, tuple[str, ...]] = {
    "follow_up": ("set_censoring",),
    "task": ("set_outcome_scale", "set_outcome_order"),
}

# ── every other decision kind ────────────────────────────────────────────────

OTHER_KINDS: dict[str, Place] = {
    # Your data: the files, the codebook, what each column is, each finding's disposition
    "join_files": Place("data", "decision:join_files", DECIDE, 2),
    "import_codebook": Place("data", "decision:import_codebook", DECIDE, 9),
    "set_feature_table": Place("data", "decision:set_feature_table", DECIDE, 8),
    "apply_repair": Place("data", "decision:finding-disposition", DECIDE, 11),
    "defer_finding": Place("data", "decision:finding-disposition", DECIDE, 11),
    "dismiss_finding": Place("data", "decision:finding-disposition", DECIDE, 11),
    "confirm_reading": Place("data", "decision:readings-ask-card", DECIDE, 28),
    "confirm_readings": Place("data", "decision:readings-ask-card", DECIDE, 28),
    "confirm_role": Place("data", "decision:confirm_role", DECIDE, 30),
    "set_categorical": Place("data", "reading:code-or-amount", DECIDE, 31),
    "set_column_unit": Place("data", "reading:energy-unit-and-days", DECIDE, 32),
    # Your question: the outcome card's own answers, and Predict's shape
    "set_outcome_scale": Place("question", "q:outcome_scale", DECIDE, 4),
    "set_outcome_order": Place("question", "q:outcome_order", DECIDE, 5),
    "set_outcome_unit": Place("question", "q:outcome_unit", DECIDE, 6),
    "set_censoring": Place("question", "q:follow_up", DECIDE, 7),
    "set_intended_use": Place("question", "q:intended_use", DECIDE, 20),
    # First look: the outcome views looked at
    "view_outcome": Place("first_look", "decision:view_outcome", RECORD, 5),
    # Models: the declarations with no Router key (disagreements 11 and 21), and the plan's lock
    "set_scales": Place("models", "decision:set_scales", DECIDE, 6),
    "set_batch": Place("models", "decision:set_batch", DECIDE, 7),
    # Before the families under Predict, the only goal it is asked under (CROSSWALK §5, "Predict";
    # MODEL_FAMILY_CONTRACT C4: the shelf waits for it)
    "set_selection": Place("models", "decision:set_selection", DECIDE, 8.5),
    "set_model_sequence": Place("models", "decision:set_model_sequence", DECIDE, 10),
    "set_sensitivity": Place("models", "decision:set_sensitivity", DECIDE, 11),
    "set_measurement_error": Place("models", "decision:set_measurement_error", DECIDE, 12),
    "set_usual_intake": Place("models", "decision:set_usual_intake", DECIDE, 21),
    "set_multiplicity": Place("models", "decision:set_multiplicity", CONFIRM, 135),
    "set_levers": Place("models", "decision:set_levers", CONFIRM, 137),
    "lock_plan": Place("models", "other:plan-lock", RECORD, 199),
    # Results: after the fit
    "respond_diagnostic": Place("results", "decision:respond_diagnostic", DECIDE, 7),
    "set_updating": Place("results", "decision:set_updating", DECIDE, 21),
    "reseal": Place("results", "decision:reseal", DECIDE, 36),
    "set_explain": Place("results", "decision:set_explain", DECIDE, 45),
}
# A revert sits where the record it undoes sits.
FOLLOWS_WHAT_IT_UNDOES = "revert"


def kind_place(kind: str) -> Place:
    """The place of a decision kind: its own (:data:`OTHER_KINDS`), else its Router question's.
    Raises ``KeyError`` for a kind the registry does not place (and for ``revert``)."""
    if kind in OTHER_KINDS:
        return OTHER_KINDS[kind]
    from turbotab.core.sequence import question_of

    question = question_of(kind)
    if question is None:
        raise KeyError(f"the stage registry does not place the decision kind {kind!r}")
    return QUESTIONS[question]


def kind_stages() -> dict[str, str]:
    """Every decision kind the log accepts -> its stage (``revert`` aside: it follows its
    record)."""
    from turbotab.core.decisions import SLOTS

    return {kind: kind_place(kind).stage for kind in sorted(SLOTS)}


def answering_kinds(key: str) -> tuple[str, ...]:
    """The decision kinds whose record answers the Router question ``key``."""
    from turbotab.core.decisions import SLOTS
    from turbotab.core.sequence import question_of

    return tuple(k for k in SLOTS if question_of(k) == key) + ALSO_ANSWERS.get(key, ())


# ── declarations: the decisions with no Router key that are lines of their own ─


def _goal(*goals: str) -> Callable[[Any, Sequence[str] | None], bool]:
    return lambda state, columns: getattr(state, "purpose", None) in goals


def _lens(*lenses: str) -> Callable[[Any], bool]:
    return lambda state: any(lens in (getattr(state, "lens", None) or ()) for lens in lenses)


def _always(state: Any, columns: Sequence[str] | None) -> bool:
    return True


def _batch_applies(state: Any, columns: Sequence[str] | None) -> bool:
    """Under an assay lens, with a column named as a batch, a run or a plate (the name reading that
    starts the batch finding, ``methods.batch.batch_columns``; a name is a guess, so a question)."""
    if state.purpose not in ("inference", "prediction"):
        return False
    if not _lens("metabolomics", "genomics")(state):
        return False
    from turbotab.core.readings import BATCH_KINDS
    from turbotab.core.recognizers import acquisition_kind

    return any(c != state.target and acquisition_kind(c) in BATCH_KINDS for c in columns or ())


def _model_one_is_current(state: Any) -> bool:
    """Model 1 holds while it is declared for the exposure (or the family) now declared."""
    from turbotab.core.decisions import EXPOSURE_FAMILY

    spec, est = state.model_sequence, state.estimand
    if spec is None:
        return False
    if est is None:
        return True
    return spec.exposure == (EXPOSURE_FAMILY if est.family else est.exposure)


def _calibration_is_current(state: Any) -> bool:
    """A calibration declared under another adjustment set is asked again (MODELING_SEQUENCE §2)."""
    from turbotab.core.stages.calibration import current_calibration

    spec, changed = current_calibration(state)
    return spec is not None and changed is None


@dataclass(frozen=True)
class Declaration:
    """A decision with no Router key that the quest log lists as a line (disagreement 11).

    ``applies(state, columns)``: whether the line is in the stage for this table, goal and lens.
    ``reads``: the Router questions whose answers its card reads; it waits for them. ``counted``:
    whether it counts toward progress (an offer nothing records declining does not). ``holds``:
    whether its recorded answer still stands (else it is asked again). ``only_recorded``: a For the
    record line, listed once something is recorded."""

    kind: str
    name: str
    applies: Callable[[Any, Sequence[str] | None], bool]
    reads: tuple[str, ...] = ()
    counted: bool = True
    holds: Callable[[Any], bool] | None = None
    only_recorded: bool = False

    @property
    def place(self) -> Place:
        return OTHER_KINDS[self.kind]


DECLARATIONS: tuple[Declaration, ...] = (
    Declaration("join_files", "joining another file", _always, counted=False),
    Declaration("import_codebook", "the data dictionary", _always, counted=False),
    Declaration("set_intended_use", "the intended-use question", _goal("prediction"),
                reads=("purpose",)),
    Declaration("view_outcome", "the outcome views looked at", _always, counted=False,
                only_recorded=True),
    Declaration("set_scales", "the scales question",
                lambda s, c: s.purpose in ("inference", "prediction") and _lens("survey")(s),
                reads=("roles",)),
    Declaration("set_batch", "the batch question", _batch_applies, reads=("roles",)),
    Declaration("set_selection", "the predictor-selection question", _goal("prediction"),
                reads=("roles",)),
    Declaration("set_model_sequence", "the Model 1 question", _goal("inference"),
                reads=("estimand", "adjustment"), holds=_model_one_is_current),
    # Once an exclusion rule is recorded: the analyses beside the primary's rows (Banna et al. 2017)
    Declaration("set_sensitivity", "the sensitivity analyses",
                lambda s, c: s.purpose in ("inference", "prediction") and bool(s.exclusions),
                reads=("exclusions",)),
    # Rows that are each person's mean of several recalls (the calibration stage's NOT_COMBINED)
    Declaration("set_measurement_error", "the regression calibration question",
                lambda s, c: (s.purpose == "inference" and _lens("dietary")(s)
                              and s.aggregation is not None),
                reads=("adjustment", "models"), holds=_calibration_is_current),
    Declaration("set_usual_intake", "the usual-intake question",
                lambda s, c: s.purpose == "describe" and _lens("dietary")(s), reads=("purpose",)),
    Declaration("set_multiplicity", "how many tests are accounted for",
                lambda s, c: (s.purpose == "inference" and s.estimand is not None
                              and bool(s.estimand.family)), reads=("estimand",)),
    Declaration("set_levers", "the rules repeated inside each fold", _goal("prediction"),
                reads=("models",)),
    Declaration("set_updating", "the shrinkage question", _goal("prediction"), reads=("models",)),
    Declaration("set_explain", "the explanations", _goal("inference", "prediction"),
                reads=("models",), counted=False),
)


def contract_tier(kind: str) -> str | None:
    """How the method contracts that record ``kind`` reach it (BLUEPRINT §13, their ``question``):
    ``"asked"`` (the Router's question or the card's own) or ``"stated"`` (a default the engine
    states); None when no contract records it. A question asked by any of them makes it asked."""
    from turbotab.core import contracts

    tiers = {"stated" if c.question.strip() == "stated" or c.question.startswith("(stated")
             else "implied" if c.question.startswith("(") else "asked"
             for c in contracts.contracts().values() if c.decision == kind}
    if "asked" in tiers:
        return "asked"
    return "stated" if "stated" in tiers else None


# Where a later ruling labels a line otherwise than its contracts' tier (FOUNDATION §3: an asked
# question is a Decide, a stated default a Confirm or For the record).
TIER_RULINGS: dict[str, str] = {
    "set_levers": "asked as in-fold rules, but set for the person in Models' Confirm sweep, where "
                  "the shelf ranks on the defaults now set (CROSSWALK §5; MODEL_FAMILY_CONTRACT C4)",
    "set_model_sequence": "the field's Model 1 is stated by its contract, but which columns it "
                          "adjusts for is asked in Models (CROSSWALK §5 Decide 10)",
}


# ── findings and noticings ───────────────────────────────────────────────────

# The crosswalk card a finding is decided on, by the question it routes to (or is held for).
FINDING_ROUTES: dict[str | None, str] = {
    None: "decision:finding-disposition",
    "lens": "noticing:lens-contradiction",
    "roles": "q:roles",
    "exclusions": "noticing:findings_routed_here",
    "missing": "noticing:findings_routed_here",
    "energy_adjustment": "noticing:findings-routed-to-models",
    "models": "noticing:findings-routed-to-models",
}
# The omics normalization is chosen in Models, before the families (CROSSWALK §5 Decide 8;
# disagreement 21): its finding is that card.
NORMALIZATION_FINDING = "omics_scale"
NORMALIZATION = Place("models", "decision:omics-normalization", DECIDE, 8)

# Explore's findings, by kind (``stages/explore.py``): where each is decided or shown.
EXPLORE_FINDINGS: dict[str, Place] = {
    "outcome_relationship": Place("first_look", "noticing:explore::relationship", RECORD, 0),
    "outcome_distribution": Place("first_look", "noticing:explore::outcome_distribution", RECORD,
                                  0),
    "quality_by_group": Place("whos_in", "noticing:explore::quality_by_group", RECORD, 0),
    "survey_design": Place("whos_in", "q:survey", DECIDE, 7),
    "low_variance": Place("models", "noticing:explore::low_variance", CONFIRM, 0),
    "wide": Place("models", "noticing:explore::wide", DECIDE, 17),
    "collinear": Place("models", "noticing:explore::collinear", DECIDE, 18),
}


def finding_place(finding: Mapping[str, Any], state: Any = None) -> tuple[Place, str | None]:
    """Where a finding is decided, and the question it is decided at (None: its own repair).

    Held for a question (``defer_finding``), at that question; else at the question it routes to;
    else in Your data, where its repair is chosen. The omics normalization's finding is Models'
    normalization card."""
    from turbotab.core.stages.finding_words import family

    disposition = ((getattr(state, "findings", None) or {}).get(str(finding.get("id")))
                   if state is not None else None)
    question = (disposition.to if disposition is not None and disposition.action == "deferred"
                and disposition.to in QUESTIONS else finding.get("routes_to"))
    question = question if question in QUESTIONS else None
    if family(str(finding.get("id"))) == NORMALIZATION_FINDING:
        return NORMALIZATION, question
    if question is None:
        return OTHER_KINDS["apply_repair"], None
    home = QUESTIONS[question]
    # Just after the question it is decided at.
    item = FINDING_ROUTES.get(question, home.item)
    return Place(home.stage, item, DECIDE, home.order + 0.5), question


@lru_cache(maxsize=1)
def noticing_stages() -> dict[str, str]:
    """Every catalog thread (noticing) -> its stage, as the crosswalk places it."""
    by_stage = json.loads(NOTICINGS_FILE.read_text("utf-8"))
    return {thread: stage for stage, threads in by_stage.items() for thread in threads}


def noticing_stage(thread: str) -> str:
    return noticing_stages()[thread]


# ── compute stages ───────────────────────────────────────────────────────────

# Each compute stage -> the quest stage that shows its result, with the crosswalk card that shows
# it (CROSSWALK.md, "Engine stages and quest stages", its "Shown in" column). Every estimate stage
# shows in Results, after Fit (``estimand.ESTIMATE_STAGES``); the substitution curve has no card
# of its own yet (disagreement 13: Results draws it).
COMPUTE: dict[str, tuple[str, str | None]] = {
    "ingest": ("data", "record:ingest-facts"),
    "oriented": ("data", "q:orientation"),
    "profile": ("data", "record:profile"),
    "roles": ("data", "q:roles"),
    "target_info": ("question", "q:task"),
    # Your data's noticings and First look's groups: the groups are what goes stale on screen
    "findings": ("first_look", "noticing:findings-stage"),
    "explore": ("first_look", "other:first-look:outcome-door"),
    "structure": ("whos_in", "q:grain"),
    # the table the analysis reads: one row per unit, as Who's in shapes it
    "working": ("whos_in", "q:unit"),
    "cohort": ("whos_in", "result:cohort"),
    "seal_plan": ("whos_in", "result:seal_plan"),
    "split": ("whos_in", "q:split"),
    "proposals": ("models", "q:adjustment"),
    "shelf": ("models", "q:models"),
    "forms": ("models", "q:form"),
    "design": ("models", "flowchart:analysis"),
    "causal_design": ("models", "q:causal"),
    "usual_intake": ("results", "exhibit:usual_intake_distribution"),
    "fit": ("results", "exhibit:performance_table"),
    "substitution": ("results", None),
    "sensitivity": ("results", "exhibit:declared_sensitivity"),
    "calibration": ("results", "exhibit:regression_calibration"),
    "secondary": ("results", "exhibit:secondary_further_adjusted"),
    "scales": ("results", "exhibit:scales_reliability"),
    "effects": ("results", "exhibit:table2"),
    "causal": ("results", "exhibit:causal_estimate"),
    "time_varying": ("results", "exhibit:time_varying_estimate"),
    "modification": ("results", "exhibit:effect_modification"),
    "explain": ("results", "exhibit:shap_importance"),
    "evaluation": ("results", "exhibit:decision_curve"),
}


@lru_cache(maxsize=1)
def _graph() -> Any:
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    return load_graph(GRAPH_FACTORY)


@lru_cache(maxsize=None)
def stage_reads(name: str) -> frozenset[str]:
    """Every slot a compute stage's key reads, its own and its upstream stages'."""
    graph = _graph()
    stage = graph[name]
    out = set(stage.reads) | set(stage.requires)
    for dep in stage.deps:
        out |= stage_reads(dep)
    return frozenset(out)


@lru_cache(maxsize=None)
def question_reads(key: str) -> frozenset[str]:
    """The slots a Router question rests on: every earlier question's (the Router asks in its
    dependency order, disagreement 18) and what the stages its card needs read."""
    from turbotab.core.interview import NEEDS, QUESTION_KEYS, SLOT_OF

    earlier = QUESTION_KEYS[:QUESTION_KEYS.index(key)]
    out = {SLOT_OF.get(k, k) for k in earlier}
    for stage in NEEDS.get(key, ()):
        out |= stage_reads(stage)
    return frozenset(out)


def written_slots(decision: Any) -> set[str]:
    """The slots one decision writes: its own, and those it writes beside it (``register_kind``'s
    ``also``, ``slot_for``, ``entries`` and ``confirms``)."""
    from turbotab.core import decisions as d

    kind = decision.kind
    out = {d.SLOTS[kind]} if kind in d.SLOTS else set()
    out |= set(d._ALSO.get(kind, {}))
    if kind in d._SLOT_FOR:
        out.add(d._SLOT_FOR[kind](decision))
    for entries in (d._ENTRIES.get(kind), d._CONFIRMS.get(kind)):
        if entries is not None:
            out |= {slot for slot, _key, _value in entries(decision)}
    return out


# ── the report ───────────────────────────────────────────────────────────────


class Waiting(BaseModel):
    """A question a line waits for: its key, its stage and how the app names it."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: str
    stage: str
    name: str


class ReopenedBy(BaseModel):
    """The record whose change asked a line again, and the stage it was decided in."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    decision_id: str
    kind: str
    stage: str


class QuestLine(BaseModel):
    """One line of a stage: a Router question, a declaration or a finding.

    ``status``: ``answered``; ``open`` (answerable now); ``waiting`` (``waiting_for`` names the
    earlier answers it needs, ``computing`` the results its card waits for); ``set_for_you`` (a
    default the engine stated, in the Confirm sweep or For the record)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    key: str
    source: Literal["question", "declaration", "finding"]
    label: Label
    name: str
    status: Literal["answered", "open", "waiting", "set_for_you"]
    counted: bool
    order: float
    decision_id: str | None = None
    reason: str | None = None
    waiting_for: list[Waiting] = []
    computing: list[str] = []
    reopened_by: ReopenedBy | None = None


class Progress(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    answered: int
    required: int


class Sweep(BaseModel):
    """The stage's one Confirm sweep: how many defaults it holds, and whether each is answered."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    lines: int
    answered: bool


class Reopened(BaseModel):
    """Why a stage dropped back: an answer decided in another stage asked its questions again or
    left its results computed for other answers."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    changed_in: str
    decision_id: str
    kind: str
    questions: list[str] = []  # the lines asked again (their ids)
    results: list[str] = []  # the compute stages out of date
    sentence: str


class QuestStage(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: str
    name: str
    reached: bool
    progress: Progress | None = None
    sweep: Sweep | None = None
    lines: list[QuestLine] = []
    reopened: list[Reopened] = []


class QuestLog(BaseModel):
    """The seven stages, each with its lines, progress and reopen reasons; ``kinds`` places every
    decision kind (a methods sentence's "change" link opens its stage, disagreement 19; a revert
    sits with the record it undoes)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    version: int = QUEST_VERSION
    stages: list[QuestStage]
    kinds: dict[str, str]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class _Log:
    """The decision log as the reopen reasons read it: live records in order, and what each
    revert undoes."""

    records: list[Any]
    live: list[Any] = field(default_factory=list)
    by_id: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from turbotab.core.decisions import reverted

        ordered = sorted(self.records, key=lambda r: r.seq)
        try:
            cancelled = reverted(ordered)
        except Exception:  # noqa: BLE001 - a log the fold refuses has no reasons to give
            cancelled = {}
        self.by_id = {r.id: r for r in ordered}
        self.live = [r for r in ordered if r.id not in cancelled]

    def undone(self, record: Any) -> Any:
        """The decision a record changes: its own, or, for a revert, what it undoes."""
        seen: set[str] = set()
        while record is not None and record.decision.kind == FOLLOWS_WHAT_IT_UNDOES:
            if record.id in seen:
                return None
            seen.add(record.id)
            record = self.by_id.get(record.decision.decision_id)
        return None if record is None else record.decision

    def last_answer(self, kinds: Sequence[str]) -> Any:
        return next((r for r in reversed(self.live) if r.decision.kind in kinds), None)

    def cause(self, slots: frozenset[str] | set[str], *, after_seq: int = 0,
              after_time: datetime | None = None) -> Any:
        """The latest live record, after ``after_seq`` (and ``after_time``), that changed one of
        ``slots``."""
        for r in reversed(self.live):
            if r.seq <= after_seq or (after_time is not None and r.at <= after_time):
                return None
            changed = self.undone(r)
            if changed is not None and written_slots(changed) & slots:
                return r
        return None

    def stage_of(self, record: Any) -> str | None:
        changed = self.undone(record)
        try:
            return kind_place(changed.kind).stage if changed is not None else None
        except KeyError:
            return None


def record_stage(record: Any, records: Sequence[Any]) -> str | None:
    """The stage a record was decided in (a revert: the stage of the record it undoes)."""
    return _Log(list(records)).stage_of(record)


def _reopened_by(log: _Log, kinds: Sequence[str], reads: frozenset[str]) -> ReopenedBy | None:
    answer = log.last_answer(kinds)
    if answer is None:
        return None
    cause = log.cause(reads, after_seq=answer.seq)
    stage = log.stage_of(cause) if cause is not None else None
    if cause is None or stage is None:
        return None
    return ReopenedBy(decision_id=cause.id, kind=cause.decision.kind, stage=stage)


def _question_lines(steps: Sequence[Any], log: _Log) -> list[tuple[str, QuestLine]]:
    from turbotab.core.voice import question_name

    first = next((s for s in steps if _get(s, "status") in ("open", "waiting")), None)
    lines = []
    for step in steps:
        key, status = _get(step, "key"), _get(step, "status")
        place = QUESTIONS.get(key)
        if place is None or status == "not_applicable":
            continue
        label = place.label
        if status == "skipped":
            label = place.label if place.label != DECIDE else STATED.get(key, CONFIRM)
        waiting_on = list(_get(step, "waiting_on") or [])
        waiting_for, computing = [], waiting_on
        if status == "waiting" and step is not first and waiting_on and waiting_on[0] in QUESTIONS:
            earlier = waiting_on[0]
            waiting_for = [Waiting(key=earlier, stage=QUESTIONS[earlier].stage,
                                   name=question_name(earlier))]
            computing = waiting_on[1:]
        reopened = None
        if status in ("open", "waiting"):
            reopened = _reopened_by(log, answering_kinds(key), question_reads(key))
        lines.append((place.stage, QuestLine(
            id=place.item, key=key, source="question", label=label, name=question_name(key),
            status="set_for_you" if status == "skipped" else status, counted=label == DECIDE,
            order=place.order, decision_id=_get(step, "decision_id"),
            reason=_get(step, "reason") if status == "skipped" else None,
            waiting_for=waiting_for, computing=computing, reopened_by=reopened)))
    return lines


def _declaration_lines(state: Any, steps: Mapping[str, Any], log: _Log,
                       columns: Sequence[str] | None) -> list[tuple[str, QuestLine]]:
    from turbotab.core.decisions import SLOTS
    from turbotab.core.interview import QUESTION_KEYS, SLOT_OF
    from turbotab.core.voice import question_name

    lines = []
    for decl in DECLARATIONS:
        place = decl.place
        recorded = getattr(state, SLOTS[decl.kind], None) is not None
        if decl.only_recorded:
            if not recorded:
                continue
        elif not decl.applies(state, columns):
            continue
        answered = recorded and (decl.holds is None or decl.holds(state))
        writer = log.last_answer((decl.kind,))
        unanswered = [k for k in decl.reads
                      if _get(steps.get(k), "status") in ("open", "waiting")]
        if answered:
            status = "answered"
        elif place.label == CONFIRM:
            status = "waiting" if unanswered else "set_for_you"
        else:
            status = "waiting" if unanswered else "open"
        reopened = None
        if not answered and writer is not None:
            # It rests on the questions its card reads, and on every question before them.
            last = max((QUESTION_KEYS.index(k) for k in decl.reads), default=-1)
            reads = frozenset(SLOT_OF.get(k, k) for k in QUESTION_KEYS[:last + 1])
            reopened = _reopened_by(log, (decl.kind,), reads)
        lines.append((place.stage, QuestLine(
            id=place.item, key=decl.kind, source="declaration", label=place.label,
            name=decl.name, status=status, counted=decl.counted and place.label == DECIDE,
            order=place.order, decision_id=writer.id if answered and writer is not None else None,
            waiting_for=[Waiting(key=k, stage=QUESTIONS[k].stage, name=question_name(k))
                         for k in unanswered],
            reopened_by=reopened)))
    return lines


def _finding_lines(state: Any, findings: Any, steps: Mapping[str, Any]
                   ) -> list[tuple[str, QuestLine]]:
    """Each finding of the findings stage as a line where it is decided: a Decide when it has a
    question or a repair to decide by, For the record otherwise. ``findings`` is the artifact as
    served (``repairs.annotate``: each finding's ``answered_by``)."""
    from turbotab.core.repairs import matching_answer

    lines = []
    for f in _get(findings, "findings") or []:
        place, question = finding_place(f, state)
        disposition = (state.findings or {}).get(str(f.get("id")))
        if disposition is not None and disposition.action == "deferred":
            # Held for a question: decided there, when its answer settles it.
            answered_by = None
            if matching_answer({**f, "routes_to": question}, state):
                answered_by = _get(steps.get(question), "decision_id")
            answered = answered_by is not None
        else:
            answered_by = f.get("answered_by")
            answered = answered_by is not None
        decide = bool(question or f.get("repairs") or f.get("routes_to"))
        lines.append((place.stage, QuestLine(
            id=f"finding:{f.get('id')}", key=str(f.get("id")), source="finding",
            label=DECIDE if decide else RECORD,
            name=str(f.get("summary") or f.get("title") or f.get("id")),
            status="answered" if answered else "open", counted=decide, order=place.order,
            decision_id=answered_by)))
    return lines


def _hold_the_families(placed: Sequence[tuple[str, QuestLine]]) -> None:
    """The families question waits for the Decide answers ordered before it that have no Router key
    (scales, batch, the omics normalization; selection under Predict; disagreement 21 and
    MODEL_FAMILY_CONTRACT C4's readiness): the Router does not hold it behind them."""
    models = next((l for _s, l in placed if l.source == "question" and l.key == "models"), None)
    if models is None or models.status not in ("open", "waiting"):
        return
    ahead = [l for stage, l in placed
             if stage == "models" and l.label == DECIDE and l.counted and l.status != "answered"
             and l.order < models.order
             and (l.source == "declaration"
                  or (l.source == "finding" and l.order == NORMALIZATION.order))]
    if not ahead:
        return
    models.status = "waiting"
    models.waiting_for = [*models.waiting_for,
                          *(Waiting(key=l.key, stage="models", name=l.name) for l in ahead)]


def _frontier(steps: Sequence[Any], stages: Mapping[str, Any]) -> int:
    """The furthest stage the Router has reached: every question answered, stated or open so far,
    and the first one still waiting. Results opens once an estimate stage has a result for the
    answers now (``usual_intake`` computes on the lens and goal alone, so it opens nothing), and
    Write-up opens with Results."""
    from turbotab.core.estimand import ESTIMATE_STAGES

    reached = [STAGE_INDEX[QUESTIONS[_get(s, "key")].stage] for s in steps
               if _get(s, "key") in QUESTIONS
               and _get(s, "status") in ("answered", "skipped", "open")]
    first = next((s for s in steps if _get(s, "status") in ("open", "waiting")), None)
    if first is not None and _get(first, "key") in QUESTIONS:
        reached.append(STAGE_INDEX[QUESTIONS[_get(first, "key")].stage])
    furthest = max(reached, default=0)
    if any(_get(stages.get(name), "status") == "fresh" for name in ESTIMATE_STAGES):
        furthest = max(furthest, STAGE_INDEX["results"])
    if furthest >= STAGE_INDEX["results"]:
        furthest = STAGE_INDEX["writeup"]
    return furthest


def _sentence(changed_in: str, here: str, questions: int, results: int) -> str:
    from turbotab.core.voice import plural

    parts = []
    if questions:
        parts.append(f"reopened {questions} {plural(questions, 'question')} in {here}")
    if results:
        where = "there" if questions else f"in {here}"
        parts.append(f"made {results} {plural(results, 'result')} {where} out of date")
    return f"Your change to {changed_in} {' and '.join(parts)}."


def _reasons(stage: str, lines: Sequence[QuestLine], log: _Log,
             shown_at: Mapping[str, datetime | None]) -> list[Reopened]:
    found: dict[str, dict[str, Any]] = {}

    def entry(record_id: str, kind: str, changed_in: str) -> dict[str, Any]:
        return found.setdefault(record_id, {"changed_in": changed_in, "kind": kind,
                                            "questions": [], "results": []})

    for line in lines:
        by = line.reopened_by
        if by is not None and by.stage != stage:
            entry(by.decision_id, by.kind, by.stage)["questions"].append(line.id)
    for name, (home, _item) in COMPUTE.items():
        if home != stage or name not in shown_at:
            continue
        cause = log.cause(stage_reads(name), after_time=shown_at[name])
        changed_in = log.stage_of(cause) if cause is not None else None
        if cause is None or changed_in is None or changed_in == stage:
            continue
        entry(cause.id, cause.decision.kind, changed_in)["results"].append(name)
    out = []
    for record_id, e in found.items():
        out.append(Reopened(
            changed_in=e["changed_in"], decision_id=record_id, kind=e["kind"],
            questions=e["questions"], results=e["results"],
            sentence=_sentence(STAGE_NAMES[e["changed_in"]], STAGE_NAMES[stage],
                               len(e["questions"]), len(e["results"]))))
    return sorted(out, key=lambda r: log.by_id[r.decision_id].seq, reverse=True)


def quest_log(state: Any, records: Sequence[Any], steps: Sequence[Any],
              stages: Mapping[str, Any] | None = None, *, findings: Any = None,
              columns: Sequence[str] | None = None,
              shown_at: Mapping[str, datetime | None] | None = None) -> QuestLog:
    """The seven stages for this project now.

    ``steps``: the Router's answer (``interview.route``). ``stages``: each compute stage's status.
    ``findings``: the findings artifact as served (``repairs.annotate``), else None. ``columns``:
    the table's columns, for the declarations that apply by name (a batch column). ``shown_at``:
    for each compute stage whose result is out of date (not fresh, with an older artifact), when
    that artifact was computed; a change after it, decided in another stage, is the reason."""
    stages = stages or {}
    log = _Log(list(records))
    by_key = {_get(s, "key"): s for s in steps}
    placed = [*_question_lines(steps, log), *_declaration_lines(state, by_key, log, columns),
              *_finding_lines(state, findings, by_key)]
    _hold_the_families(placed)
    frontier = _frontier(steps, stages)
    out = []
    for key, name in STAGES:
        mine = sorted((l for stage, l in placed if stage == key), key=lambda l: (l.order, l.name))
        reached = STAGE_INDEX[key] <= frontier
        confirms = [l for l in mine if l.label == CONFIRM]
        sweep = (Sweep(lines=len(confirms), answered=all(l.status == "answered" for l in confirms))
                 if confirms else None)
        progress = None
        if reached:
            decide = [l for l in mine if l.label == DECIDE and l.counted]
            swept = int(sweep is not None and sweep.answered)
            progress = Progress(answered=sum(l.status == "answered" for l in decide) + swept,
                                required=len(decide) + int(sweep is not None))
        # A stage not reached yet has nothing to drop back from.
        reasons = _reasons(key, mine, log, shown_at or {}) if reached else []
        out.append(QuestStage(key=key, name=name, reached=reached, progress=progress, sweep=sweep,
                              lines=mine, reopened=reasons))
    return QuestLog(stages=out, kinds=kind_stages())


__all__ = [
    "COMPUTE", "DECLARATIONS", "Declaration", "EXPLORE_FINDINGS", "FINDING_ROUTES",
    "FOLLOWS_WHAT_IT_UNDOES", "NORMALIZATION", "OTHER_KINDS", "Place", "Progress", "QUESTIONS",
    "QUEST_VERSION", "QuestLine", "QuestLog", "QuestStage", "Reopened", "ReopenedBy", "STAGES",
    "STAGE_NAMES", "STATED", "Sweep", "TIER_RULINGS", "Waiting", "answering_kinds",
    "contract_tier", "finding_place", "kind_place", "kind_stages", "noticing_stage",
    "noticing_stages", "quest_log", "question_reads", "record_stage", "stage_reads",
    "written_slots",
]
