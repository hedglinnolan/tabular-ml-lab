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
* **Noticings**: each catalog thread of the understanding layer sits where the crosswalk decides it,
  with the objective of the crosswalk item that carries it (``quest_noticings.json``,
  :func:`noticing_place`), so the thread registry (U1) places them here.
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
empty (``None``) for a stage not reached, and ``complete`` once every objective is answered (a
reached stage that asks nothing, 0 of 0, is complete: its segment is full); and why it reopened.

**Reached.** A stage is reached once the Router has asked a question in it or a later one.
Results is reached once Fit is pressed (SIZING P0.8; FOUNDATION §7): under Estimate and Describe
once the plan is locked, under Predict once Fit is pressed for the outcome; it stays reached when
its results go out of date, so it can say why it dropped back. Write-up opens with Results.
Without what Fit says (``fit``), Results is reached once an estimate has been computed, for the
answers now or for earlier ones.

**The lock is visible** (``fit``, :class:`fit_press.FitLock`): whether the plan is locked, when,
its SHA-256 and why, in plain words; whether Fit was pressed, and whether the fit waits for it.

**Reopen reasons.** A stage reopens when another answer invalidates its own: a question the Router
asks again (the decisions' invalidation relations: a form left stale, an adjustment set missing a
new covariate's answers, a new outcome's event), or a result computed for answers that have changed
since (the stage graph's freshness). The record named is the one that made the answer stop holding,
found by replaying the log (a later, unrelated change never displaces it). The reason names the
stage of the answer that changed: "Your change to Who's in reopened 2 questions in Models." A
question reopened by another answer in its own stage says so too ("Your change to another answer
in Your data reopened 1 question.", FOUNDATION §3's "the join added 12 columns, so what each new
column is gets read again"; ``within``); a result its own stage's answers left out of date is not
a drop-back (each answer redraws its stage's cards). A result whose stage is blocked (the analysis
was withdrawn) is never computed again, so it is not out of date.

**While a reading recomputes.** A question whose applicability or answer is read from a stage
(the outcome's reading, the rows' structure, the form card) waits while that stage recomputes.
Its recorded answer, while it still holds on the answers now, stands meanwhile; a question with no
such answer is listed as waiting and not counted, since whether it is asked at all is that
reading's. So the bar never shows questions reopened that a recompute settles again.

**The Confirm sweep** (P0.5, ``turbotab/core/sweep.py``): a default the engine states is a Confirm
only when another choice would change a number on this table (each default's would-change test),
else For the record with why; the Confirm lines sit last in their stage, and Your data's sweep
holds what the values settled. The sweep is answered when each line carries the person's own
answer or is covered by the stage's "Confirm all" (``confirm_sweep``), which holds each default as
it was stated, so one stated otherwise since is open again. Progress reads the registry's lines
only: the noticings of the understanding layer join them when they are wired (P0.9).

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

from turbotab.core.fit_press import FitLock

# Bumped when the shape or the meaning of the quest log changes (GET /projects/{pid}/quest).
# 2 (P0.8): Results opens when Fit is pressed, and the log carries the lock (``fit``).
# 3 (P0.5): a Confirm line is a default whose alternative would change a number here, else For the
# record; Your data's readings line; the sweep's words and its "Confirm all".
QUEST_VERSION = 3

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
# A stage lists its Decides, then its one Confirm sweep, then For the record (FOUNDATION §3).
LABELS: tuple[Label, ...] = (DECIDE, CONFIRM, RECORD)

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
    # P0.6 (disagreement 10): the design slot, before the goal; observational is stated until it
    # is answered, a Confirm line (``default:design_observational``).
    "design": Place("question", "q:study-design", DECIDE, 17),
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
# A question placed otherwise under one goal (``(key, purpose)``): under Estimate the split is For
# the record in Who's in, recorded by TurboTab with no rows held out (P0.6, disagreement 5;
# ``default:split_under_inference``).
GOAL_PLACES: dict[tuple[str, str], Place] = {
    ("split", "inference"): Place("whos_in", "default:split_under_inference", RECORD, 23),
}
# A question TurboTab answers itself once its readings settle shows that record For the record in a
# later stage, besides its own line (P0.6, disagreement 1: "Column roles recorded as you confirmed
# them in Your data", in Who's in).
COMPLETED: dict[str, tuple[Place, str, str]] = {
    "roles": (Place("whos_in", "q:roles", RECORD, 5.5), "roles_recorded",
              "Column roles recorded as you confirmed them in Your data"),
}
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
    # P0.6 (disagreement 5): a changed validation scheme, its own kind; the draw stays in Who's in.
    "set_validation": Place("models", "default:validation-scheme", CONFIRM, 136),
    "lock_plan": Place("models", "other:plan-lock", RECORD, 199),
    # Results: after the fit
    "respond_diagnostic": Place("results", "decision:respond_diagnostic", DECIDE, 7),
    "set_updating": Place("results", "decision:set_updating", DECIDE, 21),
    "reseal": Place("results", "decision:reseal", DECIDE, 36),
    "set_explain": Place("results", "decision:set_explain", DECIDE, 45),
}
# A revert sits where the record it undoes sits.
FOLLOWS_WHAT_IT_UNDOES = "revert"
# "Confirm all" sits in the stage whose sweep it confirms (P0.5; ``turbotab/core/sweep.py``).
FOLLOWS_ITS_STAGE = "confirm_sweep"
# Each stage's sweep, as the crosswalk names it (First look sets nothing for the person).
SWEEP_ITEMS: dict[str, str] = {
    "data": "other:confirm-sweep:data",
    "question": "other:confirm-sweep:question",
    "whos_in": "other:confirm-sweep:whos_in",
    "models": "other:confirm-sweep",
    "results": "other:confirm-sweep:results",
    "writeup": "other:confirm-sweep:writeup",
}


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
    record; and ``confirm_sweep``: it sits in the stage it confirms)."""
    from turbotab.core.decisions import SLOTS

    return {kind: kind_place(kind).stage for kind in sorted(SLOTS) if kind != FOLLOWS_ITS_STAGE}


def answering_kinds(key: str) -> tuple[str, ...]:
    """The decision kinds whose record answers the Router question ``key``."""
    from turbotab.core.decisions import SLOTS
    from turbotab.core.sequence import question_of

    return tuple(k for k in SLOTS if question_of(k) == key) + ALSO_ANSWERS.get(key, ())


# ── declarations: the decisions with no Router key that are lines of their own ─


@dataclass(frozen=True)
class Facts:
    """What a declaration's applicability reads beyond the answers: the table's columns (a batch
    column is read by its name), and the newest artifacts of the stages that make an offer, fresh
    or not (``usual_intake``: whether the usual-intake distribution is offered)."""

    columns: Sequence[str] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)


def _goal(*goals: str) -> Callable[[Any, Facts], bool]:
    return lambda state, facts: getattr(state, "purpose", None) in goals


def _lens(*lenses: str) -> Callable[[Any], bool]:
    return lambda state: any(lens in (getattr(state, "lens", None) or ()) for lens in lenses)


def _always(state: Any, facts: Facts) -> bool:
    return True


def _linear_family(state: Any) -> bool:
    """The unpenalized regression family is chosen, or the families are not chosen yet (the line
    then waits for them): regression calibration corrects its coefficient
    (``stages/calibration.py``: NO_LINEAR). The stage switches on the key, so this mirror does
    too, and both retire in MC-2b (``test_mc2_no_family_switches.NOT_YET``)."""
    models = getattr(state, "models", None)
    return models is None or "linear" in models


def _shrinkage_offered(state: Any) -> bool:
    """A chosen family declares shrinkage by the calibration slope among its updating
    (MODEL_FAMILY_CONTRACT C9; the relation ``shrinkage_offered``), or the families are not chosen
    yet (the line then waits for them)."""
    models = getattr(state, "models", None)
    if models is None:
        return True
    from turbotab.core.models import families  # the package registers every family

    return any("shrinkage" in f.updating for f in families() if f.key in models)


def _batch_applies(state: Any, facts: Facts) -> bool:
    """Under an assay lens, with a column named as a batch, a run or a plate (the name reading that
    starts the batch finding, ``methods.batch.batch_columns``; a name is a guess, so a question)."""
    if state.purpose not in ("inference", "prediction"):
        return False
    if not _lens("metabolomics", "genomics")(state):
        return False
    from turbotab.core.readings import BATCH_KINDS
    from turbotab.core.recognizers import acquisition_kind

    return any(c != state.target and acquisition_kind(c) in BATCH_KINDS for c in facts.columns)


def _calibration_applies(state: Any, facts: Facts) -> bool:
    """Under inference and the dietary lens, rows that are each person's mean of their recalls
    (repeats combined by the mean, never as time points), with the linear family (the crosswalk's
    fires_when; the calibration stage's NOT_COMBINED, TIME_POINTS and NO_LINEAR)."""
    if state.purpose != "inference" or not _lens("dietary")(state):
        return False
    aggregation, repeats = state.aggregation, state.repeat_kind
    if aggregation is None or aggregation.method != "mean":
        return False
    if repeats is not None and repeats.repeat_kind == "time_points":
        return False
    return _linear_family(state)


def _usual_intake_applies(state: Any, facts: Facts) -> bool:
    """Under the dietary lens and any goal but prediction, where the usual-intake stage offers the
    distribution (repeated recalls, enough people with two or more: ``UsualIntakeOffer.offered``),
    or once an answer is recorded (the crosswalk: it "runs under either purpose today")."""
    if state.purpose in (None, "prediction") or not _lens("dietary")(state):
        return False
    if getattr(state, "usual_intake", None):
        return True
    offer = (facts.artifacts.get("usual_intake") or {}).get("offer") or {}
    return bool(offer.get("offered"))


def _model_one_is_current(state: Any) -> bool:
    """Model 1 holds as the engine holds it (``estimand.current_model_sequence``): for the exposure
    now declared, with every Model 1 column still in the primary adjustment set (MODELING_SEQUENCE
    §2: another exposure, or a column leaving the set, asks it again)."""
    from turbotab.core.estimand import current_model_sequence

    return current_model_sequence(state) is not None


def _calibration_is_current(state: Any) -> bool:
    """A calibration declared under another adjustment set is asked again (MODELING_SEQUENCE §2)."""
    from turbotab.core.stages.calibration import current_calibration

    spec, changed = current_calibration(state)
    return spec is not None and changed is None


@dataclass(frozen=True)
class Declaration:
    """A decision with no Router key that the quest log lists as a line (disagreement 11).

    ``applies(state, facts)``: whether the line is in the stage for this table, goal and lens.
    ``reads``: the Router questions whose answers its card reads; it waits for them. ``counted``:
    whether it counts toward progress (an offer nothing records declining does not). ``holds``:
    whether its recorded answer still stands (else it is asked again). ``only_recorded``: a For the
    record line, listed once something is recorded. ``own_record``: answered only by a record of its
    own kind, since its slot is another answer's too (the validation scheme rides in the split's
    record until it is changed, P0.6)."""

    kind: str
    name: str
    applies: Callable[[Any, Facts], bool]
    reads: tuple[str, ...] = ()
    counted: bool = True
    holds: Callable[[Any], bool] | None = None
    only_recorded: bool = False
    own_record: bool = False

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
                lambda s, f: s.purpose in ("inference", "prediction") and _lens("survey")(s),
                reads=("roles",)),
    Declaration("set_batch", "the batch question", _batch_applies, reads=("roles",)),
    Declaration("set_selection", "the predictor-selection question", _goal("prediction"),
                reads=("roles",)),
    Declaration("set_model_sequence", "the Model 1 question", _goal("inference"),
                reads=("estimand", "adjustment"), holds=_model_one_is_current),
    # Once an exclusion rule is recorded: the analyses beside the primary's rows (Banna et al. 2017)
    Declaration("set_sensitivity", "the sensitivity analyses",
                lambda s, f: s.purpose in ("inference", "prediction") and bool(s.exclusions),
                reads=("exclusions",)),
    Declaration("set_measurement_error", "the regression calibration question",
                _calibration_applies, reads=("adjustment", "models"),
                holds=_calibration_is_current),
    # It waits for the repeats and the survey answers (its depends_on; SURVEY_UNANSWERED).
    Declaration("set_usual_intake", "the usual-intake question", _usual_intake_applies,
                reads=("repeat_kind", "survey")),
    Declaration("set_multiplicity", "how many tests are accounted for",
                lambda s, f: (s.purpose == "inference" and s.estimand is not None
                              and bool(s.estimand.family)), reads=("estimand",)),
    Declaration("set_levers", "the rules repeated inside each fold", _goal("prediction"),
                reads=("models",)),
    # An unpenalized regression family under prediction (the relation shrinkage_offered).
    Declaration("set_updating", "the shrinkage question",
                lambda s, f: s.purpose == "prediction" and _shrinkage_offered(s),
                reads=("models",)),
    Declaration("set_explain", "the explanations", _goal("inference", "prediction"),
                reads=("models",), counted=False),
    # P0.6 (disagreement 5): the scheme set for you with the draw, a Models Confirm line under
    # Predict; changing it records its own kind and keeps the draw.
    Declaration("set_validation", "the validation scheme", _goal("prediction"), reads=("split",),
                own_record=True),
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
def noticing_places() -> dict[str, tuple[str, str]]:
    """Every catalog thread (noticing) -> (its stage, its objective), as the crosswalk places it:
    the objective is the crosswalk item's that carries it (Decide, Confirm, For the record, or
    "Shown (not an objective)" for one an exhibit shows)."""
    by_stage = json.loads(NOTICINGS_FILE.read_text("utf-8"))
    return {thread: (stage, objective) for stage, threads in by_stage.items()
            for thread, objective in threads.items()}


def noticing_stages() -> dict[str, str]:
    """Every catalog thread (noticing) -> its stage."""
    return {thread: stage for thread, (stage, _objective) in noticing_places().items()}


def noticing_place(thread: str) -> tuple[str, str]:
    return noticing_places()[thread]


def noticing_stage(thread: str) -> str:
    return noticing_places()[thread][0]


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


def answer_holds(key: str) -> Callable[[Any], bool]:
    """Whether a Router question's recorded answer stands on a state, as far as the answers alone
    say (``interview.route``'s ``answers``): the follow-up for this outcome's kind, the estimand
    while its exposure is in the model, the adjustment set while every covariate has its answers,
    the causal lane for the exposure now declared, the modifiers once each is complete, each form
    on its column's present scale; else its slot holds a value. What only an artifact can say (a
    form card read on other rows) is not seen here."""
    from turbotab.core import causal, estimand
    from turbotab.core.interview import SLOT_OF
    from turbotab.core.methods import exposure_form, interaction

    by_answers: dict[str, Callable[[Any], bool]] = {
        "follow_up": lambda s: estimand.follow_up_answer(s) is not None,
        "estimand": lambda s: estimand.current_estimand(s) is not None,
        "adjustment": lambda s: estimand.adjustment_answer(s) is not None,
        "causal": lambda s: causal.current_causal(s) is not None,
        "modification": lambda s: interaction.modification_answer(s) is not None,
        "form": lambda s: not exposure_form.stale_forms(s),
    }
    slot = SLOT_OF.get(key, key)
    return by_answers.get(key, lambda s: getattr(s, slot, None) is not None)


# The stages whose reading says whether the Router asks a question at all, or whether its answer
# stands (``interview.route``'s gates and answers, and the artifacts ``ProjectService.interview``
# hands it): while one recomputes, the question waits on it and what it will be is not known yet.
READ_BY_GATE: dict[str, frozenset[str]] = {
    "orientation": frozenset({"oriented"}),
    "event": frozenset({"target_info"}),
    "task": frozenset({"target_info"}),
    "follow_up": frozenset({"target_info"}),
    "grain": frozenset({"structure"}),
    "repeat_kind": frozenset({"structure"}),
    "unit": frozenset({"structure"}),
    "aggregation": frozenset({"structure"}),
    "temporal": frozenset({"structure"}),
    "clusters": frozenset({"roles"}),
    "time_varying": frozenset({"structure", "time_varying"}),
    "form": frozenset({"forms"}),
    "causal": frozenset({"causal_design", "time_varying"}),
}


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


class ReadOption(BaseModel):
    """Another reading of a column, and the decision that records it."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    label: str
    decision: dict[str, Any] | None = None


class ReadItem(BaseModel):
    """One reading the values settled with no question asked (``readings.read_from_data``)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    kind: str
    column: str
    value: str
    words: str
    evidence: str
    options: list[ReadOption] = []
class ChangedSince(BaseModel):
    """An answer that returns with the reason (crosswalk disagreements 20 and 1): one decided early
    ("Decide now") once an answer it was decided ahead of is recorded, so what its card counted
    may have moved (``decided``); the roles TurboTab recorded from the person's confirmations once
    a proposal moves after them (``confirmed``). ``decision_id`` is the later answer."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    since: Literal["decided", "confirmed"]
    decision_id: str
    kind: str
    reason: str


class QuestLine(BaseModel):
    """One line of a stage: a Router question, a declaration, a finding or what the values settled
    (``reading``).

    ``status``: ``answered``; ``open`` (answerable now); ``waiting`` (``waiting_for`` names the
    earlier answers it needs, ``computing`` the results its card waits for); ``set_for_you`` (a
    default the engine stated, in the Confirm sweep or For the record). A default's ``reason`` is
    why it was set; on a Confirm line ``would_change`` says what another choice would change here,
    and a default For the record says in ``changes_nothing`` why no other choice changes a number
    (P0.5). ``id`` is the card that opens its options; a reading's ``items`` carry their own."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    key: str
    source: Literal["question", "declaration", "finding", "reading"]
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
    would_change: str | None = None
    changes_nothing: str | None = None
    items: list[ReadItem] = []
    changed_since: ChangedSince | None = None


class Progress(BaseModel):
    """A reached stage's objectives: each counted Decide one, its Confirm sweep one. ``complete``
    once every one is answered. A reached stage that asks nothing (0 of 0: First look until its
    noticings are wired, Results under Estimate, Write-up) is complete, its segment full; a stage
    not reached has no progress at all (empty, never "0 of N")."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    answered: int
    required: int
    complete: bool


class Sweep(BaseModel):
    """The stage's one Confirm sweep (P0.5): how many defaults it holds, and whether each is
    answered, by the person's own answer or by "Confirm all" (``confirmed_by``, the
    ``confirm_sweep`` record). ``changed``: the lines set for the person anew, or stated otherwise,
    since that confirmation. ``id`` is the crosswalk's card; ``heading`` and ``action`` are its
    words ("Here are the 6 other choices set for you", "Confirm all 6")."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    lines: int
    answered: bool
    id: str = ""
    heading: str = ""
    action: str = ""
    confirmed_by: str | None = None
    changed: list[str] = []


class Reopened(BaseModel):
    """Why a stage dropped back: an answer decided in another stage asked its questions again or
    left its results computed for other answers; or (``within``) another answer in the stage
    itself asked its questions again."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    changed_in: str
    decision_id: str
    kind: str
    questions: list[str] = []  # the lines asked again (their ids)
    results: list[str] = []  # the compute stages out of date
    within: bool = False  # the answer that changed is in this stage
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
    fit: FitLock | None = None  # Fit and the plan's lock (SIZING P0.8)


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
    ordered: list[Any] = field(default_factory=list)
    _states: dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from turbotab.core.decisions import reverted

        self.ordered = sorted(self.records, key=lambda r: r.seq)
        try:
            cancelled = reverted(self.ordered)
        except Exception:  # noqa: BLE001 - a log the fold refuses has no reasons to give
            cancelled = {}
        self.by_id = {r.id: r for r in self.ordered}
        self.live = [r for r in self.ordered if r.id not in cancelled]

    def state_at(self, seq: int) -> Any:
        """The state just after record ``seq``: the log as it stood then, folded (None when the
        fold refuses it)."""
        if seq not in self._states:
            from turbotab.core.decisions import fold

            try:
                self._states[seq] = fold([r for r in self.ordered if r.seq <= seq])
            except Exception:  # noqa: BLE001 - a prefix the fold refuses says nothing
                self._states[seq] = None
        return self._states[seq]

    def holds_at(self, holds: Callable[[Any], bool], seq: int) -> bool | None:
        state = self.state_at(seq)
        if state is None:
            return None
        try:
            return bool(holds(state))
        except Exception:  # noqa: BLE001 - an answer the check cannot read is not known
            return None

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

    def changes(self, slots: frozenset[str] | set[str], *, after_seq: int = 0,
                after_time: datetime | None = None) -> list[Any]:
        """The live records, after ``after_seq`` (and ``after_time``), that changed one of
        ``slots``, oldest first."""
        out = []
        for r in self.live:
            if r.seq <= after_seq or (after_time is not None and r.at <= after_time):
                continue
            changed = self.undone(r)
            if changed is not None and written_slots(changed) & slots:
                out.append(r)
        return out

    def cause(self, slots: frozenset[str] | set[str], *, after_seq: int = 0,
              after_time: datetime | None = None) -> Any:
        """The latest live record, after ``after_seq`` (and ``after_time``), that changed one of
        ``slots``."""
        found = self.changes(slots, after_seq=after_seq, after_time=after_time)
        return found[-1] if found else None

    def stage_of(self, record: Any) -> str | None:
        changed = self.undone(record)
        if changed is not None and changed.kind == FOLLOWS_ITS_STAGE:
            return changed.stage
        try:
            return kind_place(changed.kind).stage if changed is not None else None
        except KeyError:
            return None


def record_stage(record: Any, records: Sequence[Any]) -> str | None:
    """The stage a record was decided in (a revert: the stage of the record it undoes)."""
    return _Log(list(records)).stage_of(record)


def _reopened_by(log: _Log, kinds: Sequence[str], reads: frozenset[str],
                 holds: Callable[[Any], bool] | None = None) -> ReopenedBy | None:
    """The record that asked a line again: of the live records after its last answer that changed
    a slot it reads, the one after which its answer stopped holding (``holds``, replayed on the log
    as it stood after each), so a later change that left it as it was never displaces the cause;
    the latest of them when the answers alone cannot say (an artifact reopened it)."""
    answer = log.last_answer(kinds)
    if answer is None:
        return None
    changes = log.changes(reads, after_seq=answer.seq)
    if not changes:
        return None
    cause = changes[-1]
    if holds is not None:
        broke = None
        for record in changes:
            stands = log.holds_at(holds, record.seq)
            if stands is True:
                broke = None  # restored since: a later change is the cause
            elif stands is False and broke is None:
                broke = record
        cause = broke or cause
    stage = log.stage_of(cause)
    if stage is None:
        return None
    return ReopenedBy(decision_id=cause.id, kind=cause.decision.kind, stage=stage)


def _stated_reason(key: str) -> str | None:
    """Why a default TurboTab recorded itself holds (the split under Estimate)."""
    if key == "split":
        from turbotab.core.seal import INFERENCE_SPLIT_REASON

        return INFERENCE_SPLIT_REASON
    return None


def _changed_since_decided(key: str, writer: Any, log: _Log) -> ChangedSince | None:
    """An answer decided early returns once an answer to a question it was decided ahead of is
    recorded after it (crosswalk disagreement 20): its card counted before that answer."""
    from turbotab.core.interview import QUESTION_KEYS
    from turbotab.core.sequence import question_of
    from turbotab.core.voice import question_name

    early = getattr(writer, "early", None)
    if not early or early not in QUESTION_KEYS or key not in QUESTION_KEYS:
        return None
    ahead = QUESTION_KEYS[QUESTION_KEYS.index(early):QUESTION_KEYS.index(key)]
    kinds = {k for q in ahead for k in answering_kinds(q)}
    cause = next((r for r in reversed(log.live) if r.seq > writer.seq
                  and getattr(log.undone(r), "kind", None) in kinds), None)
    if cause is None:
        return None
    changed = log.undone(cause)
    name = question_name(question_of(changed.kind) or early)
    return ChangedSince(
        since="decided", decision_id=cause.id, kind=changed.kind,
        reason=f"Changed since you decided: {name} was answered after you decided this early, so "
               f"what its card counted may have moved.")


def _changed_since_confirmed(writer: Any, log: _Log, facts: Facts) -> ChangedSince | None:
    """The roles TurboTab recorded from the person's confirmations return once a column's proposal
    reads otherwise after its confirmation, because of a later answer the roles' reading reads
    (the outcome, the goal, what one row is; crosswalk disagreement 1)."""
    from turbotab.core.readings import proposals_of
    from turbotab.core.sequence import question_of
    from turbotab.core.voice import question_name

    recorded = dict(getattr(writer.decision, "roles", None) or {})
    reads = stage_reads("roles") - {"roles"}
    for proposal in proposals_of(facts.artifacts.get("roles")):
        column, proposed = str(proposal.get("column")), proposal.get("proposed")
        if column not in recorded or not proposed or proposed == recorded[column]:
            continue
        confirmed = _confirmed_at(log, column)
        if confirmed is None:
            continue
        cause = log.cause(reads, after_seq=confirmed.seq)
        if cause is None:
            continue
        changed = log.undone(cause)
        question = question_of(changed.kind)
        after = question_name(question) if question else changed.kind
        return ChangedSince(
            since="confirmed", decision_id=cause.id, kind=changed.kind,
            reason=f"Changed since you confirmed: `{column}` now reads as {proposed} after "
                   f"{after}; you confirmed it as {recorded[column]}.")
    return None


def _confirmed_at(log: _Log, column: str) -> Any:
    """The latest live record that confirmed ``column``'s role on its own."""
    for r in reversed(log.live):
        d = r.decision
        if d.kind == "confirm_role" and d.column == column:
            return r
        if d.kind == "confirm_reading" and d.reading == "role" and d.column == column:
            return r
        if d.kind == "confirm_readings" and any(
                i.reading == "role" and i.column == column for i in d.items):
            return r
    return None


def _question_lines(state: Any, steps: Sequence[Any], log: _Log,
                    facts: Facts | None = None) -> list[tuple[str, QuestLine]]:
    from turbotab.core.voice import question_name

    first = next((s for s in steps if _get(s, "status") in ("open", "waiting")), None)
    purpose = getattr(state, "purpose", None)
    lines = []
    for step in steps:
        key, status = _get(step, "key"), _get(step, "status")
        place = GOAL_PLACES.get((key, purpose)) or QUESTIONS.get(key)
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
        decision_id, counted = _get(step, "decision_id"), label == DECIDE
        if status == "waiting" and READ_BY_GATE.get(key, frozenset()) & set(waiting_on):
            # Its own reading recomputes: a recorded answer that still holds stands meanwhile;
            # else whether it is asked at all is that reading's, so it is not counted yet.
            answer = log.last_answer(answering_kinds(key))
            if answer is not None and answer_holds(key)(state):
                status, decision_id, waiting_for = "answered", answer.id, []
                computing = [s for s in waiting_on if s not in QUESTIONS]
            else:
                counted = False
        reopened = None
        if status in ("open", "waiting"):
            reopened = _reopened_by(log, answering_kinds(key), question_reads(key),
                                    answer_holds(key))
        writer = log.by_id.get(decision_id) if decision_id else None
        by_turbotab = getattr(writer, "recorded_by", "you") == "turbotab"
        changed = (_changed_since_decided(key, writer, log)
                   if status == "answered" and writer is not None else None)
        if status == "answered" and by_turbotab and key in COMPLETED:
            home, line_key, said = COMPLETED[key]
            lines.append((home.stage, QuestLine(
                id=home.item, key=line_key, source="question", label=home.label, name=said,
                status="set_for_you", counted=False, order=home.order, decision_id=decision_id,
                changed_since=_changed_since_confirmed(writer, log, facts or Facts()))))
        elif status == "answered" and by_turbotab and label != DECIDE:
            status = "skipped"  # a default TurboTab recorded: set for you, with a way to change it
        lines.append((place.stage, QuestLine(
            id=place.item, key=key, source="question", label=label, name=question_name(key),
            status="set_for_you" if status == "skipped" else status, counted=counted,
            order=place.order, decision_id=decision_id,
            reason=(_get(step, "reason") or _stated_reason(key)) if status == "skipped" else None,
            waiting_for=waiting_for, computing=computing, reopened_by=reopened,
            changed_since=changed)))
    return lines


def _declaration_reads(decl: Declaration) -> frozenset[str]:
    """The slots a declaration's answer rests on: the questions its card reads, and what each of
    those rests on (:func:`question_reads`: every earlier question, and the stages its card
    needs)."""
    from turbotab.core.interview import SLOT_OF

    out: set[str] = set()
    for key in decl.reads:
        out |= question_reads(key) | {SLOT_OF.get(key, key)}
    return frozenset(out)


def _declaration_lines(state: Any, steps: Mapping[str, Any], log: _Log,
                       facts: Facts) -> list[tuple[str, QuestLine]]:
    from turbotab.core.decisions import SLOTS
    from turbotab.core.voice import question_name

    lines = []
    for decl in DECLARATIONS:
        place = decl.place
        recorded = getattr(state, SLOTS[decl.kind], None) is not None
        if decl.own_record:
            recorded = recorded and log.last_answer((decl.kind,)) is not None
        if decl.only_recorded:
            if not recorded:
                continue
        elif not decl.applies(state, facts):
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
            reopened = _reopened_by(log, (decl.kind,), _declaration_reads(decl), decl.holds)
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


def _frontier(steps: Sequence[Any], stages: Mapping[str, Any],
              shown_at: Mapping[str, datetime | None], fit: FitLock | None = None) -> int:
    """The furthest stage the Router has reached: every question answered, stated or open so far,
    and the first one still waiting. Results opens when Fit is pressed (``fit``: under Estimate and
    Describe the plan locked, under Predict Fit pressed for the outcome; never with no purpose),
    and stays open when a change after the fit puts its results out of date (the outcome changed,
    the goal withdrawn: ``fit.opened``), so it drops back with its reason rather than emptying. Without ``fit``, Results opens once an estimate stage has a
    result, for the answers now (fresh) or for earlier ones (``shown_at``: out of date, or
    withdrawn since) (``usual_intake``'s offer computes on the lens and goal alone, so it opens
    nothing). Write-up opens with Results."""
    from turbotab.core.estimand import ESTIMATE_STAGES

    first = next((s for s in steps if _get(s, "status") in ("open", "waiting")), None)
    # A default stated past the first unanswered question (the design, observational until it is
    # answered; P0.6) is not a question the Router has reached.
    ahead = list(steps).index(first) if first is not None else len(steps)
    reached = [STAGE_INDEX[QUESTIONS[_get(s, "key")].stage] for i, s in enumerate(steps)
               if _get(s, "key") in QUESTIONS
               and (_get(s, "status") in ("answered", "open")
                    or (_get(s, "status") == "skipped" and i < ahead))]
    if first is not None and _get(first, "key") in QUESTIONS:
        reached.append(STAGE_INDEX[QUESTIONS[_get(first, "key")].stage])
    furthest = max(reached, default=0)
    if fit is not None:
        opened = fit.purpose is not None and (fit.locked if fit.locks else fit.pressed)
        # Opened by a press since kept (for an earlier outcome, or before the goal was withdrawn),
        # Results stays reached while a result it showed is out of date, and says why.
        opened = opened or (fit.opened and any(name in shown_at for name in ESTIMATE_STAGES
                                               if name != "usual_intake"))
    else:
        opened = any(_get(stages.get(name), "status") == "fresh" or name in shown_at
                     for name in ESTIMATE_STAGES if name != "usual_intake")
    if opened:
        furthest = max(furthest, STAGE_INDEX["results"])
    if furthest >= STAGE_INDEX["results"]:
        furthest = STAGE_INDEX["writeup"]
    return furthest


def _sentence(changed_in: str, here: str, questions: int, results: int, *,
              within: bool = False) -> str:
    from turbotab.core.voice import plural

    if within:
        return (f"Your change to another answer in {here} reopened {questions} "
                f"{plural(questions, 'question')}.")
    parts = []
    if questions:
        parts.append(f"reopened {questions} {plural(questions, 'question')} in {here}")
    if results:
        where = "there" if questions else f"in {here}"
        parts.append(f"made {results} {plural(results, 'result')} {where} out of date")
    return f"Your change to {changed_in} {' and '.join(parts)}."


def _reasons(stage: str, lines: Sequence[QuestLine], log: _Log, stages: Mapping[str, Any],
             shown_at: Mapping[str, datetime | None]) -> list[Reopened]:
    found: dict[str, dict[str, Any]] = {}

    def entry(record_id: str, kind: str, changed_in: str) -> dict[str, Any]:
        return found.setdefault(record_id, {"changed_in": changed_in, "kind": kind,
                                            "questions": [], "results": []})

    for line in lines:
        by = line.reopened_by
        if by is not None:
            entry(by.decision_id, by.kind, by.stage)["questions"].append(line.id)
    for name, (home, _item) in COMPUTE.items():
        if home != stage or name not in shown_at:
            continue
        reads = stage_reads(name)
        if _get(stages.get(name), "status") == "blocked":
            # Blocked: its requirement was withdrawn, so it is never computed again; not out of
            # date. The goal is the exception: an estimate waits for it (P0.8), and is computed
            # again once it is answered, so a change to the goal is why Results dropped back.
            reads = frozenset({"purpose"}) & reads
            if not reads:
                continue
        cause = log.cause(reads, after_time=shown_at[name])
        changed_in = log.stage_of(cause) if cause is not None else None
        # Its own stage's answers redraw its cards as they are given: no drop-back.
        if cause is None or changed_in is None or changed_in == stage:
            continue
        entry(cause.id, cause.decision.kind, changed_in)["results"].append(name)
    out = []
    for record_id, e in found.items():
        within = e["changed_in"] == stage
        out.append(Reopened(
            changed_in=e["changed_in"], decision_id=record_id, kind=e["kind"],
            questions=e["questions"], results=e["results"], within=within,
            sentence=_sentence(STAGE_NAMES[e["changed_in"]], STAGE_NAMES[stage],
                               len(e["questions"]), len(e["results"]), within=within)))
    return sorted(out, key=lambda r: log.by_id[r.decision_id].seq, reverse=True)


def quest_log(state: Any, records: Sequence[Any], steps: Sequence[Any],
              stages: Mapping[str, Any] | None = None, *, findings: Any = None,
              columns: Sequence[str] | None = None,
              artifacts: Mapping[str, Any] | None = None,
              shown_at: Mapping[str, datetime | None] | None = None,
              readings: Sequence[Mapping[str, Any]] | None = None,
              fit: FitLock | None = None) -> QuestLog:
    """The seven stages for this project now.

    ``steps``: the Router's answer (``interview.route``). ``stages``: each compute stage's status.
    ``findings``: the findings artifact as served (``repairs.annotate``), else None. ``columns``:
    the table's columns, for the declarations that apply by name (a batch column). ``artifacts``:
    the newest artifacts of the stages that make an offer (``usual_intake``), fresh or not.
    ``shown_at``: for each compute stage whose result is not for the answers now (not fresh, with
    an older artifact, blocked ones included), when that artifact was computed; a change after it,
    decided in another stage, is the reason, unless the stage is blocked (withdrawn). Any estimate
    stage among them keeps Results reached. ``fit``: Fit and the plan's lock
    (``fit_press.fit_lock``), which opens Results and is reported with the log. ``readings``: what
    the values settled with no question asked (``readings.read_from_data``), Your data's line in
    its sweep.

    P0.5 (``turbotab/core/sweep.py``): a default stated for the person is a Confirm only when
    another choice would change a number here, else For the record with why; the Confirm lines
    sit last in their stage, after its Decides, and "Confirm all" (``confirm_sweep``) answers them
    while each is still stated as it was confirmed. A noticing the triage at the gate disposed of
    (it changes no number here, or it is left as a limitation) is answered by that record."""
    from turbotab.core import sweep as sweeps

    stages = stages or {}
    shown_at = shown_at or {}
    log = _Log(list(records))
    by_key = {_get(s, "key"): s for s in steps}
    facts = Facts(columns=tuple(columns or ()), artifacts=dict(artifacts or {}))
    placed = [*_question_lines(state, steps, log, facts),
              *_declaration_lines(state, by_key, log, facts),
              *_finding_lines(state, findings, by_key), *sweeps.reading_lines(state, readings)]
    _hold_the_families(placed)
    sweeps.weigh(placed, state, facts)
    sweeps.cover_noticings(placed, state, log, findings)
    frontier = _frontier(steps, stages, shown_at, fit)
    out = []
    for key, name in STAGES:
        mine = sorted((l for stage, l in placed if stage == key),
                      key=lambda l: (LABELS.index(l.label), l.order, l.name))
        reached = STAGE_INDEX[key] <= frontier
        sweep = sweeps.sweep_of(key, mine, state, log)
        progress = None
        if reached:
            decide = [l for l in mine if l.label == DECIDE and l.counted]
            swept = int(sweep is not None and sweep.answered)
            answered = sum(l.status == "answered" for l in decide) + swept
            required = len(decide) + int(sweep is not None)
            progress = Progress(answered=answered, required=required, complete=answered >= required)
        # A stage not reached yet has nothing to drop back from.
        reasons = _reasons(key, mine, log, stages, shown_at) if reached else []
        out.append(QuestStage(key=key, name=name, reached=reached, progress=progress, sweep=sweep,
                              lines=mine, reopened=reasons))
    return QuestLog(stages=out, kinds=kind_stages(), fit=fit)


__all__ = [
    "COMPLETED", "COMPUTE", "ChangedSince", "DECLARATIONS", "Declaration", "EXPLORE_FINDINGS",
    "FINDING_ROUTES", "FOLLOWS_ITS_STAGE", "FOLLOWS_WHAT_IT_UNDOES", "Facts", "GOAL_PLACES",
    "LABELS", "NORMALIZATION", "OTHER_KINDS", "Place", "Progress", "QUESTIONS", "QUEST_VERSION",
    "QuestLine", "QuestLog", "QuestStage", "READ_BY_GATE", "ReadItem", "ReadOption", "Reopened",
    "ReopenedBy", "STAGES", "STAGE_NAMES", "STATED", "SWEEP_ITEMS", "Sweep", "TIER_RULINGS",
    "Waiting", "answer_holds", "answering_kinds", "contract_tier", "finding_place", "kind_place",
    "kind_stages", "noticing_place", "noticing_places", "noticing_stage", "noticing_stages",
    "quest_log", "question_reads", "record_stage", "stage_reads", "written_slots",
]
