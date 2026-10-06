"""The interview Router (M1_CONTRACT.md §1): which question is asked now, and why the others wait.

A pure function of the recorded state, the stage statuses and a few small artifacts. The client
renders what it says and never decides the order itself.

Rules:

* Questions are asked in :data:`QUESTION_KEYS` order. A question whose slot holds an answer is
  ``answered`` (with the record that wrote it); changing an earlier answer never reopens a later
  one — it stays answered and its stages go stale.
* ``task`` is ``skipped`` while detection is high-confidence for the current outcome and no
  answer overrides it; ``reason`` is the detection. The client's "Ask me anyway" reopens it.
* ``energy_adjustment`` is ``not_applicable`` unless the lens includes ``dietary``, a column has
  the ``energy`` role and an ``exposure`` carries energy; ``reason`` names what is missing.
  ``substitution`` is ``not_applicable`` with fewer than two exposures that carry energy, as an
  amount or as a share of energy.
* The opening sequence (M2_CONTRACT §1, OPENING_SEQUENCE §01/§03), nothing resequenced:
  ``orientation`` fires when the lens includes an assay pack and the oriented stage's reading
  (names first, then a scale-aware shape; audit WP14) is not "one row per sample" — feature-major
  or undetermined (and, while it is open, the target question waits behind it);
  ``event`` only for a binary or time-to-event outcome; ``grain`` always, but ``skipped``
  (stated) when a recognized person identifier is unique on every row and nothing repeats like a
  roster (M2_CONTRACT §10; the structure stage's ``grain.stated``); ``repeat_kind`` and ``unit``
  only when units repeat, ``repeat_kind`` usually ``skipped`` (stated from the structure stage's
  reading, with its evidence as ``reason``); ``aggregation`` only when the unit is the unit;
  ``temporal`` only when time points stay as rows. A grain of ``unknown`` ("I don't know") repeats nothing, so the four
  follow-ups do not apply. A question that does not fire is ``not_applicable`` with the reason;
  ``orientation`` stays answered once answered, since its slot turns the table whatever the lens.
* ``survey`` (audit §5 WP10) is asked after the roles, under inference, when a column reads as a
  survey weight: the surveyed population (design-based) or these participants (unweighted, and
  recorded as such). It is ``not_applicable`` under prediction and without a weight column.
* WP17 (audit §5; ``turbotab/core/estimand.py``): ``follow_up`` is asked of a yes/no outcome
  beside a column that reads as a follow-up time (stated, "skipped", when none does) and of a time
  to event, whose follow-up must be named; it is answered by the follow-up (a time to event) or by
  "the same for everyone" (``censoring``, a yes/no outcome). ``clusters`` is asked after the roles
  when a column reads as a group of participants. Under inference ``estimand`` (the exposure and
  its effect) and ``adjustment`` (each covariate's answers, complete for the current exposure) are
  asked; under prediction they are not applicable. They are the modeling sequence's steps 2 and 3
  (MODELING_SEQUENCE §1, "everything after the seal"), so they follow the opening sequence's
  eligibility, missing values and split, which nothing resequences (OPENING_SEQUENCE §01), and
  come before the energy model (step 4, the domain transforms) and the models.
* ``causal`` (the causal lane; ``turbotab/core/causal.py``) comes before the models under inference,
  so it is part of the plan declared before any estimate is shown (the plan lock records it):
  ``not_applicable`` under prediction, for an exposure family, a direct effect or an outcome that is
  neither numeric nor yes/no; otherwise ``skipped``: the primary model alone is stated, the
  top-ranked causal estimator one step away (and, when the adjustment set's candidates are many
  relative to n, why it ranks first: rung (d) ranks high), also while its card computes, so it
  never holds the models question. It is answered by ``set_causal`` for the declared exposure (a
  new exposure re-asks it).
* FORM (MODELING_SEQUENCE §1 rows 5 and 7): ``form`` comes after the domain transforms (the
  energy model, step 4), so knots, cut points and the estimand's unit belong to the exposure's
  final scale. Under inference it is asked of the declared exposure and each adjusted continuous
  confounder the ``forms`` stage's card lists, and answered while each has a form declared on its
  present scale; a later transform leaves a form stale, as does a change of the analyzed rows for
  a spline whose k Harrell's rule read on other rows, and the question opens again (``followup``
  "stale"), never keeping it. It reads only the fresh card: while the card recomputes it waits,
  and no later question is answered meanwhile. Under prediction it is stated (``skipped``).
  ``modification`` is stated until an effect modifier or a second exposure is declared, then
  answered once an interaction's second exposure has its adjustment answers (``followup``
  "adjustment" while it waits); not applicable under prediction or for an exposure family.
* V2 causal row (``turbotab/core/time_varying.py``): ``time_varying`` is asked under inference
  when the rows are a unit's time points kept as rows and one exposure is declared, after the
  adjustment set; ``not_applicable`` when the ``time_varying`` stage reads the exposure as fixed
  within every unit. It is answered by the lane for the current exposure, and the weights' lane
  only once its truncation is declared after the diagnostics.
* ``open_seal`` is the last step (M2_CONTRACT §12.1): asked once the fit is fresh (it waits on the
  fit until then), ``not_applicable`` when nothing is held out, and answered once opened. Its slot
  is ``seal_opened``.
* A skip's ``reason`` is the clause after the client's own "Not asked:" label, so it never begins
  with those words itself.
* ``task`` (audit WP18, RO-10) stays open while its answer is incomplete (``followup``): a positive,
  markedly skewed outcome's scale (``"scale"``), or the order of an ordinal text outcome's levels
  (``"order"``); a task the detection reads at high confidence is skipped only when neither waits
  (``turbotab.core.structural.task_followup``).
* The open question carries the ledger's one ask card (``ask``; BLUEPRINT §14.2): the unsettled
  readings the consumer its answer feeds will read, placed where that consumer needs them, never
  as a wall at upload (``turbotab.core.ask``).
* At most one question is ``open``: the first applicable unanswered one. It is ``waiting``
  instead while a stage it needs is still being computed (``waiting_on`` names the stage) —
  ``substitution`` waits until ``fit`` is fresh. Every later unanswered question is ``waiting``
  on the earliest unanswered question before it (plus any stage of its own still computing).
  A stage that failed or was stopped does not hold a question back: the question opens and the
  pipeline panel shows the failure.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.ask import AskCard, AskContext

QuestionKey = Literal[
    "lens", "orientation", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "time_varying", "energy_adjustment", "form",
    "modification", "causal",
    "models",
    "substitution", "open_seal",
]
QUESTION_KEYS: tuple[str, ...] = (
    "lens", "orientation", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "time_varying", "energy_adjustment", "form",
    "modification", "causal",
    "models",
    "substitution", "open_seal",
)
# The ProjectState slot a question's answer writes, where it is not the question's own name.
# FORM: the form question's answers write the forms (``set_exposure_form``, ``set_forms``), the
# modifiers' question its declared modifiers (``set_modification``).
SLOT_OF: dict[str, str] = {"open_seal": "seal_opened", "form": "exposure_forms",
                           "modification": "modifications"}
ASSAY_LENSES = ("metabolomics", "genomics")
StepStatus = Literal["answered", "open", "waiting", "skipped", "not_applicable"]

# The stage a question needs before it can be shown (its options come from it).
NEEDS: dict[str, tuple[str, ...]] = {
    "lens": ("ingest",),
    "orientation": ("oriented",),
    "target": ("oriented",),
    "event": ("target_info",),
    "task": ("target_info",),
    # WP17: the follow-up time's candidates are read with the outcome (``target_info.follow_up``).
    "follow_up": ("target_info",),
    "grain": ("structure",),
    "repeat_kind": ("structure",),
    "unit": (),
    "aggregation": ("structure",),
    "temporal": ("structure",),
    "purpose": (),
    "roles": ("roles",),
    # Its options (one per recognized weight, the pooled cycle) are read with the proposals.
    "survey": ("proposals",),
    # WP17: a grouping is read from the roles; the estimand and adjustment cards are proposals'.
    "clusters": ("roles",),
    "estimand": ("proposals",),
    "adjustment": ("proposals",),
    "exclusions": ("proposals",),
    "missing": (),
    # The checks come before the seal (audit RO-02): a repair to the outcome after the draw would
    # draw it again, so the split waits for the findings, and one that would rewrite a column the
    # draw reads is settled first (``seal._the_draw_reads_settled_values``).
    "split": ("findings",),
    # V2 causal row: whether the exposure changes within units, the proposal and the diagnostics
    # are the time_varying stage's (turbotab/core/time_varying.py).
    "time_varying": ("time_varying",),
    "energy_adjustment": ("proposals",),
    # FORM (MODELING_SEQUENCE §1 rows 5 and 7): the form question's card (the columns it asks
    # about, on their final scale, with k by the rule) is the ``forms`` stage's; the modifiers'
    # question reads the state alone.
    "form": ("forms",),
    "modification": (),
    "models": ("shelf",),
    # The causal lane: its card (the options, the assumptions and the outcome-free diagnostics) is
    # the ``causal_design`` stage's (``turbotab/core/causal.py``).
    "causal": ("causal_design",),
    "substitution": ("fit",),
    "open_seal": ("fit",),
}
MUST_BE_FRESH = {"substitution": "fit", "open_seal": "fit"}
NOT_ASKED = "Not asked:"  # the client's label before a skip's reason


class InterviewStep(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: QuestionKey
    status: StepStatus
    decision_id: str | None = None
    reason: str | None = None
    waiting_on: list[str] = []
    # M2_CONTRACT §4: findings deferred to this question, which resurface inside it, attributed.
    deferred_findings: list[str] = []
    # WP18 (audit RO-10): what an open task question still needs after its task ("scale": the
    # outcome's scale; "order": an ordinal text outcome's order). Its options are the target
    # stage's ``scale_question`` and ``order_question``.
    # FORM: the form question re-asked because a domain transform left a declared form stale
    # ("stale"); an interaction's second exposure still waiting for its adjustment answers
    # ("adjustment").
    followup: Literal["scale", "order", "stale", "adjustment"] | None = None
    # WP18 (BLUEPRINT §14.2): the one ask card, on the open question whose answer feeds a consumer
    # of unsettled readings.
    ask: AskCard | None = None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _deps() -> dict[str, tuple[str, ...]]:
    try:
        from turbotab.core.graph import load_graph
        from turbotab.core.stages import GRAPH_FACTORY

        return {s.name: s.deps for s in load_graph(GRAPH_FACTORY).stages()}
    except Exception:  # pragma: no cover - a graph that cannot load has nothing pending
        return {}


def pending_stages(stages: Mapping[str, Any], deps: Mapping[str, Sequence[str]] | None = None) -> set[str]:
    """Stages whose work for the current answers is under way or about to start.

    Queued or running; or idle/stale (not stopped by request) while a stage it depends on is
    pending — it starts when that one finishes.
    """
    deps = _deps() if deps is None else deps
    memo: dict[str, bool] = {}

    def pending(name: str) -> bool:
        if name in memo:
            return memo[name]
        memo[name] = False  # guards a (malformed) cycle
        status = stages.get(name)
        state = _get(status, "status")
        result = state in ("queued", "running") or (
            state in ("idle", "stale")
            and not _get(status, "cancelled", False)
            and any(pending(d) for d in deps.get(name, ()))
        )
        memo[name] = result
        return result

    return {name for name in stages if pending(name)}


def _live_writer(records: Sequence[Any], state: Any) -> dict[str, str]:
    """Slot -> id of the live record that wrote its current value."""
    from turbotab.core.decisions import SLOTS, Revert, reverted

    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = reverted(ordered)
    except Exception:
        cancelled = {}
    out: dict[str, str] = {}
    for record in ordered:
        decision = record.decision
        if record.id in cancelled or isinstance(decision, Revert) or decision.kind not in SLOTS:
            continue
        slot = SLOTS[decision.kind]
        if decision.kind == "set_task" and (
            decision.column != state.target or decision.task != state.task
        ):
            continue
        if decision.kind == "set_event" and (
            decision.column != state.target or decision.level != state.event
        ):
            continue
        # An opening or a re-seal is about its own outcome's seal (audit WP16, RO-05).
        if decision.kind in ("open_seal", "reseal") and getattr(decision, "target", None) not in (
            None, state.target
        ):
            continue
        if decision.kind in ("set_follow_up", "set_censoring") and decision.column != state.target:
            continue
        out[slot] = record.id
    return out


def _energy_applicability(state: Any, bearing: Callable[[str], bool]) -> str | None:
    """Why energy adjustment does not apply, or None when it does (or cannot tell yet)."""
    if state.lens is not None and "dietary" not in state.lens:
        return "The dietary lens is off, so energy adjustment does not apply."
    if state.roles is None:
        return None
    from turbotab.core.readings import unsettled

    if unsettled(state):
        # BLUEPRINT §14.1: whether a column is total energy, or an exposure, may still change with
        # a role nobody confirmed; the question is not removed on an unsettled reading.
        return None
    roles = state.roles
    if not any(r == "energy" for r in roles.values()):
        return "No column has the energy role, so there is no total energy to adjust against."
    if not any(r == "exposure" for r in roles.values()):
        return "No column is an exposure, so there is nothing to adjust."
    # BLUEPRINT §14.1 (the readings ledger): "carries no energy" is a reading of the name, never
    # settled by it; an exposure no name reads as a nutrient (``Energykcal``'s NDNS ``Protein``,
    # a food group) may still carry energy, so the estimand question is asked rather than
    # removed. The energy card says which exposures it reads as nutrients.
    return None


def _substitution_applicability(state: Any, bearing: Callable[[str], bool]) -> str | None:
    """Why a substitution does not apply, or None. An exposure carries energy as an amount
    (``fat_g``) or as a share of energy (``fat_pct_kcal``: the field's "5% of energy from X
    replaced by Y", audit B24 and D19); a swap moves energy between two of them."""
    from turbotab.core.methods.percent_energy import is_percent_of_energy

    if state.roles is None:
        return None
    n = sum(1 for c, r in state.roles.items()
            if r == "exposure" and (bearing(c) or is_percent_of_energy(c)))
    if n < 2:
        return (f"Only {n} exposure carries energy; a substitution swaps kcal between two."
                if n == 1 else "No exposure carries energy; a substitution swaps kcal between two.")
    return None


Gate = tuple[str, str | None] | None  # ("not_applicable" | "skipped", reason), or None: ask


def _orientation_gate(state: Any, oriented: Any) -> Gate:
    if state.lens is None:
        return None
    if not any(lens in ASSAY_LENSES for lens in state.lens):
        return ("not_applicable",
                "No assay lens is on, and other tables are not exported turned around.")
    reading = _get(oriented, "reading") or {}
    # WP14 (audit IN-11): under an assay lens it is asked unless the table reads as one row per
    # sample; an undetermined reading is a question, never a silent "rows are samples". BLUEPRINT
    # §14.1 (the readings ledger): only a settled reading answers it, and the orientation reader is
    # never high (its evidence is the header's grammar and the shape), so under an assay lens the
    # question is asked with the reading as its proposal.
    from turbotab.core.readings import stated_reading

    found = stated_reading("orientation", "__table__", _get(reading, "reading"),
                           _get(reading, "confidence"))
    if oriented is None or found.value != "sample_major" or not found.settled:
        return None
    sentence = _get(reading, "sentence") or ""
    return ("not_applicable", sentence or "The table reads as one row per sample.")


def _event_gate(state: Any, target_info: Any) -> Gate:
    if state.target is None:
        return None
    task = state.task
    if task is None and target_info is not None and _get(target_info, "column") == state.target:
        task = _get(target_info, "task")
    if task is None:
        return None
    if task not in ("binary", "time_to_event"):  # a time-to-event outcome's column is its event
        return ("not_applicable", f"The outcome is read as {task}, so there is no event level to choose.")
    return None


def _skip_reason(text: Any) -> str | None:
    """A skip's reason as the clause after the client's "Not asked:" label."""
    if not text:
        return None
    text = str(text).strip()
    return text[len(NOT_ASKED):].lstrip() if text.startswith(NOT_ASKED) else text


def _grain_gate(state: Any, structure: Any) -> Gate:
    """Stated, not asked, when a recognized person identifier is unique on every row (M2 §10)."""
    if state.grain is not None:
        return None
    stated = _get(_get(structure, "grain"), "stated")
    if stated and _get(stated, "column"):
        return ("skipped", _skip_reason(_get(stated, "sentence")))
    return None


def _grain_answer(state: Any, structure: Any) -> str | None:
    """The grain as answered, else as stated (``one_row_per_unit``), else None while unanswered."""
    if state.grain is not None:
        return str(_get(state.grain, "grain"))
    if _grain_gate(state, structure) is not None:
        return "one_row_per_unit"
    return None


UNKNOWN_GRAIN = "Whether a unit can appear in more than one row is not known"


def _repeat_kind_gate(state: Any, structure: Any) -> Gate:
    grain = _grain_answer(state, structure)
    if grain is None:
        return None
    if grain == "unknown":
        return ("not_applicable", f"{UNKNOWN_GRAIN}, so there are no repeats to tell apart.")
    if grain != "repeated":
        return ("not_applicable", "Each unit appears once, so there are no repeats to tell apart.")
    # BLUEPRINT §14.1 (the readings ledger): the repeat kind is skipped only on a settled reading.
    # The structure stage's stated reading is medium at most, so it is the question's proposal,
    # never its answer (the gate: recalls numbered across an ``assessment``'s occasions).
    from turbotab.core.readings import repeat_kind_reading

    units = _get(structure, "units") or {}
    found = repeat_kind_reading(state, structure if isinstance(structure, Mapping) else None)
    if (found is not None and found.settled
            and _get(units, "column") == _get(state.grain, "id_column")):
        return ("skipped", _skip_reason(found.evidence))
    return None


def _unit_gate(state: Any, structure: Any) -> Gate:
    grain = _grain_answer(state, structure)
    if grain == "unknown":
        return ("not_applicable", f"{UNKNOWN_GRAIN}, so each row is analyzed as it is.")
    if grain == "one_row_per_unit":
        return ("not_applicable", "Each unit appears once, so each row already is one.")
    return None


def _aggregation_gate(state: Any, structure: Any) -> Gate:
    grain = _grain_answer(state, structure)
    if grain == "unknown":
        return ("not_applicable", f"{UNKNOWN_GRAIN}, so nothing is combined.")
    if grain == "one_row_per_unit":
        return ("not_applicable", "Each unit appears once, so there is nothing to combine.")
    if grain == "repeated" and state.unit == "row":
        return ("not_applicable", "Records stay as they are, so nothing is combined.")
    return None


def _temporal_gate(state: Any, structure: Any) -> Gate:
    from turbotab.core.stages.working import effective_repeat_kind

    grain = _grain_answer(state, structure)
    if grain == "unknown":
        return ("not_applicable", f"{UNKNOWN_GRAIN}, so no rows are read as time points.")
    if grain == "one_row_per_unit":
        return ("not_applicable", "Each unit appears once, so no row comes later than another.")
    if grain is None:
        return None
    kind = effective_repeat_kind(state, structure if isinstance(structure, Mapping) else None)
    if kind == "repeats":
        return ("not_applicable", "The rows are repeats of one measurement, not different time points.")
    if kind == "imputed_copies":  # audit I18 (WP18)
        return ("not_applicable", "The rows are imputed copies of one record, not time points.")
    if state.unit == "unit":
        return ("not_applicable", "Each unit's rows are combined, so no time points stay as rows.")
    return None


def _survey_gate(state: Any) -> Gate:
    """Asked under inference when a column reads as a survey weight (audit ME-06, WP10)."""
    from turbotab.core.survey import not_applicable_reason

    reason = not_applicable_reason(state)
    return ("not_applicable", reason) if reason else None


def _open_seal_gate(state: Any) -> Gate:
    split = state.split
    if split is not None and float(_get(split, "holdout") or 0) == 0:
        return ("not_applicable",
                "Every row trains under cross-validation alone, so no held-out rows wait to be opened.")
    return None


def route(
    state: Any,
    stages: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    records: Sequence[Any] = (),
    *,
    deps: Mapping[str, Sequence[str]] | None = None,
    energy_bearing: Callable[[str], bool] | None = None,
    ask: AskContext | None = None,
) -> list[InterviewStep]:
    """The interview, in asking order.

    ``stages``: stage name -> StageStatus (or its dict). ``artifacts``: the fresh public artifacts
    the Router reads — ``target_info`` (the task skip, the event's binary outcome), ``oriented``
    (the shape reading orientation fires on) and ``structure`` (the stated repeats reading).
    ``records``: the decision log, for each answered step's ``decision_id``. WP17: ``roles`` (the
    roles stage's proposals) tells the cluster question which columns read as groups. ``ask``:
    what the open question's ask card may read (the table's summaries and store); without it the
    card reads the state and the artifacts alone.
    """
    from turbotab.core import causal, estimand, time_varying
    from turbotab.core.methods import exposure_form, interaction

    if energy_bearing is None:
        from turbotab.core.stages.rows import energy_bearing as bearing
    else:
        bearing = energy_bearing
    artifacts = artifacts or {}
    pending = pending_stages(stages, deps)
    writers = _live_writer(records, state)
    target_info = artifacts.get("target_info")
    oriented = artifacts.get("oriented")
    structure = artifacts.get("structure")
    # FORM (the repair of the Router race): the form question reads only the card computed for the
    # present answers. The card last computed answered for other answers (a confirmed reading that
    # makes `bmi` a continuous confounder, a changed row set that re-derives k), so while the
    # fresh one computes the question waits on it and no later question is answered meanwhile.
    # Only when the card's stage failed or was cancelled does the card last computed stand in.
    forms_status = _get(stages.get("forms"), "status")
    forms_card = artifacts.get("forms")
    if forms_card is None and (forms_status == "error"
                               or _get(stages.get("forms"), "cancelled", False)):
        forms_card = artifacts.get("forms_shown")
    gates: dict[str, Callable[[], Gate]] = {
        "orientation": lambda: _orientation_gate(state, oriented),
        "event": lambda: _event_gate(state, target_info),
        "grain": lambda: _grain_gate(state, structure),
        "repeat_kind": lambda: _repeat_kind_gate(state, structure),
        "unit": lambda: _unit_gate(state, structure),
        "aggregation": lambda: _aggregation_gate(state, structure),
        "temporal": lambda: _temporal_gate(state, structure),
        "survey": lambda: _survey_gate(state),
        "open_seal": lambda: _open_seal_gate(state),
        # WP17 (turbotab/core/estimand.py)
        "follow_up": lambda: estimand.follow_up_gate(state, target_info),
        "clusters": lambda: estimand.clusters_gate(state, artifacts.get("roles")),
        "estimand": lambda: estimand.estimand_gate(state),
        "adjustment": lambda: estimand.adjustment_gate(state),
        # The causal lane (turbotab/core/causal.py): never under prediction; else stated, its
        # top-ranked estimator one step away (rung (d) ranks high when candidates are many).
        "causal": lambda: causal.causal_gate(state, artifacts.get("causal_design"),
                                             artifacts.get("time_varying")),
        # V2 causal row (turbotab/core/time_varying.py)
        "time_varying": lambda: time_varying.lane_gate(state, structure,
                                                       artifacts.get("time_varying")),
        # FORM (turbotab/core/methods/exposure_form.py, interaction.py): the form is asked after
        # the domain transforms, on the exposure's final scale; the modifiers are stated until
        # one is declared.
        "form": lambda: exposure_form.form_gate(state, forms_card),
        "modification": lambda: interaction.modification_gate(state),
    }
    # A question whose answer is not simply its slot's value (WP17): the follow-up is answered by a
    # time to event's follow-up or a yes/no outcome's "same for everyone"; the estimand while its
    # exposure is in the model; the adjustment once every covariate has answers for that exposure.
    answers: dict[str, Callable[[], Any]] = {
        "follow_up": lambda: estimand.follow_up_answer(state, target_info),
        "estimand": lambda: estimand.current_estimand(state),
        "adjustment": lambda: estimand.adjustment_answer(state),
        "causal": lambda: causal.current_causal(state),
        # Re-asked when the fresh artifact's diagnostics are not the ones its declaration came after.
        "time_varying": lambda: time_varying.lane_answer(state, artifacts.get("time_varying")),
        # FORM: answered while every column the fresh card asks about has a form declared on its
        # present scale and rows (a stale one is asked again) and no column waits for its
        # code-or-amount reading; the modifiers once every declared modifier is complete.
        "form": lambda: exposure_form.form_answer(state, forms_card),
        "modification": lambda: interaction.modification_answer(state),
    }
    writer_slots = {"follow_up": ("follow_up", "censoring")}

    steps: list[InterviewStep] = []
    first_unanswered: str | None = None
    for key in QUESTION_KEYS:
        slot = SLOT_OF.get(key, key)
        value = answers[key]() if key in answers else getattr(state, slot, None)
        decision_id = next((writers[s] for s in writer_slots.get(key, (slot,)) if s in writers), None)
        if key == "orientation" and value is not None:  # its slot turns the table whatever the lens
            steps.append(InterviewStep(key=key, status="answered", decision_id=decision_id))
            continue
        not_applicable = None
        gate = gates[key]() if key in gates else None
        if key == "energy_adjustment":
            not_applicable = _energy_applicability(state, bearing)
        elif key == "substitution":
            not_applicable = _substitution_applicability(state, bearing)
        elif gate is not None and gate[0] == "not_applicable":
            not_applicable = gate[1] or ""
        if not_applicable is not None:
            steps.append(InterviewStep(key=key, status="not_applicable", reason=not_applicable,
                                       decision_id=decision_id if value is not None else None))
            continue
        followup = None
        undecided = False
        if key == "task":
            from turbotab.core.structural import task_followup, task_followup_possible

            followup = task_followup(state, target_info)
            # While the outcome is re-read (a new task recomputes it), an answer that may still
            # need its scale or its order waits for the reading rather than standing answered.
            undecided = (followup is None and value is not None and "target_info" in pending
                         and task_followup_possible(state))
        if key == "form" and value is None and exposure_form.stale_forms(state, forms_card):
            # asked again: a domain transform left a declared form stale, or the analyzed rows
            # changed under a spline whose k the rule read on other rows
            followup = "stale"
        if key == "modification" and value is None:
            followup = interaction.modification_followup(state)
        if value is not None and followup is None and not undecided:
            steps.append(InterviewStep(key=key, status="answered", decision_id=decision_id))
            continue
        if gate is not None and gate[0] == "skipped":
            steps.append(InterviewStep(key=key, status="skipped", reason=gate[1]))
            continue
        if (key == "task" and followup is None and state.target is not None
                and target_info is not None
                and _get(target_info, "column") == state.target
                and _get(target_info, "confidence") == "high"):
            steps.append(InterviewStep(key=key, status="skipped", reason=_get(target_info, "reason")))
            continue
        own = [s for s in NEEDS.get(key, ()) if s in pending]
        if (key in ("event", "task") and state.target is not None
                and _get(target_info, "column") != state.target and "target_info" not in own
                and _get(stages.get("target_info"), "status") != "error"
                and not _get(stages.get("target_info"), "cancelled", False)):
            # The outcome is not read yet for the outcome chosen (its stage is about to start):
            # whether an event or a task is asked at all depends on that reading.
            own.append("target_info")
        if (key == "form" and artifacts.get("forms") is None and "forms" not in own
                and "forms" in stages
                and _get(stages.get("forms"), "status") != "error"
                and not _get(stages.get("forms"), "cancelled", False)):
            # FORM: the card is not read for these answers yet (about to be computed): whether
            # the question asks anything, and about which columns, is the card's.
            own.append("forms")
        fresh_stage = MUST_BE_FRESH.get(key)
        if fresh_stage and _get(stages.get(fresh_stage), "status") != "fresh" and fresh_stage not in own:
            own.append(fresh_stage)
        if first_unanswered is None:
            first_unanswered = key
            status = "waiting" if own else "open"
            card = None
            if status == "open":
                from turbotab.core.ask import card as ask_card

                card = ask_card(key, ask if ask is not None else AskContext(state, artifacts))
            steps.append(InterviewStep(key=key, status=status, waiting_on=own, followup=followup,
                                       ask=card))
        else:
            steps.append(InterviewStep(key=key, status="waiting", waiting_on=[first_unanswered, *own],
                                       followup=followup))
    return with_deferred(steps, state)


def with_deferred(steps: list[InterviewStep], state: Any) -> list[InterviewStep]:
    """Each step with the findings deferred to it (``turbotab.core.repairs.deferred_to``)."""
    from turbotab.core.repairs import deferred_to

    held = deferred_to(state)
    return [s.model_copy(update={"deferred_findings": held[s.key]}) if s.key in held else s
            for s in steps]


def first_unanswered(steps: Sequence[InterviewStep]) -> InterviewStep | None:
    """The earliest question still waiting for an answer (``open`` or ``waiting``), if any."""
    return next((s for s in steps if s.status in ("open", "waiting")), None)


__all__ = ["InterviewStep", "NEEDS", "QUESTION_KEYS", "QuestionKey", "SLOT_OF", "first_unanswered",
           "pending_stages", "route", "with_deferred"]
