"""The Confirm sweep, For the record and the triage of open noticings (SIZING P0.5).

calm/FOUNDATION §3: every quest stage ends its lines with at most one Confirm sweep, after its
Decides, holding the defaults set for the person whose alternative would change a number on this
table, each with its reason; one action, "Confirm all", clears it and counts as the stage's one
Confirm objective. A default whose alternative changes nothing here is For the record instead,
collapsed, with why. FOUNDATION §10 (the display-order rule): any answer the engine fills in is
recorded and visible, in a Confirm sweep or For the record, never quietly.

* **The would-change test** (:data:`WOULD_CHANGE`): one explicit test per default the quest log
  can state (a Router question it states, a Confirm declaration or finding, what the values
  settled), returning whether another choice would change a number on this table and, either way,
  why in one sentence. A test reads the answers as they stand; where it cannot tell (no roles yet,
  no columns known) it answers "would change", the strictest case, so nothing leaves the sweep on a
  guess. :func:`analysis_reads` is the one reading of "a column the analysis reads" they share:
  the outcome, the models' predictors (less the columns the answers leave out), the declared
  "further adjusted for" columns, a design, time or grouping column, and any column another answer
  names (an exclusion rule's, the energy model's, a modifier).
* **Confirm all** (``confirm_sweep``): one record per stage. Its lines are the server's, read from
  the quest log as it stands (:func:`sweep_lines`), each the default as stated; a line stated
  otherwise later is no longer covered, and the sweep says which (``Sweep.changed``). A sweep with
  a line still waiting for an earlier answer is not confirmed yet; one with nothing set for the
  person has nothing to confirm.
* **The triage of open noticings** (:func:`triage`; FOUNDATION §7, UNDERSTANDING_LAYER §7 ruling 1
  as re-ruled 2026-10-09): at the gate (before the plan's lock under Estimate and Describe, and
  under an unanswered purpose, the strictest case; before the held-out rows open under Predict),
  every open noticing arrives with the engine's recommended disposition and its reason: "doesn't
  change your numbers here" (a supplement line), "could bias the estimate" (a limitation sentence)
  or "act on it" (its decision). Blockers (a critical finding: it blocks or refuses part of the
  analysis) must be resolved before the triage is confirmed. The person may change any other line;
  the one ``confirm_sweep`` record (``sweep="noticings"``) holds every disposition, each with the
  recommendation and reason it was recorded on (``SweptLine.basis``), and a noticing disposed of
  there is answered by it in the quest log while the triage still states them so: one whose
  facts changed (a column now read, a finding turned critical under its id) is open again, and
  the triage says which (``Triage.changed``). Confirming again carries every disposition that
  stands, the person's own included, and is refused once the gate has passed. The noticing
  detectors themselves are P0.9's; the triage reads the findings the engine already produces.
* **For the record** (:func:`for_the_record`, ``GET /projects/{pid}/record``): each reached stage's
  collapsed lines: what was read (the ingest's facts and warnings, the profile's basis), why a
  question was not asked, the defaults that change nothing here, what the engine filled in itself
  (the records of :data:`decisions.SYSTEM_KINDS`), and what was noted with nothing to decide.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import (
    SYSTEM_KINDS,
    ConfirmSweep,
    Refusal,
    SweptLine,
    register_completion,
    sweep_key,
)
from turbotab.core.quest import (
    CONFIRM,
    DECIDE,
    QUESTIONS,
    RECORD,
    STAGE_NAMES,
    STAGES,
    SWEEP_ITEMS,
    Facts,
    QuestLine,
    QuestLog,
    QuestStage,
    ReadItem,
    ReadOption,
    Sweep,
    _get,
    _Log,
)

Change = tuple[bool, str]
WouldChange = Callable[[Any, Facts, QuestLine], Change]
Disposition = Literal["no_change", "could_bias", "act_on_it"]
DISPOSITIONS: tuple[str, ...] = ("no_change", "could_bias", "act_on_it")

# A role that puts a column in no model (``PREDICTOR_ROLES`` are the ones that do, as
# ``models/pipeline.py`` reads them; a design, time or grouping column is read by the design, the
# follow-up or the intervals).
NOT_ANALYZED_ROLES = ("identifier", "flag", "excluded")
PREDICTOR_ROLES = ("exposure", "covariate", "energy")
# The slots that say what a column is, or what was looked at, rather than what the analysis reads:
# a column named only there is read by no number. Every other slot that names a column counts it
# as read (a new slot is strict by default). ``adjustment`` and ``missing`` are read through what
# they leave out of the model (``decisions.left_out``) and the declared secondary columns.
WHAT_A_COLUMN_IS = frozenset({
    "lens", "target", "task", "purpose", "roles", "roles_unconfirmed", "role_confirmations",
    "reading_confirmations", "shape_confirmations", "sex_codings", "column_units", "categorical",
    "numbers_read", "codebooks", "adjustment", "missing", "findings", "outcome_views", "sweeps",
    "plan_locked", "seal_opened",
})


# ── what the analysis reads ──────────────────────────────────────────────────


def _names(value: Any, column: str) -> bool:
    if isinstance(value, str):
        return value == column
    if isinstance(value, Mapping):
        return any(k == column or _names(v, column) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_names(v, column) for v in value)
    return False


def analysis_reads(state: Any, column: str) -> bool:
    """Whether any number the analysis reports reads ``column`` under the answers as they stand.
    Before the roles or the purpose are answered nothing says a column is left out: True."""
    roles = _get(state, "roles")
    if not roles or _get(state, "purpose") is None:
        return True
    if column == _get(state, "target"):
        return True
    role = roles.get(column)
    if role is None or (role not in NOT_ANALYZED_ROLES and role not in PREDICTOR_ROLES):
        return True  # no role answered for it, or a design, time or grouping column
    from turbotab.core.decisions import left_out
    from turbotab.core.estimand import fixed_effects_column, secondary_columns
    from turbotab.core.models.pipeline import predictors_from_roles

    if column in predictors_from_roles(roles, _get(state, "target"), left_out(state)):
        return True
    if column in secondary_columns(state) or column == fixed_effects_column(state):
        return True
    values = state.model_dump(mode="json") if hasattr(state, "model_dump") else dict(state)
    return any(_names(v, column) for slot, v in values.items()
               if slot not in WHAT_A_COLUMN_IS and v is not None)


def _tick(columns: Sequence[str], limit: int = 4) -> str:
    from turbotab.core.voice import listing

    return listing(list(columns), limit=limit)


# ── the would-change tests ───────────────────────────────────────────────────


def _always(why: str) -> WouldChange:
    return lambda state, facts, line: (True, why)


def _exposures(state: Any) -> list[str]:
    spec = _get(state, "estimand")
    if spec is not None and _get(spec, "exposure"):
        return [str(spec.exposure)]
    return [c for c, r in (_get(state, "roles") or {}).items() if r == "exposure"]


def _modifier_changes(state: Any, facts: Facts, line: QuestLine) -> Change:
    """A modifier is another column of the table (``methods/interaction._modifier_is_declarable``),
    not an identifier: with one, the effect is reported in each of its groups."""
    roles = _get(state, "roles") or {}
    columns = list(facts.columns) or list(roles)
    if not columns:
        return True, "A modifier would report the effect in each of its groups."
    skip = {_get(state, "target"), *_exposures(state)}
    candidates = [c for c in columns if c not in skip and roles.get(c) != "identifier"]
    if candidates:
        return True, (f"A modifier such as {_tick(candidates[:1])} would report the effect in each "
                      f"of its groups.")
    return False, ("No column besides the exposure and the outcome could modify the effect, so "
                   "there is no other choice to make here.")


def _multiplicity_changes(state: Any, facts: Facts, line: QuestLine) -> Change:
    """The family is every exposure-role column (``EstimandSpec.family``): with two or more, how
    the tests are accounted for changes their p-values."""
    n = len([c for c, r in (_get(state, "roles") or {}).items() if r == "exposure"])
    if n == 1:
        return False, "The family holds one exposure, so there is no other test to account for."
    counted = f"the {n} tests" if n else "the family's tests"
    return True, f"Another way of accounting for {counted} changes their p-values and intervals."


def _readings_change(state: Any, facts: Facts, line: QuestLine) -> Change:
    return True, ("Another reading changes which columns enter the models, or the values they "
                  "enter with.")


WOULD_CHANGE: dict[str, WouldChange] = {
    # The Router's stated answers (``interview.route``'s skipped gates) that land in a Confirm.
    "grain": _always("Who counts as one person sets every count, and the intervals."),
    "repeat_kind": _always("Whether repeats are combined or kept as time points changes each "
                           "person's values."),
    "follow_up": _always("A follow-up that varies makes the outcome a time to event, which changes "
                         "the effect measure."),
    "form": _always("Another form changes how each continuous predictor enters the models."),
    "modification": _modifier_changes,
    "causal": _always("A causal estimator, such as double machine learning, reports its own "
                      "estimate of the effect."),
    # P0.6 (crosswalk disagreement 10): observational until the design is answered. A trial or a
    # matched case-control study is analyzed another way (``designs.DESIGNS``), so the estimates
    # read as observational would not be that design's.
    "design": _always("A trial or a matched case-control study is analyzed another way, so the "
                      "estimates here would not be the ones for that design."),
    # The Confirm declarations and findings (``quest.DECLARATIONS``, ``quest.EXPLORE_FINDINGS``).
    "set_multiplicity": _multiplicity_changes,
    "set_levers": _always("An in-fold rule (a spline, a variance filter, an imbalance correction) "
                          "changes what each model is fit on."),
    # P0.6 (disagreement 5): the scheme set with the draw, under Predict.
    "set_validation": _always("Another scheme (more folds, repeated folds, the bootstrap) "
                              "changes the cross-validated scores and how much they vary."),
    "low_variance": _always("Keeping or dropping the near-constant predictors changes the models' "
                            "inputs."),
    # What the values settled (``readings.read_from_data``), Your data's line.
    "read_from_data": _readings_change,
}
GENERIC = "Another choice would change the analysis."


def _test_for(line: QuestLine) -> WouldChange | None:
    if line.key in WOULD_CHANGE:
        return WOULD_CHANGE[line.key]
    if line.source == "finding":
        return next((fn for k, fn in WOULD_CHANGE.items() if line.key.startswith(k)), None)
    return None


def weigh(placed: Sequence[tuple[str, QuestLine]], state: Any, facts: Facts) -> None:
    """Each default set for the person: a Confirm when another choice would change a number here,
    with what it would change; else For the record, with why nothing would. A line its own answer
    settled is the person's, and stays as it is."""
    for _stage, line in placed:
        if line.label != CONFIRM or line.status not in ("set_for_you", "waiting"):
            continue
        test = _test_for(line)
        changes, why = test(state, facts, line) if test is not None else (True, GENERIC)
        if changes:
            line.would_change = why
        else:
            line.label, line.counted, line.changes_nothing = RECORD, False, why


# ── what the values settled: Your data's line ────────────────────────────────

READ_ORDER = 100.0  # Your data's sweep comes after its Decides, whatever its order


def reading_changes(state: Any, item: Mapping[str, Any]) -> bool:
    """A role decides whether a column enters a model at all, so another one always changes a
    number; any other reading (a unit, codes or an amount, a coding) changes one only where the
    analysis reads the column."""
    return item.get("kind") == "role" or analysis_reads(state, str(item.get("column")))


def _item(i: Mapping[str, Any]) -> ReadItem:
    return ReadItem(kind=str(i.get("kind")), column=str(i.get("column")), value=str(i.get("value")),
                    words=str(i.get("words") or ""), evidence=str(i.get("evidence") or ""),
                    options=[ReadOption(label=str(o.get("label") or ""), decision=o.get("decision"))
                             for o in i.get("change") or []])


def reading_lines(state: Any, readings: Sequence[Mapping[str, Any]] | None
                  ) -> list[tuple[str, QuestLine]]:
    """Your data's "read from your data" (crosswalk ``default:read-from-data``): one Confirm line
    for the readings another answer would change a number on, and one For the record line for the
    rest. The outcome's kind is Your question's (``default:task_settled``)."""
    items = [i for i in readings or () if i.get("kind") != "task"]
    read = [i for i in items if reading_changes(state, i)]
    quiet = [i for i in items if not reading_changes(state, i)]
    out: list[tuple[str, QuestLine]] = []
    if read:
        out.append(("data", QuestLine(
            id="default:read-from-data", key="read_from_data", source="reading", label=CONFIRM,
            name="what your data settled", status="set_for_you", counted=False, order=READ_ORDER,
            reason="These columns were settled by their values, each with its evidence and a way "
                   "to change it.", items=[_item(i) for i in read])))
    if quiet:
        columns = list(dict.fromkeys(str(i.get("column")) for i in quiet))
        out.append(("data", QuestLine(
            id="default:read-from-data", key="read_from_data:unchanged", source="reading",
            label=RECORD, name="what your data settled that changes nothing here",
            status="set_for_you", counted=False, order=READ_ORDER,
            reason="These columns were settled by their values.",
            changes_nothing=f"{_tick(columns)} {'enters' if len(columns) == 1 else 'enter'} no "
                            f"analysis, so how {'its' if len(columns) == 1 else 'their'} values "
                            f"are read changes no number.",
            items=[_item(i) for i in quiet])))
    return out


# ── the sweep and its confirmation ───────────────────────────────────────────


def default_value(line: QuestLine) -> str:
    """The default as stated, which a confirmation holds: a reading line's readings, else the
    reason it was set (a declaration's name when it states none)."""
    if line.items:
        return "; ".join(f"{i.kind}:{i.column}={i.value}" for i in line.items)
    return line.reason or line.name


def _writer(log: _Log, stage: str, sweep: str) -> str | None:
    found = next((r for r in reversed(log.live) if r.decision.kind == "confirm_sweep"
                  and r.decision.stage == stage and r.decision.sweep == sweep), None)
    return found.id if found is not None else None


def _sweep_words(n: int) -> tuple[str, str]:
    if n == 1:
        return "Here is the other choice set for you", "Confirm it"
    return f"Here are the {n} other choices set for you", f"Confirm all {n}"


def sweep_of(stage: str, lines: Sequence[QuestLine], state: Any, log: _Log) -> Sweep | None:
    """The stage's one sweep, None when nothing set for the person would change a number. A line
    still stated as the stage's "Confirm all" recorded it is answered by that record."""
    confirms = [l for l in lines if l.label == CONFIRM]
    if not confirms:
        return None
    held = (_get(state, "sweeps") or {}).get(sweep_key(stage))
    writer = _writer(log, stage, "defaults") if held is not None else None
    stated = {(l.key, l.value) for l in held.lines} if held is not None else set()
    changed = []
    for line in confirms:
        if line.status != "set_for_you":
            continue
        if (line.key, default_value(line)) in stated:
            line.status, line.decision_id = "answered", writer
        elif held is not None:
            changed.append(line.key)
    heading, action = _sweep_words(len(confirms))
    return Sweep(lines=len(confirms), answered=all(l.status == "answered" for l in confirms),
                 id=SWEEP_ITEMS.get(stage, ""), heading=heading, action=action,
                 confirmed_by=writer, changed=changed)


def sweep_lines(stage: QuestStage) -> list[SweptLine]:
    """What "Confirm all" records for ``stage`` now: each default still set for the person (or
    covered by the stage's last confirmation), as stated. A line the person answered themselves
    is theirs, not the sweep's."""
    by = stage.sweep.confirmed_by if stage.sweep is not None else None
    return [SweptLine(id=l.id, key=l.key, value=default_value(l)) for l in stage.lines
            if l.label == CONFIRM and (l.status == "set_for_you"
                                       or (by is not None and l.decision_id == by))]


# ── the triage of open noticings ─────────────────────────────────────────────

GATES = {"models": "gate:open-noticings-before-lock", "results": "gate:open-noticings-before-seal"}
DISPOSITION_WORDS = {
    "no_change": "Doesn't change your numbers here",
    "could_bias": "Could bias the estimate",
    "act_on_it": "Act on it",
}


class TriageItem(BaseModel):
    """One open noticing at the gate, or one the last confirmation disposed of: where it is
    decided (``line``, ``stage``, the ``question`` it routes to), the engine's recommended
    disposition with its plain label and reason, whether it blocks (it must be resolved before the
    triage is confirmed), and the disposition the last confirmation recorded while it still
    stands (the recommendation and reason it was recorded on are the triage's now)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    line: str
    stage: str
    summary: str
    severity: str
    columns: list[str] = []
    question: str | None = None
    recommended: Disposition
    label: str
    reason: str
    blocker: bool
    recorded: Disposition | None = None


class Triage(BaseModel):
    """The triage at the gate: its stage and crosswalk card, each noticing (blockers first), how
    many block, whether it can be confirmed now (the gate not passed, no blocker, something to
    triage), whether every noticing has a recorded disposition, and the record that holds them.
    ``changed``: the noticings whose recorded disposition no longer stands, the recommendation or
    its reason having changed since. ``passed``: the gate has passed (the plan is fixed, or under
    Predict the held-out rows are open), so the triage is read, no longer confirmed."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    version: int = 1
    stage: str
    gate: str
    items: list[TriageItem] = []
    blockers: int = 0
    confirmable: bool = False
    answered: bool = False
    confirmed_by: str | None = None
    changed: list[str] = []
    passed: bool = False


def gate_stage(state: Any) -> str:
    """Where the triage stands: before the held-out rows open under Predict (Results), else before
    the plan's lock (Models), an unanswered purpose included: the strictest case."""
    return "results" if _get(state, "purpose") == "prediction" else "models"


def gate_passed(state: Any) -> bool:
    """Whether the triage's gate has passed: the held-out rows opened under Predict, else the plan
    fixed. A disposition recorded after it would be chosen with the numbers in view."""
    return bool(_get(state, "seal_opened" if gate_stage(state) == "results" else "plan_locked"))


def recommend(state: Any, finding: Mapping[str, Any], open_questions: set[str]
              ) -> tuple[Disposition, bool, str]:
    """The engine's recommended disposition for an open finding, whether it blocks, and why.

    A critical finding blocks or refuses part of the analysis (UNDERSTANDING_LAYER §2.2, T1): it is
    acted on first. One routed to a question still to be answered is decided there. A note (info)
    changes no number, nor does a finding about columns no analysis reads. Otherwise it concerns
    what the analysis reads, and left as it is it could bias the estimate (or, under prediction,
    the score): a limitation sentence, unless the person acts on it."""
    severity = str(finding.get("severity") or "info")
    if severity == "critical":
        return "act_on_it", True, ("It blocks or refuses part of the analysis, so it is resolved "
                                   "before any estimate.")
    routed = finding.get("routes_to")
    if routed in open_questions:
        from turbotab.core.voice import question_name

        return "act_on_it", False, f"It is decided at {question_name(str(routed))}, still to come."
    if severity == "info":
        return "no_change", False, "A note about the data: no number rests on it."
    columns = [str(c) for c in finding.get("affected_columns") or []]
    read = [c for c in columns if analysis_reads(state, c)]
    what = "score" if _get(state, "purpose") == "prediction" else "estimate"
    if columns and not read:
        return "no_change", False, f"It concerns only {_tick(columns)}, which no analysis reads."
    about = f"It concerns {_tick(read)}, which the analysis reads" if read else \
        "It concerns the whole table, which every number reads"
    return "could_bias", False, f"{about}; left as it is, it could bias the {what}."


def _open_questions(lines: Iterable[QuestLine]) -> set[str]:
    return {l.key for l in lines if l.source == "question" and l.status in ("open", "waiting")}


def basis(disposition: str, reason: str) -> str:
    """What a disposition is recorded on: the engine's recommendation and its reason then."""
    return f"{disposition}: {reason}"


def _stands(held: SweptLine | None, recommended: tuple[Disposition, bool, str]) -> bool:
    """A recorded disposition stands while the triage still recommends what, and why, it did
    when it was recorded; never on a blocker, which is resolved by its own decision."""
    disposition, blocker, reason = recommended
    return held is not None and not blocker and held.basis == basis(disposition, reason)


def _held_noticings(state: Any) -> dict[str, SweptLine]:
    held = (_get(state, "sweeps") or {}).get(sweep_key(gate_stage(state), "noticings"))
    return {l.key: l for l in held.lines} if held is not None else {}


def triage(state: Any, log: QuestLog, findings: Any, records: Sequence[Any] = ()) -> Triage:
    """Every noticing that feeds the plan (each finding still open where it is decided, and each
    the last confirmation disposed of) with the engine's recommended disposition and the one
    recorded, while it stands. ``findings``: the findings artifact as served."""
    stage = gate_stage(state)
    by_id = {str(f.get("id")): f for f in _get(findings, "findings") or []}
    lines = {l.key: (s.key, l) for s in log.stages for l in s.lines if l.source == "finding"}
    open_questions = _open_questions(l for s in log.stages for l in s.lines)
    held = _held_noticings(state)
    writer = _writer(_Log(list(records)), stage, "noticings") if held else None
    items, changed = [], []
    for fid, (where, line) in lines.items():
        covered = (line.status == "answered" and fid in held and writer is not None
                   and line.decision_id == writer)
        if not (line.status == "open" or covered) or line.label != DECIDE or fid not in by_id:
            continue
        finding = by_id[fid]
        recommended = recommend(state, finding, open_questions)
        disposition, blocker, reason = recommended
        stands = _stands(held.get(fid), recommended)
        if fid in held and not stands:
            changed.append(fid)
        label = DISPOSITION_WORDS[disposition]
        if disposition == "could_bias" and _get(state, "purpose") == "prediction":
            label = "Could bias the score"
        items.append(TriageItem(
            id=fid, line=line.id if line.id.startswith("finding:") else f"finding:{fid}",
            stage=where, summary=line.name, severity=str(finding.get("severity") or "info"),
            columns=[str(c) for c in finding.get("affected_columns") or []],
            question=finding.get("routes_to") if finding.get("routes_to") in QUESTIONS else None,
            recommended=disposition, label=label, reason=reason, blocker=blocker,
            recorded=held[fid].value if stands else None))
    items.sort(key=lambda i: not i.blocker)
    blockers = sum(i.blocker for i in items)
    passed = gate_passed(state)
    return Triage(stage=stage, gate=GATES[stage], items=items, blockers=blockers,
                  confirmable=bool(items) and not blockers and not passed,
                  answered=not blockers and all(i.recorded is not None for i in items),
                  confirmed_by=writer, changed=changed, passed=passed)


def cover_noticings(placed: Sequence[tuple[str, QuestLine]], state: Any, log: _Log,
                    findings: Any = None) -> None:
    """A noticing the triage disposed of (it changes no number here, or it is left as a
    limitation) is answered by the triage's record while that disposition stands (the triage
    still recommends what, and why, it did then); one to act on stays open for its decision.
    A noticing that blocks, or is no longer among ``findings``, is never answered by it."""
    held = _held_noticings(state)
    if not held:
        return
    by_id = {str(f.get("id")): f for f in _get(findings, "findings") or []}
    open_questions = _open_questions(l for _s, l in placed)
    writer = _writer(log, gate_stage(state), "noticings")
    for _stage, line in placed:
        if line.source != "finding" or line.status != "open" or line.key not in by_id:
            continue
        mine = held.get(line.key)
        if mine is None or mine.value not in ("no_change", "could_bias"):
            continue
        if _stands(mine, recommend(state, by_id[line.key], open_questions)):
            line.status, line.decision_id = "answered", writer


# ── "Confirm all": the record ────────────────────────────────────────────────


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    return ctx.get(name) if isinstance(ctx, Mapping) else getattr(ctx, name, None)


def _confirm_the_defaults(d: ConfirmSweep, log: QuestLog) -> ConfirmSweep:
    stage = next(s for s in log.stages if s.key == d.stage)
    name = STAGE_NAMES[d.stage]
    confirms = [l for l in stage.lines if l.label == CONFIRM]
    waiting = [l for l in confirms if l.status == "waiting"]
    if waiting:
        names = [w.name for l in waiting for w in l.waiting_for] or [l.name for l in waiting]
        raise Refusal(
            "sweep_waits",
            f"Some choices set for you in {name} wait for {', '.join(dict.fromkeys(names))}; "
            f"they are confirmed together once it is answered.",
            exits=[{"label": f"Answer {names[0]}", "decision": None}])
    current = sweep_lines(stage)
    if not current:
        raise Refusal(
            "nothing_to_confirm",
            f"{name} holds no choice set for you whose alternative would change a number here"
            + (", other than those you answered yourself" if confirms else "")
            + ", so there is nothing to confirm.",
            exits=[{"label": f"Continue past {name}", "decision": None}])
    if d.lines and sorted(d.lines, key=lambda l: l.key) != sorted(current, key=lambda l: l.key):
        shown = {(l.key, l.value) for l in d.lines}
        moved = [l.key for l in current if (l.key, l.value) not in shown]
        raise Refusal(
            "sweep_changed",
            f"The choices set for you in {name} changed since they were shown"
            + (f" ({', '.join(moved)})" if moved else "") + ". Confirm them as they stand now.",
            exits=[{"label": "Confirm them as they stand",
                    "decision": d.model_copy(update={"lines": current}).model_dump(mode="json")}])
    return d.model_copy(update={"lines": current})


def _confirm_the_triage(d: ConfirmSweep, t: Triage) -> ConfirmSweep:
    name = STAGE_NAMES[t.stage]
    if t.passed:
        raise Refusal(
            "gate_passed",
            "The open noticings are triaged before "
            + ("the held-out rows open" if t.stage == "results" else "the plan is fixed")
            + ", and that has happened: each is decided on its own now, as made after the "
            + ("held-out scores were seen." if t.stage == "results" else "estimates were seen."),
            exits=[{"label": f"Decide: {i.summary}", "decision": None}
                   for i in t.items if i.recorded is None])
    if d.stage != t.stage:
        raise Refusal(
            "not_the_gate",
            f"The open noticings are triaged in {name}, before "
            + ("the held-out rows open." if t.stage == "results" else "the plan is fixed."),
            exits=[{"label": f"Triage them in {name}",
                    "decision": d.model_copy(update={"stage": t.stage}).model_dump(mode="json")}])
    if t.blockers:
        blocking = [i for i in t.items if i.blocker]
        raise Refusal(
            "blocker_open",
            f"{len(blocking)} noticing{'s' if len(blocking) > 1 else ''} must be resolved before "
            f"the rest are triaged: {'; '.join(i.summary for i in blocking)}.",
            exits=[{"label": f"Decide: {i.summary}", "decision": None} for i in blocking])
    if not t.items:
        raise Refusal("nothing_to_confirm", "No noticing is open, so there is nothing to triage.",
                      exits=[{"label": "Continue", "decision": None}])
    given = {l.key: l.value for l in d.lines}
    stray = [k for k in given if k not in {i.id for i in t.items}]
    if stray:
        raise Refusal(
            "not_an_open_noticing",
            f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} not an open noticing here.",
            exits=[{"label": "Triage the open noticings as they stand",
                    "decision": d.model_copy(update={"lines": []}).model_dump(mode="json")}])
    unknown = [k for k, v in given.items() if v not in DISPOSITIONS]
    if unknown:
        raise Refusal(
            "unknown_disposition",
            "A noticing is triaged as one of: doesn't change your numbers here, could bias the "
            "estimate, or act on it.",
            exits=[{"label": "Keep the recommended dispositions",
                    "decision": d.model_copy(update={"lines": []}).model_dump(mode="json")}])
    # Each noticing as the person set it now, else as recorded while that stands, else as
    # recommended: a disposition the person chose is never replaced by the engine's.
    lines = [SweptLine(id=i.line, key=i.id, value=given.get(i.id) or i.recorded or i.recommended,
                       basis=basis(i.recommended, i.reason)) for i in t.items]
    return d.model_copy(update={"lines": lines})


def _confirm_reads_the_sweep(d: ConfirmSweep, ctx: Any) -> ConfirmSweep:
    """The lines are the server's: read from the quest log (the defaults) or the triage (the
    noticings) as the project stands. Without them (a replay) the record is taken as written."""
    if d.sweep == "noticings":
        reader = _ctx(ctx, "triage")
        t = reader() if callable(reader) else None
        return _confirm_the_triage(d, t) if t is not None else d
    reader = _ctx(ctx, "quest")
    log = reader() if callable(reader) else None
    return _confirm_the_defaults(d, log) if log is not None else d


register_completion("confirm_sweep", _confirm_reads_the_sweep)


def _plain(value: str) -> str:
    text = value.strip().rstrip(".")
    return text[:1].lower() + text[1:] if text[:1].isupper() and text[1:2].islower() else text


def confirm_sentence(d: Any, state: Any = None, ctx: Any = None) -> str:
    """The record quotes what was confirmed, as it was stated (FOUNDATION §10: what the engine set
    is recorded and visible)."""
    name = STAGE_NAMES.get(d.stage, d.stage)
    if d.sweep == "noticings":
        counts = {k: sum(l.value == k for l in d.lines) for k in DISPOSITIONS}
        said = [(counts["could_bias"], "left as a limitation"),
                (counts["no_change"], "noted in the supplement as changing no number here"),
                (counts["act_on_it"], "to act on")]
        return (f"The open noticings were triaged in {name}: "
                + "; ".join(f"{n} {words}" for n, words in said if n) + ".")
    if not d.lines:
        return f"Confirmed the choices set for you in {name}."
    stated = "; ".join(_plain(l.value) for l in d.lines if not l.key.startswith("read_from_data"))
    read = next((l for l in d.lines if l.key == "read_from_data"), None)
    if read is not None:
        n = len(read.value.split("; "))
        stated = "; ".join(s for s in (stated, f"{n} reading{'s' if n != 1 else ''} the values "
                                                f"settled") if s)
    count = len(d.lines)
    what = "the choice" if count == 1 else f"the {count} choices"
    return f"Confirmed {what} set for you in {name}, as stated: {stated}."


def _register_sentence() -> None:
    from turbotab.core.voice import register_sentence

    register_sentence("confirm_sweep")(confirm_sentence)


_register_sentence()


# ── For the record ───────────────────────────────────────────────────────────

RecordKind = Literal["ingest", "profile", "not_applicable", "set_for_you", "filled", "noted"]


class RecordLine(BaseModel):
    """One collapsed line: ``ingest`` (what was read and its warnings), ``profile`` (what the
    column summaries read), ``not_applicable`` (why a question was not asked), ``set_for_you`` (a
    default no other choice changes a number on here, with why), ``filled`` (an answer the engine
    recorded itself), ``noted`` (what was looked at or noticed with nothing to decide)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    kind: RecordKind
    key: str
    id: str | None = None
    text: str
    decision_id: str | None = None


class RecordStage(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: str
    name: str
    lines: list[RecordLine] = []


class ForTheRecord(BaseModel):
    """Each stage's For the record lines (FOUNDATION §3: collapsed, never counted). A stage not
    reached yet has none: what it would say may still change."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    version: int = 1
    stages: list[RecordStage]


def _sentence(*parts: str | None) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def for_the_record(log: QuestLog, steps: Sequence[Any], records: Sequence[Any], *,
                   ingest: Any = None, profile: Any = None) -> ForTheRecord:
    """``log``: the quest log now; ``steps``: the Router's; ``records``: the decision log;
    ``ingest`` and ``profile``: those stages' newest artifacts (None before they exist)."""
    from turbotab.core.quest import record_stage

    reached = {s.key for s in log.stages if s.reached}
    out: dict[str, list[RecordLine]] = {key: [] for key, _ in STAGES}
    ingest, profile = getattr(ingest, "data", ingest), getattr(profile, "data", profile)
    if ingest is not None:
        n_rows, n_cols = _get(ingest, "n_rows"), _get(ingest, "n_cols")
        if n_rows is not None and n_cols is not None:
            fingerprint = str(_get(ingest, "fingerprint") or "")
            out["data"].append(RecordLine(
                kind="ingest", key="facts", id="record:ingest-facts",
                text=f"Read {int(n_rows):,} rows and {int(n_cols):,} columns"
                     + (f"; the file's fingerprint is {fingerprint[:12]}." if fingerprint
                        else ".")))
        for i, warning in enumerate(_get(ingest, "warnings") or []):
            out["data"].append(RecordLine(kind="ingest", key=f"warning:{i}",
                                          id="noticing:ingest-warnings", text=str(warning)))
    basis = _get(profile, "basis")
    if basis:
        out["data"].append(RecordLine(kind="profile", key="basis", id="record:profile",
                                      text=str(basis)))
    for step in steps:
        key = _get(step, "key")
        if _get(step, "status") == "not_applicable" and key in QUESTIONS and _get(step, "reason"):
            place = QUESTIONS[key]
            out[place.stage].append(RecordLine(kind="not_applicable", key=key, id=place.item,
                                               text=str(_get(step, "reason"))))
    for stage in log.stages:
        for line in stage.lines:
            if line.label != RECORD:
                continue
            if line.status == "set_for_you":
                out[stage.key].append(RecordLine(
                    kind="set_for_you", key=line.key, id=line.id,
                    text=_sentence(line.reason, line.changes_nothing) or line.name))
            else:
                out[stage.key].append(RecordLine(kind="noted", key=line.key, id=line.id,
                                                 text=line.name, decision_id=line.decision_id))
    live = _Log(list(records))
    for record in live.live:
        if record.decision.kind in SYSTEM_KINDS:
            where = record_stage(record, records)
            if where is not None:
                out[where].append(RecordLine(
                    kind="filled", key=record.id, id=None,
                    text=record.sentence or record.decision.kind, decision_id=record.id))
    return ForTheRecord(stages=[RecordStage(key=key, name=name,
                                            lines=out[key] if key in reached else [])
                                for key, name in STAGES])


__all__ = [
    "DISPOSITIONS", "DISPOSITION_WORDS", "ForTheRecord", "GATES", "NOT_ANALYZED_ROLES",
    "RecordLine", "RecordStage", "Triage", "TriageItem", "WHAT_A_COLUMN_IS", "WOULD_CHANGE",
    "analysis_reads", "basis", "cover_noticings", "default_value", "for_the_record", "gate_passed",
    "gate_stage",
    "reading_changes", "reading_lines", "recommend", "sweep_lines", "sweep_of", "triage", "weigh",
]
