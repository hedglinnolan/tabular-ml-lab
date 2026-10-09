"""P0.5 · the Confirm sweep, For the record and the triage of open noticings
(``turbotab/core/sweep.py``, read by ``turbotab/core/quest.py``).

The expectations come from the rulings, not from the module: a stage's Confirm sweep holds only the
defaults set for the person whose alternative would change a number on this table, last in the
stage, and counts as one objective (calm/FOUNDATION §3); a default that changes nothing here is For
the record, with why; "Confirm all" is one record, and a default that changes after it is confirmed
is no longer covered (the display-order rule, FOUNDATION §10: nothing quietly filled in); every open
noticing at the gate arrives with a recommended disposition, and blockers must be resolved first
(UNDERSTANDING_LAYER §7, re-ruled 2026-10-09).
"""
from __future__ import annotations

import pytest

from turbotab.core import decisions, quest, sweep
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.tests.test_stage_registry import line, record, stage, steps_until

STATED = ("grain", "clusters", "modification", "causal")
NOT_ASKED = ("orientation", "event", "follow_up", "repeat_kind", "unit", "aggregation",
             "temporal", "survey", "time_varying", "energy_adjustment", "substitution", "open_seal")
ROLES = {"SEQN": "identifier", "sugar": "exposure", "age": "covariate", "bp_sys": "excluded"}
COLUMNS = ["SEQN", "sugar", "age", "bp_sys", "glucose"]


def planned(**update) -> ProjectState:
    base = dict(lens=["dietary"], target="glucose", purpose="inference", roles=dict(ROLES))
    return ProjectState(**{**base, **update})


def steps(*, stated=STATED, not_applicable=NOT_ASKED):
    return steps_until(None, stated=stated, not_applicable=not_applicable)


def reading(kind, column, value):
    return {"kind": kind, "column": column, "value": value, "words": f"is {value}",
            "evidence": f"{column}'s values", "change": [{"label": "another reading",
                                                         "decision": None}]}


READINGS = [reading("role", "SEQN", "identifier"),
            reading("task", "glucose", "regression"),
            reading("code_or_count", "sugar", "amount"),
            reading("code_or_count", "bp_sys", "amount")]


def log_of(state, records=(), the_steps=None, **kw):
    kw.setdefault("columns", COLUMNS)
    return quest.quest_log(state, list(records), the_steps or steps(), **kw)


# ── the sweep: what it holds, and where ──────────────────────────────────────


def test_the_sweep_holds_the_defaults_that_would_change_a_number_last_in_the_stage():
    log = log_of(planned(), readings=READINGS)
    whos_in, models, data = stage(log, "whos_in"), stage(log, "models"), stage(log, "data")
    # Who's in: who counts as one person is set for you (each row a person), and another answer
    # would recount every person; no column can group the rows, so nothing else could be offered.
    assert [l.key for l in whos_in.lines if l.label == "Confirm"] == ["grain"]
    assert line(log, "clusters").label == "For the record"
    # Models: no modifier declared, while `age` could modify the effect; the primary model alone,
    # with the causal estimators one step away. Both would change a number.
    assert [l.key for l in models.lines if l.label == "Confirm"] == ["modification", "causal"]
    for key in ("grain", "modification", "causal"):
        assert line(log, key).would_change and line(log, key).reason
    # Last in the stage: every Confirm after every Decide, For the record after them.
    for s in log.stages:
        ranks = [("Decide", "Confirm", "For the record").index(l.label) for l in s.lines]
        assert ranks == sorted(ranks), s.key
    assert whos_in.sweep.lines == 1 and not whos_in.sweep.answered
    assert whos_in.sweep.id == "other:confirm-sweep:whos_in"
    assert (models.sweep.heading, models.sweep.action) == (
        "Here are the 2 other choices set for you", "Confirm all 2")
    assert (whos_in.sweep.heading, whos_in.sweep.action) == (
        "Here is the other choice set for you", "Confirm it")
    # Your data: one line for what the values settled. `bp_sys` enters no analysis (left out), so
    # how its values are read changes no number; the outcome's kind is Your question's.
    [read] = [l for l in data.lines if l.label == "Confirm"]
    assert read.id == "default:read-from-data"
    assert [(i.kind, i.column) for i in read.items] == [("role", "SEQN"),
                                                        ("code_or_count", "sugar")]
    [quiet] = [l for l in data.lines if l.key == "read_from_data:unchanged"]
    assert quiet.label == "For the record" and [i.column for i in quiet.items] == ["bp_sys"]
    assert "`bp_sys`" in quiet.changes_nothing


def test_a_stage_whose_defaults_change_nothing_here_has_no_sweep():
    # Only the exposure and the outcome: no column could modify the effect, and a family of one
    # exposure has no other test to account for.
    roles = {"id": "identifier", "sugar": "exposure"}
    state = planned(roles=roles, estimand=decisions.EstimandSpec(family=True, measure="mean_difference"))
    log = log_of(state, columns=["id", "sugar", "glucose"],
                 the_steps=steps(stated=("grain", "clusters", "modification"),
                                 not_applicable=NOT_ASKED + ("causal",)))
    models = stage(log, "models")
    assert models.sweep is None
    assert [l.label for l in models.lines if l.key in ("modification", "set_multiplicity")] == [
        "For the record", "For the record"]
    assert line(log, "modification").changes_nothing
    assert models.progress.required == len([l for l in models.lines
                                            if l.label == "Decide" and l.counted])
    # With a second exposure in the family, how the tests are counted changes their p-values.
    two = state.model_copy(update={"roles": {**roles, "fat": "exposure"}})
    log = log_of(two, columns=["id", "sugar", "fat", "glucose"],
                 the_steps=steps(stated=("grain", "clusters", "modification"),
                                 not_applicable=NOT_ASKED + ("causal",)))
    assert [l.key for l in stage(log, "models").lines if l.label == "Confirm"] == [
        "set_multiplicity"]


def test_an_unanswered_purpose_reads_every_column_as_analyzed():
    # Before the roles and the purpose, nothing says a column is left out: the strictest case.
    log = log_of(ProjectState(lens=["dietary"], target="glucose"), readings=READINGS,
                 the_steps=steps_until("purpose"))
    [read] = [l for l in stage(log, "data").lines if l.label == "Confirm"]
    assert [i.column for i in read.items] == ["SEQN", "sugar", "bp_sys"]
    assert not [l for l in stage(log, "data").lines if l.key == "read_from_data:unchanged"]


def test_every_default_the_quest_log_can_state_has_a_would_change_test():
    # The questions the Router states today (``interview.route``'s skipped gates) that land in a
    # Confirm, every Confirm declaration and finding, and what the values settled.
    stated = {"grain", "repeat_kind", "follow_up", "form", "modification", "causal"}
    assert not {k for k in stated if quest.STATED.get(k) == "For the record"}
    confirms = {d.kind for d in quest.DECLARATIONS if d.place.label == "Confirm"}
    findings = {k for k, p in quest.EXPLORE_FINDINGS.items() if p.label == "Confirm"}
    assert stated | confirms | findings | {"read_from_data"} <= set(sweep.WOULD_CHANGE)


# ── Confirm all: one record, replayed ────────────────────────────────────────


def confirm(seq, stage_key, lines, sweep_kind="defaults"):
    return record(seq, {"kind": "confirm_sweep", "stage": stage_key, "sweep": sweep_kind,
                        "lines": lines})


def swept(log, key):
    return [l.model_dump() for l in sweep.sweep_lines(stage(log, key))]


def test_confirm_all_is_one_record_that_answers_the_sweep_and_replays():
    state = planned()
    before = log_of(state)
    lines = swept(before, "whos_in")
    assert lines == [{"id": "q:grain", "key": "grain", "value": "stated"}]
    records = [record(1, {"kind": "set_lens", "lenses": ["dietary"]}), confirm(2, "whos_in", lines)]
    state = planned(sweeps=decisions.fold(records).sweeps)
    after = log_of(state, records)
    whos_in = stage(after, "whos_in")
    assert whos_in.sweep.answered and whos_in.sweep.confirmed_by == "r2"
    assert (line(after, "grain").status, line(after, "grain").decision_id) == ("answered", "r2")
    assert whos_in.progress.answered == stage(before, "whos_in").progress.answered + 1
    assert not stage(after, "models").sweep.answered  # one sweep per stage
    # The fold of the log is the replay: the confirmation is in the state it folds to.
    assert decisions.fold(records).sweeps["whos_in"].lines[0].key == "grain"
    # A default stated otherwise since is no longer covered, and says so.
    changed = [s.model_copy(update={"reason": "every `SEQN` appears once"}) if s.key == "grain"
               else s for s in steps()]
    again = log_of(state, records, the_steps=changed)
    assert not stage(again, "whos_in").sweep.answered
    assert stage(again, "whos_in").sweep.changed == ["grain"]
    # Undoing the confirmation reopens the sweep.
    undone = [*records, record(3, {"kind": "revert", "decision_id": "r2"})]
    state = planned(sweeps=decisions.fold(undone).sweeps)
    assert not stage(log_of(state, undone), "whos_in").sweep.answered
    assert quest.record_stage(undone[1], undone) == "whos_in"
    assert quest.record_stage(undone[2], undone) == "whos_in"


class Ctx:
    """What the server tells a validator: the quest log and the triage as the project stands."""

    def __init__(self, log, triage=None):
        self.quest = lambda: log
        self.triage = lambda: triage


def test_confirm_all_records_exactly_what_was_shown_and_refuses_an_empty_or_stale_sweep():
    log = log_of(planned())
    d = decisions.validate({"kind": "confirm_sweep", "stage": "models"}, Ctx(log))
    assert [l.key for l in d.lines] == ["modification", "causal"]
    with pytest.raises(Refusal) as nothing:
        decisions.validate({"kind": "confirm_sweep", "stage": "first_look"}, Ctx(log))
    assert nothing.value.code == "nothing_to_confirm"
    stale = [{"id": "q:grain", "key": "grain", "value": "an older reading"}]
    with pytest.raises(Refusal) as changed:
        decisions.validate({"kind": "confirm_sweep", "stage": "whos_in", "lines": stale}, Ctx(log))
    assert changed.value.code == "sweep_changed"
    assert changed.value.exits[0]["decision"]["lines"] == swept(log, "whos_in")


def test_a_sweep_with_a_line_still_waiting_cannot_be_confirmed():
    state = ProjectState(lens=["metabolomics"], target="outcome", purpose="prediction",
                         roles={"m_001": "exposure"})
    log = quest.quest_log(state, [], steps_until("models"), columns=["m_001", "outcome"])
    assert line(log, "set_levers").status == "waiting"
    with pytest.raises(Refusal) as waits:
        decisions.validate({"kind": "confirm_sweep", "stage": "models"}, Ctx(log))
    assert waits.value.code == "sweep_waits"


# ── the triage of open noticings ─────────────────────────────────────────────

FINDINGS = {"findings": [
    {"id": "batch_confounded__run", "severity": "critical", "summary": "Run is the outcome",
     "affected_columns": ["run"], "routes_to": None, "repairs": [{"key": "x"}],
     "answered_by": None},
    {"id": "binary_text__age", "severity": "warning", "summary": "Age reads as text",
     "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "map"}],
     "answered_by": None},
    {"id": "outliers__bp_sys", "severity": "warning", "summary": "Extreme blood pressure",
     "affected_columns": ["bp_sys"], "routes_to": None, "repairs": [{"key": "cap"}],
     "answered_by": None},
    {"id": "note__SEQN", "severity": "info", "summary": "SEQN is an identifier",
     "affected_columns": ["SEQN"], "routes_to": None, "repairs": [{"key": "keep"}],
     "answered_by": None},
    {"id": "settled", "severity": "warning", "summary": "Already decided",
     "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "map"}],
     "answered_by": "r1"},
]}


def test_each_open_noticing_arrives_with_a_recommended_disposition_and_blockers_first():
    state = planned()
    log = log_of(state, findings=FINDINGS)
    t = sweep.triage(state, log, FINDINGS)
    assert (t.stage, t.gate) == ("models", "gate:open-noticings-before-lock")
    got = {i.id: (i.recommended, i.blocker) for i in t.items}
    assert got == {
        "batch_confounded__run": ("act_on_it", True),  # critical: it must be resolved first
        "binary_text__age": ("could_bias", False),  # age is in the model
        "outliers__bp_sys": ("no_change", False),  # bp_sys is left out of every model
        "note__SEQN": ("no_change", False),  # a note, nothing to act on
    }
    assert [i.id for i in t.items][0] == "batch_confounded__run"
    assert t.blockers == 1 and not t.confirmable
    assert all(i.reason and i.label for i in t.items)
    with pytest.raises(Refusal) as blocked:
        decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"},
                           Ctx(log, t))
    assert blocked.value.code == "blocker_open"
    # Under Predict the gate is before the held-out rows open, in Results.
    predict = sweep.triage(state.model_copy(update={"purpose": "prediction"}), log, FINDINGS)
    assert (predict.stage, predict.gate) == ("results", "gate:open-noticings-before-seal")
    # An unanswered purpose is the strictest case: the gate before the lock.
    assert sweep.triage(state.model_copy(update={"purpose": None}), log, FINDINGS).stage == "models"
    # A noticing routed to a question still to come is decided there.
    routed = {"id": "implausible", "severity": "warning", "affected_columns": ["sugar"],
              "routes_to": "exclusions"}
    assert sweep.recommend(state, routed, {"exclusions"})[:2] == ("act_on_it", False)
    assert sweep.recommend(state, routed, set())[:2] == ("could_bias", False)


def test_the_triage_records_every_disposition_and_the_disposed_noticings_are_answered():
    resolved = {"findings": [f for f in FINDINGS["findings"] if f["severity"] != "critical"]}
    state = planned()
    log = log_of(state, findings=resolved)
    t = sweep.triage(state, log, resolved)
    assert t.blockers == 0 and t.confirmable
    # The person changes one line: SEQN's note is acted on instead.
    d = decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                            "lines": [{"id": "finding:note__SEQN", "key": "note__SEQN",
                                       "value": "act_on_it"}]}, Ctx(log, t))
    assert {l.key: l.value for l in d.lines} == {
        "binary_text__age": "could_bias", "outliers__bp_sys": "no_change",
        "note__SEQN": "act_on_it"}
    records = [record(1, d.model_dump(mode="json"))]
    state = planned(sweeps=decisions.fold(records).sweeps)
    after = log_of(state, records, findings=resolved)
    assert line(after, "binary_text__age").status == "answered"
    assert line(after, "outliers__bp_sys").decision_id == "r1"
    assert line(after, "note__SEQN").status == "open"  # to act on: still its own decision
    assert stage(after, "models").sweep is not None  # the defaults' sweep is a separate objective
    with pytest.raises(Refusal) as unsound:
        decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                            "lines": [{"id": "finding:x", "key": "x", "value": "no_change"}]},
                           Ctx(log, t))
    assert unsound.value.code == "not_an_open_noticing"


# ── For the record ───────────────────────────────────────────────────────────


def test_for_the_record_lists_what_was_read_or_set_with_no_choice_that_matters_here():
    state = planned()
    records = [record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
               record(2, {"kind": "lock_plan", "plan": {}, "digest": "ab" * 32})]
    the_steps = steps()
    log = log_of(state, records, the_steps=the_steps, readings=READINGS)
    out = sweep.for_the_record(
        log, the_steps, records,
        ingest={"warnings": ["the file is not valid UTF-8; it was read as Latin-1"]},
        profile={"basis": "Column summaries and lens hints read all 5 rows."})
    by = {s.key: s for s in out.stages}
    kinds = {s.key: [(l.kind, l.key) for l in s.lines] for s in out.stages}
    assert ("ingest", "warning:0") in kinds["data"] and ("profile", "basis") in kinds["data"]
    assert next(l for l in by["data"].lines if l.kind == "ingest").text.startswith(
        "the file is not valid UTF-8")
    # Not applicable, with the Router's reason, where the question would sit.
    assert ("not_applicable", "orientation") in kinds["data"]
    assert ("not_applicable", "survey") in kinds["whos_in"]
    # Defaults that change nothing here, with why.
    assert ("set_for_you", "clusters") in kinds["whos_in"]
    assert ("set_for_you", "read_from_data:unchanged") in kinds["data"]
    # What the engine filled in itself is recorded and shown.
    assert ("filled", "r2") in kinds["models"]
    # A stage not reached has nothing for the record yet.
    early = log_of(ProjectState(lens=["dietary"]), the_steps=steps_until("target"))
    out = sweep.for_the_record(early, steps_until("target"), [])
    assert [l for s in out.stages if s.key == "models" for l in s.lines] == []
