"""P0.4 · the stage registry (``turbotab/core/quest.py``): every engine object in one of the seven
stages, as ``crosswalk.json`` places it, and what each stage reports.

The expectations come from outside the module: the Router's ``QUESTION_KEYS``, the decision union,
the stage graph, the understanding catalogs and the crosswalk's own placements; the reopen
sentences from the package's own example ("Your change to Who's in reopened 2 questions in
Models").

Run as a script with ``--write-noticings`` to regenerate ``quest_noticings.json`` from the
crosswalk.
"""
from __future__ import annotations

import json
import sys
import typing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pytest

from turbotab.core import decisions, quest
from turbotab.core.decisions import (BatchSpec, DecisionRecord, ProjectState, SelectionSpec,
                                     parse_decision)
from turbotab.core.interview import QUESTION_KEYS, InterviewStep, route

REPO = Path(__file__).resolve().parents[3]
CROSSWALK = REPO / "docs" / "turbotab-next" / "crosswalk" / "crosswalk.json"
CATALOGS = REPO / "docs" / "turbotab-next" / "understanding" / "catalogs"
SEVEN = ("data", "question", "first_look", "whos_in", "models", "results", "writeup")
T0 = datetime(2026, 10, 8, tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def crosswalk() -> dict:
    return json.loads(CROSSWALK.read_text("utf-8"))


@lru_cache(maxsize=1)
def items() -> dict[str, dict]:
    return {i["id"]: i for i in crosswalk()["items"]}


def catalog_threads() -> set[str]:
    return {t["id"] for path in CATALOGS.glob("*.json")
            for t in json.loads(path.read_text("utf-8")).get("threads") or []}


def crosswalk_noticings() -> dict[str, dict[str, str]]:
    """Each catalog thread's stage and objective, as the crosswalk places it: an item of its own
    (``thread:<id>``), or carried by the item that already does its job (``threads``)."""
    known = catalog_threads()
    where: dict[str, tuple[str, str]] = {}
    for item in crosswalk()["items"]:
        carried = [*(item.get("threads") or []),
                   *([item["id"][len("thread:"):]] if item["id"].startswith("thread:") else [])]
        for thread in carried:
            if thread in known:  # one stage and objective per thread, however many items carry it
                place = (item["stage"], item["objective"])
                assert where.setdefault(thread, place) == place, thread
    return {stage: {t: o for t, (s, o) in sorted(where.items()) if s == stage} for stage in SEVEN}


def decision_kinds() -> set[str]:
    union = typing.get_args(typing.get_args(decisions.Decision)[0])
    return {model.model_fields["kind"].default for model in union}


def record(seq: int, decision: dict, minutes: float | None = None) -> DecisionRecord:
    at = T0 + timedelta(minutes=seq if minutes is None else minutes)
    return DecisionRecord(id=f"r{seq}", seq=seq, at=at, decision=parse_decision(decision))


def steps_until(open_at: str | None, *, stated: tuple[str, ...] = (),
                not_applicable: tuple[str, ...] = ()) -> list[InterviewStep]:
    """The Router's steps with every question before ``open_at`` answered (or stated), ``open_at``
    open and every later one waiting on it, as ``interview.route`` lays them out."""
    out = []
    reached = open_at is None
    for key in QUESTION_KEYS:
        if key in not_applicable:
            out.append(InterviewStep(key=key, status="not_applicable", reason="does not apply"))
        elif key == open_at:
            reached = True
            out.append(InterviewStep(key=key, status="open"))
        elif reached and open_at is not None:
            out.append(InterviewStep(key=key, status="waiting", waiting_on=[open_at]))
        elif key in stated:
            out.append(InterviewStep(key=key, status="skipped", reason="stated"))
        else:
            out.append(InterviewStep(key=key, status="answered", decision_id=f"answer-{key}"))
    return out


def stage(log: quest.QuestLog, key: str) -> quest.QuestStage:
    return next(s for s in log.stages if s.key == key)


def line(log: quest.QuestLog, key: str) -> quest.QuestLine:
    return next(l for s in log.stages for l in s.lines if l.key == key)


# ── coverage: one stage each ─────────────────────────────────────────────────


def test_every_router_question_sits_in_one_of_the_seven_stages():
    assert set(quest.QUESTIONS) == set(QUESTION_KEYS)
    assert {p.stage for p in quest.QUESTIONS.values()} <= set(SEVEN)
    assert [key for key, _name in quest.STAGES] == list(SEVEN)


def test_every_decision_kind_sits_in_one_stage_and_a_revert_where_its_record_does():
    kinds = decision_kinds()
    # "Confirm all" (P0.5) sits in the stage whose sweep it confirms, as a record of its own says.
    assert set(quest.kind_stages()) == kinds - {"revert", "confirm_sweep"}
    assert set(quest.kind_stages().values()) <= set(SEVEN)
    with pytest.raises(KeyError):
        quest.kind_place("revert")
    swept = record(1, {"kind": "confirm_sweep", "stage": "whos_in"})
    assert quest.record_stage(swept, [swept]) == "whos_in"
    assert set(quest.SWEEP_ITEMS) == set(SEVEN) - {"first_look"}
    for home, item in quest.SWEEP_ITEMS.items():
        assert (items()[item]["stage"], items()[item]["objective"]) == (home, "Confirm")
    # A kind that answers a Router question and is placed on its own card is placed in that
    # question's stage, so it has one stage either way.
    from turbotab.core.sequence import question_of

    for kind, place in quest.OTHER_KINDS.items():
        if question_of(kind) is not None:
            assert place.stage == quest.QUESTIONS[question_of(kind)].stage, kind
    log = [record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
           record(2, {"kind": "set_target", "column": "glucose"}),
           record(3, {"kind": "revert", "decision_id": "r2"}),
           record(4, {"kind": "revert", "decision_id": "r3"})]
    assert quest.record_stage(log[2], log) == "question"  # undoes the outcome
    assert quest.record_stage(log[3], log) == "question"  # undoes the undoing


def test_every_compute_stage_sits_in_one_stage():
    from turbotab.core.stages import build_graph

    names = [s.name for s in build_graph().stages()]
    assert len(names) == len(set(names)) == 30
    assert set(quest.COMPUTE) == set(names)
    assert {home for home, _item in quest.COMPUTE.values()} <= set(SEVEN)


def test_the_estimate_stages_are_the_ones_that_declare_they_serve_one():
    # V2X_SEAMS seam guard 7: declared on the stage, the gate's list derived from the declarations.
    from turbotab.core.estimand import ESTIMATE_STAGES
    from turbotab.core.plan_lock import ESTIMATE_STAGES as LOCKED
    from turbotab.core.stages import ESTIMATE, build_graph

    declared = [s.name for s in build_graph().stages() if s.serves == ESTIMATE]
    assert list(ESTIMATE_STAGES) == declared and LOCKED == ESTIMATE_STAGES
    # CROSSWALK "Engine stages and quest stages": the thirteen shown in Results, after Fit, with
    # Describe's usual-intake distribution ("Describe has a gate and a lock"; P0.8).
    assert set(declared) == {"fit", "substitution", "sensitivity", "calibration", "secondary",
                             "scales", "effects", "causal", "time_varying", "modification",
                             "explain", "evaluation", "usual_intake"}


def test_every_noticing_in_the_catalogs_sits_in_one_stage_with_an_objective():
    placed = quest.noticing_stages()
    assert set(placed) == catalog_threads()
    assert len(catalog_threads()) == 367  # CROSSWALK "At a glance": all 367 noticings placed
    assert set(placed.values()) <= set(SEVEN)
    # Gaps engine 1: "a stage and an objective". A thread with a crosswalk item of its own carries
    # that item's objective.
    objectives = {"Decide", "Confirm", "For the record", "Shown (not an objective)"}
    assert {o for _s, o in quest.noticing_places().values()} == objectives
    own = [i for i in crosswalk()["items"]
           if i["id"].startswith("thread:") and i["id"][len("thread:"):] in catalog_threads()]
    assert len(own) > 200
    for item in own:
        assert quest.noticing_place(item["id"][len("thread:"):]) == (item["stage"],
                                                                     item["objective"]), item["id"]


def test_every_finding_kind_explore_emits_is_placed():
    # ``ExploreFinding.kind`` is a free string: the kinds are the ones the stage writes.
    import ast

    from turbotab.core.stages import explore

    tree = ast.parse(Path(explore.__file__).read_text("utf-8"))
    emitted = {kw.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ExploreFinding"
               for kw in node.keywords
               if kw.arg == "kind" and isinstance(kw.value, ast.Constant)}
    assert len(emitted) >= 7
    assert emitted == set(quest.EXPLORE_FINDINGS)


# ── agreement with crosswalk.json ────────────────────────────────────────────


@pytest.mark.parametrize("key", QUESTION_KEYS)
def test_each_question_sits_where_the_crosswalk_decides_it(key):
    place = quest.QUESTIONS[key]
    item = items()[place.item]
    assert (place.stage, place.label) == (item["stage"], item["objective"])


@pytest.mark.parametrize("kind", sorted(quest.OTHER_KINDS))
def test_each_other_kind_sits_where_the_crosswalk_decides_it(kind):
    place = quest.OTHER_KINDS[kind]
    item = items()[place.item]
    assert (place.stage, place.label) == (item["stage"], item["objective"])


def test_each_result_shows_where_the_crosswalk_shows_it():
    from turbotab.core.estimand import ESTIMATE_STAGES

    for name, (home, item) in quest.COMPUTE.items():
        if item is not None:
            assert items()[item]["stage"] == home, name
    # CROSSWALK "Engine stages and quest stages": the estimate stages show in Results, after Fit.
    assert {quest.COMPUTE[name][0] for name in ESTIMATE_STAGES} == {"results"}


def test_findings_land_on_the_cards_the_crosswalk_decides_them_on():
    for route_to, item in quest.FINDING_ROUTES.items():
        home = "data" if route_to is None else quest.QUESTIONS[route_to].stage
        assert items()[item]["stage"] == home, route_to
    for kind, place in [*quest.EXPLORE_FINDINGS.items(), ("normalization", quest.NORMALIZATION)]:
        assert items()[place.item]["stage"] == place.stage, kind


def test_each_line_carries_the_label_its_contracts_tier_gives_it_or_a_named_ruling():
    # FOUNDATION §3: a question the engine asks is a Decide; a default it states is a Confirm, or
    # For the record. The two the crosswalk rules otherwise are named, with why.
    tiers = {d.kind: quest.contract_tier(d.kind) for d in quest.DECLARATIONS}
    assert {k for k, t in tiers.items() if t is not None} >= {
        "set_intended_use", "set_scales", "set_selection", "set_model_sequence",
        "set_measurement_error", "set_levers", "set_explain"}
    for kind, tier in tiers.items():
        if tier is None or kind in quest.TIER_RULINGS:
            continue
        label = quest.OTHER_KINDS[kind].label
        assert (label == "Decide") == (tier == "asked"), (kind, tier, label)
    for kind in quest.TIER_RULINGS:
        assert (quest.OTHER_KINDS[kind].label == "Decide") != (tiers[kind] == "asked"), kind


def test_the_noticings_file_is_the_crosswalks_placement():
    on_disk = json.loads(quest.NOTICINGS_FILE.read_text("utf-8"))
    assert on_disk == crosswalk_noticings(), (
        "quest_noticings.json drifted from crosswalk.json: run "
        "venv/bin/python -m turbotab.core.tests.test_stage_registry --write-noticings")


def test_each_stage_keeps_the_routers_order_and_the_ruled_moves_ahead_of_the_families():
    # Disagreement 18: the quest log adopts the Router's order for the questions it asks.
    for home in SEVEN:
        asked = [k for k in QUESTION_KEYS
                 if quest.QUESTIONS[k].stage == home and quest.QUESTIONS[k].label == "Decide"]
        orders = [quest.QUESTIONS[k].order for k in asked]
        assert orders == sorted(orders), (home, asked)
    # Disagreement 21 (and the Predict order): scales, batch, the normalization and selection
    # come before the families question.
    families = quest.QUESTIONS["models"].order
    for place in (quest.OTHER_KINDS["set_scales"], quest.OTHER_KINDS["set_batch"],
                  quest.NORMALIZATION, quest.OTHER_KINDS["set_selection"]):
        assert place.order < families


# ── what each stage reports ──────────────────────────────────────────────────


def test_a_stage_not_reached_reports_no_progress_and_waits_for_the_question_ahead():
    records = [record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
               record(2, {"kind": "set_target", "column": "glucose"})]
    state = decisions.fold(records)
    steps = route(state, {}, {}, records)
    log = quest.quest_log(state, records, steps)
    first = next(s for s in steps if s.status in ("open", "waiting"))
    assert stage(log, "data").reached and stage(log, "question").reached
    models = stage(log, "models")
    assert not models.reached and models.progress is None  # empty, never "0 of N"
    estimand = line(log, "estimand")
    assert estimand.status == "waiting"
    assert [(w.key, w.stage) for w in estimand.waiting_for] == [
        (first.key, quest.QUESTIONS[first.key].stage)]
    # Your data: the lens answered; the roles, asked after Who's in's structure questions today,
    # wait; the offers (a join, a codebook) are listed and not counted.
    data = stage(log, "data")
    assert data.progress == quest.Progress(answered=1, required=2, complete=False)
    assert {l.key: l.counted for l in data.lines if l.source == "declaration"} == {
        "join_files": False, "import_codebook": False}


def test_progress_counts_each_decide_once_and_the_confirm_sweep_once():
    state = ProjectState(lens=["dietary"], target="glucose", purpose="inference")
    steps = steps_until("estimand", stated=("grain", "repeat_kind", "clusters"),
                        not_applicable=("orientation", "event", "unit", "aggregation", "temporal",
                                        "survey"))
    log = quest.quest_log(state, [], steps)
    whos_in = stage(log, "whos_in")
    # Decide: exclusions, missing, split (answered); Confirm: the grain and the repeat kind as
    # stated (one sweep, open); For the record: no grouping to offer.
    assert (whos_in.sweep.lines, whos_in.sweep.answered) == (2, False)
    assert whos_in.progress == quest.Progress(answered=3, required=4, complete=False)
    assert line(log, "clusters").label == "For the record"
    assert stage(log, "models").progress.answered == 0


def test_the_families_wait_for_batch_and_selection_under_predict():
    state = ProjectState(lens=["metabolomics"], target="outcome", purpose="prediction")
    steps = steps_until("models", stated=("modification", "causal"))
    columns = ["sample_id", "batch", "m_001", "m_002", "outcome"]
    log = quest.quest_log(state, [], steps, columns=columns)
    families = line(log, "models")
    assert families.status == "waiting"
    assert [w.key for w in families.waiting_for] == ["set_batch", "set_selection"]
    answered = state.model_copy(update={"batch": BatchSpec(column="batch", method="covariate"),
                                        "selection": SelectionSpec()})
    assert line(quest.quest_log(answered, [], steps, columns=columns), "models").status == "open"
    # Without a column named as a batch, the batch question is not in the stage at all.
    log = quest.quest_log(state, [], steps, columns=["sample_id", "m_001", "outcome"])
    assert "set_batch" not in {l.key for l in stage(log, "models").lines}


def _plan(change: dict) -> tuple[ProjectState, list[DecisionRecord]]:
    records = [
        record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
        record(2, {"kind": "set_target", "column": "glucose"}),
        record(3, {"kind": "set_purpose", "purpose": "inference"}),
        record(4, {"kind": "set_roles", "roles": {"sugar": "exposure", "age": "covariate",
                                                 "tg": "excluded"}}),
        record(5, {"kind": "set_estimand", "exposure": "sugar", "effect": "total",
                   "measure": "mean_difference"}),
        record(6, {"kind": "set_adjustment", "exposure": "sugar", "answers": {"age": {
            "causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}}}),
        record(7, change),
    ]
    return decisions.fold(records), records


def test_a_question_asked_again_by_a_change_in_another_stage_says_why():
    state, records = _plan({"kind": "confirm_role", "column": "tg", "role": "covariate"})
    steps = steps_until("adjustment")
    log = quest.quest_log(state, records, steps)
    adjustment = line(log, "adjustment")
    assert adjustment.reopened_by == quest.ReopenedBy(decision_id="r7", kind="confirm_role",
                                                      stage="data")
    assert stage(log, "models").reopened == [quest.Reopened(
        changed_in="data", decision_id="r7", kind="confirm_role", questions=["q:adjustment"],
        sentence="Your change to Your data reopened 1 question in Models.")]
    # The same question asked again by another answer in Models itself (FOUNDATION §3: "Your data
    # reopened: the join added 12 columns, so what each new column is gets read again"): the stage
    # says so too, marked as within it.
    state, records = _plan({"kind": "set_estimand", "exposure": "age", "effect": "total",
                            "measure": "mean_difference"})
    log = quest.quest_log(state, records, steps)
    assert line(log, "adjustment").reopened_by.stage == "models"
    assert stage(log, "models").reopened == [quest.Reopened(
        changed_in="models", decision_id="r7", kind="set_estimand", questions=["q:adjustment"],
        within=True, sentence="Your change to another answer in Models reopened 1 question.")]


EXCLUSION = {"kind": "set_exclusions", "rules": [
    {"column": "sugar", "low": 0, "high": 500, "reason": "implausible intake"}]}
TG_COVARIATE = {"kind": "confirm_role", "column": "tg", "role": "covariate"}


def test_a_reopen_names_the_change_that_reopened_it_whatever_came_after():
    # Triglycerides confirmed as a covariate (Your data) leave the adjustment set without their
    # answers; an exclusion rule recorded next (Who's in) changes nothing the set rests on.
    state, records = _plan(TG_COVARIATE)
    records = [*records, record(8, EXCLUSION)]
    log = quest.quest_log(decisions.fold(records), records, steps_until("adjustment"))
    assert line(log, "adjustment").reopened_by == quest.ReopenedBy(
        decision_id="r7", kind="confirm_role", stage="data")
    assert [r.sentence for r in stage(log, "models").reopened] == [
        "Your change to Your data reopened 1 question in Models."]
    # The other way round, the rule first: the confirmation is still the one named.
    records = [*_plan(EXCLUSION)[1], record(8, TG_COVARIATE)]
    log = quest.quest_log(decisions.fold(records), records, steps_until("adjustment"))
    assert line(log, "adjustment").reopened_by.decision_id == "r8"
    # Undone and then broken again by another answer: the later one is the cause.
    records = [*_plan(TG_COVARIATE)[1], record(8, {"kind": "revert", "decision_id": "r7"}),
               record(9, {"kind": "set_estimand", "exposure": "age", "effect": "total",
                          "measure": "mean_difference"}),
               record(10, EXCLUSION)]
    log = quest.quest_log(decisions.fold(records), records, steps_until("adjustment"))
    assert line(log, "adjustment").reopened_by == quest.ReopenedBy(
        decision_id="r9", kind="set_estimand", stage="models")


def test_model_one_is_asked_again_as_the_engine_asks_it_again():
    # MODELING_SEQUENCE §2: a Model 1 column that leaves the primary adjustment set re-asks it
    # (``estimand.current_model_sequence``), whichever stage the change is made in.
    model_one = {"kind": "set_model_sequence", "exposure": "sugar", "model_1": ["age"]}
    steps = steps_until(None, not_applicable=("open_seal",))
    state, records = _plan(model_one)
    assert line(quest.quest_log(state, records, steps), "set_model_sequence").status == "answered"
    # Age re-roled as left out, in Your data.
    records_1 = [*records, record(8, {"kind": "confirm_role", "column": "age", "role": "excluded"})]
    log = quest.quest_log(decisions.fold(records_1), records_1, steps)
    model_1 = line(log, "set_model_sequence")
    assert model_1.status == "open"
    assert model_1.reopened_by == quest.ReopenedBy(decision_id="r8", kind="confirm_role",
                                                   stage="data")
    assert [r.sentence for r in stage(log, "models").reopened] == [
        "Your change to Your data reopened 1 question in Models."]
    # Age answered as measured after the exposure (a possible mediator), in Models.
    records_2 = [*records, record(8, {"kind": "set_adjustment", "exposure": "sugar", "answers": {
        "age": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "yes"}}})]
    log = quest.quest_log(decisions.fold(records_2), records_2, steps)
    assert line(log, "set_model_sequence").status == "open"
    assert [(r.within, r.kind) for r in stage(log, "models").reopened] == [
        (True, "set_adjustment")]
    # No exposure in the model any longer: nothing to hold Model 1 against.
    records_3 = [*records, record(8, {"kind": "confirm_role", "column": "sugar",
                                      "role": "excluded"})]
    assert line(quest.quest_log(decisions.fold(records_3), records_3, steps),
                "set_model_sequence").status != "answered"


def test_a_result_computed_before_a_change_in_another_stage_is_out_of_date_and_says_why():
    # Disagreement 7: a Models answer changes who is in, so Who's in's participant flow drops back.
    state, records = _plan({"kind": "set_estimand", "exposure": "age", "effect": "total",
                            "measure": "mean_difference"})
    steps = steps_until("adjustment")
    stages = {"cohort": {"status": "stale"}}
    before = T0 + timedelta(minutes=6.5)
    log = quest.quest_log(state, records, steps, stages, shown_at={"cohort": before})
    assert stage(log, "whos_in").reopened == [quest.Reopened(
        changed_in="models", decision_id="r7", kind="set_estimand", results=["cohort"],
        sentence="Your change to Models made 1 result in Who's in out of date.")]
    # Computed after the change, it is out of date for some other reason: none is given.
    after = T0 + timedelta(minutes=7.5)
    log = quest.quest_log(state, records, steps, stages, shown_at={"cohort": after})
    assert stage(log, "whos_in").reopened == []


def _after_the_fit(change: dict) -> tuple[ProjectState, list[DecisionRecord], list[InterviewStep]]:
    """The plan answered and fitted at minute 7.5 under Estimate (nothing held out), then
    ``change`` recorded at minute 8."""
    state, records = _plan({"kind": "select_models", "models": ["linear"]})
    records = [*records, record(8, change)]
    return (decisions.fold(records), records,
            steps_until(None, not_applicable=("substitution", "open_seal")))


def test_results_stays_reached_after_the_fit_and_says_why_it_dropped_back():
    state, records, steps = _after_the_fit(EXCLUSION)
    fitted = T0 + timedelta(minutes=7.5)
    # Before any estimate is computed Results is not reached: empty, with no reason.
    log = quest.quest_log(state, records, steps, {"fit": {"status": "idle"}})
    assert [stage(log, k).reached for k in SEVEN] == [True] * 5 + [False] * 2
    assert stage(log, "results").progress is None
    # Fresh for the answers now: reached, and asking nothing under Estimate, complete at 0 of 0.
    log = quest.quest_log(state, records, steps, {"fit": {"status": "fresh"}})
    assert stage(log, "results").reached and stage(log, "writeup").reached
    assert stage(log, "results").progress == quest.Progress(answered=0, required=0, complete=True)
    # The rule recorded in Who's in after the fit: the fit is out of date, and Results stays
    # reached and says why, until it is computed again (or, with Fit held, until Fit is pressed).
    for status in ("stale", "queued", "running"):
        log = quest.quest_log(state, records, steps, {"fit": {"status": status}},
                              shown_at={"fit": fitted})
        results = stage(log, "results")
        assert results.reached and stage(log, "writeup").reached, status
        assert results.progress == quest.Progress(answered=0, required=0, complete=True)
        assert results.reopened == [quest.Reopened(
            changed_in="whos_in", decision_id="r8", kind="set_exclusions", results=["fit"],
            sentence="Your change to Who's in made 1 result in Results out of date.")]


def test_results_opens_when_fit_is_pressed_not_when_an_estimate_is_computed():
    # P0.8 (FOUNDATION §7): under Estimate, Results opens with the lock Fit records; a fit computed
    # live before the press opens nothing. The log carries the lock as Fit reports it.
    from turbotab.core.fit_press import fit_lock

    state, records, steps = _after_the_fit(EXCLUSION)
    fresh = {"fit": {"status": "fresh"}, "usual_intake": {"status": "fresh"}}
    unlocked = fit_lock(state, records, pressed=False, held=False, estimate=None)
    log = quest.quest_log(state, records, steps, fresh, fit=unlocked)
    assert [stage(log, k).reached for k in SEVEN] == [True] * 5 + [False] * 2
    assert log.fit == unlocked and not log.fit.locked
    lock = record(9, {"kind": "lock_plan", "plan": {"purpose": "inference"}, "digest": "a" * 64})
    locked_records = [*records, lock]
    locked = decisions.fold(locked_records)
    report = fit_lock(locked, locked_records, pressed=True, held=False, estimate=None)
    log = quest.quest_log(locked, locked_records, steps, {"fit": {"status": "running"}},
                          fit=report)
    assert stage(log, "results").reached and stage(log, "writeup").reached
    assert log.fit.locked and log.fit.sha256 == "a" * 64 and log.fit.at == lock.at
    # Under Predict, the press for this outcome opens it; with no purpose nothing does.
    predict = state.model_copy(update={"purpose": "prediction"})
    for pressed, reached in ((False, False), (True, True)):
        log = quest.quest_log(predict, records, steps, fresh,
                              fit=fit_lock(predict, records, pressed=pressed, held=False,
                                           estimate=None))
        assert stage(log, "results").reached is reached, pressed
    none = state.model_copy(update={"purpose": None})
    log = quest.quest_log(none, records, steps, fresh,
                          fit=fit_lock(none, records, pressed=True, held=False, estimate=None))
    assert not stage(log, "results").reached
    # Without what Fit says, the usual-intake offer (computed on the lens and goal alone) opens
    # nothing, though it is declared an estimate stage.
    log = quest.quest_log(state, records, steps, {"usual_intake": {"status": "fresh"}})
    assert not stage(log, "results").reached


def test_a_withdrawn_analysis_is_not_out_of_date():
    # The causal lane's estimate, computed at minute 7.5; its answer is then withdrawn, so its
    # stage is blocked and is never computed again: no reason, but Results was reached.
    state, records, steps = _after_the_fit({"kind": "set_lens", "lenses": ["dietary", "clinical"]})
    fitted = T0 + timedelta(minutes=7.5)
    log = quest.quest_log(state, records, steps, {"causal": {"status": "blocked"}},
                          shown_at={"causal": fitted})
    assert stage(log, "results").reached and stage(log, "results").reopened == []
    # Not blocked, the same artifact is out of date for the change in Your data.
    log = quest.quest_log(state, records, steps, {"causal": {"status": "stale"}},
                          shown_at={"causal": fitted})
    assert [(r.changed_in, r.results) for r in stage(log, "results").reopened] == [
        ("data", ["causal"])]


def test_a_reached_stage_that_asks_nothing_is_complete_and_one_not_reached_is_empty():
    state = ProjectState(lens=["dietary"], target="glucose", purpose="inference")
    log = quest.quest_log(state, [], steps_until("estimand"))
    first_look = stage(log, "first_look")
    assert first_look.reached and first_look.lines == []
    assert first_look.progress == quest.Progress(answered=0, required=0, complete=True)
    assert stage(log, "results").progress is None and stage(log, "writeup").progress is None
    log = quest.quest_log(state, [], steps_until(None, stated=("modification", "causal"),
                                                 not_applicable=("open_seal",)))
    assert stage(log, "whos_in").progress.complete


def test_progress_holds_while_the_outcome_is_read_again():
    # Your question settled: the outcome and the goal answered, the event not asked, the task read
    # at high confidence (stated), the follow-up not asked.
    state = ProjectState(lens=["dietary"], target="glucose", purpose="inference")
    settled = steps_until(None, stated=("task",), not_applicable=("event", "follow_up"))
    before = stage(quest.quest_log(state, [], settled), "question").progress
    assert (before.answered, before.required) == (2, 2)
    # The outcome's reading recomputes (``interview.route``): the event, the task and the
    # follow-up wait on it, since whether each is asked at all is that reading's.
    reading = [InterviewStep(key=s.key, status="waiting",
                             waiting_on=["target_info"] if s.key == "event"
                             else ["event", "target_info"])
               if s.key in ("event", "task", "follow_up") else s for s in settled]
    log = quest.quest_log(state, [], reading)
    assert stage(log, "question").progress == before
    assert not line(log, "event").counted and line(log, "event").status == "waiting"
    # A recorded answer that still holds stands meanwhile, and counts: the task, set for the
    # outcome now; one recorded for another outcome does not hold, so it waits uncounted.
    records = [record(1, {"kind": "set_target", "column": "glucose"}),
               record(2, {"kind": "set_task", "column": "glucose", "task": "regression"})]
    state = decisions.fold(records)
    task = line(quest.quest_log(state, records, reading), "task")
    assert (task.status, task.counted, task.decision_id, task.computing) == (
        "answered", True, "r2", ["target_info"])
    records = [*records, record(3, {"kind": "set_target", "column": "hba1c"})]
    state = decisions.fold(records)
    task = line(quest.quest_log(state, records, reading), "task")
    assert (task.status, task.counted) == ("waiting", False)


def test_the_declarations_apply_where_the_engine_asks_them():
    columns = ["SEQN", "sugar", "glucose"]

    def keys(state: ProjectState, artifacts: dict | None = None) -> set[str]:
        log = quest.quest_log(state, [], steps_until("estimand"), columns=columns,
                              artifacts=artifacts)
        return {l.key for s in log.stages for l in s.lines if l.source == "declaration"}

    diet = ProjectState(lens=["dietary"], target="glucose", purpose="inference")
    # Regression calibration: rows that are each person's mean of their recalls, with the linear
    # family (or the families not chosen yet: the line waits for them).
    mean = diet.model_copy(update={"aggregation": decisions.AggregationSpec(method="mean")})
    assert "set_measurement_error" in keys(mean)
    assert "set_measurement_error" not in keys(diet.model_copy(update={
        "aggregation": decisions.AggregationSpec(method="first")}))
    assert "set_measurement_error" not in keys(mean.model_copy(update={
        "models": ["elastic_net"]}))
    assert "set_measurement_error" in keys(mean.model_copy(update={"models": ["linear"]}))
    # Usual intake: where the stage offers it, under any goal but prediction.
    offered = {"usual_intake": {"offer": {"offered": True}}}
    assert "set_usual_intake" not in keys(diet)
    assert "set_usual_intake" not in keys(diet, {"usual_intake": {"offer": {"offered": False}}})
    assert "set_usual_intake" in keys(diet, offered)
    assert "set_usual_intake" not in keys(diet.model_copy(update={"purpose": "prediction"}),
                                          offered)
    # Shrinkage: an unpenalized regression family under prediction.
    predict = ProjectState(lens=["metabolomics"], target="outcome", purpose="prediction")
    assert "set_updating" in keys(predict)
    assert "set_updating" in keys(predict.model_copy(update={"models": ["linear", "ridge"]}))
    assert "set_updating" not in keys(predict.model_copy(update={"models": ["boosted_trees"]}))
    # A penalized regression family declares no shrinkage updating: the slope is its own penalty.
    assert "set_updating" not in keys(predict.model_copy(update={"models": ["elastic_net"]}))


def test_a_finding_is_decided_at_its_question_or_where_it_was_held():
    state = ProjectState(lens=["dietary"], target="glucose", purpose="inference",
                         findings={"binary_text__smoker": {"action": "deferred", "to": "missing"}})
    findings = {"findings": [
        {"id": "pack::dietary::implausible_intake", "routes_to": "exclusions", "repairs": [],
         "answered_by": None},
        {"id": "sas_zeros", "routes_to": None, "repairs": [{"key": "zero"}], "answered_by": None},
        {"id": "binary_text__smoker", "routes_to": None, "repairs": [{"key": "map"}],
         "answered_by": "r9"},
        {"id": "unnamed_columns", "routes_to": None, "repairs": [], "answered_by": None},
    ]}
    steps = steps_until("missing")
    log = quest.quest_log(state, [], steps, findings=findings)
    where = {l.key: s.key for s in log.stages for l in s.lines if l.source == "finding"}
    assert where == {"pack::dietary::implausible_intake": "whos_in", "sas_zeros": "data",
                     "binary_text__smoker": "whos_in", "unnamed_columns": "data"}
    # Held for the missing-values question, it is decided when that question is answered.
    assert line(log, "binary_text__smoker").status == "open"
    assert line(log, "unnamed_columns").label == "For the record"
    assert not line(log, "unnamed_columns").counted


def _write_noticings() -> None:
    text = json.dumps(crosswalk_noticings(), indent=1) + "\n"
    quest.NOTICINGS_FILE.write_text(text, encoding="utf-8")
    print(f"wrote {quest.NOTICINGS_FILE}", file=sys.stderr)


if __name__ == "__main__":
    if "--write-noticings" in sys.argv[1:]:
        _write_noticings()
