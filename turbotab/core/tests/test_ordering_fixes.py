"""P0.6 · the ordering fixes, engine side (crosswalk, "Where the engine order and the stage order
disagree"), and seam guard 6 (V2X_SEAMS, "A deferred option is a named value, refused").

The expectations come from the crosswalk's own fixes, not from the code under test:

* disagreement 10, the design slot: ``q:study-design`` in Your question, before the goal, with
  observational stated by default (``default:design_observational``, a Confirm);
* seam guard 6 and recommendation 6: crossover trials, repeated-measures trial models, complier and
  per-protocol effects and the direct effect are named values, each refused with a ``*_v2x`` code,
  its reason and an exit that keeps the work; a log that recorded one still loads;
* disagreement 9: the survey question is asked under every goal;
* disagreement 5: the draw and the validation scheme are two kinds, and a changed scheme
  (``set_validation``) never touches the draw's record;
* disagreement 20: "Decide now" answers a later question early only where its card is computed and
  the earlier answers it reads are in, and otherwise names what it waits for.
"""
from __future__ import annotations

import json
import typing
from datetime import datetime, timezone

import pytest

from turbotab.core import decisions as d
from turbotab.core import quest, seal
from turbotab.core.decisions import DecisionRecord, ProjectState, parse_decision
from turbotab.core.interview import QUESTION_KEYS, InterviewStep, route

T0 = datetime(2026, 10, 9, tzinfo=timezone.utc)
FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution")}
ROLES = {"pid": "identifier", "sugar": "exposure", "age": "covariate", "bmi": "covariate"}


def record(seq: int, decision: dict) -> DecisionRecord:
    return DecisionRecord(id=f"r{seq}", seq=seq, at=T0, decision=parse_decision(decision))


def refused(decision: dict, ctx: dict | None = None) -> d.Refusal:
    with pytest.raises(d.Refusal) as caught:
        d.validate(decision, ctx if ctx is not None else {})
    return caught.value


def inference(**extra) -> ProjectState:
    return ProjectState(lens=["clinical"], target="glucose", task="regression",
                        purpose="inference", roles=dict(ROLES), **extra)


# ── the design slot (disagreement 10) ────────────────────────────────────────


def test_the_design_is_a_question_in_your_question_just_before_the_goal():
    assert QUESTION_KEYS.index("design") == QUESTION_KEYS.index("purpose") - 1
    place = quest.QUESTIONS["design"]
    assert (place.stage, place.item) == ("question", "q:study-design")
    assert place.order < quest.QUESTIONS["purpose"].order


def test_unanswered_the_design_is_stated_observational_and_holds_nothing_back():
    state = ProjectState(lens=["clinical"], target="glucose", task="regression")
    steps = {s.key: s for s in route(state, FRESH)}
    assert steps["design"].status == "skipped"
    assert "observational" in steps["design"].reason
    assert steps["purpose"].status == "open"
    # In the quest log it is a default set for you, in Your question's Confirm sweep.
    log = quest.quest_log(state, [], route(state, FRESH), FRESH)
    line = next(l for s in log.stages for l in s.lines if l.key == "design")
    assert (line.label, line.status) == ("Confirm", "set_for_you")
    assert next(s.key for s in log.stages if line in s.lines) == "question"


def test_an_observational_answer_is_recorded_and_answers_the_question():
    assert d.validate({"kind": "set_design", "design": "observational"}, {}).design == \
        "observational"
    state = d.fold([record(1, {"kind": "set_design", "design": "observational"})])
    assert state.design == "observational"
    steps = {s.key: s for s in route(state, FRESH)}
    assert steps["design"].status == "answered"


@pytest.mark.parametrize("design,code", [("crossover", "crossover_v2x"),
                                         ("repeated_measures_trial",
                                          "repeated_measures_trial_v2x")])
def test_the_designs_deferred_to_v2x_are_named_values_refused_with_an_exit(design, code):
    error = refused({"kind": "set_design", "design": design})
    assert error.code == code
    assert error.message.startswith("Not available yet")
    assert "v2.x" in error.message
    assert error.exits[0]["decision"] == {"kind": "set_design", "design": "observational"}


@pytest.mark.parametrize("design", ["parallel_trial", "cluster_randomized_trial",
                                    "case_control", "matched_sets"])
def test_the_designed_experiments_wait_for_their_milestone_with_an_exit(design):
    error = refused({"kind": "set_design", "design": design})
    assert error.code == "not_available_yet"
    assert error.message.startswith("Not available yet")
    assert error.exits[0]["decision"] == {"kind": "set_design", "design": "observational"}


def test_the_design_registry_names_every_value_the_slot_accepts():
    from turbotab.core.designs import DESIGNS

    values = set(typing.get_args(d.DesignKey))
    assert values == set(DESIGNS)
    assert [k for k, v in DESIGNS.items() if v.available] == ["observational"]
    assert {k for k, v in DESIGNS.items() if v.code and v.code.endswith("_v2x")} == {
        "crossover", "repeated_measures_trial"}


# ── seam guard 6: the deferred effects ───────────────────────────────────────


@pytest.mark.parametrize("effect,code", [("direct", "direct_effect_v2x"),
                                         ("complier", "complier_effect_v2x"),
                                         ("per_protocol", "per_protocol_effect_v2x")])
def test_a_deferred_effect_is_a_named_value_refused_with_the_whole_effect_as_its_exit(effect, code):
    assert effect in typing.get_args(d.EffectKind)
    error = refused({"kind": "set_estimand", "exposure": "sugar", "effect": effect,
                     "measure": "mean_difference"}, {"state": inference()})
    assert error.code == code
    assert error.message.startswith("Not available yet")
    [whole] = [e["decision"] for e in error.exits if e["decision"]]
    assert whole == {"kind": "set_estimand", "exposure": "sugar", "family": False,
                     "effect": "total", "contrast": None, "measure": "mean_difference",
                     "multiplicity": None, "multiplicity_acknowledged": False}


def test_the_whole_effect_is_still_accepted():
    ok = d.validate({"kind": "set_estimand", "exposure": "sugar", "effect": "total",
                     "measure": "mean_difference"}, {"state": inference()})
    assert ok.effect == "total"


def test_a_log_that_recorded_a_direct_effect_or_a_deferred_design_still_loads():
    lines = [
        {"id": "a", "seq": 1, "at": "2026-10-01T00:00:00Z",
         "decision": {"kind": "set_target", "column": "glucose"}},
        {"id": "b", "seq": 2, "at": "2026-10-01T00:00:01Z",
         "decision": {"kind": "set_estimand", "exposure": "sugar", "effect": "direct",
                      "measure": "mean_difference"}},
        {"format": d.LOG_FORMAT, "id": "c", "seq": 3, "at": "2026-10-01T00:00:02Z",
         "decision": {"kind": "set_design", "design": "crossover"}},
    ]
    records = [d.read_record(json.loads(json.dumps(line))) for line in lines]
    state = d.fold(records)
    assert state.estimand.effect == "direct"
    assert state.design == "crossover"
    # The direct effect's own questions are kept for the mediation milestone (v2.x).
    from turbotab.core import estimand

    assert estimand.direct_questions is not None


def test_the_effect_card_offers_only_the_direct_part_as_not_available_yet():
    from turbotab.core.estimand import estimand_card

    card = estimand_card(inference(), "regression")
    effects = {e["effect"]: e for e in card["effects"]}
    assert effects["total"].get("available", True) is True
    assert effects["direct"]["available"] is False
    assert effects["direct"]["label"] == "Only the direct part"
    assert effects["direct"]["reason"].startswith("Not available yet")
    assert effects["direct"]["exit"]["effect"] == "total"


# ── the survey question under every goal (disagreement 9) ────────────────────


def test_the_survey_question_is_asked_under_prediction_when_a_weight_reads():
    from turbotab.core.survey import not_applicable_reason

    roles = {"SEQN": "identifier", "DR1TKCAL": "covariate", "WTDRD1": "design",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    state = ProjectState(lens=["dietary"], target="DR1TPROT", task="regression",
                         purpose="prediction", roles=roles)
    assert not_applicable_reason(state) is None
    steps = {s.key: s for s in route(state, FRESH)}
    assert steps["survey"].status != "not_applicable"
    # With no design column it does not apply, under either goal.
    plain = state.model_copy(update={"roles": {"SEQN": "identifier", "DR1TKCAL": "covariate"}})
    assert not_applicable_reason(plain) is not None


def test_under_prediction_the_survey_options_say_whose_performance_the_scores_estimate():
    from turbotab.core.survey import offered

    roles = {"SEQN": "identifier", "DR1TKCAL": "covariate", "WTDRD1": "design",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    state = ProjectState(lens=["dietary"], target="DR1TPROT", task="regression",
                         purpose="prediction", roles=roles)
    options = {o["key"]: o for o in offered(state)}
    assert "design-based cross-validation" in options["population:WTDRD1"]["consequence"]
    assert "these rows" in options["sample"]["consequence"]


# ── the draw and the validation scheme (disagreement 5) ──────────────────────


def split_log() -> list[DecisionRecord]:
    return [record(1, {"kind": "set_target", "column": "readmit_30d"}),
            record(2, {"kind": "set_purpose", "purpose": "prediction"}),
            record(3, {"kind": "set_split", "holdout": 0.2, "seed": 7, "folds": 5})]


def test_a_changed_scheme_keeps_the_draw_and_its_record():
    records = [*split_log(), record(4, {"kind": "set_validation", "validation": "repeated_kfold",
                                         "folds": 10, "repeats": 20})]
    state = d.fold(records)
    assert (state.split.holdout, state.split.seed) == (0.2, 7)
    assert (state.split.validation, state.split.folds, state.split.repeats) == (
        "repeated_kfold", 10, 20)
    # The draw's record is still the one that drew it: its time, and the seal's, do not move.
    assert seal.split_writer(records) == "r3"


def test_a_scheme_needs_a_draw_first():
    error = refused({"kind": "set_validation", "validation": "bootstrap"},
                    {"state": d.fold(split_log()[:2]), "records": lambda: split_log()[:2]})
    assert error.code == "no_draw_yet"


def test_a_scheme_names_its_cluster_only_for_internal_external_validation():
    with pytest.raises(Exception):
        parse_decision({"kind": "set_validation", "validation": "internal_external"})


def test_a_scheme_changed_after_the_opening_is_blocked_and_recorded():
    records = [*split_log(), record(4, {"kind": "select_models", "models": ["logistic"]}),
               record(5, {"kind": "open_seal", "family": "logistic",
                          "target": "readmit_30d"})]
    ctx = {"state": d.fold(records), "records": lambda: records}
    error = refused({"kind": "set_validation", "validation": "bootstrap"}, ctx)
    assert error.code == "scheme_after_opening"
    [keep] = [e["decision"] for e in error.exits if e["decision"]]
    assert keep["acknowledged"] is True
    assert d.validate(keep, ctx).acknowledged is True


def test_the_scheme_sits_in_models_and_the_draw_in_whos_in():
    assert quest.kind_place("set_validation").stage == "models"
    assert quest.kind_place("set_split").stage == "whos_in"


def test_the_lines_turbotab_records_sit_where_the_crosswalk_places_them():
    from pathlib import Path

    crosswalk = Path(__file__).resolve().parents[3] / "docs/turbotab-next/crosswalk/crosswalk.json"
    items = {i["id"]: i for i in json.loads(crosswalk.read_text("utf-8"))["items"]}
    for (key, goal), place in quest.GOAL_PLACES.items():
        item = items[place.item]
        assert (item["stage"], item["objective"]) == (place.stage, place.label), key
        assert goal in item["goals"], key
    scheme = items[quest.OTHER_KINDS["set_validation"].item]
    assert (scheme["stage"], scheme["objective"]) == ("models", "Confirm")
    stated = items["default:design_observational"]
    assert (stated["stage"], stated["objective"]) == ("question", "Confirm")


# ── what TurboTab records itself (disagreements 1 and 5) ─────────────────────


def test_a_completion_never_lands_over_the_persons_answer(tmp_path):
    log = d.DecisionLog(tmp_path / "decisions.jsonl")
    mine = log.append(d.SetRoles(roles={"sugar": "exposure"}))
    with pytest.raises(d.Refusal) as caught:
        log.append(d.SetRoles(roles={"sugar": "covariate"}), recorded_by="turbotab",
                   unless_set="roles")
    assert caught.value.code == "already_answered"
    assert [r.id for r in log.records()] == [mine.id]
    split = log.append(d.SetSplit(holdout=0.0), recorded_by="turbotab", unless_set="split")
    assert split.recorded_by == "turbotab"
    # A line says who recorded it only when it was TurboTab (the default is left unsaid).
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert '"recorded_by"' not in lines[0] and '"recorded_by":"turbotab"' in lines[1]
    assert d.DecisionLog(tmp_path / "decisions.jsonl").records()[1].recorded_by == "turbotab"


def test_the_roles_are_completed_only_once_every_reading_is_settled():
    from turbotab.core.readings import roles_completion

    artifact = {"columns": [
        {"column": "pid", "proposed": "identifier", "confidence": "high"},
        {"column": "sugar", "proposed": "exposure", "confidence": "medium"},
        {"column": "age", "proposed": "covariate", "confidence": "high"}]}
    state = ProjectState(target="glucose")
    assert roles_completion(state, artifact) is None  # `sugar` is read below high, unconfirmed
    confirmed = d.fold([record(1, {"kind": "set_target", "column": "glucose"}),
                        record(2, {"kind": "confirm_reading", "reading": "role",
                                   "column": "sugar", "value": "covariate"})])
    assert roles_completion(confirmed, artifact) == {"kind": "set_roles", "roles": {
        "pid": "identifier", "sugar": "covariate", "age": "covariate"}}
    # The person's own roles answer is never replaced.
    answered = confirmed.model_copy(update={"roles": {"sugar": "exposure"}})
    assert roles_completion(answered, artifact) is None


# ── "Decide now" (disagreement 20) ───────────────────────────────────────────


def steps_open_at(open_at: str) -> list[InterviewStep]:
    out, reached = [], False
    for key in QUESTION_KEYS:
        if key == open_at:
            reached = True
            out.append(InterviewStep(key=key, status="open"))
        elif reached:
            out.append(InterviewStep(key=key, status="waiting", waiting_on=[open_at]))
        else:
            out.append(InterviewStep(key=key, status="answered"))
    return out


def early_ctx(open_at: str, *, early: bool = True, stale: tuple[str, ...] = ()) -> dict:
    return {"interview": lambda: steps_open_at(open_at), "early": early,
            "stage": lambda name: {"status": "stale" if name in stale else "fresh"}}


def test_without_decide_now_a_later_question_is_refused_not_yet():
    from turbotab.core.sequence import _answers_in_order

    with pytest.raises(d.Refusal) as caught:
        _answers_in_order(parse_decision({"kind": "set_exclusions", "rules": []}),
                          early_ctx("survey", early=False))
    assert caught.value.code == "not_yet"


def test_decide_now_answers_early_a_question_whose_card_reads_only_answers_that_are_in():
    from turbotab.core.sequence import _answers_in_order

    # The eligibility rules' card reads the answers up to the grouping, not the survey's.
    _answers_in_order(parse_decision({"kind": "set_exclusions", "rules": []}),
                      early_ctx("survey"))


def test_decide_now_waits_for_an_earlier_answer_the_card_reads_and_names_it():
    from turbotab.core.sequence import _answers_in_order

    with pytest.raises(d.Refusal) as caught:
        _answers_in_order(parse_decision({"kind": "set_exclusions", "rules": []}),
                          early_ctx("clusters"))
    assert caught.value.code == "not_yet"
    assert caught.value.message.startswith("Waiting for")
    assert "group" in caught.value.message.lower()


def test_decide_now_waits_while_the_card_is_still_being_computed():
    from turbotab.core.sequence import _answers_in_order

    with pytest.raises(d.Refusal) as caught:
        _answers_in_order(parse_decision({"kind": "set_exclusions", "rules": []}),
                          early_ctx("survey", stale=("proposals",)))
    assert caught.value.code == "not_yet"
    assert caught.value.message.startswith("Waiting for")
