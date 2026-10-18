"""The pilot's mechanical checks (DRIVER_PLAN D5; SWARM_PLAN §7), each on constructed records both
ways: silent on the record built to keep it, firing (or saying what it could not check) on one
changed to break it. Every truth is the construction's: ``constructed.py`` builds the records, and
the known-answer tests declare their own truths beside ``truths.FIXTURE_TRUTHS``' own."""
from __future__ import annotations

import copy
import json

import pytest

from research.swarm import checks as C
from research.swarm.assemble import write
from research.swarm.record import validate
from research.swarm.tests.constructed import (
    decision,
    estimate_record,
    movement,
    predict_record,
)


def outcome(check, data, **kw):
    return check(validate(data), **kw)


# ── 1. seal integrity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("make", [predict_record, estimate_record])
def test_seal_integrity_is_silent_on_a_result_no_later_record_reached(make):
    got = outcome(C.seal_integrity, make())
    assert (got.outcome, got.problems) == ("pass", [])


def test_seal_integrity_fires_on_a_reseal():
    data = predict_record()
    data["decisions"].append(decision(6, "reseal", {"reason": "again", "target": "y"},
                                      [("reseal", {})], post_seal=True, minute=20))
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail" and got.problems == ["Record #6 re-seals the held-out rows."]


def test_seal_integrity_fires_on_a_second_opening_of_the_same_outcome():
    data = predict_record()
    data["decisions"].append(decision(6, "open_seal", {"family": "linear", "target": "y"},
                                      [("open_seal", {})], post_seal=True, minute=20))
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail"
    assert got.problems == ["The held-out rows of `y` were opened 2 times (records #5, #6)."]


def test_seal_integrity_fires_when_a_record_after_the_opening_reaches_the_result():
    data = predict_record()
    data["decisions"].append(decision(6, "select_models", {"models": ["linear"]},
                                      [("select_models", {"models": ["linear"]})],
                                      post_seal=True, minute=20))
    data["result"]["decisions"] = [1, 3, 6]
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail" and "#6" in got.problems[0]


def test_seal_integrity_fires_when_the_result_is_not_the_openings_scores():
    data = predict_record()
    data["result"]["scores"]["linear"]["auc"] = 0.75  # a later fit's score, reported instead
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result's scores are not those the opening at record #5 "
                            "recorded."]


def test_seal_integrity_does_not_pass_a_result_whose_fit_is_unknown():
    data = predict_record()
    data["result"].update(attempt=None, decisions=None)
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "not_checked" and got.unchecked


def test_seal_integrity_fires_when_a_change_made_with_the_estimates_in_view_reaches_the_result():
    data = estimate_record()
    data["result"].update(attempt="fit2", decisions=[1, 2, 3, 4, 5])
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail" and "#5" in got.problems[0]


def test_seal_integrity_fires_when_the_result_is_not_the_locked_plans():
    data = estimate_record()
    data["result"]["plan_digest"] = "another plan"
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result's plan is not the one the lock at record #4 recorded."]


# ── 2. logging ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("make", [predict_record, estimate_record])
def test_logging_is_silent_when_every_attempt_and_record_has_its_reason(make):
    got = outcome(C.logging_complete, make())
    assert (got.outcome, got.problems) == ("pass", [])


def test_logging_fires_on_an_attempt_with_no_reason():
    data = predict_record()
    data["attempts"][1]["reasons"] = []
    got = outcome(C.logging_complete, data)
    assert got.problems == ["Fit #2 has no reason: no record that changed before it (records #4) "
                            "has a note."]


def test_a_fit_after_turbotabs_own_records_only_is_not_the_analysts_attempt():
    data = estimate_record()
    data["attempts"][1].update(changed=[4], reasons=[])
    got = outcome(C.logging_complete, data)
    assert got.outcome == "pass" and "TurboTab's own records" in got.notes[0]


def test_logging_fires_on_a_family_with_no_movement_and_no_why():
    data = predict_record()
    data["attempts"][1]["movement"] = [movement("linear", "fit1", 0.4)]  # elastic_net's dropped
    got = outcome(C.logging_complete, data)
    assert got.problems == ["Fit #2 has no measured movement for `elastic_net`, and no why."]


def test_logging_fires_on_an_analysts_record_with_no_reason_and_not_on_turbotabs():
    data = predict_record()
    data["decisions"][2]["reason"] = "  "
    got = outcome(C.logging_complete, data)
    assert got.problems == ["Record #3 (confirm_reading) has no reason."]
    assert outcome(C.logging_complete, estimate_record()).outcome == "pass"  # #4 is TurboTab's


def test_a_failed_run_is_a_logging_failure(tmp_path):
    raw = predict_record()
    raw["decisions"][0]["materiality"] = {"source": None}
    path, ok = write(raw, "run-1", tmp_path)
    got = C.failed_run(path)
    assert not ok and got.outcome == "fail" and "schema" in got.problems[0]


# ── 3. single pass ───────────────────────────────────────────────────────────


def _single(data):
    for d in data["decisions"]:
        d["examined"] = [{**e, "alternative": False} for e in d["examined"]]
    data["attempts"] = data["attempts"][:1]
    data["result"].update(attempt="fit1", decisions=[1, 2, 3])
    data["decisions"][3]["materiality"] = {"source": None, "why": "no second fit"}
    return data


def test_single_pass_flags_a_run_that_examined_nothing_and_fit_once():
    got = outcome(C.single_pass, _single(predict_record()))
    assert got.outcome == "fail"
    assert got.problems == ["Single pass: before the seal's opening or the plan's lock, no "
                            "decision had an alternative examined, and the run made 1 fit."]


def test_single_pass_counts_no_option_tried_or_fitted_after_the_seal_was_opened():
    data = _single(predict_record())
    data["decisions"].append(decision(
        6, "select_models", {"models": ["elastic_net"]},
        [("select_models", {"models": ["elastic_net"]})], post_seal=True, minute=20,
        examined=[{"question": "select_models", "value": {"models": ["linear"]}, "how": "fit",
                   "at": data["attempts"][0]["at"], "seq": 2, "attempt": "fit1",
                   "alternative": True}]))
    second = copy.deepcopy(data["attempts"][0])
    second.update(id="fit2", seq=2, decisions=[1, 2, 3, 6], changed=[6],
                  at="2026-10-18T10:21:00+00:00")
    data["attempts"].append(second)
    got = outcome(C.single_pass, data)
    assert got.outcome == "fail" and "the run made 1 fit." in got.problems[0]


def test_single_pass_is_silent_on_one_alternative_examined_or_a_second_fit():
    assert outcome(C.single_pass, predict_record()).outcome == "pass"
    one_fit = predict_record()
    one_fit["attempts"] = one_fit["attempts"][:1]
    one_fit["result"].update(attempt="fit1", decisions=[1, 2, 3])
    one_fit["decisions"][3]["materiality"] = {"source": None, "why": "no second fit"}
    assert outcome(C.single_pass, one_fit).outcome == "pass"  # #4 examined #2's option
    two_fits = predict_record()
    for d in two_fits["decisions"]:
        d["examined"] = []
    assert outcome(C.single_pass, two_fits).outcome == "pass"


@pytest.mark.parametrize("minute,status", [(7, "pass"), (9, "fail")])
def test_single_pass_counts_an_alternative_previewed_before_the_opening_only(minute, status):
    """The opening is at 10:08: a preview of another option while record #2 stood counts before
    it, and not after it."""
    data = _single(predict_record())
    data["decisions"][1]["examined"] = [{"question": "select_models",
                                         "value": {"models": ["boosted_trees"]},
                                         "how": "preview", "at": f"2026-10-18T10:{minute:02d}:00Z",
                                         "refused": None, "alternative": True}]
    assert outcome(C.single_pass, data).outcome == status


def test_an_option_examined_that_is_the_one_chosen_is_not_exploring():
    data = _single(predict_record())
    data["decisions"][1]["examined"] = [{"question": "select_models",
                                         "value": {"models": ["linear"]}, "how": "preview",
                                         "at": data["decisions"][0]["at"], "refused": None,
                                         "alternative": False}]
    assert outcome(C.single_pass, data).outcome == "fail"


# ── 4. known-answer controls ─────────────────────────────────────────────────

TRUTH = {"table.csv": {"code_or_count:age": "amount", "unit:kcal": "kcal",
                       "adjust:age": "yes,yes,no"}}


def test_known_answer_is_not_judged_on_a_run_that_is_not_a_control():
    got = outcome(C.known_answer, predict_record(), truths=TRUTH)
    assert got.outcome == "not_checked" and got.unchecked == ["Not a known-answer control."]


def test_known_answer_fires_on_a_fact_reading_answered_otherwise_than_declared():
    data = predict_record("known_answer")
    data["decisions"][2]["answers"] = [{"question": "code_or_count:age", "value": "code"}]
    data["result"]["decisions"] = [1, 3, 4]
    got = outcome(C.known_answer, data, truths=TRUTH)
    assert got.outcome == "fail"
    assert got.problems == ["Record #3 answers the code_or_count:age reading as `code`; the "
                            "fixture declares `amount`."]


def test_known_answer_leaves_the_research_questions_assumptions_to_the_analyst():
    """``adjust:`` is the fixture author's causal reading, not a fact of the data."""
    data = predict_record("known_answer")
    data["decisions"][0]["answers"].append({"question": "adjust:age", "value": "no,no,no"})
    got = outcome(C.known_answer, data, truths=TRUTH)
    assert got.problems == []


def test_known_answer_says_it_cannot_judge_a_result_with_no_declared_answer():
    got = outcome(C.known_answer, predict_record("known_answer"), truths=TRUTH)
    assert got.outcome == "not_checked" and got.problems == []
    assert "no known answer is declared" in got.unchecked[0]


def test_known_answer_reads_the_repos_fixture_truths_by_default():
    data = predict_record("known_answer")
    data["dataset"]["fixture"] = "clinical_risk.csv"  # declares code_or_count:age = amount
    data["decisions"][2]["answers"] = [{"question": "code_or_count:age", "value": "code"}]
    got = outcome(C.known_answer, data)
    assert got.outcome == "fail" and "declares `amount`" in got.problems[0]


@pytest.mark.parametrize("value,status", [(0.30, "pass"), (0.10, "pass"), (0.55, "fail"),
                                          (-0.01, "fail")])
def test_known_answer_holds_an_estimate_whose_interval_covers_the_declared_value(value, status):
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": value}
    got = outcome(C.known_answer, data, truths={"table.csv": {}})
    assert got.outcome == status


def test_known_answer_holds_a_refused_result():
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": 9.0}
    data["result"].update(effects=None, refused="The plan was never locked.")
    assert outcome(C.known_answer, data, truths={"table.csv": {}}).outcome == "pass"


@pytest.mark.parametrize("low,high,status", [(0.65, 0.75, "pass"), (0.72, 0.80, "fail")])
def test_known_answer_holds_a_held_out_score_within_the_declared_range(low, high, status):
    data = predict_record("known_answer")
    data["dataset"]["known"] = {"metric": "auc", "low": low, "high": high}
    assert outcome(C.known_answer, data, truths=TRUTH).outcome == status


# ── 5. null decoys ───────────────────────────────────────────────────────────


def test_null_decoy_is_not_judged_on_a_run_that_is_not_a_decoy():
    assert outcome(C.null_decoy, estimate_record()).outcome == "not_checked"


@pytest.mark.parametrize("low,high,status", [(0.10, 0.50, "fail"), (-0.50, -0.10, "fail"),
                                             (-0.10, 0.50, "pass")])
def test_null_decoy_fires_on_an_exposure_interval_that_excludes_no_effect(low, high, status):
    data = estimate_record("null_decoy")
    data["result"]["effects"]["linear"][0].update(ci_low=low, ci_high=high)
    got = outcome(C.null_decoy, data)
    assert got.outcome == status


@pytest.mark.parametrize("low,status", [(0.52, "fail"), (0.45, "pass")])
def test_null_decoy_fires_on_a_held_out_auc_whose_interval_lies_above_no_skill(low, status):
    data = predict_record("null_decoy")
    data["result"]["intervals"]["auc"]["ci_low"] = low
    assert outcome(C.null_decoy, data).outcome == status


def test_null_decoy_does_not_pass_a_result_with_no_interval_to_judge():
    data = predict_record("null_decoy")
    data["result"].update(intervals=None, intervals_why="the fit changed after the opening")
    got = outcome(C.null_decoy, data)
    assert got.outcome == "not_checked"
    assert got.unchecked == ["the fit changed after the opening"]


def test_null_decoy_holds_a_run_that_reported_no_result():
    data = estimate_record("null_decoy")
    data["result"].update(effects=None, refused="The plan was never locked.")
    assert outcome(C.null_decoy, data).outcome == "pass"


# ── 6. robustness ────────────────────────────────────────────────────────────


def test_robustness_is_silent_on_a_run_that_exported_on_its_own():
    assert outcome(C.robustness, predict_record()).outcome == "pass"


def test_robustness_fires_on_a_run_that_never_exported_or_was_helped():
    data = predict_record()
    data["export"]["reached"] = False
    assert outcome(C.robustness, data).problems == [
        "The analyst's own export never returned the bundle."]
    helped = predict_record()
    helped["run"]["help"] = ["restarted the server"]
    assert outcome(C.robustness, helped).problems == [
        "A person helped 1 time: restarted the server"]


# ── the script over a corpus ─────────────────────────────────────────────────


def test_the_script_runs_every_check_over_the_corpus_and_its_failed_runs(tmp_path, capsys):
    write(predict_record(), "run-1", tmp_path)
    broken = estimate_record()
    broken["decisions"][0]["materiality"] = {"source": None}
    write(broken, "run-2", tmp_path)
    assert C.main([str(tmp_path)]) == 1  # the failed run fails logging
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(o["run"], o["check"]) for o in lines] == (
        [("run-1", c) for c in ("seal_integrity", "logging", "single_pass", "known_answer",
                                "null_decoy", "robustness")] + [("run-2", "logging")])
    assert lines[-1]["outcome"] == "fail"
