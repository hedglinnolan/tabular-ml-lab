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
    revert_record,
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
                                      [("seal_opened", {"reason": "again", "target": "y"})],
                                      post_seal=True, minute=20))
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail" and got.problems == ["Record #6 re-seals the held-out rows."]


def test_seal_integrity_fires_on_a_second_opening_of_the_same_outcome():
    data = predict_record()
    data["decisions"].append(decision(6, "open_seal", {"family": "linear", "target": "y"},
                                      [("seal_opened", {"family": "linear", "target": "y"})],
                                      post_seal=True, minute=20))
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail"
    assert got.problems == ["The held-out rows of `y` were opened 2 times (records #5, #6)."]


def test_seal_integrity_fires_when_a_record_after_the_opening_reaches_the_result():
    data = predict_record()
    data["decisions"].append(decision(6, "select_models", {"models": ["linear"]},
                                      [("models", {"models": ["linear"]})],
                                      post_seal=True, minute=20))
    data["result"]["decisions"] = [1, 3, 6]
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail" and "#6" in got.problems[0]


def test_seal_integrity_fires_when_the_result_is_not_the_openings_scores():
    data = predict_record()
    data["result"]["scores"]["linear"]["auc"] = 0.75  # a later fit's score, reported instead
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result's scores are not those the opening at record #5 "
                            "recorded.",
                            "The result's scores are not the held-out scores its fit #2 serves."]


# DB-V3: the result is tied to the fit it names, whatever opening or lock it names.


def test_seal_integrity_fires_on_a_record_after_the_seal_in_the_result_when_no_opening_is_named():
    """DB-V3 (a): a post-seal record #6 among the result's records, the result naming no
    opening."""
    data = predict_record()
    data["decisions"].append(decision(6, "select_models", {"models": ["linear"]},
                                      [("models", {"models": ["linear"]})],
                                      post_seal=True, minute=20))
    data["result"].update(opening=None, decisions=[1, 3, 4, 6])
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail"
    assert got.problems == [
        "Records #6, made after the seal's opening, are among those the result's fit read.",
        "The result's records (#1, #3, #4, #6) are not those its fit #2 read (#1, #3, #4).",
        "The result reports held-out scores but names no opening of the seal."]


def test_seal_integrity_fires_on_estimates_that_are_not_its_named_fits():
    """DB-V3 (b): the post-lock fit's estimate reported under the locked fit's name."""
    data = estimate_record()
    data["result"]["effects"] = copy.deepcopy(data["attempts"][1]["effects"])
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail"
    assert got.problems == ["The result's estimates are not those its fit #1 serves."]


def test_seal_integrity_fires_on_a_named_fit_that_read_a_change_made_with_the_estimates_in_view():
    """DB-V3 (c): the result names fit 2, which read record #5, while its own records stop at
    #4."""
    data = estimate_record()
    data["result"]["attempt"] = "fit2"
    got = outcome(C.seal_integrity, data)
    assert got.outcome == "fail"
    assert got.problems[:2] == [
        "Records #5, made with the estimates in view, are among those the result's fit read.",
        "The result's records (#1, #2, #3, #4) are not those its fit #2 read (#1, #2, #3, #4, "
        "#5)."]


def test_seal_integrity_reads_the_locked_fit_from_the_servers_mark_not_the_plans_digest():
    """DRV-INT: a fit and its lock can differ in an answer the fit does not read, so the fit the
    lock was recorded on is the one GET …/fits marks (``locked``), whatever its own plan's
    digest: such a result passes. A result whose fit is not marked fails, digest or not."""
    data = estimate_record()
    data["attempts"][0]["plan_digest"] = "another plan"
    assert outcome(C.seal_integrity, data).outcome == "pass"
    unmarked = estimate_record()
    unmarked["attempts"][0]["locked"] = False
    got = outcome(C.seal_integrity, unmarked)
    assert got.outcome == "fail"
    assert got.problems == ["The result's fit #1 is not the fit the lock at record #4 was "
                            "recorded on."]


def test_seal_integrity_fires_on_held_out_scores_its_named_fit_does_not_serve():
    data = predict_record()
    data["attempts"][1]["holdout"]["linear"]["auc"] = 0.69
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result's scores are not the held-out scores its fit #2 serves."]


def test_seal_integrity_does_not_pass_scores_its_named_fit_serves_none_of():
    data = predict_record()
    data["attempts"][1].update(holdout=None, holdout_why="The held-out rows are sealed.")
    got = outcome(C.seal_integrity, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == ["The result's fit #2 serves no held-out scores to compare with the "
                             "opening's: The held-out rows are sealed."]


def test_seal_integrity_fires_on_a_named_fit_that_is_not_the_one_the_opening_was_recorded_on():
    data = predict_record()
    data["attempts"][1]["changed_after_seal"] = True
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result's fit #2 is not the fit the opening at record #5 was "
                            "recorded on."]


def test_seal_integrity_fires_on_estimates_with_no_lock_named():
    data = estimate_record()
    data["result"]["lock"] = None
    got = outcome(C.seal_integrity, data)
    assert got.problems == ["The result reports estimates but names no lock of the plan."]


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


@pytest.mark.parametrize("seen,status", [(False, "pass"), (True, "fail")])
def test_seal_integrity_reads_a_record_after_the_lock_by_the_servers_mark(seen, status):
    """DB-V9: a look (``view_outcome``, outside the plan's digest) posted while a held fit ran,
    after the lock and before any estimate was served, is recorded unmarked and lands among the
    locked fit's records, its plan still the lock's: not a record made with the estimates in view.
    The same record marked ``after_estimates`` is."""
    data = estimate_record()
    look = decision(5, "view_outcome", {"view": "distribution", "target": "y"},
                    [("outcome_views:distribution:y", {"view": "distribution"})],
                    after_estimates=seen, minute=5)
    data["decisions"][4] = look  # the post-lock exclusions change gives way to the look
    data["attempts"] = data["attempts"][:1]
    data["attempts"][0]["decisions"] = [1, 2, 3, 4, 5]
    data["result"]["decisions"] = [1, 2, 3, 4, 5]
    got = outcome(C.seal_integrity, data)
    assert got.outcome == status
    assert got.problems == ([] if not seen else [
        "Records #5, made with the estimates in view, are among those the result's fit read."])


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


def test_logging_is_silent_on_a_fit_whose_only_change_is_a_revert_with_its_note():
    assert outcome(C.logging_complete, revert_record()).outcome == "pass"
    quiet = revert_record()
    quiet["decisions"][4]["reason"] = None
    quiet["attempts"][3]["reasons"] = []
    got = outcome(C.logging_complete, quiet)
    assert got.problems == ["Fit #4 has no reason: no record that changed before it (records #5) "
                            "has a note.", "Record #5 (revert) has no reason."]


def test_logging_does_not_pass_a_fit_whose_change_is_credited_to_no_record():
    data = revert_record()
    data["attempts"][3].update(changed=[], reasons=[], withdrawn=[{"seq": 3, "by": None}])
    data["decisions"][4]["materiality"] = {"source": None, "why": "constructed: none measured"}
    got = outcome(C.logging_complete, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == ["Fit #4's change from the fit before (records #3 leaving or "
                             "returning to force) is credited to no record, so its reason is not "
                             "known."]


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


FIT_UNSERVED = ("This fit cannot be matched to your recorded decisions: at no point in your "
                "decision log, up to when the fit completed, were the answers those it was "
                "computed from, so none of its numbers is served.")


def _unserved(data, i):
    """Fit ``i`` (from 0) as ``GET …/fits`` serves a fit whose records it cannot read
    (``FitRecord.decisions`` null, ``runs.ANSWERS_UNREAD``): no records, no change known, no
    numbers, and the engine's why."""
    data["attempts"][i].update(decisions=None, changed=None, reasons=[], withdrawn=[],
                               reinstated=[], movement=[], scores={},
                               movement_why="This fit cannot be matched to your recorded "
                                            "decisions.", withheld=FIT_UNSERVED)
    return data


def test_logging_does_not_judge_a_fit_whose_records_the_server_does_not_serve():
    """DRV-INT (DRV-B's verifier, P4): fit 2's records are not served, so whether its change has a
    reason is not known: not checked, with the engine's own why, never failed as "no reason". The
    other way, with every fit's records served, logging passes (the first logging test)."""
    data = _unserved(predict_record(), 1)
    data["result"]["decisions"] = [1, 3, 4]
    data["result"].update(attempt=None)
    got = outcome(C.logging_complete, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == [f"Fit #2's records are not served, so its change from the fit before "
                             f"and its reason are not known: {FIT_UNSERVED}"]


def test_logging_does_not_judge_the_fit_after_one_whose_records_are_not_served():
    data = revert_record()
    data = _unserved(data, 1)
    data["attempts"][2].update(changed=None, reasons=[])
    data["decisions"][2]["materiality"] = {"source": None, "why": "constructed: none measured"}
    data["decisions"][3]["materiality"] = {"source": None, "why": "constructed: none measured"}
    got = outcome(C.logging_complete, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == [
        f"Fit #2's records are not served, so its change from the fit before and its reason are "
        f"not known: {FIT_UNSERVED}",
        f"Fit #3's change from the fit before is not known, since fit #2's records are not "
        f"served: {FIT_UNSERVED}"]


def test_logging_does_not_judge_a_fit_whose_change_the_log_does_not_account_for():
    from research.swarm.assemble import UNACCOUNTED

    data = predict_record()
    data["attempts"][1].update(changed=None, reasons=[], withdrawn=[])
    data["decisions"][3]["materiality"] = {"source": None, "why": UNACCOUNTED}
    got = outcome(C.logging_complete, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == [f"Fit #2's change from the fit before is not known: {UNACCOUNTED}"]


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


def test_single_pass_notes_count_one_fit_as_one():
    """DRV-INT (the dry run's decoy): the note names one fit "1 fit", two "2 fits"."""
    got = outcome(C.single_pass, _single(predict_record()))
    assert got.notes == ["0 of 5 decisions made before the seal's opening or the plan's lock had "
                         "an alternative examined; 1 fit read only those decisions."]
    got = outcome(C.single_pass, predict_record())
    assert got.notes[0].endswith("; 2 fits read only those decisions.")


def test_single_pass_counts_no_option_tried_or_fitted_after_the_seal_was_opened():
    data = _single(predict_record())
    data["decisions"].append(decision(
        6, "select_models", {"models": ["elastic_net"]},
        [("models", {"models": ["elastic_net"]})], post_seal=True, minute=20,
        examined=[{"question": "models", "value": {"models": ["linear"]}, "how": "fit",
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
    data["decisions"][1]["examined"] = [{"question": "models",
                                         "value": {"models": ["boosted_trees"]},
                                         "how": "preview", "at": f"2026-10-18T10:{minute:02d}:00Z",
                                         "refused": None, "alternative": True}]
    assert outcome(C.single_pass, data).outcome == status


def test_an_option_examined_that_is_the_one_chosen_is_not_exploring():
    data = _single(predict_record())
    data["decisions"][1]["examined"] = [{"question": "models",
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


def _read_again(*later):
    """The Predict record with no opening (its result none reported, so it names no records) and
    ``later`` decisions after record #4: each a confirmation of age's code-or-count reading, or a
    revert, at its own minute."""
    data = predict_record("known_answer")
    data["decisions"] = data["decisions"][:4]
    data["result"] = {"purpose": "prediction", "attempt": None, "plan_digest": None,
                      "decisions": None, "refused": "The held-out rows were never opened.",
                      "refused_kind": "not_reported"}
    for seq, what in later:
        if what.startswith("r"):
            data["decisions"].append(decision(seq, "revert", {"decision_id": what},
                                              [(f"revert:{what}", {"decision_id": what})],
                                              minute=14 + seq))
        else:
            option = {"reading": "code_or_count", "column": "age", "value": what}
            data["decisions"].append(decision(seq, "confirm_reading", option,
                                              [("code_or_count:age", what)], minute=14 + seq))
    return data


def test_known_answer_judges_the_reading_in_force_at_the_logs_end_when_the_result_names_none():
    """DRV-INT (DRV-B's verifier, P2): the result names no records, so each reading is judged by
    the answer in force at the log's end (the log's own revert rule): #6 answers age as a code
    (wrong), #7 as an amount (right), #8 reverts #7, so #6's wrong answer is in force: it fails.
    The other way: #7 reverts #6, so #3's amount is in force: it passes."""
    wrong = outcome(C.known_answer, _read_again((6, "code"), (7, "amount"), (8, "r7")),
                    truths=TRUTH)
    assert wrong.outcome == "fail"
    assert wrong.problems == ["Record #6 answers the code_or_count:age reading as `code`; the "
                              "fixture declares `amount`."]
    right = outcome(C.known_answer, _read_again((6, "code"), (7, "r6")), truths=TRUTH)
    assert (right.outcome, right.problems) == ("pass", [])


def test_a_result_with_no_row_of_the_known_answers_family_says_which_family():
    """DRV-INT (the dry run): the known answer names the linear family, and the result holds only
    the elastic net's row of `x`: the problem names the family it looked for, since the result
    does hold an estimate of `x`. With no family named, a result with no row of `x` says so as
    before."""
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": 0.3}
    data["result"]["effects"] = {"elastic_net": copy.deepcopy(data["result"]["effects"]["linear"])}
    data["attempts"][0]["effects"] = copy.deepcopy(data["result"]["effects"])
    got = outcome(C.known_answer, data, truths={"table.csv": {}})
    assert got.problems == ["The result holds no `linear` estimate of `x`, the known answer's."]
    data["dataset"]["known"] = {"feature": "z", "value": 0.3}
    got = outcome(C.known_answer, data, truths={"table.csv": {}})
    assert got.problems == ["The result holds no estimate of `z`, the known answer's."]


def test_known_answer_holds_a_refused_result():
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": 9.0}
    data["result"].update(effects=None, refused="The plan was never locked.",
                          refused_kind="not_reported")
    assert outcome(C.known_answer, data, truths={"table.csv": {}}).outcome == "pass"
    data["result"].update(refused="The locked fit serves no exposure rows.",
                          refused_kind="not_served")
    assert outcome(C.known_answer, data, truths={"table.csv": {}}).outcome == "pass"


# DB-V5: a result the record could not identify is not a refusal.


def _unknown(data):
    data["result"].update(attempt=None, decisions=None, effects=None,
                          refused="No fit record holds the plan the lock at record #4 recorded.",
                          refused_kind="not_known")
    return data


def test_known_answer_does_not_judge_a_result_the_record_cannot_know():
    """The known answer 9.0 lies outside every interval the record could hold (constructed): a
    result whose fit is not identified is neither right nor refused, so it is not checked."""
    data = _unknown(estimate_record("known_answer"))
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": 9.0}
    got = outcome(C.known_answer, data, truths={"table.csv": {}})
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == ["The record cannot say what result the run reported, so it is not "
                             "checked: No fit record holds the plan the lock at record #4 "
                             "recorded."]


def test_null_decoy_does_not_judge_a_result_the_record_cannot_know():
    got = outcome(C.null_decoy, _unknown(estimate_record("null_decoy")))
    assert (got.outcome, got.notes) == ("not_checked", [])
    assert got.unchecked[0].startswith("The record cannot say what result the run reported")


def test_a_result_with_no_numbers_names_its_kind():
    data = estimate_record()
    data["result"].update(effects=None, refused="The plan was never locked.")
    with pytest.raises(ValueError, match="which kind of none"):
        validate(data)
    numbered = estimate_record()
    numbered["result"]["refused_kind"] = "not_known"
    with pytest.raises(ValueError, match="which kind of none"):
        validate(numbered)


# DB-V7: an estimate with no interval is not a refusal.


@pytest.mark.parametrize("value", [0.0, 0.9])
def test_known_answer_does_not_judge_an_estimate_with_no_interval(value):
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": value}
    data["result"]["effects"]["linear"][0].update(estimate=0.9, ci_low=None, ci_high=None)
    got = outcome(C.known_answer, data, truths={"table.csv": {}})
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == [f"`linear` estimates `x` at 0.9 with no interval, so whether it holds "
                             f"the known answer {value} is not checked."]


def test_known_answer_holds_a_row_the_engine_did_not_estimate():
    data = estimate_record("known_answer")
    data["dataset"]["known"] = {"family": "linear", "feature": "x", "value": 9.0}
    data["result"]["effects"]["linear"][0].update(estimate=None, ci_low=None, ci_high=None)
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
    data["result"].update(effects=None, refused="The plan was never locked.",
                          refused_kind="not_reported")
    got = outcome(C.null_decoy, data)
    assert (got.outcome, got.notes) == ("pass", ["No result was reported: The plan was never "
                                                 "locked."])
    data["result"].update(refused="The locked fit serves no exposure rows.",
                          refused_kind="not_served")
    got = outcome(C.null_decoy, data)
    assert (got.outcome, got.notes) == ("pass", ["The engine served no result: The locked fit "
                                                 "serves no exposure rows."])


def test_null_decoy_does_not_judge_an_estimate_with_no_interval():
    data = estimate_record("null_decoy")
    data["result"]["effects"]["linear"][0].update(estimate=0.9, ci_low=None, ci_high=None)
    got = outcome(C.null_decoy, data)
    assert (got.outcome, got.problems) == ("not_checked", [])
    assert got.unchecked == ["`linear` reports `x` at 0.9 with no interval, so whether it excludes "
                             "no effect is not checked."]
    refused = estimate_record("null_decoy")
    refused["result"]["effects"]["linear"][0].update(estimate=None, ci_low=None, ci_high=None)
    assert outcome(C.null_decoy, refused).outcome == "pass"  # not estimated: no finding


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
