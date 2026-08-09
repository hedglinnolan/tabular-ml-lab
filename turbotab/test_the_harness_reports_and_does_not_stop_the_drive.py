"""`turbotab/devchecks.py` — every check, induced.

**A check that nothing triggers is a check that does not exist** — `GUIDED-019`,
and the reason this file is longer than the module's happy path deserves. A
harness whose checks have never fired is a harness nobody knows the shape of,
and the first drive is the worst time to find out that the assertion which was
going to catch the bug had a typo in it.

So every check below is exercised twice: once against a state that satisfies it,
where it must stay silent, and once against a state built to break it, where it
must fire and name the thing that broke. The induced states are deliberately
minimal — a dict with one wrong number in it — because a check that only fires
on an elaborate setup is a check that will not fire on a real drive either.

Three properties beyond the individual checks, and they are the ones that decide
whether the harness is usable at all:

* **Off is the default and off costs nothing.** No directory, no monitoring, no
  session, and `check_transition` returns an empty list without looking at the
  state it was handed.
* **A violation records and continues.** One bug must not end a drive — the
  driver is looking for the second and third bug too.
* **The harness never reaches the app.** A failure inside the harness becomes a
  violation named after the harness, never an exception in an endpoint. The
  first draft raised `KeyError` inside the upload endpoint and took the whole
  drive with it, which is why `safely()` exists.

Run:  TURBOTAB_DEV_CHECKS=1 turbotab/.venv/bin/python -m pytest \\
          turbotab/test_the_harness_reports_and_does_not_stop_the_drive.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import devchecks                                        # noqa: E402
from turbotab.project import AnalysisProject                          # noqa: E402


@pytest.fixture
def on(tmp_path, monkeypatch):
    """The harness, on, writing to a throwaway directory."""
    monkeypatch.setenv(devchecks.ENV_FLAG, "1")
    session = devchecks.reset_for_test(tmp_path / "drive")
    yield session
    devchecks.stop_listening()
    devchecks.reset_for_test(None)


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(devchecks.ENV_FLAG, raising=False)
    devchecks.reset_for_test(None)
    yield
    devchecks.reset_for_test(None)


def _project() -> AnalysisProject:
    df = pd.DataFrame({"pid": ["a", "a", "b", "b", "c", "c"],
                       "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "y": [0, 1, 0, 1, 0, 1]})
    return AnalysisProject.from_dataframe(df, "t.csv")


def _checks(vs):
    return {v.check for v in vs}


# ── off is the default, and off costs nothing ────────────────────────────────

def test_the_harness_is_off_unless_asked_for(off):
    assert devchecks.enabled() is False
    assert devchecks.session() is None
    assert devchecks.check_transition(_project(), {}, {"decisions": []},
                                      {"kind": "set_target"}) == []
    # And the noisiest entry points are safe to call unconditionally, which is
    # what lets the call sites stay uncluttered by `if enabled():`.
    devchecks.swallowed("nowhere", ValueError("x"))
    devchecks.capture_action({"kind": "note"}, None, None)
    devchecks.capture_console("error", "boom")
    assert devchecks.write_index() is None


def test_a_session_directory_is_created_only_on_first_use(on, tmp_path):
    assert on.root.exists()
    assert (on.root / "README.md").exists()
    assert (on.root / "dom").is_dir()


# ── 1 · every number displayed traces to a value in the record ───────────────

def test_a_number_with_no_source_in_the_record_is_a_violation(on):
    clean = {"n_rows": 600, "n_columns": 17,
             "decisions": [{"kind": "note", "text": "600 rows, 17 columns.",
                            "payload": {}}],
             "disclosures": {}}
    assert devchecks.every_number_displayed_traces_to_the_record(clean) == []

    invented = {"n_rows": 600, "n_columns": 17,
                "decisions": [{"kind": "note",
                               "text": "600 rows, 17 columns, 42 of them useful.",
                               "payload": {}}],
                "disclosures": {}}
    vs = devchecks.every_number_displayed_traces_to_the_record(invented)
    assert [v.check for v in vs] == ["number_with_no_source"]
    assert vs[0].detail["number"] == 42.0


def test_a_column_name_carrying_digits_is_not_read_as_a_claim(on):
    """`bp_1`, `mz_0001`, `item_05`. Every one would read as an unsupported
    number, and a harness that cries wolf on column names gets switched off."""
    state = {"n_rows": 80,
             "decisions": [{"kind": "apply",
                            "text": "`mz_0001` and `bp_12` were coerced.",
                            "payload": {}}],
             "disclosures": {}}
    assert devchecks.every_number_displayed_traces_to_the_record(state) == []


def test_a_fraction_is_supported_by_the_percentage_it_renders_as(on):
    state = {"lockbox": {"fraction": 0.15, "n_test": 90},
             "decisions": [],
             "disclosures": {"seal": "90 rows (15%) are held out."}}
    assert devchecks.every_number_displayed_traces_to_the_record(state) == []


# ── 2 · the seal, recomputed live ────────────────────────────────────────────

def test_an_exploratory_seal_that_renders_as_a_clean_lock_is_a_violation(on):
    project = _project()
    honest = {"lockbox": {"seal_basis": "undetermined", "n_test": 1,
                          "fraction": 0.15, "labels": [0]},
              "disclosures": {"exploratory": True}}
    assert "an_exploratory_seal_rendered_as_a_clean_lock" not in _checks(
        devchecks.the_seal_states_its_own_basis(project, honest))

    lying = {"lockbox": {"seal_basis": "undetermined", "n_test": 1,
                         "fraction": 0.15, "labels": [0]},
             "disclosures": {"exploratory": False}}
    assert "an_exploratory_seal_rendered_as_a_clean_lock" in _checks(
        devchecks.the_seal_states_its_own_basis(project, lying))


def test_a_grouped_seal_with_a_subject_on_both_sides_is_a_violation(on):
    """The thing the seal is FOR, recomputed against the frame rather than
    trusted. `IMPORT-020` is what a clean-looking lock over a real leak costs."""
    project = _project()
    clean = {"lockbox": {"seal_basis": "grouped", "group_col": "pid",
                         "labels": [0, 1], "n_test": 2, "fraction": 0.33,
                         "group_noun": "people"},
             "disclosures": {}}
    assert "a_grouped_seal_has_subjects_on_both_sides" not in _checks(
        devchecks.the_seal_states_its_own_basis(project, clean))

    # One row of person "a" held out, the other left in training. A grouped
    # seal that does this is the leak the whole constitution exists to prevent.
    leaking = {"lockbox": {"seal_basis": "grouped", "group_col": "pid",
                           "labels": [0], "n_test": 1, "fraction": 0.16,
                           "group_noun": "people"},
               "disclosures": {}}
    vs = devchecks.the_seal_states_its_own_basis(project, leaking)
    assert "a_grouped_seal_has_subjects_on_both_sides" in _checks(vs)
    assert any("a" in v.detail.get("examples", []) for v in vs)


def test_a_seal_basis_outside_the_four_is_a_violation(on):
    bad = {"lockbox": {"seal_basis": "clean", "labels": [0], "n_test": 1,
                       "fraction": 0.1}, "disclosures": {}}
    assert "seal_basis_is_not_one_of_the_four" in _checks(
        devchecks.the_seal_states_its_own_basis(_project(), bad))


# ── 3 · decision sentences match their records ───────────────────────────────

def test_a_grain_decision_recording_a_basis_its_answer_does_not_produce(on):
    honest = {"decisions": [{"kind": "set_grain", "text": "Recorded.",
                             "payload": {"answer": "not_sure", "group_col": None,
                                         "basis": "undetermined"}}]}
    assert devchecks.decision_sentences_match_their_records(honest) == []

    # "I'm not sure" cannot produce a verified cross-sectional seal. That is
    # constitution §03's exact failure: two different claims rendering as one.
    lying = {"decisions": [{"kind": "set_grain", "text": "Recorded.",
                            "payload": {"answer": "not_sure", "group_col": None,
                                        "basis": "cross_sectional"}}]}
    assert "a_grain_decision_records_a_basis_its_answer_does_not_produce" in \
        _checks(devchecks.decision_sentences_match_their_records(lying))


def test_a_seal_decision_that_disagrees_with_its_lockbox(on):
    state = {"lockbox": {"n_test": 90, "seal_basis": "grouped"},
             "decisions": [{"kind": "seal_lockbox",
                            "text": "A test set was sealed.",
                            "payload": {"n_test": 12, "seal_basis": "grouped"}}]}
    assert "the_seal_decision_and_the_lockbox_disagree_on_n" in \
        _checks(devchecks.decision_sentences_match_their_records(state))


def test_a_decision_sentence_carrying_a_number_its_payload_does_not(on):
    state = {"n_rows": 100, "n_columns": 5,
             "decisions": [{"kind": "set_eligibility",
                            "text": "7 participants were excluded.",
                            "payload": {"n_excluded": 7}}]}
    assert devchecks.decision_sentences_match_their_records(state) == []

    state["decisions"][0]["text"] = "9 participants were excluded."
    assert "an_eligibility_decision_does_not_state_the_n_it_removed" in \
        _checks(devchecks.decision_sentences_match_their_records(state))


# ── 4 · a deferred transform leaves the table byte-identical ─────────────────

def test_a_deferred_action_that_changes_the_working_table(on):
    """Constitution §06's canonical leak, which is invisible from the outside:
    the table simply has better numbers in it afterwards."""
    before = {"fingerprint": "aaaa", "n_rows": 600}
    same = {"fingerprint": "aaaa", "n_rows": 600}
    moved = {"fingerprint": "bbbb", "n_rows": 600}

    assert devchecks.a_deferred_transform_leaves_the_table_byte_identical(
        "defer_feature", before, same) == []
    assert "a_deferred_action_changed_the_working_table" in _checks(
        devchecks.a_deferred_transform_leaves_the_table_byte_identical(
            "defer_feature", before, moved))
    # An action that is SUPPOSED to touch the table is not this check's business.
    assert devchecks.a_deferred_transform_leaves_the_table_byte_identical(
        "add_feature", before, moved) == []


# ── 5 · after an edit exactly the right things are stale ─────────────────────

def test_the_cascade_firing_the_wrong_number_of_times(on):
    before = {"stale_downstream": []}
    once = {"stale_downstream": [{"why": "a column was added"}]}
    twice = {"stale_downstream": [{"why": "a"}, {"why": "b"}]}

    assert devchecks.after_an_edit_exactly_the_right_things_are_stale(
        "add_feature", before, once) == []
    # Too FEW is a wrong number downstream.
    assert "the_cascade_fired_the_wrong_number_of_times" in _checks(
        devchecks.after_an_edit_exactly_the_right_things_are_stale(
            "add_feature", before, before))
    # Too MANY is a lost afternoon, and trains the user to ignore the cascade.
    assert "the_cascade_fired_the_wrong_number_of_times" in _checks(
        devchecks.after_an_edit_exactly_the_right_things_are_stale(
            "add_feature", before, twice))
    # Deferring executes nothing, so it invalidates nothing.
    assert "the_cascade_fired_the_wrong_number_of_times" in _checks(
        devchecks.after_an_edit_exactly_the_right_things_are_stale(
            "defer_feature", before, once))


# ── 6 · no post-seal operation changes a surviving row's label ───────────────

def test_a_sealed_label_that_no_longer_names_a_row(on):
    project = _project()
    before = {"barrier_raised": True}
    intact = {"lockbox": {"labels": [0, 1]}}
    assert devchecks.no_post_seal_operation_changes_a_surviving_rows_label(
        project, before, intact) == []

    renumbered = {"lockbox": {"labels": [0, 1, 99]}}
    vs = devchecks.no_post_seal_operation_changes_a_surviving_rows_label(
        project, before, renumbered)
    assert "a_sealed_row_label_no_longer_names_a_row" in _checks(vs)
    # Before the barrier there are no identities to preserve.
    assert devchecks.no_post_seal_operation_changes_a_surviving_rows_label(
        project, {"barrier_raised": False}, renumbered) == []


# ── 7 · a finding claiming N features names N ────────────────────────────────

def test_a_finding_that_claims_a_count_it_does_not_name(on):
    honest = {"findings": [{"id": "f1", "title": "3 column(s) have no name",
                            "params": {"columns": ["a", "b", "c"]}}]}
    assert devchecks.a_finding_claiming_n_features_names_n(honest) == []

    lying = {"findings": [{"id": "f1", "title": "5 column(s) have no name",
                           "params": {"columns": ["a", "b", "c"]}}]}
    vs = devchecks.a_finding_claiming_n_features_names_n(lying)
    assert "a_finding_claims_a_count_it_does_not_name" in _checks(vs)

    # "4 groups of columns" beside 40 affected columns is CORRECT, and reading
    # the group count against the column list would make this check unusable on
    # the one finding it fires on most.
    grouped = {"findings": [{"id": "f2",
                             "title": "4 group(s) of columns look like repeats",
                             "affected_columns": ["c%d" % i for i in range(40)],
                             "params": {"families": {"a": [], "b": [], "c": [],
                                                     "d": []}}}]}
    assert devchecks.a_finding_claiming_n_features_names_n(grouped) == []


# ── 8 · router.audit passes before rendering ─────────────────────────────────

def test_the_rendered_plan_is_the_audited_plan(on):
    from ml import router
    questions = router.plan([], target=None, detection=None, step="data",
                            deferred={}, answered=[], recommendations=[],
                            signals=None, missing_columns=[])
    rendered = [q.to_dict() for q in questions]
    assert devchecks.router_audit_passed_before_this_render(questions, rendered) == []
    assert on.audits == 1

    # A render that shows a different number of questions than were audited is
    # a plan nobody checked, dressed as one somebody did.
    assert "the_rendered_plan_is_not_the_audited_plan" in _checks(
        devchecks.router_audit_passed_before_this_render(questions, rendered[:-1])
        if rendered else
        devchecks.router_audit_passed_before_this_render(questions, [{}]))


# ── 9 · every decision taken appears in the record ───────────────────────────

def test_an_action_that_records_nothing(on):
    before = {"decisions": []}
    recorded = {"decisions": [{"kind": "set_grain"}]}
    assert devchecks.every_decision_taken_appears_in_the_record(
        "set_grain", before, recorded) == []
    assert "an_action_was_taken_and_nothing_was_recorded" in _checks(
        devchecks.every_decision_taken_appears_in_the_record(
            "set_grain", before, before))
    # A read that records a decision is the opposite failure, and it puts a
    # thing in the manuscript that nobody decided.
    assert "a_read_recorded_a_decision" in _checks(
        devchecks.every_decision_taken_appears_in_the_record(
            "eligibility_evidence", before, recorded))


# ── the silent wells ─────────────────────────────────────────────────────────

def test_an_explicit_swallow_is_written_down_with_what_the_user_did_not_see(on):
    devchecks.swallowed("engine.preview_fix::changed-cell-count",
                        KeyError("no such column"),
                        "the preview will report zero changed cells")
    assert len(on.swallows) == 1
    row = on.swallows[0]
    assert row["type"] == "KeyError"
    assert row["note"].startswith("the preview")
    lines = (on.root / "swallowed.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["where"].startswith("engine.preview_fix")


def test_the_monitoring_layer_watches_our_files_and_not_the_libraries(on):
    """Layer 2 exists because `ml/import_doctor.py` is frozen and instrumenting
    it would be new construction on a frozen path. So nothing is edited."""
    here = os.path.dirname(os.path.abspath(__file__))
    assert devchecks._watched(os.path.join(here, "engine.py")) is True
    assert devchecks._watched(os.path.join(here, "..", "ml", "import_doctor.py")) is True
    # Not the harness itself, not the tests, not a library.
    assert devchecks._watched(os.path.join(here, "devchecks.py")) is False
    assert devchecks._watched(__file__) is False
    assert devchecks._watched(pd.__file__) is False


def test_a_swallow_in_a_watched_file_is_reported_without_editing_that_file(on):
    """End to end, through `sys.monitoring`, against a real swallow site.

    `_column_repetition` catches `TypeError` on unhashable cells and returns
    `None` — a legitimate fallback, and a place where a column silently stops
    being considered as an identifier. Nothing in `grain.py` was touched to
    make this visible.
    """
    # `AUDIT-039`, `L56-B2`. `start_listening()` returns `False` for FOUR
    # different reasons — devchecks disabled, `sys.monitoring` absent, the
    # listener already on, and another tool holding `DEBUGGER_ID` — and the old
    # skip asserted the fourth without checking. Three of the four are states
    # this test can establish rather than stand down over, and only the last is
    # environmental in the way a missing JS engine is.
    #
    # So each is separated, the three are ASSERTED, and the environmental one
    # skips with the holder's real name instead of a guess: `get_tool()` reports
    # who has the id, which turns "another tool holds it" from a hypothesis into
    # a measurement. An exemption is an argument, not a keyword.
    assert devchecks.enabled(), (
        "devchecks is disabled in this run, so layer 2 reports nothing at all "
        "and every swallow claim in this file is vacuous. That is a "
        "configuration regression, not an environment.")
    monitoring = getattr(sys, "monitoring", None)
    assert monitoring is not None, (
        "`sys.monitoring` is absent; this repository requires 3.12+ and the "
        "monitoring layer is not optional on it")
    assert not devchecks._monitoring_on, (
        "the listener was already running when this test started, so a "
        "previous test left it on and the swallows below would include theirs")

    if not devchecks.start_listening():
        holder = monitoring.get_tool(monitoring.DEBUGGER_ID)
        assert holder is not None, (
            "`start_listening()` refused and no tool holds DEBUGGER_ID, so the "
            "refusal has none of the four causes it is allowed to have")
        pytest.skip(f"the monitoring DEBUGGER_ID is held by {holder!r} — "
                    f"environmental, like a missing JS engine, and named "
                    f"rather than assumed")
    try:
        from turbotab import grain as G
        assert G.repetition_evidence(pd.DataFrame({"a": [[1], [2], [3]] * 10})) == []
    finally:
        devchecks.stop_listening()

    swallowed = [s for s in on.swallows if s["layer"] == "monitoring"]
    assert swallowed, "the monitoring layer reported nothing"
    assert any("grain.py" in s["where"] for s in swallowed), \
        [s["where"] for s in swallowed]


def test_the_apps_own_refusals_are_not_reported_as_swallows(on):
    """`ProjectError` and its siblings are the REFUSE branch of the governing
    rule working correctly. Reporting them would bury the real wells under the
    app speaking properly."""
    for name in ("ProjectError", "EngineRefusal", "HTTPException",
                 "GrainContradiction", "StopIteration"):
        assert name in devchecks._DELIBERATE


# ── index.md leads with violations ───────────────────────────────────────────

def test_the_index_opens_with_violations_not_with_narrative(on):
    devchecks.record_violations(
        [devchecks.Violation("number_with_no_source", "displays 42", {})],
        {"kind": "set_target"})
    path = devchecks.write_index()
    text = path.read_text()
    head = text.split("## ")[1]
    assert head.startswith("1 violation"), text[:400]
    assert "number_with_no_source" in text
    assert text.index("violation") < text.index("Coverage")


def test_a_clean_drive_says_so_in_one_line_and_gets_out_of_the_way(on):
    text = devchecks.write_index().read_text()
    assert "No violations." in text
    assert text.index("No violations.") < text.index("Coverage")


# ── the harness never reaches the app ────────────────────────────────────────

def test_a_failure_inside_the_harness_becomes_a_violation_not_an_exception(on):
    """The rule the harness applies to the app, applied to itself.

    The first draft raised `KeyError` from `write_index` inside the upload
    endpoint and ended the drive with a stack trace pointing at the app. A
    harness that halts on its own bug has cost more than it found.
    """
    def explodes():
        raise RuntimeError("the harness has a bug")

    devchecks.safely(explodes)                             # must not raise
    assert any(v["check"] == "the_harness_itself_failed" for v in on.violations)


def test_a_check_that_raises_does_not_end_the_battery(on):
    def explodes(_state):
        raise RuntimeError("boom")

    vs = devchecks._guard(explodes, {})
    assert [v.check for v in vs] == ["check_itself_failed"]


# ── end to end, through the API, with a bug planted ──────────────────────────

def test_a_planted_wrong_number_in_a_disclosure_is_caught_over_http(on, monkeypatch):
    """The whole harness, driven, against a deliberately planted defect.

    The bug is the most plausible one in this app's design space: a disclosure
    sentence stating a number the record does not hold. The seal is drawn
    correctly, the basis is correct, the count is a count — only the sentence is
    wrong about it. Nothing in the suite constructs this state, which is exactly
    why the harness exists.

    **This test is also the record of a bug in the harness itself.** The first
    version of `every_number_displayed_traces_to_the_record` built ONE supported
    set from the entire project payload — findings, profile, every column
    summary, several hundred numbers on a real table — so almost any small
    integer was "supported" and the check could not fail. Driven against this
    exact planted bug it reported nothing: a green line asserting something
    false, inside the instrument built to catch green lines asserting something
    false. Scoping each claim to the part of the record it is ABOUT is the
    repair, and this is the test that would have caught it.
    """
    from fastapi.testclient import TestClient
    from turbotab import api, grain as grain_mod

    real = grain_mod.seal_disclosure

    def bugged(lockbox):
        # One digit. 90 rows sealed, "40" reported.
        return real(lockbox).replace(f"{lockbox.get('n_test', 0):,}", "40", 1)

    monkeypatch.setattr(grain_mod, "seal_disclosure", bugged)

    client = TestClient(api.app)
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_data", "dietary_recalls.csv")
    with open(fixture, "rb") as fh:
        project = client.post("/project",
                              files={"file": ("dietary_recalls.csv", fh, "text/csv")}).json()
    pid = project["id"]

    def decide(kind, payload):
        return client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "subject": "", "payload": payload})

    decide("set_target", {"column": "hba1c"})
    decide("set_grain", {"answer": "people_repeat", "group_col": "participant_id"})
    # People repeat, so clause 01's bracketed steps come before the seal. The
    # unit is the RECORD, so the 600 rows survive and 90 of them are held out.
    decide("set_repeat_kind", {"kind": "repeats"})
    decide("set_unit_of_analysis", {"unit": "record"})
    decide("set_eligibility", {"answer": "everyone"})
    sealed = decide("seal", {"fraction": 0.15, "seed": 42})

    assert sealed.status_code == 200
    body = sealed.json()
    assert body["lockbox"]["n_test"] == 90
    assert body["disclosures"]["seal"].startswith("40 rows")

    caught = [v for v in on.violations if v["check"] == "number_with_no_source"]
    assert caught, "the planted wrong number was not caught"
    assert any(v["detail"]["number"] == 40.0 for v in caught)
    assert any(v["message"].startswith("disclosure::seal") for v in caught)

    # AND THE DRIVE CONTINUES. One bug must not end it — the driver is looking
    # for the second and third bug too.
    assert decide("settle_features", {"skipped": True}).status_code == 200
    assert decide("settle_preprocess", {"skipped": True}).status_code == 200


def test_the_same_drive_with_no_planted_bug_is_clean(on):
    """The other half, and the half that decides whether the harness is usable.

    A check that fires on a correct drive gets switched off within a day, and
    then it is not a check. This is the same sequence with nothing planted.
    """
    from fastapi.testclient import TestClient
    from turbotab import api

    client = TestClient(api.app)
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_data", "dietary_recalls.csv")
    with open(fixture, "rb") as fh:
        pid = client.post("/project",
                          files={"file": ("dietary_recalls.csv", fh, "text/csv")}).json()["id"]

    def decide(kind, payload):
        return client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "subject": "", "payload": payload})

    decide("set_target", {"column": "hba1c"})
    decide("set_grain", {"answer": "people_repeat", "group_col": "participant_id"})
    decide("set_repeat_kind", {"kind": "repeats"})
    decide("set_unit_of_analysis", {"unit": "record"})
    decide("set_eligibility", {"answer": "everyone"})
    decide("seal", {"fraction": 0.15, "seed": 42})
    client.get(f"/project/{pid}/interview?step=explore")
    decide("defer_feature", {"transform": "standardize", "columns": ["energy_kcal"]})
    decide("settle_features", {"skipped": False})
    decide("settle_preprocess", {"skipped": True})

    assert on.violations == [], [v["message"] for v in on.violations]
    assert len(on.actions) >= 8, "the capture recorded nothing"


# ── STATE-111 · the defect the harness found on its first drive ──────────────

def test_a_serialized_project_does_not_hand_out_the_projects_own_containers():
    """`STATE-111`, found by the 'exactly the right things are stale' check.

    `select_models` appends one stale entry and the harness measured zero,
    because `to_dict()` handed both snapshots the SAME list. Two readings of
    before and after that are one object.

    The other direction is worse and was latent: a caller that appended to what
    it had been given as a *serialization* mutated the project, with nothing
    recording it.
    """
    project = _project()
    snapshot = project.to_dict()

    for key, attribute in [("findings", "findings"),
                           ("engineered", "engineered"),
                           ("deferred_transforms", "deferred_transforms"),
                           ("stale_downstream", "stale_downstream"),
                           ("missingness", "missingness"),
                           ("obligations", "obligations"),
                           ("selected_models", "selected_models")]:
        assert snapshot[key] is not getattr(project, attribute), (
            f"to_dict handed out the project's own {attribute}")

    before = project.to_dict()
    project._mark_stale("something changed")
    after = project.to_dict()
    assert len(after["stale_downstream"]) - len(before["stale_downstream"]) == 1, (
        "a snapshot taken before an action must not move when the action runs")
