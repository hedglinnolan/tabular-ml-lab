"""Questions 4 to 7 — `STATE-108`, `STATE-109`, `STATE-110`.

`OPENING_SEQUENCE.md` §03. Four questions that fire only when the grain answer is
*people repeat*, each gating the next, and the assertions here are **as much
about not firing as about firing** — which is §04's own instruction about these
fixtures.

The pair that carries the argument is `dietary_recalls.csv` and
`clinical_longitudinal.csv`. Same grain answer, same replicate index, opposite
reading, opposite aggregation advice. The only evidence separating them is
spacing, so if the spacing reading is wrong the app gives confidently opposite
advice on two files that look alike — which is why the reading is stated with
its measurement rather than asserted.

Run:  turbotab/.venv/bin/python -m pytest \\
          turbotab/test_the_repeated_measures_chain_fires_only_when_it_should.py -q
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import eligibility as E, engine, grain as G             # noqa: E402
from turbotab import repeats as R                                     # noqa: E402
from turbotab.project import AnalysisProject, ProjectError            # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def project(name: str, target: str) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(pd.read_csv(DATA / f"{name}.csv"), name)
    task = engine.detect_task_type(p.df, target)
    p.set_target(target, task["detected"], task["confidence"], task.get("reasons", []))
    return p


# ── question 4 · the reading, and the evidence it cites ──────────────────────

def test_two_recalls_a_week_apart_read_as_repeats():
    reading = R.read(pd.read_csv(DATA / "dietary_recalls.csv"), "participant_id")
    assert reading["reading"] == R.REPEATS
    assert reading["stated"] is True
    gaps = reading["spacing"]
    assert gaps["column"] == "recall_date"
    assert (gaps["min_days"], gaps["max_days"]) == (3.0, 14.0)
    assert gaps["cv"] > 0.3, "irregular"
    assert gaps["median_days"] < 14, "too close together to be a schedule"
    assert reading["sentence"].startswith(
        "Not asked: these look like repeated measurements of the same quantity")
    assert "recall_date" in reading["sentence"]
    assert "9" in reading["sentence"], "the measurement is cited, not just claimed"


def test_three_visits_ninety_days_apart_read_as_time_points():
    reading = R.read(pd.read_csv(DATA / "clinical_longitudinal.csv"), "subject_id")
    assert reading["reading"] == R.TIME_POINTS
    assert reading["stated"] is True
    assert reading["confidence"] == "high"
    gaps = reading["spacing"]
    assert gaps["cv"] < 0.15, "a schedule is regular by construction"
    assert gaps["median_days"] >= 14
    assert reading["sentence"].startswith(
        "Not asked: these look like different time points")
    assert "schedule" in reading["sentence"]


def test_the_two_fixtures_get_opposite_readings_from_spacing_alone():
    """The pair, stated as one assertion.

    Both have a replicate index numbering each person's records 1, 2, 3. Both
    have a date column. Both have the same grain answer. If the app reads them
    alike it gives confidently opposite advice on one of them.

    Discharges `lockbox-01`: repeats-or-time-points comes after grain.
    """
    dietary = R.read(pd.read_csv(DATA / "dietary_recalls.csv"), "participant_id")
    clinical = R.read(pd.read_csv(DATA / "clinical_longitudinal.csv"), "subject_id")
    assert dietary["replicate_index"] is not None
    assert clinical["replicate_index"] is not None
    assert dietary["reading"] != clinical["reading"]


def test_identical_dates_within_a_person_read_as_repeats():
    df = pd.DataFrame({"pid": ["a", "a", "b", "b", "c", "c"],
                       "when": ["2024-01-01"] * 2 + ["2024-02-01"] * 2
                               + ["2024-03-01"] * 2,
                       "x": range(6)})
    reading = R.read(df, "pid")
    assert reading["reading"] == R.REPEATS
    assert reading["confidence"] == "high"
    assert "same date" in reading["sentence"]


def test_thin_evidence_is_asked_rather_than_guessed():
    """*"Where the evidence is thin, it is asked rather than guessed."*

    Thin is MEASURED — no dates and no replicate index — not felt. Guessing
    here decides whether averaging is correct.

    Discharges `lockbox-01`: repeats-or-time-points is asked, not inferred.
    """
    df = pd.DataFrame({"pid": ["a", "a", "b", "b", "c", "c"],
                       "x": [1.0, 2, 3, 4, 5, 6],
                       "y": [9.0, 8, 7, 6, 5, 4]})
    reading = R.read(df, "pid")
    assert reading["reading"] is None
    assert reading["stated"] is False
    assert reading["sentence"].startswith("Asked rather than stated")


def test_widely_spaced_but_irregular_records_are_asked_not_stated():
    """Unscheduled encounters are still time points, so this case is genuinely
    uncertain and is the one date-bearing reading the app declines to make."""
    rng = np.random.default_rng(3)
    rows = []
    for p in range(30):
        day = 0
        for _ in range(3):
            day += int(rng.integers(20, 400))
            rows.append({"pid": f"P{p}",
                         "when": (pd.Timestamp("2023-01-01")
                                  + pd.Timedelta(days=day)).date().isoformat(),
                         "x": float(rng.normal())})
    reading = R.read(pd.DataFrame(rows), "pid")
    assert reading["reading"] is None
    assert "unscheduled encounters" in reading["sentence"]


def test_the_reading_is_overturnable_and_the_record_says_so():
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_repeat_kind(R.TIME_POINTS, overturned=True)
    assert p.repeat_kind["kind"] == R.TIME_POINTS
    assert p.repeat_kind["overturned"] is True
    assert p.repeat_kind["stated"] is False
    said = p.decisions[-1]
    assert "different time points" in said.text
    assert "overturning the reading the data suggested" in said.text


# ── the chain does not fire when it does not apply ───────────────────────────

def test_none_of_the_four_fires_when_one_row_is_one_person():
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.ONE_ROW_PER_PERSON, acknowledged_contradiction=True)
    for call, args in [(p.set_repeat_kind, (R.REPEATS,)),
                       (p.set_unit_of_analysis, (R.UNIT_PERSON,)),
                       (p.set_aggregation, (R.MEAN,)),
                       (p.set_temporal_prediction, (True,))]:
        with pytest.raises(ProjectError):
            call(*args)


@pytest.mark.parametrize("fixture,target", [
    ("metabolomics_untargeted", "responder"),
    ("survey_instrument", "sought_support"),
    ("genomics_expression", "condition"),
])
def test_the_chain_is_absent_from_the_plan_on_a_cross_sectional_table(fixture, target):
    """`OPENING_SEQUENCE.md` §02: most datasets see four to six questions.

    On a table where every row is a different participant, all four of these are
    not merely skipped — they are not in the plan.
    """
    plan = router.plan([], target=target, detection=None, step="data",
                       deferred={}, answered=["state_lens", "choose_target",
                                              "state_grain"],
                       recommendations=[], signals=None, missing_columns=[],
                       repeats=None)
    router.audit(plan)
    keys = {q.key for q in plan}
    assert keys & {"state_repeat_kind", "state_unit_of_analysis",
                   "state_aggregation", "state_temporal_prediction"} == set()


def test_each_question_gates_the_next():
    """One at a time, which is the interaction model rather than a form."""
    state = {"reading": R.REPEATS, "sentence": "…", "confidence": "medium",
             "kind": None, "unit": None, "menu": None}
    answered = ["state_lens", "choose_target", "state_grain"]

    def keys(st, ans):
        plan = router.plan([], target="y", detection=None, step="data",
                           deferred={}, answered=ans, recommendations=[],
                           signals=None, missing_columns=[], repeats=st)
        router.audit(plan)
        chain = {"state_repeat_kind", "state_unit_of_analysis",
                 "state_aggregation", "state_temporal_prediction"}
        return [q.key for q in plan if q.key in chain]

    assert keys(state, answered) == ["state_repeat_kind"]

    answered = answered + ["state_repeat_kind"]
    state = {**state, "kind": R.REPEATS}
    assert keys(state, answered) == ["state_unit_of_analysis"]

    answered = answered + ["state_unit_of_analysis"]
    state = {**state, "unit": R.UNIT_PERSON, "menu": R.menu(R.REPEATS)}
    assert keys(state, answered) == ["state_aggregation"]

    # And with the unit as the RECORD, aggregation never appears — there is
    # nothing to combine — while temporal prediction does, but only for time
    # points.
    state_record = {**state, "unit": R.UNIT_RECORD, "kind": R.REPEATS}
    assert keys(state_record, answered) == []
    state_time = {**state_record, "kind": R.TIME_POINTS}
    assert keys(state_time, answered) == ["state_temporal_prediction"]


# ── question 6 · the menu inverts ────────────────────────────────────────────

def test_the_mean_is_recommended_for_repeats_with_the_measurement_error_reason():
    menu = R.menu(R.REPEATS, lens=["dietary"])
    assert menu["recommended"] == R.MEAN
    assert menu["marker"] == "derived"
    assert "attenuates diet–outcome associations toward the null" in menu["reason"]
    assert "measurement error" in menu["reason"]
    assert [o["key"] for o in menu["options"]] == list(R.AGGREGATIONS)


def test_the_dietary_reason_is_the_packs_own_sentence_and_not_a_copy(monkeypatch):
    """`GUIDED-026`: two implementations of one rule, with the documented one
    inert.

    `repeats.py` carried its own nearly identical sentences and the pack's copy
    was unreachable — so editing the pack changed nothing, and editing
    `repeats.py` made the pack's stated prior a false description of the app.

    **The assertion is the DEPENDENCY, not an equality**, and the probe is what
    forced that. Asserting `menu["reason"] == prior["reason"]` came back GREEN
    against a `repeats.py` that restated the sentence, because the restated copy
    was character-identical — an identity check passes on a duplicate right up
    until somebody edits one side, which is the drift itself rather than a guard
    against it.

    So the pack's prior is REPLACED here and the rendered sentence has to
    follow. A restating implementation renders the original and fails.
    """
    from turbotab import packs as PK
    prior = PK.priors(["dietary"], "repeat_treatment")[0]
    menu = R.menu(R.REPEATS, lens=["dietary"])
    assert menu["reason"] == prior["reason"]
    assert menu["marker"] == prior["marker"]

    sentinel = ("Replaced for this test only: the rendered sentence must come "
                "from the pack rather than from a copy beside it.")
    pack = PK.PACKS["dietary"]
    patched = tuple(
        # The evidence badge is carried through rather than dropped: a prior is
        # not constructible without one since `GUIDED-047`, and rebuilding it
        # without the badge would be this test asserting a shape the app
        # forbids.
        PK.Prior(question=p.question, marker=p.marker, reason=sentinel,
                 evidence=p.evidence,
                 scope=p.scope, detector=p.detector, values=dict(p.values))
        if p.question == "repeat_treatment" else p
        for p in pack.priors)
    monkeypatch.setitem(PK.PACKS, "dietary",
                        dataclasses.replace(pack, priors=patched))

    assert R.menu(R.REPEATS, lens=["dietary"])["reason"] == sentinel, (
        "the rendered reason did not follow the pack; there are two "
        "implementations of the averaging rule again")
    # And the record can name where the recommendation came from. A user
    # reading "averaging reduces measurement error" is entitled to know which
    # field's convention said so.
    assert menu["from_pack"] == "dietary"
    assert menu["from_pack_label"] == PK.LENS_LABELS["dietary"]


def test_with_no_dietary_lens_the_general_argument_stands_unattributed():
    """The general case is arithmetic about noise, not a domain convention, so
    it survives without a lens — and it is NOT the dietary sentence, because
    that one is about 24-hour recalls and usual intake."""
    from turbotab import packs as PK
    plain = R.menu(R.REPEATS)
    assert plain["recommended"] == R.MEAN
    assert plain["from_pack"] is None
    assert plain["reason"] != PK.priors(["dietary"], "repeat_treatment")[0]["reason"]
    assert "24-hour recall" not in plain["reason"]
    assert "diet" not in plain["reason"]


def test_nothing_is_recommended_for_time_points_and_that_absence_is_the_finding():
    menu = R.menu(R.TIME_POINTS)
    assert menu["recommended"] is None
    assert "averaging them destroys the signal" in menu["reason"]
    # The same four options are still offered. Withholding the mean would be
    # the app deciding, and a user with a reason for it is not wrong.
    assert [o["key"] for o in menu["options"]] == list(R.AGGREGATIONS)


def test_the_unbuilt_summaries_are_named_rather_than_quietly_absent():
    for kind in R.REPEAT_KINDS:
        filed = R.menu(kind)["filed"]
        for missing in ("Slope", "area under the curve", "usual-intake"):
            assert missing in filed


# ── question 6 · what aggregation actually does ──────────────────────────────

def test_aggregating_the_mean_produces_one_row_per_person():
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_PERSON)
    before = p.df.copy()
    p.set_aggregation(R.MEAN)

    assert len(p.df) == 300
    assert p.df["participant_id"].is_unique
    assert p.aggregation["n_before"] == 600
    assert p.aggregation["n_after"] == 300
    # The values are means, checked against one person rather than trusted.
    who = before["participant_id"].iloc[0]
    expected = before.loc[before["participant_id"] == who, "energy_kcal"].mean()
    got = float(p.df.loc[p.df["participant_id"] == who, "energy_kcal"].iloc[0])
    assert got == pytest.approx(expected)
    assert "600 rows became 300" in p.decisions[-1].text


def test_a_change_score_does_not_difference_the_outcome():
    """A change score on the target asks a different research question.

    "Who improved" is not "who is ill", and silently substituting one for the
    other would be the app choosing the paper's question.
    """
    p = project("clinical_longitudinal", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "subject_id")
    p.set_repeat_kind(R.TIME_POINTS)
    p.set_unit_of_analysis(R.UNIT_PERSON)
    before = p.df.copy()
    p.set_aggregation(R.CHANGE)

    who = before["subject_id"].iloc[0]
    block = before.loc[before["subject_id"] == who].sort_values("visit_date")
    row = p.df.loc[p.df["subject_id"] == who].iloc[0]
    assert float(row["weight_kg"]) == pytest.approx(
        float(block["weight_kg"].iloc[-1]) - float(block["weight_kg"].iloc[0]))
    assert float(row["hba1c"]) == pytest.approx(float(block["hba1c"].iloc[-1]))
    assert p.aggregation["target_not_differenced"] is True
    assert "asks a different question" in p.decisions[-1].text


def test_a_categorical_that_varied_within_a_person_is_named_in_the_receipt():
    """Taking the first value is a compromise, and a compromise nobody is told
    about is indistinguishable from a computation."""
    df = pd.DataFrame({"pid": ["a", "a", "b", "b"],
                       "site": ["North", "South", "East", "East"],
                       "sex": ["F", "F", "M", "M"],
                       "x": [1.0, 3.0, 5.0, 7.0]})
    out = R.aggregate(df, "pid", R.MEAN)
    assert out["varying_categoricals"] == ["site"]
    assert "`site`" in out["sentence"]
    assert "sex" not in out["sentence"]


# ── the identity barrier · why aggregation cannot move ───────────────────────

def test_aggregation_is_refused_once_the_seal_names_rows():
    """Decision A, and the reason question 6 sits where it does.

    Combining rows after the seal would leave the lockbox's labels naming rows
    that no longer exist, with nothing able to detect it — the lockbox would
    still look perfectly well-formed.

    Discharges `lockbox-01`: aggregation cannot move.
    """
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, "hba1c", "regression", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    with pytest.raises(ProjectError, match="identity barrier"):
        p.set_aggregation(R.MEAN)
    for call, args in [(p.set_repeat_kind, (R.TIME_POINTS,)),
                       (p.set_unit_of_analysis, (R.UNIT_PERSON,)),
                       (p.set_temporal_prediction, (True,))]:
        with pytest.raises(ProjectError, match="already sealed"):
            call(*args)


def test_the_seal_is_refused_until_the_chain_is_settled():
    """Clause §01's bracketed steps are not optional when the shape calls for
    them: the seal is drawn over rows, and the chain decides what a row IS.

    Discharges `lockbox-01`: the bracketed steps sit before the seal.
    """
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, "hba1c", "regression", p.grain)

    with pytest.raises(ProjectError, match="repeated measurements or different"):
        p.seal_lockbox(drawn["labels"], **drawn["disclosure"])
    p.set_repeat_kind(R.REPEATS)
    with pytest.raises(ProjectError, match="what one row means"):
        p.seal_lockbox(drawn["labels"], **drawn["disclosure"])
    p.set_unit_of_analysis(R.UNIT_PERSON)
    with pytest.raises(ProjectError, match="have not been combined yet"):
        p.seal_lockbox(drawn["labels"], **drawn["disclosure"])


def test_aggregating_first_then_sealing_leaves_a_seal_of_whole_people():
    """The order that works, end to end, and the number the seal reports."""
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_PERSON)
    p.set_aggregation(R.MEAN)
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, "hba1c", "regression", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    assert p.lockbox["n_total"] == 300
    held = set(p.lockbox["labels"])
    assert len(held) == p.lockbox["n_test"]
    # One row per person now, so a person cannot be on both sides by
    # construction — which is the state aggregation was for.
    assert p.df.loc[[l for l in p.df.index if l in held],
                    "participant_id"].is_unique
    p.assert_identity_intact()


# ── question 7 · fires only when time points survive as rows ─────────────────

def test_temporal_prediction_is_refused_for_repeats():
    """Discharges `lockbox-01`: temporal prediction fires only when time points
    survive as rows."""
    p = project("dietary_recalls", "hba1c")
    p.set_grain(G.PEOPLE_REPEAT, "participant_id")
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    with pytest.raises(ProjectError, match="no earlier and later"):
        p.set_temporal_prediction(True)


def test_temporal_prediction_is_refused_once_the_rows_are_combined():
    p = project("clinical_longitudinal", "progressed")
    p.set_grain(G.PEOPLE_REPEAT, "subject_id")
    p.set_repeat_kind(R.TIME_POINTS)
    p.set_unit_of_analysis(R.UNIT_PERSON)
    with pytest.raises(ProjectError, match="nothing to split chronologically"):
        p.set_temporal_prediction(True)


def test_temporal_prediction_records_the_objective_and_says_it_was_not_drawn():
    """**This test used to assert the defect, and that is worth recording.**

    It was `test_temporal_prediction_routes_to_the_chronological_strategy`, its
    docstring said *"temporal prediction routes to the chronological split
    rather than to a random one"*, and it asserted
    `strategy == "chronological_grouped"` and the sentence *"at times after the
    ones it trained on"* — **verbatim, and green, for as long as `GUIDED-143`
    existed.**

    So the false claim was held in place by two mechanisms, not one. The L41
    report found the first: `recorded_kinds()` read two of the dispatcher's
    three forms, so this kind was outside the probe's denominator. The second is
    here — a passing test whose *name* asserted the routing and whose
    assertions pinned the sentence. Nothing routed anywhere; the test checked
    that a string composer returned a string, and its name supplied the claim.

    **Not the guard-testing-its-own-description class** (trap #2), where the
    assertion is a sentence about the code. Here the assertion was about a real
    value and the value was wrong, and the name asserted a consequence the
    assertion never checked. A test can pin a defect by naming it correctly.
    """
    p = project("clinical_longitudinal", "progressed")
    p.set_grain(G.PEOPLE_REPEAT, "subject_id")
    p.set_repeat_kind(R.TIME_POINTS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_temporal_prediction(True)
    assert p.temporal_prediction["strategy"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert p.temporal_prediction["honored"] is False
    assert "not drawn that way" in p.temporal_prediction["sentence"]
    assert "at times after the ones it trained on" not in \
        p.temporal_prediction["sentence"]

    q = project("clinical_longitudinal", "progressed")
    q.set_grain(G.PEOPLE_REPEAT, "subject_id")
    q.set_repeat_kind(R.TIME_POINTS)
    q.set_unit_of_analysis(R.UNIT_RECORD)
    q.set_temporal_prediction(False)
    assert q.temporal_prediction["strategy"] == R.GROUPED
    assert q.temporal_prediction["honored"] is True


# ── the clinical fixture, end to end, the way a driver takes it ──────────────

def test_clinical_longitudinal_reaches_a_sealed_project_through_all_four():
    p = project("clinical_longitudinal", "progressed")
    p.set_lens(["clinical"])
    p.set_grain(G.PEOPLE_REPEAT, "subject_id")

    reading = R.read(p.df, "subject_id")
    assert reading["reading"] == R.TIME_POINTS
    p.set_repeat_kind(reading["reading"])
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_temporal_prediction(True)
    p.set_eligibility(E.EVERYONE)

    drawn = engine.draw_holdout(p.df, "progressed", "classification", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    assert len(p.df) == 600, "the rows were NOT combined; the trajectory stands"
    assert p.lockbox["seal_basis"] == "grouped"
    kinds = [d.kind for d in p.decisions]
    assert kinds.index("set_repeat_kind") < kinds.index("set_unit_of_analysis")
    assert kinds.index("set_unit_of_analysis") < kinds.index("seal_lockbox")
    assert "set_aggregation" not in kinds
