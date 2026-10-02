"""The app's words: decision sentences say what happened, with the flow's own numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import DecisionLog, ProjectState
from turbotab.core.stages.target import task_reason


# ── finish ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, done", [
    ("**Bold** claim", "Bold claim."),
    ("Ends with a colon:", "Ends with a colon."),
    ("Doubled.. period", "Doubled. period."),
    ("space ,before  comma", "space, before comma."),
    ("Is it?", "Is it?"),
    ("A list;.", "A list."),
    ("(see §02.)", "(see §02.)"),
    ("keeps `a  b` and `**x**` as data", "keeps `a  b` and `**x**` as data."),
    ("Trailing ellipsis...", "Trailing ellipsis."),
    ("One.\n\nTwo", "One.\n\nTwo."),
])
def test_finish_leaves_one_clean_sentence(raw, done):
    assert voice.finish(raw) == done


def test_a_label_ends_without_punctuation():
    assert voice.finish("Adjust for energy.", terminal=False) == "Adjust for energy"


def test_machinery_is_named():
    assert voice.machinery("value None here") == ["None"]
    assert voice.machinery("None of them") == []
    assert voice.machinery("a {n_bins} bins") == ["{placeholder}"]
    assert voice.machinery("[object Object]") == ["[object"]
    assert voice.machinery("**bold**") == ["raw markdown"]
    assert voice.machinery("`None` is a level") == []


# ── exclusions: the flow's numbers ───────────────────────────────────────────

def frame():
    return pd.DataFrame({
        "kcal": [300, 450, 1800, 2200, 5200, 6000, 400, 2500],
        "y": [1.0, np.nan, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0],
    })


def test_an_exclusion_counts_rows_with_the_outcome_measured():
    rule = d.ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intakes")
    text = voice.sentence_for(d.SetExclusions(rules=[rule]), ProjectState(target="y"), {"frame": frame()})
    # rows 0, 4, 6 (row 1 and 5 have no outcome, so the flow never reaches them here)
    assert text == "`3` rows with `kcal` outside `500`–`5000` were excluded as implausible intakes."


def test_rules_count_in_order_each_removing_only_what_the_last_kept():
    first = d.ExclusionRule(column="kcal", low=500, reason="Implausibly low intakes")
    second = d.ExclusionRule(column="kcal", high=2000, reason="a stricter cap")
    text = voice.sentence_for(d.SetExclusions(rules=[first, second]), ProjectState(), {"frame": frame()})
    # 300, 450, 400 go first; of 1800, 2200, 5200, 6000, 2500 the cap takes four
    assert text == ("`3` rows with `kcal` below `500` were excluded as implausibly low intakes; "
                    "`4` rows with `kcal` above `2000` were excluded as a stricter cap, `7` in all.")


def test_an_exclusion_by_sex_names_each_range():
    rule = d.ExclusionRule(column="kcal", reason="implausible intakes (Willett's sex-specific cut-offs)",
                           by=d.RangeByLevel(column="sex", ranges={"F": (500, 3500), "M": (800, 4200)}))
    f = pd.DataFrame({"kcal": [3600, 3600, 700, 700], "sex": ["F", "M", "F", "M"]})
    text = voice.sentence_for(d.SetExclusions(rules=[rule]), ProjectState(), {"frame": f})
    assert text == ("`2` rows with `kcal` outside `500`–`3500` for `sex` `F` and outside "
                    "`800`–`4200` for `M` were excluded as implausible intakes (Willett's "
                    "sex-specific cut-offs).")


def test_no_exclusion_is_a_recorded_answer():
    assert voice.sentence_for(d.SetExclusions(rules=[]), ProjectState(target="glucose")) == \
        "No rows were excluded: every row with `glucose` measured stays in the analysis."


def test_without_data_the_sentence_says_less_and_never_a_placeholder():
    rule = d.ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intakes")
    assert voice.sentence_for(d.SetExclusions(rules=[rule])) == \
        "Rows with `kcal` outside `500`–`5000` were excluded as implausible intakes."


# ── revert and roles: what holds now ─────────────────────────────────────────

def test_a_revert_says_what_the_slot_holds_again(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetEnergyAdjustment(method="standard", energy_column="kcal", nutrients=["fat_total"]))
    second = log.append(d.SetEnergyAdjustment(method="residual", energy_column="kcal", nutrients=["fat_total"]))
    text = voice.sentence_for(d.Revert(decision_id=second.id), log.state(), {"records": log.records()})
    assert text == ("Decision `#2` was reverted, so the energy adjustment is the standard "
                    "(multivariate) model again.")
    target = log.append(d.SetTarget(column="glucose"))
    text = voice.sentence_for(d.Revert(decision_id=target.id), log.state(), {"records": log.records()})
    assert text == "Decision `#3` was reverted, so the outcome is unanswered again."


def test_a_changed_role_is_said_as_a_change():
    before = ProjectState(roles={"bmi": "covariate", "kcal": "energy"})
    text = voice.sentence_for(d.SetRoles(roles={"bmi": "excluded", "kcal": "energy"}), before)
    assert text == "`bmi` became excluded; every other role is unchanged."


def test_the_split_names_its_seed_grouping_and_stratification():
    text = voice.sentence_for(d.SetSplit(holdout=0.2, seed=3, folds=5),
                              ProjectState(target="event", task="binary"),
                              {"n_cohort": 5352, "repeats": {"column": "SEQN"}})
    assert text == ("A random `20%` of the rows with `event` recorded (seed `3`, keeping each `SEQN`'s rows "
                    "together, stratified by `event`) was held out for one final score; models "
                    "were compared by `5`-fold cross-validation on the rest.")


def test_the_residual_sentence_says_where_it_was_fit():
    text = voice.sentence_for(d.SetEnergyAdjustment(method="residual", energy_column="kcal",
                                                    nutrients=["protein", "fat_total"], strata="gender"))
    assert text == ("Energy was adjusted by the residual method: `protein` and `fat_total` were "
                    "each regressed on `kcal` within levels of `gender` on training rows and "
                    "replaced by the residual plus the nutrient's mean over all training rows.")


# ── the task reason ──────────────────────────────────────────────────────────

def detect(series):
    from turbotab import engine

    detection = engine.detect_task_type(series.to_frame("y"), "y")
    k = series.nunique()
    task = ("binary" if k <= 2 else "multiclass") if detection["detected"] == "classification" else "regression"
    return task_reason(series, detection, task)


def test_the_task_reason_is_one_sentence_in_the_app_voice():
    rng = np.random.default_rng(0)
    assert detect(pd.Series(rng.normal(100, 10, 500))) == \
        "Continuous, with 500 distinct values — read as a regression outcome."
    assert detect(pd.Series(["no", "yes"] * 50)) == \
        "Text with two values, `no` and `yes` — read as a binary outcome."
    assert detect(pd.Series([0, 1] * 50)) == "Only `0` and `1` — read as a binary outcome."
    assert detect(pd.Series([0.0, 1.0, np.nan] * 30)) == \
        "Only `0` and `1`, with blanks — read as a binary outcome."
    assert detect(pd.Series([1, 2, 3, 4, 5] * 20)) == (
        "Whole numbers with 5 distinct values; class codes, counts and ordinal scores all look "
        "like this — read as a multiclass outcome.")


# ── M2: the opening sequence, the seal and findings (M2_CONTRACT §6) ────────

RECALLS = pd.DataFrame({
    "participant_id": [1, 1, 2, 2, 3, 3, 4],
    "recall_date": pd.to_datetime(["2020-01-01", "2020-01-05"] * 3 + ["2020-01-01"]),
    "glucose": [99.0, 101.0, 120.0, 118.0, 88.0, 90.0, 95.0],
    "diabetes": ["no", "no", "yes", "yes", "no", "no", "no"],
})
REPEATED = ProjectState(target="glucose", grain={"grain": "repeated", "id_column": "participant_id"},
                        repeat_kind={"repeat_kind": "repeats"})


def say(decision, state=REPEATED, **ctx):
    return voice.sentence_for(decision, state, {"frame": RECALLS, **ctx})


def test_the_grain_sentence_counts_the_units_it_names():
    assert say(d.SetGrain(grain="repeated", id_column="participant_id")) == (
        "Participants were declared to appear in more than one row, identified by "
        "`participant_id`: `7` rows from `4` of them, at most `2` each.")
    assert say(d.SetGrain(grain="one_row_per_unit", id_column="participant_id")) == (
        "Each row was declared a different participant: no `participant_id` appears in more "
        "than one row.")


def test_the_aggregation_sentence_says_how_and_what_it_did_to_n():
    assert say(d.SetAggregation(method="mean")) == (
        "Each `participant_id`'s rows were combined into one by their mean: `7` rows became `4`.")
    timed = REPEATED.model_copy(update={"repeat_kind": d.RepeatSpec(repeat_kind="time_points",
                                                                    time_column="recall_date")})
    assert say(d.SetAggregation(method="last", outcome="last"), timed) == (
        "Each `participant_id`'s rows were combined into one by keeping the last row, in order of "
        "`recall_date`: `7` rows became `4`; the outcome `glucose` was taken as their last value.")


def test_the_event_sentence_says_which_level_became_1():
    """DRIVE_RUBRIC §5.3: a reader can tell which level became 1 from the sentence alone."""
    assert say(d.SetEvent(column="diabetes", level="yes")) == (
        "`yes` of `diabetes` was taken as the event and coded 1; `no` was coded 0.")
    assert say(d.SetEvent(column="flag", level="1"), levels=[0.0, 1.0]) == (
        "`1` of `flag` was taken as the event and coded 1; `0` was coded 0.")


def test_the_orientation_sentence_is_the_methods_sentence():
    assert voice.sentence_for(d.SetOrientation(orientation="feature_major"), None, {"n_rows": 396}) == (
        "The table was supplied with features in rows and samples in columns, and was transposed "
        "to one row per sample before any diagnosis was run; its `396` rows became measurement "
        "columns.")
    assert voice.sentence_for(d.SetOrientation(orientation="sample_major")) == (
        "The table was confirmed as one row per sample, as supplied, and was not transposed.")


def test_the_temporal_and_seal_sentences():
    assert say(d.SetTemporal(temporal=True, time_column="recall_date")) == (
        "The model was declared to predict a later outcome from earlier measurements: the held-out "
        "rows are the latest by `recall_date`, with each `participant_id`'s rows kept together.")
    assert say(d.OpenSeal(), n_holdout=591) == (
        "The `591` held-out rows were opened once and scored; those scores are fixed in the "
        "record, and any later change is marked as made after the seal was opened.")


def test_findings_answered_name_their_columns_and_timing():
    finding = {"id": "pack::survey::sentinel_codes", "title": "5 items carry values outside 1–5",
               "affected_columns": ["item_03", "item_14"]}
    repair = {"label": "Set to missing", "consequence": "Sentinel codes become blanks.",
              "row_local": True}
    assert say(d.ApplyRepair(finding_id=finding["id"], option="to_missing", params={"code": 9}),
               finding=finding, repair=repair) == (
        "The repair “Set to missing” was applied to `item_03` and `item_14` (code `9`): sentinel "
        "codes become blanks; it rewrote the working table before anything was counted.")
    assert say(d.DeferFinding(finding_id=finding["id"], to="missing"), finding=finding) == (
        "The finding on `item_03` and `item_14` was set aside for the missing-values question, "
        "where it will be raised again.")
    assert say(d.DismissFinding(finding_id=finding["id"], reason="These are real 9-point items"),
               finding=finding) == (
        "The finding on `item_03` and `item_14` was dismissed: these are real 9-point items.")


def test_a_repair_family_may_speak_for_itself():
    voice.register_repair_sentence(
        "test::family", lambda dec, state, ctx: "`bp_di` values of `5.4e-79` were set to `0`")
    try:
        assert voice.sentence_for(d.ApplyRepair(finding_id="test::family__bp_di", option="zero")) == \
            "`bp_di` values of `5.4e-79` were set to `0`."
    finally:
        voice._REPAIR_SENTENCES.pop("test::family", None)


def test_reverting_a_disposition_or_a_structural_answer(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetGrain(grain="repeated", id_column="participant_id"))
    grain = log.append(d.SetGrain(grain="one_row_per_unit"))
    text = voice.sentence_for(d.Revert(decision_id=grain.id), log.state(), {"records": log.records()})
    assert text == ("Decision `#2` was reverted, so the grain is repeated rows by `participant_id` "
                    "again.")
    dismissed = log.append(d.DismissFinding(finding_id="voice::identifier__SEQN"))
    text = voice.sentence_for(d.Revert(decision_id=dismissed.id), log.state(),
                              {"records": log.records(), "finding": {"affected_columns": ["SEQN"]}})
    assert text == "Decision `#3` was reverted, so the finding on `SEQN` is open again."


def test_the_outcome_sentence_carries_its_unit():
    assert voice.sentence_for(d.SetTarget(column="glucose"), None, {"frame": RECALLS}) == \
        "`glucose` was chosen as the outcome, in mg/dL."
    assert voice.sentence_for(d.SetTarget(column="diabetes"), None, {"frame": RECALLS}) == \
        "`diabetes` was chosen as the outcome."
