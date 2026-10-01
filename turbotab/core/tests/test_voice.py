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
                    "replaced by the residual plus the nutrient's mean.")


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
