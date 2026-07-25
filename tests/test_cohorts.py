"""Cohort runs: same question, different people.

The distinction these tests defend, because conflating it is the whole
problem:

    "Does my model work equally well for men and women?"
        -> one model, evaluated within groups. ml.publication.subgroup_analysis.
    "Is the relationship DIFFERENT in men and women?"
        -> fitted separately in each. That is a cohort run, and it is what
           researchers actually ask for.

A run is deliberately narrow — target and features fixed, only rows change —
so runs are comparable by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import (
    MIN_PER_CLASS, MIN_ROWS_PER_COHORT, CohortRun, cohort_mask,
    comparison_caveats, features_that_lose_variance, plan_cohorts, runs_remaining,
)
from utils.test_lockbox import ensure_lockbox

RNG = np.random.RandomState(0)


def _cohort(n=1000):
    return pd.DataFrame({
        "SEQN": range(n),
        "sex": RNG.choice(["M", "F"], n, p=[0.7, 0.3]),
        "diabetes": RNG.choice([0, 1], n, p=[0.8, 0.2]),
        "age": RNG.randint(18, 80, n),
        "outcome": RNG.choice([0, 1], n, p=[0.8, 0.2]),
    })


# ── the held-out set must look like the study ────────────────────────────

class TestLockboxIsRepresentative:
    """A test set that is 75% men when the cohort is 70% men reports a number
    that does not describe the study population. Stratifying on the outcome
    alone was not enough."""

    def _test_slice(self, df, strata):
        st.session_state.clear()
        lb = ensure_lockbox(df, "outcome", "classification",
                            fraction=0.15, seed=42, stratify_cols=strata)
        return df.loc[lb["labels"]], lb

    def test_demographics_are_balanced_when_requested(self):
        df = _cohort()
        loose, _ = self._test_slice(df, None)
        tight, _ = self._test_slice(df, ["sex"])
        truth = float((df["sex"] == "M").mean())
        assert abs(float((tight["sex"] == "M").mean()) - truth) <= \
            abs(float((loose["sex"] == "M").mean()) - truth)

    def test_the_strata_actually_used_are_recorded(self):
        _, lb = self._test_slice(_cohort(), ["sex", "diabetes"])
        assert lb["strata"] == ["outcome", "sex", "diabetes"]

    def test_the_outcome_is_always_balanced_for_classification(self):
        _, lb = self._test_slice(_cohort(), None)
        assert "outcome" in lb["strata"]

    def test_impossible_strata_degrade_instead_of_failing(self):
        # A near-unique column cannot be stratified on; the split must still
        # happen, and must report what it managed rather than claiming success.
        df = _cohort()
        df["almost_unique"] = range(len(df))
        _, lb = self._test_slice(df, ["almost_unique"])
        assert lb is not None and len(lb["labels"]) > 0
        assert "almost_unique" not in lb["strata"]

    def test_changing_the_strata_rebuilds_the_lockbox(self):
        df = _cohort()
        a, lb_a = self._test_slice(df, None)
        b, lb_b = self._test_slice(df, ["sex"])
        assert lb_a["signature"] != lb_b["signature"]

    def test_an_unknown_column_is_ignored_not_fatal(self):
        _, lb = self._test_slice(_cohort(), ["not_a_column"])
        assert lb is not None and "not_a_column" not in lb["strata"]


# ── planning the cohorts ─────────────────────────────────────────────────

class TestCohortPlanning:

    def _plan(self, df, column="sex"):
        st.session_state.clear()
        lb = ensure_lockbox(df, "outcome", "classification", fraction=0.15, seed=42)
        train_mask = pd.Series(~df.index.isin(lb["labels"]), index=df.index)
        return plan_cohorts(df, column, "outcome", "classification", train_mask=train_mask)

    def test_a_normal_split_is_offered(self):
        plan = self._plan(_cohort())
        assert plan.can_proceed
        assert {c.label for c in plan.viable} == {"M", "F"}

    def test_train_and_test_counts_are_reported_per_cohort(self):
        plan = self._plan(_cohort())
        for cell in plan.viable:
            assert cell.n_train > 0 and cell.n_test > 0
            assert cell.n_train + cell.n_test == cell.n_rows

    def test_a_level_too_small_is_refused_not_warned(self):
        df = _cohort()
        df["site"] = ["main"] * 970 + ["satellite"] * 30
        plan = self._plan(df, "site")
        blocked = {c.label for c in plan.blocked}
        assert "satellite" in blocked
        assert not plan.can_proceed

    def test_the_refusal_says_why(self):
        df = _cohort()
        df["site"] = ["main"] * 970 + ["satellite"] * 30
        plan = self._plan(df, "site")
        reason = [c.blocked_reason for c in plan.blocked][0]
        assert str(MIN_ROWS_PER_COHORT) in reason or str(MIN_PER_CLASS) in reason

    def test_too_few_events_blocks_even_when_rows_are_plentiful(self):
        n = 400
        df = pd.DataFrame({"sex": ["M"] * 200 + ["F"] * 200,
                           "outcome": [0] * 197 + [1] * 3 + [0] * 150 + [1] * 50})
        plan = plan_cohorts(df, "sex", "outcome", "classification")
        assert not [c for c in plan.viable if c.label == "M"]

    def test_splitting_by_the_outcome_is_refused(self):
        plan = plan_cohorts(_cohort(), "outcome", "outcome", "classification")
        assert plan.blocking and "predicting" in plan.blocking[0]

    def test_a_constant_column_is_refused(self):
        df = _cohort()
        df["site"] = "one"
        assert plan_cohorts(df, "site", "outcome", "classification").blocking

    def test_a_high_cardinality_column_is_refused(self):
        df = _cohort()
        df["zip"] = RNG.randint(10000, 99999, len(df))
        plan = plan_cohorts(df, "zip", "outcome", "classification")
        assert plan.blocking and "handful" in plan.blocking[0]

    def test_a_missing_column_is_refused(self):
        assert plan_cohorts(_cohort(), "nope", "outcome").blocking


class TestCohortWarnings:
    """What splitting costs, said before it happens."""

    def test_lopsided_cohorts_are_flagged(self):
        df = pd.DataFrame({"g": ["a"] * 900 + ["b"] * 100,
                           "outcome": list(RNG.choice([0, 1], 1000, p=[0.7, 0.3]))})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        assert any("very different sizes" in w for w in plan.warnings)

    def test_differing_outcome_rates_are_flagged(self):
        df = pd.DataFrame({"g": ["a"] * 500 + ["b"] * 500,
                           "outcome": [0] * 450 + [1] * 50 + [0] * 250 + [1] * 250})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        assert any("not directly comparable" in w for w in plan.warnings)

    def test_many_cohorts_raise_multiplicity(self):
        n = 1400
        df = pd.DataFrame({"g": [f"s{i % 7}" for i in range(n)],
                           "outcome": list(RNG.choice([0, 1], n, p=[0.6, 0.4]))})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        assert any("multiple-comparisons" in w for w in plan.warnings)

    def test_a_clean_two_way_split_is_not_nagged(self):
        df = pd.DataFrame({"g": ["a"] * 500 + ["b"] * 500,
                           "outcome": list(RNG.choice([0, 1], 1000, p=[0.7, 0.3]))})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        assert plan.warnings == []


class TestFeaturesLosingVariance:
    """Filtering to men makes `sex` constant. Handing that to a model is noise
    at best, and a feature list that silently changed between runs breaks the
    promise that both runs answered the same question."""

    def test_the_grouping_column_itself_is_caught(self):
        df = _cohort()
        lost = features_that_lose_variance(df, cohort_mask(df, "sex", "M"),
                                           ["sex", "age", "diabetes"])
        assert [c for c, _ in lost] == ["sex"]

    def test_the_reason_is_stated(self):
        df = _cohort()
        lost = features_that_lose_variance(df, cohort_mask(df, "sex", "M"), ["sex"])
        assert "always" in lost[0][1]

    def test_near_constant_is_caught_too(self):
        # 99 of 100 rows in cohort "a" share a value: not constant, still useless.
        df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100,
                           "rare": [0] * 99 + [1] + list(RNG.randint(0, 5, 100))})
        lost = features_that_lose_variance(df, cohort_mask(df, "g", "a"), ["rare"])
        assert lost and "same value" in lost[0][1]

    def test_values_read_as_a_person_would_write_them(self):
        df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100, "flag": [0] * 200})
        lost = features_that_lose_variance(df, cohort_mask(df, "g", "a"), ["flag"])
        assert lost[0][1] == "is always 0 in this cohort"

    def test_a_varying_feature_is_left_alone(self):
        df = _cohort()
        lost = features_that_lose_variance(df, cohort_mask(df, "sex", "M"), ["age"])
        assert lost == []


class TestRunSequencing:

    def test_remaining_runs_drive_the_next_button(self):
        df = _cohort()
        plan = plan_cohorts(df, "sex", "outcome", "classification")
        assert {c.label for c in runs_remaining(plan, [])} == {"M", "F"}
        assert [c.label for c in runs_remaining(plan, ["F"])] == ["M"]
        assert runs_remaining(plan, ["F", "M"]) == []

    def test_caveats_need_two_completed_runs(self):
        assert comparison_caveats([CohortRun("sex", "M", 600, 100, completed=True)],
                                  "classification") == []

    def test_unequal_training_sizes_are_disclosed(self):
        runs = [CohortRun("sex", "M", 600, 100, completed=True),
                CohortRun("sex", "F", 150, 30, completed=True)]
        assert any("handicapped" in c for c in comparison_caveats(runs, "classification"))

    def test_multiplicity_is_counted_for_the_manuscript(self):
        runs = [CohortRun("sex", "M", 600, 100, completed=True),
                CohortRun("sex", "F", 500, 90, completed=True)]
        assert any("Report all 2" in c for c in comparison_caveats(runs, "classification"))

    def test_the_interaction_alternative_is_always_offered(self):
        runs = [CohortRun("sex", "M", 600, 100, completed=True),
                CohortRun("sex", "F", 500, 90, completed=True)]
        assert any("interaction term" in c for c in comparison_caveats(runs, "classification"))


# ── defects found by re-auditing this feature after it was written ───────

class TestAuditFindings:
    """Five problems the first pass shipped with. Each of them produced a
    plausible-looking result, which is why none of them failed a test."""

    def test_stratification_survives_a_grouped_split_falling_back(self):
        """A group column with too few subjects fell back to an ordinary split
        carrying NO stratification, because stratification was decided before
        the fallback. The one case producing a test set unrepresentative of
        both the outcome AND the demographics — silently."""
        rng = np.random.RandomState(0)
        n = 300
        df = pd.DataFrame({"subj": np.repeat(range(3), 100),   # below the group floor
                           "sex": rng.choice(["M", "F"], n, p=[0.7, 0.3]),
                           "outcome": rng.choice([0, 1], n, p=[0.8, 0.2])})
        st.session_state.clear()
        lb = ensure_lockbox(df, "outcome", "classification", fraction=0.2, seed=1,
                            group_col="subj", stratify_cols=["sex"])
        assert lb["group_col"] is None            # it did fall back
        assert lb["strata"] == ["outcome", "sex"]  # and still stratified
        test = df.loc[lb["labels"]]
        assert abs(test["outcome"].mean() - df["outcome"].mean()) < 0.05

    def test_a_genuinely_grouped_split_still_groups(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"subj": np.repeat(range(50), 6),
                           "sex": rng.choice(["M", "F"], 300),
                           "outcome": rng.choice([0, 1], 300, p=[0.7, 0.3])})
        st.session_state.clear()
        lb = ensure_lockbox(df, "outcome", "classification", fraction=0.2, seed=1,
                            group_col="subj", stratify_cols=["sex"])
        assert lb["group_col"] == "subj"
        test_subjects = set(df.loc[lb["labels"], "subj"])
        train_subjects = set(df.loc[~df.index.isin(lb["labels"]), "subj"])
        assert not (test_subjects & train_subjects)

    def test_rows_with_no_grouping_value_are_counted_and_disclosed(self):
        """They belong to no cohort and vanish from every run. Saying nothing
        is exactly the silent exclusion this app exists to prevent."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"sex": rng.choice(["M", "F", None], 1000, p=[0.6, 0.25, 0.15]),
                           "outcome": rng.choice([0, 1], 1000, p=[0.8, 0.2])})
        plan = plan_cohorts(df, "sex", "outcome", "classification")
        assert plan.n_excluded_missing == int(df["sex"].isna().sum())
        assert "no 'sex' recorded" in plan.summary()
        assert any("selected sample" in w for w in plan.warnings)

    def test_a_couple_of_missing_values_are_not_nagged_about(self):
        df = pd.DataFrame({"g": ["a"] * 499 + ["b"] * 499 + [None] * 2,
                           "outcome": list(np.resize([0, 0, 0, 1], 1000))})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        assert not any("selected sample" in w for w in plan.warnings)

    def test_rows_without_an_outcome_do_not_inflate_viability(self):
        """100 rows of which 65 have no outcome is a 35-row cohort wearing a
        big number, and it was passing the size check on the big number."""
        df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100,
                           "outcome": [0] * 20 + [1] * 15 + [None] * 65
                                      + [0] * 60 + [1] * 40})
        plan = plan_cohorts(df, "g", "outcome", "classification")
        a = [c for c in plan.cells if c.label == "a"][0]
        assert a.n_rows == 35 and a.n_rows_total == 100
        assert not a.viable and "with an outcome recorded" in a.blocked_reason

    def test_a_train_mask_from_another_frame_cannot_corrupt_the_counts(self):
        df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100,
                           "outcome": list(np.resize([0, 1], 200))})
        stray = pd.Series(True, index=range(500, 700))     # wrong index entirely
        plan = plan_cohorts(df, "g", "outcome", "classification", train_mask=stray)
        for cell in plan.cells:
            assert cell.n_train + cell.n_test == cell.n_rows
