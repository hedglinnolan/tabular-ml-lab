"""The cohort run as the app actually experiences it: as a filter on state.

utils/cohorts.py can be perfectly correct and still change nothing, because
what makes a run real is that every page below sees different rows. These tests
are about that seam — get_data(), the lockbox ordering, and the ways a filter
can silently stop applying.

The failure this file exists to prevent: a page headed "women" that quietly
trained on everyone. Wherever the filter cannot be honored, the app must show
NOTHING rather than everything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import (
    active_cohort, apply_cohort, clear_cohort, cohort_candidates,
    cohort_filter_broken, completed_runs, plan_cohorts, record_run,
    runs_remaining, start_cohort,
)
from utils.session_state import get_data, set_data


@pytest.fixture(autouse=True)
def clean_state():
    for key in ("cohort_run", "cohort_runs_done", "_cohort_filter_broken",
                "raw_data", "filtered_data", "df_engineered", "test_lockbox",
                "_raw_data_fingerprint", "data_config", "exploratory_mode"):
        st.session_state.pop(key, None)
    yield
    for key in ("cohort_run", "cohort_runs_done", "_cohort_filter_broken",
                "raw_data", "filtered_data", "df_engineered", "test_lockbox",
                "_raw_data_fingerprint", "data_config", "exploratory_mode"):
        st.session_state.pop(key, None)


def study(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "seqn": np.arange(1, n + 1),
        "sex": rng.choice(["Male", "Female"], n),
        "smoker": rng.choice(["never", "former", "current"], n),
        "age": rng.integers(20, 80, n),
        "bmi": rng.normal(28, 5, n),
        "diabetes": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


def begin(df, column="sex", label="Female", target="diabetes"):
    plan = plan_cohorts(df, column, target, "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return start_cohort(df, plan, cell, target)


# ── the filter actually reaches the pages ────────────────────────────────

class TestGetDataHonorsTheRun:

    def test_no_cohort_means_everyone(self):
        df = study()
        st.session_state["raw_data"] = df
        assert len(get_data()) == len(df)

    def test_active_cohort_is_all_any_page_sees(self):
        df = study()
        st.session_state["raw_data"] = df
        run = begin(df)
        seen = get_data()
        assert len(seen) == run["n_rows"] < len(df)
        assert set(seen["sex"].unique()) == {"Female"}

    def test_full_study_escape_hatch_still_sees_everyone(self):
        df = study()
        st.session_state["raw_data"] = df
        begin(df)
        assert len(get_data(full_study=True)) == len(df)

    def test_engineered_frame_is_filtered_too(self):
        df = study()
        st.session_state["raw_data"] = df
        run = begin(df)
        eng = df.assign(bmi_sq=df["bmi"] ** 2)
        st.session_state["df_engineered"] = eng
        assert len(get_data()) == run["n_rows"]

    def test_applying_twice_changes_nothing(self):
        df = study()
        st.session_state["raw_data"] = df
        begin(df)
        once = apply_cohort(df)
        assert len(apply_cohort(once)) == len(once)


# ── the ways a filter silently stops applying ────────────────────────────

class TestTheFilterCannotBeLostQuietly:

    def test_survives_the_grouping_column_being_engineered_away(self):
        """One-hot encoding `sex` must not un-filter the run.

        A column-based filter would find nothing to match and fall through to
        everyone — a run labeled Female reporting the whole study.
        """
        df = study()
        st.session_state["raw_data"] = df
        run = begin(df)
        eng = df.drop(columns=["sex"]).assign(sex_male=(df["sex"] == "Male").astype(int))
        st.session_state["df_engineered"] = eng
        assert len(get_data()) == run["n_rows"]

    def test_survives_rows_being_dropped_downstream(self):
        df = study()
        st.session_state["raw_data"] = df
        begin(df)
        st.session_state["filtered_data"] = df[df["age"] >= 40]
        seen = get_data()
        assert set(seen["sex"].unique()) == {"Female"}
        assert (seen["age"] >= 40).all()

    def test_unrecognizable_rows_yield_nothing_not_everything(self):
        """The one case where being wrong would be publishable."""
        df = study()
        st.session_state["raw_data"] = df
        begin(df)
        stranger = df.drop(columns=["sex"]).copy()
        stranger.index = range(10_000, 10_000 + len(stranger))
        assert len(apply_cohort(stranger)) == 0
        assert cohort_filter_broken() is True

    def test_new_data_clears_the_run(self):
        df = study()
        set_data(df)
        begin(df)
        assert active_cohort() is not None
        set_data(study(n=250, seed=7))
        assert active_cohort() is None
        assert len(get_data()) == 250

    def test_re_setting_the_same_data_keeps_the_run(self):
        """Page 01 re-sets its working table on every visit.

        Clearing the cohort there ended the run the moment the researcher looked
        at the upload page — while the sidebar still said "Run 1 of 2, Female",
        because the sidebar had already rendered from the state being wiped
        further down the same page.
        """
        df = study()
        set_data(df)
        run = begin(df)
        for _ in range(3):
            set_data(df.copy())         # what a page revisit does
            assert active_cohort() is not None
            assert len(get_data()) == run["n_rows"]

    def test_corrected_data_with_the_same_columns_still_clears_it(self):
        df = study()
        set_data(df)
        begin(df)
        corrected = df.copy()
        corrected.loc[corrected.index[:20], "bmi"] = 99.9
        set_data(corrected)
        assert active_cohort() is None


# ── the lockbox is drawn before the filter, not after ────────────────────

class TestOneSplitSharedByEveryRun:

    def test_each_cohort_inherits_its_slice_of_one_split(self):
        from utils.test_lockbox import ensure_lockbox, train_row_mask
        df = study()
        st.session_state["raw_data"] = df
        lb = ensure_lockbox(df, "diabetes", "classification")
        sealed = set(lb["labels"])

        female = begin(df, label="Female")
        male_plan = plan_cohorts(df, "sex", "diabetes", "classification")
        male_cell = next(c for c in male_plan.viable if c.label == "Male")

        f_test = set(female["labels"]) & sealed
        st.session_state.pop("cohort_run")
        male = start_cohort(df, male_plan, male_cell, "diabetes")
        m_test = set(male["labels"]) & sealed

        # No one is in two places, and between them the two runs account for
        # the whole sealed set — which only holds if it was drawn once, first.
        assert not (f_test & m_test)
        assert f_test | m_test == sealed

    def test_status_chip_reports_this_runs_share_not_the_studys(self):
        """"n=135" beside a 490-row run is a number a researcher writes down."""
        from utils.test_lockbox import ensure_lockbox, render_lockbox_status
        df = study()
        st.session_state["raw_data"] = df
        lb = ensure_lockbox(df, "diabetes", "classification")
        run = begin(df)
        n_here = len(set(lb["labels"]) & set(run["labels"]))
        assert 0 < n_here < lb["n_test"]
        render_lockbox_status()          # must not raise in either mode
        clear_cohort()
        render_lockbox_status()

    def test_train_mask_still_reads_the_study_wide_lockbox(self):
        from utils.test_lockbox import ensure_lockbox, train_row_mask
        df = study()
        st.session_state["raw_data"] = df
        ensure_lockbox(df, "diabetes", "classification")
        begin(df)
        cohort = get_data()
        mask = train_row_mask(cohort.index)
        assert mask.index.equals(cohort.index)
        assert 0 < int(mask.sum()) < len(cohort)


# ── the registry that drives "now run the men" ───────────────────────────

class TestRunRegistry:

    def test_position_of_the_run_is_recorded(self):
        df = study()
        run = begin(df, label="Female")
        assert run["of"] == 2
        assert run["position"] in (1, 2)
        assert set(run["order"]) == {"Female", "Male"}

    def test_recording_counts_train_and_test_from_the_lockbox(self):
        from utils.test_lockbox import ensure_lockbox
        df = study()
        st.session_state["raw_data"] = df
        ensure_lockbox(df, "diabetes", "classification")
        run = begin(df)
        entry = record_run({"ROC-AUC": 0.71})
        assert entry.n_train + entry.n_test == run["n_rows"]
        assert entry.n_test > 0
        assert entry.metrics["ROC-AUC"] == 0.71

    def test_re_recording_the_same_cohort_replaces_it(self):
        df = study()
        st.session_state["raw_data"] = df
        begin(df)
        record_run({"ROC-AUC": 0.61})
        record_run({"ROC-AUC": 0.66})
        assert len(completed_runs()) == 1
        assert completed_runs()[0].metrics["ROC-AUC"] == 0.66

    def test_remaining_drives_the_next_button(self):
        df = study()
        st.session_state["raw_data"] = df
        begin(df, label="Female")
        record_run({})
        plan = plan_cohorts(df, "sex", "diabetes", "classification")
        left = runs_remaining(plan, [r.label for r in completed_runs()])
        assert [c.label for c in left] == ["Male"]

    def test_no_run_active_records_nothing(self):
        assert record_run({"ROC-AUC": 0.9}) is None
        assert completed_runs() == []


# ── what gets offered as a grouping variable ─────────────────────────────

class TestCandidates:

    def test_offers_the_columns_that_describe_people(self):
        got = cohort_candidates(study(), "diabetes")
        assert "sex" in got and "smoker" in got

    def test_never_offers_the_outcome(self):
        assert "diabetes" not in cohort_candidates(study(), "diabetes")

    def test_never_offers_an_identifier(self):
        assert "seqn" not in cohort_candidates(study(), "diabetes")

    def test_never_offers_a_measurement(self):
        got = cohort_candidates(study(), "diabetes")
        assert "bmi" not in got and "age" not in got

    def test_never_offers_the_apps_own_bookkeeping(self):
        df = study()
        df["__source_file"] = np.where(df.index % 2 == 0, "a.csv", "b.csv")
        assert "__source_file" not in cohort_candidates(df, "diabetes")

    def test_survives_duplicate_column_labels(self):
        df = study()
        df = pd.concat([df, df[["sex"]]], axis=1)
        cohort_candidates(df, "diabetes")   # must not raise


# ── the chooser as page 01 calls it ──────────────────────────────────────

class TestChooserRenders:
    """These were three smoke tests that asserted nothing.

    The audit's mutation run confirmed the cost: stubbing render_cohort_chooser
    to a no-op left them green, because in bare mode Streamlit widget calls are
    largely no-ops and "it did not raise" is nearly free. They now assert on the
    DECISIONS the chooser makes, which is the part that can be wrong.
    """

    def test_the_chooser_offers_the_groups_the_plan_found_viable(self):
        from utils.cohort_ui import _cell_table, _n_for
        df = study()
        st.session_state["raw_data"] = df
        plan = plan_cohorts(df, "sex", "diabetes", "classification")
        table = _cell_table(plan, "diabetes")
        assert list(table["Group"]) == [c.label for c in plan.cells]
        assert all(_n_for(plan, c.label) == c.n_rows for c in plan.cells)

    def test_the_table_marks_a_group_that_cannot_be_modeled(self):
        from utils.cohort_ui import _cell_table
        df = study(n=400)
        df.loc[df.index[:395], "sex"] = "Male"      # 5 Females: far too few
        plan = plan_cohorts(df, "sex", "diabetes", "classification")
        table = _cell_table(plan, "diabetes")
        row = table[table["Group"] == "Female"].iloc[0]
        assert row["Can be analyzed on its own"].startswith("no —")

    def test_the_rarer_outcome_names_the_target_and_its_value(self):
        from utils.cohort_ui import _rarer_outcome
        df = study()
        plan = plan_cohorts(df, "sex", "diabetes", "classification")
        cell = plan.cells[0]
        text = _rarer_outcome(cell, "diabetes")
        assert "diabetes =" in text, (
            "a bare count cannot say whether it is cases or non-cases")
        assert str(cell.n_events) in text.replace(",", "")

    def test_the_chip_states_the_run_and_both_denominators(self):
        from utils.cohort_ui import render_cohort_chip
        df = study()
        st.session_state["raw_data"] = df
        run = begin(df)
        html = []
        real = st.sidebar.markdown
        st.sidebar.markdown = lambda body, **kw: html.append(str(body))
        try:
            render_cohort_chip()
        finally:
            st.sidebar.markdown = real
        blob = " ".join(html)
        assert "sex" in blob and "Female" in blob
        assert f"{run['n_rows']:,}" in blob and f"{run['n_total']:,}" in blob

    def test_the_chip_says_nothing_when_no_run_is_active(self):
        from utils.cohort_ui import render_cohort_chip
        html = []
        real = st.sidebar.markdown
        st.sidebar.markdown = lambda body, **kw: html.append(str(body))
        try:
            render_cohort_chip()
        finally:
            st.sidebar.markdown = real
        assert not html
