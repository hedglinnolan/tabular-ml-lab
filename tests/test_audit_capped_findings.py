"""The findings the audit's agent cap left unverified.

Six of the seven I could check empirically were REAL — the cap was hiding
defects, not noise, which is why they were returned labeled rather than
dropped.

  * Switching cohorts left the previous run's filtered_data in place, so
    apply_cohort found no labels in it, fell through to the column path, and
    returned an EMPTY frame with no broken flag — because the fallback had
    "worked".
  * record_run banked every row as a training row, including those with no
    outcome, disagreeing with the "To train on" count the chooser had shown.
  * MIN_PER_CLASS says "in train AND test" and was counted pooled across the
    lockbox boundary, so a cohort passed with one event in the slice every
    reported metric is computed on.
  * execute_stack overwrote a __source_file column a re-uploaded frame already
    carried — destroying provenance on the one column whose entire job is
    provenance.
  * A right-hand column merely NAMED like the left key made the key column
    vanish from the merged result entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.join_doctor import execute_join
from utils.cohorts import (MIN_PER_CLASS, active_cohort, apply_cohort,
                           cohort_filter_broken, plan_cohorts, record_run,
                           start_cohort)
from utils.combine import SOURCE_COLUMN, execute_stack
from utils.session_state import DataConfig, get_data, set_data
from utils.test_lockbox import ensure_lockbox, train_row_mask


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def study(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                         "age": rng.integers(20, 80, n),
                         "y": rng.integers(0, 2, n)})


class TestASwitchNeverYieldsSilentZeroRows:

    def test_a_frame_from_another_run_is_flagged_not_returned_empty(self):
        df = study()
        set_data(df)
        plan = plan_cohorts(df, "sex", "y", "classification")
        start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
        female_frame = get_data()
        start_cohort(df, plan, next(c for c in plan.viable if c.label == "Male"), "y")
        out = apply_cohort(female_frame)
        assert len(out) == 0 and cohort_filter_broken(), (
            "an empty result with no flag is the silent-zero-rows case")

    def test_the_ui_switch_drops_the_previous_runs_row_filter(self):
        import inspect
        import utils.cohort_ui as cu
        for fn in (cu._switch_to, cu._advance_to):
            assert 'pop("filtered_data"' in inspect.getsource(fn), (
                f"{fn.__name__} leaves the previous cohort's rows in place")


class TestBankedCountsMatchWhatTheChooserShowed:

    def test_rows_with_no_outcome_are_not_banked_as_training_rows(self):
        d = study(400)
        d.loc[d.index[:120], "y"] = np.nan
        set_data(d)
        st.session_state["data_config"] = DataConfig(
            target_col="y", feature_cols=["age"], task_type="classification")
        ensure_lockbox(d, "y", "classification")
        plan = plan_cohorts(d, "sex", "y", "classification",
                            train_mask=train_row_mask(d.index))
        cell = next(c for c in plan.viable if c.label == "Female")
        start_cohort(d, plan, cell, "y")
        entry = record_run({"AUC": 0.7})
        assert entry.n_train == cell.n_train, (
            f"banked {entry.n_train}, chooser showed {cell.n_train}")
        assert entry.n_test == cell.n_test


class TestViabilityIsJudgedOnTheSliceThatIsReported:

    def test_a_cohort_whose_holdout_has_one_event_is_refused(self):
        n = 400
        y = np.zeros(n, int)
        y[:MIN_PER_CLASS] = 1                      # all events in one group
        d = pd.DataFrame({"sex": ["Female"] * 150 + ["Male"] * 250,
                          "age": np.arange(n) % 60 + 20, "y": y})
        set_data(d)
        ensure_lockbox(d, "y", "classification")
        plan = plan_cohorts(d, "sex", "y", "classification",
                            train_mask=train_row_mask(d.index))
        female = next(c for c in plan.cells if c.label == "Female")
        assert not female.viable, (
            "a score from its held-out slice would be computed on ~1 event")

    def test_a_healthy_cohort_still_passes(self):
        d = study(800, seed=2)
        set_data(d)
        ensure_lockbox(d, "y", "classification")
        plan = plan_cohorts(d, "sex", "y", "classification",
                            train_mask=train_row_mask(d.index))
        assert len(plan.viable) == 2 and plan.can_proceed


class TestStackingPreservesProvenanceItWasGiven:

    def test_an_existing_source_column_is_kept_under_a_new_name(self):
        a = pd.DataFrame({"x": [1, 2], SOURCE_COLUMN: ["cycle_1999", "cycle_1999"]})
        b = pd.DataFrame({"x": [3, 4], SOURCE_COLUMN: ["cycle_2001", "cycle_2001"]})
        got, _ = execute_stack({"combined_a": a, "combined_b": b})
        assert f"{SOURCE_COLUMN}_before" in got.columns
        assert set(got[f"{SOURCE_COLUMN}_before"]) == {"cycle_1999", "cycle_2001"}
        assert set(got[SOURCE_COLUMN]) == {"combined_a", "combined_b"}

    def test_a_frame_without_one_is_unaffected(self):
        a = pd.DataFrame({"x": [1, 2]})
        b = pd.DataFrame({"x": [3, 4]})
        got, _ = execute_stack({"f1": a, "f2": b})
        assert f"{SOURCE_COLUMN}_before" not in got.columns


class TestTheKeyColumnSurvivesTheJoin:

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_a_right_column_named_like_the_left_key(self, how):
        """demographics keyed on SEQN; labs keyed on pid but carrying SEQN notes."""
        left = pd.DataFrame({"SEQN": [1, 2, 3], "age": [40, 50, 60]})
        right = pd.DataFrame({"pid": [2, 3, 9],
                              "SEQN": ["note_x", "note_y", "note_z"],
                              "glu": [1.0, 2.0, 3.0]})
        merged, _ = execute_join(left, right, "SEQN", "pid", how, "demo", "labs")
        assert "SEQN" in merged.columns, (
            "the column the researcher joined on vanished from their data")
        assert any(c.startswith("SEQN_") for c in merged.columns), (
            "the right file's own SEQN should be kept, suffixed")


class TestWhatActuallySurvivesACohortSwitch:
    """Pressing "Now run the same analysis on Male" must switch the PEOPLE.

    The concern this pins down: does the switch deliver the Male cohort's rows,
    or does it deliver nothing and merely announce it? It delivers the rows.
    What it does NOT carry is anything that was FITTED on the previous group,
    and the button's text now says so instead of promising the predictors are
    unchanged.
    """

    def _work_in_female(self):
        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                           "age": rng.integers(20, 80, n),
                           "bmi": np.round(rng.normal(28, 5, n), 1),
                           "y": rng.integers(0, 2, n)})
        set_data(df)
        st.session_state["data_config"] = DataConfig(
            target_col="y", feature_cols=["age", "bmi"], task_type="classification")
        ensure_lockbox(df, "y", "classification")
        plan = plan_cohorts(df, "sex", "y", "classification",
                            train_mask=train_row_mask(df.index))
        start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
        st.session_state["df_engineered"] = get_data().assign(bmi_sq=lambda d: d["bmi"] ** 2)
        st.session_state["pre_fe_feature_cols"] = ["age", "bmi"]
        st.session_state["selected_features"] = ["age", "bmi", "bmi_sq"]
        st.session_state["preprocessing_config_by_model"] = {"logreg": {"numeric_scaling": "standard"}}
        st.session_state["preprocess_built_model_keys"] = ["logreg"]
        st.session_state["train_model_logreg"] = True
        st.session_state["selected_model_params"] = {"logreg": {"C": 0.5}}
        st.session_state["trained_models"] = {"logreg": "FITTED-ON-FEMALE"}
        st.session_state["filtered_data"] = get_data()
        return df

    def _press_the_button(self):
        import utils.cohort_ui as cu
        try:
            cu._advance_to("sex", "Male")
        except Exception:
            pass                      # st.rerun raises in bare mode

    def test_the_rows_become_the_male_cohort(self):
        df = self._work_in_female()
        assert set(get_data()["sex"]) == {"Female"}
        self._press_the_button()
        out = get_data()
        assert len(out) > 0, "the switch delivered nothing"
        assert set(out["sex"]) == {"Male"}
        assert len(out) == int((df["sex"] == "Male").sum())

    def test_the_two_runs_partition_one_sealed_set(self):
        self._work_in_female()
        sealed = set(st.session_state["test_lockbox"]["labels"])
        female = set(active_cohort()["labels"]) & sealed
        self._press_the_button()
        male = set(active_cohort()["labels"]) & sealed
        assert not (female & male) and (female | male) == sealed

    def test_the_question_carries_over(self):
        self._work_in_female()
        self._press_the_button()
        assert st.session_state["data_config"].target_col == "y"
        assert st.session_state.get("train_model_logreg") is True
        assert st.session_state["selected_model_params"] == {"logreg": {"C": 0.5}}

    def test_nothing_fitted_on_the_previous_group_carries_over(self):
        self._work_in_female()
        self._press_the_button()
        assert not st.session_state.get("trained_models")
        assert not st.session_state.get("preprocessing_config_by_model")
        assert not st.session_state.get("preprocess_built_model_keys"), (
            "a model would be badged 'Tuned for this model' with nothing behind it")
        assert st.session_state.get("df_engineered") is None

    def test_the_button_does_not_promise_predictors_are_unchanged(self):
        import inspect
        import utils.cohort_ui as cu
        src = inspect.getsource(cu.render_next_cohort)
        assert "the predictors and the" not in src, (
            "the text claimed predictors stay while engineering is cleared")
        assert "rebuild" in src.lower()
