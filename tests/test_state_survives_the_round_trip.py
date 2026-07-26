"""Three ways the app forgot what it had already decided.

  - Changing the OUTCOME during a one-group run trips the same refusal as any
    other change, and the refusal is right — every run shares one split. But
    the sealed rows were drawn among those with a value for the OLD outcome,
    so when the new one was assayed on a sub-sample (ordinary in nutrition and
    omics) most of them cannot be scored at all, while the chip went on
    reporting the sealed count. The audit measured 68 reported against 21
    actually evaluated.

  - Restoring a saved session dropped _working_table_source_id, so the first
    visit to Upload & Audit rebuilt the working table from the ORIGINAL file,
    discarded every cleaning action, and called set_data() with it — which
    cleared the active run and moved the data fingerprint, taking every banked
    run out of the comparison table. Silent, on the app's home page.

  - The replay that makes "run the same analysis on the male cohort" mean the
    same analysis only executed on Feature Engineering, while the button that
    promises it lives on Train & Compare.
"""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import plan_cohorts, start_cohort
from utils.replay import record, stage_for_replay, run_pending_replay, pending
from utils.test_lockbox import ensure_lockbox, get_lockbox


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def study(n=800):
    """glucose_high is complete; hba1c_high was assayed on a sub-sample."""
    rng = np.random.default_rng(3)
    sex = rng.choice(["Female", "Male"], n)
    hba1c = rng.integers(0, 2, n).astype(float)
    # missing far more often in women, as a sub-sampled assay usually is
    drop = (rng.random(n) < np.where(sex == "Female", 0.75, 0.25))
    hba1c[drop] = np.nan
    return pd.DataFrame({"sex": sex, "age": rng.integers(20, 80, n),
                         "bmi": rng.normal(27, 4, n),
                         "glucose_high": rng.integers(0, 2, n),
                         "hba1c_high": hba1c})


class Cfg:
    def __init__(self, target):
        self.target_col = target
        self.feature_cols = ["age", "bmi"]
        self.task_type = "classification"


def test_changing_the_outcome_mid_run_reports_what_can_be_scored():
    df = study()
    st.session_state["raw_data"] = df
    ensure_lockbox(df, "glucose_high", "classification")
    plan = plan_cohorts(df, "sex", "glucose_high", "classification")
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"),
                 "glucose_high")

    # the researcher changes the Target Variable selectbox without leaving page 01
    ensure_lockbox(df, "hba1c_high", "classification")

    refused = st.session_state.get("_lockbox_redraw_refused")
    assert refused, "the refusal did not fire"
    assert refused["target_changed"] is True
    assert refused["drawn_for"] == "glucose_high" and refused["target"] == "hba1c_high"
    assert refused["n_scoreable"] < refused["n_sealed"], (
        "the fixture should leave some sealed rows unscoreable")
    # and the number is the truth, not the sealed count
    sealed = get_lockbox()["labels"]
    assert refused["n_scoreable"] == int(df.loc[sealed, "hba1c_high"].notna().sum())


def test_the_lockbox_remembers_which_outcome_it_was_drawn_for():
    df = study()
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "glucose_high", "classification")
    assert lb["target_col"] == "glucose_high"


def test_a_restored_session_keeps_its_cleaned_working_table():
    """The whole round trip, through the real save and load."""
    from utils.session_manager import _collect_session_data, _restore_session_data
    rng = np.random.default_rng(8)
    raw = pd.DataFrame({"sex": rng.choice(["Female", "Male"], 412),
                        "age": rng.integers(20, 80, 412),
                        "y": rng.integers(0, 2, 412)})
    cleaned = raw.drop(index=raw.index[:13])          # "Apply: Drop duplicate rows"
    st.session_state["datasets_registry"] = {"d1": raw}
    st.session_state["working_table"] = cleaned
    st.session_state["raw_data"] = cleaned
    st.session_state["data_config"] = Cfg("y")

    archive, _ = _collect_session_data()
    st.session_state.clear()
    _restore_session_data(archive)

    assert st.session_state.get("_working_table_source_id") == "d1", (
        "page 01 will rebuild from the original file and discard the cleaning")
    assert len(st.session_state["working_table"]) == len(cleaned)


def test_a_restored_lockbox_still_knows_it_was_drawn_by_subject():
    from utils.session_manager import _collect_session_data, _restore_session_data
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"subject_id": np.repeat(np.arange(60), 3),
                       "crp": rng.normal(3, 1, 180),
                       "y": rng.integers(0, 2, 180)})
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "y", "classification")
    assert lb["group_col"] == "subject_id"

    archive, _ = _collect_session_data()
    st.session_state.clear()
    _restore_session_data(archive)

    back = get_lockbox()
    assert back["group_col"] == "subject_id", "the split forgot it was by subject"
    assert back["group_kind"] == "subject" and back["target_col"] == "y"
    assert back["n_test_groups"] == lb["n_test_groups"]


def test_the_replay_runs_without_a_visit_to_feature_engineering():
    """The promise is made on Train & Compare, so it has to hold there."""
    rng = np.random.default_rng(21)
    df = pd.DataFrame({"sex": rng.choice(["Female", "Male"], 600),
                       "bmi": rng.normal(27, 4, 600),
                       "y": rng.integers(0, 2, 600)})
    st.session_state["raw_data"] = df
    st.session_state["data_config"] = Cfg("y")
    record("math", {"column": "bmi", "transform": "square", "name": "bmi_squared"},
           ["bmi_squared"])
    # what the switch button does
    st.session_state["engineered_feature_names"] = ["bmi_squared"]
    staged = stage_for_replay("switching to Male")
    assert staged and pending()

    st.session_state.pop("df_engineered", None)
    result = run_pending_replay(df)
    assert result is not None, "nothing replayed"
    assert not pending(), "the pending replay was not consumed"


def test_running_the_replay_twice_does_nothing_the_second_time():
    rng = np.random.default_rng(23)
    df = pd.DataFrame({"bmi": rng.normal(27, 4, 300), "y": rng.integers(0, 2, 300)})
    st.session_state["raw_data"] = df
    st.session_state["data_config"] = Cfg("y")
    record("math", {"column": "bmi", "transform": "square", "name": "bmi_sq"},
           ["bmi_sq"])
    st.session_state["engineered_feature_names"] = ["bmi_sq"]
    stage_for_replay("switching")
    first = run_pending_replay(df)
    assert first is not None
    assert run_pending_replay(df) is None, "a second page replayed it again"
