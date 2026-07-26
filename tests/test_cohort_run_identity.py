"""A banked run must carry the question it answered.

See the audit findings cohort-runs-done-survives-data-and-target-change and
runs-table-not-scoped-to-question: CohortRun recorded only column+label, so a
number could neither be invalidated when the question changed nor excluded when
a different question was being asked. Filtering happens at READ time on
(grouping column, target, task, data fingerprint), so it cannot be forgotten by
a reset path that does not exist yet.
"""
import numpy as np, pandas as pd, pytest, streamlit as st
from utils.session_state import set_data, DataConfig
from utils.cohorts import (plan_cohorts, start_cohort, record_run, completed_runs,
                           all_recorded_runs, clear_cohort)
from utils.test_lockbox import ensure_lockbox

@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()

def frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sex": rng.choice(["Female","Male"], n),
                         "smoker": rng.choice(["never","current"], n),
                         "age": rng.integers(20,80,n),
                         "diabetes": rng.integers(0,2,n),
                         "hypertension": rng.integers(0,2,n)})

def run_group(df, column, label, target):
    st.session_state["data_config"] = DataConfig(
        target_col=target, feature_cols=["age"], task_type="classification")
    ensure_lockbox(df, target, "classification")
    plan = plan_cohorts(df, column, target, "classification")
    start_cohort(df, plan, next(c for c in plan.viable if c.label == label), target)
    return record_run({"ROC-AUC": 0.71})

def test_corrected_reupload_retires_the_old_number():
    df = frame(); set_data(df)
    run_group(df, "sex", "Female", "diabetes")
    assert len(completed_runs("sex")) == 1
    corrected = df.copy(); corrected.loc[:20, "age"] = 99
    set_data(corrected)                       # same columns, different values
    st.session_state["data_config"] = DataConfig(
        target_col="diabetes", feature_cols=["age"], task_type="classification")
    assert completed_runs("sex") == [], "a number from the old data survived"
    assert all_recorded_runs(), "it should still exist, just not be shown"

def test_a_target_swap_retires_it_too():
    df = frame(); set_data(df)
    run_group(df, "sex", "Female", "diabetes")
    st.session_state["data_config"] = DataConfig(
        target_col="hypertension", feature_cols=["age"], task_type="classification")
    assert completed_runs("sex") == []

def test_a_different_grouping_variable_is_not_in_the_table():
    df = frame(); set_data(df)
    run_group(df, "sex", "Female", "diabetes")
    run_group(df, "sex", "Male", "diabetes")
    run_group(df, "smoker", "never", "diabetes")
    assert sorted(r.label for r in completed_runs("sex")) == ["Female", "Male"]
    assert [r.label for r in completed_runs("smoker")] == ["never"]

def test_rerunning_the_same_group_updates_rather_than_duplicates():
    df = frame(); set_data(df)
    run_group(df, "sex", "Female", "diabetes")
    st.session_state["data_config"] = DataConfig(
        target_col="diabetes", feature_cols=["age"], task_type="classification")
    plan = plan_cohorts(df, "sex", "diabetes", "classification")
    start_cohort(df, plan, next(c for c in plan.viable if c.label=="Female"), "diabetes")
    record_run({"ROC-AUC": 0.88})
    got = completed_runs("sex")
    assert len(got) == 1 and got[0].metrics["ROC-AUC"] == 0.88
