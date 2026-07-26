"""The lockbox invariant, as the pre-PR audit broke it.

full_study=True used to mean "skip the cohort filter", not "the whole study".
Feature engineering inside a run writes a cohort-sized df_engineered, so
full_study returned one group's rows to its two callers — the test lockbox and
the cohort chooser — both of which document that they need all of them. Page 01
then redrew the split on 427 women and 56 rows sealed since upload became
trainable, while the audit caption above still promised the whole study.
"""
import numpy as np, pandas as pd, pytest, streamlit as st
from utils.session_state import get_data
from utils.cohorts import plan_cohorts, start_cohort, clear_cohort
from utils.test_lockbox import ensure_lockbox, get_lockbox, train_row_mask


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def study(n=800):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"sex": rng.choice(["Female","Male"], n),
                         "age": rng.integers(20,80,n),
                         "y": rng.integers(0,2,n)})


def test_engineering_inside_a_run_cannot_redraw_the_lockbox():
    df = study(); st.session_state["raw_data"] = df
    lb1 = ensure_lockbox(df, "y", "classification")
    sealed = set(lb1["labels"])

    plan = plan_cohorts(df, "sex", "y", "classification")
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")

    # page 03 writes an engineered frame built from the cohort view
    st.session_state["df_engineered"] = get_data().assign(age_sq=lambda d: d["age"]**2)

    whole = get_data(full_study=True)
    assert len(whole) == len(df), "full_study must be the whole study"
    assert set(whole["sex"].unique()) == {"Female", "Male"}

    ensure_lockbox(whole, "y", "classification")          # page 01 revisit
    assert set(get_lockbox()["labels"]) == sealed, "the sealed set changed"

    leaked = sealed - set(get_data(full_study=True).index[
        train_row_mask(get_data(full_study=True).index).values == False])
    assert not leaked, f"{len(leaked)} sealed rows became trainable"


def test_the_second_cohort_is_still_reachable():
    from utils.cohorts import cohort_candidates
    df = study(); st.session_state["raw_data"] = df
    ensure_lockbox(df, "y", "classification")
    plan = plan_cohorts(df, "sex", "y", "classification")
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
    st.session_state["df_engineered"] = get_data().assign(age_sq=lambda d: d["age"]**2)
    assert "sex" in cohort_candidates(get_data(full_study=True), "y")


def test_a_redraw_attempt_mid_run_is_refused_and_disclosed():
    df = study(); st.session_state["raw_data"] = df
    lb1 = ensure_lockbox(df, "y", "classification")
    plan = plan_cohorts(df, "sex", "y", "classification")
    start_cohort(df, plan, plan.viable[0], "y")
    ensure_lockbox(df.iloc[:400], "y", "classification")   # a narrowed frame
    assert get_lockbox()["labels"] == lb1["labels"]
    assert st.session_state.get("_lockbox_redraw_refused")
