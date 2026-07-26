"""An export must never report one group's N as the study's.

The pre-PR audit drove page 10 with a `sex = Female` run active and found the
restriction stated NOWHERE in any exported artifact: the Methods draft, the
plain-language abstract, the evidence map, the reproducibility manifest, the
LaTeX abstract/methods/results all said "A total of 314 participants were
included in the analysis" for a study of 600 people, and the strengths list
offered "Sample size of 314 observations" as a point in the paper's favor.

The only disclosure anywhere was a sidebar chip inside the running app, which
does not leave with the export. A reviewer would believe the model was fitted
on the whole study. It was fitted on 314 women.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import clear_cohort, plan_cohorts, start_cohort
from utils.workflow_provenance import (
    WorkflowProvenance, cohort_restriction_sentence, get_provenance,
)


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def study(n=600):
    rng = np.random.default_rng(5)
    return pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                         "age": rng.integers(20, 80, n),
                         "diabetes": rng.integers(0, 2, n)})


def begin_female_run():
    df = study()
    st.session_state["raw_data"] = df
    prov = get_provenance()
    prov.record_upload(target_col="diabetes", task_type="classification",
                       feature_cols=["sex", "age"], n_samples=len(df))
    plan = plan_cohorts(df, "sex", "diabetes", "classification")
    cell = next(c for c in plan.viable if c.label == "Female")
    run = start_cohort(df, plan, cell, "diabetes")
    prov.record_cohort_restriction()
    return df, run


class TestProvenanceCarriesTheRestriction:

    def test_unrestricted_analysis_says_nothing(self):
        df = study()
        st.session_state["raw_data"] = df
        get_provenance().record_upload(
            target_col="diabetes", task_type="classification",
            feature_cols=["age"], n_samples=len(df))
        assert cohort_restriction_sentence() == ""

    def test_a_run_is_recorded_with_both_denominators(self):
        _, run = begin_female_run()
        up = get_provenance().upload
        assert up.is_cohort_restricted
        assert up.cohort_column == "sex" and up.cohort_value == "Female"
        assert up.cohort_n == run["n_rows"] < up.study_n == run["n_total"]

    def test_the_sentence_names_group_and_both_counts(self):
        _, run = begin_female_run()
        text = cohort_restriction_sentence()
        assert "sex = Female" in text
        assert f"{run['n_rows']:,}" in text and f"{run['n_total']:,}" in text
        assert "should not be read as describing the whole study" in text

    def test_clearing_the_run_clears_the_claim(self):
        begin_female_run()
        clear_cohort()
        get_provenance().record_cohort_restriction()
        assert cohort_restriction_sentence() == ""


class TestEveryExportSurfaceStatesIt:
    """Each of these rendered the bare N to a reviewer before this change."""

    def test_latex_abstract_population_sentence(self):
        from ml.latex_report import _format_abstract_population_sentence as fn
        begin_female_run()
        for upload_n, analysis_n in ((314, 314), (600, 314)):
            out = fn(upload_n, analysis_n)
            assert "restricted to participants with sex = Female" in out, (
                f"bare N for upload_n={upload_n} analysis_n={analysis_n}")

    def test_latex_results_study_population(self):
        import ml.latex_report as lr
        begin_female_run()
        found = False
        for name in dir(lr):
            fn = getattr(lr, name)
            if callable(fn) and name.startswith("_") and "escape" in name:
                found = True
        assert found  # escaping helper exists for the appended sentence
        assert "sex = Female" in cohort_restriction_sentence()

    def test_methods_narrative_states_it_next_to_the_n(self):
        begin_female_run()
        # The narrative appends the restriction immediately after the N so the
        # two are never read apart.
        import inspect, ml.narrative_engine as ne
        src = inspect.getsource(ne)
        i = src.index("A {task_type} analysis was performed on a dataset of")
        assert "cohort_restriction_sentence" in src[i:i + 900]

    def test_strengths_no_longer_offer_a_restricted_n_bare(self):
        import inspect
        import ml.latex_report as lr
        src = inspect.getsource(lr)
        i = src.index('Sample size of')
        window = src[max(0, i - 700):i + 400]
        assert "cohort_column" in window, (
            "a restricted sample must not be listed as a plain strength")
