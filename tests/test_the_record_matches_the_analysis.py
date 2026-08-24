"""What the exported record says must match what was actually analyzed.

The cohort-disclosure work reached the abstract, Results and methods narrative
but stopped short of four artifacts, and the import-repair logging never
arrived at all:

  - The CONSORT participant-flow figure took n_total=len(df) from an already
    cohort-filtered frame, so with a 319-of-600 Female run it drew "All records
    n = 319" and no exclusion branch — the one figure whose entire job is to
    account for who was excluded positively asserting nobody was.
  - The Evidence Map, which the draft's own preamble names as the proof that
    every quantitative statement traces to a logged event, printed the cohort's
    N with no row about the filter.
  - The reproducibility manifest hashed the cohort and called it the data.
  - Two of the four clear paths never refreshed provenance, so an analysis that
    had gone back to everyone still exported a restriction sentence.
  - Import Doctor repairs were recorded before an upload record existed (so
    they were dropped) and passed rows_before=rows_after=0 (so they zeroed the
    recorded study N).
"""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.publication import generate_flow_diagram_mermaid
from utils.cohorts import plan_cohorts, start_cohort, clear_cohort
from utils.workflow_provenance import WorkflowProvenance, get_provenance


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def study(n=600):
    rng = np.random.default_rng(6)
    return pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                         "age": rng.integers(20, 80, n),
                         "y": rng.integers(0, 2, n)})


def begin_female_run(df):
    st.session_state["raw_data"] = df
    plan = plan_cohorts(df, "sex", "y", "classification")
    cell = next(c for c in plan.viable if c.label == "Female")
    return start_cohort(df, plan, cell, "y")


# ── repairs recorded before there is anywhere to put them ────────────────

def test_a_repair_recorded_before_the_config_save_survives_it():
    prov = WorkflowProvenance()
    prov.record_cleaning("Recoded -9 to missing in `income`", 0, 0,
                         {"column": "income"})
    assert prov.upload is None and len(prov.pending_cleaning_actions) == 1
    prov.record_upload("y", "classification", ["age", "bmi"], 400)
    actions = [a["action"] for a in prov.upload.cleaning_actions]
    assert any("income" in a for a in actions), (
        "the repair was erased by the configuration save")


def test_a_repair_that_removes_no_rows_does_not_zero_the_study_n():
    prov = WorkflowProvenance()
    prov.record_upload("y", "classification", ["age"], 400)
    prov.record_cleaning("Recoded 999 to missing in `bmi`", 0, 0)
    assert prov.upload.n_samples == 400


def test_a_cleaning_action_that_does_remove_rows_still_moves_the_n():
    prov = WorkflowProvenance()
    prov.record_upload("y", "classification", ["age"], 400)
    prov.record_cleaning("Dropped duplicate rows", 400, 387)
    assert prov.upload.n_samples == 387


# ── the restriction follows the run, in both directions ──────────────────

def test_clearing_the_run_clears_the_restriction_however_it_is_cleared():
    df = study()
    prov = get_provenance()
    prov.record_upload("y", "classification", ["age"], len(df))
    begin_female_run(df)
    prov.record_cohort_restriction()
    assert prov.upload.cohort_column == "sex"

    clear_cohort()          # the sidebar repair button does exactly this
    assert prov.upload.cohort_column == "", (
        "an unrestricted analysis still exports a restriction sentence")
    assert prov.upload.restriction_sentence() == ""


# ── the flow diagram accounts for who was excluded ───────────────────────

def test_the_flow_diagram_shows_the_cohort_as_an_exclusion():
    df = study()
    run = begin_female_run(df)
    n_run, n_study = run["n_rows"], run["n_total"]
    assert n_run < n_study

    mermaid = generate_flow_diagram_mermaid(
        n_total=n_study, n_excluded=n_study - n_run,
        exclusion_reasons={"Not sex = Female": n_study - n_run},
        n_analyzed=n_run,
    )
    assert f"N = {n_study:,}" in mermaid, "total records is the cohort, not the study"
    assert "Excluded" in mermaid and "Not sex = Female" in mermaid
    assert "All records" not in mermaid, "asserts nobody was excluded"


def test_page_10_passes_the_study_total_not_the_cohort():
    """The page's own arithmetic, so the figure and the caller cannot drift."""
    import ast
    src = open("pages/10_Report_Export.py", encoding='utf-8').read()
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "generate_flow_diagram_mermaid")
    kwargs = {k.arg for k in call.keywords}
    assert {"n_total", "n_excluded", "exclusion_reasons"} <= kwargs, (
        f"the flow diagram is called without an exclusion branch: {sorted(kwargs)}")
    n_total = next(k.value for k in call.keywords if k.arg == "n_total")
    assert not (isinstance(n_total, ast.Call)
                and getattr(n_total.func, "id", "") == "len"), (
        "n_total is len(df), and df is already cohort-filtered")


# ── the traceability artifacts name the filter ───────────────────────────

def test_the_evidence_map_has_a_row_for_the_restriction():
    from ml.narrative_engine import NarrativeEngine
    df = study()
    prov = get_provenance()
    prov.record_upload("y", "classification", ["age"], len(df))
    run = begin_female_run(df)
    prov.record_cohort_restriction()

    text = NarrativeEngine(prov).generate_evidence_map()
    assert "Sample Restriction" in text, "the traceability artifact omits the filter"
    assert "sex" in text and "Female" in text
    assert f"{run['n_total']:,}" in text, "the study N is nowhere in the map"


def test_the_manifest_names_the_group_it_hashed():
    import ast
    src = open("pages/10_Report_Export.py", encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_build_reproducibility_manifest")
    body = ast.dump(fn)
    assert "cohort_restriction" in body, (
        "the manifest hashes one group and calls it the data")
