"""Two things a cohort run recorded, said out loud, and then did not do.

**`dropped_features`.** Filtering to the women makes `sex` constant, and
anything that only varies with it goes flat too. `start_cohort` has always
recorded those columns; the chooser has always named them; and nothing ever
removed them. Every use in the tree was display or serialization. So a
single-sex model was fitted with a constant column in it, and page 07 ranked
`sex` at importance ≈ 0 — which reads as a finding about sex and is an artifact
of the filter.

**External validation.** Page 07 scored a one-group model against every row of
the external file. That measures a different thing — the model's performance in
a population it was not fitted for — and reports it, in the Methods, as external
validation of this model.

Both are defects in the sequential design, independent of persistence. They are
here because branches make them reachable twice as often: with two banked
groups, whichever one you switch to is fitted the same wrong way.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import (
    active_cohort, clear_cohort, constant_in_cohort, plan_cohorts, start_cohort,
    training_features,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

_WIPE = ("cohort_run", "raw_data", "filtered_data", "data_config",
         "_raw_data_fingerprint", "test_lockbox", "_cohort_filter_broken",
         "selected_features")


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in _WIPE:
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


def study(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(20, 80, n),
        "bmi": rng.normal(28, 5, n),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
    })


def enter(df, label, dropped=("sex",)):
    plan = plan_cohorts(df, "sex", "y", "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return start_cohort(df, plan, cell, "y", dropped_features=list(dropped))


# ── the working feature list ─────────────────────────────────────────────

class TestAConstantColumnLeavesTheModel:

    def test_a_flat_predictor_is_not_handed_to_the_fit(self):
        df = study()
        st.session_state["raw_data"] = df
        enter(df, "Female")
        assert training_features(["sex", "age", "bmi"]) == ["age", "bmi"]

    def test_the_shared_selection_is_untouched(self):
        """`selected_features` is the QUESTION, and it is the same question in
        both groups. Rewriting it here would change what the other branch was
        fitted for, silently, from inside a training run."""
        df = study()
        st.session_state["raw_data"] = df
        st.session_state["selected_features"] = ["sex", "age", "bmi"]
        enter(df, "Female")
        training_features(st.session_state["selected_features"])
        assert st.session_state["selected_features"] == ["sex", "age", "bmi"]

    def test_with_no_cohort_nothing_is_dropped(self):
        df = study()
        st.session_state["raw_data"] = df
        clear_cohort()
        assert training_features(["sex", "age"]) == ["sex", "age"]

    def test_a_group_that_would_lose_everything_keeps_everything(self):
        """A model with no predictors is a worse failure than one with a
        constant column, and a failure the researcher cannot act on."""
        df = study()
        st.session_state["raw_data"] = df
        enter(df, "Female", dropped=("sex", "age", "bmi"))
        assert training_features(["sex", "age", "bmi"]) == ["sex", "age", "bmi"]

    def test_the_page_can_say_which_ones_it_left_out(self):
        df = study()
        st.session_state["raw_data"] = df
        enter(df, "Female")
        assert constant_in_cohort(["sex", "age", "bmi"]) == ["sex"]

    def test_both_feature_list_sites_on_page_06_go_through_the_helper(self):
        """There are TWO independent reads of the selection on the training
        path and they share no variable: the split builds `feature_cols`, and
        `feature_names_by_model` re-reads `selected_features` from scratch
        several hundred lines later. Filtering only the first leaves the
        exported per-model feature names listing a column the model was never
        fitted on — and pages 06 and 10 both print that list.
        """
        src = (ROOT / "pages" / "06_Train_and_Compare.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Site one: what the split is given.
        split_feed = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "feature_cols" for t in node.targets)
            and "training_features" in ast.dump(node.value)
        ]
        assert split_feed, (
            "the feature list handed to the split does not go through "
            "training_features(); the fit gets the cohort's constant columns")

        # Site two: what the export says the model was fitted on.
        named = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any("feature_names_by_model" in ast.dump(t) for t in node.targets)
        ]
        assert named, "feature_names_by_model is no longer written on page 06"
        assert all("training_features" in ast.dump(node.value) for node in named), (
            "feature_names_by_model is built from the unfiltered selection, so "
            "the exported per-model feature names list a column the model was "
            "never fitted on — pages 06 and 10 both print that list")


# ── the reconcile warning is not about this ──────────────────────────────

def test_an_intended_drop_is_not_reported_as_a_state_drift():
    """The pipeline is built on page 05 over the whole study's predictors, so a
    column this cohort drops on purpose arrives at `reconcile_pipeline_columns`
    looking exactly like drift. Telling the researcher to "reconfigure
    intentionally" about a decision the app just made for them, on every
    training run, is how a correct app teaches distrust."""
    src = (ROOT / "pages" / "06_Train_and_Compare.py").read_text(encoding="utf-8")
    assert "_unexpected = [c for c in _dropped_cols" in src, (
        "the reconcile warning no longer separates a cohort's intended drops "
        "from a real drift")
    i = src.index("_unexpected = [c for c in _dropped_cols")
    after = src[i:i + 900]
    assert "if _unexpected:" in after, (
        "the warning still fires on the full dropped list")


# ── external validation ──────────────────────────────────────────────────

class TestExternalValidationRespectsTheGroup:

    def _source(self):
        return (ROOT / "pages" / "07_Explainability.py").read_text(encoding="utf-8")

    def test_the_external_frame_itself_is_filtered_not_just_the_score_inputs(self):
        """The stored record and the provenance event both read
        `ext_df.shape[0]`. Filtering only the model inputs would put the
        UNFILTERED external N into the manuscript — over-stating the validation
        cohort, which is the defect this fix exists to remove."""
        src = self._source()
        assert "ext_df = ext_df.loc[cohort_mask(ext_df" in src, (
            "the external frame is not being filtered by the active cohort")
        filtered_at = src.index("ext_df = ext_df.loc[cohort_mask(ext_df")
        recorded_at = src.index("'n_rows': int(ext_df.shape[0])")
        assert filtered_at < recorded_at, (
            "the row count is recorded before the frame is filtered")

    def test_a_missing_grouping_column_is_refused_with_a_reason(self):
        """Not silently skipped. Scoring a Female-only model against everybody
        in the external file measures something else and reports it as external
        validation of this model."""
        src = self._source()
        assert "no way to select " in src and "the same group in it" in src

    def test_an_empty_group_in_the_external_file_is_refused(self):
        src = self._source()
        assert "to be validated on" in src and "has nobody " in src

    def test_the_record_says_which_group_it_describes(self):
        src = self._source()
        assert "'cohort': (f\"{_run_now['column']}={_run_now['label']}\"" in src

    def test_the_filter_selects_the_same_group(self):
        """The mechanism, run for real rather than read: `cohort_mask` is what
        `apply_cohort` uses on the internal frame, so the external file is
        selected by the same rule the study was."""
        from utils.cohorts import cohort_mask
        df = study()
        st.session_state["raw_data"] = df
        run = enter(df, "Female")

        external = pd.DataFrame({
            "sex": ["Female", "Male", "Female", None],
            "age": [40, 50, 60, 70],
            "y": [0, 1, 1, 0],
        })
        kept = external.loc[cohort_mask(external, run["column"], run["value"])]
        assert list(kept["sex"]) == ["Female", "Female"], (
            "a missing value must belong to no group, not to this one")


# ── the counts have to describe the fit, not the selection ───────────────

class TestTheFeatureCountIsTheFitsCount:
    """Applying `dropped_features` made the models smaller than the selection.
    Three places went on printing the selection's size as the model's, so a
    one-group run said "Training on these: 8" over a model fitted on 7."""

    def test_page_06_counts_what_the_split_was_built_from(self):
        src = (ROOT / "pages" / "06_Train_and_Compare.py").read_text(encoding="utf-8")
        assert "_fitted_on = st.session_state.get('feature_names')" in src, (
            "the training summary counts the selection, not the fit")
        assert "len(_fitted_on) if _fitted_on" in src

    def test_the_report_names_the_predictors_this_group_lost(self):
        """Labeled the same way the Rows line above it is. An unlabeled count
        is the one a reader writes down."""
        src = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
        assert "and left out of this group's " in src
        assert "_fit_features = _training_features(_sel_features)" in src

    def test_the_metadata_separates_fitted_from_selected(self):
        src = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
        assert "'n_features': len(_fitted_features)" in src, (
            "metadata.json attributes a column to a model that never saw it")
        assert "'n_features_selected':" in src
        assert "'features_constant_in_cohort':" in src


class TestTheExternalRefusalNamesTheRealProblem:
    """Both cohort refusals blocked validation by pushing the grouping column
    into `missing_cols`, so the empty-group case printed "Missing columns in
    external dataset: ['sex']" about a column the file plainly HAS — sending
    the researcher to fix a schema that is already correct."""

    def _source(self):
        return (ROOT / "pages" / "07_Explainability.py").read_text(encoding="utf-8")

    def test_the_block_flag_is_not_the_missing_column_list(self):
        src = self._source()
        assert "_cohort_blocked = False" in src
        assert "missing_cols = missing_cols or [_col]" not in src, (
            "the refusal still fabricates a missing column")
        assert "if missing_cols or _cohort_blocked:" in src

    def test_genuinely_missing_columns_are_still_named(self):
        """A file with two problems must not have to be fixed twice."""
        src = self._source()
        i = src.index("if missing_cols or _cohort_blocked:")
        after = src[i:i + 900]
        assert "if missing_cols:" in after
        assert "Missing columns in external dataset" in after
