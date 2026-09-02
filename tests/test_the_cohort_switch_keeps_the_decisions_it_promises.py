"""The button says the decisions come along. Now they do.

"Now run the same analysis on Male" told the researcher, in the text right
above it, that "your preprocessing choices are carried over". The restore
appended that phrase to a list of notes and wrote nothing back. Meanwhile the
reset it followed had set `preprocessing_config_by_model` to `{}`, and
Streamlit had already dropped every `preprocess_*` widget key — they belong to
widgets on a page that was not rendered on that run — along with the
`train_model_*` checkboxes, which live above a gate the page now stops at.
The Male pipeline was then rebuilt from shipped defaults: two cohorts, two
different pipelines, presented as answering the same question.

A widget's value cannot be written back at the switch. Streamlit drops a
widget's value at the end of any run that does not render the widget, and
refuses a write once the widget has been instantiated on the current run. So
the decisions are parked under a key no widget owns, and each page claims its
own on the run that renders them, just before they are instantiated. These
tests pin the staging, the claims, the mapping from a built config back to
widget keys, and the words that describe all of it — because the previous
words described nothing.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils import replay as R
from utils.cohorts import CohortRun, plan_cohorts, record_run, start_cohort
from utils.session_state import (
    DataConfig, reset_data_dependent_state, reset_downstream_results, set_data,
)
from utils.test_lockbox import ensure_lockbox, train_row_mask


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _study(n=600):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                         "age": rng.integers(20, 80, n),
                         "bmi": np.round(rng.normal(28, 5, n), 1),
                         "chol": np.round(rng.normal(200, 30, n), 1),
                         "y": rng.integers(0, 2, n)})


def _in_female(df=None):
    df = _study() if df is None else df
    set_data(df)
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=["age", "bmi", "chol"],
        task_type="classification")
    ensure_lockbox(df, "y", "classification")
    plan = plan_cohorts(df, "sex", "y", "classification",
                        train_mask=train_row_mask(df.index))
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
    return df


FEMALE_LOGREG = {
    "numeric_imputation": "iterative",
    "numeric_scaling": "robust",
    "numeric_log_transform": False,
    "numeric_power_transform": "yeo-johnson",
    "numeric_missing_indicators": True,
    "numeric_outlier_treatment": "mad",
    "numeric_outlier_params": {"threshold": 3.0},
    "categorical_imputation": "constant",
    "categorical_encoding": "target",
    "use_kmeans_features": True,
    "kmeans_n_clusters": 4,
    "kmeans_add_distances": False,
    "kmeans_add_onehot": True,
    "use_pca": True,
    "pca_n_components": 5,
    "pca_whiten": True,
    "unit_harmonization": False,
    "plausibility_gating": True,
    "plausibility_mode": "filter",
    "interpretability_mode": "performance",
    # Data-derived — produced by the build, owned by no widget.
    "numeric_features": ["age", "bmi", "chol"],
    "categorical_features": [],
    "n_output_features": 9,
    "overrides": ["PCA components reduced to 5"],
}


def _decide_in_female(**extra):
    """What a researcher's session looks like when the button is pressed."""
    _in_female()
    st.session_state["preprocessing_config_by_model"] = {"logreg": dict(FEMALE_LOGREG)}
    st.session_state["preprocessing_config"] = {"numeric_features": ["age", "bmi", "chol"]}
    st.session_state["preprocess_built_model_keys"] = ["logreg"]
    st.session_state["train_model_logreg"] = True
    st.session_state["train_model_rf"] = False
    st.session_state["logreg_C"] = 0.5
    st.session_state["logreg_max_iter"] = 2000
    st.session_state["preprocess_config_mode"] = R.ADVANCED_MODE_LABEL
    st.session_state["interpretability_mode"] = "performance"
    for k, v in extra.items():
        st.session_state[k] = v


def _switch():
    """The three steps the switch takes, without the rerun."""
    R.stage_for_replay(reason="cohort switch")
    reset_downstream_results(clear_feature_engineering=True)
    return R.restore_decisions()


def _cull(prefixes=("preprocess_", "train_model_", "logreg_", "interpretability_mode")):
    """What Streamlit does to widget keys on a run that renders no widget."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(prefixes):
            st.session_state.pop(k, None)


class TestStagingCapturesTheDecisions:

    def test_the_built_config_is_parked_apart_from_the_recipe(self):
        _decide_in_female()
        R.stage_for_replay(reason="cohort switch")
        d = R.decisions_pending()
        assert d is not None
        assert d["from_label"] == "Female"
        assert d["preprocess"]["config_by_model"]["logreg"]["numeric_scaling"] == "robust"
        assert d["preprocess"]["mode"] == R.ADVANCED_MODE_LABEL
        assert d["preprocess"]["interpretability_mode"] == "performance"
        assert d["models"]["picks"] == ["logreg"], "an unticked model is not a pick"
        assert d["models"]["hyperparams"] == {
            "logreg": {"logreg_C": 0.5, "logreg_max_iter": 2000}}

    def test_the_parked_config_is_a_copy_not_the_pages_dict(self):
        _decide_in_female()
        R.stage_for_replay(reason="cohort switch")
        st.session_state["preprocessing_config_by_model"]["logreg"]["numeric_scaling"] = "minmax"
        d = R.decisions_pending()
        assert d["preprocess"]["config_by_model"]["logreg"]["numeric_scaling"] == "robust", (
            "a later build mutates the page's sub-dicts in place")

    def test_the_reset_leaves_the_parked_decisions_alone(self):
        _decide_in_female()
        _switch()
        assert not st.session_state.get("preprocessing_config_by_model"), (
            "the config of BUILT pipelines must not survive: nothing is built")
        assert not st.session_state.get("preprocess_built_model_keys")
        assert R.decisions_pending()["preprocess"]["config_by_model"]["logreg"]

    def test_built_models_are_the_fallback_when_no_checkbox_is_ticked(self):
        _decide_in_female()
        _cull(("train_model_",))                      # the checkboxes are gone
        R.stage_for_replay(reason="cohort switch")
        assert R.decisions_pending()["models"]["picks"] == ["logreg"]

    def test_the_list_of_built_models_is_state_not_a_choice(self):
        _decide_in_female()
        R.stage_for_replay(reason="cohort switch")
        d = R.decisions_pending()
        assert "preprocess_built_model_keys" not in d["preprocess"]["widgets"]
        assert "preprocess_config_mode" not in d["preprocess"]["widgets"], (
            "the mode is carried on its own terms, not as a stray widget")

    def test_nothing_to_carry_stages_nothing(self):
        _in_female()
        assert R.stage_for_replay(reason="cohort switch") is None
        assert R.decisions_pending() is None
        assert R.describe_pending_decisions() == ""


class TestThePreprocessPageClaimsItsWidgets:

    def test_a_config_seeds_the_widgets_that_rebuild_it(self):
        seeds = R._preprocess_widget_seeds("logreg", FEMALE_LOGREG)
        p = "preprocess_logreg_"
        assert seeds[p + "numeric_imputation"] == "iterative"
        assert seeds[p + "numeric_scaling"] == "robust"
        assert seeds[p + "numeric_power_transform"] == "yeo-johnson"
        assert seeds[p + "numeric_log_transform"] is False
        assert seeds[p + "numeric_missing_indicators"] is True
        assert seeds[p + "numeric_outlier_treatment"] == "mad"
        assert seeds[p + "outlier_mad_threshold"] == 3.0
        assert p + "outlier_lower_q" not in seeds, "the other treatment's inputs are not this choice"
        assert seeds[p + "categorical_imputation"] == "constant"
        assert seeds[p + "categorical_encoding"] == "target"
        # The config and the widgets name these differently.
        assert seeds[p + "use_kmeans"] is True
        assert seeds[p + "kmeans_n_clusters"] == 4
        assert seeds[p + "kmeans_distances"] is False
        assert seeds[p + "kmeans_onehot"] is True
        assert seeds[p + "use_pca"] is True
        assert seeds[p + "pca_mode"] == "Fixed Components"
        assert seeds[p + "pca_fixed_n"] == 5
        assert seeds[p + "pca_n_components"] == 5
        assert seeds[p + "pca_whiten"] is True
        assert seeds[p + "plausibility_gating"] is True
        assert seeds[p + "plausibility_mode"] == "filter"
        assert seeds[p + "unit_harmonization"] is False
        for data_derived in ("numeric_features", "categorical_features",
                             "n_output_features", "overrides", "interpretability_mode"):
            assert p + data_derived not in seeds, (
                f"{data_derived} is produced by the build, not chosen on a widget")

    def test_pca_by_variance_seeds_the_slider_not_the_number_input(self):
        seeds = R._preprocess_widget_seeds("ridge", {"use_pca": True, "pca_n_components": 0.9})
        assert seeds["preprocess_ridge_pca_mode"] == "Variance Threshold"
        assert seeds["preprocess_ridge_pca_variance"] == 0.9
        assert "preprocess_ridge_pca_fixed_n" not in seeds

    def test_a_legacy_log_flag_becomes_the_log1p_transform(self):
        seeds = R._preprocess_widget_seeds("ridge", {"numeric_log_transform": True})
        assert seeds["preprocess_ridge_numeric_power_transform"] == "log1p"
        assert seeds["preprocess_ridge_numeric_log_transform"] is True

    def test_percentile_outliers_carry_both_bounds(self):
        seeds = R._preprocess_widget_seeds("ridge", {
            "numeric_outlier_treatment": "percentile",
            "numeric_outlier_params": {"lower_q": 0.02, "upper_q": 0.97}})
        assert seeds["preprocess_ridge_outlier_lower_q"] == 0.02
        assert seeds["preprocess_ridge_outlier_upper_q"] == 0.97
        assert "preprocess_ridge_outlier_mad_threshold" not in seeds

    def test_the_claim_seeds_the_page_forces_advanced_and_carries_the_picks(self):
        _decide_in_female()
        _switch()
        _cull()                                       # the rerun landed on a page without them
        assert "preprocess_logreg_numeric_scaling" not in st.session_state
        got = R.claim_for_preprocess_page()
        assert got == {"from_label": "Female", "models": ["logreg"],
                       "picks": ["logreg"], "mode_forced": True}
        assert st.session_state["preprocess_logreg_numeric_scaling"] == "robust"
        assert st.session_state["preprocess_logreg_outlier_mad_threshold"] == 3.0
        assert st.session_state["preprocess_config_mode"] == R.ADVANCED_MODE_LABEL, (
            "in Smart Defaults mode the page overwrites every seeded key from "
            "the new group's profile before the build can read it")
        assert st.session_state["interpretability_mode"] == "performance"
        assert st.session_state["train_model_logreg"] is True
        assert st.session_state.get("_coach_applied") is True, (
            "the coach re-applies its own picks once per reset and would add "
            "models the researcher did not choose for the first group")

    def test_the_advanced_label_is_the_pages_own(self):
        src = open("pages/05_Preprocess.py", encoding="utf-8").read()
        assert repr(R.ADVANCED_MODE_LABEL) in src or f'"{R.ADVANCED_MODE_LABEL}"' in src, (
            "the radio option changed on the page; the carry would land in Smart Defaults")

    def test_the_claim_is_one_shot_and_leaves_the_train_page_its_share(self):
        _decide_in_female()
        _switch()
        R.claim_for_preprocess_page()
        left = R.decisions_pending()
        assert left is not None and "preprocess" not in left
        assert "picks" not in left["models"]
        assert left["models"]["hyperparams"]["logreg"]["logreg_C"] == 0.5
        assert R.claim_for_preprocess_page() is None, "claimed twice would overwrite edits"

    def test_widget_choices_without_a_built_config_still_restore(self):
        _in_female()
        st.session_state["preprocess_ridge_numeric_scaling"] = "minmax"
        _switch()
        _cull()
        got = R.claim_for_preprocess_page()
        assert st.session_state["preprocess_ridge_numeric_scaling"] == "minmax"
        assert got is not None and got["mode_forced"] is False, (
            "no built config, nothing to force the mode for")

    def test_the_built_config_outranks_a_later_unbuilt_edit(self):
        _decide_in_female(preprocess_logreg_numeric_scaling="minmax")   # edited, never rebuilt
        _switch()
        _cull()
        R.claim_for_preprocess_page()
        assert st.session_state["preprocess_logreg_numeric_scaling"] == "robust", (
            "the built config is what the previous group's models actually used")


class TestTheTrainPageClaimsItsWidgets:

    def test_hyperparameters_return_to_their_controls_after_the_cull(self):
        _decide_in_female()
        _switch()
        _cull()
        R.claim_for_preprocess_page()
        assert "logreg_C" not in st.session_state
        got = R.claim_for_train_page()
        assert got == {"from_label": "Female", "picks": [], "hyperparams": ["logreg"]}
        assert st.session_state["logreg_C"] == 0.5
        assert st.session_state["logreg_max_iter"] == 2000
        assert R.decisions_pending() is None, "everything claimed, nothing left to announce"

    def test_the_train_page_can_claim_the_picks_if_it_renders_first(self):
        _decide_in_female()
        _switch()
        _cull()
        got = R.claim_for_train_page()
        assert got["picks"] == ["logreg"]
        assert st.session_state["train_model_logreg"] is True
        assert R.decisions_pending()["preprocess"], "the preprocess share is still waiting"

    def test_a_model_key_with_an_underscore_is_named_whole(self):
        _in_female()
        st.session_state["train_model_knn_clf"] = True
        st.session_state["knn_clf_n_neighbors"] = 7
        R.stage_for_replay(reason="cohort switch")
        d = R.decisions_pending()
        assert d["models"]["hyperparams"] == {"knn_clf": {"knn_clf_n_neighbors": 7}}
        assert "KNN_CLF" in R.describe_pending_decisions()

    def test_a_tuned_result_is_not_carried_only_the_controls_are(self):
        """Optuna's best_params live in a local on page 06 and in provenance,
        which the reset nulls; the widgets keep the researcher's own values."""
        _decide_in_female()
        _switch()
        d = R.decisions_pending()
        assert "selected_model_params" not in str(d)
        assert d["models"]["hyperparams"]["logreg"] == {"logreg_C": 0.5, "logreg_max_iter": 2000}


class TestTheWordsMatchWhatHappens:

    def test_restore_notes_name_what_is_parked_not_what_was_written(self):
        _decide_in_female()
        notes = _switch()
        joined = " | ".join(notes)
        assert "preprocessing settings for LOGREG" in joined
        assert "model picks (LOGREG)" in joined
        assert "hyperparameter settings for LOGREG" in joined
        assert "your preprocessing choices" not in joined, (
            "that phrase used to be appended with nothing written back")

    def test_the_sidebar_says_what_is_waiting_and_where_until_nothing_is(self):
        _decide_in_female()
        _switch()
        text = R.describe_pending_decisions()
        assert text.startswith("Carried from the Female run:")
        assert "Preprocess" in text and "Train & Compare" in text
        R.claim_for_preprocess_page()
        text = R.describe_pending_decisions()
        assert "hyperparameter settings for LOGREG" in text
        assert "preprocessing settings" not in text, "already applied"
        assert "Preprocess" not in text.split("Applied when you open")[1]
        R.claim_for_train_page()
        assert R.describe_pending_decisions() == ""

    def test_the_chip_renders_the_waiting_caption(self):
        from utils.cohort_ui import render_cohort_chip
        _decide_in_female()
        _switch()
        captions, real = [], st.sidebar.caption
        st.sidebar.caption = lambda body, **kw: captions.append(str(body))
        try:
            render_cohort_chip()
        finally:
            st.sidebar.caption = real
        assert any("Carried from the Female run" in c for c in captions)

    def test_the_chip_is_quiet_when_nothing_is_waiting(self):
        from utils.cohort_ui import render_cohort_chip
        _in_female()
        captions, real = [], st.sidebar.caption
        st.sidebar.caption = lambda body, **kw: captions.append(str(body))
        try:
            render_cohort_chip()
        finally:
            st.sidebar.caption = real
        assert not captions

    def test_the_switch_warns_that_the_report_will_not_be_kept(self):
        from utils.cohort_ui import render_next_cohort
        _decide_in_female()
        st.session_state["latex_report"] = "\\documentclass{article}"
        st.session_state["compiled_pdf"] = b"%PDF"
        warned, real = [], st.warning
        st.warning = lambda body, **kw: warned.append(str(body))
        try:
            render_next_cohort("classification", {"Best model": "logreg", "ROC-AUC": 0.7})
        finally:
            st.warning = real
        hit = [w for w in warned if "will not be kept" in w]
        assert hit, warned
        assert "LaTeX report" in hit[0] and "compiled PDF" in hit[0]
        assert "Female" in hit[0] and "Report & Export" in hit[0]

    def test_no_report_means_no_warning_about_one(self):
        from utils.cohort_ui import render_next_cohort
        _decide_in_female()
        warned, real = [], st.warning
        st.warning = lambda body, **kw: warned.append(str(body))
        try:
            render_next_cohort("classification", {"Best model": "logreg", "ROC-AUC": 0.7})
        finally:
            st.warning = real
        assert not any("will not be kept" in w for w in warned)

    def test_the_button_text_names_every_kind_of_decision_it_carries(self):
        import utils.cohort_ui as cu
        src = inspect.getsource(cu.render_next_cohort)
        for phrase in ("preprocessing settings", "model picks", "hyperparameter choices",
                       "rebuilt from their formulas", "refit on", "a tuned hyperparameter"):
            assert phrase in src, phrase
        assert "_replay_note" not in inspect.getsource(cu), (
            "written on both switch paths and read by nothing")

    def test_the_train_page_gate_says_where_the_settings_are(self):
        src = open("pages/06_Train_and_Compare.py", encoding="utf-8").read()
        i = src.index("if pipeline is None and not pipelines_by_model:")
        window = src[i:i + 1500]
        assert "decisions_pending()" in window
        assert "waiting on the" in window and "Preprocess" in window
        assert "Please build your preprocessing pipelines first" in window, (
            "the plain message must survive for the plain case")

    def test_the_preprocess_page_claims_before_its_first_widget(self):
        src = open("pages/05_Preprocess.py", encoding="utf-8").read()
        claim = src.index("claim_for_preprocess_page()")
        first_widget = min(src.index('key="preprocess_config_mode"'),
                           src.index("_coach_applied"),
                           src.index('key=f"btn_{ck}"'))
        assert claim < first_widget, (
            "a write to an instantiated widget's key raises; the claim must come first")

    def test_the_train_page_claims_past_the_gate_and_before_its_checkboxes(self):
        src = open("pages/06_Train_and_Compare.py", encoding="utf-8").read()
        gate = src.index("if pipeline is None and not pipelines_by_model:")
        claim = src.index("claim_for_train_page()")
        first_checkbox = src.index('checkbox_key = f"train_model_{model_key}"')
        assert gate < claim < first_checkbox, (
            "claimed before the gate, the values die with the run that stops there")


class TestTheComparisonTableShowsWhatWentFlat:

    def _runs(self, dropped_a, dropped_b):
        return [CohortRun(column="sex", label="Female", n_train=200, n_test=50,
                          dropped_features=dropped_a, completed=True,
                          metrics={"ROC-AUC": 0.71}),
                CohortRun(column="sex", label="Male", n_train=210, n_test=55,
                          dropped_features=dropped_b, completed=True,
                          metrics={"ROC-AUC": 0.66})]

    def test_a_predictor_constant_in_one_group_is_named_beside_the_scores(self):
        from utils.cohort_ui import _runs_table
        table = _runs_table(self._runs([], ["pregnancies"]))
        col = table["Constant in this group"]
        assert list(col) == ["—", "pregnancies"]

    def test_no_flat_predictor_anywhere_means_no_column(self):
        from utils.cohort_ui import _runs_table
        table = _runs_table(self._runs([], []))
        assert "Constant in this group" not in table.columns


class TestParkedDecisionsDoNotOutliveTheData:

    def test_a_new_dataset_drops_them(self):
        _decide_in_female()
        _switch()
        assert R.decisions_pending()
        reset_data_dependent_state()
        assert R.decisions_pending() is None
        assert R.pending() is None

    def test_a_saved_session_keeps_them(self):
        from utils.session_manager import _PLAIN_KEYS
        assert "cohort_decisions_pending" in _PLAIN_KEYS
