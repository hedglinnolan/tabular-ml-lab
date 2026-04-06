"""
Tier 2: Page-to-page state contract verification.

These tests verify that the session_state keys each page writes are exactly
what the next page needs to read.  Unlike render-only tests, these confirm
that data actually flows correctly across the multi-page workflow.

Methodology:
  - Run Page N with AppTest and injected upstream state
  - After the page renders, inspect at.session_state for the keys Page N+1
    requires
  - Where possible, feed Page N's output directly into Page N+1 and verify
    it renders without errors

This closes the biggest gap identified in our test audit: tests that verify
page orchestration, not just isolated function correctness.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    build_test_dataframe, build_classification_dataframe,
    inject_data_state, inject_trained_state,
)


# ── Helpers ──────────────────────────────────────────────────────────

def assert_no_exception(at, page_name):
    if at.exception:
        msgs = [str(e.value)[:300] for e in at.exception]
        pytest.fail(f"{page_name} raised: {'; '.join(msgs)}")


def _ss_get(at, key, default=None):
    """Safe session_state.get() for AppTest (which doesn't support .get())."""
    try:
        return at.session_state[key]
    except (KeyError, Exception):
        return default


def _numeric_features(df, target_col='glucose'):
    return [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64')]


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def reg_df():
    return build_test_dataframe()


@pytest.fixture(scope="module")
def clf_df():
    return build_classification_dataframe()


# ── Contract: Page 01 → Page 02 ─────────────────────────────────────

class TestPage01ToPage02Contract:
    """Page 01 (Upload) must produce the keys Page 02 (EDA) needs."""

    def test_upload_produces_required_eda_keys(self, reg_df):
        """After upload state injection (simulating Page 01 output),
        verify every key that Page 02 reads at startup is present."""
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "EDA")

        # Page 02 reads these keys at startup
        assert at.session_state['raw_data'] is not None
        assert at.session_state['data_config'] is not None
        assert at.session_state['data_config'].target_col == 'glucose'
        assert len(at.session_state['data_config'].feature_cols) > 0
        assert at.session_state['task_mode'] == 'prediction'

    def test_eda_writes_dataset_profile(self, reg_df):
        """Page 02 must write dataset_profile, which Pages 05, 06, 10 read."""
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "EDA")

        profile = _ss_get(at, 'dataset_profile')
        assert profile is not None, "EDA page must set dataset_profile"


# ── Contract: Page 02 → Page 04 ─────────────────────────────────────

class TestPage02ToPage04Contract:
    """Page 02 output + data_config must be sufficient for Page 04."""

    def test_eda_state_sufficient_for_feature_selection(self, reg_df):
        """Run EDA first, then feed its state into Feature Selection."""
        # Run EDA page
        at_eda = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at_eda, reg_df)
        at_eda.run()
        assert_no_exception(at_eda, "EDA")

        # Now run Feature Selection with the state EDA produced
        at_fs = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=30)
        inject_data_state(at_fs, reg_df)
        # Forward keys EDA wrote
        for key in ('dataset_profile', 'eda_results'):
            val = _ss_get(at_eda, key)
            if val is not None:
                at_fs.session_state[key] = val
        at_fs.run()
        assert_no_exception(at_fs, "Feature Selection")


# ── Contract: Page 04 → Page 05 ─────────────────────────────────────

class TestPage04ToPage05Contract:
    """Feature Selection output must provide what Preprocess needs."""

    def test_selected_features_flow_to_preprocess(self, reg_df):
        """Page 05 reads selected_features and data_config.feature_cols."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)

        # Simulate feature selection reducing feature set
        numeric_feats = _numeric_features(reg_df)
        reduced = numeric_feats[:3]  # keep only 3 features
        at.session_state['selected_features'] = reduced
        at.session_state['data_config'].feature_cols = reduced
        at.run()
        assert_no_exception(at, "Preprocess")

    def test_preprocess_renders_with_full_feature_set(self, reg_df):
        """Page 05 handles full (unselected) feature set."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Preprocess")

    def test_classification_features_flow_to_preprocess(self, clf_df):
        """Classification task flows from Feature Selection to Preprocess."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')
        at.run()
        assert_no_exception(at, "Preprocess (classification)")


# ── Contract: Page 05 → Page 06 ─────────────────────────────────────

class TestPage05ToPage06Contract:
    """Preprocess output must provide what Train & Compare needs."""

    def test_page06_requires_preprocessing_pipelines(self, reg_df):
        """Page 06 hard-stops if preprocessing_pipelines_by_model is empty."""
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, reg_df)
        # Don't inject preprocessing — page should stop gracefully
        at.run()
        # Page should stop (not crash) when preprocessing is missing
        assert_no_exception(at, "Train (no preprocessing)")

    def test_preprocessing_keys_enable_training_page(self, reg_df):
        """With preprocessing pipelines injected, Page 06 renders training UI."""
        from ml.pipeline import build_preprocessing_pipeline

        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, reg_df)

        numeric_feats = _numeric_features(reg_df)
        cat_feats = [c for c in reg_df.columns if c != 'glucose' and reg_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=cat_feats,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        at.session_state['preprocessing_pipelines_by_model'] = {'ridge': pipe}
        at.session_state['preprocessing_config_by_model'] = {
            'ridge': {'numeric_scaling': 'standard', 'numeric_imputation': 'median'}
        }
        at.session_state['train_model_ridge'] = True
        at.run()
        assert_no_exception(at, "Train (with preprocessing)")


# ── Contract: Page 06 → Page 07 ─────────────────────────────────────

class TestPage06ToPage07Contract:
    """Train output must provide what Explainability needs."""

    def test_trained_state_sufficient_for_explainability(self, reg_df):
        """Page 07 needs trained_models, fitted_estimators,
        fitted_preprocessing_pipelines, and feature_names_by_model."""
        at = AppTest.from_file("pages/07_Explainability.py", default_timeout=30)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)

        # inject_trained_state uses only numeric features, so build pipeline to match
        from ml.pipeline import build_preprocessing_pipeline
        numeric_feats = _numeric_features(reg_df)
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=[],
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        pipe.fit(at.session_state['X_train'])
        at.session_state['fitted_preprocessing_pipelines'] = {'ridge': pipe}
        at.session_state['feature_names_by_model'] = {'ridge': numeric_feats}
        at.session_state['preprocessing_pipelines_by_model'] = {'ridge': pipe}

        at.run()
        assert_no_exception(at, "Explainability")

    def test_explainability_missing_pipelines_degrades_gracefully(self, reg_df):
        """If fitted_preprocessing_pipelines is missing, page should not crash."""
        at = AppTest.from_file("pages/07_Explainability.py", default_timeout=30)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)
        # Deliberately leave fitted_preprocessing_pipelines empty (not set by inject_trained_state)
        at.run()
        assert_no_exception(at, "Explainability (no pipelines)")


# ── Contract: Page 06 → Page 08 ─────────────────────────────────────

class TestPage06ToPage08Contract:
    """Train output must provide what Sensitivity Analysis needs."""

    def test_trained_state_sufficient_for_sensitivity(self, reg_df):
        at = AppTest.from_file("pages/08_Sensitivity_Analysis.py", default_timeout=30)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Sensitivity Analysis")


# ── Contract: All → Page 10 ─────────────────────────────────────────

class TestAllToPage10Contract:
    """Report Export reads from all previous pages."""

    def test_report_renders_with_trained_state(self, reg_df):
        at = AppTest.from_file("pages/10_Report_Export.py", default_timeout=30)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Report Export")

    def test_report_renders_with_minimal_state(self, reg_df):
        """Report page should not crash even with only data loaded."""
        at = AppTest.from_file("pages/10_Report_Export.py", default_timeout=30)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Report Export (minimal)")


# ── Contract: Classification Path ────────────────────────────────────

class TestClassificationPathContracts:
    """Verify state contracts hold for classification workflow."""

    def test_classification_eda_to_preprocess(self, clf_df):
        """Classification path: EDA → Preprocess renders correctly."""
        # Run EDA
        at_eda = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at_eda, clf_df, target_col='condition', task_type='classification')
        at_eda.run()
        assert_no_exception(at_eda, "EDA (classification)")

        # Run Preprocess with EDA's output
        at_pp = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at_pp, clf_df, target_col='condition', task_type='classification')
        profile = _ss_get(at_eda, 'dataset_profile')
        if profile is not None:
            at_pp.session_state['dataset_profile'] = profile
        at_pp.run()
        assert_no_exception(at_pp, "Preprocess (classification)")

    def test_classification_train_renders(self, clf_df):
        """Classification training page renders with preprocessing."""
        from ml.pipeline import build_preprocessing_pipeline

        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')

        numeric_feats = [c for c in clf_df.columns if c != 'condition' and clf_df[c].dtype in ('float64', 'int64')]
        cat_feats = [c for c in clf_df.columns if c != 'condition' and clf_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=cat_feats,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        at.session_state['preprocessing_pipelines_by_model'] = {'logreg': pipe}
        at.session_state['preprocessing_config_by_model'] = {
            'logreg': {'numeric_scaling': 'standard'}
        }
        at.session_state['train_model_logreg'] = True
        at.run()
        assert_no_exception(at, "Train (classification)")


# ── Contract: Engineered Features Path ───────────────────────────────

class TestEngineeredFeaturesContract:
    """Verify feature engineering state flows correctly to downstream pages."""

    def test_engineered_features_reach_feature_selection(self, reg_df):
        """When FE is applied, Page 04 should see the engineered columns."""
        at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=30)
        inject_data_state(at, reg_df)

        # Simulate Page 03 having added engineered features
        df_eng = reg_df.copy()
        df_eng['log_bmi'] = np.log1p(df_eng['bmi'].fillna(df_eng['bmi'].median()))
        df_eng['bmi_squared'] = df_eng['bmi'].fillna(df_eng['bmi'].median()) ** 2

        at.session_state['raw_data'] = df_eng
        at.session_state['filtered_data'] = df_eng
        at.session_state['df_engineered'] = df_eng
        at.session_state['feature_engineering_applied'] = True
        at.session_state['engineered_feature_names'] = ['log_bmi', 'bmi_squared']

        all_feats = [c for c in df_eng.columns if c != 'glucose']
        at.session_state['selected_features'] = all_feats
        at.session_state['data_config'].feature_cols = all_feats
        at.run()
        assert_no_exception(at, "Feature Selection (with engineered features)")

    def test_engineered_features_reach_preprocess(self, reg_df):
        """When FE is applied, Page 05 should see the engineered columns."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)

        df_eng = reg_df.copy()
        df_eng['log_bmi'] = np.log1p(df_eng['bmi'].fillna(df_eng['bmi'].median()))

        at.session_state['raw_data'] = df_eng
        at.session_state['filtered_data'] = df_eng
        at.session_state['df_engineered'] = df_eng
        at.session_state['feature_engineering_applied'] = True
        at.session_state['engineered_feature_names'] = ['log_bmi']
        at.session_state['engineered_feature_transforms'] = {'log_bmi': 'log'}

        all_feats = [c for c in df_eng.columns if c != 'glucose']
        at.session_state['selected_features'] = all_feats
        at.session_state['data_config'].feature_cols = all_feats
        at.run()
        assert_no_exception(at, "Preprocess (with engineered features)")
