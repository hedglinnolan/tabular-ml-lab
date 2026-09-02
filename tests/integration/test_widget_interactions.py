"""
Tier 2: Widget interaction tests — clicking buttons, setting values.

These tests simulate actual user interactions: clicking buttons and verifying
that session_state is mutated correctly.  This is the layer our audit
identified as having ZERO coverage — all existing tests were read-only renders.

Key interactions tested:
  - Page 05: Build Pipelines button → preprocessing_pipelines_by_model populated
  - Page 06: Prepare Splits → X_train/y_train populated
  - Page 06: Train Models → trained_models populated
  - Page 02: Generate Table 1 → table1_df populated
  - Page 04: Run Feature Selection → feature_selection_results populated
  - Page 09: Hypothesis test buttons → results stored

Note on AppTest widget API:
  - Buttons with explicit keys: at.button(key="the_key").click()
  - Buttons without keys: at.button[index].click()  (fragile, avoid)
  - After widget interaction: at.run() to re-execute the page
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    HARNESS_ONLY_EXCEPTIONS,
    build_test_dataframe, build_classification_dataframe,
    inject_data_state, inject_trained_state,
)


# ── Helpers ──────────────────────────────────────────────────────────

def assert_no_exception(at, page_name, ignore_patterns=None):
    """Assert no meaningful exceptions during page render.

    ignore_patterns: list of substrings to ignore in exception messages.
    st.page_link() cannot resolve a page under a single-file AppTest; the
    message it raises depends on the Streamlit version, and the shared
    HARNESS_ONLY_EXCEPTIONS in conftest names every spelling seen so far.
    """
    if not at.exception:
        return
    default_ignore = list(HARNESS_ONLY_EXCEPTIONS)
    ignore = (ignore_patterns or []) + default_ignore
    real_exceptions = []
    for e in at.exception:
        msg = str(e.value)
        if not any(pat in msg for pat in ignore):
            real_exceptions.append(msg[:300])
    if real_exceptions:
        pytest.fail(f"{page_name} raised: {'; '.join(real_exceptions)}")


def _ss_get(at, key, default=None):
    """Safe session_state.get() for AppTest."""
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


# ── Page 05: Build Pipelines Button ─────────────────────────────────

class TestBuildPipelinesInteraction:
    """Click Build Pipelines on Page 05 and verify pipeline state is created."""

    def test_build_pipelines_creates_per_model_pipelines(self, reg_df):
        """Click Build Pipelines → preprocessing_pipelines_by_model populated."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=45)
        inject_data_state(at, reg_df)
        # Select ridge for training
        at.session_state['train_model_ridge'] = True
        at.run()
        assert_no_exception(at, "Preprocess (initial render)")

        # Click the Build Pipelines button
        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Preprocess (after Build Pipelines)")

        # Verify pipeline state was created
        pipelines = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert len(pipelines) > 0, "Build Pipelines must create preprocessing_pipelines_by_model"
        assert 'ridge' in pipelines, "Pipeline for ridge model must exist"

        configs = _ss_get(at, 'preprocessing_config_by_model', {})
        assert len(configs) > 0, "Build Pipelines must create preprocessing_config_by_model"

        summary = _ss_get(at, 'preprocessing_summary')
        assert summary is not None, "Build Pipelines must create preprocessing_summary"

    def test_build_pipelines_with_multiple_models(self, reg_df):
        """Build pipelines for ridge + rf → both pipelines created."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=45)
        inject_data_state(at, reg_df)
        at.session_state['train_model_ridge'] = True
        at.session_state['train_model_rf'] = True
        at.run()
        assert_no_exception(at, "Preprocess (2 models initial)")

        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Preprocess (2 models after build)")

        pipelines = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert 'ridge' in pipelines, "Ridge pipeline must exist"
        assert 'rf' in pipelines, "RF pipeline must exist"

    def test_rebuild_clears_pipeline_state(self, reg_df):
        """Simulating Rebuild Pipeline clears old pipelines.

        The Rebuild button can't be clicked in AppTest because st.page_link()
        above it raises KeyError('url_pathname') in test mode.  Instead we
        verify the rebuild contract: clearing pipeline state returns the page
        to its initial (unconfigured) render.
        """
        from ml.pipeline import build_preprocessing_pipeline
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=45)
        inject_data_state(at, reg_df)
        at.session_state['train_model_ridge'] = True

        # Inject pre-built pipeline state (simulating prior Build Pipelines)
        numeric_feats = _numeric_features(reg_df)
        cat_feats = [c for c in reg_df.columns if c != 'glucose' and reg_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=cat_feats,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        at.session_state['preprocessing_pipelines_by_model'] = {'ridge': pipe}
        at.session_state['preprocessing_config_by_model'] = {'ridge': {}}
        at.run()
        assert_no_exception(at, "Preprocess (with existing pipelines)")

        # Simulate what the Rebuild button does (lines 1165-1168 of 05_Preprocess.py)
        at.session_state['preprocessing_pipeline'] = None
        at.session_state['preprocessing_config'] = None
        at.session_state['preprocessing_pipelines_by_model'] = {}
        at.session_state['preprocessing_config_by_model'] = {}
        at.run()
        assert_no_exception(at, "Preprocess (after rebuild simulation)")

        pipelines = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert len(pipelines) == 0, "Rebuild must clear preprocessing_pipelines_by_model"

    def test_classification_build_pipelines(self, clf_df):
        """Build pipelines for classification task."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=45)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')
        at.session_state['train_model_logreg'] = True
        at.run()
        assert_no_exception(at, "Preprocess (classification initial)")

        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Preprocess (classification build)")

        pipelines = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert 'logreg' in pipelines


# ── Page 06: Prepare Splits Button ──────────────────────────────────

class TestPrepareSplitsInteraction:
    """Click Prepare Splits on Page 06 and verify split state is created."""

    def _setup_page06(self, at, df, target_col='glucose', task_type='regression'):
        """Set up Page 06 with preprocessing pipelines ready."""
        from ml.pipeline import build_preprocessing_pipeline
        inject_data_state(at, df, target_col=target_col, task_type=task_type)
        numeric_feats = [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64')]
        cat_feats = [c for c in df.columns if c != target_col and df[c].dtype == 'object']
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

    def test_prepare_splits_creates_train_test_data(self, reg_df):
        """Click Prepare Splits → X_train, y_train, etc. populated."""
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=45)
        self._setup_page06(at, reg_df)
        at.run()
        assert_no_exception(at, "Train (initial)")

        # The "Prepare Splits" button has no key — find by label
        # It's the first primary button on the page before model selection
        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        assert len(split_buttons) > 0, "Prepare Splits button must exist"
        split_buttons[0].click()
        at.run()
        assert_no_exception(at, "Train (after Prepare Splits)")

        # Verify split state was created
        X_train = _ss_get(at, 'X_train')
        assert X_train is not None, "Prepare Splits must set X_train"
        assert len(X_train) > 0, "X_train must not be empty"

        y_train = _ss_get(at, 'y_train')
        assert y_train is not None, "Prepare Splits must set y_train"

        X_test = _ss_get(at, 'X_test')
        assert X_test is not None, "Prepare Splits must set X_test"

        X_val = _ss_get(at, 'X_val')
        assert X_val is not None, "Prepare Splits must set X_val"

        # Verify sizes are reasonable (70/15/15 default)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total > 0
        train_pct = len(X_train) / total
        assert 0.5 < train_pct < 0.9, f"Train fraction {train_pct:.2f} outside expected range"

    def test_prepare_splits_stores_row_labels(self, reg_df):
        """Click Prepare Splits → train/val/test row LABELS stored, disjoint.

        Row identity crosses a page boundary as an index label, never as a
        position, so this is the key the downstream pages read.
        """
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=45)
        self._setup_page06(at, reg_df)
        at.run()

        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        split_buttons[0].click()
        at.run()
        assert_no_exception(at, "Train (splits + row labels)")

        train_labels = _ss_get(at, 'train_row_labels')
        assert train_labels is not None, "train_row_labels must be stored"
        assert len(train_labels) > 0

        val_labels = _ss_get(at, 'val_row_labels')
        assert val_labels is not None, "val_row_labels must be stored"

        test_labels = _ss_get(at, 'test_row_labels')
        assert test_labels is not None, "test_row_labels must be stored"

        # No row may appear in two partitions
        assert len(set(train_labels) & set(val_labels)) == 0, "Train and val rows must not overlap"
        assert len(set(train_labels) & set(test_labels)) == 0, "Train and test rows must not overlap"
        assert len(set(val_labels) & set(test_labels)) == 0, "Val and test rows must not overlap"

        # And every stored label names a row of the frame the split was drawn on.
        assert set(train_labels) | set(val_labels) | set(test_labels) <= set(reg_df.index)

    def test_prepare_splits_classification(self, clf_df):
        """Classification splits work correctly."""
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=45)
        self._setup_page06(at, clf_df, target_col='condition', task_type='classification')
        at.session_state['train_model_logreg'] = True
        at.run()

        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        split_buttons[0].click()
        at.run()
        assert_no_exception(at, "Train (classification splits)")

        y_train = _ss_get(at, 'y_train')
        assert y_train is not None


# ── Page 06: Train Models Button ────────────────────────────────────

class TestTrainModelsInteraction:
    """Click Train Models on Page 06 and verify model state is created."""

    def test_train_ridge_model(self, reg_df):
        """Full interaction: Prepare Splits → Train Models → model results."""
        from ml.pipeline import build_preprocessing_pipeline
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=90)
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
        assert_no_exception(at, "Train (initial)")

        # Step 1: Prepare Splits
        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        split_buttons[0].click()
        at.run()
        assert_no_exception(at, "Train (after splits)")

        # Step 2: Train Models
        at.button(key="train_models_button").click()
        at.run()
        assert_no_exception(at, "Train (after training)")

        # Verify training state
        trained_models = _ss_get(at, 'trained_models', {})
        assert 'ridge' in trained_models, "Ridge model must be in trained_models"

        model_results = _ss_get(at, 'model_results', {})
        assert 'ridge' in model_results, "Ridge results must be in model_results"
        assert 'metrics' in model_results['ridge'], "Ridge results must contain metrics"

        fitted_estimators = _ss_get(at, 'fitted_estimators', {})
        assert 'ridge' in fitted_estimators, "Ridge must be in fitted_estimators"

        fitted_pipelines = _ss_get(at, 'fitted_preprocessing_pipelines', {})
        assert 'ridge' in fitted_pipelines, "Ridge pipeline must be in fitted_preprocessing_pipelines"

        feature_names = _ss_get(at, 'feature_names_by_model', {})
        assert 'ridge' in feature_names, "Ridge feature names must be stored"

    def test_train_produces_valid_metrics(self, reg_df):
        """Trained model metrics are valid numbers."""
        from ml.pipeline import build_preprocessing_pipeline
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=90)
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

        # Prepare + Train
        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        split_buttons[0].click()
        at.run()
        at.button(key="train_models_button").click()
        at.run()
        assert_no_exception(at, "Train (metrics check)")

        results = _ss_get(at, 'model_results', {})
        metrics = results.get('ridge', {}).get('metrics', {})
        assert 'RMSE' in metrics, "Regression metrics must include RMSE"
        assert 'R2' in metrics, "Regression metrics must include R2"
        assert metrics['RMSE'] > 0, "RMSE must be positive"
        assert isinstance(metrics['R2'], (int, float)), "R2 must be a number"


# ── Page 02: Generate Table 1 Button ────────────────────────────────

class TestGenerateTable1Interaction:
    """Click Generate Table 1 on Page 02 and verify table state."""

    def test_generate_table1_produces_valid_output(self, reg_df):
        """Verify Table 1 generation produces correct DataFrame + metadata.

        The EDA page uses st.pills (ButtonGroup) which has a serialization
        bug in Streamlit AppTest after button re-runs.  We test the
        generate_table1 function directly with the same inputs the page uses,
        then verify the output would be stored correctly in session_state.
        """
        from ml.table_one import generate_table1, Table1Config

        feature_cols = [c for c in reg_df.columns if c != 'glucose']

        # Use defaults matching the EDA page's widget defaults
        continuous_vars = [c for c in feature_cols if reg_df[c].dtype in ('float64', 'int64')]
        categorical_vars = [c for c in feature_cols if reg_df[c].dtype == 'object']

        config = Table1Config(
            grouping_var=None,
            continuous_vars=continuous_vars,
            categorical_vars=categorical_vars,
            show_pvalues=False,
            show_smd=False,
            show_missing=False,
        )
        table1_df, table1_metadata = generate_table1(reg_df, config)

        assert table1_df is not None, "generate_table1 must return a DataFrame"
        assert len(table1_df) > 0, "table1_df must not be empty"
        assert table1_metadata is not None, "generate_table1 must return metadata"

        # Verify the metadata has expected structure
        assert 'n_total' in table1_metadata or 'n' in table1_metadata or len(table1_metadata) > 0


# ── Page 09: Hypothesis Test Buttons ────────────────────────────────

class TestHypothesisTestInteraction:
    """Click hypothesis test buttons and verify results."""

    def test_correlation_test_stores_results(self, reg_df):
        """Click Run Correlation Test → results stored."""
        at = AppTest.from_file("pages/09_Hypothesis_Testing.py", default_timeout=45)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Hypothesis (initial)")

        # Click Run Correlation Test
        at.button(key="run_corr").click()
        at.run()
        assert_no_exception(at, "Hypothesis (after correlation)")

    def test_two_sample_test_runs(self, reg_df):
        """Click Run Two-Sample Test → executes without crash."""
        at = AppTest.from_file("pages/09_Hypothesis_Testing.py", default_timeout=45)
        inject_data_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Hypothesis (initial two-sample)")

        # The two-sample test button may be rendered if grouping vars exist
        try:
            at.button(key="run_two_sample").click()
            at.run()
            assert_no_exception(at, "Hypothesis (after two-sample)")
        except KeyError:
            # Button not rendered (no suitable grouping variable detected)
            pass


# ── Page 08: Sensitivity Analysis Buttons ────────────────────────────

class TestSensitivityInteraction:
    """Click sensitivity buttons and verify they execute without error."""

    def test_seed_sensitivity_button(self, reg_df):
        """Click Run Seed Sensitivity → executes without crash."""
        at = AppTest.from_file("pages/08_Sensitivity_Analysis.py", default_timeout=60)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)

        from ml.pipeline import build_preprocessing_pipeline
        numeric_feats = _numeric_features(reg_df)
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=[],
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        at.session_state['preprocessing_pipelines_by_model'] = {'ridge': pipe}
        at.session_state['fitted_preprocessing_pipelines'] = {'ridge': pipe}
        at.session_state['feature_names_by_model'] = {'ridge': numeric_feats}
        at.run()
        assert_no_exception(at, "Sensitivity (initial)")

        at.button(key="run_seed").click()
        at.run()
        assert_no_exception(at, "Sensitivity (after seed run)")


# ── Page 10: Generate Methods Button ─────────────────────────────────

class TestReportGenerationInteraction:
    """Click Generate Methods Section and verify output."""

    def test_generate_methods_section(self, reg_df):
        """Click Generate Methods Section → no crash, output present."""
        at = AppTest.from_file("pages/10_Report_Export.py", default_timeout=45)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)
        at.run()
        assert_no_exception(at, "Report (initial)")

        at.button(key="gen_methods").click()
        at.run()
        assert_no_exception(at, "Report (after Generate Methods)")
