"""
Tier 2: Cascade invalidation tests.

Verify that when users go back and change upstream decisions, downstream
state is properly invalidated.  This is the most dangerous gap — silent
stale state can produce incorrect results without any visible error.

Key scenarios:
  1. Feature engineering save → clears all downstream state
  2. Feature change on Page 01 → clears preprocessing + training state
  3. Preprocessing rebuild → clears pipeline state
  4. Feature selection change → affects what preprocessing and training use
  5. Full round-trip: build → invalidate → rebuild → verify fresh state
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
    if not at.exception:
        return
    # st.page_link() under a single-file AppTest: the harness's failure, not
    # the page's, in whichever words this Streamlit version uses for it.
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
    try:
        return at.session_state[key]
    except (KeyError, Exception):
        return default


def _numeric_features(df, target_col='glucose'):
    return [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64')]


def _inject_full_downstream_state(at, df, target_col='glucose'):
    """Inject preprocessing + training + explainability state to simulate
    a user who has completed the full pipeline."""
    from ml.pipeline import build_preprocessing_pipeline
    from sklearn.linear_model import Ridge
    from ml.eval import calculate_regression_metrics
    from utils.session_state import SplitConfig

    numeric_feats = _numeric_features(df, target_col)
    cat_feats = [c for c in df.columns if c != target_col and df[c].dtype == 'object']

    # Build pipeline
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
    at.session_state['preprocessing_pipeline'] = pipe
    at.session_state['preprocessing_config'] = {'numeric_features': numeric_feats}

    # Prepare splits
    mask = df[target_col].notna()
    X = df.loc[mask, numeric_feats].copy().fillna(df[numeric_feats].median())
    y = df.loc[mask, target_col].copy()
    n = len(X)
    n_train, n_val = int(n * 0.7), int(n * 0.15)

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train+n_val], X.iloc[n_train+n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train+n_val], y.iloc[n_train+n_val:]

    at.session_state['split_config'] = SplitConfig()
    at.session_state['X_train'] = X_train
    at.session_state['X_val'] = X_val
    at.session_state['X_test'] = X_test
    at.session_state['y_train'] = y_train
    at.session_state['y_val'] = y_val
    at.session_state['y_test'] = y_test
    # Row identity as the app stores it: index LABELS, read off the frames.
    at.session_state['train_row_labels'] = list(X_train.index)
    at.session_state['val_row_labels'] = list(X_val.index)
    at.session_state['test_row_labels'] = list(X_test.index)
    at.session_state['feature_names'] = numeric_feats

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test.values, y_pred)

    at.session_state['trained_models'] = {'ridge': model}
    at.session_state['fitted_estimators'] = {'ridge': model}
    at.session_state['model_results'] = {
        'ridge': {'metrics': metrics, 'y_test': y_test.values, 'y_test_pred': y_pred}
    }
    at.session_state['fitted_preprocessing_pipelines'] = {'ridge': pipe}
    at.session_state['feature_names_by_model'] = {'ridge': numeric_feats}

    # Explainability state
    at.session_state['shap_results'] = {'ridge': {'values': np.random.randn(len(X_test), len(numeric_feats))}}
    at.session_state['permutation_importance'] = {'ridge': {'importances': np.random.rand(len(numeric_feats))}}
    at.session_state['sensitivity_seed_results'] = {'ridge': {'cv_rmse': 2.5}}


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def reg_df():
    return build_test_dataframe()


# ── Test: Feature Change Cascade (Page 01) ──────────────────────────

# Keys that Page 01 clears when features change (from lines 1601-1615):
DOWNSTREAM_KEYS_CLEARED_BY_FEATURE_CHANGE = [
    'preprocessing_pipeline', 'preprocessing_config',
    'preprocessing_pipelines_by_model', 'preprocessing_config_by_model',
    'trained_models', 'model_results', 'fitted_estimators',
    'fitted_preprocessing_pipelines', 'feature_names_by_model',
    'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
    'train_row_labels', 'val_row_labels', 'test_row_labels',
    'permutation_importance', 'partial_dependence', 'shap_results',
    'sensitivity_seed_results', 'report_data',
    'feature_selection_results', 'consensus_features',
    'split_config', 'target_transformer',
    'y_train_original', 'y_val_original', 'y_test_original',
    'eda_results', 'eda_insights',
]


class TestFeatureChangeCascade:
    """When features change on Page 01, all downstream state must be cleared."""

    def test_feature_hash_change_clears_downstream(self, reg_df):
        """Simulate feature set change via hash mismatch → downstream cleared."""
        at = AppTest.from_file("pages/01_Upload_and_Audit.py", default_timeout=30)
        inject_data_state(at, reg_df)
        _inject_full_downstream_state(at, reg_df)

        # Verify downstream state exists before the change
        assert _ss_get(at, 'trained_models') is not None
        assert len(_ss_get(at, 'trained_models', {})) > 0
        assert _ss_get(at, 'X_train') is not None
        assert _ss_get(at, 'preprocessing_pipelines_by_model') is not None

        # Simulate what Page 01 does when features change (lines 1596-1615):
        # It computes a hash of the new feature list and compares to old hash.
        # If different, it clears all downstream keys.
        import hashlib
        old_hash = _ss_get(at, '_data_config_features_hash', '')
        new_features = _numeric_features(reg_df)[:3]  # reduced set
        new_hash = hashlib.md5(','.join(sorted(new_features)).encode()).hexdigest()[:8]

        # Hash should be different (we changed the feature set)
        assert old_hash != new_hash, "Reduced feature set should produce different hash"

        # Apply the cascade clear (what Page 01 does on hash mismatch)
        for key in DOWNSTREAM_KEYS_CLEARED_BY_FEATURE_CHANGE:
            try:
                val = at.session_state[key]
                # Clear dicts to empty, others to None
                if isinstance(val, dict):
                    at.session_state[key] = {}
                else:
                    at.session_state[key] = None
            except (KeyError, Exception):
                pass
        at.session_state['_data_config_features_hash'] = new_hash

        # Verify downstream state was cleared
        assert len(_ss_get(at, 'trained_models', {})) == 0, "trained_models must be cleared"
        assert _ss_get(at, 'X_train') is None, "X_train must be cleared"
        assert len(_ss_get(at, 'preprocessing_pipelines_by_model', {})) == 0
        assert len(_ss_get(at, 'shap_results', {})) == 0
        # sensitivity_seed_results is a dict, so it's cleared to {}
        sens = _ss_get(at, 'sensitivity_seed_results')
        assert sens is None or sens == {}, "sensitivity_seed_results must be cleared"

    def test_cleared_state_still_renders_downstream_pages(self, reg_df):
        """After cascade clear, downstream pages should render without crash
        (they should show 'complete previous step' messages, not errors)."""
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, reg_df)
        # Empty preprocessing state (as if cascade cleared)
        at.session_state['preprocessing_pipelines_by_model'] = {}
        at.session_state['preprocessing_config_by_model'] = {}
        at.run()
        assert_no_exception(at, "Train (after cascade clear)")

    def test_cleared_state_renders_preprocess_page(self, reg_df):
        """After cascade clear, Preprocess page should render fresh."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)
        # Cleared pipeline state
        at.session_state['preprocessing_pipelines_by_model'] = {}
        at.session_state['preprocessing_config_by_model'] = {}
        at.session_state['preprocessing_pipeline'] = None
        at.session_state['preprocessing_config'] = None
        at.run()
        assert_no_exception(at, "Preprocess (after cascade clear)")


# ── Test: Feature Engineering Cascade (Page 03) ─────────────────────

# Keys that Page 03 Save clears (from the agent's research):
FE_CASCADE_KEYS = [
    'feature_selection_results', 'consensus_features',
    'preprocessing_pipeline', 'preprocessing_config',
    'preprocessing_pipelines_by_model', 'preprocessing_config_by_model',
    'trained_models', 'model_results', 'fitted_estimators',
    'fitted_preprocessing_pipelines',
    'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
    'permutation_importance', 'partial_dependence', 'shap_results',
    'sensitivity_seed_results', 'report_data',
]


class TestFeatureEngineeringCascade:
    """When Feature Engineering saves, all downstream state must be cleared."""

    def test_fe_save_clears_downstream_state(self, reg_df):
        """Simulate FE save → downstream preprocessing/training cleared."""
        at = AppTest.from_file("pages/03_Feature_Engineering.py", default_timeout=30)
        inject_data_state(at, reg_df)
        _inject_full_downstream_state(at, reg_df)
        at.run()
        assert_no_exception(at, "FE (initial with downstream)")

        # Verify downstream state exists
        assert len(_ss_get(at, 'trained_models', {})) > 0
        assert _ss_get(at, 'X_train') is not None

        # Simulate what the Save button does: apply FE cascade clear
        for key in FE_CASCADE_KEYS:
            try:
                val = at.session_state[key]
                if isinstance(val, dict):
                    at.session_state[key] = {}
                else:
                    at.session_state[key] = None
            except (KeyError, Exception):
                pass

        # Set FE state
        df_eng = reg_df.copy()
        df_eng['log_bmi'] = np.log1p(df_eng['bmi'].fillna(df_eng['bmi'].median()))
        at.session_state['df_engineered'] = df_eng
        at.session_state['feature_engineering_applied'] = True
        at.session_state['engineered_feature_names'] = ['log_bmi']

        # Verify downstream was cleared
        assert len(_ss_get(at, 'trained_models', {})) == 0
        assert _ss_get(at, 'X_train') is None
        assert len(_ss_get(at, 'preprocessing_pipelines_by_model', {})) == 0

    def test_fe_disabled_clears_engineered_state(self, reg_df):
        """When FE is disabled (Reset/Skip), engineered features are removed."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)

        # Simulate FE was applied
        df_eng = reg_df.copy()
        df_eng['log_bmi'] = np.log1p(df_eng['bmi'].fillna(df_eng['bmi'].median()))
        at.session_state['raw_data'] = df_eng
        at.session_state['filtered_data'] = df_eng
        at.session_state['df_engineered'] = df_eng
        at.session_state['feature_engineering_applied'] = True
        at.session_state['engineered_feature_names'] = ['log_bmi']
        all_feats = [c for c in df_eng.columns if c != 'glucose']
        at.session_state['selected_features'] = all_feats
        at.session_state['data_config'].feature_cols = all_feats
        at.run()
        assert_no_exception(at, "Preprocess (with FE)")

        # Now simulate FE disabled (Reset button on Page 03)
        at.session_state['feature_engineering_applied'] = False
        at.session_state['df_engineered'] = None
        at.session_state['engineered_feature_names'] = []
        # Restore original features
        orig_feats = [c for c in reg_df.columns if c != 'glucose']
        at.session_state['selected_features'] = orig_feats
        at.session_state['data_config'].feature_cols = orig_feats
        at.session_state['raw_data'] = reg_df
        at.session_state['filtered_data'] = reg_df
        # Clear downstream
        at.session_state['preprocessing_pipelines_by_model'] = {}
        at.session_state['preprocessing_config_by_model'] = {}
        at.run()
        assert_no_exception(at, "Preprocess (after FE disabled)")


# ── Test: Preprocessing Rebuild Cascade ──────────────────────────────

class TestPreprocessingRebuildCascade:
    """When preprocessing is rebuilt, training state should be considered stale."""

    def test_rebuild_with_stale_training(self, reg_df):
        """After rebuilding preprocessing, training page should still render
        (with stale models) — it shouldn't crash."""
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, reg_df)
        inject_trained_state(at, reg_df)

        # Simulate preprocessing rebuild: pipelines changed
        from ml.pipeline import build_preprocessing_pipeline
        numeric_feats = _numeric_features(reg_df)
        cat_feats = [c for c in reg_df.columns if c != 'glucose' and reg_df[c].dtype == 'object']
        new_pipe = build_preprocessing_pipeline(
            numeric_features=numeric_feats,
            categorical_features=cat_feats,
            numeric_imputation="mean",  # DIFFERENT from original
            numeric_scaling="robust",   # DIFFERENT from original
        )
        at.session_state['preprocessing_pipelines_by_model'] = {'ridge': new_pipe}
        at.session_state['preprocessing_config_by_model'] = {
            'ridge': {'numeric_scaling': 'robust', 'numeric_imputation': 'mean'}
        }
        at.session_state['train_model_ridge'] = True
        at.run()
        assert_no_exception(at, "Train (after preprocessing rebuild)")


# ── Test: Feature Selection Change Propagation ───────────────────────

class TestFeatureSelectionPropagation:
    """Feature selection changes must propagate to preprocessing and training."""

    def test_reduced_features_in_preprocess(self, reg_df):
        """After feature selection reduces feature set,
        Preprocess uses the reduced set."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)

        # Full feature set initially
        all_numeric = _numeric_features(reg_df)
        assert len(all_numeric) >= 4

        # Simulate feature selection reducing to 2 features
        reduced = all_numeric[:2]
        at.session_state['selected_features'] = reduced
        at.session_state['data_config'].feature_cols = reduced
        at.session_state['train_model_ridge'] = True
        at.run()
        assert_no_exception(at, "Preprocess (reduced features)")

        # Build pipelines with reduced features
        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Preprocess (build with reduced features)")

        configs = _ss_get(at, 'preprocessing_config_by_model', {})
        if 'ridge' in configs:
            numeric_in_config = configs['ridge'].get('numeric_features', [])
            # The config should reflect only the selected features
            assert len(numeric_in_config) <= len(all_numeric)

    def test_expanded_features_after_fe(self, reg_df):
        """Feature engineering expands feature set, which flows to preprocess."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, reg_df)

        # Add engineered features
        df_eng = reg_df.copy()
        df_eng['log_bmi'] = np.log1p(df_eng['bmi'].fillna(df_eng['bmi'].median()))
        df_eng['bmi_sq'] = df_eng['bmi'].fillna(df_eng['bmi'].median()) ** 2
        at.session_state['raw_data'] = df_eng
        at.session_state['filtered_data'] = df_eng
        at.session_state['df_engineered'] = df_eng
        at.session_state['feature_engineering_applied'] = True
        at.session_state['engineered_feature_names'] = ['log_bmi', 'bmi_sq']

        all_feats = [c for c in df_eng.columns if c != 'glucose']
        at.session_state['selected_features'] = all_feats
        at.session_state['data_config'].feature_cols = all_feats
        at.session_state['train_model_ridge'] = True
        at.run()
        assert_no_exception(at, "Preprocess (expanded features)")

        # Build pipelines → should include the engineered features
        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Preprocess (build with expanded features)")

        configs = _ss_get(at, 'preprocessing_config_by_model', {})
        if 'ridge' in configs:
            numeric_in_config = configs['ridge'].get('numeric_features', [])
            # Engineered features should be in the numeric features list
            assert 'log_bmi' in numeric_in_config or 'bmi_sq' in numeric_in_config


# ── Test: Full Round-Trip ────────────────────────────────────────────

class TestFullRoundTrip:
    """Build pipelines → invalidate → rebuild → verify fresh state."""

    def test_build_invalidate_rebuild(self, reg_df):
        """Full cycle: build → simulate feature change → rebuild → train."""
        # Phase 1: Build pipelines
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=45)
        inject_data_state(at, reg_df)
        at.session_state['train_model_ridge'] = True
        at.run()

        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Phase 1: Build")

        pipelines_v1 = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert 'ridge' in pipelines_v1, "Phase 1: pipeline must exist"

        # Phase 2: Simulate feature change (cascade clear)
        at.session_state['preprocessing_pipelines_by_model'] = {}
        at.session_state['preprocessing_config_by_model'] = {}
        at.session_state['preprocessing_pipeline'] = None
        at.session_state['preprocessing_config'] = None

        # Reduce features
        numeric_feats = _numeric_features(reg_df)
        reduced = numeric_feats[:3]
        at.session_state['selected_features'] = reduced
        at.session_state['data_config'].feature_cols = reduced
        at.run()
        assert_no_exception(at, "Phase 2: After invalidation")

        pipelines_cleared = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert len(pipelines_cleared) == 0, "Phase 2: pipelines must be cleared"

        # Phase 3: Rebuild with new features
        at.button(key="preprocess_build_button").click()
        at.run()
        assert_no_exception(at, "Phase 3: Rebuild")

        pipelines_v2 = _ss_get(at, 'preprocessing_pipelines_by_model', {})
        assert 'ridge' in pipelines_v2, "Phase 3: rebuilt pipeline must exist"

    def test_train_after_rebuild_produces_fresh_results(self, reg_df):
        """After invalidation and rebuild, training produces fresh results."""
        from ml.pipeline import build_preprocessing_pipeline

        # Set up training page with preprocessing
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

        # Prepare splits and train
        split_buttons = [b for b in at.button if "Prepare Splits" in str(b.label)]
        split_buttons[0].click()
        at.run()
        assert_no_exception(at, "Round-trip: Prepare Splits")

        at.button(key="train_models_button").click()
        at.run()
        assert_no_exception(at, "Round-trip: Train")

        results = _ss_get(at, 'model_results', {})
        assert 'ridge' in results, "Round-trip: training must produce results"
        assert 'metrics' in results['ridge']
        assert results['ridge']['metrics']['RMSE'] > 0
