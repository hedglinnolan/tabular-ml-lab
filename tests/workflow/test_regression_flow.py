"""
Workflow test: Full regression pipeline.
Upload → Configure → EDA → Feature Selection → Preprocess → Train → Explain → Export

Verifies that each step's output is consumed correctly by downstream steps,
and that the final export reflects all decisions made along the way.

Uses module-scoped fixtures from conftest.py so that state mutations in
earlier tests are visible to later ones (simulating a real user session).
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import (
    build_regression_df, inject_uploaded_state, prepare_splits,
    train_ridge_model, make_data_config,
)


class TestRegressionPipeline:
    """Sequential regression pipeline — tests run in definition order on shared state."""

    def test_step1_upload_and_configure(self, regression_df, regression_state):
        """Step 1: Verify upload & configure produced valid state."""
        df = regression_df
        state = regression_state

        assert state.get('raw_data') is not None, "raw_data missing"
        assert state.get('data_config') is not None, "data_config missing"

        data_config = state['data_config']
        assert data_config.target_col == 'glucose'
        assert data_config.task_type == 'regression'
        assert len(data_config.feature_cols) > 0
        assert 'glucose' not in data_config.feature_cols, "target in features"

        assert state.get('_data_config_features_hash'), "Feature hash not set"
        assert state.get('dataset_profile') is not None, "Profile not computed"

    def test_step2_eda_signals(self, regression_df, regression_state):
        """Step 2: Verify EDA signals are available from the profile."""
        df = regression_df
        state = regression_state

        profile = state['dataset_profile']
        assert hasattr(profile, 'signals') or hasattr(profile, 'n_rows'), \
            "Profile structure unexpected"

        from ml.eda_recommender import DatasetSignals
        data_config = state['data_config']
        feature_cols = data_config.feature_cols
        numeric_cols = [c for c in feature_cols if df[c].dtype in ('float64', 'int64')]

        signals = DatasetSignals(n_rows=len(df), n_cols=len(feature_cols), numeric_cols=numeric_cols)
        assert signals.n_rows == 200

    def test_step3_prepare_splits(self, regression_df, regression_state):
        """Step 3: Prepare train/val/test splits and store in shared state."""
        df = regression_df
        state = regression_state

        splits = prepare_splits(df, target_col='glucose')
        for k, v in splits.items():
            state[k] = v

        total = len(state['X_train']) + len(state['X_val']) + len(state['X_test'])
        assert total > 0, "No data after split"
        assert len(state['X_train']) > len(state['X_test']), "Train should be larger than test"
        assert 'glucose' not in state['X_train'].columns, "Target leaked into features"
        assert len(state['test_indices']) == len(state['X_test'])
        assert len(state['y_train']) == len(state['X_train'])
        assert len(state['y_test']) == len(state['X_test'])

    def test_step4_train_model(self, regression_state):
        """Step 4: Train a model and verify metrics."""
        state = regression_state
        assert state.get('X_train') is not None, "Step 3 must run first (splits missing)"

        splits = {k: state[k] for k in [
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
            'train_indices', 'val_indices', 'test_indices', 'feature_names',
            'split_config',
        ] if k in state}

        result = train_ridge_model(splits)
        model = result['model']
        metrics = result['metrics']

        assert hasattr(model, 'coef_'), "Model not fitted"
        assert len(result['y_test_pred']) == len(state['y_test'])

        # R² should be positive for this synthetic data
        r2 = None
        for k, v in metrics.items():
            if 'r2' in k.lower() or 'r²' in k.lower():
                r2 = v
                break
        assert r2 is not None and r2 > 0, f"R² should be positive, got {r2}"

        # RMSE sanity
        rmse = metrics.get('rmse', metrics.get('RMSE', None))
        if rmse is not None:
            assert 0 < rmse < 100, f"RMSE unreasonable: {rmse}"

        # Store in shared state
        state['trained_models'] = {'ridge': model}
        state['model_results'] = {
            'ridge': {
                'metrics': metrics,
                'y_test': state['y_test'].values,
                'y_test_pred': result['y_test_pred'],
            }
        }

    def test_step5_explainability_compatibility(self, regression_df, regression_state):
        """Step 5: Subgroup analysis can access raw columns via test_indices (#32 fix)."""
        df = regression_df
        state = regression_state

        test_indices = state.get('test_indices')
        assert test_indices is not None, "No test indices"

        raw_df = state['raw_data']
        X_test = state['X_test']

        # Gender shouldn't be in numeric features but should be accessible from raw df
        assert 'gender' not in X_test.columns, "Gender shouldn't be in numeric features"
        subgroup_labels = raw_df.iloc[test_indices]['gender'].values
        assert len(subgroup_labels) == len(X_test), "Subgroup labels length mismatch"
        assert set(subgroup_labels).issubset({'male', 'female'})

        smoking_labels = raw_df.iloc[test_indices]['smoking'].values
        assert len(smoking_labels) == len(X_test)

    def test_step6_publication_output(self, regression_state):
        """Step 6: Publication methods section is complete."""
        state = regression_state

        from ml.publication import generate_methods_section

        data_config = state['data_config']
        sc_dict = {
            'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
            'target_trim_enabled': False,
            'target_transform': 'none',
            'stratify': False, 'use_time_split': False,
        }

        methods = generate_methods_section(
            data_config={
                'feature_cols': data_config.feature_cols,
                'target_col': 'glucose',
                'task_type': 'regression',
            },
            preprocessing_config={},
            model_configs={},
            split_config=sc_dict,
            n_total=200,
            n_train=len(state['X_train']),
            n_val=len(state['X_val']),
            n_test=len(state['X_test']),
            feature_names=list(state['X_train'].columns),
            target_name='glucose',
            task_type='regression',
            metrics_used=['RMSE', 'R²'],
        )

        assert 'glucose' in methods, "Target not mentioned in methods"
        assert 'training' in methods.lower() or 'split' in methods.lower()
        assert len(methods) > 200, f"Methods section too short ({len(methods)} chars)"
