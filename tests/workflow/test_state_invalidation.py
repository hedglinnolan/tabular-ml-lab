"""
Workflow test: State invalidation when features change mid-workflow.

Simulates the exact bug from issue #32:
1. Upload data, select features WITHOUT gender
2. Train a model
3. Go back, add gender as a feature
4. Verify all downstream state is cleared
5. Verify retraining works cleanly with new features
"""
import sys
import os
import hashlib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import (
    build_regression_df, prepare_splits, train_ridge_model,
)
from utils.session_state import DataConfig, SplitConfig


# Keys that should be cleared on feature change (from Upload & Audit)
CASCADE_KEYS = [
    'preprocessing_pipeline', 'preprocessing_config',
    'preprocessing_pipelines_by_model', 'preprocessing_config_by_model',
    'trained_models', 'model_results', 'fitted_estimators',
    'fitted_preprocessing_pipelines', 'feature_names_by_model',
    'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
    'train_indices', 'val_indices', 'test_indices',
    'permutation_importance', 'partial_dependence', 'shap_results',
    'sensitivity_seed_results', 'report_data',
    'feature_selection_results', 'consensus_features',
    'split_config', 'target_transformer',
    'y_train_original', 'y_val_original', 'y_test_original',
]


def simulate_feature_change_cascade(state, new_feature_cols):
    """Simulate the cascade invalidation that happens in Upload & Audit."""
    new_hash = hashlib.md5(','.join(sorted(new_feature_cols)).encode()).hexdigest()[:8]
    old_hash = state.get('_data_config_features_hash', '')

    changed = new_hash != old_hash
    if changed:
        state['_data_config_features_hash'] = new_hash
        for key in CASCADE_KEYS:
            state.pop(key, None)

    return changed


class TestStateInvalidation:
    """Sequential state-invalidation tests — phases build on shared mutable state."""

    def test_phase1_initial_training(self, invalidation_df, invalidation_state):
        """Phase 1: Verify initial state has trained model on numeric-only features."""
        state = invalidation_state

        assert state.get('trained_models') is not None
        assert state.get('X_train') is not None
        assert state.get('model_results') is not None
        assert 'gender' not in state['X_train'].columns

    def test_phase2_change_features(self, invalidation_df, invalidation_state):
        """Phase 2: Add gender to features — should trigger cascade clearing."""
        state = invalidation_state

        # Verify pre-change state is populated
        assert state.get('trained_models') is not None
        assert state.get('X_train') is not None
        assert state.get('shap_results') is not None

        new_features = ['age', 'bmi', 'cholesterol', 'blood_pressure', 'exercise_hours', 'gender']
        changed = simulate_feature_change_cascade(state, new_features)
        assert changed, "Feature change should be detected"

        # Verify ALL downstream state is cleared
        surviving_keys = [key for key in CASCADE_KEYS if state.get(key) is not None]
        assert not surviving_keys, f"Stale state found: {surviving_keys}"

        # Critical keys specifically
        assert state.get('trained_models') is None, "Models should be cleared"
        assert state.get('X_train') is None, "Splits should be cleared"
        assert state.get('shap_results') is None, "SHAP should be cleared"
        assert state.get('model_results') is None, "Results should be cleared"

        # Non-cascade state should survive
        assert state.get('raw_data') is not None, "Raw data should survive"
        assert state.get('_data_config_features_hash') is not None, "Hash should be updated"

    def test_phase3_retrain_with_new_features(self, invalidation_df, invalidation_state):
        """Phase 3: Retrain with new feature set — should work cleanly."""
        df = invalidation_df
        state = invalidation_state

        new_features = ['age', 'bmi', 'cholesterol', 'blood_pressure', 'exercise_hours', 'gender']
        state['data_config'] = DataConfig(
            target_col='glucose',
            feature_cols=new_features,
            task_type='regression',
        )

        splits = prepare_splits(df, target_col='glucose')
        for k, v in splits.items():
            state[k] = v

        result = train_ridge_model(splits)
        state['trained_models'] = {'ridge': result['model']}
        state['model_results'] = {
            'ridge': {
                'metrics': result['metrics'],
                'y_test': splits['y_test'].values,
                'y_test_pred': result['y_test_pred'],
            }
        }

        # Verify metrics are reasonable
        for k, v in result['metrics'].items():
            if 'r2' in k.lower() or 'r²' in k.lower():
                assert v > 0, f"R² should be positive, got {v}"
                break

    def test_phase4_no_false_cascade(self):
        """Phase 4: Re-saving with SAME features should NOT cascade (standalone)."""
        state = {}
        features = ['age', 'bmi', 'cholesterol']

        state['_data_config_features_hash'] = hashlib.md5(
            ','.join(sorted(features)).encode()
        ).hexdigest()[:8]
        state['trained_models'] = {'ridge': 'dummy'}
        state['X_train'] = 'dummy'

        changed = simulate_feature_change_cascade(state, features)
        assert not changed, "Same features should not trigger cascade"
        assert state.get('trained_models') is not None, "Models should survive"
        assert state.get('X_train') is not None, "Splits should survive"

        # Same features in different order
        changed = simulate_feature_change_cascade(state, ['cholesterol', 'age', 'bmi'])
        assert not changed, "Reordered features should not trigger cascade"

    def test_phase5_subgroup_access(self, invalidation_df, invalidation_state):
        """Phase 5: Subgroup analysis can access all raw columns after retrain."""
        state = invalidation_state

        test_indices = state.get('test_indices')
        assert test_indices is not None, "No test indices (phase 3 must run first)"

        raw_df = state['raw_data']
        X_test = state['X_test']

        for col in ['gender', 'smoking', 'age', 'bmi']:
            values = raw_df.iloc[test_indices][col].values
            assert len(values) == len(X_test), f"{col} length mismatch"

        gender_labels = raw_df.iloc[test_indices]['gender'].values
        assert len(set(gender_labels)) >= 2, f"Expected 2 gender values, got {set(gender_labels)}"
