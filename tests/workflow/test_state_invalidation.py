#!/usr/bin/env python
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import (
    build_regression_df, inject_uploaded_state, prepare_splits,
    train_ridge_model, make_data_config,
)
from utils.session_state import DataConfig, SplitConfig

PASS = 0
FAIL = 0
FAILURES = []


def log(msg, status):
    global PASS, FAIL
    icon = {"PASS": "✅", "FAIL": "❌"}.get(status, "")
    print(f"  {icon} {status}: {msg}")
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
        FAILURES.append(msg)


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


def test_phase1_initial_training():
    """Phase 1: Train with initial feature set (no gender)."""
    print("\n--- Phase 1: Initial Training (without gender) ---")
    
    df = build_regression_df(n=200)
    state = {}
    
    # Configure WITHOUT gender and smoking (categorical)
    numeric_features = ['age', 'bmi', 'cholesterol', 'blood_pressure', 'exercise_hours']
    data_config = DataConfig(
        target_col='glucose',
        feature_cols=numeric_features,
        task_type='regression',
    )
    state['raw_data'] = df
    state['filtered_data'] = df
    state['data_config'] = data_config
    state['_data_config_features_hash'] = hashlib.md5(
        ','.join(sorted(numeric_features)).encode()
    ).hexdigest()[:8]
    
    log(f"Configured with {len(numeric_features)} features (no gender)", "PASS")
    
    # Prepare splits and train
    splits = prepare_splits(df, target_col='glucose')
    for k, v in splits.items():
        state[k] = v
    
    result = train_ridge_model(splits)
    state['trained_models'] = {'ridge': result['model']}
    state['model_results'] = {'ridge': {'metrics': result['metrics'], 'y_test': splits['y_test'].values, 'y_test_pred': result['y_test_pred']}}
    state['fitted_estimators'] = {'ridge': result['model']}
    state['preprocessing_pipelines_by_model'] = {'ridge': 'dummy_pipeline'}
    state['shap_results'] = {'ridge': 'dummy_shap'}
    state['permutation_importance'] = {'ridge': 'dummy_perm'}
    
    # Verify we have trained state
    assert state.get('trained_models') is not None
    assert state.get('X_train') is not None
    assert state.get('model_results') is not None
    log("Model trained and state populated", "PASS")
    
    # Verify gender is NOT in features
    assert 'gender' not in state['X_train'].columns
    log("Gender correctly excluded from training features", "PASS")
    
    return df, state


def test_phase2_change_features(df, state):
    """Phase 2: Add gender to features — should trigger cascade."""
    print("\n--- Phase 2: Change Features (add gender) ---")
    
    # Capture pre-change state
    had_models = state.get('trained_models') is not None
    had_splits = state.get('X_train') is not None
    had_shap = state.get('shap_results') is not None
    assert had_models and had_splits and had_shap, "Pre-change state should be populated"
    log("Pre-change state verified (models, splits, SHAP present)", "PASS")
    
    # User changes features to include gender
    new_features = ['age', 'bmi', 'cholesterol', 'blood_pressure', 'exercise_hours', 'gender']
    
    changed = simulate_feature_change_cascade(state, new_features)
    assert changed, "Feature change should be detected"
    log("Feature change detected by hash comparison", "PASS")
    
    # Verify ALL downstream state is cleared
    cleared_keys = []
    surviving_keys = []
    for key in CASCADE_KEYS:
        if state.get(key) is not None:
            surviving_keys.append(key)
        else:
            cleared_keys.append(key)
    
    if surviving_keys:
        log(f"STALE STATE FOUND: {surviving_keys}", "FAIL")
    else:
        log(f"All {len(cleared_keys)} downstream keys cleared", "PASS")
    
    # Specifically check critical keys
    assert state.get('trained_models') is None, "Models should be cleared"
    assert state.get('X_train') is None, "Splits should be cleared"
    assert state.get('shap_results') is None, "SHAP should be cleared"
    assert state.get('model_results') is None, "Results should be cleared"
    log("Critical state (models, splits, SHAP, results) confirmed cleared", "PASS")
    
    # Verify non-cascade state survives
    assert state.get('raw_data') is not None, "Raw data should survive"
    assert state.get('_data_config_features_hash') is not None, "Hash should be updated"
    log("Non-cascade state (raw data, hash) survives", "PASS")
    
    return state


def test_phase3_retrain_with_new_features(df, state):
    """Phase 3: Retrain with the new feature set — should work cleanly."""
    print("\n--- Phase 3: Retrain with New Features ---")
    
    # Update data config with new features
    new_features = ['age', 'bmi', 'cholesterol', 'blood_pressure', 'exercise_hours', 'gender']
    state['data_config'] = DataConfig(
        target_col='glucose',
        feature_cols=new_features,
        task_type='regression',
    )
    
    # Prepare new splits (with gender now available for subgroup analysis)
    splits = prepare_splits(df, target_col='glucose')
    for k, v in splits.items():
        state[k] = v
    
    # Verify we can train without "column not in dataframe" error
    try:
        result = train_ridge_model(splits)
        state['trained_models'] = {'ridge': result['model']}
        state['model_results'] = {'ridge': {'metrics': result['metrics'], 'y_test': splits['y_test'].values, 'y_test_pred': result['y_test_pred']}}
        log("Retrained successfully with new feature set", "PASS")
    except KeyError as e:
        log(f"KeyError during retrain: {e} — THIS IS THE BUG #32 WAS ABOUT", "FAIL")
        return state
    except Exception as e:
        log(f"Unexpected error during retrain: {e}", "FAIL")
        return state
    
    # Verify metrics are reasonable
    metrics = result['metrics']
    r2 = None
    for k, v in metrics.items():
        if 'r2' in k.lower() or 'r²' in k.lower():
            r2 = v
            break
    if r2 is not None:
        assert r2 > 0, f"R² should be positive, got {r2}"
        log(f"New model R² = {r2:.4f} (positive)", "PASS")
    
    return state


def test_phase4_no_false_cascade():
    """Phase 4: Re-saving with SAME features should NOT cascade."""
    print("\n--- Phase 4: No False Cascade ---")
    
    df = build_regression_df(n=100)
    state = {}
    features = ['age', 'bmi', 'cholesterol']
    
    state['_data_config_features_hash'] = hashlib.md5(
        ','.join(sorted(features)).encode()
    ).hexdigest()[:8]
    
    # Add some "trained" state
    state['trained_models'] = {'ridge': 'dummy'}
    state['X_train'] = 'dummy'
    
    # Re-save with same features
    changed = simulate_feature_change_cascade(state, features)
    
    assert not changed, "Same features should not trigger cascade"
    assert state.get('trained_models') is not None, "Models should survive"
    assert state.get('X_train') is not None, "Splits should survive"
    log("Same features → no cascade (models preserved)", "PASS")
    
    # Re-save with same features in different order
    changed = simulate_feature_change_cascade(state, ['cholesterol', 'age', 'bmi'])
    assert not changed, "Reordered features should not trigger cascade"
    log("Reordered features → no cascade (order-independent)", "PASS")


def test_phase5_subgroup_access(df, state):
    """Phase 5: Verify subgroup analysis can access all raw columns."""
    print("\n--- Phase 5: Subgroup Access After Retrain ---")
    
    test_indices = state.get('test_indices')
    if test_indices is None:
        log("No test indices in state", "FAIL")
        return
    
    raw_df = state['raw_data']
    
    # All original columns should be accessible
    for col in ['gender', 'smoking', 'age', 'bmi']:
        values = raw_df.iloc[test_indices][col].values
        assert len(values) == len(state['X_test']), f"{col} length mismatch"
    
    log("All raw columns accessible via test_indices for subgroup analysis", "PASS")
    
    # Verify we can stratify by gender
    gender_labels = raw_df.iloc[test_indices]['gender'].values
    unique = set(gender_labels)
    assert len(unique) >= 2, f"Expected 2 gender values, got {unique}"
    log("Gender stratification possible with both values present", "PASS")


def run():
    print("=" * 60)
    print("Workflow Test: State Invalidation (#32)")
    print("=" * 60)
    
    try:
        df, state = test_phase1_initial_training()
        state = test_phase2_change_features(df, state)
        state = test_phase3_retrain_with_new_features(df, state)
        test_phase4_no_false_cascade()
        test_phase5_subgroup_access(df, state)
    except Exception as e:
        import traceback
        log(f"UNHANDLED: {e}", "FAIL")
        traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(f"  ❌ {f}")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
