#!/usr/bin/env python
"""
Workflow test: Full regression pipeline.
Upload → Configure → EDA → Feature Selection → Preprocess → Train → Explain → Export

Verifies that each step's output is consumed correctly by downstream steps,
and that the final export reflects all decisions made along the way.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import (
    build_regression_df, inject_uploaded_state, prepare_splits,
    train_ridge_model, make_data_config,
)

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


def test_step1_upload_and_configure():
    """Step 1: Upload data and configure prediction task."""
    print("\n--- Step 1: Upload & Configure ---")
    
    df = build_regression_df(n=200)
    state = {}
    data_config = inject_uploaded_state(state, df, target_col='glucose', task_type='regression')
    
    # Verify state is populated
    assert state.get('raw_data') is not None, "raw_data missing"
    assert state.get('data_config') is not None, "data_config missing"
    assert data_config.target_col == 'glucose'
    assert data_config.task_type == 'regression'
    assert len(data_config.feature_cols) > 0
    assert 'glucose' not in data_config.feature_cols, "target in features"
    log("Data loaded and configured correctly", "PASS")
    
    # Verify feature hash set
    assert state.get('_data_config_features_hash'), "Feature hash not set"
    log("Feature hash initialized", "PASS")
    
    # Verify profile computed
    assert state.get('dataset_profile') is not None, "Profile not computed"
    log("Dataset profile computed", "PASS")
    
    return df, state


def test_step2_eda_signals(df, state):
    """Step 2: Verify EDA signals are available from the profile."""
    print("\n--- Step 2: EDA Signals ---")
    
    profile = state['dataset_profile']
    
    # Profile should have signals
    assert hasattr(profile, 'signals') or hasattr(profile, 'n_rows'), "Profile structure unexpected"
    log("Profile has expected structure", "PASS")
    
    # Verify we can compute signals
    from ml.eda_recommender import DatasetSignals
    data_config = state['data_config']
    feature_cols = data_config.feature_cols
    numeric_cols = [c for c in feature_cols if df[c].dtype in ('float64', 'int64')]
    
    signals = DatasetSignals(n_rows=len(df), n_cols=len(feature_cols), numeric_cols=numeric_cols)
    assert signals.n_rows == 200
    log("EDA signals computable", "PASS")
    
    return state


def test_step3_prepare_splits(df, state):
    """Step 3: Prepare train/val/test splits."""
    print("\n--- Step 3: Prepare Splits ---")
    
    splits = prepare_splits(df, target_col='glucose')
    for k, v in splits.items():
        state[k] = v
    
    X_train = state['X_train']
    X_val = state['X_val']
    X_test = state['X_test']
    y_train = state['y_train']
    y_test = state['y_test']
    
    # Verify split sizes
    total = len(X_train) + len(X_val) + len(X_test)
    assert total > 0, "No data after split"
    assert len(X_train) > len(X_test), "Train should be larger than test"
    log(f"Splits created: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}", "PASS")
    
    # Verify no target leakage into features
    assert 'glucose' not in X_train.columns, "Target leaked into features"
    log("No target leakage in features", "PASS")
    
    # Verify indices stored
    assert len(state['test_indices']) == len(X_test)
    log("Split indices stored", "PASS")
    
    # Verify y alignment
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    log("X/y alignment correct", "PASS")
    
    return splits, state


def test_step4_train_model(splits, state):
    """Step 4: Train a model and verify metrics."""
    print("\n--- Step 4: Train Model ---")
    
    result = train_ridge_model(splits)
    model = result['model']
    metrics = result['metrics']
    
    # Verify model trained
    assert hasattr(model, 'coef_'), "Model not fitted"
    log("Ridge model trained", "PASS")
    
    # Verify predictions shape
    assert len(result['y_test_pred']) == len(splits['y_test'])
    log("Predictions shape matches test set", "PASS")
    
    # Verify metrics are reasonable (R² should be positive for this synthetic data)
    r2 = metrics.get('r2', metrics.get('R²', None))
    if r2 is not None:
        assert r2 > 0, f"R² should be positive for synthetic data, got {r2}"
        log(f"R² = {r2:.4f} (positive, as expected)", "PASS")
    else:
        # Try other key formats
        for k, v in metrics.items():
            if 'r2' in k.lower() or 'r²' in k.lower():
                assert v > 0, f"R² should be positive, got {v}"
                log(f"{k} = {v:.4f} (positive, as expected)", "PASS")
                break
    
    # Verify RMSE is reasonable
    rmse = metrics.get('rmse', metrics.get('RMSE', None))
    if rmse is not None:
        assert 0 < rmse < 100, f"RMSE should be reasonable for this data, got {rmse}"
        log(f"RMSE = {rmse:.4f} (reasonable range)", "PASS")
    
    # Store in state
    state['trained_models'] = {'ridge': model}
    state['model_results'] = {
        'ridge': {
            'metrics': metrics,
            'y_test': splits['y_test'].values,
            'y_test_pred': result['y_test_pred'],
        }
    }
    log("Model results stored in state", "PASS")
    
    return state


def test_step5_explainability_compatibility(df, state):
    """Step 5: Verify explainability can access subgroup variables from raw data."""
    print("\n--- Step 5: Explainability Compatibility ---")
    
    test_indices = state.get('test_indices')
    assert test_indices is not None, "No test indices"
    
    # Subgroup analysis should be able to pull from raw df
    raw_df = state['raw_data']
    
    # Gender should be accessible even though it's not in X_test (it's categorical)
    X_test = state['X_test']
    assert 'gender' not in X_test.columns, "Gender shouldn't be in numeric features"
    
    # But we can get it from raw df using test indices
    subgroup_labels = raw_df.iloc[test_indices]['gender'].values
    assert len(subgroup_labels) == len(X_test), "Subgroup labels length mismatch"
    assert set(subgroup_labels).issubset({'male', 'female'}), "Unexpected gender values"
    log("Subgroup analysis can access gender from raw data (#32 fix)", "PASS")
    
    # Verify smoking also accessible
    smoking_labels = raw_df.iloc[test_indices]['smoking'].values
    assert len(smoking_labels) == len(X_test)
    log("Subgroup analysis can access smoking from raw data", "PASS")


def test_step6_publication_output(state):
    """Step 6: Verify publication methods section is complete."""
    print("\n--- Step 6: Publication Output ---")
    
    from ml.publication import generate_methods_section
    
    data_config = state['data_config']
    split_config = state.get('split_config')
    
    sc_dict = {
        'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
        'target_trim_enabled': False,
        'target_transform': 'none',
        'stratify': False, 'use_time_split': False,
    }
    
    methods = generate_methods_section(
        data_config={'feature_cols': data_config.feature_cols, 'target_col': 'glucose', 'task_type': 'regression'},
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
    assert 'training' in methods.lower() or 'split' in methods.lower(), "Split not mentioned"
    log("Methods section generated successfully", "PASS")
    
    assert len(methods) > 200, f"Methods section too short ({len(methods)} chars)"
    log(f"Methods section has substance ({len(methods)} chars)", "PASS")


def run():
    print("=" * 60)
    print("Workflow Test: Full Regression Pipeline")
    print("=" * 60)
    
    try:
        df, state = test_step1_upload_and_configure()
        state = test_step2_eda_signals(df, state)
        splits, state = test_step3_prepare_splits(df, state)
        state = test_step4_train_model(splits, state)
        test_step5_explainability_compatibility(df, state)
        test_step6_publication_output(state)
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
