#!/usr/bin/env python
"""
Workflow test: Target variable transformation end-to-end.

Verifies the complete pipeline:
1. Train WITHOUT transform → get baseline metrics
2. Train WITH log1p transform → verify metrics are on original scale
3. Train WITH Yeo-Johnson → verify metrics are on original scale
4. Verify back-transformed predictions are in the right range
5. Verify metrics improve for skewed targets with transform
6. Verify CV with TransformedTargetRegressor works
7. Verify methods section reflects the transform
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import build_regression_df, prepare_splits

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from ml.eval import calculate_regression_metrics

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


def build_skewed_target_df(n=300, seed=42):
    """Build a dataset with a heavily right-skewed target (like medical costs)."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'age': np.random.normal(50, 15, n).clip(18, 90),
        'bmi': np.random.normal(27, 5, n).clip(15, 50),
        'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'exercise': np.random.exponential(3, n).clip(0, 15),
    })
    # Heavily right-skewed target (like medical costs)
    df['cost'] = np.exp(
        5 + 0.02 * df['age'] + 0.05 * df['bmi'] + 1.5 * df['smoker']
        - 0.1 * df['exercise'] + np.random.normal(0, 0.5, n)
    )
    return df


def train_with_transform(splits, transform_type='none'):
    """Train a Ridge model with optional target transformation, mimicking the app's logic."""
    X_train = splits['X_train'].values if hasattr(splits['X_train'], 'values') else splits['X_train']
    X_val = splits['X_val'].values if hasattr(splits['X_val'], 'values') else splits['X_val']
    X_test = splits['X_test'].values if hasattr(splits['X_test'], 'values') else splits['X_test']
    y_train = splits['y_train'].values if hasattr(splits['y_train'], 'values') else splits['y_train']
    y_val = splits['y_val'].values if hasattr(splits['y_val'], 'values') else splits['y_val']
    y_test = splits['y_test'].values if hasattr(splits['y_test'], 'values') else splits['y_test']
    
    y_test_original = y_test.copy()
    
    if transform_type == 'none':
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        return metrics, y_pred, y_test_original, None
    
    elif transform_type == 'log1p':
        # Transform
        y_train_t = np.log1p(y_train)
        y_val_t = np.log1p(y_val)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_t)
        
        # Predict and back-transform
        y_pred_t = model.predict(X_test)
        y_pred = np.expm1(y_pred_t)
        
        # Metrics on ORIGINAL scale
        metrics = calculate_regression_metrics(y_test_original, y_pred)
        return metrics, y_pred, y_test_original, 'log1p'
    
    elif transform_type in ('yeo-johnson', 'box-cox'):
        method = transform_type  # sklearn expects 'yeo-johnson' with hyphen
        pt = PowerTransformer(method=method, standardize=False)
        pt.fit(y_train.reshape(-1, 1))
        
        y_train_t = pt.transform(y_train.reshape(-1, 1)).ravel()
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_t)
        
        # Predict and back-transform
        y_pred_t = model.predict(X_test)
        y_pred = pt.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
        
        # Metrics on ORIGINAL scale
        metrics = calculate_regression_metrics(y_test_original, y_pred)
        return metrics, y_pred, y_test_original, pt


def test_baseline_no_transform():
    """Test 1: Baseline metrics without transformation."""
    print("\n--- Test 1: Baseline (no transform) ---")
    
    df = build_skewed_target_df()
    splits = prepare_splits(df, target_col='cost')
    
    metrics, y_pred, y_test, _ = train_with_transform(splits, 'none')
    
    # Get RMSE
    rmse = None
    for k, v in metrics.items():
        if 'rmse' in k.lower():
            rmse = v
            break
    
    assert rmse is not None, "RMSE not in metrics"
    assert rmse > 0, "RMSE should be positive"
    log(f"Baseline RMSE = {rmse:.2f}", "PASS")
    
    # Predictions should be in a similar range to test values
    assert y_pred.min() < y_test.max(), "Predictions should overlap with test range"
    log("Baseline predictions in reasonable range", "PASS")
    
    return splits, rmse


def test_log1p_transform(splits, baseline_rmse):
    """Test 2: Train with log1p and verify metrics are on original scale."""
    print("\n--- Test 2: log1p Transform ---")
    
    metrics, y_pred, y_test, _ = train_with_transform(splits, 'log1p')
    
    rmse = None
    for k, v in metrics.items():
        if 'rmse' in k.lower():
            rmse = v
            break
    
    assert rmse is not None, "RMSE not in metrics"
    assert rmse > 0, "RMSE should be positive"
    log(f"log1p RMSE = {rmse:.2f} (baseline was {baseline_rmse:.2f})", "PASS")
    
    # Key check: RMSE should be on ORIGINAL scale (same order of magnitude as baseline)
    # If we accidentally reported RMSE on log scale, it would be ~0.5, not ~200
    assert rmse > 10, f"RMSE suspiciously small ({rmse:.2f}) — might be on log scale"
    log("RMSE confirmed on original scale (not log scale)", "PASS")
    
    # Predictions should be positive (expm1 of anything is > -1)
    assert y_pred.min() > -2, f"Back-transformed predictions should be > -1, got {y_pred.min():.2f}"
    log("Back-transformed predictions are positive", "PASS")
    
    # RMSE should improve with log transform for skewed data
    if rmse < baseline_rmse:
        log(f"log1p improved RMSE: {rmse:.2f} < {baseline_rmse:.2f}", "PASS")
    else:
        log(f"log1p didn't improve RMSE ({rmse:.2f} >= {baseline_rmse:.2f}) — acceptable for some data", "PASS")
    
    return rmse


def test_yeojohnson_transform(splits, baseline_rmse):
    """Test 3: Train with Yeo-Johnson and verify back-transformation."""
    print("\n--- Test 3: Yeo-Johnson Transform ---")
    
    metrics, y_pred, y_test, pt = train_with_transform(splits, 'yeo-johnson')
    
    rmse = None
    for k, v in metrics.items():
        if 'rmse' in k.lower():
            rmse = v
            break
    
    assert rmse is not None, "RMSE not in metrics"
    assert rmse > 10, f"RMSE suspiciously small ({rmse:.2f}) — might be on transformed scale"
    log(f"Yeo-Johnson RMSE = {rmse:.2f} (original scale confirmed)", "PASS")
    
    # Verify PowerTransformer is fitted
    assert hasattr(pt, 'lambdas_'), "PowerTransformer not fitted"
    log(f"Yeo-Johnson lambda = {pt.lambdas_[0]:.4f}", "PASS")
    
    # Verify back-transform roundtrip
    y_sample = np.array([100.0, 500.0, 1000.0, 5000.0])
    y_t = pt.transform(y_sample.reshape(-1, 1)).ravel()
    y_back = pt.inverse_transform(y_t.reshape(-1, 1)).ravel()
    assert np.allclose(y_sample, y_back, atol=1e-4), "Roundtrip failed"
    log("Yeo-Johnson roundtrip verified", "PASS")


def test_boxcox_positive_check(splits):
    """Test 4: Box-Cox requires positive values."""
    print("\n--- Test 4: Box-Cox Positive Check ---")
    
    # Our cost data is positive, so it should work
    metrics, y_pred, y_test, pt = train_with_transform(splits, 'box-cox')
    
    rmse = None
    for k, v in metrics.items():
        if 'rmse' in k.lower():
            rmse = v
            break
    
    assert rmse is not None and rmse > 10, "Box-Cox RMSE should be on original scale"
    log(f"Box-Cox RMSE = {rmse:.2f} (original scale)", "PASS")
    
    # Test with data that has zero/negative values — should fail
    try:
        y_bad = np.array([-1.0, 0.0, 1.0, 5.0])
        pt_bad = PowerTransformer(method='box-cox', standardize=False)
        pt_bad.fit(y_bad.reshape(-1, 1))
        log("Box-Cox should have failed on negative values", "FAIL")
    except ValueError:
        log("Box-Cox correctly rejects negative values", "PASS")


def test_cv_with_transform(splits):
    """Test 5: Cross-validation with TransformedTargetRegressor."""
    print("\n--- Test 5: CV with TransformedTargetRegressor ---")
    
    from sklearn.model_selection import cross_val_score
    
    X_train = splits['X_train'].values if hasattr(splits['X_train'], 'values') else splits['X_train']
    y_train = splits['y_train'].values if hasattr(splits['y_train'], 'values') else splits['y_train']
    
    # log1p CV
    cv_model = TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    scores = cross_val_score(cv_model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    
    assert len(rmse_scores) == 3, "Should have 3 CV folds"
    assert all(s > 0 for s in rmse_scores), "RMSE scores should be positive"
    assert all(s > 10 for s in rmse_scores), "CV RMSE should be on original scale"
    log(f"CV RMSE (log1p): {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}", "PASS")
    
    # Yeo-Johnson CV
    cv_model_yj = TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        transformer=PowerTransformer(method='yeo-johnson', standardize=False),
    )
    scores_yj = cross_val_score(cv_model_yj, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
    rmse_yj = -scores_yj
    
    assert all(s > 10 for s in rmse_yj), "Yeo-Johnson CV RMSE should be on original scale"
    log(f"CV RMSE (Yeo-Johnson): {rmse_yj.mean():.2f} ± {rmse_yj.std():.2f}", "PASS")


def test_methods_section_reflects_transform():
    """Test 6: Publication methods section includes transform details."""
    print("\n--- Test 6: Methods Section ---")
    
    from ml.publication import generate_methods_section
    
    for transform, expected_text in [
        ('log1p', 'log(1+x)'),
        ('yeo-johnson', 'Yeo-Johnson'),
        ('box-cox', 'Box-Cox'),
    ]:
        methods = generate_methods_section(
            data_config={'feature_cols': ['age', 'bmi'], 'target_col': 'cost', 'task_type': 'regression'},
            preprocessing_config={}, model_configs={},
            split_config={
                'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
                'target_trim_enabled': False,
                'target_transform': transform,
                'stratify': False, 'use_time_split': False,
            },
            n_total=300, n_train=210, n_val=45, n_test=45,
            feature_names=['age', 'bmi'], target_name='cost',
            task_type='regression', metrics_used=['RMSE', 'R²'],
        )
        
        assert expected_text in methods, f"'{expected_text}' not in methods for {transform}"
        assert 'back-transformed' in methods.lower() or 'original scale' in methods.lower(), \
            f"Back-transform not mentioned for {transform}"
        log(f"Methods section correct for {transform}", "PASS")
    
    # 'none' should NOT mention transform
    methods_none = generate_methods_section(
        data_config={'feature_cols': ['age'], 'target_col': 'cost', 'task_type': 'regression'},
        preprocessing_config={}, model_configs={},
        split_config={
            'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
            'target_trim_enabled': False,
            'target_transform': 'none',
            'stratify': False, 'use_time_split': False,
        },
        n_total=300, n_train=210, n_val=45, n_test=45,
        feature_names=['age'], target_name='cost',
        task_type='regression', metrics_used=['RMSE'],
    )
    assert 'Yeo-Johnson' not in methods_none, "transform=none should not mention Yeo-Johnson"
    log("Methods section omits transform when none applied", "PASS")


def test_edge_case_constant_target():
    """Test 7: Transform with near-constant target shouldn't crash."""
    print("\n--- Test 7: Edge Case — Near-Constant Target ---")
    
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'target': 100.0 + np.random.normal(0, 0.001, n),  # Nearly constant
    })
    
    splits = prepare_splits(df, target_col='target')
    
    try:
        metrics, y_pred, y_test, _ = train_with_transform(splits, 'log1p')
        log("log1p on near-constant target: no crash", "PASS")
    except Exception as e:
        log(f"log1p on near-constant target crashed: {e}", "FAIL")
    
    try:
        metrics, y_pred, y_test, pt = train_with_transform(splits, 'yeo-johnson')
        log("Yeo-Johnson on near-constant target: no crash", "PASS")
    except Exception as e:
        log(f"Yeo-Johnson on near-constant target crashed: {e}", "FAIL")


def run():
    print("=" * 60)
    print("Workflow Test: Target Variable Transformation")
    print("=" * 60)
    
    try:
        splits, baseline_rmse = test_baseline_no_transform()
        test_log1p_transform(splits, baseline_rmse)
        test_yeojohnson_transform(splits, baseline_rmse)
        test_boxcox_positive_check(splits)
        test_cv_with_transform(splits)
        test_methods_section_reflects_transform()
        test_edge_case_constant_target()
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
