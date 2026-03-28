"""
Shared pytest fixtures for workflow integration tests.

Module-scoped fixtures provide mutable state containers that persist
across sequential test steps within each file. Tests modify the shared
state dict in-place, simulating a real user session progressing through
the app pipeline.
"""
import sys
import os
import hashlib
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import (
    build_regression_df, inject_uploaded_state, prepare_splits,
    train_ridge_model, make_data_config,
)


# ── Regression flow fixtures ────────────────────────────────────────

@pytest.fixture(scope="module")
def regression_df():
    """Synthetic regression dataset shared across the regression pipeline tests."""
    return build_regression_df(n=200)


@pytest.fixture(scope="module")
def regression_state(regression_df):
    """Mutable session-state dict pre-populated with upload & configure output."""
    state = {}
    inject_uploaded_state(state, regression_df, target_col='glucose', task_type='regression')
    return state


# ── State-invalidation fixtures ─────────────────────────────────────

@pytest.fixture(scope="module")
def invalidation_df():
    """Separate df instance for the state-invalidation test sequence."""
    return build_regression_df(n=200)


@pytest.fixture(scope="module")
def invalidation_state(invalidation_df):
    """
    Mutable session-state for the invalidation sequence.
    Starts with a trained model on numeric-only features (no gender).
    """
    from utils.session_state import DataConfig

    df = invalidation_df
    state = {}
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

    # Train
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
    state['fitted_estimators'] = {'ridge': result['model']}
    state['preprocessing_pipelines_by_model'] = {'ridge': 'dummy_pipeline'}
    state['shap_results'] = {'ridge': 'dummy_shap'}
    state['permutation_importance'] = {'ridge': 'dummy_perm'}

    return state


# ── Target-transform fixtures ───────────────────────────────────────

def _build_skewed_target_df(n=300, seed=42):
    """Dataset with heavily right-skewed target (like medical costs)."""
    np.random.seed(seed)
    df = __import__('pandas').DataFrame({
        'age': np.random.normal(50, 15, n).clip(18, 90),
        'bmi': np.random.normal(27, 5, n).clip(15, 50),
        'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'exercise': np.random.exponential(3, n).clip(0, 15),
    })
    df['cost'] = np.exp(
        5 + 0.02 * df['age'] + 0.05 * df['bmi'] + 1.5 * df['smoker']
        - 0.1 * df['exercise'] + np.random.normal(0, 0.5, n)
    )
    return df


@pytest.fixture(scope="module")
def skewed_df():
    """Skewed-target dataset for transform tests."""
    return _build_skewed_target_df()


@pytest.fixture(scope="module")
def skewed_splits(skewed_df):
    """Train/val/test splits of the skewed dataset."""
    return prepare_splits(skewed_df, target_col='cost')


@pytest.fixture(scope="module")
def baseline_rmse(skewed_splits):
    """Baseline RMSE (no transform) for comparison in transform tests."""
    from ml.eval import calculate_regression_metrics
    from sklearn.linear_model import Ridge

    X_train = skewed_splits['X_train'].values
    X_test = skewed_splits['X_test'].values
    y_train = skewed_splits['y_train'].values
    y_test = skewed_splits['y_test'].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    for k, v in metrics.items():
        if 'rmse' in k.lower():
            return v
    raise ValueError("RMSE not found in metrics")
