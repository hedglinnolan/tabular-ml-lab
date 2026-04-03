"""
Fixtures for Streamlit AppTest integration tests (Tier 2).

These tests use Streamlit's AppTest framework to render pages in-process
without a running server. Slower than unit tests but catch widget bugs,
session state issues, and page rendering crashes.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def build_test_dataframe(n=200, seed=42):
    """Synthetic regression dataset for page rendering tests."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'age': np.random.normal(50, 15, n).clip(18, 90),
        'bmi': np.random.normal(27, 5, n).clip(15, 50),
        'cholesterol': np.random.normal(200, 40, n).clip(100, 350),
        'blood_pressure': np.random.normal(120, 20, n).clip(80, 200),
        'exercise_hours': np.random.exponential(3, n).clip(0, 20),
        'gender': np.random.choice(['male', 'female'], n),
        'smoking': np.random.choice(['never', 'former', 'current'], n),
    })
    df['glucose'] = (
        50 + 0.5 * df['age'] + 2.0 * df['bmi']
        - 0.1 * df['cholesterol'] + np.random.normal(0, 10, n)
    )
    df.loc[np.random.choice(n, 10, replace=False), 'bmi'] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), 'cholesterol'] = np.nan
    return df


def build_classification_dataframe(n=200, seed=42):
    """Synthetic imbalanced classification dataset."""
    df = build_test_dataframe(n=n, seed=seed)
    threshold = df['glucose'].quantile(0.8)  # 80/20 imbalance
    df['condition'] = (df['glucose'] > threshold).astype(int)
    df = df.drop(columns=['glucose'])
    return df


def inject_data_state(at, df, target_col='glucose', task_type='regression'):
    """Inject uploaded dataset state into AppTest session."""
    from utils.session_state import DataConfig
    from ml.dataset_profile import compute_dataset_profile

    feature_cols = [c for c in df.columns if c != target_col]
    at.session_state['raw_data'] = df
    at.session_state['filtered_data'] = df
    at.session_state['task_mode'] = 'prediction'
    at.session_state['data_config'] = DataConfig(
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
    )
    at.session_state['selected_features'] = feature_cols
    at.session_state['data_audit'] = {'n_rows': len(df), 'n_cols': len(df.columns)}

    try:
        profile = compute_dataset_profile(df, target_col, feature_cols, task_type)
        at.session_state['dataset_profile'] = profile
    except Exception:
        pass

    # Feature hash
    import hashlib
    numeric_features = [c for c in feature_cols if df[c].dtype in ('float64', 'int64')]
    at.session_state['_data_config_features_hash'] = hashlib.md5(
        ','.join(sorted(numeric_features)).encode()
    ).hexdigest()[:8]


def inject_trained_state(at, df, target_col='glucose'):
    """Inject splits + trained Ridge model for downstream page tests."""
    from utils.session_state import SplitConfig
    from sklearn.linear_model import Ridge
    from ml.eval import calculate_regression_metrics

    feature_cols = [c for c in df.columns
                    if c != target_col and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols].copy().fillna(df[feature_cols].median())
    y = df.loc[mask, target_col].copy()

    n = len(X)
    n_train, n_val = int(n * 0.7), int(n * 0.15)

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train+n_val], X.iloc[n_train+n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train+n_val], y.iloc[n_train+n_val:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test.values, y_pred)

    at.session_state['split_config'] = SplitConfig()
    at.session_state['X_train'] = X_train
    at.session_state['X_val'] = X_val
    at.session_state['X_test'] = X_test
    at.session_state['y_train'] = y_train
    at.session_state['y_val'] = y_val
    at.session_state['y_test'] = y_test
    at.session_state['train_indices'] = list(range(n_train))
    at.session_state['val_indices'] = list(range(n_train, n_train + n_val))
    at.session_state['test_indices'] = list(range(n_train + n_val, n))
    at.session_state['feature_names'] = feature_cols
    at.session_state['selected_features'] = feature_cols
    at.session_state['trained_models'] = {'ridge': model}
    at.session_state['fitted_estimators'] = {'ridge': model}
    at.session_state['model_results'] = {
        'ridge': {
            'metrics': metrics,
            'y_test': y_test.values,
            'y_test_pred': y_pred,
        }
    }
