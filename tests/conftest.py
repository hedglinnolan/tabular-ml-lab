"""
Shared test fixtures for Tabular ML Lab integration tests.
"""
import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Streamlit 1.62 changed AppTest.from_file to resolve relative paths against
# the CALLING file instead of the working directory; 90 call sites across this
# suite say `AppTest.from_file("pages/…")` and every one broke at once on a
# fresh dependency resolution (CI) while older-pinned local venvs stayed green.
# Restore the one behavior every call site was written against — relative
# paths mean repo-root-relative — in this one place, for tests/ and
# tests/integration alike, rather than editing 90 sites into 90 chances to
# drift. Absolute-path call sites are untouched.
# Guarded: the pre-commit gates run single test files under an interpreter
# that deliberately carries no streamlit; this conftest must stay importable
# there (the probe-coverage test enforces exactly that). Where streamlit is
# absent, no AppTest test can run anyway, so there is nothing to patch.
try:
    from streamlit.testing.v1 import AppTest as _AppTest  # noqa: E402
except ImportError:
    _AppTest = None
else:
    _orig_from_file = _AppTest.from_file.__func__

    def _from_repo_root(cls, script_path, **kwargs):
        p = str(script_path)
        if not os.path.isabs(p):
            p = os.path.join(PROJECT_ROOT, p)
        return _orig_from_file(cls, p, **kwargs)

    _AppTest.from_file = classmethod(_from_repo_root)


def build_regression_df(n=200, seed=42, missing_rate=0.05):
    """Build a synthetic regression dataset with known relationships."""
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
    # Target with known linear relationship + noise
    df['glucose'] = (
        50 + 0.5 * df['age'] + 2.0 * df['bmi']
        - 0.1 * df['cholesterol'] + 0.3 * df['blood_pressure']
        + np.random.normal(0, 10, n)
    )
    # Add missing values
    n_missing = int(n * missing_rate)
    for col in ['bmi', 'cholesterol', 'exercise_hours']:
        idx = np.random.choice(n, n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df


def build_classification_df(n=200, seed=42):
    """Build a synthetic binary classification dataset."""
    np.random.seed(seed)
    df = build_regression_df(n=n, seed=seed, missing_rate=0.02)
    df['target_class'] = (df['glucose'] > df['glucose'].median()).astype(int)
    df = df.drop(columns=['glucose'])
    return df


def make_data_config(df, target_col, task_type='regression'):
    """Create a DataConfig from a DataFrame."""
    from utils.session_state import DataConfig
    feature_cols = [c for c in df.columns if c != target_col]
    return DataConfig(
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
    )


def inject_uploaded_state(session_state, df, target_col='glucose', task_type='regression'):
    """Simulate a completed Upload & Audit step."""
    data_config = make_data_config(df, target_col, task_type)
    feature_cols = data_config.feature_cols

    session_state['raw_data'] = df
    session_state['filtered_data'] = df
    session_state['task_mode'] = 'prediction'
    session_state['data_config'] = data_config
    session_state['data_audit'] = {'n_rows': len(df), 'n_cols': len(df.columns)}

    try:
        from ml.dataset_profile import compute_dataset_profile
        profile = compute_dataset_profile(df, target_col, feature_cols, task_type)
        session_state['dataset_profile'] = profile
    except Exception as e:
        import warnings
        warnings.warn(f"inject_uploaded_state: dataset_profile computation failed: {e}")

    # Set feature hash (as Upload & Audit would)
    import hashlib
    session_state['_data_config_features_hash'] = hashlib.md5(
        ','.join(sorted(feature_cols)).encode()
    ).hexdigest()[:8]

    return data_config


def prepare_splits(df, target_col='glucose', train_frac=0.7, val_frac=0.15):
    """Prepare train/val/test splits and return as a dict suitable for session_state."""
    from utils.session_state import SplitConfig

    feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols].copy().fillna(df[feature_cols].median())
    y = df.loc[mask, target_col].copy()

    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    return {
        'split_config': SplitConfig(),
        'X_train': X.iloc[idx_train],
        'X_val': X.iloc[idx_val],
        'X_test': X.iloc[idx_test],
        'y_train': y.iloc[idx_train],
        'y_val': y.iloc[idx_val],
        'y_test': y.iloc[idx_test],
        # Row identity is the index LABEL, as the app stores it: `pages/06`
        # writes `*_row_labels` and nothing carries positions across a page
        # boundary any more. `X.index` is the source frame's labels for the
        # rows that survived the target mask, so these are labels into `df`.
        'train_row_labels': X.index[idx_train].tolist(),
        'val_row_labels': X.index[idx_val].tolist(),
        'test_row_labels': X.index[idx_test].tolist(),
        'feature_names': feature_cols,
    }


def train_ridge_model(splits, alpha=1.0):
    """Train a Ridge model on the splits and return results dict."""
    from sklearn.linear_model import Ridge
    from ml.eval import calculate_regression_metrics

    model = Ridge(alpha=alpha)
    model.fit(splits['X_train'], splits['y_train'])

    y_pred_test = model.predict(splits['X_test'])
    y_pred_val = model.predict(splits['X_val'])
    metrics = calculate_regression_metrics(
        splits['y_test'].values if hasattr(splits['y_test'], 'values') else splits['y_test'],
        y_pred_test,
    )

    return {
        'model': model,
        'y_test_pred': y_pred_test,
        'y_val_pred': y_pred_val,
        'metrics': metrics,
    }


def inject_trained_state(session_state, df, target_col='glucose'):
    """Full injection: upload + split + train a Ridge model."""
    data_config = inject_uploaded_state(session_state, df, target_col)
    splits = prepare_splits(df, target_col)

    for k, v in splits.items():
        session_state[k] = v

    result = train_ridge_model(splits)
    session_state['trained_models'] = {'ridge': result['model']}
    session_state['fitted_estimators'] = {'ridge': result['model']}
    session_state['model_results'] = {
        'ridge': {
            'metrics': result['metrics'],
            'y_test': splits['y_test'].values if hasattr(splits['y_test'], 'values') else splits['y_test'],
            'y_test_pred': result['y_test_pred'],
        }
    }
    return splits, result
