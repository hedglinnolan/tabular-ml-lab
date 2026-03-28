"""
Workflow test: Target variable transformation end-to-end.

Verifies the complete pipeline:
1. Train WITHOUT transform → baseline metrics
2. Train WITH log1p → verify metrics on original scale
3. Train WITH Yeo-Johnson → verify back-transformation
4. Box-Cox positive-value requirement
5. CV with TransformedTargetRegressor
6. Publication methods section reflects transform choice
7. Edge case: near-constant target
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import prepare_splits

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from ml.eval import calculate_regression_metrics


def train_with_transform(splits, transform_type='none'):
    """Train a Ridge model with optional target transformation."""
    X_train = splits['X_train'].values if hasattr(splits['X_train'], 'values') else splits['X_train']
    X_test = splits['X_test'].values if hasattr(splits['X_test'], 'values') else splits['X_test']
    y_train = splits['y_train'].values if hasattr(splits['y_train'], 'values') else splits['y_train']
    y_test = splits['y_test'].values if hasattr(splits['y_test'], 'values') else splits['y_test']

    y_test_original = y_test.copy()

    if transform_type == 'none':
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        return metrics, y_pred, y_test_original, None

    elif transform_type == 'log1p':
        y_train_t = np.log1p(y_train)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_t)
        y_pred_t = model.predict(X_test)
        y_pred = np.expm1(y_pred_t)
        metrics = calculate_regression_metrics(y_test_original, y_pred)
        return metrics, y_pred, y_test_original, 'log1p'

    elif transform_type in ('yeo-johnson', 'box-cox'):
        pt = PowerTransformer(method=transform_type, standardize=False)
        pt.fit(y_train.reshape(-1, 1))
        y_train_t = pt.transform(y_train.reshape(-1, 1)).ravel()

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_t)
        y_pred_t = model.predict(X_test)
        y_pred = pt.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()

        metrics = calculate_regression_metrics(y_test_original, y_pred)
        return metrics, y_pred, y_test_original, pt


def _get_rmse(metrics):
    """Extract RMSE from metrics dict (handles various key formats)."""
    for k, v in metrics.items():
        if 'rmse' in k.lower():
            return v
    return None


class TestTargetTransform:
    """Target transformation tests — share skewed_splits and baseline_rmse fixtures."""

    def test_baseline_no_transform(self, skewed_splits):
        """Baseline metrics without transformation."""
        metrics, y_pred, y_test, _ = train_with_transform(skewed_splits, 'none')
        rmse = _get_rmse(metrics)

        assert rmse is not None, "RMSE not in metrics"
        assert rmse > 0, "RMSE should be positive"
        assert y_pred.min() < y_test.max(), "Predictions should overlap with test range"

    def test_log1p_transform(self, skewed_splits, baseline_rmse):
        """log1p: metrics must be on original scale, not log scale."""
        metrics, y_pred, y_test, _ = train_with_transform(skewed_splits, 'log1p')
        rmse = _get_rmse(metrics)

        assert rmse is not None and rmse > 0
        # If accidentally on log scale, RMSE would be ~0.5, not ~200+
        assert rmse > 10, f"RMSE suspiciously small ({rmse:.2f}) — might be on log scale"
        # Back-transformed predictions should be positive
        assert y_pred.min() > -2, f"Back-transformed predictions should be > -1, got {y_pred.min():.2f}"

    def test_yeojohnson_transform(self, skewed_splits, baseline_rmse):
        """Yeo-Johnson: verify back-transformation and roundtrip."""
        metrics, y_pred, y_test, pt = train_with_transform(skewed_splits, 'yeo-johnson')
        rmse = _get_rmse(metrics)

        assert rmse is not None
        assert rmse > 10, f"RMSE suspiciously small ({rmse:.2f}) — might be on transformed scale"
        assert hasattr(pt, 'lambdas_'), "PowerTransformer not fitted"

        # Roundtrip check
        y_sample = np.array([100.0, 500.0, 1000.0, 5000.0])
        y_t = pt.transform(y_sample.reshape(-1, 1)).ravel()
        y_back = pt.inverse_transform(y_t.reshape(-1, 1)).ravel()
        assert np.allclose(y_sample, y_back, atol=1e-4), "Roundtrip failed"

    def test_boxcox_positive_check(self, skewed_splits):
        """Box-Cox works on positive data and rejects negative values."""
        metrics, y_pred, y_test, pt = train_with_transform(skewed_splits, 'box-cox')
        rmse = _get_rmse(metrics)
        assert rmse is not None and rmse > 10, "Box-Cox RMSE should be on original scale"

        # Negative values should raise
        with pytest.raises(ValueError):
            pt_bad = PowerTransformer(method='box-cox', standardize=False)
            pt_bad.fit(np.array([-1.0, 0.0, 1.0, 5.0]).reshape(-1, 1))

    def test_cv_with_transform(self, skewed_splits):
        """Cross-validation with TransformedTargetRegressor reports original-scale RMSE."""
        from sklearn.model_selection import cross_val_score

        X_train = skewed_splits['X_train'].values if hasattr(skewed_splits['X_train'], 'values') else skewed_splits['X_train']
        y_train = skewed_splits['y_train'].values if hasattr(skewed_splits['y_train'], 'values') else skewed_splits['y_train']

        # log1p CV
        cv_model = TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        scores = cross_val_score(cv_model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
        rmse_scores = -scores

        assert len(rmse_scores) == 3
        assert all(s > 0 for s in rmse_scores)
        assert all(s > 10 for s in rmse_scores), "CV RMSE should be on original scale"

        # Yeo-Johnson CV
        cv_model_yj = TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0),
            transformer=PowerTransformer(method='yeo-johnson', standardize=False),
        )
        scores_yj = cross_val_score(cv_model_yj, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
        rmse_yj = -scores_yj
        assert all(s > 10 for s in rmse_yj), "Yeo-Johnson CV RMSE should be on original scale"

    def test_methods_section_reflects_transform(self):
        """Publication methods section includes transform details when applied."""
        from ml.publication import generate_methods_section

        base_kwargs = dict(
            data_config={'feature_cols': ['age', 'bmi'], 'target_col': 'cost', 'task_type': 'regression'},
            preprocessing_config={},
            model_configs={},
            n_total=300, n_train=210, n_val=45, n_test=45,
            feature_names=['age', 'bmi'],
            target_name='cost',
            task_type='regression',
            metrics_used=['RMSE', 'R²'],
        )

        for transform, expected_text in [
            ('log1p', 'log(1+x)'),
            ('yeo-johnson', 'Yeo-Johnson'),
            ('box-cox', 'Box-Cox'),
        ]:
            methods = generate_methods_section(
                **base_kwargs,
                split_config={
                    'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
                    'target_trim_enabled': False,
                    'target_transform': transform,
                    'stratify': False, 'use_time_split': False,
                },
            )
            assert expected_text in methods, f"'{expected_text}' not in methods for {transform}"
            assert 'back-transformed' in methods.lower() or 'original scale' in methods.lower(), \
                f"Back-transform not mentioned for {transform}"

        # 'none' should NOT mention any transform
        methods_none = generate_methods_section(
            data_config={'feature_cols': ['age'], 'target_col': 'cost', 'task_type': 'regression'},
            preprocessing_config={},
            model_configs={},
            split_config={
                'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
                'target_trim_enabled': False,
                'target_transform': 'none',
                'stratify': False, 'use_time_split': False,
            },
            n_total=300, n_train=210, n_val=45, n_test=45,
            feature_names=['age'],
            target_name='cost',
            task_type='regression',
            metrics_used=['RMSE'],
        )
        assert 'Yeo-Johnson' not in methods_none

    def test_edge_case_constant_target(self):
        """Near-constant target shouldn't crash transforms."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'target': 100.0 + np.random.normal(0, 0.001, n),
        })
        splits = prepare_splits(df, target_col='target')

        # log1p should not crash
        metrics, y_pred, y_test, _ = train_with_transform(splits, 'log1p')
        assert _get_rmse(metrics) is not None

        # Yeo-Johnson should not crash
        metrics, y_pred, y_test, pt = train_with_transform(splits, 'yeo-johnson')
        assert _get_rmse(metrics) is not None
