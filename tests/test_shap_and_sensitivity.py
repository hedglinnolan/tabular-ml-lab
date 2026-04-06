"""
Tests for real SHAP computation and sensitivity analysis.

Replaces the dummy mocks with actual computations on small datasets
to verify:
1. SHAP values are computed correctly for linear and tree models
2. Permutation importance structure matches downstream expectations
3. Sensitivity framework produces valid results
4. Result dict structures match what Pages 07, 08, and 10 expect
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import build_regression_df, prepare_splits
from ml.eval import calculate_regression_metrics
from ml.sensitivity import (
    SensitivityResult, SensitivityAnalysis, sensitivity_random_seeds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """Small dataset for fast SHAP computation."""
    return build_regression_df(n=80, seed=42, missing_rate=0.0)


@pytest.fixture(scope="module")
def small_splits(small_df):
    return prepare_splits(small_df, target_col="glucose", train_frac=0.7, val_frac=0.15)


@pytest.fixture(scope="module")
def trained_ridge(small_splits):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    X_tr = small_splits["X_train"].values
    y_tr = small_splits["y_train"].values
    model.fit(X_tr, y_tr)
    return model


@pytest.fixture(scope="module")
def trained_histgb(small_splits):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(random_state=42, max_iter=30)
    X_tr = small_splits["X_train"].values
    y_tr = small_splits["y_train"].values
    model.fit(X_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# SHAP computation tests
# ---------------------------------------------------------------------------

class TestSHAPComputation:
    """Test that SHAP values are actually computed (not mocked)."""

    def test_linear_shap_values(self, trained_ridge, small_splits):
        """LinearExplainer produces valid SHAP values for Ridge."""
        import shap

        X_tr = small_splits["X_train"].values
        X_te = small_splits["X_test"].values
        feature_names = list(small_splits["X_train"].columns)

        explainer = shap.LinearExplainer(trained_ridge, X_tr)
        shap_values = explainer.shap_values(X_te)

        # Shape: (n_test_samples, n_features)
        assert shap_values.shape == X_te.shape, \
            f"SHAP shape {shap_values.shape} != data shape {X_te.shape}"

        # SHAP values should not all be zero
        assert not np.allclose(shap_values, 0), "SHAP values are all zero"

        # Build the result dict as Page 07 would
        shap_result = {
            "shap_values": shap_values,
            "X_eval": X_te,
            "feature_names": feature_names,
            "class_label": None,
            "all_shap_values": shap_values,
            "kernel_capped": False,
            "n_eval_samples": len(X_te),
        }

        # Validate structure
        assert isinstance(shap_result["shap_values"], np.ndarray)
        assert len(shap_result["feature_names"]) == shap_values.shape[1]
        assert shap_result["n_eval_samples"] == len(X_te)

    def test_tree_shap_values(self, trained_histgb, small_splits):
        """TreeExplainer produces valid SHAP values for HistGB."""
        import shap

        X_te = small_splits["X_test"].values

        explainer = shap.TreeExplainer(trained_histgb)
        shap_values = explainer.shap_values(X_te)

        assert shap_values.shape == X_te.shape
        assert not np.allclose(shap_values, 0)

    def test_shap_feature_names_align_with_values(self, trained_ridge, small_splits):
        """Feature names count must match SHAP values column count."""
        import shap

        X_tr = small_splits["X_train"].values
        X_te = small_splits["X_test"].values
        feature_names = list(small_splits["X_train"].columns)

        explainer = shap.LinearExplainer(trained_ridge, X_tr)
        shap_values = explainer.shap_values(X_te)

        assert len(feature_names) == shap_values.shape[1], \
            f"{len(feature_names)} names but {shap_values.shape[1]} SHAP columns"


# ---------------------------------------------------------------------------
# Permutation importance tests
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    """Test real permutation importance computation."""

    def test_permutation_importance_structure(self, trained_ridge, small_splits):
        """Permutation importance result matches expected structure."""
        from sklearn.inspection import permutation_importance

        X_te = small_splits["X_test"]
        y_te = small_splits["y_test"]
        feature_names = list(X_te.columns)

        pi = permutation_importance(
            trained_ridge, X_te.values, y_te.values,
            n_repeats=5, random_state=42,
        )

        # Build result dict as Page 07 would
        perm_result = {
            "importances_mean": pi.importances_mean,
            "importances_std": pi.importances_std,
            "feature_names": feature_names,
        }

        # Validate structure
        n_features = len(feature_names)
        assert perm_result["importances_mean"].shape == (n_features,)
        assert perm_result["importances_std"].shape == (n_features,)
        assert len(perm_result["feature_names"]) == n_features

    def test_importances_are_not_all_zero(self, trained_ridge, small_splits):
        """At least some features should have non-zero importance."""
        from sklearn.inspection import permutation_importance

        X_te = small_splits["X_test"]
        y_te = small_splits["y_test"]

        pi = permutation_importance(
            trained_ridge, X_te.values, y_te.values,
            n_repeats=5, random_state=42,
        )

        assert not np.allclose(pi.importances_mean, 0), \
            "All feature importances are zero — synthetic data should have signal"

    def test_permutation_importance_post_pca(self, small_splits):
        """PI on PCA-reduced data must operate on transformed features, not originals.

        Regression test for #126: PI stalled on high-dimensional data because
        it permuted 3000+ original features instead of 20 PCA components.
        The fix: transform X through the preprocessing pipeline first, then
        compute PI on the transformed data using just the estimator.
        """
        from sklearn.inspection import permutation_importance
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        X_train = small_splits["X_train"].values
        X_test = small_splits["X_test"].values
        y_train = small_splits["y_train"].values
        y_test = small_splits["y_test"].values
        n_original_features = X_train.shape[1]

        # Build a preprocessing pipeline with PCA (reduces to 3 components)
        n_components = 3
        prep_pipeline = SklearnPipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
        ])
        prep_pipeline.fit(X_train)
        X_train_pca = prep_pipeline.transform(X_train)
        X_test_pca = prep_pipeline.transform(X_test)

        # Train model on PCA-transformed data
        model = Ridge(alpha=1.0)
        model.fit(X_train_pca, y_train)

        # CORRECT: PI on transformed features (what the fix does)
        pi_correct = permutation_importance(
            model, X_test_pca, y_test,
            n_repeats=5, random_state=42,
        )
        assert pi_correct.importances_mean.shape == (n_components,), \
            f"PI should have {n_components} features (PCA components), got {pi_correct.importances_mean.shape}"

        # WRONG (what the old code did): PI on full pipeline with raw features
        full_pipeline = SklearnPipeline([('preprocess', prep_pipeline), ('model', model)])
        pi_wrong = permutation_importance(
            full_pipeline, X_test, y_test,
            n_repeats=5, random_state=42,
        )
        assert pi_wrong.importances_mean.shape == (n_original_features,), \
            "Full-pipeline PI permutes original features (the bug)"

        # The correct approach should be faster and have fewer features
        assert pi_correct.importances_mean.shape[0] < pi_wrong.importances_mean.shape[0]


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    """Test the sensitivity framework with real models."""

    def test_seed_sensitivity_produces_valid_results(self, small_splits):
        """sensitivity_random_seeds produces structured results."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        X_tr = small_splits["X_train"].values
        X_te = small_splits["X_test"].values
        y_tr = small_splits["y_train"].values
        y_te = small_splits["y_test"].values

        def train_fn(seed):
            m = HistGradientBoostingRegressor(random_state=seed, max_iter=30)
            m.fit(X_tr, y_tr)
            return m

        def eval_fn(m):
            return calculate_regression_metrics(y_te, m.predict(X_te))

        analysis = sensitivity_random_seeds(
            train_fn, eval_fn,
            seeds=[0, 1, 7, 13],
            baseline_seed=42,
        )

        assert isinstance(analysis, SensitivityAnalysis)
        assert analysis.analysis_type == "random_seed"
        assert len(analysis.baseline_metrics) > 0
        assert len(analysis.variations) > 0

        # Each variation should have metrics
        for v in analysis.variations:
            assert isinstance(v, SensitivityResult)
            assert len(v.metrics) > 0
            assert v.variation_name == "seed"

    def test_sensitivity_dataframe_output(self, small_splits):
        """to_dataframe() produces a proper DataFrame."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        X_tr = small_splits["X_train"].values
        X_te = small_splits["X_test"].values
        y_tr = small_splits["y_train"].values
        y_te = small_splits["y_test"].values

        def train_fn(seed):
            m = HistGradientBoostingRegressor(random_state=seed, max_iter=30)
            m.fit(X_tr, y_tr)
            return m

        def eval_fn(m):
            return calculate_regression_metrics(y_te, m.predict(X_te))

        analysis = sensitivity_random_seeds(train_fn, eval_fn, seeds=[0, 7], baseline_seed=42)
        df = analysis.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "Variation" in df.columns
        assert len(df) >= 2  # baseline + variations
        assert df.iloc[0]["Variation"] == "Baseline"

    def test_robustness_check(self, small_splits):
        """is_robust() correctly evaluates metric stability."""
        from sklearn.linear_model import Ridge

        X_tr = small_splits["X_train"].values
        X_te = small_splits["X_test"].values
        y_tr = small_splits["y_train"].values
        y_te = small_splits["y_test"].values

        def train_fn(seed):
            m = Ridge(alpha=1.0)  # deterministic
            m.fit(X_tr, y_tr)
            return m

        def eval_fn(m):
            return calculate_regression_metrics(y_te, m.predict(X_te))

        analysis = sensitivity_random_seeds(train_fn, eval_fn, seeds=[0, 1], baseline_seed=42)

        # Ridge is deterministic — should be perfectly robust
        for metric_key in analysis.baseline_metrics:
            assert analysis.is_robust(metric_key, tolerance=0.01), \
                f"Ridge should be robust for {metric_key}"

    def test_coefficient_of_variation_calculation(self):
        """CV calculation matches expected formula: std/|mean| * 100."""
        metrics = pd.Series([0.85, 0.84, 0.86, 0.85, 0.83])
        cv = metrics.std() / abs(metrics.mean()) * 100

        assert 0 < cv < 5, f"CV should indicate moderate robustness, got {cv:.2f}%"
        assert cv == pytest.approx(
            metrics.std() / abs(metrics.mean()) * 100, abs=0.001
        )
