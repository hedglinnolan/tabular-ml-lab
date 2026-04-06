"""
Workflow test: Per-model preprocessing pipeline verification.

Validates the app's core differentiator — different models get different
preprocessing pipelines. Tests that:
1. Ridge (requires_scaled_numeric=True) gets scaling; RF (False) does not
2. Different pipeline configs produce different transformed outputs
3. Pipeline-to-training handoff works end-to-end with per-model data
4. Feature names after transform are tracked correctly per model
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.conftest import build_regression_df, make_data_config
from ml.pipeline import build_preprocessing_pipeline, get_feature_names_after_transform
from ml.model_registry import get_registry

MODEL_REGISTRY = get_registry()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """Synthetic regression dataset for pipeline tests."""
    return build_regression_df(n=200, seed=42, missing_rate=0.05)


@pytest.fixture(scope="module")
def feature_lists(sample_df):
    """Pre-computed numeric and categorical feature lists."""
    target = "glucose"
    numeric = [c for c in sample_df.columns
               if c != target and sample_df[c].dtype in ("float64", "int64")]
    categorical = [c for c in sample_df.columns
                   if c != target and sample_df[c].dtype == "object"]
    return {"numeric": numeric, "categorical": categorical, "target": target}


@pytest.fixture(scope="module")
def X_sample(sample_df, feature_lists):
    """Feature matrix for pipeline fitting."""
    cols = feature_lists["numeric"] + feature_lists["categorical"]
    return sample_df[cols].copy()


# ---------------------------------------------------------------------------
# Test: Different models get different pipeline configs
# ---------------------------------------------------------------------------

class TestPerModelPipelineConfigs:
    """Verify that model capabilities drive pipeline configuration."""

    def test_ridge_requires_scaling(self):
        """Ridge declares requires_scaled_numeric=True."""
        spec = MODEL_REGISTRY.get("ridge")
        assert spec is not None, "Ridge not in registry"
        assert spec.capabilities.requires_scaled_numeric is True

    def test_rf_does_not_require_scaling(self):
        """Random Forest declares requires_scaled_numeric=False."""
        spec = MODEL_REGISTRY.get("rf")
        assert spec is not None, "rf not in registry"
        assert spec.capabilities.requires_scaled_numeric is False

    def test_histgb_does_not_require_scaling(self):
        """HistGradientBoosting declares requires_scaled_numeric=False."""
        spec = MODEL_REGISTRY.get("histgb_reg")
        assert spec is not None, "histgb_reg not in registry"
        assert spec.capabilities.requires_scaled_numeric is False

    def test_registry_has_both_scaling_classes(self):
        """Registry has models requiring scaling and models that don't."""
        needs_scaling = [k for k, v in MODEL_REGISTRY.items()
                         if v.capabilities.requires_scaled_numeric]
        no_scaling = [k for k, v in MODEL_REGISTRY.items()
                      if not v.capabilities.requires_scaled_numeric]
        assert len(needs_scaling) >= 3, f"Expected >=3 models needing scaling, got {needs_scaling}"
        assert len(no_scaling) >= 3, f"Expected >=3 models not needing scaling, got {no_scaling}"


# ---------------------------------------------------------------------------
# Test: Different configs produce different transformed output
# ---------------------------------------------------------------------------

class TestPipelineOutputDifference:
    """Verify that distinct pipeline configs produce distinct outputs."""

    def test_scaled_vs_unscaled_outputs_differ(self, X_sample, feature_lists):
        """Standard-scaled output differs from unscaled output."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]

        pipe_scaled = build_preprocessing_pipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            numeric_scaling="standard",
        )
        pipe_unscaled = build_preprocessing_pipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            numeric_scaling="none",
        )
        pipe_scaled.fit(X_sample)
        pipe_unscaled.fit(X_sample)

        out_scaled = pipe_scaled.transform(X_sample)
        out_unscaled = pipe_unscaled.transform(X_sample)

        # Both produce arrays of the same shape
        assert out_scaled.shape == out_unscaled.shape, \
            f"Shape mismatch: {out_scaled.shape} vs {out_unscaled.shape}"

        # But the values differ (at least on numeric columns)
        if hasattr(out_scaled, "toarray"):
            out_scaled = out_scaled.toarray()
        if hasattr(out_unscaled, "toarray"):
            out_unscaled = out_unscaled.toarray()
        assert not np.allclose(out_scaled, out_unscaled, atol=1e-6), \
            "Scaled and unscaled outputs should differ"

    def test_robust_vs_standard_scaling_differ(self, X_sample, feature_lists):
        """Robust scaling produces different output than standard scaling."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]

        pipe_std = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="standard",
        )
        pipe_rob = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="robust",
        )
        pipe_std.fit(X_sample)
        pipe_rob.fit(X_sample)

        out_std = pipe_std.transform(X_sample)
        out_rob = pipe_rob.transform(X_sample)

        if hasattr(out_std, "toarray"):
            out_std = out_std.toarray()
        if hasattr(out_rob, "toarray"):
            out_rob = out_rob.toarray()

        assert not np.allclose(out_std, out_rob, atol=1e-6), \
            "Standard and robust scaling should produce different values"

    def test_power_transform_changes_output(self, X_sample, feature_lists):
        """Yeo-Johnson power transform changes the numeric output."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]

        pipe_none = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_power_transform="none",
        )
        pipe_yj = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_power_transform="yeo-johnson",
        )
        pipe_none.fit(X_sample)
        pipe_yj.fit(X_sample)

        out_none = pipe_none.transform(X_sample)
        out_yj = pipe_yj.transform(X_sample)

        if hasattr(out_none, "toarray"):
            out_none = out_none.toarray()
        if hasattr(out_yj, "toarray"):
            out_yj = out_yj.toarray()

        assert not np.allclose(out_none, out_yj, atol=1e-6), \
            "Power transform should change the output"


# ---------------------------------------------------------------------------
# Test: Per-model pipeline → training produces different results
# ---------------------------------------------------------------------------

class TestPipelineTrainingIntegration:
    """Verify that per-model pipelines feed different data to different models."""

    def test_ridge_and_rf_get_different_training_data(self, sample_df, X_sample, feature_lists):
        """When Ridge and RF each have their own pipeline, the data they
        train on is actually different."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]
        target = feature_lists["target"]

        # Ridge pipeline: standard scaling
        ridge_pipe = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="standard",
        )
        # RF pipeline: no scaling
        rf_pipe = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="none",
        )

        ridge_pipe.fit(X_sample)
        rf_pipe.fit(X_sample)

        X_ridge = ridge_pipe.transform(X_sample)
        X_rf = rf_pipe.transform(X_sample)

        if hasattr(X_ridge, "toarray"):
            X_ridge = X_ridge.toarray()
        if hasattr(X_rf, "toarray"):
            X_rf = X_rf.toarray()

        assert X_ridge.shape == X_rf.shape
        assert not np.allclose(X_ridge, X_rf, atol=1e-6), \
            "Ridge and RF should receive different preprocessed data"

    def test_models_trained_on_own_pipeline_produce_different_predictions(
        self, sample_df, X_sample, feature_lists
    ):
        """Train Ridge on scaled data and on unscaled data — predictions differ."""
        from sklearn.linear_model import Ridge

        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]
        target = feature_lists["target"]

        y = sample_df[target].values

        pipe_scaled = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="standard",
        )
        pipe_unscaled = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="none",
        )
        pipe_scaled.fit(X_sample)
        pipe_unscaled.fit(X_sample)

        X_s = pipe_scaled.transform(X_sample)
        X_u = pipe_unscaled.transform(X_sample)
        if hasattr(X_s, "toarray"):
            X_s = X_s.toarray()
        if hasattr(X_u, "toarray"):
            X_u = X_u.toarray()

        # Drop rows with NaN target
        mask = ~np.isnan(y)
        y_clean = y[mask]
        X_s_clean = X_s[mask]
        X_u_clean = X_u[mask]

        model_s = Ridge(alpha=1.0)
        model_u = Ridge(alpha=1.0)
        model_s.fit(X_s_clean, y_clean)
        model_u.fit(X_u_clean, y_clean)

        preds_s = model_s.predict(X_s_clean)
        preds_u = model_u.predict(X_u_clean)

        # Coefficients should differ due to different input scale
        assert not np.allclose(model_s.coef_, model_u.coef_, atol=1e-6), \
            "Coefficients should differ between scaled and unscaled inputs"

    def test_feature_names_tracked_per_model(self, X_sample, feature_lists):
        """Feature names after transform are correct and can differ per pipeline."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]
        all_features = numeric + categorical

        pipe_plain = build_preprocessing_pipeline(
            numeric_features=numeric, categorical_features=categorical,
            numeric_scaling="standard",
        )
        pipe_plain.fit(X_sample)

        names = get_feature_names_after_transform(pipe_plain, all_features)
        assert isinstance(names, list), "Should return a list"
        assert len(names) > 0, "Should have at least one feature name"
        # One-hot encoding expands categorical features
        assert len(names) >= len(numeric), \
            f"Expected at least {len(numeric)} features, got {len(names)}"


# ---------------------------------------------------------------------------
# Test: Pipeline persists in session-state dict correctly
# ---------------------------------------------------------------------------

class TestPipelineSessionStateContract:
    """Verify pipelines can be stored/retrieved as Page 05 → 06 expects."""

    def test_pipelines_by_model_dict_contract(self, X_sample, feature_lists):
        """Build a per-model dict matching the session_state contract."""
        numeric = feature_lists["numeric"]
        categorical = feature_lists["categorical"]

        pipelines_by_model = {}
        configs_by_model = {}

        for model_key, scaling in [("ridge", "standard"), ("rf", "none")]:
            pipe = build_preprocessing_pipeline(
                numeric_features=numeric, categorical_features=categorical,
                numeric_scaling=scaling,
            )
            pipe.fit(X_sample)
            pipelines_by_model[model_key] = pipe
            configs_by_model[model_key] = {
                "numeric_features": numeric,
                "categorical_features": categorical,
                "numeric_scaling": scaling,
            }

        # Contract: dict keyed by model_key, values are fitted Pipelines
        assert "ridge" in pipelines_by_model
        assert "rf" in pipelines_by_model
        assert hasattr(pipelines_by_model["ridge"], "transform")
        assert hasattr(pipelines_by_model["rf"], "transform")

        # Configs track what was applied
        assert configs_by_model["ridge"]["numeric_scaling"] == "standard"
        assert configs_by_model["rf"]["numeric_scaling"] == "none"

        # Both pipelines can transform the same input
        out_ridge = pipelines_by_model["ridge"].transform(X_sample)
        out_rf = pipelines_by_model["rf"].transform(X_sample)
        assert out_ridge.shape[0] == out_rf.shape[0] == len(X_sample)
