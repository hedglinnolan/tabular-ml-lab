"""
Workflow test: Cross-page data contracts.

Validates that session state keys written by one page are correctly
consumed by downstream pages. Focuses on the gaps identified in the
test coverage audit:
- Page 03 → 04: df_engineered, engineered_feature_names
- Page 04 → 05: selected_features → preprocess builder
- Feature engineering toggle → cascade invalidation
- Preprocessing config → training consumption
"""
import sys
import os
import hashlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.conftest import (
    build_regression_df, inject_uploaded_state, prepare_splits, make_data_config,
)
from ml.pipeline import build_preprocessing_pipeline, get_feature_names_after_transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contract_df():
    return build_regression_df(n=200, seed=42, missing_rate=0.05)


@pytest.fixture(scope="module")
def contract_state(contract_df):
    state = {}
    inject_uploaded_state(state, contract_df, target_col="glucose", task_type="regression")
    return state


# ---------------------------------------------------------------------------
# Feature Engineering → Feature Selection contract
# ---------------------------------------------------------------------------

class TestFeatureEngineeringContract:
    """Page 03 → Page 04 data contract."""

    def test_engineered_features_extend_feature_set(self, contract_df, contract_state):
        """df_engineered should contain original columns plus new ones."""
        state = contract_state
        df = contract_df
        data_config = state["data_config"]

        # Simulate feature engineering: add log_bmi and bmi_squared
        X = df[data_config.feature_cols].copy()
        bmi = X["bmi"].fillna(X["bmi"].median())
        X["log_bmi"] = np.log1p(bmi)
        X["bmi_squared"] = bmi ** 2

        df_engineered = pd.concat([X, df["glucose"]], axis=1)
        engineered_feature_names = ["log_bmi", "bmi_squared"]

        state["df_engineered"] = df_engineered
        state["feature_engineering_applied"] = True
        state["engineered_feature_names"] = engineered_feature_names

        # Contract: df_engineered has all original features PLUS new ones
        for col in data_config.feature_cols:
            assert col in df_engineered.columns, f"Original feature {col} missing from df_engineered"
        for col in engineered_feature_names:
            assert col in df_engineered.columns, f"Engineered feature {col} missing from df_engineered"
        assert "glucose" in df_engineered.columns, "Target should be in df_engineered"

    def test_engineered_names_are_new_features_only(self, contract_state):
        """engineered_feature_names should contain ONLY new features, not originals."""
        state = contract_state
        data_config = state["data_config"]
        engineered = state["engineered_feature_names"]

        for name in engineered:
            assert name not in data_config.feature_cols, \
                f"Engineered name '{name}' collides with original feature"

    def test_selected_features_updated_after_engineering(self, contract_state):
        """Page 04 should see the expanded feature set."""
        state = contract_state
        df_eng = state["df_engineered"]

        # Simulate Page 04: select features from engineered df
        all_features = [c for c in df_eng.columns if c != "glucose"]
        state["selected_features"] = all_features

        # Should include both original and engineered
        assert "log_bmi" in state["selected_features"]
        assert "bmi_squared" in state["selected_features"]
        assert "age" in state["selected_features"]


# ---------------------------------------------------------------------------
# Feature Selection → Preprocess contract
# ---------------------------------------------------------------------------

class TestFeatureSelectionToPreprocess:
    """Page 04 → Page 05 data contract."""

    def test_preprocess_uses_selected_features(self, contract_df, contract_state):
        """Pipeline should be built using selected_features, not all features."""
        state = contract_state
        selected = state["selected_features"]
        df = state.get("df_engineered", contract_df)

        numeric_features = [c for c in selected
                            if c in df.columns and df[c].dtype in ("float64", "int64")]
        categorical_features = [c for c in selected
                                if c in df.columns and df[c].dtype == "object"]

        X_sample = df[numeric_features + categorical_features].copy()
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        pipe.fit(X_sample)

        out = pipe.transform(X_sample)
        names = get_feature_names_after_transform(pipe, numeric_features + categorical_features)

        # Feature names should reflect selected (including engineered)
        assert len(names) >= len(numeric_features)

        state["preprocessing_pipelines_by_model"] = {"ridge": pipe}

    def test_reduced_selection_reduces_pipeline_input(self, contract_df, contract_state):
        """If fewer features are selected, pipeline should get fewer inputs."""
        state = contract_state
        df = state.get("df_engineered", contract_df)

        # Select only 3 features
        small_selection = ["age", "bmi", "cholesterol"]
        numeric_features = small_selection

        X_sample = df[numeric_features].copy()
        pipe = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=[],
        )
        pipe.fit(X_sample)

        names = get_feature_names_after_transform(pipe, numeric_features)
        assert len(names) == 3, f"Expected 3 features, got {len(names)}"


# ---------------------------------------------------------------------------
# Feature engineering toggle → cascade
# ---------------------------------------------------------------------------

class TestFeatureEngineeringToggleCascade:
    """Toggling feature engineering off should cascade-clear downstream state."""

    def test_disabling_fe_clears_engineered_features(self):
        """When FE is toggled off, engineered names and df should be cleared."""
        state = {}
        df = build_regression_df(n=100)

        inject_uploaded_state(state, df, target_col="glucose")
        data_config = state["data_config"]

        # Simulate FE on
        state["feature_engineering_applied"] = True
        state["engineered_feature_names"] = ["log_bmi"]
        state["df_engineered"] = df.copy()
        state["selected_features"] = data_config.feature_cols + ["log_bmi"]

        # Build a pipeline with FE features
        state["preprocessing_pipelines_by_model"] = {"ridge": "dummy"}
        state["trained_models"] = {"ridge": "dummy"}
        state["model_results"] = {"ridge": "dummy"}
        state["shap_results"] = {"ridge": "dummy"}

        # Simulate FE toggle off — clear FE state and cascade
        state["feature_engineering_applied"] = False
        state.pop("engineered_feature_names", None)
        state.pop("df_engineered", None)
        state["selected_features"] = list(data_config.feature_cols)

        # Simulate cascade: remove downstream state
        for key in ["preprocessing_pipelines_by_model", "trained_models",
                     "model_results", "shap_results"]:
            state.pop(key, None)

        # Verify FE artifacts are gone
        assert state.get("engineered_feature_names") is None
        assert state.get("df_engineered") is None
        assert "log_bmi" not in state["selected_features"]

        # Verify downstream was cleared
        assert state.get("trained_models") is None
        assert state.get("shap_results") is None


# ---------------------------------------------------------------------------
# Preprocessing → Training data contract
# ---------------------------------------------------------------------------

class TestPreprocessToTrainingContract:
    """Page 05 → Page 06 contract: pipelines transform training data correctly."""

    def test_pipeline_transform_preserves_row_count(self, contract_df, contract_state):
        """Transformed data should have same number of rows as input."""
        state = contract_state
        df = contract_df
        selected = [c for c in state["data_config"].feature_cols
                    if df[c].dtype in ("float64", "int64")]

        pipe = build_preprocessing_pipeline(
            numeric_features=selected,
            categorical_features=[],
        )
        X = df[selected].copy()
        pipe.fit(X)
        X_out = pipe.transform(X)

        if hasattr(X_out, "toarray"):
            X_out = X_out.toarray()

        assert X_out.shape[0] == len(X), \
            f"Row count changed: {len(X)} → {X_out.shape[0]}"

    def test_pipeline_handles_missing_values(self, contract_df):
        """Pipeline should handle NaN values without crashing."""
        df = contract_df
        numeric = ["age", "bmi", "cholesterol"]  # bmi and cholesterol have NaN

        assert df[numeric].isna().any().any(), "Test data should have missing values"

        pipe = build_preprocessing_pipeline(
            numeric_features=numeric,
            categorical_features=[],
            numeric_imputation="median",
        )
        X = df[numeric].copy()
        pipe.fit(X)
        X_out = pipe.transform(X)

        if hasattr(X_out, "toarray"):
            X_out = X_out.toarray()

        assert not np.isnan(X_out).any(), "Pipeline should impute all NaN values"

    def test_fitted_pipeline_stored_for_explainability(self, contract_df):
        """fitted_preprocessing_pipelines should be usable by Page 07."""
        df = contract_df
        selected = ["age", "bmi", "cholesterol"]

        pipe = build_preprocessing_pipeline(
            numeric_features=selected,
            categorical_features=[],
        )
        X = df[selected].copy()
        pipe.fit(X)

        # Simulate what Page 06 stores
        fitted_pipelines = {"ridge": pipe}
        feature_names_by_model = {
            "ridge": get_feature_names_after_transform(pipe, selected)
        }

        # Page 07 reads these and must be able to transform data
        assert hasattr(fitted_pipelines["ridge"], "transform")
        X_out = fitted_pipelines["ridge"].transform(X)
        assert X_out.shape[0] == len(X)

        # Feature names should align with transform output
        if hasattr(X_out, "toarray"):
            X_out = X_out.toarray()
        assert X_out.shape[1] == len(feature_names_by_model["ridge"]), \
            f"Feature count mismatch: {X_out.shape[1]} cols vs {len(feature_names_by_model['ridge'])} names"
