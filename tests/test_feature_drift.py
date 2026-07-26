"""Regression tests for the feature-engineering column-drift bug class.

A user hit `Training failed: Some column names are not columns of the
dataframe: {'meds_hbp_has_data', 'meds_chol_has_data'}` — an engineered
missingness indicator was created, a preprocessing pipeline was built naming
it, then Feature Selection dropped that low-signal column, leaving the stored
pipeline referencing a column no longer in the data. It only surfaced at
`ColumnTransformer.fit()`.

These pin the fixes:
- ml.pipeline.reconcile_pipeline_columns rebuilds a pipeline against the
  columns that actually exist (the training backstop).
- reset_downstream_results(clear_feature_selection=False) invalidates stale
  pipelines/splits/models when Feature Selection is applied, without wiping
  the selection just made.
- reset_downstream_results(clear_feature_engineering=True) restores the
  pre-FE feature list (the Skip/Reset fix).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _clear_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ── The training backstop: reconcile a stale pipeline against real columns ──

class TestReconcilePipelineColumns:
    def _stale_pipeline(self):
        from ml.pipeline import build_preprocessing_pipeline
        # Built when the engineered indicator still existed
        return build_preprocessing_pipeline(
            numeric_features=["age", "bmi", "meds_hbp_has_data"],
            categorical_features=[],
            random_state=42,
        )

    def test_stale_pipeline_crashes_without_reconcile(self):
        """Documents the exact reported failure so the fix stays honest."""
        X_train = pd.DataFrame({"age": [40.0, 55, 60, 33, 70],
                                "bmi": [22.1, 28.4, 31.0, 24.5, 26.6]})
        with pytest.raises(ValueError, match="not columns of the dataframe"):
            self._stale_pipeline().fit(X_train)

    def test_reconcile_drops_missing_and_fits(self):
        from ml.pipeline import reconcile_pipeline_columns
        X_train = pd.DataFrame({"age": [40.0, 55, 60, 33, 70],
                                "bmi": [22.1, 28.4, 31.0, 24.5, 26.6]})
        healed, dropped = reconcile_pipeline_columns(self._stale_pipeline(),
                                                     X_train.columns)
        assert dropped == ["meds_hbp_has_data"]
        # The reported crash no longer happens; the surviving columns transform.
        out = healed.fit_transform(X_train)
        assert out.shape == (5, 2)

    def test_reconcile_is_noop_when_columns_intact(self):
        from ml.pipeline import reconcile_pipeline_columns
        from ml.pipeline import build_preprocessing_pipeline
        pipe = build_preprocessing_pipeline(numeric_features=["age", "bmi"],
                                            categorical_features=[], random_state=42)
        same, dropped = reconcile_pipeline_columns(pipe, ["age", "bmi", "extra"])
        assert same is pipe and dropped == []


# ── The invalidation contracts ──

class TestFeatureChangeInvalidation:
    def setup_method(self):
        _clear_session()

    def teardown_method(self):
        _clear_session()

    def test_applying_feature_selection_clears_stale_pipeline(self):
        """Feature Selection applied → stored preprocessing pipelines/splits/
        models must be cleared (they named the old feature set), but the
        selection results themselves must survive."""
        from utils.session_state import reset_downstream_results

        st.session_state["preprocessing_pipelines_by_model"] = {"ridge": "STALE"}
        st.session_state["preprocessing_config_by_model"] = {"ridge": "STALE"}
        st.session_state["X_train"] = "STALE"
        st.session_state["trained_models"] = {"ridge": "STALE"}
        st.session_state["feature_selection_results"] = {"keep": "ME"}
        st.session_state["consensus_features"] = ["age", "bmi"]

        reset_downstream_results(clear_feature_engineering=False,
                                 clear_feature_selection=False)

        # Stale modeling artifacts gone
        assert st.session_state.get("preprocessing_pipelines_by_model") == {}
        assert st.session_state.get("preprocessing_config_by_model") == {}
        assert st.session_state.get("X_train") is None
        assert st.session_state.get("trained_models") == {}
        # The selection just applied survives
        assert st.session_state.get("feature_selection_results") == {"keep": "ME"}
        assert st.session_state.get("consensus_features") == ["age", "bmi"]

    def test_default_reset_still_clears_feature_selection(self):
        """The data-change path (default flags) must keep clearing FS results."""
        from utils.session_state import reset_downstream_results
        st.session_state["feature_selection_results"] = {"marker": "STALE"}
        st.session_state["consensus_features"] = ["x"]
        reset_downstream_results()
        assert st.session_state.get("feature_selection_results") is None
        assert st.session_state.get("consensus_features") is None

    def test_skip_reset_restores_pre_fe_features(self):
        """Skip/Reset route through reset_downstream_results(clear_feature_
        engineering=True), which must restore the pre-FE selection into
        selected_features (Reset previously left the engineered list, so
        downstream still referenced dropped columns)."""
        from utils.session_state import reset_downstream_results

        st.session_state["pre_fe_feature_cols"] = ["age", "bmi"]
        st.session_state["selected_features"] = ["age", "bmi", "meds_hbp_has_data"]
        st.session_state["df_engineered"] = pd.DataFrame({"age": [1.0]})
        st.session_state["feature_engineering_applied"] = True

        reset_downstream_results(clear_feature_engineering=True)

        assert st.session_state.get("selected_features") == ["age", "bmi"]
        assert st.session_state.get("df_engineered") is None
        assert st.session_state.get("feature_engineering_applied") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
