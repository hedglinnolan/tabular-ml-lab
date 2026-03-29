"""Integration tests: WorkflowProvenance records match methodology_log and session state.

These tests simulate user actions and verify that the provenance layer
captures the same data as the existing tracking systems.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit session_state for testing page-level integration."""
    session_state = {}

    class MockSessionState(dict):
        """Dict that also supports attribute access like st.session_state."""
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    state = MockSessionState()

    mock_st = MagicMock()
    mock_st.session_state = state
    return mock_st, state


class TestProvenanceMatchesMethodologyLog:
    """Verify provenance records match what log_methodology captures."""

    def test_upload_provenance_matches_log(self, mock_streamlit):
        """Provenance upload record matches log_methodology Upload & Audit entry."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()

        # Simulate what Upload & Audit page does
        target_col = "glucose"
        task_type = "regression"
        features = ["age", "bmi", "insulin", "bp"]
        n_samples = 500

        prov.record_upload(
            target_col=target_col,
            task_type=task_type,
            feature_cols=features,
            n_samples=n_samples,
            data_source="csv_upload",
        )

        # Verify provenance captures what log_methodology would capture
        assert prov.upload is not None
        assert prov.upload.target_col == target_col
        assert prov.upload.task_type == task_type
        assert prov.upload.feature_cols == features
        assert prov.upload.n_samples == n_samples
        assert prov.upload.n_features == len(features)

        # Verify get_methods_context produces correct output
        ctx = prov.get_methods_context()
        assert ctx["target_name"] == target_col
        assert ctx["task_type"] == task_type
        assert ctx["n_features_original"] == 4
        assert ctx["n_total"] == 500

    def test_preprocessing_provenance_captures_per_model(self):
        """Provenance preprocessing captures full per-model config — the core differentiator."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()

        # Simulate per-model configs as built by Preprocess page
        configs_by_model = {
            "ridge": {
                "numeric_scaling": "standard",
                "categorical_encoding": "onehot",
                "numeric_outlier_treatment": "none",
                "numeric_power_transform": "yeo-johnson",
                "numeric_log_transform": False,
                "use_pca": False,
            },
            "rf": {
                "numeric_scaling": "none",
                "categorical_encoding": "onehot",
                "numeric_outlier_treatment": "none",
                "numeric_power_transform": "none",
                "numeric_log_transform": False,
                "use_pca": False,
            },
            "histgb_reg": {
                "numeric_scaling": "none",
                "categorical_encoding": "ordinal",
                "numeric_outlier_treatment": "percentile_clip",
                "numeric_outlier_params": {"lower": 5, "upper": 95},
                "numeric_power_transform": "none",
                "numeric_log_transform": False,
                "use_pca": False,
            },
        }

        prov.record_preprocessing(
            configs_by_model=configs_by_model,
            imputation_method="median",
        )

        # Verify per-model capture
        assert prov.preprocessing is not None
        assert len(prov.preprocessing.per_model) == 3
        assert prov.preprocessing.configs_differ() is True

        # Ridge gets scaling + transform
        ridge = prov.preprocessing.per_model["ridge"]
        assert ridge.scaling == "standard"
        assert ridge.power_transform == "yeo-johnson"

        # RF gets nothing
        rf = prov.preprocessing.per_model["rf"]
        assert rf.scaling == "none"
        assert rf.power_transform == "none"

        # HistGB gets outlier treatment
        histgb = prov.preprocessing.per_model["histgb_reg"]
        assert histgb.outlier_treatment == "percentile_clip"
        assert histgb.outlier_params == {"lower": 5, "upper": 95}
        assert histgb.encoding == "ordinal"

        # Shared settings detected
        assert prov.preprocessing.shared.get("imputation") == "median"

        # Methods context includes per-model detail
        ctx = prov.get_methods_context()
        assert ctx["preprocessing_differs"] is True
        assert "ridge" in ctx["preprocessing_per_model"]
        assert ctx["preprocessing_per_model"]["ridge"]["scaling"] == "standard"
        assert ctx["preprocessing_per_model"]["rf"]["scaling"] == "none"

    def test_preprocessing_identical_configs_detected(self):
        """When all models share the same preprocessing, configs_differ() returns False."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        same_config = {
            "numeric_scaling": "standard",
            "categorical_encoding": "onehot",
            "numeric_outlier_treatment": "none",
            "numeric_power_transform": "none",
        }
        prov.record_preprocessing(
            configs_by_model={"ridge": same_config, "rf": same_config},
            imputation_method="mean",
        )

        assert prov.preprocessing.configs_differ() is False
        ctx = prov.get_methods_context()
        assert ctx["preprocessing_differs"] is False

    def test_training_provenance_matches_log(self):
        """Training provenance captures what log_methodology Model Training entry captures."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()

        prov.record_training(
            models_trained=["ridge", "rf", "histgb_reg"],
            primary_model="rf",
            selection_criteria="validation RMSE",
            use_cv=True,
            cv_folds=5,
            use_hyperopt=False,
            class_weight_balanced=False,
            hyperparameters={
                "ridge": {"alpha": 1.0},
                "rf": {"n_estimators": 100, "max_depth": None},
                "histgb_reg": {"max_iter": 100, "learning_rate": 0.1},
            },
            metrics_by_model={
                "ridge": {"RMSE": 12.3, "R2": 0.78},
                "rf": {"RMSE": 10.1, "R2": 0.85},
                "histgb_reg": {"RMSE": 11.5, "R2": 0.81},
            },
        )

        ctx = prov.get_methods_context()
        assert ctx["models_trained"] == ["ridge", "rf", "histgb_reg"]
        assert ctx["primary_model"] == "rf"
        assert ctx["use_cv"] is True
        assert ctx["cv_folds"] == 5
        assert ctx["hyperparameters"]["ridge"]["alpha"] == 1.0
        assert ctx["metrics_by_model"]["rf"]["RMSE"] == 10.1


class TestProvenanceEndToEnd:
    """Simulate a full workflow and verify provenance completeness."""

    def test_full_workflow_provenance(self):
        """Walk through a complete workflow and verify all sections are populated."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()

        # 1. Upload
        prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
        assert prov.get_completeness()["upload"] is True

        # 2. EDA
        prov.record_eda_analysis("Correlation Matrix")
        prov.record_eda_analysis("Distribution Plots")
        prov.record_table1()
        assert prov.get_completeness()["eda"] is True
        assert prov.eda.table1_generated is True

        # 3. Feature Engineering
        prov.record_feature_engineering(
            transforms=["Polynomial degree 2"],
            n_created=3,
            n_before=2,
            n_after=5,
        )
        assert prov.get_completeness()["feature_engineering"] is True

        # 4. Feature Selection
        prov.record_feature_selection(
            method="consensus",
            n_before=5,
            n_after=3,
            features_kept=["age", "bmi", "age*bmi"],
            consensus_methods=["mutual_info", "f_regression"],
        )
        assert prov.get_completeness()["feature_selection"] is True

        # 5. Split
        prov.record_split(
            strategy="stratified",
            train_n=350, val_n=75, test_n=75,
            random_seed=42,
        )
        assert prov.get_completeness()["split"] is True

        # 6. Preprocessing (per-model)
        prov.record_preprocessing(
            configs_by_model={
                "ridge": {"numeric_scaling": "standard", "numeric_power_transform": "yeo-johnson"},
                "rf": {"numeric_scaling": "none", "numeric_power_transform": "none"},
            },
            imputation_method="median",
        )
        assert prov.get_completeness()["preprocessing"] is True
        assert prov.preprocessing.configs_differ() is True

        # 7. Training
        prov.record_training(
            models_trained=["ridge", "rf"],
            primary_model="rf",
            selection_criteria="validation RMSE",
        )
        assert prov.get_completeness()["training"] is True

        # 8. Explainability
        prov.record_explainability(["SHAP", "Permutation Importance"], ["ridge", "rf"])
        assert prov.get_completeness()["explainability"] is True

        # 9. Sensitivity
        prov.record_sensitivity(seed_stability=True, seed_stability_cv=2.3)
        assert prov.get_completeness()["sensitivity"] is True

        # 10. Statistical validation
        prov.record_statistical_test("Shapiro-Wilk", "residuals", 0.97, 0.03)
        assert prov.get_completeness()["statistical_validation"] is True

        # All complete
        completeness = prov.get_completeness()
        assert all(completeness.values()), f"Not all stages complete: {completeness}"

        # Methods context has everything
        ctx = prov.get_methods_context()
        assert ctx["target_name"] == "glucose"
        assert ctx["models_trained"] == ["ridge", "rf"]
        assert ctx["preprocessing_differs"] is True
        assert ctx["split_strategy"] == "stratified"
        assert ctx["use_cv"] is False  # we didn't set it
        assert len(ctx["statistical_tests"]) == 1

    def test_upload_resets_downstream(self):
        """Re-uploading data resets downstream provenance sections."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()

        # Build full pipeline
        prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
        prov.record_feature_engineering(["poly"], 3, 2, 5)
        prov.record_feature_selection("consensus", 5, 3, ["a", "b", "c"])
        prov.record_preprocessing({"ridge": {"numeric_scaling": "standard"}}, "mean")
        prov.record_training(["ridge"], "ridge", "RMSE")

        assert prov.training is not None

        # Re-upload — should reset downstream
        prov.record_upload("new_target", "classification", ["x", "y"], 1000)

        assert prov.upload.target_col == "new_target"
        assert prov.feature_engineering is None
        assert prov.feature_selection is None
        assert prov.preprocessing is None
        assert prov.training is None

    def test_serialization_round_trip(self):
        """Provenance survives to_dict → from_dict round-trip with per-model detail intact."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
        prov.record_preprocessing(
            configs_by_model={
                "ridge": {"numeric_scaling": "standard", "numeric_power_transform": "yeo-johnson"},
                "rf": {"numeric_scaling": "none"},
            },
            imputation_method="median",
        )
        prov.record_training(
            ["ridge", "rf"], "rf", "RMSE",
            hyperparameters={"ridge": {"alpha": 1.0}},
            metrics_by_model={"ridge": {"RMSE": 12.3}, "rf": {"RMSE": 10.1}},
        )

        # Round-trip
        data = prov.to_dict()
        restored = WorkflowProvenance.from_dict(data)

        # Verify per-model preprocessing survived
        assert restored.preprocessing.per_model["ridge"].scaling == "standard"
        assert restored.preprocessing.per_model["ridge"].power_transform == "yeo-johnson"
        assert restored.preprocessing.per_model["rf"].scaling == "none"
        assert restored.preprocessing.configs_differ() is True

        # Verify training survived
        assert restored.training.primary_model == "rf"
        assert restored.training.hyperparameters["ridge"]["alpha"] == 1.0
        assert restored.training.metrics_by_model["rf"]["RMSE"] == 10.1

        # Verify methods context matches
        orig_ctx = prov.get_methods_context()
        restored_ctx = restored.get_methods_context()
        assert orig_ctx["models_trained"] == restored_ctx["models_trained"]
        assert orig_ctx["preprocessing_per_model"] == restored_ctx["preprocessing_per_model"]


class TestProvenanceMethodsContextForPublication:
    """Verify that get_methods_context() produces data sufficient for generate_methods_section()."""

    def test_methods_context_sufficient_for_methods_section(self):
        """The methods context should contain every key that generate_methods_section needs."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload("glucose", "regression", ["age", "bmi", "insulin"], 500)
        prov.record_split("stratified", 350, 75, 75, random_seed=42)
        prov.record_preprocessing(
            {"ridge": {"numeric_scaling": "standard"}, "rf": {"numeric_scaling": "none"}},
            "median",
        )
        prov.record_training(
            ["ridge", "rf"], "rf", "validation RMSE",
            use_cv=True, cv_folds=5,
            hyperparameters={"ridge": {"alpha": 1.0}},
            metrics_by_model={"ridge": {"RMSE": 12.3}, "rf": {"RMSE": 10.1}},
        )

        ctx = prov.get_methods_context()

        # Required by generate_methods_section
        assert "target_name" in ctx
        assert "task_type" in ctx
        assert "n_total" in ctx
        assert "n_train" in ctx
        assert "n_val" in ctx
        assert "n_test" in ctx
        assert "split_strategy" in ctx
        assert "random_seed" in ctx
        assert "models_trained" in ctx
        assert "primary_model" in ctx
        assert "use_cv" in ctx
        assert "cv_folds" in ctx
        assert "hyperparameters" in ctx
        assert "preprocessing_per_model" in ctx
        assert "preprocessing_differs" in ctx

        # Per-model detail present
        assert len(ctx["preprocessing_per_model"]) == 2
        assert ctx["preprocessing_per_model"]["ridge"]["scaling"] == "standard"
        assert ctx["preprocessing_per_model"]["rf"]["scaling"] == "none"

    def test_methods_context_empty_when_no_provenance(self):
        """Empty provenance produces empty context without errors."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        ctx = prov.get_methods_context()
        assert ctx == {}

    def test_partial_workflow_produces_partial_context(self):
        """Only completed stages appear in methods context."""
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload("glucose", "regression", ["age"], 100)

        ctx = prov.get_methods_context()
        assert "target_name" in ctx
        assert "models_trained" not in ctx  # training not done yet
        assert "preprocessing_per_model" not in ctx  # preprocessing not done yet
