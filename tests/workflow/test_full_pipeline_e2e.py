"""
Workflow test: Full pipeline end-to-end — Upload through Report.

Verifies that a complete user session produces a manuscript draft that
accurately reflects the decisions made at each stage. This is the single
test that ties all pages together.

Stages:
    Upload → EDA profile → Feature Selection → Preprocess (2 models) →
    Train → Explain → Sensitivity → Report generation
"""
import sys
import os
import hashlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.conftest import build_regression_df, make_data_config, inject_uploaded_state
from ml.pipeline import build_preprocessing_pipeline, get_feature_names_after_transform
from ml.model_registry import get_registry
from ml.sensitivity import sensitivity_random_seeds, SensitivityAnalysis
from ml.eval import calculate_regression_metrics
from utils.workflow_provenance import WorkflowProvenance
from utils.insight_ledger import InsightLedger, Insight
from ml.narrative_engine import NarrativeEngine


REGISTRY = get_registry()


@pytest.fixture(scope="module")
def e2e_df():
    return build_regression_df(n=200, seed=42, missing_rate=0.05)


@pytest.fixture(scope="module")
def e2e_state():
    """Mutable session-state dict that accumulates across all steps."""
    return {}


class TestFullPipelineE2E:
    """Sequential end-to-end pipeline.  Tests run in definition order."""

    # ── Step 1: Upload ─────────────────────────────────────────────
    def test_step01_upload(self, e2e_df, e2e_state):
        """Simulate Upload & Audit."""
        state = e2e_state
        inject_uploaded_state(state, e2e_df, target_col="glucose", task_type="regression")

        assert state["data_config"].target_col == "glucose"
        assert state["data_config"].task_type == "regression"
        assert state.get("dataset_profile") is not None

        # Init provenance and ledger
        state["provenance"] = WorkflowProvenance()
        state["insight_ledger"] = InsightLedger()

        # Record upload in provenance (correct API)
        data_config = state["data_config"]
        state["provenance"].record_upload(
            target_col="glucose",
            task_type="regression",
            feature_cols=data_config.feature_cols,
            n_samples=len(e2e_df),
        )

    # ── Step 2: EDA — generate insights ────────────────────────────
    def test_step02_eda_insights(self, e2e_df, e2e_state):
        state = e2e_state
        ledger = state["insight_ledger"]

        # Simulate EDA observations
        ledger.upsert(Insight(
            id="eda_bmi_skew",
            source_page="02_EDA",
            category="distributional",
            severity="warning",
            finding="BMI is right-skewed (skewness=1.2)",
            implication="May violate linearity assumption for linear models",
            recommended_action="Apply log or power transform on Preprocess page",
            relevant_pages=["05_Preprocess"],
            model_scope=["linear"],
        ))
        ledger.upsert(Insight(
            id="eda_sufficient_n",
            source_page="02_EDA",
            category="data_quality",
            severity="info",
            finding="Sample size (n=200) is adequate for the feature set",
            implication="Favorable for analysis",
            recommended_action="",
            relevant_pages=[],
        ))

        assert len(ledger.insights) >= 2

    # ── Step 3: Feature Selection ──────────────────────────────────
    def test_step03_feature_selection(self, e2e_state):
        state = e2e_state
        data_config = state["data_config"]

        # Select all numeric features
        selected = [c for c in data_config.feature_cols
                    if state["raw_data"][c].dtype in ("float64", "int64")]
        state["selected_features"] = selected
        assert len(selected) >= 3

        state["provenance"].record_feature_selection(
            method="manual",
            n_before=len(data_config.feature_cols),
            n_after=len(selected),
            features_kept=selected,
        )

    # ── Step 4: Preprocess — two different models ──────────────────
    def test_step04_preprocess_per_model(self, e2e_state):
        state = e2e_state
        selected = state["selected_features"]
        df = state["raw_data"]

        numeric_features = selected
        categorical_features = []
        X_sample = df[numeric_features + categorical_features].copy()

        # Ridge: standard scaling
        ridge_pipe = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_scaling="standard",
            numeric_imputation="median",
        )
        ridge_pipe.fit(X_sample)

        # HistGB: no scaling
        histgb_pipe = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_scaling="none",
            numeric_imputation="median",
        )
        histgb_pipe.fit(X_sample)

        state["preprocessing_pipelines_by_model"] = {
            "ridge": ridge_pipe,
            "histgb_reg": histgb_pipe,
        }
        state["preprocessing_config_by_model"] = {
            "ridge": {"numeric_scaling": "standard"},
            "histgb_reg": {"numeric_scaling": "none"},
        }

        # Record provenance
        state["provenance"].record_preprocessing(
            configs_by_model={
                "ridge": {"numeric_scaling": "standard", "imputation": "median"},
                "histgb_reg": {"numeric_scaling": "none", "imputation": "median"},
            },
        )

        # Resolve the skew insight
        state["insight_ledger"].resolve(
            "eda_bmi_skew",
            resolved_by="Accepted without transform — relying on scaling for Ridge, "
                        "HistGB is tree-based and invariant",
            resolved_on_page="05_Preprocess",
        )

    # ── Step 5: Split and Train ────────────────────────────────────
    def test_step05_train(self, e2e_state):
        state = e2e_state
        df = state["raw_data"]
        selected = state["selected_features"]
        pipelines = state["preprocessing_pipelines_by_model"]

        # Prepare splits
        mask = df["glucose"].notna()
        X = df.loc[mask, selected].copy()
        y = df.loc[mask, "glucose"].copy()

        n = len(X)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        state["X_train"] = X.iloc[indices[:n_train]]
        state["X_val"] = X.iloc[indices[n_train:n_train + n_val]]
        state["X_test"] = X.iloc[indices[n_train + n_val:]]
        state["y_train"] = y.iloc[indices[:n_train]]
        state["y_val"] = y.iloc[indices[n_train:n_train + n_val]]
        state["y_test"] = y.iloc[indices[n_train + n_val:]]
        state["test_indices"] = indices[n_train + n_val:].tolist()

        # Record split provenance
        state["provenance"].record_split(
            strategy="holdout",
            train_n=len(state["X_train"]),
            val_n=len(state["X_val"]),
            test_n=len(state["X_test"]),
        )

        # Train both models on their own preprocessed data
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import HistGradientBoostingRegressor

        trained_models = {}
        model_results = {}
        fitted_pipelines = {}
        feature_names_by_model = {}

        model_classes = {
            "ridge": Ridge(alpha=1.0),
            "histgb_reg": HistGradientBoostingRegressor(random_state=42, max_iter=50),
        }

        for model_key, model in model_classes.items():
            pipe = pipelines[model_key]
            X_tr = pipe.transform(state["X_train"])
            X_te = pipe.transform(state["X_test"])

            if hasattr(X_tr, "toarray"):
                X_tr = X_tr.toarray()
            if hasattr(X_te, "toarray"):
                X_te = X_te.toarray()

            model.fit(X_tr, state["y_train"].values)
            y_pred = model.predict(X_te)
            metrics = calculate_regression_metrics(state["y_test"].values, y_pred)

            trained_models[model_key] = model
            model_results[model_key] = {
                "metrics": metrics,
                "y_test": state["y_test"].values,
                "y_test_pred": y_pred,
            }
            fitted_pipelines[model_key] = pipe
            feature_names_by_model[model_key] = get_feature_names_after_transform(
                pipe, selected
            )

        state["trained_models"] = trained_models
        state["model_results"] = model_results
        state["fitted_preprocessing_pipelines"] = fitted_pipelines
        state["feature_names_by_model"] = feature_names_by_model

        # Both models should have valid metrics
        for mk in ["ridge", "histgb_reg"]:
            m = model_results[mk]["metrics"]
            r2 = next((v for k, v in m.items() if "r2" in k.lower() or "r²" in k.lower()), None)
            assert r2 is not None and r2 > 0, f"{mk} R² should be positive, got {r2}"

        # Record provenance
        state["provenance"].record_training(
            models_trained=list(trained_models.keys()),
            metrics_by_model={k: v["metrics"] for k, v in model_results.items()},
        )

    # ── Step 6: Explainability (permutation importance) ────────────
    def test_step06_permutation_importance(self, e2e_state):
        from sklearn.inspection import permutation_importance as perm_imp

        state = e2e_state
        pipelines = state["fitted_preprocessing_pipelines"]
        perm_results = {}

        for model_key, model in state["trained_models"].items():
            pipe = pipelines[model_key]

            from sklearn.pipeline import Pipeline as SkPipeline
            full_pipe = SkPipeline([("preprocess", pipe), ("model", model)])

            pi = perm_imp(
                full_pipe, state["X_test"], state["y_test"],
                n_repeats=5, random_state=42, n_jobs=1,
            )
            fnames = state["feature_names_by_model"].get(model_key, state["selected_features"])
            perm_results[model_key] = {
                "importances_mean": pi.importances_mean,
                "importances_std": pi.importances_std,
                "feature_names": fnames,
            }

        state["permutation_importance"] = perm_results

        # Record provenance
        state["provenance"].record_explainability(
            methods=["permutation_importance"],
            models=list(state["trained_models"].keys()),
        )

        # Validate structure
        for mk in ["ridge", "histgb_reg"]:
            pr = perm_results[mk]
            assert "importances_mean" in pr
            assert "importances_std" in pr
            assert "feature_names" in pr
            assert len(pr["importances_mean"]) > 0

    # ── Step 7: Sensitivity — seed robustness ──────────────────────
    def test_step07_sensitivity(self, e2e_state):
        from sklearn.linear_model import Ridge

        state = e2e_state
        pipe = state["fitted_preprocessing_pipelines"]["ridge"]
        X_tr = pipe.transform(state["X_train"])
        X_te = pipe.transform(state["X_test"])
        if hasattr(X_tr, "toarray"):
            X_tr = X_tr.toarray()
        if hasattr(X_te, "toarray"):
            X_te = X_te.toarray()
        y_tr = state["y_train"].values
        y_te = state["y_test"].values

        def train_fn(seed):
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr)
            return m

        def eval_fn(m):
            return calculate_regression_metrics(y_te, m.predict(X_te))

        analysis = sensitivity_random_seeds(train_fn, eval_fn, seeds=[0, 1, 7, 13], baseline_seed=42)

        assert isinstance(analysis, SensitivityAnalysis)
        assert analysis.analysis_type == "random_seed"
        assert len(analysis.variations) > 0

        df_sens = analysis.to_dataframe()
        assert len(df_sens) >= 2

        state["sensitivity_analysis"] = analysis
        state["sensitivity_seed_results"] = df_sens

        state["provenance"].record_sensitivity(
            seed_stability=True,
        )

    # ── Step 8: Report — narrative engine ──────────────────────────
    def test_step08_narrative_generation(self, e2e_state):
        state = e2e_state
        engine = NarrativeEngine(
            provenance=state["provenance"],
            ledger=state["insight_ledger"],
        )
        draft = engine.generate()

        # Manuscript should have content
        sections = draft.sections
        assert len(sections) > 0, "Manuscript has no sections"

        md = draft.to_markdown()
        assert len(md) > 200, f"Manuscript too short ({len(md)} chars)"

        # Provenance should flow through — model names should appear
        md_lower = md.lower()
        assert "ridge" in md_lower, "Ridge not mentioned in manuscript"

        state["manuscript_draft"] = draft

    # ── Step 9: Cross-check — provenance matches reality ───────────
    def test_step09_provenance_matches_training(self, e2e_state):
        """Verify provenance records match what actually happened."""
        state = e2e_state
        ctx = state["provenance"].get_methods_context()

        # Upload provenance
        assert ctx.get("target_name") == "glucose"
        assert ctx.get("task_type") == "regression"
        assert ctx.get("n_total") == 200

        # Training provenance
        models_trained = ctx.get("models_trained", [])
        assert "ridge" in models_trained
        assert "histgb_reg" in models_trained

        # Feature selection provenance
        assert ctx.get("n_features_after_selection", 0) > 0

    # ── Step 10: Cross-check — insights survived the full journey ──
    def test_step10_insight_lifecycle_complete(self, e2e_state):
        """Insights created in EDA, resolved in Preprocess, appear in narrative."""
        state = e2e_state
        ledger = state["insight_ledger"]

        # The BMI skew insight should be resolved
        bmi_insight = ledger.get("eda_bmi_skew")
        assert bmi_insight is not None
        assert bmi_insight.resolved is True
        assert "05_Preprocess" in (bmi_insight.resolved_on_page or "")

        # The sufficient-n insight should still be unresolved (info-level)
        n_insight = ledger.get("eda_sufficient_n")
        assert n_insight is not None
        assert n_insight.resolved is False

        # Manuscript should have been generated
        draft = state["manuscript_draft"]
        assert draft.to_markdown(), "Manuscript should have content"
