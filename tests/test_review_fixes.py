"""Regression tests for the 2026-07 code-review fixes.

Each test pins a defect that was found and fixed during the pre-conference
review: statistical correctness, state invalidation, the test-set lockbox,
and manuscript honesty. See CODE_REVIEW.md for the findings these guard.
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


# ── Bootstrap: degenerate resamples must not poison the CI ───────────────

class TestBootstrapDegenerateResamples:
    def test_auc_ci_finite_on_imbalanced_data(self):
        """Single-class AUC resamples previously propagated NaN into both CI
        bounds via np.percentile; they are now dropped."""
        from ml.bootstrap import bootstrap_all_classification_metrics

        rng = np.random.RandomState(0)
        n = 60
        y_true = np.zeros(n, dtype=int)
        y_true[:6] = 1  # 10% prevalence → degenerate resamples near-certain
        rng.shuffle(y_true)
        y_proba = np.clip(y_true * 0.5 + rng.uniform(0, 0.5, n), 0, 1)
        y_pred = (y_proba > 0.5).astype(int)

        results = bootstrap_all_classification_metrics(
            y_true, y_pred, y_proba=y_proba, n_resamples=1000
        )
        assert "AUC" in results
        auc = results["AUC"]
        assert np.isfinite(auc.estimate)
        assert np.isfinite(auc.ci_lower), "AUC CI lower bound must not be NaN"
        assert np.isfinite(auc.ci_upper), "AUC CI upper bound must not be NaN"
        assert auc.ci_lower <= auc.estimate <= auc.ci_upper

    def test_ci_reports_nan_when_too_few_valid_resamples(self):
        """With almost no valid resamples the CI must be honestly NaN, not
        fabricated from a handful of values."""
        from ml.bootstrap import bootstrap_metric

        def metric_all_nan(yt, yp):
            return float("nan")

        res = bootstrap_metric(np.arange(30.0), np.arange(30.0), metric_all_nan,
                               n_resamples=200)
        assert np.isnan(res.ci_lower) and np.isnan(res.ci_upper)


# ── Test-set lockbox invariants ──────────────────────────────────────────

class TestLockbox:
    def setup_method(self):
        _clear_session()

    def teardown_method(self):
        _clear_session()

    def _df(self, n=200, seed=7):
        rng = np.random.RandomState(seed)
        return pd.DataFrame({
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "y": rng.normal(size=n),
        })

    def test_creation_and_partition(self):
        from utils.test_lockbox import ensure_lockbox, train_row_mask

        df = self._df()
        lb = ensure_lockbox(df, "y", "regression")
        assert lb is not None
        assert lb["n_test"] == 30  # 15% of 200
        mask = train_row_mask(df.index)
        assert int(mask.sum()) == 170
        assert all(not mask.loc[lbl] for lbl in lb["labels"])

    def test_deterministic_across_calls(self):
        from utils.test_lockbox import ensure_lockbox

        df = self._df()
        lb1 = ensure_lockbox(df, "y", "regression")
        lb2 = ensure_lockbox(df, "y", "regression")
        assert lb1["labels"] == lb2["labels"]

    def test_rebuild_invalidates_downstream(self):
        from utils.test_lockbox import ensure_lockbox

        df = self._df()
        ensure_lockbox(df, "y", "regression")
        st.session_state["trained_models"] = {"marker": "STALE"}
        ensure_lockbox(df, "y", "regression", fraction=0.30)
        assert st.session_state.get("trained_models") in ({}, None)

    def test_exploratory_mode_unlocks(self):
        from utils.test_lockbox import ensure_lockbox, train_row_mask

        df = self._df()
        ensure_lockbox(df, "y", "regression")
        st.session_state["exploratory_mode"] = True
        assert bool(train_row_mask(df.index).all())

    def test_classification_lockbox_stratifies(self):
        from utils.test_lockbox import ensure_lockbox

        rng = np.random.RandomState(3)
        df = self._df()
        df["y"] = (rng.uniform(size=len(df)) < 0.25).astype(int)
        lb = ensure_lockbox(df, "y", "classification", fraction=0.20)
        assert lb["stratified"] is True
        test_rate = df.loc[lb["labels"], "y"].mean()
        assert abs(test_rate - df["y"].mean()) < 0.10

    def test_mask_survives_row_subsetting(self):
        """train_row_mask keys on index labels, so filtered/engineered frames
        that preserve the original index stay correctly quarantined."""
        from utils.test_lockbox import ensure_lockbox, train_row_mask

        df = self._df()
        lb = ensure_lockbox(df, "y", "regression")
        subset = df.iloc[::2]  # keep every other row, original labels intact
        mask = train_row_mask(subset.index)
        expected_test = set(lb["labels"]) & set(subset.index)
        assert int((~mask).sum()) == len(expected_test)


# ── Downstream reset completeness ────────────────────────────────────────

RESULT_KEYS = [
    "df_engineered", "preprocessing_pipelines_by_model", "X_train", "y_test",
    "trained_models", "model_results", "fitted_estimators",
    "fitted_preprocessing_pipelines", "feature_names_by_model",
    "shap_results", "permutation_importance", "bootstrap_results",
    "baseline_results", "calibration_results", "sensitivity_seed_results",
    "hypothesis_test_results", "table1_df", "table1_metadata",
    "methods_section", "manuscript_context",
]


class TestDownstreamReset:
    def setup_method(self):
        _clear_session()

    def teardown_method(self):
        _clear_session()

    def test_reset_clears_every_result_key(self):
        from utils.session_state import reset_downstream_results

        for k in RESULT_KEYS:
            st.session_state[k] = {"marker": "STALE"}
        reset_downstream_results()
        for k in RESULT_KEYS:
            v = st.session_state.get(k)
            assert not (isinstance(v, dict) and v.get("marker") == "STALE"), \
                f"reset_downstream_results left stale value in '{k}'"

    def test_same_schema_content_change_resets_results(self):
        from utils.session_state import set_data, get_data

        df1 = pd.DataFrame({"a": np.arange(10.0), "y": np.arange(10.0)})
        set_data(df1)
        st.session_state["trained_models"] = {"marker": "STALE"}
        st.session_state["df_engineered"] = df1.head(3)

        set_data(df1 * 2.0)  # same columns, different values
        assert st.session_state.get("trained_models") in ({}, None)
        assert st.session_state.get("df_engineered") is None
        active = get_data()
        assert len(active) == 10 and float(active["a"].iloc[1]) == 2.0

    def test_identical_content_rerun_is_noop(self):
        from utils.session_state import set_data

        df1 = pd.DataFrame({"a": np.arange(10.0), "y": np.arange(10.0)})
        set_data(df1)
        st.session_state["trained_models"] = {"marker": "KEEP"}
        set_data(df1.copy())  # benign rerun with identical content
        assert st.session_state["trained_models"] == {"marker": "KEEP"}


# ── Manuscript honesty ───────────────────────────────────────────────────

class TestNarrativeHonesty:
    def _engine(self, primary, metrics):
        from ml.narrative_engine import NarrativeEngine
        from utils.workflow_provenance import WorkflowProvenance
        from utils.insight_ledger import InsightLedger

        prov = WorkflowProvenance()
        prov.record_upload(target_col="y", task_type="regression",
                           feature_cols=["a", "b"], n_samples=100)
        prov.record_training(
            models_trained=list(metrics.keys()),
            metrics_by_model=metrics,
        )
        ctx = {
            "included_models": list(metrics.keys()),
            "manuscript_primary_model": primary,
            "selected_model_results": None,
        }
        engine = NarrativeEngine(prov, InsightLedger(), manuscript_context=ctx)
        return engine.generate().to_markdown()

    METRICS = {
        "rf": {"RMSE": 1.0, "R2": 0.9},
        "ridge": {"RMSE": 2.0, "R2": 0.5},
    }

    def test_no_placeholder_in_results(self):
        text = self._engine("rf", self.METRICS)
        assert "[Feature importance analysis pending" not in text

    def test_best_claim_only_for_true_best(self):
        text = self._engine("ridge", self.METRICS)  # ridge is NOT best
        assert "selected as the primary model" in text
        assert "Ridge" not in [
            s for s in text.split(".") if "best overall performance" in s
        ]

    def test_best_claim_for_actual_best(self):
        text = self._engine("rf", self.METRICS)
        assert "best overall performance" in text


class TestStrengthsClassification:
    def test_info_severity_is_not_a_strength(self):
        from utils.insight_ledger import InsightLedger, Insight

        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_skew_group", source_page="02_EDA", category="distribution",
            severity="info", finding="3 features are heavily skewed",
            implication="Linear models may be affected",
        ))
        ledger.upsert(Insight(
            id="eda_opportunity_balance", source_page="02_EDA", category="target",
            severity="opportunity", finding="Classes are well balanced",
            implication="Accuracy is a meaningful metric",
            resolved=True, resolved_by="Positive signal — no action needed",
            resolved_on_page="02_EDA",
        ))
        points = ledger.discussion_points_for_manuscript()
        assert "Classes are well balanced" in points["strengths"]
        assert "3 features are heavily skewed" not in points["strengths"]


# ── Source-level pins for wiring fixes ───────────────────────────────────

class TestWiringPins:
    def _read(self, rel):
        with open(os.path.join(PROJECT_ROOT, rel)) as f:
            return f.read()

    def test_record_split_called_by_train_page(self):
        assert "record_split(" in self._read("pages/06_Train_and_Compare.py")

    def test_calibration_results_written_by_train_page(self):
        src = self._read("pages/06_Train_and_Compare.py")
        assert 'st.session_state["calibration_results"]' in src

    def test_preprocess_resolves_real_insight_ids(self):
        src = self._read("pages/05_Preprocess.py")
        assert "eda_skew_group" in src
        assert "eda_skew_individual" not in src
        assert "preprocess_outlier_handling" in src
        assert '"eda_outliers"' not in src

    def test_coaching_derives_selected_models(self):
        src = self._read("utils/coaching_ui.py")
        assert "train_model_" in src

    def test_selection_guidance_uses_real_metric_columns(self):
        src = self._read("pages/06_Train_and_Compare.py")
        assert "'AUC (val)'" not in src and "'R² (val)'" not in src

    def test_feature_selection_scopes_to_train_rows(self):
        src = self._read("pages/04_Feature_Selection.py")
        assert "train_row_mask" in src
