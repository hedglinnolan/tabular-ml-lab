"""Tests for publication engine (methods generator, TRIPOD, flow diagram)."""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.publication import (
    TRIPODTracker, TRIPOD_ITEMS,
    generate_methods_section,
    generate_flow_diagram_mermaid,
    subgroup_analysis,
)
from ml.latex_report import generate_latex_report


def test_tripod_tracker():
    tracker = TRIPODTracker()
    done, total = tracker.get_progress()
    assert done == 0
    assert total == len(TRIPOD_ITEMS)

    tracker.mark_complete("outcome_defined", note="Glucose level", page_ref="Upload")
    done, total = tracker.get_progress()
    assert done == 1

    df = tracker.get_checklist_df()
    assert len(df) == len(TRIPOD_ITEMS)
    assert "✅" in df["Status"].values


def test_methods_section():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={"numeric_scaling": "standard", "numeric_imputation": "median"},
        model_configs={"Ridge": {}, "Random Forest": {}},
        split_config={},
        n_total=1000,
        n_train=700,
        n_val=150,
        n_test=150,
        feature_names=["age", "bmi", "glucose"],
        target_name="outcome",
        task_type="regression",
        metrics_used=["RMSE", "MAE", "R2"],
    )
    assert "1,000 observations" in text
    assert "RMSE" in text
    assert "age, bmi, glucose" in text
    assert "RIDGE" in text
    assert "bootstrap" in text.lower()


def test_flow_diagram():
    mermaid = generate_flow_diagram_mermaid(
        n_total=1000,
        n_excluded=50,
        exclusion_reasons={"Missing outcome": 30, "Age < 18": 20},
        n_train=700,
        n_val=125,
        n_test=125,
    )
    assert "graph TD" in mermaid
    assert "1,000" in mermaid
    assert "700" in mermaid


def test_subgroup_analysis():
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 100)
    y_pred = y_true + np.random.normal(0, 10, 100)
    subgroups = np.array(["Young"] * 50 + ["Old"] * 50)

    result = subgroup_analysis(y_true, y_pred, subgroups, n_bootstrap=50)
    assert isinstance(result, pd.DataFrame)
    assert "Overall" in result["Subgroup"].values
    assert "Young" in result["Subgroup"].values
    assert "Old" in result["Subgroup"].values


def test_methods_section_preserves_percentile_clipping_details():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={
            "missing_data": {"label": "median imputation"},
            "outliers": {
                "method": "percentile",
                "label": "percentile-based winsorization",
                "lower_percentile": 4,
                "upper_percentile": 96,
            },
        },
        model_configs={"Ridge": {}},
        split_config={},
        n_total=1000,
        n_train=700,
        n_val=150,
        n_test=150,
        feature_names=["age", "bmi", "glucose"],
        target_name="outcome",
        task_type="regression",
        metrics_used=["RMSE"],
    )
    assert "4th and 96th percentiles" in text
    assert "percentile-based winsorization" not in text


def test_methods_section_detects_different_model_specific_outlier_thresholds():
    import streamlit as st

    original = st.session_state.get('preprocessing_config_by_model', None)
    try:
        st.session_state.preprocessing_config_by_model = {
            "ridge": {
                "numeric_scaling": "standard",
                "categorical_encoding": "onehot",
                "numeric_outlier_treatment": "percentile",
                "numeric_outlier_params": {"lower_percentile": 4, "upper_percentile": 96},
                "numeric_power_transform": "none",
                "numeric_log_transform": False,
                "use_pca": False,
            },
            "rf": {
                "numeric_scaling": "standard",
                "categorical_encoding": "onehot",
                "numeric_outlier_treatment": "percentile",
                "numeric_outlier_params": {"lower_percentile": 1, "upper_percentile": 99},
                "numeric_power_transform": "none",
                "numeric_log_transform": False,
                "use_pca": False,
            },
        }

        text = generate_methods_section(
            data_config={},
            preprocessing_config={"missing_data": {"label": "median imputation"}},
            model_configs={"Ridge": {}, "RF": {}},
            split_config={},
            n_total=1000,
            n_train=700,
            n_val=150,
            n_test=150,
            feature_names=["age", "bmi", "glucose"],
            target_name="outcome",
            task_type="regression",
            metrics_used=["RMSE"],
        )

        assert "Model-specific preprocessing differed as follows" in text
        assert "4th and 96th percentiles" in text
        assert "1th and 99th percentiles" in text
    finally:
        if original is None:
            try:
                del st.session_state['preprocessing_config_by_model']
            except Exception:
                pass
        else:
            st.session_state.preprocessing_config_by_model = original


def test_generate_latex_report_preserves_detailed_methods_and_results_sections():
    methods_text = """### Missing Data
Missing numeric values were handled using median imputation.

### Data Preprocessing
Continuous predictors were scaled using z-score standardization.

### Feature Engineering
Polynomial degree 2 terms were added for key predictors.

### Model Interpretability
SHapley Additive exPlanations (SHAP) values were computed.

## Results (Draft)
### Model Performance
**RIDGE:** RMSE: 0.1234
"""

    latex = generate_latex_report(
        methods_section=methods_text,
        model_results={"ridge": {"metrics": {"RMSE": 0.1234}}},
        task_type="regression",
    )

    assert "\\subsection{Missing Data}" in latex
    assert "median imputation" in latex
    assert "\\subsection{Data Preprocessing}" in latex
    assert "z-score standardization" in latex
    assert "\\subsection{Feature Engineering}" in latex
    assert "Polynomial degree 2 terms were added" in latex
    assert "\\subsection{Model Interpretability}" in latex
    assert "SHapley Additive exPlanations" in latex
    assert "Table \\ref{tab:model_performance} summarizes held-out performance" in latex
    assert "RIDGE" in latex


def test_generate_latex_report_normalizes_known_text_artifacts():
    latex = generate_latex_report(
        methods_section="### Comparison with PriorWork\nmainresults are shown below.",
        task_type="regression",
    )

    assert "Prior Work" in latex
    assert "main results" in latex
    assert "PriorWork" not in latex
    assert "mainresults" not in latex


def test_methods_section_uses_consistent_feature_counts_from_workflow_state():
    import streamlit as st

    saved_pre_fe = st.session_state.get('pre_fe_feature_cols', None)
    saved_engineered = st.session_state.get('engineered_feature_names', None)
    saved_log = st.session_state.get('methodology_log', None)
    try:
        st.session_state['pre_fe_feature_cols'] = [f"base_{i}" for i in range(26)]
        st.session_state['engineered_feature_names'] = [f"eng_{i}" for i in range(9)]
        st.session_state['methodology_log'] = [
            {
                'step': 'Feature Selection',
                'details': {'methods': ['lasso'], 'n_features_before': 35, 'n_features_after': 23},
            },
            {
                'step': 'Feature Selection Applied',
                'details': {'method': 'consensus', 'n_features_selected': 23},
            },
        ]

        text = generate_methods_section(
            data_config={'feature_cols': [f"base_{i}" for i in range(26)]},
            preprocessing_config={'missing_data': {'label': 'median imputation'}},
            model_configs={'ridge': {}},
            split_config={},
            n_total=1000,
            n_train=700,
            n_val=150,
            n_test=150,
            feature_names=[f"selected_{i}" for i in range(35)],
            target_name='outcome',
            task_type='regression',
            metrics_used=['RMSE'],
        )

        assert "23 predictors in the final modeling set" in text
        assert "began with 26 original predictors, expanded this to 35 candidate predictors after feature engineering, and retained 23 predictors for final modeling" in text
        assert "reduced the feature set from 35 to 23 predictors" in text
        assert "35 predictor variables entered into final modeling" not in text
    finally:
        if saved_pre_fe is None:
            st.session_state.pop('pre_fe_feature_cols', None)
        else:
            st.session_state['pre_fe_feature_cols'] = saved_pre_fe
        if saved_engineered is None:
            st.session_state.pop('engineered_feature_names', None)
        else:
            st.session_state['engineered_feature_names'] = saved_engineered
        if saved_log is None:
            st.session_state.pop('methodology_log', None)
        else:
            st.session_state['methodology_log'] = saved_log



def test_methods_section_keeps_only_latest_seed_sensitivity_run_per_model_metric():
    import streamlit as st

    saved_log = st.session_state.get('methodology_log', None)
    saved_seed = st.session_state.get('sensitivity_seed_results', None)
    try:
        st.session_state['methodology_log'] = [
            {'step': 'Sensitivity Analysis', 'action': 'Ran seed stability analysis', 'details': {'model': 'ridge', 'metric': 'RMSE', 'n_seeds': 5}},
            {'step': 'Sensitivity Analysis', 'action': 'Ran seed stability analysis', 'details': {'model': 'ridge', 'metric': 'RMSE', 'n_seeds': 20}},
        ]
        st.session_state['sensitivity_seed_results'] = pd.DataFrame({'RMSE': [0.45, 0.5, 0.55]})

        text = generate_methods_section(
            data_config={},
            preprocessing_config={'missing_data': {'label': 'median imputation'}},
            model_configs={'ridge': {}},
            split_config={},
            n_total=100,
            n_train=70,
            n_val=15,
            n_test=15,
            feature_names=['a', 'b'],
            target_name='outcome',
            task_type='regression',
            metrics_used=['RMSE'],
        )

        assert "20 random seeds" in text
        assert "5 random seeds" not in text
        assert text.count("Seed stability analysis was performed") == 1
    finally:
        if saved_log is None:
            st.session_state.pop('methodology_log', None)
        else:
            st.session_state['methodology_log'] = saved_log
        if saved_seed is None:
            st.session_state.pop('sensitivity_seed_results', None)
        else:
            st.session_state['sensitivity_seed_results'] = saved_seed



def test_generate_latex_report_avoids_redundant_results_prose_when_table_present():
    latex = generate_latex_report(
        methods_section="""## Results (Draft)\n### Model Performance\n**RIDGE:** RMSE: 0.1234\n""",
        model_results={"ridge": {"metrics": {"RMSE": 0.1234}}},
        task_type="regression",
    )

    assert "Table \\ref{tab:model_performance} summarizes held-out performance" in latex
    assert "\\caption{Model performance on the held-out test set (regression metrics).}" in latex
    assert "\\paragraph{Model Performance}" not in latex
    assert "**RIDGE:**" not in latex



def test_generate_latex_report_repairs_due_to_duplication_artifact():
    latex = generate_latex_report(
        methods_section="### Methodological Considerations\nText due to the complexity of dynamically reconstructing column-specific pipelines) due to the workflow.",
        task_type="regression",
    )

    assert ") because the workflow" in latex
    assert ") due to the workflow" not in latex


def test_generate_latex_report_repairs_simple_fused_word_artifacts():
    latex = generate_latex_report(
        methods_section="### Methodological Considerations\nSignals were reviewed andautomatically compared withbaseline outputs.",
        task_type="regression",
    )

    assert "and automatically compared with baseline outputs" in latex
    assert "andautomatically" not in latex
    assert "withbaseline" not in latex


def test_methods_section_prefers_manuscript_context_over_session_state_model_drift():
    import streamlit as st

    saved_log = st.session_state.get('methodology_log', None)
    try:
        st.session_state['methodology_log'] = [
            {'step': 'Model Training', 'details': {'models': ['ridge', 'rf', 'xgb'], 'best_model': 'xgb'}},
        ]
        manuscript_context = {
            'included_models': ['ridge', 'rf'],
            'selected_model_results': {
                'ridge': {'metrics': {'RMSE': 0.42}},
                'rf': {'metrics': {'RMSE': 0.35}},
            },
            'selected_bootstrap_results': {},
            'feature_names_for_manuscript': ['f1', 'f2'],
            'feature_counts': {'selected': 2},
            'manuscript_primary_model': 'ridge',
            'best_model_by_metric': 'rf',
            'best_metric_name': 'RMSE',
        }

        text = generate_methods_section(
            data_config={'feature_cols': ['base1', 'base2', 'base3']},
            preprocessing_config={'missing_data': {'label': 'median imputation'}},
            model_configs={'ridge': {}, 'rf': {}, 'xgb': {}},
            split_config={},
            n_total=100,
            n_train=70,
            n_val=15,
            n_test=15,
            feature_names=['live1', 'live2', 'live3'],
            target_name='outcome',
            task_type='regression',
            metrics_used=['RMSE'],
            selected_model_results={'xgb': {'metrics': {'RMSE': 0.20}}},
            best_model_name='xgb',
            manuscript_context=manuscript_context,
        )

        assert 'The manuscript-primary model was **RIDGE**.' in text
        assert 'The best model by RMSE was **RF**.' in text
        assert '**XGB:**' not in text
        assert '**RIDGE:**' in text and '**RF:**' in text
    finally:
        if saved_log is None:
            st.session_state.pop('methodology_log', None)
        else:
            st.session_state['methodology_log'] = saved_log


def test_methods_results_draft_marks_boundary_and_orders_metrics_from_outputs():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={'missing_data': {'label': 'median imputation'}},
        model_configs={'ridge': {}},
        split_config={},
        n_total=100,
        n_train=70,
        n_val=15,
        n_test=15,
        feature_names=['f1', 'f2'],
        target_name='outcome',
        task_type='regression',
        metrics_used=['R2', 'RMSE', 'MAE'],
        selected_model_results={
            'ridge': {'metrics': {'R2': 0.8, 'MAE': 0.2, 'RMSE': 0.3}}
        },
        bootstrap_results={},
    )

    assert '## Results (Workflow-Derived Draft)' in text
    assert 'leaves interpretation to the author' in text
    assert 'Table X should summarize the held-out performance of the 1 included model(s)' in text
    assert 'No manuscript-primary model was explicitly selected in the workflow' in text
    assert '**RIDGE:** RMSE: 0.3000; MAE: 0.2000; R2: 0.8000' in text


def test_methods_results_draft_does_not_infer_manuscript_primary_from_best_model_name():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={'missing_data': {'label': 'median imputation'}},
        model_configs={'ridge': {}, 'rf': {}},
        split_config={},
        n_total=100,
        n_train=70,
        n_val=15,
        n_test=15,
        feature_names=['f1', 'f2'],
        target_name='outcome',
        task_type='regression',
        metrics_used=['RMSE'],
        selected_model_results={
            'ridge': {'metrics': {'RMSE': 0.30}},
            'rf': {'metrics': {'RMSE': 0.25}},
        },
        bootstrap_results={},
        best_model_name='rf',
    )

    assert 'The manuscript-primary model was' not in text
    assert 'The best model by held-out metric was **RF**.' in text
    assert 'No manuscript-primary model was explicitly selected in the workflow.' in text


def test_generate_latex_report_uses_frozen_manuscript_subset_and_primary_model():
    manuscript_context = {
        'included_models': ['ridge', 'rf'],
        'selected_model_results': {
            'ridge': {'metrics': {'RMSE': 0.42}},
            'rf': {'metrics': {'RMSE': 0.35}},
        },
        'selected_bootstrap_results': {},
        'feature_names_for_manuscript': ['f1', 'f2'],
        'feature_counts': {'selected': 2},
        'manuscript_primary_model': 'ridge',
        'best_model_by_metric': 'rf',
        'best_metric_name': 'RMSE',
    }

    latex = generate_latex_report(
        methods_section='### Predictor Variables\nThe following predictor variables were included: f1, f2.\n',
        model_results={
            'ridge': {'metrics': {'RMSE': 0.42}},
            'rf': {'metrics': {'RMSE': 0.35}},
            'xgb': {'metrics': {'RMSE': 0.20}},
        },
        bootstrap_results={},
        task_type='regression',
        manuscript_context=manuscript_context,
    )

    assert 'The manuscript-primary model was \\textbf{RIDGE}.' in latex
    assert 'The best model by RMSE was \\textbf{RF}.' in latex
    assert 'XGB' not in latex
    assert 'RIDGE & 0.4200' in latex
    assert 'RF & 0.3500' in latex


def test_generate_latex_report_does_not_infer_primary_model_from_best_by_metric_only():
    latex = generate_latex_report(
        methods_section='### Predictor Variables\nThe following predictor variables were included: f1, f2.\n',
        model_results={
            'ridge': {'metrics': {'RMSE': 0.42}},
            'rf': {'metrics': {'RMSE': 0.35}},
        },
        bootstrap_results={},
        task_type='regression',
        manuscript_context={
            'selected_model_results': {
                'ridge': {'metrics': {'RMSE': 0.42}},
                'rf': {'metrics': {'RMSE': 0.35}},
            },
            'best_model_by_metric': 'rf',
            'best_metric_name': 'RMSE',
        },
    )

    assert 'The manuscript-primary model was' not in latex
    assert 'The best model by RMSE was \\textbf{RF}. No manuscript-primary model was explicitly selected in the workflow.' in latex


def test_generate_latex_report_fallback_methods_use_frozen_feature_names():
    manuscript_context = {
        'feature_names_for_manuscript': ['f1', 'f2'],
        'feature_counts': {'selected': 2},
    }

    latex = generate_latex_report(
        methods_section='',
        model_results={'ridge': {'metrics': {'RMSE': 0.42}}},
        task_type='regression',
        feature_names=['live1', 'live2', 'live3'],
        manuscript_context=manuscript_context,
    )

    assert 'The following 2 predictor variables were included: f1, f2.' in latex
    assert 'live1' not in latex


def test_generate_methods_from_log_reads_from_ledger():
    """generate_methods_from_log reads InsightLedger when it has entries."""
    import streamlit as st
    from ml.publication import generate_methods_from_log
    from utils.session_state import log_methodology

    saved_ledger = st.session_state.get('insight_ledger', None)
    saved_log = st.session_state.get('methodology_log', None)
    try:
        # Reset both systems so the ledger is the only source
        from utils.insight_ledger import InsightLedger
        st.session_state['insight_ledger'] = InsightLedger()
        st.session_state['methodology_log'] = []

        log_methodology('Model Training', 'Trained Ridge', {'model': 'ridge', 'hyperparameter_optimization': True})

        result = generate_methods_from_log()
        assert 'Model Training' in result
        entries = result['Model Training']
        assert len(entries) >= 1
        assert entries[0].get('step') == 'Model Training'
    finally:
        if saved_ledger is None:
            st.session_state.pop('insight_ledger', None)
        else:
            st.session_state['insight_ledger'] = saved_ledger
        if saved_log is None:
            st.session_state.pop('methodology_log', None)
        else:
            st.session_state['methodology_log'] = saved_log


def test_methods_section_lists_model_names():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={},
        model_configs={'ridge': {}, 'rf': {}, 'histgb_reg': {}},
        split_config={},
        n_total=200,
        n_train=140,
        n_val=30,
        n_test=30,
        feature_names=['feat1', 'feat2'],
        target_name='outcome',
        task_type='regression',
        metrics_used=['RMSE', 'R2'],
    )
    assert 'RIDGE' in text or 'Ridge' in text
    assert 'RF' in text or 'Random Forest' in text or 'rf' in text.lower()


def test_generate_methods_section_accepts_ledger_narratives():
    """generate_methods_section integrates ledger_narratives when provided."""
    text = generate_methods_section(
        data_config={},
        preprocessing_config={'numeric_scaling': 'standard'},
        model_configs={'ridge': {}},
        split_config={},
        n_total=200,
        n_train=140,
        n_val=30,
        n_test=30,
        feature_names=['age', 'bmi'],
        target_name='outcome',
        task_type='regression',
        metrics_used=['RMSE'],
        ledger_narratives={'EDA': 'Skewed features were log-transformed.'},
    )
    assert 'Data Quality and Preprocessing Rationale' in text
    assert 'EDA' in text
    assert 'Skewed features were log-transformed.' in text
