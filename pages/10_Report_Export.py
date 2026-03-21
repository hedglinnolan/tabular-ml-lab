"""
Page 10: Report Export
Generate and download comprehensive modeling report with trained artifacts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
import io
import zipfile
import json
import plotly.graph_objects as go
import plotly.express as px
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig
)
from ml.pipeline import get_pipeline_recipe
from utils.storyline import get_insights_by_category, render_breadcrumb, render_page_navigation
from ml.model_registry import get_registry

logger = logging.getLogger(__name__)

init_session_state()

from utils.theme import inject_custom_css, render_step_indicator, render_guidance, render_sidebar_workflow
from utils.table_export import table
st.set_page_config(page_title="Report Export", page_icon="📄", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="10_Report_Export")
render_step_indicator(10, "Report Export")
st.title("📄 Report Export")
st.caption("This is the culmination of the workflow: package the strongest parts of your analysis into one manuscript-ready starting point.")
render_breadcrumb("10_Report_Export")
render_page_navigation("10_Report_Export")

st.markdown("""
### Export a Manuscript-Ready Starting Point

This page should feel like the end of one coherent journey: upload data, build a baseline result, explain it, optionally strengthen it, then package the outputs for drafting.

It can generate:

1. **Methods Section Draft** — Auto-generated from your actual workflow choices
2. **TRIPOD Checklist Materials** — Prediction model reporting support
3. **Results Tables** — Model performance with bootstrap CIs
4. **Figures** — Calibration, feature importance, SHAP plots

**Download everything as a ZIP** as a strong starting point for your manuscript.
""")

# Progress indicator

# Guardrail: Report Export is primarily for prediction mode
task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ **Report Export is primarily designed for Prediction mode.**")
    st.info("""
    Please go to the **Upload & Audit** page and select **Prediction** as your task mode.
    This export workflow packages trained models, metrics, explainability results, and draft manuscript materials.
    """)
    st.stop()

# Check prerequisites
df = get_data()
if df is None:
    st.warning("Please complete the modeling workflow first")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
split_config: SplitConfig = st.session_state.get('split_config')
model_config: ModelConfig = st.session_state.get('model_config')
pipeline = get_preprocessing_pipeline()
trained_models = st.session_state.get('trained_models', {})
model_results = st.session_state.get('model_results', {})
data_audit = st.session_state.get('data_audit')
profile = st.session_state.get('dataset_profile')
coach_output = st.session_state.get('coach_output')

if not data_config:
    st.warning("Please configure your data in Upload & Audit first")
    st.stop()

if not trained_models:
    st.warning("Please train models first")
    st.stop()

st.info("💡 **Recommended export posture:** finish the recommended workflow first, then include advanced analyses only when they materially strengthen your manuscript.")

# Custom CSS for better report aesthetics
st.markdown("""
<style>
.report-section {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: #fafafa;
}
.report-section h3 {
    margin-top: 0;
    color: #333;
    border-bottom: 2px solid #1e88e5;
    padding-bottom: 0.5rem;
}
.metric-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e88e5;
}
.metric-label {
    font-size: 0.85rem;
    color: #666;
}
.coef-table {
    font-size: 0.9rem;
}
.model-detail-section {
    background: #f8f9fa;
    border-left: 4px solid #1e88e5;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and branch if available."""
    try:
        import subprocess
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()[:8]
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        return {'commit': commit_hash, 'branch': branch}
    except:
        return {'commit': 'unknown', 'branch': 'unknown'}


def generate_metadata() -> Dict[str, Any]:
    """Generate comprehensive metadata for export."""
    git_info = get_git_info()
    
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'app_version': '1.0.0',  # You could read this from a version file
        'git_commit': git_info['commit'],
        'git_branch': git_info['branch'],
        'random_seed': st.session_state.get('random_seed', 42),
        'dataset': {
            'n_rows': len(df),
            'n_features': len(data_config.feature_cols),
            'target': data_config.target_col,
            'task_type': data_config.task_type,
            'features': data_config.feature_cols
        },
        'splits': {
            'train_size': split_config.train_size,
            'val_size': split_config.val_size,
            'test_size': split_config.test_size,
            'stratify': split_config.stratify,
            'use_time_split': split_config.use_time_split
        },
        'preprocessing': st.session_state.get('preprocessing_config', {}),
        'models_trained': list(trained_models.keys())
    }
    
    # Add dataset profile summary if available
    if profile:
        metadata['dataset_profile'] = {
            'data_sufficiency': profile.data_sufficiency.value,
            'n_numeric': profile.n_numeric,
            'n_categorical': profile.n_categorical,
            'p_n_ratio': profile.p_n_ratio,
            'total_missing_rate': profile.total_missing_rate,
            'n_features_with_outliers': len(profile.features_with_outliers),
            'warnings': [w.short_message for w in profile.warnings]
        }
    
    return metadata


def extract_model_coefficients(model, model_key: str, feature_names: list) -> Optional[pd.DataFrame]:
    """Extract coefficients from linear models."""
    try:
        # Try to get coefficients from the model
        coef = None
        intercept = None
        
        if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
            coef = model.model.coef_
            intercept = model.model.intercept_ if hasattr(model.model, 'intercept_') else None
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            intercept = model.intercept_ if hasattr(model, 'intercept_') else None
        
        if coef is not None:
            # Handle multi-class case
            if len(coef.shape) > 1:
                coef = coef[0]  # Take first class for binary classification
            
            coef_df = pd.DataFrame({
                'Feature': feature_names[:len(coef)],
                'Coefficient': coef
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            if intercept is not None:
                intercept_row = pd.DataFrame({
                    'Feature': ['(Intercept)'],
                    'Coefficient': [intercept if np.isscalar(intercept) else intercept[0]]
                })
                coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
            
            return coef_df
    except Exception as e:
        logger.debug(f"Could not extract coefficients for {model_key}: {e}")
    
    return None


def _get_export_best_model(model_results: Dict[str, Dict[str, Any]], task_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return the best model key and metric used for that decision."""
    if not model_results:
        return None, None

    if task_type == 'regression':
        ranked = [
            (name, results.get('metrics', {}).get('RMSE'))
            for name, results in model_results.items()
            if results.get('metrics', {}).get('RMSE') is not None
        ]
        if not ranked:
            return None, None
        return min(ranked, key=lambda item: item[1])[0], 'RMSE'

    for metric in ('F1', 'AUC', 'Accuracy'):
        ranked = [
            (name, results.get('metrics', {}).get(metric))
            for name, results in model_results.items()
            if results.get('metrics', {}).get(metric) is not None
        ]
        if ranked:
            return max(ranked, key=lambda item: item[1])[0], metric

    return None, None


def build_export_context() -> Dict[str, Any]:
    """Freeze the export view into a single snapshot for consistent rendering/export."""
    pipelines_by_model = st.session_state.get('preprocessing_pipelines_by_model') or {}
    configs_by_model = st.session_state.get('preprocessing_config_by_model') or {}
    permutation_importance = st.session_state.get('permutation_importance') or {}
    shap_results = st.session_state.get('shap_results') or {}
    legacy_shap = st.session_state.get('shap_values') or {}
    pdp_results = st.session_state.get('pdp_results') or st.session_state.get('partial_dependence') or {}
    robustness = st.session_state.get('explainability_robustness') or {}
    manuscript_primary_model = st.session_state.get('report_best_model') or None
    best_model_by_metric, best_metric_name = _get_export_best_model(model_results, data_config.task_type if data_config else None)

    readiness = {
        'models': {'status': 'present' if trained_models else 'missing', 'detail': f"{len(trained_models)} trained model(s) available" if trained_models else 'Train at least one model before export.'},
        'metrics': {'status': 'present' if model_results else 'missing', 'detail': f"Metrics available for {len(model_results)} model(s)" if model_results else 'No model metrics found.'},
        'permutation_importance': {'status': 'present' if permutation_importance else 'missing', 'detail': f"Available for {len(permutation_importance)} model(s)" if permutation_importance else 'Permutation importance not computed.'},
        'shap': {'status': 'present' if shap_results else ('inferred' if legacy_shap else 'missing'), 'detail': f"SHAP results available for {len(shap_results)} model(s)" if shap_results else ('Legacy shap_values detected, but export uses shap_results payloads.' if legacy_shap else 'SHAP results not computed.')},
        'partial_dependence': {'status': 'present' if pdp_results else 'missing', 'detail': f"PDP results available for {len(pdp_results)} model(s)" if pdp_results else 'Partial dependence not computed.'},
        'bootstrap': {'status': 'present' if st.session_state.get('bootstrap_results') else 'missing', 'detail': 'Bootstrap confidence intervals available.' if st.session_state.get('bootstrap_results') else 'Bootstrap confidence intervals not found.'},
        'tripod_table1': {'status': 'present' if st.session_state.get('table1_df') is not None else 'missing', 'detail': 'Table 1 available for manuscript export.' if st.session_state.get('table1_df') is not None else 'Table 1 has not been generated.'},
        'bland_altman': {'status': 'inferred' if len([1 for r in model_results.values() if r.get('y_test_pred') is not None]) >= 2 and (data_config.task_type if data_config else None) == 'regression' else 'missing', 'detail': 'Can be recomputed from stored test predictions for regression models; not stored as a persistent artifact.' if len([1 for r in model_results.values() if r.get('y_test_pred') is not None]) >= 2 and (data_config.task_type if data_config else None) == 'regression' else 'Requires at least two regression models with test predictions.'},
    }

    return {
        'dataset': df,
        'data_config': data_config,
        'split_config': split_config,
        'model_config': model_config,
        'trained_models': trained_models,
        'model_results': model_results,
        'data_audit': data_audit,
        'profile': profile,
        'coach_output': coach_output,
        'pipeline': pipeline,
        'pipelines_by_model': pipelines_by_model,
        'configs_by_model': configs_by_model,
        'selected_model_params': st.session_state.get('selected_model_params', {}),
        'feature_names': st.session_state.get('feature_names', []),
        'permutation_importance': permutation_importance,
        'shap_results': shap_results,
        'legacy_shap_values': legacy_shap,
        'pdp_results': pdp_results,
        'robustness': robustness,
        'bootstrap_results': st.session_state.get('bootstrap_results') or {},
        'table1_df': st.session_state.get('table1_df'),
        'shap_figs': st.session_state.get('shap_matplotlib_figs', {}),
        'include_llm': st.session_state.get('report_include_llm', False),
        'manuscript_primary_model': manuscript_primary_model,
        'best_model_by_metric': best_model_by_metric,
        'best_metric_name': best_metric_name,
        'readiness': readiness,
    }


def render_export_readiness_audit(export_ctx: Dict[str, Any]) -> None:
    """Show what export can truthfully include before download."""
    status_icon = {'present': '✅', 'inferred': '🟡', 'missing': '❌'}
    status_label = {'present': 'Present', 'inferred': 'Inferred / recomputable', 'missing': 'Missing'}
    rows: List[Dict[str, str]] = []
    for artifact, info in export_ctx['readiness'].items():
        rows.append({
            'Artifact': artifact.replace('_', ' ').title(),
            'Status': f"{status_icon.get(info['status'], '•')} {status_label.get(info['status'], info['status'])}",
            'Detail': info['detail'],
        })

    st.subheader('Export Readiness Audit')
    st.caption('This audit freezes the current session state so the export reflects what is actually available, what can be recomputed, and what is still missing.')
    table(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    present = sum(1 for info in export_ctx['readiness'].values() if info['status'] == 'present')
    inferred = sum(1 for info in export_ctx['readiness'].values() if info['status'] == 'inferred')
    missing = sum(1 for info in export_ctx['readiness'].values() if info['status'] == 'missing')
    st.info(f"Readiness summary: {present} present · {inferred} inferred/recomputable · {missing} missing")

    best_by_metric = export_ctx.get('best_model_by_metric')
    best_metric_name = export_ctx.get('best_metric_name')
    manuscript_primary = export_ctx.get('manuscript_primary_model')
    if best_by_metric:
        msg = f"Best by current held-out metric: {best_by_metric.upper()}"
        if best_metric_name:
            msg += f" ({best_metric_name})"
        if manuscript_primary and manuscript_primary != best_by_metric:
            msg += f". Manuscript primary model currently selected: {manuscript_primary.upper()}."
        st.caption(msg)


def _build_manuscript_context(
    selected_for_report: List[str],
    selected_explain: List[str],
    include_results: bool,
    best_model: Optional[str],
) -> Dict[str, Any]:
    """Freeze manuscript-scoped facts once so markdown and LaTeX share the same boundary."""
    selected_results = {k: v for k, v in model_results.items() if k in selected_for_report} if include_results else None
    selected_bootstrap = {k: v for k, v in export_ctx['bootstrap_results'].items() if k in selected_for_report} if include_results else None

    x_train = st.session_state.get('X_train')
    x_train_columns = getattr(x_train, 'columns', None)
    if x_train_columns is not None:
        workflow_feature_names = list(x_train_columns)
    else:
        feature_names_state = st.session_state.get('feature_names')
        if feature_names_state is not None:
            workflow_feature_names = list(feature_names_state)
        else:
            workflow_feature_names = list(data_config.feature_cols)

    from ml.publication import _resolve_workflow_feature_counts, generate_methods_from_log

    feature_counts = _resolve_workflow_feature_counts(
        workflow_feature_names,
        logged_steps=generate_methods_from_log(),
        data_config={'feature_cols': list(data_config.feature_cols)},
    )

    return {
        'included_models': list(selected_for_report),
        'selected_model_results': selected_results,
        'selected_bootstrap_results': selected_bootstrap,
        'feature_names_for_manuscript': workflow_feature_names,
        'feature_counts': feature_counts,
        'manuscript_primary_model': best_model,
        'best_model_by_metric': export_ctx.get('best_model_by_metric'),
        'best_metric_name': export_ctx.get('best_metric_name'),
        'explainability_methods': list(selected_explain),
    }


def _build_methods_section_for_export(
    manuscript_context: Dict[str, Any],
) -> str:
    """Build the current methods/results draft directly from workflow state."""
    from ml.publication import generate_methods_section

    train_n = len(st.session_state.get('X_train', []))
    val_n = len(st.session_state.get('X_val', []))
    test_n = len(st.session_state.get('X_test', []))

    prep_summary = st.session_state.get('preprocessing_summary') or {}
    prep_config_raw = st.session_state.get('preprocessing_config') or {}
    prep_config = dict(prep_summary) if prep_summary else dict(prep_config_raw)
    if prep_summary and prep_config_raw:
        merged_outliers = dict(prep_summary.get('outliers') or {})
        raw_outliers = prep_config_raw.get('outliers') or {}
        raw_params = raw_outliers.get('params') or {
            key: raw_outliers.get(key)
            for key in (
                'lower_percentile', 'upper_percentile', 'threshold',
                'mad_threshold', 'n_mad', 'multiplier', 'iqr_multiplier'
            )
            if raw_outliers.get(key) is not None
        }
        if raw_params:
            merged_outliers['params'] = raw_params
        if merged_outliers:
            prep_config['outliers'] = merged_outliers

    selected_for_report = manuscript_context.get('included_models', [])
    selected_results = manuscript_context.get('selected_model_results')
    selected_bootstrap = manuscript_context.get('selected_bootstrap_results')

    fs_results = st.session_state.get("feature_selection_results")
    fs_method = fs_results[0].method if fs_results else None

    first_result = next(iter((selected_results or model_results).values()), {})
    metrics_used = list(first_result.get('metrics', {}).keys()) or ["RMSE"]
    
    # Build split_strategy from split_config
    split_strategy = None
    if split_config:
        if split_config.use_time_split:
            split_strategy = "chronological"
        elif split_config.stratify:
            split_strategy = "stratified"
        else:
            split_strategy = "random"
    
    # Build model_hyperparameters — prefer explicit params, fallback to trained model objects
    model_hyperparameters = dict(st.session_state.get('selected_model_params', {}))
    # If explicit params are sparse, extract from trained model objects
    if not model_hyperparameters or all(not v for v in model_hyperparameters.values()):
        trained_models = st.session_state.get('trained_models', {})
        for model_key, model_obj in trained_models.items():
            if model_key in model_hyperparameters and model_hyperparameters[model_key]:
                continue  # already have explicit params
            try:
                params = model_obj.get_params() if hasattr(model_obj, 'get_params') else {}
                # Filter to publication-relevant params only
                key_lower = model_key.lower()
                relevant = {}
                if key_lower in ('ridge', 'lasso', 'elasticnet'):
                    for k in ('alpha', 'l1_ratio'):
                        if k in params:
                            relevant[k] = params[k]
                elif key_lower in ('histgb_reg', 'histgb_clf'):
                    for k in ('max_iter', 'max_depth', 'learning_rate', 'min_samples_leaf', 'max_leaf_nodes'):
                        if k in params and params[k] is not None:
                            relevant[k] = params[k]
                elif key_lower in ('rf', 'xgb', 'lgbm'):
                    for k in ('n_estimators', 'max_depth', 'learning_rate'):
                        if k in params and params[k] is not None:
                            relevant[k] = params[k]
                elif key_lower == 'nn':
                    for k in ('hidden_layer_sizes', 'learning_rate_init', 'max_iter', 'activation'):
                        if k in params and params[k] is not None:
                            relevant[k] = params[k]
                elif key_lower == 'svm':
                    for k in ('C', 'kernel', 'gamma'):
                        if k in params and params[k] is not None:
                            relevant[k] = params[k]
                elif key_lower == 'knn':
                    for k in ('n_neighbors', 'weights'):
                        if k in params and params[k] is not None:
                            relevant[k] = params[k]
                if relevant:
                    model_hyperparameters[model_key] = relevant
            except Exception:
                pass
    
    # Check methodology log for hyperparameter_optimization
    hyperparameter_optimization = False
    methodology_log = st.session_state.get('methodology_log', [])
    for entry in methodology_log:
        if entry.get('step') == 'Model Training':
            details = entry.get('details', {})
            if details.get('hyperparameter_optimization'):
                hyperparameter_optimization = True
                break
    
    # Build missing_data_summary from dataset_profile or data_audit
    missing_data_summary = None
    profile = st.session_state.get('dataset_profile')
    if profile and hasattr(profile, 'n_features_with_missing'):
        # Build summary from profile
        total_features = profile.n_numeric + profile.n_categorical
        if profile.n_features_with_missing > 0:
            # Try to get per-feature missing rates
            data_audit = st.session_state.get('data_audit')
            if data_audit and 'missing_counts' in data_audit:
                missing_counts = data_audit['missing_counts']
                n_rows = len(df)
                missing_rates = {k: v / n_rows for k, v in missing_counts.items() if v > 0}
                if missing_rates:
                    min_rate = min(missing_rates.values())
                    max_rate = max(missing_rates.values())
                    missing_data_summary = {
                        'n_features_with_missing': profile.n_features_with_missing,
                        'total_features': total_features,
                        'min_missing_rate': min_rate,
                        'max_missing_rate': max_rate,
                    }
                else:
                    missing_data_summary = {
                        'n_features_with_missing': profile.n_features_with_missing,
                        'total_features': total_features,
                    }
    
    # Fallback: check data_audit directly
    if not missing_data_summary:
        data_audit = st.session_state.get('data_audit')
        if data_audit and 'missing_counts' in data_audit:
            missing_counts = data_audit['missing_counts']
            features_with_missing = sum(1 for v in missing_counts.values() if v > 0)
            if features_with_missing > 0:
                n_rows = len(df)
                missing_rates = {k: v / n_rows for k, v in missing_counts.items() if v > 0}
                total_features = len(data_config.feature_cols) if data_config else len(missing_counts)
                if missing_rates:
                    min_rate = min(missing_rates.values())
                    max_rate = max(missing_rates.values())
                    missing_data_summary = {
                        'n_features_with_missing': features_with_missing,
                        'total_features': total_features,
                        'min_missing_rate': min_rate,
                        'max_missing_rate': max_rate,
                    }

    # Final fallback: compute directly from the dataframe
    if not missing_data_summary and df is not None:
        try:
            feature_cols = list(data_config.feature_cols) if data_config else list(df.columns)
            feature_df = df[feature_cols] if all(c in df.columns for c in feature_cols) else df
            missing_per_col = feature_df.isnull().sum()
            cols_with_missing = missing_per_col[missing_per_col > 0]
            if len(cols_with_missing) > 0:
                n_rows = len(feature_df)
                rates = (cols_with_missing / n_rows)
                missing_data_summary = {
                    'n_features_with_missing': len(cols_with_missing),
                    'total_features': len(feature_cols),
                    'min_missing_rate': rates.min(),
                    'max_missing_rate': rates.max(),
                }
        except Exception:
            pass

    return generate_methods_section(
        data_config={},
        preprocessing_config=prep_config,
        model_configs={name: {} for name in selected_for_report},
        split_config={},
        n_total=len(df),
        n_train=train_n,
        n_val=val_n,
        n_test=test_n,
        feature_names=manuscript_context.get('feature_names_for_manuscript', []),
        target_name=data_config.target_col,
        task_type=data_config.task_type or "regression",
        metrics_used=metrics_used,
        feature_selection_method=fs_method,
        selected_model_results=selected_results,
        bootstrap_results=selected_bootstrap,
        best_model_name=manuscript_context.get('manuscript_primary_model'),
        explainability_methods=manuscript_context.get('explainability_methods'),
        random_seed=st.session_state.get("random_seed", 42),
        manuscript_context=manuscript_context,
        model_hyperparameters=model_hyperparameters,
        hyperparameter_optimization=hyperparameter_optimization,
        split_strategy=split_strategy,
        missing_data_summary=missing_data_summary,
    )


def generate_report(export_ctx: Dict[str, Any]) -> str:
    """Generate markdown report with improved structure and aesthetics."""
    report_lines = []

    data_config = export_ctx['data_config']
    split_config = export_ctx['split_config']
    trained_models = export_ctx['trained_models']
    model_results = export_ctx['model_results']
    profile = export_ctx['profile']
    coach_output = export_ctx['coach_output']
    pipeline = export_ctx['pipeline']
    pipelines_by_model = export_ctx['pipelines_by_model']
    configs_by_model = export_ctx['configs_by_model']
    feature_names = export_ctx['feature_names']

    git_info = get_git_info()
    
    # Header with metadata
    report_lines.append("# Tabular ML Lab Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Git Commit:** {git_info['commit']} ({git_info['branch']})")
    report_lines.append(f"**Random Seed:** {st.session_state.get('random_seed', 42)}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    # Best model summary
    best_model_key = export_ctx.get('best_model_by_metric')
    best_metric_name = export_ctx.get('best_metric_name')
    manuscript_primary_model = export_ctx.get('manuscript_primary_model')
    if best_model_key and best_model_key in model_results:
        best_model = model_results[best_model_key]
        report_lines.append(f"**Best Model (by held-out {best_metric_name or 'metric'}):** {best_model_key.upper()}")
        if data_config.task_type == 'regression':
            if best_model['metrics'].get('RMSE') is not None:
                report_lines.append(f"**Test RMSE:** {best_model['metrics']['RMSE']:.4f}")
            if best_model['metrics'].get('R2') is not None:
                report_lines.append(f"**Test R²:** {best_model['metrics']['R2']:.4f}")
        else:
            if best_model['metrics'].get('Accuracy') is not None:
                report_lines.append(f"**Test Accuracy:** {best_model['metrics']['Accuracy']:.4f}")
            if best_model['metrics'].get('F1') is not None:
                report_lines.append(f"**Test F1:** {best_model['metrics']['F1']:.4f}")
    if manuscript_primary_model:
        report_lines.append(f"**Manuscript Primary Model:** {manuscript_primary_model.upper()}")
        if best_model_key and manuscript_primary_model != best_model_key:
            report_lines.append("**Note:** The manuscript primary model differs from the current best held-out metric winner.")
    
    report_lines.append("")
    
    # Key findings
    if profile and profile.warnings:
        report_lines.append("**Key Data Warnings:**")
        for w in profile.warnings[:3]:
            report_lines.append(f"- {w.short_message}")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Abstract (Draft) - auto-scaffold from known facts
    report_lines.append("## Abstract (Draft)")
    report_lines.append("")
    
    # Objective
    report_lines.append(f"**Objective:** [PLACEHOLDER: clinical context]. This study developed and validated a prediction model for {data_config.target_col} using {data_config.task_type}.")
    report_lines.append("")
    
    # Methods
    train_n = len(st.session_state.get('X_train', []))
    val_n = len(st.session_state.get('X_val', []))
    test_n = len(st.session_state.get('X_test', []))
    report_lines.append(f"**Methods:** A total of {len(df):,} observations with {len(data_config.feature_cols)} predictors were split into training (n={train_n:,}), validation (n={val_n:,}), and test (n={test_n:,}) sets. {len(model_results)} models were compared.")
    report_lines.append("")
    
    # Results - extract best model metrics
    if best_model_key and best_model_key in model_results:
        best_model = model_results[best_model_key]
        if data_config.task_type == 'regression':
            primary_metric = 'RMSE'
            primary_val = best_model['metrics'].get('RMSE')
        else:
            primary_metric = 'F1' if 'F1' in best_model['metrics'] else 'Accuracy'
            primary_val = best_model['metrics'].get(primary_metric)
        
        if primary_val is not None:
            # Check for bootstrap CIs
            bootstrap_results_ctx = export_ctx.get('bootstrap_results', {})
            ci_str = ""
            if best_model_key in bootstrap_results_ctx:
                ci = bootstrap_results_ctx[best_model_key].get(primary_metric)
                if ci and hasattr(ci, 'ci_lower') and hasattr(ci, 'ci_upper'):
                    ci_str = f" (95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}])"
            
            report_lines.append(f"**Results:** The best model ({best_model_key.upper()}) achieved {primary_metric}: {primary_val:.4f}{ci_str}.")
        else:
            report_lines.append("**Results:** [PLACEHOLDER: Summarize key results with metrics and CIs].")
    else:
        report_lines.append("**Results:** [PLACEHOLDER: Summarize key results with metrics and CIs].")
    report_lines.append("")
    
    # Conclusion
    report_lines.append("**Conclusion:** [PLACEHOLDER: Summarize clinical implications].")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Dataset Summary
    report_lines.append("## Dataset Summary")
    report_lines.append("")
    report_lines.append("| Property | Value |")
    report_lines.append("|----------|-------|")
    report_lines.append(f"| Rows | {len(df):,} |")
    report_lines.append(f"| Features | {len(data_config.feature_cols)} |")
    report_lines.append(f"| Target | `{data_config.target_col}` |")
    report_lines.append(f"| Task Type | {data_config.task_type.title()} |")
    
    if profile:
        report_lines.append(f"| Numeric Features | {profile.n_numeric} |")
        report_lines.append(f"| Categorical Features | {profile.n_categorical} |")
        report_lines.append(f"| Data Sufficiency | {profile.data_sufficiency.value.title()} |")
        report_lines.append(f"| Feature/Sample Ratio | {profile.p_n_ratio:.4f} |")
    
    report_lines.append("")
    
    # Data Sufficiency Narrative
    if profile and profile.sufficiency_narrative:
        report_lines.append("### Data Sufficiency Analysis")
        report_lines.append("")
        report_lines.append(f"> {profile.sufficiency_narrative}")
        report_lines.append("")
    
    # Task and cohort detection
    task_det = st.session_state.get('task_type_detection')
    cohort_det = st.session_state.get('cohort_structure_detection')
    if task_det or cohort_det:
        report_lines.append("### Automatic Detection")
        report_lines.append("")
        if task_det and task_det.detected:
            report_lines.append(f"- **Task Type:** {task_det.detected} ({task_det.confidence} confidence)")
        if cohort_det and cohort_det.detected:
            report_lines.append(f"- **Cohort Structure:** {cohort_det.detected} ({cohort_det.confidence} confidence)")
            if cohort_det.entity_id_final:
                report_lines.append(f"- **Entity ID Column:** `{cohort_det.entity_id_final}`")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Key observations and resolutions from the unified ledger
    from utils.insight_ledger import get_ledger as _get_report_ledger
    _report_ledger = _get_report_ledger()
    if len(_report_ledger) > 0:
        report_lines.append("## Key Observations and Resolutions")
        report_lines.append("")
        report_lines.append(_report_ledger.narrative_for_report())
        report_lines.append("")

        # Manuscript-ready narrative grouped by workflow phase
        phase_narratives = _report_ledger.to_manuscript_narrative()
        if phase_narratives:
            report_lines.append("### Methods Narrative (by workflow phase)")
            report_lines.append("")
            for phase_name, narrative in phase_narratives.items():
                report_lines.append(f"**{phase_name}:** {narrative}")
                report_lines.append("")

        report_lines.append("---")
        report_lines.append("")
    
    # Split Strategy
    report_lines.append("## 🔀 Data Split Strategy")
    report_lines.append("")
    report_lines.append("| Split | Percentage | Samples |")
    report_lines.append("|-------|------------|---------|")
    train_n = len(st.session_state.get('X_train', []))
    val_n = len(st.session_state.get('X_val', []))
    test_n = len(st.session_state.get('X_test', []))
    report_lines.append(f"| Train | {split_config.train_size*100:.1f}% | {train_n:,} |")
    report_lines.append(f"| Validation | {split_config.val_size*100:.1f}% | {val_n:,} |")
    report_lines.append(f"| Test | {split_config.test_size*100:.1f}% | {test_n:,} |")
    report_lines.append("")
    
    split_type = "Time-based" if split_config.use_time_split else ("Stratified" if split_config.stratify else "Random")
    report_lines.append(f"**Split Type:** {split_type}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Preprocessing (per-model when available)
    if pipelines_by_model:
        report_lines.append("## Preprocessing (per model)")
        report_lines.append("")
        for mk, pl in pipelines_by_model.items():
            report_lines.append(f"### {mk.upper()}")
            report_lines.append("")
            recipe = get_pipeline_recipe(pl)
            report_lines.append("```")
            report_lines.append(recipe)
            report_lines.append("```")
            cfg = configs_by_model.get(mk, {})
            ov = cfg.get("overrides", [])
            if ov:
                report_lines.append("**Overrides:**")
                for n in ov:
                    report_lines.append(f"- {n}")
                report_lines.append("")
            report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    elif pipeline:
        report_lines.append("## Preprocessing Pipeline")
        report_lines.append("")
        recipe = get_pipeline_recipe(pipeline)
        report_lines.append("```")
        report_lines.append(recipe)
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Model Performance Comparison
    report_lines.append("## Model Performance")
    report_lines.append("")
    
    # Brief narrative before table
    if best_model_key and best_model_key in model_results:
        best_model = model_results[best_model_key]
        if data_config.task_type == 'regression':
            primary_metric = 'RMSE'
            primary_val = best_model['metrics'].get('RMSE')
        else:
            primary_metric = 'F1' if 'F1' in best_model['metrics'] else 'Accuracy'
            primary_val = best_model['metrics'].get(primary_metric)
        
        if primary_val is not None:
            report_lines.append(f"Best model: **{best_model_key.upper()}** with {primary_metric} = {primary_val:.4f} on the held-out test set.")
            report_lines.append("")
    
    # Metrics table
    comparison_data = []
    for name, results in model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the table nicely
    report_lines.append("### Performance Metrics (Test Set)")
    report_lines.append("")
    try:
        report_lines.append(comparison_df.to_markdown(index=False, floatfmt='.4f'))
    except:
        headers = '| ' + ' | '.join(comparison_df.columns) + ' |'
        separators = '| ' + ' | '.join(['---'] * len(comparison_df.columns)) + ' |'
        report_lines.append(headers)
        report_lines.append(separators)
        for _, row in comparison_df.iterrows():
            values = '| ' + ' | '.join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values]) + ' |'
            report_lines.append(values)
    report_lines.append("")
    
    # Cross-Validation Results
    cv_results_exist = any(r.get('cv_results') for r in model_results.values())
    if cv_results_exist:
        report_lines.append("### Cross-Validation Results")
        report_lines.append("")
        report_lines.append("| Model | Mean Score | Std Dev |")
        report_lines.append("|-------|------------|---------|")
        for name, results in model_results.items():
            if results.get('cv_results'):
                cv = results['cv_results']
                report_lines.append(f"| {name.upper()} | {cv['mean']:.4f} | ±{cv['std']:.4f} |")
        report_lines.append("")

        from ml.eval import compare_models_paired_cv
        cv_names = [n for n, r in model_results.items() if r.get("cv_results")]
        paired = compare_models_paired_cv(
            cv_names,
            model_results,
            task_type=data_config.task_type if data_config else "regression",
        )
        if paired:
            report_lines.append("### Statistical comparison of models (CV)")
            report_lines.append("")
            report_lines.append("Pairwise paired tests on fold-level CV scores. Mean Δ = mean(A) − mean(B); p < 0.05 suggests a significant difference.")
            report_lines.append("")
            report_lines.append("| Model A | Model B | Mean Δ | Test | p | Significant |")
            report_lines.append("|---------|---------|--------|------|---|-------------|")
            for (ma, mb), v in paired.items():
                mean_d = v["mean_delta"]
                tname = v["test_name"]
                p = v["p"]
                sig = "Yes" if (p is not None and np.isfinite(p) and p < 0.05) else "No"
                p_str = f"{p:.4f}" if (p is not None and np.isfinite(p)) else "—"
                report_lines.append(f"| {ma.upper()} | {mb.upper()} | {mean_d:.4f} | {tname} | {p_str} | {sig} |")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")
    
    # Model-Specific Details
    report_lines.append("## Model-Specific Details")
    report_lines.append("")
    
    registry = get_registry()
    selected_model_params = export_ctx['selected_model_params']
    
    for model_key, model_wrapper in trained_models.items():
        spec = registry.get(model_key)
        model_name = spec.name if spec else model_key.upper()
        results = model_results.get(model_key, {})
        
        report_lines.append(f"### {model_name}")
        report_lines.append("")
        
        # Hyperparameters
        params = selected_model_params.get(model_key, spec.default_params if spec else {})
        if params:
            report_lines.append("**Hyperparameters:**")
            report_lines.append("")
            for param_name, param_value in params.items():
                report_lines.append(f"- `{param_name}`: {param_value}")
            report_lines.append("")
        
        # Linear model coefficients
        if model_key in ['ridge', 'lasso', 'elasticnet', 'glm', 'huber', 'logreg']:
            coef_df = extract_model_coefficients(model_wrapper, model_key, feature_names)
            if coef_df is not None:
                report_lines.append("**Model Coefficients (Top 10 by magnitude):**")
                report_lines.append("")
                try:
                    report_lines.append(coef_df.head(10).to_markdown(index=False, floatfmt='.4f'))
                except:
                    for _, row in coef_df.head(10).iterrows():
                        report_lines.append(f"- {row['Feature']}: {row['Coefficient']:.4f}")
                report_lines.append("")
                
                # Interpretation note
                report_lines.append("> **Interpretation:** A positive coefficient means the feature increases the target value")
                report_lines.append("> (or log-odds for classification). Coefficients are on the scale of standardized features.")
                report_lines.append("")
        
        # Neural network architecture
        if model_key == 'nn' and hasattr(model_wrapper, 'get_architecture_summary'):
            try:
                arch_summary = model_wrapper.get_architecture_summary()
                if arch_summary:
                    report_lines.append(f"**Architecture:** {arch_summary}")
                    report_lines.append("")
            except:
                pass
        
        # Training history for NN
        if model_key == 'nn' and hasattr(model_wrapper, 'get_training_history'):
            try:
                history = model_wrapper.get_training_history()
                if history and 'train_loss' in history:
                    final_train_loss = history['train_loss'][-1]
                    final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 'N/A'
                    report_lines.append(f"**Training Summary:** {len(history['train_loss'])} epochs")
                    report_lines.append(f"- Final train loss: {final_train_loss:.4f}")
                    if final_val_loss != 'N/A':
                        report_lines.append(f"- Final validation loss: {final_val_loss:.4f}")
                    report_lines.append("")
            except:
                pass
        
        # Classification-specific: confusion matrix summary
        if data_config.task_type == 'classification' and 'y_test' in results and 'y_test_pred' in results:
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(results['y_test'], results['y_test_pred'])
                report_lines.append("**Confusion Matrix:**")
                report_lines.append("")
                report_lines.append("```")
                report_lines.append(str(cm))
                report_lines.append("```")
                report_lines.append("")
            except:
                pass
        
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Feature Importance
    perm_importance = export_ctx['permutation_importance']
    if perm_importance:
        report_lines.append("## Feature Importance (Permutation)")
        report_lines.append("")
        for name, perm_data in perm_importance.items():
            report_lines.append(f"### {name.upper()}")
            importance_df = pd.DataFrame({
                'Feature': perm_data['feature_names'],
                'Importance': perm_data['importances_mean']
            }).sort_values('Importance', ascending=False)
            try:
                report_lines.append(importance_df.head(10).to_markdown(index=False, floatfmt='.4f'))
            except:
                for _, row in importance_df.head(10).iterrows():
                    report_lines.append(f"- {row['Feature']}: {row['Importance']:.4f}")
            report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # Explainability: Partial Dependence, SHAP, Bland-Altman, Robustness
    pd_data = export_ctx['pdp_results']
    shap_data = export_ctx['shap_results']
    rob = export_ctx['robustness']

    if pd_data and any(pd_data.values()):
        report_lines.append("## Partial Dependence")
        report_lines.append("")
        for name, data in pd_data.items():
            if not data:
                continue
            pd_feature_names = data.get('feature_names', [])
            pd_feature_indices = data.get('feature_indices', [])
            feats = [
                pd_feature_names[idx] if isinstance(idx, int) and idx < len(pd_feature_names) else str(idx)
                for idx in pd_feature_indices[:5]
            ]
            if not feats and data.get('pd_per_feature'):
                feats = [str(idx) for idx in list(data['pd_per_feature'].keys())[:5]]
            report_lines.append(f"**{name.upper()}:** {', '.join(feats)}{'…' if len(pd_feature_indices) > 5 else ''}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    if shap_data:
        report_lines.append("## SHAP")
        report_lines.append("")
        report_lines.append(f"Models: {', '.join(m.upper() for m in shap_data.keys())}.")
        for name, s in shap_data.items():
            ss = s.get("stats_summary", "")
            if ss:
                report_lines.append(f"- **{name.upper()}:** {ss[:120]}{'…' if len(ss) > 120 else ''}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    if rob:
        report_lines.append("## Cross-Model Robustness")
        report_lines.append("")
        report_lines.append("| Model A | Model B | Spearman ρ | Top-5 overlap |")
        report_lines.append("|---------|---------|------------|---------------|")
        for (ma, mb), v in list(rob.items())[:10]:
            r = v.get("spearman")
            r_str = f"{r:.3f}" if r is not None else "N/A"
            ov = v.get("top_k_overlap", "N/A")
            report_lines.append(f"| {ma} | {mb} | {r_str} | {ov} |")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # Bland-Altman: include only if the analysis was explicitly run and stored in workflow state.
    bland_altman_results = st.session_state.get('bland_altman_results')
    if bland_altman_results:
        report_lines.append("## 📉 Bland-Altman")
        report_lines.append("")
        report_lines.append("This section reports Bland-Altman outputs that were explicitly generated in the workflow.")
        report_lines.append("")
        report_lines.append("| Comparison | Mean diff | LoA width | % outside LoA |")
        report_lines.append("|------------|-----------|-----------|---------------|")
        for comparison, ba in list(bland_altman_results.items())[:10]:
            if not isinstance(ba, dict):
                continue
            md = ba.get("mean_diff", 0)
            w = ba.get("width_loa", 0)
            pct = ba.get("pct_outside_loa", 0)
            report_lines.append(f"| {comparison} | {md:.4f} | {w:.4f} | {pct:.1%} |")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # LLM-backed interpretations (optional)
    include_llm = export_ctx['include_llm']
    if include_llm:
        llm_items = [
            (k, v) for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith("llm_result_") and v not in ("__unavailable__", "__error__")
        ]
        if llm_items:
            report_lines.append("## LLM-backed interpretations")
            report_lines.append("")
            for k, v in llm_items[:20]:
                label = k.replace("llm_result_", "").replace("_", " ").title()
                report_lines.append(f"**{label}**")
                report_lines.append(f"{v}")
                report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

    # Recommendations
    if coach_output:
        report_lines.append("## Model Selection Coach Insights")
        report_lines.append("")
        report_lines.append(f"> {coach_output.data_sufficiency_narrative}")
        report_lines.append("")
        
        if coach_output.warnings_summary:
            report_lines.append("**Warnings:**")
            for warning in coach_output.warnings_summary[:3]:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Discussion (Draft)
    report_lines.append("## Discussion (Draft)")
    report_lines.append("")
    
    # Principal Findings - result-specific prompt
    report_lines.append("### Principal Findings")
    report_lines.append("")
    if best_model_key and best_model_key in model_results:
        best_model = model_results[best_model_key]
        if data_config.task_type == 'regression':
            primary_metric = 'RMSE'
            primary_val = best_model['metrics'].get('RMSE')
        else:
            primary_metric = 'F1' if 'F1' in best_model['metrics'] else 'Accuracy'
            primary_val = best_model['metrics'].get(primary_metric)
        
        if primary_val is not None:
            report_lines.append(f"The {best_model_key.upper()} achieved {primary_metric} of {primary_val:.4f} on held-out data. [PLACEHOLDER: Interpret this performance in clinical context]")
        else:
            report_lines.append("[PLACEHOLDER: Summarize the main results in context of the study objectives.]")
    else:
        report_lines.append("[PLACEHOLDER: Summarize the main results in context of the study objectives.]")
    report_lines.append("")
    
    # Feature importance interpretation
    perm_imp = export_ctx.get('permutation_importance', {})
    if perm_imp and best_model_key and best_model_key in perm_imp:
        pi_data = perm_imp[best_model_key]
        feat_names = pi_data.get('feature_names', [])
        importances = pi_data.get('importances_mean', [])
        if len(feat_names) > 0 and len(importances) > 0:
            sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
            top_feats = [feat_names[i] for i in sorted_idx[:3]]
            report_lines.append(f"Key predictors identified were {', '.join(top_feats)}. [PLACEHOLDER: Discuss biological plausibility and consistency with prior knowledge]")
            report_lines.append("")
    
    # Comparison with Prior Work
    report_lines.append("### Comparison with Prior Work")
    report_lines.append("")
    task_label = "regression" if data_config.task_type == "regression" else "classification"
    report_lines.append(f"[PLACEHOLDER: Compare the performance to prior work. Note: typical {task_label} models in this domain achieve...]")
    report_lines.append("")
    
    # Clinical Implications
    report_lines.append("### Clinical Implications")
    report_lines.append("")
    report_lines.append("[PLACEHOLDER: Discuss practical implications for clinical decision-making or research.]")
    report_lines.append("")
    
    # Strengths and Limitations
    report_lines.append("### Strengths and Limitations")
    report_lines.append("")
    
    # Auto-fill methodological strengths
    strength_items = []
    if len(df) > 0:
        strength_items.append(f"Sample size of {len(df):,} observations")
    if export_ctx.get('bootstrap_results'):
        strength_items.append("Bootstrap confidence intervals for uncertainty quantification")
    if export_ctx.get('shap_results'):
        strength_items.append("Model-agnostic explainability via SHAP analysis")
    if perm_imp:
        strength_items.append("Permutation importance for feature contribution assessment")
    
    if strength_items:
        report_lines.append("**Strengths:**")
        for item in strength_items:
            report_lines.append(f"- {item}")
        report_lines.append("- [PLACEHOLDER: Add study-specific strengths]")
    else:
        report_lines.append("**Strengths:** [PLACEHOLDER: Discuss methodological strengths]")
    report_lines.append("")
    
    report_lines.append("**Limitations:** [PLACEHOLDER: Discuss study limitations, data constraints, generalizability, etc.]")
    report_lines.append("")
    
    # Conclusion
    report_lines.append("### Conclusion")
    report_lines.append("")
    report_lines.append("[PLACEHOLDER: State the main conclusion and its implications.]")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Notes and Reproducibility
    # FIX 7: Decision Audit Trail Appendix
    from ml.publication import generate_decision_audit_trail
    audit_trail = generate_decision_audit_trail()
    if audit_trail:
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## Appendix: Decision Audit Trail")
        report_lines.append("")
        report_lines.append(audit_trail)
        report_lines.append("")

    report_lines.append("## 📝 Notes")
    report_lines.append("")
    report_lines.append("- This report was generated automatically by Tabular ML Lab")
    report_lines.append("- All models were evaluated on the same held-out test set")
    if pipelines_by_model:
        report_lines.append("- Preprocessing was applied per model (see Preprocessing section)")
    else:
        report_lines.append("- Preprocessing was applied consistently across all models")
    report_lines.append(f"- Random seed: {st.session_state.get('random_seed', 42)} (for reproducibility)")
    report_lines.append("")

    return "\n".join(report_lines)


# Freeze export state once per page render so downstream UI and downloads stay consistent
export_ctx = build_export_context()
render_export_readiness_audit(export_ctx)

# Generate report text once — used by Export Options and Report Preview below
report_text = generate_report(export_ctx)

# ============================================================================
# PUBLICATION TOOLS
# ============================================================================
st.header("📝 Publication Tools")

# Methods Section Generator
with st.expander("📄 Auto-Generated Methods Section", expanded=False):
    st.markdown("""
    Generate a workflow-derived draft of the methods section and, optionally, a factual results draft.
    Fill in the `[PLACEHOLDER]` sections with study-specific details and add your own interpretation separately.
    """)
    # Let user select which models to include in the report
    all_model_names = list(trained_models.keys()) if trained_models else []
    if all_model_names:
        selected_for_report = st.multiselect(
            "Models to include in report",
            options=all_model_names,
            default=all_model_names,
            key="report_model_selection",
            help="Select which models' results to include in the methods/results section.",
        )
    else:
        selected_for_report = []

    # Explainability methods to mention
    available_explain = []
    if st.session_state.get("permutation_importance"):
        available_explain.append("permutation_importance")
    if st.session_state.get("shap_results") or st.session_state.get("shap_values"):
        available_explain.append("shap")
    if st.session_state.get("pdp_results") or st.session_state.get("partial_dependence"):
        available_explain.append("partial_dependence")
    if st.session_state.get("calibration_results"):
        available_explain.append("calibration")
    if st.session_state.get("bland_altman_results"):
        available_explain.append("bland_altman")

    if available_explain:
        selected_explain = st.multiselect(
            "Explainability methods to describe",
            options=available_explain,
            default=available_explain,
            key="report_explain_selection",
            help="Select which analyses to describe in the methods section.",
        )
    else:
        selected_explain = []

    # Manuscript-primary model selection is optional and should only reflect an explicit user choice
    best_model = None
    if selected_for_report:
        default_best_model = export_ctx.get('best_model_by_metric')
        st.caption(
            f"Current best by held-out metric: {default_best_model.upper()}"
            if default_best_model else
            "Current best-by-metric model is not available."
        )
        primary_options = ["None (describe best-by-metric only)"] + selected_for_report
        stored_primary_model = st.session_state.get("report_best_model")
        default_primary_index = primary_options.index(stored_primary_model) if stored_primary_model in selected_for_report else 0
        selected_primary_option = st.selectbox(
            "Manuscript-primary model (optional)",
            options=primary_options,
            index=default_primary_index,
            key="report_best_model_selection",
            help="Select a manuscript-primary model only if you want to explicitly frame one model as primary in the draft.",
        )
        best_model = selected_primary_option if selected_primary_option in selected_for_report else None
        st.session_state["report_best_model"] = best_model

    include_results = st.checkbox("Include draft Results section with actual metrics", value=True,
                                   key="report_include_results",
                                   help="Adds a Results section populated with your model's actual performance numbers and CIs.")

    if st.button("Generate Methods Section", key="gen_methods", type="primary"):
        manuscript_context = _build_manuscript_context(
            selected_for_report=selected_for_report,
            selected_explain=selected_explain,
            include_results=include_results,
            best_model=best_model,
        )
        methods_text = _build_methods_section_for_export(manuscript_context)
        st.session_state["methods_section"] = methods_text
        st.session_state["manuscript_export_context"] = manuscript_context

    if st.session_state.get("methods_section"):
        st.markdown(st.session_state["methods_section"])
        st.download_button(
            "📥 Download Methods Section",
            st.session_state["methods_section"],
            "methods_section.md", "text/markdown",
            key="dl_methods",
        )

# CONSORT-Style Flow Diagram
with st.expander("📊 Sample Flow Diagram", expanded=False):
    st.markdown("CONSORT-style diagram showing how your sample was derived.")
    if st.button("Generate Flow Diagram", key="gen_flow"):
        from ml.publication import generate_flow_diagram_mermaid
        train_n = len(st.session_state.get('X_train', []))
        val_n = len(st.session_state.get('X_val', []))
        test_n = len(st.session_state.get('X_test', []))
        n_missing_target = df[data_config.target_col].isna().sum() if data_config.target_col else 0

        mermaid = generate_flow_diagram_mermaid(
            n_total=len(df),
            n_missing_target=n_missing_target,
            n_analyzed=len(df) - n_missing_target,
            n_train=train_n,
            n_val=val_n,
            n_test=test_n,
        )
        st.session_state["flow_diagram"] = mermaid

    if st.session_state.get("flow_diagram"):
        mermaid_code = st.session_state["flow_diagram"]
        st.text_area("Mermaid Diagram Code", value=mermaid_code, height=300, key="flow_mermaid_code",
                     help="Select all and copy, or use the download button below.")
        col_fd1, col_fd2 = st.columns(2)
        with col_fd1:
            st.download_button(
                "📥 Download Mermaid Code",
                mermaid_code,
                "flow_diagram.mmd", "text/plain",
                key="dl_flow_mermaid",
            )
        with col_fd2:
            st.link_button("🔗 Open in Mermaid Live Editor", "https://mermaid.live")
        st.caption("Paste the code into [mermaid.live](https://mermaid.live) to render as SVG/PNG for your paper.")

# TRIPOD Checklist
with st.expander("✅ TRIPOD Checklist", expanded=False):
    st.markdown("""
    The [TRIPOD statement](https://www.tripod-statement.org/) is the reporting guideline for
    prediction model studies. Track your compliance here.
    """)
    from ml.publication import TRIPODTracker, TRIPOD_ITEMS

    # TRIPOD auto-completion from ledger + workflow state
    tracker = TRIPODTracker()

    # Auto-mark from ledger resolutions (covers: missing_data, predictor_handling, etc.)
    _tripod_from_ledger = _report_ledger.get_tripod_status()
    for auto_key, completed in _tripod_from_ledger.items():
        if completed:
            # Find a resolved insight with this tripod key for the note
            note = ""
            for _ins in _report_ledger.get_resolved():
                if auto_key in _ins.tripod_keys:
                    note = _ins.resolved_by
                    break
            tracker.mark_complete(auto_key, note or "Auto-detected from analysis", "Ledger")

    # Auto-mark from workflow state (items not tracked by ledger)
    if data_config and data_config.target_col:
        tracker.mark_complete("outcome_defined", f"Target: {data_config.target_col}", "Upload & Audit")
    if data_config and data_config.feature_cols:
        tracker.mark_complete("predictors_defined", f"{len(data_config.feature_cols)} features", "Upload & Audit")
    if trained_models:
        tracker.mark_complete("model_building", f"Models: {', '.join(trained_models.keys())}", "Train & Compare")
    if model_results:
        tracker.mark_complete("performance_measures", "Test set metrics computed", "Train & Compare")
    if st.session_state.get("bootstrap_results"):
        tracker.mark_complete("performance_ci", "Bootstrap CIs computed", "Train & Compare")
    if st.session_state.get("table1_df") is not None:
        tracker.mark_complete("table1", "Table 1 generated", "EDA")
    prep_config = st.session_state.get('preprocessing_config', {})
    if prep_config:
        tracker.mark_complete("predictor_handling", "Preprocessing configured", "Preprocess")
        if prep_config.get("numeric_imputation", "none") != "none":
            tracker.mark_complete("missing_data", f"Imputation: {prep_config.get('numeric_imputation')}", "Preprocess")

    done, total = tracker.get_progress()
    st.progress(done / total)
    st.markdown(f"**{done}/{total} items addressed** (auto-completed from your workflow)")

    checklist_df = tracker.get_checklist_df()
    table(checklist_df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download TRIPOD Checklist",
        checklist_df.to_csv(index=False),
        "tripod_checklist.csv", "text/csv",
        key="dl_tripod",
    )

# Table 1 with Custom Tests
if st.session_state.get("table1_df") is not None:
    with st.expander("📋 Table 1: Study Population (with Custom Tests)", expanded=False):
        st.markdown("""
        This is your final Table 1, including any custom statistical tests you added in **Statistical Validation** (page 9).
        """)
        
        table1_display = st.session_state["table1_df"].copy()
        custom_tests = st.session_state.get('custom_table1_tests', [])
        
        if custom_tests:
            st.info(f"✅ {len(custom_tests)} custom statistical test(s) added from Statistical Validation page")
            
            # Add custom tests as additional rows
            for test in custom_tests:
                new_row = pd.DataFrame({
                    'Variable': [f"{test['variable']}"],
                    'p-value': [f"{test['p_value']:.4f}" if test['p_value'] >= 0.001 else "<0.001"]
                })
                # Add note column if not exists
                if 'Test/Note' not in table1_display.columns:
                    table1_display['Test/Note'] = ''
                new_row['Test/Note'] = f"{test['test']}: {test['statistic']} ({test['note']})"
                
                # Match other columns from original table
                for col in table1_display.columns:
                    if col not in new_row.columns:
                        new_row[col] = '—'
                
                table1_display = pd.concat([table1_display, new_row], ignore_index=True)
        
        table(table1_display, use_container_width=True, hide_index=True)
        
        # Export buttons
        col_t1_exp1, col_t1_exp2 = st.columns(2)
        with col_t1_exp1:
            csv_data = table1_display.to_csv(index=False)
            st.download_button("📥 Download CSV", csv_data, "table1_final.csv", "text/csv", key="dl_table1_final_csv")
        with col_t1_exp2:
            from ml.table_one import table1_to_latex
            try:
                latex_data = table1_to_latex(table1_display)
                st.download_button("📥 Download LaTeX", latex_data, "table1_final.tex", "text/plain", key="dl_table1_final_latex")
            except:
                st.caption("LaTeX export requires standard Table 1 format")

# LaTeX Manuscript Template
with st.expander("📝 LaTeX Manuscript Template", expanded=False):
    st.markdown("""
    Generate a **complete LaTeX manuscript** populated with your actual results.
    Compile with `pdflatex` to produce a publication-ready PDF. All placeholders
    are clearly marked with `[PLACEHOLDER]` for you to fill in study-specific details.
    """)

    col_lt1, col_lt2 = st.columns(2)
    with col_lt1:
        paper_title = st.text_input("Paper title", value="Prediction Model Development and Validation", key="latex_title")
        authors = st.text_input("Authors", value="[Author Names]", key="latex_authors")
    with col_lt2:
        affiliation = st.text_input("Affiliation", value="[Institution]", key="latex_affiliation")

    if st.button("Generate LaTeX Manuscript", key="gen_latex", type="primary"):
        from ml.latex_report import generate_latex_report

        train_n = len(st.session_state.get('X_train', []))
        val_n = len(st.session_state.get('X_val', []))
        test_n = len(st.session_state.get('X_test', []))
        table1_df_local = st.session_state.get("table1_df")
        
        # Merge custom statistical tests from page 09 into Table 1
        # Format test info within existing Table 1 columns (not as extra columns)
        custom_tests = st.session_state.get('custom_table1_tests', [])
        if custom_tests and table1_df_local is not None:
            # Add footnote marker to variable names and collect footnotes
            footnotes = []
            for idx, test in enumerate(custom_tests, start=1):
                var_name = test['variable']
                # Find matching row in table1_df_local and add footnote marker
                mask = table1_df_local.index.str.contains(var_name, case=False, na=False)
                if mask.any():
                    first_match_idx = table1_df_local.index[mask][0]
                    current_label = str(first_match_idx)
                    table1_df_local = table1_df_local.rename(index={first_match_idx: f"{current_label}^{idx}"})
                    p_str = f"{test['p_value']:.4f}" if test['p_value'] >= 0.001 else "<0.001"
                    footnotes.append(f"^{idx} {test['test']}: {test['statistic']}, p={p_str} ({test['note']})")
            
            # Store footnotes in metadata for LaTeX generation
            if footnotes:
                st.session_state['table1_custom_test_footnotes'] = footnotes
        methods_text = st.session_state.get("methods_section", "")
        if not methods_text.strip():
            manuscript_context = _build_manuscript_context(
                selected_for_report=selected_for_report,
                selected_explain=selected_explain,
                include_results=include_results,
                best_model=best_model,
            )
            methods_text = _build_methods_section_for_export(manuscript_context)
            st.session_state["methods_section"] = methods_text
            st.session_state["manuscript_export_context"] = manuscript_context

        manuscript_context = st.session_state.get("manuscript_export_context") or _build_manuscript_context(
            selected_for_report=selected_for_report,
            selected_explain=selected_explain,
            include_results=include_results,
            best_model=best_model,
        )
        st.session_state["manuscript_export_context"] = manuscript_context
        bootstrap_res = manuscript_context.get('selected_bootstrap_results') or {}
        
        # Build explainability summary from session state
        explainability_summary = {}
        perm_imp = st.session_state.get('permutation_importance', {})
        shap_results = st.session_state.get('shap_results', {})
        calibration_results = st.session_state.get('calibration_results', {})
        
        if perm_imp:
            explainability_summary['permutation_importance_available'] = True
            # Extract top features from permutation importance
            best_model_key = manuscript_context.get('manuscript_primary_model') or manuscript_context.get('best_model_by_metric')
            if best_model_key and best_model_key in perm_imp:
                pi_data = perm_imp[best_model_key]
                feat_names = pi_data.get('feature_names', [])
                importances = pi_data.get('importances_mean', [])
                if len(feat_names) > 0 and len(importances) > 0:
                    # Sort by importance
                    sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
                    explainability_summary['top_features'] = [feat_names[i] for i in sorted_idx[:10]]
        
        if shap_results:
            explainability_summary['shap_available'] = True
        
        if calibration_results:
            # Extract calibration metrics if available
            best_model_key = manuscript_context.get('manuscript_primary_model') or manuscript_context.get('best_model_by_metric')
            if best_model_key and best_model_key in calibration_results:
                cal_data = calibration_results[best_model_key]
                explainability_summary['calibration_metrics'] = {
                    k: v for k, v in cal_data.items() 
                    if isinstance(v, (int, float)) and k not in ('model', 'timestamp')
                }
        
        # Build sensitivity summary from session state
        sensitivity_summary = {}
        seed_sensitivity = st.session_state.get('sensitivity_seed_results')
        if seed_sensitivity is not None and not seed_sensitivity.empty:
            # Calculate CV% for the primary metric
            metric_cols = [c for c in seed_sensitivity.columns if c not in ('seed', '_error')]
            if metric_cols:
                metric_vals = seed_sensitivity[metric_cols[0]].dropna()
                if len(metric_vals) > 1:
                    cv_pct = (metric_vals.std() / metric_vals.mean() * 100) if metric_vals.mean() != 0 else 0
                    sensitivity_summary['seed_stability'] = {
                        'cv_percent': cv_pct,
                        'range': f"{metric_vals.min():.4f} to {metric_vals.max():.4f}"
                    }
        
        feature_dropout = st.session_state.get('sensitivity_feature_dropout')
        if feature_dropout is not None:
            sensitivity_summary['feature_dropout_conducted'] = True

        # FIX 4: Build statistical validation summary from methodology log
        stat_validation_summary = []
        methodology_log = st.session_state.get('methodology_log', [])
        for entry in methodology_log:
            if entry.get('step') == 'Statistical Validation':
                details = entry.get('details', {})
                action = entry.get('action', '')
                stat_validation_summary.append({
                    'test_name': details.get('test_name') or action,
                    'variable': details.get('variable', 'unknown variable'),
                    'statistic': details.get('statistic'),
                    'p_value': details.get('p_value'),
                })

        latex_source = generate_latex_report(
            title=paper_title,
            authors=authors,
            affiliation=affiliation,
            methods_section=methods_text,
            table1_df=table1_df_local,
            model_results=manuscript_context.get('selected_model_results'),
            bootstrap_results=bootstrap_res,
            task_type=data_config.task_type or "regression",
            feature_names=manuscript_context.get('feature_names_for_manuscript'),
            target_name=data_config.target_col,
            n_total=len(df),
            n_train=train_n,
            n_val=val_n,
            n_test=test_n,
            explainability_summary=explainability_summary if explainability_summary else None,
            sensitivity_summary=sensitivity_summary if sensitivity_summary else None,
            stat_validation_summary=stat_validation_summary if stat_validation_summary else None,
            manuscript_context=manuscript_context,
        )
        st.session_state["latex_report"] = latex_source

    if st.session_state.get("latex_report"):
        st.text_area("LaTeX Source", value=st.session_state["latex_report"], height=400, key="latex_preview")
        st.download_button(
            "📥 Download LaTeX (.tex)",
            st.session_state["latex_report"],
            "manuscript.tex", "text/plain",
            key="dl_latex",
        )
        st.caption("Compile with: `pdflatex manuscript.tex` · Requires `booktabs`, `natbib`, `hyperref` packages.")

st.markdown("---")

# ============================================================================
# EXPORT OPTIONS
# ============================================================================
st.header("💾 Export Options")

# Export configuration
with st.expander("Export Configuration"):
    export_models = st.checkbox("Include trained model artifacts (joblib/pickle)", value=True)
    export_predictions = st.checkbox("Include predictions CSV", value=True)
    export_plots = st.checkbox("Include plots in zip", value=False)
    if export_plots:
        st.caption("⚠️ Plot export renders images server-side and may be slow with many models. Select only what you need.")
        st.caption("💡 Tip: You can also download any individual plot by hovering over it and clicking the 📷 camera icon.")
        plot_col1, plot_col2, plot_col3 = st.columns(3)
        with plot_col1:
            export_plots_train = st.checkbox("Training plots", value=True, help="Predictions, residuals, confusion matrices, ROC/PR curves")
        with plot_col2:
            export_plots_explain = st.checkbox("Explainability plots", value=True, help="Permutation importance bar charts")
        with plot_col3:
            export_plots_sensitivity = st.checkbox("Sensitivity plots", value=False, help="Seed sensitivity charts")
    else:
        export_plots_train = False
        export_plots_explain = False
        export_plots_sensitivity = False
    include_raw_data = st.checkbox("Include raw data sample (first 100 rows)", value=False)
    st.checkbox("Include LLM interpretations in report", value=False, key="report_include_llm")


# Helper function to save plotly figures as images
def save_plotly_fig(fig, filename: str) -> Optional[bytes]:
    """Save plotly figure as PNG bytes."""
    try:
        return fig.to_image(format="png", width=1200, height=800)
    except Exception:
        try:
            return fig.to_image(format="png", width=1200, height=800, engine="kaleido")
        except Exception:
            return None


def save_matplotlib_fig(fig, filename: str) -> Optional[bytes]:
    """Save matplotlib figure as PNG bytes."""
    try:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


def export_model_artifact(model_wrapper, model_key: str) -> Optional[bytes]:
    """Export trained model as bytes."""
    try:
        import joblib
        buffer = io.BytesIO()
        
        # For neural networks, export the sklearn-compatible wrapper
        if model_key == 'nn' and hasattr(model_wrapper, '_sklearn_estimator'):
            # Export the sklearn estimator (lighter weight)
            joblib.dump(model_wrapper._sklearn_estimator, buffer)
        elif hasattr(model_wrapper, 'model'):
            # Export the underlying model
            joblib.dump(model_wrapper.model, buffer)
        else:
            # Export the wrapper itself
            joblib.dump(model_wrapper, buffer)
        
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Could not export model {model_key}: {e}")
        return None


# Download buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="Download Report (Markdown)",
        data=report_text,
        file_name=f"modeling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        type="primary"
    )

with col2:
    # Quick metrics CSV
    comparison_data = []
    for name, results in model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    comparison_df = pd.DataFrame(comparison_data)
    
    st.download_button(
        label="Download Metrics (CSV)",
        data=comparison_df.to_csv(index=False),
        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col3:
    # Create comprehensive zip package
    # Get selected_model_params from session_state (needed for export)
    selected_model_params = st.session_state.get('selected_model_params', {})
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Report
        zip_file.writestr("report.md", report_text)
        
        # Metadata JSON
        metadata = generate_metadata()
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
        
        # Metrics CSV
        zip_file.writestr("metrics.csv", comparison_df.to_csv(index=False))
        
        # Predictions CSV
        if export_predictions:
            for name, results in model_results.items():
                pred_df = pd.DataFrame({
                    'Actual': results['y_test'],
                    'Predicted': results['y_test_pred']
                })
                zip_file.writestr(f"predictions/{name}_predictions.csv", pred_df.to_csv(index=False))
        
        # Model artifacts
        if export_models:
            for model_key, model_wrapper in trained_models.items():
                model_bytes = export_model_artifact(model_wrapper, model_key)
                if model_bytes:
                    zip_file.writestr(f"models/{model_key}_model.joblib", model_bytes)
                    
                    # For NN, also export model info
                    if model_key == 'nn':
                        model_info = {
                            'type': 'neural_network',
                            'params': selected_model_params.get(model_key, {}),
                            'note': 'Use joblib.load() to load the sklearn-compatible estimator'
                        }
                        zip_file.writestr(f"models/{model_key}_info.json", json.dumps(model_info, indent=2))
        
        # Preprocessing artifacts
        preprocessing_manifest = {'global_pipeline': None, 'per_model': {}}
        if pipeline:
            try:
                import joblib
                pipeline_buffer = io.BytesIO()
                joblib.dump(pipeline, pipeline_buffer)
                zip_file.writestr("preprocessing_pipeline.joblib", pipeline_buffer.getvalue())
                preprocessing_manifest['global_pipeline'] = 'preprocessing_pipeline.joblib'
            except Exception as e:
                logger.warning(f"Could not export pipeline: {e}")

        for model_key, model_pipeline in export_ctx['pipelines_by_model'].items():
            artifact_path = f"preprocessing/{model_key}_pipeline.joblib"
            config_path = f"preprocessing/{model_key}_config.json"
            entry = {'pipeline_artifact': None, 'config_artifact': None}
            try:
                import joblib
                pipeline_buffer = io.BytesIO()
                joblib.dump(model_pipeline, pipeline_buffer)
                zip_file.writestr(artifact_path, pipeline_buffer.getvalue())
                entry['pipeline_artifact'] = artifact_path
            except Exception as e:
                logger.warning(f"Could not export preprocessing pipeline for {model_key}: {e}")
            cfg = export_ctx['configs_by_model'].get(model_key)
            if cfg is not None:
                zip_file.writestr(config_path, json.dumps(cfg, indent=2, default=str))
                entry['config_artifact'] = config_path
            preprocessing_manifest['per_model'][model_key] = entry
        
        # Plots
        if export_plots and (export_plots_train or export_plots_explain or export_plots_sensitivity):
            from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, precision_recall_curve, auc as sk_auc

            if export_plots_train:
                for name, results in model_results.items():
                    y_true = results['y_test']
                    y_pred = results['y_test_pred']

                    if data_config.task_type == 'regression':
                        fig = px.scatter(
                            x=y_true, y=y_pred,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title=f"{name.upper()} - Predictions vs Actual"
                        )
                        fig.add_trace(go.Scatter(
                            x=[min(y_true), max(y_true)],
                            y=[min(y_true), max(y_true)],
                            mode='lines', name='Perfect', line=dict(dash='dash', color='red')
                        ))
                        plot_bytes = save_plotly_fig(fig, f"plot_{name}.png")
                        if plot_bytes:
                            zip_file.writestr(f"plots/train/{name}_predictions.png", plot_bytes)

                        residuals = np.array(y_true) - np.array(y_pred)
                        fig_res = px.histogram(residuals, nbins=30, title=f"{name.upper()} - Residual Distribution",
                                               labels={'value': 'Residual', 'count': 'Count'})
                        plot_bytes = save_plotly_fig(fig_res, f"resid_{name}.png")
                        if plot_bytes:
                            zip_file.writestr(f"plots/train/{name}_residuals.png", plot_bytes)
                    else:
                        cm = sk_confusion_matrix(y_true, y_pred)
                        fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title=f"{name.upper()} - Confusion Matrix",
                                           labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
                        plot_bytes = save_plotly_fig(fig_cm, f"cm_{name}.png")
                        if plot_bytes:
                            zip_file.writestr(f"plots/train/{name}_confusion_matrix.png", plot_bytes)

                        y_proba = results.get('y_test_proba')
                        if y_proba is not None:
                            try:
                                unique_classes = np.unique(y_true)
                                if len(unique_classes) == 2:
                                    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                                    fpr, tpr, _ = roc_curve(y_true, proba_pos)
                                    roc_auc_val = sk_auc(fpr, tpr)
                                    fig_roc = px.area(x=fpr, y=tpr, labels=dict(x="FPR", y="TPR"),
                                                      title=f"{name.upper()} - ROC Curve (AUC={roc_auc_val:.3f})")
                                    plot_bytes = save_plotly_fig(fig_roc, f"roc_{name}.png")
                                    if plot_bytes:
                                        zip_file.writestr(f"plots/train/{name}_roc_curve.png", plot_bytes)

                                    prec, rec, _ = precision_recall_curve(y_true, proba_pos)
                                    pr_auc_val = sk_auc(rec, prec)
                                    fig_pr = px.area(x=rec, y=prec, labels=dict(x="Recall", y="Precision"),
                                                     title=f"{name.upper()} - PR Curve (AUC={pr_auc_val:.3f})")
                                    plot_bytes = save_plotly_fig(fig_pr, f"pr_{name}.png")
                                    if plot_bytes:
                                        zip_file.writestr(f"plots/train/{name}_pr_curve.png", plot_bytes)
                            except Exception:
                                pass

            if export_plots_explain:
                # Permutation importance (Plotly)
                perm_data = st.session_state.get('permutation_importance', {})
                for name, pi in perm_data.items():
                    try:
                        fn = pi.get('feature_names', [])
                        imp = pi.get('importances_mean', [])
                        if len(fn) > 0 and len(imp) > 0:
                            sort_idx = np.argsort(imp)[::-1][:15]
                            fig_pi = px.bar(x=np.array(imp)[sort_idx], y=np.array(fn)[sort_idx],
                                            orientation='h', title=f"{name.upper()} - Permutation Importance",
                                            labels={'x': 'Importance', 'y': 'Feature'})
                            fig_pi.update_layout(yaxis=dict(autorange="reversed"))
                            plot_bytes = save_plotly_fig(fig_pi, f"pi_{name}.png")
                            if plot_bytes:
                                zip_file.writestr(f"plots/explainability/{name}_permutation_importance.png", plot_bytes)
                    except Exception:
                        pass
                
                # SHAP plots (matplotlib)
                shap_figs = export_ctx['shap_figs']
                for fig_key, fig in shap_figs.items():
                    try:
                        plot_bytes = save_matplotlib_fig(fig, f"{fig_key}.png")
                        if plot_bytes:
                            zip_file.writestr(f"plots/explainability/{fig_key}.png", plot_bytes)
                    except Exception:
                        pass

            if export_plots_sensitivity:
                seed_df = st.session_state.get('sensitivity_seed_results')
                if seed_df is not None:
                    try:
                        metric_cols = [c for c in seed_df.columns if c not in ('seed', '_error')]
                        if metric_cols:
                            fig_seed = px.bar(seed_df, x='seed', y=metric_cols[0], title=f"Seed Sensitivity - {metric_cols[0]}")
                            plot_bytes = save_plotly_fig(fig_seed, "seed_sensitivity.png")
                            if plot_bytes:
                                zip_file.writestr(f"plots/sensitivity/seed_sensitivity.png", plot_bytes)
                    except Exception:
                        pass
        
        # Raw data sample
        if include_raw_data:
            zip_file.writestr("data_sample.csv", df.head(100).to_csv(index=False))
        
        # Feature names
        feature_names = st.session_state.get('feature_names', [])
        if feature_names:
            zip_file.writestr("feature_names.txt", "\n".join(feature_names))
        
        # Manifest with summary
        manifest = {
            'export_timestamp': datetime.now().isoformat(),
            'models_trained': list(trained_models.keys()),
            'metrics_summary': {name: results['metrics'] for name, results in model_results.items()},
            'preprocessing_available': pipeline is not None or bool(export_ctx['pipelines_by_model']),
            'permutation_importance_available': len(export_ctx['permutation_importance']) > 0,
            'export_readiness': export_ctx['readiness'],
            'best_model_by_metric': export_ctx.get('best_model_by_metric'),
            'best_metric_name': export_ctx.get('best_metric_name'),
            'manuscript_primary_model': export_ctx.get('manuscript_primary_model'),
            'preprocessing_artifacts': preprocessing_manifest
        }
        if selected_model_params:
            manifest['model_hyperparameters'] = selected_model_params
        zip_file.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
    
    st.download_button(
        label="Download Complete Package (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"modeling_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

st.success("Report generated successfully!")

# ============================================================================
# REPORT PREVIEW
# ============================================================================
st.header("Report Preview")

# Display in a nice container
with st.container():
    st.markdown(report_text)

# ============================================================================
# STATE DEBUG
# ============================================================================
with st.expander("Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    st.write(f"• Data shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• Trained models: {len(trained_models)}")
    st.write(f"• Dataset profile: {'Available' if profile else 'Not computed'}")
    st.write(f"• Coach output: {'Available' if coach_output else 'Not computed'}")
    st.write(f"• Git info: {get_git_info()}")
