"""
Page 06: Report Export
Generate and download comprehensive modeling report with trained artifacts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
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
from utils.storyline import render_progress_indicator, get_insights_by_category, render_breadcrumb, render_page_navigation
from ml.model_registry import get_registry

logger = logging.getLogger(__name__)

init_session_state()

from utils.theme import inject_custom_css, render_step_indicator, render_guidance
st.set_page_config(page_title="Report Export", page_icon="📄", layout="wide")
inject_custom_css()
render_step_indicator(7, "Report Export")
st.title("📄 Report Export")
render_breadcrumb("09_Report_Export")
render_page_navigation("09_Report_Export")

# Progress indicator
render_progress_indicator("09_Report_Export")

# Guardrail: Report Export is primarily for prediction mode
task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ **Report Export is primarily designed for Prediction mode.**")
    st.info("""
    Please go to the **Upload & Audit** page and select **Prediction** as your task mode.
    Comprehensive modeling reports include trained models, metrics, and explainability results.
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

if not trained_models:
    st.warning("Please train models first")
    st.stop()

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


def generate_report() -> str:
    """Generate markdown report with improved structure and aesthetics."""
    report_lines = []
    
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
    if data_config.task_type == 'regression':
        best_model = min(model_results.items(), key=lambda x: x[1]['metrics'].get('RMSE', float('inf')))
        report_lines.append(f"**Best Model:** {best_model[0].upper()}")
        report_lines.append(f"**Test RMSE:** {best_model[1]['metrics']['RMSE']:.4f}")
        report_lines.append(f"**Test R²:** {best_model[1]['metrics']['R2']:.4f}")
    else:
        best_model = max(model_results.items(), key=lambda x: x[1]['metrics'].get('F1', x[1]['metrics'].get('Accuracy', 0)))
        report_lines.append(f"**Best Model:** {best_model[0].upper()}")
        report_lines.append(f"**Test Accuracy:** {best_model[1]['metrics'].get('Accuracy', 'N/A'):.4f}")
        if 'F1' in best_model[1]['metrics']:
            report_lines.append(f"**Test F1:** {best_model[1]['metrics']['F1']:.4f}")
    
    report_lines.append("")
    
    # Key findings
    if profile and profile.warnings:
        report_lines.append("**Key Data Warnings:**")
        for w in profile.warnings[:3]:
            report_lines.append(f"- {w.short_message}")
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
    
    # Key insights after pre-processing (EDA + preprocessing)
    insights = get_insights_by_category()
    eda_insights = [i for i in insights if i.get("category") != "preprocessing"]
    prep_insights = [i for i in insights if i.get("category") == "preprocessing"]
    if eda_insights or prep_insights:
        report_lines.append("## Key insights after pre-processing")
        report_lines.append("")
        if eda_insights:
            report_lines.append("### From EDA")
            report_lines.append("")
            for insight in eda_insights:
                report_lines.append(f"**{insight.get('category', 'General').title()}:** {insight['finding']}")
                report_lines.append(f"→ {insight['implication']}")
                report_lines.append("")
        if prep_insights:
            report_lines.append("### From preprocessing")
            report_lines.append("")
            for insight in prep_insights:
                report_lines.append(f"- {insight['finding']}")
                report_lines.append(f"  → {insight['implication']}")
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
    pipelines_by_model = st.session_state.get("preprocessing_pipelines_by_model") or {}
    configs_by_model = st.session_state.get("preprocessing_config_by_model") or {}
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
    selected_model_params = st.session_state.get('selected_model_params', {})
    feature_names = st.session_state.get('feature_names', [])
    
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
    perm_importance = st.session_state.get('permutation_importance', {})
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
    pd_data = st.session_state.get("partial_dependence") or {}
    shap_data = st.session_state.get("shap_results") or {}
    rob = st.session_state.get("explainability_robustness") or {}

    if pd_data and any(pd_data.values()):
        report_lines.append("## Partial Dependence")
        report_lines.append("")
        for name, data in pd_data.items():
            if not data:
                continue
            feats = list(data.keys())[:5]
            report_lines.append(f"**{name.upper()}:** {', '.join(feats)}{'…' if len(data) > 5 else ''}")
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

    # Bland-Altman (compute from model pairs when we have predictions)
    try:
        from ml.eval import analyze_bland_altman
        model_keys = list(trained_models.keys())
        ba_pairs = []
        for i, ma in enumerate(model_keys):
            for mb in model_keys[i + 1 :]:
                ra = model_results.get(ma, {})
                rb = model_results.get(mb, {})
                ya = ra.get("y_test_pred")
                yb = rb.get("y_test_pred")
                if ya is not None and yb is not None and len(ya) == len(yb):
                    ba = analyze_bland_altman(ya, yb)
                    if ba:
                        ba_pairs.append((ma, mb, ba))
        if ba_pairs:
            report_lines.append("## 📉 Bland-Altman")
            report_lines.append("")
            report_lines.append("| Model A | Model B | Mean diff | LoA width | % outside LoA |")
            report_lines.append("|---------|---------|-----------|-----------|---------------|")
            for ma, mb, ba in ba_pairs[:10]:
                md = ba.get("mean_diff", 0)
                w = ba.get("width_loa", 0)
                pct = ba.get("pct_outside_loa", 0)
                report_lines.append(f"| {ma} | {mb} | {md:.4f} | {w:.4f} | {pct:.1%} |")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
    except Exception:
        pass

    # LLM-backed interpretations (optional)
    include_llm = st.session_state.get("report_include_llm", False)
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
    
    # Notes and Reproducibility
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


# ============================================================================
# REPORT PREVIEW
# ============================================================================
st.header("Report Preview")

# Generate report
report_text = generate_report()

# Display in a nice container
with st.container():
    st.markdown(report_text)

# ============================================================================
# PUBLICATION TOOLS
# ============================================================================
st.header("📝 Publication Tools")

# Methods Section Generator
with st.expander("📄 Auto-Generated Methods Section", expanded=False):
    st.markdown("""
    Generate a draft methods section based on your actual workflow choices.
    Fill in the `[PLACEHOLDER]` sections with study-specific details.
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
    if st.session_state.get("shap_values"):
        available_explain.append("shap")
    if st.session_state.get("bootstrap_results"):
        available_explain.append("calibration")

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

    # Best model selection
    best_model = None
    if selected_for_report:
        best_model = st.selectbox(
            "Best/primary model (highlighted in results)",
            options=selected_for_report,
            index=0,
            key="report_best_model",
        )

    include_results = st.checkbox("Include draft Results section with actual metrics", value=True,
                                   key="report_include_results",
                                   help="Adds a Results section populated with your model's actual performance numbers and CIs.")

    if st.button("Generate Methods Section", key="gen_methods", type="primary"):
        from ml.publication import generate_methods_section
        train_n = len(st.session_state.get('X_train', []))
        val_n = len(st.session_state.get('X_val', []))
        test_n = len(st.session_state.get('X_test', []))
        # Prefer the detailed preprocessing_summary; fall back to raw config
        prep_config = st.session_state.get('preprocessing_summary') or st.session_state.get('preprocessing_config', {})

        # Filter model results to selected models
        selected_results = {k: v for k, v in model_results.items() if k in selected_for_report} if include_results else None
        selected_bootstrap = {k: v for k, v in st.session_state.get("bootstrap_results", {}).items() if k in selected_for_report} if include_results else None

        fs_results = st.session_state.get("feature_selection_results")
        fs_method = fs_results[0].method if fs_results else None

        methods_text = generate_methods_section(
            data_config={},
            preprocessing_config=prep_config,
            model_configs={name: {} for name in selected_for_report},
            split_config={},
            n_total=len(df),
            n_train=train_n,
            n_val=val_n,
            n_test=test_n,
            feature_names=data_config.feature_cols,
            target_name=data_config.target_col,
            task_type=data_config.task_type or "regression",
            metrics_used=list(next(iter(model_results.values()))['metrics'].keys()) if model_results else ["RMSE"],
            feature_selection_method=fs_method,
            selected_model_results=selected_results,
            bootstrap_results=selected_bootstrap,
            best_model_name=best_model,
            explainability_methods=selected_explain,
            random_seed=st.session_state.get("random_seed", 42),
        )
        st.session_state["methods_section"] = methods_text

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

    if "tripod_tracker" not in st.session_state:
        tracker = TRIPODTracker()
        # Auto-mark items we can detect
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
        st.session_state["tripod_tracker"] = tracker

    tracker = st.session_state["tripod_tracker"]
    done, total = tracker.get_progress()
    st.progress(done / total)
    st.markdown(f"**{done}/{total} items addressed**")

    checklist_df = tracker.get_checklist_df()
    st.dataframe(checklist_df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download TRIPOD Checklist",
        checklist_df.to_csv(index=False),
        "tripod_checklist.csv", "text/csv",
        key="dl_tripod",
    )

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
        methods_text = st.session_state.get("methods_section", "")
        bootstrap_res = st.session_state.get("bootstrap_results")

        latex_source = generate_latex_report(
            title=paper_title,
            authors=authors,
            affiliation=affiliation,
            methods_section=methods_text,
            table1_df=table1_df_local,
            model_results=model_results,
            bootstrap_results=bootstrap_res,
            task_type=data_config.task_type or "regression",
            feature_names=data_config.feature_cols,
            target_name=data_config.target_col,
            n_total=len(df),
            n_train=train_n,
            n_val=val_n,
            n_test=test_n,
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
    export_plots = st.checkbox("Include plots (requires kaleido)", value=False)
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
        
        # Preprocessing pipeline
        if pipeline:
            try:
                import joblib
                pipeline_buffer = io.BytesIO()
                joblib.dump(pipeline, pipeline_buffer)
                zip_file.writestr("preprocessing_pipeline.joblib", pipeline_buffer.getvalue())
            except Exception as e:
                logger.warning(f"Could not export pipeline: {e}")
        
        # Plots
        if export_plots:
            for name, results in model_results.items():
                if data_config.task_type == 'regression':
                    fig = px.scatter(
                        x=results['y_test'], y=results['y_test_pred'],
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title=f"{name.upper()} - Predictions vs Actual"
                    )
                    fig.add_trace(go.Scatter(
                        x=[min(results['y_test']), max(results['y_test'])],
                        y=[min(results['y_test']), max(results['y_test'])],
                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')
                    ))
                    plot_bytes = save_plotly_fig(fig, f"plot_{name}.png")
                    if plot_bytes:
                        zip_file.writestr(f"plots/{name}_predictions.png", plot_bytes)
        
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
            'preprocessing_available': pipeline is not None,
            'permutation_importance_available': len(st.session_state.get('permutation_importance', {})) > 0
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
