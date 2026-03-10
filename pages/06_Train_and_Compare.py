"""
Page 04: Train and Compare Models
Train models, evaluate, compare metrics, show diagnostics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig, set_splits, add_trained_model,
    TaskTypeDetection, CohortStructureDetection, log_methodology,
)
from utils.seed import set_global_seed, get_global_seed
from utils.storyline import get_insights_by_category, render_breadcrumb, render_page_navigation
from utils.theme import inject_custom_css, render_guidance, render_reviewer_concern, render_step_indicator, render_metric_row, render_sidebar_workflow
from ml.splits import to_numpy_1d

logger = logging.getLogger(__name__)

# Lazy imports for heavy packages - only load when needed
def _get_plotly():
    """Lazy import plotly."""
    import plotly.graph_objects as go
    import plotly.express as px
    return go, px

def _get_sklearn_splits():
    """Lazy import sklearn model_selection."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
    return train_test_split, GroupShuffleSplit, GroupKFold

def _get_model_wrappers():
    """Lazy import model wrappers - these load torch/sklearn models."""
    from models.nn_whuber import NNWeightedHuberWrapper
    from models.glm import GLMWrapper
    from models.huber_glm import HuberGLMWrapper
    from models.rf import RFWrapper
    from models.registry_wrappers import RegistryModelWrapper
    return NNWeightedHuberWrapper, GLMWrapper, HuberGLMWrapper, RFWrapper, RegistryModelWrapper

def _get_eval_functions():
    """Lazy import evaluation functions."""
    from ml.eval import (
        calculate_regression_metrics, calculate_classification_metrics,
        perform_cross_validation, analyze_residuals
    )
    return calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals

def _get_visualization_functions():
    """Lazy import visualization functions with fallback."""
    try:
        from visualizations import plot_training_history, plot_predictions_vs_actual, plot_residuals
        return plot_training_history, plot_predictions_vs_actual, plot_residuals
    except ImportError:
        # Fallback if visualizations module not found
        import plotly.graph_objects as go
        def plot_training_history(history):
            fig = go.Figure()
            epochs = range(1, len(history['train_loss']) + 1)
            fig.add_trace(go.Scatter(x=list(epochs), y=history['train_loss'], name='Train Loss'))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss'))
            return fig
        def plot_predictions_vs_actual(y_true, y_pred, title=""):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'))
            return fig
        def plot_residuals(y_true, y_pred, title=""):
            residuals = y_true - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
            return fig
        return plot_training_history, plot_predictions_vs_actual, plot_residuals

init_session_state()

# Set global seed
set_global_seed(st.session_state.get('random_seed', 42))

st.set_page_config(page_title="Train & Compare", page_icon="🧠", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="05_Train")
render_step_indicator(5, "Train & Compare Models")
st.title("🧠 Train & Compare Models")
render_breadcrumb("06_Train_and_Compare")
render_page_navigation("06_Train_and_Compare")

# Progress indicator

# Global random seed control
with st.sidebar:
    st.header("Global Settings")
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.get('random_seed', 42),
        help="Controls randomness for reproducibility"
    )
    if random_seed != st.session_state.get('random_seed', 42):
        st.session_state.random_seed = random_seed
        set_global_seed(random_seed)
        st.info("Seed updated. Re-run splits and training to apply.")

# Check prerequisites
df = get_data()
if df is None:
    st.warning("Please upload data first")
    st.stop()
if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your dataset is empty. Please upload data with at least one row and one column.")
    st.stop()

# Guardrail: Training is only for prediction mode
task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ **Model Training is only available in Prediction mode.**")
    st.info("""
    Please go to the **Upload & Audit** page and select **Prediction** as your task mode.
    Model training is used to build predictive models.
    """)
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("Please configure target and features")
    st.stop()

pipelines_by_model = st.session_state.get('preprocessing_pipelines_by_model', {})
pipeline = get_preprocessing_pipeline()
if pipeline is None and not pipelines_by_model:
    st.warning("Please build preprocessing pipeline first")
    st.stop()

# Get final detection values
task_type_detection: TaskTypeDetection = st.session_state.get('task_type_detection', TaskTypeDetection())
cohort_structure_detection: CohortStructureDetection = st.session_state.get('cohort_structure_detection', CohortStructureDetection())

task_type_final = task_type_detection.final if task_type_detection.final else data_config.task_type
cohort_type_final = cohort_structure_detection.final if cohort_structure_detection.final else 'cross_sectional'
entity_id_final = cohort_structure_detection.entity_id_final

# Use final task type for downstream logic
if task_type_final:
    data_config.task_type = task_type_final

# Split configuration
st.header("Data Splitting")

# Longitudinal data handling
use_group_split = False
if cohort_type_final == 'longitudinal' and entity_id_final:
    st.info(f"Longitudinal data detected. Entity ID: `{entity_id_final}`. Using group-based splitting to prevent data leakage.")
    use_group_split = True
    if entity_id_final not in df.columns:
        st.error(f"Entity ID column '{entity_id_final}' not found in data. Please check Upload & Audit page.")
        st.stop()
elif cohort_type_final == 'longitudinal' and not entity_id_final:
    st.warning("Longitudinal data detected but no entity ID column found. Consider selecting an entity ID in Upload & Audit page.")
    if data_config.datetime_col:
        st.info("Using time-based split as fallback for longitudinal data.")

# Time-series split option
use_time_split = False
if data_config.datetime_col:
    time_split_default = st.session_state.get('train_use_time_split')
    if time_split_default is None:
        time_split_default = (cohort_type_final == 'longitudinal' and not entity_id_final)
    use_time_split = st.checkbox(
        "Use Time-Based Split",
        value=time_split_default,
        disabled=use_group_split,
        key="train_use_time_split",
        help="Split data chronologically instead of randomly (recommended for time-series)"
    )
    if not use_time_split and not use_group_split:
        st.warning("Datetime column detected but random split selected. Consider using time-based split for time-series data.")

col1, col2, col3 = st.columns(3)

# Read split sizes from session_state or use defaults
split_config_existing = st.session_state.get('split_config')
train_size_default = int((split_config_existing.train_size * 100) if split_config_existing and split_config_existing.train_size else 70)
val_size_default = int((split_config_existing.val_size * 100) if split_config_existing and split_config_existing.val_size else 15)
test_size_default = int((split_config_existing.test_size * 100) if split_config_existing and split_config_existing.test_size else 15)

with col1:
    train_size = st.slider("Train %", 50, 90, train_size_default, key="train_split_train_pct") / 100
with col2:
    val_size = st.slider("Val %", 5, 30, val_size_default, key="train_split_val_pct") / 100
with col3:
    test_size = st.slider("Test %", 5, 30, test_size_default, key="train_split_test_pct") / 100

if abs(train_size + val_size + test_size - 1.0) > 0.01:
    st.error("Splits must sum to 100%")
    st.stop()

split_config = SplitConfig(
    train_size=train_size,
    val_size=val_size,
    test_size=test_size,
    random_state=st.session_state.get('random_seed', 42),
    stratify=(task_type_final == 'classification' and not use_time_split and not use_group_split),
    use_time_split=use_time_split,
    datetime_col=data_config.datetime_col if use_time_split else None
)
st.session_state.split_config = split_config

# Cross-validation option - read from session_state
use_cv_default = st.session_state.get('use_cv', False)
use_cv = st.checkbox("Enable Cross-Validation", value=use_cv_default, key="train_use_cv")
if use_cv:
    cv_folds_default = st.session_state.get('cv_folds', 5)
    cv_folds = st.slider("CV Folds", 3, 10, cv_folds_default, key="train_cv_folds")
    st.session_state.use_cv = True
    st.session_state.cv_folds = cv_folds
else:
    st.session_state.use_cv = False

# Prepare data splits
if st.button("Prepare Splits", type="primary"):
    try:
        t0 = time.perf_counter()
        train_test_split, GroupShuffleSplit, GroupKFold = _get_sklearn_splits()
        from sklearn.preprocessing import LabelEncoder

        X = df[data_config.feature_cols].copy()
        y = df[data_config.target_col].copy()
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        original_indices = np.where(mask)[0]
        indices = np.arange(len(X))
        
        if len(X) < 2:
            st.error("Not enough samples after removing missing target values. Need at least 2 rows for train/test split.")
            st.stop()

        target_is_categorical = y.dtype.name in ("object", "category", "bool") or (
            hasattr(y.dtype, "kind") and y.dtype.kind in ("O", "b")
        )
        
        # Single-class validation for classification
        if task_type_final == 'classification':
            n_classes = y.nunique()
            if n_classes < 2:
                st.error(f"""
                **Single-class target detected:** Your target has only {n_classes} unique value(s) after removing missing values.
                
                Classification requires at least 2 classes. Please check:
                - That your target column has multiple distinct values
                - That filtering (e.g., plausibility) did not remove all samples of one class
                """)
                st.stop()
        
        le = None
        if target_is_categorical:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            st.session_state["target_label_encoder"] = le
        else:
            st.session_state.pop("target_label_encoder", None)
        
        # Split data (group-based, time-based, or random)
        if use_group_split and entity_id_final:
            groups = to_numpy_1d(df.iloc[original_indices][entity_id_final])
            y_arr = to_numpy_1d(y)

            gss = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=split_config.random_state)
            train_idx, temp_idx = next(gss.split(indices, y_arr, groups))

            groups_temp = groups[temp_idx]
            rel_val = val_size / (val_size + test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - rel_val), random_state=split_config.random_state)
            val_idx, test_idx = next(gss2.split(indices[temp_idx], y_arr[temp_idx], groups_temp))

            X_train = X.iloc[train_idx]
            X_val = X.iloc[temp_idx[val_idx]]
            X_test = X.iloc[temp_idx[test_idx]]
            y_train = y_arr[train_idx]
            y_val = y_arr[temp_idx[val_idx]]
            y_test = y_arr[temp_idx[test_idx]]

            n_train_groups = len(np.unique(groups[train_idx]))
            n_val_groups = len(np.unique(groups[temp_idx[val_idx]]))
            n_test_groups = len(np.unique(groups[temp_idx[test_idx]]))
            st.info(f"Group-based split: {n_train_groups} train groups, {n_val_groups} val groups, {n_test_groups} test groups")
        elif split_config.use_time_split and data_config.datetime_col:
            df_work = df.iloc[original_indices].copy()
            df_work["_temp_index"] = np.arange(len(df_work))
            df_work = df_work.sort_values(data_config.datetime_col)

            n_total = len(df_work)
            n_train = int(n_total * train_size)
            n_val = int(n_total * val_size)

            train_pos = df_work.iloc[:n_train]["_temp_index"].values
            val_pos = df_work.iloc[n_train : n_train + n_val]["_temp_index"].values
            test_pos = df_work.iloc[n_train + n_val :]["_temp_index"].values

            X_train = X.iloc[train_pos]
            X_val = X.iloc[val_pos]
            X_test = X.iloc[test_pos]
            y_train = to_numpy_1d(y.iloc[train_pos])
            y_val = to_numpy_1d(y.iloc[val_pos])
            y_test = to_numpy_1d(y.iloc[test_pos])
            train_indices = original_indices[train_pos]
            val_indices = original_indices[val_pos]
            test_indices = original_indices[test_pos]

            st.info(f"Time-based split: Train={df_work.iloc[0][data_config.datetime_col]} to {df_work.iloc[n_train - 1][data_config.datetime_col]}")
        elif split_config.stratify and task_type_final == 'classification':
            idx_train, idx_temp, y_train, y_temp = train_test_split(
                indices, y, test_size=(val_size + test_size),
                random_state=split_config.random_state, stratify=y
            )
            rel_val = val_size / (val_size + test_size)
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state, stratify=y_temp
            )
            X_train = X.iloc[idx_train]
            X_val = X.iloc[idx_val]
            X_test = X.iloc[idx_test]
        else:
            idx_train, idx_temp, y_train, y_temp = train_test_split(
                indices, y, test_size=(val_size + test_size),
                random_state=split_config.random_state
            )
            rel_val = val_size / (val_size + test_size)
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state
            )
            X_train = X.iloc[idx_train]
            X_val = X.iloc[idx_val]
            X_test = X.iloc[idx_test]
        
        feature_names = list(data_config.feature_cols)
        set_splits(X_train, X_val, X_test, to_numpy_1d(y_train), to_numpy_1d(y_val), to_numpy_1d(y_test), feature_names)
        elapsed = time.perf_counter() - t0
        st.session_state.setdefault("last_timings", {})["Prepare Splits"] = round(elapsed, 2)

        # Store indices for explainability (original df positions)
        if use_group_split and entity_id_final:
            st.session_state.train_indices = original_indices[train_idx].tolist()
            st.session_state.test_indices = original_indices[temp_idx[test_idx]].tolist()
        elif split_config.use_time_split and data_config.datetime_col:
            st.session_state.train_indices = train_indices.tolist()
            st.session_state.test_indices = test_indices.tolist()
        else:
            st.session_state.train_indices = original_indices[idx_train].tolist()
            st.session_state.test_indices = original_indices[idx_test].tolist()
        
        st.success(f"Splits prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except Exception as e:
        st.error(f"Error preparing splits: {e}")
        logger.exception(e)

# Check if splits are ready
splits = st.session_state.get('X_train')
if splits is None:
    st.stop()

X_train = st.session_state.X_train
X_val = st.session_state.X_val
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_val = st.session_state.y_val
y_test = st.session_state.y_test

# Small sample size warning
n_train = len(X_train)
if n_train < 50:
    st.warning(f"""
    **Small sample size detected:** Training set has only {n_train} samples.
    
    Some models may have limitations:
    - **KNN:** n_neighbors will be limited to {n_train - 1} max
    - **Cross-validation:** May not be reliable with very small folds
    - **Neural Networks:** May overfit easily
    
    Consider simpler models (Linear/Logistic Regression, Decision Trees) for small datasets.
    """)
elif n_train < 100:
    st.info(f"Training set has {n_train} samples. Some complex models may have limited performance.")

# Model Selection Coach (top section) - cached for performance
@st.cache_data
def _compute_coach_recommendations(_df_hash, target_col, task_type, cohort_type, entity_id, eda_results_keys):
    """Cached coach recommendations computation."""
    from ml.model_coach import coach_recommendations
    from ml.eda_recommender import compute_dataset_signals
    
    df = get_data()  # Get actual dataframe
    signals = compute_dataset_signals(
        df,
        target_col,
        task_type,
        cohort_type,
        entity_id,
        outlier_method=st.session_state.get("eda_outlier_method", "iqr")
    )
    eda_results = st.session_state.get('eda_results')
    return coach_recommendations(signals, eda_results, get_insights_by_category())

# Key insights after pre-processing (EDA + preprocessing-specific)
insights = get_insights_by_category()
eda_only = [i for i in insights if i.get('category') != 'preprocessing']
prep_only = [i for i in insights if i.get('category') == 'preprocessing']
if eda_only or prep_only:
    with st.expander("Key insights after pre-processing", expanded=True):
        if eda_only:
            st.markdown("**From EDA**")
            for insight in eda_only:
                st.markdown(f"• **{insight.get('category', 'General').title()}:** {insight['finding']}")
                st.caption(f"  → {insight['implication']}")
        if prep_only:
            st.markdown("**From preprocessing**")
            for insight in prep_only:
                st.markdown(f"• {insight['finding']}")
                st.caption(f"  → {insight['implication']}")

# Model selection and configuration
st.header("Model Configuration")
_prep_pipes = st.session_state.get("preprocessing_pipelines_by_model") or {}
_prep_models = [k for k in _prep_pipes.keys() if k != "default"]
if _prep_models:
    st.caption("Models with preprocessing pipelines are pre-selected. Adjust as needed.")

# Get registry and filter by task type (cached)
@st.cache_resource
def _get_registry_cached():
    """Cached registry access."""
    from ml.model_registry import get_registry
    return get_registry()

registry = _get_registry_cached()
available_models = {
    k: v for k, v in registry.items()
    if (task_type_final == 'regression' and v.capabilities.supports_regression) or
       (task_type_final == 'classification' and v.capabilities.supports_classification)
}

# Sync Train & Compare selections from Preprocessing (before any checkbox)
# Only set if not already set to avoid widget conflicts
_prep_built_sync = st.session_state.get("preprocess_built_model_keys", [])
for _k in _prep_built_sync:
    if _k not in available_models:
        continue
    _key = f"train_model_{_k}"
    # Only set if key doesn't exist to avoid widget default value conflicts
    if _key not in st.session_state:
        st.session_state[_key] = True

# Initialize model_config and tracking variables
model_config = st.session_state.get('model_config', ModelConfig())
models_to_train = []
selected_model_params = st.session_state.get('selected_model_params', {})

# Check if data has been preprocessed
_prep_built = st.session_state.get("preprocess_built_model_keys", [])
_has_preprocessing = len(_prep_built) > 0 or pipeline is not None

# Warning banner for unprocessed data
if not _has_preprocessing:
    st.warning("⚠️ **Warning:** You are about to train models with unprocessed data. It is recommended to preprocess your data first in the Preprocess page to ensure optimal model performance.")

# Group models by family, ordered by explainability (high -> medium -> low)
# Define explainability order: Linear (high) > Probabilistic (high) > Trees (medium) > Distance (medium) > Boosting (low) > Margin (low) > Neural Net (low)
EXPLAINABILITY_ORDER = {
    'Linear': 1,
    'Probabilistic': 2,
    'Trees': 3,
    'Distance': 4,
    'Boosting': 5,
    'Margin': 6,
    'Neural Net': 7
}

# Family-based model selection grouped by explainability
# Group models by family
model_groups = {}
for key, spec in available_models.items():
    group = spec.group
    if group not in model_groups:
        model_groups[group] = []
    model_groups[group].append((key, spec))

# Sort groups by explainability order (most explainable first)
sorted_groups = sorted(model_groups.keys(), key=lambda g: EXPLAINABILITY_ORDER.get(g, 999))

# Display models by group, ordered by explainability — card layout
_GROUP_ICONS_TC = {
    "Linear": "📏", "Trees": "🌳", "Distance": "📍", "Boosting": "🚀",
    "Margin": "🔲", "Probabilistic": "🎲", "Neural Net": "🧠",
}
_EXPLAIN_LABELS = {
    "Linear": "High explainability", "Probabilistic": "High explainability",
    "Trees": "Medium explainability", "Distance": "Medium explainability",
    "Boosting": "Low-medium explainability", "Margin": "Low explainability",
    "Neural Net": "Low explainability",
}

for group_name in sorted_groups:
    icon = _GROUP_ICONS_TC.get(group_name, "📦")
    explain = _EXPLAIN_LABELS.get(group_name, "")
    st.markdown(f"#### {icon} {group_name}" + (f" <span style='font-size:0.8rem; color:#94a3b8; margin-left:0.5rem;'>({explain})</span>" if explain else ""), unsafe_allow_html=True)
    group_models = model_groups[group_name]
    cols = st.columns(min(len(group_models), 3))

    for idx, (model_key, spec) in enumerate(group_models):
        checkbox_key = f"train_model_{model_key}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = False

        with cols[idx % len(cols)]:
            is_selected = st.session_state[checkbox_key]
            model_has_preprocessing = model_key in _prep_built
            border = "#667eea" if is_selected else "#e2e8f0"
            bg = "#f0f0ff" if is_selected else "#fff"
            prep_badge = "✅ Preprocessed" if model_has_preprocessing else "⚠️ No pipeline"
            prep_color = "#22c55e" if model_has_preprocessing else "#f59e0b"
            notes = "; ".join(spec.capabilities.notes) if spec.capabilities.notes else ""
            st.markdown(f"""
            <div style="border: 2px solid {border}; border-radius: 10px; padding: 0.75rem;
                        background: {bg}; margin-bottom: 0.4rem; min-height: 80px;">
                <strong>{spec.name}</strong>
                <span style="float:right; font-size:0.7rem; color:{prep_color};">{prep_badge if _has_preprocessing else ""}</span>
                {"<br/><span style='font-size:0.78rem; color:#64748b;'>" + notes + "</span>" if notes else ""}
            </div>
            """, unsafe_allow_html=True)
            is_selected = st.checkbox("Select", value=is_selected, key=checkbox_key, label_visibility="collapsed")

        if is_selected:
            models_to_train.append(model_key)
            
            # Hyperparameter controls
            if spec.hyperparam_schema:
                with st.expander(f"{spec.name} Hyperparameters"):
                    params = {}
                    automl_best = st.session_state.get("nn_automl_best_params", {}) if model_key == "nn" else {}
                    
                    # Get training sample size for KNN validation
                    n_train_samples = len(X_train) if X_train is not None else 1000
                    
                    for param_name, param_def in spec.hyperparam_schema.items():
                        param_key = f"{model_key}_{param_name}"
                        default_val = automl_best.get(param_name, param_def['default'])
                        if param_def['type'] == 'int':
                            # Dynamic max for n_neighbors based on training sample size
                            max_val = param_def['max']
                            if param_name == 'n_neighbors' and 'knn' in model_key:
                                max_val = min(param_def['max'], n_train_samples - 1)
                                if max_val < param_def['max']:
                                    st.caption(f"Max limited to {max_val} (training set has {n_train_samples} samples)")
                                # Also adjust default if needed
                                if default_val > max_val:
                                    default_val = max(1, min(5, max_val))
                            
                            params[param_name] = st.number_input(
                                param_def.get('help', param_name),
                                min_value=param_def['min'],
                                max_value=max(1, max_val),  # Ensure max is at least 1
                                value=min(default_val, max(1, max_val)),  # Ensure value doesn't exceed max
                                key=param_key
                            )
                        elif param_def['type'] == 'float':
                            format_str = "%.4f" if param_def.get('log', False) else "%.2f"
                            params[param_name] = st.number_input(
                                param_def.get('help', param_name),
                                min_value=param_def['min'],
                                max_value=param_def['max'],
                                value=default_val,
                                format=format_str,
                                key=param_key
                            )
                        elif param_def['type'] == 'select':
                            options = param_def['options']
                            default_val = automl_best.get(param_name, param_def['default'])
                            default_idx = options.index(default_val) if default_val in options else options.index(param_def['default'])
                            params[param_name] = st.selectbox(
                                param_def.get('help', param_name),
                                options=options,
                                index=default_idx,
                                key=param_key
                            )
                        elif param_def['type'] == 'int_or_none':
                            # Special handling for max_depth=None
                            use_none = st.checkbox(f"{param_name} = None (unlimited)", value=param_def['default'] is None, key=f"{param_key}_none")
                            if use_none:
                                params[param_name] = None
                            else:
                                params[param_name] = st.number_input(
                                    param_def.get('help', param_name),
                                    min_value=param_def['min'],
                                    max_value=param_def['max'],
                                    value=param_def['min'] if param_def['default'] is None else param_def['default'],
                                    key=param_key
                                )
                    
                    selected_model_params[model_key] = params

st.session_state.model_config = model_config

# Pre-training coach tips
coach_output = st.session_state.get('coach_output')
with st.expander("Pre-training Coach Tips", expanded=False):
    if coach_output and hasattr(coach_output, 'preprocessing_recommendations') and coach_output.preprocessing_recommendations:
        st.markdown("**Preprocessing checklist (from Coach):**")
        for prep in coach_output.preprocessing_recommendations[:5]:
            st.markdown(f"- **{prep.step_name}** ({prep.priority}): {prep.rationale}")
        st.caption("Configure these in the Preprocessing page before building the pipeline.")
    else:
        st.info("Run EDA and check the Model Selection Coach for preprocessing recommendations.")
    st.markdown("**Tip:** Ensure your preprocessing pipeline matches your selected models. Linear models and neural nets require scaling; tree models do not.")

# Check Optuna availability
_has_optuna = False
try:
    import optuna
    _has_optuna = True
except Exception:
    pass

# Generic Optuna optimization function
def optimize_model_hyperparameters(model_name, spec, X_train_transformed, y_train, X_val_transformed, y_val, task_type, random_seed, n_trials=30):
    """
    Generic function to optimize hyperparameters for any model using Optuna.
    
    Returns:
        dict: Best hyperparameters found
    """
    if not _has_optuna:
        return None
    
    if not spec.hyperparam_schema:
        return None  # No hyperparameters to optimize
    
    NNWeightedHuberWrapper, GLMWrapper, HuberGLMWrapper, RFWrapper, RegistryModelWrapper = _get_model_wrappers()
    
    def _objective(trial):
        # Suggest hyperparameters based on schema
        params = {}
        n_train_samples = len(X_train)
        
        for param_name, param_def in spec.hyperparam_schema.items():
            if param_def['type'] == 'int':
                max_val = param_def['max']
                # Dynamic max for n_neighbors based on training sample size
                if param_name == 'n_neighbors' and 'knn' in model_name:
                    max_val = min(max_val, n_train_samples - 1)
                params[param_name] = trial.suggest_int(param_name, param_def['min'], max(1, max_val))
            elif param_def['type'] == 'float':
                log_scale = param_def.get('log', False)
                min_val = param_def['min']
                max_val = param_def['max']
                # Optuna requires min > 0 for log=True
                if log_scale and min_val <= 0:
                    # Use a small positive value instead
                    min_val = max(1e-5, min_val)
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=log_scale)
            elif param_def['type'] == 'select':
                selected_value = trial.suggest_categorical(param_name, param_def['options'])
                # Special handling for SVR/SVC gamma parameter: convert numeric strings to floats
                if param_name == 'gamma' and selected_value not in ['scale', 'auto']:
                    try:
                        params[param_name] = float(selected_value)
                    except (ValueError, TypeError):
                        params[param_name] = selected_value
                else:
                    params[param_name] = selected_value
            elif param_def['type'] == 'int_or_none':
                # For int_or_none, suggest int with option for None
                use_none = trial.suggest_categorical(f"{param_name}_use_none", [False, True])
                if use_none:
                    params[param_name] = None
                else:
                    params[param_name] = trial.suggest_int(param_name, param_def['min'], param_def['max'])
        
        # Special handling for neural network
        if model_name == 'nn':
            num_layers = params.get('num_layers', 2)
            layer_width = params.get('layer_width', 32)
            pattern = params.get('architecture_pattern', 'constant')
            
            if pattern == 'constant':
                hidden_layers = [layer_width] * num_layers
            elif pattern == 'pyramid':
                hidden_layers = [layer_width * (2 ** i) for i in range(num_layers)]
            elif pattern == 'funnel':
                max_width = layer_width * (2 ** (num_layers - 1))
                hidden_layers = [max_width // (2 ** i) for i in range(num_layers)]
            else:
                hidden_layers = [layer_width] * num_layers
            
            model = NNWeightedHuberWrapper(
                hidden_layers=hidden_layers,
                dropout=params.get('dropout', 0.1),
                task_type=task_type,
                activation=params.get('activation', 'relu')
            )
            res = model.fit(
                X_train_transformed, y_train, X_val_transformed, y_val,
                epochs=params.get('epochs', 200),
                batch_size=params.get('batch_size', 256),
                lr=params.get('lr', 0.0015),
                weight_decay=params.get('weight_decay', 0.0002),
                patience=params.get('patience', 30),
                random_seed=random_seed
            )
            hist = res.get("history", {}) if isinstance(res, dict) else {}
            if task_type == "regression":
                vlm = hist.get("val_rmse", [])
                return vlm[-1] if vlm else float("inf")
            else:
                vlm = hist.get("val_accuracy", [])
                return 1.0 - (vlm[-1] if vlm else 1.0)  # Convert to minimize
        else:
            # For sklearn-based models
            estimator = spec.factory(task_type, random_seed)
            for param_name, param_value in params.items():
                if hasattr(estimator, param_name):
                    # Special handling for SVR/SVC gamma parameter: convert numeric strings to floats
                    if param_name == 'gamma' and isinstance(param_value, str) and param_value not in ['scale', 'auto']:
                        try:
                            param_value = float(param_value)
                        except (ValueError, TypeError):
                            pass  # Keep original value if conversion fails
                    setattr(estimator, param_name, param_value)
            
            # Fit and evaluate
            estimator.fit(X_train_transformed, y_train)
            
            if task_type == "regression":
                from sklearn.metrics import mean_squared_error
                y_pred = estimator.predict(X_val_transformed)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                from sklearn.metrics import accuracy_score
                y_pred = estimator.predict(X_val_transformed)
                return 1.0 - accuracy_score(y_val, y_pred)  # Convert to minimize
    
    direction = "minimize"  # Always minimize (we convert accuracy to error)
    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params

def _train_models(models_to_train, selected_model_params, use_optimization=False):
    """Train models with optional hyperparameter optimization."""
    # Lazy import model wrappers and evaluation functions only when training
    NNWeightedHuberWrapper, GLMWrapper, HuberGLMWrapper, RFWrapper, RegistryModelWrapper = _get_model_wrappers()
    calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals = _get_eval_functions()
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    progress_container = st.container()
    random_seed = st.session_state.get('random_seed', 42)
    
    # Training time warning with cancel option
    slow_models = {'nn', 'extratrees', 'svc', 'svr'}
    has_slow = any(m in slow_models for m in models_to_train)
    
    # Initialize cancel flag in session state
    if 'cancel_training' not in st.session_state:
        st.session_state.cancel_training = False
    
    if has_slow or use_optimization:
        col_warn, col_cancel = st.columns([4, 1])
        with col_warn:
            st.warning("""
            ⏱️ **Training in progress.** Some models (Neural Networks, ExtraTrees, SVM) or hyperparameter 
            optimization may take several minutes. Training progress shown below.
            """)
        with col_cancel:
            if st.button("🛑 Cancel Training", type="secondary", key="cancel_training_btn"):
                st.session_state.cancel_training = True
                st.warning("Training canceled. Trained models saved. Refresh page to train again.")
                st.stop()
    
    for model_name in models_to_train:
        with progress_container:
            if use_optimization:
                st.subheader(f"Optimizing and Training {model_name.upper()}")
                if model_name in slow_models:
                    st.info("⏱️ This model typically takes 2-5 minutes with optimization. Please be patient...")
                else:
                    st.info("⏱️ Hyperparameter optimization in progress...")
            else:
                st.subheader(f"Training {model_name.upper()}")
                if model_name in slow_models:
                    st.info("⏱️ This model may take 30-90 seconds to train...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Get model spec from registry
                spec = registry.get(model_name)
                model_pipeline = get_preprocessing_pipeline(model_name) or pipeline
                if model_pipeline is None:
                    st.error("Preprocessing pipeline not found for this model.")
                    continue

                # Fit preprocessing on training data only
                model_pipeline.fit(X_train)
                X_train_model = model_pipeline.transform(X_train)
                X_val_model = model_pipeline.transform(X_val)
                X_test_model = model_pipeline.transform(X_test)
                if hasattr(X_train_model, 'toarray'):
                    X_train_model = X_train_model.toarray()
                    X_val_model = X_val_model.toarray()
                    X_test_model = X_test_model.toarray()
                
                # Optimize hyperparameters if requested
                if use_optimization and spec and spec.hyperparam_schema:
                    status_text.text("Running Optuna hyperparameter optimization...")
                    best_params = optimize_model_hyperparameters(
                        model_name, spec, X_train_model, y_train, X_val_model, y_val,
                        task_type_final, random_seed, n_trials=30
                    )
                    if best_params:
                        # Update selected_model_params with optimized values
                        selected_model_params[model_name] = best_params
                        st.success(f"Optimization complete! Best parameters found for {model_name.upper()}")
                    else:
                        st.warning(f"Could not optimize {model_name}. Using default parameters.")
                
                # Handle existing wrappers (nn, rf, glm, huber) with special logic
                if model_name == 'nn':
                    params = selected_model_params.get(model_name, {})
                    
                    # Compute hidden_layers from architecture parameters
                    num_layers = params.get('num_layers', 2)
                    layer_width = params.get('layer_width', 32)
                    pattern = params.get('architecture_pattern', 'constant')
                    
                    if pattern == 'constant':
                        hidden_layers = [layer_width] * num_layers
                    elif pattern == 'pyramid':
                        # Increasing width: 32 -> 64 -> 128
                        hidden_layers = [layer_width * (2 ** i) for i in range(num_layers)]
                    elif pattern == 'funnel':
                        # Decreasing width: 128 -> 64 -> 32
                        max_width = layer_width * (2 ** (num_layers - 1))
                        hidden_layers = [max_width // (2 ** i) for i in range(num_layers)]
                    else:
                        hidden_layers = [layer_width] * num_layers
                    
                    status_text.text(f"Architecture: {hidden_layers} ({pattern})")
                    
                    model = NNWeightedHuberWrapper(
                        hidden_layers=hidden_layers,
                        dropout=params.get('dropout', model_config.nn_dropout),
                        task_type=task_type_final,
                        activation=params.get('activation', 'relu')
                    )
                    def progress_cb(epoch, train_loss, val_loss, val_metric):
                        epochs = params.get('epochs', model_config.nn_epochs)
                        progress = epoch / epochs
                        progress_bar.progress(progress)
                        if task_type_final == 'regression':
                            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_metric:.4f}")
                        else:
                            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Accuracy: {val_metric:.4f}")
                    
                    results = model.fit(
                        X_train_model, y_train, X_val_model, y_val,
                        epochs=params.get('epochs', model_config.nn_epochs),
                        batch_size=params.get('batch_size', model_config.nn_batch_size),
                        lr=params.get('lr', model_config.nn_lr),
                        weight_decay=params.get('weight_decay', model_config.nn_weight_decay),
                        patience=params.get('patience', model_config.nn_patience),
                        progress_callback=progress_cb,
                        random_seed=st.session_state.get('random_seed', 42)
                    )
                    
                    # Store architecture info in results for reporting
                    results['architecture'] = model.get_architecture_summary()
                
                elif model_name == 'rf':
                    params = selected_model_params.get(model_name, {})
                    model = RFWrapper(
                        n_estimators=params.get('n_estimators', model_config.rf_n_estimators),
                        max_depth=params.get('max_depth', model_config.rf_max_depth),
                        min_samples_leaf=params.get('min_samples_leaf', model_config.rf_min_samples_leaf),
                        task_type=task_type_final
                    )
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                elif model_name == 'glm':
                    model = GLMWrapper(task_type=task_type_final)
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                elif model_name == 'huber':
                    params = selected_model_params.get(model_name, {})
                    model = HuberGLMWrapper(
                        epsilon=params.get('epsilon', model_config.huber_epsilon),
                        alpha=params.get('alpha', model_config.huber_alpha)
                    )
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                else:
                    # New registry models: create estimator and wrap
                    if spec is None:
                        st.error(f"Model spec not found for {model_name}")
                        continue
                    
                    params = selected_model_params.get(model_name, spec.default_params.copy())
                    random_seed = st.session_state.get('random_seed', 42)
                    
                    # Create estimator from factory
                    estimator = spec.factory(task_type_final, random_seed)
                    
                    # Special handling for KNN: ensure n_neighbors <= training samples
                    if 'knn' in model_name and 'n_neighbors' in params:
                        n_train_samples = len(X_train_model)
                        original_n_neighbors = params['n_neighbors']
                        if original_n_neighbors > n_train_samples:
                            adjusted_n_neighbors = max(1, n_train_samples - 1)
                            params['n_neighbors'] = adjusted_n_neighbors
                            st.warning(f"""
                            **KNN adjusted:** n_neighbors reduced from {original_n_neighbors} to {adjusted_n_neighbors} 
                            because training set only has {n_train_samples} samples.
                            """)
                    
                    # Set hyperparameters
                    for param_name, param_value in params.items():
                        if hasattr(estimator, param_name):
                            # Special handling for SVR/SVC gamma parameter: convert numeric strings to floats
                            if param_name == 'gamma' and isinstance(param_value, str) and param_value not in ['scale', 'auto']:
                                try:
                                    param_value = float(param_value)
                                except (ValueError, TypeError):
                                    pass  # Keep original value if conversion fails
                            setattr(estimator, param_name, param_value)
                    
                    # Wrap in generic wrapper
                    model = RegistryModelWrapper(estimator, spec.name)
                    
                    # Fit model
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                # Evaluate on test set
                y_test_pred = model.predict(X_test_model)
                
                if task_type_final == 'regression':
                    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
                else:
                    y_test_proba = model.predict_proba(X_test_model) if model.supports_proba() else None
                    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
                
                # Cross-validation if enabled (skip for NN - PyTorch models don't implement sklearn interface)
                cv_results = None
                if use_cv and model_name != 'nn':
                    try:
                        cv_results = perform_cross_validation(
                            model.get_model(), X_train_model, y_train,
                            cv_folds=cv_folds, task_type=data_config.task_type
                        )
                    except Exception as cv_error:
                        st.warning(f"Cross-validation failed for {model_name}: {cv_error}. Skipping CV.")
                        logger.warning(f"CV failed for {model_name}: {cv_error}")
                elif use_cv and model_name == 'nn':
                    st.info("Cross-validation skipped for Neural Network (PyTorch models use their own validation loop during training)")
                
                # Store results
                model_results = {
                    'metrics': test_metrics,
                    'history': results.get('history', {}),
                    'y_test_pred': y_test_pred,
                    'y_test': y_test,
                    'y_test_proba': y_test_proba if data_config.task_type == 'classification' else None,
                    'cv_results': cv_results
                }
                
                add_trained_model(model_name, model, model_results)

                # Store fitted estimator for explainability
                # For explainability, we need a pipeline that can handle raw data
                # Store both the fitted model and the preprocessing pipeline
                if model_name == 'nn':
                    # NN needs special handling - store sklearn-compatible wrapper
                    fitted_estimator = model.get_sklearn_estimator()
                    if not (hasattr(fitted_estimator, 'is_fitted_') and fitted_estimator.is_fitted_):
                        fitted_estimator.fit(X_train_model[:1], y_train[:1])
                    st.session_state.fitted_estimators[model_name] = fitted_estimator
                else:
                    # For sklearn models, store the fitted model
                    sklearn_model = model.get_model()
                    st.session_state.fitted_estimators[model_name] = sklearn_model
                
                # Store preprocessing pipeline for all models (needed for explainability)
                st.session_state.fitted_preprocessing_pipelines[model_name] = model_pipeline
                from ml.pipeline import get_feature_names_after_transform
                st.session_state.feature_names_by_model[model_name] = get_feature_names_after_transform(
                    model_pipeline, data_config.feature_cols
                )
                
                progress_bar.progress(1.0)
                st.success(f"{model_name.upper()} training complete!")
                
            except Exception as e:
                with st.expander(f"Error training {model_name.upper()}", expanded=True):
                    st.error(f"Training failed: {str(e)}")
                    st.code(str(e), language='python')
                    logger.exception(e)
    
    # Log methodology action after all models are trained
    trained_models = st.session_state.get('trained_models', {})
    model_results = st.session_state.get('model_results', {})
    if trained_models:
        # Get best model and metrics
        best_model_name = None
        best_metric_value = None
        task_type_final_local = st.session_state.get('task_type_detection', TaskTypeDetection()).final or data_config.task_type
        
        for name, results in model_results.items():
            metrics = results.get('metrics', {})
            if task_type_final_local == 'regression':
                metric_val = metrics.get('RMSE', float('inf'))
                if best_metric_value is None or metric_val < best_metric_value:
                    best_metric_value = metric_val
                    best_model_name = name
            else:
                metric_val = metrics.get('Accuracy', 0)
                if best_metric_value is None or metric_val > best_metric_value:
                    best_metric_value = metric_val
                    best_model_name = name
        
        log_methodology(
            step='Model Training',
            action=f"Trained {len(trained_models)} models with validation",
            details={
                'models': list(trained_models.keys()),
                'best_model': best_model_name,
                'best_metric_value': best_metric_value,
                'use_cv': st.session_state.get('use_cv', False),
                'cv_folds': st.session_state.get('cv_folds', 5) if st.session_state.get('use_cv', False) else None,
                'hyperparameter_optimization': use_optimization
            }
        )

# Training section with two buttons
st.markdown("---")
st.header("Train Models")

if models_to_train:
    col1, col2 = st.columns(2)
    
    with col1:
        train_standard = st.button("Train Models", type="primary", key="train_models_button", width="stretch")
    
    with col2:
        train_optimized = st.button(
            "Train Models with Hyperparameter Optimization", 
            type="secondary", 
            key="train_models_optimized_button",
            width="stretch",
            help="⚠️ This will take significantly longer as it searches for optimal hyperparameters using Optuna"
        )
    
    if train_standard:
        _train_models(models_to_train, selected_model_params, use_optimization=False)
    elif train_optimized:
        if not _has_optuna:
            st.error("Optuna is not installed. Please install it with `pip install optuna` to use hyperparameter optimization.")
        else:
            _train_models(models_to_train, selected_model_params, use_optimization=True)
elif not models_to_train:
    st.info("Please select at least one model to train.")

# Results comparison
if st.session_state.get('trained_models'):
    # Lazy import plotly and visualization functions for results display
    go, px = _get_plotly()
    plot_training_history, plot_predictions_vs_actual, plot_residuals = _get_visualization_functions()
    calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals = _get_eval_functions()
    
    st.header("Results Comparison")
    
    # ================================================================
    # TRAINING CONFIGURATION SUMMARY
    # ================================================================
    st.markdown("### What You're Training On")
    
    # Get dataset statistics
    X_train_data = st.session_state.get("X_train")
    X_val_data = st.session_state.get("X_val")
    X_test_data = st.session_state.get("X_test")
    
    n_train = len(X_train_data) if X_train_data is not None else 0
    n_val = len(X_val_data) if X_val_data is not None else 0
    n_test = len(X_test_data) if X_test_data is not None else 0
    n_total = n_train + n_val + n_test
    
    # Get feature information
    selected_features = st.session_state.get('selected_features')
    feature_cols = data_config.feature_cols if data_config else []
    n_features_used = len(selected_features) if selected_features else len(feature_cols)
    n_original_features = len(feature_cols)
    n_engineered = len(st.session_state.get('engineered_feature_names', []))
    
    # Get target information
    target_name = data_config.target_col if data_config else "Unknown"
    task_type_display = task_type_final if task_type_final else (data_config.task_type if data_config else "Unknown")
    
    st.markdown(f"""
    **Dataset:**
    - Total samples: {n_total:,} ({n_train:,} train, {n_val:,} val, {n_test:,} test)
    - Original features: {n_original_features}
    - Engineered features: {n_engineered}
    - **Selected features: {n_features_used}** ← Training on these
    
    **Target:** {target_name} ({task_type_display})
    
    **Preprocessing:** Per-model pipelines (see Preprocessing page)
    """)
    
    if n_engineered > 0:
        engineered_names = st.session_state.get('engineered_feature_names', [])
        st.info(f"🧬 **{n_engineered} engineered features** included: {', '.join(engineered_names[:5])}{'...' if len(engineered_names) > 5 else ''}")
    
    st.markdown("---")
    
    # How to read results explainer
    with st.expander("How to Read These Results", expanded=False):
        if task_type_final == 'regression':
            st.markdown("""
            **Metrics:**
            - **RMSE (Root Mean Squared Error):** Average prediction error in target units. Lower is better.
            - **MAE (Mean Absolute Error):** Average absolute error. Less sensitive to outliers than RMSE.
            - **R² (R-squared):** Proportion of variance explained. 1.0 = perfect, 0 = no better than mean.
            - **MedianAE:** Median absolute error. Robust to outliers.
            
            **Cross-Validation vs Holdout:**
            - **Holdout:** Single train/test split. Fast but may be noisy.
            - **Cross-Validation:** Multiple splits. More stable estimate but slower.
            """)
        else:
            st.markdown("""
            **Metrics:**
            - **Accuracy:** Proportion of correct predictions. Can be misleading with class imbalance.
            - **F1 Score:** Harmonic mean of precision and recall. Better for imbalanced data.
            - **ROC-AUC:** Area under ROC curve. Measures separability of classes.
            - **PR-AUC:** Precision-Recall AUC. Better for imbalanced data than ROC-AUC.
            - **Log Loss:** Penalizes confident wrong predictions. Lower is better.
            
            **Calibration:**
            - Well-calibrated models: predicted probabilities match actual frequencies
            - Important for medical decision-making
            - Check calibration plots if available
            """)
    
    # Metrics table with native copy support
    from utils.table_export import table
    
    comparison_data = []
    for name, results in st.session_state.model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if data_config.task_type == 'regression':
        comparison_df = comparison_df.sort_values('RMSE')
    else:
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Model metrics table with export option
    table(comparison_df, key="model_metrics")

    # ================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ================================================================
    with st.expander("📊 Metrics with 95% Bootstrap Confidence Intervals", expanded=True):
        st.markdown("""
        **Why this matters:** Point estimates (e.g., "RMSE = 0.82") aren't sufficient for publication.
        Confidence intervals show the uncertainty in your estimates. Reviewers expect these.
        """)

        if st.button("Compute Bootstrap CIs (1000 resamples)", key="compute_bootstrap_cis"):
            from ml.bootstrap import (
                bootstrap_all_regression_metrics, bootstrap_all_classification_metrics,
                format_metric_with_ci,
            )
            bootstrap_results = {}
            progress = st.progress(0)
            model_names_list = list(st.session_state.model_results.keys())
            for i, name in enumerate(model_names_list):
                results = st.session_state.model_results[name]
                y_test_local = np.array(results["y_test"])
                y_pred_local = np.array(results["y_test_pred"])

                if data_config.task_type == "regression":
                    cis = bootstrap_all_regression_metrics(y_test_local, y_pred_local, n_resamples=1000)
                else:
                    y_proba_local = None
                    model_obj = st.session_state.trained_models.get(name)
                    if model_obj and hasattr(model_obj, 'supports_proba') and model_obj.supports_proba():
                        pipeline_local = st.session_state.get("fitted_preprocessing_pipelines", {}).get(name)
                        if pipeline_local is not None:
                            try:
                                X_test_local = pipeline_local.transform(st.session_state.get("X_test"))
                                y_proba_local = model_obj.predict_proba(X_test_local)
                                if y_proba_local is not None and y_proba_local.ndim == 2 and y_proba_local.shape[1] == 2:
                                    y_proba_local = y_proba_local[:, 1]
                            except Exception:
                                pass
                    cis = bootstrap_all_classification_metrics(y_test_local, y_pred_local, y_proba=y_proba_local, n_resamples=1000)

                bootstrap_results[name] = cis
                progress.progress((i + 1) / len(model_names_list))

            st.session_state["bootstrap_results"] = bootstrap_results

        if st.session_state.get("bootstrap_results"):
            from ml.bootstrap import format_metric_with_ci
            ci_rows = []
            for name, cis in st.session_state["bootstrap_results"].items():
                row = {"Model": name.upper()}
                for metric_name, result in cis.items():
                    row[metric_name] = format_metric_with_ci(result, decimal_places=4)
                ci_rows.append(row)
            ci_df = pd.DataFrame(ci_rows)
            table(ci_df, key="bootstrap_ci")
            st.caption("Format: estimate [95% CI lower, upper] via BCa bootstrap (1000 resamples)")

    # ================================================================
    # BASELINE MODEL COMPARISON
    # ================================================================
    with st.expander("📏 Baseline Model Comparison", expanded=False):
        st.markdown("""
        **Why this matters:** Reviewers need to see that your model outperforms trivial baselines.
        Without this comparison, they can't tell if your model actually adds value.
        """)

        if st.button("Train Baseline Models", key="train_baselines"):
            from ml.baseline_models import train_baseline_models
            X_train_base = st.session_state.get("X_train")
            y_train_base = st.session_state.get("y_train")
            X_test_base = st.session_state.get("X_test")
            y_test_base = st.session_state.get("y_test")

            if X_train_base is not None and y_test_base is not None:
                # Use the first model's preprocessing pipeline for baselines
                first_model = list(st.session_state.get("fitted_preprocessing_pipelines", {}).keys())
                if first_model:
                    pipe = st.session_state["fitted_preprocessing_pipelines"][first_model[0]]
                    try:
                        X_train_t = pipe.transform(X_train_base)
                        X_test_t = pipe.transform(X_test_base)
                    except Exception:
                        X_train_t = np.array(X_train_base)
                        X_test_t = np.array(X_test_base)
                else:
                    X_train_t = np.array(X_train_base)
                    X_test_t = np.array(X_test_base)

                baselines = train_baseline_models(
                    X_train_t, np.array(y_train_base),
                    X_test_t, np.array(y_test_base),
                    task_type=data_config.task_type or "regression",
                )
                st.session_state["baseline_results"] = baselines

        if st.session_state.get("baseline_results"):
            baselines = st.session_state["baseline_results"]
            for bname, bres in baselines.items():
                st.markdown(f"**{bname}:** {bres['description']}")
                cols_b = st.columns(len(bres["metrics"]))
                for i, (mname, mval) in enumerate(bres["metrics"].items()):
                    ci = bres.get("bootstrap_cis", {}).get(mname)
                    with cols_b[i]:
                        if ci:
                            st.metric(mname, f"{mval:.4f}", help=f"95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
                        else:
                            st.metric(mname, f"{mval:.4f}")

    # ================================================================
    # CALIBRATION ANALYSIS
    # ================================================================
    with st.expander("📐 Calibration Analysis", expanded=False):
        st.markdown("""
        **Why this matters:** A model that says "70% chance of event" should be right about 70% of the time.
        Poor calibration means predicted probabilities are unreliable — critical for clinical decisions.
        """)

        if data_config.task_type == "classification":
            for name, results in st.session_state.model_results.items():
                model_obj = st.session_state.trained_models.get(name)
                if model_obj and hasattr(model_obj, 'supports_proba') and model_obj.supports_proba():
                    pipeline_local = st.session_state.get("fitted_preprocessing_pipelines", {}).get(name)
                    if pipeline_local is not None:
                        try:
                            from ml.calibration import calibration_classification, plot_calibration_curve
                            X_test_local = pipeline_local.transform(st.session_state.get("X_test"))
                            y_proba_local = model_obj.predict_proba(X_test_local)
                            if y_proba_local is not None and y_proba_local.ndim == 2:
                                y_proba_pos = y_proba_local[:, 1] if y_proba_local.shape[1] == 2 else y_proba_local[:, -1]
                                cal = calibration_classification(np.array(results["y_test"]), y_proba_pos, model_name=name.upper())
                                fig_cal = plot_calibration_curve(cal)
                                st.plotly_chart(fig_cal, use_container_width=True, key=f"cal_{name}")
                                st.caption(f"Brier Score: {cal.brier_score:.4f} | ECE: {cal.ece:.4f} | MCE: {cal.mce:.4f}")
                        except Exception as e:
                            st.caption(f"Could not compute calibration for {name}: {e}")
        else:
            for name, results in st.session_state.model_results.items():
                from ml.calibration import calibration_regression
                cal = calibration_regression(
                    np.array(results["y_test"]), np.array(results["y_test_pred"]),
                    model_name=name.upper(),
                )
                st.markdown(f"**{name.upper()}:** Calibration slope = {cal.calibration_slope:.3f}, "
                           f"Intercept = {cal.calibration_intercept:.3f} "
                           f"(perfect: slope=1, intercept=0)")
            st.caption("Calibration slope measures systematic over/under-prediction. "
                      "Slope < 1 = predictions too extreme; slope > 1 = predictions too conservative.")
    
    # CV results if available
    if use_cv:
        st.subheader("Cross-Validation Results")
        cv_data = []
        for name, results in st.session_state.model_results.items():
            if results.get('cv_results'):
                cv_data.append({
                    'Model': name.upper(),
                    'Mean Score': results['cv_results']['mean'],
                    'Std Score': results['cv_results']['std']
                })
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            st.dataframe(cv_df, width="stretch")

            # Boxplot of CV scores
            fig = go.Figure()
            for name, results in st.session_state.model_results.items():
                if results.get('cv_results'):
                    fig.add_trace(go.Box(
                        y=results['cv_results']['scores'],
                        name=name.upper()
                    ))
            fig.update_layout(title="CV Score Distribution", yaxis_title="Score")
            st.plotly_chart(fig, width="stretch")

            # Pairwise statistical comparison (paired t or Wilcoxon on fold-level metrics)
            from ml.eval import compare_models_paired_cv
            cv_names = [n for n, r in st.session_state.model_results.items() if r.get("cv_results")]
            paired = compare_models_paired_cv(
                cv_names,
                st.session_state.model_results,
                task_type=data_config.task_type if data_config else "regression",
            )
            if paired:
                with st.expander("Statistical comparison of models (CV)", expanded=False):
                    st.caption("Pairwise paired tests on fold-level CV scores. Mean Δ = mean(A) − mean(B); p < 0.05 suggests a significant difference.")
                    rows = []
                    for (ma, mb), v in paired.items():
                        mean_d, stat, p, tname = v["mean_delta"], v["stat"], v["p"], v["test_name"]
                        sig = " *" if (p is not None and np.isfinite(p) and p < 0.05) else ""
                        rows.append({"Model A": ma.upper(), "Model B": mb.upper(), "Mean Δ": round(mean_d, 4), "Test": tname, "p": round(p, 4) if p is not None and np.isfinite(p) else None, "Significant": "Yes" if (p is not None and np.isfinite(p) and p < 0.05) else "No"})
                    st.dataframe(pd.DataFrame(rows), width="stretch")

    # ================================================================
    # MODEL INSIGHTS: WHY DID THIS MODEL WIN?
    # ================================================================
    st.markdown("---")
    st.markdown("### 🔍 Why Did This Model Win?")
    
    # Get best model
    metric_col_insights = 'AUC (val)' if task_type_final == 'classification' else 'R² (val)'
    
    if metric_col_insights in comparison_df.columns and len(comparison_df) > 0:
        # Find best performing model
        best_model_idx = comparison_df[metric_col_insights].idxmax()
        best_model_name = comparison_df.loc[best_model_idx, 'Model']
        best_metric_value = comparison_df.loc[best_model_idx, metric_col_insights]
        
        # Generate insights based on model type and data characteristics
        insights = []
        
        # Analyze data characteristics
        df = get_data()
        feature_cols_insights = st.session_state.get('selected_features') or st.session_state.get('feature_cols') or []
        n_samples = len(df)
        n_features = len(feature_cols_insights) if feature_cols_insights else 0
        
        # Calculate data properties
        numeric_features = []
        outlier_count = 0
        high_corr_count = 0
        
        if feature_cols_insights:
            try:
                numeric_features = df[feature_cols_insights].select_dtypes(include=[np.number]).columns.tolist()
                
                # Count outliers
                if len(numeric_features) > 0:
                    for col in numeric_features:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                        outlier_count += outliers
                
                # Check feature correlations
                if len(numeric_features) > 1:
                    corr_matrix = df[numeric_features].corr().abs()
                    high_corr_count = int(((corr_matrix > 0.8) & (corr_matrix < 1.0)).sum().sum() // 2)
            except Exception as e:
                logger.warning(f"Could not analyze data characteristics: {e}")
        
        # Generate model-specific insights
        if best_model_name in ['RF', 'RANDOM_FOREST', 'XGB', 'LGBM']:
            # Tree-based models
            insights.append({
                'reason': 'Handles Non-Linearity',
                'icon': '✅',
                'explanation': 'Tree-based models capture complex, non-linear relationships without manual feature engineering.'
            })
            
            if outlier_count > n_samples * 0.05:
                insights.append({
                    'reason': 'Robust to Outliers',
                    'icon': '✅',
                    'explanation': f'Your data has {outlier_count} outliers. Tree-based models handle these naturally without scaling.'
                })
            
            if high_corr_count > 0:
                insights.append({
                    'reason': 'Handles Collinearity',
                    'icon': '✅',
                    'explanation': f'Found {high_corr_count} highly correlated feature pairs. Trees naturally handle redundant features.'
                })
            
            insights.append({
                'reason': 'No Scaling Required',
                'icon': '⚙️',
                'explanation': 'Unlike linear models, trees work directly on raw feature scales (faster preprocessing).'
            })
            
        elif best_model_name in ['LOGISTIC', 'RIDGE', 'LASSO', 'ELASTIC_NET']:
            # Linear models
            insights.append({
                'reason': 'Linear Relationships',
                'icon': '✅',
                'explanation': 'Your outcome appears to have linear relationships with predictors.'
            })
            
            insights.append({
                'reason': 'High Interpretability',
                'icon': '🔍',
                'explanation': 'Coefficients directly show feature importance and direction of effect (unlike black-box models).'
            })
            
            if best_model_name in ['RIDGE', 'LASSO', 'ELASTIC_NET']:
                insights.append({
                    'reason': 'Regularization Benefit',
                    'icon': '✅',
                    'explanation': f'L1/L2 regularization prevents overfitting with {n_features} features and {n_samples} samples.'
                })
            
            insights.append({
                'reason': 'Fast Predictions',
                'icon': '⚡',
                'explanation': 'Linear models are near-instant for deployment (important for real-time applications).'
            })

        elif best_model_name in ['SVM_LINEAR', 'SVM_RBF']:
            # SVM
            insights.append({
                'reason': 'Maximum Margin',
                'icon': '✅',
                'explanation': 'SVM finds optimal decision boundary that maximizes class separation.'
            })
            
            if best_model_name == 'SVM_RBF':
                insights.append({
                    'reason': 'Non-Linear Kernel',
                    'icon': '✅',
                    'explanation': 'RBF kernel captures complex, non-linear decision boundaries.'
                })
            
            if n_samples < 1000:
                insights.append({
                    'reason': 'Works on Small Data',
                    'icon': '✅',
                    'explanation': f'SVMs often excel with smaller datasets ({n_samples} samples).'
                })

        elif best_model_name in ['NN', 'MLP']:
            # Neural networks
            insights.append({
                'reason': 'Complex Patterns',
                'icon': '✅',
                'explanation': 'Neural networks can learn highly complex, hierarchical representations.'
            })
            
            insights.append({
                'reason': 'Feature Interactions',
                'icon': '✅',
                'explanation': 'Hidden layers automatically discover feature interactions without manual engineering.'
            })
        
        # Display insights
        if insights:
            metric_name_display = metric_col_insights.replace(' (val)', '')
            st.markdown(f"**{best_model_name}** achieved the best performance ({metric_name_display}: {best_metric_value:.3f}). Here's why:")
            
            for insight in insights:
                st.markdown(f"""
{insight['icon']} **{insight['reason']}**  
{insight['explanation']}
""")
        
        # Trade-offs section
        st.markdown("### ⚖️ Trade-offs to Consider")
        
        if best_model_name in ['RF', 'RANDOM_FOREST', 'XGB', 'LGBM']:
            st.markdown("""
- ⚠️ **Less interpretable** than linear models (use SHAP for explanations)
- ⚠️ **Slower predictions** than linear models (ensemble of trees)
- ⚠️ **Larger memory** footprint for deployment
- ✅ **But:** Superior performance often worth the trade-off
""")

        elif best_model_name in ['LOGISTIC', 'RIDGE', 'LASSO']:
            st.markdown("""
- ⚠️ **Assumes linearity** (may miss complex patterns)
- ⚠️ **Requires scaling** (preprocessing adds complexity)
- ✅ **But:** Highly interpretable coefficients
- ✅ **But:** Fast, lightweight deployment
""")

        elif best_model_name in ['NN', 'MLP']:
            st.markdown("""
- ⚠️ **Black box** (hardest to interpret)
- ⚠️ **Requires tuning** (many hyperparameters)
- ⚠️ **Needs more data** (risk of overfitting on small datasets)
- ✅ **But:** Can capture any pattern given enough data
""")
        
        elif best_model_name in ['SVM_LINEAR', 'SVM_RBF']:
            st.markdown("""
- ⚠️ **Can be slow** to train on large datasets
- ⚠️ **Sensitive to scaling** (requires careful preprocessing)
- ⚠️ **Hyperparameter tuning** critical for good performance
- ✅ **But:** Strong theoretical foundation (maximum margin)
""")
        
        else:
            st.markdown("""
- Check model documentation for specific trade-offs
- Consider interpretability, speed, and deployment requirements
- Use SHAP for post-hoc explanations if needed
""")
    else:
        st.info("Train models to see performance insights")

    # ================================================================
    # MODEL SELECTION GUIDANCE
    # ================================================================
    st.markdown("---")
    st.markdown("### 🎯 How to Choose Your Model")
    
    # Helper function for complexity description
    def get_model_complexity(model_name: str) -> str:
        """Return human-readable complexity description."""
        simple_models = ['LOGISTIC', 'RIDGE', 'LASSO', 'ELASTIC_NET']
        moderate_models = ['RF', 'RANDOM_FOREST', 'SVM_LINEAR']
        complex_models = ['XGB', 'LGBM', 'NN', 'SVM_RBF']
        
        if model_name in simple_models:
            return "Simple, highly interpretable"
        elif model_name in moderate_models:
            return "Moderate complexity, good interpretability"
        elif model_name in complex_models:
            return "Complex, black-box (use SHAP for interpretation)"
        else:
            return "Unknown complexity"
    
    # Get top 3 models by performance
    metric_col = 'AUC (val)' if task_type_final == 'classification' else 'R² (val)'
    
    # Check if the metric column exists in the comparison_df
    if metric_col in comparison_df.columns and len(comparison_df) > 0:
        top_models = comparison_df.nlargest(3, metric_col)
        
        # Check if top models have overlapping confidence intervals
        if len(top_models) >= 2:
            # Get bootstrap CIs for top 2 models
            model1_name = top_models.index[0]
            model2_name = top_models.index[1]
            
            bootstrap_results = st.session_state.get("bootstrap_results", {})
            
            if bootstrap_results:
                # Get BootstrapResult objects
                metric_name = metric_col.replace(' (val)', '')  # 'AUC' or 'R²'
                model1_result = bootstrap_results.get(model1_name, {}).get(metric_name)
                model2_result = bootstrap_results.get(model2_name, {}).get(metric_name)
                
                # Extract CI bounds from BootstrapResult dataclass
                from ml.bootstrap import BootstrapResult
                if isinstance(model1_result, BootstrapResult) and isinstance(model2_result, BootstrapResult):
                    model1_ci = [model1_result.ci_lower, model1_result.ci_upper]
                    model2_ci = [model2_result.ci_lower, model2_result.ci_upper]
                    
                    # Check for overlap
                    ci_overlap = (model1_ci[0] <= model2_ci[1] and model2_ci[0] <= model1_ci[1])
                else:
                    # Bootstrap results incomplete, skip CI analysis
                    ci_overlap = None
                
                if ci_overlap is not None:
                    if ci_overlap:
                        st.info(f"""
                        **Models Perform Similarly**
                        
                        Your top models ({model1_name}, {model2_name}) have overlapping confidence intervals,
                        meaning there's no statistically significant performance difference.
                        
                        **Decision Framework:**
                        
                        1. **Interpretability** → Choose simpler model
                           - Order: Logistic Regression > Ridge/LASSO > Random Forest > Gradient Boosting > Neural Networks
                           - For publication: Simpler models are easier to explain to reviewers
                        
                        2. **Deployment** → Choose faster model
                           - Logistic/Ridge: Near-instant predictions
                           - Random Forest/XGBoost: Fast but larger memory footprint
                           - Neural Networks: Slower, requires PyTorch/TensorFlow
                        
                        3. **Calibration** → Check the Explainability page
                           - Well-calibrated models have predicted probabilities that match observed frequencies
                           - Important for risk prediction and clinical applications
                        
                        4. **Robustness** → Check Sensitivity Analysis (next page)
                           - Some models are more sensitive to random seed or feature dropout
                        
                        **Our Recommendation:**
                        """)
                        
                        # Auto-recommend based on overlap
                        if model1_name in ['LOGISTIC', 'RIDGE', 'LASSO']:
                            st.success(f"✅ **{model1_name}**: Best balance of performance, interpretability, and deployability.")
                        elif model1_name in ['RF', 'RANDOM_FOREST']:
                            st.success(f"✅ **{model1_name}**: Excellent choice. Robust, feature importances available, widely trusted.")
                        else:
                            st.success(f"✅ **{model1_name}**: Best performer, but consider if interpretability is important.")
                        
                        # Show all top 3
                        st.markdown("**Top 3 Models:**")
                        for idx, (model_name, row) in enumerate(top_models.iterrows(), 1):
                            metric_val = row[metric_col]
                            complexity = get_model_complexity(model_name)
                            st.markdown(f"{idx}. **{model_name}**: {metric_val:.3f} — {complexity}")
                    
                    else:
                        st.success(f"""
                        **Clear Winner**
                        
                        **{model1_name}** significantly outperforms other models (non-overlapping CIs).
                        
                        → **Recommendation:** Use {model1_name} unless you have specific concerns about interpretability or deployment.
                        """)
            else:
                st.info("💡 **Tip:** Compute Bootstrap CIs above to get statistical guidance on model selection.")
        
        else:
            st.info("Train multiple models to see comparison and recommendations.")
    else:
        st.info("Train models to see selection guidance.")

    # ================================================================
    # DIAGNOSTIC ASSISTANT FOR POOR PERFORMANCE
    # ================================================================
    # Check if we have model results and comparison data
    if len(comparison_df) > 0 and metric_col in comparison_df.columns:
        best_metric = comparison_df[metric_col].max()
        
        # Define poor performance thresholds
        if task_type_final == 'classification':
            poor_threshold = 0.65
            metric_name = 'AUC'
        else:
            poor_threshold = 0.40
            metric_name = 'R²'
        
        # Trigger diagnostics if performance is poor
        if best_metric < poor_threshold:
            st.markdown("---")
            st.error(f"""
            ⚠️ **Poor Model Performance Detected**
            
            Your best model achieved {metric_name} = {best_metric:.2f}, which is below the acceptable threshold ({poor_threshold:.2f}).
            """)
            
            st.markdown("### 🔍 Diagnostic Analysis")
            
            # Run diagnostics
            diagnostics = []
            
            # Get the raw data
            df = get_data()
            target = data_config.target_col if data_config else None
            feature_cols = st.session_state.get('selected_features') or (data_config.feature_cols if data_config else None)
            
            if df is not None and target and feature_cols:
                # Check 1: Feature-target correlations
                numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 0:
                    try:
                        correlations = df[numeric_features].corrwith(df[target]).abs()
                        max_corr = correlations.max()
                        
                        if max_corr < 0.1:
                            diagnostics.append({
                                'issue': 'Weak Features',
                                'severity': 'HIGH',
                                'description': f'No feature has correlation >0.1 with target (max: {max_corr:.3f})',
                                'action': 'Review EDA for feature-target relationships. Consider Feature Engineering or collect more informative data.'
                            })
                    except Exception:
                        pass  # Skip if correlation calculation fails
                
                # Check 2: Sample size
                n_samples = len(df)
                n_features = len(feature_cols) if feature_cols else 0
                samples_per_feature = n_samples / n_features if n_features > 0 else 0
                
                if samples_per_feature < 10:
                    diagnostics.append({
                        'issue': 'Insufficient Data',
                        'severity': 'HIGH',
                        'description': f'Only {samples_per_feature:.1f} samples per feature (need ≥10-20)',
                        'action': 'Reduce features via Feature Selection or collect more samples. Consider this a pilot study.'
                    })
                
                # Check 3: Class imbalance (classification only)
                if task_type_final == 'classification':
                    try:
                        class_counts = df[target].value_counts()
                        minority_pct = (class_counts.min() / class_counts.sum()) * 100
                        
                        if minority_pct < 10:
                            diagnostics.append({
                                'issue': 'Severe Class Imbalance',
                                'severity': 'HIGH',
                                'description': f'Minority class is only {minority_pct:.1f}% of data',
                                'action': 'Use stratified splits (already done), consider class weights, or collect more minority samples.'
                            })
                    except Exception:
                        pass  # Skip if class imbalance check fails
                
                # Check 4: Missing data
                try:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    if missing_pct > 20:
                        diagnostics.append({
                            'issue': 'High Missing Data',
                            'severity': 'MEDIUM',
                            'description': f'{missing_pct:.1f}% of data is missing',
                            'action': 'Review Preprocessing step. Consider multiple imputation or dropping high-missingness features.'
                        })
                except Exception:
                    pass  # Skip if missing data check fails
            
            # Display diagnostics
            if diagnostics:
                for diag in diagnostics:
                    severity_icon = '🔴' if diag['severity'] == 'HIGH' else '🟡'
                    st.markdown(f"""
**{severity_icon} {diag['issue']}**  
{diag['description']}

**→ Action:** {diag['action']}
""")
            else:
                # No specific issues detected, provide general guidance
                if df is not None:
                    n_samples = len(df)
                else:
                    n_samples = "unknown"
                    
                st.info(f"""
**No obvious data quality issues detected.**

Poor performance may be due to:
- Inherent unpredictability of the outcome
- Missing important features not in your dataset
- Need for more complex feature engineering
- Small dataset size (current: {n_samples} samples)

**Recommendation:** Frame this as an exploratory/pilot study. Report confidence intervals and discuss limitations.
""")

    # Model diagnostics (one tab per model so pred-vs-actual etc. visible for all)
    st.header("Model Diagnostics")
    model_names = list(st.session_state.trained_models.keys())
    if not model_names:
        st.info("No models to show.")
    else:
        tabs = st.tabs([f"{n.upper()}" for n in model_names])
        _fn_by_model = st.session_state.get("feature_names_by_model", {})
        for tab, name in zip(tabs, model_names):
            with tab:
                model = st.session_state.trained_models[name]
                results = st.session_state.model_results[name]
                _feats = _fn_by_model.get(name) or (data_config.feature_cols if data_config else [])
                _n_test = len(results.get("y_test", []))
                _task = data_config.task_type if data_config else None

                fitted_prep = st.session_state.get("fitted_preprocessing_pipelines", {}).get(name)
                if fitted_prep is not None:
                    from ml.pipeline import get_pipeline_recipe
                    st.subheader("Preprocessing used")
                    st.caption(f"Pipeline for **{name.upper()}**")
                    st.code(get_pipeline_recipe(fitted_prep), language=None)
                    st.markdown("---")

                st.subheader("Test Set Metrics")
                metrics = results["metrics"]
                metric_cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric_name, f"{metric_value:.4f}")

                if name == "nn" and results.get("history", {}).get("train_loss"):
                    st.subheader("Learning Curves")
                    st.plotly_chart(plot_training_history(results["history"]), width="stretch", key=f"diag_lc_{name}")
                    from ml.plot_narrative import narrative_learning_curves
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    nar = narrative_learning_curves(results["history"])
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    h = results["history"]
                    tl, vl = h.get("train_loss", []), h.get("val_loss", h.get("train_loss", []))
                    stats_summary = f"train_loss={tl[-1]:.4f}; val_loss={vl[-1]:.4f}" if tl else ""
                    ctx = build_llm_context("learning_curves", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_lc_{name}", result_session_key=f"llm_result_lc_{name}")

                if data_config.task_type == "regression":
                    st.subheader("Predictions vs Actual")
                    st.plotly_chart(
                        plot_predictions_vs_actual(results["y_test"], results["y_test_pred"], title=f"{name.upper()} Predictions"),
                        width="stretch",
                        key=f"diag_pva_{name}",
                    )
                    st.caption("The dashed red line (y = x) represents perfect agreement. Points closer to it indicate better predictions.")
                    from ml.eval import analyze_pred_vs_actual
                    from ml.plot_narrative import narrative_pred_vs_actual
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    pva_stats = analyze_pred_vs_actual(results["y_test"], results["y_test_pred"])
                    nar = narrative_pred_vs_actual(pva_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    stats_summary = f"corr={pva_stats.get('correlation', 0):.3f}; mean_err={pva_stats.get('mean_error', 0):.4f}"
                    ctx = build_llm_context("pred_vs_actual", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_pva_{name}", result_session_key=f"llm_result_pva_{name}")

                    st.subheader("Residuals")
                    st.plotly_chart(
                        plot_residuals(results["y_test"], results["y_test_pred"], title=f"{name.upper()} Residuals"),
                        width="stretch",
                        key=f"diag_resid_{name}",
                    )
                    from ml.eval import analyze_residuals_extended
                    from ml.plot_narrative import narrative_residuals
                    resid_stats = analyze_residuals_extended(results["y_test"], results["y_test_pred"])
                    nar = narrative_residuals(resid_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    else:
                        res_basic = analyze_residuals(results["y_test"], results["y_test_pred"])
                        st.caption(f"Mean residual: {res_basic['mean_residual']:.4f} | Std: {res_basic['std_residual']:.4f}")
                    stats_summary = f"skew={resid_stats.get('skew', 0):.3f}; iqr={resid_stats.get('iqr', 0):.4f}; rvp={resid_stats.get('residual_vs_predicted_corr', 0):.3f}"
                    ctx = build_llm_context("residuals", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_resid_{name}", result_session_key=f"llm_result_resid_{name}")
                else:
                    st.subheader("Classification Performance")
                    from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, precision_recall_curve, auc
                    from ml.eval import analyze_confusion_matrix
                    from ml.plot_narrative import narrative_confusion_matrix
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button

                    # ROC and PR curves (if model supports probability predictions)
                    if model.supports_proba() and "y_test_proba" in results:
                        y_proba = results["y_test_proba"]
                        y_true = results["y_test"]

                        # Handle binary vs multiclass
                        unique_classes = np.unique(y_true)
                        if len(unique_classes) == 2:
                            # Binary: ROC curve
                            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                            fpr, tpr, _ = roc_curve(y_true, proba_pos)
                            roc_auc = auc(fpr, tpr)
                            fig_roc = px.area(x=fpr, y=tpr, labels=dict(x="False Positive Rate", y="True Positive Rate"),
                                              title=f"ROC Curve (AUC = {roc_auc:.3f})")
                            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
                            fig_roc.update_layout(template="plotly_white")
                            st.plotly_chart(fig_roc, use_container_width=True, key=f"diag_roc_{name}")

                            # Precision-Recall curve
                            prec, rec, _ = precision_recall_curve(y_true, proba_pos)
                            pr_auc = auc(rec, prec)
                            fig_pr = px.area(x=rec, y=prec, labels=dict(x="Recall", y="Precision"),
                                             title=f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
                            baseline = np.mean(y_true == unique_classes[1]) if len(unique_classes) == 2 else 0.5
                            fig_pr.add_shape(type="line", x0=0, x1=1, y0=baseline, y1=baseline, line=dict(dash="dash", color="gray"))
                            fig_pr.update_layout(template="plotly_white")
                            st.plotly_chart(fig_pr, use_container_width=True, key=f"diag_pr_{name}")
                        else:
                            # Multiclass: per-class ROC curves
                            from sklearn.preprocessing import label_binarize
                            y_bin = label_binarize(y_true, classes=unique_classes)
                            fig_roc = go.Figure()
                            for i, cls in enumerate(unique_classes):
                                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                                auc_i = auc(fpr_i, tpr_i)
                                fig_roc.add_trace(go.Scatter(x=fpr_i, y=tpr_i, mode='lines', name=f"Class {cls} (AUC={auc_i:.3f})"))
                            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
                            fig_roc.update_layout(title="ROC Curves (One-vs-Rest)", xaxis_title="FPR", yaxis_title="TPR", template="plotly_white")
                            st.plotly_chart(fig_roc, use_container_width=True, key=f"diag_roc_{name}")
                    elif not model.supports_proba():
                        st.caption("This model does not support probability predictions — ROC/PR curves unavailable.")

                    # Confusion Matrix
                    cm = sk_confusion_matrix(results["y_test"], results["y_test_pred"])
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"diag_cm_{name}")
                    cm_stats = analyze_confusion_matrix(results["y_test"], results["y_test_pred"])
                    nar = narrative_confusion_matrix(cm_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    per = cm_stats.get("per_class", [])[:3]
                    stats_summary = "; ".join(f"{p.get('label','?')}: P={p.get('precision',0):.2f} R={p.get('recall',0):.2f}" for p in per) if per else ""
                    ctx = build_llm_context("confusion_matrix", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_cm_{name}", result_session_key=f"llm_result_cm_{name}")
