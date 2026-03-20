"""
Session state management for multi-page Streamlit app.
Defines schema and initialization functions.
"""
import streamlit as st
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


@dataclass
class TaskTypeDetection:
    """Task type detection results and overrides."""
    detected: Optional[Literal["regression", "classification"]] = None
    confidence: Optional[Literal["low", "med", "high"]] = None
    reasons: List[str] = field(default_factory=list)
    override_enabled: bool = False
    override_value: Optional[Literal["regression", "classification"]] = None
    
    @property
    def final(self) -> Optional[Literal["regression", "classification"]]:
        """Get final task type (override if enabled, else detected)."""
        if self.override_enabled and self.override_value is not None:
            return self.override_value
        return self.detected


@dataclass
class CohortStructureDetection:
    """Cohort structure detection results and overrides."""
    detected: Optional[Literal["cross_sectional", "longitudinal"]] = None
    confidence: Optional[Literal["low", "med", "high"]] = None
    reasons: List[str] = field(default_factory=list)
    override_enabled: bool = False
    override_value: Optional[Literal["cross_sectional", "longitudinal"]] = None
    entity_id_candidates: List[str] = field(default_factory=list)
    entity_id_detected: Optional[str] = None
    entity_id_override_enabled: bool = False
    entity_id_override_value: Optional[str] = None
    time_column_candidates: List[str] = field(default_factory=list)
    
    @property
    def final(self) -> Optional[Literal["cross_sectional", "longitudinal"]]:
        """Get final cohort type (override if enabled, else detected)."""
        if self.override_enabled and self.override_value is not None:
            return self.override_value
        return self.detected
    
    @property
    def entity_id_final(self) -> Optional[str]:
        """Get final entity ID column (override if enabled, else detected)."""
        if self.entity_id_override_enabled and self.entity_id_override_value is not None:
            return self.entity_id_override_value
        return self.entity_id_detected


@dataclass
class DataConfig:
    """Configuration for dataset and target/feature selection."""
    target_col: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)
    datetime_col: Optional[str] = None  # For time-series splits
    task_type: Optional[str] = None  # 'regression' or 'classification' (DEPRECATED: use task_type_detection.final)


@dataclass
class SplitConfig:
    """Configuration for train/val/test splits."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = False  # For classification
    use_time_split: bool = False  # Use datetime_col for splitting
    datetime_col: Optional[str] = None  # Column to use for time-based splitting


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters."""
    # Neural Network
    nn_epochs: int = 200
    nn_batch_size: int = 256
    nn_lr: float = 0.0015
    nn_weight_decay: float = 0.0002
    nn_patience: int = 30
    nn_dropout: float = 0.1
    
    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 10
    
    # GLM/Huber
    huber_epsilon: float = 1.35
    huber_alpha: float = 0.0


def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        # Data
        'raw_data': None,
        'df_engineered': None,  # Dataset after feature engineering
        'feature_engineering_applied': False,
        'engineered_feature_names': [],
        'data_config': DataConfig(),
        'data_audit': None,
        
        # Project-based dataset management
        'task_mode': None,  # 'prediction' | 'hypothesis_testing'
        'datasets_registry': {},  # Dict mapping dataset_id -> DataFrame
        'working_table': None,  # The merged/active DataFrame for analysis
        'merge_steps': [],  # List of merge operations
        'last_merge_columns': [],  # Columns from the last merge result
        
        # Detection and triage
        'task_type_detection': TaskTypeDetection(),
        'cohort_structure_detection': CohortStructureDetection(),
        
        # Preprocessing
        'preprocessing_pipeline': None,
        'preprocessing_config': None,
        'preprocessing_pipelines_by_model': {},
        'preprocessing_config_by_model': {},
        
        # Splits
        'split_config': SplitConfig(),
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'feature_names': None,
        'feature_names_by_model': {},
        
        # Models
        'model_config': ModelConfig(),
        'trained_models': {},  # Dict[str, Any] - model name -> model wrapper object
        'model_results': {},  # Dict[str, Dict] - model name -> metrics/history
        'fitted_estimators': {},  # Dict[str, Any] - model name -> fitted sklearn-compatible estimator/pipeline
        'fitted_preprocessing_pipelines': {},  # Dict[str, Pipeline] - model name -> preprocessing pipeline used
        
        # Evaluation
        'cv_results': None,  # For k-fold CV
        'use_cv': False,
        'cv_folds': 5,
        
        # Explainability
        'permutation_importance': {},
        'partial_dependence': {},
        'explainability_robustness': {},

        # EDA
        'eda_results': {},  # Dict[str, Dict] - recommendation_id -> results
        'eda_insights': [],  # LEGACY — backward compat, computed from insight_ledger
        
        # Report
        'report_data': None,
        
        # Global settings
        'random_seed': 42,  # Global random seed
        'data_source': None,  # Track data source (uploaded CSV, built-in dataset, etc.)
        'data_filename': None,  # Track filename for uploads or dataset label
        'dataset_id': None,  # Incrementing dataset identifier
        'dataset_history': [],  # Archive of replaced datasets (metadata only)
        'has_completed_tour': False,  # Guided tour dismissed/completed
        'show_guided_tour': False,  # Expand guided tour in sidebar
        'workflow_mode': 'quick',  # 'quick' | 'advanced' navigation emphasis only
        
        # Methodology logging for auto-generated methods section
        'methodology_log': [],  # List of methodology actions for publication
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Insight ledger — single logical layer for cross-page insight tracking.
    # Initialized separately because it's a class instance, not a plain default.
    if 'insight_ledger' not in st.session_state:
        from utils.insight_ledger import InsightLedger
        st.session_state.insight_ledger = InsightLedger()


def get_data() -> Optional[pd.DataFrame]:
    """Get active data from session state. 
    Priority: df_engineered (if feature engineering was applied) > filtered_data > raw_data"""
    # Explicitly check for None to avoid DataFrame boolean ambiguity
    df_eng = st.session_state.get('df_engineered')
    if df_eng is not None:
        return df_eng
    
    df_filt = st.session_state.get('filtered_data')
    if df_filt is not None:
        return df_filt
    
    return st.session_state.get('raw_data')


def set_data(df: pd.DataFrame):
    """Set raw data in session state. Clears filtered_data so it is not stale."""
    st.session_state.raw_data = df
    st.session_state.pop("filtered_data", None)


def reset_data_dependent_state():
    """Reset state that depends on the active dataset."""
    st.session_state.data_config = DataConfig()
    st.session_state.data_audit = None
    st.session_state.task_type_detection = TaskTypeDetection()
    st.session_state.cohort_structure_detection = CohortStructureDetection()
    # Note: task_mode and datasets_registry are NOT reset here
    # as they are workflow-level, not dataset-specific

    st.session_state.preprocessing_pipeline = None
    st.session_state.preprocessing_config = None
    st.session_state.preprocessing_pipelines_by_model = {}
    st.session_state.preprocessing_config_by_model = {}
    st.session_state.pop("filtered_data", None)
    
    # Clear feature engineering state
    st.session_state.pop("df_engineered", None)
    st.session_state.feature_engineering_applied = False
    st.session_state.engineered_feature_names = []
    st.session_state.pop("engineering_log", None)

    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.feature_names = None
    st.session_state.feature_names_by_model = {}

    st.session_state.trained_models = {}
    st.session_state.model_results = {}
    st.session_state.fitted_estimators = {}
    st.session_state.fitted_preprocessing_pipelines = {}

    st.session_state.cv_results = None
    st.session_state.use_cv = False
    st.session_state.cv_folds = 5

    st.session_state.permutation_importance = {}
    st.session_state.partial_dependence = {}
    st.session_state.explainability_robustness = {}
    st.session_state.eda_results = {}
    st.session_state.eda_insights = []
    st.session_state.report_data = None

    # Reset insight ledger
    from utils.insight_ledger import InsightLedger
    st.session_state.insight_ledger = InsightLedger()
    for key in (
        'methods_section', 'flow_diagram', 'tripod_tracker', 'latex_report',
        'report_best_model', 'report_model_selection', 'report_explain_selection',
        'report_include_results', 'report_include_llm', 'shap_matplotlib_figs'
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("target_label_encoder", None)
    st.session_state.pop("train_indices", None)
    st.session_state.pop("test_indices", None)


def get_preprocessing_pipeline(model_key: Optional[str] = None) -> Optional[Pipeline]:
    """Get preprocessing pipeline from session state."""
    if model_key:
        pipelines = st.session_state.get('preprocessing_pipelines_by_model', {})
        if model_key in pipelines:
            return pipelines[model_key]
    return st.session_state.get('preprocessing_pipeline')


def set_preprocessing_pipeline(pipeline: Pipeline, config: Dict[str, Any]):
    """Set preprocessing pipeline and config."""
    st.session_state.preprocessing_pipeline = pipeline
    st.session_state.preprocessing_config = config


def set_preprocessing_pipelines(pipelines_by_model: Dict[str, Pipeline], configs_by_model: Dict[str, Any], base_config: Dict[str, Any]):
    """Set model-specific preprocessing pipelines and configs."""
    st.session_state.preprocessing_pipelines_by_model = pipelines_by_model
    st.session_state.preprocessing_config_by_model = configs_by_model
    # Preserve a default pipeline for legacy access
    default_pipeline = pipelines_by_model.get('default') or next(iter(pipelines_by_model.values()), None)
    if default_pipeline:
        st.session_state.preprocessing_pipeline = default_pipeline
    st.session_state.preprocessing_config = base_config


def get_splits() -> Optional[tuple]:
    """Get train/val/test splits from session state."""
    if st.session_state.get('X_train') is None:
        return None
    return (
        st.session_state.X_train,
        st.session_state.X_val,
        st.session_state.X_test,
        st.session_state.y_train,
        st.session_state.y_val,
        st.session_state.y_test,
    )


def set_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names: List[str]):
    """Set train/val/test splits in session state."""
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.session_state.y_test = y_test
    st.session_state.feature_names = feature_names


def add_trained_model(name: str, model: Any, results: Dict[str, Any]):
    """Add a trained model and its results to session state."""
    st.session_state.trained_models[name] = model
    st.session_state.model_results[name] = results


def log_methodology(step: str, action: str, details: Optional[Dict[str, Any]] = None):
    """Log a methodology action for the final report.
    
    Args:
        step: Workflow step name (e.g., 'Feature Engineering', 'Feature Selection')
        action: Description of what was done
        details: Optional dict with additional parameters
    """
    from datetime import datetime
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'action': action,
        'details': details or {}
    }
    
    if 'methodology_log' not in st.session_state:
        st.session_state.methodology_log = []
    
    # Steps where re-doing replaces the previous entry (user iterates on a single config)
    REPLACE_STEPS = {'Upload & Audit', 'Feature Engineering', 'Feature Selection Applied',
                     'Preprocessing', 'Model Training', 'Explainability'}
    
    log = st.session_state.methodology_log
    if step in REPLACE_STEPS:
        # Last-wins: replace previous entry with same step
        for i in range(len(log) - 1, -1, -1):
            if log[i]['step'] == step:
                log[i] = entry
                return
    # Additive steps (EDA, Statistical Validation, Data Cleaning) — always append
    log.append(entry)
