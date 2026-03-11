"""
Publication engine: methods section generator, flow diagrams, TRIPOD tracking.

Generates publication-ready text, figures, and compliance checklists.
"""
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# TRIPOD Checklist
# ============================================================================

TRIPOD_ITEMS = [
    {"id": "1", "section": "Title", "item": "Identify the study as developing and/or validating a prediction model", "auto_key": "title"},
    {"id": "2", "section": "Abstract", "item": "Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions", "auto_key": "abstract"},
    {"id": "3a", "section": "Introduction", "item": "Explain the medical context and rationale for developing/validating the prediction model", "auto_key": "background"},
    {"id": "3b", "section": "Introduction", "item": "Specify the objectives, including whether developing and/or validating", "auto_key": "objectives"},
    {"id": "4a", "section": "Methods", "item": "Describe the study design or source of data", "auto_key": "study_design"},
    {"id": "4b", "section": "Methods", "item": "Specify the key study dates", "auto_key": "study_dates"},
    {"id": "5a", "section": "Methods", "item": "Specify key elements of the study setting", "auto_key": "setting"},
    {"id": "5b", "section": "Methods", "item": "Describe eligibility criteria for participants", "auto_key": "eligibility"},
    {"id": "6a", "section": "Methods", "item": "Clearly define the outcome", "auto_key": "outcome_defined"},
    {"id": "7a", "section": "Methods", "item": "Clearly define all predictors used in the model", "auto_key": "predictors_defined"},
    {"id": "8", "section": "Methods", "item": "Explain how the study size was arrived at", "auto_key": "sample_size"},
    {"id": "9", "section": "Methods", "item": "Describe how missing data were handled", "auto_key": "missing_data"},
    {"id": "10a", "section": "Methods", "item": "Describe how predictors were handled in the analyses", "auto_key": "predictor_handling"},
    {"id": "10b", "section": "Methods", "item": "Specify type of model, all model-building procedures, and method for internal validation", "auto_key": "model_building"},
    {"id": "10d", "section": "Methods", "item": "Specify all measures used to assess model performance", "auto_key": "performance_measures"},
    {"id": "13a", "section": "Results", "item": "Describe the flow of participants through the study", "auto_key": "participant_flow"},
    {"id": "13b", "section": "Results", "item": "Describe the characteristics of participants", "auto_key": "table1"},
    {"id": "14a", "section": "Results", "item": "Specify the number of participants and outcome events in each analysis", "auto_key": "sample_counts"},
    {"id": "15a", "section": "Results", "item": "Present the full prediction model to allow predictions for individuals", "auto_key": "full_model"},
    {"id": "16", "section": "Results", "item": "Report performance measures with confidence intervals", "auto_key": "performance_ci"},
    {"id": "19a", "section": "Discussion", "item": "Give an overall interpretation of results considering objectives, limitations, and results from similar studies", "auto_key": "interpretation"},
    {"id": "19b", "section": "Discussion", "item": "Discuss any limitations of the study", "auto_key": "limitations"},
]


@dataclass
class TRIPODTracker:
    """Track TRIPOD checklist completion."""
    completed: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    page_refs: Dict[str, str] = field(default_factory=dict)

    def mark_complete(self, auto_key: str, note: str = "", page_ref: str = ""):
        self.completed[auto_key] = True
        if note:
            self.notes[auto_key] = note
        if page_ref:
            self.page_refs[auto_key] = page_ref

    def get_progress(self) -> Tuple[int, int]:
        total = len(TRIPOD_ITEMS)
        done = sum(1 for item in TRIPOD_ITEMS if self.completed.get(item["auto_key"], False))
        return done, total

    def get_checklist_df(self) -> pd.DataFrame:
        rows = []
        for item in TRIPOD_ITEMS:
            key = item["auto_key"]
            rows.append({
                "Item": item["id"],
                "Section": item["section"],
                "Description": item["item"],
                "Status": "✅" if self.completed.get(key, False) else "⬜",
                "Notes": self.notes.get(key, ""),
                "Page": self.page_refs.get(key, ""),
            })
        return pd.DataFrame(rows)


# ============================================================================
# Methods Section Generator
# ============================================================================

def generate_methods_from_log() -> Dict[str, List[Dict[str, Any]]]:
    """Extract methodology actions grouped by step from session state log.
    
    Returns:
        Dictionary mapping step name to list of log entries for that step.
    """
    try:
        import streamlit as st
        log = st.session_state.get('methodology_log', [])
    except ImportError:
        # Not in Streamlit context
        return {}
    
    steps = {}
    for entry in log:
        step = entry.get('step', 'Unknown')
        if step not in steps:
            steps[step] = []
        steps[step].append(entry)
    
    return steps


def _determine_best_model(selected_model_results: Dict[str, Dict], task_type: str) -> Optional[str]:
    """Determine the best model from actual metrics.
    
    Args:
        selected_model_results: Dict of model_name -> {"metrics": {...}}
        task_type: "regression" or "classification"
    
    Returns:
        Name of the best model, or None if no results.
    """
    if not selected_model_results:
        return None
    
    if task_type == "regression":
        # Lowest RMSE wins
        best = min(
            selected_model_results.items(),
            key=lambda x: x[1].get('metrics', {}).get('RMSE', float('inf'))
        )
        return best[0] if best[1].get('metrics', {}).get('RMSE') != float('inf') else None
    else:
        # Highest F1 (or AUC if F1 not available, or Accuracy as fallback)
        def score(item):
            metrics = item[1].get('metrics', {})
            if 'F1' in metrics:
                return metrics['F1']
            elif 'AUC' in metrics:
                return metrics['AUC']
            elif 'Accuracy' in metrics:
                return metrics['Accuracy']
            else:
                return -float('inf')
        
        best = max(selected_model_results.items(), key=score)
        return best[0] if score(best) > -float('inf') else None


def generate_methods_section(
    data_config: Dict[str, Any],
    preprocessing_config: Dict[str, Any],
    model_configs: Dict[str, Any],
    split_config: Dict[str, Any],
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    feature_names: List[str],
    target_name: str,
    task_type: str,
    metrics_used: List[str],
    cv_folds: Optional[int] = None,
    feature_selection_method: Optional[str] = None,
    missing_data_strategy: Optional[str] = None,
    external_validation: bool = False,
    # NEW: actual results for a richer methods/results section
    selected_model_results: Optional[Dict[str, Dict]] = None,
    bootstrap_results: Optional[Dict[str, Dict]] = None,
    best_model_name: Optional[str] = None,
    explainability_methods: Optional[List[str]] = None,
    calibration_results: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> str:
    """Generate a draft methods section for a publication.

    Returns formatted text with placeholders for study-specific details.
    """
    sections = []

    # Study design
    sections.append("### Study Design and Participants\n")
    sections.append(
        f"[PLACEHOLDER: Describe your study design, data source, and recruitment/selection criteria.] "
        f"The analysis included {n_total:,} observations with {len(feature_names)} predictor variables. "
        f"The outcome variable was {target_name}"
        + (", treated as a continuous outcome (regression)." if task_type == "regression"
           else ", treated as a categorical outcome (classification).")
    )

    # Predictors
    sections.append("\n\n### Predictor Variables\n")
    if len(feature_names) <= 15:
        feat_list = ", ".join(feature_names)
        sections.append(f"The following predictor variables were included: {feat_list}.")
    else:
        sections.append(
            f"A total of {len(feature_names)} predictor variables were included "
            f"(see Supplementary Table S1 for full list)."
        )

    # Feature selection: use logged data as source of truth
    logged_steps = generate_methods_from_log()
    
    # Check for "Feature Selection Applied" step first (the final applied selection)
    feature_selection_logged = False
    for fs_step in ['Feature Selection Applied', 'Feature Selection']:
        if feature_selection_logged:
            break
        if fs_step not in logged_steps:
            continue
        for entry in logged_steps[fs_step]:
            details = entry.get('details', {})
            n_before = details.get('n_features_before')
            n_after = details.get('n_features_after')
            methods = details.get('methods', [])
            method = details.get('method', '')
            if n_before and n_after:
                methods_str = ", ".join(methods) if methods else method
                if n_before == n_after:
                    # All features retained — say so clearly
                    if methods_str:
                        sections.append(
                            f" Feature selection was performed using {methods_str}; "
                            f"all {n_after} features met the retention threshold and were included in modeling."
                        )
                    else:
                        sections.append(
                            f" Feature selection was performed; all {n_after} features were retained."
                        )
                else:
                    if methods_str:
                        sections.append(
                            f" Feature selection using {methods_str} reduced the feature set "
                            f"from {n_before} to {n_after} predictors."
                        )
                    else:
                        sections.append(
                            f" Feature selection reduced the feature set from {n_before} to {n_after} predictors."
                        )
                feature_selection_logged = True
                break  # Use first valid entry only
    
    # Only fall back to parameter if no log entries exist
    if not feature_selection_logged and feature_selection_method:
        sections.append(f" Feature selection was performed using {feature_selection_method}.")

    # Feature Engineering (if applied)
    try:
        import streamlit as st
        feature_engineering_applied = st.session_state.get('feature_engineering_applied', False)
        if feature_engineering_applied:
            sections.append("\n\n### Feature Engineering\n")
            engineering_log = st.session_state.get('engineering_log', [])
            engineered_feature_names = st.session_state.get('engineered_feature_names', [])
            
            sections.append("Feature engineering was performed prior to feature selection. ")
            
            # List techniques from engineering log
            if engineering_log:
                sections.append("The following transformations were applied: ")
                techniques = []
                for log_entry in engineering_log:
                    # Parse entries like "Polynomial degree 2: +45 features"
                    if ':' in log_entry:
                        technique, detail = log_entry.split(':', 1)
                        techniques.append(f"{technique.strip()} ({detail.strip()})")
                    else:
                        techniques.append(log_entry)
                sections.append("; ".join(techniques) + ". ")
            
            # Total features created
            n_engineered = len(engineered_feature_names) if engineered_feature_names else 0
            if n_engineered > 0:
                sections.append(
                    f"In total, {n_engineered} engineered features were created and included in subsequent feature selection."
                )
            else:
                sections.append(
                    "The engineered features were included in subsequent feature selection and model training."
                )
    except ImportError:
        # Not in a Streamlit context, skip feature engineering section
        pass

    # Missing data & Preprocessing (combined when preprocessing_summary available)
    _preproc = preprocessing_config or {}
    _has_summary = bool(_preproc.get("missing_data"))

    sections.append("\n\n### Missing Data\n")
    if _has_summary:
        _md = _preproc["missing_data"]
        sections.append(f"Missing numeric values were handled using {_md['label']}.")
        if _md.get("indicators"):
            sections.append(
                " Binary indicator variables were added for features with missing values "
                "to allow models to leverage missingness patterns."
            )
    elif missing_data_strategy:
        sections.append(f"Missing data were handled using {missing_data_strategy}.")
    else:
        sections.append("[PLACEHOLDER: Describe how missing data were handled.]")

    sections.append("\n\n### Data Preprocessing\n")
    
    # Check logged preprocessing data
    if 'Preprocessing' in logged_steps:
        preprocessing_logged = False
        for entry in logged_steps['Preprocessing']:
            details = entry.get('details', {})
            # Use logged preprocessing details if available
            if details:
                sentences = []
                
                # Skip imputation here (already covered in Missing Data section)
                
                if details.get('scaling') and details['scaling'] != 'none':
                    scale_labels = {
                        "standard": "z-score standardization",
                        "robust": "robust scaling",
                        "minmax": "min-max normalization"
                    }
                    scale_label = scale_labels.get(details['scaling'], details['scaling'])
                    sentences.append(f"Continuous features were scaled using {scale_label}.")
                
                if details.get('encoding'):
                    enc_labels = {
                        "onehot": "one-hot encoding",
                        "target": "target encoding",
                        "ordinal": "ordinal encoding"
                    }
                    enc_label = enc_labels.get(details['encoding'], details['encoding'])
                    sentences.append(f"Categorical variables were encoded using {enc_label}.")
                
                if details.get('outlier_handling') and details['outlier_handling'] != 'none':
                    outlier_labels = {
                        "percentile": "percentile-based winsorization",
                        "mad": "MAD-based outlier clipping",
                        "iqr": "IQR-based outlier removal"
                    }
                    outlier_label = outlier_labels.get(details['outlier_handling'], details['outlier_handling'])
                    sentences.append(f"Outliers were addressed using {outlier_label}.")
                
                if sentences:
                    sections.append(" ".join(sentences))
                    preprocessing_logged = True
                    break  # Use first logged preprocessing entry
        
        if not preprocessing_logged:
            sections.append("No additional preprocessing transformations were applied beyond imputation.")
    elif _has_summary:
        sentences = []
        _sc = _preproc.get("scaling", {})
        if _sc.get("method", "none") != "none":
            n_num = _preproc.get("n_numeric", 0)
            num_note = f" ({n_num} features)" if n_num else ""
            sentences.append(f"Continuous predictors{num_note} were scaled using {_sc['label']}.")
        _tr = _preproc.get("transforms", {})
        if _tr.get("method", "none") != "none":
            sentences.append(f"A {_tr['label']} was applied to reduce skewness.")
        _ol = _preproc.get("outliers", {})
        if _ol.get("method", "none") != "none":
            sentences.append(f"Outliers were addressed via {_ol['label']}.")
        _en = _preproc.get("encoding", {})
        if _en.get("method"):
            n_cat = _preproc.get("n_categorical", 0)
            cat_note = f" ({n_cat} variables)" if n_cat else ""
            sentences.append(f"Categorical predictors{cat_note} were transformed using {_en['label']}.")
        if sentences:
            sections.append(" ".join(sentences))
        else:
            sections.append("No additional preprocessing transformations were applied beyond imputation.")
    elif _preproc:
        steps = []
        scaling = _preproc.get("numeric_scaling", "standard")
        if scaling != "none":
            steps.append(f"numeric features were {scaling}-scaled")
        # Don't mention imputation here (already in Missing Data section)
        cat_enc = _preproc.get("categorical_encoding", "onehot")
        if cat_enc:
            steps.append(f"categorical variables were encoded using {cat_enc} encoding")
        if steps:
            sections.append(f"Preprocessing included: {'; '.join(steps)}.")
    else:
        sections.append("[PLACEHOLDER: Describe preprocessing steps.]")

    # Model development
    sections.append("\n\n### Model Development\n")
    
    # Use logged Model Training data if available
    if 'Model Training' in logged_steps:
        for entry in logged_steps['Model Training']:
            details = entry.get('details', {})
            models_trained = details.get('models', [])
            best_model = details.get('best_model')
            use_cv = details.get('use_cv', False)
            cv_folds_logged = details.get('cv_folds')
            hyperopt = details.get('hyperparameter_optimization', False)
            
            if models_trained:
                models_str = ', '.join(m.upper() for m in models_trained)
                sections.append(f"The following models were developed and compared: {models_str}.")
            
            if hyperopt:
                sections.append(" Hyperparameter optimization was performed using Optuna with 30 trials per model.")
            
            # Don't use logged best_model here - we'll determine it from actual results below
            
            break  # Use first logged training entry
    else:
        model_names = list(model_configs.keys()) if model_configs else []
        if model_names:
            sections.append(
                f"The following models were developed and compared: {', '.join(n.upper() for n in model_names)}."
            )
    
    sections.append(
        f" Data were split into training ({n_train:,}, {n_train/n_total*100:.0f}%), "
        f"validation ({n_val:,}, {n_val/n_total*100:.0f}%), and "
        f"test ({n_test:,}, {n_test/n_total*100:.0f}%) sets."
    )
    
    # Check for CV from logged data or parameters
    cv_to_use = None
    if 'Model Training' in logged_steps:
        for entry in logged_steps['Model Training']:
            details = entry.get('details', {})
            if details.get('use_cv'):
                cv_to_use = details.get('cv_folds', 5)
                break
    if cv_to_use is None and cv_folds:
        cv_to_use = cv_folds
    
    if cv_to_use:
        sections.append(f" {cv_to_use}-fold cross-validation was used for internal validation.")

    # Performance evaluation
    sections.append("\n\n### Performance Evaluation\n")
    sections.append(
        f"Model performance was evaluated using {', '.join(metrics_used)}. "
        f"95% confidence intervals were computed using 1,000 BCa bootstrap resamples. "
    )
    if external_validation:
        sections.append(
            "External validation was performed on an independent dataset. "
        )

    # Explainability - check both parameter and session state
    explainability_to_mention = set(explainability_methods or [])
    
    # Check session state for SHAP and Bland-Altman
    try:
        import streamlit as st
        if st.session_state.get('shap_results'):
            explainability_to_mention.add('shap')
        if st.session_state.get('bland_altman_results'):
            explainability_to_mention.add('bland_altman')
    except ImportError:
        pass
    
    if explainability_to_mention:
        sections.append("\n\n### Model Interpretability\n")
        method_descriptions = {
            "permutation_importance": "Permutation importance was computed to assess feature contributions by measuring the decrease in model performance when each feature was randomly shuffled.",
            "shap": "SHapley Additive exPlanations (SHAP) values were computed to quantify the contribution of each feature to individual predictions.",
            "partial_dependence": "Partial dependence plots were generated to visualize the marginal effect of individual features on the predicted outcome.",
            "calibration": "Model calibration was assessed using reliability diagrams, Brier score, and expected calibration error (ECE).",
            "subgroup": "Subgroup analysis was performed to evaluate model performance across clinically relevant subgroups.",
            "decision_curve": "Decision curve analysis was performed to assess the clinical utility of the model at various probability thresholds.",
            "bland_altman": "Bland-Altman analysis was performed to assess agreement between model predictions.",
        }
        for method in sorted(explainability_to_mention):
            desc = method_descriptions.get(method, f"{method} analysis was performed.")
            sections.append(f"{desc} ")

    # Software - get actual versions
    try:
        import sklearn
        import numpy
        import pandas
        import scipy
        
        py_version = sys.version.split()[0]
        sklearn_version = sklearn.__version__
        numpy_version = numpy.__version__
        pandas_version = pandas.__version__
        scipy_version = scipy.__version__
        
        sections.append("\n\n### Software\n")
        sections.append(
            f"All analyses were performed using Python (version {py_version}) with "
            f"scikit-learn ({sklearn_version}), NumPy ({numpy_version}), "
            f"pandas ({pandas_version}), and SciPy ({scipy_version}). "
            f"Random seed was set to {random_seed} for reproducibility."
        )
    except Exception:
        # Fallback if imports fail
        sections.append("\n\n### Software\n")
        sections.append(
            f"All analyses were performed using Python with "
            f"scikit-learn, NumPy, pandas, and SciPy. "
            f"Random seed was set to {random_seed} for reproducibility."
        )

    # Methodological Considerations
    sections.append("\n\n### Methodological Considerations\n")
    sections.append(
        "The following methodological choices and their implications are documented "
        "for transparency and reproducibility.\n\n"
    )

    # 1. CV on pre-transformed data
    if cv_folds or cv_to_use:
        _has_pca = False
        _has_feature_selection = feature_selection_logged or feature_selection_method is not None
        try:
            import streamlit as st
            _preproc_configs = st.session_state.get('preprocessing_config_by_model', {})
            for _mc in _preproc_configs.values():
                if isinstance(_mc, dict) and _mc.get('use_pca'):
                    _has_pca = True
                    break
        except ImportError:
            pass

        cv_num = cv_to_use if cv_to_use else cv_folds
        sections.append(
            f"**Cross-validation and preprocessing:** {cv_num}-fold cross-validation was performed "
            f"on data that had already been preprocessed using the full training set. "
            f"In a strict nested cross-validation framework, preprocessing would be re-fit within "
            f"each fold to avoid information leakage. For scale-invariant models (tree-based ensembles), "
            f"this has no practical effect. For scale-sensitive models (linear, SVM, neural networks), "
            f"the impact of this choice on reported cross-validation metrics is expected to be minimal "
            f"for imputation and scaling operations"
        )
        if _has_pca or _has_feature_selection:
            sections.append(
                f", though it may introduce optimistic bias for dimensionality reduction "
                f"{'(PCA was applied)' if _has_pca else ''}"
                f"{' (feature selection was applied)' if _has_feature_selection else ''}"
            )
        sections.append(
            ". Held-out test set performance, which uses a strict train/test separation "
            "for preprocessing, remains unaffected by this consideration.\n\n"
        )

    # 2. Feature dropout methodology
    try:
        import streamlit as st
        _has_dropout = st.session_state.get('sensitivity_dropout_results') is not None
    except ImportError:
        _has_dropout = False

    if _has_dropout:
        sections.append(
            "**Feature dropout analysis:** When assessing the impact of individual feature removal, "
            "models were retrained using median imputation only (without the full preprocessing pipeline) "
            "due to the complexity of dynamically reconstructing column-specific pipelines for each "
            "feature permutation. This simplification is inconsequential for tree-based models, which are "
            "invariant to monotonic feature transformations. For linear and neural network models, "
            "dropout impact estimates should be interpreted with caution, as the absence of scaling "
            "may confound the effect of feature removal. Permutation importance and SHAP values, "
            "which operate on the fully preprocessed data, provide more reliable feature importance "
            "estimates for these model types.\n\n"
        )

    # 3. Feature engineering transform detection
    try:
        import streamlit as st
        _has_eng = st.session_state.get('feature_engineering_applied', False)
        _transform_map = st.session_state.get('engineered_feature_transforms', {})
    except ImportError:
        _has_eng = False
        _transform_map = {}

    if _has_eng and _transform_map:
        sections.append(
            "**Engineered feature handling in preprocessing:** To prevent double-transformation "
            "(e.g., applying a log transform to an already log-transformed feature), engineered features "
            "were identified by naming convention and automatically excluded from redundant preprocessing "
            "transforms. These features received imputation and scaling only. "
            "This detection relies on standard naming prefixes (e.g., `log_`, `sqrt_`, `PCA_`); "
            "features with non-standard names were treated as untransformed and received full preprocessing.\n\n"
        )

    # 4. Methodology audit trail
    if logged_steps:
        step_names = sorted(logged_steps.keys())
        sections.append(
            f"**Reproducibility:** All data processing decisions were recorded in an automated "
            f"methodology log covering {len(step_names)} analysis phases "
            f"({', '.join(step_names)}). "
            f"This log captures the specific parameters used at each step and can be "
            f"exported for full reproducibility.\n\n"
        )

    # ── Results Section (if actual results provided) ──
    if selected_model_results:
        sections.append("\n\n---\n\n## Results (Draft)\n")
        sections.append(f"\n### Model Performance\n")

        # Determine best model from actual metrics
        actual_best = _determine_best_model(selected_model_results, task_type)
        model_to_highlight = actual_best or best_model_name
        
        if model_to_highlight:
            sections.append(f"The best-performing model was **{model_to_highlight.upper()}**. ")

        sections.append("Table X presents the performance of all evaluated models on the held-out test set.\n\n")

        # Build a text table
        for name, res in selected_model_results.items():
            metrics = res.get("metrics", {})
            cis = bootstrap_results.get(name, {}) if bootstrap_results else {}
            metric_strs = []
            for m, v in metrics.items():
                ci = cis.get(m)
                if ci and hasattr(ci, 'ci_lower'):
                    metric_strs.append(f"{m}: {v:.4f} (95% CI: {ci.ci_lower:.4f}–{ci.ci_upper:.4f})")
                else:
                    metric_strs.append(f"{m}: {v:.4f}")
            sections.append(f"**{name.upper()}:** {'; '.join(metric_strs)}\n\n")

        # Calibration
        if calibration_results:
            sections.append("\n### Calibration\n")
            for model_name, cal in calibration_results.items():
                if hasattr(cal, 'brier_score') and cal.brier_score is not None:
                    sections.append(
                        f"**{model_name}:** Brier score = {cal.brier_score:.4f}, "
                        f"ECE = {cal.ece:.4f}.\n\n"
                    )
                elif hasattr(cal, 'calibration_slope') and cal.calibration_slope is not None:
                    sections.append(
                        f"**{model_name}:** Calibration slope = {cal.calibration_slope:.3f}, "
                        f"intercept = {cal.calibration_intercept:.3f}.\n\n"
                    )

    return "".join(sections)


# ============================================================================
# Flow Diagram (CONSORT-style)
# ============================================================================

def generate_flow_diagram_mermaid(
    n_total: int,
    n_excluded: int = 0,
    exclusion_reasons: Optional[Dict[str, int]] = None,
    n_missing_target: int = 0,
    n_analyzed: int = 0,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
) -> str:
    """Generate a Mermaid flowchart for sample selection.

    Returns Mermaid diagram text that can be rendered in Streamlit
    or exported to SVG/PNG.
    """
    lines = ["graph TD"]
    lines.append(f'    A["Total records<br/>N = {n_total:,}"]')

    if n_excluded > 0 or exclusion_reasons:
        if exclusion_reasons:
            reasons = "<br/>".join(f"• {r}: n={n}" for r, n in exclusion_reasons.items())
            lines.append(f'    B["Excluded<br/>n = {n_excluded:,}<br/>{reasons}"]')
        else:
            lines.append(f'    B["Excluded<br/>n = {n_excluded:,}"]')
        lines.append("    A --> B")
        n_after = n_total - n_excluded
        lines.append(f'    C["After exclusions<br/>n = {n_after:,}"]')
        lines.append("    A --> C")
    else:
        lines.append(f'    C["All records<br/>n = {n_total:,}"]')
        lines.append("    A --> C")

    if n_missing_target > 0:
        lines.append(f'    D["Missing target<br/>n = {n_missing_target:,}"]')
        lines.append("    C --> D")
        n_analyzed = n_analyzed or (n_total - n_excluded - n_missing_target)
        lines.append(f'    E["Analyzed<br/>n = {n_analyzed:,}"]')
        lines.append("    C --> E")
    else:
        n_analyzed = n_analyzed or (n_total - n_excluded)
        lines.append(f'    E["Analyzed<br/>n = {n_analyzed:,}"]')
        lines.append("    C --> E")

    if n_train > 0:
        lines.append(f'    F["Training set<br/>n = {n_train:,}"]')
        lines.append(f'    G["Validation set<br/>n = {n_val:,}"]')
        lines.append(f'    H["Test set<br/>n = {n_test:,}"]')
        lines.append("    E --> F")
        lines.append("    E --> G")
        lines.append("    E --> H")

    return "\n".join(lines)


# ============================================================================
# Subgroup Analysis
# ============================================================================

def subgroup_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    task_type: str = "regression",
    metric_fn=None,
    metric_name: str = "Metric",
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute metrics stratified by subgroup with bootstrap CIs.

    Args:
        y_true: True values
        y_pred: Predicted values
        subgroup_labels: Group labels for each sample
        task_type: 'regression' or 'classification'
        metric_fn: Custom metric function (default: RMSE for regression, accuracy for classification)
        metric_name: Name of the metric
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed

    Returns:
        DataFrame with subgroup, n, metric, CI_lower, CI_upper
    """
    from ml.bootstrap import bootstrap_metric

    if metric_fn is None:
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error
            metric_fn = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
            metric_name = "RMSE"
        else:
            from sklearn.metrics import accuracy_score
            metric_fn = accuracy_score
            metric_name = "Accuracy"

    groups = np.unique(subgroup_labels)
    rows = []

    # Overall
    overall_result = bootstrap_metric(
        y_true, y_pred, metric_fn,
        n_resamples=n_bootstrap, metric_name=metric_name, random_state=random_state,
    )
    rows.append({
        "Subgroup": "Overall",
        "N": len(y_true),
        metric_name: f"{overall_result.estimate:.4f}",
        "95% CI": f"[{overall_result.ci_lower:.4f}, {overall_result.ci_upper:.4f}]",
        "_estimate": overall_result.estimate,
        "_ci_lower": overall_result.ci_lower,
        "_ci_upper": overall_result.ci_upper,
    })

    for g in groups:
        mask = subgroup_labels == g
        if mask.sum() < 5:
            continue
        result = bootstrap_metric(
            y_true[mask], y_pred[mask], metric_fn,
            n_resamples=n_bootstrap, metric_name=metric_name, random_state=random_state,
        )
        rows.append({
            "Subgroup": str(g),
            "N": int(mask.sum()),
            metric_name: f"{result.estimate:.4f}",
            "95% CI": f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
            "_estimate": result.estimate,
            "_ci_lower": result.ci_lower,
            "_ci_upper": result.ci_upper,
        })

    return pd.DataFrame(rows)


def plot_forest_subgroups(subgroup_df: pd.DataFrame, metric_name: str = "Metric"):
    """Generate a forest plot for subgroup analysis.

    Returns a Plotly figure.
    """
    import plotly.graph_objects as go

    df = subgroup_df.copy()
    df = df.iloc[::-1]  # Reverse for plotting

    fig = go.Figure()

    # Error bars
    fig.add_trace(go.Scatter(
        x=df["_estimate"],
        y=df["Subgroup"],
        mode="markers",
        marker=dict(size=10, color="steelblue"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=df["_ci_upper"] - df["_estimate"],
            arrayminus=df["_estimate"] - df["_ci_lower"],
        ),
        name=metric_name,
    ))

    # Overall reference line
    overall = df.loc[df["Subgroup"] == "Overall", "_estimate"]
    if len(overall) > 0:
        fig.add_vline(x=overall.iloc[0], line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Subgroup Analysis — {metric_name}",
        xaxis_title=metric_name,
        yaxis_title="",
        height=max(300, 60 * len(df)),
    )

    return fig
