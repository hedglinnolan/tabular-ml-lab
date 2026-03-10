"""
Publication engine: methods section generator, flow diagrams, TRIPOD tracking.

Generates publication-ready text, figures, and compliance checklists.
"""
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

    if feature_selection_method:
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
    if _has_summary:
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
            sections.append("No additional preprocessing transformations were applied.")
    elif _preproc:
        steps = []
        scaling = _preproc.get("numeric_scaling", "standard")
        if scaling != "none":
            steps.append(f"numeric features were {scaling}-scaled")
        imputation = _preproc.get("numeric_imputation", "median")
        steps.append(f"missing numeric values were imputed using the {imputation}")
        cat_enc = _preproc.get("categorical_encoding", "onehot")
        if cat_enc:
            steps.append(f"categorical variables were encoded using {cat_enc} encoding")
        if steps:
            sections.append(f"Preprocessing included: {'; '.join(steps)}.")
    else:
        sections.append("[PLACEHOLDER: Describe preprocessing steps.]")

    # Model development
    sections.append("\n\n### Model Development\n")
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
    if cv_folds:
        sections.append(f" {cv_folds}-fold cross-validation was used for internal validation.")

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

    # Explainability
    if explainability_methods:
        sections.append("\n\n### Model Interpretability\n")
        method_descriptions = {
            "permutation_importance": "Permutation importance was computed to assess feature contributions by measuring the decrease in model performance when each feature was randomly shuffled.",
            "shap": "SHapley Additive exPlanations (SHAP) values were computed to quantify the contribution of each feature to individual predictions.",
            "partial_dependence": "Partial dependence plots were generated to visualize the marginal effect of individual features on the predicted outcome.",
            "calibration": "Model calibration was assessed using reliability diagrams, Brier score, and expected calibration error (ECE).",
            "subgroup": "Subgroup analysis was performed to evaluate model performance across clinically relevant subgroups.",
            "decision_curve": "Decision curve analysis was performed to assess the clinical utility of the model at various probability thresholds.",
        }
        for method in explainability_methods:
            desc = method_descriptions.get(method, f"{method} analysis was performed.")
            sections.append(f"{desc} ")

    # Software
    sections.append("\n\n### Software\n")
    sections.append(
        f"All analyses were performed using Python (version 3.x) with "
        f"scikit-learn, NumPy, pandas, and SciPy. "
        f"Random seed was set to {random_seed} for reproducibility. "
        f"[PLACEHOLDER: Add specific software versions from the reproducibility manifest.]"
    )

    # ── Results Section (if actual results provided) ──
    if selected_model_results:
        sections.append("\n\n---\n\n## Results (Draft)\n")
        sections.append(f"\n### Model Performance\n")

        if best_model_name:
            sections.append(f"The best-performing model was **{best_model_name.upper()}**. ")

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
