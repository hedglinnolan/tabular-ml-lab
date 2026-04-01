"""NarrativeEngine — End-to-end manuscript narrative pipeline.

Single entry point that reads from WorkflowProvenance + InsightLedger
and produces a complete, internally consistent manuscript draft.

Architecture:
    WorkflowProvenance (what happened) + InsightLedger (what was considered)
        ↓
    NarrativeEngine.generate() → ManuscriptDraft
        ↓
    ManuscriptDraft.to_markdown() / .to_latex()

This replaces the stitched-together approach where Report Export assembled
prose from 100+ scattered session_state reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.workflow_provenance import WorkflowProvenance
from utils.insight_ledger import InsightLedger


# ---------------------------------------------------------------------------
# Manuscript draft
# ---------------------------------------------------------------------------

@dataclass
class ManuscriptDraft:
    """Structured manuscript output with typed sections."""

    # Methods subsections (keyed by IMRAD convention)
    study_design: str = ""
    predictor_variables: str = ""
    missing_data: str = ""
    data_preprocessing: str = ""
    model_development: str = ""
    model_evaluation: str = ""
    sensitivity_analysis: str = ""
    statistical_validation: str = ""

    # Cross-cutting
    data_observations: str = ""  # from InsightLedger (resolved + acknowledged + strengths)
    software_environment: str = ""

    # Metadata
    warnings: List[str] = field(default_factory=list)
    completeness: Dict[str, bool] = field(default_factory=dict)

    @property
    def sections(self) -> Dict[str, str]:
        """All non-empty sections as an ordered dict."""
        ordered = [
            ("Study Design", self.study_design),
            ("Predictor Variables", self.predictor_variables),
            ("Missing Data", self.missing_data),
            ("Data Preprocessing", self.data_preprocessing),
            ("Model Development", self.model_development),
            ("Model Evaluation", self.model_evaluation),
            ("Sensitivity Analysis", self.sensitivity_analysis),
            ("Statistical Validation", self.statistical_validation),
            ("Data Observations", self.data_observations),
            ("Software Environment", self.software_environment),
        ]
        return {k: v for k, v in ordered if v.strip()}

    def to_markdown(self) -> str:
        """Render as markdown with subsection headers."""
        lines = ["## Methods\n"]
        for title, content in self.sections.items():
            lines.append(f"### {title}\n")
            lines.append(content)
            lines.append("")
        if self.warnings:
            lines.append("### Completeness Notes\n")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render as LaTeX subsections."""
        import re
        
        lines = ["\\section{Methods}\n"]
        for title, content in self.sections.items():
            latex_title = title.replace("&", "\\&")
            lines.append(f"\\subsection{{{latex_title}}}")
            lines.append("")
            # Strip markdown headers from content (##, ###, etc.)
            content_cleaned = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            # Clean up resulting multiple blank lines
            content_cleaned = re.sub(r'\n\n\n+', '\n\n', content_cleaned).strip()
            lines.append(content_cleaned)
            lines.append("")
        if self.warnings:
            lines.append("% Completeness warnings:")
            for w in self.warnings:
                lines.append(f"% - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable name mappings
# ---------------------------------------------------------------------------

_MODEL_NAMES: dict = {
    "histgb_reg": "Histogram-based Gradient Boosting (Regression)",
    "histgb_clf": "Histogram-based Gradient Boosting (Classification)",
    "nn": "Neural Network (MLP)",
    "huber": "Huber Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "elasticnet": "Elastic Net",
    "rf": "Random Forest",
    "xgb": "XGBoost (Gradient Boosting)",
    "lgbm": "LightGBM",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "logistic": "Logistic Regression",
    "dt": "Decision Tree",
}

_METRIC_NAMES: dict = {
    "RMSE": "root mean squared error (RMSE)",
    "MAE": "mean absolute error (MAE)",
    "R2": "coefficient of determination (R²)",
    "MedianAE": "median absolute error (MedAE)",
    "Accuracy": "accuracy",
    "F1": "F1 score",
    "AUC": "area under the ROC curve (AUC)",
    "Precision": "precision",
    "Recall": "recall",
}


# ---------------------------------------------------------------------------
# Scale/encoding label maps
# ---------------------------------------------------------------------------

_SCALE_LABELS = {
    "standard": "z-score standardization",
    "robust": "robust scaling (median/IQR)",
    "minmax": "min-max normalization",
    "none": None,
}

_ENC_LABELS = {
    "onehot": "one-hot encoding",
    "target": "target encoding",
    "ordinal": "ordinal encoding",
}

_TRANSFORM_LABELS = {
    "yeo-johnson": "Yeo-Johnson power transform",
    "box-cox": "Box-Cox power transform",
    "log1p": "log(1+x) transform",
    "none": None,
}


# ---------------------------------------------------------------------------
# NarrativeEngine
# ---------------------------------------------------------------------------

class NarrativeEngine:
    """Generates manuscript narrative from WorkflowProvenance + InsightLedger.

    Usage:
        engine = NarrativeEngine(provenance, ledger)
        draft = engine.generate()
        print(draft.to_markdown())
    """

    def __init__(
        self,
        provenance: WorkflowProvenance,
        ledger: Optional[InsightLedger] = None,
    ):
        self.prov = provenance
        self.ledger = ledger
        self.ctx = provenance.get_methods_context()

    def generate(self) -> ManuscriptDraft:
        """Generate a complete manuscript draft from provenance + ledger."""
        draft = ManuscriptDraft()
        draft.completeness = self.prov.get_completeness()

        # Generate each section
        draft.study_design = self._gen_study_design()
        draft.predictor_variables = self._gen_predictor_variables()
        draft.missing_data = self._gen_missing_data()
        draft.data_preprocessing = self._gen_data_preprocessing()
        draft.model_development = self._gen_model_development()
        draft.model_evaluation = self._gen_model_evaluation()
        draft.sensitivity_analysis = self._gen_sensitivity_analysis()
        draft.statistical_validation = self._gen_statistical_validation()
        draft.data_observations = self._gen_data_observations()
        draft.software_environment = self._gen_software_environment()

        # Completeness warnings
        draft.warnings = self._check_completeness()

        return draft

    # -- Section generators --------------------------------------------------

    def _gen_study_design(self) -> str:
        """Study design: task type, sample size, split strategy."""
        parts = []

        task_type = self.ctx.get("task_type", "")
        n_total = self.ctx.get("n_total", 0)
        target = self.ctx.get("target_name", "")

        if task_type and n_total:
            parts.append(
                f"A {task_type} analysis was performed on a dataset of "
                f"{n_total:,} observations."
            )
        if target:
            parts.append(f"The outcome variable was {target}.")

        # Split strategy
        split_prov = self.prov.split
        if split_prov:
            strategy = split_prov.strategy
            n_train = split_prov.train_n
            n_val = split_prov.val_n
            n_test = split_prov.test_n
            seed = split_prov.random_seed

            # Target trimming — mention BEFORE the split description
            if split_prov.target_trim_enabled:
                lo_pct = round(split_prov.target_trim_lower * 100)
                hi_pct = round((1.0 - split_prov.target_trim_upper) * 100)
                if lo_pct > 0 or hi_pct > 0:
                    trim_parts = []
                    if lo_pct > 0:
                        trim_parts.append(f"lower {lo_pct}%")
                    if hi_pct > 0:
                        trim_parts.append(f"upper {hi_pct}%")
                    parts.append(
                        f"Extreme target values were removed prior to splitting "
                        f"(trimmed {' and '.join(trim_parts)} of the target distribution)."
                    )
                else:
                    parts.append(
                        "Target variable trimming was applied prior to splitting."
                    )

            # Split description
            if n_train > 0 and n_test > 0:
                # Recompute percentages from actual n values (not stored pct fields)
                total = n_train + n_val + n_test
                train_pct = round(n_train / total * 100)
                val_pct = round(n_val / total * 100) if n_val else 0
                test_pct = round(n_test / total * 100)

                split_desc = f"{strategy} " if strategy else ""
                if n_val > 0:
                    parts.append(
                        f"Data were partitioned using a {split_desc}split into training "
                        f"(n={n_train:,}, {train_pct}%), validation (n={n_val:,}, {val_pct}%), "
                        f"and test (n={n_test:,}, {test_pct}%) sets (random seed={seed})."
                    )
                else:
                    parts.append(
                        f"Data were partitioned using a {split_desc}split into training "
                        f"(n={n_train:,}, {train_pct}%) and test (n={n_test:,}, {test_pct}%) "
                        f"sets (random seed={seed})."
                    )

            # Target transform
            target_transform = split_prov.target_transform
            if target_transform and target_transform != "none":
                label = _TRANSFORM_LABELS.get(target_transform, target_transform)
                parts.append(
                    f"The target variable was transformed using {label} "
                    f"prior to model training; predictions were back-transformed for evaluation."
                )

        # Data cleaning
        cleaning = self.ctx.get("cleaning_actions", [])
        if cleaning:
            total_removed = sum(
                a.get("rows_before", 0) - a.get("rows_after", 0)
                for a in cleaning
            )
            parts.append(
                f"Prior to analysis, {len(cleaning)} data cleaning operations were performed"
                f"{f', removing {total_removed:,} observations' if total_removed > 0 else ''}."
            )

        return " ".join(parts)

    def _gen_predictor_variables(self) -> str:
        """Predictor variables: feature counts, engineering, selection."""
        parts = []

        n_original = self.ctx.get("n_features_original", 0)
        features = self.ctx.get("features_kept") or self.ctx.get("feature_cols", [])
        n_final = len(features) if features else 0

        # Feature engineering
        transforms = self.ctx.get("engineering_transforms", [])
        n_engineered = self.ctx.get("n_engineered", 0)
        if transforms:
            parts.append(
                f"Feature engineering was performed: {', '.join(transforms)}. "
                f"{n_engineered} engineered features were created."
            )

        # Feature selection
        fs_method = self.ctx.get("fs_method", "")
        n_before_sel = self.ctx.get("n_features_before_selection", 0)
        n_after_sel = self.ctx.get("n_features_after_selection", 0)
        if fs_method:
            if n_before_sel == n_after_sel and n_after_sel > 0:
                parts.append(
                    f"All {n_after_sel} candidate predictors were retained."
                )
            elif n_after_sel > 0:
                parts.append(
                    f"Feature selection was performed using {fs_method}, "
                    f"reducing from {n_before_sel} to {n_after_sel} predictors."
                )

        # Final feature count
        if n_final and n_final <= 15 and features:
            feat_list = ", ".join(features)
            parts.append(f"The final set of predictor variables included: {feat_list}.")
        elif n_final:
            parts.append(
                f"A total of {n_final} predictor variables were included in the final models "
                f"(see Supplementary Table S1 for full list)."
            )
        elif n_original:
            parts.append(f"The analysis began with {n_original} candidate predictors.")

        return " ".join(parts)

    def _gen_missing_data(self) -> str:
        """Missing data handling."""
        pp = self.ctx.get("preprocessing_per_model", {})
        if not pp:
            return ""

        # Get imputation method (should be shared across models)
        methods = set()
        for cfg in pp.values():
            imp = cfg.get("imputation", "")
            if imp:
                methods.add(imp)

        if not methods:
            return ""

        # Include feature-level missing data counts if available from provenance
        n_missing_features = self.ctx.get("n_features_with_missing", 0)
        n_total_features = self.ctx.get("n_features_original", 0) or len(
            self.ctx.get("feature_cols", [])
        )
        missing_pct_str = ""
        if n_missing_features and n_total_features:
            pct = round(n_missing_features / n_total_features * 100)
            missing_pct_str = (
                f" {n_missing_features} of {n_total_features} features "
                f"({pct}%) contained missing values."
            )
        elif n_missing_features:
            missing_pct_str = f" {n_missing_features} features contained missing values."

        if len(methods) == 1:
            method = next(iter(methods))
            return f"Missing values were handled using {method} imputation.{missing_pct_str}"
        else:
            method_list = ", ".join(sorted(methods))
            return (
                f"Missing values were handled using model-specific imputation strategies: "
                f"{method_list}.{missing_pct_str}"
            )

    def _gen_data_preprocessing(self) -> str:
        """Data preprocessing: per-model pipeline description.

        This is the core differentiator — different models may get
        different preprocessing pipelines.
        """
        pp = self.ctx.get("preprocessing_per_model", {})
        differs = self.ctx.get("preprocessing_differs", False)

        if not pp:
            return ""

        parts = []

        if not differs:
            # All models share preprocessing
            cfg = next(iter(pp.values()))
            sents = self._describe_preprocessing(cfg)
            if sents:
                parts.append(f"All models shared identical preprocessing: {'; '.join(sents)}.")
            else:
                parts.append(
                    "No additional preprocessing transformations were applied beyond imputation."
                )
        else:
            # Per-model preprocessing — the key differentiator
            parts.append(
                "Preprocessing was configured independently for each model family "
                "to respect different algorithmic assumptions:"
            )
            for model_key, cfg in pp.items():
                sents = self._describe_preprocessing(cfg)
                model_label = self._model_name(model_key)
                if sents:
                    parts.append(f"**{model_label}**: {'; '.join(sents)}.")
                else:
                    parts.append(
                        f"**{model_label}**: default preprocessing (no additional transformations)."
                    )

        return " ".join(parts)

    def _gen_model_development(self) -> str:
        """Model development: models trained, CV, hyperparameters."""
        parts = []

        models = self.ctx.get("models_trained", [])
        if not models:
            return ""

        models_str = ", ".join(self._model_name(m) for m in models)
        parts.append(
            f"The following model candidates were trained and compared: {models_str}."
        )

        # Cross-validation
        use_cv = self.ctx.get("use_cv", False)
        cv_folds = self.ctx.get("cv_folds")
        if use_cv and cv_folds:
            parts.append(
                f"{cv_folds}-fold cross-validation was used for model evaluation."
            )

        # Hyperparameter optimization
        if self.ctx.get("use_hyperopt"):
            parts.append(
                "Hyperparameter optimization was performed using grid search."
            )

        # Class weighting
        if self.ctx.get("class_weight_balanced"):
            parts.append(
                "To address class imbalance, class_weight='balanced' was applied "
                "to supported classifiers, weighting each class inversely proportional "
                "to its frequency in the training data."
            )

        # Hyperparameters
        hyperparams = self.ctx.get("hyperparameters", {})
        if hyperparams:
            hp_parts = []
            for model_name, params in hyperparams.items():
                if not params:
                    continue
                param_strs = [f"{k}={self._fmt_param(v)}" for k, v in params.items()]
                if param_strs:
                    hp_parts.append(f"{self._model_name(model_name)} ({', '.join(param_strs)})")
            if hp_parts:
                parts.append(
                    f"Key hyperparameters: {'; '.join(hp_parts)}."
                )

        # Primary model selection
        primary = self.ctx.get("primary_model", "")
        criteria = self.ctx.get("selection_criteria", "")
        if primary:
            parts.append(
                f"{self._model_name(primary)} was selected as the primary model"
                f"{f', based on {criteria}' if criteria else ''}."
            )
        elif models:
            parts.append(
                "The model demonstrating the best performance on the primary evaluation "
                "metric was selected for reporting."
            )

        return " ".join(parts)

    def _gen_model_evaluation(self) -> str:
        """Model evaluation: metrics by model."""
        metrics = self.ctx.get("metrics_by_model", {})
        if not metrics:
            return ""

        parts = []
        parts.append("Model performance was evaluated using the following metrics:")

        for model_name, model_metrics in metrics.items():
            if not model_metrics:
                continue
            metric_strs = [
                f"{self._metric_name(k)}={self._fmt_param(v)}"
                for k, v in model_metrics.items()
                if isinstance(v, (int, float))
            ]
            if metric_strs:
                parts.append(f"**{self._model_name(model_name)}**: {', '.join(metric_strs)}.")

        return " ".join(parts)

    def _gen_sensitivity_analysis(self) -> str:
        """Sensitivity analysis."""
        if not self.ctx.get("seed_stability") and not self.ctx.get("feature_dropout"):
            return ""

        parts = []
        if self.ctx.get("seed_stability"):
            parts.append(
                "Seed stability analysis was performed to assess reproducibility of results "
                "across random initializations."
            )
        if self.ctx.get("feature_dropout"):
            parts.append(
                "Feature dropout analysis was conducted to evaluate the robustness of model "
                "performance to individual predictor removal."
            )
        return " ".join(parts)

    def _gen_statistical_validation(self) -> str:
        """Statistical validation tests."""
        tests = self.ctx.get("statistical_tests", [])
        if not tests:
            return ""

        parts = []
        test_names = list(set(t.get("test_name", "") for t in tests if t.get("test_name")))
        if test_names:
            parts.append(
                f"Statistical validation was performed using: {', '.join(test_names)}."
            )

        # Summarize significant findings
        significant = [
            t for t in tests
            if t.get("p_value") is not None and t["p_value"] < 0.05
        ]
        if significant:
            parts.append(
                f"{len(significant)} of {len(tests)} tests yielded statistically "
                f"significant results (p < 0.05)."
            )

        return " ".join(parts)

    def _gen_data_observations(self) -> str:
        """Data observations from InsightLedger (resolved + acknowledged + strengths)."""
        if not self.ledger:
            return ""

        narratives = self.ledger.to_manuscript_narrative()
        if not narratives:
            return ""

        parts = []
        for phase, text in narratives.items():
            if text.strip():
                parts.append(text)

        return " ".join(parts)

    def _gen_software_environment(self) -> str:
        """Software environment boilerplate."""
        return (
            "All analyses were conducted using the Tabular ML Lab, "
            "an open-source research workbench for reproducible machine learning "
            "on tabular data (Python, scikit-learn, Streamlit). "
            "The complete analysis workflow, including all preprocessing configurations, "
            "model hyperparameters, and evaluation metrics, was automatically documented "
            "by the platform's provenance tracking system."
        )

    # -- Helpers --------------------------------------------------------------

    def _describe_preprocessing(self, cfg: Dict[str, Any]) -> List[str]:
        """Build list of preprocessing description sentences for one model config."""
        sents = []
        scaling = cfg.get("scaling", "none")
        sl = _SCALE_LABELS.get(scaling)
        if sl:
            sents.append(f"scaled using {sl}")

        encoding = cfg.get("encoding", "")
        el = _ENC_LABELS.get(encoding)
        if el:
            sents.append(f"categorical variables encoded using {el}")

        outlier = cfg.get("outlier_treatment", "none")
        if outlier and outlier != "none":
            params = cfg.get("outlier_params", {})
            if outlier == "percentile_clip" and params:
                lo = params.get("lower", 5)
                hi = params.get("upper", 95)
                sents.append(f"outliers clipped at {lo}th–{hi}th percentile")
            elif outlier == "iqr":
                mult = params.get("multiplier", 1.5)
                sents.append(f"outliers treated via IQR method (×{mult})")
            else:
                sents.append(f"outlier treatment: {outlier}")

        transform = cfg.get("power_transform", "none")
        tl = _TRANSFORM_LABELS.get(transform)
        if tl:
            sents.append(f"{tl} applied")

        log_t = cfg.get("log_transform", False)
        if log_t:
            sents.append("log transform applied")

        if cfg.get("use_pca"):
            pca_n = cfg.get("pca_n_components")
            if isinstance(pca_n, float) and pca_n < 1:
                sents.append(f"PCA applied (retaining {pca_n*100:.0f}% variance)")
            elif isinstance(pca_n, int):
                sents.append(f"PCA applied ({pca_n} components)")
            else:
                sents.append("PCA dimensionality reduction applied")

        return sents

    def _model_name(self, key: str) -> str:
        """Return human-readable model name, falling back to uppercased key."""
        return _MODEL_NAMES.get(key, key.upper())

    def _metric_name(self, key: str) -> str:
        """Return human-readable metric name, falling back to the key itself."""
        return _METRIC_NAMES.get(key, key)

    def _fmt_param(self, v: Any) -> str:
        """Format a parameter value for publication."""
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            return f"{v:.4g}"
        return str(v)

    def _check_completeness(self) -> List[str]:
        """Check for missing workflow stages and return warnings."""
        warnings = []
        completeness = self.prov.get_completeness()

        if not completeness.get("upload"):
            warnings.append("[PLACEHOLDER] Study design section requires data upload provenance.")
        if not completeness.get("preprocessing"):
            warnings.append("[PLACEHOLDER] Preprocessing section requires pipeline configuration.")
        if not completeness.get("training"):
            warnings.append("[PLACEHOLDER] Model development section requires training provenance.")
        if not completeness.get("split"):
            warnings.append("[PLACEHOLDER] Study design section requires split configuration.")
        if not completeness.get("eda"):
            warnings.append("[NOTE] No EDA analyses were recorded in provenance.")

        return warnings
