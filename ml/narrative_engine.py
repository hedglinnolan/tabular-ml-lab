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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.workflow_provenance import WorkflowProvenance
from utils.insight_ledger import InsightLedger, MODEL_DISPLAY_NAMES


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

    # Results section (IMRAD)
    results: str = ""

    # Discussion section (IMRAD)
    discussion: str = ""

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
            # Methods subsections
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

    @property
    def all_sections(self) -> Dict[str, str]:
        """All sections including Results and Discussion, for full manuscript export."""
        ordered = [
            # Methods subsections
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
            # Results and Discussion
            ("Results", self.results),
            ("Discussion", self.discussion),
        ]
        return {k: v for k, v in ordered if v.strip()}

    def to_markdown(self) -> str:
        """Render as markdown with subsection headers."""
        lines = []
        
        # Methods section with subsections
        lines.append("## Methods\n")
        for title, content in self.sections.items():
            lines.append(f"### {title}\n")
            lines.append(content)
            lines.append("")

        # Results section (top-level)
        if self.results.strip():
            lines.append("## Results\n")
            lines.append(self.results)
            lines.append("")
        
        # Discussion section (top-level)
        if self.discussion.strip():
            lines.append("## Discussion\n")
            lines.append(self.discussion)
            lines.append("")
        
        return "\n".join(lines)

    @staticmethod
    def _md_to_latex(text: str) -> str:
        """Convert markdown formatting to LaTeX equivalents."""
        import re
        # Bold **text** → \textbf{text}
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        # Italic *text* → \textit{text}  (single asterisk, not inside bold)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)
        # Escape special LaTeX chars that might appear in prose (but not our own commands)
        text = text.replace('²', '$^2$')
        text = text.replace('×', '$\\times$')
        text = text.replace('–', '--')
        return text

    def to_latex(self) -> str:
        """Render as LaTeX subsections."""
        import re
        
        lines = []
        
        # Methods section with subsections
        lines.append("\\section{Methods}\n")
        for title, content in self.sections.items():
            latex_title = title.replace("&", "\\&")
            lines.append(f"\\subsection{{{latex_title}}}")
            lines.append("")
            # Strip markdown headers from content (##, ###, etc.)
            content_cleaned = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            content_cleaned = re.sub(r'\n\n\n+', '\n\n', content_cleaned).strip()
            lines.append(self._md_to_latex(content_cleaned))
            lines.append("")
        
        if self.warnings:
            lines.append("% Completeness warnings:")
            for w in self.warnings:
                lines.append(f"% NOTE: {w}")
            lines.append("")
        
        # Results section (top-level)
        if self.results.strip():
            lines.append("\\section{Results}\n")
            content_cleaned = re.sub(r'^#+\s+.*$', '', self.results, flags=re.MULTILINE)
            content_cleaned = re.sub(r'\n\n\n+', '\n\n', content_cleaned).strip()
            lines.append(self._md_to_latex(content_cleaned))
            lines.append("")
        
        # Discussion section (top-level)
        if self.discussion.strip():
            lines.append("\\section{Discussion}\n")
            # Convert ### headings to \subsection instead of stripping
            content_cleaned = self.discussion
            content_cleaned = re.sub(
                r'^###\s+(.+)$',
                lambda m: f"\\subsection{{{m.group(1).replace('&', chr(92) + '&')}}}",
                content_cleaned, flags=re.MULTILINE
            )
            content_cleaned = re.sub(r'^##\s+(.+)$',
                lambda m: f"\\subsection{{{m.group(1)}}}",
                content_cleaned, flags=re.MULTILINE
            )
            content_cleaned = re.sub(r'\n\n\n+', '\n\n', content_cleaned).strip()
            lines.append(self._md_to_latex(content_cleaned))
            lines.append("")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable name mappings
# ---------------------------------------------------------------------------

_MODEL_NAMES: dict = {
    **MODEL_DISPLAY_NAMES,
    "lasso": "LASSO",
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


def _count_phrase(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return a count-aware noun phrase."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def _oxford_join(items: List[str]) -> str:
    """Join a list for manuscript prose."""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _feature_selection_method_label(method: str) -> str:
    """Render feature-selection methods with manuscript-friendly names."""
    labels = {
        "lasso": "LASSO",
        "rfe": "RFE-CV",
        "rfe-cv": "RFE-CV",
        "rfecv": "RFE-CV",
        "univariate": "univariate screening",
        "f_regression": "univariate screening",
        "mutual_info": "mutual information screening",
        "stability": "stability selection",
        "stability_selection": "stability selection",
    }
    key = str(method or "").strip().lower()
    return labels.get(key, str(method or "").strip())


def _polish_data_observations_text(text: str) -> str:
    """Convert ledger-derived workflow notes into smoother manuscript prose."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    replacements = {
        "Workflow Observations: ": "",
        "Preprocessing Rationale: ": "",
        "Missing Data: ": "",
        "Model Development: ": "",
        "Explainability: ": "",
        "Sensitivity Analysis: ": "",
        "Statistical Validation: ": "",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = re.sub(
        r"Large sample-to-feature ratio \(([^)]+)\) — plenty of data relative to complexity\.",
        r"The sample-to-feature ratio was \1, supporting model estimation relative to predictor dimensionality.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Pipelines built for (\d+) model\(s\): ([^.]+)\.",
        r"Preprocessing was tailored across \1 model families (\2).",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Per-model preprocessing pipelines were configured for \d+ model\(s\)\.\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Target is skewed \(([^)]+)\)\.",
        r"The outcome distribution was skewed (\1), which informed preprocessing choices.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Power transform \((.+?)\)\.",
        r"Power transformation was applied selectively by model family (\1).",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Missing values were imputed with column medians in the \d+ model pipelines\.",
        "Missing values were handled with median imputation across model pipelines.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Missing values were imputed with column medians\.",
        "Missing values were handled with median imputation.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?<!\.)\.\.(?!\.)", ".", cleaned)
    return cleaned.strip()


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
        manuscript_context: Optional[Dict[str, Any]] = None,
    ):
        self.prov = provenance
        self.ledger = ledger
        self.ctx = provenance.get_methods_context()
        self.manuscript_context = dict(manuscript_context or {})
        self._apply_manuscript_context()
        # Normalize metric keys to title-case (e.g. "rmse" → "RMSE", "r2" → "R2")
        self._normalize_metrics()

    def generate(self) -> ManuscriptDraft:
        """Generate a complete manuscript draft from provenance + ledger."""
        draft = ManuscriptDraft()
        draft.completeness = self.prov.get_completeness()

        # Generate Methods subsections
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

        # Generate Results and Discussion sections
        draft.results = self._gen_results()
        draft.discussion = self._gen_discussion()

        # Completeness warnings
        draft.warnings = self._check_completeness()

        return draft

    # -- Section generators --------------------------------------------------

    def _apply_manuscript_context(self) -> None:
        """Overlay frozen export facts onto provenance-derived context when provided."""
        if not self.manuscript_context:
            return

        selected_results = self.manuscript_context.get("selected_model_results") or {}
        if selected_results:
            frozen_metrics = {}
            for model_key, payload in selected_results.items():
                if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
                    frozen_metrics[model_key] = dict(payload["metrics"])
                elif isinstance(payload, dict):
                    frozen_metrics[model_key] = dict(payload)
            if frozen_metrics:
                self.ctx["metrics_by_model"] = frozen_metrics
                self.ctx["models_trained"] = list(frozen_metrics.keys())

        manuscript_primary = (
            self.manuscript_context.get("manuscript_primary_model")
        )
        if manuscript_primary:
            self.ctx["primary_model"] = manuscript_primary
        elif "manuscript_primary_model" in self.manuscript_context:
            self.ctx["primary_model"] = ""

        best_model_by_metric = self.manuscript_context.get("best_model_by_metric")
        if best_model_by_metric:
            self.ctx["best_model_by_metric"] = best_model_by_metric

        best_metric_name = self.manuscript_context.get("best_metric_name")
        if best_metric_name:
            self.ctx["best_metric_name"] = best_metric_name
            self.ctx["selection_criteria"] = f"validation {best_metric_name}"

        population_counts = self.manuscript_context.get("population_counts") or {}
        if population_counts:
            upload_total = population_counts.get("upload_total")
            analysis_total = population_counts.get("analysis_total")
            if upload_total is not None:
                self.ctx["n_upload_total"] = upload_total
            if analysis_total is not None:
                self.ctx["n_analysis_total"] = analysis_total
                self.ctx["n_total"] = analysis_total

        feature_counts = self.manuscript_context.get("feature_counts") or {}
        if feature_counts:
            if feature_counts.get("original") is not None:
                self.ctx["n_features_original"] = feature_counts.get("original")
            if feature_counts.get("candidate") is not None:
                self.ctx["n_features_before_selection"] = feature_counts.get("candidate")
            if feature_counts.get("selected") is not None:
                self.ctx["n_features_after_selection"] = feature_counts.get("selected")
            if feature_counts.get("engineered") is not None:
                self.ctx["n_engineered"] = feature_counts.get("engineered")

        frozen_feature_names = self.manuscript_context.get("feature_names_for_manuscript") or []
        if frozen_feature_names:
            self.ctx["feature_cols"] = list(frozen_feature_names)
            self.ctx["features_kept"] = list(frozen_feature_names)

        target_stats = self.manuscript_context.get("target_stats") or {}
        if target_stats:
            self.ctx["target_stats"] = dict(target_stats)

        top_features = self.manuscript_context.get("top_features") or []
        if top_features:
            self.ctx["top_features"] = list(top_features)

    def _gen_study_design(self) -> str:
        """Study design: task type, sample size, split strategy."""
        parts = []

        task_type = self.ctx.get("task_type", "")
        n_total = self.ctx.get("n_analysis_total") or self.ctx.get("n_total", 0)
        n_upload_total = self.ctx.get("n_upload_total", 0)
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
            n_analysis_total = n_train + n_val + n_test
            counts_reconciled = False

            # Reconcile upload-vs-analysis population before describing the split.
            if n_upload_total and n_analysis_total and n_upload_total > n_analysis_total:
                lo_pct = round(split_prov.target_trim_lower * 100)
                hi_pct = round((1.0 - split_prov.target_trim_upper) * 100)
                if split_prov.target_trim_enabled and (lo_pct > 0 or hi_pct > 0):
                    trim_parts = []
                    if lo_pct > 0:
                        trim_parts.append(f"the lower {lo_pct}%")
                    if hi_pct > 0:
                        trim_parts.append(f"the upper {hi_pct}%")
                    parts.append(
                        f"Of {n_upload_total:,} available observations, "
                        f"{n_analysis_total:,} remained for analysis after trimming "
                        f"{' and '.join(trim_parts)} of the target distribution prior to splitting."
                    )
                elif split_prov.target_trim_enabled:
                    parts.append(
                        f"Of {n_upload_total:,} available observations, "
                        f"{n_analysis_total:,} remained for analysis after target trimming "
                        f"was applied prior to splitting."
                    )
                else:
                    parts.append(
                        f"Of {n_upload_total:,} available observations, "
                        f"{n_analysis_total:,} remained for analysis after exclusion criteria "
                        f"were applied prior to splitting."
                    )
                counts_reconciled = True

            # Target trimming — mention BEFORE the split description when counts are unchanged.
            if split_prov.target_trim_enabled and not counts_reconciled:
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
        n_before_sel = self.ctx.get("n_features_before_selection", 0)
        n_after_sel = self.ctx.get("n_features_after_selection", 0)
        n_engineered = self.ctx.get("n_engineered", 0)
        engineered_candidate_count = (n_original + n_engineered) if n_original else 0
        candidate_count = max(n_before_sel or 0, engineered_candidate_count or 0)
        if not candidate_count:
            candidate_count = engineered_candidate_count or n_before_sel or 0
        final_count = n_after_sel or n_final

        # Feature engineering
        transforms = self.ctx.get("engineering_transforms", [])
        if transforms:
            creation_verb = "was" if n_engineered == 1 else "were"
            parts.append(
                f"Feature engineering was performed: {', '.join(transforms)}. "
                f"{_count_phrase(n_engineered, 'engineered feature')} {creation_verb} created."
            )

        fs_method = self.ctx.get("fs_method", "")
        consensus_methods = [
            _feature_selection_method_label(method)
            for method in (self.ctx.get("fs_consensus_methods") or [])
        ]
        consensus_phrase = _oxford_join(consensus_methods)

        # Feature funnel narrative
        if n_original and final_count:
            if candidate_count and candidate_count != n_original and final_count != candidate_count:
                added_count = max(candidate_count - n_original, 0)
                parts.append(
                    f"The raw dataset contained {n_original} predictor variables. "
                    f"Feature engineering added {added_count} predictor variables, yielding {candidate_count} candidate predictors."
                )
            elif candidate_count and candidate_count != n_original:
                parts.append(
                    f"The raw dataset contained {n_original} predictor variables. "
                    f"Feature engineering yielded {candidate_count} candidate predictors, "
                    f"all of which were retained for final modeling."
                )
            elif final_count != n_original:
                parts.append(
                    f"The workflow began with {n_original} predictor variables and retained "
                    f"{final_count} predictors for final modeling."
                )
            else:
                parts.append(f"All {final_count} candidate predictors were retained for final modeling.")

        # Feature selection detail
        if fs_method and final_count:
            if candidate_count == final_count:
                parts.append(
                    (
                        f"Consensus feature selection across {consensus_phrase} retained all {final_count} candidate predictors."
                        if fs_method == "consensus" and consensus_phrase
                        else f"Feature selection was performed using {fs_method}, and all {final_count} candidate predictors were retained."
                    )
                )
            elif candidate_count:
                parts.append(
                    (
                        f"Consensus feature selection across {consensus_phrase} reduced the candidate set from {candidate_count} to {final_count} predictors for final modeling."
                        if fs_method == "consensus" and consensus_phrase
                        else f"Feature selection was performed using {fs_method}, reducing the feature set from {candidate_count} to {final_count} predictors for final modeling."
                    )
                )

        # Final feature count
        if n_final and n_final <= 15 and features:
            feat_list = ", ".join(features)
            parts.append(f"The final set of predictor variables included: {feat_list}.")
        elif n_final:
            parts.append("The full final predictor list is provided in Supplementary Table S1.")
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

        # Hyperparameters — describe per model with human-readable prose
        hyperparams = self.ctx.get("hyperparameters", {})
        if hyperparams:
            hp_sentences = []
            for model_key, params in hyperparams.items():
                if not params:
                    continue
                model_label = self._model_name(model_key)
                hp_desc = self._describe_hyperparameters(model_key, params)
                if hp_desc:
                    hp_sentences.append(f"{model_label} was trained with {hp_desc}.")
            if hp_sentences:
                parts.extend(hp_sentences)

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

        # Primary model selection
        primary = self.ctx.get("primary_model", "")
        best_by_metric = self.ctx.get("best_model_by_metric", "")
        criteria = self.ctx.get("selection_criteria", "")
        if primary:
            parts.append(
                f"{self._model_name(primary)} was selected as the primary model"
                f"{f', based on {criteria}' if criteria else ''}."
            )
        elif best_by_metric:
            metric_phrase = criteria or (
                f"validation {self.ctx.get('best_metric_name')}"
                if self.ctx.get('best_metric_name') else ""
            )
            parts.append(
                f"{self._model_name(best_by_metric)} achieved the best held-out performance"
                f"{f' on {metric_phrase}' if metric_phrase else ''}, "
                "but no manuscript-primary model was explicitly selected."
            )
        elif models:
            parts.append(
                "The model demonstrating the best performance on the primary evaluation "
                "metric was selected for reporting."
            )

        return " ".join(parts)

    def _gen_model_evaluation(self) -> str:
        """Model evaluation: metrics by model, including confidence intervals when available."""
        metrics = self.ctx.get("metrics_by_model", {})
        if not metrics:
            return ""

        parts = []
        parts.append("Model performance was evaluated using the following metrics:")

        for model_name, model_metrics in metrics.items():
            if not model_metrics:
                continue
            
            metric_strs = []
            for k, v in model_metrics.items():
                # Skip CI keys (they're handled with their base metric)
                if k.endswith("_ci_lower") or k.endswith("_ci_upper"):
                    continue
                
                if isinstance(v, (int, float)):
                    # Check for corresponding CI bounds
                    ci_lower_key = f"{k}_ci_lower"
                    ci_upper_key = f"{k}_ci_upper"
                    ci_lower = model_metrics.get(ci_lower_key)
                    ci_upper = model_metrics.get(ci_upper_key)
                    
                    if ci_lower is not None and ci_upper is not None:
                        # Format with CI
                        metric_strs.append(
                            f"{self._metric_name(k)}={self._fmt_param(v)} "
                            f"(95% CI: {self._fmt_param(ci_lower)}–{self._fmt_param(ci_upper)})"
                        )
                    else:
                        # Format without CI
                        metric_strs.append(f"{self._metric_name(k)}={self._fmt_param(v)}")
            
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
        """Methods-facing data observations from resolved InsightLedger actions."""
        if not self.ledger:
            return ""

        narratives = self.ledger.to_manuscript_narrative()
        if not narratives:
            return ""

        parts = []
        for phase, text in narratives.items():
            if text.strip():
                polished = _polish_data_observations_text(text)
                if polished:
                    parts.append(polished)

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

    def _gen_results(self) -> str:
        """Results section: best model, performance comparison, feature importance.
        
        Reports what happened. Does NOT interpret why.
        """
        parts = []
        
        models = self.ctx.get("models_trained", [])
        if not models:
            return ""
        
        metrics = self.ctx.get("metrics_by_model", {})
        task_type = self.ctx.get("task_type", "")
        primary_model = self.ctx.get("primary_model", "")
        
        # Determine best model if not specified
        if not primary_model and metrics:
            if task_type == "regression":
                # Lower RMSE is better
                best_rmse = float("inf")
                for m, m_metrics in metrics.items():
                    rmse = m_metrics.get("RMSE", float("inf"))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        primary_model = m
            elif task_type == "classification":
                # Higher accuracy/F1 is better
                best_f1 = 0.0
                for m, m_metrics in metrics.items():
                    f1 = m_metrics.get("F1", m_metrics.get("Accuracy", 0.0))
                    if f1 > best_f1:
                        best_f1 = f1
                        primary_model = m
        
        # Report best model and key metrics
        if primary_model and metrics.get(primary_model):
            best_metrics = metrics[primary_model]
            model_label = self._model_name(primary_model)
            
            if task_type == "regression":
                r2 = best_metrics.get("R2")
                rmse = best_metrics.get("RMSE")
                r2_ci_lo = best_metrics.get("R2_ci_lower")
                r2_ci_hi = best_metrics.get("R2_ci_upper")
                rmse_ci_lo = best_metrics.get("RMSE_ci_lower")
                rmse_ci_hi = best_metrics.get("RMSE_ci_upper")
                
                parts.append(f"{model_label} demonstrated the best overall performance.")
                
                metric_strs = []
                if r2 is not None:
                    if r2_ci_lo is not None and r2_ci_hi is not None:
                        metric_strs.append(
                            f"R² = {self._fmt_param(r2)} "
                            f"(95% CI: {self._fmt_param(r2_ci_lo)}–{self._fmt_param(r2_ci_hi)})"
                        )
                    else:
                        metric_strs.append(f"R² = {self._fmt_param(r2)}")
                
                if rmse is not None:
                    if rmse_ci_lo is not None and rmse_ci_hi is not None:
                        metric_strs.append(
                            f"RMSE = {self._fmt_param(rmse)} "
                            f"(95% CI: {self._fmt_param(rmse_ci_lo)}–{self._fmt_param(rmse_ci_hi)})"
                        )
                    else:
                        metric_strs.append(f"RMSE = {self._fmt_param(rmse)}")
                
                if metric_strs:
                    parts.append(", ".join(metric_strs) + ".")
            
            elif task_type == "classification":
                acc = best_metrics.get("Accuracy")
                f1 = best_metrics.get("F1")
                auc = best_metrics.get("AUC")
                
                parts.append(f"{model_label} demonstrated the best overall performance.")
                
                metric_strs = []
                for metric_key, metric_val in [("Accuracy", acc), ("F1", f1), ("AUC", auc)]:
                    if metric_val is not None:
                        ci_lo = best_metrics.get(f"{metric_key}_ci_lower")
                        ci_hi = best_metrics.get(f"{metric_key}_ci_upper")
                        if ci_lo is not None and ci_hi is not None:
                            metric_strs.append(
                                f"{self._metric_name(metric_key)} = {self._fmt_param(metric_val)} "
                                f"(95% CI: {self._fmt_param(ci_lo)}–{self._fmt_param(ci_hi)})"
                            )
                        else:
                            metric_strs.append(f"{self._metric_name(metric_key)} = {self._fmt_param(metric_val)}")
                
                if metric_strs:
                    parts.append(", ".join(metric_strs) + ".")
        
        # Comparative table/paragraph for all models
        if len(models) > 1 and metrics:
            parts.append("Performance across all candidate models is summarized in Table 1.")
            # Optionally include a brief sentence about ranking
            if task_type == "regression":
                # Sort by RMSE (lower is better)
                sorted_models = sorted(
                    [(m, metrics[m].get("RMSE", float("inf"))) for m in models if m in metrics],
                    key=lambda x: x[1]
                )
                if len(sorted_models) >= 2:
                    best_name = self._model_name(sorted_models[0][0])
                    worst_name = self._model_name(sorted_models[-1][0])
                    parts.append(
                        f"{best_name} achieved the lowest RMSE, "
                        f"while {worst_name} exhibited the highest prediction error."
                    )
            elif task_type == "classification":
                # Sort by F1 or Accuracy (higher is better)
                sorted_models = sorted(
                    [(m, metrics[m].get("F1", metrics[m].get("Accuracy", 0))) for m in models if m in metrics],
                    key=lambda x: x[1],
                    reverse=True
                )
                if len(sorted_models) >= 2:
                    best_name = self._model_name(sorted_models[0][0])
                    worst_name = self._model_name(sorted_models[-1][0])
                    parts.append(
                        f"{best_name} achieved the highest F1 score, "
                        f"while {worst_name} demonstrated the lowest classification performance."
                    )
        
        # Feature importance findings (if available from session_state or ledger)
        # For now, placeholder — feature importance would need to be in provenance
        # or passed via an extended context
        parts.append(
            "[Feature importance analysis pending — requires explainability provenance integration.]"
        )
        
        # Note if complex models didn't beat simple ones (this is a finding, not a failure)
        if len(models) >= 2 and metrics:
            # Check if simple models (linear, logistic) are competitive with complex ones (RF, XGB, NN)
            simple_keys = {"ridge", "lasso", "elasticnet", "logistic"}
            complex_keys = {"rf", "random_forest", "xgb", "lgbm", "histgb_reg", "histgb_clf", "nn"}
            simple_models = [m for m in models if m.lower() in simple_keys]
            complex_models = [m for m in models if m.lower() in complex_keys]
            
            if simple_models and complex_models:
                if task_type == "regression":
                    simple_rmse = [metrics[m].get("RMSE", float("inf")) for m in simple_models if m in metrics]
                    complex_rmse = [metrics[m].get("RMSE", float("inf")) for m in complex_models if m in metrics]
                    if simple_rmse and complex_rmse:
                        best_simple = min(simple_rmse)
                        best_complex = min(complex_rmse)
                        # If simple model is within 5% of complex, note it
                        if best_simple <= best_complex * 1.05:
                            parts.append(
                                "Regularized linear models achieved performance comparable to "
                                "ensemble methods, suggesting that the relationship between "
                                "predictors and outcome may be approximately linear."
                            )
                elif task_type == "classification":
                    simple_f1 = [metrics[m].get("F1", metrics[m].get("Accuracy", 0)) for m in simple_models if m in metrics]
                    complex_f1 = [metrics[m].get("F1", metrics[m].get("Accuracy", 0)) for m in complex_models if m in metrics]
                    if simple_f1 and complex_f1:
                        best_simple = max(simple_f1)
                        best_complex = max(complex_f1)
                        # If simple model is within 5% of complex, note it
                        if best_simple >= best_complex * 0.95:
                            parts.append(
                                "Logistic regression achieved performance comparable to "
                                "ensemble methods, suggesting that decision boundaries may "
                                "be approximately linear."
                            )
        
        return " ".join(parts)

    def _gen_discussion(self) -> str:
        """Discussion section: results-aware scaffold plus investigator-required placeholders."""
        parts = []
        
        # Principal Findings — auto-generated summary
        parts.append("### Principal Findings\n")
        
        models = self.ctx.get("models_trained", [])
        primary_model = self.ctx.get("primary_model", "")
        task_type = self.ctx.get("task_type", "")
        metrics = self.ctx.get("metrics_by_model", {})
        top_features = self.ctx.get("top_features", [])
        target_stats = self.ctx.get("target_stats", {})

        primary_model = self._resolve_primary_model(metrics, task_type, primary_model)
        
        if primary_model and metrics.get(primary_model):
            model_label = self._model_name(primary_model)
            best_metrics = metrics[primary_model]
            
            if task_type == "regression":
                r2 = best_metrics.get("R2")
                if r2 is not None:
                    parts.append(
                        f"In this analysis, {model_label} demonstrated the strongest "
                        f"predictive performance (R² = {self._fmt_param(r2)}), "
                        f"accounting for {int(r2*100)}% of variance in the outcome. "
                    )
                else:
                    parts.append(f"In this analysis, {model_label} demonstrated the strongest predictive performance. ")
            elif task_type == "classification":
                acc = best_metrics.get("Accuracy")
                f1 = best_metrics.get("F1")
                if acc is not None:
                    parts.append(
                        f"In this analysis, {model_label} demonstrated the strongest "
                        f"classification performance (accuracy = {self._fmt_param(acc)}). "
                    )
                elif f1 is not None:
                    parts.append(
                        f"In this analysis, {model_label} demonstrated the strongest "
                        f"classification performance (F1 = {self._fmt_param(f1)}). "
                    )
                else:
                    parts.append(f"In this analysis, {model_label} demonstrated the strongest classification performance. ")
        
        # Note if multiple models were compared
        if len(models) > 1:
            parts.append(
                f"Performance was compared across {len(models)} candidate models "
                f"({', '.join(self._model_name(m) for m in models)}). "
            )

        pattern_sentence = self._discussion_model_pattern(task_type, metrics)
        if pattern_sentence:
            parts.append(pattern_sentence + " ")

        regression_context_sentence = self._discussion_regression_context(
            best_metrics=metrics.get(primary_model, {}) if primary_model else {},
            target_stats=target_stats,
        )
        if regression_context_sentence:
            parts.append(regression_context_sentence + " ")

        if top_features:
            feature_phrase = self._human_join(top_features[:3])
            parts.append(
                f"The strongest predictors in the explainability analyses were {feature_phrase}. "
            )
        
        parts.append("\n")
        
        # Comparison with Prior Work — placeholder
        parts.append("### Comparison with Prior Work\n")
        parts.append("[Investigator required: Compare these results to published studies, "
                    "discuss agreement or discrepancies, and contextualize findings within the literature.]\n\n")
        
        # Strengths and Limitations — auto-populate from InsightLedger
        parts.append("### Strengths and Limitations\n")
        
        if self.ledger:
            discussion_points = self.ledger.discussion_points_for_manuscript()
            strengths = discussion_points.get("strengths", [])
            limitations = discussion_points.get("limitations", [])

            if strengths:
                parts.append("**Strengths (auto-generated from analysis ledger):** ")
                strength_strs = strengths[:3]
                parts.append("; ".join(strength_strs) + ". ")
            
            if limitations:
                parts.append("**Limitations (auto-generated from analysis ledger):** ")
                limitation_strs = limitations[:5]
                parts.append("; ".join(limitation_strs) + ". ")
            
            if not strengths and not limitations:
                parts.append(
                    "[Investigator required: No acknowledged strengths or limitations were "
                    "captured in the analysis ledger. Document any study-specific considerations here.] "
                )
            
            parts.append("\n")
        else:
            parts.append(
                "[Investigator required: Discuss methodological strengths "
                "(e.g., sample size, data quality, validation approach) and limitations "
                "(e.g., generalizability, unmeasured confounders, missing data).]\n\n"
            )
        
        # Clinical/Practical Implications — placeholder
        parts.append("### Clinical and Practical Implications\n")
        parts.append("[Investigator required: Discuss how findings could inform "
                    "practice, policy, or future research. Consider clinical significance "
                    "beyond statistical significance.]\n\n")
        
        # Conclusions — brief auto-generated restatement
        parts.append("### Conclusions\n")
        if primary_model:
            model_label = self._model_name(primary_model)
            parts.append(
                f"This study demonstrates that {model_label} can effectively predict "
                f"the target outcome from the available predictors. "
            )
        
        parts.append(
            "Further validation in independent cohorts and exploration of causal mechanisms "
            "are warranted before clinical or policy implementation.\n"
        )
        
        return "".join(parts)

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

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    _METRIC_KEY_MAP = {
        "rmse": "RMSE", "mae": "MAE", "r2": "R2", "medianae": "MedianAE",
        "accuracy": "Accuracy", "f1": "F1", "auc": "AUC",
        "precision": "Precision", "recall": "Recall",
    }

    def _normalize_metrics(self) -> None:
        """Normalize metric dict keys to canonical casing."""
        raw = self.ctx.get("metrics_by_model")
        if not raw:
            return
        normalized: dict = {}
        for model, mdict in raw.items():
            norm: dict = {}
            for k, v in mdict.items():
                canonical = self._METRIC_KEY_MAP.get(k.lower(), k)
                # Also handle CI keys like "rmse_ci_lower" → "RMSE_ci_lower"
                if "_ci_" in k.lower():
                    base = k.split("_ci_")[0]
                    suffix = "_ci_" + k.split("_ci_")[1]
                    canonical = self._METRIC_KEY_MAP.get(base.lower(), base) + suffix
                norm[canonical] = v
            normalized[model] = norm
        self.ctx["metrics_by_model"] = normalized

    def _resolve_primary_model(self, metrics: Dict[str, Dict[str, Any]], task_type: str, primary_model: str) -> str:
        """Resolve the manuscript primary model from explicit context or metric ranking."""
        if primary_model and metrics.get(primary_model):
            return primary_model
        if not metrics:
            return primary_model

        if task_type == "regression":
            ranked = [(m, vals.get("RMSE")) for m, vals in metrics.items() if vals.get("RMSE") is not None]
            if ranked:
                return min(ranked, key=lambda item: item[1])[0]
        elif task_type == "classification":
            ranked = [
                (m, vals.get("F1", vals.get("Accuracy")))
                for m, vals in metrics.items()
                if vals.get("F1", vals.get("Accuracy")) is not None
            ]
            if ranked:
                return max(ranked, key=lambda item: item[1])[0]

        return next(iter(metrics.keys()), primary_model)

    def _human_join(self, values: List[str]) -> str:
        """Join short phrase lists into publication-style prose."""
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _discussion_model_pattern(self, task_type: str, metrics: Dict[str, Dict[str, Any]]) -> str:
        """Interpret broad performance patterns without drifting into domain claims."""
        if len(metrics) < 2:
            return ""

        # Check ledger for a prefer-simpler insight (from post-training diagnostics)
        if self.ledger:
            try:
                prefer_simpler = self.ledger.get("train_prefer_simpler")
                if prefer_simpler and not prefer_simpler.resolved:
                    return (
                        f"{prefer_simpler.finding} "
                        "This pattern suggests that the available predictive signal is largely "
                        "captured by linear effects, favoring the simpler model on grounds of "
                        "parsimony and interpretability."
                    )
            except Exception:
                pass  # Fall through to hardcoded logic

        simple_keys = {"ridge", "lasso", "elasticnet", "logistic", "logreg", "glm", "huber"}
        complex_keys = {
            "rf", "random_forest", "extratrees_reg", "extratrees_clf",
            "histgb_reg", "histgb_clf", "xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf", "nn"
        }

        if task_type == "regression":
            scored = [(m, vals.get("RMSE")) for m, vals in metrics.items() if vals.get("RMSE") is not None]
            if len(scored) < 2:
                return ""
            scored.sort(key=lambda item: item[1])
            best_score = scored[0][1]
            if all(score <= best_score * 1.05 for _, score in scored[1:]):
                return (
                    "Performance differences across candidate models were small, suggesting "
                    "that the available predictive signal was captured similarly across model families."
                )

            simple_scores = [score for model, score in scored if model.lower() in simple_keys]
            complex_scores = [score for model, score in scored if model.lower() in complex_keys]
            if simple_scores and complex_scores:
                best_simple = min(simple_scores)
                best_complex = min(complex_scores)
                if best_simple <= best_complex * 1.05:
                    return (
                        "Regularized linear models performed comparably to more complex learners, "
                        "suggesting that much of the predictive signal may be captured by approximately linear effects."
                    )
                if best_complex < best_simple * 0.95:
                    return (
                        "Tree-based or boosted models outperformed simpler linear baselines, "
                        "suggesting that non-linear effects or feature interactions contribute materially to performance."
                    )

        elif task_type == "classification":
            scored = [
                (m, vals.get("F1", vals.get("Accuracy")))
                for m, vals in metrics.items()
                if vals.get("F1", vals.get("Accuracy")) is not None
            ]
            if len(scored) < 2:
                return ""
            scored.sort(key=lambda item: item[1], reverse=True)
            best_score = scored[0][1]
            if all(score >= best_score * 0.95 for _, score in scored[1:]):
                return (
                    "Performance differences across candidate models were small, suggesting "
                    "that discrimination was similar across model families."
                )

            simple_scores = [score for model, score in scored if model.lower() in simple_keys]
            complex_scores = [score for model, score in scored if model.lower() in complex_keys]
            if simple_scores and complex_scores:
                best_simple = max(simple_scores)
                best_complex = max(complex_scores)
                if best_simple >= best_complex * 0.95:
                    return (
                        "Simpler linear classifiers performed comparably to more complex learners, "
                        "suggesting that decision boundaries may be approximately linear."
                    )
                if best_complex > best_simple * 1.05:
                    return (
                        "Tree-based or boosted classifiers outperformed simpler linear baselines, "
                        "suggesting that non-linear decision boundaries contribute materially to discrimination."
                    )

        return ""

    def _discussion_regression_context(
        self,
        best_metrics: Dict[str, Any],
        target_stats: Dict[str, Any],
    ) -> str:
        """Contextualize regression fit against the observed outcome distribution."""
        if not best_metrics:
            return ""

        parts = []
        rmse = best_metrics.get("RMSE")
        r2 = best_metrics.get("R2")
        if rmse is not None:
            std_val = target_stats.get("std")
            min_val = target_stats.get("min")
            max_val = target_stats.get("max")
            if std_val:
                parts.append(
                    f"An RMSE of {self._fmt_param(rmse)} corresponded to approximately {rmse / std_val:.2f} SD of the outcome distribution."
                )
            elif min_val is not None and max_val is not None and max_val > min_val:
                outcome_range = max_val - min_val
                parts.append(
                    f"An RMSE of {self._fmt_param(rmse)} corresponded to approximately {rmse / outcome_range:.2f} of the observed outcome range."
                )
        if r2 is not None:
            explained = int(round(r2 * 100))
            unexplained = int(round((1 - r2) * 100))
            parts.append(
                f"This model explained {explained}% of outcome variance, leaving {unexplained}% unexplained."
            )
        return " ".join(parts)

    def _model_name(self, key: str) -> str:
        """Return human-readable model name, falling back to title-cased key."""
        # Try exact match, then lowercase, then uppercase
        if key in _MODEL_NAMES:
            return _MODEL_NAMES[key]
        lk = key.lower()
        if lk in _MODEL_NAMES:
            return _MODEL_NAMES[lk]
        # Fallback: replace underscores, title-case
        return key.replace("_", " ").title()

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

    def _describe_hyperparameters(self, model_key: str, params: Dict[str, Any]) -> str:
        """Generate human-readable hyperparameter description for a model.
        
        Returns a prose description of key hyperparameters, not a raw dump.
        Only includes parameters that matter (not sklearn defaults).
        """
        if not params:
            return ""
        
        # Map of human-readable descriptions for common hyperparameters
        # Focus on what reviewers care about, not internal sklearn naming
        desc_parts = []
        
        # Linear models (ridge, lasso, elasticnet)
        if "alpha" in params:
            alpha = params["alpha"]
            if "ridge" in model_key or "lasso" in model_key:
                reg_type = "L2" if "ridge" in model_key else "L1"
                desc_parts.append(f"alpha={self._fmt_param(alpha)} ({reg_type} regularization)")
            else:
                desc_parts.append(f"alpha={self._fmt_param(alpha)}")
        
        if "l1_ratio" in params and params.get("l1_ratio") is not None:
            desc_parts.append(f"L1 ratio={self._fmt_param(params['l1_ratio'])}")
        
        # Tree-based models (rf, xgb, lgbm, histgb)
        if "n_estimators" in params:
            desc_parts.append(f"{params['n_estimators']} trees")
        
        if "max_depth" in params:
            depth = params["max_depth"]
            if depth is None:
                desc_parts.append("unrestricted depth")
            else:
                desc_parts.append(f"max depth={depth}")
        
        if "learning_rate" in params:
            desc_parts.append(f"learning rate={self._fmt_param(params['learning_rate'])}")
        
        if "max_iter" in params and "histgb" in model_key:
            desc_parts.append(f"{params['max_iter']} boosting iterations")
        
        # Neural network
        if "hidden_layers" in params:
            layers = params["hidden_layers"]
            if isinstance(layers, list):
                layer_str = "×".join(str(w) for w in layers)
                desc_parts.append(f"architecture [{layer_str}]")
        
        if "dropout" in params and params.get("dropout", 0) > 0:
            desc_parts.append(f"dropout={self._fmt_param(params['dropout'])}")
        
        if "lr" in params:
            desc_parts.append(f"learning rate={self._fmt_param(params['lr'])}")
        
        if "epochs" in params:
            desc_parts.append(f"{params['epochs']} epochs")
        
        # SVM
        if "C" in params:
            desc_parts.append(f"C={self._fmt_param(params['C'])} (regularization)")
        
        if "kernel" in params:
            desc_parts.append(f"{params['kernel']} kernel")
        
        if "gamma" in params and params["gamma"] not in ("scale", "auto"):
            desc_parts.append(f"gamma={self._fmt_param(params['gamma'])}")
        
        # KNN
        if "n_neighbors" in params:
            desc_parts.append(f"k={params['n_neighbors']} neighbors")
        
        # Huber regression
        if "epsilon" in params:
            desc_parts.append(f"epsilon={self._fmt_param(params['epsilon'])}")
        
        # If we couldn't extract any meaningful description, fall back to raw params
        # but filter out None and common defaults
        if not desc_parts:
            filtered = {
                k: v for k, v in params.items()
                if v is not None and k not in ("random_state", "random_seed", "n_jobs", "verbose")
            }
            if filtered:
                param_strs = [f"{k}={self._fmt_param(v)}" for k, v in filtered.items()]
                return ", ".join(param_strs)
        
        return ", ".join(desc_parts)

    def _check_completeness(self) -> List[str]:
        """Check for missing workflow stages and return warnings."""
        warnings = []
        completeness = self.prov.get_completeness()

        if not completeness.get("upload"):
            warnings.append("Study design section requires data upload provenance.")
        if not completeness.get("preprocessing"):
            warnings.append("Preprocessing section requires pipeline configuration.")
        if not completeness.get("training"):
            warnings.append("Model development section requires training provenance.")
        if not completeness.get("split"):
            warnings.append("Study design section requires split configuration.")
        if not completeness.get("eda"):
            warnings.append("No EDA analyses were recorded in provenance.")

        return warnings
