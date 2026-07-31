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

    #: The ownership contract shipped inside every draft. Methods/Results are
    #: compiled from recorded events; interpretation belongs to the authors.
    OWNERSHIP_PREAMBLE = (
        "> **How to read this draft.** The Methods and Results below are "
        "compiled from the recorded analysis workflow: every quantitative "
        "statement traces to a logged event (see the evidence map), and "
        "nothing is asserted that the pipeline did not record. Passages "
        "marked **[AUTHOR REQUIRED — …]** belong to the authors: "
        "interpretation, literature context, and claims of adequacy are "
        "yours, not the software's. Verify all compiled text before "
        "submission.\n"
    )

    def count_author_inputs(self) -> int:
        """Number of [AUTHOR REQUIRED — …] scaffolds awaiting the author.

        Counts section content only (not the ownership preamble, which
        mentions the marker while explaining it).
        """
        return sum(v.count("[AUTHOR REQUIRED") for v in self.all_sections.values())

    def to_markdown(self) -> str:
        """Render as markdown with subsection headers."""
        lines = [self.OWNERSHIP_PREAMBLE]

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
        """Convert markdown formatting to LaTeX equivalents.

        LaTeX specials are escaped FIRST, while the text is still plain prose
        with no commands in it — feature names like ``feat_0042`` and values
        like ``15%`` would otherwise break compilation (underscore = math
        mode, percent = comment). The bold/italic conversion runs after, so
        the commands it inserts stay intact.
        """
        import re
        for ch, rep in (('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_')):
            text = text.replace(ch, rep)
        # Bold **text** → \textbf{text}
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        # Italic *text* → \textit{text}  (single asterisk, not inside bold)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)
        text = text.replace('²', '$^2$')
        text = text.replace('×', '$\\times$')
        text = text.replace('−', '--')
        text = text.replace('–', '--')
        text = text.replace('—', '---')
        return text

    def to_latex(self) -> str:
        """Render as LaTeX subsections."""
        import re

        lines = [
            "% How to read this draft: Methods and Results are compiled from",
            "% the recorded analysis workflow — every quantitative statement",
            "% traces to a logged event. Passages marked [AUTHOR REQUIRED - ...]",
            "% belong to the authors. Verify all compiled text before submission.",
            "",
        ]

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

    def generate_evidence_map(self) -> str:
        """Markdown table tracing each compiled draft section to its recorded
        sources.

        This is the artifact that makes 'compiled from the audit trail'
        checkable rather than a slogan: for every section, which workflow
        events supplied its facts and the key recorded values. Sections whose
        sources were never recorded say so — the draft omits them rather
        than inventing content.
        """
        ctx = self.ctx
        comp = self.prov.get_completeness()

        def _n(key):
            v = ctx.get(key)
            return v if v is not None else "—"

        rows: List[tuple] = []

        if comp.get("upload") or comp.get("split"):
            vals = (f"{_n('n_total')} observations; target `{ctx.get('target_name', '—')}` "
                    f"({ctx.get('task_type', '—')})")
            if comp.get("split"):
                vals += (f"; split {_n('n_train')}/{_n('n_val')}/{_n('n_test')} "
                         f"(strategy: {ctx.get('split_strategy', '—')}, "
                         f"seed {_n('random_seed')})")
            rows.append(("Study Design", "upload + split records", vals))
        else:
            rows.append(("Study Design", "NOT RECORDED", "section omitted or generic"))

        # The draft's own preamble points a reader HERE as the proof that every
        # quantitative statement traces to a logged event. An evidence map that
        # prints one group's N as the study's, with no row naming the filter, is
        # the one artifact that must not omit it.
        _up = getattr(self.prov, "upload", None)
        if _up is not None and getattr(_up, "cohort_column", ""):
            rows.append((
                "Sample Restriction", "cohort run record",
                f"analysis restricted to `{_up.cohort_column}` = "
                f"{_up.cohort_value} — {_up.cohort_n:,} of "
                f"{_up.study_n:,} in the study; every N above is this group's",
            ))

        if comp.get("feature_selection"):
            rows.append(("Predictor Variables", "feature-selection record",
                         f"{ctx.get('fs_method', '—')}: "
                         f"{_n('n_features_before_selection')} → "
                         f"{_n('n_features_after_selection')} predictors"))
        elif comp.get("upload"):
            rows.append(("Predictor Variables", "upload record",
                         f"{_n('n_features_original')} predictors (no selection recorded)"))
        else:
            rows.append(("Predictor Variables", "NOT RECORDED", "section omitted or generic"))

        if comp.get("preprocessing"):
            models_cfg = ctx.get("models_configured") or []
            rows.append(("Missing Data / Preprocessing", "per-model preprocessing configs",
                         f"{len(models_cfg)} model pipeline(s): "
                         f"{', '.join(models_cfg) if models_cfg else '—'}"))
        else:
            rows.append(("Missing Data / Preprocessing", "NOT RECORDED",
                         "section omitted or generic"))

        if ctx.get("coach_headline") or ctx.get("coach_picks"):
            _n_picks = len(ctx.get("coach_picks") or [])
            rows.append(("Model Development — shortlist rationale", "coach record",
                         f"{_n_picks} shortlisted pick(s); "
                         f"{'probe: ' + ctx.get('coach_probe_summary') if ctx.get('coach_probe_summary') else 'shape-based rationale'}"))

        if comp.get("training"):
            models = ctx.get("models_trained") or []
            rows.append(("Model Development / Evaluation", "training record",
                         f"{len(models)} model(s): {', '.join(models) if models else '—'}; "
                         f"primary: {ctx.get('primary_model') or '—'}; "
                         f"CV: {'yes' if ctx.get('use_cv') else 'no'}"))
            rows.append(("Results", "recorded per-model metrics",
                         f"metrics for {len(ctx.get('metrics_by_model') or {})} model(s)"))
        else:
            rows.append(("Model Development / Evaluation", "NOT RECORDED",
                         "section omitted or generic"))
            rows.append(("Results", "NOT RECORDED", "section omitted"))

        if comp.get("sensitivity"):
            rows.append(("Sensitivity Analysis", "sensitivity record",
                         "seed stability / feature dropout as recorded"))
        if comp.get("statistical_validation"):
            rows.append(("Statistical Validation", "statistical-test record",
                         f"{len(ctx.get('statistical_tests') or [])} test(s)"))

        if self.ledger is not None and len(self.ledger) > 0:
            contributing = sorted(
                i.id for i in self.ledger.insights
                if i.id.startswith("eda_opportunity_") or i.acknowledged or not i.resolved
            )
            rows.append(("Data Observations & Strengths/Limitations", "insight ledger",
                         f"{len(contributing)} contributing insight(s): "
                         f"{', '.join(contributing[:8])}"
                         f"{' …' if len(contributing) > 8 else ''}"))

        rows.append(("Discussion — interpretation, prior work, implications",
                     "AUTHOR", "author-owned; evidence-citing scaffolds only"))

        lines = [
            "# Evidence Map",
            "",
            "Every compiled section of the draft traces to recorded workflow "
            "events. \"NOT RECORDED\" means the pipeline holds no evidence for "
            "that section — the draft omits it rather than inventing it.",
            "",
            "| Draft section | Compiled from | Recorded values |",
            "|---|---|---|",
        ]
        for section, source, vals in rows:
            lines.append(f"| {section} | {source} | {vals} |")
        return "\n".join(lines) + "\n"

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

        if self.manuscript_context.get("exploratory_mode"):
            self.ctx["exploratory_mode"] = True

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
            # Immediately, not in a footnote: without it this N reads as the
            # study population when it is one group's.
            from utils.workflow_provenance import cohort_restriction_sentence
            _restriction = cohort_restriction_sentence()
            if _restriction:
                parts.append(_restriction)
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

        # The quarantine timing is the study design's strongest methodological
        # guarantee — state it explicitly when the lockbox governed the run.
        if not self.ctx.get("exploratory_mode") and self.ledger is not None:
            try:
                _lb_ins = self.ledger.get("upload_test_lockbox")
            except Exception:
                _lb_ins = None
            if _lb_ins is not None and _lb_ins.resolved:
                _p = (_lb_ins.resolution_details or {}).get("params", {})
                _frac = _p.get("fraction")
                parts.append(
                    "The held-out test set"
                    + (f" ({_frac:.0%} of eligible observations)" if _frac else "")
                    + " was frozen at data upload, before any feature engineering "
                    "or feature selection, and was accessed only for the final "
                    "evaluation."
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

    def _borrowed_pipeline_note(self) -> str:
        """Models trained with another model's preprocessing, named.

        Training resolves `get_preprocessing_pipeline(key) or pipeline`, and
        that fallback is the FIRST prepared model's pipeline — so a model
        selected on Train & Compare but never prepared really is trained
        through another model's PCA or power transform. The methods section
        described only the prepared models, leaving a reader no way to know.
        """
        try:
            import streamlit as st
            by_model = st.session_state.get("preprocessing_pipelines_by_model") or {}
            built = set(st.session_state.get("preprocess_built_model_keys") or [])
            trained = set((st.session_state.get("model_results") or {}).keys())
            borrowers = sorted(m for m in trained if m not in built)
            if not borrowers or not by_model:
                return ""
            owner = "default" if "default" in by_model else next(iter(by_model))
            names = ", ".join(m.upper() for m in borrowers)
            if owner == "default":
                return f"{names} used the shared preprocessing pipeline."
            # One borrower is the common case, and the plural verb read as a
            # typo in an exported manuscript.
            was = "was" if len(borrowers) == 1 else "were"
            return (f"{names} had no preprocessing configured and {was} "
                    f"trained using the pipeline built for {owner.upper()}, "
                    f"including any transform chosen specifically for that "
                    f"model.")
        except Exception:
            return ""

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

        # "All models" is a claim about every model TRAINED, but `pp` holds only
        # the models that were PREPARED. With a borrower in the run the sentence
        # was contradicted by the very next one, which named a model that had
        # none.
        _borrowed = self._borrowed_pipeline_note()
        _all = "The prepared models shared" if _borrowed else "All models shared"

        if not differs:
            # All models share preprocessing
            cfg = next(iter(pp.values()))
            sents = self._describe_preprocessing(cfg)
            if sents:
                parts.append(f"{_all} identical preprocessing: {'; '.join(sents)}.")
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

        # A model trained through ANOTHER model's pipeline is described here or
        # nowhere. Leaving it out let a methods section name every prepared
        # model's transforms while saying nothing about the model that borrowed
        # them — including a PCA a reader would need to know about to interpret
        # the explainability at all.
        if _borrowed:
            parts.append(_borrowed)

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

        # Model-selection rationale (a TRIPOD reporting item) — compiled from
        # the coach's recorded reasoning rather than reconstructed from memory.
        # Only cited when the trained lineup actually overlaps the shortlist:
        # if the user ignored the coach, the Methods must not claim a
        # rationale that the model list visibly contradicts.
        coach_picks = self.ctx.get("coach_picks") or []
        _pick_keys = {p.get("model_key") for p in coach_picks if isinstance(p, dict)}
        _followed = bool(_pick_keys & set(models))
        coach_headline = (self.ctx.get("coach_headline") or "").strip()
        if coach_headline and _followed:
            _rationale = coach_headline.rstrip(".")
            # strip advisory emoji/prefixes from the UI register
            _rationale = _rationale.replace("⚠️ ", "").replace("Dominant constraint: ", "")
            parts.append(
                f"Candidate models were shortlisted from the dataset's "
                f"characteristics: {_rationale[0].lower() + _rationale[1:]}."
            )
        coach_probe = (self.ctx.get("coach_probe_summary") or "").strip()
        if coach_probe and _followed:
            parts.append(
                f"A preliminary cross-validated screen on the training data "
                f"informed this shortlist ({coach_probe}); screen scores were "
                f"advisory and are not reported as results."
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

        # NN config rationale (#95)
        nn_config_source = self.ctx.get("nn_config_source", "")
        if nn_config_source == "recommended":
            parts.append(
                "Neural network hyperparameters were configured using data-driven "
                "recommendations based on dataset characteristics (sample-to-feature "
                "ratio, target distribution, and data sufficiency)."
            )
        elif nn_config_source == "recommended+modified":
            mods = self.ctx.get("nn_config_modifications", {})
            if mods:
                mod_names = ", ".join(mods.keys())
                parts.append(
                    "Neural network hyperparameters were initialized from data-driven "
                    f"recommendations, with manual adjustments to: {mod_names}."
                )

        # Class weighting. `GUIDED-049`.
        #
        # This said "To address class imbalance, class_weight='balanced' was
        # applied…" — unconditionally, and approvingly, in the artifact that IS
        # the product. Van den Goorbergh et al. (JAMIA 2022;29:1525) and
        # Carriero et al. (Stat Med 2025) show rebalancing overestimates
        # minority-class probability without improving discrimination.
        #
        # What was done is still reported — a reader has to know — but it is no
        # longer reported as a remedy, and it carries the limitation.
        if self.ctx.get("class_weight_balanced"):
            from ml.imbalance_advice import manuscript_sentence
            parts.append(manuscript_sentence(self.ctx.get("model_purpose")))

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
        """Statistical validation tests, and what may be said about how many hit.

        `AUDIT-001`. This paragraph used to end with *"N of M tests yielded
        statistically significant results (p < 0.05)"* — an uncorrected count,
        with no correction named and none applied, written into the artifact
        that is the product. `research/METABOLOMICS_PACK.md` §06.3: *plotting
        raw p-values with a line at p = 0.05 on a 3,000-feature untargeted
        dataset is an anti-pattern and would be flagged in review*, and §10
        lists *asterisks without the test or correction*.

        Two branches now, and the second is the one that matters:

        * **A correction was recorded** — report the corrected count, naming
          the method and the threshold, because that IS a result.
        * **No correction was recorded** — report NO count. The number of tests
          reaching a raw p < 0.05 is the quantity the anti-pattern is made of,
          and printing it beside the word "significant" is the assertion.

        Silence was the other option offered and it is the weaker one: a
        paragraph that simply stops is indistinguishable from a family in which
        nothing was significant, and `DESIGN_LANGUAGE.md` §09's recorded-absence
        rule is exactly about that confusion. So the absence is stated — the
        tests are uncorrected, the count is not interpretable, and here is how
        many would be expected to clear the line with nothing going on. That
        last number is the pack's own coaching, and it is the sentence that
        turns an omission into information a reviewer can use.
        """
        from ml import multiplicity

        tests = self.ctx.get("statistical_tests", [])
        if not tests:
            return ""

        parts = []
        # SORTED, not `list(set(...))`. Set iteration order is not stable
        # across runs, so the same analysis produced a manuscript whose test
        # list was in a different order each time it was drafted. A record that
        # changes when nothing changed is not a record.
        test_names = sorted({t.get("test_name", "") for t in tests
                             if t.get("test_name")})
        if test_names:
            parts.append(
                f"Statistical validation was performed using: {', '.join(test_names)}."
            )

        with_p = [t for t in tests
                  if isinstance(t.get("p_value"), (int, float))
                  and t.get("p_value") is not None]
        correction = multiplicity.correction_of(tests)

        if correction and with_p:
            alpha = next((float(t.get("correction_alpha")) for t in tests
                          if t.get("correction_alpha") is not None), 0.05)
            corrected = [t for t in with_p
                         if t.get("significant_after_correction")]
            parts.append(
                f"P-values were adjusted for multiple comparisons using the "
                f"{multiplicity.method_label(correction)} method across the "
                f"{len(with_p)} tests reported here; "
                f"{len(corrected)} remained significant at q < {alpha:g}."
            )
        elif with_p:
            expected = multiplicity.expected_by_chance(len(with_p))
            parts.append(
                f"No correction for multiple comparisons was applied across "
                f"the {len(with_p)} tests reported here, so the number reaching "
                f"p < 0.05 is not interpretable as a count of findings: at "
                f"alpha = 0.05 roughly {expected:.0f} of {len(with_p)} would be "
                f"expected to do so by chance alone."
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
        
        # Best model by metric — used as the default primary AND to keep the
        # prose honest when the user deliberately picks a non-best primary
        # (e.g. an interpretable model): the Results section must not claim
        # "best overall performance" for a model the comparison sentence will
        # then describe as having the highest error.
        metric_best = ""
        if metrics:
            if task_type == "regression":
                # Lower RMSE is better
                best_rmse = float("inf")
                for m, m_metrics in metrics.items():
                    rmse = m_metrics.get("RMSE", float("inf"))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        metric_best = m
            elif task_type == "classification":
                # Higher accuracy/F1 is better
                best_f1 = 0.0
                for m, m_metrics in metrics.items():
                    f1 = m_metrics.get("F1", m_metrics.get("Accuracy", 0.0))
                    if f1 > best_f1:
                        best_f1 = f1
                        metric_best = m
        if not primary_model:
            primary_model = metric_best
        
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

                if not metric_best or primary_model == metric_best:
                    parts.append(f"{model_label} demonstrated the best overall performance.")
                else:
                    parts.append(
                        f"{model_label} was selected as the primary model for reporting; "
                        f"{self._model_name(metric_best)} achieved the best point-estimate performance."
                    )
                
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

                if not metric_best or primary_model == metric_best:
                    parts.append(f"{model_label} demonstrated the best overall performance.")
                else:
                    parts.append(
                        f"{model_label} was selected as the primary model for reporting; "
                        f"{self._model_name(metric_best)} achieved the best point-estimate performance."
                    )
                
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
        
        # Feature importance findings — from the explainability context when
        # available; omitted entirely otherwise (never a placeholder in a
        # reader-facing Results section).
        top_features = self.ctx.get("top_features") or []
        if top_features:
            parts.append(
                f"Feature importance analysis identified "
                f"{self._human_join([str(f) for f in top_features[:5]])} "
                f"as the strongest predictors."
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
        target_name = self.ctx.get("target_name") or "the outcome"

        # "strongest" implies a comparison — only claim it when there was one.
        _comparative = len(models) > 1
        headline = ""  # e.g. "R² = 0.62"; reused by the author scaffolds below

        if primary_model and metrics.get(primary_model):
            model_label = self._model_name(primary_model)
            best_metrics = metrics[primary_model]

            if task_type == "regression":
                r2 = best_metrics.get("R2")
                if r2 is not None:
                    headline = f"R² = {self._fmt_param(r2)}"
                    if _comparative:
                        lead = (f"Among the models compared, {model_label} achieved "
                                f"the strongest held-out performance ({headline})")
                    else:
                        lead = f"{model_label} achieved {headline} on the held-out test set"
                    if r2 >= 0:
                        lead += (f", accounting for {int(r2*100)}% of variance "
                                 f"in {target_name}. ")
                    else:
                        # A negative R² is worse than predicting the mean — say so
                        # rather than dressing it up.
                        lead += (", which is below a mean-only baseline: the model "
                                 "did not explain outcome variance in held-out data. ")
                    parts.append(lead)
                else:
                    parts.append(f"{model_label} was selected as the primary model. ")
            elif task_type == "classification":
                acc = best_metrics.get("Accuracy")
                f1 = best_metrics.get("F1")
                if acc is not None:
                    headline = f"accuracy = {self._fmt_param(acc)}"
                elif f1 is not None:
                    headline = f"F1 = {self._fmt_param(f1)}"
                if headline:
                    if _comparative:
                        parts.append(
                            f"Among the models compared, {model_label} achieved the "
                            f"strongest held-out classification performance ({headline}). "
                        )
                    else:
                        parts.append(
                            f"{model_label} achieved {headline} on the held-out test set. "
                        )
                else:
                    parts.append(f"{model_label} was selected as the primary model. ")
        
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
        
        # Comparison with Prior Work — author-owned, scaffolded with the
        # study's own evidence so the author starts from facts, not a blank.
        parts.append("### Comparison with Prior Work\n")
        _headline_clause = (
            f" This analysis achieved {headline} for predicting {target_name}."
            if headline else ""
        )
        parts.append(
            "[AUTHOR REQUIRED — Situate these results in the literature."
            f"{_headline_clause} Compare against published models of the same or "
            "a similar outcome, and discuss agreement, discrepancies, and "
            "plausible reasons (population, predictors, validation design).]\n\n"
        )
        
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
            
            if self.ctx.get("exploratory_mode"):
                limitations = list(limitations) + [
                    "the analysis was run in exploratory mode: the held-out test "
                    "set was not quarantined from feature engineering and "
                    "selection, so reported performance may be optimistically "
                    "biased and should not be presented as validated held-out "
                    "performance"
                ]

            if limitations:
                parts.append("**Limitations (auto-generated from analysis ledger):** ")
                limitation_strs = limitations[:5]
                parts.append("; ".join(limitation_strs) + ". ")
            
            if not strengths and not limitations:
                parts.append(
                    "[AUTHOR REQUIRED — No acknowledged strengths or limitations "
                    "were captured in the analysis ledger. Document any "
                    "study-specific considerations here.] "
                )

            parts.append("\n")
        else:
            parts.append(
                "[AUTHOR REQUIRED — Discuss methodological strengths "
                "(e.g., sample size, data quality, validation approach) and limitations "
                "(e.g., generalizability, unmeasured confounders, missing data).]\n\n"
            )

        # Clinical/Practical Implications — author-owned, scaffolded with the
        # explainability evidence when available.
        parts.append("### Clinical and Practical Implications\n")
        if top_features:
            _feats_clause = self._human_join(top_features[:3])
            parts.append(
                f"[AUTHOR REQUIRED — The leading predictors in the explainability "
                f"analyses were {_feats_clause}. Discuss whether these associations "
                "are plausible and actionable in your domain, and what use "
                "(screening, triage, hypothesis generation) the observed "
                "performance would support. These are predictive associations, "
                "not causal effects — do not present them as drivers of the "
                "outcome.]\n\n"
            )
        else:
            parts.append(
                "[AUTHOR REQUIRED — Discuss how findings could inform practice, "
                "policy, or future research. Consider clinical significance "
                "beyond statistical significance.]\n\n"
            )

        # Conclusions — state the recorded facts; the claim they support
        # belongs to the author. "Effectively predicts" is not something the
        # software can certify, at any metric value.
        parts.append("### Conclusions\n")
        if primary_model and headline:
            parts.append(
                f"In this dataset, {self._model_name(primary_model)} predicted "
                f"{target_name} with {headline} on the held-out test set. "
            )
        elif primary_model:
            parts.append(
                f"In this dataset, {self._model_name(primary_model)} was selected "
                f"as the primary model. "
            )
        parts.append(
            "[AUTHOR REQUIRED — State the conclusion this level of performance "
            "supports in your domain: whether it is adequate for the intended "
            "application, and what it does not establish.] "
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
                layer_str = "\u00d7".join(str(w) for w in layers)
                desc_parts.append(f"architecture [{layer_str}]")

        if "num_layers" in params and "hidden_layers" not in params:
            width = params.get("layer_width", "?")
            desc_parts.append(f"{params['num_layers']} layers \u00d7 {width} units")

        if params.get("use_batchnorm"):
            desc_parts.append("batch normalization")

        if "dropout" in params and params.get("dropout", 0) > 0:
            desc_parts.append(f"dropout={self._fmt_param(params['dropout'])}")

        if "lr" in params:
            desc_parts.append(f"learning rate={self._fmt_param(params['lr'])}")

        if "epochs" in params:
            desc_parts.append(f"{params['epochs']} epochs")

        if "lr_scheduler" in params and params["lr_scheduler"] != "reduce_on_plateau":
            sched_names = {"cosine_warm_restarts": "cosine annealing", "one_cycle": "one-cycle LR"}
            desc_parts.append(sched_names.get(params["lr_scheduler"], params["lr_scheduler"]))

        if params.get("grad_clip_norm") is not None:
            desc_parts.append(f"gradient clipping (max norm={self._fmt_param(params['grad_clip_norm'])})")

        if "loss_function" in params and params["loss_function"] != "mse":
            loss_names = {"huber": "Huber loss", "weighted_huber": "weighted Huber loss", "mae": "MAE loss"}
            desc_parts.append(loss_names.get(params["loss_function"], params["loss_function"]))
        
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
