"""Pre-export manuscript consistency validator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ml.narrative_engine import _MODEL_NAMES


@dataclass
class ManuscriptValidationCheck:
    """Single validation result."""

    name: str
    status: str
    location: str
    detail: str


@dataclass
class ManuscriptValidationReport:
    """Validation report for a manuscript export bundle."""

    checks: List[ManuscriptValidationCheck]

    @property
    def failed_checks(self) -> List[ManuscriptValidationCheck]:
        return [check for check in self.checks if check.status == "FAIL"]

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def to_rows(self) -> List[Dict[str, str]]:
        return [
            {
                "Status": "PASS" if check.status == "PASS" else "FAIL",
                "Check": check.name,
                "Location": check.location,
                "Detail": check.detail,
            }
            for check in self.checks
        ]


def _extract_section(text: str, heading: str, level: int) -> str:
    pattern = rf"(?ms)^{'#' * level}\s+{re.escape(heading)}\s*\n(.*?)(?=^{'#' * level}\s+|^\#{{1,{level-1}}}\s+|\Z)"
    match = re.search(pattern, text or "")
    return match.group(1).strip() if match else ""


def _extract_latex_subsection(text: str, heading: str) -> str:
    pattern = rf"(?ms)\\subsection\{{{re.escape(heading)}\}}\s*(.*?)(?=\\subsection\{{|\\section\{{|\\paragraph\{{|\\end\{{document\}}|\Z)"
    match = re.search(pattern, text or "")
    return match.group(1).strip() if match else ""


def _extract_analysis_n(text: str) -> Optional[int]:
    patterns = [
        r"Of\s+[\d,]+\s+observations,\s+([\d,]+)\s+remained for analysis",
        r"A total of\s+([\d,]+)\s+observations were available for analysis",
        r"dataset of\s+([\d,]+)\s+observations",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _extract_final_predictor_count(text: str) -> Optional[int]:
    patterns = [
        r"retained\s+([\d,]+)\s+predictors?\s+for final modeling",
        r"final modeling set contained\s+([\d,]+)\s+predictors?",
        r"([\d,]+)\s+predictors?\s+for final modeling",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _model_variants(model_key: str) -> List[str]:
    display = _MODEL_NAMES.get(model_key) or _MODEL_NAMES.get(model_key.lower()) or model_key.replace("_", " ").title()
    variants = [display, model_key.upper(), model_key.replace("_", " ").title()]
    return [variant for variant in variants if variant]


def _contains_any_variant(text: str, variants: List[str]) -> bool:
    lowered = (text or "").lower()
    return any(variant.lower() in lowered for variant in variants)


def _invalid_metric_terms_for_task(text: str, task_type: str) -> List[str]:
    invalid_terms = {
        "regression": {"accuracy", "f1", "auc", "precision", "recall"},
        "classification": {"rmse", "mae", "r2", "medianae"},
    }.get(task_type, set())
    return sorted({term for term in invalid_terms if re.search(rf"\b{re.escape(term)}\b", text or "", re.IGNORECASE)})


def validate_manuscript_bundle(
    manuscript_context: Optional[Dict[str, Any]],
    methods_text: str,
    report_text: str,
    latex_text: str,
    task_type: str,
) -> ManuscriptValidationReport:
    """Validate manuscript consistency before export."""
    context = manuscript_context or {}
    population = context.get("population_counts") or {}
    feature_counts = context.get("feature_counts") or {}
    included_models = (
        context.get("included_models")
        or list((context.get("selected_model_results") or {}).keys())
        or []
    )
    checks: List[ManuscriptValidationCheck] = []

    abstract_section = _extract_section(report_text, "Abstract (Draft)", level=2)
    study_design_section = _extract_section(methods_text, "Study Design", level=3)
    predictor_section = _extract_section(methods_text, "Predictor Variables", level=3)
    model_dev_section = _extract_section(methods_text, "Model Development", level=3)
    model_eval_section = _extract_section(methods_text, "Model Evaluation", level=3)
    latex_model_dev_section = _extract_latex_subsection(latex_text, "Model Development")
    combined_export_text = f"{report_text}\n{latex_text}"

    expected_analysis_n = population.get("analysis_total")
    abstract_analysis_n = _extract_analysis_n(abstract_section)
    study_design_n = _extract_analysis_n(study_design_section)
    analysis_match = (
        expected_analysis_n is not None
        and abstract_analysis_n == expected_analysis_n
        and study_design_n == expected_analysis_n
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Analysis population is consistent across abstract and study design",
            status="PASS" if analysis_match else "FAIL",
            location="Abstract / Methods: Study Design",
            detail=(
                f"Expected analysis N={expected_analysis_n}, abstract N={abstract_analysis_n}, "
                f"study design N={study_design_n}."
            ),
        )
    )

    split_total = sum(
        int(population.get(key) or 0)
        for key in ("train_n", "val_n", "test_n")
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Split counts reconcile to analysis population",
            status="PASS" if expected_analysis_n == split_total else "FAIL",
            location="Methods: Study Design",
            detail=f"analysis_total={expected_analysis_n}, split_sum={split_total}.",
        )
    )

    expected_predictors = feature_counts.get("selected")
    if expected_predictors is None:
        expected_predictors = len(context.get("feature_names_for_manuscript") or [])
    abstract_predictors = _extract_final_predictor_count(abstract_section)
    methods_predictors = _extract_final_predictor_count(predictor_section)
    predictor_match = (
        expected_predictors is not None
        and abstract_predictors == expected_predictors
        and methods_predictors == expected_predictors
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Final predictor count is consistent across abstract and methods",
            status="PASS" if predictor_match else "FAIL",
            location="Abstract / Methods: Predictor Variables",
            detail=(
                f"Expected predictors={expected_predictors}, abstract={abstract_predictors}, "
                f"predictor section={methods_predictors}."
            ),
        )
    )

    missing_models = []
    for model_key in included_models:
        variants = _model_variants(model_key)
        in_dev = _contains_any_variant(model_dev_section, variants)
        in_eval = _contains_any_variant(model_eval_section, variants)
        if not (in_dev and in_eval):
            missing_models.append(model_key)
    checks.append(
        ManuscriptValidationCheck(
            name="Model names match between development and evaluation sections",
            status="PASS" if not missing_models else "FAIL",
            location="Methods: Model Development / Model Evaluation",
            detail=(
                "All selected models appear in both sections."
                if not missing_models
                else f"Missing or inconsistent models: {', '.join(missing_models)}."
            ),
        )
    )

    metric_name = (context.get("best_metric_name") or "").lower()
    invalid_metric = (
        task_type == "regression" and metric_name in {"accuracy", "f1", "auc", "precision", "recall"}
    ) or (
        task_type == "classification" and metric_name in {"rmse", "mae", "r2", "medianae"}
    )
    invalid_metric_terms = _invalid_metric_terms_for_task(
        "\n".join(part for part in (model_dev_section, latex_model_dev_section) if part),
        task_type,
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Selection metric language matches task type",
            status="PASS" if not (invalid_metric or invalid_metric_terms) else "FAIL",
            location="Export Context / Methods",
            detail=(
                f"task_type={task_type}, best_metric_name={context.get('best_metric_name')}."
                if not invalid_metric_terms
                else (
                    f"task_type={task_type}, best_metric_name={context.get('best_metric_name')}, "
                    f"invalid rendered metric term(s) in model-development prose: {', '.join(invalid_metric_terms)}."
                )
            ),
        )
    )

    explicit_primary_claim = bool(
        re.search(r"\bselected as the primary model\b|\bmanuscript-primary model was\b", f"{model_dev_section}\n{latex_model_dev_section}", re.IGNORECASE)
    )
    no_primary_claim = "no manuscript-primary model was explicitly selected" in combined_export_text.lower()
    expected_primary_model = context.get("manuscript_primary_model")
    primary_conflict = (
        (explicit_primary_claim and no_primary_claim)
        or (explicit_primary_claim and not expected_primary_model)
        or (no_primary_claim and bool(expected_primary_model))
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Primary model statements are internally consistent",
            status="PASS" if not primary_conflict else "FAIL",
            location="Methods / Results",
            detail=(
                f"manuscript_primary_model={expected_primary_model}, "
                f"explicit_primary_claim={explicit_primary_claim}, no_primary_claim={no_primary_claim}."
            ),
        )
    )

    original_count = feature_counts.get("original")
    selected_count = feature_counts.get("selected")
    reduction_language = bool(re.search(r"feature selection|retained .* predictors|reduced", abstract_section or "", re.IGNORECASE))
    should_not_reduce = original_count is not None and selected_count is not None and original_count == selected_count
    checks.append(
        ManuscriptValidationCheck(
            name="Abstract feature-selection language matches actual reduction",
            status="PASS" if not (should_not_reduce and reduction_language) else "FAIL",
            location="Abstract (Draft)",
            detail=(
                "No reduction language detected."
                if not (should_not_reduce and reduction_language)
                else f"Abstract still describes feature reduction even though original={original_count} and selected={selected_count}."
            ),
        )
    )

    artifact_patterns = {
        "placeholder prompt": r"\[PLACEHOLDER(?::|\])",
        "note tag": r"\[NOTE(?::|\])",
        "markdown heading": r"(?:^|\n)##\s+\S+|\\#\\#",
        "markdown bold": r"\*\*[^*]+\*\*",
    }
    artifact_tokens = [
        label for label, pattern in artifact_patterns.items()
        if re.search(pattern, latex_text or "", re.IGNORECASE)
    ]
    checks.append(
        ManuscriptValidationCheck(
            name="LaTeX output is free of markdown and note artifacts",
            status="PASS" if not artifact_tokens else "FAIL",
            location="LaTeX export",
            detail=(
                "No raw placeholder or markdown tokens detected."
                if not artifact_tokens
                else f"Detected raw artifacts: {', '.join(artifact_tokens)}."
            ),
        )
    )

    internal_keys = sorted({key for key in _MODEL_NAMES if "_" in key})
    leaked_keys = []
    for key in internal_keys:
        if re.search(rf"\b{re.escape(key)}\b", combined_export_text) or re.search(rf"\b{re.escape(key.upper())}\b", combined_export_text):
            leaked_keys.append(key.upper())
    checks.append(
        ManuscriptValidationCheck(
            name="No internal model keys leak into export text",
            status="PASS" if not leaked_keys else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No internal model identifiers detected."
                if not leaked_keys
                else f"Leaked keys: {', '.join(leaked_keys)}."
            ),
        )
    )

    coaching_patterns = [
        "no action needed",
        "favorable to analysis",
        "workflow-derived abstract",
        "[applicable to",
    ]
    found_patterns = [pattern for pattern in coaching_patterns if pattern in combined_export_text.lower()]
    checks.append(
        ManuscriptValidationCheck(
            name="No coaching language patterns remain in export text",
            status="PASS" if not found_patterns else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No coaching language detected."
                if not found_patterns
                else f"Found coaching patterns: {', '.join(found_patterns)}."
            ),
        )
    )

    punctuation_issues = []
    if re.search(r"\.\.", combined_export_text):
        punctuation_issues.append("double periods")
    if re.search(r"\b(Table|Figure)\s+X\b", combined_export_text):
        punctuation_issues.append("dangling Table/Figure X reference")
    if re.search(r"[—-]\.", combined_export_text):
        punctuation_issues.append("dash followed by period")
    checks.append(
        ManuscriptValidationCheck(
            name="No obvious dangling punctuation or placeholder references remain",
            status="PASS" if not punctuation_issues else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No obvious dangling references detected."
                if not punctuation_issues
                else f"Detected issues: {', '.join(punctuation_issues)}."
            ),
        )
    )

    return ManuscriptValidationReport(checks=checks)
