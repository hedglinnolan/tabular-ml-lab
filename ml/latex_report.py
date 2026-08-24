"""
LaTeX report generator.

Generates a complete LaTeX manuscript template populated with actual results
from the modeling workflow. Ready to compile with pdflatex.
"""
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from ml.table_one import format_pvalue
from datetime import datetime

logger = logging.getLogger(__name__)


def compile_latex_to_pdf(latex_source: str, timeout: int = 30) -> Optional[bytes]:
    """Compile a LaTeX source string to PDF bytes using pdflatex.

    Returns the PDF bytes, or None if pdflatex is unavailable or compilation
    failed. Runs pdflatex twice so cross-references resolve.

    Used by Page 10 to render the in-app PDF preview, and by integration
    tests to verify the compile pipeline end-to-end.
    """
    if not shutil.which("pdflatex"):
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "manuscript.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_source)
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                    capture_output=True, text=True, timeout=timeout,
                )
            pdf_path = os.path.join(tmpdir, "manuscript.pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    return f.read()
    except Exception as exc:
        logger.debug("PDF compilation failed: %s", exc)
    return None


def _normalize_generated_latex_text(text: str) -> str:
    """Clean minor generation artifacts without changing substantive content."""
    if not text:
        return ""

    replacements = {
        "mainresults": "main results",
        "PriorWork": "Prior Work",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"due to the([^\n.]+)\) due to", r"due to the\1) because", cleaned)
    # Guard against occasional fused-word artifacts in generated handoff text.
    # Match "and"/"with" fused to the next word, but not inside legitimate words
    # like "random", "withhold", "android", "standard", "mandate", "bandwidth".
    cleaned = re.sub(r"\band(?!om|roid|ard|ate|width|rew)(?=[a-z]{4,})", "and ", cleaned)
    cleaned = re.sub(r"\bwith(?!hold|out|in\b|draw|stand|er\b|al\b)(?=[a-z]{4,})", "with ", cleaned)
    # Fix accidentally doubled words, both fused ("waswas") and spaced ("was was")
    cleaned = re.sub(r'\b(was|the|of|in|to|and|for|is|on|at|by|an|or|as|it|that|from|with|this|were|are|been|has|had|have|not|but|all|can|its|may|will|one|our|out|per)\s*\1\b', r'\1', cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_metrics_list(model_results: Optional[Dict[str, Dict]], task_type: str = "regression") -> str:
    """Return a human-readable list of metrics known to be available."""
    if not model_results:
        return "RMSE, MAE, R$^2$, and MedianAE" if task_type == "regression" else "accuracy, F1 score, and AUC"

    preferred = ["RMSE", "MAE", "R2", "MedianAE"] if task_type == "regression" else ["Accuracy", "F1", "AUC"]
    present = []
    seen = set()
    for results in model_results.values():
        for metric in results.get("metrics", {}).keys():
            if metric not in seen:
                present.append(metric)
                seen.add(metric)

    ordered = [metric for metric in preferred if metric in seen] + [metric for metric in present if metric not in preferred]
    display_map = {"R2": "R$^2$", "F1": "F1 score", "AUC": "AUC"}
    return ", ".join(display_map.get(metric, metric) for metric in ordered) if ordered else (
        "RMSE, MAE, R$^2$, and MedianAE" if task_type == "regression" else "accuracy, F1 score, and AUC"
    )


def _resolve_latex_manuscript_context(
    manuscript_context: Optional[Dict[str, Any]],
    model_results: Optional[Dict[str, Dict]],
    bootstrap_results: Optional[Dict],
    feature_names: Optional[List[str]],
) -> Dict[str, Any]:
    """Prefer export-frozen manuscript facts over live/default arguments."""
    context = manuscript_context or {}
    selected_model_results = context.get('selected_model_results')
    selected_bootstrap_results = context.get('selected_bootstrap_results')
    feature_names_for_manuscript = context.get('feature_names_for_manuscript')
    return {
        'model_results': selected_model_results if selected_model_results is not None else model_results,
        'bootstrap_results': selected_bootstrap_results if selected_bootstrap_results is not None else bootstrap_results,
        'feature_names': list(feature_names_for_manuscript) if feature_names_for_manuscript is not None else feature_names,
        'manuscript_primary_model': context.get('manuscript_primary_model'),
        'best_model_by_metric': context.get('best_model_by_metric'),
        'best_metric_name': context.get('best_metric_name'),
        'feature_counts': dict(context.get('feature_counts') or {}),
        'population_counts': dict(context.get('population_counts') or {}),
        # `MODELS-009`: the Results section reads `manuscript_facts` for the
        # baseline comparison and this resolver never put it there, so the
        # "null and simple baselines on the same held-out test set" sentence —
        # the anchor for "is the model better than trivial?" — was composed from
        # an empty dict in every run and never printed at all.
        'baseline_results': dict(context.get('baseline_results') or {}),
    }


def _demote_results_subsections(results_latex: str) -> str:
    """Keep draft-results detail without colliding with manuscript section structure."""
    if not results_latex:
        return ""

    normalized = _normalize_generated_latex_text(results_latex)
    normalized = re.sub(r"\\subsection\{([^}]*)\}", r"\\paragraph{\1}", normalized)
    return normalized.strip()


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '<': r'\textless{}', '>': r'\textgreater{}',
    }
    for char, replacement in chars.items():
        text = text.replace(char, replacement)
    return text


def _styled_placeholder(text: str) -> str:
    """Wrap placeholder/investigator markers in deliberate LaTeX formatting.

    Preserves the literal marker text (e.g., '[PLACEHOLDER: ...]') so that
    existing tests matching on those strings continue to pass.
    """
    return rf"\textcolor{{gray}}{{\textit{{{text}}}}}"


def _format_abstract_population_sentence(upload_n: int, analysis_n: int) -> str:
    """Describe the abstract population without contradicting split counts."""
    if analysis_n and upload_n and analysis_n != upload_n:
        sentence = (
            f"Of {upload_n:,} observations, {analysis_n:,} remained for analysis "
            "after trimming/exclusion criteria were applied prior to splitting."
        )
    else:
        total = analysis_n or upload_n
        sentence = f"A total of {total:,} observations were available for analysis."
    # A cohort run makes this N the GROUP's, not the study's, and stating it
    # bare tells a reviewer the model was fitted on everyone. Appended after
    # BOTH branches — the first version of this fix patched only one, and the
    # exclusion-criteria wording is exactly where a restricted N is most
    # likely to be mistaken for the full study.
    from utils.workflow_provenance import cohort_restriction_sentence
    restriction = cohort_restriction_sentence()
    return f"{sentence} {restriction}".strip() if restriction else sentence


def _format_abstract_predictor_sentence(feature_counts: Dict[str, Any], feature_names: Optional[List[str]]) -> str:
    """Describe predictor counts consistently in the abstract."""
    selected_count = feature_counts.get('selected') or (len(feature_names) if feature_names else 0)
    original_count = feature_counts.get('original')
    candidate_count = feature_counts.get('candidate')

    # `DRIVE8-21`: the FE clause is written only where the record holds
    # engineered columns. `candidate != original` alone is not that evidence.
    from ml.publication import feature_engineering_ran
    if (original_count and candidate_count and selected_count
            and candidate_count != original_count and selected_count != candidate_count
            and feature_engineering_ran(feature_counts)):
        return (
            f"The raw dataset contained {original_count} predictor variables, "
            f"feature engineering yielded {candidate_count} candidates, and "
            f"feature selection retained {selected_count} predictors for final modeling."
        )
    if original_count and selected_count and selected_count != original_count:
        return (
            f"The workflow began with {original_count} predictor variables and retained "
            f"{selected_count} predictors for final modeling."
        )
    return f"The final modeling set contained {selected_count or 'N'} predictors."


_CALIBRATION_METRIC_LABELS = (
    ('brier_score', 'Brier score', 4),
    ('ece', 'expected calibration error', 4),
    ('mce', 'maximum calibration error', 4),
    ('c_statistic', 'c-statistic', 3),
    ('weak_slope', 'weak calibration slope', 3),
    ('weak_intercept', 'weak calibration intercept', 3),
    ('calibration_slope', 'calibration slope', 3),
    ('calibration_intercept', 'calibration intercept', 3),
    ('calibration_r2', r'calibration $R^2$', 3),
)


def _calibration_prose_by_model(
    calibration_by_model: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    """One Calibration sentence per model that has computed artifacts.

    `MISC-102`. Only the metrics present on a record are named — the
    classification and regression records carry different quantities, and the
    weak-calibration pair keeps its own name so it is not read as the
    observed-on-predicted regression slope.
    """
    if not calibration_by_model:
        return []
    lines: List[str] = []
    for model_key, metrics in calibration_by_model.items():
        if not isinstance(metrics, dict):
            continue
        parts = []
        for field_name, label, digits in _CALIBRATION_METRIC_LABELS:
            value = metrics.get(field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            parts.append(f"{label} = {float(value):.{digits}f}")
        if not parts:
            continue
        lines.append(
            f"For \\textbf{{{_escape_latex(_model_display_name(model_key))}}}, "
            f"{'; '.join(parts)}."
        )
    return lines


def _model_display_name(key: Optional[str]) -> str:
    """Return a human-readable model label without hard depending on Streamlit modules."""
    if not key:
        return "the selected model"

    try:
        from utils.insight_ledger import model_display_name as ledger_model_display_name
        return ledger_model_display_name(key)
    except Exception:
        fallback_names = {
            "ridge": "Ridge Regression",
            "lasso": "Lasso Regression",
            "elasticnet": "Elastic Net",
            "rf": "Random Forest",
            "extratrees_reg": "Extra Trees (Regressor)",
            "extratrees_clf": "Extra Trees (Classifier)",
            "histgb_reg": "HistGradientBoosting (Regressor)",
            "histgb_clf": "HistGradientBoosting (Classifier)",
            "nn": "Neural Network (MLP)",
            "knn_reg": "k-Nearest Neighbors (Regressor)",
            "knn_clf": "k-Nearest Neighbors (Classifier)",
            "svr": "Support Vector Regressor",
            "svc": "Support Vector Classifier",
            "naive_bayes": "Naive Bayes",
            "gaussian_nb": "Gaussian Naive Bayes",
            "lda": "Linear Discriminant Analysis",
            "xgb_reg": "XGBoost (Regressor)",
            "xgb_clf": "XGBoost (Classifier)",
            "lgbm_reg": "LightGBM (Regressor)",
            "lgbm_clf": "LightGBM (Classifier)",
        }
        return fallback_names.get(str(key).lower(), str(key).upper())


def _human_join(items: List[str]) -> str:
    """Join short lists into manuscript-style prose."""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _build_structured_abstract_sections(
    task_type: str,
    target_name: str,
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    model_results: Optional[Dict[str, Dict]],
    bootstrap_results: Optional[Dict],
    manuscript_context: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build a structured abstract scaffold from workflow facts."""
    manuscript_facts = _resolve_latex_manuscript_context(
        manuscript_context,
        model_results,
        bootstrap_results,
        feature_names,
    )
    resolved_results = manuscript_facts.get('model_results') or {}
    resolved_bootstrap = manuscript_facts.get('bootstrap_results') or {}
    resolved_feature_names = manuscript_facts.get('feature_names') or feature_names
    feature_counts = manuscript_facts.get('feature_counts', {})
    context = manuscript_context or {}
    population_counts = context.get('population_counts', {})
    upload_n = population_counts.get('upload_total') or n_total
    analysis_n = population_counts.get('analysis_total') or (n_train + n_val + n_test)
    dataset_descriptor = context.get('dataset_descriptor')
    cohort_type = context.get('cohort_type')
    target_stats = context.get('target_stats') or {}
    top_features = context.get('top_features') or []

    best_model_key = (
        manuscript_facts.get('manuscript_primary_model')
        or manuscript_facts.get('best_model_by_metric')
        or next(iter(resolved_results.keys()), None)
    )
    best_result = resolved_results.get(best_model_key, {}) if best_model_key else {}
    metrics_dict = best_result.get('metrics', {})

    background_bits = ["[INVESTIGATOR: describe the clinical or scientific question and why it matters]."]
    if dataset_descriptor and str(dataset_descriptor).lower() not in {"unknown", "none"}:
        background_bits.append(f"Data were drawn from {dataset_descriptor}.")
    if cohort_type:
        background_bits.append(f"The workflow treated the dataset as a {str(cohort_type).replace('_', ' ')} cohort.")

    methods_sentence = (
        f"{_format_abstract_population_sentence(upload_n, analysis_n)} "
        f"{_format_abstract_predictor_sentence(feature_counts, resolved_feature_names)} "
        f"These observations were split into training (n={n_train:,}), validation (n={n_val:,}), "
        f"and test (n={n_test:,}) sets. {len(resolved_results)} models were compared."
    )

    results_bits = []
    if best_model_key and metrics_dict:
        model_label = _model_display_name(best_model_key)
        if task_type == 'regression':
            rmse = metrics_dict.get('RMSE')
            r2 = metrics_dict.get('R2')
            if rmse is not None:
                rmse_str = f"{rmse:.4f}"
                ci = resolved_bootstrap.get(best_model_key, {}).get('RMSE')
                if ci and hasattr(ci, 'ci_lower') and hasattr(ci, 'ci_upper'):
                    rmse_str += f" (95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}])"
                results_bits.append(f"The best-performing model was {model_label} (RMSE {rmse_str}).")
                std_val = target_stats.get('std')
                if std_val:
                    results_bits.append(
                        f"This corresponded to approximately {rmse / std_val:.2f} SD of the outcome distribution."
                    )
            if r2 is not None:
                results_bits.append(
                    f"The model explained {int(round(r2 * 100))}% of outcome variance, leaving "
                    f"{int(round((1 - r2) * 100))}% unexplained."
                )
        else:
            primary_metric = 'F1' if metrics_dict.get('F1') is not None else 'Accuracy'
            primary_val = metrics_dict.get(primary_metric)
            if primary_val is not None:
                results_bits.append(
                    f"The best-performing model was {model_label} ({primary_metric} {primary_val:.4f})."
                )
            auc = metrics_dict.get('AUC')
            if auc is not None:
                results_bits.append(f"Discrimination was supported by an AUC of {auc:.4f}.")

    if top_features:
        results_bits.append(
            f"The most influential predictors included {_human_join(top_features[:3])}."
        )
    if not results_bits:
        results_bits.append("[INVESTIGATOR: summarize the key results and any leading predictors].")

    conclusions_bits = [
        "[INVESTIGATOR: interpret the practical importance of these findings and note key limitations]."
    ]
    if task_type == 'regression':
        conclusions_bits.insert(0, "These results indicate measurable predictive signal but still leave substantial unexplained variation.")
    else:
        conclusions_bits.insert(0, "These results indicate measurable predictive signal that requires domain interpretation before use.")

    return {
        'background_objective': " ".join(background_bits),
        'methods': methods_sentence,
        'results': " ".join(results_bits),
        'conclusions': " ".join(conclusions_bits),
    }


#: Top-level markdown headings the LaTeX skeleton has a home for. A draft
#: section whose heading is not one of these is compiled material with nowhere
#: to go, and it is printed under its own heading rather than dropped.
_DRAFT_SECTION_ALIASES = {
    'methods': 'methods',
    'results': 'results',
    'discussion': 'discussion',
}


def _draft_section_key(heading: str) -> Optional[str]:
    """Which manuscript section a `## ` heading names, if any."""
    normalized = re.sub(r'\(.*?\)', '', heading).strip().strip(':').lower()
    return _DRAFT_SECTION_ALIASES.get(normalized)


def _split_draft_markdown(markdown_text: str) -> List[Tuple[str, str]]:
    """The draft's top-level sections, in order, as `(heading, body)`.

    The heading is `""` for the text before the first `## ` — the ownership
    preamble the narrative engine puts there.
    """
    sections: List[Tuple[str, str]] = []
    last_heading = ""
    last_start = 0
    for match in re.finditer(r'(?m)^##[ \t]+(.+?)[ \t]*$', markdown_text):
        body = markdown_text[last_start:match.start()]
        if last_heading or body.strip():
            sections.append((last_heading, body))
        last_heading = match.group(1).strip()
        last_start = match.end()
    tail = markdown_text[last_start:]
    if last_heading or tail.strip():
        sections.append((last_heading, tail))
    return sections


def _convert_markdown_to_latex(markdown_text: str) -> Dict[str, Any]:
    """Convert the compiled markdown draft to LaTeX, section by section.

    `RECORD-007`: this used to split the draft at the FIRST `## Results` and
    return two strings — everything after that heading, Discussion included,
    became `results_latex`, which the caller appended only when there were no
    model results. In every real run there are model results, so the compiled
    Discussion, with every `[AUTHOR REQUIRED]` scaffold the workflow generated,
    was dropped from the .tex while the .md kept it. The two files in one ZIP
    then made different claims, and the honesty guards the project tests
    applied to the markdown only.

    Sectioning is now structural: the draft's `## ` headings are matched to the
    manuscript sections, and a heading with no home comes back under
    ``unmapped`` so the caller can print it instead of losing it.

    Returns:
        ``{'methods': str, 'results': str, 'discussion': str,
           'unmapped': List[Tuple[str, str]]}`` — LaTeX, not markdown.
    """
    out: Dict[str, Any] = {'methods': "", 'results': "", 'discussion': "",
                           'unmapped': []}
    if not markdown_text:
        return out

    def convert_section(md_text):
        if not md_text:
            return ""
        
        sections = []
        
        # Split on ### headers (handle both \n### and ^###)
        parts = re.split(r'(?:\n|^)### ', md_text)
        
        # First part (before any ###) is intro text
        if parts[0].strip():
            intro = parts[0].strip()
            # Remove markdown separators and stray escaped subsection typos from upstream text
            intro = re.sub(r'\n?---\s*\n?', '\n\n', intro).strip()
            intro = re.sub(r'(?m)^##\s+[^\n]+\s*$', '', intro).strip()
            intro = intro.replace('\\subelection', '\\subsection')
            # Convert markdown formatting (this handles escaping internally)
            intro_processed = _convert_inline_markdown(intro)
            # Escape any remaining text that wasn't in markdown formatting
            # We need to escape text NOT inside LaTeX commands
            intro_final = _escape_remaining_text(intro_processed)
            sections.append(intro_final)
        
        # Process each subsection
        for part in parts[1:]:
            lines = part.split('\n', 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            
            # Remove markdown separators (anywhere in body)
            body = re.sub(r'\n?---\s*\n?', '\n\n', body)
            body = body.replace('\\subelection', '\\subsection')
            body = body.strip()
            
            # Convert inline markdown (handles escaping internally)
            title_processed = _convert_inline_markdown(title)
            body_processed = _convert_inline_markdown(body)
            
            # Escape remaining text
            title_final = _escape_remaining_text(title_processed)
            body_final = _escape_remaining_text(body_processed)
            
            # Create subsection
            sections.append(f"\\subsection{{{title_final}}}\n\n{body_final}")
        
        return "\n\n".join(sections)
    
    def _escape_remaining_text(text):
        """Escape text that's not already inside LaTeX commands."""
        # Split on LaTeX commands (\textbf{...}, \texttt{...}, etc.)
        # This is a simple approach: find all LaTeX command blocks and escape everything else
        result = []
        last_end = 0
        
        # Find all LaTeX commands
        for match in re.finditer(r'\\(?:textbf|texttt|textit|emph)\{[^}]*\}', text):
            # Escape text before this command
            before = text[last_end:match.start()]
            result.append(_escape_latex(before))
            # Keep the command as-is
            result.append(match.group(0))
            last_end = match.end()
        
        # Escape any remaining text
        result.append(_escape_latex(text[last_end:]))
        
        return ''.join(result)
    
    def _convert_inline_markdown(text):
        """Convert markdown inline formatting to LaTeX.
        
        This function must be called BEFORE _escape_latex to preserve LaTeX commands.
        """
        # Convert **bold** to \textbf{bold} - escape the content
        def escape_bold(match):
            content = _escape_latex(match.group(1))
            return f"\\textbf{{{content}}}"
        text = re.sub(r'\*\*(.+?)\*\*', escape_bold, text)
        
        # Convert `code` to \texttt{code} - escape the content
        def escape_code(match):
            content = _escape_latex(match.group(1))
            return f"\\texttt{{{content}}}"
        text = re.sub(r'`(.+?)`', escape_code, text)
        
        return text
    
    for heading, body in _split_draft_markdown(markdown_text):
        # The text before the first heading is the draft's ownership preamble;
        # it introduces the Methods, which is where it already appeared.
        key = 'methods' if heading == "" else _draft_section_key(heading)
        latex = convert_section(body)
        if not latex.strip():
            continue
        if key is None:
            out['unmapped'].append((heading, latex))
        elif out[key]:
            out[key] = out[key] + "\n\n" + latex
        else:
            out[key] = latex

    return out


def _metrics_to_latex_table(
    model_results: Dict[str, Dict],
    task_type: str = "regression",
    bootstrap_results: Optional[Dict] = None,
) -> str:
    """Generate a LaTeX metrics comparison table with width containment."""
    if task_type == "regression":
        metric_names = ["RMSE", "MAE", "R2", "MedianAE"]
        caption = "Model performance on the held-out test set (regression metrics)."
    else:
        metric_names = ["Accuracy", "F1", "AUC"]
        caption = "Model performance on the held-out test set (classification metrics)."

    # Determine which metrics are actually present
    all_metrics = set()
    for res in model_results.values():
        all_metrics.update(res.get("metrics", {}).keys())
    metric_names = [m for m in metric_names if m in all_metrics]

    if not metric_names:
        return ""

    n_models = len(model_results)
    n_metrics = len(metric_names)
    is_wide = n_metrics > 4 or n_models > 4
    
    # Use p{3.5cm} for first column to allow wrapping without cramping model names
    col_spec = "p{3.5cm}" + "c" * len(metric_names)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\begin{spacing}{1.0}")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:model_performance}")

    # Width containment: wrap in adjustbox
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")

    # Use smaller font for wide tables
    if is_wide:
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{5pt}")
    else:
        lines.append(r"\small")
    
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Model & " + " & ".join(metric_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for name, res in model_results.items():
        metrics = res.get("metrics", {})
        cis = {}
        if bootstrap_results and name in bootstrap_results:
            cis = bootstrap_results[name]

        cells = [_escape_latex(_model_display_name(name))]
        for m in metric_names:
            val = metrics.get(m)
            ci = cis.get(m)
            if val is not None:
                if ci and hasattr(ci, 'ci_lower') and hasattr(ci, 'ci_upper'):
                    cells.append(f"{val:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")

    # `MINE-027`: a cell in this table is not the held-out score unless the N it
    # was computed on is the held-out N. When a degenerate target back-transform
    # made some predictions non-finite, those pairs were dropped before scoring
    # and the truncation reached a Streamlit warning and nothing else — least of
    # all the table a reviewer reads. The count is NOT a metric column (it was,
    # briefly, and rendered as a model score); it is a footnote to the numbers
    # it qualifies.
    notes = []
    for name, res in model_results.items():
        disclosure = res.get("test_scoring") or {}
        if disclosure.get("n_dropped_nonfinite"):
            notes.append(
                f"{_escape_latex(_model_display_name(name))}: computed on "
                f"{disclosure['n_scored']} of {disclosure.get('n_pairs', '?')} "
                f"pairs; {disclosure['n_dropped_nonfinite']} non-finite "
                f"pair(s) excluded")
    if notes:
        lines.append(r"\vspace{2pt}")
        lines.append(r"{\footnotesize \textit{Note:} " + "; ".join(notes) + ".}")

    lines.append(r"\end{spacing}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _seed_stability_table_latex(by_model: List[Dict[str, Any]]) -> str:
    """The per-model seed spread page 08 shows on screen, as a LaTeX table.

    `MISC-104`. Page 08 sweeps every eligible model and renders mean / SD /
    range / CV per model; the export carried one model's numbers and named no
    model. Emitted only when more than one model was swept — for a single model
    the sentence above already says it.
    """
    rows = [row for row in (by_model or []) if isinstance(row, dict)]
    if len(rows) < 2:
        return ""

    metric = str(rows[0].get('metric') or 'the primary metric')
    lines = [
        r"\begin{table}[htbp]",
        r"\begin{spacing}{1.0}",
        r"\centering",
        r"\caption{Across-seed " + _escape_latex(metric)
        + " by model (fresh split per seed).}",
        r"\label{tab:seed-stability}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Seeds & Mean & SD & Range & CV (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join([
                _escape_latex(_model_display_name(str(row.get('model') or ''))),
                str(int(row.get('n_seeds') or 0)),
                f"{float(row.get('mean', 0.0)):.4f}",
                f"{float(row.get('sd', 0.0)):.4f}",
                f"{float(row.get('min', 0.0)):.4f}--{float(row.get('max', 0.0)):.4f}",
                f"{float(row.get('cv_percent', 0.0)):.1f}",
            ]) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{spacing}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _table1_to_latex(
    table1_df: pd.DataFrame,
    footnotes: Optional[List[str]] = None,
) -> str:
    """Convert Table 1 DataFrame to LaTeX with width containment.

    `MISC-104`: `footnotes` are the custom-test notes pages/10 writes when it
    appends `^N` markers to row labels. Without them the markers were dangling
    superscripts — a reference to a note the manuscript did not contain.
    """
    if table1_df is None or table1_df.empty:
        return ""

    n_cols = len(table1_df.columns)
    is_wide = n_cols > 4
    
    # Use p{4cm} for first column (Characteristic) to allow wrapping
    col_spec = "p{4cm}" + "c" * n_cols

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\begin{spacing}{1.0}")
    lines.append(r"\centering")
    lines.append(r"\caption{Characteristics of the study population.}")
    lines.append(r"\label{tab:table1}")

    # Width containment: wrap in adjustbox
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")

    # Use smaller font for wide tables
    if is_wide:
        lines.append(r"\small")
    
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = "Characteristic & " + " & ".join(_escape_latex(str(c)) for c in table1_df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for idx, row in table1_df.iterrows():
        cells = [_escape_latex(str(idx))]
        for val in row.values:
            cells.append(_escape_latex(str(val)) if val else "")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")

    clean_footnotes = [str(note).strip() for note in (footnotes or []) if str(note).strip()]
    if clean_footnotes:
        lines.append(r"\begin{flushleft}")
        lines.append(r"\footnotesize")
        for note in clean_footnotes:
            lines.append(_escape_latex(note) + r" \\")
        lines.append(r"\end{flushleft}")

    lines.append(r"\end{spacing}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_report(
    title: str = "Prediction Model Development and Validation",
    authors: str = "[Author Names]",
    affiliation: str = "[Institution]",
    abstract: str = "[ABSTRACT PLACEHOLDER]",
    methods_section: str = "",
    table1_df: Optional[pd.DataFrame] = None,
    table1_footnotes: Optional[List[str]] = None,
    model_results: Optional[Dict[str, Dict]] = None,
    bootstrap_results: Optional[Dict] = None,
    task_type: str = "regression",
    feature_names: Optional[List[str]] = None,
    target_name: str = "outcome",
    n_total: int = 0,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    tripod_checklist: Optional[pd.DataFrame] = None,
    data_config: Optional[Dict] = None,
    calibration_text: str = "",
    limitations: str = "[Discuss limitations here]",
    explainability_summary: Optional[Dict[str, Any]] = None,
    sensitivity_summary: Optional[Dict[str, Any]] = None,
    stat_validation_summary: Optional[List[Dict[str, Any]]] = None,
    manuscript_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a complete LaTeX manuscript template.

    Returns compilable LaTeX source populated with actual results.
    """
    manuscript_facts = _resolve_latex_manuscript_context(manuscript_context, model_results, bootstrap_results, feature_names)
    model_results = manuscript_facts['model_results']
    bootstrap_results = manuscript_facts['bootstrap_results']
    feature_names = manuscript_facts['feature_names']
    population_counts = manuscript_facts.get('population_counts', {})
    analysis_n = population_counts.get('analysis_total') or (n_train + n_val + n_test) or n_total

    sections = []

    # ── Preamble ──
    sections.append(r"""\documentclass[12pt, a4paper]{article}

% ── Packages ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathpazo}          % Palatino serif (submission-quality typography)
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black]{hyperref}
\usepackage{natbib}
\usepackage{float}
\usepackage{setspace}
\usepackage{caption}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{longtable}
\usepackage{microtype}         % Improved line-breaking and spacing
\usepackage{xcolor}            % For placeholder styling

\doublespacing

% ── Title ──""")

    sections.append(f"\\title{{{_escape_latex(title)}}}")
    sections.append(f"\\author{{{_escape_latex(authors)} \\\\ \\small{{{_escape_latex(affiliation)}}}}}")
    sections.append(r"\date{\today}")
    sections.append("")
    sections.append(r"\begin{document}")
    sections.append(r"\maketitle")
    sections.append(r"\thispagestyle{empty}")

    # ── Abstract ──
    # Auto-scaffold abstract from known facts
    if abstract == "[ABSTRACT PLACEHOLDER]" and model_results and (analysis_n > 0 or n_total > 0):
        abstract_sections = _build_structured_abstract_sections(
            task_type=task_type,
            target_name=target_name,
            n_total=n_total,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            model_results=model_results,
            bootstrap_results=bootstrap_results,
            manuscript_context=manuscript_context,
            feature_names=feature_names,
        )
        abs_parts = []
        abs_parts.append(r"\noindent \textbf{Background/Objective:} " + _escape_latex(abstract_sections['background_objective']))
        abs_parts.append(r"\textbf{Methods:} " + _escape_latex(abstract_sections['methods']))
        abs_parts.append(r"\textbf{Results:} " + _escape_latex(abstract_sections['results']))
        abs_parts.append(r"\textbf{Conclusions:} " + _escape_latex(abstract_sections['conclusions']))
        
        abstract = " ".join(abs_parts)
    
    sections.append(r"""
\begin{abstract}
\begin{spacing}{1.0}""")
    sections.append(f"{abstract if not abstract.startswith('[ABSTRACT') else _escape_latex(abstract)}")
    sections.append(r"""\end{spacing}
\end{abstract}

\vspace{1.5em}
""")

    # ── Introduction ──
    sections.append(r"\section{Introduction}")
    sections.append("")
    sections.append(_styled_placeholder("[PLACEHOLDER: Provide background on the clinical/research context and rationale for developing this prediction model. Cite relevant prior work.]"))
    sections.append(r"""
\subsection{Objectives}""")
    sections.append(_styled_placeholder("[PLACEHOLDER: State the specific objectives of this study, including whether you are developing, validating, or both.]"))
    sections.append("")

    # ── Methods ──
    sections.append(r"\section{Methods}")
    sections.append("")

    draft_discussion = ""
    draft_unmapped: List[Tuple[str, str]] = []
    if methods_section:
        # Convert markdown to LaTeX properly
        draft_latex = _convert_markdown_to_latex(methods_section)
        if draft_latex['methods']:
            sections.append(_normalize_generated_latex_text(draft_latex['methods']))
        # Held for the Results and Discussion sections below. Every compiled
        # section reaches the .tex or is printed as unmapped material — none of
        # it may be dropped on the way (RECORD-007).
        draft_results = _demote_results_subsections(draft_latex['results'])
        draft_discussion = _normalize_generated_latex_text(draft_latex['discussion']).strip()
        draft_unmapped = draft_latex['unmapped']
    else:
        draft_results = ""
        sections.append(r"""
\subsection{Study Design and Participants}""")
        sections.append(_styled_placeholder("[PLACEHOLDER: Describe the study design, data source, eligibility criteria, and key dates.]"))
        sections.append(r"""
\subsection{Outcome Definition}
""")
        sections.append(f"The outcome variable was {_escape_latex(target_name)}.")
        sections.append(r"""
\subsection{Predictor Variables}""")
        if feature_names:
            if len(feature_names) <= 15:
                feat_list = ", ".join(_escape_latex(f) for f in feature_names)
                sections.append(f"The following {len(feature_names)} predictor variables were included: {feat_list}.")
            else:
                sections.append(f"A total of {len(feature_names)} predictor variables were included (see Supplementary Table S1).")

        sections.append(r"""
\subsection{Missing Data}""")
        sections.append(_styled_placeholder("[PLACEHOLDER: Describe how missing data were handled, including the mechanism (MCAR/MAR/MNAR) and imputation strategy.]"))
        sections.append(r"""
\subsection{Model Development}""")
        sections.append(_styled_placeholder("[PLACEHOLDER: Describe preprocessing, model selection, and internal validation strategy.]"))
        sections.append("")
        if n_total > 0:
            sections.append(f"Data were split into training (n={n_train:,}), validation (n={n_val:,}), and test (n={n_test:,}) sets.")

        metrics_text = _format_metrics_list(model_results, task_type)
        sections.append(r"""
\subsection{Performance Evaluation}
""")
        sections.append(
            f"Model performance was assessed using {metrics_text} with 95\\% confidence intervals computed via 1,000 BCa bootstrap resamples."
        )

    # ── Results ──
    sections.append(r"""
\section{Results}

\subsection{Study Population}""")

    if analysis_n > 0:
        sections.append(f"A total of {analysis_n:,} participants were included in the analysis.")
    from utils.workflow_provenance import cohort_restriction_sentence
    _restriction = cohort_restriction_sentence()
    if _restriction:
        sections.append(_escape_latex(_restriction))

    # Table 1
    if table1_df is not None and not table1_df.empty:
        sections.append(_table1_to_latex(table1_df, table1_footnotes))
    else:
        sections.append(_styled_placeholder("[INSERT TABLE 1: Characteristics of the study population]"))

    # Model Performance — avoid duplicating a prose dump when the structured table is present.
    sections.append(r"""
\subsection{Model Performance}""")

    if model_results:
        primary_model = manuscript_facts.get('manuscript_primary_model')
        best_model_by_metric = manuscript_facts.get('best_model_by_metric')
        best_metric_name = manuscript_facts.get('best_metric_name') or 'held-out metric'
        if primary_model:
            sections.append(f"The manuscript-primary model was \\textbf{{{_escape_latex(_model_display_name(primary_model))}}}.")
            if best_model_by_metric and best_model_by_metric != primary_model:
                sections.append(
                    f"The best model by {_escape_latex(best_metric_name)} was \\textbf{{{_escape_latex(_model_display_name(best_model_by_metric))}}}."
                )
        elif best_model_by_metric:
            sections.append(
                f"The best model by {_escape_latex(best_metric_name)} was \\textbf{{{_escape_latex(_model_display_name(best_model_by_metric))}}}. "
                "No manuscript-primary model was explicitly selected in the workflow."
            )
        sections.append("Table \\ref{tab:model_performance} summarizes held-out performance across the evaluated models.")
        sections.append(_metrics_to_latex_table(model_results, task_type, bootstrap_results))

        baselines = manuscript_facts.get('baseline_results') or {}
        if baselines:
            b_parts = []
            b_recipes = []
            for bname, bdata in baselines.items():
                metrics = (bdata or {}).get('metrics', {})
                m_txt = ", ".join(f"{_escape_latex(str(k))} = {v:.4f}" for k, v in metrics.items())
                b_parts.append(f"{_escape_latex(str(bname))} ({m_txt})")
                recipe = (bdata or {}).get('preprocessing')
                if recipe and recipe not in b_recipes:
                    b_recipes.append(recipe)
            sections.append(
                "For reference, null and simple baseline models evaluated on the same "
                "held-out test set achieved: " + "; ".join(b_parts) + "."
            )
            # `MODELS-009`: "the same held-out test set" is true and "the same
            # preprocessing" is not — the baselines have their own fixed recipe.
            # A comparison whose recipe is unstated invites the reader to assume
            # the models' one.
            if b_recipes:
                sections.append(
                    "The baseline features went through their own fixed recipe ("
                    + "; ".join(_escape_latex(str(r)) for r in b_recipes)
                    + "), not the per-model preprocessing pipelines above."
                )

        # The table above is generated from the same recorded results, so
        # reproducing the draft's Results prose would state the numbers twice.
        # It stays omitted — but the omission is STATED, and any author-input
        # scaffold the prose carries is reproduced: a compiled passage that
        # vanishes between the .md and the .tex is what RECORD-007 is about.
        if draft_results:
            sections.append(
                "The compiled draft's Results narrative is not reproduced here; "
                "it accompanies this manuscript in the exported markdown draft."
            )
            outstanding = [p.strip() for p in re.split(r'\n\s*\n', draft_results)
                           if '[AUTHOR REQUIRED' in p]
            if outstanding:
                sections.append(r"\paragraph{Outstanding author input}")
                sections.extend(outstanding)
            sections.append("")
    elif draft_results:
        sections.append(draft_results)
        sections.append("\n")
    else:
        sections.append(_styled_placeholder(r"[INSERT TABLE: Model performance metrics with 95\% CIs]"))

    # Calibration
    calibration_prose = _calibration_prose_by_model(
        (explainability_summary or {}).get('calibration_by_model'))
    if calibration_text:
        sections.append(r"""
\subsection{Calibration}""")
        sections.append(_escape_latex(calibration_text))
    elif calibration_prose:
        # `MISC-102`: the placeholder used to stand even when page 06 had
        # calibrated every model, so the export asked its own author to supply
        # numbers the session already held. The placeholder still stands when
        # nothing was computed — that is a real absence.
        sections.append(r"""
\subsection{Calibration}""")
        sections.extend(calibration_prose)
        sections.append("")
    else:
        if task_type == "regression":
            sections.append(r"""
\subsection{Calibration}""")
            sections.append(_styled_placeholder(r"[PLACEHOLDER: Report calibration results --- calibration slope/intercept, predicted vs.\ observed plots, residual diagnostics. Include calibration plot as a figure.]"))
            sections.append("")
        else:
            sections.append(r"""
\subsection{Calibration}""")
            sections.append(_styled_placeholder("[PLACEHOLDER: Report calibration results --- Brier score, ECE, calibration slope/intercept. Include calibration plot as a figure.]"))
            sections.append("")
    
    # Explainability results (if provided)
    if explainability_summary:
        sections.append(r"""
\subsection{Feature Importance and Explainability}""")
        
        # Feature importance (top features)
        top_features = explainability_summary.get('top_features', [])
        if top_features:
            feat_list = ", ".join(_escape_latex(f) for f in top_features[:5])
            sections.append(f"The most important predictors were: {feat_list}.")
            sections.append("")
        
        # Permutation importance availability
        if explainability_summary.get('permutation_importance_available'):
            sections.append("Permutation importance analysis was conducted to assess feature contributions.")
            sections.append("")
        
        # SHAP availability
        if explainability_summary.get('shap_available'):
            sections.append("SHAP (SHapley Additive exPlanations) analysis was performed to explain individual predictions.")
            sections.append("")
        
        # Calibration metrics (if not already reported above)
        calibration_metrics = explainability_summary.get('calibration_metrics')
        if calibration_metrics and not calibration_text and not calibration_prose:
            sections.append(r"\paragraph{Calibration Metrics}")
            for metric_name, metric_val in calibration_metrics.items():
                sections.append(f"{_escape_latex(metric_name)}: {metric_val:.4f}. ")
            sections.append("")
    
    # Sensitivity analysis results (if provided)
    if sensitivity_summary:
        sections.append(r"""
\subsection{Sensitivity Analysis}""")
        
        # Seed stability
        seed_stability = sensitivity_summary.get('seed_stability')
        if seed_stability:
            cv_pct = seed_stability.get('cv_percent')
            metric_range = seed_stability.get('range')
            # `MISC-104`: whose coefficient of variation, of what. Page 08
            # re-seeds every eligible model; a bare percentage in a manuscript
            # that reports five models reads as a statement about all of them.
            # The model and metric are named when the export knows them, and
            # the sentence stays as it was when it does not.
            seed_model = seed_stability.get('model')
            seed_metric = seed_stability.get('metric')
            subject = "Random seed sensitivity analysis"
            if seed_model:
                subject += f" of \\textbf{{{_escape_latex(_model_display_name(seed_model))}}}"
            if seed_metric:
                subject += f" ({_escape_latex(str(seed_metric))})"
            if cv_pct is not None:
                n_seeds = seed_stability.get('n_seeds')
                seeds_clause = f" across {n_seeds} seeds" if n_seeds else " across seeds"
                sections.append(
                    f"{subject} showed a coefficient of variation of "
                    f"{cv_pct:.1f}\\%{seeds_clause}.")
            if metric_range:
                sections.append(f"Performance range: {metric_range}.")
            sections.append("")

            by_model = seed_stability.get('by_model')
            if by_model:
                sections.append(_seed_stability_table_latex(by_model))
                sections.append("")
        
        # Feature dropout
        if sensitivity_summary.get('feature_dropout_conducted'):
            sections.append("Feature dropout sensitivity analysis was conducted to assess model robustness to missing predictors.")
            sections.append("")
        
        sections.append(_styled_placeholder("[PLACEHOLDER: Interpret sensitivity results in context]"))
        sections.append("")

    # FIX 4: Statistical Validation subsection
    if stat_validation_summary:
        sections.append(r"""
\subsection{Statistical Validation}""")
        for entry in stat_validation_summary:
            test_name = _escape_latex(entry.get('test_name', 'Statistical test'))
            variable = _escape_latex(entry.get('variable', 'unknown variable'))
            statistic = entry.get('statistic')
            p_value = entry.get('p_value')
            
            if statistic is not None and p_value is not None:
                # `DRIVE8-32`: `:.4f` renders anything below 5e-5 as "0.0000",
                # which asserts p = 0. The floor is stated as an inequality.
                _p = format_pvalue(p_value)
                _p_clause = (f"$p$ {_p}" if _p.startswith("<")
                             else f"$p$ = {_p}")
                sections.append(f"{test_name} was performed on {variable}: statistic = {statistic:.4f}, {_p_clause}.")
                sections.append("")

        # Multiple testing caveat if >3 distinct comparisons. `DRIVE8-20`: an
        # override re-run of one comparison is not a second test.
        from ml.narrative_engine import _distinct_comparisons
        if len(_distinct_comparisons(stat_validation_summary)) > 3:
            sections.append(
                r"Note: Multiple statistical tests were performed; "
                r"readers should consider the increased risk of Type I error "
                r"when interpreting individual $p$-values."
            )
            sections.append("")

    # ── Discussion ──
    sections.append(r"""
\section{Discussion}""")

    # The compiled Discussion is the one the app vouches for: its Principal
    # Findings state the metric the way the honesty guards require (a negative
    # R² as "below a mean-only baseline"), and its Strengths and Limitations
    # carry the ledger's evidence-cited caveats and [AUTHOR REQUIRED] scaffolds.
    # It used to be dropped from the .tex in every run that had model results
    # (RECORD-007), leaving a Discussion rebuilt from generic placeholders while
    # the .md in the same ZIP said something else. The placeholder scaffold is
    # now the FALLBACK, printed only when nothing was compiled — the two name
    # the same subsections, so exactly one of them may appear.
    if draft_discussion:
        sections.append(draft_discussion)
        sections.append("")
        # A limitations paragraph the caller passed explicitly is a second
        # source and is appended rather than dropped.
        if limitations and limitations.strip() != "[Discuss limitations here]":
            sections.append(r"""
\paragraph{Limitations}
""")
            sections.append(_escape_latex(limitations))
            sections.append("")
    else:
        sections.append(r"""
\subsection{Principal Findings}""")

        # Result-specific prompts instead of generic placeholders
        best_model_key = manuscript_facts.get('manuscript_primary_model') or manuscript_facts.get('best_model_by_metric')
        if best_model_key and model_results and best_model_key in model_results:
            best_metrics = model_results[best_model_key].get('metrics', {})
            if task_type == 'regression':
                primary_metric = 'RMSE'
                primary_val = best_metrics.get('RMSE')
            else:
                primary_metric = 'F1' if 'F1' in best_metrics else 'Accuracy'
                primary_val = best_metrics.get(primary_metric)
        
            if primary_val is not None:
                sections.append(
                    f"The {_escape_latex(_model_display_name(best_model_key))} achieved {primary_metric} of {primary_val:.4f} on held-out data. "
                    + _styled_placeholder("[PLACEHOLDER: Interpret this performance in clinical context]")
                )
            else:
                sections.append(_styled_placeholder("[PLACEHOLDER: Summarize the main results in context of the study objectives.]"))
        else:
            sections.append(_styled_placeholder("[PLACEHOLDER: Summarize the main results in context of the study objectives.]"))
    
        sections.append("")
    
        # Feature importance interpretation prompt
        if explainability_summary and explainability_summary.get('top_features'):
            top_feats = explainability_summary['top_features'][:3]
            feat_str = ", ".join(_escape_latex(f) for f in top_feats)
            sections.append(f"Key predictors identified were {feat_str}. " + _styled_placeholder("[PLACEHOLDER: Discuss biological plausibility and consistency with prior knowledge]"))
            sections.append("")
    
        sections.append(r"""
\subsection{Comparison with Prior Work and Implications}""")
        if task_type and best_model_key and model_results:
            task_label = "regression" if task_type == "regression" else "classification"
            sections.append(_styled_placeholder(
                f"[PLACEHOLDER: Compare the {primary_metric if 'primary_metric' in locals() else 'performance'} "
                f"to published benchmarks for {task_label} in this domain and discuss practical or clinical implications.]"
            ))
        else:
            sections.append(_styled_placeholder("[PLACEHOLDER: Compare your results with existing literature and discuss implications.]"))
        sections.append(r"""
\subsection{Strengths and Limitations}

\paragraph{Strengths}""")
    
        # Auto-fill methodological strengths from what we know
        strength_items = []
        if analysis_n > 0:
            from utils.workflow_provenance import get_provenance as _gp
            try:
                _up = getattr(_gp(), "upload", None)
            except Exception:
                _up = None
            if _up is not None and getattr(_up, "cohort_column", ""):
                # Listing a restricted sample as a plain strength invites the reader
                # to treat it as the study's size. Name the group it is a sample of.
                strength_items.append(
                    f"Sample size of {analysis_n:,} observations within "
                    f"{_up.cohort_column} = {_up.cohort_value} "
                    f"(the analysis was restricted to this group)")
            else:
                strength_items.append(f"Sample size of {analysis_n:,} observations")
        if bootstrap_results:
            strength_items.append("Bootstrap confidence intervals for uncertainty quantification")
        if explainability_summary:
            if explainability_summary.get('shap_available'):
                strength_items.append("Model-agnostic explainability via SHAP analysis")
            if explainability_summary.get('permutation_importance_available'):
                strength_items.append("Permutation importance for feature contribution assessment")
        if sensitivity_summary and sensitivity_summary.get('seed_stability'):
            strength_items.append("Random seed sensitivity analysis for robustness assessment")
    
        strength_items = strength_items[:4]
        if strength_items:
            sections.append(r"\begin{itemize}")
            for item in strength_items:
                sections.append(f"\\item {item}")
            sections.append(r"\end{itemize}")
            sections.append("")
            sections.append(_styled_placeholder("[PLACEHOLDER: Add study-specific strengths]"))
        else:
            sections.append(_styled_placeholder("[PLACEHOLDER: Discuss methodological strengths]"))
    
        sections.append(r"""

\paragraph{Limitations}
""")
        sections.append(_escape_latex(limitations))
    
        sections.append(r"""

\subsection{Conclusion}""")
        sections.append(_styled_placeholder("[PLACEHOLDER: State the main conclusion and its implications.]"))
        sections.append("")
        sections.append("")

    # ── References ──
    sections.append(r"""
\section*{References}
\begin{spacing}{1.0}
\begin{enumerate}""")
    sections.append(r"\item " + _styled_placeholder("[PLACEHOLDER: Add references in journal format]"))
    sections.append(r"\item Collins GS, et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ. 2015;350:g7594.")
    sections.append(r"\item Steyerberg EW, et al. Assessing the performance of prediction models: a framework for some traditional and novel measures. Epidemiology. 2010;21(1):128--138.")
    sections.append(r"\item Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman \& Hall; 1993.")
    if explainability_summary and explainability_summary.get('shap_available'):
        sections.append(r"\item Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems. 2017;30.")
    sections.append(r"""\end{enumerate}
\end{spacing}
""")

    # ── Supplementary ──
    sections.append(r"""
\clearpage
\appendix
\begin{spacing}{1.0}
\renewcommand{\thesubsection}{S\arabic{subsection}}
\section{Supplementary Material}

\subsection{TRIPOD Checklist}""")
    sections.append(_styled_placeholder("[See exported TRIPOD checklist CSV/PDF]"))
    sections.append(r"""
\subsection{Reproducibility}
This analysis was conducted using Tabular ML Lab (Python). Full reproducibility manifest including software versions, random seeds, and data hashes is available in the exported analysis package.

""")

    # FIX 7: Decision Audit Trail in LaTeX supplementary
    from ml.publication import generate_decision_audit_trail
    audit_trail = generate_decision_audit_trail()
    if audit_trail:
        sections.append(r"\subsection{Decision Audit Trail}")
        sections.append(r"\small")
        sections.append("The following audit trail documents key methodological decisions made during the analysis workflow.")
        sections.append("")
        in_list = False
        # Parse the numbered list from audit_trail and convert to LaTeX enumerate
        for line in audit_trail.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("### "):
                if in_list:
                    sections.append(r"\end{enumerate}")
                    in_list = False
                sections.append(rf"\paragraph{{{_escape_latex(stripped[4:])}}}")
                sections.append(r"\begin{enumerate}")
                in_list = True
                continue

            match = re.match(r'^\d+\.\s*(.+)$', stripped)
            if match:
                if not in_list:
                    sections.append(r"\begin{enumerate}")
                    in_list = True
                content = _escape_latex(match.group(1))
                sections.append(f"\\item {content}")
        if in_list:
            sections.append(r"\end{enumerate}")
        sections.append("")

    # Compiled draft sections the manuscript skeleton has no home for. They are
    # printed here, named, rather than dropped on the way to the .tex: a section
    # this function does not recognize is a gap in the mapping, and the author
    # has to be able to see it (RECORD-007).
    if draft_unmapped:
        sections.append(r"\subsection{Unmapped Compiled Draft Sections}")
        sections.append(
            "The generated draft contained the following section(s), which this "
            "template has no place for. They are reproduced verbatim so that "
            "nothing compiled from the workflow is lost."
        )
        sections.append("")
        for _heading, _latex in draft_unmapped:
            sections.append(rf"\paragraph{{{_escape_latex(_heading)}}}")
            sections.append(_normalize_generated_latex_text(_latex))
            sections.append("")

    sections.append(r"\end{spacing}")
    sections.append(r"\end{document}")

    return "\n".join(sections)
