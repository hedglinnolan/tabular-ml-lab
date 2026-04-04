"""
LaTeX report generator.

Generates a complete LaTeX manuscript template populated with actual results
from the modeling workflow. Ready to compile with pdflatex.
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


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
        return (
            f"Of {upload_n:,} observations, {analysis_n:,} remained for analysis "
            "after trimming/exclusion criteria were applied prior to splitting."
        )
    total = analysis_n or upload_n
    return f"A total of {total:,} observations were available for analysis."


def _format_abstract_predictor_sentence(feature_counts: Dict[str, Any], feature_names: Optional[List[str]]) -> str:
    """Describe predictor counts consistently in the abstract."""
    selected_count = feature_counts.get('selected') or (len(feature_names) if feature_names else 0)
    original_count = feature_counts.get('original')
    candidate_count = feature_counts.get('candidate')

    if original_count and candidate_count and selected_count and candidate_count != original_count and selected_count != candidate_count:
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


def _convert_markdown_to_latex(markdown_text: str) -> Tuple[str, str]:
    """Convert markdown methods section to LaTeX, separating Methods and Results.
    
    Returns:
        Tuple of (methods_latex, results_latex)
    """
    if not markdown_text:
        return "", ""
    
    # Split on ## Results / ## Results (Draft) to separate Methods from Results
    parts = re.split(r'\n## Results(?:\s*\(Draft\))?.*?\n', markdown_text, maxsplit=1)
    methods_md = parts[0]
    results_md = parts[1] if len(parts) > 1 else ""
    
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
    
    methods_latex = convert_section(methods_md)
    results_latex = convert_section(results_md)
    
    return methods_latex, results_latex


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
    lines.append(r"\end{spacing}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _table1_to_latex(table1_df: pd.DataFrame) -> str:
    """Convert Table 1 DataFrame to LaTeX with width containment."""
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

    if methods_section:
        # Convert markdown to LaTeX properly
        methods_latex, results_latex = _convert_markdown_to_latex(methods_section)
        if methods_latex:
            sections.append(_normalize_generated_latex_text(methods_latex))
        # Store results_latex for later use in Results section
        draft_results = _demote_results_subsections(results_latex)
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

    # Table 1
    if table1_df is not None and not table1_df.empty:
        sections.append(_table1_to_latex(table1_df))
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
    elif draft_results:
        sections.append(draft_results)
        sections.append("\n")
    else:
        sections.append(_styled_placeholder(r"[INSERT TABLE: Model performance metrics with 95\% CIs]"))

    # Calibration
    if calibration_text:
        sections.append(r"""
\subsection{Calibration}""")
        sections.append(_escape_latex(calibration_text))
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
        if calibration_metrics and not calibration_text:
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
            if cv_pct is not None:
                sections.append(f"Random seed sensitivity analysis showed a coefficient of variation of {cv_pct:.1f}\\% across seeds.")
            if metric_range:
                sections.append(f"Performance range: {metric_range}.")
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
                sections.append(f"{test_name} was performed on {variable}: statistic = {statistic:.4f}, $p$ = {p_value:.4f}.")
                sections.append("")
        
        # Multiple testing caveat if >3 tests
        if len(stat_validation_summary) > 3:
            sections.append(
                r"Note: Multiple statistical tests were performed; "
                r"readers should consider the increased risk of Type I error "
                r"when interpreting individual $p$-values."
            )
            sections.append("")

    # ── Discussion ──
    sections.append(r"""
\section{Discussion}

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

    sections.append(r"\end{spacing}")
    sections.append(r"\end{document}")

    return "\n".join(sections)
