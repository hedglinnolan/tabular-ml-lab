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


def _convert_markdown_to_latex(markdown_text: str) -> Tuple[str, str]:
    """Convert markdown methods section to LaTeX, separating Methods and Results.
    
    Returns:
        Tuple of (methods_latex, results_latex)
    """
    if not markdown_text:
        return "", ""
    
    # Split on ## Results to separate Methods from Results
    parts = re.split(r'\n## Results.*?\n', markdown_text, maxsplit=1)
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
            # Remove --- separators
            intro = re.sub(r'\n?---\s*\n?', '\n\n', intro).strip()
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
            
            # Remove --- separators (anywhere in body)
            body = re.sub(r'\n?---\s*\n?', '\n\n', body)
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
    """Generate a LaTeX metrics comparison table."""
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

    n_cols = 1 + len(metric_names)
    col_spec = "l" + "c" * len(metric_names)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:model_performance}")
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

        cells = [_escape_latex(name.upper())]
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
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _table1_to_latex(table1_df: pd.DataFrame) -> str:
    """Convert Table 1 DataFrame to LaTeX."""
    if table1_df is None or table1_df.empty:
        return ""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Characteristics of the study population.}")
    lines.append(r"\label{tab:table1}")

    n_cols = len(table1_df.columns) + 1  # +1 for index
    col_spec = "l" + "c" * len(table1_df.columns)
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
) -> str:
    """Generate a complete LaTeX manuscript template.

    Returns compilable LaTeX source populated with actual results.
    """
    sections = []

    # ── Preamble ──
    sections.append(r"""\documentclass[12pt, a4paper]{article}

% ── Packages ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{float}
\usepackage{setspace}
\usepackage{caption}

\doublespacing

% ── Title ──""")

    sections.append(f"\\title{{{_escape_latex(title)}}}")
    sections.append(f"\\author{{{_escape_latex(authors)} \\\\ \\small{{{_escape_latex(affiliation)}}}}}")
    sections.append(r"\date{\today}")
    sections.append("")
    sections.append(r"\begin{document}")
    sections.append(r"\maketitle")

    # ── Abstract ──
    sections.append(r"""
\begin{abstract}""")
    sections.append(f"{_escape_latex(abstract)}")
    sections.append(r"""\end{abstract}

\clearpage
""")

    # ── Introduction ──
    sections.append(r"""\section{Introduction}

[PLACEHOLDER: Provide background on the clinical/research context and rationale for developing this prediction model. Cite relevant prior work.]

\subsection{Objectives}
[PLACEHOLDER: State the specific objectives of this study, including whether you are developing, validating, or both.]

""")

    # ── Methods ──
    sections.append(r"\section{Methods}")
    sections.append("")

    if methods_section:
        # Convert markdown to LaTeX properly
        methods_latex, results_latex = _convert_markdown_to_latex(methods_section)
        if methods_latex:
            sections.append(methods_latex)
        # Store results_latex for later use in Results section
        draft_results = results_latex
    else:
        draft_results = ""
        sections.append(r"""
\subsection{Study Design and Participants}
[PLACEHOLDER: Describe the study design, data source, eligibility criteria, and key dates.]

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
\subsection{Missing Data}
[PLACEHOLDER: Describe how missing data were handled, including the mechanism (MCAR/MAR/MNAR) and imputation strategy.]

\subsection{Model Development}
[PLACEHOLDER: Describe preprocessing, model selection, and internal validation strategy.]
""")
        if n_total > 0:
            sections.append(f"Data were split into training (n={n_train:,}), validation (n={n_val:,}), and test (n={n_test:,}) sets.")

        sections.append(r"""
\subsection{Performance Evaluation}
Model performance was assessed using [METRICS] with 95\% confidence intervals computed via 1,000 BCa bootstrap resamples.
""")

    # ── Results ──
    sections.append(r"""
\section{Results}

\subsection{Study Population}""")

    if n_total > 0:
        sections.append(f"A total of {n_total:,} participants were included in the analysis.")

    # Table 1
    if table1_df is not None and not table1_df.empty:
        sections.append(_table1_to_latex(table1_df))
    else:
        sections.append(r"[INSERT TABLE 1: Characteristics of the study population]")

    # Model Performance — always use proper LaTeX table when results available
    sections.append(r"""
\subsection{Model Performance}""")

    if model_results:
        # Use the proper LaTeX tabular table (not prose from draft_results)
        sections.append(_metrics_to_latex_table(model_results, task_type, bootstrap_results))
    elif draft_results:
        # Fallback: use draft results text if no structured model_results
        sections.append("\n")
        sections.append(draft_results)
    else:
        sections.append(r"[INSERT TABLE: Model performance metrics with 95\% CIs]")

    # Calibration
    if calibration_text:
        sections.append(r"""
\subsection{Calibration}""")
        sections.append(_escape_latex(calibration_text))
    else:
        sections.append(r"""
\subsection{Calibration}
[PLACEHOLDER: Report calibration results --- Brier score, ECE, calibration slope/intercept. Include calibration plot as a figure.]
""")

    # ── Discussion ──
    sections.append(r"""
\section{Discussion}

\subsection{Principal Findings}
[PLACEHOLDER: Summarize the main results in context of the study objectives.]

\subsection{Comparison with Prior Work}
[PLACEHOLDER: Compare your results with existing literature.]

\subsection{Clinical Implications}
[PLACEHOLDER: Discuss practical implications for clinical decision-making or research.]

\subsection{Strengths and Limitations}
""")
    sections.append(_escape_latex(limitations))

    sections.append(r"""
\subsection{Conclusion}
[PLACEHOLDER: State the main conclusion and its implications.]

""")

    # ── References ──
    sections.append(r"""
\section*{References}
\begin{enumerate}
\item [PLACEHOLDER: Add references in journal format]
\item Collins GS, et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ. 2015;350:g7594.
\end{enumerate}
""")

    # ── Supplementary ──
    sections.append(r"""
\clearpage
\appendix
\section{Supplementary Material}

\subsection{TRIPOD Checklist}
[See exported TRIPOD checklist CSV/PDF]

\subsection{Reproducibility}
This analysis was conducted using Tabular ML Lab (Python). Full reproducibility manifest including software versions, random seeds, and data hashes is available in the exported analysis package.

""")

    sections.append(r"\end{document}")

    return "\n".join(sections)
