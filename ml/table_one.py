"""
Table 1 Generator: Characteristics of Study Population.

Generates publication-ready descriptive statistics tables stratified by groups,
with proper statistical tests and formatting for journal submission.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class Table1Config:
    """Configuration for Table 1 generation."""
    grouping_var: Optional[str] = None
    continuous_vars: List[str] = field(default_factory=list)
    categorical_vars: List[str] = field(default_factory=list)
    show_pvalues: bool = True
    show_smd: bool = False
    show_missing: bool = True
    normal_test_alpha: float = 0.05
    use_median_iqr_if_skewed: bool = True
    decimal_places: int = 1


def partition_table1_variables(
    df: pd.DataFrame,
    feature_names: List[str],
    grouping_var: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Split final manuscript predictors into continuous and categorical Table 1 lists.

    Preserves the incoming feature order and excludes the grouping variable.
    """
    available_features = [feature for feature in feature_names if feature in df.columns and feature != grouping_var]
    numeric_set = set(df.select_dtypes(include=[np.number]).columns.tolist())
    continuous = [feature for feature in available_features if feature in numeric_set]
    categorical = [feature for feature in available_features if feature not in numeric_set]
    return continuous, categorical


def generate_feature_table1(
    df: pd.DataFrame,
    feature_names: List[str],
    grouping_var: Optional[str] = None,
    show_pvalues: bool = True,
    show_smd: bool = False,
    show_missing: bool = True,
    decimal_places: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any], Table1Config]:
    """Generate a manuscript Table 1 for a specific finalized feature set."""
    continuous_vars, categorical_vars = partition_table1_variables(
        df,
        feature_names,
        grouping_var=grouping_var,
    )
    config = Table1Config(
        grouping_var=grouping_var,
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars,
        show_pvalues=show_pvalues,
        show_smd=show_smd,
        show_missing=show_missing,
        decimal_places=decimal_places,
    )
    table_df, metadata = generate_table1(df, config)
    return table_df, metadata, config


def _is_normal(series: pd.Series, alpha: float = 0.05) -> bool:
    """Test normality using Shapiro-Wilk (n<5000) or D'Agostino-Pearson."""
    clean = series.dropna()
    if len(clean) < 8:
        return False
    try:
        if len(clean) < 5000:
            _, p = stats.shapiro(clean)
        else:
            _, p = stats.normaltest(clean)
        return p > alpha
    except Exception:
        return False


def _smd(group1: pd.Series, group2: pd.Series) -> float:
    """Compute standardized mean difference (Cohen's d)."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(), g2.std()
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return abs(m1 - m2) / pooled_std


def _format_pvalue(p: float) -> str:
    """Format p-value for publication."""
    if p is None or np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def _continuous_row(series: pd.Series, is_normal: bool, dp: int) -> str:
    """Format a continuous variable summary."""
    clean = series.dropna()
    if len(clean) == 0:
        return "—"
    if is_normal:
        return f"{clean.mean():.{dp}f} ± {clean.std():.{dp}f}"
    else:
        q25, q50, q75 = clean.quantile([0.25, 0.5, 0.75])
        return f"{q50:.{dp}f} [{q25:.{dp}f}, {q75:.{dp}f}]"


def _categorical_row(series: pd.Series, category: Any) -> str:
    """Format a categorical variable count and percentage."""
    clean = series.dropna()
    n = (clean == category).sum()
    pct = n / len(clean) * 100 if len(clean) > 0 else 0
    return f"{n} ({pct:.1f}%)"


def generate_table1(
    df: pd.DataFrame,
    config: Table1Config,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate Table 1: Characteristics of Study Population.

    Args:
        df: Input DataFrame
        config: Table1Config with variable lists and options

    Returns:
        (table_df, metadata) where table_df is the formatted table
        and metadata contains test info and raw statistics
    """
    metadata = {"tests_used": {}, "normality": {}, "raw_stats": {}}

    if config.grouping_var and config.grouping_var in df.columns:
        groups = df[config.grouping_var].dropna().unique()
        groups = sorted(groups, key=str)
    else:
        groups = None

    rows = []
    row_labels = []
    col_headers = []

    # Build column headers
    n_total = len(df)
    if groups is not None:
        col_headers = [f"Overall (N={n_total})"]
        for g in groups:
            n_g = (df[config.grouping_var] == g).sum()
            col_headers.append(f"{g} (n={n_g})")
        if config.show_pvalues:
            col_headers.append("P-value")
        if config.show_smd and len(groups) == 2:
            col_headers.append("SMD")
    else:
        col_headers = [f"Overall (N={n_total})"]

    # Continuous variables
    for var in config.continuous_vars:
        if var not in df.columns:
            continue

        series = df[var]
        is_norm = _is_normal(series, config.normal_test_alpha)
        use_median = config.use_median_iqr_if_skewed and not is_norm
        metadata["normality"][var] = is_norm

        label = var
        if use_median:
            label += ", median [IQR]"
        else:
            label += ", mean ± SD"

        row = [_continuous_row(series, not use_median, config.decimal_places)]

        if groups is not None:
            group_series = []
            for g in groups:
                gs = df.loc[df[config.grouping_var] == g, var]
                group_series.append(gs)
                row.append(_continuous_row(gs, not use_median, config.decimal_places))

            # Statistical test
            if config.show_pvalues:
                clean_groups = [gs.dropna() for gs in group_series if len(gs.dropna()) > 0]
                if len(clean_groups) >= 2:
                    if len(clean_groups) == 2:
                        if all(_is_normal(cg) for cg in clean_groups):
                            stat, p = stats.ttest_ind(clean_groups[0], clean_groups[1])
                            test_name = "t-test"
                        else:
                            stat, p = stats.mannwhitneyu(
                                clean_groups[0], clean_groups[1], alternative='two-sided'
                            )
                            test_name = "Mann-Whitney U"
                    else:
                        if all(_is_normal(cg) for cg in clean_groups):
                            stat, p = stats.f_oneway(*clean_groups)
                            test_name = "ANOVA"
                        else:
                            stat, p = stats.kruskal(*clean_groups)
                            test_name = "Kruskal-Wallis"
                    row.append(_format_pvalue(p))
                    metadata["tests_used"][var] = test_name
                else:
                    row.append("—")

            # SMD (only for 2 groups)
            if config.show_smd and len(groups) == 2:
                smd_val = _smd(group_series[0], group_series[1])
                row.append(f"{smd_val:.3f}" if not np.isnan(smd_val) else "—")

        row_labels.append(label)
        rows.append(row)

        # Missing count
        if config.show_missing and series.isna().sum() > 0:
            n_miss = series.isna().sum()
            pct_miss = n_miss / len(series) * 100
            miss_row = [f"{n_miss} ({pct_miss:.1f}%)"]
            if groups is not None:
                for g in groups:
                    gs = df.loc[df[config.grouping_var] == g, var]
                    nm = gs.isna().sum()
                    pm = nm / len(gs) * 100 if len(gs) > 0 else 0
                    miss_row.append(f"{nm} ({pm:.1f}%)")
                if config.show_pvalues:
                    miss_row.append("")
                if config.show_smd and len(groups) == 2:
                    miss_row.append("")
            row_labels.append(f"  Missing")
            rows.append(miss_row)

    # Categorical variables
    for var in config.categorical_vars:
        if var not in df.columns:
            continue

        series = df[var]
        categories = sorted(series.dropna().unique(), key=str)

        # Variable header row
        header_row = [""] * len(col_headers)
        row_labels.append(f"{var}, n (%)")
        rows.append(header_row)

        for cat in categories:
            row = [_categorical_row(series, cat)]

            if groups is not None:
                group_series = []
                for g in groups:
                    gs = df.loc[df[config.grouping_var] == g, var]
                    group_series.append(gs)
                    row.append(_categorical_row(gs, cat))

            row_labels.append(f"  {cat}")
            rows.append(row)

        # P-value for categorical (chi-square or Fisher's exact)
        if groups is not None and config.show_pvalues:
            contingency = pd.crosstab(df[var].dropna(), df[config.grouping_var].dropna())
            try:
                if contingency.shape[0] <= 2 and contingency.shape[1] <= 2 and contingency.min().min() < 5:
                    _, p = stats.fisher_exact(contingency.values[:2, :2])
                    test_name = "Fisher's exact"
                else:
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    test_name = "Chi-square"
                # Attach p-value to the header row
                header_idx = len(rows) - len(categories) - 1
                p_col_idx = len(col_headers) - 1
                if config.show_smd and len(groups) == 2:
                    p_col_idx -= 1
                rows[header_idx][p_col_idx] = _format_pvalue(p)
                metadata["tests_used"][var] = test_name
            except Exception:
                pass

        # Missing
        if config.show_missing and series.isna().sum() > 0:
            n_miss = series.isna().sum()
            pct_miss = n_miss / len(series) * 100
            miss_row = [f"{n_miss} ({pct_miss:.1f}%)"]
            if groups is not None:
                for g in groups:
                    gs = df.loc[df[config.grouping_var] == g, var]
                    nm = gs.isna().sum()
                    pm = nm / len(gs) * 100 if len(gs) > 0 else 0
                    miss_row.append(f"{nm} ({pm:.1f}%)")
                if config.show_pvalues:
                    miss_row.append("")
                if config.show_smd and len(groups) == 2:
                    miss_row.append("")
            row_labels.append(f"  Missing")
            rows.append(miss_row)

    # Build DataFrame
    table_df = pd.DataFrame(rows, columns=col_headers, index=row_labels)
    table_df.index.name = "Characteristic"

    return table_df, metadata


def table1_to_latex(table_df: pd.DataFrame, caption: str = "Characteristics of study population") -> str:
    """Export Table 1 to LaTeX format with width containment."""
    if table_df is None or table_df.empty:
        return ""
    
    n_cols = len(table_df.columns)
    is_wide = n_cols > 4
    
    # Build LaTeX manually for better control over formatting
    # Use p{4cm} for first column (Characteristic/index) to allow wrapping
    col_spec = "p{4cm}" + "c" * n_cols
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:table1}")
    
    # Width containment: wrap in adjustbox
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    
    # Use smaller font for wide tables
    if is_wide:
        lines.append(r"\small")
    
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    index_name = table_df.index.name if table_df.index.name else "Characteristic"
    header = index_name + " & " + " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in table_df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Rows
    for idx, row in table_df.iterrows():
        idx_str = str(idx).replace("_", "\\_").replace("%", "\\%").replace("±", "$\\pm$")
        cells = [idx_str]
        for val in row.values:
            val_str = str(val) if val and val != "" else "—"
            val_str = val_str.replace("_", "\\_").replace("%", "\\%").replace("±", "$\\pm$")
            cells.append(val_str)
        lines.append(" & ".join(cells) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def table1_to_csv(table_df: pd.DataFrame) -> str:
    """Export Table 1 to CSV."""
    return table_df.to_csv()
