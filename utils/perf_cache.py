"""Cached heavy computations for wide datasets.

Streamlit re-executes the whole page script on every widget interaction. On
wide data (hundreds to thousands of columns) the uncached O(columns) scans in
the audit and EDA sections turn each click into a multi-second stall. These
helpers wrap the hot computations in st.cache_data so they run once per
dataset, not once per rerun.

All functions take the DataFrame as a hashable cache key (Streamlit hashes
the underlying data). Return values are cached copies — mutating them does
not corrupt the cache.
"""
import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Parsing uploaded file…", max_entries=8)
def cached_parse_upload(file_bytes: bytes, filename: str,
                        transpose: bool, excel_sheet: int) -> pd.DataFrame:
    """Parse an uploaded file once per (content, options) combination.

    The upload page re-renders while a file sits in the uploader; without
    this cache the file is re-parsed on every rerun.
    """
    from data_processor import load_tabular_data
    return load_tabular_data(
        io.BytesIO(file_bytes), filename=filename,
        transpose=transpose, excel_sheet=excel_sheet,
    )


@st.cache_data(show_spinner="Auditing columns (one-time per dataset)…", max_entries=8)
def cached_audit_tables(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """(cardinality_data, dtype_data) for the audit section, vectorized.

    Replaces two per-column Python loops (nunique/isnull/count per Series)
    with whole-frame C-level scans. Semantics match the original loops; the
    sample values are drawn from the first 200 rows.
    """
    n_total = len(df)
    nunique = df.nunique()
    nulls = df.isnull().sum()
    nonnull = n_total - nulls
    head = df.head(200)

    mixed_type_cols = set()
    for col in df.select_dtypes(include=["object"]).columns:
        nn = int(nonnull[col])
        if nn > 0:
            try:
                numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
                if 0 < numeric_count < nn:
                    mixed_type_cols.add(col)
            except Exception:
                pass

    cardinality_data = []
    dtype_data = []
    for col in df.columns:
        nu = int(nunique[col])
        n_null = int(nulls[col])
        pct_unique = (nu / n_total) * 100 if n_total > 0 else 0

        if nu == 1:
            card_type, card_flag = "Constant", "⚠️"
        elif nu == 2:
            card_type, card_flag = "Binary", ""
        elif nu == n_total:
            card_type, card_flag = "Unique (potential ID)", "🔑"
        elif nu <= 10:
            card_type, card_flag = "Low cardinality", ""
        elif nu <= 50:
            card_type, card_flag = "Moderate cardinality", ""
        elif pct_unique > 90:
            card_type, card_flag = "High cardinality (near-unique)", ""
        else:
            card_type, card_flag = "High cardinality", ""

        cardinality_data.append({
            'Column': col,
            'Unique': nu,
            '% Unique': f"{pct_unique:.1f}%",
            'Type': card_type,
            'Flag': card_flag,
        })

        sample_vals = head[col].dropna().head(3).tolist()
        sample_str = str(sample_vals)[:50] + "..." if len(str(sample_vals)) > 50 else str(sample_vals)
        validity_issues = []
        if n_null > 0:
            validity_issues.append(f"{n_null} missing")
        if col in mixed_type_cols:
            validity_issues.append("mixed types")

        dtype_data.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-null': int(nonnull[col]),
            'Null': n_null,
            'Unique': nu,
            'Sample': sample_str,
            'Issues': ', '.join(validity_issues) if validity_issues else 'OK',
        })

    return cardinality_data, dtype_data


@st.cache_data(show_spinner="Computing summary statistics (one-time per dataset)…", max_entries=8)
def cached_numeric_summary(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """describe().T + skew + missing_% for the given numeric columns.

    df.describe() over thousands of columns costs several seconds; cache it
    per dataset so reruns render the table from memory.
    """
    sub = df[list(cols)]
    out = sub.describe().T
    out["skew"] = sub.skew()
    out["kurtosis"] = sub.kurtosis()
    out["missing_%"] = sub.isnull().mean() * 100
    return out


@st.cache_data(show_spinner="Scanning feature–target correlations (one-time per dataset)…", max_entries=8)
def cached_target_correlations(
    df: pd.DataFrame, target_col: str, cols: Tuple[str, ...]
) -> Tuple[pd.Series, pd.Series]:
    """(pearson, spearman) correlation of each column vs the target.

    Vectorized: one corrwith pass for Pearson, and Spearman via the
    rank-transform identity (pearson of ranks), replacing thousands of
    per-column Series.corr calls. With missing values the rank-based
    Spearman can differ from pairwise-deletion Spearman in the 3rd decimal;
    the consumers threshold at coarse cutoffs (0.7, 0.08) where this is
    immaterial.
    """
    sub = df[list(cols)]
    target = df[target_col]
    pearson = sub.corrwith(target)
    spearman = sub.rank().corrwith(target.rank())
    return pearson, spearman
