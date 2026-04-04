"""Tests for missingness-as-feature encoding logic (issue #90).

These tests verify the core encoding logic without requiring Streamlit.
"""
import numpy as np
import pandas as pd


# ── has_data binary indicator ──────────────────────────────────────────────

def test_has_data_indicator_basic():
    """Binary has_data indicator: 1 where observed, 0 where missing."""
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    result = (~s.isnull()).astype(int)
    assert list(result) == [1, 0, 1, 0, 1]


def test_has_data_indicator_no_missing():
    """All observed → all ones."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = (~s.isnull()).astype(int)
    assert list(result) == [1, 1, 1]


def test_has_data_indicator_all_missing():
    """All missing → all zeros."""
    s = pd.Series([np.nan, np.nan, np.nan])
    result = (~s.isnull()).astype(int)
    assert list(result) == [0, 0, 0]


def test_has_data_indicator_string_column():
    """Works with string/categorical columns too."""
    s = pd.Series(["a", None, "b", np.nan, "c"])
    result = (~s.isnull()).astype(int)
    assert list(result) == [1, 0, 1, 0, 1]


# ── Missingness detection ─────────────────────────────────────────────────

def test_identify_features_with_missing():
    """Correctly identifies which features have missing values."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1, np.nan, 3],
        "c": [np.nan, np.nan, 3],
        "d": [1, 2, 3],
    })
    missing_counts = df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    assert set(features_with_missing.index) == {"b", "c"}
    assert features_with_missing["b"] == 1
    assert features_with_missing["c"] == 2


def test_missing_rate_threshold():
    """Threshold filtering: only features above X% missing are selected."""
    df = pd.DataFrame({
        "low_miss": [1, 2, np.nan] + [4] * 97,     # 1% missing
        "mid_miss": [np.nan] * 20 + [1.0] * 80,     # 20% missing
        "high_miss": [np.nan] * 60 + [1.0] * 40,    # 60% missing
    })
    n_rows = len(df)
    missing_counts = df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

    # 5% threshold
    threshold = 0.05
    eligible = features_with_missing[features_with_missing / n_rows >= threshold]
    assert set(eligible.index) == {"mid_miss", "high_miss"}

    # 50% threshold
    eligible_50 = features_with_missing[features_with_missing / n_rows >= 0.50]
    assert set(eligible_50.index) == {"high_miss"}


# ── Conditional ordinal encoding ──────────────────────────────────────────

def test_conditional_ordinal_basic():
    """Ordinal encoding: missing → 0, categories → sequential integers."""
    s = pd.Series(["B", None, "A", "C", np.nan, "A"])
    categories = s.dropna().unique()
    cat_map = {cat: i + 1 for i, cat in enumerate(sorted(categories))}
    result = s.map(cat_map).fillna(0).astype(int)
    # sorted: A=1, B=2, C=3
    assert list(result) == [2, 0, 1, 3, 0, 1]


def test_conditional_ordinal_no_missing():
    """When no missing, values are 1-indexed (no zeros)."""
    s = pd.Series(["X", "Y", "X", "Z"])
    categories = s.dropna().unique()
    cat_map = {cat: i + 1 for i, cat in enumerate(sorted(categories))}
    result = s.map(cat_map).fillna(0).astype(int)
    assert 0 not in list(result)
    assert min(result) == 1


def test_conditional_ordinal_all_missing():
    """All missing → all zeros."""
    s = pd.Series([np.nan, None, np.nan])
    categories = s.dropna().unique()
    cat_map = {cat: i + 1 for i, cat in enumerate(sorted(categories))}
    result = s.map(cat_map).fillna(0).astype(int)
    assert list(result) == [0, 0, 0]


# ── Integration-style test ────────────────────────────────────────────────

def test_full_missingness_encoding_workflow():
    """End-to-end: detect, create indicators, verify DataFrame shape."""
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 50],
        "meds": ["yes", None, "no", None, "yes"],
        "glucose": [100, 110, 105, 120, np.nan],
        "complete": [1, 2, 3, 4, 5],
    })
    X = df.copy()

    # Step 1: detect
    missing_counts = X.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    assert len(features_with_missing) == 3  # age, meds, glucose

    # Step 2: create has_data indicators for numeric
    for feat in ["age", "glucose"]:
        X[f"{feat}_has_data"] = (~df[feat].isnull()).astype(int)

    # Step 3: create ordinal for categorical
    s = df["meds"]
    categories = s.dropna().unique()
    cat_map = {cat: i + 1 for i, cat in enumerate(sorted(categories))}
    X["meds_ordinal"] = s.map(cat_map).fillna(0).astype(int)

    assert X.shape == (5, 7)  # 4 original + 2 has_data + 1 ordinal
    assert list(X["age_has_data"]) == [1, 1, 0, 1, 1]
    assert list(X["glucose_has_data"]) == [1, 1, 1, 1, 0]
    assert X["meds_ordinal"].iloc[1] == 0  # missing → 0
    assert X["meds_ordinal"].iloc[0] > 0   # observed → nonzero
