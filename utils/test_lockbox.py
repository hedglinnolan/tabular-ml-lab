"""Test-set lockbox: freeze held-out test rows the moment the modeling
problem is defined.

Methodological contract (split-first workflow):
- Immediately after the user confirms data + target + task type on Upload &
  Audit, a seeded (stratified where feasible) test fraction is drawn and its
  row labels are frozen in st.session_state['test_lockbox'].
- Every target-aware step upstream of Train & Compare — feature-engineering
  fits, feature selection, target-association views — operates on training
  rows only, via train_row_mask().
- Train & Compare consumes the frozen labels as THE test set and only divides
  the remaining rows into train/validation. The test set is opened once.
- 'Exploratory mode' disables the quarantine explicitly; downstream metrics
  and the manuscript are watermarked accordingly (never silently).

The lockbox stores index LABELS (not positions) so membership survives
feature engineering and row filtering, which preserve the original index.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_TEST_FRACTION = 0.15
_MIN_ROWS_FOR_LOCKBOX = 10


def is_exploratory() -> bool:
    """Explicit, user-chosen escape hatch. Never enabled by default."""
    return bool(st.session_state.get("exploratory_mode", False))


def get_lockbox() -> Optional[Dict[str, Any]]:
    lb = st.session_state.get("test_lockbox")
    if lb and lb.get("labels") is not None:
        return lb
    return None


def _lockbox_signature(df: pd.DataFrame, target_col: str, task_type: str,
                       fraction: float, seed: int) -> str:
    try:
        content = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        content = df.shape
    return f"{content}|{df.shape}|{target_col}|{task_type}|{fraction:.4f}|{seed}"


def ensure_lockbox(df: pd.DataFrame, target_col: str, task_type: str,
                   fraction: Optional[float] = None,
                   seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Create or refresh the lockbox; rebuild only when its inputs change.

    Returns the lockbox dict, or None when a lockbox cannot be drawn
    (no target, too few rows). A rebuild that REPLACES an existing lockbox
    invalidates downstream results — models evaluated against a different
    test set are no longer comparable.
    """
    if df is None or not target_col or target_col not in df.columns:
        return get_lockbox()

    fraction = float(fraction if fraction is not None
                     else st.session_state.get("test_lockbox_fraction", DEFAULT_TEST_FRACTION))
    seed = int(seed if seed is not None else st.session_state.get("random_seed", 42))

    y = df[target_col]
    eligible = df.index[y.notna()]
    if len(eligible) < _MIN_ROWS_FOR_LOCKBOX:
        return get_lockbox()

    sig = _lockbox_signature(df, target_col, task_type, fraction, seed)
    existing = get_lockbox()
    if existing and existing.get("signature") == sig:
        return existing

    from sklearn.model_selection import train_test_split

    stratify = None
    if task_type == "classification":
        y_eligible = y.loc[eligible]
        counts = y_eligible.value_counts()
        # Stratification needs >=2 members per class and >=1 expected per class
        if counts.min() >= 2 and (counts * fraction).min() >= 1:
            stratify = y_eligible

    try:
        _, test_labels = train_test_split(
            eligible, test_size=fraction, random_state=seed, stratify=stratify
        )
    except ValueError:
        _, test_labels = train_test_split(
            eligible, test_size=fraction, random_state=seed
        )

    lockbox = {
        "labels": list(test_labels),
        "fraction": fraction,
        "seed": seed,
        "n_total": int(len(eligible)),
        "n_test": int(len(test_labels)),
        "signature": sig,
        "stratified": stratify is not None,
    }

    if existing is not None and existing.get("labels") != lockbox["labels"]:
        # Different test set → previous results are not comparable
        from utils.session_state import reset_downstream_results
        reset_downstream_results(clear_feature_engineering=False)

    st.session_state["test_lockbox"] = lockbox
    return lockbox


def train_row_mask(index: pd.Index) -> pd.Series:
    """Boolean Series over `index`: True for rows a target-aware step may see.

    In exploratory mode (or with no lockbox) every row is visible. Works on
    any frame that preserves the original row labels (df_engineered,
    filtered_data, column subsets).
    """
    lb = get_lockbox()
    if lb is None or is_exploratory():
        return pd.Series(True, index=index)
    test_set = set(lb["labels"])
    return pd.Series([lbl not in test_set for lbl in index], index=index)


def render_lockbox_status(context: str = "") -> None:
    """The quiet, consistent status chip shown on workflow pages."""
    if is_exploratory():
        st.warning(
            "🔓 **Exploratory mode** — the test-set quarantine is OFF. "
            "Target-aware steps see all rows; downstream metrics are for "
            "exploration and will be flagged as such in any export.",
            icon="🔓",
        )
        return
    lb = get_lockbox()
    if lb is None:
        return
    extra = f" {context}" if context else ""
    st.caption(
        f"🔒 Test set: {lb['fraction']:.0%} (n={lb['n_test']}"
        f"{', stratified' if lb.get('stratified') else ''}) held out since upload — "
        f"opened once at Train & Compare.{extra}"
    )
