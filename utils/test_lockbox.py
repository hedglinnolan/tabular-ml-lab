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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_TEST_FRACTION = 0.15
_MIN_ROWS_FOR_LOCKBOX = 10
# Splitting by subject needs enough subjects for the fraction to be meaningful.
_MIN_GROUPS_FOR_GROUPED_LOCKBOX = 8


def is_exploratory() -> bool:
    """Explicit, user-chosen escape hatch. Never enabled by default."""
    return bool(st.session_state.get("exploratory_mode", False))


def get_lockbox() -> Optional[Dict[str, Any]]:
    lb = st.session_state.get("test_lockbox")
    if lb and lb.get("labels") is not None:
        return lb
    return None


def _lockbox_signature(df: pd.DataFrame, target_col: str, task_type: str,
                       fraction: float, seed: int, group_col: Optional[str] = None) -> str:
    try:
        content = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        # Unhashable cells (e.g. a list from nested JSON) must not silently
        # collapse the signature to something stable — that would stop the
        # lockbox from ever being redrawn. Fall back to a coarse but
        # content-sensitive descriptor.
        try:
            content = f"{df.shape}|{tuple(df.columns)}|{df.notna().sum().sum()}"
        except Exception:
            content = str(df.shape)
    return (f"{content}|{df.shape}|{target_col}|{task_type}|"
            f"{fraction:.4f}|{seed}|{group_col or ''}")


def detect_repeated_subjects(df: pd.DataFrame,
                             candidate_cols: Optional[list] = None
                             ) -> Optional[Tuple[str, int, int]]:
    """Find a column that looks like a subject ID appearing on several rows.

    Returns (column, n_subjects, n_rows) or None. Used to catch the case that
    silently defeats the quarantine: a merge with repeated measures puts the
    SAME subject in both the training rows and the sealed test rows, so the
    "held-out" set was already trained on.
    """
    if df is None or df.empty:
        return None
    n = len(df)
    cols = candidate_cols if candidate_cols is not None else list(df.columns)
    best = None
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):
            continue
        try:
            k = int(s.nunique(dropna=True))
        except TypeError:
            continue
        if k < 2 or k >= n:
            continue                     # unique per row => no repetition
        # An identifier repeats a handful of times; a category repeats hundreds.
        rows_per = n / k
        if rows_per < 1.5 or rows_per > 50:
            continue
        name = str(col).lower()
        looks_like_id = any(t in name for t in
                            ("id", "seqn", "subject", "participant", "patient",
                             "record", "case", "person"))
        if not looks_like_id:
            continue
        if best is None or k > best[1]:
            best = (str(col), k, n)
    return best


def ensure_lockbox(df: pd.DataFrame, target_col: str, task_type: str,
                   fraction: Optional[float] = None,
                   seed: Optional[int] = None,
                   group_col: Optional[str] = None,
                   stratify_cols: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
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

    # If the caller did not name a subject column, look for one anyway: a
    # merge with repeated measures otherwise splits the SAME subject across
    # train and test, so the "held-out" rows were already trained on and the
    # quarantine is silently worthless.
    if not group_col:
        detected = detect_repeated_subjects(df)
        if detected:
            group_col = detected[0]
    if group_col and group_col not in df.columns:
        group_col = None

    sig = _lockbox_signature(df, target_col, task_type, fraction, seed,
                            f"{group_col}|{'+'.join(sorted(stratify_cols or []))}")
    existing = get_lockbox()
    if existing and existing.get("signature") == sig:
        return existing

    # Every cohort run inherits its slice of ONE split, drawn before the study
    # was ever divided. Redrawing mid-run silently re-partitions the study: rows
    # sealed since upload become trainable, and two runs can no longer be
    # compared because run 2's test people may have been run 1's training
    # people. Refuse, and let the page say so rather than proceeding quietly.
    if existing is not None:
        from utils.cohorts import active_cohort
        run = active_cohort()
        if run is not None:
            st.session_state["_lockbox_redraw_refused"] = {
                "column": run["column"], "label": run["label"],
            }
            return existing

    from sklearn.model_selection import train_test_split

    def _build_stratum():
        """Composite stratum for the split, and the columns it ended up using.

        The outcome always — a test set holding twice the event rate makes every
        metric meaningless. Plus any demographic the researcher named, because a
        test set that is 75% men when the cohort is 71% men reports a number that
        does not describe the study population.
        """
        parts: List[pd.Series] = []
        used: List[str] = []
        if task_type == "classification":
            parts.append(y.loc[eligible].astype(str))
            used.append(target_col)
        for col in (stratify_cols or []):
            if col in df.columns and col != target_col:
                parts.append(df.loc[eligible, col].astype(str))
                used.append(col)
        # Every extra variable multiplies the cells, and one singleton cell makes
        # the split impossible. Drop the least important variable until it fits,
        # rather than failing or silently falling back to no stratification.
        while parts:
            combined = parts[0]
            for extra in parts[1:]:
                combined = combined.str.cat(extra, sep="|")
            counts = combined.value_counts()
            if counts.min() >= 2 and (counts * fraction).min() >= 1:
                return combined, used
            parts.pop()
            used.pop()
        return None, []

    grouped = False
    test_labels = None
    if group_col:
        groups = df.loc[eligible, group_col]
        n_groups = int(groups.nunique(dropna=False))
        if n_groups >= _MIN_GROUPS_FOR_GROUPED_LOCKBOX:
            from sklearn.model_selection import GroupShuffleSplit
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                _, test_pos = next(gss.split(eligible, groups=groups))
                test_labels = eligible[test_pos]
                grouped = True
            except Exception:
                test_labels = None
        if test_labels is None:
            group_col = None            # too few groups to split by subject

    # Stratification is decided AFTER the grouped attempt resolves. Deciding it
    # earlier meant that a group column with too few subjects to split by fell
    # back to an ordinary split carrying NO stratification at all — the one
    # case that produced a test set unrepresentative of both the outcome and
    # the demographics, silently.
    stratify: Optional[pd.Series] = None
    strata_used: List[str] = []
    if test_labels is None:
        stratify, strata_used = _build_stratum()
        try:
            _, test_labels = train_test_split(
                eligible, test_size=fraction, random_state=seed, stratify=stratify
            )
        except ValueError:
            stratify, strata_used = None, []
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
        "strata": list(strata_used),
        "group_col": group_col if grouped else None,
        "n_test_groups": (int(df.loc[test_labels, group_col].nunique())
                          if grouped else None),
    }

    if existing is not None and existing.get("labels") != lockbox["labels"]:
        # Different test set → previous results are not comparable
        from utils.session_state import reset_downstream_results
        reset_downstream_results(clear_feature_engineering=False)
        # Let the page disclose the redraw — a silent reset reads as data loss
        st.session_state["_lockbox_redrawn"] = True

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

    # A cohort run works on a subset, and the study-wide n is then simply not
    # this run's test set: "n=135" beside a 490-row run is a number the
    # researcher would write down and be wrong about.
    from utils.cohorts import active_cohort
    run = active_cohort()
    if run is not None:
        n_here = len(set(lb["labels"]) & set(run["labels"]))
        st.caption(
            f"🔒 Test set for this run ({run['column']} = {run['label']}): "
            f"n={n_here:,} — this run's share of the {lb['n_test']:,} rows drawn "
            f"once at upload, before the study was split, so every run is "
            f"evaluated against the same held-out people. Opened once at "
            f"Train & Compare.{extra}"
        )
        return

    if lb.get("group_col"):
        st.caption(
            f"🔒 Test set: {lb['fraction']:.0%} (n={lb['n_test']} rows from "
            f"{lb.get('n_test_groups', '?')} subjects, split by '{lb['group_col']}' so no "
            f"subject appears on both sides) held out since upload — opened once at "
            f"Train & Compare.{extra}"
        )
    else:
        st.caption(
            f"🔒 Test set: {lb['fraction']:.0%} (n={lb['n_test']}"
            f"{', stratified' if lb.get('stratified') else ''}) held out since upload — "
            f"opened once at Train & Compare.{extra}"
        )
