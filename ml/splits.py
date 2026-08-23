"""
Train / validation / test splitting — the real thing.

Extracted from `pages/06_Train_and_Compare.py:380-760`, where ~370 lines of the
app's most safety-critical logic lived inside a Streamlit script with no test
around it. This module is headless: it imports no Streamlit, reads no session
state, and writes nothing. It takes a frame and a config, and returns a
partition.

**Four mutually exclusive branches, in priority order.** The order is itself a
safety property — grouping outranks a time split because a subject spanning
partitions is the worse leak:

1. ``grouped``       — ``GroupShuffleSplit`` when the cohort is longitudinal and
                       an entity id exists. Same-subject rows never straddle.
2. ``chronological`` — sort by the datetime column and slice. No shuffle.
3. ``lockbox``       — the sealed labels *are* the test set. Only the remaining
                       rows are divided into train and validation.
4. ``stratified`` / ``random`` — ``train_test_split``.

**Row identity.** `Split` carries both, and the distinction is `T0-ID-001`:

- ``*_labels`` are index labels — the identity that survives filtering, and the
  convention the lockbox already uses.
- ``*_positions`` are offsets into the *split frame* (the rows left after
  dropping a missing target and trimming). They are valid only for the frame
  this call was handed, and must not cross a page boundary.

The two agree only while nothing renumbers the rows in between, which is exactly
what `T0-ID-001`'s identity barrier exists to guarantee. Reading a partition
back later goes through :func:`resolve_split_rows`, which resolves labels
against the current frame and refuses when any of them is gone.

**The identity barrier.** Splitting is a post-barrier operation: rows have
identities by the time they get here, and this module never renumbers them.
:meth:`Split.assert_identity_preserved` states that as a check callers can run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def to_numpy_1d(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """
    Convert input to 1D numpy array.

    Args:
        x: Can be numpy array, pandas Series, or DataFrame column

    Returns:
        1D numpy array
    """
    arr = np.asarray(x)
    return arr.reshape(-1)


class SplitError(ValueError):
    """The data cannot be split as asked, and guessing would be worse."""


@dataclass
class SplitSpec:
    """What the caller wants. Serializable; holds no fitted object.

    Deliberately not `utils.session_state.SplitConfig`: that dataclass belongs to
    the Streamlit host, and this module must stay importable without it. The
    page adapts one to the other.
    """
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = False
    use_time_split: bool = False
    datetime_col: Optional[str] = None
    entity_id_col: Optional[str] = None
    use_group_split: bool = False
    target_trim_enabled: bool = False
    target_trim_lower: float = 0.0
    target_trim_upper: float = 1.0

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 0.01:
            raise SplitError(f"Split fractions must sum to 1.0, got {total:.3f}.")
        for name in ("train_size", "val_size", "test_size"):
            if getattr(self, name) <= 0:
                raise SplitError(f"{name} must be positive.")


@dataclass
class Split:
    """One partition, with both identities and the CV scheme it implies."""

    strategy: str                       # grouped | chronological | lockbox | stratified | random
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

    # Row identity — labels first, because labels are the identity.
    train_labels: List[Any] = field(default_factory=list)
    val_labels: List[Any] = field(default_factory=list)
    test_labels: List[Any] = field(default_factory=list)

    # Positions into the split frame, for consumers that still index that way.
    train_positions: List[int] = field(default_factory=list)
    val_positions: List[int] = field(default_factory=list)
    test_positions: List[int] = field(default_factory=list)

    # Cross-validation must inherit the split's leakage semantics, or the folds
    # leak what the split was careful about.
    cv_strategy: str = "standard"       # standard | group | time
    cv_groups_train: Optional[np.ndarray] = None

    label_encoder: Optional[Any] = None
    lockbox_applied: bool = False
    n_trimmed_rows: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.X_train), "val": len(self.X_val),
                "test": len(self.X_test)}

    def assert_disjoint(self) -> None:
        """No row may appear in two partitions. The leak, checked."""
        tr = set(self.train_labels)
        va = set(self.val_labels)
        te = set(self.test_labels)
        if tr & te:
            raise SplitError(f"{len(tr & te)} row(s) are in both train and test.")
        if (tr & va) or (va & te):
            raise SplitError("Validation overlaps another partition.")

    def assert_identity_preserved(self, source: pd.DataFrame) -> None:
        """Every partitioned label must still name a row in `source`.

        The identity-barrier check (`T0-ID-001`). Splitting happens after rows
        acquire identities, so a label that is no longer in the source frame
        means something renumbered the rows underneath the split.
        """
        known = set(source.index)
        every = list(self.train_labels) + list(self.val_labels) + list(self.test_labels)
        missing = [l for l in every if l not in known]
        if missing:
            raise SplitError(
                f"{len(missing)} split label(s) are not in the source frame "
                f"(e.g. {missing[:5]}). A post-barrier operation renumbered the "
                "rows, so these labels no longer name the rows they came from.")


class SplitIdentityError(SplitError):
    """The stored split no longer names rows that exist in the current frame."""


def resolve_split_rows(df: Optional[pd.DataFrame],
                       labels: Optional[Sequence[Any]],
                       part: str = "test") -> pd.DataFrame:
    """The rows of `df` that a stored split's LABELS name — or a refusal.

    The single way to read a partition back after the split is drawn. Identity
    across a page boundary is the label, never the position: the frame a later
    page fetches can gain or lose rows in between, and a position then names a
    different person with no error at all. A label that is gone means this is
    not the frame the split was drawn on, and there is no correct answer left
    to give — only a refusal.

    Rows come back in the order the split recorded them, so a partition's rows
    stay aligned with the y/y_pred vectors stored beside it.
    """
    if labels is None or len(labels) == 0:
        raise SplitIdentityError(
            f"No {part} row labels were recorded for this split. Re-run "
            "Prepare Splits on the Train & Compare page.")
    if df is None:
        raise SplitIdentityError(
            f"There is no active dataset to resolve the {part} rows against.")

    wanted = pd.Index(list(labels))
    known = wanted.isin(df.index)
    if not known.all():
        missing = list(wanted[~known])
        raise SplitIdentityError(
            f"{len(missing)} of {len(wanted)} {part} row(s) are no longer in "
            f"the active dataset (e.g. {missing[:5]}); it now has {len(df)} "
            "rows. The row set changed after the split was drawn — rebuilding "
            "pipelines with plausibility filtering, re-applying feature "
            "engineering, or a row-dropping repair all do this. Re-run Prepare "
            "Splits on Train & Compare, or undo that change, before reading "
            "held-out results.")

    # A duplicated label makes `.loc` return more rows than the split had,
    # which would silently misalign every vector stored beside it.
    if df.index.has_duplicates and wanted.isin(df.index[df.index.duplicated()]).any():
        raise SplitIdentityError(
            f"The active dataset has duplicate row labels among the {part} "
            "rows, so a label no longer names one row. Re-run Prepare Splits "
            "on Train & Compare.")

    return df.loc[wanted]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_is_categorical(y: pd.Series) -> bool:
    return y.dtype.name in ("object", "category", "bool", "str") or (
        hasattr(y.dtype, "kind") and y.dtype.kind in ("O", "b", "U"))


def _relative_val(spec: "SplitSpec") -> float:
    """Validation's share of the train+val remainder, when test is already fixed."""
    den = spec.train_size + spec.val_size
    return (spec.val_size / den) if den > 0 else 0.18


def choose_strategy(df: pd.DataFrame, spec: "SplitSpec",
                    lockbox_labels: Optional[Sequence[Any]] = None) -> str:
    """Which branch a given input routes to. The priority order, on its own.

    Separated from the split itself so the ordering can be tested — and read —
    without running a partition. Grouping outranks time, and both outrank the
    lockbox, because each of the first two draws its own test set with its own
    leakage semantics.
    """
    if spec.use_group_split and spec.entity_id_col and spec.entity_id_col in df.columns:
        return "grouped"
    if spec.use_time_split and spec.datetime_col and spec.datetime_col in df.columns:
        return "chronological"
    if lockbox_labels:
        return "lockbox"
    return "stratified" if spec.stratify else "random"


# ─────────────────────────────────────────────────────────────────────────────
# the split
# ─────────────────────────────────────────────────────────────────────────────

def make_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    task_type: str,
    spec: Optional["SplitSpec"] = None,
    lockbox_labels: Optional[Sequence[Any]] = None,
) -> Split:
    """Partition `df` into train / validation / test.

    Pure: `df` is never mutated, no global RNG is touched, and nothing is
    written anywhere. The seed travels in `spec.random_state`.

    `lockbox_labels` are index labels frozen at upload. When present and
    applicable they *are* the test set — this function only divides what is
    left. Group and time splits draw their own test set and therefore bypass
    the lockbox, which is disclosed in `Split.notes` rather than done silently.
    """
    spec = spec or SplitSpec()
    spec.validate()

    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import LabelEncoder

    if target_col not in df.columns:
        raise SplitError(f"Target column {target_col!r} is not in the frame.")
    feature_cols = list(feature_cols)
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise SplitError(
            f"{len(missing_features)} selected feature(s) are not in the data: "
            f"{missing_features[:8]}.")
    if not feature_cols:
        raise SplitError("No features were selected.")

    notes: List[str] = []

    # ── rows with a target ────────────────────────────────────────────────
    # Labels are captured BEFORE any positional work, so identity survives.
    mask = df[target_col].notna()
    kept_labels = list(df.index[mask])
    X = df.loc[mask, feature_cols].copy()
    y = df.loc[mask, target_col].copy()
    X.index = pd.RangeIndex(len(X))
    y.index = pd.RangeIndex(len(y))

    if len(X) < 2:
        raise SplitError(
            f"Not enough samples after removing missing target values ({len(X)}). "
            "Need at least 2 rows to split.")

    strategy = choose_strategy(df, spec, lockbox_labels)

    # ── the lockbox ───────────────────────────────────────────────────────
    lockbox_applied = False
    is_test_row = np.zeros(len(X), dtype=bool)
    if lockbox_labels and strategy in ("grouped", "chronological"):
        notes.append(
            f"A lockbox exists but the {strategy} split draws its own test set "
            "(grouping and chronological ordering have their own leakage "
            "semantics), so the upload lockbox does not apply to this split.")
    elif lockbox_labels:
        sealed = set(lockbox_labels)
        is_test_row = np.array([lbl in sealed for lbl in kept_labels])
        if int(is_test_row.sum()) < 2 or int((~is_test_row).sum()) < 4:
            notes.append(
                "Too few lockbox test rows survive the current filters — "
                "falling back to a fresh random split for this run.")
            is_test_row = np.zeros(len(X), dtype=bool)
            strategy = "stratified" if spec.stratify else "random"
        else:
            lockbox_applied = True

    # ── target trimming (regression only, before the split) ───────────────
    n_trimmed = 0
    if task_type == "regression" and spec.target_trim_enabled:
        # With a lockbox, thresholds come from training rows only and test rows
        # are never dropped — otherwise the trim both leaks test-target
        # statistics and evaluates on a truncated population.
        basis = y[~is_test_row] if lockbox_applied else y
        q_lo = float(basis.quantile(spec.target_trim_lower))
        q_hi = float(basis.quantile(spec.target_trim_upper))
        trim_mask = (y >= q_lo) & (y <= q_hi)
        if lockbox_applied:
            trim_mask = trim_mask | pd.Series(is_test_row, index=y.index)
        n_trimmed = int((~trim_mask).sum())
        keep = trim_mask.to_numpy()
        X = X[keep].reset_index(drop=True)
        y = y[keep].reset_index(drop=True)
        kept_labels = [l for l, k in zip(kept_labels, keep) if k]
        is_test_row = is_test_row[keep]
        notes.append(
            f"Target trimmed before split: {n_trimmed} row(s) removed "
            f"(quantiles [{spec.target_trim_lower:.2f}, {spec.target_trim_upper:.2f}] "
            f"-> range [{q_lo:.3g}, {q_hi:.3g}]).")

    positions = np.arange(len(X))

    # ── classification guards + label encoding ────────────────────────────
    label_encoder = None
    if task_type == "classification" and y.nunique() < 2:
        raise SplitError(
            f"Single-class target: {y.nunique()} unique value(s) after removing "
            "missing values. Classification needs at least 2 classes.")
    if _target_is_categorical(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    cv_strategy: str = "standard"
    cv_groups_train: Optional[np.ndarray] = None

    # ── branch 1 · grouped ────────────────────────────────────────────────
    if strategy == "grouped":
        groups = to_numpy_1d(df.loc[kept_labels, spec.entity_id_col])
        y_arr = to_numpy_1d(y)

        gss = GroupShuffleSplit(n_splits=1, test_size=(spec.val_size + spec.test_size),
                                random_state=spec.random_state)
        train_idx, temp_idx = next(gss.split(positions, y_arr, groups))
        # Same-entity rows must stay together in CV too, or the folds leak.
        cv_strategy = "group"
        cv_groups_train = groups[train_idx]

        rel_val = spec.val_size / (spec.val_size + spec.test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - rel_val),
                                 random_state=spec.random_state)
        val_rel, test_rel = next(gss2.split(positions[temp_idx], y_arr[temp_idx],
                                            groups[temp_idx]))
        idx_train = train_idx
        idx_val = temp_idx[val_rel]
        idx_test = temp_idx[test_rel]

    # ── branch 2 · chronological ──────────────────────────────────────────
    elif strategy == "chronological":
        cv_strategy = "time"
        order_frame = df.loc[kept_labels, [spec.datetime_col]].copy()
        order_frame["_pos"] = np.arange(len(order_frame))
        # mergesort is stable: ties keep their original order, so the partition
        # is reproducible when several rows share a timestamp.
        order_frame = order_frame.sort_values(spec.datetime_col, kind="mergesort")

        n_total = len(order_frame)
        n_train = int(n_total * spec.train_size)
        n_val = int(n_total * spec.val_size)
        idx_train = order_frame.iloc[:n_train]["_pos"].to_numpy()
        idx_val = order_frame.iloc[n_train:n_train + n_val]["_pos"].to_numpy()
        idx_test = order_frame.iloc[n_train + n_val:]["_pos"].to_numpy()

    # ── branch 3 · lockbox ────────────────────────────────────────────────
    elif strategy == "lockbox":
        idx_test = positions[is_test_row]
        rest = positions[~is_test_row]
        strat = None
        if spec.stratify and task_type == "classification":
            candidate = y.iloc[rest]
            if candidate.value_counts().min() >= 2:
                strat = candidate
        try:
            idx_train, idx_val = train_test_split(
                rest, test_size=_relative_val(spec),
                random_state=spec.random_state, stratify=strat)
        except ValueError:
            idx_train, idx_val = train_test_split(
                rest, test_size=_relative_val(spec), random_state=spec.random_state)
        notes.append(
            f"Test set from the upload lockbox: n={len(idx_test)}, sealed before "
            "feature engineering or selection could see it. The remaining rows "
            "were divided into train and validation.")

    # ── branch 4 · stratified / random ────────────────────────────────────
    else:
        strat_all = y if (strategy == "stratified" and task_type == "classification") else None
        idx_train, idx_temp = train_test_split(
            positions, test_size=(spec.val_size + spec.test_size),
            random_state=spec.random_state, stratify=strat_all)
        rel_val = spec.val_size / (spec.val_size + spec.test_size)
        strat_temp = y.iloc[idx_temp] if strat_all is not None else None
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=(1 - rel_val),
            random_state=spec.random_state, stratify=strat_temp)

    idx_train = np.asarray(idx_train, dtype=int)
    idx_val = np.asarray(idx_val, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)

    split = Split(
        strategy=strategy,
        X_train=X.iloc[idx_train], X_val=X.iloc[idx_val], X_test=X.iloc[idx_test],
        y_train=to_numpy_1d(y.iloc[idx_train]),
        y_val=to_numpy_1d(y.iloc[idx_val]),
        y_test=to_numpy_1d(y.iloc[idx_test]),
        feature_names=list(feature_cols),
        train_labels=[kept_labels[i] for i in idx_train],
        val_labels=[kept_labels[i] for i in idx_val],
        test_labels=[kept_labels[i] for i in idx_test],
        train_positions=idx_train.tolist(),
        val_positions=idx_val.tolist(),
        test_positions=idx_test.tolist(),
        cv_strategy=cv_strategy,
        cv_groups_train=cv_groups_train,
        label_encoder=label_encoder,
        lockbox_applied=lockbox_applied,
        n_trimmed_rows=n_trimmed,
        notes=notes,
    )
    split.assert_disjoint()
    return split
