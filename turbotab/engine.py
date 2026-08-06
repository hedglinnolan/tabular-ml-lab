"""
turbotab.engine — a thin adapter over the real `ml/` engine.

**No analysis logic lives here.** Every number this module hands out was computed
by a function in `ml/`. The only work done in this file is:

  * putting the repository root on ``sys.path`` so ``ml.*`` resolves no matter
    where the server was started from,
  * reading an uploaded file into a DataFrame,
  * turning the engine's dataclasses into JSON-safe dictionaries,
  * merging two already-sorted finding streams into one list, using the engine's
    own severity vocabulary as the sort key,
  * refusing, loudly, in the two cases where the engine would otherwise answer a
    question it cannot actually answer (empty frame, duplicate target label).

If you find yourself computing a statistic in this file, it belongs in `ml/`.

Headless: this module imports and runs with Streamlit absent. See
``docs/turbotab/ARCHITECTURE.md`` §01 and ``turbotab/test_skeleton.py::
test_engine_imports_with_streamlit_blocked``.
"""
from __future__ import annotations

import dataclasses
import enum
import io
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The adapter is the only place that knows where the engine lives.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml import binary_text, card_evidence, import_doctor, missingness_plan, triage  # noqa: E402
from ml.dataset_profile import (                            # noqa: E402
    DatasetProfile,
    DataWarning,
    compute_dataset_profile,
)
from ml.import_doctor import ShapeFinding                   # noqa: E402


class EngineRefusal(Exception):
    """The engine was not asked a question it can answer.

    Raised instead of returning a plausible-looking answer. The governing rule
    in `PRODUCT_VISION.md` §07 is *never assert falsely*; refusing is allowed,
    guessing is not.
    """


# ─────────────────────────────────────────────────────────────────────────────
# JSON safety
#
# The engine returns numpy scalars, Enums, tuples and NaN. `json.dumps` emits a
# bare `NaN` token for the last of those, which is not valid JSON and makes
# `JSON.parse` throw in the browser — so the frontend would see a network error
# instead of a dataset with missing values. Everything crossing the wire goes
# through `_plain` first.
# ─────────────────────────────────────────────────────────────────────────────

def _plain(value: Any) -> Any:
    """Recursively convert engine output into something `json.dumps` accepts."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        # NaN and ±inf are real engine outputs (skew of a constant column, for
        # one). JSON has no literal for them; null is the honest carrier.
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        # Class-count keys arrive as numpy scalars or Timestamps; JSON object
        # keys must be strings.
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

def read_table(raw: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Read an uploaded delimited file exactly as the Streamlit app would.

    Deliberately plain: `pd.read_csv` with its default type inference, because
    that is the frame `ml.import_doctor` was written to inspect. The doctor's
    job is to catch what pandas' inference *missed* — "72 kg", "1,200", a
    decimal comma — so pre-cleaning here would delete its findings before it
    ever saw them. Equally, reading everything as `str` would make
    `check_numeric_stored_as_text` fire on every numeric column.
    """
    sep = "\t" if filename.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    if df.empty or len(df.columns) == 0:
        raise EngineRefusal(
            f"'{filename}' parsed to {len(df)} rows and {len(df.columns)} columns. "
            "There is nothing to diagnose."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# The three engine calls
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(df: pd.DataFrame, target: Optional[str] = None) -> List[ShapeFinding]:
    """Structural diagnosis, from two engine modules rather than one.

    `ml.import_doctor.diagnose` is pure — it never mutates `df` and applies no
    fix. Preview before apply is the engine's existing contract, not something
    added here.

    `ml.binary_text` is folded in on top of it because the doctor reaches for
    numeric coercion before binary when text parses truthy: a `True`/`False`
    column with blanks draws "Convert to numbers" at *high* confidence, which is
    arithmetically fine and diagnostically wrong. The binary reading supersedes
    the numeric one for the columns it claims — two proposals for one column
    would make the user adjudicate the engine's own disagreement (GUIDED-001).

    Once a target is known it is routed to a different question. For a feature
    the reading is the decision; for the outcome the reading is nearly forced
    and the decision is *which level is the event*, because that sets the sign
    of every effect estimate. Before a target is chosen there is no such
    column, and every column is read as a feature.

    Both merges are decisions about *which* finding to show, not new statistics;
    the statistics are computed in `ml/`.
    """
    return binary_text.diagnose_with_binary(df, import_doctor.diagnose(df),
                                            target=target)


def detect_task_type(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Task-type detection. Straight through to `ml.triage.detect_task_type`.

    One guard: with duplicate column labels `df[target]` yields a DataFrame
    rather than a Series, and `detect_task_type` would compute `.nunique()`
    across a frame and return a confident answer about the wrong thing. The
    engine reports duplicate labels as its own finding; until that is fixed the
    question has no single answer, so we refuse it.
    """
    if target not in list(df.columns):
        raise EngineRefusal(f"No column named '{target}' in this table.")
    if list(df.columns).count(target) > 1:
        raise EngineRefusal(
            f"'{target}' names {list(df.columns).count(target)} different columns in "
            "this file. Rename them before choosing it as the target — until then "
            "there is no single column to detect a task type for."
        )
    return triage.detect_task_type(df, target)


def profile(
    df: pd.DataFrame,
    target: Optional[str] = None,
    task_type: Optional[str] = None,
) -> DatasetProfile:
    """Dataset profile. Straight through to `ml.dataset_profile.compute_dataset_profile`."""
    return compute_dataset_profile(df, target_col=target, task_type=task_type)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

# The engine speaks two severity vocabularies: `ShapeFinding.severity`
# (critical/warning/info) and `WarningLevel` (critical/warning/caution/info).
# This is the union, in the engine's own order. It is the single ordering
# judgment this adapter makes, and it introduces no new tiers.
SEVERITY_RANK = {"critical": 0, "warning": 1, "caution": 2, "info": 3}
CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _with_deferral(d: Dict[str, Any]) -> Dict[str, Any]:
    """Attach where this finding goes if deferred, and what to call that step.

    Delegated to `ml.router.defer_destination` rather than decided here: the
    Router owns "which step can act on this", and a destination chosen by the
    renderer is one the record cannot honor. The API has always rejected a
    deferral with no target — the button simply never said what the target was
    (GUIDED-008).
    """
    from ml import router
    step, label = router.defer_destination(d)
    d["defer_target"] = step
    d["defer_target_label"] = label
    return d


def shape_finding_to_dict(f: ShapeFinding) -> Dict[str, Any]:
    """One structural finding, flattened. Field-for-field; nothing is invented."""
    return {
        "id": f.id,
        "source": "structure",
        "severity": f.severity,
        "confidence": f.confidence,
        "title": f.title,
        "detail": f.detail,
        "why_it_matters": f.why_it_matters,
        "fix_label": f.fix_label or None,
        "fix_kind": f.fix_kind,
        # `auto_suggestable` is the engine's property, not a re-derivation. It is
        # the switch behind "high confidence is the only tier the UI may
        # pre-select" (ARCHITECTURE.md §02).
        "auto_suggestable": bool(f.auto_suggestable),
        "affected_columns": _plain(f.affected_columns),
        "params": _plain(f.params),
        "suggested_actions": [],
    }


# Which `FeatureProfile` flag names the columns a warning is about. The
# `DataWarning` dataclass carries no column list — `affected_models`, and
# nothing else — so the columns have to come from the per-feature profile the
# same computation already produced.
#
# **Read from the profile, never parsed out of `detailed_message`.** The prose
# does name them, and reading it back would be a substring match against a
# sentence somebody may reword — which is the failure this project has now
# filed five times under a different name.
#
# Each category reads a STRUCTURED source. `physio_plausibility_flags` is prose
# — `'glucose: 9.2% outside NHANES reference (70.0-200.0 mg/dL)'` — and the
# column name is before the colon, so reading it back would be a substring match
# against a sentence somebody may reword. `card_evidence.plausibility_report`
# returns the same finding with a `column` key, and that is what is read.
def _outlier_columns(w, prof, df):
    return list(getattr(prof, "features_with_outliers", None) or [])


def _cardinality_columns(w, prof, df):
    return list(getattr(prof, "high_cardinality_features", None) or [])


def _missing_columns(w, prof, df):
    if df is None:
        return []
    return [str(c) for c in df.columns
            if not isinstance(df[c], pd.DataFrame) and bool(df[c].isna().any())]


def _physiologic_columns(w, prof, df):
    if df is None:
        return []
    report = card_evidence.plausibility_report(df)
    seen: List[str] = []
    for tier in ("impossible", "improbable"):
        for row in report.get(tier) or []:
            column = str(row.get("column") or "")
            if column and column not in seen:
                seen.append(column)
    return seen


_WARNING_COLUMNS = {
    "outliers": _outlier_columns,
    "cardinality": _cardinality_columns,
    "missingness": _missing_columns,
    "physiologic_plausibility": _physiologic_columns,
}


def warning_columns(w: DataWarning, prof: Optional[DatasetProfile] = None,
                    df: Optional[pd.DataFrame] = None) -> List[str]:
    """The columns this warning is about, or an empty list where it is about
    the table rather than about columns.

    `sample_size`, `imbalance` and `dimensionality` are properties of the whole
    dataset, and returning every column for them would be worse than returning
    none: an option offered over "all 400 columns" because the table is small
    is an option about nothing.
    """
    reader = _WARNING_COLUMNS.get(w.category)
    if reader is None:
        return []
    try:
        return [str(c) for c in reader(w, prof, df)]
    except Exception as exc:
        from turbotab import devchecks
        devchecks.swallowed(
            f"engine.warning_columns::{w.category}", exc,
            "a profile warning lost the columns it is about, so every option "
            "it suggests will have nothing to act on")
        return []


def data_warning_to_dict(w: DataWarning, ordinal: int,
                         prof: Optional[DatasetProfile] = None,
                         df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """One profile warning, flattened.

    `DataWarning` carries no confidence field, so `confidence` is None and
    `auto_suggestable` is False — a profile warning may never pre-select
    anything. It also carries no fix, hence `fix_kind='none'`: the engine has
    named a problem, not a repair.

    `affected_columns` was hardcoded empty, which was honest about the dataclass
    and left every option the warning suggests with nothing to act on
    (`GUIDED-031`). The columns come from the per-feature profile now.
    """
    return {
        "id": f"profile_{w.category}_{ordinal}",
        "source": "profile",
        "severity": w.level.value if isinstance(w.level, enum.Enum) else str(w.level),
        "confidence": None,
        "title": w.short_message,
        "detail": w.detailed_message,
        "why_it_matters": "",
        "fix_label": None,
        "fix_kind": "none",
        "auto_suggestable": False,
        "affected_columns": warning_columns(w, prof, df),
        "params": {"category": w.category,
                   "affected_models": _plain(w.affected_models),
                   # Named so a consumer can tell "about the table" from "about
                   # columns we failed to find" — two different claims.
                   "scope": ("columns" if w.category in _WARNING_COLUMNS
                             else "dataset")},
        "suggested_actions": _plain(w.suggested_actions),
    }


def profile_to_dict(p: DatasetProfile) -> Dict[str, Any]:
    """The whole profile as plain data. `_plain` walks the nested dataclasses."""
    return _plain(p)


# ─────────────────────────────────────────────────────────────────────────────
# Preview
#
# `import_doctor.apply_fix` already returns `(new_frame, description)` and never
# mutates its input — the engine has always worked preview-first. Everything
# below is the diff of two frames the engine produced. No repair is computed
# here, and no frame produced here is installed anywhere.
# ─────────────────────────────────────────────────────────────────────────────

MAX_PREVIEW_ROWS = 8


def _differs(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Elementwise "this cell changed", with NaN == NaN.

    Two missing values are the same absence, and `!=` disagrees.
    """
    try:
        ne = before.ne(after)
    except (TypeError, ValueError):
        # Mixed-type object columns can refuse to compare. Falling back to text
        # can only over-report a change, never hide one.
        ne = before.astype(str).ne(after.astype(str))
    both_missing = before.isna() & after.isna()
    return ne & ~both_missing


def _differs_as_shown(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Elementwise "this cell *reads* differently", with NaN == NaN.

    Deliberately not `_differs`. That one compares values, which is right for
    deciding whether row identity survived; this one compares what the user will
    see in the two panes, which is right for deciding what to count and what to
    highlight — and the two disagree in both directions:

      * `"1200"` → `1200` is a dtype change and no visible change. Value
        comparison called it 8 changed cells while every rendered cell was
        identical, so the panel said "8 cells change" over an unmarked table.
      * `True` → `1.0` is the reverse: pandas holds `True == 1`, so value
        comparison saw nothing while the panes plainly differ.

    The count, the highlighted columns and the highlighted cells now all come
    from this one function, so the header and the table cannot contradict each
    other. Type-only changes are still reported — by the `dtype of …` row in the
    statistics, which is where a type change belongs.
    """
    try:
        ne = before.astype(str).ne(after.astype(str))
    except (TypeError, ValueError):
        ne = before.ne(after)
    both_missing = before.isna() & after.isna()
    return ne & ~both_missing


def _cell(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT else str(v)


def preview_fix(df: pd.DataFrame, finding: ShapeFinding,
                choice: Optional[str] = None) -> Dict[str, Any]:
    """What this finding's fix would change, computed but not applied.

    The frame handed to `apply_fix` is a copy, and the result is described and
    discarded. Nothing here touches the project.

    The interesting part is `row_identity_preserved`. Four of the nine fix kinds
    end in `.reset_index(drop=True)` — `promote_header`, `drop_empty_rows`,
    `drop_rows`, and `melt_repeated` rebuilds the index outright. This project
    keys rows by index label (`TRANSITION_PLAN.md` §02.2), so a renumbering
    means every stored label now names a *different row*. That is the corruption
    the transition plan calls its highest risk, and a preview that did not
    mention it would be hiding the most consequential thing about the fix.

    It is detected by content, not by fix kind: the surviving labels are looked
    up in the original frame and the rows compared. Dropping trailing footer
    rows from a clean `RangeIndex` renumbers nothing that matters and is
    correctly reported as preserved; dropping from the middle is not.
    """
    if finding.fix_kind == "none":
        return {
            "finding_id": finding.id,
            "fix_kind": "none",
            "fix_label": finding.fix_label or None,
            "applicable": False,
            "description": "No automatic change is possible here; this needs a human decision.",
        }

    before = df
    after, description = _dispatch_fix(df.copy(deep=True), finding, choice=choice)

    cols_before, cols_after = list(before.columns), list(after.columns)
    added = [c for c in cols_after if c not in cols_before]
    removed = [c for c in cols_before if c not in cols_after]
    common_cols = [c for c in cols_after if c in cols_before]

    # ── did the surviving rows keep their identities? ──────────────────────
    labels_known = bool(len(after)) and bool(after.index.isin(before.index).all())
    identity_ok, identity_note = False, None
    if labels_known and not after.index.has_duplicates and common_cols:
        try:
            aligned = before.loc[after.index, common_cols]
            # A fix edits the columns it names, so those are excluded from the
            # test. Identity is judged on every *remaining* column — all of
            # them, not any of them. Judging on "any column still lines up"
            # would be satisfied by a constant column, which lines up under any
            # renumbering whatsoever and would wave through exactly the
            # corruption this check exists to catch.
            touched_by_fix = set(map(str, finding.affected_columns or []))
            witness = [c for c in common_cols if str(c) not in touched_by_fix] or common_cols
            identity_ok = not bool(
                _differs(aligned[witness], after[witness]).to_numpy().any()
            )
        except (KeyError, ValueError, IndexError):
            identity_ok = False
    if len(after) == len(before) and after.index.equals(before.index):
        identity_ok = True
    if not identity_ok:
        identity_note = (
            "This fix renumbers the rows, so every row label stored before it — "
            "including a sealed test set — would afterwards name a different row. "
            "The engine resets the index for this repair."
        )

    # ── the cell-level diff, over rows and columns present on both sides ────
    changed_count, changed_cols, changed_labels = 0, [], []
    if identity_ok and common_cols and len(after):
        try:
            aligned = before.loc[after.index, common_cols]
            mask = _differs_as_shown(aligned, after[common_cols])
            changed_count = int(mask.to_numpy().sum())
            changed_cols = [c for c in common_cols if bool(mask[c].any())]
            changed_labels = list(after.index[mask.any(axis=1)])
        except (KeyError, ValueError) as exc:
            # A well. When this fires the preview reports NO changed cells and
            # no changed columns, which reads as "this repair changes nothing" —
            # the most misleading thing a before/after card can say. Nothing
            # surfaces it; that is what the harness is for.
            from turbotab import devchecks
            devchecks.swallowed(
                "engine.preview_fix::changed-cell-count", exc,
                "the preview will report zero changed cells, which reads as "
                "'this repair changes nothing'")

    # ── the rows to show ───────────────────────────────────────────────────
    # When identity holds, before and after are shown *aligned by label*, with
    # the rows that changed first, and a cell-level diff is meaningful.
    #
    # When it does not, there is no correspondence between a before row and an
    # after row, so no cell diff can honestly be drawn. Both frames' first rows
    # are shown side by side as samples, carrying their own labels, and nothing
    # is marked changed — an unaligned diff would paint every cell as modified
    # and read as data loss when the data is merely reshaped.
    display_cols = ([c for c in common_cols if c in changed_cols or c in cols_after[:4]]
                    + added)[:6] or cols_after[:6]
    rows = []

    if identity_ok:
        show = list(changed_labels[:MAX_PREVIEW_ROWS])
        for lbl in list(after.index)[:MAX_PREVIEW_ROWS * 3]:
            if len(show) >= MAX_PREVIEW_ROWS:
                break
            if lbl not in show:
                show.append(lbl)
        for lbl in show[:MAX_PREVIEW_ROWS]:
            b_row, a_row, flags = [], [], []
            for c in display_cols:
                b_txt = _cell(before.at[lbl, c]) if (lbl in before.index and c in cols_before) else ""
                a_txt = _cell(after.at[lbl, c]) if (lbl in after.index and c in cols_after) else ""
                b_row.append(b_txt)
                a_row.append(a_txt)
                flags.append(b_txt != a_txt)
            rows.append({"label": _plain(lbl), "before_label": _plain(lbl),
                         "before": b_row, "after": a_row, "changed": flags})
    else:
        b_head, a_head = before.head(MAX_PREVIEW_ROWS), after.head(MAX_PREVIEW_ROWS)
        for i in range(max(len(b_head), len(a_head))):
            b_lbl = b_head.index[i] if i < len(b_head) else None
            a_lbl = a_head.index[i] if i < len(a_head) else None
            b_row = [_cell(b_head.at[b_lbl, c]) if (b_lbl is not None and c in cols_before) else ""
                     for c in display_cols]
            a_row = [_cell(a_head.at[a_lbl, c]) if (a_lbl is not None and c in cols_after) else ""
                     for c in display_cols]
            rows.append({
                "label": _plain(a_lbl) if a_lbl is not None else "",
                "before_label": _plain(b_lbl) if b_lbl is not None else "",
                "before": b_row, "after": a_row,
                "changed": [False] * len(display_cols),
            })

    def _missing(frame: pd.DataFrame, cols: List[str]) -> int:
        cols = [c for c in cols if c in frame.columns]
        return int(frame[cols].isna().to_numpy().sum()) if cols else 0

    touched = list(finding.affected_columns) or changed_cols
    stats = [
        {"key": "rows", "before": int(len(before)), "after": int(len(after))},
        {"key": "columns", "before": len(cols_before), "after": len(cols_after)},
        {"key": "cells changed", "before": 0, "after": changed_count},
    ]
    if touched:
        stats.append({"key": f"missing in {touched[0]}",
                      "before": _missing(before, touched[:1]),
                      "after": _missing(after, touched[:1])})
        if touched[0] in cols_before and touched[0] in cols_after:
            stats.append({"key": f"dtype of {touched[0]}",
                          "before": str(before[touched[0]].dtype),
                          "after": str(after[touched[0]].dtype)})

    return {
        "finding_id": finding.id,
        "fix_kind": finding.fix_kind,
        "fix_label": finding.fix_label or None,
        "applicable": True,
        "description": description,
        "confidence": finding.confidence,
        "auto_suggestable": bool(finding.auto_suggestable),
        "shape": {"before": [int(len(before)), len(cols_before)],
                  "after": [int(len(after)), len(cols_after)]},
        "columns_added": [str(c) for c in added],
        "columns_removed": [str(c) for c in removed],
        "rows_removed": max(0, int(len(before) - len(after))),
        "rows_added": max(0, int(len(after) - len(before))),
        "row_identity_preserved": bool(identity_ok),
        "row_identity_note": identity_note,
        "changed_cells": changed_count,
        "changed_columns": [str(c) for c in changed_cols],
        "sample": {"columns": [str(c) for c in display_cols], "rows": rows,
                   "aligned": bool(identity_ok)},
        "stats": _plain(stats),
    }


def _dispatch_fix(df: pd.DataFrame, finding: ShapeFinding,
                  choice: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Route one fix to the engine module that owns its kind.

    `ml/import_doctor.py` owns nine kinds and is frozen as engine-move-only
    (`TRANSITION_PLAN.md` §05), so `read_as_binary` and `set_positive_class`
    live in `ml/binary_text.py` and are dispatched here rather than added to the
    doctor's own table.
    """
    if finding.fix_kind == "read_as_binary":
        return binary_text.apply_read_as_binary(df, finding)
    if finding.fix_kind == "set_positive_class":
        return binary_text.apply_positive_class(df, finding, event=choice)
    return import_doctor.apply_fix(df, finding)


def apply_fix(df: pd.DataFrame, finding: ShapeFinding,
              choice: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Apply one fix, on a deep copy of the frame.

    The engine documents that it never mutates its input; the copy means a
    future change to that promise cannot corrupt a project that is still
    holding the original frame.

    `choice` carries an answer the finding cannot supply itself — today, which
    level of the outcome is the event. A fix that needs one raises rather than
    defaulting.
    """
    return _dispatch_fix(df.copy(deep=True), finding, choice=choice)


def fix_encoding(df: pd.DataFrame,
                 finding: ShapeFinding) -> Optional[Dict[str, Any]]:
    """WHICH ORIGINAL VALUE BECOMES 1, for a fix that rewrites a column to 0/1.

    `GUIDED-157`. The bulk repair recorded the columns it touched and not the
    mapping, so a record saying *"1 feature (`gender`) was read as binary"*
    left *"is the coefficient on `male` or on `female`"* with no answer
    anywhere. No number was wrong; a reported number was uninterpretable, which
    is trap #7's shape — the machine-readable form lossier than the sentence —
    crossed with the governing rule.

    **This reports what the transform does; it does not decide anything.** The
    plan comes from `ml.binary_text.read_as_binary_plan`, which is the same
    call `apply_read_as_binary` makes on the same frame one line later, so the
    record and the rewrite cannot disagree. Nothing here re-derives a mapping.

    **The spellings are the file's own**, for the reason `apply_read_as_binary`
    already states about its description: the tokens the plan compares on are
    normalized, and a methods sentence saying `true = 1` about a column written
    `True` describes a frame nobody uploaded. `positive`/`negative` are the
    spelling the transform's own sentence quotes; `positive_values` /
    `negative_values` are **every** spelling that maps to that side, because a
    column holding `Male` and `male` has two of them and a record naming one
    would be a claim about half the rows.

    Returns **None** for every other fix kind, and None where the column is no
    longer binary. A kind with no encoding gets no encoding — trap 9's rule at
    the record layer: return nothing rather than a mapping that was invented
    here.
    """
    if finding is None or finding.fix_kind != "read_as_binary":
        return None
    params = finding.params or {}
    column = params.get("column") or (finding.affected_columns or [None])[0]
    if column is None or column not in df.columns:
        return None
    series = df[column]
    if isinstance(series, pd.DataFrame):                # duplicate labels
        return None
    plan = binary_text.read_as_binary_plan(series)
    if not plan:
        return None

    positive, negative = plan["positive"], plan["negative"]
    spellings: Dict[str, List[str]] = {positive: [], negative: []}
    for value in series.tolist():
        token = binary_text._normalize(value)
        if token in spellings and str(value) not in spellings[token]:
            spellings[token].append(str(value))

    pos_values = spellings[positive] or [str(positive)]
    neg_values = spellings[negative] or [str(negative)]
    mapping = {v: 1 for v in pos_values}
    mapping.update({v: 0 for v in neg_values})
    return {
        "column": str(column),
        "positive": pos_values[0],
        "negative": neg_values[0],
        "positive_values": pos_values,
        "negative_values": neg_values,
        "mapping": mapping,
        # Whether the direction came from `KNOWN_PAIRS` or from the plan's
        # declared sorted-order fallback. The finding already says this to the
        # user; the record says it to everything downstream, because "the
        # engine recognized this pair" and "the engine picked deterministically
        # and said so" are different claims about the same 1.
        "positive_known": bool(plan["positive_known"]),
        "n_positive": int(plan["counts"][positive]),
        "n_negative": int(plan["counts"][negative]),
        "n_missing": int(plan["n_missing"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
#
# Every finding card puts its evidence on the table: the flagged entries with
# their values, or the plot the claim is about (GUIDED-003 / GUIDED-005). None
# of it is computed here — `ml.card_evidence` and `ml.missingness_plan` do the
# work and this hands the result across the wire.
# ─────────────────────────────────────────────────────────────────────────────

def plausibility(df: pd.DataFrame) -> Dict[str, Any]:
    """Impossible and improbable entries, in two tiers, with rows named."""
    return _plain(card_evidence.plausibility_report(df))


def histograms(df: pd.DataFrame, columns: Optional[List[str]] = None,
               page: int = 0, per_page: int = 6) -> Dict[str, Any]:
    return _plain(card_evidence.histogram_gallery(df, columns, page, per_page))


def correlations(df: pd.DataFrame,
                 columns: Optional[List[str]] = None) -> Dict[str, Any]:
    return _plain(card_evidence.correlation_matrix(df, columns))


def column_histogram(df: pd.DataFrame, column: str) -> Optional[Dict[str, Any]]:
    if column not in df.columns:
        raise EngineRefusal(f"No column named '{column}' in this table.")
    return _plain(card_evidence.histogram(df[column]))


def missingness(df: pd.DataFrame,
                mechanisms: Optional[Dict[str, str]] = None,
                provenance: Optional[Dict[str, Any]] = None
                ) -> List[Dict[str, Any]]:
    """Dtype-routed missingness decisions, each naming its column.

    `mechanisms` is what the user has ALREADY answered to §07's question, per
    column, and it is passed through rather than computed here — the engine
    takes a frame and knows nothing about a project. It decides whether the
    card's concern about a signal-destroying fill reads *"this is refused"* or
    *"this would be refused if you answered yes"* (`GUIDED-163`).
    """
    # `AUDIT-028`. THIS DOOR HAS NO FOLDS and the card must say so, because the
    # card is one click away from the declaration it writes and
    # `test_the_two_missingness_doors_agree` compares the two sentences.
    # `turbotab/training.py:416` is the source: nothing under `turbotab/`
    # imports `KFold`, `cross_val_score` or `cross_validate`.
    from turbotab import missingness as _miss

    return _plain(missingness_plan.missingness_cards(
        df, mechanisms=mechanisms, provenance=provenance,
        scope=_miss.TRAIN_ROWS))


def imputation_preview(df: pd.DataFrame, column: str,
                       strategy: str) -> Optional[Dict[str, Any]]:
    if column not in df.columns:
        raise EngineRefusal(f"No column named '{column}' in this table.")
    return _plain(missingness_plan.imputation_preview(df[column], strategy))


def find_shape_finding(structural: List[ShapeFinding], finding_id: str) -> ShapeFinding:
    for f in structural:
        if f.id == finding_id:
            return f
    raise EngineRefusal(
        f"No structural finding '{finding_id}' in this table. Findings are recomputed "
        "after every change, so an id from before a fix may no longer exist."
    )


def rank_findings(
    structural: List[ShapeFinding],
    prof: Optional[DatasetProfile] = None,
    lens: Sequence[str] = (),
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Merge both finding streams into one ranked list, READ UNDER THE LENS.

    Sorted by the engine's severity, then its confidence, then id — a total
    order, so the same table always ranks the same way. `import_doctor.diagnose`
    already sorts its own output by severity; this re-sorts only because the two
    streams have to interleave.

    **This is where the lens reaches the diagnosis, and the placement is the
    answer to a question worth stating.** `OPENING_SEQUENCE.md` orders the lens
    before the diagnosis because the diagnosis is field-sensitive; the
    detectors in `ml/import_doctor.py` take a frame and nothing else, are
    field-blind by construction, and are frozen. So the lens is a parameter of
    **the function that produces the finding list the app presents**, not of the
    detector pass underneath it.

    That is not a fig leaf, and there is a reason it must not be one. Reframing
    ANNOTATES and never deletes: a user who reads *"these are different
    analytes, not one analyte measured twice"* and still wants to reshape the
    table can. `apply` and `preview` re-run `diagnose()` and need the real
    `fix_kind` to execute the repair — so a lens that erased the reading at
    GENERATION would take that route away, and the annotation would become a
    deletion by another name.

    The governing rule is about what the app **asserts**, not about what it
    computes. Nothing reaches a user except through here, and
    `test_the_lens_reaches_every_finding_list_the_app_presents` is what makes
    that a check rather than a habit.
    """
    items = [_with_deferral(shape_finding_to_dict(f)) for f in structural]
    if prof is not None:
        items += [_with_deferral(data_warning_to_dict(w, i, prof, df))
                  for i, w in enumerate(prof.warnings)]

    if lens and df is not None:
        from turbotab import packs as _packs
        items = _packs.reframe(items, lens, df)
        # `_with_deferral` HERE TOO, and its absence was `GUIDED-153`. The pack
        # stream was appended raw, so every pack finding shipped with
        # `defer_target: null`, the button read "Decide later" instead of naming
        # a step, and pressing it recorded a target the API's fallback chose —
        # which `ml.router.defer_destination`'s own docstring forbids.
        items = items + [_with_deferral(f) for f in _packs.findings(df, lens)]

    items.sort(key=lambda d: (
        SEVERITY_RANK.get(d["severity"], 99),
        CONFIDENCE_RANK.get(d["confidence"], 1),   # unrated sits with 'medium'
        str(d["id"]),
    ))
    for i, d in enumerate(items):
        d["rank"] = i
    return items


# ─────────────────────────────────────────────────────────────────────────────
# The seal — drawn against the recorded grain answer, never against a guess
# ─────────────────────────────────────────────────────────────────────────────

def draw_holdout(df: pd.DataFrame, target: str, task_type: str,
                 grain: Dict[str, Any], fraction: float = 0.15,
                 seed: int = 42, time_col: Optional[str] = None,
                 temporal: bool = False) -> Dict[str, Any]:
    """Choose the rows to seal, using the grain answer the user gave.

    Returns the labels plus the disclosure the seal has to carry. Three of the
    four bases draw differently and all four are reported:

    * `grouped` — whole people held out, so nobody is on both sides.
    * `repetition_found_grouping_abandoned` — repetition was stated and believed
      but there are too few people to hold any out by person, so the split is by
      row and SAYS SO. Not the same claim as "no repetition".
    * `cross_sectional` — the user stated one row per person and nothing
      contradicted it.
    * `undetermined` — the user does not know. Sealed anyway, by row, and
      labeled exploratory: an advisory, not a hard block (constitution §03).

    The achieved ROW fraction is reported, not the requested one. A grouped
    split's `test_size` is a proportion of GROUPS, and with unequal group sizes
    those differ badly — the audit measured 15% requested against 37% of rows
    actually held out (`IMPORT-255`, `grouped-lockbox-fraction-mislabel`).
    """
    # numpy only, deliberately. `turbotab/requirements.txt` states that the
    # whole diagnose -> profile -> detect path needs pandas and numpy and
    # nothing else, and the seal is now on that path. A seeded permutation is
    # all a holdout draw is; reaching for scikit-learn here would put 60 MB
    # behind the one step the Guided door cannot skip.
    rng = np.random.default_rng(seed)

    if target not in df.columns:
        raise EngineRefusal(f"No column named '{target}' in this table.")
    y = df[target]
    eligible = list(df.index[y.notna()])
    if len(eligible) < 10:
        raise EngineRefusal(
            f"Only {len(eligible)} rows have a value for '{target}', which is too "
            f"few to hold any out and still have a study left.")

    basis = grain.get("basis")
    group_col = grain.get("group_col")
    labels: List[Any] = []
    n_test_groups = None

    chronological = False

    # ── `GUIDED-143` · the chronological grouped draw ────────────────────────
    #
    # **It refuses; it does not fall back.** If the user said the task is
    # predicting a later outcome from earlier measurements and this draw cannot
    # honor that, drawing at random and reporting `grouped` would be the false
    # assertion L42 just removed, arriving through the splitter instead of
    # through the sentence. `EngineRefusal` is the app's refuse branch and the
    # governing rule permits it; a silent fallback is the branch it forbids.
    # **REFUSE WHEN A COLUMN WAS NAMED AND CANNOT BE USED; DISCLOSE WHEN NONE
    # WAS NAMED.** The distinction is the whole of this block and it took a
    # correction to get right.
    #
    # The first version refused whenever `temporal` was set and no time column
    # was recorded — which is stronger than the rule asks and is wrong. L42
    # built the honest three-state disclosure for exactly that case:
    # `chronological_requested_not_drawn`, `honored: False`, the sentence in
    # the seal, `exploratory: True`. That is not a *silent* fallback, it is a
    # loud one, and it was accepted. Refusing there would delete a path a user
    # with no clean date column needs — the shelf being shortened.
    #
    # What must never happen is the app being told *which column is time* and
    # then drawing at random anyway. That is the silent fallback, and it is
    # the false assertion L42 removed arriving through the splitter instead of
    # through the sentence. So: named-and-unusable refuses.
    if temporal and time_col:
        if basis != "grouped" or not group_col or group_col not in df.columns:
            raise EngineRefusal(
                "A temporal validation holds out the LATEST people, so it "
                "needs to know who a row belongs to. This table's grain is "
                f"recorded as {basis!r}"
                + (" with no identifier column" if not group_col else "")
                + ". Record the repeated-measures grain first, or clear the "
                  "time column and the split will be drawn at random within "
                  "people and described as that.")
        if time_col not in df.columns:
            raise EngineRefusal(
                f"No column named '{time_col}' in this table, so the temporal "
                f"validation you asked for cannot be drawn.")
        when = pd.to_datetime(df.loc[eligible, time_col],
                              errors="coerce", format="mixed")
        if when.isna().all():
            raise EngineRefusal(
                f"'{time_col}' was recorded as the time column and none of its "
                f"values parse as a date, so the held-out set cannot be the "
                f"latest one.")
        unreadable = int(when.isna().sum())
        if unreadable and unreadable > len(when) * 0.10:
            raise EngineRefusal(
                f"{unreadable} of {len(when)} rows have no readable date in "
                f"'{time_col}'. Ordering people by their last observation "
                f"would put those people somewhere arbitrary, and the seal "
                f"would report a chronology it did not draw.")

    if temporal and time_col:
        # WHOLE PEOPLE, ORDERED BY THEIR LAST OBSERVATION. A person split
        # across the boundary is the grain violation the seal exists to
        # prevent, so a chronological split that broke grouping would trade
        # one leak for another. Ordering by `max` rather than by `min` or by
        # the mean is the choice that makes the held-out people the ones whose
        # LAST visit is latest — anything else can hold out a person whose
        # follow-up runs past the training data.
        groups = df.loc[eligible, group_col]
        last_seen = when.groupby(groups.values).max()
        # `NaT` last-seen means every row of that person was unreadable; those
        # people sort last under pandas' default and would be held out for
        # being unparseable rather than for being late. Dropped from the draw
        # and reported, never silently placed.
        ordered = list(last_seen.dropna().sort_values().index)
        undated = [g for g in pd.unique(groups.dropna()) if g not in set(ordered)]
        n_hold = max(1, int(round(len(ordered) * fraction)))
        held = set(ordered[-n_hold:])                      # THE TAIL, not a draw
        labels = [lbl for lbl in eligible if groups.loc[lbl] in held]
        n_test_groups = len(held)
        chronological = True

    elif basis == "grouped" and group_col and group_col in df.columns:
        # Whole people, so nobody is on both sides. Drawn over GROUPS, which is
        # why the achieved row fraction below is reported rather than assumed.
        groups = df.loc[eligible, group_col]
        uniq = list(pd.unique(groups.dropna()))
        rng.shuffle(uniq)
        n_hold = max(1, int(round(len(uniq) * fraction)))
        held = set(uniq[:n_hold])
        labels = [lbl for lbl in eligible if groups.loc[lbl] in held]
        n_test_groups = len(held)
    else:
        strata = None
        if task_type == "classification":
            counts = y.loc[eligible].value_counts()
            if len(counts) >= 2 and counts.min() >= 2 and (counts * fraction).min() >= 1:
                strata = y.loc[eligible]
        if strata is not None:
            # Proportional within each class, so a rare outcome is present on
            # both sides rather than absent from one by luck.
            for _, idx in strata.groupby(strata, observed=True).groups.items():
                members = list(idx)
                rng.shuffle(members)
                take = max(1, int(round(len(members) * fraction)))
                labels.extend(members[:take])
        else:
            members = list(eligible)
            rng.shuffle(members)
            labels = members[:max(1, int(round(len(members) * fraction)))]

    # Report the achieved ROW fraction, never the requested one. A grouped
    # split's fraction is a proportion of GROUPS, and with unequal group sizes
    # those differ badly -- the audit measured 15% requested against 37% of rows
    # actually held out (`IMPORT-255`, `grouped-lockbox-fraction-mislabel`).
    achieved = len(labels) / len(eligible) if len(eligible) else fraction
    return {
        "labels": labels,
        "disclosure": {
            "fraction": float(achieved),
            "fraction_requested": float(fraction),
            "seed": int(seed),
            "n_total": int(len(eligible)),
            "n_test_groups": n_test_groups,
            "exploratory": basis == "undetermined",
            # `GUIDED-143`. What the draw ACTUALLY did, reported by the thing
            # that did it — so the seal states its basis from the draw rather
            # than from the question's answer. The two disagreeing is the
            # defect this row is about.
            "chronological": chronological,
            "time_col": str(time_col) if chronological else None,
            "n_undated_groups": len(undated) if chronological else 0,
            "boundary": (str(pd.Timestamp(min(
                last_seen[g] for g in held)).date())
                if chronological and held else None),
        },
    }


def offer_simulation(df: pd.DataFrame, columns: Sequence[str], binding: str,
                     variant: Optional[str] = None,
                     n: int = 6) -> Dict[str, Any]:
    """What a distribution-learning operation would move, without moving it.

    `GUIDED-031`. A deferred operation has no row-by-row before/after to show
    until it is fitted, so a preview of one has to be a preview of the
    DISTRIBUTION it would learn — how many values it would touch and where the
    bounds fall. That is the real computation on a copy rather than a
    description of one, which is the distinction this project keeps finding to
    matter.

    Computed on the working table and discarded. Nothing here writes.
    """
    cols = [str(c) for c in columns if str(c) in df.columns
            and pd.api.types.is_numeric_dtype(df[str(c)])][:12]
    rows: List[Dict[str, Any]] = []
    total = 0
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        entry: Dict[str, Any] = {"column": col, "n": int(len(s))}
        if binding == "outliers" and variant == "winsorize":
            low, high = float(s.quantile(0.01)), float(s.quantile(0.99))
            touched = int(((s < low) | (s > high)).sum())
            entry.update(low=round(low, 4), high=round(high, 4),
                         n_touched=touched,
                         observed_min=round(float(s.min()), 4),
                         observed_max=round(float(s.max()), 4))
        elif binding in ("impute_median", "impute_mean"):
            fill = float(s.median() if binding == "impute_median" else s.mean())
            touched = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            entry.update(fill=round(fill, 4), n_touched=touched)
        elif binding == "indicator":
            touched = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            entry.update(new_column=f"{col}_was_missing", n_touched=touched)
        else:
            entry.update(n_touched=0)
        total += int(entry.get("n_touched") or 0)
        rows.append(entry)
    return {"kind": "distribution", "binding": binding, "variant": variant,
            "columns": [r["column"] for r in rows], "rows": rows[:n],
            "n_columns": len(rows), "n_values_touched": total,
            "truncated": len(rows) > n}
