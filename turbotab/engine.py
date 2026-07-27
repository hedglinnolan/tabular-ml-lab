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
from typing import Any, Dict, List, Optional, Tuple

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


def data_warning_to_dict(w: DataWarning, ordinal: int) -> Dict[str, Any]:
    """One profile warning, flattened.

    `DataWarning` carries no confidence field, so `confidence` is None and
    `auto_suggestable` is False — a profile warning may never pre-select
    anything. It also carries no fix, hence `fix_kind='none'`: the engine has
    named a problem, not a repair.
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
        "affected_columns": [],
        "params": {"category": w.category, "affected_models": _plain(w.affected_models)},
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
        except (KeyError, ValueError):
            pass

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


def missingness(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dtype-routed missingness decisions, each naming its column."""
    return _plain(missingness_plan.missingness_cards(df))


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
) -> List[Dict[str, Any]]:
    """Merge both finding streams into one ranked list.

    Sorted by the engine's severity, then its confidence, then id — a total
    order, so the same table always ranks the same way. `import_doctor.diagnose`
    already sorts its own output by severity; this re-sorts only because the two
    streams have to interleave.
    """
    items = [_with_deferral(shape_finding_to_dict(f)) for f in structural]
    if prof is not None:
        items += [_with_deferral(data_warning_to_dict(w, i))
                  for i, w in enumerate(prof.warnings)]

    items.sort(key=lambda d: (
        SEVERITY_RANK.get(d["severity"], 99),
        CONFIDENCE_RANK.get(d["confidence"], 1),   # unrated sits with 'medium'
        str(d["id"]),
    ))
    for i, d in enumerate(items):
        d["rank"] = i
    return items
