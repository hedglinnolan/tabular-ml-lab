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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# The adapter is the only place that knows where the engine lives.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml import import_doctor, triage                        # noqa: E402
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

def diagnose(df: pd.DataFrame) -> List[ShapeFinding]:
    """Structural diagnosis. Straight through to `ml.import_doctor.diagnose`.

    That function is pure — it never mutates `df` and applies no fix. Preview
    before apply is the engine's existing contract, not something added here.
    """
    return import_doctor.diagnose(df)


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
# judgement this adapter makes, and it introduces no new tiers.
SEVERITY_RANK = {"critical": 0, "warning": 1, "caution": 2, "info": 3}
CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


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
    items = [shape_finding_to_dict(f) for f in structural]
    if prof is not None:
        items += [data_warning_to_dict(w, i) for i, w in enumerate(prof.warnings)]

    items.sort(key=lambda d: (
        SEVERITY_RANK.get(d["severity"], 99),
        CONFIDENCE_RANK.get(d["confidence"], 1),   # unrated sits with 'medium'
        str(d["id"]),
    ))
    for i, d in enumerate(items):
        d["rank"] = i
    return items
