"""
turbotab.project — the minimal `AnalysisProject`.

This is the container `st.session_state` is standing in for today: what is
currently true about one analysis. It holds a dataframe handle, the target, an
append-only list of decisions, and the findings last computed. It computes
nothing — `turbotab.engine` does that — so this module imports neither the
engine nor pandas' analysis surface, and can be reasoned about on its own.

Three commitments carried over from the transition documents:

**Row identity is index labels, never positions** (`TRANSITION_PLAN.md` §02.2).
The lockbox already stores labels; the split block stores positions; page 07
reads the second against a re-fetched frame. They agree only while the index is
a pristine `RangeIndex`, which cohort filtering, row-dropping repairs and joins
all break. This project picks labels and says so in one place —
:meth:`AnalysisProject.rows`, which uses ``.loc`` and refuses ``.iloc``.

**Decisions are append-only** (`ARCHITECTURE.md` §03). The record is what
happened; the project is what is currently true. Changing the target does not
edit the earlier decision, it appends a new one — otherwise "the target was
chosen before exploration" stops being a statement anyone can check.

**Nothing is written to disk** (`ARCHITECTURE.md` §02, the ``_NEVER_PERSIST``
contract). :meth:`to_dict` exists to put a project on the wire, not in a file.
The frame itself never appears in it.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _label(value: Any) -> Any:
    """Coerce one index label to something JSON can carry.

    Labels are scalars — ints, strings, occasionally timestamps. Anything with
    an `item()` is a numpy scalar hiding as a Python one.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


class ProjectError(Exception):
    """The project was asked for something it cannot honestly provide."""


@dataclass
class Decision:
    """One recorded answer. Append-only: never edited, never deleted.

    `kind` is the vocabulary the frontend and the record share:
      ``set_target`` · ``apply`` · ``defer`` · ``dismiss`` · ``flag`` · ``note``
    """
    id: str
    kind: str
    subject: str            # a finding id, a column name, or "" for a bare note
    text: str               # the sentence that goes in the transcript
    at: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "subject": self.subject,
            "text": self.text, "at": self.at, "payload": self.payload,
        }


@dataclass
class AnalysisProject:
    """One analysis. Serializable apart from the frame it points at."""

    id: str
    name: str
    created_at: str
    df: pd.DataFrame                       # the handle — never serialized
    target: Optional[str] = None
    task_type: Optional[str] = None
    task_confidence: Optional[str] = None
    task_reasons: List[str] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    # Set when an answer changes underneath findings that were computed from the
    # old one. The findings are *marked*, never dropped: "the past is editable,
    # never silently destroyed" (PRODUCT_VISION.md §07.4).
    findings_stale: bool = False
    # True when the user contradicted the detection rather than confirming it.
    task_overridden: bool = False
    # (finding_id, frame) for each applied fix, most recent last. Holds whole
    # frames, so it is memory the skeleton spends to keep fixes reversible.
    _history: List[Tuple[str, pd.DataFrame]] = field(default_factory=list, repr=False)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str) -> "AnalysisProject":
        if df is None or df.empty:
            raise ProjectError(f"'{name}' has no rows to analyse.")
        # Duplicate index labels would make row identity ambiguous — the exact
        # failure this project model exists to prevent. Catch it at the door
        # rather than at the first `.loc`.
        if df.index.has_duplicates:
            raise ProjectError(
                f"'{name}' has repeated row labels "
                f"({df.index.duplicated().sum()} of {len(df)}). Row identity in this "
                "project is the index label, so repeated labels leave no way to say "
                "which row a decision refers to."
            )
        return cls(id=uuid.uuid4().hex[:12], name=name, created_at=_now(), df=df)

    # ── row identity: labels, not positions ─────────────────────────────────

    @property
    def row_labels(self) -> List[Any]:
        """The row identities of this project, in frame order.

        These are index *labels*. A consumer that stores positions instead —
        `np.where(mask)[0]` — is storing something that stops meaning the same
        row the moment a filter or a repair changes the frame.
        """
        return [_label(v) for v in self.df.index]

    def rows(self, labels: List[Any]) -> pd.DataFrame:
        """Fetch rows by label. The one supported way to name a subset.

        Uses ``.loc``. Never ``.iloc`` — that is the convention clash in
        `TRANSITION_PLAN.md` §02.2, and it stays fixed here by not being
        available.
        """
        missing = [l for l in labels if l not in self.df.index]
        if missing:
            raise ProjectError(
                f"{len(missing)} row label(s) are not in this table: {missing[:5]}. "
                "Labels are identities — a missing one means the frame changed "
                "underneath, not that the caller should fall back to a position."
            )
        return self.df.loc[labels]

    # ── columns ─────────────────────────────────────────────────────────────

    @property
    def columns(self) -> List[Dict[str, Any]]:
        """Columns by position, with their labels.

        Positional, because `_each_column()` in the engine is positional for a
        reason: with duplicate labels `df[col]` returns a frame, not a series
        (ARCHITECTURE.md §02). `duplicated` is surfaced so the UI can decline to
        offer an ambiguous column as a target.
        """
        labels = list(self.df.columns)
        seen: Dict[Any, int] = {}
        for l in labels:
            seen[l] = seen.get(l, 0) + 1
        out = []
        for pos in range(len(labels)):
            series = self.df.iloc[:, pos]
            out.append({
                "position": pos,
                "name": str(labels[pos]),
                "dtype": str(series.dtype),
                "duplicated": seen[labels[pos]] > 1,
                "n_missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            })
        return out

    # ── state transitions ───────────────────────────────────────────────────

    def record(self, kind: str, text: str, subject: str = "",
               payload: Optional[Dict[str, Any]] = None) -> Decision:
        """Append one decision. The only way the record grows."""
        d = Decision(id=uuid.uuid4().hex[:10], kind=kind, subject=subject,
                     text=text, at=_now(), payload=payload or {})
        self.decisions.append(d)
        return d

    def set_target(self, column: str, task_type: Optional[str],
                   confidence: Optional[str], reasons: List[str]) -> Decision:
        """Choose the target and record having chosen it.

        Marks existing findings stale rather than clearing them: they were
        computed against a different question and are no longer current, but
        throwing them away would delete work the user can still see and reason
        about.
        """
        if column not in list(self.df.columns):
            raise ProjectError(f"No column named '{column}' in this table.")
        changed = self.target is not None and self.target != column
        self.target = column
        self.task_type = task_type
        self.task_confidence = confidence
        self.task_reasons = list(reasons or [])
        self.task_overridden = False
        if changed:
            self.findings_stale = True
        return self.record(
            kind="set_target", subject=column,
            text=f"{column} was chosen as the target; the task was detected as "
                 f"{task_type} at {confidence} confidence.",
            payload={"task_type": task_type, "confidence": confidence,
                     "reasons": self.task_reasons, "replaced": changed},
        )

    def override_task_type(self, task_type: str) -> Decision:
        """Let the user disagree with the detection, and record that they did.

        Required, not optional. `ml/triage.py` returns `low` confidence for a
        low-cardinality integer target and says so in its own words — *"counts
        or ordinal scores should be treated as regression. Verify or override
        below."* An interface that reports that verdict and offers no way to
        contradict it has made the choice itself, at a confidence tier the
        governing rule reserves for the user (`PRODUCT_VISION.md` §07.1).
        Classic has this control; Guided has to as well.

        The detected value is kept alongside the override, because the record
        has to be able to say what the app thought *and* what the user decided.
        """
        if task_type not in ("classification", "regression"):
            raise ProjectError(
                f"'{task_type}' is not a task type. Expected classification or regression.")
        if not self.target:
            raise ProjectError("Choose a target before setting its task type.")
        detected = self.task_type
        self.task_type = task_type
        self.task_overridden = task_type != detected
        if self.task_overridden:
            self.findings_stale = True
        return self.record(
            kind="set_task_type", subject=self.target,
            text=(f"{self.target} was treated as {task_type}, overriding the detected "
                  f"{detected} ({self.task_confidence} confidence)."
                  if self.task_overridden else
                  f"The detected task type for {self.target}, {task_type}, was confirmed."),
            payload={"task_type": task_type, "detected": detected,
                     "overridden": self.task_overridden},
        )

    def fingerprint(self) -> str:
        """A content hash of the working table: values, row labels, dtypes.

        Exists so "declining a preview left the project untouched" is a thing a
        test can assert rather than a thing a comment claims.
        """
        h = hashlib.sha256()
        h.update(self.df.to_csv(index=True).encode("utf-8"))
        h.update("|".join(f"{c}:{d}" for c, d in
                          zip(map(str, self.df.columns), map(str, self.df.dtypes))).encode("utf-8"))
        return h.hexdigest()

    def apply_fix(self, new_df: pd.DataFrame, finding_id: str, title: str,
                  description: str, row_identity_preserved: bool) -> Decision:
        """Install a repaired frame, keeping the one it replaced.

        The previous frame is pushed onto a stack rather than dropped, because
        `ARCHITECTURE.md` §02 requires fixes to be *reversible*, and a fix you
        cannot undo is a fix the user has to be certain about before they can
        see what it did — which is the blind consent the preview exists to
        avoid.

        `row_identity_preserved` is recorded on the decision rather than acted
        on. Four fix kinds renumber the index, and when that happens every row
        label stored earlier stops naming the same row. The skeleton has no
        lockbox to invalidate yet, so the honest thing is to write down that it
        happened; the step that seals a test set will need to read this.
        """
        self._history.append((finding_id, self.df))
        self.df = new_df
        self.findings_stale = True
        return self.record(
            kind="apply", subject=finding_id, text=description,
            payload={"title": title,
                     "row_identity_preserved": bool(row_identity_preserved),
                     "reverts_to": len(self._history) - 1},
        )

    def revert_last_fix(self) -> Decision:
        """Undo the most recent applied fix. Appends; never erases the record."""
        if not self._history:
            raise ProjectError("No applied fix to undo.")
        finding_id, previous = self._history.pop()
        self.df = previous
        self.findings_stale = True
        return self.record(
            kind="revert", subject=finding_id,
            text="That change was undone; the table is back as it was.",
            payload={"undid": finding_id},
        )

    @property
    def applied_fixes(self) -> List[str]:
        return [fid for fid, _ in self._history]

    def set_findings(self, findings: List[Dict[str, Any]],
                     profile: Optional[Dict[str, Any]] = None) -> None:
        """Install a freshly computed finding set. Clears the stale mark."""
        self.findings = list(findings)
        if profile is not None:
            self.profile = profile
        self.findings_stale = False

    def finding(self, finding_id: str) -> Dict[str, Any]:
        for f in self.findings:
            if f.get("id") == finding_id:
                return f
        raise ProjectError(f"No finding with id '{finding_id}' in this project.")

    # ── the wire ────────────────────────────────────────────────────────────

    def to_dict(self, include_rows: bool = False) -> Dict[str, Any]:
        """Serialize. The frame is a handle, not content — it never goes in.

        `row_labels` is included only on request: it is the row identity, and on
        a large table it is also the largest thing here.
        """
        out: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "n_rows": int(len(self.df)),
            "n_columns": int(len(self.df.columns)),
            "row_identity": "index_labels",
            "target": self.target,
            "task_type": self.task_type,
            "task_confidence": self.task_confidence,
            "task_overridden": self.task_overridden,
            "task_reasons": list(self.task_reasons),
            "columns": self.columns,
            "decisions": [d.to_dict() for d in self.decisions],
            "findings": self.findings,
            "findings_stale": self.findings_stale,
            "applied_fixes": self.applied_fixes,
            "fingerprint": self.fingerprint(),
            "profile": self.profile,
        }
        if include_rows:
            out["row_labels"] = self.row_labels
        return out

    def head(self, n: int = 8) -> Dict[str, Any]:
        """A small sample for the interface, carrying its row labels with it.

        The labels travel with the rows so the frontend never has to say "row 3"
        and mean the fourth position.
        """
        sample = self.df.head(n)
        return {
            "columns": [str(c) for c in sample.columns],
            "labels": [_label(v) for v in sample.index],
            "rows": [["" if pd.isna(v) else str(v) for v in row]
                     for row in sample.itertuples(index=False, name=None)],
        }


class ProjectStore:
    """In-memory project registry.

    Deliberately a dict. `ARCHITECTURE.md` §02 records a ``_NEVER_PERSIST``
    contract that is stronger than its documentation, and §04 flags persistence
    as unsolved rather than solved. A skeleton that quietly wrote projects to
    disk would resolve that open question by accident.
    """

    def __init__(self) -> None:
        self._projects: Dict[str, AnalysisProject] = {}

    def add(self, project: AnalysisProject) -> AnalysisProject:
        self._projects[project.id] = project
        return project

    def get(self, project_id: str) -> AnalysisProject:
        if project_id not in self._projects:
            raise ProjectError(f"No project '{project_id}'. Projects live in memory "
                               "only and do not survive a server restart.")
        return self._projects[project_id]

    def __len__(self) -> int:
        return len(self._projects)
