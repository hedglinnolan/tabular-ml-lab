"""What the working table is, and every decision that made it that way.

Upload & Audit is very good at explaining a decision BEFORE you commit to it —
the join preview says what will be kept, what will be dropped, and what it will
do to your n. It has had nothing to say afterwards. Committed steps stayed on
screen still dressed as live controls, so a table that had since lost a column
was described by a headline reading "156 rows x 10 columns" directly above a
line reading "156 rows x 9 columns". The one number a researcher most needs —
that the 60 people they enrolled became 52 in the table — appeared only inside
a preview that scrolls away.

So this module keeps the account. Every action that changes the shape of the
working table records a Step here, in order, with what it cost. The page reads
it back as a standing statement of what you have and how you got it, and, where
the frame was small enough to hold onto, as an undo.

Rows are not people. Where an identifier is present the ledger tracks distinct
subjects alongside rows, because a repeated-measures join multiplies rows while
holding subjects flat, and "156" is then the wrong answer to "how big is my
study".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_LEDGER_KEY = "working_table_ledger"
_CONFIRM_KEY = "working_table_confirmed_shape"
_SNAPSHOT_KEY = "_working_table_snapshots"

# Undo keeps a copy of the previous frame. On an omics file that is expensive,
# so it is offered only where it is affordable and refused honestly elsewhere
# rather than silently dropping steps out of the history.
_MAX_UNDO_CELLS = 4_000_000
_MAX_UNDO_STEPS = 8

# Kinds of change, so the card can rank what is worth saying out loud.
ADD = "add"                 # a file became the working table
COMBINE = "combine"         # a join or a stack
REPAIR = "repair"           # an Import Doctor fix
CLEAN = "clean"             # a suggested action on the audit
RESHAPE = "reshape"         # transpose, header promotion


@dataclass
class Step:
    """One decision, and what it did to the table."""
    action: str
    kind: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    detail: str = ""
    subjects_before: Optional[int] = None
    subjects_after: Optional[int] = None
    columns_removed: List[str] = field(default_factory=list)
    columns_added: List[str] = field(default_factory=list)
    undoable: bool = False

    # -- what it cost, in the words the card uses ------------------------
    @property
    def rows_delta(self) -> int:
        return self.rows_after - self.rows_before

    @property
    def cols_delta(self) -> int:
        return self.cols_after - self.cols_before

    @property
    def subjects_delta(self) -> Optional[int]:
        if self.subjects_before is None or self.subjects_after is None:
            return None
        return self.subjects_after - self.subjects_before

    @property
    def is_lossy(self) -> bool:
        """True when this step removed rows, subjects, or columns."""
        return (self.rows_delta < 0 or self.cols_delta < 0
                or (self.subjects_delta or 0) < 0)

    def cost_sentence(self) -> str:
        """The plain arithmetic, or "" when nothing about the shape moved."""
        parts: List[str] = []
        if self.rows_delta:
            verb = "lost" if self.rows_delta < 0 else "gained"
            parts.append(f"{verb} {abs(self.rows_delta):,} row"
                         f"{'' if abs(self.rows_delta) == 1 else 's'}")
        sd = self.subjects_delta
        if sd:
            verb = "lost" if sd < 0 else "gained"
            parts.append(f"{verb} {abs(sd):,} subject{'' if abs(sd) == 1 else 's'}")
        if self.cols_delta:
            verb = "lost" if self.cols_delta < 0 else "gained"
            named = ", ".join(f"`{c}`" for c in
                              (self.columns_removed if self.cols_delta < 0
                               else self.columns_added)[:4])
            tail = f" ({named})" if named else ""
            parts.append(f"{verb} {abs(self.cols_delta):,} column"
                         f"{'' if abs(self.cols_delta) == 1 else 's'}{tail}")
        return ", ".join(parts)

    def loss_sentence(self) -> str:
        """Only what this step took away.

        A join can lose 8 people and gain 5 columns in the same breath, so the
        card's "what you no longer have" list has to say the first half without
        the second, or it reads as a gain.
        """
        parts: List[str] = []
        sd = self.subjects_delta
        if sd and sd < 0:
            parts.append(f"{abs(sd):,} subject{'' if abs(sd) == 1 else 's'}")
        if self.rows_delta < 0:
            parts.append(f"{abs(self.rows_delta):,} row"
                         f"{'' if abs(self.rows_delta) == 1 else 's'}")
        if self.cols_delta < 0:
            named = ", ".join(f"`{c}`" for c in self.columns_removed[:4])
            tail = f" ({named})" if named else ""
            parts.append(f"{abs(self.cols_delta):,} column"
                         f"{'' if abs(self.cols_delta) == 1 else 's'}{tail}")
        return ", ".join(parts)

    def shape_sentence(self) -> str:
        """An arrow only where something actually moved.

        "412 × 6 → 412 × 6" is a strange way to say a file was opened.
        """
        after = f"{self.rows_after:,} × {self.cols_after}"
        if (self.rows_before, self.cols_before) == (self.rows_after, self.cols_after):
            return after
        return f"{self.rows_before:,} × {self.cols_before} → {after}"


# ── identity ─────────────────────────────────────────────────────────────

def subject_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    """The column that names a person, when one is recognizable.

    Reuses the lockbox's ranking so the ledger, the split, and the chip all
    mean the same thing by "subject" — three different answers on one page
    would be worse than none.
    """
    if df is None or df.empty:
        return None
    try:
        from utils.test_lockbox import rank_grouping_candidates, _id_kind
        ranked = rank_grouping_candidates(df)
        for cand in ranked:
            if cand["kind"] == "subject":
                return cand["column"]
        # a near-unique ID column does not repeat, so ranking skips it
        for col in df.columns:
            if _id_kind(col) == "subject":
                return str(col)
    except Exception:
        pass
    return None


def count_subjects(df: Optional[pd.DataFrame],
                   col: Optional[str] = None) -> Optional[int]:
    """Distinct subjects in `df`, or None when nothing identifies one."""
    if df is None:
        return None
    col = col or subject_column(df)
    if not col or col not in df.columns:
        return None
    try:
        return int(df[col].nunique(dropna=True))
    except Exception:
        return None


def shape_of(df: Optional[pd.DataFrame]) -> Tuple[int, int]:
    if df is None:
        return (0, 0)
    return (int(df.shape[0]), int(df.shape[1]))


# ── the ledger ───────────────────────────────────────────────────────────

def steps() -> List[Step]:
    import streamlit as st
    raw = st.session_state.get(_LEDGER_KEY) or []
    return [s for s in raw if isinstance(s, Step)]


def clear() -> None:
    """Start a new account. Called when the project's files change."""
    import streamlit as st
    st.session_state.pop(_LEDGER_KEY, None)
    st.session_state.pop(_SNAPSHOT_KEY, None)
    st.session_state.pop(_CONFIRM_KEY, None)


def record(action: str, kind: str,
           before: Optional[pd.DataFrame], after: Optional[pd.DataFrame],
           detail: str = "", subject_col: Optional[str] = None) -> Step:
    """Add a step, measuring the cost from the two frames themselves.

    Measuring rather than trusting a caller's numbers is deliberate: the
    caller that reported `rows_before=0, rows_after=0` for an in-place recode
    is exactly how the study N was once overwritten with zero.
    """
    import streamlit as st
    ra, ca = shape_of(after)
    # `before=None` means "this is where the table came from" — the first file
    # adopted, with nothing before it. Recording 0 -> 156 there would put
    # "gained 156 rows" at the head of the account, which reads as a change
    # rather than a starting point.
    origin_step = before is None
    rb, cb = (ra, ca) if origin_step else shape_of(before)
    col = subject_col or subject_column(after if after is not None else before)
    before_cols = set(after.columns) if origin_step else (
        set(before.columns) if before is not None else set())
    after_cols = set(after.columns) if after is not None else set()

    step = Step(
        action=action, kind=kind,
        rows_before=rb, rows_after=ra, cols_before=cb, cols_after=ca,
        detail=detail,
        subjects_before=count_subjects(after if origin_step else before, col),
        subjects_after=count_subjects(after, col),
        columns_removed=sorted(str(c) for c in before_cols - after_cols),
        columns_added=sorted(str(c) for c in after_cols - before_cols),
    )
    step.undoable = _keep_snapshot(before, len(steps()))

    st.session_state[_LEDGER_KEY] = steps() + [step]
    # The table moved, so any confirmation the researcher gave describes a
    # table that no longer exists. Withdraw it rather than let a stale tick
    # sit beside changed numbers.
    _withdraw_confirmation_if_shape_moved((ra, ca))
    return step


def _keep_snapshot(before: Optional[pd.DataFrame], index: int) -> bool:
    import streamlit as st
    if before is None:
        return False
    rows, cols = shape_of(before)
    if rows * max(cols, 1) > _MAX_UNDO_CELLS:
        return False
    snaps: Dict[int, pd.DataFrame] = st.session_state.get(_SNAPSHOT_KEY) or {}
    snaps[index] = before.copy()
    for old in sorted(snaps)[:-_MAX_UNDO_STEPS]:
        snaps.pop(old, None)
    st.session_state[_SNAPSHOT_KEY] = snaps
    return True


def snapshot_for(index: int) -> Optional[pd.DataFrame]:
    """The frame as it stood BEFORE step `index`, when it was cheap to keep."""
    import streamlit as st
    snaps = st.session_state.get(_SNAPSHOT_KEY) or {}
    frame = snaps.get(index)
    return frame.copy() if frame is not None else None


def undo_to(index: int) -> Optional[pd.DataFrame]:
    """Roll the ledger back to just before step `index` and return that frame."""
    import streamlit as st
    frame = snapshot_for(index)
    if frame is None:
        return None
    st.session_state[_LEDGER_KEY] = steps()[:index]
    snaps = st.session_state.get(_SNAPSHOT_KEY) or {}
    st.session_state[_SNAPSHOT_KEY] = {k: v for k, v in snaps.items() if k < index}
    st.session_state.pop(_CONFIRM_KEY, None)
    return frame


# ── the account, summarized ──────────────────────────────────────────────

def origin() -> Optional[Step]:
    return steps()[0] if steps() else None


def lossy_steps() -> List[Step]:
    return [s for s in steps() if s.is_lossy]


def net_change() -> Optional[Dict[str, Any]]:
    """From the first thing brought in to the table as it stands now."""
    all_steps = steps()
    if not all_steps:
        return None
    first, last = all_steps[0], all_steps[-1]
    return {
        "rows_before": first.rows_before, "rows_after": last.rows_after,
        "cols_before": first.cols_before, "cols_after": last.cols_after,
        "subjects_before": first.subjects_before,
        "subjects_after": last.subjects_after,
        "n_steps": len(all_steps),
    }


# ── the researcher's sign-off ────────────────────────────────────────────

def confirm(df: Optional[pd.DataFrame]) -> None:
    import streamlit as st
    st.session_state[_CONFIRM_KEY] = shape_of(df)


def withdraw_confirmation() -> None:
    import streamlit as st
    st.session_state.pop(_CONFIRM_KEY, None)


def confirmed_shape() -> Optional[Tuple[int, int]]:
    import streamlit as st
    val = st.session_state.get(_CONFIRM_KEY)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    return None


def is_confirmed(df: Optional[pd.DataFrame]) -> bool:
    """True only when the table still has the shape that was signed off."""
    want = confirmed_shape()
    return want is not None and want == shape_of(df)


def _withdraw_confirmation_if_shape_moved(new_shape: Tuple[int, int]) -> None:
    want = confirmed_shape()
    if want is not None and want != new_shape:
        withdraw_confirmation()


# ── persistence ──────────────────────────────────────────────────────────

def to_list() -> List[Dict[str, Any]]:
    """Serializable ledger. Snapshots are NOT saved — undo is session-local."""
    out = []
    for s in steps():
        d = asdict(s)
        d["undoable"] = False
        out.append(d)
    return out


def from_list(data: Any) -> None:
    import streamlit as st
    restored: List[Step] = []
    for d in data or []:
        if not isinstance(d, dict):
            continue
        try:
            restored.append(Step(**{k: v for k, v in d.items()
                                    if k in Step.__dataclass_fields__}))
        except Exception:
            continue
    if restored:
        st.session_state[_LEDGER_KEY] = restored
