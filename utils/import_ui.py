"""Show a file's structural problems at upload, with one-click reversible fixes.

`ml/import_doctor.py` knows how to *find* the things that make real research
files unusable — a title row sitting above the header, 999 meaning "refused",
a lab value column typed as text because one cell reads "<0.01". This module
is what the researcher actually sees.

The contract it renders is the app's standing one:

    never silently guess — diagnose visibly, propose reversibly, record it.

So: nothing is auto-applied. Every fix is a button the user presses, every
press appends a plain-language line to a log that can go into a methods
section, and "Undo all fixes" returns the file to exactly how it was read.
Low-confidence findings are shown as questions and are never pre-selected —
in a survey export "none" is a legitimate answer, and recoding it to missing
would destroy real data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ml.import_doctor import ShapeFinding, apply_fix, diagnose, summarize

_SEVERITY_ICON = {"critical": "🛑", "warning": "⚠️", "info": "ℹ️"}

# How sure the app is, said the way a person would say it.
_CONFIDENCE_NOTE = {
    "high": None,                      # obvious enough that it needs no hedge
    "medium": "Check this one against what you know about the file.",
    "low": "This is a question, not a recommendation — only apply it if you "
           "know these values really do mean 'missing'.",
}


def _frame_signature(df: pd.DataFrame) -> str:
    """Identity for the frame as it was read, to reset stale repairs.

    CONTENT is part of the identity. Keying on shape and column names alone
    meant a corrected re-upload with the same dimensions matched the old
    signature, so repaired_frame() handed back version 1 — and version 1 is
    what page 01 committed to the project, the working table, the audit and the
    lockbox. The preview showed the old numbers too, so nothing was visible to
    notice. Same reasoning as the lockbox signature, and the same fallback for
    frames holding unhashable cells.
    """
    try:
        content = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        # The old fallback was shape + dtypes — no values at all, i.e. exactly
        # the identity this function exists to replace, and any frame with an
        # unhashable cell took it. A list column read back from Parquet arrives
        # as numpy.ndarray cells, which are unhashable, so two versions of such
        # a file differing in a numeric column produced the same signature and
        # repaired_frame served version 1. Stringify per column instead: slower,
        # but it is a fallback, and it actually looks at the data.
        try:
            parts = []
            for col in df.columns:
                s = df[col]
                try:
                    parts.append(int(pd.util.hash_pandas_object(s, index=False).sum()))
                except Exception:
                    parts.append(int(pd.util.hash_pandas_object(
                        s.astype(str), index=False).sum()))
            content = "|".join(str(p) for p in parts)
        except Exception:
            content = "?"
    return (f"{df.shape[0]}x{df.shape[1]}:"
            f"{hash(tuple(map(str, df.columns)))}:{content}")


def _state(key_prefix: str) -> Tuple[str, str, str]:
    return (f"_impdoc_frame_{key_prefix}",
            f"_impdoc_log_{key_prefix}",
            f"_impdoc_sig_{key_prefix}")


def applied_fixes(key_prefix: str) -> List[str]:
    """The plain-language description of every fix applied to this file."""
    return list(st.session_state.get(f"_impdoc_log_{key_prefix}", []))


def repaired_frame(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """The frame with this session's fixes applied, or `df` if there are none.

    Safe to call without rendering — the upload page uses it to make sure the
    frame it commits to the project is the one the user saw and repaired, not
    a fresh re-parse of the original file.
    """
    frame_key, _, sig_key = _state(key_prefix)
    stored = st.session_state.get(frame_key)
    if stored is None or st.session_state.get(sig_key) != _frame_signature(df):
        return df
    return stored


def _reset(key_prefix: str, df: pd.DataFrame) -> None:
    frame_key, log_key, sig_key = _state(key_prefix)
    st.session_state.pop(frame_key, None)
    st.session_state.pop(log_key, None)
    st.session_state[sig_key] = _frame_signature(df)


def _render_finding(finding: ShapeFinding, current: pd.DataFrame,
                    key_prefix: str, idx: int) -> Optional[pd.DataFrame]:
    """One problem, its consequence, and its fix. Returns a new frame if applied."""
    icon = _SEVERITY_ICON.get(finding.severity, "•")
    st.markdown(f"{icon} **{finding.title}**")
    st.markdown(finding.detail)
    st.caption(f"Why this matters: {finding.why_it_matters}")

    # A finding that states its own uncertainty wins over the tier default —
    # the tier sentence is about missing-value sentinels and does not describe
    # every low-confidence finding (DRIVE8 finding 27).
    note = getattr(finding, "uncertainty_note", None) or _CONFIDENCE_NOTE.get(finding.confidence)
    if note:
        st.caption(note)

    if finding.fix_kind == "none":
        # A finding the app can describe but must not repair — mixed units in
        # one column, for instance. Guessing there would corrupt the numbers.
        st.caption("There is no safe automatic fix for this one; it needs a "
                   "decision only you can make about the source file.")
        return None

    if st.button(finding.fix_label, key=f"{key_prefix}_fix_{idx}_{finding.id}"):
        try:
            new_df, description = apply_fix(current, finding)
        except Exception as exc:
            st.error(f"That fix could not be applied: {exc}")
            return None
        frame_key, log_key, sig_key = _state(key_prefix)
        st.session_state[frame_key] = new_df
        st.session_state[log_key] = applied_fixes(key_prefix) + [description]
        return new_df
    return None


def render_import_doctor(df: pd.DataFrame, key_prefix: str,
                         subject: str = "file") -> pd.DataFrame:
    """Render the structural review for one frame and return the frame to use.

    `df` is the frame exactly as it was read. The return value is that frame
    with any fixes the user has applied in this session — which is what the
    caller must add to the project.

    `subject` names what is under review in the prose. The uploader reviews a
    *file*; page 01 also reviews the *working table*, which is what every later
    page actually reads and which may never have passed through the uploader at
    all (`DRIVE-067`).
    """
    _, _, sig_key = _state(key_prefix)
    if st.session_state.get(sig_key) != _frame_signature(df):
        # The file was re-read differently (transpose toggled, other sheet
        # chosen), so repairs computed against the old shape no longer apply.
        _reset(key_prefix, df)

    current = repaired_frame(df, key_prefix)
    log = applied_fixes(key_prefix)
    findings = diagnose(current)

    if log:
        st.success(f"**{len(log)} fix{'es' if len(log) != 1 else ''} applied**")
        for line in log:
            st.caption(f"• {line}")
        if st.button("Undo all fixes", key=f"{key_prefix}_undo"):
            _reset(key_prefix, df)
            st.rerun()

    if not findings:
        if not log:
            st.caption(f"✅ Structural review: this {subject} reads as a clean table.")
        return current

    critical = [f for f in findings if f.severity == "critical"]
    rest = [f for f in findings if f.severity != "critical"]

    if critical:
        st.markdown(f"**Structural review — {summarize(findings)}**")
        st.caption(f"These are worth fixing before the {subject} goes into your "
                   "analysis. Nothing changes until you press a button, and "
                   "anything you apply can be undone.")
    else:
        st.caption(f"Structural review — {summarize(findings)}")

    for idx, finding in enumerate(critical):
        updated = _render_finding(finding, current, key_prefix, idx)
        if updated is not None:
            st.rerun()
        st.markdown("")

    if rest:
        with st.expander(f"Also worth a look ({len(rest)})"):
            for idx, finding in enumerate(rest, start=len(critical)):
                updated = _render_finding(finding, current, key_prefix, idx)
                if updated is not None:
                    st.rerun()
                st.markdown("")

    return current


def structural_headline(df: pd.DataFrame) -> Tuple[int, str]:
    """(critical count, one-line summary) — for a file list, without rendering."""
    findings = diagnose(df)
    crit = sum(1 for f in findings if f.severity == "critical")
    return crit, summarize(findings)
