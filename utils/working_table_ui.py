"""The standing statement of what the working table is.

Two surfaces, both reading the same ledger:

`render_working_table_card` sits at the head of the audit, because the audit is
about THIS table and the researcher should not have to scroll up through two
committed join previews to find out what it currently holds.

`render_exit_assurance` closes the page. It restates the table, names anything
about it a researcher would want to have noticed, and asks for a sign-off that
is withdrawn automatically if the table moves afterwards. Leaving Upload &
Audit should feel like signing for a delivery, not like hoping.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from utils import table_ledger as ledger


def _shape_line(df: pd.DataFrame, subj_col: Optional[str]) -> str:
    rows, cols = ledger.shape_of(df)
    line = f"**{rows:,} rows × {cols} columns**"
    n_sub = ledger.count_subjects(df, subj_col)
    if n_sub is not None and n_sub != rows:
        # Rows are not people. A repeated-measures join multiplies rows while
        # subjects stay flat, and reporting only rows invites a study to be
        # written up with the wrong n.
        line += f" — {n_sub:,} distinct `{subj_col}` values across those rows"
    return line


def render_working_table_card(df: Optional[pd.DataFrame],
                              context: str = "") -> None:
    """What you have, how it got that way, and how to take a step back."""
    if df is None or df.empty:
        return
    subj_col = ledger.subject_column(df)
    all_steps = ledger.steps()

    with st.container(border=True):
        st.markdown(f"#### 📋 Your working table\n{_shape_line(df, subj_col)}")
        if context:
            st.caption(context)

        net = ledger.net_change()
        if net and net["n_steps"]:
            bits = []
            if net["rows_before"] != net["rows_after"]:
                bits.append(f"{net['rows_before']:,} → {net['rows_after']:,} rows")
            if net["cols_before"] != net["cols_after"]:
                bits.append(f"{net['cols_before']} → {net['cols_after']} columns")
            if (net["subjects_before"] is not None
                    and net["subjects_after"] is not None
                    and net["subjects_before"] != net["subjects_after"]):
                bits.append(f"{net['subjects_before']:,} → "
                            f"{net['subjects_after']:,} subjects")
            if bits:
                st.caption("Since the first file you added: " + " · ".join(bits))

        lossy = [(s, s.loss_sentence()) for s in ledger.lossy_steps()]
        lossy = [(s, loss) for s, loss in lossy if loss]
        if lossy:
            st.markdown("**What you no longer have**")
            for s, loss in lossy:
                st.markdown(f"- **{loss}** — {s.action}")

        if all_steps:
            with st.expander(f"Every step that shaped this table ({len(all_steps)})",
                             expanded=False):
                _render_lineage(all_steps)


def _render_lineage(all_steps) -> None:
    """The ordered account, with an undo where one is affordable."""
    for i, s in enumerate(all_steps):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"**{i + 1}. {s.action}** — {s.shape_sentence()}")
            cost = s.cost_sentence()
            if cost:
                st.caption(("⚠️ " if s.is_lossy else "") + cost)
            if s.detail:
                st.caption(s.detail)
        with right:
            if s.undoable and i == len(all_steps) - 1:
                if st.button("Undo", key=f"wt_undo_{i}",
                             help="Put the table back as it was before this step"):
                    _undo(i)
            elif s.undoable:
                if st.button("Back to here", key=f"wt_undo_{i}",
                             help=f"Undo this and the {len(all_steps) - i - 1} "
                                  f"step(s) after it"):
                    _undo(i)
    if any(not s.undoable for s in all_steps):
        st.caption(
            "Steps without an undo button cannot be reversed here — either they "
            "happened before the table existed, or the table was too large to "
            "keep a copy of. Re-add the file to start over."
        )


def _undo(index: int) -> None:
    from utils.session_state import set_data
    from utils.state_reconcile import reconcile_state_with_df

    frame = ledger.undo_to(index)
    if frame is None:
        st.warning("That step cannot be undone — no copy of the earlier table was kept.")
        return
    st.session_state["working_table"] = frame
    # A restored table has the columns the undone step removed, so this is a
    # schema change in the direction that matters: anything downstream that was
    # narrowed to fit the smaller table has to be let back out.
    set_data(frame)
    try:
        reconcile_state_with_df(frame, st.session_state)
    except Exception:
        pass
    st.session_state["_working_table_undo_note"] = (
        f"Undone. The table is back to {len(frame):,} rows × {frame.shape[1]} columns."
    )
    st.rerun()


# ── the closing statement ────────────────────────────────────────────────

def _columns_absent_from_a_source(df: pd.DataFrame) -> list:
    """Columns that are wholly missing from at least one contributing file.

    Returns (column, files_it_is_missing_from, files_it_came_from). Only
    meaningful for a stacked table, which is the only place the source of each
    row is recorded.
    """
    try:
        from utils.combine_ui import SOURCE_COLUMN
    except Exception:
        return []
    if SOURCE_COLUMN not in df.columns:
        return []
    out = []
    try:
        sources = [str(s) for s in df[SOURCE_COLUMN].dropna().unique()]
        if len(sources) < 2:
            return []
        for col in df.columns:
            if str(col) == SOURCE_COLUMN or not df[col].isna().any():
                continue
            absent, present = [], []
            for src in sources:
                rows = df.loc[df[SOURCE_COLUMN].astype(str) == src, col]
                (absent if rows.isna().all() else present).append(src)
            if absent and present:
                out.append((str(col), absent, present))
    except Exception:
        return []
    return out


def _surprises(df: pd.DataFrame, subj_col: Optional[str]) -> list:
    """Things true of this table that a researcher would want to have noticed.

    Every one of these is legitimate and may be exactly what was intended. The
    point is that none of them should be discovered later, in a result.
    """
    out = []
    rows, cols = ledger.shape_of(df)
    # count_subjects re-derives the column when given None, so resolving it
    # here keeps the number and the name it is attributed to in agreement —
    # without this the note read "`None` repeats".
    subj_col = subj_col or ledger.subject_column(df)

    n_sub = ledger.count_subjects(df, subj_col)
    if n_sub is not None and n_sub < rows:
        per = rows / max(n_sub, 1)
        # What separates a repeated-measures design from a few stray duplicates
        # is not the average rows per person — 1.2 can mean "every subject has
        # 1.2 visits", which is impossible, or "a fifth of them are duplicated",
        # which is a data problem. It is whether repetition is systematic.
        try:
            counts = df[subj_col].value_counts()
            share_repeating = float((counts > 1).sum()) / max(len(counts), 1)
        except Exception:
            share_repeating = 1.0 if per >= 1.2 else 0.0
        if share_repeating >= 0.5:
            # A genuine repeated-measures design.
            out.append(
                f"**Your n is {n_sub:,}, not {rows:,}.** `{subj_col}` repeats — "
                f"each person contributes about {per:.1f} rows. Anything averaged "
                f"over rows weights people by how many measurements they have, "
                f"and the held-out set is split by person for the same reason."
            )
        else:
            # A handful of repeats. Saying "each person contributes about 1.0
            # rows" is noise, and the duplicate-row line below tells the real
            # story — but the count still matters, because it is what makes the
            # split subject-aware.
            extra = rows - n_sub
            out.append(
                f"**{extra:,} row{'' if extra == 1 else 's'} share a "
                f"`{subj_col}` with another row**, so this table holds {n_sub:,} "
                f"people in {rows:,} rows. The held-out set is split by person, "
                f"not by row, because of it."
            )

    net = ledger.net_change()
    if net and net["subjects_before"] and net["subjects_after"] is not None:
        lost = net["subjects_before"] - net["subjects_after"]
        if lost > 0:
            pct = 100 * lost / max(net["subjects_before"], 1)
            out.append(
                f"**{lost:,} of the {net['subjects_before']:,} people you started "
                f"with ({pct:.0f}%) are not in this table.** If being present in "
                f"every file is related to what you are predicting, the people "
                f"who remain are a selected sample."
            )

    dropped = [c for s in ledger.steps() for c in s.columns_removed]
    if dropped:
        one = len(dropped) == 1
        out.append(
            f"**{len(dropped)} column{'' if one else 's'} "
            f"{'was' if one else 'were'} removed:** "
            + ", ".join(f"`{c}`" for c in dropped[:8])
            + ("…" if len(dropped) > 8 else "")
            + f". {'It is' if one else 'They are'} not available to any later step."
        )

    # Stacking files with different columns is the shape change that hides
    # best. Nothing is empty, nothing is dropped, the row count goes up exactly
    # as promised — and a column measured in only one of the files is now
    # missing for every row from the others. A researcher who models it loses
    # half their sample and finds out, if at all, from a sample size in a
    # results table. This used to pass as "nothing about this table looks
    # surprising".
    for col, absent_from, present_in in _columns_absent_from_a_source(df):
        out.append(
            f"**`{col}` was not measured in {', '.join(absent_from)}.** It is "
            f"missing for every one of those rows — {int(df[col].isna().sum()):,} "
            f"of {rows:,} — and present only in {', '.join(present_in)}. Modeling "
            f"it silently restricts your sample to those files."
        )

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        one = len(empty_cols) == 1
        out.append(
            f"**{len(empty_cols)} column{'' if one else 's'} "
            f"{'is' if one else 'are'} entirely empty** ("
            + ", ".join(f"`{c}`" for c in empty_cols[:6])
            + f"). Stacking files with different columns does this; "
              f"{'it carries' if one else 'they carry'} no information and "
              f"cannot be modeled."
        )

    dupes = int(df.duplicated().sum())
    if dupes:
        out.append(f"**{dupes:,} rows are exact duplicates** of another row.")

    # Two choices made per-file, before this table existed, that change what
    # every number here means. A transpose inverts the table outright; a
    # recode rewrites values in place and leaves the shape untouched, so
    # neither shows up in any before/after count.
    origins = st.session_state.get("_dataset_origin") or {}
    transposed = [i.get("filename") or k for k, i in origins.items()
                  if i.get("transposed")]
    if transposed:
        one = len(transposed) == 1
        out.append(
            f"**{'A file was' if one else f'{len(transposed)} files were'} "
            f"transposed on import** ("
            + ", ".join(f"`{f}`" for f in transposed[:4])
            + f"). {'Its' if one else 'Their'} rows and columns were swapped, so "
              f"what is now a participant was a column in the original file. "
              f"Check the preview above shows one row per participant."
        )

    repairs = [(i.get("filename") or k, fx) for k, i in origins.items()
               for fx in (i.get("repairs") or [])]
    if repairs:
        out.append(
            f"**{len(repairs)} value-level repair(s) were applied at import**, "
            f"before anything else saw the data: "
            + "; ".join(f"`{f}` — {fx}" for f, fx in repairs[:4])
            + ("…" if len(repairs) > 4 else "")
            + ". These changed values, not counts, so they appear in no "
              "before/after number on this page."
        )

    return out


def render_exit_assurance(df: Optional[pd.DataFrame]) -> None:
    """The sign-off. Restate the table, name what is notable, ask for a yes."""
    if df is None or df.empty:
        return
    subj_col = ledger.subject_column(df)
    rows, cols = ledger.shape_of(df)

    st.markdown("---")
    st.markdown("### Is this the data you meant?")
    with st.container(border=True):
        st.markdown(
            f"Everything from here on — the plots, the models, the numbers in "
            f"your manuscript — describes this table and nothing else.\n\n"
            f"{_shape_line(df, subj_col)}"
        )

        notes = _surprises(df, subj_col)
        if notes:
            st.markdown("**Worth knowing before you continue**")
            for n in notes:
                st.markdown(f"- {n}")
        else:
            st.caption("Nothing about this table looks surprising.")

        already = ledger.is_confirmed(df)
        checked = st.checkbox(
            "This is the table I want to analyze.",
            value=already, key="wt_confirm_box",
            help="Recorded with the table's current shape. If the table changes "
                 "afterwards, this is withdrawn and you will be asked again.",
        )
        if checked and not already:
            ledger.confirm(df)
        elif not checked and already:
            ledger.withdraw_confirmation()

        prior = ledger.confirmed_shape()
        if prior is not None and prior != (rows, cols):
            st.warning(
                f"⚠️ You confirmed a table of {prior[0]:,} rows × {prior[1]} "
                f"columns. It is now {rows:,} × {cols}. Check the change above "
                f"and confirm again."
            )
