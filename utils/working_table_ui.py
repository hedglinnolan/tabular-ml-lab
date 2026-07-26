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

def _surprises(df: pd.DataFrame, subj_col: Optional[str]) -> list:
    """Things true of this table that a researcher would want to have noticed.

    Every one of these is legitimate and may be exactly what was intended. The
    point is that none of them should be discovered later, in a result.
    """
    out = []
    rows, cols = ledger.shape_of(df)

    n_sub = ledger.count_subjects(df, subj_col)
    if n_sub is not None and n_sub < rows:
        out.append(
            f"**Your n is {n_sub:,}, not {rows:,}.** `{subj_col}` repeats — each "
            f"person contributes about {rows / max(n_sub, 1):.1f} rows. Anything "
            f"averaged over rows weights people by how many measurements they "
            f"have, and the held-out set is split by person for the same reason."
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
