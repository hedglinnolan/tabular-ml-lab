"""Step 2 of Upload & Audit: turn several clean files into one working table.

The user story this is built for, in the researcher's own words:

    "Can I come to this app with multiple datasets that are independently
     clean, but which I haven't taken the time to combine into one working
     frame?"

The answer has to be yes, and it has to be reachable by someone who has never
written a line of SQL. So this screen makes three promises:

1. **One question, in your language.** Not "inner/left/outer join" but "are
   these the same measurements on different people, or different measurements
   on the same people?" — and the app guesses first, so the default is usually
   right.
2. **See the result before you commit.** The exact row count, which rows get
   dropped, and which subjects repeat, are all shown ABOVE the button. No
   surprises after the fact.
3. **Problems come with a fix, not a stack trace.** An ID stored as text in one
   file and numbers in another is a one-click repair with an explanation, not
   "You are trying to merge on str and int64 columns".

All the reasoning lives in ml/join_doctor.py and utils/combine.py, which are
pure and unit-tested. This module is only presentation and state.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.combine import (
    SOURCE_COLUMN, execute_stack, plan_stack, relationship_hint,
)
from ml.join_doctor import (
    KeyCandidate, diagnose_join, execute_join, find_key_candidates, plain_summary,
)

# How the four join types read to someone who has never used a database.
LINK_MODES = {
    "Keep only people found in every file": "inner",
    "Keep everyone from the first file": "left",
    "Keep everyone from every file": "outer",
}
_MODE_BY_HOW = {v: k for k, v in LINK_MODES.items()}


def _file_cards(frames: Dict[str, pd.DataFrame]) -> None:
    """What the user brought, stated plainly, with a peek at each file."""
    st.markdown("**Your files**")
    for name, df in frames.items():
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"📄 **{name}** — {len(df):,} rows × {df.shape[1]} columns")
        with c2:
            show = st.toggle("Preview", key=f"combine_peek_{name}",
                             label_visibility="collapsed")
        if show:
            st.dataframe(df.head(5), width="stretch")


def _render_link(frames: Dict[str, pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, str]]:
    """Different measurements on the same people — link them by a shared ID."""
    names = list(frames)
    base_name = st.selectbox(
        "Start from which file?", names, key="combine_base_file",
        help="Everything else gets attached to this one. Usually the file with "
             "your main outcome, or the largest.",
    )
    others = [n for n in names if n != base_name]

    mode_label = st.radio(
        "Which people should end up in the combined table?",
        list(LINK_MODES),
        key="combine_link_mode",
        help="If in doubt, the first option is the safest: it keeps only people "
             "who appear in every file, so nothing is missing.",
    )
    how = LINK_MODES[mode_label]

    result = frames[base_name]
    steps: List[str] = []
    blocked = False

    for other in others:
        st.markdown(f"##### Attaching **{other}**")
        cands = find_key_candidates(result, frames[other])
        usable = [c for c in cands if c.confidence != "low"]
        if not usable:
            st.warning(
                f"No shared ID was found between your data so far and **{other}**. "
                f"These files may not describe the same people — or the ID columns "
                f"may hold different things. You can pick the columns yourself below."
            )
            usable = cands[:5]

        if not usable:
            st.error(f"**{other}** has no column that lines up with your data, so it "
                     f"cannot be attached.")
            blocked = True
            continue

        options = [f"{c.left_col} ↔ {c.right_col}" for c in usable]
        chosen_label = st.selectbox(
            f"Which columns identify the same person in both files?",
            options, key=f"combine_key_{other}",
        )
        chosen: KeyCandidate = usable[options.index(chosen_label)]
        st.caption(chosen.headline("your data so far", other))

        diag = diagnose_join(result, frames[other], chosen.left_col, chosen.right_col,
                             how, "your data so far", other)
        for b in diag.blocking:
            st.error(f"🛑 {b}")
        for w in diag.warnings:
            st.warning(f"⚠️ {w}")
        for n in diag.notes:
            st.caption(f"ℹ️ {n}")

        if not diag.can_proceed and not diag.dtype_mismatch:
            blocked = True
            continue

        st.markdown(plain_summary(diag, "your data so far", other))
        try:
            # A type mismatch is repaired as part of the combine, which is why
            # a blocked-but-repairable diagnosis is allowed through here.
            result, desc = execute_join(
                result, frames[other], chosen.left_col, chosen.right_col, how,
                base_name, other,
            )
            steps.append(desc)
        except Exception as exc:
            st.error(f"Could not attach **{other}**: {exc}")
            blocked = True

    if blocked:
        st.info("Fix the issues above, or choose different columns, and this will update.")
        return None
    return result, " ".join(steps)


def _render_stack(frames: Dict[str, pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, str]]:
    """Same measurements on different people — stack them end to end."""
    chosen = st.multiselect(
        "Which files should be stacked?", list(frames), default=list(frames),
        key="combine_stack_files",
    )
    if len(chosen) < 2:
        st.info("Pick at least two files to stack.")
        return None

    subset = {n: frames[n] for n in chosen}
    plan = plan_stack(subset)
    for b in plan.blocking:
        st.error(f"🛑 {b}")
    for w in plan.warnings:
        st.warning(f"⚠️ {w}")
    for n in plan.notes:
        st.caption(f"ℹ️ {n}")
    if not plan.can_proceed:
        return None

    st.markdown(plan.summary())
    if plan.partial_columns:
        with st.expander(f"Which columns are missing from which file "
                         f"({len(plan.partial_columns)})"):
            for col, missing in plan.partial_columns.items():
                st.caption(f"**{col}** — not in: {', '.join(missing)}")
    return execute_stack(subset)


def render_combine_step(frames: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Render Step 2 and return the combined table once the user confirms.

    Returns None until the user presses the button, so the caller only commits
    a working table on an explicit action.
    """
    st.header("Step 2: Combine your files")
    st.caption(
        "You brought several files that haven't been combined yet. That is normal "
        "— this step does it for you, and shows exactly what you'll get before "
        "anything changes."
    )

    _file_cards(frames)
    st.markdown("---")

    hint = relationship_hint(frames)
    choices = [
        "Different measurements on the same people",
        "The same measurements on different people",
        "Just use one file for now",
    ]
    default_idx = {"link": 0, "stack": 1}.get(hint, 0)

    st.markdown("**How do these files relate?**")
    if hint == "stack":
        st.caption("💡 These look like the **same measurements on different people** — "
                   "they share nearly all of their columns.")
    elif hint == "link":
        st.caption("💡 These look like **different measurements on the same people** — "
                   "they have mostly different columns, so they probably link by an ID.")

    relation = st.radio(
        "How do these files relate?", choices, index=default_idx,
        key="combine_relation", label_visibility="collapsed",
    )
    st.caption({
        choices[0]: "For example: demographics + lab results + diet, all for the same "
                    "participants. The columns add up.",
        choices[1]: "For example: survey cycles, study sites, or different years of the "
                    "same measurements. The rows add up.",
        choices[2]: "Continue with a single file; the others stay uploaded for later.",
    }[relation])

    st.markdown("---")

    if relation == choices[2]:
        pick = st.selectbox("Which file?", list(frames), key="combine_single_pick")
        st.caption(f"{len(frames[pick]):,} rows × {frames[pick].shape[1]} columns")
        if st.button("Use this file", type="primary", key="combine_use_single"):
            st.session_state["_combine_description"] = f"Used '{pick}' without combining."
            return frames[pick]
        return None

    outcome = (_render_link(frames) if relation == choices[0]
               else _render_stack(frames))
    if outcome is None:
        return None

    combined, description = outcome
    st.markdown("---")
    if st.button("Combine files", type="primary", key="combine_confirm"):
        st.session_state["_combine_description"] = description
        return combined
    st.caption("Nothing has changed yet — press **Combine files** when the result above "
               "looks right.")
    return None


def render_combined_summary(df: pd.DataFrame) -> None:
    """After combining: what you got, and where it came from."""
    desc = st.session_state.get("_combine_description")
    st.success(f"**Combined table ready** — {len(df):,} rows × {df.shape[1]} columns")
    if desc:
        st.caption(desc)
    if SOURCE_COLUMN in df.columns:
        counts = df[SOURCE_COLUMN].value_counts()
        st.caption("Rows per file: " +
                   ", ".join(f"{k} ({v:,})" for k, v in counts.items()))
