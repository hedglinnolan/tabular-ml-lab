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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.combine import (
    SOURCE_COLUMN, execute_stack, plan_combination, plan_stack, relationship_hint,
    set_reserved_columns,
)
from ml.join_doctor import (
    KeyCandidate, _slug, diagnose_join, execute_join, find_key_candidates,
    plain_summary,
)

# How the four join types read to someone who has never used a database.
LINK_MODES = {
    "Keep only people found in every file": "inner",
    "Keep everyone from the first file": "left",
    "Keep everyone from every file": "outer",
}
_MODE_BY_HOW = {v: k for k, v in LINK_MODES.items()}


from html import escape as _esc

from utils.combine_preview import (
    FROM_ADDED, FROM_KEY, FROM_LEFT, FROM_RIGHT, FROM_SHARED,
    ChangeMap, blank_cell_mask, describe_join, describe_stack,
)

# One color per origin, used identically in the column map and the preview so
# the two read as one picture.
_ORIGIN_COLOR = {
    FROM_KEY: "#667eea",
    FROM_LEFT: "#0ea5e9",
    FROM_RIGHT: "#22c55e",
    FROM_SHARED: "#667eea",
    FROM_ADDED: "#94a3b8",
}
_OUTCOME_COLOR = {"kept": "#22c55e", "blanked": "#f59e0b", "dropped": "#cbd5e1"}
_MAX_PREVIEW_COLS = 12

_JOIN_KEY_REASON = ("the ID these files were merged on — it identifies a row, "
                    "it does not explain the outcome")


def _row_ledger(cm: ChangeMap) -> None:
    """Every row accounted for, as one proportional bar.

    A researcher's first question is not "how many rows" but "did I lose
    anybody" — and a bar answers that before any number is read.
    """
    total = sum(g.n for g in cm.row_groups) or 1
    segments = "".join(
        f'<div title="{_esc(g.label)}" style="width:{100 * g.n / total:.2f}%;'
        f'background:{_OUTCOME_COLOR.get(g.outcome, "#cbd5e1")};"></div>'
        for g in cm.row_groups if g.n
    )
    st.markdown(
        f'<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;'
        f'margin:0.35rem 0 0.6rem 0;">{segments}</div>', unsafe_allow_html=True)
    for g in cm.row_groups:
        if not g.n:
            continue
        verb = {"kept": "kept", "blanked": "kept, partly blank", "dropped": "dropped"}[g.outcome]
        st.markdown(
            f'<div style="font-size:0.86rem;margin:0.1rem 0;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
            f'background:{_OUTCOME_COLOR[g.outcome]};margin-right:0.5rem;"></span>'
            f'<strong>{g.n:,}</strong> {_esc(g.label)} — <em>{verb}</em>'
            f'<span style="color:#64748b;"> · {_esc(g.detail)}</span></div>',
            unsafe_allow_html=True)


def _preview_table(result: pd.DataFrame, cm: ChangeMap) -> None:
    """The result itself, with the seam the combine created marked on it.

    For a link the seam runs between columns — everything left of it came from
    one file, everything right of it from the other. For a stack it runs
    between rows. Showing the rows either side of that line is the difference
    between "450 rows" and understanding what happened.
    """
    origin = {c.name: c.origin for c in cm.columns}
    cols = [c for c in result.columns]
    trimmed = False
    if len(cols) > _MAX_PREVIEW_COLS:
        # Keep the seam visible: some from each side rather than the first N.
        left_cols = [c for c in cols if origin.get(str(c)) in (FROM_KEY, FROM_LEFT, FROM_SHARED)]
        right_cols = [c for c in cols if origin.get(str(c)) in (FROM_RIGHT, FROM_ADDED)]
        half = _MAX_PREVIEW_COLS // 2
        cols = left_cols[:half] + right_cols[:_MAX_PREVIEW_COLS - half]
        trimmed = True

    if cm.operation == "stack":
        # Show the last rows of the first file and the first rows of the next,
        # so the join between them is literally on screen.
        counts = [g.n for g in cm.row_groups]
        seam = counts[0] if counts else 0
        idx = list(range(max(0, seam - 2), min(len(result), seam + 2)))
        seam_after = 1 if seam >= 2 else 0
    else:
        idx = list(range(min(5, len(result))))
        seam_after = -1

    head = result.iloc[idx][cols] if idx else result.iloc[:0][cols]
    # Two different blanks that look identical in a table and mean opposite
    # things: "we never measured this person" versus "this person was not in
    # that file". Only the second is caused by the combine.
    merge_blanks = blank_cell_mask(result, cm)

    def _cell(v: Any, from_merge: bool) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
            if from_merge:
                return ('<span style="color:#b45309;background:#fef3c7;padding:0 4px;'
                        'border-radius:3px;font-size:0.78rem;">no match</span>')
            return '<span style="color:#94a3b8;">—</span>'
        text = str(v)
        return _esc(text if len(text) <= 22 else text[:21] + "…")

    ths = "".join(
        f'<th style="padding:5px 8px;text-align:left;font-size:0.78rem;'
        f'border-bottom:2px solid {_ORIGIN_COLOR.get(origin.get(str(c)), "#cbd5e1")};'
        f'white-space:nowrap;">{_esc(str(c))}</th>' for c in cols)
    trs = []
    for n, (_, row) in enumerate(head.iterrows()):
        border = ("border-top:3px solid #667eea;" if n == seam_after + 1 and seam_after >= 0
                  else "")
        tds = "".join(
            f'<td style="padding:4px 8px;font-size:0.82rem;{border}'
            f'white-space:nowrap;">'
            f'{_cell(row[c], bool(merge_blanks.at[row.name, c]) if c in merge_blanks.columns else False)}'
            f"</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    st.markdown(
        f'<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px;'
        f'padding:0.25rem 0.5rem;"><table style="border-collapse:collapse;width:100%;">'
        f"<thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>",
        unsafe_allow_html=True)

    legend = []
    for label, key in (("the ID", FROM_KEY), (cm.before[0][0], FROM_LEFT),
                       (cm.before[-1][0] if cm.operation == "link" else "in every file",
                        FROM_RIGHT if cm.operation == "link" else FROM_SHARED)):
        legend.append(f'<span style="border-bottom:2px solid {_ORIGIN_COLOR[key]};">'
                      f'{_esc(str(label))}</span>')
    note = " · ".join(legend)
    if bool(merge_blanks.to_numpy().any()):
        note += ('  ·  <span style="color:#b45309;background:#fef3c7;padding:0 4px;'
                 'border-radius:3px;">no match</span> = blank because of this '
                 'combine, not a missing measurement')
    if cm.operation == "stack" and seam_after >= 0:
        note += ' · <span style="color:#667eea;">the line is where the second file begins</span>'
    if trimmed:
        note += f" · showing {len(cols)} of {result.shape[1]} columns"
    st.caption(f"Column colors: {note}", unsafe_allow_html=True)


def _column_map(cm: ChangeMap) -> None:
    """Where every column in the result came from."""
    renamed = cm.renamed_columns
    with st.expander(f"Where your {cm.after_cols} columns come from"
                     + (f" — {len(renamed)} renamed to avoid a clash" if renamed else "")):
        for c in cm.columns:
            chip = (f'<span style="display:inline-block;width:8px;height:8px;'
                    f'border-radius:2px;background:{_ORIGIN_COLOR.get(c.origin, "#cbd5e1")};'
                    f'margin-right:0.5rem;"></span>')
            note = (f' <span style="color:#b45309;">(renamed from '
                    f'<code>{_esc(c.renamed_from)}</code> — both files had one)</span>'
                    if c.was_renamed else "")
            st.markdown(f'{chip}<code>{_esc(c.name)}</code> '
                        f'<span style="color:#64748b;font-size:0.85rem;">'
                        f'{_esc(c.source_file)}</span>{note}', unsafe_allow_html=True)


def render_change_map(cm: ChangeMap, result: Optional[pd.DataFrame] = None) -> None:
    """Before → after, in the three registers a researcher needs.

    WHO happens to the rows, WHERE the columns go, and SO WHAT it means for
    the study. The last one is the reason this exists: the app could always
    predict the row count, and a row count does not tell anybody whether their
    cohort just became a selected subsample.
    """
    st.markdown(f"##### {cm.headline()}")
    _row_ledger(cm)
    if result is not None and len(result.columns):
        _preview_table(result, cm)
    _column_map(cm)
    for line in cm.consequences:
        st.warning(line)


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


def _render_link(frames: Dict[str, pd.DataFrame],
                 running_label: Optional[str] = None) -> Optional[Tuple[pd.DataFrame, str]]:
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
    # What to call the frame as it accumulates. "your data so far" was a
    # placeholder that leaked into the headline as though it were a filename,
    # so a user who had just named their group "demographics" was shown
    # "your data so far (600 rows) + labs (480 rows)".
    running = running_label or base_name
    steps: List[str] = []
    # The columns this link actually joined on. They identify rows, so the
    # feature picker must not offer them — but only once the user commits, so
    # the key of an abandoned preview never bars a real predictor.
    join_keys: List[str] = []
    blocked = False

    for other in others:
        st.markdown(f"##### Attaching **{other}**")
        cands = find_key_candidates(result, frames[other])
        usable = [c for c in cands if c.confidence != "low"]
        if not usable:
            st.warning(
                f"No shared ID was found between **{running}** and **{other}**. "
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
        st.caption(chosen.headline(running, other))

        diag = diagnose_join(result, frames[other], chosen.left_col, chosen.right_col,
                             how, running, other)
        for b in diag.blocking:
            st.error(f"🛑 {b}")
        # Every warning, including the row-counting ones. Suppressing "have no
        # match" and "have no ID at all" as redundant with the change map was
        # wrong twice over: the map states row COUNTS, not which columns failed
        # to line up, and those two warnings are the only place a junk key or a
        # deleted stratum announces itself in words. Saying it twice is a
        # smaller cost than a wrongly merged table nobody was told about.
        for w in diag.warnings:
            st.warning(f"⚠️ {w}")
        for n in diag.notes:
            st.caption(f"ℹ️ {n}")

        if not diag.can_proceed and not diag.dtype_mismatch:
            blocked = True
            continue

        try:
            # A type mismatch is repaired as part of the combine, which is why
            # a blocked-but-repairable diagnosis is allowed through here. The
            # join runs now so the preview below is the REAL result, not a
            # description of one — nothing is committed until the button.
            change = describe_join(result, frames[other], chosen.left_col,
                                   chosen.right_col, how, running, other)
            preview, desc = execute_join(
                result, frames[other], chosen.left_col, chosen.right_col, how,
                base_name, other,
            )
            render_change_map(change, preview)
            result = preview
            steps.append(desc)
            for k in (chosen.left_col, chosen.right_col):
                if k in preview.columns and k not in join_keys:
                    join_keys.append(str(k))
        except Exception as exc:
            st.error(f"Could not attach **{other}**: {exc}")
            blocked = True

    if blocked:
        st.info("Fix the issues above, or choose different columns, and this will update.")
        return None
    st.session_state["_combine_join_keys"] = join_keys
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

    stacked, desc = execute_stack(subset)
    render_change_map(describe_stack(subset), stacked)
    return stacked, desc


def _render_grouped(frames: Dict[str, pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, str]]:
    """Several kinds of measurement, each split across files.

    Two cycles of demographics and two of labs need stacking WITHIN a kind and
    linking ACROSS kinds. Asked to pick one operation for all four files, the
    researcher gets 400 half-empty rows or a join proposed on 'age' — both
    wrong, both plausible enough to keep working with.
    """
    plan = plan_combination(frames)
    labels = [g.label for g in plan.groups]

    st.markdown("**What each file holds**")
    st.caption("Files holding the same measurements get stacked together first, "
               "then the results are linked by a shared ID.")

    # The grouping is a proposal. Anything the app inferred, the user can move.
    assignment: Dict[str, str] = {}
    default_for = {m: g.label for g in plan.groups for m in g.members}
    for name in frames:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"📄 {name}")
        with c2:
            choice = st.selectbox(
                f"group for {name}", labels,
                index=labels.index(default_for.get(name, labels[0])),
                key=f"combine_group_{name}", label_visibility="collapsed",
            )
        assignment[name] = choice

    grouped: Dict[str, List[str]] = {}
    for name, label in assignment.items():
        grouped.setdefault(label, []).append(name)

    if len(grouped) < 2:
        st.warning("Everything is in one group, so there is nothing to link. "
                   "Choose **The same measurements on different people** above "
                   "to just stack them.")
        return None

    # ── stack within each group ──────────────────────────────────────────
    stacked: Dict[str, pd.DataFrame] = {}
    steps: List[str] = []
    for label, members in grouped.items():
        if len(members) == 1:
            stacked[label] = frames[members[0]]
            continue
        subset = {m: frames[m] for m in members}
        plan_s = plan_stack(subset)
        st.markdown(f"##### Stacking into **{label}**")
        for b in plan_s.blocking:
            st.error(f"🛑 {b}")
        for w in plan_s.warnings:
            st.warning(f"⚠️ {w}")
        if not plan_s.can_proceed:
            return None
        st.caption(plan_s.summary())
        group_frame, desc = execute_stack(subset)
        # Name the bookkeeping column after the group. Two stacked groups both
        # carrying "__source_file" collide when linked, and the app then warned
        # the user about a name clash in a column the APP invented, telling them
        # to "check which one you actually want".
        if SOURCE_COLUMN in group_frame.columns:
            group_frame = group_frame.rename(
                columns={SOURCE_COLUMN: f"{SOURCE_COLUMN}_{_slug(label)}"})
        stacked[label] = group_frame
        steps.append(desc)

    # ── then link the stacked results ────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**Linking {len(stacked)} combined tables**")
    outcome = _render_link(stacked, running_label=list(stacked)[0])
    if outcome is None:
        return None
    result, link_desc = outcome
    return result, " ".join(steps + [link_desc])


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
    # Decided afresh on every render: a key left over from a different relation
    # choice must not end up registered against this table.
    st.session_state["_combine_join_keys"] = []

    # The app groups the files by how much their columns overlap and proposes
    # a shape. Its guess picks the default answer; the user always decides.
    plan = plan_combination(frames)
    choices = [
        "Different measurements on the same people",
        "The same measurements on different people",
        "Both — several kinds of measurement, each split across files",
        "Just use one file for now",
    ]
    default_idx = {"link": 0, "stack": 1, "stack_then_link": 2}.get(plan.shape, 0)

    st.markdown("**How do these files relate?**")
    if plan.shape == "stack":
        st.caption("💡 These look like the **same measurements on different people** — "
                   "they share nearly all of their columns.")
    elif plan.shape == "link":
        st.caption("💡 These look like **different measurements on the same people** — "
                   "they have mostly different columns, so they probably link by an ID.")
    elif plan.shape == "stack_then_link":
        st.caption(f"💡 {plan.describe()}")

    relation = st.radio(
        "How do these files relate?", choices, index=default_idx,
        key="combine_relation", label_visibility="collapsed",
    )
    st.caption({
        choices[0]: "For example: demographics + lab results + diet, all for the same "
                    "participants. The columns add up.",
        choices[1]: "For example: survey cycles, study sites, or different years of the "
                    "same measurements. The rows add up.",
        choices[2]: "For example: two survey cycles of demographics AND two of labs. "
                    "Each kind is stacked first, then the kinds are linked by ID.",
        choices[3]: "Continue with a single file; the others stay uploaded for later.",
    }[relation])

    st.markdown("---")

    if relation == choices[3]:
        pick = st.selectbox("Which file?", list(frames), key="combine_single_pick")
        st.caption(f"{len(frames[pick]):,} rows × {frames[pick].shape[1]} columns")
        if st.button("Use this file", type="primary", key="combine_use_single"):
            st.session_state["_combine_description"] = f"Used '{pick}' without combining."
            set_reserved_columns([], _JOIN_KEY_REASON, role="join_key")
            return frames[pick]
        return None

    if relation == choices[0]:
        outcome = _render_link(frames)
    elif relation == choices[1]:
        outcome = _render_stack(frames)
    else:
        outcome = _render_grouped(frames)
    if outcome is None:
        return None

    combined, description = outcome
    st.markdown("---")
    if st.button("Combine files", type="primary", key="combine_confirm"):
        st.session_state["_combine_description"] = description
        # Registered at the moment the merge is COMMITTED, which is the only
        # moment the app knows which column identifies a row in this table.
        set_reserved_columns(st.session_state.get("_combine_join_keys", []),
                             _JOIN_KEY_REASON, role="join_key")
        return combined
    st.caption("Nothing has changed yet — press **Combine files** when the result above "
               "looks right.")
    return None


def render_combined_summary(df: pd.DataFrame) -> None:
    """Where the combined rows came from.

    The shape and the description are deliberately NOT repeated here. The
    working-table card states the shape once, so there is a single place on the
    page that answers "what have I got" — two places invited exactly the drift
    that once left a stale "× 10 columns" sitting above a live "× 9 columns".
    What only this function knows is the per-file breakdown, so that is what it
    says.
    """
    if SOURCE_COLUMN in df.columns:
        counts = df[SOURCE_COLUMN].value_counts()
        st.caption("Rows per file: " +
                   ", ".join(f"{k} ({v:,})" for k, v in counts.items()))
