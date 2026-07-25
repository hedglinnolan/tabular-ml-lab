"""Choosing, showing, and comparing cohort runs on screen.

The engine in utils/cohorts.py decides what is analyzable. This decides what a
researcher sees, and the whole design rests on one distinction that is easy to
state and easy to get wrong:

    "Does my model work as well in women as in men?"   -> one model, everyone,
                                                          checked within groups.
    "Is the relationship DIFFERENT in women and men?"  -> two models, two sets
                                                          of people, compared.

Only the second is a cohort run, and the page says so before it offers the
control. Everything else here follows from three rules the UI must never let a
user break by accident:

  * The lockbox is drawn on the whole study, once, before any cohort exists.
  * The target and the feature list are chosen on the whole study and do not
    change between runs. Same question, different people.
  * A run in progress is stated on every page, because a filter you cannot see
    is a filter you will forget.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from utils.cohorts import (
    CohortCell, CohortPlan, active_cohort, clear_cohort, cohort_candidates,
    cohort_filter_broken, cohort_mask, comparison_caveats, completed_runs,
    features_that_lose_variance, plan_cohorts, record_run, runs_remaining,
    start_cohort,
)

_KEEP_EVERYONE = "— keep everyone together (one model for the whole study) —"


def _rarer_outcome(cell: CohortCell, target_col: str) -> str:
    """"88" alone is ambiguous — 88 cases and 88 non-cases mean opposite things."""
    if not cell.class_counts:
        return "—"
    value = min(cell.class_counts, key=lambda k: cell.class_counts[k])
    return f"{cell.class_counts[value]:,} with {target_col} = {value}"


def _cell_table(plan: CohortPlan, target_col: str) -> pd.DataFrame:
    rows = []
    for c in plan.cells:
        rows.append({
            "Group": c.label,
            "Rows": f"{c.n_rows:,}",
            "To train on": f"{c.n_train:,}",
            "Held out": f"{c.n_test:,}",
            "Rarer outcome": _rarer_outcome(c, target_col),
            "Can be analyzed on its own": "yes" if c.viable else f"no — {c.blocked_reason}",
        })
    return pd.DataFrame(rows)


def render_cohort_chooser(df: pd.DataFrame, target_col: str, task_type: str,
                          feature_cols: Sequence[str],
                          group_col: Optional[str] = None) -> None:
    """The optional last step of configuration: who is this run about?

    `df` must be the WHOLE study — the cohorts are carved out of it here, and
    the counts shown are only meaningful against the full population.
    """
    from utils.test_lockbox import train_row_mask

    st.markdown("---")
    st.subheader("Step 5: Analyze one group at a time (optional)")
    st.caption(
        "By default the app builds one model from everyone. That is usually "
        "right. Choose a group here only if your question is about a "
        "*difference between* groups."
    )

    with st.expander("Which of these two questions are you asking?", expanded=False):
        st.markdown(
            "**“Does my model work as well in women as in men?”**  \n"
            "Keep everyone together. One model, checked separately within each "
            "group — the Explainability page does this, and it is the stronger "
            "design because the model still learns from all your data.\n\n"
            "**“Is the relationship between my predictors and the outcome "
            "*different* in women and men?”**  \n"
            "That is a separate analysis in each group, which is what this step "
            "sets up. You run the women, then you run the men, and you report "
            "both.\n\n"
            "Either way, whether the difference between the groups is real is a "
            "question neither approach answers on its own — that needs one model "
            "on everyone with an interaction term, which tests the difference "
            "directly."
        )

    candidates = [c for c in cohort_candidates(df, target_col)]
    if not candidates:
        st.caption(
            "No column in this data looks like a grouping variable "
            "(a handful of categories describing people), so there is nothing "
            "to split by."
        )
        return

    active = active_cohort()
    options = [_KEEP_EVERYONE] + candidates
    default = 0
    if active and active["column"] in candidates:
        default = options.index(active["column"])
    column = st.selectbox(
        "Split the study by", options, index=default, key="cohort_split_column",
        help="Only columns with 2–20 values are offered. A subject ID or a lab "
             "value cannot define a group.",
    )

    if column == _KEEP_EVERYONE:
        if active:
            st.info(
                f"You are currently working in one group only "
                f"(**{active['column']} = {active['label']}**, "
                f"{active['n_rows']:,} of {active['n_total']:,} rows)."
            )
            if st.button("Go back to analyzing everyone", key="cohort_clear_btn"):
                _switch_to(None)
        return

    plan = plan_cohorts(
        df, column, target_col, task_type,
        train_mask=train_row_mask(df.index), group_col=group_col,
    )
    for b in plan.blocking:
        st.error(f"🛑 {b}")
    if not plan.cells:
        return

    st.caption(plan.summary())
    st.dataframe(_cell_table(plan, target_col), width="stretch", hide_index=True)
    for w in plan.warnings:
        st.warning(w)
    if not plan.can_proceed:
        return

    st.markdown(
        "**The held-out test set was drawn before this split**, on the whole "
        "study, so each group brings its own slice of the same one. That is what "
        "lets the runs be compared: a fresh split per group could put this run's "
        "test people in the next run's training data."
    )

    labels = [c.label for c in plan.viable]
    current = active["label"] if active and active["column"] == column else labels[0]
    choice = st.radio(
        "Which group is this run about?", labels,
        index=labels.index(current) if current in labels else 0,
        key=f"cohort_pick_{column}", horizontal=len(labels) <= 4,
        format_func=lambda lb: f"{lb}  (n={_n_for(plan, lb):,})",
    )
    cell = next(c for c in plan.viable if c.label == choice)

    lost = features_that_lose_variance(
        df, cohort_mask(df, column, cell.value), list(feature_cols))
    if lost:
        names = ", ".join(f"`{c}` ({why})" for c, why in lost[:6])
        head = (f"**One of your predictors carries no information inside this "
                f"group**" if len(lost) == 1 else
                f"**{len(lost)} of your predictors carry no information inside "
                f"this group**")
        st.info(
            f"{head} and will be left out of this run: {names}. That is expected "
            f"— filtering to {cell.label} makes `{column}` constant, and anything "
            f"that only varies with it goes too. The same predictors are dropped "
            f"in the other group's run, so the two stay comparable."
        )

    already = active and active["column"] == column and active["label"] == choice
    if already:
        st.success(
            f"✅ This run is **{column} = {choice}** — {active['n_rows']:,} of "
            f"{active['n_total']:,} rows. Every page below works on these people.")
        if st.button("Go back to analyzing everyone", key="cohort_clear_btn2"):
            _switch_to(None)
    else:
        if st.button(f"Analyze {choice} only  ({cell.n_rows:,} rows)",
                     type="primary", key="cohort_start_btn"):
            _switch_to((df, plan, cell, target_col, [c for c, _ in lost]))


def _n_for(plan: CohortPlan, label: str) -> int:
    for c in plan.cells:
        if c.label == label:
            return c.n_rows
    return 0


def _switch_to(payload) -> None:
    """Change who the analysis is about, and throw away what described others.

    Every model, split, pipeline and figure in session state was computed from a
    different set of people. Keeping any of it is how a run ends up reporting
    the previous cohort's numbers under this cohort's heading.
    """
    from utils.session_state import reset_downstream_results
    if payload is None:
        clear_cohort()
    else:
        df, plan, cell, target_col, dropped = payload
        start_cohort(df, plan, cell, target_col, dropped_features=dropped)
    # The restriction has to reach the manuscript, not just the sidebar. A chip
    # inside the running app does not leave with the export, and every exported
    # artifact otherwise reports this group's N as the study's.
    try:
        from utils.workflow_provenance import get_provenance
        get_provenance().record_cohort_restriction()
    except Exception:
        pass
    reset_downstream_results(clear_feature_engineering=True)
    st.rerun()


# ── the run is stated everywhere, because a hidden filter is a lie ───────

def render_cohort_chip() -> None:
    """The sidebar marker. Rendered on every page by the shared sidebar."""
    if cohort_filter_broken():
        st.sidebar.error(
            "⚠️ The rows for this run can no longer be identified, so the app is "
            "showing nothing rather than quietly showing everyone.")
        if st.sidebar.button("Clear the group filter", key="cohort_repair_btn"):
            clear_cohort()
            st.rerun()
        return
    run = active_cohort()
    if run is None:
        return
    st.sidebar.markdown(
        f"""
        <div style="background:#1e293b;border-left:3px solid #38bdf8;
                    padding:0.5rem 0.7rem;border-radius:4px;margin:0.3rem 0 0.6rem 0;">
          <div style="font-size:0.68rem;color:#7dd3fc;letter-spacing:0.06em;
                      text-transform:uppercase;font-weight:700;">
            Run {run['position']} of {run['of']}
          </div>
          <div style="font-size:0.92rem;color:#f1f5f9;font-weight:600;margin-top:0.1rem;">
            {run['column']} = {run['label']}
          </div>
          <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.1rem;">
            {run['n_rows']:,} of {run['n_total']:,} rows
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cohort_note(context: str = "") -> None:
    """In-page one-liner, for pages whose numbers would otherwise look wrong."""
    run = active_cohort()
    if run is None:
        return
    extra = f" {context}" if context else ""
    st.caption(
        f"👥 This run covers **{run['column']} = {run['label']}** only — "
        f"{run['n_rows']:,} of {run['n_total']:,} rows.{extra}"
    )


# ── after the models are fitted: now do the other group ──────────────────

def render_next_cohort(task_type: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """"You ran the women. Now run the men." — and then compare them honestly."""
    run = active_cohort()
    if run is None:
        return

    record_run(metrics)
    # Scoped to THIS grouping variable and this question. The table used to
    # render every banked run: split by sex, run Female and Male, then split by
    # smoker and run 'never', and the header said "split by smoker" above a
    # table listing Female / Male / never — three overlapping row sets whose
    # "Trained on" counts double-count the same people, under a caveat
    # announcing "you fitted this model in 3 groups".
    done = completed_runs(run["column"])
    done_labels = [r.label for r in done]
    remaining = [lb for lb in run.get("order", []) if lb not in done_labels]

    st.markdown("---")
    st.subheader(f"👥 One group at a time — split by `{run['column']}`")

    if len(done) >= 2:
        st.dataframe(_runs_table(done), width="stretch", hide_index=True)
        for c in comparison_caveats(done, task_type):
            st.warning(c)
    else:
        st.caption(
            f"Finished the **{run['label']}** run. There is nothing to compare "
            f"it with until another group has been run.")

    if not remaining:
        st.success(
            f"Every group of `{run['column']}` has been analyzed. Report all "
            f"{len(done_labels)}, not the one that worked.")
        return

    nxt = remaining[0]
    st.markdown(
        f"**Same question, next group.** The target, the predictors and the "
        f"held-out set stay exactly as they are; only the people change. "
        f"Everything fitted — imputers, scalers, models — is refitted on "
        f"{nxt}'s own rows, because carrying a fit across groups is leakage."
    )
    if st.button(f"Now run the same analysis on {nxt}", type="primary",
                 key="cohort_next_btn"):
        _advance_to(run["column"], nxt)


def _runs_table(done: Sequence[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_keys: List[str] = []
    for r in done:
        for k in r.metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    for r in done:
        row = {"Group": r.label, "Trained on": f"{r.n_train:,}",
               "Held out": f"{r.n_test:,}"}
        for k in metric_keys:
            v = r.metrics.get(k)
            row[k] = f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else v)
        rows.append(row)
    return pd.DataFrame(rows)


def _advance_to(column: str, label: str) -> None:
    """Switch to the next cohort without re-deriving the plan from a filtered frame."""
    from utils.session_state import get_data, reset_downstream_results
    from utils.test_lockbox import train_row_mask

    full = get_data(full_study=True)
    dc = st.session_state.get("data_config")
    target_col = getattr(dc, "target_col", None)
    task_type = getattr(dc, "task_type", "classification")
    if full is None or not target_col:
        return
    plan = plan_cohorts(full, column, target_col, task_type,
                        train_mask=train_row_mask(full.index))
    cell = next((c for c in plan.viable if c.label == label), None)
    if cell is None:
        st.error(f"'{label}' can no longer be analyzed on its own.")
        return
    lost = features_that_lose_variance(
        full, cohort_mask(full, column, cell.value),
        list(getattr(dc, "feature_cols", []) or []))
    start_cohort(full, plan, cell, target_col,
                 dropped_features=[c for c, _ in lost])
    try:
        from utils.workflow_provenance import get_provenance
        get_provenance().record_cohort_restriction()
    except Exception:
        pass
    reset_downstream_results(clear_feature_engineering=True)
    st.rerun()
