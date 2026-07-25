"""Analyzing one cohort at a time — same question, different people.

Researchers ask for this constantly: "model the men, then model the women."
The app already had a button labeled subgroup analysis, but it answers a
DIFFERENT question, and the two get conflated with real consequences:

    Does my model work equally well for men and women?
        -> train on everyone, evaluate the one model within each group.
           A fairness / generalization check. This is what page 07 does.

    Is the relationship between predictors and outcome DIFFERENT in men
    and women?
        -> fit separately in each group and compare. An effect-modification
           question. This is what people are actually asking for, and it is
           what a cohort run is.

A run is deliberately narrow: **same question, different people.** The target
and the feature list are fixed across runs; only the rows change. That makes
runs comparable by construction, makes the manuscript sentence writable
("in men (n=1,204) … in women (n=1,388) …"), and removes the whole class of
bugs where two "runs" quietly answered different questions.

Three invariants keep it honest, and each is enforced here rather than left to
the UI:

1. THE LOCKBOX IS DRAWN BEFORE THE FILTER. Every cohort inherits its slice of
   one split. Draw a fresh split per cohort and run 2's test people may have
   been run 1's training people, and the runs cannot be compared at all.
2. DECISIONS REPLAY, FITS DO NOT. A run reuses the recorded CHOICES — target,
   features, transform kinds, model, hyperparameters — and refits every
   fitted object on its own rows. Carrying a scaler or an imputer across
   cohorts is leakage that produces a beautiful, irreproducible result.
3. A CELL TOO SMALL TO MODEL IS REFUSED, NOT WARNED. 80 women with 6 events
   cannot support a model, and a number produced from them is worse than no
   number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# A cohort needs this many rows before a model fitted in it means anything.
MIN_ROWS_PER_COHORT = 60
# Classification also needs this many of the rarer class, in train AND test.
MIN_PER_CLASS = 10
# Above this many cells, the multiplicity problem outweighs the finding.
MAX_SENSIBLE_CELLS = 6
# A feature this uniform inside a cohort carries no signal there.
NEAR_CONSTANT_SHARE = 0.99


@dataclass
class CohortCell:
    """One level of the grouping variable, and whether it can be modeled."""
    label: str
    value: Any
    n_rows: int
    n_subjects: int
    n_train: int
    n_test: int
    viable: bool = True
    blocked_reason: str = ""
    # Rows in this cohort including those with no outcome. n_rows counts only
    # the rows that can actually be modeled.
    n_rows_total: int = 0
    class_counts: Dict[Any, int] = field(default_factory=dict)

    @property
    def n_events(self) -> Optional[int]:
        """Size of the rarer class, or None for regression."""
        if not self.class_counts:
            return None
        return int(min(self.class_counts.values()))


@dataclass
class CohortPlan:
    """Splitting the study by one variable, before anything is run."""
    column: str
    cells: List[CohortCell] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)
    # Rows that belong to NO cohort because the grouping value is missing.
    # They vanish from every run, so they have to be counted and said out loud.
    n_excluded_missing: int = 0

    @property
    def viable(self) -> List[CohortCell]:
        return [c for c in self.cells if c.viable]

    @property
    def blocked(self) -> List[CohortCell]:
        return [c for c in self.cells if not c.viable]

    @property
    def can_proceed(self) -> bool:
        return not self.blocking and len(self.viable) >= 2

    def summary(self) -> str:
        if not self.cells:
            return f"'{self.column}' has no usable values."
        parts = ", ".join(f"{c.label} ({c.n_rows:,})" for c in self.viable)
        n = len(self.viable)
        out = (f"Analyzing separately by **{self.column}**: "
               f"{n} group{'' if n == 1 else 's'} — {parts}.")
        if self.blocked:
            k = len(self.blocked)
            out += (f" {k} {'group is' if k == 1 else 'groups are'} too small "
                    f"to model, and {'is' if k == 1 else 'they are'} not offered.")
        if self.n_excluded_missing:
            m = self.n_excluded_missing
            out += (f" {m:,} row{'' if m == 1 else 's'} "
                    f"{'has' if m == 1 else 'have'} no '{self.column}' recorded "
                    f"and {'is' if m == 1 else 'are'} in none of them.")
        return out


def cohort_mask(df: pd.DataFrame, column: str, value: Any) -> pd.Series:
    """Rows belonging to one cohort. NaN never belongs to any cohort."""
    col = df[column]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return col.isna()
    return col.eq(value) & col.notna()


def plan_cohorts(df: pd.DataFrame, column: str, target_col: str,
                 task_type: str = "classification",
                 train_mask: Optional[pd.Series] = None,
                 group_col: Optional[str] = None) -> CohortPlan:
    """Enumerate the cohorts this column would create, and vet each one.

    `train_mask` is True for rows OUTSIDE the held-out lockbox. Passing it is
    what lets a cell be judged on the rows it would actually train on, rather
    than on its total size — a cohort can look ample and still have almost
    nothing left once the test set is removed.
    """
    plan = CohortPlan(column=column)
    if column not in df.columns:
        plan.blocking.append(f"'{column}' is not a column in your data.")
        return plan
    if column == target_col:
        plan.blocking.append(
            f"'{column}' is what you are predicting. Splitting the study by the "
            f"outcome would leave each group with only one answer in it.")
        return plan

    levels = df[column].dropna().unique()
    if len(levels) < 2:
        plan.blocking.append(
            f"'{column}' has the same value for everyone, so there is nothing "
            f"to split.")
        return plan
    if len(levels) > 20:
        plan.blocking.append(
            f"'{column}' has {len(levels):,} different values. Splitting the "
            f"study that many ways leaves almost nobody in each group — pick a "
            f"variable with a handful of categories.")
        return plan

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)
    elif not train_mask.index.equals(df.index):
        # A mask built against a different frame would align by index and
        # silently produce nonsense counts.
        train_mask = train_mask.reindex(df.index, fill_value=True)

    plan.n_excluded_missing = int(df[column].isna().sum())

    for value in sorted(levels, key=lambda v: str(v)):
        mask = cohort_mask(df, column, value)
        rows = df.loc[mask]
        in_train = mask & train_mask
        in_test = mask & ~train_mask
        n_subjects = (int(rows[group_col].nunique())
                      if group_col and group_col in df.columns else int(mask.sum()))
        # A row with no outcome cannot be trained on or scored, so counting it
        # toward viability overstates what the cohort can support: 100 rows of
        # which 65 have no outcome is a 35-row cohort wearing a big number.
        has_y = (df[target_col].notna() if target_col in df.columns
                 else pd.Series(True, index=df.index))
        usable = mask & has_y
        cell = CohortCell(
            label=str(value), value=value, n_rows=int(usable.sum()),
            n_subjects=n_subjects, n_train=int((in_train & has_y).sum()),
            n_test=int((in_test & has_y).sum()),
            n_rows_total=int(mask.sum()),
        )

        if task_type == "classification" and target_col in df.columns:
            counts = rows[target_col].value_counts().to_dict()
            cell.class_counts = {k: int(v) for k, v in counts.items()}

        reasons: List[str] = []
        if cell.n_rows < MIN_ROWS_PER_COHORT:
            shortfall = (f"only {cell.n_rows} rows with an outcome recorded"
                         if cell.n_rows_total > cell.n_rows
                         else f"only {cell.n_rows} rows")
            reasons.append(f"{shortfall} (needs {MIN_ROWS_PER_COHORT})")
        elif cell.n_train < MIN_ROWS_PER_COHORT // 2:
            reasons.append(f"only {cell.n_train} rows left to train on once the "
                           f"held-out set is set aside")
        if task_type == "classification":
            if len(cell.class_counts) < 2:
                reasons.append("everyone in it has the same outcome")
            elif cell.n_events is not None and cell.n_events < MIN_PER_CLASS:
                reasons.append(f"only {cell.n_events} in the smaller outcome group "
                               f"(needs {MIN_PER_CLASS})")
        if reasons:
            cell.viable = False
            cell.blocked_reason = "; ".join(reasons)
        plan.cells.append(cell)

    plan.warnings = _cohort_warnings(plan, task_type)
    if len(plan.viable) < 2:
        plan.blocking.append(
            f"Only one group in '{column}' is big enough to model on its own, "
            f"so there is nothing to compare it with. Keep everyone together and "
            f"add '{column}' as a predictor instead.")
    return plan


def _cohort_warnings(plan: CohortPlan, task_type: str) -> List[str]:
    """What splitting the study this way costs, said before it is done."""
    out: List[str] = []
    viable = plan.viable

    if len(viable) > MAX_SENSIBLE_CELLS:
        out.append(
            f"**{len(viable)} separate analyses is a lot.** Every extra group is "
            f"another chance for one of them to look significant by accident. If "
            f"you report the one that worked, that is the multiple-comparisons "
            f"problem, and it needs disclosing.")

    if viable:
        smallest = min(viable, key=lambda c: c.n_rows)
        largest = max(viable, key=lambda c: c.n_rows)
        if largest.n_rows >= 3 * max(smallest.n_rows, 1):
            out.append(
                f"**The groups are very different sizes** — {largest.label} has "
                f"{largest.n_rows:,} rows and {smallest.label} has "
                f"{smallest.n_rows:,}. The smaller model will look worse partly "
                f"because it had less to learn from, which is not the same as the "
                f"relationship being weaker in that group.")

    if task_type == "classification":
        rates = {}
        for c in viable:
            if c.class_counts and c.n_rows:
                rates[c.label] = min(c.class_counts.values()) / c.n_rows
        if len(rates) >= 2 and (max(rates.values()) - min(rates.values())) > 0.1:
            hi = max(rates, key=rates.get)
            lo = min(rates, key=rates.get)
            out.append(
                f"**The outcome is much more common in one group** ({hi}: "
                f"{rates[hi]:.0%}, {lo}: {rates[lo]:.0%}). Accuracy and AUC are "
                f"not directly comparable across groups with different rates — a "
                f"difference between them can be case mix rather than model "
                f"quality.")

    total = sum(c.n_rows_total for c in plan.cells) + plan.n_excluded_missing
    if plan.n_excluded_missing and total:
        share = plan.n_excluded_missing / total
        if share >= 0.02:
            out.append(
                f"**{plan.n_excluded_missing:,} row(s) ({share:.0%}) have no "
                f"'{plan.column}' recorded**, so they appear in none of these "
                f"analyses. If who is missing that value is related to your "
                f"outcome, every cohort here is a selected sample.")

    if plan.blocked:
        names = ", ".join(f"{c.label} ({c.blocked_reason})" for c in plan.blocked[:3])
        out.append(
            f"**Some groups are too small to model on their own:** {names}. They stay "
            f"in your data — they are simply not offered as separate analyses, "
            f"because a number computed from that few people would mislead you.")
    return out


def _show(v: Any) -> str:
    """A value as a person would write it — not as numpy reprs it.

    repr() on a pandas value yields "np.int64(0)", which is noise in a sentence
    a researcher is meant to read.
    """
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.floating,)):
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def features_that_lose_variance(df: pd.DataFrame, mask: pd.Series,
                                feature_cols: Sequence[str]) -> List[Tuple[str, str]]:
    """Features that carry no signal INSIDE this cohort.

    Filtering to men makes `sex` constant, and any interaction involving it is
    gone with it. Handing a model a column with no variance is at best noise
    and at worst a crash, so the caller drops these — and says so, because a
    feature list that silently changed between runs breaks the promise that
    the runs answered the same question.
    """
    out: List[Tuple[str, str]] = []
    sub = df.loc[mask]
    if sub.empty:
        return out
    for col in feature_cols:
        if col not in sub.columns:
            continue
        s = sub[col]
        nn = s.dropna()
        if nn.empty:
            out.append((col, "has no values at all in this group"))
            continue
        try:
            counts = nn.value_counts()
        except TypeError:
            continue
        if len(counts) == 1:
            out.append((col, f"is always {_show(counts.index[0])} in this group"))
        elif counts.iloc[0] / len(nn) >= NEAR_CONSTANT_SHARE:
            out.append((col, f"is {counts.iloc[0] / len(nn):.0%} the same value "
                             f"({_show(counts.index[0])}) in this group"))
    return out


# ── the run registry ─────────────────────────────────────────────────────

@dataclass
class CohortRun:
    """One completed pass of the same analysis over one cohort.

    A run carries the QUESTION it answered, not just the group it answered it
    for. Without that, a banked number cannot be invalidated when the question
    changes and cannot be excluded when a different question is being asked —
    and the comparison table is precisely where a stale number turns into a
    published sentence.
    """
    column: str
    label: str
    n_train: int
    n_test: int
    dropped_features: List[str] = field(default_factory=list)
    completed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    # The question. Same target, same task, same data — otherwise these runs
    # are not comparable and must not appear beside each other.
    target_col: str = ""
    task_type: str = ""
    data_fingerprint: str = ""

    @property
    def question(self) -> Tuple[str, str, str, str]:
        return (self.column, self.target_col, self.task_type, self.data_fingerprint)


def runs_remaining(plan: CohortPlan, done: Sequence[str]) -> List[CohortCell]:
    """Cohorts still to analyze, in order. Drives the 'now do the men' button."""
    seen = set(done)
    return [c for c in plan.viable if c.label not in seen]


def cohort_candidates(df: pd.DataFrame, target_col: str,
                      max_levels: int = 20) -> List[str]:
    """Columns it would make sense to split the study by.

    A grouping variable is a handful of categories that describe PEOPLE. A
    subject ID has one row each and a lab value has hundreds of levels; neither
    is a cohort, and offering them buries the two or three columns that are.
    """
    out: List[str] = []
    if df is None or df.empty:
        return out
    n = len(df)
    for col in df.columns:
        if col == target_col or str(col).startswith("__source_file"):
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):     # duplicate column labels
            continue
        try:
            k = int(s.nunique(dropna=True))
        except TypeError:
            continue
        if k < 2 or k > max_levels:
            continue
        # A column with as many levels as rows is an identifier, not a group.
        if k >= max(n * 0.5, 2):
            continue
        out.append(str(col))
    return out


# ── the active run: what "different people" currently means ──────────────
#
# Held as index LABELS, not as a column filter. Feature engineering may one-hot
# `sex` out of existence, and a filter that silently stopped applying would let
# a run labeled "women" quietly train on everyone — the exact failure this
# whole feature exists to prevent. Labels survive every row-preserving step,
# which is the same invariant the test lockbox already depends on.

_ACTIVE_KEY = "cohort_run"
_DONE_KEY = "cohort_runs_done"
_BROKEN_KEY = "_cohort_filter_broken"


def active_cohort() -> Optional[Dict[str, Any]]:
    """The run in progress, or None when the study is being analyzed whole."""
    import streamlit as st
    run = st.session_state.get(_ACTIVE_KEY)
    return run if isinstance(run, dict) and run.get("labels") else None


def clear_cohort() -> None:
    import streamlit as st
    st.session_state.pop(_ACTIVE_KEY, None)
    st.session_state.pop(_BROKEN_KEY, None)


def start_cohort(df: pd.DataFrame, plan: CohortPlan, cell: CohortCell,
                 target_col: str, dropped_features: Optional[Sequence[str]] = None,
                 ) -> Dict[str, Any]:
    """Make `cell` the rows every downstream page works on."""
    import streamlit as st
    mask = cohort_mask(df, plan.column, cell.value)
    order = [c.label for c in plan.viable]
    run = {
        "column": plan.column,
        "value": cell.value,
        "label": cell.label,
        "labels": list(df.index[mask]),
        "n_rows": int(mask.sum()),
        "n_total": int(len(df)),
        "position": order.index(cell.label) + 1 if cell.label in order else 1,
        "of": len(order),
        "order": order,
        "target_col": target_col,
        "dropped_features": list(dropped_features or []),
    }
    st.session_state[_ACTIVE_KEY] = run
    st.session_state.pop(_BROKEN_KEY, None)
    return run


def apply_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to the active run. Idempotent; no-op when none is set.

    If neither the labels nor the grouping column can be found, this returns
    NOTHING rather than everything. An empty table is a visible failure the
    banner offers a way out of; a full table under a heading that says "women"
    is a result the researcher would publish.
    """
    import streamlit as st
    run = active_cohort()
    if run is None or df is None:
        return df
    labels = set(run["labels"])
    hit = df.index.isin(labels)
    if hit.any():
        return df.loc[hit]
    col, value = run.get("column"), run.get("value")
    if col and col in df.columns:
        return df.loc[cohort_mask(df, col, value)]
    st.session_state[_BROKEN_KEY] = True
    return df.iloc[0:0]


def cohort_filter_broken() -> bool:
    import streamlit as st
    return bool(st.session_state.get(_BROKEN_KEY))


def _current_question(column: str = "") -> Tuple[str, str, str, str]:
    """(grouping column, target, task, data fingerprint) as things stand now."""
    import streamlit as st
    dc = st.session_state.get("data_config")
    run = active_cohort()
    return (
        column or (run["column"] if run else ""),
        str(getattr(dc, "target_col", "") or ""),
        str(getattr(dc, "task_type", "") or ""),
        str(st.session_state.get("_raw_data_fingerprint", "")),
    )


def all_recorded_runs() -> List[CohortRun]:
    """Everything ever banked, including runs of other questions."""
    import streamlit as st
    raw = st.session_state.get(_DONE_KEY) or []
    return [r for r in raw if isinstance(r, CohortRun)]


def completed_runs(column: str = "") -> List[CohortRun]:
    """Runs that answered the question being asked RIGHT NOW.

    Filtered at read time rather than cleared on every path that could
    invalidate them: banked runs used to survive a corrected re-upload and a
    target swap, so a women's AUC computed on the old data — or on a different
    outcome entirely — sat in the comparison table beside a men's AUC computed
    on the new one, with nothing to distinguish them. Filtering here cannot be
    forgotten by a reset path that does not exist yet.
    """
    return [r for r in all_recorded_runs() if r.question == _current_question(column)]


def record_run(metrics: Optional[Dict[str, Any]] = None) -> Optional[CohortRun]:
    """Bank the run in progress so the next cohort can be compared to it."""
    import streamlit as st
    run = active_cohort()
    if run is None:
        return None
    from utils.test_lockbox import get_lockbox
    lb = get_lockbox()
    test_labels = set(lb["labels"]) if lb else set()
    in_cohort = set(run["labels"])
    _col, _target, _task, _fp = _current_question(run["column"])
    entry = CohortRun(
        column=run["column"], label=run["label"],
        n_train=len(in_cohort - test_labels),
        n_test=len(in_cohort & test_labels),
        dropped_features=list(run.get("dropped_features") or []),
        completed=True, metrics=dict(metrics or {}),
        target_col=_target, task_type=_task, data_fingerprint=_fp,
    )
    # Replace by (question, group), so re-running the same group under the same
    # question updates it while a run of a DIFFERENT question is left alone —
    # it simply stops being returned by completed_runs().
    done = [r for r in all_recorded_runs()
            if not (r.question == entry.question and r.label == entry.label)]
    done.append(entry)
    st.session_state[_DONE_KEY] = done
    return entry


def comparison_caveats(runs: Sequence[CohortRun], task_type: str) -> List[str]:
    """What NOT to conclude from putting two cohort runs side by side.

    This is where over-interpretation happens: two AUCs differing by 0.04 look
    like a finding and are frequently just different case mix or different
    training sizes.
    """
    out: List[str] = []
    done = [r for r in runs if r.completed]
    if len(done) < 2:
        return out

    sizes = [r.n_train for r in done]
    if max(sizes) >= 3 * max(min(sizes), 1):
        out.append(
            "The groups trained on very different numbers of rows, so the "
            "smaller model is handicapped before the comparison starts. A worse "
            "score there is not evidence that the relationship is weaker.")
    if task_type == "classification":
        out.append(
            "Accuracy and AUC depend on how common the outcome is. Comparing "
            "them across groups with different outcome rates compares two "
            "different things.")
    out.append(
        f"You fitted this model in {len(done)} groups. Report all "
        f"{len(done)}, not the one that worked — otherwise the result is a "
        f"multiple-comparisons artifact, and a reviewer will ask.")
    out.append(
        "Whether the difference between the groups is REAL is a question these "
        "separate fits cannot answer. That needs one model on everyone with an "
        "interaction term, which tests the difference directly.")
    return out
