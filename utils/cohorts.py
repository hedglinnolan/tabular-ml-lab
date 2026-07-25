"""Analysing one cohort at a time — same question, different people.

Researchers ask for this constantly: "model the men, then model the women."
The app already had a button labelled subgroup analysis, but it answers a
DIFFERENT question, and the two get conflated with real consequences:

    Does my model work equally well for men and women?
        -> train on everyone, evaluate the one model within each group.
           A fairness / generalisation check. This is what page 07 does.

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
    """One level of the grouping variable, and whether it can be modelled."""
    label: str
    value: Any
    n_rows: int
    n_subjects: int
    n_train: int
    n_test: int
    viable: bool = True
    blocked_reason: str = ""
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
            return f"'{self.column}' has no usable levels."
        parts = ", ".join(f"{c.label} ({c.n_rows:,})" for c in self.viable)
        out = (f"Analysing separately by **{self.column}**: {len(self.viable)} "
               f"cohort(s) — {parts}.")
        if self.blocked:
            out += (f" {len(self.blocked)} level(s) are too small to model and "
                    f"are not offered.")
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
            f"outcome would leave each cohort with only one answer in it.")
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
            f"study that many ways leaves nothing in each cohort — pick a "
            f"variable with a handful of categories.")
        return plan

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    for value in sorted(levels, key=lambda v: str(v)):
        mask = cohort_mask(df, column, value)
        rows = df.loc[mask]
        in_train = mask & train_mask
        in_test = mask & ~train_mask
        n_subjects = (int(rows[group_col].nunique())
                      if group_col and group_col in df.columns else int(mask.sum()))
        cell = CohortCell(
            label=str(value), value=value, n_rows=int(mask.sum()),
            n_subjects=n_subjects, n_train=int(in_train.sum()),
            n_test=int(in_test.sum()),
        )

        if task_type == "classification" and target_col in df.columns:
            counts = rows[target_col].value_counts().to_dict()
            cell.class_counts = {k: int(v) for k, v in counts.items()}

        reasons: List[str] = []
        if cell.n_rows < MIN_ROWS_PER_COHORT:
            reasons.append(f"only {cell.n_rows} rows (needs {MIN_ROWS_PER_COHORT})")
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
            f"Only {len(plan.viable)} level of '{column}' is big enough to model "
            f"on its own, so there is nothing to compare. Keep everyone together "
            f"and add '{column}' as a predictor instead.")
    return plan


def _cohort_warnings(plan: CohortPlan, task_type: str) -> List[str]:
    """What splitting the study this way costs, said before it is done."""
    out: List[str] = []
    viable = plan.viable

    if len(viable) > MAX_SENSIBLE_CELLS:
        out.append(
            f"**{len(viable)} separate analyses is a lot.** Every extra cohort is "
            f"another chance for one of them to look significant by accident. If "
            f"you report the one that worked, that is the multiple-comparisons "
            f"problem, and it needs disclosing.")

    if viable:
        smallest = min(viable, key=lambda c: c.n_rows)
        largest = max(viable, key=lambda c: c.n_rows)
        if largest.n_rows >= 3 * max(smallest.n_rows, 1):
            out.append(
                f"**The cohorts are very different sizes** — {largest.label} has "
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
                f"**The outcome is much commoner in one cohort** ({hi}: "
                f"{rates[hi]:.0%}, {lo}: {rates[lo]:.0%}). Accuracy and AUC are "
                f"not directly comparable across groups with different rates — a "
                f"difference between them can be case mix rather than model "
                f"quality.")

    if plan.blocked:
        names = ", ".join(f"{c.label} ({c.blocked_reason})" for c in plan.blocked[:3])
        out.append(
            f"**Some levels cannot be modelled on their own:** {names}. They stay "
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
            out.append((col, "has no values at all in this cohort"))
            continue
        try:
            counts = nn.value_counts()
        except TypeError:
            continue
        if len(counts) == 1:
            out.append((col, f"is always {_show(counts.index[0])} in this cohort"))
        elif counts.iloc[0] / len(nn) >= NEAR_CONSTANT_SHARE:
            out.append((col, f"is {counts.iloc[0] / len(nn):.0%} the same value "
                             f"({_show(counts.index[0])}) in this cohort"))
    return out


# ── the run registry ─────────────────────────────────────────────────────

@dataclass
class CohortRun:
    """One completed pass of the same analysis over one cohort."""
    column: str
    label: str
    n_train: int
    n_test: int
    dropped_features: List[str] = field(default_factory=list)
    completed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


def runs_remaining(plan: CohortPlan, done: Sequence[str]) -> List[CohortCell]:
    """Cohorts still to analyse, in order. Drives the 'now do the men' button."""
    seen = set(done)
    return [c for c in plan.viable if c.label not in seen]


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
            "The cohorts trained on very different numbers of rows, so the "
            "smaller model is handicapped before the comparison starts. A worse "
            "score there is not evidence that the relationship is weaker.")
    if task_type == "classification":
        out.append(
            "Accuracy and AUC depend on how common the outcome is. Comparing "
            "them across cohorts with different outcome rates compares two "
            "different things.")
    out.append(
        f"You fitted this model in {len(done)} cohorts. Report all "
        f"{len(done)}, not the one that worked — otherwise the result is a "
        f"multiple-comparisons artefact, and a reviewer will ask.")
    out.append(
        "Whether the difference between cohorts is REAL is a question these "
        "separate fits cannot answer. That needs one model on everyone with an "
        "interaction term, which tests the difference directly.")
    return out
