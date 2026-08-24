"""turbotab.teaching — layer 3, and it is the preview mechanic, not a text panel.

Two rulings from the product owner, and the second is the mechanism.

**The app keeps asking hard questions and invests in making them answerable.**
The alternative — deciding more on the user's behalf — was considered and
rejected, because *answering these questions is itself an educational moment: it
gives the user internal clarity to make informed decisions downstream that they
can own.* So `DESIGN_LANGUAGE.md` §10's layers stop being aspirational.

**And there is no LLM and none is coming.** Layer 3 is the preview mechanic
pointed at interview questions instead of at repairs. Teaching means showing
consequences, and consequences are **computable**. Nothing here parses natural
language in either direction, and nothing here generates prose about a concept in
general.

## The four sub-questions, answered by computing

For each of the three hardest questions — grain, unit of analysis, aggregation —
the same four, because these are what a user predictably asks:

1. **What does each answer do to my data?** The real row-count change, the real
   split, the named columns. Not "aggregation reduces rows" — *"600 rows become
   300, one per `participant_id`."*
2. **A worked example on their own rows.** Not the concept in general:
   *"`P001` appears in rows 0 and 1. Under one row per person those become one
   row, and their `energy_kcal` values 2,347 and 1,982 become 2,164.5."*
3. **Are my repeats comparable?** Spacing regularity, and which columns actually
   differ within a person — because "comparable" is a measurement and not a
   reassurance.
4. **Which should I pick?** The app says it cannot answer this, plainly.

**That refusal is content, not a gap.** It is the fourth sub-question and it gets
the same weight as the other three, because a teaching panel that answered it
would be the app deciding on the user's behalf — which is the thing ruling 1
rejected.

## Why a test can demand it

A question with no layer-3 content fails
`test_every_hard_question_carries_its_teaching`. A new interview question cannot
ship without its teaching, which is the same shape as every other check in this
project: the failure mode is silence, so silence is a test failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# The questions layer 3 covers, and the ones a test demands content for. Named
# rather than derived from the Router, because "the hardest three" is a judgment
# and a list derived from the plan would silently grow.
TAUGHT = ("state_grain", "state_unit_of_analysis", "state_aggregation")

# The refusal, stated once. It is the answer to *"which should I pick?"* and it
# is the same sentence at every question, because the reason is the same one.
CANNOT_ANSWER = (
    "This one is yours, and the app will not guess at it. Everything above is "
    "what each answer does to your data; which of them describes your study is "
    "a fact about the study rather than about the table. Guessing here is what "
    "produced the leak the whole sealing rule exists to prevent.")


@dataclass
class Panel:
    """Layer 3 for one question: computed consequences, on their table."""
    question: str
    title: str
    consequences: List[Dict[str, Any]] = field(default_factory=list)
    worked_example: Optional[Dict[str, Any]] = None
    comparability: Optional[Dict[str, Any]] = None
    cannot_answer: str = CANNOT_ANSWER

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "title": self.title,
                "consequences": self.consequences,
                "worked_example": self.worked_example,
                "comparability": self.comparability,
                "cannot_answer": self.cannot_answer}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _repeating_column(df: pd.DataFrame, grain: Optional[Dict[str, Any]]
                      ) -> Optional[str]:
    """The identifier the record names, or the best shape-only candidate.

    Falls back to the suggestion rather than refusing, because layer 3 has to be
    able to teach the grain question BEFORE it is answered — that is the moment
    the teaching is for.
    """
    named = (grain or {}).get("group_col")
    if named and named in df.columns:
        return str(named)
    from turbotab import grain as _grain
    for candidate in _grain.suggestion(df)["columns"]:
        if candidate in df.columns:
            return str(candidate)
    return None


def _example_subject(df: pd.DataFrame, column: str) -> Optional[Any]:
    """A subject that actually repeats, so the worked example is a real case."""
    counts = df[column].value_counts(dropna=True)
    repeated = counts[counts > 1]
    return repeated.index[0] if len(repeated) else None


def _column_to_show(df: pd.DataFrame, column: str,
                    numeric: Sequence[str],
                    target: Optional[str] = None) -> Optional[str]:
    """The numeric column a worked example should use.

    **Not simply the first that varies.** The first version picked
    `recall_number` — the replicate index — which is the one column whose
    variation carries no information at all, so the example read *"their
    recall_number values 1 and 2 become 1.5"* and taught nothing. A worked
    example on a bookkeeping column is a worked example about bookkeeping.

    So the index and the ordering column are excluded, and what remains is
    ranked by how much it varies WITHIN a person relative to its own scale — the
    column whose combination the user would actually notice.
    """
    from turbotab import repeats as _rep

    reading = _rep.read(df, column)
    # THE TARGET IS EXCLUDED, and that is correctness rather than taste. The
    # first version picked `progressed` on `clinical_longitudinal` and showed
    # the user their OUTCOME being averaged to 0.3333 — which is not what
    # happens: `change_from_baseline` deliberately does not difference the
    # target, because a change score on the outcome asks a different research
    # question. A worked example that shows the one column the operation treats
    # differently is a worked example that teaches the wrong thing.
    skip = {reading.get("replicate_index"),
            (reading.get("spacing") or {}).get("column"), column, target}
    candidates = [c for c in numeric if c not in skip]
    if not candidates:
        candidates = [c for c in numeric if c != column]
    if not candidates:
        return None

    # Ranked by WITHIN-person spread against the column's OVERALL spread, which
    # is the thing "would the user notice this being combined" actually means. A
    # column that varies as much inside a person as across the study is one
    # aggregation changes a lot; one that is nearly constant within a person is
    # one it barely touches. Relative-to-the-mean was the first attempt and it
    # ranked a near-zero column top for having a small denominator.
    grouped = df.groupby(column, dropna=True)
    best, best_score = None, -1.0
    for c in candidates:
        try:
            within = float(grouped[c].std().mean())
            overall = float(df[c].std())
        except Exception:
            continue
        if not np.isfinite(within) or not np.isfinite(overall) or overall <= 0:
            continue
        score = within / overall
        if score > best_score:
            best, best_score = c, score
    return best or candidates[0]


def _varying_columns(block: pd.DataFrame, column: str) -> List[str]:
    return [str(c) for c in block.columns
            if str(c) != column and block[c].nunique(dropna=False) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# The three panels
# ─────────────────────────────────────────────────────────────────────────────

def grain_panel(df: pd.DataFrame, grain=None, target=None) -> Panel:
    """*"Can one person appear in more than one row?"*

    The consequence is the SPLIT, and it is computed both ways so the difference
    is a number rather than a warning. This is the question where a wrong answer
    produces optimistic held-out numbers that nothing on screen can show —
    except here, where it can.
    """
    panel = Panel(question="state_grain",
                  title="What each answer does to your held-out rows")
    column = _repeating_column(df, grain)
    n = len(df)

    if column is None:
        panel.consequences.append({
            "answer": "one_row_per_person",
            "headline": f"{n:,} rows, and nothing in this table repeats",
            "detail": ("No column has values that recur like a roster, so a "
                       "random split holds out rows and rows are people."),
            "n_rows": n})
        panel.comparability = {
            "verdict": "nothing_repeats",
            "detail": "No column repeats, so there is nothing to compare."}
        return panel

    groups = int(df[column].nunique(dropna=True))
    per = n / max(groups, 1)
    fraction = 0.15
    n_test_rows = int(round(n * fraction))
    n_test_groups = max(1, int(round(groups * fraction)))

    panel.consequences.append({
        "answer": "one_row_per_person",
        "headline": f"{n_test_rows:,} of {n:,} rows held out, chosen at random",
        "detail": (f"`{column}` has {groups:,} distinct values across {n:,} rows, "
                   f"about {per:.1f} each. If those are the same people, a "
                   f"random draw puts some of them on both sides of the split "
                   f"and the held-out score reads better than the model is."),
        "n_rows": n, "n_held_out": n_test_rows, "leaks": per > 1.05})
    panel.consequences.append({
        "answer": "people_repeat",
        "headline": (f"{n_test_groups:,} of {groups:,} {column} values held out "
                     f"whole — about {n_test_groups * per:.0f} rows"),
        "detail": (f"Whole values of `{column}` go to one side or the other, so "
                   f"nothing about a held-out subject appears in training."),
        "n_rows": n, "n_groups": groups, "n_held_out_groups": n_test_groups,
        "leaks": False})

    subject = _example_subject(df, column)
    if subject is not None:
        block = df[df[column] == subject]
        varying = _varying_columns(block, column)
        panel.worked_example = {
            "subject": str(subject), "column": column,
            "rows": [_plain(i) for i in block.index[:4]],
            "n_rows": int(len(block)),
            "varying_columns": varying[:6],
            "sentence": (
                f"`{subject}` appears in {len(block)} rows "
                f"({', '.join(str(_plain(i)) for i in block.index[:4])}"
                + (", …" if len(block) > 4 else "") + "). "
                + (f"Between them {len(varying)} column(s) differ, including "
                   f"`{'`, `'.join(varying[:3])}`. "
                   if varying else "Nothing differs between them. ")
                + "Under a random split some of these rows could be held out "
                  "while the others train the model."),
        }

    panel.comparability = _comparability(df, column)
    return panel


def unit_panel(df: pd.DataFrame, grain=None, target=None) -> Panel:
    """*"When you analyze this, what is one row?"*

    The question with **no default**, so the teaching carries more weight here
    than anywhere else: the app is refusing to choose and owes the user the
    means to.
    """
    panel = Panel(question="state_unit_of_analysis",
                  title="What each answer does to your table")
    column = _repeating_column(df, grain)
    n = len(df)
    if column is None:
        panel.consequences.append({
            "answer": "record", "headline": f"{n:,} rows, unchanged",
            "detail": "Nothing repeats, so there is nothing to combine.",
            "n_rows_before": n, "n_rows_after": n})
        return panel

    groups = int(df[column].nunique(dropna=True))
    numeric = [str(c) for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and str(c) != column]
    other = [str(c) for c in df.columns
             if str(c) not in numeric and str(c) != column]
    varying_other = [c for c in other
                     if int(df.groupby(column, dropna=True)[c]
                            .nunique(dropna=True).max() or 0) > 1]

    panel.consequences.append({
        "answer": "person",
        "headline": f"{n:,} rows become {groups:,} — one per {column}",
        "detail": (f"{len(numeric):,} numeric column(s) are combined into one "
                   f"value each. "
                   + (f"{len(varying_other):,} non-numeric column(s) vary within "
                      f"a person and would take the first value: "
                      f"`{'`, `'.join(varying_other[:3])}`."
                      if varying_other else
                      "No non-numeric column varies within a person, so nothing "
                      "there loses information.")),
        "n_rows_before": n, "n_rows_after": groups,
        "n_numeric": len(numeric),
        "loses": varying_other[:6]})
    panel.consequences.append({
        "answer": "record",
        "headline": f"{n:,} rows stay {n:,}",
        "detail": (f"Every record is a row and whole {column} values are held "
                   f"out, so no {column} appears on both sides. Nothing is "
                   f"averaged and nothing is dropped."),
        "n_rows_before": n, "n_rows_after": n, "loses": []})

    subject = _example_subject(df, column)
    if subject is not None and numeric:
        block = df[df[column] == subject]
        col = _column_to_show(df, column, numeric, target)
        if col is None:
            return panel
        values = [_plain(v) for v in block[col].tolist()[:4]]
        combined = _plain(round(float(block[col].mean()), 4))
        panel.worked_example = {
            "subject": str(subject), "column": column, "shown_column": col,
            "rows": [_plain(i) for i in block.index[:4]],
            "values": values, "combined": combined,
            "sentence": (
                f"`{subject}` appears in rows "
                f"{', '.join(str(_plain(i)) for i in block.index[:4])}"
                + (", …" if len(block) > 4 else "")
                + f". Under one row per person those become one row, and their "
                  f"`{col}` values "
                + " and ".join(str(v) for v in values)
                + f" become {combined}."),
        }
    panel.comparability = _comparability(df, column)
    return panel


def aggregation_panel(df: pd.DataFrame, grain=None, target=None) -> Panel:
    """*"How should each person's rows be combined?"*

    Four options, and the consequence of each is computed on one real subject —
    so *"the mean"* and *"the last"* are two numbers the user can compare rather
    than two words.
    """
    from turbotab import repeats as _rep

    panel = Panel(question="state_aggregation",
                  title="What each way of combining does to your numbers")
    column = _repeating_column(df, grain)
    if column is None:
        return panel

    subject = _example_subject(df, column)
    numeric = [str(c) for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and str(c) != column]
    reading = _rep.read(df, column)
    order = (reading.get("spacing") or {}).get("column") or \
        reading.get("replicate_index")

    show = None
    if subject is not None and numeric:
        block = df[df[column] == subject]
        if order and order in df.columns:
            key = block[order]
            if not pd.api.types.is_numeric_dtype(key):
                key = pd.to_datetime(key, errors="coerce")
            block = block.assign(_o=key).sort_values("_o").drop(columns=["_o"])
        show = _column_to_show(df, column, numeric, target)
        series = block[show].dropna() if show else pd.Series(dtype=float)
        if len(series):
            outcomes = {
                _rep.MEAN: float(series.mean()),
                _rep.FIRST: float(series.iloc[0]),
                _rep.LAST: float(series.iloc[-1]),
                _rep.CHANGE: float(series.iloc[-1] - series.iloc[0]),
            }
            for key, value in outcomes.items():
                panel.consequences.append({
                    "answer": key,
                    "headline": f"`{show}` for `{subject}` becomes "
                                f"{round(value, 4)}",
                    "detail": _AGG_DETAIL[key],
                    "value": round(value, 4)})
            panel.worked_example = {
                "subject": str(subject), "column": column,
                "shown_column": show,
                "ordered_by": order,
                "values": [_plain(round(float(v), 4)) for v in series.tolist()[:5]],
                "sentence": (
                    f"`{subject}`'s `{show}` values are "
                    + ", ".join(str(_plain(round(float(v), 4)))
                               for v in series.tolist()[:5])
                    + (", …" if len(series) > 5 else "")
                    + (f", in `{order}` order" if order else "")
                    + f". The mean is {round(float(series.mean()), 4)}, the "
                      f"first is {round(float(series.iloc[0]), 4)}, the last is "
                      f"{round(float(series.iloc[-1]), 4)}, and the change is "
                      f"{round(float(series.iloc[-1] - series.iloc[0]), 4)}."),
            }
    panel.comparability = _comparability(df, column)
    return panel


_AGG_DETAIL = {
    "mean": ("Averages every record. Correct when the records are repeated "
             "measurements of one quantity, because the noise being averaged "
             "away is measurement error."),
    "first": ("Keeps the earliest record and drops the rest. A baseline "
              "measurement, which is a different variable from an average."),
    "last": ("Keeps the latest record and drops the rest. The most recent "
             "state, which is what a prediction at follow-up usually wants."),
    "change_from_baseline": (
        "The difference between the last and the first. A change score answers "
        "a different question from a level — who improved rather than who is "
        "ill."),
}


def _comparability(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """*"Are my repeats comparable?"* — measured, not reassured.

    Spacing regularity and which columns actually differ within a person. Both
    are computations, and both are the evidence the repeats-or-time-points
    reading turns on, shown here rather than asserted there.
    """
    from turbotab import repeats as _rep

    reading = _rep.read(df, column)
    spacing = reading.get("spacing")
    out: Dict[str, Any] = {"verdict": reading.get("reading") or "thin_evidence",
                           "evidence": list(reading.get("evidence") or []),
                           "replicate_index": reading.get("replicate_index")}
    if spacing:
        out["spacing"] = {k: _plain(v) for k, v in spacing.items()}
        out["detail"] = (
            f"A person's records in `{spacing['column']}` are "
            f"{spacing['median_days']:.0f} days apart at the median, ranging "
            f"{spacing['min_days']:.0f} to {spacing['max_days']:.0f} "
            f"(coefficient of variation {spacing['cv']:.2f}). "
            + ("Regular enough to be a schedule, which makes these time points."
               if spacing["cv"] <= 0.35 and spacing["median_days"] >= 14 else
               "Too uneven or too close together to be a schedule."))
    else:
        out["detail"] = ("There is no date column, so nothing here spaces a "
                         "person's records out in time.")

    # Which columns actually differ, across every subject rather than one.
    grouped = df.groupby(column, dropna=True)
    differ = []
    for c in df.columns:
        if str(c) == column:
            continue
        try:
            if int(grouped[c].nunique(dropna=False).max() or 0) > 1:
                differ.append(str(c))
        except Exception:
            continue
    out["varying_columns"] = differ[:12]
    out["n_varying"] = len(differ)
    out["constant_columns"] = [str(c) for c in df.columns
                               if str(c) != column and str(c) not in differ][:12]
    return out


PANELS = {
    "state_grain": grain_panel,
    "state_unit_of_analysis": unit_panel,
    "state_aggregation": aggregation_panel,
}


def panel(question: str, df: pd.DataFrame, grain=None,
          target=None) -> Optional[Dict[str, Any]]:
    """Layer 3 for one question, or `None` where none is defined."""
    build = PANELS.get(question)
    if build is None:
        return None
    return build(df, grain, target).to_dict()
