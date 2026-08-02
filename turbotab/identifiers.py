"""turbotab.identifiers — a column that names a row is not a predictor.

`GUIDED-108`, found at L37 while rejecting a trigger for the resolution card,
and the measurement is the argument: **`respondent_id` is 299 of
`survey_instrument.csv`'s 344 candidate parameters.** `admission_id` is 159 of
`leaky_sepsis.csv`'s 164. `training._feature_frame` drops the target and the
grain's group column and nothing else, so every one of those reaches the
one-hot encoder and then the model.

Two consequences, and the second is the quiet one:

**A level that appears exactly once is a row's name.** A model with capacity
can separate the outcome perfectly through it and has learned nothing that
transfers. This is not leakage in the strict sense — the value was there at
prediction time — but the failure mode is the same shape: apparent performance
that no held-out set can reproduce.

**It inflates every parameter count the app reasons with.** Including the one
`GUIDED-102`'s card would have used had its third trigger shipped, which is how
this was found: `parameters > training rows` fired on five of six fixtures and
every one of them was this.

## What the core knows, and what it does not

`AUDIT-008` is *the core already holds the capability and the path that needs
it does not read it*, and this began as an instance of it —
`ml/dataset_profile.py` computes `id_like_features`. **Driven, that capability
answers `False` for every column that motivated the finding.** Its rule is

    unique_count == n and is_numeric and is_integer_dtype and not is_bool

and `respondent_id`, `admission_id`, `patient_id` and `sample_id` are all
*strings*. So the core's answer is read here, and it is not sufficient on its
own: reading it and stopping would have closed the row while fixing none of the
four cases in the argument for it. Filed as `GUIDED-120`.

## The rule, and why it is not a name list

`patient_id` is a name, and names cannot close this — a column called `code`
can be an identifier and a column called `patient_id` can be a real category on
a table with one row per visit. The arithmetic fact is:

> **A column whose observed level count equals its row count has no level that
> appears twice, so nothing about it can generalize.**

That is checkable without knowing what the column is called, it is the same
arithmetic the core uses, and it is stated in the sentence the user reads so
they can disagree with it.

## The shelf is never shortened

`PRODUCT_VISION.md`'s rule, and it decides the shape here. An identifier is
**excluded with a sentence, not removed in silence**, and the exclusion is a
recorded decision the user can overturn in one click. A column dropped without
a receipt is indistinguishable from a column the app never saw.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

#: The arithmetic threshold, and it is not tuned: **1.0** means every observed
#: level appears exactly once. Stated as a constant rather than inlined because
#: it is the number a reader would want to argue with, and it is reported in
#: the sentence beside the count it came from.
#:
#: It is deliberately not lowered. A column at 0.95 distinct-per-row is *nearly*
#: an identifier and the app does not know whether the 5% are real repeats or a
#: typo, so it says nothing rather than guessing — `GUIDED-121` is where the
#: near-unique case is filed, unbuilt.
UNIQUE_PER_ROW = 1.0


def detect(frame: pd.DataFrame, target: Optional[str] = None,
           group_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Columns that name a row rather than describe one.

    **Named `detect` rather than `survey`, and the rename was forced by a
    guard.** `rankings.CALL_SITES` tracks every function called `survey` as a
    ranking primitive, because `GUIDED-088` established that the ORDER a user
    picks from is a parameter estimated from data and must not see the held-out
    rows. This is not a ranking — nobody picks from it — but the collision was
    the right question asked at the wrong name, and the answer to the right
    question is below: `excluded` reads the TRAINING rows.

    The core's `is_id_like` is asked first and its answer is carried, so a
    column it flags is flagged here for the core's reason. The arithmetic below
    catches the ones its dtype gate excludes — which, driven, is all four of
    the columns in `GUIDED-108`'s own evidence.
    """
    from ml.dataset_profile import compute_feature_profile

    drop = {str(target)} if target else set()
    if group_col:
        # THE GRAIN'S GROUPING COLUMN IS NOT AN IDENTIFIER TO EXCLUDE, even
        # when it is unique per row. It is already dropped from the feature
        # frame, and it is the answer to a question the user was asked — naming
        # it here would report a decision back to them as a defect.
        drop.add(str(group_col))

    n = len(frame)
    out: List[Dict[str, Any]] = []
    if n == 0:
        return out

    for column in frame.columns:
        name = str(column)
        if name in drop:
            continue
        series = frame[column]
        if isinstance(series, pd.DataFrame):                # pragma: no cover
            continue
        present = series.dropna()
        if len(present) < n:
            # A column with blanks cannot have one level per ROW, and treating
            # it as unique-per-present-value would flag any sparse column.
            continue
        levels = int(present.nunique())
        ratio = levels / float(n)
        if ratio < UNIQUE_PER_ROW:
            continue

        core = compute_feature_profile(frame, name, n)
        numeric = pd.api.types.is_numeric_dtype(series)

        # **UNIQUE PER ROW IS NOT ENOUGH, AND DRIVING IS WHAT SHOWED IT.**
        #
        # The first version flagged every column at one level per row and, on
        # `metabolomics_untargeted.csv`, that was `sample_id` plus about NINETY
        # `mz_*` columns — the study's actual predictors. A continuous
        # measurement is unique per row because it is continuous; every float
        # differs. Excluding them would have deleted the analysis to protect it.
        #
        # The distinction is what the model can DO with the values. A numeric
        # column is used for its ORDER, and an order over n distinct values is
        # exactly what a continuous predictor is. A text column with one level
        # per row has no order to use: one-hot encoding spends n−1 parameters
        # and each is true for exactly one row, which is a row's name written
        # as a matrix.
        #
        # So a numeric column is flagged only where the CORE flags it — its
        # rule adds integer dtype, which is the arithmetic for *this is a row
        # number* rather than *this is a measurement*. That is `AUDIT-008`
        # honored where the core is right, and extended where its dtype gate
        # excludes the four cases in this finding's own evidence.
        if numeric and not getattr(core, "is_id_like", False):
            continue
        out.append({
            "column": name,
            "n_levels": levels,
            "n_rows": int(n),
            "distinct_per_row": round(ratio, 4),
            # WHETHER THE CORE AGREES, carried rather than recomputed. `False`
            # here on a flagged column is not a contradiction — it is the
            # dtype gate, and it is the reason this module exists.
            "core_says_id_like": bool(getattr(core, "is_id_like", False)),
            "kind": "numeric" if numeric else f"{levels} text levels",
            # WHAT IT COSTS THE MODEL, because the count is the argument.
            "parameters": 1 if numeric else max(0, levels - 1),
            "sentence": _sentence(name, levels, n, numeric),
        })
    out.sort(key=lambda d: -d["parameters"])
    return out


def _sentence(column: str, levels: int, n: int, numeric: bool) -> str:
    """What the user reads, and what they are being invited to overturn.

    States the arithmetic rather than the conclusion, because the conclusion is
    theirs: a near-unique code CAN be a real category, and the app does not
    know this table's subject.
    """
    cost = ("one column" if numeric
            else f"{max(0, levels - 1):,} columns after encoding")
    return (
        f"`{column}` has {levels:,} different values across {n:,} rows — one "
        f"for every row. A value that appears exactly once cannot tell the "
        f"model anything about a row it has not seen, and as a predictor it "
        f"would cost {cost}. It has been left out of the models. If it is a "
        f"real measurement that happens to be unique here, put it back.")


def excluded(project: Any) -> List[str]:
    """Which identifier columns this project is currently keeping out.

    The detected set minus anything the user put back. Computed rather than
    stored, so a column that stops being unique — because rows were trimmed —
    stops being excluded without anybody having to remember to un-exclude it.
    """
    frame = _seen_rows(project)
    if frame is None:
        return []
    kept = {str(c) for c in (getattr(project, "kept_identifiers", None) or [])}
    group_col = (getattr(project, "grain", None) or {}).get("group_col")
    return [row["column"] for row in detect(frame, project.target, group_col)
            if row["column"] not in kept]


def _seen_rows(project: Any) -> Optional[pd.DataFrame]:
    """THE TRAINING ROWS, not the whole table.

    Prompted by the `rankings` guard colliding on the old name, and the
    question it was really asking is a fair one: *which rows is this allowed to
    see?* Uniqueness is a structural property rather than an outcome-related
    one, so reading the whole table would probably be harmless — but *probably
    harmless* is the argument `GUIDED-088` was filed against, and reading the
    training rows costs nothing.

    It is also the more conservative direction: a column unique across 300 rows
    is still unique across its 240-row training subset, and a column that is
    unique only within the training rows is one the model would memorize
    exactly as thoroughly.
    """
    if getattr(project, "training_rows", None) is not None:
        return project.training_rows
    return getattr(project, "df", None)                     # pragma: no cover


def receipt(project: Any) -> Optional[Dict[str, Any]]:
    """What to show the user, or `None` when there is nothing to say.

    `None` rather than an empty card: a table with no identifier columns has
    not had any excluded, and a labeled empty region would read as a finding of
    nothing.
    """
    frame = _seen_rows(project)
    if frame is None:
        return None
    group_col = (getattr(project, "grain", None) or {}).get("group_col")
    found = detect(frame, project.target, group_col)
    if not found:
        return None
    kept = {str(c) for c in (getattr(project, "kept_identifiers", None) or [])}
    left_out = [row for row in found if row["column"] not in kept]
    put_back = [row for row in found if row["column"] in kept]
    return {
        "columns": found,
        "excluded": [row["column"] for row in left_out],
        "kept": [row["column"] for row in put_back],
        "parameters_saved": sum(row["parameters"] for row in left_out),
        "headline": (
            f"{len(left_out):,} column(s) that name a row rather than describe "
            f"one have been left out of the models"
            if left_out else
            f"{len(put_back):,} column(s) that name a row are included at your "
            f"request"),
        "rule": (
            f"A column is treated this way when every one of its values is "
            f"different — {UNIQUE_PER_ROW:.0%} distinct across the rows the "
            f"models see. Nothing about how it is named is used."),
    }
