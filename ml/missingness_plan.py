"""Missingness, named and routed by dtype — and timed correctly.

The drive: *"2 features with high missingness"* does not say which two. A count
with no names is a claim the reader cannot check, and the engine knew the names
the whole time.

Naming them is half of it. The other half is that "what do I do about missing
values" is not one question — it is three, and which one you are being asked
depends on the column's type:

  * **binary** — is the absence itself informative? A medication history that
    was never asked is different from one recorded as "no". The options are an
    explicit indicator, imputation, or leaving it missing for models that
    handle it.
  * **numeric** — which imputation, with the before/after distribution shown,
    because mean and median imputation do visibly different things to a skewed
    variable.
  * **categorical** — an explicit `Missing` level, or the mode. The first keeps
    the absence as a fact about the participant; the second asserts a value
    nobody recorded.

**The action-timing ruling, which is why the sentences read the way they do.**
Structural repairs execute immediately on the working table: they change what
the table *is*, and everything downstream reads the repaired frame. Statistical
transforms — imputation, scaling, encoding — are *recorded as decisions* and
execute inside each model's pipeline, on training folds only, because a
statistic fitted on all the rows before splitting leaks the held-out rows into
every fold. Both are decided here; only the first happens here.

The user is told this as methods prose, in the decision sentence itself — "will
be imputed with the training-fold median" — never as a note about how the
software works and never hidden. If the sentence could not appear in a methods
section, it is not a decision sentence (`DESIGN_LANGUAGE.md` §06).

Finding: GUIDED-002.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ml.binary_text import read_as_binary_plan

# Below this share of missing values a column is not worth a card of its own.
# It matches `dataset_profile`'s own "high missingness" tier so the two agree
# about which columns the interface is talking about.
HIGH_MISSING_SHARE = 0.20

# Structural repairs run now; statistical transforms run inside the per-model
# pipeline on training folds. Exposed as data so the frontend cannot invent a
# third timing, and so a test can assert that no card claims one.
TIMING_IMMEDIATE = "immediate"
TIMING_IN_PIPELINE = "in_pipeline"
# A third, because two of these options are genuinely BOTH (`DRIVE-008`). Adding
# a was-it-missing indicator is row-local and runs now; filling the value under
# it learns from a distribution and runs in the fold. Calling that compound
# "in_pipeline" understated what had already happened to the table, and calling
# it "immediate" would have overstated it — and clause §06's whole point is that
# the user is told WHEN.
TIMING_MIXED = "mixed"

_TIMING_PROSE = {
    TIMING_IMMEDIATE: "applied to the working table now",
    TIMING_IN_PIPELINE: "fitted inside each model's pipeline, on training folds only",
    TIMING_MIXED: ("the indicator is added now; the value under it is fitted "
                   "inside each model's pipeline, on training folds only"),
}


# How many rows of real data travel with a card. Three blank and three present:
# enough to see what a blank looks like beside what a value looks like, and few
# enough that the card is still a card rather than a table viewer.
SNIPPET_ROWS = 3

# Columns shown beside the one in question. The card is about ONE column, and a
# snippet of that column alone is a list of the word "missing" — the question is
# what the rows where it is blank have in common, which needs neighbors.
SNIPPET_COLUMNS = 3


def _snippet(df: "pd.DataFrame", column: str,
             series: "pd.Series") -> Dict[str, Any]:
    """Real rows from this table: some where the column is blank, some not.

    `DRIVE-008`. Row LABELS, not positions, because that is what row identity
    is in this project and it is what makes the snippet checkable against the
    file the user has open.

    Neighboring columns are the first few that are not the column itself and
    are not mostly blank themselves — a neighbor with no values in it answers
    nothing and spends a column doing it.
    """
    labels = [str(c) for c in df.columns if str(c) != column]
    neighbors: List[str] = []
    for name in labels:
        col = df[name]
        if isinstance(col, pd.DataFrame):
            continue
        if float(col.isna().mean()) > 0.5:
            continue
        neighbors.append(name)
        if len(neighbors) >= SNIPPET_COLUMNS:
            break

    blank_idx = list(series[series.isna()].index[:SNIPPET_ROWS])
    present_idx = list(series[series.notna()].index[:SNIPPET_ROWS])

    def row(label: Any, missing: bool) -> Dict[str, Any]:
        cells = {}
        for name in neighbors:
            value = df.at[label, name]
            cells[name] = None if pd.isna(value) else _plainly(value)
        return {"row": _plainly(label), "missing": missing,
                "value": None if missing else _plainly(series.loc[label]),
                "cells": cells}

    return {
        "column": column,
        "neighbors": neighbors,
        "rows": ([row(i, True) for i in blank_idx]
                 + [row(i, False) for i in present_idx]),
        "n_blank_shown": len(blank_idx),
        "n_present_shown": len(present_idx),
    }


def _plainly(value: Any) -> Any:
    """A cell as JSON, without pretending a numpy scalar is a Python one."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):                    # pragma: no cover
            pass
    return str(value)


def _kind_of(series: pd.Series) -> str:
    """binary | numeric | categorical — the routing key."""
    if isinstance(series, pd.DataFrame):
        return "categorical"
    present = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "binary" if set(np.unique(present.to_numpy())) <= {0, 1} and len(present) else "numeric"
    return "binary" if read_as_binary_plan(series) is not None else "categorical"


def _option(key: str, label: str, sentence: str, timing: str,
            consequence: str, recommended: bool = False) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        # The sentence the transcript will carry, in methods register. Past or
        # future tense, a named column, a named statistic — publishable as is.
        "decision_sentence": sentence,
        "timing": timing,
        "timing_prose": _TIMING_PROSE[timing],
        "consequence": consequence,
        "recommended": recommended,
    }


def _binary_options(column: str, n_missing: int, n_rows: int) -> List[Dict[str, Any]]:
    return [
        _option(
            "indicator",
            "Keep the absence as information",
            (f"Missingness in `{column}` was treated as informative: a binary "
             f"indicator was added and the value itself imputed within each "
             f"training fold."),
            # The sentence above says both halves; the timing now does too.
            TIMING_MIXED,
            (f"Adds one column. The model can learn from *whether* {column} was "
             f"recorded, which is the right reading when the absence has a cause "
             f"— not asked, not applicable, refused."),
            recommended=True,
        ),
        _option(
            "impute_mode",
            "Treat it as the common level",
            (f"Missing values in `{column}` were imputed with the most frequent "
             f"level of the training folds."),
            TIMING_IN_PIPELINE,
            (f"Asserts a value for {n_missing:,} participant(s) nobody recorded. "
             "Defensible when the absence is clerical and unrelated to the "
             "outcome; indefensible when it is not, and nothing in the file says "
             "which."),
        ),
        _option(
            "leave",
            "Leave it missing",
            (f"Missing values in `{column}` were left missing; models that accept "
             f"missing values were fitted on the column as recorded."),
            TIMING_IN_PIPELINE,
            ("Gradient boosting handles this natively. Linear and neural models "
             "do not, and will refuse the column or the row."),
        ),
    ]


def _numeric_options(column: str, series: pd.Series, n_missing: int) -> List[Dict[str, Any]]:
    present = pd.to_numeric(series, errors="coerce").dropna()
    skew = float(present.skew()) if len(present) > 2 else 0.0
    skewed = abs(skew) > 1.0
    median_first = skewed
    return [
        _option(
            "impute_median",
            "Impute with the median",
            (f"Missing values in `{column}` were imputed with the training-fold "
             f"median."),
            TIMING_IN_PIPELINE,
            (f"`{column}` is skewed (skew = {skew:.2f}), so the mean sits away "
             "from the bulk of the data and the median is the more representative "
             "filler."
             if skewed else
             "Robust to the tails. Shrinks the variance of the column, which "
             "biases any standard error computed from it toward zero."),
            recommended=median_first,
        ),
        _option(
            "impute_mean",
            "Impute with the mean",
            (f"Missing values in `{column}` were imputed with the training-fold "
             f"mean."),
            TIMING_IN_PIPELINE,
            ("Preserves the column mean exactly and nothing else. On a skewed "
             "column it places every filled value where few real ones are."
             if skewed else
             "Preserves the column mean exactly; shrinks its variance."),
            recommended=not median_first,
        ),
        _option(
            "impute_iterative",
            "Impute from the other columns",
            (f"Missing values in `{column}` were imputed by iterative "
             f"regression on the remaining features, fitted within each "
             f"training fold."),
            TIMING_IN_PIPELINE,
            ("Uses the correlations in the data rather than one number. Costs "
             "run time and makes the imputation itself a model you have to "
             "describe."),
        ),
        _option(
            "indicator_and_impute",
            "Impute, and record that it was missing",
            (f"Missing values in `{column}` were imputed with the training-fold "
             f"median and a missingness indicator was retained."),
            # Compound, like the binary `indicator` above and for the same
            # reason: the indicator is row-local and lands now, the median is
            # fitted in the fold.
            TIMING_MIXED,
            (f"Adds one column and keeps the fact that {n_missing:,} value(s) "
             "were absent available to the model."),
        ),
    ]


def _categorical_options(column: str, series: pd.Series, n_missing: int) -> List[Dict[str, Any]]:
    present = series.dropna()
    mode = present.mode()
    mode_label = str(mode.iloc[0]) if len(mode) else "the most frequent level"
    return [
        _option(
            "explicit_missing",
            "Make Missing its own level",
            (f"Missing values in `{column}` were encoded as an explicit "
             f"`Missing` category."),
            # ROW-LOCAL, and this said `in_pipeline` (`DRIVE-008`). A blank
            # becoming the literal level `Missing` consults nothing but that
            # row's own cell, and `project.route_missingness` has always
            # executed it immediately — so the card was stating a timing the
            # server contradicted, on the one clause that is about timing.
            TIMING_IMMEDIATE,
            ("Keeps the absence as a fact about the participant, and lets the "
             "model estimate whether it carries signal. Adds one level to the "
             "encoding."),
            recommended=True,
        ),
        _option(
            "impute_mode",
            f"Treat them as {mode_label}",
            (f"Missing values in `{column}` were imputed with the most frequent "
             f"level of the training folds."),
            TIMING_IN_PIPELINE,
            (f"Asserts `{mode_label}` for {n_missing:,} participant(s) nobody "
             "recorded, and inflates that level's share."),
        ),
        _option(
            "drop_rows",
            "Drop the affected rows",
            (f"{n_missing:,} row(s) with no value for `{column}` were excluded "
             f"from the analysis."),
            TIMING_IMMEDIATE,
            ("A complete-case analysis for this column. This is an exclusion "
             "criterion and belongs in the participant flow — it changes who the "
             "study is about."),
        ),
    ]


def missingness_cards(df: pd.DataFrame,
                      columns: Optional[Sequence[str]] = None,
                      threshold: float = HIGH_MISSING_SHARE) -> List[Dict[str, Any]]:
    """One decision card per column with meaningful missingness.

    The card names the column, states the count and share, routes by dtype, and
    carries the decision sentence each option would write — so the transcript
    the user is agreeing to is visible before they agree to it.
    """
    if df is None or df.empty:
        return []
    cols = list(columns) if columns is not None else list(df.columns)
    cards: List[Dict[str, Any]] = []

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            continue
        n_rows = int(len(series))
        n_missing = int(series.isna().sum())
        if not n_missing or not n_rows:
            continue
        share = n_missing / n_rows
        if share < threshold:
            continue

        kind = _kind_of(series)
        if kind == "binary":
            options = _binary_options(str(col), n_missing, n_rows)
            question = (f"Is the missingness in `{col}` informative?")
            because = ("A binary variable that was not recorded is not the same "
                       "as one recorded as absent, and only you know which this "
                       "is.")
        elif kind == "numeric":
            options = _numeric_options(str(col), series, n_missing)
            question = f"How should the missing values in `{col}` be filled?"
            because = ("Every choice below changes the distribution the model "
                       "sees. The before/after is shown for each.")
        else:
            options = _categorical_options(str(col), series, n_missing)
            question = f"What should `{col}` say where nothing was recorded?"
            because = ("An explicit Missing level keeps the absence; the mode "
                       "replaces it with a value nobody recorded.")

        cards.append({
            "id": f"missing__{col}",
            "column": str(col),
            "dtype_route": kind,
            "n_missing": n_missing,
            "n_rows": n_rows,
            "share": share,
            "question": question,
            "because": because,
            "options": options,
            # THE ACTUAL ROWS (`DRIVE-008`). The panel stated a count, a share
            # and what each option would write into the transcript, and showed
            # none of the data — so a user was asked what a blank in this
            # column MEANS while looking at no blanks.
            #
            # Blank rows and present rows together, deliberately. A snippet of
            # only the blanks answers "how many" a second time; the question is
            # what distinguishes the rows where it is missing from the rows
            # where it is not, and that needs both.
            "snippet": _snippet(df, str(col), series),
            # Named here so the deferral affordance can state its destination
            # rather than saying "later" (GUIDED-008).
            "target_step": "preprocess",
            "target_step_label": "Preprocess",
        })

    cards.sort(key=lambda c: -c["share"])
    return cards


def imputation_preview(series: pd.Series, strategy: str) -> Optional[Dict[str, Any]]:
    """Before/after summary for one numeric imputation, on this column.

    Computed on the whole column for *display*. The transform itself runs on
    training folds inside the pipeline; this is a picture of what the choice
    does to the shape, not the fitted statistic.
    """
    values = pd.to_numeric(series, errors="coerce")
    present = values.dropna()
    if present.empty:
        return None
    if strategy == "impute_mean":
        fill = float(present.mean())
    elif strategy in ("impute_median", "indicator_and_impute"):
        fill = float(present.median())
    else:
        return None
    filled = values.fillna(fill)
    return {
        "strategy": strategy,
        "fill_value": fill,
        "n_filled": int(values.isna().sum()),
        "before": {"mean": float(present.mean()), "std": float(present.std(ddof=1)) if len(present) > 1 else 0.0,
                   "median": float(present.median()), "n": int(len(present))},
        "after": {"mean": float(filled.mean()), "std": float(filled.std(ddof=1)) if len(filled) > 1 else 0.0,
                  "median": float(filled.median()), "n": int(len(filled))},
        "note": ("Shown on the whole column. The value actually used is fitted "
                 "inside each training fold, so it will differ slightly per fold."),
    }
