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

from typing import Any, Dict, List, Mapping, Optional, Sequence

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
# A fourth, and it is the honest one for `leave`. Nothing is applied and
# nothing is scheduled, which the card used to render as `in_pipeline` — a
# claim that something would be fitted in the fold when the whole point of the
# option is that nothing is. `turbotab/missingness.py`'s own `_BECAUSE` for
# `leave` says it: *nothing is computed and nothing is deferred.*
TIMING_RECORDED_ONLY = "recorded_only"

_TIMING_PROSE = {
    TIMING_IMMEDIATE: "applied to the working table now",
    TIMING_IN_PIPELINE: "fitted inside each model's pipeline, on training folds only",
    TIMING_MIXED: ("the indicator is added now; the value under it is fitted "
                   "inside each model's pipeline, on training folds only"),
    TIMING_RECORDED_ONLY: ("recorded and nothing else — no fill is applied now "
                           "and none is scheduled"),
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


#: WHAT EACH STRATEGY COSTS, in the card's register. The strategy's own
#: `because` is clause §06's litmus answer — *why it is fitted where it is* —
#: and this is the other question a user is asking: *what does choosing it do
#: to my study?* Both travel on the card.
#:
#: Callables take `(column, n_missing, series)` where the sentence depends on
#: the data, which is `DESIGN_LANGUAGE.md` §09's rule that a finding carries its
#: evidence.
def _skew_note(column: str, n_missing: int, series: pd.Series) -> str:
    present = pd.to_numeric(series, errors="coerce").dropna()
    skew = float(present.skew()) if len(present) > 2 else 0.0
    if abs(skew) > 1.0:
        return (f"`{column}` is skewed (skew = {skew:.2f}), so the mean sits "
                "away from the bulk of the data and the median is the more "
                "representative filler.")
    return ("Robust to the tails. Shrinks the variance of the column, which "
            "biases any standard error computed from it toward zero.")


def _mode_note(column: str, n_missing: int, series: pd.Series) -> str:
    mode = series.dropna().mode()
    label = str(mode.iloc[0]) if len(mode) else "the most frequent level"
    return (f"Asserts `{label}` for {n_missing:,} participant(s) nobody "
            "recorded, and inflates that level's share.")


_CONSEQUENCE = {
    "explicit_category": lambda c, n, s: (
        "Keeps the absence as a fact about the participant, and lets the model "
        "estimate whether it carries signal. Adds one level to the encoding."),
    "indicator": lambda c, n, s: (
        f"Adds one column. The model can learn from *whether* {c} was recorded, "
        "which is the right reading when the absence has a cause — not asked, "
        "not applicable, refused. The value itself stays blank, which gradient "
        "boosting reads natively and a linear model cannot."),
    "indicator_and_impute": lambda c, n, s: (
        f"Adds one column and keeps the fact that {n:,} value(s) were absent "
        "available to the model, while still giving every model a number to "
        "work with."),
    "impute_median": _skew_note,
    "impute_mean": lambda c, n, s: (
        "Preserves the column mean exactly and nothing else. On a skewed "
        "column it places every filled value where few real ones are."),
    "impute_mode": _mode_note,
    "impute_mice": lambda c, n, s: (
        "Uses the correlations in the data rather than one number. Costs run "
        "time and makes the imputation itself a model you have to describe."),
    "leave": lambda c, n, s: (
        "Gradient boosting handles this natively. Linear and neural models do "
        "not — the app fills the blank for those and says so on the run, per "
        "model."),
}

#: Which option the coach would take, per branch. Ranking carries the judgment;
#: absence never does (`PRODUCT_VISION.md`, the shelf is never shortened).
_RECOMMENDED = {"categorical": "explicit_category", "numeric": "indicator"}


def _mechanism_question(column: str) -> Dict[str, Any]:
    """§07's fork, in the Explore door's own card.

    **The same question and the same copy Preprocess asks**, from the same
    module, because the two doors gating one constitutional decision
    differently is what `GUIDED-091` was. The adjudicator's ruling: the Explore
    card is the noticing, and a noticing may not answer a constitutional
    question on the user's behalf in order to offer a shortcut past the step
    that asks it.
    """
    from turbotab import missingness as _miss

    return {
        "question": _miss.MECHANISM_QUESTION.format(column=column),
        "why": _miss.MECHANISM_WHY,
        "consumer": _miss.MECHANISM_CONSUMER,
        "options": list(_miss.MECHANISM_OPTIONS),
        "values": list(_miss.MECHANISMS),
    }


def _timing_of(spec: Dict[str, Any]) -> str:
    if spec["defers"] and spec["executes_now"]:
        return TIMING_MIXED
    if spec["defers"]:
        return TIMING_IN_PIPELINE
    if spec["executes_now"]:
        return TIMING_IMMEDIATE
    return TIMING_RECORDED_ONLY


def _options_for(column: str, branch: str, series: pd.Series,
                 n_missing: int,
                 mechanism: Optional[str] = None) -> List[Dict[str, Any]]:
    """The strategies this branch offers, FROM THE ONE TABLE THAT DECIDES.

    `GUIDED-090`. This used to be three hand-written lists, and they disagreed
    with `turbotab.missingness.STRATEGIES_BY_BRANCH` in both directions: `leave`
    was on the numeric branch's list and never on the numeric card, so a user in
    Explore could not choose to leave the blanks alone on exactly the column
    where the absence carries signal; and `impute_mode` was on the binary card
    while the record refuses it for a numeric column, so the card offered a
    route the record would reject.

    That is the product owner's own ruling at a surface nobody had compared:
    **judgment renders as ranking, never as absence.** `GUIDED-086` made the
    CHECK read this table; this is the half that OFFERS.

    The decision sentence is `missingness.sentence_for`, so the sentence on the
    card and the sentence on the record are one object rather than two that
    used to contradict each other (`GUIDED-098`).

    **AND THE LIST IS ORDERED, WITH THE APP'S CONCERN ON THE OPTIONS IT IS
    ABOUT** (`GUIDED-163`). `turbotab.missingness` already refused a median
    fill over an informatively-missing column with a typed 409; what it did not
    do was say so *before* the click. On the drive's `meds_hbp` — observed
    `{True: 5527, False: 770}`, 15,552 blanks, median 1 — the fill that puts
    every person of unknown medication status on blood pressure medication was
    offered as an equal peer of the two routes that keep the signal.

    The shelf is not shortened: every strategy the branch permits is still
    here, and `blocked` never removes one. `missingness.shelf_rank` decides the
    order and `missingness.concern` writes the sentence, so the rule lives with
    `blocks` and this composes what it says. The concern is folded into the
    option's `consequence` as well as carried structurally, because
    `consequence` is the field a person reads and a concern nothing renders is
    trap #6.
    """
    from turbotab import missingness as _miss

    out: List[Dict[str, Any]] = []
    for key in _miss.STRATEGIES_BY_BRANCH[branch]:
        spec = _miss.strategy(key)
        consequence = _CONSEQUENCE.get(key)
        option = _option(
            key, spec["label"].replace("`", ""),
            _miss.sentence_for(column, branch, key),
            _timing_of(spec),
            consequence(column, n_missing, series) if consequence else
            spec["because"],
            recommended=(key == _RECOMMENDED.get(branch)),
        )
        option.update(_miss.reading(column, key, mechanism, n_missing))
        if option["concern"]:
            option["consequence"] = (option["consequence"] + " "
                                     + option["concern"])
        out.append(option)
    # STABLE, so the branch table still decides everything the constitution has
    # no opinion about. `sorted` on one key preserves the order of equals, and
    # that order is `STRATEGIES_BY_BRANCH` — which is where the coach's pick
    # already sits first.
    out.sort(key=lambda o: o["shelf_rank"])

    # THE ONE OFFER THAT IS NOT A STRATEGY, and it stays on the card.
    #
    # Clause §04: dropping every row with no value changes who the study is
    # about, so it is an eligibility criterion reported in participant flow —
    # and `declare` refuses it with exactly that reason. It is kept here on the
    # rule that *a gap that becomes routing is worth more than a transform*: a
    # user who reaches for it deserves the argument and somewhere to go, not an
    # absence. Marked so nothing can mistake it for part of the branch table.
    for key, reason in _miss.NOT_A_STRATEGY.items():
        option = _option(
            key, "Drop the affected rows",
            (f"{n_missing:,} row(s) with no value for `{column}` were excluded "
             f"from the analysis."),
            TIMING_IMMEDIATE, reason)
        option["is_strategy"] = False
        # THE CONSTITUTIONAL FIELDS, STATED RATHER THAN DERIVED, because the
        # derivation would say a false thing here. `blocks` answers a question
        # about MISSINGNESS STRATEGIES, and `drop_rows` is not one — so
        # `blocked_under('drop_rows')` is empty, and copying that onto the
        # option as `blocked: False` would assert the constitution permits a
        # route `declare` refuses under every mechanism. `None` is *this field
        # does not answer for this option*; the reason it is refused is clause
        # §04's and is already the whole of its `consequence`.
        option.update({"blocked": None, "blocked_under": [], "concern": None,
                       "shelf_rank": max(
                           [o["shelf_rank"] for o in out] or [0]) + 1})
        out.append(option)
    for option in out:
        option.setdefault("is_strategy", True)
    return out


def missingness_cards(df: pd.DataFrame,
                      columns: Optional[Sequence[str]] = None,
                      threshold: float = HIGH_MISSING_SHARE,
                      mechanisms: Optional[Mapping[str, str]] = None,
                      provenance: Optional[Mapping[str, Any]] = None
                      ) -> List[Dict[str, Any]]:
    """One decision card per column with meaningful missingness.

    The card names the column, states the count and share, routes by dtype, and
    carries the decision sentence each option would write — so the transcript
    the user is agreeing to is visible before they agree to it.

    `mechanisms` is the answer ALREADY ON THE RECORD for a column, keyed by
    column name, and it is the difference between the app saying *"this is
    refused"* and *"this would be refused if you answered yes"*. Absent, and
    absent per column, is the normal case and is `None` — *not yet asked* —
    which is `GUIDED-091`'s rule and is why this is not defaulted to
    `not_sure`. Nothing is inferred from the data here; the only source is a
    `route_missingness` decision the user already made.
    """
    if df is None or df.empty:
        return []
    answered = {str(k): v for k, v in (mechanisms or {}).items()}
    made = {str(k): v for k, v in (provenance or {}).items()}
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
        # TWO ROUTINGS, AND THEY ARE DIFFERENT QUESTIONS.
        #
        # `dtype_route` is PRESENTATION — how to word the question, whether a
        # histogram belongs beside it — and it has three values because a 0/1
        # column reads differently from a continuous one.
        #
        # `branch` is what the RECORD will use, and it is the dtype, because
        # that is what decides whether a strategy would change what the column
        # IS (`GUIDED-086`). They disagreed on a 0/1 numeric column: the card
        # offered `impute_mode` and `declare` refuses it there, so the Explore
        # door offered a route the record would reject. One table now decides
        # the offer, keyed on the branch the record keys on.
        branch = ("numeric" if pd.api.types.is_numeric_dtype(series)
                  else "categorical")
        mechanism = answered.get(str(col))
        options = _options_for(str(col), branch, series, n_missing, mechanism)
        if kind == "binary":
            question = (f"Is the missingness in `{col}` informative?")
            because = ("A binary variable that was not recorded is not the same "
                       "as one recorded as absent, and only you know which this "
                       "is.")
        elif kind == "numeric":
            question = f"How should the missing values in `{col}` be filled?"
            because = ("Every choice below changes the distribution the model "
                       "sees. The before/after is shown for each.")
        else:
            question = f"What should `{col}` say where nothing was recorded?"
            because = ("An explicit Missing level keeps the absence; the mode "
                       "replaces it with a value nobody recorded.")

        cards.append({
            "id": f"missing__{col}",
            "column": str(col),
            "dtype_route": kind,
            "branch": branch,
            "n_missing": n_missing,
            "n_rows": n_rows,
            "share": share,
            "question": question,
            "because": because,
            # §07'S FORK, CARRIED BY THIS DOOR TOO (`GUIDED-091`). The card had
            # no `mechanism` field at all, so the page's `c.mechanism ||
            # "not_sure"` was unconditional and every column routed from here
            # recorded an answer the user was never asked for — which made
            # clause §07's blocker unreachable from this door by any user on
            # any column. `None` is *not yet asked*, and the strategies are not
            # offered until it is answered, exactly as Preprocess does it.
            #
            # It is `None` UNLESS THE USER ALREADY ANSWERED IT — a column that
            # carries a `route_missingness` decision has the answer on the
            # record, and re-asking a question the transcript already holds is
            # the app forgetting what it was told (`GUIDED-163`). Still never
            # inferred: `mechanisms` comes from decisions and from nowhere else.
            "mechanism": mechanism,
            "mechanism_question": _mechanism_question(str(col)),
            # WHERE THESE BLANKS CAME FROM (`GUIDED-166`), or `None` where the
            # app made none of them. Composed by `turbotab.missingness` and
            # passed in, because the Preprocess survey says the same thing from
            # the same reading and two doors answering one question differently
            # is what `GUIDED-091` was.
            "provenance": made.get(str(col)),
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
