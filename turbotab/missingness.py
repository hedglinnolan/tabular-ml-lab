"""turbotab.missingness — clause §07: routing by dtype **and** mechanism.

> Prediction is not inference, and the distinction is load-bearing: the
> missing-indicator method discouraged for causal estimation is defensible and
> often helpful for prediction under informative missingness.

That sentence is why this module cannot import a causal-inference default and
call the clause satisfied. The two branches fail in different ways and are
therefore separate objects here:

**Binary / categorical** — ask FIRST whether the missingness is informative
(*"could a blank here mean something?"*); in EHR data it usually is. Default to
an explicit `Missing` category or an indicator, which preserve the signal.
Imputing an informatively-missing field is a **blocker with typed
acknowledgment**, and the **stability assumption** — that missingness means the
same thing at deployment — is recorded as a methods assumption, because it may
not hold across sites.

**Numeric** — single vs multiple imputation and the strategy; fit **inside the
fold**; and **never place the outcome in the imputation model**, which is a
blocker in any configuration.

**Almost nothing here executes**, and that is clause §06 rather than a
limitation. Every routine below is stateful by the litmus — a median, a mode, a
MICE model and a frequency table are all statements about the column's
distribution — so this module produces **declarations** that per-model pipelines
fit inside training folds. The one exception is the explicit-`Missing` category
for a categorical column, which is row-local (a blank becomes a literal token,
using nothing but that row's own cell) and is marked so.

The consequence for the interface is the awkward part and is dealt with in
`plan_receipt`: the user answers several questions, the table does not change,
and the step still has to be believable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# The mechanism answers. `NOT_SURE` is first-class for the same reason it is on
# the grain question: the app cannot know, and a forced guess is worse than a
# recorded uncertainty.
INFORMATIVE = "informative"
NOT_INFORMATIVE = "not_informative"
NOT_SURE = "not_sure"
MECHANISMS = (INFORMATIVE, NOT_INFORMATIVE, NOT_SURE)

# What may be done about it, per branch.
EXPLICIT_CATEGORY = "explicit_category"      # row-local
INDICATOR = "indicator"                      # row-local
IMPUTE_MODE = "impute_mode"                  # stateful
IMPUTE_MEDIAN = "impute_median"              # stateful
IMPUTE_MEAN = "impute_mean"                  # stateful
IMPUTE_MICE = "impute_mice"                  # stateful
LEAVE = "leave"                              # nothing, recorded

ROW_LOCAL_STRATEGIES = frozenset({EXPLICIT_CATEGORY, INDICATOR, LEAVE})


class MissingnessRefusal(Exception):
    """The step was asked for something clause §07 forbids."""


# ── the mechanism question, asked before the strategy ────────────────────────

MECHANISM_QUESTION = "Could a blank in `{column}` mean something?"

MECHANISM_WHY = (
    "In records collected during care, a missing value is often a decision "
    "rather than an accident — a test not ordered because the patient looked "
    "well is different from a test that was ordered and lost. If a blank "
    "carries information, filling it in throws that information away.")

MECHANISM_OPTIONS = (
    "Yes — a blank here means something",
    "No — these are accidents of collection",
    "I'm not sure",
)

MECHANISM_CONSUMER = (
    "The answer decides how this column is handled and what the methods section "
    "has to say. 'Yes' routes to an explicit Missing category or an indicator, "
    "which keep the signal, and makes plain imputation a blocked choice that "
    "needs a typed acknowledgment. It also records the stability assumption — "
    "that a blank will still mean the same thing wherever the model is "
    "deployed — as a stated limitation, because it may not hold across sites.")

# §07's own words, recorded when the mechanism is stated informative. Not a
# warning: an assumption, written where a methods section can quote it.
STABILITY_ASSUMPTION = (
    "This analysis assumes that a blank in `{column}` will mean the same thing "
    "wherever the model is used as it means here. That assumption is not "
    "checkable from this dataset — missingness patterns are a property of how "
    "a site collects data, and a model that reads a blank as a signal will read "
    "a differently-collected blank as the same signal.")

# The CONSEQUENCE. Plain imputation over a column the user has just said carries
# information is not a mistake the app can correct on their behalf — they may
# know something it does not — so it resolves or is attested, never a dead end.
INFORMATIVE_IMPUTATION_BLOCKER = (
    "You said a blank in `{column}` means something, and {strategy} would "
    "replace every one of those {n_missing:,} blanks with {filler}. The fact "
    "that the value was missing would no longer be in the data at all, and no "
    "model can recover it afterward.\n\n"
    "If that is what you want, say so and it is recorded as a stated "
    "limitation. If it is not, an explicit `Missing` category keeps the blank "
    "as its own answer and costs nothing.")

BLOCKER_EXITS = (
    {"id": "explicit_category", "kind": "resolve",
     "label": "Keep the blanks as their own category",
     "detail": "A blank becomes a literal `Missing` value, so the model can use "
               "it the way it uses any other level."},
    {"id": "attest", "kind": "attest",
     "label": "Fill them anyway — I know what these blanks are",
     "detail": "Recorded as a stated limitation: the missingness signal is "
               "removed deliberately, and the methods section says so."},
)

# The numeric branch's blocker, which is not a judgment call at all.
OUTCOME_IN_IMPUTATION_REFUSAL = (
    "The outcome `{target}` cannot be one of the columns the imputation model "
    "reads. An imputer fitted with the outcome in scope writes the outcome's "
    "own information into the feature columns, so every number scored "
    "afterwards is scored against features that already encode the answer. "
    "There is no configuration in which this is acceptable, so it is not "
    "offered as a choice.")


def survey(df: pd.DataFrame, target: Optional[str] = None) -> List[Dict[str, Any]]:
    """Which columns have missing values, and which branch each one is on.

    Reports; decides nothing. The dtype half of clause §07 is mechanical and
    lives here; the mechanism half cannot be — it is the user's answer, and
    inferring it is the same error the grain question exists to prevent.
    """
    out: List[Dict[str, Any]] = []
    for col in df.columns:
        if target is not None and str(col) == str(target):
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):        # duplicated label
            continue
        n_missing = int(s.isna().sum())
        if not n_missing:
            continue
        numeric = bool(pd.api.types.is_numeric_dtype(s))
        out.append({
            "column": str(col),
            "branch": "numeric" if numeric else "categorical",
            "n_missing": n_missing,
            "n_rows": int(len(s)),
            "fraction": round(n_missing / max(1, len(s)), 4),
            # Asked, never inferred — so this is None until the user answers.
            "mechanism": None,
            "strategies": list(NUMERIC_STRATEGIES if numeric
                               else CATEGORICAL_STRATEGIES),
        })
    out.sort(key=lambda d: (-d["fraction"], d["column"]))
    return out


CATEGORICAL_STRATEGIES = (EXPLICIT_CATEGORY, INDICATOR, IMPUTE_MODE, LEAVE)
NUMERIC_STRATEGIES = (INDICATOR, IMPUTE_MEDIAN, IMPUTE_MEAN, IMPUTE_MICE, LEAVE)

_LABELS = {
    EXPLICIT_CATEGORY: "Keep blanks as an explicit `Missing` category",
    INDICATOR: "Add a was-it-missing column and leave the value blank",
    IMPUTE_MODE: "Fill with the most common value",
    IMPUTE_MEDIAN: "Fill with the median",
    IMPUTE_MEAN: "Fill with the mean",
    IMPUTE_MICE: "Fill by modeling it from the other columns (MICE)",
    LEAVE: "Leave it alone for now",
}

_FILLERS = {
    IMPUTE_MODE: "the column's most common value",
    IMPUTE_MEDIAN: "the column's median",
    IMPUTE_MEAN: "the column's mean",
    IMPUTE_MICE: "a value modeled from the other columns",
}

# Why each strategy is where it is on clause §06's litmus. Held as data so the
# interface can show the reasoning rather than assert the classification, the
# same way the transform catalogue does.
_BECAUSE = {
    EXPLICIT_CATEGORY:
        "Row-local: a blank becomes a literal `Missing` token using nothing but "
        "that row's own cell, so it can execute now.",
    INDICATOR:
        "Row-local: the new column is 1 where this row's value is blank and 0 "
        "where it is not. Nothing about any other row is consulted.",
    IMPUTE_MODE:
        "Stateful: the most common value is a fact about the whole column, so "
        "computing it over the full table would compute it over the held-out "
        "rows too.",
    IMPUTE_MEDIAN:
        "Stateful: the median is a fact about the whole column. Fitted inside "
        "each training fold, never over the sealed rows.",
    IMPUTE_MEAN:
        "Stateful: the mean is a fact about the whole column, and a more "
        "fragile one than the median — one extreme value moves it.",
    IMPUTE_MICE:
        "Stateful, and the most so: MICE fits a model per column against the "
        "others, so it learns the joint distribution of the training rows.",
    LEAVE:
        "Nothing is computed and nothing is deferred. Recorded so that "
        "'decided to leave it' and 'never looked at it' are different states.",
}


def strategy(key: str) -> Dict[str, Any]:
    if key not in _LABELS:
        raise MissingnessRefusal(
            f"'{key}' is not a missingness strategy. Known: "
            f"{', '.join(sorted(_LABELS))}.")
    return {"key": key, "label": _LABELS[key], "because": _BECAUSE[key],
            "defers": key not in ROW_LOCAL_STRATEGIES}


def blocks(mechanism: Optional[str], strategy_key: str) -> bool:
    """Is this pairing the CONSEQUENCE clause §07 names?

    Only the informative branch blocks, and only for a strategy that destroys
    the signal. `NOT_SURE` deliberately does not block: the user has said they
    do not know, and turning an admission of uncertainty into a wall teaches
    people to stop admitting it.
    """
    return mechanism == INFORMATIVE and strategy_key in _FILLERS


def blocker(column: str, mechanism: Optional[str], strategy_key: str,
            n_missing: int) -> Optional[Dict[str, Any]]:
    """The interruption, with both terminal exits attached.

    `DESIGN_LANGUAGE.md` §09: a CONSEQUENCE resolves or is attested, never a
    dead end. Both exits travel with the refusal so an interface cannot render
    the interruption without also rendering its way out.
    """
    if not blocks(mechanism, strategy_key):
        return None
    return {
        "column": str(column),
        "kind": "blocker",
        "strategy": strategy_key,
        "message": INFORMATIVE_IMPUTATION_BLOCKER.format(
            column=column, strategy=_LABELS[strategy_key].lower(),
            n_missing=int(n_missing), filler=_FILLERS[strategy_key]),
        "exits": [dict(e) for e in BLOCKER_EXITS],
        "acknowledgment_kind": "typed",
    }


def declare(column: str, branch: str, mechanism: str, strategy_key: str,
            target: Optional[str] = None,
            uses_columns: Optional[Sequence[str]] = None,
            acknowledged: bool = False,
            n_missing: int = 0) -> Dict[str, Any]:
    """Record how one column's missingness will be handled. Executes nothing.

    Refuses three things, each for a different reason:

    * an unknown mechanism or strategy — a typo would otherwise become a silent
      default;
    * the outcome inside a MICE scope — clause §07's hard blocker, and the one
      that is not a judgment call, so it is refused rather than offered;
    * an informative-missingness imputation with no acknowledgment — the
      CONSEQUENCE, which resolves or is attested.
    """
    if mechanism not in MECHANISMS:
        raise MissingnessRefusal(
            f"{mechanism!r} is not one of {list(MECHANISMS)}. The mechanism is "
            f"asked, never inferred — `not_sure` is a real answer.")
    spec = strategy(strategy_key)

    scope = [str(c) for c in (uses_columns or [])]
    if strategy_key == IMPUTE_MICE and target and str(target) in scope:
        raise MissingnessRefusal(
            OUTCOME_IN_IMPUTATION_REFUSAL.format(target=target))

    if blocks(mechanism, strategy_key) and not acknowledged:
        raise MissingnessRefusal(
            blocker(column, mechanism, strategy_key, n_missing)["message"])

    record: Dict[str, Any] = {
        "column": str(column),
        "branch": branch,
        "mechanism": mechanism,
        "strategy": strategy_key,
        "label": spec["label"],
        "because": spec["because"],
        "defers": spec["defers"],
        "fit_on": ("training folds only" if spec["defers"] else "row-local, applied now"),
        "uses_columns": scope or None,
        "acknowledged_signal_loss": bool(blocks(mechanism, strategy_key) and acknowledged),
        "sentence": _sentence(column, mechanism, strategy_key, spec),
    }
    if mechanism == INFORMATIVE:
        # §07: recorded as a methods ASSUMPTION rather than a warning, because a
        # warning is something a user dismisses and an assumption is something a
        # manuscript carries.
        record["assumption"] = STABILITY_ASSUMPTION.format(column=column)
    return record


def _sentence(column: str, mechanism: str, key: str, spec: Dict[str, Any]) -> str:
    """The methods-prose line, carrying the TIMING for anything deferred."""
    if key == LEAVE:
        return (f"Missing values in `{column}` are left as they are; no "
                f"imputation is applied and none is scheduled.")
    if key == EXPLICIT_CATEGORY:
        return (f"Missing values in `{column}` are kept as an explicit "
                f"`Missing` category rather than filled.")
    if key == INDICATOR:
        return (f"A was-it-missing indicator is added for `{column}`; the "
                f"underlying value is left blank.")
    where = " within each training fold" if spec["defers"] else ""
    return (f"Missing values in `{column}` will be filled using "
            f"{_LABELS[key].lower().replace('fill with ', '').replace('fill by ', '')}"
            f"{where}.")


def plan_receipt(declared: Sequence[Dict[str, Any]],
                 n_columns_with_missing: int) -> Dict[str, Any]:
    """What the user reads at the end of a step in which nothing visibly changed.

    **The hard part of this step.** Almost every preprocessing transform is
    stateful by clause §06's litmus, so the output is a set of recorded
    decisions that fire inside training folds — the user answers questions, the
    table does not change, and the step still has to be believable.

    The Features step's settle-sentence is the pattern: name the counts in each
    category, and name where the deferred ones will happen. What this adds is
    that the counts can legitimately be `0 applied now`, so the sentence has to
    be honest about a zero rather than embarrassed by it.
    """
    now = [d for d in declared if not d["defers"] and d["strategy"] != LEAVE]
    later = [d for d in declared if d["defers"]]
    left = [d for d in declared if d["strategy"] == LEAVE]
    attested = [d for d in declared if d.get("acknowledged_signal_loss")]
    assumptions = [d["assumption"] for d in declared if d.get("assumption")]

    parts: List[str] = []
    if now:
        parts.append(f"{len(now)} column(s) changed now")
    parts.append(f"{len(later)} recorded to be fitted inside the training folds")
    if left:
        parts.append(f"{len(left)} deliberately left alone")

    unanswered = max(0, n_columns_with_missing - len(declared))
    return {
        "n_applied_now": len(now),
        "n_deferred": len(later),
        "n_left": len(left),
        "n_unanswered": unanswered,
        "assumptions": assumptions,
        "n_attested": len(attested),
        "headline": "Missingness settled: " + ", ".join(parts) + ".",
        # THE SENTENCE THAT MAKES A ZERO BELIEVABLE. Stated plainly rather than
        # hidden, because a step that says nothing after a user answered six
        # questions reads as a step that did nothing.
        "why_nothing_changed": (
            "Your table looks the same because it is the same. Filling a blank "
            "with a median means computing that median, and computing it over "
            "every row would compute it over the held-out rows too — so the "
            "decision is recorded now and the arithmetic happens inside each "
            "training fold, where it can only see training data. What you just "
            "did is the part that cannot be automated; what is left is "
            "bookkeeping the pipeline does on its own."
            if later else
            "Nothing was deferred, so nothing is waiting: every answer here "
            "either changed the table or deliberately left it alone."),
        "outstanding": (
            f"{unanswered} column(s) with missing values have not been "
            f"answered yet." if unanswered else ""),
    }
