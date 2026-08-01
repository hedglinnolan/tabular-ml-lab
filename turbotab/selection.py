"""turbotab.selection — feature selection, declared and deferred.

The sharpest case in the whole project, and the one clause §06 exists for.

*"Feature selection on the full dataset"* is a named leak in Kapoor &
Narayanan's taxonomy, and it is subtle in a way that makes it easy to ship: no
held-out row is copied anywhere, no value crosses the boundary, and yet **the
selected SET encodes test signal**. Choosing the top-20 features by mutual
information over the whole table means the identity of those twenty was decided
partly by the rows you promised not to look at. Every later number is then
conditioned on that choice.

So selection is never performed here. This module produces a **specification** —
what will be selected, by which method, to what size — and the per-model
pipeline fits it inside each training fold.

**Two levels, and Classic gets one of them.** `pages/04_Feature_Selection.py`
masks to training rows with `train_row_mask` and says so on screen, which is the
good example in this codebase and the reason this module exists at all. But it
then runs selection ONCE over all training rows and stores a consensus set, so
validation-fold signal is inside the selected set even though test-set signal is
not. Fold-local selection is strictly better, and saying which of the two is
meant matters more than the improvement:

    scope="train_rows"   test rows excluded  (Classic's behavior)
    scope="train_folds"  refitted per fold   (what this module declares)

Both are recorded explicitly, so a manuscript can state which one happened
rather than implying the stronger one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

TRAIN_ROWS = "train_rows"
TRAIN_FOLDS = "train_folds"


class SelectionRefusal(Exception):
    """Selection was asked to do something that would leak."""


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    # The methods-prose sentence, carrying the TIMING. `{n}` is the size.
    sentence: str
    # What the method needs to see. Every one of these is stateful by
    # construction — a ranking is a statement about the columns relative to
    # each other, across rows.
    why_stateful: str
    explainability_cost: str = "medium"


_ACROSS_ROWS = (
    "A ranking compares columns to each other across every row it is given, so "
    "the selected set is a property of the rows it saw. Fitted on the whole "
    "table it is a property of the held-out rows too.")

METHODS: Dict[str, Method] = {m.key: m for m in [
    Method("mutual_info", "Mutual information with the outcome",
           "The top {n} features by mutual information with `{target}` will be "
           "selected within each training fold.", _ACROSS_ROWS,
           explainability_cost="low"),
    Method("lasso", "LASSO (L1 penalty)",
           "Features surviving a LASSO penalty will be selected within each "
           "training fold, with the penalty tuned on that fold.", _ACROSS_ROWS,
           explainability_cost="low"),
    Method("rfe", "Recursive feature elimination",
           "Features will be eliminated recursively down to {n}, refitting "
           "within each training fold.", _ACROSS_ROWS,
           explainability_cost="medium"),
    Method("univariate", "Univariate association with the outcome",
           "The top {n} features by univariate association with `{target}` "
           "will be selected within each training fold.", _ACROSS_ROWS,
           explainability_cost="low"),
    Method("stability", "Stability selection over resamples",
           "Features selected in at least half of resamples will be kept, with "
           "the resampling done inside each training fold.", _ACROSS_ROWS,
           explainability_cost="high"),
]}


def method(key: str) -> Method:
    m = METHODS.get(key)
    if m is None:
        raise SelectionRefusal(
            f"'{key}' is not a selection method. Known: "
            f"{', '.join(sorted(METHODS))}.")
    return m


def declare(method_key: str, target: str, candidates: Sequence[str],
            n_features: Optional[int] = None,
            consensus_min_methods: Optional[int] = None,
            scope: str = TRAIN_FOLDS) -> Dict[str, Any]:
    """Record what will be selected. Never selects.

    `scope` is explicit and has no default that hides the weaker option:
    `train_folds` is what this module declares, `train_rows` exists so a
    project that inherits Classic's behavior can SAY so rather than imply the
    stronger claim.
    """
    m = method(method_key)
    if scope not in (TRAIN_ROWS, TRAIN_FOLDS):
        raise SelectionRefusal(
            f"scope must be {TRAIN_ROWS!r} or {TRAIN_FOLDS!r}; got {scope!r}. "
            f"There is no third option, and in particular there is no option "
            f"that fits on the whole table.")
    if not candidates:
        raise SelectionRefusal("Selection needs candidate features.")
    if target in candidates:
        raise SelectionRefusal(
            f"'{target}' is the outcome and cannot also be a candidate "
            f"feature: selecting the target predicts it perfectly.")
    if n_features is not None and n_features > len(candidates):
        raise SelectionRefusal(
            f"Asked for {n_features} of {len(candidates)} candidates.")

    sentence = m.sentence.format(n=n_features or len(candidates), target=target)
    if scope == TRAIN_ROWS:
        sentence = sentence.replace(
            "within each training fold",
            "once over the training rows (held-out rows excluded)")
    return {
        "method": method_key,
        "label": m.label,
        "target": target,
        "candidates": [str(c) for c in candidates],
        "n_features": n_features,
        "consensus_min_methods": consensus_min_methods,
        "scope": scope,
        "fit_on": ("training folds only" if scope == TRAIN_FOLDS
                   else "training rows only"),
        "sentence": sentence,
        "because": m.why_stateful,
        "explainability_cost": m.explainability_cost,
        # Never populated here. A spec that carried a selected set would be a
        # selection that already ran, which is the thing this module refuses.
        "selected": None,
    }


def evidence(df: pd.DataFrame, target: str, candidates: Sequence[str],
             train_mask: Optional[pd.Series] = None,
             top: int = 12) -> Dict[str, Any]:
    """The evidence a selection CHOICE is shown beside, on training rows only.

    `DESIGN_LANGUAGE.md` §09: a finding carries its evidence — a proposed
    interaction shows its correlation, not a bullet describing one. So the
    interface can show what the ranking would look like, and it must compute it
    where selection would: on training rows.

    This is emphatically NOT selection. It ranks, it does not choose, nothing
    is stored, and the returned scores are marked `preview_not_applied`. The
    distinction is the same one clause §06 draws for a deferred transform's
    preview.
    """
    if target not in df.columns:
        raise SelectionRefusal(f"No column named '{target}' in this table.")
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise SelectionRefusal("None of those candidates are in this table.")

    frame = df if train_mask is None else df.loc[train_mask.reindex(df.index, fill_value=False)]
    n_seen = int(len(frame))
    # THE CLAIM IS ABOUT THE ROWS, NOT ABOUT THE ARGUMENT. `scope` used to read
    # `train_rows` whenever a mask object was passed, so a mask that excludes
    # nothing — which is what `project.training_mask` correctly returns before
    # the seal — would have made this sentence say *ranked on training rows
    # only* about a ranking that saw every row. A scope derived from what was
    # actually withheld cannot say that.
    n_held_out = int(len(df)) - n_seen
    y = frame[target]
    scored: List[Dict[str, Any]] = []
    for c in cols:
        s = frame[c]
        if not pd.api.types.is_numeric_dtype(s) or not pd.api.types.is_numeric_dtype(y):
            scored.append({"feature": str(c), "score": None,
                           "measure": "not numeric — not ranked here"})
            continue
        try:
            r = float(s.corr(y))
        except Exception:
            r = float("nan")
        scored.append({"feature": str(c),
                       "score": None if pd.isna(r) else round(abs(r), 4),
                       "signed": None if pd.isna(r) else round(r, 4),
                       "measure": "absolute correlation with the outcome"})
    scored.sort(key=lambda d: (d["score"] is None, -(d["score"] or 0)))
    return {
        "preview_not_applied": True,
        "n_rows_seen": n_seen,
        "n_rows_withheld": n_held_out,
        "scope": TRAIN_ROWS if n_held_out else "all rows",
        "ranked": scored[:top],
        "note": ("Ranked on training rows only, and not applied. What is "
                 "actually selected is refitted inside each training fold, so "
                 "this ordering is indicative rather than the answer."
                 if n_held_out else
                 "Nothing was withheld from this ranking, so it saw every row "
                 "in the table. Treat it as exploratory."),
    }
