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
pipeline fits it as a step inside the estimator, so it is refitted wherever
that pipeline is refitted. **How many times that is, is a property of the
door, not of this module** (`AUDIT-027`): the Guided door fits each model once,
on the training partition, so there is one fold and `FITTED_SCOPE` says so.
Every sentence here about where selection lands is composed from that constant
rather than from the stronger claim.

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

#: The clause each scope contributes to a sentence, and the machine-readable
#: half beside it. One table rather than a `str.replace` in `declare` and a
#: hand-written sentence in `evidence` — `AUDIT-027` was those two drifting
#: apart, and `missingness._SCOPE_PHRASE` is the same table one module over.
_SCOPE_PHRASE = {
    TRAIN_ROWS: "once over the training rows (held-out rows excluded)",
    TRAIN_FOLDS: "within each training fold",
}
_FIT_ON = {
    TRAIN_ROWS: "training rows only",
    TRAIN_FOLDS: "training folds only",
}

#: WHAT THIS DOOR ACTUALLY FITS, whatever the record requested (`AUDIT-027`).
#:
#: `training.train` fits each model ONCE, on the training partition, and scores
#: on the sealed rows. There is one fold. `pipeline_plan` has recorded exactly
#: this as `scope_fitted` since `GUIDED-095` — it reads this name — and states a
#: `Divergence` when a spec asked for `TRAIN_FOLDS` and got this instead. The
#: constant is here so a sentence about where selection is fitted cannot be
#: written from memory in one module while another one records the truth.
#:
#: `TRAIN_FOLDS` becomes the value the day `GUIDED-103`'s resampling policy
#: lands, and then it moves in one place.
FITTED_SCOPE = TRAIN_ROWS


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


#: `evidence` takes a parameter named `method` — the name the query string and
#: the recorded spec both already use — so the lookup gets a name the parameter
#: cannot shadow.
_method_or_refuse = method


# ─────────────────────────────────────────────────────────────────────────────
# What each method can honestly PREVIEW, which is not the same as what it ranks
# ─────────────────────────────────────────────────────────────────────────────
#
# `GUIDED-177`. The evidence table used to compute `abs(pearson)` whatever the
# user had chosen, and head the table *absolute correlation with the outcome*
# under a sentence reading *the top 5 features by mutual information*. The
# evidence did not vary with the choice it was evidence for.
#
# The line that decides whether a preview is possible is whether the method's
# score is a property of ONE COLUMN against the outcome, or a property of a
# FITTED SET:
#
#   mutual_info · univariate   a per-column statistic. Previewable, and the
#                              preview is the real quantity the fold will rank
#                              by — `pipeline_plan._selector` uses the same two
#                              sklearn scorers.
#   lasso · rfe · stability    ranked by what survives a fit over all
#                              candidates at once. RFE does expose `ranking_`
#                              and stability a survival frequency, but getting
#                              either means RUNNING THE SELECTOR — which is a
#                              selection, and this panel ranks and does not
#                              choose.
#
# THE SHELF IS NOT SHORTENED BY THIS. All five methods stay offered and stay
# fitted inside the model's own pipeline; the three that cannot be previewed
# say what was not computed instead of borrowing a number from a different
# method.
_NO_PER_FEATURE_SCORE: Dict[str, str] = {
    "lasso": ("LASSO ranks nothing column by column: it fits one penalized "
              "model over all the candidates together and keeps the "
              "coefficients that survive."),
    "rfe": ("Recursive elimination ranks nothing column by column: it drops "
            "the weakest feature of the current fit and refits, repeatedly."),
    "stability": ("Stability selection's score is how often a column survives "
                  "a resample, and producing it means running the selector."),
}

#: Fewer complete pairs than this and there is no statistic, only a number.
#: `pandas.Series.corr` returns `nan` below 2 and the sklearn scorers raise, so
#: the floor is stated here rather than left to whichever one was called.
MIN_PAIRS = 3

#: Fixed, and on purpose. `mutual_info_classif`/`_regression` add noise to break
#: ties in the nearest-neighbour estimate, so an unseeded preview returns
#: different numbers each time the user presses the button — which reads as the
#: data changing. `pipeline_plan._selector` seeds for the same reason and takes
#: the run's seed; a preview has no run, so it gets a constant.
PREVIEW_SEED = 0


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
        sentence = sentence.replace(_SCOPE_PHRASE[TRAIN_FOLDS],
                                    _SCOPE_PHRASE[TRAIN_ROWS])
    return {
        "method": method_key,
        "label": m.label,
        "target": target,
        "candidates": [str(c) for c in candidates],
        "n_features": n_features,
        "consensus_min_methods": consensus_min_methods,
        "scope": scope,
        "fit_on": _FIT_ON[scope],
        "sentence": sentence,
        "because": m.why_stateful,
        "explainability_cost": m.explainability_cost,
        # Never populated here. A spec that carried a selected set would be a
        # selection that already ran, which is the thing this module refuses.
        "selected": None,
    }


def _preview_measure(m: Optional[Method],
                     task_type: Optional[str]) -> Dict[str, Any]:
    """Which statistic this method's preview may honestly show, and its name.

    Returns `kind` — the branch `evidence` scores by — together with the one
    string that heads the table and the sentence that says what was or was not
    computed. `kind == "none"` is the refusal: the columns are still listed,
    and no number is invented for them.
    """
    if m is None:
        return {"kind": "correlation",
                "measure": "absolute correlation with the outcome",
                "sentence": ("No method is chosen yet, so this is the plain "
                             "linear association between each column and the "
                             "outcome.")}
    if m.key in _NO_PER_FEATURE_SCORE:
        return {"kind": "none",
                "measure": (f"not computed here — {m.label} has no per-column "
                            f"score"),
                "sentence": (
                    f"{_NO_PER_FEATURE_SCORE[m.key]} So there is no column-by-"
                    f"column number to show you, and the columns below are "
                    f"your candidates in table order rather than a ranking. "
                    f"{m.label} is still offered and is still fitted "
                    f"{_SCOPE_PHRASE[FITTED_SCOPE]} inside the model's own "
                    f"pipeline — what is missing is the preview, not the "
                    f"method.")}
    if task_type not in ("classification", "regression"):
        return {"kind": "none",
                "measure": ("not computed — the outcome's task type is not "
                            "recorded"),
                "sentence": (
                    f"{m.label} scores a column against a classification "
                    f"outcome differently than against a regression one, and "
                    f"this project has not recorded which this is. Rather than "
                    f"pick one, nothing is scored.")}
    classify = task_type == "classification"
    if m.key == "mutual_info":
        return {"kind": "mutual_info",
                "measure": "mutual information with the outcome",
                "sentence": ("Scored with "
                             + ("mutual_info_classif" if classify
                                else "mutual_info_regression")
                             + ", the same estimator the selector will rank by.")}
    return {"kind": "f_test",
            "measure": ("ANOVA F statistic against the outcome" if classify
                        else "univariate linear F statistic against the "
                             "outcome"),
            "sentence": ("Scored with " + ("f_classif" if classify
                                           else "f_regression")
                         + ", the same estimator the selector will rank by.")}


def _sklearn_score(kind: str, x: pd.Series, y: pd.Series,
                   classify: bool, seed: int) -> Optional[float]:
    """One column against the outcome, on the rows where both are present.

    PAIRWISE-COMPLETE, matching what `Series.corr` already did on this path:
    the sklearn scorers raise on a `NaN` anywhere, and dropping every row that
    is blank in ANY candidate would silently rank a different set of columns on
    a different set of rows than the sentence above the table claims.
    """
    from sklearn.feature_selection import (f_classif, f_regression,
                                           mutual_info_classif,
                                           mutual_info_regression)

    X = x.to_frame()
    if kind == "mutual_info":
        fn = mutual_info_classif if classify else mutual_info_regression
        out = fn(X, y, random_state=seed)
    else:
        out = (f_classif if classify else f_regression)(X, y)[0]
    v = float(out[0])
    return None if pd.isna(v) else v


def evidence(df: pd.DataFrame, target: str, candidates: Sequence[str],
             train_mask: Optional[pd.Series] = None,
             top: int = 12, *,
             method: Optional[str] = None,
             task_type: Optional[str] = None,
             seed: int = PREVIEW_SEED) -> Dict[str, Any]:
    """The evidence a selection CHOICE is shown beside, on training rows only.

    `DESIGN_LANGUAGE.md` §09: a finding carries its evidence — a proposed
    interaction shows its correlation, not a bullet describing one. So the
    interface can show what the ranking would look like, and it must compute it
    where selection would: on training rows.

    This is emphatically NOT selection. It ranks, it does not choose, nothing
    is stored, and the returned scores are marked `preview_not_applied`. The
    distinction is the same one clause §06 draws for a deferred transform's
    preview.

    **`method` is the choice this is evidence FOR** (`GUIDED-177`). Without it
    the ranking was `abs(pearson)` for all five methods, headed *absolute
    correlation with the outcome* under a recorded sentence that said *by
    mutual information* — the evidence not varying with the choice it was
    evidence for, and closest to wrong exactly where the user chose to look
    past correlation. A method with no per-column score returns `score: None`
    and a `measure` naming what was not computed; it never borrows another
    method's number.
    """
    if target not in df.columns:
        raise SelectionRefusal(f"No column named '{target}' in this table.")
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise SelectionRefusal("None of those candidates are in this table.")
    m = _method_or_refuse(method) if method else None

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
    plan = _preview_measure(m, task_type)
    kind = plan["kind"]
    classify = task_type == "classification"
    # The correlation branch is the only one that needs a numeric OUTCOME; the
    # sklearn classification scorers take a string label happily. The old code
    # attributed the refusal to the FEATURE either way — every column of a
    # string-outcome project came back `not numeric — not ranked here`, which
    # is false of a column of floats.
    if kind == "correlation" and not pd.api.types.is_numeric_dtype(y):
        kind = "none"
        plan = {**plan,
                "measure": f"'{target}' is not numeric — correlation not computed",
                "sentence": (
                    f"A correlation needs two numbers and '{target}' holds "
                    f"labels, so none was computed. Choosing a selection "
                    f"method gives this table a statistic that can read a "
                    f"label outcome.")}
    scored: List[Dict[str, Any]] = []
    for c in cols:
        s = frame[c]
        if not pd.api.types.is_numeric_dtype(s):
            scored.append({"feature": str(c), "score": None,
                           "measure": "not numeric — not ranked here"})
            continue
        if kind == "none":
            scored.append({"feature": str(c), "score": None,
                           "measure": plan["measure"]})
            continue
        pair = pd.concat([s, y], axis=1).dropna()
        if len(pair) < MIN_PAIRS:
            scored.append({"feature": str(c), "score": None,
                           "measure": (f"only {len(pair)} rows have both "
                                       f"values — not computed")})
            continue
        xs, ys = pair.iloc[:, 0], pair.iloc[:, 1]
        if kind == "correlation":
            try:
                r = float(xs.corr(ys))
            except Exception:
                r = float("nan")
            scored.append({"feature": str(c),
                           "score": None if pd.isna(r) else round(abs(r), 4),
                           "signed": None if pd.isna(r) else round(r, 4),
                           "measure": plan["measure"]})
            continue
        try:
            v = _sklearn_score(kind, xs, ys, classify, seed)
        except Exception as exc:
            # NO NUMBER RATHER THAN A NUMBER FROM SOMEWHERE ELSE (trap 9). The
            # reason travels with the row so the table can show it.
            scored.append({"feature": str(c), "score": None,
                           "measure": (f"not computed — "
                                       f"{type(exc).__name__}")})
            continue
        scored.append({"feature": str(c),
                       "score": None if v is None else round(v, 4),
                       "measure": plan["measure"]})
    scored.sort(key=lambda d: (d["score"] is None, -(d["score"] or 0)))
    # `AUDIT-027`. THE PANEL SAID THE STRONGER THING THAN THE RECORD.
    #
    #   before: "Ranked on training rows only, and not applied. What is
    #            actually selected is refitted inside each training fold, so
    #            this ordering is indicative rather than the answer."
    #   after:  "…What is actually selected is fitted once over the training
    #            rows (held-out rows excluded) — this door fits each model one
    #            time, so there is a single fold — so this ordering is
    #            indicative rather than the answer."
    #
    # Same subject, weaker claim, true. *Refitted inside each training fold* is
    # a claim that selection sits inside a resampling loop, which §A5.5 is
    # precisely about; there is no loop here. `api` records
    # `scope=TRAIN_ROWS` for this door and `pipeline_plan` fits `FITTED_SCOPE`,
    # so the panel was the one surface still asserting the fold refit — and it
    # bypassed `declare`'s own rewrite, three functions up, that exists to stop
    # exactly this. The phrase now comes from the same table `declare` uses, so
    # the two cannot drift again.
    note = (f"Ranked on training rows only, and not applied. What is "
            f"actually selected is fitted {_SCOPE_PHRASE[FITTED_SCOPE]} — "
            f"this door fits each model one time, so there is a single fold "
            f"— so this ordering is indicative rather than the answer."
            if n_held_out else
            "Nothing was withheld from this ranking, so it saw every row "
            "in the table. Treat it as exploratory.")
    if kind == "none":
        # A sentence about an ordering, over a list that is not ordered, would
        # be the same class of false claim this finding is about.
        note = ("Nothing here is ranked. " + plan["sentence"] + " "
                + ("Nothing was withheld." if not n_held_out
                   else "Only training rows were read."))
    return {
        "preview_not_applied": True,
        "n_rows_seen": n_seen,
        "n_rows_withheld": n_held_out,
        "scope": TRAIN_ROWS if n_held_out else "all rows",
        # WHERE THE SELECTION ITSELF WILL BE FITTED, machine-readable beside
        # the prose (trap 7) and named apart from `scope` above, which is about
        # the rows THIS RANKING saw. `AUDIT-027`: the note asserted one of these
        # and the recorded spec stored the other, and nothing on the wire let a
        # reader see that they disagreed. These two are the record's own
        # vocabulary — `declare` returns the same pair of keys' values — so a
        # parity check is an equality rather than a substring search.
        "selection_scope": FITTED_SCOPE,
        "selection_fit_on": _FIT_ON[FITTED_SCOPE],
        # WHAT THIS IS EVIDENCE FOR, on the wire and not only in the prose
        # (trap 7). The page heads the table from `measure` rather than from
        # whichever row happened to sort first.
        "method": m.key if m else None,
        "method_label": m.label if m else None,
        "measure": plan["measure"],
        "is_ranked": kind != "none",
        "ranked": scored[:top],
        "note": note + " " + plan["sentence"] if kind != "none" else note,
    }
