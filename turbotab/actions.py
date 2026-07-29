"""turbotab.actions — the engine's suggestions, classified into things to do.

`GUIDED-031`. The product owner drove the app, clicked *"Show me what this
means"* on a finding with no proposed repair, and landed in a branch that prints
`suggested_actions` as em-dash bullets and closes with *"the engine reports this
without proposing a repair."* Honest about the engine, and a dead end for the
user, who is standing at the exact moment a decision is due.

And `suggested_actions` was already **a list of options wearing prose**:

    — Consider winsorizing or capping
    — Tree models are robust to outliers
    — Investigate if outliers are errors or genuine

Three different decisions rendered as three paragraphs. That is
`DESIGN_LANGUAGE.md` §01.4 — *three attributes wearing a sentence costume* — the
critique that started this project, surviving the rewrite by moving branch.

## Two kinds, because they are not the same object

* **An OPERATION is something the app does to the data.** It carries a binding
  into a catalogue that already resolves it, and it previews. Winsorize, add a
  missingness indicator, impute, target-encode, reduce dimensions.
* **An EARMARK goes somewhere.** Either to a person — *"verify units and data
  entry"* is a thing a human does and the app must not pretend otherwise — or
  to a later step that owns the decision, like model choice. It lands in the
  record naming where it resurfaces, which is what makes it different from a
  dead end.

**The app says which is which.** Claiming it can verify data entry would be the
governing rule broken in the place built to honor it.

## Classified by PHRASE, not by finding type

The engine already carries the list; the work is classifying it. So the table
below is keyed on the engine's own sentences rather than hand-authored per
finding, which has two consequences worth the design:

* a phrase used by three warnings is classified once, and cannot drift between
  them;
* `test_every_suggested_action_the_engine_can_emit_is_classified` walks
  `ml/dataset_profile.py` for every literal it can produce and fails on one
  this table does not know — so a new suggestion cannot arrive as a bullet
  again.

## Timing is clause §06's, and the user never has to know it

A row-local operation runs now and posts a receipt. A distribution-learning one
is recorded and fitted inside the training folds, and its decision sentence
carries the timing as methods prose — *"extreme values in 15 features will be
winsorized at the 1st and 99th percentiles, computed within each training
fold."* A deferred option still previews, as a read-only simulation labeled
*preview, not applied*, which clause §06 explicitly permits and
`features.preview` already implements against training rows only.

## What an unbindable operation does

Some suggestions name an operation the app does not have — frequency encoding,
rare-category grouping, SMOTE. Those become earmarks naming the step that would
own them, and say so. **That is a legitimate outcome and it is not a dead end**,
because the earmark goes somewhere and the record says where.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Where an operation is bound. Each is a catalogue that already resolves the
# operation and states its own clause §06 scope, so nothing here re-decides
# timing — it reads it.
FEATURE = "feature"          # turbotab.features — row-local or deferred
RECIPE = "recipe"            # turbotab.recipes — per-model, always stateful
MISSINGNESS = "missingness"  # turbotab.missingness — dtype-routed
SELECTION = "selection"      # turbotab.selection — declared, fitted in folds

# Who an earmark is for. `you` is the one that matters: the app saying out loud
# that it cannot do this.
YOU = "you"


@dataclass(frozen=True)
class Operation:
    """Something the app does to the data, bound to a catalogue that resolves it."""
    key: str
    label: str
    catalogue: str
    binding: str                       # the catalogue's own key
    variant: Optional[str] = None
    kind: str = "operation"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "key": self.key, "label": self.label,
                "catalogue": self.catalogue, "binding": self.binding,
                "variant": self.variant}


@dataclass(frozen=True)
class Earmark:
    """Something that goes somewhere. A person, or a step that owns it."""
    key: str
    label: str
    target_step: str                   # a STEP, or `you`
    why: str
    kind: str = "earmark"

    @property
    def is_for_a_person(self) -> bool:
        return self.target_step == YOU

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "key": self.key, "label": self.label,
                "target_step": self.target_step, "why": self.why,
                "is_for_a_person": self.is_for_a_person}


def _norm(phrase: str) -> str:
    """One suggestion, in the form the table is keyed on.

    Case, punctuation and parentheticals only — *"Use regularized linear models
    (Ridge, Lasso)"* and *"Use regularized models (Ridge, Lasso, ElasticNet)"*
    are the same suggestion with different examples, and a table that
    distinguished them would classify the same decision twice and let the two
    drift.
    """
    text = re.sub(r"\([^)]*\)", " ", str(phrase)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# The table
#
# Keyed on the engine's own sentences. Every entry is a judgment and each one is
# stated: an OPERATION claims the app can do this and preview it, and an EARMARK
# claims it cannot and names where the decision lives instead.
# ─────────────────────────────────────────────────────────────────────────────

_TABLE: Dict[str, Any] = {}


def _op(phrase: str, key: str, label: str, catalogue: str, binding: str,
        variant: Optional[str] = None) -> None:
    _TABLE[_norm(phrase)] = Operation(key=key, label=label, catalogue=catalogue,
                                      binding=binding, variant=variant)


def _mark(phrase: str, key: str, label: str, target_step: str, why: str) -> None:
    _TABLE[_norm(phrase)] = Earmark(key=key, label=label,
                                    target_step=target_step, why=why)


# ── operations: the app does this, and shows you what it does ────────────────

_op("Consider winsorizing or capping", "winsorize",
    "Winsorize the extreme values", RECIPE, "outliers", "winsorize")
_op("Consider adding missingness indicators", "missing_indicator",
    "Add a was-it-missing indicator", MISSINGNESS, "indicator")
_op("Use imputation (mean/median for simple, KNN/iterative for better)",
    "impute", "Fill the blanks", MISSINGNESS, "impute_median")
_op("Consider target encoding", "target_encode",
    "Encode categories by their target mean", RECIPE, "encode", "target")
_op("Consider dimensionality reduction (PCA)", "pca",
    "Reduce to principal components", FEATURE, "pca")
_op("Remove low-variance or redundant features", "select_features",
    "Select a subset of features", SELECTION, "mutual_info")

# ── earmarks for a PERSON. The app cannot do these and says so ───────────────

_mark("Verify units and data entry", "verify_units",
      "Verify units and data entry", YOU,
      "Only you can check a value against the instrument or the chart that "
      "produced it. The app can show you which entries look wrong and cannot "
      "tell you whether they are.")
_mark("Investigate if outliers are errors or genuine", "triage_outliers",
      "Decide whether the extreme values are errors or real", YOU,
      "Whether an extreme value is a mistake or a finding is a question about "
      "your study, not about the number. The app can show you the rows; it "
      "cannot know which.")
_mark("Consider collecting more data", "collect_more",
      "Collect more data", YOU,
      "Nothing in the app changes the sample size. Recorded here so the "
      "limitation reaches the manuscript rather than being noticed at review.")
_mark("Be cautious interpreting results", "interpret_with_caution",
      "Read the results with this in mind", YOU,
      "Carried into the report as a stated limitation rather than left as "
      "advice nobody wrote down.")
_mark("Review plausible ranges for affected features", "review_ranges",
      "Review the flagged ranges", "explore",
      "The plausibility view lists every flagged entry in two tiers — "
      "impossible and merely improbable — with the rows named. That is where "
      "this is done.")
_mark("Investigate if missingness is random (MAR) or informative (MNAR)",
      "mechanism", "Say what a blank means here", "preprocess",
      "This is the mechanism question, and Preprocess asks it directly rather "
      "than leaving it as advice. Answering it there is answering this.")

# ── earmarks for a STEP that owns the decision ───────────────────────────────

for _phrase, _key in (
        ("Use regularized linear models (Ridge, Lasso)", "prefer_regularized"),
        ("Use regularized models (Ridge, Lasso, ElasticNet)", "prefer_regularized"),
        ("Use regularization", "prefer_regularized")):
    _mark(_phrase, _key, "Prefer regularized models", "preprocess",
          "Model choice happens at Preprocess, where the shelf is ordered by "
          "the shape of your data and this concern is stated on it.")

_mark("Prefer simpler models", "prefer_simple", "Prefer simpler models",
      "preprocess",
      "Model choice happens at Preprocess, where the shelf is ordered and "
      "every concern is stated beside the model it is about.")
_mark("Tree models are robust to outliers", "prefer_trees_outliers",
      "Prefer tree models, which are robust to this", "preprocess",
      "A property of the models rather than of your data, so it belongs where "
      "the models are chosen.")
_mark("Tree models can handle missing values natively", "prefer_trees_missing",
      "Prefer tree models, which handle blanks natively", "preprocess",
      "A property of the models rather than of your data, so it belongs where "
      "the models are chosen.")
_mark("Tree models handle high cardinality better", "prefer_trees_cardinality",
      "Prefer tree models, which handle many categories better", "preprocess",
      "A property of the models rather than of your data, so it belongs where "
      "the models are chosen.")
_mark("Consider robust models (Huber loss)", "prefer_robust_loss",
      "Prefer a robust loss", "preprocess",
      "Model choice happens at Preprocess; a robust loss is one of the "
      "options the shelf carries.")
_mark("Use class weights in training", "class_weights",
      "Weight the classes", "train",
      "Set where the models are fitted. Not built in the Guided door yet, and "
      "the earmark says so rather than the card pretending otherwise.")
_mark("Consider SMOTE or other resampling (with caution)", "resampling",
      "Resample the minority class", "train",
      "Resampling changes what the training rows ARE, so it belongs beside "
      "fitting. Not built in the Guided door yet.")
_mark("Focus on precision-recall metrics, not accuracy", "pr_metrics",
      "Score on precision and recall rather than accuracy", "train",
      "Which metric answers your question is decided where the scores are "
      "produced.")
_mark("Adjust classification threshold based on costs", "threshold",
      "Choose the classification threshold", "train",
      "The threshold turns a probability into a decision, and what it costs to "
      "be wrong in each direction is yours to say.")
_mark("Increase cross-validation folds", "more_folds",
      "Use more cross-validation folds", "train",
      "Set where the models are fitted.")
_mark("Use cross-validation", "cross_validate", "Use cross-validation", "train",
      "Set where the models are fitted.")
_mark("Consider frequency encoding", "frequency_encode",
      "Encode categories by how often they occur", "features",
      "The transform catalogue does not carry this one. Earmarked to Features, "
      "which is where it would live — the app is saying it cannot do this "
      "rather than offering a control that does nothing.")
_mark("Group rare categories", "group_rare",
      "Group the rare categories together", "features",
      "The transform catalogue does not carry this one. Earmarked to Features, "
      "which is where it would live.")
_mark("Consider unit harmonization or plausibility gating", "harmonize_units",
      "Harmonize the units, or gate on plausibility", "explore",
      "The plausibility view carries both readings — the impossibility band "
      "proposes setting an entry to missing, and the unit reference names what "
      "each column is measured in. That is where this is done.")


def classify(phrase: str):
    """One engine suggestion, as an `Operation` or an `Earmark`, or `None`.

    `None` means **this table does not know the phrase**, which is a gap rather
    than a classification — and the interface renders it as the prose it always
    was, so a new engine suggestion degrades to the old behavior instead of
    disappearing. The test is what stops that being permanent.
    """
    return _TABLE.get(_norm(phrase))


def known_phrases() -> Dict[str, Any]:
    return dict(_TABLE)


# ─────────────────────────────────────────────────────────────────────────────
# What a finding offers
# ─────────────────────────────────────────────────────────────────────────────

def _columns_for(finding: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    named = [str(c) for c in (finding.get("affected_columns") or [])
             if str(c) in df.columns]
    if named:
        return named
    params = finding.get("params") or {}
    return [str(c) for c in (params.get("columns") or []) if str(c) in df.columns]


def _timing(catalogue: str, binding: str, variant: Optional[str],
            columns: Sequence[str]) -> Tuple[str, bool, str]:
    """Clause §06's answer for one operation: what runs when, and the prose.

    Read from the catalogue that owns the operation rather than decided here.
    Two homes for one timing rule is two rules that will disagree, and this
    module is not one of them.
    """
    n = len(columns)
    where = f"{n} feature(s)" if n != 1 else f"`{columns[0]}`"
    if catalogue == FEATURE:
        from turbotab import features as _feat
        spec = _feat.get(binding)
        defers = bool(spec.defers)
        return (spec.because, defers,
                f"{spec.label} over {where}"
                + (", fitted within each training fold." if defers else
                   ", applied now."))
    if catalogue == MISSINGNESS:
        from turbotab import missingness as _miss
        spec = _miss.strategy(binding)
        defers = bool(spec["defers"])
        return (spec["because"], defers,
                f"{spec['label']} for {where}"
                + (", computed within each training fold." if defers else
                   ", applied now."))
    if catalogue == RECIPE:
        from turbotab import recipes as _rec
        op = _rec.operation(binding)
        return (op.because, True,
                f"{op.label} set to {variant} for {where}, fitted within each "
                f"training fold.")
    from turbotab import selection as _sel
    method = _sel.METHODS[binding]
    return ("Feature selection learns which columns matter from the rows it "
            "sees, so it is fitted inside the training folds.", True,
            f"{method.label} over {where}, selected within each training fold.")


def offers(finding: Dict[str, Any], df: pd.DataFrame,
           target: Optional[str] = None) -> Dict[str, Any]:
    """Everything this finding offers to do, split into options and earmarks.

    **Never empty where the engine said anything**, which is the whole point:
    the branch this replaces printed the same list as paragraphs and stopped.
    """
    options: List[Dict[str, Any]] = []
    earmarks: List[Dict[str, Any]] = []
    unclassified: List[str] = []
    columns = _columns_for(finding, df)

    for phrase in finding.get("suggested_actions") or []:
        found = classify(phrase)
        if found is None:
            unclassified.append(str(phrase))
            continue
        if isinstance(found, Earmark):
            earmarks.append({**found.to_dict(), "phrase": str(phrase)})
            continue
        try:
            because, defers, sentence = _timing(
                found.catalogue, found.binding, found.variant, columns)
        except Exception:
            # A binding the catalogue no longer resolves. Degraded to an
            # earmark rather than dropped, and loudly, because an option that
            # silently disappears is the dead end returning by another route.
            from turbotab import devchecks
            devchecks.swallowed(
                f"actions.offers::{found.key}", _last_exception(),
                f"the {found.key!r} option was withdrawn from a finding card "
                f"and the user sees one fewer thing to do")
            earmarks.append({
                "kind": "earmark", "key": found.key, "label": found.label,
                "target_step": "features", "is_for_a_person": False,
                "why": "The catalogue does not resolve this operation here.",
                "phrase": str(phrase)})
            continue
        options.append({**found.to_dict(), "phrase": str(phrase),
                        "columns": columns, "because": because,
                        "defers": defers, "sentence": sentence,
                        "previewable": bool(columns)})

    return {"options": options, "earmarks": earmarks,
            "unclassified": unclassified,
            "n_columns": len(columns), "columns": columns,
            # Stated rather than inferred by a renderer counting lists. A
            # finding whose every suggestion is a human task is a legitimate
            # outcome and a different sentence from one nobody classified.
            "summary": _summary(options, earmarks, unclassified)}


def _summary(options, earmarks, unclassified) -> str:
    if options:
        return (f"{len(options)} of these the app can do and show you; "
                f"{len(earmarks)} are decisions that live elsewhere.")
    if earmarks:
        return ("Nothing here is an operation on the data. "
                + (f"{sum(1 for e in earmarks if e['is_for_a_person'])} of "
                   f"these are yours to do and the rest belong to a later "
                   f"step — each is earmarked where it resurfaces."))
    if unclassified:
        return ("The engine's suggestions here are not yet classified, so they "
                "are shown as it wrote them.")
    return "The engine reports this without suggesting anything."


def _last_exception() -> BaseException:
    import sys
    return sys.exc_info()[1] or RuntimeError("unknown")
