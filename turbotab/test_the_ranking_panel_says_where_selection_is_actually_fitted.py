"""`AUDIT-027` · the 'Rank them for me' panel and the record say the same scope.

## The false sentence

`selection.evidence` returned, whenever any row was withheld:

    Ranked on training rows only, and not applied. **What is actually selected
    is refitted inside each training fold**, so this ordering is indicative
    rather than the answer.

Nothing in this app refits selection inside a fold. `api` records
`scope=train_rows` for this door — with a comment saying so — `training.train`
does exactly one `pipe.fit(X_train, y_train)` per model, and `pipeline_plan`
records `scope_fitted=train_rows` and raises a `Divergence` when a spec asked
for the stronger one. The panel was the last surface still asserting it, and it
bypassed `declare`'s own rewrite three functions above it — the rewrite that
exists precisely so a door that fits once can SAY `train_rows` instead of
implying the stronger claim (`GUIDED-104`).

`CLINICAL_SURVEY_PACK.md` §A5.5 is why it matters rather than being pedantry:
*internal validation must resample the entire modeling pipeline — imputation,
transformation, selection, tuning.* Telling a researcher that selection is
refitted per fold is telling them their selection sits inside a resampling
loop. It does not. What it sits inside is the single train/test split the same
paragraph calls the weakest option.

## The correction

Same subject, weaker claim, true:

    …What is actually selected is fitted **once over the training rows
    (held-out rows excluded)** — this door fits each model one time, so there
    is a single fold — so this ordering is indicative rather than the answer.

The phrase now comes from `selection._SCOPE_PHRASE[FITTED_SCOPE]`, the same
table `declare` rewrites with, so the panel and the record cannot drift again;
and `pipeline_plan` reads `FITTED_SCOPE` for `scope_fitted`, so there is one
name for *what this door fits* rather than a literal in each module.

## Driven, not described

Every assertion below drives the real route the row names — Guided door →
Features step → 'Rank them for me' → `GET /project/{id}/selection/evidence` —
through `TestClient(api.app)`. Nothing here reads source text.

## Fixture shapes — `GUIDED-097`

Two shapes of different target type: a continuous outcome and a three-level
string outcome. `SHAPES_NOT_COVERED` names what is not driven and why.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import selection as _sel                          # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: The sentence the row was filed against, quoted so a revert names itself.
THE_FALSE_CLAIM = "refitted inside each training fold"

#: `GUIDED-097`. Two target shapes, deliberately of different type: the
#: evidence note is composed before any scoring branch is chosen, so a shape
#: whose preview refuses (`kind == "none"`) and one whose preview computes must
#: both be exercised — a continuous outcome scores with `f_regression`, a
#: three-level string outcome with `mutual_info_classif`.
TARGET_SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c", ["age", "bmi", "sbp"]),
    "multiclass_string": ("multiclass_stage.csv", "disease_stage",
                          ["age", "bmi", "hba1c"]),
}

#: NOT COVERED, named here rather than discovered later.
SHAPES_NOT_COVERED = {
    "binary numeric (0/1)": (
        "`leaky_sepsis.csv` has a 0/1 target. The note is composed from "
        "`n_rows_withheld` and `FITTED_SCOPE` and reads neither the target nor "
        "the task type, so the behavior is expected to be identical — which is "
        "exactly why `GUIDED-097` says to say it is undriven rather than to "
        "assume it."),
    "a project that recorded scope=train_folds explicitly": (
        "The API's `set_selection` defaults to `TRAIN_ROWS` and no surface "
        "offers the other value, so this is unreachable through the door. If "
        "it ever becomes reachable, the note stays true — it states what is "
        "FITTED — and `pipeline_plan` is the surface that must then report the "
        "divergence from what was RECORDED. Undriven here."),
    "no seal drawn": (
        "Covered by the existing "
        "`test_selection_evidence_without_a_mask_says_it_saw_everything`, "
        "which pins the other branch of this same note."),
}


# ═══════════ 3 · the vocabulary has one home ═══════════

def test_the_recorded_sentence_and_the_panel_are_composed_from_one_table():
    """`declare`'s rewrite and the panel's note draw the same clause.

    The defect was two hand-written sentences about one fact. This pins them to
    `_SCOPE_PHRASE`, so a future edit to either wording moves both — the shape
    `missingness._SCOPE_PHRASE` already has one module over.
    """
    spec = _sel.declare("mutual_info", "y", ["a", "b", "c"], n_features=2,
                        scope=_sel.TRAIN_ROWS)
    clause = _sel._SCOPE_PHRASE[_sel.FITTED_SCOPE]
    assert clause in spec["sentence"], (
        f"the recorded sentence does not carry the scope clause: "
        f"{spec['sentence']!r}")
    assert _sel.FITTED_SCOPE == _sel.TRAIN_ROWS, (
        "this door fits each model once; if that changed, `training.train` "
        "changed and every sentence composed from FITTED_SCOPE moved with it")
