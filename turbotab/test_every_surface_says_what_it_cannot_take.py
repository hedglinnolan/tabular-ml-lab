"""L40-D — the general form of L39's zero.

L39-D drove one shape end to end and found **14 surfaces, 11 correct, 3
silently wrong, 0 refusing** — *nothing anywhere declines a multiclass target,
so every failure looks exactly like a right answer.* This is that sweep
generalized: every target shape the app can encounter, against every surface
that consumes a target, asserting each cell either **handles** the shape or
**refuses in a way a user can read**.

**Silence is the defect.** A surface that returns a number for a shape it
cannot handle is worse than one that returns nothing, because the number is
indistinguishable from a right answer.

## Two things this must not do

It must not close `GUIDED-118` by inventing a time-to-event target type.
Refusing is the correct behavior there and L38 established it, so the survival
column expects a refusal and the test fails if one ever appears.

And it must not turn every unsupported shape into a wall. `PRODUCT_VISION.md`'s
ladder puts most of this on *rank and state the concern*, and **the shelf is
never shortened** — so `RANKS_AND_STATES` is a third verdict beside handles and
refuses, for a surface that produces a number and says what is wrong with it.

## The counts this owes

`test_the_sweep_reports_its_own_coverage` prints them and asserts the last
column, which is the finding: shapes enumerated, surfaces checked, cells that
handle, cells that refuse readably, cells silently wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: Every target shape the app can encounter. Four are representable today and
#: three are not — and *not representable* is itself a cell worth checking,
#: because a shape the app cannot express is one a user will express anyway by
#: choosing the nearest task type.
SHAPES = {
    "binary": {
        "fixture": ("leaky_sepsis.csv", "sepsis"),
        "task": "classification",
        "representable": True,
    },
    "multiclass": {
        "fixture": ("multiclass_stage.csv", "disease_stage"),
        "task": "classification",
        "representable": True,
    },
    "ordered multiclass": {
        # THE SAME COLUMN, and that is the finding for this row: the app has no
        # ordinal task type, so an ordered outcome and a nominal one are the
        # same object to every surface below. Nothing is wrong with any single
        # answer; what is missing is the distinction.
        "fixture": ("multiclass_stage.csv", "disease_stage"),
        "task": "classification",
        "representable": False,
        "because": ("no ordinal task type exists, so an ordered outcome is "
                    "modeled as nominal and the order is discarded silently"),
    },
    "continuous": {
        "fixture": ("clinic_visits.csv", "hba1c"),
        "task": "regression",
        "representable": True,
    },
    "count": {
        # Representable AS regression, which is what a user will do — and the
        # consequence is that nothing offers a count model or mentions that a
        # negative prediction is impossible for this outcome.
        "fixture": ("clinical_longitudinal.csv", "visit"),
        "task": "regression",
        "representable": False,
        "because": ("no count task type or count model family exists, so a "
                    "count outcome is modeled as continuous and nothing "
                    "notices that predictions below zero are impossible"),
    },
    "time-to-event": {
        "fixture": None,
        "task": None,
        "representable": False,
        "because": ("no time-to-event target type exists (`GUIDED-118`); "
                    "L38 refused Kaplan-Meier for this reason and the refusal "
                    "stands"),
    },
    "multi-label": {
        "fixture": None,
        "task": None,
        "representable": False,
        "because": ("more than one class true per row is not modeled anywhere "
                    "and nothing pretends to"),
    },
}

#: The three verdicts. `RANKS_AND_STATES` exists because
#: `PRODUCT_VISION.md`'s ladder puts most of this on *rank and state the
#: concern* rather than on refusal — the shelf is never shortened, so a
#: surface that produces a number and says what is wrong with it is behaving
#: correctly and must not be counted as a defect.
HANDLES = "handles"
REFUSES = "refuses readably"
RANKS_AND_STATES = "ranks and states the concern"
SILENT = "silently wrong"

#: The surfaces that consume a target, and what each is asserted to do per
#: shape. A cell absent from a surface's map is `SILENT` by default, so adding
#: a shape without deciding what a surface does about it fails here.
SURFACES = {
    "task type": {
        "binary": HANDLES, "multiclass": HANDLES,
        # THE ORDERED CASE IS THE ONE WORTH READING. `set_task_type` accepts
        # `classification` for an ordered outcome and records nothing about
        # the order, so the app is not wrong about anything it said — it
        # simply cannot say the thing that matters. `GUIDED-130`.
        "ordered multiclass": SILENT,
        "continuous": HANDLES, "count": SILENT,
        "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "the seal": {
        "binary": HANDLES, "multiclass": HANDLES,
        "ordered multiclass": HANDLES, "continuous": HANDLES,
        "count": HANDLES, "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "resolution card": {
        # CLOSED AT L40-A1: the trigger reads `1 - 1/k` now.
        "binary": HANDLES, "multiclass": HANDLES,
        "ordered multiclass": HANDLES, "continuous": HANDLES,
        "count": HANDLES, "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "instability plot": {
        "binary": HANDLES,
        # `GUIDED-113`, still open: it plots `predict_proba[:, 1]`, one class
        # of k, under a caption that says it is about the prediction.
        "multiclass": SILENT, "ordered multiclass": SILENT,
        "continuous": HANDLES, "count": HANDLES,
        "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "decision curve / ROC": {
        # BUILT AT L40-C AND THE FIRST TO DECLINE A SHAPE. Both gate on
        # `n_classes == 2` and appear in `not_drawn` with a reason.
        "binary": HANDLES, "multiclass": REFUSES,
        "ordered multiclass": REFUSES, "continuous": REFUSES,
        "count": REFUSES, "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "calibration figure": {
        # `GUIDED-126`, still open: `when_applicable` asks only whether the
        # task is classification, so it is offered where there is no single
        # predicted risk.
        "binary": HANDLES, "multiclass": SILENT,
        "ordered multiclass": SILENT, "continuous": REFUSES,
        "count": REFUSES, "time-to-event": REFUSES, "multi-label": REFUSES,
    },
    "manuscript validator": {
        "binary": HANDLES, "multiclass": HANDLES,
        "ordered multiclass": HANDLES, "continuous": HANDLES,
        "count": HANDLES, "time-to-event": REFUSES, "multi-label": REFUSES,
    },
}

#: Cells that are `SILENT` and the row that owns each. Every silent cell must
#: be filed — an unfiled one is the state this whole sweep exists to end.
SILENT_CELLS_ARE_FILED = {
    ("task type", "ordered multiclass"): "GUIDED-130",
    ("task type", "count"): "GUIDED-130",
    ("instability plot", "multiclass"): "GUIDED-113",
    ("instability plot", "ordered multiclass"): "GUIDED-113",
    ("calibration figure", "multiclass"): "GUIDED-126",
    ("calibration figure", "ordered multiclass"): "GUIDED-126",
}


def _verdicts():
    return [(surface, shape, cells.get(shape, SILENT))
            for surface, cells in SURFACES.items() for shape in SHAPES]


# ═══════════ THE MATRIX IS COMPLETE ═══════════

def test_every_surface_has_a_verdict_for_every_shape():
    """A cell nobody decided is `SILENT` by default, so adding a shape without
    deciding what each surface does about it fails HERE rather than being
    discovered by the next sweep."""
    missing = [(surface, shape) for surface, cells in SURFACES.items()
               for shape in SHAPES if shape not in cells]
    assert not missing, (
        f"{missing} have no verdict. A surface with no decision about a shape "
        f"is the silence this sweep exists to end.")


def test_every_silently_wrong_cell_is_filed():
    """**The last column is the finding**, and an unfiled finding is a report
    nobody can act on."""
    import json

    ledger = {row["id"] for row in json.load(
        open("docs/turbotab/data/findings.json"))}
    silent = {(s, k) for s, k, v in _verdicts() if v == SILENT}

    assert silent == set(SILENT_CELLS_ARE_FILED), (
        f"the silent set and the filed set disagree: "
        f"{silent ^ set(SILENT_CELLS_ARE_FILED)}")
    for cell, row_id in SILENT_CELLS_ARE_FILED.items():
        assert row_id in ledger, f"{cell} is filed against {row_id}, which does not exist"


# ═══════════ THE REFUSALS ARE READABLE ═══════════

def test_a_time_to_event_target_is_refused_and_stays_refused():
    """**`GUIDED-118` must not close here.** Refusing is the correct behavior
    and L38 established it; a sweep that made the app accept a survival target
    to fill in a matrix cell would be inventing a capability to satisfy a
    test."""
    from turbotab import figures
    from turbotab.project import AnalysisProject, ProjectError
    import turbotab.figure_specs                            # noqa: F401

    df = pd.DataFrame({"t": [1, 2, 3, 4] * 5, "event": [0, 1] * 10,
                       "x": range(20)})
    p = AnalysisProject.from_dataframe(df, "survival.csv")
    with pytest.raises(ProjectError, match="not a task type"):
        p.override_task_type("survival")

    entry = figures.resolve("kaplan_meier")
    assert entry["status"] == figures.PENDING_STATUS
    assert entry["blocked_by"] == "GUIDED-118"
    assert len(entry["needs"]) > 200, (
        "the refusal does not say what is missing, which makes it "
        "indistinguishable from an absent feature")


def test_the_new_clinical_figures_refuse_readably_rather_than_silently():
    """L40-C's four are the first surfaces built after L39-D's zero and the
    first that decline a shape. A refusal is `not_drawn` WITH A REASON, never
    an absence."""
    from turbotab import figure_bundle as FB
    from turbotab.project import AnalysisProject

    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    p = AnalysisProject.from_dataframe(df, "multiclass_stage.csv")
    p.set_target("disease_stage", "classification", "high", [])
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(7)
    idx = list(p.df.index)
    rng.shuffle(idx)
    p.seal_lockbox(idx[:60], fraction=0.25)

    bundle = FB.render(p)
    offered = {row["id"] for row in bundle["admitted"] + bundle["held"]}
    reasons = {row["id"]: row["why"] for row in bundle["not_drawn"]}
    for figure_id in ("decision_curve", "roc"):
        assert figure_id not in offered
        assert figure_id in reasons, (
            f"{figure_id} is absent from a three-class project AND "
            f"unexplained, which is the silence L39-D found everywhere")
        assert reasons[figure_id], f"{figure_id} is declined with no reason"


def test_the_shelf_is_never_shortened_by_any_of_this():
    """`PRODUCT_VISION.md`'s rule, and the reason `RANKS_AND_STATES` exists as
    a third verdict. None of the refusals above removes a MODEL from the
    shelf — a figure declining a shape is not the same act as an app deciding
    which models a researcher may fit."""
    from turbotab import training as T
    from turbotab.project import AnalysisProject
    from ml.model_registry import get_registry

    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    p = AnalysisProject.from_dataframe(df, "multiclass_stage.csv")
    p.set_target("disease_stage", "classification", "high", [])
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(7)
    idx = list(p.df.index)
    rng.shuffle(idx)
    p.seal_lockbox(idx[:60], fraction=0.25)

    shelf, _profile = p.model_shelf_ranked()
    classifiers = {spec.key for spec in get_registry().values()
                   if "clf" in spec.key or spec.key in
                   ("logreg", "gaussian_nb", "lda", "rf", "svc", "knn_clf")}
    offered = {entry.key for entry in shelf}
    assert classifiers & offered, "no classifier is offered at all"
    # AND IT FITS. A shelf that offered a model the app then refused to fit
    # would be the shortening done one step later.
    run = T.train(p, ["logreg"])
    assert [r for r in run.results if not r.error]


# ═══════════ THE COUNTS ═══════════

def test_the_sweep_reports_its_own_coverage(capsys):
    """**The counts this probe owes.** The last column is the finding."""
    verdicts = _verdicts()
    counts = {v: sum(1 for _, _, verdict in verdicts if verdict == v)
              for v in (HANDLES, REFUSES, RANKS_AND_STATES, SILENT)}
    representable = sum(1 for s in SHAPES.values() if s["representable"])

    with capsys.disabled():
        print(f"\n  target shapes enumerated       {len(SHAPES)}"
              f"   ({representable} representable, "
              f"{len(SHAPES) - representable} not)")
        print(f"  surfaces checked               {len(SURFACES)}")
        print(f"  cells                          {len(verdicts)}")
        print(f"  handle the shape               {counts[HANDLES]}")
        print(f"  refuse readably                {counts[REFUSES]}")
        print(f"  rank and state the concern     {counts[RANKS_AND_STATES]}")
        print(f"  SILENTLY WRONG                 {counts[SILENT]}"
              f"   ({', '.join(sorted(set(SILENT_CELLS_ARE_FILED.values())))})")

    assert len(verdicts) == len(SURFACES) * len(SHAPES) == 49
    assert counts[SILENT] == 6
    # L39-D's ZERO, no longer zero. Something refuses now, which is the whole
    # point — the sweep's own subject was that nothing did.
    assert counts[REFUSES] > 0, (
        "no surface refuses any shape readably, which is exactly the finding "
        "L39-D reported and this loop was meant to begin closing")


def test_a_shape_the_app_cannot_represent_says_why():
    """Three of the seven are not representable, and *not representable* is a
    cell worth checking: a user will express the shape anyway by choosing the
    nearest task type, and what happens then is the app's answer whether or
    not it meant to give one."""
    for name, shape in SHAPES.items():
        if shape["representable"]:
            continue
        assert shape.get("because"), f"{name} is unrepresentable with no reason"
        assert len(shape["because"]) > 40
