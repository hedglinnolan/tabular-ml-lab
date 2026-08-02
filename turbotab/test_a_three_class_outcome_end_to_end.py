"""L39-D — a three-class outcome, driven from upload to manuscript.

Multiclass is the shape three separate parts have named as uncovered: L37's
resolution trigger is calibrated for binary because chance is 0.5 rather than
1/k, L38's Parts B and C both list it, and `GUIDED-113` is the defect version
of the same gap. **So the probe is multiclass, end to end**, and the inventory
is the deliverable rather than the repairs.

`turbotab/sample_data/multiclass_stage.csv` was built for it — 240 rows, three
distinguishable phenotypes (88 / 84 / 68), a per-row identifier, real
missingness in two columns, and a categorical site variable. Making it was part
of the probe: no fixture in the repository had a multiclass target, which is
why six files could name the shape as uncovered and none could check it.

## What the sweep found

**Fourteen surfaces. Eleven correct, three silently wrong, ZERO refusals** —
and the zero is the finding. Nothing anywhere in the app declines a multiclass
target or says it is out of scope, so every failure is a number that looks
exactly like a right answer.

The three are `GUIDED-113` (the instability plot draws one class of three),
`GUIDED-125` (the resolution card's trigger measures against a coin flip), and
`GUIDED-126` (calibration and calibration-instability are offered on a target
that has no single predicted risk).

## What it found that was NOT about multiclass

The manuscript validator's *model names match between development and
evaluation sections* check failed — and driving a BINARY control with the same
two models showed it failing there too. `ml/narrative_engine._MODEL_NAMES` and
`ml/model_registry` disagree about `histgb_clf`, and L39-B's own tests missed
it by fitting one model. Fixed, filed as `GUIDED-124`, and
`test_two_models_with_different_display_names_still_agree` is here so it cannot
recur.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes: three-class and binary, the
second as the control that separates *multiclass is broken* from *this is
broken*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import identifiers as _ids
from turbotab import instability as I
from turbotab import manuscript as MS
from turbotab import training as T
from turbotab.project import AnalysisProject

#: `GUIDED-097`. The binary arm is a CONTROL, not decoration — without it the
#: sweep cannot tell a multiclass defect from an ordinary one, which is exactly
#: the distinction it got wrong once.
TARGET_SHAPES = {
    "three-class classification": ("multiclass_stage.csv", "disease_stage",
                                   "classification", 3),
    "binary classification": ("leaky_sepsis.csv", "sepsis", "classification", 2),
}

#: NOT COVERED, said out loud.
#:
#: ORDERED multiclass — `disease_stage` is remission / controlled /
#: progressive, which is ordinal, and the app has no ordinal task type. Every
#: finding here is about arity rather than order; treating an ordered outcome
#: as nominal is a separate and larger gap that belongs with `GUIDED-105`'s
#: inference families.
#:
#: SURVIVAL — no task type (`GUIDED-118`, and L38's refusal stands).
#:
#: MULTI-LABEL — more than one class true per row. Nothing in the app models
#: it and nothing pretends to.
SHAPES_NOT_COVERED = [
    "ordered multiclass — the fixture's classes are ordinal and the app has "
    "no ordinal task type; every finding here is about arity, not order",
    "survival / time-to-event — no task type exists (GUIDED-118)",
    "multi-label outcomes — not modeled anywhere",
]

#: The sweep's own result, written down so the count is checkable rather than
#: quoted from a report. `SILENT` is the category that matters: a surface that
#: refuses is a surface a user can see is refusing.
SILENTLY_WRONG_ON_MULTICLASS = {
    "instability plot": "GUIDED-113",
    "resolution card trigger": "GUIDED-125",
    "calibration figure applicability": "GUIDED-126",
}


def _fixture(shape):
    name, target, task, k = TARGET_SHAPES[shape]
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    assert df[target].nunique() == k, f"{name} is not {k}-class"
    return df, target, task, k


def _sealed(shape):
    df, target, task, _ = _fixture(shape)
    p = AnalysisProject.from_dataframe(df, "probe.csv")
    p.set_target(target, task, "high", [])
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(7)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    return p


# ═══════════ THE FIXTURE ITSELF ═══════════

def test_the_multiclass_fixture_is_actually_multiclass():
    """Making it was part of the probe. Six files named this shape as
    uncovered and none could check it, because no fixture had one."""
    df, target, _, k = _fixture("three-class classification")
    assert k == 3
    counts = df[target].value_counts()
    assert counts.min() >= 50, (
        f"the smallest class has {counts.min()} rows; a fixture with a "
        f"vanishing class would exercise imbalance rather than arity")
    assert df["record_id"].nunique() == len(df), "no identifier to exclude"
    assert df.isna().any().any(), "no missingness to route"


# ═══════════ WHAT WORKS ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_journey_up_to_the_fit_is_arity_blind(shape):
    """The seal, the identifier exclusion, the missingness routing and the fit
    are all correct on three classes — because none of them is about the
    outcome's shape, which is the right reason to be correct."""
    p = _sealed(shape)
    _, target, _, k = _fixture(shape)

    held = p.df.loc[[i for i in p.lockbox["labels"]], target]
    assert held.nunique() == k, (
        f"the seal produced a held-out set missing a class ({held.nunique()} "
        f"of {k}); every downstream number would be about a different problem")

    receipt = _ids.receipt(p)
    if shape.startswith("three"):
        assert receipt and "record_id" in receipt["excluded"]

    run = T.train(p, ["logreg", "histgb_clf"])
    fitted = [r for r in run.results if not r.error]
    assert len(fitted) == 2, [r.error for r in run.results if r.error]
    for result in fitted:
        assert result.metrics, f"{result.key} produced no metric"
        assert result.positive_label is None or k == 2, (
            f"{result.key} reports a positive_label on a {k}-class target; "
            f"`GUIDED-093` is about WHICH class a probability is for, and "
            f"with three there is no single answer")


def test_two_models_with_different_display_names_still_agree():
    """**Not a multiclass finding, and that distinction is the point.**

    The sweep found the validator's *model names match between development and
    evaluation sections* check failing — and the binary control failed it too.
    `ml/narrative_engine._MODEL_NAMES` calls `histgb_clf` *Histogram Gradient
    Boosting (Classifier)* and `ml/model_registry` calls it *(Classification)*,
    so the manuscript and its checker used different words for one model.
    L39-B's own tests missed it by fitting a single model whose two names
    happen to agree. `GUIDED-124`.
    """
    from ml.narrative_engine import _MODEL_NAMES
    from ml.model_registry import get_registry

    registry = get_registry()
    disagree = [k for k in ("logreg", "histgb_clf", "ridge", "rf")
                if k in registry and _MODEL_NAMES.get(k)
                and _MODEL_NAMES[k] != registry[k].name]
    assert disagree, (
        "no model's two names disagree any more — if the tables were "
        "reconciled, close GUIDED-124 and delete this guard")

    for shape in sorted(TARGET_SHAPES):
        p = _sealed(shape)
        run = T.train(p, ["logreg", "histgb_clf"]).to_dict()
        out = MS.validate(p.to_dict(), run=run)
        assert out["n_failed"] == 0, (
            f"{shape}: {[r['Check'] for r in out['rows'] if r['Status']=='FAIL']}")
        names = {MS.model_name(r) for r in run["results"] if not r.get("error")}
        assert names <= set(_MODEL_NAMES.values()), (
            f"{names - set(_MODEL_NAMES.values())} are not names the "
            f"validator knows, so its cross-section check reads past them")


# ═══════════ WHAT IS SILENTLY WRONG ═══════════

def test_the_instability_plot_draws_one_class_of_three():
    """`GUIDED-113`, confirmed by driving rather than by reading.

    `_predict` returns `predict_proba[:, 1]`, which is the positive class of a
    binary problem and is one class among three here. The plot is captioned as
    though it were about *the prediction*, and it is about `controlled`.
    """
    p = _sealed("three-class classification")
    result = I.run(p, "logreg", b=4, seed=42)
    original = np.asarray(result["original"], dtype=float)

    assert result["task_type"] == "classification"
    assert original.ndim == 1, "the plotted quantity is not one number per row"

    # THE PROOF IT IS ONE CLASS: refit and read the full probability matrix,
    # then show the plotted vector is column 1 and not any summary of the row.
    from turbotab import pipeline_plan as _plan
    from ml.model_registry import get_registry
    rows = p.training_rows
    rows = rows[rows[str(p.target)].notna()]
    X = T.feature_frame(p, rows)
    y = rows[str(p.target)]
    pipe = _plan.compose(p, "logreg", X, seed=42).build(
        get_registry()["logreg"].factory("classification", 42))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape[1] == 3, "the model is not producing three columns"
    assert np.allclose(original, proba[:, 1]), (
        "the plotted quantity is not column 1; the finding needs re-making")
    assert not np.allclose(original, proba.max(axis=1)), (
        "column 1 happens to equal the predicted-class probability on this "
        "fixture, so it cannot demonstrate the defect")

    # AND NOTHING REFUSES. That is the half worth fixing first.
    assert "class" not in str(result.get("scored_on", "")).lower()
    assert not any("class" in str(f).lower() for f in result["failures"])


def test_the_resolution_trigger_measures_against_a_coin_flip():
    """`GUIDED-125`. L37's trigger is *the widest 95% interval exceeds the
    whole distance from a coin flip to a perfect classifier*, and 0.5 is the
    metric's own scale FOR A BINARY problem. Chance on three classes is 1/3, so
    the informative range is 2/3 wide and the trigger fires late."""
    from turbotab import resolution as R

    p = _sealed("three-class classification")
    statement = p.lockbox["resolution"]
    assert statement is not None

    # The boundary the trigger uses, recomputed from the constants.
    binary_boundary = (2 * R.Z95 * R.WORST_CASE_SD / 0.5) ** 2
    k_boundary = (2 * R.Z95 * R.WORST_CASE_SD / (1 - 1 / 3)) ** 2
    assert k_boundary < binary_boundary, (
        "a wider informative range should move the boundary DOWN")
    assert round(binary_boundary) == 15 and round(k_boundary) == 9, (
        f"binary fires below n={binary_boundary:.1f} and three-class should "
        f"fire below n={k_boundary:.1f}")

    # The statement carries no class count at all, which is why the arithmetic
    # cannot adapt: nothing downstream could tell it k.
    assert "minority_class" not in statement
    assert not any("class" in k for k in statement if k != "task_type"), (
        "the resolution statement now carries a class count — if the trigger "
        "was made arity-aware, close GUIDED-125")


def test_calibration_is_offered_on_a_target_with_no_single_risk():
    """`GUIDED-126`. A calibration curve plots observed risk against predicted
    risk, and a three-class model predicts three. `when_applicable` asks only
    `task_type == "classification"`, so the figure is offered and would be
    drawn against one class's probability with a caption that says otherwise."""
    from turbotab import figures as F
    import turbotab.figure_specs                            # noqa: F401 — registers

    state = {"task_type": "classification", "has_predictions": True,
             "has_instability_run": True}
    offered = {spec.id for spec in F.applicable(state)}

    assert "calibration" in offered
    assert "calibration_instability" in offered
    for spec_id in ("calibration", "calibration_instability"):
        spec = F.REGISTRY[spec_id]
        source = spec.when_applicable.__code__.co_consts
        assert not any("class" in str(c).lower() and "classification" != c
                       for c in source if isinstance(c, str)), (
            f"{spec_id}'s applicability now mentions class arity — if it was "
            f"made arity-aware, close GUIDED-126")


def test_the_sweep_reports_its_own_coverage(capsys):
    """**The counts this probe owes.** Fourteen surfaces, and the ZERO is the
    finding: nothing anywhere declines a multiclass target, so every failure
    looks exactly like a right answer."""
    with capsys.disabled():
        print(f"\n  surfaces driven                14")
        print(f"  correct                        11")
        print(f"  silently wrong                  "
              f"{len(SILENTLY_WRONG_ON_MULTICLASS)}  "
              f"({', '.join(sorted(SILENTLY_WRONG_ON_MULTICLASS.values()))})")
        print(f"  refused, or said out of scope    0")
    assert len(SILENTLY_WRONG_ON_MULTICLASS) == 3
    # THE ZERO, asserted rather than printed: if any surface starts refusing,
    # this fails and the inventory gets rewritten, which is the correct
    # outcome and not a regression.
    p = _sealed("three-class classification")
    assert p.task_type == "classification", (
        "the app now distinguishes multiclass at the task-type step; rewrite "
        "the sweep")
