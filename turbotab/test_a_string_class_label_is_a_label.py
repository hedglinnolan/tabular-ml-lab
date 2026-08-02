"""`GUIDED-093` — the fit succeeded and the app said it failed.

## The defect, and the three false statements from one line

`result.metrics` was assigned, and the **next** statement was
`result.predictions = [float(v) for v in y_pred]`. A string class label cannot
be coerced, so the assignment that had already succeeded was followed by an
exception, and the handler labeled the whole result a failure. Measured on
`sample_data/clinic_visits.csv`, target `outcome` (`responder` /
`non-responder`): every model carried **Accuracy 0.857 AND** `error="ValueError:
could not convert string to float"` **AND** zero predictions.

1. The page rendered *"did not fit"* beside a model that fitted and scored,
   because it reads `error` while an API consumer reads `metrics`.
2. An API consumer saw the metric with no sign anything went wrong, so the two
   doors disagreed about whether the user had a model at all.
3. **The calibration figure said *"the models that were fitted do not produce
   probabilities, so there is nothing to calibrate"* — about logistic
   regression.** That sentence lives inside the not-drawn apparatus
   `GUIDED-065` built one loop earlier to be precise about which of four states
   applies, and it was stating a falsehood about the user's model class to
   explain the app's own serialization bug. Silence would have been permitted.
   That is an assertion.

## And a fourth thing, found while fixing it

`predict_proba`'s second column is `classes_[1]`. On a 0/1 target that is `1`
and nobody ever had to ask. On `responder` / `non-responder` it is whichever
sorts second — and a calibration curve binarized against the *other* class is a
picture of the complementary event, drawn confidently, with every annotation
number wrong in a way no reader could detect. So the run records which class
the probabilities are about, the payload carries it, and `predictions_for`
returns `None` rather than guessing where it is absent.

## THE FIXTURE RULE, which is the real lesson (`GUIDED-097`)

This survived two loops because every Train and calibration claim used
`leaky_sepsis.csv`, whose target is `0`/`1`, so `float()` succeeded. String
outcomes are the ordinary case in clinical research — died/survived,
case/control, responder/non-responder — and `clinic_visits.csv` is the app's
own clinical fixture.

That is `GUIDED-075` one level over. The project learned *don't verify through
the API, drive the page*; it had not learned the data version: **don't verify
against the fixture that works.** So the Train claims below are parametrized
over `TARGET_SHAPES`, and `test_the_shapes_this_file_does_not_cover_are_named`
makes the uncovered ones an explicit list rather than a silence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, figure_bundle, training                     # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


#: **The fixture rule made mechanical.** Every claim about the Train step runs
#: against at least two of these, and the shapes NOT covered are named below
#: rather than left out.
TARGET_SHAPES = {
    "binary_numeric": ("leaky_sepsis.csv", "sepsis", "classification"),
    "binary_string": ("clinic_visits.csv", "outcome", "classification"),
    "continuous": ("clinic_visits.csv", "hba1c", "regression"),
}

#: Shapes no fixture in this repository has, stated so the coverage claim is
#: honest. A sweep that reports only what it covered has not reported its
#: coverage.
SHAPES_NOT_COVERED = {
    "multiclass": (
        "No fixture has a three-or-more-level outcome. `_metrics` routes "
        "multiclass through `roc_auc_score(multi_class='ovr')` and the "
        "calibration figure declines it, and neither branch is exercised "
        "here."),
    "boolean_dtype": (
        "No fixture has a true `bool` outcome column. `_serialize` has a "
        "branch for it — a bool is neither a number nor a label — and nothing "
        "drives it end to end."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _trained(client, fixture, target, *, models=None, fraction=0.25):
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column=target)
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=fraction)
    project = api.STORE.get(pid)
    shelf = [e.to_dict()["key"] for e in project.model_shelf()]
    keys = [k for k in (models or shelf[:2]) if k in shelf] or shelf[:1]
    run = training.train(project, keys)
    project.training_run = run
    return pid, project, run


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES),
                         ids=sorted(TARGET_SHAPES))
def test_a_result_carries_a_score_or_a_reason_and_never_both(client, shape):
    """**The guard, corrected.** `test_the_page_says_what_the_record_says`
    asserted `result['metrics'] or result['error']`, which is satisfied when
    BOTH are set — so it could not see the state the app was actually in.

    Mutual exclusion, on every target shape. If serialization failed the metric
    is not trustworthy either, and the honest form is one or the other.
    """
    fixture, target, _ = TARGET_SHAPES[shape]
    pid, project, run = _trained(client, fixture, target)
    assert run.results, "nothing was fitted, so this asserts nothing"
    for result in run.results:
        assert bool(result.metrics) != bool(result.error), (
            f"{result.key} on the {shape} target carries "
            f"{result.metrics!r} AND {result.error!r}")


def test_a_string_class_label_reaches_the_result_as_a_label(client):
    """The line itself. `[float(v) for v in y_pred]` on `non-responder`.

    Asserted on what a consumer receives, because that is what broke: the
    predictions were empty and the metric was real.
    """
    pid, project, run = _trained(client, "clinic_visits.csv", "outcome",
                                 models=["logreg", "knn_clf"])
    for result in run.results:
        assert result.error is None, (result.key, result.error)
        assert result.metrics.get("Accuracy") is not None
        assert result.predictions, (
            f"{result.key} scored and serialized no prediction at all")
        assert set(result.predictions) <= {"responder", "non-responder"}, (
            "the class labels were coerced into something they are not")
        assert result.probabilities and len(result.probabilities) == \
            len(result.predictions)


def test_a_continuous_prediction_is_still_a_number(client):
    """The other half of `_serialize`, and the reason it is not `str()` for
    everything: a regression consumer must be unchanged."""
    pid, project, run = _trained(client, "clinic_visits.csv", "hba1c")
    scored = [r for r in run.results if r.metrics]
    assert scored, [r.error for r in run.results]
    for value in scored[0].predictions[:5]:
        assert isinstance(value, float), (value, type(value))
    assert scored[0].probabilities is None, (
        "a regression run produced probabilities, which are not a thing it has")


def test_the_calibration_figure_does_not_blame_the_model_for_a_serialization_bug(
        client):
    """**The third false statement, and the one that matters.**

    The sentence *"the models that were fitted do not produce probabilities"*
    is `GUIDED-065`'s four-state apparatus doing its job — on a regression task
    it is exactly right. Said about logistic regression it is a falsehood about
    the user's model class, offered to explain a bug in the app.
    """
    pid, project, run = _trained(client, "clinic_visits.csv", "outcome",
                                 models=["logreg", "knn_clf"])
    reason = figure_bundle._no_predictions_because(project)
    assert "do not produce probabilities" not in reason, (
        "the app says logistic regression produces no probabilities, which is "
        f"false: {reason}")

    figures = client.get(f"/project/{pid}/figures").json()
    drawn = [f for f in figures["admitted"] + figures["held"]
             if f["id"] == "calibration"]
    assert drawn, (
        "the calibration curve is not drawn for a string-labeled "
        "classification, which is the ordinary shape of a clinical outcome")
    assert drawn[0]["payload"]["scored_on"] == "held-out rows only"


def test_the_curve_names_the_event_it_is_about(client):
    """`predict_proba[:, 1]` is `classes_[1]`, and with string labels that is
    whichever sorts second. A curve binarized against the other class is the
    complementary event drawn confidently, and every annotation number would be
    wrong in a way no reader could detect."""
    pid, project, run = _trained(client, "clinic_visits.csv", "outcome",
                                 models=["logreg"])
    result = run.results[0]
    assert result.positive_label == "responder", result.positive_label

    y_true, y_proba, name, event = figure_bundle.predictions_for(project)
    assert event == "responder"
    assert set(y_true) <= {0, 1}
    raw = training.y_true_for(project)
    assert sum(y_true) == sum(1 for v in raw if v == "responder"), (
        "the binarization does not agree with the label it claims to be about")

    figures = client.get(f"/project/{pid}/figures").json()
    drawn = [f for f in figures["admitted"] + figures["held"]
             if f["id"] == "calibration"][0]
    assert drawn["payload"]["event"] == "responder", (
        "the curve does not say which event it is about, so a reader cannot "
        "tell it from its mirror image")

    # THE POSITIVE CONTROL. Binarizing against the other class must give a
    # different curve, or this fixture cannot tell the two apart and the
    # assertion above would hold whichever class had been chosen.
    from turbotab import figure_specs as _specs

    mirrored = [0 if v else 1 for v in y_true]
    assert (_specs.calibration_payload(y_true, y_proba)["c_statistic"]
            != _specs.calibration_payload(mirrored, y_proba)["c_statistic"]), (
        "the C-statistic is the same against either class on this fixture, so "
        "the claim above would pass whichever one the run had recorded")


def test_a_run_that_cannot_say_which_class_draws_nothing(client):
    """**Return nothing rather than a wrong value.** Where the positive class
    was not recorded, the curve is not drawn against a guess — guessing `1`
    would be right on a 0/1 target and silently wrong on every other one."""
    pid, project, run = _trained(client, "clinic_visits.csv", "outcome",
                                 models=["logreg"])
    assert figure_bundle.predictions_for(project) is not None    # control
    for result in run.results:
        result.positive_label = None
    assert figure_bundle.predictions_for(project) is None, (
        "the curve was drawn against an assumed positive class")


def test_the_shapes_this_file_does_not_cover_are_named():
    """**The fixture rule's own honesty clause** (`GUIDED-097`).

    Two fixtures of different target shape is the requirement; naming the
    shapes NOT covered is what stops the requirement becoming the next
    `leaky_sepsis.csv`. A sweep that reports only what it covered has not
    reported its coverage.
    """
    shapes = {kind for _, _, kind in TARGET_SHAPES.values()}
    assert len(TARGET_SHAPES) >= 2 and len(shapes) >= 2, (
        "the Train claims here run against one target shape, which is exactly "
        "how GUIDED-093 survived two loops")
    fixtures = {name for name, _, _ in TARGET_SHAPES.values()}
    assert len(fixtures) >= 2, fixtures
    assert SHAPES_NOT_COVERED, (
        "no uncovered shape is named, which reads as complete coverage")
    for shape, reason in SHAPES_NOT_COVERED.items():
        assert len(reason) > 80, f"{shape}: the reason is a shrug"
        assert shape not in TARGET_SHAPES, (
            f"{shape} is declared uncovered and is covered — a stale "
            "exclusion is worse than none")


def test_every_fixture_this_file_claims_a_target_shape_for_has_it():
    """The declarations have to stay true of the files. A fixture whose target
    quietly changed dtype would make every claim above about something else."""
    for shape, (fixture, target, task) in TARGET_SHAPES.items():
        frame = pd.read_csv(DATA / fixture)
        assert target in frame.columns, (shape, target)
        values = frame[target].dropna()
        numeric = pd.api.types.is_numeric_dtype(values)
        if shape == "binary_numeric":
            assert numeric and set(values.unique()) <= {0, 1}, values.unique()
        elif shape == "binary_string":
            assert not numeric and len(values.unique()) == 2, values.unique()
        else:
            assert numeric and values.nunique() > 20, values.nunique()
