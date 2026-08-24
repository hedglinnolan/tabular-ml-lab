"""`GUIDED-054` — IRLS returned its last iterate when it had not converged.

`ml/calibration.py`'s `weak_calibration` ran a fixed number of IRLS steps and,
when the loop was exhausted, **returned whatever `beta` happened to hold** with
no signal that it had not converged.

That is not a slow fit reported early. Under complete or quasi-complete
separation the iteration does not converge slowly — it **diverges**: the
coefficients run off toward infinity and the last iterate is an arbitrary point
along that path. It has the type of a measurement and the meaning of a
coordinate in an optimizer's history, and it would have been printed in the
annotation box beside numbers that are real.

**And separation is not an exotic case here.** It is what a very good model on a
small sample produces, which is exactly the situation a calibration plot is
drawn for. The failure mode was therefore concentrated on the runs where the
figure matters most.

## The fix is the one the file already argued for

One line above, the docstring says `(0.0, 1.0)` must not be returned for "could
not compute", because those are the values of *perfect* calibration. Returning
a divergent iterate is the same error with more decimals: a number where there
is no number. Non-convergence now joins every other undefined case and returns
`(None, None)`.

## Two jobs, kept apart

The `annotation_box` checklist item still **fails** when a number is missing,
and it should — the figure is not publication-grade without the intercept and
slope. What changed is that the box **renders the absence**: *"not estimable"*,
with the reason, rather than a blank cell beside five real numbers. A blank
reads as a rendering fault; the app declining to state a quantity it does not
have is a different claim and has to look like one.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.calibration import weak_calibration                           # noqa: E402
from turbotab import figure_specs as FS                               # noqa: E402


def _logistic(n=4000, extreme=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-x)))
    return y, 1.0 / (1.0 + np.exp(-extreme * x))


def _separable(n=60):
    """Complete separation: every non-event below every event, no overlap.

    Written as the shape rather than as random data, because the whole point is
    that the likelihood has no interior maximum — a fixture that merely
    discriminated well would converge and test nothing.
    """
    y = np.r_[np.zeros(n), np.ones(n)]
    p = np.r_[np.linspace(0.001, 0.02, n), np.linspace(0.98, 0.999, n)]
    return y, p


# ── the two cases where a number exists ─────────────────────────────────────

def test_a_calibrated_model_reports_slope_one_and_intercept_zero():
    """The reading that makes the number worth printing at all."""
    intercept, slope = weak_calibration(*_logistic())
    assert intercept is not None and slope is not None
    assert abs(intercept) < 0.15, intercept
    assert abs(slope - 1.0) < 0.15, slope


def test_predictions_twice_too_extreme_report_a_slope_near_a_half():
    """A slope below 1 means predictions that are too extreme, which is the
    signature of overfitting — and the amount is readable, not just the sign.

    Predictions built with twice the true log-odds should report a slope near
    0.5, and that is the arithmetic rather than a tolerance chosen to pass.
    """
    y, _ = _logistic()
    _, too_extreme = _logistic(extreme=2.0)
    intercept, slope = weak_calibration(y, too_extreme)
    assert slope is not None
    assert 0.40 < slope < 0.60, slope
    assert abs(intercept) < 0.2, intercept


# ── the three where none does ───────────────────────────────────────────────

def test_separable_data_reports_no_fit_rather_than_a_divergent_iterate():
    """The defect, at the case that produces it.

    IRLS on separated data diverges. The last iterate is a coordinate in an
    optimizer's history and was being returned as a calibration slope.
    """
    assert weak_calibration(*_separable()) == (None, None)


def test_one_outcome_class_reports_no_fit():
    """Guarded twice, and the probe is what showed it.

    Removing the explicit `len(unique(y)) < 2` check leaves this green, because
    a single outcome class is also perfectly separated and the convergence
    guard catches it. Both are kept: the explicit check is the cheap, legible
    one and states the condition in the reader's terms, and the convergence
    guard is the one that holds when a case nobody enumerated arrives. A
    redundant guard is only waste when the thing it guards is cheap to be wrong
    about, and this one is not.
    """
    assert weak_calibration(np.ones(40), np.linspace(0.1, 0.9, 40)) == (None, None)
    assert weak_calibration(np.zeros(40), np.linspace(0.1, 0.9, 40)) == (None, None)


def test_constant_predictions_report_no_fit():
    """No variation in the predictor means no slope to estimate — and the
    honest report of that is silence, not a slope of zero."""
    y = np.r_[np.ones(20), np.zeros(20)]
    assert weak_calibration(y, np.full(40, 0.4)) == (None, None)


def test_no_undefined_case_ever_reports_perfect_calibration():
    """The rule the module states, checked across every undefined case at once.

    `(0.0, 1.0)` are the values of PERFECT calibration. Any of these returning
    them would be the app reporting an ideal result where it has none, which is
    the governing rule's failure in two floats.
    """
    undefined = [
        _separable(),
        (np.ones(40), np.linspace(0.1, 0.9, 40)),
        (np.r_[np.ones(20), np.zeros(20)], np.full(40, 0.4)),
        (np.array([1.0, 0.0]), np.array([0.6, 0.4])),          # n < 3
    ]
    for y, p in undefined:
        assert weak_calibration(y, p) == (None, None), (y[:3], p[:3])


# ── what the figure does with the absence ───────────────────────────────────

def test_the_annotation_box_renders_the_absence_rather_than_a_blank():
    """A blank cell beside five real numbers reads as a rendering fault.

    The app declining to state a quantity it does not have is a different claim
    and has to look like one — so the row says *not estimable* and carries the
    reason.
    """
    payload = FS.calibration_render(*_separable())
    box = {row["key"]: row for row in payload["annotation_box"]}

    for key in ("calibration_intercept", "calibration_slope"):
        assert box[key]["value"] == "not estimable", box[key]
        assert box[key]["why"], f"{key} states no reason for the absence"
        assert "separation" in box[key]["why"]
        assert box[key]["value"] != "", "a blank reads as a rendering fault"

    # The numbers that DO exist are still numbers, so the absence is legible as
    # an absence rather than as the whole box having failed.
    assert box["n"]["value"] and box["n"]["why"] == ""
    assert box["c_statistic"]["value"] != "not estimable"


def test_the_checklist_still_fails_when_a_number_is_missing():
    """Rendering honestly and passing the checklist are different jobs.

    The figure is not publication-grade without the intercept and slope, so the
    item fails — and it must keep failing, or "render the absence" would have
    quietly become "the absence is fine".
    """
    scored = {r["id"]: r for r in
              FS.CALIBRATION.score(FS.calibration_render(*_separable()))}
    assert scored["annotation_box"]["passed"] is False, (
        "the checklist passed with two of its six numbers missing, so "
        "'render the absence' has quietly become 'the absence is fine'")
    assert scored["annotation_box"]["because"]
    # And the items that do not depend on those two numbers still pass, so the
    # failure is attributed rather than smeared across the checklist.
    assert scored["risk_distribution"]["passed"] is True
    assert scored["no_truncation"]["passed"] is True


def test_the_caption_uses_the_same_words_as_the_box():
    """One vocabulary for one fact. A caption saying "not computed" beside a box
    saying "not estimable" is two claims about one absence."""
    payload = FS.calibration_render(*_separable())
    caption = FS.CALIBRATION.caption(payload)
    assert "not estimable" in caption
    assert "not computed" not in caption
