"""`AUDIT-003` — the number upstream of every skew-driven transform sentence.

The row is *"a log transform is recommended from skewness alone"*. `L52`
corrected what the app SAYS about that skew (`ml/eda_recommender.py`'s R5 card,
`ml/model_coach.py`'s preprocessing action). This file is about the skew
itself, one level up, where `ml/dataset_profile.py` had:

    if len(valid) > 2:
        try:
            profile.skewness = float(valid.skew())
        except:
            profile.skewness = 0.0

**`0.0` is not a null here.** It is the value of a perfectly symmetric
distribution, returned from a failure to compute one — trap 9's exact shape
("`(None, None)`, never `(0.0, 1.0)` — those are the values of *perfect*
calibration, and returning them from ignorance asserts perfection").

It reaches a person. `pages/06_Train_and_Compare.py:866` renders

    f", target skew {abs(_target_prof.skewness):.2f}"

so the app printed *"target skew 0.00"* about a column nothing measured. And
`ml/dataset_profile.py`'s feature loop gates `highly_skewed_features` on
`abs(fp.skewness) > 1.0`, so an unmeasurable column was asserted symmetric
INTO the set the transform advice is composed from — the row's own subject.

## The correction

`skewness` stays `None`, which is what the field's default already means and
what both of its readers already guard on (`pages/06_Train_and_Compare.py:865`
and `ml/nn_recommender.py:146` both test `is not None`). The app says nothing
rather than saying zero. Nothing is deleted: the measurement is still taken and
still reported wherever it can be taken, and
`test_a_skew_that_can_be_computed_is_still_reported` is the positive control
(`GUIDED-045`) that fails if this were "fixed" by blanking the field.

## Fixture shapes — `GUIDED-097`

| target shape | fixture |
|---|---|
| **int64 regression** | `clinical_risk.csv::length_of_stay_days` (1–20) |
| **float64 regression** | `dietary_recalls.csv::energy_kcal` (257–7,801) |

**Not covered, said out loud.** A **classification** target:
`compute_target_profile` takes its other branch and never computes a skew, so
`skewness` is `None` there already and there is no fabrication to correct. A
column with **fewer than three valid values**: the `len(valid) > 2` gate means
no skew is attempted, which is the same honest `None` by a different route and
is not what this file drives.

## How the failure is provoked

`pandas.Series.skew` is monkeypatched to raise. That is the branch's real
trigger — an `except` clause with no reachable input is a claim nobody can
check, and the honest way to check this one is to make the call fail.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.dataset_profile import (                                 # noqa: E402
    compute_dataset_profile, compute_feature_profile, compute_target_profile,
)

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

INT_TARGET = ("clinical_risk.csv", "length_of_stay_days")
FLOAT_TARGET = ("dietary_recalls.csv", "energy_kcal")


@pytest.fixture()
def skew_cannot_be_computed(monkeypatch):
    """Make `Series.skew` raise, which is the only way into the branch."""
    def _boom(self, *a, **k):
        raise ValueError("skew is not computable on this column")
    monkeypatch.setattr(pd.Series, "skew", _boom, raising=True)


# ── the positive control, first ─────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target", [INT_TARGET, FLOAT_TARGET])
def test_a_skew_that_can_be_computed_is_still_reported(fixture, target):
    """`GUIDED-045`. The correction is to what the app says when it CANNOT
    measure. If it stopped measuring, this fails."""
    frame = pd.read_csv(DATA / fixture)
    tp = compute_target_profile(frame, target, "regression")
    assert tp.skewness is not None, "the skew stopped being measured at all"
    assert isinstance(tp.skewness, float), type(tp.skewness)


# ── the fabrication ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target", [INT_TARGET, FLOAT_TARGET])
def test_an_unmeasurable_target_skew_is_not_reported_as_zero(fixture, target,
                                                             skew_cannot_be_computed):
    """`0.00` is a reading of perfect symmetry. `pages/06_Train_and_Compare.py`
    prints it beside the model choice, so this one is rendered."""
    frame = pd.read_csv(DATA / fixture)
    tp = compute_target_profile(frame, target, "regression")
    assert tp.skewness is None, (
        f"the target's skew could not be computed and the profile reports "
        f"{tp.skewness!r}, which is a measurement of a symmetric distribution")


def test_an_unmeasurable_feature_skew_is_not_reported_as_zero(skew_cannot_be_computed):
    frame = pd.read_csv(DATA / FLOAT_TARGET[0])
    fp = compute_feature_profile(frame, "bmi", len(frame))
    assert fp.skewness is None, (
        f"the feature's skew could not be computed and the profile reports "
        f"{fp.skewness!r}")


def test_an_unmeasurable_column_is_not_asserted_symmetric_into_the_transform_advice(
        skew_cannot_be_computed):
    """The join to this row's own subject. `highly_skewed_features` is the set
    `ml/model_coach.py`'s preprocessing action composes the transform sentence
    from; a fabricated `0.0` placed every unmeasurable column on the "nothing
    to see" side of that gate without anything having looked."""
    frame = pd.read_csv(DATA / FLOAT_TARGET[0])
    profile = compute_dataset_profile(frame, target_col=FLOAT_TARGET[1],
                                      task_type="regression")
    assert profile.feature_profiles, "nothing was profiled at all"
    reported = {c: fp.skewness for c, fp in profile.feature_profiles.items()
                if fp.skewness is not None}
    assert not reported, (
        "no skew was computable and these columns still carry one: "
        + repr(reported))


# ── the readers, and that they were already guarded ─────────────────────────

def test_the_train_page_source_still_guards_the_skew_before_printing_it():
    """A SOURCE read and the name says so (trap 3b). `pages/06_Train_and_Compare.py`
    is frozen Streamlit; what is checked is that the guard the correction
    relies on is still there, so `None` goes quiet rather than raising."""
    page = (ROOT / "pages" / "06_Train_and_Compare.py").read_text(encoding="utf-8")
    assert "_target_prof.skewness is not None" in page, (
        "the Train page prints the target skew without checking it was "
        "measured; None would now raise there")


def test_the_nn_recommender_source_still_guards_the_skew():
    """The other reader. Same shape, same reason."""
    src = (ROOT / "ml" / "nn_recommender.py").read_text(encoding="utf-8")
    assert "target_profile.skewness is not None" in src, src[:0] or (
        "ml/nn_recommender.py reads the skew without checking it was measured")
