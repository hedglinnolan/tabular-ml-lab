"""`AUDIT-012` — one outlier number carrying two situations that want opposite advice.

`ml/model_coach.py` recommended Huber Regression with this sentence, and with
this sentence only, for every regression target that tripped the fence:

    "The outcome itself contains outliers (12% of values) — Huber downweights
     extreme residuals so they don't steer the fit. Feature outliers are handled
     in preprocessing instead."

The rate is `ml/outliers.py:44`'s IQR fences, computed at
`ml/dataset_profile.py`. On `clinical_labs.csv::sbp` it counts a systolic
pressure of **0 mmHg** and one of **244 mmHg** as the same kind of thing.

`research/CLINICAL_SURVEY_PACK.md`, Cross-cutting 7, is why they are not:

> Impossible values and abnormal-but-real values are different findings with
> different remedies. An impossible entry is a data error and is repaired; an
> extreme but attainable measurement is the phenomenon under study and is
> modeled. Collapsing them into one "outlier" rate invites the analyst to
> delete the finding.

Huber **downweights** both. For the 0 mmHg that is the wrong remedy (it belongs
on the plausibility card, repaired); for the 244 mmHg it discards the signal the
study is about. And the band that separates them was already computed one module
over — `turbotab/engine.plausibility` → `ml/card_evidence.plausibility_report` —
and nothing on the coaching path read it.

## What the correction is, and what it is not

It is **not** the removal of the Huber recommendation: Huber is still the pick in
all three branches below, and `test_huber_is_still_the_pick_in_every_branch` is
the positive control that says so (`GUIDED-045` — and `PRODUCT_VISION.md`'s "the
shelf is never shortened").

It is the narrower true sentence, per situation:

| situation | fixture | what the sentence must do |
|---|---|---|
| recognized, and some values are impossible | `clinical_labs.csv::sbp` (4 of 288, band 40–300 mmHg) | name the impossible count, name the band, say Huber downweights rather than repairs |
| recognized, and none are | `clinical_labs.csv::dbp` (band 15–220 mmHg) | say the fence hits read as real extremes |
| not recognized at all | `clinical_labs.csv::bnp` | say the app **cannot tell them apart here**, and name why |

The third is the clause the row is really about: where the app cannot make the
distinction it must say so rather than keep the wording that implies it did.

`ml/dataset_profile.py` reports `impossible_count`/`impossible_rate` as **None**,
never `0`, where no band was read — a `0` there would assert a measurement that
was never taken (trap 9).

## Fixture shapes — `GUIDED-097`

Three float64 targets covering all three branches, plus one **constructed
int64** target (`bp_sys` with a 0 sentinel), because no `sample_data` file has
an integer regression target whose IQR rate clears
`compute_target_profile`'s 5% `has_outliers` gate — `longitudinal_visits.csv::bp_sys`
reads 0.0%.

**Not covered:** a classification target (the Huber branch is regression-only,
so there is no sentence to correct), and a target column carrying string
sentinels such as `clinical_labs.csv::ferritin` (`">1500"`), which is `object`
dtype and never reaches `compute_target_profile`'s regression branch.

## Where this lands

`pages/05_Preprocess.py:208` renders `pick.why` verbatim in the Model Coach
container. Guided's `turbotab/models.py:98` calls `model_viability` and not
`select_top_picks`, so the corrected sentence is a **Classic** surface; that is
stated rather than implied, and no test here claims a Guided rendering.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.dataset_profile import (                                     # noqa: E402
    compute_dataset_profile, compute_target_profile, read_target_impossibility,
)
from ml.model_coach import select_top_picks                          # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: The sentence that carried all three situations alike.
BLIND = "The outcome itself contains outliers"

IMPOSSIBLE = ("clinical_labs.csv", "sbp")        # 4 impossible, band 40–300
CLEAN = ("clinical_labs.csv", "dbp")             # fence hits, none impossible
UNRECOGNIZED = ("clinical_labs.csv", "bnp")      # no reference variable


def _int_target_frame() -> pd.DataFrame:
    """An int64 `bp_sys` with entry-error zeros — the shape no fixture has.

    Constructed, and said so. 0 mmHg is below the published floor of 40, so
    this exercises the impossible branch on an integer column end to end.
    """
    rng = np.random.default_rng(12)
    values = rng.integers(105, 145, size=200).astype("int64")
    values[:14] = 0
    return pd.DataFrame({
        "bp_sys": values,
        "age": rng.integers(30, 80, size=200),
        "chol": rng.normal(190, 30, size=200),
    })


def _why(frame: pd.DataFrame, target: str) -> str:
    profile = compute_dataset_profile(frame, target_col=target,
                                      task_type="regression")
    picks, _, _ = select_top_picks(profile)
    huber = [p for p in picks if p.model_key == "huber"]
    assert huber, ("the Huber pick did not fire, so there is no sentence to "
                   f"check: {[p.model_key for p in picks]}")
    return huber[0].why


def _file_why(fixture: str, target: str) -> str:
    return _why(pd.read_csv(DATA / fixture), target)


# ── the reading, before any sentence is composed ────────────────────────────

def test_the_impossible_tier_is_read_for_a_recognized_outcome():
    frame = pd.read_csv(DATA / IMPOSSIBLE[0])
    reading = read_target_impossibility(frame, IMPOSSIBLE[1])
    assert reading["physio_read"] == "matched", reading
    assert reading["physio_variable"] == "bp_sys", reading
    assert reading["impossible_count"] == 4, reading
    assert reading["impossibility_band"][:2] == (40.0, 300.0), reading


def test_an_unrecognized_outcome_reports_no_count_rather_than_zero():
    """Trap 9. `0` would say "nothing here is impossible" about a column the
    reference has never heard of."""
    frame = pd.read_csv(DATA / UNRECOGNIZED[0])
    reading = read_target_impossibility(frame, UNRECOGNIZED[1])
    assert reading["physio_read"] == "unrecognized", reading
    assert reading["impossible_count"] is None, reading
    assert reading["impossible_rate"] is None, reading


def test_the_reading_reaches_the_target_profile_the_coach_is_given():
    """`select_top_picks` sees a `DatasetProfile` and never the frame. If the
    tier is not on `TargetProfile` the coach cannot say it."""
    frame = pd.read_csv(DATA / IMPOSSIBLE[0])
    tp = compute_target_profile(frame, IMPOSSIBLE[1], "regression")
    assert tp.physio_read == "matched"
    assert tp.impossible_count == 4
    assert tp.outlier_rate > tp.impossible_rate > 0, (
        "the two tiers should be different numbers; if they are equal the "
        "fence rate is being reported twice")


# ── the sentence, per situation ─────────────────────────────────────────────

def test_an_impossible_entry_is_not_offered_downweighting_as_its_remedy():
    """The sharpest form. 4 of 288 systolic readings are outside 40–300 mmHg;
    those are entry errors and Huber downweights rather than repairs them."""
    said = _file_why(*IMPOSSIBLE)
    assert BLIND not in said, said
    assert "40–300 mmHg" in said, said
    assert "entry errors" in said, said
    assert "repair them on the plausibility card" in said.lower(), said


def test_a_clean_recognized_outcome_says_the_extremes_read_as_real():
    """The other half of the pair. `dbp` trips the fence and nothing is outside
    15–220 mmHg, so the advice is the same recommendation with the opposite
    reason — which is the distinction the old sentence could not make."""
    said = _file_why(*CLEAN)
    assert "15–220 mmHg" in said, said
    assert "real extremes rather than entry errors" in said, said
    assert "entry errors rather than extreme" not in said, said


def test_an_unrecognized_outcome_says_the_app_cannot_tell_them_apart():
    """The clause this row exists for. Where the distinction cannot be made,
    the sentence says so and names the reason instead of keeping wording that
    implies a check that never ran."""
    said = _file_why(*UNRECOGNIZED)
    assert "cannot tell a physiologically impossible entry" in said, said
    assert "abnormal-but-real" in said, said
    assert "'bnp' matches no variable in the physiologic reference" in said, said
    assert "IQR fence count and nothing more" in said, said


def test_an_integer_outcome_takes_the_same_two_tier_reading():
    """The constructed int64 shape. 14 zeros below the 40 mmHg floor."""
    said = _why(_int_target_frame(), "bp_sys")
    assert BLIND not in said, said
    assert "40–300 mmHg" in said, said
    assert "entry errors" in said, said


# ── the positive control ────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target", [IMPOSSIBLE, CLEAN, UNRECOGNIZED])
def test_huber_is_still_the_pick_in_every_branch(fixture, target):
    """`AUDIT-012` narrows what the app CLAIMS, not what it offers. If a future
    edit satisfies the assertions above by dropping the recommendation, this
    fails — the shelf is never shortened."""
    frame = pd.read_csv(DATA / fixture)
    profile = compute_dataset_profile(frame, target_col=target,
                                      task_type="regression")
    picks, _, _ = select_top_picks(profile)
    assert picks, "no picks at all"
    assert any(p.model_key == "huber" for p in picks), [p.model_key for p in picks]
    assert all(p.why for p in picks), "a pick was left with no reason at all"


def test_the_classic_page_source_still_interpolates_the_pick_reason():
    """A SOURCE read, and the name says so (trap 3b): `pages/05_Preprocess.py`
    is frozen Streamlit and cannot be driven from here. What it checks is that
    the page still interpolates `pick.why` — if that line goes, this row is
    closed against a surface that no longer exists."""
    page = (ROOT / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")
    assert "select_top_picks" in page
    assert "{pick.why}" in page, (
        "the Model Coach container no longer renders a pick's reason")
