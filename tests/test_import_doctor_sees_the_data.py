"""The Import Doctor has to look at the values, and stay silent only on purpose.

Two gates that over-corrected, and one blind spot:

  - The "a code sits beyond the observations" gate compared candidates against
    the raw min and max, so ONE row above a code block reclassified the whole
    block as observations. A 1-5 Likert with 7=refused, 8=don't know, 9=missing
    plus a single row holding 10 lost all three, and diagnose() then reported
    nothing at all for the column. The principle was right and the test was
    wrong: it has to be a real population on both sides, not one row.

  - The content-hash frame identity fell back to shape + dtypes when a cell was
    unhashable — which is exactly the identity it was written to replace. A
    list column read back from Parquet arrives as numpy.ndarray cells, so two
    versions of that file differing in a numeric column matched, and the stale
    repair was served to the corrected upload.

  - 999999 is the standard top-code in income and expenditure columns and was
    not in the sentinel list at all.
"""
import numpy as np
import pandas as pd
import pytest

from ml.import_doctor import check_numeric_sentinels
from utils.import_ui import _frame_signature


def likert(n_scale=380, codes=(7, 8, 9), n_code=10, n_stray=0, seed=1):
    rng = np.random.default_rng(seed)
    parts = [rng.integers(1, 6, n_scale)]
    for c in codes:
        parts.append(np.full(n_code, c))
    if n_stray:
        parts.append(np.full(n_stray, 10))
    return pd.DataFrame({"likert": np.concatenate(parts).astype(float)})


@pytest.mark.parametrize("n_stray", [0, 1, 2, 3])
def test_a_handful_of_values_above_the_codes_does_not_silence_them(n_stray):
    found = check_numeric_sentinels(likert(n_stray=n_stray))
    assert found, f"{n_stray} row(s) at 10 silenced the whole 7/8/9 code block"
    detail = found[0].detail
    assert all(str(c) in detail for c in (7, 8, 9))


@pytest.mark.parametrize("seed", range(6))
def test_clean_clinical_columns_stay_silent(seed):
    """The false positives the gate was built to stop, still stopped."""
    rng = np.random.default_rng(seed)
    for name, col in [("age", rng.integers(18, 81, 500)),
                      ("systolic", rng.integers(90, 175, 500)),
                      ("diastolic", rng.integers(55, 100, 500)),
                      ("bmi", rng.normal(27, 4, 500)),
                      ("crp", rng.gamma(2, 1.5, 500))]:
        found = check_numeric_sentinels(pd.DataFrame({name: col.astype(float)}))
        assert not found, f"{name} (seed {seed}): {found[0].detail}"


@pytest.mark.parametrize("code,lo,hi", [
    (999.0, 18, 81), (9999.0, 18, 81), (-9.0, 0, 101), (999999.0, 10_000, 90_000),
])
def test_real_sentinels_are_still_caught(code, lo, hi):
    rng = np.random.default_rng(5)
    col = np.r_[rng.integers(lo, hi, 380), np.full(20, code)]
    found = check_numeric_sentinels(pd.DataFrame({"v": col.astype(float)}))
    assert found, f"{code:g} sailed through a {lo}-{hi} column"


@pytest.mark.parametrize("shape", ["continuous", "dense_int", "narrow", "unit", "count"])
def test_a_negative_code_is_caught_whatever_the_column_looks_like(shape):
    """-9 in a column of non-negative measurements is never an observation.

    The distance test puts the threshold at lo - 0.5*spread, which on a
    continuous 0-100 score is -50: -9 cleared the candidate filter, failed the
    distance test, and was averaged into every result.
    """
    rng = np.random.default_rng(4)
    base = {"continuous": rng.uniform(0, 100, 380),
            "dense_int": rng.integers(0, 101, 380).astype(float),
            "narrow": rng.uniform(40, 90, 380),
            "unit": rng.uniform(0, 1, 380),
            "count": rng.integers(0, 31, 380).astype(float)}[shape]
    col = np.r_[base, np.full(20, -9.0)]
    assert check_numeric_sentinels(pd.DataFrame({"v": col})), f"-9 survived a {shape} column"


@pytest.mark.parametrize("seed", range(4))
def test_a_column_that_genuinely_goes_negative_is_left_alone(seed):
    """Change scores and log ratios really do hold -9."""
    rng = np.random.default_rng(seed)
    col = np.r_[rng.normal(0, 3, 380), np.full(20, -9.0)]
    assert not check_numeric_sentinels(pd.DataFrame({"log_ratio": col}))


def test_a_six_digit_top_code_is_a_candidate():
    """999999 = 'refused' is the norm in income columns and was not in the list."""
    from ml.import_doctor import NUMERIC_SENTINELS
    assert 999999.0 in NUMERIC_SENTINELS


# ── frame identity ───────────────────────────────────────────────────────

def frames_with_unhashable_cells():
    v1 = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "glucose": [95.0, 101.0, 88.0],
        "tags": [np.array([1, 2]), np.array([3]), np.array([4, 5])],
    })
    v2 = v1.copy()
    v2["glucose"] = [195.0, 101.0, 88.0]      # the corrected upload
    return v1, v2


def test_the_fallback_signature_still_looks_at_the_values():
    v1, v2 = frames_with_unhashable_cells()
    with pytest.raises(TypeError):
        pd.util.hash_pandas_object(v1, index=False)   # the fallback really is taken
    assert _frame_signature(v1) != _frame_signature(v2), (
        "two versions of the file share a signature, so the stale repair is served")


def test_an_identical_frame_still_matches_itself():
    v1, _ = frames_with_unhashable_cells()
    assert _frame_signature(v1) == _frame_signature(v1.copy())
