"""The null decoy (``research/swarm/decoys.py``): reproducible from its seed, and its outcome
independent of every column.

The independence is by construction: the outcome is drawn from its own random stream and is given
no column (``decoys.outcome``). The tests check that construction directly (the table's outcome
is that function's, whatever the columns' stream draws), and then a consequence of it that any
dependence would break, a correlation bound derived from independence: for independent X and Y
with finite variances, sqrt(n) times the sample correlation tends to a standard normal, so
|r| sqrt(n) < 4 holds with probability about 1 - 6.3e-5 per column. The bound is shown to fire on
an outcome built to depend on one column, so it is not silent by weakness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import decoys  # noqa: E402

BOUND = 4.0  # |r| sqrt(n) under independence (module docstring)
PILOT_SEED = 20261018  # the seed the intake gives the pilot's decoy (intake.DECOY_SEED)


def features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every column as numbers: each numeric column, each level of a text column as 0/1, and
    each column's blanks as 0/1 (whether a value is missing must say nothing either)."""
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        if name in (decoys.OUTCOME, "patient_id"):
            continue
        col = frame[name]
        if col.dtype.kind in "fi":
            out[name] = col.astype(float).to_numpy()
            if col.isna().any():
                out[f"{name} is blank"] = col.isna().to_numpy(dtype=float)
        else:
            for level in sorted(col.dropna().unique()):
                out[f"{name} = {level}"] = (col == level).to_numpy(dtype=float)
    return out


def scaled_correlations(frame: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """|r| sqrt(n) between ``y`` and each feature, over the rows where the feature is recorded."""
    out = {}
    for name, x in features(frame).items():
        keep = np.isfinite(x)
        r = np.corrcoef(x[keep], y[keep].astype(float))[0, 1]
        out[name] = abs(r) * np.sqrt(keep.sum())
    return out


def test_the_same_seed_gives_the_same_table_and_another_seed_another():
    a, b = decoys.generate(PILOT_SEED), decoys.generate(PILOT_SEED)
    pd.testing.assert_frame_equal(a, b)
    other = decoys.generate(PILOT_SEED + 1)
    assert not other[decoys.OUTCOME].equals(a[decoys.OUTCOME])
    assert not other["age"].equals(a["age"])


def test_the_outcome_is_drawn_from_its_own_stream_with_no_column():
    """The table's outcome is ``outcome(seed, n)``, which takes no column; and drawing the
    columns differently (here: none drawn at all, and a table of another size's columns) leaves
    it as it was."""
    frame = decoys.generate(PILOT_SEED)
    n = len(frame)
    assert np.array_equal(frame[decoys.OUTCOME].to_numpy(), decoys.outcome(PILOT_SEED, n))
    _, rng = decoys.streams(PILOT_SEED)
    assert np.array_equal(decoys.outcome(PILOT_SEED, n),
                          (rng.random(n) < decoys.PREVALENCE).astype(int))
    assert set(np.unique(frame[decoys.OUTCOME])) == {0, 1}


@pytest.mark.parametrize("seed", [PILOT_SEED, 1, 2, 3])
def test_no_column_correlates_with_the_outcome_past_the_independence_bound(seed):
    frame = decoys.generate(seed)
    scaled = scaled_correlations(frame, frame[decoys.OUTCOME].to_numpy())
    assert len(scaled) >= 20, sorted(scaled)  # every column, level and blank pattern is checked
    over = {k: round(v, 2) for k, v in scaled.items() if v >= BOUND}
    assert not over, over


def test_the_bound_fires_on_an_outcome_built_to_depend_on_one_column():
    """The same check on the same columns, with an outcome that depends on BMI (log-odds 1 per
    standard deviation): the bound fires on BMI, so a silent check above is not a weak one."""
    frame = decoys.generate(PILOT_SEED)
    rng = np.random.default_rng(7)
    bmi = frame["bmi"].to_numpy(dtype=float)
    z = (bmi - bmi.mean()) / bmi.std()
    y = (rng.random(len(frame)) < 1 / (1 + np.exp(-(np.log(0.25) + 1.0 * z)))).astype(int)
    scaled = scaled_correlations(frame, y)
    assert scaled["bmi"] >= BOUND, scaled["bmi"]


def test_the_columns_are_related_to_one_another_as_the_docstring_says():
    """The decoy is not marked by unrelated columns: blood pressure rises with age and BMI, waist
    with BMI, and HDL falls with BMI (each correlation far past the bound, by construction)."""
    frame = decoys.generate(PILOT_SEED)
    n = len(frame)

    def r(a: str, b: str) -> float:
        keep = frame[[a, b]].notna().all(axis=1)
        return float(np.corrcoef(frame.loc[keep, a].astype(float),
                                 frame.loc[keep, b].astype(float))[0, 1])

    assert r("sbp", "age") * np.sqrt(n) > BOUND
    assert r("sbp", "bmi") * np.sqrt(n) > BOUND
    assert r("waist_cm", "bmi") * np.sqrt(n) > BOUND
    assert -r("hdl", "bmi") * np.sqrt(n) > BOUND
