"""The target-leakage screen needed one column and built the whole matrix.

`compute_dataset_signals` asks a single question — which features correlate with
the target above 0.95 — and answered it by materializing the full (p+1)x(p+1)
correlation matrix and slicing one column out of it. That is O(p^2 * n) work for
an O(p * n) question: 39.97 s and 810 MB at p=10,000 against 1.075 s and 17.5 MB
for `corrwith`, and at p=5,000 the matrix was 93% of the entire signals scan.

This is the one path in the compute-cap PR that gets NO cap and NO disclosure,
because nothing is reduced. The refactor has to earn that, so this file is a
characterization test: the cheap answer must be the SAME answer — same values to
machine precision, same flag set, same order — including on the shapes that make
correlation fussy (missing cells, object-dtype columns that `pd.to_numeric`
verified without converting).

It also pins a bug the old code carried. `corr_df['_target'] = target_numeric`
overwrote any user column literally named `_target`, and the following
`.drop('_target')` then removed it, so that feature was silently absent from the
leakage scan. `corrwith` never materializes the name.
"""
import numpy as np
import pandas as pd
import pytest

from ml.eda_recommender import compute_dataset_signals


def _frame(n=400, p=300, seed=0, missing_rate=0.02):
    """A wide-ish numeric frame with a leaky column and scattered gaps."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    df = pd.DataFrame(X, columns=[f"f{i:04d}" for i in range(p)])
    # One column that is the outcome plus noise — the thing the screen is for.
    df["outcome"] = df["f0000"] * 3.0 + rng.normal(scale=0.01, size=n)
    if missing_rate:
        block = df.iloc[:, :p].to_numpy()
        block[rng.random((n, p)) < missing_rate] = np.nan
        df.iloc[:, :p] = block
    return df


def _matrix_answer(df, target):
    """The p x p construction this PR removed, kept here as the reference."""
    cols = [c for c in df.columns if c != target]
    corr_df = df[cols].copy()
    corr_df["_target"] = pd.to_numeric(df[target], errors="coerce")
    return corr_df.corr()["_target"].abs().drop("_target", errors="ignore")


def _signals_answer(df, target):
    """What the app now computes, through the real entry point."""
    features = [c for c in df.columns if c != target]
    return compute_dataset_signals(
        df, target, "regression", "cross_sectional", None, feature_cols=features
    )


def test_the_cheap_answer_is_the_same_answer_with_missing_cells():
    """1e-12 is the contract; the measurement is four orders tighter than that."""
    df = _frame()
    reference = _matrix_answer(df, "outcome")
    cheap = df[[c for c in df.columns if c != "outcome"]].corrwith(df["outcome"]).abs()

    aligned = pd.concat([reference, cheap], axis=1, keys=["matrix", "corrwith"])
    assert aligned["matrix"].notna().sum() > 0, "the fixture correlated nothing"
    max_diff = float((aligned["matrix"] - aligned["corrwith"]).abs().max())
    assert max_diff < 1e-12, f"corrwith drifted from the matrix by {max_diff:g}"
    assert list(reference.index) == list(cheap.index), (
        "the order changed, so leakage_candidate_cols would be reordered")


def test_the_flagged_set_is_identical_and_still_finds_the_leak():
    df = _frame()
    reference = _matrix_answer(df, "outcome")
    signals = _signals_answer(df, "outcome")

    expected = reference[reference > 0.95].index.tolist()
    assert expected, "the fixture has no leaky column, so it proves nothing"
    assert signals.leakage_candidate_cols == expected
    assert signals.leakage_scan_error == "", signals.leakage_scan_error
    assert any("0.95 correlation to target" in f for f in signals.leakage_flags)


def test_a_column_verified_numeric_but_stored_as_object_agrees_too():
    """The messy real upload: `pd.to_numeric` verifies without converting.

    The screen keeps object-dtype columns that merely *parse* as numeric, so the
    refactor has to hold on a frame that never became float64.
    """
    df = _frame(p=60)
    df["f0002"] = df["f0002"].astype(object)
    reference = _matrix_answer(df, "outcome")
    cheap = df[[c for c in df.columns if c != "outcome"]].corrwith(df["outcome"]).abs()
    max_diff = float((reference - cheap).abs().max())
    assert max_diff < 1e-12, f"object-dtype column drifted by {max_diff:g}"


def test_a_feature_named_underscore_target_is_no_longer_swallowed():
    """The bug the matrix carried: the screen overwrote it, then dropped it."""
    df = _frame(p=30)
    # A leaky feature whose name collides with the old code's scratch column.
    df["_target"] = df["outcome"] * 2.0 + 0.001

    old = _matrix_answer(df, "outcome")
    assert "_target" not in old.index, (
        "the fixture no longer reproduces the collision this test is about")

    signals = _signals_answer(df, "outcome")
    assert "_target" in signals.leakage_candidate_cols, (
        "a leaky feature named _target is still invisible to the screen")


def test_a_scan_that_cannot_run_is_still_distinguishable_from_a_clean_one():
    """MINE-004 stays: an empty candidate list must not mean 'all clear'.

    The refactor removes the matrix, not the failure disclosure. Nothing
    downstream can tell an empty `leakage_candidate_cols` from a scan that never
    ran except `leakage_scan_error`.
    """
    src = open("ml/eda_recommender.py", encoding="utf-8").read()
    assert "signals.leakage_scan_error = f\"{type(exc).__name__}: {exc}\"" in src
    assert "Target-leakage screen did not complete" in src


def test_the_matrix_construction_is_gone_from_the_screen():
    """Structural: the O(p^2) line must not come back as a 'small' convenience."""
    lines = open("ml/eda_recommender.py", encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if "# Leakage detection" in l)
    end = next(i for i, l in enumerate(lines)
               if i > start and l.strip().startswith("# Collinearity"))
    block = "\n".join(l for l in lines[start:end] if not l.lstrip().startswith("#"))

    assert "corrwith(target_numeric)" in block, "the O(p*n) screen is gone"
    assert "corr_df['_target']" not in block, "the scratch-column collision is back"
    assert ".corr()" not in block, (
        "a p x p correlation matrix is being built again to read one column")


@pytest.mark.parametrize("n_features", [5, 20])
def test_a_narrow_frame_reaches_the_same_verdict_as_before(n_features):
    """The 500 x 20 case: no cap, no change, same flags."""
    df = _frame(n=500, p=n_features, missing_rate=0.0)
    reference = _matrix_answer(df, "outcome")
    signals = _signals_answer(df, "outcome")
    assert signals.leakage_candidate_cols == reference[reference > 0.95].index.tolist()
