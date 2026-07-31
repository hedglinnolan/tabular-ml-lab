"""
Dataset signals must survive copy-on-write.

Copy-on-write is the default from pandas 3. Under it, `DataFrame.values` hands
back a read-only array, so an in-place `np.fill_diagonal(corr_matrix.values, 0)`
raises — and in compute_dataset_signals that raise was caught by a bare
`except Exception: pass` wrapping the whole collinearity block. The result was
not a crash or a warning: collinearity detection silently vanished. No
correlation-cluster insights, no collinearity coaching, no VIF nudge, and
nothing on screen to say anything had been skipped.

CI installs from an unpinned `pandas>=2.0.0` and so runs pandas 3, while a
developer venv pinned earlier runs pandas 2 — which is exactly how this reached
main-adjacent code with a green local suite. These tests force the pandas 3
behavior on whichever version is installed.
"""
import numpy as np
import pandas as pd
import pytest

from ml.eda_recommender import compute_dataset_signals


@pytest.fixture
def copy_on_write():
    """Force copy-on-write, the pandas 3 default, whatever version is here."""
    try:
        previous = pd.options.mode.copy_on_write
    except (AttributeError, KeyError):
        # pandas 3 removed the toggle; the behavior is unconditional there.
        yield
        return
    pd.options.mode.copy_on_write = True
    try:
        yield
    finally:
        pd.options.mode.copy_on_write = previous


def _collinear_frame(n=300, seed=7):
    rng = np.random.default_rng(seed)
    bmi = rng.normal(27, 5, n)
    return pd.DataFrame({
        "bmi": bmi,
        "weight": bmi * 2.9 + rng.normal(0, 0.01, n),   # r ~ 1.00 with bmi
        "waist": bmi * 2.4 + rng.normal(0, 0.01, n),
        "age": rng.normal(50, 12, n),
        "glucose": rng.normal(100, 15, n),
    })


def _signals(df):
    return compute_dataset_signals(
        df, "glucose", "regression", "cross_sectional", None,
        feature_cols=[c for c in df.columns if c != "glucose"],
    )


class TestCollinearitySurvivesCopyOnWrite:

    def test_the_readonly_values_trap_is_real(self):
        """Pin the mechanism, so the fix is not mistaken for cosmetics."""
        corr = _collinear_frame().corr().abs()
        if not hasattr(pd.options.mode, "copy_on_write"):
            pytest.skip("pandas 3: copy-on-write is unconditional")
        previous = pd.options.mode.copy_on_write
        pd.options.mode.copy_on_write = True
        try:
            corr_cow = _collinear_frame().corr().abs()
            assert not corr_cow.values.flags.writeable, (
                "copy-on-write no longer yields a read-only .values — "
                "this test's premise needs revisiting"
            )
        finally:
            pd.options.mode.copy_on_write = previous
        # to_numpy(copy=True), which the fix uses, is always writeable.
        assert corr.to_numpy(dtype=float, copy=True).flags.writeable

    def test_high_corr_pairs_are_found(self, copy_on_write):
        signals = _signals(_collinear_frame())
        pairs = signals.collinearity_summary.get("high_corr_pairs")
        assert pairs, "collinearity analysis was silently skipped under copy-on-write"
        involved = {c for a, b, _ in pairs for c in (a, b)}
        assert {"bmi", "weight", "waist"} <= involved

    def test_max_corr_excludes_the_diagonal(self, copy_on_write):
        """Left in, the diagonal pins max_corr at 1.0 for any frame at all."""
        rng = np.random.default_rng(3)
        independent = pd.DataFrame({
            "a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300),
            "c": rng.normal(0, 1, 300), "glucose": rng.normal(0, 1, 300),
        })
        max_corr = _signals(independent).collinearity_summary.get("max_corr")
        assert max_corr is not None
        assert max_corr < 0.5, f"diagonal leaked into max_corr: {max_corr}"

    def test_matches_the_answer_without_copy_on_write(self):
        """The fix must not change results, only stop them disappearing."""
        df = _collinear_frame()
        if not hasattr(pd.options.mode, "copy_on_write"):
            pytest.skip("pandas 3: cannot compare against the legacy mode")
        previous = pd.options.mode.copy_on_write
        try:
            pd.options.mode.copy_on_write = False
            legacy = _signals(df).collinearity_summary
            pd.options.mode.copy_on_write = True
            cow = _signals(df).collinearity_summary
        finally:
            pd.options.mode.copy_on_write = previous

        assert cow["max_corr"] == pytest.approx(legacy["max_corr"])
        assert {(a, b) for a, b, _ in cow["high_corr_pairs"]} == \
               {(a, b) for a, b, _ in legacy["high_corr_pairs"]}

    def test_signals_still_computed_for_a_clean_frame(self, copy_on_write):
        rng = np.random.default_rng(11)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200),
            "glucose": rng.normal(0, 1, 200),
        })
        signals = _signals(df)
        assert signals.n_rows == 200
        assert "max_corr" in signals.collinearity_summary
        assert signals.collinearity_summary["high_corr_pairs"] == []
