"""The signed change from the per-fold scores the record keeps (SWARM_PLAN §7; ``signed.py``),
against numbers derived by hand and against the engine's own corrected repeated k-fold t
(``turbotab.core.models.validation.corrected_t``), a separate implementation of the paired case."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from research.swarm import signed as S


def score(folds, repeats=1, low=None, high=None, estimator="fold_mean"):
    return {"folds": list(folds), "repeats": repeats, "estimator": estimator, "ci_low": low,
            "ci_high": high}


def test_a_paired_change_is_the_corrected_repeated_k_fold_t_by_hand():
    """Differences 1, 2, 3, 4, 5 over one repeat of five folds: mean 3, s² = 2.5, se =
    √((1/5 + 1/4)·2.5) = √1.125, on 4 degrees of freedom."""
    before = score([10, 10, 10, 10, 10], low=9.0, high=11.0)
    after = score([11, 12, 13, 14, 15])
    got = S.change(after, before, same_folds=True, higher_is_better=True)
    se = math.sqrt(1.125)
    q = stats.t.ppf(0.975, 4)
    assert got["paired"] is True and got["difference"] == pytest.approx(3.0)
    assert got["se"] == pytest.approx(se) and got["df"] == 4
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((3 - q * se, 3 + q * se))
    assert got["better"] is True and got["excludes_zero"] is True
    assert got["half_width"] == pytest.approx(1.0) and got["why"] is None


def test_a_paired_change_agrees_with_the_engines_own_corrected_t():
    from turbotab.core.models.validation import corrected_t

    rng = np.random.default_rng(7)
    r, k = 10, 5
    a, b = rng.normal(0.70, 0.03, r * k), rng.normal(0.69, 0.03, r * k)
    got = S.change(score(a, repeats=r), score(b, repeats=r), same_folds=True,
                   higher_is_better=True)
    engine = corrected_t(list(a), list(b), test_share=None, repeats=r)
    assert got["difference"] == pytest.approx(engine.mean)
    assert got["se"] == pytest.approx(engine.se)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((engine.ci_low, engine.ci_high))


def test_an_unpaired_change_inflates_each_arm_by_the_same_factor_with_welchs_degrees():
    """After 1–5 (s² = 2.5) against a constant before: each arm's variance of its mean is
    (1/5 + 1/4)·s², so 1.125 and 0; the difference 3 with se √1.125 on Welch's 4 degrees."""
    got = S.change(score([1, 2, 3, 4, 5]), score([0, 0, 0, 0, 0]), same_folds=False,
                   higher_is_better=True)
    se = math.sqrt(1.125)
    q = stats.t.ppf(0.975, 4)
    assert got["paired"] is False and got["difference"] == pytest.approx(3.0)
    assert got["se"] == pytest.approx(se) and got["df"] == pytest.approx(4.0)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((3 - q * se, 3 + q * se))


def test_an_unpaired_change_over_two_varying_arms_by_hand():
    a, b = [0.70, 0.72, 0.75, 0.71, 0.74], [0.60, 0.66, 0.61, 0.69, 0.64, 0.62, 0.65, 0.63, 0.67, 0.68]
    got = S.change(score(a), score(b, repeats=2), same_folds=False, higher_is_better=True)
    va = (1 / 5 + 1 / 4) * np.var(a, ddof=1)
    vb = (1 / 10 + 1 / 4) * np.var(b, ddof=1)  # two repeats of five folds: K = 5
    df = (va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 9)
    se = math.sqrt(va + vb)
    diff = np.mean(a) - np.mean(b)
    q = stats.t.ppf(0.975, df)
    assert got["difference"] == pytest.approx(diff) and got["se"] == pytest.approx(se)
    assert got["df"] == pytest.approx(df)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((diff - q * se, diff + q * se))
    assert got["folds"] == [5, 10]


def test_a_loss_is_better_lower():
    got = S.change(score([0.5, 0.4, 0.45, 0.42, 0.41]), score([0.6, 0.55, 0.58, 0.57, 0.6]),
                   same_folds=True, higher_is_better=False)
    assert got["difference"] < 0 and got["better"] is True


def test_no_change_is_neither_better_nor_worse():
    same = score([1, 2, 3, 4, 5])
    got = S.change(same, same, same_folds=True, higher_is_better=True)
    assert got["difference"] == 0 and got["better"] is None and got["excludes_zero"] is False


@pytest.mark.parametrize("after,before,same,why", [
    (score(range(10), repeats=2), score(range(5)), True, "not the same folds"),
    (score([1]), score([1]), False, "no k-fold mean"),
    (score([1, 2, 3, 4], estimator="optimism"), score([1, 2, 3, 4]), False, "no k-fold mean"),
    (score([1, None, None, None]), score([1, 2, 3, 4]), True, "Fewer than two folds"),
    (score([1, None, None, None]), score([1, 2, 3, 4]), False, "fewer than two fold scores"),
    (score([1, 2, 3, 4], repeats=2), score([1, 2, 3, 4, 5, 6, 7, 8], repeats=2), True,
     "not the same scheme"),
])
def test_a_change_the_folds_cannot_give_has_no_number_and_says_why(after, before, same, why):
    got = S.change(after, before, same_folds=same, higher_is_better=True)
    assert got["difference"] is None and why in got["why"]


def test_folds_per_repeat_reads_k_from_the_score():
    assert S.folds_per_repeat(score(range(50), repeats=10)) == 5
    assert S.folds_per_repeat(score(range(5))) == 5
    assert S.folds_per_repeat(score(range(7), repeats=2)) is None
    assert S.folds_per_repeat({"folds": None}) is None
