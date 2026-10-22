"""The signed change from the per-fold scores the record keeps (SWARM_PLAN §7; ``signed.py``),
against numbers derived by hand and against the engine's own corrected repeated k-fold t
(``turbotab.core.models.validation.corrected_t``), a separate implementation of the paired case,
with the share of rows each fold scores the engine measures on folds it draws itself
(``stages.rows.draw_split``, ``metrics.CrossValidated.test_share``). Where those facts do not fix
the share (time-ordered, internal–external, grouped folds), or a pooled score is not the mean of
its fold scores, the change is refused, and each guard is tested both ways."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from research.swarm import signed as S


def score(folds, repeats=1, low=None, high=None, estimator="fold_mean"):
    return {"folds": list(folds), "repeats": repeats, "estimator": estimator, "ci_low": low,
            "ci_high": high}


def split(n=100, k=5, repeats=1, scheme="random", validation=None, grouped_by=None):
    """The split stage's facts for ``n`` training rows in ``k`` random folds: with ``n`` a
    multiple of ``k`` every fold scores n/k rows and is fit on the rest, so n_test/n_train is
    exactly 1/(k − 1) (1/4 for five folds)."""
    return {"n_train": n, "folds": k, "repeats": repeats, "fold_scheme": scheme,
            "validation": validation or ("repeated_kfold" if repeats > 1 else "kfold"),
            "grouped_by": grouped_by}


FIVE = split()


def test_a_paired_change_is_the_corrected_repeated_k_fold_t_by_hand():
    """Differences 1, 2, 3, 4, 5 over one repeat of five folds: mean 3, s² = 2.5, se =
    √((1/5 + 1/4)·2.5) = √1.125, on 4 degrees of freedom."""
    before = score([10, 10, 10, 10, 10], low=9.0, high=11.0)
    after = score([11, 12, 13, 14, 15])
    got = S.change(after, before, same_folds=True, higher_is_better=True, after_split=FIVE)
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
                   higher_is_better=True, after_split=split(repeats=r))
    engine = corrected_t(list(a), list(b), test_share=None, repeats=r)
    assert got["difference"] == pytest.approx(engine.mean)
    assert got["se"] == pytest.approx(engine.se)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((engine.ci_low, engine.ci_high))


def test_an_unpaired_change_inflates_each_arm_by_the_same_factor_with_welchs_degrees():
    """After 1–5 (s² = 2.5) against a constant before: each arm's variance of its mean is
    (1/5 + 1/4)·s², so 1.125 and 0; the difference 3 with se √1.125 on Welch's 4 degrees."""
    got = S.change(score([1, 2, 3, 4, 5]), score([0, 0, 0, 0, 0]), same_folds=False,
                   higher_is_better=True, after_split=FIVE, before_split=FIVE)
    se = math.sqrt(1.125)
    q = stats.t.ppf(0.975, 4)
    assert got["paired"] is False and got["difference"] == pytest.approx(3.0)
    assert got["se"] == pytest.approx(se) and got["df"] == pytest.approx(4.0)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((3 - q * se, 3 + q * se))


def test_an_unpaired_change_over_two_varying_arms_by_hand():
    a, b = [0.70, 0.72, 0.75, 0.71, 0.74], [0.60, 0.66, 0.61, 0.69, 0.64, 0.62, 0.65, 0.63, 0.67, 0.68]
    got = S.change(score(a), score(b, repeats=2), same_folds=False, higher_is_better=True,
                   after_split=FIVE, before_split=split(repeats=2))
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
                   same_folds=True, higher_is_better=False, after_split=FIVE)
    assert got["difference"] < 0 and got["better"] is True


def test_no_change_is_neither_better_nor_worse():
    """Differences +1, −1, +2, −2, 0: a mean of exactly zero, with a spread."""
    got = S.change(score([2, 1, 5, 2, 5]), score([1, 2, 3, 4, 5]), same_folds=True,
                   higher_is_better=True, after_split=FIVE)
    assert got["difference"] == 0 and got["better"] is None and got["excludes_zero"] is False
    assert got["se"] > 0


# ── differences with no spread (the verifier's V2) ───────────────────────────

BEFORE = [0.70, 0.72, 0.69, 0.71, 0.73]


@pytest.mark.parametrize("after,before,shift", [
    ([b + 0.01 for b in BEFORE], BEFORE, 0.01),  # the verifier's case: exactly equal in float64
    # +0.3 on these scores leaves differences apart by float64's rounding alone (s² ≈ 2e-33)
    ([b + 0.3 for b in [0.1, 0.2, 0.3, 0.7, 0.9]], [0.1, 0.2, 0.3, 0.7, 0.9], 0.3),
    (BEFORE, BEFORE, 0.0),  # no fold's score changed
])
def test_a_paired_change_whose_differences_have_no_spread_is_refused(after, before, shift):
    """Every fold's difference the same: s²_d is zero (or float64's rounding of the scores), so
    the corrected interval has no width and certifies nothing; the engine's own paired movement
    refuses the same input (``runs.movement``). The change has no number, and says why."""
    d = np.subtract(after, before)
    assert np.ptp(d) <= 1e-15 and d.mean() == pytest.approx(shift, abs=1e-12)  # the construction
    got = S.change(score(after, low=0.65, high=0.75), score(before, low=0.66, high=0.74),
                   same_folds=True, higher_is_better=True, after_split=FIVE)
    assert got["difference"] is None and got["excludes_zero"] is None and got["se"] is None
    assert "no fold-to-fold spread" in got["why"]
    assert got["half_width"] == pytest.approx(0.04)  # the earlier fit's own, as for any change


@pytest.mark.parametrize("after,before", [([0.8] * 5, [0.7] * 5), ([0.8] * 5, [0.8] * 5)])
def test_an_unpaired_change_whose_arms_both_have_no_spread_is_refused(after, before):
    got = S.change(score(after), score(before), same_folds=False, higher_is_better=True,
                   after_split=FIVE, before_split=FIVE)
    assert got["difference"] is None and got["excludes_zero"] is None and got["df"] is None
    assert "no fold-to-fold spread" in got["why"]


def test_differences_with_a_spread_keep_their_number():
    """The same scores with one fold moved by 0.001 more: a spread, so a number (paired), and an
    unpaired change with one arm constant keeps Welch's interval on the other arm's spread."""
    after = [b + 0.01 for b in BEFORE]
    after[2] += 0.001
    got = S.change(score(after), score(BEFORE), same_folds=True, higher_is_better=True,
                   after_split=FIVE)
    d = np.subtract(after, BEFORE)
    se = math.sqrt((1 / 5 + 1 / 4) * np.var(d, ddof=1))
    assert got["why"] is None and got["difference"] == pytest.approx(d.mean())
    assert got["se"] == pytest.approx(se) and got["se"] > 0
    one = S.change(score([0.8] * 5), score(BEFORE), same_folds=False, higher_is_better=True,
                   after_split=FIVE, before_split=FIVE)
    assert one["why"] is None and one["df"] == pytest.approx(4.0)
    assert one["se"] == pytest.approx(math.sqrt((1 / 5 + 1 / 4) * np.var(BEFORE, ddof=1)))


FOUR, TWO_BY_FOUR, TWO_BY_TWO = split(k=4), split(k=4, repeats=2), split(k=2, repeats=2)


@pytest.mark.parametrize("after,before,same,splits,why", [
    (score(range(10), repeats=2), score(range(5)), False, (split(repeats=2), FIVE),
     None),  # unpaired: two repeats against one is a number
    (score(range(10), repeats=2), score(range(5)), True, (FIVE, FIVE), "draws 1 × 5"),
    (score([1]), score([1]), False, (FIVE, FIVE), "no k-fold score"),
    (score([1, 2, 3, 4], estimator="optimism"), score([1, 2, 3, 4]), False, (FOUR, FOUR),
     "no k-fold score"),
    (score([1, None, None, None]), score([1, 2, 3, 4]), True, (FOUR, FOUR),
     "Fewer than two folds"),
    (score([1, None, None, None]), score([1, 2, 3, 4]), False, (FOUR, FOUR),
     "fewer than two fold scores"),
    (score([1, 2, 3, 4], repeats=2), score([1, 2, 3, 4, 5, 6, 7, 8], repeats=2), True,
     (TWO_BY_TWO, TWO_BY_FOUR), "draws 2 × 2"),
    (score([1, 2, 3, 4, 5]), score([1, 2, 3, 4, 5]), True, (None, None), "not read"),
])
def test_a_change_the_folds_cannot_give_has_no_number_and_says_why(after, before, same, splits,
                                                                    why):
    got = S.change(after, before, same_folds=same, higher_is_better=True,
                   after_split=splits[0], before_split=splits[1])
    if why is None:
        assert got["difference"] is not None and got["why"] is None
    else:
        assert got["difference"] is None and why in got["why"] and got["test_share"] is None


def test_folds_per_repeat_reads_k_from_the_score():
    assert S.folds_per_repeat(score(range(50), repeats=10)) == 5
    assert S.folds_per_repeat(score(range(5))) == 5
    assert S.folds_per_repeat(score(range(7), repeats=2)) is None
    assert S.folds_per_repeat({"folds": None}) is None


# ── the share, as the engine measures it on folds it draws (the verifier's P4) ──


def engine_folds(n, k, *, seed=3, repeats=1, y=None, groups=None, order=None, clusters=None):
    """The engine's own split of ``n`` rows (``draw_split``, none held out) and, on its folds, the
    share of rows each fold scores over the rows it is fit on as the fit measures it
    (``metrics.repeated_pairs``, ``CrossValidated.test_share``)."""
    from turbotab.core.models.metrics import CrossValidated, repeated_pairs
    from turbotab.core.stages.modeling import repeat_columns
    from turbotab.core.stages.rows import draw_split

    frame, info = draw_split(np.arange(n), holdout=0.0, seed=seed, folds=k, y=y, groups=groups,
                             grouped_by="unit" if groups is not None else None, order=order,
                             repeats=repeats, clusters=clusters,
                             cluster_column="site" if clusters is not None else None)
    train = frame["partition"].to_numpy() == "train"
    columns = [frame.loc[train, c].to_numpy() for c in ["fold", *repeat_columns(frame)]]
    pairs, _ = repeated_pairs(columns, info["fold_scheme"])
    measured = CrossValidated(per_fold=[], sizes=[(int(f.sum()), int(t.sum()))
                                                  for _, f, t in pairs]).test_share
    validation = ("internal_external" if clusters is not None
                  else "repeated_kfold" if repeats > 1 else "kfold")
    return {**info, "validation": validation}, measured, len(pairs)


@pytest.mark.parametrize("n,k,repeats,stratified", [
    (100, 5, 1, False), (103, 5, 1, False), (61, 3, 1, True), (157, 10, 3, True),
    (103, 5, 4, False),
])
def test_the_share_on_random_folds_is_the_one_the_engine_measures_on_its_own_folds(
        n, k, repeats, stratified):
    y = np.array(["yes" if i % 3 == 0 else "no" for i in range(n)]) if stratified else None
    facts, measured, scored = engine_folds(n, k, repeats=repeats, y=y)
    assert scored == k * repeats
    share, why = S.test_share(S.split_facts(facts))
    assert why is None and share == pytest.approx(measured, rel=1e-12)
    if n % k:
        assert share != pytest.approx(1 / (k - 1), rel=1e-9)  # the sizes differ: not 1/(k − 1)


def test_a_paired_change_on_folds_of_unequal_size_is_the_engines_corrected_t_with_its_share():
    from turbotab.core.models.validation import corrected_t

    facts, measured, _ = engine_folds(103, 5, repeats=4)
    rng = np.random.default_rng(11)
    a, b = rng.normal(0.70, 0.03, 20), rng.normal(0.69, 0.03, 20)
    got = S.change(score(a, repeats=4), score(b, repeats=4), same_folds=True,
                   higher_is_better=True, after_split=S.split_facts(facts))
    engine = corrected_t(list(a), list(b), test_share=measured, repeats=4)
    assert got["difference"] == pytest.approx(engine.mean, rel=1e-12)
    assert got["se"] == pytest.approx(engine.se, rel=1e-12) and got["df"] == engine.df
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((engine.ci_low, engine.ci_high),
                                                            rel=1e-12)
    assert got["test_share"] == pytest.approx([measured, measured], rel=1e-12)


def test_the_bootstraps_validation_scores_its_headline_on_random_folds_and_is_changed():
    """Under the bootstrap's validation the split draws one run of random K-fold, as for "kfold"
    (its optimism is estimated beside it): the share is the engine's measured one on those folds.
    The dry run's exploring run takes it, the seal plan's first option at its size."""
    from turbotab.core.models.validation import corrected_t

    facts, measured, scored = engine_folds(103, 5)
    facts = {**facts, "validation": "bootstrap"}
    share, why = S.test_share(S.split_facts(facts))
    assert why is None and scored == 5 and share == pytest.approx(measured, rel=1e-12)
    a, b = [0.50, 0.52, 0.49, 0.55, 0.53], [0.52, 0.55, 0.50, 0.58, 0.54]
    got = S.change(score(a), score(b), same_folds=True, higher_is_better=False,
                   after_split=S.split_facts(facts))
    engine = corrected_t(a, b, test_share=measured, repeats=1, higher_is_better=False)
    assert got["se"] == pytest.approx(engine.se, rel=1e-12)
    assert got["difference"] == pytest.approx(-engine.mean, rel=1e-12)


def test_time_ordered_folds_are_refused_since_their_share_is_not_the_k_folds():
    """Five time-ordered folds give four scores, each fold fit on the folds before it: the share
    the engine measures (0.52 here) is not 1/(K − 1), and the record holds no fold's size, so the
    change has no number rather than the wrong one."""
    facts, measured, scored = engine_folds(120, 4, order=np.arange(120, dtype=float))
    assert facts["fold_scheme"] == "time_ordered" and scored == facts["folds"] == 4
    assert measured != pytest.approx(1 / (scored - 1), rel=1e-3)
    a = score([0.80, 0.82, 0.79, 0.85])
    b = score([0.78, 0.80, 0.80, 0.80])
    got = S.change(a, b, same_folds=True, higher_is_better=True, after_split=S.split_facts(facts))
    assert got["difference"] is None and "time-ordered" in got["why"]
    unpaired = S.change(a, b, same_folds=False, higher_is_better=True, after_split=FOUR,
                        before_split=S.split_facts(facts))
    assert unpaired["difference"] is None and "time-ordered" in unpaired["why"]


def test_internal_external_folds_are_refused_since_a_site_is_a_fold_of_its_own_size():
    clusters = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5, dtype=object)
    facts, measured, scored = engine_folds(100, 5, clusters=clusters)
    assert scored == 4 and measured != pytest.approx(1 / 3, rel=1e-3)
    got = S.change(score([0.7, 0.71, 0.69, 0.72]), score([0.7, 0.7, 0.7, 0.7]), same_folds=True,
                   higher_is_better=True, after_split=S.split_facts(facts))
    assert got["difference"] is None and "internal_external" in got["why"]


def test_folds_that_keep_a_units_rows_together_are_refused():
    groups = np.repeat(np.arange(40), [1, 2, 3, 4] * 10)
    facts, _, _ = engine_folds(len(groups), 5, groups=groups)
    assert facts["grouped_by"] == "unit"
    got = S.change(score([0.7, 0.71, 0.69, 0.72, 0.7]), score([0.7] * 5), same_folds=True,
                   higher_is_better=True, after_split=S.split_facts(facts))
    assert got["difference"] is None and "together" in got["why"]


# ── a numeric outcome's pooled MSE (the verifier's P3) ───────────────────────


def regression_fits(n, k=5, seed=5):
    """Two fits of a numeric outcome through the engine's own cross-validation on its own folds:
    least squares on x, and the training rows' mean; each with the summary the fit record keeps
    (``metrics.summarize``) and its out-of-fold predictions."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    from turbotab.core.models.metrics import cross_validate, fold_pairs

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 1.5 * X[:, 0] + rng.normal(size=n)
    facts, measured, _ = engine_folds(n, k)
    from turbotab.core.stages.rows import draw_split

    frame, _ = draw_split(np.arange(n), holdout=0.0, seed=3, folds=k)
    pairs = fold_pairs(frame["fold"].to_numpy())
    out = {}
    for name, make in (("linear", LinearRegression), ("mean", DummyRegressor)):
        cv = cross_validate("regression", make, X, y, pairs, keep_predictions=True)
        rows, truth, oof = cv.out_of_fold()
        out[name] = (cv.summary("regression")["mse"], float(np.mean((truth - oof) ** 2)))
    return out, S.split_facts(facts), measured


def test_a_pooled_mse_on_folds_of_one_size_changes_by_the_pooled_difference():
    """100 rows in five folds of 20: the pooled MSE is the mean of the fold MSEs, so the change is
    the difference of the two pooled MSEs (numpy, over every out-of-fold prediction), and its
    interval is the engine's corrected t on the fold MSEs."""
    from turbotab.core.models.validation import corrected_t

    fits, facts, measured = regression_fits(100)
    (after, after_np), (before, before_np) = fits["linear"], fits["mean"]
    assert after["estimator"] == before["estimator"] == "pooled"
    got = S.change(after, before, same_folds=True, higher_is_better=False, metric="mse",
                   after_split=facts)
    assert got["why"] is None
    assert got["difference"] == pytest.approx(after_np - before_np, rel=1e-10)
    assert got["better"] is True  # least squares on the signal beats the mean
    engine = corrected_t(after["folds"], before["folds"], test_share=measured, repeats=1,
                         higher_is_better=False)  # signed so that a lower loss is positive
    assert got["difference"] == pytest.approx(-engine.mean, rel=1e-10)
    assert got["se"] == pytest.approx(engine.se, rel=1e-10)
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((-engine.ci_high, -engine.ci_low),
                                                            rel=1e-10)


def test_a_pooled_mse_on_folds_of_different_sizes_is_refused_and_says_why():
    """103 rows in five folds: folds of 21 and 20, so the pooled MSE is their size-weighted mean,
    not the mean of the fold MSEs, which differ by the construction here."""
    fits, facts, _ = regression_fits(103)
    (after, after_np), (before, before_np) = fits["linear"], fits["mean"]
    assert np.mean(after["folds"]) - np.mean(before["folds"]) != pytest.approx(
        after_np - before_np, rel=1e-9)  # the mean of the fold scores is not the score
    got = S.change(after, before, same_folds=True, higher_is_better=False, metric="mse",
                   after_split=facts)
    assert got["difference"] is None and "21 and 20 rows" in got["why"]


def test_a_pooled_score_that_is_no_mean_of_its_folds_is_refused():
    fits, facts, _ = regression_fits(100)
    rmse = {**fits["linear"][0], "estimator": "pooled"}
    got = S.change(rmse, fits["mean"][0], same_folds=True, higher_is_better=False,
                   metric="rmse", after_split=facts)
    assert got["difference"] is None and "no weighted mean" in got["why"]
