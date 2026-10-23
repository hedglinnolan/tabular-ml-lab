"""The comparison substrate's repeats by the training units (the owner's ruling of 2026-10-23,
option 3; ``models/folds.py``): under prediction the families and the no-predictor baseline are
compared on repeated k-fold of at least 5 repeats of K folds where the training units number 5,000
or more, and of at least 10 below, the split's own repeats first and kept where they are more.

Every count expected here is the ruling's, written out (never read from the code): 4,999 training
units keep 10 repeats and 5,000 take 5, counted in units, not rows, where a unit's rows repeat. A
fit on 5,000 training rows pairs its families over the 25 folds it ran, by the corrected repeated
k-fold t written out from Nadeau & Bengio's formula (``references_ms6``) on the artifact's own fold
scores and fold sizes, and says so in its chain; a fit on 4,999 pairs them over 50. On a true null
built here (two exchangeable least-squares models on 5,000 rows, 5 × 5 folds) the interval calls
one ahead in at most the nominal share of tables, as it does at 10 × 5 on small ones
(``test_rf1_sentences``). No test here names the substrate's constants or its claim by its id
(meta-test 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
from turbotab.core.models.folds import comparison_folds
from turbotab.core.models.validation import corrected_t, validation_plan
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.acceptance import references_ms6 as ref
from turbotab.core.tests.stage_harness import Ingested


def _headline(n: int, seed: int, folds: int = 5) -> np.ndarray:
    """One balanced random partition of ``n`` rows into ``folds`` folds."""
    return np.random.default_rng(seed).permutation(np.arange(n) % folds)


def _pairs_of_rows(units: int) -> np.ndarray:
    """Each of ``units`` units holds two rows, in order: 0, 0, 1, 1, …"""
    return np.repeat(np.arange(units), 2)


def _whole_units(columns: list[np.ndarray]) -> bool:
    """Every column puts both rows of each unit (rows 2u and 2u + 1) in one fold."""
    return all(np.array_equal(c[0::2], c[1::2]) for c in columns)


# ── the rule, on comparison_folds ─────────────────────────────────────────────


@pytest.mark.parametrize("validation", ["kfold", "bootstrap"])
def test_from_5000_training_units_the_substrate_takes_5_repeats(validation):
    """5,000 training rows, one row each, and 5,000 units of two rows each: 5 repeats of the 5
    folds, the split's own run first; the units' rows stay together in every repeat."""
    rows = _headline(5_000, 1)
    columns, shared, note = comparison_folds([rows], validation=validation, scheme="random",
                                             n=5_000, folds=5, seed=4)
    assert (len(columns), shared, note) == (5, 1, None)
    assert np.array_equal(columns[0], rows)
    assert all(sorted(np.bincount(c).tolist()) == [1_000] * 5 for c in columns)
    assert len({c.tobytes() for c in columns}) == 5  # five partitions, none drawn twice
    groups = _pairs_of_rows(5_000)
    own = np.repeat(_headline(5_000, 2), 2)
    columns, shared, _ = comparison_folds([own], validation=validation, scheme="random",
                                          n=10_000, groups=groups, folds=5, seed=4)
    assert (len(columns), shared) == (5, 1) and _whole_units(columns)


def test_below_5000_training_units_the_substrate_keeps_10_repeats():
    """4,999 training rows, one row each: 10 repeats. 4,999 units of two rows each, 9,998 rows: the
    units are counted, so 10 repeats too, though the rows number more than 5,000."""
    columns, shared, _ = comparison_folds([_headline(4_999, 1)], validation="kfold",
                                          scheme="random", n=4_999, folds=5, seed=4)
    assert (len(columns), shared) == (10, 1)
    groups = _pairs_of_rows(4_999)
    columns, shared, _ = comparison_folds([np.repeat(_headline(4_999, 2), 2)], validation="kfold",
                                          scheme="random", n=9_998, groups=groups, folds=5, seed=4)
    assert (len(columns), shared) == (10, 1) and _whole_units(columns)


@pytest.mark.parametrize("n,own,want", [
    (5_000, 3, 5), (5_000, 7, 7), (4_999, 3, 10), (4_999, 12, 12)])
def test_the_splits_own_repeats_come_first_and_stay_where_they_are_more(n, own, want):
    """Repeated k-fold: the split's own repeats are the substrate's first, and the substrate runs
    at least the rule's repeats (5 from 5,000 training units, 10 below), more where the split
    drew more."""
    headline = [_headline(n, s) for s in range(own)]
    columns, shared, _ = comparison_folds(headline, validation="repeated_kfold", scheme="random",
                                          n=n, folds=5, seed=4)
    assert (len(columns), shared) == (want, own)
    assert all(np.array_equal(a, b) for a, b in zip(columns, headline))


@pytest.mark.parametrize("n,want", [(5_000, 5), (4_999, 10)])
def test_beside_internal_external_validation_the_substrate_is_drawn_afresh_by_the_rule(n, want):
    """Internal–external validation's folds are a grouping's levels, so none of the substrate's
    repeats is the headline's."""
    sites = np.arange(n) % 4
    columns, shared, _ = comparison_folds([sites], validation="internal_external",
                                          scheme="random", n=n, folds=5, seed=4)
    assert (len(columns), shared) == (want, 0)
    assert all(sorted(np.unique(c).tolist()) == [0, 1, 2, 3, 4] for c in columns)


@pytest.mark.parametrize("n", [5_000, 4_999])
def test_time_ordered_folds_run_once_at_any_size(n):
    columns, shared, note = comparison_folds([_headline(n, 1)], validation="kfold",
                                             scheme="time_ordered", n=n, folds=5, seed=4)
    assert (len(columns), shared) == (1, 1) and note


# ── the validation plan counts the same refits ───────────────────────────────


def test_the_plans_cost_counts_25_comparison_refits_from_5000_units_and_says_when_50():
    """The split question's costs, counted before any holdout is chosen, over every analyzed unit:
    from 5,000 the comparisons' 5 × 5 folds, 10 × 5 should a holdout leave fewer than 5,000 to
    train; below 5,000 the comparisons' 10 × 5, whatever is held out. Repeated k-fold's own 10
    repeats are at least either, so its cost is 50 at both sizes; with 3 of its own on 6,000
    units, 25, and 50 should a holdout leave fewer than 5,000 to train."""
    large = {o.validation: o.cost for o in validation_plan("prediction", 13_921).options}
    assert large["bootstrap"] == ("about 500 refits of each family, besides 25 for the "
                                  "comparisons' 5 × 5-fold repeats (50, 10 × 5, if fewer than "
                                  "5,000 rows train)")
    assert large["kfold"] == ("25 refits of each family: the score's 5-fold run is the first of "
                              "the comparisons' 5 × 5-fold repeats (50, 10 × 5, if fewer than "
                              "5,000 rows train)")
    assert large["repeated_kfold"] == "50 refits of each family"
    small = {o.validation: o.cost for o in validation_plan("prediction", 4_999).options}
    assert small["bootstrap"] == ("about 500 refits of each family, besides 50 for the "
                                  "comparisons' 10 × 5-fold repeats")
    assert small["kfold"] == ("50 refits of each family: the score's 5-fold run is the first of "
                              "the comparisons' 10 × 5-fold repeats")
    assert small["repeated_kfold"] == "50 refits of each family"
    three = {o.validation: o.cost for o in validation_plan("prediction", 6_000, repeats=3).options}
    assert three["repeated_kfold"] == ("25 refits of each family (50, 10 × 5, if fewer than 5,000 "
                                       "rows train)")


# ── the fit compares over the folds it ran ───────────────────────────────────


def _regression(n: int, seed: int) -> pd.DataFrame:
    """y = 1 + 0.8·x1 − 0.5·x2 + noise, x3 noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["pid"] = np.arange(n)
    frame["y"] = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    return frame


def _fit(frame: pd.DataFrame, folder) -> tuple[pd.DataFrame, object]:
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "table.csv"
    frame.to_csv(source, index=False)
    state = ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction",
        roles={"pid": "identifier", "x1": "covariate", "x2": "covariate", "x3": "covariate"},
        missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=SplitSpec(holdout=0.0, seed=5, folds=5, validation="kfold"),
        models=["linear", "ridge"])
    table = Ingested(source, folder / "i")
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, state, {"split": split, "target_info": info})
    fit = table.run(fit_stage, state, {"design": design, "split": split, "target_info": info})
    return split.frames["assignment"], fit


@pytest.mark.parametrize("n,repeats", [(5_000, 5), (4_999, 10)])
def test_a_fit_pairs_its_families_over_the_folds_it_ran_by_the_corrected_t(tmp_path, n, repeats):
    """No rows held out, one 5-fold run as the headline: the fit compares linear and ridge
    regression, and each with the baseline, over 5 repeats of the 5 folds on 5,000 training rows
    (25 paired folds, 24 df) and over 10 on 4,999 (50, 49 df). The t, its SE and df are Nadeau &
    Bengio's formula written out on the artifact's fold scores, with every fold's rows trained and
    tested counted on the substrate's own folds, to 10⁻¹⁰; the chain says the repeats run."""
    frame = _regression(n, 8)
    assignment, fit = _fit(frame, tmp_path)
    data = fit.data
    rk = 5 * repeats
    assert (data["comparison"]["folds"], data["comparison"]["repeats"],
            data["comparison"]["shared"]) == (5, repeats, 1)
    models = {m["family"]: m for m in data["models"]}
    for m in models.values():
        assert len(m["compared_on"]["folds"]) == rk and m["compared_on"]["repeats"] == repeats
        assert m["versus_baseline"]["df"] == rk - 1
    cmp = data["comparisons"][0]
    assert (cmp["a"], cmp["b"], cmp["df"], cmp["folds"], cmp["repeats"]) == (
        "linear", "ridge", rk - 1, rk, repeats)
    a = assignment.sort_values("row_id").reset_index(drop=True)
    columns, _, _ = comparison_folds([a["fold"].to_numpy()], validation="kfold", scheme="random",
                                     n=n, folds=5, seed=5)
    n_train = [int((c != k).sum()) for c in columns for k in range(5)]
    n_test = [int((c == k).sum()) for c in columns for k in range(5)]
    t, df, se = ref.corrected_repeated_t(models["linear"]["compared_on"]["folds"],
                                         models["ridge"]["compared_on"]["folds"], n_train, n_test,
                                         lower_is_better=True)
    assert df == rk - 1
    assert abs(cmp["t"] - t) < 1e-10 and abs(cmp["se"] - se) < 1e-10
    assert f"{rk - 1} df, {repeats} × 5 folds" in cmp["sentence"]
    link = next(c for c in data["chain"] if c["relation"] == "families_compared_on_substrate")
    assert link["then"] == ("every family and the no-predictor baseline were compared on 5-fold "
                            f"cross-validation repeated {repeats} times ({rk} paired folds, "
                            f"{rk - 1} degrees of freedom)")


# ── the corrected t at 5 × 5 on a large table, under a true null ────────────


def _null_cv(rng: np.random.Generator, n: int, r: int, K: int = 5):
    """Two least-squares models, each on one noise column, so neither predicts better: their
    fold scores (negative mean squared error) over r repeats of K folds."""
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    out = {0: [], 1: []}
    for _ in range(r):
        folds = rng.permutation(n) % K
        for k in range(K):
            test, train = folds == k, folds != k
            for j in (0, 1):
                X = np.column_stack([np.ones(train.sum()), x[train, j]])
                beta = np.linalg.lstsq(X, y[train], rcond=None)[0]
                pred = beta[0] + beta[1] * x[test, j]
                out[j].append(-float(np.mean((y[test] - pred) ** 2)))
    return out[0], out[1]


def test_at_5_by_5_folds_on_5000_rows_a_true_null_rarely_calls_a_family_ahead():
    """Two exchangeable models on 5,000 rows, 5 repeats of 5 folds, 300 tables: the corrected
    interval (written out here: (1/(rK) + 1/(K − 1))·s², the t quantile on rK − 1 df from scipy,
    checked against ``corrected_t`` on the first table) excludes zero in at most 5% of them."""
    rng = np.random.default_rng(23)
    r, K = 5, 5
    ahead = 0
    for i in range(300):
        a, b = _null_cv(rng, 5_000, r)
        d = np.asarray(a) - np.asarray(b)
        se = np.sqrt((1 / (r * K) + 1 / (K - 1)) * d.var(ddof=1))
        half = stats.t.ppf(0.975, r * K - 1) * se
        lo, hi = d.mean() - half, d.mean() + half
        if i == 0:
            got = corrected_t(a, b, test_share=None, repeats=r)
            assert got.df == r * K - 1
            assert (got.ci_low, got.ci_high) == pytest.approx((lo, hi), rel=1e-12)
        ahead += int(lo > 0 or hi < 0)
    assert ahead / 300 <= 0.05
