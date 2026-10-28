"""A coefficient that rests on a row of leverage one gets no HC3 standard error, its row refused and
named; and the survey multinomial and the proportional-odds tables run a separation check, as the
logistic table does, refusing a separated table and naming the separating column (T1V3-5; the
methods lead's ruling 4 of 2026-10-27).

On the base a 0/1 column held by one row (leverage one: the fit passes through that row) had an
HC3 standard error of rounding (6.16, 5.19 and 7.38 on three outcomes; R's ``sandwich::vcovHC``
gives NaN, "HC3 covariances are ... undefined if exactly 1"); and with that row at the lowest
class or level, the survey multinomial served its log-odds at −34.5 and the proportional-odds
table at −23.5 (SE 126,231), where the maximum is at minus infinity.

Truth: the hat matrix of a design with a row of leverage one is that row's alone plus the other
rows' (h = 1 makes it an eigenvector), so a coefficient that does not rest on it has the HC3
covariance of the fit without that row, statsmodels' OLS (HC3) on the other rows; and the
derivation that the likelihood rises without end as the single row's coefficient runs to minus
infinity (it raises that row's probability of the lowest class or level toward one and moves no
other row's), while a middle level's single row, which can be pushed neither way, is no
separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import (
    INDEPENDENT,
    Clusters,
    inference_table,
    separated_multinomial,
    separated_ordinal,
)
from turbotab.core.models.ordinal import ordinal_table, survey_ordinal_table
from turbotab.core.models.survey import SurveyDesign, survey_table

N = 200


def _lone(seed: int = 0):
    rng = np.random.default_rng(5)
    x = rng.normal(size=N)
    one = np.zeros(N)
    one[17] = 1.0
    y = 1 + 0.5 * x + np.random.default_rng(seed).normal(size=N)
    return x, one, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_single_rows_indicator_is_refused_by_row_and_the_rest_are_the_other_rows_hc3(seed):
    x, one, y = _lone(seed)
    table = inference_table("regression", pd.DataFrame({"x": x + 1000, "one": one}), y, None,
                            INDEPENDENT, study=["x"])
    assert table.info["refused"] is None and table.leverage_one == ["one"]
    assert table.concerns[0] == (
        "1 of these 200 rows has leverage one: the fit passes through it exactly, and HC3 divides "
        "its residual by one less its leverage, zero by zero, so no estimate, interval or p-value "
        "is reported for `one`, which rests on it; the other coefficients' HC3 standard errors are "
        "those of the other rows.")
    rows = {r["feature"]: r for r in table.rows}
    assert all(rows["one"][k] is None for k in ("estimate", "se", "ci_low", "ci_high", "p"))
    keep = np.arange(N) != 17
    ref = sm.OLS(y[keep], sm.add_constant(x[keep] + 1000)).fit(cov_type="HC3", use_t=True)
    low, high = ref.conf_int().T
    for name, k in (("(intercept)", 0), ("x", 1)):
        row = rows[name]
        assert row["estimate"] == pytest.approx(ref.params[k], rel=1e-9), name
        assert row["se"] == pytest.approx(ref.bse[k], rel=1e-9), name
        assert row["df"] == ref.df_resid
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((low[k], high[k]), rel=1e-9)
        assert row["p"] == pytest.approx(ref.pvalues[k], rel=1e-7)


def test_a_studied_coefficient_on_a_row_of_leverage_one_refuses_the_table_whole():
    """Studied, `one` refuses the whole table; with nothing named as studied (the effects stage's
    tables) the others are served and `one` is refused by row, since no other coefficient depends
    on it (unlike a dependency's, whose estimates depend on which column a fit leaves out)."""
    x, one, y = _lone()
    matrix = pd.DataFrame({"x": x, "one": one})
    table = inference_table("regression", matrix, y, None, INDEPENDENT, study=["one"])
    assert table.rows == [] and table.info["refused"] == (
        "1 of these 200 rows has leverage one: the fit passes through it exactly, and HC3 "
        "divides its residual by one less its leverage, zero by zero, so `one` has no HC3 "
        "standard error. No estimate, interval or p-value is reported.")
    assert table.info["exits"][0]["label"] == "Leave `one` out, or merge its rare levels"
    unnamed = inference_table("regression", matrix, y, None, INDEPENDENT, study=None)
    assert unnamed.info["refused"] is None and unnamed.leverage_one == ["one"]
    assert {r["feature"]: r for r in unnamed.rows}["x"]["se"] is not None


def test_a_single_row_as_the_reference_refuses_the_intercept_with_its_column():
    """Its complement (`other` is 0 on the one row): the intercept as it stands is that row's
    level, so it rests on the row too, and both are refused; `x` is the other rows'."""
    x, one, y = _lone()
    table = inference_table("regression", pd.DataFrame({"x": x, "other": 1 - one}), y, None,
                            INDEPENDENT, study=["x"])
    assert table.leverage_one == ["(intercept)", "other"]
    assert "reported for the intercept and `other`, which rest on it" in table.concerns[0]
    keep = np.arange(N) != 17
    ref = sm.OLS(y[keep], sm.add_constant(x[keep])).fit(cov_type="HC3")
    row = {r["feature"]: r for r in table.rows}["x"]
    assert row["se"] == pytest.approx(ref.bse[1], rel=1e-9)


def test_an_indicator_held_by_two_rows_is_served_at_hc3():
    """The silent side: two rows hold the indicator, neither has leverage one, and every row is
    statsmodels' HC3 on every row."""
    x, one, y = _lone()
    one[40] = 1.0
    table = inference_table("regression", pd.DataFrame({"x": x, "one": one}), y, None,
                            INDEPENDENT, study=["x"])
    assert table.info["refused"] is None and table.leverage_one == []
    ref = sm.OLS(y, sm.add_constant(np.column_stack([x, one]))).fit(cov_type="HC3")
    for row, se in zip(table.rows, ref.bse):
        assert row["se"] == pytest.approx(se, rel=1e-9), row["feature"]


# ── separation in the multinomial and proportional-odds tables ────────────────


def _levels():
    rng = np.random.default_rng(7)
    n = 360
    x1, x4 = rng.normal(size=(2, n))
    eta = 0.4 * x1 - 0.3 * x4
    yo = np.digitize(eta + rng.logistic(size=n), [-0.7, 0.7])
    design = SurveyDesign(row_ids=np.arange(n), weight=rng.uniform(1, 3, n),
                          stratum=np.repeat(np.arange(12), n // 12).astype(np.int64),
                          psu=np.repeat(np.arange(36), n // 36).astype(np.int64),
                          weight_column="wt", strata_column="st", psu_column="psu")
    codes = np.repeat(np.arange(60), n // 60)
    return x1, x4, yo, design, Clusters(column="pid", codes=codes, n_clusters=60, n_missing=0)


def _single(yo: np.ndarray, level: int) -> np.ndarray:
    lone = np.zeros(len(yo))
    lone[int(np.flatnonzero(yo == level)[0])] = 1.0
    return lone


def _tables(yo, design, cl):
    s = {"study": ["x1"]}
    return {
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT, **s),
        "proportional odds (CR1)": lambda m: ordinal_table(m, yo, [0, 1, 2], cl, **s),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design, **s),
        "survey multinomial": lambda m: survey_table("multiclass", m, yo, [0, 1, 2], design, **s),
        "survey multinomial (ordered levels)": lambda m: survey_table("ordinal", m, yo, [0, 1, 2],
                                                                      design, **s),
    }


@pytest.mark.parametrize("family", list(_tables(*_levels()[2:])))
def test_a_single_rows_indicator_at_the_lowest_level_refuses_the_table_naming_it(family):
    x1, x4, yo, design, cl = _levels()
    build = _tables(yo, design, cl)[family]
    matrix = pd.DataFrame({"x1": x1, "lone": _single(yo, 0), "x4": x4})
    table = build(matrix)
    refused = table.info["refused"]
    assert refused is not None and refused.startswith("`lone` separates the outcome's ")
    assert "(a maximum-likelihood coefficient is infinite)" in refused
    assert all(r["estimate"] is None for r in table.rows)
    assert table.info["exits"][0]["label"] == "Leave `lone` out, or merge its rare levels"
    spread = build(pd.DataFrame({"x1": x1, "lone": (np.arange(len(yo)) % 7 == 0).astype(float),
                                 "x4": x4}))
    assert spread.info["refused"] is None, spread.info["refused"]  # the silent side


def test_the_proportional_odds_check_reads_a_middle_levels_single_row_as_no_separation():
    """At the lowest or the top level a single row's coefficient runs to minus or plus infinity;
    at a middle level it can be pushed neither way (its two margins pull against each other), so
    nothing is separated; a multinomial model separates a single row in any class (its own
    class's coefficient alone runs off)."""
    x1, x4, yo, *_ = _levels()
    for level, want in ((0, [1]), (2, [1]), (1, [])):
        X = np.column_stack([x1, _single(yo, level), x4])
        assert separated_ordinal(X, yo) == want, level
        Xc = np.column_stack([np.ones(len(yo)), X])
        assert separated_multinomial(Xc, yo, 3) == [2], level
    plain = np.column_stack([x1, x4])
    assert separated_ordinal(plain, yo) == []
    assert separated_multinomial(np.column_stack([np.ones(len(yo)), plain]), yo, 3) == []
