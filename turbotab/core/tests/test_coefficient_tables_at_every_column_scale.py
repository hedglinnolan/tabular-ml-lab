"""A column's unit changes only its own coefficient: rescaling a column by c divides its estimate
and standard error by c and leaves every other row of the table as it is, in every table of a
family whose linear predictor is Xβ (``inference.by_identified`` fits each on its columns scaled to
a root mean square of one and scales the rows back).

On the base, statsmodels' least-squares, logistic and multinomial fits, the CR2 bread
``pinv(XᵀX)``, the proportional-odds and survey covariances (a pseudo-inverse of the unscaled
information) and the GEE's served wrong numbers once a column's unit was some 10⁴ or more times
another's: a standard-normal `x` beside a column of mean 3·10⁷ had its clustered least-squares SE
cut to rounding, a date in epoch seconds gave `x` an SE of 8.4·10⁻²² against 0.0648 with the date
in years, the logistic and multinomial SEs were off by factors of 398 and 221 at units 10⁻⁵ and
10⁵, and at 10⁻⁹ and 10⁹ the least-squares estimate of `x1` was 0.

Truth from construction: the exact identity above, at factors from 10⁻⁹ to 10⁹ (and an offset
date, whose intercept the identity moves by a known amount, so only the other columns' rows and
the date's own SE are compared); at c = 1 the tables are statsmodels' at ordinary scales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
from turbotab.core.models.ordinal import ordinal_table, survey_ordinal_table
from turbotab.core.models.repeated import gee_table, mixed_table
from turbotab.core.models.survey import SurveyDesign, survey_table
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N, G = 400, 80
FACTORS = (1e-9, 1e-5, 1e5, 1e9)


def _data():
    rng = np.random.default_rng(42)
    codes = np.repeat(np.arange(G), N // G)
    clusters = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)
    x1, x2, x4 = rng.normal(size=(3, N))
    eta = 0.4 * x1 - 0.3 * x4 + 0.2 * x2
    yr = 1 + eta + rng.normal(size=N) + rng.normal(size=G)[codes]
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    yo = np.digitize(eta + rng.logistic(size=N), [-0.7, 0.7])
    t = rng.exponential(np.exp(-eta))
    ys = survival_outcome(rng.random(N) < 0.8, t)
    design = SurveyDesign(row_ids=np.arange(N), weight=rng.uniform(1, 3, N),
                          stratum=np.repeat(np.arange(10), N // 10),
                          psu=np.repeat(np.arange(40), N // 40), weight_column="wt",
                          strata_column="st", psu_column="psu")
    return (x1, x2, x4), clusters, yr, yb, yo, ys, design


def _tables():
    _, cl, yr, yb, yo, ys, design = _data()
    return {
        "least squares (HC3)": lambda m: inference_table("regression", m, yr, None),
        "least squares (CR2)": lambda m: inference_table("regression", m, yr, None, cl),
        "logistic": lambda m: inference_table("binary", m, yb, [0, 1]),
        "logistic (CR2)": lambda m: inference_table("binary", m, yb, [0, 1], cl),
        "multinomial": lambda m: inference_table("multiclass", m, yo, [0, 1, 2]),
        "multinomial (CR2)": lambda m: inference_table("multiclass", m, yo, [0, 1, 2], cl),
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT),
        "proportional odds (CR1)": lambda m: ordinal_table(m, yo, [0, 1, 2], cl),
        "mixed model": lambda m: mixed_table(m, yr, cl),
        "GEE": lambda m: gee_table(m, yr, cl, "regression"),
        "GEE (logit)": lambda m: gee_table(m, yb, cl, "binary", [0, 1]),
        "Cox": lambda m: cox_table(m, ys, INDEPENDENT),
        "Cox (clustered)": lambda m: cox_table(m, ys, cl),
        "survey least squares": lambda m: survey_table("regression", m, yr, None, design),
        "survey logistic": lambda m: survey_table("binary", m, yb, [0, 1], design),
        "survey multinomial": lambda m: survey_table("multiclass", m, yo, [0, 1, 2], design),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design),
        "survey Cox": lambda m: survey_cox_table(m, ys, design),
    }


def _matrix(a: float = 1.0) -> pd.DataFrame:
    (x1, x2, x4), *_ = _data()
    return pd.DataFrame({"x1": x1, "x2": a * x2, "x4": x4}, index=pd.RangeIndex(N))


def _base(feature: str) -> str:
    return feature.split(" [")[0]


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("family", list(_tables()))
def test_rescaling_a_column_moves_only_its_own_row(family, factor):
    """`x2` in units ``factor`` times its own: its estimate, standard error and interval divided by
    ``factor``, every other row's unchanged, its df and p-value unchanged (relative 10⁻⁸). The
    mixed model's REML fit factors the design by QR and is not scaled (``repeated.mixed_table``):
    its γ̂ is found by a bounded search, and its standard errors and Satterthwaite df, which follow
    γ̂, agree to some 10⁻⁸, so it is held to 10⁻⁶."""
    build = _tables()[family]
    rel = 1e-6 if family == "mixed model" else 1e-8
    reference = build(_matrix())
    table = build(_matrix(factor))
    assert reference.info["refused"] is None
    assert table.info["refused"] is None, table.info["refused"]
    assert [r["feature"] for r in table.rows] == [r["feature"] for r in reference.rows]
    for row, ref in zip(table.rows, reference.rows):
        f = factor if _base(row["feature"]) == "x2" else 1.0
        for key in ("estimate", "se", "ci_low", "ci_high"):
            assert row[key] * f == pytest.approx(ref[key], rel=rel, abs=1e-14), (row["feature"], key)
        for key in ("df", "p"):
            if ref[key] is not None:
                assert row[key] == pytest.approx(ref[key], rel=rel, abs=1e-14)


@pytest.mark.parametrize("factor", FACTORS)
def test_the_firth_table_moves_only_the_rescaled_row(factor):
    """A group with no events separates the outcome, so the table is Firth's: `x2` in units
    ``factor`` times its own divides its estimate by ``factor`` and leaves the separated group's
    row, and every other, as it is. The data separate, so each table serves its estimates alone,
    refused for its intervals in the same words (T5d, V5c2-1; before, its profile intervals moved
    with the estimates)."""
    (x1, x2, _), *_ = _data()
    rng = np.random.default_rng(11)
    group = (np.arange(N) % 10 == 0).astype(float)
    y = np.where(group == 1, 0, (rng.random(N) < 1 / (1 + np.exp(-0.5 * x1))).astype(int))

    def build(a: float):
        return inference_table("binary", pd.DataFrame({"x1": x1, "x2": a * x2, "group": group}),
                               y, [0, 1])

    reference, table = build(1.0), build(factor)
    assert reference.info["separated"] == ["group"] and table.info["separated"] == ["group"]
    assert table.info["refused"] == reference.info["refused"]
    assert reference.info["refused"].startswith("`group` separates the outcome")
    for row, ref in zip(table.rows, reference.rows):
        f = factor if row["feature"] == "x2" else 1.0
        assert row["estimate"] * f == pytest.approx(ref["estimate"], rel=1e-7), row["feature"]
        assert all(row[k] is None for k in ("se", "ci_low", "ci_high", "p")), row["feature"]


def test_at_ordinary_scales_the_tables_are_statsmodels():
    """The identity's anchor: at c = 1 the least-squares HC3, logistic and multinomial rows are
    statsmodels' estimates and standard errors."""
    _, _, yr, yb, yo, _, _ = _data()
    matrix = _matrix()
    exog = sm.add_constant(matrix)
    for build, fit in ((lambda: inference_table("regression", matrix, yr, None),
                        sm.OLS(yr, exog).fit(cov_type="HC3")),
                       (lambda: inference_table("binary", matrix, yb, [0, 1]),
                        sm.Logit(yb, exog).fit(disp=0)),
                       (lambda: inference_table("multiclass", matrix, yo, [0, 1, 2]),
                        sm.MNLogit(yo, exog).fit(disp=0))):
        table = build()
        np.testing.assert_allclose([r["estimate"] for r in table.rows],
                                   np.asarray(fit.params).ravel(order="F"), rtol=1e-7)
        np.testing.assert_allclose([r["se"] for r in table.rows],
                                   np.asarray(fit.bse).ravel(order="F"), rtol=1e-6)


@pytest.mark.parametrize("family", ["least squares (CR2)", "logistic (CR2)", "proportional odds",
                                    "mixed model", "GEE", "Cox"])
def test_a_date_in_epoch_seconds_beside_a_standard_normal_column(family):
    """A date in seconds since 1970 (2015 to 2020) against the same date in years since 2015: an
    affine map, so `x1`'s and `x4`'s rows are unchanged and the date's standard error is its SE in
    years divided by the seconds in a year. On the base the clustered least-squares table gave
    `x1` an SE of 8.4·10⁻²² and the proportional-odds table 5.6·10⁻²²."""
    rng = np.random.default_rng(7)
    years = rng.uniform(0, 5, size=N)
    year = 365.25 * 86400
    build = _tables()[family]
    (x1, _, x4), *_ = _data()
    in_years = pd.DataFrame({"x1": x1, "date": years, "x4": x4}, index=pd.RangeIndex(N))
    in_seconds = in_years.assign(date=1_420_070_400 + years * year)
    a, b = build(in_years), build(in_seconds)
    assert b.info["refused"] is None, b.info["refused"]
    rows_a = {r["feature"]: r for r in a.rows}
    for row in b.rows:
        if row["feature"] in ("x1", "x4"):
            for key in ("estimate", "se"):
                assert row[key] == pytest.approx(rows_a[row["feature"]][key], rel=1e-8)
        elif row["feature"] == "date":
            assert row["se"] * year == pytest.approx(rows_a["date"]["se"], rel=1e-8)
            assert row["estimate"] * year == pytest.approx(rows_a["date"]["estimate"], rel=1e-8)
