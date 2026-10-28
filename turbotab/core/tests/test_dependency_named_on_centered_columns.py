"""Which columns a dependency holds is read on the centered columns each scaled to unit length, the
design the identification uses, and the intercept (the cut-points, the baseline) is in it only where
its own combination of the centered coefficients has weight in it (T1V3-1, the methods lead's
ruling 1 of 2026-10-27); each column's length is taken over its largest absolute value, so none
overflows or underflows (T1V3-8, ruling 7).

On the base the dependency's columns were weighted by their norms as they stood: a year (a mean
1,400 SDs from zero) or a date in epoch seconds beside a near copy ``x + ε·d`` (ε below R's
tolerance) was named in the copy's dependency, with the intercept, and the table refused whole
where it was studied, though R's ``lm`` leaves out the copy and serves both (year 0.15174025 with
HC3 SE 0.035457856 on the verifier's rows); and beside an exact shifted copy ``x + M`` with
M ≥ 10⁶, `x` was left out of the dependency it is in (only ``β_x + β_a`` is identified), so `x`
was served at 0.50124. A column of 10²⁰⁰'s or 10⁻²⁰⁰'s squared length was infinite or zero, and
every family refused it by row in exact-dependency words, though it is independent noise.

Truth: statsmodels' OLS with HC3 on the design with the near copy left out (R's ``lm``) or on the
identified columns; the derivation that ``(M, 1, −1)`` is the shifted copy's null vector; the
rescaling identity (a column's unit moves its own row by the unit's factor and no other row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
from turbotab.core.models.ordinal import ordinal_table, survey_ordinal_table
from turbotab.core.models.repeated import aliased, gee_table
from turbotab.core.models.survey import SurveyDesign, survey_table
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N = 400


def _rows():
    rng = np.random.default_rng(11)
    x, d, v = rng.normal(size=(3, N))
    year = 2000 + rng.integers(0, 5, N).astype(float)
    t = 1.7e9 + rng.uniform(0, 86400, N)  # epoch seconds over a day
    eta = 0.5 * x - 0.4 * v + 0.1 * (year - 2002)
    y = eta + rng.normal(size=N)
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    yo = np.digitize(eta + rng.logistic(size=N), [-0.7, 0.7])
    ys = survival_outcome(rng.random(N) < 0.7, rng.exponential(np.exp(-eta)))
    return x, d, v, year, t, y, yb, yo, ys


def _hc3(design: np.ndarray, y: np.ndarray):
    return sm.OLS(y, design).fit(cov_type="HC3")


@pytest.mark.parametrize("eps", [5e-8, 2e-8])
def test_a_year_beside_a_near_copy_is_served_at_lm_with_the_copy_left_out(eps):
    """Below R's tolerance `x2` is aliased; `year` and the intercept are in no dependency, so they
    are served as statsmodels' OLS (HC3) on ``(1, x, v, year)`` gives them, and the words name only
    `x` and `x2`. With `year` studied the table is served, not refused whole."""
    x, d, v, year, _, y, *_ = _rows()
    matrix = pd.DataFrame({"x": x, "x2": x + eps * d, "v": v, "year": year})
    found = aliased(matrix.to_numpy())
    assert found.count == 1 and found.columns.tolist() == [1, 2]
    table = inference_table("regression", matrix, y, None, INDEPENDENT, study=["v"])
    assert table.info["refused"] is None and table.unidentified == ["x", "x2"]
    words = table.concerns[0]
    assert "`x` and `x2` are fixed combinations" in words and "nor is the intercept" not in words
    assert "`year`" not in words
    ref = _hc3(np.column_stack([np.ones(N), x, v, year]), y)
    rows = {r["feature"]: r for r in table.rows}
    for name, k in (("(intercept)", 0), ("v", 2), ("year", 3)):
        assert rows[name]["estimate"] == pytest.approx(ref.params[k], rel=1e-9), name
        assert rows[name]["se"] == pytest.approx(ref.bse[k], rel=1e-7), name
    studied = inference_table("regression", matrix, y, None, INDEPENDENT, study=["year"])
    assert studied.info["refused"] is None and studied.unidentified == ["x", "x2"]
    assert {r["feature"]: r for r in studied.rows}["year"]["estimate"] == pytest.approx(
        ref.params[3], rel=1e-9)


def _families(yb, yo, ys):
    return {
        "logistic": lambda m, s: inference_table("binary", m, yb, [0, 1], INDEPENDENT, study=s),
        "proportional odds": lambda m, s: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT, study=s),
        "Cox": lambda m, s: cox_table(m, ys, INDEPENDENT, study=s),
    }


@pytest.mark.parametrize("family", ["logistic", "proportional odds", "Cox"])
def test_an_epoch_date_beside_a_near_copy_is_in_no_dependency_in_every_family(family):
    """A date in epoch seconds spanning a day beside ``x + 10⁻⁸·d``: each family refuses `x` and
    `x2` by row and serves `t`, studied or not, as its own table without `x2` does (the identity
    that a coefficient in no dependency does not depend on which aliased column is left out)."""
    x, d, v, _, t, _, yb, yo, ys = _rows()
    build = _families(yb, yo, ys)[family]
    matrix = pd.DataFrame({"x": x, "x2": x + 1e-8 * d, "v": v, "t": t})
    table = build(matrix, ["v"])
    assert table.info["refused"] is None
    assert sorted(table.unidentified) == ["x", "x2"]
    studied = build(matrix, ["t"])
    assert studied.info["refused"] is None
    reference = {r["feature"]: r for r in build(matrix.drop(columns="x2"), ["t"]).rows}
    for row in studied.rows:
        if row["feature"] in ("t", "v"):
            assert row["estimate"] == pytest.approx(reference[row["feature"]]["estimate"],
                                                    rel=1e-7)
            assert row["se"] == pytest.approx(reference[row["feature"]]["se"], rel=1e-6)


@pytest.mark.parametrize("M", [1e3, 1e5, 1e6, 1e7])
def test_a_shifted_copy_holds_its_column_and_the_intercept_at_every_shift(M):
    """``a = x + M`` is ``(M, 1, −1)``'s dependency (``M + x − a = 0``): `x`, `a` and the intercept
    are each in it at every M, so each is refused by row and `b` served at statsmodels' OLS (HC3) on
    ``(1, x, b)``; with `x` studied the table is refused whole. On the base `x` was left out of it
    from M = 10⁶ and served at 0.50124."""
    rng = np.random.default_rng(3)
    x, b = rng.normal(size=(2, 300))
    y = 1 + 0.5 * x + 0.3 * b + rng.normal(size=300)
    matrix = pd.DataFrame({"x": x, "a": x + M, "b": b})
    assert aliased(matrix.to_numpy()).columns.tolist() == [0, 1, 2]
    table = inference_table("regression", matrix, y, None, INDEPENDENT, study=["b"])
    assert table.info["refused"] is None
    assert table.unidentified == ["(intercept)", "x", "a"]
    assert "`x` and `a` are fixed combinations" in table.concerns[0]
    assert "nor is the intercept" in table.concerns[0]
    row = {r["feature"]: r for r in table.rows}["b"]
    ref = _hc3(np.column_stack([np.ones(300), x, b]), y)
    assert row["estimate"] == pytest.approx(ref.params[2], rel=1e-9)
    assert row["se"] == pytest.approx(ref.bse[2], rel=1e-7)
    whole = inference_table("regression", matrix, y, None, INDEPENDENT, study=["x"])
    assert whole.info["refused"] is not None and whole.rows == []


@pytest.mark.parametrize("case, columns", [
    ("copy", [1, 2]),           # x2 = x: the intercept is in no dependency
    ("constant", [0, 2]),       # x2 = 3: a constant column's dependency is with the intercept
    ("zero", [2]),              # x2 = 0: a column of zeros is a dependency of its own
    ("sum", [1, 2, 3]),         # x2 = 2x + v
])
def test_the_exact_dependencies_keep_their_names(case, columns):
    """The silent side of the intercept's reading and the exact cases: a copy about a mean near
    zero, a constant, a column of zeros and a sum keep the columns their null vectors hold."""
    rng = np.random.default_rng(4)
    x, v = rng.normal(size=(2, 200))
    x2 = {"copy": x.copy(), "constant": np.full(200, 3.0), "zero": np.zeros(200),
          "sum": 2 * x + v}[case]
    found = aliased(np.column_stack([x, x2, v]))
    assert found.count == 1 and found.exact
    assert found.columns.tolist() == columns


def _scale_families():
    rng = np.random.default_rng(7)
    G = 60
    codes = np.repeat(np.arange(G), 360 // G)
    cl = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)
    x1, x2, x4 = rng.normal(size=(3, 360))
    eta = 0.4 * x1 - 0.3 * x4 + 0.25 * x2
    yr = 1 + eta + rng.normal(size=360) * (1 + 0.3 * np.abs(x4)) + rng.normal(size=G)[codes]
    yb = (rng.random(360) < 1 / (1 + np.exp(-eta))).astype(int)
    yo = np.digitize(eta + rng.logistic(size=360), [-0.7, 0.7])
    ys = survival_outcome(rng.random(360) < 0.7, rng.exponential(np.exp(-eta)))
    design = SurveyDesign(row_ids=np.arange(360), weight=rng.uniform(1, 3, 360),
                          stratum=np.repeat(np.arange(12), 30).astype(np.int64),
                          psu=np.repeat(np.arange(36), 10).astype(np.int64),
                          weight_column="wt", strata_column="st", psu_column="psu")
    S = {"study": ["x1"]}
    families = {
        "least squares (HC3)": lambda m: inference_table("regression", m, yr, None, INDEPENDENT, **S),
        "least squares (CR2)": lambda m: inference_table("regression", m, yr, None, cl, **S),
        "logistic": lambda m: inference_table("binary", m, yb, [0, 1], INDEPENDENT, **S),
        "multinomial": lambda m: inference_table("multiclass", m, yo, [0, 1, 2], INDEPENDENT, **S),
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT, **S),
        "Cox": lambda m: cox_table(m, ys, INDEPENDENT, **S),
        "GEE": lambda m: gee_table(m, yr, cl, "regression", **S),
        "survey least squares": lambda m: survey_table("regression", m, yr, None, design, **S),
        "survey logistic": lambda m: survey_table("binary", m, yb, [0, 1], design, **S),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design, **S),
        "survey Cox": lambda m: survey_cox_table(m, ys, design, **S),
    }
    return (x1, x2, x4), families


@pytest.mark.parametrize("c", [1e200, 1e-200])
def test_a_column_far_past_the_squares_range_is_in_no_dependency(c):
    """`x2` times 10²⁰⁰ or 10⁻²⁰⁰ is independent noise: no dependency, and in every family each row
    is the unscaled table's (`x2`'s estimate and SE times c), to 10⁻⁹. On the base its length
    overflowed or underflowed and every family refused it by row as a fixed combination of the
    other columns."""
    (x1, x2, x4), families = _scale_families()
    assert aliased(np.column_stack([x1, c * x2, x4])).count == 0
    index = pd.RangeIndex(360)
    for name, build in families.items():
        plain = build(pd.DataFrame({"x1": x1, "x2": x2, "x4": x4}, index=index))
        scaled = build(pd.DataFrame({"x1": x1, "x2": c * x2, "x4": x4}, index=index))
        assert scaled.info["refused"] is None and scaled.unidentified == [], name
        for a, b in zip(scaled.rows, plain.rows):
            if a["feature"].startswith(("(intercept)", "(cut-point")):
                continue
            factor = c if a["feature"].startswith("x2") else 1.0
            for key in ("estimate", "se"):
                assert a[key] * factor == pytest.approx(b[key], rel=1e-9), (name, a["feature"], key)
            assert a["p"] == pytest.approx(b["p"], rel=1e-7, abs=1e-15), (name, a["feature"])
