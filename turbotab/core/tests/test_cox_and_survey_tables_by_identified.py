"""The Cox and survey coefficient tables on dependent columns, where the caller names what is
studied (`x`, here), serve the coefficients the rows identify and refuse by row each one in a
dependency, in words that name exactly the dependency's columns (``inference.by_identified``), as
the linear family's tables do.

On the base the survey least-squares and logistic tables served min-norm numbers with intervals
for coefficients that are not identified (the intercept 0.2105 and a constant column 0.4210, of
which only their sum is), and the Cox table was refused in the near-singular concern's words,
which named `x`, in no dependency, beside a column of zeros, said "`c` are close to a fixed
combination of each other" of one constant column, and quoted condition numbers that at an exact
dependency are rounding.

Truth from construction: a column of zeros, a constant column (which a Cox model's baseline
hazard and a survey model's intercept absorb) and a column twice another, beside `x` and `w` in no
dependency. A Cox coefficient in no dependency is statsmodels' PHReg (Efron ties) on the design
with the aliased column left out; a survey coefficient is the survey table's on that design (the
identity that it does not depend on which aliased column is left out), its estimate statsmodels'
weighted least squares or weighted logistic regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

from turbotab.core.models.inference import INDEPENDENT
from turbotab.core.models.ordinal import survey_ordinal_table
from turbotab.core.models.survey import SurveyDesign, survey_table
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N = 300
STUDIED = ["x"]  # in no dependency of any construction below


def _data():
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(2, N))
    t = rng.exponential(np.exp(-0.5 * x))
    event = rng.random(N) < 0.7
    idx = pd.RangeIndex(N)
    design = SurveyDesign(row_ids=np.asarray(idx), weight=rng.uniform(1, 3, N),
                          stratum=np.repeat(np.arange(10), 30), psu=np.repeat(np.arange(30), 10),
                          weight_column="wt", strata_column="st", psu_column="psu")
    yr = 1 + 0.5 * x + rng.normal(size=N)
    yb = (rng.random(N) < 1 / (1 + np.exp(-x))).astype(int)
    yo = np.digitize(0.8 * x + rng.logistic(size=N), [-0.7, 0.7])
    return x, w, t, event, idx, design, yr, yb, yo


# By construction: the matrix, the columns in its dependency, whether the constant is in it, and
# the column a fit keeping the earlier columns leaves out.
def _case(kind: str):
    x, w, *_ , idx = _data()[:5]
    if kind == "zeros":
        return pd.DataFrame({"x": x, "w": w, "z": np.zeros(N)}, index=idx), ["z"], False, "z"
    if kind == "a constant":
        return pd.DataFrame({"x": x, "w": w, "c": np.full(N, 2.0)}, index=idx), ["c"], True, "c"
    return pd.DataFrame({"x": x, "w": w, "w2": 2 * w}, index=idx), ["w", "w2"], False, "w2"


KINDS = ("zeros", "a constant", "a copy")


def _words(named: list[str], constant: bool, left: str, beside: str, extra: str | None) -> str:
    one = len(named) == 1
    listed = " and ".join(f"`{c}`" for c in named)
    verb = "is a fixed combination" if one else "are fixed combinations"
    whose = "its coefficient is" if one else "their coefficients are"
    what = f"{listed} {verb} of the other columns and {beside}, so {whose} not identified"
    unreported = listed
    if constant and extra:
        what += f", nor is {extra}"
        unreported = f"{listed} and {extra}"
    return (f"On these {N} rows the model matrix is singular: {what}. No estimate, interval or "
            f"p-value is reported for {unreported}; the other coefficients, which these rows "
            f"identify, are estimated with `{left}` left out. To report every coefficient, leave "
            f"one of the dependent columns out.")


@pytest.mark.parametrize("kind", KINDS)
def test_the_cox_table_refuses_the_dependency_by_row_and_reports_the_rest_as_phreg(kind):
    """Each column in the dependency (and no other: never `x`) is a row with no estimate; `x` and
    `w` where it is in none are PHReg's on the design without the aliased column, estimate and
    standard error; the words name the dependency's columns alone."""
    x, w, t, event, *_ = _data()
    matrix, named, _, left = _case(kind)
    y = survival_outcome(event, t)
    table = cox_table(matrix, y, INDEPENDENT, study=STUDIED)
    assert table.info["refused"] is None
    assert table.unidentified == named
    assert table.concerns[0] == _words(named, False, left, "a constant", None)
    kept = matrix.drop(columns=[left])
    ph = PHReg(t, kept, status=event.astype(int), ties="efron").fit()
    rows = {r["feature"]: r for r in table.rows}
    for name in named:
        assert rows[name]["estimate"] is None and rows[name]["se"] is None
    for name, b, s in zip(kept.columns, ph.params, ph.bse):
        if name in named:
            continue
        assert rows[name]["estimate"] == pytest.approx(float(b), rel=1e-7)
        assert rows[name]["se"] == pytest.approx(float(s), rel=1e-6)


def test_an_independent_cox_design_is_served_as_phreg():
    """The silent side: `x` and `w` are served at PHReg's numbers, nothing refused."""
    x, w, t, event, idx, *_ = _data()
    table = cox_table(pd.DataFrame({"x": x, "w": w}, index=idx), survival_outcome(event, t),
                      INDEPENDENT)
    ph = PHReg(t, np.column_stack([x, w]), status=event.astype(int), ties="efron").fit()
    assert table.info["refused"] is None and table.unidentified == []
    np.testing.assert_allclose([r["estimate"] for r in table.rows], ph.params, rtol=1e-7)
    np.testing.assert_allclose([r["se"] for r in table.rows], ph.bse, rtol=1e-6)


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize("kind", KINDS)
def test_the_survey_tables_refuse_the_dependency_by_row(task, kind):
    """The survey least-squares and logistic tables: the dependency's rows (the intercept's too,
    beside a constant) have no estimate; every other is the survey table's on the design without
    the aliased column, and its estimate statsmodels' weighted fit. On the base both tables served
    a constant column and the intercept with intervals (0.4210 and 0.2105 where only their sum,
    1.0526, is identified), and `w` and `w2 = 2w` likewise."""
    *_, design, yr, yb, _ = _data()
    matrix, named, constant, left = _case(kind)
    y, classes = (yr, None) if task == "regression" else (yb, [0, 1])
    table = survey_table(task, matrix, y, classes, design, study=STUDIED)
    assert table.info["refused"] is None
    assert table.unidentified == (["(intercept)"] if constant else []) + named
    assert table.concerns[0] == _words(named, constant, left, "the intercept", "the intercept")
    kept = matrix.drop(columns=[left])
    reference = {r["feature"]: r for r in survey_table(task, kept, y, classes, design).rows}
    exog = sm.add_constant(kept, has_constant="add")
    weight = np.asarray(design.weight)
    fit = (sm.WLS(y, exog, weights=weight).fit() if task == "regression" else
           sm.GLM(y, exog, family=sm.families.Binomial(), var_weights=weight).fit())
    truth = dict(zip(["(intercept)" if c == "const" else c for c in exog.columns], fit.params))
    for row in table.rows:
        if row["feature"] in table.unidentified:
            assert row["estimate"] is None and row["se"] is None and row["ci_low"] is None
            continue
        for key in ("estimate", "se", "df", "ci_low", "ci_high", "p"):
            assert row[key] == pytest.approx(reference[row["feature"]][key], rel=1e-8, abs=1e-12)
        assert row["estimate"] == pytest.approx(float(truth[row["feature"]]), rel=1e-6)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_an_independent_survey_design_is_served_whole(task):
    """The silent side: nothing refused, every row with its interval, at the weighted fit."""
    x, w, *_, design, yr, yb, _ = _data()
    matrix = pd.DataFrame({"x": x, "w": w}, index=pd.RangeIndex(N))
    y, classes = (yr, None) if task == "regression" else (yb, [0, 1])
    table = survey_table(task, matrix, y, classes, design)
    assert table.info["refused"] is None and table.unidentified == []
    assert all(r["ci_low"] is not None for r in table.rows)


def test_the_survey_ordinal_and_cox_tables_follow_the_same_rule():
    """Beside a column twice `w`, the survey proportional-odds and Cox tables refuse `w` and `w2`
    by row and report `x` as on the design without `w2`."""
    x, w, t, event, idx, design, _, _, yo = _data()
    matrix, named, _, left = _case("a copy")
    y = survival_outcome(event, t)
    for build in (lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design, study=STUDIED),
                  lambda m: survey_cox_table(m, y, design, study=STUDIED)):
        table = build(matrix)
        reference = {r["feature"]: r for r in build(matrix.drop(columns=[left])).rows}
        assert table.info["refused"] is None
        assert table.unidentified == named
        assert table.concerns[0] == _words(named, False, left, "a constant", None)
        rows = {r["feature"]: r for r in table.rows}
        for key in ("estimate", "se", "ci_low", "ci_high"):
            assert rows["x"][key] == pytest.approx(reference["x"][key], rel=1e-8)


@pytest.mark.parametrize("family", ["Cox", "survey least squares", "survey logistic",
                                    "survey proportional odds", "survey Cox"])
def test_with_nothing_named_studied_a_dependency_refuses_the_table_whole(family):
    """The guard's other side: where the caller names nothing studied, the copy `w2` = 2`w`
    refuses the whole table, with no coefficient, in the dependency's words (the survey ones keep
    a row per column, each with no estimate); `x`, named as studied, keeps it served by row."""
    x, w, t, event, idx, design, yr, yb, yo = _data()
    matrix, *_ = _case("a copy")
    y = survival_outcome(event, t)
    build = {"Cox": lambda m, **k: cox_table(m, y, INDEPENDENT, **k),
             "survey least squares": lambda m, **k: survey_table("regression", m, yr, None,
                                                                 design, **k),
             "survey logistic": lambda m, **k: survey_table("binary", m, yb, [0, 1], design, **k),
             "survey proportional odds": lambda m, **k: survey_ordinal_table(m, yo, [0, 1, 2],
                                                                             design, **k),
             "survey Cox": lambda m, **k: survey_cox_table(m, y, design, **k)}[family]
    beside = "the intercept" if family in ("survey least squares", "survey logistic") else \
        "a constant"
    table = build(matrix)
    assert table.info["refused"] == (
        f"On these {N} rows the model matrix is singular: `w` and `w2` are fixed combinations of "
        f"the other columns and {beside}, so their coefficients are not identified. No estimate, "
        f"interval or p-value is reported; leave one of the dependent columns out.")
    assert all(r["estimate"] is None for r in table.rows)
    served = build(matrix, study=STUDIED)
    assert served.info["refused"] is None and served.unidentified == ["w", "w2"]
