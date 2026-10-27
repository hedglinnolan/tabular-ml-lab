"""A least-squares, logistic, multinomial, proportional-odds, mixed-model or GEE table on a model
matrix whose columns are dependent is refused before the fit, with no estimate, in words that name
the dependent columns (``inference.dependent_refusal``).

There the coefficients in a dependency have no unique estimate, and what a fit made of them was
rounding: the least-squares table served pseudo-inverse values with intervals beside a concern that
called them not identified (a 4-level `educ` beside its own 2-level recoding, the Sullivan 2021
walks' pattern); the logistic table on the same rows was refused in words about a missing standard
error, and on ``[1, x, 2x]`` served standard errors of 10⁷; the proportional-odds table served
pseudo-inverse ones; the multinomial table was refused or not by rounding; and a column of zeros
or of ones ended the logistic table with ``LinAlgError: Singular matrix``. Dependence is decided at
working precision on the column-scaled design (``repeated.dependencies``, the mixed model's rule).

Truth from construction: a column twice another, a column of ones or of zeros beside the intercept,
two indicators that sum to one, and an indicator that is the sum of two others; each column named
is a fixed combination of the others and the intercept, and a column outside every dependency is
not named. The silent side: independent columns are fit as before, at statsmodels' estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
from turbotab.core.models.ordinal import ordinal_table

N = 240


def _rows() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(21)
    x = rng.normal(size=N)
    v = rng.normal(size=N)
    latent = x - 0.5 * v + rng.normal(size=N)
    return x, v, {"regression": latent, "binary": (latent > 0).astype(int),
                  "multiclass": np.digitize(latent, [-0.6, 0.6]),
                  "ordinal": np.digitize(latent, [-0.6, 0.6])}


def _matrix(kind: str) -> pd.DataFrame:
    x, v, _ = _rows()
    if kind == "a copy":
        return pd.DataFrame({"x": x, "v": v, "x_twice": 2 * x})
    if kind == "a constant":
        return pd.DataFrame({"x": x, "v": v, "one": np.ones(N)})
    if kind == "zeros":
        return pd.DataFrame({"x": x, "v": v, "zero": np.zeros(N)})
    if kind == "indicators that sum to one":
        return pd.DataFrame({"x": x, "v": v, "low": (x <= 0).astype(float),
                             "high": (x > 0).astype(float)})
    # a recoding: `both` is the indicator of `a` or `b`, as a 2-level recoding of a 3-level code
    code = np.arange(N) % 3
    return pd.DataFrame({"x": x, "v": v, "a": (code == 1).astype(float),
                         "b": (code == 2).astype(float), "both": (code >= 1).astype(float)})


NAMED = {"a copy": "`x` and `x_twice` are fixed combinations of the other columns and {beside}, so "
                   "their coefficients are",
         "a constant": "`one` is a fixed combination of the other columns and {beside}, so its "
                       "coefficient is",
         "zeros": "`zero` is a fixed combination of the other columns and {beside}, so its "
                  "coefficient is",
         "indicators that sum to one": "`low` and `high` are fixed combinations of the other "
                                       "columns and {beside}, so their coefficients are",
         "a recoding": "`a`, `b` and `both` are fixed combinations of the other columns and "
                       "{beside}, so their coefficients are"}
TASKS = ("regression", "binary", "multiclass", "ordinal")
ESTIMATORS = {"regression": "ordinary least squares",
              "binary": "logistic regression (maximum likelihood)",
              "multiclass": "multinomial logistic regression (maximum likelihood)",
              "ordinal": "proportional-odds (cumulative logit) model, maximum likelihood"}


def _table(task: str, matrix: pd.DataFrame, clusters: Clusters = INDEPENDENT):
    y = _rows()[2][task]
    if task == "ordinal":
        return ordinal_table(matrix, y, None, clusters)
    classes = {"regression": None, "binary": [0, 1], "multiclass": [0, 1, 2]}[task]
    return inference_table(task, matrix, y, classes, clusters)


def _clustered() -> Clusters:
    return Clusters(column="pid", codes=np.repeat(np.arange(N // 4), 4), n_clusters=N // 4,
                    n_missing=0)


@pytest.mark.parametrize("clustered", [False, True], ids=["independent rows", "clustered"])
@pytest.mark.parametrize("kind", list(NAMED))
@pytest.mark.parametrize("task", TASKS)
def test_dependent_columns_refuse_the_table_in_words_that_name_them(task, kind, clustered):
    """Refused with no estimate, its words naming the dependent columns and not `v`, outside every
    dependency; on the base these tables were served, refused in other words, or ended in a
    ``LinAlgError``."""
    table = _table(task, _matrix(kind), _clustered() if clustered else INDEPENDENT)
    beside = "a constant" if task == "ordinal" else "the intercept"
    words = (f"On these {N} rows the model matrix is singular: "
             f"{NAMED[kind].format(beside=beside)} not identified. No estimate, interval or "
             f"p-value is reported; leave one of the dependent columns out.")
    assert table.info["refused"] == words
    assert table.rows == []
    assert table.info["estimator"] == ESTIMATORS[task]
    assert table.info["caption"] == f"No coefficients: {words}"
    assert [e["label"] for e in table.info["exits"]] == [
        "Leave one of the dependent columns out of the model"]


@pytest.mark.parametrize("family", ["mixed", "gee"])
@pytest.mark.parametrize("kind", list(NAMED))
def test_the_mixed_and_gee_tables_are_refused_by_the_same_rule(family, kind):
    """The mixed model's and the GEE's tables on units that repeat: refused with no estimate in the
    same words, where on the base they were refused in the near-singular concern's words, which
    quote a condition number that at an exact dependency is rounding (some 10¹⁵, or "too large to
    compute") and name the columns Belsley's proportions put in it (`x` beside a column of zeros,
    in none), the estimates kept."""
    from turbotab.core.models.repeated import gee_table, mixed_table

    y = _rows()[2]["regression"]
    table = (mixed_table(_matrix(kind), y, _clustered()) if family == "mixed" else
             gee_table(_matrix(kind), y, _clustered(), "regression"))
    assert table.info["refused"] == (
        f"On these {N} rows the model matrix is singular: "
        f"{NAMED[kind].format(beside='the intercept')} not identified. No estimate, interval or "
        f"p-value is reported; leave one of the dependent columns out.")
    assert table.rows == []


def test_two_dependencies_are_counted():
    """A copy of `x` and a column of ones: two dependencies, counted, every column in one named."""
    x, v, ys = _rows()
    matrix = pd.DataFrame({"x": x, "v": v, "x_twice": 2 * x, "one": np.ones(N)})
    table = inference_table("regression", matrix, ys["regression"], None)
    assert table.info["refused"] == (
        f"On these {N} rows the model matrix is singular: `x`, `x_twice` and `one` are, in 2 ways, "
        f"fixed combinations of the other columns and the intercept, so their coefficients are not "
        f"identified. No estimate, interval or p-value is reported; leave out one dependent column "
        f"for each of the 2 ways.")


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_independent_columns_are_fit_at_statsmodels_estimates(task):
    """The guard's silent side: `x` and `v` are not refused, and the estimates are statsmodels'
    (OLS, Logit, MNLogit), with every row's interval."""
    x, v, ys = _rows()
    matrix = pd.DataFrame({"x": x, "v": v})
    table = _table(task, matrix)
    assert table.info["refused"] is None
    exog = sm.add_constant(matrix)
    model = {"regression": sm.OLS, "binary": sm.Logit, "multiclass": sm.MNLogit}[task]
    params = np.asarray(model(ys[task], exog).fit(disp=0).params if task != "regression" else
                        model(ys[task], exog).fit().params, dtype=float).ravel(order="F")
    np.testing.assert_allclose([r["estimate"] for r in table.rows], params, rtol=1e-6)
    assert all(r["ci_low"] is not None for r in table.rows)


def test_independent_columns_in_the_proportional_odds_table_are_not_refused():
    """The silent side for the proportional-odds table: every row has its interval."""
    x, v, _ = _rows()
    table = _table("ordinal", pd.DataFrame({"x": x, "v": v}))
    assert table.info["refused"] is None
    assert len(table.rows) == 4 and all(r["ci_low"] is not None for r in table.rows)


def test_a_nearly_dependent_column_is_not_refused():
    """The rule is working precision, not a near-dependency: `x` beside `x` plus noise of SD 10⁻⁴
    (a scaled condition number near 10⁴) is fit by least squares, at statsmodels' estimates."""
    x, v, ys = _rows()
    near = x + np.random.default_rng(3).normal(scale=1e-4, size=N)
    matrix = pd.DataFrame({"x": x, "v": v, "near": near})
    table = inference_table("regression", matrix, ys["regression"], None)
    assert table.info["refused"] is None
    ols = sm.OLS(ys["regression"], sm.add_constant(matrix)).fit()
    np.testing.assert_allclose([r["estimate"] for r in table.rows], ols.params, rtol=1e-6)


def test_through_the_fit_stage_the_linear_tables_are_refused(tmp_path):
    """The real split, shelf, design and fit stages under inference, one row per participant: a
    4-level `educ` beside its own 2-level recoding `educ2` (its level 2 is `educ`'s 3 and 4). The
    least-squares and logistic tables are refused in words naming `educ_3`, `educ_4` and
    `educ2_2`, not `educ_2` or `age`, and no concern beside them calls the matrix nearly singular;
    on the base the least-squares table was served with intervals for all three."""
    from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _stages

    rng = np.random.default_rng(12)
    n = 400
    educ = rng.integers(1, 5, n)
    age = rng.integers(30, 70, n).astype(float)
    sbp = 110 + 0.3 * age + rng.normal(0, 5, n)
    frame = pd.DataFrame({"participant_id": np.arange(n) + 1001, "educ": educ,
                          "educ2": np.where(educ <= 2, 1, 2), "age": age, "sbp": sbp,
                          "high": (sbp > np.median(sbp)).astype(int)})
    words = (f"On these {n} rows the model matrix is singular: `educ_3`, `educ_4` and `educ2_2` "
             f"are fixed combinations of the other columns and the intercept, so their "
             f"coefficients are not identified. No estimate, interval or p-value is reported; "
             f"leave one of the dependent columns out.")
    for target, task in (("sbp", "regression"), ("high", "binary")):
        st = ProjectState(lens=["clinical"], target=target, task=task, purpose="inference",
                          roles={"participant_id": "identifier", "educ": "covariate",
                                 "educ2": "covariate", "age": "exposure"},
                          missing="complete_case", split=SplitSpec(holdout=0.0, seed=0, folds=5),
                          models=["linear"],
                          grain=GrainSpec(grain="one_row_per_unit", id_column="participant_id"),
                          shape_confirmations={"code_or_count:age": "amount",
                                               "code_or_count:educ": "code",
                                               "code_or_count:educ2": "code"})
        other = "high" if target == "sbp" else "sbp"
        _, _, fit = _stages(frame.drop(columns=[other]), st, ["educ", "educ2", "age"], task,
                            tmp_path / task)
        model = fit.data["models"][0]
        assert model["inference"]["refused"] == words
        assert not model["coefficients"]
        assert words in model["concerns"]
        assert not any("nearly singular" in c for c in model["concerns"])
