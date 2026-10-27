"""A least-squares, logistic, multinomial, proportional-odds, mixed-model or GEE table on a model
matrix whose columns are dependent serves the coefficients its rows identify and refuses by row
each one in a dependency, in words that name the dependency's columns and count them
(``inference.by_identified``); the whole table is refused, with no estimate, only where a column
its caller names as studied is in a dependency, or no coefficient is identified.

There the coefficients in a dependency have no unique estimate, and what a fit made of them was
rounding: the least-squares table served pseudo-inverse values with intervals beside a concern that
called them not identified (a 4-level `educ` beside its own 2-level recoding, the Sullivan 2021
walks' pattern), the logistic table on ``[1, x, 2x]`` standard errors of 10⁷, and a column of zeros
or of ones ended the logistic table with ``LinAlgError: Singular matrix``. Package A refused every
such table whole, with no estimate; the tail ruling of 2026-10-27 serves a coefficient in no
dependency at the fit with the aliased columns left out, as R's ``lm`` and ``glm`` report it, so a
specification curve's departure that leaves a covariate constant keeps what is studied.
Dependence is decided at working precision on the column-scaled design (``repeated.basis``, the
mixed model's rule).

Truth from construction: a column twice another, a column of ones or of zeros beside the intercept,
two indicators that sum to one, and an indicator that is the sum of two others. A coefficient in no
dependency is statsmodels' (OLS with HC3, Logit, MNLogit, OrderedModel) on the design with the
aliased column left out, its estimate and standard error; with rows that repeat, it is the same
table's on that design (the identity that a coefficient in no dependency does not depend on which
aliased column is left out). The silent side: independent columns are fit as before, at
statsmodels' estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

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


# By construction: the columns in the dependency, whether the constant is in it, and the column a
# fit keeping the earlier columns leaves out.
DEPENDENCY = {"a copy": (["x", "x_twice"], False, "x_twice"),
              "a constant": (["one"], True, "one"),
              "zeros": (["zero"], False, "zero"),
              "indicators that sum to one": (["low", "high"], True, "high"),
              "a recoding": (["a", "b", "both"], False, "both")}
TASKS = ("regression", "binary", "multiclass", "ordinal")
ESTIMATORS = {"regression": "ordinary least squares",
              "binary": "logistic regression (maximum likelihood)",
              "multiclass": "multinomial logistic regression (maximum likelihood)",
              "ordinal": "proportional-odds (cumulative logit) model, maximum likelihood"}
CUTS = ["(cut-point 0 | 1)", "(cut-point 1 | 2)"]


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _words(kind: str, task: str) -> str:
    """The by-row refusal's words, written from the construction."""
    named, constant, left = DEPENDENCY[kind]
    beside = "a constant" if task == "ordinal" else "the intercept"
    one = len(named) == 1
    verb = "is a fixed combination" if one else "are fixed combinations"
    whose = "its coefficient is" if one else "their coefficients are"
    what = (f"{_listed([f'`{c}`' for c in named])} {verb} of the other columns and {beside}, so "
            f"{whose} not identified")
    unreported = [f"`{c}`" for c in named]
    if constant:
        what += ", nor are the cut-points" if task == "ordinal" else ", nor is the intercept"
        unreported.append("the cut-points" if task == "ordinal" else "the intercept")
    return (f"On these {N} rows the model matrix is singular: {what}. No estimate, interval or "
            f"p-value is reported for {_listed(unreported)}; the other coefficients, which these "
            f"rows identify, are estimated with `{left}` left out. To report every coefficient, "
            f"leave one of the dependent columns out.")


def _table(task: str, matrix: pd.DataFrame, clusters: Clusters = INDEPENDENT, **kw):
    y = _rows()[2][task]
    if task == "ordinal":
        return ordinal_table(matrix, y, None, clusters, **kw)
    classes = {"regression": None, "binary": [0, 1], "multiclass": [0, 1, 2]}[task]
    return inference_table(task, matrix, y, classes, clusters, **kw)


def _clustered() -> Clusters:
    return Clusters(column="pid", codes=np.repeat(np.arange(N // 4), 4), n_clusters=N // 4,
                    n_missing=0)


def _refused_rows(task: str, kind: str) -> list[str]:
    named, constant, _ = DEPENDENCY[kind]
    if task == "multiclass":
        return [f"{c} [{k}]" for k in (1, 2) for c in (["(intercept)"] if constant else []) + named]
    if task == "ordinal":
        return named + (CUTS if constant else [])
    return (["(intercept)"] if constant else []) + named


def _statsmodels(task: str, design: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Each coefficient's (estimate, standard error) from statsmodels on ``design`` (no aliased
    column): OLS with HC3, Logit, MNLogit, and OrderedModel (its cut-points' standard errors are
    on its own parameterization, so only the columns' are kept, and the first cut-point)."""
    y = _rows()[2][task]
    if task == "ordinal":
        fit = OrderedModel(y, design, distr="logit").fit(method="newton", disp=0, maxiter=200)
        out = {c: (float(fit.params[c]), float(fit.bse[c])) for c in design.columns}
        out[CUTS[0]] = (float(fit.params.iloc[len(design.columns)]), float("nan"))
        return out
    exog = sm.add_constant(design)
    names = ["(intercept)", *design.columns]
    if task == "regression":
        fit = sm.OLS(y, exog).fit(cov_type="HC3")
        return {n: (float(b), float(s)) for n, b, s in zip(names, fit.params, fit.bse)}
    if task == "binary":
        fit = sm.Logit(y, exog).fit(disp=0)
        return {n: (float(b), float(s)) for n, b, s in zip(names, fit.params, fit.bse)}
    fit = sm.MNLogit(y, exog).fit(disp=0)
    B, S = np.asarray(fit.params), np.asarray(fit.bse)
    return {f"{n} [{k + 1}]": (float(B[i, k]), float(S[i, k])) for k in range(2)
            for i, n in enumerate(names)}


@pytest.mark.parametrize("kind", list(DEPENDENCY))
@pytest.mark.parametrize("task", TASKS)
def test_dependent_columns_are_refused_by_row_and_the_others_are_statsmodels(task, kind):
    """Independent rows: each coefficient in the dependency (and the intercept or the cut-points
    where the constant is in it) is a row with no estimate, interval or p-value; every other is
    statsmodels' on the design with the aliased column left out, its estimate and standard error;
    the words, the table's first concern, name the dependency's columns and not `v`, in no
    dependency. On the base each table was refused whole, with no rows."""
    matrix = _matrix(kind)
    table = _table(task, matrix)
    _, _, left = DEPENDENCY[kind]
    assert table.info["refused"] is None
    assert table.info["estimator"] == ESTIMATORS[task]
    refused = _refused_rows(task, kind)
    assert table.unidentified == refused
    assert table.concerns[0] == _words(kind, task)
    truth = _statsmodels(task, matrix.drop(columns=[left]))
    for row in table.rows:
        if row["feature"] in refused:
            assert all(row[k] is None for k in ("estimate", "se", "ci_low", "ci_high", "p", "df"))
            continue
        assert row["ci_low"] is not None and row["p"] is not None
        estimate, se = truth[row["feature"]] if row["feature"] in truth else (None, None)
        if estimate is None:  # the proportional-odds model's later cut-point: its own parameter
            continue
        assert row["estimate"] == pytest.approx(estimate, rel=1e-5, abs=1e-7)
        if np.isfinite(se):
            assert row["se"] == pytest.approx(se, rel=1e-4)


@pytest.mark.parametrize("kind", list(DEPENDENCY))
@pytest.mark.parametrize("task", TASKS)
def test_with_rows_that_repeat_the_others_are_the_table_without_the_aliased_column(task, kind):
    """Rows that repeat (CR2, or the proportional-odds CR1): each coefficient in no dependency is
    the row the same table gives on the design with the aliased column left out, its estimate,
    standard error, df and interval; on the base the table was refused whole."""
    matrix = _matrix(kind)
    _, _, left = DEPENDENCY[kind]
    table = _table(task, matrix, _clustered())
    reference = {r["feature"]: r for r in _table(task, matrix.drop(columns=[left]),
                                                 _clustered()).rows}
    assert table.info["refused"] is None
    assert table.unidentified == _refused_rows(task, kind)
    assert table.concerns[0] == _words(kind, task)
    for row in table.rows:
        if row["feature"] in table.unidentified:
            assert row["estimate"] is None and row["se"] is None and row["ci_low"] is None
            continue
        for key in ("estimate", "se", "df", "ci_low", "ci_high", "p"):
            assert row[key] == pytest.approx(reference[row["feature"]][key], rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("family", ["mixed", "gee"])
@pytest.mark.parametrize("kind", list(DEPENDENCY))
def test_the_mixed_and_gee_tables_follow_the_same_rule(family, kind):
    """The mixed model's and the GEE's tables on units that repeat: the dependency's rows refused
    by row in the same words, every other the same table's on the design with the aliased column
    left out. On the base they were refused whole (and before package A, in the near-singular
    concern's words, which quote a condition number that at an exact dependency is rounding)."""
    from turbotab.core.models.repeated import gee_table, mixed_table

    y = _rows()[2]["regression"]
    _, _, left = DEPENDENCY[kind]
    build = ((lambda m: mixed_table(m, y, _clustered())) if family == "mixed" else
             (lambda m: gee_table(m, y, _clustered(), "regression")))
    table = build(_matrix(kind))
    reference = {r["feature"]: r for r in build(_matrix(kind).drop(columns=[left])).rows}
    assert table.info["refused"] is None
    assert table.unidentified == _refused_rows("regression", kind)
    assert table.concerns[0] == _words(kind, "regression")
    for row in table.rows:
        if row["feature"] in table.unidentified:
            assert row["estimate"] is None and row["se"] is None
            continue
        for key in ("estimate", "se", "df", "ci_low", "ci_high"):
            assert row[key] == pytest.approx(reference[row["feature"]][key], rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("family", ["mixed", "gee"])
def test_with_no_unit_that_repeats_the_dependency_is_decided_first(family):
    """No identifier repeats: the mixed and GEE tables are refused for that, keeping their
    estimates, but `w` and `w2 = 2w` get none, and `x` is least squares on the design without
    `w2` (statsmodels). On the base the no-units refusal came first and kept `w` = 0.0563 and
    `w2` = 0.1126, the pseudo-inverse's numbers for coefficients that are not identified."""
    from turbotab.core.models.repeated import gee_table, mixed_table

    rng = np.random.default_rng(3)
    n = 200
    x, w = rng.normal(size=(2, n))
    matrix = pd.DataFrame({"x": x, "w": w, "w2": 2 * w})
    y = 1 + 0.5 * x + 0.3 * w + rng.normal(size=n)
    table = (mixed_table(matrix, y, INDEPENDENT) if family == "mixed" else
             gee_table(matrix, y, INDEPENDENT, "regression"))
    assert table.info["refused"].startswith("No identifier repeats in these rows")
    rows = {r["feature"]: r for r in table.rows}
    assert rows["w"]["estimate"] is None and rows["w2"]["estimate"] is None
    ols = sm.OLS(y, sm.add_constant(matrix[["x", "w"]])).fit()
    assert rows["x"]["estimate"] == pytest.approx(float(ols.params["x"]), rel=1e-9)
    assert rows["(intercept)"]["estimate"] == pytest.approx(float(ols.params["const"]), rel=1e-9)
    assert table.unidentified == ["w", "w2"]
    assert table.concerns[0].startswith("On these 200 rows the model matrix is singular: `w` and "
                                        "`w2` are fixed combinations")


def test_two_dependencies_are_counted():
    """A copy of `x` and a column of ones: two dependencies, counted, every column in one named
    and the intercept with them; only `v` is reported."""
    x, v, ys = _rows()
    matrix = pd.DataFrame({"x": x, "v": v, "x_twice": 2 * x, "one": np.ones(N)})
    table = inference_table("regression", matrix, ys["regression"], None)
    assert table.info["refused"] is None
    assert table.concerns[0] == (
        f"On these {N} rows the model matrix is singular: `x`, `x_twice` and `one` are, in 2 ways, "
        f"fixed combinations of the other columns and the intercept, so their coefficients are not "
        f"identified, nor is the intercept. No estimate, interval or p-value is reported for `x`, "
        f"`x_twice`, `one` and the intercept; the other coefficients, which these rows identify, "
        f"are estimated with `x_twice` and `one` left out. To report every coefficient, leave out "
        f"one dependent column for each of the 2 ways.")
    assert [r["feature"] for r in table.rows if r["estimate"] is not None] == ["v"]


@pytest.mark.parametrize("task", TASKS)
def test_what_is_studied_in_a_dependency_refuses_the_whole_table(task):
    """Named as studied, `x` beside its copy refuses the table with no coefficient, in words that
    name the dependency; named so, `v`, in none, leaves the table served by row."""
    beside = "a constant" if task == "ordinal" else "the intercept"
    table = _table(task, _matrix("a copy"), study=["x"])
    words = (f"On these {N} rows the model matrix is singular: `x` and `x_twice` are fixed "
             f"combinations of the other columns and {beside}, so their coefficients are not "
             f"identified. No estimate, interval or p-value is reported; leave one of the "
             f"dependent columns out.")
    assert table.info["refused"] == words
    assert table.rows == []
    assert table.info["caption"] == f"No coefficients: {words}"
    assert [e["label"] for e in table.info["exits"]] == [
        "Leave one of the dependent columns out of the model"]
    served = _table(task, _matrix("a copy"), study=["v"])
    assert served.info["refused"] is None
    assert any(r.startswith("x") for r in served.unidentified)


def test_a_design_with_no_identified_coefficient_is_refused_whole():
    """A column of ones as the model's only column: it and the intercept are a dependency, so no
    coefficient is identified, and the table is refused with none."""
    table = inference_table("regression", pd.DataFrame({"one": np.ones(N)}),
                            _rows()[2]["regression"], None)
    assert table.rows == []
    assert table.info["refused"] == (
        f"On these {N} rows the model matrix is singular: `one` is a fixed combination of the "
        f"other columns and the intercept, so its coefficient is not identified. No estimate, "
        f"interval or p-value is reported; leave one of the dependent columns out.")


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_independent_columns_are_fit_at_statsmodels_estimates(task):
    """The guard's silent side: `x` and `v` are not refused, and the estimates are statsmodels'
    (OLS, Logit, MNLogit), with every row's interval."""
    x, v, ys = _rows()
    matrix = pd.DataFrame({"x": x, "v": v})
    table = _table(task, matrix)
    assert table.info["refused"] is None and table.unidentified == []
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
    assert table.info["refused"] is None and table.unidentified == []
    assert len(table.rows) == 4 and all(r["ci_low"] is not None for r in table.rows)


def test_a_nearly_dependent_column_is_not_refused():
    """The rule is working precision, not a near-dependency: `x` beside `x` plus noise of SD 10⁻⁴
    (a scaled condition number near 10⁴) is fit by least squares, at statsmodels' estimates."""
    x, v, ys = _rows()
    near = x + np.random.default_rng(3).normal(scale=1e-4, size=N)
    matrix = pd.DataFrame({"x": x, "v": v, "near": near})
    table = inference_table("regression", matrix, ys["regression"], None)
    assert table.info["refused"] is None and table.unidentified == []
    ols = sm.OLS(ys["regression"], sm.add_constant(matrix)).fit()
    np.testing.assert_allclose([r["estimate"] for r in table.rows], ols.params, rtol=1e-6)


def test_through_the_fit_stage_the_linear_tables_are_served_by_row(tmp_path):
    """The real split, shelf, design and fit stages under inference, one row per participant: a
    4-level `educ` beside its own 2-level recoding `educ2` (its level 2 is `educ`'s 3 and 4). The
    least-squares and logistic tables refuse by row `educ_3`, `educ_4` and `educ2_2`, in words
    naming them and not `educ_2` or `age`, and report the intercept, `educ_2` and `age` at
    statsmodels' fit with `educ2_2` left out; no concern calls the matrix nearly singular. On the
    base both tables were refused whole, with no coefficient (before package A the least-squares
    table was served with intervals for all three)."""
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
             f"coefficients are not identified. No estimate, interval or p-value is reported for "
             f"`educ_3`, `educ_4` and `educ2_2`; the other coefficients, which these rows "
             f"identify, are estimated with `educ2_2` left out. To report every coefficient, "
             f"leave one of the dependent columns out.")
    identified = pd.DataFrame({"educ_2": (educ == 2).astype(float),
                               "educ_3": (educ == 3).astype(float),
                               "educ_4": (educ == 4).astype(float), "age": age})
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
        assert model["inference"]["refused"] is None
        rows = {r["feature"]: r for r in model["coefficients"]}
        for name in ("educ_3", "educ_4", "educ2_2"):
            assert rows[name]["estimate"] is None and rows[name]["ci_low"] is None
        exog = sm.add_constant(identified)
        truth = (sm.OLS(sbp, exog).fit() if task == "regression" else
                 sm.Logit(frame["high"], exog).fit(disp=0)).params
        assert rows["age"]["estimate"] == pytest.approx(float(truth["age"]), rel=1e-6)
        assert rows["educ_2"]["estimate"] == pytest.approx(float(truth["educ_2"]), rel=1e-6)
        assert rows["age"]["ci_low"] is not None
        assert words in model["concerns"]
        assert not any("nearly singular" in c for c in model["concerns"])
