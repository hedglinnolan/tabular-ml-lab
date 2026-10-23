"""A coefficient table fit on no more rows than it has coefficients is refused, its estimates and
intervals both (the last pre-freeze verifier's V3: Model 3 on six rows reported a least-squares fit
of nine coefficients, which six rows cannot determine, with no interval, no refusal and a caption
"on t(0)").

With n rows and p coefficients, the intercept among them, least squares on a matrix of full rank
leaves n − p residual degrees of freedom. At n < p the rank is at most n, below p, so many
coefficient vectors fit the rows equally well; at n = p a matrix of full rank fits them exactly,
and nothing is left to measure its uncertainty. A logistic or multinomial likelihood on n ≤ p rows
has no unique maximum either: the rank is below p (a flat direction), or the square matrix
separates the outcome (the logistic table raised numpy's ``LinAlgError`` from Firth's fit; the
multinomial gave NaN estimates, unrefused).
Each table is refused in its existing refusal path (no rows, ``refused`` and the ways forward:
fewer columns, or more rows), wherever the linear family's table is built: the fit's, the declared
models' (``effects``), the secondary model's and the sensitivity analyses'. One row more than
coefficients and the table is fit, its estimates statsmodels' own.

Expected values come from construction (the counts of rows and of columns written out here) and
from statsmodels (the estimates).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.models import inference as I
from turbotab.core.models.inference import Outcome
from turbotab.core.tests.acceptance import estimand_fixtures as ef

COLUMNS = 8  # the matrix's columns; with the intercept, 9 coefficients


def words(p: int, n: int, per: str = "") -> str:
    return (f"The model has {p:,} coefficients{per}, the intercept among them, and is fit on "
            f"{n:,} rows: with no more rows than coefficients, the rows cannot both determine the "
            f"estimates and measure their uncertainty, so no estimate or interval is reported.")


def ways(p: int) -> list[str]:
    return ["Leave columns out of the model, so it has fewer coefficients than rows",
            f"Fit the model on more than {p:,} rows"]


def _matrix(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, COLUMNS)), columns=[f"x{j}" for j in range(COLUMNS)])


def _check_refused(table: I.InferenceTable, p: int, n: int, estimator: str,
                   per: str = "") -> None:
    reason = words(p, n, per)
    assert table.rows == []
    assert table.info["refused"] == reason
    assert table.info["covariance"] == "none" and table.info["estimator"] == estimator
    assert table.info["caption"] == f"No coefficients: {reason}"
    assert [e["label"] for e in table.info["exits"]] == ways(p)
    assert table.concerns == [reason] and table.cov is None


# ── the tables themselves ────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [6, COLUMNS + 1], ids=["fewer rows", "as many rows"])
def test_least_squares_on_no_more_rows_than_coefficients_is_refused(n):
    """9 coefficients on 6 rows (many fits), and on 9 (an exact fit with nothing left over): no
    estimate and no interval, with the reason and the ways forward. The base reported the
    pseudo-inverse's estimates, unrefused."""
    X, y = _matrix(n), np.random.default_rng(1).normal(size=n)
    _check_refused(I.inference_table("regression", X, y, None), COLUMNS + 1, n,
                   "ordinary least squares")


@pytest.mark.parametrize("n", [COLUMNS + 2, 60], ids=["one row more", "many rows more"])
def test_least_squares_on_more_rows_than_coefficients_is_fit_as_statsmodels_fits_it(n):
    X, y = _matrix(n), np.random.default_rng(1).normal(size=n)
    table = I.inference_table("regression", X, y, None)
    reference = sm.OLS(y, sm.add_constant(X)).fit()
    assert table.info["refused"] is None and len(table.rows) == COLUMNS + 1
    np.testing.assert_allclose([r["estimate"] for r in table.rows], reference.params.to_numpy(),
                               rtol=1e-9, atol=1e-12)


def test_a_logistic_table_on_six_rows_is_refused_where_it_raised():
    """The logistic table on 6 rows of 9 coefficients raised ``LinAlgError`` (Firth's fit on a
    separated outcome); it is refused in the same words. On 400 rows it is fit as statsmodels'
    ``Logit`` fits it."""
    y = np.array([0, 1, 0, 1, 1, 0])
    _check_refused(I.inference_table("binary", _matrix(6), y, [0, 1]), COLUMNS + 1, 6,
                   "logistic regression (maximum likelihood)")
    X = _matrix(400, seed=2)
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 1 / (1 + np.exp(-(0.5 * X["x0"] - 0.4 * X["x1"])))).astype(int)
    table = I.inference_table("binary", X, y, [0, 1])
    reference = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows], reference.params.to_numpy(),
                               rtol=1e-6, atol=1e-8)


def test_a_multinomial_table_on_six_rows_is_refused_where_it_was_nan():
    """Three classes on 6 rows: 9 coefficients for each level against the reference, and the
    estimates were NaN with no refusal; refused, the count said per level. On 600 rows it is fit
    as statsmodels' ``MNLogit`` fits it."""
    y = np.array([0, 1, 2, 0, 1, 2])
    per = " for each level against the reference"
    _check_refused(I.inference_table("multiclass", _matrix(6), y, [0, 1, 2]), COLUMNS + 1, 6,
                   "multinomial logistic regression (maximum likelihood)", per)
    X = _matrix(600, seed=4)
    y = np.random.default_rng(5).integers(0, 3, 600)
    table = I.inference_table("multiclass", X, y, [0, 1, 2])
    reference = sm.MNLogit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows],
                               np.asarray(reference.params).ravel(order="F"), rtol=1e-5,
                               atol=1e-7)


def test_the_refused_table_puts_nothing_on_a_scale_and_says_its_rows():
    """Through the linear family, as every stage reaches it: the refused table has no scale's
    lead in its caption (there is no estimate to read on one) and says the rows it was handed,
    with no sentence that anything was estimated from them."""
    from turbotab.core.models import get_family

    y = np.array(["no", "yes", "no", "yes", "yes", "no"])
    table = get_family("linear").inference_matrix(
        _matrix(6), (y == "yes").astype(int), task="binary", classes=[0, 1],
        clusters=I.INDEPENDENT, outcome=Outcome(name="dm", labels={0: "no", 1: "yes"}),
        rows="all")
    _check_refused(table, COLUMNS + 1, 6, "logistic regression (maximum likelihood)")
    assert (table.info["n_rows"], table.info["rows"]) == (6, "all")


# ── where the stages build it ────────────────────────────────────────────────


def _cohort(recorded: int) -> pd.DataFrame:
    """``ef.cohort(300)`` with `bmi` recorded on its first ``recorded`` rows (both values of `sex`
    among the first six, so each model codes its `male` column there)."""
    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[recorded:], "bmi"] = np.nan
    assert frame["sex"].iloc[:6].nunique() == 2
    return frame


def _ols(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    X = ef.design_matrix(frame, ["fiber", *columns])
    return sm.OLS(frame["glucose"].to_numpy(dtype=float), X).fit().params


PRIMARY = ["age", "sex", "smoking", "activity"]  # Model 2: with `fiber` and the intercept, 6
FURTHER = [*PRIMARY, "bmi"]  # Model 3 and the secondary model: 7


def _effects(tmp_path: Path, recorded: int, spline: bool = False) -> dict:
    forms = {"fiber": d.ExposureFormSpec(form="spline", knots=4)} if spline else None
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  **({"exposure_forms": forms} if forms else {}))
    return ef.sequence(ef.run(_cohort(recorded), tmp_path, st, fit=False)["effects"])


def test_model_three_on_six_rows_with_nine_coefficients_is_refused(tmp_path):
    """The verifier's V3 probe: `bmi` on six rows, `fiber` a spline of 4 knots (3 columns), so
    Model 3 has the intercept, `age`, `sex`'s `male`, `smoking`, `activity`, `bmi` and the three:
    9 coefficients on 6 rows. It reports no estimate, where it reported `fiber` −2.809 with no
    interval; Model 2, on all 300 rows, is fit."""
    seq = _effects(tmp_path, recorded=6, spline=True)
    three = seq["model_3"]
    assert three["n_rows"] == 6 and three["effects"] is None
    assert three["inference"]["refused"] == words(9, 6)
    assert [e["label"] for e in three["inference"]["exits"]] == ways(9)
    assert seq["model_2"]["n_rows"] == 300 and seq["model_2"]["effects"]


def test_model_three_on_as_many_rows_as_coefficients_is_refused(tmp_path):
    """`fiber` a straight line, `bmi` on 7 rows: Model 3 has 7 coefficients on 7 rows, refused,
    so no estimate of it has the primary's adjustment set beside it (the comparison shows the
    exposure's rows Model 3 reports, none here)."""
    three = _effects(tmp_path, recorded=7)["model_3"]
    assert three["n_rows"] == 7
    assert three["effects"] is None and three["inference"]["refused"] == words(7, 7)
    assert not three["comparison"]


def test_model_three_on_more_rows_than_coefficients_is_fit_as_statsmodels_fits_it(tmp_path):
    frame = _cohort(60)
    three = _effects(tmp_path, recorded=60)["model_3"]
    assert three["inference"]["refused"] is None
    assert three["effects"][0]["estimate"] == pytest.approx(
        _ols(frame.iloc[:60], FURTHER)["fiber"], rel=1e-8)


def _secondary(tmp_path: Path, recorded: int) -> dict:
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.secondary import secondary_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _cohort(recorded)
    st = ef.state(target="glucose", task="regression", measure="mean_difference")
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    design = design_stage(mf.context(st, inputs, paths))
    [linear] = secondary_stage(mf.context(st, {**inputs, "design": design}, paths)).data["families"]
    return {f["label"]: f for f in linear["fits"]}


def test_the_secondary_model_on_as_many_rows_as_coefficients_is_refused(tmp_path):
    """Both models on the 7 rows that record `bmi` (complete cases): the primary's 6 coefficients
    are fit, `fiber`'s estimate statsmodels'; the model further adjusted for `bmi` has 7, refused."""
    frame = _cohort(7)
    fits = _secondary(tmp_path, recorded=7)
    primary, further = fits["Primary"], fits["Further adjusted for `bmi`"]
    assert further["coefficients"] == [] and further["inference"]["refused"] == words(7, 7)
    assert primary["inference"]["refused"] is None
    assert primary["coefficients"][0]["estimate"] == pytest.approx(
        _ols(frame.iloc[:7], PRIMARY)["fiber"], rel=1e-8)


def _sensitivity(tmp_path: Path, keep: int) -> dict:
    """The sensitivity stage beside the primary (every row), with an analysis that keeps the first
    ``keep`` rows (a rule on a `visit` column that numbers them, which no model reads)."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _cohort(300).assign(visit=np.arange(300.0))
    rule = d.ExclusionRule(column="visit", low=0, high=keep - 1, reason=f"the first {keep}")
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  answers={k: v for k, v in ef.ANSWERS.items() if k != "bmi"},
                  roles={k: v for k, v in ef.ROLES.items() if k != "bmi"},
                  sensitivity=[d.SensitivityAnalysis(label="few", rules=[rule])])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    design = design_stage(mf.context(st, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    out = sensitivity_stage(mf.context(st, {**inputs, "design": design, "ingest": ingest}, paths))
    [linear] = out.data["families"]
    return {f["label"]: f for f in linear["fits"]}


@pytest.mark.parametrize("keep", [6, 7])
def test_a_sensitivity_analysis_on_no_more_rows_than_coefficients_is_refused(tmp_path, keep):
    """The primary's 6 coefficients refit on an analysis's rows: on 6 rows refused, on 7 fit,
    `fiber`'s estimate statsmodels' on those rows; the primary on all 300 is fit either way."""
    frame = _cohort(300)
    fits = _sensitivity(tmp_path, keep)
    assert fits["Primary"]["inference"]["refused"] is None and fits["Primary"]["n_rows"] == 300
    few = fits["few"]
    assert few["n_rows"] == keep
    if keep == 6:
        assert few["coefficients"] == [] and few["inference"]["refused"] == words(6, 6)
        return
    assert few["inference"]["refused"] is None
    fiber = next(r for r in few["coefficients"] if r["feature"] == "fiber")
    assert fiber["estimate"] == pytest.approx(_ols(frame.iloc[:keep], PRIMARY)["fiber"], rel=1e-8)
