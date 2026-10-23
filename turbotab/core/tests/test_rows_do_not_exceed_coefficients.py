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

The same rule in the other tables (the repair round): the mixed model's (which raised a bare
``ValueError``), the GEE's (whose refusal below the unit floor kept the fit's estimates) and the
proportional-odds model's (unrefused), each counting its own coefficients (a proportional-odds
model's cut-points in place of the intercept); the scales stage's uncorrected table (refused, it
crashed the stage with a ``StopIteration``; the score is now not corrected, in the table's words);
and, under prediction, the sensitivity analyses' own coefficients of a family with such a table.

The words made true (the fin-freeze verifier's F1-F3): a proportional-odds likelihood on n ≤ p
rows can have a unique maximum (its cut-points are shared by every row, not free per row), so its
table is refused at the same count in words that say only that its rows are no more than its
coefficients, tested on rows where statsmodels' ``OrderedModel`` finds one (and, under prediction,
on a sensitivity analysis's own proportional-odds fit, which the fit stage reports on such rows,
so no rule across TurboTab is claimed); under prediction a multiclass fit's count is said for
each of the blocks it shows, one for every level and none against a reference, and for none
where its rows hold two levels and it shows one block (a binary fit); and the
scales stage checks the uncorrected table before the correction runs, so on fewer rows than
coefficients the score is not corrected in the table's words, not the correction's.

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


# ── the same rule in the other families' tables (repair round 1) ─────────────


def _units(n: int, G: int) -> I.Clusters:
    return I.Clusters(column="u", codes=np.arange(n) % G, n_clusters=G)


FEW = 5  # the matrix's columns for the families below; with the intercept, 6 coefficients
MIXED = "random-intercept linear mixed model (REML)"
GEE = "GEE, exchangeable working correlation"


def _few(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, FEW)), columns=[f"x{j}" for j in range(FEW)])


def test_a_mixed_table_on_as_many_rows_as_coefficients_is_refused_where_it_raised():
    """6 coefficients on 6 rows in 3 units: the base raised a bare ``ValueError`` ("6 rows for 6
    coefficients: the model cannot be fit."), a crash in every stage that builds the table. It is
    refused in the table's words, and through the family it is put on no scale and says only the
    rows it was handed (no "Estimated from" sentence beside no estimate)."""
    from sklearn.pipeline import Pipeline

    from turbotab.core.models import get_family
    from turbotab.core.models.repeated import RandomInterceptRegressor, mixed_table

    y = np.random.default_rng(1).normal(size=6)
    _check_refused(mixed_table(_few(6), y, _units(6, 3)), FEW + 1, 6, MIXED)
    # A pipeline of the model step alone: the matrix the table reads is X itself.
    only = Pipeline([("model", RandomInterceptRegressor())])
    table = get_family("mixed").inference(only, _few(6), y, task="regression",
                                          clusters=_units(6, 3), rows="all")
    _check_refused(table, FEW + 1, 6, MIXED)
    assert (table.info["n_rows"], table.info["rows"]) == (6, "all")


def test_a_mixed_table_on_more_rows_than_coefficients_is_fit_as_least_squares_by_construction():
    """8 rows in 2 units of 4, every column centered within its unit: the columns' span is then
    invariant under the random-intercept covariance, so its generalized least squares estimate is
    the ordinary one whatever the variance components (statsmodels' OLS, to 1e-9)."""
    from turbotab.core.models.repeated import mixed_table

    rng = np.random.default_rng(4)
    codes = np.repeat([0, 1], 4)
    M = _few(8, seed=4)
    M = M - M.groupby(codes).transform("mean")
    y = 1 + M.to_numpy() @ np.array([0.5, -0.2, 0.3, 0.1, 0.0]) + np.array([1.0, -1.0])[codes] \
        + rng.normal(size=8)
    table = mixed_table(M, y, I.Clusters(column="u", codes=codes, n_clusters=2))
    assert table.info["refused"] is None and len(table.rows) == FEW + 1
    np.testing.assert_allclose([r["estimate"] for r in table.rows],
                               sm.OLS(y, sm.add_constant(M)).fit().params.to_numpy(), rtol=1e-9,
                               atol=1e-12)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_a_gee_table_on_as_many_rows_as_coefficients_withholds_its_estimates(task):
    """6 coefficients on 6 rows in 3 units: the base's table was refused for the unit floor but
    kept the fit's estimates (0.300, 2.262, … for the identity link). Refused here before the
    floor, with no estimate."""
    from turbotab.core.models.repeated import gee_table

    binary = task == "binary"
    y = np.array([0, 1, 1, 0, 1, 0]) if binary else np.random.default_rng(1).normal(size=6)
    link = " (logit link)" if binary else " (identity link)"
    table = gee_table(_few(6), y, _units(6, 3), task, [0, 1] if binary else None)
    _check_refused(table, FEW + 1, 6, GEE + link)


def test_a_gee_table_on_more_rows_than_coefficients_is_fit_as_statsmodels_fits_it():
    """24 rows in 8 units (the floor), 6 coefficients: fit, its estimates statsmodels' ``GEE``
    with an exchangeable working correlation (to 1e-9)."""
    from turbotab.core.models.repeated import gee_table

    rng = np.random.default_rng(6)
    codes = np.repeat(np.arange(8), 3)
    M = _few(24, seed=6)
    y = M.to_numpy() @ np.array([0.5, -0.2, 0.3, 0.1, 0.0]) + rng.normal(size=8)[codes] \
        + rng.normal(size=24)
    table = gee_table(M, y, I.Clusters(column="u", codes=codes, n_clusters=8), "regression")
    reference = sm.GEE(y, sm.add_constant(M).to_numpy(float), groups=codes,
                       family=sm.families.Gaussian(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200, ctol=1e-10)
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows], reference.params, rtol=1e-9,
                               atol=1e-12)


ORDINAL = "proportional-odds (cumulative logit) model, maximum likelihood"


def ordinal_words(p: int, cuts: int, n: int) -> str:
    """The proportional-odds table's own words (the fin-freeze verifier's F1, and its P1): its
    likelihood on n ≤ p rows can have a unique maximum, and the fit stage under prediction reports
    such a fit, so they say neither that the rows cannot determine the estimates nor a rule across
    TurboTab, only that the rows are no more than the coefficients."""
    return (f"The model has {p:,} coefficients, its {cuts:,} cut-points among them, and is fit on "
            f"{n:,} rows, no more than its coefficients, so no estimate or interval is reported "
            f"here.")


def _finite(n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """The F1 probe: 6 rows of 4 normal slopes and a permutation of two of each of 3 levels
    (numpy's ``default_rng(8)``), so 6 coefficients, the 2 cut-points among them, on 6 rows. At
    ``n`` = 7 a seventh row (``default_rng(9)``, the middle level) is added."""
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(6, 4)), columns=[f"x{j}" for j in range(4)])
    y = rng.permutation(np.array([0, 1, 2] * 2))
    if n == 7:
        extra = pd.DataFrame(np.random.default_rng(9).normal(size=(1, 4)), columns=X.columns)
        X, y = pd.concat([X, extra], ignore_index=True), np.append(y, 1)
    return X, y


def _ordered(X: pd.DataFrame, y: np.ndarray):
    import warnings

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OrderedModel(y, X, distr="logit").fit(method="newton", maxiter=200, disp=False)


@pytest.mark.parametrize("n", [6, 7], ids=["as many rows", "one row more"])
def test_a_proportional_odds_table_whose_likelihood_has_a_finite_maximum(n):
    """On these 6 rows statsmodels' ``OrderedModel`` finds a finite maximum (converged, its
    negative Hessian positive definite: about 0.084 its smallest eigenvalue, so standard errors
    exist), where the base refused the table saying the rows cannot determine the estimates. It is
    refused in words true of it. With a seventh row the table is fit, its slopes and cut-points
    ``OrderedModel``'s to 1e-6."""
    from turbotab.core.models.ordinal import ordinal_table

    X, y = _finite(n)
    reference = _ordered(X, y)
    params = reference.params.to_numpy()
    assert reference.mle_retvals["converged"] and np.all(np.abs(params) < 10)
    assert np.linalg.eigvalsh(-reference.model.hessian(params)).min() > 0.01
    table = ordinal_table(X, y, ["a", "b", "c"], I.INDEPENDENT)
    if n == 6:
        reason = ordinal_words(6, 2, 6)
        assert table.rows == [] and table.info["refused"] == reason
        assert table.info["caption"] == f"No coefficients: {reason}"
        assert [e["label"] for e in table.info["exits"]] == ways(6)
        assert "cannot" not in reason
        return
    cuts = reference.model.transform_threshold_params(params)[1:-1]
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows],
                               np.concatenate([params[:4], cuts]), atol=1e-6)


@pytest.mark.parametrize("n", [6, FEW + 2], ids=["fewer rows", "as many rows"])
def test_a_proportional_odds_table_on_no_more_rows_than_coefficients_is_refused(n):
    """Three levels: 5 slopes and 2 cut-points, 7 coefficients. On 6 or 7 rows the base reported
    estimates of 38.6, 6.3, −39.6 … (a likelihood with no maximum), unrefused; refused here, the
    cut-points named as the coefficients that are not an input's."""
    from turbotab.core.models import get_family
    from turbotab.core.models.ordinal import ordinal_table

    y = np.array([0, 1, 2] * 3)[:n]
    table = ordinal_table(_few(n), y, ["a", "b", "c"], I.INDEPENDENT)
    reason = ordinal_words(FEW + 2, 2, n)
    assert table.rows == [] and table.info["refused"] == reason
    assert table.info["estimator"] == ORDINAL
    assert table.info["caption"] == f"No coefficients: {reason}"
    assert [e["label"] for e in table.info["exits"]] == ways(FEW + 2)
    through = get_family("proportional_odds").inference_matrix(
        _few(n), y, task="ordinal", classes=["a", "b", "c"], clusters=I.INDEPENDENT, rows="all")
    assert through.info["caption"] == f"No coefficients: {reason}"
    assert (through.info["n_rows"], through.info["rows"]) == (n, "all")


def test_a_proportional_odds_table_on_more_rows_than_coefficients_is_fit_as_statsmodels_fits_it():
    """600 rows: fit, its slopes and cut-points statsmodels' ``OrderedModel`` (logit, Newton). The
    engine stops at a Newton decrement of 1e-10, which leaves each estimate within about 1e-7 of
    the maximum, so they agree to 1e-6 (``test_wp12a_ordinal``'s tolerance), not 1e-9."""
    import warnings

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    from turbotab.core.models.ordinal import ordinal_table

    rng = np.random.default_rng(4)
    X = _few(600, seed=8)
    y = np.digitize(X.to_numpy() @ np.array([0.5, -0.3, 0.2, 0.0, 0.1]) + rng.logistic(size=600),
                    [-0.5, 0.7])
    table = ordinal_table(X, y, ["a", "b", "c"], I.INDEPENDENT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = OrderedModel(y, X, distr="logit").fit(method="newton", maxiter=200,
                                                          disp=False)
    params = reference.params.to_numpy()
    cuts = reference.model.transform_threshold_params(params)[1:-1]
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows],
                               np.concatenate([params[:FEW], cuts]), atol=1e-6)


# ── the scales stage's uncorrected table (P1) ────────────────────────────────


def _scale(tmp_path: Path, n: int, noise: int = 8) -> dict:
    """``linear_scale_table``'s first ``n`` rows, its 8 items scored as `sat_score` (internal
    consistency, regression calibration), beside `age`, `bmi` and ``noise`` columns of noise: the
    outcome model has the intercept, the score, the two and the noise, 12 coefficients."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance.scales_fixtures import SAT, SAT_REVERSE, linear_scale_table

    frame = linear_scale_table(n=300).iloc[:n].drop(columns=["sat_ref"]).copy()
    rng = np.random.default_rng(3)
    for j in range(noise):
        frame[f"z{j}"] = rng.normal(size=n)
    roles = {"age": "covariate", "bmi": "covariate", **{c: "covariate" for c in SAT},
             **{f"z{j}": "covariate" for j in range(noise)}}
    st = mf.state(purpose="inference", target="sbp", task="regression", models=["linear"],
                  roles=roles, role_confirmations=dict(roles), lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), shape_confirmations={},
                  scales=[d.ScaleSpec(name="sat_score", items=SAT, reverse=SAT_REVERSE, low=1,
                                      high=5, kind="reflective",
                                      correction="regression_calibration", n_boot=50)])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(n), holdout=0.0, seed=7)
    ti = mf.target_info("regression", "sbp")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    out = scales_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                  paths)).data
    [scale] = out["scales"]
    return {"frame": frame, "scale": scale}


@pytest.mark.parametrize("n", [11, 12], ids=["fewer rows", "as many rows"])
def test_a_scale_whose_uncorrected_table_has_no_more_rows_than_coefficients_is_not_corrected(
        tmp_path, n):
    """12 coefficients on 12 rows: the base's stage raised a bare ``StopIteration`` reading the
    refused table's row for the score (and before this package reported a correction on no
    residual degree of freedom). The score is not corrected, the table's reason its words. On 11
    the correction's error-variance refusal came first (the covariates fit those rows exactly, so
    nothing was left after them, the fin-freeze verifier's F3); the table is checked first now."""
    scale = _scale(tmp_path, n)["scale"]
    assert scale["correction"] is None
    assert scale["not_corrected"] == words(12, n)


def test_a_scale_on_one_row_more_than_coefficients_is_corrected_beside_statsmodels_table(
        tmp_path):
    """13 rows: corrected, its uncorrected estimate statsmodels' OLS of `sbp` on the score (the
    keyed items' sum, by pandas), `age`, `bmi` and the noise."""
    from turbotab.core.tests.acceptance.scales_fixtures import SAT, SAT_REVERSE, keyed

    got = _scale(tmp_path, 13)
    frame, scale = got["frame"], got["scale"]
    assert scale["not_corrected"] is None and scale["correction"] is not None
    score = keyed(frame, SAT, SAT_REVERSE, 1, 5).sum(axis=1).rename("sat_score")
    X = sm.add_constant(pd.concat([score, frame[["age", "bmi", *[f"z{j}" for j in range(8)]]]],
                                  axis=1))
    reference = sm.OLS(frame["sbp"].to_numpy(dtype=float), X).fit().params["sat_score"]
    assert scale["correction"]["naive"] == pytest.approx(reference, rel=1e-9)


# ── the sensitivity analyses under prediction ────────────────────────────────


LEVELS = np.array(["low", "mid", "high"])
IN_TURN = LEVELS[np.arange(300) % 3]  # two of each level among the first six rows


def _predicting(tmp_path: Path, keep: int, grade: np.ndarray | None = None,
                ordinal: bool = False) -> dict:
    """:func:`_sensitivity` under prediction: the linear family's own (least-squares) coefficients
    on each analysis's training rows (every row: nothing is held out). With ``grade`` the outcome
    is that 3-level column: the linear family's multinomial fit (one block for each level its
    rows hold), or with ``ordinal`` the proportional-odds model's (5 slopes and the cut-points)."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _cohort(300).assign(visit=np.arange(300.0))
    target, task, measure = "glucose", "regression", "mean_difference"
    extra: dict = {}
    if grade is not None:
        frame = frame.drop(columns=["glucose", "dm"]).assign(grade=grade)
        target, task, measure = "grade", "multiclass", "relative_risk_ratio"
        if ordinal:
            task, measure = "ordinal", "cumulative_odds_ratio"
            extra = {"models": ["proportional_odds"], "outcome_order": list(LEVELS)}
    rule = d.ExclusionRule(column="visit", low=0, high=keep - 1, reason=f"the first {keep}")
    st = ef.state(target=target, task=task, measure=measure,
                  answers={k: v for k, v in ef.ANSWERS.items() if k != "bmi"},
                  roles={k: v for k, v in ef.ROLES.items() if k != "bmi"},
                  sensitivity=[d.SensitivityAnalysis(label="few", rules=[rule])], **extra)
    st = st.model_copy(update={"purpose": "prediction"})
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    design = design_stage(mf.context(st, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    return sensitivity_stage(mf.context(st, {**inputs, "design": design, "ingest": ingest},
                                        paths)).data


@pytest.mark.parametrize("keep", [6, 7])
def test_under_prediction_an_analysis_on_no_more_rows_than_coefficients_reports_none(tmp_path,
                                                                                    keep):
    """The primary's 6 least-squares coefficients refit on an analysis's rows under prediction: on
    6 rows the base reported them (`fiber` −0.608) and took them into the range of `fiber`'s
    estimates; refused here with the table's reason, so the range is the primary's alone. On 7
    they are reported, `fiber`'s statsmodels' OLS on those rows."""
    frame = _cohort(300)
    out = _predicting(tmp_path, keep)
    [linear] = out["families"]
    fits = {f["label"]: f for f in linear["fits"]}
    primary = next(r for r in fits["Primary"]["coefficients"] if r["feature"] == "fiber")
    assert primary["estimate"] == pytest.approx(_ols(frame, PRIMARY)["fiber"], rel=1e-9)
    few = fits["few"]
    assert few["n_rows"] == keep
    [change] = out["changes"]["linear"]
    if keep == 6:
        assert few["coefficients"] == [] and few["inference"]["refused"] == words(6, 6)
        assert few["concerns"] == [words(6, 6)]
        assert [e["label"] for e in few["inference"]["exits"]] == ways(6)
        assert change["lowest"] == change["highest"] == change["primary"] == primary["estimate"]
        return
    assert few["inference"] is None
    fiber = next(r for r in few["coefficients"] if r["feature"] == "fiber")
    assert fiber["estimate"] == pytest.approx(_ols(frame.iloc[:keep], PRIMARY)["fiber"], rel=1e-9)
    assert {change["lowest"], change["highest"]} == {primary["estimate"], fiber["estimate"]}


@pytest.mark.parametrize("keep", [6, 7])
def test_under_prediction_a_multiclass_analysis_counts_its_coefficients_per_level_as_shown(
        tmp_path, keep):
    """Three levels under prediction: the primary reports 3 blocks of 6 coefficients (the
    intercept and the 5 inputs, one block for each level, none against a reference), and an
    analysis on 6 rows is refused in words that count them so, where the base said "for each
    level against the reference" (the fin-freeze verifier's F2). On 7 rows all 18 are reported."""
    out = _predicting(tmp_path, keep, IN_TURN)
    [linear] = out["families"]
    fits = {f["label"]: f for f in linear["fits"]}
    per_level = 1 + len(PRIMARY) + 1  # the intercept, `fiber` and the 4 (`sex` one indicator)
    shown = [str(r["feature"]) for r in fits["Primary"]["coefficients"]]
    assert len(shown) == 3 * per_level
    assert sorted(f for f in shown if f.startswith("(intercept)")) == [
        "(intercept) [high]", "(intercept) [low]", "(intercept) [mid]"]
    few = fits["few"]
    assert few["n_rows"] == keep
    if keep == 6:
        reason = words(per_level, 6, " for each of its 3 levels")
        assert few["coefficients"] == [] and few["inference"]["refused"] == reason
        assert few["concerns"] == [reason] and "reference" not in reason
        assert [e["label"] for e in few["inference"]["exits"]] == ways(per_level)
        return
    assert few["inference"] is None and len(few["coefficients"]) == 3 * per_level


def test_under_prediction_a_multiclass_analysis_on_two_levels_counts_the_one_block_it_shows(
        tmp_path):
    """The analysis's 6 rows alternate `low` and `mid` (the outcome's third level, `high`, is on
    later rows), so its fit is binary and shows one block of 6 coefficients, one intercept among
    them. The base said "for each level against the reference" (the task stayed multiclass; the
    fin-freeze verifier's P2); the count is said for no level now, as the one block shows it."""
    grade = np.where(np.arange(300) < 6, LEVELS[np.arange(300) % 2], IN_TURN)
    assert set(grade[:6]) == {"low", "mid"} and set(grade) == set(LEVELS)
    out = _predicting(tmp_path, 6, grade)
    [linear] = out["families"]
    few = {f["label"]: f for f in linear["fits"]}["few"]
    per_level = 1 + len(PRIMARY) + 1  # the intercept, `fiber` and the 4 (`sex` one indicator)
    reason = words(per_level, 6)
    assert few["n_rows"] == 6 and few["coefficients"] == []
    assert few["inference"]["refused"] == reason and "reference" not in reason
    assert few["concerns"] == [reason]


@pytest.mark.parametrize("keep", [7, 8])
def test_under_prediction_a_proportional_odds_analysis_on_its_coefficients_rows_is_refused(
        tmp_path, keep):
    """The proportional-odds model's own coefficients under prediction: 5 slopes and the 2
    cut-points of 3 levels (all three among the first 7 rows), 7 coefficients. On 7 rows they are
    refused in the proportional-odds words, which claim no rule across TurboTab (the fit stage
    reports such a fit under prediction: the fin-freeze verifier's P1); on 8 all 7 are reported."""
    assert set(IN_TURN[:7]) == set(LEVELS)
    out = _predicting(tmp_path, keep, IN_TURN, ordinal=True)
    [family] = out["families"]
    fits = {f["label"]: f for f in family["fits"]}
    slopes = 1 + len(PRIMARY)  # `fiber` and the 4 (`sex` one indicator)
    shown = [str(r["feature"]) for r in fits["Primary"]["coefficients"]]
    assert len(shown) == slopes + 2 and sum(f.startswith("(cut-point") for f in shown) == 2
    few = fits["few"]
    assert few["n_rows"] == keep
    if keep == 7:
        reason = ordinal_words(slopes + 2, 2, 7)
        assert few["coefficients"] == [] and few["inference"]["refused"] == reason
        assert few["concerns"] == [reason]
        assert [e["label"] for e in few["inference"]["exits"]] == ways(slopes + 2)
        return
    assert few["inference"] is None and len(few["coefficients"]) == slopes + 2
