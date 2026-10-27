"""The mixed model's fit on a model matrix whose columns are dependent is fit on the columns that
identify the coefficients in no dependency, as ``lme4::lmer`` drops a rank-deficient design's
columns, where it ended the fit stage and the effects stage with ``LinAlgError: the model matrix is
singular`` or ``LinAlgError: Singular matrix`` (the Sullivan 2021 walks, seeds 35, 36 and 39: a
5-level education code beside its own 4-level recoding, the mean of two days' energy intakes
beside both, a column constant on the rows), and then, from package A to the tail tables package's
first round, refused the whole fit stage in words even where the coefficient studied was
identified (T1V-4, 2026-10-27).

REML's likelihood holds ``log|XᵀV⁻¹X|``, which a matrix with dependent columns sends to −∞, and its
estimates are not unique there, so the random-intercept fit has none on them; on the columns kept
(``repeated.basis``: each dependent column that is a combination of those kept before it left out)
it has, and its predictions are the same whichever such column is left out, since the kept columns
span what all of them do. Each coefficient in a dependency (the intercept too where the constant is
in one) has no estimate (``unidentified_``). Dependence is decided at working precision on the
column-scaled design (``repeated.dependencies``), not by LAPACK's factorization of ``XᵀV⁻¹X``, whose
pivot at an exact dependency is rounding of either sign; and the fit factors the design by QR rather
than forming ``XᵀV⁻¹X``, so where the columns are nearly, not exactly, dependent it runs. The fit is
refused, in words that name every column in a dependency and count the dependencies, only where
working precision cannot tell which columns to keep. Truth from construction: a column twice
another, a column of ones beside the intercept, and two indicators that sum to one; statsmodels'
``MixedLM`` (REML) on the design with the aliased column left out gives the fixed effects and the
predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.models.repeated import RandomInterceptRegressor, fit_random_intercept

G, M = 20, 6
N = G * M


def _rows() -> tuple[np.ndarray, pd.Series, np.ndarray]:
    rng = np.random.default_rng(4)
    units = pd.Series(np.repeat(np.arange(G), M), index=pd.RangeIndex(N))
    x = rng.normal(size=N)
    y = 1 + x + rng.normal(size=G)[units.to_numpy()] + rng.normal(size=N)
    return x, units, y


def _dependent(kind: str, x: np.ndarray) -> pd.DataFrame:
    if kind == "a copy":
        return pd.DataFrame({"x": x, "x_twice": 2 * x})
    if kind == "a constant":
        return pd.DataFrame({"x": x, "one": np.ones(N)})
    return pd.DataFrame({"x": x, "low": (x <= 0).astype(float), "high": (x > 0).astype(float)})


NAMED = {"a copy": "`x` and `x_twice` are fixed combinations of the other columns and the "
                   "intercept, so their coefficients are",
         "a constant": "`one` is a fixed combination of the other columns and the intercept, so "
                       "its coefficient is",
         "indicators that sum to one": "`low` and `high` are fixed combinations of the other "
                                       "columns and the intercept, so their coefficients are"}


# By construction: the columns (and the intercept) in the dependency, and the design a fit keeping
# the earlier columns fits on.
KEPT = {"a copy": (["x", "x_twice"], ["x"]),
        "a constant": (["(intercept)", "one"], ["x"]),
        "indicators that sum to one": (["(intercept)", "low", "high"], ["x", "low"])}


def _reml_predictions(columns: pd.DataFrame, units: pd.Series, y: np.ndarray) -> np.ndarray:
    """statsmodels' ``MixedLM`` by REML on ``[1, columns]``: its fixed-effect predictions."""
    X = sm.add_constant(columns.to_numpy(dtype=float), has_constant="add")
    reml = sm.MixedLM(y, X, groups=units.to_numpy()).fit(reml=True)
    return X @ np.asarray(reml.fe_params, dtype=float)


@pytest.mark.parametrize("kind", list(NAMED))
def test_dependent_columns_fit_on_the_columns_that_identify_the_rest(kind):
    """The fit is served: its predictions are REML's (statsmodels ``MixedLM``) on the design with
    the aliased column left out, each coefficient in the dependency has no estimate in its
    coefficient rows, and the others are REML's. On the base the fit stage was refused in the
    dependency's words, with no prediction."""
    from turbotab.core.models.repeated import MIXED

    x, units, y = _rows()
    frame = _dependent(kind, x)
    model = RandomInterceptRegressor(units=units).fit(frame, y)
    named, kept = KEPT[kind]
    assert model.unidentified_ == named
    truth = _reml_predictions(frame[kept], units, y)
    np.testing.assert_allclose(model.predict(frame), truth, rtol=1e-5, atol=1e-6)
    rows = {r["feature"]: r for r in MIXED.coefficients(Pipeline([("model", model)]), frame, y,
                                                        task="regression", purpose="prediction")}
    for name in named:
        assert rows[name]["estimate"] is None
    for name in ["(intercept)", *frame.columns]:
        if name not in named:
            assert rows[name]["estimate"] is not None


def test_two_dependencies_are_counted_and_an_independent_column_is_not_named():
    """`x_twice` is twice `x` and `one` is the intercept's copy: two dependencies, every column in
    one of them, and the intercept, without an estimate, and `z`, in none, estimated (an
    independent column's weight in the dependencies' directions is rounding); the predictions are
    REML's on ``[1, x, z]``."""
    x, units, y = _rows()
    z = np.random.default_rng(5).normal(size=N)
    X = pd.DataFrame({"x": x, "z": z, "x_twice": 2 * x, "one": np.ones(N)})
    model = RandomInterceptRegressor(units=units).fit(X, y)
    assert model.unidentified_ == ["(intercept)", "x", "x_twice", "one"]
    np.testing.assert_allclose(model.predict(X), _reml_predictions(X[["x", "z"]], units, y),
                               rtol=1e-5, atol=1e-6)


def test_where_working_precision_cannot_tell_what_to_keep_the_fit_is_refused(monkeypatch):
    """The guard's firing side: where ``basis`` cannot tell which columns to keep (it returns
    None, a float64 edge no construction here pins), the fit is refused, typed
    (``singular_model_matrix``), in words naming the dependent columns."""
    from turbotab.core.models import repeated

    x, units, y = _rows()
    real = repeated.basis
    monkeypatch.setattr(repeated, "basis", lambda X: (*real(X)[:2], None))
    with pytest.raises(StageRefusal) as refused:
        RandomInterceptRegressor(units=units).fit(_dependent("a copy", x), y)
    shown, refusal = failure_of(refused.value)
    words = (f"On these {N} rows the model matrix is singular: {NAMED['a copy']} not identified. "
             f"The random-intercept mixed model is fit by REML, which needs every coefficient "
             f"identified, so it is not fit; leave one of the dependent columns out.")
    assert shown == f"ValueError: {words}"
    assert refusal is not None and refusal["code"] == "singular_model_matrix"
    assert [e["label"] for e in refusal["exits"]] == [
        "Leave one of the dependent columns out of the model"]


def test_independent_columns_fit_at_the_reml_estimate():
    """The guard's silent side: with the copy left out the fit is not refused, and its fixed
    effects are the REML fit's, which statsmodels' ``MixedLM`` agrees with."""
    x, units, y = _rows()
    model = RandomInterceptRegressor(units=units).fit(pd.DataFrame({"x": x}), y)
    X = np.column_stack([np.ones(N), x])
    fit = fit_random_intercept(X, y, units.to_numpy(), satterthwaite=False)
    np.testing.assert_allclose([model.intercept_, *model.coef_], fit.beta, rtol=1e-12)
    reml = sm.MixedLM(y, X, groups=units.to_numpy()).fit(reml=True)
    np.testing.assert_allclose(fit.beta, reml.fe_params, rtol=1e-4)
    assert model.sigma2_u_ > 0


def test_without_units_that_repeat_the_dependent_columns_fit_as_least_squares():
    """Where no unit repeats the model is least squares, which a dependent matrix does not stop
    (the minimum-norm solution): nothing to refuse, as before."""
    x, _, y = _rows()
    model = RandomInterceptRegressor(units=None).fit(_dependent("a copy", x), y)
    fitted = model.predict(_dependent("a copy", x))
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(fitted, ols.fittedvalues, rtol=1e-8)


def test_a_nearly_dependent_matrix_is_not_refused():
    """The rule is working precision (``repeated.dependent``: the smallest scaled singular value at
    most max(N, P) machine epsilons of the largest), not a near-dependency: `x` beside `x` plus
    noise of SD 10⁻³ (a scaled condition number near 10³) fits, at the REML estimate."""
    x, units, y = _rows()
    near = x + np.random.default_rng(9).normal(scale=1e-3, size=N)
    model = RandomInterceptRegressor(units=units).fit(pd.DataFrame({"x": x, "near": near}), y)
    X = np.column_stack([np.ones(N), x, near])
    reml = sm.MixedLM(y, X, groups=units.to_numpy()).fit(reml=True)
    np.testing.assert_allclose([model.intercept_, *model.coef_], reml.fe_params, rtol=1e-3)


@pytest.mark.parametrize("noise", [1e-8, 1e-12])
def test_nearly_dependent_columns_fit_where_no_dependency_is_found(noise):
    """Between the two rules: `x` beside `x` plus noise of SD 10⁻⁸ or 10⁻¹² (a scaled condition
    number near 2 × 10⁸ or 2 × 10¹²) has no dependency to working precision, and the fit runs. On
    the base, which factored ``XᵀV⁻¹X``, whose condition number is the design's squared, past
    float64's 10¹⁶, it ended in a ``LinAlgError`` and then in a refusal that called the columns
    dependent. Truth: REML depends on the design's span alone, and ``[1, x, near − x]`` spans what
    ``[1, x, near]`` does (``near − x`` is computed exactly, by Sterbenz's lemma), so γ̂ and the
    fitted values are the dense profile's minimum on that well-conditioned design. The fit is
    REML's exactly on a design within rounding of this one (a Householder QR is backward stable),
    so it is as close as the condition number allows: γ̂ and the fitted values (against the
    largest of them) to 10⁻⁷ at 10⁻⁸ and 10⁻³ at 10⁻¹²."""
    from scipy.optimize import minimize_scalar

    from turbotab.core.models.repeated import dependencies
    from turbotab.core.tests.test_no_standard_error_refusal import _dense_profile

    x, units, y = _rows()
    codes = units.to_numpy()
    near = x + np.random.default_rng(9).normal(scale=noise, size=N)
    frame = pd.DataFrame({"x": x, "near": near})
    assert dependencies(np.column_stack([np.ones(N), x, near]))[0] == 0
    model = RandomInterceptRegressor(units=units).fit(frame, y)
    d = near - x
    span = np.column_stack([np.ones(N), x, d / np.linalg.norm(d)])
    found = minimize_scalar(lambda z: _dense_profile(span, y, codes, float(np.exp(z))),
                            bounds=(-10.0, 5.0), method="bounded", options={"xatol": 1e-10})
    gamma = float(np.exp(found.x))
    Hi = np.linalg.inv(np.eye(N) + gamma * (codes[:, None] == codes[None, :]))
    fitted = span @ np.linalg.solve(span.T @ Hi @ span, span.T @ Hi @ y)
    tol = 1e-7 if noise > 1e-10 else 1e-3
    assert model.sigma2_u_ / model.sigma2_e_ == pytest.approx(gamma, rel=tol)
    np.testing.assert_allclose(model.predict(frame), fitted, rtol=0,
                               atol=tol * float(np.abs(fitted).max()))


def _with_a_recoding() -> pd.DataFrame:
    """60 participants × 4 visits: a 4-level education code and its own 2-level recoding (1–2 and
    3–4), as the Sullivan 2021 walks held a 5-level code beside its 4-level recoding; the
    recoding's indicator is the sum of two of the code's, so the model matrix is singular."""
    rng = np.random.default_rng(12)
    units, per = 60, 4
    pid = np.repeat(np.arange(units) + 1001, per)
    educ = np.repeat(rng.integers(1, 5, units), per)
    age = np.repeat(rng.integers(30, 70, units), per).astype(float)
    sbp = 110 + 0.3 * age + np.repeat(rng.normal(0, 4, units), per) + rng.normal(0, 5, units * per)
    return pd.DataFrame({"participant_id": pid, "educ": educ, "educ2": np.where(educ <= 2, 1, 2),
                         "age": age, "sbp": sbp})


def test_through_the_fit_stage_the_mixed_model_is_fit(tmp_path):
    """The real split, shelf, design and fit stages, the mixed model chosen under prediction: the
    fit stage is served, where it was refused in the dependency's words (and before package A
    failed with a bare ``LinAlgError``), and `educ_3`, `educ_4` and `educ2_2`, the recoding's
    dependency, have no estimate in its coefficient rows while `educ_2` and `age` do."""
    from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _stages

    frame = _with_a_recoding()
    st = ProjectState(lens=["clinical"], target="sbp", task="regression", purpose="prediction",
                      roles={"participant_id": "identifier", "educ": "covariate",
                             "educ2": "covariate", "age": "covariate"},
                      missing="complete_case", split=SplitSpec(holdout=0.0, seed=0, folds=5),
                      models=["mixed"],
                      grain=GrainSpec(grain="repeated", id_column="participant_id"),
                      shape_confirmations={"code_or_count:age": "amount",
                                           "code_or_count:educ": "code",
                                           "code_or_count:educ2": "code"})
    _, _, fit = _stages(frame, st, ["educ", "educ2", "age"], "regression", tmp_path)
    rows = {r["feature"]: r for r in fit.data["models"][0]["coefficients"]}
    for name in ("educ_3", "educ_4", "educ2_2"):
        assert rows[name]["estimate"] is None
    for name in ("educ_2", "age"):
        assert rows[name]["estimate"] is not None


# R 4.6.1, lme4 2.0.6: ``lmer(y ~ x + z + site + (1|pid), REML = TRUE)`` with bobyqa's
# ``rhoend = 2e-11``, on the rows below (written to CSV in full precision and read back), after its
# "fixed-effect model matrix is rank deficient so dropping 1 column / coefficient": fixef and
# sqrt(diag(vcov)) of `x` and `z`.
LMER = {"x": (0.926279721170037, 0.064729697703992), "z": (0.437512749265501, 0.0626790540358699)}


def test_the_mixed_table_beside_a_constant_column_serves_what_is_studied_at_lmer():
    """The table the fit stage reaches (T1V-4's construction: 40 units of 6 rows, `site` constant):
    named as studied, `x` is served at ``lme4::lmer``'s REML fit after it drops `site`, its
    estimate and standard error, and `z` too; `site` and the intercept, which the constant makes a
    dependency, are refused by row. (statsmodels' ``MixedLM`` agrees on the estimates to 10⁻⁶ but
    its standard errors differ from lmer's by 8·10⁻⁴.)"""
    from turbotab.core.models.inference import Clusters
    from turbotab.core.models.repeated import mixed_table

    rng = np.random.default_rng(6)
    units, per = 40, 6
    pid = np.repeat(np.arange(units), per)
    x, z = rng.normal(size=(2, units * per))
    y = 1 + x + 0.3 * z + rng.normal(size=units)[pid] + rng.normal(size=units * per)
    frame = pd.DataFrame({"x": x, "z": z, "site": 4.0})
    clusters = Clusters(column="pid", codes=pid.astype(np.int64), n_clusters=units, n_missing=0)
    table = mixed_table(frame, y, clusters, study=["x"])
    assert table.info["refused"] is None
    assert table.unidentified == ["(intercept)", "site"]
    rows = {r["feature"]: r for r in table.rows}
    for name, (estimate, se) in LMER.items():
        assert rows[name]["estimate"] == pytest.approx(estimate, rel=1e-8)
        assert rows[name]["se"] == pytest.approx(se, rel=1e-7)
    assert rows["site"]["estimate"] is None and rows["(intercept)"]["estimate"] is None
