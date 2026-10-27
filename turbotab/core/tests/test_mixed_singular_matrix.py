"""The mixed model's fit on a model matrix whose columns are dependent is refused in words, typed as
a fit-time refusal (SWARM_PLAN §8), where it ended the fit stage and the effects stage with
``LinAlgError: the model matrix is singular`` or ``LinAlgError: Singular matrix`` (the Sullivan 2021
walks, seeds 35, 36 and 39: a 5-level education code beside its own 4-level recoding, the mean of
two days' energy intakes beside both, a column constant on the rows).

REML's likelihood holds ``log|XᵀV⁻¹X|``, which a matrix with dependent columns sends to −∞, and its
estimates are not unique there, so the random-intercept fit has none. Dependence is decided at
working precision on the column-scaled design (``repeated.dependencies``), not by LAPACK's
factorization of ``XᵀV⁻¹X``, whose pivot at an exact dependency is rounding of either sign. The
refusal names every column in a dependency (its weight in the null space's directions) and counts
the dependencies, and quotes no condition number, which at an exact dependency is rounding. Truth
from construction: a column twice another, a column of ones beside the intercept, and two
indicators that sum to one; each column of the dependency is a fixed combination of the others and
the intercept. A matrix of independent columns fits as before, at the REML estimate (statsmodels
``MixedLM``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

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


@pytest.mark.parametrize("kind", list(NAMED))
def test_dependent_columns_refuse_the_mixed_fit_in_words(kind):
    """The fit is refused, typed (``singular_model_matrix``), its words naming the dependent
    columns and why REML cannot fit them; on the base it raised a bare ``LinAlgError``."""
    x, units, y = _rows()
    with pytest.raises(StageRefusal) as refused:
        RandomInterceptRegressor(units=units).fit(_dependent(kind, x), y)
    shown, refusal = failure_of(refused.value)
    words = (f"On these {N} rows the model matrix is singular: {NAMED[kind]} not identified. The "
             f"random-intercept mixed model is fit by REML, which needs every coefficient "
             f"identified, so it is not fit; leave one of the dependent columns out.")
    assert shown == f"ValueError: {words}"
    assert refusal is not None and refusal["code"] == "singular_model_matrix"
    assert [e["label"] for e in refusal["exits"]] == [
        "Leave one of the dependent columns out of the model"]


def test_two_dependencies_are_counted_and_an_independent_column_is_not_named():
    """`x_twice` is twice `x` and `one` is the intercept's copy: two dependencies, every column in
    one of them named and `z`, in none, not named (an independent column's weight in the
    dependencies' directions is rounding)."""
    x, units, y = _rows()
    z = np.random.default_rng(5).normal(size=N)
    X = pd.DataFrame({"x": x, "z": z, "x_twice": 2 * x, "one": np.ones(N)})
    with pytest.raises(StageRefusal) as refused:
        RandomInterceptRegressor(units=units).fit(X, y)
    assert refused.value.message == (
        f"On these {N} rows the model matrix is singular: `x`, `x_twice` and `one` are, in 2 ways, "
        f"fixed combinations of the other columns and the intercept, so their coefficients are not "
        f"identified. The random-intercept mixed model is fit by REML, which needs every "
        f"coefficient identified, so it is not fit; leave out one dependent column for each of the "
        f"2 ways.")


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


def test_through_the_fit_stage_the_refusal_is_typed(tmp_path):
    """The real split, shelf, design and fit stages, the mixed model chosen under prediction: the
    fit stage is refused, typed with the refusal's code, where it failed with a bare
    ``LinAlgError`` (the job record's ``fit_failed``)."""
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
    with pytest.raises(StageRefusal) as refused:
        _stages(frame, st, ["educ", "educ2", "age"], "regression", tmp_path)
    shown, refusal = failure_of(refused.value)
    assert refusal is not None and refusal["code"] == "singular_model_matrix"
    assert shown.startswith("ValueError: On these ")
    assert " rows the model matrix is singular: `educ" in shown
