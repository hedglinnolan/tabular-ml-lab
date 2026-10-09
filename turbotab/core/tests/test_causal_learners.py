"""A causal learner fits a yes/no variable the same way whether it is named or given as a factory.

``dml_plr`` once keyed a yes/no outcome's or exposure's classifier status on the learner being a
name other than ``"linear"``, so a flexible learner given as a factory ``(classifier, seed) ->
learner`` cross-fitted a yes/no variable as a regression rather than as a probability
(docs/turbotab-next/INBOX.md, "dml_plr fits a factory learner on a yes/no outcome as a
regressor"). Here a factory and the equivalent name give identical estimates in every estimator,
the least-squares learner keeps DoubleML's ``regr.lm`` linear-probability form however it is
given, and the partially linear model's cross-fit is computed again by hand with scikit-learn.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from turbotab.core.models import causal as est


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Boosted trees on 200 rows are dominated by thread start-up when OpenMP is unpinned (30 s a
    fit against 0.3 s); one thread keeps the suite quick, and the comparisons are like for like."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield

N = 200
FLEXIBLE = ("lasso", "nuisance_forest", "untuned_boosted_trees")


def _data(seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    d_binary = (rng.random(N) < 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))).astype(float)
    d_numeric = 0.7 * X[:, 0] + rng.normal(size=N)
    signal = 0.6 * X[:, 0] + 0.4 * X[:, 2]
    y_binary = (rng.random(N) < 1 / (1 + np.exp(-(signal + 0.9 * d_binary)))).astype(float)
    y_numeric = signal + 0.5 * d_numeric + rng.normal(size=N)
    return {"X": X, "d_binary": d_binary, "d_numeric": d_numeric, "y_binary": y_binary,
            "y_numeric": y_numeric, "y_binary_numeric_d": (
                rng.random(N) < 1 / (1 + np.exp(-(signal + 0.5 * d_numeric)))).astype(float),
            "y_numeric_binary_d": signal + 0.9 * d_binary + rng.normal(size=N)}


def _factory(name: str) -> est.Factory:
    """A factory written as a caller would write it, not the module's own ``learner_factory``."""
    return lambda classifier, seed: est.make_learner(name, classifier, seed)


def _same(a: est.Estimate, b: est.Estimate) -> None:
    assert a.estimate == b.estimate
    assert a.se == b.se
    assert a.repetitions == b.repetitions


# (outcome, exposure, outcome is yes/no)
PLR_CASES = {
    "yes/no outcome, numeric exposure": ("y_binary_numeric_d", "d_numeric", True),
    "numeric outcome, yes/no exposure": ("y_numeric_binary_d", "d_binary", False),
    "yes/no outcome, yes/no exposure": ("y_binary", "d_binary", True),
}


def test_only_the_least_squares_learner_is_least_squares_however_it_is_given():
    assert est.least_squares("linear")
    assert est.least_squares(est.learner_factory("linear"))
    assert est.least_squares(_factory("linear"))
    for name in FLEXIBLE:
        assert not est.least_squares(name)
        assert not est.least_squares(est.learner_factory(name))
        assert not est.least_squares(_factory(name))


@pytest.mark.parametrize("case", list(PLR_CASES))
@pytest.mark.parametrize("name", est.LEARNERS)
def test_the_partially_linear_model_gives_a_factory_the_named_learners_estimate(name, case):
    data = _data()
    y_key, d_key, outcome_binary = PLR_CASES[case]
    y, d, X = data[y_key], data[d_key], data["X"]
    splits = est.sample_splits(N, 5, 1, seed=3)
    named = est.dml_plr(y, d, X, learner=name, splits=splits, outcome_binary=outcome_binary,
                        seed=7)
    given = est.dml_plr(y, d, X, learner=_factory(name), splits=splits,
                        outcome_binary=outcome_binary, seed=7)
    _same(named, given)
    assert named.extra["exposure_r2"] == given.extra["exposure_r2"]
    assert named.extra["final_stage"] == given.extra["final_stage"]


def _by_hand(y: np.ndarray, d: np.ndarray, X: np.ndarray, split: list, seed: int,
             l_model, m_model) -> tuple[float, float]:
    """DoubleML's partialling-out θ and its standard error on one sample split, from scratch:
    ``l_model(seed)`` / ``m_model(seed)`` return an unfitted scikit-learn estimator and a function
    of it giving its out-of-fold prediction."""
    l_hat = np.full(N, np.nan)
    m_hat = np.full(N, np.nan)
    for k, (train, test) in enumerate(split):
        model, predict = l_model(seed + k)
        model.fit(X[train], y[train])
        l_hat[test] = predict(model, X[test])
        model, predict = m_model(seed + 500 + k)
        model.fit(X[train], d[train])
        m_hat[test] = predict(model, X[test])
    v, u = d - m_hat, y - l_hat
    theta = float(np.sum(v * u) / np.sum(v * v))
    psi = -v * v * theta + v * u
    J = float(np.mean(-v * v))
    return theta, math.sqrt(float(np.mean(psi ** 2)) / J ** 2 / N)


def _proba(model, Xt):
    return model.predict_proba(Xt)[:, list(model.classes_).index(1)]


def _value(model, Xt):
    return model.predict(Xt)


def test_an_untuned_boosted_trees_factory_cross_fits_a_yes_no_outcome_and_exposure_as_probabilities():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    data = _data()
    y, d, X = data["y_binary"], data["d_binary"], data["X"]
    splits = est.sample_splits(N, 5, 1, seed=3)
    found = est.dml_plr(y, d, X, learner=_factory("untuned_boosted_trees"), splits=splits,
                        outcome_binary=True, seed=7)

    def classifier(seed):
        return HistGradientBoostingClassifier(random_state=seed), _proba

    def regressor(seed):
        return HistGradientBoostingRegressor(random_state=seed), _value

    theta, se = _by_hand(y, d.astype(int), X, splits[0], 7, classifier, classifier)
    assert found.estimate == pytest.approx(theta, rel=1e-12, abs=1e-14)
    assert found.se == pytest.approx(se, rel=1e-12)
    # What the factory used to be fit as: a regression of each yes/no variable. A different number.
    theta_regressed, _ = _by_hand(y, d, X, splits[0], 7, regressor, regressor)
    assert abs(theta_regressed - theta) > 1e-3


def test_a_least_squares_factory_keeps_the_linear_probability_form():
    from sklearn.linear_model import LinearRegression

    data = _data()
    y, d, X = data["y_binary"], data["d_binary"], data["X"]
    splits = est.sample_splits(N, 5, 1, seed=3)
    found = est.dml_plr(y, d, X, learner=_factory("linear"), splits=splits, outcome_binary=True,
                        seed=7)

    def ols(seed):
        return LinearRegression(), _value

    theta, se = _by_hand(y, d, X, splits[0], 7, ols, ols)
    assert found.estimate == pytest.approx(theta, rel=1e-9)
    assert found.se == pytest.approx(se, rel=1e-9)


@pytest.mark.parametrize("score", ["ATE", "ATT"])
@pytest.mark.parametrize("name", ("linear", "lasso", "untuned_boosted_trees"))
def test_the_interactive_model_gives_a_factory_the_named_learners_estimate(name, score):
    data = _data()
    splits = est.sample_splits(N, 5, 1, seed=3)
    for y, binary in ((data["y_binary"], True), (data["y_numeric_binary_d"], False)):
        named = est.dml_irm(y, data["d_binary"], data["X"], learner=name, splits=splits,
                            score=score, outcome_binary=binary, seed=7)
        given = est.dml_irm(y, data["d_binary"], data["X"], learner=_factory(name), splits=splits,
                            score=score, outcome_binary=binary, seed=7)
        _same(named, given)
        assert np.array_equal(named.extra["propensity"], given.extra["propensity"])
        assert named.extra.get("risk_ratio") == given.extra.get("risk_ratio")


@pytest.mark.parametrize("name", ("linear", "lasso", "untuned_boosted_trees"))
def test_tmle_gives_a_factory_the_named_learners_estimate(name):
    data = _data()
    splits = est.sample_splits(N, 5, 1, seed=3)
    for y, family in ((data["y_binary"], "binomial"), (data["y_numeric_binary_d"], "gaussian")):
        named = est.tmle(y, data["d_binary"], data["X"], learner=name, family=family,
                         splits=splits, seed=7)
        given = est.tmle(y, data["d_binary"], data["X"], learner=_factory(name), family=family,
                         splits=splits, seed=7)
        _same(named, given)
        assert named.extra.get("risk_ratio") == given.extra.get("risk_ratio")


def test_tmle_fits_a_least_squares_factory_on_every_row_as_it_does_the_name():
    data = _data()
    for y, family in ((data["y_binary"], "binomial"), (data["y_numeric_binary_d"], "gaussian")):
        named = est.tmle(y, data["d_binary"], data["X"], learner="linear", family=family)
        given = est.tmle(y, data["d_binary"], data["X"], learner=_factory("linear"), family=family)
        _same(named, given)
    with pytest.raises(ValueError, match="cross-fitted"):
        est.tmle(data["y_binary"], data["d_binary"], data["X"], learner=_factory("lasso"),
                 family="binomial")
