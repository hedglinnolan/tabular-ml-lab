"""The causal lasso's own folds keep each unit whole (RT-11; RECIPES F9).

The lasso nuisance learners choose their penalty by an inner cross-validation: ``LassoCV`` for a
number, :class:`~turbotab.core.models.causal.L1LogisticCV` for a yes/no variable. Drawn by row
(``KFold`` / ``StratifiedKFold``), one person's rows sit on both sides of an inner fold, and the
penalty is chosen on rows that share a person with the rows scoring it. Under a cluster or survey
design the inner folds are now drawn from whole units (PSUs), through the inner-split rule
(:func:`turbotab.core.models.inner_cv.inner_splits`). Without repeated units nothing changes.

References are independent of the code: scikit-learn's own splitters (``KFold``,
``StratifiedKFold``, ``GroupKFold``) and ``LassoCV`` fit on explicit folds.
"""
from __future__ import annotations

import numpy as np
import pytest
import sklearn.linear_model._coordinate_descent as coordinate_descent
import sklearn.linear_model._logistic as logistic
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.model_selection import check_cv as sklearn_check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from turbotab.core.models import causal as est

PEOPLE = 120
ROWS_EACH = 3


def _people_data(seed: int = 5) -> dict[str, np.ndarray]:
    """120 people with 3 rows each: a person's own effect is shared by all of their rows."""
    rng = np.random.default_rng(seed)
    n = PEOPLE * ROWS_EACH
    person = np.repeat(np.arange(PEOPLE), ROWS_EACH)
    labels = np.array([f"p{i:03d}" for i in person], dtype=object)
    own = rng.normal(size=PEOPLE)[person]
    X = rng.normal(size=(n, 4)) + 0.8 * own[:, None]
    d_numeric = 0.6 * X[:, 0] + 0.5 * own + rng.normal(size=n)
    d_binary = (rng.random(n) < 1 / (1 + np.exp(-(0.7 * X[:, 0] - 0.4 * X[:, 1] + own)))).astype(
        float)
    y = 0.5 * X[:, 0] - 0.3 * X[:, 2] + 0.4 * d_numeric + own + rng.normal(size=n)
    return {"X": X, "y": y, "d_numeric": d_numeric, "d_binary": d_binary, "person": person,
            "labels": labels}


class _Recorder:
    """A recording splitter: wraps the splitter each lasso resolves and keeps the folds it yields."""

    def __init__(self) -> None:
        self.folds: list[list[tuple[np.ndarray, np.ndarray]]] = []

    def check_cv(self, cv=5, y=None, *, classifier=False):
        inner = sklearn_check_cv(cv, y, classifier=classifier)
        recorder = self

        class Recording:
            def get_n_splits(self, *args, **kwargs):
                return inner.get_n_splits(*args, **kwargs)

            def split(self, *args, **kwargs):
                folds = [(np.asarray(a), np.asarray(b)) for a, b in inner.split(*args, **kwargs)]
                recorder.folds.append(folds)
                return iter(folds)

        return Recording()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    found = _Recorder()
    monkeypatch.setattr(coordinate_descent, "check_cv", found.check_cv)
    monkeypatch.setattr(logistic, "check_cv", found.check_cv)
    return found


def _fit_rows(splits: list, fits: int) -> list[np.ndarray]:
    """The training rows of each lasso fit, in the order cross_fit makes them."""
    return [splits[k % len(splits)][0] for k in range(fits)]


def _straddlers(person: np.ndarray, rows: np.ndarray, folds: list) -> set[int]:
    out: set[int] = set()
    for train, test in folds:
        out |= set(person[rows[train]]) & set(person[rows[test]])
    return out


def test_no_person_sits_on_both_sides_of_a_lasso_fold_in_the_partially_linear_model(recorder):
    data = _people_data()
    person, n = data["person"], len(data["y"])
    design = est.Design(psu=person, label="clusters")
    splits = est.sample_splits(n, 5, 1, seed=3, groups=person)
    est.dml_plr(data["y"], data["d_numeric"], data["X"], learner="lasso", splits=splits,
                design=design, seed=3)
    assert len(recorder.folds) == 10  # ℓ and m, five outer folds each
    for rows, folds in zip(_fit_rows(splits[0], 10), recorder.folds):
        assert len(folds) == 5
        assert sum(len(test) for _, test in folds) == len(rows)
        assert _straddlers(person, rows, folds) == set()


def test_no_person_sits_on_both_sides_of_an_l1_logistic_fold(recorder):
    data = _people_data()
    person, n = data["person"], len(data["y"])
    splits = est.sample_splits(n, 5, 1, seed=4, groups=person)
    learner = est.learner_factory("lasso")
    est.cross_fit(learner, True, data["X"], data["d_binary"], splits[0], seed=4, groups=person)
    assert len(recorder.folds) == 5
    for rows, folds in zip(_fit_rows(splits[0], 5), recorder.folds):
        assert sum(len(test) for _, test in folds) == len(rows)
        assert _straddlers(person, rows, folds) == set()
        for train, _ in folds:  # every training fold still holds both levels
            assert len(np.unique(data["d_binary"][rows[train]])) == 2


def test_the_interactive_model_and_tmle_keep_people_whole_in_every_lasso_fold(recorder):
    data = _people_data()
    person, n = data["person"], len(data["y"])
    design = est.Design(psu=person, label="clusters")
    splits = est.sample_splits(n, 5, 1, seed=6, groups=person)
    est.dml_irm(data["y"], data["d_binary"], data["X"], learner="lasso", splits=splits,
                design=design, seed=6)
    est.tmle(data["y"], data["d_binary"], data["X"], learner="lasso", design=design,
             splits=splits, seed=6)
    assert len(recorder.folds) == 15 + 10
    d = data["d_binary"]
    expected: list[np.ndarray] = []
    for mask in (None, d == 0, d == 1):  # m, g₀, g₁
        for train, _ in splits[0]:
            expected.append(train if mask is None else train[mask[train]])
    expected += [train for train, _ in splits[0]] * 2  # TMLE's Q, then g
    for rows, folds in zip(expected, recorder.folds):
        assert sum(len(test) for _, test in folds) == len(rows)
        assert _straddlers(person, rows, folds) == set()


def test_without_repeats_the_folds_are_scikit_learns_own(recorder):
    """One row per unit (no design, or a PSU per row): the regression lasso draws
    ``KFold(5, shuffle)`` and the logistic lasso ``StratifiedKFold(5, shuffle)`` by the fit's
    seed, as before."""
    data = _people_data()
    X, y, d = data["X"], data["y"], data["d_binary"]
    n = len(y)
    rows = np.arange(n)
    split = [(rows[rows % 4 != k], rows[rows % 4 == k]) for k in range(4)]
    for groups in (None, np.arange(n)):
        recorder.folds.clear()
        est.cross_fit(est.learner_factory("lasso"), False, X, y, split, seed=9, groups=groups)
        est.cross_fit(est.learner_factory("lasso"), True, X, d, split, seed=9, groups=groups)
        assert len(recorder.folds) == 8
        for k, (train, _) in enumerate(split):
            want = KFold(5, shuffle=True, random_state=9 + k).split(X[train])
            for (a, b), (c, e) in zip(recorder.folds[k], want):
                assert np.array_equal(a, c) and np.array_equal(b, e)
            want = StratifiedKFold(5, shuffle=True, random_state=9 + k).split(X[train], d[train])
            for (a, b), (c, e) in zip(recorder.folds[4 + k], want):
                assert np.array_equal(a, c) and np.array_equal(b, e)


def _row_lasso(classifier: bool, seed: int) -> est._Sklearn:
    """The lasso learners as they were drawn by row: explicit scikit-learn splitters."""
    if classifier:
        return est._Sklearn(make_pipeline(StandardScaler(), est.L1LogisticCV(
            seed=seed, cv=StratifiedKFold(5, shuffle=True, random_state=seed))), True)
    return est._Sklearn(make_pipeline(StandardScaler(), LassoCV(
        alphas=100, eps=1e-4, cv=KFold(5, shuffle=True, random_state=seed), max_iter=50_000,
        tol=1e-7, random_state=seed)), False)


@pytest.mark.parametrize("unit", ("none", "a PSU per row"))
def test_without_repeats_the_dml_estimates_are_bit_equal_to_row_folds(unit):
    data = _people_data()
    X, y = data["X"], data["y"]
    n = len(y)
    design = None if unit == "none" else est.Design(psu=np.arange(n), label="clusters")
    splits = est.sample_splits(n, 5, 1, seed=2)
    for d, method in ((data["d_numeric"], "plr"), (data["d_binary"], "irm")):
        run = est.dml_plr if method == "plr" else est.dml_irm
        named = run(y, d, X, learner="lasso", splits=splits, design=design, seed=2)
        by_row = run(y, d, X, learner=_row_lasso, splits=splits, design=design, seed=2)
        assert named.estimate == by_row.estimate
        assert named.se == by_row.se


def test_the_grouped_lasso_chooses_the_alpha_lassocv_chooses_on_explicit_group_folds():
    """Reference: scikit-learn's ``LassoCV`` given ``GroupKFold(5, shuffle)`` folds over the same
    people by the same seed chooses the same penalty."""
    data = _people_data()
    X, y, labels = data["X"], data["y"], data["labels"]
    keep = np.arange(len(y)) >= 30  # a training fold's rows: whole people, not a full table
    X, y, labels = X[keep], y[keep], labels[keep]
    seed = 7
    learner = est.make_learner("lasso", False, seed).fit(X, y, groups=labels)
    found = learner.model_.steps[-1][1]
    folds = list(GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=labels))
    reference = make_pipeline(StandardScaler(), LassoCV(
        alphas=100, eps=1e-4, cv=folds, max_iter=50_000, tol=1e-7, random_state=seed)).fit(X, y)
    assert found.alpha_ == reference.steps[-1][1].alpha_
    # The folds by row would have chosen another penalty here: the units matter.
    by_row = make_pipeline(StandardScaler(), LassoCV(
        alphas=100, eps=1e-4, cv=KFold(5, shuffle=True, random_state=seed), max_iter=50_000,
        tol=1e-7, random_state=seed)).fit(X, y)
    assert by_row.steps[-1][1].alpha_ != found.alpha_


@pytest.mark.parametrize("classifier", (False, True), ids=("numeric", "yes/no"))
def test_a_partition_drawn_by_unit_keeps_people_whole_in_a_cross_fit_that_names_no_units(
        recorder, classifier):
    """The causal stage's positivity gate and the method preview read the estimator's own
    propensity with ``cross_fit(factory, ..., splits[0], weights, seed)`` and name no units. A
    partition drawn by unit carries its units, so that propensity's lasso folds keep people whole
    as the estimator's do, and it is the propensity the estimator itself fits."""
    data = _people_data()
    person, n = data["person"], len(data["y"])
    d = data["d_binary"] if classifier else data["d_numeric"]
    splits = est.sample_splits(n, 5, 1, seed=8, groups=person)
    lasso = est.learner_factory("lasso")
    [gate] = est.cross_fit(lasso, classifier, data["X"], d, splits[0], None, 8)
    assert len(recorder.folds) == 5
    for rows, folds in zip(_fit_rows(splits[0], 5), recorder.folds):
        assert sum(len(test) for _, test in folds) == len(rows)
        assert _straddlers(person, rows, folds) == set()
    [own] = est.cross_fit(lasso, classifier, data["X"], d, splits[0], None, 8, groups=person)
    assert np.array_equal(gate, own)


def test_a_partition_without_repeated_units_still_draws_the_lasso_folds_by_row(recorder):
    """A PSU per row deals the same partition's folds by unit, yet no unit repeats: the lasso's own
    folds stay ``KFold(5, shuffle)`` by the fit's seed."""
    data = _people_data()
    X, y = data["X"], data["y"]
    n = len(y)
    splits = est.sample_splits(n, 4, 1, seed=1, groups=np.arange(n))
    est.cross_fit(est.learner_factory("lasso"), False, X, y, splits[0], seed=9)
    assert len(recorder.folds) == 4
    for k, (train, _) in enumerate(splits[0]):
        want = KFold(5, shuffle=True, random_state=9 + k).split(X[train])
        for (a, b), (c, e) in zip(recorder.folds[k], want):
            assert np.array_equal(a, c) and np.array_equal(b, e)
