"""RECIPES F11 (audit §3.6): a model that stops early scores its stopping rows with steps that never
saw them.

The early-stopping branch of ``inner_cv.fit_pipeline`` draws the stopping units first (whole units,
stratified by class, the latest when the folds follow time), fits the steps before the model on the
remaining rows only, transforms the stopping rows with those fitted steps, and only then fits the
model (RECIPES §4.3, ``fit_parts`` steps 1–4). The probe is the audit's: a step that reads the
outcome records every row it is fit on, and the model records the rows it stops on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from turbotab.core.models.inner_cv import fit_pipeline, row_keys, validation_rows


class TopByCorrelation(BaseEstimator, TransformerMixin):
    """An outcome-reading step: keeps the ``k`` columns most correlated with the outcome on the rows
    it is fit on, and records those rows and the inner splits it was handed."""

    def __init__(self, k: int = 3, cv: object = 5):
        self.k = k
        self.cv = cv

    def fit(self, X, y=None):
        self.fit_rows_ = list(X.index)
        yv = np.asarray(y, dtype=float)
        corr = [abs(np.corrcoef(X[c].to_numpy(), yv)[0, 1]) for c in X.columns]
        self.cols_ = list(np.asarray(X.columns)[np.argsort(corr)[::-1][: self.k]])
        return self

    def transform(self, X):
        return X[self.cols_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.cols_, dtype=object)


def _table(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
    X.index = pd.Index(np.arange(n) + 50_000, name="row_id")  # row ids are not positions
    return X, X["x0"].to_numpy() + rng.normal(size=n)


def _spied(monkeypatch, cls) -> list[tuple[list, list]]:
    """Each fit's (training rows, stopping rows) as the model is handed them."""
    seen: list[tuple[list, list]] = []
    original = cls.fit

    def fit(self, X, y, *args, **kwargs):
        val = kwargs.get("X_val")
        seen.append((list(X.index), None if val is None else list(val.index)))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", fit)
    return seen


def _pipeline(model) -> Pipeline:
    return Pipeline([("select", TopByCorrelation()), ("model", model)]).set_output(transform="pandas")


@pytest.mark.parametrize("units", ["rows", "people", "time"])
def test_an_outcome_reading_step_never_sees_the_stopping_rows(monkeypatch, units):
    """The selection step is fit on exactly the rows the model trains on; the stopping rows are
    transformed by that fitted step, never fit by it. Reference: the stopping rows are drawn by
    ``validation_rows`` over the same units, independently of the pipeline."""
    seen = _spied(monkeypatch, HistGradientBoostingRegressor)
    X, y = _table(900, seed=11)
    person = np.arange(len(X)) // 3
    groups = person if units != "rows" else None
    order = (np.random.default_rng(2).permutation(person.max() + 1).astype(float)[person]
             if units == "time" else None)
    model = HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.2,
                                          max_iter=30, random_state=0)
    fitted = fit_pipeline(_pipeline(model), X, y, groups=groups, order=order)
    keys = row_keys(X, y) if groups is None else None
    held = validation_rows(0.2, groups=groups, keys=keys, order=order, seed=0)
    stopping = set(X.index[held])
    train_rows, val_rows = seen[-1]
    assert set(val_rows) == stopping and len(stopping) > 0
    step = fitted.named_steps["select"]
    assert not set(step.fit_rows_) & stopping, "the outcome-reading step saw the stopping rows"
    assert step.fit_rows_ == train_rows
    # the model stops on the stopping rows as the fitted step transforms them
    assert fitted[-1].n_features_in_ == step.k
    # the step's own inner splits are drawn over the rows it is fit on, never the stopping rows
    cv = step.cv
    assert isinstance(cv, list) and cv
    positions = np.concatenate([np.concatenate([a, b]) for a, b in cv])
    assert positions.min() >= 0 and positions.max() == len(step.fit_rows_) - 1
    for a, b in cv:
        assert not set(a) & set(b)


def test_the_stopping_score_comes_from_steps_fit_on_the_training_rows_only(monkeypatch):
    """The early-stopped pipeline equals one assembled by hand: the selection fit on the rows that
    are not stopping rows, the stopping rows transformed by it, the model stopping on them. On a
    yes/no outcome the stopping units are drawn stratified by class."""
    X, y_num = _table(800, seed=5)
    y = (y_num > 0).astype(int)  # 1: yes
    model = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.25,
                                           max_iter=40, random_state=0)
    fitted = fit_pipeline(_pipeline(clone(model)), X, y)
    held = validation_rows(0.25, keys=row_keys(X, y), y=y, seed=0)
    assert abs(y[held].mean() - y.mean()) < 0.01  # stratified by class
    select = TopByCorrelation().fit(X[~held], y[~held])
    by_hand = clone(model).fit(select.transform(X[~held]), y[~held],
                               X_val=select.transform(X[held]), y_val=y[held])
    assert fitted.named_steps["select"].cols_ == select.cols_
    assert fitted[-1].n_iter_ == by_hand.n_iter_
    np.testing.assert_allclose(fitted.predict_proba(X), by_hand.predict_proba(select.transform(X)),
                               rtol=0, atol=1e-12)


def test_a_model_without_steps_stops_on_the_same_rows_as_before(monkeypatch):
    """Nothing before the model: the fit is unchanged by F11's fix (the same stopping rows, the
    model trained on the rest)."""
    seen = _spied(monkeypatch, HistGradientBoostingRegressor)
    X, y = _table(600, seed=3)
    model = HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.1,
                                          max_iter=20, random_state=0)
    fit_pipeline(Pipeline([("model", model)]), X, y)
    held = validation_rows(0.1, keys=row_keys(X, y), seed=0)
    assert seen[-1][1] == list(X.index[held]) and seen[-1][0] == list(X.index[~held])
