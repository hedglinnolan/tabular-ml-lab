"""The selection step in the Models preview, and a selection that keeps no predictor.

Two failures from Nolan's first walk on Windows (2026-10-20), neither of them Windows':

1. With a selection answer under Predict, every preview of the model choice answered 500
   ("Selection needs the outcome on the fold."): the preview fitted each family's steps without
   the outcome, and the selection step reads it. The preview now shows the candidate columns the
   selection chooses among, and its caption names the selection and says it runs in each training
   fold; the matrix it shows is the one without a selection, the candidates themselves.
2. The fit stopped inside the tuning with scikit-learn's "at least one array or dtype is
   required" when a training fold's selection kept no predictor and the scaling after it got a
   table with no column. The model there would be the intercept alone, which this version does not
   fit, so the step now refuses in words, with the ways forward, when it would hand on no column;
   the fit that kept nothing stays inspectable, as the inclusion frequencies across folds read it.

The empty selection is built, not found: an outcome and candidates that are columns of a Hadamard
matrix are exactly uncorrelated, so no screen, penalty or elimination has anything to keep; one
candidate made to carry the outcome is kept, the guard's other side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard

from turbotab.core import decisions as d
from turbotab.core.models.variable_selection import NAMES, Selector
from turbotab.core.tests.acceptance import test_previews_leash as L
from turbotab.core.tests.acceptance.preview_harness import Project

N = 64
EMPTYING = ["univariable", "elastic_net", "stepwise"]


def orthogonal(carried: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Six candidates exactly uncorrelated with the outcome (and with each other); with
    ``carried``, the first is the outcome plus an orthogonal column, correlation 1/sqrt(1.25)."""
    H = hadamard(N).astype(float)
    y = pd.Series(H[:, 1], name="y")
    X = pd.DataFrame(H[:, 2:8], columns=[f"x{i}" for i in range(6)])
    if carried:
        X["x0"] = H[:, 1] + 0.5 * H[:, 9]
    return X, y


def selector(method: str, columns: list[str]) -> Selector:
    return Selector(method, "regression", None, None, None, 50, 5, 0, None, columns)


@pytest.mark.parametrize("method", EMPTYING)
def test_a_selection_that_keeps_no_predictor_refuses_in_words(method):
    X, y = orthogonal(carried=False)
    assert np.allclose(X.T.to_numpy() @ (y - y.mean()).to_numpy(), 0)  # the construction
    fitted = selector(method, list(X.columns)).fit(X, y)
    assert fitted.selected_ == []  # the fit itself stays inspectable (inclusion frequencies)
    with pytest.raises(ValueError) as refused:
        fitted.transform(X)
    said = str(refused.value)
    assert said.startswith(NAMES[method][0].upper() + NAMES[method][1:] + " kept no predictor")
    assert f"on {N:,} training rows" in said
    assert "intercept alone" in said and "No selection" in said


@pytest.mark.parametrize("method", EMPTYING)
def test_a_selection_that_keeps_a_predictor_fits_as_before(method):
    X, y = orthogonal(carried=True)
    fitted = selector(method, list(X.columns)).fit(X, y)
    assert "x0" in fitted.kept_


def test_no_selection_never_refuses_on_the_same_rows():
    X, y = orthogonal(carried=False)
    assert selector("none", list(X.columns)).fit(X, y).kept_ == list(X.columns)


@pytest.fixture(scope="module")
def previews(tmp_path_factory):
    """The Models preview of the linear family on the walk's table under Predict, with the elastic
    net chosen as the selection, and with no selection."""
    out = {}
    for method in ("elastic_net", "none"):
        state = L.walk_state(purpose="prediction", selection=d.SelectionSpec(method=method))
        project = Project(L.walk_table(False), Path(tmp_path_factory.mktemp(method)), state,
                          upto=["design", "cohort", "split"])
        try:
            out[method] = L.served(project, d.SelectModels(models=["linear"]), state)
        finally:
            project.close()
    return out


def test_the_models_preview_with_a_selection_shows_the_candidates_and_names_the_selection(previews):
    chosen, none = previews["elastic_net"], previews["none"]
    assert "refused" not in chosen and chosen["views"], chosen
    caption = chosen["views"][0]["caption"]
    assert f"{NAMES['elastic_net']} picking among them in each training fold" in caption
    # The candidates it picks among are the matrix without a selection, column for column.
    assert chosen["views"][0]["after"] == none["views"][0]["after"]


def test_the_models_preview_without_a_selection_names_none(previews):
    caption = previews["none"]["views"][0]["caption"]
    assert "picking among them" not in caption


@pytest.fixture(scope="module")
def selection_previews(tmp_path_factory):
    """The selection question's own preview of the elastic net under Predict, on the walk's table
    (glucose depends on the diet, age, BMI and sex) and on the same table with its outcome re-paired
    (glucose given to the rows in a fixed random order, so it relates to nothing)."""
    out = {}
    state = L.walk_state(purpose="prediction")
    for repaired in (False, True):
        project = Project(L.walk_table(repaired), Path(tmp_path_factory.mktemp(f"sel{repaired}")),
                          state, upto=["design", "cohort", "split"])
        try:
            out[repaired] = L.served(project, d.SetSelection(method="elastic_net"), state)
        finally:
            project.close()
    return out


def test_a_selection_that_keeps_nothing_says_so_and_that_fit_would_refuse(selection_previews):
    served = selection_previews[True]
    assert "refused" not in served and served["views"], served
    caption = served["views"][0]["caption"]
    assert "it keeps none of" in caption and "Fit would refuse" in caption


def test_a_selection_that_keeps_columns_says_how_many_as_before(selection_previews):
    caption = selection_previews[False]["views"][0]["caption"]
    assert "it keeps none" not in caption and "every training fold repeats it" in caption


def test_the_kept_columns_of_a_fit_that_kept_nothing_are_none_without_a_refusal():
    X, y = orthogonal(carried=False)
    fitted = selector("elastic_net", list(X.columns)).fit(X, y)
    assert list(fitted.kept_columns(X).columns) == []
    X, y = orthogonal(carried=True)
    fitted = selector("elastic_net", list(X.columns)).fit(X, y)
    assert fitted.kept_columns(X).equals(fitted.transform(X))
