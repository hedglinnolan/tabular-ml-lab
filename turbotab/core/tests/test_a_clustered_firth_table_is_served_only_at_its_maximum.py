"""A Firth table on clustered rows, or on too few units, serves its estimates only where the fit is
at its penalized maximum, as one on independent rows does (T5dV-4).

The clustering refusal (and the unit floor's) was set first, and the at-maximum check ran only
where no refusal was set, so on clustered separated rows the estimates of a Firth fit that ran its
300 Newton steps without converging were served, 7 to 9.6 standard errors from the maximum
(beside a near copy of `x1`, ``x1 + 10⁻⁶·N(0, 1)``, which passes the aliasing check), where the
same fit on independent rows is refused with no estimate. Now the check runs whatever else
refuses the intervals: a fit short of its maximum gives no estimate, in the at-maximum words
followed by the clustering (or floor) words, with their exits; a converged fit's estimates are
served beside the clustering words as before.

Truth: separated by construction (``test_firth_serves_its_estimates_alone_under_separation``'s
designs, checked by scipy's linear program there); Firth's maximum by scipy's BFGS on the
penalized log-likelihood written out with numpy, from several starts (that file's reference); the
words from ``inference.at_maximum`` with the steps and distance the fit reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import inference as I
from turbotab.core.models.inference import inference_table, min_clusters
from turbotab.core.tests.test_firth_serves_its_estimates_alone_under_separation import (
    COLUMNS, _design, _indicator, _listed, _reference, _separated)


def _clusters(n: int, size: int) -> I.Clusters:
    codes = np.arange(n) // size
    return I.Clusters(column="pid", codes=codes, n_clusters=int(codes.max()) + 1, n_missing=0)


def _repeat_words(named: list[str]) -> str:
    verb = "separates" if len(named) == 1 else "separate"
    return (f"{_listed(named)} {verb} the outcome and the rows repeat by `pid`: Firth's profile "
            f"intervals assume independent rows, so no interval or p-value is reported.")


def _complete() -> tuple[pd.DataFrame, np.ndarray]:
    rows = _separated(0, 120)
    return rows[COLUMNS], rows["y"].to_numpy()


def _with_near_copy(matrix: pd.DataFrame, eps: float) -> pd.DataFrame:
    return matrix.assign(near=matrix["x1"] + eps * np.random.default_rng(7).normal(size=len(matrix)))


DESIGNS = {"complete 0": _complete, "indicator 11": lambda: _indicator(11)}


@pytest.mark.parametrize("design", list(DESIGNS))
def test_on_clustered_rows_a_converged_firth_fits_estimates_are_served(design):
    """The silent side: rows clustered in fours, the fit converged, so the estimates are served,
    each within 10⁻⁶ of a standard error of the independent maximum, refused for every interval
    and p-value in the clustering words with their exits (as before)."""
    matrix, y = DESIGNS[design]()
    X, yv = _design(matrix), y.astype(float)
    assert I.firth_fit(X, yv).converged
    table = inference_table("binary", matrix, y, [0, 1], _clusters(len(y), 4),
                            study=[matrix.columns[0]])
    named = table.info["separated"]
    assert table.info["estimator"] == I.FIRTH and named
    assert table.info["refused"] == _repeat_words(named)
    assert [e["label"] for e in table.info["exits"]] == [
        "Combine each `pid`'s rows into one (the unit question, before the seal)",
        f"Leave {_listed(named)} out, or merge its rare levels"]
    beta, _, se = _reference(X, yv)
    for j, row in enumerate(table.rows):
        assert row["se"] is None and row["p"] is None
        assert abs(row["estimate"] - beta[j]) <= 1e-6 * se[j], (design, row["feature"])


@pytest.mark.parametrize("design", list(DESIGNS))
def test_on_clustered_rows_a_firth_fit_that_did_not_converge_serves_no_estimate(design):
    """Beside a near copy of `x1` (``10⁻⁶`` of noise) the fit runs its 300 Newton steps without
    converging; on rows clustered in fours no estimate is served, in the at-maximum words (the
    steps the fit reports) followed by the clustering words, with the clustering exits. On the
    base every estimate was served."""
    matrix, y = DESIGNS[design]()
    matrix = _with_near_copy(matrix, 1e-6)
    X, yv = _design(matrix), y.astype(float)
    fit = I.firth_fit(X, yv)
    assert not fit.converged and fit.iterations == 300
    table = inference_table("binary", matrix, y, [0, 1], _clusters(len(y), 4),
                            study=[matrix.columns[0]])
    named = table.info["separated"]
    short = I.at_maximum("Firth's penalized fit", I.firth_distance(X, yv, fit.beta),
                         converged=False, steps=300)
    assert table.info["refused"] == f"{short} {_repeat_words(named)}"
    assert table.info["refused"].startswith(
        "Firth's penalized fit did not converge in 300 Newton steps, so no estimate")
    assert len(table.info["exits"]) == 2
    assert all(row["estimate"] is None for row in table.rows)
    independent = inference_table("binary", matrix, y, [0, 1], I.INDEPENDENT,
                                  study=[matrix.columns[0]])
    assert independent.info["refused"] == short  # as on independent rows


@pytest.mark.parametrize("clusters", ["in fours", "too few units"])
def test_a_firth_fit_stopped_early_on_clustered_rows_serves_no_estimate(clusters, monkeypatch):
    """A fit stopped after 4 steps that claims convergence: on rows clustered in fours, and on
    rows in fewer units than the floor (refused for that), it is refused as stopped short of its
    maximum with no estimate, in words that state the distance the table compares, followed by
    the clustering or floor words; the same rows with the fit run to its maximum serve every
    estimate (the floor's case here, the clustered case above)."""
    matrix, y = _complete()
    size = 4 if clusters == "in fours" else 30
    groups = _clusters(len(y), size)
    assert (groups.n_clusters >= min_clusters()) == (clusters == "in fours")
    served = inference_table("binary", matrix, y, [0, 1], groups, study=["x1"])
    assert all(r["estimate"] is not None for r in served.rows)
    floor_or_repeat = served.info["refused"]
    real = I.firth_fit
    seen: dict[str, float] = {}

    def early(X: np.ndarray, y: np.ndarray, **kw: object) -> I.FirthFit:
        fit = real(X, y, max_iter=4)
        seen["distance"] = I.firth_distance(X, y, fit.beta)
        return I.FirthFit(beta=fit.beta, loglik=fit.loglik, cov=fit.cov,
                          iterations=fit.iterations, converged=True)

    monkeypatch.setattr(I, "firth_fit", early)
    table = inference_table("binary", matrix, y, [0, 1], groups, study=["x1"])
    assert seen["distance"] > 10 * I.SHORT_OF_MAXIMUM
    short = I.at_maximum("Firth's penalized fit", seen["distance"])
    assert table.info["refused"] == f"{short} {floor_or_repeat}"
    assert table.info["exits"] == served.info["exits"]
    assert all(r["estimate"] is None for r in table.rows)
