"""The guard that refuses a table whose fit gives some coefficient no standard error
(``inference.no_interval``), tested both ways in each table builder that makes Wald or t rows and
is not tested in ``test_no_standard_error_refusal.py``: the logistic and multinomial Wald tables,
the proportional-odds model's Wald and cluster-robust (CR1) tables, the Cox model's Wald and
Lin–Wei tables, and GEE's (its early return past the CR2 table, ``repeated.gee_table``).

Where it fires, the covariance is constructed directly, so the case is exact on every platform
(no rounding, and no NaN handed to LAPACK, whose handling of one differs between builds): one
coefficient's entries in the logistic, multinomial and Cox covariances set to zero (the first two
where the tables take them, the inverse information from the working design's QR,
``inference.gram_inverse``, which they read in place of statsmodels' since T1V2-2), the
proportional-odds model's information set to zero where the table takes it (``ordinal.on_qr``;
its pseudo-inverse, the table's bread, is then exactly zero), and GEE's working residuals set to
zero (so its sandwich's meat is). A zero
row and column of a bread leave that coefficient's sandwich variance exactly zero too. Each table
is refused in words that name what has no standard error and what the fit could not compute, and
keeps every estimate, as the unaltered fit's table reports it, with no interval. Where it stays
silent, an ordinary fit on the same rows is served with an interval that holds each estimate.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import inference, ordinal, repeated, survival
from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table

G, M = 40, 5  # 40 units: above the unit floor and the few-units count
N = G * M
CODES = np.repeat(np.arange(G), M)
CLUSTERS = Clusters(column="pid", codes=CODES, n_clusters=G, n_missing=0)
END = "no finite, positive standard error, so no interval or p-value is reported."


def _rows() -> tuple[pd.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(21)
    return pd.DataFrame({"x": rng.normal(size=N), "z": rng.normal(size=N)}), rng


def _outcome(kind: str) -> np.ndarray:
    matrix, rng = _rows()
    x = matrix["x"].to_numpy()
    if kind == "binary":
        return (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-x))).astype(int)
    if kind in ("multiclass", "ordinal"):
        return np.digitize(x + rng.normal(size=N), [-0.5, 0.5])
    if kind == "survival":
        time = rng.exponential(size=N) * np.exp(-0.5 * x)
        return survival.survival_outcome(rng.uniform(size=N) < 0.7, time)
    return 1.0 + x + rng.normal(size=G)[CODES] + rng.normal(size=N)


def _without_variance(cov: np.ndarray, j: int) -> np.ndarray:
    """A covariance with coefficient ``j``'s row and column set to zero."""
    cov = np.array(cov, dtype=float)
    cov[j, :] = 0.0
    cov[:, j] = 0.0
    return cov


def _information(monkeypatch: pytest.MonkeyPatch, j: int) -> None:
    """The inverse information the logistic and multinomial tables take
    (``inference.gram_inverse``), with coefficient ``j`` (class by class) left no variance."""
    inverse = inference.gram_inverse

    def altered(*args, **kwargs):
        return _without_variance(inverse(*args, **kwargs), j)

    monkeypatch.setattr(inference, "gram_inverse", altered)


def _ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    on_qr = ordinal.on_qr

    def altered(*args, **kwargs):
        information, scores, carry = on_qr(*args, **kwargs)
        return np.zeros_like(information), scores, carry

    monkeypatch.setattr(ordinal, "on_qr", altered)


def _cox(monkeypatch: pytest.MonkeyPatch) -> None:
    fit = survival.cox_fit

    def altered(*args, **kwargs):
        result = fit(*args, **kwargs)
        return dataclasses.replace(result, cov=_without_variance(result.cov, 1))

    monkeypatch.setattr(survival, "cox_fit", altered)


def _gee(monkeypatch: pytest.MonkeyPatch) -> None:
    working = repeated.gee_working

    def altered(*args, **kwargs):
        W, r = working(*args, **kwargs)
        return W, np.zeros_like(r)

    monkeypatch.setattr(repeated, "gee_working", altered)


CUTS = "`(cut-point 0 | 1)` and `(cut-point 1 | 2)`"
# name: (outcome, the table, how its covariance is left without a variance, the words)
CASES = {
    "logistic": ("binary", lambda m, y: inference_table("binary", m, y, [0, 1]),
                 lambda mp: _information(mp, 2),
                 f"The logistic model's information gives `z` {END}"),
    "multinomial": ("multiclass", lambda m, y: inference_table("multiclass", m, y, [0, 1, 2]),
                    lambda mp: _information(mp, 2),
                    f"The multinomial model's information gives `z [1]` {END}"),
    "proportional odds": ("ordinal", lambda m, y: ordinal.ordinal_table(m, y, [0, 1, 2],
                                                                        INDEPENDENT),
                          _ordinal,
                          f"The proportional-odds model's information gives `x`, `z`, {CUTS} "
                          f"{END}"),
    "proportional odds, CR1": ("ordinal", lambda m, y: ordinal.ordinal_table(m, y, [0, 1, 2],
                                                                             CLUSTERS),
                               _ordinal,
                               f"The cluster-robust sandwich by `pid` gives `x`, `z`, {CUTS} "
                               f"{END}"),
    "Cox": ("survival", lambda m, y: survival.cox_table(m, y, INDEPENDENT), _cox,
            f"The Cox partial likelihood's information gives `z` {END}"),
    "Cox, Lin–Wei": ("survival", lambda m, y: survival.cox_table(m, y, CLUSTERS), _cox,
                     f"The Lin–Wei cluster-robust sandwich by `pid` gives `z` {END}"),
    "GEE": ("quantity", lambda m, y: repeated.gee_table(m, y, CLUSTERS, "regression"), _gee,
            f"The cluster-robust (CR2) covariance by `pid` gives the intercept, `x` and `z` {END}"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_fit_left_with_no_standard_error_is_refused_in_words(case, monkeypatch):
    """The guard fires: the table is refused in words that name each coefficient with no standard
    error and the covariance that gives it none, keeps every estimate the unaltered fit reports,
    and gives no interval or p-value. With the guard removed it was served, the coefficient's
    estimate beside no interval."""
    kind, table, alter, words = CASES[case]
    matrix, _ = _rows()
    y = _outcome(kind)
    served = table(matrix, y)
    alter(monkeypatch)
    refused = table(matrix, y)
    assert refused.info["refused"] == words
    assert refused.info["covariance"] == "none"
    assert refused.info["caption"].endswith(f"No intervals: {words}")  # after the scale, if any
    assert [r["feature"] for r in refused.rows] == [r["feature"] for r in served.rows]
    for kept, row in zip(served.rows, refused.rows):
        assert row["estimate"] == kept["estimate"]
        assert row["ci_low"] is None and row["ci_high"] is None and row["p"] is None


@pytest.mark.parametrize("case", list(CASES))
def test_an_ordinary_fit_is_served_with_its_intervals(case):
    """The guard stays silent: on the same rows, unaltered, every coefficient has a standard error
    and an interval that holds its estimate."""
    kind, table, _, _ = CASES[case]
    matrix, _ = _rows()
    result = table(matrix, _outcome(kind))
    assert result.info["refused"] is None
    assert result.rows
    for row in result.rows:
        assert row["se"] > 0 and row["ci_low"] < row["estimate"] < row["ci_high"]
