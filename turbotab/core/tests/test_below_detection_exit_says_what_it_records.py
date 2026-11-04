"""The censoring-aware exit of ``median_below_detection`` says what its record does
(``decisions._non_detections_are_not_filled_by_the_median``). Candidate 7 of the PP2 scope: under
inference, with a single fill kept on the attestation, the exit read "a censored-normal draw below
the limit, given the outcome" whatever the strategy, and recorded strategy="impute" with
below_detection="censoring_aware": a fill once, outcome-free, from an intercept-only censored
normal (``missing.BelowDetectionFill``), with no draw and no outcome.

The truth is the code path the record takes, by construction: the exit keeps the answer's
strategy; the design reads that record's below-detection answer into the detection step
(``models.pipeline.design_spec`` then ``detect_step``); and that step's fill on a column with a
known limit L is E[X | X < L] under a censored normal fitted to the column, against scipy (a
maximum-likelihood fit of the left-censored normal by ``scipy.optimize``, then the truncated
normal's mean, ``scipy.stats.truncnorm``), one value for every blank whatever the outcome. The
guard's other direction: under multiple imputation the draw's words stay, and under prediction a
fill's words stay as they were.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from turbotab.core.decisions import ProjectState, Refusal, SetMissing, validate

LIMIT = -0.5  # negative, so the censored normal is fitted on the column's own scale


def ctx(purpose: str) -> dict:
    state = ProjectState(target="y", purpose=purpose, roles={"m": "exposure"})
    return {"state": state, "columns": ["m", "y"]}


def first_exit(purpose: str, strategy: str, **answer) -> dict:
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy=strategy, censored_columns=["m"], **answer), ctx(purpose))
    assert refused.value.code == "median_below_detection"
    return refused.value.exits[0]


def censored_column() -> pd.DataFrame:
    """400 draws of N(0, 1), every value below LIMIT a blank, and the smallest detected value set
    to LIMIT, so the limit the fill reads (the smallest detected value) is LIMIT."""
    x = np.random.default_rng(11).normal(0.0, 1.0, 400)
    x[x < LIMIT] = np.nan
    x[np.nanargmin(x)] = LIMIT
    return pd.DataFrame({"m": x})


def scipy_fill(x: np.ndarray) -> float:
    """E[X | X < LIMIT] under the left-censored normal's maximum-likelihood fit."""
    detected, blanks = x[np.isfinite(x)], int(np.isnan(x).sum())

    def minus_loglik(theta: np.ndarray) -> float:
        mu, sigma = theta[0], np.exp(theta[1])
        return -(stats.norm.logpdf(detected, mu, sigma).sum()
                 + blanks * stats.norm.logcdf((LIMIT - mu) / sigma))

    fit = optimize.minimize(minus_loglik, np.array([0.5, 0.0]), method="BFGS",
                            options={"gtol": 1e-10})
    mu, sigma = fit.x[0], float(np.exp(fit.x[1]))
    return float(stats.truncnorm(-np.inf, (LIMIT - mu) / sigma, loc=mu, scale=sigma).mean())


def test_under_inference_a_single_fill_says_the_expected_value_below_the_limit():
    way = first_exit("inference", "impute", acknowledged=True)
    assert way["label"] == "Censoring-aware: the expected value below the limit under a " \
                           "censored-normal fit"
    record = way["decision"]
    assert (record["strategy"], record["below_detection"], record["censored_columns"]) == \
        ("impute", "censoring_aware", ["m"])
    validate(SetMissing(**{k: v for k, v in record.items() if k != "kind"}), ctx("inference"))
    # The design's detection step for that record (``design_spec``: the answer's below-detection
    # method over its censored columns; no logged normalization here).
    from turbotab.core.methods.missing import BelowDetectionFill
    from turbotab.core.models.pipeline import detect_step

    step = detect_step(SimpleNamespace(normalization=None, censored={
        "method": record["below_detection"], "columns": record["censored_columns"]}))
    assert isinstance(step, BelowDetectionFill) and step.method == "censoring_aware"
    frame = censored_column()
    blank = frame["m"].isna().to_numpy()
    assert 0.2 < blank.mean() < 0.4
    outcome = np.random.default_rng(12).normal(size=len(frame))
    filled = [step.fit(frame, y).transform(frame)["m"].to_numpy()[blank]
              for y in (None, outcome, -outcome)]
    for values in filled:  # one value for every blank, the same whatever the outcome: no draw
        assert np.unique(values).size == 1
        assert values[0] == pytest.approx(scipy_fill(frame["m"].to_numpy()), rel=1e-6)
        assert values[0] < LIMIT
    assert filled[0][0] == filled[1][0] == filled[2][0]


def test_under_inference_multiple_imputation_keeps_the_draws_words():
    way = first_exit("inference", "multiple_imputation")
    assert way["label"] == "Censoring-aware: a censored-normal draw below the limit, given the " \
                           "outcome"
    assert (way["decision"]["strategy"], way["decision"]["below_detection"]) == \
        ("multiple_imputation", "censoring_aware")


def test_under_prediction_a_fill_keeps_its_words():
    way = first_exit("prediction", "impute")
    assert way["label"] == "Censoring-aware: the expected value below the limit, fit in each " \
                           "training fold"
    assert (way["decision"]["strategy"], way["decision"]["below_detection"]) == \
        ("impute", "censoring_aware")
