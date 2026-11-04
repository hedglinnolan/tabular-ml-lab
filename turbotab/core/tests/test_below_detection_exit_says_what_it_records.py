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

The verifier's P1 (repair round 1): with fewer than 3 detected values that fill fits nothing and
sets every blank to half the smallest detected value (both fills: ``BelowDetectionFill`` on the
column's own scale, ``omics.LogScaleCensoredFill`` on the log scale the pipeline reads next), and
the metabolomics pack flags columns detected on one or two participants as left-censored. The label
says both, at the threshold the fills keep: at 3 detected values each fill is E[X | X < L] under its
censored normal (scipy, as above, on the log scale through ``scipy.stats.lognorm``), and at 2 it is
L / 2 (4.0 and 9.0 detected: 2.0).

The verifier's R1-P1 (repair round 2): two reachable edges of that fallback. On the own scale a
smallest detected value below zero (a table logged before upload) is the fill itself, not its half
(``limit / 2 if limit > 0 else limit``: -1.0 and 2.0 detected fill -1.0). And where a log2
normalization includes any of the censored columns, every one of them is filled on the log scale
(``pipeline.detect_step``: one ``LogScaleCensoredFill`` over them all), which sets a column whose
detected values are all equal to half that value with no fit (4.0 three times: 2.0); on the own
scale the same column is refused, as a censored normal on equal values has no finite maximum
(σ → 0 raises the likelihood without bound), so no half is claimed there. The label states both,
the log2 normalizations named by their own step labels (``omics.STEP_LABELS``; log-CPM is not
one).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from turbotab.core.decisions import ProjectState, Refusal, SetMissing, validate

LIMIT = -0.5  # negative, so the censored normal is fitted on the column's own scale


# The label a single fill's exit reads under inference: the fit, and its fallback with no fit, at
# the threshold the fills keep and where a log2 normalization sends equal values to it, a negative
# smallest value kept as it is (the tests below, both ways).
SINGLE_FILL = ("Censoring-aware: the expected value below the limit under a censored-normal fit, "
               "or, with no fit, half the smallest detected value (the value itself where it is "
               "negative) where a column has fewer than 3 detected values, or has every detected "
               "value equal and a log2 normalization includes any of these columns")


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


def censored_normal_fit(z: np.ndarray, limit: float) -> tuple[float, float]:
    """(mu, sigma) of the left-censored normal's maximum-likelihood fit to ``z``, its blanks
    censored at ``limit``."""
    detected, blanks = z[np.isfinite(z)], int(np.isnan(z).sum())

    def minus_loglik(theta: np.ndarray) -> float:
        mu, sigma = theta[0], np.exp(theta[1])
        return -(stats.norm.logpdf(detected, mu, sigma).sum()
                 + blanks * stats.norm.logcdf((limit - mu) / sigma))

    start = np.array([float(np.mean(detected)) - 1.0, float(np.log(np.std(detected) + 1.0))])
    fit = optimize.minimize(minus_loglik, start, method="BFGS", options={"gtol": 1e-10})
    return float(fit.x[0]), float(np.exp(fit.x[1]))


def scipy_fill(x: np.ndarray, limit: float = LIMIT) -> float:
    """E[X | X < limit] under the left-censored normal's maximum-likelihood fit."""
    mu, sigma = censored_normal_fit(x, limit)
    return float(stats.truncnorm(-np.inf, (limit - mu) / sigma, loc=mu, scale=sigma).mean())


def scipy_log_fill(x: np.ndarray, limit: float) -> float:
    """E[X | X < limit] under the left-censored normal's maximum-likelihood fit to log x: the
    log-normal's mean below the limit (``scipy.stats.lognorm``, by numerical integration)."""
    mu, sigma = censored_normal_fit(np.log(x), float(np.log(limit)))
    return float(stats.lognorm(s=sigma, scale=np.exp(mu)).expect(lb=0.0, ub=limit,
                                                                 conditional=True))


def test_under_inference_a_single_fill_says_the_expected_value_below_the_limit():
    way = first_exit("inference", "impute", acknowledged=True)
    assert way["label"] == SINGLE_FILL
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


def detected_on(values: list[float], blanks: int = 37) -> pd.DataFrame:
    """A column detected only at ``values`` (its smallest the limit), every other row a blank."""
    x = np.full(len(values) + blanks, np.nan)
    x[np.linspace(0, len(x) - 1, len(values)).astype(int)] = values
    return pd.DataFrame({"m": x})


def fills(frame: pd.DataFrame) -> dict[str, float]:
    """Each censoring-aware fill's one value for the blanks of ``frame``'s column: on its own scale
    (``BelowDetectionFill``) and on the log scale (``omics.LogScaleCensoredFill``)."""
    from turbotab.core.methods.missing import BelowDetectionFill
    from turbotab.core.methods.omics import LogScaleCensoredFill

    blank = frame["m"].isna().to_numpy()
    out = {}
    for name, step in (("own", BelowDetectionFill(["m"], "censoring_aware")),
                       ("log", LogScaleCensoredFill(["m"]))):
        values = step.fit(frame).transform(frame)["m"].to_numpy()[blank]
        assert np.unique(values).size == 1
        out[name] = float(values[0])
    return out


def test_at_three_detected_values_each_fill_is_the_expected_value_below_the_limit():
    """At the threshold the label names, each fill fits its censored normal: on the own scale (the
    limit below zero, so no log), and on the log scale; neither is half the limit."""
    own = detected_on([-0.5, 0.4, 1.1])
    assert fills(own)["own"] == pytest.approx(scipy_fill(own["m"].to_numpy(), -0.5), rel=1e-5)
    assert fills(own)["own"] < -0.5
    logged = detected_on([4.0, 6.0, 9.0])
    assert fills(logged)["log"] == pytest.approx(scipy_log_fill(logged["m"].to_numpy(), 4.0),
                                                 rel=1e-5)
    assert fills(logged)["log"] != pytest.approx(2.0, rel=1e-3)
    assert 0 < fills(logged)["log"] < 4.0


def test_below_three_detected_values_each_fill_is_half_the_smallest_detected_value():
    """Detected on two participants (4.0 and 9.0, the pack's case), each fill sets every blank to
    half the limit, 4.0 / 2, with no fit; and the exit's label says so."""
    assert fills(detected_on([4.0, 9.0])) == {"own": 2.0, "log": 2.0}
    assert fills(detected_on([4.0])) == {"own": 2.0, "log": 2.0}
    assert first_exit("inference", "impute", acknowledged=True)["label"] == SINGLE_FILL
    assert "half the smallest detected value (the value itself where it is negative) where a " \
           "column has fewer than 3 detected values" in SINGLE_FILL


def own_fill(frame: pd.DataFrame) -> float:
    """The own-scale censoring-aware fill's one value for the blanks of ``frame``'s column."""
    from turbotab.core.methods.missing import BelowDetectionFill

    blank = frame["m"].isna().to_numpy()
    values = BelowDetectionFill(["m"], "censoring_aware").fit(frame).transform(frame)["m"]
    values = values.to_numpy()[blank]
    assert np.unique(values).size == 1
    return float(values[0])


def test_below_three_detected_values_a_negative_smallest_value_is_the_fill_itself():
    """Detected at -1.0 and 2.0 (a table logged before upload; no log2 normalization takes a column
    at or below zero, so the fill is the own-scale one), every blank is set to the smallest
    detected value itself, -1.0 by construction, not its half; at 0.0 the value and its half agree,
    and a positive smallest value is halved (the test above). The label says so."""
    assert own_fill(detected_on([-1.0, 2.0])) == -1.0
    assert own_fill(detected_on([-1.0])) == -1.0
    assert own_fill(detected_on([0.0, 2.0])) == 0.0
    assert own_fill(detected_on([0.5, 2.0])) == 0.25
    assert first_exit("inference", "impute", acknowledged=True)["label"] == SINGLE_FILL
    assert "half the smallest detected value (the value itself where it is negative)" in SINGLE_FILL


def equal_and_varied() -> pd.DataFrame:
    """`m` detected three times at 4.0 and `k` at 4.0, 6.0 and 9.0, every other row a blank."""
    return pd.DataFrame({"m": detected_on([4.0, 4.0, 4.0])["m"],
                         "k": detected_on([4.0, 6.0, 9.0])["m"]})


@pytest.mark.parametrize("method", ["log2", "pqn_log2", "qc_pqn_log2"])
def test_where_a_log2_normalization_includes_a_column_equal_values_are_halved_with_no_fit(method):
    """A log2 normalization (its own step label says log2) that includes `k` alone: the detection
    step fills every censored column on the log scale, `m` too, so `m`'s three equal detected
    values (no fit has a finite maximum on them) set its blanks to 4.0 / 2 with no fit, while `k`,
    whose values differ, is fitted (E[X | X < 4] under the log-normal, scipy). The label says so."""
    from turbotab.core.methods.omics import STEP_LABELS, LogScaleCensoredFill
    from turbotab.core.models.pipeline import detect_step

    assert "log2" in STEP_LABELS[method][0].lower() and "log2" in STEP_LABELS[method][1]
    step = detect_step(SimpleNamespace(normalization={"method": method, "columns": ["k"]},
                                       censored={"method": "censoring_aware",
                                                 "columns": ["m", "k"]}))
    assert isinstance(step, LogScaleCensoredFill) and list(step.columns) == ["m", "k"]
    frame = equal_and_varied()
    filled = step.fit(frame).transform(frame)
    assert set(filled["m"][frame["m"].isna()]) == {2.0}
    k = filled["k"][frame["k"].isna()]
    assert np.unique(k).size == 1
    assert k.iloc[0] == pytest.approx(scipy_log_fill(frame["k"].to_numpy(), 4.0), rel=1e-5)
    assert first_exit("inference", "impute", acknowledged=True)["label"] == SINGLE_FILL
    assert SINGLE_FILL.endswith(", or has every detected value equal and a log2 normalization "
                                "includes any of these columns")


@pytest.mark.parametrize("method", [None, "log_cpm", "log_cpm_tmm"])
def test_without_a_log2_normalization_equal_values_are_refused_not_halved(method):
    """No normalization, or log-CPM (its step label says log-CPM, not log2): the fill is the
    own-scale one, which refuses `m`'s three equal detected values (a censored normal on them has
    no finite maximum: σ → 0 raises the likelihood without bound), so the label claims no half
    there."""
    from turbotab.core.graph import StageRefusal
    from turbotab.core.methods.missing import BelowDetectionFill
    from turbotab.core.methods.omics import STEP_LABELS
    from turbotab.core.models.pipeline import detect_step

    if method is not None:
        assert "log2" not in " ".join(STEP_LABELS[method]).lower()
    step = detect_step(SimpleNamespace(
        normalization=None if method is None else {"method": method, "columns": ["m", "k"]},
        censored={"method": "censoring_aware", "columns": ["m", "k"]}))
    assert isinstance(step, BelowDetectionFill) and step.method == "censoring_aware"
    with pytest.raises(StageRefusal, match="cannot be filled censoring-aware"):
        step.fit(equal_and_varied())
