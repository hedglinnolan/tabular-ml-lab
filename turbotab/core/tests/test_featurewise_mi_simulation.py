"""The pooled feature-wise table, simulated by construction (the plasma scope's item 1, truth (3);
``methods.featurewise_mi``).

Each replicate is one study factor on the plasma panel's design, 58 events and 42 others (n = 100),
one covariate: x = 10 + δ·event + 0.3·z + N(0, 1), blank below its 30% or 60% quantile (left-censored
there). Many study factors in one table are independent replicates on that design, so a table of
1,000 of them estimates coverage and the null rejection rate with a Monte Carlo error of
√(0.95 · 0.05 / 1,000) = 0.0069. The truth is δ, known by construction.

* Coverage of the pooled 95% intervals and the rejection rate at 0.05 under the null lie within
  three Monte Carlo errors of 0.95 and 0.05 at 30% and at 60% censored. Today's way forward, one
  censoring-aware fill per feature (``missing.BelowDetectionFill``, intercept-only), covers δ = 0.5
  at 60% censored in fewer than 90% of its intervals (the scope's ``fill_sim.py``: 0.865).
* The false-discovery proportion after Benjamini–Hochberg on the pooled p-values, over tables of
  200 study factors (20 with δ = 0.8, 180 null, 40% censored), averages at most q = 0.05 within
  three of its Monte Carlo errors (Benjamini & Hochberg's bound under independence is
  q · 180/200 = 0.045). The scope did not check this.
* On rows that repeat by participant beside an event that is one value per participant (the
  verifier's clustered panel), every table is drawn, its estimate unbiased and its intervals at
  their nominal coverage within three Monte Carlo errors (the second repair round).

Marked slow (under a minute since the censored-normal fit is Newton's, about four before; about
five more for the clustered panel): run it by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.methods.imputation import Censoring

N1, N0 = 58, 42


def panel(seed: int, R: int, delta: float | np.ndarray, share: float
          ) -> tuple[pd.DataFrame, np.ndarray]:
    """``R`` study factors on one design (the event and one covariate `z0`), each left-censored
    below its own ``share`` quantile."""
    rng = np.random.default_rng(seed)
    n = N1 + N0
    event = np.r_[np.ones(N1), np.zeros(N0)]
    z = rng.standard_normal(n)
    deltas = np.broadcast_to(np.asarray(delta, dtype=float), (R,))
    cols = {}
    for j in range(R):
        x = 10 + deltas[j] * event + 0.3 * z + rng.standard_normal(n)
        limit = np.quantile(x, share)
        cols[f"f{j:04d}"] = np.where(x < limit, np.nan, x)
    frame = pd.DataFrame(cols)
    frame["z0"] = z
    return frame, event


def pooled(frame: pd.DataFrame, event: np.ndarray, seed: int) -> list[dict]:
    from turbotab.core.methods import featurewise_mi as fm

    features = [c for c in frame.columns if c.startswith("f")]
    censored = {f: Censoring(float(np.nanmin(frame[f].to_numpy(float))), False) for f in features}
    table, _ = fm.pooled_featurewise_table(frame, event, "binary", features, censored=censored,
                                           seed=seed)
    return table.rows


def single_fill_coverage(frame: pd.DataFrame, event: np.ndarray, delta: float) -> float:
    """Today's way forward: each feature filled once (``BelowDetectionFill``, censoring-aware),
    then least squares of it on the event and `z0`, the classical 95% interval."""
    from turbotab.core.methods.missing import BelowDetectionFill

    X = np.column_stack([np.ones(len(event)), event, frame["z0"].to_numpy(float)])
    df = len(event) - X.shape[1]
    q = stats.t.ppf(0.975, df)
    inv = np.linalg.inv(X.T @ X)[1, 1]
    covered = []
    for f in [c for c in frame.columns if c.startswith("f")]:
        filled = BelowDetectionFill(columns=[f], method="censoring_aware").fit(frame[[f]])
        x = filled.transform(frame[[f]])[f].to_numpy(float)
        beta, rss = np.linalg.lstsq(X, x, rcond=None)[:2]
        se = float(np.sqrt(rss[0] / df * inv))
        covered.append(beta[1] - q * se <= delta <= beta[1] + q * se)
    return float(np.mean(covered))


MC = float(np.sqrt(0.95 * 0.05 / 1000))  # the Monte Carlo error of a rate near 0.95 or 0.05


@pytest.mark.slow
@pytest.mark.parametrize("share", [0.3, 0.6])
def test_coverage_and_null_rejection_are_nominal_at_30_and_60_percent_censored(monkeypatch,
                                                                              share):
    from turbotab.core.methods import featurewise_mi as fm

    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 10 ** 6)  # one replicate per study factor
    frame, event = panel(101 + int(share * 10), 1000, 0.5, share)
    rows = pooled(frame, event, seed=1)
    low = np.array([r["ci_low"] for r in rows])
    high = np.array([r["ci_high"] for r in rows])
    coverage = float(np.mean((low <= 0.5) & (0.5 <= high)))
    assert abs(coverage - 0.95) <= 3 * MC, coverage
    frame0, event0 = panel(201 + int(share * 10), 1000, 0.0, share)
    p = np.array([r["p"] for r in pooled(frame0, event0, seed=2)])
    rejection = float(np.mean(p < 0.05))
    assert abs(rejection - 0.05) <= 3 * MC, rejection
    if share == 0.6:
        # Today's way forward, on the same replicates: its intervals miss δ far more often.
        assert single_fill_coverage(frame, event, 0.5) < 0.90


@pytest.mark.slow
def test_the_false_discovery_proportion_after_benjamini_hochberg_stays_at_q():
    deltas = np.r_[np.full(20, 0.8), np.zeros(180)]
    fdp = []
    for rep in range(40):
        frame, event = panel(5000 + rep, 200, deltas, 0.4)
        q = np.array([r["q"] for r in pooled(frame, event, seed=rep)])
        found = q < 0.05
        fdp.append(found[20:].sum() / max(int(found.sum()), 1))
    mean, error = float(np.mean(fdp)), float(np.std(fdp, ddof=1) / np.sqrt(len(fdp)))
    assert mean <= 0.05 + 3 * error, (mean, error)


@pytest.mark.slow
def test_a_censored_factor_on_rows_repeating_beside_a_unit_level_outcome_is_drawn_and_covers():
    """The verifier's clustered panel (FWMI-R1-1's, its ``cluster_sim.py``): 40 participants of 3
    rows, the event one value per participant, `z0` row by row, a participant effect (SD 0.7);
    x = 8 + δ·event + 0.3·z0 + effect + N(0, 1), δ = 0.5, blank below its 30% quantile; 400
    independent replicates, each its own table, clustered by participant (CR2). The chained
    model's participant mean of the event repeats it; the censored-normal fit drops it, as
    survreg drops an aliased column. Every replicate is drawn (the base refused about 65% in
    words, by rounding), the mean estimate lies within three Monte Carlo errors of δ, and the
    coverage of the 95% intervals within three of 0.95 (√(0.95 · 0.05 / 400) = 0.011). Over 1,000
    replicates the mean was 0.501 (Monte Carlo error 0.010) and the coverage 0.960."""
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.models.linear import as_clusters

    G, R, delta = 40, 3, 0.5
    codes = np.repeat(np.arange(G), R)
    estimates, covered = [], []
    for rep in range(400):
        rng = np.random.default_rng(31000 + rep)
        event = np.repeat((rng.random(G) < 0.5).astype(float), R)
        z = rng.standard_normal(G * R)
        effect = np.repeat(rng.standard_normal(G) * 0.7, R)
        x = 8 + delta * event + 0.3 * z + effect + rng.standard_normal(G * R)
        x = np.where(x < np.quantile(x, 0.3), np.nan, x)
        frame = pd.DataFrame({"c0": x, "z0": z})
        table, _ = fm.pooled_featurewise_table(
            frame, event, "binary", ["c0"], as_clusters(codes),
            censored={"c0": Censoring(float(np.nanmin(x)), False)}, seed=rep)
        (row,) = table.rows
        estimates.append(row["estimate"])
        covered.append(row["ci_low"] <= delta <= row["ci_high"])
    error = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(float(np.mean(estimates)) - delta) <= 3 * error, (np.mean(estimates), error)
    assert abs(float(np.mean(covered)) - 0.95) <= 3 * np.sqrt(0.95 * 0.05 / 400), np.mean(covered)
