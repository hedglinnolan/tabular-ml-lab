"""The design-based curve's fit (``survey.design_fit``) is fit on its columns centered and scaled,
as the coefficient tables are, and mapped back (T1V3-4; the methods lead's ruling 3 of
2026-10-27): a date in epoch seconds spanning a day gives the slopes and their influence the same
date at its own origin gives.

On the base the fit took the columns as they stood: beside such a date the least-squares fit cut
the date as nearly a copy of the intercept (its slope 0, the others' moved by up to 2.4% on the
verifier's rows), and the logistic fit stopped at zero.

Truth: the shift identity (``t`` and ``t − M`` span the same columns with the intercept, so every
slope and each row's influence on it are unmoved and the intercept moves by ``β_t·M``), with
``t − M`` exact in floating point (Sterbenz); statsmodels' WLS and its GLM with frequency weights
on ``(1, x1, x2, t − M)``, and each row's influence ``(XᵀWX)⁻¹ x_i w_i e_i`` written out with numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.survey import SurveyDesign, design_fit

N, M = 300, 1.7e9


def _data():
    rng = np.random.default_rng(21)
    x1, x2 = rng.normal(size=(2, N))
    t = M + rng.uniform(0, 86400, N)
    own = t - M  # exact: t and M are within a factor of two of each other
    eta = 0.4 * x1 - 0.2 * x2 + 2e-5 * (own - 43200)
    yr = 1 + eta + rng.normal(size=N)
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    design = SurveyDesign(row_ids=np.arange(N), weight=rng.uniform(1, 3, N),
                          stratum=np.repeat(np.arange(10), N // 10).astype(np.int64),
                          psu=np.repeat(np.arange(30), N // 30).astype(np.int64),
                          weight_column="w", strata_column="s", psu_column="p")
    index = pd.RangeIndex(N)
    epoch = pd.DataFrame({"x1": x1, "x2": x2, "t": t}, index=index)
    return epoch, epoch.assign(t=own), yr, yb, design


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_a_date_in_epoch_seconds_gives_the_slopes_and_influence_of_the_date_at_its_origin(task):
    epoch, origin, yr, yb, design = _data()
    y, classes = (yr, None) if task == "regression" else (yb, [0, 1])
    a = design_fit(task, epoch, y, classes, design)
    b = design_fit(task, origin, y, classes, design)
    np.testing.assert_allclose(a.beta[1:], b.beta[1:], rtol=1e-9)
    assert a.beta[0] == pytest.approx(b.beta[0] - b.beta[3] * M, rel=1e-6)
    np.testing.assert_allclose(a.influence[:, 1:], b.influence[:, 1:], rtol=1e-7,
                               atol=1e-9 * np.abs(b.influence[:, 1:]).max())
    # The mean at the epoch date is X @ β in seconds, where β_t·t (about 3.5e4) and the intercept
    # cancel to leave about 1: a dot product's rounding is within γ_4·Σ_j|β_j x_ij| (Higham), and
    # the intercept's mapping back adds at most 2u·|β_t·m|, so 8u·Σ_j|β_j x_ij| = 4·eps·Σ bounds
    # it (6.2e-11 here). CI's Linux runner differed by 8.5e-12 on a mean of -0.004, past rtol
    # 1e-9 alone; the base, which cut the date, moved the slopes by up to 2.4%.
    terms = np.abs(np.column_stack([np.ones(N), epoch.to_numpy()]) * a.beta).sum(axis=1)
    np.testing.assert_allclose(a.mean(epoch)[0], b.mean(origin)[0], rtol=1e-9,
                               atol=4 * np.finfo(float).eps * terms.max())
    # and the fit at the origin is statsmodels' weighted fit, its influence the bread times the
    # row's weighted score
    X = sm.add_constant(origin.to_numpy())
    w = design.weight
    if task == "regression":
        ref = sm.WLS(yr, X, weights=w).fit()
        mu, var = X @ ref.params, np.ones(N)
    else:
        ref = sm.GLM(yb, X, family=sm.families.Binomial(), freq_weights=w).fit(tol=1e-14)
        mu = 1 / (1 + np.exp(-(X @ ref.params)))
        var = mu * (1 - mu)
    np.testing.assert_allclose(b.beta, ref.params, rtol=1e-8)
    bread = np.linalg.inv(X.T @ (X * (w * var)[:, None]))
    influence = (X * (w * (y - mu))[:, None]) @ bread
    np.testing.assert_allclose(b.influence[:, 1:], influence[:, 1:], rtol=1e-6,
                               atol=1e-9 * np.abs(influence[:, 1:]).max())
