"""A Cox fit on nearly collinear columns finds the maximum R's ``coxph`` finds, where its
coefficients are some 40 SDs and opposite in sign: the divergence bound is on the linear
predictor's SD (``survival._spread``), not on each coefficient, and a monotone likelihood is still
refused.

On the base ``cox_fit`` stopped after its first Newton step, a coefficient past 20 SDs, reported
not converged, and the Cox table was refused as having "no stable maximum" for `x1` and `x2`
(as was the SMC-FCS Cox draw on package B's near-alias construction).

Truth from construction and from R: x2 = x1 + 0.02·d with d standard normal and the log hazard
0.5·x1 + d, so on (x1, x2) the coefficients are about (0.5 − 50, 50) while the linear predictor's
SD is about 1.1; R 4.6.1's ``survival::coxph(Surv(time, event) ~ x1 + x2, ties = "efron")`` on
these rows (written by this module's ``_data``, read back in full precision) converges in 4
iterations to the numbers below, and statsmodels' PHReg agrees to 10⁻¹⁰. A group with no events
orders every risk set, and its likelihood has no maximum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbotab.core.models.inference import INDEPENDENT
from turbotab.core.models.survival import cox_fit, cox_table, survival_outcome

# R 4.6.1, survival 3.8.6: coef(fit) and sqrt(diag(vcov(fit))), eps = 1e-12.
R_COEF = (-42.7401463228503, 43.2440181980344)
R_SE = (3.69791052035482, 3.71493280341787)


def _data(seed: int = 5, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    d = rng.normal(size=n)
    x2 = x1 + 0.02 * d
    eta = 0.5 * x1 + d
    t = rng.exponential(np.exp(-eta))
    c = rng.exponential(2.0, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "time": np.minimum(t, c),
                         "event": (t <= c).astype(int)})


def test_the_fit_finds_r_s_maximum_on_nearly_collinear_columns():
    """Converged, at R's coefficients and standard errors; the table is served with them."""
    f = _data()
    y = survival_outcome(f.event.to_numpy() == 1, f.time.to_numpy())
    fit = cox_fit(f[["x1", "x2"]].to_numpy(), y)
    assert fit.converged
    np.testing.assert_allclose(fit.beta, R_COEF, rtol=1e-8)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), R_SE, rtol=1e-7)
    table = cox_table(f[["x1", "x2"]], y, INDEPENDENT)
    assert table.info["refused"] is None
    np.testing.assert_allclose([r["estimate"] for r in table.rows], R_COEF, rtol=1e-8)
    np.testing.assert_allclose([r["se"] for r in table.rows], R_SE, rtol=1e-7)


def test_a_group_with_no_events_beside_them_is_still_refused():
    """The bound's other side: a group with no events orders every risk set, so the likelihood
    has no maximum along it, and the table is refused naming the group alone."""
    f = _data()
    group = (np.arange(len(f)) % 8 == 0).astype(float)
    event = (f.event.to_numpy() == 1) & (group == 0)
    y = survival_outcome(event, f.time.to_numpy())
    matrix = f[["x1", "x2"]].assign(group=group)
    fit = cox_fit(matrix.to_numpy(), y)
    assert not fit.converged
    table = cox_table(matrix, y, INDEPENDENT)
    assert table.info["refused"] == (
        "The partial likelihood has no maximum for `group`: at every event time the rows with "
        "the event have the lowest `group` of the rows at risk (as when a group has no events), "
        "so its hazard ratio runs to zero, and no interval or p-value is reported.")
    assert all(r["se"] is None for r in table.rows)
