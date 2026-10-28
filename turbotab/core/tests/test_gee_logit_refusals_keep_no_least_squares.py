"""A GEE logit table refused before its fit (no unit repeats, a nearly singular matrix, a column
that separates the outcome) reports no estimate; the identity-link GEE refused so keeps its
least-squares estimates, which are its own (T1V2-3, 2026-10-27).

On the base each such refusal kept the least-squares start of the fit, ``lstsq(X, y)`` on the 0/1
outcome, and the table's caption, on the logit scale, called those numbers log-odds with odds
ratios exp(estimate): with no unit repeating, `x`'s 0.2144 (odds ratio 1.239) where the logistic
fit on the same rows gives 1.1196 (3.06).

Truth: statsmodels' OLS on the rows, for the identity link; for the logit link, every estimate is
None, by construction of the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import INDEPENDENT, Clusters, _on_scale
from turbotab.core.models.repeated import gee_table

N, G = 500, 100


def _data():
    rng = np.random.default_rng(1)
    x, v, d = rng.normal(size=(3, N))
    eta = -0.3 + 1.0 * x + 0.5 * v
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    yr = eta + rng.normal(size=N)
    return x, v, d, yb, yr


def _clusters() -> Clusters:
    return Clusters(column="pid", codes=np.repeat(np.arange(G), N // G).astype(np.int64),
                    n_clusters=G, n_missing=0)


def _refusals():
    x, v, d, yb, _ = _data()
    f = (np.arange(N) % 9 == 0).astype(float)  # every row where f = 1 has the event
    separated = np.where(f == 1, 1, yb)
    return {
        "no unit repeats": (pd.DataFrame({"x": x, "v": v}), yb, INDEPENDENT,
                            "No identifier repeats"),
        "nearly singular": (pd.DataFrame({"x": x, "x2": x + 1e-4 * d, "v": v}), yb, _clusters(),
                            "The model matrix is nearly singular"),
        "separation": (pd.DataFrame({"x": x, "v": v, "f": f}), separated, _clusters(),
                       "`f` separates the outcome"),
    }


@pytest.mark.parametrize("case", list(_refusals()))
def test_a_gee_logit_table_refused_before_its_fit_reports_no_estimate(case):
    """Each refusal before the fit reports no estimate, and so no odds ratio, on the logit scale."""
    matrix, y, clusters, words = _refusals()[case]
    table = _on_scale(gee_table(matrix, y, clusters, "binary", [0, 1], study=["v"]), "binary",
                      [0, 1], None)
    assert table.info["refused"] is not None and table.info["refused"].startswith(words), (
        table.info["refused"])
    assert table.rows and all(r["estimate"] is None and r.get("ratio") is None
                              for r in table.rows)


def test_the_identity_gee_refused_with_no_unit_repeating_keeps_least_squares():
    """The silent side: the identity-link GEE with no unit repeating keeps its estimates, which are
    least squares' on these rows."""
    x, v, _, _, yr = _data()
    matrix = pd.DataFrame({"x": x, "v": v})
    table = gee_table(matrix, yr, INDEPENDENT, "regression", study=["v"])
    assert table.info["refused"].startswith("No identifier repeats")
    ols = sm.OLS(yr, sm.add_constant(matrix)).fit()
    np.testing.assert_allclose([r["estimate"] for r in table.rows], ols.params, rtol=1e-10)
