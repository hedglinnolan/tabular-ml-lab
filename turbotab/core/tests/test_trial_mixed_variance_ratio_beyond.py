"""A cluster-randomized trial analyzed by the mixed model is refused, in the mixed table's words,
where REML's ratio of the between- to the within-cluster variance lies past the fit's search
(``RandomInterceptFit.beyond``), as the mixed model's coefficient table is.

On the base the trial path ignored the flag and served Kenward–Roger intervals from the GLS fit at
the search's top, whose variances are not REML's: the defect package A fixed in the coefficient
table, left in ``trials._mixed``.

Truth from construction: an outcome that is the cluster's level plus the arm and a covariate,
with noise of SD 10⁻⁶ within clusters beside clusters of SD 0.8, puts REML's ratio near 10¹², past
the grid's 10⁸; with noise of SD 2 it is near 0.16 and the trial is analyzed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import trials as T
from turbotab.core.models.repeated import GAMMA_GRID

SPEC = T.TrialSpec(arm="arm", control="ctl", outcome="y", trial_design="cluster_randomized_trial",
                   cluster="cluster", covariates=("x",))


def _trial(noise: float, seed: int = 7, K: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.integers(4, 40, K)
    df = pd.DataFrame({"cluster": np.repeat(np.arange(1, K + 1), m)})
    df["arm"] = df.cluster.map({c: ("ctl" if c % 2 else "trt") for c in range(1, K + 1)})
    u = rng.normal(0, 0.8, K + 1)
    df["x"] = rng.normal(0, 1, len(df))
    df["y"] = (u[df.cluster] + 0.6 * (df.arm == "trt") + 0.8 * df.x
               + rng.normal(0, noise, len(df)))
    return df


def test_a_variance_ratio_past_the_search_is_refused_in_the_tables_words():
    """Noise of SD 10⁻⁶ within clusters: refused, in the words the mixed table uses, with the
    estimating equations and a description of the outcome as the ways forward."""
    df = _trial(1e-6)
    with pytest.raises(T.TrialRefused) as refused:
        T.estimate(df, SPEC, cluster_method="mixed")
    top = GAMMA_GRID[-1]
    assert str(refused.value) == (
        f"The random-intercept model's REML likelihood by `cluster` on these 14 units is still "
        f"rising where the between-unit variance is {top:,.0f} times the within-unit variance, "
        f"the most the fit searches: within a `cluster` unit the outcome varies so little beyond "
        f"what the predictors explain that REML puts the within-unit SD below "
        f"{1 / np.sqrt(top):.3g} times the between-unit SD. The fit does not estimate variances "
        f"that far apart, so no interval or p-value is reported.")
    assert [e.get("cluster_method") for e in refused.value.exits][:1] == ["gee"]


def test_an_ordinary_variance_ratio_is_analyzed():
    """The silent side: noise of SD 2 within clusters, the trial is analyzed by the mixed model,
    its contrast with an interval."""
    result = T.estimate(_trial(2.0), SPEC, cluster_method="mixed")
    assert result.estimator.startswith("linear mixed model")
    (contrast,) = result.contrasts
    assert contrast.lower < contrast.estimate < contrast.upper
