"""The standard candidate below an effective size of 300 (RECIPES_AND_TUNING §4.6, ruled
2026-10-11; notes/SMALL_SAMPLE_STANDARD.md; claims backtest A-001).

Below an effective size of 300 the plan draws no search and boosted trees and XGBoost are fit once,
at their standard candidate. §4.6's reason is that settings ranked on so few units vary from sample
to sample and miscalibrate, so one fit at the standard should predict about as well as nested
tuning and be calibrated. At the libraries' defaults it is neither: the study behind the ruling
found them the worst arm in every cell, by 0.10 to 0.39 nats, with calibration slopes of 0.2 to
0.5.

**Known truth.** T1's *signal* generator (ten N(0, 1) predictors; logit −0.85 + 0.8x₁ − 0.6x₂ +
0.5x₁x₃), n = 200 rows, about 66 events, so an effective size near 53. Each model is scored on
20,000 fresh rows against the true probabilities, the outcome integrated out:

* **the calibration slope:** the logistic recalibration's slope, fit by statsmodels' binomial GLM
  of the true probability on the model's logit;
* **excess log loss:** the expected log loss minus the Bayes risk, in nats.

The comparator is nested tuning by the app's own search, forced on as T1 forces it (the plan made
as at an effective size of 300, so the Sobol candidates are drawn; the K floor reads the real event
count): what the claim says the standard does at least as well as.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families

P, N, FRESH, DATASETS = 10, 200, 20_000, 6
COLUMNS = [f"x{j}" for j in range(P)]


def _truth(X: np.ndarray) -> np.ndarray:
    lp = -0.85 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    return 1.0 / (1.0 + np.exp(-lp))


def _slope(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    import statsmodels.api as sm

    logit = np.log(p_hat / (1.0 - p_hat))
    fit = sm.GLM(p_true, sm.add_constant(logit), family=sm.families.Binomial()).fit()
    return float(fit.params[1])


def _excess(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    loss = -(p_true * np.log(p_hat) + (1.0 - p_true) * np.log(1.0 - p_hat))
    bayes = -(p_true * np.log(p_true) + (1.0 - p_true) * np.log(1.0 - p_true))
    return float(np.mean(loss - bayes))


def _scored(key: str, nested: bool) -> tuple[list[float], list[float], list[int]]:
    """(slopes, excess log losses, candidate counts) over the datasets, each fit as the design
    makes it (``make_plans`` on the training rows), or with the search forced on."""
    from turbotab.core.models.base import get_family
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.pipeline import make_plans
    from turbotab.core.models.tuning import TunedPipeline, make_plan

    family = get_family(key)
    fresh = np.random.default_rng(99).standard_normal((FRESH, P))
    p_true = _truth(fresh)
    slopes, excess, sizes = [], [], []
    for seed in range(DATASETS):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, P))
        y = (rng.random(N) < _truth(X)).astype(int)
        plan = make_plans([family], "binary", y, folds=5, split_seed=seed)[key]
        assert plan.n_plan < 300
        if nested:
            plan = make_plan(family, task="binary", loss="log_loss", n_plan=300,
                             plan_rows=plan.plan_rows, unit=plan.unit, split_seed=seed,
                             rarest=plan.n_plan)
        pipe = TunedPipeline([("model", family.build("binary", "prediction", N, P))], search=plan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fit_pipeline(pipe, pd.DataFrame(X, columns=COLUMNS), y, seed=plan.split_seed)
        p_hat = fitted.predict_proba(pd.DataFrame(fresh, columns=COLUMNS))[:, 1]
        p_hat = np.clip(p_hat, 1e-12, 1.0 - 1e-12)
        slopes.append(_slope(p_hat, p_true))
        excess.append(_excess(p_hat, p_true))
        sizes.append(len(plan.candidates))
    return slopes, excess, sizes


@pytest.mark.parametrize("key", ["boosted_trees", "xgboost"])
def test_below_300_the_standard_fit_is_calibrated_and_predicts_as_well_as_nested_tuning(key):
    """Below 300 the plan makes one fit at the standard candidate. Its mean calibration slope over
    six datasets lies within [0.7, 1.3] (direct library fits at the ruled values gave 0.82 and
    0.93 here; the libraries' defaults gave 0.28 and 0.21), and its mean excess log loss is at most nested
    tuning's plus 0.01 nats (the study: the standard 0.010 and 0.014 nats better at n = 200; the
    defaults 0.15 and 0.29 worse)."""
    slopes, excess, sizes = _scored(key, nested=False)
    assert set(sizes) == {1}
    _, nested_excess, nested_sizes = _scored(key, nested=True)
    assert set(nested_sizes) == {9}
    assert 0.7 <= float(np.mean(slopes)) <= 1.3, slopes
    assert float(np.mean(excess)) <= float(np.mean(nested_excess)) + 0.01, (excess, nested_excess)
