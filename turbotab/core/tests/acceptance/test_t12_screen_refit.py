"""T12 · F4 is fixed (RECIPES_AND_TUNING §8; RT-5f). Slow, and a scheduled heavy run
(WAVE_C6A_PLAN §6: 30 to 90 minutes on the dev machine): it runs only when
``TURBOTAB_SCHEDULED_RUNS=1`` is set, so neither CI tier spends an hour on it.

The screened elastic net used to tune its penalty on columns screened with the outcomes of the
inner validation rows: the screen was fit on the whole outer training fold before the inner
cross-validation split it. On pure noise the screen then hands the inner folds columns that look
predictive on the rows that score the penalty, and the penalty comes out too small. The path search
now refits the screen in every inner split (the screen is a step before the model, and the search
refits those per split).

**Procedures**, on a p ≫ n null fixture (n = 120, p = 2,000, every column and the outcome
independent N(0, 1)), 50 datasets, the same plan and seeds for both:

* **N (ours):** the screened family's pipeline (screen, scale, elastic net) through the engine:
  the screen refit on every inner split's training rows; its model built at the table's width,
  as the design stage builds it (the wide build, which fits the screened columns it is handed).
* **F (F4, what not to do):** the screen fit once on the fit's rows, written out here, then the
  same path search on the screened columns alone (scale, elastic net).

**Assertions:**

* paired over the datasets, N's chosen penalty per row (α) is larger than F's by more than three
  standard errors of the mean paired log ratio;
* each procedure's outer five-fold score (the screen refit in each outer training fold for both, so
  both are honest) is within two standard errors of its fresh-data truth: the same procedure fit on
  all 120 rows and scored on 20,000 fresh rows, whose expected squared error is 1 plus the fit's
  spread around 0.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("TURBOTAB_SCHEDULED_RUNS") != "1",
                       reason="T12 is a scheduled heavy run (WAVE_C6A_PLAN §6): set "
                              "TURBOTAB_SCHEDULED_RUNS=1"),
]

N_ROWS, N_COLUMNS, N_DATASETS, FRESH, FOLDS = 120, 2_000, 50, 20_000, 5


def _plan(n: int, seed: int):
    from turbotab.core.models.base import get_family
    from turbotab.core.models.tuning import make_plan

    return make_plan(get_family("screened_elastic_net"), task="regression", loss="mse", n_plan=n,
                     plan_rows=n, unit="units", split_seed=seed)


def _ours(X: pd.DataFrame, y: np.ndarray, seed: int):
    """N: the screen inside every inner split, as the family's pipeline runs it."""
    from sklearn.preprocessing import StandardScaler

    from turbotab.core.methods.omics import UnivariateScreen
    from turbotab.core.models.base import get_family
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.tuning import TunedPipeline

    family = get_family("screened_elastic_net")
    pipe = TunedPipeline([("screen", UnivariateScreen(list(X.columns))),
                          ("scale", StandardScaler()),
                          # built at the table's width, as the design stage builds it
                          ("model", family.build("regression", "prediction", len(y),
                                                 X.shape[1]))],
                         search=_plan(len(y), seed)).set_output(transform="pandas")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_pipeline(pipe, X, y, seed=seed)


class _ScreenedOnce:
    """F: the screen fit once on the fit's rows, then the path search on its columns alone."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, seed: int):
        from sklearn.preprocessing import StandardScaler

        from turbotab.core.methods.omics import UnivariateScreen
        from turbotab.core.models.base import get_family
        from turbotab.core.models.inner_cv import fit_pipeline
        from turbotab.core.models.tuning import TunedPipeline

        self.screen = UnivariateScreen(list(X.columns)).fit(X, y)
        kept = self.screen.transform(X)
        family = get_family("screened_elastic_net")
        pipe = TunedPipeline([("scale", StandardScaler()),
                              ("model", family.build("regression", "prediction", len(y),
                                                     X.shape[1]))],
                             search=_plan(len(y), seed)).set_output(transform="pandas")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = fit_pipeline(pipe, kept, y, seed=seed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.fitted.predict(self.screen.transform(X))


def _outer_score(fit, X: pd.DataFrame, y: np.ndarray, seed: int) -> float:
    """Five-fold cross-validated squared error, each procedure refit on each training fold."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(len(y)) % FOLDS)
    errors = np.empty(len(y))
    for k in range(FOLDS):
        train, test = fold != k, fold == k
        model = fit(X.loc[train], y[train], seed)
        errors[test] = (y[test] - model.predict(X.loc[test])) ** 2
    return float(errors.mean())


def test_t12_the_screen_refit_in_every_inner_split_chooses_a_larger_penalty_on_noise():
    log_ratio, gaps = [], {"N": [], "F": []}
    columns = [f"g{j}" for j in range(N_COLUMNS)]
    for d in range(N_DATASETS):
        rng = np.random.default_rng(10_000 + d)
        X = pd.DataFrame(rng.normal(size=(N_ROWS, N_COLUMNS)), columns=columns)
        y = rng.normal(size=N_ROWS)
        fresh = pd.DataFrame(rng.normal(size=(FRESH, N_COLUMNS)), columns=columns)
        fresh_y = rng.normal(size=FRESH)
        ours, once = _ours(X, y, d), _ScreenedOnce(X, y, d)
        log_ratio.append(np.log(ours[-1].alpha) - np.log(once.fitted[-1].alpha))
        for name, fit, model in (("N", _ours, ours), ("F", _ScreenedOnce, once)):
            truth = float(np.mean((fresh_y - model.predict(fresh)) ** 2))
            gaps[name].append(_outer_score(fit, X, y, d) - truth)
    ratio = np.asarray(log_ratio)
    se = ratio.std(ddof=1) / np.sqrt(len(ratio))
    assert ratio.mean() > 3 * se, (ratio.mean(), se)
    for name, gap in gaps.items():
        gap = np.asarray(gap)
        assert abs(gap.mean()) <= 2 * gap.std(ddof=1) / np.sqrt(len(gap)), (name, gap.mean())
