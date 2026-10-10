"""T1 · Tuning outside nested cross-validation is optimistic; nesting is not (RECIPES_AND_TUNING §8
T1, §4.6; WAVE_C6A_PLAN §6). **Slow and scheduled with Nolan: never run unannounced.**

It is marked ``slow`` and also skipped unless ``TURBOTAB_SCHEDULED_RUNS=1``: the full CI tier runs
every slow test, and this one takes about 30 to 60 minutes. ``TURBOTAB_T1_JOBS`` sets how many
datasets run at once (1 by default: the dev machine is beside the bed).

**Generators** (n = 200, ten N(0, 1) predictors):

* (a) null: y ~ Bernoulli(0.3), so no model can beat the entropy, 0.6109 nats;
* (b) signal: logit = −0.85 + 0.8·x₁ − 0.6·x₂ + 0.5·x₁·x₃.

**Procedure F (flat, what not to do)**, in plain scikit-learn: the plan's candidates scored by
stratified 5-fold cross-validation on every row, the best one's mean log loss reported.
**Procedure N (ours)**: boosted trees' cross-validated log loss over the app's own outer folds,
each outer fit running the plan's search nested inside it, with the automatic plan forced on (made
as at an effective size of 300, so the Sobol candidates are drawn; the K floor still reads the
real event count).

**Truth:** each procedure retrained on 4/5 of the rows and scored on 20,000 fresh rows from the
same generator. **Assertions over 200 datasets,** with the Monte Carlo standard error of the mean
of the paired difference: F's mean lies below its truth by more than 3 standard errors; N's mean
lies within 2 of its truth; with ridge added, the BBC-CV selection-corrected estimate (Tsamardinos
et al. 2018) lies within 2 standard errors of the chosen family's truth, or above it
(conservative). Under (a) every truth averages at least the entropy. **Reported, not asserted:**
nested tuning against the standard settings on fresh data (§4.6's threshold of 300).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import tuning as T
from turbotab.core.models.base import get_family
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.metrics import cross_validate, fold_pairs
from turbotab.core.models.pipeline import DesignSpec, build_pipeline, make_plans
from turbotab.core.models.selection import OutOfFold, selection_optimism
from turbotab.core.stages.rows import draw_split

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("TURBOTAB_SCHEDULED_RUNS") != "1",
                       reason="T1 takes 30 to 60 minutes: scheduled with Nolan "
                              "(TURBOTAB_SCHEDULED_RUNS=1)"),
]

DATASETS = 200
N, P, FRESH = 200, 10, 20_000
ENTROPY = -(0.3 * math.log(0.3) + 0.7 * math.log(0.7))  # 0.6109 nats
COLS = [f"x{i}" for i in range(1, P + 1)]


def _draw(kind: str, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    X = rng.normal(size=(n, P))
    if kind == "null":
        p = np.full(n, 0.3)
    else:
        eta = -0.85 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
        p = 1 / (1 + np.exp(-eta))
    return pd.DataFrame(X, columns=COLS), (rng.random(n) < p).astype(int)


def _fresh_loss(model, X: pd.DataFrame, y: np.ndarray) -> float:
    return float(log_loss(y, model.predict_proba(X)[:, 1], labels=[0, 1]))


def _forced_plan(y: np.ndarray, seed: int):
    """The automatic plan, forced on: made as at an effective size of at least 300."""
    family = get_family("boosted_trees")
    n_plan, plan_rows, unit = T.plan_size("binary", y, None, folds=5)
    return T.make_plan(family, task="binary", loss="log_loss", n_plan=max(n_plan, T.SMALL_N),
                       plan_rows=plan_rows, unit=unit, split_seed=seed, rarest=n_plan)


def _one(kind: str, r: int) -> dict[str, float]:
    rng = np.random.default_rng([1, r, 0 if kind == "null" else 1])
    X, y = _draw(kind, N, rng)
    X_new, y_new = _draw(kind, FRESH, rng)
    trees = get_family("boosted_trees")
    plan = _forced_plan(y, seed=r)
    assert len(plan.candidates) > 1 and plan.chooses()
    spec = DesignSpec(predictors=COLS, inputs=COLS, categorical=[], numeric=COLS, energy=None,
                      impute=False)
    spec = replace(spec, plans={**{k: p.to_dict() for k, p in
                                   make_plans([get_family("ridge")], "binary", y, folds=5,
                                              split_seed=r).items()},
                                trees.key: plan.to_dict()})
    pipes = {k: build_pipeline(spec, get_family(k), "binary", "prediction", N, P)
             for k in ("boosted_trees", "ridge")}

    # F: every candidate scored by 5-fold CV on all rows, the best one's loss reported
    skf = list(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y))
    flat = []
    for c in plan.candidates:
        params = T.estimator_params(trees, "binary", c.values, n_units=N, n_rows=N, plan=plan)
        losses = [_fresh_loss(HistGradientBoostingClassifier(early_stopping=False, **params)
                              .fit(X.iloc[tr], y[tr]), X.iloc[te], y[te]) for tr, te in skf]
        flat.append((float(np.mean(losses)), params))
    f_est, f_params = min(flat, key=lambda t: t[0])
    train = skf[0][0]  # 4/5 of the rows
    f_truth = _fresh_loss(HistGradientBoostingClassifier(early_stopping=False, **f_params)
                          .fit(X.iloc[train], y[train]), X_new, y_new)

    # N: the app's outer folds, the search nested in every outer fit
    split, _ = draw_split(np.arange(N), holdout=0.0, seed=r, folds=5, y=y)
    pairs = fold_pairs(split.sort_values("row_id")["fold"].to_numpy())
    oof = OutOfFold("binary", X, y, pairs)
    results = {}
    for key, pipe in pipes.items():
        fit = oof.wrap(key, lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf, seed=r))
        results[key] = cross_validate("binary", lambda _p=pipe: clone(_p), X, y, pairs, fit=fit)
    n_est = results["boosted_trees"].summary("binary")["log_loss"]["estimate"]
    truths = {k: _fresh_loss(fit_pipeline(clone(p), X.iloc[train], y[train], seed=r),
                             X_new, y_new) for k, p in pipes.items()}
    chosen = selection_optimism("binary", "log_loss", results, oof)
    standard = _fresh_loss(HistGradientBoostingClassifier().fit(X.iloc[train], y[train]),
                           X_new, y_new)
    return {"f_est": f_est, "f_truth": f_truth, "n_est": n_est,
            "n_truth": truths["boosted_trees"], "corrected": chosen["corrected"],
            "chosen_truth": truths[chosen["best"]], "standard_truth": standard}


def _mean_se(diff: np.ndarray) -> tuple[float, float]:
    return float(np.mean(diff)), float(np.std(diff, ddof=1) / math.sqrt(len(diff)))


@pytest.mark.parametrize("kind", ["null", "signal"])
def test_t1_flat_tuning_is_optimistic_and_nested_tuning_is_not(kind):
    from joblib import Parallel, delayed

    jobs = int(os.environ.get("TURBOTAB_T1_JOBS", "1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = Parallel(n_jobs=jobs)(delayed(_one)(kind, r) for r in range(DATASETS))
    table = pd.DataFrame(rows)
    f_gap, f_se = _mean_se((table["f_truth"] - table["f_est"]).to_numpy())
    n_gap, n_se = _mean_se((table["n_est"] - table["n_truth"]).to_numpy())
    c_gap, c_se = _mean_se((table["corrected"] - table["chosen_truth"]).to_numpy())
    s_gap, s_se = _mean_se((table["n_truth"] - table["standard_truth"]).to_numpy())
    print(f"\nT1 ({kind}, {DATASETS} datasets): flat optimism {f_gap:.4f} ± {f_se:.4f}; nested "
          f"bias {n_gap:+.4f} ± {n_se:.4f}; corrected − truth {c_gap:+.4f} ± {c_se:.4f}; "
          f"nested − standard on fresh data {s_gap:+.4f} ± {s_se:.4f} (reported, §4.6)")
    assert f_gap > 3 * f_se, (f_gap, f_se)
    assert abs(n_gap) <= 2 * n_se, (n_gap, n_se)
    assert c_gap >= -2 * c_se, (c_gap, c_se)
    if kind == "null":
        for column in ("f_truth", "n_truth", "chosen_truth", "standard_truth"):
            assert table[column].mean() >= ENTROPY, (column, table[column].mean())
