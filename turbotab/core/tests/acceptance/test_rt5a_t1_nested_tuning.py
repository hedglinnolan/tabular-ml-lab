"""T1 · Tuning outside nested cross-validation is optimistic; nesting is not (RECIPES_AND_TUNING §8
T1, §4.6; WAVE_C6A_PLAN §6; notes/STRATIFICATION_OPTIMISM.md). **Slow and scheduled with Nolan:
never run unannounced.**

It is marked ``slow`` and also skipped unless ``TURBOTAB_SCHEDULED_RUNS=1``: the full CI tier runs
every slow test, and this one takes about an hour at 4 jobs (600 datasets per generator, about 12 s
of one core each). ``TURBOTAB_T1_JOBS`` sets how many datasets run at once (1 by default: the dev
machine is beside the bed). ``TURBOTAB_T1_DATASETS`` shortens a smoke run of the code path; the
assertions are sized for 600 and are expected to fail on far fewer.

**Generators** (n = 200, ten N(0, 1) predictors):

* (a) null: y ~ Bernoulli(0.3), so no model can beat the entropy, 0.6109 nats;
* (b) signal: logit = −0.85 + 0.8·x₁ − 0.6·x₂ + 0.5·x₁·x₃.

**Procedure F (flat, what not to do)**, in plain scikit-learn: the plan's candidates scored by
stratified 5-fold cross-validation on every row, the best one's mean log loss reported.
**Procedure N (ours)**: boosted trees' cross-validated log loss over the app's own outer folds,
each outer fit running the plan's search nested inside it, with the automatic plan forced on (made
as at an effective size of 300, so the Sobol candidates are drawn; the K floor still reads the
real event count). **Procedure C**: with ridge added, BBC-CV's selection-corrected estimate
(Tsamardinos et al. 2018) for the family with the better cross-validated score. F and N use the
same five folds (the split stage's are scikit-learn's ``StratifiedKFold`` at the same seed,
checked per dataset).

**Truth:** a procedure's own five outer-fold models, each scored by its expected log loss on
20,000 fresh rows, the outcome integrated out with its true probability, averaged over the folds.
Every fold model is the procedure refit on 4/5 of the rows (fold 0's alone was the truth before),
so the mean is unchanged and the noise of a single refit is gone.

**The stratification term.** Stratified folds make every fold's event rate the sample's, so a
cross-validated log loss never pays for estimating the base rate, while fresh data does: the
estimate is low by about ρ/n nats (1/n = 0.005 under the null; derived and simulated in
notes/STRATIFICATION_OPTIMISM.md). It is not a tuning leak, so every assertion removes it:
(a) by pairing F with N on the same folds, (b) and (c) with a **control** on the same folds that
fits nothing but the base rate (the true logit, its intercept refit on each training fold, which
under the null is the training fold's event rate). The control's optimism is the term alone.

**Assertions over 600 datasets per generator,** each at a one-sided α = 0.05, with the Monte Carlo
standard error of the mean of the paired, per-dataset difference (optimism O = truth − estimate):

* (a) tuning outside nesting is optimistic: O_F − O_N > 0;
* (b) nesting removes the optimism: O_N − O_control is equivalent to 0 within δ = 0.005 nats (two
  one-sided tests: the 90% interval lies inside ±δ). δ is half of the smallest log-loss gain the
  app calls meaningful (``baseline.MIN_GAIN``). F's own leak here is of the same size, so it is
  (a) that tells nesting from flat tuning; (b) bounds what is left;
* (c) BBC-CV's corrected estimate is not optimistic: O_C − O_control < δ (non-inferiority);
* under (a), every truth averages at least the entropy.

Power (pilot of 40 datasets per generator; notes/STRATIFICATION_OPTIMISM.md §6): at least 0.93
for each assertion at the effects a correct implementation shows, about 0.93 that all six pass.
**Reported, not asserted:** each optimism, the control's against ρ/n, and nested tuning against
the standard settings on fresh data (§4.6's threshold of 300; notes/SMALL_SAMPLE_STANDARD.md).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import tuning as T
from turbotab.core.models.base import get_family
from turbotab.core.models.baseline import MIN_GAIN
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.metrics import cross_validate, fold_pairs
from turbotab.core.models.pipeline import DesignSpec, build_pipeline, make_plans
from turbotab.core.models.selection import OutOfFold, selection_optimism
from turbotab.core.stages.rows import draw_split

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("TURBOTAB_SCHEDULED_RUNS") != "1",
                       reason="T1 takes about an hour at 4 jobs: scheduled with Nolan "
                              "(TURBOTAB_SCHEDULED_RUNS=1)"),
]

DATASETS = int(os.environ.get("TURBOTAB_T1_DATASETS", "600"))
N, P, FRESH = 200, 10, 20_000
ENTROPY = -(0.3 * math.log(0.3) + 0.7 * math.log(0.7))  # 0.6109 nats
COLS = [f"x{i}" for i in range(1, P + 1)]
Z = 1.6448536269514722  # the normal's 95th percentile: one-sided α = 0.05
DELTA = MIN_GAIN / 2  # 0.005 nats: the equivalence and non-inferiority margin


def _draw(kind: str, n: int, rng: np.random.Generator
          ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Predictors, outcome and each row's true probability (the draws T1 has always made)."""
    X = rng.normal(size=(n, P))
    if kind == "null":
        p = np.full(n, 0.3)
    else:
        eta = -0.85 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
        p = 1 / (1 + np.exp(-eta))
    return pd.DataFrame(X, columns=COLS), (rng.random(n) < p).astype(int), p


def _expected_loss(p_true: np.ndarray, p_hat: np.ndarray) -> float:
    """The log loss on these rows with each outcome integrated out at its true probability."""
    q = np.clip(p_hat, 1e-15, 1 - 1e-15)
    return float(np.mean(-(p_true * np.log(q) + (1 - p_true) * np.log(1 - q))))


def _held_out(y: np.ndarray, p_hat: np.ndarray) -> float:
    return float(log_loss(y, p_hat, labels=[0, 1]))


def _offset(eta: np.ndarray, y: np.ndarray) -> float:
    """The intercept's maximum-likelihood shift with the logit ``eta`` held fixed (Newton)."""
    c = 0.0
    for _ in range(50):
        q = expit(eta + c)
        step = (y.sum() - q.sum()) / max(float((q * (1 - q)).sum()), 1e-12)
        c += step
        if abs(step) < 1e-12:
            break
    return float(c)


def _forced_plan(y: np.ndarray, seed: int):
    """The automatic plan, forced on: made as at an effective size of at least 300."""
    family = get_family("boosted_trees")
    n_plan, plan_rows, unit = T.plan_size("binary", y, None, folds=5)
    return T.make_plan(family, task="binary", loss="log_loss", n_plan=max(n_plan, T.SMALL_N),
                       plan_rows=plan_rows, unit=unit, split_seed=seed, rarest=n_plan)


def _one(kind: str, r: int) -> dict[str, float]:
    """One dataset, every fit on one OpenMP thread (the plan's; plain scikit-learn's by the
    limit), so parallel datasets never oversubscribe the machine."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="openmp"):
        return _one_dataset(kind, r)


def _one_dataset(kind: str, r: int) -> dict[str, float]:
    rng = np.random.default_rng([1, r, 0 if kind == "null" else 1])
    X, y, p = _draw(kind, N, rng)
    X_new, _, p_new = _draw(kind, FRESH, rng)

    def truth(models) -> float:
        return float(np.mean([_expected_loss(p_new, m.predict_proba(X_new)[:, 1])
                              for m in models]))

    trees = get_family("boosted_trees")
    plan = _forced_plan(y, seed=r)
    assert len(plan.candidates) > 1 and plan.chooses()
    spec = DesignSpec(predictors=COLS, inputs=COLS, categorical=[], numeric=COLS, energy=None,
                      impute=False)
    spec = replace(spec, plans={**{k: q.to_dict() for k, q in
                                   make_plans([get_family("ridge")], "binary", y, folds=5,
                                              split_seed=r).items()},
                                trees.key: plan.to_dict()})
    pipes = {k: build_pipeline(spec, get_family(k), "binary", "prediction", N, P)
             for k in ("boosted_trees", "ridge")}

    # The app's outer folds; F and the control use the same ones, so the stratification term
    # is shared (and cancels in F − N).
    skf = list(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y))
    split, _ = draw_split(np.arange(N), holdout=0.0, seed=r, folds=5, y=y)
    pairs = fold_pairs(split.sort_values("row_id")["fold"].to_numpy())
    assert (sorted(tuple(np.flatnonzero(te)) for _, _, te in pairs)
            == sorted(tuple(te) for _, te in skf)), "F and N must share their folds"

    # F: every candidate scored by 5-fold CV on all rows, the best one's loss reported
    flat = []
    for c in plan.candidates:
        params = T.estimator_params(trees, "binary", c.values, n_units=N, n_rows=N, plan=plan)
        params.pop("n_threads")  # the family's pin; plain scikit-learn runs inside _one's limit
        models, losses = [], []
        for tr, te in skf:
            m = HistGradientBoostingClassifier(early_stopping=False, **params)
            models.append(m.fit(X.iloc[tr], y[tr]))
            losses.append(_held_out(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        flat.append((float(np.mean(losses)), models))
    f_est, f_models = min(flat, key=lambda t: t[0])  # ties: the first candidate

    # N: the search nested in every outer fit; ridge beside it for BBC-CV
    oof = OutOfFold("binary", X, y, pairs)
    results, fitted = {}, {k: [] for k in pipes}
    for key, pipe in pipes.items():
        def fit_and_keep(m, Xf, yf, rows, _key=key):
            model = fit_pipeline(m, Xf, yf, seed=r)
            fitted[_key].append(model)
            return model
        results[key] = cross_validate("binary", lambda _p=pipe: clone(_p), X, y, pairs,
                                      fit=oof.wrap(key, fit_and_keep))
    n_est = results["boosted_trees"].summary("binary")["log_loss"]["estimate"]
    truths = {k: truth(models) for k, models in fitted.items()}
    chosen = selection_optimism("binary", "log_loss", results, oof)

    # The control: the true logit with its intercept refit on each training fold
    eta, eta_new = logit(p), logit(p_new)
    c_est, c_truth = [], []
    for tr, te in skf:
        shift = _offset(eta[tr], y[tr])
        c_est.append(_held_out(y[te], expit(eta[te] + shift)))
        c_truth.append(_expected_loss(p_new, expit(eta_new + shift)))

    # Reported: the standard settings on the same folds, and N's ρ (the stratification
    # term's factor, notes/STRATIFICATION_OPTIMISM.md §3)
    standard = truth([HistGradientBoostingClassifier().fit(X.iloc[tr], y[tr]) for tr, _ in skf])
    pbar = float(p_new.mean())
    rho = float(np.mean([np.mean(p_new * (1 - q)) / pbar + np.mean(q * (1 - p_new)) / (1 - pbar)
                         for q in (m.predict_proba(X_new)[:, 1] for m in fitted["boosted_trees"])]))
    return {"f_est": f_est, "f_truth": truth(f_models), "n_est": n_est,
            "n_truth": truths["boosted_trees"], "c_est": chosen["corrected"],
            "c_truth": truths[chosen["best"]], "control_est": float(np.mean(c_est)),
            "control_truth": float(np.mean(c_truth)), "standard_truth": standard, "rho": rho}


def _mean_se(diff: pd.Series) -> tuple[float, float]:
    x = diff.to_numpy(dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


@pytest.mark.parametrize("kind", ["null", "signal"])
def test_t1_flat_tuning_is_optimistic_and_nested_tuning_is_not(kind):
    from joblib import Parallel, delayed

    jobs = int(os.environ.get("TURBOTAB_T1_JOBS", "1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = Parallel(n_jobs=jobs)(delayed(_one)(kind, r) for r in range(DATASETS))
    t = pd.DataFrame(rows)
    opt = {k: t[f"{k}_truth"] - t[f"{k}_est"] for k in ("f", "n", "c", "control")}
    a, a_se = _mean_se(opt["f"] - opt["n"])
    b, b_se = _mean_se(opt["n"] - opt["control"])
    c, c_se = _mean_se(opt["c"] - opt["control"])
    s, s_se = _mean_se(opt["control"])
    g, g_se = _mean_se(t["n_truth"] - t["standard_truth"])
    print(f"\nT1 ({kind}, {DATASETS} datasets; optimism = truth − estimate, nats):"
          + "".join(f" {k} {m:+.4f} ± {e:.4f};" for k, (m, e) in
                    {key: _mean_se(v) for key, v in opt.items()}.items())
          + f"\n (a) F − N {a:+.5f} ± {a_se:.5f} (z {a / a_se:.2f}, needs > {Z:.3f});"
          f" (b) N − control {b:+.5f} ± {b_se:.5f}, 90% interval [{b - Z * b_se:+.5f},"
          f" {b + Z * b_se:+.5f}] inside ±{DELTA};"
          f" (c) C − control {c:+.5f} ± {c_se:.5f}, upper {c + Z * c_se:+.5f} < {DELTA}"
          f"\n control (the stratification term) {s:+.5f} ± {s_se:.5f} against ρ/n with N's ρ"
          f" {t['rho'].mean():.3f}: {t['rho'].mean() / N:.5f}; nested − standard on fresh data"
          f" {g:+.4f} ± {g_se:.4f} (reported, §4.6)")
    assert a - Z * a_se > 0, ("(a) flat tuning is not more optimistic than nested", a, a_se)
    assert -DELTA < b - Z * b_se and b + Z * b_se < DELTA, ("(b) nested's optimism", b, b_se)
    assert c + Z * c_se < DELTA, ("(c) BBC-CV's corrected estimate is optimistic", c, c_se)
    if kind == "null":
        for column in ("f_truth", "n_truth", "c_truth", "control_truth", "standard_truth"):
            assert t[column].mean() >= ENTROPY, (column, t[column].mean())
