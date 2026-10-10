"""The engine's path search with the elastic net on it: a class an inner split lacks, the imbalance
correction, and the time estimate (RT-5f's verifier; RECIPES_AND_TUNING §4.3, §4.4).

* **A class an inner split lacks** (the verifier's rare.py and rare_design.py). The path models
  the classes of the rows it is fit on; the fit's other classes had no column, and the pooled
  predictions could not be stacked ("all the input array dimensions … size 4 … size 3"). A class
  held in one PSU (inner splits by PSU are not stratified), or one row of a class in a fit, raised
  where the cross-validated estimator it replaced did not. Reference: a searched family's rule,
  written out here: a class the split's model never saw has probability 0, clipped as scikit-learn's
  ``log_loss`` clips it, so every grid point pays the same for those rows.
* **The imbalance correction** (imb.py). A candidate is the wrapped, recalibrated model (RECIPES
  §4.3: the search scores exactly what will be deployed), but the path was run on the bare,
  unweighted logistic loss: its curve was bit-identical with and without the correction, and the
  plan counted 25 fits where 6 were made. Reference: the deployed wrapper itself, fit on each inner
  split's training rows at each grid point's C = 1/(n·r·λ_max) of those rows, its log loss on the
  validation rows computed here.
* **The time estimate** (cost.py). One plain fit at the grid's center times the plan's fits missed
  that every inner split runs the whole path (six mixes × 100 ratios): 33× short at 1,500 × 30. The
  path family's search is timed itself.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from turbotab.core.models import tuning as T
from turbotab.core.models.base import get_family
from turbotab.core.models.inner_cv import fit_pipeline

EPS = np.finfo(float).eps


def _fit(pipe, X, y, **kw):
    drawn: list[T.Drawn] = []
    with T.observing(drawn.append), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit_pipeline(pipe, X, y, seed=3, **kw)
    return fitted, [d for d in drawn if d.kind == "inner"]


def _pipe(key: str, task: str, plan, n: int, p: int):
    family = get_family(key)
    return T.TunedPipeline([("scale", StandardScaler()),
                            ("model", family.build(task, "prediction", n, p))],
                           search=plan).set_output(transform="pandas")


@pytest.mark.parametrize("key", ["elastic_net", "ridge"])
def test_a_class_one_row_holds_is_missing_from_an_inner_split_and_costs_every_point_alike(key):
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = np.digitize(X["a"].to_numpy() + rng.normal(0, 0.5, n), [-0.5, 0.5])
    y[0] = 3  # one row of a fourth class
    family = get_family(key)
    plan = T.make_plan(family, task="multiclass", loss="log_loss", n_plan=n, plan_rows=n,
                       unit="rarest_class", rarest=40, split_seed=3)
    fitted, inner = _fit(_pipe(key, "multiclass", plan, n, 4), X, y)
    record = fitted.tuning_
    losses = np.asarray(record.losses, dtype=float)
    assert np.isfinite(losses).all() and record.inner_k_used == plan.inner_k
    assert list(fitted[-1].classes_) == [0, 1, 2, 3]  # the refit sees every class
    lacking = [d for d in inner if 3 not in set(y[np.asarray(d.train, dtype=int)])]
    assert len(lacking) == 1  # the split whose validation rows hold the one row
    # the one row costs −log(eps) at every grid point: the curve is the same less that constant
    rows = sum(len(d.validation) for d in inner)
    floor = -np.log(EPS) / rows
    assert np.all(losses > floor)


def test_a_class_held_in_one_psu_is_scored_under_the_population_answer():
    rng = np.random.default_rng(1)
    n = 300
    strata, psu = np.repeat(np.arange(15), 20), np.tile(np.repeat([1, 2], 10), 15)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = np.digitize(X["a"].to_numpy() + rng.normal(0, 0.5, n), [-0.5, 0.5])
    y[(strata == 0) & (psu == 1)] = 3  # a class in one PSU, 10 rows: the plan's floor is met
    design = T.FitDesign(strata=strata, psu=psu, weights=rng.uniform(0.5, 2, n))
    for key in ("elastic_net", "ridge"):
        plan = T.make_plan(get_family(key), task="multiclass", loss="log_loss", n_plan=n,
                           plan_rows=n, unit="rarest_class", rarest=10, psus=30, weighted=True,
                           split_seed=3)
        fitted, inner = _fit(_pipe(key, "multiclass", plan, n, 4), X, y, design=design)
        assert np.isfinite(np.asarray(fitted.tuning_.losses, dtype=float)).all(), key
        assert any(3 not in set(y[np.asarray(d.train, dtype=int)]) for d in inner), key


def test_under_the_imbalance_correction_the_path_search_scores_the_wrapped_model():
    from sklearn.base import clone

    from turbotab.core.methods.levers import wrap_model

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (X["a"] + rng.normal(0, 1, n) > 1.6).astype(int).to_numpy()
    family = get_family("elastic_net")
    curves = {}
    for lever in ("none", "weights"):
        plan = T.make_plan(family, task="binary", loss="log_loss", n_plan=n, plan_rows=n,
                           unit="events", rarest=int(y.sum()), split_seed=3,
                           imbalance=lever != "none")
        model = wrap_model(family.build("binary", "prediction", n, 5), {"imbalance": lever},
                           "binary")
        pipe = T.TunedPipeline([("scale", StandardScaler()), ("model", model)],
                               search=plan).set_output(transform="pandas")
        fitted, inner = _fit(pipe, X, y)
        curves[lever] = np.asarray(fitted.tuning_.losses, dtype=float)
    assert not np.array_equal(curves["none"], curves["weights"])
    K, C = plan.inner_k, len(plan.candidates)
    assert plan.fits() == T.IMBALANCE_FITS * ((K - 1) * C + 1)
    assert fitted.tuning_.n_fits == K * C + 1
    # the deployed wrapper on each split at each grid point (the engine's documented parts, RECIPES
    # §4.3 steps 1–3, for the head and the wrapper's own inner folds), C from λ_max's definition
    # on the split's scaled training rows, the pooled log loss by hand
    from sklearn.pipeline import Pipeline

    total = np.zeros(C)
    seed = T.fit_seed(plan)
    for d in inner:
        train, val = np.asarray(d.train, dtype=int), np.asarray(d.validation, dtype=int)
        steps = Pipeline([("scale", StandardScaler()), ("model", clone(model))])
        head = T.fit_head(steps, X, y, train, seed=seed, stop_share=0.0, classify=True)
        Zt, Zv = np.asarray(head.Z, dtype=float), np.asarray(head.transform(X.iloc[val]))
        yt = np.asarray(head.y)
        for c, cand in enumerate(plan.candidates):
            mix, ratio = cand.values["l1_ratio"], cand.values["ratio"]
            r = (yt - yt.mean())[:, None]
            top = float(np.max(np.abs((Zt - Zt.mean(axis=0)).T @ r))) / (len(yt) * mix)
            wrapped = clone(steps.steps[-1][1]).set_params(
                estimator__C=1.0 / (len(yt) * ratio * top), estimator__l1_ratio=mix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T.fit_model(wrapped, head, seed=seed)
                p1 = wrapped.predict_proba(Zv)[:, 1]
            p1 = np.clip(p1, EPS, 1 - EPS)
            total[c] += -(y[val] * np.log(p1) + (1 - y[val]) * np.log(1 - p1)).sum()
    by_hand = total / sum(len(d.validation) for d in inner)
    assert np.max(np.abs(curves["weights"] - by_hand)) <= 1e-9 * by_hand.min()


def test_a_path_familys_time_estimate_times_its_search(monkeypatch):
    from turbotab.core.decisions import ProjectState
    from turbotab.core.models import cost

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
    frame["y"] = frame["x1"] + rng.normal(size=120)

    class Store:
        def materialize(self, columns, ids):
            return frame.loc[np.asarray(ids), list(columns)]

    family = get_family("elastic_net")
    plan = T.make_plan(family, task="regression", loss="mse", n_plan=120, plan_rows=120,
                       unit="units", split_seed=1)
    real = cost.time_one_fit
    kinds: list[str] = []

    def timed(pipeline, X, y):
        with T.observing(lambda d: kinds.append(d.kind)):
            real(pipeline, X, y)
        return 0.5

    monkeypatch.setattr(cost, "time_one_fit", timed)
    state = ProjectState(target="y", task="regression", purpose="prediction",
                         roles={"x1": "exposure", "x2": "covariate"})
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120), [family], folds=5,
                             plans={family.key: plan})
    assert kinds.count("inner") == plan.inner_k  # the timed fit ran the search
    assert out["elastic_net"].seconds == 0.5 * 5  # the search and refit, once per fold
    assert out["elastic_net"].text.endswith(cost.TUNING_CLAUSE)
