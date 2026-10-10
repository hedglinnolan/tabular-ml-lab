"""RT-1b · the search engine, nested in every fit (RECIPES_AND_TUNING §4.2–§4.7; WAVE_C6A_PLAN §2,
§5).

No family is tuned yet (RT-5a and RT-5f declare the trees' and the elastic net's tuning), so each
test registers a small probe family for its own duration, with a declaration made here. Every
expected value is computed independently of ``models/tuning.py``:

* **T3** the inner splits: partitions worked by hand from the fixture's persons, times and PSUs,
  and the identity with scikit-learn's ``GroupKFold`` over the units sorted as text, at the seed
  recomputed here from the SHA-256 of the canonical JSON of (split seed, "inner splits"); the
  weighted inner loss refit by hand, each candidate a direct ``HistGradientBoostingRegressor`` on
  the observed inner rows, Σ w·(y − ŷ)² / Σ w written out; the oversample lever's stopping and
  recalibration rows read off spies on the wrapped trees;
* **T4** perturbation: held-out rows changed (values scaled, outcomes permuted, so class counts
  hold) leave the fold's record and fit unchanged bit for bit; a training row changes them;
* **T9** the floor: a fit with fewer than 2 × K of the rarest class keeps K and is counted;
* **T13** replay: the same seed reproduces bit for bit; another split seed draws other candidates;
  pinned replay through the plain path reproduces the deployed predictions with one fit;
* **T17** one plan, observed through ``observing()`` in every outer fold, bootstrap resample and
  the final refit;
* **F13** folds of 9,999 and 10,001 rows stop alike under one plan, against scikit-learn's own
  ``"auto"`` rule, which stops one and not the other;
* **T6 (partial)** an instrumented counter of the rows every model fit is handed, over the fit's
  rows, equals ``plan.fits()`` (searched, with the imbalance lever, out of bag, on a path, and
  with nothing to choose); out-of-bag losses equal direct ``RandomForestRegressor`` fits'; a
  path's pooled losses equal direct ``Ridge`` fits' on the observed splits;
* Cancel stops a search within about 2 seconds; the estimate counts the plan's fits.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from turbotab.core.jobs import Cancelled
from turbotab.core.models import base as B
from turbotab.core.models import tuning as T
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.tuning import Dimension, TunedPipeline, TuningDecl

SEEN: list[tuple[int, int]] = []  # each spied model fit: (training rows, stopping rows)
SLOW = {"seconds": 0.0}


def _dim(name, low, high, scale, **more):
    return Dimension(name, f"plain words for {name}", f"term for {name}", low, high, scale,
                     source="Friedman 2001", **more)


STOPPING = {"param": "max_iter", "rounds": 30, "patience_param": "n_iter_no_change",
            "patience": 3, "share": 0.1}
TREES = TuningDecl(
    "search",
    dimensions=(_dim("learning_rate", 0.05, 0.3, "log"),
                _dim("max_leaf_nodes", 4, 16, "log_int"),
                _dim("max_iter", 5, 25, "log_int", active="without_early_stopping")),
    standard={"learning_rate": 0.1, "max_leaf_nodes": 8, "max_iter": 15},
    standard_source="a probe's own", early_stopping=STOPPING, space_version="probe_trees/1",
    max_drawn=4)
FOREST = TuningDecl(
    "search",
    dimensions=(_dim("max_features", 0.3, 1.0, "linear"), _dim("min_samples_leaf", 1, 20, "log_int")),
    standard={"max_features": 1.0, "min_samples_leaf": 5}, out_of_bag=True,
    space_version="probe_forest/1", max_drawn=4)
RIDGE = TuningDecl("path", dimensions=(_dim("lambda", 1e-3, 10.0, "log", points=8),),
                   space_version="probe_ridge/1")


class SpyRegressor(HistGradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None, sample_weight_val=None):
        SEEN.append((len(X), 0 if X_val is None else len(X_val)))
        if SLOW["seconds"]:
            time.sleep(SLOW["seconds"])
        return super().fit(X, y, sample_weight=sample_weight, X_val=X_val, y_val=y_val,
                           sample_weight_val=sample_weight_val)


class SpyClassifier(HistGradientBoostingClassifier):
    rows: list = []

    def fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None, sample_weight_val=None):
        SEEN.append((len(X), 0 if X_val is None else len(X_val)))
        SpyClassifier.rows.append((list(X.index), None if X_val is None else list(X_val.index)))
        return super().fit(X, y, sample_weight=sample_weight, X_val=X_val, y_val=y_val,
                           sample_weight_val=sample_weight_val)


class SpyForest(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None):
        SEEN.append((len(X), 0))
        return super().fit(X, y, sample_weight=sample_weight)


class SpyRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        SEEN.append((len(X), 0))
        return super().fit(X, y, sample_weight=sample_weight)


def _trees(task, purpose=None, n_rows=0, n_features=0):
    cls = SpyRegressor if task == "regression" else SpyClassifier
    return cls(max_iter=15, max_leaf_nodes=8, random_state=0, early_stopping="auto")


def _ridge_path(Z, y, grid, *, task, weights=None):
    """The ridge path by its normal equations, centered, the intercept unpenalized, α = n·λ."""
    Z, y = np.asarray(Z, dtype=float), np.asarray(y, dtype=float)
    lambdas = np.asarray(grid["lambda"], dtype=float)
    mean_z, mean_y = Z.mean(axis=0), y.mean()
    Zc, yc = Z - mean_z, y - mean_y
    p = Z.shape[1]
    coefs = np.stack([np.linalg.solve(Zc.T @ Zc + len(y) * lam * np.eye(p), Zc.T @ yc)
                      for lam in lambdas])
    intercepts = mean_y - coefs @ mean_z
    return T.PathFit(values=lambdas[None, :], coefs=coefs[None, :, None, :],
                     intercepts=intercepts[None, :, None])


def _family(key, decl, build, *, path=None, settings=None, tasks=("regression", "binary")):
    return SimpleNamespace(key=key, label=f"Probe {key}", tasks=tasks, tuning=decl,
                           defaults_version="1", settings=settings, path=path, preprocess=None,
                           needs_scaling=False, build_for=None, cost_model="cells",
                           build=build)


@pytest.fixture(autouse=True)
def one_thread():
    """Every fit at a stated thread count (RECIPES §4.7; CI runs two workers): one, which also
    keeps the many small fits here fast."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def probe(monkeypatch):
    SEEN.clear()
    SpyClassifier.rows = []
    SLOW["seconds"] = 0.0
    family = _family("probe_trees", TREES, _trees)
    monkeypatch.setitem(B._REGISTRY, family.key, family)
    return family


def _seed(*parts) -> int:
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _plan(family, task="regression", *, n_plan=400, plan_rows=None, seed=7, **more):
    unit = "units" if task == "regression" else "events"
    loss = "mse" if task == "regression" else "log_loss"
    return T.make_plan(family, task=task, loss=loss, n_plan=n_plan,
                       plan_rows=n_plan if plan_rows is None else plan_rows, unit=unit,
                       split_seed=seed, **more)


def _tuned(family, plan, task="regression", model=None):
    return TunedPipeline([("model", model if model is not None
                           else family.build(task, "prediction", 0, 0))], search=plan)


def _people(n_people=100, rows_each=3, *, seed=0, task="regression"):
    rng = np.random.default_rng(seed)
    person = np.repeat(np.arange(n_people), rows_each)
    effect = rng.normal(size=n_people)[person]
    X = pd.DataFrame(rng.normal(size=(len(person), 4)), columns=[f"x{i}" for i in range(4)])
    X.index = pd.Index(np.arange(len(person)) + 10_000, name="row_id")  # row ids, not positions
    signal = X["x0"].to_numpy() + 0.5 * X["x1"].to_numpy() ** 2 + effect
    if task == "regression":
        return X, signal + rng.normal(scale=0.5, size=len(person)), person
    return X, (signal + rng.normal(size=len(person)) > 0.8).astype(int), person


def _observed(fit):
    drawn: list[T.Drawn] = []
    with T.observing(drawn.append):
        out = fit()
    return out, drawn


def _group_kfold(labels, k, seed):
    """scikit-learn's GroupKFold over the units sorted as text, as positions."""
    labels = np.asarray(labels, dtype=object)
    canon = np.argsort(labels, kind="stable")
    folds = GroupKFold(k, shuffle=True, random_state=seed).split(np.arange(len(labels)),
                                                                 groups=labels[canon])
    return [np.sort(canon[test]) for _, test in folds]


# ── T3 · inner splits follow the outer ones ───────────────────────────────────


def test_t3_no_person_sits_on_both_sides_of_an_inner_or_a_stopping_split(probe):
    X, y, person = _people()
    plan = _plan(probe, n_plan=1_600)  # the Sobol candidates stop early
    assert plan.early_stopping and plan.inner_k == 3 and len(plan.candidates) == 5
    fitted, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), X, y, groups=person,
                                                   seed=plan.split_seed))
    of = pd.Series(person, index=X.index)
    inner = [d for d in drawn if d.kind == "inner"]
    refit = [d for d in drawn if d.kind == "refit"]
    assert len(inner) == 3 and len(refit) == 1
    for d in [*inner, *refit]:
        train = set(of[d.train])
        stop = set() if d.stopping is None else set(of[d.stopping])
        assert not train & stop
        if d.validation is not None:
            assert not (train | stop) & set(of[d.validation])
    # every inner split draws its stopping persons (some candidate stops): a tenth of its persons
    for d in inner:
        persons = len(set(of[d.train]) | set(of[d.stopping]))
        assert len(set(of[d.stopping])) == round(0.1 * persons)
    # the validation persons are GroupKFold's partition of the persons, at the derived seed
    labels = [str(p) for p in person]
    expected = _group_kfold(labels, 3, _seed(plan.split_seed, "inner splits"))
    assert [sorted(d.validation) for d in inner] == [sorted(X.index[e]) for e in expected]
    assert sorted(np.concatenate([d.validation for d in inner])) == sorted(X.index)
    assert fitted.tuning_.inner_k_used == 3 and not fitted.tuning_.below_floor


def test_t3_under_time_every_inner_validation_unit_is_later_than_every_training_unit(probe):
    X, y, person = _people()
    when = np.random.default_rng(3).permutation(100)  # each person's rank in time
    order = when[person].astype(float)
    plan = _plan(probe, n_plan=1_600)
    _, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), X, y, groups=person,
                                              order=order, seed=plan.split_seed))
    t = pd.Series(order, index=X.index)
    inner = [d for d in drawn if d.kind == "inner"]
    assert len(inner) == 3
    for d in inner:
        assert max(t[d.train].max(), t[d.stopping].max()) < t[d.validation].min()
        assert t[d.train].max() < t[d.stopping].min()  # the stopping units are the latest
    for d in drawn:
        if d.kind == "refit" and d.stopping is not None:
            assert t[d.train].max() < t[d.stopping].min()


def _nhanes_shaped(seed=5):
    """15 strata × 2 PSUs, 10 rows in each PSU, survey weights."""
    rng = np.random.default_rng(seed)
    stratum = np.repeat(np.arange(15), 20)
    psu = np.tile(np.repeat([1, 2], 10), 15)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["x0", "x1", "x2"])
    X.index = pd.Index(np.arange(300) + 500, name="row_id")
    y = X["x0"].to_numpy() + np.sin(2 * X["x1"].to_numpy()) + rng.normal(scale=0.5, size=300)
    return X, y, stratum, psu, rng.uniform(0.5, 3.0, size=300)


def test_t3_under_the_population_answer_inner_splits_keep_whole_psus_and_the_loss_is_weighted(probe):
    from turbotab.core.models.design_cv import design_folds

    X, y, stratum, psu, w = _nhanes_shaped()
    fold, k, _ = design_folds(stratum, psu, len(y), 2, seed=11)
    assert k == 2
    # by hand: each outer training fold holds one PSU of every stratum, so P = 15 and
    # K = min(3, ⌊15/2⌋) = 3
    fewest = min(len(set(zip(stratum[fold != f], psu[fold != f]))) for f in (0, 1))
    assert fewest == 15
    plan = _plan(probe, n_plan=400, psus=fewest, weighted=True)
    assert plan.inner_k == 3 and not plan.early_stopping and not plan.standard_stops
    for f in (0, 1):
        rows = np.flatnonzero(fold != f)
        design = T.FitDesign(strata=stratum[rows], psu=psu[rows], weights=w[rows])
        Xf, yf = X.iloc[rows], y[rows]
        fitted, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), Xf, yf,
                                                       design=design, seed=plan.split_seed))
        record = fitted.tuning_
        assert record.inner_k_used == 3 and not record.below_floor
        label = pd.Series([f"{s}|{p}" for s, p in zip(stratum[rows], psu[rows])], index=Xf.index)
        inner = [d for d in drawn if d.kind == "inner"]
        for d in inner:
            assert not set(label[d.train]) & set(label[d.validation])
        expected = _group_kfold(label.to_numpy(), 3, _seed(plan.split_seed, "inner splits"))
        assert [sorted(d.validation) for d in inner] == [sorted(Xf.index[e]) for e in expected]
        # the weighted inner loss, by hand: each candidate a direct fit on the observed rows
        weight = pd.Series(w[rows], index=Xf.index)
        target = pd.Series(yf, index=Xf.index)
        for c, candidate in enumerate(plan.candidates):
            num = den = 0.0
            for d in inner:
                model = HistGradientBoostingRegressor(random_state=0, early_stopping=False,
                                                      **candidate.values)
                model.fit(Xf.loc[d.train], target[d.train])
                residual = target[d.validation].to_numpy() - model.predict(Xf.loc[d.validation])
                num += float(np.sum(weight[d.validation].to_numpy() * residual ** 2))
                den += float(np.sum(weight[d.validation].to_numpy()))
            assert record.losses[c] == pytest.approx(num / den, rel=1e-12, abs=1e-12)


def test_t3_the_oversample_lever_keeps_people_whole_and_never_resamples_a_stopping_row(
        probe, monkeypatch):
    from turbotab.core.methods import levers

    X, y, person = _people(task="binary")
    plan = _plan(probe, "binary", n_plan=1_600, imbalance=True, mode="lighter")
    assert plan.early_stopping and plan.imbalance and len(plan.candidates) == 3
    wrapper = levers.wrap_model(_trees("binary"), {"imbalance": "oversample"}, "binary")
    handed: list[tuple[list, object]] = []
    original = levers.ImbalanceCorrected.fit

    def spied(self, X_, y_, X_val=None, y_val=None, **kw):
        handed.append((list(X_.index), self.cv))
        return original(self, X_, y_, X_val, y_val, **kw)

    monkeypatch.setattr(levers.ImbalanceCorrected, "fit", spied)
    _, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan, "binary", wrapper), X, y,
                                              groups=person, seed=plan.split_seed))
    of = pd.Series(person, index=X.index)
    stopping_sets = []
    for d in drawn:
        if d.stopping is not None:
            assert not set(of[d.train]) & set(of[d.stopping])  # whole persons
            stopping_sets.append(set(d.stopping))
    assert stopping_sets  # every inner split stops: the Sobol candidates stop
    # no stopping row is resampled: every trees fit handed stopping rows trains on none of them
    for train_ids, val_ids in SpyClassifier.rows:
        if val_ids is not None:
            assert not set(train_ids) & set(val_ids)
            assert set(val_ids) in stopping_sets
    # the recalibration splits cover exactly the candidate's own rows, persons whole
    assert handed
    for rows, cv in handed:
        assert isinstance(cv, list) and cv
        tested = np.sort(np.concatenate([np.asarray(b) for _, b in cv]))
        assert np.array_equal(tested, np.arange(len(rows)))
        persons = of[rows].to_numpy()
        for a, b in cv:
            assert not set(persons[np.asarray(a)]) & set(persons[np.asarray(b)])


# ── T4 · no leakage, by perturbation ─────────────────────────────────────────


def test_t4_held_out_rows_never_move_a_folds_choice_and_a_training_row_does(probe):
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=[f"x{i}" for i in range(4)])
    X.index = pd.Index(np.arange(400) + 1, name="row_id")
    y = X["x0"].to_numpy() + X["x1"].to_numpy() ** 2 + rng.normal(size=400)
    fold = rng.permutation(np.arange(400) % 5)
    n_plan, plan_rows, unit = T.plan_size("regression", y, None, folds=5)
    assert (n_plan, plan_rows) == (320, 320)  # by hand: ⌊400·4/5⌋
    plan = T.make_plan(probe, task="regression", loss="mse", n_plan=n_plan, plan_rows=plan_rows,
                       unit=unit, split_seed=3)
    assert len(plan.candidates) == 5
    probe_rows = pd.DataFrame(rng.normal(size=(50, 4)), columns=X.columns)
    k = 2
    train, held = fold != k, fold == k

    def fit_fold(Xa, ya):
        fitted = fit_pipeline(_tuned(probe, plan), Xa.loc[train], ya[train], seed=3)
        return fitted.tuning_, fitted.predict(probe_rows)

    record, predicted = fit_fold(X, y)
    # held-out rows changed: values scaled, outcomes permuted among them (counts kept)
    Xp, yp = X.copy(), y.copy()
    Xp.loc[held, :] = Xp.loc[held, :].to_numpy() * rng.uniform(0.5, 2.0, size=(held.sum(), 4))
    yp[held] = rng.permutation(yp[held])
    assert T.plan_size("regression", yp, None, folds=5) == (n_plan, plan_rows, unit)
    again, again_predicted = fit_fold(Xp, yp)
    assert again.losses == record.losses and again.chosen == record.chosen
    assert again.chosen_params == record.chosen_params
    assert again.chosen_options == record.chosen_options
    assert np.array_equal(again_predicted, predicted)
    # one training row changed: the fold's search sees it
    Xt = X.copy()
    first = X.index[np.flatnonzero(train)[0]]
    Xt.loc[first, "x0"] += 5.0
    moved, _ = fit_fold(Xt, y)
    assert moved.losses != record.losses


def test_t4_a_class_outcomes_plan_reads_its_counts_so_the_perturbation_keeps_them():
    y = np.asarray([0] * 70 + [1] * 30)
    before = T.plan_size("binary", y, None, folds=5)
    assert before == (24, 80, "events")  # by hand: ⌊30·4/5⌋ events, ⌊100·4/5⌋ rows
    assert T.plan_size("binary", np.random.default_rng(0).permutation(y), None, folds=5) == before
    flipped = y.copy()
    flipped[:2] = 1  # 32 events: ⌊32·4/5⌋ = 25
    assert T.plan_size("binary", flipped, None, folds=5) == (25, 80, "events")


# ── T9 · small samples: the floor, counted ────────────────────────────────────


def test_t9_a_fit_below_the_floor_keeps_the_plans_k_and_the_record_counts_it(probe):
    plan = _plan(probe, "binary", n_plan=400, rarest=5)
    assert plan.inner_k == 2 == min(3, 5 // 2)
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    few = np.zeros(60, dtype=int)
    few[[3, 17, 41]] = 1  # 3 units of the rarer class: fewer than 2 × K = 4
    record = fit_pipeline(_tuned(probe, plan, "binary"), X, few, seed=7).tuning_
    assert record.inner_k_used == 2 and record.below_floor
    enough = np.zeros(60, dtype=int)
    enough[[3, 9, 17, 29, 41, 55]] = 1  # 6: at the floor or above it
    record = fit_pipeline(_tuned(probe, plan, "binary"), X, enough, seed=7).tuning_
    assert record.inner_k_used == 2 and not record.below_floor
    # with 3 of the rarest class in the plan's fit, no fit draws inner folds, whatever it holds
    none = _plan(probe, "binary", n_plan=400, rarest=3)
    assert none.inner_k == 0 and len(none.candidates) == 1
    SEEN.clear()
    fitted, drawn = _observed(lambda: fit_pipeline(_tuned(probe, none, "binary"), X, enough,
                                                   seed=7))
    assert [d.kind for d in drawn] == ["refit"] and len(SEEN) == 1
    assert fitted.tuning_.inner_k_used == 0 and fitted.tuning_.losses == (None,)


# ── T13 · reproducibility and pinned replay ───────────────────────────────────


def test_t13_the_same_seed_reproduces_another_draws_other_candidates_and_pinned_replay_refits_once(
        probe):
    X, y, person = _people(80)
    plan = _plan(probe, n_plan=1_600)
    first = fit_pipeline(_tuned(probe, plan), X, y, groups=person, seed=plan.split_seed)
    second = fit_pipeline(_tuned(probe, plan), X, y, groups=person, seed=plan.split_seed)
    a, b = first.tuning_, second.tuning_
    assert a.losses == b.losses and a.chosen == b.chosen and a.chosen_params == b.chosen_params
    assert np.array_equal(first.predict(X), second.predict(X))
    assert a.order == tuple(range(len(plan.candidates)))  # evaluated in the plan's order
    assert all(v is not None for v in a.losses)
    other = _plan(probe, n_plan=1_600, seed=8)
    assert [c.values for c in other.candidates[1:]] != [c.values for c in plan.candidates[1:]]
    # pinned replay: the plain pipeline at the recorded settings, through the plain path at the
    # split's seed, is one fit and the deployed predictions
    SEEN.clear()
    pinned = fit_pipeline(_tuned(probe, plan).at(a.chosen_params), X, y, groups=person,
                          seed=plan.split_seed)
    assert type(pinned) is Pipeline and len(SEEN) == 1
    assert np.array_equal(pinned.predict(X), first.predict(X))
    # the refit is a direct trees fit at the chosen parameters when it does not stop early
    if not a.chosen_params["early_stopping"]:
        direct = HistGradientBoostingRegressor(random_state=0).set_params(**a.chosen_params)
        assert np.array_equal(direct.fit(X, y).predict(X), first.predict(X))


def test_t13_without_stopping_the_refit_is_a_direct_fit_at_the_chosen_parameters(probe):
    X, y, _ = _people(80, 1)
    plan = _plan(probe, n_plan=400)
    fitted = fit_pipeline(_tuned(probe, plan), X, y, seed=plan.split_seed)
    params = dict(fitted.tuning_.chosen_params)
    assert params["early_stopping"] is False
    values = plan.candidates[fitted.tuning_.chosen].values
    direct = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **values)
    assert np.array_equal(direct.fit(X, y).predict(X), fitted.predict(X))


# ── T17 · one plan ────────────────────────────────────────────────────────────


def test_t17_every_outer_fold_resample_and_refit_follows_the_one_plan(probe):
    X, y, _ = _people(150, 2)
    rng = np.random.default_rng(4)
    fold = rng.permutation(np.arange(len(y)) % 5)
    plan = _plan(probe, n_plan=1_600, mode="lighter")
    template = _tuned(probe, plan)
    fits = [(fold != k) for k in range(5)]
    fits += [rng.integers(0, len(y), size=len(y)) for _ in range(2)]  # bootstrap resamples
    fits += [np.ones(len(y), dtype=bool)]  # the final refit

    def run():
        for rows in fits:
            Xf = X.iloc[rows] if rows.dtype != bool else X.loc[rows]
            yf = y[rows]
            fit_pipeline(clone(template), Xf.reset_index(drop=True), yf, seed=plan.split_seed)

    _, drawn = _observed(run)
    assert drawn and all(d.plan == plan for d in drawn)
    assert sum(d.kind == "refit" for d in drawn) == len(fits)
    inner = [d for d in drawn if d.kind == "inner"]
    assert len(inner) == 3 * len(fits)
    assert all(d.stopping is not None for d in inner)  # decided once, by the plan: every fit
    # at 3 of the rarest class no fit draws inner folds, whatever its own count
    Xb, yb, _ = _people(150, 2, task="binary")
    tiny = _plan(probe, "binary", n_plan=400, rarest=3)
    _, drawn = _observed(lambda: [fit_pipeline(_tuned(probe, tiny, "binary"),
                                               Xb.loc[fold != k], yb[fold != k], seed=7)
                                  for k in range(5)])
    assert [d.kind for d in drawn] == ["refit"] * 5 and all(d.plan == tiny for d in drawn)


@pytest.mark.xfail(strict=True, reason="RT-3: a \"Try both\" slot needs one head per option")
def test_t17_at_214_every_fit_carries_the_two_standard_candidates(probe):
    plan = _plan(probe, n_plan=214, options=(("missing", ("native", "fill")),))
    assert [c.standard for c in plan.candidates] == [True, True]
    X, y, _ = _people(70, 1)
    record = fit_pipeline(_tuned(probe, plan), X, y, seed=7).tuning_
    assert len(record.losses) == 2


# ── F13 · one plan decides early stopping across fold sizes ───────────────────


def test_f13_folds_of_9999_and_10001_rows_stop_alike_under_one_plan(probe):
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(10_001, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + rng.normal(size=10_001)
    # scikit-learn's own rule stops one and not the other
    auto = [HistGradientBoostingRegressor(max_iter=15, random_state=0).fit(X.iloc[:n], y[:n])
            for n in (9_999, 10_001)]
    assert [m.do_early_stopping_ for m in auto] == [False, True]
    for plan_rows, stops in ((9_000, False), (12_000, True)):
        plan = _plan(probe, n_plan=400, plan_rows=plan_rows, mode="standard")
        assert plan.standard_stops is stops and len(plan.candidates) == 1
        for n in (9_999, 10_001):
            fitted, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), X.iloc[:n], y[:n],
                                                           seed=plan.split_seed))
            assert fitted[-1].do_early_stopping_ is stops
            [refit] = drawn
            assert (refit.stopping is not None) is stops


# ── cancel ────────────────────────────────────────────────────────────────────


def test_cancel_is_checked_before_every_candidate_fit_and_stops_within_two_seconds(probe):
    X, y, _ = _people(60, 2)
    plan = _plan(probe, n_plan=400)
    with T.cancel_scope(lambda: True), pytest.raises(Cancelled):
        fit_pipeline(_tuned(probe, plan), X, y, seed=7)
    assert SEEN == []  # nothing was fit
    SLOW["seconds"] = 0.25  # 16 fits: about 4 seconds uncancelled
    pressed: dict[str, float] = {}
    timer = threading.Timer(0.6, lambda: pressed.setdefault("at", time.perf_counter()))
    timer.start()
    try:
        with T.cancel_scope(lambda: "at" in pressed), pytest.raises(Cancelled):
            fit_pipeline(_tuned(probe, plan), X, y, seed=7)
        stopped = time.perf_counter()
    finally:
        timer.cancel()
    assert stopped - pressed["at"] < 2.0
    assert len(SEEN) < 16


# ── T6 (partial) · the fits counted, and what the scores are ─────────────────


def _counted(n_rows: int) -> float:
    """The fits made, in fits on the fit's own rows: the rows each model fit was handed (its
    stopping rows included), over the rows of the fit."""
    return sum(a + b for a, b in SEEN) / n_rows


def test_t6_the_fits_counted_equal_the_plans_count(probe, monkeypatch):
    X, y, _ = _people(120, 2)
    n = len(y)
    searched = _plan(probe, n_plan=400)
    SEEN.clear()
    fit_pipeline(_tuned(probe, searched), X, y, seed=7)
    assert _counted(n) == searched.fits() == (3 - 1) * 5 + 1  # w·[(K − 1)·C + 1], by hand
    nothing = _plan(probe, n_plan=400, mode="standard")
    SEEN.clear()
    fit_pipeline(_tuned(probe, nothing), X, y, seed=7)
    assert _counted(n) == nothing.fits() == 1
    # the imbalance correction: each candidate is the wrapped, recalibrated model (w = 5)
    from turbotab.core.methods import levers

    Xb, yb, _ = _people(120, 2, task="binary")
    balanced = _plan(probe, "binary", n_plan=400, imbalance=True)
    wrapper = levers.wrap_model(_trees("binary"), {"imbalance": "weights"}, "binary")
    SEEN.clear()
    fitted = fit_pipeline(_tuned(probe, balanced, "binary", wrapper), Xb, yb, seed=7)
    assert _counted(len(yb)) == balanced.fits() == 5 * ((3 - 1) * 5 + 1)
    # the candidate's settings reached the wrapped trees, the switch stayed the wrapper's own
    step = fitted[-1]
    chosen = balanced.candidates[fitted.tuning_.chosen].values
    assert step.estimator.get_params()["learning_rate"] == chosen["learning_rate"]
    assert step.get_params(deep=False)["early_stopping"] is False


def test_t6_out_of_bag_scores_each_candidate_once_as_a_direct_forest_does(monkeypatch):
    SEEN.clear()
    forest = _family("probe_forest", FOREST,
                     lambda task, *a: SpyForest(n_estimators=20, oob_score=True, random_state=0),
                     tasks=("regression",))
    monkeypatch.setitem(B._REGISTRY, forest.key, forest)
    X, y, _ = _people(150, 1)
    plan = _plan(forest, n_plan=400, out_of_bag=True)
    assert plan.out_of_bag and plan.inner_k == 0 and len(plan.candidates) == 5
    fitted = fit_pipeline(_tuned(forest, plan), X, y, seed=7)
    assert _counted(len(y)) == plan.fits() == 5 + 1
    for c, candidate in enumerate(plan.candidates):
        direct = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=0,
                                       **candidate.values).fit(X, y)
        by_hand = float(np.mean((y - direct.oob_prediction_) ** 2))
        assert fitted.tuning_.losses[c] == pytest.approx(by_hand, rel=1e-12)


def test_t6_a_path_is_one_fit_per_split_and_its_curve_is_direct_ridge_fits(monkeypatch):
    SEEN.clear()
    ridge = _family("probe_ridge", RIDGE, lambda task, *a: SpyRidge(), path=_ridge_path,
                    settings=lambda values, **kw: {"alpha": kw["n_rows"] * values["lambda"]},
                    tasks=("regression",))
    monkeypatch.setitem(B._REGISTRY, ridge.key, ridge)
    X, y, _ = _people(120, 1)
    plan = _plan(ridge, n_plan=400)
    assert plan.kind == "path" and plan.inner_k == 5 and len(plan.candidates) == 8
    fitted, drawn = _observed(lambda: fit_pipeline(_tuned(ridge, plan), X, y, seed=7))
    record = fitted.tuning_
    assert record.n_fits == 5 + 1 and plan.fits() == 5
    lambdas = np.geomspace(10.0, 1e-3, 8)  # the strongest penalty first
    inner = [d for d in drawn if d.kind == "inner"]
    target = pd.Series(y, index=X.index)
    for c, lam in enumerate(lambdas):
        residuals = []
        for d in inner:
            model = Ridge(alpha=len(d.train) * lam).fit(X.loc[d.train], target[d.train])
            residuals.append(target[d.validation].to_numpy() - model.predict(X.loc[d.validation]))
        by_hand = float(np.mean(np.concatenate(residuals) ** 2))
        assert record.losses[c] == pytest.approx(by_hand, rel=1e-8)
    best = int(np.argmin(record.losses))
    assert record.chosen == best and record.path.chosen == (best,)
    assert record.path.at_edge is (best in (0, 7)) and record.path.names == ("lambda",)
    assert fitted[-1].alpha == pytest.approx(len(y) * lambdas[best], rel=1e-12)
    # a path re-tunes in its band refits, as before; a search holds its chosen settings
    from turbotab.core.stages.modeling import pinned_to_full_fit

    assert pinned_to_full_fit(clone(_tuned(ridge, plan)), fitted).search == plan


def test_a_searched_familys_band_refit_holds_the_final_fits_settings(probe):
    from turbotab.core.stages.modeling import pinned_to_full_fit

    X, y, _ = _people(60, 2)
    plan = _plan(probe, n_plan=400)
    fitted = fit_pipeline(_tuned(probe, plan), X, y, seed=7)
    pinned = pinned_to_full_fit(clone(_tuned(probe, plan)), fitted)
    assert type(pinned) is Pipeline
    params = pinned[-1].get_params()
    assert all(params[k] == v for k, v in fitted.tuning_.chosen_params.items())


# ── the boundary: the pipeline, the design's plans and the estimate ──────────


def _spec():
    from turbotab.core.models.pipeline import DesignSpec

    return DesignSpec(predictors=["x0", "x1"], inputs=["x0", "x1"], categorical=[],
                      numeric=["x0", "x1"], energy=None, impute=False,
                      roles={"x0": "exposure", "x1": "covariate"})


def test_a_tuned_family_is_built_from_its_plan_and_refused_without_one(probe):
    from turbotab.core.models.pipeline import build_pipeline

    spec = _spec()
    with pytest.raises(T.MissingPlan):
        build_pipeline(spec, probe, "regression", "prediction", 100, 2)
    plain = build_pipeline(spec, probe, "regression", "prediction", 100, 2, for_timing=True)
    assert type(plain) is Pipeline
    plan = _plan(probe)
    tuned = build_pipeline(replace(spec, plans={probe.key: plan.to_dict()}), probe, "regression",
                           "prediction", 100, 2)
    assert isinstance(tuned, TunedPipeline) and tuned.search == plan
    assert isinstance(clone(tuned), TunedPipeline) and clone(tuned).search == plan
    assert type(tuned[:1]) is TunedPipeline and tuned[:1].search is None  # slicing works
    other = _plan(probe, "binary").to_dict()
    with pytest.raises(T.MissingPlan):
        build_pipeline(replace(spec, plans={probe.key: other}), probe, "regression", "prediction",
                       100, 2)
    # a spec's plans survive its round trip (the design stage keeps it as a dict)
    from turbotab.core.models.pipeline import DesignSpec

    held = replace(spec, plans={probe.key: plan.to_dict()})
    assert DesignSpec.from_dict(held.to_dict()).plans == held.plans


def test_the_design_makes_each_tuned_familys_plan_from_the_headline_split(probe):
    from turbotab.core.graph import Bundle
    from turbotab.core.stages.modeling import design_for, fit_designs, tuning_plans

    rng = np.random.default_rng(2)
    ids = np.arange(400) + 1
    person = np.repeat(np.arange(200), 2)
    stratum = np.repeat([0, 1], 200)  # 2 strata of 3 PSUs each
    psu = np.tile(np.repeat([1, 2, 3], [67, 67, 66]), 2)
    table = pd.DataFrame({"y": rng.normal(size=400), "person": person, "w": rng.uniform(1, 2, 400),
                          "s": stratum, "p": psu}, index=pd.Index(ids, name="row_id"))
    train = (person % 4) != 0  # 300 training rows, whole persons, in every PSU
    fold = np.where(train, person % 5, np.nan)
    assignment = pd.DataFrame({"row_id": ids, "partition": np.where(train, "train", "test"),
                               "fold": fold})

    class Store:
        columns = list(table.columns)

        def materialize(self, columns, row_ids):
            return table.loc[np.asarray(row_ids), list(columns)]

    split = Bundle(data={"folds": 5, "seed": 9, "grouped_by": "person"},
                   frames={"assignment": assignment})
    state = SimpleNamespace(target="y", split=None, survey=None, purpose="prediction")
    ctx = SimpleNamespace(state=state, inputs={"split": split})
    linear = SimpleNamespace(key="plain", tasks=("regression",), tuning=None)
    plans = tuning_plans(ctx, "regression", [probe, linear], store=Store(), train_ids=ids[train])
    assert set(plans) == {probe.key}
    plan = plans[probe.key]
    # by hand: 150 persons in 300 training rows, 5 folds: ⌊150·4/5⌋ persons, ⌊300·4/5⌋ rows
    assert (plan.n_plan, plan.plan_rows, plan.unit, plan.split_seed) == (120, 240, "units", 9)
    assert plan.inner_k == 3 and len(plan.candidates) == 1 and not plan.weighted
    # under the population answer: weighted, and K floored by the fewest PSUs in a training fold
    state.survey = SimpleNamespace(estimand="population", weight="w", strata="s", psu="p",
                                   cycle=None, four_year_weight=None)
    plans = tuning_plans(ctx, "regression", [probe], store=Store(), train_ids=ids[train])
    weighted = plans[probe.key]
    s, p, f = stratum[train], psu[train], fold[train]
    # by hand: each training fold of the split holds all 6 PSUs; evaluation's design-based folds
    # keep a PSU of every stratum in each fold, so 3 folds (3 PSUs a stratum), and each of their
    # training folds holds 2 PSUs of each of the 2 strata: 4, so K = min(3, ⌊4/2⌋) = 2
    assert min(len(set(zip(s[f != k], p[f != k]))) for k in range(5)) == 6
    assert weighted.weighted and weighted.inner_k == 2
    designs = fit_designs(state, Store(), ids[train])
    design = design_for(designs, ids[train][[5, 5, 0]])  # a resample's repeats included
    assert list(design.psu) == [p[5], p[5], p[0]] and list(design.strata) == [s[5], s[5], s[0]]
    assert np.allclose(design.weights, table.loc[ids[train][[5, 5, 0]], "w"].to_numpy())


def test_the_estimate_times_the_center_once_and_counts_the_plans_fits(probe, monkeypatch):
    from turbotab.core.decisions import ProjectState
    from turbotab.core.models import cost

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x0": rng.normal(size=120), "x1": rng.normal(size=120)})
    frame["y"] = frame["x0"] + rng.normal(size=120)

    class Store:
        def materialize(self, columns, ids):
            return frame.loc[np.asarray(ids), list(columns)]

    timed = []
    monkeypatch.setattr(cost, "time_one_fit", lambda pipeline, X, y: timed.append(pipeline) or 0.5)
    state = ProjectState(target="y", task="regression", purpose="prediction",
                         roles={"x0": "exposure", "x1": "covariate"})
    plan = _plan(probe, n_plan=400)
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120), [probe], folds=5,
                             plans={probe.key: plan})
    assert plan.fits() == 11
    assert out[probe.key].seconds == 0.5 * 5 * 11  # one fit × the folds × the plan's fits
    assert out[probe.key].text == "about 30 seconds, most of it tuning"
    [pipeline] = timed
    center = {"learning_rate": float(np.exp((np.log(0.05) + np.log(0.3)) / 2)),
              "max_leaf_nodes": int(np.floor(np.exp((np.log(4) + np.log(17)) / 2))),
              "max_iter": int(np.floor(np.exp((np.log(5) + np.log(26)) / 2)))}
    params = pipeline[-1].get_params()
    assert type(pipeline) is Pipeline and params["early_stopping"] is False
    assert {k: params[k] for k in center} == pytest.approx(center)
    # without a plan the tuned family is timed once at its built settings
    timed.clear()
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120), [probe], folds=5)
    assert out[probe.key].seconds == 2.5 and type(timed[0]) is Pipeline


def test_the_design_holds_the_plan_and_every_fit_of_the_fit_stage_follows_it(tmp_path,
                                                                              monkeypatch):
    """Through the stages: the design stage makes the plan once from the split (its size from
    the training rows, its seed the split's) and builds the family's pipeline from it; every fit
    the fit stage makes of the family, on every fold of the comparison and the final refit,
    carries that plan. Boosted trees are given a probe declaration for the test's duration."""
    from turbotab.core.models import get_family
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    trees = get_family("boosted_trees")
    decl = replace(TREES, standard={"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 100})
    monkeypatch.setattr(trees, "tuning", decl)
    frame = mf.nhanes_like(300, seed=3)
    paths = mf.ingest_frame(frame, tmp_path)
    frame = frame[frame["glucose"].notna()]
    split = mf.split_bundle(frame.index.to_numpy(), seed=4)
    st = mf.state(models=["boosted_trees"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    plan = T.TuningPlan.from_dict(design.objects["spec"]["plans"]["boosted_trees"])
    assert isinstance(design.objects["pipelines"]["boosted_trees"], TunedPipeline)
    # by hand: one row per person, 5 folds: ⌊n_train·4/5⌋ units and rows; the split's seed
    n_train = split.data["n_train"]
    assert (plan.n_plan, plan.plan_rows, plan.split_seed) == (n_train * 4 // 5,
                                                              n_train * 4 // 5, 4)
    drawn: list[T.Drawn] = []
    with T.observing(drawn.append):
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                   paths))
    assert len(drawn) >= 10 * 5 + 1  # the comparison's folds and the final refit, at least
    assert all(d.plan == plan for d in drawn)
    assert fit.objects["fitted"]["boosted_trees"].tuning_.plan == plan
