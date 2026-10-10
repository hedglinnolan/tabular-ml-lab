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
* **T4** perturbation, held-out rows changed (values scaled, outcomes permuted among them, so
  class counts hold): handed to the engine's parts beside a fit's rows (V2X_SEAMS row 1) they move
  no split, stopping set or model; through the store they leave the plan as it was, while a
  training row's class moves n_plan by hand; through the design and fit stages they leave the
  plan and every record bit for bit, while one training row moves exactly the records of the fits
  that train on it (the fixture's own count of the comparison's folds, and the refit);
* **T9** the floor: a fit with fewer than 2 × K of the rarest class keeps K and is counted;
* **T13** replay: the same seed reproduces bit for bit; another split seed draws other candidates;
  pinned replay through the plain path, at the seed of the search's own draws recomputed here
  from the SHA-256 of (split seed, "stopping sets"), reproduces the deployed predictions with one
  fit; each stopping set is ``train_test_split``'s of its persons at that seed;
* **T17** one plan, observed through ``observing()`` in every outer fold, bootstrap resample and
  the final refit;
* **F13** folds of 9,999 and 10,001 rows stop alike under one plan, against scikit-learn's own
  ``"auto"`` rule, which stops one and not the other;
* **T6 (partial)** an instrumented counter of the rows every model fit is handed, over the fit's
  rows, equals ``plan.fits()`` (searched, with the imbalance lever, out of bag, on a path, and
  with nothing to choose); out-of-bag losses equal direct ``RandomForestRegressor`` fits'; a
  path's pooled losses equal direct ``Ridge`` fits' on the observed splits;
* Cancel stops a search within about 2 seconds; the estimate counts the plan's fits, its center
  resolved on the timing sample's outcome and matrix (``settings`` gets both, checked here);
* out of bag, a row no tree left out is left out of the loss, as the hand computation from each
  tree's sample does; under the population answer a unit across PSUs joins them (blocks by hand,
  GroupKFold's identity); a declaration with no early stopping holds the model's switch off,
  where scikit-learn's own ``"auto"`` would stop;
* the boundary: ``pipeline.with_plans`` gives a spec the design's plan (by hand), so the fast
  tier's held-out guarantee builds a tuned family; the shelf and design read every answer a plan
  reads; every fit the stages make passes the split's seed and the design in a cancel scope.
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


def _perturbed(X, y, held, rng):
    """Held-out rows changed: their values scaled, their outcomes permuted among them, so every
    class count (of all rows, and of the rows kept) holds."""
    Xp, yp = X.copy(), np.asarray(y).copy()
    scale = rng.uniform(0.5, 2.0, size=(held.sum(), X.shape[1]))
    Xp.loc[held, :] = Xp.loc[held, :].to_numpy() * scale
    yp[held] = rng.permutation(yp[held])
    return Xp, yp


def test_t4_rows_outside_a_fits_rows_never_reach_its_splits_its_stopping_set_or_its_model(probe):
    """The engine's parts take their rows as an argument (V2X_SEAMS row 1), so the data they are
    handed here holds held-out rows too. Changing those rows (class counts kept) moves no inner
    split, no stopping set and no fitted model; changing one of the fit's own rows does."""
    X, y, person = _people(120, 2, task="binary")
    rng = np.random.default_rng(21)
    held = np.isin(person, rng.choice(120, 24, replace=False))
    rows = np.flatnonzero(~held)
    plan = _plan(probe, "binary", n_plan=1_600)
    assert plan.inner_k == 3 and plan.early_stopping
    probe_rows = pd.DataFrame(rng.normal(size=(40, 4)), columns=X.columns)

    def parts(Xa, ya):
        splits = T.inner_splits_for(plan, Xa, ya, rows, groups=person, seed=11)
        head = T.fit_head(Pipeline([("model", _trees("binary"))]), Xa, ya, rows, groups=person,
                          seed=11, stop_share=0.1, classify=True)
        fitted = T.fit_parts(Pipeline([("model", _trees("binary"))]), Xa, ya, rows,
                             groups=person, seed=11, stop_share=0.1)
        return ([(a.tolist(), b.tolist()) for a, b in splits], head.rows.tolist(),
                head.stopping.tolist(), fitted.predict_proba(probe_rows))

    before = parts(X, y)
    Xp, yp = _perturbed(X, y, held, rng)
    assert np.array_equal(np.bincount(yp), np.bincount(y))
    assert np.array_equal(np.bincount(yp[rows]), np.bincount(y[rows]))
    after = parts(Xp, yp)
    assert after[:3] == before[:3]
    assert np.array_equal(after[3], before[3])
    # every split and the stopping set hold the fit's rows only (positions into the data)
    assert all(set(a) | set(b) <= set(rows.tolist()) for a, b in before[0])
    assert set(before[1]) | set(before[2]) == set(rows.tolist())
    # one of the fit's own rows changed: its model sees it
    Xt = X.copy()
    Xt.iloc[rows[0], 0] += 50.0
    assert not np.array_equal(parts(Xt, y)[3], before[3])


def test_t4_the_plan_reads_the_training_rows_counts_and_never_a_held_out_row(probe):
    """The plan reads the outcome's class counts (an allowed read): through the store, held-out
    rows changed with the counts kept leave it as it was; a training row's class moves n_plan, by
    hand."""
    from turbotab.core.graph import Bundle
    from turbotab.core.stages.modeling import tuning_plans

    rng = np.random.default_rng(6)
    ids = np.arange(500) + 1
    y = (rng.random(500) < 0.3).astype(int)
    train = rng.random(500) < 0.8
    assignment = pd.DataFrame({"row_id": ids, "partition": np.where(train, "train", "holdout"),
                               "fold": np.where(train, np.arange(500) % 5, -1)})
    split = Bundle(data={"folds": 5, "seed": 9, "grouped_by": None},
                   frames={"assignment": assignment})

    def plan_of(outcome):
        table = pd.DataFrame({"y": outcome, "x": rng.normal(size=500)},
                             index=pd.Index(ids, name="row_id"))

        class Store:
            columns = list(table.columns)

            def materialize(self, columns, row_ids):
                return table.loc[np.asarray(row_ids), list(columns)]

        state = SimpleNamespace(target="y", split=None, survey=None, purpose="prediction")
        ctx = SimpleNamespace(state=state, inputs={"split": split})
        return tuning_plans(ctx, "binary", [probe], store=Store(), train_ids=ids[train])[probe.key]

    plan = plan_of(y)
    events = int(min(y[train].sum(), (1 - y[train]).sum()))
    assert plan.n_plan == events * 4 // 5 and plan.unit == "events"  # by hand: ⌊events·4/5⌋
    permuted = y.copy()
    permuted[~train] = rng.permutation(y[~train])
    assert permuted.sum() == y.sum()
    assert plan_of(permuted) == plan
    flipped = y.copy()
    flipped[np.flatnonzero(train & (y == 0))[:5]] = 1  # five more training events
    assert plan_of(flipped).n_plan == (events + 5) * 4 // 5


def test_t4_through_the_stages_held_out_rows_move_nothing_and_a_training_row_moves_its_folds(
        tmp_path, monkeypatch):
    """Through the design and fit stages, every fit of a tuned family recorded: the held-out
    partition changed (values scaled, outcomes permuted among its rows) leaves the plan and every
    record (losses, choice, settings) bit for bit; one training row changed moves exactly the
    records of the fits that train on it: the comparison's folds that hold it (the fixture's own
    count of the fit stage's folds) and the final refit."""
    from turbotab.core.models import get_family
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    monkeypatch.setattr(get_family("boosted_trees"), "tuning", replace(TREES, max_drawn=2))
    seen: list[tuple[frozenset, tuple]] = []
    original = T._tuned_fit

    def recorded(pipe, X, y, **kw):
        out = original(pipe, X, y, **kw)
        r = out.tuning_
        seen.append((frozenset(X.index.tolist()),
                     (r.plan, r.losses, r.chosen, tuple(sorted(r.chosen_params.items())),
                      r.inner_k_used, r.below_floor, r.n_fits)))
        return out

    monkeypatch.setattr(T, "_tuned_fit", recorded)
    frame = mf.nhanes_like(500, seed=3)  # 400 training rows: n_plan 320, so the plan searches
    split = mf.split_bundle(frame.index.to_numpy(), seed=4)
    st = mf.state(models=["boosted_trees"])
    ti = mf.target_info("regression")

    def run(table, folder):
        paths = mf.ingest_frame(table, tmp_path / folder)
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        seen.clear()
        fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        records = dict(seen)
        assert len(records) == len(seen)  # each fit's training rows its own
        return design.objects["spec"]["plans"], records

    plans, records = run(frame, "a")
    plan = T.TuningPlan.from_dict(plans["boosted_trees"])
    assert len(plan.candidates) == 3 and plan.inner_k == 3  # a search, so choices can move
    a = split.frames["assignment"]
    held = a.set_index("row_id").loc[frame.index, "partition"].to_numpy() == "holdout"
    rng = np.random.default_rng(17)
    changed = frame.copy()
    numbers = [c for c in frame.select_dtypes("number").columns
               if c not in ("SEQN", "cycle_begin_year", "glucose")]
    changed.loc[held, numbers] = (changed.loc[held, numbers].to_numpy()
                                  * rng.uniform(0.5, 2.0, size=(held.sum(), len(numbers))))
    changed.loc[held, "glucose"] = rng.permutation(changed.loc[held, "glucose"].to_numpy())
    plans_p, records_p = run(changed, "b")
    assert plans_p == plans and records_p == records
    # one training row changed: the fits that train on it move, and no other
    row = int(a.loc[a["partition"] == "train", "row_id"].iloc[0])
    moved = frame.copy()
    moved.loc[row, "bmi"] += 40.0
    moved.loc[row, "glucose"] += 60.0
    plans_m, records_m = run(moved, "c")
    assert plans_m == plans and set(records_m) == set(records)
    every = frozenset(a.loc[a["partition"] == "train", "row_id"].tolist())
    expected = {s for s in mf.comparison_train_sets(split) | {every} if row in s}
    assert {s for s in records if records_m[s] != records[s]} == expected


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
    # seed of the search's own draws (SHA-256 of the split's seed and "stopping sets"), is one fit
    # and the deployed predictions
    own = _seed(plan.split_seed, "stopping sets")
    SEEN.clear()
    pinned = fit_pipeline(_tuned(probe, plan).at(a.chosen_params), X, y, groups=person, seed=own)
    assert type(pinned) is Pipeline and len(SEEN) == 1
    assert np.array_equal(pinned.predict(X), first.predict(X))
    # a refit that stops early draws its stopping persons at that seed: the replay needs it
    stopping = _plan(probe, n_plan=1_600, plan_rows=12_000, mode="standard")
    assert stopping.standard_stops and len(stopping.candidates) == 1
    deployed = fit_pipeline(_tuned(probe, stopping), X, y, groups=person,
                            seed=stopping.split_seed)
    assert deployed.tuning_.chosen_params["early_stopping"] is True
    at = _tuned(probe, stopping).at(deployed.tuning_.chosen_params)
    replay = fit_pipeline(clone(at), X, y, groups=person, seed=own)
    assert np.array_equal(replay.predict(X), deployed.predict(X))
    raw = fit_pipeline(clone(at), X, y, groups=person, seed=stopping.split_seed)
    assert not np.array_equal(raw.predict(X), deployed.predict(X))
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
        # a row no tree left out has no out-of-bag prediction (scikit-learn writes 0 there)
        kept = np.zeros(len(y), dtype=bool)
        for sample in direct.estimators_samples_:
            kept |= ~np.isin(np.arange(len(y)), sample)
        by_hand = float(np.mean((y[kept] - direct.oob_prediction_[kept]) ** 2))
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


# ── a tuned family wherever a pipeline is built ──────────────────────────────


def test_with_plans_gives_a_spec_the_plan_the_design_would_make(probe):
    from turbotab.core.models.pipeline import build_pipeline, with_plans

    X, y, person = _people(150, 2)
    spec = with_plans(_spec(), [probe], "regression", y, units=person, folds=5, split_seed=9)
    plan = T.TuningPlan.from_dict(spec.plans[probe.key])
    # by hand: 150 persons in 300 rows, 5 folds: ⌊150·4/5⌋ persons, ⌊300·4/5⌋ rows, the seed given
    assert (plan.n_plan, plan.plan_rows, plan.unit, plan.split_seed) == (120, 240, "units", 9)
    assert isinstance(build_pipeline(spec, probe, "regression", "prediction", 300, 2),
                      TunedPipeline)
    # one row per person reads no units, as the design stage does
    alone = with_plans(_spec(), [probe], "regression", y, units=np.arange(300), split_seed=9)
    assert T.TuningPlan.from_dict(alone.plans[probe.key]).n_plan == 240
    # an untuned family adds no plan
    linear = SimpleNamespace(key="plain", tasks=("regression",), tuning=None)
    assert with_plans(_spec(), [linear], "regression", y).plans is None


def test_the_held_out_guarantee_builds_a_tuned_family_from_its_plan(monkeypatch):
    """The fast tier's held-out guarantee builds every registered family; a tuned one is built
    from a plan there too, so it holds (held-out rows through the training-fitted steps, those
    steps the training rows' own) for boosted trees given a probe declaration, under the plain
    design, the imbalance lever and a selection step."""
    from turbotab.core.models import get_family
    from turbotab.core.tests import test_heldout_guarantee as H

    monkeypatch.setattr(get_family("boosted_trees"), "tuning", TREES)
    frame = H._table()
    for task, option in (("regression", "as_given"), ("binary", "lever_imbalance_oversample"),
                         ("regression", "select_univariable")):
        H.test_held_out_predictions_are_the_training_fitted_steps_applied_by_hand(
            frame, "boosted_trees", task, option)
    final, *_ = H._fitted("boosted_trees", "regression", H.OPTIONS["as_given"](), frame)
    assert isinstance(final, TunedPipeline) and final.tuning_.plan.family == "boosted_trees"


def test_the_stages_that_make_plans_read_every_answer_a_plan_reads():
    """The shelf and the design make the plans (the shelf to count their fits), so each reads
    every answer a plan reads: the levers and the selection step (out of bag or not, the
    imbalance correction), the survey answer (PSUs, the weighted loss) and the purpose."""
    from turbotab.core.stages import build_graph

    stages = {s.name: s for s in build_graph().stages()}
    for name in ("shelf", "design"):
        assert {"levers", "selection", "survey", "purpose"} <= set(stages[name].reads), name


def test_every_stage_fit_takes_the_splits_seed_and_design_inside_a_cancel_scope():
    """F15: every fit the fit, substitution and evaluation stages make passes the split's seed and
    the design, inside a cancel scope, so a tuned family searches as planned wherever it is
    refit and a pressed Cancel stops it."""
    import ast
    import inspect

    from turbotab.core.stages import evaluation, modeling

    def calls(node, scoped):
        if isinstance(node, ast.With):
            scoped = scoped or any(isinstance(i.context_expr, ast.Call)
                                   and getattr(i.context_expr.func, "id", None) == "cancel_scope"
                                   for i in node.items)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "fit_pipeline":
            yield node, scoped
        for child in ast.iter_child_nodes(node):
            yield from calls(child, scoped)

    found = 0
    for module in (modeling, evaluation):
        for call, scoped in calls(ast.parse(inspect.getsource(module)), False):
            found += 1
            where = f"{module.__name__}:{call.lineno}"
            assert {"seed", "design"} <= {k.arg for k in call.keywords}, where
            assert scoped, where
    assert found >= 5


# ── early stopping: the plan alone decides ───────────────────────────────────


def test_a_tuned_fit_stops_early_only_where_its_plan_says(monkeypatch):
    """A family whose declaration names no early stopping, around trees that keep scikit-learn's
    "auto" (which stops above 10,000 rows on a split it draws by position: F11, F13), never stops
    early in a tuned fit: the switch is held off in every fit and recorded. A model with no
    switch (the forest) records none."""
    SEEN.clear()
    decl = TuningDecl("search", dimensions=TREES.dimensions[:2],
                      standard={"learning_rate": 0.1, "max_leaf_nodes": 8},
                      space_version="probe_plain/1", max_drawn=2)
    family = _family("probe_plain", decl, _trees, tasks=("regression",))
    monkeypatch.setitem(B._REGISTRY, family.key, family)
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(10_001, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + rng.normal(size=10_001)
    assert HistGradientBoostingRegressor(max_iter=15, random_state=0).fit(X, y).do_early_stopping_
    plan = _plan(family, n_plan=12_000, plan_rows=12_000, mode="standard")
    fitted, drawn = _observed(lambda: fit_pipeline(_tuned(family, plan), X, y, seed=7))
    assert fitted[-1].do_early_stopping_ is False
    assert fitted.tuning_.chosen_params["early_stopping"] is False
    assert [d.stopping for d in drawn] == [None] and SEEN == [(10_001, 0)]
    forest = _family("probe_forest", FOREST,
                     lambda task, *a: RandomForestRegressor(n_estimators=5, random_state=0),
                     tasks=("regression",))
    monkeypatch.setitem(B._REGISTRY, forest.key, forest)
    Xs, ys, _ = _people(40, 1)
    record = fit_pipeline(_tuned(forest, _plan(forest, n_plan=100)), Xs, ys, seed=7).tuning_
    assert "early_stopping" not in record.chosen_params


# ── out of bag: only rows some tree left out ─────────────────────────────────


def test_t6_out_of_bag_scores_only_rows_some_tree_left_out(monkeypatch):
    """scikit-learn predicts 0 for a row no tree left out (it counts such a row once), so with few
    trees the out-of-bag loss leaves those rows out, as the hand computation does: each tree
    predicts the rows outside its sample (``estimators_samples_``), averaged per row over the
    trees that left it out."""
    import warnings

    forest = _family("probe_forest", FOREST,
                     lambda task, *a: RandomForestRegressor(n_estimators=3, oob_score=True,
                                                            random_state=0),
                     tasks=("regression",))
    monkeypatch.setitem(B._REGISTRY, forest.key, forest)
    X, y, _ = _people(60, 1)
    plan = _plan(forest, n_plan=400, out_of_bag=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Some inputs do not have OOB scores"
        fitted = fit_pipeline(_tuned(forest, plan), X, y, seed=7)
        for c, candidate in enumerate(plan.candidates):
            direct = RandomForestRegressor(n_estimators=3, random_state=0,
                                           **candidate.values).fit(X, y)
            Xa = X.to_numpy(dtype=np.float32)
            total, count = np.zeros(len(y)), np.zeros(len(y))
            for tree, sample in zip(direct.estimators_, direct.estimators_samples_):
                out = np.setdiff1d(np.arange(len(y)), sample)
                total[out] += tree.predict(Xa[out])
                count[out] += 1
            kept = count > 0
            assert 0 < (~kept).sum() < len(y)  # some row was never left out
            by_hand = float(np.mean((y[kept] - total[kept] / count[kept]) ** 2))
            assert fitted.tuning_.losses[c] == pytest.approx(by_hand, rel=1e-12)


# ── the population answer: a unit across PSUs ────────────────────────────────


def test_t3_under_the_population_answer_a_unit_across_psus_joins_them(probe):
    """A unit not nested in one PSU joins the PSUs it sits in, so neither a unit nor a PSU is ever
    split between an inner split's two sides. By hand: persons straddle PSUs 0|1 and 0|2, 0|2 and
    1|1, and 2|1 and 2|2, so those PSUs fall into two blocks, each named by its first PSU as text;
    the inner splits are GroupKFold's over the blocks at the derived seed."""
    X, y, stratum, psu, w = _nhanes_shaped()
    person = np.arange(300) // 2  # two rows each, nested in the PSUs (10 rows a PSU)
    for a, b in ((9, 10), (19, 20), (49, 50)):
        person[b] = person[a]  # rows 9|10, 19|20 and 49|50 sit in neighboring PSUs
    plan = _plan(probe, n_plan=400, psus=30, weighted=True)
    assert plan.inner_k == 3
    design = T.FitDesign(strata=stratum, psu=psu, weights=w)
    fitted, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), X, y, groups=person,
                                                   design=design, seed=plan.split_seed))
    label = np.asarray([f"{s}|{p}" for s, p in zip(stratum, psu)], dtype=object)
    block = label.copy()
    block[np.isin(label, ["0|2", "1|1"])] = "0|1"
    block[label == "2|2"] = "2|1"
    of_person = pd.Series(person, index=X.index)
    of_psu = pd.Series(label, index=X.index)
    inner = [d for d in drawn if d.kind == "inner"]
    assert len(inner) == 3
    for d in inner:
        assert not set(of_person[d.train]) & set(of_person[d.validation])
        assert not set(of_psu[d.train]) & set(of_psu[d.validation])
    expected = _group_kfold(block, 3, _seed(plan.split_seed, "inner splits"))
    assert [sorted(d.validation) for d in inner] == [sorted(X.index[e]) for e in expected]


# ── seeds: every draw of a search from the split's, by SHA-256 ───────────────


def test_t13_a_searchs_stopping_sets_and_nested_splits_draw_at_a_derived_seed(probe, monkeypatch):
    """RECIPES §4.7: inner splits, and stopping sets likewise, are seeded from the SHA-256 of the
    split's seed and their name, never the split's seed itself. Each inner split's stopping
    persons are scikit-learn's ``train_test_split`` of its persons, sorted as text, at
    (split seed, "stopping sets"); every head and model fit of the search draws there."""
    from sklearn.model_selection import train_test_split

    X, y, person = _people(80)
    plan = _plan(probe, n_plan=1_600)
    own = _seed(plan.split_seed, "stopping sets")
    assert own != plan.split_seed
    seeds: list[int] = []
    for name in ("fit_head", "fit_model"):
        original = getattr(T, name)

        def spied(*a, _original=original, **kw):
            seeds.append(kw["seed"])
            return _original(*a, **kw)

        monkeypatch.setattr(T, name, spied)
    _, drawn = _observed(lambda: fit_pipeline(_tuned(probe, plan), X, y, groups=person,
                                              seed=plan.split_seed))
    assert seeds and set(seeds) == {own}
    of = pd.Series([str(p) for p in person], index=X.index)
    for d in (d for d in drawn if d.kind == "inner"):
        names = np.unique(of[np.concatenate([d.train, d.stopping])].to_numpy())
        _, chosen = train_test_split(names, test_size=max(1, int(round(0.1 * len(names)))),
                                     random_state=own)
        assert set(of[d.stopping]) == set(chosen)


# ── the time estimate resolves the center on the timing rows ─────────────────


def test_the_center_is_resolved_on_the_timing_rows_outcome_and_matrix(probe, monkeypatch):
    """``settings`` is always called with the fit's outcome and model matrix: ``at`` a candidate
    takes the rows it will be fit on, and the estimate passes its timing sample, so a family
    whose settings read the outcome (XGBoost's hessian) is timed at its center."""
    from turbotab.core.decisions import ProjectState
    from turbotab.core.models import cost

    def settings(values, *, task, n_units, n_rows, y, Z, plan):
        assert y is not None and Z is not None and len(y) == len(Z) == n_rows
        return {**values, "l2_regularization": float(np.var(np.asarray(y)))}

    family = _family("probe_set", TREES, _trees, settings=settings)
    monkeypatch.setitem(B._REGISTRY, family.key, family)
    plan = _plan(family, n_plan=400)
    pipe = _tuned(family, plan)
    with pytest.raises(TypeError):
        pipe.at(T.center(plan))
    X, y, person = _people(60, 2)
    pinned = pipe.at(T.center(plan), X=X, y=y, groups=person)
    assert type(pinned) is Pipeline
    assert pinned[-1].get_params()["l2_regularization"] == pytest.approx(float(np.var(y)))
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
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120), [family], folds=5,
                             plans={family.key: plan})
    assert out[family.key].seconds == 0.5 * 5 * plan.fits()
    [timed_pipeline] = timed
    by_hand = float(np.var(frame["y"].to_numpy()))  # the timing sample is every row here
    assert timed_pipeline[-1].get_params()["l2_regularization"] == pytest.approx(by_hand)
