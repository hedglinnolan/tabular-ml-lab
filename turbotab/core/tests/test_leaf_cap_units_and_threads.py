"""WAVE_C6A_PLAN §7 ruling 14: a searched leaf's cap counts the plan's units when the plan knows
them, its rows otherwise, and is the same in every fit the plan makes; the tuning record states
the thread counts the fits ran at."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import tuning
from turbotab.core.models.base import get_family
from turbotab.core.models.boosted_trees import leaf_cap
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.tuning import TunedPipeline, estimator_params, make_plan


def _leaf(task, plan, *, n_units, n_rows, asked=10_000):
    family = get_family("boosted_trees")
    values = {**plan.candidates[0].values, "min_samples_leaf": asked}
    return estimator_params(family, task, values, n_units=n_units, n_rows=n_rows,
                            plan=plan)["min_samples_leaf"]


def _plan(task, *, n_plan, plan_rows, unit, **kw):
    return make_plan(get_family("boosted_trees"), task=task,
                     loss="mse" if task == "regression" else "log_loss", n_plan=n_plan,
                     plan_rows=plan_rows, unit=unit, split_seed=7, **kw)


# 100 people, 10 rows each: 1,000 rows. The plan's outer training fold holds 900 rows, 90 people.


def test_a_measured_outcome_with_repeated_rows_caps_at_a_twentieth_of_the_plans_people():
    plan = _plan("regression", n_plan=90, plan_rows=900, unit="units")
    assert _leaf("regression", plan, n_units=90, n_rows=900) == 90 // 20 == 4  # not 900 // 20


# The old switch, copied as the reference: the plan's units when it counts units, else its rows.
def _reference_cap(plan):
    return max(1, (int(plan.n_plan) if plan.unit == "units" else int(plan.plan_rows)) // 20)


# (units, rows) of fits whose people hold unequal numbers of rows: the inner folds, the refit and
# another outer fold of one plan (the verifier's probe)
_FITS = [(200, 4000), (160, 3000), (160, 3400), (200, 4600)]


@pytest.mark.parametrize("task,unit,n_plan", [("regression", "units", 200),
                                              ("binary", "events", 60),
                                              ("multiclass", "rarest_class", 40)])
def test_the_cap_is_the_same_in_every_fit_the_plan_makes(task, unit, n_plan):
    plan = _plan(task, n_plan=n_plan, plan_rows=4000, unit=unit)
    caps = {_leaf(task, plan, n_units=u, n_rows=r) for u, r in _FITS}
    assert caps == {_reference_cap(plan)}
    assert caps == ({10} if unit == "units" else {200})


@pytest.mark.parametrize("task,unit", [("binary", "events"), ("multiclass", "rarest_class")])
def test_a_class_outcome_caps_at_a_twentieth_of_the_plans_rows_since_it_counts_no_units(
        task, unit):
    # 30 events among 90 people: the plan counts 30, which is not what a leaf holds, and records
    # no count of people, so its 900 rows stand: 45, not an estimate of its people from a fit
    plan = _plan(task, n_plan=30, plan_rows=900, unit=unit)
    assert _leaf(task, plan, n_units=100, n_rows=1000) == 900 // 20 == 45
    assert _leaf(task, plan, n_units=80, n_rows=700) == 45
    one_per_row = _plan(task, n_plan=60, plan_rows=400, unit=unit)
    assert _leaf(task, one_per_row, n_units=500, n_rows=500) == 400 // 20 == 20


def test_without_a_plan_the_cap_is_a_twentieth_of_the_fits_own_units():
    assert leaf_cap(n_units=100) == 5
    assert leaf_cap(n_units=7) == 1


# ── the record's threads ──────────────────────────────────────────────────────


def test_the_record_states_openmp_at_the_plans_threads_not_the_process_s():
    from threadpoolctl import threadpool_info, threadpool_limits

    if not any(p.get("user_api") == "openmp" for p in threadpool_info()):
        pytest.skip("no OpenMP pool is loaded in this process")
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = (X["a"] + rng.normal(size=400)).to_numpy()
    family = get_family("boosted_trees")
    plan = _plan("regression", n_plan=400, plan_rows=400, unit="units", threads=1)
    pipe = TunedPipeline([("model", family.build("regression", "prediction", 0, 0))], search=plan)
    with threadpool_limits(limits=3, user_api="openmp"):
        fitted = fit_pipeline(pipe, X, y, seed=plan.split_seed)
    record = fitted.tuning_
    assert record.threads["plan"] == plan.threads == 1
    assert record.threads["openmp"] == plan.threads  # the process held 3; the fits ran at 1
    assert record.chosen_params["n_threads"] == plan.threads


def test_the_record_keeps_the_blas_pools_at_the_process_count(monkeypatch):
    # no family pins BLAS: a linear or elastic-net fit's linear algebra runs at the process's count
    import threadpoolctl

    pools = [{"user_api": "openmp", "internal_api": "openmp", "num_threads": 10},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 4}]
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: pools)
    plan = _plan("regression", n_plan=400, plan_rows=400, unit="units", threads=2)
    assert tuning._threads(plan) == {"plan": 2, "openmp": 2, "openblas": 4}
