"""WAVE_C6A_PLAN §7 ruling 14: a searched leaf's cap counts units, not rows, when the units are
known; the tuning record states the thread count the fits ran at."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models.base import get_family
from turbotab.core.models.boosted_trees import leaf_cap
from turbotab.core.models.forest import leaf_rows
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.tuning import TunedPipeline, estimator_params, make_plan


def _leaf(family, task, plan, *, n_units, n_rows, asked=10_000):
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
    family = get_family("boosted_trees")
    assert _leaf(family, "regression", plan, n_units=90, n_rows=900) == 90 // 20 == 4  # not 45


@pytest.mark.parametrize("task,unit", [("binary", "events"), ("multiclass", "rarest_class")])
def test_a_class_outcome_with_repeated_rows_caps_at_a_twentieth_of_the_people_not_the_rows(
        task, unit):
    # 30 events among 90 people: the plan counts 30, which is not what a leaf holds; its rows are
    # 900, and the fit's rows per person are 10, so its people are 900 * 100 // 1000 = 90
    plan = _plan(task, n_plan=30, plan_rows=900, unit=unit)
    family = get_family("boosted_trees")
    assert _leaf(family, task, plan, n_units=100, n_rows=1000) == 900 * 100 // 1000 // 20 == 4
    # a smaller inner fit of the same people-per-row ratio keeps the cap
    assert _leaf(family, task, plan, n_units=80, n_rows=800) == 4
    # the old cap counted rows
    assert 900 // 20 == 45


@pytest.mark.parametrize("task,unit", [("binary", "events"), ("multiclass", "rarest_class")])
def test_a_class_outcome_with_one_row_per_unit_still_caps_at_a_twentieth_of_the_plans_rows(
        task, unit):
    plan = _plan(task, n_plan=60, plan_rows=400, unit=unit)
    family = get_family("boosted_trees")
    assert _leaf(family, task, plan, n_units=500, n_rows=500) == 400 // 20 == 20


def test_without_a_plan_the_cap_is_a_twentieth_of_the_fits_own_units():
    assert leaf_cap(n_units=100, n_rows=1000) == 5
    assert leaf_cap(n_units=100) == 5
    assert leaf_cap(n_units=7, n_rows=70) == 1


def test_the_forests_leaf_is_a_share_of_the_rows_so_it_already_counts_whole_units():
    # a share of the people is the same share of the rows: one person of 100 holds 10 of 1,000 rows
    assert leaf_rows(0.01, n_rows=1000) == 10
    assert leaf_rows(0.1, n_rows=1000) == 100


# ── the record's threads ──────────────────────────────────────────────────────


def test_the_record_states_the_threads_the_fits_ran_at_not_the_process_s():
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = (X["a"] + rng.normal(size=400)).to_numpy()
    family = get_family("boosted_trees")
    plan = _plan("regression", n_plan=400, plan_rows=400, unit="units", threads=1)
    pipe = TunedPipeline([("model", family.build("regression", "prediction", 0, 0))], search=plan)
    with threadpool_limits(limits=3, user_api="openmp"):
        fitted = fit_pipeline(pipe, X, y, seed=plan.split_seed)
    record = fitted.tuning_
    assert plan.threads == 1
    assert record.threads["plan"] == plan.threads
    assert set(record.threads.values()) == {plan.threads}
    assert record.chosen_params["n_threads"] == plan.threads
