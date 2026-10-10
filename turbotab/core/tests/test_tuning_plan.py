"""RT-1a · the tuning plan, fixed once (RECIPES_AND_TUNING §4.1, §4.2, §4.6, §4.7; WAVE_C6A_PLAN §2).

Every expected value here is computed independently of ``models/tuning.py``:

* **T2(b)** the candidates: the standard settings first (one per "Try both" option, in order), then
  scipy's scrambled Sobol sample drawn with ``qmc.Sobol(d, scramble=True, rng=seed)``, mapped
  through each documented scale by formulas written out below, with the seed recomputed from the
  SHA-256 of the canonical JSON of (split seed, family key, space version);
* **T2(e)** ties: hand-built losses, equal after rounding to 10⁻⁹ of the smallest;
* **T9 and T17** plan arithmetic: the effective size, the plan size, the candidate counts, the
  inner folds' floor and the early-stopping switches, each worked by hand from the rules in
  RECIPES §4.2 and §4.6, including the worked example (17,000 participants, 37 fits per outer fit);
* the pooled loss, by hand.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import qmc

from turbotab.core.models import tuning as T
from turbotab.core.models.tuning import Dimension, TuningDecl


def _dim(name, low, high, scale, **more):
    return Dimension(name, f"plain words for {name}", f"term for {name}", low, high, scale,
                     source="Friedman 2001", **more)


# The boosted trees' search space as the plan sketches it (WAVE_C6A_PLAN §2), under a probe's key.
SEARCH = TuningDecl(
    "search",
    dimensions=(
        _dim("learning_rate", 0.01, 0.3, "log"),
        _dim("max_leaf_nodes", 4, 128, "log_int"),
        _dim("min_samples_leaf", 2, 200, "log_int"),
        _dim("l2_regularization", 1e-3, 10, "log"),
        _dim("max_features", 0.3, 1.0, "linear"),
        _dim("max_iter", 25, 500, "log_int", active="without_early_stopping"),
    ),
    standard={"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
              "l2_regularization": 0.0, "max_features": 1.0, "max_iter": 100},
    standard_source="scikit-learn's defaults",
    early_stopping={"param": "max_iter", "rounds": 1000, "patience_param": "n_iter_no_change",
                    "patience": 20, "share": 0.1},
    space_version="probe_trees/1",
    structural=("loss",),
)
# Every other scale: an integer, a choice and a share of the plan's units.
SCALES = TuningDecl(
    "search",
    dimensions=(
        _dim("max_depth", 2, 10, "int"),
        _dim("criterion", 0, 0, "choice", choices=("squared_error", "friedman_mse", "poisson")),
        _dim("min_samples_leaf", 0, 0.1, "share_of_units"),
        _dim("subsample", 0.5, 1.0, "linear"),
    ),
    standard={"max_depth": 6, "criterion": "squared_error", "min_samples_leaf": "default",
              "subsample": 1.0},
    space_version="probe_scales/3",
    max_drawn=8,
)
RIDGE = TuningDecl("path", dimensions=(_dim("lambda", 1e-5, 1e2, "log", points=50),),
                   space_version="probe_ridge/1")
HUBER = TuningDecl("none", by_hand=(_dim("t", 1, 3, "linear"),), standard={"t": 1.345},
                   space_version="probe_huber/1")
TRY_BOTH = (("missing", ("native", "fill")),)


def _family(decl, key="probe_trees", tasks=("regression", "binary", "multiclass")):
    return SimpleNamespace(key=key, label="Probe", tasks=tasks, tuning=decl, defaults_version="2",
                           settings=None)


def _seed(*parts):
    """The documented derivation, rewritten: SHA-256 of the canonical JSON (keys sorted, no spaces,
    ASCII), its first four bytes read big-endian."""
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _mapped(dim, u, n_plan):
    """Each documented scale, rewritten from RECIPES §4.1 and the plan's ruled reading of
    ``share_of_units`` (log-uniform from one unit of the plan, 1/n_plan, to ``high``)."""
    lo, hi = dim.low, dim.high
    if dim.scale == "linear":
        return lo + u * (hi - lo)
    if dim.scale == "log":
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if dim.scale == "int":  # every integer in [lo, hi] equally likely
        return min(int(hi), int(math.floor(lo + u * (hi - lo + 1))))
    if dim.scale == "log_int":  # integer k with weight log((k + 1) / k)
        return min(int(hi), int(math.floor(math.exp(
            math.log(lo) + u * (math.log(hi + 1) - math.log(lo))))))
    if dim.scale == "choice":
        return dim.choices[min(int(u * len(dim.choices)), len(dim.choices) - 1)]
    assert dim.scale == "share_of_units"
    one = 1.0 / n_plan
    return math.exp(math.log(one) + u * (math.log(hi) - math.log(one)))


def _same(got, want):
    if isinstance(want, float):
        assert isinstance(got, float) and got == pytest.approx(want, rel=1e-12, abs=0)
    else:
        assert got == want and type(got) is type(want)


# ── T2(b): the candidates ────────────────────────────────────────────────────


def test_t2b_the_standard_candidates_come_first_then_scipys_sobol_through_each_scale():
    plan = T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=1200,
                       plan_rows=1500, unit="units", split_seed=7, options=TRY_BOTH)
    seed = _seed(7, "probe_trees", "probe_trees/1")
    assert plan.seed == seed and plan.strategy == "sobol"
    standard = dict(SEARCH.standard)
    assert [(c.index, dict(c.values), dict(c.options), c.standard)
            for c in plan.candidates[:2]] == [(0, standard, {"missing": "native"}, True),
                                              (1, standard, {"missing": "fill"}, True)]
    # 1,200 units: 16 Sobol candidates; early stopping is off, so the number of trees is searched.
    dims = SEARCH.dimensions
    U = qmc.Sobol(len(dims) + 1, scramble=True, rng=seed).random_base2(4)
    sobol = plan.candidates[2:]
    assert len(sobol) == 16 and [c.index for c in sobol] == list(range(2, 18))
    for row, cand in zip(U, sobol):
        assert not cand.standard and set(cand.values) == {d.name for d in dims}
        for dim, u in zip(dims, row):
            _same(cand.values[dim.name], _mapped(dim, float(u), 1200))
        assert dict(cand.options) == {"missing": ("native", "fill")[min(int(row[-1] * 2), 1)]}


def test_t2b_under_early_stopping_the_number_of_trees_leaves_the_sample():
    plan = T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=1600,
                       plan_rows=2000, unit="units", split_seed=7)
    assert plan.early_stopping and not plan.standard_stops
    active = [d for d in SEARCH.dimensions if d.name != "max_iter"]
    U = qmc.Sobol(len(active), scramble=True, rng=_seed(7, "probe_trees", "probe_trees/1")
                  ).random_base2(4)
    assert [dict(c.values) for c in plan.candidates[:1]] == [dict(SEARCH.standard)]
    for row, cand in zip(U, plan.candidates[1:], strict=True):
        assert "max_iter" not in cand.values
        for dim, u in zip(active, row):
            _same(cand.values[dim.name], _mapped(dim, float(u), 1600))


def test_t2b_every_scale_maps_as_documented():
    plan = T.make_plan(_family(SCALES, key="probe_scales"), task="regression", loss="mse",
                       n_plan=2500, plan_rows=3000, unit="units", split_seed=11)
    U = qmc.Sobol(4, scramble=True, rng=_seed(11, "probe_scales", "probe_scales/3")).random_base2(3)
    # max_drawn caps the sample at 8, as the forest's least tunable space is capped (RECIPES §4.2).
    assert len(plan.candidates) == 1 + 8
    for row, cand in zip(U, plan.candidates[1:], strict=True):
        for dim, u in zip(SCALES.dimensions, row):
            _same(cand.values[dim.name], _mapped(dim, float(u), 2500))
    # the ends of each scale, by hand
    leaf = SCALES.dimensions[2]
    assert T.map_unit(leaf, 0.0, n_plan=400) == pytest.approx(1 / 400)
    assert T.map_unit(leaf, 1 - 1e-15, n_plan=400) == pytest.approx(0.1)
    assert T.map_unit(SEARCH.dimensions[1], 0.0, n_plan=400) == 4
    assert T.map_unit(SEARCH.dimensions[1], 1 - 1e-15, n_plan=400) == 128
    assert T.map_unit(SCALES.dimensions[0], 0.999999, n_plan=400) == 10
    assert T.map_unit(SCALES.dimensions[1], 0.5, n_plan=400) == "friedman_mse"


def test_t2b_the_seed_is_sha256_and_another_split_seed_draws_other_candidates():
    assert T.derive_seed(7, "probe_trees", "probe_trees/1") == _seed(7, "probe_trees",
                                                                   "probe_trees/1")
    assert T.derive_seed({"b": 1, "a": [2, 3]}) == _seed({"a": [2, 3], "b": 1})
    assert T.derive_seed(np.int64(7), "x") == _seed(7, "x")

    def values(split_seed):
        plan = T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=1200,
                           plan_rows=1500, unit="units", split_seed=split_seed)
        return [dict(c.values) for c in plan.candidates]

    assert values(7) == values(7)
    assert values(7)[1:] != values(8)[1:]
    assert values(7)[0] == values(8)[0] == dict(SEARCH.standard)


def test_t2b_values_set_by_hand_are_held_in_every_candidate_and_leave_the_sample():
    plan = T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=1200,
                       plan_rows=1500, unit="units", split_seed=7,
                       manual={"learning_rate": 0.05})
    free = [d for d in SEARCH.dimensions if d.name != "learning_rate"]
    U = qmc.Sobol(len(free), scramble=True, rng=_seed(7, "probe_trees", "probe_trees/1")
                  ).random_base2(4)
    assert dict(plan.candidates[0].values) == {**SEARCH.standard, "learning_rate": 0.05}
    for row, cand in zip(U, plan.candidates[1:], strict=True):
        assert cand.values["learning_rate"] == 0.05
        for dim, u in zip(free, row):
            _same(cand.values[dim.name], _mapped(dim, float(u), 1200))
    with pytest.raises(ValueError, match="'depth', which probe_trees does not tune"):
        T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=1200,
                    plan_rows=1500, unit="units", split_seed=7, manual={"depth": 3})


def test_a_path_familys_candidates_are_its_grid_from_the_strongest_penalty():
    plan = T.make_plan(_family(RIDGE, key="probe_ridge"), task="regression", loss="mse",
                       n_plan=340, plan_rows=340, unit="units", split_seed=0)
    grid = [10 ** (2 - 7 * i / 49) for i in range(50)]  # 50 points from 10² down to 10⁻⁵
    assert [c.values["lambda"] for c in plan.candidates] == pytest.approx(grid, rel=1e-12)
    assert [c.index for c in plan.candidates] == list(range(50))
    assert list(T.path_grid(RIDGE)) == ["lambda"]


# ── T2(e): ties go to the lower index ────────────────────────────────────────


def test_t2e_losses_equal_after_rounding_go_to_the_lower_index():
    from turbotab.core.models import elastic_net

    # rounded to multiples of 10⁻⁹ × 2.0: 2.0 → 10⁹, 2.0·(1 + 3·10⁻¹⁰) → 10⁹ + 0.3 → 10⁹
    assert T.choose([3.0, 2.0 * (1 + 3e-10), 2.0]) == 1
    assert T.choose([2.0, 2.0 * (1 + 3e-10)]) == 0
    assert T.choose([2.0 * (1 + 3e-9), 2.0]) == 1  # 10⁹ + 3 against 10⁹: a real difference
    assert T.choose([None, 5.0, float("nan"), 5.0]) == 1
    assert T.choose([float("inf"), float("nan")]) == 0
    assert T.LOSS_PRECISION == 1e-9
    assert elastic_net.lowest_rounded is T.choose  # moved, and re-exported where it was


# ── T9 and T17: plan arithmetic ──────────────────────────────────────────────


def test_t9_the_effective_size_counts_units_or_the_rarest_class_by_each_units_own_class():
    y = np.r_[np.ones(150), np.zeros(4850)]
    assert T.effective_size("binary", y) == (150, "events")
    assert T.effective_size("regression", np.arange(9.0), units=np.repeat([1, 2, 3], 3)) == (
        3, "units")
    # Each unit counts once, by its most common class; a tie goes to the smaller label (pandas'
    # ``mode``, as the stratified inner splits break it): A → 1, B → 0, C (0, 1) → 0, D → 1,
    # E → 0. Class 0 holds 3 units, class 1 holds 2.
    units = np.array(list("AAABBCCDE"))
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0])
    assert T.effective_size("binary", y, units=units) == (2, "events")
    assert T.effective_size("multiclass", np.array([0, 1, 2, 2, 1, 2]), units=None) == (
        1, "rarest_class")
    # a class that no unit holds most often counts zero units
    assert T.effective_size("binary", np.array([0, 0, 1, 0]), units=np.array([1, 1, 1, 2])) == (
        0, "events")


def test_t9_t17_the_plan_size_is_one_outer_training_folds():
    # 5,000 units at 3% prevalence with 5 folds: 150 events, 120 in a training fold.
    y = np.r_[np.ones(150), np.zeros(4850)]
    assert T.plan_size("binary", y, None, folds=5) == (120, 4000, "events")
    # T17: 425 units with 5 folds: 340 in a training fold
    assert T.plan_size("regression", np.zeros(425), None, folds=5) == (340, 340, "units")
    # the worked example: 17,000 participants, folds of 5: n_plan 13,600
    assert T.plan_size("regression", np.zeros(17_000), None, folds=5) == (13_600, 13_600, "units")
    # rows that repeat: 3 rows per person, 100 people
    assert T.plan_size("regression", np.zeros(300), np.repeat(np.arange(100), 3), folds=5) == (
        80, 240, "units")
    # Folds that follow time: 12 people in time order, 3 folds over 4 blocks of 3. The training
    # folds hold 3, 6 and 9 people; the median is 6.
    assert T.plan_size("regression", np.zeros(12), None, folds=3, order=np.arange(12.0)) == (
        6, 6, "units")
    # 4 folds over 5 blocks: training folds of 2, 5, 7 and 10 of 12 rows, by forward_blocks' cuts
    # nearest 12·j/5 (2.4 → 2, 4.8 → 5, 7.2 → 7, 9.6 → 10); the lower median is 5.
    assert T.plan_size("regression", np.zeros(12), None, folds=4, order=np.arange(12.0)) == (
        5, 5, "units")


def test_t9_the_inner_folds_and_their_floor():
    assert T.inner_k("search", 13_600) == 3
    assert T.inner_k("path", 340) == 5 and T.inner_k("path", 99) == 3
    assert T.inner_k("path", 5001) == 3
    assert T.inner_k("none", 13_600) == 0
    # every inner fold holds at least 2 units of the rarest class: K = min(K, ⌊m/2⌋), 0 below 2
    assert T.inner_k("search", 5, rarest=5) == 2
    assert T.inner_k("search", 3, rarest=3) == 0
    assert T.inner_k("path", 340, rarest=9) == 4
    # and, under the population answer, at least 2 PSUs
    assert T.inner_k("search", 13_600, psus=15) == 3
    assert T.inner_k("search", 13_600, psus=5) == 2
    assert T.inner_k("path", 13_600, psus=3) == 0


def test_t9_t17_the_candidate_counts_follow_the_plans_size():
    def plan(n_plan, decl=SEARCH, **more):
        return T.make_plan(_family(decl), task="regression", loss="mse", n_plan=n_plan,
                           plan_rows=n_plan, unit="units", split_seed=0, **more)

    def sizes(p):
        return (sum(c.standard for c in p.candidates), sum(not c.standard for c in p.candidates))

    assert sizes(plan(299)) == (1, 0)
    assert sizes(plan(300)) == (1, 8)
    assert sizes(plan(999)) == (1, 8)
    assert sizes(plan(1000)) == (1, 16)
    assert sizes(plan(1000, mode="lighter")) == (1, 8)
    assert sizes(plan(340, mode="lighter")) == (1, 4)
    assert sizes(plan(13_600, mode="standard")) == (1, 0)
    assert sizes(plan(13_600, mode="manual", manual={"learning_rate": 0.2})) == (1, 0)
    assert sizes(plan(13_600, decl=SCALES)) == (1, 8)  # max_drawn
    assert sizes(plan(13_600, decl=SCALES, mode="lighter")) == (1, 4)
    # T9 and T17 at an effective size of 214: exactly the two standard candidates under "Try both"
    small = plan(214, options=TRY_BOTH)
    assert sizes(small) == (2, 0) and small.inner_k == 3
    assert [dict(c.options) for c in small.candidates] == [{"missing": "native"},
                                                           {"missing": "fill"}]
    assert small.fits() == (3 - 1) * 2 + 1


def test_t9_with_too_few_of_the_rarest_class_no_fit_draws_inner_folds():
    def plan(rarest, **more):
        return T.make_plan(_family(SEARCH), task="binary", loss="log_loss", n_plan=rarest,
                           plan_rows=400, unit="events", split_seed=0, options=TRY_BOTH, **more)

    five = plan(5)
    assert five.inner_k == 2 and len(five.candidates) == 2
    three = plan(3)
    # no inner folds: the first standard candidate, the slot's first option, and the refit alone
    assert three.inner_k == 0
    assert [(dict(c.values), dict(c.options)) for c in three.candidates] == [
        (dict(SEARCH.standard), {"missing": "native"})]
    assert three.fits() == 1
    # A path family cannot choose its penalty without inner folds: a stated refusal.
    with pytest.raises(T.NoInnerFolds, match="each inner fold needs at least 2"):
        T.make_plan(_family(RIDGE, key="probe_ridge"), task="binary", loss="log_loss", n_plan=3,
                    plan_rows=400, unit="events", split_seed=0)


def test_t17_early_stopping_is_decided_once_by_the_plan():
    def plan(n_plan, plan_rows, decl=SEARCH):
        return T.make_plan(_family(decl), task="regression", loss="mse", n_plan=n_plan,
                           plan_rows=plan_rows, unit="units", split_seed=0)

    # Sobol candidates stop early from an effective size of 1,500; the standard ones follow
    # scikit-learn's own rule, more than 10,000 rows in the plan's fit.
    assert (plan(1499, 1499).early_stopping, plan(1500, 1500).early_stopping) == (False, True)
    assert (plan(1500, 10_000).standard_stops, plan(1500, 10_001).standard_stops) == (False, True)
    xgb_like = replace(SEARCH, early_stopping={**SEARCH.early_stopping, "rounds": 2000,
                                               "patience": 50, "standard_rows": None})
    assert plan(20_000, 20_000, decl=xgb_like).standard_stops is False
    no_stopping = replace(SEARCH, early_stopping=None, dimensions=SEARCH.dimensions[:5],
                          standard={k: v for k, v in SEARCH.standard.items() if k != "max_iter"})
    assert (plan(20_000, 20_000, decl=no_stopping).early_stopping,
            plan(20_000, 20_000, decl=no_stopping).standard_stops) == (False, False)


def test_the_worked_example_costs_37_fits_per_outer_fit():
    """RECIPES §4.2: 17,000 training participants, folds of 5, boosted trees at their defaults
    ("Try both" on blanks): n_plan 13,600, s = 2, S = 16, C = 18, K = 3, early stopping on;
    F = (K − 1)·C + 1 = 37."""
    n_plan, plan_rows, unit = T.plan_size("regression", np.zeros(17_000), None, folds=5)
    plan = T.make_plan(_family(SEARCH), task="regression", loss="mse", n_plan=n_plan,
                       plan_rows=plan_rows, unit=unit, split_seed=3, options=TRY_BOTH)
    assert (len(plan.candidates), plan.inner_k, plan.early_stopping) == (18, 3, True)
    assert plan.fits() == 2 * 18 + 1 == 37
    assert T.STRATEGIES["sobol"].fit_count(plan) == 37
    assert list(T.STRATEGIES["sobol"].order(plan)) == list(range(18))
    # with the imbalance correction, each fit is 5 (1 + 5 folds × 4/5)
    weighted = T.make_plan(_family(SEARCH), task="binary", loss="log_loss", n_plan=n_plan,
                           plan_rows=plan_rows, unit="events", split_seed=3, options=TRY_BOTH,
                           imbalance=True)
    assert weighted.fits() == 5 * 37
    # out of bag: every candidate once, then the refit
    oob = replace(SCALES, out_of_bag=True)
    bagged = T.make_plan(_family(oob), task="regression", loss="mse", n_plan=n_plan,
                         plan_rows=plan_rows, unit=unit, split_seed=3, out_of_bag=True)
    assert bagged.fits() == len(bagged.candidates) + 1 == 1 + 8 + 1
    # a path: F = w·[(K − 1)·r + 1], whatever the grid's length
    ridge = T.make_plan(_family(RIDGE, key="probe_ridge"), task="regression", loss="mse",
                        n_plan=340, plan_rows=340, unit="units", split_seed=0)
    assert ridge.inner_k == 5 and ridge.fits() == (5 - 1) * 1 + 1
    # nothing to choose: the refit alone
    huber = T.make_plan(_family(HUBER, key="probe_huber", tasks=("regression",)),
                        task="regression", loss="mse", n_plan=13_600, plan_rows=13_600,
                        unit="units", split_seed=0, manual={"t": 2.0})
    assert [dict(c.values) for c in huber.candidates] == [{"t": 2.0}] and huber.fits() == 1


def test_a_family_with_no_tuning_has_no_plan_and_per_task_declarations_are_read_per_task():
    assert T.make_plan(_family(None), task="regression", loss="mse", n_plan=500, plan_rows=500,
                       unit="units", split_seed=0) is None
    per_task = _family({"regression": SEARCH, "binary": SCALES})
    assert T.tuning_for(per_task, "regression") is SEARCH
    assert T.tuning_for(per_task, "binary") is SCALES
    assert T.tuning_for(per_task, "multiclass") is None
    assert T.tuning_for(_family(SEARCH, tasks=("regression",)), "binary") is None


# ── the plan as a record ─────────────────────────────────────────────────────


def test_the_plan_is_frozen_canonical_and_pickles():
    plan = T.make_plan(_family(SCALES, key="probe_scales"), task="binary", loss="log_loss",
                       n_plan=1200, plan_rows=5000, unit="events", split_seed=5,
                       options=TRY_BOTH, manual={"max_depth": 4}, weighted=True, threads=2)
    d = plan.to_dict()
    text = json.dumps(d, sort_keys=True)  # JSON-safe as it stands
    assert T.TuningPlan.from_dict(json.loads(text)) == plan
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert d["candidates"][0] == {"index": 0, "standard": True, "options": {"missing": "native"},
                                  "values": {**SCALES.standard, "max_depth": 4}}
    assert d["options"] == [["missing", ["native", "fill"]]]
    with pytest.raises(FrozenInstanceError):
        plan.seed = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="strategy 'halving'"):
        T.TuningPlan(family="x", kind="search", task="regression", loss="mse",
                     strategy="halving")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'shuffled'"):
        T.TuningPlan.from_dict({**d, "shuffled": True})


def test_strategies_are_keyed_by_strategy_never_by_family():
    """Seam guard 3 (V2X_SEAMS): a v2.x strategy is one more name; no family key is a strategy."""
    from turbotab.core.models import families

    assert list(T.STRATEGIES) == ["sobol"]
    assert not set(T.STRATEGIES) & {f.key for f in families()}
    assert isinstance(T.STRATEGIES["sobol"], T.Strategy)
    assert T.STRATEGIES["sobol"].name == "sobol"


def test_sobol_sample_is_scipys_scrambled_sample_at_a_power_of_two():
    want = qmc.Sobol(3, scramble=True, rng=12345).random_base2(4)
    assert np.array_equal(T.sobol_sample(3, 16, 12345), want)
    assert T.sobol_sample(3, 0, 1).shape == (0, 3)
    with pytest.raises(ValueError, match="a power of two"):
        T.sobol_sample(3, 12, 1)


def test_the_fit_design_takes_the_fits_rows():
    design = T.FitDesign(strata=np.array([1, 1, 2, 2]), psu=np.array([1, 2, 1, 2]),
                         weights=np.array([1.0, 2.0, 3.0, 4.0]))
    part = design.take(np.array([3, 0]))
    assert part.strata.tolist() == [2, 1] and part.psu.tolist() == [2, 1]
    assert part.weights.tolist() == [4.0, 1.0]
    assert T.FitDesign().take(np.array([0])).weights is None


def test_the_pooled_loss_is_the_weighted_mean_of_every_inner_rows_loss():
    y = np.array([1.0, 2.0, 4.0])
    p = np.array([1.5, 2.0, 3.0])
    assert T.pooled_loss("regression", "mse", y, p, classes=None) == pytest.approx(
        (0.25 + 0 + 1) / 3)
    assert T.pooled_loss("regression", "mse", y, p, classes=None,
                         weights=np.array([1.0, 2.0, 3.0])) == pytest.approx(
        (1 * 0.25 + 2 * 0 + 3 * 1) / 6)
    P = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    assert T.pooled_loss("binary", "log_loss", np.array([0, 1, 1]), P, classes=[0, 1]) == (
        pytest.approx(-(math.log(0.8) + math.log(0.7) + math.log(0.6)) / 3))


def test_a_declaration_states_its_new_fields_and_refuses_what_cannot_run():
    """The defaulted fields RT-1a adds to a dimension, and the checks a declaration runs when it is
    made: a path's grid needs points or choices, a dimension that only searches without early
    stopping needs early stopping, a search states a standard value for every dimension."""
    d = _dim("lambda", 1e-5, 1e2, "log")
    assert (d.points, d.active, d.tunability) == (0, "always", None)
    with pytest.raises(ValueError, match="'lambda' needs points or choices on a path"):
        TuningDecl("path", dimensions=(d,), space_version="p/1")
    with pytest.raises(ValueError, match="'max_iter' is searched only without early stopping"):
        TuningDecl("search", dimensions=(_dim("max_iter", 25, 500, "log_int",
                                              active="without_early_stopping"),),
                   standard={"max_iter": 100})
    with pytest.raises(ValueError, match=r"no standard value for \['max_depth'\]"):
        TuningDecl("search", dimensions=(_dim("max_depth", 2, 10, "int"),))
    with pytest.raises(ValueError, match="early_stopping lacks"):
        TuningDecl("search", dimensions=SEARCH.dimensions, standard=SEARCH.standard,
                   early_stopping={"param": "max_iter"})
    with pytest.raises(ValueError, match="a power of two"):
        replace(SCALES, max_drawn=6)


# ── repairs after review ─────────────────────────────────────────────────────


def test_nothing_to_choose_is_the_refit_alone_out_of_bag_and_on_a_path_too():
    """RECIPES §4.2: "Nothing to choose (C = 1): F = w, the refit alone", whatever else the plan
    says. Out of bag below an effective size of 300 the forest has only its standard candidate (S =
    0), and a path whose penalty is set by hand has one point; neither has anything to score."""
    forest_like = replace(SCALES, out_of_bag=True)
    for n_plan in (250, 299):
        small = T.make_plan(_family(forest_like), task="regression", loss="mse", n_plan=n_plan,
                            plan_rows=n_plan, unit="units", split_seed=0, out_of_bag=True)
        assert len(small.candidates) == 1 and small.fits() == 1 and not small.chooses()
    held = T.make_plan(_family(RIDGE, key="probe_ridge"), task="regression", loss="mse",
                       n_plan=340, plan_rows=340, unit="units", split_seed=0,
                       manual={"lambda": 0.1})
    assert [dict(c.values) for c in held.candidates] == [{"lambda": 0.1}]
    assert held.fits() == 1 and not held.chooses()
    assert T.make_plan(_family(RIDGE, key="probe_ridge"), task="regression", loss="mse",
                       n_plan=340, plan_rows=340, unit="units", split_seed=0).chooses()
    # with the imbalance correction, w = 5: the refit is the wrapped model's five fits
    wrapped = T.make_plan(_family(RIDGE, key="probe_ridge"), task="binary", loss="log_loss",
                          n_plan=340, plan_rows=2000, unit="events", split_seed=0,
                          manual={"lambda": 0.1}, imbalance=True)
    assert wrapped.fits() == 5
    # A penalty set by hand needs no inner folds to choose it, so too few events is no refusal.
    tiny = T.make_plan(_family(RIDGE, key="probe_ridge"), task="binary", loss="log_loss",
                       n_plan=3, plan_rows=400, unit="events", split_seed=0,
                       manual={"lambda": 0.1})
    assert [dict(c.values) for c in tiny.candidates] == [{"lambda": 0.1}] and tiny.fits() == 1


def _split_stage_sizes(y, order, folds):
    """The split stage's own time-ordered folds (``stages.rows._assign_folds``, which keys each row
    by its position), and the median training fold's events and rows counted by hand: fold j is
    scored by a model fit on folds 0 … j − 1."""
    from turbotab.core.stages.rows import _assign_folds

    fold, k = _assign_folds(len(y), y.astype(str), None, folds, 0, True, [], order=order)
    sizes = []
    for j in range(1, k + 1):
        train = fold < j
        sizes.append((min(int((y[train] == 1).sum()), int((y[train] == 0).sum())),
                      int(train.sum())))
    middle = (len(sizes) - 1) // 2
    return sorted(s[0] for s in sizes)[middle], sorted(s[1] for s in sizes)[middle]


def test_t9_time_ordered_plan_size_draws_the_split_stages_own_folds_when_times_tie():
    """Tied times (survey cycles, shared visit dates) are broken by the unit's key, so the plan must
    key rows as the split stage does (by position, ``2`` before ``10``), not as padded text."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        order = rng.integers(0, 6, 60).astype(float)
        y = (rng.random(60) < 0.3).astype(int)
        assert T.plan_size("binary", y, None, folds=3, order=order)[:2] == _split_stage_sizes(
            y, order, 3), seed


def test_the_plan_round_trips_tuple_values_and_hashes():
    """A tuple setting (a network's layer sizes) is a list in the canonical form and a tuple again
    after :meth:`from_dict`, so the round trip is exact; equal plans hash alike."""
    layers = TuningDecl("search", dimensions=(
        _dim("hidden_layer_sizes", 0, 0, "choice", choices=((50,), (100, 50))),),
        standard={"hidden_layer_sizes": (100,)}, space_version="probe_net/1")
    plan = T.make_plan(_family(layers, key="probe_net"), task="regression", loss="mse",
                       n_plan=1200, plan_rows=1200, unit="units", split_seed=4)
    d = plan.to_dict()
    assert d["candidates"][0]["values"] == {"hidden_layer_sizes": [100]}
    again = T.TuningPlan.from_dict(json.loads(json.dumps(d)))
    assert again == plan
    assert again.candidates[0].values["hidden_layer_sizes"] == (100,)
    assert {c.values["hidden_layer_sizes"] for c in again.candidates} <= {(50,), (100, 50), (100,)}
    assert hash(again) == hash(plan) and len({plan, again}) == 1
    held = T.make_plan(_family(layers, key="probe_net"), task="regression", loss="mse",
                       n_plan=1200, plan_rows=1200, unit="units", split_seed=4,
                       manual={"hidden_layer_sizes": (50,)})
    assert T.TuningPlan.from_dict(json.loads(json.dumps(held.to_dict()))) == held


def test_sobol_sample_seeds_scipy_before_1_15_by_seed(monkeypatch):
    """RECIPES §4.7: ``rng=`` from scipy 1.15, ``seed=`` before; both seed the same generator
    (``numpy.random.default_rng(seed)``), so the sample is the same."""
    real = qmc.Sobol

    class OldSobol:  # scipy 1.13's signature: seed=, no rng=
        def __init__(self, d, *, scramble=True, bits=None, seed=None, optimization=None):
            self._inner = real(d, scramble=scramble, rng=np.random.default_rng(seed))

        def random_base2(self, m):
            return self._inner.random_base2(m)

    monkeypatch.setattr(qmc, "Sobol", OldSobol)
    assert np.array_equal(T.sobol_sample(3, 16, 12345),
                          real(3, scramble=True, rng=12345).random_base2(4))
