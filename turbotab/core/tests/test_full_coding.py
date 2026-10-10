"""Full coding for the penalized families (RECIPES_AND_TUNING §2.2, §2.4; RT-5f).

Ridge and the elastic net shrink every coefficient toward zero. With the first level as the
reference, a category's coefficients are its levels' differences from that level, so the penalty
pulls every level toward whichever one sorts first, and the predictions change when the levels are
merely renamed. Coded with every level its own column, the columns are the same whatever the
names, the penalty treats them alike, and the predictions do not move.

**Independent reference.** The same table with its levels renamed (a permutation of which sorts
first) must give the same predictions, through both one-hot paths (the one-hot step, and the
``levels`` step that keeps blanks as a level), for a number and a yes/no outcome; the inner splits
are held by unit, so both fits search on the same rows. Least squares, which no penalty touches,
keeps its reference coding, and its predictions do not move either (a reparameterization).
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models.base import get_family
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import (DesignSpec, MissingLevelEncoder, build_pipeline,
                                           describe_steps, onehot_drop, with_plans)

RENAMED = {"north": "zz_north", "south": "aa_south", "east": "mm_east", "west": "bb_west"}
SMOKER = {"never": "z_never", "former": "a_former", "current": "m_current"}


def _table(n: int = 240, seed: int = 3, task: str = "regression"
           ) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    region = rng.choice(list(RENAMED), size=n, p=[0.4, 0.3, 0.2, 0.1])
    smoker = rng.choice(list(SMOKER), size=n).astype(object)
    smoker[rng.random(n) < 0.15] = None  # blanks, a level of their own on the levels path
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "region": region,
                      "smoker": smoker}, index=pd.Index(np.arange(n) + 300, name="row_id"))
    effect = pd.Series(region).map({"north": 0.0, "south": 0.9, "east": -0.6, "west": 1.4})
    puff = pd.Series(smoker).map({"never": 0.0, "former": 0.4, "current": 1.0}).fillna(0.7)
    signal = (X["x1"].to_numpy() + 0.5 * X["x2"].to_numpy() + effect.to_numpy() + puff.to_numpy()
              + rng.normal(0, 1.0, n))
    if task == "regression":
        return X, signal
    return X, (signal > np.median(signal)).astype(int)


def _spec(levels: bool) -> DesignSpec:
    return DesignSpec(predictors=["x1", "x2", "region", "smoker"],
                      inputs=["x1", "x2", "region", "smoker"],
                      categorical=["region", "smoker"], numeric=["x1", "x2"], energy=None,
                      impute=False, roles={"x1": "exposure", "x2": "covariate",
                                           "region": "covariate", "smoker": "covariate"},
                      levels=["smoker"] if levels else [])


def _renamed(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["region"] = out["region"].map(RENAMED)
    out["smoker"] = out["smoker"].map(SMOKER)
    return out


def _predictions(key: str, task: str, X: pd.DataFrame, y: np.ndarray, levels: bool) -> np.ndarray:
    family = get_family(key)
    spec = _spec(levels)
    if not levels:
        X = X.assign(smoker=X["smoker"].fillna("never" if "never" in set(X["smoker"]) else
                                               "z_never"))
    units = np.arange(len(y))  # every row its own unit: the inner splits read units, not values
    spec = with_plans(spec, [family], task, y, units=units, split_seed=11)
    pipe = build_pipeline(spec, family, task, "prediction", len(y), len(spec.inputs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit_pipeline(pipe, X[spec.inputs], y, groups=units, seed=11)
    if task == "regression":
        return np.asarray(fitted.predict(X[spec.inputs]), dtype=float)
    return np.asarray(fitted.predict_proba(X[spec.inputs]), dtype=float)


@pytest.mark.parametrize("levels", [False, True], ids=["onehot", "blanks-as-a-level"])
@pytest.mark.parametrize("key,task", [("elastic_net", "regression"), ("elastic_net", "binary"),
                                      ("ridge", "regression"), ("ridge", "binary")])
def test_a_penalized_familys_predictions_do_not_depend_on_which_level_sorts_first(key, task,
                                                                                  levels):
    X, y = _table(task=task)
    first = _predictions(key, task, X, y, levels)
    again = _predictions(key, task, _renamed(X), y, levels)
    assert np.max(np.abs(first - again)) <= 1e-8 * max(1.0, float(np.std(first)))


def test_least_squares_keeps_its_reference_and_renaming_moves_nothing():
    X, y = _table()
    first = _predictions("linear", "regression", X, y, levels=True)
    again = _predictions("linear", "regression", _renamed(X), y, levels=True)
    assert np.max(np.abs(first - again)) <= 1e-8 * float(np.std(first))


@pytest.mark.parametrize("key,drop", [("ridge", None), ("elastic_net", None), ("linear", "first"),
                                      ("boosted_trees", "first")])
def test_each_family_declares_its_coding_and_both_one_hot_paths_read_it(key, drop):
    family = get_family(key)
    assert onehot_drop(family) == drop
    X, y = _table()
    for levels in (False, True):
        spec = replace(with_plans(_spec(levels), [family], "regression", y, split_seed=11))
        pipe = build_pipeline(spec, family, "regression", "prediction", len(y), 4)
        frame = X if levels else X.assign(smoker=X["smoker"].fillna("never"))
        head = pipe[:-1].fit(frame[spec.inputs], y)
        names = [str(c) for c in head.get_feature_names_out()]
        region = sorted(c for c in names if c.startswith("region_"))
        smoker = sorted(c for c in names if c.startswith("smoker_"))
        every = sorted(f"region_{v}" for v in RENAMED)
        assert region == (every if drop is None else every[1:]), (key, levels, region)
        want = {"current", "former", "never"} | ({"Missing"} if levels else set())
        if drop == "first":
            want -= {"current"}
        assert {c.removeprefix("smoker_") for c in smoker} == want, (key, levels, smoker)
        steps = {s["key"]: s["detail"] for s in describe_steps(spec, family, "regression",
                                                               "prediction")}
        said = steps["levels" if levels else "onehot"]
        assert said.endswith("every level has its own column." if drop is None
                             else "the first level is the reference.")


def test_the_levels_encoder_with_every_level_reads_an_unseen_level_as_none_of_them():
    frame = pd.DataFrame({"c": ["b", "a", "c", None, "a"]})
    full = MissingLevelEncoder(["c"], drop=None).fit(frame)
    assert list(full.get_feature_names_out()) == ["c_a", "c_b", "c_c", "c_Missing"]
    out = full.transform(pd.DataFrame({"c": ["c", "zz", None]}))
    assert out.to_numpy().tolist() == [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    first = MissingLevelEncoder(["c"]).fit(frame)
    assert list(first.get_feature_names_out()) == ["c_b", "c_c", "c_Missing"]
    no_blank = MissingLevelEncoder(["c"], drop=None).fit(frame.dropna())
    # a blank where the fitting rows had none reads as their most frequent level
    assert no_blank.transform(pd.DataFrame({"c": [None]})).to_numpy().tolist() == [[1, 0, 0]]
