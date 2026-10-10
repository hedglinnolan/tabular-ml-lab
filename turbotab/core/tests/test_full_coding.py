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


# ── a category's tie under a pure lasso (the verifier's tie.py) ──────────────


def _coded(names: dict[str, str], seed: int = 0, n: int = 300, task: str = "regression"):
    """A number and two categories, sex (two levels) and region (three), each level its own
    column, standardized as the family's scale step does; ``names`` renames the levels."""
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(seed)
    sex = rng.choice(["female", "male"], n)
    region = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    x = rng.normal(size=n)
    signal = (x + 1.2 * (sex == "male") - 0.8 * (region == "b") + 0.5 * (region == "c")
              + rng.normal(size=n))
    y = signal if task == "regression" else (signal > np.median(signal)).astype(int)
    cats = pd.DataFrame({"sex": [names.get(v, v) for v in sex],
                         "region": [names.get(v, v) for v in region]})
    enc = OneHotEncoder(drop=None, sparse_output=False).fit(cats)
    back = {v: k for k, v in names.items()}
    labels = ["x"] + [f"{c.split('_', 1)[0]}_{back.get(c.split('_', 1)[1], c.split('_', 1)[1])}"
                      for c in enc.get_feature_names_out()]
    D = np.column_stack([x, enc.transform(cats)])
    Z = (D - D.mean(axis=0)) / D.std(axis=0)
    return Z, y, labels


SWAPPED = {"female": "z_female", "male": "a_male", "a": "z_a", "b": "y_b", "c": "x_c"}


def _by_label(coef, labels) -> dict[str, float]:
    return dict(zip(labels, np.ravel(coef)))


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_a_pure_lasso_reports_a_two_level_categorys_tie_at_its_middle_whatever_sorts_first(task):
    """The verifier's tie.py: under a pure lasso, with every level its own column, female and male
    can trade their shared contrast along a direction that changes neither the fit nor the penalty,
    and the exact path stopped at one end (female −0.6625, male 0; renamed, male +0.6625, female
    0). The reference is computed here: scikit-learn's own solver (coordinate descent, or ``saga``
    for the logistic loss) run tight stops somewhere on the tie; the middle is half the contrast
    each way, β_female = (β_f − β_m)/2 = −β_male (the two standardized columns are each other's
    negatives). Ours equals that to 1e-6 under either naming, its predictions are the same, its
    penalty is the lowest (the same ‖β‖₁ as the reference's), and region, three levels, needs no
    move: its lasso solution is one point, one level exactly 0, the same one whatever the names."""
    from sklearn.linear_model import ElasticNet, LogisticRegression

    from turbotab.core.models.elastic_net import (ExactElasticNet, ExactLogisticRegression,
                                                  lambda_max)

    ours = []
    for names in ({}, SWAPPED):
        Z, y, labels = _coded(names, task=task)
        top = lambda_max(Z, y, 1.0, task=task)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "regression":
                model = ExactElasticNet(alpha=0.05 * top, l1_ratio=1.0).fit(Z, y)
                theirs = ElasticNet(alpha=0.05 * top, l1_ratio=1.0, tol=1e-14,
                                    max_iter=1_000_000).fit(Z, y)
                fit, their_fit = model.predict(Z), theirs.predict(Z)
            else:
                C = 1.0 / (len(y) * 0.05 * top)
                model = ExactLogisticRegression(C=C, l1_ratio=1.0).fit(Z, y)
                theirs = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", tol=1e-12,
                                            max_iter=500_000, random_state=0).fit(Z, y)
                fit, their_fit = model.predict_proba(Z), theirs.predict_proba(Z)
        got, ref = _by_label(model.coef_, labels), _by_label(theirs.coef_, labels)
        contrast = ref["sex_female"] - ref["sex_male"]
        middle = {**ref, "sex_female": contrast / 2, "sex_male": -contrast / 2}
        assert abs(contrast) > 0.1  # a real contrast, so the tie is not at zero
        assert max(abs(got[k] - middle[k]) for k in labels) <= 1e-6, (names, got, middle)
        assert np.max(np.abs(fit - their_fit)) <= 1e-6
        assert np.abs(np.ravel(model.coef_)).sum() <= np.abs(np.ravel(theirs.coef_)).sum() + 1e-9
        assert sum(got[f"region_{v}"] == 0.0 for v in "abc") == 1  # one point: one level at 0
        ours.append((got, fit))
    (first, fit_first), (again, fit_again) = ours
    assert max(abs(first[k] - again[k]) for k in first) <= 1e-10
    assert np.max(np.abs(fit_first - fit_again)) <= 1e-10


def test_more_than_two_classes_under_a_pure_lasso_report_both_ties_at_their_middles():
    """Three classes: a number added to a column in every class changes no probability (the
    classes' tie, ``exact_path.middle_of_ties``) and a category's columns move together within a
    class (its tie). Both are reported at their middles: the coefficients follow the levels, not
    their names (1e-10), each column's median over the classes is 0, and sex's two columns are
    each other's negatives in every class (half the contrast each way); the probabilities do not
    move."""
    from turbotab.core.models.elastic_net import ExactLogisticRegression, lambda_max

    out = []
    for names in ({}, SWAPPED):
        Z, signal, labels = _coded(names, seed=3)
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
        C = 1.0 / (len(y) * 0.05 * lambda_max(Z, y, 1.0, task="multiclass"))
        model = ExactLogisticRegression(C=C, l1_ratio=1.0).fit(Z, y)
        W = pd.DataFrame(model.coef_, columns=labels)[sorted(labels)]
        out.append((W, model.predict_proba(Z)))
        assert np.max(np.abs(np.median(W.to_numpy(), axis=0))) <= 1e-12
        assert np.max(np.abs(W["sex_female"] + W["sex_male"])) <= 1e-12
    assert np.max(np.abs(out[0][0].to_numpy() - out[1][0].to_numpy())) <= 1e-10
    assert np.max(np.abs(out[0][1] - out[1][1])) <= 1e-10


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_pure_lasso_path_is_the_refit_at_every_point_and_follows_the_levels(task):
    """The family's path at a pure lasso holds each point at the middle of its tie, as the refit
    does (the refit at r·λ_max is that point of the path, to 1e-10), so the shrinkage path's lines
    follow the levels and not their names, and a row whose level the fit never saw (every
    indicator 0) is predicted the same whichever level sorts first."""
    from turbotab.core.models.elastic_net import ELASTIC_NET, ExactElasticNet, ExactLogisticRegression

    ratios = np.exp(np.linspace(0.0, np.log(1e-3), 8))
    paths, unseen = [], []
    for names in ({}, SWAPPED):
        Z, y, labels = _coded(names, task=task)
        fitted = ELASTIC_NET.path(Z, y, {"l1_ratio": [1.0], "ratio": ratios}, task=task)
        coefs = np.asarray(fitted.coefs)[0]  # (points, 1, p)
        intercepts = np.asarray(fitted.intercepts)[0]
        g = 4
        value = float(np.asarray(fitted.values)[0, g])
        if task == "regression":
            refit = ExactElasticNet(alpha=value, l1_ratio=1.0).fit(Z, y)
        else:
            refit = ExactLogisticRegression(C=1.0 / (len(y) * value), l1_ratio=1.0).fit(Z, y)
        assert np.max(np.abs(np.ravel(refit.coef_) - coefs[g, 0])) <= 1e-10
        assert abs(float(np.ravel(refit.intercept_)[0]) - float(intercepts[g, 0])) <= 1e-10
        order = np.argsort(labels)
        paths.append(coefs[:, 0, order])
        none = np.zeros((1, Z.shape[1]))
        none[0, 0] = 0.3
        lo = Z.min(axis=0)
        for j, label in enumerate(labels):
            if label != "x":
                none[0, j] = lo[j]  # every indicator at its "not this level" value
        unseen.append(none @ coefs[:, 0, :].T + intercepts[:, 0])
    assert np.max(np.abs(paths[0] - paths[1])) <= 1e-10
    assert np.max(np.abs(unseen[0] - unseen[1])) <= 1e-10


def test_a_full_coded_family_is_described_with_the_columns_it_fits():
    """The verifier's cols2.py: the design describes every family from the shared matrix, coded
    with the first level dropped (19 columns on NHANES-like data), but the elastic net's head had
    one more column per category. Its scale step now names its own width: the fitted head's, one
    more column for each of region and smoker than least squares' first-level coding."""
    from turbotab.core.models.pipeline import shared_steps, transformer

    X, y = _table()
    frame = X.assign(smoker=X["smoker"].fillna("never"))
    spec = _spec(False)
    shared = transformer(shared_steps(spec)).fit(frame[spec.inputs], y)
    n_shared = len(shared.get_feature_names_out())
    family = get_family("elastic_net")
    spec = with_plans(spec, [family], "regression", y, split_seed=11)
    pipe = build_pipeline(spec, family, "regression", "prediction", len(y), n_shared)
    head = pipe[:-1].fit(frame[spec.inputs], y)
    width = len(head.get_feature_names_out())
    assert width == n_shared + 2  # region and smoker, each one more column
    steps = {s["key"]: s["detail"] for s in describe_steps(spec, family, "regression",
                                                           "prediction", n_shared)}
    assert steps["scale"].startswith(f"Centers and scales all {width} columns")
    ridge = {s["key"]: s["detail"] for s in describe_steps(spec, get_family("ridge"),
                                                            "regression", "prediction", n_shared)}
    assert ridge["scale"].startswith(f"Centers and scales all {width} columns")
