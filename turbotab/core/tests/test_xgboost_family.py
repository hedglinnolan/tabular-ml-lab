"""RT-5e · the XGBoost family (RECIPES_AND_TUNING §2.2, §4.1, §8 T11 and T13; WAVE_C6A_PLAN §3).

Each check stands on a reference the family's code does not compute:

* **native ``xgb.train``** with its parameters written out here, on the same stopping rows: the same
  best iteration and the same margins, bit for bit;
* **xgboost's own ``XGBRegressor`` / ``XGBClassifier``** at their defaults and the same thread
  count, the labels encoded by hand: the standard settings are XGBoost's defaults, exactly;
* **xgboost's ``pred_contribs``** against the engine's numpy TreeSHAP on the converted trees (rule
  ``<``, each split's default branch for blanks), to 1e-6, with rows on a threshold;
* **xgboost's own covers** (``trees_to_dataframe``) for the mean hessian at the base score, and the
  shares p̄ worked by hand;
* **names** written out by hand, and xgboost's own refusal of the unsafe ones;
* **T13:** two fits at one plan and thread count are identical; the seed is the SHA-256 written out
  here; another split seed gives other predictions;
* **the shelf:** boosted trees' scores less 1.0, at least 0.5, worked by hand.

Fixtures stay small (n ≤ 400, depth ≤ 3) so the file runs in seconds.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import explain as E
from turbotab.core.models import xgboost_family as XF
from turbotab.core.models.base import Situation, assessment, get_family, rank
from turbotab.core.models.tuning import estimator_params, make_plan

REPO = Path(__file__).resolve().parents[3]
THREADS = 2
OBJECTIVE = {"regression": "reg:squarederror", "binary": "binary:logistic",
             "multiclass": "multi:softprob"}


def _data(task: str, n: int = 400, seed: int = 5, blanks: float = 0.08
          ) -> tuple[pd.DataFrame, np.ndarray]:
    """Values exact in single precision (xgboost reads single precision), with blanks."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(np.round(rng.normal(size=(n, 4)), 2).astype(np.float32).astype(float),
                     columns=["a", "b", "c", "d"])
    signal = 1.5 * X["a"] + np.sin(2 * X["b"]) + X["c"] * X["d"]
    X = X.mask(rng.random(X.shape) < blanks)
    noise = rng.normal(size=n)
    if task == "regression":
        return X, (signal + noise).to_numpy()
    if task == "binary":
        return X, np.where(signal + noise > 0.8, "yes", "no")
    return X, np.asarray(["low", "mid", "high"])[np.digitize(signal + noise, [-0.5, 1.0])]


def _codes(y: np.ndarray) -> np.ndarray:
    """Labels to 0…K−1 in sorted order, by hand."""
    order = sorted(set(y.tolist()))
    return np.asarray([order.index(v) for v in y])


def _native(task: str, y: np.ndarray, **params: object) -> dict[str, object]:
    out = {"objective": OBJECTIVE[task], "eta": 0.3, "max_depth": 3, "min_child_weight": 1.0,
           "subsample": 1.0, "colsample_bytree": 1.0, "lambda": 1.0, "alpha": 0.0,
           "booster": "gbtree", "tree_method": "hist", "nthread": THREADS, "seed": 0,
           "verbosity": 0}
    if task == "multiclass":
        out["num_class"] = len(set(y.tolist()))
    out.update(params)
    return out


def _step(task: str, **params: object):
    return get_family("xgboost").build(task, "prediction", 400, 4).set_params(
        n_jobs=THREADS, **params)


def _margin(step, X: pd.DataFrame, task: str) -> np.ndarray:
    return step.predict(X) if task == "regression" else step.decision_function(X)


# ── T11: native xgb.train on the same stopping rows ──────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_it_stops_where_native_xgb_train_stops_on_the_same_rows(task: str) -> None:
    """The stopping rows are every tenth row, chosen here; native ``xgb.train`` runs up to 2,000
    rounds with patience 50 and reports its best iteration. The family's step, handed the same
    rows as ``X_val``, stops at the same round, keeps no round after it, and gives the same
    margins on every row, bit for bit."""
    X, y = _data(task)
    held = np.arange(len(y)) % 10 == 0
    fit_rows, stop_rows = ~held, held
    codes = y if task == "regression" else _codes(y)

    train = xgb.DMatrix(X[fit_rows].to_numpy(), label=codes[fit_rows])
    stop = xgb.DMatrix(X[stop_rows].to_numpy(), label=codes[stop_rows])
    native = xgb.train(_native(task, y), train, num_boost_round=2000, evals=[(stop, "stop")],
                       early_stopping_rounds=50, verbose_eval=False)
    expected = native.predict(xgb.DMatrix(X.to_numpy()), output_margin=True,
                              iteration_range=(0, native.best_iteration + 1))

    step = _step(task, max_depth=3, n_estimators=2000, early_stopping=True,
                 early_stopping_rounds=50)
    step.fit(X[fit_rows], y[fit_rows], X_val=X[stop_rows], y_val=y[stop_rows])
    assert step.best_iteration_ == native.best_iteration
    assert step.best_iteration_ < 1950  # it did stop early
    assert step.n_rounds_ == native.best_iteration + 1
    assert step.stopping_rows_ == int(held.sum())
    assert np.array_equal(_margin(step, X, task), np.asarray(expected, dtype=float))


def test_a_fit_that_does_not_stop_early_refuses_stopping_rows() -> None:
    X, y = _data("regression", n=120)
    with pytest.raises(ValueError, match="does not stop early"):
        _step("regression").fit(X[:100], y[:100], X_val=X[100:], y_val=y[100:])


def test_the_fit_hands_it_whole_stopping_units_drawn_first() -> None:
    """Through ``inner_cv.fit_pipeline``, a step that stops early is handed a tenth of the units
    as stopping rows: 400 rows, each its own unit, give 40 (round(0.1 × 400))."""
    from turbotab.core.models.inner_cv import fit_pipeline

    X, y = _data("binary")
    pipe = Pipeline([("model", _step("binary", max_depth=2, n_estimators=2000,
                                     early_stopping=True))]).set_output(transform="pandas")
    fit_pipeline(pipe, X, y, seed=3)
    assert pipe[-1].stopping_rows_ == 40
    assert pipe[-1].best_iteration_ is not None


# ── T11: the standard settings are XGBoost's defaults ─────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_the_standard_settings_are_xgboosts_own_defaults(task: str) -> None:
    """At the standard candidate (``estimator_params`` on a plan at 2 threads), the step equals
    xgboost's own estimator at its defaults and the same thread count, the labels encoded by hand:
    predictions, probabilities and margins bit for bit."""
    X, y = _data(task)
    family = get_family("xgboost")
    plan = make_plan(family, task=task, loss="mse" if task == "regression" else "log_loss",
                     n_plan=400, plan_rows=400, unit="units" if task == "regression" else
                     ("events" if task == "binary" else "rarest_class"), split_seed=7,
                     threads=THREADS)
    params = estimator_params(family, task, plan.candidates[0].values, n_units=len(y),
                              n_rows=len(y), y=y, plan=plan)
    assert params["min_child_weight"] == 1.0 and params["n_jobs"] == THREADS
    step = family.build(task, "prediction", len(y), 4).set_params(**params).fit(X, y)
    if task == "regression":
        direct = xgb.XGBRegressor(n_jobs=THREADS).fit(X, y)
        assert np.array_equal(step.predict(X), direct.predict(X).astype(float))
        return
    codes = _codes(y)
    direct = xgb.XGBClassifier(n_jobs=THREADS).fit(X, codes)
    assert np.array_equal(step.predict_proba(X), direct.predict_proba(X).astype(float))
    assert np.array_equal(step.decision_function(X),
                          direct.predict(X, output_margin=True).astype(float))
    assert np.array_equal(step.predict(X), np.asarray(sorted(set(y)))[direct.predict(X)])


# ── T11: SHAP and the trees ───────────────────────────────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_pred_contribs_equal_the_engines_treeshap_on_the_converted_trees(task: str) -> None:
    """xgboost's ``pred_contribs`` (the family's ``tree_shap``) against the engine's numpy
    TreeSHAP on the converted trees, every output, to 1e-6, blanks included; both add up to
    xgboost's own margin (local accuracy)."""
    X, y = _data(task, n=300)
    family = get_family("xgboost")
    step = _step(task, max_depth=3, n_estimators=20, subsample=0.8, colsample_bytree=0.75,
                 random_state=4).fit(X, y)
    Z = X.to_numpy(dtype=float)
    phi, expected = family.tree_shap(step, Z)
    ours, ours_expected = E.tree_shap(family.trees(step), Z)
    assert phi.shape == ours.shape == (len(Z), 4, 1 if task != "multiclass" else 3)
    np.testing.assert_allclose(ours, phi, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ours_expected, expected, rtol=0, atol=1e-6)
    margin = np.asarray(_margin(step, X, task)).reshape(len(Z), -1)
    np.testing.assert_allclose(ours_expected + ours.sum(axis=1), margin, rtol=0, atol=1e-5)


def test_a_row_on_a_threshold_goes_right_as_xgboost_sends_it() -> None:
    """One split on a column holding 0, 1, 2 and 3: xgboost's threshold is one of those values,
    so rows sit on it. Under ``<`` they go right; the converted trees' SHAP values add up to
    xgboost's margin on every row, and the tree view says the rule is ``<``."""
    x = np.repeat([0.0, 1.0, 2.0, 3.0], 25)
    X = pd.DataFrame({"x": x, "z": np.zeros_like(x)})
    y = np.where(x >= 2.0, 10.0, 0.0)
    step = _step("regression", max_depth=1, n_estimators=1).fit(X, y)
    ensemble = get_family("xgboost").trees(step)
    root = ensemble.tables[0][0][0]
    assert ensemble.rule == "lt" and float(root["num_threshold"]) in (1.0, 2.0, 3.0)
    on = x == float(root["num_threshold"])
    phi, expected = E.tree_shap(ensemble, X.to_numpy())
    total = expected[0] + phi[:, :, 0].sum(axis=1)
    np.testing.assert_allclose(total, step.predict(X), rtol=0, atol=1e-6)
    right = ensemble.tables[0][0][int(root["right"])]
    assert on.any()  # rows sit on the threshold, and go right
    np.testing.assert_allclose(total[on], ensemble.base[0] + float(right["value"]), rtol=0,
                               atol=1e-6)

    pipe = Pipeline([("model", step)]).set_output(transform="pandas")
    pipe.fit(X, y)
    view = E.tree_structure(E.anatomy(pipe, ["x", "z"], "regression"), ["x", "z"],
                            family="xgboost")
    assert view.rule == "lt"
    assert view.first_tree[0].column == "x"


def test_the_tree_view_of_boosted_trees_keeps_its_rule() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y = _data("regression", n=200)
    pipe = Pipeline([("model", HistGradientBoostingRegressor(max_iter=5))]).set_output(
        transform="pandas").fit(X, y)
    view = E.tree_structure(E.anatomy(pipe, list(X.columns), "regression"), list(X.columns),
                            family="boosted_trees")
    assert view.rule == "le"


def _rows_by_hand(booster, X: pd.DataFrame, tree: int) -> dict[int, int]:
    """Each node of xgboost's own tree ``tree`` (``trees_to_dataframe``) and how many rows of
    ``X`` reach it, routed by hand: left below the split (in single precision), a blank where
    ``Missing`` says."""
    frame = booster.trees_to_dataframe()
    frame = frame[frame["Tree"] == tree].set_index("ID")
    reached: dict[int, int] = {}

    def route(node_id: str, rows: pd.DataFrame) -> None:
        node = frame.loc[node_id]
        reached[int(node["Node"])] = len(rows)
        if node["Feature"] == "Leaf":
            return
        x = rows[node["Feature"]].astype(np.float32)  # xgboost compares in single precision
        left = (x < np.float32(node["Split"])) | (x.isna() & (node["Missing"] == node["Yes"]))
        route(node["Yes"], rows[left.to_numpy()])
        route(node["No"], rows[~left.to_numpy()])

    route(f"{tree}-0", X)
    return reached


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_the_tree_view_counts_rows_not_the_hessian_cover(task: str) -> None:
    """On classes xgboost's cover is a hessian sum (about n·p(1 − p), well below n); the tree
    view's ``n`` is the fit's rows reaching each node, routed by hand through xgboost's own
    first tree, while the converted tables keep the cover for TreeSHAP."""
    X, y = _data(task, n=300)
    pipe = Pipeline([("model", _step(task, max_depth=3, n_estimators=4))]).set_output(
        transform="pandas").fit(X, y)
    view = E.tree_structure(E.anatomy(pipe, list(X.columns), task), list(X.columns),
                            family="xgboost")
    booster = pipe.named_steps["model"].booster_
    by_hand = _rows_by_hand(booster, X, tree=0)
    assert view.first_tree[0].n == 300
    assert {node.id: node.n for node in view.first_tree} == {
        node.id: by_hand[node.id] for node in view.first_tree}
    root_cover = float(booster.trees_to_dataframe().query("Tree == 0 and Node == 0")["Cover"]
                       .iloc[0])
    assert root_cover < 0.6 * 300  # the cover is not a row count here
    table = get_family("xgboost").trees(pipe.named_steps["model"]).tables[0][0]
    assert float(table[0]["count"]) == pytest.approx(root_cover, rel=1e-6)


def test_the_row_counts_leave_out_the_stopping_rows() -> None:
    """Under early stopping the counts are the training rows only (360 of 400, the other 40
    being the stopping rows), in every kept tree, and a split's rows are its children's."""
    X, y = _data("binary")
    held = np.arange(len(y)) % 10 == 0
    step = _step("binary", max_depth=3, n_estimators=2000, early_stopping=True,
                 early_stopping_rounds=20)
    step.fit(X[~held], y[~held], X_val=X[held], y_val=y[held])
    tables = get_family("xgboost").trees(step).tables[0]
    assert len(tables) == step.best_iteration_ + 1
    for table in tables:
        assert int(table[0]["rows"]) == 360
        for node in table[~table["is_leaf"].astype(bool)]:
            assert node["rows"] == table[node["left"]]["rows"] + table[node["right"]]["rows"]


def test_the_explanations_read_pred_contribs_through_the_family() -> None:
    """``explain.attributions`` reads the family's ``tree_shap``: xgboost's values, as
    ``Booster.predict(pred_contribs=True)`` gives them on the model matrix."""
    X, y = _data("binary", n=200)
    pipe = Pipeline([("model", _step("binary", max_depth=2, n_estimators=10))]).set_output(
        transform="pandas").fit(X, y)
    anat = E.anatomy(pipe, list(X.columns), "binary")
    assert E.model_kind("xgboost", anat) == "trees"
    found = E.attributions(anat, X, X, kind="trees", family="xgboost")
    booster = pipe[-1].booster_
    direct = booster.predict(xgb.DMatrix(X.to_numpy(), feature_names=booster.feature_names),
                             pred_contribs=True)
    assert np.array_equal(found.phi.to_numpy(), direct[:, :-1].astype(float))
    assert found.expected == float(direct[0, -1]) and found.scale == "margin"


# ── the least child weight, in rows ────────────────────────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_the_mean_hessian_is_xgboosts_own_at_the_base_score(task: str) -> None:
    """One round of stumps: each tree's root cover (xgboost's hessian sum,
    ``trees_to_dataframe``) is n times the mean hessian. By hand: 1 for a number; p̄(1 − p̄) for
    two classes; for several, xgboost's softmax hessian 2·p̄ₖ(1 − p̄ₖ), averaged over the classes.
    A child weight of 10 rows is 10 times that."""
    X, y = _data(task)
    n = len(y)
    if task == "regression":
        by_hand = 1.0
    else:
        shares = [float(np.mean(y == c)) for c in sorted(set(y.tolist()))]
        by_hand = (shares[0] * (1 - shares[0]) if len(shares) == 2
                   else float(np.mean([2 * p * (1 - p) for p in shares])))
    step = _step(task, max_depth=1, n_estimators=1).fit(X, y)
    frame = step.booster_.trees_to_dataframe()
    covers = frame[frame["Node"] == 0]["Cover"].to_numpy()
    assert covers.mean() / n == pytest.approx(by_hand, rel=1e-5)
    assert XF.mean_hessian(y, classify=task != "regression") == pytest.approx(by_hand, rel=1e-12)
    family = get_family("xgboost")
    params = family.settings({"min_child_weight": 10.0}, task=task, n_units=n, n_rows=n, y=y,
                             Z=None, plan=None)
    assert params["min_child_weight"] == pytest.approx(10.0 * by_hand, rel=1e-12)


def test_the_child_weight_reads_only_the_fits_own_rows() -> None:
    """Two fits whose rows differ in prevalence get different weights at the same candidate."""
    family = get_family("xgboost")
    rare = np.asarray([1] * 5 + [0] * 95)
    common = np.asarray([1] * 50 + [0] * 50)
    weights = [family.settings({"min_child_weight": 64.0}, task="binary", n_units=100,
                               n_rows=100, y=y, Z=None, plan=None)["min_child_weight"]
               for y in (rare, common)]
    assert weights == pytest.approx([64 * 0.05 * 0.95, 64 * 0.25])


def test_its_plan_never_stops_the_standard_settings_early() -> None:
    """XGBoost's defaults run 100 rounds whatever the rows (``standard_rows: None``); its Sobol
    candidates stop early from an effective size of 1,500, holding no round count."""
    family = get_family("xgboost")
    plan = make_plan(family, task="regression", loss="mse", n_plan=4000, plan_rows=50_000,
                     unit="units", split_seed=1)
    assert plan.early_stopping and not plan.standard_stops
    assert plan.candidates[0].values["n_estimators"] == 100
    assert all("n_estimators" not in c.values for c in plan.candidates[1:])
    assert len(plan.candidates) == 1 + 16


# ── names ────────────────────────────────────────────────────────────────────


def test_unsafe_names_are_replaced_and_mapped_back() -> None:
    """xgboost refuses ``[``, ``]`` and ``<`` in a name. Replaced, kept distinct, and read back."""
    names = ["dose [mg]", "age<65", "x[", "x(", "a]b", "plain"]
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, len(names))), columns=names)
    y = X["plain"].to_numpy() + rng.normal(size=120)
    with pytest.raises(ValueError):
        xgb.XGBRegressor(n_estimators=2).fit(X, y)  # xgboost's own refusal
    safe = ["dose (mg)", "agelt65", "x(", "x(~2", "a)b", "plain"]
    assert XF.safe_names(names) == safe
    step = _step("regression", n_estimators=5).fit(X, y)
    assert step.booster_.feature_names == safe
    assert list(step.feature_names_in_) == names
    assert step.original_names_ == dict(zip(safe, names))
    assert np.array_equal(step.predict(X), step.predict(X.to_numpy()))
    with pytest.raises(ValueError, match="not the ones this model was fit on"):
        step.predict(X[names[::-1]])


# ── T13: reproducibility at a pinned thread count ─────────────────────────────


def test_one_plan_and_thread_count_reproduce_a_fit_and_another_seed_does_not() -> None:
    """At a candidate that subsamples rows and columns: two fits at one plan and 2 threads are
    identical (within replay's 1e-12, here exactly); the seed is the SHA-256 of the canonical
    JSON of (split seed, family, space version), its first four bytes; another split seed
    changes the predictions."""
    X, y = _data("regression")
    family = get_family("xgboost")
    values = {"learning_rate": 0.1, "max_depth": 3, "min_child_weight": 4.0, "subsample": 0.7,
              "colsample_bytree": 0.6, "reg_lambda": 2.0, "n_estimators": 40}

    def fit(split_seed: int, threads: int) -> np.ndarray:
        plan = make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                         unit="units", split_seed=split_seed, threads=threads)
        text = json.dumps([split_seed, "xgboost", "xgboost/1"], sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)
        assert plan.seed == int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        params = estimator_params(family, "regression", values, n_units=400, n_rows=400, y=y,
                                  plan=plan)
        assert params["random_state"] == plan.seed and params["n_jobs"] == threads
        return family.build("regression", "prediction", 400, 4).set_params(**params).fit(
            X, y).predict(X)

    for threads in (1, THREADS):
        first, again = fit(11, threads), fit(11, threads)
        assert np.max(np.abs(first - again)) <= family.replay_tolerance
        assert np.array_equal(first, again)
    assert not np.allclose(fit(11, THREADS), fit(12, THREADS))


# ── the shelf ────────────────────────────────────────────────────────────────


def test_its_assessment_is_boosted_trees_less_one_at_least_half() -> None:
    """Boosted trees score 3.0 from 2,000 rows, 1.0 from 200 and 0.5 below (by hand from
    ``boosted_trees.assess``); XGBoost reads 2.0, 0.5 and 0.5, with the same concerns."""
    xgboost, trees = get_family("xgboost"), get_family("boosted_trees")
    for rows, by_hand in ((2400, 2.0), (300, 0.5), (150, 0.5)):
        s = Situation("regression", "prediction", rows, 4)
        judged = assessment(xgboost, s)
        assert judged.score == by_hand
        assert judged.concerns == trees.assess(s).concerns
        assert xgboost.assess(s) == judged
        ranked = {f.key: a.score for f, a in rank(s)}
        assert ranked["xgboost"] == by_hand


# ── loading ──────────────────────────────────────────────────────────────────


def test_the_engine_imports_and_the_family_refuses_when_xgboost_cannot_load() -> None:
    """With xgboost unimportable, importing the engine registers every family, xgboost
    included; the other families fit; an XGBoost fit refuses, saying "XGBoost could not load"."""
    script = textwrap.dedent("""
        import sys
        sys.modules["xgboost"] = None  # import xgboost now raises ImportError
        import numpy as np
        from turbotab.core.models import get_family
        from turbotab.core.models.xgboost_family import XGBoostUnavailable, load_error
        family = get_family("xgboost")
        X, y = np.random.default_rng(0).normal(size=(50, 2)), np.arange(50) % 2
        get_family("linear").build("binary", "prediction", 50, 2).fit(X, y)
        try:
            family.build("binary", "prediction", 50, 2).fit(X, y)
        except XGBoostUnavailable as e:
            print("refused:", e)
        print("reason:", load_error())
    """)
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "refused: XGBoost could not load: ModuleNotFoundError" in out.stdout
    assert "reason: XGBoost could not load" in out.stdout
