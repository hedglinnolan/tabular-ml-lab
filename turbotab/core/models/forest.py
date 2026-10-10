"""``random_forest``: Breiman's random forest (scikit-learn's ``RandomForest*``), blanks native
(RECIPES_AND_TUNING §2.2, §4.1, §4.2; MODEL_FAMILY_CONTRACT §3.2; WAVE_C6A_PLAN §3 row RT-5d).

**Standard settings** (randomForest's and ranger's defaults, as RECIPES §2.2 adopts them): 500
trees, each on a bootstrap draw of every row (``replace=TRUE``, ``sampsize = nrow(x)``); the columns
each split tries, ``floor(sqrt(p))`` for classes and ``max(floor(p/3), 1)`` for a number
(randomForest's ``mtry``); the smallest leaf, 10 rows for classes, whose averaged class shares are
a probability forest's (ranger's ``min.node.size`` for ``probability = TRUE``), and 5 for a number
(randomForest's ``nodesize``). Two departures, stated: scikit-learn's regressor tries every column
at every split (``max_features=1.0``), which is bagging, not a random forest; and ranger's 10 is
the smallest node it splits, where here 10 is the smallest leaf, which is stricter.

**The estimator** is a thin subclass of scikit-learn's forest under the same class name (the
identity names it), with the same parameters and the same trees, that

* predicts single-threaded, so the trees' outputs are summed in tree order and a prediction is the
  same at every thread count (RECIPES §4.7); it fits on the plan's threads;
* for classes, adds a margin ``decision_function``: the log-odds of the probability of the class
  coded 1, the probability first held :func:`clip_share` (half of one tree's vote) away from 0
  and 1, so a region every tree calls pure has a finite log-odds (MODEL_FAMILY_CONTRACT C9);
* gives its out-of-bag predictions (:meth:`out_of_bag_prediction`), a row no tree left out marked
  blank rather than predicted as 0, for out-of-bag tuning (RECIPES §4.2: each candidate fit once
  and scored on the rows its trees did not draw, tuneRanger's method).

**Tuning** (RECIPES §4.1): a search over the share of columns per split, the share of rows each tree
draws (with replacement) and the smallest leaf, at most 8 Sobol candidates (the least tunable of
the tree families), scored out of bag where the engine allows it. Symbolic standards and the leaf's
share are resolved per fit by :meth:`RandomForest.settings`.

**Explanations.** ``trees`` gives the fitted trees as an ``explain.TreeEnsemble``: each tree's
leaf value divided by the number of trees, so the trees add up to the forest's prediction: the
number for a number, the probability of each class for classes (``scale="probability"``).
``tree_shap`` is the ``shap`` package's path-dependent TreeExplainer, which reads its inputs as
float32 exactly as the trees do. A yes/no forest's SHAP values are on its probability, the scale
its trees average, where path-dependent TreeSHAP is exact and adds up; its curves and interactions
are on the log-odds of the clipped probability, the scale every family's curves share; the
explanation labels each (``explain.explain``).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier as _SklearnClassifier
from sklearn.ensemble import RandomForestRegressor as _SklearnRegressor

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Situation,
    Source,
    register_family,
)
from turbotab.core.models.tuning import Dimension, TuningDecl, derive_seed

STANDARD = "default"  # a symbolic standard, resolved per task by ``settings``
THIRD = 1.0 / 3.0  # scikit-learn's max(1, int(p/3)): randomForest's max(floor(p/3), 1) for every p
LEAF = {"numbers": 5, "classes": 10}  # randomForest's nodesize; ranger's probability forest
SMALL_N, MID_N = 500, 2000  # RECIPES §2.6's rows for the forest's score


# ═════════════════════════════════════════════════════════════════════════════
# the estimator
# ═════════════════════════════════════════════════════════════════════════════


def clip_share(n_trees: int) -> float:
    """How far a probability is held from 0 and 1 before its log-odds is taken: half of one tree's
    vote, 1/(2T). At 500 trees that is 0.001, so the margin stays within ±6.9."""
    return 1.0 / (2 * max(int(n_trees), 1))


def _one_thread(model: Any) -> Any:
    """A shallow copy of the fitted ``model`` (the same trees) that predicts with one thread. The
    model itself is never changed, so two threads predicting with it at once neither race on its
    ``n_jobs`` nor see it change."""
    one = object.__new__(type(model))
    one.__dict__.update(model.__dict__)
    one.n_jobs = 1
    return one


def _never_left_out(model: Any) -> np.ndarray:
    """The training rows every tree drew (no out-of-bag prediction), from the forest's own record
    of each tree's bootstrap (``estimators_samples_``)."""
    left_out = np.zeros(int(model._n_samples), dtype=bool)
    for drawn in model.estimators_samples_:
        bag = np.zeros_like(left_out)
        bag[drawn] = True
        left_out |= ~bag
    return ~left_out


class RandomForestRegressor(_SklearnRegressor):
    """scikit-learn's random forest for a number, predicted single-threaded."""

    def predict(self, X: Any) -> np.ndarray:
        return _SklearnRegressor.predict(_one_thread(self), X)

    def out_of_bag_prediction(self) -> np.ndarray:
        """Each training row's mean prediction over the trees that left it out; blank (NaN) for a
        row every tree drew. Needs ``oob_score=True`` at the fit."""
        out = np.asarray(self.oob_prediction_, dtype=float).copy()
        out[_never_left_out(self)] = np.nan
        return out


class RandomForestClassifier(_SklearnClassifier):
    """scikit-learn's random forest for classes (its averaged class shares, a probability forest),
    predicted single-threaded, with a margin ``decision_function``."""

    def predict_proba(self, X: Any) -> np.ndarray:
        return _SklearnClassifier.predict_proba(_one_thread(self), X)

    def decision_function(self, X: Any) -> np.ndarray:
        """For a yes/no outcome, the log-odds of the clipped probability of the class coded 1,
        (n,); for several classes, the log of each clipped probability less their mean (logits
        that sum to zero), (n, K)."""
        eps = clip_share(len(self.estimators_))
        P = np.clip(self.predict_proba(X), eps, 1 - eps)
        if P.shape[1] == 2:
            return np.log(P[:, 1]) - np.log1p(-P[:, 1])
        L = np.log(P)
        return L - L.mean(axis=1, keepdims=True)

    def out_of_bag_prediction(self) -> np.ndarray:
        """Each training row's class probabilities over the trees that left it out, in
        ``classes_`` order; blank (NaN) for a row every tree drew. Needs ``oob_score=True``."""
        out = np.asarray(self.oob_decision_function_, dtype=float).copy()
        out[_never_left_out(self)] = np.nan
        return out


def out_of_bag_loss(model: Any, y: Any, *, task: str, loss: str, weights: Any = None) -> float:
    """The pooled out-of-bag loss of a fitted forest (``tuning.pooled_loss`` over the rows some tree
    left out): squared error, log loss or ranked probability score, as the plan's loss says."""
    from turbotab.core.models.tuning import pooled_loss

    prediction = model.out_of_bag_prediction()
    scored = np.isfinite(prediction if prediction.ndim == 1 else prediction[:, 0])
    w = None if weights is None else np.asarray(weights, dtype=float)[scored]
    classes = None if task == "regression" else list(model.classes_)
    return pooled_loss(task, loss, np.asarray(y)[scored], prediction[scored], classes=classes,
                       weights=w)


# ═════════════════════════════════════════════════════════════════════════════
# the trees, for TreeSHAP and the tree view
# ═════════════════════════════════════════════════════════════════════════════

NODE_DTYPE = np.dtype([
    ("value", np.float64), ("count", np.float64), ("feature_idx", np.int64),
    ("num_threshold", np.float64), ("missing_go_to_left", np.uint8), ("left", np.int64),
    ("right", np.int64), ("depth", np.int64), ("is_leaf", np.uint8)])


def float64_threshold(threshold: np.ndarray) -> np.ndarray:
    """Thresholds that route a float64 input as scikit-learn's trees route it.

    The trees read their inputs as float32: a row goes left when ``float32(x) <= t``. Rounding to
    float32 is monotone, so that holds exactly when ``x <= t'``, t' the largest float64 that rounds
    to a float32 at most t: with f the largest float32 at most t and m the midpoint between f and
    the next float32 (exact in float64), t' is m when m rounds down to f (ties go to the even
    one) and the float64 just below m otherwise. A float32 input routes alike under t and t', so
    the shap package's values do not move; a float64 input just above t that rounds to t or below
    now goes left, as it does in the forest."""
    t = np.asarray(threshold, dtype=np.float64)
    out = t.copy()
    finite = np.isfinite(t) & (np.abs(t) < float(np.finfo(np.float32).max))
    tf = t[finite]
    f = tf.astype(np.float32)
    f = np.where(f.astype(np.float64) > tf, np.nextafter(f, np.float32(-np.inf)), f)
    up = np.nextafter(f, np.float32(np.inf))
    m = (f.astype(np.float64) + up.astype(np.float64)) / 2.0
    out[finite] = np.where(m.astype(np.float32).astype(np.float64) <= tf, m,
                           np.nextafter(m, -np.inf))
    return out


def _node_table(tree: Any, values: np.ndarray) -> np.ndarray:
    """One fitted tree's nodes in ``explain.TreeEnsemble``'s field names: ``values`` per node, the
    bootstrap-weighted training cover (``weighted_n_node_samples``, as the shap package reads it),
    and a row going left when its value is ``<=`` the threshold (:func:`float64_threshold`, so a
    float64 input goes the way the tree sends its float32), a blank where ``missing_go_to_left``
    says."""
    left, right = tree.children_left, tree.children_right
    depth = np.zeros(tree.node_count, dtype=np.int64)
    for i in range(tree.node_count):  # a child always comes after its parent
        if left[i] >= 0:
            depth[left[i]] = depth[right[i]] = depth[i] + 1
    nodes = np.zeros(tree.node_count, dtype=NODE_DTYPE)
    nodes["value"] = values
    nodes["count"] = tree.weighted_n_node_samples
    nodes["feature_idx"] = np.maximum(tree.feature, 0)
    nodes["num_threshold"] = np.where(left >= 0, float64_threshold(tree.threshold), tree.threshold)
    nodes["missing_go_to_left"] = np.asarray(tree.missing_go_to_left, dtype=np.uint8)
    nodes["left"], nodes["right"] = left, right
    nodes["depth"] = depth
    nodes["is_leaf"] = left < 0
    return nodes


def forest_ensemble(model: Any) -> Any:
    """A fitted forest as an ``explain.TreeEnsemble``: each tree's leaf value divided by the number
    of trees, so that the trees add up to the forest's prediction. A number: one output, on its own
    scale (``"margin"``). Classes: each tree's class shares (``predict_proba`` of the tree), on the
    ``"probability"`` scale, one output for a yes/no outcome (the class coded 1) and one per class
    for several."""
    from turbotab.core.models.explain import TreeEnsemble, leaf_paths

    T = len(model.estimators_)
    classes = is_classifier(model)
    outputs = ([1] if len(model.classes_) == 2 else list(range(len(model.classes_)))
               ) if classes else [0]
    tables: list[list[np.ndarray]] = [[] for _ in outputs]
    for estimator in model.estimators_:
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        if classes:
            value = value / value.sum(axis=1, keepdims=True)
        for slot, k in enumerate(outputs):
            tables[slot].append(_node_table(tree, value[:, k] / T))
    trees = [[leaf_paths(nodes) for nodes in per_output] for per_output in tables]
    return TreeEnsemble(base=np.zeros(len(outputs)), trees=trees, tables=tables, rule="le",
                        scale="probability" if classes else "margin")


def compiled_tree_shap(model: Any, Z: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """The shap package's path-dependent TreeSHAP of a fitted forest, ``(phi (n, p, K), expected
    (K,))`` on :func:`forest_ensemble`'s outputs; None when the package is not installed (the
    engine's own TreeSHAP is used then)."""
    try:
        import shap
    except ImportError:
        return None
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(np.asarray(Z, dtype=float), check_additivity=False)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    expected = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if values.ndim == 2:
        values = values[:, :, None]
    if is_classifier(model) and len(model.classes_) == 2:
        values, expected = values[:, :, 1:], expected[1:]
    return values.astype(float), expected


# ═════════════════════════════════════════════════════════════════════════════
# the family
# ═════════════════════════════════════════════════════════════════════════════

_LEAF = Dimension("min_samples_leaf", "the smallest leaf, as a share of the rows",
                  "minimum node size", 0.0, 0.1, "share_of_units",
                  source="Probst, Boulesteix & Bischl 2019; Kruppa et al. 2014")
_TUNING = TuningDecl(
    "search",
    dimensions=(
        Dimension("max_features", "the share of columns each split tries", "mtry", 0.05, 1.0,
                  "linear", source="Probst, Wright & Boulesteix 2019"),
        Dimension("max_samples", "the share of rows each tree draws", "sample fraction", 0.2,
                  1.0, "linear", source="Probst, Wright & Boulesteix 2019"),
        _LEAF,
    ),
    standard={"max_features": STANDARD, "max_samples": None, "min_samples_leaf": STANDARD},
    standard_source="randomForest's and ranger's defaults",
    fixed={"n_estimators": 500, "bootstrap": True},
    out_of_bag=True,
    space_version="random_forest/1",
    reason="Forests gain little from tuning, so at most 8 settings are tried, each scored on the "
           "rows its trees left out.",
    structural=("criterion",),
    max_drawn=8,
)


def leaf_rows(leaf: Any, *, n_rows: int) -> int:
    """The smallest leaf, in rows of one fit (RECIPES §4.1; WAVE_C6A_PLAN §5.2).

    A share (a float in (0, a tenth]), searched or set by hand, is the share applied: it was drawn
    log-uniformly from one unit of the plan, ``1/n_plan``, to a tenth (``map_unit``) and is stored
    as it is, so the record's share and the fit's leaf agree with or without a plan. It becomes
    ``max(1, round(share × n_rows))`` rows of this fit. A whole number is a count of rows already,
    as scikit-learn reads an integer ``min_samples_leaf``, so resolved parameters fed back in are
    kept. Anything else is refused, in words."""
    if isinstance(leaf, (bool, np.bool_)):
        raise ValueError(f"the smallest leaf is a share of the rows or a number of rows, not "
                         f"{leaf!r}")
    if isinstance(leaf, (int, np.integer)):
        if leaf < 1:
            raise ValueError(f"the smallest leaf holds at least 1 row, not {int(leaf)}")
        return int(leaf)
    s = float(leaf)
    high = float(_LEAF.high)
    if not (0 < s <= high * (1 + 1e-12)):
        raise ValueError(f"the smallest leaf, as a share of the rows, is above 0 and at most "
                         f"{high:g}, not {s:g}")
    return max(1, int(math.floor(min(s, high) * int(n_rows) + 0.5)))


def seed_of(plan: Any = None) -> int:
    """The forest's ``random_state``: the plan's seed, ``derive_seed(split seed, "random_forest",
    space version)`` (RECIPES §4.7), so it is threaded from the split; without a plan, the same
    derivation at split seed 0, the seed a plan made at split seed 0 gives."""
    if plan is not None:
        return int(plan.seed)
    return derive_seed(0, RandomForest.key, _TUNING.space_version)


class RandomForest(FamilyBase):
    key = "random_forest"
    label = "Random forest"
    inductive_bias = ("Many deep trees, each on resampled rows with random columns, averaged; "
                      "effects are step functions.")
    strengths = (
        "Finds curves and interactions without being told.",
        "Needs little tuning: its standard settings are usually close to its best.",
        "Uses rows with missing values as they are.",
    )
    cautions = (
        "No coefficients: effects are read from curves.",
        "Validated by cross-validation: Harrell's bootstrap overstates a near-interpolating "
        "learner's performance.",
    )
    needs_scaling = False
    handles_missing = True
    # Its trees nearly memorize their rows, so the bootstrap's optimism is understated
    # (MODEL_FAMILY_CONTRACT §3.2, C11: not sound).
    bootstrap_optimism = False
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="RandomForestRegressor; RandomForestClassifier",
                        seed_policy="random_state: the plan's seed, derive_seed(split seed, "
                                    "family, space version); fit on the plan's threads, "
                                    "predicted single-threaded")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = True
    inference_decl = InferenceDecl(table="description_only")
    invariances = ("monotone_per_column",)
    bias_terms = (Named(plain="Choosing from a random handful of columns at each split is this "
                              "forest's penalty knob.",
                        known_as="randomization as regularization",
                        source=Source("mentch2020randomization")),)
    curve_shape = "piecewise_constant"
    complexity = (
        Knob(setting="max_features", more_means="more_flexible",
             source=Source("mentch2020randomization")),
        Knob(setting="min_samples_leaf", more_means="simpler", source=Source("curth2024forests")),
    )
    output = "probability"
    raw_scale = CLASS_SCALES
    attribution = "trees"
    architecture = ("trees",)
    review_lenses = ("shared",)
    sources = (Source("breiman2001forests"), Source("liaw2002randomforest"),
               Source("wright2017ranger"), Source("probst2019rf"))
    tuning = _TUNING
    consequence = ("Many deep trees averaged: finds curves and interactions with little tuning; "
                   "gives no coefficients.")

    def methods_label(self, task: Task | None) -> str:
        return "random forest"

    def settings(self, values: Mapping[str, Any], *, task: str, n_units: int, n_rows: int,
                 y: Any = None, Z: Any = None, plan: Any = None) -> dict[str, Any]:
        """A candidate's values as the forest's parameters on one fit: the symbolic standards per
        task (``max_features``: ``"sqrt"`` for classes, a third of the columns for a number;
        ``min_samples_leaf``: 10 or 5 rows), the leaf's share as rows (:func:`leaf_rows`), the
        plan's seed (:func:`seed_of`), the plan's threads for the fit, and the out-of-bag
        predictions kept when the plan scores candidates on them. Its output fed back in gives
        itself."""
        out = dict(values)
        classes = task != "regression"
        if out.get("max_features") == STANDARD:
            out["max_features"] = "sqrt" if classes else THIRD
        leaf = out.get("min_samples_leaf")
        if isinstance(leaf, str) and leaf == STANDARD:
            out["min_samples_leaf"] = LEAF["classes" if classes else "numbers"]
        elif leaf is not None:
            out["min_samples_leaf"] = leaf_rows(leaf, n_rows=n_rows)
        out["random_state"] = seed_of(plan)
        out["n_jobs"] = int(plan.threads) if plan is not None else 1
        out["oob_score"] = bool(plan is not None and plan.out_of_bag)
        return out

    def trees(self, step: Any) -> Any:
        """Its fitted model step's trees, for TreeSHAP and the tree view (C10)."""
        return forest_ensemble(step)

    def tree_shap(self, step: Any, Z: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """The shap package's TreeSHAP of its fitted step (C10; RECIPES RT-5d)."""
        return compiled_tree_shap(step, Z)

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        from turbotab.core.models.tuning import estimator_params

        params = estimator_params(self, task, _TUNING.standard, n_units=n_rows, n_rows=n_rows)
        cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
        return cls(**params)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        if task == "regression":
            return ("Random forest",
                    "500 trees, each on a bootstrap draw of the rows; each split tries a third of "
                    "the columns, and every leaf holds at least 5 rows.")
        return ("Random forest",
                "500 trees, each on a bootstrap draw of the rows; each split tries the square root "
                "of the number of columns, and every leaf holds at least 10 rows; the trees' class "
                "shares are averaged into probabilities.")

    def assess(self, s: Situation) -> Assessment:
        """RECIPES §2.6: 2.0 from 2,000 training rows, 1.5 from 500, "fair" at 1.0 below; "fair" at
        1.5 at most when predictors outnumber rows; under inference, as boosted trees (conventions:
        no validated rule says which family will predict best)."""
        concerns: list[str] = []
        fit = "good"
        score = 2.0 if s.n_rows >= MID_N else 1.5
        if s.n_rows < SMALL_N:
            concerns.append(f"{s.n_rows:,} rows is small for a random forest: its cross-validated "
                            f"score is noisy.")
            fit, score = "fair", 1.0
        if s.n_features > s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: most columns a "
                            f"split tries will be noise.")
            fit, score = "fair", min(score, 1.5)
        if s.purpose == "inference":
            concerns.append("No coefficients or intervals: effects are read from curves instead.")
            fit = "fair" if fit == "good" else fit
            score = min(score, 1.0)
        return Assessment(score, fit, tuple(concerns))


RANDOM_FOREST = register_family(RandomForest())
