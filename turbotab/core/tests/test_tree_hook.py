"""RT-1a · the tree hook: explanations read a tree family's ``trees`` and ``tree_shap`` members, not
scikit-learn's histogram-boosting internals (RECIPES F8; MODEL_FAMILY_CONTRACT C10;
WAVE_C6A_PLAN §2).

The references are independent of the hook:

* **before and after, bit for bit:** the code as it was before the hook, copied here, reads the
  fitted model's ``_predictors`` directly; the boosted trees' SHAP values, expected value and tree
  view through the hook must equal it exactly (``tests/acceptance/test_explain.py`` keeps the
  check of the values against the ``shap`` package);
* **the split rule:** a one-split tree worked by hand, with a row on the threshold, which goes left
  under scikit-learn's ``<=`` and right under XGBoost's ``<``;
* **a compiled TreeSHAP and the scale:** a probe family whose ``tree_shap`` returns stated values;
* **"shrunk to zero":** said by the family's declared architecture, for a fitted step that has no
  ``alpha_`` of its own.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import explain as E
from turbotab.core.models.base import get_family, register_family, unregister_family
from turbotab.core.models.boosted_trees import BoostedTrees


def _fitted(kind: str) -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(600, 4)), columns=["a", "b", "c", "d"])
    X = X.mask(rng.random(X.shape) < 0.08)
    f = X["a"].fillna(0) * 2 + np.sin(X["b"].fillna(0)) + (X["c"] * X["d"]).fillna(0)
    y = f + rng.normal(size=len(X))
    if kind == "regression":
        model, target = HistGradientBoostingRegressor(max_iter=30, random_state=0), y
    else:
        model, target = HistGradientBoostingClassifier(max_iter=30, random_state=0), (y > 0.3)
    pipe = Pipeline([("model", model)]).set_output(transform="pandas")
    return pipe.fit(X, np.asarray(target).astype(float if kind == "regression" else int)), X


# ── the code as it was before the hook (turbotab-next @ 553f6f27), copied ────


def _old_ensemble(model: Any) -> E.TreeEnsemble:
    base = np.atleast_1d(np.asarray(model._baseline_prediction, dtype=float).ravel())
    outputs = len(model._predictors[0])
    trees: list[list[E.LeafPaths]] = [[] for _ in range(outputs)]
    for iteration in model._predictors:
        for k, predictor in enumerate(iteration):
            trees[k].append(E.leaf_paths(predictor.nodes))
    return E.TreeEnsemble(base=base, trees=trees)


def _old_tree_structure(anat: E.Anatomy, columns: list[str], levels: int = E.TREE_LEVELS
                        ) -> E.TreeStructure:
    model = anat.model
    trees = [p[0].nodes for p in model._predictors]
    first = trees[0]
    shown = []
    queue = [0]
    while queue:
        i = queue.pop(0)
        node = first[i]
        depth = int(node["depth"])
        leaf = bool(node["is_leaf"])
        expand = not leaf and depth + 1 < levels
        column = None if leaf else str(columns[int(node["feature_idx"])])
        shown.append(E.TreeNode(
            id=i, depth=depth, column=column,
            input=None if column is None else anat.group.get(column, column),
            threshold=None if leaf else float(node["num_threshold"]),
            blanks=None if leaf else ("left" if bool(node["missing_go_to_left"]) else "right"),
            n=int(node["count"]), value=float(node["value"]) if leaf else None,
            left=int(node["left"]) if expand else None,
            right=int(node["right"]) if expand else None))
        if expand:
            queue.extend([int(node["left"]), int(node["right"])])
    root: dict[str, list[float]] = defaultdict(list)
    splits: dict[str, int] = defaultdict(int)
    for nodes in trees:
        for node in nodes:
            if bool(node["is_leaf"]) or int(node["depth"]) >= levels:
                continue
            name = anat.group.get(str(columns[int(node["feature_idx"])]),
                                  str(columns[int(node["feature_idx"])]))
            splits[name] += 1
            if int(node["depth"]) == 0:
                root[name].append(float(node["num_threshold"]))
    counts = [E.SplitCount(input=name, root=len(root.get(name, [])), splits=n,
                           median_threshold=float(np.median(root[name])) if root.get(name)
                           else None)
              for name, n in splits.items()]
    counts.sort(key=lambda s: (-s.root, -s.splits, s.input))
    return E.TreeStructure(n_trees=len(trees), levels=levels, first_tree=shown, splits=counts)


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_boosted_trees_shap_is_bit_equal_through_the_hook(kind: str) -> None:
    pipe, X = _fitted(kind)
    anat = E.anatomy(pipe, list(X.columns), kind)
    old_phi, old_expected = E.tree_shap(_old_ensemble(pipe[-1]), X.to_numpy(dtype=float))
    found = E.attributions(anat, X, X, kind="trees", family="boosted_trees")
    assert found.scale == "margin"
    assert np.array_equal(found.phi.to_numpy(), old_phi[:, :, 0])
    assert found.expected == float(old_expected[0])
    # the family's own ensemble carries scikit-learn's node tables, tree for tree
    ensemble = get_family("boosted_trees").trees(pipe[-1])
    assert (ensemble.rule, ensemble.scale) == ("le", "margin")
    assert len(ensemble.tables[0]) == len(pipe[-1]._predictors)
    assert all(np.array_equal(t, p[0].nodes) for t, p in zip(ensemble.tables[0],
                                                             pipe[-1]._predictors))


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_the_tree_view_is_bit_equal_through_the_hook(kind: str) -> None:
    pipe, X = _fitted(kind)
    anat = E.anatomy(pipe, list(X.columns), kind)
    columns = [str(c) for c in X.columns]
    assert (E.tree_structure(anat, columns, family="boosted_trees").model_dump()
            == _old_tree_structure(anat, columns).model_dump())


# ── the split rule, by hand ───────────────────────────────────────────────────

_NODE = np.dtype([("value", "f8"), ("count", "f8"), ("feature_idx", "i8"),
                  ("num_threshold", "f8"), ("missing_go_to_left", "u1"), ("left", "i8"),
                  ("right", "i8"), ("depth", "i8"), ("is_leaf", "u1")])


def _stump() -> np.ndarray:
    """A split on column 0 at 1.0: 30 training rows went left (value 10), 70 right (value 20);
    a blank goes left."""
    return np.array([(0.0, 100, 0, 1.0, 1, 1, 2, 0, 0),
                     (10.0, 30, 0, 0.0, 0, 0, 0, 1, 1),
                     (20.0, 70, 0, 0.0, 0, 0, 0, 1, 1)], dtype=_NODE)


@pytest.mark.parametrize("rule,on_threshold", [("le", -7.0), ("lt", 3.0)])
def test_the_split_rule_decides_where_the_threshold_goes(rule: str, on_threshold: float) -> None:
    """E[f] = 0.3·10 + 0.7·20 = 17. With one split the only feature's Shapley value is f(x) − 17:
    −7 on the left, 3 on the right; the other column's is 0. The row at 1.0 goes left under
    ``<=`` and right under ``<``; a blank goes left under both."""
    nodes = _stump()
    ensemble = E.TreeEnsemble(base=np.zeros(1), trees=[[E.leaf_paths(nodes)]], tables=[[nodes]],
                              rule=rule)
    X = np.array([[1.0, 5.0], [0.5, 5.0], [2.0, 5.0], [np.nan, 5.0]])
    phi, expected = E.tree_shap(ensemble, X)
    assert expected.tolist() == pytest.approx([17.0], abs=1e-12)
    assert phi[:, 0, 0].tolist() == pytest.approx([on_threshold, -7.0, 3.0, -7.0], abs=1e-12)
    assert phi[:, 1, 0].tolist() == [0.0, 0.0, 0.0, 0.0]


# ── a compiled TreeSHAP, and a probability scale ──────────────────────────────


def test_a_compiled_tree_shap_replaces_the_engines_and_a_probability_keeps_its_scale() -> None:
    pipe, X = _fitted("binary")
    anat = E.anatomy(pipe, list(X.columns), "binary")
    stated = np.arange(X.size, dtype=float).reshape(len(X), X.shape[1], 1)

    def compiled(self: Any, step: Any, Z: Any) -> Any:
        assert step is pipe[-1] and Z.shape == X.shape
        return stated, np.array([0.25])

    def as_probability(self: Any, step: Any) -> Any:
        return replace(BoostedTrees.trees(self, step), scale="probability")

    register_family(type("Compiled", (BoostedTrees,), {
        "key": "compiled_probe", "tree_shap": compiled, "trees": as_probability})())
    try:
        found = E.attributions(anat, X, X, kind="trees", family="compiled_probe")
        assert np.array_equal(found.phi.to_numpy(), stated[:, :, 0])
        assert (found.expected, found.scale) == (0.25, "probability")
        # a class probability is drawn for a yes/no outcome only
        reg, Xr = _fitted("regression")
        assert E.attributions(E.anatomy(reg, list(Xr.columns), "regression"), Xr, Xr,
                              kind="trees", family="compiled_probe") is None
    finally:
        unregister_family("compiled_probe")
    s = E.Setting(task="binary", purpose="prediction", target="y", event="case", X=X,
                  y=np.zeros(len(X)))
    assert E.probability_of(s) == "probability of `case`"


# ── "shrunk to zero" is the architecture's to say ─────────────────────────────


def test_a_zero_is_shrunk_to_zero_when_the_family_declares_shrinkage() -> None:
    """A lasso step has no ``alpha_`` (only the CV estimators do), so the old ``hasattr`` read
    every zero as "of zero"; the declared architecture says which it is."""
    from sklearn.linear_model import Lasso

    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = 3 * X["a"] + 0.01 * X["b"] + rng.normal(size=200)
    pipe = Pipeline([("model", Lasso(alpha=0.5))]).set_output(transform="pandas").fit(X, y)
    assert int(np.sum(pipe[-1].coef_ == 0)) == 2 and not hasattr(pipe[-1], "alpha_")
    anat = E.anatomy(pipe, list(X.columns), "regression")

    def text(family: str) -> str:
        return E.linear_equation(anat, X, target="y", scale="predicted `y`", outcome_unit=None,
                                 units={}, roles={}, family=family).text

    assert "shrinkage" in get_family("elastic_net").architecture
    assert text("elastic_net").endswith("; 2 coefficients shrunk to zero")
    assert text("linear").endswith("; 2 coefficients of zero")
