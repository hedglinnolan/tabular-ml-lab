"""MC-1 and MC-2a · every model family declares the contract, and the switches the new families hit
read those declarations (MODEL_FAMILY_CONTRACT §1, §3.1, §3.3, §5).

Expected values: §3.1's table of today's nine families and §3.2's of the four v2 adds, written out
here; the tables and labels the retired switches held, copied from the code before MC-2a; ridge's
hat matrix by explicit inverse (ESL §3.4.1) against scikit-learn's own elastic net; and the citation
registry's verified records.

**Rules, not a roster** (WAVE_C6A_PLAN §3, RT-1a): a family that registers later in this wave (ridge,
Huber, the random forest, XGBoost) meets these tests as it stands, with no edit here. What each
family declares is checked against its row below, and what it adds is checked by the rules that tie
a member to a declaration: ``trees`` is set exactly when the attribution or the architecture reads
trees (TreeSHAP or the tree view), ``path`` exactly when a task's tuning is a path, ``inference``
exactly when the table has intervals.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import get_family
from turbotab.core.models.base import (
    CONSEQUENCE_WORDS,
    TASKS,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Source,
    contract_problems,
    families,
    inference_default,
    info,
    register_family,
)

REPO = Path(__file__).resolve().parents[4]
CONTRACT = REPO / "docs" / "turbotab-next" / "MODEL_FAMILY_CONTRACT.md"

# §3.1, by clause: C3's invariance, C2's inference table (description only for boosted trees, which
# gives curves and no table under inference, as §3.2 declares the forest), C11's bootstrap
# soundness (not applicable to the feature-wise tests, which make no predictions) and flexibility,
# and C10's explanation path today.
EXPECTED = {
    "linear": (("linear_maps",), "intervals", True, False, "linear"),
    "elastic_net": (("column_scale",), "shrunk_no_intervals", True, False, "linear"),
    "boosted_trees": (("monotone_per_column",), "description_only", False, True, "trees"),
    "featurewise": ((), "intervals", None, False, "none"),
    "proportional_odds": (("linear_maps",), "intervals", True, False, "none"),
    "mixed": (("linear_maps",), "intervals", True, False, "none"),
    "gee": (("linear_maps",), "intervals", True, False, "none"),
    "cox": (("linear_maps",), "intervals", True, False, "none"),
    "screened_elastic_net": (("column_scale",), None, True, False, "linear"),
    # §3.2: the four v2 families, by the keys RECIPES §2.2 registers them under. Ridge gives a
    # shrunk table under inference; Huber is for prediction only; the forest and XGBoost describe.
    "ridge": (("rotation_after_scaling",), "shrunk_no_intervals", True, False, "linear"),
    "huber": (("linear_maps",), None, True, False, "linear"),
    "random_forest": (("monotone_per_column",), "description_only", False, True, "trees"),
    "xgboost": (("monotone_per_column",), "description_only", False, True, "trees"),
}
# Today's nine (§3.1), every one registered; §3.2's four join as their packages land.
NINE = ("linear", "elastic_net", "boosted_trees", "featurewise", "proportional_odds", "mixed",
        "gee", "cox", "screened_elastic_net")


def _families() -> dict[str, Any]:
    from turbotab.core.contracts import contracts

    contracts()  # the omics chain registers the screened elastic net
    return {f.key: f for f in families()}


def test_each_family_declares_what_sections_3_1_and_3_2_say():
    found = _families()
    assert set(NINE) <= set(found) <= set(EXPECTED), sorted(set(found) - set(EXPECTED))
    for key, f in found.items():
        invariances, table, sound, flexible, attribution = EXPECTED[key]
        assert contract_problems(f) == [], key
        decl = f.inference_decl
        declared = (tuple(f.invariances), decl.table if decl else None, f.bootstrap_optimism,
                    f.flexible, f.attribution)
        assert declared == (invariances, table, sound, flexible, attribution), key


def test_the_declarations_say_what_the_tables_they_will_replace_say():
    """Until MC-2b moves these reads onto the declarations, each declaration agrees with the table
    or the check it replaces, so the move changes nothing."""
    from turbotab.core.methods.interaction import SUPPORTED
    from turbotab.core.models.survey import has_design_estimator
    from turbotab.core.reference.catalog import FAMILY_LENSES
    from turbotab.core.stages.effects import SEQUENCE_FAMILIES

    for f in _families().values():
        decl = f.inference_decl or InferenceDecl(table="description_only")
        assert tuple(f.review_lenses) == FAMILY_LENSES[f.key], f.key
        assert decl.product_terms == (f.key in SUPPORTED), f.key
        assert decl.matrix_table == (f.key in SEQUENCE_FAMILIES), f.key
        assert decl.design_based == any(has_design_estimator(f, t) for t in f.tasks), f.key
    # MC-2a: stages/scales.py's FAMILY_FOR, now each family's default_for
    assert {t: getattr(inference_default(t), "key", None) for t in TASKS} == {
        "regression": "linear", "binary": "linear", "multiclass": None,
        "ordinal": "proportional_odds", "time_to_event": None}


def test_the_methods_text_names_each_family_as_it_did():
    """MC-2a: voice._FAMILY_LABEL and _LINEAR_LABEL, copied from the code they left, are now each
    family's methods_label; the screened elastic net, which had no entry and fell back to its key,
    is named."""
    from turbotab.core.voice import _family_label

    _families()
    before = {
        ("linear", "regression"): "linear regression",
        ("linear", "binary"): "logistic regression",
        ("linear", "multiclass"): "multinomial logistic regression",
        ("linear", "ordinal"): "multinomial logistic regression, which ignores the levels' order",
        ("linear", None): "a linear model",
        ("elastic_net", "binary"): "elastic net",
        ("boosted_trees", "regression"): "gradient-boosted trees",
        ("featurewise", "regression"):
            "feature-wise least-squares tests with Benjamini–Hochberg false-discovery control",
        ("proportional_odds", "ordinal"): "a proportional-odds (cumulative logit) model",
        ("mixed", "regression"): "a random-intercept mixed model",
        ("gee", "binary"): "generalized estimating equations",
        ("cox", "time_to_event"): "Cox proportional hazards",
        ("screened_elastic_net", "regression"): "screened elastic net",
        ("no_such_family", "regression"): "`no_such_family`",
    }
    assert {k: _family_label(*k, None) for k in before} == before
    assert _family_label("linear", "binary", {"model_labels": {"linear": "my model"}}) == "my model"


def test_flexible_is_declared_not_derived():
    """MC-2a: models/selection.py:is_flexible reads ``flexible``; it no longer falls back to
    ``not bootstrap_optimism``."""
    from turbotab.core.models.selection import is_flexible

    assert {k: is_flexible(f) for k, f in _families().items()} == {
        k: EXPECTED[k][3] for k in _families()}
    assert not is_flexible(SimpleNamespace(flexible=False, bootstrap_optimism=False))
    assert is_flexible(SimpleNamespace(flexible=True, bootstrap_optimism=True))


def test_explanations_follow_the_declared_attribution():
    """MC-2a: models/explain.py reads ``attribution`` instead of the estimator's class name. The
    reference is what the class-name gate gave for the same fitted steps: linear SHAP for a
    logistic regression, TreeSHAP for histogram boosting, nothing for a step wrapped by Explore's
    imbalance correction or for a family that declares no attribution."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.methods.levers import ImbalanceCorrected
    from turbotab.core.models import explain as E

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    yes = (X["a"] + rng.normal(size=120) > 0).astype(int).to_numpy()

    def kind(key: str, model: Any, task: str, y: Any) -> str | None:
        pipe = Pipeline([("model", model)]).set_output(transform="pandas").fit(X, y)
        return E.model_kind(key, E.anatomy(pipe, list(X.columns), task))

    assert kind("linear", LogisticRegression(), "binary", yes) == "linear"
    trees = HistGradientBoostingClassifier(max_iter=5)
    assert kind("boosted_trees", trees, "binary", yes) == "trees"
    assert kind("linear", ImbalanceCorrected(LogisticRegression()), "binary", yes) is None
    assert kind("mixed", LinearRegression(), "regression", X["b"].to_numpy()) is None


def test_the_family_info_carries_the_user_facing_declarations():
    trees = info(get_family("boosted_trees"))
    assert (trees.flexible, trees.bootstrap_optimism, trees.inference_table, trees.invariances) \
        == (True, False, "description_only", ["monotone_per_column"])
    assert [(t.known_as, t.source.key) for t in trees.bias_terms] == [("gradient boosting",
                                                                         "friedman2001")]
    assert info(get_family("linear")).inference_table == "intervals"


def test_every_cited_source_is_a_verified_record():
    """Every source key a declaration cites is one of ``models.sources``, and each of those is a
    record of SIZING X4's citation registry (every DOI checked against Crossref) whose reference
    names that record. A source joins before the family that cites it lands only when the recipes
    the family is built from cite it (RECIPES_AND_TUNING), so no key is an orphan. Each
    dimension's source string resolves in the same registry (C6)."""
    from turbotab.core.export.citations import is_internal, registry, resolve, segments
    from turbotab.core.models.base import _declared_tunings
    from turbotab.core.models.sources import SOURCES

    records = registry()
    assert sorted(k for k in SOURCES if k not in records) == []
    assert sorted(k for k, ref in SOURCES.items() if k not in resolve(ref, records)) == []
    families = _families().values()
    cited = {s.key for f in families
             for s in [*f.sources, *(t.source for t in f.bias_terms),
                       *(k.source for k in f.complexity if k.source)]}
    assert sorted(cited - set(SOURCES)) == []
    recipes = (CONTRACT.parent / "RECIPES_AND_TUNING.md").read_text(encoding="utf-8")
    assert sorted(set(SOURCES) - cited - set(resolve(recipes, records))) == []
    unresolved = [(f.key, d.name, part) for f in families
                  for decl in _declared_tunings(f)[0].values()
                  for d in (*decl.dimensions, *decl.by_hand) for part in segments(d.source)
                  if not is_internal(part) and not resolve(part)]
    assert unresolved == []


def test_the_elastic_net_ridge_part_is_its_hat_matrix():
    """C7: the elastic net's declared penalty knob names ``elastic_net_ridge_part``. At l1_ratio 0
    scikit-learn's elastic net is its ridge part alone, so its fit is the hat matrix
    Z(ZᵀZ + κI)⁻¹Zᵀ at κ = nα, built here by explicit inverse; df is that matrix's trace
    (ESL §3.4.1, eq. 3.50), to 1e-10."""
    from sklearn.linear_model import ElasticNet

    from turbotab.core.models.formulas import FORMULAS

    from turbotab.core.models.base import _declared_tunings

    family = get_family("elastic_net")
    (knob,) = [k for k in family.complexity if k.formula]
    # The knob names the penalty the formula reads: the CV grid ``alpha_`` comes from, or a path
    # dimension once its tuning declares one (RT-5f).
    path = {d.name for decl in _declared_tunings(family)[0].values() if decl.kind == "path"
            for d in decl.dimensions}
    assert knob.setting in ({"alphas"} if not path else path) and knob.more_means == "simpler"
    formula = FORMULAS[knob.formula]
    rng = np.random.default_rng(11)
    n, p = 60, 6
    Z = rng.normal(size=(n, p)) @ rng.normal(size=(p, p))
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    y = Z @ rng.normal(size=p) + rng.normal(size=n)

    def hat(kappa: float) -> np.ndarray:
        return Z @ np.linalg.inv(Z.T @ Z + kappa * np.eye(p)) @ Z.T

    alpha = 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scikit-learn advises against l1_ratio 0 by name
        fit = ElasticNet(alpha=alpha, l1_ratio=0.0, tol=1e-14, max_iter=1_000_000).fit(Z, y)
    assert np.allclose(fit.predict(Z), y.mean() + hat(n * alpha) @ (y - y.mean()), atol=1e-8)
    assert formula(Z, alpha, 0.0).df == pytest.approx(np.trace(hat(n * alpha)), abs=1e-10)
    # with a lasso part: the ridge part's κ is nα(1 − ρ)
    assert formula(Z, alpha, 0.75).df == pytest.approx(np.trace(hat(n * alpha * 0.25)), abs=1e-10)
    # no penalty keeps every pattern: df is the rank
    assert formula(Z, alpha, 1.0).df == pytest.approx(p)


# ── register_family refuses a family that breaks the contract, naming what ───


class _Bare(FamilyBase):
    key = "bare_probe"
    label = "Bare probe"
    inductive_bias = "Straight lines."

    def build(self, task: Any, purpose: Any, n_rows: int, n_features: int) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()

    def describe(self, task: Any, purpose: Any) -> tuple[str, str]:
        return "Probe", "A probe."

    def assess(self, situation: Any) -> Any:
        raise NotImplementedError


def test_a_family_missing_a_declaration_is_refused_by_name():
    with pytest.raises(TypeError, match=r"bare_probe: the model-family contract asks for "
                                        r"identity, purposes, predicts, flexible, "
                                        r"bootstrap_optimism, methods_label, which it does not "
                                        r"declare"):
        register_family(_Bare())
    assert "bare_probe" not in {f.key for f in families()}


def _decl(name: str, low: float, high: float, scale: str, *, kind: str = "search",
          points: int = 0, source: str = "Friedman 2001") -> Any:
    from turbotab.core.models.tuning import Dimension, TuningDecl

    dim = Dimension(name, f"plain words for {name}", f"term for {name}", low, high, scale,
                    source=source, points=points)
    return TuningDecl(kind, dimensions=(dim,), space_version=f"probe/{name}",
                      standard={} if kind == "path" else {name: low})


def _probe(key: str = "probe", **declared: Any) -> Any:
    """The linear family's declarations with ``declared`` changed, under a key of its own and
    default for no task (each task has one default inference family, and linear is it)."""
    from turbotab.core.models.linear import Linear

    decl = replace(Linear.inference_decl, default_for=())
    return type("Probe", (Linear,), {"key": key, "inference_decl": decl, **declared})()


@pytest.mark.parametrize("declared,says", [
    ({"invariances": ("rotation",)}, "invariances ['rotation'] are not among"),
    ({"identity": Identity("trained_network", "torch", "MLP")},
     "must declare learning_rule, initialization, parameterization, seed_policy (C1)"),
    ({"identity": Identity("pretrained_prior", "tabpfn", "TabPFNClassifier", prior="v2")},
     "checkpoint and SHA-256"),
    ({"purposes": ("prediction",)}, "inference is not among its purposes"),
    ({"inference_decl": InferenceDecl("intervals", intervals=("bootstrap",))},
     "interval kinds ['bootstrap']"),
    ({"bias_terms": (Named("Lines.", "least squares", Source("nobody2099")),)},
     "cites 'nobody2099', which is not a verified source"),
    ({"complexity": (Knob("alpha", "simpler", formula="no_such_formula"),)},
     "'no_such_formula', which models.formulas does not hold"),
    ({"same_kind_as": ("xgboost", -1.0)}, "names 'xgboost', which is not another registered"),
    ({"raw_scale": {"regression": "value"}}, "['binary', 'multiclass', 'ordinal'] have none"),
    ({"raw_scale": {**get_family("linear").raw_scale, "binary": "not drawn: "}},
     "raw_scale['binary'] is 'not drawn: '"),
    ({"review_lenses": ("cardiology",)}, "review_lenses ['cardiology']"),
    ({"replay_tolerance": 0.0}, "replay_tolerance must be a positive number"),
    ({"inference_decl": replace(get_family("linear").inference_decl, default_for=("binary",))},
     "default_for ['binary']: 'linear' is already the default inference family there"),
    ({"inference_decl": InferenceDecl("description_only", default_for=("regression",))},
     "a task's default inference family gives a table with intervals"),
    ({"same_kind_as": ("cox", -1.0)}, "names 'cox', which does not model ['binary', 'multiclass', "
                                      "'ordinal', 'regression']"),
    ({"same_kind_as": ("boosted_trees", float("nan"))}, "same_kind_as must be (family key, rank "
                                                        "offset)"),
    ({"bootstrap_optimism": None}, "bootstrap_optimism must be True or False for a family that "
                                   "predicts"),
    ({"predicts": False, "raw_scale": {}, "invariances": (), "attribution": "none"},
     "bootstrap_optimism is None for a family that makes no predictions"),
    ({"predicts": False, "raw_scale": {}, "invariances": (), "attribution": "none",
      "bootstrap_optimism": None}, "attribution or curve shape: they are not applicable"),
    ({"preprocess": "spline"}, "preprocess must be a method of the family, or None"),
    ({"build": lambda self, *a: __import__("sklearn.linear_model").linear_model.LinearRegression()},
     "has no decision_function"),
    # RT-1a: the tuning members and the tree hook
    ({"tuning": _decl("depth", 2, 10, "int")},
     "its tuning names ['depth'], which are not parameters of the estimator it builds for "
     "['regression', 'binary', 'multiclass', 'ordinal'], and it has no settings to resolve them"),
    ({"tuning": {"binary": _decl("lambda", 1e-3, 1e2, "log")},
      "settings": lambda self, values, **fit: {"C": 1.0 / values["lambda"], "depth": 3}},
     "its tuning names ['depth'], which are not parameters of the estimator it builds for "
     "['binary'], and its settings do not resolve them"),
    ({"tuning": {"binary": _decl("lambda", 1e-3, 1e2, "log", kind="path", points=10)},
      "path": lambda self, Z, y, grid, **kw: None,
      "settings": lambda self, values, **fit: {"fit_intercept": True}},
     "its settings drop ['lambda'] for ['binary']: moving them leaves the estimator's parameters "
     "unchanged"),
    ({"tuning": {"binary": _decl("C", 1e-3, 1e2, "log")},
      "settings": lambda self, values, **fit: {**values, "C": 1.0}},
     "its settings drop ['C'] for ['binary']"),
    ({"tuning": {"binary": _decl("lambda", 1e-3, 1e2, "log")},
      "settings": lambda self, values, **fit: {"C": 1.0 / values["lambda"] / fit["Z"].missing}},
     "its settings failed on a probe of its binary tuning (AttributeError"),
    ({"tuning": {"binary": _decl("C", 1e-3, 1e2, "log", kind="path", points=10)}},
     "its tuning for ['binary'] is a path, so it declares path"),
    ({"path": lambda self, Z, y, grid, **kw: None}, "path is declared, but no task's tuning is a "
                                                    "path"),
    ({"tuning": {"binary": _decl("C", 1e-3, 1e2, "log", source="Nobody 2099")}},
     "the dimension 'C' cites 'Nobody 2099', which names no record of the citation registry"),
    ({"tuning": {"time_to_event": _decl("C", 1e-3, 1e2, "log")}},
     "tuning is declared for ['time_to_event'], which it does not model"),
    ({"attribution": "trees"}, "its attribution or architecture reads trees, so it declares trees"),
    ({"architecture": ("trees",)}, "its attribution or architecture reads trees, so it declares "
                                   "trees"),
    ({"trees": lambda self, step: None}, "trees is declared, but neither its attribution nor"),
    ({"consequence": " ".join(["word"] * (CONSEQUENCE_WORDS + 1))},
     f"consequence must say what choosing it means in at most {CONSEQUENCE_WORDS} words"),
    ({"defaults_version": ""}, "defaults_version must name the version of its defaults"),
])
def test_a_declaration_outside_the_contract_is_refused_by_name(declared, says):
    from turbotab.core.models.base import unregister_family

    try:
        with pytest.raises(ValueError, match=re.escape(says)):
            register_family(_probe(**declared))
        assert "probe" not in {f.key for f in families()}
    finally:
        unregister_family("probe")  # a probe let through must not leak into the next case


# ── what the declarations now drive ──────────────────────────────────────────


def test_a_same_kind_family_is_ranked_on_the_assessment_it_names():
    """C4 and RECIPES §2.2: XGBoost's declaration, ("boosted_trees", -1.0), gives it "boosted trees'
    assessment less 1.0 (at least 0.5)" on the shelf, never its own. Boosted trees score 3.0 from
    2,000 rows, 1.0 from 200 and 0.5 below (``models/boosted_trees.py``); RECIPES' worked order has
    XGBoost at 2.0 from 2,000 rows. The floor never lifts it above the family it reads."""
    from turbotab.core.models.base import Situation, rank, unregister_family
    from turbotab.core.models.boosted_trees import BoostedTrees

    def assess(self: Any, situation: Any) -> Any:
        raise AssertionError("a same-kind family's own assess is never read")

    probe = type("SameKind", (BoostedTrees,), {"key": "same_kind_probe",
                                               "same_kind_as": ("boosted_trees", -1.0),
                                               "assess": assess})()
    register_family(probe)
    try:
        for rows, trees, same in ((2400, 3.0, 2.0), (300, 1.0, 0.5), (150, 0.5, 0.5)):
            ranked = {f.key: a for f, a in rank(Situation("regression", "prediction", rows, 4))}
            assert (ranked["boosted_trees"].score, ranked["same_kind_probe"].score) == (trees, same)
            assert ranked["same_kind_probe"].concerns == ranked["boosted_trees"].concerns
            assert ranked["same_kind_probe"].fit == ranked["boosted_trees"].fit
        # It reads the assessment, so the shelf's own cost for an ordered outcome still applies.
        ranked = {f.key: a for f, a in rank(Situation("ordinal", "prediction", 2400, 4))}
        assert ranked["same_kind_probe"].score == 2.0 - 1.0
    finally:
        unregister_family("same_kind_probe")


def test_a_same_kind_family_reads_no_family_that_reads_another():
    from turbotab.core.models.base import unregister_family
    from turbotab.core.models.boosted_trees import BoostedTrees

    first = type("SameKind", (BoostedTrees,), {"key": "same_kind_probe",
                                               "same_kind_as": ("boosted_trees", -1.0)})()
    register_family(first)
    try:
        second = type("SameKind2", (BoostedTrees,), {"key": "same_kind_probe_2",
                                                     "same_kind_as": ("same_kind_probe", -1.0)})()
        assert contract_problems(second) == [
            "same_kind_as names 'same_kind_probe', which reads another family's assessment itself; "
            "name the family whose assessment it is (C4)"]
    finally:
        unregister_family("same_kind_probe")


def test_each_task_has_at_most_one_default_inference_family_with_intervals():
    """FAMILY_FOR, which ``default_for`` replaced, was one family per task, each with intervals,
    since the scales stage refits it to correct a coefficient."""
    defaults = [(t, f.key) for f in _families().values() if f.inference_decl
                for t in f.inference_decl.default_for]
    assert sorted(defaults) == [("binary", "linear"), ("ordinal", "proportional_odds"),
                                ("regression", "linear")]
    assert all(get_family(k).inference_decl.table == "intervals" for _, k in defaults)


def test_what_a_family_adds_is_declared_on_every_family():
    """§1: what a family adds of its own are members of the protocol, None where it adds nothing,
    each tied to a declaration by a rule: ``inference`` exactly when its table has intervals
    (§3.1's C2 row), ``trees`` exactly when its attribution is "trees" or its architecture
    draws trees (C10: TreeSHAP and the tree view read it), ``path`` exactly
    when a task's tuning is a path (C6), ``tree_shap`` only beside ``trees``. Among today's nine,
    the screened elastic net adds its screen, the feature-wise tests build from the design, and
    three families refit a model matrix."""
    from turbotab.core.models.base import MEMBERS, OPTIONAL_MEMBERS, _declared_tunings

    def reads_trees(f: Any) -> bool:
        return f.attribution == "trees" or "trees" in f.architecture

    found = _families()
    assert set(OPTIONAL_MEMBERS) <= set(MEMBERS)
    # A family with a tree view and no TreeSHAP meets the same rule register_family enforces.
    viewed = _probe(key="tree_view_probe", architecture=("trees",), attribution="none",
                    trees=lambda self, step: None)
    assert contract_problems(viewed) == [] and reads_trees(viewed)
    for key, f in found.items():
        decl = f.inference_decl
        assert (f.inference is not None) == (decl is not None and decl.table == "intervals"), key
        assert (f.trees is not None) == reads_trees(f), key
        assert (f.path is not None) == any(d.kind == "path"
                                           for d in _declared_tunings(f)[0].values()), key
        assert f.tree_shap is None or f.trees is not None, key
    adds = {m: sorted(k for k in NINE if getattr(found[k], m) is not None)
            for m in ("preprocess", "build_for", "describe_step", "inference_matrix")}
    assert adds == {
        "preprocess": ["screened_elastic_net"],
        "build_for": ["featurewise"],
        "describe_step": ["screened_elastic_net"],
        "inference_matrix": ["cox", "linear", "proportional_odds"],
    }


def test_each_identity_names_the_classes_its_family_builds():
    """C1: the identity's estimator is for provenance, so it must name every class the family
    builds, for every task and purpose, narrow and wide (the elastic net's wide step past its
    exact path, and its yes/no step since fix round 2), and every class it names must be one of
    them."""
    for f in _families().values():
        built = {type(f.build(t, p, rows, columns)).__name__ for t in f.tasks for p in f.purposes
                 for rows, columns in ((100, 2), (10, 20), (10, 600))}
        named = f.identity.estimator
        assert {n for n in built if not re.search(rf"\b{n}\b", named)} == set(), f.key
        # and names no class it does not build (the elastic net's three CV classes, no more)
        classes = set(re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b", named))
        assert classes - built == set(), f.key


def test_each_knob_is_a_parameter_of_the_estimator_its_family_builds():
    """C7: ``Knob.setting`` is the estimator's own parameter for some task the family models, a
    setting its tuning declares (which ``settings`` resolves), or "time" when its tuning stops
    early."""
    from turbotab.core.models.base import _declared_tunings

    for f in _families().values():
        params = set().union(*(f.build(t, "prediction" if "prediction" in f.purposes else
                                       "inference", 100, 2).get_params() for t in f.tasks))
        decls = _declared_tunings(f)[0].values()
        params |= {n for d in decls for n in d.names()}
        params |= {"time"} if any(d.early_stopping is not None for d in decls) else set()
        assert [k.setting for k in f.complexity if k.setting not in params] == [], f.key


def test_featurewise_declares_what_does_not_apply_to_it_as_not_applicable():
    """C3, C5, C10, C11: it makes no predictions, so it has no invariances, raw scale, attribution
    or curve shape, and Harrell's bootstrap has no score of its to correct; the methods reference
    says so instead of "yes" (C11)."""
    from turbotab.core.reference.methods import family_section

    f = get_family("featurewise")
    assert (f.predicts, f.bootstrap_optimism, f.curve_shape, f.invariances, dict(f.raw_scale),
            f.attribution) == (False, None, "any", (), {}, "none")
    assert info(f).bootstrap_optimism is None
    said = [line for line in family_section(f) if "bootstrap optimism" in line]
    assert said and said[0].endswith("**Harrell's bootstrap optimism is sound for it:** not "
                                     "applicable, as it makes no predictions.")


def test_the_family_info_lists_the_profile_fields_read():
    """C4: ``reads`` is declared per family, empty for today's nine until MC-4's profile exists."""
    probe = _probe(reads=("rows", "columns"))
    assert info(probe).reads == ["rows", "columns"]
    found = _families()
    assert {k: info(found[k]).reads for k in NINE} == {k: [] for k in NINE}


def test_explanations_read_the_declared_raw_scale():
    """C10: a family that declares a task "not drawn" is not explained there, whatever its
    attribution; on a declared scale it is."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.models import explain as E
    from turbotab.core.models.base import NOT_DRAWN, unregister_family

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    yes = (X["a"] > 0).astype(int).to_numpy()
    register_family(_probe(key="scale_probe", raw_scale={
        **get_family("linear").raw_scale, "binary": f"{NOT_DRAWN}a probe's reason"}))
    try:
        def kind(model: Any, task: str, y: Any) -> str | None:
            pipe = Pipeline([("model", model)]).set_output(transform="pandas").fit(X, y)
            return E.model_kind("scale_probe", E.anatomy(pipe, list(X.columns), task))

        assert kind(LinearRegression(), "regression", X["b"].to_numpy()) == "linear"
        assert kind(LogisticRegression(), "binary", yes) is None
    finally:
        unregister_family("scale_probe")


def test_the_methods_text_names_a_family_neither_survey_table_holds_by_its_label():
    """Under the population answer, a family with a design-based estimator that the estimator
    words do not hold is weighted, not blocked, and a family with none that the blocked words do
    not hold is named by its methods label, never by its key (the gap §3.1 cites for the screened
    elastic net)."""
    from turbotab.core.models.base import unregister_family
    from turbotab.core.models.boosted_trees import BoostedTrees
    from turbotab.core.models.survey import models_sentence

    weighted = _probe(key="design_probe", methods_label=lambda self, task: "probe regression")
    blocked = type("Blocked", (BoostedTrees,), {
        "key": "blocked_probe", "methods_label": lambda self, task: "probe trees"})()
    register_family(weighted)
    register_family(blocked)
    try:
        state = SimpleNamespace(purpose="inference",
                                survey=SimpleNamespace(estimand="population", weight="wt"))
        assert models_sentence(state, ["design_probe", "blocked_probe"], "regression") == (
            "For the surveyed population, probe regression was weighted by `wt`, with standard "
            "errors by Taylor linearization over the survey design; probe trees has no "
            "design-based estimator, so its estimates were blocked and not reported")
        assert models_sentence(state, ["linear", "boosted_trees"], "regression") == (
            "For the surveyed population, least squares was weighted by `wt`, with standard errors "
            "by Taylor linearization over the survey design; the gradient-boosted tree model has "
            "no design-based estimator, so its estimates were blocked and not reported")
    finally:
        unregister_family("design_probe")
        unregister_family("blocked_probe")


def test_a_tuning_declaration_keeps_structural_settings_out_of_the_search():
    """C6: ``TuningDecl.structural`` names identity, never a searched dimension, and every
    dimension has its plain label, quiet term, scale and source."""
    from turbotab.core.models.tuning import Dimension, TuningDecl

    rate = Dimension("learning_rate", "how big each correction step is", "learning rate", 0.01,
                     0.3, "log", source="Probst et al. 2019")
    assert TuningDecl("search", dimensions=(rate,), structural=("booster",),
                      standard={"learning_rate": 0.1}).structural == ("booster",)
    with pytest.raises(ValueError, match=re.escape("['learning_rate'] are structural")):
        TuningDecl("search", dimensions=(rate,), structural=("learning_rate",))
    with pytest.raises(ValueError, match=re.escape(
            "the dimension 'depth' needs a quiet term, a source, a scale among")):
        TuningDecl("search", dimensions=(Dimension("depth", "how deep", "", 2, 10, "cubic"),))


def test_no_code_probes_what_a_family_adds_by_its_presence():
    """§1: every family now has ``preprocess``, ``build_for``, ``describe_step``, ``inference`` and
    ``inference_matrix``, None where it adds nothing, so ``hasattr`` is true of every family and a
    ``getattr`` default is never used. Code reads the member and tests it against None. (A
    ``hasattr(family, "inference")`` left behind would treat boosted trees as having a coefficient
    table, so the fit would pool imputations for it.)"""
    import ast

    from turbotab.core.models.base import OPTIONAL_MEMBERS

    root = REPO / "turbotab"
    probes = []
    for path in sorted([*(root / "core").rglob("*.py"), *(root / "server").rglob("*.py")]):
        if "tests" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in OPTIONAL_MEMBERS):
                probes.append(f"{path.relative_to(root)}:{node.lineno}: {ast.unparse(node)}")
    assert probes == []


def test_every_family_states_its_tuning_and_its_consequence_within_the_contract():
    """RT-1a: every family has ``tuning`` (None, or a declaration per task it models),
    ``defaults_version`` and a ``consequence`` of at most 20 words (optional until MC-2b-1);
    boosted trees' line is today's teaching option's, word for word, and its trees reach the
    explanations through its ``trees`` member."""
    from turbotab.core.models.base import _declared_tunings
    from turbotab.core.teaching.content import MODELS

    for key, f in _families().items():
        decls, problems = _declared_tunings(f)
        assert problems == [] and set(decls) <= set(f.tasks), key
        assert isinstance(f.defaults_version, str) and f.defaults_version, key
        assert len(f.consequence.split()) <= CONSEQUENCE_WORDS, key
    taught = {o["value"]: o["consequence"] for o in MODELS["options"]}
    trees = get_family("boosted_trees")
    assert trees.consequence == taught["boosted_trees"]
    assert trees.trees is not None and trees.tree_shap is None


def test_a_tuned_family_whose_settings_resolve_its_names_registers():
    """C6: a name that is not the estimator's own parameter is allowed when ``settings`` turns it
    into one: a logistic penalty λ given as C = 1/λ, probed at the standard and center values."""
    from turbotab.core.models.base import unregister_family

    seen: list[dict[str, Any]] = []

    def settings(self: Any, values: Any, **fit: Any) -> dict[str, Any]:
        seen.append({**values, "rows": fit["n_rows"], "plan": fit["plan"].family})
        return {"C": 1.0 / values["lambda"]}

    probe = _probe(key="settings_probe", tuning={"binary": _decl("lambda", 1e-3, 1e2, "log")},
                   settings=settings)
    assert contract_problems(probe) == []
    register_family(probe)
    unregister_family("settings_probe")
    assert seen[0] == {"lambda": 1e-3, "rows": 200, "plan": "settings_probe"}
    assert seen[1]["lambda"] == pytest.approx(10 ** ((-3 + 2) / 2))
