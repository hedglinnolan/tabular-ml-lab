"""MC-1 and MC-2a · every model family declares the contract, and the switches the new families hit
read those declarations (MODEL_FAMILY_CONTRACT §1, §3.1, §3.3, §5).

Expected values: §3.1's table of today's nine families, written out here; the tables and labels the
retired switches held, copied from the code before MC-2a; ridge's hat matrix by explicit inverse
(ESL §3.4.1) against scikit-learn's own elastic net; and §7's verified references, read from the
contract itself.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import get_family
from turbotab.core.models.base import (
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
# soundness and flexibility, and C10's explanation path today.
EXPECTED = {
    "linear": (("linear_maps",), "intervals", True, False, "linear"),
    "elastic_net": (("column_scale",), "shrunk_no_intervals", True, False, "linear"),
    "boosted_trees": (("monotone_per_column",), "description_only", False, True, "trees"),
    "featurewise": ((), "intervals", True, False, "none"),
    "proportional_odds": (("linear_maps",), "intervals", True, False, "none"),
    "mixed": (("linear_maps",), "intervals", True, False, "none"),
    "gee": (("linear_maps",), "intervals", True, False, "none"),
    "cox": (("linear_maps",), "intervals", True, False, "none"),
    "screened_elastic_net": (("column_scale",), None, True, False, "linear"),
}


def _families() -> dict[str, Any]:
    from turbotab.core.contracts import contracts

    contracts()  # the omics chain registers the screened elastic net
    return {f.key: f for f in families()}


def test_the_nine_families_declare_what_section_3_1_says():
    found = _families()
    assert sorted(found) == sorted(EXPECTED)
    for key, (invariances, table, sound, flexible, attribution) in EXPECTED.items():
        f = found[key]
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

    assert [k for k, f in _families().items() if is_flexible(f)] == ["boosted_trees"]
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


def test_every_cited_source_is_one_of_section_7s_verified_references():
    """Until SIZING X4's registry: each source key's reference is a line of the contract's §7."""
    from turbotab.core.models.sources import SOURCES

    text = CONTRACT.read_text(encoding="utf-8")
    section = text[text.index("## 7 · Sources"):text.index("## What changed after review")]
    lines = {line[2:] for line in section.splitlines() if line.startswith("- ")}
    assert sorted(k for k, ref in SOURCES.items() if ref not in lines) == []
    cited = {s.key for f in _families().values()
             for s in [*f.sources, *(t.source for t in f.bias_terms),
                       *(k.source for k in f.complexity if k.source)]}
    assert cited == set(SOURCES)  # each listed because a declaration cites it


def test_the_elastic_net_ridge_part_is_its_hat_matrix():
    """C7: the elastic net's declared penalty knob names ``elastic_net_ridge_part``. At l1_ratio 0
    scikit-learn's elastic net is its ridge part alone, so its fit is the hat matrix
    Z(ZᵀZ + κI)⁻¹Zᵀ at κ = nα, built here by explicit inverse; df is that matrix's trace
    (ESL §3.4.1, eq. 3.50), to 1e-10."""
    from sklearn.linear_model import ElasticNet

    from turbotab.core.models.formulas import FORMULAS

    (knob,) = [k for k in get_family("elastic_net").complexity if k.formula]
    assert (knob.setting, knob.more_means) == ("alpha", "simpler")
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


def _probe(**declared: Any) -> Any:
    """The linear family's declarations with ``declared`` changed, under a key of its own."""
    from turbotab.core.models.linear import Linear

    return type("Probe", (Linear,), {"key": "probe", **declared})()


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
    ({"build": lambda self, *a: __import__("sklearn.linear_model").linear_model.LinearRegression()},
     "has no decision_function"),
])
def test_a_declaration_outside_the_contract_is_refused_by_name(declared, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        register_family(_probe(**declared))
    assert "probe" not in {f.key for f in families()}


def test_a_family_may_read_a_registered_familys_assessment():
    assert contract_problems(_probe(same_kind_as=("boosted_trees", -1.0))) == []
