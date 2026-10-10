"""MC-2b-1 · the family lists are read from the registry, and say what they said before.

The tables below are the hand-kept ones as they stood before this package (the catalog's
``FAMILY_LENSES``, the teaching card's ``MODELS`` options, labels, consequences and order, and the
tasks ``decisions.model_families`` gave), copied here as the reference. The registry must reproduce each exactly; and a family
registered with a lens, a label and a consequence of its own appears in every list without an edit
to the catalog or the teaching content.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SHARED = "shared"

LENSES = {
    "linear": (SHARED,),
    "elastic_net": (SHARED,),
    "ridge": (SHARED,),
    "huber": (SHARED,),
    "boosted_trees": (SHARED,),
    "random_forest": (SHARED,),
    "xgboost": (SHARED,),
    "featurewise": ("metabolomics", "genomics"),
    "proportional_odds": (SHARED,),
    "mixed": (SHARED,),
    "gee": (SHARED,),
    "cox": (SHARED,),
    "screened_elastic_net": ("metabolomics", "genomics"),
}

# The teaching card's options, word for word and in the card's order: key -> (label, consequence).
CARD = {
    "linear": ("Linear model",
               "OLS or logistic regression: one reportable coefficient per predictor, with "
               "intervals for inference."),
    "elastic_net": ("Elastic net",
                    "A penalized linear model: shrinks correlated nutrients together, tuned "
                    "inside training folds."),
    # C6a phase 2: the families registered after this package, with their own declarations' lines.
    "ridge": ("Ridge",
              "A penalized straight-line model that shrinks every effect and keeps every "
              "predictor."),
    "huber": ("Robust linear regression",
              "Straight-line effects where rows far from the line count less: steadier against "
              "outliers; for prediction only."),
    "boosted_trees": ("Boosted trees",
                      "Many shallow trees: finds curves and interactions; gives no coefficients."),
    "random_forest": ("Random forest",
                      "Many deep trees averaged: finds curves and interactions with little "
                      "tuning; gives no coefficients."),
    "xgboost": ("XGBoost",
                "The same kind of model as boosted trees, in the XGBoost library reviewers "
                "often name."),
    "featurewise": ("Feature-wise tests",
                    "Tests each factor on its own, adjusted for the covariates, with "
                    "Benjamini–Hochberg false-discovery control; no predictions."),
    "screened_elastic_net": ("Screened elastic net",
                             "Keeps the features most tied to the outcome in each training fold, "
                             "then an elastic net."),
    "proportional_odds": ("Proportional-odds model",
                          "Cumulative odds ratios for an ordered outcome, the same at every "
                          "cut-point; Brant's test checks it."),
    "mixed": ("Mixed model",
              "A random intercept per unit: model-based intervals when rows repeat, even within "
              "few units."),
    "gee": ("GEE",
            "Population-average effects, with intervals robust to how a unit's repeated rows "
            "correlate."),
    "cox": ("Cox model",
            "Hazard ratios for a time-to-event outcome, using every row's follow-up, censored "
            "or not."),
}

# The card's order as it stood: the two omics families side by side, after the boosted trees; the
# phase-2 families in their registration order (models/__init__.py).
ORDER = ["linear", "elastic_net", "ridge", "huber", "boosted_trees", "random_forest", "xgboost",
         "featurewise", "screened_elastic_net", "proportional_odds", "mixed", "gee", "cox"]

TASKS = {
    "linear": {"binary", "multiclass", "ordinal", "regression"},
    "elastic_net": {"binary", "multiclass", "ordinal", "regression"},
    "ridge": {"binary", "multiclass", "ordinal", "regression"},
    "huber": {"regression"},
    "boosted_trees": {"binary", "multiclass", "ordinal", "regression"},
    "random_forest": {"binary", "multiclass", "ordinal", "regression"},
    "xgboost": {"binary", "multiclass", "ordinal", "regression"},
    "featurewise": {"binary", "regression"},
    "proportional_odds": {"ordinal"},
    "mixed": {"regression"},
    "gee": {"binary", "regression"},
    "cox": {"time_to_event"},
    "screened_elastic_net": {"binary", "regression"},
}


def _registered() -> dict[str, Any]:
    from turbotab.core.contracts import contracts
    from turbotab.core.models import families

    contracts()  # the omics chain registers the screened elastic net
    return {f.key: f for f in families()}


def _probe(**over: Any) -> Any:
    from dataclasses import replace

    from turbotab.core.models.linear import Linear

    declared = replace(Linear.inference_decl, default_for=())  # linear stays the default
    return type("Probe", (Linear,), {"key": "mc2b1_probe", "label": "Probe family",
                                     "inference_decl": declared, **over})()


def _card() -> list[tuple[str, str, str]]:
    from turbotab.core.teaching import entry

    return [(o.value, o.label, o.consequence) for o in entry("models").options]


def test_the_catalog_reads_each_familys_lenses_from_the_registry_as_its_table_gave_them():
    from turbotab.core.models.base import register_family, unregister_family
    from turbotab.core.reference import catalog

    found = _registered()
    assert not hasattr(catalog, "FAMILY_LENSES")
    assert {k: tuple(f.review_lenses) for k, f in found.items()} == LENSES
    assert {key: catalog.lenses_of_family(key) for key in LENSES} == LENSES
    probe = register_family(_probe(review_lenses=("genomics",), consequence="A line."))
    try:
        assert catalog.lenses_of_family("mc2b1_probe") == ("genomics",)
    finally:
        unregister_family(probe.key)


def test_the_card_says_what_it_said_in_its_order_and_a_family_registered_later_joins_it():
    card = _card()  # the teaching content is loaded and the card built before the probe registers
    assert {value: (label, line) for value, label, line in card} == CARD
    assert [value for value, _, _ in card] == ORDER

    from turbotab.core.models.base import register_family, unregister_family

    probe = register_family(_probe(review_lenses=("genomics",),
                                   consequence="Stands in for any family added later."))
    try:
        assert _card() == card + [("mc2b1_probe", "Probe family",
                                   "Stands in for any family added later.")]
    finally:
        unregister_family(probe.key)
    assert _card() == card


def test_a_later_omics_family_joins_the_omics_families_on_the_card():
    from turbotab.core.models.base import register_family, unregister_family

    probe = register_family(_probe(review_lenses=("metabolomics", "genomics"),
                                   consequence="Stands in for a later omics family."))
    try:
        omics = ORDER.index("screened_elastic_net") + 1
        assert [v for v, _, _ in _card()] == ORDER[:omics] + ["mc2b1_probe"] + ORDER[omics:]
    finally:
        unregister_family(probe.key)


def test_the_validator_reads_every_family_and_its_tasks_from_the_registry_in_a_fresh_process():
    root = Path(__file__).resolve().parents[4]  # the checkout holding turbotab/
    code = ("import json; from turbotab.core.decisions import model_families; "
            "print(json.dumps({k: sorted(v) for k, v in model_families().items()}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         check=True, timeout=600)
    found = json.loads(out.stdout.strip().splitlines()[-1])
    assert {k: set(v) for k, v in found.items()} == TASKS


def test_a_family_registered_with_a_lens_and_a_consequence_is_in_every_list():
    from turbotab.core.decisions import model_families
    from turbotab.core.models.base import register_family, unregister_family
    from turbotab.core.reference import catalog
    from turbotab.core.reference import methods as ref

    _registered()
    probe = register_family(_probe(review_lenses=("genomics",),
                                   consequence="Stands in for any family added later."))
    try:
        assert catalog.lenses_of_family("mc2b1_probe") == ("genomics",)
        assert model_families()["mc2b1_probe"] == {str(t) for t in probe.tasks}
        assert "mc2b1_probe" in [f.key for f in ref.families()]
        assert ("mc2b1_probe", "Probe family", "Stands in for any family added later.") in _card()
    finally:
        unregister_family(probe.key)


def test_a_family_with_no_consequence_is_refused():
    from turbotab.core.models.base import contract_problems

    assert any("consequence" in p for p in contract_problems(_probe(consequence="")))
    assert any("consequence" in p for p in contract_problems(_probe(consequence="   ")))
    assert not any("consequence" in p for p in contract_problems(_probe(consequence="A line.")))


def test_every_registered_family_states_the_cards_consequence_within_twenty_words():
    found = {k: f.consequence for k, f in _registered().items()}
    assert found == {key: line for key, (_, line) in CARD.items()}
    assert all(line.strip() and len(line.split()) <= 20 for line in found.values())
