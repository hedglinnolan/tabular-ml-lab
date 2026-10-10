"""MC-2b-1 · the family lists are read from the registry, and say what they said before.

The tables below are the hand-kept ones as they stood before this package (the catalog's
``FAMILY_LENSES``, the teaching card's ``MODELS`` options and the tasks ``decisions.model_families``
gave), copied here as the reference. The registry must reproduce each exactly; and a family
registered with a lens, a label and a consequence of its own appears in every list without an edit
to the catalog or the teaching content.
"""
from __future__ import annotations

from typing import Any

import pytest

SHARED = "shared"

LENSES = {
    "linear": (SHARED,),
    "elastic_net": (SHARED,),
    "boosted_trees": (SHARED,),
    "featurewise": ("metabolomics", "genomics"),
    "proportional_odds": (SHARED,),
    "mixed": (SHARED,),
    "gee": (SHARED,),
    "cox": (SHARED,),
    "screened_elastic_net": ("metabolomics", "genomics"),
}

# The teaching card's options, word for word: key -> (label, consequence).
CARD = {
    "linear": ("Linear model",
               "OLS or logistic regression: one reportable coefficient per predictor, with "
               "intervals for inference."),
    "elastic_net": ("Elastic net",
                    "A penalized linear model: shrinks correlated nutrients together, tuned "
                    "inside training folds."),
    "boosted_trees": ("Boosted trees",
                      "Many shallow trees: finds curves and interactions; gives no coefficients."),
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

TASKS = {
    "linear": {"binary", "multiclass", "ordinal", "regression"},
    "elastic_net": {"binary", "multiclass", "ordinal", "regression"},
    "boosted_trees": {"binary", "multiclass", "ordinal", "regression"},
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


def test_the_registry_gives_the_lenses_the_catalog_table_gave():
    from turbotab.core.reference import catalog

    found = _registered()
    assert {k: tuple(f.review_lenses) for k, f in found.items()} == LENSES
    for key, lenses in LENSES.items():
        assert catalog.lenses_of_family(key) == lenses


def test_the_catalog_keeps_no_table_of_families():
    from turbotab.core.reference import catalog

    assert not hasattr(catalog, "FAMILY_LENSES")


def test_the_teaching_card_gives_every_family_its_label_and_consequence_as_before():
    from turbotab.core.teaching import entry

    registered = _registered()
    options = {o.value: (o.label, o.consequence) for o in entry("models").options}
    assert options == CARD
    assert [o.value for o in entry("models").options] == list(registered)


def test_the_registry_gives_the_families_and_their_tasks_the_validator_read():
    from turbotab.core.decisions import model_families

    _registered()
    assert model_families() == TASKS


def test_a_family_registered_with_a_lens_and_a_consequence_is_in_every_list():
    from turbotab.core.decisions import model_families
    from turbotab.core.models.base import register_family, unregister_family
    from turbotab.core.reference import catalog
    from turbotab.core.reference import methods as ref
    from turbotab.core.teaching.content import family_options

    _registered()
    probe = register_family(_probe(review_lenses=("genomics",),
                                   consequence="Stands in for any family added later."))
    try:
        assert catalog.lenses_of_family("mc2b1_probe") == ("genomics",)
        assert "mc2b1_probe" in model_families()
        assert "mc2b1_probe" in [f.key for f in ref.families()]
        card = {o["value"]: o for o in family_options()}["mc2b1_probe"]
        assert (card["label"], card["consequence"]) == (
            "Probe family", "Stands in for any family added later.")
    finally:
        unregister_family(probe.key)


def test_a_family_with_no_consequence_is_refused():
    from turbotab.core.models.base import contract_problems

    assert any("consequence" in p for p in contract_problems(_probe(consequence="")))
    assert any("consequence" in p for p in contract_problems(_probe(consequence="   ")))
    assert not any("consequence" in p for p in contract_problems(_probe(consequence="A line.")))


@pytest.mark.parametrize("key", sorted(LENSES))
def test_every_registered_family_states_its_consequence_within_twenty_words(key):
    f = _registered()[key]
    assert f.consequence.strip() and len(f.consequence.split()) <= 20
    assert f.consequence == CARD[key][1]
