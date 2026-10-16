"""A claim's sentence renders only inside its regime, and only when it is built (CLAIMS_REGISTRY.md
§2.2, §6; R0_PLAN.md R0a acceptance 2).

* Strict mode raises when the context fails a regime predicate; outside it, the named fallback
  claim renders. ``strict=None`` reads ``TURBOTAB_CLAIMS_STRICT``, which the core test session
  sets to "1" (``conftest.py``).
* A proposed or retracted claim never renders, in either mode.
* A sentence is its template filled from the claim's parameters; the expected sentences here are
  the planted templates filled by hand.
* Predicates are resolved when they are checked, never at registration or without a context.

Each planted registry has its status frozen as the generator freezes it
(``test_claims_fixtures.registry``); rendering from that file is ``test_claims_frozen.py``'s.
"""
from __future__ import annotations

import os

import pytest

from turbotab.core import claims
from turbotab.core.claims import Fires, NotRenderable, OutsideRegime, Param, Regime
from turbotab.core.tests.test_claims_fixtures import HERE, registry, rule, teaching

ID = "planted.few_clusters"
CARD = "Fewer than 10 clusters: intervals can still be too narrow."
METHODS = "With fewer than 10 clusters, the concern is named."
FALLBACK_CARD = "Your call: here is what each choice does."
FALLBACK_METHODS = "The analyst chose between the options."
INSIDE = {"outcome": "binary"}
OUTSIDE = {"outcome": "continuous"}


def test_the_core_test_session_runs_in_strict_mode():
    assert os.environ.get("TURBOTAB_CLAIMS_STRICT") == "1"


def test_inside_its_regime_a_claim_renders_its_template_filled_from_its_parameters(tmp_path):
    reg = registry(tmp_path, rule())
    for strict in (True, False):
        assert reg.card(ID, INSIDE, strict=strict) == CARD
        assert reg.methods(ID, INSIDE, strict=strict) == METHODS


def test_strict_mode_raises_outside_the_regime(tmp_path):
    reg = registry(tmp_path, rule(fallback="planted.your_call"), teaching())
    with pytest.raises(OutsideRegime, match=ID):
        reg.card(ID, OUTSIDE, strict=True)
    with pytest.raises(OutsideRegime, match=ID):
        reg.methods(ID, OUTSIDE, strict=True)


def test_outside_its_regime_non_strict_renders_the_named_fallback(tmp_path):
    reg = registry(tmp_path, rule(fallback="planted.your_call"), teaching())
    assert reg.card(ID, OUTSIDE, strict=False) == FALLBACK_CARD
    assert reg.methods(ID, OUTSIDE, strict=False) == FALLBACK_METHODS


def test_a_fallback_outside_its_own_regime_falls_back_again(tmp_path):
    middle = teaching(id="planted.middle", card="A middle sentence.", methods="A middle clause.",
                      regime=Regime("yes/no only", requires=(f"{HERE}:outcome_is_binary",)),
                      fallback="planted.your_call")
    reg = registry(tmp_path, rule(fallback="planted.middle"), middle, teaching())
    assert reg.card(ID, OUTSIDE, strict=False) == FALLBACK_CARD


def test_outside_its_regime_with_no_fallback_even_non_strict_refuses(tmp_path):
    reg = registry(tmp_path, rule())
    with pytest.raises(OutsideRegime, match="no fallback"):
        reg.card(ID, OUTSIDE, strict=False)


def test_strict_none_reads_the_environment(tmp_path, monkeypatch):
    reg = registry(tmp_path, rule(fallback="planted.your_call"), teaching())
    monkeypatch.setenv("TURBOTAB_CLAIMS_STRICT", "1")
    with pytest.raises(OutsideRegime):
        reg.card(ID, OUTSIDE)
    monkeypatch.setenv("TURBOTAB_CLAIMS_STRICT", "0")
    assert reg.card(ID, OUTSIDE) == FALLBACK_CARD
    monkeypatch.delenv("TURBOTAB_CLAIMS_STRICT")
    assert reg.card(ID, OUTSIDE) == FALLBACK_CARD


@pytest.mark.parametrize("strict", [True, False, None])
def test_a_proposed_claim_never_renders(tmp_path, strict):
    proposed = rule(enforced_by=("planted.engine:not_written_yet",))
    reg = registry(tmp_path, proposed)
    assert reg.status(ID) == "proposed"
    for ctx in (INSIDE, None):
        with pytest.raises(NotRenderable, match="proposed"):
            reg.card(ID, ctx, strict=strict)
        with pytest.raises(NotRenderable, match="proposed"):
            reg.methods(ID, ctx, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_a_retracted_claim_never_renders(tmp_path, strict):
    reg = registry(tmp_path, rule(retracted="2026-10-16: a planted retraction"))
    with pytest.raises(NotRenderable, match="retracted"):
        reg.card(ID, INSIDE, strict=strict)


def test_a_fallback_that_is_proposed_does_not_render_either(tmp_path):
    gone = Fires("planted/test_rule.py::test_gone", "construction")
    reg = registry(tmp_path, rule(fallback="planted.your_call"), teaching(tests=(gone,)))
    assert reg.status("planted.your_call") == "proposed"
    with pytest.raises(NotRenderable, match="planted.your_call"):
        reg.card(ID, OUTSIDE, strict=False)


def test_a_fallback_loop_is_refused_not_followed_forever(tmp_path):
    a = teaching(id="planted.a", regime=Regime("never", requires=(f"{HERE}:outcome_is_binary",)),
                 fallback="planted.b")
    b = teaching(id="planted.b", regime=Regime("never", requires=(f"{HERE}:outcome_is_binary",)),
                 fallback="planted.a")
    reg = registry(tmp_path, a, b)
    with pytest.raises(OutsideRegime, match="loop"):
        reg.card("planted.a", OUTSIDE, strict=False)


def test_predicates_are_resolved_only_when_a_context_is_checked(tmp_path):
    """A predicate naming no module registers, and renders without a context; checking it fails."""
    reg = registry(tmp_path, rule(regime=Regime("r", requires=("nowhere.at_all:predicate",))))
    assert reg.card(ID) == CARD
    with pytest.raises(claims.ClaimError, match="nowhere.at_all:predicate"):
        reg.card(ID, INSIDE, strict=True)


def test_a_predicate_must_answer_with_a_bool(tmp_path):
    reg = registry(tmp_path, rule(regime=Regime("r", requires=(f"{HERE}:not_a_bool",))))
    with pytest.raises(TypeError, match="not_a_bool"):
        reg.card(ID, INSIDE, strict=True)


def test_every_predicate_must_hold(tmp_path):
    both = Regime("r", requires=(f"{HERE}:any_design", f"{HERE}:outcome_is_binary"))
    reg = registry(tmp_path, rule(regime=both))
    assert reg.card(ID, INSIDE, strict=True) == CARD
    with pytest.raises(OutsideRegime, match="outcome_is_binary"):
        reg.card(ID, OUTSIDE, strict=True)


def test_value_reads_the_parameter_and_names_what_is_missing(tmp_path):
    reg = registry(tmp_path, rule())
    assert reg.value(ID, "few") == 10
    with pytest.raises(KeyError, match="many"):
        reg.value(ID, "many")
    with pytest.raises(KeyError, match="planted.nothing"):
        reg.value("planted.nothing", "few")


def test_a_claim_with_no_card_sentence_refuses_to_render_one(tmp_path):
    reg = registry(tmp_path, teaching(card=""))
    with pytest.raises(claims.ClaimError, match="no card"):
        reg.card("planted.your_call")
    assert reg.methods("planted.your_call") == FALLBACK_METHODS


def test_a_format_spec_in_a_placeholder_formats_the_parameter(tmp_path):
    reg = registry(tmp_path, rule(card="below a coefficient of variation of {cv:.1f}",
                                  methods="below {cv:.0%} of the mean",
                                  params={"cv": Param(0.6, unit="ratio")}))
    assert reg.card(ID, INSIDE) == "below a coefficient of variation of 0.6"
    assert reg.methods(ID, INSIDE) == "below 60% of the mean"


def test_the_module_functions_read_the_default_registry():
    assert claims.registry() is claims.REGISTRY
    for fn in (claims.card, claims.methods, claims.status):
        with pytest.raises(KeyError, match="planted.not_registered"):
            fn("planted.not_registered")
    with pytest.raises(KeyError, match="planted.not_registered"):
        claims.value("planted.not_registered", "few")
