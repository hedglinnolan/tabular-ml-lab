"""A claim is checked against the registry's vocabulary when it registers (CLAIMS_REGISTRY.md §1,
§2.1, §2.4), as a method contract is (``contracts.register_contract``).

Each guard is tested both ways: the conforming planted claim registers, and the same claim with one
field broken is refused, naming the field. The registry's copies of the leash rungs and the
purposes are held to the contracts' own, since the registry may not import them.
"""
from __future__ import annotations

import dataclasses

import pytest

from turbotab.core import claims
from turbotab.core.claims import Basis, Claim, Fires, HeldOut, Param, Part, Regime, Registry, Silent
from turbotab.core.tests.test_claims_fixtures import FIRES, SILENT, rule, teaching


def test_a_conforming_rule_and_a_conforming_teaching_claim_register():
    reg = Registry([rule(), teaching()])
    assert [c.id for c in reg] == ["planted.few_clusters", "planted.your_call"]
    assert reg["planted.few_clusters"] == rule()


def test_registering_the_same_claim_twice_is_one_claim_and_a_different_one_is_refused():
    reg = Registry([rule()])
    reg.register(rule())
    assert len(reg) == 1
    with pytest.raises(ValueError, match="planted.few_clusters"):
        reg.register(rule(card="Fewer than {few} clusters: a different sentence."))


BROKEN = {
    "id": dict(id="FewClusters"),
    "id with no area": dict(id="few_clusters"),
    "kind": dict(kind="guardrail"),
    "verdict": dict(verdict="maybe"),
    "goal": dict(goal=("description",)),
    "grade": dict(parts=(Part("value", grade="settled", basis=Basis(reason="r")),)),
    "no part": dict(parts=()),
    "two parts of one name": dict(parts=(Part("value", grade="judgment", basis=Basis(reason="r")),
                                         Part("value", grade="unsupported"))),
    "truth": dict(tests=(Fires(FIRES.test, truth="trust me"), SILENT)),
    "independent truth with no library": dict(tests=(Fires(FIRES.test, truth="independent:"),
                                                     SILENT)),
    "derivation truth with no path": dict(tests=(Fires(FIRES.test, truth="derivation:"), SILENT)),
    "test with no function": dict(tests=(Fires("planted/test_rule.py", "construction"), SILENT)),
    "test outside a .py file": dict(tests=(Fires("planted/rule.txt::test_x", "construction"),
                                           SILENT)),
    "rule with no fires test": dict(tests=(SILENT,)),
    "rule with no silent test": dict(tests=(FIRES,)),
    "rule with no enforcer": dict(enforced_by=()),
    "rule with no verdict": dict(verdict=""),
    "enforcer not module:function": dict(enforced_by=("planted.engine.act",)),
    "predicate not module:function": dict(regime=Regime("r", requires=("outcome_is_binary",))),
    "placeholder with no parameter": dict(card="Fewer than {many} clusters."),
    "positional placeholder": dict(card="Fewer than {0} clusters."),
    "attribute placeholder": dict(card="Fewer than {few.real} clusters."),
    "unbalanced brace": dict(card="Fewer than {few clusters."),
    "parameter that is not a number": dict(params={"few": Param("ten", unit="clusters")}),
    "parameter that is a bool": dict(params={"few": Param(True, unit="clusters")}),
    "parameter that is not finite": dict(params={"few": Param(float("nan"), unit="clusters")}),
    "parameter with no unit": dict(params={"few": Param(10, unit="")}),
    "constant not module:NAME": dict(params={"few": Param(10, unit="clusters", constant="FEW")}),
    "no author": dict(author=""),
    "version zero": dict(version=0),
    "retraction with no date": dict(retracted="superseded"),
    "falls back to itself": dict(fallback="planted.few_clusters"),
    "supersedes itself": dict(supersedes=("planted.few_clusters",)),
    "disjoint from itself": dict(regime=Regime("r", disjoint_from=("planted.few_clusters",))),
    "held-out rate above one": dict(parts=(Part("value", grade="replicated", basis=Basis(
        heldout=(HeldOut("a", cases=10, rate=1.5), HeldOut("b", cases=10, rate=0.5)))),)),
    "held-out fixture with no case": dict(parts=(Part("value", grade="replicated", basis=Basis(
        heldout=(HeldOut("a", cases=0, rate=0.5), HeldOut("b", cases=10, rate=0.5)))),)),
}


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_a_claim_outside_the_vocabulary_is_refused_at_registration(broken):
    with pytest.raises(ValueError):
        Registry([rule(**BROKEN[broken])])


def test_a_teaching_claim_needs_no_test_enforcer_or_verdict():
    """The both-ways rule (§2.1) is for a rule: a teaching claim with none of them registers."""
    reg = Registry([teaching(tests=(), enforced_by=(), verdict="")])
    assert "planted.your_call" in reg


def test_a_retraction_with_its_date_and_reason_registers():
    reg = Registry([rule(retracted="2026-10-16: superseded by a planted claim")])
    assert reg["planted.few_clusters"].retracted.startswith("2026-10-16")


def test_the_registry_copies_the_contracts_rungs_and_purposes():
    """The registry imports no engine module, so it holds copies; they must not drift."""
    from turbotab.core import contracts

    assert claims.RUNGS == contracts.RUNGS
    assert claims.PURPOSES == contracts.PURPOSES
    assert claims.VERDICTS == (*contracts.RUNGS, "noted")


def test_a_claim_shows_its_weakest_part():
    """§2.4: a sentence shows the weakest part it asserts, whatever order the parts are in."""
    parts = (Part("value", grade="sourced", basis=Basis(cites=("friedman2001",), where="p. 1")),
             Part("scope", grade="judgment", basis=Basis(reason="r")),
             Part("direction", grade="certified", basis=Basis(tests=(FIRES.test,))))
    assert rule(parts=parts).grade == "judgment"
    assert rule(parts=parts[:1] + parts[2:]).grade == "sourced"
    assert rule(parts=(Part("value", grade="unsupported"),) + parts).grade == "unsupported"


def test_equal_content_hashes_equal_and_any_change_moves_the_hash():
    assert claims.content_hash(rule()) == claims.content_hash(rule())
    changed = [rule(card="Fewer than {few} clusters."),
               rule(params={"few": Param(11, unit="clusters", constant="planted.engine:FEW")}),
               rule(parts=(Part("value", grade="unsupported"),)),
               rule(tests=(FIRES, Silent(FIRES.test, truth="construction"))),
               rule(version=2)]
    hashes = {claims.content_hash(c) for c in changed}
    assert len(hashes) == len(changed)
    assert claims.content_hash(rule()) not in hashes


def test_a_fires_test_and_a_silent_test_on_one_function_are_two_tests():
    """Fires and Silent differ by side, so the content (and its hash) keeps which is which."""
    assert Fires(FIRES.test, "construction") != Silent(FIRES.test, "construction")
    a = claims.content(rule(tests=(Fires(FIRES.test, "construction"), SILENT)))
    b = claims.content(rule(tests=(Silent(FIRES.test, "construction"), SILENT)))
    assert a != b


def test_a_claim_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule().card = "changed"  # type: ignore[misc]


def test_a_claim_with_its_fields_named_as_the_spec_example_names_them():
    """§1's example, with its test names and citation key made planted ones, registers."""
    c = Claim(
        id="trials.few_clusters_example", kind="rule",
        card="Fewer than {few_clusters} clusters: intervals can still be too narrow.",
        methods="With fewer than {few_clusters} clusters, the concern is named.",
        goal=("inference",),
        regime=Regime(words="a cluster-randomized trial, GEE with a small-sample correction",
                      requires=("turbotab.core.tests.test_claims_fixtures:outcome_is_binary",)),
        decision="trials.correction", options=("gee_corrected", "mixed_kr"), verdict="noted",
        parts=(Part("value", grade="sourced", basis=Basis(cites=("friedman2001",), where="§3")),
               Part("scope", grade="judgment", basis=Basis(reason="simulated binary only"))),
        params={"few_clusters": Param(10, unit="clusters")},
        tests=(FIRES, SILENT), enforced_by=("planted.engine:act",), supersedes=(),
        graded_by="basis audit, 2026-10-15", author="the R0a builder", version=1)
    assert Registry([c])["trials.few_clusters_example"].grade == "judgment"
