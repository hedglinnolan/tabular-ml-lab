"""The six meta-tests (CLAIMS_REGISTRY.md §6; R0_PLAN.md R0a acceptance 1), each run three ways: on
the default registry (no claim is registered yet, so each must pass), on a planted registry that
breaks it (it must name the violation) and on a planted registry that keeps it (it must name none).

1. every claim's tests exist and declare their truth;
2. no claim test imports or names its own claim's constant, or names its own claim's id (so it
   cannot read the claim's value, sentence or entry, however it spells the call);
3. every part's grade meets §2.4's requirement, with a grader other than the author above
   judgment, and regime predicates wherever a part is graded above judgment (§2.2);
4. no proposed claim renders: no built claim falls back to one, and none is shown;
5. no two live claims conflict on one decision in overlapping regimes;
6. the digit gate, over every claim's templates outside their placeholders.

On the default registry, the meta-tests that read the app's strings (4 and 6) read every
``Relation.says`` and teaching string, as ``test_claims_digits.py`` collects them.

Every planted case is built here from files the test writes, so its truth is the construction.
"""
from __future__ import annotations

import pytest

from turbotab.core import claims
from turbotab.core.claims import Basis, Fires, HeldOut, Param, Part, Regime, Silent
from turbotab.core.tests.test_claims_fixtures import (DERIVATION, FIRES, HERE, SILENT, registry,
                                                       rule, teaching)
from turbotab.core.tests.test_claims_digits import relation_says, teaching_strings

ID = "planted.few_clusters"
OTHER = "a basis audit"
REQUIRES = (f"{HERE}:outcome_is_binary",)


def app_strings(meta):
    """What the default run passes ``meta`` beside the registry: the app's strings, where it reads
    them."""
    if meta is claims.none_proposed_renders:
        return {"shown": [text for _, text in relation_says() + teaching_strings()]}
    if meta is claims.digit_gate:
        return {"strings": relation_says() + teaching_strings()}
    return {}


@pytest.mark.parametrize("meta", claims.META_TESTS, ids=lambda m: m.__name__)
def test_each_meta_test_passes_on_the_default_registry(meta):
    assert meta(claims.registry(), **app_strings(meta)) == []


def test_there_are_six_meta_tests_in_the_order_section_6_gives():
    assert [m.__name__ for m in claims.META_TESTS] == [
        "tests_exist", "no_self_pinning", "grades_met", "none_proposed_renders", "no_conflicts",
        "digit_gate"]


# ── 1. every claim's tests exist and declare their truth ─────────────────────────────────────

def test_1_fires_where_live_code_names_a_test_that_does_not_exist(tmp_path):
    gone = Silent("planted/test_rule.py::test_never_written", "construction")
    found = claims.tests_exist(registry(tmp_path, rule(tests=(FIRES, SILENT, gone))))
    assert len(found) == 1 and ID in found[0] and "test_never_written" in found[0]


def test_1_fires_where_a_derivation_truth_names_no_file(tmp_path):
    found = claims.tests_exist(registry(tmp_path, rule(
        tests=(Fires(FIRES.test, "derivation:notes/missing.md"), SILENT))))
    assert len(found) == 1 and "notes/missing.md" in found[0]


def test_1_is_silent_where_every_test_exists_and_its_derivation_does(tmp_path):
    reg = registry(tmp_path, rule(tests=(Fires(FIRES.test, f"derivation:{DERIVATION}"), SILENT)))
    assert claims.tests_exist(reg) == []


def test_1_is_silent_on_a_ruling_whose_code_is_not_written_yet(tmp_path):
    """A proposed claim whose enforcer does not exist yet may name tests not written yet (§6)."""
    gone = Silent("planted/test_rule.py::test_never_written", "construction")
    reg = registry(tmp_path, rule(tests=(FIRES, gone), enforced_by=("planted.engine:not_yet",)))
    assert reg.status(ID) == "proposed"
    assert claims.tests_exist(reg) == []


# ── 2. no claim test pins the claim to its own constant or value ─────────────────────────────

PINNED = {
    "imports the constant": "from planted.engine import FEW\n",
    "imports everything from its module": "from planted.engine import *  # noqa: F403\n",
    "names it on its module": "import planted.engine as engine\nFEW_SEEN = engine.FEW\n",
    "names it as a string": "import planted.engine\nN = getattr(planted.engine, 'FEW')\n",
    "reads its own claim's value": "from turbotab.core import claims\n"
                                   "N = claims.value('planted.few_clusters', 'few')\n",
    "reads its value by the bare function": "from turbotab.core.claims import value\n"
                                            "N = value(id='planted.few_clusters', param='few')\n",
    "reads its value through a variable": "from turbotab.core import claims\n"
                                          "ID = 'planted.few_clusters'\n"
                                          "N = claims.value(ID, 'few')\n",
    "reads its value through the registry": "from turbotab.core import claims\n"
                                            "N = claims.registry()['planted.few_clusters']"
                                            ".params['few'].value\n",
    "reads its value through an alias": "from turbotab.core.claims import value as v\n"
                                        "N = v('planted.few_clusters', 'few')\n",
    "renders its own sentence": "from turbotab.core import claims\n"
                                "SAYS = claims.card('planted.few_clusters', {'outcome': 'binary'})"
                                "\n",
}
CLEAN = {
    "states the number from the construction": "N = 10\n",
    "reads another claim's value": "from turbotab.core import claims\n"
                                   "N = claims.value('planted.other', 'few')\n",
    "imports the module but not the constant": "import planted.engine\nACT = planted.engine.act\n",
    "reads another claim's value through a variable": "from turbotab.core import claims\n"
                                                      "OTHER = 'planted.other'\n"
                                                      "N = claims.value(OTHER, 'few')\n",
    "names its claim inside its docstring": '"""The known-truth tests behind planted.few_clusters'
                                            '."""\n',
}
BODY = ("\n\ndef test_nine_clusters_name_the_concern():\n    assert 9 < 10\n\n\n"
        "def test_forty_clusters_say_nothing():\n    assert not 40 < 10\n")


def _pinned_registry(tmp_path, header):
    reg = registry(tmp_path, rule(tests=(
        Fires("planted/test_pinned.py::test_nine_clusters_name_the_concern", "construction"),
        Silent("planted/test_pinned.py::test_forty_clusters_say_nothing", "construction"))))
    (reg.root / "planted" / "test_pinned.py").write_text(header + BODY, encoding="utf-8")
    return reg


@pytest.mark.parametrize("case", sorted(PINNED))
def test_2_fires_on_a_test_file_that_pins_the_claim_to_itself(tmp_path, case):
    found = claims.no_self_pinning(_pinned_registry(tmp_path, PINNED[case]))
    assert found and all(ID in v and "test_pinned.py" in v for v in found)


@pytest.mark.parametrize("case", sorted(CLEAN))
def test_2_is_silent_on_a_test_file_whose_truth_is_its_own(tmp_path, case):
    assert claims.no_self_pinning(_pinned_registry(tmp_path, CLEAN[case])) == []


def test_2_reads_the_constant_from_the_parameter_that_declares_it(tmp_path):
    """With no constant declared, naming ``FEW`` is not this claim's constant."""
    reg = _pinned_registry(tmp_path, PINNED["imports the constant"])
    undeclared = rule(tests=reg[ID].tests, params={"few": Param(10, unit="clusters")})
    reg2 = claims.Registry([undeclared], root=reg.root)
    assert claims.no_self_pinning(reg2) == []


# ── 3. every grade meets its requirement ─────────────────────────────────────────────────────

def _graded(*parts, graded_by=OTHER, requires=REQUIRES):
    return rule(parts=parts, graded_by=graded_by, regime=Regime("r", requires=requires))


UNMET = {
    "proved with no theorem or derivation": _graded(Part("value", "proved")),
    "proved by a derivation that is not there": _graded(
        Part("value", "proved", Basis(derivation="notes/missing.md"))),
    "certified with no test": _graded(Part("value", "certified")),
    "certified by a test the claim does not declare": _graded(
        Part("value", "certified", Basis(tests=("planted/test_rule.py::test_elsewhere",)))),
    "replicated on one fixture": _graded(
        Part("value", "replicated", Basis(heldout=(HeldOut("nhanes", 40, 0.9),
                                                   HeldOut("nhanes", 12, 0.8))))),
    "sourced by a key that does not resolve": _graded(
        Part("value", "sourced", Basis(cites=("nobody1999",), where="p. 3"))),
    "sourced with no passage": _graded(Part("value", "sourced", Basis(cites=("friedman2001",)))),
    "judgment with no reason": _graded(Part("value", "judgment"), graded_by=""),
    "above judgment with no grader": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")), graded_by=""),
    "above judgment graded by its author": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")),
        graded_by="the R0a builder"),
    "a part graded by its author over another grader": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             graded_by="the R0a builder")),
    "above judgment with no regime predicate": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")), requires=()),
}
MET = {
    "proved by a theorem": _graded(Part("value", "proved", Basis(theorem="Lean: planted.thm"))),
    "proved by a derivation another re-derived": _graded(
        Part("value", "proved", Basis(derivation=DERIVATION))),
    "certified by the claim's own test": _graded(
        Part("value", "certified", Basis(tests=(FIRES.test,)))),
    "replicated on two fixtures with rates": _graded(
        Part("value", "replicated", Basis(heldout=(HeldOut("nhanes", 40, 0.9),
                                                   HeldOut("framingham", 25, 0.84))))),
    "sourced by a key that resolves, with its passage": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="§10.10, p. 1203"))),
    "judgment with its reason, graded by its author": _graded(
        Part("value", "judgment", Basis(reason="we use it")), graded_by="the R0a builder",
        requires=()),
    "unsupported with nothing": _graded(Part("value", "unsupported"), graded_by="", requires=()),
    "a part graded by another over its author's claim-level grade": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"), graded_by=OTHER),
        graded_by="the R0a builder"),
}


@pytest.mark.parametrize("case", sorted(UNMET))
def test_3_fires_on_a_grade_its_basis_does_not_meet(tmp_path, case):
    found = claims.grades_met(registry(tmp_path, UNMET[case]))
    assert len(found) == 1 and ID in found[0]


@pytest.mark.parametrize("case", sorted(MET))
def test_3_is_silent_on_a_grade_its_basis_meets(tmp_path, case):
    assert claims.grades_met(registry(tmp_path, MET[case])) == []


def test_3_checks_every_part_on_its_own(tmp_path):
    claim = _graded(Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")),
                    Part("direction", "certified"), Part("scope", "judgment"))
    found = claims.grades_met(registry(tmp_path, claim))
    assert len(found) == 2
    assert any("direction" in v for v in found) and any("scope" in v for v in found)


# ── 4. no proposed claim renders ─────────────────────────────────────────────────────────────

def _proposed(**over):
    return teaching(tests=(Fires("planted/test_rule.py::test_not_yet", "construction"),), **over)


def test_4_fires_where_a_built_claim_falls_back_to_a_proposed_one(tmp_path):
    found = claims.none_proposed_renders(registry(tmp_path, rule(fallback="planted.your_call"),
                                                  _proposed()))
    assert len(found) == 1 and ID in found[0] and "proposed" in found[0]


def test_4_fires_where_a_built_claim_falls_back_to_a_retracted_one(tmp_path):
    found = claims.none_proposed_renders(registry(
        tmp_path, rule(fallback="planted.your_call"),
        teaching(retracted="2026-10-16: a planted retraction")))
    assert len(found) == 1 and "retracted" in found[0]


def test_4_fires_where_a_fallback_is_not_registered(tmp_path):
    found = claims.none_proposed_renders(registry(tmp_path, rule(fallback="planted.nowhere")))
    assert len(found) == 1 and "planted.nowhere" in found[0]


def test_4_fires_where_fallbacks_loop(tmp_path):
    a = teaching(id="planted.a", fallback="planted.b")
    b = teaching(id="planted.b", fallback="planted.a")
    found = claims.none_proposed_renders(registry(tmp_path, a, b))
    assert found and all("loop" in v for v in found)


def test_4_fires_where_a_proposed_claim_s_sentence_is_shown(tmp_path):
    reg = registry(tmp_path, _proposed())
    shown = ["Because you chose X: Your call: here is what each choice does."]
    found = claims.none_proposed_renders(reg, shown=shown)
    assert len(found) == 1 and "planted.your_call" in found[0] and "shown" in found[0]


def test_4_reads_the_app_s_own_sentences_on_the_default_run(tmp_path):
    """A proposed claim worded as a real ``Relation.says`` is caught by the default run's call."""
    says = next(t for _, t in relation_says() if "{" not in t and "}" not in t)
    found = claims.none_proposed_renders(registry(tmp_path, _proposed(card=says)),
                                         **app_strings(claims.none_proposed_renders))
    assert len(found) == 1 and "planted.your_call" in found[0] and "shown" in found[0]


def test_4_is_silent_where_built_claims_fall_back_to_built_ones_and_nothing_proposed_shows(
        tmp_path):
    reg = registry(tmp_path, rule(fallback="planted.your_call"), teaching(),
                   _proposed(id="planted.ruled", card="A ruled sentence."))
    assert claims.none_proposed_renders(reg, shown=["Your call: here is what each choice does."]
                                        ) == []


def test_4_is_silent_where_a_proposed_claim_falls_back_to_a_proposed_one(tmp_path):
    """Neither renders, so neither is shown through the other."""
    reg = registry(tmp_path, _proposed(id="planted.a", fallback="planted.b"),
                   _proposed(id="planted.b"))
    assert claims.none_proposed_renders(reg) == []


# ── 5. no two live claims conflict on one decision in overlapping regimes ────────────────────

def _on(decision, **over):
    return teaching(decision=decision, **over)


CONFLICTS = {
    "a superseded claim still live": (
        _on("planted.d", id="planted.old", verdict="recommended"),
        _on("planted.d", id="planted.new", verdict="recommended", supersedes=("planted.old",))),
    "two verdicts on one decision": (
        _on("planted.d", id="planted.a", verdict="recommended"),
        _on("planted.d", id="planted.b", verdict="refused")),
    "a supersedes naming no claim": (
        _on("planted.d", id="planted.new", supersedes=("planted.never_registered",)),),
    "a disjoint regime naming no claim": (
        _on("planted.d", id="planted.a", regime=Regime("r", disjoint_from=("planted.nowhere",))),),
}
AGREEMENTS = {
    "the superseded claim retracted": (
        _on("planted.d", id="planted.old", verdict="recommended",
            retracted="2026-10-16: superseded by planted.new"),
        _on("planted.d", id="planted.new", verdict="refused", supersedes=("planted.old",))),
    "regimes declared disjoint": (
        _on("planted.d", id="planted.a", verdict="recommended",
            regime=Regime("continuous outcomes", disjoint_from=("planted.b",))),
        _on("planted.d", id="planted.b", verdict="refused", regime=Regime("yes/no outcomes"))),
    "two decisions": (
        _on("planted.d", id="planted.a", verdict="recommended"),
        _on("planted.e", id="planted.b", verdict="refused")),
    "one verdict on one decision": (
        _on("planted.d", id="planted.a", verdict="noted"),
        _on("planted.d", id="planted.b", verdict="noted")),
    "a parameter beside a rule on one decision": (
        _on("planted.d", id="planted.a", verdict="refused"),
        _on("planted.d", id="planted.b", verdict="")),
}


@pytest.mark.parametrize("case", sorted(CONFLICTS))
def test_5_fires_on_two_live_claims_that_conflict(tmp_path, case):
    found = claims.no_conflicts(registry(tmp_path, *CONFLICTS[case]))
    assert len(found) == 1


@pytest.mark.parametrize("case", sorted(AGREEMENTS))
def test_5_is_silent_where_the_claims_do_not_conflict(tmp_path, case):
    assert claims.no_conflicts(registry(tmp_path, *AGREEMENTS[case])) == []


# ── 6. the digit gate, over the claims' templates ────────────────────────────────────────────

def test_6_fires_on_a_digit_typed_into_a_template(tmp_path):
    reg = registry(tmp_path, teaching(card="Below a coefficient of variation of 0.6, use it."))
    found = claims.digit_gate(reg, allowlist={})
    assert len(found) == 1 and "planted.your_call" in found[0] and "0.6" in found[0]


def test_6_counts_a_superscript_as_a_digit(tmp_path):
    reg = registry(tmp_path, teaching(methods="Scored by R² on held-out rows."))
    assert len(claims.digit_gate(reg, allowlist={})) == 1


def test_6_fires_on_a_digit_beside_a_citation_but_outside_it(tmp_path):
    reg = registry(tmp_path, teaching(card="It is named (below 0.6, Friedman 2001)."))
    assert len(claims.digit_gate(reg, allowlist={})) == 1


@pytest.mark.parametrize("template", [
    "Below a coefficient of variation of {cv}, use it.",
    "Below a coefficient of variation of {cv:.1f}, use it.",
    "Boosting fits stagewise (Friedman 2001).",
    "Boosting fits stagewise (Friedman 2001, Ann Stat 29:1189).",
    "A plain sentence with no number in it.",
])
def test_6_is_silent_on_a_placeholder_a_citation_or_no_digit(tmp_path, template):
    reg = registry(tmp_path, teaching(card=template, params={"cv": Param(0.6, unit="ratio")}))
    assert claims.digit_gate(reg, allowlist={}) == []


def test_6_is_silent_on_a_template_allowlisted_with_its_tag(tmp_path):
    text = "Exported with its SHA-256."
    reg = registry(tmp_path, teaching(card=text))
    assert claims.digit_gate(reg, allowlist={text: "name"}) == []
    assert len(claims.digit_gate(reg, allowlist={})) == 1
