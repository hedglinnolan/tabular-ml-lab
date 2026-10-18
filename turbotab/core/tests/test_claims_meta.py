"""The eight meta-tests (CLAIMS_REGISTRY.md §6; R0_PLAN.md R0a acceptance 1; R1_PLAN.md R1-G), each
run three ways: on the default registry (each must pass), on a planted registry that breaks it (it
must name the violation) and on a planted registry that keeps it (it must name none).

1. every claim's tests exist and declare their truth (once its code is live, or at once where it
   names no code), and none is marked to skip or xfail unconditionally;
2. no claim test imports or names its own claim's constant, or names its own claim's id (so it
   cannot read the claim's value, sentence or entry, however it spells the call);
3. every part's grade meets §2.4's requirement (a certified part's test is declared and written),
   with a grader other than the author above judgment, however the author's name is written (its
   case, spacing, a leading "the" or a parenthetical), and regime predicates wherever a part is
   graded above judgment (§2.2), and a basis audit, where a part names one, by someone other than
   the author;
4. no proposed claim renders: no built claim falls back to one, and none is shown;
5. no two live claims conflict on one decision in overlapping regimes;
6. the digit gate, over every claim's templates outside their placeholders;
7. a swarm claim's part is graded by someone other than its author, and above judgment it has two
   distinct referees who upheld it, neither its author nor a fork (ORCHESTRATOR.md §3), its grade's
   own requirement and a basis audit by someone other than its author (CLAIMS_REGISTRY.md §8);
8. every part's grade is its last history line's, each line giving its date, grader and reason and
   the grade it moved from, and a swarm claim's every part entered unsupported.

On the default registry, the meta-tests that read the app's strings (4 and 6) read every
``Relation.says`` and teaching string, as ``test_claims_digits.py`` collects them.

Every planted case is built here from files the test writes, so its truth is the construction.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from turbotab.core import claims
from turbotab.core.claims import (Basis, Fires, HeldOut, Param, Part, Referee, Regime, Registry,
                                  Silent)
from turbotab.core.tests.test_claims_fixtures import (DERIVATION, FIRES, HERE, SILENT, registry,
                                                       rule, swarm, teaching, write_history)
from turbotab.core.tests.test_claims_digits import relation_says, teaching_strings

ID = "planted.few_clusters"
OTHER = "a basis audit, 2026-10-16"
REQUIRES = (f"{HERE}:outcome_is_binary",)
UNWRITTEN = Fires("planted/test_rule.py::test_not_written", truth="construction")


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


def test_there_are_eight_meta_tests_in_the_order_section_6_and_r1_g_give():
    assert [m.__name__ for m in claims.META_TESTS] == [
        "tests_exist", "no_self_pinning", "grades_met", "none_proposed_renders", "no_conflicts",
        "digit_gate", "swarm_gate", "grades_recorded"]


# ── 1. every claim's tests exist and declare their truth ─────────────────────────────────────

def test_1_fires_where_live_code_names_a_test_that_does_not_exist(tmp_path):
    gone = Silent("planted/test_rule.py::test_never_written", "construction")
    found = claims.tests_exist(registry(tmp_path, rule(tests=(FIRES, SILENT, gone))))
    assert len(found) == 1 and ID in found[0] and "test_never_written" in found[0]


def test_1_fires_where_a_derivation_truth_names_no_file(tmp_path):
    found = claims.tests_exist(registry(tmp_path, rule(
        tests=(Fires(FIRES.test, "derivation:notes/missing.md"), SILENT))))
    assert len(found) == 1 and "notes/missing.md" in found[0]


def test_1_names_the_live_code_where_the_claim_s_enforcers_resolve(tmp_path):
    gone = Silent("planted/test_rule.py::test_never_written", "construction")
    found = claims.tests_exist(registry(tmp_path, rule(tests=(FIRES, SILENT, gone))))
    assert found == ["planted.few_clusters: its code is live but its silent test "
                     "planted/test_rule.py::test_never_written does not exist"]


def test_1_says_a_claim_that_names_no_code_must_already_have_its_tests(tmp_path):
    """With no ``enforced_by`` there is no code to wait for, so a missing test is a violation now;
    the message says so, not that code is live."""
    gone = Fires("planted/test_rule.py::test_never_written", "construction")
    found = claims.tests_exist(registry(tmp_path, teaching(tests=(gone,))))
    assert found == ["planted.your_call: it names no code, so its fires test "
                     "planted/test_rule.py::test_never_written must already exist, but it does "
                     "not exist"]


def test_1_says_a_test_marked_to_skip_never_asserts(tmp_path):
    reg = registry(tmp_path, rule())
    (reg.root / "planted" / "test_rule.py").write_text(
        "import pytest\n\n\ndef test_nine_clusters_name_the_concern():\n    assert 9 < 10\n\n\n"
        "@pytest.mark.skip\ndef test_forty_clusters_say_nothing():\n    assert not 40 < 10\n",
        encoding="utf-8")
    assert claims.tests_exist(reg) == [
        "planted.few_clusters: its code is live but its silent test "
        "planted/test_rule.py::test_forty_clusters_say_nothing is marked to skip or xfail "
        "unconditionally, so it never asserts"]


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
    "certified by a declared test that is not written": dataclasses.replace(
        _graded(Part("value", "certified", Basis(tests=(UNWRITTEN.test,)))),
        tests=(UNWRITTEN, SILENT), enforced_by=("planted.engine:not_yet",)),
    "replicated on one fixture": _graded(
        Part("value", "replicated", Basis(heldout=(HeldOut("nhanes", 40, 0.9),
                                                   HeldOut("nhanes", 12, 0.8))))),
    "sourced by a key that does not resolve": _graded(
        Part("value", "sourced", Basis(cites=("nobody1999",), where="p. 3"))),
    "sourced with no passage": _graded(Part("value", "sourced", Basis(cites=("friedman2001",)))),
    "judgment with no reason": _graded(Part("value", "judgment"), graded_by=""),
    "above judgment graded by its author, dated": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")),
        graded_by="the R0a builder, 2026-10-16"),
    "a part graded by its author over another grader": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             graded_by="the R0a builder, 2026-10-16")),
    "above judgment with no regime predicate": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3")), requires=()),
    "a basis audit by its author": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by="the R0a builder, 2026-10-16")),
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
        graded_by="the R0a builder, 2026-10-16"),
    "a basis audit by someone else": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by="a domain reviewer, 2026-10-18")),
}


@pytest.mark.parametrize("case", sorted(UNMET))
def test_3_fires_on_a_grade_its_basis_does_not_meet(tmp_path, case):
    found = claims.grades_met(registry(tmp_path, UNMET[case]))
    assert len(found) == 1 and ID in found[0]


@pytest.mark.parametrize("case", sorted(MET))
def test_3_is_silent_on_a_grade_its_basis_meets(tmp_path, case):
    assert claims.grades_met(registry(tmp_path, MET[case])) == []


# The author is "the R0a builder" (test_claims_fixtures.rule). A grader is the part before its date,
# compared with the author once each is casefolded, its whitespace collapsed, a leading "the" and
# any parenthetical dropped; equal, or either inside the other, is the author grading.
SELF_GRADERS = {
    "the author, dated": "the R0a builder, 2026-10-16",
    "the author without its article": "R0a builder, 2026-10-16",
    "the author in other case and spacing": "The  R0a   BUILDER, 2026-10-16",
    "the author with a parenthetical": "the R0a builder (self-check), 2026-10-16",
    "a parenthetical alone": "(self), 2026-10-16",
    "a name inside the author's": "builder, 2026-10-16",
    "a name holding the author's": "the R0a builder and a basis audit, 2026-10-16",
    "a name that holds the author's only once its article is dropped": "the builder, 2026-10-16",
    "a bare self-reference": "self, 2026-10-16",
    "the author named as the author": "the author, 2026-10-16",
    "the author hyphenated": "R0a-builder, 2026-10-16",
    "the author run together": "R0abuilder, 2026-10-16",
}
OTHER_GRADERS = {
    "a basis audit": OTHER,
    "section 1's own example": "basis audit, 2026-10-15",
    "a domain reviewer": "a domain reviewer (methods), 2026-10-16",
    "a referee pair": "a referee pair (A, B), 2026-10-16",
}


def _sourced(grader, level):
    part = Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"))
    if level == "part":
        return _graded(dataclasses.replace(part, graded_by=grader))
    return _graded(part, graded_by=grader)


@pytest.mark.parametrize("level", ["claim", "part"])
@pytest.mark.parametrize("case", sorted(SELF_GRADERS))
def test_3_fires_where_the_author_grades_above_judgment_however_it_is_written(
        tmp_path, case, level):
    found = claims.grades_met(registry(tmp_path, _sourced(SELF_GRADERS[case], level)))
    assert len(found) == 1 and ID in found[0] and "author" in found[0]


@pytest.mark.parametrize("level", ["claim", "part"])
@pytest.mark.parametrize("case", sorted(OTHER_GRADERS))
def test_3_is_silent_where_someone_else_grades_with_a_date(tmp_path, case, level):
    assert claims.grades_met(registry(tmp_path, _sourced(OTHER_GRADERS[case], level))) == []


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


# ── 7. a swarm claim's grade above judgment passes the two-referee gate (§8) ──────────────────

SWARM = "planted.swarm_found"
GRADER = "the orchestrator, 2026-10-18"
AUDIT = "a basis audit, 2026-10-18"
FRESH_A = Referee("referee A, 2026-10-18", "upheld")
FRESH_B = Referee("referee B, 2026-10-18", "upheld")
PASSAGE = Basis(cites=("friedman2001",), where="§10.10, p. 1203")


def _promoted(**over):
    """A swarm claim whose one part is graded sourced through the whole gate, a field replaced."""
    fields = dict(name="value", grade="sourced", basis=PASSAGE, graded_by=GRADER,
                  referees=(FRESH_A, FRESH_B), audited_by=AUDIT)
    return swarm(parts=(Part(**{**fields, **over}),))


# Each case breaks one piece of the gate; the phrase is how the violation names that piece.
GATE_UNMET = {
    "no referee": (_promoted(referees=()), "two distinct referees"),
    "one referee": (_promoted(referees=(FRESH_A,)), "two distinct referees"),
    "two upholders who are one person, written twice": (
        _promoted(referees=(FRESH_A, Referee("Referee  a, 2026-10-19", "upheld"))),
        "two distinct referees"),
    "a referee who is its author": (
        _promoted(referees=(FRESH_A, Referee("theorist A, 2026-10-18", "upheld"))),
        "may be its author"),
    "a referee named as its author's fork": (
        _promoted(referees=(FRESH_A, Referee("theorist A (fork 2), 2026-10-18", "upheld"))),
        "may be its author"),
    "a referee declared a fork of its author": (
        _promoted(referees=(FRESH_A, Referee("referee C, 2026-10-18", "upheld",
                                             fork_of="theorist A"))),
        "a fork of its author"),
    "a referee declared a fork of another agent": (
        _promoted(referees=(FRESH_A, Referee("referee C, 2026-10-18", "upheld",
                                             fork_of="the orchestrator"))),
        "a fork of 'the orchestrator'"),
    "two upholders and a third referee who did not uphold": (
        _promoted(referees=(FRESH_A, FRESH_B, Referee("referee C, 2026-10-18", "not_upheld"))),
        "did not uphold"),
    "a pair, one of whom did not uphold": (
        _promoted(referees=(FRESH_A, Referee("referee B, 2026-10-18", "not_upheld"))),
        "did not uphold"),
    "its grade's own requirement unmet": (
        _promoted(basis=Basis(cites=("friedman2001",))), "requirement"),
    "no basis audit": (_promoted(audited_by=""), "basis audit"),
    "a basis audit by its author": (_promoted(audited_by="theorist A, 2026-10-18"), "basis audit"),
    "proved by a derivation, with no referee": (
        _promoted(grade="proved", basis=Basis(derivation=DERIVATION), referees=()),
        "two distinct referees"),
    "judgment graded by its author": (
        swarm(parts=(Part("value", "judgment", Basis(reason="r"),
                          graded_by="theorist A, 2026-10-18"),)),
        "someone other than its author"),
    "judgment naming no grader": (
        swarm(parts=(Part("value", "judgment", Basis(reason="r")),)),
        "someone other than its author"),
}
GATE_MET = {
    "the whole gate": _promoted(),
    "the whole gate, proved by a derivation another re-derived": _promoted(
        grade="proved", basis=Basis(derivation=DERIVATION)),
    "two fresh upholders beside the author's and a fork's dissent, which do not count": _promoted(
        referees=(FRESH_A, FRESH_B, Referee("theorist A, 2026-10-18", "not_upheld"),
                  Referee("referee C, 2026-10-18", "not_upheld", fork_of="theorist A"))),
    "judgment graded by someone else": swarm(parts=(
        Part("value", "judgment", Basis(reason="r"), graded_by=GRADER),)),
    "unsupported, as it entered": swarm(),
    "an engine claim's sourced part, with no referee": swarm(
        origin="engine", parts=(Part("value", "sourced", PASSAGE, graded_by=GRADER),)),
    "an audit claim's sourced part, with no referee": swarm(
        origin="audit", parts=(Part("value", "sourced", PASSAGE, graded_by=GRADER),)),
    "a retracted swarm claim, which renders nowhere": swarm(
        parts=(Part("value", "sourced", PASSAGE, graded_by=GRADER),),
        retracted="2026-10-18: withdrawn by the swarm"),
}


@pytest.mark.parametrize("case", sorted(GATE_UNMET))
def test_7_fires_where_a_swarm_part_misses_a_piece_of_the_gate(tmp_path, case):
    claim, phrase = GATE_UNMET[case]
    found = claims.swarm_gate(registry(tmp_path, claim))
    assert found and all(v.startswith(f"{SWARM}: part 'value'") for v in found)
    assert any(phrase in v for v in found), found


@pytest.mark.parametrize("case", sorted(GATE_MET))
def test_7_is_silent_where_a_swarm_part_passes_the_gate_or_no_gate_applies(tmp_path, case):
    assert claims.swarm_gate(registry(tmp_path, GATE_MET[case])) == []


def test_7_s_planted_pass_meets_meta_test_3_too(tmp_path):
    """The planted pass meets meta-test 3 as well, so 7's silence on it is not a planted error."""
    assert claims.grades_met(registry(tmp_path, _promoted())) == []


# ── 8. no grade differs from its last history line ───────────────────────────────────────────

def _line(grade, was="", *, id=ID, part="value", date="2026-10-18", grader="a basis audit",
          reason="a planted reason"):
    return {"id": id, "part": part, "grade": grade, "was": was, "date": date, "grader": grader,
            "reason": reason}


def _history(reg, *lines):
    reg.history_path.parent.mkdir(parents=True, exist_ok=True)
    reg.history_path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return reg


def test_8_is_silent_on_the_history_the_generator_wrote(tmp_path):
    reg = registry(tmp_path, rule(), teaching(), swarm())
    write_history(reg)
    assert claims.grades_recorded(reg) == []


def test_8_fires_where_a_grade_is_edited_after_its_history_line(tmp_path):
    reg = registry(tmp_path, rule())
    write_history(reg)
    edited = Registry([rule(parts=(Part("value", "unsupported"),))], root=reg.root)
    found = claims.grades_recorded(edited)
    assert len(found) == 1 and found[0].startswith(f"{ID}: part 'value' is graded unsupported")
    assert "judgment" in found[0]


def test_8_fires_where_a_part_has_no_grade_line(tmp_path):
    reg = registry(tmp_path, rule())
    write_history(reg)
    more = Registry([rule(parts=(*rule().parts, Part("scope", "judgment", Basis(reason="r"))))],
                    root=reg.root)
    found = claims.grades_recorded(more)
    assert len(found) == 1 and "part 'scope'" in found[0] and "no grade line" in found[0]


def test_8_fires_where_there_is_no_history(tmp_path):
    found = claims.grades_recorded(registry(tmp_path, rule()))
    assert len(found) == 1 and "no history" in found[0]


HAND_EDITS = {
    "a reason left blank": {"reason": ""},
    "a grader left blank": {"grader": ""},
    "a date that is no day": {"date": "2026-13-40"},
    "a date in another form": {"date": "18 October 2026"},
    "a grade outside the vocabulary": {"grade": "settled"},
    "a line that moves to the grade it moved from": {"was": "judgment"},
}


@pytest.mark.parametrize("case", sorted(HAND_EDITS))
def test_8_fires_on_a_grade_line_edited_by_hand(tmp_path, case):
    reg = _history(registry(tmp_path, rule()), {**_line("judgment"), **HAND_EDITS[case]})
    found = claims.grades_recorded(reg)
    assert found and any("history line 1" in v for v in found), found


def test_8_is_silent_on_the_same_line_as_the_generator_writes_it(tmp_path):
    """The other side of the hand edits: the unedited planted line passes."""
    assert claims.grades_recorded(_history(registry(tmp_path, rule()), _line("judgment"))) == []


def test_8_fires_on_a_line_that_does_not_read_or_lacks_a_field(tmp_path):
    reg = _history(registry(tmp_path, rule()), _line("judgment"))
    with reg.history_path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps({"id": ID, "part": "value", "grade": "judgment"}) + "\n")
    found = claims.grades_recorded(reg)
    assert any("history line 2" in v for v in found) and any("history line 3" in v for v in found)


def test_8_fires_where_a_line_moves_from_a_grade_the_line_before_did_not_give(tmp_path):
    reg = _history(registry(tmp_path, rule(parts=(Part("value", "unsupported"),))),
                   _line("judgment"), _line("unsupported", was="certified"))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and "history line 2" in found[0] and "certified" in found[0]


def test_8_is_silent_on_a_chain_of_moves_whose_last_is_the_grade_in_force(tmp_path):
    """A downgrade is a line like any other; the last line for a part is the one in force, and a
    version line between grade lines is not read as one."""
    reg = registry(tmp_path, rule(parts=(Part("value", "unsupported"),)))
    version = {"id": ID, "version": 1, "hash": "0" * 64, "claim": {}}
    _history(reg, version, _line("judgment"), _line("sourced", was="judgment"), version,
             _line("unsupported", was="sourced"))
    assert claims.grades_recorded(reg) == []


def test_8_fires_where_a_swarm_part_entered_above_unsupported(tmp_path):
    graded = Part("value", "judgment", Basis(reason="r"), graded_by=GRADER)
    reg = _history(registry(tmp_path, swarm(parts=(graded,))), _line("judgment", id=SWARM))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'value'")
    assert "enters unsupported" in found[0]


def test_8_is_silent_where_a_swarm_part_entered_unsupported_and_was_graded_after(tmp_path):
    graded = Part("value", "judgment", Basis(reason="r"), graded_by=GRADER)
    reg = _history(registry(tmp_path, swarm(parts=(graded,))),
                   _line("unsupported", id=SWARM, grader="theorist A"),
                   _line("judgment", was="unsupported", id=SWARM, grader="the orchestrator"))
    assert claims.grades_recorded(reg) == []
