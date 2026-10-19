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
   the author, of the part as it stands;
4. no proposed claim renders: no built claim falls back to one, and none is shown;
5. no two live claims conflict on one decision in overlapping regimes;
6. the digit gate, over every claim's templates outside their placeholders;
7. a swarm claim's part is graded (or retired) by someone other than its author, and above
   judgment it has two distinct referees who upheld it at its grade, neither its author nor a fork
   (ORCHESTRATOR.md §3), its grade's own requirement and a basis audit by someone other than its
   author (CLAIMS_REGISTRY.md §8); each referee and the audit read the part as it stands, and each
   uphold and the audit are dated no earlier than the day it took its previous grade, so a verdict
   on a lower grade or an older basis, or an uphold from before the part last moved, does not
   count, while a dissent on the part as it stands still blocks;
8. every part's grade is its last history line's, each line giving its date, grader and reason and
   the grade it moved from, dated no earlier than the line before it and naming a grader and date
   other than that line's, and a swarm claim's every part entered unsupported; a claim the history
   first recorded as the swarm's is still the swarm's; a swarm claim's author, the grader of each
   part above unsupported while it holds the grade that grader gave, and every referee as given
   stay as the history recorded them, so a referee is added, never edited or dropped (an engine or
   audit claim's are not held so); and a part the history holds that its claim no longer names was
   retired through a line (by someone other than a swarm claim's author), not dropped or renamed.

On the default registry, the meta-tests that read the app's strings (4 and 6) read every
``Relation.says`` and teaching string, as ``test_claims_digits.py`` collects them.

Every planted case is built here from files the test writes, so its truth is the construction.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta

import pytest

from turbotab.core import claims
from turbotab.core.claims import (Audit, Basis, Fires, HeldOut, Param, Part, Referee, Regime,
                                  Registry, Silent)
from turbotab.core.reference import claims as reference
from turbotab.core.tests.test_claims_fixtures import (DERIVATION, FIRES, HERE, SILENT, bound,
                                                       registry, rule, swarm, teaching,
                                                       write_history)
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
    "a basis audit by its author": bound(_graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by=Audit("the R0a builder, 2026-10-16")))),
    "a basis audit of another version of the part": _graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by=Audit("a domain reviewer, 2026-10-18", checked=claims.checked(
                 _graded(Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 9"))),
                 Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 9")))))),
    # Audits are appended, oldest first, and the last is the one in force.
    "a basis audit in force by its author, after someone else's": bound(_graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by=(Audit("a domain reviewer, 2026-10-16"),
                         Audit("the R0a builder, 2026-10-18"))))),
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
    "a basis audit by someone else, of the part as it stands": bound(_graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by=Audit("a domain reviewer, 2026-10-18")))),
    "a basis audit in force by someone else, after its author's": bound(_graded(
        Part("value", "sourced", Basis(cites=("friedman2001",), where="p. 3"),
             audited_by=(Audit("the R0a builder, 2026-10-16"),
                         Audit("a domain reviewer, 2026-10-18"))))),
    "a retired part, which asserts nothing": _graded(
        Part("value", "judgment", Basis(reason="we use it")), Part("scope", "retired"),
        graded_by="", requires=()),
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


# ── the history, as the generator writes it (meta-tests 7 and 8 read it) ─────────────────────

def _line(grade, was="", *, id=ID, part="value", date="2026-10-18", grader="a basis audit",
          reason="a planted reason"):
    return {"id": id, "part": part, "grade": grade, "was": was, "date": date, "grader": grader,
            "reason": reason}


def _history(reg, *lines):
    reg.history_path.parent.mkdir(parents=True, exist_ok=True)
    reg.history_path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return reg


def _generate(reg, today):
    """Run the generator on ``reg`` as its own history's writer, ``today`` the day a line with no
    dated grader is recorded on; its exit code."""
    path = reg.history_path
    return reference.main(["--out", str(path.with_name("CLAIMS_REFERENCE.md")), "--history",
                           str(path), "--date", today], registry=reg)


# ── 7. a swarm claim's grade above judgment passes the two-referee gate (§8) ──────────────────

SWARM = "planted.swarm_found"
GRADER = "the orchestrator, 2026-10-18"
AUDIT = Audit("a basis audit, 2026-10-18")
FRESH_A = Referee("referee A, 2026-10-18", "upheld")
FRESH_B = Referee("referee B, 2026-10-18", "upheld")
PASSAGE = Basis(cites=("friedman2001",), where="§10.10, p. 1203")
ENTERED = "2026-10-17"  # the day the planted swarm part entered the history, unsupported
STALE = "0123456789ab"  # a well-formed fingerprint of no version of the planted part


def _promoted(**over):
    """A swarm claim whose one part is graded sourced through the whole gate, a field replaced. Its
    referees and basis audit are bound to the part as it stands (``bound``), except where a
    replaced field names another grade or fingerprint."""
    fields = dict(name="value", grade="sourced", basis=PASSAGE, graded_by=GRADER,
                  referees=(FRESH_A, FRESH_B), audited_by=AUDIT)
    return bound(swarm(parts=(Part(**{**fields, **over}),)))


def _entered(reg, *lines):
    """``reg`` with the history the generator wrote when its swarm part entered, unsupported, on
    ENTERED, and any grade lines after it."""
    return _history(reg, _line("unsupported", id=SWARM, date=ENTERED, grader="theorist A"),
                    *lines)


def _as(referee, **over):
    return dataclasses.replace(referee, **over)


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
    # Fork status and authorship are the agent's, not one verdict's (RG's verifier, P1): a name
    # one record marks stops counting in every record of the part, compared as same_person does.
    "an upholder another of the part's records declares a fork of its author": (
        _promoted(referees=(FRESH_A, FRESH_B, _as(FRESH_A, fork_of="theorist A"))),
        "'referee A' is marked a fork, or possibly its author, by another record of the part"),
    "an upholder another of the part's records declares a fork of another agent": (
        _promoted(referees=(FRESH_A, FRESH_B, _as(FRESH_B, fork_of="the orchestrator"))),
        "'referee B' is marked a fork, or possibly its author, by another record of the part"),
    "an upholder whose name another record of the part gives as possibly its author's": (
        _promoted(referees=(Referee("theorist B, 2026-10-18", "upheld"), FRESH_A,
                            Referee("theorist, 2026-10-18", "upheld"))),
        "'theorist B' is marked a fork, or possibly its author, by another record of the part"),
    "two upholders and a third referee who did not uphold": (
        _promoted(referees=(FRESH_A, FRESH_B, Referee("referee C, 2026-10-18", "not_upheld"))),
        "did not uphold"),
    "a pair, one of whom did not uphold": (
        _promoted(referees=(FRESH_A, Referee("referee B, 2026-10-18", "not_upheld"))),
        "did not uphold"),
    "two upholders of a lower grade": (
        _promoted(grade="proved", basis=Basis(derivation=DERIVATION),
                  referees=(_as(FRESH_A, grade="sourced"), _as(FRESH_B, grade="sourced"))),
        "judged it as sourced, not proved"),
    "two upholders of another version of the part": (
        _promoted(referees=(_as(FRESH_A, checked=STALE), _as(FRESH_B, checked=STALE))),
        "read another version"),
    "two upholders dated before the part entered": (
        _promoted(referees=(Referee("referee A, 2020-01-01", "upheld"),
                            Referee("referee B, 2020-01-01", "upheld"))),
        f"before {ENTERED}"),
    "a dissent on the part as it stands, given before it took its previous grade": (
        _promoted(referees=(FRESH_A, FRESH_B, Referee("referee C, 2020-01-01", "not_upheld"))),
        "'referee C' did not uphold"),
    "its grade's own requirement unmet": (
        _promoted(basis=Basis(cites=("friedman2001",))), "requirement"),
    "no basis audit": (_promoted(audited_by=""), "basis audit"),
    "a basis audit by its author": (_promoted(audited_by=Audit("theorist A, 2026-10-18")),
                                    "basis audit"),
    "a basis audit of another version of the part": (
        _promoted(audited_by=Audit(AUDIT.who, STALE)), "basis audit read another version"),
    "a basis audit dated before the part entered": (
        _promoted(audited_by=Audit("a basis audit, 2020-01-01")), f"before {ENTERED}"),
    "a basis audit in force by its author, after someone else's": (
        _promoted(audited_by=(AUDIT, Audit("theorist A, 2026-10-18"))),
        "its basis audit is by 'theorist A'"),
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
    "two fresh upholders beside a fork of its author who is neither of them": _promoted(
        referees=(FRESH_A, FRESH_B, Referee("referee F, 2026-10-18", "upheld",
                                            fork_of="theorist A"))),
    "a basis audit in force by someone else, after its author's": _promoted(
        audited_by=(Audit("theorist A, 2026-10-18"), AUDIT)),
    "referees and a basis audit dated the day the part entered": _promoted(
        referees=(Referee(f"referee A, {ENTERED}", "upheld"),
                  Referee(f"referee B, {ENTERED}", "upheld")),
        audited_by=Audit(f"a basis audit, {ENTERED}")),
    "a dissent on a lower grade beside two upholders of the part as it stands": _promoted(
        referees=(FRESH_A, FRESH_B, Referee("referee C, 2026-10-18", "not_upheld",
                                            grade="judgment"))),
    "a dissent on another version beside two upholders of the part as it stands": _promoted(
        referees=(FRESH_A, FRESH_B, Referee("referee C, 2026-10-18", "not_upheld",
                                            checked=STALE))),
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
    found = claims.swarm_gate(_entered(registry(tmp_path, claim)))
    assert found and all(v.startswith(f"{SWARM}: part 'value'") for v in found)
    assert any(phrase in v for v in found), found


@pytest.mark.parametrize("case", sorted(GATE_MET))
def test_7_is_silent_where_a_swarm_part_passes_the_gate_or_no_gate_applies(tmp_path, case):
    assert claims.swarm_gate(_entered(registry(tmp_path, GATE_MET[case]))) == []


def test_7_s_planted_pass_meets_meta_test_3_too(tmp_path):
    """The planted pass meets meta-test 3 as well, so 7's silence on it is not a planted error."""
    assert claims.grades_met(registry(tmp_path, _promoted())) == []


def test_7_fires_where_the_history_holds_no_line_for_a_promoted_part(tmp_path):
    """With no line, nothing dates the part's previous grade, so its referees cannot be held to
    it: in doubt, the gate refuses."""
    found = claims.swarm_gate(registry(tmp_path, _promoted()))
    assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'value'")
    assert "holds no line" in found[0]


SINCE_JUDGED = "2026-10-15"  # the planted part took judgment on this day, after entering


@pytest.mark.parametrize("recorded", [False, True], ids=["sourced not yet recorded",
                                                           "sourced recorded"])
def test_7_holds_referees_to_the_day_the_part_took_its_previous_grade(tmp_path, recorded):
    """The part entered on 2026-10-10 and took judgment on SINCE_JUDGED; its referees for sourced
    count from that day, whether or not the generator has recorded sourced yet."""
    lines = [_line("unsupported", id=SWARM, date="2026-10-10", grader="theorist A"),
             _line("judgment", "unsupported", id=SWARM, date=SINCE_JUDGED,
                   grader="the orchestrator")]
    if recorded:
        lines.append(_line("sourced", "judgment", id=SWARM, date="2026-10-18",
                           grader="the orchestrator"))
    early = (Referee("referee A, 2026-10-12", "upheld"), Referee("referee B, 2026-10-12", "upheld"))
    found = claims.swarm_gate(_history(registry(tmp_path, _promoted(referees=early)), *lines))
    assert found and all(f"before {SINCE_JUDGED}" in v for v in found), found
    on_time = (Referee(f"referee A, {SINCE_JUDGED}", "upheld"), FRESH_B)
    assert claims.swarm_gate(_history(registry(tmp_path, _promoted(referees=on_time)),
                                      *lines)) == []


def test_7_fires_where_referees_and_an_audit_are_carried_to_a_higher_grade_on_a_new_basis(
        tmp_path):
    """The verifier's sequence (RG-1), through the generator: the part enters unsupported, rises
    to sourced through the whole gate, then to proved on a derivation with the same referees and
    basis audit, who never read proved or the derivation. Re-refereed and re-audited, its earlier
    referees and basis audit kept beside the new as the history recorded them (audits are
    appended, the last in force), it passes meta-tests 7 and 8."""
    reg = registry(tmp_path, swarm())
    assert _generate(reg, "2026-10-18") == 0
    sourced = _promoted(referees=(Referee("referee A, 2026-10-19", "upheld"),
                                  Referee("referee B, 2026-10-19", "upheld")),
                        audited_by=Audit("a basis audit, 2026-10-19"),
                        graded_by="the orchestrator, 2026-10-19", regraded="two referees upheld it")
    reg = registry(tmp_path, sourced)
    assert _generate(reg, "2026-10-19") == 0 and claims.swarm_gate(reg) == []
    was = sourced.parts[0]
    carried = swarm(parts=(dataclasses.replace(
        was, grade="proved", basis=Basis(derivation=DERIVATION),
        graded_by="the orchestrator, 2026-10-20", regraded="a derivation proves it"),))
    reg = registry(tmp_path, carried)
    assert _generate(reg, "2026-10-20") == 0
    found = claims.swarm_gate(reg)
    assert sum("judged it as sourced, not proved" in v for v in found) == 1
    assert sum("basis audit read another version" in v for v in found) == 1
    assert sum("two distinct referees" in v for v in found) == 1
    redone = _promoted(grade="proved", basis=Basis(derivation=DERIVATION),
                       referees=(*was.referees, Referee("referee C, 2026-10-20", "upheld"),
                                 Referee("referee D, 2026-10-20", "upheld")),
                       audited_by=(was.audited_by, Audit("a basis audit, 2026-10-20")),
                       graded_by="the orchestrator, 2026-10-20",
                       regraded="a derivation proves it")
    reg = registry(tmp_path, redone)
    assert _generate(reg, "2026-10-20") == 0
    assert claims.swarm_gate(reg) == [] and claims.grades_recorded(reg) == []


def _with_scope(scope):
    return swarm(parts=(Part("value", "unsupported"), scope))


def test_7_fires_where_a_swarm_part_is_retired_by_its_author(tmp_path):
    claim = _with_scope(Part("scope", "retired", graded_by="theorist A, 2026-10-18",
                             regraded="the sentence no longer says it"))
    found = claims.swarm_gate(_entered(registry(tmp_path, claim)))
    assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'scope' is retired")
    assert "someone other than its author" in found[0]


def test_7_is_silent_where_a_swarm_part_is_retired_by_someone_else(tmp_path):
    claim = _with_scope(Part("scope", "retired", graded_by=GRADER,
                             regraded="the sentence no longer says it"))
    assert claims.swarm_gate(_entered(registry(tmp_path, claim))) == []


# ── 8. no grade differs from its last history line ───────────────────────────────────────────

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
    "a first line that says it moved from a grade": {"was": "judgment"},
    "a first line that retires the part": {"grade": "retired"},
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
                   _line("judgment"), _line("unsupported", was="certified", date="2026-10-19"))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and "history line 2" in found[0] and "certified" in found[0]


def test_8_is_silent_on_a_chain_of_moves_whose_last_is_the_grade_in_force(tmp_path):
    """A downgrade is a line like any other; the last line for a part is the one in force, and a
    version line between grade lines is not read as one."""
    reg = registry(tmp_path, rule(parts=(Part("value", "unsupported"),)))
    version = {"id": ID, "version": 1, "hash": "0" * 64, "claim": {}}
    _history(reg, version, _line("judgment", date="2026-10-16", grader="the R0a builder"),
             _line("sourced", was="judgment", date="2026-10-17"), version,
             _line("unsupported", was="sourced", date="2026-10-19", grader="a domain reviewer"))
    assert claims.grades_recorded(reg) == []


# A part's second line, after its first, a judgment by "a basis audit" on 2026-10-18 (RG-2): the
# line moves it to unsupported, by the line before's grader and date, unless a case says otherwise.
OUT_OF_ORDER = {
    "dated before the line before it": (dict(date="2019-05-01", grader="a domain reviewer"),
                                        "before the line before it"),
    "the grader and date of the line before it": (dict(), "grader and date of the line before"),
    "the grade it moved from, by another grader on a later day": (
        dict(grade="judgment", date="2026-10-19", grader="a domain reviewer"),
        "moves it to the grade it moved from, judgment"),
}
IN_ORDER = {
    "another grader, the same day": dict(grader="a domain reviewer"),
    "the same grader, a later day": dict(date="2026-10-19"),
}


@pytest.mark.parametrize("case", sorted(OUT_OF_ORDER))
def test_8_fires_on_a_second_line_out_of_order_or_that_moves_the_part_nowhere(
        tmp_path, case):
    """The part is graded as the second line grades it, so the line is the only fault."""
    over, phrase = OUT_OF_ORDER[case]
    over = dict(over)
    grade = over.pop("grade", "unsupported")
    part = Part("value", grade, Basis(reason="r") if grade == "judgment" else Basis())
    reg = _history(registry(tmp_path, rule(parts=(part,))),
                   _line("judgment"), _line(grade, was="judgment", **over))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and "history line 2" in found[0] and phrase in found[0], found


@pytest.mark.parametrize("case", sorted(IN_ORDER))
def test_8_is_silent_on_a_change_by_another_grader_or_on_a_later_day(tmp_path, case):
    reg = _history(registry(tmp_path, rule(parts=(Part("value", "unsupported"),))),
                   _line("judgment"), _line("unsupported", was="judgment", **IN_ORDER[case]))
    assert claims.grades_recorded(reg) == []


def test_8_fires_where_a_swarm_part_entered_above_unsupported(tmp_path):
    graded = Part("value", "judgment", Basis(reason="r"), graded_by=GRADER)
    reg = _history(registry(tmp_path, swarm(parts=(graded,))),
                   _line("judgment", id=SWARM, grader="the orchestrator"))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'value'")
    assert "enters unsupported" in found[0]


def test_8_is_silent_where_a_swarm_part_entered_unsupported_and_was_graded_after(tmp_path):
    graded = Part("value", "judgment", Basis(reason="r"), graded_by=GRADER)
    reg = _history(registry(tmp_path, swarm(parts=(graded,))),
                   _line("unsupported", id=SWARM, grader="theorist A"),
                   _line("judgment", was="unsupported", id=SWARM, grader="the orchestrator"))
    assert claims.grades_recorded(reg) == []


def _version(origin=None, id=SWARM):
    claim = {"id": id} if origin is None else {"id": id, "origin": origin}
    return {"id": id, "version": 1, "hash": "0" * 64, "claim": claim}


SWARM_LINES = (_line("unsupported", id=SWARM, grader="theorist A"),)


@pytest.mark.parametrize("now", ["engine", "audit"])
def test_8_fires_where_a_claim_first_recorded_as_the_swarm_s_is_relabeled(tmp_path, now):
    """Relabeled, a swarm claim would leave meta-test 7's gate; the history keeps its origin."""
    reg = _history(registry(tmp_path, swarm(origin=now)), _version("swarm"), *SWARM_LINES,
                   _version(now))
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and found[0].startswith(SWARM) and "swarm" in found[0]
    assert now in found[0]


@pytest.mark.parametrize("first, now", [("swarm", "swarm"), (None, "engine"),
                                        ("engine", "audit"), ("audit", "engine")])
def test_8_is_silent_where_a_claim_s_origin_never_left_the_swarm(tmp_path, first, now):
    """A version written before claims named an origin (R0's) holds none, and is no swarm's."""
    reg = _history(registry(tmp_path, swarm(origin=now)), _version(first), *SWARM_LINES)
    assert claims.grades_recorded(reg) == []


# A part is retired through a line, never dropped or renamed (RG-4). The planted claim's history
# holds 'value' at judgment and 'scope' at sourced; each case is the claim's parts now, its lines
# after those two, the part the violation names, and the phrase it names it by.
TWO_PARTS = (_line("judgment"), _line("sourced", part="scope", date="2026-10-16"))
JUDGED = Part("value", "judgment", Basis(reason="r"))
NARROWED = Part("value_scope", "judgment", Basis(reason="r"))
RETIRING = Part("scope", "retired", graded_by="a domain reviewer, 2026-10-19",
                regraded="the sentence no longer says it")
RETIRED_LINE = _line("retired", "sourced", part="scope", date="2026-10-19",
                     grader="a domain reviewer", reason="the sentence no longer says it")
DROPS = {
    "a part dropped": ((JUDGED,), ()),
    "a part renamed": ((JUDGED, NARROWED), (_line("judgment", part="value_scope"),)),
}
RETIREMENTS = {
    "a part retired and still named": ((JUDGED, RETIRING), (RETIRED_LINE,)),
    "a part retired, then dropped": ((JUDGED,), (RETIRED_LINE,)),
}


@pytest.mark.parametrize("case", sorted(DROPS))
def test_8_fires_where_a_part_the_history_holds_is_dropped_or_renamed(tmp_path, case):
    parts, lines = DROPS[case]
    reg = _history(registry(tmp_path, rule(parts=parts)), *TWO_PARTS, *lines)
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and found[0].startswith(f"{ID}: part 'scope'"), found
    assert "no longer names it" in found[0] and "sourced" in found[0]


@pytest.mark.parametrize("case", sorted(RETIREMENTS))
def test_8_is_silent_where_a_part_left_through_a_line_that_retires_it(tmp_path, case):
    parts, lines = RETIREMENTS[case]
    reg = _history(registry(tmp_path, rule(parts=parts)), *TWO_PARTS, *lines)
    assert claims.grades_recorded(reg) == []


@pytest.mark.parametrize("by, fires", [("theorist A", True), ("the orchestrator", False)])
def test_8_holds_a_swarm_part_s_retirement_to_someone_other_than_its_author(tmp_path, by, fires):
    """Once a retired part leaves its claim, meta-test 7 no longer sees it; its line still names
    who retired it, and for a swarm claim that is not its author."""
    reg = _history(registry(tmp_path, swarm()), *SWARM_LINES,
                   _line("unsupported", id=SWARM, part="scope", grader="theorist A"),
                   _line("retired", "unsupported", id=SWARM, part="scope", date="2026-10-19",
                         grader=by))
    found = claims.grades_recorded(reg)
    if fires:
        assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'scope'")
        assert "may be its author" in found[0]
    else:
        assert found == []


# A swarm claim's review stays as the history recorded it (RG-V2-1): its author, each part's
# grader while its grade stands, every referee as given and every basis audit (audits are appended,
# the last in force). Each case is the planted swarm claim's versions after it entered, each
# written by the generator a day after the one before, except the last, which makes the edit:
# the generator refuses it, so it is appended as the generator would write it without that
# refusal (:func:`_by_hand`), as a history written by hand holds it. Meta-test 7, which reads the
# claim as it now stands, passes it.
SELF = Referee("theorist A, 2026-10-18", "upheld")  # its author, as a referee
FORK = Referee("referee F, 2026-10-18", "upheld", fork_of="theorist A")
DISSENT = Referee("referee C, 2026-10-18", "not_upheld")
LATER = Referee("referee D, 2026-10-19", "upheld")
NEW_PASSAGE = Basis(cites=("friedman2001",), where="§10.11, p. 1204")
NEW_AUDIT = Audit("a basis audit, 2026-10-19")
OWN = Part("value", "judgment", Basis(reason="r"), graded_by="theorist A, 2026-10-18",
           regraded="the author judged it")


def _reviewed(referees, author="theorist A", basis=PASSAGE, audit=AUDIT):
    """The planted swarm claim, its part graded sourced, each referee and the audit it names bound
    to the part as it stands where it names no grade or fingerprint (``bound``)."""
    part = Part("value", "sourced", basis, graded_by=GRADER, referees=referees, audited_by=audit,
                regraded="two referees upheld it")
    return bound(swarm(author=author, parts=(part,)))


def _day(n):
    return (date.fromisoformat(ENTERED) + timedelta(days=n)).isoformat()


def _through(tmp_path, *versions, last_by_hand=False):
    """Each version of one planted claim in turn, written by the generator a day after the one
    before, from ENTERED (the last by :func:`_by_hand` where ``last_by_hand``); the registry of the
    last."""
    for day, c in enumerate(versions):
        reg = registry(tmp_path, c)
        if last_by_hand and day == len(versions) - 1:
            _by_hand(reg, _day(day))
        else:
            assert _generate(reg, _day(day)) == 0, c.id
    return reg


def _by_hand(reg, today):
    """Append to ``reg``'s history the version lines and grade lines the generator would, with
    none of its refusals: a history written before a refusal, or by hand, holds such lines."""
    path = reg.history_path
    todo = reference.pending(reg, reference.read_history(path))
    graded, _ = reference.grade_lines(reg, reference.read_grades(path), today)
    with path.open("a", encoding="utf-8") as f:
        for c in reg:
            f.writelines(reference.line(r) for r in todo if r["id"] == c.id)
            f.writelines(reference.line(g) for g in graded if g["id"] == c.id)


DISSENTED = _reviewed((FRESH_A, FRESH_B, DISSENT))
REVIEWED = _reviewed((FRESH_A, FRESH_B))
RECORD_EDITS = {
    "its author relabeled, so its own uphold and audit count": (
        (_reviewed((SELF, FRESH_B), author="theorist Z", audit=Audit("theorist A, 2026-10-18")),),
        "first recorded its author as 'theorist A'"),
    "a fork's fork_of cleared": (
        (_reviewed((FRESH_A, FORK)), _reviewed((FRESH_A, _as(FORK, fork_of="")))),
        "the agent it was forked from as ''"),
    "a recorded dissent deleted": ((DISSENTED, _reviewed((FRESH_A, FRESH_B))),
                                   "no longer names that referee"),
    "a recorded dissent turned to an uphold": (
        (DISSENTED, _reviewed((FRESH_A, FRESH_B, _as(DISSENT, verdict="upheld")))),
        "its verdict as 'upheld'"),
    "two upholds rebound to a basis they never read": (
        (REVIEWED, _reviewed((FRESH_A, FRESH_B), basis=NEW_PASSAGE,
                             audit=(REVIEWED.parts[0].audited_by, NEW_AUDIT))),
        "the fingerprint it read"),
    "a part's grader relabeled while its grade stands": (
        (swarm(parts=(OWN,)), swarm(parts=(dataclasses.replace(OWN, graded_by=GRADER),))),
        "records that grade by theorist A, 2026-10-18"),
    "its author's basis audit relabeled, its fingerprint kept": (
        (_reviewed((FRESH_A, FRESH_B), audit=Audit("theorist A, 2026-10-18")), REVIEWED),
        "recorded the basis audit by 'theorist A, 2026-10-18'"),
    "a basis audit rebound to a basis it never read": (
        (REVIEWED, _reviewed((*REVIEWED.parts[0].referees, LATER,
                              Referee("referee E, 2026-10-19", "upheld")),
                             basis=NEW_PASSAGE, audit=AUDIT)),
        "the basis audit by 'a basis audit, 2026-10-18' now gives the fingerprint it read"),
}
RECORD_KEPT = {
    "a referee added": (REVIEWED, _reviewed((FRESH_A, FRESH_B, LATER))),
    "a dissent kept, answered by a new basis two more referees upheld": (
        DISSENTED,
        _reviewed((*DISSENTED.parts[0].referees, LATER, Referee("referee E, 2026-10-19", "upheld")),
                  basis=NEW_PASSAGE, audit=(DISSENTED.parts[0].audited_by, NEW_AUDIT))),
    "a basis audit added after the one recorded": (
        REVIEWED, _reviewed((FRESH_A, FRESH_B), audit=(REVIEWED.parts[0].audited_by, NEW_AUDIT))),
}


@pytest.mark.parametrize("case", sorted(RECORD_EDITS))
def test_8_fires_where_a_swarm_claim_s_review_is_edited_after_the_history_recorded_it(
        tmp_path, case):
    versions, phrase = RECORD_EDITS[case]
    reg = _through(tmp_path, swarm(), *versions, last_by_hand=True)
    assert claims.swarm_gate(reg) == [], "the edit fools meta-test 7, which reads only the claim"
    found = claims.grades_recorded(reg)
    assert found and all(v.startswith(SWARM) and phrase in v for v in found), found


@pytest.mark.parametrize("case", sorted(RECORD_EDITS))
def test_the_generator_refuses_an_edited_swarm_review_and_writes_nothing(tmp_path, capsys, case):
    """INBOX (RG's repairs): the generator refuses the version that edits what the history
    recorded of a swarm claim's review, before it appends any line, so no history it writes holds
    the edit; ``--check`` exits 1 on it too. A review added to is written
    (:func:`test_8_is_silent_where_a_swarm_claim_s_later_version_adds_to_its_review`)."""
    versions, phrase = RECORD_EDITS[case]
    _through(tmp_path, swarm(), *versions[:-1])
    reg = registry(tmp_path, versions[-1])
    out = reg.history_path.with_name("CLAIMS_REFERENCE.md")
    before = (reg.history_path.read_bytes(), out.read_bytes())
    capsys.readouterr()
    assert _generate(reg, _day(len(versions))) == 1
    assert phrase in capsys.readouterr().err
    assert (reg.history_path.read_bytes(), out.read_bytes()) == before
    assert reference.main(["--check", "--out", str(out), "--history", str(reg.history_path)],
                          registry=reg) == 1
    assert phrase in capsys.readouterr().err


def test_8_fires_where_a_swarm_part_s_grader_is_cleared_while_its_grade_stands(tmp_path):
    """Meta-test 7 refuses it too, as naming no grader; meta-test 8 holds it to its line."""
    judged = dataclasses.replace(OWN, graded_by=GRADER)
    reg = _through(tmp_path, swarm(), swarm(parts=(judged,)),
                   swarm(parts=(dataclasses.replace(judged, graded_by=""),)), last_by_hand=True)
    found = claims.grades_recorded(reg)
    assert len(found) == 1 and found[0].startswith(f"{SWARM}: part 'value' names no dated grader")
    assert "records that grade by the orchestrator, 2026-10-18" in found[0]


@pytest.mark.parametrize("case", sorted(RECORD_KEPT))
def test_8_is_silent_where_a_swarm_claim_s_later_version_adds_to_its_review(tmp_path, case):
    """A later verdict is a referee added beside those recorded, and a later audit one appended
    after them; one on a version since changed stays, and no longer counts. The generator writes
    each version (``_through`` holds it to exit 0)."""
    reg = _through(tmp_path, swarm(), *RECORD_KEPT[case])
    assert claims.grades_recorded(reg) == []
    assert claims.swarm_gate(reg) == []


@pytest.mark.parametrize("origin", ["engine", "audit"])
def test_8_is_silent_where_an_engine_or_audit_claim_s_author_or_grader_is_edited(tmp_path, origin):
    """Meta-test 7 gates only a swarm claim, so only a swarm claim's review is held as recorded."""
    edited = Part("value", "judgment", Basis(reason="a planted judgment"),
                  graded_by="a domain reviewer, 2026-10-19")
    reg = _through(tmp_path, rule(origin=origin),
                   rule(origin=origin, author="the R0b builder", parts=(edited,)))
    assert claims.grades_recorded(reg) == []


# ── 7 again, through the generator: a later record that marks an upholder a fork ─────────────

@pytest.mark.parametrize("added, counts", [
    (_as(FRESH_A, fork_of="theorist A"), False),
    (FORK, True),
], ids=["a record marking referee A a fork of its author", "a fork of its author named apart"])
def test_7_holds_a_name_one_record_marks_a_fork_to_every_record_of_the_part(tmp_path, added,
                                                                            counts):
    """RG's verifier's probe (INBOX, open: RM): referees A and B uphold the part fresh, through
    the generator, and a later version adds a record of A as a fork of its author. A fork is the
    agent's, not one verdict's, so A stops counting in every record of the part and the gate
    refuses; a fork named apart from A and B leaves both counting. The record added is a referee
    added, so meta-test 8 and the generator keep it either way."""
    reg = _through(tmp_path, swarm(), _reviewed((FRESH_A, FRESH_B)))
    assert claims.swarm_gate(reg) == []
    reg = registry(tmp_path, _reviewed((FRESH_A, FRESH_B, added)))
    assert _generate(reg, "2026-10-19") == 0 and claims.grades_recorded(reg) == []
    found = claims.swarm_gate(reg)
    if counts:
        assert found == []
    else:
        assert len(found) == 1 and "two distinct referees" in found[0], found
        assert ("'referee A' is marked a fork, or possibly its author, by another record of the "
                "part") in found[0]
