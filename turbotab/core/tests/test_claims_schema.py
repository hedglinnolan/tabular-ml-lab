"""A claim is checked against the registry's vocabulary when it registers (CLAIMS_REGISTRY.md §1,
§2.1, §2.4), as a method contract is (``contracts.register_contract``).

Each guard is tested both ways: the conforming planted claim registers, and the same claim with one
field broken is refused, naming the field. Among them: above judgment, a part's grader reads
"<who>, YYYY-MM-DD"; both templates fill from the claim's own parameters (Python's ``format`` is
the truth); and a test's path is neither absolute nor leaves the repository root. Every claim
names its origin (engine, audit or swarm; R1_PLAN.md R1-G), a part's basis audit is an ``Audit``
and it and each of its referees read "<who>, YYYY-MM-DD" as a grader does, each names the
fingerprint of the part it read (``claims.checked``), a referee names the grade it judged, and its
verdict is upheld or not. A part may be retired (a sentence that no longer asserts it), but a
claim keeps one part it asserts, and its grade is its weakest asserted part's. The fingerprint
moves with everything a referee or an audit reads (the claim's sentences, parameters, regime and
decision, and the part's grade and basis), and with nothing else. The registry's copies of the
leash rungs and the purposes are held to the contracts' own, since the registry may not import
them.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from turbotab.core import claims
from turbotab.core.claims import (Audit, Basis, Claim, Fires, HeldOut, Param, Part, Referee,
                                  Regime, Registry, Silent, checked)
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


# A well-formed fingerprint (twelve lowercase hex digits), of no part in particular.
SEEN = "0123456789ab"

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
    "origin outside the vocabulary": dict(origin="agent"),
    "origin left blank": dict(origin=""),
    "basis audit with no date": dict(parts=(Part("value", grade="judgment", basis=Basis(reason="r"),
                                                 audited_by=Audit("a basis audit", SEEN)),)),
    "basis audit given as a string": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by="a basis audit, 2026-10-18"),)),
    "basis audit naming no fingerprint": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by=Audit("a basis audit, 2026-10-18")),)),
    "basis audit fingerprint not twelve hex digits": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by=Audit("a basis audit, 2026-10-18", "not a hash")),)),
    "basis audits given as an empty tuple": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"), audited_by=()),)),
    "basis audits holding a string": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by=(Audit("a basis audit, 2026-10-18", SEEN), "a domain reviewer, 2026-10-19")),)),
    "a later basis audit with no date": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by=(Audit("a basis audit, 2026-10-18", SEEN), Audit("a domain reviewer", SEEN))),)),
    "basis audits out of date order": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        audited_by=(Audit("a basis audit, 2026-10-19", SEEN),
                    Audit("a domain reviewer, 2026-10-18", SEEN))),)),
    "referee with no date": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A", "upheld", grade="judgment", checked=SEEN),)),)),
    "referee verdict outside the vocabulary": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A, 2026-10-18", "probably", grade="judgment",
                          checked=SEEN),)),)),
    "referee given as a string": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"), referees=("referee A, 2026-10-18",)),)),
    "referee naming no grade": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A, 2026-10-18", "upheld", checked=SEEN),)),)),
    "referee grade outside the vocabulary": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A, 2026-10-18", "upheld", grade="settled", checked=SEEN),)),)),
    "referee naming no fingerprint": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A, 2026-10-18", "upheld", grade="judgment"),)),)),
    "referee fingerprint in capitals": dict(parts=(Part(
        "value", grade="judgment", basis=Basis(reason="r"),
        referees=(Referee("referee A, 2026-10-18", "upheld", grade="judgment",
                          checked=SEEN.upper()),)),)),
    "every part retired": dict(parts=(Part("value", grade="retired"),)),
}


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_a_claim_outside_the_vocabulary_is_refused_at_registration(broken):
    with pytest.raises(ValueError):
        Registry([rule(**BROKEN[broken])])


def test_a_claim_that_names_no_origin_cannot_be_written():
    """``origin`` has no default: a claim written without one is a ``TypeError`` before it can
    register, so no claim lacks one."""
    fields = {f.name: getattr(rule(), f.name) for f in dataclasses.fields(Claim)
              if f.name != "origin"}
    with pytest.raises(TypeError, match="origin"):
        Claim(**fields)


@pytest.mark.parametrize("origin", ["engine", "audit", "swarm"])
def test_each_origin_registers(origin):
    assert claims.ORIGINS == ("engine", "audit", "swarm")
    assert Registry([rule(origin=origin)])["planted.few_clusters"].origin == origin


def test_a_dated_basis_audit_and_dated_referees_register():
    part = Part("value", grade="judgment", basis=Basis(reason="r"),
                audited_by=Audit("a basis audit, 2026-10-18", SEEN),
                referees=(Referee("referee A, 2026-10-18", "upheld", grade="judgment",
                                  checked=SEEN),
                          Referee("referee B, 2026-10-18", "not_upheld",
                                  fork_of="the orchestrator", grade="sourced", checked=SEEN)))
    assert Registry([rule(parts=(part,))])["planted.few_clusters"].parts == (part,)


def test_basis_audits_appended_in_date_order_register_and_the_last_is_in_force():
    """Audits are append-only, as referees are (R1_PLAN R1-M's package, item 3): a tuple of them,
    oldest first, each dated no earlier than the one before; the last is the audit in force. A
    part that names none, or one audit in today's form, is as it was."""
    first = Audit("theorist A, 2026-10-18", SEEN)
    later = Audit("a basis audit, 2026-10-18", SEEN)
    last = Audit("a domain reviewer, 2026-10-19", SEEN)
    part = Part("value", grade="judgment", basis=Basis(reason="r"), audited_by=(first, later, last))
    assert Registry([rule(parts=(part,))])["planted.few_clusters"].parts == (part,)
    assert claims.audits(part) == (first, later, last) and claims.audit_in_force(part) == last
    one = Part("value", grade="judgment", basis=Basis(reason="r"), audited_by=first)
    assert claims.audits(one) == (first,) and claims.audit_in_force(one) == first
    none = Part("value", grade="judgment", basis=Basis(reason="r"))
    assert claims.audits(none) == () and claims.audit_in_force(none) is None


def test_a_retired_part_registers_beside_one_asserted_and_the_grade_is_the_asserted_one_s():
    """A retired part is no grade: the claim shows the weakest part it still asserts."""
    claim = rule(parts=(Part("value", "judgment", Basis(reason="r")), Part("scope", "retired")))
    assert Registry([claim])["planted.few_clusters"].grade == "judgment"
    assert claims.RETIRED == "retired" and "retired" not in claims.GRADES


# ── the fingerprint of what a referee or a basis audit read ─────────────────────────────────

PASSAGE = Basis(cites=("friedman2001",), where="§10.10, p. 1203")


def _read():
    """A claim whose one part a referee reads, with a second part beside it."""
    return rule(parts=(Part("value", "sourced", PASSAGE, graded_by="a basis audit, 2026-10-18"),
                       Part("scope", "judgment", Basis(reason="r"))))


def _value(c):
    return checked(c, c.parts[0])


def _with_value(c, **over):
    return dataclasses.replace(c, parts=(dataclasses.replace(c.parts[0], **over), *c.parts[1:]))


READ = {
    "its card": lambda c: dataclasses.replace(c, card="Fewer than {few} clusters: wider."),
    "its methods sentence": lambda c: dataclasses.replace(c, methods="Below {few}, it is named."),
    "a parameter's value": lambda c: dataclasses.replace(
        c, params={"few": Param(12, unit="clusters", constant="planted.engine:FEW")}),
    "its regime's words": lambda c: dataclasses.replace(c, regime=dataclasses.replace(
        c.regime, words="any outcome")),
    "its regime's predicates": lambda c: dataclasses.replace(c, regime=dataclasses.replace(
        c.regime, requires=())),
    "its decision": lambda c: dataclasses.replace(c, decision="planted.other"),
    "its verdict": lambda c: dataclasses.replace(c, verdict="available"),
    "its options": lambda c: dataclasses.replace(c, options=("gee_corrected",)),
    "its goal": lambda c: dataclasses.replace(c, goal=("prediction",)),
    "its id": lambda c: dataclasses.replace(c, id="planted.other_claim"),
    "the part's grade": lambda c: _with_value(c, grade="certified"),
    "the part's basis": lambda c: _with_value(c, basis=Basis(cites=("friedman2001",),
                                                             where="p. 1190")),
    "the part's name": lambda c: _with_value(c, name="direction"),
}
NOT_READ = {
    "its grader": lambda c: _with_value(c, graded_by="a domain reviewer, 2026-10-19"),
    "the claim's grader": lambda c: dataclasses.replace(c,
                                                        graded_by="the orchestrator, 2026-10-19"),
    "its referees": lambda c: _with_value(c, referees=(
        Referee("referee A, 2026-10-19", "upheld", grade="sourced", checked=SEEN),)),
    "its basis audit": lambda c: _with_value(c, audited_by=Audit("a basis audit, 2026-10-19",
                                                                 SEEN)),
    "why it was regraded": lambda c: _with_value(c, regraded="the passage says it"),
    "another part": lambda c: dataclasses.replace(c, parts=(c.parts[0],
                                                            Part("scope", "unsupported"))),
    "its version": lambda c: dataclasses.replace(c, version=2),
    "its tests": lambda c: dataclasses.replace(c, tests=(SILENT, FIRES)),
}


def test_the_fingerprint_is_twelve_hex_digits_and_repeats():
    assert re.fullmatch(r"[0-9a-f]{12}", _value(_read()))
    assert _value(_read()) == _value(_read())


@pytest.mark.parametrize("change", sorted(READ))
def test_the_fingerprint_moves_with_what_a_referee_or_an_audit_reads(change):
    assert _value(READ[change](_read())) != _value(_read())


@pytest.mark.parametrize("change", sorted(NOT_READ))
def test_the_fingerprint_stays_where_only_its_review_or_another_part_changes(change):
    assert _value(NOT_READ[change](_read())) == _value(_read())


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
               rule(version=2),
               rule(origin="audit"),
               rule(parts=(Part("value", grade="judgment", basis=Basis(reason="a planted judgment"),
                                audited_by="a basis audit, 2026-10-18"),)),
               rule(parts=(Part("value", grade="judgment", basis=Basis(reason="a planted judgment"),
                                referees=(Referee("referee A, 2026-10-18", "upheld"),)),)),
               rule(parts=(Part("value", grade="judgment", basis=Basis(reason="a planted judgment"),
                                regraded="a planted reason"),))]
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
        graded_by="basis audit, 2026-10-15", author="the R0a builder", origin="engine", version=1)
    assert Registry([c])["trials.few_clusters_example"].grade == "judgment"


# ── above judgment, a grader reads '<who>, YYYY-MM-DD' (§1's example, §2.4) ─────────────────────

SOURCED = Part("value", grade="sourced", basis=Basis(cites=("friedman2001",), where="p. 3"))
UNDATED = {
    "no grader": dict(graded_by=""),
    "a grader with no date": dict(graded_by="a basis audit"),
    "a date with no grader": dict(graded_by="2026-10-16"),
    "a comma and a date with no grader": dict(graded_by=" , 2026-10-16"),
    "a date in another form": dict(graded_by="a basis audit, 16 October 2026"),
    "a date with no comma before it": dict(graded_by="a basis audit 2026-10-16"),
    "a date that is no day of the year": dict(graded_by="a basis audit, 2026-13-40"),
    "a date with a line after it": dict(graded_by="a basis audit, 2026-10-16\n"),
    "a part's own grader with no date over a dated claim grader": dict(
        parts=(dataclasses.replace(SOURCED, graded_by="a basis audit"),),
        graded_by="a basis audit, 2026-10-16"),
}
DATED = {
    "the claim's grader": dict(graded_by="a basis audit, 2026-10-16"),
    "section 1's own example": dict(graded_by="basis audit, 2026-10-15"),
    "a part's own grader": dict(
        parts=(dataclasses.replace(SOURCED, graded_by="a domain reviewer, 2026-10-16"),)),
    "a grader whose name holds a comma": dict(graded_by="a referee pair (A, B), 2026-10-16"),
    "an undated grader of a part at judgment": dict(
        parts=(Part("value", grade="judgment", basis=Basis(reason="r")),),
        graded_by="the R0a builder"),
}


@pytest.mark.parametrize("case", sorted(UNDATED))
def test_above_judgment_a_grader_not_written_who_then_date_is_refused_at_registration(case):
    with pytest.raises(ValueError, match=r"^planted\.few_clusters: part 'value' is graded sourced, "
                                         r"so its grader reads '<who>, YYYY-MM-DD'"):
        Registry([rule(**{"parts": (SOURCED,), **UNDATED[case]})])


@pytest.mark.parametrize("case", sorted(DATED))
def test_above_judgment_a_dated_grader_registers_and_at_judgment_no_date_is_needed(case):
    reg = Registry([rule(**{"parts": (SOURCED,), **DATED[case]})])
    assert "planted.few_clusters" in reg


# ── a template its own parameters cannot fill does not register ──────────────────────────────

@pytest.mark.parametrize("register", ["card", "methods"])
def test_a_template_its_own_parameters_cannot_fill_is_refused_naming_claim_and_register(register):
    """Python's format refuses the integer code ``d`` on a float, so ``{cv:d}`` on 0.6 cannot
    render; registration says so, with the claim and the register, before any sentence is asked
    for."""
    templates = {"card": "Below {cv}.", "methods": "Below {cv}.", register: "Below {cv:d}."}
    with pytest.raises(ValueError, match=rf"^planted\.few_clusters: the {register} template "):
        Registry([rule(**templates, params={"cv": Param(0.6, unit="ratio")})])


@pytest.mark.parametrize("template, value", [("Below {cv:.1f}.", 0.6), ("Below {cv:d}.", 6)])
def test_a_template_its_own_parameters_fill_registers(template, value):
    """The same code ``d`` fills an integer, and ``.1f`` fills 0.6."""
    reg = Registry([rule(card=template, methods=template,
                         params={"cv": Param(value, unit="ratio")})])
    assert "planted.few_clusters" in reg


# ── a test path stays inside the repository ──────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["../outside_test.py", "planted/../../outside_test.py",
                                  "./../outside_test.py", "/tmp/outside_test.py",
                                  "//host/outside_test.py"])
def test_a_test_path_that_is_absolute_or_leaves_the_root_is_refused(path):
    with pytest.raises(ValueError, match=r"^planted\.few_clusters: test .* is absolute or leaves "
                                         r"the repository root"):
        Registry([rule(tests=(Fires(f"{path}::test_x", "construction"), SILENT))])


@pytest.mark.parametrize("path", ["planted/test_rule.py", "./planted/test_rule.py",
                                  "planted/../planted/test_rule.py"])
def test_a_test_path_inside_the_root_registers(path):
    reg = Registry([rule(tests=(Fires(f"{path}::test_nine_clusters_name_the_concern",
                                      "construction"), SILENT))])
    assert "planted.few_clusters" in reg
