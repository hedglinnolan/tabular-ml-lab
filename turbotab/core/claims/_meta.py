"""The six meta-tests (CLAIMS_REGISTRY.md §6), each a function over a registry that returns its
violations, one sentence each, naming the claim. ``turbotab/core/tests/test_claims_meta.py`` runs
each on the default registry and on planted ones that break and keep it.

1. :func:`tests_exist` — every claim's tests exist and declare their truth. A claim whose code is
   not written yet may name tests not written yet (it is proposed, "ruled, not built"); once every
   enforcer it names resolves, a missing test is a violation: the code acts, untested.
2. :func:`no_self_pinning` — no claim test's file imports or names its claim's constant, or holds
   its own claim id as a string: through ``value()``, ``card()`` or the registry, however the call
   is spelled, its expected value could be the claim's own (§2.1).
3. :func:`grades_met` — every part meets §2.4's requirement for its grade (certified, by a declared
   test its file defines, whether or not the claim is built yet); above judgment it has a grader
   other than the author, and its claim has regime predicates (§2.2).
4. :func:`none_proposed_renders` — no built claim falls back to one that cannot render, and no
   proposed or retracted claim's sentence is among the strings the app shows.
5. :func:`no_conflicts` — no two live claims on one decision in overlapping regimes supersede one
   another or give different verdicts.
6. :func:`digit_gate` — every digit in a claim template outside its placeholders, and in every
   string the caller passes, is a placeholder, inside a citation, or allowlisted with its tag (§3).
"""
from __future__ import annotations

import ast
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

from turbotab.core.claims._digits import (CITATIONS, citation_keys, citation_spans, load_allowlist,
                                          stray_digits)
from turbotab.core.claims._registry import Registry, tree
from turbotab.core.claims._schema import ABOVE_JUDGMENT, Claim, fill, literal


def tests_exist(reg: Registry) -> list[str]:
    """Meta-test 1."""
    out = []
    for c in reg:
        if c.retracted:
            continue
        live = all(reg.resolves(e) for e in c.enforced_by)
        for t in c.tests:
            if live and not reg.test_exists(t.test):
                out.append(f"{c.id}: its code is live but its {t.side} test {t.test} does not "
                           "exist")
            kind, _, where = t.truth.partition(":")
            if kind == "derivation" and not (reg.root / where).is_file():
                out.append(f"{c.id}: the derivation {where} that {t.test} declares as its truth "
                           "does not exist")
    return out


def _pins(module: ast.Module, c: Claim) -> list[str]:
    """How the test file ``module`` pins claim ``c`` to itself.

    A string equal to the claim's id counts wherever it stands (an argument, a variable, a
    subscript), so ``value(ID, …)``, an alias of ``value``, ``card(ID)`` and ``registry()[ID]`` are
    caught alike; an id built at run time is not seen."""
    constants = {p.constant.partition(":")[2]: p.constant.partition(":")[0]
                 for p in c.params.values() if p.constant}
    found = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in constants:
                    found.append(f"imports {alias.name}")
                elif alias.name == "*" and node.module in constants.values():
                    found.append(f"imports everything from {node.module}")
        elif isinstance(node, ast.Name) and node.id in constants:
            found.append(f"names {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in constants:
            found.append(f"names {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in constants:
                found.append(f"names {node.value!r}")
            elif node.value == c.id:
                found.append(f"names its own claim id {c.id!r}")
    return sorted(set(found))


def no_self_pinning(reg: Registry) -> list[str]:
    """Meta-test 2."""
    out = []
    for c in reg:
        for path in sorted({t.test.partition("::")[0] for t in c.tests}):
            module = tree(reg.root / path)
            if module is None:
                continue
            for how in _pins(module, c):
                out.append(f"{c.id}: its test file {path} {how}, so it can take its expected "
                           "value from the claim it tests, not from known truth")
    return out


def _grade_unmet(reg: Registry, c: Claim, part, keys: frozenset[str]) -> str:
    b, grade = part.basis, part.grade
    if grade == "proved":
        if b.derivation and not (reg.root / b.derivation).is_file():
            return f"its derivation {b.derivation} does not exist"
        if not (b.theorem or b.derivation):
            return "proved needs a theorem or a re-derived derivation"
    elif grade == "certified":
        declared = {t.test for t in c.tests}
        if not b.tests:
            return "certified needs a Fires or Silent test with declared truth"
        if any(t not in declared for t in b.tests):
            return "certified by a test the claim does not declare"
        unwritten = [t for t in b.tests if not reg.test_exists(t)]
        if unwritten:
            return f"certified by a test that is not written ({', '.join(unwritten)})"
    elif grade == "replicated":
        if len({h.fixture for h in b.heldout}) < 2:
            return "replicated needs held-out evidence from at least two fixtures, with rates"
    elif grade == "sourced":
        if not b.cites or any(k not in keys for k in b.cites):
            return f"sourced needs citation keys that resolve, not {list(b.cites)}"
        if not b.where.strip():
            return "sourced needs the passage that says it (where)"
    elif grade == "judgment":
        if not b.reason.strip():
            return "judgment needs its reason"
    if grade in ABOVE_JUDGMENT:
        grader = (part.graded_by or c.graded_by).strip()
        if not grader:
            return f"{grade} needs a grader (graded_by)"
        if grader.casefold() == c.author.strip().casefold():
            return f"{grade} is graded by its author; above judgment, someone else grades"
    return ""


def grades_met(reg: Registry, citations: Path = CITATIONS) -> list[str]:
    """Meta-test 3."""
    keys = citation_keys(citations)
    out = []
    for c in reg:
        for part in c.parts:
            unmet = _grade_unmet(reg, c, part, keys)
            if unmet:
                out.append(f"{c.id}: part {part.name!r}: {unmet}")
        if any(p.grade in ABOVE_JUDGMENT for p in c.parts) and not c.regime.requires:
            out.append(f"{c.id}: a part above judgment needs the regime as predicates (§2.2)")
    return out


def none_proposed_renders(reg: Registry, shown: Iterable[str] = ()) -> list[str]:
    """Meta-test 4. ``shown`` is every string the app shows (its relations, teaching, …)."""
    out = []
    for c in reg:
        if not c.fallback or reg.status(c.id) != "built":
            continue
        chain, f = [c.id], c.fallback
        while f:
            if f not in reg:
                out.append(f"{c.id}: its fallback {f} is not registered")
                break
            if f in chain:
                out.append(f"{c.id}: its fallbacks loop ({' → '.join(chain)} → {f})")
                break
            status = reg.status(f)
            if status != "built":
                out.append(f"{c.id}: it falls back to {f}, which is {status} and cannot render")
                break
            chain.append(f)
            f = reg[f].fallback
    shown = list(shown)
    for c in reg:
        status = reg.status(c.id)
        if status == "built":
            continue
        for template in (c.card, c.methods):
            sentence = fill(template, c.params) if template else ""
            if sentence and any(sentence in s for s in shown):
                out.append(f"{c.id}: it is {status}, but its sentence is shown: {sentence!r}")
    return out


def no_conflicts(reg: Registry) -> list[str]:
    """Meta-test 5."""
    out = []
    for c in reg:
        for other in c.supersedes:
            if other not in reg:
                out.append(f"{c.id}: it supersedes {other}, which is not registered")
        for other in c.regime.disjoint_from:
            if other not in reg:
                out.append(f"{c.id}: its regime is disjoint from {other}, which is not registered")
    live = [c for c in reg if not c.retracted and c.decision]
    for a, b in combinations(live, 2):
        if a.decision != b.decision:
            continue
        if b.id in a.regime.disjoint_from or a.id in b.regime.disjoint_from:
            continue
        if b.id in a.supersedes or a.id in b.supersedes:
            new, old = (a, b) if b.id in a.supersedes else (b, a)
            out.append(f"{new.id}: it supersedes {old.id}, which is still live on {a.decision} in "
                       "an overlapping regime; retract it, or declare the regimes disjoint")
        elif a.verdict and b.verdict and a.verdict != b.verdict:
            out.append(f"{a.id}: its verdict {a.verdict!r} conflicts with {b.id}'s {b.verdict!r} "
                       f"on {a.decision} in an overlapping regime")
    return out


def digit_gate(reg: Registry, strings: Iterable[tuple[str, str]] = (),
               allowlist: Mapping[str, str] | None = None,
               citations: Path = CITATIONS) -> list[str]:
    """Meta-test 6. ``strings`` are ``(origin, text)`` pairs the app shows; a text equal to a
    claim's filled sentence is that claim's, and is gated by its template instead."""
    allowlist = load_allowlist() if allowlist is None else allowlist
    rendered = set()
    items: list[tuple[str, str, str]] = []  # origin, the text as allowlisted, the text gated
    for c in reg:
        for register in ("card", "methods"):
            template = getattr(c, register)
            if template:
                rendered.add(fill(template, c.params))
                items.append((f"claim:{c.id}.{register}", template, literal(template)))
    items += [(origin, text, text) for origin, text in strings if text not in rendered]
    out = []
    for origin, text, gated in items:
        if text in allowlist:
            continue
        stray = stray_digits(gated, citation_spans(gated, citations))
        if stray:
            out.append(f"{origin}: the digits {stray!r} are not a placeholder, a citation or an "
                       f"allowlisted string: {text[:100]!r}")
    return out


META_TESTS = (tests_exist, no_self_pinning, grades_met, none_proposed_renders, no_conflicts,
              digit_gate)
