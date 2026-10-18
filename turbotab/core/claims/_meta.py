"""The eight meta-tests (CLAIMS_REGISTRY.md §6; R1_PLAN.md R1-G adds 7 and 8), each a function over
a registry that returns its violations, one sentence each, naming the claim.
``turbotab/core/tests/test_claims_meta.py`` runs each on the default registry and on planted ones
that break and keep it.

1. :func:`tests_exist` — every claim's tests exist and declare their truth. A claim whose code is
   not written yet may name tests not written yet (it is proposed, "ruled, not built"); once every
   enforcer it names resolves, a missing test is a violation: the code acts, untested. A claim
   that names no code has nothing to wait for, so its tests must exist already. A test marked to
   skip or xfail unconditionally never asserts, and does not count.
2. :func:`no_self_pinning` — no claim test's file imports or names its claim's constant, or holds
   its own claim id as a string: through ``value()``, ``card()`` or the registry, however the call
   is spelled, its expected value could be the claim's own (§2.1).
3. :func:`grades_met` — every part meets §2.4's requirement for its grade (certified, by a declared
   test its file defines, whether or not the claim is built yet); above judgment its grader (the
   "<who>" of "<who>, YYYY-MM-DD") is not its author, compared as ``_schema.same_person`` does,
   which refuses in doubt, and its claim has regime predicates (§2.2). A part that names a basis
   audit (who checked its basis against the source, and when) names someone other than its author.
4. :func:`none_proposed_renders` — no built claim falls back to one that cannot render, and no
   proposed or retracted claim's sentence is among the strings the app shows.
5. :func:`no_conflicts` — no two live claims on one decision in overlapping regimes supersede one
   another or give different verdicts.
6. :func:`digit_gate` — every digit in a claim template outside its placeholders, and in every
   string the caller passes, is a placeholder, inside a citation, or allowlisted with its tag (§3).
7. :func:`swarm_gate` — a part of a live claim the research swarm proposed is graded by someone
   other than its author once it leaves unsupported; above judgment it passes the two-referee gate
   (``_schema.referee_gate``: two distinct referees who upheld it, neither its author nor a fork,
   and none who counts against it), meets its grade's own requirement (§2.4, as meta-test 3 reads
   it) and names a basis audit by someone other than its author (§8).
8. :func:`grades_recorded` — every part's grade is the grade of its last line in the history
   (``history.jsonl``, which the generator writes): a grade moves only through a line with its
   date, its grader and its reason, which says the grade it moved from, downgrades included. Each
   grade line reads so, the first for a part moving from no grade; and a swarm claim's every part
   entered unsupported.
"""
from __future__ import annotations

import ast
import json
import re
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from turbotab.core.claims._digits import (CITATIONS, citation_keys, citation_spans, load_allowlist,
                                          stray_digits)
from turbotab.core.claims._registry import GENERATOR, Registry, tree
from turbotab.core.claims._schema import (ABOVE_JUDGMENT, GRADED_FORM, GRADES, Claim, fill, grader,
                                          literal, referee_gate, same_person)

_NEVER_RUNS = "is marked to skip or xfail unconditionally, so it never asserts"


def tests_exist(reg: Registry) -> list[str]:
    """Meta-test 1."""
    out = []
    for c in reg:
        if c.retracted:
            continue
        live = all(reg.resolves(e) for e in c.enforced_by)
        for t in c.tests:
            state = reg.test_state(t.test) if live else "runs"
            if state != "runs":
                gap = "does not exist" if state == "missing" else _NEVER_RUNS
                if c.enforced_by:
                    out.append(f"{c.id}: its code is live but its {t.side} test {t.test} {gap}")
                else:
                    out.append(f"{c.id}: it names no code, so its {t.side} test {t.test} must "
                               f"already exist, but it {gap}")
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
            return ("certified by a test that is not written or never runs "
                    f"({', '.join(unwritten)})")
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
        by = part.graded_by or c.graded_by
        who = grader(by)
        if who is None:  # registration refuses it (_schema.check), so only a claim set past it
            return f"{grade} needs a grader that reads {GRADED_FORM}, not {by!r}"
        if same_person(who, c.author):
            return (f"{grade} is graded by {who!r}, which may be its author {c.author!r}; above "
                    "judgment, someone else grades")
    if part.audited_by:
        auditor = grader(part.audited_by)
        if auditor is None:  # registration refuses it, so only a claim set past it
            return f"its basis audit reads {GRADED_FORM}, not {part.audited_by!r}"
        if same_person(auditor, c.author):
            return (f"its basis audit is by {auditor!r}, which may be its author {c.author!r}; "
                    "someone else checks a basis against its source")
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


def swarm_gate(reg: Registry, citations: Path = CITATIONS) -> list[str]:
    """Meta-test 7."""
    keys = citation_keys(citations)
    out = []
    for c in reg:
        if c.origin != "swarm" or c.retracted:
            continue
        for part in c.parts:
            if part.grade == "unsupported":
                continue
            where = f"{c.id}: part {part.name!r} is graded {part.grade}"
            who = grader(part.graded_by or c.graded_by)
            if who is None or same_person(who, c.author):
                named = ("it names no grader" if who is None
                         else f"{who!r}, its grader, may be its author")
                out.append(f"{where}, but a swarm claim's part leaves unsupported only as graded "
                           f"by someone other than its author, {GRADED_FORM}: {named}")
            if part.grade not in ABOVE_JUDGMENT:
                continue
            out += [f"{where}: {lack}" for lack in referee_gate(part.referees, c.author)]
            unmet = _grade_unmet(reg, c, part, keys)
            if unmet:
                out.append(f"{where}, but its grade's own requirement is unmet: {unmet}")
            auditor = grader(part.audited_by) if part.audited_by else None
            if auditor is None:
                out.append(f"{where}, but it names no basis audit, {GRADED_FORM}: who checked its "
                           "basis against the source, and when")
            elif same_person(auditor, c.author):
                out.append(f"{where}, but its basis audit is by {auditor!r}, who may be its author")
    return out


# A grade line's fields, each a string; a version line has no "part".
GRADE_LINE: tuple[str, ...] = ("id", "part", "grade", "was", "date", "grader", "reason")
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _a_day(text: str) -> bool:
    if not _DAY.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _line_faults(n: int, record: dict[str, Any], prev: dict[str, Any] | None) -> list[str]:
    """What is wrong with grade line ``n``, given the part's line before it."""
    where = f"{record['id']}: part {record['part']!r}: history line {n}"
    out = []
    if record["grade"] not in GRADES:
        out.append(f"{where} grades it {record['grade']!r}, not one of {GRADES}")
    if not _a_day(record["date"]):
        out.append(f"{where} is dated {record['date']!r}, not YYYY-MM-DD")
    if not record["grader"].strip():
        out.append(f"{where} names no grader")
    if not record["reason"].strip():
        out.append(f"{where} gives no reason")
    if prev is None and record["was"]:
        out.append(f"{where} says it moved from {record['was']}, but it is the part's first line")
    elif prev is not None and record["was"] != prev["grade"]:
        out.append(f"{where} says it moved from {record['was'] or 'no grade'}, but the line "
                   f"before it graded it {prev['grade']}")
    elif record["grade"] == record["was"]:
        out.append(f"{where} moves it to the grade it moved from, {record['grade']}")
    return out


def grades_recorded(reg: Registry, history: Path | None = None) -> list[str]:
    """Meta-test 8. ``history`` is the registry's own (``Registry.history_path``) unless given."""
    path = Path(history) if history is not None else reg.history_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [f"there is no history at {path}: run {GENERATOR}, which records each part's grade "
                "with its date, grader and reason"]
    out: list[str] = []
    first: dict[tuple[str, str], dict[str, Any]] = {}
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            out.append(f"history line {n} does not read as JSON")
            continue
        if not isinstance(record, dict):
            out.append(f"history line {n} is not one JSON object")
            continue
        if "part" not in record:
            continue  # a version line
        lacking = [k for k in GRADE_LINE if not isinstance(record.get(k), str)]
        if lacking:
            out.append(f"history line {n}: a grade line gives {', '.join(GRADE_LINE)} as text, "
                       f"and it lacks {', '.join(lacking)}")
            continue
        key = (record["id"], record["part"])
        out += _line_faults(n, record, last.get(key))
        first.setdefault(key, record)
        last[key] = record
    for c in reg:
        for part in c.parts:
            key = (c.id, part.name)
            line = last.get(key)
            if line is None:
                out.append(f"{c.id}: part {part.name!r} is graded {part.grade}, but the history "
                           f"holds no grade line for it: run {GENERATOR}, which records it with "
                           "its date, grader and reason")
            elif line["grade"] != part.grade:
                out.append(f"{c.id}: part {part.name!r} is graded {part.grade}, but its last "
                           f"history line grades it {line['grade']}: a grade moves only through a "
                           f"line with its date, grader and reason, which {GENERATOR} writes")
            if c.origin == "swarm" and key in first and first[key]["grade"] != "unsupported":
                out.append(f"{c.id}: part {part.name!r}: a swarm claim's part enters unsupported, "
                           f"but its first history line grades it {first[key]['grade']}")
    return out


META_TESTS = (tests_exist, no_self_pinning, grades_met, none_proposed_renders, no_conflicts,
              digit_gate, swarm_gate, grades_recorded)
