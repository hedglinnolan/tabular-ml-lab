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
   audit (who checked its basis against the source, and when) names someone other than its
   author, who checked the part as it stands (its fingerprint, ``_schema.checked``). A retired
   part asserts nothing, so has no requirement.
4. :func:`none_proposed_renders` — no built claim falls back to one that cannot render, and no
   proposed or retracted claim's sentence is among the strings the app shows.
5. :func:`no_conflicts` — no two live claims on one decision in overlapping regimes supersede one
   another or give different verdicts.
6. :func:`digit_gate` — every digit in a claim template outside its placeholders, and in every
   string the caller passes, is a placeholder, inside a citation, or allowlisted with its tag (§3).
7. :func:`swarm_gate` — a part of a live claim the research swarm proposed is graded (or retired)
   by someone other than its author once it leaves unsupported; above judgment it passes the
   two-referee gate (``_schema.referee_gate``: two distinct referees who upheld it, neither its
   author nor a fork, a name any of the part's records marks a fork or possibly its author
   counting in none of them, and none who counts against it), meets its grade's own requirement
   (§2.4, as meta-test 3 reads it) and names a basis audit by someone other than its author (§8).
   A referee
   or the audit counts only where it was on the part as it stands: the referee judged the grade
   the part holds, and each read the part's fingerprint now (``_schema.checked``). An uphold and
   the audit also count only where dated no earlier than the day the part took its previous
   grade, read from the history, while a dissent on the part as it stands blocks whenever it was
   given; where the history holds no line for the part, nothing dates that day, and the gate
   refuses.
8. :func:`grades_recorded` — every part's grade is the grade of its last line in the history
   (``history.jsonl``, which the generator writes): a grade moves only through a line with its
   date, its grader and its reason, which says the grade it moved from, downgrades included. Each
   grade line reads so, the first for a part moving from no grade, and each after it is dated no
   earlier than the line before it and names a grader and date other than that line's; a swarm
   claim's every part entered unsupported; and a claim whose first version line the history holds
   as the swarm's is still the swarm's, so no relabeling takes it out of meta-test 7's gate (a
   version written before claims named an origin holds none, and is no swarm's). A swarm claim's
   review also stays as the history's version lines recorded it, since meta-test 7 reads only the
   claim as it now stands: its author is the one the first version that names an author gives; a
   part graded above unsupported names the grader and date of its last grade line while it holds
   that line's grade; and every referee a version recorded on a part it still names is still
   named, each field as recorded, so a later verdict is a referee added beside the others. What
   was edited before the generator wrote it, the history never saw. A part the history holds that
   its claim no longer names left it through a line that retires it, never dropped or renamed; a
   swarm claim's part is retired by someone other than its author.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import re
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from turbotab.core.claims._digits import (CITATIONS, citation_keys, citation_spans, load_allowlist,
                                          stray_digits)
from turbotab.core.claims._registry import GENERATOR, Registry, tree
from turbotab.core.claims._schema import (ABOVE_JUDGMENT, GRADED_FORM, GRADES, RETIRED, Audit,
                                          Claim, Referee, checked, dated, fill, grader, literal,
                                          referee_gate, same_person)

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


def _grade_unmet(reg: Registry, c: Claim, part, keys: frozenset[str],
                 with_audit: bool = True) -> str:
    """What the part's grade lacks of its requirement and its grader, and, unless ``with_audit``
    is False (meta-test 7 reads the audit itself), of a basis audit it names; "" where nothing."""
    b, grade = part.basis, part.grade
    if grade == RETIRED:
        return ""
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
    audit = part.audited_by
    if with_audit and audit:
        auditor = grader(audit.who) if isinstance(audit, Audit) else None
        if auditor is None:  # registration refuses it, so only a claim set past it
            return f"its basis audit is an Audit that reads {GRADED_FORM}, not {audit!r}"
        if same_person(auditor, c.author):
            return (f"its basis audit is by {auditor!r}, which may be its author {c.author!r}; "
                    "someone else checks a basis against its source")
        if audit.checked != checked(c, part):
            return (f"its basis audit read another version of the part, since changed (its "
                    "sentence, regime, decision, grade or basis): re-audit it, or name no audit")
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


def _since(held: list[dict[str, Any]], grade: str) -> str | None:
    """The day a part took the grade before ``grade``, from its grade lines ``held``, oldest
    first: the date of the line before its last where the last records ``grade``, or of its last
    where ``grade`` is not recorded yet. None where the history holds no line for the part, or the
    line's date is no day."""
    if not held:
        return None
    line = held[-2] if held[-1]["grade"] == grade and len(held) > 1 else held[-1]
    return line["date"] if _a_day(line["date"]) else None


def swarm_gate(reg: Registry, citations: Path = CITATIONS,
               history: Path | None = None) -> list[str]:
    """Meta-test 7. ``history`` is the registry's own (``Registry.history_path``) unless given."""
    keys = citation_keys(citations)
    held = _read_history(Path(history) if history is not None else reg.history_path).parts
    out = []
    for c in reg:
        if c.origin != "swarm" or c.retracted:
            continue
        for part in c.parts:
            if part.grade == "unsupported":
                continue
            where = (f"{c.id}: part {part.name!r} is retired" if part.grade == RETIRED
                     else f"{c.id}: part {part.name!r} is graded {part.grade}")
            who = grader(part.graded_by or c.graded_by)
            if who is None or same_person(who, c.author):
                named = ("it names no grader" if who is None
                         else f"{who!r}, its grader, may be its author")
                leaves = ("is retired only by" if part.grade == RETIRED
                          else "leaves unsupported only as graded by")
                out.append(f"{where}, but a swarm claim's part {leaves} someone other than its "
                           f"author, {GRADED_FORM}: {named}")
            if part.grade not in ABOVE_JUDGMENT:
                continue
            now = checked(c, part)
            since = _since([r for _, r in held.get((c.id, part.name), [])], part.grade)
            if since is None:
                out.append(f"{where}, but the history holds no line for it, so nothing dates the "
                           "day it took its previous grade, from which its referees and basis "
                           f"audit count: a swarm claim's part enters unsupported through "
                           f"{GENERATOR}, and is graded after")
            out += [f"{where}: {lack}"
                    for lack in referee_gate(part.referees, c.author, part.grade, now, since)]
            unmet = _grade_unmet(reg, c, part, keys, with_audit=False)
            if unmet:
                out.append(f"{where}, but its grade's own requirement is unmet: {unmet}")
            audit = part.audited_by if isinstance(part.audited_by, Audit) else None
            when = dated(audit.who) if audit is not None else None
            if when is None:
                out.append(f"{where}, but it names no basis audit, an Audit reading {GRADED_FORM}: "
                           "who checked its basis against the source, and when")
            elif same_person(when[0], c.author):
                out.append(f"{where}, but its basis audit is by {when[0]!r}, who may be its "
                           "author")
            elif audit.checked != now:
                out.append(f"{where}, but its basis audit read another version of it, since "
                           "changed (its sentence, regime, decision, grade or basis)")
            elif since is not None and when[1] < since:
                out.append(f"{where}, but its basis audit is dated {when[1]}, before {since}, "
                           "when it took its previous grade")
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


# A referee's fields as a version line holds them, each with the default a version that names
# none of it implies, and how a violation names each.
_REFEREE = tuple((f.name, f.default) for f in dataclasses.fields(Referee))
_REFEREE_FIELD = {"who": "who it is", "verdict": "its verdict",
                  "fork_of": "the agent it was forked from", "grade": "the grade it judged",
                  "checked": "the fingerprint it read"}


def _referee(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: record.get(name, default) for name, default in _REFEREE}


class _History:
    """What a history holds: each part's grade lines with their line numbers, oldest first, in
    the order the history first names each part; each claim's origin in its first version line,
    and its author in the first that names one, with its line number; each part's referees, as
    its versions recorded them, each with the line of the first version that recorded it; and
    what does not read, one sentence each. ``missing`` is the path where there is none."""

    def __init__(self) -> None:
        self.parts: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        self.born: dict[str, str] = {}
        self.author: dict[str, tuple[int, str]] = {}
        self.referees: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        self.unread: list[str] = []
        self.missing: Path | None = None

    def version(self, n: int, id: str, claim: Mapping[str, Any]) -> None:
        """Read version line ``n``, of claim ``id``."""
        self.born.setdefault(id, str(claim.get("origin") or ""))
        if isinstance(claim.get("author"), str):
            self.author.setdefault(id, (n, claim["author"]))
        parts = claim.get("parts")
        for p in parts if isinstance(parts, list) else ():
            if not isinstance(p, dict) or not isinstance(p.get("name"), str):
                continue
            held = self.referees.setdefault((id, p["name"]), [])
            referees = p.get("referees")
            for r in referees if isinstance(referees, list) else ():
                if isinstance(r, dict) and _referee(r) not in (x for _, x in held):
                    held.append((n, _referee(r)))


def _read_history(path: Path) -> _History:
    h = _History()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        h.missing = path
        return h
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            h.unread.append(f"history line {n} does not read as JSON")
            continue
        if not isinstance(record, dict):
            h.unread.append(f"history line {n} is not one JSON object")
            continue
        if "part" not in record:  # a version line
            claim = record.get("claim")
            if isinstance(record.get("id"), str) and isinstance(claim, dict):
                h.version(n, record["id"], claim)
            continue
        lacking = [k for k in GRADE_LINE if not isinstance(record.get(k), str)]
        if lacking:
            h.unread.append(f"history line {n}: a grade line gives {', '.join(GRADE_LINE)} as "
                            f"text, and it lacks {', '.join(lacking)}")
            continue
        h.parts.setdefault((record["id"], record["part"]), []).append((n, record))
    return h


def _line_faults(n: int, record: dict[str, Any], prev: dict[str, Any] | None) -> list[str]:
    """What is wrong with grade line ``n``, given the part's line before it."""
    where = f"{record['id']}: part {record['part']!r}: history line {n}"
    out = []
    if record["grade"] not in GRADES and record["grade"] != RETIRED:
        out.append(f"{where} grades it {record['grade']!r}, not one of {GRADES}, or {RETIRED!r}")
    if not _a_day(record["date"]):
        out.append(f"{where} is dated {record['date']!r}, not YYYY-MM-DD")
    if not record["grader"].strip():
        out.append(f"{where} names no grader")
    if not record["reason"].strip():
        out.append(f"{where} gives no reason")
    if prev is None:
        if record["was"]:
            out.append(f"{where} says it moved from {record['was']}, but it is the part's first "
                       "line")
        elif record["grade"] == RETIRED:
            out.append(f"{where} retires it, but it is the part's first line: a part is retired "
                       "only once the history holds it")
        return out
    if record["was"] != prev["grade"]:
        out.append(f"{where} says it moved from {record['was'] or 'no grade'}, but the line "
                   f"before it graded it {prev['grade']}")
    elif record["grade"] == record["was"]:
        out.append(f"{where} moves it to the grade it moved from, {record['grade']}")
    if _a_day(record["date"]) and _a_day(prev["date"]) and record["date"] < prev["date"]:
        out.append(f"{where} is dated {record['date']}, before the line before it "
                   f"({prev['date']})")
    if (record["grader"], record["date"]) == (prev["grader"], prev["date"]):
        out.append(f"{where} gives the grader and date of the line before it ({prev['grader']}, "
                   f"{prev['date']}), so it names no one who made this change, or when")
    return out


_KEPT = ("a swarm claim's review stays as the history recorded it, so meta-test 7 reads the "
         "verdicts as they were given: a later verdict is a referee added beside them")


def _review_edits(c: Claim, h: _History) -> list[str]:
    """How swarm claim ``c`` edits what the history recorded of its review: its author, a part's
    grader while the part holds the grade that grader gave, or a referee, dropped or changed."""
    out = []
    first = h.author.get(c.id)
    if first is not None and c.author != first[1]:
        out.append(f"{c.id}: the history first recorded its author as {first[1]!r} (history line "
                   f"{first[0]}), but it now names {c.author!r}: a swarm claim's author stays as "
                   "recorded, so no relabeling counts its author's own verdict or audit toward "
                   "meta-test 7's gate")
    for part in c.parts:
        key = (c.id, part.name)
        lines, by = h.parts.get(key), dated(part.graded_by or c.graded_by)
        if lines and part.grade != "unsupported":
            n, line = lines[-1]
            if line["grade"] == part.grade and by != (line["grader"], line["date"]):
                named = f"its grader as {by[0]}, {by[1]}" if by else "no dated grader"
                out.append(f"{c.id}: part {part.name!r} names {named}, but its last history line "
                           f"(line {n}) records that grade by {line['grader']}, {line['date']}: a "
                           "swarm claim's part changes its grader only with its grade, through a "
                           f"line {GENERATOR} writes")
        now = [dataclasses.asdict(r) for r in part.referees]
        for n, was in h.referees.get(key, []):
            if was in now:
                continue
            # The referee it names under the same "<who>, YYYY-MM-DD", the closest where several.
            same = max((r for r in now if r["who"] == was["who"]), default=None,
                       key=lambda r: sum(r[k] == was[k] for k, _ in _REFEREE))
            if same is None:
                verdict = str(was["verdict"]).replace("_", " ")
                at = f" as {was['grade']}" if was["grade"] else ""
                fork = f", a fork of {was['fork_of']}" if was["fork_of"] else ""
                out.append(f"{c.id}: part {part.name!r}: history line {n} recorded the verdict of "
                           f"the referee {was['who']!r}, {verdict}{at}{fork}, but the part no "
                           f"longer names that referee: {_KEPT}")
            else:
                moved = "; ".join(f"{_REFEREE_FIELD[k]} as {same[k]!r}, where history line {n} "
                                  f"recorded {was[k]!r}" for k, _ in _REFEREE if same[k] != was[k])
                out.append(f"{c.id}: part {part.name!r}: the referee {was['who']!r} now gives "
                           f"{moved}: {_KEPT}")
    return out


def grades_recorded(reg: Registry, history: Path | None = None) -> list[str]:
    """Meta-test 8. ``history`` is the registry's own (``Registry.history_path``) unless given."""
    h = _read_history(Path(history) if history is not None else reg.history_path)
    if h.missing is not None:
        return [f"there is no history at {h.missing}: run {GENERATOR}, which records each part's "
                "grade with its date, grader and reason"]
    out: list[str] = list(h.unread)
    faults: list[tuple[int, str]] = []
    for lines in h.parts.values():
        prev = None
        for n, record in lines:
            faults += [(n, f) for f in _line_faults(n, record, prev)]
            prev = record
    out += [f for _, f in sorted(faults, key=lambda x: x[0])]
    first = {key: lines[0][1] for key, lines in h.parts.items()}
    last = {key: lines[-1][1] for key, lines in h.parts.items()}
    for c in reg:
        if h.born.get(c.id) == "swarm" and c.origin != "swarm":
            out.append(f"{c.id}: the history first recorded it as the swarm's, but it now names "
                       f"its origin {c.origin}: a swarm claim stays the swarm's, and is graded "
                       "through meta-test 7's gate")
        named = {p.name for p in c.parts}
        swarm = c.origin == "swarm" or h.born.get(c.id) == "swarm"
        if swarm:
            out += _review_edits(c, h)
        for (id, name), lines in h.parts.items():
            if id != c.id:
                continue
            if name not in named and lines[-1][1]["grade"] != RETIRED:
                out.append(f"{c.id}: part {name!r} is in the history, last graded "
                           f"{lines[-1][1]['grade']}, but the claim no longer names it, and no "
                           "line retires it: a part is retired through a line with its date, "
                           "grader and reason, never dropped or renamed, so name it retired, with "
                           f"its dated grader and its reason, and run {GENERATOR}")
            for n, record in lines:
                if swarm and record["grade"] == RETIRED and same_person(record["grader"],
                                                                        c.author):
                    out.append(f"{c.id}: part {name!r}: history line {n} retires it by "
                               f"{record['grader']!r}, who may be its author: a swarm claim's "
                               "part is retired only by someone other than its author")
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
