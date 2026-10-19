"""The claim and its parts (CLAIMS_REGISTRY.md §1, §2.1, §2.2, §2.4, §8), and the checks a claim
meets when it registers.

A claim's vocabulary is checked at registration, as a method contract's is: its kind, its origin,
its parts' grades, its verdict, its purposes, each test's declared truth, the form of every name it
gives as a string (a test's path stays inside the repository), the grader of each part above
judgment, its basis audits (each an :class:`Audit`, in the order they were made) and each of its
referees ("<who>, YYYY-MM-DD"), the
grade each referee judged and the fingerprint each referee and audit read, each referee's verdict,
that a claim keeps one part it asserts, and that its parameters fill both its templates. What the
vocabulary cannot settle (whether a test exists, whether a citation key resolves, whether the
grader is the author, whether two referees are one person, whether a verdict was on the part as it
stands, whether two claims conflict) is the meta-tests' (``_meta``).

:func:`checked` is the fingerprint of a part as it stands: what a referee or a basis audit read.
:func:`referee_gate` is the two-referee gate (§8) a swarm claim's part passes above judgment; given
the part's grade, fingerprint and the day it took its previous grade, it counts only the verdicts
on the part as it stands. It reads only the referees and what it is given, so a research tool
outside the app (the decoy queue in ``research/registry/``) applies the same gate to its own queue.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import posixpath
import re
import string
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Iterable, Mapping

KINDS: tuple[str, ...] = ("rule", "parameter", "teaching")
# Strongest first (§2.4); a claim shows its weakest part's grade.
GRADES: tuple[str, ...] = ("proved", "certified", "replicated", "sourced", "judgment",
                           "unsupported")
ABOVE_JUDGMENT: tuple[str, ...] = GRADES[:4]
# A part its sentence no longer asserts. It is no grade: it moves there through a history line, as
# a grade does (meta-test 8), so a part is retired, never dropped or renamed, and the claim shows
# the weakest part it still asserts.
RETIRED = "retired"
# Hex digits of a part's fingerprint (:func:`checked`), as a referee or a basis audit names it.
CHECKED_DIGITS = 12
# Copies of ``contracts.PURPOSES`` and ``contracts.RUNGS``: the registry imports no engine module,
# and ``test_claims_schema.py`` holds the copies to the originals.
PURPOSES: tuple[str, ...] = ("prediction", "inference")
RUNGS: tuple[str, ...] = ("recommended", "available", "rank_lower", "block_and_record", "refused",
                          "not_offered")
# A verdict is a leash rung, or "noted" for a concern (§1's V).
VERDICTS: tuple[str, ...] = (*RUNGS, "noted")
# Where a test's truth comes from (§2.1): "construction", "independent:<library>" or
# "derivation:<notes path>".
TRUTHS: tuple[str, ...] = ("construction", "independent", "derivation")
# Where a claim came from (R1_PLAN.md R1-G): written by the engine's builders from its own code,
# registered for a finding an audit made (the claims backtest's), or proposed by the research swarm
# (CLAIMS_REGISTRY.md §8), whose claim is held proposed until its parts are graded.
ORIGINS: tuple[str, ...] = ("engine", "audit", "swarm")
# A referee upheld the part at its grade, or did not (the backtest's "overstated", "false" and
# "not established" alike).
REFEREE_VERDICTS: tuple[str, ...] = ("upheld", "not_upheld")

_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_DOTTED = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
# "module:function" (a predicate or an enforcer), "module:Class.method", or "module:NAME".
_TARGET = re.compile(rf"^{_DOTTED}:{_DOTTED}$")
# "repo/relative/path.py::test_name", "path.py::Class::test_name", either with a "[param]" suffix.
_TEST = re.compile(r"^(?P<path>[\w./-]+\.py)::(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)"
                   r"(?:\[.*\])?$")
_RETRACTED = re.compile(r"^\d{4}-\d{2}-\d{2}: \S")
# Above judgment, a grader is "<who>, YYYY-MM-DD" (§1's "basis audit, 2026-10-15").
GRADED_FORM = "'<who>, YYYY-MM-DD'"
_GRADED = re.compile(r"(?P<who>.*\S), (?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})")
_CHECKED = re.compile(rf"[0-9a-f]{{{CHECKED_DIGITS}}}")
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_NAME_PUNCTUATION = re.compile(r"[-_.'\u2019]")
_SELF_REFERENCES = frozenset({"self", "myself", "me", "author", "same", "same author", "same person"})


@dataclass(frozen=True)
class Param:
    """A number with its unit (§1's parameter). ``constant`` names the engine constant that will
    read it, as ``"module:NAME"``; meta-test 2 holds the claim's tests apart from it."""

    value: int | float
    unit: str
    constant: str | None = None


@dataclass(frozen=True)
class Regime:
    """The regime the evidence covers (§2.2): in words, and as predicates.

    ``requires`` names ``"module:function"`` predicates that take the context and return a bool,
    resolved only when a context is checked. ``disjoint_from`` names claims whose regimes do not
    overlap this one (meta-test 5): two claims on one decision overlap unless either names the
    other.
    """

    words: str
    requires: tuple[str, ...] = ()
    disjoint_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeldOut:
    """Held-out evidence for a replicated part (§2.4): cases from one fixture or dataset not used
    to fit the rule, and the rate the rule held on them."""

    fixture: str
    cases: int
    rate: float


@dataclass(frozen=True)
class Basis:
    """What a part's grade rests on (§2.4's table). Which fields a grade needs is meta-test 3's."""

    cites: tuple[str, ...] = ()  # citation keys (turbotab/core/export/data/citations.json)
    where: str = ""  # the passage, quoted by the grader, that says this in this regime
    reason: str = ""  # a judgment's reason
    theorem: str = ""  # a Lean theorem built in CI, or a published theorem
    derivation: str = ""  # a repo-relative notes path holding a hand derivation
    tests: tuple[str, ...] = ()  # the claim's own Fires or Silent tests that certify the part
    heldout: tuple[HeldOut, ...] = ()


@dataclass(frozen=True)
class Referee:
    """One referee's verdict on a part (CLAIMS_REGISTRY.md §8): ``who`` reads "<who>, YYYY-MM-DD",
    ``verdict`` is one of :data:`REFEREE_VERDICTS`, and ``fork_of`` names the agent whose context
    the referee's was forked from, empty for one started fresh. A fork never counts toward the
    gate: it inherits its parent's context (ORCHESTRATOR.md §3).

    ``grade`` is the grade the referee judged the part at, and ``checked`` the fingerprint
    (:func:`checked`) of the part the referee read. A registered part's referee names both; a
    research queue's (the decoy queue's), whose items each have one state, names neither. The gate
    counts a registered part's referee only while both are the part's as it stands."""

    who: str
    verdict: str
    fork_of: str = ""
    grade: str = ""
    checked: str = ""


@dataclass(frozen=True)
class Audit:
    """A part's basis audit (CLAIMS_REGISTRY.md §8): ``who`` reads "<who>, YYYY-MM-DD", who checked
    the basis against its source, and when; ``checked`` is the fingerprint (:func:`checked`) of
    the part they checked, which registration requires. An audit stands only while the part is as
    they checked it (meta-tests 3 and 7). A part's audits are append-only, as its referees are: a
    later audit is added after those recorded (``Part.audited_by``), and the last is the one in
    force (:func:`audit_in_force`)."""

    who: str
    checked: str = ""


@dataclass(frozen=True)
class Part:
    """One thing a sentence asserts (a value, a direction, a scope), graded on its own. Its
    ``graded_by`` overrides the claim's; above judgment, the one in force reads
    "<who>, YYYY-MM-DD".

    ``grade`` is one of :data:`GRADES`, or :data:`RETIRED` for a part its sentence no longer
    asserts.

    ``audited_by`` is the basis audit, an :class:`Audit` ("" names none), or the audits made,
    oldest first, as a tuple of them, each dated no earlier than the one before: append-only, as
    the referees are, so a re-audit is added after the audit it follows, never written over it,
    and the last is the audit in force (:func:`audit_in_force`), which the meta-tests read. The
    audit in force is by someone other than the author, on the part as it stands (meta-test 3). A
    swarm claim's part above judgment names one, and the ``referees`` who upheld it (meta-test 7);
    once the history records a swarm claim's audit, it stays (meta-test 8). An engine or audit
    claim's part may name one; where it does not, its grader
    stands as its basis audit, since each grade's requirement (§2.4) has the grader quote the
    passage, re-derive the derivation or read the test against its reference, and meta-test 3
    holds that grader apart from the author. ``regraded`` is why the part moved to the grade it
    holds: the generator writes it, with the grader and date, as the part's history line
    (meta-test 8)."""

    name: str
    grade: str
    basis: Basis = Basis()
    graded_by: str = ""
    referees: tuple[Referee, ...] = ()
    audited_by: Audit | tuple[Audit, ...] | str = ""
    regraded: str = ""


def audits(part: Part) -> tuple[Audit, ...]:
    """The part's basis audits, oldest first: none for "", one for an :class:`Audit`, and each
    :class:`Audit` of a tuple (registration refuses anything else in one)."""
    audit = part.audited_by
    if isinstance(audit, Audit):
        return (audit,)
    if isinstance(audit, tuple):
        return tuple(a for a in audit if isinstance(a, Audit))
    return ()


def audit_in_force(part: Part) -> Audit | None:
    """The part's last basis audit, the one in force, or None where it names none."""
    made = audits(part)
    return made[-1] if made else None


@dataclass(frozen=True)
class _Test:
    """A known-truth test (§2.1): ``test`` is ``"path.py::test_name"``, ``truth`` where its
    expected value comes from."""

    test: str
    truth: str
    side: ClassVar[str] = ""


@dataclass(frozen=True)
class Fires(_Test):
    """The rule fires where it should."""

    side: ClassVar[str] = "fires"


@dataclass(frozen=True)
class Silent(_Test):
    """The rule stays quiet where it should not fire."""

    side: ClassVar[str] = "silent"


@dataclass(frozen=True, kw_only=True)
class Claim:
    """A statement the app relies on to set, rank, refuse or explain a decision (§1).

    ``card`` and ``methods`` are its sentence in two registers, each a template over ``params``.
    ``fallback`` names the weaker claim that renders outside the regime (§2.2). ``ruled`` points
    to where a ruling was made (§6). ``retracted`` is "YYYY-MM-DD: reason", the only status anyone
    declares; every other status is derived (``Registry.status``). ``graded_by`` is
    "<who>, YYYY-MM-DD" (§1's "basis audit, 2026-10-15") for every part above judgment that does
    not name its own grader. ``origin`` is one of :data:`ORIGINS`, and has no default, so no claim
    lacks one.
    """

    id: str
    kind: str
    parts: tuple[Part, ...]
    author: str
    origin: str
    card: str = ""
    methods: str = ""
    goal: tuple[str, ...] = ()  # G: the purposes it serves
    regime: Regime = Regime("")  # R
    decision: str = ""  # D: a contract key or decision kind
    options: tuple[str, ...] = ()  # O
    verdict: str = ""  # V
    params: Mapping[str, Param] = field(default_factory=dict)
    tests: tuple[_Test, ...] = ()
    enforced_by: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    fallback: str = ""
    graded_by: str = ""
    version: int = 1
    ruled: str = ""
    retracted: str = ""

    @property
    def grade(self) -> str:
        """The weakest grade of the parts it asserts (§2.4), which the sentence shows; a retired
        part is not asserted."""
        return max((p.grade for p in self.parts if p.grade != RETIRED), key=GRADES.index,
                   default="unsupported")


def grader(graded_by: str) -> str | None:
    """Who graded, the part of ``graded_by`` before its date, or None when it is not
    ``'<who>, YYYY-MM-DD'`` with a real date."""
    m = _GRADED.fullmatch(graded_by)
    if m is None:
        return None
    try:
        date.fromisoformat(m["date"])
    except ValueError:
        return None
    return m["who"]


def dated(by: str) -> tuple[str, str] | None:
    """``(who, date)`` from ``'<who>, YYYY-MM-DD'`` with a real date, or None."""
    who = grader(by)
    return None if who is None else (who, _GRADED.fullmatch(by)["date"])


def same_person(who_graded: str, author: str) -> bool:
    """Whether ``who_graded`` may be ``author``. Each is casefolded, its parentheticals dropped, its
    hyphens, underscores, periods and apostrophes read as spaces, its whitespace collapsed and a
    leading "the" dropped. Then they may be one person when the two are equal, either holds the
    other (with or without its spaces), the grader is a self-reference such as "self" or "the
    author", or either names no one. In doubt it says yes: a false yes costs a re-grade, a false
    no lets a claim grade itself."""
    def who(name: str) -> str:
        while _PARENTHETICAL.search(name):
            name = _PARENTHETICAL.sub(" ", name)
        name = " ".join(_NAME_PUNCTUATION.sub(" ", name).casefold().split())
        return "" if name == "the" else name.removeprefix("the ")
    g, a = who(who_graded), who(author)
    if not g or not a or g in _SELF_REFERENCES:
        return True
    g_run, a_run = g.replace(" ", ""), a.replace(" ", "")
    return g in a or a in g or g_run in a_run or a_run in g_run


def referee_gate(referees: Iterable[Referee], author: str | None, grade: str | None = None,
                 fingerprint: str | None = None, since: str | None = None) -> list[str]:
    """What the two-referee gate (CLAIMS_REGISTRY.md §8) still lacks for a part ``author`` wrote,
    one sentence each; ``[]`` once it passes.

    A referee counts unless it may be the author (``same_person``, which refuses in doubt) or is a
    fork, of the author or of anyone (ORCHESTRATOR.md §3). Being a fork, or the author, is the
    agent's, not one verdict's: a name any of the part's records marks a fork, or names as
    possibly its author, counts in none of them, each record's name compared with the marked ones
    by ``same_person``. For a registered part, ``grade`` is the
    grade it holds, ``fingerprint`` the part as it stands (:func:`checked`) and ``since`` the day
    it took its previous grade (its history's line before this grade). A referee then counts only
    where it judged that grade and read that fingerprint: a verdict on another grade, or on an
    older basis or sentence, upheld or not, was not on the part as it stands. An uphold also
    counts only where it is dated no earlier than ``since``, so one given before the part last
    moved is not carried past that move; a dissent on the part as it stands counts whenever it was
    given, since nothing has changed that could answer it. The gate passes when at least two
    referees who count, two distinct people, upheld the part, and none who counts did not. A
    referee is named by the "<who>" of its "<who>, YYYY-MM-DD", or by the whole string where it
    gives no date (a research queue's referee, for which none of ``grade``, ``fingerprint`` and
    ``since`` is given). ``author`` None is an author not recorded, so no referee is checked
    against one."""
    upheld: list[str] = []
    lacks: list[str] = []
    uncounted: list[str] = []
    named = [(r, grader(r.who) or r.who.strip()) for r in referees]
    marked = [name for r, name in named
              if r.fork_of.strip() or (author is not None and same_person(name, author))]
    for r, name in named:
        if author is not None and same_person(name, author):
            uncounted.append(f"{name!r} may be its author")
            continue
        if r.fork_of.strip():
            whose = ("its author" if author is not None and same_person(r.fork_of, author)
                     else repr(r.fork_of.strip()))
            uncounted.append(f"{name!r} is a fork of {whose}")
            continue
        if any(same_person(name, m) for m in marked):
            uncounted.append(f"{name!r} is marked a fork, or possibly its author, by another "
                             "record of the part")
            continue
        if grade is not None and r.grade != grade:
            uncounted.append(f"{name!r} judged it as {r.grade or 'no grade'}, not {grade}")
            continue
        if fingerprint is not None and r.checked != fingerprint:
            uncounted.append(f"{name!r} read another version of it, since changed")
            continue
        if r.verdict != "upheld":
            lacks.append(f"{name!r} did not uphold it")
            continue
        when = dated(r.who)
        if since is not None and (when is None or when[1] < since):
            uncounted.append(f"{name!r} is dated {when[1] if when else 'no day'}, before {since}, "
                             "when it took its previous grade")
            continue
        if not any(same_person(name, u) for u in upheld):
            upheld.append(name)
    if len(upheld) < 2:
        why = f" (not counted: {'; '.join(uncounted)})" if uncounted else ""
        at = f" as {grade}, on the part as it stands" if grade is not None else ""
        lacks.append(f"it needs two distinct referees who upheld it{at}, neither its author nor a "
                     f"fork (ORCHESTRATOR.md §3), and has {len(upheld)}{why}")
    return lacks


def placeholders(template: str) -> list[str]:
    """The placeholder names in ``template``; a malformed template raises ``ValueError``."""
    return [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]


def literal(template: str) -> str:
    """``template`` outside its placeholders (format specs included), braces unescaped."""
    return "".join(text for text, _, _, _ in string.Formatter().parse(template))


def fill(template: str, params: Mapping[str, Param]) -> str:
    return template.format(**{k: p.value for k, p in params.items()})


def _truth_ok(truth: str) -> bool:
    kind, sep, what = truth.partition(":")
    if kind == "construction":
        return not sep
    return kind in TRUTHS and bool(sep) and bool(what.strip())


def check(c: Claim) -> None:
    """Refuse a claim outside the registry's vocabulary, naming the field (``ValueError``)."""
    def fail(what: str) -> None:
        raise ValueError(f"{c.id}: {what}")

    if not _ID.match(c.id):
        fail("an id is dotted lowercase, area first ('trials.few_clusters')")
    if c.kind not in KINDS:
        fail(f"kind {c.kind!r} is not one of {KINDS}")
    if not c.author.strip():
        fail("a claim names its author")
    if c.origin not in ORIGINS:
        fail(f"origin {c.origin!r} is not one of {ORIGINS}")
    if not isinstance(c.version, int) or c.version < 1:
        fail("a version is a whole number from 1")
    if not any(p.grade != RETIRED for p in c.parts):
        fail("a claim asserts at least one part, each graded (§2.4); one that asserts none is "
             "retracted, not retired")
    names = [p.name for p in c.parts]
    if len(set(names)) != len(names):
        fail("two parts share a name")
    fingerprint = f"{CHECKED_DIGITS} lowercase hex digits (claims.checked)"
    for p in c.parts:
        if p.grade not in GRADES and p.grade != RETIRED:
            fail(f"part {p.name!r}: grade {p.grade!r} is not one of {GRADES}, or {RETIRED!r}")
        by = p.graded_by or c.graded_by
        if p.grade in ABOVE_JUDGMENT and grader(by) is None:
            fail(f"part {p.name!r} is graded {p.grade}, so its grader reads {GRADED_FORM} (§2.4), "
                 f"not {by!r}")
        audit = p.audited_by
        made = audit if isinstance(audit, tuple) else (audit,) if isinstance(audit, Audit) else ()
        if isinstance(audit, tuple) and not audit:
            fail(f"part {p.name!r}: its basis audits are one Audit or more; a part that names "
                 "none names ''")
        if audit != "" and not made:
            fail(f"part {p.name!r}: a basis audit is an Audit, naming who checked the basis and "
                 f"the part they checked, not {audit!r}")
        day = ""
        for a in made:
            if not isinstance(a, Audit):
                fail(f"part {p.name!r}: a basis audit is an Audit, naming who checked the basis "
                     f"and the part they checked, not {a!r}")
            when = dated(a.who)
            if when is None:
                fail(f"part {p.name!r}: its basis audit reads {GRADED_FORM}, not {a.who!r}")
            if not _CHECKED.fullmatch(a.checked):
                fail(f"part {p.name!r}: its basis audit names the part it checked as its "
                     f"fingerprint, {fingerprint}, not {a.checked!r}")
            if when[1] < day:
                fail(f"part {p.name!r}: its basis audits are appended in the order they were "
                     f"made, each dated no earlier than the one before, so the last is in force: "
                     f"{a.who!r} follows one dated {day}")
            day = when[1]
        for r in p.referees:
            if not isinstance(r, Referee):
                fail(f"part {p.name!r}: a referee is a Referee, not {r!r}")
            if grader(r.who) is None:
                fail(f"part {p.name!r}: a referee reads {GRADED_FORM}, not {r.who!r}")
            if r.verdict not in REFEREE_VERDICTS:
                fail(f"part {p.name!r}: referee {r.who!r} gives the verdict {r.verdict!r}, not one "
                     f"of {REFEREE_VERDICTS}")
            if r.grade not in GRADES:
                fail(f"part {p.name!r}: referee {r.who!r} names the grade it judged, one of "
                     f"{GRADES}, not {r.grade!r}")
            if not _CHECKED.fullmatch(r.checked):
                fail(f"part {p.name!r}: referee {r.who!r} names the part it read as its "
                     f"fingerprint, {fingerprint}, not {r.checked!r}")
        for h in p.basis.heldout:
            if not isinstance(h.cases, int) or h.cases < 1 or not 0 <= h.rate <= 1:
                fail(f"part {p.name!r}: held-out evidence has cases and a rate in [0, 1]")
    if c.verdict and c.verdict not in VERDICTS:
        fail(f"verdict {c.verdict!r} is not one of {VERDICTS}")
    if any(g not in PURPOSES for g in c.goal):
        fail(f"a goal is one of {PURPOSES}")
    for name, p in c.params.items():
        if isinstance(p.value, bool) or not isinstance(p.value, (int, float)):
            fail(f"parameter {name!r} is a number")
        if not math.isfinite(p.value):
            fail(f"parameter {name!r} is finite")
        if not p.unit.strip():
            fail(f"parameter {name!r} has a unit")
        if p.constant is not None and not _TARGET.match(p.constant):
            fail(f"parameter {name!r}: its constant is named 'module:NAME'")
    for register in ("card", "methods"):
        template = getattr(c, register)
        try:
            fields = placeholders(template)
        except ValueError as e:
            fail(f"the {register} template does not parse ({e})")
        for f in fields:
            if f not in c.params:
                fail(f"the {register} template's {{{f}}} names no parameter")
        try:
            fill(template, c.params)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            fail(f"the {register} template cannot be filled from its parameters ({e})")
    for t in c.tests:
        if not isinstance(t, (Fires, Silent)):
            fail("a test is Fires or Silent")
        named = _TEST.match(t.test)
        if not named:
            fail(f"test {t.test!r} is named 'path.py::test_name'")
        path = named["path"]
        if path.startswith("/") or posixpath.normpath(path).split("/")[0] == "..":
            fail(f"test {t.test!r} is absolute or leaves the repository root; name it from the "
                 "root")
        if not _truth_ok(t.truth):
            fail(f"test {t.test!r}: truth {t.truth!r} is 'construction', 'independent:<library>' "
                 "or 'derivation:<path>'")
    for target in (*c.enforced_by, *c.regime.requires):
        if not _TARGET.match(target):
            fail(f"{target!r} is named 'module:function'")
    if c.kind == "rule":
        # §2.1: every rule is tested both ways, whatever its grade, and names the code it acts in.
        if not any(isinstance(t, Fires) for t in c.tests):
            fail("a rule names a Fires test (§2.1)")
        if not any(isinstance(t, Silent) for t in c.tests):
            fail("a rule names a Silent test (§2.1)")
        if not c.enforced_by:
            fail("a rule names the code that enforces it")
        if not c.verdict:
            fail("a rule has a verdict")
    for other in (c.fallback, *c.supersedes, *c.regime.disjoint_from):
        if other and not _ID.match(other):
            fail(f"{other!r} is not a claim id")
        if other == c.id:
            fail("a claim names itself as its fallback, a superseded claim or a disjoint regime")
    if c.retracted and not _RETRACTED.match(c.retracted):
        fail("a retraction is 'YYYY-MM-DD: reason'")


def _plain(o: Any) -> Any:
    if isinstance(o, _Test):
        return {"side": o.side, "test": o.test, "truth": o.truth}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _plain(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, Mapping):
        return {str(k): _plain(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def content(c: Claim) -> dict[str, Any]:
    """The claim as plain JSON data: what its hash covers, and what the history keeps (§5)."""
    return _plain(c)


def content_hash(c: Claim) -> str:
    text = json.dumps(content(c), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# What a referee or a basis audit reads of a claim, beside the part's own name, grade and basis.
_READ = ("id", "kind", "card", "methods", "params", "goal", "regime", "decision", "options",
         "verdict")


def checked(c: Claim, part: Part) -> str:
    """The fingerprint of ``part`` as it stands in ``c``: the first :data:`CHECKED_DIGITS` hex
    digits of the SHA-256 of what a referee or a basis audit reads, the claim's id, kind,
    sentences, parameters, goal, regime, decision, options and verdict, and the part's name, grade
    and basis. Its graders, referees, basis audit and reason for regrading are left out, as are
    the claim's other parts, so recording a review moves no fingerprint, and any change to what
    was reviewed does."""
    whole = content(c)
    read = {"claim": {k: whole[k] for k in _READ},
            "part": {"name": part.name, "grade": part.grade, "basis": _plain(part.basis)}}
    text = json.dumps(read, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CHECKED_DIGITS]
