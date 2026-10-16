"""The claim and its parts (CLAIMS_REGISTRY.md §1, §2.1, §2.2, §2.4), and the checks a claim meets
when it registers.

A claim's vocabulary is checked at registration, as a method contract's is: its kind, its parts'
grades, its verdict, its purposes, each test's declared truth, the form of every name it gives as a
string (a test's path stays inside the repository), the grader of each part above judgment
("<who>, YYYY-MM-DD"), and that its parameters fill both its templates. What the vocabulary cannot
settle (whether a test exists, whether a citation key resolves, whether the grader is the author,
whether two claims conflict) is the meta-tests' (``_meta``).
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
from typing import Any, ClassVar, Mapping

KINDS: tuple[str, ...] = ("rule", "parameter", "teaching")
# Strongest first (§2.4); a claim shows its weakest part's grade.
GRADES: tuple[str, ...] = ("proved", "certified", "replicated", "sourced", "judgment",
                           "unsupported")
ABOVE_JUDGMENT: tuple[str, ...] = GRADES[:4]
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
class Part:
    """One thing a sentence asserts (a value, a direction, a scope), graded on its own. Its
    ``graded_by`` overrides the claim's; above judgment, the one in force reads
    "<who>, YYYY-MM-DD"."""

    name: str
    grade: str
    basis: Basis = Basis()
    graded_by: str = ""


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
    not name its own grader.
    """

    id: str
    kind: str
    parts: tuple[Part, ...]
    author: str
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
        """The weakest part's grade (§2.4), which the sentence shows."""
        return max((p.grade for p in self.parts), key=GRADES.index, default="unsupported")


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
    if not isinstance(c.version, int) or c.version < 1:
        fail("a version is a whole number from 1")
    if not c.parts:
        fail("a claim asserts at least one part, each graded (§2.4)")
    names = [p.name for p in c.parts]
    if len(set(names)) != len(names):
        fail("two parts share a name")
    for p in c.parts:
        if p.grade not in GRADES:
            fail(f"part {p.name!r}: grade {p.grade!r} is not one of {GRADES}")
        by = p.graded_by or c.graded_by
        if p.grade in ABOVE_JUDGMENT and grader(by) is None:
            fail(f"part {p.name!r} is graded {p.grade}, so its grader reads {GRADED_FORM} (§2.4), "
                 f"not {by!r}")
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
