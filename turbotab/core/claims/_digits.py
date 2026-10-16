"""The digit gate's parts (CLAIMS_REGISTRY.md §3): where a string's citations are, and the
allowlist of today's digit-bearing strings.

A digit is any character ``str.isdigit`` accepts, so "²" counts. It passes when it sits inside a
citation: a span one of the citation registry's ``cited_as`` patterns matches, less the bounded gap
a pattern may allow between a work's author and its year (:data:`_GAP`), and carried on, when the
span sits in a parenthesis, over its bibliographic tail and nothing else (:data:`_TAIL`: the rest
of a token the match ends inside, as in "STROBE 16a", a span of years, a journal's volume and page,
a section, a page or an item). So "(Friedman 2001, Ann Stat 29:1189, §10.10)" is one citation,
while "(below 0.6, Friedman 2001)", "(Friedman 2001, below 0.6)" and "Van Calster et al. recommend
at least 10 events per variable (2024)" are each a number beside one. The registry is read as a
file: the claims package imports no engine module.

Otherwise the whole string must be allowlisted in ``digit_allowlist.json``, tagged with one of
:data:`TAGS`. The allowlist was seeded from the app's digit-bearing strings on 2026-10-16; a
string not in it fails the gate, and an entry the app no longer has is stale.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ALLOWLIST = HERE / "digit_allowlist.json"
CITATIONS = HERE.parent / "export" / "data" / "citations.json"
TAGS: tuple[str, ...] = ("name", "guideline item", "example", "mathematical constant", "citation",
                         "threshold-pending-claim")


@lru_cache(maxsize=4)
def _records(path: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple((r["key"], tuple(r.get("cited_as", ()))) for r in data["records"])


def citation_keys(path: Path = CITATIONS) -> frozenset[str]:
    """Every key the citation registry holds."""
    return frozenset(k for k, _ in _records(str(path)))


# A bounded gap a ``cited_as`` pattern allows between a work's author and its year: a negated
# class repeated from zero, such as ``[^;()]{0,90}?``. A digit it matches is not this citation's,
# though another citation's match may hold it.
_GAP = re.compile(r"(?<!\\)\[\^[^\]]*\]\{0,\d+\}\??")
_GAP_GROUP = "_gap"


def _grouped(pattern: str) -> str:
    """``pattern`` with each bounded gap in a named group, so a match reports where its gaps are."""
    n = count()
    return _GAP.sub(lambda m: f"(?P<{_GAP_GROUP}{next(n)}>{m.group(0)})", pattern)


@lru_cache(maxsize=4)
def _patterns(path: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(_grouped(p)) for _, ps in _records(path) for p in ps)


def _opening(text: str, start: int) -> int | None:
    """Where the parenthesis enclosing ``start`` opens, if one does."""
    depth = 0
    for i in range(start - 1, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            if depth == 0:
                return i
            depth -= 1
    return None


# One piece of a citation's bibliographic tail, read from where the work's match ends: a span of
# years ("2011–2016"), then, each after a comma or a space, a journal's volume and page
# (", Ann Stat 29:1189"), a section ("§3.2.3"), a page ("p. 1203") or an item ("item 15").
_TAIL = re.compile(
    r"[–-]\d{4}\b"
    r"|(?:,\s*|\s+)(?:"
    r"[A-Z][A-Za-z.&]*(?:\s[A-Z][A-Za-z.&]*)*\s\d+(?:\(\d+\))?:\d+(?:[–-]\d+)?"
    r"|§\s?\d+(?:\.\d+)*"
    r"|pp?\.\s?\d+(?:[–-]\d+)?"
    r"|items?\s\d+[a-z]?(?:(?:,\s*|\s+and\s+)\d+[a-z]?)*"
    r")\b")
_TOKEN = re.compile(r"\w+")


def _tail(text: str, end: int) -> int:
    """Where a citation whose work's match ends at ``end`` ends: past its bibliographic tail."""
    if 0 < end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end = _TOKEN.match(text, end).end()
    while m := _TAIL.match(text, end):
        end = m.end()
    return end


def citation_spans(text: str, citations: Path = CITATIONS) -> list[tuple[int, int]]:
    """The spans of ``text`` that cite a work the citation registry holds: each match but its
    gaps, then its tail."""
    spans = []
    for pattern in _patterns(str(citations)):
        gaps = [g for g in pattern.groupindex if g.startswith(_GAP_GROUP)]
        for m in pattern.finditer(text):
            start = m.start()
            end = _tail(text, m.end()) if _opening(text, start) is not None else m.end()
            for a, b in sorted(m.span(g) for g in gaps if m.start(g) >= 0):
                spans.append((start, a))
                start = b
            spans.append((start, end))
    return spans


def stray_digits(text: str, spans: Iterable[tuple[int, int]] = ()) -> str:
    """The digits of ``text`` outside every span, in order."""
    spans = list(spans)
    return "".join(ch for i, ch in enumerate(text)
                   if ch.isdigit() and not any(a <= i < b for a, b in spans))


def load_allowlist(path: Path = ALLOWLIST) -> dict[str, str]:
    """``{text: tag}``; an entry with no tag, an unknown tag, no digit, or a repeat is refused."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in raw["strings"]:
        text, tag = entry.get("text"), entry.get("tag")
        if not isinstance(text, str) or tag not in TAGS:
            raise ValueError(f"{path.name}: {text!r} needs a tag from {TAGS}, not {tag!r}")
        if not any(ch.isdigit() for ch in text):
            raise ValueError(f"{path.name}: {text!r} holds no digit, so it needs no entry")
        if text in out:
            raise ValueError(f"{path.name}: {text!r} is listed twice")
        out[text] = tag
    return out


def stale_allowlist(texts: Iterable[str], allowlist: dict[str, str] | None = None) -> list[str]:
    """The allowlisted strings not among ``texts`` (every string the gate reads), sorted."""
    allowlist = load_allowlist() if allowlist is None else allowlist
    present = set(texts)
    return sorted(t for t in allowlist if t not in present)
