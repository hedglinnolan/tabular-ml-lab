"""The digit gate's parts (CLAIMS_REGISTRY.md §3): where a string's citations are, and the
allowlist of today's digit-bearing strings.

A digit is any character ``str.isdigit`` accepts, so "²" counts. It passes when it sits inside a
citation: a span one of the citation registry's ``cited_as`` patterns matches, carried on to the
close of the parenthesis it sits in (or the next "; " inside it), so "(Friedman 2001, Ann Stat
29:1189)" is one citation and "(below 0.6, Friedman 2001)" is a number beside one.
The registry is read as a file: the claims package imports no engine module.

Otherwise the whole string must be allowlisted in ``digit_allowlist.json``, tagged with one of
:data:`TAGS`. The allowlist was seeded from the app's digit-bearing strings on 2026-10-16; a
string not in it fails the gate, and an entry the app no longer has is stale.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
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


@lru_cache(maxsize=4)
def _patterns(path: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for _, ps in _records(path) for p in ps)


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


def _close(text: str, end: int) -> int:
    """The end of a citation inside a parenthesis: its close, or the next "; " before it."""
    depth = 0
    for j in range(end, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            if depth == 0:
                return j + 1
            depth -= 1
        elif text[j] == ";" and depth == 0:
            return j
    return len(text)


def citation_spans(text: str, citations: Path = CITATIONS) -> list[tuple[int, int]]:
    """The spans of ``text`` that cite a work the citation registry holds."""
    spans = []
    for pattern in _patterns(str(citations)):
        for m in pattern.finditer(text):
            end = _close(text, m.end()) if _opening(text, m.start()) is not None else m.end()
            spans.append((m.start(), end))
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
