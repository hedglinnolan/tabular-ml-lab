"""The citation registry (SIZING X4; V2 definition of done §1 and gate 7: "``refs.bib`` with every
DOI checked against Crossref").

Every source the app cites lives here once, under a stable citation key, as a verified record in
``data/citations.json``: the method contracts' ``sources`` and the works their option labels,
soundness notes and relations name; the model families' ``Source`` keys
(:mod:`turbotab.core.models.sources`); the teaching cards; and the noticings' catalog entries.

A record is one of two things:

* **a work with a DOI**, whose authors, year, title, venue, volume, issue and pages are copied from
  its Crossref record (``verified_by: "crossref"``), or from DataCite's for a DOI DataCite
  registers, as arXiv's and Bioconductor's are (``verified_by: "datacite"``);
* **a work with no DOI**, marked as such: a ``book``, ``chapter``, ``proceedings`` paper,
  ``report``, ``standard``, ``software``, ``web`` page or ``periodical`` article (a journal that
  registers no DOIs), with its ISBN or URL and the reason it has no DOI (``no_doi``). Nothing here
  invents a DOI.

Each record says how the app's own text cites it (``cited_as``: regular expressions, each matched
at the start of a citation), so a citation in prose resolves to exactly one record, and a source
string in a ``sources`` list resolves to at least one. Where the app's text disagrees with the
Crossref record (a year, a volume, a page), the record lists it under ``disagreements``; the claim
is not rewritten here (INBOX carries each one).

``python -m turbotab.core.export.citations_check`` re-reads every DOI from Crossref (or DataCite)
and rewrites the metadata; the tests read only the committed file, so they never touch the network.
:func:`bibtex` writes ``refs.bib`` from the verified records alone.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DATA_FILE = Path(__file__).resolve().parent / "data" / "citations.json"
FORMAT = 1
_REPO = Path(__file__).resolve().parents[3]
# The noticings' sources live in their catalogs until the noticings carry them in code.
CATALOGS = _REPO / "docs" / "turbotab-next" / "understanding" / "catalogs"
NOTICINGS_FILE = Path(__file__).resolve().parents[1] / "quest_noticings.json"

KINDS: tuple[str, ...] = ("article", "chapter", "book", "proceedings", "preprint", "report",
                          "standard", "software", "web", "periodical")
# The kinds a record may be without a DOI; each such record carries an ISBN or a URL. A
# ``periodical`` is an article in a journal that registers no DOIs (Survey Methodology).
NON_DOI_KINDS: tuple[str, ...] = ("book", "chapter", "proceedings", "report", "standard",
                                  "software", "web", "periodical")
# How a record was verified: its DOI's metadata from Crossref or DataCite; or, for a DOI Crossref's
# API does not serve, doi.org's handle with PubMed's metadata (``doi_handle``); or a non-DOI
# record's ISBN (Open Library) or URL.
DOI_VERIFIERS: tuple[str, ...] = ("crossref", "datacite", "doi_handle")
VERIFIERS: tuple[str, ...] = (*DOI_VERIFIERS, "isbn", "url")

# Pointers into the project's own documents and code: not bibliography, so not in refs.bib.
INTERNAL = re.compile(
    r"^(?:MODELING_SEQUENCE|BLUEPRINT|V2 definition of done|NUTRITION_PACK|CLINICAL_SURVEY_PACK"
    r"|GENOMICS_PACK|METABOLOMICS_PACK|research/|turbotab/|tests/|ml/|docs/|Critic's "
    r"|find_fixtures|_tt_tmp)")

# A citation in prose: authors (a surname with optional initials, "et al.", "&" or commas), an
# optional venue of up to six words, then a year. It finds where to look; ``cited_as`` decides.
_SURNAME = (r"(?:(?:van der|van den|van|de|Di|Van|Le) )?[A-Z][a-zà-ÿ'’\-]*(?:[A-Z][a-zà-ÿ]+)*"
            r"(?: [A-Z]{1,3})?")
_AUTHORS = rf"{_SURNAME}(?:(?:,| &| and) {_SURNAME})*(?: et al\.?)?"
_YEAR = r"(?:1[89]\d\d|20[0-2]\d)(?!\d)"
# Not after "Hastie, T., " (a later author of a list that cites by initials).
MENTION = re.compile(rf"(?<!\., )\b{_AUTHORS},?(?: [A-Za-z][\w.+:\-]*(?: [A-Za-z][\w.+:\-]*){{0,5}})?,?"
                     rf" \(?{_YEAR}")
# Spans the detector finds that cite nothing (a placeholder date, a survey cycle).
NOT_CITATIONS = re.compile(r"^(?:Placeholder dates|Family K1|Combining NHANES cycles"
                           r"|Wiley 2009)")


@dataclass(frozen=True)
class Citation:
    """One verified source (module docstring)."""

    key: str
    kind: str
    cited_as: tuple[str, ...]
    title: str = ""
    year: int | None = None  # None for an undated web page
    authors: tuple[str, ...] = ()  # "Family, Given"; an organization as its name
    venue: str = ""  # journal, book series, proceedings, or publisher's site
    volume: str = ""
    issue: str = ""
    pages: str = ""  # "1189-1232" or an article number
    publisher: str = ""
    edition: str = ""
    doi: str | None = None
    isbn: str = ""
    url: str = ""
    no_doi: str = ""  # why a non-DOI record has none
    verified_by: str = ""
    verified_on: str = ""
    verification_note: str = ""
    disagreements: tuple[str, ...] = ()
    years_seen: tuple[int, ...] = ()  # every year Crossref gives it (print, online, issued)
    crossref_type: str = ""
    note: str = ""

    @property
    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return _compiled(self.key, self.cited_as)

    def reference(self) -> str:
        """The record as one plain reference line."""
        who = "; ".join(self.authors) if self.authors else self.publisher
        bits = [f"{who} ({self.year})." if self.year else f"{who}.", f"{self.title}."]
        where = self.venue + (f" {self.volume}" if self.volume else "") + (
            f"({self.issue})" if self.issue else "") + (f":{self.pages}" if self.pages else "")
        if where.strip():
            bits.append(where.strip() + ".")
        bits.append(f"https://doi.org/{self.doi}" if self.doi else (self.url or f"ISBN {self.isbn}"))
        return " ".join(bits)


@lru_cache(maxsize=None)
def _compiled(key: str, patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


def load(path: Path = DATA_FILE) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


@lru_cache(maxsize=1)
def registry() -> dict[str, Citation]:
    """Every verified record, by citation key."""
    data = load()
    if data.get("format") != FORMAT:
        raise ValueError(f"{DATA_FILE.name} is format {data.get('format')}, not {FORMAT}")
    names = {f.name for f in dataclasses.fields(Citation)}
    out: dict[str, Citation] = {}
    for r in data["records"]:
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r.items() if k in names}
        c = Citation(**kw)
        if c.key in out:
            raise ValueError(f"two records share the key {c.key!r}")
        out[c.key] = c
    return out


# ---------------------------------------------------------------- what the app cites

@dataclass(frozen=True)
class Cited:
    """A piece of the app's text that cites: ``whole`` when it is an entry of a sources list
    (it must name at least one record), else prose (each citation in it must name one)."""

    origin: str
    text: str
    whole: bool = False
    # The sources list of the contract the text belongs to: a citation the registry cannot tell
    # apart ("Collins et al. 2024") names the one record that list also names.
    context: tuple[str, ...] = ()


def _strings(o: Any, path: str, seen: set[int] | None = None) -> Iterator[tuple[str, str]]:
    seen = set() if seen is None else seen
    if isinstance(o, str):
        yield path, o
        return
    if id(o) in seen:
        return
    seen.add(id(o))
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        for f in dataclasses.fields(o):
            yield from _strings(getattr(o, f.name), f"{path}.{f.name}", seen)
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(v, f"{path}.{k}", seen)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _strings(v, f"{path}[{i}]", seen)


def _contract_texts() -> Iterator[Cited]:
    from turbotab.core.contracts import contracts

    for key, c in sorted(contracts().items()):
        for i, s in enumerate(c.sources):
            yield Cited(f"contract:{key}.sources[{i}]", s, whole=True)
        for path, s in _strings(dataclasses.replace(c, sources=()), f"contract:{key}"):
            yield Cited(path, s, context=tuple(c.sources))


def family_source_keys() -> dict[str, set[str]]:
    """Each source key a model family's declarations cite, with the families citing it."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.models.base import Source, families

    out: dict[str, set[str]] = {}

    def visit(o: Any, family: str, seen: set[int]) -> None:
        if isinstance(o, Source):
            out.setdefault(o.key, set()).add(family)
        elif isinstance(o, (str, bytes, int, float, type(None))) or id(o) in seen:
            return
        else:
            seen.add(id(o))
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                for f in dataclasses.fields(o):
                    visit(getattr(o, f.name), family, seen)
            elif isinstance(o, dict):
                for v in o.values():
                    visit(v, family, seen)
            elif isinstance(o, (list, tuple, set, frozenset)):
                for v in o:
                    visit(v, family, seen)

    for fam in families():
        seen: set[int] = set()
        for name in dir(fam):
            if not name.startswith("_"):
                try:
                    value = getattr(fam, name)
                except Exception:  # noqa: BLE001 - a property that needs a fit
                    continue
                if not callable(value):
                    visit(value, fam.key, seen)
    return out


def _family_texts() -> Iterator[Cited]:
    from turbotab.core.models.sources import SOURCES

    for key, text in sorted(SOURCES.items()):
        yield Cited(f"family_source:{key}", text, whole=True)


def _teaching_texts() -> Iterator[Cited]:
    import turbotab.core.teaching.content as content

    for name in sorted(vars(content)):
        if name.isupper():
            for path, s in _strings(getattr(content, name), f"teaching:{name}"):
                yield Cited(path, s)


NOTICING_PROSE = ("title", "meaning", "sentence", "magic", "failure_prevented")


def _noticing_texts() -> Iterator[Cited]:
    ids = set(json.loads(NOTICINGS_FILE.read_text("utf-8"))["data"])
    if not CATALOGS.is_dir():
        return
    for path in sorted(CATALOGS.glob("*.json")):
        data = json.loads(path.read_text("utf-8"))
        for t in data.get("threads", []) if isinstance(data, dict) else []:
            if t.get("id") not in ids:
                continue
            for i, s in enumerate(t.get("sources") or []):
                yield Cited(f"noticing:{t['id']}.sources[{i}]", s, whole=True)
            for f in NOTICING_PROSE:
                if isinstance(t.get(f), str):
                    yield Cited(f"noticing:{t['id']}.{f}", t[f])


def cited_texts() -> list[Cited]:
    """Every piece of the app's text that may cite a source (module docstring)."""
    return [*_contract_texts(), *_family_texts(), *_teaching_texts(), *_noticing_texts()]


# ---------------------------------------------------------------- resolving a citation

def is_internal(text: str) -> bool:
    """Whether ``text`` points into the project's own documents or code."""
    return bool(INTERNAL.match(text.strip()))


def mentions(text: str) -> list[re.Match[str]]:
    """The citations in prose ``text`` (author and year), placeholders excepted."""
    return [m for m in MENTION.finditer(text) if not NOT_CITATIONS.match(m.group(0))]


def resolve_at(text: str, pos: int, reg: dict[str, Citation] | None = None) -> list[str]:
    """The records whose ``cited_as`` matches ``text`` at ``pos``."""
    reg = registry() if reg is None else reg
    return [k for k, c in reg.items() if any(p.match(text, pos) for p in c.patterns)]


_STARTS = re.compile(r"(?:; |\()")


def starts(text: str) -> list[int]:
    """Where a citation may start in a source string: its start, after "; " or "(", and at each
    author-and-year citation in it."""
    found = {0, *(m.end() for m in _STARTS.finditer(text)), *(m.start() for m in mentions(text))}
    return sorted(found)


def resolve(text: str, reg: dict[str, Citation] | None = None) -> list[str]:
    """Every record a source string cites (each matched where a citation starts), in the order
    the records are listed."""
    reg = registry() if reg is None else reg
    at = starts(text)
    return [k for k, c in reg.items() if any(p.match(text, i) for p in c.patterns for i in at)]


def cited_keys_at(c: Cited, pos: int, reg: dict[str, Citation] | None = None) -> list[str]:
    """The records the citation at ``pos`` in ``c`` names, its contract's sources breaking a tie."""
    keys = resolve_at(c.text, pos, reg)
    if len(keys) > 1 and c.context:
        named = {k for s in c.context for k in resolve(s, reg)}
        keys = [k for k in keys if k in named] or keys
    return keys


def problems(cited: Iterable[Cited] | None = None,
             reg: dict[str, Citation] | None = None) -> list[str]:
    """What in the app's citing text does not resolve, one line each; empty when all of it does."""
    reg = registry() if reg is None else reg
    out: list[str] = []
    for c in cited_texts() if cited is None else cited:
        if c.whole and not is_internal(c.text) and not resolve(c.text, reg):
            out.append(f"{c.origin}: {c.text!r} names no record")
        for m in mentions(c.text):
            keys = cited_keys_at(c, m.start(), reg)
            if len(keys) != 1:
                what = "no record" if not keys else f"{len(keys)} records {keys}"
                out.append(f"{c.origin}: {m.group(0)!r} names {what}")
    for key, fams in sorted(family_source_keys().items()):
        if key not in reg:
            out.append(f"model families {sorted(fams)} cite {key!r}, which has no record")
    return out


def keys_cited(cited: Iterable[Cited] | None = None,
               reg: dict[str, Citation] | None = None) -> list[str]:
    """Every record the app cites, in registry order."""
    reg = registry() if reg is None else reg
    hit: set[str] = set(family_source_keys())
    for c in cited_texts() if cited is None else cited:
        if c.whole:
            hit.update(resolve(c.text, reg))
        for m in mentions(c.text):
            hit.update(cited_keys_at(c, m.start(), reg))
    return [k for k in reg if k in hit]


def record_problems(c: Citation) -> list[str]:
    """What a record lacks to count as verified (module docstring)."""
    out = []
    if c.kind not in KINDS:
        out.append(f"{c.key}: kind {c.kind!r} is not one of {KINDS}")
    if not c.doi:
        if c.kind not in NON_DOI_KINDS:
            out.append(f"{c.key}: a {c.kind} has no DOI and is not marked as a non-DOI kind")
        if not (c.isbn or c.url):
            out.append(f"{c.key}: a record without a DOI carries an ISBN or a URL")
        if not c.no_doi:
            out.append(f"{c.key}: a record without a DOI says why it has none")
    elif not re.fullmatch(r"10\.\d{4,9}/\S+", c.doi):
        out.append(f"{c.key}: {c.doi!r} is not a DOI")
    if c.verified_by not in VERIFIERS or not c.verified_on:
        out.append(f"{c.key}: not verified ({c.verified_by!r})")
    if c.doi and c.verified_by not in DOI_VERIFIERS:
        out.append(f"{c.key}: a DOI is verified against Crossref, DataCite or doi.org")
    if not c.doi and c.verified_by in DOI_VERIFIERS:
        out.append(f"{c.key}: verified by {c.verified_by} without a DOI")
    if not c.title or not (c.authors or c.publisher):
        out.append(f"{c.key}: a record names its title and its authors or publisher")
    if not c.cited_as:
        out.append(f"{c.key}: a record says how the app cites it")
    return out


# ---------------------------------------------------------------- refs.bib

_BIB_TYPE = {"article": "article", "chapter": "incollection", "book": "book",
             "proceedings": "inproceedings", "preprint": "misc", "report": "techreport",
             "standard": "misc", "software": "misc", "web": "misc", "periodical": "article"}
# BibTeX counts braces even after a backslash, so a brace in the text is dropped, not escaped.
_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": "", "}": "", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s: str) -> str:
    return "".join(_ESCAPE.get(ch, ch) for ch in s)


def _protect(title: str) -> str:
    """A title with every capitalized word after the first braced (and the first, when it is an
    acronym), so a style that lowercases titles keeps "NHANES", "Bayes" and "R" as written."""
    out = []
    for i, word in enumerate(_tex(title).split(" ")):
        core = word.strip("(),.:;?!\"'‘’“”")
        capital = any(ch.isupper() for ch in core[1:]) or (i > 0 and core[:1].isupper())
        out.append("{" + word + "}" if capital else word)
    return " ".join(out)


def _name(author: str) -> str:
    return _tex(author) if "," in author else "{" + _tex(author) + "}"


def bibtex_entry(c: Citation) -> str:
    fields: list[tuple[str, str]] = []
    if c.authors:
        fields.append(("author", " and ".join(_name(a) for a in c.authors)))
    fields.append(("title", _protect(c.title)))
    container = {"article": "journal", "periodical": "journal", "chapter": "booktitle",
                 "proceedings": "booktitle"}
    if c.venue:
        fields.append((container.get(c.kind, "howpublished" if c.kind in ("web", "software",
                       "standard", "preprint") else "series"), _tex(c.venue)))
    if c.year:
        fields.append(("year", str(c.year)))
    for name, value in (("volume", c.volume), ("number", c.issue),
                        ("pages", c.pages.replace("-", "--") if c.pages else ""),
                        ("edition", c.edition)):
        if value:
            fields.append((name, _tex(value)))
    if c.publisher:
        fields.append(("institution" if c.kind == "report" else "publisher", _tex(c.publisher)))
    if c.doi:
        fields.append(("doi", c.doi))
    if c.isbn:
        fields.append(("isbn", c.isbn))
    if c.url:
        fields.append(("url", c.url))
    if c.url and c.kind in ("web", "software", "standard") and c.verified_on:
        fields.append(("urldate", c.verified_on))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{_BIB_TYPE[c.kind]}{{{c.key},\n{body}\n}}\n"


def bibtex(keys: Sequence[str] | None = None, reg: dict[str, Citation] | None = None) -> str:
    """``refs.bib`` for ``keys`` (every record when omitted), from verified records only."""
    reg = registry() if reg is None else reg
    chosen = list(reg) if keys is None else list(dict.fromkeys(keys))
    entries = []
    for k in chosen:
        c = reg[k]
        if record_problems(c):
            raise ValueError(f"{k} is not a verified record: {record_problems(c)}")
        entries.append(bibtex_entry(c))
    return "\n".join(entries)


def write_bib(path: Path, keys: Sequence[str] | None = None) -> Path:
    path.write_text(bibtex(keys), "utf-8")
    return path


__all__ = ["Citation", "Cited", "DATA_FILE", "KINDS", "NON_DOI_KINDS", "bibtex", "bibtex_entry",
           "cited_texts", "family_source_keys", "is_internal", "keys_cited", "mentions",
           "problems", "record_problems", "registry", "resolve", "resolve_at", "write_bib"]
