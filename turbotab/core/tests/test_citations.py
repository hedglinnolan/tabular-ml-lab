"""SIZING X4: the citation registry and ``refs.bib`` (V2 definition of done §1 and gate 7).

Every source the app cites resolves to a verified record; no record lacks a DOI unless it is marked
as a kind that has none; ``refs.bib`` parses, under a strict reader here and under BibTeX itself
where it is installed; and where the app's text disagrees with Crossref, the registry lists it and
INBOX carries it. The tests read only the committed registry; none touches the network.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from turbotab.core.export import citations as C
from turbotab.core.export import citations_check as check

REFS_BIB = C.DATA_FILE.parent / "refs.bib"
INBOX = Path(__file__).resolve().parents[3] / "docs" / "turbotab-next" / "INBOX.md"


@pytest.fixture(scope="module")
def cited() -> list[C.Cited]:
    return C.cited_texts()


def test_the_registry_reads_the_contracts_the_families_the_teaching_and_the_noticings(cited):
    origins = {c.origin.split(":")[0] for c in cited}
    assert origins == {"contract", "family_source", "teaching", "noticing"}
    assert any(c.whole for c in cited if c.origin.startswith("noticing:"))


def test_every_citation_in_the_app_names_one_record(cited):
    assert C.problems(cited) == []


def test_every_cited_key_resolves_to_a_verified_record(cited):
    reg = C.registry()
    keys = C.keys_cited(cited)
    assert set(C.family_source_keys()) <= set(keys)
    assert {k: C.record_problems(reg[k]) for k in keys if C.record_problems(reg[k])} == {}
    # And nothing sits in the registry that the app no longer cites.
    assert set(reg) - set(keys) == set()


def test_no_record_lacks_a_doi_unless_it_is_marked_as_a_kind_without_one():
    for c in C.registry().values():
        if c.doi:
            assert re.fullmatch(r"10\.\d{4,9}/\S+", c.doi), c.key
            assert c.verified_by in C.DOI_VERIFIERS, c.key
        else:
            assert c.kind in C.NON_DOI_KINDS, c.key
            assert c.isbn or c.url, c.key
            assert c.no_doi, c.key
            assert c.verified_by in ("isbn", "url"), c.key
    kinds = {c.kind for c in C.registry().values() if not c.doi}
    assert kinds <= set(C.NON_DOI_KINDS)


def test_a_doi_the_app_spells_out_is_its_record_s_doi(cited):
    reg = C.registry()
    spelled = 0
    for c in cited:
        for m in re.finditer(r"(?:doi\.org/|doi:\s?)(10\.\d{4,9}/[^\s,;)]+)", c.text):
            keys = C.resolve(c.text) if c.whole else C.cited_keys_at(c, 0)
            assert m.group(1).rstrip(".").lower() in {(reg[k].doi or "").lower() for k in keys}
            spelled += 1
    assert spelled >= 1  # the gradient-boosting family's source carries its DOI
    for name, key in (("tripod_ai.json", "collins2024tripodai"),
                      ("strobe_nut.json", "lachat2016strobenut")):
        source = json.loads((C.DATA_FILE.parent / name).read_text("utf-8"))["source"]
        assert source["doi"].lower() == reg[key].doi.lower()


def test_a_citation_that_names_no_record_is_reported():
    out = C.problems([C.Cited("teaching:X", "as shown before (Nobody et al. 2019).")])
    assert out and "Nobody et al. 2019" in out[0] and "no record" in out[0]
    out = C.problems([C.Cited("contract:x.sources[0]", "Nobody 2019, J Imag 1:1", whole=True)])
    assert any("names no record" in line for line in out)


def test_a_citation_two_records_share_is_settled_only_by_its_contract_s_sources():
    text = "at least 500 resamples (Collins et al. 2024)"
    alone = C.problems([C.Cited("contract:x.options", text)])
    assert alone and "2 records" in alone[0]
    settled = C.Cited("contract:x.options", text,
                      context=("Collins et al., BMJ 2024;384:e074819",))
    assert C.problems([settled]) == []
    assert C.cited_keys_at(settled, text.index("Collins")) == ["collins2024evaluation"]


def test_internal_pointers_are_not_bibliography():
    assert C.is_internal("MODELING_SEQUENCE §2")
    assert C.is_internal("turbotab/core/decisions.py:1630 SetIntendedUse")
    assert not C.is_internal("Rubin 1987")


# ---------------------------------------------------------------- disagreements with Crossref

FRIEDMAN = {"key": "f", "verified_by": "crossref", "doi": "10.1214/aos/1013203451",
            "authors": ["Friedman, Jerome H."], "year": 2001, "years_seen": [2001],
            "volume": "29", "pages": "1189-1232"}


@pytest.mark.parametrize("span, says", [
    ("Friedman 2001, Ann Stat 29:1189–1232", None),
    ("Friedman 2002, Ann Stat 29:1189–1232", "year 2002"),
    ("Friedman 2001, Ann Stat 28:1189", "volume 28"),
    ("Friedman 2001, Ann Stat 29:1190", "first page 1190"),
    ("Friedman 2001, Ann Stat 29:1189–1200", "last page 1200"),
    ("Freedman 2001, Ann Stat 29:1189", "author Freedman"),
])
def test_a_citation_that_differs_from_its_record_is_listed(span, says):
    found = check.disagreements(FRIEDMAN, [span])
    if says is None:
        assert found == []
    else:
        assert len(found) == 1 and says in found[0]


def test_every_disagreement_is_listed_in_the_registry_and_in_inbox():
    reg = C.registry()
    disagreeing = {k for k, c in reg.items() if c.disagreements}
    data = {r["key"]: r for r in C.load()["records"]}
    for k in disagreeing:
        assert k in INBOX.read_text("utf-8"), f"{k} disagrees with Crossref but INBOX does not say so"
    for k, r in data.items():
        if r.get("overrides"):  # a corrected field says why, and keeps Crossref's value beside it
            assert r["overrides"]["why"] and set(r["crossref_original"]) == set(
                r["overrides"]["fields"]), k
    # Holding the text to the committed metadata finds what the registry lists, and no more.
    spans = check.citing_spans(C.load())
    for k, r in data.items():
        original = r.get("crossref_original") or {}
        found = check.disagreements({**r, **original}, spans[k])
        assert len(found) == len(r.get("disagreements") or []), k


# ---------------------------------------------------------------- refs.bib

_TYPES = {"article": ("author", "title", "journal", "year"),
          "book": ("title", "publisher", "year"),
          "incollection": ("author", "title", "booktitle", "year"),
          "inproceedings": ("author", "title", "booktitle", "year"),
          "techreport": ("author", "title", "institution", "year"),
          "misc": ("title",)}
_FIELDS = {"author", "editor", "title", "journal", "booktitle", "howpublished", "series", "year",
           "volume", "number", "pages", "edition", "publisher", "institution", "doi", "isbn",
           "url", "urldate"}


def parse_bib(text: str) -> dict[str, tuple[str, dict[str, str]]]:
    """A strict BibTeX reader: ``@type{key, field = {value}, ...}`` with balanced braces."""
    entries: dict[str, tuple[str, dict[str, str]]] = {}
    i, n = 0, len(text)
    while True:
        i = text.find("@", i)
        if i < 0:
            return entries
        m = re.compile(r"@(\w+)\{([^,\s]+),\s*").match(text, i)
        assert m, f"malformed entry at {text[i:i + 40]!r}"
        kind, key, i = m.group(1), m.group(2), m.end()
        assert key not in entries, f"duplicate key {key}"
        fields: dict[str, str] = {}
        while True:
            while i < n and text[i] in " \n\t,":
                i += 1
            if text[i] == "}":
                i += 1
                break
            f = re.compile(r"(\w+)\s*=\s*\{").match(text, i)
            assert f, f"{key}: malformed field at {text[i:i + 40]!r}"
            name, i, depth, start = f.group(1).lower(), f.end(), 1, f.end()
            while depth:
                assert i < n, f"{key}.{name}: unbalanced braces"
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            assert name not in fields, f"{key}: {name} twice"
            fields[name] = text[start:i - 1]
        entries[key] = (kind, fields)


def test_refs_bib_is_written_from_verified_records_only():
    reg = C.registry()
    entries = parse_bib(C.bibtex())
    assert set(entries) == set(reg)
    for key, (kind, fields) in entries.items():
        assert kind in _TYPES, key
        assert set(fields) <= _FIELDS, (key, set(fields) - _FIELDS)
        for name in _TYPES[kind]:
            assert fields.get(name), f"{key} ({kind}) lacks {name}"
        if kind == "book":
            assert fields.get("author") or fields.get("editor"), key
        c = reg[key]
        assert fields.get("doi") == (c.doi or None) or (not c.doi and "doi" not in fields), key
        if not c.doi:
            assert fields.get("isbn") or fields.get("url"), key
        for value in fields.values():
            assert not re.search(r"(?<!\\)[&%#]", value), (key, value)
    unverified = C.Citation(key="x", kind="article", cited_as=("X 2000",), title="T", year=2000,
                            authors=("X, Y",), doi="10.1000/x")
    with pytest.raises(ValueError, match="not a verified record"):
        C.bibtex(["x"], reg={"x": unverified})


def test_the_committed_refs_bib_is_the_registry_s():
    assert REFS_BIB.read_text("utf-8") == C.bibtex()


@pytest.mark.skipif(shutil.which("bibtex") is None, reason="BibTeX is not installed")
def test_bibtex_itself_reads_refs_bib_without_a_warning(tmp_path):
    (tmp_path / "refs.bib").write_text(C.bibtex(), "utf-8")
    (tmp_path / "t.aux").write_text("\\citation{*}\n\\bibstyle{plain}\n\\bibdata{refs}\n")
    run = subprocess.run(["bibtex", "t"], cwd=tmp_path, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stdout
    log = (tmp_path / "t.blg").read_text("utf-8", "replace")
    assert "Warning--" not in log and "error message" not in log, log[-2000:]
    assert (tmp_path / "t.bbl").read_text("utf-8", "replace").count("\\bibitem") == len(C.registry())
