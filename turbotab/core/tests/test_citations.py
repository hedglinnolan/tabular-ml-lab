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
        # A DOI may hold a balanced parenthesis (10.1016/S0140-6736(86)90837-8); a closing one
        # without its opening ends the DOI, as when the DOI itself sits in parentheses.
        for m in re.finditer(r"(?:doi\.org/|doi:\s?)(10\.\d{4,9}/(?:[^\s,;()]|\([^\s,;()]*\))+)",
                             c.text):
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


def test_each_work_in_a_sources_entry_names_a_record():
    assert C.segments("Rubin 1987; Little & Rubin 2019 (ch. 10; ref. 4)") == [
        "Rubin 1987", "Little & Rubin 2019 (ch. 10; ref. 4)"]
    text = "Friedman 2001, Ann Stat 29:1189; Nonexistent Journal of Things, vol. 3"
    out = C.problems([C.Cited("contract:x.sources[0]", text, whole=True)])
    assert len(out) == 1 and "'Nonexistent Journal of Things, vol. 3' names no record" in out[0]
    assert C.problems([C.Cited("contract:x.sources[0]", "Friedman 2001; turbotab/grain.py:58",
                               whole=True)]) == []


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


def test_a_year_the_text_cites_is_held_to_the_year_refs_bib_prints():
    online_first = {**FRIEDMAN, "years_seen": [2000, 2001]}
    found = check.disagreements(online_first, ["Friedman 2000, Ann Stat 29:1189"])
    assert len(found) == 1 and "year 2000, the record 2001" in found[0] and "also dates it 2000" in found[0]
    reg = C.registry()
    for key, cited, printed in (("willett2013nutepi", "2013", 2012), ("bates2023cv", "2023", 2024)):
        assert reg[key].year == printed
        assert any(f"year {cited}, the record {printed}" in d for d in reg[key].disagreements), key


def test_a_page_the_doi_ends_with_is_still_held_to_the_record_s_page():
    egems = {"key": "k", "verified_by": "crossref", "doi": "10.13063/2327-9214.1244",
             "authors": ["Kahn, Michael G."], "year": 2016, "years_seen": [2016], "volume": "4",
             "pages": "18"}
    found = check.disagreements(egems, ["Kahn et al. 2016, eGEMs 4:1244"])
    assert len(found) == 1 and "first page 1244, the record 18" in found[0]
    kahn = C.registry()["kahn2016dq"]
    assert kahn.pages == "1244" and kahn.venue.startswith("eGEMs (Generating Evidence & Methods")
    assert any("first page 1244" in d and "keeps pages 1244" in d for d in kahn.disagreements)


def test_a_record_without_a_doi_is_held_to_the_year_the_text_cites():
    page = {"key": "w", "kind": "web", "verified_by": "url", "year": 2001,
            "authors": ["Maturin, Larry"], "url": "https://example.org"}
    found = check.disagreements(page, ["Bacteriological Analytical Manual (2026 edition"])
    assert len(found) == 1 and "year 2026, the record 2001" in found[0]
    # A survey's cycles and a date in the content are not the year of the work.
    assert check.disagreements({**page, "year": 2018}, ["NHANES Analytic Guidelines 2011–2016 §3.2",
                               "DEMO documentation (top-coded at 85 in 1999-2006 and at 80 from 2007"]) == []


def test_the_manual_s_record_is_the_edition_the_app_reads():
    from turbotab.core.stages.finding_words import TNTC_SOURCE

    bam = C.registry()["fda_bam_ch3"]
    assert "January 2026" in TNTC_SOURCE and "January 2026" in bam.title and bam.year == 2026
    assert bam.authors[0].startswith("Zhang") and "191248" in bam.url and not bam.disagreements


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


# ---------------------------------------------------------------- the records' own metadata

@pytest.mark.parametrize("isbn, ok", [
    ("978-92-832-1132-7", True), ("978-92-832-1132-2", False), ("92-832-1132-4", True),
    ("0199754039", True), ("0-8044-2957-X", True), ("0-8044-2957-5", False), ("97892832", False),
])
def test_an_isbn_is_held_to_its_check_digit(isbn, ok):
    assert C.isbn_valid(isbn) is ok


def test_every_isbn_passes_its_checksum_and_a_non_doi_record_s_isbn_was_looked_up():
    reg = C.registry()
    for c in reg.values():
        if c.isbn:
            assert C.isbn_valid(c.isbn), (c.key, c.isbn)
            if not c.doi:
                assert c.verified_by == "isbn", c.key
    assert reg["breslow1980iarc"].isbn == "978-92-832-1132-7"
    bad = C.Citation(key="b", kind="book", cited_as=("B 1980",), title="T", year=1980,
                     authors=("B, N",), isbn="978-92-832-1132-2", url="https://example.org",
                     no_doi="none", verified_by="url", verified_on="2026-10-09")
    found = C.record_problems(bad)
    assert any("checksum" in p for p in found) and any("looked up" in p for p in found)


def test_a_group_author_is_one_name_not_a_person():
    pubmed = {"lastName": "Of The Stratos Initiative",
              "firstName": "On Behalf Of The Measurement Error And Misclassification Topic Group Tg"}
    assert "," not in check._pubmed_person(pubmed)
    assert check._pubmed_person({"lastName": "Boe", "firstName": "Lillian A"}) == "Boe, Lillian A"
    boe = C.registry()["boe2023rc"]
    assert boe.authors[-1] == "Measurement Error and Misclassification Topic Group (TG4) of the STRATOS Initiative"
    assert "{Measurement Error and Misclassification Topic Group (TG4) of the STRATOS Initiative}" in (
        C.bibtex_entry(boe))
    garbled = C.Citation(key="g", kind="article", cited_as=("G 2023",), title="T", year=2023,
                         authors=("Of The Stratos Initiative, On Behalf Of The Group",),
                         doi="10.1000/g", verified_by="crossref", verified_on="2026-10-09")
    assert any("written as a person" in p for p in C.record_problems(garbled))


def test_a_publisher_s_running_head_and_markup_stay_out_of_the_metadata():
    meta = check.from_crossref({
        "type": "book", "author": [{"family": "Helsel", "given": "Dennis R."}],
        "title": ["Statistics for Censored Environmental Data Using Minitab® and R"],
        "subtitle": ["Helsel/Statistics for Environmental Data 2E"],
        "container-title": ["eGEMs (Generating Evidence &amp; Methods)"]})
    assert meta["title"] == "Statistics for Censored Environmental Data Using Minitab® and R"
    assert meta["venue"] == "eGEMs (Generating Evidence & Methods)"
    # A slash that is the title's own stays ("Double/debiased machine learning").
    own = check.from_crossref({"type": "journal-article", "author": [{"family": "Chernozhukov"}],
                               "title": ["Causal learning"], "subtitle": ["Double/debiased"]})
    assert own["title"] == "Causal learning: Double/debiased"
    chapter = check.from_crossref({"type": "book-chapter", "title": ["T"], "container-title": [
        "Lecture Notes in Computer Science", "Advances in Knowledge Discovery and Data Mining"]})
    assert (chapter["venue"], chapter["series"]) == (
        "Advances in Knowledge Discovery and Data Mining", "Lecture Notes in Computer Science")
    for c in C.registry().values():
        for value in (c.title, c.venue, c.publisher):
            assert not re.search(r"&\w+;|<[a-z/]", value), (c.key, value)
        for a in c.authors:  # no "Surname/Short Title" running head after a colon
            assert f": {a.split(',')[0]}/" not in c.title, c.key
        if c.kind == "chapter":
            assert c.editors and c.series, c.key


# ---------------------------------------------------------------- refs.bib

_TYPES = {"article": ("author", "title", "journal", "year"),
          "book": ("title", "publisher", "year"),
          "incollection": ("author", "title", "booktitle", "year"),
          "inproceedings": ("author", "title", "booktitle", "year"),
          "techreport": ("author", "title", "institution", "year"),
          "misc": ("title",)}
_FIELDS = {"author", "editor", "title", "journal", "booktitle", "howpublished", "series", "year",
           "volume", "number", "pages", "edition", "publisher", "institution", "organization",
           "doi", "isbn", "url", "urldate"}


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
        if kind == "incollection":  # a chapter names its book's editors
            assert fields.get("editor"), key
        if kind in ("article", "misc"):  # biblatex has no publisher on these
            assert "publisher" not in fields, key
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


@pytest.mark.skipif(shutil.which("biber") is None, reason="biber is not installed")
def test_biber_reads_refs_bib_against_biblatex_s_data_model_without_a_warning(tmp_path):
    (tmp_path / "refs.bib").write_text(C.bibtex(), "utf-8")
    run = subprocess.run(["biber", "--tool", "--validate-datamodel", "--output-file",
                          str(tmp_path / "out.bib"), "refs.bib"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:]
    assert not re.findall(r"(?m)^(?:WARN|ERROR) .*", run.stdout + run.stderr), run.stdout[-3000:]
