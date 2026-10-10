"""C6a SRC: the citation records the ridge, Huber, random-forest and XGBoost families quote.

The short forms below are written as RECIPES_AND_TUNING and MODEL_FAMILY_CONTRACT cite them; each
must name exactly one record, and each record's SOURCES entry must name that record and no other.
The bibliographic facts (DOI, volume, pages) are copied here from the publishers' own pages
(Crossref, JMLR, the CRAN issue PDF), not read back from the registry.
"""
from __future__ import annotations

import pytest

from turbotab.core.export import citations as C
from turbotab.core.models.sources import SOURCES

SHORT = {
    "Huber 1964": "huber1964",
    "Holland & Welsch 1977": "holland1977irls",
    "Breiman 2001, Random Forests": "breiman2001forests",
    "Liaw & Wiener 2002": "liaw2002randomforest",
    "Wright & Ziegler 2017": "wright2017ranger",
    "Probst, Wright & Boulesteix 2019": "probst2019rf",
    "Probst et al. 2019": "probst2019tunability",
    "Probst, Boulesteix & Bischl 2019": "probst2019tunability",
    "Chen & Guestrin 2016": "chen2016xgboost",
    "Bergstra & Bengio 2012": "bergstra2012random",
    "Bischl et al. 2023": "bischl2023hpo",
    "Cawley & Talbot 2010": "cawley2010overfitting",
    "Riley et al. 2021": "riley2021penalization",
    "Van Calster et al. 2020": "vancalster2020shrinkage",
    "Martin et al. 2021": "martin2021tuning",
    "Kruppa et al. 2014": "kruppa2014theory",
    "Josse et al. 2024": "josse2024missing",
    "Perez-Lebel et al. 2022": "perezlebel2022missing",
    "Van Ness et al. 2023": "vanness2023indicator",
    "Gelman 2008": "gelman2008twosd",
    "Ng 2004": "ng2004l1l2",
    "Mentch & Zhou 2020": "mentch2020randomization",
    "McElfresh et al. 2023": "mcelfresh2023tabular",
    "Kobak et al. 2020": "kobak2020ridge",
    "Curth et al. 2024": "curth2024forests",
}

# key: (DOI or None, volume, first page), from the publishers' pages.
FACTS = {
    "huber1964": ("10.1214/aoms/1177703732", "35", "73"),
    "holland1977irls": ("10.1080/03610927708827533", "6", "813"),
    "breiman2001forests": ("10.1023/A:1010933404324", "45", "5"),
    "wright2017ranger": ("10.18637/jss.v077.i01", "77", ""),
    "probst2019rf": ("10.1002/widm.1301", "9", "e1301"),
    "chen2016xgboost": ("10.1145/2939672.2939785", "", "785"),
    "bischl2023hpo": ("10.1002/widm.1484", "13", "e1484"),
    "riley2021penalization": ("10.1016/j.jclinepi.2020.12.005", "132", "88"),
    "vancalster2020shrinkage": ("10.1177/0962280220921415", "29", "3166"),
    "martin2021tuning": ("10.1177/09622802211046388", "30", "2545"),
    "kruppa2014theory": ("10.1002/bimj.201300068", "56", "534"),
    "josse2024missing": ("10.1007/s00362-024-01550-4", "65", "5447"),
    "perezlebel2022missing": ("10.1093/gigascience/giac013", "11", "giac013"),
    "vanness2023indicator": ("10.1145/3580305.3599911", "", "5004"),
    "gelman2008twosd": ("10.1002/sim.3107", "27", "2865"),
    "ng2004l1l2": ("10.1145/1015330.1015435", "", "78"),
    "mcelfresh2023tabular": ("10.52202/075280-3337", "", "76336"),
    "curth2024forests": ("10.48550/arXiv.2402.01502", "", ""),
    "liaw2002randomforest": (None, "2", "18"),
    "probst2019tunability": (None, "20", "1"),
    "bergstra2012random": (None, "13", "281"),
    "cawley2010overfitting": (None, "11", "2079"),
    "kobak2020ridge": (None, "21", "1"),
    "mentch2020randomization": (None, "21", "1"),
}


def test_every_new_key_is_a_source_and_a_record():
    reg = C.registry()
    assert len(FACTS) == 24 and set(SHORT.values()) <= set(FACTS)
    for key in FACTS:
        assert key in SOURCES, key
        assert key in reg, key
        assert C.record_problems(reg[key]) == [], key


@pytest.mark.parametrize("key", sorted(FACTS))
def test_a_sources_entry_names_its_own_record_and_no_other(key):
    assert C.resolve(SOURCES[key]) == [key]


@pytest.mark.parametrize("short,key", sorted(SHORT.items()))
def test_a_short_form_names_one_record(short, key):
    assert C.resolve(short) == [key]


def test_bare_breiman_2001_is_not_claimed_by_the_forest_paper():
    # "Breiman 2001" alone is the Rashomon citation (Statistical Modeling: The Two Cultures).
    assert C.resolve("Breiman 2001") == []


@pytest.mark.parametrize("key,facts", sorted(FACTS.items()))
def test_each_record_carries_the_published_facts(key, facts):
    doi, volume, first = facts
    rec = C.registry()[key]
    assert (rec.doi or None) == doi
    assert rec.volume == volume
    assert (rec.pages.split("-")[0] if rec.pages else "") == first
    if doi is None:
        assert rec.kind == "periodical" and rec.url.startswith("https://") and rec.verified_by == "url"
    else:
        assert rec.verified_by in ("crossref", "datacite")


# The issue numbers of the JMLR records, from each paper's JMLR page (its citation_issue meta tag).
JMLR_ISSUES = {
    "probst2019tunability": "53",
    "bergstra2012random": "10",
    "cawley2010overfitting": "70",
    "kobak2020ridge": "169",
    "mentch2020randomization": "171",
}


@pytest.mark.parametrize("key,issue", sorted(JMLR_ISSUES.items()))
def test_a_jmlr_record_carries_its_issue_number(key, issue):
    rec = C.registry()[key]
    assert rec.issue == issue
    assert f"{rec.volume}({issue}):{rec.pages.split('-')[0]}" in SOURCES[key]


def test_bergstra_and_cawley_read_volume_issue_pages_in_the_bib_file():
    from pathlib import Path
    bib = (Path(C.__file__).parent / "data" / "refs.bib").read_text(encoding="utf-8")
    for key, (volume, number, pages) in {"bergstra2012random": ("13", "10", "281--305"),
                                         "cawley2010overfitting": ("11", "70", "2079--2107")}.items():
        entry = bib[bib.index("{" + key + ","):]
        entry = entry[:entry.index("\n}")]
        assert f"volume = {{{volume}}}" in entry and f"number = {{{number}}}" in entry, key
        assert f"pages = {{{pages}}}" in entry, key


def _listed_lines() -> list[str]:
    """The '- ' lines of MODEL_FAMILY_CONTRACT §7 and of RECIPES_AND_TUNING's Sources."""
    from pathlib import Path
    docs = Path(__file__).resolve().parents[3] / "docs" / "turbotab-next"
    contract = (docs / "MODEL_FAMILY_CONTRACT.md").read_text(encoding="utf-8")
    recipes = (docs / "RECIPES_AND_TUNING.md").read_text(encoding="utf-8")
    sections = (contract[contract.index("## 7 · Sources"):contract.index("## What changed after review")],
                recipes[recipes.index("### Sources"):recipes.index("## Rulings folded in")])
    return [line for s in sections for line in s.splitlines() if line.startswith("- ")]


@pytest.mark.parametrize("key", sorted(SOURCES))
def test_every_source_is_listed_in_section_7_or_the_recipes_sources(key):
    """sources.py's documented rule: a source joins only once the contract's §7 or RECIPES' Sources
    lists it. The listing line names the first author and the year, and either the title's first
    long word or the volume and first page; the spec's own line is the reference."""
    import re
    rec = C.registry()[key]
    family = rec.authors[0].split(",")[0]
    word = next(w for w in re.findall(r"[A-Za-z]+", rec.title) if len(w) >= 5).lower()
    first = rec.pages.split("-")[0] if rec.pages else ""
    hits = [line for line in _listed_lines() if family in line and str(rec.year) in line
            and (word in line.lower() or (rec.volume and first and rec.volume in line and first in line))]
    assert hits, key
