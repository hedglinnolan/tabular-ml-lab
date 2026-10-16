"""The digit gate over the app's own sentences (CLAIMS_REGISTRY.md §3; R0_PLAN.md R0a acceptance 4).

The gate reads every claim template outside its placeholders, every ``Relation.says`` in the
method contracts and every teaching string (the uppercase names of
``turbotab.core.teaching.content``, as the citation scanner reads them), one character at a time
with ``str.isdigit``, so "²" counts. A digit passes when it is a placeholder, sits inside a
citation the citation registry recognizes (the work, then only its bibliographic tail: years,
volume and page, section, page or item; where the registry finds the work by its author, a bounded
gap and its year, the author and the year, not the gap), or the whole string is allowlisted with
its tag in
``turbotab/core/claims/digit_allowlist.json``. Today's digit-bearing strings seed the allowlist,
each tagged; a new one fails, and an entry the app no longer has fails too.

This test imports the engine to read its strings; the claims package never does.
"""
from __future__ import annotations

import json

import pytest

from turbotab.core import claims
from turbotab.core.tests.test_claims_fixtures import registry, rule

# The six tags the R0 plan names, written out here rather than read from the code.
SIX_TAGS = ("name", "guideline item", "example", "mathematical constant", "citation",
            "threshold-pending-claim")


def relation_says() -> list[tuple[str, str]]:
    from turbotab.core.contracts import contracts

    return [(f"relation:{key}.{r.name}", r.says)
            for key, c in sorted(contracts().items()) for r in c.relations]


def teaching_strings() -> list[tuple[str, str]]:
    from turbotab.core.export.citations import cited_texts

    return [(c.origin, c.text) for c in cited_texts() if c.origin.startswith("teaching:")]


def test_the_gate_reads_the_strings_it_says_it_reads():
    """Floors, so the gate cannot pass by reading nothing."""
    says, teaching = relation_says(), teaching_strings()
    assert len(says) >= 300 and len(teaching) >= 700
    digits = [t for _, t in says + teaching if any(ch.isdigit() for ch in t)]
    assert len(digits) >= 150


def test_every_relation_and_teaching_string_passes_the_gate():
    found = claims.digit_gate(claims.registry(), strings=relation_says() + teaching_strings())
    assert found == [], f"{len(found)} strings fail the digit gate:\n" + "\n".join(found[:20])


def test_the_allowlist_holds_no_string_the_app_no_longer_has():
    texts = {t for _, t in relation_says() + teaching_strings()}
    texts |= {t for c in claims.registry() for t in (c.card, c.methods) if t}
    stale = claims.stale_allowlist(texts)
    assert stale == [], "allowlisted strings no longer in the app:\n" + "\n".join(stale[:20])


def test_every_allowlist_entry_carries_one_of_the_six_tags():
    assert claims.TAGS == SIX_TAGS
    raw = json.loads(claims.ALLOWLIST.read_text(encoding="utf-8"))
    entries = raw["strings"]
    assert entries, "the allowlist is seeded from today's strings"
    assert all(e["tag"] in SIX_TAGS for e in entries)
    assert [e["text"] for e in entries] == sorted({e["text"] for e in entries})
    assert claims.load_allowlist() == {e["text"]: e["tag"] for e in entries}


@pytest.mark.parametrize("text", [
    "fewer than 12 clusters are named among the concerns",
    "Scored by R² on held-out rows",
    "Refused below 0.6 (Friedman 2001)",
])
def test_a_new_digit_bearing_string_fails_the_gate(text):
    found = claims.digit_gate(claims.Registry(), strings=[("planted:new", text)], allowlist={})
    assert len(found) == 1 and "planted:new" in found[0]


@pytest.mark.parametrize("text, stray", [
    ("It is named (Friedman 2001, below 0.6).", "06"),
    ("It is named (Li & Redden 2015: fewer than 10 clusters).", "10"),
    ("(Friedman 2001 recommends a shrinkage of 0.1 and 500 trees)", "01500"),
    ("It is named (Friedman 2001, Ann Stat 29:1189, below 0.6).", "06"),
    ("It is named (Friedman 2001, §10.10, 500 trees).", "500"),
])
def test_a_number_after_a_citation_inside_its_parenthesis_is_not_part_of_it(text, stray):
    """A citation runs on only over its bibliographic tail (years, volume and page, section, page,
    item); the threshold after it is the sentence's own number."""
    found = claims.digit_gate(claims.Registry(), strings=[("planted:new", text)], allowlist={})
    assert len(found) == 1 and f"the digits {stray!r} " in found[0]


@pytest.mark.parametrize("text, stray", [
    ("Van Calster et al. recommend at least 10 events per variable (2024).", "10"),
    ("Bustin et al. require at least 3 replicates per sample (2009).", "3"),
    ("Sumner et al. ask for 2 orthogonal properties to name a metabolite (2007).", "2"),
    ("Lachat and colleagues ask that intakes under 500 kcal be reported (2016).", "500"),
])
def test_a_number_between_a_citation_s_author_and_its_year_is_not_part_of_it(text, stray):
    """A citation the registry recognizes by its author, a bounded gap and its year cites the
    author and the year; a threshold written in the gap is the sentence's own number."""
    found = claims.digit_gate(claims.Registry(), strings=[("planted:new", text)], allowlist={})
    assert len(found) == 1 and f"the digits {stray!r} " in found[0]


@pytest.mark.parametrize("text", [
    "Boosting fits stagewise (Friedman 2001).",
    "Van Calster et al. recommend it for small samples (2024).",
    "Van Calster et al., STRATOS TG6, 2024, list the measures.",
    "Boosting fits stagewise (Friedman 2001, Ann Stat 29:1189); a later work disagrees.",
    "Boosting fits stagewise (Friedman 2001, Ann Stat 29:1189, §10.10, p. 1203).",
    "The arms are described, never tested (Moher et al. 2010, item 15).",
    "The unadjusted estimate is shown (STROBE 16a).",
    "The subgroup's degrees of freedom (NHANES Analytic Guidelines 2011–2016 §3.2.3).",
    "A sentence with no number in it.",
])
def test_a_digit_inside_a_citation_or_no_digit_passes(text):
    assert claims.digit_gate(claims.Registry(), strings=[("planted:new", text)],
                             allowlist={}) == []


def test_a_tagged_string_passes_and_the_same_string_untagged_fails():
    text = "The plan is exported with its SHA-256."
    strings = [("planted:new", text)]
    assert claims.digit_gate(claims.Registry(), strings=strings, allowlist={text: "name"}) == []
    assert len(claims.digit_gate(claims.Registry(), strings=strings, allowlist={})) == 1


def test_an_engine_sentence_filled_from_a_claim_is_gated_by_the_claim_s_template(tmp_path):
    """A ``says`` filled by ``claims.card`` passes, since its number is a placeholder; the same
    words with no claim behind them, or with another number, fail."""
    reg = registry(tmp_path, rule())
    filled = reg.card("planted.few_clusters")
    assert claims.digit_gate(reg, strings=[("relation:x", filled)], allowlist={}) == []
    assert len(claims.digit_gate(claims.Registry(), strings=[("relation:x", filled)],
                                 allowlist={})) == 1
    other = filled.replace("10", "12")
    assert len(claims.digit_gate(reg, strings=[("relation:x", other)], allowlist={})) == 1


def test_a_stale_entry_is_named_and_a_present_one_is_not():
    allowlist = {"kept 1": "name", "gone 2": "example"}
    assert claims.stale_allowlist({"kept 1", "other"}, allowlist=allowlist) == ["gone 2"]
    assert claims.stale_allowlist({"kept 1", "gone 2"}, allowlist=allowlist) == []


@pytest.mark.parametrize("entries", [
    [{"text": "a 1", "tag": "vibes"}],
    [{"text": "a 1"}],
    [{"text": "a 1", "tag": "name"}, {"text": "a 1", "tag": "example"}],
    [{"text": "no digit here", "tag": "name"}],
])
def test_the_allowlist_refuses_an_entry_with_no_tag_a_wrong_one_a_repeat_or_no_digit(
        tmp_path, entries):
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"strings": entries}), encoding="utf-8")
    with pytest.raises(ValueError):
        claims.load_allowlist(path)


def test_a_well_formed_allowlist_loads(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"strings": [{"text": "SHA-256", "tag": "name"}]}),
                    encoding="utf-8")
    assert claims.load_allowlist(path) == {"SHA-256": "name"}
