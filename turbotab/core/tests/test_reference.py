"""The methods reference and the review packets stay what the code generates (V2 definition of done
§4: "a methods reference generated from the contracts"; "a per-domain review packet").

* The committed ``METHODS_REFERENCE.md`` is regenerated from the registry and must be identical:
  a contract, an option, a label, a relation or a family that changes without regenerating it
  fails here, naming the command that fixes it.
* Every committed review packet is regenerated from the registry and its stored journey captures
  (no fit runs here): a registry that moved on fails until the packet is rebuilt with ``--reuse``.
* The lens catalog covers the registry exactly: a contract or a family registered without a lens
  fails, and so does a lens entry for something no longer registered.
* Every gap names real code: its decision kinds are kinds the decision log accepts, its modules
  import, its labels are a question ``custom_sound`` labels, and the contracts it says cover part
  of it exist.
* Every sentence a contract names by ``module:function`` resolves.
* The v2 citation draft is Citation File Format 1.2.0 with the paper as its preferred citation, and
  the root ``CITATION.cff`` (Classic's) is not the draft.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import get_args

import pytest

from turbotab.core.reference import catalog, journeys, methods, packet

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "docs" / "turbotab-next" / "release" / "CITATION.cff.draft"


def test_the_methods_reference_is_what_the_contracts_generate():
    committed = methods.OUT.read_text(encoding="utf-8") if methods.OUT.is_file() else ""
    assert committed == methods.render(), (
        f"{methods.OUT.relative_to(REPO)} is stale: run `{methods.COMMAND}`")


@pytest.mark.parametrize("lens", catalog.LENSES)
def test_each_review_packet_is_what_the_registry_and_its_captures_generate(lens):
    path = packet.OUT / f"{lens}.md"
    assert path.is_file(), f"no packet for {lens}: run `{packet.COMMAND} --lens {lens}`"
    assert path.read_text(encoding="utf-8") == packet.render(lens, packet.captures_for(lens)), (
        f"{path.relative_to(REPO)} is stale: run `{packet.COMMAND} --lens {lens} --reuse`")


def test_every_contract_and_family_has_its_lenses_and_nothing_else_does():
    from turbotab.core.contracts import contracts

    registered = set(contracts())
    assert registered, "no contracts registered: the check would pass vacuously"
    assert sorted(registered - set(catalog.CONTRACT_LENSES)) == [], \
        "contracts with no lens: add them to turbotab/core/reference/catalog.py"
    assert sorted(set(catalog.CONTRACT_LENSES) - registered) == [], \
        "lens entries for contracts that are not registered"
    families = {f.key for f in methods.families()}
    assert sorted(families ^ set(catalog.FAMILY_LENSES)) == []
    allowed = {*catalog.LENSES, catalog.SHARED}
    for table in (catalog.CONTRACT_LENSES, catalog.FAMILY_LENSES):
        for key, lenses in table.items():
            assert lenses and set(lenses) <= allowed, key
            assert not (catalog.SHARED in lenses and len(lenses) > 1), key


def test_every_gap_names_real_code_and_no_contract():
    from turbotab.core import custom_sound
    from turbotab.core.contracts import contracts
    from turbotab.core.decisions import Decision

    union = get_args(Decision)[0]
    kinds = {m.model_fields["kind"].default for m in get_args(union)}
    registered = contracts()
    keys = [g.key for g in catalog.GAPS]
    assert len(keys) == len(set(keys))
    labeled = set(get_args(custom_sound.LabeledQuestion.model_fields["question"].annotation))
    for g in catalog.GAPS:
        assert g.key not in registered, f"{g.key} has a contract: it is no longer a gap"
        assert set(g.decisions) <= kinds, (g.key, sorted(set(g.decisions) - kinds))
        for module in g.modules:
            importlib.import_module(module)
        if g.labels:
            assert g.labels.split(":", 1)[1] in labeled, g.key
        assert set(g.contracted_parts) <= set(registered), g.key
        assert set(g.lenses) <= {*catalog.LENSES, catalog.SHARED}, g.key


def test_every_sentence_a_contract_names_resolves():
    for c in methods.ordered_contracts():
        if isinstance(c.sentence, str) and re.fullmatch(r"[\w.]+:[\w.]+", c.sentence):
            assert callable(methods.resolve(c.sentence)), c.key


def test_every_lens_has_a_journey_for_each_purpose_and_every_capture_one_journey():
    for lens in catalog.LENSES:
        assert sorted(j.purpose for j in journeys.for_lens(lens)) == ["inference", "prediction"]
    stored = {p.stem for p in journeys.CAPTURES.glob("*.json")}
    assert stored <= set(journeys.JOURNEYS), sorted(stored - set(journeys.JOURNEYS))


def test_the_citation_draft_cites_the_paper_first_and_v2_as_the_software():
    yaml = pytest.importorskip("yaml")
    draft = yaml.safe_load(DRAFT.read_text(encoding="utf-8"))
    assert draft["cff-version"] == "1.2.0"
    assert draft["type"] == "software" and str(draft["version"]) == "2.0.0"
    for key in ("message", "title", "authors"):
        assert draft[key], key
    paper = draft["preferred-citation"]
    assert paper["type"] == "article" and paper["title"] and paper["authors"]
    classic = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(classic["version"]) == "1.0.0"  # the root file is still Classic's
    assert any(str(r.get("version")) == str(classic["version"]) for r in draft["references"])
