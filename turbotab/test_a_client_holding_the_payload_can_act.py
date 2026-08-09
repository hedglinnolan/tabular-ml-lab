"""`GUIDED-064` and `GUIDED-072` — the machine-readable form catches up with
the sentence.

One class, two surfaces, and the unifying test is the title: **a client holding
only the payload can act correctly.**

`GUIDED-072` — a 409 carries `exits=[revise, attest]`, the attest exit says
*"My answer is right — the data really is like this"*, and nothing in the
payload says what to send. The adjudicator read it, sent what it described, and
got a second 409, because the key is `acknowledge_contradiction` and the exit
never said so. Four attest exits, two different keys, and `acknowledge_blocker`
beside them: a client had to hold an out-of-band map from *which endpoint
refused* to *which key unlocks it*. That is the coupling `api._disclosures`
argues against in its own words, with the direction reversed.

`GUIDED-064` — a statement carries one badge and can make two claims the field
holds at different statuses. Four instances: the genomics counts finding
(SETTLED model ranking, DISPUTED normalization), the dietary energy-adjustment
finding (SETTLED that it is needed, CONVENTION which model), the volcano
(SETTLED y-axis rule, CONVENTION fold-change cut) and the diverging bar
(CONVENTION figure, DISPUTED neutral treatment). Nothing false reached a reader
because the finer status was in the prose. **The defect is that the badge a
machine reads was coarser than the sentence a human reads**, which inverts what
the badge is for.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, exits, figures, grain, missingness      # noqa: E402
from turbotab import figure_specs, packs as P, purpose            # noqa: E402

FIXTURES = Path(__file__).parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


# ── GUIDED-072 · the exit carries the way through ──────────────────────────

def _every_attest_exit():
    """Every attest exit the app can serve, from the modules that build them."""
    rows = [P._lens_attest("because the data really is like this"),
            grain._attest("because the data really is like this")]
    rows += [e for e in missingness.BLOCKER_EXITS if e["kind"] == exits.ATTEST]
    rows += [e for e in purpose.INDICATOR_EXITS if e["kind"] == exits.ATTEST]
    return rows


@pytest.mark.parametrize("exit_row", _every_attest_exit(),
                         ids=lambda e: e.get("payload_key", "?"))
def test_every_attest_exit_names_the_key_and_carries_the_retry(exit_row):
    assert exit_row["payload_key"] in exits.PAYLOAD_KEYS
    retry = exit_row["retry"]["payload"]
    assert retry == {exit_row["payload_key"]: True}
    assert exits.is_actionable(exit_row)
    assert exit_row["payload_key"] in exit_row["retry"]["how"]


def test_an_exit_cannot_be_built_around_a_key_nothing_reads():
    """A typo produces exactly the defect this repairs: an exit that renders
    perfectly, describes a real way through, and unlocks nothing."""
    with pytest.raises(exits.ExitError, match="not a key any decision handler"):
        exits.attest("label", "detail", "acknowledge_the_contradiction")


def test_a_client_with_only_the_409_can_construct_the_retry(client):
    """**The gate, end to end.** Refuse, read the exit, merge its payload into
    the request that was refused, post it again. Nothing out of band."""
    with open(FIXTURES / "metabolomics_untargeted.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("m.csv", fh, "text/csv")}).json()["id"]

    # A stated CLINICAL lens over a 396-column assay panel. The contradiction
    # detector fires in both directions, and this is the direction the
    # adjudicator hit.
    request = {"kind": "set_lens", "payload": {"lens": ["clinical"]}}
    refused = client.post(f"/project/{pid}/decision", json=request)
    assert refused.status_code == 409, refused.text

    detail = refused.json()["detail"]
    attest = next(e for e in detail["exits"] if e["kind"] == exits.ATTEST)
    # A CLIENT DOES THIS, holding nothing but the response body.
    request["payload"].update(attest["retry"]["payload"])
    accepted = client.post(f"/project/{pid}/decision", json=request)
    assert accepted.status_code == 200, accepted.text


def test_the_grain_contradiction_opens_the_same_way(client):
    with open(FIXTURES / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("d.csv", fh, "text/csv")}).json()["id"]

    request = {"kind": "set_grain",
               "payload": {"answer": "one_row_per_person",
                           "group_col": None}}
    refused = client.post(f"/project/{pid}/decision", json=request)
    # `AUDIT-039`, `L56-B2`. This stood down when the answer was ACCEPTED —
    # which is the regression, not a reason to stop looking. `dietary_recalls.csv`
    # carries repeated recalls per person, so `one_row_per_person` contradicts
    # the data, and that is a deterministic property of a shipped fixture rather
    # than something this test may decline over. If the app stops refusing, the
    # test that checks the refusal opens the same way must go RED.
    assert refused.status_code == 409, (
        f"`one_row_per_person` was answered {refused.status_code} on "
        f"dietary_recalls.csv, which carries more than one recall per person. "
        f"Either the grain check stopped contradicting it or the fixture "
        f"changed; both are findings, and neither is a skip. AUDIT-039.")
    attest = next(e for e in refused.json()["detail"]["exits"]
                  if e["kind"] == exits.ATTEST)
    request["payload"].update(attest["retry"]["payload"])
    assert client.post(f"/project/{pid}/decision",
                       json=request).status_code == 200


def test_a_resolve_exit_needs_nothing_and_says_nothing():
    """`revise` sends the user back to the question. Giving it a payload key
    would be inventing a mechanism to be symmetrical with."""
    assert exits.is_actionable(P._LENS_RESOLVE)
    assert "payload_key" not in P._LENS_RESOLVE


# ── GUIDED-064 · the badge is as fine as the sentence ──────────────────────

def test_a_two_claim_finding_carries_a_badge_per_claim():
    df = pd.read_csv(FIXTURES / "genomics_expression.csv")
    finding = next(f for f in P.findings(df, [P.GENOMICS])
                   if f["id"] == "pack::genomics::counts_p_over_n")
    badge = finding["evidence"]
    claims = {c["key"]: c for c in badge["claims"]}
    assert claims["model_ranking"]["evidence_status"] == P.SETTLED
    assert claims["normalization"]["evidence_status"] == P.DISPUTED
    assert claims["normalization"]["both_sides"]
    assert all(len(c["statement"]) > 40 for c in badge["claims"])


def test_the_headline_may_preselect_goes_false_when_a_claim_is_disputed():
    """**The part that makes this more than a display change.** The headline
    evidence is SETTLED, so `may_preselect` read True while one of the two
    things the finding says is DISPUTED — and *DISPUTED is never defaulted
    silently* is the badge's own obligation. A machine acting on the headline
    alone would have pre-selected across a disagreement it could not see."""
    df = pd.read_csv(FIXTURES / "genomics_expression.csv")
    badge = next(f for f in P.findings(df, [P.GENOMICS])
                 if f["id"] == "pack::genomics::counts_p_over_n")["evidence"]
    assert badge["evidence_status"] == P.SETTLED
    assert badge["may_preselect"] is False
    assert badge["weakest_status"] == P.DISPUTED
    # The SETTLED claim keeps its own permission; the coarsening is undone
    # rather than pushed down.
    settled = next(c for c in badge["claims"] if c["key"] == "model_ranking")
    assert settled["may_preselect"] is True


def test_a_convention_claim_lowers_the_headline_without_forbidding_it():
    """The dietary instance: the adjustment being needed is SETTLED and which
    model to use is a convention. CONVENTION may still pre-select, stated AS
    convention, so `may_preselect` stays True and `weakest_status` says which."""
    df = pd.read_csv(FIXTURES / "dietary_recalls.csv")
    badge = next(f for f in P.findings(df, [P.DIETARY])
                 if f["id"] == "pack::dietary::energy_adjustment")["evidence"]
    assert badge["evidence_status"] == P.SETTLED
    assert badge["weakest_status"] == P.CONVENTION_STATUS
    assert badge["may_preselect"] is True


def test_a_single_claim_statement_is_unchanged():
    """Additive. A finding that says one thing carries no claims and no
    `weakest_status`, because inventing one would make every consumer handle a
    field that means nothing on most findings."""
    df = pd.read_csv(FIXTURES / "dietary_recalls.csv")
    badge = next(f for f in P.findings(df, [P.DIETARY])
                 if f["id"] == "pack::dietary::compositional")["evidence"]
    assert "claims" not in badge
    assert "weakest_status" not in badge
    assert badge["may_preselect"] is True


@pytest.mark.parametrize("figure_id,weakest", [
    ("volcano", P.CONVENTION_STATUS),
    ("diverging_stacked_bar", P.DISPUTED),
])
def test_the_two_figures_carry_their_claims_to_the_bundle(figure_id, weakest):
    """The figure layer's half. `to_dict` and the bundle row both carry the
    claims, so a consumer reading either sees the same granularity the caption
    has always had."""
    spec = figures.REGISTRY[figure_id]
    served = spec.to_dict()
    assert served["weakest_status"] == weakest
    assert len(served["claims"]) == 2
    for claim in served["claims"]:
        assert claim["source"].startswith("research/")


def test_the_disputed_figure_claim_states_both_sides():
    served = figures.REGISTRY["diverging_stacked_bar"].to_dict()
    disputed = next(c for c in served["claims"]
                    if c["evidence_status"] == P.DISPUTED)
    assert len(disputed["both_sides"]) > 80
    assert served["may_preselect"] is False


def test_a_claim_cannot_be_badged_with_a_dict_or_left_unstated():
    with pytest.raises(P.EvidenceError, match="must be an `Evidence`"):
        P.Claim("k", "a statement long enough to be about something",
                {"evidence_status": "SETTLED"})
    with pytest.raises(P.EvidenceError, match="a claim states what it is"):
        P.Claim("k", "too short",
                P.Evidence(status=P.SETTLED,
                           source="research/GENOMICS_PACK.md#08 · Modeling at p >> n"))


def test_every_claim_source_resolves_through_the_gate():
    """The gate walks them now, so a claim citing nothing fails the commit."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "docs" / "turbotab" / "tools"))
    import importlib
    tool = importlib.import_module("evidence")
    assert tool.check() == 0
