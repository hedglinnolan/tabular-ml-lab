"""`GUIDED-060` — the refusal, probed from outside, with its offer resolved.

`prevalence_of_inadequacy` refuses in four cases and each refusal names what it
CAN draw instead, because **a refusal that offers nothing is indistinguishable
from a missing feature** and the user still has a real question. Two problems,
and the second is the one that generalizes:

1. Two of the four offers named a figure that does not exist. The registry holds
   `calibration`, `pca_scores` and `shrinkage`; the offers named
   `distribution_against_ai` and `distribution_against_ear_and_rda`. **The AI
   case is the flagship** — the reason the nutrition pack was built first — and
   it offered a renderer that cannot run. An offer naming an unbuilt figure is
   the same failure the offer exists to prevent, arriving one layer later and at
   a worse moment.
2. `test_every_refusal_offers_something_it_can_draw` asserted the three strings
   were truthy. That is the shape of the claim and not its resolution, and it
   passed every time. This project resolves a prior's source (`evidence.py`) and
   a `FIXED` row's test (`ledger.py check`); an offer's draw target is the third
   reference of that kind and nothing resolved it.

**The two figures were deliberately not built.** They need a Dietary Reference
Intake table, none ships in this repository, and `DOMAIN_SCIENCE.md` §04 says
they must be read from NASEM rather than remembered — a wrong EAR does not look
wrong. `GUIDED-067` carries that, and `figures.PENDING` says so where an offer
can reach it.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, figures, nutrition                          # noqa: E402
from turbotab import figure_specs                                     # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _decide(client, pid, what, **payload):
    response = client.post(f"/project/{pid}/decision",
                           json={"kind": what, "payload": payload})
    assert response.status_code == 200, (what, response.text)


@pytest.fixture(scope="module")
def dietary(client):
    with open(FIXTURES / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("d.csv", fh, "text/csv")}).json()["id"]
    _decide(client, pid, "set_lens", lens=["dietary"])
    _decide(client, pid, "set_target", column="hba1c")
    _decide(client, pid, "set_grain", answer="people_repeat",
            group_col="participant_id")
    _decide(client, pid, "set_repeat_kind", kind="repeats")
    return pid


# ── the gate: the whole path, from outside ──────────────────────────────────

def test_a_single_days_intake_is_refused_and_the_shrinkage_plot_arrives_drawn(
        client, dietary):
    """The gate. A dietary-lens project asks for a prevalence of inadequacy
    from a single day's intake and gets three things: the refusal, the reason,
    and a figure it can look at — not the name of one."""
    response = client.get(
        f"/project/{dietary}/nutrition/prevalence"
        f"?nutrient=calcium&basis=single_day&reference_kind=EAR")
    assert response.status_code == 200, (
        "a refusal is an answer, not a malformed request")
    body = response.json()

    assert body["refused"] is True
    # The reason, in the app's own words rather than a status code.
    assert "usual-intake distribution" in body["reason"]
    assert "overstated, in both" in body["reason"]
    # The badge, because a refusal is the sharpest claim the pack makes.
    assert body["evidence_status"] == "SETTLED"
    assert body["source"].startswith("research/NUTRITION_PACK.md#")

    offer = body["offer"]
    assert offer["pending"] is False
    assert offer["resolved"]["status"] == figures.REGISTERED_STATUS

    figure = body["figure"]
    assert figure["id"] == "shrinkage"
    assert figure["caption"], "the figure arrived without its caption"
    assert len(figure["annotations"]) == 6
    assert all(a["value"] != "not estimable" for a in figure["annotations"])
    assert all(item["passed"] for item in figure["checklist"]), (
        "the offered figure fails its own publication-grade checklist")
    # And it is the figure the refusal is ABOUT: the narrowing is the size of
    # the error a prevalence computed from one day would have carried.
    payload = figure["payload"]
    assert payload["spread_single_day"] > payload["spread_usual_intake"]


def test_the_naive_mean_gets_the_same_refusal_and_the_same_figure(client, dietary):
    """*"We averaged the two 24-hour recalls to obtain usual intake"* followed
    by a prevalence claim is a documented failure, not a simplification."""
    body = client.get(
        f"/project/{dietary}/nutrition/prevalence"
        f"?nutrient=calcium&basis=naive_mean&reference_kind=EAR").json()
    assert body["refused"] is True
    assert "naive mean of the available days" in body["reason"]
    assert body["figure"]["id"] == "shrinkage"


# ── the two that name a figure nobody has built ─────────────────────────────

@pytest.mark.parametrize("nutrient,kind,target", [
    ("fiber", "AI", "distribution_against_ai"),
    ("calcium", "RDA", "distribution_against_ear_and_rda"),
])
def test_an_unbuilt_offer_arrives_as_a_record_and_not_as_a_promise(
        client, dietary, nutrient, kind, target):
    """The offer is not withdrawn — the target is planned, `NUTRITION_PACK.md`
    §07 figure E specifies it. What changes is that it comes back saying it is
    pending, what it needs, and which row is blocking it, so the user can tell
    *"the app cannot draw this yet"* from *"the app will not draw this"*."""
    body = client.get(
        f"/project/{dietary}/nutrition/prevalence"
        f"?nutrient={nutrient}&basis=usual_intake&reference_kind={kind}").json()
    assert body["refused"] is True
    resolved = body["offer"]["resolved"]
    assert body["offer"]["pending"] is True
    assert resolved["id"] == target
    assert resolved["status"] == figures.PENDING_STATUS
    assert "Dietary Reference Intake" in resolved["needs"]
    assert resolved["blocked_by"] == "GUIDED-067"
    # NOT drawn, and not pretended. A pending figure has no payload.
    assert "figure" not in body


def test_a_pending_figures_own_citation_resolves(client):
    """The same rule the evidence gate applies to a prior's source: a citation
    nobody can follow is a citation nobody can check."""
    assert figures.PENDING, "the pending table is empty; nothing is asserted"
    for entry in figures.PENDING.values():
        filename, _, section = entry.specified_in.partition("#")
        path = ROOT / "docs" / "turbotab" / filename
        assert path.exists(), (entry.id, filename)
        headings = {m.group(1).strip() for m in re.finditer(
            r"^#{1,6}\s+(.*?)\s*$", path.read_text(encoding="utf-8"), re.M)}
        assert section in headings, (entry.id, section)


def test_a_pending_figure_names_a_ledger_row_that_exists(client):
    """`blocked_by` is a reference like any other, so it resolves too. A
    pending figure blocked by a row nobody filed is a promise nothing tracks."""
    import json
    rows = {r["id"] for r in json.loads(
        (ROOT / "docs" / "turbotab" / "data" / "findings.json")
        .read_text(encoding="utf-8"))}
    for entry in figures.PENDING.values():
        assert entry.blocked_by in rows, (entry.id, entry.blocked_by)


# ── resolution refuses what it cannot resolve ───────────────────────────────

def test_a_draw_target_in_neither_table_is_refused_rather_than_shrugged_at():
    """An id in neither table is not a pending figure. It is a typo, or a
    figure somebody imagined, and offering one is worse than offering nothing
    because it reads as a feature."""
    with pytest.raises(figures.FigureError, match="neither a registered"):
        figures.resolve("intake_against_the_moon")
    with pytest.raises(figures.FigureError, match="neither a registered"):
        figures.resolve_offer({"draw": "", "label": "x"})


def test_a_figure_cannot_be_both_built_and_pending():
    """The offer that named it would resolve to two different answers."""
    with pytest.raises(figures.FigureError, match="cannot be both"):
        figures.register_pending(figures.Pending(
            id="shrinkage", title="x",
            specified_in="research/NUTRITION_PACK.md#07 · EDA and presentation",
            needs="n", blocked_by="GUIDED-060"))


def test_every_refusal_in_the_module_resolves_its_own_offer():
    """Wider than the prevalence path: every `PackRefusal` the nutrition module
    can raise, including the usual-intake ones the figure layer raises when a
    person has one recall. A sweep that stopped at the four this finding named
    would terminate where the sweeper's attention ended (`LOOP.md` §06.5)."""
    import pandas as pd

    raised = []
    for nutrient, basis, kind in [("fiber", nutrition.USUAL_INTAKE, "AI"),
                                  ("calcium", nutrition.USUAL_INTAKE, "RDA"),
                                  ("calcium", nutrition.SINGLE_DAY, "EAR"),
                                  ("calcium", nutrition.NAIVE_MEAN, "EAR")]:
        with pytest.raises(nutrition.PrevalenceRefusal) as caught:
            nutrition.prevalence_of_inadequacy(nutrient, basis=basis,
                                               reference_kind=kind)
        raised.append(caught.value)

    one_day = pd.DataFrame({"pid": [f"P{i}" for i in range(30)],
                            "kcal": [1800 + i * 10 for i in range(30)]})
    with pytest.raises(nutrition.UsualIntakeRefusal) as caught:
        nutrition.usual_intake_series(one_day, person_col="pid",
                                      value_col="kcal")
    raised.append(caught.value)

    flat = pd.DataFrame({"pid": [f"P{i // 2}" for i in range(60)],
                         "kcal": [1000 if i % 2 else 3000 for i in range(60)]})
    with pytest.raises(nutrition.UsualIntakeRefusal) as caught:
        nutrition.usual_intake_series(flat, person_col="pid", value_col="kcal")
    raised.append(caught.value)

    assert len(raised) == 6
    for refusal in raised:
        resolved = figures.resolve_offer(refusal.offer)["resolved"]
        assert resolved["status"] in (figures.REGISTERED_STATUS,
                                      figures.PENDING_STATUS)
        assert refusal.to_dict()["evidence_status"], "an unbadged refusal"


# ── the endpoint's own boundaries ───────────────────────────────────────────

def test_the_prevalence_question_needs_the_dietary_lens(client):
    """The app does not infer the field from column names. Answering the lens
    is what licenses the pack's reference logic."""
    with open(FIXTURES / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("d.csv", fh, "text/csv")}).json()["id"]
    _decide(client, pid, "set_lens", lens=["other"])
    response = client.get(
        f"/project/{pid}/nutrition/prevalence?nutrient=calcium")
    assert response.status_code == 409
    assert "does not infer the field" in response.json()["detail"]


def test_the_valid_case_answers_and_says_which_method(client, dietary):
    """The path is not all refusal. A usual-intake distribution against the EAR
    is the case the cut-point method is for, and it comes back badged."""
    body = client.get(
        f"/project/{dietary}/nutrition/prevalence"
        f"?nutrient=calcium&basis=usual_intake&reference_kind=EAR").json()
    assert body["refused"] is False
    assert body["method"] == "cut_point"
    assert body["evidence_status"] == "SETTLED"


def test_iron_in_menstruating_women_routes_rather_than_refuses(client, dietary):
    """A skewed requirement distribution, so the cut-point method does not
    apply and the probability approach does. The question has an answer, by a
    different route — routing is not refusal and the payload says which."""
    body = client.get(
        f"/project/{dietary}/nutrition/prevalence?nutrient=iron"
        f"&basis=usual_intake&reference_kind=EAR&stratum=menstruating").json()
    assert body["refused"] is False
    assert body["method"] == "probability_approach"
    assert "skewed requirement" in body["note"]


def test_an_unknown_basis_is_refused_with_the_three_that_are_known(client, dietary):
    body = client.get(
        f"/project/{dietary}/nutrition/prevalence"
        f"?nutrient=calcium&basis=two_day_average")
    assert body.status_code == 400
    assert "usual_intake" in body.json()["detail"]
