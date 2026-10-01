"""``GET /api/teaching`` (M1_CONTRACT §5) and the findings' M1 fields over HTTP."""
from __future__ import annotations

from turbotab.core.teaching import TEACHING_KEYS
from turbotab.server.tests.conftest import open_by_path, wait_for


def test_teaching_answers_one_entry_per_question(client):
    response = client.get("/api/teaching")
    assert response.status_code == 200
    entries = response.json()
    assert [e["key"] for e in entries] == list(TEACHING_KEYS)
    energy = next(e for e in entries if e["key"] == "energy_adjustment")
    assert {o["value"] for o in energy["options"]} == {
        "none", "standard", "residual", "density_multivariate", "density", "partition"}
    assert energy["evidence"]["status"] == "CONVENTION"
    assert all(s["evidence"]["source"].startswith("research/") for s in energy["drawer"]["sections"])


def test_findings_carry_their_lever_over_http(client):
    pid = open_by_path(client)  # a project of its own: the shared one's lens stays unanswered
    response = client.post(f"/api/projects/{pid}/decisions", json={"kind": "set_lens", "lenses": ["dietary"]})
    assert response.status_code == 200, response.text
    wait_for(client, pid, {"findings": "fresh"})
    artifact = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]
    energy = next(f for f in artifact["findings"] if f["id"] == "pack::dietary::energy_adjustment")
    assert energy["routes_to"] == "energy_adjustment"
    assert energy["lever_label"] == "Adjust for energy"
    assert energy["summary"].endswith("nutrient effects are tangled with total energy.")
    assert all({"summary", "routes_to", "lever_label", "group"} <= set(f) for f in artifact["findings"])


def test_m2_sentences_coach_lines_and_units_over_http(client):
    """The recorded sentence counts what the server holds; the proposals carry the cards' coach
    lines; the outcome carries its unit; an evidence view carries its notes."""
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})
    for decision in ({"kind": "set_lens", "lenses": ["dietary"]}, {"kind": "set_target", "column": "hba1c"}):
        assert client.post(f"/api/projects/{pid}/decisions", json=decision).status_code == 200
    wait_for(client, pid, {"target_info": "fresh", "proposals": "fresh", "findings": "fresh"})
    record = client.post(f"/api/projects/{pid}/decisions",
                         json={"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    assert record.status_code == 200, record.text
    sentence = record.json()["decisions"][-1]["sentence"]
    assert sentence == ("Participants were declared to appear in more than one row, identified by "
                        "`participant_id`: `600` rows from `300` of them, at most `2` each.")
    info = client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"]
    assert (info["unit"], info["unit_source"]) == ("%", "pack")
    proposals = client.get(f"/api/projects/{pid}/stages/proposals").json()["artifact"]
    assert proposals["coach"]["exclusions"]["text"].endswith("kcal: likely under-reporting.")
    shown = client.get(f"/api/projects/{pid}/findings/pack::dietary::implausible_intake/evidence")
    assert shown.status_code == 200, shown.text
    assert any(n["text"].endswith("likely under-reporting.")
               for v in shown.json()["views"] for n in v["coach"])
