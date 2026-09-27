"""``GET /api/teaching`` (M1_CONTRACT §5) and the findings' M1 fields over HTTP."""
from __future__ import annotations

from turbotab.core.teaching import QUESTION_KEYS
from turbotab.server.tests.conftest import open_by_path, wait_for


def test_teaching_answers_one_entry_per_question(client):
    response = client.get("/api/teaching")
    assert response.status_code == 200
    entries = response.json()
    assert [e["key"] for e in entries] == list(QUESTION_KEYS)
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
