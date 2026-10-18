"""``GET /api/teaching`` (M1_CONTRACT §5) and the findings' M1 fields over HTTP."""
from __future__ import annotations

from turbotab.core.teaching import TEACHING_KEYS
from turbotab.server.tests.conftest import open_by_path, prepare, wait_for


def test_teaching_answers_one_entry_per_question(client):
    response = client.get("/api/teaching")
    assert response.status_code == 200
    entries = response.json()
    assert [e["key"] for e in entries] == list(TEACHING_KEYS)
    energy = next(e for e in entries if e["key"] == "energy_adjustment")
    assert {o["value"] for o in energy["options"]} == {
        "none", "standard", "residual", "residual_energy_dropped", "density_multivariate", "density",
        "partition", "all_components"}
    assert energy["evidence"]["status"] == "CONVENTION"
    assert all(s["evidence"]["source"].startswith("research/") for s in energy["drawer"]["sections"])


def test_the_roles_card_is_served_in_the_goals_words(client):
    """W1-R round 2 (R2's P3a, the walkthrough's finding 3c): ``?purpose=prediction`` serves the
    roles card that names no study factor (``content.ROLES_UNDER_PREDICT``); with no goal, or under
    inference, every card is as before, so a client that names none reads what it read."""
    from turbotab.core.teaching.content import ROLES, ROLES_UNDER_PREDICT

    default = client.get("/api/teaching").json()
    predict = client.get("/api/teaching", params={"purpose": "prediction"}).json()
    infer = client.get("/api/teaching", params={"purpose": "inference"}).json()
    assert infer == default
    roles = {key: next(e for e in served if e["key"] == "roles")
             for key, served in (("default", default), ("predict", predict))}
    assert roles["default"]["one_liner"] == ROLES["one_liner"]
    assert roles["predict"]["one_liner"] == ROLES_UNDER_PREDICT["one_liner"]
    assert [o["label"] for o in roles["predict"]["options"]] == [
        o["label"] for o in ROLES_UNDER_PREDICT["options"]]
    # only the roles card differs
    assert [e["key"] for e in predict] == [e["key"] for e in default]
    assert [e["key"] for e, f in zip(default, predict) if e != f] == ["roles"]
    assert client.get("/api/teaching", params={"purpose": "describe"}).status_code == 422


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
    grain = {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"}
    prepare(client, pid, grain)  # the purpose question comes first (M2_CONTRACT §12.2)
    record = client.post(f"/api/projects/{pid}/decisions", json=grain)
    assert record.status_code == 200, record.text
    sentence = record.json()["decisions"][-1]["sentence"]
    assert sentence == ("Participants were declared to appear in more than one row, identified by "
                        "`participant_id`: `600` rows from `300` of them, at most `2` each.")
    # Audit IN-05: a unit the name does not spell out is proposed, never stated, until recorded.
    info = client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"]
    assert (info["unit"], info["unit_source"], info["proposed_unit"]) == (None, None, "%")
    record = client.post(f"/api/projects/{pid}/decisions",
                         json={"kind": "set_outcome_unit", "column": "hba1c", "unit": "%"})
    assert record.status_code == 200, record.text
    assert record.json()["decisions"][-1]["sentence"] == "The unit of `hba1c` was recorded as percent."
    wait_for(client, pid, {"target_info": "fresh"})
    info = client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"]
    assert (info["unit"], info["unit_source"]) == ("%", "decision")
    # BLUEPRINT §14.3: a day count is recorded, never read from a band of values; the fixture's
    # truth (one 24-hour recall per row) is recorded, and the misreporting words follow it.
    record = client.post(f"/api/projects/{pid}/decisions", json={
        "kind": "set_column_unit", "column": "energy_kcal", "unit": "kcal", "days": 1})
    assert record.status_code == 200, record.text
    wait_for(client, pid, {"proposals": "fresh", "findings": "fresh"})
    proposals = client.get(f"/api/projects/{pid}/stages/proposals").json()["artifact"]
    assert proposals["coach"]["exclusions"]["text"].endswith("kcal: likely under-reporting.")
    shown = client.get(f"/api/projects/{pid}/findings/pack::dietary::implausible_intake/evidence")
    assert shown.status_code == 200, shown.text
    assert any(n["text"].endswith("likely under-reporting.")
               for v in shown.json()["views"] for n in v["coach"])
