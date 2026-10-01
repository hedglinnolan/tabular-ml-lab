"""The working table over HTTP (M2_CONTRACT §2, Tier B): once rows are combined, the table, the
column summaries, the previews and the Router all read one row per person."""
from __future__ import annotations

from turbotab.server import schemas
from turbotab.server.tests.conftest import open_by_path, wait_for


def decide(client, pid, decision, status=200):
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == status, response.text
    return response.json()


def test_combining_a_persons_recalls_reaches_the_table_the_previews_and_the_router(client):
    pid = open_by_path(client)  # dietary_recalls.csv: 300 people × 2 recalls
    wait_for(client, pid, {"ingest": "fresh", "oriented": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    wait_for(client, pid, {"structure": "fresh", "working": "fresh"})
    assert client.get(f"/api/projects/{pid}/table?limit=1").json()["total_rows"] == 600

    refusal = decide(client, pid, {"kind": "set_grain", "grain": "one_row_per_unit"}, status=409)
    schemas.Refusal.model_validate(refusal)
    assert refusal["error"]["code"] == "data_repeats"
    assert refusal["error"]["exits"][-1]["decision"]["acknowledged"] is True

    decide(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    view = wait_for(client, pid, {"structure": "fresh"})
    steps = {s["key"]: s for s in client.get(f"/api/projects/{pid}").json()["interview"]}
    assert steps["repeat_kind"]["status"] == "skipped" and "`recall_date`" in steps["repeat_kind"]["reason"]
    structure = client.get(f"/api/projects/{pid}/stages/structure").json()["artifact"]
    schemas.StructureArtifact.model_validate(structure)
    assert structure["aggregation"]["recommended"] == "mean"

    decide(client, pid, {"kind": "set_unit", "unit": "unit"})
    decide(client, pid, {"kind": "set_aggregation", "method": "mean"})
    view = wait_for(client, pid, {"working": "fresh", "cohort": "fresh", "target_info": "fresh"})
    working = client.get(f"/api/projects/{pid}/stages/working").json()["artifact"]
    schemas.WorkingArtifact.model_validate(working)
    assert working["n_rows"] == 300 and working["aggregation"]["n_source_rows"] == 600
    assert view["summary"]["n_rows"] == 600  # the file is still the file

    window = client.get(f"/api/projects/{pid}/table?limit=3").json()
    assert window["total_rows"] == 300
    summaries = {c["name"]: c for c in client.get(f"/api/projects/{pid}/columns").json()}
    assert summaries["participant_id"]["n_unique"] == 300 and summaries["energy_kcal"]["n"] == 300

    rule = {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    preview = client.post(f"/api/projects/{pid}/preview",
                          json={"kind": "set_exclusions", "rules": [rule]}).json()
    assert preview["views"][0]["kind"] == "row_flow"
    assert preview["views"][0]["after"][0]["n"] == 300  # the flow starts from people, not recalls
    steps = {s["key"]: s["status"] for s in view["interview"]}
    assert steps["aggregation"] == "answered" and steps["temporal"] == "not_applicable"
