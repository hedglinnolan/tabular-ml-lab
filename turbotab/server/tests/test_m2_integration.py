"""What the five M2 branches had to agree on once merged (Tier B, over HTTP), each found on the
five-lens journeys: the split's sentence names the seal's own draw, a repair's sentence is the
registry's, the missing-values sentence counts a kept Missing level, an assay's counts are not a
roster, the aggregation coach reads the stated repeats, and a binary model is about the event."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbotab.core.stages.modeling import coded_outcome
from turbotab.server.tests.conftest import SAMPLES, open_by_path, wait_for


def decide(client, pid, decision, status=200):
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == status, response.text
    return response.json()


def sentence(view) -> str:
    return view["decisions"][-1]["sentence"]


def roles(client, pid) -> dict[str, str]:
    artifact = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    return {c["column"]: c["proposed"] for c in artifact["columns"]}


def test_a_chronological_seal_is_said_as_drawn_and_a_repair_speaks_its_own_sentence(client):
    pid = open_by_path(client, SAMPLES / "clinical_longitudinal.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    wait_for(client, pid, {"findings": "fresh"})
    found = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    impossible = next(f for f in found if f["id"] == "pack::clinical::impossible_vs_extreme")
    option = next(o for o in impossible["repairs"] if o["key"] == "set_missing")
    said = sentence(decide(client, pid, option["decision"]))
    assert said == option["sentence"] and "{" not in said  # the registry's words, not its params

    decide(client, pid, {"kind": "set_target", "column": "progressed"})
    decide(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"})
    decide(client, pid, {"kind": "set_unit", "unit": "row"})
    decide(client, pid, {"kind": "set_temporal", "temporal": True, "time_column": "visit_date"})
    wait_for(client, pid, {"roles": "fresh"})
    decide(client, pid, {"kind": "set_roles", "roles": roles(client, pid)})
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    wait_for(client, pid, {"seal_plan": "fresh"})
    said = sentence(decide(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}))
    assert said.startswith("The latest `20%`") and "`visit_date`" in said and "`subject_id`" in said
    assert "random" not in said and "stratified" not in said
    split = wait_for(client, pid, {"split": "fresh"})
    artifact = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    assert artifact["chronology"]["drawn"] and artifact["basis"]["state"] == "grouped"
    assert split["stages"]["split"]["status"] == "fresh"


def test_a_kept_missing_level_is_counted_as_keeping_its_rows(client, tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame({
        "pid": [f"P{i:03d}" for i in range(n)],
        "age": rng.integers(30, 70, n),
        "meds": np.where(rng.random(n) < 0.6, None, np.where(rng.random(n) < 0.5, "yes", "no")),
        "y": rng.normal(100, 10, n).round(1),
    })
    path = tmp_path / "meds_blank.csv"
    frame.to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "y"})
    decide(client, pid, {"kind": "set_roles", "roles": {"pid": "identifier", "age": "covariate",
                                                       "meds": "covariate"}})
    wait_for(client, pid, {"cohort": "fresh"})
    said = sentence(decide(client, pid, {"kind": "set_missing", "strategy": "complete_case",
                                         "categorical": "missing_category"}))
    assert f"`{n}` of `{n}` rows remain" in said or f"all `{n}` rows remain" in said, said


def test_an_assay_lens_keeps_gene_counts_out_of_the_roster_reading(client):
    pid = open_by_path(client, SAMPLES / "genomics_expression.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["genomics"]})
    decide(client, pid, {"kind": "set_target", "column": "condition"})
    wait_for(client, pid, {"structure": "fresh"})
    structure = client.get(f"/api/projects/{pid}/stages/structure").json()["artifact"]
    assert not any(c.startswith("gene_") for c in structure["grain"]["suggested"])
    assert structure["grain"]["if_one_row"] is None
    decide(client, pid, {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "sample_id"})
    wait_for(client, pid, {"findings": "fresh"})
    found = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    counts = [f for f in found if f["id"].startswith("sentinel_missing__gene_")]
    assert counts and all("holds low counts" in f["summary"] for f in counts)


def test_the_aggregation_coach_reads_the_stated_repeats_and_their_dates(client):
    pid = open_by_path(client)  # dietary_recalls.csv
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    decide(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    decide(client, pid, {"kind": "set_unit", "unit": "unit"})
    wait_for(client, pid, {"structure": "fresh"})

    def coach(method):
        body = client.post(f"/api/projects/{pid}/preview",
                           json={"kind": "set_aggregation", "method": method}).json()
        return [n["text"] for v in body["views"] for n in v.get("coach") or []]

    assert any("replicates" in t for t in coach("mean"))  # stated repeats, not only answered
    assert not any("No time column" in t for t in coach("first"))  # recall_date orders them


def test_a_binary_outcome_is_coded_by_the_event_the_user_named():
    y = np.array(["case", "control", "control", "case"], dtype=object)
    assert coded_outcome("binary", y, "case").tolist() == [1, 0, 0, 1]
    assert coded_outcome("binary", y, "control").tolist() == [0, 1, 1, 0]
    assert coded_outcome("binary", np.array([0.0, 1.0, 1.0]), "1").tolist() == [0, 1, 1]
    assert coded_outcome("regression", y, "case") is y and coded_outcome("binary", y, None) is y


def preview(client, pid, decision):
    response = client.post(f"/api/projects/{pid}/preview", json=decision)
    assert response.status_code == 200, response.text
    return response.json()


def test_every_structural_answer_previews_its_picture(client, tmp_path):
    """M2_CONTRACT §9: the canvas previews each structural choice, not "Nothing… can be shown"."""
    pid = open_by_path(client, SAMPLES / "clinical_longitudinal.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "progressed"})
    wait_for(client, pid, {"target_info": "fresh", "oriented": "fresh"})
    event = preview(client, pid, {"kind": "set_event", "column": "progressed", "level": "1"})
    assert event["note"].startswith("`1` (`161` rows) becomes 1")
    grain = preview(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"})
    assert [v["kind"] for v in grain["views"]] == ["table_focus"]
    assert "`200` `subject_id` values, at most `3` each" in grain["views"][0]["caption"]
    decide(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"})
    decide(client, pid, {"kind": "set_unit", "unit": "row"})
    wait_for(client, pid, {"structure": "fresh", "cohort": "fresh"})
    yes = preview(client, pid, {"kind": "set_temporal", "temporal": True})  # the stated column
    flow = yes["views"][0]
    assert flow["after"][-1]["key"] == "holdout" and "`visit_date`" in flow["caption"]
    recorded = decide(client, pid, {"kind": "set_temporal", "temporal": True})
    assert recorded["state"]["temporal"]["time_column"] == "visit_date"  # the record names it

    m = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv").set_index("sample_id")
    turned = m.select_dtypes("number").T
    turned.index.name = "feature_id"
    source = tmp_path / "metabolomics_T.csv"
    turned.to_csv(source)
    tid = open_by_path(client, source)
    wait_for(client, tid, {"ingest": "fresh"})
    decide(client, tid, {"kind": "set_lens", "lenses": ["metabolomics"]})
    wait_for(client, tid, {"oriented": "fresh"})
    turn = preview(client, tid, {"kind": "set_orientation", "orientation": "feature_major"})
    focus = turn["views"][0]
    assert focus["kind"] == "table_focus" and [f["label"] for f in focus["story"]] == [
        "As supplied: features in rows", "Turned: one row per sample"]
    assert focus["story"][1]["columns"][0] == "sample_id" and len(turn["views"]) == 1
    # the corner at a table_focus's limits: 8 features by 8 samples, each cell the same either way
    supplied, after = focus["story"]
    assert len(supplied["rows"]) == len(after["rows"]) == 8 and len(after["columns"]) == 9
    first = supplied["rows"][0]["values"]
    assert all(after["rows"][j]["values"][str(first["feature_id"])] == first[s]
               for j, s in enumerate(supplied["columns"][1:]))
    assert f"`{turned.shape[1]}` rows, one per sample" in focus["caption"]
