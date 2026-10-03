"""Previews, the interview and recorded sentences over HTTP (Tier B)."""
from __future__ import annotations

import pytest

from turbotab.core import voice
from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, words
from turbotab.server import schemas
from turbotab.server.tests.conftest import (
    answer_settled, open_by_path, prepare, settle_reads, wait_for,
)


def decide(client, pid, decision):
    prepare(client, pid, decision)  # the questions before it, answered as usual (M2 §12.2)
    # each reading below high confirmed on its own (BLUEPRINT §14.1, the readings ledger)
    response = answer_settled(client, pid, None, decision)
    assert response.status_code == 200, response.text
    return response.json()


def recorded(client, pid) -> int:
    return len(client.get(f"/api/projects/{pid}").json()["decisions"])


def preview(client, pid, decision):
    prepare(client, pid, decision)  # a question the Router has not reached refuses its preview too
    settle_reads(client, pid, decision)  # the readings it asks about, answered from the truth
    before = recorded(client, pid)
    response = client.post(f"/api/projects/{pid}/preview", json=decision)
    assert response.status_code == 200, response.text
    assert recorded(client, pid) == before  # a preview records nothing
    body = response.json()
    result = schemas.PreviewResult.model_validate(body)
    assert 1 <= len(result.views) <= 3, body
    for view in result.views:
        assert words(view.title) <= TITLE_WORDS, view.title
        assert words(view.caption) <= CAPTION_WORDS, view.caption
    return body


@pytest.fixture(scope="module")
def project(client):
    """dietary_recalls.csv under the dietary lens, outcome hba1c, roles, cohort and target read."""
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    wait_for(client, pid, {"roles": "fresh", "target_info": "fresh", "cohort": "fresh"})
    return pid


def test_every_row_kind_previews_within_its_word_budgets_and_records_nothing(client, project):
    pid = project
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    schemas.RolesArtifact.model_validate(roles)
    proposed = {c["column"]: c["proposed"] for c in roles["columns"]}

    lineage = preview(client, pid, {"kind": "set_roles", "roles": proposed})["views"][0]
    assert lineage["kind"] == "lineage" and lineage["before"] is None
    matrix = {n["column"] for n in lineage["after"]["nodes"] if n["lane"] == "matrix"}
    assert {"energy_kcal", "protein_g"} <= matrix and "participant_id" not in matrix

    decide(client, pid, {"kind": "set_roles", "roles": proposed})
    rule = {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    views = preview(client, pid, {"kind": "set_exclusions", "rules": [rule]})["views"]
    assert [v["kind"] for v in views] == ["row_flow", "distribution"]
    assert views[0]["after"][-1]["n"] == 580 and "`20`" in views[0]["caption"]
    assert [(m["value"], m["label"], m["group"]) for m in views[1]["marks"]] == [
        (500.0, "500 kcal", None), (5000.0, "5,000 kcal", None)]

    views = preview(client, pid, {"kind": "set_missing", "strategy": "complete_case"})["views"]
    assert views[0]["kind"] == "row_flow" and views[0]["after"][-1]["key"] == "complete_cases"

    body = preview(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    fork = body["views"][0]
    assert [s["key"] for s in fork["after"][-2:]] == ["train", "holdout"]
    assert "participant_id" in fork["caption"]  # people repeat, so the split is grouped
    # Drawing the seal reads the identifier on every row, and the basis says so (M2 §3's audit).
    assert body["basis"].startswith("Reads participant_id on all ") and "no outcome value" in body["basis"]


def test_after_the_split_previews_say_the_held_out_rows_are_sealed(client, project):
    pid = project
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    decide(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    wait_for(client, pid, {"cohort": "fresh", "split": "fresh"})
    split = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    schemas.SplitArtifact.model_validate(split)
    assert split["grouped_by"] == "participant_id" and split["n_holdout"] + split["n_train"] == 600
    schemas.CohortArtifact.model_validate(client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"])
    body = preview(client, pid, {"kind": "set_exclusions", "rules": []})
    assert "held-out rows stay sealed" in body["basis"]
    assert body["views"][0]["before"][0]["label"] == "Rows not held out"


def test_a_preview_of_a_refused_option_is_the_refusal(client, project):
    bad = {"kind": "set_exclusions", "rules": [{"column": "sex", "low": 1, "high": 2, "reason": "r"}]}
    prepare(client, project, bad)
    response = client.post(f"/api/projects/{project}/preview", json=bad)
    assert response.status_code == 409 and response.json()["error"]["code"] == "not_numeric"


def test_a_kind_without_a_picture_says_so(client, project):
    body = client.post(f"/api/projects/{project}/preview", json={"kind": "set_purpose", "purpose": "inference"}).json()
    assert body["views"] == [] and body["note"]


def test_the_project_view_carries_the_interview(client, project):
    view = schemas.ProjectView.model_validate(client.get(f"/api/projects/{project}").json())
    steps = {s.key: s for s in view.interview}
    assert [s.key for s in view.interview][:5] == ["lens", "orientation", "target", "event", "task"]
    assert steps["orientation"].status == "not_applicable"  # no assay lens: never asked
    assert steps["lens"].status == "answered" and steps["lens"].decision_id
    assert [s.status for s in view.interview].count("open") <= 1


def test_each_record_carries_the_sentence_the_voice_writes(client, monkeypatch):
    seen = []

    def sentence_for(decision, state_before, ctx):
        seen.append((decision.kind, state_before.lens, ctx.columns is not None))
        return f"Recorded `{decision.kind}`."

    monkeypatch.setattr(voice, "sentence_for", sentence_for)
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    view = decide(client, pid, {"kind": "set_purpose", "purpose": "prediction"})
    assert [r["sentence"] for r in view["decisions"]] == [
        "Recorded `set_lens`.", "Recorded `set_target`.", "Recorded `set_purpose`."]
    assert seen == [("set_lens", None, True), ("set_target", ["dietary"], True),
                    ("set_purpose", ["dietary"], True)]


def test_recorded_sentences_count_rows_as_the_participant_flow_does(client):
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    wait_for(client, pid, {"roles": "fresh", "target_info": "fresh", "cohort": "fresh"})
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    rule = {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    decide(client, pid, {"kind": "set_exclusions", "rules": [rule]})
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    wait_for(client, pid, {"cohort": "fresh"})
    steps = {s["key"]: s for s in client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["steps"]}
    view = decide(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    said = {r["decision"]["kind"]: r["sentence"] for r in view["decisions"]}

    assert said["set_exclusions"].startswith(f"`{steps['exclusion:0']['dropped']}` rows with `energy_kcal`")
    done = steps["complete_cases"]
    if done["dropped"]:
        assert f"`{done['n']:,}` of `{done['n'] + done['dropped']:,}` rows remain" in said["set_missing"]
    else:  # nothing dropped: said plainly, not "580 of 580 rows remain"
        assert f"no row is missing any predictor, so all `{done['n']:,}` rows remain" in said["set_missing"]
    # The split names no count: the analysis count moves with later answers (the banner has it).
    assert "of the rows with `hba1c` recorded" in said["set_split"]
    assert "keeping each `participant_id`'s rows together" in said["set_split"]
    assert all(not voice.machinery(s) for s in said.values()), said
