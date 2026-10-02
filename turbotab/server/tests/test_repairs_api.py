"""Findings with preview-before-apply, deferral and memory, over HTTP (M2_CONTRACT §4; Tier B).

Until the ``working`` stage exists (the sequence agent's), "the working table changes" is checked
through the expressions it will select: ``repairs.column_expressions`` over the served artifact and
the recorded dispositions, evaluated on the project's own table.
"""
from __future__ import annotations

import pytest

from turbotab.core import repairs
from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, words
from turbotab.core.tests.stage_harness import NHANES
from turbotab.server import schemas
from turbotab.server.tests.conftest import SAMPLES, open_by_path, prepare, wait_for


def decide(client, pid, decision, status=200):
    if status == 200:
        prepare(client, pid, decision)  # the questions before it, answered as usual (M2 §12.2)
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == status, response.text
    return response.json()


def findings(client, pid):
    body = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]
    schemas.FindingsArtifact.model_validate(body)
    return {f["id"]: f for f in body["findings"]}, body


@pytest.fixture(scope="module")
def survey(client):
    pid = open_by_path(client, SAMPLES / "survey_sentinels.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["survey"]})
    decide(client, pid, {"kind": "set_target", "column": "sought_support"})
    wait_for(client, pid, {"findings": "fresh", "target_info": "fresh"})
    return pid


def test_apply_previews_then_records_and_the_working_values_change(client, survey):
    pid = survey
    found, _ = findings(client, pid)
    item = found["sentinel_missing__item_14"]
    option = item["repairs"][0]
    assert option["label"] == "Treat as missing" and option["effect"] == "values"

    n_before = len(client.get(f"/api/projects/{pid}").json()["decisions"])
    response = client.post(f"/api/projects/{pid}/preview", json=option["decision"])
    assert response.status_code == 200, response.text
    body = schemas.PreviewResult.model_validate(response.json())
    assert [v.kind for v in body.views] == ["table_focus", "distribution"]
    for view in body.views:
        assert words(view.title) <= TITLE_WORDS and words(view.caption) <= CAPTION_WORDS
    focus, dist = body.views
    assert all(column == "item_14" for _, column in focus.changed) and focus.changed
    assert all(row.before["item_14"] == 9 and row.after["item_14"] is None for row in focus.rows)
    assert [(m.value, m.label) for m in dist.marks] == [(9.0, "code 9")]
    assert dist.after.n_missing - dist.before.n_missing == 33
    assert body.basis == "Every one of the 300 rows, as the finding read them."
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == n_before  # nothing recorded

    # The client may name the option alone: the server fills its params from the finding.
    view = decide(client, pid, {"kind": "apply_repair", "finding_id": item["id"], "option": "set_missing"})
    record = view["decisions"][-1]
    assert record["decision"]["params"] == {"values": {"item_14": [9.0]}}
    assert view["state"]["findings"][item["id"]]["action"] == "applied"
    found, artifact = findings(client, pid)
    assert found[item["id"]]["disposition"]["option"] == "set_missing"
    assert found[item["id"]]["answered_by"] == record["id"]

    state = schemas.ProjectState.model_validate(view["state"])
    expressions = repairs.column_expressions(artifact, state)
    assert set(expressions) == {"item_14"}
    data = client.app.state.service.workspace.data_path(pid)
    after = repairs.evaluate(data, ["item_14"], expressions)
    assert not (after["item_14"] == 9).any() and int(after["item_14"].isna().sum()) == 33

    # The bulk repair covers the four other items, and folds the per-item findings it answers.
    pack = found["pack::survey::sentinel_codes"]
    view = decide(client, pid, pack["repairs"][0]["decision"])
    bulk = view["decisions"][-1]["id"]
    found, _ = findings(client, pid)
    for column in ("item_05", "item_09", "item_22", "item_33"):
        assert found[f"sentinel_missing__{column}"]["answered_by"] == bulk
    assert found[item["id"]]["answered_by"] == record["id"]  # its own record still answers it

    # Reverting restores what was there before: the finding is open again.
    view = decide(client, pid, {"kind": "revert", "decision_id": record["id"]})
    found, _ = findings(client, pid)
    assert found[item["id"]]["disposition"] is None
    assert found[item["id"]]["answered_by"] == bulk  # the bulk repair still does its work


def test_defer_resurfaces_in_its_question_and_dismiss_folds(client, survey):
    pid = survey
    found, _ = findings(client, pid)
    sex = found["binary_text__sex"]
    assert [o["decision"]["params"]["one"] for o in sex["repairs"]] == ["f", "m"]
    view = decide(client, pid, {"kind": "defer_finding", "finding_id": sex["id"], "to": "roles"})
    steps = {s["key"]: s for s in view["interview"]}
    assert steps["roles"]["deferred_findings"] == [sex["id"]]
    found, _ = findings(client, pid)
    assert found[sex["id"]]["disposition"]["action"] == "deferred"
    assert found[sex["id"]]["answered_by"] == view["decisions"][-1]["id"]

    wide = next(f for f in found.values() if f["id"].startswith("wide_repeated"))
    view = decide(client, pid, {"kind": "dismiss_finding", "finding_id": wide["id"],
                                "reason": "a questionnaire is wide by design"})
    found, _ = findings(client, pid)
    assert found[wide["id"]]["disposition"]["action"] == "dismissed"
    assert found[wide["id"]]["answered_by"] == view["decisions"][-1]["id"]


def test_a_dismissal_recomputes_nothing_and_an_applied_repair_does(client, survey):
    pid = survey
    decide(client, pid, {"kind": "set_roles", "roles": {"age": "covariate", "item_01": "exposure"}})
    view = wait_for(client, pid, {"cohort": "fresh"})
    key = view["stages"]["cohort"]["key"]
    found, _ = findings(client, pid)
    ident = next(f for f in found.values() if f["id"].startswith("voice::identifier"))
    view = decide(client, pid, {"kind": "dismiss_finding", "finding_id": ident["id"]})
    assert view["stages"]["cohort"]["key"] == key and view["stages"]["cohort"]["status"] == "fresh"
    view = decide(client, pid, found["sentinel_missing__item_33"]["repairs"][0]["decision"])
    assert view["stages"]["cohort"]["key"] != key


def test_refused_repairs_say_why_and_offer_the_ones_there_are(client, survey):
    pid = survey
    found, _ = findings(client, pid)
    item = found["sentinel_missing__item_22"]
    bad = {"kind": "apply_repair", "finding_id": item["id"], "option": "set_missing",
           "params": {"values": {"item_22": [3.0]}}}  # 3 is an answer, not a code the data showed
    for path in ("decisions", "preview"):
        response = client.post(f"/api/projects/{pid}/{path}", json=bad)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "repair_params"
        assert error["exits"][0]["decision"]["params"] == {"values": {"item_22": [8.0]}}
    response = client.post(f"/api/projects/{pid}/decisions",
                           json={"kind": "defer_finding", "finding_id": item["id"], "to": "nowhere"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "unknown_question"


@pytest.mark.skipif(not NHANES.is_file(), reason="the real NHANES export is not here")
def test_nhanes_can_keep_meds_hbp_blanks_as_a_missing_level(client):
    """M2_CONTRACT §9: on NHANES, meds_hbp can be kept as a Missing category, offered beside
    imputing it (which would put every unasked respondent on blood-pressure medication)."""
    pid = open_by_path(client, NHANES)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary", "clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "glucose"})
    wait_for(client, pid, {"roles": "fresh", "proposals": "fresh"})
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    wait_for(client, pid, {"cohort": "fresh"})
    prepare(client, pid, {"kind": "set_missing"})  # the eligibility question comes first
    # SEQN is unique on every row, so the grain was stated, not asked (M2_CONTRACT §10)
    grain = next(s for s in client.get(f"/api/projects/{pid}").json()["interview"] if s["key"] == "grain")
    assert grain["status"] == "skipped" and "`SEQN` appears once" in grain["reason"]

    impute = client.post(f"/api/projects/{pid}/preview", json={"kind": "set_missing", "strategy": "impute"}).json()
    level = {"kind": "set_missing", "strategy": "impute", "categorical": "missing_category",
             "drop_columns": [], "indicators": False, "m": 20, "below_detection": None,
             "censored_columns": [], "acknowledged": False, "reason": None}
    assert level in [e["decision"] for e in impute["caution"]["exits"]]
    kept = client.post(f"/api/projects/{pid}/preview", json=level).json()
    assert kept["caution"] is None
    assert [v["title"] for v in kept["views"]][:2] == ["Rows kept when values are missing",
                                                       "Blanks kept as their own level"]
    assert all(row["after"]["meds_hbp"] in ("Missing", True, False) for row in kept["views"][1]["rows"])
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case", "categorical": "missing_category"})
    view = wait_for(client, pid, {"cohort": "fresh"})
    cohort = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]
    assert cohort["steps"][-1] == {**cohort["steps"][-1], "key": "complete_cases", "dropped": 0}
    assert view["state"]["missing"]["categorical"] == "missing_category"


@pytest.fixture(scope="module")
def labs(client):
    pid = open_by_path(client, SAMPLES / "clinical_labs.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "readmitted"})
    decide(client, pid, {"kind": "set_roles", "roles": {"age": "covariate", "sbp": "covariate",
                                                       "dbp": "covariate"}})
    wait_for(client, pid, {"findings": "fresh", "cohort": "fresh"})
    return pid


def test_the_three_routes_for_impossible_values_preview_their_own_pictures(client, labs):
    pid = labs
    found, _ = findings(client, pid)
    f = found["pack::clinical::impossible_vs_extreme"]
    by_key = {o["key"]: o["decision"] for o in f["repairs"]}

    def preview(decision):
        response = client.post(f"/api/projects/{pid}/preview", json=decision)
        assert response.status_code == 200, response.text
        body = schemas.PreviewResult.model_validate(response.json())
        for view in body.views:
            assert words(view.title) <= TITLE_WORDS and words(view.caption) <= CAPTION_WORDS, view.caption
        return body

    blank = preview(by_key["set_missing"])
    assert [v.kind for v in blank.views] == ["table_focus", "distribution"]
    assert [m.label for m in blank.views[1].marks] == ["floor 40 mmHg", "ceiling 300 mmHg"]
    rows = preview(by_key["exclude_rows"])
    assert [v.kind for v in rows.views] == ["row_flow", "distribution"]
    step = rows.views[0].after[-1]
    assert step.key == "repair:0" and step.dropped == 4
    gone = preview(by_key["unusable"])
    assert [v.kind for v in gone.views] == ["lineage"]
    matrix = {n.column for n in gone.views[0].after.nodes if n.lane == "matrix"}
    assert "sbp" not in matrix and "age" in matrix

    view = decide(client, pid, by_key["exclude_rows"])
    view = wait_for(client, pid, {"cohort": "fresh"})
    cohort = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]
    assert [s["key"] for s in cohort["steps"]][-1] == "repair:0"
    assert cohort["steps"][-1]["dropped"] == 4 and cohort["n_final"] == 284
