"""Tier B over HTTP on the real NHANES export (M1_CONTRACT §12): every finding's evidence, the
missing-values offer and its preview, nested nutrients on the roles, and the storyboard and marks
in previews. Skipped where the export is not on the machine."""
from __future__ import annotations

import pytest

from turbotab.core.consequences import CAPTION_WORDS, FRAME_WORDS, TITLE_WORDS, words
from turbotab.core.tests.stage_harness import NHANES
from turbotab.server import schemas
from turbotab.server.tests.conftest import open_by_path, wait_for

pytestmark = pytest.mark.skipif(not NHANES.is_file(), reason="the NHANES export is not on this machine")

# The view a finding's evidence leads with, by family.
LEADS = {
    "pack::dietary::energy_adjustment": "relationship",
    "pack::dietary::implausible_intake": "distribution",
    "voice::flag": "table_focus",
    "voice::identifier": "table_focus",
    "binary_text": "table_focus",
    "voice::pooled_cycles": "table_focus",
}


def decide(client, pid, decision):
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == 200, response.text
    return response.json()


def valid(body: dict) -> schemas.PreviewResult:
    result = schemas.PreviewResult.model_validate(body)
    for view in result.views:
        assert words(view.title) <= TITLE_WORDS, view.title
        assert words(view.caption) <= CAPTION_WORDS, view.caption
        for frame in view.story:
            assert words(frame.label) <= FRAME_WORDS, frame.label
    return result


@pytest.fixture(scope="module")
def nhanes(client):
    pid = open_by_path(client, NHANES)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "glucose"})
    wait_for(client, pid, {"findings": "fresh", "roles": "fresh", "cohort": "fresh",
                           "proposals": "fresh"}, timeout=120)
    return pid


def test_every_finding_on_nhanes_shows_its_evidence(client, nhanes):
    findings = client.get(f"/api/projects/{nhanes}/stages/findings").json()["artifact"]["findings"]
    assert len(findings) >= 10
    seen = set()
    for finding in findings:
        response = client.get(f"/api/projects/{nhanes}/findings/{finding['id']}/evidence")
        assert response.status_code == 200, (finding["id"], response.text)
        result = valid(response.json())
        family = result.kind
        seen.add(family)
        if family == "voice::survey_design_absent":  # about columns the table lacks
            assert result.views == [] and result.note
            continue
        assert 1 <= len(result.views) <= 3, finding["id"]
        lead = LEADS.get(family)
        if lead:
            assert result.views[0].kind == lead, (finding["id"], result.views[0].kind)
        assert "rows" in result.basis
    assert {"pack::dietary::energy_adjustment", "pack::dietary::implausible_intake",
            "voice::flag", "voice::identifier"} <= seen


def test_the_evidence_says_what_the_finding_says(client, nhanes):
    get = lambda fid: client.get(f"/api/projects/{nhanes}/findings/{fid}/evidence").json()  # noqa: E731
    energy = get("pack::dietary::energy_adjustment")["views"][0]
    assert energy["x_label"] == "kcal" and energy["y_label_before"] == "fat_total"
    assert energy["points_before"] == energy["points_after"] and 0.8 < energy["r_before"] < 0.95
    intake = get("pack::dietary::implausible_intake")["views"][0]
    assert [(m["value"], m["label"]) for m in intake["marks"]] == [(500.0, "500 kcal"), (5000.0, "5,000 kcal")]
    assert intake["caption"].startswith("`501` of `21,849` rows")
    flag = get("voice::flag__imputed_bmi")["views"][0]
    assert flag["columns_before"] == ["imputed_bmi", "bmi"] and "`306` rows" in flag["caption"]
    assert all(column == "bmi" for _, column in flag["changed"]) and flag["changed"]
    blank = get("binary_text__meds_hbp")["views"][0]
    assert blank["changed"] and all(r["before"]["meds_hbp"] is None for r in blank["rows"])


def test_an_unknown_finding_is_a_404(client, nhanes):
    response = client.get(f"/api/projects/{nhanes}/findings/no_such_thing/evidence")
    assert response.status_code == 404 and response.json()["error"]["code"] == "unknown_finding"


def test_the_roles_and_proposals_read_nested_nutrients_and_blanks_that_mean_not_asked(client, nhanes):
    roles = client.get(f"/api/projects/{nhanes}/stages/roles").json()["artifact"]
    schemas.RolesArtifact.model_validate(roles)
    nested = {c["column"]: c["nested_in"] for c in roles["columns"] if c["nested_in"]}
    assert nested == {"sugar": "carb", "fat_sat": "fat_total", "fat_mon": "fat_total",
                      "fat_poly": "fat_total"}
    proposals = client.get(f"/api/projects/{nhanes}/stages/proposals").json()["artifact"]
    schemas.ProposalsArtifact.model_validate(proposals)
    offer = proposals["missing"]["leave_out"]
    assert sorted(offer["columns"]) == ["meds_chol", "meds_hbp"] and offer["share"] > 0.8


def test_the_missing_question_previews_the_rows_leaving_the_blank_columns_out_saves(client, nhanes):
    roles = client.get(f"/api/projects/{nhanes}/stages/roles").json()["artifact"]
    proposed = {c["column"]: c["proposed"] for c in roles["columns"]}
    assert proposed["meds_hbp"] == "covariate" and proposed["meds_chol"] == "covariate"
    decide(client, nhanes, {"kind": "set_roles", "roles": proposed})
    rule = {"column": "kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    decide(client, nhanes, {"kind": "set_exclusions", "rules": [rule]})
    option = {"kind": "set_missing", "strategy": "complete_case", "drop_columns": ["meds_hbp", "meds_chol"]}
    response = client.post(f"/api/projects/{nhanes}/preview", json=option)
    assert response.status_code == 200, response.text
    flow, saved = valid(response.json()).views
    assert flow.kind == "row_flow" and flow.after[-1].key == "complete_cases"
    assert flow.after[-1].n == 21_348
    assert "`18,405` rows complete cases would drop" in flow.caption
    assert saved.kind == "table_focus" and saved.columns_after == [c for c in saved.columns_before
                                                                    if c not in ("meds_hbp", "meds_chol")]
    view = decide(client, nhanes, option)
    said = view["decisions"][-1]["sentence"]
    assert said.startswith("`meds_hbp` and `meds_chol` were left out of the predictors; then a "
                           "complete-case analysis was applied: no row is missing any other "
                           "predictor, so all `21,348` rows remain")
    assert view["state"]["missing"] == {"strategy": "complete_case", "drop_columns": ["meds_hbp", "meds_chol"]}
    wait_for(client, nhanes, {"cohort": "fresh"})
    cohort = client.get(f"/api/projects/{nhanes}/stages/cohort").json()["artifact"]
    assert cohort["n_final"] == 21_348 and not {"meds_hbp", "meds_chol"} & set(cohort["predictors"])
    bad = {"kind": "set_missing", "strategy": "impute", "drop_columns": ["SEQN"]}
    refused = client.post(f"/api/projects/{nhanes}/decisions", json=bad)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "not_a_predictor"


def test_a_by_sex_rule_marks_each_level(client, nhanes):
    rule = {"column": "kcal", "by": {"column": "gender", "ranges": {"female": [500, 3500], "male": [800, 4200]}},
            "reason": "Willett"}
    body = client.post(f"/api/projects/{nhanes}/preview", json={"kind": "set_exclusions", "rules": [rule]}).json()
    dist = valid(body).views[1]
    assert [(m.value, m.label, m.group) for m in dist.marks] == [
        (500.0, "500 kcal", "female"), (3500.0, "3,500 kcal", "female"),
        (800.0, "800 kcal", "male"), (4200.0, "4,200 kcal", "male")]
