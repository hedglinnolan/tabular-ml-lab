"""The export's routes on a project that has only been opened (Tier B; the journeys are
``core/tests/acceptance/test_export.py``)."""
from __future__ import annotations

from turbotab.server.tests.conftest import open_by_path, wait_for


def test_a_fresh_project_is_refused_its_bundle_and_shown_what_the_checklist_waits_for(client):
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})
    r = client.get(f"/api/projects/{pid}/export")
    assert r.status_code == 409
    error = r.json()["error"]
    assert error["code"] == "unanswered_questions"
    assert error["message"].startswith("The lens question")
    assert error["exits"][0] == {"label": "Answer the lens question", "decision": None}
    c = client.get(f"/api/projects/{pid}/checklist")
    assert c.status_code == 200
    report = c.json()
    assert report["checklist"] == "STROBE-nut" and report["counts"]["items"] == 58
    # nothing recorded answers an item yet; the provenance record a bundle carries answers part of
    # STROBE-nut 22.2 (the data and the tools are the author's)
    assert report["counts"]["unanswered"] == 57 and report["waiting"]
    assert [i["id"] for i in report["items"] if i["status"] != "unanswered"] == ["nut-22.2"]
    assert report["waiting"][0].startswith("The lens question")


def test_an_unknown_project_has_no_bundle_and_no_checklist(client):
    for path in ("export", "checklist"):
        r = client.get(f"/api/projects/p0000000000/{path}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_project"
