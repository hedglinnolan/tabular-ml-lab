"""PREVIEWS · the new previews through the server's own route (``POST /api/projects/{pid}/preview``).

The other preview tests plan as the server plans (``preview_harness``); this one goes through it,
so the service's wiring is held too: the decision validated and completed before it is previewed (a
refused option previews as its refusal), the log handed to the planner (a preview's state is the
log's fold with the answer; a revert previews the answer it restores), nothing recorded by a
preview, and the recorded answer's stage agreeing with what its preview showed.

The causal lane's cohort (``test_causal_lane.cohort``): the plan declared for ``heavy_user``, the
lane's positivity refused until trimmed.
"""
from __future__ import annotations

import re

import pytest

from turbotab.core.consequences import CAPTION_WORDS, MAX_VIEWS, words

ALL4 = ["no_unmeasured_confounding", "positivity", "consistency", "time_ordering"]


def _preview(drive, body: dict) -> dict:
    before = len(drive.view()["decisions"])
    response = drive.c.post(f"/api/projects/{drive.pid}/preview", json=body)
    assert response.status_code == 200, response.text
    assert len(drive.view()["decisions"]) == before  # a preview records nothing
    found = response.json()
    assert 1 <= len(found["views"]) <= MAX_VIEWS or found["note"]
    for v in found["views"]:
        assert words(v["caption"]) <= CAPTION_WORDS
    return found


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.acceptance.test_causal_lane import _open_cohort, _to_the_plan

    folder = tmp_path_factory.mktemp("previews_server")
    with local_server(folder / "home") as client:
        d = _open_cohort(folder, client)
        _to_the_plan(d, "heavy_user")
        d.answer_plan("heavy_user")
        d.artifact("causal_design")
        yield d


def test_server_a_refused_option_previews_as_its_refusal_and_its_exit_previews(drive):
    """The lane's positivity: an untrimmed TMLE is refused (409, its exits), and the trimming exit
    previews the estimator's own propensity with the rows the trim keeps."""
    body = {"kind": "set_causal", "exposure": "heavy_user", "method": "tmle", "learner": "linear",
            "assumptions": ALL4}
    refused = drive.c.post(f"/api/projects/{drive.pid}/preview", json=body)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "positivity"
    trim = next(e["decision"] for e in refused.json()["error"]["exits"] if e["decision"].get("trim"))
    found = _preview(drive, trim)
    dist, flow = found["views"]
    assert (dist["kind"], flow["kind"]) == ("distribution", "row_flow")
    assert found["basis"] == ("The estimator's own propensity on all 1,500 complete analyzed rows, "
                              "its learner and folds; no outcome read.")
    kept = int(re.search(r"keeps `([\d,]+)` rows", flow["caption"]).group(1).replace(",", ""))
    drive.decide(trim)
    drive.decide({"kind": "select_models", "models": ["linear"]})
    stage = drive.artifact("causal")
    assert stage["n"] == kept  # the rows the preview said the trim keeps are the lane's


def test_server_a_preview_reads_the_log_and_a_revert_previews_the_answer_it_restores(drive):
    """The model sequence's preview runs on the log's fold with the answer; after a second
    declaration, reverting it previews the first one again."""
    first = {"kind": "set_model_sequence", "exposure": "heavy_user", "model_1": ["age"]}
    second = {"kind": "set_model_sequence", "exposure": "heavy_user",
              "model_1": ["age", "smoker"]}
    found = _preview(drive, first)
    assert found["views"][0]["caption"].startswith("Model 1 adjusts for `age`;")
    drive.decide(first)
    drive.decide(second)
    record = drive.view()["decisions"][-1]
    assert record["decision"]["kind"] == "set_model_sequence"
    restored = _preview(drive, {"kind": "revert", "decision_id": record["id"]})
    assert restored["views"][0]["caption"] == found["views"][0]["caption"]
