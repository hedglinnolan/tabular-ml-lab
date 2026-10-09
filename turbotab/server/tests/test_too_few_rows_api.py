"""Too few rows to analyze, over HTTP (``turbotab.core.row_floor``; the metabolomics zero-row crash).

The no-dead-end drive's path through the real server: ``metabolomics_untargeted.csv`` under
prediction, the roles recorded as proposed (the features ride along unconfirmed), complete cases,
then the features confirmed at the models question. The confirmation left the cohort 0 rows and
the design failed on scikit-learn's "Found array with 0 sample(s)"; now its preview and its
recording are refused (409) with the blanks named and a fill as the way back, the fill is accepted,
the confirmation goes through after it, and no stage fails.
"""
from __future__ import annotations

import time

import pandas as pd

from turbotab.server.tests.conftest import SAMPLES, open_by_path, wait_for

METABOLOMICS = SAMPLES / "metabolomics_untargeted.csv"


def post(client, pid, body):
    return client.post(f"/api/projects/{pid}/decisions", json=body)


def answer(client, pid, key, body, timeout=180.0):
    """Wait until ``key`` is the open question, then record ``body``."""
    end = time.monotonic() + timeout
    while True:
        steps = client.get(f"/api/projects/{pid}").json()["interview"]
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if first is not None and first["key"] == key and first["status"] == "open":
            break
        assert time.monotonic() < end, f"{key} never opened: {first}"
        time.sleep(0.05)
    response = post(client, pid, body)
    assert response.status_code == 200, response.text[:900]


def test_confirming_the_features_under_complete_cases_is_refused_with_a_fill(client):
    frame = pd.read_csv(METABOLOMICS)
    measured = int(frame["responder"].notna().sum())
    pid = open_by_path(client, METABOLOMICS)
    wait_for(client, pid, {"ingest": "fresh"})
    answer(client, pid, "lens", {"kind": "set_lens", "lenses": ["metabolomics"]})
    answer(client, pid, "orientation", {"kind": "set_orientation", "orientation": "sample_major"})
    answer(client, pid, "target", {"kind": "set_target", "column": "responder"})
    answer(client, pid, "event", {"kind": "set_event", "column": "responder", "level": "1"})
    answer(client, pid, "purpose", {"kind": "set_purpose", "purpose": "prediction"})
    answer(client, pid, "grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    wait_for(client, pid, {"roles": "fresh"}, timeout=180)
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    answer(client, pid, "roles", {"kind": "set_roles", "roles": {
        c["column"]: c["proposed"] for c in roles["columns"] if c["proposed"]}})
    answer(client, pid, "exclusions", {"kind": "set_exclusions", "rules": []})
    # Accepted: the features ride along unconfirmed, so their blanks drop no row yet (§14.1).
    answer(client, pid, "missing", {"kind": "set_missing", "strategy": "complete_case"})
    answer(client, pid, "split", {"kind": "set_split", "holdout": 0.2})
    wait_for(client, pid, {"shelf": "fresh"}, timeout=180)
    asked = post(client, pid, {"kind": "select_models", "models": ["elastic_net"]})
    assert asked.status_code == 409 and asked.json()["error"]["code"] == "reading_unsettled"
    confirm = next(e["decision"] for e in asked.json()["error"]["exits"]
                   if (e["decision"] or {}).get("kind") == "confirm_readings")
    assert any(i["column"].startswith("mz_") for i in confirm["items"])

    previewed = client.post(f"/api/projects/{pid}/preview", json=confirm)
    assert previewed.status_code == 409, previewed.text[:600]
    assert previewed.json()["error"]["code"] == "too_few_rows"
    refused = post(client, pid, confirm)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    # Under prediction the check reads the rows outside the held-out ones, as a preview does.
    held = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]["n_holdout"]
    assert error["message"].startswith(
        f"Recorded, this answer would leave none of the {measured - held} rows outside the held-out "
        f"ones, and the design needs at least 6. Complete cases would remove {measured - held} rows: "
        f"each of them is blank in at least one of the ")
    assert error["message"].endswith(
        "Fill the blanks instead of complete cases (the missing-values question).")
    fill = error["exits"][0]
    assert fill["label"].startswith("Fill the blanks instead: single fill in each training fold")
    assert fill["decision"]["kind"] == "set_missing" and fill["decision"]["strategy"] == "impute"
    assert error["exits"][-1] == {"label": "Keep the answers as they are", "decision": None}

    assert post(client, pid, fill["decision"]).status_code == 200
    confirmed = post(client, pid, confirm)
    assert confirmed.status_code == 200, confirmed.text[:600]
    view = wait_for(client, pid, {"cohort": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["n_final"] == measured
    assert not [s for s, st in view["stages"].items() if st["status"] == "error"]


# ── the verifier's renal tables, through the server ──────────────────────────


def renal_project(client, tmp_path, frame, purpose="prediction"):
    """``frame`` opened as a project, answered up to the missing-values question with complete
    cases (the labs' roles confirmed, as a user confirming each one would)."""
    from turbotab.server.tests.conftest import answer_settled, prepare

    path = tmp_path / "renal.csv"
    frame.assign(pt_code=[f"P{i:04d}" for i in range(len(frame))]).to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "egfr_decline"},
              {"kind": "set_purpose", "purpose": purpose}):
        prepare(client, pid, d)
        assert answer_settled(client, pid, None, d).status_code == 200
    prepare(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    return pid


def test_a_repair_that_blanks_codes_under_complete_cases_is_refused_over_http(client, tmp_path):
    """The verifier's renal_s.csv: complete cases keep 10 rows, `sbp` holds 999 in six of them.
    Treating 999 as missing previewed its 9 changed cells and was recorded (200), and the cohort
    failed with 4 rows. Its preview and its recording are refused now (409), counted on the values
    the repair would leave; the fill is accepted, then the repair, and no stage fails."""
    from turbotab.core.tests.test_row_floor import renal

    pid = renal_project(client, tmp_path, renal())
    assert post(client, pid, {"kind": "set_missing", "strategy": "complete_case"}).status_code == 200
    view = wait_for(client, pid, {"findings": "fresh", "working": "fresh", "cohort": "fresh"},
                    timeout=180)
    assert client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["n_final"] == 10
    found = {f["id"]: f for f in
             client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]}
    repair = next(o["decision"] for o in found["sentinel_missing__sbp"]["repairs"]
                  if o["key"] == "set_missing")
    n_records = len(view["decisions"])

    previewed = client.post(f"/api/projects/{pid}/preview", json=repair)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, repair)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    assert error["message"] == (
        "Recorded, this answer would leave 4 of the 10 rows analyzed now, and the design needs at "
        "least 6. Complete cases would remove 6 rows: each of them is blank in `sbp`. Fill the "
        "blanks instead of complete cases (the missing-values question).")
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == n_records

    fill = error["exits"][0]["decision"]
    assert fill["kind"] == "set_missing" and fill["strategy"] == "impute"
    assert post(client, pid, fill).status_code == 200
    assert post(client, pid, repair).status_code == 200
    view = wait_for(client, pid, {"working": "fresh", "cohort": "fresh"}, timeout=180)
    assert not [s for s, st in view["stages"].items() if st["status"] == "error"]


def test_complete_cases_that_leave_one_outcome_value_are_refused_over_http(client, tmp_path):
    """The verifier's renal_1class.csv: the 12 complete rows all have `egfr_decline` 0. Recorded,
    the fit crashed with "IndexError: list index out of range" and the seal said "List index out
    of range.". Complete cases are refused now (preview and record, 409), naming the outcome left
    one value, with a fill as the way back."""
    from turbotab.core.tests.test_row_floor import renal

    frame = renal(complete=12)
    frame.loc[:11, "egfr_decline"] = 0
    pid = renal_project(client, tmp_path, frame)
    answer = {"kind": "set_missing", "strategy": "complete_case"}
    previewed = client.post(f"/api/projects/{pid}/preview", json=answer)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, answer)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "one_outcome_value"
    assert error["message"].startswith(
        "Recorded, this answer would leave 12 rows, and `egfr_decline` is `0` in every one of them: "
        "the models need rows with another value of the outcome to learn from. Complete cases "
        "would remove 108 rows: ")
    fill = error["exits"][0]["decision"]
    assert fill["strategy"] == "impute" and post(client, pid, fill).status_code == 200


def test_a_sensitivity_analysis_that_keeps_no_row_is_refused_over_http(client, tmp_path):
    """The verifier's survey case: a sensitivity analysis whose rule keeps no row was accepted
    (200), and the sensitivity stage then showed scikit-learn's "Found array with 0 sample(s)".
    Its preview and its recording are refused now (409); leaving the analysis out, or keeping
    one that keeps enough rows, is accepted."""
    from turbotab.core.tests.test_row_floor import renal

    pid = renal_project(client, tmp_path, renal(complete=40))
    assert post(client, pid, {"kind": "set_missing", "strategy": "complete_case"}).status_code == 200
    nobody = {"label": "centenarians", "rules": [
        {"column": "age_years", "low": 100, "high": 120, "reason": "the oldest"}]}
    older = {"label": "over 50", "rules": [
        {"column": "age_years", "low": 50, "reason": "older adults"}]}
    answer = {"kind": "set_sensitivity", "analyses": [older, nobody]}
    previewed = client.post(f"/api/projects/{pid}/preview", json=answer)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, answer)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    assert error["message"] == (
        "Recorded, this answer would leave the sensitivity analysis “centenarians” no rows, "
        "against 40 in the primary analysis, and its fit needs at least 6. “`age_years` within "
        "`100`–`120`” would remove 120 rows. Drop or widen the rule that removes them (the "
        "sensitivity question).")  # the flow's own order: the rule, then complete cases
    left = next(e["decision"] for e in error["exits"]
                if e["label"] == "Leave out the analysis “centenarians”")
    assert [a["label"] for a in left["analyses"]] == ["over 50"]
    assert post(client, pid, left).status_code == 200
