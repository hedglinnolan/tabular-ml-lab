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
