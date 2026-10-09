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

import numpy as np
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


def test_a_refused_revert_over_http_never_offers_the_answer_already_recorded(client, tmp_path):
    """The verifier's revert, through the server, on the renal table: complete cases keep 10 rows
    and are recorded, then a fill, then `sbp`'s code 999 as blank (accepted: the fill keeps every
    row). Reverting the fill brings complete cases back over the repaired values, 4 rows, and is
    refused (409). None of its exits is the fill already recorded (taking it changed nothing), and
    each one the record accepts."""
    from turbotab.core.tests.test_row_floor import renal

    pid = renal_project(client, tmp_path, renal())
    assert post(client, pid, {"kind": "set_missing", "strategy": "complete_case"}).status_code == 200
    fill = {"kind": "set_missing", "strategy": "impute"}
    assert post(client, pid, fill).status_code == 200
    filled = client.get(f"/api/projects/{pid}").json()["decisions"][-1]
    wait_for(client, pid, {"findings": "fresh", "working": "fresh", "cohort": "fresh"}, timeout=180)
    found = {f["id"]: f for f in
             client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]}
    repair = next(o["decision"] for o in found["sentinel_missing__sbp"]["repairs"]
                  if o["key"] == "set_missing")
    assert post(client, pid, repair).status_code == 200
    wait_for(client, pid, {"working": "fresh"}, timeout=180)
    reverted = post(client, pid, {"kind": "revert", "decision_id": filled["id"]})
    assert reverted.status_code == 409, reverted.text[:600]
    error = reverted.json()["error"]
    assert error["code"] == "too_few_rows"
    ways = [e["decision"] for e in error["exits"] if e["decision"] is not None]
    assert ways, error["exits"]
    recorded = {k: v for k, v in filled["decision"].items() if v not in (None, [], False)}
    for way in ways:
        assert {k: v for k, v in way.items() if v not in (None, [], False)} != recorded, way
        previewed = client.post(f"/api/projects/{pid}/preview", json=way)
        assert previewed.status_code == 200, previewed.text[:600]


def test_combining_records_into_too_few_units_is_refused_when_it_is_recorded(client, tmp_path):
    """The integrator's residue: an answer that changes the working table's rows was counted only
    when the cohort and design stages ran. Four people seen five times each: 20 records, and
    combining each person's records into one row leaves 4 rows, where the design needs 6. The
    aggregation answer was accepted (200) and the cohort stage then failed. Its preview and its
    recording are refused now (409), counted on the table the working stage would build, with
    analyzing each record as its own row among the exits, which the record accepts."""
    from turbotab.server.tests.conftest import answer_settled, prepare

    rng = np.random.default_rng(3)
    people = np.repeat([f"P{i:02d}" for i in range(4)], 5)
    frame = pd.DataFrame({
        "participant_id": people,
        "visit_date": np.tile(pd.date_range("2024-01-01", periods=5, freq="30D").strftime("%Y-%m-%d"), 4),
        "sbp_mm": np.round(rng.normal(130, 12, 20), 1),
        "ldl_mmol": np.round(rng.normal(3.2, 0.6, 20), 2),
        "glucose_mmol": np.round(rng.normal(5.6, 0.7, 20), 2),
    })
    assert frame.groupby("participant_id").ngroups == 4 and len(frame) == 20  # the reference
    path = tmp_path / "visits.csv"
    frame.to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "glucose_mmol"},
              {"kind": "set_purpose", "purpose": "prediction"},
              {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"},
              {"kind": "set_repeat_kind", "repeat_kind": "repeats"},
              {"kind": "set_unit", "unit": "unit"}):
        prepare(client, pid, d)
        assert answer_settled(client, pid, None, d).status_code == 200, d
    combine = {"kind": "set_aggregation", "method": "mean", "outcome": "mean"}
    prepare(client, pid, combine)
    n_records = len(client.get(f"/api/projects/{pid}").json()["decisions"])
    previewed = client.post(f"/api/projects/{pid}/preview", json=combine)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, combine)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    assert error["message"] == (
        "Recorded, this answer would leave 4 rows, one per unit, where 20 rows are analyzed now, "
        "and the design needs at least 6. Analyze each record as its own row (the unit question), "
        "or keep the answers as they are.")
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == n_records
    assert [e["label"] for e in error["exits"]] == ["Analyze each record as its own row",
                                                    "Keep the answers as they are"]
    apart = error["exits"][0]["decision"]
    assert apart == {"kind": "set_unit", "unit": "row"}
    assert post(client, pid, apart).status_code == 200


def test_leaving_out_reference_rows_that_leave_too_few_is_refused_when_it_is_recorded(client,
                                                                                      tmp_path):
    """The other row-changing answer: a repair that leaves the pooled QC rows out of the working
    table (WP18). Five participants beside eight pooled QCs: the outcome `Class` is recorded in
    all 13 rows, and without the QCs 5 are left, where the design needs 6. The exclusion was
    accepted and the cohort stage then failed; it is refused now, counted on the table the working
    stage would build."""
    from turbotab.server.tests.conftest import answer_settled, prepare

    df = pd.read_csv(METABOLOMICS)
    qc = df[df["sample_type"].eq("pooled_qc")].head(8)
    people = df[~df["sample_type"].eq("pooled_qc")].head(5)
    table = pd.concat([people, qc]).reset_index(drop=True)
    cls = np.where(table["sample_type"].eq("pooled_qc"), "QC",
                   np.array(["Case", "Control", "Case", "Control", "Case"] + [""] * 8))
    table = table.drop(columns=["sample_type", "bmi", "responder"], errors="ignore")
    table.insert(1, "Class", cls)
    assert (table["Class"] != "QC").sum() == 5 and (table["Class"] == "QC").sum() == 8  # reference
    path = tmp_path / "met_few.csv"
    table.to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    for d in ({"kind": "set_lens", "lenses": ["metabolomics"]},
              {"kind": "set_target", "column": "Class"}):
        prepare(client, pid, d)
        assert answer_settled(client, pid, None, d).status_code == 200, d
    view = wait_for(client, pid, {"findings": "fresh", "cohort": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["n_final"] == 13
    findings = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    exclude = next(o["decision"] for f in findings for o in f.get("repairs") or []
                   if o["key"] == "exclude_rows")
    assert exclude["params"]["levels"] == ["QC"]
    previewed = client.post(f"/api/projects/{pid}/preview", json=exclude)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, exclude)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    assert error["message"] == (
        "Recorded, this answer would leave 5 rows once the reference rows have left the table, "
        "where 13 rows are analyzed now, and the design needs at least 6. “`Class` is not `QC`” "
        "would remove 8 rows. Keep the reference rows in the table.")
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == len(view["decisions"])
    assert error["exits"][-1] == {"label": "Keep the answers as they are", "decision": None}


def test_a_repair_on_combined_rows_is_counted_on_the_units_it_leaves(client, tmp_path):
    """The same gap on a combined table: once each person's records are one row, a repair that
    blanks values was counted on the table as it stood (``_rewritten`` cannot map a combined
    table's rows), so the cohort stage failed after it. Eight people seen three times each, `sbp`
    coded 999 in every record of three of them: treating 999 as missing under complete cases
    leaves 5 people (pandas, below), and is refused now, counted on the table the working stage
    would build."""
    from turbotab.server.tests.conftest import answer_settled, prepare

    rng = np.random.default_rng(5)
    people = np.repeat([f"P{i:02d}" for i in range(8)], 3)
    frame = pd.DataFrame({
        "participant_id": people,
        "visit_date": np.tile(pd.date_range("2024-01-01", periods=3, freq="30D").strftime("%Y-%m-%d"), 8),
        "sbp_mm": np.round(rng.normal(130, 12, 24), 1),
        "ldl_mmol": np.round(rng.normal(3.2, 0.6, 24), 2),
        "glucose_mmol": np.round(rng.normal(5.6, 0.7, 24), 2),
    })
    frame.loc[frame["participant_id"].isin(["P01", "P04", "P06"]), "sbp_mm"] = 999.0
    units = frame.assign(sbp_mm=frame["sbp_mm"].mask(frame["sbp_mm"] == 999)).groupby(
        "participant_id")[["sbp_mm", "ldl_mmol"]].mean()
    assert int(units.notna().all(axis=1).sum()) == 5  # the reference
    path = tmp_path / "coded.csv"
    frame.to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "glucose_mmol"},
              {"kind": "set_purpose", "purpose": "prediction"},
              {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"},
              {"kind": "set_repeat_kind", "repeat_kind": "repeats"},
              {"kind": "set_unit", "unit": "unit"},
              {"kind": "set_aggregation", "method": "mean", "outcome": "mean"},
              {"kind": "set_missing", "strategy": "complete_case"}):
        prepare(client, pid, d)
        assert answer_settled(client, pid, None, d).status_code == 200, d
    wait_for(client, pid, {"findings": "fresh", "working": "fresh", "cohort": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["n_final"] == 8
    found = {f["id"]: f for f in
             client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]}
    repair = next(o["decision"] for o in found["sentinel_missing__sbp_mm"]["repairs"]
                  if o["key"] == "set_missing")
    previewed = client.post(f"/api/projects/{pid}/preview", json=repair)
    assert previewed.status_code == 409, previewed.text[:600]
    refused = post(client, pid, repair)
    assert refused.status_code == 409, refused.text[:600]
    error = refused.json()["error"]
    assert error["code"] == "too_few_rows"
    assert error["message"] == (
        "Recorded, this answer would leave 5 of the 8 rows analyzed now, and the design needs at "
        "least 6. Complete cases would remove 3 rows: each of them is blank in `sbp_mm`. Fill the "
        "blanks instead of complete cases (the missing-values question).")
    fill = error["exits"][0]["decision"]
    assert fill["kind"] == "set_missing" and fill["strategy"] == "impute"
    assert post(client, pid, fill).status_code == 200
    assert post(client, pid, repair).status_code == 200


def statin_table() -> pd.DataFrame:
    """The verifier's 60-row table: `statin_use` (yes/no) blank on 70% of rows, and complete cases
    on `age_y` and `bmi` keep 4 rows."""
    rng = np.random.default_rng(5)
    n = 60
    age = rng.integers(30, 80, n).astype(float)
    bmi = rng.normal(27.0, 4.0, n).round(1)
    age[:28] = np.nan
    bmi[28:56] = np.nan
    statin = np.where(rng.random(n) < 0.5, "yes", "no").astype(object)
    statin[rng.permutation(n)[:42]] = None
    sbp = (118 + 0.3 * np.nan_to_num(bmi, nan=27.0) + rng.normal(0, 9, n)).round(0)
    return pd.DataFrame({"pt_code": [f"P{i:03d}" for i in range(n)], "age_y": age, "bmi": bmi,
                         "statin_use": statin, "sbp": sbp})


def test_the_not_asked_cautions_exits_are_each_accepted(client, tmp_path):
    """The verifier's case: previewing a single fill offered the "not asked" caution's "Leave it
    out first" (complete cases with `statin_use` left out), which keeps 4 rows and was itself
    refused, 409 too_few_rows, at preview and at recording. Every exit the caution offers now
    previews and records; leaving the column out is still offered, with the other blanks filled."""
    from turbotab.server.tests.conftest import answer_settled, prepare

    path = tmp_path / "statins.csv"
    statin_table().to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "sbp"},
              {"kind": "set_purpose", "purpose": "prediction"}):
        prepare(client, pid, d)
        assert answer_settled(client, pid, None, d).status_code == 200
    fill = {"kind": "set_missing", "strategy": "impute"}
    prepare(client, pid, fill)
    wait_for(client, pid, {"proposals": "fresh"}, timeout=180)
    reading = client.get(f"/api/projects/{pid}/stages/proposals").json()["artifact"]["missing"]
    assert [e["column"] for e in reading["columns"] if e["likely_not_asked"]] == ["statin_use"]

    previewed = client.post(f"/api/projects/{pid}/preview", json=fill)
    assert previewed.status_code == 200, previewed.text[:600]
    caution = previewed.json()["caution"]
    assert "`statin_use`" in caution["text"] and "not asked" in caution["text"]
    exits = [e["decision"] for e in caution["exits"]]
    assert any(d["drop_columns"] == ["statin_use"] for d in exits)
    for d in exits:
        taken = client.post(f"/api/projects/{pid}/preview", json=d)
        assert taken.status_code == 200, (d, taken.text[:600])
    leave = next(d for d in exits if d["drop_columns"] == ["statin_use"])
    assert post(client, pid, leave).status_code == 200
