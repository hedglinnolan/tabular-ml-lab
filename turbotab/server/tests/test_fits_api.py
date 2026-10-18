"""SWARM_ENGINE_PLAN §2 E1 and E2 over HTTP: the record of every fit, and the results as one document.

* **One record per distinct fit that completes**, in completion order, written once: a press of Fit
  that computes nothing new and a revert back to a fit already recorded write none, and no line is
  ever rewritten. Each record names the fit's key, the records in force it was computed from (an
  answer given again replaces its first record there), the plan's digest (the plan export's own
  SHA-256 over its plan, computed here with hashlib) and the seconds and workers it ran with.
* **Each number is withheld wherever the stage route would withhold it for that fit**, read both
  ways and for each record's own scope: under Predict no cross-validated score before Fit is
  pressed for the record's own outcome, and every one after it, the fit's own; no held-out score
  before the seal and, after it, the sealed frame's scores of a record scored on the rows the
  opening opened, and none of a record scored on rows drawn with another seed (the sealed frame's
  row ids, hashed here with hashlib); on the opened rows, the opened fit's unmarked, a fit
  computed after the opening and the current fit's with the marks the stage route served beside
  them, and none of a fit computed before the opening until it is the current fit. Under Estimate
  no estimate before the plan's lock, and after it the exposure's rows of the fit the lock was
  recorded on, whether or not an answer it does not read came between it and the lock, and none of
  a fit computed before the lock under another plan, until it is the fit the stage route serves.
  What is served counts as shown, as on the stage route.
* **The movement under Predict** is ``|mean Δ| / SD(Δ)`` of the primary metric's fold scores,
  computed here with numpy on the two records' served fold scores, where the two fits' splits draw
  the same assignment; across a split drawn with another seed there is none, and the record says so.
* **GET /results**: every field equals the stage route's own (the fit's and the split's, and under
  Estimate the effects stage's), before Fit and after it, before the seal and after it, and after
  a change made after the opening. A fit served while another waits counts as shown for its own
  outcome, so a family fitted alone after it is not declared before any score was seen.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import pytest

from turbotab.core import seal
from turbotab.core.graph import artifact_dir, read_artifact
from turbotab.core.models.selection import read_seen
from turbotab.core.runs import RUNS_FILE
from turbotab.core.stages.modeling import read_assignment
from turbotab.server.service import STALE_OUTCOME_UNREAD, ProjectService
from turbotab.server.tests.conftest import prepare, wait_for
from turbotab.server.tests.test_quest_api import needs_nhanes
from turbotab.server.tests.test_seal_api import assert_nothing_held_out_in, decide, sealed_project


def service_of(client) -> ProjectService:
    return client.app.state.service


def fits(client, pid: str) -> list[dict]:
    response = client.get(f"/api/projects/{pid}/fits")
    assert response.status_code == 200, response.text
    return response.json()["records"]


def results(client, pid: str) -> dict:
    response = client.get(f"/api/projects/{pid}/results")
    assert response.status_code == 200, response.text
    return response.json()


def stage(client, pid: str, name: str) -> dict:
    return client.get(f"/api/projects/{pid}/stages/{name}").json()["artifact"] or {}


def settled(client, pid: str, n: int, timeout: float = 60.0) -> list[dict]:
    """The records once ``n`` are written (each is written on the recorder's thread when its fit
    turns fresh); the recorder's queue drained before each read."""
    service = service_of(client)
    end = time.monotonic() + timeout
    while True:
        service._recorder.submit(lambda: None).result(timeout=timeout)
        found = fits(client, pid)
        if len(found) >= n:
            return found
        assert time.monotonic() < end, f"{len(found)} records, waiting for {n}"
        time.sleep(0.05)


def fit_key(client, pid: str) -> str:
    return service_of(client).engine.status(pid)["fit"].key


def plan_sha(client, pid: str) -> str:
    """The SHA-256 of the plan the plan export serves, over its canonical JSON (hashlib's own)."""
    plan = json.loads(client.get(f"/api/projects/{pid}/plan").content)["plan"]
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def by_family(fit: dict) -> dict[str, dict]:
    return {m["family"]: m for m in fit.get("models") or []}


def split_key(client, pid: str) -> str:
    return service_of(client).engine.status(pid)["split"].key


def assignment(client, pid: str, key: str):
    """The rows' partition and folds the split at ``key`` drew (the fit's own reader)."""
    cache = service_of(client).workspace.cache_dir(pid)
    return read_assignment(read_artifact(cache, "split", key))


def kinds(client, pid: str, kind: str) -> list[dict]:
    view = client.get(f"/api/projects/{pid}").json()
    return [r for r in view["decisions"] if r["decision"]["kind"] == kind]


def held_out_rows(client, pid: str, key: str) -> list[int]:
    """The row ids the split at ``key`` holds out: its sealed frame, read with pandas."""
    cache = service_of(client).workspace.cache_dir(pid)
    frame = pd.read_parquet(artifact_dir(cache, "split", key) / "frames" / "sealed.parquet")
    return sorted(int(i) for i in frame["row_id"])


def rows_sha(rows: list[int]) -> str:
    """The SHA-256 of row ids, sorted and comma-joined (hashlib's own)."""
    return hashlib.sha256(",".join(map(str, sorted(rows))).encode("utf-8")).hexdigest()


SEALED = "The held-out rows are sealed: no held-out score is served until the seal is opened."
SEAL_FIELDS = ("at_opening", "final_note", "changed_after_seal", "post_seal_decisions")


def assert_split_as_served(client, pid: str, document: dict) -> None:
    """/results' split is the split stage's own basis, chronology and exploratory flag."""
    split = stage(client, pid, "split")
    assert document["split"]["stage"]["key"] == split_key(client, pid)
    for k in ("basis", "chronology", "exploratory"):
        assert document["split"][k] == split[k], k
    assert document["split"]["basis"]["state"] == "grouped"  # participant_id repeats: grouped


STORED = ("id", "seq", "at", "purpose", "target", "fit_key", "plan_digest", "decisions", "press",
          "families", "seconds", "workers", "primary_metric", "folds", "seed", "held_out")


def test_one_record_per_distinct_fit_withheld_as_the_stage_route_and_never_rewritten(client):
    service = service_of(client)
    pid = sealed_project(client)
    pdir = service.workspace.project_dir(pid)
    started = time.monotonic()
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    [first] = settled(client, pid, 1)
    took = time.monotonic() - started
    plan_then, split_then = plan_sha(client, pid), split_key(client, pid)
    roles_then = kinds(client, pid, "set_roles")[-1]
    chosen = kinds(client, pid, "select_models")[-1]

    assert first["seq"] == 1 and first["fit_key"] == fit_key(client, pid)
    assert first["purpose"] == "prediction" and first["target"] == "hba1c"
    assert first["families"] == ["linear"] and first["press"] is None  # computed before any press
    assert first["workers"] == service.runner.workers == 2
    assert 0 < first["seconds"] < took
    assert {chosen["seq"], roles_then["seq"]} <= set(first["decisions"])
    assert max(first["decisions"]) <= chosen["seq"]  # nothing recorded after the fit
    assert first["plan_digest"] == plan_then
    assert first["movement"] == []
    assert first["movement_why"] == "No earlier fit is recorded to compare this one with."

    # Before Fit: no score, as the stage route serves none, and nothing is marked seen.
    report, document = fits(client, pid)[0], results(client, pid)
    assert read_seen(pdir) == {}
    served = stage(client, pid, "fit")
    assert served["withheld_for"] == "fit" and by_family(served)["linear"]["cv"] == {}
    assert report["scores"] == {"linear": {}} and report["holdout"] is None
    assert report["withheld"] == served["withheld"]
    assert document["withheld"] == served["withheld"]
    assert document["families"] == [{
        "family": "linear", "label": by_family(served)["linear"]["label"], "cv": {},
        "holdout": None, "intervals": None, "calibration": None, "effect": None, "sequence": [],
        "marginal": None, "e_values": []}]
    assert document["result"] == served.get("result")

    # A press that fits nothing new writes no record, however often it is pressed.
    for _ in range(2):
        assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    assert len(settled(client, pid, 1)) == 1
    lines = (pdir / RUNS_FILE).read_bytes()

    # After Fit: every score the fit computed, as the stage route serves it; served counts as seen.
    [report] = fits(client, pid)
    assert read_seen(pdir) == {"hba1c": ["linear"]}
    computed = read_artifact(service.workspace.cache_dir(pid), "fit", first["fit_key"], public=True)
    served = stage(client, pid, "fit")
    assert report["scores"]["linear"] == by_family(computed)["linear"]["cv"]
    assert report["scores"]["linear"] == by_family(served)["linear"]["cv"] != {}
    assert report["withheld"] is None and report["holdout"] is None  # still sealed
    assert {k: report[k] for k in STORED} == {k: first[k] for k in STORED}
    document = results(client, pid)
    assert document["families"][0]["cv"] == by_family(served)["linear"]["cv"]
    assert document["result"] == served["result"] and document["withheld"] is None

    # An answer given again: a new fit, its own record; the first is kept byte for byte.
    roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
    decide(client, pid, {"kind": "set_roles", "roles": {**roles, "sodium_mg": "excluded"}})
    roles_now = kinds(client, pid, "set_roles")[-1]
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    one, two = settled(client, pid, 2)
    assert (pdir / RUNS_FILE).read_bytes().startswith(lines)
    assert {k: one[k] for k in STORED} == {k: first[k] for k in STORED}
    assert two["seq"] == 2 and two["fit_key"] == fit_key(client, pid) != first["fit_key"]
    assert roles_now["seq"] in two["decisions"] and roles_then["seq"] not in two["decisions"]
    assert roles_now["seq"] not in one["decisions"] and roles_then["seq"] in one["decisions"]
    assert two["plan_digest"] == plan_sha(client, pid) != first["plan_digest"]

    # The same folds (the two fits' splits draw one assignment): the movement is paired_folds on
    # the two records' served fold scores.
    assert assignment(client, pid, split_then).equals(assignment(client, pid, split_key(client, pid)))
    assert one["folds"] == two["folds"] and two["movement_why"] is None
    metric = two["primary_metric"]
    before = np.asarray(one["scores"]["linear"][metric]["folds"], dtype=float)
    after = np.asarray(two["scores"]["linear"][metric]["folds"], dtype=float)
    d = after - before
    assert d.any()  # sodium left the model: its scores moved
    [moved] = two["movement"]
    assert moved["family"] == "linear" and moved["against"] == one["id"] and moved["why"] is None
    assert moved["movement"]["instrument"] == "paired_folds"
    assert moved["movement"]["value"] == pytest.approx(abs(d.mean()) / d.std(ddof=1), rel=1e-9)

    # Back to the first answers: the first fit's key, already recorded, writes nothing.
    back = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": roles_now["id"]})
    assert back.status_code == 200, back.text
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert fit_key(client, pid) == first["fit_key"]
    assert [r["fit_key"] for r in settled(client, pid, 2)] == [first["fit_key"], two["fit_key"]]

    # The goal withdrawn: the stage route withholds every score, and so does every record, its
    # movement with it; a fit that cannot compute writes no record.
    goal = kinds(client, pid, "set_purpose")[-1]
    undo = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": goal["id"]})
    assert undo.status_code == 200, undo.text
    served = stage(client, pid, "fit")
    assert served["withheld_for"] == "purpose" and by_family(served)["linear"]["cv"] == {}
    withdrawn = settled(client, pid, 2)
    assert len(withdrawn) == 2
    for r in withdrawn:
        assert r["scores"] == {"linear": {}} and r["movement"] == []
        assert r["withheld"] == served["withheld"]
    assert withdrawn[1]["movement_why"] is None  # it has a movement: withheld, not absent


def test_held_out_scores_are_the_opened_rows_only_and_other_folds_carry_no_movement(client):
    service = service_of(client)
    pid = sealed_project(client)
    cache = service.workspace.cache_dir(pid)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    settled(client, pid, 1)
    split_then = split_key(client, pid)
    decide(client, pid, {"kind": "set_split", "holdout": 0.25, "seed": 1, "folds": 5})
    wait_for(client, pid, {"fit": "fresh", "split": "fresh"}, timeout=180)
    one, two = settled(client, pid, 2)

    # Another seed draws other folds: no paired movement, and why.
    assert not assignment(client, pid, split_then).equals(
        assignment(client, pid, split_key(client, pid)))
    assert one["seed"] == 0 and two["seed"] == 1 and one["folds"] != two["folds"]
    assert two["movement"] == []
    assert two["movement_why"] == (
        "The previous fit's folds were drawn with seed 0, this one's with seed 1.")
    assert two["scores"]["linear"]  # the scores are served: the movement is absent, not withheld

    # Each record names the rows its split held out; seed 1 holds out other rows than seed 0.
    rows_then, rows_now = held_out_rows(client, pid, split_then), held_out_rows(
        client, pid, split_key(client, pid))
    assert rows_then != rows_now
    assert one["held_out"] == rows_sha(rows_then) and two["held_out"] == rows_sha(rows_now)

    # Before the seal: no held-out score anywhere in either document, as on the stage route.
    sealed = {f"{r['seq']}:{family}": scores
              for r in (one, two) for family, scores in
              (seal.read_sealed_scores(cache, r["fit_key"]) or {}).items()}
    assert len(sealed) == 2
    bodies = [client.get(f"/api/projects/{pid}/fits").json(), results(client, pid)]
    assert_nothing_held_out_in(bodies, sealed)
    assert all(r["holdout"] is None and r["holdout_why"] == SEALED for r in fits(client, pid))
    served = stage(client, pid, "fit")
    document = results(client, pid)
    assert document["holdout_sealed"] is served["holdout_sealed"] is True
    assert document["families"][0]["holdout"] is None and document["families"][0]["intervals"] is None
    for k in SEAL_FIELDS:
        assert document[k] == served[k], k
    assert_split_as_served(client, pid, document)

    # The seal opened on seed 1's rows: the record scored on them serves its sealed frame's scores.
    # Seed 0's rows were never opened (only a recorded re-seal draws them again), so its record
    # serves none, and says why.
    prepare(client, pid, {"kind": "open_seal", "family": "linear"})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    decide(client, pid, {"kind": "open_seal", "family": "linear"})
    opening = kinds(client, pid, "open_seal")[-1]
    one, two = fits(client, pid)
    assert two["holdout"]["linear"] == seal.read_sealed_scores(cache, two["fit_key"])["linear"]
    assert two["holdout"]["linear"] and two["holdout_why"] is None
    assert one["holdout"] is None
    assert one["holdout_why"] == (
        f"This fit was scored on other held-out rows than those the opening at decision "
        f"#{opening['seq']} opened, so its held-out scores are not served.")
    assert_nothing_held_out_in([one], {"1:linear": sealed["1:linear"]})
    served, document = stage(client, pid, "fit"), results(client, pid)
    model = by_family(served)["linear"]
    [family] = document["families"]
    assert family["holdout"] == model["holdout"]
    assert family["holdout"] == seal.read_sealed_scores(cache, two["fit_key"])["linear"]
    assert family["intervals"] == model["holdout_detail"]["intervals"]
    assert family["calibration"] == model["holdout_detail"]["calibration"]
    assert family["cv"] == model["cv"]
    assert document["holdout_sealed"] is served["holdout_sealed"] is False
    assert document["final_model"] == served["final_model"] == "linear"
    assert document["result"] == served["result"]
    assert document["fit"] == {"key": two["fit_key"], "fresh": True, "status": "fresh"}
    for k in SEAL_FIELDS:
        assert document[k] == served[k], k
    assert document["at_opening"]["current"] is True and document["changed_after_seal"] is False
    assert document["at_opening"]["scores"]["linear"] == two["holdout"]["linear"]
    assert_split_as_served(client, pid, document)

    # A change after the opening (sodium left out) fits again on the opened rows: its record serves
    # its held-out scores as the stage route does, and /results says what the screen says beside
    # them: the scores at the opening stay the reported result, and these are post-seal.
    roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
    decide(client, pid, {"kind": "set_roles", "roles": {**roles, "sodium_mg": "excluded"}})
    change = kinds(client, pid, "set_roles")[-1]
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    one, two, three = settled(client, pid, 3)
    assert three["held_out"] == two["held_out"]  # the opened rows hold still
    assert three["holdout"]["linear"] == seal.read_sealed_scores(cache, three["fit_key"])["linear"]
    assert three["holdout"]["linear"] != two["holdout"]["linear"] and three["holdout_why"] is None
    assert one["holdout"] is None
    served, document = stage(client, pid, "fit"), results(client, pid)
    for k in SEAL_FIELDS:
        assert document[k] == served[k], k
    assert document["changed_after_seal"] is True and document["at_opening"]["current"] is False
    assert document["final_note"] == document["at_opening"]["note"]
    assert change["id"] in document["post_seal_decisions"]
    assert document["at_opening"]["scores"]["linear"] == two["holdout"]["linear"]
    assert document["families"][0]["holdout"] == by_family(served)["linear"]["holdout"]
    assert document["families"][0]["holdout"] == three["holdout"]["linear"]
    assert_split_as_served(client, pid, document)


RECORD_MARKS = ("at_opening", "final_note", "changed_after_seal")


@pytest.mark.slow  # three fits and two reverts on the dietary recalls: about 40 s
def test_after_the_opening_a_records_held_out_scores_carry_the_marks_the_stage_route_puts(client):
    """One draw of held-out rows, fits that differ in their predictors only. After the opening a
    record serves its held-out scores where the stage route has served that fit's: the opened fit's
    unmarked, a fit computed after the opening (by a recorded change) marked as the stage route
    marked it while it was current, and the current fit's exactly the stage route's. A fit computed
    before the opening that is not the opened one serves none and says why, until a recorded change
    makes it the current fit, when its record is marked as the stage route marks it."""
    service = service_of(client)
    pid = sealed_project(client)
    cache = service.workspace.cache_dir(pid)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    settled(client, pid, 1)
    roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
    decide(client, pid, {"kind": "set_roles", "roles": {**roles, "sodium_mg": "excluded"}})
    sodium = kinds(client, pid, "set_roles")[-1]
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    prepare(client, pid, {"kind": "open_seal", "family": "linear"})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    opened_key = fit_key(client, pid)
    decide(client, pid, {"kind": "open_seal", "family": "linear"})
    opening = kinds(client, pid, "open_seal")[-1]
    one, two = settled(client, pid, 2)
    assert two["fit_key"] == opened_key != one["fit_key"]
    assert one["held_out"] == two["held_out"] is not None  # one draw: the same held-out rows
    sealed_one = seal.read_sealed_scores(cache, one["fit_key"])["linear"]
    sealed_two = seal.read_sealed_scores(cache, two["fit_key"])["linear"]
    assert sealed_one != sealed_two

    # At the opening: the opened fit's record is the stage route's, unmarked.
    at_opening = stage(client, pid, "fit")
    assert two["holdout"] == {"linear": sealed_two} and two["holdout_why"] is None
    for k in RECORD_MARKS:
        assert two[k] == at_opening[k], k
    assert two["changed_after_seal"] is False and two["at_opening"]["current"] is True
    assert two["at_opening"]["scores"]["linear"] == sealed_two
    # The fit computed before the opening, on the same rows: no held-out score, and why.
    before = (f"This fit was computed before the opening at decision #{opening['seq']}, which "
              f"opened another fit's held-out scores, so this fit's are not served.")
    assert one["holdout"] is None and one["holdout_why"] == before
    assert one["changed_after_seal"] is True
    assert one["at_opening"] is None and one["final_note"] is None
    assert_nothing_held_out_in([one], {"1:linear": sealed_one})

    # A change after the opening (fiber left out too): the current fit's record is the stage
    # route's, marked post-seal; the opened fit's stays unmarked; the earlier fit's still withheld.
    roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
    decide(client, pid, {"kind": "set_roles", "roles": {**roles, "fiber_g": "excluded"}})
    fiber = kinds(client, pid, "set_roles")[-1]
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    one, two, three = settled(client, pid, 3)
    post_seal = stage(client, pid, "fit")
    assert three["fit_key"] == fit_key(client, pid) and three["held_out"] == two["held_out"]
    assert three["holdout"] == {"linear": by_family(post_seal)["linear"]["holdout"]}
    assert three["holdout"]["linear"] == seal.read_sealed_scores(cache, three["fit_key"])["linear"]
    for k in RECORD_MARKS:
        assert three[k] == post_seal[k], k
    assert three["changed_after_seal"] is True and three["at_opening"]["current"] is False
    assert three["final_note"] == three["at_opening"]["note"]
    assert two["holdout"] == {"linear": sealed_two}
    for k in RECORD_MARKS:
        assert two[k] == at_opening[k], k
    assert one["holdout"] is None and one["holdout_why"] == before

    # Reverted: the opened fit is current again. The post-seal fit's record keeps the marks the
    # stage route served beside its scores while it was current.
    back = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": fiber["id"]})
    assert back.status_code == 200, back.text
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert fit_key(client, pid) == two["fit_key"]
    one, two, three = settled(client, pid, 3)
    assert three["holdout"] == {"linear": by_family(post_seal)["linear"]["holdout"]}
    for k in RECORD_MARKS:
        assert three[k] == post_seal[k], k
    served = stage(client, pid, "fit")
    for k in RECORD_MARKS:
        assert two[k] == served[k], k
    assert one["holdout"] is None and one["holdout_why"] == before

    # The sodium change reverted too: the fit computed before the opening is the current one, so
    # its record serves its held-out scores as the stage route does, marked as it marks them.
    back = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": sodium["id"]})
    assert back.status_code == 200, back.text
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert fit_key(client, pid) == one["fit_key"]
    one, two, three = settled(client, pid, 3)
    served = stage(client, pid, "fit")
    assert one["holdout"] == {"linear": sealed_one} == {
        "linear": by_family(served)["linear"]["holdout"]}
    assert one["holdout_why"] is None
    for k in RECORD_MARKS:
        assert one[k] == served[k], k
    assert one["changed_after_seal"] is True and one["at_opening"]["current"] is False
    assert two["holdout"] == {"linear": sealed_two} and two["at_opening"]["current"] is True


def stored_estimates(client, pid: str, record_id: str) -> dict[str, float]:
    """The exposure's estimates and interval ends a record kept, as written in its line."""
    path = service_of(client).workspace.project_dir(pid) / RUNS_FILE
    [line] = [json.loads(t) for t in path.read_text("utf-8").splitlines()
              if t.strip() and json.loads(t)["id"] == record_id]
    return {f"{family}:{row['feature']}:{k}": row[k] for family, rows in line["effects"].items()
            for row in rows for k in ("estimate", "ci_low", "ci_high")}


def nhanes_under_estimate(client):
    """The NHANES fixture under Estimate up to a fresh fit of the linear family: glucose by sugar,
    energy-adjusted, kcal kept within 500–5000, no rows held out. Returns the drive."""
    from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
    from turbotab.core.tests.stage_harness import NHANES
    from turbotab.core.tests.truths import fixture_truth
    from turbotab.server.tests.test_quest_api import ROLES, rule

    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "glucose"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(ROLES)
    drive.answer("exclusions", rule(500, 5000))
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    answer_plan(drive, "sugar")
    drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["sugar"]})
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, drive.pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    return drive


@pytest.mark.slow  # three NHANES fits under Estimate: over a minute
@needs_nhanes
def test_under_estimate_no_estimate_before_the_lock_and_only_the_locked_plans_after_it(client):
    from turbotab.server.tests.test_quest_api import rule

    service = service_of(client)
    drive = nhanes_under_estimate(client)
    pid = drive.pid
    first_key = fit_key(client, pid)
    [record] = settled(client, pid, 1)
    assert record["purpose"] == "inference" and record["fit_key"] == first_key
    assert record["scores"] == {} and record["movement_why"].startswith("This fit is under Estimate")
    assert record["plan_digest"] == plan_sha(client, pid)

    # Another exclusion rule before any lock (a fork Gelman & Loken name): a second plan and fit.
    drive.decide(rule(800, 4000))
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    first, second = settled(client, pid, 2)
    assert second["fit_key"] == fit_key(client, pid) != first_key
    assert second["plan_digest"] == plan_sha(client, pid) != first["plan_digest"]
    assert first["id"] == record["id"]

    # Before the lock: no estimate in either document, as the stage routes serve none.
    served, effects = stage(client, pid, "fit"), stage(client, pid, "effects")
    assert served["withheld_for"] == "fit" and by_family(served)["linear"]["coefficients"] is None
    for r in (first, second):
        assert r["effects"] == {"linear": None} and r["withheld"] == served["withheld"]
    document = results(client, pid)
    assert document["estimand"] is None and served.get("estimand") is None
    assert document["families"][0]["effect"] is None and document["families"][0]["sequence"] == []
    assert effects["families"] == [] and document["withheld"] == served["withheld"]
    assert drive.view()["state"]["plan_locked"] is None  # read, withheld, nothing locked

    # Fit locks the second plan and computes nothing new; then every estimate of that fit, the
    # stage routes' own. The first fit's plan is not the locked one and it was computed before the
    # lock: none of its estimates is served, and it says why.
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    lock = kinds(client, pid, "lock_plan")[-1]
    first, second = settled(client, pid, 2)
    assert second["plan_digest"] == lock["decision"]["digest"] != first["plan_digest"]
    text = json.dumps(lock["decision"]["plan"], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    assert second["plan_digest"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert service._shown_lock(pid) == lock["id"]  # an estimate served under the lock: it stands
    served, effects = stage(client, pid, "fit"), stage(client, pid, "effects")
    rows = by_family(served)["linear"]["coefficients"]
    assert rows and {r["feature"] for r in rows} == set(served["estimand"]["features"])
    assert second["effects"] == {"linear": rows} and second["withheld"] is None
    assert all(r["estimate"] is not None and r["ci_low"] < r["ci_high"] for r in rows)
    assert first["effects"] == {"linear": None}
    assert first["withheld"] == (
        f"No estimate of this fit is served: its plan is not the one the analysis-plan lock at "
        f"decision #{lock['seq']} recorded, and it was computed before that lock.")
    unlocked = stored_estimates(client, pid, first["id"])
    assert unlocked and set(unlocked.values()).isdisjoint(
        stored_estimates(client, pid, second["id"]).values())
    body = json.dumps(client.get(f"/api/projects/{pid}/fits").json())
    for value in unlocked.values():  # the first fit's own numbers, as its line kept them
        assert json.dumps(value) not in body, value
    document = results(client, pid)
    assert document["estimand"] == served["estimand"] and document["withheld"] is None
    [family] = document["families"]
    [effect] = effects["families"]
    assert family["effect"] == rows
    assert family["sequence"] == effect["sequence"] and family["sequence"]
    assert family["marginal"] == effect.get("marginal")
    assert family["e_values"] == [{"feature": s["feature"], "e_value": s["e_value"]}
                                  for s in effect["sensitivity"] if s.get("e_value") is not None]
    assert family["e_values"]
    assert document["effects"] == {"key": service.engine.status(pid)["effects"].key, "fresh": True,
                                   "status": "fresh"}

    # A third rule after the lock (recorded as made after the estimates): a fit computed under the
    # lock, its record served once it is no longer the current fit, the lock among its records.
    drive.decide(rule(600, 4500))
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    first, second, third = settled(client, pid, 3)
    assert third["plan_digest"] != lock["decision"]["digest"] and lock["seq"] in third["decisions"]
    rows_third = by_family(stage(client, pid, "fit"))["linear"]["coefficients"]
    assert rows_third and third["effects"] == {"linear": rows_third} and third["withheld"] is None
    later = kinds(client, pid, "set_exclusions")[-1]
    back = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": later["id"]})
    assert back.status_code == 200, back.text
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    assert fit_key(client, pid) == second["fit_key"]
    first, second, third = settled(client, pid, 3)
    assert third["effects"] == {"linear": rows_third} and third["withheld"] is None
    assert second["effects"] == {"linear": rows} and first["effects"] == {"linear": None}

    # The first rule change reverted too: the first fit is the one the stage route serves again, so
    # its record serves the same estimates, its line's own; the others keep theirs.
    change = kinds(client, pid, "set_exclusions")[-2]
    assert change["decision"]["rules"][0]["low"] == 800
    back = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": change["id"]})
    assert back.status_code == 200, back.text
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    assert fit_key(client, pid) == first_key
    first, second, third = settled(client, pid, 3)
    rows_back = by_family(stage(client, pid, "fit"))["linear"]["coefficients"]
    assert rows_back and first["effects"] == {"linear": rows_back} and first["withheld"] is None
    assert {f"linear:{r['feature']}:{k}": r[k] for r in rows_back
            for k in ("estimate", "ci_low", "ci_high")} == unlocked
    assert second["effects"] == {"linear": rows} and third["effects"] == {"linear": rows_third}


@pytest.mark.slow  # three NHANES fits under Estimate: over a minute
@needs_nhanes
def test_under_estimate_the_lock_covers_the_fit_it_was_recorded_on_whatever_else_is_answered(
        client):
    """An answer the fit does not read (the model sequence, which the effects stage reads) lands
    between the fit and the press of Fit: the lock's plan then holds it and the fit's record, whose
    plan is the one it completed under, does not, though the lock was recorded on that fit (its
    key is the one the answers at the lock give). After a change made after the lock, that fit's
    record serves the estimates the stage route served while it was current; a fit computed before
    the lock under another exclusion rule serves none, and says why."""
    from turbotab.server.tests.test_quest_api import rule

    drive = nhanes_under_estimate(client)
    pid = drive.pid
    drive.decide(rule(800, 4000))
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    first, second = settled(client, pid, 2)
    assert second["fit_key"] == fit_key(client, pid) != first["fit_key"]
    assert second["plan_digest"] == plan_sha(client, pid)

    drive.decide({"kind": "set_model_sequence", "exposure": "sugar", "model_1": ["age", "gender"]})
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    assert fit_key(client, pid) == second["fit_key"]  # the fit does not read the sequence
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    lock = kinds(client, pid, "lock_plan")[-1]
    first, second = settled(client, pid, 2)  # Fit computed nothing new
    assert lock["decision"]["digest"] == plan_sha(client, pid) != second["plan_digest"]
    assert lock["seq"] not in second["decisions"]
    rows = by_family(stage(client, pid, "fit"))["linear"]["coefficients"]
    assert rows and all(r["estimate"] is not None and r["ci_low"] < r["ci_high"] for r in rows)
    assert second["effects"] == {"linear": rows} and second["withheld"] is None

    # A change after the lock: the locked fit is no longer current; its record serves the same rows.
    drive.decide(rule(600, 4500))
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    first, second, third = settled(client, pid, 3)
    assert third["fit_key"] == fit_key(client, pid) != second["fit_key"]
    assert second["effects"] == {"linear": rows} and second["withheld"] is None
    assert first["effects"] == {"linear": None}
    assert first["withheld"] == (
        f"No estimate of this fit is served: its plan is not the one the analysis-plan lock at "
        f"decision #{lock['seq']} recorded, and it was computed before that lock.")
    body = json.dumps(client.get(f"/api/projects/{pid}/fits").json())
    for value in stored_estimates(client, pid, first["id"]).values():
        assert json.dumps(value) not in body, value


def test_a_records_exposure_rows_are_its_own_whatever_the_plan_names_now():
    """From construction: the rows a record kept under the estimand it was computed for are served
    wherever the served fit shows its coefficients (a later exposure moves the fit's own rows to
    its adjustment terms under the current caption), and none where the fit withholds them."""
    from datetime import datetime, timezone

    from turbotab.core.runs import FitRecord
    from turbotab.server.service import _served_record

    sugar = {"feature": "sugar", "estimate": -0.028, "ci_low": -0.041, "ci_high": -0.015}
    fiber = {"feature": "fiber", "estimate": 0.012, "ci_low": 0.004, "ci_high": 0.020}
    rec = FitRecord(id="r" * 32, seq=1, at=datetime(2026, 10, 18, tzinfo=timezone.utc),
                    purpose="inference", target="glucose", fit_key="k", plan_digest="p",
                    decisions=[1], press=None, families=["linear"], seconds=1.0, workers=2,
                    primary_metric=None, folds=None, seed=0, effects={"linear": [sugar]})
    now = {"models": [{"family": "linear", "cv": {}, "holdout": None, "coefficients": [fiber],
                       "adjustment_terms": [{**sugar, "why": "an adjustment term"}]}],
           "n_holdout": 0, "withheld": None}
    out = _served_record(rec, now)
    assert out["effects"] == {"linear": [sugar]} and out["withheld"] is None
    withheld = {"models": [{"family": "linear", "cv": {}, "holdout": None,
                            "coefficients": None}], "n_holdout": 0, "withheld": "Not yet."}
    out = _served_record(rec, withheld)
    assert out["effects"] == {"linear": None} and out["withheld"] == "Not yet."


def test_scores_wait_for_a_press_of_fit_for_the_records_own_outcome(client):
    service = service_of(client)
    pid = sealed_project(client)
    pdir = service.workspace.project_dir(pid)
    decide(client, pid, {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    [first] = settled(client, pid, 1)
    unpressed = stage(client, pid, "fit")["withheld"]  # the stage's reason before Fit
    assert unpressed and first["target"] == "hba1c"
    assert first["scores"] == {"linear": {}} and first["withheld"] == unpressed

    # Another outcome, and Fit pressed for it: that record's scores are served, the stage route's
    # own. hba1c's still wait for a press for hba1c, and none of them was ever marked seen.
    decide(client, pid, {"kind": "set_target", "column": "bmi"})
    wait_for(client, pid, {"roles": "fresh"}, timeout=180)
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles",
                         "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    one, two = settled(client, pid, 2)
    served = stage(client, pid, "fit")
    assert two["target"] == "bmi" and two["fit_key"] == fit_key(client, pid)
    assert two["scores"]["linear"] == by_family(served)["linear"]["cv"] != {}
    assert two["withheld"] is None
    assert one["id"] == first["id"] and one["target"] == "hba1c"
    assert one["scores"] == {"linear": {}} and one["withheld"] == unpressed
    assert read_seen(pdir) == {"bmi": ["linear"]}


def held_for_fit(client, pid: str, timeout: float = 60.0) -> dict:
    """The view once the scheduler holds the fit for a press of Fit."""
    end = time.monotonic() + timeout
    while True:
        view = client.get(f"/api/projects/{pid}").json()
        if view["stages"]["fit"].get("held") == "fit":
            return view
        assert time.monotonic() < end, view["stages"]["fit"]
        time.sleep(0.05)


def test_results_count_a_waiting_fits_scores_as_seen_for_its_own_outcome(client, monkeypatch):
    from turbotab.core import fit_press, runs

    service = service_of(client)
    pid = sealed_project(client)
    pdir = service.workspace.project_dir(pid)
    decide(client, pid, {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    decide(client, pid, {"kind": "select_models", "models": ["ridge"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    service._recorder.submit(lambda: None).result(timeout=60)  # its record written (not read:
    [ridge] = runs.RunLog(pdir / RUNS_FILE).records()  # GET /fits would mark its scores seen)
    assert read_seen(pdir) == {}

    # Linear alone, expected to take far longer than the press confirmed: the scheduler holds it
    # for Fit (RECIPES §4.4), so the ridge fit is the one both routes serve meanwhile.
    monkeypatch.setattr(fit_press, "fit_estimate", lambda shelf, models: 1000.0)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    held_for_fit(client, pid)

    # Its outcome named by no record and no prefix of the log: none of its scores is served.
    path = pdir / RUNS_FILE
    aside = path.with_name("fits.aside.jsonl")
    path.rename(aside)
    with monkeypatch.context() as m:
        m.setattr(runs, "keyed_prefix", lambda *args, **kwargs: None)
        document = results(client, pid)
    aside.rename(path)
    assert document["fit"]["key"] == ridge.fit_key and document["fit"]["fresh"] is False
    assert document["families"][0]["family"] == "ridge" and document["families"][0]["cv"] == {}
    assert document["withheld"] == STALE_OUTCOME_UNREAD and document["result"] is None
    assert read_seen(pdir) == {}

    # Its record names hba1c: its scores are served as the stage route serves them, and are kept as
    # seen for hba1c (the record alone names it here: no prefix of the log is read).
    with monkeypatch.context() as m:
        m.setattr(runs, "keyed_prefix", lambda *args, **kwargs: None)
        document = results(client, pid)
    served = stage(client, pid, "fit")
    assert document["fit"]["key"] == ridge.fit_key and document["fit"]["fresh"] is False
    assert document["families"][0]["cv"] == by_family(served)["ridge"]["cv"] != {}
    assert document["withheld"] is None
    assert read_seen(pdir) == {"hba1c": ["ridge"]}

    # Fit pressed: linear is fitted alone after ridge's score was shown, so no result is declared.
    monkeypatch.undo()
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    document, served = results(client, pid), stage(client, pid, "fit")
    assert document["fit"]["fresh"] is True and [f["family"] for f in document["families"]] == [
        "linear"]
    assert document["result"] == served["result"]
    assert document["result"]["basis"] == "not_declared" and document["result"]["estimate"] is None


def test_a_records_gate_is_the_stage_routes_read_for_its_own_goal_outcome_and_plan():
    """From construction, each case both ways: the gate ``_record_gate`` serves a record under,
    against the stage's own rule for it (``fit_press``: the goal first, then under Predict a press
    for the record's outcome, under Estimate the lock), and the lock in force covering the
    record: its plan the lock's, its fit the one the answers at the lock key, or the lock among
    its records."""
    from types import SimpleNamespace

    from turbotab.core import fit_press
    from turbotab.core.runs import FitRecord
    from turbotab.server.service import FIT_ANSWERS_UNREAD

    def record(purpose, target="hba1c", plan="p1", decisions=(1, 2)):
        return FitRecord(id="r" * 32, seq=1, at="2026-10-18T00:00:00Z", purpose=purpose,
                         target=target, fit_key="k", plan_digest=plan, decisions=list(decisions),
                         press=None, families=["linear"], seconds=1.0, workers=2,
                         primary_metric="mse", folds=None, seed=0)

    def gate(rec, *, purpose, locked=None, press=None, lock=None, question=None,
             locked_key=None):
        state = SimpleNamespace(purpose=purpose, plan_locked=locked, target="bmi")
        return ProjectService._record_gate(SimpleNamespace(replay=False), rec, state, question,
                                           press, lock, locked_key)

    def reason(found):
        return None if found is None else found["reason"]

    # No goal now: the goal's own gate, whatever the record.
    assert reason(gate(record("prediction"), purpose=None)) == fit_press.PURPOSE_FIRST
    # The record's answers unread, or another goal: everything withheld, and why.
    assert gate(record(None), purpose="prediction")["reason"] == FIT_ANSWERS_UNREAD == (
        "This fit cannot be matched to your recorded decisions: at no point in your decision "
        "log, up to when the fit completed, were the answers those it was computed from, so none "
        "of its numbers is served.")
    other = gate(record("prediction"), purpose="inference", locked=True)
    assert other["reason"] == ("This fit was computed under prediction, and the goal is now "
                               "inference: a fit's numbers are served only under the goal it was "
                               "computed for.") and other["scores"] is True
    # Predict: Fit pressed for the record's own outcome, not the outcome now (bmi).
    pressed = {"target": "hba1c", "seconds": 1.0, "seq": 3}
    assert gate(record("prediction"), purpose="prediction", press=pressed) is None
    assert reason(gate(record("prediction"), purpose="prediction",
                       press={**pressed, "target": "bmi"})) == fit_press.FIT_FIRST
    assert reason(gate(record("prediction"), purpose="prediction",
                       press={**pressed, "withdrawn": True})) == fit_press.FIT_FIRST
    question = {"question": "follow_up", "reason": "Answer the follow-up first.", "exits": [],
                "purpose": "prediction"}
    assert reason(gate(record("prediction"), purpose="prediction", press=pressed,
                       question=question)) == "Answer the follow-up first."
    # Estimate: nothing before the lock; after it, the lock's own plan, or a fit computed under it.
    lock = SimpleNamespace(seq=7, decision=SimpleNamespace(digest="p1"))
    assert reason(gate(record("inference"), purpose="inference")) == fit_press.LOCK_FIRST
    assert gate(record("inference"), purpose="inference", locked=True, lock=lock) is None
    assert gate(record("inference", plan="p2", decisions=(1, 7)), purpose="inference",
                locked=True, lock=lock) is None
    # The fit the lock was recorded on, though an answer it does not read came between the fit
    # and the press (its plan is not the lock's): covered by its key.
    assert gate(record("inference", plan="p2"), purpose="inference", locked=True, lock=lock,
                locked_key="k") is None
    refused = ("No estimate of this fit is served: its plan is not the one the analysis-plan lock "
               "at decision #7 recorded, and it was computed before that lock.")
    assert reason(gate(record("inference", plan="p2"), purpose="inference", locked=True,
                       lock=lock)) == refused
    assert reason(gate(record("inference", plan="p2"), purpose="inference", locked=True,
                       lock=lock, locked_key="another")) == refused
    # The digest's case rests on this: the fit reads only slots the plan holds, so two plans alike
    # are two fits alike under one engine (the graph's own reads, not the gate's).
    from turbotab.core import plan_lock, seal
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    reads = seal.slots_read_by(load_graph(GRAPH_FACTORY), "fit")
    assert reads and reads <= set(plan_lock.plan_slots()) - set(plan_lock.LOOKS)


def test_a_fit_no_recorded_decisions_match_is_refused_in_plain_words():
    """The three refusals served when no point in the decision log gives a fit's answers (in
    /fits' withheld and movement_why, and /results' withheld) say so in the reader's words, not
    the engine's (a prefix of the log, a key)."""
    from turbotab.core import runs
    from turbotab.server.service import FIT_ANSWERS_UNREAD

    assert runs.ANSWERS_UNREAD == (
        "This fit cannot be matched to your recorded decisions: at no point in your decision log, "
        "up to when the fit completed, were the answers those it was computed from, so its folds "
        "are not known and no movement is measured.")
    assert STALE_OUTCOME_UNREAD == (
        "This fit is not the current one, and neither its record nor any point in your decision "
        "log gives the answers it was computed from, so the outcome its scores are for is not "
        "known and none of them is served.")
    for text in (FIT_ANSWERS_UNREAD, runs.ANSWERS_UNREAD, STALE_OUTCOME_UNREAD):
        assert "prefix" not in text and "keys" not in text, text


def test_a_records_held_out_scores_are_served_only_on_the_rows_the_opening_opened():
    """From construction, each case both ways: ``_held_out_why`` against the stage route's terms
    (rows held out, the seal opened, the record's outcome the opened seal's, the same rows), and
    the fit's own: the opened one, one computed after the opening (the opening among its
    records), or the current one; never another computed before the opening."""
    from turbotab.core.runs import FitRecord
    from turbotab.server.service import _held_out_why

    def record(target="hba1c", held_out="rows-a", key="k", decisions=(1, 24)):
        return FitRecord(id="r" * 32, seq=1, at="2026-10-18T00:00:00Z", purpose="prediction",
                         target=target, fit_key=key, plan_digest="p", decisions=list(decisions),
                         press=None, families=["linear"], seconds=1.0, workers=2,
                         primary_metric="mse", folds=None, seed=0, held_out=held_out)

    opened_fit, opened = {"n_holdout": 150, "holdout_sealed": False}, (24, "rows-a", "k")
    assert _held_out_why(record(), opened_fit, opened, "hba1c") is None
    # Computed before the opening (it is not among its records): the opened fit's own, the
    # current fit's, and no other's.
    before = record(key="k0", decisions=(1, 2))
    assert _held_out_why(before, opened_fit, (24, "rows-a", "k0"), "hba1c") is None
    assert _held_out_why(before, opened_fit, opened, "hba1c", current="k0") is None
    assert _held_out_why(before, opened_fit, opened, "hba1c", current="k") == (
        "This fit was computed before the opening at decision #24, which opened another fit's "
        "held-out scores, so this fit's are not served.")
    assert _held_out_why(before, opened_fit, (24, "rows-a", None), "hba1c").startswith(
        "This fit was computed before the opening at decision #24")
    assert _held_out_why(record(key="k2"), opened_fit, opened, "hba1c", current="k") is None
    # A record whose fit is no longer kept: whether it is the opened fit, from the keys alone.
    from turbotab.server.service import _served_record

    gone = _served_record(before, None, opened=opened)
    assert gone["changed_after_seal"] is True and gone["at_opening"] is None
    assert _served_record(record(), None, opened=opened)["changed_after_seal"] is False
    assert _served_record(before, None, opened=None)["changed_after_seal"] is False
    assert _held_out_why(record(), {"n_holdout": 0}, opened, "hba1c") == (
        "No rows are held out of this fit's split, so it has no held-out score.")
    assert _held_out_why(record(), {"n_holdout": 150, "holdout_sealed": True}, None,
                         "hba1c") == SEALED
    assert _held_out_why(record(target="bmi"), opened_fit, opened, "hba1c") == (
        "This fit predicts `bmi`, and the seal opened is `hba1c`'s: a fit's held-out scores are "
        "served only on the rows the opening in force opened.")
    assert _held_out_why(record(held_out=None), opened_fit, opened, "hba1c").startswith(
        "The held-out rows this fit was scored on")
    assert _held_out_why(record(), opened_fit, (24, None, "k"), "hba1c").startswith(
        "The held-out rows this fit was scored on")
    assert _held_out_why(record(held_out="rows-b"), opened_fit, opened, "hba1c") == (
        "This fit was scored on other held-out rows than those the opening at decision #24 "
        "opened, so its held-out scores are not served.")
