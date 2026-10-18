"""SWARM_ENGINE_PLAN §2 E1 and E2 over HTTP: the record of every fit, and the results as one document.

* **One record per distinct fit that completes**, in completion order, written once: a press of Fit
  that computes nothing new and a revert back to a fit already recorded write none, and no line is
  ever rewritten. Each record names the fit's key, the records in force it was computed from (an
  answer given again replaces its first record there), the plan's digest (the plan export's own
  SHA-256 over its plan, computed here with hashlib) and the seconds and workers it ran with.
* **Each number is withheld exactly where the stage route withholds it**, read both ways: under
  Predict no cross-validated score before Fit and every one after it, the fit's own; no held-out
  score before the seal and, after it, the sealed frame's scores of each record's own fit; under
  Estimate no estimate before the plan's lock and the exposure's rows after it, with the lock's own
  digest. What is served counts as shown, as on the stage route.
* **The movement under Predict** is ``|mean Δ| / SD(Δ)`` of the primary metric's fold scores,
  computed here with numpy on the two records' served fold scores, where the two fits' splits draw
  the same assignment; across a split drawn with another seed there is none, and the record says so.
* **GET /results**: every field equals the stage route's own (the fit's, and under Estimate the
  effects stage's), before Fit and after it, before the seal and after it.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest

from turbotab.core import seal
from turbotab.core.graph import read_artifact
from turbotab.core.models.selection import read_seen
from turbotab.core.runs import RUNS_FILE
from turbotab.core.stages.modeling import read_assignment
from turbotab.server.service import ProjectService
from turbotab.server.tests.conftest import prepare, wait_for
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


STORED = ("id", "seq", "at", "purpose", "target", "fit_key", "plan_digest", "decisions", "press",
          "families", "seconds", "workers", "primary_metric", "folds", "seed")


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


def test_held_out_scores_wait_for_the_seal_and_other_folds_carry_no_movement(client):
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

    # Before the seal: no held-out score anywhere in either document, as on the stage route.
    sealed = {f"{r['seq']}:{family}": scores
              for r in (one, two) for family, scores in
              (seal.read_sealed_scores(cache, r["fit_key"]) or {}).items()}
    assert len(sealed) == 2
    bodies = [client.get(f"/api/projects/{pid}/fits").json(), results(client, pid)]
    assert_nothing_held_out_in(bodies, sealed)
    assert all(r["holdout"] is None for r in fits(client, pid))
    served = stage(client, pid, "fit")
    document = results(client, pid)
    assert document["holdout_sealed"] is served["holdout_sealed"] is True
    assert document["families"][0]["holdout"] is None and document["families"][0]["intervals"] is None

    # After the seal: each record's held-out scores are its own fit's sealed frame's.
    prepare(client, pid, {"kind": "open_seal", "family": "linear"})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    decide(client, pid, {"kind": "open_seal", "family": "linear"})
    for r in fits(client, pid):
        assert r["holdout"]["linear"] == seal.read_sealed_scores(cache, r["fit_key"])["linear"]
        assert r["holdout"]["linear"]
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


def test_under_estimate_no_estimate_before_the_lock_and_every_one_after_it(client):
    from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
    from turbotab.core.tests.stage_harness import NHANES
    from turbotab.core.tests.truths import fixture_truth
    from turbotab.server.tests.test_quest_api import ROLES, rule

    service = service_of(client)
    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    pid = drive.pid
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
    wait_for(client, pid, {"fit": "fresh", "effects": "fresh"}, timeout=240)
    [record] = settled(client, pid, 1)
    assert record["purpose"] == "inference" and record["fit_key"] == fit_key(client, pid)
    assert record["scores"] == {} and record["movement_why"].startswith("This fit is under Estimate")

    # Before the lock: no estimate in either document, as the stage routes serve none.
    served, effects = stage(client, pid, "fit"), stage(client, pid, "effects")
    assert served["withheld_for"] == "fit" and by_family(served)["linear"]["coefficients"] is None
    assert record["effects"] == {"linear": None} and record["withheld"] == served["withheld"]
    document = results(client, pid)
    assert document["estimand"] is None and served.get("estimand") is None
    assert document["families"][0]["effect"] is None and document["families"][0]["sequence"] == []
    assert effects["families"] == [] and document["withheld"] == served["withheld"]
    assert drive.view()["state"]["plan_locked"] is None  # read, withheld, nothing locked

    # Fit locks the plan and computes nothing new; then every estimate, the stage routes' own.
    assert client.post(f"/api/projects/{pid}/fit").status_code == 200
    lock = kinds(client, pid, "lock_plan")[-1]
    [record] = settled(client, pid, 1)
    assert record["plan_digest"] == lock["decision"]["digest"]
    text = json.dumps(lock["decision"]["plan"], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    assert record["plan_digest"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert service._shown_lock(pid) == lock["id"]  # an estimate served under the lock: it stands
    served, effects = stage(client, pid, "fit"), stage(client, pid, "effects")
    rows = by_family(served)["linear"]["coefficients"]
    assert rows and {r["feature"] for r in rows} == set(served["estimand"]["features"])
    assert record["effects"] == {"linear": rows} and record["withheld"] is None
    assert all(r["estimate"] is not None and r["ci_low"] < r["ci_high"] for r in rows)
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
