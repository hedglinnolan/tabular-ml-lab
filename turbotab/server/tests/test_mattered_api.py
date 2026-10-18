"""E3 over HTTP (SWARM_ENGINE_PLAN): ``GET /projects/{pid}/mattered`` on the NHANES journey under
Estimate (sugar and glucose, the drive of ``test_fit_lock_repair``).

* Before Fit locks the plan the curve is refused with the line saying when it opens, and the stage
  is blocked on the lock: nothing is fitted.
* After the lock it is served: its primary is the fit's own row of the exposure, as
  ``GET …/stages/fit`` serves it, and the lock it names is the lock record in force.
* A choice changed after the lock: the old curve is never served as current (it is withheld while
  a question the change reopened is open, as the estimates are); the curve is computed again, and
  its primary is the new fit's row.
* With the goal changed to Predict the curve is refused again, and the stage route withholds it.

Each expectation is read from the record or another route, never from the code under test.
"""
from __future__ import annotations

import pytest

from turbotab.core.methods.spec_curve import LOCK_EXIT, SEALED
from turbotab.core.tests.acceptance.server_drive import settle_forms
from turbotab.server.tests.conftest import wait_for
from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press, record_of


def mattered(client, pid: str):
    return client.get(f"/api/projects/{pid}/mattered")


def fit_row(client, pid: str, exposure: str = "sugar") -> dict:
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    model = next(m for m in fit["models"] if m["family"] == "linear")
    return next(r for r in model["coefficients"] if r["feature"] == exposure)


def fresh_curve(client, pid: str) -> dict:
    wait_for(client, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=300)
    r = mattered(client, pid)
    assert r.status_code == 200, r.text[:900]
    body = r.json()
    assert body["fresh"] is True and body["status"] == "fresh" and body["curve"], body
    return body


@pytest.mark.slow
def test_the_curve_is_refused_before_the_lock_and_served_after_it(client):
    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)

    # Before the lock: refused with the line saying when it opens; the stage waits on the lock.
    sealed = mattered(client, pid)
    assert sealed.status_code == 409, sealed.text
    error = sealed.json()["error"]
    assert error["code"] == "curve_sealed" and error["message"] == SEALED
    assert [e["label"] for e in error["exits"]] == [LOCK_EXIT["label"]]
    stage = client.get(f"/api/projects/{pid}").json()["stages"]["mattered"]
    assert stage["status"] == "blocked" and "plan_locked" in stage["missing"]

    # After the lock: the curve, its primary the reported estimate.
    report = press(client, pid)
    assert report["locked"] is True
    lock = record_of(client, pid, "lock_plan")
    body = fresh_curve(client, pid)
    curve = body["curve"]
    assert body["plan_lock"] == curve["plan_lock"] == lock["id"]
    assert curve["crossing"] == "one_at_a_time" and curve["sealed"] is None
    primary = next(s for s in curve["specs"] if s["primary"])
    row = fit_row(client, pid)
    assert primary["estimate"] == pytest.approx(row["estimate"], rel=1e-9)
    assert primary["low"] == pytest.approx(row["ci_low"], rel=1e-9)
    assert primary["high"] == pytest.approx(row["ci_high"], rel=1e-9)
    assert body["reported"]["estimate"] == row["estimate"]
    # One departure per recorded choice, each changing that choice alone.
    choices = [c["key"] for c in curve["choices"]]
    assert "exclusion:0" in choices and any(c.startswith("adjusted:") for c in choices)
    assert len(curve["specs"]) + len(curve["not_estimated"]) == 1 + len(choices)
    for s in curve["specs"]:
        changed = [k for k, v in s["picks"].items() if v != "recorded"]
        assert len(changed) == (0 if s["primary"] else 1), s["key"]
    served = client.get(f"/api/projects/{pid}/stages/mattered").json()["artifact"]
    assert served["curve"]["specs"] == curve["specs"] and served["curve"]["plan_lock"] == lock["id"]
    assert served.get("withheld") is None

    # A choice changed after the lock: never the old curve as current, then the new one.
    old_key = body["key"]
    drive.decide({"kind": "set_exclusions", "rules": [
        {"column": "kcal", "low": 600, "high": 4500, "reason": "implausible intake"}]})
    stale = mattered(client, pid)
    if stale.status_code == 409:
        # The new rows reopen a question the estimates rest on (the functional form, read on the
        # analyzed rows): the curve is withheld with it, as the estimates are.
        assert stale.json()["error"]["code"] == "curve_withheld", stale.text
        held = client.get(f"/api/projects/{pid}/stages/mattered").json()["artifact"]
        assert held["withheld"] == stale.json()["error"]["message"]
        assert held["curve"]["specs"] == [] and held["reported"] is None
    else:
        body = stale.json()
        assert (body["fresh"] is False and body["curve"] is None) or body["key"] != old_key
    settle_forms(drive)
    again = fresh_curve(client, pid)
    assert again["key"] != old_key
    primary = next(s for s in again["curve"]["specs"] if s["primary"])
    row = fit_row(client, pid)
    assert primary["estimate"] == pytest.approx(row["estimate"], rel=1e-9)
    rule = again["curve"]["choices"][0]
    assert rule["key"] == "exclusion:0" and "`kcal`" in rule["label"] and "600" in rule["label"]

    # The goal changed to Predict: refused, and the stage route withholds every specification.
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})
    gone = mattered(client, pid)
    assert gone.status_code == 409 and gone.json()["error"]["code"] == "curve_not_offered"
    withheld = client.get(f"/api/projects/{pid}/stages/mattered").json()["artifact"]
    assert withheld["withheld"] and withheld["curve"]["specs"] == []
    assert withheld["reported"] is None
