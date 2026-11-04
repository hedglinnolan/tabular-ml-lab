"""CH1: the curve of which decisions mattered computes live after the lock until a press of Fit can
release it (RECIPES_AND_TUNING §4.4; ``core/fit_press.py``; ``core/mattered.py``).

``fit_press.curve_holds`` judges the curve held over about 2 minutes until a press for the outcome
confirmed a curve estimate (the press's ``curve_seconds``). No press the product makes records one:
``ProjectService.press_fit`` keeps the outcome, the fit's estimate and the last record before it,
and nothing in the product shows the curve's cost before a press. So a scheduler's hold built on
``curve_holds`` alone would hold a long curve for good, with nothing serving why. Until a press
shows and records that cost, the server's hold (``ProjectService._hold``) holds nothing for the
curve's stage, as ``fit_press``'s module docstring says, and the curve computes once the fit is
made and the plan is locked.

Every expectation comes from outside the code under test:

* **the specifications**: the eight ``test_mattered.departures`` lists for its state (the primary,
  each of its two exclusion rules dropped, each of its four adjusted covariates dropped, and `bmi`
  added), by construction;
* **the seconds**: the least-squares family's ``fit_seconds`` the test writes into the fit's
  artifact, so the curve's estimate is their product, set by arithmetic on either side of
  ``fit_press.HOLD_SECONDS``;
* **the press** in the fast test is kept as ``press_fit`` keeps it (``fit_press.record_press``
  with no ``curve_seconds``); in the slow one it is the press the product makes, ``POST …/fit``;
* over HTTP, the stage's status as ``GET /projects/{pid}`` serves it, and the curve as
  ``GET /projects/{pid}/mattered`` serves it.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from turbotab.core import fit_press
from turbotab.core.graph import HoldView
from turbotab.core.tests import test_mattered as TM
from turbotab.server.tests.conftest import DIETARY

SPECIFICATIONS = len(TM.departures())  # 8, by test_mattered's construction


@pytest.fixture
def project(client) -> tuple[Any, str, Any]:
    """A project of its own (dietary_recalls.csv, opened fresh): the service, its id and the folder
    a press is kept in. The state the hold reads is the test's own (``HoldView``)."""
    r = client.post("/api/projects", json={"path": str(DIETARY)})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    service = client.app.state.service
    return service, pid, service.workspace.project_dir(pid)


def view(pid: str, state: Any, linear: float | None) -> HoldView:
    """The project as the hold reads it once the fit is made: ``state``, and the fit's artifact
    timing least squares at ``linear`` seconds (None: not fitted)."""
    models = [] if linear is None else [{"family": "linear", "fit_seconds": linear}]
    fit = {"models": models}
    return HoldView(pid, state, lambda name: fit if name == "fit" else None, lambda name: False)


def test_no_press_the_product_makes_can_release_the_curve_so_the_scheduler_holds_none(project):
    """Least squares at 16 s makes the eight specifications 128 s, over about 2 minutes, so
    ``curve_holds`` judges the curve held after a press that confirmed the fit's estimate alone,
    the only press ``press_fit`` keeps. The server's hold holds the curve's stage for none: not
    before a press, not after the press as the product keeps it, and not at 15 s (120 s, not over
    the threshold). At 3099fa4d it held ``"fit"`` over the threshold, before and after the press."""
    from turbotab.core.mattered import expected_seconds

    service, pid, folder = project
    st = TM.state(models=["linear"])
    over = 16.0
    assert over * SPECIFICATIONS == 128.0 > fit_press.HOLD_SECONDS
    assert expected_seconds(st, view(pid, st, over).artifact("fit")) == 128.0  # known, not None
    at = fit_press.HOLD_SECONDS / SPECIFICATIONS
    assert at == 15.0
    assert service._hold("mattered", view(pid, st, over)) is None
    assert service._hold("mattered", view(pid, st, at)) is None
    fit_press.record_press(folder, target="glucose", seconds=40.0, seq=3)
    kept = fit_press.read_press(folder)
    assert fit_press.curve_holds(over * SPECIFICATIONS, kept, "glucose") is True
    assert service._hold("mattered", view(pid, st, over)) is None
    assert service._hold("mattered", view(pid, st, at)) is None


def _mattered(client, pid: str) -> dict[str, Any]:
    return client.get(f"/api/projects/{pid}").json()["stages"]["mattered"]


@pytest.mark.slow
def test_after_a_press_of_fit_the_curve_is_computed_whatever_its_estimate(client, monkeypatch):
    """Through the server, under Estimate on the NHANES drive, with the hold's threshold at 0 s so
    the curve's measured estimate (``mattered.expected_seconds``) is over it: Fit is pressed
    through ``POST …/fit``, the press the product makes, which locks the plan. The curve's stage is
    never held, and the curve is computed and served for the lock. At 3099fa4d the stage stayed
    idle, held ``"fit"`` with no job, and ``GET …/mattered`` served no curve."""
    from turbotab.core import decisions, mattered
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press, record_of

    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    monkeypatch.setattr(fit_press, "HOLD_SECONDS", 0.0)
    service = client.app.state.service
    state = decisions.fold(service.log(pid).records())
    estimate = mattered.expected_seconds(state, service.engine.get(pid, "fit").artifact)
    assert estimate is not None and estimate > fit_press.HOLD_SECONDS
    assert press(client, pid)["locked"] is True
    end = time.monotonic() + 300
    while (status := _mattered(client, pid))["status"] != "fresh":
        assert status["held"] is None and status["status"] != "error", status
        assert time.monotonic() < end, status
        time.sleep(0.1)
    served = client.get(f"/api/projects/{pid}/mattered")
    assert served.status_code == 200, served.text[:600]
    body = served.json()
    assert body["fresh"] is True and body["curve"]["specs"], body
    assert body["plan_lock"] == record_of(client, pid, "lock_plan")["id"]
