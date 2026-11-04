"""CH1: the curve of which decisions mattered is held as Fit is held (RECIPES_AND_TUNING §4.4;
``core/fit_press.py``; ``core/mattered.py``).

The ``mattered`` stage fits the plan once per specification after the lock, so the scheduler's hold
(``ProjectService._hold``) holds it as it holds the fit: a curve whose estimate (the followed
family's fit seconds times the specifications, ``mattered.expected_seconds``) is over about 2
minutes (``fit_press.HOLD_SECONDS``) waits until a press of Fit for the outcome confirmed a curve
estimate it is within 1.5 times of (the press's ``curve_seconds``, ``fit_press.curve_holds``); a
curve under it computes live, as before.

Every expected value comes from outside the code under test:

* **the specifications**: the eight ``test_mattered.departures`` lists for its state (the primary,
  each of its two exclusion rules dropped, each of its four adjusted covariates dropped, and `bmi`
  added), by construction;
* **the seconds**: the least-squares family's ``fit_seconds`` the test writes into the fit's
  artifact (the family the curve follows for a numeric outcome), so the estimate is their product,
  set by arithmetic on either side of the threshold; a random forest beside it fits for 500 s, so
  a hold that summed the fit's families would differ;
* **the press** is the one the test keeps beside the project (``fit_press.record_press``), as a
  press of Fit keeps it;
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


def curve_state(**slots: Any) -> Any:
    return TM.state(models=["linear", "random_forest"], **slots)


def view(pid: str, state: Any, linear: float | None, forest: float = 500.0) -> HoldView:
    """The project as the hold reads it once the fit is made: ``state``, and the fit's artifact
    timing least squares at ``linear`` seconds (None: not fitted) and the forest at ``forest``."""
    models = [{"family": "random_forest", "fit_seconds": forest}]
    if linear is not None:
        models.insert(0, {"family": "linear", "fit_seconds": linear})
    fit = {"models": models}
    return HoldView(pid, state, lambda name: fit if name == "fit" else None, lambda name: False)


def test_a_curve_expected_over_about_two_minutes_waits_until_a_press_confirms_it(project):
    """Least squares fits in 16 s, so the eight specifications take 128 s, over about 2 minutes:
    with no press the curve waits for Fit (held ``"fit"``, as the fit is); a press that confirmed
    the fit's estimate alone (as the lock's press keeps it today) releases nothing, nor one for
    another outcome; a press for the outcome that confirmed 128 s releases it. It holds again when
    its estimate grows past 1.5 times the one confirmed (192 s), and when the press is withdrawn.
    On the base the server's hold held nothing for this stage, so the curve computed live."""
    service, pid, folder = project
    st = curve_state()
    seconds = 16.0
    estimate = seconds * SPECIFICATIONS
    assert estimate == 128.0 > fit_press.HOLD_SECONDS
    assert service._hold("mattered", view(pid, st, seconds)) == "fit"
    fit_press.record_press(folder, target="glucose", seconds=40.0, seq=3)
    assert service._hold("mattered", view(pid, st, seconds)) == "fit"
    fit_press.record_press(folder, target="insulin", seconds=40.0, seq=3, curve_seconds=estimate)
    assert service._hold("mattered", view(pid, st, seconds)) == "fit"
    fit_press.record_press(folder, target="glucose", seconds=40.0, seq=3, curve_seconds=estimate)
    assert service._hold("mattered", view(pid, st, seconds)) is None
    # within 1.5 times the confirmed 128 s (8 x 24 = 192 s) it stays released; past it, held
    assert service._hold("mattered", view(pid, st, 24.0)) is None
    assert service._hold("mattered", view(pid, st, 24.5)) == "fit"
    fit_press.withdraw_press(folder)
    assert service._hold("mattered", view(pid, st, seconds)) == "fit"


def test_a_curve_expected_within_about_two_minutes_computes_live(project):
    """No press is kept. Least squares at 15 s makes the eight specifications 120 s, about 2
    minutes and not over it; at 1 s, 8 s, with the forest's 500 s beside it, which the curve does
    not refit: neither holds the curve. Nor does a curve the goal does not offer (Predict), or a
    fit that times no family the curve follows: an estimate not known holds nothing."""
    service, pid, _ = project
    st = curve_state()
    at = fit_press.HOLD_SECONDS / SPECIFICATIONS
    assert at == 15.0 and at * SPECIFICATIONS == fit_press.HOLD_SECONDS
    assert service._hold("mattered", view(pid, st, at)) is None
    assert service._hold("mattered", view(pid, st, 1.0)) is None
    predict = st.model_copy(update={"purpose": "prediction"})
    assert service._hold("mattered", view(pid, predict, 16.0)) is None
    assert service._hold("mattered", view(pid, st, None)) is None


def _mattered(client, pid: str) -> dict[str, Any]:
    return client.get(f"/api/projects/{pid}").json()["stages"]["mattered"]


@pytest.mark.slow
def test_after_the_lock_a_long_curve_waits_in_the_scheduler_until_a_press_confirms_it(
        client, monkeypatch):
    """Through the server, under Estimate on the NHANES drive: every measured curve is a long one
    (the threshold at 0 s). Fit is pressed, which locks the plan and confirms the fit's estimate
    alone, so the curve's stage waits in the scheduler (held ``"fit"``, no job) and
    ``GET …/mattered`` serves no curve. A press kept for the outcome that confirmed the curve's
    estimate releases it: the curve is computed and served. On the base it computed at once."""
    from turbotab.core import decisions, mattered
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press

    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    monkeypatch.setattr(fit_press, "HOLD_SECONDS", 0.0)
    assert press(client, pid)["locked"] is True
    end = time.monotonic() + 120
    while (status := _mattered(client, pid))["held"] != "fit":
        assert status["status"] in ("blocked", "idle", "stale"), status  # never started
        assert time.monotonic() < end, status
        time.sleep(0.1)
    time.sleep(1.0)  # held, not merely slow to start
    status = _mattered(client, pid)
    assert status["status"] in ("idle", "stale") and status["held"] == "fit", status
    assert status["job_id"] is None
    served = client.get(f"/api/projects/{pid}/mattered")
    assert served.status_code == 200, served.text[:600]
    assert served.json()["curve"] is None and served.json()["fresh"] is False

    service = client.app.state.service
    folder = service.workspace.project_dir(pid)
    kept = fit_press.read_press(folder)
    assert kept is not None and "curve_seconds" not in kept  # the lock's press: the fit's alone
    state = decisions.fold(service.log(pid).records())
    estimate = mattered.expected_seconds(state, service.engine.get(pid, "fit").artifact)
    assert estimate is not None and estimate > 0
    fit_press.record_press(folder, target=kept["target"], seconds=kept["seconds"],
                           seq=kept["seq"], curve_seconds=estimate)
    service.engine.on_decision(pid)
    end = time.monotonic() + 240
    while (status := _mattered(client, pid))["status"] != "fresh":
        assert status["held"] is None and status["status"] != "error", status
        assert time.monotonic() < end, status
        time.sleep(0.1)
    body = client.get(f"/api/projects/{pid}/mattered").json()
    assert body["fresh"] is True and body["curve"]["specs"], body
