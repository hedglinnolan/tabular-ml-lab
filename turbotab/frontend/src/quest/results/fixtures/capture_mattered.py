"""Captures the fixture of Estimate Results' "Which of my decisions mattered?" (front-end audit
G10; the owner's Q10): what the real server serves for the specification curve (the ``mattered``
stage) and the materiality ledger, and what the stage itself makes where the curve is refused.

* ``nhanes``: the materiality journey of ``turbotab/server/tests/test_materiality_api.py`` (sugar
  and fasting glucose under Estimate, a 500–5,000 kcal screen with the every-row analysis declared
  beside it, the linear model), its drive imported from there, through the real server in process
  (``server_drive.local_server``). Two snapshots: ``before``, every stage computed and Fit not
  pressed (the plan not locked, so the stage waits on the lock and nothing is fitted), with GET
  /mattered's refusal beside it; ``locked``, the triage confirmed and Fit pressed, which locks the
  plan, once the effects, sensitivity and ``mattered`` stages are fresh, with GET /mattered and
  the fit's own row of `sugar` beside it. One specification of its curve is not estimated: the
  departure adding `hdl` waits for that column's code-or-amount reading.
* ``cox``: journey B of ``turbotab/core/tests/acceptance/test_estimand_chain.py`` (a time to an
  event, the Cox model), its rows and answers imported from there, once Fit is pressed: the
  ``mattered`` stage as served (withheld: the refit does not follow a time to an event) and GET
  /mattered's refusal beside it.
* ``refused``: the ``mattered`` stage's artifact for an exposure family, which the curve refuses
  with a way forward, run through the real design and fit stages on
  ``turbotab/core/tests/test_mattered.py``'s cohort (its ``Run`` and that test's family state).
  With the gate clear the server serves such an artifact unchanged (``service._serve``:
  ``mattered.with_lock`` names the lock on a curve, and a refusal has none).
* ``binary`` (its own file, ``mattered-binary.json``): the ``mattered`` stage's artifact under a
  yes/no outcome and the odds ratio, the logistic curve of that test's ``binary`` run, as served
  with the gate clear (``mattered.with_lock``, the ``nhanes`` capture's lock named), and the fit's
  own row of the exposure the curve follows (its odds ratio, as Table 2 reports it).

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_mattered.py

writes ``mattered-estimate.json`` and ``mattered-binary.json`` beside this file; with ``--binary``
it writes only the second, naming the lock ``mattered-estimate.json`` holds, and runs no server.
Nothing in either is written by hand: a project view keeps only the state the Results read
(``capture_estimate._slim_view``), the record and the stages, and every other served value is kept
exactly as served. The run stops itself after :data:`LIMIT` seconds.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # capture_estimate, beside this file

import capture_estimate as CE  # noqa: E402
from turbotab.core import decisions as d  # noqa: E402
from turbotab.core import mattered as M  # noqa: E402
from turbotab.core.tests import test_mattered as TM  # noqa: E402
from turbotab.core.tests.acceptance import estimand_fixtures as ef  # noqa: E402
from turbotab.core.tests.acceptance import test_estimand_chain as B  # noqa: E402
from turbotab.core.tests.acceptance.server_drive import (  # noqa: E402
    local_server, open_project, press_fit, release)
from turbotab.core.tests.truths import Truth, answer_adjustment  # noqa: E402
from turbotab.server.tests import test_materiality_api as MA  # noqa: E402
from turbotab.server.tests.test_confirm_sweep_api import post  # noqa: E402
from turbotab.server.tests.test_quest_api import settle  # noqa: E402

OUT = Path(__file__).with_name("mattered-estimate.json")
BINARY = Path(__file__).with_name("mattered-binary.json")
LIMIT = 540  # seconds: a local run stays under ten minutes
STAGES = ("effects", "mattered")  # the stages whose served results the Results read


def _stop(*_: Any) -> None:
    raise TimeoutError(f"the capture ran past {LIMIT} s")


def _fresh(drive: Any, stages: tuple[str, ...], timeout: float = 300.0) -> None:
    """Wait until every one of ``stages`` is fresh, releasing a stage held for Fit as the user
    does."""
    end = time.monotonic() + timeout
    while True:
        view = drive.view()["stages"]
        for name in stages:
            release(drive.c, drive.pid, view[name])
        if all(view[name]["status"] == "fresh" for name in stages):
            return
        assert not any(view[name]["status"] == "error" for name in stages), view
        assert time.monotonic() < end, {n: view[n]["status"] for n in stages}
        time.sleep(0.1)


def _snapshot(client: Any, pid: str) -> dict[str, Any]:
    get = lambda path: client.get(f"/api/projects/{pid}{path}").json()  # noqa: E731
    return {"view": CE._slim_view(get("")), "quest": get("/quest"),
            "stages": {s: get(f"/stages/{s}") for s in STAGES},
            "materiality": get("/materiality")}


def _route(client: Any, pid: str) -> dict[str, Any]:
    r = client.get(f"/api/projects/{pid}/mattered")
    return {"status": r.status_code, "body": r.json()}


def nhanes(folder: Path, notes: list[str], t0: float) -> dict[str, Any]:
    with local_server(folder / "home") as client:
        drive = MA.journey(client)  # settled before Fit
        out: dict[str, Any] = {"before": {**_snapshot(client, drive.pid),
                                          "route": _route(client, drive.pid)}}
        assert out["before"]["view"]["stages"]["mattered"]["status"] == "blocked"
        assert out["before"]["route"]["status"] == 409, out["before"]["route"]
        notes.append(f"nhanes: every stage computed, Fit not pressed ({round(time.monotonic() - t0)} s)")
        r = post(drive, {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"})
        assert r.status_code == 200, r.text[:600]
        assert drive.press_fit(), "Fit was refused"
        _fresh(drive, ("effects", "sensitivity", "mattered"))
        settle(drive)
        fit = client.get(f"/api/projects/{drive.pid}/stages/fit").json()["artifact"]
        linear = next(m for m in fit["models"] if m["family"] == "linear")
        out["locked"] = {**_snapshot(client, drive.pid), "route": _route(client, drive.pid),
                         "fit_row": next(r for r in linear["coefficients"]
                                         if r["feature"] == "sugar")}
        assert out["locked"]["quest"]["fit"]["locked"], "not locked after Fit"
        assert out["locked"]["route"]["status"] == 200, out["locked"]["route"]
        notes.append(f"nhanes: Fit pressed, the plan locked, the curve fresh "
                     f"({round(time.monotonic() - t0)} s)")
        return out


def cox(folder: Path, notes: list[str], t0: float) -> dict[str, Any]:
    csv = folder / "cohort.csv"
    B._cohort().to_csv(csv, index=False)
    truth = Truth(dict(CE.TRUTH), fixture="the time-to-event chain")
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, truth)
        B._open(drive, "clinical", "event", "1")
        assert drive.reach("follow_up")["status"] == "open"
        drive.decide({"kind": "set_task", "column": "event", "task": "time_to_event"})
        drive.decide({"kind": "set_follow_up", "column": "event", "time_column": "followup_years"})
        B._seal(drive, CE.ROLES)
        drive.reach("estimand")
        drive.decide({"kind": "set_estimand", "exposure": "x1", "measure": "hazard_ratio"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], truth)
        drive.answer("models", {"kind": "select_models", "models": ["cox"]})
        settle(drive)
        assert press_fit(client, drive.pid), "Fit was refused"
        _fresh(drive, ("effects", "mattered"))
        out = {"locked": {**_snapshot(client, drive.pid), "route": _route(client, drive.pid)}}
        assert out["locked"]["route"]["status"] == 409, out["locked"]["route"]
        notes.append(f"cox: Fit pressed, the plan locked, the curve's stage fresh "
                     f"({round(time.monotonic() - t0)} s)")
        return out


def refused(folder: Path, lock_id: str, notes: list[str], t0: float) -> dict[str, Any]:
    numeric = TM.Run(TM.cohort(), folder, TM.state())
    roles = {**ef.ROLES, "supplement": "exposure"}
    family = numeric.state.model_copy(update={
        "roles": roles, "role_confirmations": dict(roles),
        "estimand": d.EstimandSpec(family=True, measure="mean_difference",
                                   multiplicity="fdr_bh")})
    art = numeric.mattered(family)
    assert art["curve"] is None and art["refused"]["exits"], art
    served = M.with_lock(art, lock_id)
    assert served == art, "a refusal is served unchanged"
    notes.append(f"refused: the exposure family's curve ({round(time.monotonic() - t0)} s)")
    return served


def binary(folder: Path, lock_id: str, notes: list[str], t0: float) -> dict[str, Any]:
    run = TM.Run(TM.cohort(), folder, TM.state(target="dm", task="binary", measure="odds_ratio",
                                               event="yes"))
    art = run.mattered()
    assert art["refused"] is None and art["curve"]["specs"], art
    served = M.with_lock(art, lock_id)
    reported = art["reported"]
    model = next(m for m in run.fit.data["models"] if m["family"] == reported["family"])
    row = next(r for r in model["coefficients"] if r["feature"] == reported["feature"])
    notes.append(f"binary: the logistic curve under the odds ratio ({round(time.monotonic() - t0)} s)")
    return {"artifact": served, "fit_row": row}


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()


def _write(path: Path, meta: dict[str, Any], out: dict[str, Any]) -> None:
    out = {"meta": {**meta, "captured": date.today().isoformat(), "head": _head(),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture_mattered.py"},
           **out}
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size:,} bytes) in {meta['seconds']} s")


def main(argv: list[str]) -> int:
    t0 = time.monotonic()
    signal.signal(signal.SIGALRM, _stop)
    signal.alarm(LIMIT)
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mattered-capture-") as folder:
        root = Path(folder)
        if "--binary" in argv:
            lock_id = json.loads(OUT.read_text())["nhanes"]["locked"]["route"]["body"]["plan_lock"]
        else:
            (root / "nhanes").mkdir()
            (root / "cox").mkdir()
            out: dict[str, Any] = {"nhanes": nhanes(root / "nhanes", notes, t0),
                                   "cox": cox(root / "cox", notes, t0)}
            lock_id = out["nhanes"]["locked"]["route"]["body"]["plan_lock"]
            out["refused"] = refused(root / "engine", lock_id, notes, t0)
            _write(OUT, {"journey": "estimate-mattered",
                         "source": "test_materiality_api.py's journey, test_estimand_chain.py "
                                   "journey B, test_mattered.py's cohort",
                         "seconds": round(time.monotonic() - t0),
                         "view_state": list(CE.STATE_KEYS), "notes": notes}, out)
            notes = []
        t1 = time.monotonic()
        part = binary(root / "binary", lock_id, notes, t1)
    _write(BINARY, {"journey": "estimate-mattered-binary",
                    "source": "test_mattered.py's cohort and its binary run",
                    "seconds": round(time.monotonic() - t1), "notes": notes}, {"binary": part})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
