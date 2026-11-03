"""Captures Estimate Results' fixture (front-end audit G02 v0, G11, G18): what the real server
serves on a light time-to-event walk under Estimate, before the plan's lock, once Fit is pressed,
and once the failed diagnostic's first way forward is recorded.

The walk is journey B of ``turbotab/core/tests/acceptance/test_estimand_chain.py``, its rows, its
fixture's truth and its answers imported from there: 600 rows whose hazard ratio for ``x1`` changes
over follow-up (log hazard ratio 1.0 before t = 0.5 and -0.3 after), the clinical lens, the event
``event`` with ``followup_years`` as its follow-up, the hazard ratio of ``x1`` as the estimand, the
adjustment answered from the truth, the Cox model; all through the real server in process
(``server_drive.local_server``). Three snapshots:

* ``before``: the models chosen and every stage computed, Fit not pressed, so the plan is not
  locked and every estimate is withheld (FOUNDATION §5 rule 6);
* ``locked``: Fit pressed, which locks the plan; the effects stage serves the declared sequence and
  the proportional-hazards check, failed for ``x1``, with its ways forward; GET /plan beside it;
* ``responded``: the check's first way forward (``respond_diagnostic``) recorded, as a person
  taking it does, and the effects stage computed again with the response.

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_estimate.py

writes ``cox-estimate.json`` beside this file. Nothing in it is written by hand: a project view
keeps only the state the Results read (``meta.view_state``), the record and the stages, and every
other served value is kept exactly as served. The run stops itself after :data:`LIMIT` seconds.
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

from turbotab.core.tests.acceptance import test_estimand_chain as B  # noqa: E402
from turbotab.core.tests.acceptance.server_drive import local_server, open_project, press_fit  # noqa: E402
from turbotab.core.tests.truths import Truth, answer_adjustment  # noqa: E402

OUT = Path(__file__).with_name("cox-estimate.json")
LIMIT = 540  # seconds: a local run stays under ten minutes
# Journey B's fixture truth and roles (test_estimand_chain.py's ``cox`` fixture).
TRUTH = {"code_or_count:x2": "amount", "code_or_count:activity": "amount",
         "code_or_count:pid": "amount", "adjust:x2": "yes,yes,no",
         "adjust:grp": "yes,yes,no", "adjust:activity": "no,yes,no"}
ROLES = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "grp": "covariate",
         "activity": "covariate", "followup_years": "time"}
# The view's parts the Results read: the folded state's goal, outcome and lock, the record, the
# stages.
STATE_KEYS = ("target", "purpose", "task", "plan_locked", "estimand", "diagnostic_responses")


def _slim_view(view: dict[str, Any]) -> dict[str, Any]:
    state = view.get("state") or {}
    return {"id": view.get("id"), "state": {k: state.get(k) for k in STATE_KEYS if k in state},
            "decisions": view.get("decisions") or [], "stages": view.get("stages") or {}}


def _stop(*_: Any) -> None:
    raise TimeoutError(f"the capture ran past {LIMIT} s")


def main() -> int:
    t0 = time.monotonic()
    signal.signal(signal.SIGALRM, _stop)
    signal.alarm(LIMIT)
    out: dict[str, Any] = {}
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="estimate-capture-") as folder:
        csv = Path(folder) / "cohort.csv"
        B._cohort().to_csv(csv, index=False)
        truth = Truth(dict(TRUTH), fixture="the time-to-event chain")
        with local_server(Path(folder) / "home") as client:
            drive = open_project(client, csv, truth)
            pid = drive.pid
            get = lambda path: client.get(f"/api/projects/{pid}{path}").json()  # noqa: E731

            def snapshot() -> dict[str, Any]:
                return {"view": _slim_view(get("")), "quest": get("/quest"),
                        "stages": {"effects": get("/stages/effects")}}

            def settled(until: Any, timeout: float = 300.0) -> None:
                """Wait, without pressing Fit, until the effects stage is fresh and ``until``
                holds of its served artifact."""
                end = time.monotonic() + timeout
                while True:
                    status = drive.view()["stages"]["effects"]
                    if status["status"] == "fresh" and until(get("/stages/effects")["artifact"]):
                        return
                    assert status["status"] != "error", status
                    assert time.monotonic() < end, f"effects never settled: {status}"
                    time.sleep(0.1)

            B._open(drive, "clinical", "event", "1")
            assert drive.reach("follow_up")["status"] == "open"
            drive.decide({"kind": "set_task", "column": "event", "task": "time_to_event"})
            drive.decide({"kind": "set_follow_up", "column": "event",
                          "time_column": "followup_years"})
            B._seal(drive, ROLES)
            drive.reach("estimand")
            drive.decide({"kind": "set_estimand", "exposure": "x1", "measure": "hazard_ratio"})
            drive.reach("adjustment")
            answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], truth)
            B._fresh(drive, "proposals", lambda p: "activity" in (p.get("adjustment") or {}).get(
                "answered", {}))
            drive.answer("models", {"kind": "select_models", "models": ["cox"]})
            settled(lambda a: True)
            out["before"] = snapshot()
            assert not out["before"]["view"]["state"].get("plan_locked"), "locked before Fit"
            notes.append(f"every stage computed, Fit not pressed ({round(time.monotonic() - t0)} s)")

            assert press_fit(client, pid), "Fit was refused"
            settled(lambda a: bool(a.get("families")))
            out["locked"] = {**snapshot(), "plan": get("/plan")}
            notes.append(f"Fit pressed, the plan locked ({round(time.monotonic() - t0)} s)")

            [check] = out["locked"]["stages"]["effects"]["artifact"]["families"][0]["diagnostics"]
            assert check["status"] == "failed" and check["exits"], check
            taken = check["exits"][0]["decision"]
            drive.decide(taken)
            settled(lambda a: a["families"][0]["diagnostics"][0]["response"] is not None)
            out["responded"] = {**snapshot(), "taken": taken}
            notes.append(f"the check's first way forward recorded ({round(time.monotonic() - t0)} s)")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()
    out = {"meta": {"journey": "estimate-cox", "source": "test_estimand_chain.py journey B",
                    "captured": date.today().isoformat(), "head": head,
                    "seconds": round(time.monotonic() - t0),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture_estimate.py",
                    "view_state": list(STATE_KEYS), "notes": notes}, **out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) in {out['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
