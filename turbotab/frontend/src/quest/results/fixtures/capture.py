"""Captures Predict Results' fixture (W1 part 2, W1-G): what the real server serves on a light
NHANES walk under Predict, before the opening and after it.

The walk is the light reference walk of ``turbotab/core/tests/test_w1_press_fit_references.py``:
the NHANES export (fasting glucose), seven predictors, every row with glucose recorded, a random
20% held out, the linear model and ridge, Fit pressed when the Router reaches it, all through the
real server in process (``server_drive.local_server``). Where that walk names the final model in
its answer to the seal, this one walks the seal's chain of refusals as a person pressing its exits
does: the opening with no family is posted, and each refusal's first way forward that records a
decision is taken (the one naming the linear model, where the refusal asks for the final model).
An exit that is itself the opening is the next attempt; after any other, the last attempt is
posted again. Once the fit is served (the sealed snapshot), the explanations are asked for with
the engine's defaults (``set_explain`` with no field set); they read the training rows only.

The run stops itself after :data:`LIMIT` seconds, writing what was served so far beside the
fixture (``nhanes-predict.partial.json``).

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture.py

writes ``nhanes-predict.json`` beside this file. Nothing in it is written by hand. Large lists the
Results do not read are dropped (``meta.dropped`` names each path), a project view keeps only the
state the Results read (``meta.view_state``), the record and the stages, and a step of the chain
the server recorded keeps the record it made and the state after it; every other served value is
kept exactly as served.
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`

from turbotab.core.reference import journeys as J  # noqa: E402
from turbotab.core.tests.acceptance.server_drive import Drive, local_server  # noqa: E402

OUT = Path(__file__).with_name("nhanes-predict.json")
LIMIT = 570  # seconds: a local run stays under ten minutes
PREDICTORS = ("age", "gender", "bmi", "waist", "bp_sys", "hdl", "triglycerides")
# The explain artifact's per-row lists, which the Results do not draw (the beeswarm, the per-row
# attributions, the explained rows' ids).
DROPPED = ("artifact.families[].beeswarm", "artifact.families[].observations", "artifact.row_ids")
# The view's parts the Results read: the folded state's seal and outcome, the record, the stages.
STATE_KEYS = ("target", "purpose", "task", "seal_opened", "models", "explain", "updating")


def _slim_view(view: dict[str, Any]) -> dict[str, Any]:
    state = view.get("state") or {}
    return {"id": view.get("id"), "state": {k: state.get(k) for k in STATE_KEYS if k in state},
            "decisions": view.get("decisions") or [], "stages": view.get("stages") or {}}


def _trim_explain(served: dict[str, Any]) -> dict[str, Any]:
    art = served.get("artifact")
    if not isinstance(art, dict):
        return served
    art = {k: v for k, v in art.items() if k != "row_ids"}
    art["families"] = [{k: v for k, v in f.items() if k not in ("beeswarm", "observations")}
                       for f in art.get("families") or []]
    return {**served, "artifact": art}


def _spec(columns: list[str], hook: Any) -> Any:
    roles = {c: ("identifier" if c == "SEQN" else "covariate" if c in PREDICTORS else "excluded")
             for c in columns if c != "glucose"}
    return J.Journey(
        "w1g-results", "clinical", "prediction",
        "How well do age, sex and body measures predict fasting glucose?",
        J.NHANES_FILE, J.NHANES_WHY, J.nhanes_path,
        lambda: J.truth(J.nhanes_readings(), fixture=J.NHANES_KEY),
        target="glucose", lenses=("clinical",), roles=roles,
        answers={"exclusions": {"kind": "set_exclusions", "rules": []},
                 "missing": {"kind": "set_missing", "strategy": "complete_case"},
                 "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
                 "models": {"kind": "select_models", "models": ["linear", "ridge"]}},
        before={"open_seal": hook}, timeout=540.0)


def _stop(*_: Any) -> None:
    raise TimeoutError(f"the capture ran past {LIMIT} s")


def main() -> int:
    t0 = time.monotonic()
    signal.signal(signal.SIGALRM, _stop)
    signal.alarm(LIMIT)
    out: dict[str, Any] = {}

    def snapshot(run: Any) -> dict[str, Any]:
        c, pid = run.client, run.pid
        get = lambda path: c.get(f"/api/projects/{pid}{path}").json()  # noqa: E731
        return {"view": _slim_view(get("")), "quest": get("/quest"), "triage": get("/triage"),
                "stages": {"fit": get("/stages/fit"), "evaluation": get("/stages/evaluation"),
                           "explain": _trim_explain(get("/stages/explain")),
                           "split": get("/stages/split")}}

    def wait_fresh(run: Any, *stages: str, timeout: float = 420.0) -> None:
        for stage in stages:
            run.drive.artifact(stage, timeout=timeout)

    def at_the_seal(run: Any, view: dict[str, Any]) -> None:
        c, pid = run.client, run.pid
        post = lambda body: c.post(f"/api/projects/{pid}/decisions", json=body)  # noqa: E731
        wait_fresh(run, "fit")
        run.note(f"the fit is served ({round(time.monotonic() - t0)} s)")
        out["sealed"] = snapshot(run)
        # The explain offer, with the engine's defaults: the explanations read the training rows
        # only, so they compute beside the opening.
        r = post({"kind": "set_explain"})
        assert r.status_code == 200, r.text[:600]
        out["explain_posted"] = {"posted": {"kind": "set_explain"}, "status": r.status_code}
        chain: list[dict[str, Any]] = []

        def step(body: dict[str, Any]) -> Any:
            r = post(body)
            entry: dict[str, Any] = {"posted": body, "status": r.status_code}
            if r.status_code == 200:  # the record it made, and the folded state after it
                served = _slim_view(r.json())
                entry["response"] = {"record": served["decisions"][-1], "state": served["state"]}
            else:
                entry["response"] = r.json()
            chain.append(entry)
            out["chain"] = chain
            return r

        # The opening, as a person pressing the seal's exits retries it: an exit that is itself
        # the opening (naming the final model) is the next attempt; any other exit is recorded
        # and the last attempt is posted again.
        attempt: dict[str, Any] = {"kind": "open_seal"}
        r = step(attempt)
        for _ in range(40):  # NHANES: the final model, the triage, then each noticing in turn
            if r.status_code == 200:
                break
            exits = [e for e in r.json()["error"]["exits"] if e.get("decision")]
            assert exits, r.text[:600]
            named = [e for e in exits if (e["decision"] or {}).get("family") == "linear"]
            decision = (named or exits)[0]["decision"]
            if decision.get("kind") == "open_seal":
                attempt = decision
                r = step(attempt)
            else:
                taken = step(decision)
                assert taken.status_code == 200, taken.text[:600]
                r = step(attempt)
        assert r.status_code == 200, json.dumps(chain)[:3000]
        run.note(f"the held-out rows are open ({round(time.monotonic() - t0)} s)")
        wait_fresh(run, "fit", "explain", "evaluation", timeout=480.0)
        out["opened"] = snapshot(run)
        run.note(f"every Results stage is served ({round(time.monotonic() - t0)} s)")

    path = J.nhanes_path()
    spec = _spec(list(pd.read_csv(path, nrows=1).columns), at_the_seal)
    run = J.Run(spec)
    with tempfile.TemporaryDirectory(prefix="w1g-capture-") as home, local_server(Path(home)) as client:
        run.client = J.Recording(client, run)
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        run.pid = r.json()["id"]
        truth = spec.truth()
        truth.guess = J._server_guess(client, run.pid)
        truth.ranked = J._server_ranked(client, run.pid)
        run.drive = Drive(run.client, run.pid, truth)
        run.drive.artifact("ingest", timeout=300)
        try:
            followed = J.follow(run)
        finally:
            if "opened" not in out:  # what was served before the stop, for the reader
                OUT.with_suffix(".partial.json").write_text(json.dumps(out, ensure_ascii=False))
        assert followed and "opened" in out, run.notes
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()
    out = {"meta": {"journey": spec.name, "question": spec.question, "source": "NHANES export",
                    "captured": date.today().isoformat(), "head": head,
                    "seconds": round(time.monotonic() - t0),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture.py",
                    "dropped": list(DROPPED), "view_state": list(STATE_KEYS),
                    "log": run.log, "notes": run.notes}, **out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) in {out['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
