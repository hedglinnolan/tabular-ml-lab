"""Fit, pressed while the plan cannot be locked yet, over HTTP (calm/FOUNDATION §7; the front-end
audit's G05/V1 and the integrator's first-pass inbox line).

Under Estimate and Describe pressing Fit locks the analysis plan, and the open noticings are
triaged before the lock, so the server holds the press:

* while the open noticings' triage is not confirmed: refused, naming each open noticing and the way
  forward (the triage in Models, or its "Confirm all"), recording no press and no lock; once the
  triage is confirmed the press locks the plan, the dispositions in it;
* while the fit, or a stage it waits on, is refused: refused with that stage's own refusal, its
  words and its ways forward, recording no press and no lock; once nothing refuses it, the press
  locks the plan as before.

Under Predict nothing changes (its triage's gate is the held-out rows): the press is taken with the
triage open. An Estimate fit with nothing to triage is pressed as before.

What the triages hold rests on each table's construction, read here with pandas: NHANES's
`gender` and the dietary file's `sex` are two-level text, each a noticing of which level counts as
1. The refused fit is a refusal written here, served as the fit's status (the stage it names is
the one that refused, a stage the fit waits on): through the walk the server refuses an answer
that would leave the fit refused before the fit runs, so a fit refuses only where those checks
cannot see (SWARM_PLAN §8), which no fixture reaches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from turbotab.core import fit_press
from turbotab.core.decisions import parse_decision
from turbotab.core.graph import Refused
from turbotab.server.service import ProjectService
from turbotab.server.tests.conftest import DIETARY, wait_for
from turbotab.server.tests.test_quest_api import needs_nhanes
from turbotab.server.tests.test_seal_api import decide, sealed_project

CONFIRM = parse_decision({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                          "lines": []}).model_dump(mode="json")


def press(client, pid: str):
    return client.post(f"/api/projects/{pid}/fit")


def triage(client, pid: str) -> dict[str, Any]:
    return client.get(f"/api/projects/{pid}/triage").json()


def kinds(client, pid: str) -> list[str]:
    return [r["decision"]["kind"] for r in client.get(f"/api/projects/{pid}").json()["decisions"]]


def nothing_pressed_or_locked(client, pid: str, before: list[str]) -> None:
    service: ProjectService = client.app.state.service
    view = client.get(f"/api/projects/{pid}").json()
    assert [r["decision"]["kind"] for r in view["decisions"]] == before  # no lock recorded
    assert "lock_plan" not in before and view["state"]["plan_locked"] is None
    assert fit_press.read_press(service.workspace.project_dir(pid)) is None  # no press kept
    fit = client.get(f"/api/projects/{pid}/quest").json()["fit"]
    assert fit["pressed"] is False and fit["locked"] is False


def nhanes_under_estimate(client) -> Any:
    """The NHANES journey (sugar → fasting glucose, under Estimate) to a fresh fit, Fit not
    pressed, as ``test_fit_api`` walks it."""
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
    wait_for(client, drive.pid, {"fit": "fresh"}, timeout=240)
    return drive


def walking_under_estimate(client, folder: Path) -> str:
    """The Estimate fixture's table (``test_export_stage_refusal.walking_bp``) walked by its
    journey under Estimate, complete cases and the linear model, to a fresh fit, Fit not pressed."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.acceptance.server_drive import Drive
    from turbotab.core.tests.test_export_stage_refusal import FIXTURE, ROLES, walking_bp
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FIXTURE
    walking_bp().to_csv(path, index=False)
    spec = J.Journey("fit-press-holds", "clinical", "inference", "?", str(path), "test",
                     lambda: path, lambda: J.truth(dict(FIXTURE_TRUTHS[FIXTURE]), fixture=path.name),
                     target="sbp_mmhg", lenses=("clinical",), exposure="walks_daily",
                     roles=dict(ROLES),
                     answers={"missing": {"kind": "set_missing", "strategy": "complete_case"},
                              "models": {"kind": "select_models", "models": ["linear"]}})
    run = J.Run(spec)
    run.client = J.Recording(client, run)
    r = client.post("/api/projects", json={"path": str(path)})
    assert r.status_code == 200, r.text
    run.pid = r.json()["id"]
    truth = spec.truth()
    truth.guess, truth.ranked = J._server_guess(client, run.pid), J._server_ranked(client, run.pid)
    run.drive = Drive(run.client, run.pid, truth)
    run.drive.exposure = spec.exposure
    run.drive.artifact("ingest", timeout=600)
    assert J.follow(run), run.notes
    wait_for(client, run.pid, {"fit": "fresh"}, timeout=240)
    return run.pid


# ── the triage, before the lock ──────────────────────────────────────────────


@pytest.mark.slow
@needs_nhanes
def test_under_estimate_fit_waits_for_the_open_noticings_triage_and_locks_once_it_is_confirmed(
        client):
    from turbotab.core.tests.stage_harness import NHANES

    gender = pd.read_csv(NHANES, usecols=["gender"])["gender"]
    assert sorted(gender.dropna().unique()) == ["female", "male"]  # the construction: two-level text
    drive = nhanes_under_estimate(client)
    pid = drive.pid
    t = triage(client, pid)
    assert (t["stage"], t["blockers"], t["answered"]) == ("models", 0, False)
    still = [i for i in t["items"] if i["recorded"] is None]
    assert "binary_text__gender" in {i["id"] for i in still}
    before = kinds(client, pid)

    r = press(client, pid)
    assert r.status_code == 409, r.text[:600]
    error = r.json()["error"]
    assert error["code"] == "triage_open"
    n = len(still)
    named = "; ".join(f"{i['summary'].strip().rstrip('.')} ({i['label'].lower()})" for i in still)
    assert error["message"] == (
        f"Fit waits: pressing it locks the analysis plan, and the open noticings are triaged "
        f"before the plan is locked. {n} {'is' if n == 1 else 'are'} still open, each with its "
        f"recommended disposition: {named}. Triage {'it' if n == 1 else 'them'} in Models first, "
        f"so each is decided before any estimate is seen.")
    assert error["exits"] == [{"label": "Triage the open noticings in Models", "decision": None},
                              {"label": "Keep the recommended dispositions", "decision": CONFIRM}]
    nothing_pressed_or_locked(client, pid, before)
    assert client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]["withheld_for"] == "fit"

    # The way forward taken: "Confirm all", each as recommended; Fit then locks the plan with the
    # dispositions in it.
    confirmed = drive.post(error["exits"][1]["decision"])
    assert confirmed.status_code == 200, confirmed.text[:600]
    triaged = confirmed.json()["decisions"][-1]
    assert triaged["decision"]["kind"] == "confirm_sweep"
    assert {l["key"]: l["value"] for l in triaged["decision"]["lines"]} == {
        i["id"]: i["recommended"] for i in still}
    assert triage(client, pid)["answered"]
    taken = press(client, pid)
    assert taken.status_code == 200, taken.text[:600]
    assert taken.json()["locked"] is True
    view = drive.view()
    lock = view["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan" and view["state"]["plan_locked"] is True
    assert lock["decision"]["plan"]["triage"] == triaged["decision"]["lines"]
    service: ProjectService = client.app.state.service
    assert fit_press.read_press(service.workspace.project_dir(pid))["target"] == "glucose"


def test_under_estimate_with_nothing_to_triage_fit_locks_the_plan_as_before(client, tmp_path):
    pid = walking_under_estimate(client, tmp_path / "data")
    t = triage(client, pid)
    assert t["stage"] == "models" and t["items"] == [] and t["answered"]
    r = press(client, pid)
    assert r.status_code == 200, r.text[:600]
    assert r.json()["locked"] is True and r.json()["pressed"] is True
    assert kinds(client, pid)[-1] == "lock_plan"


def test_under_prediction_fit_is_taken_with_the_triage_open(client):
    """Under Predict the triage's gate is the held-out rows, not Fit: the press is taken while the
    triage at Results is open, and nothing locks."""
    sex = pd.read_csv(DIETARY, usecols=["sex"])["sex"]
    assert sorted(sex.dropna().unique()) == ["F", "M"]  # the construction: two-level text
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}").json()["state"]["purpose"] == "prediction"
    t = triage(client, pid)
    assert t["stage"] == "results" and not t["answered"]
    assert next(i for i in t["items"] if i["id"] == "binary_text__sex")["recorded"] is None
    before = kinds(client, pid)
    r = press(client, pid)
    assert r.status_code == 200, r.text[:600]
    assert r.json()["pressed"] is True and r.json()["locked"] is False
    assert kinds(client, pid) == before  # a job command, not a decision
    service: ProjectService = client.app.state.service
    assert fit_press.read_press(service.workspace.project_dir(pid))["target"] == "hba1c"


# ── a fit that cannot run ────────────────────────────────────────────────────


def test_under_estimate_a_refused_fit_refuses_the_press_with_its_own_refusal(client, tmp_path,
                                                                              monkeypatch):
    """A refusal written here served as the fit's status, as the engine serves a stage the fit
    waits on that refused (``graph.Engine._status``): the press is refused with its code, words and
    ways forward, and nothing is pressed or locked; the fit's own refusal likewise. With no refusal
    served, the same press locks the plan."""
    service: ProjectService = client.app.state.service
    pid = walking_under_estimate(client, tmp_path / "data")
    before = kinds(client, pid)
    real = service.engine.status
    for refusal in (
            Refused(stage="cohort", code="too_few_rows",
                    message="The eligibility rules written here leave too few rows to analyze.",
                    exits=[{"label": "Keep every row",
                            "decision": parse_decision({"kind": "set_exclusions", "rules": []})
                            .model_dump(mode="json")},
                           {"label": "Change the eligibility rules", "decision": None}]),
            Refused(stage="fit", code="singular_model_matrix",
                    message="The model matrix written here is singular.",
                    exits=[{"label": "Leave one of the dependent columns out of the model",
                            "decision": None}])):

        def refused(p: str, *args: Any, _r: Refused = refusal, **kwargs: Any) -> dict[str, Any]:
            stages = dict(real(p, *args, **kwargs))
            if p == pid:
                stages["fit"] = stages["fit"].model_copy(update={
                    "status": "error", "fresh": False, "refusal": _r,
                    "error": f"Needs {_r.stage!r}, which failed." if _r.stage != "fit" else
                             _r.message})
            return stages

        monkeypatch.setattr(service.engine, "status", refused)
        assert client.get(f"/api/projects/{pid}").json()["stages"]["fit"]["refusal"] == (
            refusal.model_dump(mode="json"))  # the premise, as the server serves it
        r = press(client, pid)
        assert r.status_code == 409, r.text[:600]
        assert r.json()["error"] == {"code": refusal.code, "message": refusal.message,
                                     "exits": refusal.exits}
        nothing_pressed_or_locked(client, pid, before)
        monkeypatch.undo()

    assert service.engine.status(pid)["fit"].refusal is None
    taken = press(client, pid)
    assert taken.status_code == 200, taken.text[:600]
    assert taken.json()["locked"] is True and kinds(client, pid)[-1] == "lock_plan"
