"""Fit, pressed while the plan cannot be locked yet, over HTTP (calm/FOUNDATION §7; the front-end
audit's G05/V1 and the integrator's first-pass inbox line).

Under Estimate and Describe pressing Fit locks the analysis plan, and the open noticings are
triaged before the lock, so the server holds the press:

* while the open noticings' triage is not confirmed: refused, naming each open noticing and the way
  forward (the triage in Models, or its "Confirm all"), recording no press and no lock;
* while a noticing the triage recorded "act on it" is not acted on (the owner's Q8): refused,
  naming it, where it is acted on and the way forward that changes its disposition (recorded as a
  limitation), recording no press and no lock; once its disposition is changed, or once it is acted
  on (NHANES's SAS zeros read as zero by their repair, so the triage no longer lists them), the
  press locks the plan, the dispositions in it;
* while the fit, or a stage it waits on, is refused: refused with that stage's own refusal, its
  words and its ways forward, recording no press and no lock; once nothing refuses it, the press
  locks the plan as before.

Under Predict nothing changes (its triage's gate is the held-out rows): the press is taken with the
triage open, and with the fit refused. An Estimate fit with nothing to triage is pressed as before.

What the triages hold rests on each table's construction, read here with pandas: NHANES's
`gender` and the dietary file's `sex` are two-level text, each a noticing of which level counts as
1, and NHANES's misread SAS transport zeros (cells of about 5.4e-79) a finding with a repair, acted
on in Your data. The refused fit is a refusal written here, served as the fit's status (the stage it names is
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
    # each open noticing in a sentence of its own, numbered when there are several, with its
    # recommended disposition after it, as the triage serves them
    n = len(still)
    said = [f"{i['summary'].strip().rstrip('.')}. Recommended: {i['label'].lower()}."
            for i in still]
    listed = (f"1 noticing is still open: {said[0]}" if n == 1 else
              f"{n} noticings are still open. "
              + " ".join(f"({k}) {e}" for k, e in enumerate(said, 1)))
    assert error["message"] == (
        f"Fit waits: pressing it locks the analysis plan, and the open noticings are triaged "
        f"before the plan is locked. {listed} Triage {'it' if n == 1 else 'them'} in Models "
        f"first, so {'it is' if n == 1 else 'each is'} decided before any estimate is seen.")
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
    kept = triage(client, pid)
    assert kept["answered"]

    # Kept is not acted on (the owner's Q8): the SAS zeros, recorded "act on it", hold the press
    # until they are acted on or their disposition is changed.
    to_act = sas_zeros_kept_to_act_on(kept)
    before = kinds(client, pid)
    held = press(client, pid)
    assert held.status_code == 409, held.text[:600]
    error = held.json()["error"]
    assert error["code"] == "triage_open"
    summary = to_act["summary"]
    assert error["message"] == (
        f"Fit waits: pressing it locks the analysis plan, and 1 noticing marked \"act on it\" is "
        f"not acted on yet: {summary.strip().rstrip('.')}. Act on it where it is decided, or change "
        f"its disposition by recording it as a limitation, before any estimate is seen.")
    change = parse_decision({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                             "lines": [{"id": to_act["line"], "key": "sas_zeros",
                                        "value": "could_bias"}]}).model_dump(mode="json")
    assert error["exits"] == [{"label": f"Act on it in Your data: {summary}", "decision": None},
                              {"label": f"Record as a limitation: {summary}", "decision": change}]
    nothing_pressed_or_locked(client, pid, before)

    # Its disposition changed by that way forward: recorded as could bias the estimate, every other
    # disposition as kept; Fit then locks the plan with the dispositions in it.
    changed = drive.post(change)
    assert changed.status_code == 200, changed.text[:600]
    triaged = changed.json()["decisions"][-1]
    assert {l["key"]: l["value"] for l in triaged["decision"]["lines"]} == {
        **{i["id"]: i["recorded"] for i in kept["items"]}, "sas_zeros": "could_bias"}
    taken = press(client, pid)
    assert taken.status_code == 200, taken.text[:600]
    assert taken.json()["locked"] is True
    view = drive.view()
    lock = view["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan" and view["state"]["plan_locked"] is True
    assert lock["decision"]["plan"]["triage"] == triaged["decision"]["lines"]
    service: ProjectService = client.app.state.service
    assert fit_press.read_press(service.workspace.project_dir(pid))["target"] == "glucose"


def sas_cells() -> tuple[int, int]:
    """The construction: how many cells of the NHANES export hold a misread SAS transport zero
    (about 5.4e-79: nonzero and below 1e-70 in magnitude), and in how many columns, by pandas."""
    import numpy as np

    from turbotab.core.tests.stage_harness import NHANES

    numbers = pd.read_csv(NHANES).select_dtypes("number").abs()
    tiny = (numbers > 0) & (numbers < 1e-70)
    return int(np.sum(tiny.to_numpy())), int(tiny.any().sum())


def sas_zeros_kept_to_act_on(kept: dict[str, Any]) -> dict[str, Any]:
    """The triage as served once "Keep the recommended dispositions" is recorded: every noticing
    with a disposition, and the one recorded "act on it" the SAS zeros, the finding on the cells
    pandas counts (:func:`sas_cells`), decided in Your data where its repair is chosen."""
    cells, columns = sas_cells()
    assert cells and columns  # the construction: the export holds misread SAS zeros
    to_act = [i for i in kept["items"] if i["recorded"] == "act_on_it"]
    assert [i["id"] for i in to_act] == ["sas_zeros"], to_act
    assert f"`{cells}` cells in `{columns}` columns" in to_act[0]["summary"]
    assert to_act[0]["stage"] == "data"
    return to_act[0]


def answered_as_asked(drive: Any, timeout: float = 300.0) -> None:
    """The findings and the fit computed, each reading a stage asks for on the way (a stage refused
    in words that asks one, as the design asks whether a column's whole numbers are amounts or
    codes) answered as the fixture's truth declares it (``truths.answers``), as a person who knows
    the table answers it. Any other stage in error fails."""
    import time

    from turbotab.core.tests.truths import answers

    end, given = time.monotonic() + timeout, []
    while True:
        stages = drive.view()["stages"]
        if all(stages[s]["status"] == "fresh" for s in ("findings", "fit")):
            return
        failed = [s for s in stages.values() if s["status"] == "error"]
        asks = next((answers(s["refusal"], drive.truth) for s in failed if s.get("refusal")
                     and answers(s["refusal"], drive.truth)), [])
        assert asks or not failed, failed
        for decision in asks:
            if decision in given:  # answered already: the stage has not computed again yet
                continue
            r = drive.post(decision)
            assert r.status_code == 200, r.text[:600]
            given.append(decision)
        assert time.monotonic() < end, {k: s["status"] for k, s in stages.items()}
        time.sleep(0.1)


@pytest.mark.slow
@needs_nhanes
def test_under_estimate_a_noticing_kept_to_act_on_and_acted_on_holds_fit_no_longer(client):
    """Both ways, the other way through: the SAS zeros kept to act on, then acted on as a person
    acts on them, by their repair as the findings serve it (Read as zero, in Your data). Acted on,
    they drop out of the triage, no refusal names them, and Fit locks the plan (after the triage's
    "Confirm all" again, where a disposition no longer stands once the cells are read as zero)."""
    drive = nhanes_under_estimate(client)
    pid = drive.pid
    first = press(client, pid)
    assert first.status_code == 409 and first.json()["error"]["code"] == "triage_open"
    assert drive.post(CONFIRM).status_code == 200
    sas_zeros_kept_to_act_on(triage(client, pid))

    findings = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    repair = next(f for f in findings if f["id"] == "sas_zeros")["repairs"][0]["decision"]
    assert repair["kind"] == "apply_repair" and repair["finding_id"] == "sas_zeros"
    acted = drive.post(repair)
    assert acted.status_code == 200, acted.text[:600]
    # Read as zero, `kcal`'s misread cells leave it whole numbers alone, so the model's design asks
    # whether they are amounts or codes; answered as the fixture declares (an amount).
    answered_as_asked(drive)
    after = triage(client, pid)
    assert "sas_zeros" not in {i["id"] for i in after["items"]}
    assert not [i for i in after["items"] if i["recorded"] == "act_on_it"]

    answers = [press(client, pid)]
    if answers[-1].status_code == 409 and "still open" in answers[-1].json()["error"]["message"]:
        assert drive.post(CONFIRM).status_code == 200
        answers.append(press(client, pid))
    assert answers[-1].status_code == 200, answers[-1].text[:600]
    assert answers[-1].json()["locked"] is True
    assert not any("not acted on yet" in a.text for a in answers)
    assert kinds(client, pid)[-1] == "lock_plan"


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

# Refusals written here: one of a stage the fit waits on, one of the fit's own.
REFUSALS = (
    Refused(stage="cohort", code="too_few_rows",
            message="The eligibility rules written here leave too few rows to analyze.",
            exits=[{"label": "Keep every row",
                    "decision": parse_decision({"kind": "set_exclusions", "rules": []})
                    .model_dump(mode="json")},
                   {"label": "Change the eligibility rules", "decision": None}]),
    Refused(stage="fit", code="singular_model_matrix",
            message="The model matrix written here is singular.",
            exits=[{"label": "Leave one of the dependent columns out of the model",
                    "decision": None}]))


def serve_refused(client, monkeypatch, pid: str, refusal: Refused) -> None:
    """``refusal`` served as the fit's status for ``pid``, as the engine serves a stage the fit
    waits on that refused (``graph.Engine._status``), or the fit's own; the premise read back as
    the server serves it."""
    service: ProjectService = client.app.state.service
    real = service.engine.status

    def refused(p: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        stages = dict(real(p, *args, **kwargs))
        if p == pid:
            stages["fit"] = stages["fit"].model_copy(update={
                "status": "error", "fresh": False, "refusal": refusal,
                "error": f"Needs {refusal.stage!r}, which failed." if refusal.stage != "fit" else
                         refusal.message})
        return stages

    monkeypatch.setattr(service.engine, "status", refused)
    assert client.get(f"/api/projects/{pid}").json()["stages"]["fit"]["refusal"] == (
        refusal.model_dump(mode="json"))  # the premise, as the server serves it


def test_under_estimate_a_refused_fit_refuses_the_press_with_its_own_refusal(client, tmp_path,
                                                                              monkeypatch):
    """A refusal written here served as the fit's status, as the engine serves a stage the fit
    waits on that refused (``graph.Engine._status``): the press is refused with its code, words and
    ways forward, and nothing is pressed or locked; the fit's own refusal likewise. With no refusal
    served, the same press locks the plan."""
    service: ProjectService = client.app.state.service
    pid = walking_under_estimate(client, tmp_path / "data")
    before = kinds(client, pid)
    for refusal in REFUSALS:
        serve_refused(client, monkeypatch, pid, refusal)
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


def test_under_prediction_a_refused_fit_does_not_hold_the_press(client, monkeypatch):
    """Under Predict nothing changes: with the fit served as refused, by a stage it waits on or by
    itself (the refusals written here), the press is taken as before: pressed, nothing locked, no
    record added, the press kept for the outcome (withdrawn after each, so the next is a new
    press)."""
    service: ProjectService = client.app.state.service
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}").json()["state"]["purpose"] == "prediction"
    pdir = service.workspace.project_dir(pid)
    assert not fit_press.pressed_for(fit_press.read_press(pdir), "hba1c")
    before = kinds(client, pid)
    for refusal in REFUSALS:
        serve_refused(client, monkeypatch, pid, refusal)
        r = press(client, pid)
        assert r.status_code == 200, r.text[:600]
        assert r.json()["pressed"] is True and r.json()["locked"] is False
        assert kinds(client, pid) == before  # a job command, not a decision
        assert fit_press.pressed_for(fit_press.read_press(pdir), "hba1c")
        monkeypatch.undo()
        fit_press.withdraw_press(pdir)  # the next refusal meets Fit not pressed
        assert not fit_press.pressed_for(fit_press.read_press(pdir), "hba1c")
