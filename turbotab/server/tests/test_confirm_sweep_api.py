"""P0.5 · the Confirm sweep, For the record and the triage on the server, on the NHANES journey
(sugar → fasting glucose, under Estimate), as ``test_quest_api`` drives it.

The expectations come from outside the sweep: the stages' crosswalk cards; the fixture's declared
causal reading (``truths.py``: blood pressure, HDL, triglycerides and the medications are
downstream of the diet, so the total effect's model leaves them out, and how their values are read
changes no number); the table's own size; and the rulings that Confirm all is one record, replayed
from the log, and that a sweep stated otherwise since is not covered by it.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.decisions import DecisionLog, fold
from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import FIXTURE_TRUTHS, fixture_truth
from turbotab.server.tests.test_quest_api import ROLES, needs_nhanes, quest, rule, settle

TRUTH = FIXTURE_TRUTHS["_tt_tmp_nhanes.csv"]
# The author's reading: changed by sugar or measured after it, so not in the total effect's model.
DOWNSTREAM = {k.split(":", 1)[1] for k, v in TRUTH.items()
              if k.startswith("adjust:") and v.endswith(",yes") and v.startswith("no,")}


def confirms(stage: dict[str, Any]) -> list[str]:
    return [l["key"] for l in stage["lines"] if l["label"] == "Confirm"]


def post(drive, body: dict[str, Any]):
    return drive.c.post(f"/api/projects/{drive.pid}/decisions", json=body)


@needs_nhanes
def test_each_stage_sweeps_what_would_change_a_number_and_confirm_all_is_recorded(client):
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
    settle(drive)
    stages = quest(drive)

    # ── per stage, the defaults whose alternative would change a number, last ──
    sweeps = {k: s["sweep"] and s["sweep"]["id"] for k, s in stages.items()}
    assert sweeps == {"data": "other:confirm-sweep:data",
                      "question": "other:confirm-sweep:question", "first_look": None,
                      "whos_in": "other:confirm-sweep:whos_in", "models": "other:confirm-sweep",
                      "results": None, "writeup": None}
    # Who's in: each SEQN is one person (stated, not asked); Models: no modifier declared, and the
    # primary model alone with the causal estimators one step away.
    assert confirms(stages["question"]) == ["design"]
    [design] = [l for l in stages["question"]["lines"] if l["key"] == "design"]
    assert "trial" in design["would_change"]
    assert confirms(stages["whos_in"]) == ["grain"]
    assert confirms(stages["models"]) == ["modification", "causal"]
    for key in ("whos_in", "models"):
        labels = [l["label"] for l in stages[key]["lines"]]
        assert labels == sorted(labels, key=("Decide", "Confirm", "For the record").index), key
    # Your data: what the values settled. A role decides whether a column enters a model at all;
    # the readings of the columns downstream of sugar change no number on the total effect.
    [read] = [l for l in stages["data"]["lines"] if l["label"] == "Confirm"]
    assert read["key"] == "read_from_data" and read["would_change"]
    got = {(i["kind"], i["column"]) for i in read["items"]}
    assert {("role", "SEQN"), ("role", "kcal"), ("code_or_count", "sugar")} <= got
    assert not {c for k, c in got if k != "role"} & DOWNSTREAM
    [quiet] = [l for l in stages["data"]["lines"] if l["key"] == "read_from_data:unchanged"]
    assert quiet["label"] == "For the record"
    assert {i["column"] for i in quiet["items"]} == {"bp_sys", "bp_di"} == (
        {i["column"] for i in quiet["items"]} & DOWNSTREAM)
    # The outcome's kind is Your question's, read at high confidence: For the record there.
    assert "task" not in {i["kind"] for i in read["items"]}
    assert next(l for l in stages["question"]["lines"] if l["key"] == "task")["label"] == (
        "For the record")

    # ── Confirm all: one record, the defaults as stated ──
    whos_in_before = stages["whos_in"]["progress"]
    r = post(drive, {"kind": "confirm_sweep", "stage": "whos_in"})
    assert r.status_code == 200, r.text[:600]
    record = r.json()["decisions"][-1]
    grain = next(l for l in stages["whos_in"]["lines"] if l["key"] == "grain")
    assert record["decision"]["lines"] == [{"id": "q:grain", "key": "grain",
                                            "value": grain["reason"], "basis": None}]
    assert "`SEQN`" in record["sentence"] and "Who's in" in record["sentence"]
    stages = quest(drive)
    sweep = stages["whos_in"]["sweep"]
    assert sweep["answered"] and sweep["confirmed_by"] == record["id"]
    assert stages["whos_in"]["progress"]["answered"] == whos_in_before["answered"] + 1
    assert not stages["models"]["sweep"]["answered"]  # one sweep, one stage
    # Replayed: the log read back from disk folds to the same confirmation.
    path = client.app.state.service.workspace.decisions_path(drive.pid)
    replayed = fold(DecisionLog(path).records())
    assert replayed.sweeps["whos_in"].model_dump(mode="json") == {
        "stage": "whos_in", "sweep": "defaults", "lines": record["decision"]["lines"]}

    # A sweep shown before a default changed is refused, with the sweep as it stands as the exit.
    stale = post(drive, {"kind": "confirm_sweep", "stage": "models", "lines": [
        {"id": "q:modification", "key": "modification", "value": "an older reading"}]})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "sweep_changed"
    assert [l["key"] for l in stale.json()["error"]["exits"][0]["decision"]["lines"]] == [
        "modification", "causal"]
    nothing = post(drive, {"kind": "confirm_sweep", "stage": "first_look"})
    assert nothing.status_code == 409 and nothing.json()["error"]["code"] == "nothing_to_confirm"
    # Undoing the confirmation reopens the sweep.
    assert post(drive, {"kind": "revert", "decision_id": record["id"]}).status_code == 200
    assert not quest(drive)["whos_in"]["sweep"]["answered"]

    # ── For the record ──
    out = drive.c.get(f"/api/projects/{drive.pid}/record")
    assert out.status_code == 200, out.text[:600]
    lines = {s["key"]: {(l["kind"], l["key"]): l for l in s["lines"]} for s in out.json()["stages"]}
    n_seqn = int(drive.view()["summary"]["n_rows"])
    assert f"{n_seqn:,} rows" in lines["data"][("ingest", "facts")]["text"]
    assert ("profile", "basis") in lines["data"]
    assert ("not_applicable", "orientation") in lines["data"]  # no assay lens
    assert ("not_applicable", "survey") in lines["whos_in"]  # no survey weight
    assert ("set_for_you", "task") in lines["question"]
    assert ("set_for_you", "clusters") in lines["whos_in"]
    assert "`bp_sys`" in lines["data"][("set_for_you", "read_from_data:unchanged")]["text"]

    # ── the triage at the gate, before the plan's lock ──
    t = drive.c.get(f"/api/projects/{drive.pid}/triage").json()
    assert (t["stage"], t["gate"]) == ("models", "gate:open-noticings-before-lock")
    assert all(i["blocker"] == (i["severity"] == "critical") for i in t["items"])
    # Gender is a confounder in the author's reading, so it is in the model as an adjustment term
    # only: which of its two values is 1 changes no number for sugar, exactly (WAVE_C6A_PLAN Q-a:
    # the model matrix spans the same space either way, Frisch–Waugh–Lovell).
    assert TRUTH["adjust:gender"] == "yes,yes,no"
    gender = next(i for i in t["items"] if i["id"] == "binary_text__gender")
    assert gender["recommended"] == "no_change" and "`gender`" in gender["reason"]
    assert gender["band"] == 0 and gender["calibrated"]
    assert t["confirmable"] == (t["blockers"] == 0)
    r = post(drive, {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"})
    assert r.status_code == 200, r.text[:600]
    triaged = r.json()["decisions"][-1]
    assert {l["key"] for l in triaged["decision"]["lines"]} == {i["id"] for i in t["items"]}
    line = next(l for s in quest(drive).values() for l in s["lines"]
                if l["key"] == "binary_text__gender")
    assert (line["status"], line["decision_id"]) == ("answered", triaged["id"])
    assert drive.c.get(f"/api/projects/{drive.pid}/triage").json()["answered"]
    # Each disposition is recorded on the recommendation and reason the triage showed.
    shown = {i["id"]: f"{i['recommended']}: {i['reason']}" for i in t["items"]}
    assert {l["key"]: l["basis"] for l in triaged["decision"]["lines"]} == shown

    # ── once the plan is fixed, the triage is read, no longer confirmed ──
    drive.artifact("fit")  # the first estimate served locks the plan under inference
    assert drive.view()["state"]["plan_locked"]
    t = drive.c.get(f"/api/projects/{drive.pid}/triage").json()
    assert t["passed"] and not t["confirmable"]
    late = post(drive, {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"})
    assert late.status_code == 409 and late.json()["error"]["code"] == "gate_passed"
