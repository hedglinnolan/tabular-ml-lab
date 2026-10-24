"""The export, through the server, where a stage it reports ended in a typed refusal (a fit-time
refusal in words, SWARM_PLAN §8; ``graph.StageRefusal``), as the swarm's random walks met it.

A short journey (``turbotab.core.reference.journeys``) under Estimate on a table written here as
the Estimate fixture is (``walking_bp.csv``'s columns and truth: walking lowers systolic pressure,
pain raises both, the concern score is a collider), whose models answer is ridge alone. Ridge has
no coefficient table, so Table 2 (the declared models, ``effects``) is refused with the words and
ways forward ``test_no_coefficient_table_refusal`` pins: the families registered for a numeric
outcome whose declarations give a coefficient table under inference are the linear model, the
feature-wise regression, the random-intercept mixed model and GEE, the linear model first; the
last two model the unit, and ``person_id`` takes a value of its own on every row, so they are not
offered. The export says that refusal as the stage served it (its code, its words said as refused,
its ways forward with their decisions), where it said Table 2 "failed", quoted the exception's
name and offered to compute it again, which refuses again. The first way forward taken (the
linear model added) is not refused so again: the bundle is made.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

N, SEED = 150, 7
FIXTURE = "walking_bp.csv"
ROLES = {"person_id": "identifier", "age": "covariate", "chronic_pain": "covariate",
         "walks_daily": "exposure", "height_cm": "covariate", "health_concern_score": "covariate"}
# Written out by hand (``test_no_coefficient_table_refusal``'s EFFECTS_WORDS and LABELS).
WORDS = ("None of the chosen model families has a coefficient table, so there is no effect to "
         "report; choose the linear model.")
SAID = f"Table 2 (the declared models) was refused, so its result cannot be reported: {WORDS}"
OFFERED = [("Add Linear model, which has a coefficient table", "linear"),
           ("Add Feature-wise regression, which has a coefficient table", "featurewise")]


def walking_bp() -> pd.DataFrame:
    """The Estimate fixture's columns (``test_replay_spline_rule.walking_bp`` at its seed): 30
    cells of ``chronic_pain`` blank, so a complete-case analysis keeps 120 rows."""
    rng = np.random.default_rng(SEED)
    age = rng.integers(30, 80, N)
    pain = (rng.random(N) < 0.35).astype(int)
    walks = (rng.random(N) < 1 / (1 + np.exp(-(0.3 - 1.0 * pain)))).astype(int)
    height = np.round(rng.normal(170, 9, N), 1)
    sbp = np.round(120 - 6 * walks + 8 * pain + 0.4 * (age - 55) + 0.01 * (age - 55) ** 2
                   + rng.normal(0, 8, N), 1)
    score = np.round(50 - 5 * walks + 0.5 * (sbp - 125) + rng.normal(0, 5, N), 1)
    frame = pd.DataFrame({"person_id": [f"P{i + 1:05d}" for i in range(N)],
                          "age": age.astype(int), "chronic_pain": pd.array(pain, dtype="Int64"),
                          "walks_daily": walks, "height_cm": height,
                          "health_concern_score": score, "sbp_mmhg": sbp})
    frame.loc[rng.choice(N, 30, replace=False), "chronic_pain"] = pd.NA
    return frame


def _export(run: Any) -> Any:
    return run.drive.c.get(f"/api/projects/{run.pid}/export")


@pytest.fixture(scope="module")
def walk(tmp_path_factory) -> dict[str, Any]:
    """The journey with ridge alone, run once: the effects stage's status as served, the export
    before Fit is pressed and after, and the export once the first way forward is taken."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server, press_fit
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder = tmp_path_factory.mktemp("export-refused")
    path = folder / FIXTURE
    walking_bp().to_csv(path, index=False)
    spec = J.Journey("refused", "clinical", "inference", "?", str(path), "test", lambda: path,
                     lambda: J.truth(dict(FIXTURE_TRUTHS[FIXTURE]), fixture=path.name),
                     target="sbp_mmhg", lenses=("clinical",), exposure="walks_daily",
                     roles=dict(ROLES),
                     answers={"missing": {"kind": "set_missing", "strategy": "complete_case"},
                              "models": {"kind": "select_models", "models": ["ridge"]}})
    seen: dict[str, Any] = {}
    run = J.Run(spec)
    with local_server(folder / "home") as client:
        run.client = J.Recording(client, run)
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        run.pid = r.json()["id"]
        truth = spec.truth()
        truth.guess, truth.ranked = J._server_guess(client, run.pid), J._server_ranked(client,
                                                                                      run.pid)
        run.drive = Drive(run.client, run.pid, truth)
        run.drive.exposure = spec.exposure
        run.drive.artifact("ingest", timeout=600)
        assert J.follow(run), run.notes
        assert not J.wait_results(run), run.notes  # Table 2 is refused
        seen["effects"] = run.drive.view()["stages"]["effects"]
        seen["open"] = _export(run)  # the plan still open: Fit not pressed
        assert press_fit(client, run.pid)
        seen["locked"] = _export(run)
        first = seen["locked"].json()["error"]["exits"][0]
        if first["decision"] is not None:
            J.post_answer(run, first["decision"])
        else:  # "Compute it again", as the export offered before it read the refusal
            client.post(f"/api/projects/{run.pid}/stages/effects/run")
        J.wait_results(run)
        seen["taken"] = J.export(run)  # Fit pressed again where the change withdrew the lock
        seen["notes"] = run.notes
    return seen


def _ways(exits: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    from turbotab.core.decisions import parse_decision

    return [(e["label"], parse_decision(e["decision"]).models) for e in exits if e["decision"]]


def test_the_effects_stage_serves_its_refusal_typed(walk):
    """The premise, as the server serves it: the effects stage's status carries the refusal."""
    effects = walk["effects"]
    assert effects["status"] == "error"
    refusal = effects["refusal"]
    assert (refusal["stage"], refusal["code"], refusal["message"]) == (
        "effects", "no_coefficient_table", WORDS)
    assert _ways(refusal["exits"]) == [(label, ["ridge", key]) for label, key in OFFERED]


def test_the_export_reports_a_refused_stage_as_its_refusal_with_its_ways_forward(walk):
    """With the plan locked, the export's refusal is the stage's own: its code and its words said
    as refused, with no exception's name, and each way forward a decision (no "Compute again")."""
    r = walk["locked"]
    assert r.status_code == 409
    error = r.json()["error"]
    assert error["code"] == "no_coefficient_table"
    assert error["message"] == SAID
    assert "ValueError" not in error["message"] and "failed" not in error["message"]
    assert all(e["decision"] for e in error["exits"])
    assert _ways(error["exits"]) == [(label, ["ridge", key]) for label, key in OFFERED]


def test_before_fit_the_open_plan_comes_first_and_the_refusal_is_said_beside_it(walk):
    error = walk["open"].json()["error"]
    assert error["code"] == "plan_open"
    assert error["message"].endswith(f" {SAID}")
    assert error["exits"][0] == {"label": "Press Fit on the analysis flowchart; it locks the plan",
                                 "decision": None}
    assert _ways(error["exits"][1:]) == [(label, ["ridge", key]) for label, key in OFFERED]


def test_the_first_way_forward_taken_is_not_refused_again(walk):
    """The linear model added beside ridge: Table 2 refits the linear model, and the bundle is
    made, where computing it again refused again."""
    r = walk["taken"]
    code = r.json()["error"]["code"] if r.status_code != 200 else None
    assert code != "no_coefficient_table", r.text[:900]
    assert r.status_code == 200, (r.text[:900], walk["notes"])
