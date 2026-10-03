"""What the M1 review found in the server's wiring (Tier A for the seal, Tier B otherwise).

* The held-out rows hold while the split recomputes: the service takes them from the newest split
  artifact, fresh or not, so no preview or evidence reads them in the window after a changed answer.
* A preview's basis names the rows it was computed on: an exact count over the rows not held out,
  and a sample of the training rows, never one number standing in for the other.
* Recording an answer again restarts a stage the user stopped for that answer.
"""
from __future__ import annotations

import numpy as np

from turbotab.core.graph import StageStatus
from turbotab.server.service import ProjectService
from turbotab.server.tests.conftest import answer_settled, open_by_path, prepare, wait_for


def decide(client, pid, decision):
    prepare(client, pid, decision)  # the questions before it, answered as usual (M2 §12.2)
    # each reading below high confirmed on its own (BLUEPRINT §14.1, the readings ledger)
    response = answer_settled(client, pid, None, decision)
    assert response.status_code == 200, response.text
    return response.json()


def _split_project(client) -> str:
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    wait_for(client, pid, {"roles": "fresh", "target_info": "fresh", "cohort": "fresh"})
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    decide(client, pid, {"kind": "set_split", "holdout": 0.25, "seed": 0, "folds": 5})
    wait_for(client, pid, {"split": "fresh", "cohort": "fresh", "proposals": "fresh"})
    return pid


def test_the_seal_holds_while_the_split_recomputes(client):
    pid = _split_project(client)
    service: ProjectService = client.app.state.service
    stages = service.engine.status(pid)
    sealed = service.sealed_rows(pid, stages)
    assert sealed is not None and len(sealed)
    rule = {"column": "energy_kcal", "low": 800, "high": 4000, "reason": "implausible intake"}
    decide(client, pid, {"kind": "set_exclusions", "rules": [rule]})
    # The window the reviewers instrumented: the split is stale (its key moved) and recomputing.
    stale = dict(service.engine.status(pid))
    stale["split"] = StageStatus(stage="split", status="stale", key="a-key-with-no-artifact")
    stale["cohort"] = StageStatus(stage="cohort", status="running", key="another-key")
    np.testing.assert_array_equal(service.sealed_rows(pid, stale), sealed)
    # And once it is fresh again the held-out rows are the same rows: exclusions never move one.
    wait_for(client, pid, {"split": "fresh"})
    np.testing.assert_array_equal(service.sealed_rows(pid, service.engine.status(pid)), sealed)


def test_a_preview_names_the_rows_it_read(client):
    pid = _split_project(client)
    view = client.get(f"/api/projects/{pid}").json()
    split = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    proposals = client.get(f"/api/projects/{pid}/stages/proposals").json()["artifact"]
    rows = view["summary"]["n_rows"]
    service: ProjectService = client.app.state.service
    held = len(service.sealed_rows(pid, service.engine.status(pid)))

    rule = {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    counted = client.post(f"/api/projects/{pid}/preview",
                          json={"kind": "set_exclusions", "rules": [rule]}).json()
    assert counted["basis"] == (f"Counts on the {rows - held:,} rows not held out; held-out rows "
                                f"stay sealed.")

    energy = proposals["energy"]
    decision = {"kind": "set_energy_adjustment", "method": "residual",
                "energy_column": energy["energy_column"], "nutrients": energy["nutrients"]}
    sampled = client.post(f"/api/projects/{pid}/preview", json=decision).json()
    n_train = split["n_train"]
    assert f"all {n_train:,} training rows" in sampled["basis"], sampled["basis"]
    assert "not held out" not in sampled["basis"]  # the training rows are named as such
    assert sampled["basis"].endswith("held-out rows stay sealed.")


def test_recording_an_answer_again_restarts_what_was_stopped_for_it():
    class Engine:
        def __init__(self):
            from turbotab.core.stages import build_graph

            self.graph = build_graph()
            self.ensured: list[str] = []

        def status(self, pid):
            return {
                "fit": StageStatus(stage="fit", status="stale", key="k", cancelled=True),
                "design": StageStatus(stage="design", status="fresh", key="d", fresh=True),
                "substitution": StageStatus(stage="substitution", status="error", key="s",
                                            error="boom"),
            }

        def ensure(self, pid, stage):
            self.ensured.append(stage)

    service = ProjectService.__new__(ProjectService)
    service.engine = Engine()
    service._restart_stopped("p", "select_models")
    assert service.engine.ensured == ["fit"]  # fit reads models and was stopped; design is fresh
    service._restart_stopped("p", "set_substitution")
    assert service.engine.ensured == ["fit", "substitution"]
    service._restart_stopped("p", "set_lens")  # nothing that reads the lens was stopped
    assert service.engine.ensured == ["fit", "substitution"]
