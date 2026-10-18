"""Ruling R-g and U6 over HTTP: a group's one press, its record, and the served predictor groups.

A table built here: three lab values the roles stage reads alike (one role, one confidence, one
reason word for word), a nutrient that carries energy and total energy. Under Predict the roles answer records the proposals; Your data's quest log then lists the
three lab values as one line whose one press the server accepts, after which each is settled and
the record and the methods say the press in one sentence, the reason read from the served roles
stage. The roles stage is served with the project's predictor groups (W1 part 2's U6): named by the
lens, as the roles ruling of 2026-10-17 names them, and with the columns that carry energy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from turbotab.server.tests.conftest import open_by_path, wait_for


def post(client, pid, body):
    return client.post(f"/api/projects/{pid}/decisions", json=body)


def answer(client, pid, key, body, timeout=180.0):
    """Wait until ``key`` is the open question, then record ``body``."""
    end = time.monotonic() + timeout
    while True:
        steps = client.get(f"/api/projects/{pid}").json()["interview"]
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if first is not None and first["key"] == key and first["status"] == "open":
            break
        assert time.monotonic() < end, f"{key} never opened: {first}"
        time.sleep(0.05)
    response = post(client, pid, body)
    assert response.status_code == 200, response.text[:900]


def _table(tmp_path):
    rng = np.random.default_rng(7)
    n = 160
    kcal = rng.normal(2100, 400, n)
    frame = pd.DataFrame({"pid": np.arange(n), "y": rng.normal(100, 12, n), "kcal": kcal,
                          "fat_g": 0.035 * kcal + rng.normal(0, 8, n)})
    for c in ("lab_a", "lab_b", "lab_c"):
        frame[c] = rng.normal(5, 1, n)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "groups.csv"
    frame.to_csv(path, index=False)
    return path


def _opened(client, tmp_path, lens: str) -> tuple[str, dict]:
    pid = open_by_path(client, _table(tmp_path))
    wait_for(client, pid, {"ingest": "fresh"})
    answer(client, pid, "lens", {"kind": "set_lens", "lenses": [lens]})
    if lens == "genomics":  # an assay lens asks which way the table runs
        answer(client, pid, "orientation", {"kind": "set_orientation",
                                            "orientation": "sample_major"})
    answer(client, pid, "target", {"kind": "set_target", "column": "y"})
    answer(client, pid, "purpose", {"kind": "set_purpose", "purpose": "prediction"})
    wait_for(client, pid, {"roles": "fresh"}, timeout=180)
    return pid, client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]


def test_the_roles_stage_is_served_with_the_projects_predictor_groups(client, tmp_path):
    clinical, served = _opened(client, tmp_path / "c", "clinical")
    groups = served["predictor_groups"]
    # the ruling's names: under the clinical lens a measure, under an assay lens an assay feature
    assert (groups["no_fill"]["exposure"]["one"], groups["no_fill"]["exposure"]["head"]) == (
        "a measure", ["Measure", "Measures"])
    assert groups["fill"]["covariate"]["one"] == "a characteristic"
    # the fill is named only in the groups where it runs
    assert "energy-aware fill" in groups["fill"]["exposure"]["note"]
    assert "energy-aware fill" not in groups["no_fill"]["exposure"]["note"]
    # of the table's columns, the nutrient carries energy; total energy and the lab values do not
    assert groups["energy_bearing"] == ["fat_g"]
    genomics, served = _opened(client, tmp_path / "g", "genomics")
    assert served["predictor_groups"]["no_fill"]["exposure"]["one"] == "an assay feature"


def test_a_groups_one_press_is_accepted_settles_each_column_and_is_said_in_one_sentence(
        client, tmp_path):
    pid, served = _opened(client, tmp_path, "clinical")
    proposals = {p["column"]: p for p in served["columns"]}

    def judged(c: str) -> tuple[str, str, str]:
        return proposals[c]["proposed"], proposals[c]["confidence"], proposals[c]["reason"]

    # The lab values' judgment, and every column the roles stage judged alike (read from its
    # proposals, in their order: the table's measured columns may share it).
    assert proposals["lab_a"]["confidence"] != "high"
    labs = [c for c in proposals if judged(c) == judged("lab_a")]
    assert {"lab_a", "lab_b", "lab_c"} <= set(labs)
    reason = proposals["lab_a"]["reason"]
    answer(client, pid, "roles", {"kind": "set_roles", "roles": {
        c["column"]: c["proposed"] for c in served["columns"] if c["proposed"]}})
    log = client.get(f"/api/projects/{pid}/quest").json()
    lines = [l for s in log["stages"] for l in s["lines"] if l["id"] == "decision:confirm_role"]
    [group] = [l for l in lines if l["items"][0]["kind"] == "role_group"]
    assert [i["column"] for i in group["items"][1:]] == labs
    press = group["items"][0]["options"][0]["decision"]
    assert press == {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": c, "value": proposals[c]["proposed"]} for c in labs]}
    # silent: part of the group is refused, naming it
    part = post(client, pid, {"kind": "confirm_readings", "items": press["items"][:2]})
    assert part.status_code == 409 and part.json()["error"]["code"] == "roles_one_at_a_time"
    response = post(client, pid, press)
    assert response.status_code == 200, response.text[:900]
    view = response.json()
    assert {c: view["state"]["role_confirmations"][c] for c in labs} == {
        c: proposals[c]["proposed"] for c in labs}
    sentence = view["decisions"][-1]["sentence"]
    first = (f"`{labs[0]}`, `{labs[1]}` and {len(labs) - 2} more" if len(labs) > 3 else
             f"`{labs[0]}`, `{labs[1]}` and `{labs[2]}`")
    assert sentence == (f"The {len(labs)} columns {first}, proposed as measures at "
                        f"{proposals['lab_a']['confidence']} confidence, below high, for one reason "
                        f"(\"{reason}\"), were confirmed together as one group, after the evidence "
                        f"for their proposal was read.")
    methods = client.get(f"/api/projects/{pid}/methods").json()
    assert sentence in methods["text"]
    after = client.get(f"/api/projects/{pid}/quest").json()
    [done] = [l for s in after["stages"] for l in s["lines"] if l["key"] == group["key"]]
    assert done["status"] == "answered" and done["decision_id"] == view["decisions"][-1]["id"]
