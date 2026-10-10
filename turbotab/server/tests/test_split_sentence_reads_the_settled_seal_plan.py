"""The split's recorded sentence names the grouping the seal draws by, even when the split is
recorded while the seal plan recomputes (WAVE_C6A §7 ruling 5, Q-b).

E1c's integration found a split recorded while the seal's basis recomputed stored a methods
sentence without "keeping each `participant_id`'s rows together": the sentence read the seal plan
only while it was fresh, and said less otherwise. The test made that race certain here: the seal
plan takes a second and a half to compute, and the roles' reading of the repeats (the sentence's
other source of the grouping) is not there, as when both recompute at once.

Reference: the fixture's design, by hand. ``dietary_recalls.csv`` holds two recall days per
participant (``participant_id`` repeats), so the seal keeps each participant's rows together.
"""
from __future__ import annotations

import time

from turbotab.server.tests.conftest import open_by_path, wait_for
from turbotab.server.tests.test_previews import decide


def test_a_split_recorded_while_the_seal_plan_recomputes_names_the_grouping(client, monkeypatch):
    from turbotab.core import seal
    from turbotab.server import service

    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    wait_for(client, pid, {"roles": "fresh", "target_info": "fresh", "cohort": "fresh"})
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    wait_for(client, pid, {"cohort": "fresh", "seal_plan": "fresh"})

    real = seal.plan

    def slow(*args, **kwargs):
        time.sleep(1.5)
        return real(*args, **kwargs)

    monkeypatch.setattr(seal, "plan", slow)
    # the roles' repeats are not read either: only the seal plan can name the grouping
    monkeypatch.setattr(service.SentenceFacts, "repeats", property(lambda self: None))
    # an exclusion moves the cohort, so the seal plan recomputes (slowly) behind the split
    rule = {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intake"}
    decide(client, pid, {"kind": "set_exclusions", "rules": [rule]})
    status = client.get(f"/api/projects/{pid}").json()["stages"]["seal_plan"]["status"]
    assert status != "fresh", status  # the race this test is about
    view = decide(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    said = [r["sentence"] for r in view["decisions"] if r["decision"]["kind"] == "set_split"][-1]
    assert "keeping each `participant_id`'s rows together" in said, said
