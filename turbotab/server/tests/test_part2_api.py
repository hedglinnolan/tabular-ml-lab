"""What M2 part 1 left open, over HTTP (M2_CONTRACT §12): answers in the Router's order, the
grain the seal needs (stated, answered, or "I don't know"), and the cost of a fit stated before it.

The order refusals are Tier A at the route: an answer the Router has not reached is refused with
the way forward, a preview of it too, and a change to an answered question never is."""
from __future__ import annotations

from turbotab.server import schemas
from turbotab.server.tests.conftest import SAMPLES, open_by_path, prepare, wait_for


def post(client, pid, decision):
    return client.post(f"/api/projects/{pid}/decisions", json=decision)


def decide(client, pid, decision):
    response = post(client, pid, decision)
    assert response.status_code == 200, response.text
    return response.json()


def steps(client, pid) -> dict:
    return {s["key"]: s for s in client.get(f"/api/projects/{pid}").json()["interview"]}


def artifact(client, pid, stage) -> dict:
    return client.get(f"/api/projects/{pid}/stages/{stage}").json()["artifact"]


def until_open(client, pid, key: str, timeout: float = 60.0) -> None:
    """Wait until ``key`` is the open question (the stages it waits on have caught up)."""
    import time

    end = time.monotonic() + timeout
    while steps(client, pid)[key]["status"] != "open":
        assert time.monotonic() < end, steps(client, pid)[key]
        time.sleep(0.05)


def test_an_answer_out_of_order_is_refused_with_its_way_forward_and_a_change_never_is(client):
    pid = open_by_path(client)  # dietary_recalls.csv
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    until_open(client, pid, "purpose")

    grain = {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"}
    for route in ("decisions", "preview"):
        refused = client.post(f"/api/projects/{pid}/{route}", json=grain)
        assert refused.status_code == 409, refused.text
        error = schemas.Refusal.model_validate(refused.json()).error
        assert error.code == "not_yet"
        assert [(e.label, e.decision) for e in error.exits] == [("Answer the purpose question first", None)]
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == 2  # nothing recorded

    purpose = decide(client, pid, {"kind": "set_purpose", "purpose": "prediction"})["decisions"][-1]
    decide(client, pid, grain)
    # The purpose is withdrawn: the grain stays answered, and changing it is still allowed...
    decide(client, pid, {"kind": "revert", "decision_id": purpose["id"]})
    until_open(client, pid, "purpose")
    assert steps(client, pid)["grain"]["status"] == "answered"
    changed = decide(client, pid, {"kind": "set_grain", "grain": "unknown"})
    assert changed["state"]["grain"]["grain"] == "unknown"
    # ...while a question the Router has not reached is still refused, naming the first one it
    # holds open. (The new grain re-reads the outcome, and while it does the event question waits
    # ahead of the purpose, as the Record shows; so the Router is let settle first.)
    until_open(client, pid, "purpose")
    refused = post(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "not_yet"
    assert "purpose" in refused.json()["error"]["exits"][0]["label"]


def test_the_seal_waits_for_the_grain_and_i_dont_know_draws_an_undetermined_one(client):
    pid = open_by_path(client, SAMPLES / "genomics_expression.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["genomics"]})
    prepare(client, pid, {"kind": "set_target"})  # an assay lens: the table's shape is read first
    assert steps(client, pid)["orientation"]["status"] == "not_applicable"
    decide(client, pid, {"kind": "set_target", "column": "condition"})
    prepare(client, pid, {"kind": "set_grain"})
    assert steps(client, pid)["grain"]["status"] == "open"  # sample_id names samples, not people
    split = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
    early = post(client, pid, split)
    assert early.status_code == 409 and early.json()["error"]["code"] == "not_yet"
    assert early.json()["error"]["exits"][0]["label"] == (
        "Answer the question of whether people repeat first")

    said = decide(client, pid, {"kind": "set_grain", "grain": "unknown"})["decisions"][-1]["sentence"]
    assert "not known" in said and said.endswith("exploratory.")
    assert all(steps(client, pid)[k]["status"] == "not_applicable"
               for k in ("repeat_kind", "unit", "aggregation", "temporal"))
    prepare(client, pid, split)
    wait_for(client, pid, {"seal_plan": "fresh"})
    plan = artifact(client, pid, "seal_plan")
    assert plan["basis"]["state"] == "undetermined" and plan["exploratory"] is True

    # the split's caution resolves (answer the grain) or is attested (keep it, exploratory)
    shown = client.post(f"/api/projects/{pid}/preview", json=split).json()
    caution = shown["caution"]
    assert caution["text"] == plan["basis"]["sentence"]
    kinds = [e["decision"]["kind"] for e in caution["exits"]]
    assert kinds[-1] == "set_split" and caution["exits"][-1]["label"] == "Keep this split, labeled exploratory"
    assert "set_grain" in kinds[:-1]
    decide(client, pid, split)
    wait_for(client, pid, {"split": "fresh"})
    drawn = artifact(client, pid, "split")
    assert drawn["basis"]["state"] == "undetermined" and drawn["exploratory"] and drawn["grouped_by"] is None


def test_a_unique_person_identifier_states_the_grain_and_ask_me_anyway_reopens_it(client):
    pid = open_by_path(client, SAMPLES / "survey_instrument.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["survey"]})
    decide(client, pid, {"kind": "set_target", "column": "sought_support"})
    prepare(client, pid, {"kind": "set_roles"})
    grain = steps(client, pid)["grain"]
    assert grain["status"] == "skipped" and grain["decision_id"] is None
    assert grain["reason"] == "every `respondent_id` appears once, so each person is one row."
    structure = schemas.StructureArtifact.model_validate(artifact(client, pid, "structure"))
    assert structure.grain.stated is not None and structure.grain.stated.column == "respondent_id"
    wait_for(client, pid, {"seal_plan": "fresh"})
    basis = artifact(client, pid, "seal_plan")["basis"]
    assert (basis["state"], basis["source"], basis["exploratory"]) == ("one_row_per_unit", "stated", False)
    assert "every `respondent_id` appears once" in basis["sentence"]

    # "Ask me anyway": the answer outranks the statement, whatever comes after it
    decide(client, pid, {"kind": "set_grain", "grain": "unknown"})
    assert steps(client, pid)["grain"]["status"] == "answered"
    wait_for(client, pid, {"seal_plan": "fresh"})
    assert artifact(client, pid, "seal_plan")["basis"]["state"] == "undetermined"


def test_the_shelf_says_what_each_fit_will_take_before_it_runs(client):
    pid = open_by_path(client)  # dietary_recalls.csv
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    prepare(client, pid, {"kind": "select_models"})
    wait_for(client, pid, {"shelf": "fresh"})
    shelf = schemas.ARTIFACT_MODELS["shelf"].model_validate(artifact(client, pid, "shelf"))
    # every family that models a numeric outcome (the feature-wise tests are WP11's, the mixed
    # model and GEE WP12b's)
    from turbotab.core.models import families

    assert {f.key for f in shelf.families} == {f.key for f in families("regression")}
    assert {"linear", "elastic_net", "boosted_trees", "featurewise", "mixed", "gee"} <= \
        {f.key for f in shelf.families}
    for family in shelf.families:
        assert family.estimate_seconds is not None and family.estimate_seconds >= 0
        assert family.estimate and family.estimate.startswith(("under", "about"))
    # a fit on 480 rows of a dozen columns takes seconds, and the estimate says so: no columns named
    assert all("columns" not in f.estimate for f in shelf.families)


def test_the_event_finding_routes_to_the_event_question_and_learns_it_was_answered(client):
    """`positive_class__<target>` once said "No control for this yet." beside a recorded event."""
    pid = open_by_path(client, SAMPLES / "clinical_longitudinal.csv")
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "progressed"})
    until_open(client, pid, "event")

    def finding() -> dict:
        wait_for(client, pid, {"findings": "fresh"})
        found = [f for f in artifact(client, pid, "findings")["findings"]
                 if f["id"] == "positive_class__progressed"]
        assert len(found) == 1
        return found[0]

    asked = finding()
    assert asked["routes_to"] == "event" and asked["answered_by"] is None
    assert "No control" not in asked["summary"]
    event = decide(client, pid, {"kind": "set_event", "column": "progressed", "level": "1"})["decisions"][-1]
    assert finding()["answered_by"] == event["id"]
