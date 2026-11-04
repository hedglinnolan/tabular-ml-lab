"""G12 (the owner's Q3) over HTTP: a refused answer's way forward that names no decision is served
with the card it opens (``target``: its quest stage and line), and the card it names is one the
project's own quest log serves (GET /quest), read independently of the refusal. A way forward the
server says in its own refusals (``ApiError``) carries its target the same way, and one that names
none is served as before, label and decision alone."""
from __future__ import annotations

from turbotab.server import schemas
from turbotab.server.errors import ApiError
from turbotab.server.tests.conftest import open_by_path, wait_for
from turbotab.server.tests.test_part2_api import decide, post, until_open


def test_an_answer_out_of_order_names_the_card_of_the_question_before_it(client):
    pid = open_by_path(client)  # dietary_recalls.csv
    wait_for(client, pid, {"ingest": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    until_open(client, pid, "purpose")

    refused = post(client, pid, {"kind": "set_grain", "grain": "repeated",
                                 "id_column": "participant_id"})
    assert refused.status_code == 409, refused.text
    body = refused.json()
    error = schemas.Refusal.model_validate(body).error  # the served shape the contract declares
    assert error.code == "not_yet"
    (way,) = body["error"]["exits"]
    assert way["decision"] is None and way["label"] == "Answer the purpose question first"
    # The quest log serves the purpose question as a line of the stage the target names.
    log = client.get(f"/api/projects/{pid}/quest").json()
    where = [s["key"] for s in log["stages"] if any(l["key"] == "purpose" for l in s["lines"])]
    assert where, "the quest log serves the purpose question"
    assert way["target"] == {"stage": where[0], "key": "purpose"}
    assert error.exits[0].target is not None and error.exits[0].target.key == "purpose"


def test_the_server_s_own_refusals_carry_a_target_and_none_where_none_is_named():
    served = ApiError(409, "outcome_not_yet", "Not yet.", [
        {"label": "Decide which rows to hold out", "decision": None,
         "target": {"stage": "whos_in", "key": "split"}},
        {"label": "Keep the answers as they are", "decision": None},
    ]).body()["error"]["exits"]
    assert served[0]["target"] == {"stage": "whos_in", "key": "split"}
    assert served[1] == {"label": "Keep the answers as they are", "decision": None}
    for way in served:
        schemas.Exit.model_validate(way)  # each in the contract's shape
