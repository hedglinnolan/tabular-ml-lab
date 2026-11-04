"""A test driver answers from the fixture's truth only a refusal that asks for readings
(``truths.answer_refusal``). INBOX (the wording batch's truths.py item): ``role_unconfirmed`` was
read as one such refusal whatever it carried, but a revisit refused because a recorded answer reads
what it would unsettle (``decisions._answers_keep_settled_readings`` and
``_answers_keep_settled_roles``) asks nothing: it carries routes, a reading's other value in place
of the one revisited (a ``confirm_reading`` exit), the answer withdrawn first, and the answers
kept. A driver that answered it confirmed the reading the truth declares and posted the revisit
again, round after round, or, where the truth declares the other value, changed a settled reading
and so took a route nobody chose.

By construction, through an in-memory log that records what ``decisions.validate`` accepts: the
revisit refusal is returned as it is, nothing posted and the log unchanged, whatever the truth
declares; and, the guard's other direction, a ``role_unconfirmed`` that asks (an energy adjustment
naming a nutrient whose role rode along) is still answered from the truth and the answer recorded.
"""
from __future__ import annotations

from typing import Any

from turbotab.core import decisions as d
from turbotab.core.tests.test_revisit_offers_the_routes_it_names import (
    LINEAR, REVISIT, ctx, plan, rec,
)
from turbotab.core.tests.test_settled_roles_revisit_offers_its_routes import ADJ, confirmed
from turbotab.core.tests.test_settled_roles_revisit_offers_its_routes import ctx as roles_ctx
from turbotab.core.tests.truths import Truth, answer_refusal


class Response:
    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status_code, self._body = status_code, body
        self.text = str(body)

    def json(self) -> dict[str, Any]:
        return self._body


class Log:
    """The decisions log as the server keeps it: a decision ``validate`` accepts is recorded, and a
    refused one answered 409 with the refusal."""

    def __init__(self, records: list[d.DecisionRecord], context: Any):
        self.records, self.context, self.posted = list(records), context, []

    def post(self, body: dict[str, Any]) -> Response:
        self.posted.append(body)
        try:
            done = d.validate(body, self.context(self.records))
        except d.Refusal as refused:
            return Response(409, refused.to_dict())
        self.records.append(rec(self.records[-1].seq + 1, done.model_dump(mode="json")))
        return Response(200, {})


def drive(log: Log, body: dict[str, Any], truth: Truth) -> Response:
    return answer_refusal(log.post, log.post(body), truth, lambda: log.post(body))


def test_the_readings_revisit_is_returned_unanswered_whatever_the_truth_declares():
    for value in ("code", "amount"):
        log = Log(plan(LINEAR), ctx)
        before = [r.id for r in log.records]
        truth = Truth({"code_or_count:charlson_index": value}, fixture=f"charlson as {value}")
        response = drive(log, REVISIT, truth)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "role_unconfirmed"
        assert [e["decision"]["kind"] for e in response.json()["error"]["exits"]
                if e["decision"]] == ["confirm_reading", "revert"]
        assert log.posted == [REVISIT], value
        assert [r.id for r in log.records] == before
        assert d.fold(log.records).shape_confirmations == {"code_or_count:charlson_index": "code"}


def test_the_roles_revisit_is_returned_with_its_withdrawal_not_taken():
    log = Log(confirmed(ADJ), roles_ctx)
    revisit = {"kind": "revert", "decision_id": "r3"}
    response = drive(log, revisit, Truth({"role:ALC": "exposure"}, fixture="ALC"))
    error = response.json()["error"]
    assert response.status_code == 409 and error["code"] == "role_unconfirmed"
    assert any((e["decision"] or {}).get("kind") == "revert" for e in error["exits"])
    assert log.posted == [revisit]
    assert d.fold(log.records).energy_adjustment is not None


def test_a_role_unconfirmed_that_asks_is_still_answered_from_the_truth():
    """The energy adjustment names `ALC`, whose role rode along in a bulk answer and was never
    confirmed on its own: the refusal asks for its role, the driver confirms it as the truth
    declares, and the adjustment is then recorded."""
    roles = {"kcal": "energy", "fat_g": "exposure", "ALC": "exposure", "age": "covariate"}
    head = [rec(1, {"kind": "set_target", "column": "y"}),
            rec(2, {"kind": "set_roles", "roles": roles, "unconfirmed": ["ALC"]})]
    log = Log(head, roles_ctx)
    first = log.post(ADJ)
    assert first.status_code == 409 and first.json()["error"]["code"] == "role_unconfirmed"
    response = answer_refusal(log.post, first, Truth({"role:ALC": "exposure"}, fixture="ALC"),
                              lambda: log.post(ADJ))
    assert response.status_code == 200, response.text
    assert d.fold(log.records).energy_adjustment.nutrients == ["fat_g", "ALC"]
    assert d.fold(log.records).role_confirmations == {"ALC": "exposure"}
