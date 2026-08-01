"""`GUIDED-086` — a fill that changes what the column IS, accepted silently.

`route_missingness` took any strategy for any column. `explicit_category` on a
**numeric** column was accepted, wrote the literal string `Missing` into it, and
turned a column of numbers into a column of text — while the recorded sentence
said only that the blanks were kept as their own category. Nothing downstream
was told: the profile re-read it as `object`, and every numeric candidate list
in the app — the feature catalogue, the selection ranking, the recipe lattice's
`numeric_columns` — silently lost a column.

**The strategy lists per branch already existed** and nothing enforced them,
which is `AUDIT-008`'s shape one more time: the capability is in the module and
the path that needs it does not consult it.

Found at L34-B by driving the blocker's own way through. That exit offered
`explicit_category` for every column including numeric ones, so the app's
recommended escape from the blocker was the route that corrupts the column.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, missingness as M                            # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _project(client):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sample_data", "clinic_visits.csv")
    with open(path, "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "hba1c"}),
                          ("set_purpose", {"answer": "prediction"})]:
        client.post(f"/project/{pid}/decision",
                    json={"kind": what, "payload": payload})
    return pid


def _numeric_column_with_blanks(client, pid):
    served = client.get(f"/project/{pid}/preprocess").json()
    for col in served["columns"]:
        if col["branch"] == "numeric":
            return col["column"]
    pytest.skip("this fixture has no numeric column with blanks")


def test_a_categorical_strategy_is_refused_on_a_numeric_column(client):
    """The gate, and it is asserted on the REFUSAL rather than on the dtype:
    a check that only looked at the result would pass against a version that
    accepted the strategy and quietly did nothing."""
    pid = _project(client)
    col = _numeric_column_with_blanks(client, pid)
    refused = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": col, "mechanism": "not_informative",
                    "strategy": M.EXPLICIT_CATEGORY}})
    assert refused.status_code >= 400, (
        "a categorical fill was accepted for a numeric column")
    assert "not offered for a numeric column" in refused.text
    assert "stops being one" in refused.text, (
        "the refusal does not say what would have happened, so it reads as a "
        "rule rather than as a consequence")


def test_the_column_keeps_its_type(client):
    """What the refusal is protecting, measured on the table."""
    pid = _project(client)
    col = _numeric_column_with_blanks(client, pid)
    before = [c for c in client.get(f"/project/{pid}").json()["columns"]
              if c["name"] == col][0]["dtype"]
    client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": col, "mechanism": "not_informative",
                    "strategy": M.EXPLICIT_CATEGORY}})
    after = [c for c in client.get(f"/project/{pid}").json()["columns"]
             if c["name"] == col][0]["dtype"]
    assert before == after, (
        f"{col} was {before} and is now {after}; the refusal did not hold")
    assert "int" in after or "float" in after                    # control


def test_every_offered_strategy_is_one_the_record_accepts(client):
    """The offer and the check read one table. Two lists that happen to agree
    are two lists, and this is the assertion that keeps them one."""
    pid = _project(client)
    served = client.get(f"/project/{pid}/preprocess").json()
    assert served["columns"], "nothing to check"                 # control
    for col in served["columns"]:
        allowed = M.STRATEGIES_BY_BRANCH[col["branch"]]
        for key in col["strategies"]:
            assert key in allowed, (
                f"{col['column']} ({col['branch']}) is offered {key!r}, which "
                f"the record refuses for that branch")


def test_the_blockers_way_through_is_available_on_the_branch_it_is_offered_for(
        client):
    """The defect that found this one. The blocker's resolve exit was
    hand-written as `explicit_category` for every column, so on a numeric column
    the app's recommended escape was the route that corrupts it."""
    pid = _project(client)
    col = _numeric_column_with_blanks(client, pid)
    refused = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": col, "mechanism": "informative",
                    "strategy": M.IMPUTE_MEDIAN}})
    assert refused.status_code == 409                            # control
    exits = refused.json()["detail"]["exits"]
    resolve = [e for e in exits if e["kind"] == "resolve"]
    assert resolve, "the blocker offers no way through but the attestation"
    assert resolve[0]["id"] in M.NUMERIC_STRATEGIES, (
        f"the way out of the blocker is {resolve[0]['id']!r}, which is not a "
        "strategy this column can take")
    # `GUIDED-072`: a client holding only the payload can act on it.
    assert resolve[0]["retry"]["payload"]["strategy"] == resolve[0]["id"]
    taken = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": col, "mechanism": "informative",
                    **resolve[0]["retry"]["payload"]}})
    assert taken.status_code == 200, (
        "the blocker's own way through is refused by the record", taken.text[:200])


def test_a_categorical_column_still_gets_the_category_exit(client):
    """The fix is a branch table, not a deletion: where keeping the blanks as a
    level is the right answer, it is still the one offered."""
    exits = M.blocker_exits("categorical")
    resolve = [e for e in exits if e["kind"] == "resolve"]
    assert resolve and resolve[0]["id"] == M.EXPLICIT_CATEGORY
    assert resolve[0]["retry"]["payload"]["strategy"] == M.EXPLICIT_CATEGORY
