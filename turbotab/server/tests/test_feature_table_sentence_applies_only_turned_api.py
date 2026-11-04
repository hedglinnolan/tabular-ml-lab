"""FT1 over HTTP: a features-in-rows reading is said only while the table is turned.

The turn check's exits record ``set_feature_table`` (``stages.working.turn_plan``), and a table
turns only on `feature_major` (``stages.working.oriented_stage``). The reading posted on a table no
answer has turned, and kept while the orientation is answered as one row per sample, writes no
sentence in the methods the server serves (``GET …/methods``) nor in the export bundle's own
sections and words (``GET …/methods/sections``); answered as features in rows, the table turns and
the reading's sentence is said in force; answered back as samples, it is folded out again. The
table: the untargeted metabolomics sample supplied turned round (features in rows), opened through
the real server.

After a seen lock the methods fold nothing out, but a reading recorded on a table no answer turns,
and withdrawn before one does, is read at no moment, so neither the methods nor the exported plan
(``GET …/plan``: ``plan_lock.plan_document``, which the bundle's ``analysis_plan.json`` holds too)
states it or its revert, and the plan's count of decisions made after the estimates were seen
leaves both out. One recorded on the turned table
is listed and counted, and stays so, not in force, once the table is answered back as samples
(``provenance.never_read``). The lock is recorded in the project's log as the server records one
after estimates were shown under prediction (``seen="prediction"``; the routing gate), so no fit
is run; every other decision is posted to the real server.
"""
from __future__ import annotations

import time

import pandas as pd

from turbotab.server.tests.conftest import SAMPLES, open_by_path, wait_for

LEAD = "For the features-in-rows table, "


def _post(client, pid, body):
    response = client.post(f"/api/projects/{pid}/decisions", json=body)
    assert response.status_code == 200, response.text[:900]
    return response.json()


def _answer(client, pid, key, body, timeout=180.0):
    """Wait until ``key`` is the open question, then record ``body``."""
    end = time.monotonic() + timeout
    while True:
        steps = client.get(f"/api/projects/{pid}").json()["interview"]
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if first is not None and first["key"] == key and first["status"] == "open":
            break
        assert time.monotonic() < end, f"{key} never opened: {first}"
        time.sleep(0.05)
    return _post(client, pid, body)


def _said(client, pid, record_id) -> tuple[list, list]:
    """The reading's lines in the served methods, and its entries in the export's sections."""
    methods = client.get(f"/api/projects/{pid}/methods")
    assert methods.status_code == 200, methods.text[:600]
    sections = client.get(f"/api/projects/{pid}/methods/sections")
    assert sections.status_code == 200, sections.text[:600]
    lines = [line for line in methods.json()["lines"] if line["kind"] == "set_feature_table"]
    entries = [e for s in sections.json()["sections"] for e in s["entries"]
               if record_id in (e.get("record_ids") or [e.get("record_id")])]
    assert (LEAD.rstrip(", ") in methods.json()["text"]) == bool(lines)
    return lines, entries


def _opened(client, tmp_path) -> str:
    """The untargeted metabolomics sample turned round (features in rows), its lens answered."""
    m = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv").set_index("sample_id")
    turned = m.select_dtypes("number").T
    turned.index.name = "feature_id"
    source = tmp_path / "features_in_rows.csv"
    turned.to_csv(source)
    pid = open_by_path(client, source)
    wait_for(client, pid, {"ingest": "fresh"})
    _answer(client, pid, "lens", {"kind": "set_lens", "lenses": ["metabolomics"]})
    return pid


def test_a_feature_table_reading_is_said_only_while_the_table_is_turned(client, tmp_path):
    pid = _opened(client, tmp_path)

    # A turn exit pressed on a table no answer has turned: recorded, and said nowhere.
    recorded = _post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                   "annotations": []})
    reading = next(r for r in recorded["decisions"] if r["decision"]["kind"] == "set_feature_table")
    assert reading["sentence"].startswith(LEAD)
    assert _said(client, pid, reading["id"]) == ([], [])

    # Rows are samples: the table is not turned, so the reading still says nothing.
    _answer(client, pid, "orientation", {"kind": "set_orientation", "orientation": "sample_major"})
    assert _said(client, pid, reading["id"]) == ([], [])

    # Features in rows: the table turns, and the reading is how it reads.
    wait_for(client, pid, {"oriented": "fresh"})
    _post(client, pid, {"kind": "set_orientation", "orientation": "feature_major"})
    lines, entries = _said(client, pid, reading["id"])
    assert [(line["record_id"], line["sentence"], line["in_force"]) for line in lines] == [
        (reading["id"], reading["sentence"], True)]
    assert len(entries) == 1 and entries[0]["text"].startswith(LEAD)

    # Answered back as samples: folded out again.
    _post(client, pid, {"kind": "set_orientation", "orientation": "sample_major"})
    assert _said(client, pid, reading["id"]) == ([], [])


COUNTED = "made after the estimates were seen, listed with it and marked so in the methods."


def _later(client, pid) -> tuple[list[int], list[int], str]:
    """The seqs the exported plan lists as made after the estimates were seen, the seqs of the
    served methods' lines so marked, and the plan's text."""
    plan = client.get(f"/api/projects/{pid}/plan")
    assert plan.status_code == 200, plan.text[:600]
    methods = client.get(f"/api/projects/{pid}/methods").json()
    return ([e["seq"] for e in plan.json()["after_estimates"]],
            [line["seq"] for line in methods["lines"] if line["after_estimates"]],
            plan.json()["text"])


def _counts(text: str, n: int) -> bool:
    """Whether the plan's text counts ``n`` decisions made after the estimates were seen."""
    if n == 0:
        return COUNTED not in text
    return f" {n:,} decision{'s were' if n != 1 else ' was'} {COUNTED}" in text


def _reading(response: dict) -> dict:
    return next(r for r in reversed(response["decisions"])
                if r["decision"]["kind"] == "set_feature_table")


def test_after_a_seen_lock_the_plan_lists_a_reading_only_once_the_table_turned(client, tmp_path):
    from turbotab.core.decisions import LockPlan
    from turbotab.core.plan_lock import digest

    pid = _opened(client, tmp_path)
    _answer(client, pid, "orientation", {"kind": "set_orientation", "orientation": "sample_major"})
    client.app.state.service.log(pid).append(
        LockPlan(plan={}, digest=digest({}), seen="prediction"))
    listed, stated, text = _later(client, pid)
    assert listed == stated == [] and _counts(text, 0)

    # A turn exit pressed on the unturned table, then withdrawn: neither listed nor counted.
    unread = _reading(_post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                          "annotations": []}))
    assert unread["after_estimates"] is True and unread["sentence"]
    listed, stated, text = _later(client, pid)
    assert listed == stated == [] and _counts(text, 0)
    undone = _post(client, pid, {"kind": "revert", "decision_id": unread["id"]})["decisions"][-1]
    assert undone["decision"]["kind"] == "revert" and undone["sentence"]
    listed, stated, text = _later(client, pid)
    assert listed == stated == [] and _counts(text, 0)

    # Turned after the lock, the table reads by the reading recorded then: listed and counted.
    wait_for(client, pid, {"oriented": "fresh"})
    turn = _post(client, pid, {"kind": "set_orientation", "orientation": "feature_major"})
    turn = turn["decisions"][-1]
    read = _reading(_post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                        "annotations": []}))
    assert read["sentence"].startswith("After the estimates were seen, for the features-in-rows")
    listed, stated, text = _later(client, pid)
    assert listed == stated == [turn["seq"], read["seq"]] and _counts(text, 2)

    # Answered back as samples: the reading is a change made after the estimates were seen, kept.
    back = _post(client, pid, {"kind": "set_orientation", "orientation": "sample_major"})
    back = back["decisions"][-1]
    listed, stated, text = _later(client, pid)
    assert listed == stated == [turn["seq"], read["seq"], back["seq"]] and _counts(text, 3)
    lines = client.get(f"/api/projects/{pid}/methods").json()["lines"]
    assert [(line["record_id"], line["in_force"]) for line in lines
            if line["kind"] == "set_feature_table"] == [(read["id"], False)]
