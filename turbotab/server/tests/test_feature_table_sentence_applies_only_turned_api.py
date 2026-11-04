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
and withdrawn before one does, is read at no moment, so the methods state neither it nor its
revert; the exported plan (``GET …/plan``: ``plan_lock.plan_document``, which the bundle's
``analysis_plan.json`` holds too) still lists and counts both, as the Record holds them, and its
text names each as left out of the methods (V2-1). One recorded on the turned table is listed,
counted and marked so in the methods, and stays so, not in force, once the table is answered back
as samples (``provenance.never_read``). A reading in force when it was recorded but unturned before
the lock and since is stated nowhere in the methods, so its revert after the lock writes no
sentence either and is named as left out; one in force at the lock is in the plan as declared, and
its revert is said and marked (V2-2). The lock is recorded in the project's log as the server
records one after estimates were shown under prediction (``seen="prediction"``; the routing gate),
so no fit is run; every other decision is posted to the real server.
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


MADE = "made after the estimates were seen, listed with it"
NOW = "nothing in the current analysis."


def _later(client, pid) -> tuple[list[int], list[int], str]:
    """The seqs the exported plan lists as made after the estimates were seen, the seqs of the
    served methods' lines so marked, and the plan's text."""
    plan = client.get(f"/api/projects/{pid}/plan")
    assert plan.status_code == 200, plan.text[:600]
    methods = client.get(f"/api/projects/{pid}/methods").json()
    return ([e["seq"] for e in plan.json()["after_estimates"]],
            [line["seq"] for line in methods["lines"] if line["after_estimates"]],
            plan.json()["text"])


def _reading(response: dict) -> dict:
    return next(r for r in reversed(response["decisions"])
                if r["decision"]["kind"] == "set_feature_table")


def _lock(client, pid) -> None:
    from turbotab.core.decisions import LockPlan
    from turbotab.core.plan_lock import digest

    client.app.state.service.log(pid).append(
        LockPlan(plan={}, digest=digest({}), seen="prediction"))


def test_after_a_seen_lock_the_plan_counts_every_reading_and_names_the_unread(client, tmp_path):
    pid = _opened(client, tmp_path)
    _answer(client, pid, "orientation", {"kind": "set_orientation", "orientation": "sample_major"})
    _lock(client, pid)
    listed, stated, text = _later(client, pid)
    assert listed == stated == []
    assert text.endswith(" No decision has been made since the estimates were seen.")

    # A turn exit pressed on the unturned table, then withdrawn: listed and counted as the Record
    # holds them, and named as left out of the methods, which state neither.
    unread = _reading(_post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                          "annotations": []}))
    assert unread["after_estimates"] is True and unread["sentence"]
    listed, stated, text = _later(client, pid)
    assert (listed, stated) == ([unread["seq"]], [])
    assert text.endswith(f" 1 decision was {MADE}; decision #{unread['seq']} is left out of the "
                         f"methods, as it changes {NOW}")
    undone = _post(client, pid, {"kind": "revert", "decision_id": unread["id"]})["decisions"][-1]
    assert undone["decision"]["kind"] == "revert" and undone["sentence"]
    left = f"decisions #{unread['seq']} and #{undone['seq']}"
    listed, stated, text = _later(client, pid)
    assert (listed, stated) == ([unread["seq"], undone["seq"]], [])
    assert text.endswith(f" 2 decisions were {MADE}; {left} are left out of the methods, as they "
                         f"change {NOW}")

    # Turned after the lock, the table reads by the reading recorded then: listed, counted and
    # marked so in the methods, beside the two still named as left out.
    wait_for(client, pid, {"oriented": "fresh"})
    turn = _post(client, pid, {"kind": "set_orientation", "orientation": "feature_major"})
    turn = turn["decisions"][-1]
    read = _reading(_post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                        "annotations": []}))
    assert read["sentence"].startswith("After the estimates were seen, for the features-in-rows")
    listed, stated, text = _later(client, pid)
    assert listed == [unread["seq"], undone["seq"], turn["seq"], read["seq"]]
    assert stated == [turn["seq"], read["seq"]]
    assert text.endswith(f" 4 decisions were {MADE}; 2 of them are marked so in the methods, and "
                         f"{left} are left out of them, as they change {NOW}")

    # Answered back as samples: the reading is a change made after the estimates were seen, kept.
    back = _post(client, pid, {"kind": "set_orientation", "orientation": "sample_major"})
    back = back["decisions"][-1]
    listed, stated, text = _later(client, pid)
    assert listed == [unread["seq"], undone["seq"], turn["seq"], read["seq"], back["seq"]]
    assert stated == [turn["seq"], read["seq"], back["seq"]]
    assert text.endswith(f" 5 decisions were {MADE}; 3 of them are marked so in the methods, and "
                         f"{left} are left out of them, as they change {NOW}")
    lines = client.get(f"/api/projects/{pid}/methods").json()["lines"]
    assert [(line["record_id"], line["in_force"]) for line in lines
            if line["kind"] == "set_feature_table"] == [(read["id"], False)]


def _withdrawn_after_the_lock(client, tmp_path, *, unturned: bool) -> tuple[dict, dict, str]:
    """A reading recorded on the turned table before the lock, the table answered back as samples
    first when ``unturned``, and reverted after the lock: the reading, its revert, and the pid."""
    pid = _opened(client, tmp_path)
    _answer(client, pid, "orientation", {"kind": "set_orientation", "orientation": "feature_major"})
    wait_for(client, pid, {"oriented": "fresh"})
    reading = _reading(_post(client, pid, {"kind": "set_feature_table", "label": "feature_id",
                                           "annotations": []}))
    target = "bmi"  # a column once the table is turned (features in columns)
    if unturned:
        _post(client, pid, {"kind": "set_orientation", "orientation": "sample_major"})
        wait_for(client, pid, {"oriented": "fresh"})
        target = "S002"  # a column of the table as supplied (one per sample)
    _post(client, pid, {"kind": "set_target", "column": target})
    _lock(client, pid)
    undone = _post(client, pid, {"kind": "revert", "decision_id": reading["id"]})["decisions"][-1]
    assert undone["decision"]["kind"] == "revert" and undone["after_estimates"] is True
    assert undone["sentence"]
    return reading, undone, pid


def test_a_reading_withdrawn_after_the_lock_unturned_since_writes_no_revert(client, tmp_path):
    reading, undone, pid = _withdrawn_after_the_lock(client, tmp_path, unturned=True)
    lines = client.get(f"/api/projects/{pid}/methods").json()["lines"]
    assert not {reading["id"], undone["id"]} & {line["record_id"] for line in lines}
    listed, stated, text = _later(client, pid)
    assert (listed, stated) == ([undone["seq"]], [])
    assert text.endswith(f" 1 decision was {MADE}; decision #{undone['seq']} is left out of the "
                         f"methods, as it changes {NOW}")


def test_a_reading_in_force_at_the_lock_and_withdrawn_after_is_said(client, tmp_path):
    reading, undone, pid = _withdrawn_after_the_lock(client, tmp_path, unturned=False)
    lines = {line["record_id"]: line
             for line in client.get(f"/api/projects/{pid}/methods").json()["lines"]}
    assert lines[reading["id"]]["sentence"] == reading["sentence"]  # the plan as declared
    assert lines[undone["id"]]["after_estimates"] is True
    listed, stated, text = _later(client, pid)
    assert listed == stated == [undone["seq"]]
    assert text.endswith(f" 1 decision was {MADE} and marked so in the methods.")
