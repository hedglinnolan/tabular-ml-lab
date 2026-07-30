"""`GUIDED-041` — the reopen affordance answered the question it was reopening.

Decision B permits the Router to skip a question only where a high-confidence
finding makes a question of *fact* moot, and only if the skip is **visible and
reversible**. The visible half was built: a muted provenance row carrying its
reason, with *"Ask me anyway"* beside it.

The reversible half sent this:

    decide("set_task_type", "", {task_type: P.task_type})

`P.task_type` is **the engine's own reading** — the thing the user is reaching
past when they press the button. So pressing *"Ask me anyway"* recorded that
reading as the user's answer. The question left the plan as ANSWERED, the skip
disappeared because the question was gone, and the transcript then said a human
had confirmed something no human had looked at.

That is worse than having no affordance at all. A skip with no reopen is
honestly incomplete; a reopen that discards teaches that opening a skip loses
your place, and the next skip goes unopened.

## What the fix has to be

Not a flag. `unskip` is a **recorded decision** carrying the question key and no
answer, for the reason §09's recorded-absence rule gives about everything else
here: *"I did not accept the engine's reading of this"* is a sentence a methods
section can carry, and a mutated boolean is not. An `unskip` with nothing after
it is a question still open, which is what it should look like.

And it is generic in the key, so it closes the class rather than the task-type
instance — the same move `DRIVE-001` needed. Every rendered skip the Router
serves is reopenable the day it exists, including the pack-settled missingness
blocks, where a single skip stands for hundreds of columns and the cost of
being unable to reopen it is correspondingly larger.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _skipped(client, pid, step="data"):
    plan = client.get(f"/project/{pid}/interview?step={step}").json()
    return {q["key"]: q for q in plan["questions"] if q["status"] == "skipped"}


def _asked(client, pid, step="data"):
    plan = client.get(f"/project/{pid}/interview?step={step}").json()
    return {q["key"]: q for q in plan["questions"]
            if q["mode"] == "push" and q["status"] == "asked"}


def _upload(client, name):
    with open(DATA / f"{name}.csv", "rb") as fh:
        return client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()


# ── the effect, read back ────────────────────────────────────────────────────

def test_reopening_a_skipped_question_brings_it_back_asked_and_unanswered():
    """The read-back, and the assertion that would have failed before.

    Two facts, and the second is the one nine tests would have missed: the
    question is back, AND no answer was recorded for it. The old code satisfied
    neither, but a test written only against the first would have gone green the
    moment somebody made the reopen re-ask the question while still writing the
    engine's guess into the record.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})

    skipped = _skipped(client, pid)
    assert "confirm_task_type" in skipped, (
        "the task-type question is not skipped on this fixture, so there is "
        "nothing to reopen and this test proves nothing")

    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "unskip", "subject": "confirm_task_type",
                          "payload": {"key": "confirm_task_type"}})
    assert r.status_code == 200, r.text

    after = _asked(client, pid)
    assert "confirm_task_type" in after, (
        "the question did not come back; a reopen that does not reopen is the "
        "defect with a different implementation")
    assert "confirm_task_type" not in _skipped(client, pid)

    # THE HALF THE OLD CODE GOT WRONG. No answer may have been recorded, and in
    # particular not the engine's own reading.
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    assert "set_task_type" not in kinds, (
        "reopening the question recorded an answer to it — and the answer is "
        "the engine's own reading, which is the thing the user pressed the "
        "button to dispute")
    assert "unskip" in kinds


def test_the_reopen_survives_the_engine_still_being_certain():
    """A reopened question stays asked.

    The skip is granted on the engine's confidence, and the engine's confidence
    does not change when a human disagrees with it — so without this the
    question would be re-skipped on the very next render and the reopen would
    appear to do nothing at all. The user's asking outranks the engine's
    certainty, which is the same asymmetry §02 draws for the grain.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": "confirm_task_type"}})

    asked = _asked(client, pid)
    assert "confirm_task_type" in asked, (
        "the reopened question was skipped again on the very next render, so "
        "the reopen appears to the driver to do nothing at all")
    q = asked["confirm_task_type"]
    assert q["confidence"] == "high", (
        "the engine stopped being certain, so this test is no longer about "
        "what it says it is about")
    assert q["status"] == "asked"
    # And again, because the plan is recomputed per render and a reopen that
    # survives one render and not the next is the same defect, slower.
    assert "confirm_task_type" in _asked(client, pid), (
        "the reopened question was skipped again on a later render")


def test_answering_a_reopened_question_settles_it_and_keeps_the_reopen_recorded():
    """The reopen stays in the record after the answer.

    It is not bookkeeping: *"the user declined the engine's reading and answered
    this themselves"* is a different sentence from *"the user answered this"*,
    and the manuscript can carry the first only if the record keeps both.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": "confirm_task_type"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_task_type",
                      "payload": {"task_type": "classification"}})

    assert "confirm_task_type" not in _asked(client, pid)
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    assert "unskip" in kinds and "set_task_type" in kinds
    assert kinds.index("unskip") < kinds.index("set_task_type")


def test_a_pack_settled_missingness_block_is_reopenable_too():
    """The class, not the instance.

    One skip standing for 306 columns is where being unable to reopen costs the
    most, and it is a different code path from the task-type skip — so it is
    asserted rather than assumed to have come along.
    """
    client = _client()
    project = _upload(client, "metabolomics_untargeted")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "responder"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["metabolomics"]}})

    settled = [k for k in _skipped(client, pid, "preprocess")
               if k.startswith("missingness_settled::")]
    assert settled, (
        "no pack-settled missingness block on this fixture; the metabolomics "
        "left-censoring prior is what makes this test meaningful")
    key = settled[0]

    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": key}})
    assert key in _asked(client, pid, "preprocess"), (
        f"{key} did not come back asked")
    assert key not in _skipped(client, pid, "preprocess")


def test_the_router_refuses_to_skip_a_key_the_user_reopened():
    """Enforced in the one place Decision B lives, so a second skip site cannot
    forget it. `_skip_is_permitted` is where the constitution is checked rather
    than remembered."""
    assert router._skip_is_permitted("high", "task_type") is True
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["confirm_task_type"]) is False
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["something_else"]) is True


# ── what the driver presses ──────────────────────────────────────────────────

def test_the_page_sends_a_reopen_and_not_an_answer():
    """Read back off the page's own click handler.

    The defect was entirely in what the button sent, so a test that did not
    watch the wire would have missed it. This one dispatches at the real
    affordance and asserts the request body.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    project = client.get(f"/project/{pid}").json()
    plan = client.get(f"/project/{pid}/interview?step=data").json()

    body = H.run(
        """
        var html = __harness.html('skipNote');
        var m = /data-unskip="([^"]+)"/.exec(html);
        if (!m) throw new Error('no reopen affordance rendered');
        __harness.dispatch('click', __harness.target(
          {'data-unskip': m[1], 'data-unskip-title': 'x'}, ['again']));
        var posts = __harness.posts();
        __emit(posts.length ? posts[posts.length - 1] : null);
        """,
        routes={
            f"/project/{pid}": project,
            f"/project/{pid}/interview?step=data": plan,
            f"/project/{pid}/interview?step=explore": {"questions": []},
            f"/project/{pid}/evidence/missingness": {"cards": []},
        }, search=f"?project={pid}")

    assert body, "pressing the reopen affordance sent nothing"
    assert body["body"]["kind"] == "unskip", (
        "the reopen affordance still sends an answer instead of a reopen: "
        f"{body['body']['kind']}")
    assert body["body"]["payload"]["key"] == "confirm_task_type"

    # And the server accepts exactly that body.
    replay = client.post(f"/project/{pid}/decision", json=body["body"])
    assert replay.status_code == 200, replay.text
    assert "confirm_task_type" in _asked(client, pid)
