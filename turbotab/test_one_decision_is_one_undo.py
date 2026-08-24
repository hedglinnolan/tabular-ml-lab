"""`DRIVE-033` — the transcript said one decision and the undo stack held nine.

A human applied *"read as binary"* to a nine-column group and pressed **"Undo
the last change"**. One column came back. The other eight stayed `Int64`, while
the apply's own tooltip had promised *"records one decision covering the whole
group. It can be undone."*

**A partial undo leaves the table in a state no decision describes** — the
record disagreeing with the frame, which is the one thing the transcript exists
to prevent.

## Established before it was fixed

The loop prompt did not reproduce this and said so. It reproduces: on a
nine-column boolean frame, `apply_bulk` changed 9 dtypes and `revert` restored
**1**. The cause is arithmetic rather than intent — `apply_fix_quietly` took an
undo entry per member, `revert_last_fix` pops exactly one, and no control
anywhere steps back a second time.

`apply_fix_quietly`'s own docstring argued for the per-member entries:
*reversibility is per frame and a bulk repair that could not be stepped back
would be nine changes the user has to accept together or not at all.* The
reasoning is sound and the affordance it assumes was never built, so what
shipped was nine changes the user could accept together and un-accept one at a
time.

## What this file also pins

`DRIVE-020`'s numbering half, because it is the same shape one surface over: a
card writing its own number instead of rendering the position the Router served.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _nine_column_frame() -> bytes:
    """The tester's shape: six `imputed_*`, two `meds_*`, and `gender`."""
    rng = np.random.default_rng(3)
    n = 400
    cols: Dict[str, object] = {
        f"imputed_{k}": rng.choice([True, False], n)
        for k in ("bmi", "bp_di", "bp_sy", "glu", "hdl", "tri")}
    cols["meds_hbp"] = rng.choice([True, False], n)
    cols["meds_chol"] = rng.choice([True, False], n)
    cols["gender"] = rng.choice([True, False], n)
    cols["age"] = rng.integers(18, 85, n)
    return pd.DataFrame(cols).to_csv(index=False).encode()


def _dtypes(payload) -> Dict[str, str]:
    return {c["name"]: c["dtype"] for c in payload["columns"] if c["name"] != "age"}


def test_undoing_a_bulk_repair_restores_every_column_it_changed(capsys):
    """The load-bearing claim, and it counts rather than sampling.

    Asserted as *every column the apply changed*, derived from the frames rather
    than from a list of names — a test that checked one column would have passed
    against the defect, since one column is exactly what came back.
    """
    client = _client()
    pid = client.post("/project", files={
        "file": ("t.csv", io.BytesIO(_nine_column_frame()), "text/csv")}
    ).json()["id"]

    before = _dtypes(client.get(f"/project/{pid}").json())

    plan = client.get(f"/project/{pid}/interview?step=data").json()
    groups = [q["key"] for q in plan["questions"]
              if q["key"].startswith("repair_bulk::")]
    assert groups, "no bulk repair group is served on this frame"
    kind = groups[0].split("::", 1)[1]
    members = [m["id"] for m in
               client.get(f"/project/{pid}/repair_group/{kind}").json()["members"]]
    assert len(members) >= 5, (
        f"only {len(members)} members; this claim is about a group large enough "
        f"for a partial undo to be visible")

    applied = client.post(f"/project/{pid}/decision",
                          json={"kind": "apply_bulk", "subject": kind,
                                "payload": {"findings": members}})
    assert applied.status_code == 200, applied.text[:300]
    after = _dtypes(applied.json())
    changed = [c for c in before if before[c] != after.get(c)]
    # THE PRECONDITION, FROM THE DATA. If the apply changed nothing the undo
    # assertion below would be true of a project where nothing happened.
    assert len(changed) == len(members), (
        f"the apply changed {len(changed)} columns for {len(members)} members")

    undone = client.post(f"/project/{pid}/decision",
                         json={"kind": "revert", "payload": {}})
    # NAMED, because the other way this breaks is the opposite one: remembering
    # nothing at all leaves the stack empty and `revert` refuses outright, which
    # is a group that cannot be undone rather than one undone in part.
    assert undone.status_code == 200, (
        f"the bulk apply left nothing to undo — revert answered "
        f"{undone.status_code}: {undone.text[:200]}")
    back = _dtypes(undone.json())

    still = sorted(c for c in changed if back.get(c) != before[c])
    assert not still, (
        f"one undo restored {len(changed) - len(still)} of {len(changed)} "
        f"columns; these are still changed and no decision describes the "
        f"table's state: {still}")
    with capsys.disabled():
        print(f"\n  applied {len(changed)} columns, one undo restored "
              f"{len(changed)}")


def test_a_single_apply_still_undoes_on_its_own(capsys):
    """The other side of the same stack, so the fix does not buy the group's
    undo with the individual one.

    `remember=False` is passed only to bulk members after the first; a lone
    `apply` goes through `apply_fix` and must be unaffected.
    """
    # A SHIPPED FIXTURE THAT ACTUALLY SERVES ONE, established by asking rather
    # than hoped for. The first version drove the nine-column frame, which
    # serves only the group — so it skipped, and a test that declines to look
    # when the thing is absent is `TEST-059`'s shape, green over the case it
    # was written for.
    client = _client()
    with (DATA / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]
    before = _dtypes(client.get(f"/project/{pid}").json())

    plan = client.get(f"/project/{pid}/interview?step=data").json()
    singles = [q["key"].split("::", 1)[1] for q in plan["questions"]
               if q["key"].startswith("repair::")]
    assert singles, (
        "clinical_risk.csv no longer serves a single repair, so this claim has "
        "nothing to drive — pick another fixture rather than skipping")
    one = singles[0]
    applied = client.post(f"/project/{pid}/decision",
                          json={"kind": "apply", "subject": one, "payload": {}})
    assert applied.status_code == 200, applied.text[:300]
    after = _dtypes(applied.json())
    changed = [c for c in before if before[c] != after.get(c)]
    assert changed, "the single apply changed nothing, so this proves nothing"

    undone = client.post(f"/project/{pid}/decision",
                         json={"kind": "revert", "payload": {}})
    assert undone.status_code == 200
    back = _dtypes(undone.json())
    assert all(back.get(c) == before[c] for c in changed), (
        "a single apply no longer undoes cleanly")
    with capsys.disabled():
        print(f"\n  single apply: {len(changed)} column(s), restored")


def test_the_undo_stack_holds_one_entry_for_one_bulk_decision(capsys):
    """The mechanism, named — so a future change that re-adds per-member entries
    fails here with the reason rather than only at the behavior above."""
    from turbotab import api

    client = _client()
    pid = client.post("/project", files={
        "file": ("t.csv", io.BytesIO(_nine_column_frame()), "text/csv")}
    ).json()["id"]
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    kind = [q["key"] for q in plan["questions"]
            if q["key"].startswith("repair_bulk::")][0].split("::", 1)[1]
    members = [m["id"] for m in
               client.get(f"/project/{pid}/repair_group/{kind}").json()["members"]]
    client.post(f"/project/{pid}/decision",
                json={"kind": "apply_bulk", "subject": kind,
                      "payload": {"findings": members}})

    # `api.STORE` is the one the routes use — read through it rather than
    # skipping, so this cannot go quiet the way the first draft did.
    project = api.STORE.get(pid)
    assert len(project.applied_fixes) == 1, (
        f"{len(project.applied_fixes)} undo entries for one decision covering "
        f"{len(members)} features: {project.applied_fixes}")
    with capsys.disabled():
        print(f"\n  {len(members)} features, {len(project.applied_fixes)} undo entry")


# ── `DRIVE-020`, the numbering half ──────────────────────────────────────────

def test_the_target_card_renders_the_position_the_router_served(capsys):
    """A surface was numbering itself instead of rendering what it was served.

    The lens card and the target card both showed **01** in two consecutive
    human drives. `ml/router.py`'s `SEQUENCE` is the constitution's table in the
    one place a renderer can read it — `state_lens` 01, `choose_target` 02 — and
    the generic channel already renders `q.seq`. The target card held its own
    copy in the markup.
    """
    from turbotab import api, pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    with (DATA / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]

    plan = client.get(f"/project/{pid}/interview?step=data").json()
    served = {q["key"]: q.get("seq") for q in plan["questions"]}
    assert served.get("state_lens") == "01" and served.get("choose_target") == "02", (
        f"the Router's own positions moved; this claim quotes them: {served}")

    ids = sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"',
                                PAGE.read_text(encoding="utf-8"))))
    routes: Dict[str, object] = {}
    for step in ("data", "explore", "preprocess", "features"):
        path = f"/project/{pid}/interview?step={step}"
        resp = client.get(path)
        if resp.status_code == 200:
            routes[path] = resp.json()
    reader = ('var IDS = ' + json.dumps(ids) + ';\n'
              'var blob = ""; IDS.forEach(function(id){\n'
              '  var e = document.getElementById(id); if (e) blob += (e.innerHTML || ""); });\n'
              '__emit({blob: blob, asked: __harness.html("askedQuestions"),\n'
              '        targetNum: (__harness.el("targetNum") || {}).textContent,\n'
              '        calls: __harness.calls().map(function(c){\n'
              '          return {method: c.method, path: c.path}; })});')
    seen: set = set()
    out = {}
    for _ in range(6):
        out = pageharness.run(reader, routes=routes, search=f"?project={pid}")
        calls = {(c["method"], c["path"]) for c in out["calls"]}
        if calls <= seen:
            break
        seen |= calls
        for call in out["calls"]:
            if call["method"] != "GET" or call["path"] in routes:
                continue
            resp = client.get(call["path"])
            if resp.status_code == 200:
                try:
                    routes[call["path"]] = resp.json()
                except ValueError:
                    pass
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe nothing here")

    lens = re.findall(r'data-asked="state_lens"[^>]*>.*?sub-num">([^<]*)<',
                      out["asked"] or "", re.S)
    assert lens, "the lens card did not render, so the comparison has one side"
    assert out["targetNum"] == served["choose_target"], (
        f"the target card renders {out['targetNum']!r} and the Router serves "
        f"{served['choose_target']!r}")
    assert lens[0] == served["state_lens"]
    assert out["targetNum"] != lens[0], (
        f"the lens and the target both render {lens[0]!r}, which is DRIVE-020 "
        f"and is what a human read in two consecutive drives")
    with capsys.disabled():
        print(f"\n  lens {lens[0]} · target {out['targetNum']} — "
              f"both from the Router")
