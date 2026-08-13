"""`DRIVE-038` — `<span class="chip arrived">[object Object]</span>`.

Run 3 saw it once on the positive-class finding, persisting across a lens
change. The chip is written in one place, `findingCard`'s `arrived` parameter,
and `esc()` calls `String(s)` on whatever it is handed — so an object arrives on
screen as developer text.

**The sentence this chip carries is the server's, quoted** (`DESIGN_LANGUAGE`
§05.1 rule 3). A value that is not a string is therefore not a sentence the
server composed, and rendering it asserts something false about where it came
from. Silence is the honest fallback; the console carries the fault, because a
user cannot act on it and whoever is debugging can.

## What was ruled out, and what is still open

The obvious path is not it: `stackCards(st.pushed, st.promoted,
st.promoted_because)` is the only three-argument call site, and
`attention.stack` returns `promoted_because` as a string on **every** branch —
an f-string at `attention.py:463`, returned as `x if promoted else ""` at
`:523`. The other two call sites pass two arguments, so `arrived` is `undefined`
and `esc()` correctly yields empty.

Run 3 also drove a **current page against a 28-hour-old API**, so the payload
feeding the chip was not the one this tree serves. **The trigger is not
reproduced and stays open on the row.** This file pins the defensive half, which
is owed either way — it is `TEST-083`'s shape, a positional argument in the
wrong slot silently stringified, for the second time in two loops.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _page_ids():
    return sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"',
                                 PAGE.read_text(encoding="utf-8"))))


_READER = """
var IDS = __IDS__;
var blob = "";
IDS.forEach(function(id){
  var e = document.getElementById(id); if (e) blob += (e.innerHTML || "");
});
__emit({blob: blob, profList: __harness.html("profList"),
        calls: __harness.calls().map(function(c){
          return {method: c.method, path: c.path}; })});
"""


def _routes(client, pid: str, doctor=None) -> Dict[str, Any]:
    """Every route the page fetches, to a fixpoint, with `doctor` applied to the
    project payload afterwards."""
    from turbotab import pageharness

    reader = _READER.replace("__IDS__", json.dumps(_page_ids()))
    routes: Dict[str, Any] = {}
    for step in ("data", "explore", "preprocess", "features"):
        path = f"/project/{pid}/interview?step={step}"
        resp = client.get(path)
        if resp.status_code == 200:
            routes[path] = resp.json()
    seen: set = set()
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
    if doctor is not None:
        key = f"/project/{pid}"
        routes[key] = doctor(json.loads(json.dumps(routes[key])))
    return routes


def _render(client, pid, doctor=None):
    from turbotab import pageharness

    reader = _READER.replace("__IDS__", json.dumps(_page_ids()))
    out = pageharness.run(reader, routes=_routes(client, pid, doctor),
                          search=f"?project={pid}")
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe no "
        f"absence from this run")
    return out


def _project(client):
    with (DATA / "clinical_labs.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_labs.csv", handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "readmitted"}})
    return pid


@pytest.fixture(autouse=True)
def _needs_node():
    from turbotab import pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")


def test_a_promoted_card_renders_the_servers_sentence(capsys):
    """The positive control, and it has to come first.

    Every assertion below is that `[object Object]` is absent, and absence is
    free over a render that produced no chip at all. This drives the chip into
    existence with a real string first.
    """
    from turbotab import api
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    pid = _project(client)

    sentence = "Moved up when you dismissed a card above."

    def promote(payload):
        stack = payload.get("explore_stack") or {}
        pushed = list(stack.get("pushed") or [])
        assert pushed, "this fixture serves no pushed stack; pick another"
        stack["promoted"] = [pushed[0]]
        stack["promoted_because"] = sentence
        payload["explore_stack"] = stack
        return payload

    out = _render(client, pid, promote)
    assert sentence in (out["profList"] or ""), (
        f"the promoted card does not carry the server's sentence:\n"
        f"  {(out['profList'] or '')[:300]}")
    with capsys.disabled():
        print(f"\n  string  → chip carries the sentence")


def test_an_object_in_that_slot_reaches_no_chip(capsys):
    """The guard. Same drive, same card, a non-string in the same field.

    Driven through a doctored project payload rather than by calling
    `findingCard` — it lives inside the page's closure, and a test that reached
    around the controller would be asserting about a function rather than about
    what a person sees.
    """
    from turbotab import api
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    pid = _project(client)

    def promote_with_an_object(payload):
        stack = payload.get("explore_stack") or {}
        pushed = list(stack.get("pushed") or [])
        assert pushed, "this fixture serves no pushed stack; pick another"
        stack["promoted"] = [pushed[0]]
        # What run 3's screen implies arrived here.
        stack["promoted_because"] = {"verb": "dismissed", "n": 1}
        payload["explore_stack"] = stack
        return payload

    out = _render(client, pid, promote_with_an_object)
    assert "[object Object]" not in out["blob"], (
        "an object reached the page as developer text — the chip stringified a "
        "value the server did not compose as a sentence")
    assert "chip arrived" not in (out["profList"] or ""), (
        "the chip rendered from a non-string; it must be silent instead")
    with capsys.disabled():
        print(f"\n  object  → no chip, no [object Object]")


def test_no_render_of_this_project_says_object_Object(capsys):
    """The sweep the row is really about, over the whole render rather than one
    element — `[object Object]` anywhere is the same defect wherever it lands."""
    from turbotab import api
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    pid = _project(client)
    out = _render(client, pid)
    n = out["blob"].count("[object Object]")
    assert n == 0, f"{n} occurrence(s) of [object Object] in this render"
    with capsys.disabled():
        print(f"\n  undoctored render: {n} occurrences across "
              f"{len(out['blob']):,} chars")
