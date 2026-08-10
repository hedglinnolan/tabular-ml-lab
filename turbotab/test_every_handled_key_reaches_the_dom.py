"""`DRIVE-022` — a list membership stood in for a renderer, and the guard
checked the membership.

## What was wrong with the guard this replaces

`test_the_page_renders_every_question_the_router_can_serve` asserts that every
key the Router can serve is **in** one of two lists. Its own failure message
names the loophole:

> Add the key to ANSWERABLE … **or to `HANDLED_QUESTION_KEYS` if a dedicated
> section already asks it.**

The `if` is the unchecked half. `HANDLED_QUESTION_KEYS`' comment says every key
in it has *"a dedicated section above"*, `renderAsked` believes that
unconditionally and filters the key out — so **adding a key to that list is the
documented way to make the guard green without building anything.** Three keys
were in it with no section: `state_grain`, `state_eligibility` and
`choose_preparation_mode`. The first two are why a human could not fit a model
through the Guided door at all (`DRIVE-017`).

The name of the old test carries a consequence verb — *renders* — that nothing
in its body observes. That is trap #3b, and this file is the assertion the name
was promising.

## What this checks instead

For every key in `HANDLED_QUESTION_KEYS`:

1. **Drive a real project into a state where the Router serves that key
   `asked`** — and assert that from the served plan, not from the drive's
   intent. A precondition established from the data, then a consequence
   asserted unconditionally, is trap #3d's rule.
2. **Render the page's real controller** over that project under
   `pageharness`, with every route it fetches answered from the same
   `TestClient`.
3. **Assert a control that can ANSWER the question is in the DOM** — a
   `data-*` attribute the page actually delegates clicks on. Not the title, not
   a heading: `DESIGN_LANGUAGE` aside, a sentence about a question is not a way
   to answer it, and `GUIDED-006` is the row for controls that look like one and
   are not.

**A key that cannot be driven into an asked state is the finding, not a skip.**
`DRIVES` and `ANSWERING_CONTROL` are required for every key in the list; a key
missing from either fails here, naming itself. That is what makes appending to
`HANDLED_QUESTION_KEYS` cost something.

## The reader, and why it is built this way

Three facts about `pageharness` that cost two wrong answers before they were
written down, none of them documented in that module:

* `document.querySelectorAll("[id]")` returns **nothing** and
  `document.body.innerHTML` is **empty**. A reader built on either reports
  every question as unrendered. So the ids are harvested from the markup and
  handed to the reader.
* `__harness.matches(el, sel)` is a **predicate, not a query**.
* **Assert the reader read something before believing any absence.** Every
  claim in here is an absence claim, and an absence claim over an empty string
  is free.

`state_lens` is the positive control: it rides the generic channel, it is not in
`HANDLED_QUESTION_KEYS`, and it must render. If it does not, the rig is broken
and no verdict in this file means anything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"

#: How to reach the state where the Router serves each key `asked`: the fixture,
#: the decisions that get there, and the interview step to read.
#:
#: Every key in `HANDLED_QUESTION_KEYS` needs an entry. A key with none fails —
#: `test_every_handled_key_has_a_drive_and_a_control`.
DRIVES: Dict[str, Tuple[str, List[Tuple[str, dict]], str]] = {
    "choose_target": ("clinical_risk.csv", [], "data"),
    "confirm_task_type": (
        "clinical_risk.csv", [("set_target", {"column": "age"})], "data"),
    "choose_features": (
        "clinical_risk.csv", [("set_target", {"column": "age"})], "features"),
    # The shelf is gated on the seal and says so at Preprocess — *"models are
    # chosen after the held-out set is sealed"* — so the drive seals first. That
    # gate is the reason the Guided door showed a Train heading with no shelf
    # before L58: the seal was unreachable, not the shelf missing.
    "choose_models": (
        "clinical_risk.csv",
        [("set_target", {"column": "age"}),
         ("set_grain", {"answer": "one_row_per_person"}),
         ("set_eligibility", {"answer": "everyone"}),
         ("seal", {})],
        "preprocess"),
    # Asked once the models are chosen. Driven with the shelf's own first two
    # keys rather than a hard-coded model name, because the shelf is a function
    # of the task type and a name would drift with the registry.
    "choose_preparation_mode": (
        "clinical_risk.csv",
        [("set_target", {"column": "age"}),
         ("set_grain", {"answer": "one_row_per_person"}),
         ("set_eligibility", {"answer": "everyone"}),
         ("seal", {}),
         ("select_models", "__FIRST_TWO_SHELF_MODELS__")],
        "preprocess"),
}

#: The `data-*` attribute of a control that ANSWERS the question, per key.
#:
#: Asserted to be one the page delegates clicks on, so an entry cannot be
#: satisfied by a heading or a sentence that happens to carry a matching string.
#:
#: **An entry may pin a VALUE** — `data-answer-key="state_grain"` — and where a
#: question rides a shared channel it must. Five sections each emit their own
#: attribute and those are unique by construction; the generic channel emits one
#: attribute for every question it draws, so a bare `data-answer-key` would be
#: satisfied by *any other question's* card. That is the matcher-fires-on-prose
#: family one layer in, and the difference decides whether a re-listed
#: `state_grain` is caught.
ANSWERING_CONTROL: Dict[str, str] = {
    "choose_target": "data-target-col",
    "confirm_task_type": "data-task",
    "choose_features": "data-feat-settle",
    "choose_models": "data-pick-model",
    "choose_preparation_mode": "data-prep-mode",
}


def _attr_name(declaration: str) -> str:
    """The attribute, without any value pinned to it."""
    return declaration.split("=", 1)[0]


def _needle(declaration: str) -> str:
    """What to look for in the render.

    A bare attribute is searched as `attr="`, so `data-task` cannot be
    satisfied by `data-task-something-else`. A pinned value is searched
    verbatim.
    """
    return declaration if "=" in declaration else declaration + '="'

#: `DRIVE-026`. The third key in the list with no section, found by this file on
#: its first run and left as a failing test rather than a green suite over an
#: unreachable question — trap #1's rule, and `GUIDED-119`'s `xfail(strict=True)`
#: is the model. Strict, so the day the row is built this file goes red and the
#: entry has to be removed rather than quietly outliving the defect.
#:
#: Measured: driven to `asked` on `clinical_risk.csv` with the seal drawn and
#: two models selected, `GET /interview?step=preprocess` serves it `asked`, and
#: across 93,880 characters of rendered page the title appears 0 times,
#: `data-answer-key="choose_preparation_mode"` 0 times, and the strings
#: `preparation_mode` and `prep-mode` 0 times. `#prepPlan` — the overlay whose
#: whole job is to draw an asked question no other surface holds — is 0
#: characters, because `drawnElsewhere` also reads `HANDLED_QUESTION_KEYS`. The
#: list silences the safety net as well as the renderer.
NOT_BUILT: Dict[str, str] = {
    "choose_preparation_mode":
        "DRIVE-026 — the preparation-mode row does not exist. The page never "
        "names `set_preparation_mode`, and `drawnElsewhere` reads the same "
        "list, so `renderPreprocessPlan` will not draw its open row either.",
}


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _handled_keys() -> List[str]:
    """Read out of the page, never restated here. A list written in one place
    and applied in another is the same silence as a capability with no row."""
    text = _page()
    start = text.index("var HANDLED_QUESTION_KEYS = [")
    return re.findall(r'"([^"]+)"', text[start:text.index("];", start)])


def _page_ids() -> List[str]:
    """`document.querySelectorAll("[id]")` is empty under the shim, so the
    reader is handed the ids rather than asked to find them."""
    return sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', _page())))


_READER = r"""
var IDS = __IDS__;
var blob = "";
IDS.forEach(function(id){
  var e = document.getElementById(id);
  if (e) blob += (e.innerHTML || "");
});
__emit({blob: blob,
        keys: Array.prototype.slice.call(document.querySelectorAll("[data-answer-key]"))
                .map(function(b){ return b.getAttribute("data-answer-key"); }),
        calls: __harness.calls().map(function(c){
          return {method: c.method, path: c.path}; })});
"""


def _serve(client, calls, routes):
    for call in calls:
        if call["method"] != "GET" or call["path"] in routes:
            continue
        try:
            resp = client.get(call["path"])
        except Exception:
            continue
        if resp.status_code == 200:
            try:
                routes[call["path"]] = resp.json()
            except ValueError:
                pass
    return routes


def _render(client, pid: str) -> Dict[str, Any]:
    """Every route the page fetches, answered to a FIXPOINT, then the render.

    Pass zero is seeded with the four interview plans. With no route answered at
    all the controller throws a `TypeError` inside `paintPalette` —
    `plan.questions` of `{}` — and exits non-zero, so there is no first pass to
    read calls from. Each later pass unlocks fetches the previous one could not
    reach, so it iterates until the call set stops growing. Serving fewer routes
    than that makes a card which failed for want of data read as a defect.
    """
    from turbotab import pageharness

    reader = _READER.replace("__IDS__", json.dumps(_page_ids()))
    search = f"?project={pid}"
    routes: Dict[str, Any] = {}
    for step in ("data", "explore", "preprocess", "features"):
        path = f"/project/{pid}/interview?step={step}"
        resp = client.get(path)
        if resp.status_code == 200:
            routes[path] = resp.json()

    seen: set = set()
    out: Dict[str, Any] = {}
    for _ in range(6):
        out = pageharness.run(reader, routes=routes, search=search)
        calls = {(c["method"], c["path"]) for c in out["calls"]}
        if calls <= seen:
            break
        seen |= calls
        routes = _serve(client, out["calls"], routes)
    out["n_routes"] = len(routes)
    return out


def _drive(client, key: str) -> Tuple[str, dict]:
    """Reach the state, and return the project id with the served question."""
    fixture, decisions, step = DRIVES[key]
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in decisions:
        if payload == "__FIRST_TWO_SHELF_MODELS__":
            shelf = client.get(f"/project/{pid}/models").json()
            keys = [m["key"] for g in shelf.get("groups", [])
                    for m in (g.get("models") or [])]
            assert keys, "the shelf served no models, so the drive cannot proceed"
            payload = {"models": keys[:2]}
        resp = client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload})
        assert resp.status_code == 200, (key, kind, resp.status_code, resp.text[:300])
    plan = client.get(f"/project/{pid}/interview?step={step}").json()
    served = [q for q in plan["questions"] if q["key"] == key]
    assert served, (
        f"{key} is in HANDLED_QUESTION_KEYS and the drive in DRIVES does not "
        f"reach a plan that serves it. A key that cannot be driven into an "
        f"asked state is itself the finding — fix the drive or file the key.")
    return pid, served[0]


# ── the instrument's own preconditions ───────────────────────────────────────

def test_every_handled_key_has_a_drive_and_a_control():
    """The clause that makes appending to the list cost something.

    Without this, a key added to `HANDLED_QUESTION_KEYS` with no `DRIVES` entry
    would simply not be checked — the old guard's loophole rebuilt one file
    over, which is trap #2 arriving in the guard written to replace it.
    """
    keys = set(_handled_keys())
    assert keys, "HANDLED_QUESTION_KEYS could not be read out of the page"
    missing_drive = sorted(keys - set(DRIVES))
    missing_control = sorted(keys - set(ANSWERING_CONTROL))
    assert not missing_drive, (
        f"no drive reaches these keys, so nothing checks that they render: "
        f"{missing_drive}. Add a DRIVES entry that puts the Router in a state "
        f"where the key is served `asked`.")
    assert not missing_control, (
        f"no answering control is declared for these keys: {missing_control}. "
        f"Name the `data-*` attribute of the control a person presses to "
        f"answer the question.")


def test_every_declared_control_is_one_the_page_delegates():
    """An entry in `ANSWERING_CONTROL` names a CONTROL, not a string.

    Without this the map could be satisfied by a heading whose markup happens
    to contain the attribute name — trap #3, a fixture manufacturing the thing
    whose absence is the defect, in the declaration rather than in the fixture.
    """
    from turbotab import (
        test_every_control_the_page_delegates_survives_being_pressed as base)

    delegated = set(base.delegated_attributes(_page()))
    assert len(delegated) >= 50, (
        f"only {len(delegated)} delegated attributes were derived; the "
        f"extractor is broken and every check below would pass vacuously")
    for key, declaration in ANSWERING_CONTROL.items():
        if key in NOT_BUILT:
            continue
        attr = _attr_name(declaration)
        assert attr in delegated, (
            f"{key} declares {declaration!r} as its answering control and no "
            f"click delegate dispatches on {attr!r}, so pressing it does "
            f"nothing")


def test_the_positive_control_renders():
    """`state_lens` rides the generic channel and must reach the DOM.

    Every assertion in this file is an absence claim, and an absence claim over
    a broken rig is free. If this fails, no verdict below means anything.
    """
    from turbotab import api, pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    with (DATA / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]
    out = _render(client, pid)
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe no "
        f"absence from this rig")
    assert "state_lens" in set(out["keys"]), (
        "the positive control did not render, so the rig cannot tell a "
        "question that is missing from one it cannot see")


# ── the check `DRIVE-022` says was missing ───────────────────────────────────

@pytest.mark.parametrize("key", sorted(DRIVES))
def test_a_handled_key_reaches_a_control_in_the_dom(key, capsys):
    """The assertion the old guard's name was promising.

    A key in `HANDLED_QUESTION_KEYS` claims a dedicated section draws it. This
    drives the page into the state where the Router serves it and looks for a
    control that answers it.
    """
    from turbotab import api, pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    if key in NOT_BUILT:
        pytest.xfail(NOT_BUILT[key])

    client = TestClient(api.app)
    pid, question = _drive(client, key)
    # Established from the DATA and asserted, then the consequence asserted
    # unconditionally. A test that returned early on `status != "asked"` would
    # go quiet exactly where the defect lives (`TEST-059`).
    assert question["status"] == "asked", (
        f"{key} is served {question['status']!r} in this drive; the check needs "
        f"it `asked`")

    out = _render(client, pid)
    assert len(out["blob"]) > 5_000, (
        f"{key}: the reader collected {len(out['blob'])} characters over "
        f"{out['n_routes']} routes; believe no absence from this run")

    declaration = ANSWERING_CONTROL[key]
    found = out["blob"].count(_needle(declaration))
    with capsys.disabled():
        print(f"\n  {key:<26} {declaration:<32} {found} control(s) · "
              f"{len(out['blob']):,} chars · {out['n_routes']} routes")
    assert found, (
        f"the Router serves {key!r} as `asked` and no control matching "
        f"{_needle(declaration)!r} is anywhere in the {len(out['blob']):,} "
        f"characters this render produced. `HANDLED_QUESTION_KEYS` says a "
        f"dedicated section draws it; nothing does. That is DRIVE-022, and the "
        f"remedy is to build the section or move the key to ANSWERABLE — not "
        f"to adjust this list.")


def test_the_two_keys_this_loop_unlisted_are_gone_from_the_list():
    """`DRIVE-017`, named specifically.

    The general check above would pass again the day somebody put `state_grain`
    back — it would simply have no `DRIVES` entry and the first test would
    catch that instead. This says the thing directly, because the grain and
    eligibility cards are the two that stopped a human reaching a model.
    """
    handled = set(_handled_keys())
    assert "state_grain" not in handled, (
        "`state_grain` is back in HANDLED_QUESTION_KEYS. That list asserts a "
        "dedicated section draws it, and none exists — this is DRIVE-017, "
        "which is a human unable to answer question 03 at all.")
    assert "state_eligibility" not in handled
    text = _page()
    start = text.index("var ANSWERABLE = {")
    body = text[start:text.index("\n  };", start)]
    assert '"set_grain"' in body or "set_grain" in body
    assert "set_eligibility" in body
