"""`DRIVE-026` — the third key in `HANDLED_QUESTION_KEYS` with no section.

`HANDLED_QUESTION_KEYS` declares *"this key has a dedicated section above"* and
`renderAsked` believes it unconditionally. Three keys were in it with nothing
behind them: `state_grain` and `state_eligibility`, which is why a human could
not fit a model at all, and `choose_preparation_mode`, which L58-B's own guard
found on its first run and pinned as a **strict xfail** rather than leaving a
green suite over an unreachable question.

Measured then: driven to `asked` with the seal drawn and two models selected,
the title appeared **0 times** across 93,880 characters, and `#prepPlan` — the
overlay whose whole job is to draw an asked question no other surface holds —
rendered **0 characters**, because `drawnElsewhere` reads the same list and the
list silences the safety net as well as the renderer.

## What was missing was the surface

The server side has been complete the whole time: `set_preparation_mode` is
handled at `api.py:726` and folded into `answered` at `:3087`. So this is the
same build `state_grain` and `state_eligibility` got — the key rides the generic
channel, which draws from what the Router says about a question rather than from
a copy of it in the page.

**And the values are keys, not prose.** `ml/router.py` serves `per_model` and
`uniform` as `option_values`, read off `AnalysisProject.PREPARATION_MODES`
rather than restated — `DRIVE-023` was exactly this defect on the eligibility
question, where `to_dict`'s `option_values or options` fallback put the prose on
the wire and every press took a 400.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _page_ids():
    return sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"',
                                 PAGE.read_text(encoding="utf-8"))))


_READER = """
var IDS = __IDS__;
var blob = "";
IDS.forEach(function(i){ var e = document.getElementById(i); if (e) blob += (e.innerHTML || ""); });
__emit({blob: blob, prepPlan: __harness.html("prepPlan"),
        asked: Array.prototype.slice.call(document.querySelectorAll("[data-asked]"))
                 .map(function(b){ return b.getAttribute("data-asked"); }),
        posts: __harness.posts(),
        calls: __harness.calls().map(function(c){
          return {method: c.method, path: c.path}; })});
"""


def _at_preparation_mode(client) -> str:
    """A project where the Router serves `choose_preparation_mode` as asked."""
    with (DATA / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_target", {"column": "age"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})):
        resp = client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload})
        assert resp.status_code == 200, (kind, resp.text[:200])
    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf.get("groups", []) for m in (g.get("models") or [])]
    assert keys, "the shelf served no models"
    client.post(f"/project/{pid}/decision",
                json={"kind": "select_models", "payload": {"models": keys[:2]}})
    return pid


def _render(client, pid, tail: str = "") -> Dict[str, Any]:
    from turbotab import pageharness

    reader = _READER.replace("__IDS__", json.dumps(_page_ids()))
    routes: Dict[str, Any] = {}
    for step in ("data", "explore", "preprocess", "features"):
        path = f"/project/{pid}/interview?step={step}"
        resp = client.get(path)
        if resp.status_code == 200:
            routes[path] = resp.json()
    seen: set = set()
    out: Dict[str, Any] = {}
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
    if tail:
        out = pageharness.run(tail + "\n" + reader, routes=routes,
                              search=f"?project={pid}")
        if out.get("error"):
            raise AssertionError(f"the page could not be driven: {out['error']}")
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe no absence")
    return out


@pytest.fixture(autouse=True)
def _needs_node():
    from turbotab import pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")


def test_the_router_serves_keys_rather_than_prose(capsys):
    """`DRIVE-023`'s defect, checked before the card is built on top of it."""
    from turbotab.project import AnalysisProject

    client = _client()
    pid = _at_preparation_mode(client)
    plan = client.get(f"/project/{pid}/interview?step=preprocess").json()
    question = next(q for q in plan["questions"]
                    if q["key"] == "choose_preparation_mode")
    assert question["status"] == "asked"
    assert question["option_values"] == list(AnalysisProject.PREPARATION_MODES), (
        f"the wire carries {question['option_values']!r}; "
        f"set_preparation_mode accepts "
        f"{list(AnalysisProject.PREPARATION_MODES)!r}")
    assert len(question["options"]) == len(question["option_values"])
    with capsys.disabled():
        print(f"\n  values {question['option_values']} "
              f"for {len(question['options'])} options")


def test_the_card_renders_and_its_press_records_the_mode(capsys):
    """Served → rendered → pressed → the body the page composed → the record.

    The assertion is that the RECORD changed, not that the page contains a
    string.
    """
    client = _client()
    pid = _at_preparation_mode(client)

    out = _render(client, pid)
    assert "choose_preparation_mode" in set(out["asked"]), (
        "the preparation-mode question renders nowhere; DRIVE-026 is back")

    from turbotab import pageharness

    buttons = [b for b in pageharness.elements(out["blob"])
               if b.get("data-answer-key") == "choose_preparation_mode"]
    values = [b.get("data-answer-value") for b in buttons]
    from turbotab.project import AnalysisProject

    assert set(values) == set(AnalysisProject.PREPARATION_MODES), values

    pressed = _render(client, pid, '''
var b = document.querySelectorAll('[data-answer-key="choose_preparation_mode"]')
          .filter(function(e){ return e.getAttribute("data-answer-value") === "per_model"; })[0];
if (!b) { __emit({error: "no per_model control"}); }
__harness.dispatch("click", b);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
''')
    posts = pressed["posts"]
    assert len(posts) == 1, f"one press, {len(posts)} request(s)"
    assert posts[0]["body"]["kind"] == "set_preparation_mode"
    assert posts[0]["body"]["payload"] == {"mode": "per_model"}

    replayed = client.post(f"/project/{pid}/decision", json=posts[0]["body"])
    assert replayed.status_code == 200, replayed.text[:300]
    assert replayed.json()["preparation_mode"] == "per_model"
    said = [d["text"] for d in replayed.json()["decisions"]
            if d["kind"] == "set_preparation_mode"][-1]
    assert said.strip() and len(said) > 25, said
    with capsys.disabled():
        print(f"\n  pressed per_model → recorded, {len(said)}-char sentence")


def test_the_other_option_records_too(capsys):
    """Both options post in one press — asserted, because a card that offered
    two and could only take one is `GUIDED-006`'s shape."""
    client = _client()
    pid = _at_preparation_mode(client)
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_preparation_mode",
                             "payload": {"mode": "uniform"}})
    assert resp.status_code == 200, resp.text[:200]
    assert resp.json()["preparation_mode"] == "uniform"
    with capsys.disabled():
        print("\n  uniform records too")


def test_the_question_leaves_the_plan_once_answered(capsys):
    """And the card carries an acknowledgment, so it does not simply vanish —
    `DRIVE-004`, which is why `ACK_LABEL` gained a row."""
    client = _client()
    pid = _at_preparation_mode(client)
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_preparation_mode", "payload": {"mode": "per_model"}})
    plan = client.get(f"/project/{pid}/interview?step=preprocess").json()
    still = [q["key"] for q in plan["questions"]
             if q["key"] == "choose_preparation_mode" and q["status"] == "asked"]
    assert not still, "the question is still asked after being answered"

    page = PAGE.read_text(encoding="utf-8")
    start = page.index("var ACK_LABEL = {")
    body = page[start:page.index("\n  };", start)]
    assert "set_preparation_mode" in body, (
        "the answered card has no acknowledgment row, so it vanishes when "
        "pressed — DRIVE-004")
    with capsys.disabled():
        print("\n  answered: leaves the plan, and is acknowledged")


def test_the_key_is_no_longer_claimed_by_a_section_that_does_not_exist(capsys):
    """The list itself, named — this is the third key it claimed falsely."""
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("var HANDLED_QUESTION_KEYS = [")
    listed = re.findall(r'"([^"]+)"', page[start:page.index("];", start)])
    assert "choose_preparation_mode" not in listed, (
        "`choose_preparation_mode` is back in HANDLED_QUESTION_KEYS, which "
        "asserts a dedicated section draws it. None exists — that is DRIVE-026")
    assert "state_grain" not in listed and "state_eligibility" not in listed
    with capsys.disabled():
        print(f"\n  the list now claims {len(listed)}: {listed}")
