"""`DRIVE-017` — a human could not fit a model through the Guided door, and the
reason was two questions that rendered nowhere.

A product-owner drive on a 21,849 × 29 NHANES file reached the interview, was
asked lens, target, task type and purpose, and then landed at Explore. No grain
card, no eligibility card, no seal. Question 03 of the pre-seal sequence was
served `asked` by the Router on every render and drawn by nothing, so the seal —
which refuses until the grain is answered — was unreachable, and so was every
model behind it.

The cause was one line: `state_grain` and `state_eligibility` sat in
`HANDLED_QUESTION_KEYS`, whose comment says every key in it has *"a dedicated
section above"*. `renderAsked` believes that unconditionally and filters the key
out. **No such section had ever been written**, and no test could tell, because
the guard for this checked list membership rather than the renderer
(`DRIVE-022`, and `test_every_handled_key_reaches_the_dom.py` is its remedy).

This file is the consumer half. It drives the page's real controller and asserts
**the record changed** — served, rendered, pressed, the body the page composed
replayed against the real API, and the project read back. A page that contains
the string `state_grain` would satisfy nothing here.

## Two fixture shapes, and the one not covered

`GUIDED-097`: every claim about a journey step runs against at least two
fixtures of different shape. `clinical_risk.csv` is the happy path — a
continuous target and one row per person, so the answer records in one press.
`clinical_longitudinal.csv` is the contradiction — 200 subjects across 600 rows,
so `one_row_per_person` returns a 409 whose exits travel with it.

**Not covered here:** a classification target. The grain question does not read
the target's dtype at all — `set_grain` takes the answer and the frame's shape —
so the shape that matters for this claim is the *table's*, and both fixtures are
driven. A string-outcome fixture is covered by the Train claims.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _page_ids() -> List[str]:
    return sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', _page())))


_READER = r"""
var IDS = __IDS__;
var blob = "";
IDS.forEach(function(id){
  var e = document.getElementById(id);
  if (e) blob += (e.innerHTML || "");
});
__emit({blob: blob,
        asked: Array.prototype.slice.call(document.querySelectorAll("[data-asked]"))
                 .map(function(b){ return b.getAttribute("data-asked"); }),
        refusal: __harness.html("refusal"),
        refusal_class: (__harness.el("refusal") || {}).className,
        at_grain: __harness.html("ac-ans-state_grain"),
        upload_class: (__harness.el("sub-upload") || {}).className,
        upErr: (__harness.el("upErr") || {}).textContent,
        teach_body: __harness.html("teachprobe") ||
                    ((document.querySelectorAll('[data-teach-body="state_grain"]')[0] || {}).innerHTML || ""),
        posts: __harness.posts(),
        calls: __harness.calls().map(function(c){ return {method: c.method, path: c.path}; })});
"""


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, fixture: str) -> str:
    with (DATA / fixture).open("rb") as handle:
        resp = client.post("/project", files={
            "file": (fixture, handle, "text/csv")})
    assert resp.status_code == 200, resp.text[:300]
    return resp.json()["id"]


def _routes(client, pid: str, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """Every route the page fetches, answered to a fixpoint from the real API.

    Pass zero is seeded with the four interview plans: with nothing answered the
    controller throws a `TypeError` inside `paintPalette` on `plan.questions` of
    `{}`, so there is no first pass to read calls from. Serving fewer routes
    than the page asks for makes a card that failed for want of data read as a
    defect, which is the opposite of what this file is for.
    """
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
    routes.update(extra or {})
    return routes


def _render(client, pid: str, tail: str = "", extra=None) -> Dict[str, Any]:
    from turbotab import pageharness

    body = tail + "\n" + _READER.replace("__IDS__", json.dumps(_page_ids()))
    out = pageharness.run(body, routes=_routes(client, pid, extra),
                          search=f"?project={pid}")
    # A PRESS WITH NO TARGET IS THE FINDING, AND IT MUST SAY SO. `__emit` ENDS
    # the run, so a tail that cannot find its control short-circuits the reader
    # and every field below becomes a `KeyError` — a red for the wrong reason
    # sitting exactly where the interesting red belongs. Caught and named.
    if isinstance(out, dict) and out.get("error"):
        raise AssertionError(
            f"the page could not be driven: {out['error']}. The control this "
            f"press needs is not in the render.")
    # THE READER'S OWN GUARD. Every claim below is an absence or a presence
    # claim over this string, and both are free over an empty one.
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe nothing "
        f"this run reports")
    return out


_PRESS = '''
var b = document.querySelectorAll('[data-answer-key="%s"]')
          .filter(function(e){ return e.getAttribute("data-answer-value") === "%s"; })[0];
if (!b) { __emit({error: "no control for %s = %s"}); }
__harness.dispatch("click", b);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
'''


def _press(client, pid, key, value, extra=None):
    return _render(client, pid, _PRESS % (key, value, key, value), extra)


@pytest.fixture(autouse=True)
def _needs_node():
    from turbotab import pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")


# ── the consumer, end to end ─────────────────────────────────────────────────

def test_a_human_can_answer_the_grain_question_and_reach_eligibility(capsys):
    """The whole of `DRIVE-017`, as a driven sequence.

    Served → rendered → pressed → the body the page composed → replayed against
    the real API → the record changed → the next question appears → rendered →
    pressed → recorded → the seal, which was the unreachable thing.
    """
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_target", "payload": {"column": "age"}})
    assert resp.status_code == 200

    # 1 · the Router serves it, established from the plan rather than assumed.
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    grain = next(q for q in plan["questions"] if q["key"] == "state_grain")
    assert grain["status"] == "asked" and grain["seq"] == "03"

    # 2 · the page renders it, with a control per option carrying the VALUE.
    out = _render(client, pid)
    assert "state_grain" in set(out["asked"]), (
        "the grain card is not in the render; DRIVE-017 is back")
    values = re.findall(
        r'data-answer-key="state_grain" [^>]*data-answer-value="([^"]+)"',
        out["blob"])
    assert set(values) == set(grain["option_values"]), (
        f"the card renders {sorted(set(values))} and the Router serves "
        f"{sorted(grain['option_values'])}; the shelf is never shortened")

    # 3 · pressing it composes a body the API accepts, and the record changes.
    pressed = _press(client, pid, "state_grain", "one_row_per_person")
    posts = pressed["posts"]
    assert len(posts) == 1, f"one press, {len(posts)} request(s): {posts}"
    assert posts[0]["body"]["kind"] == "set_grain"
    assert posts[0]["body"]["payload"] == {"answer": "one_row_per_person"}

    replay = client.post(f"/project/{pid}/decision", json=posts[0]["body"])
    assert replay.status_code == 200, replay.text[:300]
    record = replay.json()
    assert record["grain"]["answer"] == "one_row_per_person"
    assert record["grain"]["basis"] == "cross_sectional"

    # 4 · eligibility now appears, and it is drawn.
    plan2 = client.get(f"/project/{pid}/interview?step=data").json()
    elig = next(q for q in plan2["questions"] if q["key"] == "state_eligibility")
    assert elig["status"] == "asked"
    out2 = _render(client, pid)
    assert "state_eligibility" in set(out2["asked"])

    # 5 · and answering it records, so the seal's two preconditions are met.
    pressed2 = _press(client, pid, "state_eligibility", "everyone")
    assert len(pressed2["posts"]) == 1
    body = pressed2["posts"][0]["body"]
    assert body["kind"] == "set_eligibility"
    assert body["payload"] == {"answer": "everyone"}
    replay2 = client.post(f"/project/{pid}/decision", json=body)
    assert replay2.status_code == 200, replay2.text[:300]
    assert replay2.json()["eligibility"]["answer"] == "everyone"

    # 6 · THE THING THAT WAS UNREACHABLE. Before this loop the seal refused with
    # *"The grain question comes before the seal"* forever, because nothing on
    # the page could answer it.
    sealed = client.post(f"/project/{pid}/decision",
                         json={"kind": "seal", "payload": {}})
    assert sealed.status_code == 200, sealed.text[:300]
    assert sealed.json()["barrier_raised"] is True

    with capsys.disabled():
        print(f"\n  grain → eligibility → seal, driven: "
              f"{len(out['blob']):,} chars, {len(values)} grain options")


def test_the_eligibility_option_the_app_cannot_perform_says_so(capsys):
    """`GUIDED-006` / `DRIVE-011`'s mechanism, on a second question.

    *"Yes → which column, and what range?"* needs a column-and-range control
    that is not built, and `build_criterion` additionally refuses a criterion
    with no reason. A solid-bordered button that silently no-ops asserts a
    capability that does not exist; one that posts and takes a 400 is worse.
    """
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "age"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_grain",
                      "payload": {"answer": "one_row_per_person"}})

    out = _render(client, pid)
    card = re.search(
        r'<div class="stepcard" data-asked="state_eligibility".*?(?=<div class="stepcard"|$)',
        out["blob"], re.S)
    assert card, "the eligibility card is not in the render"
    from turbotab import pageharness

    buttons = [b for b in pageharness.elements(card.group(0))
               if b.get("data-answer-key") == "state_eligibility"]
    by_value = {b["data-answer-value"]: b for b in buttons}
    # `DRIVE-023`. The Router served this question with no `option_values`, and
    # `Question.to_dict` falls back to `option_values or options` — so the wire
    # said the value of the first option was the sentence a person reads, and
    # `set_eligibility` accepts only the two keys. A card built on that posts
    # its label instead of the answer it records, which is a 400 on a control
    # that looks live. `GUIDED-037` is the same defect on question 01.
    from turbotab import eligibility as _elig

    assert set(by_value) == set(_elig.ANSWERS), (
        f"the card posts its label instead of the answer it records: "
        f"{sorted(by_value)} — see DRIVE-023")

    assert "notbuilt" not in by_value["everyone"].get("class", "")
    assert by_value["everyone"].get("aria-disabled") is None
    assert "notbuilt" in by_value["restricted"].get("class", ""), (
        "the option the app cannot perform is offered as though it can be")
    assert by_value["restricted"].get("aria-disabled") == "true"
    # AND THE REASON IS ON SCREEN, not only in the class name.
    assert "not built in this build" in card.group(0)
    with capsys.disabled():
        print(f"\n  eligibility: everyone live, restricted unbuilt with its reason")


def test_the_grain_card_carries_its_teaching_panel(capsys):
    """`DESIGN_LANGUAGE.md` §10 layer 3, on the hardest question in the
    sequence.

    A generic card that dropped layer 3 here would be a regression: the server
    has served `GET /teaching/state_grain` for six loops and nothing pressed it.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "age"}})

    # The teaching route is fetched only on the press, so the fixpoint above
    # never sees it. Served explicitly, from the real API.
    served = client.get(f"/project/{pid}/teaching/state_grain")
    assert served.status_code == 200, served.text[:300]
    panel = served.json()

    out = _render(client, pid, '''
var t = document.querySelectorAll('[data-teach="state_grain"]')[0];
if (!t) { __emit({error: "no teach control on the grain card"}); }
__harness.dispatch("click", t);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
''', extra={f"/project/{pid}/teaching/state_grain": panel})

    fetched = [c["path"] for c in out["calls"]
               if c["path"].endswith("/teaching/state_grain")]
    assert fetched, (
        "pressing the teaching control fetched no teaching route, so layer 3 "
        "is a button that does nothing")
    # THE PANEL REACHED THE DOM, and it is the SERVER'S panel — asserted on a
    # string only the server composes, so a card rendering a hopeful
    # placeholder, or the harness's own error path, would not satisfy it. It is
    # read out of `[data-teach-body]` rather than the id sweep, because the
    # panel has no id and the reader collects ids.
    body = out["teach_body"] or ""
    assert panel["cannot_answer"] in body, (
        f"the teaching panel was fetched and its refusal is not in the card's "
        f"own panel: {body[:300]!r}")
    assert panel["title"] in body
    with capsys.disabled():
        print(f"\n  teaching: {len(panel.get('consequences') or [])} consequences, "
              f"worked example {'yes' if panel.get('worked_example') else 'no'}")


def test_the_contradiction_renders_with_the_way_out_it_travels_with(capsys):
    """`api.py`: the exits travel WITH the refusal *"so an interface cannot
    render the interruption without also rendering its way out."*

    Driven through a REFUSING route, which is the only way to reach this path at
    all — and the refusal body is the real one, provoked against the API rather
    than hand-written, so a change to the exits changes this test.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "age"}})
    refused = client.post(f"/project/{pid}/decision",
                          json={"kind": "set_grain",
                                "payload": {"answer": "one_row_per_person"}})
    assert refused.status_code == 409, refused.status_code
    detail = refused.json()["detail"]
    assert [e["id"] for e in detail["exits"]] == ["revise", "attest"]

    route = {f"POST /project/{pid}/decision":
             {"__status": 409, "body": {"detail": detail}}}
    out = _press(client, pid, "state_grain", "one_row_per_person", extra=route)

    band = out["refusal"] or ""
    assert "is-hidden" not in (out["refusal_class"] or ""), "the band stayed hidden"
    assert detail["message"] in band, "the refusal's own sentence is not rendered"
    for exit_ in detail["exits"]:
        assert exit_["label"] in band, f"exit {exit_['id']} has no control"
        assert exit_["detail"] in band, f"exit {exit_['id']} renders no detail"

    # THE NEGATIVE CONTROL. `showRefusal` returns early on an empty exits list,
    # so a band that filled anyway would mean this test reads something other
    # than the exits — which is the whole claim.
    stripped = json.loads(json.dumps(detail))
    stripped["exits"] = []
    out_n = _press(client, pid, "state_grain", "one_row_per_person",
                   extra={f"POST /project/{pid}/decision":
                          {"__status": 409, "body": {"detail": stripped}}})
    assert not (out_n["refusal"] or ""), (
        "the band rendered with no exits to render, so its content is not the "
        "exits")

    with capsys.disabled():
        print(f"\n  409: {len(band)} chars, "
              f"{len(detail['exits'])} exits rendered, 0 without them")


def test_taking_the_attested_exit_records_the_disagreement(capsys):
    """The exit is LIVE, not decoration.

    `GUIDED-072` gave every attest exit a ready-to-post `retry.payload` so the
    client need not hold a map from which endpoint refused to which key unlocks
    it. This asserts the page holds none: the body it sends is the original
    request with the server's payload merged in, and it is replayed against the
    real API.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "age"}})
    detail = client.post(f"/project/{pid}/decision",
                         json={"kind": "set_grain",
                               "payload": {"answer": "one_row_per_person"}}
                         ).json()["detail"]
    route = {f"POST /project/{pid}/decision":
             {"__status": 409, "body": {"detail": detail}}}

    out = _render(client, pid, (_PRESS % ("state_grain", "one_row_per_person",
                                          "state_grain", "one_row_per_person")) + '''
var xs = document.querySelectorAll("[data-refusal-i]");
__harness.dispatch("click", xs[xs.length - 1]);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
''', extra=route)
    bodies = [p["body"] for p in out["posts"]]
    assert len(bodies) == 2, f"the retry did not happen: {bodies}"
    assert bodies[1]["payload"]["acknowledge_contradiction"] is True
    assert bodies[1]["payload"]["answer"] == bodies[0]["payload"]["answer"]

    replay = client.post(f"/project/{pid}/decision", json=bodies[1])
    assert replay.status_code == 200, replay.text[:300]
    assert replay.json()["grain"]["contradiction_acknowledged"] is True
    with capsys.disabled():
        print("\n  attest: the retry the SERVER described, replayed and recorded")


def test_the_explore_exclusion_route_goes_through_once_the_grain_is_answered(capsys):
    """`DRIVE-019`, and it is CHECKED after Part A rather than closed by it.

    The Explore impossibility card has always offered *"Exclude those rows from
    the study"* with a typed reason box and an operable button, and it printed
    its own two preconditions: *the grain question answered* and *the held-out
    set not yet sealed*. The first was unanswerable — that is `DRIVE-017` — so
    the route was a live-looking control that could only ever be refused.

    **Both presses are through the page**, which is what couples this claim to
    the fix. A test that answered the grain over the API would still pass with
    the grain card deleted, and the row is precisely about the grain card.
    """
    client = _client()
    pid = _project(client, "clinical_labs.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "readmitted"}})

    # The card's own account of what blocks it, read from the server.
    block = client.get(f"/project/{pid}/evidence/plausibility"
                       ).json()["impossible"][0]
    route = next(r for r in block["routes"] if r["id"] == "exclude_the_rows")
    assert "the grain question answered" in " ".join(route["needs"])

    # Before the grain, the route is refused — the state the row describes.
    payload = dict(route["decision"]["payload"])
    payload["reason"] = "Entry errors outside the physiologic range."
    blocked = client.post(f"/project/{pid}/decision",
                          json={"kind": route["decision"]["kind"],
                                "subject": route["decision"]["subject"],
                                "payload": payload})
    assert blocked.status_code == 400
    assert "grain question comes before eligibility" in blocked.text

    # 1 · answer the grain THROUGH THE PAGE. `one_row_per_person` is a 409 on
    # this fixture by design, so the drive presses the answer that is true of it.
    pressed = _press(client, pid, "state_grain", "people_repeat")
    assert len(pressed["posts"]) == 1
    answered = client.post(f"/project/{pid}/decision",
                           json=pressed["posts"][0]["body"])
    assert answered.status_code == 200, answered.text[:300]

    # 2 · press the Explore route THROUGH THE PAGE, with a typed reason.
    out = _render(client, pid, '''
var b = document.querySelectorAll('[data-plaus-route="exclude_the_rows"]')[0];
if (!b) { __emit({error: "the impossibility card has no exclusion route"}); }
var box = document.querySelectorAll("[data-plaus-reason]")[0];
if (box) box.value = "Entry errors outside the physiologic range.";
__harness.dispatch("click", b);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
''')
    posts = [p for p in out["posts"] if p["body"]["kind"] == "set_eligibility"]
    assert posts, f"the exclusion button posted nothing: {out['posts']}"
    body = posts[-1]["body"]
    assert body["payload"]["reason"], "the typed reason did not travel"

    recorded = client.post(f"/project/{pid}/decision", json=body)
    assert recorded.status_code == 200, recorded.text[:300]
    elig = recorded.json()["eligibility"]
    assert elig["answer"] == "restricted"
    assert elig["column"] == block["column"]
    assert elig["n_excluded"] > 0, (
        "the criterion recorded and removed nothing, so participant flow has "
        "no line to report")
    with capsys.disabled():
        print(f"\n  DRIVE-019: {elig['n_excluded']} of {elig['n_before']} "
              f"excluded on `{elig['column']}`, both presses through the page")


def test_a_refusal_to_a_generic_answer_reaches_the_control(capsys):
    """`DRIVE-025`. A plain 400 has no `exits`, so the band does not draw — and
    `#upErr` is inside a `display:none` subtree from the first render.

    The at-control copy is the only one a person can see, and `AT_CONTROL` is
    assigned in exactly one delegate. `[data-answer-key]` was not in its
    selector, so the whole generic channel produced presses with no visible
    response — `GUIDED-167`'s own failure, on the channel that now carries
    questions 03 and 08.
    """
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "age"}})

    sentence = "The grain question comes before the seal."
    # `_press` dispatches on a `data-answer-key` control — the attribute this
    # docstring names — so the press under test really is the generic channel's
    # and not some other posting surface that happens to be on the page.
    assert 'data-answer-key="%s"' in _PRESS, (
        "the press helper no longer targets a data-answer-key control, so this "
        "claim is about a different surface from the one it names")
    out = _press(client, pid, "state_grain", "one_row_per_person",
                 extra={f"POST /project/{pid}/decision":
                        {"__status": 400, "body": {"detail": sentence}}})
    # The band is correctly empty — this refusal carries no way out.
    assert not (out["refusal"] or "")
    # AND THE SLOT IS THE PRESSED CONTROL'S OWN, derived from the key rather
    # than hard-coded, so a control whose `data-ac` stopped matching its
    # `data-answer-key` would fail here instead of writing into a stranger's
    # slot.
    assert out["at_grain"] is not None, "the at-control slot ac-ans-state_grain does not exist"
    # THE PRECONDITION, ESTABLISHED FROM THE RENDER rather than asserted from
    # the comment that describes it. If `#sub-upload` were visible this test
    # would not be measuring the case the at-control mechanism exists for, and
    # it would pass for the wrong reason.
    assert "is-hidden" in (out["upload_class"] or ""), (
        f"`#sub-upload` reads {out['upload_class']!r}; the canonical sink is "
        f"not hidden, so this claim is about a state the app is not in")
    assert sentence in (out["upErr"] or ""), (
        "the sentence did not even reach the hidden sink, so the request path "
        "is broken and the at-control claim below would be red for the wrong "
        "reason")
    assert sentence in (out["at_grain"] or ""), (
        f"the refusal reached the hidden sink and nothing a person can see: "
        f"{out['at_grain']!r}")
    with capsys.disabled():
        print(f"\n  plain 400: 0 chars in the band, "
              f"{len(out['at_grain'] or '')} at the control")
