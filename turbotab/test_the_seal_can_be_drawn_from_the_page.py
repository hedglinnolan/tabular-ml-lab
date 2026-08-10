"""`DRIVE-017`'s second half — the seal had no control anywhere on the page.

Grain and eligibility rendered nowhere because a list claimed cards that were
never built. The seal is a different absence and it needed a different fix:
**it is not a Router question at all.** No key, no plan entry, no `asked`
status — so the interview's generic channel could never draw it and there was
nothing to un-list. `renderDisclosures` rendered the seal's *sentence*, and only
once the seal already existed. `EFFECTS.seal` held the promise and nothing could
make it.

So a human could answer every question in the pre-seal sequence and still not
reach a model, because the model shelf is gated on `barrier_raised`.

## What this asserts

1. **The gate is the SERVER'S, not the page's.** `GET /seal` serves what the
   `seal` handler itself would raise, from one function both call. A page that
   decided for itself when the control was pressable would hold a second copy
   of clause §01's order — and would silently omit the fourth gate, the repeat
   chain, which fires only where people repeat.
2. **The control says what it costs before it is pressed**, not after. Sealed
   once is the whole design of the lockbox.
3. **Pressing it draws the split**, driven under `pageharness` and replayed
   against the real API, with the record read back.
4. **Once drawn, the control is gone and the disclosure is the record** — one
   claim, one object.

## Two fixture shapes

`clinical_risk.csv` reaches `can_draw` in three answers. `clinical_longitudinal.csv`
is the shape that exposes the fourth gate: answering the grain leaves the repeat
chain open, and the blocker sentence changes to that rather than to eligibility.
A page holding its own copy of the order would show the wrong sentence there, and
that is the case a single-fixture test would miss (`GUIDED-097`).
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
        seal: __harness.html("sealBox"),
        disclosures: __harness.html("disclosuresBox"),
        at_seal: __harness.html("ac-seal"),
        posts: __harness.posts(),
        calls: __harness.calls().map(function(c){
          return {method: c.method, path: c.path}; })});
"""


_ATTRS = re.compile(r'\s[a-zA-Z-]+="[^"]*"')


def _visible_text(html: str) -> str:
    """The copy, with every attribute value removed.

    A control's effect sentence appears three times in this page's markup — as
    rendered copy, in `data-tip` and in `aria-label` — so a substring search
    over the raw markup cannot tell a disclosed consequence from a hovered one.
    `FEATURE_PARITY.md`'s rule about substrings, one layer in.
    """
    return re.sub(r"<[^>]*>", " ", _ATTRS.sub(" ", html))


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, fixture: str) -> str:
    with (DATA / fixture).open("rb") as handle:
        return client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]


def _decide(client, pid, kind, payload, expect=200):
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": kind, "payload": payload})
    assert resp.status_code == expect, (kind, resp.status_code, resp.text[:300])
    return resp


def _render(client, pid: str, tail: str = "", extra=None) -> Dict[str, Any]:
    """Every route the page fetches, answered to a fixpoint, then the render."""
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
    out = pageharness.run(tail + "\n" + reader, routes=routes,
                          search=f"?project={pid}")
    if isinstance(out, dict) and out.get("error"):
        raise AssertionError(f"the page could not be driven: {out['error']}")
    assert len(out["blob"]) > 5_000, (
        f"the reader collected {len(out['blob'])} characters; believe nothing "
        f"this run reports")
    return out


@pytest.fixture(autouse=True)
def _needs_node():
    from turbotab import pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")


def test_the_page_fetches_the_seal_state_rather_than_deriving_it(capsys):
    """The load-bearing design claim, asserted behaviorally.

    Reading the page for `P.target && P.grain` would prove nothing — a grep
    cannot tell a page that reads a field from one that mentions it, which is
    `pageharness`' own reason for existing. So this asserts the FETCH happened
    and that the rendered sentence is the server's, character for character.
    """
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    _decide(client, pid, "set_target", {"column": "age"})

    served = client.get(f"/project/{pid}/seal").json()
    assert served["can_draw"] is False and served["blocked_by"]

    out = _render(client, pid)
    assert any(c["path"].endswith("/seal") for c in out["calls"]), (
        "the page never asked the server whether the seal can be drawn, so "
        "whatever it renders is its own second copy of clause §01")
    assert served["blocked_by"] in (out["seal"] or ""), (
        f"the control does not quote the server's reason:\n  {out['seal']!r}")
    with capsys.disabled():
        print(f"\n  blocked: {served['blocked_by'][:70]}…")


def test_the_control_says_it_cannot_be_redrawn_before_it_is_pressed(capsys):
    """Sealed once is the whole design, and a consequence disclosed afterwards
    is not a disclosure.

    **What this cannot tell, said rather than implied.** The sentence renders
    twice — as the card's always-visible `why` line and inside a `<details>`
    disclosure — and `pageharness` knows nothing about layout, so it cannot
    distinguish copy that is on screen from copy one click away. Removing
    either alone leaves this green; removing both turns it red. So the measured
    claim is *the consequence is in the card's rendered copy rather than only in
    a hover string*, which is strictly weaker than *a user sees it* and strictly
    stronger than anything a text search over the file could ask. The driver
    remains the check for visibility.
    """
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    _decide(client, pid, "set_target", {"column": "age"})
    _decide(client, pid, "set_grain", {"answer": "one_row_per_person"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})

    served = client.get(f"/project/{pid}/seal").json()
    assert served["can_draw"] is True

    out = _render(client, pid)
    card = out["seal"] or ""
    # READ THE TEXT, NOT THE ATTRIBUTES. Every control on this page carries its
    # effect in `data-tip` and `aria-label`, so `"cannot be redrawn" in card`
    # is satisfied by a hover string — and a consequence a user has to hover to
    # discover has not been disclosed. Measured: dropping the `<p class="why">`
    # left this assertion GREEN on the tooltip alone, which is why it reads the
    # rendered copy instead.
    text = _visible_text(card)
    assert "cannot be redrawn" in text, (
        f"the control says the seal is once-only only in a hover string:\n"
        f"  {text!r}")
    assert served["once"] in text, (
        "the server's own once-only sentence is not rendered as copy")

    from turbotab import pageharness

    button = [b for b in pageharness.elements(card) if "data-seal" in b]
    assert len(button) == 1, f"expected one seal control, found {len(button)}"
    assert button[0].get("aria-disabled") is None, (
        "the control is not pressable in a state the server says can draw")
    assert "cannot be redrawn" in button[0].get("data-tip", "")
    with capsys.disabled():
        print(f"\n  pressable, and it says so first: {len(card)} chars")


def test_pressing_it_draws_the_held_out_set(capsys):
    """The consumer. Pressed under the harness, the body replayed, the record
    read back — and the shelf that was gated on it is measured after."""
    client = _client()
    pid = _project(client, "clinical_risk.csv")
    _decide(client, pid, "set_target", {"column": "age"})
    _decide(client, pid, "set_grain", {"answer": "one_row_per_person"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})

    out = _render(client, pid, '''
var b = document.querySelectorAll("[data-seal]")[0];
if (!b) { __emit({error: "no seal control in the render"}); }
__harness.dispatch("click", b);
for (var i = 0; i < 6; i++) await new Promise(function(r){ setTimeout(r, 0); });
''')
    posts = [p for p in out["posts"] if p["body"]["kind"] == "seal"]
    assert len(posts) == 1, f"the seal control posted {len(posts)} time(s)"
    assert posts[0]["body"]["payload"] == {}, (
        "the page sent its own fraction or seed, so it is choosing the split")

    drawn = client.post(f"/project/{pid}/decision", json=posts[0]["body"])
    assert drawn.status_code == 200, drawn.text[:300]
    body = drawn.json()
    assert body["barrier_raised"] is True
    lockbox = body["lockbox"]
    assert lockbox["n_test"] > 0 and lockbox["seal_basis"] == "cross_sectional"
    assert body["disclosures"]["seal"]

    # AND THE THING IT UNGATES. `renderTrainStep` returns early on
    # `!P.barrier_raised`, so before this the Train step had no shelf at all —
    # which is what the drive reported as "a heading with no shelf".
    shelf = client.get(f"/project/{pid}/models").json()
    models = [m["key"] for g in shelf.get("groups", [])
              for m in (g.get("models") or [])]
    assert models, "the shelf served nothing after the seal"

    after = _render(client, pid)
    assert not (after["seal"] or "").strip(), (
        "the seal control is still on screen after the seal was drawn; the "
        "record and the act would then be two objects for one claim")
    assert "disc-mark" in (after["disclosures"] or ""), (
        "the disclosure band does not carry the sealed marker")
    with capsys.disabled():
        print(f"\n  drew {lockbox['n_test']} of {lockbox['n_total']} rows; "
              f"the shelf then serves {len(models)} models")


def test_the_blocker_is_the_repeat_chain_where_the_repeat_chain_is_open(capsys):
    """The fourth gate, and the reason the page must not hold its own copy.

    Clause §01's bracketed steps sit BETWEEN the grain and eligibility and fire
    only on a table where people repeat. A control keyed on
    `target && grain && eligibility` would read as pressable here and be
    refused, or would name the wrong question — and this fixture is the only one
    that shows the difference.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    _decide(client, pid, "set_target", {"column": "age"})
    _decide(client, pid, "set_grain",
            {"answer": "people_repeat", "group_col": "subject_id"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})

    served = client.get(f"/project/{pid}/seal").json()
    assert served["can_draw"] is False
    assert "repeated measurements" in served["blocked_by"], served["blocked_by"]

    out = _render(client, pid)
    card = out["seal"] or ""
    assert served["blocked_by"] in card, (
        f"the control names a different obstacle from the one the engine "
        f"would refuse with:\n  {card!r}")
    from turbotab import pageharness

    button = [b for b in pageharness.elements(card) if "data-seal" in b]
    assert button and button[0].get("aria-disabled") == "true", (
        "the control is pressable while the engine would refuse it")

    # Answering the chain clears it, and the sentence moves on.
    _decide(client, pid, "set_repeat_kind", {"kind": "time_points"})
    _decide(client, pid, "set_unit_of_analysis", {"unit": "record"})
    _decide(client, pid, "set_temporal_prediction", {"temporal": False})
    now = client.get(f"/project/{pid}/seal").json()
    assert now["can_draw"] is True, now["blocked_by"]
    with capsys.disabled():
        print(f"\n  fourth gate named: {served['blocked_by'][:66]}…")


def test_the_gate_the_control_reads_is_the_gate_the_handler_enforces(capsys):
    """One rule, two readers — swept over every prefix of the pre-seal sequence
    on both fixtures, and the counts are reported including the passes.

    A control that agreed with the handler on the fixture it was written against
    and disagreed one answer earlier is the failure this sweep exists for.
    """
    from turbotab import api

    client = _client()
    drives = {
        "clinical_risk.csv": [
            ("set_target", {"column": "age"}),
            ("set_grain", {"answer": "one_row_per_person"}),
            ("set_eligibility", {"answer": "everyone"}),
        ],
        "clinical_longitudinal.csv": [
            ("set_target", {"column": "age"}),
            ("set_grain", {"answer": "people_repeat", "group_col": "subject_id"}),
            ("set_repeat_kind", {"kind": "time_points"}),
            ("set_unit_of_analysis", {"unit": "record"}),
            ("set_temporal_prediction", {"temporal": False}),
            ("set_eligibility", {"answer": "everyone"}),
        ],
    }
    checked = agreed = 0
    for fixture, steps in drives.items():
        for cut in range(len(steps) + 1):
            pid = _project(client, fixture)
            for kind, payload in steps[:cut]:
                _decide(client, pid, kind, payload)
            served = client.get(f"/project/{pid}/seal").json()
            attempt = client.post(f"/project/{pid}/decision",
                                  json={"kind": "seal", "payload": {}})
            checked += 1
            if served["can_draw"]:
                assert attempt.status_code == 200, (
                    f"{fixture} after {cut}: the control says it can draw and "
                    f"the handler refused — {attempt.text[:200]}")
            else:
                assert attempt.status_code == 400, (
                    f"{fixture} after {cut}: the control says it cannot draw "
                    f"and the handler drew it anyway")
                assert served["blocked_by"] in attempt.text, (
                    f"{fixture} after {cut}: the control names a different "
                    f"reason from the refusal:\n  control: "
                    f"{served['blocked_by']}\n  handler: {attempt.text[:200]}")
            agreed += 1
    with capsys.disabled():
        print(f"\n  {agreed} of {checked} states agree, across "
              f"{len(drives)} fixtures — one rule, {api.seal_blocker.__name__}")
    assert checked == 11, f"the sweep covered {checked} states, expected 11"
