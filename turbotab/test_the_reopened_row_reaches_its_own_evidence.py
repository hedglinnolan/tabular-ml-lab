"""L48-A1 — `GUIDED-167` closed on the press the row was actually filed about.

The row was marked `FIXED` at L47 on a test that pressed a `data-dismiss`. The
mechanism was right and the wiring reached three controls of thirty, and
**`data-miss-choose` was not one of them** — so on the press in the row's own
evidence `t.getAttribute("data-ac")` was null, `AT_CONTROL` became `""`,
`atControl("")` returned false, and the typed 409 landed only in `#upErr`, which
is inside `#sub-upload` and hidden by `renderData()` from the first render on.
The adjudicator reopened it to `PARTIAL`. This is `MISC-019`'s shape: a row
describing a class, a test covering a different instance, and the closed count
moving anyway.

## What is driven here

The two responses the drive produced, both of them correct on the wire and
neither reaching a person:

- **`impute_median` on an informative mechanism → a typed 409.** `detail` is an
  OBJECT, so `showRefusal` fires and the reason must appear at the control.
- **`drop_rows` → a 400 with the complete-case explanation.** `detail` is a
  STRING, so `showRefusal` never fires and this goes through `setErr` — a
  different code path to the same slot, which is why both are driven rather
  than one.

Two fixtures of different target shape (`GUIDED-097`): `clinic_visits.csv`
routes a **categorical** column through `impute_mode`, `metabolomics_untargeted.
csv` a **numeric** one through `impute_median`. The shape not covered is named
in `SHAPES_NOT_COVERED` below.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "Whether the slot is on screen. Nothing without layout can tell, and "
    "`pageharness.py` says so in its own docstring.",
    "A project with NO target. `renderExplore` returns early without one, so "
    "the card does not render at all and there is nothing to press — the "
    "invisible-response question does not arise before a target is chosen.",
    "The `explicit_category` and `indicator` options, which SUCCEED. This "
    "drives the two that refuse, because an invisible refusal is the defect.",
)


#: fixture -> the target that reaches Explore, and its shape. `GUIDED-097`:
#: `outcome` is a two-level string and `bmi` is continuous, so the two runs are
#: a classification project and a regression one.
TARGETS = {"clinic_visits.csv": "outcome",
           "metabolomics_untargeted.csv": "bmi"}


def _project(fixture: str):
    """A project driven as far as Explore, because that is where the card is.

    `renderMissingness` is called from `renderExplore` alone, and `renderExplore`
    returns early with no target — a drive that skipped the target question got
    an empty `#missBox` and would have reported "the slot is missing" for a card
    that never rendered. That is the false negative Part D's sweep names.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": TARGETS[fixture]}})
    return client, pid


#: THE PRESS IS BUILT FROM THE RENDER, NEVER HAND-SPECIFIED — trap #3.
#:
#: The first version of this drive passed `{'data-miss-choose': col, 'data-ac':
#: 'miss-' + col}` to `__harness.target`, and the revert probe reported
#: `GREEN — NOT LOAD-BEARING`: deleting `data-ac` from the button changed
#: nothing, because the fixture was supplying the attribute whose absence IS the
#: defect. The synthetic target now reads every `data-*` off the rendered button,
#: so a page that stops emitting `data-ac` produces a press that does not carry
#: one, which is exactly what a user's press would be.
_PRESS_FROM_RENDER = """
function pressed(html, attr, value, opt){
  var re = /<button\\b([^>]*)>/g, m;
  while ((m = re.exec(html))){
    var raw = m[1], attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(raw))) attrs[k[1]] = k[2];
    if (attrs[attr] !== value) continue;
    if (opt && attrs['data-miss-opt'] !== opt) continue;
    return attrs;
  }
  return null;
}
"""


def _routes(client, pid):
    got = client.get(f"/project/{pid}").json()
    out = {f"/project/{pid}": got}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    return out


@pytest.mark.parametrize("fixture,column,option,expect", [
    ("clinic_visits.csv", "notes", "impute_mode", "most common value"),
    ("metabolomics_untargeted.csv", "mz_0022", "impute_median", "median"),
], ids=["categorical column", "numeric column"])
def test_the_typed_409_lands_at_the_control_that_caused_it(
        fixture, column, option, expect):
    """The reopening's own test: `route_missingness`, informative, a fill.

    The mechanism is set through the page's own `data-miss-mech-for` control
    rather than injected into the payload, because the strategies do not render
    at all until it is answered (`GUIDED-091`) — injecting it would drive a card
    no user can reach.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture)
    routes = _routes(client, pid)
    cards = routes[f"/project/{pid}/evidence/missingness"].get("cards") or []
    assert any(c["column"] == column for c in cards), (
        f"{fixture} no longer serves a missingness card for `{column}`, so "
        f"this drive is pressing a control the fixture does not render")

    live = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "card_option": option,
                    "mechanism": "informative"}})
    assert live.status_code == 409, (
        f"the server no longer refuses {option} on an informative mechanism "
        f"({live.status_code}); this test is pinned to that refusal")
    routes[f"POST /project/{pid}/decision"] = {
        "__status": 409, "body": live.json()}

    out = PH.run(
        _PRESS_FROM_RENDER +
        "__harness.dispatch('click', __harness.target("
        "  {'data-miss-mech-for': %(col)s, 'data-miss-mech-value': 'informative'}));\n"
        "for (var i = 0; i < 4; i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var box = __harness.html('missBox') || '';\n"
        "var btn = pressed(box, 'data-miss-choose', %(col)s, %(opt)s);\n"
        "if (btn) __harness.dispatch('click', __harness.target(btn));\n"
        "for (var j = 0; j < 10; j++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({rendered: btn, box: box.slice(0, 400),"
        "        at: __harness.html('ac-miss-' + %(col)s),"
        "        band: __harness.html('refusal'),"
        "        canonical: __harness.el('upErr') ?"
        "                   __harness.el('upErr').textContent : null});"
        % {"col": repr(column), "opt": repr(option)},
        routes=routes, search=f"?project={pid}")

    assert out["rendered"], (
        f"the card for `{column}` did not render its choose control after the "
        f"mechanism was answered, so nothing was pressed: {out['box']!r}")
    assert out["rendered"].get("data-ac") == f"miss-{column}", (
        f"the rendered button names no at-control slot, so the press cannot be "
        f"answered where it was made: {out['rendered']!r}")
    assert out["at"], (
        f"the 409 rendered NOTHING at the control. This is the reopening's "
        f"exact reproduction. The band held: {(out['band'] or '')[:120]!r}; "
        f"`#upErr` held: {out['canonical']!r}")
    assert expect in out["at"], (
        f"something arrived at the control and it is not the server's reason: "
        f"{out['at'][:300]!r}")
    assert column in out["at"], (
        "the sentence at the control does not name the column it is about")


def test_the_complete_case_400_lands_there_too():
    """The string-`detail` path, which `showRefusal` never sees.

    `drop_rows` returns a 400 whose `detail` is a plain string, so `api()` does
    not attach `err.detail` and the refusal band stays empty — the response
    reaches the user only through `setErr`. Same slot, different route into it,
    and a fix that wired only the `showRefusal` half would pass the test above
    and fail this one.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project("clinic_visits.csv")
    routes = _routes(client, pid)
    live = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": "notes",
        "payload": {"column": "notes", "card_option": "drop_rows",
                    "mechanism": "informative"}})
    assert live.status_code == 400
    assert isinstance(live.json()["detail"], str), (
        "the complete-case refusal now sends a structured detail, so this test "
        "is no longer driving the string path it was written for")
    routes[f"POST /project/{pid}/decision"] = {
        "__status": 400, "body": live.json()}

    out = PH.run(
        _PRESS_FROM_RENDER +
        "__harness.dispatch('click', __harness.target("
        "  {'data-miss-mech-for': 'notes', 'data-miss-mech-value': 'informative'}));\n"
        "for (var i = 0; i < 4; i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var btn = pressed(__harness.html('missBox') || '', 'data-miss-choose',"
        "                  'notes', 'drop_rows');\n"
        "if (btn) __harness.dispatch('click', __harness.target(btn));\n"
        "for (var j = 0; j < 10; j++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({pressed: btn, at: __harness.html('ac-miss-notes'),"
        "        band: __harness.html('refusal')});",
        routes=routes, search=f"?project={pid}")

    assert out["pressed"], "the drop-rows option did not render on this card"

    assert out["at"], (
        "the complete-case 400 rendered nothing at the control. It is a string "
        "`detail`, so `showRefusal` never fires and `setErr` is the only path")
    assert "complete-case" in out["at"] or "eligibility criterion" in out["at"], (
        f"what arrived is not the server's explanation: {out['at'][:300]!r}")


def test_the_card_emits_a_slot_for_every_column_it_renders():
    """The static half — the slot is per-CARD and every card has one.

    Driven above on two columns; asserted here across every card the fixture
    produces, because a per-column id that is emitted for only the first card is
    the shape `GUIDED-167` had at the page level one loop ago.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project("clinic_visits.csv")
    routes = _routes(client, pid)
    cards = routes[f"/project/{pid}/evidence/missingness"]["cards"]
    assert len(cards) >= 2, "this fixture used to render at least two cards"

    out = PH.run("__emit({box: __harness.html('missBox')});",
                 routes=routes, search=f"?project={pid}")
    box = out["box"] or ""
    for card in cards:
        assert f'id="ac-miss-{card["column"]}"' in box, (
            f'the card for `{card["column"]}` renders no at-control slot, so '
            f"every response to a press on it is invisible")
