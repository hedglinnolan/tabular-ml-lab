"""`GUIDED-176` — the cut-point refusal was computed, correct, and unreadable.

`bin_fixed` is *"binning by supplied cut-points"*, and the page's
*"Show me what it does"* sends `?transform=bin_fixed&columns=<col>` with **no
`params`**. The server refuses, correctly and well:

    Binning by supplied cut-points needs at least two edges. Without them the
    edges would have to come from the data, which is a different transform and
    defers.

`FeatureRefusal` becomes `HTTPException(400, str(exc))`, so `detail` is a
**string**. `api()` attaches `err.detail` only when the detail is an object, so
`showRefusal` never fires — and it could not have anyway: `showRefusal` is
called from exactly two places in the page, `post()` and the train-start
handler, and this is neither. What was left was `.catch(setErr(err.message))`,
and `setErr` writes two places: `#upErr`, which `renderData()` hides from the
first render after upload, and `ac-<data-ac>` at the control that was pressed.

**The preview button carried no `data-ac`.** So `AT_CONTROL` became `""`,
`atControl("")` returned false, and the sentence went only to a hidden node.
Driven below on two fixtures before the fix: `#featprev-bin_fixed` held 0
characters, `#refusal` held 0 characters and kept `is-hidden`, the row's own
slot held 0 characters, and the only copy of the server's sentence was inside
`#sub-upload`.

## Which of (a) (b) (c), and why

**Not (c).** The branch is reachable. `bin_fixed` is served in `row_local` on
`clinic_visits.csv` and on `dietary_recalls.csv`, and the button renders as soon
as a column is picked. `GUIDED-182`'s *"no shipped fixture produces a
previewable offer at all"* is about `data-offer-preview`, the row inside a
refusal card — a different control. It does not cover this one, and this test
asserts the catalogue rather than assuming it.

**Not (a) alone.** Making the server send a structured detail here would render
nothing: this handler never calls `showRefusal`, and `showRefusal` returns early
when the detail carries no `exits`. That is a capability with no consumer, which
is `AGENT_ONBOARD` §07.1 in one move.

**(b), and without touching `api()`.** `err.message` already carries the
server's sentence and `setErr` already routes it to `ac-<data-ac>`. The only
thing missing was the slot name on the control.

**The general rule, stated because the instance is not the finding:** `data-ac`
is not a property of *posting* controls. It is a property of every control that
can produce a response — a GET preview refuses exactly as a POST does. L48-A
swept the posting controls and terminated there, which is §08.5's *did a sweep
terminate where the sweeper's attention ended* in its own words.

## The 400s, re-counted

The prompt's figure was **55 sites, 44 of them `str(exc)`**, and that is a grep's
answer. Parsed instead of matched, `turbotab/` raises `HTTPException` with status
400 at **70 sites**, all of them in `turbotab/api.py` and none anywhere else in
the package. **70 of 70 carry a string detail** — 45 literally `str(exc)`, 13 a
bare literal, 11 an f-string, and one a prose variable (`repeat_chain_gap()`).
**Not one 400 in the app carries a structured detail.** A grep for
`HTTPException(400` misses fourteen of them because the call wraps onto the next
line, which is trap #5 exactly. Pinned by
`test_no_four_hundred_in_this_app_carries_a_structured_detail` below, so the
count is a record rather than a claim in a report.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"
API = Path(__file__).resolve().parent / "api.py"

#: NOT COVERED, said out loud (`GUIDED-097`, §10 rule 4).
SHAPES_NOT_COVERED = (
    "Whether the slot is on screen. `pageharness.py` has no layout and says so; "
    "this asserts the sentence reaches an addressable node beside the button, "
    "never that a person can see it.",
    "`ordinal_declared`, the sibling branch — *encoding in a stated order* "
    "refuses for the same reason through the same `str(exc)` path. It is the "
    "same code on both sides and is not separately driven.",
    "The SUCCESS path of this preview. Every transform whose params the button "
    "can supply returns 200 and writes into `#featprev-<key>`; an invisible "
    "refusal is the defect, so the two that refuse are what is driven.",
    "A project with no target. `renderFeatures` needs `/features`, which needs "
    "the target, so the row does not render and there is nothing to press.",
)

#: Two fixtures of different target shape (`GUIDED-097`): `outcome` is a
#: two-level string and `bmi` is continuous, so one run is a classification
#: project and one a regression project. The COLUMN fed to `bin_fixed` is
#: numeric in both, because that is what the picker offers.
FIXTURES = [
    ("clinic_visits.csv", "outcome", "age", "classification"),
    ("dietary_recalls.csv", "bmi", "energy_kcal", "regression"),
]

TRANSFORM = "bin_fixed"


def _project(fixture: str, target: str):
    """Driven as far as the Features step, because that is where the row is."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_target", {"column": target}),
                          ("set_purpose", {"answer": "prediction"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"})):
        client.post(f"/project/{pid}/decision",
                    json={"kind": kind, "payload": payload})
    return client, pid


def _routes(client, pid):
    """Every response one render of this page asks for."""
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    return out


#: THE PRESS IS BUILT FROM THE RENDER, NEVER HAND-SPECIFIED — trap #3.
#:
#: `data-ac` is the attribute whose absence IS this defect, so a synthetic
#: `{'data-feat-preview': 'bin_fixed', 'data-ac': '...'}` would let the fixture
#: supply it and the revert probe would report `GREEN — NOT LOAD-BEARING`. Every
#: attribute pressed here is read off the button the page emitted.
_PRESS_FROM_RENDER = """
function buttons(html){
  var re = /<button\\b([^>]*)>/g, m, out = [];
  while ((m = re.exec(html))){
    var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(m[1]))) attrs[k[1]] = k[2];
    out.push(attrs);
  }
  return out;
}
"""


@pytest.mark.parametrize("fixture,target,column,shape", FIXTURES,
                         ids=["classification target", "regression target"])
def test_the_cut_point_refusal_lands_at_the_control_that_asked_for_it(
        fixture, target, column, shape):
    """The reproduction and its fix, driven end to end.

    The mechanism is set through the page's own `data-feat-col` picker rather
    than injected, because the preview button does not render until a column is
    chosen — injecting it would drive a control no user can reach.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture, target)
    routes = _routes(client, pid)

    # THE BRANCH IS REACHABLE, asserted rather than assumed (`GUIDED-182`).
    served = [r["key"] for r in
              (routes[f"/project/{pid}/features"].get("row_local") or [])]
    assert TRANSFORM in served, (
        f"`{TRANSFORM}` is no longer in the catalogue this fixture serves, so "
        f"the user-supplied cut-point branch is unreachable and this test is "
        f"driving nothing: {served}")
    assert column in (routes[f"/project/{pid}/features"].get("numeric_columns")
                      or []), (
        f"`{column}` is not offered by the picker on {fixture}, so the press "
        f"below is on a control the page does not render")

    path = (f"/project/{pid}/feature/preview?transform={TRANSFORM}"
            f"&columns={column}")
    live = client.get(path)
    assert live.status_code == 400, (
        f"the server no longer refuses a cut-point preview with no cut-points "
        f"({live.status_code}); this test is pinned to that refusal")
    assert isinstance(live.json()["detail"], str), (
        "the refusal now sends a structured detail, so this test is no longer "
        "driving the string path it was written for")
    routes[path] = {"__status": 400, "body": live.json()}

    out = PH.run(
        _PRESS_FROM_RENDER +
        "var sel = __harness.target({'data-feat-col': %(t)s, 'data-feat-slot': '0'});\n"
        "sel.value = %(col)s;\n"
        "__harness.dispatch('change', sel);\n"
        "for (var i = 0; i < 4; i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var body = __harness.html('featbody-' + %(t)s) || '';\n"
        "var btns = buttons(body).filter(function(b){ return b['data-feat-preview']; });\n"
        "if (btns.length) __harness.dispatch('click', __harness.target(btns[0]));\n"
        "for (var j = 0; j < 10; j++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({rendered: btns[0] || null, row: body.slice(0, 400),\n"
        "        at: btns.length && btns[0]['data-ac']\n"
        "              ? __harness.html('ac-' + btns[0]['data-ac']) : null,\n"
        "        prev: __harness.html('featprev-' + %(t)s),\n"
        "        band: __harness.html('refusal'),\n"
        "        canonical: __harness.el('upErr')\n"
        "                     ? __harness.el('upErr').textContent : null,\n"
        "        fetched: __harness.calls()\n"
        "                   .filter(function(c){ return c.path.indexOf('feature/preview') !== -1; })\n"
        "                   .map(function(c){ return c.path; })});"
        % {"t": repr(TRANSFORM), "col": repr(column)},
        routes=routes, search=f"?project={pid}")

    assert out["rendered"], (
        f"the `{TRANSFORM}` row did not render its preview control after a "
        f"column was picked, so nothing was pressed: {out['row']!r}")
    assert out["fetched"], (
        "the preview control was pressed and asked the server nothing")

    # 1 · THE CONTROL NAMES A SLOT. This is the assertion the fix is about: a
    #     GET preview refuses exactly as a POST does, and a control with no
    #     `data-ac` makes `AT_CONTROL` empty and `atControl("")` return false.
    assert out["rendered"].get("data-ac"), (
        f"the preview control names no at-control slot, so every response to a "
        f"press on it — including a refusal the server composed correctly — "
        f"goes only to `#upErr`, which `renderData()` hid on the first render "
        f"after upload. The sentence was there and unreadable: "
        f"{(out['canonical'] or '')[:120]!r}. Button: {out['rendered']!r}")

    # 2 · and the server's own sentence arrives there.
    assert out["at"], (
        f"the cut-point 400 rendered NOTHING at the control. `#featprev-"
        f"{TRANSFORM}` held {len(out['prev'] or '')} characters and `#refusal` "
        f"held {len(out['band'] or '')}")
    assert "cut-point" in out["at"], (
        f"what arrived at the control is not the server's explanation: "
        f"{out['at'][:300]!r}")
    assert out["at"] in live.text or live.json()["detail"] in out["at"] \
        or all(w in out["at"] for w in ("edges", "defers")), (
        f"the sentence at the control is not the one the server sent, so the "
        f"page is composing its own: {out['at'][:300]!r}")

    # 3 · and the refusal BAND stays empty, because a string detail never
    #     reaches `showRefusal` — the same slot, a different road into it.
    assert not (out["band"] or ""), (
        f"the refusal band drew something. `detail` is a string here, so "
        f"`api()` attaches no `err.detail` and `showRefusal` is not even "
        f"called from this handler: {(out['band'] or '')[:200]!r}")


def test_no_four_hundred_in_this_app_carries_a_structured_detail():
    """The count this row's decision rests on, parsed rather than grepped.

    The decision not to structure this branch's detail is only defensible if
    structuring it would be a lone exception with no consumer, so the claim is
    recorded here instead of in a report: **every** 400 in `turbotab/` sends a
    string. A grep for `HTTPException(400` answers 56 because fourteen of the
    calls wrap onto the next line — trap #5, in the file this test reads.

    The numbers are asserted loosely enough that adding a 400 does not break
    the build, and tightly enough that the first STRUCTURED one does.
    """
    tree = ast.parse(API.read_text())
    string_detail, structured, sites = 0, [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else getattr(func, "attr", None))
        if name != "HTTPException":
            continue
        status = node.args[0] if node.args else None
        detail = node.args[1] if len(node.args) > 1 else None
        for kw in node.keywords:
            if kw.arg == "status_code":
                status = kw.value
            if kw.arg == "detail":
                detail = kw.value
        if not (isinstance(status, ast.Constant) and status.value == 400):
            continue
        sites += 1
        if isinstance(detail, ast.Dict):
            structured.append(node.lineno)
        else:
            string_detail += 1

    assert sites >= 55, (
        f"only {sites} sites raise a 400 in api.py; the refusal apparatus this "
        f"row is about has shrunk and the decision should be re-taken")
    assert not structured, (
        f"a 400 now carries a structured detail (line(s) {structured}). That "
        f"is not wrong — but `showRefusal` renders nothing without `exits` and "
        f"is called from two handlers only, so a structured 400 needs a "
        f"consumer shipped with it. `GUIDED-176`'s decision assumed there were "
        f"none")
    assert string_detail == sites
