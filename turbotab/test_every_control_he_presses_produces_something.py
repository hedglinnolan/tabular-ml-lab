"""L48-D — the loop's gate, driven on the nearest thing to his file.

> **The loop's gate:** upload an `nhanes_*` fixture, walk it to Preprocess, and
> **every control pressed produces something a person can see** — or is listed,
> by name, with the reason it does not.

`nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` (21,849 × 29) is not in the
repository, so this drives the three that are — `nhanes_dietary.csv`,
`nhanes_kilojoules.csv` and `nhanes_partial_design.csv`, all 120 rows, the
first two 10 columns and the third 8. **The gap is stated rather than papered
over**: 120 rows will not reproduce a finding whose evidence is 15,552 blanks,
and `SEQN` is `int64` here where his is `float64` (§00, and the reconnaissance
that said otherwise was corrected at L47).

## What "produces something" means here, and what it cannot mean

Nothing without layout can say a response is *visible*, and `pageharness.py`
refuses to pretend otherwise. So the measurable form is:

> **A press produced something when at least one of three observable things
> changed: the page POSTed, the control's own at-control slot became non-empty,
> or ANY node in the page with an id changed its markup or its classes.**

The third clause is *any addressable node*, not *the surface the control lives
on*, and the difference was found the hard way. `innerHTML` on this shim is what
was **assigned**, never a serialization of children — the shim says so in its
own comment — so a panel written into `pv-<id>` inside `#profList` does not
change `#profList`'s string. Watching the surfaces reported six `data-panel`
presses as producing nothing when every one of them had rendered a panel: a
false negative dressed as a finding, which is exactly what L47-D's docstring
warns against one door over.

Strictly weaker than *the user saw it*. Strictly stronger than *the handler
returned* — which is what a delegate-coverage sweep can ask, and is why four
sweeps over this door found none of the 24 drive findings.

## The negative control

`CONTROL_PRESS` is `data-refusal-i` with no refusal open — a guard read off the
page, `if (!LAST_REFUSAL) return;`, before anything else in that handler. It
must come back **produced nothing**: a sweep with no negative case cannot tell
you it is working, and this one is reporting its own numbers.

The first choice was a `data-dismiss` on a finding id that does not exist, and
the instrument reported it as *produced something* — correctly, because
`decide()` does not check that the subject exists and the press POSTs. The
negative control failing on itself is how that was found.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"

#: The three that ship, and the target each is driven to. `GUIDED-097`:
#: `DR1TKCAL` is continuous (regression) on two of them and the third is driven
#: to `DR1TPROT`, so the walk is not one target shape three times.
FIXTURES = (
    ("nhanes_dietary.csv", "DR1TKCAL"),
    ("nhanes_kilojoules.csv", "DR1TKCAL"),
    ("nhanes_partial_design.csv", "DR1TPROT"),
)

#: The routes one full render fetches. Nineteen — a harness that stubs four
#: gets a controller that throws on the rest and a sweep that reports the throw
#: as a finding.
ROUTES = ("interview?step=data", "interview?step=explore",
          "interview?step=features", "capabilities", "features", "recipes",
          "preprocess", "figures", "draft", "manuscript", "models", "training",
          "instability", "explain", "sensitivity", "evidence/plausibility",
          "evidence/missingness")

#: The surfaces the walk reads controls off, in journey order.
SURFACES = ("structList", "profList", "exploreStack", "missBox", "targetCols",
            "taskOverride", "featBuild", "featDecided", "selBuild", "prepCols",
            "repairGroups", "askedQuestions", "skipNote", "blockerBand",
            "appliedBox", "ledgerList", "paletteBox")

#: The negative control: `data-refusal-i` with no refusal open.
#:
#: **The first choice was wrong and the sweep said so.** A `data-dismiss` on a
#: finding id that does not exist looked like the obvious silent press — and it
#: POSTs, because `decide()` does not check that the subject exists. The
#: instrument reported it as *produced something*, correctly, and that is the
#: negative control doing its job on itself. This one is a guard read off the
#: page: `if (!LAST_REFUSAL) return;`, before anything else in the handler.
CONTROL_PRESS = {"data-refusal-i": "0", "data-ac": "refusal"}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "Whether anything is on screen. Layout is the one thing this cannot see.",
    "His file's SCALE. 120 rows against 21,849, and 10 columns against 29 — "
    "four of the 24 drive findings have evidence (15,552 blanks in `meds_hbp`, "
    "a 26-feature histogram pager) that 120 rows cannot produce.",
    "`SEQN` as `float64`. All three shipped fixtures carry it as `int64` and "
    "`identifiers.detect` flags it in all three, so the detection-fails case "
    "is unreachable from any fixture here (L47 §00).",
    "Train and everything after it. The walk stops at Preprocess, which is "
    "where the prompt's gate stops.",
    "SECOND presses. Several handlers toggle and this presses once.",
)


def _walk(client, pid, target):
    """Upload → target → explore → features → preprocess, over HTTP."""
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["dietary"]}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    client.get(f"/project/{pid}/interview?step=explore")
    client.get(f"/project/{pid}/features")
    client.get(f"/project/{pid}/preprocess")


def _routes(client, pid) -> Dict[str, Any]:
    project = client.get(f"/project/{pid}").json()
    out = {f"/project/{pid}": project}
    for path in ROUTES:
        got = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = got.json() if got.status_code == 200 else {}
    # THE DECISION POST ANSWERS WITH A REAL PROJECT, which is what the server
    # returns and what `decide()` assigns to `P`. An unstubbed POST hands the
    # controller a payload with no `findings`, and `findingById` reads
    # `P.findings.length` with no guard — so the sweep's FIRST press crashed the
    # page. That is a real fragility and it is filed rather than fixed here
    # (`GUIDED-188`); it is not what this gate is measuring, and a harness that
    # feeds the app something the server never sends is testing the harness.
    out[f"POST /project/{pid}/decision"] = project
    return out


_SWEEP_JS = """
/* Every control the walk rendered, pressed once, from its OWN attributes.
   Reading the button off the render rather than composing one is the whole
   difference between this and a delegate sweep: a control that stopped
   emitting `data-ac` produces a press that does not carry one. */
function buttons(html){
  var out = [], re = /<button\\b([^>]*)>/g, m;
  while ((m = re.exec(html || ""))){
    var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, kv;
    while ((kv = a.exec(m[1]))) attrs[kv[1]] = kv[2];
    out.push(attrs);
  }
  return out;
}
function settle(n){
  var p = Promise.resolve();
  for (var i = 0; i < n; i++) p = p.then(function(){});
  return p;
}
var SURFACES = __SURFACES__;
var DELEGATED = __DELEGATED__;
var seen = {}, results = [];

/* EVERY ADDRESSABLE NODE, not the seventeen surfaces.
   `innerHTML` on this shim is what was ASSIGNED, never a serialization of
   children — the shim says so in its own comment — so a write into `pv-<id>`
   inside `#profList` does not change `#profList`'s string. Watching only the
   surfaces reported six `data-panel` presses as producing nothing when each had
   written a panel into its own node: a false negative dressed as a finding,
   which is the exact failure L47-D's docstring warns about one door over.
   `__byId` is the shim's own registry of everything with an id. */
function pageState(){
  var s = {};
  Object.keys(__byId).forEach(function(id){
    var el = __byId[id];
    s[id] = (el.innerHTML || "") + "\\u0000" + Object.keys(el._classes).join(" ");
  });
  return s;
}
function movedIds(before, after){
  var out = [];
  Object.keys(after).forEach(function(id){
    if (before[id] !== after[id]) out.push(id);
  });
  return out;
}

async function pressOne(attrs, where){
  var key = null;
  for (var k in attrs) if (DELEGATED.indexOf(k) !== -1) { key = k; break; }
  if (key === null) return null;
  var id = key + "=" + attrs[key];
  if (seen[id]) return null;
  seen[id] = 1;
  if ("disabled" in attrs) {
    results.push({attr: key, id: id, where: where, disabled: true,
                  posted: 0, at: "", moved: []});
    return null;
  }
  var before = pageState();
  var nPosts = __harness.posts().length;
  __harness.dispatch('click', __harness.target(attrs));
  await settle(14);
  var moved = movedIds(before, pageState());
  results.push({
    attr: key, id: id, where: where, disabled: false,
    posted: __harness.posts().length - nPosts,
    at: (attrs['data-ac'] ? (__harness.html('ac-' + attrs['data-ac']) || "") : ""),
    moved: moved.slice(0, 6)});
  return null;
}

(async function(){
  for (var i = 0; i < SURFACES.length; i++){
    var host = SURFACES[i];
    var bs = buttons(__harness.html(host));
    for (var j = 0; j < bs.length; j++) await pressOne(bs[j], host);
  }
  /* THE NEGATIVE CONTROL, through the same delegate. */
  var cbefore = pageState(), cposts = __harness.posts().length;
  __harness.dispatch('click', __harness.target(__CONTROL__));
  await settle(14);
  __emit({results: results,
          control: {posted: __harness.posts().length - cposts,
                    at: __harness.html('ac-' + __CONTROL_AC__) || "",
                    moved: movedIds(cbefore, pageState())}});
})();
"""


def _sweep(fixture: str, target: str) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from turbotab import api, pageharness as PH
    from turbotab import (
        test_every_control_the_page_delegates_survives_being_pressed as base)

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    _walk(client, pid, target)
    routes = _routes(client, pid)

    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    delegated = base.delegated_attributes(page)

    body = (_SWEEP_JS
            .replace("__SURFACES__", json.dumps(list(SURFACES)))
            .replace("__DELEGATED__", json.dumps(delegated))
            .replace("__CONTROL__", json.dumps(CONTROL_PRESS))
            .replace("__CONTROL_AC__", json.dumps(CONTROL_PRESS["data-ac"])))
    out = PH.run(body, routes=routes, search=f"?project={pid}")
    out["pid"] = pid
    return out


def _produced(row: Dict[str, Any]) -> bool:
    return bool(row["posted"] or row["at"] or row["moved"])


@pytest.fixture(scope="module")
def swept():
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    return {fixture: _sweep(fixture, target) for fixture, target in FIXTURES}


def test_the_negative_control_produces_nothing(swept):
    """The sweep can say no. Without this its yes means nothing.

    A `data-dismiss` on a finding id that is not in the project: same delegate,
    same `decide()`, same POST path. It must come back silent.
    """
    for fixture, out in swept.items():
        control = out["control"]
        assert not (control["posted"] or control["at"] or control["moved"]), (
            f"{fixture}: the negative control — a `data-dismiss` on "
            f"{CONTROL_PRESS['data-dismiss']!r} — produced "
            f"posted={control['posted']} at={control['at'][:80]!r} "
            f"moved={control['moved']}. Every 'produced something' below is "
            f"then unfalsifiable")


def test_the_walk_reaches_controls_at_all(swept):
    """The positive control. Zero presses is the same output as zero failures."""
    for fixture, out in swept.items():
        rows = out["results"]
        assert len(rows) >= 8, (
            f"{fixture}: the walk found only {len(rows)} control(s) across "
            f"{len(SURFACES)} surfaces, so the sweep below is measuring almost "
            f"nothing. Either the render broke or the walk stopped early")


def test_every_control_pressed_produces_something(swept, capsys):
    """The loop's gate.

    Reports first and asserts second — a sweep that prints only its failures
    has not said how much it looked at (`LOOP.md` §10).
    """
    silent: List[str] = []
    with capsys.disabled():
        print("\n  ── L48-D · every control he presses, on the nearest file ──")
        print("  DEFINITION: a press produced something when the page POSTed,")
        print("  or its own `ac-<data-ac>` slot became non-empty, or ANY node")
        print("  with an id changed its markup or classes. Not visibility —")
        print("  layout is the one thing this cannot see.")
        for fixture, out in swept.items():
            rows = out["results"]
            live = [r for r in rows if not r["disabled"]]
            produced = [r for r in live if _produced(r)]
            quiet = [r for r in live if not _produced(r)]
            print(f"\n  {fixture}")
            print(f"    controls rendered and reached     {len(rows)}")
            print(f"      rendered `disabled`             "
                  f"{len(rows) - len(live)}")
            print(f"      pressed                         {len(live)}")
            print(f"        produced something            {len(produced)}")
            print(f"        produced nothing              {len(quiet)}")
            for row in quiet:
                print(f"            {row['id'][:58]:<58} on #{row['where']}")
                silent.append(f"{fixture}: {row['id']} on #{row['where']}")
        print(f"\n  shapes NOT covered                  "
              f"{len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")

    assert not silent, (
        "these controls rendered on the walk, were pressed, and produced "
        "nothing observable — no POST, nothing at the control, no surface "
        "moved:\n  " + "\n  ".join(silent))
