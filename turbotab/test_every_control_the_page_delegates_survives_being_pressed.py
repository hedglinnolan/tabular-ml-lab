"""L42-C — `GUIDED-140`, in the form its own `act` field asks for.

`nudge()` was deleted at `DRIVE-006` and **seven call sites outlived it**, so
every pull affordance in the door threw `ReferenceError` into the panel that
means *we do not have this* (`GUIDED-139`). Nothing in this repository could
find the survivors: the page is one 4,000-line script and no check enumerates
the names it calls against the names it defines.

## The regex version is not built, and that is the finding rather than a gap

`GUIDED-140`'s row records the measurement: a regex over the page reported
**37** false positives one way and **15** the other. Stripping comments before
strings lets a `/*` inside a string literal eat the file; stripping strings
first lets an apostrophe in a comment — *"the page's own"* — open a literal that
runs to the next apostrophe and swallows every `function` declaration in
between. **A guard with a double-digit false-positive rate is one the next
person deletes**, which the standing route check's own comment already says in
those words.

So the row's `act` names two options and prefers the second:

> *Either run the script through node with a linter that reports undefined
> identifiers, or extend `turbotab/pageharness.py` — it already loads the script
> under a DOM shim in node, so a pass that collects `ReferenceError`s while
> exercising every registered click handler is the same harness with a wider
> drive. **The second is preferable because it also catches a name that exists
> and is not a function.***

This is the second. It has no exemption list because it has no false positives:
a `ReferenceError` from the page's own controller is not a heuristic.

## What it does not do, said plainly

A behavioral check finds a dead name **on the paths it drives**. Six of
`nudge`'s seven call sites were on paths nothing drove, which is exactly why
`GUIDED-139` survived — so this narrows that surface without closing it, and
`GUIDED-140` stays `OPEN` for the static half.

**The two are not redundant and that matters** (the loop prompt asks the
question directly). Part D's drive presses the controls the page *rendered* on
one project; this presses every control the click delegate *declares*, whether
or not a fixture happens to render one. A name behind a control no fixture
produces is invisible to D and visible here. Neither subsumes the other, and
two independent detectors for a `critical` class is the right number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"

#: **Derived from the delegates themselves.** The page declares which attributes
#: it handles in `closest(...)` selectors; hand-copying them here is how a control
#: added next loop goes undriven, which is this file's own subject one level up.
#:
#: **AND IT WAS DERIVING ONLY ONE OF THREE.** Found at L47-D. This pattern was
#: `e\.target\.closest\(`, and the page's second and third click delegates are
#: written `ev.target.closest(` — so it matched neither, and the winner-takes-all
#: below then discarded whatever it did match beyond the largest. **Two
#: independent structural bugs, either of which alone hides a delegate:**
#:
#: 1. The event parameter's NAME was hard-coded. Any delegate whose parameter is
#:    not `e` was invisible.
#: 2. `if len(found) > len(best): best = found` keeps the single largest match
#:    and drops the rest. The second and third delegates dispatch through five
#:    separate one- and two-selector `closest` calls, so even with the name fixed
#:    each would have lost the length race to the 47-attribute one.
#:
#: The cost: **five delegated attributes were never pressed by this file** —
#: `data-earmark`, `data-offer-preview`, `data-answer-key`, `data-answer-commit`,
#: `data-teach` — and three of those five post. One of them, `data-earmark`, is
#: `GUIDED-161`'s own control: the guard that would have caught it could not see
#: it.
#: The selector list is grammar'd as `"…" ( + "…" )*` with the `+` REQUIRED
#: between strings — JS string concatenation has no other spelling, and making
#: it optional (`\s*\+?\s*` inside the repetition) gave two adjacent `\s*` runs
#: exponentially many ways to split the whitespace between strings when no
#: closing paren follows (CodeQL: inefficient regular expression). Verified
#: against the page: both forms extract the identical attribute set.
_DELEGATE = re.compile(
    r"[A-Za-z_$][\w$]*\.target\.closest\(\s*"
    r"(\"[^\"]*\"(?:\s*\+\s*\"[^\"]*\")*)\s*\)")


def delegated_attributes(page: str) -> list:
    """Every `data-*` attribute any click delegate dispatches on.

    A UNION, not the largest single match — see the second bug above.
    """
    found: set = set()
    for match in _DELEGATE.finditer(page):
        selector = "".join(re.findall(r'"([^"]*)"', match.group(1)))
        found |= set(re.findall(r"\[([a-z-]+)\]", selector))
    # `data-tip` is dispatched from the MOUSEMOVE delegate, not a click one. It
    # is not pressable and pressing it would be this file inventing a control.
    found.discard("data-tip")
    return sorted(found)


#: NOT COVERED, said out loud.
#:
#: THE STATIC HALF. `GUIDED-140` stays `OPEN`: this finds a dead name on the
#: paths it drives, and six of `nudge`'s seven call sites were on paths nothing
#: drove. A tokenizer would find them without a drive and is not built here.
#:
#: NON-CLICK HANDLERS. `change`, `input` and `submit` have their own listeners.
#: The delegate this reads is the click one, which is where `runPull` lives and
#: where `GUIDED-139` landed.
#:
#: A CONTROL'S SECOND PRESS. Each is pressed once. A handler that throws only on
#: the toggle-off path is not reached.
SHAPES_NOT_COVERED = [
    "the static half — a tokenizer over the page's own names; GUIDED-140 stays "
    "OPEN for it",
    "change / input / submit handlers — this reads the click delegate",
    "a control's second press, so a toggle-off path that throws is not reached",
]


@pytest.fixture(scope="module")
def driven():
    """One project, far enough along that most controls exist."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "clinical_labs.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_labs.csv", handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": ["clinical"]}),
                          ("set_target", {"column": "readmitted"})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, ok.text[:200]

    def get(tail):
        return client.get(f"/project/{pid}{tail}").json()

    routes = {
        f"/project/{pid}": get(""),
        f"/project/{pid}/interview?step=data": get("/interview?step=data"),
        f"/project/{pid}/interview?step=explore": get("/interview?step=explore"),
        f"/project/{pid}/evidence/missingness": get("/evidence/missingness"),
        f"/project/{pid}/evidence/plausibility": get("/evidence/plausibility"),
        f"/project/{pid}/capabilities": get("/capabilities"),
        f"/project/{pid}/draft": get("/draft"),
        f"/project/{pid}/figures": get("/figures"),
    }
    return routes, pid


#: **A thrown error is only half of it, and the missing half is the half that
#: matters.** The first version caught exceptions around `dispatch` and a revert
#: probe reported `GREEN — NOT LOAD-BEARING` on `GUIDED-139` itself: `nudge(box)`
#: sits inside `runPull`'s `.then()`, so the `ReferenceError` is raised
#: asynchronously, caught by the page's own `.catch()`, and **rendered into the
#: panel as text**. Nothing escapes to the caller.
#:
#: That is the app's error handling working — and it is exactly why the defect
#: was invisible for as long as it was. So the check reads both: what threw, and
#: what the page *wrote down* about a name it could not resolve.
#:
#: **Two signatures, and the narrowing is deliberate.** `Cannot read properties
#: of undefined` was in this tuple and had to come out: pressing
#: `data-target-col` with a column name no table has produces exactly that, so
#: it fires on the probe's own synthetic input rather than on a defect. What is
#: left are the two failures a bad *value* cannot cause — an unresolved NAME
#: (`is not defined`) and a name that resolved to something uncallable (`is not
#: a function`), which are the two halves of `GUIDED-140`'s class.
_ERROR_TEXT = ("is not defined", "is not a function")

_PRESS = """
var ATTRS = %s;
var IDS = %s;
var errors = [];
var pressed = 0;
/* Each control is dispatched THROUGH THE PAGE'S OWN DELEGATE. `__harness.target`
   builds a synthetic element with the attribute set, which is what `closest()`
   answers on — so the handler under test is the page's, not a stand-in for it. */
ATTRS.forEach(function(attr){
  var a = {};
  /* A value the handler will not find in any registry, deliberately: the
     question is whether the code PATH survives, and a handler that returns
     early on a missing lookup has still executed the lines before the lookup —
     which is where `nudge(box)` sat. */
  a[attr] = "probe";
  a["data-endpoint"] = "plausibility";
  try {
    __harness.dispatch("click", __harness.target(a, []));
    pressed++;
  } catch (e) {
    errors.push("threw :: " + attr + " :: " + (e && e.name) + ": " + (e && e.message));
  }
});
/* Two ticks, because a handler that fetches resolves its `.then()` on the
   microtask queue and a single `setTimeout(0)` can run before it. */
setTimeout(function(){ setTimeout(function(){
  var dom = "";
  IDS.forEach(function(k){ dom += (__harness.html(k) || ""); });
  __emit({pressed: pressed, errors: errors, dom: dom.slice(0, 120000)});
}, 0); }, 0);
"""


def test_the_attribute_list_comes_from_the_delegate(driven):
    """Derived, not hand-listed. A control added next loop is pressed because
    the page declares it, not because somebody remembered."""
    attrs = delegated_attributes(PAGE.read_text(encoding="utf-8"))
    assert len(attrs) > 30, f"the delegate scan found {len(attrs)}: {attrs}"
    for expected in ("data-look", "data-panel", "data-promote", "data-flag"):
        assert expected in attrs, f"{expected} is handled and was not found"


def test_no_delegated_control_throws_a_reference_error(driven):
    """**`GUIDED-139` as a standing check.** A `ReferenceError` from the page's
    own controller is not a heuristic — there is nothing to exempt and nothing
    to tune, which is the whole argument for building this half rather than the
    regex one.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    from turbotab import fieldsweep as FS

    routes, pid = driven
    page = PAGE.read_text(encoding="utf-8")
    attrs = delegated_attributes(page)
    out = PH.run(_PRESS % (json.dumps(attrs),
                           json.dumps(FS.container_ids(page))),
                 routes=routes, search=f"?project={pid}")

    assert out["pressed"] >= len(attrs) - 2, (
        f"only {out['pressed']} of {len(attrs)} controls were dispatched at "
        f"all, so this proves little about the rest")
    fatal = [e for e in out["errors"]
             if "ReferenceError" in e or "is not a function" in e]
    # AND WHAT THE PAGE WROTE DOWN, which is where `GUIDED-139` actually landed.
    for signature in _ERROR_TEXT:
        if signature in out["dom"]:
            at = out["dom"].index(signature)
            fatal.append(f"rendered :: …{out['dom'][max(0, at - 90):at + 40]}…")
    assert not fatal, (
        "pressing these controls reached a name the page does not define:\n  "
        + "\n  ".join(fatal)
        + "\n\nThat is GUIDED-139's shape: `nudge()` was deleted at DRIVE-006 "
          "and seven callers outlived it. A ReferenceError inside a `.then()` "
          "is caught by the page's own error handling and RENDERED, which is "
          "why the text is checked as well as the throw.")


def test_the_probe_can_fail(driven):
    """The positive control, and this file is worthless without it. A press
    that reached a dead name must be reported — otherwise the green above means
    only that nothing was dispatched.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    routes, pid = driven
    out = PH.run(
        """
        var errors = [];
        try { aFunctionNobodyDefined(1); }
        catch (e) { errors.push("probe :: " + e.name + ": " + e.message); }
        __emit({errors: errors});
        """,
        routes=routes, search=f"?project={pid}")
    assert any("ReferenceError" in e for e in out["errors"]), (
        "a call to a name nothing defines did not surface as a ReferenceError, "
        "so the check above cannot see one either")


def test_this_and_the_field_sweep_catch_guided_139_independently():
    """**The loop prompt asks this directly**: is this the only thing that
    would have caught `GUIDED-139`?

    No, and the two are not redundant. Part D's drive presses the controls a
    fixture *rendered*; this presses every control the delegate *declares*.
    A name behind a control no fixture produces is invisible to D and visible
    here — and a name reached only after a real payload lands is the reverse.

    Asserted rather than claimed: the delegate declares controls that the
    clinical fixture's DOM does not contain, so the two press sets genuinely
    differ.
    """
    page = PAGE.read_text(encoding="utf-8")
    declared = set(delegated_attributes(page))
    # What Part B's drive presses, read from its own source rather than
    # restated — two lists of one thing are two things to drift.
    from turbotab import fieldsweep as FS
    pressed_by_the_sweep = set(re.findall(r'"(data-[a-z-]+)"', FS._READ_ALL))
    assert pressed_by_the_sweep, "the sweep presses nothing"
    assert declared - pressed_by_the_sweep, (
        "the delegate declares nothing the field sweep does not already press, "
        "so this check is the sweep restated and one of them should go")
    assert len(declared) > len(pressed_by_the_sweep)


def test_the_regex_version_is_declared_unbuilt_rather_than_shipped_quiet():
    """`GUIDED-140` stays `OPEN`, and the reason is a measurement rather than a
    preference. A guard with 15 to 37 false positives is one the next person
    deletes."""
    import json as _json

    rows = _json.load(open("docs/turbotab/data/findings.json"))
    row = next(r for r in rows if r["id"] == "GUIDED-140")
    assert row["status"] == "OPEN", (
        "GUIDED-140 was closed and the static half is not built; a behavioral "
        "check finds a dead name only on the paths it drives, and six of "
        "nudge's seven call sites were on paths nothing drove")
