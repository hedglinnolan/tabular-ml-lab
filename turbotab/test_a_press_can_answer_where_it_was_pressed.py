"""L47-D — the fifth granularity. Can a control answer where it was pressed?

Four automated sweeps — **route, field, name, stand-in**, built across L42–L44 —
ran over this door in the four preceding loops and **found none of the 24 drive
findings.** Not one of the 24 is a missing capability; every one is a seam
between two correct things, and no sweep that asks *does this exist* can see a
seam.

This is the sweep that would have found `GUIDED-167`, `173` and `161`.

## What it measures, and the definition it publishes

`ROADMAP.md` condition 7's third half has no instrument and `PRODUCT_VISION.md`
§06b calls that the honest gap: **nothing measures whether what arrives is
legible**, and nothing without layout can. This does not attempt it. It converts
an unmeasurable property — *is it on screen* — into a measurable one:

> **A press can answer AT THE CONTROL when the page can write into a node whose
> id is `ac-<the control's own `data-ac`>`.** A control with no such slot is
> counted as **no slot**, not as *renders elsewhere* — because without layout
> this instrument genuinely cannot tell "renders elsewhere" from "renders
> nowhere", and reporting the two as one would be the overstatement
> `pageharness.py` refuses in its own docstring.

Strictly weaker than visibility. Strictly stronger than *somewhere in the page*,
which is what every check before it could ask.

**And say what the definition costs**, because a subtree measure has edges in
both directions. `data-dismiss`'s undo note renders as the *immediate next
sibling* of the article it hid — outside the slot by this definition and exactly
where the user is looking. `data-att-send` renders into a full-width band —
inside the page, far away by eye. **The definition decides the numbers, so the
numbers travel with it.**

## The false negative this must not manufacture

At least twelve delegated handlers return early on **page state** rather than on
the DOM: `data-refusal-i` on `LAST_REFUSAL`, `data-exit-for` and `data-att-send`
on `LAST_EXPLORE_PLAN`, `data-answer-key`/`data-answer-commit` on `ANSWERABLE`,
`data-rg-pick`/`data-rg-apply` on `RG_OPEN` and `PICKED`, `data-miss-choose` on a
dataset the mechanism control writes — **and `decide()` itself returns early on
`!P`, which is every one of the twenty-five posting handlers at once.**
`data-att-send` is `disabled` until a typed sentence matches, and the delegate
bails on `t.disabled` before any handler runs.

**A synthetic attribute-only press on any of these sends nothing**, and a sweep
that scored them *"pressed, nothing changed"* would be publishing a false
negative dressed as a finding. They are counted as **unconstructible** and named.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"

#: Handlers that return early on page state a synthetic press cannot establish.
#: Each is a real guard read off the page, not a guess — the value is the guard.
UNCONSTRUCTIBLE: Dict[str, str] = {
    "data-refusal-i": "if (!LAST_REFUSAL) return;",
    "data-exit-for": "the exit is looked up in LAST_EXPLORE_PLAN.questions",
    "data-att-send": "the button is `disabled` until the typed sentence matches, "
                     "and the delegate bails on `t.disabled` first",
    "data-answer-key": "spec = ANSWERABLE[key]; if (!spec) return;",
    "data-answer-commit": "submitAnswer reads ANSWERABLE[ckey] and returns if absent",
    "data-miss-choose": "returns unless data-miss-mech is set, which only "
                        "data-miss-mech-for writes",
    "data-rg-pick": "g0 = RG_OPEN[pkk]; if (!g0) return;",
    "data-rg-apply": "ga = RG_OPEN[ak]; if (!ga) return; and if (!picks.length) return;",
}

#: Attributes whose handler posts only on one branch. Counted apart, because
#: "this control posts" is false for the other branch and a sweep that called
#: them posters would be measuring a path that does not exist.
CONDITIONAL = {
    "data-answer-key": "posts only on the single-select branch; multi-select "
                       "accumulates into PENDING and returns",
    "data-exit-for": "posts nothing when the exit is typed — that post is "
                     "data-att-send's",
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "Whether an at-control response is VISIBLE. Nothing without layout can, and "
    "this file does not pretend otherwise.",
    "The three non-delegated POST paths — `pickBtn`, the `fileInput` change and "
    "the drop zone — which all reach the same `POST /project` and are not "
    "presses of a delegated control.",
    "Second presses. Several handlers toggle, and the sweep presses once.",
)


def _delegated() -> List[str]:
    from turbotab import (
        test_every_control_the_page_delegates_survives_being_pressed as base)

    return base.delegated_attributes(PAGE.read_text(encoding="utf-8"))


def _script() -> str:
    page = PAGE.read_text(encoding="utf-8")
    return page[page.index("<script>"):page.rindex("</script>")]


_HANDLER = re.compile(r'hasAttribute\("([a-z-]+)"\)|closest\("\[([a-z-]+)\]"\)')


def _posters(script: str) -> List[str]:
    """Attributes whose handler body reaches a POST.

    Derived by slicing the script between one handler's guard and the next and
    looking for a posting call in between — rather than hand-listed, because a
    hand list is how a control added next loop goes unmeasured, which is this
    file's own subject one level up.
    """
    hits = [(m.start(), m.group(1) or m.group(2))
            for m in _HANDLER.finditer(script)]
    out = []
    for i, (start, attr) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(script)
        body = script[start:end]
        if re.search(r"\bdecide\(|\bpost\(|method\s*:\s*\"POST\"", body):
            out.append(attr)
    return sorted(set(out))


_ELEMENT = re.compile(r"<(?:button|option|select|input|a)\b")


def _slots(script: str) -> List[str]:
    """Attributes emitted on an element that also carries `data-ac`.

    **The derivation is per-ELEMENT, not per-window**, and that changed at L48.
    The first version scanned 400 characters backwards from each `data-ac="` and
    credited every `data-*` it found there — which credits a neighbour's
    attribute to a slot that is not its own. It happened not to be wrong on the
    page it was written against (re-measured under both readings, L47's three
    are the same three either way, and the loose reading additionally credited
    `data-panel`, which is not a poster and so never reached the number). But a
    gate is about to rest on this count, and a heuristic that is right by
    coincidence is not a gate.

    Each `<button`/`<option`/`<select`/`<input`/`<a` opens a chunk that runs to
    the next such opener or to the element's own close tag, whichever comes
    first. Attributes inside one chunk belong to one control.
    """
    out = set()
    starts = [m.start() for m in _ELEMENT.finditer(script)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(script)
        close = script.find("</", start)
        if close != -1 and close < end:
            end = close
        chunk = script[start:end]
        attrs = set(re.findall(r"data-[a-z-]+(?==)", chunk))
        if "data-ac" in attrs:
            out |= attrs - {"data-ac"}
    return sorted(out)


def test_the_enumeration_is_derived_rather_than_listed():
    """The instrument's own precondition.

    If this file hand-listed its controls it would measure the page it was
    written against, which is what the delegate test did for five attributes and
    four loops.
    """
    attrs = _delegated()
    assert len(attrs) >= 50, (
        f"only {len(attrs)} delegated attributes were derived; the page has "
        f"three click delegates and the extractor used to see one")
    posters = _posters(_script())
    assert posters, "no posting handler was derived at all"
    assert "data-dismiss" in posters and "data-earmark" in posters


def test_the_sweep_reports_its_own_coverage(capsys):
    """`LOOP.md` §10. Counts, a stated definition, and what was dropped."""
    script = _script()
    attrs = _delegated()
    posters = [a for a in _posters(script) if a in attrs]
    slots = _slots(script)

    at_control = sorted(a for a in posters if a in slots)
    unconstructible = sorted(a for a in posters if a in UNCONSTRUCTIBLE)
    no_slot = sorted(a for a in posters
                     if a not in slots and a not in UNCONSTRUCTIBLE)

    with capsys.disabled():
        print("\n  ── L47-D · can a press answer where it was pressed ──")
        print("  DEFINITION: a press can answer AT THE CONTROL when the page")
        print("  can write into a node whose id is `ac-<its own data-ac>`.")
        print("  A control with no such slot is NO SLOT, not 'elsewhere' —")
        print("  without layout this cannot tell elsewhere from nowhere.")
        print()
        print(f"  delegated attributes (3 delegates)  {len(attrs)}")
        print(f"  of those, handlers that post        {len(posters)}")
        print(f"    can answer at the control         {len(at_control)}")
        for a in at_control:
            print(f"        {a}")
        print(f"    unconstructible by a synthetic press  {len(unconstructible)}")
        for a in unconstructible:
            print(f"        {a:<22} {UNCONSTRUCTIBLE[a][:60]}")
        print(f"    no slot                           {len(no_slot)}")
        print(f"        {', '.join(no_slot)}")
        print(f"  conditional posters                 {len(CONDITIONAL)}")
        for a, why in CONDITIONAL.items():
            print(f"        {a:<22} {why[:60]}")
        print(f"  shapes NOT covered                  {len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")

    assert at_control, (
        "no posting control can answer where it was pressed, so L47-C shipped "
        "nothing")


#: Response-producing controls that deliberately carry no `data-ac`, each with
#: the reason. A control listed here is a DECISION; a control silently absent is
#: a hole — the same distinction `GUIDED-180` draws one layer down.
#:
#: **All three are the mechanism under a different id.** Each writes BOTH its
#: answer and its failure into a dedicated per-row panel it addresses by name —
#: `missprev-<col>-<opt>`, `[data-rg-body]`, `[data-teach-body]` — which is
#: adjacency by the same argument `ac-<key>` is, and predates it. Giving them a
#: second slot would put two receipts on one press.
NO_SLOT_BY_DESIGN: Dict[str, str] = {
    "data-miss-preview": "writes success AND `.catch` into `missprev-<col>-"
                         "<opt>`, its own per-option panel",
    "data-rg-show": "writes success AND `.catch` into `[data-rg-body]`, its "
                    "own per-group panel",
    "data-teach": "writes success AND `.catch` into `[data-teach-body]`, its "
                  "own per-question panel",
}

#: Attributes whose handler reaches a GET but never a POST.
#:
#: **L48-A's gate said "posting controls" and that was too narrow**, which the
#: `GUIDED-176` work found: `data-feat-preview` is a GET, its 400 carries a
#: string `detail`, and its whole failure path went to `setErr` — the exact
#: defect the posting sweep was built for, on a control the sweep did not look
#: at. A press produces a response or it does not; whether the verb is GET or
#: POST is not a property of what a user sees.
_FETCH = re.compile(r"\bapi\(|\bjob\(")


def _fetchers(script: str) -> List[str]:
    """Attributes whose handler reaches a GET and no POST.

    Derived the same way `_posters` is, and for the same reason: a hand list is
    how the control added next loop goes unmeasured.
    """
    hits = [(m.start(), m.group(1) or m.group(2))
            for m in _HANDLER.finditer(script)]
    out = []
    for i, (start, attr) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(script)
        body = script[start:end]
        if _FETCH.search(body) and not re.search(
                r"\bdecide\(|\bpost\(|method\s*:\s*\"POST\"", body):
            out.append(attr)
    return sorted(set(out))


def test_every_posting_control_can_answer_where_it_was_pressed(capsys):
    """L48-A1's gate. `GUIDED-167`, as an invariant rather than a number.

    L47 shipped the mechanism and wired three of thirty consumers, which is trap
    #1 — a capability ahead of its consumers — with the capability's own sweep
    publishing the shortfall. The row was marked `FIXED` on a test that pressed
    a finding card while the row's own evidence is a `data-miss-choose`, and the
    adjudicator reopened it to `PARTIAL` for exactly that. This is the version
    that cannot close on the wrong instance: **every** posting control, or a
    named exception carrying its reason.

    The count is a floor and never a ceiling — a control added next loop with no
    slot fails here, which is the whole point of deriving the enumeration rather
    than listing it.
    """
    script = _script()
    attrs = _delegated()
    posters = [a for a in _posters(script) if a in attrs]
    fetchers = [a for a in _fetchers(script) if a in attrs]
    responders = sorted(set(posters) | set(fetchers))
    slots = _slots(script)

    missing = sorted(a for a in responders
                     if a not in slots and a not in NO_SLOT_BY_DESIGN)
    declared = sorted(a for a in responders if a in NO_SLOT_BY_DESIGN)

    # The row's own instance, named rather than left to the aggregate. A count
    # of thirty can be reached while the one control the finding was filed about
    # is still missing, which is precisely how the row closed on the wrong
    # instance last loop.
    assert "data-miss-choose" in slots, (
        "`data-miss-choose` still carries no slot. That is the press in "
        "`GUIDED-167`'s own evidence, and the row was reopened for closing "
        "without it")

    with capsys.disabled():
        print("\n  ── L48-A1 · the gate ──")
        print("  SCOPE: every control that PRODUCES A RESPONSE, post or GET.")
        print("  L48-A said `posting` and that was too narrow — see the note on")
        print("  NO_SLOT_BY_DESIGN; a GET refuses exactly as a post does.")
        print(f"  controls that post                  {len(posters)}")
        print(f"  controls that GET and never post    {len(fetchers)}")
        print(f"  response-producing, together        {len(responders)}")
        print(f"    with a slot                       "
              f"{len([a for a in responders if a in slots])}")
        print(f"    declared as having none           {len(declared)}")
        for a in declared:
            print(f"        {a:<22} {NO_SLOT_BY_DESIGN[a][:52]}")
        print(f"    neither                           {len(missing)}")

    assert not missing, (
        f"{len(missing)} posting control(s) can neither answer where they were "
        f"pressed nor say why not: {missing}. Add `data-ac` and an "
        f"`atControlSlot` beside the control, or declare it in "
        f"NO_SLOT_BY_DESIGN with the reason.")


def test_every_unconstructible_guard_is_really_in_the_page():
    """The positive control.

    A list of excuses nobody checks is worse than no list: it would let a real
    "pressed, nothing changed" hide behind an invented guard. Each entry names a
    guard, and each guard has to be findable.
    """
    script = _script()
    for attr, guard in UNCONSTRUCTIBLE.items():
        assert f'"{attr}"' in script, f"{attr} is not dispatched on at all"
    for literal in ("if (!LAST_REFUSAL) return;", "if (!g0) return;",
                    "if (!picks.length) return;", "if (!spec) return;"):
        assert literal in script, (
            f"{literal!r} is not in the page, so an entry in UNCONSTRUCTIBLE "
            f"rests on a guard that does not exist")
    # AND `decide` ITSELF. The universal one — every posting handler that routes
    # through it no-ops with no project, which is twenty-five controls at once.
    assert re.search(r"function decide\([^)]*\)\{\s*\n\s*if \(!P\) return;",
                     script), "decide's own project guard is gone"


def test_a_control_that_can_answer_actually_does(capsys):
    """The end-to-end half — the sweep is not only a static count.

    Drives a real refusal into a real slot, so *"N can answer at the control"*
    is a measured claim rather than a count of attributes.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / "clinical_labs.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_labs.csv", handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["clinical"]}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "readmitted"}})

    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                            else {})
    routes[f"POST /project/{pid}/decision"] = {
        "__status": 400, "body": {"detail": "swept refusal"}}

    out = PH.run(
        "var m = (__harness.html('profList') || '').match(/id=\"ac-(find-[^\"]+)\"/);\n"
        "if (!m) { __emit({none: true}); } else {\n"
        "  __harness.dispatch('click', __harness.target("
        "    {'data-dismiss': m[1].slice(5), 'data-ac': m[1]}));\n"
        "  for (var i = 0; i < 8; i++) "
        "    await new Promise(function(r){ setTimeout(r, 0); });\n"
        "  __emit({at: __harness.html('ac-' + m[1])});\n}",
        routes=routes, search=f"?project={pid}")
    assert not out.get("none"), "no finding card rendered a slot to press"
    assert "swept refusal" in (out["at"] or ""), (
        f"the slot exists and nothing arrived in it: {out['at']!r}")


#: `ACTION_CONTRACT`'s hole and `EFFECTS`' — **the same lens one surface over**,
#: and bigger than the thing this file was pointed at. Reported, and filed as
#: rows; deliberately not fixed here, because fixing them is a loop.
def test_the_same_lens_one_surface_over(capsys):
    """Which decision kinds nothing watches, and which say nothing to the user.

    Derived, both of them. `ACTION_CONTRACT.get(kind)` returns `None` for an
    unlisted kind and all three consumers then **skip the check entirely** —
    `a_deferred_transform_leaves_the_table_byte_identical`,
    `after_an_edit_exactly_the_right_things_are_stale`, and
    `every_decision_taken_appears_in_the_record`. A missing row is not flagged as
    missing; it is silently unchecked. `effectOf` does the same one layer up,
    falling back to *"Records your answer."*
    """
    from turbotab import api, devchecks

    source = Path(api.__file__).read_text(encoding="utf-8")
    live = set(re.findall(r'decision\.kind == "([a-z_]+)"', source))
    # The `in (...)` branches, one entry each — the first version took only the
    # FIRST such branch and split it carelessly, which produced a phantom empty
    # kind and a duplicate. A counting instrument that miscounts is worse than
    # none, and this one is printing its own numbers.
    for group in re.findall(r'decision\.kind in \(([^)]*)\)', source):
        live |= {k.strip().strip('"\'') for k in group.split(",")}
    # The terminal whitelist: kinds with no branch of their own, accepted by the
    # generic tail at the bottom of the handler.
    live |= {"dismiss", "undismiss", "flag", "unflag", "note", "defer"}
    live = {k for k in live if k and k.replace("_", "").isalpha()}

    script = _script()
    effects = set(re.findall(r"^\s{4}([a-z_]+): function\(", script, re.M))

    no_contract = sorted(live - set(devchecks.ACTION_CONTRACT))
    no_effect = sorted(live - effects)

    with capsys.disabled():
        print("\n  ── the same lens one surface over ──")
        print(f"  live decision kinds                 {len(live)}")
        print(f"  with no ACTION_CONTRACT row         {len(no_contract)}")
        print(f"      {', '.join(no_contract)}")
        print(f"  with no EFFECTS sentence            {len(no_effect)}")
        print(f"      {', '.join(no_effect)}")
        print("  Both tables SKIP an unlisted kind rather than flagging it:")
        print("    devchecks.py:641/:661/:740 all `.get(kind)` and return [].")
        print("    index.html effectOf falls back to 'Records your answer.'")
        print("  Filed, not fixed — GUIDED-180 and GUIDED-181.")

    # The two bulk mutators are the point, so they are asserted rather than
    # printed: they rewrite the working table with no contract watching.
    assert "apply_bulk" in no_contract or "apply_bulk" in devchecks.ACTION_CONTRACT
    assert live, "no decision kinds derived at all"
