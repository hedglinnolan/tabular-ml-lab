"""`DRIVE-006` — the nudge skipped past the card the user was reading.

The product owner's reading, and it is the specification: *if cards simply build
out, they scroll themselves.*

## The history, because it is the argument

Three positions on one rule. The original was **never auto-scroll**. Building
the prototype appeared to disprove it and DESIGN_LANGUAGE §05 was revised to
*new content is nudged into view only when it sits below the viewport*. That
revision was written against a prototype where a section held two or three
cards, so "below the viewport" meant "the next card" and the nudge landed on the
thing that had just appeared.

On `metabolomics_untargeted.csv` there are nine structural findings. Revealing
the Explore section scrolled the user from the middle of those findings to the
top of a section below them — past the card they were reading, every time.

The thing worth extracting is not that the threshold was wrong. It is **why a
threshold was there at all**: the revised rule had a size-dependent condition in
it, so it was right at one dataset size and wrong at another, and nothing in the
interface could tell which one it was in. The rule that replaces it has no free
parameter, which is why it cannot be wrong at the next scale.

## What this asserts

That the page moves the viewport **zero** times across a whole drive — asserted
as a count rather than as "not during reveal", because a nudge reintroduced
anywhere else would be the same defect from a different function, and the
absence of a whole category is the right shape when the guarantee is a
subtraction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

pytestmark = pytest.mark.skipif(
    not H.available(), reason="no JS engine on this machine")


def _drive(name: str, target: str, lens=None):
    """Boot the page on a real project and report what it did to the viewport."""
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    if lens:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": lens}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    project = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness":
            client.get(f"/project/{pid}/evidence/missingness").json(),
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/draft": {"paragraphs": []},
        f"/project/{pid}/gaps": {"gaps": []},
    }
    return H.run(
        # `drainRaf` is called deliberately: the old nudge deferred its scroll
        # into `requestAnimationFrame`, so a harness that never drained the
        # queue would report zero scrolls whether or not the defect was there.
        #
        # That was only HALF of what made it observable, and the revert probe
        # found the other half. The shim's `getBoundingClientRect` returned
        # zeros, which is not "no layout" but a claim that the element sits at
        # the top of the viewport — so the nudge's own `is this below the fold?`
        # guard was false, the scroll never ran, and the probe reported the
        # restored defect as GREEN. The shim now reports elements just below the
        # fold, which is the direction a guard should be wrong in.
        "__harness.drainRaf(); __emit({scrolls: __harness.scrolls(), "
        "findings: ((__harness.html('structList') + __harness.html('profList'))"
        ".match(/class=\"find/g) || []).length});",
        routes=routes, search=f"?project={pid}")


def _drive_and_press(name: str, target: Optional[str], *, press: str,
                     hide: bool = False):
    """Boot the page, press one rail step, and report what moved.

    Separate from `_drive` deliberately: every assertion above is about a drive
    in which **nothing is pressed**, and folding a press into that helper would
    make the zero-scroll claim quietly conditional on which arguments a caller
    passed.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    if target is not None:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for step in ("data", "explore", "preprocess", "features", "train"):
        path = f"/project/{pid}/interview?step={step}"
        got = client.get(path)
        if got.status_code == 200:
            routes[path] = got.json()
    for path in (f"/project/{pid}/evidence/missingness",
                 f"/project/{pid}/findings", f"/project/{pid}/figures",
                 "/capabilities", "/dev/status"):
        got = client.get(path)
        if got.status_code == 200:
            try:
                routes[path] = got.json()
            except ValueError:
                pass
    routes.setdefault(f"/project/{pid}/evidence/plausibility", {"columns": []})
    routes.setdefault(f"/project/{pid}/draft", {"paragraphs": []})
    routes.setdefault(f"/project/{pid}/gaps", {"gaps": []})
    hide_it = ("sect.classList.add('is-hidden');\n" if hide else "")
    return H.run(
        "var sect = document.getElementById('sec-" + press + "');\n"
        + hide_it +
        "var revealed = !!sect && !sect.classList.contains('is-hidden');\n"
        "__harness.dispatch('click', __harness.el('map-" + press + "'));\n"
        "__harness.drainRaf();\n"
        "__emit({scrolls: __harness.scrolls(), revealed: revealed});",
        routes=routes, search=f"?project={pid}")


def test_a_drive_with_many_findings_never_scrolls_the_page():
    """The fixture the defect was reported on.

    The finding count is asserted too: on a table with one finding the old nudge
    was nearly harmless, so a test that passed on a short page would prove
    nothing about the case that was broken.
    """
    out = _drive("metabolomics_untargeted", "responder", ["metabolomics"])
    assert out["findings"] >= 5, (
        f"only {out['findings']} finding cards rendered; this test is about "
        f"what the nudge did on a LONG page and needs one")
    assert out["scrolls"] == [], (
        f"the page moved the viewport {len(out['scrolls'])} time(s) during a "
        f"drive: {out['scrolls']}")


@pytest.mark.parametrize("name,target", [
    ("clinic_visits", "outcome"),
    ("clinical_longitudinal", "progressed"),
    ("dietary_recalls", "hba1c"),
])
def test_no_fixture_produces_a_scroll(name, target):
    """The absence of a whole category, across shapes.

    A nudge reintroduced in a different function would be the same defect from
    a new direction, so what is asserted is that nothing anywhere scrolls —
    never that one call site behaves.
    """
    assert _drive(name, target)["scrolls"] == []


def test_the_scroll_helper_is_gone_rather_than_disabled():
    """A dormant helper is a nudge waiting for a caller.

    Checked on the page source, and it is the one thing a text search genuinely
    settles: `window.scrollTo` either appears in the controller or it does not.
    """
    page = H.PAGE.read_text(encoding="utf-8")
    script = page[page.index("<script>"):page.rindex("</script>")]
    assert "window.scrollTo" not in script, (
        "the controller can still move the viewport")
    assert "function nudge(" not in script, (
        "the nudge helper is still defined, so it is one call away from "
        "returning")


# ── `DRIVE-047` · the one movement a user asks for ──────────────────────────
#
# **`LOOP.md` §06.2 is invoked here, in those words, and this comment is the
# record of it.** This file's guard is amended in the same loop as the finding
# that pressured it, which §06 forbids by default — *never accept a moved
# threshold in the same loop as the change that pressured it* — and the
# exception's test is whether the entry changes PURPOSE or VALUE.
#
# It changes the purpose, and narrows it rather than relaxing it:
#
#   before: the page moves the viewport zero times.
#   after:  the page moves the viewport zero times EXCEPT from a control whose
#           only purpose is to go somewhere — and the rail must actually go.
#
# `DRIVE-006`'s rule was about the page moving a reader who did not ask. Its
# own stated rationale says so: *cards build downward and the user's scroll
# follows them.* A rail step is the user pressing the word "Train".
# `research/INTERACTION_PACK.md` §04.1 is the sourced half — the Layout
# Instability API excludes a shift within **500 ms of user input** from CLS
# entirely, because temporal proximity makes the causality legible. **[SETTLED]**
# for the spec.
#
# Three things make this an amendment rather than a hole. Every drive above
# still asserts **zero** scrolls and none of them presses the rail. The
# exemption is one attribute, named, and anything else that scrolls fails. And
# the rail is now required to navigate, so the amendment cannot be spent on
# nothing — a later loop that quietly stops the rail moving fails here rather
# than passing more easily.

def test_the_rail_navigates_when_a_user_presses_it():
    """`DRIVE-047`. The control that already knew the active step now goes to it.

    Driven: a project sealed far enough that Train is revealed, the rail's Train
    step pressed, and the harness asked what moved.
    """
    out = _drive_and_press("clinic_visits", "outcome", press="train")
    assert out["revealed"], (
        "the Train section was never revealed, so pressing its rail step "
        "correctly does nothing and this test asserts nothing")
    assert out["scrolls"], (
        "the rail step was pressed and the viewport did not move — which is "
        "DRIVE-047 exactly, and is what this test was added to prevent "
        "returning")


def test_pressing_a_step_that_has_not_been_reached_moves_nothing():
    """A jump to a hidden element is a jump to nothing.

    The rail renders every step from the first render, so most of them point at
    sections that are not on screen yet. Moving to one would land the reader on
    a collapsed nothing and read as a broken control.
    """
    # **The section is put back into its unreached state before the press.**
    #
    # Driving to a genuinely unreached step and pressing it is what a user does,
    # and it is not what this can assert: the page reveals every section during
    # its own bootstrap on a project that already carries findings, so by the
    # time a body runs there is nothing hidden left to press. Measured, not
    # assumed — the first draft of this test tried `report` on a targeted
    # project and then `train` on an untargeted one, and both were revealed.
    #
    # So the state is restored rather than found. That is weaker than a drive
    # and it is the honest form of what is being checked, which is a guard on
    # one branch: `is-hidden` in, no scroll out.
    out = _drive_and_press("clinic_visits", "outcome", press="train", hide=True)
    assert not out["revealed"], "the section was not hidden before the press"
    assert out["scrolls"] == [], out["scrolls"]


def test_nothing_but_the_rail_may_move_the_viewport():
    """The exemption is one attribute wide, checked on the source.

    `scrollIntoView` is now permitted **once**, inside the `data-map` branch.
    A second call site anywhere is the amendment being spent on something it
    was not argued for.
    """
    page = H.PAGE.read_text(encoding="utf-8")
    script = page[page.index("<script>"):page.rindex("</script>")]
    assert script.count("scrollIntoView") == 1, (
        f"the controller calls scrollIntoView {script.count('scrollIntoView')} "
        f"times; DRIVE-047 argued for exactly one, in the rail's own handler")
    branch = script[script.index('if (t.hasAttribute("data-map"))'):]
    assert "scrollIntoView" in branch[:600], (
        "the one permitted scroll is no longer inside the rail's handler")
