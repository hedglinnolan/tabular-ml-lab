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
