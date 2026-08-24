"""`DRIVE-005` — ten flags, ten cards, accidental infinite scroll.

The product owner's words are the specification:

> Ten features with improbable values produced ten dedicated cards with no
> action button, because there is no action to take — the finding is for
> awareness. Aggregate cards carrying the same kind of information across
> features into one card with horizontal paging, rather than stacking them
> vertically. **A finding with no action does not earn a card of its own.**

## The line, and why it is not severity

The impossible blocks in the same panel keep their own cards, and they are not
kept because they are worse. They carry a **decision** — *set these entries to
missing* / *keep as is* — so each needs its own space and produces its own
recorded answer. The improbable blocks carry a **fact**: *"advisory; a value can
be unusual and correct."* Facts of the same kind across many columns are one
fact with a column axis, and a column axis is a pager.

So the test asserts both halves. Collapsing the ones with actions would be the
same defect inverted — a decision hidden behind a pager is a decision the user
has to go looking for.

## What is asserted, and where from

Off the render, through the page's own controller and its own click handler, on
a fixture that really produces five of each. `clinical_longitudinal.csv` is that
fixture; the drive that reported this had ten.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

pytestmark = pytest.mark.skipif(
    not H.available(), reason="no JS engine on this machine")


def _plausibility_render(name="clinical_longitudinal", target="progressed",
                         after="", report=None):
    """Boot the page on a real project and read back the plausibility panel.

    `report` overrides what the server would return. Used for exactly one case —
    the single-member card — because no fixture in the tree produces both a
    physiologic-plausibility finding to hang the panel on AND exactly one
    improbable column, and the renderer's response to a one-member report is
    still a real contract the server can hand it.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    project = client.get(f"/project/{pid}").json()
    rep = report if report is not None else client.get(
        f"/project/{pid}/evidence/plausibility").json()
    slot = next((f["id"] for f in project["findings"]
                 if (f.get("params") or {}).get("category")
                 == "physiologic_plausibility"), None)
    assert slot, "no physiologic-plausibility finding on this fixture"

    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": rep,
        f"/project/{pid}/draft": {"paragraphs": []},
        f"/project/{pid}/gaps": {"gaps": []},
    }
    body = (after +
            "__emit({html: __harness.html('ev-" + slot + "'), "
            "n_improbable: " + str(len(rep.get("improbable", []))) + ", "
            "n_impossible: " + str(len(rep.get("impossible", []))) + "});")
    return H.run(body, routes=routes, search=f"?project={pid}")


def test_many_improbable_features_become_one_card_with_a_pager():
    """The defect, as the property it broke."""
    out = _plausibility_render()
    assert out["n_improbable"] >= 3, (
        f"only {out['n_improbable']} improbable features on this fixture; the "
        f"test is about what happens when there are many")
    html = out["html"]

    cards = re.findall(r'<div class="ev paged" data-pg="improbable"', html)
    assert len(cards) == 1, (
        f"expected one aggregate card, found {len(cards)}. Ten flags becoming "
        f"ten cards is the finding.")
    pages = re.findall(r'data-pg-page="improbable"', html)
    assert len(pages) == out["n_improbable"], (
        f"{len(pages)} pages for {out['n_improbable']} features — the card is "
        f"dropping some, which is worse than stacking them")
    assert 'data-pg-step="1"' in html and 'data-pg-step="-1"' in html, (
        "the aggregate card has no pager, so all but the first feature are "
        "unreachable")


def test_the_card_says_how_many_before_the_user_decides_to_look():
    """Nothing is hidden, only unstacked.

    Ten stacked cards told you their count by making you scroll to the bottom.
    One card that does not say `10` has replaced that with a worse problem.
    """
    out = _plausibility_render()
    head = out["html"][:out["html"].index("data-pg-page")]
    assert str(out["n_improbable"]) in head, (
        f"the card does not say how many features it covers: {head[-200:]!r}")
    assert re.search(r'data-pg-at="improbable"[^>]*>1 / %d<' % out["n_improbable"],
                     out["html"]), "the page indicator does not show the total"


def test_the_findings_that_carry_a_decision_keep_their_own_cards():
    """The other half, and the one a fix could plausibly get wrong.

    The impossible blocks are not kept separate because they are worse. They
    carry a decision the user has to make, and a decision behind a pager is a
    decision the user has to go looking for.
    """
    out = _plausibility_render()
    assert out["n_impossible"] >= 2, (
        "this fixture has fewer than two impossible columns, so the assertion "
        "below cannot distinguish 'kept separate' from 'there was only one'")
    applies = re.findall(r'data-impossible="([^"]+)"', out["html"])
    assert len(applies) == out["n_impossible"], (
        f"{len(applies)} apply controls for {out['n_impossible']} impossible "
        f"columns — a decision was collapsed into the pager")
    assert 'data-pg="impossible"' not in out["html"]


def test_the_pager_moves_and_wraps():
    """Read back off the page's own click handler.

    Wrapping rather than stopping at the ends: a set of ten facts has no first
    or last one, and an arrow that dies at the edge is a control that stops
    working for a reason the user has to infer.
    """
    def press(times, step="1"):
        js = ("".join(
            "__harness.dispatch('click', __harness.target("
            "{'data-pg-step': '%s', 'data-pg-for': 'improbable'}, ['pg-btn']));"
            % step for _ in range(times)))
        return _plausibility_render(after=js)

    first = _plausibility_render()
    n = first["n_improbable"]
    shown = re.search(r'data-pg-at="improbable"[^>]*>([^<]+)<', first["html"])
    assert shown.group(1).strip() == f"1 / {n}"

    second = press(1)
    at = re.search(r'data-pg-at="improbable"[^>]*>([^<]+)<', second["html"])
    assert at.group(1).strip() == f"2 / {n}", (
        f"the pager did not move: {at.group(1)!r}")
    # The pane that is visible is the second one, not merely the label.
    visible = [i for i, m in enumerate(
        re.finditer(r'class="pg-page( is-hidden)?" data-pg-page="improbable"',
                    second["html"]))
        if not m.group(1)]
    assert visible == [1], (
        f"the label moved and the panes did not: visible panes {visible}")

    wrapped = press(1, step="-1")
    at = re.search(r'data-pg-at="improbable"[^>]*>([^<]+)<', wrapped["html"])
    assert at.group(1).strip() == f"{n} / {n}", (
        f"stepping back from the first page did not wrap: {at.group(1)!r}")


def test_one_member_is_not_a_pager():
    """A single improbable column renders as the plain block it always was.

    Wrapping one thing in navigation asserts a set where there is not one, which
    is §08's rule — structural devices must encode something true — applied to a
    control rather than to a number.
    """
    one = {"impossible": [], "improbable": [{
        "column": "glucose", "n_flagged": 3, "low": 3.0, "high": 20.0,
        "unit": "mmol/L", "truncated": False,
        "entries": [{"row": 1, "value": 41.0, "side": "above", "bound": 20.0}]}]}
    out = _plausibility_render(report=one)
    assert out["n_improbable"] == 1
    assert 'data-pg="improbable"' not in out["html"]
    assert "data-pg-step" not in out["html"], (
        "one thing was wrapped in navigation, which asserts a set where there "
        "is not one")
    assert "Improbable values — glucose" in out["html"]
