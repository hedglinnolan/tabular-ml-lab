"""`DRIVE-007`, the half that is cheap — a list you can take to the file.

> 125 entries of `bp_di` are physiologically impossible and the card shows 12.
> The product owner's first instinct was to go back to the CSV and repair it at
> source, which the app makes hard by not supplying the row list.

The card is right to show twelve. A hundred and twenty-five identifiers are not
a reading, and a card that opened with them would trade one unusable card for
another. What was missing is the **way out**: the app was the only thing that
knew which rows, and it kept them.

So every affected row label travels with the block, closed by default, selectable,
with a copy control. Not an export feature — it is the list you paste into a
filter in the file you are about to fix.

## The unclean-feature mark is deliberately not here

`DRIVE-007` names two things. The second — marking a feature unclean so feature
selection can suggest excluding it — is a new mechanic that carries a judgment
across steps, and it pairs with `DRIVE-010`'s working-feature-set question. They
get designed together; half of a cross-step mechanic is worse than none, because
the mark would be recordable and nothing would read it.

## Two assertions that are easy to get wrong

**The count is `n_flagged`, never `len(all_rows)`.** They differ when the server
caps the list, and a summary that counted what it was handed would report
`20,000` for a column with more — the card asserting a completeness the payload
does not have.

**And the list is the WHOLE list**, checked against the count the same card
prints, not against "more than twelve".
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import card_evidence                                          # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _report(name="clinical_longitudinal", target="progressed"):
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return client, pid, client.get(f"/project/{pid}/evidence/plausibility").json()


# ── the payload ──────────────────────────────────────────────────────────────

def test_every_flagged_row_travels_with_the_block_not_only_the_twelve_shown():
    """The defect, at the point it bites.

    `glucose` on this fixture flags 53 rows and the card shows twelve — the
    same shape as the 125-of-`bp_di` the drive reported.
    """
    _, _, rep = _report()
    blocks = rep["impossible"] + rep["improbable"]
    assert blocks, "no plausibility blocks on this fixture"

    truncated = [b for b in blocks if b["truncated"]]
    assert truncated, (
        "no block on this fixture shows fewer entries than it flagged, so this "
        "test cannot tell a full list from a short one")

    for b in blocks:
        assert "all_rows" in b, f"{b['column']} carries no row list"
        if not b["all_rows_truncated"]:
            assert len(b["all_rows"]) == b["n_flagged"], (
                f"{b['column']}: the row list holds {len(b['all_rows'])} of "
                f"{b['n_flagged']} flagged rows and does not say it is short")


def test_the_row_list_is_the_rows_and_not_the_positions():
    """Row LABELS, which is what row identity is in this project.

    `TRANSITION_PLAN.md` §02.2 settles identity as index labels, and a list of
    positions would be unusable against a file whose rows were dropped or
    reordered — the copy would point at the wrong lines and look right.
    """
    client, pid, rep = _report()
    block = next(b for b in rep["improbable"] if b["n_flagged"] > 12)
    shown = {e["row"] for e in block["entries"]}
    assert shown <= set(block["all_rows"]), (
        "the rows the card displays are not in the list it hands out, so the "
        "two disagree about which rows are affected")


def test_the_cap_is_declared_rather_than_silent():
    """*"Every affected row"* is a claim.

    A list that quietly stopped at the cap would be the card asserting a
    completeness it does not have — which is the governing rule's own failure
    in an export affordance.
    """
    import numpy as np
    import pandas as pd

    n = card_evidence.MAX_ROW_LIST + 50
    frame = pd.DataFrame({"sbp": np.full(n, 9999.0)})
    hit = frame["sbp"]
    block = card_evidence._entry_block(
        "sbp", "systolic_bp", hit, hit, hit, 40.0, 300.0, "mmHg",
        tier="impossible")
    assert block["n_flagged"] == n, "the COUNT must stay exact"
    assert len(block["all_rows"]) == card_evidence.MAX_ROW_LIST
    assert block["all_rows_truncated"] is True, (
        "the list was capped and does not say so")


# ── what the driver sees ─────────────────────────────────────────────────────

@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_card_offers_the_whole_list_and_counts_it_honestly():
    """Read back off the render.

    The summary says 53 because `n_flagged` says 53 — not because the list
    happens to be that long today.
    """
    client, pid, rep = _report()
    project = client.get(f"/project/{pid}").json()
    slot = next(f["id"] for f in project["findings"]
                if (f.get("params") or {}).get("category")
                == "physiologic_plausibility")
    html = H.run("__emit(__harness.html('ev-" + slot + "'));", routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": rep,
    }, search=f"?project={pid}")

    assert "data-copy-rows=" in html, (
        "the card offers no way to take the row list out of the app")

    big = max(rep["impossible"] + rep["improbable"],
              key=lambda b: b["n_flagged"])
    assert f"All {big['n_flagged']:,} affected rows" in html, (
        f"the summary does not carry the real count for {big['column']}")

    # The whole list is in the box, not the twelve the table shows.
    boxes = re.findall(r"<textarea[^>]*>([^<]*)</textarea>", html)
    assert boxes, "no selectable row list rendered"
    longest = max(boxes, key=len).strip().split("\n")
    assert len(longest) == big["n_flagged"], (
        f"the box holds {len(longest)} rows where {big['n_flagged']} are "
        f"flagged — the card is still handing out a truncation")


@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_a_capped_list_reports_the_real_count_and_says_it_is_short():
    """The case no fixture produces, and the reason the count is `n_flagged`.

    A revert probe found this: with `n = rows.length` instead of
    `block.n_flagged`, every assertion above stayed green, because on real data
    nothing is capped and the two numbers agree. The distinction only bites
    where the server truncated — and that is exactly where the card would
    otherwise report `12` for a column with 4,000, which is the card asserting
    a completeness the payload does not have.

    So the capped block is handed to the renderer directly. It is a shape the
    server really produces; no fixture in this tree is large enough to make it.
    """
    client, pid, _ = _report()
    project = client.get(f"/project/{pid}").json()
    slot = next(f["id"] for f in project["findings"]
                if (f.get("params") or {}).get("category")
                == "physiologic_plausibility")
    capped = {"impossible": [{
        "column": "sbp", "n_flagged": 4321, "low": 40.0, "high": 300.0,
        "unit": "mmHg", "truncated": True, "whole_column_suspect": False,
        "entries": [{"row": i, "value": 9999.0, "side": "above", "bound": 300.0}
                    for i in range(12)],
        "all_rows": [str(i) for i in range(12)],
        "all_rows_truncated": True}], "improbable": []}

    html = H.run("__emit(__harness.html('ev-" + slot + "'));", routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": capped,
    }, search=f"?project={pid}")

    assert "All 4,321 affected rows" in html, (
        "the summary counted the list it was handed rather than the rows that "
        "are flagged, so a capped payload reads as a complete one")
    assert "Showing the first 12 of 4,321" in html, (
        "the card hands out a truncated list without saying it is truncated")


@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_list_is_closed_until_it_is_asked_for():
    """The card's job is the reading.

    Fifty-three identifiers rendered open would trade one unusable card for
    another, which is the `DRIVE-005` complaint arriving inside a single card.
    """
    client, pid, rep = _report()
    project = client.get(f"/project/{pid}").json()
    slot = next(f["id"] for f in project["findings"]
                if (f.get("params") or {}).get("category")
                == "physiologic_plausibility")
    html = H.run("__emit(__harness.html('ev-" + slot + "'));", routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": rep,
    }, search=f"?project={pid}")
    assert "<details class=\"rowlist\"" in html
    assert "<details class=\"rowlist\" open" not in html, (
        "the row list is open by default, so every card now opens with a wall "
        "of identifiers")
