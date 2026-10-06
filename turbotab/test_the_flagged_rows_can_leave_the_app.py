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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import card_evidence                                          # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


# ── the payload ──────────────────────────────────────────────────────────────


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
