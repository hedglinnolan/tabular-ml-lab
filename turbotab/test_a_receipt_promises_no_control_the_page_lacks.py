"""`DRIVE-036`'s false promise, pulled — and kept out.

**Two human drives read the same sentence and went looking for the same
control.** The *what one row is* receipt, on `people repeat` with no identifier
named, ended: *"Your numbers are labeled exploratory until a person column is
named, **and you can name it at any point before the seal.**"*

There is no such control. The identifier follow-up is declared in the API
contract — `GET /grain` carries `follow_up: "which column identifies the
person?"` — and rendered nowhere, so a reader who believes the receipt finds
only the aggregation menu's refusal: *"there is no identifier column recorded,
so there is nothing to combine rows by."*

That is the governing rule's **assert something false** branch wearing an offer.
It is worse than silence rather than milder, because silence costs a reader
nothing and a promise costs them the session.

**What this file does and does not claim.** It does not assert that the control
should exist — building it is the other half of `DRIVE-036` and it stays open.
It asserts the narrow thing that is true today: **the receipt does not promise
an action the page cannot perform.** If somebody builds the control, the
promise becomes true and this file says exactly which assertion to change.

The positive control matters here more than usual. Every assertion below is an
absence, and the sentence being absent from a receipt nobody renders would
satisfy all of them.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import pathlib


from turbotab import grain

ROOT = pathlib.Path(__file__).resolve().parent

#: The clause that was pulled. Matched loosely on purpose — a rewording that
#: still promises the action is the same defect.
PROMISED = ("at any point before the seal",
            "you can name it at any")

#: The receipts this file is about: the two `people repeat` / `design not
#: described` branches where NO column was named. They are the ones whose
#: subject is a control that does not exist.
UNGROUPED_RECEIPTS = (grain._PEOPLE_REPEAT_UNGROUPED,
                      grain._DESIGN_NOT_DESCRIBED_UNGROUPED)


def _receipt(key: str) -> str:
    return grain._ANSWERED[key]


def test_the_ungrouped_receipt_promises_no_naming_control():
    """The rule. No receipt whose condition is *no identifier named* may offer
    the act of naming one, while nothing on the page performs it."""
    offenders = []
    for key in UNGROUPED_RECEIPTS:
        text = _receipt(key).lower()
        for phrase in PROMISED:
            if phrase in text:
                offenders.append((key, phrase))
    assert not offenders, (
        f"{offenders} — the receipt offers naming a person column and no "
        f"control on the page does it. `GET /grain` declares the follow-up "
        f"'which column identifies the person?' and nothing renders it, so a "
        f"reader who believes this sentence spends the session hunting. Build "
        f"the control (DRIVE-036's open half) and this assertion is the one to "
        f"change.")


def test_the_receipt_still_says_what_is_true_of_the_split():
    """**The other half, and it is why this is a pull rather than a deletion.**

    The removed clause was one sentence-ending, not the paragraph. What must
    survive is the honest content: the split is by row, the same person can sit
    on both sides, the numbers are exploratory, and the condition that lifts it
    is named. A receipt trimmed until it promises nothing would also say
    nothing, and silence about a split that puts one person on both sides is
    not the safe direction.
    """
    text = _receipt(grain._PEOPLE_REPEAT_UNGROUPED)
    for owed in ("BY ROW",
                 "both sides",
                 "exploratory",
                 "until a person column is named"):
        assert owed in text, (
            f"the receipt no longer says {owed!r}; pulling the promise was "
            f"meant to remove a claim about a CONTROL, not a claim about the "
            f"SPLIT")


def test_the_matcher_would_see_the_promise_if_it_came_back():
    """The negative control's control. A matcher that fires on nothing has
    silence that means nothing, and every phrase above is an absence claim."""
    revived = ("Your numbers are labeled exploratory until a person column is "
               "named, and you can name it at any point before the seal.")
    assert any(phrase in revived.lower() for phrase in PROMISED), (
        "the matcher no longer recognizes the exact sentence that was pulled, "
        "so its silence about the current receipt means nothing")


