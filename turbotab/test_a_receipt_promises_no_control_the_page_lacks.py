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
from __future__ import annotations

import pathlib

import pytest

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


def test_no_control_names_the_person_column_yet():
    """**The positive control, and it is the load-bearing one.**

    The assertion above is only meaningful while the control is genuinely
    absent — if somebody builds it, the receipt SHOULD promise it, and a guard
    that still forbade the sentence would be pinning the defect in place
    (`AGENT_ONBOARD.md` trap #3c).

    So the absence is checked rather than assumed, and this test is the thing
    that fails first when `DRIVE-036`'s open half lands. Read as an
    instruction: when it goes red, delete it and invert the assertion above.
    """
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    # The page's own convention for a control that submits an answer to a
    # follow-up is a `data-` attribute naming it. Two spellings, because the
    # identifier follow-up has no shipped name yet and either would do.
    for marker in ('data-person-col', 'data-identifier-for'):
        assert marker not in page, (
            f"{marker} is on the page — the naming control may exist now. If "
            f"it does, `DRIVE-036`'s open half has landed: the receipt may "
            f"promise it again, and this file's first assertion should be "
            f"inverted rather than left forbidding a true sentence.")


def test_the_matcher_would_see_the_promise_if_it_came_back():
    """The negative control's control. A matcher that fires on nothing has
    silence that means nothing, and every phrase above is an absence claim."""
    revived = ("Your numbers are labeled exploratory until a person column is "
               "named, and you can name it at any point before the seal.")
    assert any(phrase in revived.lower() for phrase in PROMISED), (
        "the matcher no longer recognizes the exact sentence that was pulled, "
        "so its silence about the current receipt means nothing")


@pytest.mark.parametrize("key", sorted(grain._ANSWERED))
def test_no_grain_receipt_promises_an_action_with_no_verb_behind_it(key):
    """The same lens, one surface over — `AGENT_ONBOARD.md` §08 item 5.

    The pulled clause was one instance of *a receipt offering an action*. The
    other receipts make offers too, and two of them survive this check for a
    real reason rather than by luck: *'you can settle it at any point before
    training'* on `not_sure`, and the `design_not_described` pair. Re-answering
    a grain question is a `revise` exit, and `revise` is the one verb the page
    implements (`index.html`, the exits block) — so those promises are backed.
    Asserted here so the distinction is checked rather than remembered.
    """
    text = grain._ANSWERED[key].lower()
    if "you can name it" not in text:
        return
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert '"revise"' in page, (
        f"{key} offers an action and the page implements no verb for it")
