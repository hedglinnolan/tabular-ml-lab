"""`GUIDED-090` / `GUIDED-091` / `GUIDED-098` — one question, two doors, three
disagreements.

The Explore card and the Preprocess panel both ask what to do about a blank.
Both surfaces are legitimate and the vocabularies were never the defect —
`PRODUCT_VISION.md` §04 already names their relationship: deferral is a
first-class disposition and a noticing resurfaces at the step it targets. What
they may not do is answer the same question differently.

They did, three times.

**`GUIDED-090` — the shelf was shortened.** Measured on `clinic_visits.csv`, a
numeric column: the Explore card offered four strategies, `/preprocess` offered
five, and the missing one was `leave`. `_numeric_options` never emitted it while
`_binary_options` did, and `CARD_STRATEGY` already knew it. That is the product
owner's own ruling at a surface nobody had compared — *judgment renders as
ranking, never as absence* — and it was missing on exactly the column where the
absence carries signal.

**`GUIDED-091` — the mechanism was never asked.** The card carried no
`mechanism` field at all, so the page's `c.mechanism || "not_sure"` was
unconditional and every column routed from that door recorded `not_sure`.
`blocks()` fires only on `informative`, so **§07's blocker was unreachable from
that door by any user on any column** — not bypassed, unreachable.

**`GUIDED-098`, found while fixing those two — one click, two methods
sentences.** The card's `indicator_and_impute` promised *"imputed with the
training-fold median and a missingness indicator was retained"* and
`CARD_STRATEGY` mapped it to `INDICATOR`, whose recorded sentence is *"the
underlying value is left blank."* The binary branch's `indicator` did the same.
After `GUIDED-095` the pipeline honors the record, so the fill the card promised
does not happen — the contradiction stopped being about prose and became about
the fit.

## The fix, in one sentence each

**One table decides what both doors offer.** `_options_for` iterates
`missingness.STRATEGIES_BY_BRANCH`; `GUIDED-086` made the CHECK read it and this
is the half that OFFERS. **One composer writes the sentence.**
`missingness.sentence_for` is asked by the card and by `declare`. **The card
asks the mechanism**, with the same copy and the same order Preprocess uses.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import missingness_plan as MP                                 # noqa: E402
from turbotab import missingness as M                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: Enough shapes that a branch cannot be right by accident: a continuous
#: column, a 0/1 numeric one (which the two doors used to route DIFFERENTLY),
#: and a text one.
FIXTURES = ("clinic_visits.csv", "survey_instrument.csv",
            "metabolomics_untargeted.csv")


# ── one table decides what both doors offer ──────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_both_doors_offer_the_same_strategies_for_the_same_column(fixture):
    """`GUIDED-090`. The offer, not the check.

    Every strategy the Explore card offers must be one the record permits on
    that column's branch, AND every strategy the branch permits must be
    offered. A shorter shelf on one door is the app making a decision in the
    user's name at the surface where they are looking.
    """
    # `threshold=0.0`: this claim is about what the card OFFERS, not about
    # which columns are notable enough to earn one, so every column with a
    # blank is exercised rather than only the loud ones.
    frame = pd.read_csv(DATA / fixture)
    cards = MP.missingness_cards(frame, threshold=0.0)
    assert cards, f"{fixture} produced no card, so this asserts nothing"
    for card in cards:
        permitted = set(M.STRATEGIES_BY_BRANCH[card["branch"]])
        offered = {o["key"] for o in card["options"] if o.get("is_strategy")}
        assert offered == permitted, (
            f"{fixture}:{card['column']} ({card['branch']}) — the card offers "
            f"{sorted(offered)} and the record permits {sorted(permitted)}; "
            f"missing from the card: {sorted(permitted - offered)}")


def test_leave_is_on_the_numeric_card_where_the_absence_carries_signal():
    """The instance, named, because a set comparison passing does not say
    WHICH option came back."""
    frame = pd.read_csv(DATA / "clinic_visits.csv")
    numeric = [c for c in MP.missingness_cards(frame)
               if c["branch"] == "numeric"]
    assert numeric, "clinic_visits has no numeric card"
    for card in numeric:
        keys = {o["key"] for o in card["options"]}
        assert "leave" in keys, (
            f"{card['column']} cannot be left alone from the Explore door, and "
            "it is the option that matters most where a blank means something")


def test_the_card_and_the_record_route_a_column_to_the_same_branch():
    """The other half of the same divergence, and it was invisible.

    `_kind_of` called a 0/1 numeric column `binary` and offered `impute_mode`;
    `declare` calls it `numeric` by dtype and REFUSES `impute_mode` there
    (`GUIDED-086`), so the card offered a route the record would reject. The
    card keeps its three-way `dtype_route` for how it words the question and
    routes the OFFER on the record's branch.
    """
    frame = pd.DataFrame({
        "zero_one": [1, 0, None, 1, None, 0, 1, None, 0, 1] * 4,
        "text": ["a", "b", None, "c", None, "a", "b", None, "c", "a"] * 4,
    })
    cards = {c["column"]: c for c in MP.missingness_cards(frame)}
    assert cards["zero_one"]["dtype_route"] == "binary"
    assert cards["zero_one"]["branch"] == "numeric", (
        "the card routes a 0/1 numeric column to a branch the record does not")
    for card in cards.values():
        for option in card["options"]:
            if not option.get("is_strategy"):
                continue
            strategy = M.strategy_for_card_option(option["key"])
            M.declare(card["column"], card["branch"], "not_sure", strategy)


def test_the_one_option_that_is_not_a_strategy_is_still_offered_with_its_reason():
    """**Do not settle this by deleting the card.** `drop_rows` is an
    eligibility criterion wearing a missingness costume, and the right answer
    is the argument plus somewhere to go — a gap that becomes routing is worth
    more than a transform."""
    frame = pd.read_csv(DATA / "clinic_visits.csv")
    for card in MP.missingness_cards(frame):
        drop = [o for o in card["options"] if o["key"] == "drop_rows"]
        assert drop, f"{card['column']} offers no route for dropping the rows"
        assert drop[0]["is_strategy"] is False
        assert "participant flow" in drop[0]["consequence"]
        with pytest.raises(M.MissingnessRefusal):
            M.strategy_for_card_option("drop_rows")


# ── one composer writes the sentence ─────────────────────────────────────────

@pytest.mark.parametrize("scope", [M.TRAIN_ROWS, M.TRAIN_FOLDS],
                         ids=["the guided door, fitted once", "classic, per fold"])
@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_card_and_the_record_write_the_same_sentence(fixture, scope):
    """`GUIDED-098`, and it is asserted as EQUALITY of the served strings
    rather than as identity, because the card is composed on one request and
    the record on another — two processes, one composer.

    The pairing that was wrong is the one that matters: an option promising a
    training-fold median mapped to a declaration whose sentence says the value
    is left blank. One click, two methods sentences, opposite claims.

    **Parametrized over the fitting scope by `AUDIT-028`, and that is a
    widening rather than an accommodation.** The row's fix gave the two doors
    different scopes — the guided door fits once on the training rows
    (`turbotab/training.py:416`: nothing under `turbotab/` imports `KFold`,
    `cross_val_score` or `cross_validate`), Classic re-fits per fold when
    `pages/06`'s CV box is ticked. The first merge of that fix moved only the
    RECORD and left the card pinned at `TRAIN_FOLDS`, so this gate went red on
    a real defect: the guided door showed a fold sentence on the button and
    wrote a rows sentence into the transcript. Comparing at a fixed scope would
    have hidden that. Comparing at BOTH scopes says the composer is one
    composer *whatever scope it is asked for*, which is the claim `GUIDED-098`
    was always making.
    """
    frame = pd.read_csv(DATA / fixture)
    for card in MP.missingness_cards(frame, threshold=0.0, scope=scope):
        for option in card["options"]:
            if not option.get("is_strategy"):
                continue
            strategy = M.strategy_for_card_option(option["key"])
            recorded = M.declare(card["column"], card["branch"], "not_sure",
                                 strategy, scope=scope)
            assert option["decision_sentence"] == recorded["sentence"], (
                f"{fixture}:{card['column']} option {option['key']!r} at "
                f"scope {scope!r} — the card says "
                f"{option['decision_sentence']!r} and the record says "
                f"{recorded['sentence']!r}")
