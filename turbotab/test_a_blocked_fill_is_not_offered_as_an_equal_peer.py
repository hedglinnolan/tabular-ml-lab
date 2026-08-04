"""`GUIDED-163` — the constitution refused it and the card offered it anyway.

## What was wrong, and what was NOT wrong

`missingness.blocks('informative', 'impute_median')` has returned `True` since
§07 was written, and `POST route_missingness` with that pairing returns a typed
409 naming the column and counting the blanks. **The server was right.** Driven
before the fix, on `metabolomics_untargeted.csv` with target `bmi`:

    0. indicator              1. indicator_and_impute   2. impute_median
    3. impute_mean            4. impute_mice            5. leave
    6. drop_rows

    POST informative/impute_median -> 409
    POST informative/impute_mean   -> 409
    POST informative/impute_mice   -> 409

Three routes that the constitution refuses, sitting **above** `leave`, which it
permits — in one flat list, under a heading that reads as a list of things you
may do. On the product owner's NHANES drive the column was `meds_hbp`, observed
`{True: 5527, False: 770}` with 15,552 blanks and a median of 1, so *"fill with
the median"* is the option that puts every person of unknown medication status
on blood pressure medication and takes the column to 96.5% ones — with *not
asked* and *yes* encoding identically.

**The shelf is not shortened to fix it.** Every strategy the branch permits is
still offered and still clickable; §09's resolve-or-attest exit is how a user
who knows something the app does not says so. What changes is that the app
**orders** the list and **states its own concern** on the options it is about,
which are the two moves `PRODUCT_VISION.md` names as the alternatives to
deletion.

## What each test here observes

The flag is checked **against the real refusal**, not against itself: for every
option on the card, `blocked_under` containing `informative` and the API
answering 409 to that pairing are asserted to be the same set. A machine-
readable field that agrees with a hand-written list and not with the server is
trap #3 in the payload.

`GUIDED-097`: two fixtures of different target shape — `clinic_visits.csv`
target `outcome` (a string classification outcome) and
`metabolomics_untargeted.csv` target `bmi` (continuous). The shape not covered
is a multiclass target; `multiclass_stage.csv` has no column blank enough to
earn a card at the default threshold.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                              # noqa: E402
from turbotab import missingness as M                                 # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(file, target, column)` — the two target shapes, and the column each claim
#: is driven on. `notes` is the categorical branch and `mz_0022` the numeric
#: one, so both `_FILLERS` families are exercised: `impute_mode` on one side,
#: `impute_median` / `impute_mean` / `impute_mice` on the other.
SHAPES = [
    pytest.param("clinic_visits.csv", "outcome", "notes",
                 id="classification-target"),
    pytest.param("metabolomics_untargeted.csv", "bmi", "mz_0022",
                 id="regression-target"),
]


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _project(client, fixture: str, target: str) -> str:
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    answer = client.post(f"/project/{pid}/decision",
                         json={"kind": "set_target",
                               "payload": {"column": target}})
    assert answer.status_code == 200, (
        f"{fixture} would not take `{target}` as its outcome, so nothing "
        f"below is being driven on the shape it claims")
    return pid


def _card(client, pid: str, column: str):
    body = client.get(f"/project/{pid}/evidence/missingness").json()
    card = next((c for c in body["cards"] if c["column"] == column), None)
    assert card is not None, (
        f"no missingness card for `{column}`, so this test asserts nothing")
    return card


# ═══════════ the order, on the served payload ═══════════════════════════════

@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_every_refusable_fill_is_served_below_every_permitted_one(
        client, fixture, target, column):
    """The ordering half, asserted as a property of the list the server sends.

    Not *"impute_median is at index 3"* — that pins one arrangement and says
    nothing about the rule. The claim is the relation: no option the
    constitution can refuse is served above one it cannot.
    """
    card = _card(client, _project(client, fixture, target), column)
    strategies = [o for o in card["options"] if o.get("is_strategy")]
    assert len(strategies) > 1, "a one-option card cannot show an order"

    refusable = [i for i, o in enumerate(strategies) if o["blocked_under"]]
    permitted = [i for i, o in enumerate(strategies) if not o["blocked_under"]]
    assert refusable and permitted, (
        f"{fixture}:{column} offers no mix of refusable and permitted fills, "
        f"so the order asserts nothing: "
        f"{[(o['key'], o['blocked_under']) for o in strategies]}")
    assert min(refusable) > max(permitted), (
        f"{fixture}:{column} serves a fill the constitution can refuse above "
        f"one it never refuses: "
        f"{[(o['key'], o['blocked_under']) for o in strategies]}")


@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_the_ordering_keeps_the_coach_pick_first_and_shortens_nothing(
        client, fixture, target, column):
    """The shelf is never shortened, and the reorder did not do it by accident.

    Every strategy the record permits on this branch is still offered, and the
    option the coach would take is still the first thing read.
    """
    card = _card(client, _project(client, fixture, target), column)
    offered = [o["key"] for o in card["options"] if o.get("is_strategy")]
    assert set(offered) == set(M.STRATEGIES_BY_BRANCH[card["branch"]]), (
        f"the reorder changed WHICH strategies are offered on "
        f"{fixture}:{column}: {sorted(offered)}")
    assert card["options"][0]["recommended"] is True, (
        "the coach's pick is no longer the first option read")
    assert card["options"][-1]["key"] == "drop_rows", (
        "the one offer that is not a missingness strategy is no longer last")


# ═══════════ the flag, checked against the refusal it stands for ════════════

@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_the_blocked_under_flag_names_exactly_the_pairings_the_api_refuses(
        client, fixture, target, column):
    """Trap #3, at the field level: the flag is verified against the server.

    Each option is POSTed to `route_missingness` with `mechanism=informative`
    on a project of its own — a fresh one per option, because a column answered
    once is answered, and the second POST would be refused for the wrong
    reason. The set that answers 409 and the set whose `blocked_under` carries
    `informative` have to be the same set.
    """
    card = _card(client, _project(client, fixture, target), column)
    flagged, refused = set(), set()
    for option in card["options"]:
        key = option["key"]
        if "informative" in option["blocked_under"]:
            flagged.add(key)
        pid = _project(client, fixture, target)
        answer = client.post(
            f"/project/{pid}/decision",
            json={"kind": "route_missingness", "subject": column,
                  "payload": {"column": column, "mechanism": "informative",
                              "strategy": key, "card_option": key}})
        if answer.status_code == 409:
            refused.add(key)
    assert flagged == refused, (
        f"{fixture}:{column} — the card flags {sorted(flagged)} as refused "
        f"under an informative mechanism and the API refuses {sorted(refused)}")
    assert refused, (
        "nothing 409s on this column, so the agreement above is vacuous")


@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_an_unanswered_mechanism_returns_no_verdict_rather_than_a_false_one(
        client, fixture, target, column):
    """Trap #9 at the field level. `blocked` is `None` before the question is
    answered — not `False`, which would assert the constitution permits a fill
    it may well refuse."""
    card = _card(client, _project(client, fixture, target), column)
    assert card["mechanism"] is None
    for option in card["options"]:
        assert option["blocked"] is None, (
            f"{option['key']} claims blocked={option['blocked']!r} on a card "
            f"whose mechanism question has not been answered")


# ═══════════ the sentence, and what answering the question does to it ═══════

@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_the_concern_names_the_column_and_counts_the_real_blanks(
        client, fixture, target, column):
    """A finding carries its evidence (`DESIGN_LANGUAGE.md` §09), so the
    concern quotes this column's own blank count and not a rounded gesture."""
    card = _card(client, _project(client, fixture, target), column)
    stated = [o for o in card["options"] if o["concern"]]
    assert stated, f"{fixture}:{column} states no concern on any option"
    for option in stated:
        assert f"`{column}`" in option["concern"]
        assert f"{card['n_missing']:,}" in option["concern"], (
            f"{option['key']}'s concern does not count the blanks it is "
            f"about: {option['concern']!r}")
        assert "offered after the choices that keep the signal" in \
            option["concern"]
    for option in card["options"]:
        if option.get("is_strategy") and not option["blocked_under"]:
            assert option["concern"] is None, (
                f"{option['key']} carries a concern the constitution has no "
                f"opinion about — a second, uncalibrated layer of caution")


@pytest.mark.parametrize("fixture,target,column", SHAPES)
def test_answering_the_mechanism_makes_the_conditional_concern_definite(
        client, fixture, target, column):
    """The record reaches the card. Before the answer the app says *this would
    be refused if…*; after it, *you said a blank means something, so this is
    refused* — and `blocked` stops being `None`."""
    pid = _project(client, fixture, target)
    before = _card(client, pid, column)
    conditional = {o["key"]: o["concern"] for o in before["options"]
                   if o["concern"]}
    assert conditional, "nothing to make definite"
    for concern in conditional.values():
        assert concern.startswith("This one is refused if you answer")

    # `indicator`, and named rather than picked off the list: it is on both
    # branches, the constitution permits it under every mechanism, and it
    # leaves the blanks blank — `explicit_category` would answer the question
    # and then remove the column's last blank, so there would be no card left
    # to re-read and the test would pass by disappearing.
    permitted = "indicator"
    assert permitted in [o["key"] for o in before["options"]
                         if o.get("is_strategy") and not o["blocked_under"]]
    answer = client.post(
        f"/project/{pid}/decision",
        json={"kind": "route_missingness", "subject": column,
              "payload": {"column": column, "mechanism": "informative",
                          "strategy": permitted, "card_option": permitted}})
    assert answer.status_code == 200, answer.text

    after = _card(client, pid, column)
    assert after["mechanism"] == "informative", (
        "the card re-asks a question the transcript already holds")
    for option in after["options"]:
        if option["key"] in conditional:
            assert option["blocked"] is True
            assert option["concern"].startswith(
                "You said a blank in"), option["concern"]
        elif option.get("is_strategy"):
            assert option["blocked"] is False
            assert option["concern"] is None


# ═══════════ and it reaches a person ════════════════════════════════════════

@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_page_renders_the_concern_and_the_servers_order_for_the_options(
        client):
    """Trap #6: computed correctly, correct on the wire, invisible to a person.

    The card's strategies do not render until §07's question is answered, so
    the mechanism pill is pressed by its `data-miss-mech-value` and the
    rendered `missBox` is read afterwards. The claim is two things about that
    render: the concern sentence is in it, and the option rows carry the
    server's order — `data-miss-opt="impute_mode"` below every permitted one.

    The page is not edited to make this pass. The concern travels on the
    option's `consequence`, which the option row already renders, because a
    sentence composed for a field nothing reads is the defect this is fixing
    one surface over.
    """
    pid = _project(client, "clinic_visits.csv", "outcome")
    routes = {
        f"/project/{pid}": client.get(f"/project/{pid}").json(),
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness":
            client.get(f"/project/{pid}/evidence/missingness").json(),
        f"/project/{pid}/evidence/plausibility": {"columns": []},
    }
    out = H.run(
        """
        var before = __harness.html('missBox');
        var m = /data-miss-mech-for="(notes)"/.exec(before);
        if (!m) throw new Error('no mechanism question rendered for notes');
        __harness.dispatch('click', __harness.target(
          {'data-miss-mech-for': 'notes', 'data-miss-mech-value': 'informative'},
          ['pill']));
        __emit({after: __harness.html('missBox')});
        """, routes=routes, search=f"?project={pid}")

    rendered = out["after"]
    assert 'data-miss-opt="impute_mode"' in rendered, (
        "the strategies did not come out after the mechanism was answered")
    assert "offered after the choices that keep the signal" in rendered, (
        "the server's concern about the fill it can refuse reaches nobody — "
        "computed, correct on the wire, invisible on the page")
    blocked_at = rendered.index('data-miss-opt="impute_mode"')
    for key in ("explicit_category", "indicator", "indicator_and_impute",
                "leave"):
        assert rendered.index(f'data-miss-opt="{key}"') < blocked_at, (
            f"the page renders {key} below the fill the constitution refuses, "
            f"so the server's order is not what a person reads")
