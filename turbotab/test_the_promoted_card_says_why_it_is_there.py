"""L46-D — clearing a card frees its slot, and the card that fills it is marked.

`GUIDED-154`, ruled by the product owner after he opened L45's prototype. A
dismissed or deferred finding stops consuming the bound's budget and the
highest-ranked collapsed finding is promoted into the vacancy, so the stack keeps
its full budget of **live** findings rather than its full count of cards.

## The failure mode the ruling creates, which is why this file exists

A card the user did not ask for appears exactly where they cleared one, and the
honest reading of that is *my dismissal did not work*. The answer is not motion —
`DESIGN_LANGUAGE.md` §05.2's list stays closed at four, the app has no mechanism
for animating a change of content, and a fifth slot pulls in `GUIDED-073`. So the
promoted card is **marked where it stands**, and §09's recorded-absence rule from
the other side is the argument: an object appearing with no explanation is as
unexplained as one vanishing without it.

**That makes this arithmetic before it is design**, and the arithmetic is what is
asserted here.

## The ledger, and why it is stated in a different form than the prompt asked

The loop prompt asks for `rendered + collapsed + dismissed = served`. Taken
literally that double-counts, because a dismissed card is still **rendered** —
it collapses to a `.gone` card and its *"Still in the record, out of your way"*
undo note, which is the shelf not being shortened. The disjoint form of the same
ledger is

    live + cleared + collapsed == served

and both halves of `pushed` are served on the payload so a reader can check it
without recomputing which ids were cleared. Asserted at every size below.

## `GUIDED-097` — two lenses, and what neither reaches

Clinical and metabolomics. `SHAPES_NOT_COVERED` names what neither drives.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from turbotab import attention as A

DATA = Path(__file__).resolve().parent / "sample_data"

#: Two lenses of deliberately different stack shape. Clinical has one gating
#: finding and a long tail; metabolomics has four gating findings and almost no
#: tail, which is the case where clearing a card can promote nothing because
#: there is nothing behind the affordance.
#: `(fixture, lens, target, bound)`. The bound is stated per lens because
#: `MIN_COLLAPSE` leaves `metabolomics_untargeted.csv` with no remainder at the
#: shipping bound, so metabolomics is driven at 2 here — a bound the module
#: supports and the prototype compares — and the second lens exercises the rule
#: rather than skipping past it. A probe that ran only where the default happens
#: to bite is `GUIDED-097`'s one-fixture failure with the parameter moved.
#:
#: **THIS COMMENT USED TO SAY `MIN_COLLAPSE` LEFT *EXACTLY ONE* FIXTURE IN THIS
#: REPOSITORY WITH A REMAINDER AT THE SHIPPING BOUND, AND THAT WAS FALSE.**
#: Swept through the API at `A.BOUND = 5`, `metabolomics_merged_modes.csv` has a
#: four-card remainder under the metabolomics lens (`live=8, collapsed=4`). The
#: claim was never checked by anything, which is how it survived — and it is why
#: `PAGE_LENSES` below can drive the page-level claim on two lenses at the
#: shipping bound instead of skipping one of them (`AUDIT-039`).
#:
#: These two are kept as they are: the tests that read `LENSES` are about the
#: PARTITION rather than about the page, and they call `A.stack(..., bound=…)`
#: directly, where a bound of 2 is a real and supported case.
LENSES = {
    "clinical": ("clinical_labs.csv", "clinical", "readmitted", None),
    "metabolomics": ("metabolomics_untargeted.csv", "metabolomics", "responder", 2),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "Undoing a dismissal. `undismiss` restores the finding to the live set and "
    "`spent_ids` reads it, but no drive here presses undo, so the promotion "
    "does not get checked in reverse.",
    "A deferral cleared from inside the collapsed group. The page only renders "
    "the group's cards once it is open, and the drives below clear from the "
    "pushed list, which is where a user actually is.",
    "The three other packs. Two lenses, per `GUIDED-097`, and dietary, survey "
    "and genomics are not driven here.",
)


def _driven(fixture: str, lens, target, bound=None):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    if lens:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": [lens]}})
    if target:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


@pytest.fixture(scope="module")
def projects():
    return {label: _driven(*spec) for label, spec in LENSES.items()}


def _ledger(st: Dict[str, Any]) -> None:
    """The disjoint form, asserted the same way at every call site."""
    assert len(st["pushed"]) + len(st["collapsed"]) == st["served"], st
    assert len(st["live"]) + len(st["cleared"]) + len(st["collapsed"]) == st["served"], (
        f"{len(st['live'])} live + {len(st['cleared'])} cleared + "
        f"{len(st['collapsed'])} collapsed != {st['served']} served")
    assert st["remainder"]["n"] == len(st["collapsed"]), st["remainder"]


# ── the partition, driven against the real record ───────────────────────────

@pytest.mark.parametrize("label", sorted(LENSES))
def test_clearing_a_card_frees_its_slot_and_the_ledger_still_balances(
        projects, label):
    """Dismiss down the pushed list one card at a time, to exhaustion.

    Driven against the real record for the shipping bound and against
    `attention.stack` for the other, because the API has no way to pass one — the
    findings are the same real driven findings either way.
    """
    fixture, lens, target, bound = LENSES[label]
    client, pid = _driven(fixture, lens, target)
    project = client.get(f"/project/{pid}").json()
    findings = project["findings"]
    spent: Dict[str, str] = {}

    def current():
        if bound is None:
            return client.get(f"/project/{pid}").json()["explore_stack"]
        return A.stack(findings, bound=bound, spent=spent)

    def clear(fid: str):
        if bound is None:
            client.post(f"/project/{pid}/decision",
                        json={"kind": "dismiss", "subject": fid})
        else:
            spent[fid] = "dismiss"

    st = current()
    _ledger(st)
    live_at_start = len(st["live"])
    dismissals = 0

    while st["collapsed"]:
        # Always the first LIVE, non-gating card — a gating finding is never
        # collapsed so clearing one frees a slot that was never bounded.
        victim = next((i for i in st["live"]
                       if not A.gates_a_decision(
                           next(f for f in findings if f["id"] == i))), None)
        if victim is None:
            break
        before = list(st["collapsed"])
        clear(victim)
        st = current()
        dismissals += 1
        _ledger(st)

        # THE PROMOTED SET IS A PREFIX OF WHAT WAS BEHIND THE AFFORDANCE, in
        # rank order. A prefix rather than "the single next one" because
        # `MIN_COLLAPSE` means a remainder that would drop to one is shown
        # instead — so one dismissal can promote two, and that is the two
        # rulings interacting rather than a bug.
        fresh = [i for i in st["pushed"] if i in before]
        assert fresh == before[:len(fresh)], (
            f"{label}: promoted {fresh}, which is not a prefix of {before}")
        assert set(st["promoted"]) >= set(fresh), (
            f"{label}: {sorted(set(fresh) - set(st['promoted']))} arrived and "
            f"is not marked as promoted")
        # THE LIVE BUDGET IS KEPT WHILE THERE IS ANYTHING TO KEEP IT WITH. Once
        # the remainder empties the count can exceed the budget, because
        # `MIN_COLLAPSE` shows the last card rather than hiding it alone — so
        # the invariant is *never shrinks*, and *stays put* only while something
        # is still behind the affordance.
        assert len(st["live"]) >= live_at_start, (
            f"{label}: live SHRANK {live_at_start} → {len(st['live'])} after "
            f"{dismissals} dismissals")
        if st["collapsed"]:
            assert len(st["live"]) == live_at_start, (
                f"{label}: live went {live_at_start} → {len(st['live'])} after "
                f"{dismissals} dismissals with {len(st['collapsed'])} still "
                f"behind the affordance")

    assert dismissals, (
        f"{label}: nothing was ever collapsed at bound "
        f"{bound if bound is not None else A.BOUND}, so nothing was driven")
    # AND THE END STATE. Nothing behind the affordance, so the slot says so.
    assert st["complete"], st
    assert not st["promoted_because"] or st["promoted"], (
        "a promotion sentence with nothing promoted")


@pytest.mark.parametrize("label", sorted(LENSES))
def test_the_affordance_disappears_rather_than_saying_zero(projects, label):
    """The case that would actually break it.

    An affordance reading *"0 more"* is a control promising something it does not
    have, and `complete` is what stops it: the slot switches to the
    recorded-absence sentence instead of counting to nothing.
    """
    fixture, lens, target, bound = LENSES[label]
    client, pid = _driven(fixture, lens, target)
    findings = client.get(f"/project/{pid}").json()["findings"]
    spent = {}
    st = A.stack(findings, bound=bound, spent=spent)
    while st["collapsed"]:
        victim = next((i for i in st["live"]
                       if not A.gates_a_decision(
                           next(f for f in findings if f["id"] == i))), None)
        if victim is None:
            break
        spent[victim] = "dismiss"
        st = A.stack(findings, bound=bound, spent=spent)
        assert "0 more" not in st["affordance"], (
            f"{label}: the affordance is counting to nothing: "
            f"{st['affordance']!r}")
        if st["complete"]:
            assert st["affordance"].startswith("All "), st["affordance"]
            assert not st["affordance_open"], (
                f"{label}: a complete stack still offers an expand")
            assert not st["affordance_detail"], st["affordance_detail"]
    assert st["complete"], f"{label}: the remainder never emptied"


def test_nothing_that_gates_a_decision_is_ever_promoted_late():
    """Structural, and worth stating as its own claim.

    A finding that gates a decision is never collapsed, so it can never be behind
    the affordance, so it can never arrive late. The assertion is over the
    partition rather than over one drive, because the property is about the rule.
    """
    gating = [{"id": f"g{i}", "severity": "critical", "source": "pack",
               "pack": "clinical", "rank": i} for i in range(3)]
    ordinary = [{"id": f"o{i}", "severity": "warning", "source": "profile",
                 "rank": 10 + i} for i in range(9)]
    findings = gating + ordinary
    for cleared in ([], ["o0"], ["o0", "o1"], ["o0", "o1", "o2", "o3"],
                    ["g0"], ["g0", "o0"]):
        st = A.stack(findings, spent={i: "dismiss" for i in cleared})
        assert not [i for i in st["promoted"] if i.startswith("g")], (
            f"a finding that gates a decision arrived late: {st['promoted']}")
        assert not [i for i in st["collapsed"] if i.startswith("g")], st
        assert len(st["live"]) + len(st["cleared"]) + len(st["collapsed"]) == 12


def test_clearing_a_gating_finding_frees_nothing_because_it_cost_nothing():
    """A critical sits outside the bound, so dismissing one cannot promote.

    The sharp edge of "criticals are outside the bound": they never consumed
    budget, so clearing one cannot release any. A rule that promoted here would
    be paying out a slot that was never spent.
    """
    findings = ([{"id": "g", "severity": "critical", "source": "pack",
                  "pack": "clinical", "rank": 0}]
                + [{"id": f"o{i}", "severity": "warning", "source": "profile",
                    "rank": 1 + i} for i in range(9)])
    st = A.stack(findings, spent={"g": "dismiss"})
    assert st["promoted"] == [], (
        f"dismissing a finding outside the bound promoted {st['promoted']}")
    assert len(st["live"]) == A.BOUND, st["live"]


def test_the_marker_names_the_verb_that_actually_happened():
    """`dismiss` and `defer` are different decisions and the sentence says which.

    Rounding both to "cleared" when only one occurred would be the interface
    generalizing a recorded decision, which is the record layer of the governing
    rule.
    """
    findings = [{"id": f"o{i}", "severity": "warning", "source": "profile",
                 "rank": i} for i in range(9)]
    said = lambda spent: A.stack(findings, spent=spent)["promoted_because"]
    assert "dismissed" in said({"o0": "dismiss"})
    assert "deferred" in said({"o0": "defer"})
    assert "cleared" in said({"o0": "dismiss", "o1": "defer"})
    assert A.stack(findings)["promoted_because"] == "", (
        "a promotion sentence with nothing cleared")


# ── the marker, on the page ─────────────────────────────────────────────────

_CARD = re.compile(r'<article class="[^"]*" id="find-([^"]+)"')
_MARK = re.compile(r'<span class="chip arrived">([^<]*)</span>')


def _routes(client, pid, project):
    return {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }


#: `AUDIT-039`, `L56-B2`. **The page-level claim gets its own two lenses, and
#: they are both at the SHIPPING bound**, because that is the only bound the
#: page can ever see: `project.py` builds `explore_stack` as
#: `_att.stack(findings, spent=…)` with **no bound argument**, so the API serves
#: `A.BOUND` and nothing else. Driving this claim at `bound=2` was never
#: possible; the old code parametrized over `LENSES` and then skipped the
#: metabolomics arm, which is `GUIDED-097`'s two-lens rule silently reduced to
#: one and reported green.
#:
#: **The docstring above was wrong and this is the correction.** It claimed
#: `MIN_COLLAPSE` left *exactly ONE* fixture in this repository with a remainder
#: at the shipping bound. Swept through the API at `A.BOUND = 5`,
#: `metabolomics_merged_modes.csv` has a **four-card** remainder under the
#: metabolomics lens — `live=8, collapsed=4` — against
#: `metabolomics_untargeted.csv`'s `live=10, collapsed=0`. So the second lens
#: does not need a different bound; it needs a different fixture.
PAGE_LENSES = {
    "clinical": ("clinical_labs.csv", "clinical", "readmitted"),
    "metabolomics": ("metabolomics_merged_modes.csv", "metabolomics", "responder"),
}


@pytest.mark.parametrize("label", sorted(PAGE_LENSES))
@pytest.mark.parametrize("n_dismissals", [1, 2])
def test_the_promoted_card_says_why_it_is_there(label, n_dismissals):
    """The marker is on the promoted card and on nothing else.

    Two dismissals as well as one, because the second is where a marker that was
    never cleared would show: a card marked *"moved up"* on a render where it did
    not move is the interface asserting something false about its own history.

    **`AUDIT-039`. Both skips are gone and the precondition is asserted from the
    data instead.** The second one was the dangerous one: it stood down exactly
    when `explore_stack["collapsed"]` was empty — which is the state a stack
    regression produces. A change to `BOUND`, to `MIN_COLLAPSE`, to
    `gates_a_decision` or to a fixture's finding count would have turned the
    test that carries this file's name **quiet** rather than red, and pytest
    counts a skip as not-a-failure.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    fixture, lens, target = PAGE_LENSES[label]
    client, pid = _driven(fixture, lens, target)
    project = client.get(f"/project/{pid}").json()

    # THE PRECONDITION, ESTABLISHED FROM THE DATA AND ASSERTED. Both fixtures
    # are shipped and their stack shape at the shipping bound is a deterministic
    # fact about them, so "there is a remainder to promote from" is something
    # this test may require rather than something it may decline over.
    assert project["explore_stack"]["collapsed"], (
        f"{label}: {fixture} has no collapsed remainder at the shipping bound "
        f"A.BOUND={A.BOUND}, so nothing can be promoted and this test's subject "
        f"does not exist. That is a change in the stack, not a reason to stand "
        f"down — AUDIT-039.")

    for i in range(n_dismissals):
        st = project["explore_stack"]
        victim = next((i for i in st["live"]
                       if not A.gates_a_decision(
                           next(f for f in project["findings"] if f["id"] == i))),
                      None)
        assert victim is not None, (
            f"{label}: every live card gates a decision after {i} dismissal(s), "
            f"so there is nothing this test is allowed to clear")
        assert st["collapsed"], (
            f"{label}: the remainder emptied after {i} dismissal(s), so the "
            f"{i + 1}th promotion has nothing to promote FROM. This is the "
            f"state a stack regression produces and it used to be a skip.")
        project = client.post(f"/project/{pid}/decision",
                              json={"kind": "dismiss", "subject": victim}).json()

    st = project["explore_stack"]
    out = PH.run("__emit({list: __harness.html('profList'),"
                 "        more: __harness.html('profMore')});",
                 routes=_routes(client, pid, project), search=f"?project={pid}")
    html = out["list"] or ""

    rendered = _CARD.findall(html)
    assert rendered == st["pushed"], (
        f"{label}: the page pushed {rendered}, the server said {st['pushed']}")

    marked = []
    for fid in rendered:
        block = html.split(f'id="find-{fid}"', 1)[1].split("</article>")[0]
        if _MARK.search(block):
            marked.append(fid)
    assert marked == st["promoted"], (
        f"{label}: the page marks {marked} and the server promoted "
        f"{st['promoted']}")
    if st["promoted"]:
        assert st["promoted_because"] in html, (
            f"{label}: the marker does not carry the server's sentence")
    assert len(_MARK.findall(html)) == len(st["promoted"]), (
        f"{label}: {len(_MARK.findall(html))} markers rendered for "
        f"{len(st['promoted'])} promotions")

    # AND THE PAGE'S OWN READING OF THE RECORD AGREES WITH THE SERVER'S.
    # `statusOf` and `attention.spent_ids` are two readers of one record — a
    # card the page draws as dismissed must not still be costing budget, or the
    # two have drifted and the bound would be silently wrong.
    gone = re.findall(r'<article class="([^"]*gone[^"]*)" id="find-([^"]+)"', html)
    assert sorted(g[1] for g in gone) == sorted(st["cleared"]), (
        f"{label}: the page draws {sorted(g[1] for g in gone)} as cleared and "
        f"the partition freed {sorted(st['cleared'])}")


def test_the_probe_reports_its_own_coverage(capsys):
    """`LOOP.md` §10: sizes driven, dismissals driven, promotions observed,
    criticals ever promoted late, and any size where the arithmetic fails."""
    sizes: List[int] = []
    dismissals = promotions = late_criticals = 0
    broken: List[str] = []

    findings = ([{"id": "g", "severity": "critical", "source": "pack",
                  "pack": "clinical", "rank": 0}]
                + [{"id": f"o{i}", "severity": "warning", "source": "profile",
                    "rank": 1 + i} for i in range(24)])
    for n in range(0, 26):
        subset = findings[:n]
        sizes.append(n)
        spent: Dict[str, str] = {}
        for _ in range(n):
            st = A.stack(subset, spent=spent)
            if (len(st["live"]) + len(st["cleared"]) + len(st["collapsed"])
                    != st["served"]):
                broken.append(f"n={n}, {len(spent)} cleared")
            promotions += len(st["promoted"])
            late_criticals += sum(1 for i in st["promoted"]
                                  if i == "g")
            nxt = next((i for i in st["live"] if i != "g" and i not in spent), None)
            if nxt is None or not st["collapsed"]:
                break
            spent[nxt] = "dismiss"
            dismissals += 1

    with capsys.disabled():
        print("\n  ── L46-D · the promoted card ──")
        print(f"  stack sizes driven             0–{max(sizes)} ({len(sizes)} sizes)")
        print(f"  dismissals driven              {dismissals}")
        print(f"  promotions observed            {promotions}")
        print(f"  criticals promoted late        {late_criticals}   <- must be 0")
        print(f"  ledger failures                {len(broken)}")
        for b in broken:
            print(f"      {b}")
        print(f"  lenses driven through the API  {len(LENSES)} "
              f"({', '.join(sorted(LENSES))})")
        print(f"  shapes NOT covered             {len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")

    assert late_criticals == 0
    assert not broken
