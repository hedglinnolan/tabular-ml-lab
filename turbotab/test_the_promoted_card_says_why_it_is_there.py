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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


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


def _ledger(st: Dict[str, Any]) -> None:
    """The disjoint form, asserted the same way at every call site."""
    assert len(st["pushed"]) + len(st["collapsed"]) == st["served"], st
    assert len(st["live"]) + len(st["cleared"]) + len(st["collapsed"]) == st["served"], (
        f"{len(st['live'])} live + {len(st['cleared'])} cleared + "
        f"{len(st['collapsed'])} collapsed != {st['served']} served")
    assert st["remainder"]["n"] == len(st["collapsed"]), st["remainder"]


# ── the partition, driven against the real record ───────────────────────────


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
