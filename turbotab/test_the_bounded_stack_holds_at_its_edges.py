"""L45-D — a bounded list lies at its edges, so drive it there. `GUIDED-149`.

Every edge here is reachable by a real project: a table nothing was found in, a
table with one finding, a table sitting exactly on the bound, a table one over
it, `clinical_labs.csv`'s twenty-one, a stack that is all `critical`, a stack
with none, and a project where no lens was answered so no pack ever ran.

## The two properties, asserted rather than eyeballed

Both are one-liners and both are the kind of thing that silently stops being
true, which is why they are asserted at **every** size rather than at the one
the loop happened to build against:

1. **No finding that gates a decision is ever inside the collapsed group.**
   `ROADMAP.md` Decision B: *a blocker that only offers is not gating*, and
   *blockers rank first*.
2. **Rendered + collapsed == served.** The number in the affordance is the
   number behind it. A count that is off by one at an edge is the app asserting
   something false about its own contents, in a sentence the user reads.

## Why it is driven twice

`attention.stack` is checked directly, because that is where the rule lives and
because Python can construct sizes no fixture produces — a stack of zero, a
stack of exactly two, twenty-one criticals. **And the page is driven**, because
this door's oldest habit is a server that computes correctly beside an interface
that renders something else (`GUIDED-142`, `GUIDED-075`, `GUIDED-058`, and six
measured surfaces). An arithmetic that is exact on the wire and wrong on screen
is the version of this defect that matters, since the affordance is a sentence a
person reads and acts on.

## The findings are real, and that is deliberate

`LOOP.md` trap #3 — *a guard that manufactures the thing whose absence is the
defect*. Every card at every size here is a finding a real driven project served;
sizes are composed by **resampling** that pool with fresh ids, never by inventing
a payload or by overriding a severity to make a case appear. A stack of
twenty-one criticals is twenty-one copies of criticals the metabolomics lens
actually produced.

## `GUIDED-097` — two lenses, and what neither covers

Clinical and metabolomics, plus the no-lens project. `SHAPES_NOT_COVERED` names
what none of them reaches.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from turbotab import attention as A
from turbotab import engine

DATA = Path(__file__).resolve().parent / "sample_data"

#: Two lenses of deliberately different stack shape, plus a project where the
#: lens question was never answered. `(fixture, lens, target)`.
LENSES = {
    "clinical": ("clinical_labs.csv", "clinical", "readmitted"),
    "metabolomics": ("metabolomics_untargeted.csv", "metabolomics", "responder"),
    # A LENS WITH NO PACK, which in this app is *no lens answered* — the pack
    # stream is empty and the Explore stack is profile findings alone. It still
    # carries a target, because `renderEda` returns early without one and the
    # Explore section has not been reached at all: a probe driving a page that
    # never rendered would report the surface as dead rather than as bounded.
    "no lens answered": ("longitudinal_visits.csv", None, "outcome"),
}

#: NOT COVERED, said out loud. A probe that reports only what it drove has not
#: reported its coverage.
SHAPES_NOT_COVERED = (
    "A stack containing a `blocker`-severity FINDING. `ml/router.py:77` ranks "
    "that severity beside `critical` and `NEVER_COLLAPSED` holds both, but "
    "nothing in this repository emits it onto a finding — blockers are "
    "Questions, built from signals (`GUIDED-151`). So the `blocker` half of the "
    "never-collapse rule is exercised synthetically below and by no real "
    "project.",
    "A stack whose findings carry no `rank` at all. `_rank` falls back to "
    "arrival order and that fallback is driven, but every live producer goes "
    "through `engine.rank_findings`, which always sets one.",
    "A real table producing exactly two Explore findings. The small end of the "
    "sixteen fixtures is 1 · 3 · 3 · 3 · 3, so two is synthetic here and is "
    "absent from `prototypes/explore-stack.html` for the same reason.",
)

#: The sizes driven. Zero and one are the degenerate ends; `BOUND` and
#: `BOUND + 1` are where a bound reads as arbitrary if it is going to; 13 and 21
#: are `clinical_labs.csv`'s Explore stack and its whole finding set.
SIZES = (0, 1, 2, A.BOUND - 1, A.BOUND, A.BOUND + 1, 13, 21)


# ── the pool ────────────────────────────────────────────────────────────────


def _resample(source: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """`n` findings drawn from `source`, each a real one with a fresh id."""
    return [dict(source[i % len(source)], id=f"{source[i % len(source)]['id']}#{i}")
            for i in range(n)]


def _ranked(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank them the way `engine.rank_findings` does, using its own tables.

    Reusing the engine's ordering rather than restating it: a probe with its own
    sort would be checking the partition against a rule the app does not use.
    """
    ordered = sorted(findings, key=lambda d: (
        engine.SEVERITY_RANK.get(d.get("severity"), 99),
        engine.CONFIDENCE_RANK.get(d.get("confidence"), 1),
        str(d.get("id")),
    ))
    return [dict(f, rank=i) for i, f in enumerate(ordered)]


#: `(label, n, n_gating)` — the eight the prompt names, plus the mixes that make
#: "all critical" and "no critical" a property rather than one case.
CASES = (
    [(f"{n} findings, none gating", n, 0) for n in SIZES]
    + [(f"{n} findings, all gating", n, n) for n in SIZES if n]
    + [("21 findings, one gating", 21, 1),
       ("21 findings, all but one gating", 21, 20)]
)


# ── the two properties, at every size ───────────────────────────────────────


def test_a_bound_of_zero_still_cannot_hide_something_that_gates_a_decision():
    """The sharpest form of rule 1: no bound can bury a blocker, including one
    that pushes nothing else at all.

    Three ordinary findings rather than one, because `MIN_COLLAPSE` means a
    remainder of one is shown — so a two-finding fixture would collapse nothing
    and the claim would pass without the rule being exercised. That is `L45`'s
    own lesson about a fixture nothing is wrong for, arriving on the test that
    lesson was written in.
    """
    gating = {"id": "g", "severity": "blocker", "source": "pack",
              "pack": "clinical", "rank": 7}
    ordinary = [{"id": f"o{i}", "severity": "warning", "source": "profile",
                 "rank": i} for i in range(3)]
    st = A.stack(ordinary + [gating], bound=0)
    assert st["pushed"] == ["g"], (
        f"a bound of zero collapsed something that gates a decision: {st}")
    assert st["collapsed"] == ["o0", "o1", "o2"]
    # AND IT IS FIRST, ahead of a finding the engine ranked above it.
    # `engine.SEVERITY_RANK` has no `blocker` key, so `rank_findings` would sort
    # that severity to 99 and put it LAST while `ml/router.py:77` ranks it 0.
    # The surface re-asserts the one clause the constitution makes absolute
    # rather than trusting a rank table that disagrees with itself
    # (`GUIDED-151`).
    st2 = A.stack(ordinary + [gating], bound=40)
    assert st2["pushed"][0] == "g", (
        f"a blocker is not first: {st2['pushed']}. A blocker third in a list of "
        f"nine is a blocker in name only.")


def test_a_finding_with_no_rank_is_placed_by_arrival_and_never_dropped():
    """The fallback `_rank` takes when a producer sets no rank.

    Three, not two: `MIN_COLLAPSE` shows a remainder of one, so a two-finding
    fixture at bound 1 collapses nothing and says nothing about arrival order.
    """
    raw = [{"id": k, "severity": "warning", "source": "profile"}
           for k in ("a", "b", "c")]
    st = A.stack(raw, bound=1)
    assert st["served"] == 3
    assert st["pushed"] == ["a"] and st["collapsed"] == ["b", "c"]


def test_the_structural_stream_is_not_in_this_stack():
    """`structure` findings render in their own card at the Data step, filtered
    by the repair groups. Pulling them in here would put a question the user
    already answered back in front of them."""
    st = A.stack([{"id": "s", "severity": "critical", "source": "structure"},
                  {"id": "p", "severity": "warning", "source": "profile"}])
    assert st["served"] == 1 and st["pushed"] == ["p"]


def test_a_negative_bound_is_refused_rather_than_clamped():
    with pytest.raises(A.StackError):
        A.stack([], bound=-1)


# ── the same arithmetic, on the page ────────────────────────────────────────

def _routes(run, project):
    pid = run["pid"]
    client = run["client"]
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


_CARD = re.compile(r'<article class="[^"]*" id="find-([^"]+)"')


