"""L47-A2 — the page built for the product owner may not state an older build.

`docs/turbotab/prototypes/explore-stack.html` is this project's answer to the one
condition with no automated instrument: `ROADMAP.md` condition 7's third half,
*beautiful*, which `pageharness.py` says in its own docstring it cannot reach. The
prototype's whole standing rests on **being what the server actually serves** —
its own header says *"Nothing here is typed by hand."*

**And it disagreed with itself.** At L47 the page rendered `CAPTURE.bound_because`
into its "What the bound is" box while `n_collapsing`, derived in the same
capture, said something else on the line above it:

    becauseLine  "…collapses something on three of them and nothing on thirteen"
    popTable     "At the shipping bound, 1 of 16 collapse anything at all."

L46 changed `MIN_COLLAPSE`, re-derived the count, and **snapshotted the old
constant** — because the capture copies `attention.BOUND_BECAUSE` verbatim and
nothing then checked that the copy was current. The remedy shipped with it was
*"re-run the capture after anything that changes what a fixture produces"*, in
prose, in a docstring.

**`LOOP.md` §05 has already ruled on that shape**: an instruction a tired agent
can skip is not a gate. The five pre-commit gates are a hook rather than a
paragraph because a commit once went out with the spelling test red, the gates
having been chained with a newline instead of `&&`. This file is the same move
one artifact over.

## What it does not check, stated because a probe that overstates is worse

It compares the **strings and numbers the page quotes** against the module they
were quoted from. It does not check that the captured *findings* are current — a
detector could change what `clinical_labs.csv` produces and this file would stay
green while the panels went stale. That is a real gap and it is left open
deliberately: re-deriving every fixture here would make the suite own a
several-minute drive, and `capture_explore_stack.py` already prints the
population every time it runs. **What is closed is the class that actually
fired**: a constant restated in two places, one of which moved.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from turbotab import attention as A

PAGE = (Path(__file__).resolve().parents[1] / "docs" / "turbotab" /
        "prototypes" / "explore-stack.html")

#: The markers `capture_explore_stack.py` writes between. Read rather than
#: assumed: if the capture ever stops writing them the parse fails loudly here,
#: which is the correct direction to be wrong in.
_ISLAND = re.compile(r"/\* ⟦CAPTURE-BEGIN⟧ \*/\s*var CAPTURE = (.*?);\s*"
                     r"/\* ⟦CAPTURE-END⟧ \*/", re.S)


@pytest.fixture(scope="module")
def capture():
    assert PAGE.exists(), f"{PAGE} is gone; the prototype is the exit gate"
    found = _ISLAND.search(PAGE.read_text(encoding="utf-8"))
    assert found, (
        "the capture markers are not in the prototype, so nothing can tell "
        "whether its numbers are current")
    return json.loads(found.group(1))


def test_the_prototype_quotes_the_bound_the_build_ships(capture):
    assert capture["ships"] == A.BOUND, (
        f"the page says the build ships bound {capture['ships']} and it ships "
        f"{A.BOUND}. Re-run docs/turbotab/prototypes/capture_explore_stack.py.")
    assert str(A.BOUND) in [str(b) for b in capture["bounds"]], (
        f"the shipping bound {A.BOUND} is not among the bounds the page "
        f"compares ({capture['bounds']}), so the panel labeled 'the build ships "
        f"this one' shows a partition the build does not produce")


def test_the_prototype_quotes_the_reason_the_build_ships(capture):
    """The one that actually fired.

    A verbatim comparison rather than a substring or a keyword: the failure was a
    *stale* sentence, and every keyword that mattered — median, sixteen, tail —
    appeared in both versions. Only equality could see it.
    """
    assert capture["bound_because"] == A.BOUND_BECAUSE, (
        "the prototype states a reason the build does not hold.\n"
        f"  page  : …{capture['bound_because'][160:300]}…\n"
        f"  build : …{A.BOUND_BECAUSE[160:300]}…\n"
        "Re-run docs/turbotab/prototypes/capture_explore_stack.py.")


def test_the_pages_own_two_statements_of_the_count_agree(capture):
    """The page derives `n_collapsing` and also quotes a sentence containing it.

    This is the shape of the defect rather than its instance: one number, two
    renderers, and the derived one was right. Asserted against the population the
    same capture holds, so the check needs nothing from outside the file.
    """
    derived = sum(1 for row in capture["population"] if row["collapsed_at_ships"])
    assert derived == capture["n_collapsing"], (
        f"the page's summary line says {capture['n_collapsing']} tables collapse "
        f"and its own table shows {derived}")
    assert len(capture["population"]) == 16, (
        f"the population is {len(capture['population'])} tables and every "
        f"sentence about it says sixteen")


def test_every_panel_shows_a_partition_the_module_still_produces(capture):
    """The captured partitions are recomputed here from the captured findings.

    Cheap, because the findings travel in the island — no API drive, no fixture
    read. It catches a change to `attention.stack`'s RULE (the bound, the
    never-collapse set, `MIN_COLLAPSE`) without pretending to catch a change to
    what a detector produces, which is the gap the docstring names.
    """
    for case in capture["cases"]:
        findings = [{"id": c["id"], "severity": c["severity"],
                     "source": "pack" if c["source_label"].endswith(" lens")
                               else "profile",
                     "pack": c["source_label"].replace(" lens", ""),
                     "rank": i}
                    for i, c in enumerate(case["cards"])]
        for bound, shown in case["stacks"].items():
            fresh = A.stack(findings, bound=int(bound))
            assert fresh["pushed"] == shown["pushed"], (
                f"{case['fixture']} at bound {bound}: the page shows "
                f"{len(shown['pushed'])} pushed and the module now produces "
                f"{len(fresh['pushed'])}")
            assert fresh["collapsed"] == shown["collapsed"], (
                f"{case['fixture']} at bound {bound}: the collapsed group moved")
            assert fresh["affordance"] == shown["affordance"], (
                f"{case['fixture']} at bound {bound}: the affordance now reads "
                f"{fresh['affordance']!r} and the page shows "
                f"{shown['affordance']!r}")
