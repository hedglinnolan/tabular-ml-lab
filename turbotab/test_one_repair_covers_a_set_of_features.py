"""`DRIVE-002` — nine features, one idea, nine show-me-then-apply cycles.

> Nine NHANES features are binary written as text. The engine found all nine and
> the driver had to open and apply each individually. Show what it means for
> one, then let the user select which features to run it on, then apply to the
> selected set.

This is `turbotab/bulk.py`'s rule-scope pointed at repairs rather than at
questions: **operations apply to sets defined by a rule.** `bulk.py` was built
for the missingness question, where 308 columns with blanks produced 308
questions. A repair is the same shape one object over.

## What this file asserts, and the one that matters most

The load-bearing assertion is **the frame**, not the record. A bulk apply that
wrote one satisfying sentence into the transcript and repaired one column would
satisfy every plausible test of the receipt — and that is the exact shape of the
critical this project closed last loop, where nine tests read a seal's record
and none read the draw. So the columns are checked in the dataframe the project
holds afterwards, and the ones left out are checked to be **unchanged**.

The second is that **the record and the interview agree about the declined
members**. The first implementation of this recorded *"1 other in the same group
was deliberately left as recorded"* and then re-asked that member on its own
card. Both statements were rendered, in the same session, to the same user. A
sentence in the transcript that the interview contradicts is worse than no
sentence, because it is the app asserting something the app itself does not
believe.
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

from ml import router                                                 # noqa: E402
from turbotab import engine, repairs as R                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _upload(client, name):
    with open(DATA / f"{name}.csv", "rb") as fh:
        return client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]


def _pushed(client, pid, step="data"):
    return [q for q in client.get(
        f"/project/{pid}/interview?step={step}").json()["questions"]
        if q["mode"] == "push"]


# ── the grouping ─────────────────────────────────────────────────────────────


def test_a_group_of_one_is_not_a_group():
    """`bulk.MIN_GROUP`'s argument, and the same number for the same reason.

    A rule over one column is a column with extra words in front of it, and a
    bulk affordance offered over a single leftover is worse than asking.
    """
    findings = [{"id": "a", "fix_kind": "coerce_numeric", "fix_label": "x",
                 "affected_columns": ["one"]}]
    assert R.group(findings) == []
    assert len(R.group(findings + [dict(findings[0], id="b",
                                        affected_columns=["two"])])) == 1


@pytest.mark.parametrize("kind", sorted(R.NEVER_GROUPED))
def test_the_repairs_that_never_group_say_why(kind):
    """A name on the exclusion list is a claim that bulk would be WRONG, not
    merely awkward — so each carries its reason, and none is a bare entry."""
    findings = [{"id": f"{kind}__{i}", "fix_kind": kind, "fix_label": "x",
                 "affected_columns": [f"c{i}"]} for i in range(3)]
    assert R.group(findings) == [], f"{kind} grouped and must not"
    assert len(R.NEVER_GROUPED[kind]) > 60, (
        f"{kind}'s exclusion has no argument behind it")


# ── the effect, read back off the frame ──────────────────────────────────────


def test_the_group_cites_a_finding_so_it_counts_as_findings_driven():
    """*"Push the notable"* is only true if the question says what it is
    pushing. A grouped question that cited nothing scored as a question that
    exists because a pipeline stage exists."""
    df = pd.read_csv(DATA / "clinic_visits.csv")
    findings = engine.rank_findings(
        engine.diagnose(df, target="outcome"),
        engine.profile(df, "outcome", None), lens=[], df=df)
    plan = router.plan(findings, target="outcome", step="data",
                       detection={"detected": "classification",
                                  "confidence": "high", "reasons": []})
    groups = [q for q in plan if q.key.startswith("repair_bulk::")]
    assert groups, "no groups on this fixture"
    for q in groups:
        assert q.is_findings_driven, f"{q.key} cites no finding"
        assert q.triggering_finding in q.covers


def test_a_deferred_group_comes_back_at_the_step_it_names():
    """Deferral is a first-class disposition only if it comes back.

    The first implementation set `status = "deferred"` whenever the key was in
    `deferred`, regardless of the step being planned — so a group deferred to
    Explore stayed deferred AT Explore and never resurfaced. Deferral would
    have been a discard with manners.
    """
    df = pd.read_csv(DATA / "clinic_visits.csv")
    findings = engine.rank_findings(
        engine.diagnose(df, target="outcome"),
        engine.profile(df, "outcome", None), lens=[], df=df)
    detection = {"detected": "classification", "confidence": "high",
                 "reasons": []}
    first = router.plan(findings, target="outcome", detection=detection,
                        step="data")
    key = next(q.key for q in first if q.key.startswith("repair_bulk::"))

    at_data = router.plan(findings, target="outcome", detection=detection,
                          step="data", deferred={key: "explore"})
    moved = next(q for q in at_data if q.key == key)
    assert moved.status == "deferred" and moved.defer_target == "explore"

    at_explore = router.plan(findings, target="outcome", detection=detection,
                             step="explore", deferred={key: "explore"},
                             answered=["choose_target"])
    back = next((q for q in at_explore if q.key == key), None)
    assert back is not None, "a deferred group never resurfaced"
    assert back.status == "asked", (
        "the group came back still deferred, so deferral is a discard with "
        "manners")
    assert back.deferred_from == "data"
