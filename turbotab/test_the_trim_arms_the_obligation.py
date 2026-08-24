"""Clause §05's arming half: a train-only trim records what the report must say.

`STATE-103` names why this clause is the one most likely to go unbuilt: it is
the only clause whose obligation fires at a **different step** from the action
that arms it. Every other clause is checkable where it happens — the seal states
its basis at the seal, the grain is asked before the seal, a structural repair
posts its receipt when it executes. This one deliberately spends its friction
later, at export, and a requirement armed in one step and enforced in another is
exactly the kind that gets built halfway.

**So it was split, and the split is the point.** Under one row, landing the
arming half would have read as progress on the whole clause. Under two, this one
closes and the other stays visibly open:

    STATE-103   the arming half — a trim records the obligation      (here)
    STATE-105   the firing half — export refuses without the breakdown (open)

The firing half is deliberately NOT built. There is no Report step, and a
blocker with nothing to block cannot be tested — which would make it the
half-built clause this row exists to prevent.

The tests below assert two different things and it is worth naming which:

* the trim obeys §04 — training rows only, sealed rows never touched;
* the trim arms §05 — the obligation is recorded, with the numbers the report
  needs and cannot recover for itself.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import (eligibility as E, engine, grain as G,          # noqa: E402
                      obligations as OB)
from turbotab.project import AnalysisProject, ProjectError           # noqa: E402


def study(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        # A wide spread, so a trim leaves rows outside the range on BOTH sides
        # of the seal. Without that the extrapolation number is zero and the
        # tests would pass on an implementation that never computed it.
        "age": rng.integers(18, 95, n).astype(float),
        "glucose": rng.normal(100, 30, n),
        "outcome": rng.integers(0, 2, n),
    })


def _sealed() -> AnalysisProject:
    p = AnalysisProject.from_dataframe(study(), "t")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    return p


# ── the arming half ──────────────────────────────────────────────────────────

def test_a_train_only_trim_arms_the_extrapolation_obligation():
    """§05: the trim is a legitimate choice and earns no blocker, so it ARMS a
    requirement instead. Something has to record that it was armed, or the
    clause has no memory between the step that arms it and the step that fires.

    Clause: `lockbox-05`
    """
    p = _sealed()
    assert not p.obligations, "nothing is armed before anything is trimmed"

    p.trim_training_rows("age", minimum=40, maximum=75,
                         reason="The cohort of clinical interest is 40 to 75.")

    assert len(p.obligations) == 1
    ob = p.obligations[0]
    assert ob["kind"] == OB.EXTRAPOLATION
    assert ob["discharged_at"] == "report", (
        "the obligation does not name where it is discharged, so nothing can "
        "look it up — which is how a two-step obligation becomes two halves")
    assert ob["discharged"] is False
    assert OB.outstanding(p.obligations) == [ob]


def test_the_obligation_carries_the_numbers_the_report_cannot_recover():
    """The count of held-out rows outside the trimmed range is computed AT TRIM
    TIME, and it has to be, because the training rows outside the range are gone
    afterwards — the report cannot count what is no longer there.

    This is the number the clause is actually about: the held-out rows were
    never trimmed (§04 forbids it), so some of them lie outside the range the
    model was fitted on, and a single aggregate metric averages them in silently.

    Clause: `lockbox-05`
    """
    p = _sealed()
    p.trim_training_rows("age", minimum=40, maximum=75,
                         reason="The cohort of clinical interest is 40 to 75.")
    ob = p.obligations[0]

    assert ob["n_train_trimmed"] > 0, (
        "this fixture trims no training rows, so it demonstrates nothing")
    assert ob["n_test_outside_range"] > 0, (
        "no held-out row falls outside the trimmed range in this fixture, so "
        "the extrapolation the clause is about does not arise and the test "
        "would pass on an implementation that never computed it")
    assert ob["n_test_outside_range"] < ob["n_test_total"]
    assert ob["column"] == "age"
    assert ob["minimum"] == 40 and ob["maximum"] == 75
    assert "40 to 75" in ob["reason"]

    # And the sentence a report has to be able to produce, produced now.
    assert str(ob["n_test_outside_range"]) in ob["sentence"]
    assert "separately for in-range and out-of-range" in ob["sentence"]
    assert "stratified" in ob["requires"]


def test_the_trim_touches_training_rows_only():
    """§04, at the operation rather than as a promise. `STATE-101` measured what
    happens when a filter runs over the whole frame instead: 7 of 60 sealed rows
    disappeared and evaluation ran on 53 while the chip said 60.

    Clause: `lockbox-05`
    """
    p = _sealed()
    sealed = set(p.lockbox["labels"])
    n_before = len(p.df)

    p.trim_training_rows("age", minimum=40, maximum=75, reason="cohort of interest")

    still_there = {l for l in p.df.index if l in sealed}
    assert still_there == sealed, (
        f"{len(sealed) - len(still_there)} sealed row(s) were removed by a "
        "robustness trim; the test set is never touched")
    assert len(p.df) < n_before, "the trim removed nothing, so it proves nothing"
    p.assert_identity_intact()

    # The sealed rows that survive include ones OUTSIDE the range — that is the
    # whole point, and if they had been trimmed to match the extrapolation
    # would vanish along with the disclosure.
    outside = [l for l in sealed if not (40 <= p.df.loc[l, "age"] <= 75)]
    assert outside, (
        "every surviving sealed row is inside the trimmed range, so this test "
        "cannot tell 'sealed rows were kept' from 'sealed rows were trimmed too'")


def test_a_trim_before_the_seal_is_refused_and_names_the_other_object():
    """There is no training partition until the test set is sealed. Before the
    seal, narrowing the study is an ELIGIBILITY criterion — a different object,
    a different question, and it changes N. §04's whole table.

    Clause: `lockbox-05`
    """
    p = AnalysisProject.from_dataframe(study(), "t")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    with pytest.raises(ProjectError) as exc:
        p.trim_training_rows("age", minimum=40, reason="cohort")
    said = str(exc.value)
    assert "post-seal by definition" in said
    assert "eligibility criterion" in said, (
        "the refusal does not name the object the user actually wants, so it "
        "is a dead end rather than a route")


def test_a_trim_with_no_reason_is_refused():
    """The reason is what the report prints beside the breakdown. Without it the
    disclosure would say some rows were outside a range nobody can explain.

    Clause: `lockbox-05`
    """
    p = _sealed()
    with pytest.raises(ProjectError, match="what the report has to print"):
        p.trim_training_rows("age", minimum=40, maximum=75)


def test_a_trim_with_no_bounds_is_refused():
    """A trim that narrows nothing arms nothing, and an obligation with no
    extrapolation in it would be a disclosure with nothing to disclose.

    Clause: `lockbox-05`
    """
    p = _sealed()
    with pytest.raises(ProjectError, match="narrows nothing"):
        p.trim_training_rows("age", reason="no bounds given")


def test_the_obligation_survives_the_save_file():
    """§05 spans two steps, so the obligation has to survive the gap between
    them — and a save and restore is the longest form that gap takes. A restored
    project that lost it would export a single aggregate metric with nothing
    left to object.

    Clause: `lockbox-05`
    """
    from turbotab import archive
    p = _sealed()
    p.trim_training_rows("age", minimum=40, maximum=75, reason="cohort of interest")

    back = archive.from_bytes(archive.to_bytes(p))
    assert len(back.obligations) == 1
    assert back.obligations[0]["n_test_outside_range"] == \
        p.obligations[0]["n_test_outside_range"]
    assert OB.outstanding(back.obligations), (
        "the restored obligation is not outstanding, so the report would think "
        "it had already been discharged")


def test_the_firing_half_is_deliberately_absent_and_tracked():
    """The other half of the split, asserted rather than assumed.

    `STATE-105` is the row. This test exists so that "the firing half is not
    built" is a statement the suite makes rather than a thing a reader has to
    notice — and so that building it turns this test red, which is the signal to
    close the row and rewrite this.

    Clause: `lockbox-05`
    """
    import json
    with open(os.path.join(PROJECT_ROOT, "docs", "turbotab", "data",
                           "findings.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    rows = {r["id"]: r for r in (data["findings"] if isinstance(data, dict) else data)}

    firing = rows.get("STATE-105")
    assert firing is not None, "the firing half has no row"
    assert firing["status"] in ("OPEN", "PARTIAL"), (
        "STATE-105 is closed, so the firing half is claimed to be built. If it "
        "is, this test should be replaced by one that drives an export and "
        "watches it refuse.")
    assert "lockbox-05" in json.dumps(firing)

    # And nothing here pretends to fire it. `outstanding` reports; it does not
    # block, because there is no export to block.
    assert not hasattr(OB, "refuse_export"), (
        "something in the arming half has started doing the firing half's job, "
        "which is the halfway state the split exists to prevent")
