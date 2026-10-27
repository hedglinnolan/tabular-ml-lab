"""Keeping the table as it is answers an unanswered orientation without a re-seal (the swarm's C3).

The plasma panel's crash sweep never reached Fit under the metabolomics lens: the lens asks which
way round the table is, the sweep answered `sample_major` after the split was drawn, and the seal
refused it as a change to what a row is ("sealed"); the genomics lens ended on the same refusal.
An unanswered orientation is the table as it is: the oriented stage turns the table only on
`feature_major` (``stages.working.oriented_stage``), and the sequence offers `sample_major` as
"Keep the table as it is" (``sequence._orientation_turns_before_the_target``). So `sample_major`
on an unanswered orientation changes no row, and the seal's three comparisons of what a row is
(the answer, a revert, and the draw an opened seal holds) read it so. Turning the table is still
a change: `feature_major` waits for a re-seal, and so does a revert that turns it back.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import DecisionRecord, Refusal

KEEP = d.SetOrientation(orientation="sample_major")
TURN = d.SetOrientation(orientation="feature_major")


def record(seq: int, decision, rid: str | None = None) -> DecisionRecord:
    return DecisionRecord(id=rid or f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                          decision=decision)


def drawn(holdout: float) -> list[DecisionRecord]:
    """A metabolomics project with its split drawn and the orientation never answered."""
    return [record(1, d.SetLens(lenses=["metabolomics"])), record(2, d.SetTarget(column="y")),
            record(3, d.SetSplit(holdout=holdout, seed=0, folds=5), "split")]


def ctx_of(records: list[DecisionRecord]) -> dict:
    return {"state": d.fold(records), "records": records}


@pytest.mark.parametrize("holdout", [0.0, 0.2], ids=["folds only", "held out"])
def test_keeping_the_table_as_it_is_on_an_unanswered_orientation_is_accepted(holdout):
    records = drawn(holdout)
    assert d.fold(records).orientation is None
    seal._decision_a_waits_for_a_reseal(KEEP, ctx_of(records))
    d.validate(KEEP, ctx_of(records))


@pytest.mark.parametrize("holdout", [0.0, 0.2], ids=["folds only", "held out"])
def test_turning_the_table_on_an_unanswered_orientation_still_waits_for_a_reseal(holdout):
    records = drawn(holdout)
    with pytest.raises(Refusal) as refused:
        seal._decision_a_waits_for_a_reseal(TURN, ctx_of(records))
    assert refused.value.code == "sealed"
    assert "which way round the table is" in refused.value.message


def test_reverting_the_kept_layout_after_the_seal_is_not_a_change():
    """`sample_major` recorded after the draw, then reverted: the orientation is unanswered again,
    the table as it was."""
    records = [*drawn(0.2), record(4, KEEP, "keep")]
    seal._revert_keeps_the_seal(d.Revert(decision_id="keep"), {"records": records})


def test_reverting_a_turn_made_before_the_seal_still_waits_for_a_reseal():
    """`feature_major` answered before the outcome and the draw: reverting it turns the table back."""
    records = [record(1, d.SetLens(lenses=["metabolomics"])), record(2, TURN, "turn"),
               record(3, d.SetTarget(column="y")),
               record(4, d.SetSplit(holdout=0.2, seed=0, folds=5), "split")]
    with pytest.raises(Refusal) as refused:
        seal._revert_keeps_the_seal(d.Revert(decision_id="turn"), {"records": records})
    assert refused.value.code == "sealed"


def opened(then: list[DecisionRecord]) -> list[DecisionRecord]:
    seq = max(r.seq for r in then)
    return [*then, record(seq + 1, d.SelectModels(models=["linear"])),
            record(seq + 2, d.OpenSeal(), "open")]


def test_the_opened_seal_reads_the_kept_layout_as_the_draw_it_opened():
    """Opened with the orientation unanswered, then `sample_major` recorded: the held-out rows are
    the ones opened, so a later revert of an unrelated answer is not refused as drawing them again."""
    records = [*opened(drawn(0.2)), record(7, KEEP, "keep"),
               record(8, d.SetPurpose(purpose="inference"), "purpose")]
    then = d.fold(records[:5])
    assert then.orientation is None and d.fold(records).orientation == "sample_major"
    assert seal._membership_change(then, d.fold(records), {}) is None
    seal._an_opened_seal_holds_still(d.Revert(decision_id="purpose"), {"records": records})


def test_the_opened_seal_still_reads_a_turn_as_drawing_the_rows_again():
    then = d.fold(opened(drawn(0.2)))
    turned = then.model_copy(update={"orientation": "feature_major"})
    assert seal._membership_change(then, turned, {}) == (
        "changing which way round the table is would draw them again")
