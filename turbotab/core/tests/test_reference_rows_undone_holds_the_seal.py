"""Undoing a reference-row answer after the split is held by the seal as the answer is (INBOX:1024).

The exclusion of the pooled QCs (or QC-RLSC, after which they leave the same way) is refused once
rows are held out where the rows leaving would move the draw (``reference_rows._before_the_seal``).
That check was registered for ``apply_repair`` alone, so a dismissal, a deferral or a revert of the
applied answer brought the rows back unchecked, and a revert of a dismissal took them out again
unchecked. The seal's own checks on those kinds (``seal._the_draw_holds_until_a_reseal``,
``seal._revert_keeps_the_seal``) compare the outcome, the task, the temporal answer and the draw
columns' values, never which rows the working table keeps.

The truth is construction, on the pilot's plasma panel (100 participants, 28 pooled QCs) drawn with
20% held out. Where the QCs carry the outcome (a copy with ``disease`` written on every QC row), the
QC rows are in the draw: drawn with them out and with them back, the held-out participants by source
row differ (numpy's symmetric difference over the split's assignments joined to the row maps), so
the undoing would move participants across the seal. Where they carry none (the panel as run), the
draw holds every participant in place, but the rows that stay are numbered again and the seal names
rows by number: until the split is drawn again, the previews read every row but the old split's
sealed numbers, which then name held-out participants, unless every QC comes after every
participant. Holding nothing out (cross-validation alone), each undoing is accepted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import consequences, row_previews
from turbotab.core.decisions import (
    DecisionRecord,
    DeferFinding,
    DismissFinding,
    ProjectState,
    Refusal,
    Revert,
    fold,
    parse_decision,
    validate,
)
from turbotab.core.stages.working import row_map
from turbotab.core.tests.test_qc_rows_after_the_split import (
    PLASMA,
    PLASMA_BASE,
    QC_LEVEL,
    REFUSED_WORDS,
    Project,
    _laid_out,
    _variant,
    ctx_of,
    data,
    option,
    placed,
)

T0 = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLOSE_IN = "Start a new analysis, or keep them in and say so."
BACK_WORDS = ("The held-out rows are drawn over the rows the analysis holds, so reference rows "
              "leave before the seal; bringing them back now would draw it again. Start a new "
              "analysis, or keep them out.")
BACK_RENUMBERED_WORDS = ("The held-out rows are drawn and named by their row numbers, so reference "
                         "rows leave before the seal; bringing them back now would renumber the "
                         "rows with `disease` recorded that come after them. Start a new analysis, "
                         "or keep them out.")
UNDOINGS = ("dismiss", "defer", "revert")


def logged(project: Project, holdout: float, *answers: dict) -> list[DecisionRecord]:
    """The plasma answers as the decision log holds them: the lens, the orientation, the outcome,
    the grain and complete cases, then ``answers``, then the split (seed 0, five folds)."""
    shown = [{"kind": "set_lens", "lenses": ["metabolomics"]},
             {"kind": "set_orientation", "orientation": "sample_major"},
             {"kind": "set_target", "column": "disease"},
             {"kind": "set_grain", "grain": "one_row_per_unit"},
             {"kind": "set_missing", "strategy": "complete_case"},
             *answers,
             {"kind": "set_split", "holdout": holdout, "seed": 0, "folds": 5}]
    return [DecisionRecord(id=f"r{i}", seq=i, at=T0 + timedelta(minutes=i),
                           decision=parse_decision(d)) for i, d in enumerate(shown, 1)]


def excluded(project: Project) -> dict:
    """The exclusion of the pooled QCs, as the finding offers it."""
    return option(project.qc, "exclude_rows")["decision"]


def dismissed(project: Project) -> dict:
    return {"kind": "dismiss_finding", "finding_id": project.qc["id"]}


def undoing(how: str, project: Project, log: list[DecisionRecord]) -> Any:
    """The answer that brings the excluded QC rows back: a dismissal, a deferral, or a revert of
    the exclusion's record."""
    if how == "dismiss":
        return DismissFinding(finding_id=project.qc["id"])
    if how == "defer":
        return DeferFinding(finding_id=project.qc["id"], to="exclusions")
    return Revert(decision_id=next(r.id for r in log if r.decision.kind == "apply_repair"))


def checked(decision: Any, project: Project, log: list[DecisionRecord],
            split: bool = True) -> tuple[Refusal | None, ProjectState, dict]:
    """``decision`` validated as the server would, against the log's state and its run (the split
    fresh, or recomputing): the refusal or None, the state and the run."""
    state = fold(log)
    out = project.draw(state)
    try:
        validate(decision, {**ctx_of(state, out, split=split), "records": log})
    except Refusal as caught:
        return caught, state, out
    return None, state, out


def held_out(out: dict) -> np.ndarray:
    """The source rows of the participants a run holds out."""
    at = placed(out)
    return at.index[at["partition"] == "holdout"].to_numpy(dtype=np.int64)


@pytest.fixture(scope="module")
def carried(tmp_path_factory):
    """The plasma panel with ``disease`` recorded on every pooled QC (0), as INBOX:1024 measured
    it: the QC rows are in the draw."""
    folder = tmp_path_factory.mktemp("carried")
    project = Project(_variant(folder, "outcome"), folder / "p", PLASMA_BASE)
    yield project
    project.run.close()


def test_the_carried_panel_holds_its_outcome_on_every_row(tmp_path):
    """The construction, read with pandas: on the copy every QC row has ``disease`` recorded, and
    every participant's is the panel's own."""
    raw, df = pd.read_csv(PLASMA), pd.read_csv(_variant(tmp_path, "outcome"))
    qc = df["sample_type"].eq(QC_LEVEL)
    assert int(qc.sum()) == 28 and df.loc[qc, "disease"].eq(0).all()
    assert raw.loc[qc, "disease"].isna().all()
    pd.testing.assert_series_equal(df.loc[~qc, "disease"], raw.loc[~qc, "disease"])


@pytest.mark.parametrize("how", UNDOINGS)
def test_undoing_an_exclusion_whose_rows_carry_the_outcome_is_refused_while_rows_are_held_out(
        carried, how):
    """By construction, the exclusion recorded before a split that holds 20% out: drawn with the
    QC rows out and with them back (dismissed, deferred or the exclusion reverted, each folding to
    the QC rows back), participants change sides of the seal (29 of the 100 on this draw, counted
    here, not assumed), so each undoing is refused, with the split fresh or recomputing, in words
    that say the rows come back. Under cross-validation alone nothing is held out and each is
    accepted. On the base every one was accepted."""
    for holdout in (0.2, 0.0):
        log = logged(carried, holdout, excluded(carried))
        decision = undoing(how, carried, log)
        refusal, state, out = checked(decision, carried, log)
        back = carried.draw(fold([*log, DecisionRecord(id="u", seq=len(log) + 1,
                                                        at=T0 + timedelta(hours=1),
                                                        decision=decision)]))
        assert data(out["working"])["n_rows"] == 100 and data(back["working"])["n_rows"] == 128
        participants = placed(out).index.to_numpy(dtype=np.int64)
        crossed = np.intersect1d(np.setxor1d(held_out(out), held_out(back)), participants)
        if holdout:
            assert len(held_out(out)) == 20 and len(crossed) > 0
            assert refusal is not None and refusal.code == "rows_after_the_seal"
            assert refusal.message == BACK_WORDS
            assert [e["label"] for e in refusal.exits] == ["Keep the rows as they are"]
            assert checked(decision, carried, log, split=False)[0].message == BACK_WORDS
        else:
            assert len(crossed) == 0 and refusal is None


def test_a_revert_that_takes_the_rows_out_again_is_held_as_the_exclusion_is(carried):
    """The exclusion and then its dismissal, both before the split: reverting the dismissal brings
    the exclusion back, so the QC rows leave after the split. By construction participants change
    sides of the seal, so it is refused with the exclusion's own words, as applying it then is;
    under cross-validation alone it is accepted. On the base it was accepted."""
    for holdout in (0.2, 0.0):
        log = logged(carried, holdout, excluded(carried), dismissed(carried))
        revert = Revert(decision_id=next(r.id for r in log if r.decision.kind == "dismiss_finding"))
        refusal, state, out = checked(revert, carried, log)
        assert data(out["working"])["n_rows"] == 128
        gone = carried.draw(fold([*log, DecisionRecord(id="u", seq=len(log) + 1,
                                                        at=T0 + timedelta(hours=1),
                                                        decision=revert)]))
        assert data(gone["working"])["n_rows"] == 100
        participants = placed(gone).index.to_numpy(dtype=np.int64)
        crossed = np.intersect1d(np.setxor1d(held_out(out), held_out(gone)), participants)
        if holdout:
            assert len(crossed) > 0
            assert refusal is not None and refusal.code == "rows_after_the_seal"
            assert refusal.message == REFUSED_WORDS
            assert refusal.message.endswith(CLOSE_IN)
        else:
            assert len(crossed) == 0 and refusal is None


@pytest.mark.parametrize("layout", ["as_run", "last"])
def test_until_the_split_is_drawn_again_no_preview_reads_a_participant_the_undoing_renumbers(
        tmp_path, layout):
    """The plasma panel's QCs carry no outcome. The exclusion recorded before a split that holds
    20% out, then undone: by construction the redraw holds every participant on its side (their
    source rows), but the rows are numbered again, and until the split is drawn again the previews
    read every working row but the old split's sealed numbers, and train on the cohort's rows but
    those (``server.service.training_rows``). The held-out participants are the source rows those
    numbers named in the old working table. With the QCs as run they come back before held-out
    participants, and the window would read some: each undoing is refused, in words that say the
    rows come back. With every QC after every participant no participant is renumbered, the window
    reads none, and each is accepted."""
    source = PLASMA if layout == "as_run" else _laid_out(tmp_path, layout)
    project = Project(source, tmp_path / "p", PLASMA_BASE)
    try:
        log = logged(project, 0.2, excluded(project))
        for how in UNDOINGS:
            decision = undoing(how, project, log)
            refusal, state, old = checked(decision, project, log)
            sealed = row_previews.sealed_rows(old["split"])
            named = row_map(old["working"]).set_index("row_id")["source_row_id"]
            participants = set(named.loc[sealed].tolist())
            assert len(participants) == 20
            after = fold([*log, DecisionRecord(id="u", seq=len(log) + 1,
                                               at=T0 + timedelta(hours=1), decision=decision)])
            new = project.run.run(after, upto=["cohort"])
            n_rows = int(data(new["working"])["n_rows"])
            assert n_rows == 128
            pd.testing.assert_frame_equal(placed(project.draw(after)), placed(old))
            preview = consequences.PreviewContext(
                project_id="p", state=state, datastore=SimpleNamespace(n_rows=n_rows),
                artifact=lambda _: None, training_row_ids=None, cohort_row_ids=None,
                sealed_row_ids=sealed)
            cohort = new["cohort"].frames["rows"]["row_id"].to_numpy(dtype=np.int64)
            training = np.setdiff1d(cohort, sealed)
            source_of = row_map(new["working"]).set_index("row_id")["source_row_id"]
            read = set().union(*(participants & set(source_of.loc[pool].tolist())
                                 for pool in (preview.unsealed_row_ids(), training)))
            assert bool(read) == (layout == "as_run")
            if layout == "as_run":
                assert refusal is not None and refusal.code == "rows_after_the_seal"
                assert refusal.message == BACK_RENUMBERED_WORDS
            else:
                assert refusal is None
    finally:
        project.run.close()


@pytest.fixture(scope="module")
def plasma(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plasma")
    project = Project(PLASMA, folder / "p", PLASMA_BASE)
    yield project
    project.run.close()


def test_an_answer_that_changes_no_reference_rows_is_not_held(plasma):
    """While 20% are held out and the QC rows are in the table (no exclusion recorded), dismissing
    or deferring the pooled-QC finding keeps every row where it is: accepted, with the tables at
    hand and with nothing to read (only the state). With the exclusion applied and nothing to read,
    the guard cannot show the seal stays, so its dismissal is refused; on the base it was
    accepted."""
    log = logged(plasma, 0.2)
    for decision in (DismissFinding(finding_id=plasma.qc["id"]),
                     DeferFinding(finding_id=plasma.qc["id"], to="exclusions")):
        assert checked(decision, plasma, log)[0] is None
        validate(decision, {"state": fold(log)})
    sealed = fold(logged(plasma, 0.2, excluded(plasma)))
    with pytest.raises(Refusal) as caught:
        validate(DismissFinding(finding_id=plasma.qc["id"]), {"state": sealed})
    assert caught.value.code == "rows_after_the_seal"
    assert caught.value.message == BACK_WORDS
    validate(DismissFinding(finding_id=plasma.qc["id"]), {"state": sealed.model_copy(
        update={"split": sealed.split.model_copy(update={"holdout": 0.0})})})


def test_with_nothing_to_read_the_refusal_says_which_way_the_rows_move():
    """No table at hand, 20% held out: the refusal names the change the answer makes. Rows that
    would leave are "excluding them now" (today's words, kept), rows that would come back
    "bringing them back now", and an answer that takes some out and brings others back (another
    level of the same column in place of the one excluded) "changing which of them leave now",
    each closing with what keeping the rows as they are means there."""
    from turbotab.core import repairs  # noqa: F401 - registers the reference-rows family
    from turbotab.core.methods import qc_drift as Q

    def exclusion(level: str) -> dict:
        return {"kind": "apply_repair", "finding_id": Q.FINDING, "option": "exclude_rows",
                "params": {"column": "sample_type", "levels": [level]}}

    def refused(decision: dict, findings: dict) -> str:
        state = ProjectState(lens=["metabolomics"], target="disease", findings=findings,
                             split={"holdout": 0.2, "seed": 0, "folds": 5})
        with pytest.raises(Refusal) as caught:
            validate(decision, {"state": state})
        assert caught.value.code == "rows_after_the_seal"
        return caught.value.message

    applied = {Q.FINDING: {"action": "applied", "option": "exclude_rows",
                           "params": exclusion(QC_LEVEL)["params"]}}
    assert refused(exclusion(QC_LEVEL), {}) == REFUSED_WORDS
    assert refused({"kind": "dismiss_finding", "finding_id": Q.FINDING}, applied) == BACK_WORDS
    assert refused(exclusion("blank"), applied) == (
        "The held-out rows are drawn over the rows the analysis holds, so reference rows leave "
        "before the seal; changing which of them leave now would draw it again. Start a new "
        "analysis, or keep the rows as they are.")
