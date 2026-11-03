"""Reference rows that leave after the split: refused only where leaving would move the draw.

Under Estimate the engine records its own split at the missing answer, holding nothing out (P0.6),
so every pooled-QC answer reached after it was refused ``rows_after_the_seal`` on the plasma panel,
and PQN against the pooled QCs (which waits for that answer, ``qc_rows_unsettled``) with it: a dead
end the pilot's analysts left by reverting the missing answer and the split.

The held-out rows are drawn over the rows with the outcome recorded, sorted by row id and drawn by
position (``stages.rows.draw_split``); the folds over the training rows the same way. So rows that
leave with the outcome blank on them, as the plasma panel's pooled QCs do, move no participant
across the seal or between folds: the truth here is construction, the same participants' places in
the draws with and without the rows leaving, mapped to source rows. But the rows that stay are
numbered again, and the seal names rows by number: until the split is drawn again over the new
working table, the previews read every row but the old split's sealed numbers, which then name
other participants. So while rows are held out the guard refuses where a row that leaves comes
before a row in the draw (it renumbers it), as well as where a row that leaves has its outcome
recorded (a ``QC`` level of the outcome, the audit's export; or an outcome written on the QC rows).
A grouped draw names a drawn row with no identifier by its working row id (``__missing_{rid}``),
so that guard holds it still too. An answer that brings the rows back (the exclusion dismissed,
deferred or reverted) or takes them out again (a revert of its dismissal) is held by the same guard
(``test_reference_rows_undone_holds_the_seal.py``).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import consequences, row_previews
from turbotab.core.decisions import ApplyRepair, ProjectState, Refusal, validate
from turbotab.core.methods import qc_drift as Q
from turbotab.core.stages.working import row_map
from turbotab.core.tests.acceptance.omics_chains import metabolomics_run
from turbotab.core.tests.graph_runner import GraphRun

REPO = Path(__file__).resolve().parents[3]
PLASMA = REPO / "research" / "swarm" / "fixtures" / "plasma_panel.csv"
QC_LEVEL = "pooled_qc"
REFUSED_WORDS = ("The held-out rows are drawn over the rows the analysis holds, so reference rows "
                 "leave before the seal; excluding them now would draw it again. Start a new "
                 "analysis, or keep them in and say so.")
RENUMBERED_WORDS = ("The held-out rows are drawn and named by their row numbers, so reference rows "
                    "leave before the seal; excluding them now would renumber the rows with "
                    "`disease` recorded that come after them. Start a new analysis, or keep them "
                    "in and say so.")
PLASMA_BASE = {"lens": ["metabolomics"], "orientation": "sample_major", "target": "disease",
               "grain": {"grain": "one_row_per_unit"}}


def data(artifact: Any) -> Any:
    return getattr(artifact, "data", artifact)


def ctx_of(state: ProjectState, out: dict, *, split: bool = True) -> dict:
    """The validators' context, as the server builds it: the columns and their summaries, the
    state before the answer, each fresh public artifact and each fresh stage's files. ``split``
    False: the split is still recomputing (no fresh split artifact)."""
    info = {c["name"]: c for c in data(out["ingest"])["columns"]}

    def fresh(name: str) -> Any:
        if name not in out or (name == "split" and not split):
            return None
        return out[name]

    return {"state": state, "columns": list(info), "column_info": info,
            "artifact": lambda name: data(fresh(name)) if fresh(name) is not None else None,
            "bundle": fresh}


def findings_of(out: dict) -> dict[str, dict]:
    return {f["id"]: f for f in data(out["findings"])["findings"]}


def option(finding: dict, key: str) -> dict:
    """The first offer of ``key`` (QC-RLSC is offered per batch first, then over the whole run)."""
    return next(o for o in finding["repairs"] if o["key"] == key)


def answer(finding: dict, key: str) -> ApplyRepair:
    d = option(finding, key)["decision"]
    return ApplyRepair(finding_id=d["finding_id"], option=d["option"], params=d["params"])


def applied(finding: dict, key: str) -> dict:
    return {finding["id"]: {"action": "applied", "option": key,
                            "params": option(finding, key)["decision"]["params"]}}


def placed(out: dict) -> pd.DataFrame:
    """Each analyzed row's side of the seal and fold, by its source row."""
    frame = out["split"].frames["assignment"].merge(row_map(out["working"]), on="row_id")
    return frame.set_index("source_row_id")[["partition", "fold"]].sort_index()


class Project:
    """One table through the real stage graph, in-process: its answers before the pooled-QC
    answer (``base``), and the runs it is drawn under."""

    def __init__(self, source: Path, folder: Path, base: dict):
        self.run = GraphRun(source, folder)
        self.base = base
        first = self.run.run(ProjectState.model_validate(base), upto=["findings"])
        self.findings = findings_of(first)
        self.qc = self.findings[Q.FINDING]

    def state(self, holdout: float, **extra: Any) -> ProjectState:
        return ProjectState.model_validate({
            **self.base, "missing": {"strategy": "complete_case"},
            "split": {"holdout": holdout, "seed": 0, "folds": 5}, **extra})

    def draw(self, state: ProjectState) -> dict:
        return self.run.run(state, upto=["split"])


@pytest.fixture(scope="module")
def plasma(tmp_path_factory):
    """The pilot's plasma panel: 100 participants and 28 pooled QCs, ``disease`` blank on every
    QC, one row per participant."""
    folder = tmp_path_factory.mktemp("plasma")
    project = Project(PLASMA, folder / "p", PLASMA_BASE)
    yield project
    project.run.close()


def test_the_plasma_qcs_carry_no_outcome():
    """The construction the acceptance rests on, read with pandas: every pooled QC's outcome is
    blank, every participant's is recorded."""
    df = pd.read_csv(PLASMA)
    qc = df["sample_type"].eq(QC_LEVEL)
    assert int(qc.sum()) == 28 and int((~qc).sum()) == 100
    assert df.loc[qc, "disease"].isna().all() and df.loc[~qc, "disease"].notna().all()


@pytest.mark.parametrize("holdout", [0.2, 0.0])
@pytest.mark.parametrize("key", ["qc_rlsc_lc", "exclude_rows"])
def test_the_pooled_qcs_leaving_moves_no_participant_across_the_seal_or_a_fold(
        plasma, holdout, key):
    """By construction: drawn with the QC rows in the table and with them gone (QC-RLSC, or the
    exclusion alone), every participant sits on the same side of the seal and in the same fold,
    mapped to source rows; the 28 QC rows are in neither draw."""
    gone = plasma.draw(plasma.state(holdout, findings=applied(plasma.qc, key)))
    kept = plasma.draw(plasma.state(holdout))
    assert data(gone["working"])["n_rows"] == 100 and data(kept["working"])["n_rows"] == 128
    df = pd.read_csv(PLASMA)
    participants = sorted(np.flatnonzero(df["sample_type"].ne(QC_LEVEL).to_numpy()).tolist())
    a, b = placed(gone), placed(kept)
    assert list(a.index) == list(b.index) == participants
    pd.testing.assert_frame_equal(a, b)
    assert int((a["partition"] == "holdout").sum()) == (20 if holdout else 0)


@pytest.mark.parametrize("key", ["qc_rlsc_lc", "exclude_rows"])
def test_a_pooled_qc_answer_after_the_engines_no_holdout_split_is_accepted(plasma, key):
    """The engine's own no-holdout split under Estimate, recorded before the pooled-QC answer: the
    QC rows carry no outcome, so the draw holds none of them, and nothing is held out, so no seal
    names a row by its number: the answer is accepted (on the base it was refused
    ``rows_after_the_seal``), with the split fresh or recomputing."""
    before = plasma.state(0.0)
    out = plasma.draw(before)
    validate(answer(plasma.qc, key), ctx_of(before, out))
    validate(answer(plasma.qc, key), ctx_of(before, out, split=False))


def _laid_out(folder: Path, layout: str, cut: int = 0) -> Path:
    """The plasma panel with its rows in another order (``run_order`` keeps each injection's
    place): ``first`` puts the 28 pooled QCs before every participant, ``last`` after every one;
    ``participant_first`` moves the first participant ahead of the first QC and keeps every other
    row as run; ``between`` puts every QC after the participants on source rows up to ``cut`` and
    before the participants after it."""
    df = pd.read_csv(PLASMA)
    qc = df["sample_type"].eq(QC_LEVEL)
    if layout == "participant_first":
        lead = int(np.flatnonzero(~qc.to_numpy())[0])
        parts = [df.iloc[[lead, *(i for i in range(len(df)) if i != lead)]]]
    elif layout == "between":
        early = ~qc & (df.index <= cut)
        parts = [df[early], df[qc], df[~qc & ~early]]
    else:
        parts = [df[qc], df[~qc]] if layout == "first" else [df[~qc], df[qc]]
    path = folder / f"plasma_{layout}.csv"
    pd.concat(parts).to_csv(path, index=False)
    return path


def _sealed(project: Project) -> tuple[ProjectState, dict, Any, set[int]]:
    """``project`` drawn with 20% held out: the state, the run, the split's sealed row ids, and
    the source rows they name in that run's working table (the held-out participants)."""
    before = project.state(0.2)
    old = project.draw(before)
    sealed = row_previews.sealed_rows(old["split"])
    named = row_map(old["working"]).set_index("row_id")["source_row_id"]
    return before, old, sealed, set(named.loc[sealed].tolist())


@pytest.mark.parametrize("layout", ["as_run", "first", "participant_first", "between", "last"])
@pytest.mark.parametrize("key", ["qc_rlsc_lc", "exclude_rows"])
def test_until_the_split_is_drawn_again_no_preview_reads_a_held_out_participant(
        tmp_path, layout, key):
    """The window between the working table and the split: a pooled-QC answer accepted while 20%
    are held out numbers the rows that stay again, and until the split is drawn again over them the
    previews and the evidence read every working row but the newest split's sealed row ids
    (``consequences.PreviewContext.unsealed_row_ids``; ``server.service.sealed_rows``, fresh or
    still recomputing) and train on the new cohort's rows but those
    (``server.service.training_rows``). By construction the held-out participants are the source
    rows the old split's sealed ids named in the old working table, and neither pool may hold one.
    The window is built for every layout, as it would open were the answer accepted.

    With the QCs as run, all first, or as run behind a first participant (held out on this draw),
    a QC comes before held-out participants, and the window would read them (18 of the 20 as run
    or all first, 17 behind the first participant): the answer is refused. A guard that refused
    only a leaving row with its outcome recorded accepted all three; one that compares the leaving
    rows against the first row in the draw, not the last, accepts ``participant_first``. With
    every QC after the last held-out participant but before five training participants
    (``between``), the window would read none, and the answer is refused all the same: the guard
    refuses wherever a row with the outcome recorded is renumbered, not only a held-out one, so
    accepting there would be a change of rule. With every QC after every participant, no row in
    the draw is renumbered: the answer is accepted, and the window reads no held-out participant."""
    cut = 0
    if layout == "between":  # the last held-out participant's source row, the panel as run
        panel = Project(PLASMA, tmp_path / "as_run", PLASMA_BASE)
        try:
            cut = max(_sealed(panel)[3])
        finally:
            panel.run.close()
    source = PLASMA if layout == "as_run" else _laid_out(tmp_path, layout, cut)
    project = Project(source, tmp_path / "p", PLASMA_BASE)
    try:
        before, old, sealed, held_out = _sealed(project)
        assert len(held_out) == 20
        df = pd.read_csv(source)
        qc_rows = np.flatnonzero(df["sample_type"].eq(QC_LEVEL).to_numpy())
        last_participant = int(np.flatnonzero(df["sample_type"].ne(QC_LEVEL).to_numpy())[-1])
        if layout == "participant_first":
            assert 0 in held_out and int(qc_rows[0]) == 1
        if layout == "between":  # every held-out row before every QC, five drawn rows after them
            assert max(held_out) < int(qc_rows[0])
            assert last_participant - int(qc_rows[-1]) == 5
        outcomes = []
        for split in (True, False):
            try:
                validate(answer(project.qc, key), ctx_of(before, old, split=split))
                outcomes.append(None)
            except Refusal as caught:
                outcomes.append(caught)
        refusal = outcomes[0]
        assert (outcomes[1] is None) == (refusal is None)
        new = project.run.run(project.state(0.2, findings=applied(project.qc, key)),
                              upto=["cohort"])
        n_rows = int(data(new["working"])["n_rows"])
        assert n_rows == 100
        preview = consequences.PreviewContext(
            project_id="p", state=before, datastore=SimpleNamespace(n_rows=n_rows),
            artifact=lambda _: None, training_row_ids=None, cohort_row_ids=None,
            sealed_row_ids=sealed)
        cohort = new["cohort"].frames["rows"]["row_id"].to_numpy(dtype=np.int64)
        training = np.setdiff1d(cohort, sealed)  # service.training_rows, the split recomputing
        source_of = row_map(new["working"]).set_index("row_id")["source_row_id"]
        read = set().union(*(held_out & set(source_of.loc[pool].tolist())
                             for pool in (preview.unsealed_row_ids(), training)))
        assert bool(read) == (layout in ("as_run", "first", "participant_first"))
        assert refusal is not None or not read
        assert (refusal is None) == (layout == "last")
        if refusal is not None:
            assert refusal.code == "rows_after_the_seal"
            assert refusal.message == RENUMBERED_WORDS
    finally:
        project.run.close()


def test_pqn_against_the_pooled_qcs_follows_the_qc_answer_after_the_split(plasma):
    """``qc_rows_unsettled`` asks for the pooled-QC answer first; with the split recorded that
    answer is now accepted, and PQN against the QCs after it."""
    before = plasma.state(0.0)
    out = plasma.draw(before)
    pqn = answer(findings_of(out)["omics_scale"], "qc_pqn_log2")
    with pytest.raises(Refusal) as caught:
        validate(pqn, ctx_of(before, out))
    assert caught.value.code == "qc_rows_unsettled"
    validate(answer(plasma.qc, "qc_rlsc_lc"), ctx_of(before, out))
    after = plasma.state(0.0, findings=applied(plasma.qc, "qc_rlsc_lc"))
    validate(pqn, ctx_of(after, plasma.draw(after)))


def _variant(folder: Path, how: str) -> Path:
    """The plasma panel with its pooled QCs in the draw: ``class`` names the participants' classes
    and the QCs in one column, the outcome (the audit's export shape); ``outcome`` writes a recorded
    ``disease`` on every QC row."""
    df = pd.read_csv(PLASMA)
    qc = df["sample_type"].eq(QC_LEVEL)
    if how == "class":
        df.insert(1, "Class", np.where(qc, "QC", np.where(df["disease"] == 1, "Case", "Control")))
        df = df.drop(columns=["sample_type", "disease"])
    else:
        df.loc[qc, "disease"] = 0
    path = folder / f"plasma_{how}.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("how", ["class", "outcome"])
def test_rows_held_out_and_a_leaving_row_with_its_outcome_recorded_still_refuse(tmp_path, how):
    """Where the QC rows carry the outcome they are in the draw: while rows are held out the answer
    is refused with today's words, and under cross-validation alone, holding nothing out, it is
    accepted."""
    target = "Class" if how == "class" else "disease"
    project = Project(_variant(tmp_path, how), tmp_path / "p",
                      {"lens": ["metabolomics"], "orientation": "sample_major", "target": target,
                       "grain": {"grain": "one_row_per_unit"}})
    try:
        leaving = option(project.qc, "exclude_rows")["decision"]["params"]
        assert leaving == {"column": "Class" if how == "class" else "sample_type",
                           "levels": ["QC" if how == "class" else QC_LEVEL]}
        for key in ("qc_rlsc_lc", "exclude_rows"):
            sealed = project.state(0.2)
            with pytest.raises(Refusal) as caught:
                validate(answer(project.qc, key), ctx_of(sealed, project.draw(sealed)))
            assert caught.value.code == "rows_after_the_seal"
            assert caught.value.message == REFUSED_WORDS
            folds_only = project.state(0.0)
            validate(answer(project.qc, key), ctx_of(folds_only, project.draw(folds_only)))
    finally:
        project.run.close()


def _repeated(folder: Path, blank_ids: int, layout: str) -> Path:
    """Chain 3's run, smaller (two batches of 40 injections, 40 features): two samples from each
    participant, grouped by ``participant_id``; the pooled QCs have no outcome and no identifier.
    ``blank_ids`` participant rows, spread across the run, lose their identifier. ``layout``
    ``last`` puts every QC row after every participant row (``injection_order`` keeps the run)."""
    frame, _ = metabolomics_run(batches=2, per_batch=40, p=40, near=4)
    qc = frame["sample_type"].eq("QC")
    study = np.flatnonzero((~qc).to_numpy())
    if blank_ids:
        frame.loc[study[:: len(study) // blank_ids][:blank_ids], "participant_id"] = None
    if layout == "last":
        frame = pd.concat([frame[~qc], frame[qc]])
    path = folder / f"repeated_{blank_ids}_{layout}.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(params=[(0, "as_run"), (20, "as_run"), (0, "last"), (20, "last")],
                ids=["every_id", "twenty_blank_ids", "every_id_qcs_last", "twenty_blank_ids_qcs_last"])
def repeated(request, tmp_path):
    blank_ids, layout = request.param
    project = Project(_repeated(tmp_path, blank_ids, layout), tmp_path / "p",
                      {"lens": ["metabolomics"], "orientation": "sample_major", "target": "case",
                       "grain": {"grain": "repeated", "id_column": "participant_id"},
                       "unit": "row", "repeat_kind": {"repeat_kind": "repeats"}})
    project.blank_ids, project.layout = blank_ids, layout
    yield project
    project.run.close()


def test_a_grouped_draw_holds_still_where_no_row_in_it_is_renumbered(repeated):
    """By construction, with 20% held out and the draw grouped by ``participant_id``: a drawn row
    with no identifier is a unit of its own named by its working row id. With the QCs as run they
    come before rows in the draw, and leaving renumbers them: with every participant row identified
    the draw holds still all the same, with twenty unidentified it moves (rows change side or
    fold), and either way the answer is refused, with the split recomputing too. With every QC
    after every participant no row in the draw is renumbered, and the draw holds still with twenty
    unidentified too: the answer is accepted (on the base, refused there). Under cross-validation
    alone it is accepted in each."""
    sealed = repeated.state(0.2)
    out = repeated.draw(sealed)
    assert data(out["split"])["grouped_by"] == "participant_id"
    gone = placed(repeated.draw(repeated.state(0.2, findings=applied(repeated.qc, "qc_rlsc_lc"))))
    kept = placed(out)
    assert list(gone.index) == list(kept.index)
    moved = int(((gone["partition"] != kept["partition"]) | (gone["fold"] != kept["fold"])).sum())
    assert (moved > 0) == (repeated.blank_ids > 0 and repeated.layout == "as_run")
    if repeated.layout == "as_run":
        for split in (True, False):
            with pytest.raises(Refusal) as caught:
                validate(answer(repeated.qc, "qc_rlsc_lc"), ctx_of(sealed, out, split=split))
            assert caught.value.code == "rows_after_the_seal"
            assert caught.value.message == RENUMBERED_WORDS.replace("`disease`", "`case`")
    else:
        validate(answer(repeated.qc, "qc_rlsc_lc"), ctx_of(sealed, out))
    folds_only = repeated.state(0.0)
    validate(answer(repeated.qc, "qc_rlsc_lc"), ctx_of(folds_only, repeated.draw(folds_only)))


def test_with_nothing_to_read_the_rows_held_out_still_refuse():
    """No table at hand (only the state): while rows are held out the guard cannot show the seal
    stays, so it refuses; holding nothing out, it accepts."""
    from turbotab.core import repairs  # noqa: F401 - registers the reference-rows family

    decision = {"kind": "apply_repair", "finding_id": Q.FINDING, "option": "exclude_rows",
                "params": {"column": "sample_type", "levels": [QC_LEVEL]}}
    sealed = ProjectState(lens=["metabolomics"], target="disease",
                          split={"holdout": 0.2, "seed": 0, "folds": 5})
    with pytest.raises(Refusal) as caught:
        validate(decision, {"state": sealed})
    assert caught.value.code == "rows_after_the_seal"
    validate(decision, {"state": sealed.model_copy(update={"split": sealed.split.model_copy(
        update={"holdout": 0.0})})})
