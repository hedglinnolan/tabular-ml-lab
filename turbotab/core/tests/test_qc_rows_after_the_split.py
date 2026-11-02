"""Reference rows that leave after the split: refused only where leaving would draw it again.

Under Estimate the engine records its own split at the missing answer, holding nothing out (P0.6),
so every pooled-QC answer reached after it was refused ``rows_after_the_seal`` on the plasma panel,
and PQN against the pooled QCs (which waits for that answer, ``qc_rows_unsettled``) with it: a dead
end the pilot's analysts left by reverting the missing answer and the split.

The held-out rows are drawn over the rows with the outcome recorded, sorted by row id and drawn by
position (``stages.rows.draw_split``); the folds over the training rows the same way. So rows that
leave with the outcome blank on them, as the plasma panel's pooled QCs do, move no participant
across the seal or between folds: the truth here is construction, the same participants' places in
the draws with and without the rows leaving, mapped to source rows. The guard still refuses where
the split holds rows out and a row that leaves has its outcome recorded (a ``QC`` level of the
outcome, the audit's export; or an outcome written on the QC rows), and where the draw groups by an
identifier some drawn row leaves blank: such a row is a unit of its own named by its working row id
(``__missing_{rid}``), which the leaving rows renumber, and the draw shown here moves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

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
    project = Project(PLASMA, folder / "p", {
        "lens": ["metabolomics"], "orientation": "sample_major", "target": "disease",
        "grain": {"grain": "one_row_per_unit"}})
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


@pytest.mark.parametrize("holdout", [0.0, 0.2])
@pytest.mark.parametrize("key", ["qc_rlsc_lc", "exclude_rows"])
def test_a_pooled_qc_answer_after_a_split_that_holds_none_of_them_is_accepted(plasma, holdout, key):
    """The engine's own no-holdout split under Estimate, and a 20% held-out set, recorded before
    the pooled-QC answer: the QC rows carry no outcome, so neither draw holds them, and the answer
    is accepted (on the base it was refused ``rows_after_the_seal``)."""
    before = plasma.state(holdout)
    out = plasma.draw(before)
    validate(answer(plasma.qc, key), ctx_of(before, out))
    validate(answer(plasma.qc, key), ctx_of(before, out, split=False))


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


def _repeated(folder: Path, blank_ids: int) -> Path:
    """Chain 3's run, smaller (two batches of 40 injections, 40 features): two samples from each
    participant, grouped by ``participant_id``; the pooled QCs have no outcome and no identifier.
    ``blank_ids`` participant rows, spread across the run, lose their identifier."""
    frame, _ = metabolomics_run(batches=2, per_batch=40, p=40, near=4)
    study = np.flatnonzero(frame["sample_type"].ne("QC").to_numpy())
    if blank_ids:
        frame.loc[study[:: len(study) // blank_ids][:blank_ids], "participant_id"] = None
    path = folder / f"repeated_{blank_ids}.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(params=[0, 20], ids=["every_id", "twenty_blank_ids"])
def repeated(request, tmp_path):
    project = Project(_repeated(tmp_path, request.param), tmp_path / "p",
                      {"lens": ["metabolomics"], "orientation": "sample_major", "target": "case",
                       "grain": {"grain": "repeated", "id_column": "participant_id"},
                       "unit": "row", "repeat_kind": {"repeat_kind": "repeats"}})
    project.blank_ids = request.param
    yield project
    project.run.close()


def test_a_grouped_draw_holds_still_unless_a_drawn_row_has_no_identifier(repeated):
    """By construction, with 20% held out and the draw grouped by ``participant_id``: with every
    participant row identified, the QC rows leaving moves no row across the seal or between folds,
    and the answer is accepted; with twenty rows unidentified, each a unit named by its working row
    id, the QC rows leaving renumbers them and the draw moves (rows change side or fold), so the
    answer is refused, with the split recomputing too. Under cross-validation alone it is accepted
    either way."""
    sealed = repeated.state(0.2)
    out = repeated.draw(sealed)
    assert data(out["split"])["grouped_by"] == "participant_id"
    gone = placed(repeated.draw(repeated.state(0.2, findings=applied(repeated.qc, "qc_rlsc_lc"))))
    kept = placed(out)
    assert list(gone.index) == list(kept.index)
    moved = int(((gone["partition"] != kept["partition"]) | (gone["fold"] != kept["fold"])).sum())
    if repeated.blank_ids:
        assert moved > 0
        for split in (True, False):
            with pytest.raises(Refusal) as caught:
                validate(answer(repeated.qc, "qc_rlsc_lc"), ctx_of(sealed, out, split=split))
            assert caught.value.code == "rows_after_the_seal"
    else:
        assert moved == 0 and (gone["fold"] == kept["fold"]).all()
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
