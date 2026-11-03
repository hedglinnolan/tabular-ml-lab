"""A blank is missing in the working table, whatever wrote it (run_27's stuck export).

QC-RLSC (``methods/qc_drift.correct_table``) wrote the corrected features back with each blank as a
float NaN, and the working stage counted ``n_missing`` as Arrow's null count, so every reader of it
(complete cases, the missing-values previews, the row floor) saw no blank on the plasma panel while
the design step, reading pandas, saw 76 of the 100 participants with one: complete cases "dropped
0 rows", and the design refused. Ingest already reads a NaN as missing
(``datastore._null_the_nans``); now the correction writes nulls and the working stage turns any NaN
left into a null before it counts, as ingest does.

The truth is pandas on the raw table: each column's blanks on the rows that stay, and the rows with
no blank among the cohort's predictors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.methods import qc_drift as Q
from turbotab.core.stages.rows import _columns_with_missing
from turbotab.core.stages.working import row_map
from turbotab.core.tests.acceptance.omics_chains import metabolomics_run
from turbotab.core.tests.graph_runner import GraphRun

REPO = Path(__file__).resolve().parents[3]
PLASMA = REPO / "research" / "swarm" / "fixtures" / "plasma_panel.csv"


def data(artifact: Any) -> Any:
    return getattr(artifact, "data", artifact)


def small_run() -> pd.DataFrame:
    """Chain 3's run, smaller: two batches of 40 injections, 40 features, four of them near the
    detection limit and blank below it (on QCs and on study samples)."""
    frame, _ = metabolomics_run(batches=2, per_batch=40, p=40, near=4)
    return frame


def features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("mz_")]


def test_the_qc_correction_writes_each_blank_as_missing(tmp_path):
    """``correct_table`` on a constructed table: a non-detection stays as it is (QC-RLSC divides
    only detected values), so each feature's null count after the correction is pandas' blank count
    before it, and no NaN is left."""
    frame = small_run()
    feats = features(frame)
    blanks = frame[feats].isna().sum()
    assert int(blanks.sum()) > 0
    path = tmp_path / "table.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    plan = Q.QCPlan(qc_column="sample_type", qc_levels=("QC",), features=tuple(feats),
                    order_column="injection_order", batch_column="batch")
    Q.correct_table(path, plan, tmp_path / Q.QC_FILE)
    table = pq.read_table(path)
    for c in feats:
        col = table.column(c)
        assert col.null_count == int(blanks[c]), c
        assert not pc.any(pc.is_nan(col)).as_py(), c


def _graph_counts(folder: Path, source: Path, base: dict, key: str = "qc_rlsc_lc") -> dict:
    """The table through the real stage graph with the pooled-QC answer recorded: the working
    artifact and the cohort's."""
    run = GraphRun(source, folder)
    try:
        found = {f["id"]: f for f in data(run.run(ProjectState.model_validate(base),
                                                  upto=["findings"])["findings"])["findings"]}
        qc = found[Q.FINDING]
        offer = next(o for o in qc["repairs"] if o["key"] == key)
        state = ProjectState.model_validate({
            **base, "findings": {qc["id"]: {"action": "applied", "option": key,
                                            "params": offer["decision"]["params"]}},
            "missing": {"strategy": "complete_case"},
            "split": {"holdout": 0.0, "seed": 0, "folds": 5}})
        out = run.run(state, upto=["cohort"])
        rows = out["cohort"].frames["rows"].merge(row_map(out["working"]), on="row_id")
        return {"working": data(out["working"]), "cohort": data(out["cohort"]),
                "kept": sorted(rows["source_row_id"].tolist())}
    finally:
        run.close()


def _nan_writer(real):
    """``correct_table`` as the base wrote it: every blank of the features a float NaN."""

    def write(path: Any, plan: Any, qc_out: Any) -> dict:
        record = real(path, plan, qc_out)
        table = pq.read_table(path)
        for c in plan.features:
            if c in table.column_names:
                i = table.column_names.index(c)
                table = table.set_column(i, c, pc.fill_null(table.column(c), float("nan")))
        pq.write_table(table, path, compression="zstd")
        return record

    return write


@pytest.mark.parametrize("writer", ["correction", "nan_writer"])
def test_the_working_table_counts_each_blank_on_the_rows_that_stay(tmp_path, monkeypatch, writer):
    """A constructed run through QC-RLSC: the working table's ``n_missing`` of every column is
    pandas' blank count on the participant rows (the pooled QCs leave), and complete cases keep
    exactly the rows pandas finds complete among the cohort's predictors. ``nan_writer``: the
    correction writes NaN (as the base did), and the working stage still counts them, as ingest
    does."""
    if writer == "nan_writer":
        monkeypatch.setattr(Q, "correct_table", _nan_writer(Q.correct_table))
    frame = small_run()
    path = tmp_path / "run.csv"
    frame.to_csv(path, index=False)
    roles = {c: "exposure" for c in features(frame)}
    roles.update({"participant_id": "identifier", "sample_id": "excluded",
                  "injection_order": "excluded", "batch": "excluded"})
    got = _graph_counts(tmp_path / "p", path, {
        "lens": ["metabolomics"], "orientation": "sample_major", "target": "case",
        "grain": {"grain": "repeated", "id_column": "participant_id"}, "unit": "row",
        "repeat_kind": {"repeat_kind": "repeats"}, "roles": roles})
    stay = frame[frame["sample_type"].ne("QC")]
    blanks = stay.isna().sum()
    columns = {c["name"]: int(c["n_missing"]) for c in got["working"]["columns"]}
    assert got["working"]["n_rows"] == len(stay)
    assert columns == {c: int(blanks[c]) for c in columns}
    predictors = list(got["cohort"]["predictors"])
    assert predictors and set(predictors) <= set(features(frame))
    complete = stay.index[stay[predictors].notna().all(axis=1)].tolist()
    assert got["kept"] == complete
    assert 0 < len(complete) < len(stay)


def test_the_plasma_panel_through_the_qc_repair_counts_its_76_incomplete_participants(tmp_path):
    """The pilot's plasma panel through QC-RLSC (run_27's answer): the working table counts each
    column's blanks on the 100 participants as pandas does on the raw file, the columns with a
    blank are the ones pandas finds, and complete cases keep the 24 participants pandas finds
    complete among the cohort's predictors, dropping 76 (on the base: 0)."""
    raw = pd.read_csv(PLASMA)
    feats = features(raw)
    roles = {c: "exposure" for c in feats}
    roles.update({"age": "covariate", "sex": "covariate", "bmi": "covariate",
                  "sample_id": "identifier"})
    got = _graph_counts(tmp_path / "p", PLASMA, {
        "lens": ["metabolomics"], "orientation": "sample_major", "target": "disease",
        "grain": {"grain": "one_row_per_unit"}, "roles": roles})
    stay = raw[raw["sample_type"].ne("pooled_qc")]
    blanks = stay.isna().sum()
    columns = {c["name"]: int(c["n_missing"]) for c in got["working"]["columns"]}
    assert columns == {c: int(blanks[c]) for c in columns}
    assert set(_columns_with_missing(got["working"])) == set(blanks.index[blanks > 0])
    predictors = list(got["cohort"]["predictors"])
    complete = stay.index[stay[predictors].notna().all(axis=1)].tolist()
    assert got["kept"] == complete and len(complete) == 24
    step = next(s for s in got["cohort"]["steps"] if s["key"] == "complete_cases")
    assert step["dropped"] == 76 and got["cohort"]["n_final"] == 24


def _written(path: Path, nested: bool) -> pa.Table:
    """A table as the working stage writes it: ``__row_id``, ``x`` with a NaN on row 1 and ``y``
    with a null on row 0 and a NaN on row 2. ``nested`` puts a struct of two whole numbers before
    ``x`` and a list of them between ``x`` and ``y``, so a field's place among the top-level fields
    is not its place among the leaf columns Parquet stores (``x`` is field 2 and leaf 3, ``y`` field
    4 and leaf 5)."""
    columns: dict[str, Any] = {"__row_id": pa.array([0, 1, 2, 3], pa.int64())}
    if nested:
        columns["visit"] = pa.array([{"a": i, "b": 10 * i} for i in range(4)],
                                    pa.struct([("a", pa.int64()), ("b", pa.int64())]))
    columns["x"] = pa.array([1.5, float("nan"), 2.5, 3.5], pa.float64())
    if nested:
        columns["tags"] = pa.array([[1], [], [2, 3], None], pa.list_(pa.int64()))
    columns["y"] = pa.array([None, 4.0, float("nan"), 5.0], pa.float64())
    table = pa.table(columns)
    pq.write_table(table, path)
    return table


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_each_float_columns_nans_are_read_by_the_columns_own_place(tmp_path, nested):
    """By construction: ``_nans_as_missing`` finds the float columns that hold a NaN and rewrites
    each NaN as a null, every other cell as it was. The flat table reads the same before the fix and
    after (no working table changes); with a nested column ahead of them, the base read the leaf
    columns at the floats' field places (the struct's ``b`` and the list), not ``x`` and ``y``."""
    import duckdb

    from turbotab.core.stages.working import _nans_as_missing

    path = tmp_path / "table.parquet"
    table = _written(path, nested)
    con = duckdb.connect()
    try:
        assert _nans_as_missing(con, path) == ["x", "y"]
    finally:
        con.close()
    got = pq.read_table(path)
    assert sorted(got.column_names) == sorted(table.column_names)  # the row id is written last
    assert got.column("x").to_pylist() == [1.5, None, 2.5, 3.5]
    assert got.column("y").to_pylist() == [None, 4.0, None, 5.0]
    for name in set(table.column_names) - {"x", "y"}:
        assert got.column(name).to_pylist() == table.column(name).to_pylist(), name


def test_a_failed_nan_rewrite_leaves_nothing_beside_the_table(tmp_path, monkeypatch):
    """``datastore._null_the_nans`` writes the table again to ``<table>.<token>.nan`` and moves it
    over the table. Where the move fails, the working stage's rewrite raises, the table is as it
    was, and no file is left beside it (ingest removes its own the same way); on the base the
    ``.nan`` file stayed."""
    import duckdb

    from turbotab.core import datastore
    from turbotab.core.stages.working import _nans_as_missing

    path = tmp_path / "table.parquet"
    table = _written(path, nested=False)

    def full(src: Any, dst: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(datastore.atomic, "replace", full)
    con = duckdb.connect()
    try:
        with pytest.raises(OSError):
            _nans_as_missing(con, path)
    finally:
        con.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["table.parquet"]

    def cells(t: pa.Table) -> dict[str, list[Any]]:  # a NaN spelled out: it equals nothing
        return {c: ["nan" if v != v else v for v in t.column(c).to_pylist()]
                for c in t.column_names}

    assert cells(pq.read_table(path)) == cells(table)
    assert cells(table)["x"] == [1.5, "nan", 2.5, 3.5]
