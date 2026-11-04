"""A query that groups by an alias reads a table holding a column of the alias's name (GA1).

DuckDB binds a name in ``GROUP BY`` to a table column before a select-list alias, so
``SELECT CAST(s AS VARCHAR) AS v ... GROUP BY v`` groups by the table's own ``v`` when it has one
and every other column's read raised ``BinderException: column "s" must appear in the GROUP BY
clause``. Each read below groups by position instead. Every expected value comes from pandas on
the same column or from the table's construction; each read is checked on a table holding a column
named like its alias, on one whose column of that name is the one read (where the old query
worked), and on an ordinary table, which must give the same.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from turbotab.core.datastore import ROW_ID, DataStore, ingest
from turbotab.core.decisions import FeatureTableSpec, ProjectState
from turbotab.core.readings import by_values

BUDGET = 1 << 30


def _store(tmp_path: Path, frame: pd.DataFrame, name: str = "t") -> tuple[Path, DataStore]:
    source = tmp_path / f"{name}.csv"
    frame.to_csv(source, index=False)
    dest = tmp_path / f"{name}.parquet"
    ingest(source, dest)
    return dest, DataStore(dest, BUDGET)


def _registry_verdict(values: pd.Series, *, as_text: bool):
    """The registry's one value test over each distinct value and its row count as pandas counts
    them (``value_counts``), in sorted order: the reading the store's query must hand it."""
    present = values.dropna()
    counts = (present.astype(str) if as_text else present.astype(float)).value_counts().sort_index()
    index = pd.Series(list(counts.index), dtype=object if as_text else float)
    return by_values("code_or_count", index, counts=counts.to_numpy(dtype=float))


# ── DataStore.text_numbers and DataStore.whole_numbers (alias v) ──────────────────

LABELS = ["a", "b", "a", "c", None, "b", "a"]
TEXT_NUMBERS = ["1.5", "2.25", ".", "1.5", "10", None, "2.25"]
CODES = [250.02, 307.1, 250.02, 307.51, 307.1, 250.02, None]  # ICD-9-CM codes with decimals


def _datastore_frame(shadow: str) -> pd.DataFrame:
    return pd.DataFrame({shadow: [1.25, 2.5, 3.75, 4.0, 5.5, 6.0, 7.0],
                         "s": LABELS, "t": TEXT_NUMBERS, "y": CODES})


@pytest.mark.parametrize("shadow", ["v", "w"], ids=["a column named v", "ordinary"])
def test_text_numbers_reads_each_text_column_beside_a_column_named_v(tmp_path, shadow):
    """``s`` holds three labels; ``t`` five numbers written as text (three distinct, 1.5 to 10)
    and one SAS ``.`` mark, by construction."""
    frame = _datastore_frame(shadow)
    _, store = _store(tmp_path, frame)
    with store:
        got = store.text_numbers(["s", "t"])
    assert got["s"]["text_numbers"] is False and got["s"]["labels"] is True
    assert got["s"]["n_values"] == frame["s"].nunique() == 3
    assert got["s"]["evidence"] == _registry_verdict(frame["s"], as_text=True).evidence
    t = got["t"]
    numbers = pd.to_numeric(frame["t"], errors="coerce").dropna()
    assert t["text_numbers"] is True and t["numbers"] == len(numbers) == 5
    assert t["distinct"] == numbers.nunique() == 3 and t["whole"] is False
    assert (t["min"], t["max"]) == (numbers.min(), numbers.max()) == (1.5, 10.0)
    assert t["marks"] == {".": 1}
    want = _registry_verdict(frame["t"], as_text=True)
    assert t["evidence"] == want.evidence and t["distinct"] == want.detail["distinct"]


def test_text_numbers_of_the_column_named_v_itself(tmp_path):
    """The read the old query already made right (it grouped by the column itself) is unchanged."""
    frame = pd.DataFrame({"v": TEXT_NUMBERS, "y": CODES})
    _, store = _store(tmp_path, frame)
    with store:
        got = store.text_numbers(["v"])["v"]
    want = _registry_verdict(frame["v"], as_text=True)
    assert got["text_numbers"] is True and got["evidence"] == want.evidence
    assert (got["distinct"], got["min"], got["max"]) == (3, 1.5, 10.0)


@pytest.mark.parametrize("shadow", ["v", "w"], ids=["a column named v", "ordinary"])
def test_whole_numbers_reads_a_column_with_decimals_beside_a_column_named_v(tmp_path, shadow):
    """``y`` holds three ICD-9-CM codes (250.02 three times, 307.1 twice, 307.51 once); whether its
    values settle "amount" is the registry's test over pandas' counts of them."""
    frame = _datastore_frame(shadow)
    _, store = _store(tmp_path, frame)
    with store:
        got = store.whole_numbers(["y"])["y"]
    y = frame["y"].dropna()
    assert got["whole"] is False and got["n_values"] == y.nunique() == 3
    assert (got["min"], got["max"]) == (float(np.min(y)), float(np.max(y))) == (250.02, 307.51)
    want = _registry_verdict(frame["y"], as_text=False)
    assert got["amount_by_values"] is bool(want.settles) and got["evidence"] == want.evidence


def test_whole_numbers_of_the_column_named_v_itself(tmp_path):
    frame = pd.DataFrame({"v": CODES, "s": LABELS})
    _, store = _store(tmp_path, frame)
    with store:
        got = store.whole_numbers(["v"])["v"]
    want = _registry_verdict(frame["v"], as_text=False)
    assert got["n_values"] == 3 and got["evidence"] == want.evidence
    assert got["amount_by_values"] is bool(want.settles)


# ── working.turn_plan's label check (alias v) ─────────────────────────────────


def _feature_table(tmp_path: Path, name: str, sample: str, genes: list[str]) -> tuple[Path, dict]:
    frame = pd.DataFrame({"gene": genes, "s1": np.arange(len(genes), dtype=float) + 0.5,
                          sample: np.arange(len(genes), dtype=float)[::-1] + 0.25})
    source = tmp_path / f"{name}.csv"
    frame.to_csv(source, index=False)
    dest = tmp_path / f"{name}.parquet"
    return dest, ingest(source, dest).to_dict()


@pytest.mark.parametrize("sample", ["v", "w"], ids=["a sample named v", "ordinary"])
def test_a_feature_table_with_a_sample_named_v_is_checked_for_its_names(tmp_path, sample):
    """Distinct feature names pass and are the label; two rows named ``g2`` (by construction) are
    refused with that name, whatever the samples are called."""
    from turbotab.core.stages.working import turn_plan

    declared = FeatureTableSpec(label="gene")
    con = duckdb.connect()
    try:
        plan = turn_plan(con, *_feature_table(tmp_path, "distinct", sample, ["g1", "g2", "g3"]),
                         declared)
        assert plan["refusal"] is None and plan["label_column"] == "gene"
        assert plan["samples"] == ["s1", sample]
        plan = turn_plan(con, *_feature_table(tmp_path, "dupe", sample, ["g1", "g2", "g2"]),
                         declared)
    finally:
        con.close()
    assert plan["code"] == "duplicate_features" and "`g2`" in plan["refusal"]


# ── working.text_amounts (alias v) ───────────────────────────────────────────


@pytest.mark.parametrize("shadow", ["v", "w"], ids=["a column named v", "ordinary"])
def test_a_text_column_confirmed_as_amounts_is_read_as_pandas_reads_it_beside_v(tmp_path, shadow):
    """A text BMI with a SAS ``.`` confirmed "amount" reads as ``pd.to_numeric``, ``.`` blank."""
    from turbotab.core.repairs import evaluate
    from turbotab.core.stages.working import text_amounts

    bmi = ["21.5", ".", "30.25", "21.5", None, "18"]
    path = tmp_path / "oriented.parquet"
    pd.DataFrame({"bmi": bmi, shadow: np.arange(len(bmi), dtype=float),
                  ROW_ID: np.arange(len(bmi))}).to_parquet(path)
    state = {"reading_confirmations": {"code_or_count:bmi": "amount"}}
    expressions = text_amounts(state, path, ["bmi"], {})
    assert set(expressions) == {"bmi"}
    got = evaluate(path, ["bmi"], expressions)["bmi"].to_numpy(dtype=float)
    want = pd.to_numeric(pd.Series(bmi), errors="coerce").to_numpy(dtype=float)
    np.testing.assert_array_equal(got, want)  # NaN == NaN here


# ── structure_previews: the aggregation's reshape (alias u) ────────────────────

# Five units by construction: P1, P3 and P4 have two records each and P5 three; P2 has one. The
# reshape shows the units with more than one record in file order while their records fit eight
# rows past the first two units: P1, P3, P4 (six rows); P5's three would make nine.
UNITS = ["P1", "P1", "P2", "P3", "P3", "P4", "P4", "P5", "P5", "P5"]
SHOWN = ["P1", "P3", "P4"]


def _records(shadow: str) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({"pid": UNITS,
                         shadow: np.round(rng.uniform(1, 9, len(UNITS)), 3),
                         "sbp": np.round(rng.uniform(100, 160, len(UNITS)), 2),
                         "y": [{"P1": 1.5, "P2": 2.5, "P3": 3.5, "P4": 4.5, "P5": 5.5}[p]
                               for p in UNITS]})


@pytest.mark.parametrize("shadow", ["u", "w"], ids=["a column named u", "ordinary"])
def test_the_reshape_preview_shows_the_units_with_several_records_beside_a_column_named_u(
        tmp_path, shadow):
    """The records shown are the units pandas finds with more than one record, in file order, and
    each combined row is pandas' mean of its unit's records."""
    from turbotab.core import structure_previews as SP
    from turbotab.core.consequences import PreviewContext, TableFocusView
    from turbotab.core.tests.graph_runner import GraphRun

    frame = _records(shadow)
    source = tmp_path / "records.csv"
    frame.to_csv(source, index=False)
    run = GraphRun(source, tmp_path / "project")
    try:
        s = ProjectState.model_validate({"lens": ["clinical"], "target": "y",
                                         "grain": {"grain": "repeated", "id_column": "pid"},
                                         "unit": "unit"})
        out = run.run(s, upto=["oriented", "structure", "findings"])
        recorded = s.model_copy(update={"aggregation": None})
        with DataStore(run.raw, BUDGET) as store:
            ctx = PreviewContext(project_id="p", state=recorded, datastore=store,
                                 artifact=out.get, training_row_ids=None, cohort_row_ids=None)
            views = SP.aggregation_views(SimpleNamespace(method="mean", outcome=None, columns={}),
                                         ctx)
    finally:
        run.close()
    (reshape,) = [v for v in views if isinstance(v, TableFocusView)]
    records, combined = reshape.story
    several = frame.groupby("pid", sort=False).size()
    assert [p for p in several.index if several[p] > 1][:3] == SHOWN  # the construction holds
    assert [r.unit for r in records.rows] == [p for p in UNITS if p in SHOWN]
    assert [r.unit for r in combined.rows] == SHOWN
    assert {shadow, "sbp"} <= set(combined.columns)
    means = frame[frame["pid"].isin(SHOWN)].groupby("pid")[[shadow, "sbp"]].mean()
    for row in combined.rows:
        for column in (shadow, "sbp"):
            assert row.values[column] == pytest.approx(means.loc[row.unit, column], rel=1e-12)
