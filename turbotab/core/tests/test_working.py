"""The working table (M2_CONTRACT §2), Tier A: row identity, the transpose, aggregation, the row map.

* a transposed assay table turned back round holds the same values, and its rows are named by the
  sample identifiers in the header;
* combining a unit's rows (mean, first, last, change) equals a plain pandas computation of the same
  thing, each column by what it holds (audit MA-14), on ``dietary_recalls.csv`` and
  ``clinical_longitudinal.csv``;
* the row map sends every source row to exactly one working row, the one for its unit;
* every stage after ``working`` reads the working table (a spy on the DataStore);
* with nothing structural recorded, the working table is the raw file, referenced and not copied.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.stages.working import (
    SAMPLE_COLUMN,
    StructureError,
    orientation_reading,
    row_map,
)
from turbotab.core.tests.graph_runner import GraphRun

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"
DIETARY = SAMPLES / "dietary_recalls.csv"
CLINICAL = SAMPLES / "clinical_longitudinal.csv"
METABOLOMICS = SAMPLES / "metabolomics_untargeted.csv"
NUMERIC = ("numeric", "integer")


def state(**slots) -> ProjectState:
    return ProjectState.model_validate(slots)


@pytest.fixture
def graph(tmp_path):
    runs: list[GraphRun] = []

    def make(source: Path) -> GraphRun:
        run = GraphRun(source, tmp_path / f"p{len(runs)}")
        runs.append(run)
        return run

    yield make
    for run in runs:
        run.close()


def table(bundle) -> pd.DataFrame:
    return pd.read_parquet(bundle.files["table.parquet"])


# ── the transpose ─────────────────────────────────────────────────────────────


def transposed_copy(folder: Path, with_text: bool = False) -> tuple[Path, pd.DataFrame]:
    """The metabolomics fixture exported features-in-rows: feature ids first, sample ids as headers."""
    m = pd.read_csv(METABOLOMICS).set_index("sample_id")
    block = m.select_dtypes("number")
    if with_text:
        block = pd.concat([m[["sample_type"]], block], axis=1)
    turned = block.T
    turned.index.name = "feature_id"
    path = folder / ("metabolomics_T_text.csv" if with_text else "metabolomics_T.csv")
    turned.to_csv(path)
    return path, block


@pytest.mark.parametrize("with_text", [False, True])
def test_turning_a_feature_major_table_round_trips_every_value_and_names_rows_by_sample(
        graph, tmp_path, with_text):
    source, original = transposed_copy(tmp_path, with_text)
    run = graph(source)
    assert run.info["n_rows"] == original.shape[1] and run.info["n_cols"] == 81
    out = run.run(state(lens=["metabolomics"], orientation="feature_major"), upto=["oriented"])
    oriented = out["oriented"]
    assert oriented.data["transposed"] is True and oriented.data["turn"]["label_column"] == "feature_id"
    turned = table(oriented)
    # 80 samples × (sample_id + every feature), rows named by the header, in header order
    assert turned.shape == (80, 1 + original.shape[1] + 1)
    assert list(turned[SAMPLE_COLUMN]) == list(original.index)
    assert list(turned["__row_id"]) == list(range(80))
    assert list(turned.columns[1:-1]) == list(original.columns)
    for column in original.columns:
        want, got = original[column], turned[column].set_axis(original.index)
        if column == "sample_type":
            assert got.tolist() == want.tolist()  # text stays text, value for value
            continue
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got.to_numpy(), want.to_numpy(dtype=float))  # NaN == NaN here
    # the DataStore reads it as described: the sidecar the stage wrote is the table's own
    assert oriented.data["n_rows"] == 80 and oriented.data["n_cols"] == 1 + original.shape[1]


def test_a_table_with_two_rows_of_one_name_is_not_turned(graph, tmp_path):
    source, _ = transposed_copy(tmp_path)
    frame = pd.read_csv(source)
    frame.loc[5, "feature_id"] = frame.loc[4, "feature_id"]
    dupe = tmp_path / "dupe.csv"
    frame.to_csv(dupe, index=False)
    run = graph(dupe)
    checked = run.run(state(lens=["metabolomics"]), upto=["oriented"])["oriented"].data["turn"]
    assert checked["code"] == "duplicate_features" and frame.loc[4, "feature_id"] in checked["refusal"]
    with pytest.raises(StructureError, match="both named"):
        run.run(state(lens=["metabolomics"], orientation="feature_major"), upto=["oriented"])


def test_the_streamed_reading_is_the_orientation_modules_reading(tmp_path):
    """The statistic streamed over the Parquet file is the same shape reading taken in memory.

    Since audit WP14 (IN-11) the shape is scale-aware and leaves feature-description columns (m/z,
    RT) out of the block (``turbotab.core.detectors.orientation``); the in-memory reference is that
    rule's ``read_frame`` with the names left out, so this checks streaming, not the rule."""
    from turbotab.core.datastore import ingest
    from turbotab.core.detectors import orientation as rule
    from turbotab.core.stages.working import is_feature_annotation

    source, _ = transposed_copy(tmp_path)
    for path in (source, METABOLOMICS, DIETARY, CLINICAL, SAMPLES / "genomics_expression.csv"):
        parquet = tmp_path / f"{path.stem}.parquet"
        info = ingest(path, parquet).to_dict()
        streamed = orientation_reading(parquet, info)
        frame = pd.read_csv(path)
        block = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
                 and not pd.api.types.is_bool_dtype(frame[c]) and not is_feature_annotation(c)
                 and frame[c].notna().mean() > 0.5]
        x = frame[block].to_numpy(dtype=float)
        memory = rule.shape(np.nanmean(x, axis=1), np.nanmean(x, axis=0), float(np.nanmin(x)),
                            float(np.nanmax(x)), len(frame), len(block))
        assert streamed["reading"] == memory["reading"], path.name
        assert streamed["ratio"] == pytest.approx(memory["ratio"], abs=2e-3), path.name


# ── aggregation ───────────────────────────────────────────────────────────────


def _kind(df: pd.DataFrame, key: str, c: str, numeric: bool) -> str:
    """What a column holds, in plain pandas (audit MA-14): constant within every unit; a category
    (text); a code (whole numbers, at most 10 values, each seen 3 times on average, not 0/1); an
    amount."""
    if not (df.groupby(key)[c].nunique() > 1).any():
        return "constant"
    if not numeric:
        return "category"
    x = df[c].dropna().astype(float)
    whole = bool((x == np.floor(x)).all())
    indicator = bool(x.isin([0.0, 1.0]).all())
    if whole and not indicator and x.nunique() <= 10 and len(x) >= 3 * x.nunique():
        return "code"
    return "amount"


def reference(oriented: pd.DataFrame, info: dict, *, key: str, method: str, target: str,
              outcome: str | None, order: str | None) -> pd.DataFrame:
    """The same combination in plain pandas: a unit's dated records ordered by time, then file
    order; undated ones after them, never first or last; each column by what it holds."""
    dtypes = {c["name"]: c["dtype"] for c in info["columns"]}
    df = oriented.copy()
    df["_rid"] = df.index.to_numpy()
    if order is not None:
        t = df[order]
        df["_t"] = t if dtypes[order] in NUMERIC or np.issubdtype(t.dtype, np.datetime64) \
            else pd.to_datetime(t, errors="coerce")
    else:
        df["_t"] = df["_rid"]
    df = df.sort_values([key, "_t", "_rid"], kind="stable", na_position="last")
    kinds = {c["name"]: _kind(df, key, c["name"], dtypes[c["name"]] in NUMERIC)
             for c in info["columns"] if c["name"] not in (key, target, order)}

    def mode(g: pd.DataFrame, c: str):
        counts = g[c].value_counts()
        top = set(counts[counts == counts.max()].index)
        return next(v for v in g[c] if v in top)  # ties: the earliest record

    rows = []
    for _, g in sorted(df.groupby(key, sort=False), key=lambda kv: kv[1]["_rid"].min()):
        dated = g[g["_t"].notna()]
        first = dated.iloc[0] if len(dated) else None
        last = dated.iloc[-1] if len(dated) else None

        def at(record, c):
            return np.nan if record is None else record[c]

        row = {}
        for c in info["columns"]:
            c = c["name"]
            if c == key:
                row[c] = g[c].iloc[0]
            elif c == target:
                values = g[c].dropna()
                if df.groupby(key)[c].nunique().max() <= 1:
                    row[c] = values.iloc[0] if len(values) else np.nan
                else:
                    row[c] = {"mean": values.mean(), "first": at(first, c),
                              "last": at(last, c)}[outcome]
            elif c == order:
                row[c] = at(last if method == "last" else first, c)
            else:
                rule = {"constant": "constant", "amount": method}.get(
                    kinds[c], {"mean": "mode", "change": "first"}.get(method, method))
                if rule == "constant":
                    values = g[c].dropna()
                    row[c] = values.iloc[0] if len(values) else np.nan
                elif rule == "mean":
                    row[c] = g[c].mean()
                elif rule == "mode":
                    row[c] = mode(g, c)
                elif rule == "change":
                    row[c] = (at(last, c) - at(first, c)) if len(dated) > 1 else np.nan
                else:
                    row[c] = at(last if rule == "last" else first, c)
        rows.append(row)
    return pd.DataFrame(rows)


def same(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b)) or pd.isna(b)
    if isinstance(a, (int, float, np.integer, np.floating)) and not isinstance(a, bool):
        return b is not None and not pd.isna(b) and math.isclose(float(a), float(b), rel_tol=1e-9,
                                                                 abs_tol=1e-9)
    return pd.Timestamp(a) == pd.Timestamp(b) if isinstance(a, (pd.Timestamp, np.datetime64)) else a == b


CASES = [
    (DIETARY, "participant_id", "hba1c", None),
    (CLINICAL, "subject_id", "progressed", "last"),
]


@pytest.mark.parametrize("source,key,target,outcome", CASES, ids=["dietary", "clinical"])
@pytest.mark.parametrize("method", ["mean", "first", "last", "change"])
def test_combining_each_units_rows_equals_pandas(graph, source, key, target, outcome, method):
    run = graph(source)
    combined = state(lens=["dietary" if source == DIETARY else "clinical"], target=target,
                     grain={"grain": "repeated", "id_column": key}, unit="unit",
                     aggregation={"method": method, "outcome": outcome})
    out = run.run(combined, upto=["working"])
    working, structure = out["working"], out["structure"]
    receipt = working.data["aggregation"]
    order = receipt["time_column"]
    assert order == structure["repeats"]["spacing"]["column"]  # the reading's date column orders them
    assert receipt["ordered_by"] == order
    from turbotab.core.datastore import DataStore

    with DataStore(run.raw, 1 << 30) as store:
        oriented = store.materialize()
    want = reference(oriented, run.info, key=key, method=method, target=target, outcome=outcome,
                     order=order)
    got = table(working)
    assert len(got) == len(want) == receipt["n_units"] and receipt["n_source_rows"] == 600
    assert list(got["__row_id"]) == list(range(len(got)))  # dense over its rows
    assert list(got.columns[:-1]) == [c["name"] for c in run.info["columns"]]
    for column in want.columns:
        bad = [i for i, (a, b) in enumerate(zip(want[column], got[column])) if not same(a, b)]
        assert not bad, (method, column, want[column].iloc[bad[0]], got[column].iloc[bad[0]])
    assert receipt["outcome"]["rule"] == ("constant" if outcome is None else outcome)


def test_an_outcome_that_varies_within_a_unit_must_say_which_to_keep(graph):
    run = graph(CLINICAL)
    with pytest.raises(StructureError, match="which value to keep"):
        run.run(state(lens=["clinical"], target="progressed",
                      grain={"grain": "repeated", "id_column": "subject_id"}, unit="unit",
                      aggregation={"method": "first"}), upto=["working"])


def test_the_row_map_sends_every_source_row_to_its_units_one_row(graph):
    run = graph(DIETARY)
    out = run.run(state(lens=["dietary"], target="hba1c",
                        grain={"grain": "repeated", "id_column": "participant_id"}, unit="unit",
                        aggregation={"method": "mean"}), upto=["working"])
    working = out["working"]
    mapping = row_map(working)
    source = pd.read_parquet(run.raw, columns=["participant_id", "__row_id"]).set_index("__row_id")
    combined = table(working).set_index("__row_id")
    assert sorted(mapping["source_row_id"]) == list(range(600))  # every source row, once
    assert mapping["row_id"].nunique() == len(combined) == 300
    for row_id, rows in mapping.groupby("row_id")["source_row_id"]:
        unit = combined.loc[row_id, "participant_id"]
        assert set(rows) == set(source.index[source["participant_id"] == unit])
    firsts = mapping.groupby("row_id")["source_row_id"].min()
    assert firsts.is_monotonic_increasing  # working rows come in order of each unit's first row


# ── the rewiring ──────────────────────────────────────────────────────────────

READS_WORKING = ("target_info", "roles", "proposals", "cohort", "split", "design", "fit",
                 "substitution")
READS_ORIENTED = ("profile", "findings", "structure")


def test_every_stage_after_working_depends_on_it_or_on_one_that_does():
    from turbotab.core.stages import build_graph

    graph = build_graph()
    for name in READS_WORKING:
        assert "working" in graph[name].deps, name  # it opens the table, so it must hold it
    assert "working" in graph.upstream("shelf")
    assert graph["findings"].deps == ("oriented",) and graph["oriented"].deps == ("ingest",)
    assert set(graph["working"].deps) == {"oriented", "findings", "structure"}


def test_every_downstream_stage_reads_the_working_table(graph, monkeypatch):
    from turbotab.core import datastore

    opened: dict[str, list[Path]] = {}
    current = {"stage": None}
    real_init = datastore.DataStore.__init__

    def spy(self, parquet, budget):
        opened.setdefault(current["stage"], []).append(Path(parquet).resolve())
        real_init(self, parquet, budget)

    monkeypatch.setattr(datastore.DataStore, "__init__", spy)

    def before(stage):
        current["stage"] = stage

    run = graph(DIETARY)
    base = dict(lens=["dietary"], target="hba1c", purpose="prediction",
                grain={"grain": "repeated", "id_column": "participant_id"}, unit="unit",
                aggregation={"method": "mean"})
    roles = run.run(state(**base), upto=["roles"])["roles"]
    proposed = {c["column"]: c["proposed"] for c in roles["columns"]}
    out = run.run(state(**base, roles=proposed, exclusions=[], missing="impute",
                        split={"holdout": 0.2}, energy_adjustment={"method": "none"},
                        models=["linear"],
                        substitution={"donor": "fat_g", "recipient": "carbohydrate_g"}),
                  before=before)
    working_table = out["working"].files["table.parquet"].resolve()
    oriented_table = out["oriented"].files["table.parquet"].resolve()
    assert working_table != oriented_table and len(table(out["working"])) == 300
    for name in READS_WORKING:
        assert name in out, f"{name} did not run"
        assert opened.get(name), f"{name} opened no table"
        assert set(opened[name]) == {working_table}, name
    for name in ("profile", "findings", "structure"):
        assert set(opened[name]) == {oriented_table}, name
    assert out["cohort"].data["steps"][0]["n"] == 300  # the participant flow counts people now


def test_with_nothing_structural_recorded_the_working_table_is_the_raw_file(graph):
    run = graph(DIETARY)
    out = run.run(state(lens=["dietary"], target="hba1c"), upto=["working"])
    for stage in ("oriented", "working"):
        assert out[stage].data.get("pass_through", True) is True
        assert out[stage].files["table.parquet"] == run.raw.resolve()  # the same file, by reference
    folder = run.cache / "working"
    written = [p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
               and p.name not in ("meta.json", "artifact.json", "latest")]
    assert written == []  # copies nothing: only the artifact's own JSON is written
    linked = [p for p in folder.rglob("table.parquet")]
    assert linked and all(os.path.islink(p) for p in linked)
    identity = row_map(out["working"])
    assert (identity["row_id"] == identity["source_row_id"]).all() and len(identity) == 600


def test_a_written_table_is_described_as_the_ingest_describes_it(tmp_path):
    """The working table's sidecar (counted with Arrow) says what DuckDB's ingest pass says."""
    import shutil

    from turbotab.core.datastore import ingest
    from turbotab.core.stages.working import _describe

    for path in (DIETARY, CLINICAL, SAMPLES / "nhanes_dietary.csv", SAMPLES / "survey_sentinels.csv"):
        parquet = tmp_path / f"{path.stem}.parquet"
        want = ingest(path, parquet).to_dict()
        copy = tmp_path / f"{path.stem}-copy.parquet"
        shutil.copyfile(parquet, copy)
        got = _describe(copy, tmp_path / f"{path.stem}-copy.info.json", 0.0)
        assert got["n_rows"] == want["n_rows"] and got["n_cols"] == want["n_cols"]
        for a, b in zip(got["columns"], want["columns"]):
            keys = ("name", "dtype", "physical_type", "n_missing", "n_unique", "sample")
            assert {k: a[k] for k in keys} == {k: b[k] for k in keys}, path.name
