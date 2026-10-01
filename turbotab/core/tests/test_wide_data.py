"""Wide data (M2_CONTRACT §5): the Arrow paths answer exactly what the DuckDB paths answer.

Tier A (BLUEPRINT §8): column statistics, summaries and materialized frames feed every count
and every model, so the Arrow paths that replace DuckDB projections at 20,000 columns are held
equal to the DuckDB ones on a table with every type, missing values, NaN, ``-0.0``, a column
with no values, a constant and a text column with more than a thousand values. Row identity is
held on a file of many row groups.
"""
from __future__ import annotations

import datetime as dt
import decimal
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from turbotab.core import datastore as ds_mod
from turbotab.core.datastore import ROW_ID, DataStore, ingest

BUDGET = 2 << 30
N = 3_000


def _mixed_table(n: int = N, seed: int = 11) -> pa.Table:
    rng = np.random.default_rng(seed)
    ints = rng.integers(-50, 50, size=n)
    floats = rng.normal(10, 3, size=n)
    floats[::7] = -0.0
    floats[::11] = 0.0
    big = rng.integers(2**40, 2**52, size=n)
    flags = rng.random(n) < 0.3
    cats = rng.choice(["a", "bb", "ccc", "dé"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    # More than EXACT_TOP_MAX_UNIQUE values, with a clear top five (counts 60, 50, 40, 30, 20).
    text = [f"id{i:05d}" for i in range(n)]
    for k, (word, count) in enumerate((("alpha", 60), ("beta", 50), ("gamma", 40),
                                       ("delta", 30), ("eps", 20))):
        for j in range(count):
            text[(k * 200 + j * 3) % n] = word
    base = dt.date(2020, 1, 1)
    dates = [base + dt.timedelta(days=int(d)) for d in rng.integers(0, 400, size=n)]
    stamps = [dt.datetime(2021, 5, 1) + dt.timedelta(minutes=int(m))
              for m in rng.integers(0, 50_000, size=n)]
    times = [dt.time(int(h), int(m)) for h, m in zip(rng.integers(0, 24, n), rng.integers(0, 60, n))]
    money = [decimal.Decimal(int(c)) / 100 for c in rng.integers(0, 10**7, size=n)]

    def holes(values: list | np.ndarray, every: int, offset: int = 0) -> list:
        out = list(values.tolist() if isinstance(values, np.ndarray) else values)
        for i in range(offset, n, every):
            out[i] = None
        return out

    return pa.table({
        "int_col": pa.array(holes(ints, 13), pa.int64()),
        "small_int": pa.array(rng.integers(0, 3, size=n).astype(np.int16)),
        "float_col": pa.array(holes(floats, 17, 3), pa.float64()),
        "float32_col": pa.array(rng.normal(0, 1, size=n).astype(np.float32)),
        "big_int": pa.array(big, pa.int64()),
        "flag": pa.array(holes(flags, 19), pa.bool_()),
        "category": pa.array(holes(cats, 23), pa.string()),
        "text": pa.array(holes(text, 29, 5), pa.string()),
        "day": pa.array(holes(dates, 31), pa.date32()),
        "stamp": pa.array(holes(stamps, 37), pa.timestamp("us")),
        "clock": pa.array(times, pa.time64("us")),
        "money": pa.array(money, pa.decimal128(12, 2)),
        "empty": pa.array([None] * n, pa.float64()),
        "constant": pa.array([7] * n, pa.int64()),
        "x.1": pa.array(rng.normal(size=n)),   # a dotted name pyarrow must not read as a path
        ROW_ID: pa.array(np.arange(n, dtype=np.int64)),
    })


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> Path:
    """The mixed table as a Parquet file of 7 row groups, read through a DataStore."""
    folder = tmp_path_factory.mktemp("mixed")
    path = folder / "raw.parquet"
    pq.write_table(_mixed_table(), path, row_group_size=450, compression="zstd")
    return path


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


# ── column statistics (ingest) ───────────────────────────────────────────────

def test_column_stats_through_arrow_equal_duckdbs(mixed):
    con = ds_mod._connect()
    try:
        by_duckdb, extras_duckdb = ds_mod._column_stats(con, mixed, N, arrow=False)
        by_arrow, extras_arrow = ds_mod._column_stats(con, mixed, N, arrow=True)
    finally:
        con.close()
    assert [c.to_dict() for c in by_arrow] == [c.to_dict() for c in by_duckdb]
    assert extras_arrow["nan_columns"] == extras_duckdb["nan_columns"]
    assert extras_arrow["avg_len"].keys() == extras_duckdb["avg_len"].keys()
    for name, length in extras_duckdb["avg_len"].items():
        assert math.isclose(extras_arrow["avg_len"][name], length, rel_tol=1e-12), name
    # -0.0 and 0.0 are one value, as DuckDB's count(DISTINCT) has them.
    floats = next(c for c in by_arrow if c.name == "float_col")
    values = _mixed_table().column("float_col").to_pandas().dropna()
    assert floats.n_unique == values.nunique()


def test_nan_is_found_and_counted_once_through_arrow(tmp_path):
    path = tmp_path / "nan.parquet"
    pq.write_table(pa.table({"x": [1.0, float("nan"), float("nan"), None, -0.0, 0.0],
                             "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             ROW_ID: np.arange(6, dtype=np.int64)}), path)
    con = ds_mod._connect()
    try:
        results = [ds_mod._column_stats(con, path, 6, arrow=flag) for flag in (False, True)]
    finally:
        con.close()
    (duck, duck_extras), (arrow, arrow_extras) = results
    assert [c.to_dict() for c in arrow] == [c.to_dict() for c in duck]
    assert arrow_extras["nan_columns"] == duck_extras["nan_columns"] == ["x"]
    assert arrow[0].n_unique == 3  # 1, NaN, 0


# ── summaries ────────────────────────────────────────────────────────────────

def test_summaries_through_arrow_equal_duckdbs(mixed):
    with DataStore(mixed, BUDGET) as store:
        cols = store.columns
        by_duckdb = store._compute_summaries(cols, arrow=False)
        by_arrow = store._compute_summaries(cols, arrow=True)
    assert by_arrow.keys() == by_duckdb.keys()
    for name in cols:
        a, d = by_arrow[name], by_duckdb[name]
        for key in ("name", "dtype", "n", "n_missing", "n_unique"):
            assert a[key] == d[key], (name, key)
        for key in ("mean", "std", "min", "q25", "median", "q75", "max"):
            assert _same(a[key], d[key]), (name, key, a[key], d[key])
            assert type(a[key]) is type(d[key]), (name, key)  # an integer min stays an integer
        if a["dtype"] in ("numeric", "integer", "boolean") or a["n_unique"] <= ds_mod.EXACT_TOP_MAX_UNIQUE:
            assert a["top"] == d["top"], name
            continue
        # Past 1,000 values DuckDB finds candidates approximately (approx_top_k), and misses:
        # on `text` it ranks a value seen once above `eps`, seen 20 times. Arrow counts every
        # value, so its top five are the exact ones and never count fewer.
        exact = _mixed_table().column(name).to_pandas().dropna().map(ds_mod.json_safe)
        ranked = sorted(exact.value_counts().items(), key=lambda kv: (-kv[1], str(kv[0])))[:5]
        assert a["top"] == [{"value": v, "count": int(c)} for v, c in ranked], name
        assert [t["count"] for t in a["top"]] >= [t["count"] for t in d["top"]], name


def test_summaries_of_a_wide_short_table_go_through_arrow(tmp_path, monkeypatch):
    """The dispatch: past one batch of columns on a short table, the Arrow path answers."""
    rng = np.random.default_rng(3)
    data = {f"g{j:04d}": rng.poisson(3.0, size=40) for j in range(ds_mod.BATCH_COLUMNS + 50)}
    source = tmp_path / "wide.csv"
    pd.DataFrame(data).to_csv(source, index=False)
    dest = tmp_path / "data" / "raw.parquet"
    ingest(source, dest)
    calls: list[str] = []
    real = DataStore._summaries_arrow

    def spy(self, cols):
        calls.append("arrow")
        return real(self, cols)

    monkeypatch.setattr(DataStore, "_summaries_arrow", spy)
    with DataStore(dest, BUDGET) as store:
        summaries = store.summaries()
        assert calls == ["arrow"]
        assert ds_mod.arrow_columnwise(len(store.columns), store.n_rows)
    frame = pd.DataFrame(data)
    for s in summaries[:20]:
        col = frame[s["name"]]
        assert s["n"] == 40 and s["min"] == int(col.min()) and s["max"] == int(col.max())
        assert math.isclose(s["mean"], float(col.mean()), rel_tol=1e-12)
        assert math.isclose(s["std"], float(col.std()), rel_tol=1e-9)
        assert math.isclose(s["median"], float(col.median()), rel_tol=1e-12)
        assert math.isclose(s["q25"], float(col.quantile(0.25)), rel_tol=1e-12)


# ── materialize ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["all", "sample", "shuffled", "repeats", "none", "one_group"])
def test_materialize_through_arrow_equals_duckdb(mixed, which):
    rng = np.random.default_rng(5)
    ids = {
        "all": None,
        "sample": np.sort(rng.choice(N, size=400, replace=False)),
        "shuffled": rng.permutation(N)[:1_234],
        "repeats": np.array([5, 2_999, 5, 0, 1_800, 0]),
        "none": np.array([], dtype=np.int64),
        "one_group": np.arange(900, 1_000),
    }[which]
    with DataStore(mixed, BUDGET) as store:
        physical = {c.name: c.physical_type for c in store.info().columns}
        cols = [c for c in store.columns if ds_mod.arrow_safe(physical[c])]
        assert set(store.columns) - set(cols) == {"money"}  # a decimal goes through DuckDB
        expected = store._materialize_duckdb(cols, ids)
        got = store._materialize_arrow(cols, ids)
        public = store.materialize(cols, ids)
        everything = store.materialize(None, ids)
        pd.testing.assert_frame_equal(everything, store._materialize_duckdb(store.columns, ids),
                                      check_exact=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    pd.testing.assert_frame_equal(public, expected, check_exact=True)
    assert got.index.name == "row_id"
    if ids is not None:
        assert got.index.to_numpy().tolist() == [int(i) for i in ids]


def test_materialize_on_a_file_not_in_row_id_order(tmp_path):
    """A table written in another order (a working table may be) still answers by row id."""
    table = _mixed_table(n=900).drop_columns(["money"])
    order = np.random.default_rng(8).permutation(900)
    path = tmp_path / "shuffled.parquet"
    pq.write_table(table.take(pa.array(order)), path, row_group_size=200)
    picked = np.array([899, 3, 450, 3, 0])
    with DataStore(path, BUDGET) as store:
        cols = store.columns
        for ids in (None, picked, np.arange(100, 300)):
            pd.testing.assert_frame_equal(store._materialize_arrow(cols, ids),
                                          store._materialize_duckdb(cols, ids), check_exact=True)
        assert store.materialize(["int_col"], picked).index.tolist() == picked.tolist()


def test_arrow_materialize_reads_only_the_row_groups_it_needs(mixed, monkeypatch):
    read: list[list[int]] = []

    class Spy(pq.ParquetFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            real = self.reader

            class Reader:
                def read_row_groups(self, groups, *a, **k):
                    read.append(list(groups))
                    return real.read_row_groups(groups, *a, **k)

                def __getattr__(self, name):
                    return getattr(real, name)

            self.reader = Reader()

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    with DataStore(mixed, BUDGET) as store:
        frame = store.materialize(["int_col", "x.1"], [10, 20, 2_000])
    assert [g for batch in read for g in batch] == [0, 4]  # 450-row groups
    assert frame.index.tolist() == [10, 20, 2_000]


def test_a_zoned_timestamp_goes_through_duckdb(tmp_path):
    """Arrow would answer in UTC where DuckDB answers in the session's zone: not Arrow's."""
    path = tmp_path / "tz.parquet"
    stamps = pa.array([dt.datetime(2022, 1, 1, 12, tzinfo=dt.timezone.utc)] * 3,
                      pa.timestamp("us", tz="UTC"))
    pq.write_table(pa.table({"t": stamps, "x": [1, 2, 3], ROW_ID: np.arange(3, dtype=np.int64)}),
                   path)
    with DataStore(path, BUDGET) as store:
        physical = {c.name: c.physical_type for c in store.info().columns}
        assert not ds_mod.arrow_safe(physical["t"]) and ds_mod.arrow_safe(physical["x"])
        pd.testing.assert_frame_equal(store.materialize(), store._materialize_duckdb(["t", "x"], None))


# ── row groups and very long lines ───────────────────────────────────────────

def test_row_group_rows():
    assert ds_mod.row_group_rows(30) is None                        # DuckDB's default
    assert ds_mod.row_group_rows(20_000) == 2_048                   # whole vectors, at least one
    assert ds_mod.row_group_rows(1_000) == 8_192
    assert ds_mod.row_group_rows(20_000, n_rows=500) is None        # a few groups' rows: default
    assert ds_mod.row_group_rows(20_000, n_rows=50_000) == 2_048


def test_a_wide_table_gets_small_row_groups_and_keeps_row_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ds_mod, "ROW_GROUP_CELLS", 500_000)          # 250 columns -> 2,048 rows
    n_rows, n_cols = 8_000, 250
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.integers(0, 1_000, size=(n_rows, n_cols)),
                         columns=[f"c{j}" for j in range(n_cols)])
    source = tmp_path / "wide.csv"
    frame.to_csv(source, index=False)
    dest = tmp_path / "data" / "raw.parquet"
    ingest(source, dest)
    meta = pq.read_metadata(dest)
    assert meta.num_row_groups >= 3
    assert all(meta.row_group(i).num_rows <= 2_048 for i in range(meta.num_row_groups))
    ids = pq.read_table(dest, columns=[ROW_ID]).column(ROW_ID).to_numpy()
    assert ids.tolist() == list(range(n_rows))
    with DataStore(dest, BUDGET) as store:
        window = store.window(2_040, 20, ["c0", "c249"])
        assert window["rows"] == frame.loc[2_040:2_059, ["c0", "c249"]].values.tolist()
        picked = np.array([7_999, 0, 2_048, 2_047, 4_100])
        got = store.materialize(None, picked)
        pd.testing.assert_frame_equal(got, store._materialize_duckdb(store.columns, picked))
        assert got["c7"].tolist() == frame.loc[picked, "c7"].tolist()


def test_max_line_size_is_raised_only_for_very_wide_files(tmp_path):
    narrow = tmp_path / "narrow.csv"
    narrow.write_text("a,b,c\n1,2,3\n")
    assert ds_mod.max_line_size(narrow) is None
    wide = tmp_path / "wide.csv"
    wide.write_text(",".join(f"feature_{i:06d}" for i in range(40_000)) + "\n")
    need = ds_mod.max_line_size(wide)
    assert need is not None and need >= 40_000 * ds_mod.FIELD_BYTES > ds_mod.DUCKDB_MAX_LINE
    assert "max_line_size" in ds_mod._line_options(need)


def test_a_line_over_two_megabytes_is_read(tmp_path):
    """DuckDB refuses a line over 2 MB by default. A 1.2 MB header raises the limit to twice
    its length, and a 2.3 MB value then reads whole. (The realistic case, 40,000 columns of
    long names, is in the benchmark: it costs 4 GB of DuckDB's memory, too much for a test.)"""
    name = "n" * 1_200_000
    value = "v" * 2_300_000
    source = tmp_path / "long.csv"
    source.write_text(f"id,{name}\n1,{value}\n2,short\n")
    assert ds_mod.max_line_size(source) == 2 * 1_200_004
    info = ingest(source, tmp_path / "data" / "raw.parquet")
    assert (info.n_rows, info.n_cols) == (2, 2) and info.columns[1].name == name
    with DataStore(tmp_path / "data" / "raw.parquet", BUDGET) as store:
        rows = store.window(0, 2, [name])["rows"]
    assert len(rows[0][0]) == len(value) and rows[1][0] == "short"


# ── the elastic net on a wide matrix ─────────────────────────────────────────

def test_the_wide_elastic_net_chooses_what_the_default_chooses():
    """Single precision changes the cost, not the fit: on two p > n tables, the same penalty,
    mix and support as sklearn's float64 fit, and the same predictions to 1e-4 of the outcome's sd.
    (A looser tolerance would not pass this: it moves the penalty and the support.)"""
    import warnings

    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    from turbotab.core.bench.synth import wide_table
    from turbotab.core.models.elastic_net import ELASTIC_NET, L1_RATIOS
    from turbotab.core.models.wide import Float32ElasticNetCV

    narrow = ELASTIC_NET.build("regression", "prediction", 1_000, 50)
    assert type(narrow) is ElasticNetCV                               # p <= n: untouched
    for seed in (4, 5):
        table = wide_table(n=150, genes=800, seed=seed)
        genes = [c for c in table.column_names if c.startswith("gene_")]
        X = StandardScaler().fit_transform(np.column_stack([table[c].to_numpy() for c in genes])
                                           .astype(float))
        X = pd.DataFrame(X, columns=genes)
        y = table["bmi"].to_numpy()
        wide = ELASTIC_NET.build("regression", "prediction", 150, 800)
        assert type(wide) is Float32ElasticNetCV and wide.tol == 1e-4 and wide.n_jobs >= 1
        default = ElasticNetCV(l1_ratio=list(L1_RATIOS), cv=5, max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wide.fit(X, y)
            default.fit(X, y)
        assert wide.l1_ratio_ == default.l1_ratio_, seed
        assert math.isclose(wide.alpha_, default.alpha_, rel_tol=1e-6), seed  # one grid point
        assert set(np.flatnonzero(wide.coef_)) == set(np.flatnonzero(default.coef_)), seed
        assert list(wide.feature_names_in_) == genes
        scale = float(np.abs(default.coef_).max())
        assert np.allclose(wide.coef_, default.coef_, atol=1e-4 * scale), seed
        assert np.allclose(wide.predict(X), default.predict(X), atol=1e-4 * float(np.std(y))), seed


# ── the packs' shape readings, once per stage ────────────────────────────────

def test_the_shape_memo_reads_once_and_changes_no_finding(tmp_path, monkeypatch):
    """findings and lens hints are the same with the memo as without it; inside it a reading
    of the same frame is computed once (``reframe`` asked once per finding: 2 minutes at 20k)."""
    from turbotab import packs
    from turbotab.core.decisions import ProjectState
    from turbotab.core.shape_memo import remembered
    from turbotab.core.stages.data import profile_stage
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.tests.stage_harness import SAMPLES, Ingested

    h = Ingested(SAMPLES / "genomics_expression.csv", tmp_path)
    state = ProjectState(lens=["genomics"], target="condition")
    assert h.run(findings_stage, state) == h.run(findings_stage.__wrapped__, state)
    assert h.run(profile_stage, state) == h.run(profile_stage.__wrapped__, state)

    frame = h.frame().reset_index(drop=True)
    calls: list[int] = []
    real = packs._numeric
    monkeypatch.setattr(packs, "_numeric", lambda df: calls.append(1) or real(df))
    with remembered():
        first, second = packs.count_matrix(frame), packs.count_matrix(frame)
    assert first == second and first is not second and len(calls) == 1
    packs.count_matrix(frame)                      # outside the block: computed again
    assert len(calls) == 2
