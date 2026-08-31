"""`F-01` — the Classic transpose destroyed the identifiers it exists to keep.

The checkbox on the upload page says *"use this if your features are in rows
instead of columns"*, which names one kind of user: someone holding an omics
matrix, a string `gene_id` column beside one numeric column per sample. For
exactly that file, `data_processor.transpose_dataframe`'s bare `df.T` produced

* columns named `0, 1, 2, …` — the source RangeIndex — so the gene names were
  gone from every reading, every picker and the methods section after it;
* the gene names back as a phantom first row of strings, counted as a sample;
* an all-`object` frame, because that one string row types the whole column.

None of it raised. The three assertions below are the three failures, and they
go through `load_tabular_data(transpose=True)` — the real Classic path, the one
the page and `utils/perf_cache.cached_parse_upload` call — rather than through
`turbotab.orientation` directly, because that delegation is the thing under
test. `turbotab/test_a_transposed_assay_table_is_turned_around_before_diagnosis.py`
already covers the module itself.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_processor import load_tabular_data                    # noqa: E402
from turbotab import orientation as O                           # noqa: E402

GENES = [f"ENSG{i:05d}" for i in range(12)]
SAMPLES = [f"S{i:02d}" for i in range(1, 9)]


def _assay_bytes(gene_ids=None) -> io.BytesIO:
    """A features-in-rows assay export: one identifier column, samples across."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "gene_id": list(gene_ids if gene_ids is not None else GENES),
        **{s: rng.lognormal(i, 1, 12) for i, s in enumerate(SAMPLES, start=1)},
    })
    return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))


def _turned() -> pd.DataFrame:
    return load_tabular_data(_assay_bytes(), filename="counts.csv", transpose=True)


# ── the three failures ───────────────────────────────────────────────────────

def test_the_gene_names_are_the_columns():
    """`df.T` named them `0…11`. An identifier that becomes a position is not
    an identifier, and nothing downstream can get it back."""
    df = _turned()
    assert set(df.columns) - {"sample_id"} == set(GENES)


def test_no_gene_name_survives_as_a_cell():
    """The phantom row: the identifiers used to come back as row 0, a row of
    strings that reads as a thirteenth sample and is counted as one."""
    df = _turned()
    values = set(df.astype(str).to_numpy().ravel().tolist())
    assert not values & set(GENES)
    assert len(df) == len(SAMPLES)


def test_every_measurement_column_is_numeric():
    """Scoped to the measurements on purpose. `sample_id` holds the sample
    names and is legitimately `object`; the twelve gene columns are the ones
    the phantom string row used to type as `object` at a measured 40.0
    bytes/cell against 8."""
    df = _turned()
    measurements = df.drop(columns=["sample_id"])
    assert not list(measurements.select_dtypes(include=["object"]).columns)
    assert measurements.shape == (len(SAMPLES), len(GENES))


def test_the_sample_names_are_carried_rather_than_dropped():
    """Where the extra column came from. `df.T` put the sample names in an
    index and `reset_index` was never called, so they were discarded in
    silence — the samples were as anonymous as the genes."""
    df = _turned()
    assert df["sample_id"].tolist() == SAMPLES


# ── the refusal, and that it is reachable ────────────────────────────────────

def test_a_duplicated_gene_name_is_refused_rather_than_silently_merged():
    """Two rows with one name become two columns with one name, and every
    consumer after that sees whichever pandas hands it. The first assertion is
    the one that matters: if the identifier column were not recognized, the
    names would be quietly replaced by `row_0, row_1, …` and this refusal would
    never fire — the test would pass without exercising anything."""
    dupes = ["ENSG00000", "ENSG00000"] + GENES[2:]
    frame = pd.read_csv(_assay_bytes(dupes))
    assert O.label_column(frame) == "gene_id"
    with pytest.raises(O.OrientationError, match="Two rows are both named"):
        load_tabular_data(_assay_bytes(dupes), filename="counts.csv",
                          transpose=True)


# ── the refusals that the identifier rule cannot reach on its own ────────────
#
# Every case below reached `orientation.transpose`'s `else` branch, where the
# identifier column becomes a row and the per-column coercion replaces it with
# NaN. Each one is an ordinary omics export, and for each the delegation was
# measured to be WORSE than the `df.T` it replaced: `df.T` stranded the names as
# cells, this destroyed them. `data_processor._transpose_refusal` is the guard.


def _gene_ids_survive(df: pd.DataFrame, ids) -> bool:
    """Anywhere at all — as columns, or stranded in cells the way `df.T` left
    them. The bar is deliberately this low: a refusal is acceptable, quietly
    returning a frame with the identifiers nowhere in it is not."""
    seen = set(df.columns) | set(df.astype(str).to_numpy().ravel().tolist())
    return bool(seen & {str(i) for i in ids})


def test_gene_symbols_repeated_too_often_to_be_read_as_labels_are_refused():
    """A symbol-keyed expression matrix repeats symbols routinely, and probe
    arrays repeat them by design. Three duplicates in twelve rows puts the
    column under `label_column`'s 90%-distinct bar, so it is not recognized as
    the identifiers at all and the refusal behind that recognition never fires.
    Measured before the guard: shape (9, 13), columns `row_0…row_11`, and the
    symbols in no cell of the result."""
    dupes = ["ENSG00000", "ENSG00000", "ENSG00000"] + GENES[3:]
    assert O.label_column(pd.read_csv(_assay_bytes(dupes))) is None
    with pytest.raises(O.OrientationError, match="Two rows are both named"):
        load_tabular_data(_assay_bytes(dupes), filename="counts.csv",
                          transpose=True)


def test_the_duplicate_refusal_fires_below_ten_rows_where_the_margin_rounds_away():
    """`values.nunique() >= 0.9 * len(df)` cannot separate a duplicate from a
    clean column on a short frame: at n=5, four distinct names clear 4.5 — no,
    they do not, and at n=8 seven clear 7.2 — they do not either, so the column
    is rejected and nothing refuses. A targeted panel really is this short."""
    short = ["mz_1", "mz_1", "mz_3", "mz_4", "mz_5"]
    frame = pd.DataFrame({"mz_id": short,
                          **{s: np.arange(5, dtype=float) for s in SAMPLES[:3]}})
    buf = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    with pytest.raises(O.OrientationError, match="Two rows are both named"):
        load_tabular_data(buf, filename="panel.csv", transpose=True)


def test_a_blank_in_the_identifier_column_is_refused_rather_than_dropped():
    """`label_column` requires the column be fully populated, so a single blank
    disqualifies it however distinct the other 11 values are — and the 11 good
    names go with it. The message has to name the blanks, because from the
    user's side the column looks perfectly fine."""
    holed = [GENES[0], None] + GENES[2:]
    frame = pd.read_csv(_assay_bytes(holed))
    assert O.label_column(frame) is None
    with pytest.raises(O.OrientationError, match="blank"):
        load_tabular_data(_assay_bytes(holed), filename="counts.csv",
                          transpose=True)


def test_an_annotation_column_beside_the_identifiers_is_refused():
    """The identifiers ARE recognized here — this is the case that survives the
    guard's first question and still corrupts. `chrom` is not a sample, so it
    becomes a row of text that fails the numeric coercion and lands as an
    all-NaN 'sample' in the middle of the study."""
    frame = pd.DataFrame({
        "gene_id": GENES, "chrom": (["1"] * 6) + (["X"] * 6),
        **{s: np.arange(12, dtype=float) for s in SAMPLES[:3]},
    })
    assert O.label_column(frame) == "gene_id"
    buf = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    with pytest.raises(O.OrientationError, match="holds text"):
        load_tabular_data(buf, filename="counts.csv", transpose=True)


@pytest.mark.parametrize("ids", [
    pytest.param(["ENSG00000"] * 3 + GENES[3:], id="repeated-symbols"),
    pytest.param([GENES[0], None] + GENES[2:], id="one-blank"),
])
def test_no_refused_frame_ever_comes_back_with_the_names_destroyed(ids):
    """The property behind all of the above, stated once. Either the transpose
    refuses, or the identifiers are somewhere in what it returns. What must
    never happen — and what happened before the guard — is a frame handed back
    with the names in neither place and no error raised."""
    try:
        df = load_tabular_data(_assay_bytes(ids), filename="counts.csv",
                               transpose=True)
    except O.OrientationError:
        return
    assert _gene_ids_survive(df, [i for i in ids if i is not None])


# ── the frame with nothing to name its rows ──────────────────────────────────

def test_a_frame_with_no_identifier_column_gets_stable_row_names():
    """Not every features-in-rows export ships an identifier column. Positions
    are all there is to name those rows by, so they become `row_0, row_1, …` —
    no better than the `0, 1, 2` `df.T` gave, but stable strings rather than
    integers that later read as a numeric column, and the samples are still
    carried instead of dropped."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({s: rng.lognormal(i, 1, 12)
                          for i, s in enumerate(SAMPLES, start=1)})
    buf = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    df = load_tabular_data(buf, filename="counts.csv", transpose=True)
    assert list(df.columns) == ["sample_id"] + [f"row_{i}" for i in range(12)]
    assert df["sample_id"].tolist() == SAMPLES
    assert not list(df.drop(columns=["sample_id"])
                      .select_dtypes(include=["object"]).columns)


# ── and the file that is not an assay at all ─────────────────────────────────

def test_an_ordinary_csv_is_untouched_when_the_box_is_not_ticked():
    """The whole point of the scoping: this changes what ticking the box does,
    and nothing else. A 500 × 20 upload that never asks for a transpose must
    come back byte-identical."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({f"col_{i}": rng.normal(0, 1, 500) for i in range(19)})
    frame["group"] = ["a", "b"] * 250
    raw = frame.to_csv(index=False).encode("utf-8")
    loaded = load_tabular_data(io.BytesIO(raw), filename="study.csv")
    assert loaded.shape == (500, 20)
    pd.testing.assert_frame_equal(loaded, pd.read_csv(io.BytesIO(raw)))
