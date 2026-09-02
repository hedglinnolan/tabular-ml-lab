"""`utils/admission.py` — the upload gate reads the data, not its formatting.

**The defect, measured.** The gate this replaces compared `uploaded_file.size`
against 50 MB. One identical 20,000 x 300 matrix — the same 47.0 MB DataFrame
in memory, the same analysis, the same everything — was written out at 19.66 MB
as integer counts, 40.44 MB at four decimal places, and 106.32 MB at pandas'
default float repr. Two were admitted and one was refused, and the only
difference between them was decimal places. A gene expression matrix written at
full precision was refused for being legible.

So the load-bearing test in this file is not that some particular frame is
refused. It is that those three renderings reach the SAME verdict, because a
gate that answers differently for the same data is not a limit, it is a
coin toss with a scientific cost.

**The other two claims are about what a gate must not do.** It must not refuse
because a memory probe failed — `available_memory_bytes()` returns `None` for
*cannot estimate*, and a researcher blocked by a missing psutil is a worse and
far more common outcome than the OOM being guessed at. And it must not offer an
override on a memory refusal: the upload page renders its "Load anyway"
checkbox from `Verdict.warnings`, so a refusal that carried warnings alongside
it would grow a way past itself without anyone editing the page. That is
asserted structurally here rather than left to the page's indentation.

Pure functions of a shape and an integer, so none of this needs a Streamlit
runtime, a browser, or a file on disk.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.admission import (                                    # noqa: E402
    PREFILTER_REFUSE_MB,
    WIDE_COLUMN_WARN,
    Verdict,
    admission_verdict,
    estimated_frame_bytes,
    measured_frame_bytes,
    prefilter_verdict,
)

GB = 1024 ** 3

#: The audit's case: 20,000 rows x 300 columns, one matrix, three renderings.
OMICS_SHAPE = (20_000, 300)
#: Megabytes on disk the audit measured for each rendering of that one matrix.
THREE_RENDERINGS_MB = (19.66, 40.44, 106.32)


# ── the ordinary CSV must not notice this change at all ──────────────────────

@pytest.mark.parametrize("available", [
    None,          # no probe at all
    256 * 2**20,   # a quarter of a gigabyte free, a laptop under pressure
    2 * GB,
    64 * GB,       # a departmental server
])
def test_a_five_hundred_row_csv_is_admitted_in_silence(available):
    """The must-not-regress case. 500 x 20 is what this app is used for daily.

    Not merely "not refused" — `clean` means the page renders no warning and
    grows no checkbox, at every plausible reading of the host including no
    reading at all. A gate that adds friction to the ordinary file has failed
    even when it lets the file through.
    """
    verdict = admission_verdict(500, 20, "cohort.csv", available)
    assert verdict.clean
    assert not verdict.refused
    assert verdict.warnings == ()


def test_an_ordinary_csv_is_not_pre_filtered_on_bytes():
    """The cheap pre-parse backstop must be invisible to a real file."""
    # 500 x 20 of float text is on the order of a hundred kilobytes.
    assert prefilter_verdict(200 * 1024, "cohort.csv").clean


# ── the defect: one matrix, three renderings, one answer ─────────────────────

@pytest.mark.parametrize("available", [4 * GB, 32 * GB, None])
def test_three_renderings_of_one_matrix_reach_one_verdict(available):
    """The whole point. Decimal places are not a property of the data.

    The shape gate never sees the file's bytes, so this is close to a tautology
    at the level of the function — which is exactly the claim being locked
    down. The old gate could not express it at all.
    """
    verdicts = [admission_verdict(*OMICS_SHAPE, f"expr_{mb}.csv", available)
                for mb in THREE_RENDERINGS_MB]
    assert len({(v.refused, len(v.warnings)) for v in verdicts}) == 1


def test_the_old_byte_cap_would_have_split_those_three():
    """Evidence that the case above is real, not a straw man.

    The replaced gate was `uploaded_file.size > 50 MB`. Run the audit's three
    measured sizes through it and it answers two ways.
    """
    old_cap_verdicts = {mb > 50 for mb in THREE_RENDERINGS_MB}
    assert old_cap_verdicts == {False, True}


# ── a failed probe warns; it never refuses ───────────────────────────────────

@pytest.mark.parametrize("rows,cols", [
    (500, 20),
    (20_000, 300),
    (5_000_000, 4_000),   # far past any machine's memory
])
def test_an_unmeasurable_host_never_refuses(rows, cols):
    """`None` means cannot estimate. Refusing on it inverts the probe's own
    contract, and it would land hardest on the lean installs least equipped to
    work out why their file was rejected."""
    verdict = admission_verdict(rows, cols, "x.csv", None)
    assert not verdict.refused


def test_an_unmeasurable_host_is_still_reported_when_it_mattered():
    """"Cannot measure" is not "unlimited" either — the user is told."""
    verdict = admission_verdict(5_000_000, 4_000, "huge.csv", None)
    assert verdict.warnings
    assert "could not be read" in " ".join(verdict.warnings)


# ── the memory refusal ───────────────────────────────────────────────────────

def test_a_frame_larger_than_the_host_is_refused_with_the_numbers():
    """A refusal a researcher can act on names all three quantities.

    Shape, need and headroom — because "too large" without them is
    indistinguishable from a bug, and the reader cannot tell whether to subset
    columns or to raise the container's limit.
    """
    verdict = admission_verdict(200_000, 20_000, "atlas.csv", 2 * GB)
    assert verdict.refused
    text = verdict.refusal
    assert "200,000 rows" in text and "20,000 columns" in text
    assert "2.0 GB" in text                       # what was available
    assert "119.2 GB" in text                     # what was estimated
    assert "APP_MEMORY_LIMIT" in text             # a concrete way forward


def test_a_refusal_never_carries_the_load_anyway_affordance():
    """Structural, not a matter of how the page happens to be indented.

    The upload page renders its "Load anyway" checkbox from `warnings`, so a
    refusal that also warned would hand the user a way past a memory refusal.
    The far side of that override is an OOM kill, which reaches the researcher
    as a blank tab with no traceback anywhere they can see.
    """
    memory = admission_verdict(200_000, 20_000, "atlas.csv", 2 * GB)
    bytes_ = prefilter_verdict(int((PREFILTER_REFUSE_MB + 500) * 1024 * 1024), "x.parquet")
    for verdict in (memory, bytes_):
        assert verdict.refused
        assert verdict.warnings == ()


def test_the_estimate_is_cells_times_eight_times_the_safety_factor():
    """The arithmetic the refusal message quotes, pinned.

    Four is the measured multiplier (the audit's 6.6-7.2x RAM per CSV byte,
    corroborated by three live copies of the frame this repo makes on the way
    to a successful add). If it moves, it should move deliberately.
    """
    assert estimated_frame_bytes(1_000, 1_000) == 1_000 * 1_000 * 8 * 4
    assert estimated_frame_bytes(0, 5_000) == 0


# ── and eight bytes a cell is a FLOOR, not the estimate ──────────────────────
#
# The shape rule assumes float64. A table of text costs 40-58 bytes a cell — the
# transpose work in this same branch measured string cells at a constant 40.0
# against 8 — so for an object frame the whole 8x4 budget lands BELOW one
# resting copy, and the app holds three. The gate has the parsed frame in hand
# at both call sites, so it measures rather than assumes.


def _survey_frame(rows=20_000, cols=30):
    """An ordinary categorical export. Nothing exotic, nothing omics."""
    pd = pytest.importorskip("pandas")
    answers = ["strongly agree", "agree", "neutral", "disagree"]
    return pd.DataFrame({f"q{i}": [answers[(i + r) % 4] for r in range(rows)]
                         for i in range(cols)})


def test_a_text_frame_is_measured_rather_than_assumed_to_be_float64():
    """The under-estimate, stated as a ratio so it cannot drift back.

    Measured at 200,000 x 30: 0.32 GB for one copy at rest against a 0.18 GB
    total budget — 0.56x, i.e. the safety factor is entirely consumed before a
    single copy is made, and the docstring claiming object columns were "covered
    by the safety factor" was false by this repo's own number.
    """
    frame = _survey_frame()
    floor = estimated_frame_bytes(*frame.shape)
    measured = measured_frame_bytes(frame)
    assert measured > 3 * floor
    assert measured >= 4 * int(frame.memory_usage(deep=True, index=True).sum())


def test_a_numeric_frame_measures_exactly_what_the_shape_rule_said():
    """The other half: this is a generalization, not a new rule. float64 cells
    really do cost 8, so the two answers agree on the matrix the shape rule was
    written for, and only diverge where it was wrong."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(np.zeros((5_000, 20)))
    floor = estimated_frame_bytes(*frame.shape)
    # The index costs a few hundred bytes on top; the cells are the claim.
    assert floor <= measured_frame_bytes(frame) < floor * 1.01


def test_the_measurement_can_only_ever_refuse_more_than_the_shape_rule():
    """The property that makes this safe to land in a PR that must move no
    existing result: a categorical column weighs little at rest and costs
    dearly once anything copies or encodes it, so `measured_frame_bytes` is
    floored at the shape figure and can never admit what the shape rule
    refused."""
    pd = pytest.importorskip("pandas")
    frames = [_survey_frame(rows=500, cols=8),
              pd.DataFrame({"c": pd.Categorical(["a", "b"] * 5_000)}),
              pd.DataFrame({"n": range(1_000)})]
    for frame in frames:
        assert measured_frame_bytes(frame) >= estimated_frame_bytes(*frame.shape)


def test_the_ordinary_csv_is_still_admitted_in_silence_when_measured():
    """Requirement 5 re-checked against the more accurate number: a 500 x 20
    upload with a text column must still pass without a word."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({**{f"c{i}": [float(r) for r in range(500)]
                             for i in range(19)},
                          "group": ["a", "b"] * 250})
    verdict = admission_verdict(*frame.shape, "study.csv", 8 * GB,
                                estimated_bytes=measured_frame_bytes(frame))
    assert verdict.clean


def test_an_omitted_estimate_still_falls_back_to_the_shape_floor():
    """`estimated_bytes` is optional so shape-only callers and these tests keep
    working; the app never omits it."""
    assert admission_verdict(200_000, 20_000, "atlas.csv", 2 * GB).refused


# ── the width warning ────────────────────────────────────────────────────────

def test_a_wide_frame_that_fits_comfortably_still_warns():
    """Memory is not the only way a frame can be unusable.

    3,000 columns of 1,000 rows is under a hundred megabytes and will load
    instantly — and then EDA's uncapped O(p²) scans will work through four and
    a half million column pairs. This warning is the only protection there is
    until those paths get their caps, so it must not be conditional on memory
    pressure.
    """
    verdict = admission_verdict(1_000, 3_000, "expr.csv", 64 * GB)
    assert not verdict.refused
    assert verdict.warnings
    joined = " ".join(verdict.warnings)
    assert "3,000 columns" in joined
    assert f"{(3_000 * 2_999) // 2:,}" in joined   # the actual pair count


@pytest.mark.parametrize("cols,expected_warning", [
    (WIDE_COLUMN_WARN - 1, False),
    (WIDE_COLUMN_WARN, False),
    (WIDE_COLUMN_WARN + 1, True),
])
def test_the_width_threshold_is_where_it_says_it_is(cols, expected_warning):
    verdict = admission_verdict(1_000, cols, "expr.csv", 512 * GB)
    assert bool(verdict.warnings) is expected_warning


# ── the pre-parse backstop is a backstop, not a byte cap ─────────────────────

def test_the_pre_filter_sits_far_above_any_real_research_file():
    """It exists to stop the app materializing the absurd, not to gate uploads.

    Set anywhere near a real file's size it would be the old byte cap under a
    new name — and it would be worst on parquet, which is routinely ~10x
    compressed and so is the format where bytes predict memory least.
    """
    assert PREFILTER_REFUSE_MB >= 1000
    assert prefilter_verdict(900 * 1024 * 1024, "big_but_real.csv").clean


def test_an_unknown_upload_size_is_not_a_reason_to_refuse():
    """Not every uploader object reports a size. Missing is not enormous."""
    assert prefilter_verdict(None, "x.csv").clean
    assert prefilter_verdict(0, "x.csv").clean


def test_the_server_ceiling_stays_above_the_app_s_own_threshold():
    """The bug that made the old check unreachable, asserted so it cannot recur.

    `.streamlit/config.toml` used to set `maxUploadSize` to exactly the app's
    own 50 MB limit, so Streamlit refused the POST with a generic 413 before a
    line of page code ran and the app's message was dead code. If these two
    numbers ever meet again, the same thing happens to `PREFILTER_REFUSE_MB`.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(root, ".streamlit", "config.toml"),
                 os.path.join(root, "Dockerfile")):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        sizes = [int(line.split("=", 1)[1].strip())
                 for line in text.splitlines()
                 if line.strip().startswith("maxUploadSize")]
        assert sizes, f"no maxUploadSize found in {path}"
        for size in sizes:
            assert size > PREFILTER_REFUSE_MB, (
                f"{path} caps uploads at {size} MB, at or below the app's own "
                f"{PREFILTER_REFUSE_MB} MB threshold — the server would refuse "
                f"first and the app's explanation would never render.")


def test_a_clean_verdict_is_the_default_shape():
    """`Verdict()` with nothing said is the ordinary case, and is falsy-clean."""
    assert Verdict().clean
    assert not Verdict().refused



# ── Excel says its price before it is parsed ─────────────────────────────────
#
# The shape gate runs after the parse, and for Excel the parse is the cost:
# measured ~7 s per million cells on the audit's box and 10.1 s on this one,
# 56x slower than CSV. An .xlsx sheet's cell count is readable from its
# dimension record without parsing a cell, so the price can be stated before
# the wait rather than discovered during it.

import io                                                         # noqa: E402
import pathlib                                                    # noqa: E402

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import inspect                                                    # noqa: E402

from utils.admission import (                                     # noqa: E402
    EXCEL_QUIET_BELOW_BYTES, EXCEL_QUIET_BELOW_CELLS,
    EXCEL_SECONDS_PER_MILLION_CELLS, excel_batch_price, excel_parse_seconds,
    excel_price, excel_sheet_cells,
)


PROJECT_ROOT_FOR_PIN = pathlib.Path(__file__).resolve().parent.parent


def _workbook(**sheets):
    """An .xlsx in memory with the given sheets, in the given order."""
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def test_the_cell_count_comes_from_the_dimension_record_of_the_sheet_asked_for():
    """Sheets are indexed in the workbook's own order — the order the page's
    selector offers — and the count includes the header row the parser reads."""
    book = _workbook(
        second=pd.DataFrame({"a": [1, 2, 3]}),
        first=pd.DataFrame({"a": range(100), "b": range(100), "c": range(100)}),
    )
    assert excel_sheet_cells(book, 0) == 4 * 1
    assert excel_sheet_cells(book, 1) == 101 * 3
    assert excel_sheet_cells(book, 2) is None
    assert pd.ExcelFile(io.BytesIO(book)).sheet_names == ["second", "first"]


def test_the_count_matches_what_the_parser_then_reads():
    frame = pd.DataFrame(np.random.RandomState(0).rand(240, 7))
    book = _workbook(data=frame)
    parsed = pd.read_excel(io.BytesIO(book))
    assert excel_sheet_cells(book, 0) == (parsed.shape[0] + 1) * parsed.shape[1]


@pytest.mark.parametrize("payload", [
    b"\xd0\xcf\x11\xe0 an old-format .xls is not a zip",
    b"",
    b"PK\x03\x04 a zip header and nothing else",
])
def test_a_workbook_that_cannot_be_counted_is_none_not_an_error(payload):
    assert excel_sheet_cells(payload, 0) is None


def test_the_two_measured_rates_are_the_range_at_one_million_cells():
    lo, hi = excel_parse_seconds(1_000_000)
    assert (lo, hi) == EXCEL_SECONDS_PER_MILLION_CELLS == (7.0, 10.0)


def test_the_exponent_is_superlinear():
    """Ten times the cells costs more than ten times the seconds — the audit's
    1.21, and the reason Excel is the ingest path worth pricing."""
    lo1, _ = excel_parse_seconds(1_000_000)
    lo10, _ = excel_parse_seconds(10_000_000)
    assert 10 * lo1 < lo10 < 20 * lo1


def test_an_ordinary_spreadsheet_is_not_priced():
    assert excel_price(200 * 1024, "survey.xlsx", 5_000) is None
    assert excel_price(200 * 1024, "survey.xls", None) is None
    assert excel_price(EXCEL_QUIET_BELOW_BYTES - 1, "survey.xls", None) is None


def test_a_large_sheet_is_priced_in_cells_and_minutes_before_the_parse():
    note = excel_price(150 * 1024 * 1024, "expression.xlsx", 12_000_000)
    assert "**expression.xlsx**" in note
    assert "12,000,000 cells" in note
    assert "7 to 10 seconds per million cells" in note
    lo, hi = excel_parse_seconds(12_000_000)
    assert f"about {round(lo / 60)} to {round(hi / 60)} minutes" in note
    assert "CSV" in note and "fifty times faster" in note


def test_a_workbook_that_could_not_be_counted_is_priced_on_its_bytes():
    note = excel_price(40 * 1024 * 1024, "legacy.xls", None)
    assert "**legacy.xls** is a 40 MB Excel workbook" in note
    assert "could not be counted" in note
    assert "minutes rather than seconds" in note


def test_the_batch_price_sums_the_counted_sheets_and_names_the_rest():
    note = excel_batch_price([
        ("a.xlsx", 600_000, 8 * 1024 * 1024),
        ("b.xlsx", 2_000_000, 20 * 1024 * 1024),
        ("c.xls", None, 9 * 1024 * 1024),
        ("tiny.xls", None, 100 * 1024),
    ])
    assert "2 of these are Excel workbooks with 2,600,000 cells between them" in note
    assert "c.xls is a large Excel workbook" in note
    assert "tiny.xls" not in note


def test_a_batch_of_small_spreadsheets_is_not_priced():
    assert excel_batch_price([("a.xlsx", 1_000, 50_000), ("b.xlsx", None, 1_000)]) is None
    assert excel_batch_price([]) is None


def test_the_price_sits_above_the_parse_in_both_upload_paths():
    """Source order, pinned: a price stated after the parse is a receipt."""
    src = (PROJECT_ROOT_FOR_PIN / "pages" / "01_Upload_and_Audit.py").read_text(encoding="utf-8")
    bulk = src.index('if st.button(f"Add all {len(uploaded_files)} files to project"')
    assert 0 < src.index("excel_batch_price(") < bulk
    expander = src.index('with st.expander(f"Configure: {uploaded_file.name}"')
    price = src.index("excel_price(", expander)
    parse = src.index("df_preview = cached_parse_upload(", expander)
    assert expander < price < parse
