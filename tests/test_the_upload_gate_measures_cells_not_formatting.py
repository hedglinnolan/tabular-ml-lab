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


# ── the combined table is admitted at the join, not only at the door ─────────
#
# The gate above decides per file. Step 2 then links or stacks admitted files
# into one working table, and the combine preview runs the REAL merge on every
# rerun — so two files that each fit could be built into one that does not,
# with the memory spent before anything could refuse. The projection is taken
# from the change map's predicted shape and the inputs' real dtypes, before the
# merge, and the verdict follows the door's rules.

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from utils.admission import (                                     # noqa: E402
    combination_verdict, projected_bytes_per_row, projected_join_bytes,
    projected_stack_bytes,
)


def _numeric(rows, cols, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.rand(rows, cols), columns=[f"x{i}" for i in range(cols)])
    df.insert(0, "id", np.arange(rows))
    return df


def test_a_numeric_row_weighs_eight_bytes_a_cell():
    df = _numeric(500, 9)                     # 10 columns with the id
    assert projected_bytes_per_row(df) == 10 * 8


def test_a_text_row_weighs_what_its_strings_weigh_not_eight():
    df = pd.DataFrame({"id": range(200), "site": ["a long site name"] * 200})
    assert projected_bytes_per_row(df) > 2 * 8


def test_a_link_is_projected_as_both_sides_times_the_predicted_rows():
    left, right = _numeric(1_000, 9), _numeric(1_000, 9, seed=1)
    # 1,000 matched rows, 10 + 10 - 1 shared key = 19 columns.
    projected = projected_join_bytes(left, right, 1_000, 19)
    per_row = projected_bytes_per_row(left) + projected_bytes_per_row(right)
    assert projected == int(per_row * 1_000 * 4)
    assert projected >= estimated_frame_bytes(1_000, 19)


def test_a_many_to_many_link_is_projected_on_the_rows_it_will_produce():
    """The change map predicts the multiplied row count; the projection has to
    use that, not the inputs' sizes, or a key that pairs every row with every
    row is admitted on the size of the two small files that cause it."""
    left, right = _numeric(1_000, 9), _numeric(1_000, 9, seed=1)
    modest = projected_join_bytes(left, right, 1_000, 19)
    multiplied = projected_join_bytes(left, right, 1_000 * 1_000, 19)
    assert multiplied == modest * 1_000


def test_a_stack_is_projected_as_every_row_plus_the_blanks_plus_the_source_column():
    a, b = _numeric(600, 9), _numeric(400, 9, seed=1)
    b["only_in_b"] = 1.0
    cm_rows, cm_cols = 1_000, 12                 # union of 11 columns + source
    projected = projected_stack_bytes({"a": a, "b": b}, cm_rows, cm_cols)
    own_rows = projected_bytes_per_row(a) * 600 + projected_bytes_per_row(b) * 400
    blanks = 600 * 1 * 8                          # a lacks one column of b
    source = 1_000 * 40
    assert projected == int((own_rows + blanks + source) * 4)
    assert projected >= estimated_frame_bytes(cm_rows, cm_cols)


@pytest.mark.parametrize("available", [None, 2 * GB, 32 * GB])
def test_an_ordinary_link_is_admitted_in_silence(available):
    left, right = _numeric(500, 6), _numeric(500, 6, seed=1)
    projected = projected_join_bytes(left, right, 500, 13)
    verdict = combination_verdict(500, 13, projected, available,
                                  "linking demo (500 rows) with labs (500 rows)")
    assert verdict.clean


def test_two_files_that_each_fit_are_refused_when_their_link_would_not():
    """The case the door cannot see: each side is admitted on its own shape,
    and the link of the two exceeds the headroom. The refusal is at the join,
    before the merge, and it carries the numbers."""
    available = 64 * 2 ** 20                     # 64 MB free
    left, right = _numeric(20_000, 49), _numeric(20_000, 49, seed=1)
    for side in (left, right):
        assert not admission_verdict(*side.shape, "side.csv", available,
                                     estimated_bytes=measured_frame_bytes(side)).refused, \
            "each file fits on its own — that is the premise"
    # A link on a key that pairs every row with every row: 400M rows.
    projected = projected_join_bytes(left, right, 20_000 * 20_000, 99)
    verdict = combination_verdict(20_000 * 20_000, 99, projected, available,
                                  "linking left (20,000 rows) with right (20,000 rows)")
    assert verdict.refused
    assert "400,000,000 rows x 99 columns" in verdict.refusal
    assert "GB to work with" in verdict.refusal
    assert "0.1 GB is available" in verdict.refusal
    assert "was not built" in verdict.refusal
    assert verdict.warnings == (), "a refusal carries no load-anyway affordance"


def test_an_unmeasurable_host_never_refuses_a_link_and_says_so_when_it_mattered():
    left, right = _numeric(20_000, 49), _numeric(20_000, 49, seed=1)
    big = projected_join_bytes(left, right, 20_000 * 200, 99)   # ~12 GB projected
    verdict = combination_verdict(20_000 * 200, 99, big, None,
                                  "linking left (20,000 rows) with right (20,000 rows)")
    assert not verdict.refused
    assert len(verdict.warnings) == 1
    assert "could not be read" in verdict.warnings[0]
    assert "4,000,000 rows x 99 columns" in verdict.warnings[0]


def test_a_combined_table_wider_than_the_analysis_pages_are_built_for_is_warned():
    left, right = _numeric(300, 1_200), _numeric(300, 1_200, seed=1)
    cols = 1_201 + 1_201 - 1
    verdict = combination_verdict(300, cols, projected_join_bytes(left, right, 300, cols),
                                  32 * GB, "linking a (300 rows) with b (300 rows)")
    assert not verdict.refused
    assert any(f"{cols:,} columns" in w and "O(columns" in w for w in verdict.warnings)


def test_the_description_opens_the_sentence_in_the_researchers_terms():
    verdict = combination_verdict(10 ** 9, 50, 10 ** 12, GB,
                                  "stacking 3 files (1,000,000,000 rows in all)")
    assert verdict.refusal.startswith("**Not combined.** Stacking 3 files")


# ── and the real combine step refuses BEFORE it runs the merge ───────────────

class _Stub:
    """Enough of Streamlit for `render_combine_step` to run; everything said
    is recorded so the test can assert on the refusal it rendered."""

    def __init__(self, relation_index=0):
        self.session_state = {}
        self.said = []
        self.relation_index = relation_index

    def selectbox(self, label, options, index=0, **kw):
        return list(options)[index or 0]

    def radio(self, label, options, index=0, **kw):
        if label == "How do these files relate?":
            return list(options)[self.relation_index]
        return list(options)[index]

    def multiselect(self, label, options, default=None, **kw):
        return list(default if default is not None else options)

    def button(self, *a, **kw):
        return True

    def columns(self, spec, **kw):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def _record(*a, **kw):
            self.said.append((name, a[0] if a else None))
            return self
        return _record


def _combine_step(monkeypatch, frames, available, relation_index=0):
    import utils.combine_ui as combine_ui

    stub = _Stub(relation_index=relation_index)
    monkeypatch.setattr(combine_ui, "st", stub)
    monkeypatch.setattr(combine_ui, "available_memory_bytes", lambda: available)

    def _never(*a, **kw):
        raise AssertionError("the merge ran on a table that was refused")
    if available is not None and available < 2 ** 20:
        monkeypatch.setattr(combine_ui, "execute_join", _never)
        monkeypatch.setattr(combine_ui, "execute_stack", _never)
    return combine_ui.render_combine_step(frames), stub


def test_the_link_is_refused_before_the_merge_runs(monkeypatch):
    demo = pd.DataFrame({"SEQN": [1, 2, 3], "age": [50, 60, 70]})
    labs = pd.DataFrame({"SEQN": [1, 2, 3], "chol": [4.0, 5.0, 6.0]})
    out, stub = _combine_step(monkeypatch, {"demo": demo, "labs": labs}, available=1)
    assert out is None, "nothing to commit"
    errors = [msg for kind, msg in stub.said if kind == "error"]
    assert any("**Not combined.** Linking demo (3 rows) with labs (3 rows)" in e
               for e in errors), errors


def test_the_stack_is_refused_before_it_runs(monkeypatch):
    a = pd.DataFrame({"SEQN": [1, 2, 3], "age": [50, 60, 70]})
    b = pd.DataFrame({"SEQN": [4, 5, 6], "age": [51, 61, 71]})
    out, stub = _combine_step(monkeypatch, {"a": a, "b": b}, available=1,
                              relation_index=1)
    assert out is None
    errors = [msg for kind, msg in stub.said if kind == "error"]
    assert any("**Not combined.** Stacking 2 files (6 rows in all)" in e
               for e in errors), errors


def test_an_ordinary_link_still_combines(monkeypatch):
    demo = pd.DataFrame({"SEQN": [1, 2, 3], "age": [50, 60, 70]})
    labs = pd.DataFrame({"SEQN": [1, 2, 3], "chol": [4.0, 5.0, 6.0]})
    out, stub = _combine_step(monkeypatch, {"demo": demo, "labs": labs},
                              available=32 * GB)
    assert out is not None and "chol" in out.columns
    assert not [msg for kind, msg in stub.said if kind == "error"]


def test_the_admission_sits_above_the_merge_in_every_path():
    """Source order, pinned: the projection is only worth anything if it runs
    before the merge or the stack it is about to refuse."""
    import inspect
    import utils.combine_ui as combine_ui

    link = inspect.getsource(combine_ui._render_link)
    assert link.index("_admit_combined(") < link.index("execute_join(")
    stack = inspect.getsource(combine_ui._render_stack)
    assert stack.index("_admit_combined(") < stack.index("execute_stack(")
    grouped = inspect.getsource(combine_ui._render_grouped)
    assert grouped.index("_admit_combined(") < grouped.index("execute_stack(")
