"""Whether an uploaded frame is admitted, decided on its shape.

**The bug this replaces.** The upload page gated on `uploaded_file.size` — the
number of bytes of *text* — against a 50 MB cap. Bytes of text are a proxy for
the wrong variable. One identical 20,000 x 300 matrix, which becomes the same
47.0 MB DataFrame however it is written down, measured 19.66 MB on disk as
integer counts, 40.44 MB at four decimal places, and 106.32 MB at pandas'
default float repr. Same data, same memory, three different verdicts, and the
only thing that changed was formatting. Cells predict memory; characters do
not, and a gene expression matrix written out at full precision was refused for
being *legible*.

**And the check could never fire anyway.** `.streamlit/config.toml` set
`maxUploadSize` to the same 50, and Streamlit refuses the POST with a generic
413 before a line of app code runs — so the friendly warning and its "Load
anyway" escape hatch were unreachable for every file they were written to
catch. Raising the server ceiling *above* the app's own thresholds is what
makes the app's messages reachable at all, and is why the two numbers must
never be equal again.

**What lives here and why it is not in the page.** These are pure functions of
a shape and a memory reading: no Streamlit, no session state, no widgets. That
keeps them testable without a browser or a script run, and it lets both ingest
paths on the upload page — the per-file expander and the "Add all N files"
button, which had no size check of any kind — reach one identical decision
instead of two that drift.

Deciding is this module's job; measuring is `utils.host_resources`'s, and it is
emphatic that its `None` means *cannot estimate* and never *unlimited*. The
rule below honors that in the only direction that is safe for a researcher: a
failed probe never refuses a file, and never silently pretends the ceiling was
checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: Bytes per cell in the shape-only floor. A pandas frame of float64 or int64 —
#: what a numeric research matrix parses to — costs exactly this per cell, so it
#: is the right base figure rather than a guess.
#:
#: It is a FLOOR and not the estimate. An object column costs far more: this
#: PR's own transpose work measured string cells at a constant 40.0 bytes
#: against 8, and an ordinary 200,000 x 30 survey export of short categorical
#: answers measures 57.5 bytes/cell at rest — so for that frame 8 x 4 is 0.56x
#: of ONE resting copy, and the app holds three. Anything holding the frame
#: therefore calls `measured_frame_bytes`, which reads the real dtypes; this
#: figure is what remains when only a shape is known.
_BYTES_PER_CELL = 8

#: How many times the dense-cell figure a frame actually costs to get through
#: this app, and the one number here that came from measurement rather than
#: arithmetic.
#:
#: The audit measured 6.6-7.2x resident RAM per byte of CSV through the real
#: app path. That ratio is against CSV bytes and this estimate is against
#: cells, so it is corroboration rather than the constant itself — but the
#: copies behind it are countable in this repository, which is the reason 4 is
#: defensible and not merely cautious. `cached_parse_upload` is decorated
#: `@st.cache_data` (`utils/perf_cache.py`), so the parsed frame is retained in
#: the cache AND a copy is handed back on every call; `_commit_dataset` in
#: `pages/01_Upload_and_Audit.py` does `df = df.copy()` again before
#: registering it. That is three live copies at the moment of a successful add,
#: before the audit tables that run next allocate anything. 4 leaves one
#: copy's worth of margin for them.
#:
#: Deliberately a single flat factor. A per-dtype model would be more accurate
#: and would be a threshold nobody had measured, baked into the abstraction
#: every future caller reads first.
_MEMORY_SAFETY_FACTOR = 4

#: Column count above which the analysis pages are warned about.
#:
#: Not a memory number — a frame this wide can fit in RAM comfortably and still
#: make the app unusable. The EDA and explainability pages carry uncapped
#: O(p^2) work (a full correlation matrix, pairwise scans), and this PR is
#: deliberately not touching those paths: their caps depend on timings the
#: audit never measured, and guessing them here would bake unverified numbers
#: into the abstraction everything downstream reads. Until that measurement
#: exists, telling the user plainly is the only protection there is, so this
#: warning is load-bearing rather than advisory.
WIDE_COLUMN_WARN = 2000

#: File size above which the file is refused WITHOUT being parsed.
#:
#: A backstop, not a return of the byte cap. Bytes are still a proxy for the
#: wrong variable, and this number is set high enough that no real research
#: file reaches it — its only job is to stop the app materializing something
#: absurd purely to discover it cannot hold it, because the shape gate cannot
#: run until the frame exists. It is loose in the one direction that matters:
#: parquet is routinely ~10x compressed, so a 1500 MB parquet is a frame no
#: byte figure would have predicted, and the shape gate is what catches it a
#: moment later.
#:
#: Kept BELOW the server's `maxUploadSize` on purpose. When the two numbers
#: match, the server refuses first with a generic error and the app's own
#: explanation is dead code — which is exactly the bug this module replaces.
PREFILTER_REFUSE_MB = 1500

#: Estimated frame size below which a failed memory probe is not worth
#: mentioning. An unmeasurable host is only interesting if there was a real
#: question to answer, and a frame this small fits on any machine that can load
#: pandas, scikit-learn and Streamlit at all — so warning about it would put
#: friction on the ordinary 500-row CSV to report a risk that does not exist.
#: Above it, the user is told the ceiling could not be checked.
_UNMEASURED_QUIET_BELOW_BYTES = 1024 ** 3


@dataclass(frozen=True)
class Verdict:
    """One admission decision: a refusal, or zero or more things to say.

    The two fields are mutually exclusive by construction, and that is the
    point rather than a convention. The old code paired its warning with a
    "Load anyway" checkbox; a shape gate that refuses on memory must not be
    overridable that way, because the thing on the far side of the override is
    an OOM kill that reaches the researcher as a blank browser tab. A refusal
    is therefore returned with an empty `warnings`, so a caller rendering the
    checkbox from `warnings` cannot reach it from a refusal even by mistake.
    """

    refusal: Optional[str] = None
    warnings: Tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    @property
    def clean(self) -> bool:
        """Nothing to show the user at all — the ordinary case."""
        return self.refusal is None and not self.warnings


def estimated_frame_bytes(rows: int, cols: int) -> int:
    """The floor: what a rows x cols frame of NUMBERS costs to get through here.

    Dense cells times the measured multiplier. Not what the frame weighs at
    rest — what peak resident memory reaches while the app parses, caches,
    copies and audits it.

    Correct only for the numeric matrix it describes, and an under-estimate for
    anything holding strings. A caller that has the frame should call
    `measured_frame_bytes` instead; this is for a caller that has only a shape.
    """
    return int(max(rows, 0) * max(cols, 0) * _BYTES_PER_CELL * _MEMORY_SAFETY_FACTOR)


def measured_frame_bytes(df) -> int:
    """The same estimate, taken from the frame's real dtypes rather than assumed.

    `memory_usage(deep=True)` follows every object column out to the Python
    strings it points at, which is the whole difference: a categorical or
    free-text export costs 40-58 bytes per cell where the shape floor assumes 8,
    so gating on shape alone under-estimates an ordinary survey table by ~7x and
    admits it silently. The frame already exists at both call sites — it has to,
    because a transposed file's true width is not known until it is parsed — so
    this costs a walk over data that is already resident and guesses nothing.

    Never returns less than `estimated_frame_bytes`. The floor stays in force
    because *at rest* is not the quantity being budgeted: a categorical column
    weighs almost nothing until something copies, encodes or one-hots it, and a
    frame that measures small must not thereby be admitted where the same shape
    of float64 would have been refused. So this can only ever refuse MORE than
    the shape rule did, never less.
    """
    try:
        at_rest = int(df.memory_usage(deep=True, index=True).sum())
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - exotic dtype
        # An extension dtype that cannot report itself is not a reason to refuse
        # a file or to crash an upload; fall back to the floor, which is exactly
        # the answer this module gave before it could measure at all.
        at_rest = 0
    rows, cols = (df.shape + (0, 0))[:2]
    return max(int(at_rest * _MEMORY_SAFETY_FACTOR),
               estimated_frame_bytes(rows, cols))


def _gb(n: float) -> str:
    return f"{n / (1024 ** 3):.1f} GB"


def prefilter_verdict(size_bytes: Optional[int], filename: str) -> Verdict:
    """The cheap pre-parse backstop. Refuses only the genuinely absurd.

    Takes `Optional[int]` because an uploader object is not guaranteed to
    report a size; an unknown size is not a reason to refuse, so it passes.
    """
    if not size_bytes:
        return Verdict()
    size_mb = size_bytes / (1024 * 1024)
    if size_mb <= PREFILTER_REFUSE_MB:
        return Verdict()
    return Verdict(refusal=(
        f"**{filename}** is {size_mb:,.0f} MB, past the {PREFILTER_REFUSE_MB:,} MB "
        f"point at which this app will not attempt to read a file at all — "
        f"reading it is what would exhaust memory, so it is stopped before that "
        f"rather than after. Split the file by rows, or subset the columns you "
        f"intend to analyze, and upload the pieces; Step 2 combines them again."
    ))


def admission_verdict(rows: int, cols: int, filename: str,
                      available_bytes: Optional[int],
                      estimated_bytes: Optional[int] = None) -> Verdict:
    """Decide on a parsed frame's shape against a memory reading.

    `available_bytes` is `utils.host_resources.available_memory_bytes()` —
    passed in rather than probed here so the decision stays a pure function of
    its inputs, and so a caller handling several files reads the host once.

    `estimated_bytes` is `measured_frame_bytes(df)` when the caller holds the
    frame, which both real call sites do. Omitting it falls back to the
    shape-only floor — right for a numeric matrix, and an under-estimate for a
    frame of strings, so it is a default for tests and shape-only callers rather
    than the path the app takes.

    **`None` warns and proceeds; it never refuses.** The probe returns `None`
    for *cannot estimate* — a lean install without psutil, or a locked-down
    host where reading `/proc` raises. Refusing a researcher's file because a
    memory probe failed is a worse and far more common failure than the OOM it
    would be guessing at, and it would hit hardest on the minimal installs
    least able to diagnose it. But "cannot measure" is not "unlimited" either,
    so a frame large enough for the answer to have mattered is told plainly
    that the ceiling was not checked.
    """
    est = (estimated_frame_bytes(rows, cols) if estimated_bytes is None
           else max(int(estimated_bytes), 0))
    shape = f"{rows:,} rows x {cols:,} columns"

    if available_bytes is not None and est > available_bytes:
        # No warnings alongside a refusal: see `Verdict`. The workaround has to
        # be concrete, and it has to include getting the file back out of the
        # uploader — the parse that produced this shape is still held in
        # `cached_parse_upload`'s cache, so the memory this refusal is about is
        # already spent until the file is removed.
        return Verdict(refusal=(
            f"**{filename}** is too large for this machine to analyze. "
            f"It parsed to {shape}, which needs about {_gb(est)} to work with, "
            f"and only {_gb(available_bytes)} is available.\n\n"
            f"Remove the file from the uploader above to release the memory the "
            f"parse already used, then try one of:\n\n"
            f"- upload a subset of the columns you actually intend to model on;\n"
            f"- split the file by rows and combine the pieces in Step 2;\n"
            f"- give the app more memory — in Docker, raise `APP_MEMORY_LIMIT` "
            f"(see `UNIVERSITY_DEPLOYMENT.md`) and restart the container."
        ))

    warnings = []

    if available_bytes is None and est >= _UNMEASURED_QUIET_BELOW_BYTES:
        warnings.append(
            f"**{filename}** is {shape} and needs roughly {_gb(est)} to work "
            f"with, but this machine's free memory could not be read, so that "
            f"was not checked against anything. Loading it anyway is fine; if "
            f"the app becomes unresponsive or the tab goes blank, this is why."
        )

    if cols > WIDE_COLUMN_WARN:
        # Specific on purpose. "May be slow" is what the old warning said and
        # it told the user nothing they could act on.
        pairs = (cols * (cols - 1)) // 2
        warnings.append(
            f"**{filename}** has {cols:,} columns. It will load, but the "
            f"analysis pages are not built for this width yet: EDA's "
            f"correlation and pairwise scans are O(columns²) and uncapped, so "
            f"they will work through {pairs:,} column pairs and can take many "
            f"minutes per page — Explainability longer still. Nothing will "
            f"stop you or warn you again. Cutting to the columns you intend to "
            f"model on before Step 2 is the difference between minutes and "
            f"seconds."
        )

    return Verdict(warnings=tuple(warnings))


# ── the combined table: admitted at the join, not only at the door ───────────
#
# The gate above decides per uploaded file. Step 2 then links or stacks the
# admitted files into one working table, and until now nothing measured THAT
# frame before it was built. Two files that each fit can add up to one that
# does not — a link on a many-to-many key multiplies rows, an outer link and a
# stack add every row of every file — and the combine preview runs the real
# merge on every rerun, so the memory was spent before any check could have
# run. These are projections from the inputs' real dtypes and the change map's
# predicted shape, taken before the merge, so a refusal costs nothing and a
# clean verdict costs one pass over data that is already resident.

#: What one string cell costs at rest, measured by this module's own transpose
#: work at a constant 40.0 bytes against 8 for a number. Used for the
#: provenance column a stack adds (one short filename per row).
_STRING_CELL_BYTES = 40


def projected_bytes_per_row(df) -> float:
    """What one row of `df` weighs at rest, from its real dtypes.

    The index is excluded: a stack renumbers it and a link rebuilds it, so the
    inputs' indexes do not survive into the combined table. Never less than
    the shape floor's per-row figure, for the same reason `measured_frame_bytes`
    keeps its floor — a categorical column weighs almost nothing until
    something encodes it.
    """
    rows = max(int(df.shape[0]), 1)
    cols = int(df.shape[1])
    try:
        at_rest = int(df.memory_usage(deep=True, index=False).sum())
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - exotic dtype
        at_rest = 0
    return max(at_rest / rows, cols * _BYTES_PER_CELL)


def projected_join_bytes(left, right, after_rows: int, after_cols: int) -> int:
    """Peak bytes a link of `left` and `right` would cost, projected before it runs.

    A linked row carries one row of each side, so its cost is the sum of the
    two sides' per-row costs (the shared key is counted twice, which
    over-estimates by one column — the safe direction), times the rows the
    change map predicts, times the same multiplier `measured_frame_bytes`
    applies. Never less than the shape floor for the predicted shape.
    """
    per_row = projected_bytes_per_row(left) + projected_bytes_per_row(right)
    rows = max(int(after_rows), 0)
    return max(int(per_row * rows * _MEMORY_SAFETY_FACTOR),
               estimated_frame_bytes(rows, after_cols))


def projected_stack_bytes(frames: Dict[str, "object"], after_rows: int,
                          after_cols: int) -> int:
    """Peak bytes a stack of `frames` would cost, projected before it runs.

    Every row keeps its own file's weight; a column a file lacks becomes a
    blank cell in that file's rows (8 bytes each, since concat upcasts to a
    float that can hold the blank); and the provenance column the stack adds is
    one short string per row.
    """
    parts = [f for f in frames.values() if f is not None]
    union = set()
    for f in parts:
        union.update(str(c) for c in f.columns)
    total = 0.0
    for f in parts:
        rows = int(f.shape[0])
        total += projected_bytes_per_row(f) * rows
        total += max(len(union) - int(f.shape[1]), 0) * _BYTES_PER_CELL * rows
    rows_after = max(int(after_rows), 0)
    total += rows_after * _STRING_CELL_BYTES
    return max(int(total * _MEMORY_SAFETY_FACTOR),
               estimated_frame_bytes(rows_after, after_cols))


def combination_verdict(rows: int, cols: int, projected_bytes: int,
                        available_bytes: Optional[int], description: str) -> Verdict:
    """Decide on the combined table's projected shape before it is built.

    `description` reads as a gerund phrase in the researcher's terms —
    "linking demographics (600 rows) with labs (480 rows)" — so the message can
    say what was not done. The rules are the door's rules: a refusal on memory
    carries no override and no warnings; `None` for available memory warns
    when the answer would have mattered and never refuses; a table wider than
    `WIDE_COLUMN_WARN` is told what the analysis pages will do with it.
    """
    est = max(int(projected_bytes), 0)
    shape = f"{rows:,} rows x {cols:,} columns"
    what = description.strip() or "combining these files"

    if available_bytes is not None and est > available_bytes:
        return Verdict(refusal=(
            f"**Not combined.** {what[0].upper() + what[1:]} would produce "
            f"{shape}, which needs about {_gb(est)} to work with, and only "
            f"{_gb(available_bytes)} is available. Each file fit on its own; "
            f"the combined table would not, so it was not built.\n\n"
            f"Try one of:\n\n"
            f"- keep only the people found in every file, if you were keeping "
            f"everyone — a table with fewer rows costs less;\n"
            f"- cut each file to the columns you intend to model on and "
            f"upload those instead;\n"
            f"- give the app more memory — in Docker, raise `APP_MEMORY_LIMIT` "
            f"(see `UNIVERSITY_DEPLOYMENT.md`) and restart the container."
        ))

    warnings = []

    if available_bytes is None and est >= _UNMEASURED_QUIET_BELOW_BYTES:
        warnings.append(
            f"{what[0].upper() + what[1:]} would produce {shape} and needs "
            f"roughly {_gb(est)} to work with, but this machine's free memory "
            f"could not be read, so that was not checked against anything. "
            f"Combining anyway is fine; if the app becomes unresponsive or the "
            f"tab goes blank afterwards, this is why."
        )

    if cols > WIDE_COLUMN_WARN:
        pairs = (cols * (cols - 1)) // 2
        warnings.append(
            f"The combined table would have {cols:,} columns. It will be "
            f"built, but the analysis pages are not built for this width yet: "
            f"EDA's correlation and pairwise scans are O(columns²) and "
            f"uncapped, so they will work through {pairs:,} column pairs and "
            f"can take many minutes per page — Explainability longer still. "
            f"Cutting each file to the columns you intend to model on before "
            f"combining is the difference between minutes and seconds."
        )

    return Verdict(warnings=tuple(warnings))




# ── Excel: the one ingest path whose price is worth quoting before the parse ─
#
# The shape gate above runs after the parse, because a frame's true shape is
# not known until it exists. For CSV that is fine — read_csv measured ~2 s per
# 100 MB. Excel is different: the audit measured ~7 s per million cells, 56x
# slower than CSV and the only ingest path with a superlinear exponent (1.21),
# so a large workbook is parsed in full, for minutes, before anything at all
# can be said about it. An .xlsx sheet's cell count can be read WITHOUT
# parsing a cell: the worksheet part opens with a `<dimension ref="A1:AX20001"/>`
# record, and reading it costs a zip directory lookup and a few KB — measured
# 3 ms against 10 s for the parse of the same 12 MB workbook.

#: Seconds to parse one million .xlsx cells, measured twice on the same
#: 20,000 x 50 float sheet: ~7 s on the audit's 14-core box and 10.1 s on the
#: 8-core Windows laptop this app targets. Two machines, so a range rather
#: than a constant — the exponent below is portable and the coefficient is
#: not, which is the rule the runtime estimator (F-10) is built on.
EXCEL_SECONDS_PER_MILLION_CELLS = (7.0, 10.0)

#: The audit's fitted exponent for Excel ingest in cells. Superlinear, and the
#: only ingest path that is.
EXCEL_PARSE_EXPONENT = 1.21

#: Below this many cells the parse is a few seconds and a notice is friction
#: on the ordinary spreadsheet. 500,000 cells is ~4-5 s at the measured rates.
EXCEL_QUIET_BELOW_CELLS = 500_000

#: The same threshold for a workbook whose cells cannot be counted — a .xls is
#: not a zip, and an .xlsx from an unusual writer may carry no dimension
#: record. The float sheet above measured 12.6 bytes per cell on disk, so
#: this is roughly the byte size of the cell threshold.
EXCEL_QUIET_BELOW_BYTES = 6 * 1024 * 1024


def _column_number(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def excel_sheet_cells(file_bytes: bytes, sheet: int = 0) -> Optional[int]:
    """Cells in one sheet of an .xlsx, from the sheet's dimension record.

    The sheet index follows the workbook's own order (`xl/workbook.xml`,
    resolved through its relationships file), which is the order
    `pd.ExcelFile.sheet_names` reports and the page's selector offers. The
    count includes the header row, because the parser reads it too.

    `None` for anything that is not a readable .xlsx — a .xls is not a zip —
    or for a sheet part with no dimension record; the caller then falls back
    to the byte size. Never raises: a workbook this cannot read is exactly the
    workbook `pd.read_excel` is about to be handed, and the parse's own error
    is the one worth showing.
    """
    import io
    import re
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            names = set(archive.namelist())
            if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
                return None
            workbook = archive.read("xl/workbook.xml").decode("utf-8", "replace")
            sheet_ids = re.findall(r'<(?:\w+:)?sheet\b[^>]*?\b\w+:id="([^"]+)"', workbook)
            if not 0 <= int(sheet) < len(sheet_ids):
                return None
            rels = archive.read("xl/_rels/workbook.xml.rels").decode("utf-8", "replace")
            target = None
            for element in re.findall(r"<Relationship\b[^>]*/?>", rels):
                if re.search(r'\bId="%s"' % re.escape(sheet_ids[int(sheet)]), element):
                    found = re.search(r'\bTarget="([^"]+)"', element)
                    target = found.group(1) if found else None
                    break
            if not target:
                return None
            part = target.lstrip("/")
            if not part.startswith("xl/"):
                part = "xl/" + part
            if part not in names:
                return None
            with archive.open(part) as handle:
                head = handle.read(16 * 1024)
    except Exception:
        return None
    found = re.search(rb'<dimension\s+ref="([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?"', head)
    if not found:
        return None
    first_col, first_row, last_col, last_row = found.groups()
    if last_col is None:
        return 1
    cols = _column_number(last_col.decode()) - _column_number(first_col.decode()) + 1
    rows = int(last_row) - int(first_row) + 1
    return max(rows, 0) * max(cols, 0)


def excel_parse_seconds(cells: int) -> Tuple[float, float]:
    """The measured range of seconds to parse `cells` of .xlsx: slow box, fast box."""
    millions = max(int(cells), 0) / 1e6
    lo, hi = EXCEL_SECONDS_PER_MILLION_CELLS
    return (lo * millions ** EXCEL_PARSE_EXPONENT, hi * millions ** EXCEL_PARSE_EXPONENT)


def _span(lo_seconds: float, hi_seconds: float) -> str:
    """'about 40 seconds', 'about 1 to 2 minutes', 'about 1.2 to 1.8 hours'."""
    if hi_seconds < 90:
        lo, hi = int(round(lo_seconds / 5.0) * 5), int(round(hi_seconds / 5.0) * 5)
        return f"about {lo} seconds" if lo == hi else f"about {lo} to {hi} seconds"
    if hi_seconds < 3 * 3600:
        lo, hi = int(round(lo_seconds / 60.0)), int(round(hi_seconds / 60.0))
        return f"about {lo} minutes" if lo == hi else f"about {lo} to {hi} minutes"
    lo, hi = lo_seconds / 3600.0, hi_seconds / 3600.0
    return f"about {lo:.1f} to {hi:.1f} hours"


def excel_price(size_bytes: Optional[int], filename: str,
                cells: Optional[int]) -> Optional[str]:
    """The sentence to show BEFORE an Excel workbook is parsed, or None.

    None when the parse is a few seconds and the sentence would be friction.
    Worded to stay true after the parse as well, because the upload page
    re-renders while the file sits in the uploader and the notice stays on
    screen above a table that has since loaded.
    """
    if cells is not None:
        if cells < EXCEL_QUIET_BELOW_CELLS:
            return None
        lo, hi = excel_parse_seconds(cells)
        return (
            f"**{filename}** is an Excel workbook, and the sheet being read "
            f"holds {cells:,} cells. Excel is the slowest format this app "
            f"reads — measured at 7 to 10 seconds per million cells — so "
            f"reading this sheet takes {_span(lo, hi)}, during which nothing "
            f"can be said about the file. The same table saved as CSV reads "
            f"about fifty times faster."
        )
    if not size_bytes or size_bytes < EXCEL_QUIET_BELOW_BYTES:
        return None
    return (
        f"**{filename}** is a {size_bytes / (1024 * 1024):,.0f} MB Excel "
        f"workbook whose cells could not be counted without reading it. Excel "
        f"is the slowest format this app reads — measured at 7 to 10 seconds "
        f"per million cells — so expect minutes rather than seconds, during "
        f"which nothing can be said about the file. The same table saved as "
        f"CSV reads about fifty times faster."
    )


def excel_batch_price(entries) -> Optional[str]:
    """One sentence for the 'Add all N files' button, from `(filename, cells,
    size_bytes)` triples for the Excel files in the batch, or None.

    Counted sheets are summed into one span; workbooks that could not be
    counted are named as such, above the byte threshold.
    """
    counted = [(name, cells) for name, cells, _ in entries if cells is not None]
    uncounted = [name for name, cells, size in entries
                 if cells is None and size and size >= EXCEL_QUIET_BELOW_BYTES]
    total = sum(cells for _, cells in counted)
    if total < EXCEL_QUIET_BELOW_CELLS and not uncounted:
        return None
    parts = []
    if total >= EXCEL_QUIET_BELOW_CELLS:
        lo, hi = excel_parse_seconds(total)
        n = len(counted)
        parts.append(
            f"{n} of these {'is an' if n == 1 else 'are'} Excel "
            f"workbook{'' if n == 1 else 's'} with {total:,} cells "
            f"{'on the sheet being read' if n == 1 else 'between them'}: "
            f"reading {'it' if n == 1 else 'them'} takes {_span(lo, hi)} "
            f"(Excel measured 7 to 10 seconds per million cells; the same "
            f"tables as CSV read about fifty times faster)."
        )
    if uncounted:
        parts.append(
            f"{', '.join(uncounted)} {'is a large Excel workbook' if len(uncounted) == 1 else 'are large Excel workbooks'} "
            f"whose cells could not be counted without reading "
            f"{'it' if len(uncounted) == 1 else 'them'}; expect minutes rather "
            f"than seconds."
        )
    return " ".join(parts)
