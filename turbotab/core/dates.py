"""Dates written as text: which formats a column's values fit, and when the text cannot say.

``03/01/2020`` is the first of March to a US reader and the third of January to most others. When
every value in a column has a day of 12 or less, nothing in the file says which. DuckDB's sniffer
then picks day-first without a word; on US visit dates coarsened to the first of the month (a common
de-identification) that changed 1,840 of 2,000 dates, all to January (audit MA-05). So a column's
text is tested here against every format at once, in DuckDB, over every value:

* an **unambiguous** format names the month in words or puts the year first (``Mar 3, 2021``,
  ``14-Mar-2021``, ``2021-03-14``). It has one reading, so it is read.
* a **numeric day-month pair** (``%m/%d/%Y`` and ``%d/%m/%Y``) is read only when one member reads
  values the other cannot: a day above 12 settles it. When both read every value the column is
  **ambiguous**, and it stays text until the user says which (``repairs`` family ``date_reading``).
* values neither reads are **undated**: counted, never placed.

One implementation serves the ingest (``datastore``), the structure reading and the combining order
(``stages.working``), and the date-reading repair (``repairs``), so they cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

# Formats with one reading: the year first, or the month as a word. Order does not matter: no
# value can be read by two of them differently.
_TIMES = ("", " %H:%M", " %H:%M:%S", " %H:%M:%S.%f", "T%H:%M", "T%H:%M:%S", "T%H:%M:%S.%f")
UNAMBIGUOUS: tuple[str, ...] = tuple(
    [f"%Y-%m-%d{t}" for t in _TIMES]
    + [f"%Y/%m/%d{t}" for t in _TIMES if not t.startswith("T")]
    + ["%Y.%m.%d"]
    + [f"{d}{t}" for d in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y", "%d %b %Y",
                           "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y", "%d %b, %Y",
                           "%d/%b/%Y", "%b-%d-%Y", "%Y-%b-%d", "%d.%b.%Y", "%b. %d, %Y")
       for t in ("", " %H:%M", " %H:%M:%S")]
)

MONTH_FIRST, DAY_FIRST = "month_first", "day_first"


def _pairs() -> list[tuple[str, str]]:
    out = []
    for sep in ("/", "-", "."):
        for year in ("%Y", "%y"):
            for t in ("", " %H:%M", " %H:%M:%S"):
                out.append((f"%m{sep}%d{sep}{year}{t}", f"%d{sep}%m{sep}{year}{t}"))
    return out


# (month-first, day-first): the same text, two dates.
PAIRS: tuple[tuple[str, str], ...] = tuple(_pairs())
_SWAP = {m: d for m, d in PAIRS} | {d: m for m, d in PAIRS}

READS_AS_DATES = 0.9   # share of a column's values a format must read for the column to be dates
EXAMPLES = 3


def swapped(fmt: str | None) -> str | None:
    """The other member of a day-month pair (``%d/%m/%Y`` ↔ ``%m/%d/%Y``); None for any other."""
    return _SWAP.get(str(fmt)) if fmt else None


def order_of(fmt: str) -> str | None:
    """``month_first`` or ``day_first`` for a member of a pair, else None."""
    for m, d in PAIRS:
        if fmt == m:
            return MONTH_FIRST
        if fmt == d:
            return DAY_FIRST
    return None


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def format_list(formats: Sequence[str]) -> str:
    """A DuckDB list literal of strptime formats."""
    return "[" + ", ".join(_lit(f) for f in formats) + "]"


def parse_sql(x: str, formats: Sequence[str]) -> str:
    """SQL that reads text ``x`` with the first of ``formats`` that fits; NULL when none does."""
    if not formats:
        return "CAST(NULL AS TIMESTAMP)"
    return f"try_strptime(CAST({x} AS VARCHAR), {format_list(formats)})"


def has_time(fmt: str) -> bool:
    return "%H" in fmt


@dataclass
class DateReading:
    """How a text column reads as dates, over every value."""

    column: str
    n: int                                   # non-missing values
    kind: str                                # dates | ambiguous | mixed | none
    formats: list[str] = field(default_factory=list)  # what reads it (empty unless dates)
    parsed: int = 0                          # values those formats read
    pair: tuple[str, str] | None = None      # (month-first, day-first) when the pair decides
    examples: list[dict[str, Any]] = field(default_factory=list)  # ambiguous: both readings

    @property
    def share(self) -> float:
        return self.parsed / self.n if self.n else 0.0

    @property
    def undated(self) -> int:
        return self.n - self.parsed

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "n": self.n, "kind": self.kind, "formats": list(self.formats),
                "parsed": self.parsed, "pair": list(self.pair) if self.pair else None,
                "examples": list(self.examples)}


def read_text_dates(con: Any, rel: str, column: str, declared: str | None = None) -> DateReading:
    """Read text column ``column`` of relation ``rel`` as dates, every value at once.

    ``declared`` is the user's answer for an ambiguous column (one member of a pair): it joins the
    unambiguous formats, and the column reads as dates by it.
    """
    x = f'CAST("{column.replace(chr(34), chr(34) * 2)}" AS VARCHAR)'
    aggs = [f"count({x})", f"count(try_strptime({x}, {format_list(UNAMBIGUOUS)}))"]
    for m, d in PAIRS:
        a, b = f"try_strptime({x}, {_lit(m)})", f"try_strptime({x}, {_lit(d)})"
        aggs += [f"count({a})", f"count({b})", f"count_if({a} IS NOT NULL AND {b} IS NOT NULL)"]
    if declared:
        aggs.append(f"count(try_strptime({x}, {format_list([*UNAMBIGUOUS, declared])}))")
    row = con.execute(f"SELECT {', '.join(aggs)} FROM {rel}").fetchone()
    n, u = int(row[0] or 0), int(row[1] or 0)
    if n == 0:
        return DateReading(column, 0, "none")
    if declared:
        parsed = int(row[-1] or 0)
        kind = "dates" if parsed else "none"
        return DateReading(column, n, kind, [*UNAMBIGUOUS, declared] if parsed else [], parsed,
                           pair=(declared, swapped(declared)) if swapped(declared) else None)
    best: tuple[int, int, int, tuple[str, str]] | None = None
    for i, pair in enumerate(PAIRS):
        m, d, both = (int(v or 0) for v in row[2 + 3 * i: 5 + 3 * i])
        if best is None or max(m, d) > max(best[0], best[1]):
            best = (m, d, both, pair)
    assert best is not None
    m, d, both, pair = best
    if u >= max(m, d):
        return DateReading(column, n, "dates" if u else "none", list(UNAMBIGUOUS) if u else [], u)
    if m == d == both:  # every value one member reads, the other reads too
        examples = _examples(con, rel, x, pair)
        return DateReading(column, n, "ambiguous", [], 0, pair=pair, examples=examples)
    if min(m, d) > both:  # values only month-first reads AND values only day-first reads
        return DateReading(column, n, "mixed", [], 0, pair=pair)
    member = pair[0] if m > d else pair[1]
    formats = [*UNAMBIGUOUS, member]
    parsed = int(con.execute(f"SELECT count(try_strptime({x}, {format_list(formats)})) "
                             f"FROM {rel}").fetchone()[0] or 0)
    return DateReading(column, n, "dates", formats, parsed, pair=pair)


def _examples(con: Any, rel: str, x: str, pair: tuple[str, str]) -> list[dict[str, Any]]:
    """Up to three values (first seen first) whose two readings differ, with both readings."""
    m, d = pair
    rows = con.execute(
        f"SELECT v, any_value(a), any_value(b) FROM (SELECT {x} AS v, try_strptime({x}, {_lit(m)}) AS a, "
        f"try_strptime({x}, {_lit(d)}) AS b, row_number() OVER () AS r FROM {rel}) "
        f"WHERE a IS NOT NULL AND b IS NOT NULL AND a <> b GROUP BY v ORDER BY min(r) "
        f"LIMIT {EXAMPLES}").fetchall()

    def iso(t: Any) -> str:
        return t.isoformat(sep=" ") if has_time(m) else t.date().isoformat()

    return [{"text": str(v), MONTH_FIRST: iso(a), DAY_FIRST: iso(b)} for v, a, b in rows]


def read_series(values: Any, declared: str | None = None, column: str = "value") -> DateReading:
    """:func:`read_text_dates` over a pandas Series or a list of strings."""
    import duckdb
    import pandas as pd
    import pyarrow as pa

    s = pd.Series(values, dtype="object")
    text = [None if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in s.tolist()]
    con = duckdb.connect()
    try:
        con.register("__dates", pa.table({column: pa.array(text, pa.string())}))
        return read_text_dates(con, "__dates", column, declared)
    finally:
        con.close()


def to_timestamps(values: Any, formats: Sequence[str]) -> Any:
    """``values`` read by ``formats`` as a pandas datetime64 Series (NaT where none fits)."""
    import duckdb
    import pandas as pd
    import pyarrow as pa

    s = pd.Series(values, dtype="object")
    text = [None if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in s.tolist()]
    con = duckdb.connect()
    try:
        con.register("__dates", pa.table({"v": pa.array(text, pa.string())}))
        got = con.execute(f"SELECT {parse_sql('v', formats)} AS t FROM __dates").to_arrow_table()
    finally:
        con.close()
    # Through Arrow, not numpy: a masked numpy array would turn a missing value into 1970-01-01.
    column = got.column("t").to_pandas() if got.num_rows else pd.Series([], dtype="datetime64[ns]")
    return pd.Series(pd.to_datetime(column).to_numpy(), index=s.index)


__all__ = ["DAY_FIRST", "DateReading", "MONTH_FIRST", "PAIRS", "READS_AS_DATES", "UNAMBIGUOUS",
           "format_list", "has_time", "order_of", "parse_sql", "read_series", "read_text_dates",
           "swapped", "to_timestamps"]
