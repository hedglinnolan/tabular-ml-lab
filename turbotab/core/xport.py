"""SAS transport (XPORT) files, versions 5 and 8 (V2 definition of done §1: SAS XPT upload).

NHANES and most US federal surveys publish their tables as SAS transport files. The format is
public: SAS Technical Support document TS-140, *The Record Layout of a Data Set in SAS Transport
(XPORT) Format* (version 5), and *Record Layout of a SAS Version 8 or 9 Data Set in SAS Transport
Format* (version 8). Everything below follows those two documents:

* every record is 80 bytes; a header record reads ``HEADER RECORD*******<name> HEADER
  RECORD!!!!!!!`` followed by six five-digit fields;
* version 5 opens with ``LIBRARY``, version 8 with ``LIBV8``; a member (one data set) is announced
  by ``MEMBER``/``MEMBV8``, described by ``DSCRPTR``/``DSCPTV8``, and lists its variables in
  ``NAMESTR``/``NAMSTV8`` records of 140 bytes (136 on VAX/VMS) each, packed and padded to 80;
* version 8 adds a 32-character variable name inside the namestr, and labels longer than 40
  characters in ``LABELV8``/``LABELV9`` records;
* the observations follow ``OBS``/``OBSV8``, packed, each ``sum(lengths)`` bytes; the last record is
  padded with blanks;
* a number is an IBM System/360 hexadecimal floating-point number, big-endian, 2–8 bytes (a short
  length keeps the leading bytes); a missing value is ``.``, ``.A``–``.Z`` or ``._`` in the first
  byte with every other byte zero; a character value is padded with blanks.

The conversion to IEEE doubles is exact for every value SAS wrote from a double (the IBM fraction
has 53 to 56 significant bits) and rounds to nearest otherwise, as R's ``foreign::read.xport``
does. Numbers with a SAS date or datetime format are SAS dates (days since 1960-01-01) and
datetimes (seconds since 1960-01-01 00:00:00) and are read as such; a time format's seconds stay
numbers, since a SAS time may pass 24 hours. Only the first member of a library is read; a note
says when there are more.

The reader is streamed: the header is parsed once, and the observations are converted a block of
rows at a time into Arrow record batches, so a file larger than memory is read in bounded memory.
"""
from __future__ import annotations

import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

RECORD = 80
_PREFIX = b"HEADER RECORD*******"
_MIDDLE = b"HEADER RECORD!!!!!!!"
MISSING_FIRST_BYTES = frozenset([0x2E, 0x5F, *range(0x41, 0x5B)])  # ".", "_", "A"–"Z"
SAS_EPOCH = np.datetime64("1960-01-01", "D")
SAS_EPOCH_S = np.datetime64("1960-01-01T00:00:00", "s")
CHUNK_BYTES = 64 << 20  # observation bytes converted at once

# SAS's date, datetime and time formats (SAS 9.4 Formats and Informats: Reference, "Formats by
# Category"); a numeric variable carrying one of them holds a date, a datetime or a time.
DATE_FORMATS = frozenset({
    "DATE", "DAY", "DDMMYY", "DDMMYYB", "DDMMYYC", "DDMMYYD", "DDMMYYN", "DDMMYYP", "DDMMYYS",
    "DOWNAME", "E8601DA", "B8601DA", "JULDAY", "JULIAN", "MMDDYY", "MMDDYYB", "MMDDYYC",
    "MMDDYYD", "MMDDYYN", "MMDDYYP", "MMDDYYS", "MMYY", "MMYYC", "MMYYD", "MMYYN", "MMYYP",
    "MMYYS", "MONNAME", "MONTH", "MONYY", "NENGO", "QTR", "QTRR", "WEEKDATE", "WEEKDATX",
    "WEEKDAY", "WEEKU", "WEEKV", "WEEKW", "WORDDATE", "WORDDATX", "YEAR", "YYMM", "YYMMC",
    "YYMMD", "YYMMN", "YYMMP", "YYMMS", "YYMMDD", "YYMMDDB", "YYMMDDC", "YYMMDDD", "YYMMDDN",
    "YYMMDDP", "YYMMDDS", "YYMON", "YYQ", "YYQC", "YYQD", "YYQN", "YYQP", "YYQS", "YYQR",
    "YYQRC", "YYQRD", "YYQRN", "YYQRP", "YYQRS", "MINGUO", "PDJULG", "PDJULI",
})
DATETIME_FORMATS = frozenset({"DATETIME", "DATEAMPM", "DTDATE", "E8601DT", "B8601DT", "E8601DZ",
                              "B8601DZ", "MDYAMPM", "DTMONYY", "DTWKDATX", "DTYEAR", "DTYYQC"})
TIME_FORMATS = frozenset({"TIME", "TIMEAMPM", "TOD", "HHMM", "HOUR", "MMSS", "E8601TM",
                          "B8601TM", "E8601LZ", "B8601LZ"})


class XportError(ValueError):
    """The file is not a SAS transport file TurboTab can read, with the reason in plain words."""


@dataclass
class Variable:
    """One variable as its namestr (and, in version 8, its long label record) describes it."""

    number: int
    name: str
    label: str
    type: str          # "numeric" | "text"
    length: int        # bytes in each observation
    position: int      # offset in each observation
    format: str        # the format's name, without width or decimals ("DATE", "" when none)
    format_width: int = 0
    format_decimals: int = 0
    informat: str = ""

    @property
    def kind(self) -> str:
        """``number``, ``date``, ``datetime``, ``time`` or ``text``: what the values are read as."""
        if self.type == "text":
            return "text"
        name = self.format.upper().rstrip(".")
        name = re.sub(r"\d+$", "", name)
        if name in DATE_FORMATS:
            return "date"
        if name in DATETIME_FORMATS:
            return "datetime"
        if name in TIME_FORMATS:
            return "time"
        return "number"

    def to_dict(self) -> dict[str, Any]:
        return {"variable": self.name, "label": self.label, "type": self.type,
                "length": self.length, "format": self.format, "read_as": self.kind}


@dataclass
class Member:
    """The data set the reader reads: its name, label, variables and where its observations lie."""

    name: str
    label: str
    type: str
    sas_version: str
    os: str
    created: str
    variables: list[Variable]
    data_start: int
    data_end: int
    obs_length: int
    n_rows: int


@dataclass
class XportFile:
    """A parsed SAS transport file: its version, its first member, and notes for the reader."""

    path: Path
    version: int
    member: Member
    n_members: int
    notes: list[str] = field(default_factory=list)
    # Text variables whose bytes were not UTF-8, and the encoding they were read in.
    encodings: dict[str, str] = field(default_factory=dict)
    # Numeric variables holding SAS special missing values (.A–.Z, ._), and how many: counted
    # while the observations are converted (:func:`batches`).
    special: dict[str, int] = field(default_factory=dict)


def _header(rec: bytes, *names: str) -> str | None:
    """The header record's name when ``rec`` is one of ``names`` (any when none given), else None."""
    if len(rec) < RECORD or not rec.startswith(_PREFIX) or rec[28:48] != _MIDDLE:
        return None
    name = rec[20:28].decode("ascii", "replace").strip()
    if names and name not in names:
        return None
    return name


def _fields(rec: bytes) -> list[int]:
    """The numbers a header record carries after its name: six five-digit fields
    (``000000000500000…``: the NAMESTR count is the second), or blank-padded numbers (the
    ``LABELV8`` and ``OBSV8`` records write ``              1              0``)."""
    tail = rec[48:78]
    if tail.isdigit():
        return [int(tail[5 * i: 5 * i + 5]) for i in range(6)]
    out = [int(t) for t in tail.split() if t.isdigit()]
    return out + [0] * (6 - len(out))


def _text(raw: bytes) -> str:
    """A header's or a namestr's text field: trailing blanks and NULs gone, decoded."""
    raw = raw.rstrip(b" \x00")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _u16(buf: bytes, at: int) -> int:
    return int.from_bytes(buf[at:at + 2], "big", signed=True)


def _u32(buf: bytes, at: int) -> int:
    return int.from_bytes(buf[at:at + 4], "big", signed=True)


def read_header(path: str | Path) -> XportFile:
    """Parse ``path``'s library and first member headers (no observation is read)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(RECORD)
        if head.startswith(b"**COMPRESSED**"):
            raise XportError(f"{path.name!r} is a SAS CPORT file (made by PROC CPORT), not a transport "
                             "file; export the data set with the XPORT engine (PROC COPY or "
                             "%loc2xpt) or as CSV")
        library = _header(head)
        if library not in ("LIBRARY", "LIBV8"):
            raise XportError(f"{path.name!r} is not a SAS transport (XPORT) file: it does not open "
                             "with a library header record")
        version = 5 if library == "LIBRARY" else 8
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return _parse(path, mm, size, version)


def _parse(path: Path, mm: Any, size: int, version: int) -> XportFile:
    notes: list[str] = []
    at = 3 * RECORD  # the library header and its two real-header records
    member_name, descriptor, namestr_name, obs_name = (
        ("MEMBER", "DSCRPTR", "NAMESTR", "OBS") if version == 5 else
        ("MEMBV8", "DSCPTV8", "NAMSTV8", "OBSV8"))
    rec = mm[at:at + RECORD]
    if _header(rec, member_name) is None:
        raise XportError(f"{path.name!r} has no data set in it (no {member_name} header record)")
    namestr_length = int(rec[74:78]) if rec[74:78].strip().isdigit() else 140
    if namestr_length not in (136, 140):
        raise XportError(f"{path.name!r} declares {namestr_length}-byte variable descriptors; "
                         "the format has 140 (136 on VAX/VMS)")
    at += RECORD
    if _header(mm[at:at + RECORD], descriptor) is None:
        raise XportError(f"{path.name!r} is damaged: the member header has no descriptor record")
    at += RECORD
    first, second = mm[at:at + RECORD], mm[at + RECORD:at + 2 * RECORD]
    at += 2 * RECORD
    if version == 5:
        ds_name, sas_version, os_name, created = (_text(first[8:16]), _text(first[24:32]),
                                                  _text(first[32:40]), _text(first[64:80]))
    else:
        ds_name, sas_version, os_name, created = (_text(first[8:40]), _text(first[48:56]),
                                                  _text(first[56:64]), _text(first[64:80]))
    ds_label, ds_type = _text(second[32:72]), _text(second[72:80])
    rec = mm[at:at + RECORD]
    if _header(rec, namestr_name) is None:
        raise XportError(f"{path.name!r} is damaged: no {namestr_name} record lists its variables")
    n_vars = _fields(rec)[1]
    at += RECORD
    block = namestr_length * n_vars
    raw = mm[at:at + block]
    if len(raw) < block:
        raise XportError(f"{path.name!r} ends inside its variable descriptors")
    at += -(-block // RECORD) * RECORD
    variables = [_namestr(raw[i * namestr_length:(i + 1) * namestr_length], version)
                 for i in range(n_vars)]
    # Version 8: labels longer than 40 characters (and, in LABELV9, long format names).
    while True:
        rec = mm[at:at + RECORD]
        name = _header(rec)
        if name in ("LABELV8", "LABELV9"):
            at = _long_labels(mm, at + RECORD, _fields(rec)[0], variables, name == "LABELV9")
            continue
        break
    if _header(rec, obs_name) is None:
        raise XportError(f"{path.name!r} is damaged: no {obs_name} record before its observations")
    at += RECORD
    data_start = at
    # The data run to the next member's header (on an 80-byte boundary), else to the end.
    data_end, n_members = size, 1
    probe = data_start
    marker = _PREFIX + (b"MEMBER  " if version == 5 else b"MEMBV8  ") + _MIDDLE
    while True:
        found = mm.find(marker, probe)
        if found < 0:
            break
        if (found - data_start) % RECORD == 0:
            if n_members == 1:
                data_end = found
            n_members += 1
        probe = found + 1
    if n_members > 1:
        notes.append(f"the transport file holds {n_members} data sets; only the first, "
                     f"\"{ds_name}\", was read")
    obs_length = max((v.position + v.length for v in variables), default=0)
    n_rows = _count_rows(mm, data_start, data_end, obs_length)
    member = Member(name=ds_name, label=ds_label, type=ds_type, sas_version=sas_version,
                    os=os_name, created=created, variables=variables, data_start=data_start,
                    data_end=data_end, obs_length=obs_length, n_rows=n_rows)
    return XportFile(path=path, version=version, member=member, n_members=n_members, notes=notes)


def _namestr(raw: bytes, version: int) -> Variable:
    ntype, nlng, nvar0 = _u16(raw, 0), _u16(raw, 4), _u16(raw, 6)
    short_name = _text(raw[8:16])
    label = _text(raw[16:56])
    form = _text(raw[56:64])
    nfl, nfd = _u16(raw, 64), _u16(raw, 66)
    informat = _text(raw[72:80])
    npos = _u32(raw, 84)
    name = short_name
    if version == 8 and len(raw) >= 120:
        long_name = _text(raw[88:120])
        name = long_name or short_name
    if ntype not in (1, 2):
        raise XportError(f"variable {name!r} has type {ntype}; the format has 1 (numeric) and 2 "
                         "(character)")
    if ntype == 1 and not 2 <= nlng <= 8:
        raise XportError(f"numeric variable {name!r} is {nlng} bytes long; the format stores 2 to 8")
    return Variable(number=nvar0, name=name, label=label, type="numeric" if ntype == 1 else "text",
                    length=nlng, position=npos, format=form, format_width=nfl,
                    format_decimals=nfd, informat=informat)


def _long_labels(mm: Any, at: int, count: int, variables: list[Variable], v9: bool) -> int:
    """Read a LABELV8/LABELV9 block from ``at``; returns the offset after its padding."""
    by_number = {v.number: v for v in variables}
    start = at
    for _ in range(count):
        head = mm[at:at + (10 if v9 else 6)]
        number, name_len, label_len = _u16(head, 0), _u16(head, 2), _u16(head, 4)
        format_len = informat_len = 0
        if v9:
            format_len, informat_len = _u16(head, 6), _u16(head, 8)
        at += len(head)
        name = _text(mm[at:at + name_len])
        at += name_len
        label = _text(mm[at:at + label_len])
        at += label_len
        fmt = _text(mm[at:at + format_len])
        at += format_len + informat_len
        target = by_number.get(number) or next((v for v in variables if v.name == name), None)
        if target is not None:
            target.label = label
            if fmt:
                target.format = re.sub(r"[\d.]+$", "", fmt) or fmt
    used = at - start
    return start + -(-used // RECORD) * RECORD


def _count_rows(mm: Any, start: int, end: int, obs_length: int) -> int:
    """Observations between ``start`` and ``end``. The last record is padded with blanks: whole
    observations of blanks inside that padding are not data (an all-blank observation can only be
    one whose every variable is character, the format's own ambiguity)."""
    if obs_length <= 0 or end <= start:
        return 0
    n = (end - start) // obs_length
    # The padding is shorter than one record, so a padding "observation" starts inside the last
    # 80 bytes; only an observation shorter than a record can fit there.
    while n > 0:
        first = start + (n - 1) * obs_length
        if first > end - RECORD and mm[first:first + obs_length] == b" " * obs_length:
            n -= 1
            continue
        break
    return n


# ── values ───────────────────────────────────────────────────────────────────


def ibm_to_double(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IBM hexadecimal floats (``raw``: n × length uint8, length 2–8, big-endian) to IEEE doubles,
    and the mask of SAS missing values (``.``, ``.A``–``.Z``, ``._``: that first byte, every other
    byte zero), which are NaN in the doubles."""
    n, length = raw.shape
    full = np.zeros((n, 8), dtype=np.uint8)
    full[:, :length] = raw
    rest_zero = ~full[:, 1:].any(axis=1)
    missing = rest_zero & np.isin(full[:, 0], np.fromiter(MISSING_FIRST_BYTES, dtype=np.uint8))
    word = full.view(">u8").reshape(n).astype(np.uint64)
    negative = (word >> np.uint64(63)) != 0
    exponent = ((word >> np.uint64(56)) & np.uint64(0x7F)).astype(np.int64)
    fraction = word & np.uint64(0x00FFFFFFFFFFFFFF)
    # value = fraction / 2**56 × 16**(exponent − 64); the fraction's conversion rounds to nearest.
    values = np.ldexp(fraction.astype(np.float64), (4 * (exponent - 64) - 56).astype(np.int32))
    values = np.where(negative, -values, values)
    values[missing] = np.nan
    return values, missing


def _decode_text(raw: np.ndarray) -> tuple[list[str | None], str]:
    """Character values: trailing blanks (and NULs) dropped; blank is SAS's missing character
    value, read as missing. UTF-8 when every value decodes so, else Windows-1252 (SAS's WLATIN1),
    else Latin-1. Returns the values and the encoding they were read in."""
    items = [bytes(r).rstrip(b" \x00") for r in raw]
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return [s.decode(encoding) if s else None for s in items], encoding
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes every byte")  # pragma: no cover


def batches(xpt: XportFile, rows_per_batch: int | None = None) -> Iterator[Any]:
    """The member's observations as Arrow record batches, in file order. Numbers are doubles (a
    SAS date or datetime as such), text is UTF-8 strings, SAS missing values are nulls; the text
    encodings and special missing values met are noted on ``xpt``."""
    import pyarrow as pa

    m = xpt.member
    if rows_per_batch is None:
        rows_per_batch = max(1, CHUNK_BYTES // max(1, m.obs_length))
    names = _arrow_names(m.variables)
    schema = pa.schema([pa.field(n, _arrow_type(v)) for n, v in zip(names, m.variables)])
    with open(xpt.path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for first in range(0, m.n_rows, rows_per_batch):
            n = min(rows_per_batch, m.n_rows - first)
            start = m.data_start + first * m.obs_length
            # A copy of the block (slicing the map), so no view outlives the map.
            block = np.frombuffer(mm[start:start + n * m.obs_length],
                                  dtype=np.uint8).reshape(n, m.obs_length)
            arrays = [_column(block[:, v.position:v.position + v.length], v, xpt)
                      for v in m.variables]
            del block
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)
        if m.n_rows == 0:
            yield pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in schema],
                                             schema=schema)


def _arrow_names(variables: list[Variable]) -> list[str]:
    return [v.name for v in variables]


def _arrow_type(v: Variable) -> Any:
    import pyarrow as pa

    return {"text": pa.string(), "date": pa.date32(),
            "datetime": pa.timestamp("ms")}.get(v.kind, pa.float64())


def _column(raw: np.ndarray, v: Variable, xpt: XportFile | None = None) -> Any:
    """One variable's values in a block of observations, as an Arrow array.

    A SAS date is a whole number of days since 1960-01-01; a fraction is dropped as SAS's date
    formats drop it. A datetime is seconds since 1960-01-01 00:00:00, kept to the millisecond. A
    time keeps its seconds as a number: SAS times may pass 24 hours (a duration), which no time of
    day holds."""
    import pyarrow as pa

    if v.type == "text":
        values, encoding = _decode_text(raw)
        if encoding != "utf-8" and xpt is not None:
            xpt.encodings[v.name] = encoding
        return pa.array(values, type=pa.string())
    cells = np.ascontiguousarray(raw)
    values, missing = ibm_to_double(cells)
    if xpt is not None and missing.any():
        special = int((missing & (cells[:, 0] != 0x2E)).sum())
        if special:
            xpt.special[v.name] = xpt.special.get(v.name, 0) + special
    present = np.isfinite(values)
    kind = v.kind
    if kind == "date":
        days = np.where(present, np.floor(values), 0).astype(np.int64)
        out = (SAS_EPOCH + days).astype("datetime64[D]")
        return pa.array(out, mask=~present, type=pa.date32())
    if kind == "datetime":
        ms = np.where(present, np.round(values * 1000.0), 0).astype(np.int64)
        out = SAS_EPOCH_S.astype("datetime64[ms]") + ms.astype("timedelta64[ms]")
        return pa.array(out, mask=~present, type=pa.timestamp("ms"))
    return pa.array(values, mask=~present, type=pa.float64())


def special_missing(xpt: XportFile) -> dict[str, int]:
    """Per numeric variable, how many values are SAS special missing values (``.A``–``.Z``,
    ``._``), which TurboTab reads as plain missing values. Counted by :func:`batches`; a header
    whose observations were never converted is read for them here."""
    if not xpt.special and xpt.member.n_rows:
        for _ in batches(xpt):
            pass
    return dict(xpt.special)


def read_frame(path: str | Path) -> Any:
    """The whole first member as a pandas frame (for tests and small files)."""
    import pyarrow as pa

    xpt = read_header(path)
    table = pa.Table.from_batches(list(batches(xpt)))
    return table.to_pandas()


__all__ = ["DATE_FORMATS", "DATETIME_FORMATS", "Member", "TIME_FORMATS", "Variable", "XportError",
           "XportFile", "batches", "ibm_to_double", "read_frame", "read_header", "special_missing"]
