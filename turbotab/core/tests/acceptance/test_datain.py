"""DATAIN: SAS transport upload, codebook import and minimal joins (V2 definition of done §1).

Acceptance, item by item (each expected value from a path independent of the code under test):

1. **SAS XPT, versions 5 and 8.** The NHANES 2017–2018 files DEMO_J, DR1TOT_J and BMX_J (byte-
   faithful windows of the CDC files, ``datain_data/build_fixtures.py``) read identically to R's
   ``foreign::read.xport`` and to pandas' ``read_sas``; version 8 files read identically to R's
   ``haven::read_xpt`` (ReadStat, the C library pyreadstat wraps; pyreadstat is not installed
   here). A file built byte by byte in this test from the two SAS layout documents (TS-140 and the
   version 8 layout), its numbers encoded to IBM hexadecimal floats by exact integer arithmetic,
   reads as its known truth. Variable labels are kept as codebook entries (``lookup.xport``).
2. **Codebook import.** A variable/label/unit/codes table (CSV and Excel), the NHANES codebook
   pages, and the XPT labels. Structured fields settle readings as the user's own documentation,
   with the codebook as their evidence; free-text labels only strengthen the guess on the ask
   card; a codebook the values contradict is asked, never applied. On the NHANES export with its
   codebooks, the fit's ask card shrinks by exactly the readings the codebooks settle, and the
   rest still ask.
3. **Joins.** DEMO + DR1TOT + BMX on SEQN reproduce pandas' merge counts exactly; one-to-many
   too; a many-to-many join is refused with its reason (pandas' own ``validate`` agrees it is
   many-to-many).
4. **The methods sentence** names the codebook and the join, asserted verbatim.

R is used only as an independent reference, by subprocess, on files this test writes; the tests
that need it skip when ``Rscript`` is absent.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from html import unescape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import assembly, codebook as cb, readings as R, xport
from turbotab.core import decisions as d
from turbotab.core.datastore import DataStore, JoinInput, ingest, read_labels
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.truths import asked

HERE = Path(__file__).resolve().parent
DATA = HERE / "datain_data"
NHANES = ("DEMO_J", "DR1TOT_J", "BMX_J")
RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (the independent reference) is absent")
PANDAS_ZERO = 2.0 ** -260  # pandas reads an IBM zero as 2**-260: TS-140's bit recipe applied to 0


# ── helpers ──────────────────────────────────────────────────────────────────


def unpack(folder: Path, name: str, kind: str = "XPT") -> Path:
    path = folder / f"{name}.{kind}"
    if not path.exists():
        path.write_bytes(gzip.decompress((DATA / f"{name}.{kind}.gz").read_bytes()))
    return path


def r(code: str) -> None:
    done = subprocess.run([RSCRIPT, "-e", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def r_csv(path: Path) -> pd.DataFrame:
    """A CSV R wrote with every number as ``%.17g`` (exact round trip), read exactly."""
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False,
                       na_values=["NA"], dtype=str)


def ours(src: Path, folder: Path) -> pd.DataFrame:
    out = folder / f"{src.stem}.parquet"
    ingest(src, out)
    return pd.read_parquet(out).drop(columns="__row_id")


def same_numbers(a: Any, b: Any) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def as_float(column: pd.Series) -> np.ndarray:
    return np.array([float(v) if isinstance(v, str) and v not in ("", "NA") else np.nan
                     for v in column], dtype=float)


R_EXACT = ('x[] <- lapply(x, function(c) if (is.numeric(c) && !inherits(c, "Date") && '
           '!inherits(c, "POSIXt")) sprintf("%.17g", c) else as.character(c)); '
           'write.csv(x, "{out}", row.names = FALSE, na = "NA")')


# ═════════════════════════════════════════════════════════════════════════════
# 1 · SAS XPT, versions 5 and 8
# ═════════════════════════════════════════════════════════════════════════════


@needs_r
@pytest.mark.parametrize("name", NHANES)
def test_1_nhanes_v5_values_are_identical_to_r_foreign(tmp_path, name):
    """Every value, every column and the row count of the NHANES files as ``foreign::read.xport``
    reads them (R prints each number to 17 significant digits; read back exactly)."""
    src = unpack(tmp_path, name)
    out = tmp_path / "r.csv"
    r(f'library(foreign); x <- read.xport("{src}"); ' + R_EXACT.format(out=out))
    ref, got = r_csv(out), ours(src, tmp_path)
    assert list(got.columns) == list(ref.columns)
    assert len(got) == len(ref) > 0
    for c in ref.columns:
        assert same_numbers(got[c], as_float(ref[c])), c


@needs_r
@pytest.mark.parametrize("name", NHANES)
def test_1_nhanes_variable_labels_are_kept_as_codebook_entries(tmp_path, name):
    """The labels sidecar holds every variable's label as ``foreign::lookup.xport`` reports it,
    and the codebook read from it carries them (labels only: it settles nothing)."""
    src = unpack(tmp_path, name)
    out = tmp_path / "labels.csv"
    r(f'library(foreign); i <- lookup.xport("{src}")[[1]]; '
      f'write.csv(data.frame(name = i$name, label = i$label), "{out}", row.names = FALSE)')
    ref = pd.read_csv(out, keep_default_na=False)
    ingest(src, tmp_path / "raw.parquet")
    side = read_labels(tmp_path / "raw.parquet")
    assert side is not None and side["sources"][0]["file"] == src.name
    entries = side["sources"][0]["entries"]
    assert [e["variable"] for e in entries] == list(ref["name"])
    assert [e["label"] for e in entries] == list(ref["label"])
    book = cb.from_labels(side["sources"], src.name)
    assert book.form == "xpt" and {e.variable: e.label for e in book.entries} == \
        dict(zip(ref["name"], ref["label"]))
    assert not any(e.unit or e.codes or e.type for e in book.entries)


@pytest.mark.parametrize("name", NHANES)
def test_1_nhanes_v5_values_match_pandas_read_sas_but_its_zero_defect(tmp_path, name):
    """pandas' own reader (no R needed): identical, except that pandas reads an IBM zero (eight
    zero bytes) as 2**-260; each of those cells reads 0 here, and its eight bytes are zero (read
    straight from the file at the variable's position)."""
    src = unpack(tmp_path, name)
    ref = pd.read_sas(src, format="xport")
    got = ours(src, tmp_path)
    assert list(got.columns) == list(ref.columns) and len(got) == len(ref)
    raw = src.read_bytes()
    head = xport.read_header(src).member
    positions = {v.name: (v.position, v.length) for v in head.variables}
    for c in ref.columns:
        a, b = got[c].to_numpy(float), ref[c].to_numpy(float)
        differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        for i in np.flatnonzero(differ):
            assert a[i] == 0.0 and b[i] == PANDAS_ZERO, (c, i, a[i], b[i])
            at = head.data_start + i * head.obs_length + positions[c][0]
            assert raw[at:at + positions[c][1]] == b"\0" * positions[c][1]


@needs_r
def test_1_v8_files_read_as_haven_reads_them(tmp_path):
    """A version 8 file from ``haven::write_xpt`` (a 26-character name, a label past 40
    characters in a LABELV8 record, UTF-8 text, a SAS special missing ``.A``, dates and
    datetimes) reads as ``haven::read_xpt`` reads it; its version 5 twin as ``foreign`` reads it.
    Blank text is SAS's missing character value (haven shows it as "")."""
    v8, v5 = tmp_path / "edge_v8.xpt", tmp_path / "edge_v5.xpt"
    r(f'''library(haven)
x <- data.frame(respondent_sequence_number = c(93703, 93704, 93705, 93706, 93707),
  intake_g = c(12.5, tagged_na("A"), NA, -0.000123, 1234567.875),
  note = c("plain", "naïve café", "", "x", "  lead"),
  visit = as.Date(c("2017-01-15", NA, "1959-12-31", "2020-02-29", "1960-01-01")),
  at = as.POSIXct(c("2017-01-15 08:30:00", NA, "1960-01-01 00:00:01", "1999-12-31 23:59:59",
                    "2020-02-29 12:00:00"), tz = "UTC"), stringsAsFactors = FALSE)
attr(x$respondent_sequence_number, "label") <- "Respondent sequence number, a label longer than forty characters"
attr(x$intake_g, "label") <- "Intake (g)"
write_xpt(x, "{v8}", version = 8, name = "EDGECASES")
names(x) <- c("SEQN", "INTAKE", "NOTE", "VISIT", "AT")
write_xpt(x, "{v5}", version = 5, name = "EDGE")''')
    out8, out5, lab = tmp_path / "h8.csv", tmp_path / "f5.csv", tmp_path / "lab.csv"
    r(f'library(haven); x <- read_xpt("{v8}"); '
      f'write.csv(data.frame(name = names(x), label = sapply(x, function(c) {{ l <- attr(c, "label"); '
      f'if (is.null(l)) "" else l }})), "{lab}", row.names = FALSE); '
      + R_EXACT.format(out=out8))
    r(f'library(foreign); x <- read.xport("{v5}"); ' + R_EXACT.format(out=out5))
    got8 = ours(v8, tmp_path)
    ref8 = r_csv(out8)
    assert list(got8.columns) == list(ref8.columns) == ["respondent_sequence_number", "intake_g",
                                                         "note", "visit", "at"]
    for c in ("respondent_sequence_number", "intake_g"):
        assert same_numbers(got8[c], as_float(ref8[c])), c
    assert [v if v is not None else "" for v in got8["note"]] == list(ref8["note"].fillna(""))
    assert [str(v) if v is not None else "NA" for v in got8["visit"]] == \
        list(ref8["visit"].fillna("NA"))
    stamps = [v.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(v) else "NA" for v in got8["at"]]
    assert stamps == list(ref8["at"].fillna("NA"))
    labels = pd.read_csv(lab, keep_default_na=False)
    ingest(v8, tmp_path / "raw8.parquet")
    entries = read_labels(tmp_path / "raw8.parquet")["sources"][0]["entries"]
    assert [e["label"] for e in entries] == [s.rstrip() for s in labels["label"]]
    # The special missing value is read as missing, and the ingest says so.
    info = json.loads((tmp_path / "raw8.info.json").read_text())["info"]
    assert any("SAS special missing" in w and "intake_g" in w for w in info["warnings"])
    # Version 5, against foreign: numbers identical (dates are their SAS day numbers there).
    got5, ref5 = ours(v5, tmp_path), r_csv(out5)
    for c in ("SEQN", "INTAKE"):
        assert same_numbers(got5[c], as_float(ref5[c])), c
    sas_days = (pd.to_datetime(got5["VISIT"]) - pd.Timestamp("1960-01-01")).dt.days
    assert same_numbers(sas_days, as_float(ref5["VISIT"]))


# ── a transport file built byte by byte from the layout documents ────────────


def ibm(x: float) -> bytes:
    """IEEE double → 8-byte IBM hexadecimal float, by exact integer arithmetic (TS-140: sign bit,
    7-bit excess-64 exponent of 16, 56-bit fraction)."""
    if x == 0:
        return b"\0" * 8
    m, e = math.frexp(abs(x))
    mant = int(m * 2 ** 53)                 # abs(x) = mant · 2**(e − 53), exactly
    expo = -(-e // 4) + 64                  # abs(x) = F · 16**(expo − 64) · 2**−56
    shift = e - 53 - 4 * (expo - 64) + 56   # 0–3: F's top hexadecimal digit is not zero
    fraction = mant << shift
    return bytes([(0x80 if x < 0 else 0) | expo]) + fraction.to_bytes(7, "big")


def exact(b: bytes) -> float:
    """An IBM float's value, exactly, then rounded once to a double."""
    b = b.ljust(8, b"\0")
    word = int.from_bytes(b, "big")
    value = Fraction(word & ((1 << 56) - 1), 1 << 56) * Fraction(16) ** (((word >> 56) & 0x7F) - 64)
    return float(-value if word >> 63 else value)


MISSING = {".": b".", ".A": b"A", "._": b"_"}


def header(name: str, fields: str = "0" * 30) -> bytes:
    return (b"HEADER RECORD*******" + name.ljust(8).encode() + b"HEADER RECORD!!!!!!!"
            + fields.encode()).ljust(80)


def namestr(ntype: int, length: int, number: int, name: str, label: str, position: int,
            fmt: str = "", long_name: str = "") -> bytes:
    rest = (long_name.encode().ljust(32) + struct.pack(">h", len(label)) + b"\0" * 18
            if long_name else b"\0" * 52)
    return struct.pack(">hhhh8s40s8shhh2s8shhi52s", ntype, 0, length, number,
                       name.encode().ljust(8), label[:40].encode().ljust(40),
                       fmt.encode().ljust(8), 0, 0, 0, b"\0\0", b"".ljust(8), 0, 0, position,
                       rest)


def pad(data: bytes, fill: bytes = b" ") -> bytes:
    return data + fill * ((-len(data)) % 80)


def build_xport(version: int, variables: list[dict[str, Any]], rows: list[bytes],
                second_member: bool = False) -> bytes:
    v8 = version == 8
    stamp = b"05OCT26:00:00:00"
    out = header("LIBV8" if v8 else "LIBRARY")
    out += b"SAS     SAS     SASLIB  9.4     X64_10PR" + b" " * 24 + stamp
    out += stamp + b" " * 64
    member_count = 2 if second_member else 1
    for m in range(member_count):
        out += header("MEMBV8" if v8 else "MEMBER", "000000000000000001600000000140")
        out += header("DSCPTV8" if v8 else "DSCRPTR")
        ds = f"MEMBER{m}"
        if v8:
            out += b"SAS     " + ds.encode().ljust(32) + b"SASDATA 9.4     X64_10PR" + stamp
        else:
            out += (b"SAS     " + ds.encode().ljust(8) + b"SASDATA 9.4     X64_10PR" + b" " * 24
                    + stamp)
        out += stamp + b" " * 16 + b"a test data set".ljust(40) + b"DATA    "
        out += header("NAMSTV8" if v8 else "NAMESTR", f"000000{len(variables):04d}" + "0" * 20)
        out += pad(b"".join(namestr(**v) for v in variables), b"\0")
        long = [v for v in variables if len(v["label"]) > 40]
        if v8 and long:
            out += header("LABELV8", f"{len(long):>15}{0:>15}")
            block = b""
            for v in long:
                name = (v.get("long_name") or v["name"]).encode()
                block += struct.pack(">hhh", v["number"], len(name), len(v["label"]))
                block += name + v["label"].encode()
            out += pad(block)
        out += header("OBSV8" if v8 else "OBS", f"{len(rows):>15}{0:>15}" if v8 else "0" * 30)
        out += pad(b"".join(rows))
    return out


def test_1_a_file_built_from_the_layout_documents_reads_as_its_known_truth(tmp_path):
    """Versions 5 and 8, built in this test: positive, negative, fractional, tiny and huge numbers;
    the missing values ``.``, ``.A`` and ``._``; numbers stored in 3 and 5 bytes (a short length
    keeps the leading bytes); blank-padded text and a blank (missing) value; in version 8 a
    32-character name and a 100-character label (LABELV8); a second member (only the first is
    read, and a note says so); and, at an observation shorter than one record, the blank padding
    never read as rows."""
    numbers = [0.1, -2.5e10, 0.0, 123456.789, 1e-70, 7.2e75, -1.0, 2.0 ** -40]
    short = [0.1, -3.14159, 1000.0, 0.333333, 5.0, -0.001, 42.0, 1e10]
    missing = [None, ".", None, ".A", None, None, "._", None]
    text = ["alpha", "", "  lead", "z", "naive", "x" * 10, "end", "q"]
    long_label = "A label longer than forty characters, as version 8 keeps it whole in LABELV8"
    for version in (5, 8):
        name0 = "a_thirty_two_character_name_abcd" if version == 8 else "NUMBER"
        variables = [
            dict(ntype=1, length=8, number=1, name="NUMBER", label=long_label if version == 8
                 else "A number", position=0, long_name=name0 if version == 8 else ""),
            dict(ntype=1, length=3, number=2, name="SHORT3", label="three bytes", position=8,
                 long_name="SHORT3" if version == 8 else ""),
            dict(ntype=1, length=5, number=3, name="SHORT5", label="five bytes", position=11,
                 long_name="SHORT5" if version == 8 else ""),
            dict(ntype=2, length=10, number=4, name="TEXT", label="text", position=16,
                 long_name="TEXT" if version == 8 else ""),
        ]
        rows, truth = [], {name0: [], "SHORT3": [], "SHORT5": [], "TEXT": []}
        for i in range(8):
            if missing[i] is None:
                num = ibm(numbers[i])
                truth[name0].append(exact(num))
            else:
                num = MISSING[missing[i]] + b"\0" * 7
                truth[name0].append(np.nan)
            s3, s5 = ibm(short[i])[:3], ibm(short[i])[:5]
            truth["SHORT3"].append(exact(s3))
            truth["SHORT5"].append(exact(s5))
            truth["TEXT"].append(text[i].rstrip() or None)
            rows.append(num + s3 + s5 + text[i].encode().ljust(10))
        assert truth[name0][0] == 0.1 and truth[name0][5] == 7.2e75  # whole doubles round-trip
        path = tmp_path / f"built_v{version}.xpt"
        path.write_bytes(build_xport(version, variables, rows, second_member=True))
        head = xport.read_header(path)
        assert head.version == version and head.member.n_rows == 8 and head.n_members == 2
        assert any("2 data sets" in n for n in head.notes)
        if version == 8:
            assert head.member.variables[0].label == long_label
        got = ours(path, tmp_path)
        assert list(got.columns) == [name0, "SHORT3", "SHORT5", "TEXT"]
        for c in (name0, "SHORT3", "SHORT5"):
            assert same_numbers(got[c], truth[c]), (version, c)
        assert list(got["TEXT"]) == truth["TEXT"]
        assert xport.special_missing(head) == {name0: 2}
    # An observation of 10 bytes: 3 rows, then 50 bytes of padding a reader must not count.
    variables = [dict(ntype=2, length=10, number=1, name="CODE", label="code", position=0)]
    path = tmp_path / "short_obs.xpt"
    path.write_bytes(build_xport(5, variables, [b"ab".ljust(10), b"".ljust(10), b"cd".ljust(10)]))
    assert list(ours(path, tmp_path)["CODE"]) == ["ab", None, "cd"]


def test_1_an_xpt_upload_is_read_through_the_server(tmp_path):
    """The upload endpoint accepts a ``.XPT`` and the ingest reads it (DEMO_J: 300 rows, 46
    columns, its labels kept)."""
    src = unpack(tmp_path, "DEMO_J")
    with local_server(tmp_path / "home") as client:
        with open(src, "rb") as fh:
            resp = client.post("/api/projects/upload",
                               files={"file": ("DEMO_J.XPT", fh, "application/octet-stream")})
        assert resp.status_code == 200, resp.text[:400]
        from turbotab.core.tests.acceptance.server_drive import Drive

        drive = Drive(client, resp.json()["id"])
        info = drive.artifact("ingest")
        ref = pd.read_sas(src, format="xport")
        assert (info["n_rows"], info["n_cols"]) == (len(ref), ref.shape[1])
        pdir = client.app.state.service.workspace.project_dir(drive.pid)
        assert read_labels(pdir / "data" / "raw.parquet")["sources"][0]["dataset"] == "DEMO_J"


# ═════════════════════════════════════════════════════════════════════════════
# The NHANES export: DEMO_J + DR1TOT_J + BMX_J joined on SEQN, in a project folder
# ═════════════════════════════════════════════════════════════════════════════


def pandas_frames(folder: Path) -> dict[str, pd.DataFrame]:
    """The three files through pandas' own reader, its zero defect undone (2**-260 → 0)."""
    out = {}
    for name in NHANES:
        frame = pd.read_sas(unpack(folder, name), format="xport")
        out[name] = frame.mask(frame == PANDAS_ZERO, 0.0)
    return out


class Project:
    """A project folder as the server keeps one (``data/raw.parquet`` with its sidecars,
    ``files/<id>/``, ``codebooks/<id>/``) and a decision log, driven through the real validators,
    completions and fold, without the server's job runner."""

    def __init__(self, folder: Path):
        self.dir = folder / "project"
        (self.dir / "data").mkdir(parents=True)
        self.files: dict[str, str] = {}
        self.records: list[d.DecisionRecord] = []
        self.primary = unpack(folder, "DEMO_J")
        for i, name in enumerate(("DR1TOT_J", "BMX_J")):
            fid = f"f{i:010x}"
            (self.dir / "files" / fid).mkdir(parents=True)
            info = ingest(unpack(folder, name), self.dir / "files" / fid / "raw.parquet")
            (self.dir / "files" / fid / "file.json").write_text(json.dumps(
                {"id": fid, "name": f"{name}.XPT", "n_rows": info.n_rows}))
            self.files[name] = fid
        self.read()

    @property
    def state(self) -> d.ProjectState:
        return d.fold(self.records)

    @property
    def table(self) -> Path:
        return self.dir / "data" / "raw.parquet"

    def read(self) -> None:
        """The ingest stage: the table and every join recorded so far."""
        joins = [JoinInput(parquet=self.dir / "files" / s.file / "raw.parquet", name=s.name,
                           on=s.on, right_on=s.right_on, how=s.how)
                 for s in (self.state.joins or {}).values()]
        ingest(self.primary, self.table, joins=joins)

    def ctx(self) -> dict[str, Any]:
        info = json.loads((self.dir / "data" / "raw.info.json").read_text())["info"]
        return {"project_dir": str(self.dir), "state": self.state, "ingest_status": "fresh",
                "columns": [c["name"] for c in info["columns"]],
                "records": lambda: list(self.records)}

    def decide(self, decision: Any) -> d.DecisionRecord:
        done = d.validate(decision, self.ctx())
        from turbotab.core import voice

        record = d.DecisionRecord(id=f"{len(self.records) + 1:032x}", seq=len(self.records) + 1,
                                  at=datetime.now(timezone.utc),
                                  sentence=voice.sentence_for(done, self.state, None),
                                  decision=done)
        self.records.append(record)
        if done.kind == "join_files":
            self.read()
        return record

    def stage_codebook(self, path: Path) -> str:
        return cb.stage(cb.read(path), self.dir, path)


@pytest.fixture(scope="module")
def nhanes(tmp_path_factory):
    folder = tmp_path_factory.mktemp("nhanes")
    project = Project(folder)
    return folder, project


# ═════════════════════════════════════════════════════════════════════════════
# 3 · Joins
# ═════════════════════════════════════════════════════════════════════════════


def pandas_counts(left: pd.DataFrame, right: pd.DataFrame, on: str) -> dict[str, int]:
    """The join's counts from pandas' merge (``indicator``), with blank identifiers dropped first
    (pandas pairs a blank with a blank; a missing identifier is no shared one)."""
    left, right = left[left[on].notna()], right[right[on].notna()]
    outer = left.merge(right, on=on, how="outer", indicator=True)
    sides = outer["_merge"].value_counts()
    return {"rows": len(left.merge(right, on=on, how="left")),
            "table_unmatched": int(sides.get("left_only", 0)),
            "file_unmatched": int(sides.get("right_only", 0)),
            "matched_keys": int(left[on][left[on].isin(right[on])].nunique()),
            "table_rows": len(left), "file_rows": len(right)}


def test_3_demo_dr1tot_bmx_on_seqn_reproduce_pandas_merge_counts(nhanes):
    """The preview's counts before each join, the counts recorded with it, and the joined table
    after both, against pandas' merge of pandas' own reading of the three files."""
    folder, project = nhanes
    frames = pandas_frames(folder)
    left = frames["DEMO_J"]
    for name in ("DR1TOT_J", "BMX_J"):
        right = frames[name]
        want = pandas_counts(left, right, "SEQN")
        # pandas' own validation: one row per SEQN on each side.
        left.merge(right, on="SEQN", validate="one_to_one")
        fid = project.files[name]
        plan = assembly.preview(project.table, project.dir / "files" / fid / "raw.parquet",
                                on="SEQN", right_name=f"{name}.XPT")
        assert plan.refusal is None and plan.relation == "one-to-one"
        got = plan.counts()
        assert {k: got[k] for k in want} == want, name
        record = project.decide(d.JoinFiles(file=fid, on="SEQN"))
        assert record.decision.counts.model_dump() == got  # the preview is what is recorded
        left = left.merge(right, on="SEQN", how="left")
    store = DataStore(project.table, 1 << 30)
    try:
        joined = store.materialize()
    finally:
        store.close()
    assert len(joined) == len(left) and list(joined.columns) == list(left.columns)
    for c in ("SEQN", "RIDAGEYR", "DR1TKCAL", "DR1TFIBE", "BMXWT", "BMXWAIST"):
        assert same_numbers(joined[c], left[c]), c


def recalls(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A long table: 1–3 rows per SEQN (a person's recalls or visits), for some SEQNs only."""
    rng = np.random.default_rng(seed)
    seqn = frame["SEQN"].to_numpy()[rng.random(len(frame)) < 0.7]
    reps = rng.integers(1, 4, len(seqn))
    return pd.DataFrame({"SEQN": np.repeat(seqn, reps),
                         "recall": np.concatenate([np.arange(1, k + 1) for k in reps]),
                         "kcal": rng.normal(2000, 400, int(reps.sum())).round()})


def test_3_one_to_many_and_many_to_one_reproduce_pandas_and_many_to_many_is_refused(tmp_path):
    demo = pd.read_sas(unpack(tmp_path, "DEMO_J"), format="xport")[["SEQN", "RIDAGEYR"]]
    long_a, long_b = recalls(demo, 1), recalls(demo, 2)
    paths = {}
    for name, frame in (("demo", demo), ("long_a", long_a), ("long_b", long_b)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        ingest(tmp_path / f"{name}.csv", tmp_path / f"{name}.parquet")
        paths[name] = tmp_path / f"{name}.parquet"
    one_many = assembly.preview(paths["demo"], paths["long_a"], on="SEQN")
    demo.merge(long_a, on="SEQN", validate="one_to_many")
    assert one_many.relation == "one-to-many" and one_many.refusal is None
    assert {k: one_many.counts()[k] for k in pandas_counts(demo, long_a, "SEQN")} == \
        pandas_counts(demo, long_a, "SEQN")
    many_one = assembly.preview(paths["long_a"], paths["demo"], on="SEQN")
    long_a.merge(demo, on="SEQN", validate="many_to_one")
    assert many_one.relation == "many-to-one"
    assert {k: many_one.counts()[k] for k in pandas_counts(long_a, demo, "SEQN")} == \
        pandas_counts(long_a, demo, "SEQN")
    # Many-to-many: pandas refuses to call it anything less ...
    for validate in ("one_to_many", "many_to_one"):
        with pytest.raises(pd.errors.MergeError):
            long_a.merge(long_b, on="SEQN", validate=validate)
    refused = assembly.preview(paths["long_a"], paths["long_b"], on="SEQN")
    # ... and the join is refused, with its reason, an example from each side, and the exits.
    a_top = long_a["SEQN"].value_counts()
    b_top = long_b["SEQN"].value_counts()
    a_key = int(min(a_top[a_top == a_top.max()].index))
    b_key = int(min(b_top[b_top == b_top.max()].index))
    assert refused.refusal["code"] == "many_to_many"
    assert refused.refusal["message"] == (
        f"`SEQN` repeats in both: `{a_key}` names {a_top.max():,} rows of the table and `{b_key}` "
        f"{b_top.max():,} rows of the file. A join on it pairs every row of a value with every "
        f"row of that value on the other side, so a row's partner would be no single row, and "
        f"every count after it would multiply. Join on a column that names one row in one of the "
        f"two files.")
    assert [e["label"] for e in refused.refusal["exits"]] == [
        "Join on a column that names one row per unit in one of the files",
        "Combine the file's rows to one per identifier first (deep assembly, planned for v2.x)"]


def test_3_blank_identifiers_never_match_and_types_must_agree(tmp_path):
    """A blank SEQN is no shared identifier: its rows are unmatched on both sides (pandas, which
    pairs blanks with blanks, agrees once they are dropped). An identifier written as text in one
    file and as numbers in the other is refused, with its reason."""
    left = pd.DataFrame({"SEQN": [1, 2, None, 4], "a": [1, 2, 3, 4]})
    right = pd.DataFrame({"SEQN": [2, None, 4, 5], "b": [5, 6, 7, 8]})
    text = pd.DataFrame({"SEQN": ["2", "4"], "c": [1, 2]})
    for name, frame in (("l", left), ("r", right), ("t", text)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        ingest(tmp_path / f"{name}.csv", tmp_path / f"{name}.parquet")
    text_csv = tmp_path / "t.csv"
    text_csv.write_text("SEQN,c\nA2,1\nA4,2\n")
    ingest(text_csv, tmp_path / "t.parquet")
    plan = assembly.preview(tmp_path / "l.parquet", tmp_path / "r.parquet", on="SEQN")
    assert plan.counts()["table_unmatched"] == 2 and plan.counts()["file_unmatched"] == 2
    assert plan.counts()["rows"] == 4 and plan.matched_keys == 2
    assert (plan.left.blank_keys, plan.right.blank_keys) == (1, 1)
    refused = assembly.preview(tmp_path / "l.parquet", tmp_path / "t.parquet", on="SEQN")
    assert refused.refusal["code"] == "identifier_types_differ"
    assert refused.refusal["message"] == ("`SEQN` holds numbers in the table and text in the file, "
                                          "so no value of one can equal a value of the other.")


def test_3_a_many_to_many_join_is_refused_through_the_server(tmp_path):
    """Through the API: the preview carries the refusal and no sentence; recording the join is
    refused (409) with the same reason and exits, and nothing is recorded."""
    demo = pd.read_sas(unpack(tmp_path, "DEMO_J"), format="xport")[["SEQN", "RIDAGEYR"]]
    long_a, long_b = recalls(demo, 1), recalls(demo, 2)
    long_a.to_csv(tmp_path / "long_a.csv", index=False)
    long_b.to_csv(tmp_path / "long_b.csv", index=False)
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, tmp_path / "long_a.csv")
        added = client.post(f"/api/projects/{drive.pid}/files",
                            json={"path": str(tmp_path / "long_b.csv")})
        assert added.status_code == 200, added.text[:400]
        fid = added.json()["id"]
        preview = client.post(f"/api/projects/{drive.pid}/join-preview",
                              json={"file": fid, "on": "SEQN"}).json()
        assert preview["refusal"]["code"] == "many_to_many" and preview["sentence"] == ""
        before = len(drive.view()["decisions"])
        resp = drive.post({"kind": "join_files", "file": fid, "on": "SEQN"})
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "many_to_many" and error["message"] == preview["refusal"]["message"]
        assert [e["label"] for e in error["exits"]] == \
            [e["label"] for e in preview["refusal"]["exits"]]
        assert len(drive.view()["decisions"]) == before


# ═════════════════════════════════════════════════════════════════════════════
# 2 · Codebook import
# ═════════════════════════════════════════════════════════════════════════════


def page_entry(html: str, variable: str) -> tuple[str, dict[str, str]]:
    """A variable's SAS label and code rows read off the NHANES page with plain string searches
    (independent of the parser under test)."""
    block = html.split(f'id="{variable}"', 1)[1].split('<div class="pagebreak">', 1)[0]
    label = block.split("<dt>SAS Label: </dt>", 1)[1].split("<dd>", 1)[1].split("</dd>", 1)[0]
    table = block.split('<table class="values">', 1)[1].split("</table>", 1)[0] \
        if '<table class="values">' in block else ""
    rows = {}
    for row in table.split("<tr>")[2:]:
        cells = [re.sub(r"<[^>]+>", "", c.split(">", 1)[1].split("</td>", 1)[0]).strip()
                 for c in row.split("<td")[1:3]]
        if len(cells) == 2:
            rows[" ".join(unescape(cells[0]).split())] = " ".join(unescape(cells[1]).split())
    return " ".join(unescape(label).split()), rows


def test_2_an_nhanes_codebook_page_is_read_variable_by_variable(tmp_path):
    """Every variable on the page, its SAS label, its codes and its documented range."""
    for name in NHANES:
        html = unpack(tmp_path, name, "htm").read_text(encoding="utf-8")
        book = cb.read(unpack(tmp_path, name, "htm"))
        assert book.form == "nhanes"
        assert len(book.entries) == html.count("<dt>Variable Name: </dt>")
        for e in book.entries:
            label, rows = page_entry(html, e.variable)
            assert e.label == label, e.variable
            codes = {k: v for k, v in rows.items()
                     if k != "." and v != "Range of Values"}
            assert e.codes == codes, e.variable
            ranges = [k for k, v in rows.items() if v == "Range of Values"]
            if ranges:
                lo, hi = (float(x.replace(",", "")) for x in ranges[0].split(" to "))
                assert e.range == (lo, hi), e.variable
            else:
                assert e.range is None, e.variable
    demo = {e.variable: e for e in cb.read(unpack(tmp_path, "DEMO_J", "htm")).entries}
    assert demo["RIAGENDR"].codes == {"1": "Male", "2": "Female"}
    assert demo["RIDAGEYR"].range == (0.0, 79.0) and demo["RIDAGEYR"].codes == {
        "80": "80 years of age and over"}


TABLE = [  # a researcher's own data dictionary for the NHANES export (units from the CDC pages)
    {"variable": "SEQN", "label": "Respondent sequence number", "unit": "", "type": "identifier",
     "codes": ""},
    {"variable": "RIAGENDR", "label": "Gender", "unit": "", "type": "categorical",
     "codes": "1=Male; 2=Female"},
    {"variable": "RIDAGEYR", "label": "Age in years at screening", "unit": "years",
     "type": "continuous", "codes": ""},
    {"variable": "RIDRETH3", "label": "Race/Hispanic origin w/ NH Asian", "unit": "",
     "type": "nominal", "codes": "1=Mexican American; 2=Other Hispanic; 3=Non-Hispanic White; "
                                 "4=Non-Hispanic Black; 6=Non-Hispanic Asian; 7=Other Race"},
    {"variable": "BMXWT", "label": "Weight (kg)", "unit": "kg", "type": "continuous",
     "codes": ""},
    {"variable": "BMXHT", "label": "Standing Height (cm)", "unit": "cm", "type": "continuous",
     "codes": ""},
    {"variable": "DR1TKCAL", "label": "Energy (kcal)", "unit": "kcal", "type": "continuous",
     "codes": ""},
    {"variable": "DR1TPROT", "label": "Protein (gm)", "unit": "g", "type": "continuous",
     "codes": ""},
    {"variable": "BMXBMI", "label": "Body Mass Index (kg/m**2)", "unit": "kg/m²",
     "type": "continuous", "codes": ""},
]


def write_table(folder: Path, rows: list[dict[str, str]], name: str) -> tuple[Path, Path]:
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / f"{name}.csv", index=False)
    frame.to_excel(folder / f"{name}.xlsx", index=False)
    return folder / f"{name}.csv", folder / f"{name}.xlsx"


def test_2_a_variable_table_reads_the_same_from_csv_excel_and_redcap_codes(tmp_path):
    csv, xlsx = write_table(tmp_path, TABLE, "dictionary")
    a, b = cb.read(csv), cb.read(xlsx)
    assert [e.to_dict() for e in a.entries] == [e.to_dict() for e in b.entries]
    entry = {e.variable: e for e in a.entries}
    assert entry["RIAGENDR"].codes == {"1": "Male", "2": "Female"}
    assert entry["BMXWT"].unit == "kg" and entry["BMXWT"].type == "continuous"
    # REDCap's data dictionary writes its choices as "1, Male | 2, Female".
    redcap = tmp_path / "redcap.csv"
    redcap.write_text('Variable / Field Name,Field Label,Field Type,"Choices, Calculations, OR '
                      'Slider Labels"\nsex,Sex,radio,"1, Male | 2, Female"\n')
    book = cb.read(redcap)
    assert book.entries[0].codes == {"1": "Male", "2": "Female"}
    assert cb.type_reading(book.entries[0].type) == "code"


def import_codebook(project: Project, path: Path) -> d.DecisionRecord:
    cid = project.stage_codebook(path)
    return project.decide(d.ImportCodebook(codebook=cid))


def test_2_structured_fields_settle_labels_only_guide_and_contradictions_are_asked(tmp_path):
    """The table's structured fields settle each reading as the user's documentation (its
    evidence names the codebook) and reach the consumers that read them: the Goldberg screen's
    body units, an energy source's kcal per gram, the outcome's unit in a sentence, the sex coding,
    codes in the fit. A label settles nothing. A unit the magnitudes reject (height in m with a
    median of 160; energy in kJ whose values match their macronutrients' energy in kcal), codes
    the data do not hold, and a categorical type over a measurement's values are asked, never
    applied: the readings stay as they were, and the exits offer both answers."""
    project = Project(tmp_path)
    for name in ("DR1TOT_J", "BMX_J"):
        project.decide(d.JoinFiles(file=project.files[name], on="SEQN"))
    # The values, read independently (pandas): BMXHT's median is a human height in cm only.
    frames = pandas_frames(tmp_path)
    height = frames["BMX_J"]["BMXHT"].dropna()
    assert 100 <= float(np.median(height)) <= 230
    wrong = [dict(r) for r in TABLE]
    for row in wrong:
        if row["variable"] == "BMXHT":
            row["unit"] = "m"
        if row["variable"] == "RIDRETH3":
            row["codes"] = "0=Mexican American; 1=Other Hispanic"
        if row["variable"] == "DR1TKCAL":
            row["unit"] = "kJ"
    wrong.append({"variable": "DR1TFIBE", "label": "Dietary fiber (gm)", "unit": "",
                  "type": "categorical", "codes": ""})
    csv, _ = write_table(tmp_path, wrong, "dictionary")
    state0 = project.state
    record = import_codebook(project, csv)
    state = project.state
    imported = record.decision
    settled = {(i.reading, i.column): i.value for i in imported.items}
    assert settled == {("code_or_count", "RIAGENDR"): "code",
                       ("sex_coding", "RIAGENDR"): "female=2,male=1",
                       ("code_or_count", "RIDAGEYR"): "amount", ("unit", "RIDAGEYR"): "years",
                       ("code_or_count", "BMXWT"): "amount", ("unit", "BMXWT"): "kg",
                       ("code_or_count", "DR1TPROT"): "amount", ("unit", "DR1TPROT"): "g",
                       ("code_or_count", "BMXBMI"): "amount"}
    asked_fields = {(x.column, x.field) for x in imported.asked}
    assert asked_fields == {("BMXHT", "unit"), ("RIDRETH3", "codes"), ("DR1TKCAL", "unit"),
                            ("DR1TFIBE", "type")}
    # Asked, never applied: none of the contradicted entries' fields reached the state.
    for column in ("BMXHT", "RIDRETH3", "DR1TKCAL", "DR1TFIBE"):
        for kind in ("unit", "code_or_count"):
            assert R.confirmation(state, kind, column) is None, (kind, column)
    # The exits answer both ways: the values' reading and the codebook's.
    preview = cb.assessment_for(cb.load(project.dir, imported.codebook), project.dir, state0)
    exits = {x["column"]: [e["decision"]["value"] for e in x["exits"]] for x in preview.asked}
    assert exits["BMXHT"] == ["cm", "m"] and exits["DR1TKCAL"] == ["kcal", "kj"]
    assert exits["RIDRETH3"] == ["amount", "code"]
    # Settled readings name the codebook and reach their consumers.
    weight = R.body_unit_reading("BMXWT", "weight", state)
    assert weight.settled and weight.value == "kg"
    assert weight.evidence == "from the codebook `dictionary.csv`, the user's own documentation"
    assert R.sex_coding_reading(state, "RIAGENDR", [1.0, 2.0]).settled
    assert R.sex_levels(state, "RIAGENDR", [1.0, 2.0]) == {"2": "female", "1": "male"}
    protein = R.kcal_per_unit(state, "DR1TPROT")
    assert protein.settled and protein.factor == 4.0
    assert R.code_or_count_reading(state, "RIAGENDR", {"whole": True, "n_values": 2},
                                   scope="combine").settled
    # The outcome's documented unit (kg/m², no reading kind's) is stated once it is the outcome.
    from turbotab.core.units import outcome_unit, recorded_unit

    assert R.codebook_unit(state, "BMXBMI") == "kg/m²"
    assert R.confirmation(state, "unit", "BMXBMI") is None
    outcome = state.model_copy(update={"target": "BMXBMI"})
    assert recorded_unit(outcome, "BMXBMI") == "kg/m²"
    assert outcome_unit("BMXBMI", recorded=recorded_unit(outcome, "BMXBMI")) == ("kg/m²",
                                                                                 "decision")
    assert recorded_unit(state, "BMXBMI") is None  # only while it is the outcome
    # A label settles nothing: BMXHT's label says cm, and its unit is still unrecorded; the ask
    # leads with the label's guess and shows it beside the evidence.
    height_reading = R.reading("unit", "BMXHT", "m", confidence="medium", evidence="", state=state)
    assert not height_reading.settled
    (guided,) = R.labeled([height_reading], state)
    assert guided.value == "cm" and not guided.settled
    assert guided.evidence == 'your codebook labels it "Standing Height (cm)"'


def test_2_the_users_own_answer_stands_and_one_revert_undoes_the_import(tmp_path):
    project = Project(tmp_path)
    project.decide(d.JoinFiles(file=project.files["BMX_J"], on="SEQN"))
    project.decide(d.ConfirmReading(reading="unit", column="BMXWT", value="lb"))
    csv, _ = write_table(tmp_path, TABLE, "dictionary")
    record = import_codebook(project, csv)
    assert record.decision.kept == ["unit:BMXWT"]
    state = project.state
    assert R.confirmation(state, "unit", "BMXWT") == "lb"
    assert R.body_unit_reading("BMXWT", "weight", state).evidence == "recorded by the user"
    assert "`1` reading already recorded otherwise keeps its recorded answer (`BMXWT`'s unit)" \
        in record.sentence
    project.decide(d.Revert(decision_id=record.id))
    state = project.state
    assert R.confirmation(state, "sex_coding", "RIAGENDR") is None
    assert not state.codebooks and R.confirmation(state, "unit", "BMXWT") == "lb"


def test_2_a_codebook_imported_again_after_a_join_settles_the_new_columns(tmp_path):
    """Imported before DR1TOT_J is joined, the dictionary documents none of its columns; imported
    again after the join, it settles DR1TPROT's grams, and the readings it had settled before still
    name it as their evidence."""
    project = Project(tmp_path)
    csv, _ = write_table(tmp_path, TABLE, "dictionary")
    first = import_codebook(project, csv)
    assert ("unit", "DR1TPROT") not in {(i.reading, i.column) for i in first.decision.items}
    project.decide(d.JoinFiles(file=project.files["DR1TOT_J"], on="SEQN"))
    again = import_codebook(project, csv)
    state = project.state
    assert R.confirmation(state, "unit", "DR1TPROT") == "g"
    assert R.kcal_per_unit(state, "DR1TPROT").factor == 4.0
    assert R.recorded_evidence(state, "sex_coding", "RIAGENDR") == \
        "from the codebook `dictionary.csv`, the user's own documentation"
    assert len(state.codebooks) == 1 and again.decision.codebook == first.decision.codebook


FIT_ROLES = {"SEQN": "identifier", "DR1TFIBE": "exposure", "DR1TKCAL": "energy",
             "RIDAGEYR": "covariate", "RIAGENDR": "covariate", "RIDRETH3": "covariate",
             "DMDEDUC2": "covariate", "DMDMARTL": "covariate", "DMDHHSIZ": "covariate",
             "INDFMPIR": "covariate", "SDMVPSU": "design", "SDMVSTRA": "design",
             "WTDRD1": "design"}


def the_card(project: Project) -> tuple[set[tuple[str, str]], R.Unsettled]:
    store = DataStore(project.table, 1 << 30)
    try:
        R.predictors_or_ask(project.state, None, store=store)
    except R.Unsettled as ask:
        return {(r.kind, r.column) for r in ask.readings}, ask
    finally:
        store.close()
    return set(), R.Unsettled("")


def test_2_on_the_nhanes_export_the_card_shrinks_by_what_the_codebooks_settle(nhanes):
    """The NHANES export (DEMO + DR1TOT + BMX on SEQN) with the three NHANES codebook pages. The
    fit's ask card before: every whole-valued predictor with three or more values (read here with
    pandas), its code-or-amount reading. After importing the pages: exactly those the pages settle
    (a code table whose descriptions are words: RIDRETH3, DMDEDUC2, DMDMARTL, read off the page with
    string searches) have left it; the rest still ask (an age's 0–79 range and its top code 80, a
    household size whose codes restate their numbers, a total energy's range), each led by its
    label's guess, which settles nothing."""
    folder, project = nhanes
    if not project.state.joins:
        for name in ("DR1TOT_J", "BMX_J"):
            project.decide(d.JoinFiles(file=project.files[name], on="SEQN"))
    project.decide(d.SetTarget(column="BMXWAIST"))
    project.decide(d.SetRoles(roles=FIT_ROLES))
    frames = pandas_frames(folder)
    joined = frames["DEMO_J"].merge(frames["DR1TOT_J"], on="SEQN", how="left") \
        .merge(frames["BMX_J"], on="SEQN", how="left")
    predictors = [c for c, role in FIT_ROLES.items() if role in ("exposure", "covariate", "energy")]
    whole = {c for c in predictors
             if np.all(np.mod(joined[c].dropna(), 1) == 0) and joined[c].nunique() >= 3}
    before, _ = the_card(project)
    assert before == {("code_or_count", c) for c in whole}, before
    html = unpack(folder, "DEMO_J", "htm").read_text(encoding="utf-8")
    worded = set()
    for c in sorted(whole):
        _label, rows = page_entry(html, c) if f'id="{c}"' in html else ("", {})
        codes = {k: v for k, v in rows.items() if k != "." and v != "Range of Values"}
        ranged = any(v == "Range of Values" for v in rows.values())
        if codes and not ranged and not all(v.split()[0] == k for k, v in codes.items()
                                            if v not in ("Refused", "Don't know")):
            worded.add(c)
    assert worded == {"RIDRETH3", "DMDEDUC2", "DMDMARTL"}
    for name in NHANES:
        import_codebook(project, unpack(folder, name, "htm"))
    state = project.state
    settled = {(i.split(":", 1)[0], i.split(":", 1)[1]) for spec in state.codebooks.values()
               for i in spec.settled}
    after, ask = the_card(project)
    assert before & settled == {("code_or_count", c) for c in worded}
    assert after == before - settled and after, (before, after)
    for r in ask.readings:   # each still asked, led by its label's guess, still unsettled
        label = R.codebook_label(state, r.column)
        assert label and f'your codebook labels it "{label}"' in r.evidence and not r.settled
    guesses = {r.column: r.value for r in ask.readings}
    assert guesses["RIDAGEYR"] == "amount" and guesses["DMDHHSIZ"] == "amount"
    # What left the card reaches the fit's design as the user's codes: one indicator per level.
    assert {"RIDRETH3", "DMDEDUC2", "DMDMARTL"} <= set(R.confirmed_codes(state))
    assert 'your codebook labels it "Age in years at screening"' in str(ask)
    # The sex coding the screens read: asked before, the codebook's after, named as its evidence.
    values = frames["DEMO_J"]["RIAGENDR"]
    assert not R.sex_coding_reading(d.fold(project.records[:4]), "RIAGENDR", values).settled
    coding = R.sex_coding_reading(state, "RIAGENDR", values)
    assert coding.settled and coding.value == "female=2,male=1"
    assert coding.evidence == "from the codebook `DEMO_J.htm`, the user's own documentation"


def test_2_xpt_labels_import_settles_nothing_and_guides_the_guesses(tmp_path):
    project = Project(tmp_path)
    side = read_labels(project.table)
    book = cb.from_labels(side["sources"], "DEMO_J.XPT")
    cid = cb.stage(book, project.dir)
    record = project.decide(d.ImportCodebook(codebook=cid))
    assert record.decision.form == "xpt" and record.decision.items == []
    assert record.decision.labels["RIDAGEYR"] == "Age in years at screening"
    assert record.sentence == (
        "The variable labels carried by `DEMO_J.XPT` describe `46` of the table's columns; a label "
        "is a name, so they settle no reading and are shown beside the guesses the questions lead "
        "with.")


# ═════════════════════════════════════════════════════════════════════════════
# 4 · The methods sentences, verbatim
# ═════════════════════════════════════════════════════════════════════════════


def test_4_the_join_and_the_codebook_are_named_in_their_methods_sentences(tmp_path):
    project = Project(tmp_path)
    frames = pandas_frames(tmp_path)
    c = pandas_counts(frames["DEMO_J"], frames["DR1TOT_J"], "SEQN")
    record = project.decide(d.JoinFiles(file=project.files["DR1TOT_J"], on="SEQN"))
    added = frames["DR1TOT_J"].shape[1] - 1
    assert record.sentence == (
        f"`DR1TOT_J.XPT` (`{c['file_rows']:,}` rows) was joined one-to-one to the table "
        f"(`{c['table_rows']:,}` rows) on `SEQN`, keeping every row of the table; "
        f"`{c['table_rows'] - c['table_unmatched']:,}` of the table's rows found a partner and "
        f"`{c['table_unmatched']:,}` did not, their `{added}` new columns blank; "
        f"`{c['file_rows'] - c['file_unmatched']:,}` of its rows matched and "
        f"`{c['file_unmatched']:,}` matched no row of the table and were left out. The joined "
        f"table has `{c['rows']:,}` rows.")
    record = import_codebook(project, unpack(tmp_path, "BMX_J", "htm"))
    bmx = cb.read(unpack(tmp_path, "BMX_J", "htm"))
    worded = [e.variable for e in bmx.entries if e.codes and e.range is None]
    # BMX_J is not joined here: its codebook documents only SEQN of this table.
    assert record.decision.n_matched == 1 and record.sentence == (
        "The codebook `BMX_J.htm` (an NHANES codebook page) documents `1` of the table's columns. "
        "Its structured fields settled no reading. Its labels of `1` column are shown beside the "
        "guesses the questions lead with and settle nothing.")
    assert worded  # (it lists the comment codes of columns this table does not have)
    record = import_codebook(project, unpack(tmp_path, "DEMO_J", "htm"))
    html = unpack(tmp_path, "DEMO_J", "htm").read_text(encoding="utf-8")
    n_vars = html.count("<dt>Variable Name: </dt>")
    items = record.decision.items
    codes = [i.column for i in items if i.reading == "code_or_count"]
    assert record.sentence == (
        f"The codebook `DEMO_J.htm` (an NHANES codebook page) documents `{n_vars}` of the table's "
        f"columns. Its structured fields settled `{len(codes)}` columns as codes for categories "
        f"(`{codes[0]}`, `{codes[1]}`, `{codes[2]}`, `{codes[3]}` and `{len(codes) - 4}` more); and "
        f"the sex coding of `RIAGENDR` (`2` female, `1` male) and `DMDHRGND` (`2` female, `1` "
        f"male), as the user's own documentation. Its labels of `{n_vars}` columns are shown "
        f"beside the guesses the questions lead with and settle nothing.")


# ═════════════════════════════════════════════════════════════════════════════
# The chain (BLUEPRINT §13): upload → join → codebook → the card, through the server
# ═════════════════════════════════════════════════════════════════════════════


def test_chain_upload_join_codebook_card_through_the_server(tmp_path):
    """XPT upload; two files added and joined after their previews (the preview's counts are the
    recorded ones: the relation "implies the row counts are previewed"); the three codebook pages
    staged (their previews are what the import records) and imported; the methods record holds,
    in order, the joins and the codebooks, each naming its file; the readings the codebooks
    settled are recorded with the codebook as their evidence."""
    for name in NHANES:
        unpack(tmp_path, name)
        unpack(tmp_path, name, "htm")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, tmp_path / "DEMO_J.XPT")
        url = f"/api/projects/{drive.pid}"
        for name in ("DR1TOT_J", "BMX_J"):
            fid = client.post(f"{url}/files", json={"path": str(tmp_path / f"{name}.XPT")}).json()["id"]
            preview = client.post(f"{url}/join-preview", json={"file": fid, "on": "SEQN"}).json()
            assert preview["refusal"] is None and preview["relation"] == "one-to-one"
            drive.decide({"kind": "join_files", "file": fid, "on": "SEQN"})
            recorded = drive.view()["decisions"][-1]
            assert recorded["decision"]["counts"]["rows"] == preview["rows"]
            assert recorded["sentence"] == preview["sentence"]
            drive.artifact("ingest")
        imported = []
        for name in NHANES:
            staged = client.post(f"{url}/codebooks", json={"path": str(tmp_path / f"{name}.htm")})
            assert staged.status_code == 200, staged.text[:400]
            preview = staged.json()
            drive.decide({"kind": "import_codebook", "codebook": preview["id"]})
            record = drive.view()["decisions"][-1]
            assert record["sentence"] == preview["sentence"]
            assert [(i["reading"], i["column"], i["value"]) for i in record["decision"]["items"]] \
                == [(s["reading"], s["column"], s["value"]) for s in preview["settles"]]
            imported.append(record)
        sentences = [r["sentence"] for r in drive.view()["decisions"]]
        assert sentences[0].startswith("`DR1TOT_J.XPT` (") and sentences[1].startswith("`BMX_J.XPT` (")
        assert [s.split("`")[1] for s in sentences[2:5]] == ["DEMO_J.htm", "DR1TOT_J.htm", "BMX_J.htm"]
        state = d.ProjectState.model_validate(drive.view()["state"])
        reading = R.sex_coding_reading(state, "RIAGENDR", [1, 2])
        assert reading.settled and "`DEMO_J.htm`" in reading.evidence
        labels = client.post(f"{url}/codebooks", json={"labels": True}).json()
        assert labels["form"] == "xpt" and labels["settles"] == []


def test_contracts_declare_every_field_of_the_method_contract():
    """BLUEPRINT §13: slot, data scope, needs, routing (question, options, leash), storyboard,
    sentence and relations, for the join and the codebook import, each in the one registry."""
    from turbotab.core.contracts import CONTRACTS

    for contract in (assembly.CONTRACT, cb.CONTRACT):
        assert CONTRACTS[contract.key] is contract and contract.slot == "ingest"
        assert contract.scope_note and contract.needs and contract.question and contract.place
        assert contract.options and contract.storyboard and contract.sentence and contract.relations
    assert assembly.CONTRACT.scope == "row_local" and cb.CONTRACT.scope == "descriptive"
    many = assembly.CONTRACT.relation("conflicts", "many_to_many")
    assert many.rung == "refused" and many.exits
    assert cb.CONTRACT.relation("contradicted by the values").says == "asked, never applied"
    assert cb.CONTRACT.relation("free-text label").says.startswith("strengthens the guess")
