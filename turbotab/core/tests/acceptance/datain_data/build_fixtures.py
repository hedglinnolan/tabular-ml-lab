"""Rebuild the DATAIN fixtures from the public NHANES files (run by hand, not by pytest).

    venv/bin/python turbotab/core/tests/acceptance/datain_data/build_fixtures.py <folder> [NAME ...]

``<folder>`` holds the files as CDC publishes them (fetched 2026-10-05); with NAMEs, only those
fixtures are rebuilt:

    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT   (+ DEMO_J.htm)
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.XPT (+ DR1TOT_J.htm)
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.XPT    (+ BMX_J.htm)
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.XPT (+ DR1TOT_I.htm)
        (SHA-256 185b8f112cdf22175470fe8cc7d4d20800fe28e8ac37c50dc4bf197d40333b6e; its page
        e72f7d4aae58648b513049b970bc76bf7a06c113ca4e604d9b5a543009acb9b7)

Each fixture XPT is the real file cut byte for byte: its library, member and variable headers
unchanged, and the observations whose ``SEQN`` falls in a window (or is one of the listed rows),
padded with blanks to an 80-byte record as TS-140 pads the last record. The windows differ, so a
join has rows with no partner on each side. DR1TOT_I keeps the eight rows that hold the maximum of
twelve variables whose maximum, read as an IEEE double, is one unit in the last place above the
bound the page prints (DR1TOT_I's DR1TSFAT: 223.75900000000001 against "0 to 223.759"). The
codebook pages are copied whole. Everything is gzipped. This script parses only what the cut needs
(the NAMESTR count, each variable's length and position, the OBS record); it shares no code with
``turbotab.core.xport``.
"""
from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path
from typing import Callable, Sequence

HERE = Path(__file__).resolve().parent
WINDOWS = {"DEMO_J": (93703, 94002), "DR1TOT_J": (93800, 94120), "BMX_J": (93703, 93950)}
ROWS = {"DR1TOT_I": (83966, 85549, 85963, 87422, 87895, 88545, 91053, 91611)}


def ibm(b: bytes) -> float | None:
    b = b.ljust(8, b"\0")
    if b[1:] == b"\0" * 7 and b[0] != 0:
        return None
    (word,) = struct.unpack(">Q", b)
    sign = -1.0 if word >> 63 else 1.0
    return sign * (word & ((1 << 56) - 1)) * 16.0 ** (((word >> 56) & 0x7F) - 64) / 2.0 ** 56


def cut(src: Path, keep: "Callable[[float], bool]") -> bytes:
    raw = src.read_bytes()
    recs = [raw[i:i + 80] for i in range(0, len(raw), 80)]
    namestr = next(i for i, r in enumerate(recs) if r[20:28] == b"NAMESTR ")
    n_vars = int(recs[namestr][53:58])
    start = (namestr + 1) * 80
    variables = []
    for k in range(n_vars):
        ns = raw[start + 140 * k: start + 140 * (k + 1)]
        length = struct.unpack(">h", ns[4:6])[0]
        name = ns[8:16].decode().strip()
        position = struct.unpack(">i", ns[84:88])[0]
        variables.append((name, length, position))
    obs_at = next(i for i, r in enumerate(recs) if r[20:28] == b"OBS     ") + 1
    data_start = obs_at * 80
    obs_len = max(p + n for _, n, p in variables)
    seqn = next((n, p) for name, n, p in variables if name == "SEQN")
    n_obs = (len(raw) - data_start) // obs_len
    kept = []
    for i in range(n_obs):
        row = raw[data_start + i * obs_len: data_start + (i + 1) * obs_len]
        value = ibm(row[seqn[1]:seqn[1] + seqn[0]])
        if value is not None and keep(value):
            kept.append(row)
    body = b"".join(kept)
    body += b" " * ((-len(body)) % 80)
    return raw[:data_start] + body


def main(folder: Path, names: Sequence[str] = ()) -> None:
    keeps: dict[str, Callable[[float], bool]] = {
        **{name: (lambda v, lo=lo, hi=hi: lo <= v <= hi) for name, (lo, hi) in WINDOWS.items()},
        **{name: (lambda v, rows=frozenset(rows): v in rows) for name, rows in ROWS.items()}}
    for name, keep in keeps.items():
        if names and name not in names:
            continue
        data = cut(folder / f"{name}.XPT", keep)
        (HERE / f"{name}.XPT.gz").write_bytes(gzip.compress(data, mtime=0))
        page = (folder / f"{name}.htm").read_bytes()
        (HERE / f"{name}.htm.gz").write_bytes(gzip.compress(page, mtime=0))
        print(name, len(data), "bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
