"""Rebuild ``plausibility.json``'s NHANES percentiles, and the test extract they are checked on.

Run by hand, never at import (it needs the NHANES files, which are not in the repository)::

    venv/bin/python -m turbotab.core.detectors.data.build_plausibility <folder with the .xpt files>

The folder holds the NHANES 2017–2018 (cycle J) public files ``DEMO_J``, ``BPX_J``, ``BMX_J``,
``GLU_J``, ``TRIGLY_J``, ``GHB_J`` and ``TCHOL_J`` as published at
``https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<FILE>.xpt``.

The percentiles are the 1st and 99th of adults aged 20 and over, weighted by the examination
weight (``WTMEC2YR``), or by the fasting-subsample weight (``WTSAF2YR``) for the two analytes
measured on the morning fasting subsample (glucose, triglycerides). A weighted percentile is the
smallest value whose cumulative weight share reaches the probability. The extract the acceptance
test recomputes them from (``turbotab/core/tests/acceptance/wp14_data``) is written beside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXTRACT = HERE.parents[1] / "tests" / "acceptance" / "wp14_data" / "nhanes_2017_2018_adults.csv.gz"

#: variable -> (NHANES file, NHANES column, weight column, unit, decimals kept)
SOURCES = {
    "bp_sys": ("BPX_J", "BPXSY1", "WTMEC2YR", "mmHg", 0),
    "bp_di": ("BPX_J", "BPXDI1", "WTMEC2YR", "mmHg", 0),
    "weight": ("BMX_J", "BMXWT", "WTMEC2YR", "kg", 1),
    "height": ("BMX_J", "BMXHT", "WTMEC2YR", "cm", 1),
    "bmi": ("BMX_J", "BMXBMI", "WTMEC2YR", "kg/m²", 1),
    "waist": ("BMX_J", "BMXWAIST", "WTMEC2YR", "cm", 1),
    "glucose": ("GLU_J", "LBXGLU", "WTSAF2YR", "mg/dL", 0),
    "triglyceride": ("TRIGLY_J", "LBXTR", "WTSAF2YR", "mg/dL", 0),
    "hba1c": ("GHB_J", "LBXGH", "WTMEC2YR", "%", 1),
    "cholesterol": ("TCHOL_J", "LBDTCSI", "WTMEC2YR", "mmol/L", 2),
}


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    share = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(share, q, side="left")])


def extract(folder: Path) -> pd.DataFrame:
    read = lambda name: pd.read_sas(folder / f"{name}.xpt", format="xport")  # noqa: E731
    frame = read("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR"]]
    for name in ("BPX_J", "BMX_J", "GLU_J", "TRIGLY_J", "GHB_J", "TCHOL_J"):
        table = read(name)
        wanted = ["SEQN"] + [c for _, (f, c, w, _, _) in SOURCES.items() if f == name]
        if "WTSAF2YR" in table.columns and "WTSAF2YR" not in frame.columns:
            wanted.append("WTSAF2YR")
        frame = frame.merge(table[list(dict.fromkeys(wanted))], on="SEQN", how="left")
    frame = frame[frame["RIDAGEYR"] >= 20].reset_index(drop=True)
    return frame


def bands(frame: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for var, (file, column, weight, unit, decimals) in SOURCES.items():
        ok = frame[column].notna() & frame[weight].notna() & (frame[weight] > 0)
        x = frame.loc[ok, column].to_numpy(dtype=float)
        w = frame.loc[ok, weight].to_numpy(dtype=float)
        out[var] = {"p01": round(weighted_quantile(x, w, 0.01), decimals),
                    "p99": round(weighted_quantile(x, w, 0.99), decimals),
                    "unit": unit, "nhanes_file": file, "nhanes_variable": column,
                    "weight": weight, "n": int(ok.sum())}
    return out


def main(folder: str) -> None:
    frame = extract(Path(folder))
    EXTRACT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXTRACT, index=False, compression="gzip", float_format="%.6g")
    computed = bands(pd.read_csv(EXTRACT))
    path = HERE / "plausibility.json"
    table = json.loads(path.read_text("utf-8"))
    for var, band in computed.items():
        table["variables"][var]["improbable"].update(band)
    path.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(computed, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
