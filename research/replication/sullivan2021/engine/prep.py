"""The prep for the Sullivan 2021 replication: only what TurboTab's engine cannot do today.

SULLIVAN_PLAN ruling 2: the prep does only what the engine cannot do, and each step it takes is an
engine gap the report names. What the engine does is left to it, through the API: it reads the
files (SAS transport, through its own reader), joins them by ``SEQN`` (``join_files``), applies
every exclusion (``set_exclusions``: age, pregnancy, the rating's codes, the recall status, a
zero-energy recall day), and sets the outcome, the exposure, the covariates, their forms (PIR's
bands are a declared ``categories`` form), the survey design and the estimand.

What is left here, each because no decision or route does it (``git grep`` of decisions.py,
recognizers.py and methods/ on 2026-10-18; the report lists each with its INBOX line):

1. **Stacking the four cycles** (``stack``): each component's four files, one under the other.
   ``turbotab.core.methods.cycles.stack_cycles`` exists but is engine only (no decision or route
   reaches it). The cycle column is NHANES's own ``SDDSRVYR`` (7, 8, 9, 10), already in DEMO.
   The files are read with the engine's own XPT reader (``turbotab.core.xport``), which reads a
   SAS zero as 0 (pandas reads it as 5.4e-79).
2. **Insulin's source file**: 2011–2012 ships insulin in GLU_G, later cycles in INS_H/I/J. The
   stacked INS file takes ``LBXIN`` from each (a per-cycle rename the engine cannot declare). The
   fasting weight comes from GLU alone: INS's ``WTSAF2YR`` equals GLU's row for row (checked).
3. **Derived columns** (``derived.parquet``, keyed by ``SEQN``), each a function of several columns
   (no derivation decision exists):
   - ``race5``: RIDRETH3 with Mexican American (1) and Other Hispanic (2) merged into Hispanic (1);
     3, 4, 6 and 7 kept (a category merge: no decision merges levels);
   - ``educ4``: DMDEDUC2 1–2 → 1 (less than high school), 3 → 2, 4 → 3, 5 → 4; refused (7) and
     don't know (9) blank (a category merge);
   - ``smoke_now``: 1 when SMQ020 = 1 and SMQ040 is 1 or 2 (every day, some days); 0 when SMQ020
     = 2 or SMQ040 = 3; blank otherwise;
   - ``pa_rec_min``: weekly moderate recreational minutes (PAQ670 × PAD675) + 2 × weekly vigorous
     ones (PAQ655 × PAD660), each 0 when its yes/no item (PAQ665, PAQ650) is 2; blank when an item
     is refused or not known (7, 9, 77, 99, 7777, 9999) or missing;
   - ``kcal_mean``: the mean of the two days' energy, (DR1TKCAL + DR2TKCAL) / 2 (M3's total energy);
   - ``glu_med``: glucose-lowering medication, 1 when DIQ050 = 1 (insulin now) or DIQ070 = 1
     (diabetic pills); 0 when DIQ010 is answered (1, 2, 3) and neither is 1 nor refused or not
     known; blank otherwise (M4's medication);
   - ``water_only_alt``: 1 when a recall day has zero energy and plain water reported
     (DRxTKCAL = 0 and DRx_320Z > 0); the §3 alternative to the zero-energy rule the engine
     applies (which needs no derivation);
   - ``LBXIN_xw``: insulin with 2011–2012's Roche values put on the Tosoh scale of 2013–2018 by
     the INS_H codebook's forward Deming equation, 10^(1.024 · log10(x) − 0.0802); later cycles as
     measured (an unstated decision the codebooks raise; §3 measures it).

Nothing else is changed: no recode of DBQ700 (ruling 3: the engine fits the raw codes and the
report compares −β), no row is dropped (the engine excludes), no value is imputed.

    scripts/py research/replication/sullivan2021/engine/prep.py

writes ``research/data/derived/prep/*.parquet`` and ``prep_log.json`` beside them (ignored by git).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from turbotab.core.xport import read_frame  # noqa: E402  (the engine's own XPT reader)

DATA = REPO / "research" / "data" / "nhanes"
OUT = REPO / "research" / "data" / "derived" / "prep"
CYCLES = {"2011-2012": "G", "2013-2014": "H", "2015-2016": "I", "2017-2018": "J"}
# Each stacked component: its source file stem per cycle (None: the file of another component)
# and the variables kept. Only the variables the analysis reads are kept; the engine could leave
# the rest out by their roles, so this is a convenience, not a gap (the report says so).
COMPONENTS: dict[str, dict] = {
    "DEMO": {"vars": ["SEQN", "SDDSRVYR", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2",
                      "INDFMPIR", "RIDEXPRG", "SDMVPSU", "SDMVSTRA"]},
    "DBQ": {"vars": ["SEQN", "DBQ700"]},
    "DR1TOT": {"vars": ["SEQN", "DR1DRSTZ", "DR1TKCAL"]},
    "DR2TOT": {"vars": ["SEQN", "DR2DRSTZ", "DR2TKCAL", "WTDR2D"]},
    "GLU": {"vars": ["SEQN", "LBXGLU", "WTSAF2YR"]},
    "INS": {"vars": ["SEQN", "LBXIN"], "source": {"G": "GLU_G"}},
    "GHB": {"vars": ["SEQN", "LBXGH"]},
    "BMX": {"vars": ["SEQN", "BMXBMI"]},
    "DIQ": {"vars": ["SEQN", "DIQ010"]},
}
# Read for the derived columns only (not joined as files of their own).
SOURCES_ONLY = {"SMQ": ["SEQN", "SMQ020", "SMQ040"],
                "PAQ": ["SEQN", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"],
                "DIQX": ["SEQN", "DIQ010", "DIQ050", "DIQ070"],
                "DR1X": ["SEQN", "DR1TKCAL", "DR1_320Z"], "DR2X": ["SEQN", "DR2TKCAL", "DR2_320Z"]}
FILE_OF = {"DIQX": "DIQ", "DR1X": "DR1TOT", "DR2X": "DR2TOT"}
# The INS_H codebook's forward Deming regression (Roche 2011–2012 → Tosoh-equivalent).
XW_SLOPE, XW_INTERCEPT = 1.024, -0.0802


def read(stem: str, cycle: str) -> pd.DataFrame:
    return read_frame(DATA / cycle / f"{stem}.xpt")


def stack(name: str, variables: list[str], source: dict[str, str] | None = None) -> pd.DataFrame:
    """One component's four files stacked (``SDDSRVYR`` is DEMO's; the others carry ``SEQN``)."""
    parts = []
    for cycle, s in CYCLES.items():
        stem = (source or {}).get(s, f"{FILE_OF.get(name, name)}_{s}")
        frame = read(stem, cycle)
        missing = [v for v in variables if v not in frame.columns]
        if missing:
            raise SystemExit(f"{stem} lacks {missing}")
        parts.append(frame[variables])
    out = pd.concat(parts, ignore_index=True)
    if not out["SEQN"].is_unique:
        raise SystemExit(f"{name}: SEQN repeats across cycles")
    return out


def plain(x: pd.Series, valid: tuple[float, float]) -> pd.Series:
    """``x`` where it lies in ``valid`` (the codebook's range), blank elsewhere (its refused and
    don't-know codes)."""
    return x.where((x >= valid[0]) & (x <= valid[1]))


def weekly(yes: pd.Series, days: pd.Series, minutes: pd.Series) -> pd.Series:
    """Weekly minutes of one kind of recreational activity: 0 when its yes/no item is 2 (no),
    days × minutes per day when it is 1 and both are recorded, blank otherwise."""
    out = pd.Series(np.nan, index=yes.index)
    out[yes == 2] = 0.0
    on = yes == 1
    out[on] = (plain(days, (1, 7)) * plain(minutes, (1, 1440)))[on]
    return out


def derive(demo: pd.DataFrame, cycle_of: pd.Series) -> tuple[pd.DataFrame, dict]:
    smq = stack("SMQ", SOURCES_ONLY["SMQ"])
    paq = stack("PAQ", SOURCES_ONLY["PAQ"])
    diq = stack("DIQX", SOURCES_ONLY["DIQX"])
    dr1 = stack("DR1X", SOURCES_ONLY["DR1X"])
    dr2 = stack("DR2X", SOURCES_ONLY["DR2X"])
    ins = stack("INS", ["SEQN", "LBXIN"], {"G": "GLU_G"})
    out = demo[["SEQN"]].copy()
    r = demo["RIDRETH3"]
    out["race5"] = r.where(r != 2, 1.0)
    e = demo["DMDEDUC2"]
    out["educ4"] = e.map({1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0})
    s = out[["SEQN"]].merge(smq, on="SEQN", how="left")
    smoke = pd.Series(np.nan, index=s.index)
    smoke[(s["SMQ020"] == 2) | ((s["SMQ020"] == 1) & (s["SMQ040"] == 3))] = 0.0
    smoke[(s["SMQ020"] == 1) & s["SMQ040"].isin([1, 2])] = 1.0
    out["smoke_now"] = smoke.to_numpy()
    p = out[["SEQN"]].merge(paq, on="SEQN", how="left")
    vig = weekly(p["PAQ650"], p["PAQ655"], p["PAD660"])
    mod = weekly(p["PAQ665"], p["PAQ670"], p["PAD675"])
    out["pa_rec_min"] = (mod + 2.0 * vig).to_numpy()
    k = out[["SEQN"]].merge(dr1, on="SEQN", how="left").merge(dr2, on="SEQN", how="left")
    out["kcal_mean"] = ((k["DR1TKCAL"] + k["DR2TKCAL"]) / 2.0).to_numpy()
    water = (((k["DR1TKCAL"] == 0) & (k["DR1_320Z"] > 0))
             | ((k["DR2TKCAL"] == 0) & (k["DR2_320Z"] > 0)))
    out["water_only_alt"] = water.astype(float).to_numpy()
    d = out[["SEQN"]].merge(diq, on="SEQN", how="left")
    med = pd.Series(np.nan, index=d.index)
    unsure = d["DIQ050"].isin([7, 9]) | d["DIQ070"].isin([7, 9])
    med[d["DIQ010"].isin([1, 2, 3]) & ~unsure] = 0.0
    med[(d["DIQ050"] == 1) | (d["DIQ070"] == 1)] = 1.0
    out["glu_med"] = med.to_numpy()
    i = out[["SEQN"]].merge(ins, on="SEQN", how="left")
    x = i["LBXIN"].to_numpy(dtype=float)
    roche = (cycle_of.to_numpy() == 7) & np.isfinite(x) & (x > 0)
    xw = x.copy()
    xw[roche] = 10.0 ** (XW_SLOPE * np.log10(x[roche]) + XW_INTERCEPT)
    out["LBXIN_xw"] = xw
    log = {
        "race5": {str(int(k)): int(v) for k, v in out["race5"].value_counts().sort_index().items()},
        "educ4": {str(int(k)): int(v) for k, v in out["educ4"].value_counts().sort_index().items()},
        "smoke_now": {str(int(k)): int(v) for k, v in out["smoke_now"].value_counts().items()},
        "pa_rec_min_blank": int(out["pa_rec_min"].isna().sum()),
        "kcal_mean_blank": int(out["kcal_mean"].isna().sum()),
        "water_only_alt": int(out["water_only_alt"].sum()),
        "glu_med": {str(int(k)): int(v) for k, v in out["glu_med"].value_counts().items()},
        "insulin_crosswalked_rows": int(roche.sum()),
    }
    return out, log


def write(frame: pd.DataFrame, name: str) -> dict:
    path = OUT / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": str(path.relative_to(REPO)), "rows": len(frame), "columns": list(frame.columns),
            "sha256": digest}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"files": {}, "checks": {}}
    stacked = {name: stack(name, c["vars"], c.get("source")) for name, c in COMPONENTS.items()}
    # The fasting weight is taken from GLU alone: INS's equals it row for row.
    for cycle, s in CYCLES.items():
        if s == "G":
            continue
        g, ins = read(f"GLU_{s}", cycle), read(f"INS_{s}", cycle)
        m = g[["SEQN", "WTSAF2YR"]].merge(ins[["SEQN", "WTSAF2YR"]], on="SEQN", how="outer",
                                          suffixes=("_glu", "_ins"))
        same = bool(np.array_equal(m["WTSAF2YR_glu"].to_numpy(), m["WTSAF2YR_ins"].to_numpy(),
                                   equal_nan=True))
        log["checks"][f"WTSAF2YR GLU_{s} = INS_{s}"] = same
        if not same:
            raise SystemExit(f"GLU_{s} and INS_{s} disagree on WTSAF2YR")
    demo = stacked["DEMO"]
    derived, log["derived"] = derive(demo, demo["SDDSRVYR"])
    for name, frame in stacked.items():
        log["files"][name] = write(frame, name)
    log["files"]["derived"] = write(derived, "derived")
    log["checks"]["DEMO rows"] = len(demo)
    (OUT / "prep_log.json").write_text(json.dumps(log, indent=1) + "\n")
    for name, f in log["files"].items():
        print(f"{name}: {f['rows']:,} rows, {len(f['columns'])} columns")
    print(json.dumps(log["derived"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
