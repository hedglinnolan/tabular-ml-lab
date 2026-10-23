"""The prep for the Bang 2009 replication: only what TurboTab's engine cannot do today.

As SU-A's prep (``research/replication/sullivan2021/engine/prep.py``), it does only what no
decision or route does, and the report names each step as an engine gap. The table is the intake's
(``research/datasets/bang2009/card.json``: NHANES 1999–2004 stacked and joined by ``SEQN``, every
participant kept), held to its manifest's SHA-256. What the engine does is left to it: the
eligibility (age, pregnancy, known diabetes by its codes, the fasting subsample) is the analyst's
rules, recorded in the app; the survey design, the model and its fit are the engine's.

What is left here is the six predictors of the paper's Table 2, each a function of one or several
columns, coded as small integers with 0 the reference level (the engine codes a category's levels
in sorted order, the first the reference, and no decision names another reference or derives a
column). The rules are the paper's words as the notes quote them (``bang2009.md`` §2.3); each
option of an unstated decision (§6) is its own column, so a run names its option by the column it
fits (``prereg.json``):

* ``age_band``: 0 under 40, 1 for 40–49, 2 for 50–59, 3 for 60 or older (``RIDAGEYR``, whole
  years);
* ``male``: 1 for ``RIAGENDR`` 1 (male), 0 for 2 (female);
* ``famhx``: 1 for ``MCQ250A`` 1, a blood relative told of diabetes (grandparents included), else
  0; ``famhx_close``: 1 for a parent or a sibling among ``MCQ260AA``, ``AB``, ``AG``, ``AH`` (codes
  1, 2, 7 and 8), else 0;
* ``htn``: 1 for a history of hypertension (``BPQ020`` 1), the prescribed medicine now taken
  (``BPQ050A`` 1), or the mean of the recorded readings at or above the systolic or diastolic cut
  (:data:`SBP_HIGH`, :data:`DBP_HIGH` mmHg), else 0; ``htn_first`` reads the first recorded
  reading in place of the mean; ``htn_told_rx`` reads ``BPQ040A`` 1 (told to take the medicine)
  in place of ``BPQ050A``;
* ``obesity``: the higher of the BMI's category and the waist's (sex-specific, in inches), 0 not
  overweight or obese, 1 overweight, 2 obese, 3 extremely obese (:data:`BMI_CUTS`,
  :data:`WAIST_CUTS`); blank where both are missing; ``obesity_absent`` is 0 there instead;
* ``active``: 1 for ``PAQ520`` 1 (more active than others of the same age), else 0.

A categorical answer that is missing or unknown (refused, don't know, blank) is the condition
absent, as the paper says ("If information was missing or unknown in categorical variables, we
defined the condition as absent"); a measurement that is missing is read as not taken, so the
mean of the blood pressure readings is over the readings recorded, and a category made from two
measurements uses the one recorded. A diastolic reading of 0 is a recorded reading, as the file
holds it.

**The fasting subsample's rows** (:func:`subsample`). The table keeps only the participants with a
record in the fasting glucose file, those whose ``WTSAF2YR`` is recorded (zero where they did not
fast as NCHS requires): no other participant has a glucose result, so none is ever analyzed. The
engine needs it: its survey answer reads the weights over every row of the table and refuses the
NCHS six-year fasting weight wherever ``WTSAF4YR`` is blank on a 1999–2002 row
(``turbotab/core/methods/survey.py``'s ``analysis_weights``), which is every 1999–2002 participant
outside the subsample, whose two-year fasting weight is blank as well. Each stratum keeps its
PSUs (checked), so the design's linearized variance, which counts the PSUs within each stratum, is
the one over the whole sample. Nothing else is changed: no other row is dropped and no value is
imputed.

    scripts/py research/replication/bang2009/prep.py

writes ``research/data/derived/bang2009/derived.parquet`` and ``prep_log.json`` beside it (ignored
by git).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = REPO / "research" / "datasets" / "bang2009" / "manifest.json"
OUT = REPO / "research" / "data" / "derived" / "bang2009"
DERIVED = OUT / "derived.parquet"

# The paper's cut points (bang2009.md §2.3, Table 2 and its footnote).
AGE_CUTS = (40, 50, 60)  # years: under 40 (the reference), 40-49, 50-59, 60 or older
BMI_CUTS = (25.0, 30.0, 40.0)  # kg/m2: overweight, obese, extremely obese from each
WAIST_CUTS = {1: (37.0, 40.0, 50.0), 2: (31.5, 35.0, 49.0)}  # inches, by RIAGENDR: 1 male, 2 female
CM_PER_INCH = 2.54  # BMXWAIST is in cm (bang2009.md §3)
SBP_HIGH, DBP_HIGH = 140.0, 90.0  # mmHg: a systolic or diastolic pressure at or above it
YES = 1  # the NHANES code for yes in BPQ020, BPQ040A, BPQ050A and MCQ250A
MALE = 1  # RIAGENDR
MORE_ACTIVE = 1  # PAQ520: "more active"
# MCQ260AA-AI each hold one relative's code; a parent or a sibling (bang2009.md §6, choice 6).
CLOSE_RELATIVES = {"MCQ260AA": 1, "MCQ260AB": 2, "MCQ260AG": 7, "MCQ260AH": 8}
SYSTOLIC = [f"BPXSY{i}" for i in range(1, 5)]
DIASTOLIC = [f"BPXDI{i}" for i in range(1, 5)]

# What the derived table keeps of the intake's: the identifier, the survey design, the columns the
# eligibility rules read, and the outcome (``LBXGLU``, its source, stays out of the model's table).
KEPT = ["SEQN", "SDDSRVYR", "SDMVSTRA", "SDMVPSU", "WTSAF2YR", "WTSAF4YR", "RIDAGEYR", "RIDEXPRG",
        "DIQ010", "DIQ050", "DIQ070", "diabetes_fpg"]
PREDICTORS = ["age_band", "male", "famhx", "famhx_close", "htn", "htn_first", "htn_told_rx",
              "obesity", "obesity_absent", "active"]


def _is(series: pd.Series, code: float) -> pd.Series:
    """True where ``series`` holds ``code``; a blank is not it (the condition absent)."""
    return pd.to_numeric(series, errors="coerce").eq(code).fillna(False).astype(bool)


def codes(values: Any) -> pd.Series:
    """Integer codes as a nullable integer column."""
    return pd.Series(values).astype("Int64")


def age_band(age: pd.Series) -> pd.Series:
    """0 under the first cut, then one more at each cut reached (:data:`AGE_CUTS`)."""
    a = pd.to_numeric(age, errors="coerce")
    band = sum((a >= c).astype(int) for c in AGE_CUTS)
    return codes(band.where(a.notna()))


def category(value: pd.Series, cuts: Any) -> pd.Series:
    """How many of the three cuts ``value`` reaches (0 to 3); blank where ``value`` is."""
    v = pd.to_numeric(value, errors="coerce")
    cuts = np.asarray(cuts, dtype=float)
    if cuts.ndim == 1:
        cuts = np.broadcast_to(cuts, (len(v), len(cuts)))
    out = (v.to_numpy()[:, None] >= cuts).sum(axis=1).astype(float)
    out[v.isna().to_numpy()] = np.nan
    return pd.Series(out, index=v.index)


def obesity(frame: pd.DataFrame, missing_as_absent: bool = False) -> pd.Series:
    """The paper's four levels: the higher of the BMI's category and the waist's (sex-specific,
    the waist in inches); blank where neither is measured, or 0 with ``missing_as_absent``."""
    by_bmi = category(frame["BMXBMI"], BMI_CUTS)
    sex = pd.to_numeric(frame["RIAGENDR"], errors="coerce")
    cuts = np.array([WAIST_CUTS.get(int(s), (np.nan,) * 3) if pd.notna(s) else (np.nan,) * 3
                     for s in sex], dtype=float).reshape(len(frame), 3)
    waist = pd.to_numeric(frame["BMXWAIST"], errors="coerce") / CM_PER_INCH
    by_waist = category(waist, cuts)
    by_waist[waist.isna() | sex.isna()] = np.nan
    level = pd.concat([by_bmi, by_waist], axis=1).max(axis=1, skipna=True)
    if missing_as_absent:
        level = level.fillna(0)
    return codes(level)


def first_recorded(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """The lowest-numbered reading recorded among ``columns``; blank where none is."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return values.bfill(axis=1).iloc[:, 0]


def hypertension(frame: pd.DataFrame, reading: str = "mean", medicine: str = "BPQ050A") -> pd.Series:
    """1 for a history of hypertension, the medicine (``medicine`` 1), or a systolic or diastolic
    pressure at or above its cut (``reading``: the mean of the recorded readings, or the first
    recorded); else 0, a blank or unknown answer and an unmeasured pressure being the condition
    absent."""
    if reading == "mean":
        sbp = frame[SYSTOLIC].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        dbp = frame[DIASTOLIC].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    elif reading == "first":
        sbp, dbp = first_recorded(frame, SYSTOLIC), first_recorded(frame, DIASTOLIC)
    else:
        raise ValueError(f"reading is mean or first, not {reading!r}")
    told = _is(frame["BPQ020"], YES)
    rx = _is(frame[medicine], YES)
    high = sbp.ge(SBP_HIGH).fillna(False) | dbp.ge(DBP_HIGH).fillna(False)
    return codes((told | rx | high).astype(int))


def family_history(frame: pd.DataFrame, close: bool = False) -> pd.Series:
    """1 for a blood relative told of diabetes (``MCQ250A`` 1), or with ``close`` a parent or a
    sibling among the relatives named (:data:`CLOSE_RELATIVES`); else 0."""
    if not close:
        return codes(_is(frame["MCQ250A"], YES).astype(int))
    hit = np.zeros(len(frame), dtype=bool)
    for column, code in CLOSE_RELATIVES.items():
        hit |= _is(frame[column], code).to_numpy()
    return codes(hit.astype(int))


def derive(table: pd.DataFrame) -> pd.DataFrame:
    """The derived table: :data:`KEPT` as the intake has them (the outcome as an integer) and every
    predictor column of :data:`PREDICTORS`, row for row."""
    out = table[KEPT].copy()
    out["diabetes_fpg"] = codes(out["diabetes_fpg"])
    out["age_band"] = age_band(table["RIDAGEYR"])
    sex = pd.to_numeric(table["RIAGENDR"], errors="coerce")
    out["male"] = codes(sex.eq(MALE).astype(int).where(sex.notna()))
    out["famhx"] = family_history(table)
    out["famhx_close"] = family_history(table, close=True)
    out["htn"] = hypertension(table)
    out["htn_first"] = hypertension(table, reading="first")
    out["htn_told_rx"] = hypertension(table, medicine="BPQ040A")
    out["obesity"] = obesity(table)
    out["obesity_absent"] = obesity(table, missing_as_absent=True)
    out["active"] = codes(_is(table["PAQ520"], MORE_ACTIVE).astype(int))
    return out


FOUR_YEAR_CYCLES = (1, 2)  # SDDSRVYR: 1999-2000 and 2001-2002, the cycles WTSAF4YR spans


def subsample(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the fasting subsample (``WTSAF2YR`` recorded); refused where a 1999–2002 row
    there lacks ``WTSAF4YR``, or a stratum loses a PSU the whole table has."""
    kept = table[pd.to_numeric(table["WTSAF2YR"], errors="coerce").notna()]
    early = pd.to_numeric(kept["SDDSRVYR"], errors="coerce").isin(FOUR_YEAR_CYCLES)
    lacking = int((early & pd.to_numeric(kept["WTSAF4YR"], errors="coerce").isna()).sum())
    if lacking:
        raise ValueError(f"{lacking} rows of the 1999-2002 fasting subsample lack WTSAF4YR")

    def psus(frame: pd.DataFrame) -> dict[Any, set[Any]]:
        return {s: set(g["SDMVPSU"].dropna()) for s, g in frame.groupby("SDMVSTRA")}

    if psus(kept) != psus(table):
        raise ValueError("the fasting subsample does not keep every stratum's PSUs")
    return kept


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def intake_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    """The intake's table, refused unless its bytes are the manifest's."""
    entry = json.loads(MANIFEST.read_text("utf-8"))["table"]
    path = REPO / entry["file"]
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise SystemExit(f"{entry['file']} is not the manifest's table: run "
                         f"python -m research.swarm.intake --check")
    return pd.read_parquet(path), entry


def main() -> int:
    table, entry = intake_table()
    out = derive(subsample(table))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DERIVED, index=False)
    log = {"from": entry, "rows": int(len(out)), "rows_outside_the_subsample": int(len(table) - len(out)),
           "columns": list(out.columns),
           "derived": {"file": str(DERIVED.relative_to(REPO)), "sha256": sha256(DERIVED)},
           "levels": {c: {str(k): int(v) for k, v in out[c].value_counts(dropna=False).items()}
                      for c in PREDICTORS}}
    (OUT / "prep_log.json").write_text(json.dumps(log, indent=1) + "\n", "utf-8")
    print(json.dumps({k: log[k] for k in ("rows", "derived")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
