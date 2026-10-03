"""Clinical plausibility: values no living patient shows, and values outside a reference sample's
central 98% (audit IN-09, MI-04; CLINICAL_SURVEY_PACK §A1.2).

``ml.physiology_reference`` (Classic's, not edited) held one band table labeled "NHANES
(reference population, demo defaults)" whose numbers no NHANES file produced (on the real NHANES
export 9.1% of triglycerides fell outside its "p01/p99"), applied adult limits to children, put
total energy intake among the physiological variables (nine 0–93 kcal recalls became
"physiologically impossible … entry errors"), and called a diastolic pressure of 0 an entry error
although NHANES documents it. The findings stage now serves this module's reading:

* **Bands carry their source, cycle and status** (``data/plausibility.json``). The *improbable*
  tier is the weighted 1st–99th percentile of adults aged 20 and over in NHANES 2017–2018
  (cycle J), computed by ``data/build_plausibility.py`` from the public files; it is described
  as "outside the central 98% of NHANES 2017–2018 adults", never as "abnormal". The *impossible*
  tier is CLINICAL_SURVEY_PACK §A1.2's suggested EHR-cleaning limits, which the pack marks
  CONVENTION ("plausibility limits are institution- and observation-specific"), or TurboTab's own
  where the pack gives none, and says so. Every band is CONVENTION.
* **Total energy is not a physiological variable.** A very low recall is a dietary report, judged
  by the dietary pack's intake screens.
* **Children are not judged by adult limits** (§A1.2: "Pediatric and growth data: never apply adult
  bounds"). With an age column, rows under 20 leave the adult percentile tier; their weight,
  height and BMI are judged by the CDC 2000 growth charts' modified z-scores (2 to 20 years,
  sex-specific LMS tables in ``data/``, as published at
  ``https://www.cdc.gov/growthcharts/data/zscore/{wtage,statage,bmiagerev}.csv``): "computed by
  extrapolating one-half of the distance between 0 and +2 (or between 0 and -2) z-scores to the
  distribution's tails", flagged below −5
  or above 8 for weight-for-age, below −5 or above 4 for height-for-age, below −4 or above 8 for
  BMI-for-age (CDC, SAS Program for CDC Growth Charts, Table 2). Without a sex column a child is
  flagged only when both sexes' charts flag the value. A table that holds children judges body
  size by any-age limits for the repair, since the adult floor (20 kg) would blank real children.
* **A diastolic pressure of 0** is described per the NHANES documentation ("Diastolic BP can be
  zero", BPX_J, Data Processing and Editing): a recorded reading, not an entry error; setting it
  to missing is the usual convention, and the repair offers that.

Units are read by Classic's reader (``ml.clinical_units.infer_unit``) and a column whose reading
is doubtful as a whole (``ml.card_evidence.interpretation_verdict``: wrong units, a derived
name) is set aside rather than flagged, as before.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
ADULT_AGE = 20.0
CHILD_MIN_AGE = 2.0
BODY_SIZE = ("weight", "height", "bmi")
#: CDC SAS Program for CDC Growth Charts, Table 2: modified z-score cut-offs for extreme values.
BIV_CUTOFFS = {"weight": (-5.0, 8.0), "height": (-5.0, 4.0), "bmi": (-4.0, 8.0)}
LMS_FILES = {"weight": "wtage.csv", "height": "statage.csv", "bmi": "bmiagerev.csv"}
CDC_SOURCE = ("CDC, SAS Program for CDC Growth Charts (2000 growth charts, 2 to 20 years), "
              "Table 2: modified z-scores below -5 or above 8 (weight-for-age), below -5 or "
              "above 4 (height-for-age), below -4 or above 8 (BMI-for-age)")

_AGE_NAMES = {"age", "ridageyr", "age_years", "age_yrs", "ageyears", "age_yr", "age_y", "agey"}
_MONTH_NAMES = {"agemos", "age_months", "age_mo", "ridagemn", "ridexagm", "age_in_months"}
_SEX_NAMES = {"sex", "gender", "riagendr", "male", "female", "is_male", "is_female", "sex_male"}


@lru_cache(maxsize=1)
def reference() -> dict[str, Any]:
    """The band table (``data/plausibility.json``)."""
    return json.loads((DATA / "plausibility.json").read_text("utf-8"))


@lru_cache(maxsize=3)
def _lms(table: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / LMS_FILES[table])
    frame = frame[pd.to_numeric(frame["Sex"], errors="coerce").isin([1, 2])]
    return frame[["Sex", "Agemos", "L", "M", "S"]].astype(float).reset_index(drop=True)


def _normal(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())


def variable_of(column: str) -> str | None:
    """The band variable a column is, by exact key or declared alias (never by substring)."""
    target = _normal(column)
    if not target:
        return None
    for key, spec in reference()["variables"].items():
        if any(_normal(n) == target for n in [key, *spec.get("aliases", [])]):
            return key
    return None


# ── age and sex ──────────────────────────────────────────────────────────────


def age_in_months(df: pd.DataFrame) -> pd.Series | None:
    """Each row's age in months, from a months column or a years column (whole years read at
    mid-year), or None when the table has neither."""
    for c in df.columns:
        if _normal(c) in {_normal(n) for n in _MONTH_NAMES} and pd.api.types.is_numeric_dtype(df[c]):
            months = pd.to_numeric(df[c], errors="coerce")
            if months.notna().any() and months.dropna().between(0, 1500).all():
                return months
    for c in df.columns:
        if _normal(c) in {_normal(n) for n in _AGE_NAMES} and pd.api.types.is_numeric_dtype(df[c]):
            years = pd.to_numeric(df[c], errors="coerce")
            if not years.notna().any() or not years.dropna().between(0, 125).all():
                continue
            whole = bool(np.all(np.mod(years.dropna().to_numpy(dtype=float), 1) == 0))
            return years * 12.0 + (6.0 if whole else 0.0)
    return None


def sex_codes(df: pd.DataFrame) -> pd.Series | None:
    """1 (male) or 2 (female) per row, the CDC tables' coding, or None without a sex column."""
    for c in df.columns:
        if _normal(c) not in {_normal(n) for n in _SEX_NAMES}:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = set(pd.to_numeric(s, errors="coerce").dropna().unique())
            if values <= {1.0, 2.0} and values:
                return pd.to_numeric(s, errors="coerce")
            continue
        text = s.astype("string").str.strip().str.lower()
        mapped = text.map({"male": 1.0, "m": 1.0, "man": 1.0, "boy": 1.0,
                           "female": 2.0, "f": 2.0, "woman": 2.0, "girl": 2.0})
        if mapped.notna().sum() >= 0.9 * text.notna().sum() and mapped.notna().any():
            return mapped.astype(float)
    return None


def modified_z(values: pd.Series, months: pd.Series, sex: pd.Series | float, table: str) -> pd.Series:
    """The CDC modified z-score of each value for its age (months) and sex (1 or 2).

    LMS by linear interpolation between the table's half-month ages; X_{±2} = M (1 ± 2 L S)^{1/L};
    the modified z is (X − M) / ((X_{+2} − M) / 2) above the median and (X − M) / ((M − X_{−2}) / 2)
    below it. NaN outside 24–240 months.
    """
    lms = _lms(table)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    sex_s = sex if isinstance(sex, pd.Series) else pd.Series(float(sex), index=values.index)
    for code in (1.0, 2.0):
        part = lms[lms["Sex"] == code].sort_values("Agemos")
        rows = (sex_s == code) & months.between(part["Agemos"].min(), part["Agemos"].max()) \
            & values.notna()
        if not rows.any():
            continue
        age = months[rows].to_numpy(dtype=float)
        L = np.interp(age, part["Agemos"], part["L"])
        M = np.interp(age, part["Agemos"], part["M"])
        S = np.interp(age, part["Agemos"], part["S"])
        x = values[rows].to_numpy(dtype=float)
        plus2 = M * (1 + 2 * L * S) ** (1 / L)
        minus2 = M * (1 - 2 * L * S) ** (1 / L)
        z = np.where(x >= M, (x - M) / ((plus2 - M) / 2), (x - M) / ((M - minus2) / 2))
        out[rows] = z
    return out


def _biv(values: pd.Series, months: pd.Series, sex: pd.Series | None, var: str) -> pd.Series:
    """Whether each child's value is a CDC biologically implausible value (both sexes' charts
    must flag it when the sex is not known)."""
    lo, hi = BIV_CUTOFFS[var]
    table = var

    def flags(code: pd.Series | float) -> pd.Series:
        z = modified_z(values, months, code, table)
        return (z < lo) | (z > hi)

    if sex is None:
        return flags(1.0) & flags(2.0)
    known = sex.isin([1.0, 2.0])
    out = flags(sex.where(known, 1.0)) & known
    unknown = ~known
    if unknown.any():
        out |= unknown & flags(1.0) & flags(2.0)
    return out


# ── the reading ──────────────────────────────────────────────────────────────


def _is_zero(x: pd.Series) -> pd.Series:
    from turbotab.core.repairs import is_sas_zero

    arr = x.to_numpy(dtype=float)
    return pd.Series((arr == 0) | is_sas_zero(arr), index=x.index)


def read(df: pd.DataFrame) -> dict[str, Any]:
    """Per recognized column: the impossible values, the values outside the reference sample's
    central 98% (adult rows), the children's growth-chart flags, and what was set aside."""
    from ml.card_evidence import READING_ENTRIES, interpretation_verdict
    from ml.clinical_units import infer_unit

    ref = reference()
    months = age_in_months(df)
    sex = sex_codes(df)
    adult = months >= ADULT_AGE * 12 if months is not None else pd.Series(True, index=df.index)
    child = ((months >= CHILD_MIN_AGE * 12) & (months < ADULT_AGE * 12)) if months is not None \
        else pd.Series(False, index=df.index)
    has_children = bool(child.any())
    columns: list[dict[str, Any]] = []
    set_aside: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame) or not pd.api.types.is_numeric_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            continue
        var = variable_of(str(col))
        if var is None:
            continue
        spec = ref["variables"][var]
        present = pd.to_numeric(s, errors="coerce").dropna()
        if present.empty:
            continue
        unit = infer_unit(str(col), present) or {}
        factor = unit.get("conversion_factor")
        if not factor:
            set_aside.append({"column": str(col), "variable": var,
                              "why": "its unit could not be read"})
            continue
        x = present * float(factor)
        improbable = spec["improbable"]
        p01, p99 = float(improbable["p01"]), float(improbable["p99"])
        tier = "any_age" if (var in BODY_SIZE and has_children) else "impossible"
        band = spec.get(tier) or spec["impossible"]
        low, high = float(band["low"]), float(band["high"])
        if var in BODY_SIZE and months is None and float(x.median()) < p01:
            set_aside.append({
                "column": str(col), "variable": var,
                "why": (f"its values sit below adults' (median {float(x.median()):g} against the "
                        f"adult 1st percentile {p01:g} {spec['unit']}) and there is no age column, "
                        f"so children cannot be told from errors; children are judged by "
                        f"age-specific z-scores, never by adult limits")})
            band = spec.get("any_age") or band
            low, high = float(band["low"]), float(band["high"])
            tier = "any_age"
        impossible = x[(x < low) | (x > high)]
        ifcc = None
        if var == "hba1c":
            # IN-09: an HbA1c above 30% that the NGSP master equation (NGSP = 0.09148 × IFCC +
            # 2.152, ngsp.org) reads as a plausible percentage is a unit question (IFCC mmol/mol),
            # not an impossible value.
            converted = 0.09148 * impossible + 2.152
            ifcc = impossible[(impossible > high) & (converted >= low) & (converted <= high)]
            impossible = impossible.drop(ifcc.index)
        verdict = interpretation_verdict(str(col), var, x, low, high, impossible,
                                         reference=(p01, p99))
        if verdict.reading != READING_ENTRIES:
            set_aside.append({"column": str(col), "variable": var,
                              "why": verdict.statement or f"the reading is {verdict.reading}"})
            continue
        adult_x = x[adult.reindex(x.index, fill_value=False)]
        possible = adult_x[(adult_x >= low) & (adult_x <= high)]
        outside_98 = possible[(possible < p01) | (possible > p99)]
        entry: dict[str, Any] = {
            "column": str(col), "variable": var, "unit": spec["unit"],
            "n_present": int(len(x)),
            "n_impossible": int(len(impossible)), "impossible_band": [low, high],
            "impossible_source": band["source"], "impossible_tier": tier,
            "n_outside_central_98": int(len(outside_98)),
            # The key the repair family and the voice read; the same count.
            "n_abnormal_but_possible": int(len(outside_98)),
            "normal_band": [p01, p99], "central_98_band": [p01, p99],
            "reference_sample": {k: improbable.get(k) for k in (
                "source", "cycle", "population", "nhanes_file", "nhanes_variable", "weight", "n",
                "status")},
            "status": "CONVENTION",
            "n_adult_rows_read": int(len(adult_x)),
        }
        if ifcc is not None and len(ifcc):
            entry["unit_question"] = {
                "n": int(len(ifcc)), "unit": "mmol/mol (IFCC)",
                "as_percent": [round(float(v), 1) for v in (0.09148 * ifcc + 2.152).head(5)],
                "source": "NGSP master equation: NGSP = [0.09148 * IFCC] + 2.152 (ngsp.org)"}
        if spec.get("zero"):
            zeros = _is_zero(present)
            entry["n_zero"] = int(zeros.sum())
            entry["zero_source"] = spec["zero"]
        if var in BODY_SIZE and has_children:
            kids = x[child.reindex(x.index, fill_value=False)]
            flags = _biv(kids, months.reindex(kids.index), sex.reindex(kids.index)
                         if sex is not None else None, var)
            entry["children"] = {"n_read": int(len(kids)), "n_flagged": int(flags.sum()),
                                 "rows": [int(i) if isinstance(i, (int, np.integer)) else str(i)
                                          for i in flags[flags].index[:50]],
                                 "cutoffs": list(BIV_CUTOFFS[var]),
                                 "sex_known": sex is not None, "source": CDC_SOURCE}
        columns.append(entry)
    return {"columns": columns, "set_aside": set_aside, "has_age": months is not None,
            "n_children": int(child.sum()), "n_adults_read": int(adult.sum()),
            "version": ref["version"], "status": ref["status"]}


def _num(v: float) -> str:
    return f"{v:g}"


def impossible_vs_extreme_finding(df: pd.DataFrame) -> dict[str, Any] | None:
    """``pack::clinical::impossible_vs_extreme``: impossible values beside values that are only
    unusual, from sourced bands, adults and children each judged by their own rule."""
    from turbotab.clinical import CLINICAL, IMPOSSIBLE_VS_EXTREME_EVIDENCE
    from turbotab.packs import _finding

    r = read(df)
    flagged = [e for e in r["columns"] if e["n_impossible"] or e.get("unit_question")
               or (e.get("children") or {}).get("n_flagged")]
    if not flagged:
        return None
    flagged.sort(key=lambda e: (-e["n_impossible"], -e["n_outside_central_98"]))
    lead = flagged[0]
    unit = f" {lead['unit']}" if lead["unit"] else ""
    lo, hi = lead["impossible_band"]
    p01, p99 = lead["central_98_band"]
    sample = lead["reference_sample"]
    parts = []
    if lead["n_impossible"]:
        documented = int(lead.get("n_zero") or 0) >= lead["n_impossible"]
        parts.append(
            f"{lead['n_impossible']:,} value{'' if lead['n_impossible'] == 1 else 's'} in "
            f"`{lead['column']}` {'is' if lead['n_impossible'] == 1 else 'are'} outside "
            f"{_num(lo)}–{_num(hi)}{unit}, limits no living "
            f"{'person' if lead['impossible_tier'] == 'any_age' else 'adult'} is expected to show "
            f"(a convention: {lead['impossible_source']})."
            + ("" if documented else " Recording errors are the usual reason; setting them to "
                                     "missing and reporting the count is the usual treatment."))
    for e in flagged:
        if e.get("n_zero"):
            parts.append(
                f"{e['n_zero']:,} of `{e['column']}`'s values {'is' if e['n_zero'] == 1 else 'are'} "
                f"0. NHANES records a diastolic pressure of 0 on purpose ({e['zero_source']}): it "
                f"is a reading, not an entry error, and it is conventionally set to missing.")
    if lead["n_outside_central_98"]:
        parts.append(
            f"**This is different from the {lead['n_outside_central_98']:,} adult values outside "
            f"{_num(p01)}–{_num(p99)}{unit}, the central 98% of {sample.get('population')} in "
            f"NHANES {sample.get('cycle')} (weighted 1st and 99th percentiles, "
            f"{sample.get('nhanes_variable')}): unusual but real, and they must be kept**, because "
            f"excluding them would remove the sickest patients.")
    for e in flagged:
        unit_q = e.get("unit_question")
        if unit_q:
            parts.append(
                f"{unit_q['n']:,} of `{e['column']}`'s values are above 30% but read as HbA1c in "
                f"{unit_q['unit']}: by the NGSP master equation they are "
                f"{', '.join(f'{v:g}%' for v in unit_q['as_percent'][:3])}. That is a question "
                f"about units, not impossible values, so no repair is offered for this column.")
    kids = lead.get("children")
    if kids:
        parts.append(
            f"{kids['n_read']:,} rows are children aged 2 to 19: adult limits are not applied to "
            f"them. By the CDC growth charts' modified z-scores (below {_num(kids['cutoffs'][0])} "
            f"or above {_num(kids['cutoffs'][1])}), {kids['n_flagged']:,} "
            f"{'is' if kids['n_flagged'] == 1 else 'are'} biologically implausible"
            + ("" if kids["sex_known"] else ", counted only where both sexes' charts agree, since "
                                            "there is no sex column") + ".")
    if len(flagged) > 1:
        parts.append("The same reading applies to " + ", ".join(
            f"`{e['column']}`" for e in flagged if e is not lead) + ".")
    if r["set_aside"]:
        parts.append("Set aside rather than judged: " + "; ".join(
            f"`{a['column']}` ({a['why']})" for a in r["set_aside"][:3]) + ".")
    if lead["n_impossible"]:
        title = (f"`{lead['column']}` holds impossible values and unusual ones, and they are "
                 f"different categories")
    elif lead.get("unit_question"):
        title = f"`{lead['column']}` holds values in another unit"
    else:
        title = f"`{lead['column']}` holds children's values the growth charts call implausible"
    return _finding(
        "pack::clinical::impossible_vs_extreme", "warning", title,
        " ".join(parts),
        ("Physiologically impossible and statistically extreme are different categories, and no "
         "generic outlier rule can tell them apart. The impossible values are a data-quality "
         "repair with a count that belongs in the paper; the unusual ones are the case mix the "
         "model exists to learn. Every limit here is a convention, with its source."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=IMPOSSIBLE_VS_EXTREME_EVIDENCE,
        columns=[e["column"] for e in flagged],
        # ``columns`` are the entries the repair acts on (impossible values, a column-wide band);
        # children's growth-chart flags are per row and per sex, so no column-wide band blanks them.
        params={"columns": [e for e in flagged if e["n_impossible"] and not e.get("unit_question")],
                "children": [{"column": e["column"], **e["children"]} for e in flagged
                             if e.get("children")],
                "reference_version": r["version"], "status": r["status"],
                "has_age": r["has_age"], "n_children": r["n_children"],
                "columns_the_core_could_not_read": [a["column"] for a in r["set_aside"]],
                "set_aside": r["set_aside"]},
        fix_label="", fix_kind="none")


__all__ = ["BIV_CUTOFFS", "age_in_months", "impossible_vs_extreme_finding", "modified_z", "read",
           "reference", "sex_codes", "variable_of"]
