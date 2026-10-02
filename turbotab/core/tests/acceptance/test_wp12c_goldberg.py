"""WP12 acceptance test 5: the Goldberg screen and the primary-plus-sensitivity answer
(docs/turbotab-next/audit/AUDIT_REPORT.md §5, WP12 · Methods a reviewer expects).

    5. "Goldberg cut-offs with Black 2000 parameters reproduce published thresholds for a stated PAL
       and d; a "primary plus sensitivity" answer renders the estimate under each exclusion rule.
       Source check: Banna et al. 2017."

It closes ME-16 ("No misreporting screen beyond fixed kcal cut-offs, and no with/without-exclusion
sensitivity analysis").

**Published thresholds.** Black 2000 (*Int J Obes* 24:1119), abstract (PubMed 11033980, read
2026-10-02): "The suggested value for average within-subject variation in energy intake is 23%
(unchanged) … For within-subject variation in measured and estimated BMR, 4% and 8.5% respectively
are suggested (previously 2.5% and 8%), and for total between-subject variation in PAL, the
suggested value is 15% (previously 12.5%)." The thresholds those parameters give are published in
the EFSA EU Menu guidance, Appendix 8.2.1 (EFSA 2014, "Guidance on the EU Menu methodology", the
appendix on under- and over-reporting), which applies "The revised factors by Black … CVwEI = 23%;
CVwB = 8.5%; CVtP=15%" with "d is the number of days of diet assessment (in our study it is 2)" and
prints, in its Table 4, "the lower and upper cut-off for each PAL value for n=1":

    PAL 1.4 → 0.872, 2.249 · PAL 1.6 → 0.996, 2.570 · PAL 1.8 → 1.120, 2.892 · PAL 2.0 → 1.245, 3.213

and, for its group example ("n=30 and apply PAL=1.8"), "the calculated specific lower cut-off
(1.65)". The same appendix prints the Schofield equations in kcal/d ("Schofield equations for
estimating BMR (kcal/d) from weight (kg) and height (m)" and "from weight (kg)"), which are the
independent path for the BMR the screen uses (the engine holds Schofield's own MJ/day
coefficients). Henry 2005 (*Public Health Nutr* 8:1133) prints each Oxford equation in MJ and in
kcal (Tables 12 and 15) and the mean weight and height of each group (Table 16) beside the mean
BMR the equations predict for it (Table 17), a second published path.

**The sensitivity answer.** Banna et al. 2017 (*Front Nutr* 4:45), the source check, read in full:
"Regardless of which method is used, for the time being, analyses in the total sample without
exclusion of participants should also be conducted and reported. This will allow the researcher to
examine the impact of using each method, and will also address the potential bias that may be
introduced when participants are excluded from the analysis." And on cut-offs that read body
weight: "body weight is included in both the calculation of implausible rEI and the outcome
variable in subsequent analyses of how dietary variables are associated with the outcome (13),
which could artificially elevate the association." Driven through the real server, the stage fits
the primary (Goldberg-screened), a fixed 500–3,500 kcal screen, and the every-row analysis it adds
itself; each analysis's rows are recounted longhand (EFSA's kcal equations and EFSA's printed
cut-offs) and each estimate refit by NumPy least squares with HC3 written out by definition
(``references.hc3_by_definition``), none of which imports the engine.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.methods.misreporting import (CV_TP, CV_WB_ESTIMATED, CV_WEI, EQUATIONS,
                                                bmr_kcal, goldberg_cutoffs, screen)
from turbotab.core.tests.acceptance.references import hc3_by_definition
from turbotab.core.tests.acceptance.server_drive import local_server, open_project

# ── the published numbers ────────────────────────────────────────────────────

# EFSA 2014, EU Menu guidance Appendix 8.2.1, Table 4 (n = 1, d = 2, Black's factors).
EFSA_TABLE_4 = {1.4: (0.872, 2.249), 1.6: (0.996, 2.570), 1.8: (1.120, 2.892), 2.0: (1.245, 3.213)}
EFSA_GROUP_EXAMPLE = {"pal": 1.8, "n": 30, "days": 2, "lower": 1.65}

# EFSA 2014 Appendix 8.2.1, "Schofield equations for estimating BMR (kcal/d)": (age band,
# weight, height in m, constant) as printed; age bands 10-17, 18-29, 30-59, 60+.
EFSA_SCHOFIELD_WH = {
    "male": [((10, 18), "16.2", "137", "516"), ((18, 30), "15.0", "-10", "706"),
             ((30, 60), "11.5", "-2.6", "877"), ((60, 120), "9.1", "972", "-834")],
    "female": [((10, 18), "8.4", "466", "200"), ((18, 30), "13.6", "283", "98"),
               ((30, 60), "8.1", "1.4", "844"), ((60, 120), "7.9", "458", "17.7")],
}
EFSA_SCHOFIELD_W = {
    "male": [((10, 18), "17.7", "658.2"), ((18, 30), "15.0", "692.1"), ((30, 60), "11.5", "873.0"),
             ((60, 120), "11.7", "587.7")],
    "female": [((10, 18), "13.4", "692.6"), ((18, 30), "14.8", "486.6"), ((30, 60), "8.1", "845.6"),
               ((60, 120), "9.1", "658.4")],
}
# Henry 2005, Table 12 (weight) and Table 15 (weight and height), the kcal/day columns as printed,
# and Table 16's group means (age, height m, weight kg) beside Table 17's mean predicted BMR (MJ):
# (weight alone, weight + height).
HENRY_KCAL_W = {
    "male": [((3, 10), "23.3", "514"), ((10, 18), "18.4", "581"), ((18, 30), "16.0", "545"),
             ((30, 60), "14.2", "593"), ((60, 120), "13.5", "514")],
    "female": [((3, 10), "20.1", "507"), ((10, 18), "11.1", "761"), ((18, 30), "13.1", "558"),
               ((30, 60), "9.74", "694"), ((60, 120), "10.1", "569")],
}
HENRY_KCAL_WH = {
    "male": [((3, 10), "15.1", "74.2", "306"), ((10, 18), "15.6", "266", "299"),
             ((18, 30), "14.4", "313", "113"), ((30, 60), "11.4", "541", "-137"),
             ((60, 120), "11.4", "541", "-256")],
    "female": [((3, 10), "15.9", "210", "349"), ((10, 18), "9.40", "249", "462"),
               ((18, 30), "10.4", "615", "-282"), ((30, 60), "8.18", "502", "-11.6"),
               ((60, 120), "8.52", "421", "10.7")],
}
HENRY_GROUPS = {  # (mean age, height m, weight kg) -> (BMR weight alone, BMR weight + height), MJ
    "male": [((6.6, 1.17, 21.4), (4.168, 4.168)), ((12.7, 1.49, 40.0), (5.506, 5.505)),
             ((22.7, 1.70, 61.0), (6.364, 6.366)), ((40.8, 1.69, 65.3), (6.347, 6.349)),
             ((70.9, 1.70, 71.3), (6.173, 6.178))],
    "female": [((7.1, 1.22, 23.6), (4.100, 4.096)), ((13.0, 1.50, 43.4), (5.202, 5.199)),
               ((22.4, 1.60, 53.2), (5.239, 5.232)), ((41.6, 1.59, 59.1), (5.306, 5.307)),
               ((69.8, 1.56, 60.0), (4.931, 4.934))],
}
# Two kcal figures Henry prints for the boys aged 3–10 contradict the MJ figure beside them: Table
# 12's "23.3W" beside "0.0937W" (0.0937 MJ is 22.4 kcal) and Table 15's "74.2H" beside "1.31H"
# (1.31 MJ is 313 kcal). The MJ figures are the ones Table 13 repeats and Table 17 confirms: at
# Table 16's means (21.4 kg, 1.17 m) they give 4.155 and 4.165 MJ against Table 17's 4.168, where
# the kcal equations give 4.24 and 2.99 MJ. (key, sex, band low, coefficient position).
HENRY_KCAL_MISPRINTS = {("henry", "male", 3, 0), ("henry_height", "male", 3, 1)}
KCAL_PER_MJ = 239.005736  # 1 / 0.004184: the thermochemical calorie, written out here


def _unit(printed: str) -> float:
    """One unit of the last printed digit: "16.2" → 0.1, "137" → 1, "17.7" → 0.1."""
    return 10.0 ** -(len(printed.split(".")[1]) if "." in printed else 0)


def _agree(mj: float, kcal_text: str) -> bool:
    """An MJ/day coefficient and the kcal/day one printed for it agree: within one unit of the kcal
    figure's last digit plus half a unit of the MJ figure's, converted (both are rounded)."""
    return abs(mj * KCAL_PER_MJ - float(kcal_text)) <= (_unit(kcal_text)
                                                        + KCAL_PER_MJ * _unit(repr(mj)) / 2 + 1e-9)


# ── 5a · the cut-offs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("pal", sorted(EFSA_TABLE_4))
def test_5_black_cutoffs_reproduce_efsa_table_4(pal):
    """EFSA 2014 Table 4: for n = 1 and d = 2 recall days, Black's factors give these cut-offs to
    the three decimals printed (PAL 1.4, 1.6, 1.8 and 2.0).

    Seven of the eight printed figures agree to half a unit of the third decimal. The eighth,
    0.872 for PAL 1.4, is 1.4 × exp(−2 × 0.2370) = 0.87149, which rounds to 0.871 (0.8715 rounded
    again gives EFSA's figure). The bound is therefore 0.0006, half a printed unit plus that
    discrepancy; the cut-offs Goldberg's 1991 factors give (BMR 8%, PAL 12.5%, the values Black
    revised) differ from every printed figure by more than 0.02, so the bound tells them apart.
    """
    assert (CV_WEI, CV_WB_ESTIMATED, CV_TP) == (23.0, 8.5, 15.0)  # Black 2000's values, as quoted
    lower, upper = goldberg_cutoffs(pal, days=2, n=1)
    printed_lower, printed_upper = EFSA_TABLE_4[pal]
    assert abs(lower - printed_lower) <= 0.0006, (pal, lower, printed_lower)
    assert abs(upper - printed_upper) <= 0.0006, (pal, upper, printed_upper)
    old_lower, old_upper = goldberg_cutoffs(pal, days=2, n=1, cv_wb=8.0, cv_tp=12.5)
    assert abs(old_lower - printed_lower) > 0.02 and abs(old_upper - printed_upper) > 0.02


def test_5_the_group_cutoff_reproduces_efsa_example_a():
    """EFSA 2014 example A: "n=30 and apply PAL=1.8" gives "the calculated specific lower cut-off
    (1.65)"; the group's cut-offs narrow with √n as Black's equation says."""
    ex = EFSA_GROUP_EXAMPLE
    lower, upper = goldberg_cutoffs(ex["pal"], days=ex["days"], n=ex["n"])
    assert round(lower, 2) == ex["lower"]
    one, _ = goldberg_cutoffs(ex["pal"], days=ex["days"], n=1)
    assert math.log(ex["pal"] / lower) == pytest.approx(math.log(ex["pal"] / one) / math.sqrt(30))
    # More recall days narrow the individual's limits too (S falls with d), never below the
    # BMR and PAL terms: d = 2 is EFSA's, d = 7 a week of records.
    week, _ = goldberg_cutoffs(ex["pal"], days=7)
    assert one < week < ex["pal"] * math.exp(-2 * math.hypot(8.5, 15) / 100) + 1e-12


# ── 5b · the BMR equations ───────────────────────────────────────────────────


@pytest.mark.parametrize("sex", ["male", "female"])
def test_5_schofield_equations_agree_with_efsa_kcal_tables(sex):
    """The engine's Schofield MJ/day coefficients, times 1000/4.184, agree with EFSA's printed kcal/d
    coefficients band by band (within the rounding of both printed figures, ``_agree``), and the
    BMR the screen computes agrees with EFSA's equations to within 0.4% over a grid of bodies."""
    for key, efsa, height in (("schofield_height", EFSA_SCHOFIELD_WH, True),
                              ("schofield", EFSA_SCHOFIELD_W, False)):
        bands = EQUATIONS[key].bands[sex]
        assert len(bands) == len(efsa[sex])
        for band, printed in zip(bands, efsa[sex]):
            (lo, hi), *coefs = printed
            assert (band.low, band.high or 120) == (lo, hi)
            mine = [band.weight, band.height, band.constant] if height else [band.weight, band.constant]
            for value, text in zip(mine, coefs):
                assert _agree(value, text), (key, sex, lo, value, text)
        # The BMR on a grid of ages, weights (kg) and heights (cm) against EFSA's kcal equations.
        ages, weights, heights = np.meshgrid([12, 25, 45, 70], [40, 60, 80, 110], [150, 170, 190])
        ages, weights, heights = ages.ravel(), weights.ravel(), heights.ravel()
        got = bmr_kcal(key, np.full(ages.size, sex, dtype=object), ages, weights,
                       heights if height else None, height_unit="cm")
        want = np.empty(ages.size)
        for i, (a, w, h) in enumerate(zip(ages, weights, heights / 100)):
            (_, *c) = next(p for p in efsa[sex] if p[0][0] <= a < p[0][1])
            c = [float(x) for x in c]
            want[i] = c[0] * w + c[1] * h + c[2] if height else c[0] * w + c[1]
        assert np.max(np.abs(got / want - 1)) < 0.004, key


@pytest.mark.parametrize("sex", ["male", "female"])
def test_5_henry_equations_reproduce_henrys_kcal_columns_and_group_means(sex):
    """Henry 2005: each Oxford equation in MJ/day (the engine's) matches the kcal/day column Henry
    prints beside it, within the rounding of both figures (ages 3 and over: the screen's range),
    except Henry's two misprinted kcal figures (``HENRY_KCAL_MISPRINTS``), which are asserted to
    contradict Table 17 where the MJ figures agree with it; and at Table 16's mean weight and
    height of each group the equations give Table 17's mean predicted BMR to within 0.02 MJ (the
    means are printed to 0.1 kg and 0.01 m)."""
    for key, printed, height in (("henry", HENRY_KCAL_W, False), ("henry_height", HENRY_KCAL_WH, True)):
        bands = [b for b in EQUATIONS[key].bands[sex] if b.low >= 3]
        for band, row in zip(bands, printed[sex]):
            (lo, hi), *coefs = row
            assert (band.low, band.high or 120) == (lo, hi)
            mine = [band.weight, band.height, band.constant] if height else [band.weight, band.constant]
            for i, (value, text) in enumerate(zip(mine, coefs)):
                misprint = (key, sex, lo, i) in HENRY_KCAL_MISPRINTS
                assert _agree(value, text) is not misprint, (key, sex, lo, value, text)
    if sex == "male":  # the misprints, against Table 17's 4.168 MJ at 21.4 kg and 1.17 m
        kcal_alone = (23.3 * 21.4 + 514) / KCAL_PER_MJ
        kcal_both = (15.1 * 21.4 + 74.2 * 1.17 + 306) / KCAL_PER_MJ
        assert abs(kcal_alone - 4.168) > 0.05 and abs(kcal_both - 4.168) > 1.0
    for (age, h, w), (alone, both) in HENRY_GROUPS[sex]:
        got_alone = bmr_kcal("henry", np.array([sex], dtype=object), [age], [w])[0] / KCAL_PER_MJ
        got_both = bmr_kcal("henry_height", np.array([sex], dtype=object), [age], [w], [h],
                            height_unit="m")[0] / KCAL_PER_MJ
        assert abs(got_alone - alone) < 0.02, (age, got_alone, alone)
        assert abs(got_both - both) < 0.02, (age, got_both, both)


def test_5_mifflin_is_the_published_equation():
    """Mifflin et al. 1990 (*Am J Clin Nutr* 51:241), abstract: "REE (males) = 10 x weight (kg) +
    6.25 x height (cm) - 5 x age (y) + 5; REE (females) = 10 x weight (kg) + 6.25 x height (cm) -
    5 x age (y) - 161"."""
    sex = np.array(["male", "female", "male", "female"], dtype=object)
    age, w, h = np.array([25.0, 40.0, 70.0, 19.0]), np.array([70.0, 62.0, 85.0, 50.0]), np.array([180.0, 165.0, 172.0, 158.0])
    want = 10 * w + 6.25 * h - 5 * age + np.where(sex == "male", 5, -161)
    np.testing.assert_allclose(bmr_kcal("mifflin", sex, age, w, h), want, rtol=1e-12)
    # Derived on adults: a 15-year-old is not screened by it.
    assert np.isnan(bmr_kcal("mifflin", np.array(["male"], dtype=object), [15.0], [60.0], [170.0])[0])


def test_5_the_offered_screen_states_the_recall_days_each_row_averages(tmp_path):
    """The proposal's d is the number of recalls each analysis row's energy averages, read from the
    working table's row map: 2 when every person's two recalls were combined by the mean; with 1 to
    3 recalls, the fewest, and the label says so (NUTRITION_PACK §02: "applying Goldberg to a
    single day without adjusting `d` in the S term … a multi-day cut-off on 1-day data
    over-excludes"); 1 when nothing was combined."""
    from turbotab.core.graph import Bundle
    from turbotab.core.stages.proposals import recall_days

    def working(units: list[int], method: str | None = "mean") -> Bundle:
        path = tmp_path / f"row_map_{len(units)}_{method}.parquet"
        pd.DataFrame({"row_id": units, "source_row_id": range(len(units))}).to_parquet(path)
        aggregation = {"method": method} if method else None
        return Bundle(data={"aggregation": aggregation, "n_rows": len(set(units))},
                      files={"row_map.parquet": path})

    assert recall_days(working([0, 0, 1, 1, 2, 2])) == (2.0, None)
    days, note = recall_days(working([0, 1, 1, 2, 2, 2]))
    assert days == 1.0 and note == "the fewest recalls anyone has; others have up to 3"
    assert recall_days(working([0, 0, 1, 1], method="first")) == (1.0, None)
    assert recall_days(Bundle(data={"aggregation": None, "n_rows": 4})) == (1.0, None)


# ── 5c · the screen, row by row ──────────────────────────────────────────────


def efsa_bmr_kcal(sex: np.ndarray, age: np.ndarray, weight: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    """BMR by EFSA's printed kcal/d Schofield equations (weight and height), NaN under age 10."""
    out = np.full(len(age), np.nan)
    for i in range(len(age)):
        if age[i] < 10 or sex[i] not in EFSA_SCHOFIELD_WH:
            continue
        (_, *c) = next(p for p in EFSA_SCHOFIELD_WH[sex[i]] if p[0][0] <= age[i] < p[0][1])
        c = [float(x) for x in c]
        out[i] = c[0] * weight[i] + c[1] * height_cm[i] / 100 + c[2]
    return out


def misreporting_table(seed: int = 12, n: int = 1500) -> pd.DataFrame:
    """One row per adult: body size, two-recall mean energy with under- and over-reporters, fiber
    (misreported with the energy), activity category, and LDL depending on true fiber intake.

    Rows whose EI:BMR would fall within 1.5% of a cut-off are moved off it, so the EFSA kcal
    equations and the printed (rounded) cut-offs classify every row as the exact ones do."""
    rng = np.random.default_rng(seed)
    male = rng.random(n) < 0.5
    sex = np.where(male, "male", "female")
    age = rng.integers(20, 80, n).astype(float)
    height = np.where(male, rng.normal(176, 7, n), rng.normal(162, 7, n)).round(1)
    weight = (rng.normal(27, 4.5, n).clip(17, 45) * (height / 100) ** 2).round(1)
    bmr = efsa_bmr_kcal(sex, age, weight, height)
    tee = bmr * rng.normal(1.65, 0.18, n).clip(1.25, 2.3)
    kind = rng.choice(["plausible", "under", "over"], n, p=[0.74, 0.21, 0.05])
    factor = np.where(kind == "under", rng.uniform(0.35, 0.6, n),
                      np.where(kind == "over", rng.uniform(1.55, 1.9, n), rng.normal(1.0, 0.1, n)))
    kcal = tee * factor
    lower, upper = EFSA_TABLE_4[1.6]
    for _ in range(20):  # move rows off the cut-offs, away from the nearer one
        ratio = kcal / bmr
        near = (np.abs(ratio / lower - 1) < 0.015) | (np.abs(ratio / upper - 1) < 0.015)
        if not near.any():
            break
        side = np.where(np.abs(ratio - lower) < np.abs(ratio - upper),
                        np.sign(ratio - lower), np.sign(ratio - upper))
        kcal[near] *= 1 + 0.04 * np.where(side[near] == 0, 1, side[near])
    kcal = kcal.round(1)
    true_fiber = tee / 1000 * rng.normal(9.0, 2.5, n).clip(2)
    fiber = (true_fiber * factor).round(1)
    ldl = (150 - 1.2 * true_fiber + 0.3 * (age - 50) + 5 * male + rng.normal(0, 12, n)).round(1)
    activity = rng.choice(["low", "moderate", "vigorous"], n, p=[0.4, 0.45, 0.15])
    return pd.DataFrame({"participant_id": [f"G{i:05d}" for i in range(n)], "sex": sex, "age": age,
                         "height": height, "weight": weight, "activity": activity, "kcal": kcal,
                         "fiber_g": fiber, "ldl": ldl})


GOLDBERG = {"kind": "goldberg", "column": "kcal", "energy_unit": "kcal", "days": 2, "sex": "sex",
            "female": ["female"], "male": ["male"], "age": "age", "weight": "weight",
            "height": "height", "height_unit": "cm", "equation": "schofield_height", "pal": 1.6,
            "reason": "implausible energy reports (Goldberg cut-offs, Black 2000)"}
FIXED = {"kind": "range", "column": "kcal", "low": 500, "high": 3500,
         "reason": "implausible intakes (500–3,500 kcal a day)"}


def efsa_keeps(frame: pd.DataFrame, pal: Any = 1.6) -> np.ndarray:
    """EFSA's procedure for individuals, longhand: EI over BMR from the kcal equations, inside
    Table 4's printed cut-offs for the person's PAL."""
    ratio = frame["kcal"].to_numpy() / efsa_bmr_kcal(frame["sex"].to_numpy(), frame["age"].to_numpy(),
                                                     frame["weight"].to_numpy(), frame["height"].to_numpy())
    pals = np.broadcast_to(np.asarray(pal, dtype=float), ratio.shape)
    lower = np.array([EFSA_TABLE_4[round(float(p), 1)][0] for p in pals])
    upper = np.array([EFSA_TABLE_4[round(float(p), 1)][1] for p in pals])
    return (ratio >= lower) & (ratio <= upper)


def test_5_the_screen_classifies_every_row_as_efsa_procedure_does():
    """Each row's status is EFSA's individual-level procedure computed longhand: Schofield (kcal/d)
    from weight and height, EI:BMR against Table 4's printed cut-offs for PAL 1.6 and d = 2; the
    same with energy in kJ; with a PAL per activity category (EFSA's 18–69 values: low 1.4,
    moderate 1.6, vigorous 1.8); and rows the screen cannot judge are counted as such, never
    classified."""
    from turbotab.core.decisions import GoldbergRule

    frame = misreporting_table(seed=3, n=800)
    rule = GoldbergRule.model_validate(GOLDBERG)
    got = screen(frame, rule)
    keep = efsa_keeps(frame)
    assert (got["inside"].to_numpy() == keep).all()
    ratio = frame["kcal"] / efsa_bmr_kcal(frame["sex"].to_numpy(), frame["age"].to_numpy(),
                                          frame["weight"].to_numpy(), frame["height"].to_numpy())
    assert (got["status"].eq("under") == (ratio < 0.996)).all()
    assert (got["status"].eq("over") == (ratio > 2.570)).all()
    assert 0.1 < (~keep).mean() < 0.4  # the fixture has misreporters for the screen to find

    kj = frame.assign(kcal=frame["kcal"] * 4.184)
    in_kj = screen(kj, rule.model_copy(update={"energy_unit": "kj"}))
    assert (in_kj["status"] == got["status"]).all()

    pal_by = {"column": "activity", "values": {"low": 1.4, "moderate": 1.6, "vigorous": 1.8}}
    by_level = screen(frame, GoldbergRule.model_validate({**GOLDBERG, "pal": None, "pal_by": pal_by}))
    pals = frame["activity"].map(pal_by["values"]).to_numpy()
    want = efsa_keeps(frame, pals)
    # The printed cut-offs are rounded; judge only the rows not within 1.5% of their own.
    ratio_v = ratio.to_numpy()
    lo = np.array([EFSA_TABLE_4[p][0] for p in pals])
    hi = np.array([EFSA_TABLE_4[p][1] for p in pals])
    clear = (np.abs(ratio_v / lo - 1) > 0.015) & (np.abs(ratio_v / hi - 1) > 0.015)
    assert clear.mean() > 0.9
    assert (by_level["inside"].to_numpy()[clear] == want[clear]).all()

    edge = frame.head(4).copy()
    edge.loc[edge.index[0], "weight"] = np.nan  # not recorded
    edge.loc[edge.index[1], "age"] = 8.0  # no Schofield band below 10: not screened
    edge.loc[edge.index[2], "sex"] = "unknown"  # a level read as neither: not screened
    s = screen(edge, rule)
    assert list(s["status"][:3]) == ["not recorded", "not screened", "not screened"]
    assert not s["screened"][:3].any()


# ── 5d · primary plus sensitivity, through the server ─────────────────────────


ROLES = {"participant_id": "identifier", "sex": "covariate", "age": "covariate", "kcal": "energy",
         "fiber_g": "exposure", "height": "excluded", "weight": "excluded", "activity": "excluded"}


def run_sensitivity(path: Path, home: Path, purpose: str) -> tuple[dict, dict, set[int]]:
    """The opening sequence with the Goldberg screen as the primary rule, then the sensitivity
    answer naming a fixed 500–3,500 kcal screen; returns (sensitivity, fit, sealed rows)."""
    with local_server(home) as client:
        d = open_project(client, path)
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "ldl"})
        d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": purpose})
        d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit",
                           "id_column": "participant_id"})
        d.reach("roles")
        d.decide({"kind": "set_roles", "roles": ROLES})
        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": [GOLDBERG]})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        d.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["fiber_g"]})
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["linear"]})
        d.decide({"kind": "set_sensitivity",
                  "analyses": [{"label": "500–3,500 kcal", "rules": [FIXED]}]})
        sensitivity = d.artifact("sensitivity")
        fit = d.artifact("fit")
        sealed = d.sealed()
    return sensitivity, fit, sealed


def ols_hc3(frame: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """``ldl ~ fiber_g + kcal + age + [sex = male]`` by NumPy least squares, with HC3 written out
    (MacKinnon & White 1985) and a t(n − p) interval: {column: (estimate, low, high)}."""
    X = np.column_stack([np.ones(len(frame)), frame["fiber_g"], frame["kcal"], frame["age"],
                         (frame["sex"] == "male").astype(float)])
    y = frame["ldl"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    V = hc3_by_definition(X, y - X @ beta)
    t = stats.t.ppf(0.975, len(y) - X.shape[1])
    se = np.sqrt(np.diag(V))
    names = ["(intercept)", "fiber_g", "kcal", "age", "sex_male"]
    return {n: (b, b - t * s, b + t * s) for n, b, s in zip(names, beta, se)}


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> tuple[pd.DataFrame, Path]:
    frame = misreporting_table()
    path = tmp_path_factory.mktemp("wp12c_goldberg") / "misreporting.csv"
    frame.to_csv(path, index=False)
    return frame, path


@pytest.fixture(scope="module")
def inference_run(table, tmp_path_factory):
    _, path = table
    return run_sensitivity(path, tmp_path_factory.mktemp("wp12c_inference_home"), "inference")


def _coefficient(fit: dict, feature: str) -> dict:
    return next(c for c in fit["coefficients"] if c["feature"] == feature)


def test_5_the_answer_renders_the_estimate_under_each_exclusion_rule(table, inference_run):
    """Under inference: three analyses, in order: the primary (Goldberg), the named 500–3,500 kcal
    screen, and the every-row analysis the stage adds because the primary excludes rows (Banna).
    Each counts the rows its own rule keeps (EFSA's procedure longhand, and a pandas range), every
    eligible row including the held-out ones (BLUEPRINT §12 ruling 3), and each fiber coefficient
    and HC3 interval is the independent refit's on exactly those rows."""
    frame, _ = table
    sensitivity, _, sealed = inference_run
    assert sensitivity["purpose"] == "inference" and sensitivity["rows"] == "all eligible rows"
    labels = [(a["label"], a["primary"], a["added"]) for a in sensitivity["analyses"]]
    assert labels == [("Primary", True, False), ("500–3,500 kcal", False, False),
                      ("Every row", False, True)]
    keeps = [efsa_keeps(frame), ((frame["kcal"] >= 500) & (frame["kcal"] <= 3500)).to_numpy(),
             np.ones(len(frame), dtype=bool)]
    assert sealed and sealed <= set(np.flatnonzero(keeps[2]))  # held-out rows exist, and count
    family = sensitivity["families"][0]
    assert family["family"] == "linear" and len(family["fits"]) == 3
    references = []
    for analysis, fit_, keep in zip(sensitivity["analyses"], family["fits"], keeps):
        assert analysis["n_rows"] == fit_["n_rows"] == int(keep.sum()), analysis["label"]
        ref = ols_hc3(frame[keep])["fiber_g"]
        got = _coefficient(fit_, "fiber_g")
        assert got["estimate"] == pytest.approx(ref[0], rel=1e-9, abs=1e-12)
        assert got["ci_low"] == pytest.approx(ref[1], rel=1e-7)
        assert got["ci_high"] == pytest.approx(ref[2], rel=1e-7)
        assert fit_["inference"]["covariance"] == "HC3"
        references.append(ref[0])
    # The rule moves the answer, which is why both are reported.
    assert len(set(np.round(references, 6))) == 3
    change = sensitivity["changes"]["linear"]
    fiber = next(c for c in change if c["feature"] == "fiber_g")
    assert fiber["primary"] == pytest.approx(references[0], rel=1e-9)
    assert fiber["lowest"] == pytest.approx(min(references), rel=1e-9)
    assert fiber["highest"] == pytest.approx(max(references), rel=1e-9)
    assert sensitivity["exposures"] == ["fiber_g"]


def test_5_the_rules_and_the_reason_are_stated_with_banna(inference_run):
    """The methods sentence names every analysis's rule (the Goldberg screen with its equation, PAL
    and days), and the added every-row analysis says why, quoting Banna et al. 2017."""
    sensitivity, _, _ = inference_run
    methods = sensitivity["methods"]
    assert "Goldberg cut-offs for PAL `1.6` and `2` days" in methods
    assert "Schofield (weight and height) BMR" in methods
    assert "excluding rows with `kcal` outside `500`–`3500` or not recorded" in methods
    assert "Every row (keeping every row)" in methods
    assert any("analyses in the total sample without exclusion of participants should also be "
               "conducted and reported" in c for c in sensitivity["concerns"])
    primary_rules = sensitivity["analyses"][0]["rules"]
    assert primary_rules and "Goldberg cut-offs" in primary_rules[0]


def test_5_under_prediction_no_analysis_reads_a_held_out_row(table, tmp_path_factory):
    """Under prediction the seal holds: each analysis is fit on the rows its rule keeps less every
    sealed row (including sealed rows the primary's rule excludes, which the every-row analysis
    would otherwise bring back), and a concern says the excluded people are still in the
    population the model will be used on (audit ME-16's prediction leash)."""
    frame, path = table
    sensitivity, _, sealed = run_sensitivity(path, tmp_path_factory.mktemp("wp12c_prediction_home"),
                                             "prediction")
    assert sensitivity["purpose"] == "prediction"
    keeps = [efsa_keeps(frame), ((frame["kcal"] >= 500) & (frame["kcal"] <= 3500)).to_numpy(),
             np.ones(len(frame), dtype=bool)]
    for analysis, keep in zip(sensitivity["analyses"], keeps):
        rows = set(np.flatnonzero(keep)) - sealed
        assert analysis["n_rows"] == len(rows), analysis["label"]
    fits = sensitivity["families"][0]["fits"]
    every = fits[2]
    assert every["n_rows"] == len(frame) - len(sealed)
    est = _coefficient(every, "fiber_g")["estimate"]
    open_rows = np.array(sorted(set(range(len(frame))) - sealed))
    assert est == pytest.approx(ols_hc3(frame.iloc[open_rows])["fiber_g"][0], rel=1e-9)
    assert any("still among those the model will be used on" in c for c in sensitivity["concerns"])


def test_5_a_goldberg_rule_that_reads_the_outcome_is_refused(table, tmp_path_factory):
    """Banna et al. 2017: body weight "is included in both the calculation of implausible rEI and
    the outcome variable … which could artificially elevate the association". With weight as the
    outcome, the Goldberg screen reads the outcome (its BMR), so it is refused as a primary rule and
    inside a sensitivity analysis (audit RO-01), each with a way forward."""
    _, path = table
    with local_server(tmp_path_factory.mktemp("wp12c_outcome_home")) as client:
        d = open_project(client, path)
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "weight"})
        d.answer("task", {"kind": "set_task", "column": "weight", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit",
                           "id_column": "participant_id"})
        d.reach("roles")
        d.decide({"kind": "set_roles",
                  "roles": {**{c: r for c, r in ROLES.items() if c != "weight"}, "ldl": "covariate"}})
        d.reach("exclusions")
        r = d.post({"kind": "set_exclusions", "rules": [GOLDBERG]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "rule_on_outcome" and error["exits"]
        r = d.post({"kind": "set_sensitivity", "analyses": [{"label": "Goldberg", "rules": [GOLDBERG]}]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "rule_on_outcome" and "In “Goldberg”" in error["message"]
        assert any(e["label"] == "Leave out “Goldberg”" for e in error["exits"])
