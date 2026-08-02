"""Generate the five domain fixtures. Deterministic; safe to re-run.

    turbotab/.venv/bin/python turbotab/sample_data/make_fixtures.py

**Why the generator is committed and not just its output.** A fixture whose
generator is lost is a fixture nobody can adjust: the next person who needs one
more impossible value, or one fewer QC row, has to hand-edit a 400-column CSV or
regenerate from scratch and lose every other property it was carrying. The
companion `.md` states what each file must produce; this states how it got that
way, and the two together are what make a drive a comparison against an
expectation rather than a guess.

**Every characteristic below is deliberate**, and each one is named in the
companion `.md` as either *must surface* or *must not*. Nothing here is
incidental noise — where a number looks arbitrary it is pinned to something the
engine actually reads:

* the impossible values are chosen outside `ml/physiology_reference.py`'s
  **impossibility band** (`floor`/`ceiling`), which is the tier that earns a
  repair proposal;
* the implausible dietary intakes are chosen INSIDE that band and outside the
  **reference interval** (`p01`/`p99`), which is the tier that stays advisory —
  because an implausible intake is an eligibility criterion the user states, not
  an entry error the app repairs (`DOMAIN_PACKS.md` §07);
* the column names are chosen knowing `_REPEAT_SUFFIX` in `ml/import_doctor.py`
  matches a trailing one-or-two-digit run, so `mz_0001` DOES form a family and
  the wide-shape false alarm fires. That is the point: the metabolomics,
  genomics and survey fixtures must reproduce the false alarm the lens exists to
  correct, or the lens has nothing to demonstrate.

Shapes follow the loop brief rather than `OPENING_SEQUENCE.md` §04's earlier
table, which said 80 × 1,847 for metabolomics and 500 people for dietary. Both
are the same shape an order of magnitude apart; the smaller ones load instantly,
which is the property the drive needs.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# 1 · metabolomics_untargeted — 80 × 400
# ─────────────────────────────────────────────────────────────────────────────

def metabolomics() -> pd.DataFrame:
    """80 samples, 392 features, 8 pooled QC rows, one run-order column.

    Four properties are load-bearing:

    * **Log-normal intensities.** Concentrations are bounded below by zero and
      combine multiplicatively; the log-normality is by construction, which is
      why the pack's log-transform default is `derived` rather than convention.
    * **Left-censored missingness.** Missing values are placed by ABUNDANCE
      RANK, not at random: the lowest-abundance features lose up to ~55% of
      their values and the highest lose none. A median imputation would put
      non-detections in the middle of the distribution.
    * **Instrument drift along `run_order`.** A multiplicative ramp, so
      intensity correlates with acquisition order — detection is derived,
      correction is never automatic because it alters every value.
    * **Pooled QC rows every tenth injection.** They look exactly like
      participants. They carry no age, no sex and no outcome, they repeat one
      pooled sample so their between-row variance is small, and modeling them
      is an error with no legitimate reading.
    """
    rng = np.random.RandomState(20260728)
    n, n_feat = 80, 392

    qc_positions = set(range(0, n, 10))                # 8 rows: 0,10,20,…,70
    is_qc = np.array([i in qc_positions for i in range(n)])

    # Per-feature abundance on the log scale, spanning four orders of magnitude.
    log_mu = rng.uniform(np.log(50.0), np.log(500_000.0), size=n_feat)
    log_sd = rng.uniform(0.25, 0.85, size=n_feat)

    # Instrument drift: a smooth multiplicative ramp along acquisition order,
    # different per feature in magnitude but shared in direction — which is what
    # makes it a batch effect rather than noise.
    run_order = np.arange(1, n + 1)
    # Calibrated, not guessed: at this strength about 40% of features reach
    # |r| > 0.3 against acquisition order, which is what "instrument drift is
    # often the largest single variance component" looks like in a table. Too
    # weak and there is nothing for a detector to find; too strong and the
    # fixture proves only that an obvious thing is detectable.
    drift_strength = rng.uniform(-0.9, 1.1, size=n_feat)
    ramp = (run_order - run_order.mean()) / run_order.std()

    values = np.zeros((n, n_feat))
    for j in range(n_feat):
        biological = rng.normal(0.0, log_sd[j], size=n)
        # A pooled QC is one sample injected repeatedly: the biology is
        # identical and only technical variation remains.
        biological[is_qc] = rng.normal(0.0, log_sd[j] * 0.12, size=is_qc.sum())
        values[:, j] = np.exp(log_mu[j] + biological + drift_strength[j] * ramp * 0.55)

    # Left censoring. The detection limit is the same instrument threshold for
    # every feature, so which features lose values is decided by where they sit
    # relative to it — abundance rank — rather than by a random draw.
    rank = np.argsort(np.argsort(log_mu)) / max(n_feat - 1, 1)   # 0 = lowest
    for j in range(n_feat):
        p_missing = float(np.clip(0.55 * (1.0 - rank[j]) ** 3, 0.0, 0.55))
        if p_missing <= 0.005:
            continue
        # Censoring is not random WITHIN a feature either: the smallest values
        # are the ones that fall below the limit.
        k = int(round(p_missing * n))
        if k:
            lowest = np.argsort(values[:, j])[:k]
            values[lowest, j] = np.nan

    sample_id = [f"QC{i // 10 + 1:02d}" if is_qc[i] else f"S{i + 1:03d}"
                 for i in range(n)]
    age = np.where(is_qc, np.nan, rng.randint(28, 78, size=n).astype(float))
    sex = np.where(is_qc, "", rng.choice(["F", "M"], size=n))
    bmi = np.where(is_qc, np.nan, np.round(rng.normal(27.5, 4.6, size=n), 1))
    responder = np.where(is_qc, np.nan, rng.binomial(1, 0.42, size=n).astype(float))

    frame = {
        "sample_id": sample_id,
        "sample_type": np.where(is_qc, "pooled_qc", "participant"),
        "run_order": run_order,
        "batch": np.where(run_order <= 40, "B1", "B2"),
        "age": age,
        "sex": sex,
        "bmi": bmi,
    }
    for j in range(n_feat):
        frame[f"mz_{j + 1:04d}"] = np.round(values[:, j], 2)
    frame["responder"] = responder
    return pd.DataFrame(frame)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · dietary_recalls — 300 people × 2 recalls
# ─────────────────────────────────────────────────────────────────────────────

def dietary() -> pd.DataFrame:
    """Two 24-hour recalls per person, macronutrient percentages summing to 100.

    The recall dates are the evidence question 4 turns on, so they are built to
    be exactly what they are: gaps of 3 to 14 days, irregular, with nothing that
    forms a schedule. That is what makes *repeats* the stated reading and a visit
    series the wrong one.

    `hba1c` is measured once at the clinic and is therefore CONSTANT within a
    person. That is deliberate and it is the easy case: the outcome does not
    vary across the rows being combined, so aggregation raises no "which
    outcome?" question. `clinical_longitudinal` is the hard case, and it varies
    on purpose.
    """
    rng = np.random.RandomState(4242)
    n_people, n_recalls = 300, 2
    base = date(2024, 3, 4)

    rows = []
    for p in range(n_people):
        pid = f"P{p + 1:03d}"
        age = int(rng.randint(21, 80))
        sex = str(rng.choice(["F", "M"]))
        bmi = float(np.round(rng.normal(28.0, 5.2), 1))
        # One clinic measurement per person, not per recall.
        hba1c = float(np.round(np.clip(rng.normal(5.7, 0.9), 4.0, 12.0), 1))
        usual_kcal = float(np.clip(rng.normal(2150 if sex == "M" else 1780, 380), 1000, 3900))
        first = base + timedelta(days=int(rng.randint(0, 120)))
        gap = int(rng.randint(3, 15))                  # 3–14 days, irregular
        for r in range(n_recalls):
            when = first if r == 0 else first + timedelta(days=gap)
            # Within-person day-to-day variation is large: a single recall is a
            # noisy estimate of usual intake, which is the whole reason the mean
            # of two is recommended.
            # Clipped to a range no reasonable person would call implausible, so
            # the only implausible intakes in this file are the seeded ones and
            # a count is a count rather than a count plus a tail.
            kcal = float(np.clip(rng.normal(usual_kcal, usual_kcal * 0.28), 620, 4400))
            prot = float(np.clip(rng.normal(16.5, 3.4), 6, 34))
            fat = float(np.clip(rng.normal(34.0, 6.5), 14, 55))
            alc = float(max(0.0, rng.normal(1.6, 3.2)))
            carb = 100.0 - prot - fat - alc
            if carb < 20.0:                            # keep the composition real
                scale = (100.0 - 20.0) / (prot + fat + alc)
                prot, fat, alc = prot * scale, fat * scale, alc * scale
                carb = 100.0 - prot - fat - alc
            rows.append({
                "participant_id": pid,
                "recall_number": r + 1,
                "recall_date": when.isoformat(),
                "age": age, "sex": sex, "bmi": bmi,
                "energy_kcal": round(kcal, 0),
                "protein_pct_kcal": round(prot, 2),
                "fat_pct_kcal": round(fat, 2),
                "carbohydrate_pct_kcal": round(carb, 2),
                "alcohol_pct_kcal": round(alc, 2),
                "protein_g": round(kcal * prot / 100.0 / 4.0, 1),
                "fat_g": round(kcal * fat / 100.0 / 9.0, 1),
                "carbohydrate_g": round(kcal * carb / 100.0 / 4.0, 1),
                "fiber_g": round(float(np.clip(rng.normal(18.0, 7.0), 1, 60)), 1),
                "sodium_mg": round(float(np.clip(rng.normal(3200, 1100), 300, 9000)), 0),
                "hba1c": hba1c,
            })

    df = pd.DataFrame(rows)

    # Implausible intakes. INSIDE the impossibility band (100–30000 kcal) and
    # outside the reference interval (800–4500), so the impossibility pass must
    # leave them alone and the dietary pack must OFFER an exclusion criterion.
    # 12 under-reports and 8 over-reports, on 20 distinct people.
    under = rng.choice(np.arange(0, len(df), 2), size=12, replace=False)
    remaining = [i for i in np.arange(0, len(df), 2) if i not in set(under)]
    over = rng.choice(remaining, size=8, replace=False)
    for i in under:
        df.loc[i, "energy_kcal"] = float(rng.randint(240, 430))
    for i in over:
        df.loc[i, "energy_kcal"] = float(rng.randint(6100, 7900))
    # The gram columns are recomputed so the composition still adds up — an
    # implausible total with plausible parts would be a different defect.
    for i in list(under) + list(over):
        k = float(df.loc[i, "energy_kcal"])
        df.loc[i, "protein_g"] = round(k * df.loc[i, "protein_pct_kcal"] / 100.0 / 4.0, 1)
        df.loc[i, "fat_g"] = round(k * df.loc[i, "fat_pct_kcal"] / 100.0 / 9.0, 1)
        df.loc[i, "carbohydrate_g"] = round(k * df.loc[i, "carbohydrate_pct_kcal"] / 100.0 / 4.0, 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3 · clinical_longitudinal — 200 people × 3 visits
# ─────────────────────────────────────────────────────────────────────────────

def clinical() -> pd.DataFrame:
    """Three scheduled visits per person, with impossible vitals seeded in.

    The visit dates are on a SCHEDULE — baseline, ~90 days, ~180 days, each ±5
    days — which is the evidence that makes *time points* the stated reading.
    Averaging three visits of a trajectory destroys the trajectory, so the
    aggregation menu must not recommend the mean here.

    `progressed` is measured AT EACH VISIT and varies within a person. That is
    what makes question 6 incoherent without question 2 answered: combining a
    person's rows requires deciding which outcome, and there is no obvious one.

    Every impossible value below is outside `physiology_reference`'s
    impossibility band, so each earns a repair proposal rather than an advisory.
    """
    rng = np.random.RandomState(90210)
    n_people, n_visits = 200, 3
    base = date(2023, 1, 9)

    rows = []
    for p in range(n_people):
        pid = f"P{p + 1:03d}"
        age0 = int(rng.randint(35, 85))
        sex = str(rng.choice(["F", "M"]))
        height = float(np.round(rng.normal(171 if sex == "M" else 159, 7.5), 1))
        weight0 = float(np.round(rng.normal(84 if sex == "M" else 72, 15.0), 1))
        sbp0 = float(np.round(rng.normal(134, 17), 0))
        dbp0 = float(np.round(rng.normal(81, 10), 0))
        gluc0 = float(np.round(rng.normal(108, 26), 0))
        a1c0 = float(np.round(np.clip(rng.normal(6.2, 1.1), 4.2, 13.0), 1))
        trend = float(rng.normal(0.0, 1.0))
        enrol = base + timedelta(days=int(rng.randint(0, 200)))
        for v in range(n_visits):
            when = enrol + timedelta(days=90 * v + int(rng.randint(-5, 6)))
            drift = v * trend
            rows.append({
                "subject_id": pid,
                "visit": v + 1,
                "visit_date": when.isoformat(),
                "age": age0,
                "sex": sex,
                "height_cm": height,
                "weight_kg": round(weight0 + drift * 1.4 + rng.normal(0, 0.8), 1),
                "sbp": round(sbp0 + drift * 3.1 + rng.normal(0, 5), 0),
                "dbp": round(dbp0 + drift * 1.6 + rng.normal(0, 4), 0),
                "heart_rate": round(float(np.clip(rng.normal(74, 11), 42, 130)), 0),
                "glucose": round(gluc0 + drift * 4.2 + rng.normal(0, 7), 0),
                "hba1c": round(float(np.clip(a1c0 + drift * 0.12 + rng.normal(0, 0.15), 4.0, 15.0)), 1),
                "progressed": int(rng.rand() < (0.18 + 0.06 * v + 0.04 * max(trend, 0))),
            })

    df = pd.DataFrame(rows)

    # Fourteen impossible cells across five columns, on fourteen distinct rows.
    # A cuff that failed reads 0, a transcription error moves a decimal, a
    # glucose in mmol/L pasted into a mg/dL column reads absurd.
    picks = rng.choice(len(df), size=14, replace=False)
    seeded = [("dbp", 0.0), ("dbp", 0.0), ("dbp", 0.0), ("dbp", 0.0),
              ("sbp", 400.0), ("sbp", 400.0), ("sbp", 999.0),
              ("weight_kg", 0.0), ("weight_kg", 0.0),
              ("glucose", 5000.0), ("glucose", 5000.0),
              ("height_cm", 0.0), ("height_cm", 0.0), ("height_cm", 1.7)]
    for row, (col, value) in zip(picks, seeded):
        df.loc[row, col] = value
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3b · clinical_risk — 480 encounters, a model that does NOT separate
#
# `GUIDED-135`, and it is `GUIDED-097`'s fixture rule at the opposite polarity.
# That rule was written from *do not verify against the fixture that works*;
# this is the mirror — **the fixture that degenerately fails.**
#
# `leaky_sepsis.csv` is the fixture behind every calibration claim in this
# repository and its held-out C-statistic is **1.000**: 24 rows, 16 events,
# complete separation. So `weak_calibration` correctly returns `(None, None)`,
# the annotation box correctly renders *not estimable* for the intercept and
# the slope, and the `annotation_box` checklist item correctly FAILS — because
# a figure missing two of its six required numbers is not publication-grade.
#
# Which means the flagship clinical figure has been asserted for six loops only
# in the one state where two of its numbers cannot exist, and its own checklist
# item had never been observed passing anywhere.
#
# `leaky_sepsis` keeps its job. It is the right fixture for leakage and for the
# not-estimable branch, and that branch is real and worth holding. What it
# cannot do is show the figure passing, and until this file nothing else could.
# ─────────────────────────────────────────────────────────────────────────────

def clinical_risk(n: int = 480) -> pd.DataFrame:
    """A 30-day readmission cohort whose logistic fit lands around C = 0.72.

    **Every coefficient here is chosen to keep the model ORDINARY.** A fixture
    that separates proves nothing about the figure and a fixture at chance
    proves nothing either; what the calibration plot needs in order to be
    checkable is a model that is genuinely mediocre, which is what a real
    30-day readmission model is — the LACE-index literature sits around
    C = 0.65–0.70 and nothing in this space does much better.

    Measured on the sealed 25% holdout with `logreg`, at `numpy` seed 42 for
    the split:

    | | |
    |---|---|
    | held-out rows / events | 120 / 31 |
    | C-statistic | 0.719 |
    | calibration intercept | +0.034 |
    | calibration slope | 0.795 |
    | E:avg / E:max | 0.066 / 0.365 |

    The slope below 1 is not a defect in the fixture — it is the small-sample
    overfitting signature the figure's own caption exists to explain, arriving
    honestly rather than being asserted about a model that has none.

    **The outcome is generated from a logistic model of the predictors plus
    Bernoulli noise, so the true risk is genuinely partly unpredictable.** That
    is the property `leaky_sepsis` deliberately lacks: there,
    `abx_escalation_score` carries r = 0.9996 with the outcome, which is the
    whole point of that file and the reason it separates.

    No column here is measured after the outcome and none is a proxy for it, so
    this table must NOT raise a leakage finding. `encounter_id` is one level per
    row and is excluded by `turbotab/identifiers.py` with a sentence, which is
    the ordinary path rather than a property of this fixture.
    """
    rng = np.random.default_rng(20260802)

    age = np.clip(rng.normal(68, 13, n), 22, 97).round(0)
    sex = rng.choice(["F", "M"], n)
    charlson = rng.poisson(2.2, n).clip(0, 12)
    prior = rng.poisson(0.9, n).clip(0, 8)
    albumin = np.clip(rng.normal(3.6, 0.45, n), 1.6, 5.2).round(1)
    creatinine = np.clip(rng.lognormal(np.log(1.05), 0.42, n), 0.3, 9.0).round(2)
    hemoglobin = np.clip(rng.normal(11.9, 1.7, n), 6.0, 18.0).round(1)
    sodium = np.clip(rng.normal(138, 4.0, n), 118, 155).round(0)
    los = np.clip(rng.gamma(2.2, 2.1, n), 1, 40).round(0)

    # Centered at the cohort's own means so the intercept IS the log-odds of
    # the base rate, which is what makes the ~23% event fraction readable off
    # the -1.35 rather than emerging from arithmetic nobody can follow.
    z = (-1.35
         + 0.030 * (age - 68)
         + 0.26 * (charlson - 2.2)
         + 0.50 * (prior - 0.9)
         - 0.95 * (albumin - 3.6)
         + 0.45 * np.log(creatinine / 1.05)
         - 0.16 * (hemoglobin - 11.9)
         + 0.075 * (los - 4.6))
    readmit = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)

    return pd.DataFrame({
        "encounter_id": [f"ENC{i:04d}" for i in range(n)],
        "age": age.astype(int),
        "sex": sex,
        "charlson_index": charlson.astype(int),
        "prior_admissions_12mo": prior.astype(int),
        "albumin_g_dl": albumin,
        "creatinine_mg_dl": creatinine,
        "hemoglobin_g_dl": hemoglobin,
        "sodium_mmol_l": sodium.astype(int),
        "length_of_stay_days": los.astype(int),
        "readmit_30d": readmit,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3c · clinical_labs — the messy EHR export the clinical detectors were written for
#
# `research/CLINICAL_SURVEY_PACK.md` §A1.1 and §A1.3, and `DOMAIN_SCIENCE.md`
# §03b's clinical rows. Until L41 the clinical pack held one prior and zero
# detectors against a 1,209-line research file, and the argument for the
# thinness — *physiologic bounds and unit harmonization already live in the
# core* — was true of §A1.2 and **not true of §A1.3**, which describes machinery
# nothing in this repository has.
#
# One table rather than six, because §A1.3's own framing is that these arrive
# together: a real multi-site lab extract carries censoring tokens, a column
# typed as text because of them, an analyte reported in two units by two sites,
# and a vitals column with a manual-entry spike, all at once. Splitting them
# into one property per file would make each detector easy and the interaction
# untested.
#
# It is LONGITUDINAL — 96 patients × 3 visits — because temporal plausibility
# (§A1.2, Kahn et al.) is a claim about a trajectory and cannot be built on a
# cross-section.
# ─────────────────────────────────────────────────────────────────────────────

def clinical_labs(n_patients: int = 96, n_visits: int = 3) -> pd.DataFrame:
    """288 rows carrying every §A1.3 format problem at once, each one seeded.

    Every property below is named in the companion `.md` as *must surface* or
    *must not*, and each is pinned to something a detector actually reads.

    | Column | What it carries | Section |
    |---|---|---|
    | `hs_crp` | `<0.3` on ~19% of rows — one modal detection limit | §A1.3 left censoring |
    | `ferritin` | `>1500` above the upper limit of quantitation | §A1.3 ULOQ |
    | `wbc` | `TNTC` and `QNS` — **measurement failures, not censoring** | §A1.3 |
    | `troponin` | `0.04` and `negative` in one column | §A1.3 mixed quant/qual |
    | `glucose` | two sites, mmol/L and mg/dL, ratio 18.0 | §A1.1 mixed units |
    | `sbp` / `dbp` | mass at 120/80, four impossible, many abnormal-but-real | §A1.2, §03b |
    | `temp_f` | mass at 98.6 | §03b default values |
    | `height_cm` | one adult growing 9 cm between visits | §A1.2 temporal |
    | `weight_kg` | one adult losing 34% in 21 days | §A1.2 temporal |
    | `platelets` | `252,000` — a thousands separator | §A1.3 |
    | `creatinine` | `1,05` — a European decimal comma | §A1.3 |
    | `bnp` | `1.2E3` — scientific notation | §A1.3 |

    **The censored fraction is deliberately 19%**, which sits above §A1.3's 10%
    warning threshold and inside the *"above ~20% it is not defensible"* band it
    calls the boundary of the substitution dispute. A fixture at 4% would make
    the DISPUTED clause unreachable and a fixture at 40% would make it
    uninteresting; 19% is the only region where both positions are live.
    """
    rng = np.random.default_rng(770118)
    base = date(2024, 2, 5)

    rows = []
    for p in range(n_patients):
        pid = f"PT{p + 1:04d}"
        age = int(rng.integers(31, 88))
        sex = str(rng.choice(["F", "M"]))
        height = float(np.round(rng.normal(172 if sex == "M" else 160, 7.0), 1))
        weight = float(np.round(rng.normal(86 if sex == "M" else 73, 14.0), 1))
        # A REAL SEVERE-HYPERTENSION TAIL, and it is the whole point of the
        # coaching sentence. One patient in nine runs at hypertensive-urgency
        # pressures — above the 200 mmHg the reference population's 99th
        # percentile sits at — and those readings are abnormal, real, and the
        # sickest people in the cohort. A ±3 SD screen deletes them. Without
        # this tail the fixture would carry four impossible values and nothing
        # to contrast them against, which is the sentence's other half missing.
        severe = rng.random() < 0.11
        sbp0 = float(np.round(rng.normal(212 if severe else 134, 14 if severe else 16), 0))
        dbp0 = float(np.round(rng.normal(118 if severe else 80, 9 if severe else 10), 0))
        # WHICH SITE THIS PATIENT WAS SEEN AT, and it decides the glucose unit.
        # A mixed-unit column is not noise: it is a variable whose meaning
        # changes between rows, and the reason it is patient-level here rather
        # than random is that this is how it actually happens — two hospitals
        # reporting into one extract.
        site = "NORTH" if rng.random() < 0.62 else "SOUTH"
        enrolled = base + timedelta(days=int(rng.integers(0, 240)))
        for v in range(n_visits):
            when = enrolled + timedelta(days=90 * v + int(rng.integers(-6, 7)))
            gluc_mgdl = float(np.round(np.clip(rng.normal(112, 30), 55, 340), 0))
            rows.append({
                "patient_id": pid,
                "visit_date": when.isoformat(),
                "site": site,
                "age": age,
                "sex": sex,
                "height_cm": height,
                "weight_kg": round(weight + rng.normal(0, 1.1), 1),
                "sbp": round(sbp0 + rng.normal(0, 6), 0),
                "dbp": round(dbp0 + rng.normal(0, 5), 0),
                "temp_f": round(float(np.clip(rng.normal(98.4, 0.8), 95.0, 104.0)), 1),
                # mmol/L at SOUTH, mg/dL at NORTH. 18.0 exactly — §A1.1's own
                # first row.
                "glucose": (round(gluc_mgdl / 18.0, 1) if site == "SOUTH"
                            else gluc_mgdl),
                "hs_crp": round(float(np.clip(rng.lognormal(0.4, 1.0), 0.05, 90)), 2),
                "ferritin": round(float(np.clip(rng.lognormal(4.7, 0.8), 5, 4000)), 0),
                "wbc": round(float(np.clip(rng.normal(8.1, 2.6), 1.2, 28)), 1),
                "troponin": round(float(np.clip(rng.lognormal(-3.4, 0.9), 0.005, 4)), 3),
                # Reported per microliter rather than as ×10⁹/L, which is the
                # common US lab convention and the reason the column carries a
                # thousands separator at all — a `252` never needs one.
                "platelets": int(np.clip(rng.normal(252, 70), 40, 900)) * 1000,
                "creatinine": round(float(np.clip(rng.lognormal(np.log(1.02), 0.35), 0.3, 8)), 2),
                "bnp": round(float(np.clip(rng.lognormal(4.6, 1.2), 5, 9000)), 0),
                "readmitted": int(rng.random() < 0.21),
            })

    df = pd.DataFrame(rows)
    n = len(df)

    # ── §A1.3 · censoring, as TEXT, which is what makes the column text ──────
    #
    # One detection limit per analyte, because that is what an assay has. A
    # fixture with three different `<X` values per analyte would make the modal
    # inference trivially wrong rather than trivially right.
    crp = df["hs_crp"].astype(object)
    below = rng.choice(n, size=int(round(0.19 * n)), replace=False)
    crp.iloc[below] = "<0.3"
    df["hs_crp"] = crp.astype(str)

    ferritin = df["ferritin"].astype(object)
    above = rng.choice(np.setdiff1d(np.arange(n), below), size=22, replace=False)
    ferritin.iloc[above] = ">1500"
    df["ferritin"] = ferritin.astype(str)

    # TNTC AND QNS ARE NOT CENSORING. Too numerous to count and quantity not
    # sufficient are measurement FAILURES — the specimen was unusable, or the
    # count was uncountable — and treating them as extreme values would put a
    # number where the assay produced none.
    wbc = df["wbc"].astype(object)
    failures = rng.choice(np.arange(n), size=14, replace=False)
    wbc.iloc[failures[:8]] = "QNS"
    wbc.iloc[failures[8:]] = "TNTC"
    df["wbc"] = wbc.astype(str)

    # A troponin column holding both `0.04` and `negative` — §A1.3's own
    # example, and the one a generic profiler reads as a categorical.
    trop = df["troponin"].astype(object)
    qualitative = rng.choice(np.arange(n), size=41, replace=False)
    trop.iloc[qualitative] = np.where(rng.random(41) < 0.75, "negative", "positive")
    df["troponin"] = trop.astype(str)

    # ── §A1.3 · number formats ───────────────────────────────────────────────
    df["platelets"] = [f"{v:,}" for v in df["platelets"]]
    df["creatinine"] = [f"{v:.2f}".replace(".", ",") for v in df["creatinine"]]
    df["bnp"] = [f"{v:.1E}" for v in df["bnp"]]

    # ── §03b · repeated-digit / default-value mass ───────────────────────────
    #
    # Value preference and manual entry, not measurement. 120/80 on the same
    # rows, because a cuff nobody read is written down as a pair.
    preferred = rng.choice(n, size=int(round(0.12 * n)), replace=False)
    df.loc[df.index[preferred], "sbp"] = 120.0
    df.loc[df.index[preferred], "dbp"] = 80.0
    df.loc[df.index[rng.choice(n, size=int(round(0.09 * n)), replace=False)],
           "temp_f"] = 98.6

    # ── §A1.2 · four impossible systolic values, and the abnormal ones kept ──
    #
    # THE COACHING SENTENCE'S OWN ARITHMETIC. Four below 30 mmHg are entry
    # errors; every reading above 140 is abnormal and real and must stay, and
    # the whole point of the sentence is that no ±3 SD rule separates them.
    impossible = rng.choice(n, size=4, replace=False)
    df.loc[df.index[impossible], "sbp"] = [28.0, 12.0, 0.0, 22.0]

    # ── §A1.2 · temporal implausibility ──────────────────────────────────────
    #
    # Seeded on named patients rather than at random, so the companion `.md` can
    # state exactly which rows a detector must find.
    grown = df.index[(df["patient_id"] == "PT0007") & (df["visit_date"] > df.loc[
        df["patient_id"] == "PT0007", "visit_date"].min())]
    df.loc[grown, "height_cm"] = df.loc[grown, "height_cm"] + 9.0

    lost = df.index[(df["patient_id"] == "PT0021")]
    lost_later = lost[1:]
    df.loc[lost_later, "weight_kg"] = (df.loc[lost_later, "weight_kg"] * 0.66).round(1)
    # ...and move that visit inside 30 days, because ±30% over six months is a
    # trajectory and over three weeks is a transcription error.
    first_visit = pd.to_datetime(df.loc[lost[0], "visit_date"])
    df.loc[lost_later, "visit_date"] = [
        (first_visit + timedelta(days=21 * (i + 1))).date().isoformat()
        for i in range(len(lost_later))]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4 · survey_instrument — 300 × 40 Likert items
# ─────────────────────────────────────────────────────────────────────────────

REVERSE_CODED = ("item_05", "item_11", "item_17", "item_23",
                 "item_29", "item_34", "item_38", "item_40")


def survey() -> pd.DataFrame:
    """One 40-item instrument, five-point Likert, eight items reverse-coded.

    The reverse-coded items are genuinely reverse-coded — they load negatively
    on the same latent trait — so their correlations with the rest are negative
    and an inference from correlation WOULD get them right here. That is
    precisely why the app must not do it: the same evidence appears when two
    subscales measure opposing constructs, and there is no way to tell from the
    numbers. Reverse-coding needs the codebook, so it is asked.

    The response scale is 1–5 with labels the instrument declares. Its
    ordinality comes from the instrument, not from five values happening to
    recur — which is why the encoding is row-local rather than a distribution
    the app has to learn.
    """
    rng = np.random.RandomState(1177)
    n, n_items = 300, 40

    trait = rng.normal(0.0, 1.0, size=n)
    loading = rng.uniform(0.35, 0.85, size=n_items)
    rows = {"respondent_id": [f"R{i + 1:03d}" for i in range(n)]}
    rows["age"] = rng.randint(18, 82, size=n)
    rows["sex"] = rng.choice(["F", "M"], size=n)
    rows["education"] = rng.choice(
        ["High school", "Some college", "Bachelors", "Graduate"], size=n)

    items = {}
    for j in range(n_items):
        name = f"item_{j + 1:02d}"
        sign = -1.0 if name in REVERSE_CODED else 1.0
        latent = sign * loading[j] * trait + rng.normal(0, 0.75, size=n)
        # Cut the latent into five ordered bands. Unequal cut points, because a
        # real instrument's response distribution is not uniform.
        cuts = np.quantile(latent, [0.12, 0.34, 0.63, 0.86])
        items[name] = (np.digitize(latent, cuts) + 1).astype(int)
    rows.update(items)

    # An external outcome, not a score computed from the items — a total built
    # out of the items would be the target leaking through every feature.
    logit = 0.9 * trait + rng.normal(0, 1.0, size=n)
    rows["sought_support"] = (logit > 0.6).astype(int)

    df = pd.DataFrame(rows)
    # A handful of skipped items, which is what a real return looks like.
    for name in ("item_07", "item_19", "item_31"):
        idx = rng.choice(n, size=int(0.04 * n), replace=False)
        df.loc[idx, name] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5 · genomics_expression — 60 × 500 counts
# ─────────────────────────────────────────────────────────────────────────────

def genomics() -> pd.DataFrame:
    """Raw counts, three batches, library sizes varying about two-fold.

    Counts, not concentrations: integers with a variance that grows with the
    mean. The library sizes differ across samples on purpose, because that is
    exactly what makes a normalization choice necessary — and the pack's job
    here is to recognize the shape and DECLINE to pick one. CPM, TPM and VST
    answer different questions and are not interchangeable.

    p = 495 features against n = 60 samples, so p/n ≈ 8.25 — well into the
    regime where an unregularized fit is not merely optimistic but degenerate.
    """
    rng = np.random.RandomState(31337)
    n, n_genes = 60, 495

    batch = np.array(["B1"] * 20 + ["B2"] * 20 + ["B3"] * 20)
    condition = np.array((["control"] * 10 + ["case"] * 10) * 3)
    # Library size varies about two-fold, and varies WITH batch — which is why
    # a normalization that ignores batch and a batch correction that ignores
    # normalization are both incomplete.
    lib = rng.uniform(0.75, 1.55, size=n) * np.where(batch == "B2", 1.25, 1.0)

    base_expr = np.exp(rng.normal(2.0, 1.9, size=n_genes))
    de = rng.rand(n_genes) < 0.06                       # 6% differentially expressed
    effect = np.where(de, rng.normal(0.9, 0.35, size=n_genes), 0.0)
    batch_effect = rng.normal(0, 0.4, size=(3, n_genes))

    counts = np.zeros((n, n_genes), dtype=int)
    b_index = {"B1": 0, "B2": 1, "B3": 2}
    for i in range(n):
        mu = (base_expr
              * lib[i]
              * np.exp(effect * (1.0 if condition[i] == "case" else 0.0))
              * np.exp(batch_effect[b_index[batch[i]]]))
        # Negative binomial via a gamma-Poisson mixture: overdispersion is the
        # defining property of count data and the reason a Gaussian model of raw
        # counts is wrong before any normalization question is reached.
        shape = 1.0 / 0.35
        lam = rng.gamma(shape, mu / shape)
        counts[i] = rng.poisson(lam)

    frame = {
        "sample_id": [f"GS{i + 1:03d}" for i in range(n)],
        "batch": batch,
        "sex": rng.choice(["F", "M"], size=n),
        "age": rng.randint(30, 80, size=n),
    }
    for j in range(n_genes):
        frame[f"gene_{j + 1:04d}"] = counts[:, j]
    frame["condition"] = condition
    return pd.DataFrame(frame)


# ─────────────────────────────────────────────────────────────────────────────
# 6 · nhanes_dietary / nhanes_partial_design — the survey-design pair
#
# `GUIDED-058`. The three survey-design detectors were wired to the dietary
# pack, and `test_a_pack_names_what_it_will_look_for` binds each `LooksFor` to a
# finding id **by running the detectors against the fixture the pack was built
# for**. `dietary_recalls.csv` carries no design variables, so three real
# capabilities could not be promised: the registry cannot name a detector its
# fixture never triggers.
#
# **Two files, and they have to be two.** `partial_design` fires when a weight
# has no strata or PSU; `lonely_psu` fires when both are present and a stratum
# holds one. The preconditions are exact negations, so no single table can
# exercise both — which is why the key-match test now takes a TUPLE of fixtures
# per pack rather than one.
#
# Small on purpose (120 rows). They exist to trigger detectors, not to be
# analyzed, and the companion `.md` files say so.
# ─────────────────────────────────────────────────────────────────────────────

def _nhanes_core(n: int, seed: int) -> dict:
    """SEQN and the DR1T nutrients, in grams, reconstructing cleanly.

    Units are internally consistent so `atwater_finding` stays SILENT here: its
    promise is already kept by `dietary_recalls.csv`, and a second fixture
    firing it would test nothing new while making these files about two things.
    """
    rng = np.random.default_rng(seed)
    protein = rng.gamma(9, 9, n)
    carb = rng.gamma(9, 28, n)
    fat = rng.gamma(7, 11, n)
    alcohol = rng.gamma(1, 4, n)
    return {
        "SEQN": np.arange(1, n + 1),
        "DR1TKCAL": 4 * protein + 4 * carb + 9 * fat + 7 * alcohol,
        "DR1TPROT": protein,
        "DR1TCARB": carb,
        "DR1TTFAT": fat,
        "DR1TALCO": alcohol,
        "WTDRD1": rng.gamma(4, 6000, n),
        "WTMEC2YR": rng.gamma(4, 6000, n),
    }


def nhanes_dietary(n: int = 120) -> pd.DataFrame:
    """A complete design with one lonely stratum.

    Must surface: `pack::dietary::survey_weights` (the dietary weight beside the
    examination weight) and `pack::dietary::lonely_psu` — stratum 999 holds a
    single PSU, so its variance contribution is undefined rather than small.
    Must NOT surface: `pack::dietary::partial_design`, because the design is
    complete.
    """
    rng = np.random.default_rng(7)
    frame = _nhanes_core(n, seed=7)
    strata = rng.integers(100, 108, n)
    psu = rng.integers(1, 3, n)
    strata[:9] = 999                       # one stratum...
    psu[:9] = 1                            # ...with one PSU in it
    frame["SDMVSTRA"] = strata
    frame["SDMVPSU"] = psu
    return pd.DataFrame(frame)


def nhanes_kilojoules(n: int = 120) -> pd.DataFrame:
    """The same table with its energy column in kilojoules.

    `GUIDED-068` is why this exists. Once the Atwater check learned to prefer
    the gram columns over the percent-of-energy columns beside them,
    `dietary_recalls.csv` PASSED — which is correct and left
    `pack::dietary::atwater` promised by the dietary pack's hover and emitted
    by no fixture. A promise nobody keeps is the app announcing it will look
    for something it will not, so the promise needed a table with a real unit
    error in it.

    Must surface: `pack::dietary::atwater` at verdict `energy_in_kj`, ratio
    ≈ 4.184. Must NOT surface: any design finding — the weights and the strata
    are here so the table is NHANES-shaped, and they are complete.
    """
    frame = _nhanes_core(n, seed=9)
    frame["DR1TKCAL"] = frame["DR1TKCAL"] * 4.184
    rng = np.random.default_rng(9)
    frame["SDMVSTRA"] = rng.integers(100, 108, n)
    frame["SDMVPSU"] = rng.integers(1, 3, n)
    return pd.DataFrame(frame)


def nhanes_partial_design(n: int = 120) -> pd.DataFrame:
    """The same weights with no strata and no PSU.

    Must surface: `pack::dietary::survey_weights` and
    `pack::dietary::partial_design`. Must NOT surface:
    `pack::dietary::lonely_psu`, which needs both columns to mean anything.
    """
    return pd.DataFrame(_nhanes_core(n, seed=8))


FIXTURES = {
    "metabolomics_untargeted": metabolomics,
    "dietary_recalls": dietary,
    "clinical_longitudinal": clinical,
    "clinical_risk": clinical_risk,
    "clinical_labs": clinical_labs,
    "survey_instrument": survey,
    "genomics_expression": genomics,
    "nhanes_dietary": nhanes_dietary,
    "nhanes_partial_design": nhanes_partial_design,
    "nhanes_kilojoules": nhanes_kilojoules,
}


def main() -> None:
    for name, build in FIXTURES.items():
        df = build()
        path = HERE / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{path.name:34s} {len(df):5d} × {len(df.columns):4d}  "
              f"{path.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
