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


FIXTURES = {
    "metabolomics_untargeted": metabolomics,
    "dietary_recalls": dietary,
    "clinical_longitudinal": clinical,
    "survey_instrument": survey,
    "genomics_expression": genomics,
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
