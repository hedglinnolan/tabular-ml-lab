# Pre-registration — the real NHANES drive, 2026-08-04

**Written before the product owner drove the app, and committed before he reported.** He asked to
drive a real dataset and give pointed feedback on how the app serves a specific user. That feedback
is worth more if his observations and the adjudicator's are independent, so this file records what
the adjudicator saw in a pre-flight **and did not tell him.**

The idiom is this repository's own: `VALUE_CHECK_PREREG.md` was frozen before the routing measurement
and stayed unedited even when it turned out ambiguous at an edge, and the ambiguity was adjudicated
in a separate file rather than by touching the prereg. **Same rule here. This file is not edited
after his report lands.** Anything it got wrong is corrected in the adjudication, not here.

---

## The file

`nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — **21,849 × 29**, 4.4 MB, uploads in ~1.4 s,
well inside the 64 MB ceiling. Fasting subsample, dietary variables, pooled across **nine** cycle
years, with six `imputed_*` boolean flags.

## What the app said

14 findings on the dietary lens; the attention stack pushed 5 and collapsed 0, so **no affordance
renders on this file either.**

| | |
|---|---|
| `critical` | 2 features with high missingness |
| `warning` ×9 | `imputed_bmi`, `imputed_bp_di`, `imputed_bp_sys`, `imputed_height`, `imputed_waist`, `imputed_weight`, `meds_chol`, `meds_hbp`, `gender` — "binary variable written as text" |
| `warning` | Nutrient associations need energy adjustment |
| `caution` | 16 features with outliers · 2 physiologic flags |
| `info` | 501 records report an implausible daily intake |

The dietary pack fired correctly on energy adjustment and on implausible intake, and the
high-missingness critical is right — `meds_hbp` is 71% null and `meds_chol` 79%.

## What the adjudicator noticed and did not say

**Recorded as predictions, not as findings.** Each is either confirmed or refuted after the drive.

### 1 · `SEQN` is not recognized as an identifier — and this is a rule tuned on a fixture meeting a real dataset

`SEQN` is the NHANES respondent sequence number: **21,849 distinct values on 21,849 rows**, `float64`.
`ml/dataset_profile.py`'s predicate is

```
unique_count == n and not is_bool_dtype and (not is_numeric or is_integer_dtype)
```

so a float that is unique on every row answers **`False`**, and `turbotab.identifiers.detect()`
returns nothing for this file. **Driven and confirmed before the drive.**

The dtype test is not an accident and its reason is recorded at `GUIDED-120`: dropping it flagged 88
continuous `mz_*` columns on `metabolomics_untargeted.csv`, which are the study's own predictors.
That reasoning was sound and it was measured on one fixture.

**What this file adds is the other side of the same test.** Of its 29 columns, exactly one is unique
on every row. The nutrient columns come close and none arrives — `fat_mon` 19,504 of 21,849,
`carb` 18,959, `sugar` 17,172, `protein` 15,129. So on this table *unique-per-row* discriminates
perfectly, and on the metabolomics table it does not.

**Prediction:** the app hands `SEQN` to the model as a predictor unless the user removes it. The
honest resolution is probably not a change to the predicate but a **lens-aware** one — an untargeted
assay is the one shape where unique-per-row floats are measurements, and the lens is already
recorded before Explore runs.

### 2 · The survey design is absent, and nothing says so

This is an NHANES file with **no `WTMEC2YR`, no `WTDRD1`, no `WTSAF2YR`, no `SDMVSTRA`, no
`SDMVPSU`.** Every published NHANES estimate is weighted, and this table cannot produce one.

The dietary pack holds three survey-design detectors — `survey_weights_finding`,
`partial_design_finding`, `lonely_psu_finding` — and every one fires on the **presence** of a design
variable in some incomplete state. **None fires on total absence.**

**Prediction:** the app says nothing about it. Whether it should is a genuine question rather than an
obvious defect: a table with no weights may be a convenience sample where weighting is meaningless.
But the columns here are literally `SEQN`, `kcal`, `protein`, `WTDRD1`-shaped nutrient variables from
NHANES, so the app has the evidence to ask.

### 3 · Nine cycles are pooled and nothing notices

`cycle_begin_year` carries **nine distinct values**. Pooling NHANES cycles requires dividing the
weights by the number of cycles; unpooled weights over pooled cycles inflate the effective N by
roughly nine and every standard error is then wrong.

**Prediction:** nothing fires. Related to 2 — with no weights present there is nothing to re-weight,
so the two findings compose into one: *this looks like pooled NHANES and carries no design variables
at all.*

### 4 · The six `imputed_*` flags are missingness indicators and are read as ordinary binaries

`imputed_bmi` is exactly the missing-indicator method the clinical pack discusses, already computed
upstream, and it pairs with a `bmi` column in the same frame. The app reports all six as *"binary
variable written as text"* — they are `bool` dtype, not text, which is a second and smaller
observation about that message's wording.

**Prediction:** the app does not connect `imputed_bmi` to `bmi`. Not obviously a defect for
prediction, where the indicator is legitimate and observable at deployment; it is a defect if the
pair reaches a `set_missingness` question that asks how to impute a column already imputed.

---

## What was deliberately withheld from him, and why

Only one thing beyond the above: **nothing about the four observations.** He was told the file loads,
that it produces 14 findings, and that **nothing collapses on it so the L45/L46 attention work will
not show** — which is operational rather than a finding, and telling him saved a wasted drive.

If his report names any of the four independently, that is a stronger result than the adjudicator
finding it, and this file exists so that can be said honestly.
