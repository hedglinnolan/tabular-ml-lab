# Sullivan 2021 through the Estimate lane (SU-A)

Written 2026-10-18 by the SU-A builder, under `plans/SULLIVAN_PLAN.md`, and revised the same day
in its repair round under the orchestrator's three rulings on its open issues (the shared template
home, §1; acceptance 4 re-ruled, §6; the refits added after the ladder's numbers were seen, §3 and
§4). Branch `feat/sullivan-a`, from `feat/w1p2-int` at `712747ff`; no engine code changed
(ruling 6). The code and the plan file are in `research/replication/sullivan2021/engine/`; the
numbers below are read from its `results.json`, which `results.py` writes from the runs'
captures.

## TL;DR

- **Table 4 matched.** All four associations (fasting glucose, insulin, HbA1c, and BMI) came out
  of TurboTab at the paper's printed values: the same β, the same SE, and exactly the same number
  of people (6,832, 6,655, 14,795 and 15,278). The same exclusion rules reproduce the paper's
  whole participant flow line for line (39,156 people down to 16,913), counted outside the engine,
  whose own flow starts at each outcome's measured rows.
- **The nested models (Appendix Table 2) mostly matched.** Models 0 and 1 matched everywhere.
  Models 2 and 3 matched for insulin and HbA1c; for glucose they were off by one in the last
  printed digit (−3.6 (0.43) against −3.7 (0.44), −3.5 (0.44) against −3.6 (0.45)). With income
  in the three bands of the paper's Table 1 instead of as a number (the paper does not say which
  it used), every printed value of the appendix matches (Model 4 read as below), and Table 4
  still does. Those ladder refits with income in bands were added after the ladder's numbers,
  this glucose gap among them, had been seen, so their match was looked for, not predicted: it
  is evidence, not proof. The headline numbers stay income as a number, chosen before any run.
- **Model 4, as the paper describes it, does not reproduce.** "Model 2 plus medication" gives
  glucose −2.0 against the printed −1.1. "The Table 4 model plus medication" (a reading the blind
  R agent found after the stated one missed) gives −1.0 or −1.1 depending on income's form, and
  matches insulin and HbA1c exactly.
- **What mattered among the choices the paper left unstated:** the form of age (bands or a curve
  move glucose from −2.0 to −1.8 or −1.9 at printed precision) and the form of income (above).
  Dividing the weights by four, how "only water" is identified and the insulin assay crosswalk
  moved nothing at printed precision, nor did SAS's default variance multiplier (applied by
  neither pipeline) in Table 4 or the ladder. Every Table 4 option was in the plan file before any run;
  21 ladder refits (income's bands in Models 2–4, age's bands in Models 2 and 3, Model 3's energy
  from day 1) were added after the ladder's numbers were seen (§4).
- **The biggest swing is an alternative, not a replication choice:** leaving out people with
  diagnosed diabetes takes the glucose association from −2.0 to −0.2 (p = 0.36).
- **What the engine could not do:** stack the four survey cycles, derive a variable from several
  columns (current smoking, weekly activity minutes, medication use, mean energy), merge
  categories (race, education), reverse the rating's coding, keep a single code in an exclusion
  rule, show a flow that starts before the outcome, set the paper's p < 0.01, state an
  association without an effect's words, or name a variable by its label. Its file join works but
  is slow through the API (nine joins took about nine minutes). Each is an INBOX line in §7.

## 1. What ran

1. **Intake** (`intake.py`): 43 XPT files and their 43 codebook pages for 2011–2018, 162.7 MB,
   from wwwn.cdc.gov with no identifier in any request. `manifest.json` (committed) gives each
   file's URL, download date, size, SHA-256, the "First Published" and "Last Revised" dates its
   codebook page states, and the server's Last-Modified header. The data are in
   `research/data/nhanes/<cycle>/`, ignored by git.
2. **Prep** (`prep.py`), only what the engine cannot do (§7): each component's four cycles
   stacked, read with the engine's own XPT reader (pandas reads a SAS zero as 5.4e-79, which
   would have hidden the 20 zero-energy recalls); insulin taken from GLU_G in 2011–2012 and from
   INS_H, INS_I and INS_J after (the fasting weight is GLU's, which equals INS's row for row,
   checked); and one `derived.parquet` keyed by `SEQN` with the derived columns. Each component
   keeps only the variables the analysis reads; the engine could leave the rest out by their
   roles, so that is a convenience, not a gap.
3. **The table** (`run.py --template`): TurboTab opened the stacked DEMO file and joined the other
   nine files on `SEQN` through the API (`join_files`, each after its preview, all one-to-one,
   39,156 rows). Every run starts from a copy of that project's home, so its log holds the nine
   joins, recorded once and copied. The orchestrator accepted this shared home for this
   replication (2026-10-18) because of gap 7: nine joins through the API take about nine
   minutes, so a run that joined its own would exceed the ten-minute command budget. One run's
   bundle replays from its files and log alone (§10).
4. **The runs** (`run.py`): one project per outcome and model, each a journey
   (`turbotab.core.reference.journeys`' loop, recording client and capture) through the server's
   API in local mode with two workers, to its fit and its export. The plan file
   `sullivan2021.json` holds the paper as data: the joins, the readings (from the CDC
   codebooks), the exclusions, the survey design, the models and the variants. It is the swarm
   driver's first plan file.
5. **Checks** (no TurboTab code): `check_flow.py` counts the paper's flow from the prep's tables
   by the plan file's rules; `check_rows.py` refits every run from its analytic-rows CSV with
   numpy (survey-weighted least squares, Taylor-linearized SE from the textbook sandwich).

## 2. The decisions, stated and unstated

| Decision | The paper | How it was taken here | Measured shift |
|---|---|---|---|
| Cycles | 2011–2018 pooled (stated) | stacked in the prep (gap 1); `SDDSRVYR` the cycle column | — |
| Age < 20 | excluded (stated) | engine rule `RIDAGEYR` ≥ 20 | flow matches (−16,539) |
| Pregnancy | excluded (stated) | engine rule: `RIDEXPRG` 1 out; 3 ("cannot ascertain") and untested kept (my reading) | flow matches (−247) |
| No self-rating | excluded (stated) | engine rule `DBQ700` 1–5 (drops 7, 9 and blanks) | flow matches (−10) |
| No recall, one recall | excluded; "complete and reliable" recalls (stated) | engine rules `DR1DRSTZ` ≤ 1, `DR2DRSTZ` ≤ 1 (status 1; a one-value range is refused, gap 6) | flow matches (−2,982, −2,445) |
| Only water on a day | excluded (stated); how it was identified is not | engine rules `DR1TKCAL` ≥ 1, `DR2TKCAL` ≥ 1 (zero energy; NHANES energy is whole kcal, the smallest positive value 3 on day 1 and 2 on day 2) | flow matches (−20); options in §4 |
| Exposure | DBQ700 as a 1–5 score, poor = 1 (stated) | engine, a straight line on NHANES's codes (excellent = 1); the paper's β compared with −β (gap 5) | — |
| Model | survey-weighted linear regression, SAS 9.4 (stated) | engine: survey-weighted least squares, Taylor linearization, t on PSUs − strata (df 62) | — |
| Variance multiplier | not stated; SAS SURVEYREG's default `VADJUST=DF`, as recalled, multiplies the variance by (n − 1)/(n − p) | not applied (nor by SU-R's R) | SE larger by 0.003% to 0.13% (0.11% at most in Table 4 and the ladder); no Table 4 or ladder match changes; one §4 cell flips (§4) |
| Weights | WTSAF2YR for glucose and insulin, WTDR2D otherwise (stated) | engine survey answer | — |
| Weights across cycles | not stated | ÷ 4 (`SDDSRVYR` named) | zero, as derived (§4) |
| Covariates | age, sex, race (5 groups with non-Hispanic Asian), education (4), PIR, current smoking, recreational activity (moderate + 2 × vigorous minutes), BMI; BMI out of the BMI model (stated) | engine covariates; race and education merged and smoking and activity derived in the prep (gaps 3, 4) | — |
| Age's form | not stated | a straight line | material for glucose and HbA1c (§4) |
| PIR's form | not stated (Table 1 shows three bands) | a straight line | the bands reproduce glucose in M2, M3 and M4 as printed, in ladder refits added after the gap was seen (§3, §4) |
| Education groups | 4 (stated); which 4 not stated | DMDEDUC2 1–2, 3, 4, 5; 7 and 9 blank | not measured |
| Smoking, activity definitions | stated in words | SMQ020 = 1 and SMQ040 1–2; PAQ670 × PAD675 + 2 × PAQ655 × PAD660 a week, 0 when PAQ665 or PAQ650 is 2, refused and unknown blank | not measured (the complete-case n's match exactly) |
| Missing covariates | not stated | complete cases (SAS drops them) | n matches all four outcomes; options in §4 |
| Total energy (M3) | in M3 (stated); which recall days not stated | the mean of the two days | day 1 alone, a refit added after the ladder's numbers were seen: §4 |
| Medication (M4) | "medication(s) that affect risk factor" (stated); which variables not stated | DIQ050 = 1 (insulin) or DIQ070 = 1 (pills); 0 when DIQ010 is answered | not measured (RXQ_RX not fetched) |
| Insulin assay | not stated; the INS_H codebook gives a Roche→Tosoh crosswalk for 2011–2012 | as measured | §4 |
| Significance | p < 0.01, 99% intervals (stated) | not settable: the engine reports 95% (gap 10) | — |

Rows per outcome (Table 4 fits; the paper's n is the last column's):

| Outcome | measured and eligible | complete cases | dropped for a blank covariate | complete with a positive weight (n) |
|---|---|---|---|---|
| glucose | 7,936 | 7,180 | 756 | 6,832 |
| insulin | 7,732 | 6,998 | 734 | 6,655 |
| hba1c | 16,346 | 14,795 | 1,551 | 14,795 |
| bmi | 16,745 | 15,278 | 1,467 | 15,278 |

The zero-weight rows are the fasting subsample's participants with `WTSAF2YR` = 0. The engine
keeps them in the design as a domain; the paper's n, like the engine's estimate, counts only
positive weights.

## 3. Our numbers against the paper's

β is −1 × the engine's coefficient per raw DBQ700 step (ruling 3). Each match is at the paper's
printed precision (halves rounded away from zero). df is 62 in every fit (121 PSUs in 59 strata).

| Outcome | ours: −β (SE) | n | df | p | ours at the paper's precision | paper: β (SE) | paper n | paper p | β, SE, n match |
|---|---|---|---|---|---|---|---|---|---|
| Glucose (mg/dL) | −1.957 (0.419) | 6,832 | 62 | < 0.001 | −2.0 (0.4) | −2.0 (0.4) | 6,832 | <0.001 | yes, yes, yes |
| Insulin (μU/mL) | −0.561 (0.182) | 6,655 | 62 | 0.0031 | −0.6 (0.2) | −0.6 (0.2) | 6,655 | 0.003 | yes, yes, yes |
| HbA1c (%) | −0.0459 (0.0095) | 14,795 | 62 | < 0.001 | −0.05 (0.01) | −0.05 (0.01) | 14,795 | <0.001 | yes, yes, yes |
| BMI (kg/m², no BMI term) | −1.634 (0.086) | 15,278 | 62 | < 0.001 | −1.6 (0.1) | −1.6 (0.1) | 15,278 | <0.001 | yes, yes, yes |

| Outcome | Model | ours: −β (SE) | n | paper: β (SE) | β matches | SE matches |
|---|---|---|---|---|---|---|
| glucose | M0 | −2.740 (0.365) | 7,534 | −2.7 (0.37) | yes | yes |
| glucose | M1 | −4.057 (0.365) | 7,534 | −4.1 (0.36) | yes | yes |
| glucose | M2 | −3.647 (0.434) | 6,900 | −3.7 (0.44) | **no** | **no** |
| glucose | M3 | −3.544 (0.443) | 6,886 | −3.6 (0.45) | **no** | **no** |
| glucose | M4 as stated (M2 + medication) | −2.002 (0.338), p < 0.001 (paper p = 0.004) | 6,894 | −1.1 (0.36) | **no** | **no** |
| glucose | M4 as printed (Table 4 model + medication) | −1.047 (0.355), p = 0.0045 (paper p = 0.004) | 6,826 | −1.1 (0.36) | **no** | yes |
| insulin | M0 | −2.331 (0.219) | 7,335 | −2.3 (0.22) | yes | yes |
| insulin | M1 | −2.487 (0.213) | 7,335 | −2.5 (0.21) | yes | yes |
| insulin | M2 | −2.225 (0.229) | 6,723 | −2.2 (0.23) | yes | yes |
| insulin | M3 | −2.086 (0.230) | 6,709 | −2.1 (0.23) | yes | yes |
| insulin | M4 as stated (M2 + medication) | −1.851 (0.225), p < 0.001 (paper p = 0.03) | 6,717 | −0.4 (0.19) | **no** | **no** |
| insulin | M4 as printed (Table 4 model + medication) | −0.413 (0.190), p = 0.033 (paper p = 0.03) | 6,649 | −0.4 (0.19) | yes | yes |
| hba1c | M0 | −0.0774 (0.0094) | 16,346 | −0.08 (0.01) | yes | yes |
| hba1c | M1 | −0.1267 (0.0094) | 16,346 | −0.13 (—) | yes | — |
| hba1c | M2 | −0.0983 (0.0102) | 14,954 | −0.10 (—) | yes | — |
| hba1c | M3 | −0.0918 (0.0100) | 14,928 | −0.09 (—) | yes | — |
| hba1c | M4 as stated (M2 + medication) | −0.0564 (0.0074) | 14,943 | −0.03 (0.01) | **no** | yes |
| hba1c | M4 as printed (Table 4 model + medication) | −0.0262 (0.0073) | 14,784 | −0.03 (0.01) | yes | yes |

**Model 4.** The supplement states it as "Model 2 plus use of medication(s) that affect risk
factor". That reading misses all three printed values by far (glucose −2.0 against −1.1, insulin
−1.9 against −0.4, HbA1c −0.06 against −0.03). The Table 4 model plus the medication reproduces
insulin and HbA1c exactly, and glucose to −1.0 (printed −1.1), its p 0.0045 (printed 0.004). That
second reading was not found here: it reached this package from the blind R agent's note of
2026-10-18 (`estimate_candidates.md` §1), found after the stated one missed (the fifth of five
readings it tried), so its match is evidence of what the paper ran, not an independent result.

**PIR in the Table 1 bands.** The table below refits the same models with PIR in the paper's
Table 1 bands. Its Table 4 rows (`pir_bands`) were in the plan file before any run. Its ladder rows
(`pir_bands_M2`, `pir_bands_M3`, `pir_bands_M4` and `pir_bands_M4t4`) were not: they were added
after the ladder's numbers, glucose's Models 2 and 3 among them, had been seen (the timeline is in
§4).

| Outcome | Model | −β (SE) | n | paper: β (SE) | β matches | SE matches |
|---|---|---|---|---|---|---|
| glucose | M2 | −3.678 (0.438) | 6,900 | −3.7 (0.44) | yes | yes |
| glucose | M3 | −3.577 (0.446) | 6,886 | −3.6 (0.45) | yes | yes |
| glucose | M4 as stated | −2.011 (0.339), p < 0.001 | 6,894 | −1.1 (0.36) | **no** | **no** |
| glucose | M4 as printed | −1.056 (0.357), p = 0.0044 | 6,826 | −1.1 (0.36) | yes | yes |
| glucose | Table 4 | −1.980 (0.421) | 6,832 | −2.0 (0.4) | yes | yes |
| insulin | M2 | −2.231 (0.228) | 6,723 | −2.2 (0.23) | yes | yes |
| insulin | M3 | −2.093 (0.229) | 6,709 | −2.1 (0.23) | yes | yes |
| insulin | M4 as stated | −1.851 (0.224), p < 0.001 | 6,717 | −0.4 (0.19) | **no** | **no** |
| insulin | M4 as printed | −0.411 (0.190), p = 0.035 | 6,649 | −0.4 (0.19) | yes | yes |
| insulin | Table 4 | −0.562 (0.183) | 6,655 | −0.6 (0.2) | yes | yes |
| hba1c | M2 | −0.0994 (0.0101) | 14,954 | −0.10 (—) | yes | — |
| hba1c | M3 | −0.0929 (0.0099) | 14,928 | −0.09 (—) | yes | — |
| hba1c | M4 as stated | −0.0570 (0.0074), p < 0.001 | 14,943 | −0.03 (0.01) | **no** | yes |
| hba1c | M4 as printed | −0.0267 (0.0073), p < 0.001 | 14,784 | −0.03 (0.01) | yes | yes |
| hba1c | Table 4 | −0.0470 (0.0096) | 14,795 | −0.05 (0.01) | yes | yes |

So with PIR in its Table 1 bands every printed value of Table 4 and of the appendix's M0–M3
matches, and M4 matches under the Table 4 + medication reading (glucose −1.1 (0.36), p 0.0044).
PIR continuous was the plan file's answer before any run; it stays the headline. The bands are
reported as what they are: the unstated option that reproduces the printed table, found by
refits added after the gap was seen, so the match is a post hoc one.

## 4. The unstated decisions, measured

Each option is refit through the engine with everything else as the plan file answers it. Δ is
the option's −β and SE minus the plan file's answer for the same outcome and model; "printed
change" says whether the paper's rounding of β or SE would change.

**When each option was added** (times UTC on 2026-10-18, from the SU-A builder's transcript). The
plan file was first written at 10:43 with every Table 4 option below (age's three bands and
spline, PIR's bands, blanks as their own category with PIR continuous or banded, the weights not
divided, the two "only water" readings, the insulin crosswalk), both models of Model 4, and every
alternative of §5. The ladder's first numbers came back at 11:40 (Model 4 for glucose) and 11:49
(Models 0–3 for glucose, Model 2's −3.65 (0.43) among them). Seven variants, 21 runs, were added
after that, each marked † below: Model 3's energy from day 1 alone (`energy_day1`, 12:34); PIR's
bands and age's bands in Models 2 and 3 (12:40); and PIR's bands in both readings of Model 4
(12:46, 20 seconds after the bands' Model 2 results came back). Among them, PIR's bands give
glucose's printed β and SE in Models 2 and 3 and its printed β in Model 4 as printed, and day-1
energy gives glucose's printed SE in Model 3 (0.45): values the plan's answers miss.

| Decision | Option | Outcome (model) | −β (SE) | n | Δβ | ΔSE | printed β / SE change |
|---|---|---|---|---|---|---|---|
| age's form | three bands, 20–39, 40–59, 60+ | glucose (Table 4) | −1.775 (0.420) | 6,832 | +0.183 | +0.001 | **yes** / no |
| age's form | three bands, 20–39, 40–59, 60+ | insulin (Table 4) | −0.595 (0.177) | 6,655 | −0.033 | −0.005 | no / no |
| age's form | three bands, 20–39, 40–59, 60+ | hba1c (Table 4) | −0.0374 (0.0099) | 14,795 | +0.0085 | +0.0004 | **yes** / no |
| age's form | three bands, 20–39, 40–59, 60+ | bmi (Table 4) | −1.613 (0.086) | 15,278 | +0.021 | 0 | no / no |
| age's form | spline (k by the engine's rule) | glucose (Table 4) | −1.886 (0.414) | 6,832 | +0.071 | −0.006 | **yes** / no |
| age's form | spline (k by the engine's rule) | insulin (Table 4) | −0.561 (0.181) | 6,655 | +0.001 | −0.001 | no / no |
| age's form | spline (k by the engine's rule) | hba1c (Table 4) | −0.0431 (0.0097) | 14,795 | +0.0028 | +0.0002 | **yes** / no |
| age's form | spline (k by the engine's rule) | bmi (Table 4) | −1.572 (0.081) | 15,278 | +0.062 | −0.005 | no / no |
| PIR's form | Table 1 bands | glucose (Table 4) | −1.980 (0.421) | 6,832 | −0.022 | +0.002 | no / no |
| PIR's form | Table 1 bands | insulin (Table 4) | −0.562 (0.183) | 6,655 | −0.001 | +0.001 | no / no |
| PIR's form | Table 1 bands | hba1c (Table 4) | −0.0470 (0.0096) | 14,795 | −0.0011 | 0 | no / no |
| PIR's form | Table 1 bands | bmi (Table 4) | −1.630 (0.086) | 15,278 | +0.003 | 0 | no / no |
| PIR's form | continuous, blanks as their own category | glucose (Table 4) | −1.940 (0.381) | 7,534 | +0.017 | −0.038 | **yes** / no |
| PIR's form | continuous, blanks as their own category | insulin (Table 4) | −0.646 (0.183) | 7,335 | −0.084 | 0 | no / no |
| PIR's form | continuous, blanks as their own category | hba1c (Table 4) | −0.0504 (0.0092) | 16,346 | −0.0044 | −0.0003 | no / no |
| PIR's form | continuous, blanks as their own category | bmi (Table 4) | −1.624 (0.084) | 16,745 | +0.010 | −0.002 | no / no |
| PIR's form | bands, blanks as their own category | glucose (Table 4) | −1.962 (0.383) | 7,534 | −0.004 | −0.037 | no / no |
| PIR's form | bands, blanks as their own category | insulin (Table 4) | −0.647 (0.184) | 7,335 | −0.086 | +0.002 | no / no |
| PIR's form | bands, blanks as their own category | hba1c (Table 4) | −0.0514 (0.0093) | 16,346 | −0.0055 | −0.0003 | no / no |
| PIR's form | bands, blanks as their own category | bmi (Table 4) | −1.621 (0.084) | 16,745 | +0.013 | −0.002 | no / no |
| weights across cycles | not divided by 4 | glucose (Table 4) | −1.957 (0.419) | 6,832 | 0 | 0 | no / no |
| weights across cycles | not divided by 4 | insulin (Table 4) | −0.561 (0.182) | 6,655 | 0 | 0 | no / no |
| weights across cycles | not divided by 4 | hba1c (Table 4) | −0.0459 (0.0095) | 14,795 | 0 | 0 | no / no |
| weights across cycles | not divided by 4 | bmi (Table 4) | −1.634 (0.086) | 15,278 | 0 | 0 | no / no |
| only water | no exclusion | glucose (Table 4) | −1.959 (0.417) | 6,840 | −0.002 | −0.002 | no / no |
| only water | no exclusion | insulin (Table 4) | −0.560 (0.182) | 6,662 | +0.001 | 0 | no / no |
| only water | no exclusion | hba1c (Table 4) | −0.0459 (0.0095) | 14,809 | +0.0001 | 0 | no / no |
| only water | no exclusion | bmi (Table 4) | −1.635 (0.086) | 15,295 | −0.001 | 0 | no / no |
| only water | zero energy with plain water reported | glucose (Table 4) | −1.957 (0.419) | 6,832 | 0 | 0 | no / no |
| only water | zero energy with plain water reported | insulin (Table 4) | −0.561 (0.182) | 6,655 | 0 | 0 | no / no |
| only water | zero energy with plain water reported | hba1c (Table 4) | −0.0459 (0.0095) | 14,795 | 0 | 0 | no / no |
| only water | zero energy with plain water reported | bmi (Table 4) | −1.635 (0.086) | 15,279 | −0.001 | 0 | no / no |
| insulin assay | 2011–2012 put on the Tosoh scale | insulin (Table 4) | −0.584 (0.181) | 6,655 | −0.023 | −0.001 | no / no |
| Model 3's energy † | day 1 alone | glucose (M3) | −3.526 (0.447) | 6,886 | +0.018 | +0.003 | no / **yes** |
| Model 3's energy † | day 1 alone | insulin (M3) | −2.067 (0.229) | 6,709 | +0.020 | −0.001 | no / no |
| Model 3's energy † | day 1 alone | hba1c (M3) | −0.0916 (0.0100) | 14,928 | +0.0001 | +0.0001 | no / no |
| PIR's form (ladder) † | Table 1 bands | glucose (M2) | −3.678 (0.438) | 6,900 | −0.032 | +0.004 | **yes** / **yes** |
| PIR's form (ladder) † | Table 1 bands | insulin (M2) | −2.231 (0.228) | 6,723 | −0.006 | −0.001 | no / no |
| PIR's form (ladder) † | Table 1 bands | hba1c (M2) | −0.0994 (0.0101) | 14,954 | −0.0012 | −0.0001 | no / no |
| PIR's form (ladder) † | Table 1 bands | glucose (M3) | −3.577 (0.446) | 6,886 | −0.033 | +0.003 | **yes** / **yes** |
| PIR's form (ladder) † | Table 1 bands | insulin (M3) | −2.093 (0.229) | 6,709 | −0.007 | −0.001 | no / no |
| PIR's form (ladder) † | Table 1 bands | hba1c (M3) | −0.0929 (0.0099) | 14,928 | −0.0011 | −0.0001 | no / no |
| PIR's form (ladder) † | Table 1 bands | glucose (M4) | −2.011 (0.339) | 6,894 | −0.009 | +0.002 | no / no |
| PIR's form (ladder) † | Table 1 bands | insulin (M4) | −1.851 (0.224) | 6,717 | −0.001 | −0.001 | no / no |
| PIR's form (ladder) † | Table 1 bands | hba1c (M4) | −0.0570 (0.0074) | 14,943 | −0.0006 | −0.0001 | no / no |
| PIR's form (ladder) † | Table 1 bands | glucose (M4 as printed) | −1.056 (0.357) | 6,826 | −0.009 | +0.002 | **yes** / no |
| PIR's form (ladder) † | Table 1 bands | insulin (M4 as printed) | −0.411 (0.190) | 6,649 | +0.001 | +0.001 | no / no |
| PIR's form (ladder) † | Table 1 bands | hba1c (M4 as printed) | −0.0267 (0.0073) | 14,784 | −0.0005 | 0 | no / no |
| age's form (ladder) † | three bands | glucose (M2) | −3.440 (0.435) | 6,900 | +0.206 | +0.001 | **yes** / no ‡ |
| age's form (ladder) † | three bands | insulin (M2) | −2.238 (0.225) | 6,723 | −0.012 | −0.004 | no / no |
| age's form (ladder) † | three bands | hba1c (M2) | −0.0898 (0.0105) | 14,954 | +0.0085 | +0.0002 | **yes** / no |
| age's form (ladder) † | three bands | glucose (M3) | −3.340 (0.444) | 6,886 | +0.204 | +0.001 | **yes** / no |
| age's form (ladder) † | three bands | insulin (M3) | −2.105 (0.227) | 6,709 | −0.019 | −0.004 | no / no |
| age's form (ladder) † | three bands | hba1c (M3) | −0.0827 (0.0103) | 14,928 | +0.0091 | +0.0003 | **yes** / no |

- **Age** is the most material choice in Table 4: three bands move glucose by +0.18 and HbA1c by
  +0.008, each a change in the printed value; a spline moves them less but still changes both;
  in Models 2 and 3, three bands move glucose by +0.2. A straight line is what reproduces the
  paper.
- **PIR**: the bands move Table 4 by at most 0.02 (no printed change) and Models 2–3 glucose by
  −0.03 with SE +0.003 to +0.004, enough to change their printed values to the paper's; in M4 as
  printed they move glucose by −0.009, which takes its printed β from −1.0 to the paper's −1.1.
  The ladder's bands refits (†) were added after the gap they close was seen.
  Blanks as their own category (the missing-indicator method, which the engine applies to every
  covariate with a blank, and blocks and records under inference; kept by its "Keep it, recorded"
  exit) add 680 to 1,551 rows, move insulin by −0.08 and change glucose's printed β to −1.9; the
  paper's n's rule them out.
- **The weights' division by 4** shifts nothing, to every digit: a common factor on every weight
  cancels in β and in the linearized SE.
- **Only water**: no exclusion adds 7 to 17 rows and moves β by at most 0.002; "zero energy with
  plain water reported" (2 zero-energy days had no water) moves only BMI, by one row.
- **The insulin crosswalk** moves insulin by −0.02 (no printed change).
- **Model 3's energy** as the first day alone (†, added after the ladder's numbers were seen)
  moves glucose by +0.02 (no printed change in β; its SE rounds to the paper's 0.45 instead of
  0.44).
- **SAS's variance multiplier** (`VADJUST=DF`, as recalled: the variance times (n − 1)/(n − p),
  n the rows with a positive weight and p the model's coefficients), which neither the engine nor
  SU-R's R applies, makes every SE larger by 0.003% (Model 0, HbA1c) to 0.13% (blanks as a
  category with PIR banded, insulin); 0.11% at most in Table 4 and the ladder (Model 4 as printed,
  insulin). It changes no match with the paper in Table 4 or the ladder, and does not close
  glucose's Model 2 or 3 gap (0.43404 still rounds to 0.43, 0.44354 to 0.44). It flips one cell
  above (‡): with age in bands in Model 2, glucose's SE becomes 0.43520, which rounds to the
  paper's 0.44 while the plan's answer stays 0.43, so its printed SE would change. It is measured
  in `results.json` (`sas_df_adjusted`, per run and in all), not applied to any number here.
- **Missing covariates**: complete cases drop the rows in "Rows per outcome" (§2), and give the
  paper's n exactly for all four outcomes.

## 5. Alternatives worth discussing (not the paper's analysis)

| Alternative | Outcome (model) | −β (SE) | n | p | Δβ against the paper's model |
|---|---|---|---|---|---|
| Table 4 without BMI | glucose (Table 4) | −3.529 (0.450) | 6,886 | < 0.001 | −1.571 |
| Table 4 without BMI | insulin (Table 4) | −2.087 (0.228) | 6,709 | < 0.001 | −1.525 |
| Table 4 without BMI | hba1c (Table 4) | −0.0918 (0.0101) | 14,928 | < 0.001 | −0.0458 |
| diagnosed diabetes excluded | glucose (Table 4) | −0.173 (0.186) | 5,845 | 0.36 | +1.785 |
| diagnosed diabetes excluded | insulin (Table 4) | −0.477 (0.174) | 5,701 | 0.008 | +0.085 |
| diagnosed diabetes excluded | hba1c (Table 4) | −0.0111 (0.0059) | 12,699 | 0.067 | +0.0349 |
| diagnosed diabetes excluded | bmi (Table 4) | −1.518 (0.091) | 13,123 | < 0.001 | +0.115 |
| M2 plus energy alone | glucose (M2) | −3.665 (0.427) | 6,900 | < 0.001 | −0.018 |
| M2 plus energy alone | insulin (M2) | −2.226 (0.232) | 6,723 | < 0.001 | 0 |
| M2 plus energy alone | hba1c (M2) | −0.0983 (0.0101) | 14,954 | < 0.001 | −0.0001 |
| insulin on the log scale | insulin (Table 4) | −0.048 (0.011) | 6,655 | < 0.001 | — (another scale) |

- **BMI out** of the Table 4 model roughly doubles the glucose and HbA1c associations and nearly
  quadruples insulin's: most of Table 4's attenuation from the ladder's Model 3 is BMI.
- **Excluding diagnosed diabetes** (DIQ010 = 1) leaves glucose at −0.2 (SE 0.19, p 0.36) and
  HbA1c at −0.011 (p 0.07); insulin keeps most of its association. Table 4's glucose and HbA1c
  associations come mostly from people with diagnosed diabetes.
- **Energy on its own** beside Model 2 moves nothing (−3.67 against −3.65 for glucose): the
  paper's "did not meaningfully alter" holds when energy is tested alone too.
- **Insulin on the log scale**: −0.048 in mean log insulin per step (about 4.7% lower geometric
  mean per step), p < 0.001, against p 0.003 on the original scale.

## 6. The methods the engine wrote (Table 4, glucose)

Quoted from the export bundle's `methods.md` of `t4-glucose` (its "Reproducibility" paragraph
left out), each sentence as written, the marks after it. The marks follow SULLIVAN_PLAN §4
acceptance 4 as the orchestrator re-ruled it on 2026-10-18: the methods must misstate no decision
the paper stated; a stated decision they cannot state is traced to its engine gap; and TurboTab's
own reporting conventions are listed as TurboTab's, not the paper's. Marks: **[stated ✓]** a
decision the paper stated, said here; **[omitted: …, gap k]** a stated decision the text cannot
state, and the gap (§7) that keeps it out; **[TurboTab's: …]** a convention of TurboTab's that
the paper did not state; **[unstated: …]** our answer to a decision the paper left open;
**[misleads: …]** a true sentence that a reader of the paper would misread.

> _Each sentence below is the decision record's own, as it stood when this bundle was exported: answers changed before anything was seen are folded out, every change made after the estimates were seen is kept and marked, and a sentence that counts rows counts them as they stand. Sections follow STROBE and STROBE-nut; a paragraph that is not the record's was written by the analysis on these answers._
>
> ## Study design (STROBE 4)
>
> The analysis was declared for `inference`: associations are estimated with their uncertainty. **[stated ✓: associations with their uncertainty]**
>
> ## Participants (STROBE 6)
>
> `2,136` rows with `RIDAGEYR` below `20` or not recorded were excluded as younger than 20 (the paper excludes age < 20); **[stated ✓: age < 20]** `96` rows with `RIDEXPRG` outside `2`–`3` were excluded as pregnant at the exam (RIDEXPRG 1, a positive test; men, and women not tested, are kept); **[stated ✓: pregnant]** `4` rows with `DBQ700` outside `1`–`5` or not recorded were excluded as having no self-rating of the diet (no answer, 7 refused, or 9 don't know); **[stated ✓: no self-rating]** `807` rows with `DR1DRSTZ` above `1` or not recorded were excluded as having no reliable first-day recall (DR1DRSTZ 2, 4 or 5, or no recall); **[stated ✓: no recall; misleads: counted within the glucose rows, not the paper's 2,982, gap 8]** `1,080` rows with `DR2DRSTZ` above `1` or not recorded were excluded as having no reliable second-day recall (DR2DRSTZ 2, 4 or 5, or no recall); **[stated ✓: only one recall]** `1` row with `DR1TKCAL` below `1` or not recorded was excluded as reporting only water on the first recall day, read as zero energy (NHANES energy is in whole kcal); `8` rows with `DR2TKCAL` below `1` or not recorded were excluded as reporting only water on the second recall day, read as zero energy (NHANES energy is in whole kcal), `4,132` in all; under the survey design they leave the estimate but keep their strata and PSUs in the variance (a domain analysis). **[stated ✓: only water; unstated: read as zero energy]**
>
> ## Variables (STROBE 7)
>
> `LBXGLU` was chosen as the outcome; **[stated ✓: fasting glucose; omitted: its plain name, gap 14]** it is measured on `12,068` of `39,156` rows. `LBXGLU` was modeled as a `regression` task. Column roles were set for `29` columns: exposure `DBQ700`; covariates `RIDAGEYR`, `RIAGENDR`, `INDFMPIR` and 5 more; identifier `SEQN`; design columns `SDDSRVYR`, `SDMVPSU`, `SDMVSTRA` and 2 more; excluded `RIDRETH3`, `DMDEDUC2`, `RIDEXPRG` and 11 more. **[stated ✓: the covariates, by column; omitted: what `race5`, `educ4`, `smoke_now` and `pa_rec_min` are, gaps 3 and 4]** The 5 columns `SDDSRVYR`, `SDMVPSU` and 3 more, proposed as part of the survey design at medium confidence, below high, for one reason ("An NHANES survey design variable: a weight, stratum or sampling unit. Confirm it is part of the design."), were confirmed together as one group, after the evidence for their proposal was read. The 7 columns `RIDAGEYR`, `RIAGENDR` and 5 more, proposed as covariates at low confidence, below high, for one reason ("Not a nutrient or feature: read as a covariate until you say otherwise."), were confirmed together as one group, after the evidence for their proposal was read. `smoke_now` was confirmed as a covariate, on its own, after the evidence for its reading was read. `DR1TKCAL` was recorded as a day's energy in kcal. `DR2TKCAL` was recorded as a day's energy in kcal. The analysis estimates the total effect of `DBQ700` on `LBXGLU`, as a difference in the mean outcome per unit of `DBQ700`. **[TurboTab's: "the total effect", the only kind of estimand the engine has under inference (gap 13), where the paper estimates a cross-sectional association; stated ✓: per unit of a 1–5 score; omitted: the paper's reversed scale, gap 5]** Confirmed together, each as the question showed it: `pa_rec_min` was confirmed to hold amounts or counts; `RIDAGEYR` was confirmed to hold amounts or counts. Confirmed together, each as the question showed it: `DBQ700` was confirmed to hold amounts or counts; `race5` was confirmed to hold codes for categories; `educ4` was confirmed to hold codes for categories.
>
> Read from the values, no question asked: `SEQN` is the unit's identifier (`SEQN` is the NHANES respondent number; `39,156` whole numbers, each on one row, within a span of `40,796`: a measurement spread over so narrow a span would repeat about `18,790` times); `INDFMPIR` is an amount (one slope) (`501` values with decimals that fill their grid as a measurement's do (a code list's sit apart)); `BMXBMI` is an amount (one slope) (`516` values with decimals that fill their grid as a measurement's do (a code list's sit apart)). **[TurboTab's: the engine's readings of the values]**
>
> ## Data sources and measurement (STROBE 8)
>
> `DBQ.parquet` (`39,156` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `39,156` of the table's rows found a partner; `39,156` of its rows matched. The joined table has `39,156` rows. **[stated ✓: the files merged; omitted: the four cycles and their years, gap 1]** `DR1TOT.parquet` (`37,399` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `37,399` of the table's rows found a partner and `1,757` did not, their `2` new columns blank; `37,399` of its rows matched. The joined table has `39,156` rows. `DR2TOT.parquet` (`37,399` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `37,399` of the table's rows found a partner and `1,757` did not, their `3` new columns blank; `37,399` of its rows matched. The joined table has `39,156` rows. `GLU.parquet` (`12,795` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `12,795` of the table's rows found a partner and `26,361` did not, their `2` new columns blank; `12,795` of its rows matched. The joined table has `39,156` rows. `INS.parquet` (`12,795` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `12,795` of the table's rows found a partner and `26,361` did not, their `1` new columns blank; `12,795` of its rows matched. The joined table has `39,156` rows. `GHB.parquet` (`26,673` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `26,673` of the table's rows found a partner and `12,483` did not, their `1` new columns blank; `26,673` of its rows matched. The joined table has `39,156` rows. `BMX.parquet` (`37,399` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `37,399` of the table's rows found a partner and `1,757` did not, their `1` new columns blank; `37,399` of its rows matched. The joined table has `39,156` rows. `DIQ.parquet` (`37,606` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `37,606` of the table's rows found a partner and `1,550` did not, their `1` new columns blank; `37,606` of its rows matched. The joined table has `39,156` rows. `derived.parquet` (`39,156` rows) was joined one-to-one to the table (`39,156` rows) on `SEQN`, keeping every row of the table; `39,156` of the table's rows found a partner; `39,156` of its rows matched. The joined table has `39,156` rows. The table was read through the `dietary` lens. **[TurboTab's: the lens]**
>
> ## Quantitative variables (STROBE 11)
>
> `RIDAGEYR` entered the models as a straight line; `INDFMPIR` entered the models as a straight line; `BMXBMI` entered the models as a straight line; `pa_rec_min` entered the models as a straight line. **[unstated: age, PIR, BMI and activity as straight lines]**
>
> ## Statistical methods (STROBE 12)
>
> The estimates describe the surveyed population: rows were weighted by `WTSAF2YR` divided by the number of cycles pooled in `SDDSRVYR`, and standard errors were estimated by Taylor series linearization over `SDMVPSU` nested within `SDMVSTRA` (`121` PSUs in `59` strata), first-stage units taken as sampled with replacement, with t intervals on the PSUs minus the strata that hold the analysis rows. **[stated ✓: WTSAF2YR, strata and PSUs; unstated: ÷ 4; omitted: which cycles and how many, gap 1]** Rows missing any predictor were dropped (a complete-case analysis): `7,180` of `7,936` rows remain; complete cases give unbiased coefficients only if whether a value is missing does not depend on the outcome, given the predictors (Groenwold et al. 2012; White & Carlin 2010); under the survey design they leave the estimate but keep their strata and PSUs in the variance (a domain analysis). **[unstated: complete cases; TurboTab's: the assumption's sentence]** No rows were held out: every analyzed row estimates the coefficients, and no cross-validated score is reported under inference: the declared model is reported by its estimates. **[TurboTab's, and true: no split]** For the effect of `DBQ700`, by the disjunctive cause criterion: `RIDAGEYR`, `RIAGENDR`, `INDFMPIR`, `smoke_now` and `pa_rec_min` are confounders, adjusted for. **[TurboTab's: the disjunctive cause criterion's words and "confounders", a causal reading; the paper adjusted without saying why]** For the effect of `DBQ700`, by the disjunctive cause criterion: `BMXBMI` is a confounder, adjusted for. For the effect of `DBQ700`, by the disjunctive cause criterion: `race5` and `educ4` are confounders, adjusted for. One model family was chosen: linear regression. **[stated ✓: survey-weighted linear regression]** For the surveyed population, least squares was weighted by `WTSAF2YR`, with standard errors by Taylor linearization over the survey design.
>
> The estimate of `DBQ700` is reported across a declared sequence of models fit on all 7,180 analyzed rows: unadjusted; Model 2, the primary, adjusted for `RIDAGEYR`, `RIAGENDR`, `INDFMPIR`, `BMXBMI`, `race5`, `educ4`, `smoke_now` and `pa_rec_min`. **[TurboTab's: the unadjusted model, and the label "Model 2, the primary" for the Table 4 model, which is not the paper's Model 2 (age, sex, race, education and PIR); misleads: the estimate uses the 6,832 rows with a positive weight, gap 9]** Only the exposure's estimates are shown as effects; every other coefficient is listed apart as an adjustment term, not an effect estimate (Westreich & Greenland 2013, Am J Epidemiol 177:292). Each model is design-based: weighted by `WTSAF2YR`, with Taylor-linearized intervals over the survey's strata and PSUs, so the estimates describe the surveyed population. Sensitivity to unmeasured confounding is reported by the E-value for the estimate and for the confidence limit nearer the null, the difference standardized by the outcome's design-weighted standard deviation in the surveyed population, never as a pass or a fail. **[TurboTab's: the E-value]**
>
> The analysis plan recorded above was declared in TurboTab before any estimate was displayed (SHA-256 `f5f401dfa234`); **[TurboTab's: the plan lock]** every later change is marked as made after the estimates were seen.

Against acceptance 4 as re-ruled:

- **Misstated:** none of the paper's stated decisions is given another value. The exclusions, the
  outcome, the 1–5 score, the weights, the design, the covariates and the model are as the paper
  states them. Two of TurboTab's conventions read against the paper's own words, and a reader
  holding the paper could take either for a misstatement: "the total effect" of `DBQ700` (the
  paper reports cross-sectional associations, as this text's first sentence says; the engine has
  no associational estimand under inference, gap 13), and "Model 2, the primary" for the Table 4
  model (the paper's Model 2 is the ladder's age, sex, race, education and PIR model, without BMI,
  smoking or activity). They are listed below as TurboTab's.
- **Stated by the paper, not stated here, each traced to its gap:** the years pooled (gap 1:
  `methods/cycles.py` has a methods sentence naming the cycles a stack holds, `stack_sentence`,
  but no route reaches the stack, so the text says only "the number of cycles pooled in
  `SDDSRVYR`"); what `race5`,
  `educ4`, `smoke_now` and `pa_rec_min` are (gaps 3 and 4: the race and education groupings, the
  smoking definition and the activity formula were derived in the prep, so no decision records
  them); that the paper's rating runs the other way (gap 5); the paper's p < 0.01 and 99%
  intervals (gap 10); and the variables' plain names, such as "fasting plasma glucose" for
  `LBXGLU` (gap 14: the methods name columns; this replication imported no codebook, and an
  imported codebook's labels feed only the readings' guesses, not the methods).
- **TurboTab's, not the paper's:** "the total effect" and the label "Model 2 (primary)" (above);
  the unadjusted model beside the primary; "by the disjunctive cause criterion" and "confounders"
  (the engine needs the causal answers to fit; the paper adjusted without saying why); the
  complete-case assumption's sentence; the no-split sentence; the E-value; the dietary lens; the
  engine's readings of the values; and the plan lock.
- **What misleads** (true of the engine's rows, misread against the paper): the exclusion counts
  are within the 12,068 people whose glucose was measured, so they are not the paper's flow
  (gap 8); and "fit on all 7,180 analyzed rows" counts the 348 zero-weight rows that the estimate
  (n 6,832) does not use (gap 9).

## 7. Engine gaps

Each prep step and each workaround, with its INBOX line. None stopped a number (ruling 6's
exception did not arise).

1. **Stacking survey cycles.** No decision or route reaches `methods/cycles.py`'s
   `stack_cycles`; the prep concatenated each component's four files.
   - [data-in, cycles] Stacking NHANES cycles has no route: `methods/cycles.stack_cycles` is engine only, so the Sullivan 2021 replication stacked 2011–2018 in its prep — research/replication/sullivan2021 (SU-A)
2. **A per-cycle source.** 2011–2012 ships insulin in GLU_G and later cycles in INS_H–J; a
   stack's per-cycle rename would carry it (`stack_cycles` has `renames`), with no route.
   - [data-in, cycles] A component whose file changes name across cycles (insulin: GLU_G, then INS_H–J) needs a per-cycle rename in the stack; none is reachable — research/replication/sullivan2021 (SU-A)
3. **Derivations from several columns.** Current smoking (SMQ020, SMQ040), weekly recreational
   minutes (six PAQ items), glucose-lowering medication (DIQ050, DIQ070), the two days' mean
   energy, the plain-water flag and the insulin crosswalk were all derived in the prep, so the
   methods cannot state how.
   - [decisions] No decision derives a column from several (current smoking from SMQ020 and SMQ040; weekly activity minutes from six PAQ items; a medication flag from DIQ050 or DIQ070; a two-day mean); the Sullivan 2021 replication derived six in its prep and its methods cannot state them — research/replication/sullivan2021 (SU-A)
4. **Merging categories.** Race (RIDRETH3 1 and 2 as Hispanic) and education (DMDEDUC2 1 and 2;
   7 and 9 blank) were merged in the prep. PIR's bands needed no merge: the engine's
   `categories` form cut it.
   - [decisions] No decision merges a coded column's levels (RIDRETH3 1+2 as Hispanic; DMDEDUC2 1+2, with 7 and 9 as blanks); the Sullivan 2021 replication merged both in its prep — research/replication/sullivan2021 (SU-A)
5. **Reversing a coded exposure.** DBQ700 codes excellent = 1; the paper scores poor = 1. The
   engine fits the raw codes, and the report compares −β.
   - [decisions, estimand] No recode reverses an ordinal exposure's scale (DBQ700: excellent = 1, the paper's poor = 1), so the estimate and its methods are per NHANES code — research/replication/sullivan2021 (SU-A)
6. **Keeping one code.** A range rule with `low` = `high` (keep `DR1DRSTZ` = 1) is refused as
   `empty_range`, whose only exit drops the rule. A one-sided bound (`at most 1`) does it here
   because no code lies below 1; a code in the middle of a list could not be kept.
   - [decisions, exclusions] An exclusion rule cannot keep one code: low = high is refused as empty_range and its only exit drops the rule (DR1DRSTZ = 1 was kept as "at most 1", true only because no code lies below 1) — research/replication/sullivan2021 (SU-A)
7. **The join's speed.** `assembly.plan`'s join guards each side's identifier inside the ON
   clause of a LEFT JOIN, which keeps DuckDB from hashing it (its plan is a blockwise
   nested-loop join, checked with EXPLAIN, for a float and an integer `SEQN` alike), and each
   ingest re-applies every join so far. The nine joins over 39,156 rows took 518.6 s, from 19.9 s
   for the first to 99.4 s for the ninth. A project per fit would spend about nine minutes
   joining, so every run copies one joined home. The replay (§10), which rebuilds the same
   joined table from the log in a fresh home, then fits and exports, took 27.7 s in all, so the
   cost lies in the API's per-join path rather than in one pass of the nine joins (not profiled).
   - [data-in, performance] join_files runs as a blockwise nested-loop join (the identifier guards sit in a LEFT JOIN's ON clause) and every ingest re-applies every join: nine NHANES files on SEQN over 39,156 rows took 518.6 s, 19.9 s to 99.4 s each — core/assembly.py::plan (Sullivan SU-A)
8. **The flow before the outcome.** The cohort's flow starts at the rows whose outcome is
   measured, so the paper's flow over the whole sample (39,156 → 16,913) is not one the engine
   shows; `check_flow.py` counts it outside the engine. The methods' exclusion counts are within
   each outcome's rows, which a reader of the paper would misread.
   - [export, flow] The participant flow begins at the outcome's measured rows, so a paper's flow before any outcome (Sullivan 2021: 39,156 → 16,913, then each outcome's n) cannot be shown, and the methods' exclusion counts read as a different flow — research/replication/sullivan2021 (SU-A)
9. **Zero weights in the methods' counts.** "Fit on all 7,180 analyzed rows" while the estimate
   uses the 6,832 with a positive weight; the fit's caption says so, the methods do not.
   - [export, methods] The methods say the models were "fit on all 7,180 analyzed rows" when 348 of them carry a zero survey weight and the estimate's n is 6,832; the inference caption says it, the methods sentence does not — research/replication/sullivan2021 (SU-A)
10. **The significance level.** No decision sets the test level or the interval's (the paper:
    p < 0.01 and 99% intervals); every interval is 95%.
    - [inference] No decision sets the significance level or the interval level (Sullivan 2021 declares p < 0.01 and reports 99% intervals); the methods cannot state it — research/replication/sullivan2021 (SU-A)
11. **Adjusted means per rating.** Table 4's adjusted means per rating level (Poor → Excellent)
    were not attempted: no survey stage gives an ordinal-as-linear exposure's means per level
    (`methods/trials.py`'s are for trial arms).
    - [inference, survey] No adjusted (predictive-margin) means per level of an exposure fit as a line under a survey design, as Sullivan 2021's Table 4 prints for each rating — research/replication/sullivan2021 (SU-A)
12. **The missing-indicator option's words.** Its keep exit attests "a blank here means not
    asked, or the covariates are baseline ones in a randomized trial", neither true of income
    nonresponse; it applies to every covariate with a blank; and the single-fill sentence says
    "imputed in each training fold" under Estimate, where nothing is held out.
    - [missing] Under inference the missing-indicator keep exit's attestation names only "not asked" or a trial's baseline (not nonresponse), the method cannot be limited to one column, and its sentence says "imputed in each training fold" when no rows are held out — research/replication/sullivan2021 (SU-A)
13. **No associational estimand.** Under inference the estimand is an effect, whose kinds are
    total, direct, complier and per protocol (`decisions.py`'s `EffectKind`), so a
    cross-sectional association's methods say "the total effect of `DBQ700` on `LBXGLU`" and
    name every covariate a confounder "by the disjunctive cause criterion". The paper estimates
    associations.
    - [inference, estimand] No associational estimand under inference: EffectKind offers only total, direct, complier and per_protocol, so a cross-sectional association (Sullivan 2021) is written as "the total effect of DBQ700 on LBXGLU" with its covariates as confounders by the disjunctive cause criterion — research/replication/sullivan2021 (SU-A)
14. **Plain names.** The methods name each variable by its column (`LBXGLU`, `DBQ700`). This
    replication imported no codebook; an imported codebook's labels feed only the readings'
    guesses (`readings.codebook_label`, read by the readings' ask alone), not the methods.
    - [export, methods] The methods name each variable only by its column (LBXGLU, never "fasting plasma glucose"); an imported codebook's labels feed only the readings' guesses (readings.codebook_label), so no sentence can give a variable's plain name — research/replication/sullivan2021 (SU-A)

**Not gaps, but noted.** The energy screens on `DR1TKCAL` and `DR2TKCAL` waited for each day
count to be recorded (the leash working; answered from the codebook). The engine names every
primary "Model 2 (primary)" and adds an unadjusted model beside it, which collides with a paper's
own model numbers (§6 lists both as TurboTab's). `journeys.Run.source_of` credits `set_forms`,
`set_estimand` and `set_clusters` to the acceptance drivers whenever any journey posts them;
`run.py` overrides it (an open issue in a file this package does not own).

## 8. The runs

Read from `results.json`'s `totals` and `totals.groups`, which `results.py` computes from the
captures (medians by `statistics.median`, so an even group's is the mean of its middle two).

| Group | Projects (fits) | Seconds, total | Seconds, median |
|---|---|---|---|
| Table 4 | 4 (8) | 173 | 43 |
| Appendix Table 2, M0–M4 as stated | 15 (30) | 627 | 41 |
| M4 as printed (Table 4 model + medication) | 3 (6) | 139 | 45 |
| unstated decisions' refits | 54 (108) | 2,242 | 41 |
| alternatives | 11 (22) | 443 | 41 |
| all | 87 (174) | 3,624 | 41 |

Each project fits two models, the primary and the unadjusted model the engine sets beside it on the same rows; every one of the 87 exports returned 200. The joined template took 518.6 s, once (DBQ 19.9 s, DR1TOT 26.9 s, DR2TOT 40.9 s, GLU 45.0 s, INS 65.9 s, GHB 56.8 s, BMX 73.6 s, DIQ 90.2 s, derived 99.4 s). Runs went in batches of eight, each one command under ten minutes, one local server at a time with two workers. `check_rows.py` refits every run from its CSV: the largest relative difference from the engine's β or SE is 3.5e-11, and every n matches. The seconds are the final pass's: the whole set ran twice, the first pass's exclusion reasons reading wrongly in the methods ("excluded as adults"), its numbers identical.

## 9. Where the task's premise was wrong

- **Merging by SEQN is an engine step**, not a prep step: `join_files` joins an added file on an
  identifier. It went through the engine (gap 7 is its speed).
- **Counting reliable recalls needs no derivation:** two range rules on the status columns keep
  exactly the people with two reliable recalls, and the flow matches line for line.
- **"Only water" needs no flag:** zero energy on a day is a range rule on the day's energy, and it
  excludes exactly the paper's 20. (The plain-water flag of §4's option is a derivation.)
- **PIR's bands are not a category merge:** the engine's `categories` form cuts them, blocked and
  recorded for a confounder in three groups and kept by its exit.
- **The cycle column** is NHANES's own `SDDSRVYR`; the prep added none.

## 10. Not run, and doubts

- Not run: the comparison with the blind R tables (SU-R, not read, by design; the verifier made
  it); a replay of any bundle but `t4-glucose`; a medication flag from RXQ_RX; other education or
  race groupings, pregnancy code 3 excluded, other smoking or activity definitions, activity's
  form; adjusted means. No test suite ran: no engine code changed.
- The first dry run joined its nine files inside the run and took 667 s, over the ten-minute
  command rule; every later command stayed under it.
- **One bundle replays** (the orchestrator's ruling on the shared template). From this worktree
  at `7d5abb6e`, whose `turbotab/` is the bundle's `68090ef8`'s:
  `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py -m turbotab.replay research/data/derived/engine/bundles/t4-glucose.zip --home <an empty scratch folder>`.
  It printed "Reproduced." and exited 0, in 27.7 s: all 10 input files match the SHA-256 their
  record holds; the 32 decision records replayed as recorded; the model matrix (7,180 rows by 14
  columns) is identical, its Parquet SHA-256 too; 212 of 212 estimates reproduced, the largest
  difference 0 (tolerance 1e-12); the result stages' keys are the same; and `methods.md`,
  `methods.json` and `methods.txt` differ only in the source-revision line (`68090ef8fa11`
  against `7d5abb6e370d`). So a run whose log holds the template's nine joins, copied, replays
  from its files and log alone. The verifier ran the same replay from a scratch worktree at
  `712747ff`, with the same result.
