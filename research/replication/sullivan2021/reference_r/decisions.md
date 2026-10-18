# Sullivan 2021, the blind R reference: decisions

Package SU-R of `docs/turbotab-next/plans/SULLIVAN_PLAN.md` (ruling 7). Written 2026-10-18.

## TL;DR

- **Inputs.** This reference was built from the paper's facts (`notes/replication/estimate_candidates.md`
  §1) and the CDC codebook pages alone, in R, without reading any TurboTab code.
- **Sample steps.** Every step matches the paper's count exactly except the last ("only water on
  a recall day"): it removes 18 here against the paper's 20. The other 2 are people with a day on
  which nothing at all was reported.
- **Table 4.** All four outcomes match the paper at its printed precision (β and SE). n matches
  exactly for glucose, insulin and HbA1c; BMI is one over (15,279 against 15,278), and that one
  person is from the water step.
- **The ladder.** M0–M3 match for all three outcomes. M4, as §1 defines it (M2 plus
  glucose-lowering medication), does not match for any outcome. Checked afterward, the paper's six
  printed M4 numbers and its two M4 p values are all reproduced by the Table 4 model plus
  medication. So §1's account of M4 is probably a misreading of the supplement. That needs a check
  against the supplement, which this package could not reach (its web scope is CDC only).
- **Code.** `download.R`, `prep.R`, `models.R` and `fit.R` run in about a minute in all. The
  numbers are in `results.json`, the file list in `manifest.json`.

## How to run

From the repository root, in order (each under a minute; the downloads are 163 MB, fetched once):

```
Rscript research/replication/sullivan2021/reference_r/download.R
Rscript research/replication/sullivan2021/reference_r/prep.R
Rscript research/replication/sullivan2021/reference_r/fit.R
```

R 4.6.1, survey 4.5, haven 2.5.5, jsonlite and digest, all already installed; nothing was installed.

## What the paper states, and how it is coded

| The paper states | Coded as | Codebook page |
|---|---|---|
| NHANES 2011–2012, 2013–2014, 2015–2016, 2017–2018, combined | DEMO of each cycle stacked (39,156 records); every other file merged onto it by SEQN | DEMO_G/H/I/J |
| Adults; age < 20 excluded | `RIDAGEYR >= 20` | DEMO_G (RIDAGEYR) |
| Pregnant excluded | `RIDEXPRG == 1` (positive lab test or self-reported pregnant at exam) excluded | DEMO_G (RIDEXPRG) |
| Exposure: "how healthy is your overall diet", poor = 1 to excellent = 5, one continuous score; 7 and 9 dropped | `rating = 6 - DBQ700` for DBQ700 in 1–5; missing, 7 and 9 excluded | DBQ_G (DBQ700: 1 = excellent ... 5 = poor) |
| Two complete and reliable recalls | `DR1DRSTZ == 1` and `DR2DRSTZ == 1` | DR1TOT_G, DR2TOT_G (DRxDRSTZ: 1 = reliable and met the minimum criteria) |
| Outcomes: fasting glucose (mg/dL), insulin (μU/mL), HbA1c (%), BMI (kg/m²) | `LBXGLU`, `LBXIN`, `LBXGH`, `BMXBMI` | GLU, INS (GLU_G in 2011–2012), GHB, BMX |
| Weights: WTSAF2YR for glucose and insulin, WTDR2D for HbA1c and BMI | as stated | GLU_G, DR2TOT_G |
| Survey-weighted linear regression | `svyglm(..., family = gaussian())` on `svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ..., nest = TRUE)` over all 39,156 records, the analysis rows taken with `subset()` (a domain) | DEMO_G (SDMVPSU, SDMVSTRA) |
| Table 4 covariates: age, sex, race/ethnicity (5 groups, with non-Hispanic Asian), education (4 groups), PIR, current smoking, recreational physical activity, BMI; no BMI in the BMI model | as stated; the forms are decisions below | DEMO_G, SMQ_G, PAQ_G, BMX_G |
| Recreational activity = moderate + 2 × vigorous recreational minutes per week | `mod_min + 2 * vig_min`, each days per week × minutes per day | PAQ_G (PAQ650–PAD675) |
| Ladder: M0 none; M1 age, sex; M2 + race, education, PIR; M3 = M2 + smoking, activity, total energy; M4 = M2 + medication for the risk factor (glucose-lowering drugs) | as §1 states (but see "Where this differs from the paper", 2) | — |

## Decisions the paper left unstated

Each choice was made before any comparison with the paper's numbers. Where a later comparison
differs, the difference is reported below; no choice was changed after seeing it. The labels are
the ones cited in `prep.R`.

1. **D-age, the form of age.** Linear in years (`RIDAGEYR`, top-coded at 80 in the release). The
   paper names "age" with no bands or terms; a single linear term is the default reading of an
   unqualified continuous covariate. Page: DEMO_G (RIDAGEYR, "80 years of age and over").
2. **D-pir, the form of PIR.** Three bands, as the paper's Table 1 reports PIR: < 1.30,
   1.30–1.85 (both ends in the middle band), > 1.85, with > 1.85 as the reference. This is the only form
   the paper gives PIR in; the cut points are the 130% and 185% poverty-guideline lines. Missing
   PIR is not a category (see D-missing). Page: DEMO_G (INDFMPIR, 0–5, top-coded at 5).
3. **D-race, the five groups.** RIDRETH3 1 (Mexican American) and 2 (other Hispanic) are
   Hispanic; 3 non-Hispanic White (reference); 4 non-Hispanic Black; 6 non-Hispanic Asian;
   7 other race, including multiracial. RIDRETH3 is the variable with a non-Hispanic Asian code,
   so this is the only five-group reading that keeps Asian apart. Page: DEMO_G (RIDRETH3).
4. **D-educ, the four groups.** DMDEDUC2 1 and 2 (less than 9th grade; 9th–11th) are "less than
   high school"; 3 high school or GED; 4 some college or AA; 5 college graduate or above
   (reference); 7 and 9 missing. The adult education variable has five levels, and merging its
   two lowest is the standard four-group form. Page: DEMO_G (DMDEDUC2).
5. **D-smoke, current smoking.** Current = smoked at least 100 cigarettes in life (`SMQ020 == 1`)
   and smokes now every day or some days (`SMQ040` 1 or 2). Never smokers (`SMQ020 == 2`) and
   former smokers (`SMQ040 == 3`) are not current. Refused or don't know is missing. This is the
   CDC's definition of current cigarette smoking. Page: SMQ_G (SMQ020, SMQ040).
6. **D-pa, recreational activity.** Weekly minutes are days per week × minutes per typical day,
   vigorous from PAQ650/PAQ655/PAD660 and moderate from PAQ665/PAQ670/PAD675. "No" to the activity
   counts as 0 minutes. Refused or don't know (7/9 on the yes/no item, 77/99 on days, 7777/9999 on
   minutes) is missing. No cap is applied. Activity enters the model as one linear term in minutes
   per week, the quantity the paper defines; the paper does not say it was banded. Page: PAQ_G.
7. **D-energy, total energy in M3.** The mean of the two recall days' total energy
   (`(DR1TKCAL + DR2TKCAL) / 2`, kcal per day), one linear term. Every analytic person has two
   reliable days, and the paper restricts to them; using both days is the natural reading. Page:
   DR1TOT_G, DR2TOT_G (DRxTKCAL).
8. **D-med, glucose-lowering medication in M4.** 1 if now taking insulin (`DIQ050 == 1`) or
   diabetic pills to lower blood sugar (`DIQ070 == 1`), else 0. DIQ070 is skipped for people never
   told of diabetes or prediabetes, and a skip counts as "no". Refused or don't know with no "yes"
   is missing. These are the questionnaire's two glucose-lowering items. The prescription file
   (RXQ_RX) was not used. Page: DIQ_G (DIQ050, DIQ070).
   *Compared afterward:* the flag's weighted share by rating in the analytic sample (WTDR2D / 4) is
   10.4%, 11.6%, 9.8%, 7.2% and 5.9% for poor to excellent. §1 gives Table 1's range as 5.9–11.6%,
   the same two ends.
9. **D-weights, the division across cycles.** Each two-year weight is divided by 4, the number of
   cycles: the NHANES Analytic Guidelines' rule for combined two-year cycles, as I know it. DEMO_J
   refers "combining datasets from previous NHANES cycles" to those guidelines, which were not
   fetched (the web scope was the data files and their codebooks). A common factor on every weight
   moves neither β nor a linearized SE, so this choice changes no number in `results.json`. Records
   outside a subsample get weight 0 and so fall outside every domain. Page: DEMO_J ("Sample
   Weights"), GLU_J (weight 0 outside the fasting subsample).
10. **D-fasting, who counts as fasting.** A glucose or insulin value enters only with
    `WTSAF2YR > 0`. GLU_J states that people in the subsample who did not meet the 8 to < 24 hour
    fasting criterion keep their glucose value but get weight 0, "a reweighting would be required"
    to use them. Between 180 and 245 such values per cycle are in the files. No further
    fasting-hours filter is applied. Page: GLU_J ("Glucose was measured in a fasting subsample").
11. **D-insulin-weight.** For insulin, the WTSAF2YR in INS_H/I/J (in 2011–2012, GLU_G's, the file
    that holds insulin that cycle). It is identical to the glucose file's WTSAF2YR in every cycle
    (maximum absolute difference 0, checked in R). Page: INS_H.
12. **D-crosswalk, lab method changes.** None applied: the released values are used. The insulin
    method changed from 2011–2012 to 2013–2014 (Roche to Tosoh; INS_H gives Deming equations), and
    the glucose instrument changed in 2015–2016 (GLU_I gives forward and backward equations, and its
    C501 results are already adjusted forward in the release). The codebooks offer the equations
    "according to the analytic aims"; the paper names none. Page: INS_H, GLU_I, GLU_J.
13. **D-reliable, "complete and reliable".** Status code 1 on the day (`DRxDRSTZ == 1`). A day
    with no reliable recall on day 1 is "no dietary recall"; a reliable day 1 with no reliable
    day 2 is "only one recall". Page: DR1TOT_G ("Dietary recall status code").
14. **D-water, "only water reported on a recall day".** A reliable day with zero total energy and
    at least one item reported (`DRxTKCAL == 0` and `DRxTNUMF >= 1`). DR1TOT_G says a water-only
    day is coded reliable, counts its water items in the number of foods, and "will have zero
    energy intake for the day". A separate case also has zero energy: a reliable day with no
    water, food or beverage at all (`DRxTNUMF == 0`). It is a different thing from "only water",
    so it is not removed. Page: DR1TOT_G ("Participants who reported consuming only water").
15. **D-pregnancy, "cannot ascertain".** Only `RIDEXPRG == 1` is excluded; 3 (cannot ascertain)
    stays. "Pregnant" names the positive code. Page: DEMO_G (RIDEXPRG).
16. **D-missing, missing covariates.** Complete cases, per fit, on that fit's own variables, as
    SAS's survey procedures drop observations with a missing model variable. So the ladder's n
    differs by model. Table 4 drops (rows with the outcome and a positive weight, then
    used): glucose 7,535 → 6,832; insulin 7,335 → 6,655; HbA1c 16,347 → 14,795; BMI 16,747 →
    15,279. PIR is most of it (632, 609, 1,387 and 1,439 rows missing it); the counts by covariate
    are in `results.json`.
17. **D-variance, the variance estimator.** Taylor linearization with PSUs taken as sampled with
    replacement within strata (survey's default for this design, and the usual one for NHANES).
    svyglm applies no small-sample multiplier. SAS's SURVEYREG, as I recall its documentation,
    multiplies the variance by (n − 1)/(n − p) by default. I could not check that here (web scope
    CDC only). Applied to these fits it would raise each SE by a factor of at most 1.0011 and
    changes no printed comparison (checked for all 19 fits).
18. **D-df, the t reference.** p is two-sided from t with df = 121 PSUs − 59 strata = 62 of the
    full design, the degrees of freedom SAS's survey procedures use by default. svyglm's own
    residual df (design df of the domain + 1 − number of coefficients) and its p are reported
    beside it.
19. **D-domain, the domain.** The analysis rows are a `subset()` of the design over all 39,156
    records, so each SE is the domain SE. `fit.R` derives that SE again by hand (A⁻¹BA⁻¹ with
    zero contributions outside the domain and the full design's PSU counts) and stops unless the two agree
    to 1e-8. They agree to about 1e-14. For these four Table 4 domains every PSU has domain rows,
    so a design built on the domain's rows alone gives the same SE here; the two would part only
    for a domain that leaves a PSU empty.
20. **Reference levels.** Male, non-Hispanic White, college graduate and PIR > 1.85. They do not
    affect the rating's β.

## The sample steps, ours beside the paper's

| Step | Paper removes | Ours removes | Paper remaining | Ours remaining |
|---|---|---|---|---|
| Start (four cycles combined) | | | 39,156 | 39,156 |
| Age < 20 | 16,539 | 16,539 | 22,617 | 22,617 |
| Pregnant | 247 | 247 | 22,370 | 22,370 |
| No self-rating | 10 | 10 | 22,360 | 22,360 |
| No dietary recall | 2,982 | 2,982 | 19,378 | 19,378 |
| Only one recall | 2,445 | 2,445 | 16,933 | 16,933 |
| Only water on a recall day | 20 | 18 | 16,913 | 16,915 |

## Where this differs from the paper

The comparison, at the paper's printed precision, is in `results.json` (`beta_matches_paper`,
`se_matches_paper`). Of the 19 fits, 16 match in β, and every printed SE matches except M4's for
glucose and insulin. The three that miss are the three M4 fits.

**1. The water step: 18 removed, against 20.** Two people in the sample have a reliable day with
nothing reported at all: SEQN 79221 (2013–2014, day 2) and SEQN 93792 (2017–2018, day 1). Removing
them too, the zero-energy reading of the step, gives the paper's 20 and 16,913. Only SEQN 93792
enters a Table 4 fit (BMI; SEQN 79221 lacks PIR). Under that reading every Table 4 n equals the
paper's, and BMI's β moves from −1.6316 to −1.6305 (SE 0.0860 to 0.0859); the other three fits
do not change. So the paper probably removed every zero-energy day. This reference keeps its own
reading (D-water) and lists the difference rather than adopting the reading that matches. Both
are in `results.json`: the fits, and `premise_checks`. The rows are flagged
`nothing_reported_day` in the analytic CSVs.

**2. M4: the stated model does not reproduce the paper.** M0–M3 match for every outcome. M4 as §1
defines it (M2 + glucose-lowering medication) does not:

| Outcome | Ours, M2 + medication: β (SE) | Paper M4: β (SE) |
|---|---|---|
| Glucose | −2.01 (0.34) | −1.1 (0.36), p = 0.004 |
| Insulin | −1.86 (0.22) | −0.4 (0.19), p = 0.03 |
| HbA1c | −0.057 (0.007) | −0.03 (0.01) |

The medication flag is probably not the cause: its share by rating reproduces Table 1's range
(D-med). In these data, adding it to M2 cuts insulin's slope by 17% (−2.23 → −1.86), while the
paper's M2 → M4 cuts it by 82% (−2.2 → −0.4). The paper's M4 numbers are smaller than its
BMI-adjusted Table 4 numbers, which suggests BMI is in its M4. After the miss, five readings of M4 were tried, in this order, and all
are reported in `results.json` (`premise_checks`):

| M4 read as | Glucose β (SE) | Insulin β (SE) | HbA1c β (SE) | All six printed values match |
|---|---|---|---|---|
| M2 + medication + BMI | −1.12 (0.34) | −0.50 (0.18) | −0.031 (0.007) | no |
| M2 + BMI | −2.02 (0.40) | −0.65 (0.18) | −0.052 (0.010) | no |
| M3 + medication | −1.90 (0.35) | −1.72 (0.23) | −0.051 (0.007) | no |
| M3 + medication + BMI | −1.04 (0.36) | −0.41 (0.19) | −0.026 (0.007) | no (glucose β rounds to −1.0) |
| Table 4 model + medication | −1.06 (0.36) | −0.41 (0.19) | −0.027 (0.007) | **yes**; p = 0.0044 and 0.033, printed 0.004 and 0.03 |

The Table 4 model plus medication (age, sex, race, education, PIR, smoking, activity, BMI and
medication; no energy) reproduces all six printed M4 values and both printed p values. It was
found after the fact, among five tries, so it is evidence about what the paper did, not a
result. **Open premise:** §1's "M4: M2 plus the medication" should be checked against the
supplement (NIHMS1722501-supplement-1.pdf). `results.json`'s M4 rows keep §1's definition, as the
task states it.

**3. BMI n is one over** (15,279 against 15,278). That is SEQN 93792, item 1.

## Not done here

- The adjusted means per rating category (Table 4's last column) and the race-stratified appendix
  fits were not in this package's task.
- No unstated decision's alternatives were refit here, apart from the two premise checks above.
  SU-A measures each option's shift through the engine (`SULLIVAN_PLAN.md` §3).
