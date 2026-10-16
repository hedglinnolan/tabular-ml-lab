# Candidates for the pilot's "Estimate" slot (replacing Yang 2014)

**Chosen (Nolan, 2026-10-15): Sullivan 2021.** The replication pilot is Bang 2009 (Predict), Menke 2015 (Describe) and Sullivan 2021 (Estimate). Kant 2015 stays the candidate for a second Estimate paper later.

Checked 2026-10-15. Each fact is tagged **[V]** (verified at the URL or file named) or **[U]** (unverified: inferred, not stated by the source, or not checked).

Working files: `scratchpad/replication/est/` holds the E-utilities helpers (`eu.py`, `abs.py`, `pmc.py`, `tables.py`), the PMC XML and text (`PMC*.xml`, `PMC*.txt`), the search results (`res*.txt`), the HEAD results (`heads_est.tsv`) and the codebook pages (`cb/*.htm`). No XPT file was downloaded; every XPT check was a HEAD request.

Source URL patterns used below:
- PubMed record: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=<PMID>&retmode=xml`
- PMC full text: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=<PMCID>`
- NHANES file: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>.xpt` (and `.htm` for its codebook)

## Ranking

| Rank | Paper | Exposure → outcome | Known answers | Verdict |
|---|---|---|---|---|
| 1 | **Sullivan 2021**, *Am J Prev Med* | Self-rated diet quality (DBQ700) → fasting glucose, insulin, HbA1c | β and SE, n per outcome, and a five-model nested ladder in the appendix | **Near-exact** |
| 2 | **Kant 2015**, *Int J Obes* | Meals away from home and fast-food meals per week (DBD895, DBD900) → HbA1c, fasting glucose | Adjusted (geometric) means and adjusted percentages with 95% CIs, and trend p-values | **Approximate (close)** |
| 3 | **Tucker 2018**, *Nutrients* | Fiber in g per 1000 kcal (one recall) → HOMA-IR | Adjusted HOMA means ± SE and F-tests under five covariate sets | **Approximate** |

**Why this order.**
- **Sullivan 2021** meets every criterion with the least preparation:
  - the exposure is one questionnaire item;
  - the outcomes are NHANES variables used as they come, with no HOMA to derive;
  - the weights and exclusion counts are stated;
  - the known answers are coefficients, and the appendix has a model ladder that shows the BMI, energy and medication choices moving the estimate.
- **Kant 2015** is the cleanest methods section of the three:
  - weights named by variable;
  - exclusion counts that add up;
  - a stated reason for leaving BMI out of the model.

  Its answers are predictive margins rather than coefficients, and its glycemic results are null.
- **Tucker 2018** is closest to the domain (a nutrient, energy-adjusted, against HOMA-IR) and has the choices most worth disputing. But it fails part of criteria 4 and 5:
  - the weights are not named;
  - the exclusions have no counts;
  - the answers are means ± SE, not coefficients;
  - it has internal inconsistencies.

---

## 1. Sullivan 2021: self-rated diet quality and cardiometabolic health, NHANES 2011–2018

### Citation [V]

Sullivan VK, Johnston EA, Firestone MJ, Yi SS, Beasley JM. "Self-Rated Diet Quality and Cardiometabolic Health Among U.S. Adults, 2011-2018." *Am J Prev Med* 2021;61(4):563-575.

- PMID 34246527
- DOI 10.1016/j.amepre.2021.04.033
- PMCID PMC8523030 (an NIH author manuscript, NIHMS1722501)
- Funding: NIH (NIDDK, NHLBI, NIMHD).

[V: PubMed efetch id=34246527; PMC efetch PMC8523030]

### Design [V]

- Cross-sectional, four cycles pooled (2011–12, 2013–14, 2015–16, 2017–18).
- The exposure is the single item "In general, how healthy is your overall diet?", modeled as a continuous 1–5 score (poor = 1 to excellent = 5).
- Each risk factor is fit by survey-weighted linear regression.
- Analysis in SAS 9.4 survey procedures.

[V: PMC8523030 body, Methods]

### Sample and exclusions [V]

| Step | n |
|---|---|
| Start (four cycles combined) | 39,156 |
| Age < 20 | −16,539 |
| Pregnant | −247 |
| No self-rating | −10 |
| No dietary recall | −2,982 |
| Only one recall | −2,445 |
| Only water reported on a recall day | −20 |
| **Analytic sample** | **16,913** |

The arithmetic checks. Only recalls "determined to be complete and reliable per NHANES documentation" were kept.

Analysis n per outcome (Table 4, unweighted):

| Outcome | n |
|---|---|
| Glucose | 6,832 |
| Insulin | 6,655 |
| HbA1c | 14,795 (printed "14.795") |
| BMI | 15,278 |

[V: PMC8523030 body and Table 4]

### Weights [V]

- "Day 2 dietary weights were used in accordance with the least common denominator approach … Fasting subsample weights were substituted in analyses of fasting laboratory data."
- That is, WTDR2D for HbA1c and BMI, and WTSAF2YR for glucose and insulin.
- The variable names are confirmed in the codebooks (DR2TOT_G: "WTDR2D – Dietary two-day sample weight"; GLU_G: "WTSAF2YR – Fasting Subsample 2 Year MEC Weight").
- How the weights were divided across the four cycles is not stated. NCHS's rule (divide by 4) is presumed. [U]

[V: PMC8523030 Methods; https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DR2TOT_G.htm; …/2011/DataFiles/GLU_G.htm]

### Covariates [V unless marked]

- **Main Table 4:** age, sex, race/ethnicity (5 groups, with non-Hispanic Asian), education (4 groups), poverty–income ratio, current smoking, recreational physical activity, and BMI.
  - PIR appears in Table 1 as <130%, 130–185% and >185%. How it was modeled is not stated. [U]
  - The form of age is not stated. [U]
  - Recreational physical activity = weekly moderate recreational minutes + 2 × vigorous recreational minutes.
  - BMI is dropped from the BMI and waist models.
- **Energy:** "Total calorie intake did not meaningfully alter estimates and was omitted from the final models (Appendix Tables 1 and 2)."
- **Threshold:** significance at p < 0.01.

People with diabetes and users of glucose-lowering drugs are **not** excluded. Table 1 shows 5.9–11.6% using glucose-lowering medications across the five rating categories. [V: Table 1]

### Exposure coding [V]

DBQ700 in NHANES is coded:

| Code | Meaning | n in DBQ_G (2011–12) |
|---|---|---|
| 1 | Excellent | 597 |
| 2 | Very good | 1,316 |
| 3 | Good | 2,641 |
| 4 | Fair | 1,293 |
| 5 | Poor | 321 |
| 7 | Refused | 1 |
| 9 | Don't know | 5 |
| . | Missing | 3,582 |

The paper reverses this (poor = 1), so a replication on the raw codes gets the sign flipped. Codes 7 and 9 must be dropped.

[V: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DBQ_G.htm, code table]

### Known answers

**Table 4 (main model, BMI-adjusted)** [V: PMC8523030 Table 4]. Adjusted means run Poor → Excellent.

| Outcome | n | β per rating step | SE | p | Adjusted means (Poor → Excellent) |
|---|---|---|---|---|---|
| Glucose (mg/dL) | 6,832 | −2.0 | 0.4 | <0.001 | 111.8, 112.9, 110.5, 108.5, 105.2 |
| Insulin (μU/mL) | 6,655 | −0.6 | 0.2 | 0.003 | 15.2, 14.8, 14.2, 13.5, 13.2 |
| HbA1c (%) | 14,795 | −0.05 | 0.01 | <0.001 | 5.86, 5.91, 5.86, 5.76, 5.77 |
| BMI (kg/m², no BMI adjustment) | 15,278 | −1.6 | 0.1 | <0.001 | — |

**Appendix Table 2 (nested models, none with BMI)** [V: supplement NIHMS1722501-supplement-1.pdf, read in a Chrome tab with pdf.js; https://pmc.ncbi.nlm.nih.gov/articles/instance/8523030/bin/NIHMS1722501-supplement-1.pdf]

Model definitions:
- M0: no covariates.
- M1: age and sex.
- M2: age, sex, race, education and PIR.
- M3: M2 plus smoking, recreational physical activity and **total energy**.
- M4: **M2** plus the medication for the risk factor (glucose-lowering drugs for glucose, insulin and HbA1c).

β per rating step (SE):

| Outcome | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| Glucose | −2.7 (0.37) | −4.1 (0.36) | −3.7 (0.44) | −3.6 (0.45) | −1.1 (0.36), p = 0.004 |
| Insulin | −2.3 (0.22) | −2.5 (0.21) | −2.2 (0.23) | −2.1 (0.23) | −0.4 (0.19), p = 0.03 |
| HbA1c | −0.08 (0.01) | −0.13 | −0.10 | −0.09 | −0.03 (0.01) |

**Appendix, race-stratified, BMI-adjusted** [V: same supplement]:

| Outcome | Group | n | β (SE) |
|---|---|---|---|
| Glucose | White | 2,831 | −2.1 (0.5) |
| Glucose | Hispanic | 1,557 | −2.8 (0.9) |
| Glucose | Black | 1,430 | −1.1 (0.9) |
| Glucose | Asian | 759 | −1.2 (1.4) |
| Insulin | White | 2,770 | −0.7 (0.3) |

### Analytic choices worth discussing [V for what the paper did; the reading is mine]

- **BMI in the adjustment set.** The paper's own numbers show it.
  - The no-BMI model with energy (M3) gives −3.6 for glucose; the BMI-adjusted Table 4 gives −2.0.
  - Is BMI a confounder of rating and glucose, or the path from diet to glucose?
  - The two models differ in more than BMI (M3 has energy, Table 4 does not), so the comparison is not clean either.
- **Diagnosed diabetes kept in, medication handled only in M4.** Adjusting for glucose-lowering drugs cuts the glucose association by about 70% (−3.7 → −1.1).
  - Treated people with diabetes probably rate their diet differently and have high glucose. That is reverse causation and treatment confounding in one variable.
  - A defensible alternative is to exclude diagnosed diabetes, as Kant and Tucker do.
- **Energy adjustment.** It was dismissed as "did not meaningfully alter estimates", but energy was only tested together with smoking and physical activity (M2 → M3), never alone.
- **Sample restricted to two-recall completers.** This is needed for the HEI aim but not for the cardiometabolic one, and it makes a weighting conflict:
  - WTDR2D adjusts for day-2 nonresponse;
  - WTSAF2YR, used for the fasting outcomes, does not;
  - NHANES gives no weight for "fasting ∩ two recalls".
- **Ordinal as linear.** A 1–5 rating is treated as equally spaced. The adjusted means are not monotone (Fair is above Poor for glucose and HbA1c).
- **Insulin untransformed** despite its skew. So is glucose, in a sample that includes people with diabetes.
- **p < 0.01** in place of 0.05, and 99% CIs in the text.

### Data files

Fourteen of the 74 files were HEAD-checked; all returned 200. [V: HEAD, `est/heads_est.tsv`]

| Cycle | Files | HEAD |
|---|---|---|
| 2011–12 | DEMO_G, DBQ_G, DR1TOT_G, DR2TOT_G, GLU_G (glucose **and insulin**: INS_G returns 404), GHB_G, BMX_G, SMQ_G, PAQ_G | all 200 |
| 2013–14 | DEMO_H, **DBQ_H**, DR1TOT_H, DR2TOT_H, **GLU_H**, **INS_H**, **GHB_H**, BMX_H, SMQ_H, PAQ_H | bold 200 |
| 2015–16 | DEMO_I, **DBQ_I**, DR1TOT_I, DR2TOT_I, **GLU_I**, **INS_I**, GHB_I, BMX_I, SMQ_I, PAQ_I | bold 200 |
| 2017–18 | **DEMO_J**, **DBQ_J**, DR1TOT_J, **DR2TOT_J**, **GLU_J**, **INS_J**, **GHB_J**, BMX_J, SMQ_J, **PAQ_J** | bold 200 |
| Each cycle | BPQ, DIQ, RXQ_RX (medication use; descriptive, and M4 only) | not checked [U] |

Files not in bold were not HEAD-checked. They follow the same naming pattern. [U]

Variables:
- In the codebooks [V]:
  - DBQ700 (DBQ_G, DBQ_J);
  - LBXGLU, LBXIN, WTSAF2YR (GLU_G); LBXIN, WTSAF2YR (INS_H);
  - DR2DRSTZ, WTDR2D (DR2TOT_G);
  - PAQ650, PAQ655, PAD660, PAQ665, PAQ670, PAD675 (PAQ_G, PAQ_J);
  - RIDRETH3, DMDEDUC2, INDFMPIR, RIDEXPRG, SDMVPSU, SDMVSTRA (DEMO_G);
  - SMQ020, SMQ040 (SMQ_G).
- LBXGH in GHB is not checked. [U]
- 2011–12 overlaps Menke 2015's GLU_G and GHB_G. [U: depends on Menke's file list]

### Engine gaps (read-only check)

| Need | Status |
|---|---|
| **Pooling four cycles** | **Partial.** The stacking module `turbotab/core/methods/cycles.py` is "engine only until C1 asks it" (contracts.py comment). A pre-stacked table with a cycle column gets weights divided by the number of cycles (`methods/survey.py` `analysis_weights`). [V] |
| Weight recognition (WTSAF2YR "fasting subsample", WTDR2D "dietary days 1 and 2") | Recognized (`recognizers.py` ~l.1799–1803). [V] |
| Survey-weighted linear regression with domains and t(df = PSUs − strata) | Exists (`models/survey.py`). [V] |
| **Recreational PA minutes** (days × minutes, vigorous doubled) and **current smoking** (SMQ020 + SMQ040) | **Gap:** multi-column derivations (the known gap). |
| Reversing DBQ700 | No recode found; compare −β. Dropping codes 7/9 is an exclusion rule (`SetExclusions`). [V that the decision exists; U how it is phrased] |
| **Adjusted means per rating category** | **Gap.** Not found for survey fits; `methods/trials.py` has emmeans-style arm means for trials only. The β (the headline answer) is unaffected. [V: grep] |
| Exposure as a straight line | Supported (`exposure_form.py` "linear"). [V] |
| Medication-adjusted M4 | Needs the medication flags, a multi-column derivation for glucose-lowering drugs (DIQ050/DIQ070). [U] |

### Verdict: **near-exact**

The Table 4 coefficients are fully specified except:
- the forms of age and PIR;
- how the 20 "only water" respondents were identified.

Rounding (one decimal place) makes the β easy to match. The paper was not built for the Estimate slot, but its appendix ladder makes it an unusually good teaching case.

---

## 2. Kant 2015: meals away from home and biomarkers, NHANES 2005–2010

### Citation [V]

Kant AK, Whitley MI, Graubard BI. "Away from home meals: associations with biomarkers of chronic disease and dietary intake in American adults, NHANES 2005-2010." *Int J Obes (Lond)* 2015;39(5):820-7.

- PMID 25319744
- DOI 10.1038/ijo.2014.183
- PMCID PMC4400187 (NIH author manuscript NIHMS629165)
- Funding: NIH intramural. COI: none.
- Graubard is an NCI survey statistician. [U, from memory; not on the record]

[V: PubMed efetch id=25319744; PMC efetch PMC4400187]

### Design [V]

- **Metabolic outcomes:** NHANES 2007–08 and 2009–10, cross-sectional.
- **Exposures:**
  - weekly meals "prepared away from home" (DBD895), coded 0, 1–2, 3–5 and ≥6;
  - weekly fast-food or pizza meals (DBD900), coded 0, 1, 2–3 and ≥4;
  - each also as a continuous trend variable.
- **Outcomes:** BMI, lipids, HbA1c, fasting glucose. Each is modeled continuously (log-transformed, back-transformed to geometric means) and as a binary at a risk cut point (HbA1c ≥ 5.7%, glucose ≥ 100 mg/dL).
- **Software:** SUDAAN. Predicted margins give the adjusted means and proportions; Satterthwaite-adjusted F tests give the p-values.

[V: PMC4400187 Methods]

### Sample [V]

| Step | n |
|---|---|
| Eligible: non-pregnant, non-lactating, age ≥ 20, answered the AFH question, at least one metabolic biomarker | 10,953 |
| Users of cholesterol- or glucose-lowering medication excluded | −2,639 |
| **Analytic sample** (4,070 men and 4,244 women) | **8,314** |

- The arithmetic checks.
- The adjusted N is 8,297 after excluding missing education (11), smoking (4) and physical activity (2). This also checks.
- For glucose, people fasting < 9 h are excluded.
- Analysis n: glucose 3,668; HbA1c 8,234.

[V: Methods; Table 1 footnote; Tables 2–3]

### Weights [V]

"1/2 MEC weight (WTMEC2YR or WTSAF2YR) was used for each cycle"; the fasting weight for glucose, LDL and triglycerides. [V: Methods]

### Covariates, chosen a priori [V]

| Covariate | Coding |
|---|---|
| Age | 20–39, 40–59, ≥60 |
| Sex | — |
| Race/ethnicity | 4 groups |
| PIR | ≤1.3, >1.3–3.5, >3.5, unknown |
| Education | 4 groups |
| Serum cotinine | continuous |
| Alcohol status | never, former, current, unknown |
| Physical activity | none, then tertiles of MET-min/week |
| Fasting hours | continuous |

- **BMI deliberately excluded** "because of concerns about BMI potentially being on the causal pathway". With BMI added, results were "essentially unchanged (data not shown)".
- Where an exposure × age or × sex interaction had p < 0.05, results are stratified.

[V: Methods]

### Known answers [V: PMC4400187 Tables 2–3; supplement NIHMS629165-supplement-1.docx, read in a Chrome tab from the article page https://pmc.ncbi.nlm.nih.gov/articles/PMC4400187/]

Adjusted geometric means (95% CI) across the exposure categories:

| Table | Exposure | Outcome | Group (n) | Values | P (trend) |
|---|---|---|---|---|---|
| 2 | Away from home (0 / 1–2 / 3–5 / ≥6) | Fasting glucose, mg/dL | all (3,668) | 99 (98, 101); 99 (98, 100); 98 (97, 99); 99 (98, 100) | 0.8 |
| 2 | Away from home | HbA1c, % | all (8,234) | 5.41 (5.38, 5.44); 5.41 (5.38, 5.44); 5.39 (5.37, 5.41); 5.40 (5.37, 5.44) | 0.4 |
| 2 | Away from home | HbA1c, % | <50 y | 5.30 … 5.29 | 0.1 |
| 2 | Away from home | HbA1c, % | ≥50 y | 5.59 (5.55, 5.63) … 5.68 (5.61, 5.76) | 0.1 |
| 3 | Fast food (0 / 1 / 2–3 / ≥4) | Fasting glucose, mg/dL | all | 99 (98, 100) in every category | 0.5 |
| 3 | Fast food | Fasting glucose, mg/dL | men (1,750) | 102; 102; 101; 101 | 0.1 |
| 3 | Fast food | Fasting glucose, mg/dL | women (1,918) | 97 (95, 98); 95 (94, 97); 97 (96, 99); 98 (96, 101) | 0.2 |
| 3 | Fast food | HbA1c, % | all | 5.42 (5.39, 5.44); 5.39 (5.36, 5.42); 5.39 (5.36, 5.42); 5.40 (5.37, 5.44) | 0.2 |

Adjusted percentage above the risk cut point (95% CI), from logistic models:

| Table | Exposure | Outcome | Group | Values | P (trend) |
|---|---|---|---|---|---|
| Supp. 3 | Away from home | HbA1c ≥ 5.7% | all | 23 (21, 26); 25 (23, 28); 24 (22, 25); 23 (19, 24) | 0.1 |
| Supp. 3 | Away from home | Glucose ≥ 100 | all | 43 (38, 48); 44 (40, 48); 41 (37, 44); 45 (40, 49) | 0.7 |
| Supp. 4 | Fast food | HbA1c ≥ 5.7% | all | 25 (23, 27); 22 (20, 25); 23 (20, 25); 23 (20, 27) | 0.1 |
| Supp. 4 | Fast food | HbA1c ≥ 5.7% | <50 y | 16; 14; 17; 13 | 0.03 |
| Supp. 4 | Fast food | HbA1c ≥ 5.7% | ≥50 y | 40; 39; 35; 51 | 0.2 |
| Supp. 4 | Fast food | Glucose ≥ 100 | all | 42; 43; 42; 46 | 0.5 |
| Supp. 4 | Fast food | Glucose ≥ 100 | women | 32 (27, 37); 30 (25, 36); 36 (28, 44); 45 (35, 55) | 0.04 |

Note on Supp. Table 3: the ≥6 category's HbA1c entry "23 (19, 24)" sits at the edge of its own interval, so it is probably a typo. Score that cell loosely. [V for the printed value; U that it is an error]

### Analytic choices worth discussing [V for what the paper did]

- **BMI left out as a mediator.** This is the mirror image of Sullivan, and the two pair well for a pedagogy beat. The sensitivity analysis with BMI is "data not shown", so it cannot be scored.
- **Medication users excluded** (2,639 people). This conditions on treatment, which follows from the outcome. Defensible, but it removes the people with the highest glucose.
- **Fasting hours and cotinine as covariates.**
- **Missing-indicator categories** ("unknown" PIR and alcohol).
- **Continuous confounders cut coarsely** (age in 3 groups).
- **Stratify-on-interaction rule** (p < 0.05), which makes the reported table data-dependent.
- **Exposure validity.** The away-from-home question changed wording between 2005–06 and 2007–10, which is why 2005–06 is only used for the nutrient biomarkers. No validation of the question exists, as the paper says.
- **Inconsistent top category.** It is written ">6" in Methods and "≥6" in the tables. Either way 6 belongs to the top category, but the inconsistency is a cut-point detail to record.

### Data files

All needed for 2007–08 and 2009–10. [V: HEAD, `est/heads_est.tsv`]

| File | HEAD |
|---|---|
| DEMO_E, DEMO_F | 200 |
| DBQ_E, DBQ_F | 200 |
| GHB_E, GHB_F | 200 |
| GLU_E, GLU_F | 200 |
| FASTQX_E, FASTQX_F | 200 |
| ALQ_E | 200 |
| PAQ_E | 200 |
| BPQ_E | 200 |
| DIQ_E | 200 |
| **COTNAL_E, COTNAL_F** (cotinine) | 200; COT_E and COT_F return 404 |
| BMX_E/F, ALQ_F, PAQ_F, BPQ_F, DIQ_F, RHQ_E/F (breastfeeding), RXQ_RX | not checked [U] |

Variables in the codebooks [V]:
- DBD895 and DBD900 in DBQ_E, DBQ_F, DBQ_G and DBQ_J;
- LBXCOT in COTNAL_E and COTNAL_F;
- LBXGLU, LBXIN, WTSAF2YR and PHAFSTHR in GLU_E;
- PHAFSTHR in FASTQX_E;
- the GPAQ items (PAQ610 through PAD675) in PAQ_E;
- ALQ101, ALQ110 and ALQ120Q in ALQ_E.

DBD895 codes (DBQ_E) [V: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DBQ_E.htm]:
- 0 to 21 meals;
- 5555 = "More than 21 meals per week" (n = 1);
- 7777 = Refused, 9999 = Don't know.

The 5555, 7777 and 9999 codes must be handled before any continuous trend variable is built.

The paper does not name the medication variables it used. They are presumably BPQ090D/BPQ100D and DIQ050/DIQ070. [U]

### Engine gaps

| Need | Status |
|---|---|
| **Adjusted means and percentages (predictive margins) per exposure category** under the survey design | **Gap.** Graubard–Korn margins exist only for the substitution curve (`models/survey.py` `design_curve`). Without them, the replication can only compare the regression coefficients behind the margins, which the paper does not print, or the trend p-values. [V: grep] |
| Survey-weighted linear and logistic regression; log outcome (`SetOutcomeScale`); exposure categories at declared cuts (`categories`); halved two-year weights (pooled cycles) | Supported. [V] |
| **Age cut into three groups as a confounder** | **Blocked** under inference by the coarse-confounder rule (`exposure_form.py` `_cuts_are_declared_or_recorded`, Brenner & Blettner). It can be kept "as a recorded limitation". A good place for the app to argue with the paper. [V] |
| PA MET-min tertiles (GPAQ days × minutes × MET), alcohol status (three items), medication exclusion flags | **Gap:** multi-column derivations. |
| Missing-indicator ("unknown") categories | Not checked against `methods/missing.py`. [U] |
| Stratifying when an interaction test passes | A modification decision exists (`SetModification`); not checked whether it can reproduce the automatic rule. [U] |

### Verdict: **approximate (close)**

The data, eligibility, weights and covariates are all specified. What stops an exact score is the margins gap (the known answers are margins) and the derived covariates. Its glycemic results are null, which is still a fair test.

---

## 3. Tucker 2018: fiber intake and insulin resistance, NHANES 1999–2006

### Citation [V]

Tucker LA. "Fiber Intake and Insulin Resistance in 6374 Adults: The Role of Abdominal Obesity." *Nutrients* 2018;10(2):237.

- PMID 29461482
- DOI 10.3390/nu10020237
- PMCID PMC5852813 (MDPI open access)
- COI: "The author declares no conflict of interest."
- Single author.

[V: PubMed efetch id=29461482; PMC efetch PMC5852813]

### Design [V]

- Cross-sectional, four cycles 1999–2006, chosen because those are "the only years NHANES measured body fat percentage … using DXA".
- **Exposure:** total fiber per 1000 kcal from **one** 24-h recall, in **sex-specific quartiles**, the middle two collapsed into three groups: Low (Q1), Moderate (Q2–Q3), High (Q4).
  - Women: Low ≤ 5.0; Moderate 5.1–9.8; High > 9.8 g/1000 kcal.
  - Men: Low ≤ 4.2; Moderate 4.3–8.5; High > 8.5 g/1000 kcal.
- Also continuous, and dichotomized at the Dietary Guidelines' 14 g/1000 kcal.
- **Outcome:** HOMA = insulin (μU/mL) × glucose (mg/dL) / 405, log-transformed for modeling.
- **Fit:** SAS PROC SURVEYREG, with "least-squares means" for the adjusted means.
- **Design df:** 117 clusters − 58 strata = 59.

[V: PMC5852813 Methods]

### Sample [V]

- **Fasting specimens usable for HOMA (age ≥ 12):** 2,986 (1999–2000), 3,379 (2001–02), 3,120 (2003–04), 3,080 (2005–06).
- **Restricted to:**
  - age 20–84;
  - no diabetes (FPG ≥ 126, physician diagnosis, insulin or oral medication);
  - complete data on fiber, energy, glucose, insulin, age, sex, race, smoking, BMI, body fat, waist and PA.
- **Final:** 6,374 (3,321 women, 3,053 men).
- **Missing: counts for each exclusion step** (criterion 4 partly unmet). [V]

### Weights

"individual sample weights for the subsample of fasting participants". The variable and the eight-year combination are not named. [V that it is unnamed]

Presumably the combination is:
- 1999–2002: WTSAF4YR × ½;
- 2003–2006: WTSAF2YR × ¼.

[U]

### Covariates, entered in sequence [V]

1. Demographics (age, sex, race).
2. \+ misreported energy intake (MEI), smoking pack-years and PA.
3. \+ DXA body fat %, **or** BMI, **or** waist circumference, each in its own model.

Their construction:
- **MEI** = self-reported kcal − (Mifflin RMR × a PAL assigned from the 4-level PA item: 1.45 / 1.55 / 1.65 / 1.75).
- **Pack-years** = cigarettes per day × years smoked / 20.
- **PA** = the four-sentence daily-activity item (PAQ180 in 1999).
- **Body fat %** comes from NHANES's multiply imputed DXA files, entered as sex-specific quintiles plus a sixth "missing" category (n = 633).

[V: Methods; PAQ180 in https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/PAQ.htm]

### Known answers [V: PMC5852813 Table 2 and Results]

Adjusted HOMA mean ± SE (Low / Moderate / High fiber), F with 2 and 59 df:

| Covariates | Low | Moderate | High | F | p |
|---|---|---|---|---|---|
| Demographics | 2.62 ± 0.10 | 2.37 ± 0.07 | 2.21 ± 0.07 | 5.4 | 0.0072 |
| + MEI, pack-years, PA | 2.59 ± 0.10 | 2.32 ± 0.07 | 2.06 ± 0.07 | 8.0 | 0.0009 |
| + body fat % | 2.55 ± 0.10 | 2.28 ± 0.07 | 2.14 ± 0.07 | 7.0 | 0.0019 |
| + BMI (in place of body fat) | 2.48 ± 0.09 | 2.27 ± 0.06 | 2.19 ± 0.06 | 4.9 | 0.0108 |
| + waist (in place of BMI) | 2.55 ± 0.09 | 2.35 ± 0.06 | 2.31 ± 0.06 | 2.3 | 0.1050 |

Fiber as a continuous variable:

| Model | F | p |
|---|---|---|
| Unadjusted | 7.7 | 0.0075 |
| Demographics | 10.2 | 0.0022 |
| + smoking, PA, MEI | 19.0 | <0.0001 |
| Fiber per estimated 1000 kcal | 79.8 | <0.0001 |
| Estimated energy + waist | 2.98 | 0.0895 |

The text adds that "for each 10 g decrease in fiber (per 1000 kcal), HOMA was 0.4 points higher".

The 14 g/1000 kcal split:
- 501 people meet it and 5,873 do not.
- HOMA is 34.7% lower in those who meet it (F = 16.4, p = 0.0002, with demographics, MEI, smoking and PA).
- With waist added: F = 4.5, p = 0.0390.

Descriptives (Table 1): mean HOMA 2.2 (SE 0.05); fiber 7.3 g/1000 kcal.

**Internal inconsistencies to flag before scoring** [V]:
- **Text vs table for the same model.** For demographics + MEI + pack-years + PA, the Results text gives F = 9.9 (MEI only) and then F = 13.9, p < 0.0001. Table 2 and the abstract give F = 8.0, p = 0.0009.
- **Fiber mean.** The text gives 15.7 ± 0.2 g; Table 1 gives 15.8 (SE 0.20).
- **Race counts.** Table 1's footnote gives 1,666 "other races" and 204 "Other Hispanic". That looks swapped for NHANES 1999–2006. [U: inference]
- **Scale of the means.** The means are reported on the HOMA scale although the models are on log HOMA. Whether they are back-transformed geometric means is not stated. [U]

### Analytic choices worth discussing (the richest of the three)

- **Energy adjustment by density** (g/1000 kcal) is the only energy handling.
  - It is a ratio exposure, which the engine's own `energy.py` calls "rescaled relative effect; obscure" (density without E).
  - The engine offers the residual and standard models instead.
- **The "misreported energy intake" covariate is built from weight and height** (Mifflin). Adjusting for it partly adjusts for body size. That may be why the association "strengthened".
- **Waist circumference treated as the "mediator" that "eliminates" the association.** This is the over-adjustment and mediation question in its sharpest form. The paper concludes "much of the association is due to differences in waist circumference".
- **One recall** standing in for usual intake, with sex-specific quartiles collapsed 1–2–1. That puts regression dilution and classification error into the exposure.
- **A data-dependent cut** (quartiles from these data) beside a guideline cut (14 g/1000 kcal) that only 7.9% meet (501 of 6,374).
- **No dietary weight.** The fasting weight is used while the exposure comes from the day-1 recall.

### Data files

The 1999–2006 laboratory files overlap Menke 2015's checks (`replication/heads_continuous.tsv`).

| Cycle | Files | HEAD [V: `est/heads_est.tsv`] |
|---|---|---|
| 1999–00 | DEMO, DRXTOT, LAB10AM (glucose, insulin, WTSAF2YR and WTSAF4YR), BMX, SMQ, PAQ, DXX, DIQ | all 200 |
| 2001–02 | DEMO_B, DRXTOT_B, L10AM_B (WTSAF4YR), BMX_B, SMQ_B, PAQ_B, DXX_B, DIQ_B | DRXTOT_B, L10AM_B, DXX_B 200; DEMO_B, BMX_B, DIQ_B 200 in the earlier check; the rest unchecked |
| 2003–04 | DEMO_C, DR1TOT_C, L10AM_C, BMX_C, SMQ_C, PAQ_C, DXX_C, DIQ_C | DR1TOT_C, L10AM_C, DXX_C, PAQ_C 200 |
| 2005–06 | DEMO_D, DR1TOT_D, **GLU_D (glucose and insulin; INS_D returns 404)**, BMX_D, SMQ_D, PAQ_D, DXX_D, DIQ_D | DEMO_D, DR1TOT_D, GLU_D, DXX_D, PAQ_D, SMQ_D 200 |

Earlier check for 2001–02: [V: `replication/heads_continuous.tsv`].

Variables in the codebooks [V]:
- DRXTFIBE and DRXTKCAL in DRXTOT;
- DR1TFIBE and DR1TKCAL in DR1TOT_D;
- LBXGLU, LBXIN, WTSAF2YR and WTSAF4YR in LAB10AM and L10AM_B;
- LBXGLU, LBXIN and WTSAF2YR in L10AM_C and GLU_D;
- SMD070, SMD030 and SMD055 in SMQ;
- DIQ010, DIQ050 and DIQ070 in DIQ;
- PAQ180 in PAQ.

Not checked:
- The DXA variable names (body fat %, the implicate index). [U]
- DRXTFIBE in DRXTOT_B. [U]

### Engine gaps

| Need | Status |
|---|---|
| **HOMA-IR** (insulin × glucose / 405), **MEI** (kcal − Mifflin × PAL), **pack-years** | **Gap:** each is a multi-column derivation. The Goldberg screen with Mifflin exists (`methods/misreporting.py`), but as a screen, not a covariate. [V] |
| **Sex-specific quartile cut points**, collapsed 1–2–1 | **Gap.** The forms are linear, spline, quintiles, or one list of declared cuts per column (`ExposureFormSpec`). Sex-specific cuts need a derived column. [V] |
| **DXA multiple imputations** (five implicate rows per person) | **Gap.** A join would make it one row per implicate. Combining to one row per person is "deep assembly, planned for v2.x" (`assembly.py`). The body-fat model (one row of five) is out of reach today. [V] |
| Eight-year fasting weight with the 1999–2002 four-year rule | Supported for a pre-stacked table (`methods/survey.py` `analysis_weights`; `methods/cycles.py`). Stacking itself is engine-only. [V] |
| Density energy adjustment, log outcome, survey-weighted linear regression, the 14 g dichotomy as a declared cut | Supported. [V] |
| Least-squares (adjusted) means | **Gap** (as for Kant). |

### Verdict: **approximate**

Four of the five Table 2 models (all but body fat) are reachable after outside preparation of HOMA, MEI and pack-years. Even then, scoring is against means ± SE with an unnamed weight and inconsistent F values, so close agreement is the best outcome.

A sibling paper fixes the coefficient problem but has the same weight gap: **Tucker 2022, *Nutrients* 14(5):918**.
- PMID 35267895, DOI 10.3390/nu14050918, PMCID PMC8912746.
- Ten macronutrients as % of energy, averaged over **two** recalls, against HOMA-IR in 2011–2018 (n = 5,665).
- It reports regression slopes ± SE (fiber −0.023 ± 0.008, F = 9.1, p = 0.0037; total carbohydrate 0.011 ± 0.004; total fat −0.013 ± 0.005; 62 df).
- But it also does not name its weight, gives no exclusion counts, and adjusts every model for waist.

[V: PMC8912746 Table 2 and Methods]

---

## Strong-looking papers rejected

| Paper | What it offers | Why rejected |
|---|---|---|
| Sneed 2023, *PLoS One* (PMID 37339144, PMC10281581): added sugar and prediabetes, 2013–2018 | Survey-weighted logistic regression, ORs with CIs, WTSAF2YR named [V] | Added sugar from the **FPED**, a USDA file, not an NHANES one [V]. **NCI-method usual intake** as the exposure [V]; the engine's `usual_intake.py` estimates the distribution of usual intake (its header), and regression calibration was not looked for [U]. The abstract's CI "OR 1.01, 95% CI .99–1.00" excludes its own estimate [V]. |
| Park 2017, *J Nutr* (PMID 28298537, PMC5368583): Mediterranean diet and HOMA, mediated by BMI and waist | Exactly the mediation debate | **NHANES III** [V: abstract]. |
| Lin 2022, *IJERPH* (PMID 36613000, PMC9819548): sugar-sweetened beverages and prediabetes | ORs | Sugar from SSBs built by **summing food-level records** (the day-1 and day-2 individual-food files) [V]: the known gap. The weights are not named. |
| Tester 2020, *Prev Chronic Dis* (PMID 33092687, PMC7587298): whole grains and impaired fasting glucose in adolescents | aOR 0.60 (0.38–0.93) [V] | Whole grains from the **FPED** [V]; adolescents; two-day average. |
| Vernarelli 2015, *Eur J Nutr* (PMID 24664188, PMC4176562): dietary energy density and HOMA-IR | Careful definition (food-only energy density, beverages excluded by USDA food code) [V] | Needs **food-level summing** (kcal and grams of foods only) [V]: the known gap. Otherwise a good backup if food-file aggregation lands. |
| Wu 2023, *Nutrients* (PMID 37630736): HEI-2015 vs aMed and prediabetes | ORs | HEI and aMed need the FPED and food-level data. Not read in full. [V: abstract only] |
| Al-Ibrahim 2019, *Nutr J* (PMID 30995902) and Curlin 2023, *BMC Nutr* (PMID 37924091) | Diet indices against glycemic markers | HEI, AHEI and plant-based indices need the FPED and food-level data. [V: abstracts] |
| Mazidi 2018, *Lipids Health Dis* (PMID 30223829): whole grains, adiposity and HOMA | — | Whole-grain intake from food-pattern data; analysis of covariance and mediation, not a regression with published coefficients. [V: abstract] |
| UFA/TF 2023, *Lipids Health Dis* (PMID 38053162, PMC10696799): unsaturated fat ratio and insulin resistance | — | **No survey weighting described** (Mann-Whitney and plain logistic regression in R). People with diabetes included. Implausible intakes (mean carbohydrate 438.8 g/day). [V] |
| Sun 2018, *Nutrients* (PMID 30513866, PMC6316834): organic food purchase and diabetes | — | Outcome is prevalent diabetes. HEI-2010 in the adjustment set needs the FPED or MPED. The weight is not named. [V] |
| McAtee 2020, *Nutrients* (PMID 32344821, PMC7281998): home food availability and prediabetes | WTDR2D named [V] | The exposure is availability, not intake. The score is a sum of five items (a derived column). Prediabetes by HbA1c alone [V]. A reasonable fourth choice. |
| Leung 2012, *AJCN* (PMID 22170370, PMC3238460): SNAP participation | — | Outcomes are prevalence ratios from Poisson models. No survey Poisson family was found in `models/survey.py`, which lists linear, logistic, multinomial, Cox and ordinal [V]. The exposure is program participation, not diet. |
| Lee 2020, *Nutrients* (PMID 32549382, PMC7353167): caffeine and HOMA | — | The exposure is **urinary** caffeine metabolites, a biomarker (n = 994 subsample) [V: abstract]. |
| Austin & Krueger 2013, *J Nutr* (PMID 23946345): % energy from sugar and fats against HOMA-IR, 2005–2006 | Isocaloric substitution, the best energy-adjustment teaching case found [V: abstract] | **Not in PMC.** Full text is free (bronze) at the publisher but returned 403 to curl from both Elsevier and OUP. Methods and tables were **not read** [U]. Worth reading in a browser; the engine has `percent_energy.py` substitution. |
| Kant 2015's 2005–06 nutrient-biomarker half | — | The outcomes are serum vitamins, not metabolic markers. |
| Tucker 2022, *Nutrients* (PMC8912746) | Coefficients ± SE | See §3: weights and exclusion counts unstated. |
| Industry-linked beverage and nut papers: Leahy 2017 (PMID 28837084), O'Neil 2015 (PMID 26123047) | — | Exposures need food-level summing [V: abstracts]. Funding not checked [U]. |

## How the search was run [V]

- **PubMed E-utilities searches:** NHANES (in title/abstract, or the survey's full name) combined with the glycemic outcomes (insulin resistance, HOMA, fasting insulin, HbA1c, fasting glucose, prediabetes) and each of the exposures:
  - sugar-sweetened beverages, fiber, coffee/caffeine, diet quality and diet indices, whole grains, minerals, nuts/dairy/eggs, juice/fruit, sugars/carbohydrate;
  - protein/fats/sodium, vitamins, alcohol/meals/breakfast, sweeteners/water, physical activity/sitting/sleep;
  - the author names Graubard, Kant and Tucker;
  - a nutrition-journal filter, and a broad 2001–2022 sweep (441 hits).
- About 30 abstracts were read (`est/abs1.txt`, `est/abs2.txt`).
- Nine PMC full texts were read in XML: PMC8523030, PMC4400187, PMC5852813, PMC8912746, PMC10696799, PMC6316834, PMC7281998, plus excerpts of PMC10281581, PMC9819548, PMC4176562 and PMC7587298.
- Two supplements were read in one Chrome tab, which was closed afterwards. Nothing from them was saved to disk.
