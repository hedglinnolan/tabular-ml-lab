# Dietary intake / nutrition pack — research specification

Same structure as the other packs: by modeling step, and within each step by **diagnostic**,
**coaching**, and **presentation**. Every recommendation carries **SETTLED** · **CONVENTION** ·
**DISPUTED**.

**Scope:** 24-hour recalls (24HR), food-frequency questionnaires (FFQ), food records, body
composition and anthropometry.

**Research caveat.** Academic hosts (PMC, PLOS, medRxiv, CDC, ScienceDirect) were blocked by the
session egress policy, so this is assembled from search-surfaced excerpts of primary sources plus
domain knowledge. A verification-gaps list is at §10.

---

## 01 · Import and structural recognition

### Nutrient column recognition — match on three signals jointly, never names alone

1. **Name patterns:** `energy|kcal|kj|calor`, `prot`, `carb|cho`, `fat|lipid|tfat|sfa|mufa|pufa`,
   `fibe|fiber|fibre`, `sugr|sugar`, `chol`, `sodi|na_`, `pota|k_`, `calc|ca_`, `iron|fe_`, `zinc`,
   `vit[a-e]|folate|folic|niacin|thia|ribo|retinol|carotene|tocopherol`, `alco|etoh`, `caff`, `mois`
2. **NHANES/WWEIA schema:** `SEQN` plus `DR1T*`/`DR2T*` (2003→), `DRXT*` (1999–2002), `DR1I*`/`DR2I*`
   (food-level, long format), `DS1TOT`/`DSQTOT` (supplements), `DR1TVARA` (µg RAE), `DR1TFDFE`
   (µg DFE), `DR1TVD` (µg)
3. **Unit suffixes:** `_g`, `_mg`, `_mcg`, `_ug`, `_iu`, `_kcal`, `_kj`, `_per1000kcal`, `_pct_energy`

### ★ The Atwater reconstruction check — unit inference with a real basis [SETTLED as a diagnostic]

Compute `E_hat = 4·protein_g + 4·carbohydrate_g + 9·fat_g + 7·alcohol_g` and compare to the declared
energy column:

| `E_declared / E_hat` | Inference |
|---|---|
| 0.90–1.10 | Energy in **kcal**, macros in **g**. Pass. (Residual discrepancy is normal — fiber contributes ~2 kcal/g in some systems; organic acids and polyols differ by database.) |
| ~4.18 | Energy column is **kJ** (1 kcal = 4.184 kJ) |
| ~0.239 | The inverse mislabeling |
| <0.5 or >2, no clean factor | Macros are likely **% of energy**, not grams — check whether the four sum to ~100 |
| ratio drifts with total energy | **Mixed units across rows** (multi-source merge) — hard fail |

**Median-magnitude plausibility priors** (adults/day) as a second signal: energy 1,600–2,600 kcal
(7,000–11,000 → kJ) · sodium 2,800–3,800 mg (2.8–3.8 → g) · calcium 700–1,100 mg · vitamin A 500–900
µg RAE (2,000–6,000 → IU) · vitamin D 3–8 µg (120–320 → IU) · folate 350–600 µg DFE (200–350 → food
folate, not DFE) · fiber 14–22 g · height 150–185 cm · weight 55–110 kg.

### The unit-conversion table the app must hard-code correctly [SETTLED — and the pack's #1 embarrassment risk]

- Energy: **1 kcal = 4.184 kJ**
- **Vitamin D: 1 µg = 40 IU** (µg = IU × 0.025)
- **Vitamin A, retinol: 1 IU = 0.3 µg RAE**; dietary β-carotene **12 µg = 1 µg RAE**; other
  provitamin-A carotenoids **24:1**; supplemental β-carotene **2:1**
- Vitamin E: **1 IU natural (RRR-α-tocopherol) = 0.67 mg**; **1 IU synthetic (all-rac) = 0.45 mg**
- Folate: **µg DFE = µg food folate + 1.7 × µg folic acid**
- Niacin: **mg NE = mg niacin + (mg tryptophan ÷ 60)**

**The failure this prevents.** `mcg = IU × 0.025` is the **vitamin D** conversion. Applying it to
vitamin A retinol (correct: ×0.3) is off by **12×**. This exact conflation appears in secondary web
summaries. The app must hard-code per-nutrient factors *with a compound qualifier*, and prefer taking
µg RAE / µg DFE **directly from the source database** over converting.

### Structure detection

- **Duplicate participant IDs → repeated-measures structure.** Detect the occasion variable (`day`,
  `visit`, `recall`, `occasion`, `DR1`/`DR2` prefix pairs, a date column). Wide-format twin columns
  (`*_d1`/`*_d2`, `DR1TKCAL` + `DR2TKCAL`) are the same structure in wide form; offer reshape.
- Report the recalls-per-person distribution. Detect **non-consecutive vs consecutive** days —
  consecutive days are correlated, which inflates apparent between-person variance.
- Detect **day-of-week** and **sequence** (day 1 vs day 2) — both are standard covariates in
  usual-intake models.
- NHANES recall-status flags (`DR1DRSTZ`, `DR2DRSTZ`); status ≠ 1 means the recall did not meet
  minimum criteria.
- **FFQ detection:** many columns of small integers on a bounded frequency scale, or column names
  containing *food* names rather than *nutrient* names → **frequency responses, not grams**; nutrients
  must be derived via a food composition database before modeling.

### Survey design detection

Flag `WTINT2YR`, `WTMEC2YR`, `WTDRD1`, `WTDR2D`, `SDMVPSU`, `SDMVSTRA`, `SDDSRVYR`, or generic
`weight|wt|pweight|strata|psu|cluster|fpc`. If a weight-like column exists but no strata/PSU, flag as
**partially specified design**.

### Body composition

`BMXWT`, `BMXHT`, `BMXBMI`, `BMXWAIST`, `DXDTOFAT`, `DXDTOPF`, or generic
`fat_mass|lean|ffm|bodyfat|bia|dxa|skinfold|waist|hip|whr`. **Recompute BMI from weight/height² and
compare to any declared BMI** (±0.2 kg/m²) — a mismatch means a unit error or a stale derived column.

**★ NHANES DXA multiple-imputation structure [SETTLED, and a silent-failure trap].** NHANES 1999–2004
whole-body DXA public files contain **5 records per participant** because missing DXA was multiply
imputed (missingness related to age, BMI, weight, height — not MCAR). Detect exactly-5× row
multiplicity keyed on `SEQN`. If present and unhandled, **the naive N is inflated 5×.**

### Coaching

> *"Your energy column reconstructs to 4·protein + 4·carbohydrate + 9·fat + 7·alcohol within 3%, so
> units are internally consistent (kcal and grams)."* — Every downstream step (implausibility screens,
> energy adjustment, densities, compositional models) is a function of total energy. A unit error
> there propagates into every result and is invisible in the final tables.

> *"Vitamin A is reported around 3,400 per day, which is the IU range, not µg RAE. I'll convert
> (retinol 1 IU = 0.3 µg RAE) — but only if this column is retinol. If it mixes retinol and
> carotenoids the correct factor differs (12:1 for dietary β-carotene) and you should take the µg RAE
> column straight from the food composition database instead of converting."*

> *"This dataset carries `WTDRD1`, `SDMVPSU` and `SDMVSTRA` — NHANES complex-survey design variables.
> Without them: (a) your point estimates are biased toward the oversampled groups, because NHANES
> deliberately oversamples specific race/ethnicity, age and income groups, so unweighted means are not
> US-population means; (b) your standard errors are too small because clustering within PSUs is
> ignored, so confidence intervals are too narrow. NCHS's own guidance states that variance estimates
> computed under a simple-random-sample assumption are generally too low and biased for NHANES."*

> *"Use `WTDRD1` for Day-1 dietary analyses and `WTDR2D` for anything using both days — **not**
> `WTMEC2YR`. The dietary weights add adjustments for recall non-response and for the deliberate
> weekday/Friday/Saturday/Sunday allocation of recall days."* **[SETTLED]**

> *"To restrict to a subgroup, do not delete rows. Use subpopulation/domain analysis (`subset()` on a
> `svydesign` object; `DOMAIN=` in SAS). Deleting rows drops PSUs and strata from the variance
> calculation and gives wrong standard errors."* **[SETTLED]**

> *"Combining NHANES cycles: divide the 2-year weights by the number of cycles combined, and confirm
> the same dietary methodology applies across them."* **[SETTLED]**

> *"Your DXA table has exactly 5 rows per participant. These are multiple imputations, not repeat
> scans. Analyze all 5 with Rubin's rules and report N from one implicate — otherwise your N is 5×
> too large and your standard errors are far too small."* **[SETTLED]**

### Presentation

- **Data dictionary table (exportable).** column → detected role (energy / macronutrient /
  micronutrient / food group / frequency item / anthropometry / design variable / covariate) →
  detected unit → inferred-vs-declared unit → n missing → median [IQR] → plausibility verdict. This is
  what a nutrition reviewer actually wants to see behind Table 1.
- **Structure card.** Participants, recalls per participant, day-of-week distribution,
  weekday-vs-weekend proportion, day-1-vs-day-2 sequence counts.
- **Design card.** Weight variable in use, sum of weights (should approximate the target population
  size), strata count, PSU count, **minimum PSUs per stratum** — a stratum with 1 PSU breaks
  Taylor-series variance estimation; flag it and offer the standard `lonely.psu` remedies.
- **★ Atwater reconstruction scatter.** Declared energy (y) vs computed energy (x), identity line,
  colored by suspected unit group. Makes a unit error visible in one glance.

### Anti-patterns

Asserting a single IU→µg factor for vitamin A · reporting `Sodium (g)` because the median was 3,400 ·
using `WTMEC2YR` for dietary analyses · row-filtering a complex survey dataset before variance
estimation · silently averaging the 5 DXA implicates · treating FFQ frequency codes as continuous
intake.

---

## 02 · Implausible intake exclusions

### Diagnostic

1. **Fixed kcal screens** — flag against the competing conventions and show how many each catches:
   - Willett / Nurses' Health Study **[CONVENTION]**: women **<500 or >3,500 kcal/d**; men **<800 or
     >4,200 kcal/d**
   - Variants in circulation: men upper bound **4,000** or **5,000**; sex-neutral **500–5,000**;
     sex-neutral **500–3,500**
   - **The app must show the sensitivity of N to the choice**, because the conventions genuinely
     differ across literatures.
2. **EI:BMR ratio.** BMR equations offered with explicit provenance:
   - **Schofield (1985; FAO/WHO/UNU)** — weight-only, age/sex banded, kJ/d: men 18–30 `63·W + 2896`;
     men 30–60 `48·W + 3653`; women 18–30 `62·W + 2036`; women 30–60 `34·W + 3538` (÷4.184 for kcal).
     **This is the equation the Goldberg literature is built on.**
   - **Henry/Oxford (2005)** — recommended over Schofield by several current advisory groups, which
     find Schofield overestimates RMR in contemporary and non-European populations.
   - **Mifflin–St Jeor** — `10·W + 6.25·H − 5·age + 5 (men) / −161 (women)` kcal/d; most accurate at
     the individual level in several validations.
   - **Harris–Benedict (revised)** — men `88.362 + 13.397·W + 4.799·H − 5.677·age`; women
     `447.593 + 9.247·W + 3.098·H − 4.330·age`.
3. **Goldberg cut-off, Black (2000) revision [CONVENTION, the field standard for misreporting].**
   `Cut-off = PAL × exp(±SD_limit × S / 100)` where `S = sqrt(CV²wEI/d + CV²wB + CV²tP)`, with Black's
   revised inputs **CV_wEI = 23%**, **CV_wB = 8.5%** for *estimated* BMR (4% if measured),
   **CV_tP = 15%**, `d` = assessment days, SD_limit = **±2** for 95% limits.
   - The blanket **EI:BMR < 1.35** is Goldberg's "CUT-OFF 1"; it assumes *measured* BMR and is a weak
     screen for individuals.
   - **PAL = 1.55 is not automatically the right anchor.** Sensitivity improves with low/medium/high
     activity strata and stratum-specific PAL.
4. **EI:EER** as an alternative denominator; Huang/McCrory ±1 or ±2 SD approaches.
5. Recall reliability flags, zero-energy days, days with too few foods reported.
6. Special-day flags: "much more / much less than usual", fasting, illness, special diet.
7. **★ Correlates of misreporting — to be tabulated, not just counted.** BMI, sex, age,
   education/income, weight-consciousness. **Under-reporting is systematically, not randomly,
   distributed** — concentrated in higher-BMI and weight-conscious participants, exactly the direction
   that creates spurious diet–obesity associations.
8. **Magnitude context to display.** Doubly-labeled-water validation of the USDA AMPM 24HR found ~**11%
   overall under-reporting** of energy vs expenditure, and **<3% in normal-weight** participants. In
   OPEN, men under-reported by **12–14% on 24HR and 31–36% on FFQ**; women by **16–20% and 34–38%**.

### Coaching

> *"I flagged 214 participants (2.4%) outside the 500–3,500 / 800–4,200 kcal/d range, and 981 (11.1%)
> below the Goldberg lower cut-off using estimated BMR (Schofield) and PAL = 1.55. These are different
> screens catching different people — the fixed-kcal screen catches data-entry failures; Goldberg
> catches biologically implausible reporting relative to body size."*

> *"Exclusion is conventional but it does not fix the problem. Bias in nutrition–health associations is
> *reduced but not eliminated* by Goldberg cut-offs — in one evaluation only 14 of 24 nutrition–outcome
> pairs improved. And stratifying by implausible-reporter status does not reliably reduce attenuation:
> under non-differential error, attenuation can be *greater* in both strata than unstratified,
> contrary to the intuition that estimates among 'plausible reporters' are cleaner."*
> **[SETTLED that exclusion is insufficient; DISPUTED whether to exclude at all]**

> *"Recommended default: (1) drop only unreliable/incomplete recalls and hard-impossible values;
> (2) run the primary analysis on the full sample with energy adjustment; (3) present the
> misreporter-excluded analysis as a prespecified sensitivity analysis; (4) report misreporter status
> as a covariate or stratifier rather than a deletion criterion where feasible."* **[CONVENTION — a
> reviewer in an obesity journal may expect exclusion; the app should say so]**

> *"Because under-reporting is concentrated in people with higher BMI, excluding under-reporters when
> your outcome is adiposity removes a non-random slice of exactly the population you are studying.
> Report the excluded group's characteristics."* **[SETTLED]**

> *"Choose the BMR equation before you look at the results and name it in the methods. Switching
> equations moves your exclusion count."* **[DISPUTED — no consensus on which]**

### Presentation

- **★ STROBE participant-flow diagram with itemized dietary exclusions** — the field expectation, and
  **the single most-checked figure in a nutrition methods review.** Each box carries an n and a
  reason, in order: enrolled → completed ≥1 recall → recall met minimum criteria → eligible age /
  non-pregnant → implausible energy excluded (**with the rule stated in the box**) → missing covariates
  → analytic sample. **[SETTLED that this must appear]**
- **Exclusion sensitivity table:** rule | n excluded | % | resulting N | primary effect estimate under
  that rule. Preempts "what if you'd used a different cutoff?"
- **EI:BMR distribution plot** with the Goldberg cut-off lines drawn, exclusion zones shaded, PAL
  anchor annotated.
- **Misreporting-by-BMI panel.** EI:BMR vs BMI scatter with LOESS, or box plots by BMI category.
  Publication-grade because it *demonstrates* differential misreporting rather than asserting it.
- **Excluded-vs-retained characteristics table.**

### Anti-patterns

"Implausible intakes were excluded" with no rule, equation, PAL or n **[STROBE-nut violation]** ·
claiming exclusion removed misreporting bias · a blanket EI:BMR < 1.35 applied to a sedentary elderly
cohort or to athletes · applying Goldberg to a *single* day without adjusting `d` in the S term (the
cut-off widens as `d` shrinks — a multi-day cut-off on 1-day data over-excludes) · excluding on energy
and then reporting energy as an exposure.

---

## 03 · ★ Repeated recalls and measurement error

### The core statistical fact the pack must teach

A single day is `observed = usual (between-person) + day-to-day + error`. The observed one-day
distribution is therefore **wider** than the true usual-intake distribution. Two consequences, and
they are **different problems**:

1. **Distributional.** Percentiles, and any prevalence computed from the tails (% below the EAR, %
   above the UL), are wrong — over-stated in *both* tails — if computed from one day or a naive
   two-day mean.
2. **Association (regression dilution).** Regressing an outcome on error-prone intake biases the slope
   toward the null by the attenuation factor λ.

Under classical error, for the mean of `n` days:

```
lambda = sigma_b^2 / (sigma_b^2 + sigma_w^2/n) = 1 / (1 + (sigma_w^2/sigma_b^2)/n)
rho(observed mean, true usual) = sqrt(lambda)
days needed for correlation r:  n = [r^2 / (1 - r^2)] * (sigma_w^2 / sigma_b^2)
de-attenuation:  r_corrected = r_obs * sqrt(1 + (sigma_w^2/sigma_b^2)/n)
```

**Empirical magnitudes to quote [SETTLED]:**

- Within:between variance ratios across nutrients have ranged **1.3 to 26.9**.
- Days needed to estimate an individual's intake within **10%** of true mean with 95% confidence:
  **10–35 days for energy and major nutrients**, **15–640 days for micronutrients**. Within 20%:
  **3–9** and **4–160** days.
- **OPEN attenuation factors** (doubly-labeled water + urinary nitrogen as unbiased references):
  absolute energy and protein — **FFQ 0.04–0.16**; **single 24HR 0.10–0.20**; **four 24HRs
  0.20–0.37**. Protein *density* — FFQ 0.3–0.4; single 24HR 0.15–0.25; four 24HRs 0.35–0.50. The
  authors concluded the FFQ **cannot be recommended** for studying *absolute* energy or protein
  intake and disease.
- Sample size for constant power scales as **1/λ²** — λ = 0.3 means roughly an **11-fold** sample-size
  penalty relative to an error-free exposure.

### Diagnostic

- With ≥2 recalls, fit a random-intercept model per nutrient
  (`nutrient ~ 1 + day_of_week + sequence + (1|id)` on a normalizing scale) and report σ²_w, σ²_b,
  the **within:between ratio**, **ICC**, **λ for n = 1, 2, 3, 4 and observed n**, **days needed** for
  r = 0.7/0.8/0.9, and the **implied sample-size inflation 1/λ²**.
- With only 1 recall: report that σ²_w is **not identifiable from these data**, and offer external
  variance-component estimates as an explicitly-labeled assumption — noting that using an incorrect
  within-person-variation ratio biases prevalence estimates.
- Detect **consecutive-day** recalls (correlated error → σ²_w underestimated → λ overestimated → false
  confidence). NHANES uses non-consecutive days by design; a 2-consecutive-day diary does not.
- Detect whether a **calibration substudy** exists (biomarker, DLW, urinary nitrogen/potassium, or a
  reference 24HR on a subsample). If yes, regression calibration is available; if no, say so.
- Detect **repeated FFQs over follow-up** — enables cumulative-average modeling.
- Check **day-of-week** and **sequence** effects.

### Coaching — the decision rule stated explicitly

> **Why two recalls exist:** *"A single 24-hour recall measures one day, not usual diet. Two or more
> non-consecutive recalls let the model separate day-to-day variation from real between-person
> differences. With one recall you cannot do that separation from your own data at all."* **[SETTLED]**

**When simple averaging is adequate [CONVENTION]:**

> *"If your goal is to rank people — regression, classification, quantiles of exposure, a predictive
> model — the mean of your available recalls is an acceptable exposure. It is still attenuated (λ =
> 0.42 with 2 days for this nutrient, so an unadjusted slope is roughly 42% of the true slope), but it
> is unbiased in direction under classical error and it is what most published cohort analyses use."*

**When simple averaging is NOT adequate — three named situations [SETTLED]:**

1. **Any prevalence or percentile claim about usual intake.** Averaging 2 days leaves too much
   within-person variance in the distribution; the tails are too fat, so prevalence of *both*
   inadequacy and excess is over-estimated.
2. **Episodically consumed items** (fish, nuts, organ meats, alcohol, vitamin A from liver). Many
   recorded zeros are non-consumption *days*, not non-consumers. A simple average puts a spike at zero
   that does not exist in usual intake.
3. **When the exposure coefficient must be unbiased in magnitude** — attenuation differs by nutrient,
   so *relative* rankings of effect size across nutrients are distorted, not just shrunk.

> *"Your data have 2 recalls per person and no calibration substudy. That is enough for the NCI method
> to estimate the usual-intake *distribution*, and enough for regression calibration at the population
> level, but not enough to produce a trustworthy usual intake *for an individual*. Do not put NCI
> individual predictions into a Table 1 or a per-person deliverable."* **[SETTLED — NCI explicitly
> warns INDIVINT output does not represent individual usual intake.]**

**The NCI method, concretely:** *"Usual intake is modeled as a latent variable. For nearly-daily
nutrients it is a one-part mixed-effects model: Box–Cox-transform daily intake toward normality, fit a
linear mixed model with a person-level random effect plus covariates (day of week, sequence, mode,
age, sex, FFQ-derived frequency), then back-transform by Monte Carlo integration to recover the
usual-intake distribution with the within-person component removed. For episodically consumed items it
is a two-part model: Part 1 logistic regression for the probability of consumption with a person-level
random effect; Part 2 linear regression for the amount on consumption days, also with a person-level
effect; the two effects are allowed to be correlated. Usual intake = probability × conditional amount.
Implemented in the SAS macros MIXTRAN and DISTRIB; INDIVINT supports regression calibration."*
**[SETTLED description]** Equivalent-standing alternatives: **ISU/ISUF**, **MSM**, **SPADE (RIVM, R)**,
NCI's newer Intake program / SIMPLE macro. **[CONVENTION]**

> *"With repeated FFQs across follow-up, the standard cohort approach is the **cumulative average** —
> averaging all FFQs up to each event time — which reduces within-person error relative to
> baseline-only. Hu et al. (1999, Nurses' Health Study) found cumulative averaging strengthened the
> fat–CHD associations. Caveat: if exposure changes *because* of preclinical disease, cumulative
> averaging imports reverse causation, so a lag or a stop-updating-at-diagnosis rule is
> conventional."* **[CONVENTION]**

### Presentation

- **★ Shrinkage plot — the signature figure of this step.** Overlaid kernel densities of the same
  nutrient: (a) single-day intake, (b) mean of available days, (c) modeled usual intake. **The visible
  narrowing from (a) to (c) is the entire argument for usual-intake modeling in one image.** Annotate
  the 5th and 95th percentiles of each.
- **Variance-components table:** nutrient | σ²_w | σ²_b | ratio | ICC | λ(n=1) | λ(n=2) | λ(observed) |
  days for r = 0.8. Reviewers in this field ask for it.
- **Attenuation panel.** λ as a function of number of days, one curve per nutrient, with the study's
  actual n marked. Makes "why we can't just use one day" self-evident.
- **Day-1 vs Day-2 Bland–Altman** for energy and key nutrients — the field's standard agreement figure.
- **Corrected-vs-uncorrected effect estimates side by side** — crude β, de-attenuated β, with CIs that
  widen appropriately.

### Anti-patterns and documented failures

- **"We averaged the two 24-hour recalls to obtain usual intake"** followed by percentile or prevalence
  claims. **[SETTLED failure]**
- De-attenuating a coefficient without widening the CI to reflect uncertainty in λ.
- Reporting NCI/INDIVINT individual predictions as measured usual intakes.
- Treating consecutive-day records as independent replicates.
- **Assuming FFQ error is classical.** It is not — OPEN showed **person-specific bias correlated with
  true intake and with the reference method's error**, which is why FFQ attenuation for absolute
  energy is as low as 0.04–0.16, and why 24HR-validated FFQs *overstate* their own validity.
- **★ Leakage in a predictive-modeling context (TurboTab-specific).** If a person contributes multiple
  recalls, rows from the same person must never be split across train and test folds — use
  participant-level splitting. Likewise, variance components, λ, NCI parameters, and any residual-based
  energy adjustment must be estimated **inside** the training fold.

---

## 04 · Energy adjustment — the methodological signature

### Why adjust at all — three reasons [SETTLED]

1. **Total energy is a strong determinant of nutrient intake**, so absolute nutrient intake is
   confounded by body size, physical activity, metabolic efficiency and reporting scale.
2. **Diet composition, not absolute quantity, is usually the hypothesis** — nutrients act within a
   caloric budget.
3. **Measurement-error cancellation.** Because nutrient and energy errors are strongly positively
   correlated in self-report, energy adjustment makes the correlated components partly cancel.

Willett's 1997 *AJCN* paper and Chapter 11 of *Nutritional Epidemiology* are canonical. Energy
adjustment is preferable to adjusting for body weight and physical activity as proxies.

### The five models, formally

Let `N` = nutrient, `E` = total energy, `Y` = outcome, `C` = covariates.

| # | Model | Specification | What the coefficient means | Standing |
|---|---|---|---|---|
| 1 | **Standard / multivariate** | `Y ~ N + E + C` | ↑N with total energy fixed → **implicitly a substitution** for the average of all other energy sources | [CONVENTION], very common |
| 2 | **Willett residual** | `N ~ E` (OLS, both usually logged); `N_adj = residual + N̂(Ē)`; then `Y ~ N_adj + C` | Same substitution estimand as #1. The added constant restores units and removes negatives so quantile means are interpretable | [CONVENTION], the field default |
| 3 | **Multivariate nutrient density** | `Y ~ (N/E) + E + C` | Composition, with total energy as a separate term | [CONVENTION] |
| 4 | **Nutrient density alone** | `Y ~ (N/E) + C` | Rescaled relative effect; **interpretation obscure** without the energy term | [CONVENTION but weakest] |
| 5 | **Energy partition** | `Y ~ E_from_N + E_from_other + C` (all kcal) | Effect of **adding** calories from N holding others fixed — an *addition*, not a substitution | [CONVENTION] |

### ★ The result that most surprises practitioners [SETTLED, under-appreciated]

**The standard model and the residual model are mathematically equivalent** — the residual approach
yields the identical coefficient and p-value for the nutrient as including N and E together. Tomova et
al. (2022, *AJCN*, "Adjustment for energy intake in nutritional research: a causal inference
perspective") formalize this and show:

- Standard and residual models estimate the **average relative causal effect** (a substitution) but
  are **biased even absent confounding**. The mechanism is **composite variable bias** — information
  lost when two or more components with distinct effects are collapsed into a single total. That the
  "substituted" mixture is the population-average mixture of all other energy sources is the paper's
  **definition of the estimand**, stated in an adjacent sentence, not the source of the bias.
  *(This bullet read "…, because the substituted mixture is the population-average mixture" until
  2026-08-09. The quoted phrase is near-verbatim from Tomova and is correct; the causal connective
  was not, and it made `PRODUCT_VISION.md` §06c's substitution curve read as a remedy for a bias it
  does not remedy — the total is in the model either way.)*
- The **energy partition model** estimates the **total causal effect**, unbiased only when there is no
  confounding *or* all other nutrients have equal effects.
- The **nutrient density model** has an obscure interpretation.
- All four **only partially account for confounding by common dietary causes** — each evaluates a
  *different estimand*.

Also: an *AJE* (2004) analysis of correlated measurement error found that with energy in the model,
correlated errors can introduce **spurious** nutrient–outcome associations and destabilize estimates.

### Diagnostic

- Regress each nutrient on total energy; report **R²** (macros typically 0.4–0.9; micronutrients
  0.1–0.6). A very low R² means energy adjustment barely changes the variable — report that.
- Report r(N, E) before and after; after residual adjustment it should be **≈ 0 by construction** —
  **verify it**, because a nonzero value means residuals were computed on a different sample than the
  one being analyzed.
- **Sample-boundary check [frequently botched].** Were residuals computed on the *final analytic
  sample*, after exclusions? If exclusions happened after, residuals no longer have mean zero and the
  added constant no longer corresponds to the analytic sample's mean energy.
- **Stratification check.** Pooled residuals across sexes carry a sex effect into the "adjusted"
  variable.
- Compute all five variants and report coefficient, SE, and estimand side by side.
- Check for **energy in the model twice** (density *and* residual *and* total energy).
- Detect whether the outcome is itself energy-related (weight, BMI, adiposity, diabetes) — if so,
  escalate the mediation/collider warning.

### Coaching

> *"Default recommendation: the Willett residual method, computed **within the final analytic sample**
> and **within sex**, on log-transformed nutrient and energy, with the predicted nutrient at the cohort
> mean energy added back. Report the constant you added."* **[CONVENTION — genuinely the field default,
> but not uncontested]**

> *"Interpretation you must write into your results: an energy-adjusted coefficient is a **substitution**
> estimate. It answers 'what if this person got more of their calories from X and correspondingly fewer
> from everything else, at the same total intake?' — not 'what if this person ate more X?'"* **[SETTLED]**

> *"The residual method and simply putting total energy in your model give numerically identical
> nutrient coefficients. The residual method's advantages are practical, not inferential: an adjusted
> variable in interpretable units, uncorrelated with energy, that you can cut into quintiles without
> energy confounding the boundaries. **It does not buy extra confounding control.**"* **[SETTLED]**

> ⚠ *"Your outcome is BMI/adiposity. Total energy is plausibly on the causal pathway from diet
> composition to adiposity, and adiposity causes under-reporting of energy. Conditioning on reported
> total energy here can be simultaneously over-adjustment (mediator) and collider-stratification bias.
> Present both adjusted and unadjusted models, and flag this in limitations."* **[DISPUTED — the app
> must not pick a side]**

> *"Do not categorize total energy and then adjust for the categories. The residual method has been
> shown to be more robust than the standard method precisely when the adjustment variable is
> categorized."* **[SETTLED]**

### Presentation

- **Nutrient-vs-energy scatter** with the fitted line and residuals illustrated as arrow segments,
  annotated with R² and the added constant. Explains the residual method to a reader in one figure.
- **Before/after density panel.**
- **★ Method-comparison forest plot.** The same exposure–outcome estimate under all five
  energy-adjustment models, one row each, **with the estimand named in the row label** ("substitution",
  "addition", "obscure"). Anticipates "why this method?" and demonstrates robustness — or honestly
  shows fragility.
- **Methods sentence generator:** *"Nutrient intakes were energy-adjusted using the residual method
  (Willett & Stampfer), by regressing log-transformed nutrient intake on log-transformed total energy
  within sex in the analytic sample and adding the predicted intake at the sex-specific mean energy
  (men: 2,180 kcal/d; women: 1,720 kcal/d) to the residuals."*

### Anti-patterns

"We adjusted for energy" without naming the model **[STROBE-nut violation]** · interpreting an adjusted
coefficient as an addition effect · including density *and* energy *and* the residual · computing
residuals on the full cohort then analyzing a subset · **fitting the `N ~ E` residual regression on
train+test before cross-validation** · reporting "energy-adjusted" for a nutrient whose R² on energy
is 0.02 as though it were meaningful.

---

## 05 · Compositional structure and substitution modeling

### Diagnostic

- **Closure detection [SETTLED must-have].** Test whether a set of columns sums to a constant:
  `%E_protein + %E_carb + %E_fat + %E_alcohol ≈ 100` (±2%, allowing for fiber/organic acids/rounding),
  and whether food-group columns sum to total energy or to 100%.
- **Rank-deficiency check.** If closure is detected, warn that a design matrix containing *all* parts
  (± the total) is **singular** — the fitter will silently drop a term, alias it, or blow up variance
  inflation. Report the condition number and which term was dropped.
- **Unit compatibility.** Substitution modeling requires all components in the **same unit that sums
  to the total** — kcal for isocaloric substitution, grams for gram-for-gram food substitution. Mixed
  units yield an estimand the literature calls obscure and potentially extremely misleading.
- Detect zeros in any component intended for log-ratio analysis; **alcohol is the usual offender.**
- Detect a **residual "other" category** — if food groups do not exhaust total energy, substitution is
  ill-defined.

### Coaching

> *"Your macronutrient percentages sum to 100. That means they are **compositional**: they carry only
> relative information, and one part is fully determined by the others. Ordinary regression on parts of
> a whole is not just collinear — it is misleading, because a coefficient on 'percent energy from fat'
> has no meaning until you say what the fat replaced."* **[SETTLED]**

> **Three legitimate ways forward:**
> **(a) Leave-one-out substitution model** — all components except one, plus total energy, all in kcal.
> Each coefficient is the effect of substituting that component for the omitted one. **Name the omitted
> component in your results.** This is the approach that established the SFA/trans → MUFA/PUFA
> replacement findings for coronary heart disease.
> **(b) Compositional data analysis** — isometric log-ratio (ilr) coordinates as covariates. Each is an
> interpretable *balance*, and the model is full-rank by construction. Requires a zero-handling rule.
> **(c) Explicit difference-in-coefficients** — fit components as separate terms and report `β_A − β_B`
> with a variance accounting for their covariance.
> **[CONVENTION for (a) and (c); (b) is emerging and increasingly respected but less familiar to
> reviewers]**

> ⚠ *"Adjusting for total energy does not by itself make a substitution isocaloric. The components must
> also be measured and analyzed in calories. If your components are in grams or servings, the
> 'substitution' you report is not the one you think it is."* **[SETTLED — the documented failure mode
> of the substitution literature]**

> ⚠ *"A review of 100 substitution-modeling studies (2018–2024, 21 countries) found 53% used
> **unvalidated** FFQ-derived variables; among those reporting validation, correlation coefficients
> with reference methods ranged **0.12 to 0.77 (median 0.43)**; 62% provided minimal or no
> documentation; and in some cases deviations from reference values exceeded 450%. Studies with
> unvalidated inputs were frequently published in high-impact journals."* **[SETTLED as a documented
> failure of the literature]**

> *"Song & Giovannucci (2018) title their paper 'proceed with caution' for a reason: the substituted-for
> component is a modeling choice, not a fact in the data."*

### Presentation

- **★ Isocaloric substitution forest plot** — the field's expected figure. Rows of the form "5% of
  energy from X replaced by Y", estimate + 95% CI, null reference line, grouped by replaced component.
  **State the substitution magnitude and unit in the axis title.**
- **Ternary plot** of the three-part macronutrient composition, colored by outcome or covariate
  tertile. Makes the simplex visible.
- **Composition stacked-bar by exposure quantile**, showing the constraint.
- **Closure verification table** in the supplement.
- If CoDA: **balance dendrogram / sequential binary partition diagram** naming each ilr coordinate.

### Anti-patterns

`Y ~ pct_protein + pct_carb + pct_fat + pct_alcohol + total_energy` — singular, and whatever survives
is uninterpretable · reporting a %-energy coefficient with no named substituted component · mixing g,
servings and % energy in one substitution model · adding an arbitrary ε to zeros without reporting it ·
**running LASSO over a closed composition and reporting "the model selected fat"** — the selection
among perfectly-dependent parts is arbitrary.

---

## 06 · Missing data

### Diagnostic — four distinct dietary missingness types, labeled per column

1. **Item non-response on an FFQ** (blank food line) — conventionally coded 0
2. **Structural zero vs episodic zero on a 24HR** — a 0 g day for fish is a *day* without fish, not a
   non-consumer
3. **Whole-recall missing** (day 2 not completed) — the classic NHANES pattern
4. **Nutrient not in the food composition database** — often materializes as 0 or NA for an entire
   nutrient in a subset of foods, producing a systematically low derived intake

Also: quantify FFQ item non-response per participant and flag above the study's blank-item threshold ·
day-2 completion rate, comparing day-1 characteristics of completers vs non-completers (this is the MAR
evidence, and it is what `WTDR2D` exists to partly correct) · the NHANES DXA **5-implicate** structure ·
**device/method mixing** across sites or visits for BIA/DXA (a hidden "missing method" variable).

### Coaching

> *"Blank FFQ items: the field convention is to code them zero. That convention has a documented cost —
> zero imputation introduces bias because not all blanks are true zeros, and the bias grows with how
> commonly the food is consumed. Zero imputation creates little bias except for frequently consumed
> foods, where it is suboptimal once more than 5–10% of items are missing."* **[CONVENTION with
> documented limits]**

> *"Missing FFQ values are often **missing not at random** — a blank frequently means 'never',
> especially in older participants and in those with many blanks. Standard MI assumes MAR, so MI is an
> improvement, not a solution."* **[SETTLED that the mechanism is non-ignorable; DISPUTED how to
> handle it]**

> *"A zero on a 24-hour recall is a zero **for that day**. Do not read it as 'this person never eats
> fish', and do not let a model treat the zero spike as a real feature of usual intake. This is exactly
> the case the NCI two-part model exists for."* **[SETTLED]**

> *"Complete-case analysis after listwise deletion across 30 nutrients plus body composition will
> silently remove a large, non-random fraction of your sample. I dropped from N=8,506 to N=4,918 (42%)
> — here is who was dropped."*

### Presentation

Missingness heatmap and UpSet co-missingness plot · **missing-vs-observed comparison table**
(characteristics with vs without day-2 recall, with vs without DXA — this is what justifies MAR) ·
imputation diagnostics (observed vs imputed distributions overlaid) · **sensitivity strip**: primary
estimate under complete case / zero-imputation / MI / MI-with-delta, as a small forest plot · the
exclusion cascade in the STROBE flow diagram.

### Anti-patterns

Silent listwise deletion with no N cascade · **imputing nutrient intakes without including total energy
in the imputation model** (breaks the energy–nutrient relationship every downstream step depends on) ·
imputing then energy-adjusting without propagating imputation uncertainty · treating the 5 DXA
implicates as 5 observations · pooling BIA-derived and DXA-derived fat mass into one column.

---

## 07 · EDA and presentation

The section that determines whether the output looks like nutritional epidemiology or like generic data
science.

### Diagnostic — what to compute before drawing anything

Per nutrient: n, missing, median [IQR], mean ± SD, skewness, kurtosis, % zeros, min/max, normality
verdict on raw and log scale · whether values are **per day**, **per 1000 kcal**, or **% energy** —
label every axis accordingly · whether **supplements** are included in totals (compute both if
supplement data exist) · Spearman correlation matrix among **energy-adjusted** nutrients with
hierarchical clustering · for each nutrient with a DRI: **EAR, RDA/AI, and UL for the participant's
age/sex/pregnancy/lactation stratum** — these are stratum-specific and pregnancy and lactation have
separate DRIs · body composition: BMI, FMI, FFMI, %BF, waist, WHR, WHtR, and **verify the identity
BMI = FMI + FFMI** as a free data-quality check.

### The figure catalogue

**A. Per-nutrient distribution plots [SETTLED that these appear].** Histogram + density, raw and log₁₀
side by side, median and IQR marked, n and unit in the axis label. *Publication-grade requires:* units
stated as "per day"; explicit note whether supplements are included; **a marker for whether the plotted
variable is a single day, a mean of days, or modeled usual intake**; a companion percentile table
(5/10/25/50/75/90/95) with standard errors when survey-weighted.

**B. Energy-adjusted distributions.** Paired panel, crude vs adjusted, same units, with the added
constant annotated and the stratification stated.

**C. ★ Nutrient correlation structure with hierarchical clustering — the highest-value EDA figure in
this pack.** Spearman heatmap of **energy-adjusted** nutrients, ordered by dendrogram, diverging scale
centered at zero.

> *Why it matters and what to say about it:* **nutrients cluster by food source, not by biology.** The
> typical structure: an **animal-source cluster** (protein, B12, zinc, heme iron, retinol, saturated
> fat, cholesterol), a **plant-source cluster** (fiber, folate, magnesium, potassium, vitamin C,
> vitamin K, carotenoids), a **refined-grain/added-sugar cluster**, and a **fortification cluster**
> (folic acid, thiamin, riboflavin, niacin, iron in fortified-grain economies).

*Publication-grade requires:* energy adjustment before correlating (otherwise the whole matrix is
dominated by a single "total intake" factor); Spearman rather than Pearson; and **a caption that names
the food-source interpretation of each cluster** rather than leaving it to the reader.

**D. Dietary pattern derivation — four routes, each with its own figure set:**

| Method | Figures | Publication-grade requirements | Standing |
|---|---|---|---|
| **PCA / EFA** | scree plot with the chosen cut marked; **factor loading heatmap** with loadings ≥ 0.20 (or 0.30) highlighted; % variance per factor and cumulative; score distribution | Prespecify: input variables (**food groups, not nutrients**, conventionally ~30–40), whether energy-adjusted/standardized, number of factors and criterion, rotation (varimax conventional), pattern names. Report that first-few-factor variance explained is modest (often ~15–30% total) — **do not hide it**. State that naming is subjective. | [CONVENTION] |
| **Reduced rank regression** | loading plot per pattern; **explained variation in the response variables *and* the predictors**; validation in an independent/split sample | Response variables chosen **a priori** and justified. RRR patterns are population-specific and reproduce poorly — **validation is not optional.** | [CONVENTION; response choice DISPUTED] |
| **Cluster analysis** | cluster-mean profile heatmap; cluster sizes; silhouette or gap statistic; stability across resamples | **Energy-adjust or standardize inputs first** — otherwise clusters recover total energy intake, not diet pattern. | [CONVENTION] |
| **Index / a priori scores** | radar or bar chart of components; total-score distribution; component contribution | Name the index and version (HEI-2020, AHEI-2010, DASH, aMED, DII, PHDI), the scoring method, the food-group mapping. **HEI-2020**: 13 components, most scored **per 1,000 kcal**, fatty-acid component as the **(PUFA+MUFA)/SFA ratio**, total 100 points. **State which HEI scoring approach was used** — *simple/per-person* (needed for individual-level associations) vs the *population ratio method* — they are not interchangeable. Tooling: the `dietaryindex` R package. | [SETTLED for HEI mechanics; CONVENTION for which index] |

**E. ★ Intake distribution against reference intakes.** Usual-intake density or CDF per nutrient with
**vertical lines at the EAR, RDA (or AI), and UL** for the relevant stratum; the area below the EAR
shaded and labeled with the prevalence estimate and its CI. **The rules that make it correct rather
than embarrassing [SETTLED]:**

- The distribution must be a **usual-intake** distribution, not one day and not a naive 2-day mean.
- **EAR cut-point method:** prevalence of inadequacy = proportion with usual intake below the EAR.
  Valid when the requirement distribution is symmetric, intake and requirement are independent, and
  intake variance exceeds requirement variance.
- **Exceptions that must be hard-coded:** **iron in menstruating women** has a skewed requirement
  distribution → use the **probability approach**, not the cut-point. **Energy** has no EAR-style
  cut-point (use EER). **Nutrients with only an AI** (fiber, potassium under some DRI versions)
  **cannot yield a prevalence of inadequacy** — the app must **refuse to compute one** and say why.
- The **UL** side: report % above the UL only for total intake including supplements.
- Reference intakes join on age band, sex, pregnancy **and** lactation status.

**F. Body composition figures.**

- **★ FMI vs FFMI scatter with BMI iso-lines** (Hattori-style). Because **BMI = FMI + FFMI** exactly,
  lines of constant BMI are −45° diagonals and lines of constant %BF are rays through the origin.
  Sex-stratified, with reference percentile bands. *Reference values to overlay:* FFMI 25th–75th ≈
  **18.7–21.0 kg/m² (men)** and **14.9–17.2 (women)** in an Italian DXA population; 5th–95th ≈
  **16.3–22.3** and **13.3–17.8** in a Korean DXA reference. **State the reference population** —
  these are population- and method-specific.
- **Bland–Altman (BIA vs DXA)** if both exist. Annotate the known direction: BIA **overestimates
  fat-free mass by ~3.4–8.3 kg** and **underestimates fat mass by ~2.5–5.7 kg** vs DXA across BMI
  18.5–40, device-dependent.
- **BMI vs %BF scatter** showing the misclassification band — BMI correctly classified only ~63% of
  males and ~67% of females against body-composition criteria in one comparison.
- **Sarcopenia quadrant plot** if ASM/ASMI is available, with consensus cut-points annotated (EWGSOP2
  ASM/height²: <7.0 kg/m² men, <5.5 women; AWGS2019 DXA: <7.0 / <5.4; BIA: <7.0 / <5.7).

**G. Dose–response figure.** **Restricted cubic spline** of outcome vs energy-adjusted intake: 3–5
knots at conventional percentiles (**3 knots at 10/50/90**, or 4 at 5/35/65/95), reference at a stated
percentile, shaded 95% CI band, **a rug or histogram of the exposure distribution underneath**, and a
reported **p for non-linearity**. Truncate at the 1st–99th percentile. (Desquilbet & Mariotti, *Stat
Med* 2010.) **[CONVENTION, now near-default; quintiles remain expected alongside]**

**H. The nutrition "Table 1" [CONVENTION, near-universal].** Baseline characteristics **by quantile
(usually quintile) of the energy-adjusted primary exposure**, continuous as mean ± SD or median [IQR],
categorical as n (%), and — the field-specific touch — **age-standardized for all variables except age
itself**. **Include energy intake in the table so the reader can verify the adjustment worked** (energy
should be similar across quintiles of a residual-adjusted exposure).

### Coaching

> *"I energy-adjusted before computing the correlation matrix. Without that step the matrix mostly
> measures 'ate more of everything' — the first principal component of raw nutrients is essentially
> total energy, and every nutrient pair looks correlated at r > 0.6."* **[SETTLED]**

> *"Your nutrients cluster into an animal-source group and a plant-source group. That is the central
> interpretive fact of nutrient analyses: an association with any single nutrient in a cluster is,
> statistically, an association with the food source. Say so in your discussion rather than letting a
> reviewer say it for you."*

> *"Before I compute '% below the EAR', I need a usual-intake distribution. Applying the EAR cut-point
> to single-day or 2-day-mean intakes will overestimate the prevalence of inadequacy — the tails are
> too fat — and simultaneously overestimate the proportion above the UL."* **[SETTLED; the single most
> common quantitative error in nutrition surveillance analyses]**

> *"Fiber has an AI, not an EAR. I can show the distribution against the AI, but I cannot compute a
> prevalence of inadequacy from an AI, and neither can anyone else."* **[SETTLED]**

> *"For iron in menstruating women I will use the probability approach rather than the EAR cut-point,
> because the iron requirement distribution is skewed."* **[SETTLED]**

> *"BMI cannot distinguish fat mass from lean mass. If you have DXA or BIA, report FMI and FFMI, not
> just %BF — %BF is a ratio whose denominator includes muscle, so a high %BF can mean excess fat or lost
> muscle. FMI separates them."* **[SETTLED]**

### Anti-patterns

Correlation heatmap on **unadjusted** nutrients presented as "nutrient relationships" · EAR cut-point on
one-day intakes · prevalence of inadequacy computed against an **AI** or against the **RDA** (an
individual-level target, not a group cut-point) **[SETTLED error]** · PCA on nutrients rather than food
groups without saying so (nutrient-level PCA recovers food-composition-table structure, not eating
behavior) · scree plot without the retained-factor decision marked · k-means on unstandardized intakes
→ clusters = energy tertiles · **RCS plot without the exposure distribution shown**, so the reader
cannot see the dramatic upturn is driven by 11 people · reporting BMI categories as "body composition" ·
overlaying DXA-derived FFMI reference percentiles onto BIA-derived values.

---

## 08 · Feature selection and modeling

### Diagnostic

Pairwise |r| among candidates (energy-adjusted), reporting all pairs above 0.7 and 0.9 · **VIF**, and
for relative-risk regression the **condition indices and variance-decomposition proportions** · EPV,
flagging <10 · **effective sample size under survey weights (Kish design effect)** — the effective N for
power can be far below the row count · **selection frequency** of each nutrient across bootstrap
resamples · λ-adjusted power (∝ 1/λ²) · the compositional/closure trap among selected features ·
participant-level grouping before any CV split.

### Coaching

> *"Sixteen of your 30 nutrients have a pairwise correlation above 0.7 with at least one other. In this
> situation automatic selection is not identifying a causal nutrient — it is arbitrarily picking one
> marker of a shared food source. Across 200 bootstrap resamples, LASSO selected magnesium 41% of the
> time, potassium 37%, and fiber 33%; they are interchangeable. **Report selection frequency, not the
> single selected set.**"* **[SETTLED behavior; the presentation recommendation is CONVENTION]**

> *"Collinearity in diet data is not a nuisance to be dispatched — **it is the biology.** The four
> established remedies are: (1) drop collinear variables and analyze one at a time with clear reporting,
> (2) express components as proportions of energy, (3) use the residual approach to purge shared
> variance, (4) penalized regression. **None of them creates causal identification.**"* **[SETTLED as a
> description of the options]**

> **Dietary pattern scores vs individual nutrients — a real trade-off, not a right answer:**
> *Patterns* collapse collinearity into a few stable, interpretable, translatable variables, better
> reflect how people actually eat, and are usually better powered — at the cost of population-specific
> derivation that often does not reproduce (RRR especially), meaning that depends on the food-group
> list, and lost mechanistic specificity. *Single nutrients* give a mechanistic hypothesis and
> cross-study comparability — at the cost of severe collinearity and a coefficient confounded by the
> whole food source it marks. A common defensible strategy: **lead with a prespecified pattern/index
> and present single-nutrient models as secondary.** **[DISPUTED — genuine, ongoing disagreement]**

> *"Your exposure's attenuation factor is λ = 0.35. Relative to an error-free exposure, achieving the
> same power requires roughly 1/λ² ≈ 8× the sample size. This is why null results in nutrition are hard
> to interpret. Increasing N is an incomplete fix — the attenuated signal can become too small to
> distinguish from unmeasured confounding."* **[SETTLED]**

> *"Predictive-modeling hygiene for this data type: **split by participant, not by row**; fit
> energy-adjustment residual models, PCA/RRR loadings, standardization, and imputation **inside the
> training fold only**; and if survey weights are present, decide explicitly whether you are estimating
> a population-generalizable model (use weights, with design-based or replicate-weight resampling for
> validation) or a within-sample predictor (unweighted, but say so)."* **[SETTLED for ML practice; the
> weighted-CV question is DISPUTED]**

### Presentation

**Bootstrap selection-frequency bar chart** for every candidate · **correlation-clustered feature map**
with selected features highlighted, so the reader sees which cluster each came from · **nested model
table**: Model 1 (age, sex, energy) → 2 (+ demographics/lifestyle) → 3 (+ body composition) → 4
(+ mutually adjusted nutrients) — the stepwise-adjustment table is the field convention and reviewers
expect to watch the estimate move · **quintile table + spline figure side by side**, with **p for trend
using the median of each quintile as a continuous score, not the quintile number** · forest plot of
subgroup/sensitivity analyses with prespecified/post-hoc status marked · calibration plot and
discrimination metric if this is genuinely a prediction model.

### Anti-patterns

Stepwise selection over 40 correlated nutrients, reporting the survivors as "diet predictors of X" ·
**p-for-trend computed on quintile rank codes 1–5** when the underlying intakes are unevenly spaced ·
adjusting for BMI when BMI is a mediator without saying so · ignoring that with correlated measurement
error and energy in the model, **spurious** associations can be induced · reporting a null and
concluding "no association" without a measurement-error-aware power statement · **row-level train/test
split with repeated recalls.**

---

## 09 · Reporting standards

**STROBE-nut** — Lachat C, et al., *PLoS Medicine* 2016;13(6):e1002036. Developed by 21 experts with a
3-round Delphi involving 53 external experts; adds **24 nutrition-specific recommendations** to STROBE;
applies to cohort, case-control and cross-sectional studies. Explanation & Elaboration: *Adv Nutr*
2017;8(5):652. In the author instructions of a growing set of journals.

### The checklist engine — what the app can auto-fill vs what only the user can supply

| Requirement | Source |
|---|---|
| Dietary assessment method, instrument name/version, mode, number of administrations, time frame | user |
| Whether validated; **the reference method, when, and in what population**; validity statistics; whether reproducibility was tested | user + app template |
| **Food composition database and version** (e.g. USDA FNDDS release year), and how non-matching foods were handled | user |
| Units, and whether intakes are per day / per 1000 kcal / % energy | **app** |
| Whether **supplements** were included, and how doses were quantified | app detects + user confirms |
| **Energy adjustment method, with the model named** | **app** |
| **Misreporting handling:** definition, equation, cut-off, PAL, n excluded | **app** |
| Usual-intake handling: number of recalls, method, covariates | **app** |
| Missing data: mechanism assumed, method, n affected | **app** |
| Complex survey design: weights, strata, PSU, subpopulation handling, variance method | **app** |
| **Nutrient requirement standards** (which DRI edition/country) and the inadequacy method | **app** |
| Participant flow with dietary exclusions | **app** |
| Population context: country, food supply, fortification policy, season | user |
| Data and code availability | user |

### Coaching

> *"A nutrition reviewer reads your methods in a fixed order and checks six things: (1) what instrument,
> administered how many times, covering what period; (2) what food composition database and version;
> (3) what validation exists **in a population like yours, with the actual correlation coefficients**;
> (4) how you handled misreporting, with the equation and cut-off; (5) which energy adjustment model,
> named; (6) how within-person variation was handled. Missing any one is the most common reason for a
> methods revise-and-resubmit in this field."*

> *"Report your validation coefficients as numbers, not adjectives. 'The FFQ was previously validated'
> is not reporting. In the substitution-modeling literature, published validation correlations range
> from 0.12 to 0.77 with a median of 0.43 — a reader cannot judge your study without knowing where in
> that range your instrument falls."*

> **Limitations paragraph the field expects — the app drafts it:** *(a) self-reported intake is subject
> to substantial random and systematic error, with attenuation factors for absolute intakes as low as
> 0.04–0.20 depending on instrument; (b) residual confounding by correlated dietary and lifestyle
> factors cannot be excluded, because nutrients cluster by food source; (c) energy adjustment yields a
> substitution estimand under stated assumptions; (d) [cross-sectional] temporality cannot be
> established; (e) findings apply to the food supply, fortification policy and dietary patterns of
> [setting].*

### Presentation

Rendered **STROBE-nut checklist** with section anchors, auto-filled where known, unfilled items
highlighted, downloadable as the supplementary file journals request · auto-generated methods paragraph
with every numeric decision inlined · STROBE participant flow diagram · **analysis provenance card**:
package versions, seeds, weight variable, design specification, and **the exact residual-adjustment
constants** — so the analysis is reproducible from the paper.

### Anti-patterns

"Dietary intake was assessed by a validated FFQ" as the entire instrument description · naming no food
composition database · "intakes were energy-adjusted" with no model · "implausible reporters were
excluded" with no rule · **reporting a nutrient in µg without saying RAE/DFE/NE** · NHANES analysis with
no statement of weights, strata, PSU and variance method.

---

## 10 · Where confident automation would embarrass us

Hard-coded guardrails with citations, not heuristics.

1. **Vitamin A IU↔µg RAE** (retinol 1 IU = 0.3 µg RAE; dietary β-carotene 12:1) vs **vitamin D**
   (1 µg = 40 IU). Routinely conflated in secondary sources; conflating them is a **12× error**.
2. **Schofield ≠ Harris–Benedict.** Schofield is weight-only and age/sex-banded. The
   `88.362 + 13.397·W + …` form is revised Harris–Benedict. Web summaries mislabel this; naming the
   wrong equation in a methods section is a citable error.
3. **The EAR cut-point applies to usual intake only**, requires an EAR (not an AI), is not the RDA, and
   fails for iron in menstruating women.
4. **The residual model is mathematically identical to the standard model** for the nutrient
   coefficient. Claiming it "removes confounding by energy" the standard model does not is wrong.
5. **Energy adjustment yields a substitution estimand.** "The effect of eating more X" is wrong.
6. **NHANES dietary weights are WTDRD1/WTDR2D, not WTMEC2YR**, and subsetting must be domain analysis.
7. **NHANES DXA public files have 5 imputed records per person.**
8. **Excluding misreporters does not remove bias** — reduced, not eliminated, and stratification can
   worsen attenuation in both strata.
9. **"500–3500 / 800–4200 kcal" is a convention, not a standard.** Show the sensitivity.
10. **Compositional closure:** never fit all parts plus the total.
11. **FFQ error is not classical.** Validation studies using 24HR as the reference overstate FFQ
    validity because the errors are correlated; only biomarker-based studies give unbiased attenuation.
12. **`WTDRD1` sums to a population, so unweighted NHANES means are not US means.**

### Verification gaps — re-verify from primary PDFs before shipping

All WebFetch calls in the research session returned HTTP 403 from the egress proxy. **Not directly
verified:** the exact numbering and verbatim text of individual STROBE-nut checklist items (only the
count — 24 — the development process, and the substantive content areas were confirmable); the exact
NCI *User's Guide* parameterization of MIXTRAN/DISTRIB/INDIVINT beyond the documented model structure;
the precise NHANES variable names for the 2021–2023 cycle.

**Before shipping, re-verify:** (a) the full STROBE-nut item list from the *PLoS Med* 2016 supporting
information; (b) Black AE, *Int J Obes* 2000;24:1119–30 for the exact Goldberg/Black cut-off algebra
and the SD convention; (c) the NCI *User's Guide for Analysis of Usual Intakes* v2.1; (d) **current DRI
tables (EAR/RDA/AI/UL by age/sex/pregnancy/lactation) from the NASEM DRI database — which the pack must
ship as data, not as text.**
