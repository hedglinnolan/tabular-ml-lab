# The modeling sequence — everything after the seal

**Status: DRAFT.** Written by the orchestrator as the project's methods expert (Nolan, 2026-10-02:
*"I am asking you to be one"*). Adversarial review against the primary literature is owed before
anything is built on it (§7).

**What this covers.** Everything from the drawn seal to a declared, calibrated, compared result. In
Classic this was EDA → feature engineering → feature selection → model-specific preprocessing →
train and compare. It is the densest concentration of intelligence in the app.

**What governs it:** BLUEPRINT §12 (Nolan's rulings: residual keeps energy; all-components first for
substitution under inference; the seal is purpose-scoped; imputation by purpose), North star 5
(customary vs sound, as separate labels), §11.3 (the leash), §13 (the method contract and its
relations), and the lockbox constitution §06 (the data-scope test).

**The organizing fact.** Purpose splits this segment more than any other. **Prediction** may learn
choices from the training data, as long as the choice is made *inside* the folds. **Inference** may
not choose its model from the data it will report on: the exposure, the estimand, the adjustment set
and the functional form are declared before estimates are seen, and anything the data chose is
reported as such. The same five Classic stages therefore become two related sequences, not one.

---

## 1 · The sequence

| # | Step | Fires | Prediction | Inference |
|---|---|---|---|---|
| 1 | **Explore** | always | stack of findings on training rows only, each tied to a lever below | the same, except that outcome-relationship views are *recorded as looked at* and the methods state it (forking-paths disclosure) |
| 2 | **Exposure and estimand** | inference | — | *"Which exposure, and which effect?"*: total vs direct; for nutrients, substitution vs addition (energy partition); per what unit. No coefficient is shown before this is answered |
| 3 | **Adjustment set** | inference | — | a role for each covariate: confounder · precision · mediator · collider · not relevant. Mediators in a total-effect model are blocked and recorded; the disjunctive cause criterion is offered as guidance. **No data-driven confounder selection** |
| 4 | **Functional form** | a continuous exposure (inference) or continuous predictors with nonlinearity evidence (prediction) | linear · restricted cubic spline (knots at the pack's percentiles) · offered from evidence on training rows | declared before estimates: linear · spline with an overall and a nonlinearity test · categories for *presentation* (quintiles with a trend test on medians) — never categories as the analysis model by default |
| 5 | **Domain transforms** (method contracts) | by lens and roles | energy adjustment (choice matters little: said so) · omics normalization in-fold · scale scores · ilr coordinates | energy model with its estimand (all-components first for substitution) · omics normalization · scale scores with reliability and attenuation · regression calibration when there are repeated recalls |
| 6 | **Missing data** (asked before the seal, executed here) | missing predictors exist | in-fold, outcome-free imputation (deployable) | multiple imputation with the outcome and energy, m ≥ 20, Rubin's rules; complete cases with its assumption stated |
| 7 | **Feature construction** | prediction, or declared interactions under inference | domain features (contracts), declared interactions, in-fold PCA for omics; each states its scope | only declared, scientifically motivated interactions (effect modification), stated in the methods |
| 8 | **Selection** | prediction | none (all candidates) · penalized (elastic net selects) · stability selection · in-fold screening at p ≫ n (genomics). **Never on the full data** | not offered: a data-driven selection is *refused* for the reported model (post-selection inference); allowed only as a labeled sensitivity analysis |
| 9 | **The shelf** | always | ordered by expected performance for the data's shape (n, p, events), each family's inductive bias stated; the **price of explainability** measured after the fit | ordered by interpretability of the declared estimand (regression families first: linear, logistic, Cox, mixed models/GEE for repeated units, design-based for surveys); flexible learners offered for sensitivity or as nuisance models |
| 10 | **Model-specific preprocessing** | not a question | stated per family in the lineage (scaling, encoding, native missing values); overridable in an advanced disclosure | the same, plus the scale the coefficients are reported on |
| 11 | **Tuning and comparison** | stated | nested CV on the same folds; paired differences with intervals; selection optimism estimated; calibration; primary metric by task (proper scores first) | not a comparison: one declared model. Alternative specifications (energy model, exclusion rules, MI vs complete cases) are *declared secondary analyses*, shown side by side and labeled |
| 12 | **Declaration** | before opening the seal (prediction) / before showing estimates (inference) | the final family is declared on CV, then the holdout is opened once | the analysis plan is locked; any change after estimates are seen is recorded "after the estimates were seen" |

## 2 · Relations: how one action leads to another (§13)

- **exposure declared** *enables* the effect display and the estimand sentence; *invalidates* the
  adjustment-set answers that were given for another exposure.
- **adjustment set** *conflicts with* data-driven selection under inference.
- **energy model** *constrains* substitution: a substitution curve needs every energy source in the
  model, and an omitted source is a stated concern (block and record under inference). The
  all-components model *enables* substitution of any pair.
- **log transform of a nutrient** *changes what a substitution means* (ratio rather than
  difference) — the curve's label says so.
- **spline** *replaces* the single coefficient with a curve plus an overall test and a
  nonlinearity test (the table row becomes a figure).
- **multiple imputation** *invalidates* median fill for the inference table and *implies* pooled
  intervals (Rubin's rules) everywhere a coefficient is shown.
- **omics raw values** *conflict with* linear families until a normalization is chosen or the
  values are declared normalized.
- **QC drift correction** (before the seal; reference-rows scope) *implies* that the QC rows leave
  the cohort, and *precedes* in-fold normalization.
- **repeated units** *imply* cluster-aware intervals or mixed models/GEE; i.i.d. intervals on
  repeated rows are refused.
- **survey design under inference** *implies* design-based estimation, or a recorded attestation
  of a sample-only estimand.

## 3 · What the canvas shows at each step (pedagogy, §11.2)

Every view below comes from the existing closed vocabulary; nothing needs a bespoke picture.

| Step | Primary view | The question it answers |
|---|---|---|
| Explore | distribution, relationship, table focus | is my data okay; what will matter |
| Exposure and estimand | lineage (exposure highlighted) and a one-line estimand | what is this choice |
| Adjustment set | lineage with roles (confounder/mediator lanes) | what will it change |
| Functional form | relationship with the fitted form drawn (linear vs spline), storyboard: the fitted line bends | what will it change in my model |
| Domain transforms | the method's own storyboard (energy: fit the line → residuals → add back the mean) | what will it change in my data |
| Selection | lineage: the columns that leave, per fold | what will it change in my model |
| Shelf / comparison | metric comparison with the baseline and the price of explainability | why it matters for my result |
| Declaration | the record: the locked plan, one sentence per decision | what did I decide; can a reviewer reproduce it |

## 4 · The leash, by step

| Step | Prediction | Inference |
|---|---|---|
| Explore of outcome relationships | right as is (training rows only) | longer leash is wrong: *record and disclose*, never block |
| Exposure/estimand | — | **required**: no estimates until answered |
| Mediators in a total-effect set | — | **block and record** |
| Data-driven selection | allowed in-fold | **refuse** for the reported model; sensitivity only |
| Categories as the analysis model | rank lower, state the information loss | **block and record**, offer splines (categories for presentation) |
| Omitted energy sources in substitution | state the concern | **block and record** above a stated omitted share |
| Single split at small n | rank lower than bootstrap optimism, state why | not applicable (no holdout) |
| Median fill | allowed in-fold | **block and record** for the inference table |

## 5 · What already exists, and what this specifies new

Built or being fixed by the audit loops: per-family preprocessing, the shelf, nested tuning, calibration and optimism (WP9), declared final model (WP8), splines/quintiles/ordinal (WP12a), Cox and mixed models (WP12b), MI by purpose (WP7), energy estimands (WP6). **New from this spec:** the Explore stack after the seal with forking-paths disclosure; the exposure/estimand and adjustment-set questions (WP17 begins them); functional-form as a declared question; selection as a purpose-routed question; secondary analyses as declared objects; the analysis-plan lock; the price of explainability; the relations in §2 as contracts with a chain test.

## 6 · Sources to verify in review

VanderWeele 2019 (confounder selection, disjunctive cause criterion); Westreich & Greenland 2013
(Table 2 fallacy); Gelman & Loken 2013 (garden of forking paths); Harrell, *Regression Modeling
Strategies* (splines, knots, against categorization; bootstrap validation); Altman & Royston 2006
(the cost of dichotomizing); Ambroise & McLachlan 2002 (selection bias in CV); Berk et al. 2013 /
Taylor & Tibshirani 2015 (post-selection inference); Varma & Simon 2006 (nested CV); Steyerberg,
*Clinical Prediction Models* (2019); Van Calster et al. 2019 (calibration); Tomova et al. 2022
(all-components); Willett, *Nutritional Epidemiology* (energy adjustment); Moons 2006 / Sterne 2009
/ Sisk 2023 (missing data by purpose); Dunn et al. 2011 (QC-RLSC); Riley et al. 2020 (sample size
for prediction models).

## 7 · Owed before building

1. An adversarial literature review of every row of §1, §2 and §4 (customary? sound? the leash
   right?).
2. Fixture journeys for one prediction and one inference analysis per lens.
3. A decision on how "Quick" vs "Advanced" verbosity (PRODUCT_VISION §08) changes which of these
   steps are stated rather than asked.
