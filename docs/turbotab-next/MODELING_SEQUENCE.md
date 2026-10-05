# The modeling sequence — everything after the seal

**Status: DRAFT 2, reviewed.** Written by the orchestrator as the project's methods expert (Nolan,
2026-10-02: *"I am asking you to be one"*). Draft 1 was reviewed adversarially against primary
sources on 2026-10-03 by four reviewers, one each for inference, prediction, domain methods and end-to-end chains. There were 69 findings: 3 critical, 47 major, 19 minor or holds. The full record,
with quoted sources, is `docs/turbotab-next/audit/modeling-sequence-review.json`. §0 gives the
rulings. Where a work-package prompt written before this revision disagrees with it (for example,
WP17's five free-hand covariate roles), **this revision governs**.

**What this covers.** Everything from the drawn seal to a declared, calibrated, compared result. In
Classic this was EDA → feature engineering → feature selection → model-specific preprocessing →
train and compare. It is the densest concentration of intelligence in the app.

**What governs it:**
- BLUEPRINT §12, Nolan's rulings:
  - residual keeps energy;
  - all-components comes first for substitution under inference;
  - the seal is purpose-scoped;
  - imputation follows the purpose.
- North star 5: customary and sound are separate labels.
- §11.3: the leash.
- §13: the method contract and its relations.
- §14: the readings ledger.
- Lockbox constitution §06: the data-scope test.

**The organizing fact.** Purpose splits this segment more than any other.
- **Prediction** may learn choices from the training data, as long as the choice is made *inside*
  the resampling. A choice made by hand from what the analyst saw is disclosed as outside it.
- **Inference** may not choose its model from the data it will report on. The exposure, the
  estimand, the adjustment set and the functional form are declared before estimates are seen.
  Anything the data chose is reported as such.

---

## 0 · Rulings on the review (2026-10-03)

The reviewers were told to refute. Most of what they found holds up against its sources, and I
accept it. These are the calls where the review offered a choice, or where the scope of v2 is
touched.

1. **Data-driven adjustment has four rungs, not one.** Draft 1's blanket refusal contradicted v2's
   own DoubleML/TMLE.
   - (a) Choosing which exposures or terms to *report* by significance is refused (selective
     inference).
   - (b) Stepwise or p-value selection of confounders is refused as the primary model, labeled
     *customary, unsound*.
   - (c) Change-in-estimate among covariates already judged pre-exposure causes is block-and-record.
     The label states that it cannot see mediators or colliders and is invalid for an OR or HR with
     a common outcome.
   - (d) Data-adaptive adjustment with valid post-selection inference (post-double-selection lasso;
     cross-fitted DML/TMLE) over a candidate set chosen from subject knowledge is allowed. It ranks
     high when the candidates are many relative to n.
2. **Exposure quintiles under inference rank lower; they are not blocked.** They estimate a coarser
   contrast, not a false number, and they are the field's primary-analysis convention. The spline
   ranks first and the quintile table is produced beside it by default. Block-and-record moves to
   two cases:
   - data-derived "optimal" cut points;
   - a continuous confounder cut into three or fewer groups (residual confounding).
3. **Explore under prediction records what the analyst saw.** Without a holdout, "training rows"
   means all rows, so a lever pulled by hand from an outcome view sits outside the resampling while
   the score is labeled optimism-corrected. Three changes, never a block:
   - outcome views are recorded as looked at;
   - each Explore lever is offered first as an in-fold rule the resampling repeats;
   - a lever applied by hand is disclosed as outside the corrected score.
4. **Prediction selects and declares on a strictly proper score.** AUC and the C-index are always
   reported, labeled *customary headline* (the engine compares on them today: a defect, MS6).
   Families are compared on repeated k-fold. The choice among families is corrected by BBC-CV. With
   no holdout, the result is the selection-corrected estimate.
5. **Multiple imputation must be compatible with the analysis model** (critical). Today the engine
   imputes on the raw scale from a linear model, then derives logs, residuals, splines and
   interactions in each copy. This "passive" imputation biases curvature and interactions toward
   the null, and its negative draws break the log residual. The required order is in §1.1 and the
   relations in §2. Passive imputation with a declared nonlinear term is block-and-record under
   inference.
6. **A population estimand under a survey design binds every family and every display.**
   - Survey-weighted Cox (Binder linearization), a weighted proportional-odds model and weighted
     substitution refits are built (MS4).
   - Where no design-based estimator exists, the result is block-and-record. The exit is the
     sample-only attestation.
7. **Regression calibration is a declared secondary analysis.**
   - It sits beside the uncorrected estimate, runs inside each imputed copy, and puts every
     outcome-model covariate in its calibration equation.
   - Its variance comes from a bootstrap over the whole chain, resampling PSUs within strata, or
     clusters.
   - **Multivariate calibration (Rosner 1990) is in v2.** Without it, RC conflicts with the
     all-components model that §12 ranks first, and two of our own rulings would collide.
8. **Scale reliability is ω, not α** (omega-total for a unidimensional scale; omega-hierarchical
   for a multidimensional one). α is shown only with its customary label.
   - Disattenuation by α or ω is **refused for formative indices** (diet-quality scores, FFQ-derived
     scores). It needs test–retest ICC or a calibration substudy.
   - For reflective scales, the correction is conditional on the covariates (done by regression
     calibration) and labeled as omitting transient error.
   - Latent-variable (SEM) correction goes to INBOX for v2.x.
9. **The effect measure is part of the estimand.**
   - The estimand question asks: difference or ratio; conditional or marginal.
   - ORs and HRs are labeled conditional and non-collapsible.
   - Marginal standardization (g-computation) to an RD or RR is offered, ranked first for a common
     binary outcome.
10. **Unmeasured-confounding sensitivity is in v2,** under the causal row's "diagnostics". It is
    offered for every inference result and required in the causal lane:
    - the E-value for ratio measures, never with a pass/fail threshold;
    - the Cinelli–Hazlett robustness value for linear outcomes, benchmarked against a named
      measured confounder, ranked first.
11. **Share reallocation (compositional, ilr) goes to INBOX for v2.x.** In v2, a request to
    reallocate shares is refused with that reason, and the kcal substitution is offered instead.
12. **Clustered multiple imputation in v2** imputes time-invariant variables once per unit, at the
    unit level, and includes cluster means of the other variables when imputing time-varying ones.
    Full two-level FCS goes to INBOX. Single-level imputation on clustered rows is block-and-record
    under inference.

13. **Cross-validated scores under the surveyed population (2026-10-05, from REPAIR-SURVEY's open
    question).** Prediction then estimates performance *in the population*. Folds keep whole PSUs
    together within strata, and every loss, calibration and comparison is survey-weighted (Wieczorek,
    Guerin & McMahon 2022, *Stat* 11:e454, design-based K-fold CV). The record labels these
    "design-based cross-validation". Under the sample answer, scores stay unweighted, labeled as the
    procedure's performance on these rows. Under inference no cross-validated score is shown (§1 row
    11).
14. **The E-value of a difference is standardized by the SD the estimand speaks of.** Under the
    surveyed-population answer that is the design-weighted SD of the outcome; under the sample answer
    it is the sample SD (VanderWeele & Ding 2017's approximation RR ≈ exp(0.91·d)). The effects
    stage and the causal lane use the same SD.

Everything else in the review is accepted as written in §1–§4.

## 1 · The sequence (question order)

This is the order in which the Router asks. Domain transforms now come before the functional form:
knots, cut points and the unit of the estimate belong to the exposure's *final* scale.

| # | Step | Fires | Prediction | Inference |
|---|---|---|---|---|
| 1 | **Explore** | always | findings on training rows, each tied to a lever offered first as an in-fold rule; outcome views recorded as looked at; data-quality checks compared across sociodemographic groups (TRIPOD+AI 7) | the same; outcome-relationship views are recorded and disclosed (forking paths), never blocked |
| 2 | **Exposure and estimand** | inference | intended use (decision support or not), which gates the decision curve and threshold questions | *"Which exposure, and which effect?"*: total vs direct; substitution vs addition (the energy partition); the effect measure (difference or ratio; conditional or marginal); per what unit; **one exposure or an exposure family** (feature-wise, with its multiplicity method); for a food with many non-consumers, the whole population vs consumers only (an estimand change, STROBE-nut nut-14). No coefficient is shown before this is answered. |
| 3 | **Adjustment set** | inference | — | **the modified disjunctive cause criterion, asked as questions** per covariate: a cause of the exposure? of the outcome? could the exposure have changed it, or was it measured after? a known instrument? a proxy for an unmeasured common cause? The role is derived from the answers (confounder, precision, mediator, collider, instrument, proxy). Unknown timing (cross-sectional BMI) → a declared with-and-without pair. Other dietary components default to confounders through common dietary causes, never to "not relevant". A direct effect also asks for mediator–outcome confounders. **No data-driven selection, except rung (d) of ruling 1.** |
| 4 | **Domain transforms** (method contracts) | by lens and roles | energy adjustment (choice matters little, and the app says so); omics chain (§1.1); scale scores; count normalization; batch (outcome-free, in-fold) | the energy model with its estimand (all-components or leave-one-out first for substitution); log before residual; scale scores with ω; regression calibration *planned* here, run as a secondary analysis (ruling 7); batch as a covariate first |
| 5 | **Functional form** | a continuous exposure or confounder (inference); continuous predictors with nonlinearity evidence (prediction) | linear · restricted cubic spline · chosen by an in-fold rule or inner CV | declared before estimates, on the transformed scale: linear · RCS with **k by a declared rule** (4 by default; 3 at small n; 5 at large n), knots at Harrell's percentiles of observed values, fixed across imputations; the **overall test** is the test of association, and a non-significant nonlinearity test never silently refits a linear model. Quintiles: ranked lower, produced beside the spline, with boundaries and the reference stated, and the statistic labeled "p for linear trend (customary)". **Continuous confounders get a declared form too.** Mass at zero → non-consumers as their own category plus a form among consumers. |
| 6 | **Missing data** (asked before the seal, executed here) | missing values exist | outcome-free imputation fitted in-fold (deployable) | multiple imputation **compatible with the analysis model** (§1.1): the outcome, energy, design variables, cluster structure and every declared term; m ≥ max(20, % of rows with any imputed value), with Monte Carlo error reported; complete cases with its assumption stated |
| 7 | **Construction** | prediction, or declared modifiers under inference | domain features (contracts), in-fold PCA for omics, each with its scope | **effect modification** (the exposure's effect across strata of M; the exposure's adjustment set) and **interaction** (two exposures; step 3 re-run for the second) are separate declared objects, reported on both the additive scale (RERI) and the multiplicative scale; post hoc subgroups labeled "suggested by data inspection" and counted in the family |
| 8 | **Selection** | prediction | none · elastic net · stability selection (for a short, reproducible panel; ranked after elastic net) · in-fold screening or variance filters at p ≫ n · customary options (stepwise, univariable screens, VIP > 1) ranked lower, with instability stated · **selection outside the resampling refused**; asks whether any predictor was pre-selected on these rows' outcome (TRIPOD+AI 9a); inclusion frequencies reported | not offered for the reported model (ruling 1); as a labeled sensitivity analysis under MI, selection uses Rubin's-rules Wald tests across all copies |
| 9 | **The shelf** | always | **Riley minimum n first**; ordered by soundness at this size; the no-predictor baseline and a regression-with-splines benchmark always fitted; tree ensembles move up only at large n with nonlinearity evidence; each family's inductive bias stated | ordered by the declared estimand: linear, logistic, Cox, ordinal, mixed models/GEE for repeated units, design-based for surveys; flexible learners as nuisance models (rung d) or for sensitivity |
| 10 | **Model-specific preprocessing** | not a question | stated per family in the lineage; native missing-value handling either replaces or follows the in-fold imputation (the lineage says which); imbalance corrections in-fold, followed by recalibration | the same, plus the scale the coefficients are reported on |
| 11 | **Tuning and comparison** | stated | tuning nested in every outer fold and every bootstrap resample; families compared on **repeated k-fold** (≥ 10 × K), by paired differences with the corrected t; choice among families corrected by **BBC-CV**; strictly proper primary score, AUC/C reported; calibration for every task (time-to-event at a declared horizon; ordinal and multiclass by level; else "not assessed"); intervals worded "expected performance of this procedure at this n"; at p ≫ n, the nested-CV interval (Bates et al.) or the label "likely too narrow"; bootstrap ≥ 500; subgroup performance with CIs, decision curve if decision support, shrinkage offered as model updating, internal–external CV when clusters exist | **not a comparison.** The declared object is a crude model, a declared adjustment sequence (Model 1: age, sex and energy; Model 2: plus confounders; optional Model 3: plus possible mediators, labeled) and the primary model, shown **for the exposure only** (the Table 2 fallacy: adjustment terms are moved to an appendix titled "adjustment terms, not effect estimates"). Diagnostics are reported (proportional hazards by Schoenfeld; residuals and influence). Declared secondary analyses go side by side. Sensitivity to unmeasured confounding (ruling 10). |
| 12 | **Declaration** | before opening the seal (prediction); before showing estimates (inference) | (a) holdout drawn: the family is declared, any threshold or recalibration is locked, then the holdout is opened once; the record says which fit is reported and deployed. (b) No holdout: the selection-corrected estimate; only a family declared before any score was seen may report its own corrected score. | **the analysis-plan lock** covers the exposure or family, the estimand and effect measure, the adjustment set and roles, the forms, the energy model, the missing-data method, the secondaries, modifiers and subgroups, and the multiplicity policy. A change after estimates are seen is recorded as "after the estimates were seen". The lock is **never described as "prespecified" or "preregistered"**: it says what was declared in the software before estimates were displayed. The plan exports with a timestamp and a content hash for external registration. |

### 1.1 · Execution order (what the engine runs)

The question order is not the run order. The run order:

- **Before the seal, reference rows only:**
  - QC drift and inter-batch correction (QC-RLSC);
  - QC-RSD and QC detection-rate filters;
  - the QC rows leave;
  - PQN against a pooled-QC reference (that variant only).
- **Prediction, inside each training fold:**
  1. outcome-free imputation;
  2. PQN with a study-sample reference, and count normalization;
  3. detection-limit handling, censoring-aware:
     - half-minimum is customary at ≤ 10% censored;
     - above 10%, QRILC or a censored-normal draw;
  4. log or glog;
  5. batch adjustment, outcome-free (reference batch or add-on);
  6. scaling;
  7. construction;
  8. filters and selection;
  9. tuning;
  10. the fit.

  A zero that a log would turn missing is sent to the detection-limit question, never to median
  fill.
- **Inference, for each imputed copy m = 1…M:**
  1. imputation, compatible with the analysis model:
     - logged quantities are imputed on the log scale;
     - energy sources and "other" are imputed, and E is derived as their sum;
     - splines, interactions and logistic or Cox outcomes use SMC-FCS;
     - just-another-variable imputation only for a linear outcome;
     - design variables and cluster structure are in the model;
     - detection-limit blanks use the censored model, never MAR;
  2. derived terms are computed in the copy: the energy residual is re-estimated per copy;
     densities; scale scores from imputed items;
  3. regression calibration, when declared (inside the copy; outcome never in the calibration
     model);
  4. the spline basis, using the fixed knots;
  5. the fit.

  Then pooling: Rubin's rules for scalars; D1 for multi-df tests (overall, nonlinearity,
  interaction, global quintile); ν_com = the design degrees of freedom under a survey; every
  displayed estimate is pooled, including curves and contrasts. Calibrated estimates take their
  intervals from a bootstrap over the whole chain.

## 2 · Relations: how one action leads to another (§13)

**Kept from draft 1:**
- **Exposure declared**:
  - *enables* the effect display and the estimand sentence;
  - *invalidates* the adjustment-set answers given for another exposure.
- **Adjustment set** *conflicts with* data-driven selection under inference, except rung (d).
- **Energy model**:
  - *constrains* substitution: an omitted energy source is block-and-record under inference;
  - all-components and leave-one-out *enable* substitution of any pair.
- **QC drift correction** *implies* that the QC rows leave, and *precedes* every in-fold
  normalization.
- **Omics raw values** *conflict with* linear families until a normalization is chosen or the
  values are declared normalized.

**Corrected:**
- **A log or spline on an energy component** *conflicts with* reading a substitution as a
  coefficient difference. The curve still moves k kcal, but the effect now depends on k and on each
  person's intake. The curve is labeled as an average over the stated population at the stated k.
  The all-components contrast is undefined on logged components. *(Draft 1 said "ratio rather than
  difference"; that was wrong.)*
- **A spline or categories on a residual** *breaks* the identity with the standard model: it is the
  nutrient's curve on the energy-adjusted scale at mean energy, not the substitution curve. Its
  label changes, and spline(N) + E is offered as the route to the substitution curve.
- **Multiple imputation** *implies* pooling of **every** estimate shown under inference, including
  substitution curves (per copy, pooled per k) and form tests (D1). A single-fill curve under
  inference is a defect (MS3).
- **Repeated units or clusters**:
  - *imply*, under inference: cluster-aware intervals or mixed models/GEE at the highest correlated
    level, and a small-sample sandwich (CR2, Bell–McCaffrey df) with refusal below a floor;
  - *imply*, under prediction: grouped folds and a grouped holdout, a bootstrap by unit, and
    performance heterogeneity across sites.

**New:**
- **A domain transform of the exposure** (log, energy model, calibration, scale scoring)
  *invalidates* the functional-form answer, the knots, the cut points and the estimand's unit. They
  are re-asked, never silently kept.
- **A declared nonlinear or derived term under MI** *implies* an imputation model compatible with
  the analysis model (§1.1). A change of form or interactions *invalidates* the imputations.
- **Survey design with a population estimand**:
  - *implies* design variables in the imputation model, and ν_com = the design df;
  - *implies* a design-based estimator for every family and display, or block-and-record with the
    sample-only exit.
- **Clusters under MI** *imply* clustered imputation (ruling 12).
- **Regression calibration**:
  - *implies* every outcome-model covariate in the calibration model;
  - *implies* multivariate calibration when several intakes are error-prone;
  - *implies* a whole-chain bootstrap variance;
  - a change to the adjustment set *invalidates* it;
  - for residual or density models, energy is adjusted per recall day first, then calibrated.
- **Detection-limit readings** *imply* censoring-aware handling; MAR imputation of those blanks is
  refused under inference.
- **Zeros in a food exposure**:
  - *conflict with* log and log-residual;
  - exits: a two-part exposure, or a consumers-only domain (an estimand change, asked at step 2);
  - usual intake for an episodically consumed food goes to the NCI two-part model.
- **Batch**:
  - perfectly confounded with the outcome → refused under both purposes;
  - correction that protects the outcome (ComBat with the outcome) → refused for testing under
    inference (figures only) and refused under prediction (leakage);
  - correction *precedes* in-fold screening.
- **An exposure family** *implies* multiplicity control: BH q-values for feature-wise analyses; for
  a few prespecified nutrient hypotheses, the number of tests stated. Every member is shown. This
  is not selection.
- **The effect measure**:
  - an OR or HR with a common outcome *conflicts with* change-in-estimate;
  - adding a precision covariate under an OR or HR *changes* the conditional estimand, and the app
    says so.
- **A direct effect** *implies* asking for mediator–outcome confounders. Exposure–mediator
  interaction routes to counterfactual mediation methods.

## 3 · What the canvas shows at each step (pedagogy, §11.2)

Every view below comes from the existing closed vocabulary; nothing needs a bespoke picture.

| Step | Primary view | The question it answers |
|---|---|---|
| Explore | distribution, relationship, table focus | is my data okay; what will matter |
| Exposure and estimand | lineage (exposure highlighted) and a one-line estimand | what is this choice |
| Adjustment set | lineage with derived roles (confounder, mediator, collider lanes) | what will it change |
| Domain transforms | the method's own storyboard (energy: fit the line → residuals → add back the mean) | what will it change in my data |
| Functional form | relationship with the declared form drawn on the transformed scale (linear vs spline) | what will it change in my model |
| Selection | lineage: the columns that leave, per fold, with inclusion frequencies | what will it change in my model |
| Shelf / comparison | metric comparison against the baseline and the spline benchmark; what the interpretable model costs or gains | why it matters for my result |
| Declaration | the record: the locked plan, one sentence per decision | what did I decide; can a reviewer reproduce it |

## 4 · The leash, by step

| Step | Prediction | Inference |
|---|---|---|
| Outcome views in Explore | record; offer in-fold rules; disclose hand-chosen levers; never block | record and disclose; never block |
| Exposure and estimand | intended use asked | **required**: no estimates until answered |
| Data-driven adjustment | — | (a) refuse · (b) refuse as primary · (c) block and record · (d) allow (ruling 1) |
| Mediators in a total-effect set | — | **block and record**; "further adjusted for BMI" offered as a labeled secondary |
| Selection outside the resampling | **refuse** (a false performance number) | — |
| Exposure quintiles as the analysis model | rank lower, state the information loss | rank lower; spline first; quintile table beside it |
| Optimal cut points; confounders cut into ≤ 3 groups | rank lower | **block and record** |
| Passive MI with a declared nonlinear term | — | **block and record**; the form tests are biased toward the null |
| Single-level MI on clustered rows | — | **block and record** |
| MAR imputation of detection-limit blanks | rank lower | **refuse**, censoring-aware instead |
| Population estimand without a design-based estimator | — | **block and record**; exit: the sample-only attestation |
| Disattenuation by α/ω of a formative index | — | **refuse**; test–retest ICC or a calibration substudy |
| Outcome-protected batch correction | **refuse** (leakage) | **refuse** for testing; figures only |
| Omitted energy sources in substitution | state the concern | **block and record** above a stated omitted share |
| Single split at small n | rank lower than resampling, state why | not applicable (no holdout) |
| Median fill | allowed in-fold | **block and record** for the inference table |
| Winner's own corrected score as "the result" (no holdout) | **refuse**; report the selection-corrected estimate | not applicable |

## 5 · What exists, what is wrong in it, and what is new

**Engine defects the review found in shipped code.** These are methods-layer work packages, run
with the completeness pass (V2 definition of done §6) because they overlap with it:
- **MS1 · MI compatibility.**
  - Logs are imputed on the log scale; energy sources are imputed and E is derived.
  - SMC-FCS for splines, interactions and logistic or Cox outcomes.
  - Knots are fixed across copies.
  - A NaN after a log in a copy never reaches the median imputer.
- **MS2 · The MI frame.**
  - Design variables and ν_com = the design df.
  - Clustered imputation (ruling 12).
  - The m rule and Monte Carlo error.
  - D1 pooling for multi-df tests.
- **MS3 · Pool everything under MI.** Substitution curves (an exact pooled linear contrast for
  all-components; per-copy curves pooled per k for nonlinear models), margins and form tests.
- **MS4 · Survey design across families.** Survey-weighted Cox (Binder), weighted proportional
  odds, weighted substitution refits; block-and-record where no design-based estimator exists.
- **MS5 · Regression calibration.** A secondary analysis; inside MI; covariates in the calibration
  model; multivariate (Rosner); a whole-chain bootstrap by PSU within strata, or by cluster.
- **MS6 · Prediction validation.**
  - A proper-score primary for comparison and declaration, with AUC/C reported.
  - Repeated k-fold as the comparison substrate; BBC-CV.
  - The no-holdout declaration.
  - Bootstrap ≥ 500, with a compute estimate.
  - Calibration for time-to-event, ordinal and multiclass tasks, or "not assessed".
  - Grouped folds everywhere for repeated units.
- **MS7 · Omics order.**
  - Detection-limit handling before the log; zeros to the detection-limit question.
  - The QC-reference PQN variant; QC filters.
  - Batch relations, and outcome-free in-fold batch adjustment.
- **MS8 · Scales.**
  - ω-total or ω-hierarchical, with α labeled.
  - Disattenuation refused for formative indices.
  - Conditional correction; item-level MI before scoring.

**Built and verified by the audit loops:** per-family preprocessing, the shelf, nested tuning,
calibration and optimism (WP9), the declared final model (WP8), splines, quintiles and ordinal
(WP12a), Cox and mixed models (WP12b), MI by purpose (WP7), energy estimands (WP6).

**New from this spec (the M3.5 build):**
- Explore after the seal, with forking-paths disclosure and in-fold levers.
- The exposure and estimand question, including the effect measure, exposure families and
  consumers-only domains.
- The adjustment set asked through the disjunctive cause criterion (WP17 begins it).
- The functional form as a declared question, on the transformed scale, with confounders' forms.
- Selection as a purpose-routed question, with the customary options labeled.
- Effect modification and interaction as declared objects.
- The model sequence and the Table 2 display.
- Diagnostics.
- g-computation standardization.
- E-values and robustness values.
- Secondary analyses as declared objects; the analysis-plan lock and its export.
- The intended-use question, the decision curve, subgroup performance and shrinkage.
- What the interpretable model costs or gains (renamed from "the price of explainability": the
  signed paired difference between the best interpretable family and the best flexible family on
  the proper score and on calibration, BBC-corrected).
- The relations in §2 as contracts, with a chain test per reference chain (§6).

## 6 · Reference chains (the chain test)

Each chain runs end to end in the acceptance harness. It asserts the order in §1.1, asserts that
every relation in §2 fires, and asserts the methods sentence the reviewers wrote for it:
1. **NHANES with linked mortality.** Log-residual fiber with energy kept, an RCS exposure, survey-weighted
   Cox, MI with the event indicator, Nelson–Aalen, energy and the design variables, pooled on the
   design df.
2. **Repeated 24-h recalls.** Multivariate RC of all energy sources, all-components substitution, a
   bootstrap by PSU within strata that repeats calibration, imputation and the outcome model.
3. **Metabolomics prediction.** QC-RLSC on pooled QCs, then in-fold PQN, detection-limit
   imputation, log, autoscaling; an elastic net in nested CV; grouped folds for repeat samples.
4. **A survey scale.**
   - A reflective scale: ω, conditional correction by RC, both estimates reported.
   - A formative diet score: disattenuation refused, with test–retest ICC asked instead.
5. **Genomics, p ≫ n.** Batch as a covariate, in-fold screening, penalized logistic.
   - The inference twin: feature-wise models with BH.
6. **A multi-site cohort with repeated visits.** Clustered MI, GEE with CR2 and few-site
   Bell–McCaffrey df; grouped folds and site heterogeneity under prediction.
7. **An episodically consumed food.** Zeros → a two-part exposure, or a consumers-only domain stated
   for nut-14.

## 7 · Sources

The full quotes are in the review record. The core sources:
- **Confounder selection and the Table 2 fallacy:** VanderWeele 2019 (the disjunctive cause
  criterion; change-in-estimate and non-collapsibility); Westreich & Greenland 2013 (the Table 2
  fallacy); Daniel, Zhang & Farewell 2021 (conditional ≠ adjusted).
- **Interaction, sensitivity and forking paths:** Knol & VanderWeele 2012 and VanderWeele & Knol
  2014 (interaction reporting); VanderWeele & Ding 2017 (E-value); Cinelli & Hazlett 2020
  (robustness value); Gelman & Loken 2013 (forking paths).
- **Splines, categorization and validation:** Harrell, *Regression Modeling Strategies* (knots;
  validation repeating all steps involving Y); Grambsch & O'Brien 1991 (pretesting for
  nonlinearity); Altman & Royston 2006 (dichotomizing).
- **Missing data:** Bartlett, Seaman, White & Carpenter 2015 (SMC-FCS); Seaman, Bartlett & White
  2012 (JAV); von Hippel 2009; White, Royston & Wood 2011 (m and Monte Carlo error); Moons 2006 /
  Sterne 2009 / Sisk 2023 (missing data by purpose).
- **Model validation and selection:** Ambroise & McLachlan 2002; Varma & Simon 2006; Tsamardinos et
  al. 2018 (BBC-CV); Bates, Hastie & Tibshirani 2023 (what CV estimates); Nadeau & Bengio 2003
  (corrected t); Moscovich & Rosset 2022 (preprocessing bias); Meinshausen & Bühlmann 2010.
- **Prediction modeling:** Steyerberg, *Clinical Prediction Models* (2019); Van Calster et al. 2019
  (calibration); Riley et al. 2020 (sample size); Collins et al. 2024 (TRIPOD+AI; bootstrap ≥ 500);
  Rudin 2019.
- **Energy adjustment and measurement error:** Willett, *Nutritional Epidemiology*; Tomova et al.
  2022 (estimands of energy models); Rosner, Spiegelman & Willett 1990 (multivariate RC); Freedman
  et al. 2011 (RC with energy-adjusted intakes).
- **Omics and batch:** Dunn et al. 2011 (QC-RLSC); Nygaard, Rødland & Hovig 2016 (ComBat with
  unbalanced groups).
- **Reliability:** McNeish 2018 (α vs ω).
- **Reporting:** STROBE and STROBE-nut (Lachat et al. 2016).

Two caveats from the reviewers:
- The VanderWeele 2019 and Knol & VanderWeele 2012 quotes came from page summaries.
- Several later sources were read as abstracts after the search budget ran out.

The human expert review packet (V2 definition of done §4) must check those against the full texts.

## 8 · Owed before building

1. ~~An adversarial literature review~~ — done (2026-10-03).
2. Fixture journeys for one prediction and one inference analysis per lens, plus the seven reference
   chains in §6.
3. A decision on how "Quick" vs "Advanced" verbosity (PRODUCT_VISION §08) changes which steps are
   stated rather than asked. My proposal: Quick *states* steps 8–10 and the default rungs, and
   *asks* steps 2, 3 and 5 under inference, because those are the ones a reviewer will question.
