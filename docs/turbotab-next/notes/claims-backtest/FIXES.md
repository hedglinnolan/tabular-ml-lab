# Claims backtest: the fix list

Each entry is a claim the app relies on that the adjudicator found wrong, overstated, unbuilt, or unlabeled. The adjudicator had the repo, both blind referees' verdicts, and the basis audit. One adjudicator ruled on each claim; no second adjudicator checked it. So every S1 fix starts by writing a known-truth or independent-reference test that fails on today's code. If no such test can be made to fail, the finding is not confirmed. Record it as a retraction here, not as a fix.

Severity:

- **S1:** a realistic case gets a misleading number, interval, recommendation, refusal or lesson.
- **S2:** the behavior is defensible in the main regime, but the stated reason or scope is wrong or overstated.
- **S3:** cosmetic, or rarely reached.

Strata: *random* is the seeded sample, *tier2* is explanatory text, and *targeted* is the inventory agents' most-suspect picks (see README.md).


## S1 (15)

### A-001 · defect · targeted

**Claim.** For boosted trees (scikit-learn's HistGradientBoosting) and XGBoost under prediction, when the effective size of one outer training fold (n_plan: units for a numeric outcome; units in the rarest class for a yes/no or class outcome) is below 300, fitting once at the library's default settings with no search (HistGB: learning rate 0.1, 31 leaves, smallest leaf 20, no L2, 100 trees; XGBoost: learning rate 0.3, depth 6, child weight 1, λ 1, 100 rounds) predicts at least as well as nested tuning and is better calibrated, because settings ranked on so few units vary from sample to sample and miscalibrate.

**Where.** `turbotab/core/models/tuning.py:92`

**Fix.** Implement the 2026-10-11 small-sample standard (RECIPES §4.6) as the below-300 standard candidate in turbotab/core/models/boosted_trees.py:71 (100 trees, <=8 leaves, lr 0.02, leaf 20, L2 1) and turbotab/core/models/xgboost_family.py:502 (depth 3, lr 0.02, child weight 20 rows-equivalent, lambda 1), update the methods sentence at boosted_trees.py:289, and replace the bit-for-bit pin in test_boosted_trees_tuned.py::test_below_300_the_tuned_fit_is_a_direct_default_fit with a held-out check against nested tuning.

**Evidence.** tuning.py:92 (SMALL_N=300) sets S=0 so boosted trees and XGBoost are delivered at the library defaults below n_plan 300; both referees' simulations (calibration slopes 0.2-0.5, log loss worse in every data set) and the repo's own SMALL_SAMPLE_STANDARD.md §2 (defaults the worst arm in all 14 cells, regret 0.10-0.39 nats, slope ~0.3) refute 'at least as well and better calibrated'. I accept the audit: the cited Riley/Van Calster/Martin papers concern penalized regression, and the ruled fix is not built, so every small-table boosted fit is overconfident and loses model comparisons. (confidence 0.95)

### A-029 · defect · targeted

**Claim.** A gain of 0.01 on the primary metric's own scale (baseline.MIN_GAIN) is the smallest gain that matters for every task, including the mean squared error of a numeric outcome in its squared outcome units, log loss in nats, the ranked probability score and the Brier score at a horizon, so a gain below 0.01 is 'too small to matter whatever the folds say'.

**Where.** `turbotab/core/models/baseline.py:40`

**Fix.** Make the 0.01 floor in turbotab/core/models/baseline.py:96-97 relative for the loss primaries other than regression's R². For example, require a gain of at least 0.01 x the baseline's Brier at the horizon, log loss or RPS, which is a scaled-Brier or McFadden gain of 0.01, as the R² verdict already does for regression. Correct the docstring at baseline.py:25-27, and stop explain.floor_of (explain.py:1199) from saying 'is not shown to beat' when the interval clears zero.

**Evidence.** The inventory's MSE premise is wrong. Regression's verdict is read on R² (stages/modeling.py:1989-1991, stages/evaluation.py:270), so the referees' mg/dL objection never reaches the app. Their rare-event point does hold for brier_t, the time-to-event primary, which the app reads unscaled. In my simulation, a model with AUC 0.73 at a 10% horizon risk gains about 0.007 in Brier (0.002 at 5% risk with AUC 0.79). The app then calls that gain 'too little to matter whatever the folds say', and floor_of refuses its explanation curves as 'noise' even when the 95% interval clears zero. Binary log loss bites the same way below about 3% prevalence. (confidence 0.75)

### A-102 · defect · random

**Claim.** For the proportional-odds model, Wald intervals from the inverse observed information are valid for independent rows, and when rows repeat within units the CR1 sandwich over units' summed scores with a G/(G−1) correction and a t(G−1) reference gives valid 95% intervals (no interval below the unit floor).

**Where.** `turbotab/core/models/ordinal.py:362`

**Fix.** Stop prepending the linear family's inference._cluster_concerns (inference.py:638-649) at ordinal.py:377; give the ordinal table its own lines that say CR1 with G/(G-1) on t(G-1), Cameron & Miller's stated minimum that still over-rejects with few or unequal units, and below FEW_CLUSTERS (30) either flag the intervals as possibly too narrow or add a few-cluster method (a CR2-type correction or wild cluster score bootstrap), with a coverage test at G = 8-30 with unequal units.

**Evidence.** Running ordinal_table with G = 12 shows the caption 'sandwich ... G/(G-1) correction ... on t(11)' beside concern lines saying 'cluster-robust (CR2, G = 12)' and 'the intervals use the CR2 small-sample correction with Bell-McCaffrey degrees of freedom, which Cameron & Miller recommend when clusters are few', so for every ordinal table with 8-29 units the user is told a method and df that were not used, the very correction the source recommends for this case. Both referees agree, and their simulations concur, that CR1 + t(G-1) undercovers (about 80-88%) with unequal units or exposure in few units; I accept that, and the audit's reading of Cameron & Miller ('at a minimum'). The independent-rows Wald part is fine. (confidence 0.85)

### A-124 · defect · random

**Claim.** Before the lock under Estimate, the share of rows an exclusion rule (an energy-intake screen) removes, multiplied by the largest standardized mean difference (Austin 2009; pooled-SD for numbers, per-level proportion difference for categories) between the removed and kept rows on the adjustment set, is an outcome-blind predictor of how far keeping every row would move the exposure's estimate in units of its 95% interval half-width, good enough to band the noticing as doesn't change / could bias / act on it.

**Where.** `turbotab/core/materiality.py:286`

**Fix.** In turbotab/core/materiality.py, keep rows_smd from ever predicting band 0 for an energy screen: set calibrated=false in materiality_calibration.json, since 9 in-sample cases from one fixture are too few. Reword recommend()'s band-0 text at materiality.py:1035-1037 ('below what would move the estimate'), and the band words, so they call the product an outcome-blind screening convention rather than a predictor of the movement. Make the every-row analysis the default verifier for any energy screen.

**Evidence.** Both referees show that rows x max SMD on the adjustment set does not track the coefficient's movement. Adjusted-for covariate imbalance does not move a correctly specified coefficient; misreported exposure among the leavers does; and the proxy has no n. Referee A's simulation ranks the cases backwards. The repo's own calibration confirms this: the pairs are non-monotone, half the band-1 predictions (4/8) were realized at band 2 ('act on it'), and band_1 was refit in-sample to 0.0037. Yet the entry is marked calibrated:true. A narrow screen below 0.0037 is told 'That is below what would move the estimate', and the repo's own test documents a narrow screen at 0.019 that moved 0.63 half-widths. The 'checked again after the plan is fixed' promise holds only if an every-row analysis was declared, since realized_from_sensitivity returns None otherwise. That makes this a recommendation change in the reassuring direction. (confidence 0.65)

### B-211 · defect · random

**Claim.** When the study-design question is unanswered, analyzing the table as an observational study (people observed as they were, not randomized, not sampled by the outcome, not matched) is a safe default under every goal, provided it is stated and listed for confirmation.

**Where.** `turbotab/core/designs.py:90`

**Fix.** When the declared design is case_control or matched_sets (turbotab/core/designs.py:64-74, 129-137), the 'Analyze it as an observational study' exit must keep case_control.py's refusals of prevalence, absolute risk and risk difference, and the Record must name the real design instead of voice.py:456-458's 'not ... sampled by their outcome'. Route prevalence_signal and matched_sets_signal to ask the design question instead of leaving them as noticings only.

**Evidence.** The app never says 'safe', but it treats the observational lane as a safe fallback in two places. It is the unanswered default, and it is the first exit offered to a declared case-control or matched sample, whose refusal says the data 'can be analyzed as an observational study, and the methods say so'. Taking that exit records the study as 'not ... sampled by their outcome' (voice.py:456-458). It also runs a lane that ranks a marginal risk difference first for a common-looking outcome (explore.py:591) and reports what case_control.py says is refused 'under every goal'. No methods text records the original design. I accept the referees' Prentice-Pyke point and the audit's reading. (confidence 0.6)

### B-281 · defect · random

**Claim.** When some group of trial participants all had the event or none did (logistic separation), the model's risks run to 0% or 100% and no finite estimate exists, so the standardized risk difference and ratio are refused.

**Where.** `turbotab/core/methods/trials.py:746`

**Fix.** In turbotab/core/methods/trials.py:715-751, stop refusing the standardized risk difference under separation: compute the arm risks from the fitted limit (or a Firth-penalized fit, or with the separated stratum collapsed), report the RD always and the RR when both arms' standardized risks are positive, use an interval that survives separation (Firth/profile or a bootstrap), and change the message to 'the logistic model's coefficients have no finite values' instead of 'no finite estimate exists'; distinguish the |beta| > 25 / risk < 1e-10 triggers from true separation.

**Evidence.** Both referees and the audit agree, and Referee A's simulation shows it: under separation only the coefficients diverge, while the g-computed arm risks are averages of bounded fitted risks with finite limits, so the RD (and usually the RR) exists. The app refuses the trial's pre-specified adjusted effect for that false reason, and its only exit drops every covariate (covariates: []); with zero events in one arm (common for rare outcomes) even the unadjusted fit separates and no RD is given at all, although it is finite (e.g. 0/50 vs 20/50 = 0.40). (confidence 0.75)

### B-416 · defect · random

**Claim.** For a numeric exposure, a cross-fitted R² of 0.9 or more of the exposure on the covariates (a variance inflation factor of 10) is a practical positivity violation: the effect rests on little of the exposure's own variation and extrapolates where it does not vary, so the estimate is refused until every row is kept on the record.

**Where.** `turbotab/core/models/causal.py:1163`

**Fix.** Stop withholding at stages/causal.py:740-746 for a numeric exposure when variation().violated (models/causal.py:1159-1183); report the cross-fitted R² as a precision note (the residual SD and the sqrt(VIF) widening, already in the interval), remove 'Positivity is practically violated ... extrapolates where x barely varies' from the methods sentence (core/causal.py:791-797) and 'revisit the adjustment set' from _variation_reason (stages/causal.py:584-588), and if a continuous-exposure positivity check is wanted, base it on support of the residual at the contrasted values, with a known-truth test.

**Evidence.** My simulation (DML-PLR, OLS nuisances that are correctly specified, n = 300, 500 reps) gave cross-fitted R² 0.95 with bias +0.005 and coverage 0.95, and R² 0.99 with bias +0.02 and coverage 0.94: high R² only widens the interval, as both referees say (Frisch-Waugh), yet the app returns a 409, withholds the estimate until the user records that 'positivity is practically violated', and writes that into the methods sentence. With random-forest nuisances (the default from 500 rows), the real hazard of amplified misfit bias showed (bias -0.09 to -0.12, coverage 0.53-0.67), but there the cross-fitted R² stayed at 0.86-0.88, below the 0.9 line, so the gate does not catch the problem it is said to catch. Its only source is Kutner's collinearity rule of thumb, which the audit confirms says nothing about positivity. (confidence 0.72)

### B-435 · defect · random

**Claim.** When the Newton fit drives a coefficient beyond 20 SDs of its column on the log-hazard scale (or does not converge), the partial likelihood has no finite maximum (monotone likelihood, 'in one of its groups every row has the event, or none does'), so the hazard ratio is infinite and no interval or p-value is reported.

**Where.** `turbotab/core/models/survival.py:472`

**Fix.** In cox_fit (turbotab/core/models/survival.py:270-296), detect monotone likelihood without relying on |beta_z|>20. For example, flag a coefficient whose standardized value exceeds about 10 while its information eigenvalue collapses, or check each column for a level with no events in its risk sets, and refuse when pinv returns a zero or non-finite SE. Reword the parenthetical at :475-476 and :577-578 to 'a predictor orders the risk sets perfectly (for example, a group with no events)'. Add a test with a zero-event group plus one other covariate, and consider the Firth exit (Heinze & Schemper 2001).

**Evidence.** Both referees are right that 'every row has the event' is not a Cox monotone-likelihood condition. In my run of the app's cox_fit, an all-event group interleaved with censored controls converges at beta*SD of -0.01. The worse problem is that the refusal misses the canonical monotone case. I ran cox_table with a binary group that has zero events plus one other covariate, 40 seeds per design at n=60-1000. It was refused in only 0-2 of 40 runs per design (1 at n=1000). Most runs instead reported a log HR of about -36 with SE about 1e-17, p = 0.0 and a zero-width 95% interval (26/40 at n=60). Others gave a Wald interval of plus or minus 1.5e7 with p=1.0, or no interval and no refusal. Newton's decrement falls below TOL=1e-18 at |beta_z| about 18, before the 20-SD trigger, and pinv zeroes the vanishing information. A user is shown a 'certain' infinite hazard ratio, with p = 0.0. (confidence 0.85)

### BX-040 · defect · tier2

**Claim.** The share of usual intake above the Tolerable Upper Intake Level (UL) is a share at risk of excess only if the recalls include supplements, since the UL applies to total intake.

**Where.** `turbotab/core/methods/usual_intake.py:1309`

**Fix.** In share_phrase (methods/usual_intake.py:1309-1311), recognize by name, as reads_as_iron does, the nutrients whose UL covers only supplemental or synthetic forms (magnesium from supplements and medicines, folic acid, niacin, supplemental alpha-tocopherol, preformed vitamin A), and say that for them only that form is compared with the UL. For the other nutrients, say that a foods-only share is a lower bound on the share above the UL.

**Evidence.** I accept both referees and the audit: 'since the UL applies to total intake' is false for magnesium, folate, niacin, vitamin E and preformed vitamin A (IOM DRI UL definitions). A user whose recalls include supplements for these nutrients is told the share above the UL is a share at risk of excess, which overstates it; food magnesium alone often exceeds the 350 mg supplemental UL. The app prints this sentence unconditionally, never checks the nutrient or whether supplements were included, and has no test. (confidence 0.65)

### C-123 · defect · random

**Claim.** A column holding a single value carries no variation any model can use, so it is settled as excluded without asking.

**Where.** `turbotab/core/readings.py:838`

**Fix.** In turbotab/core/readings.py:838-843, settle 'excluded' only when the column has no blanks (values.isna().sum() == 0), and require n_missing == 0 likewise at turbotab/core/stages/rows.py:591. Otherwise ask, or offer value-versus-blank as a binary. Add a 'yes'/blank fixture to test_discriminate.

**Evidence.** Both referees judged the strict claim true and both warned about blanks, and the code takes the blank case. constant_values drops blanks before counting (readings.py:425-428), and my probe returned settles=True with 'every row holds the same value' for ['yes', None, None, 'yes', None] and for [1.0, NaN, 1.0]. A checkbox-style column, such as a 1/blank multi-select export or 'yes'/blank supplement use, is then dropped without asking, for a false reason. That removes a usable predictor or confounder. (confidence 0.75)

### C-143 · defect · random

**Claim.** For NHANES data the correct survey weight is the weight of the smallest subsample on which all analysis variables were measured (the least common denominator): an OGTT analyte requires the OGTT subsample weight (WTSOG*), a fasting analyte (glucose, triglycerides, LDL, insulin, apoB) the fasting subsample weight (WTSAF*), dietary recall variables the day-1 dietary weight (WTDRD1*, or the two-day WTDR2D* when day-2 variables are present), and otherwise the MEC examination weight; a variable recorded only where a subsample weight is positive (at most 1% of its rows outside, missing on at least 5% of the table, at least 10 rows) is that subsample's variable whatever its name; that weight is offered first, then the weights of the samples that contain it.

**Where.** `turbotab/core/recognizers.py:1913`

**Fix.** Add the August 2021-August 2023 phlebotomy weight (WTPH2YR) to WEIGHT_TIERS between examination and fasting (recognizers.py:1798-1805) or to NHANES_WEIGHT_PREFIX (recognizers.py:1714), so blood analytes name it. In least_common_denominator (recognizers.py:1985-1990), offer WTINT first when no analysis variable is recorded only on examined rows. Drop DBQ/DBD from _DIETARY_VARIABLE (recognizers.py:1817).

**Evidence.** I accept referee A's decisive point, which the audit missed. With WTPH2YR in the table and total cholesterol recorded only on phlebotomy rows, I ran least_common_denominator: it returned WTMEC2YR with settled=True. WTPH2YR is read as a weight (read_design) but has no tier, and its prefix is unknown, so the values check never runs. The recommended weight therefore skips the phlebotomy nonresponse adjustment that CDC added for this cycle. The interview-only (MEC named instead of WTINT) and DBQ-as-dietary departures that the audit found only cost rows, not bias. (confidence 0.65)

### C-170 · defect · random

**Claim.** Under Predict, a highly collinear predictor pair always warrants a decision by the user (a penalty fit inside each fold), so the collinear line is a Decide regardless of materiality.

**Where.** `turbotab/core/materiality.py:952 collinear_label`

**Fix.** Under Predict, make materiality.collinear_label (materiality.py:957-958) return For the record (or Confirm, as calm/BEATS_PREDICT M4 has it) and change _collinear_lever (stages/explore.py:666-674) to recommend 'keep both; a penalized family shrinks on its own; importance read grouped', dropping 'the penalty spreads the weight' from the elastic-net option, since set_selection elastic_net is a selection filter put before every family (pipeline.py:783), not a penalty in the fitted model.

**Evidence.** Both referees are right that a |r| >= 0.9 pair does not harm in-distribution prediction, so a Decide 'regardless of materiality' has no basis, and the audit notes it contradicts the app's own Predict beats and its selection menu, which ranks 'none' soundest. The bite is worse than either referee found: the Decide's recommended answer is the elastic-net selector applied to every family. In my simulation (n = 500, a pair at r = 0.95, one U-shaped predictor) it kept both collinear predictors in 10 of 10 runs, so nothing was shared, left OLS unchanged (MSE 6.14 vs 6.14), and raised the random forest's test MSE from 2.26 to 4.17 by dropping the nonlinear predictor half the time. (confidence 0.72)

### C-193 · defect · random

**Claim.** Under the genomics or metabolomics lens, a block of at least 20 numeric non-outcome, non-identifier, non-acquisition columns is unnormalized (needs a normalization before modeling) if, for counts, at least 20 of them are non-negative whole numbers with more than two distinct values reaching a maximum of at least 100 (genomics lens), or, for intensities, they are non-negative and their positive values span at least 100-fold and either reach at least 1,000 or have median per-feature skewness of at least 0.5 (metabolomics lens); values failing these tests are treated as already normalized or log-scale.

**Where.** `turbotab/core/methods/omics.py:167`

**Fix.** In turbotab/core/methods/omics.py:189-194, when the genomics lens sees a non-negative block that is non-integer, has a maximum of at least RAW_MAX and has widely varying row sums, raise the omics_scale question as 'ask: estimated counts or FPKM/TPM?' per GENOMICS_PACK.md §02 row 2, rather than returning None. Add genomics_estimated_counts.csv to test_the_reading_names_raw_values_and_stays_quiet_on_normalized_ones with the expected 'ask'.

**Evidence.** I ran scale_reading under the genomics lens on the repo's own fixtures. genomics_estimated_counts.csv (non-integer, unnormalized, 'library sizes still vary widely', and marked 'ask' by its .md and by GENOMICS_PACK §02) returns None, the same as normalized CPM. Raw integer counts are read correctly. So salmon/kallisto/RSEM estimated counts, a common RNA-seq export, get no omics_scale finding, hence no normalization question and no linear-family refusal. That is the false negative both referees describe (non-integer counts fail the whole-number test), and it contradicts the repo's own pack. The cut-offs (100, 1,000, 0.5) are unsourced conventions, but the lognormal skew arithmetic checks out (Referees A and B). (confidence 0.75)

### D-058 · defect · random

**Claim.** Under inference, a site that differs in both the exposure and the outcome confounds the estimate and intervals that treat its participants as independent are too narrow, so each site gets its own intercept and cluster-robust intervals (few groups needing a small-sample correction), 'cluster only' leaves between-group differences in the estimate, and under prediction validation keeps a group together.

**Where.** `turbotab/core/teaching/content.py:1341-1344`

**Fix.** In estimand._clusters_name_a_grouping (estimand.py:1828-1840) and the fixed-effects wiring (models/pipeline.py:103-107, 402-406), stop offering 'Adjust and cluster' as one-hot intercepts for a binary (or time-to-event) outcome with small groups. Use the conditional logistic fit already in methods/case_control.py (or a stratified Cox), or refuse with the incidental-parameter reason, and scope content.py:1341-1344 and its 'fixed effects' term to linear models or large groups.

**Evidence.** Both referees' incidental-parameter objection holds against the code. The fixed-effects column enters every family as one indicator per group, and the binary inference family is unpenalized LogisticRegression (linear.py:171). No validator checks group size or task, and 'Adjust for X and cluster by it' is the first exit. My simulation of that exact estimator (true log-OR 1) gave 2.11 for pairs, 1.60 for groups of 3, 1.46 for 4 and 1.12 for 10. Household groupings with a yes/no outcome therefore get a badly inflated odds ratio and g-computed risk difference. The linear-outcome fixture in the audit and the test is correct. (confidence 0.8)

### D-086 · design doc only (not built) · random

**Claim.** For a least-squares estimate on an outcome with a clinical cut-off c, writing y = min(y, c) + max(y − c, 0) splits the coefficient exactly into a part from values above the cut and a part from values below, which add to the primary's coefficient, and the estimate is 'mostly the tail' when the part above is more than half and its interval excludes zero.

**Where.** `docs/turbotab-next/calm/BEATS_MODELS_RESULTS.md §6 (line 650)`

**Fix.** Before BE23 is built, change BEATS_MODELS_RESULTS.md §6 (about line 650) so the verdict compares the above-cut share with P(Y > c), the share a uniform shift would give ('mostly the tail' only when the share clearly exceeds it, interval-based). State which side of the cut-off is the tail, define the share on the primary's sign, and add the sum-to-primary test.

**Evidence.** Nothing implements it (no match in turbotab or the frontend). The additivity is exact, as Referee A and the audit say. But Referee B's objection is decisive: under a location shift E[(Y-c)+] moves by delta*P(Y>c). My simulation with a BMI-like outcome and a uniform effect labeled it 'mostly the tail' in 100% of runs at the cut-off 25 (64% above), and in 0% at 30 or 40. The rule as written would therefore misread uniform effects for common lens cut-offs where most people are above (BMI 25, LDL 100), and for low-side cut-offs. It is S1 if built; today it only needs a doc fix. (confidence 0.72)


## S2 (39)

### A-055 · convention shown as a finding · random

**Claim.** Under inference with repeated units at or above the unit floor, GEE with an exchangeable working correlation and CR2 (bias-reduced linearization) intervals on Bell–McCaffrey degrees of freedom gives near-nominal type I error even with about 12 units, and below the unit floor its intervals are refused with the mixed model as the exit.

**Where.** `turbotab/core/models/repeated.py:717`

**Fix.** Reword the shelf concern at turbotab/core/models/repeated.py:729-731 from 'fewer than the 8 a sandwich interval needs' to 'fewer than the 8 units TurboTab requires (a convention, the seal's unit floor)', matching floor_refusal at inference.py:263; add a binary-outcome GEE type-I test beside test_gee_type_one_error_at_twelve_units.

**Evidence.** I ran the app's gee_table under the null. At 12 units it rejects 0.048 for a numeric outcome. With a rare binary outcome (about 4%) and 3 of 12 units exposed it rejects 0.016-0.029, so it is conservative, not liberal. That rejects Referee B's 0.123 for this implementation and supports Referee A. With the floor bypassed at 6 units it rejects 0.064, so the floor of 8 is a defensible conservative convention with no methods source (both referees and the audit agree). floor_refusal labels the floor as 'TurboTab requires', but the shelf text states it as what 'a sandwich interval needs'. The exits are task-correct: mixed model for a number, combining rows for yes/no. Aside: gee_working raises an uncaught ValueError when the estimated exchangeable alpha is negative enough for large units, e.g. ICC near 0 with 30-40 rows per unit. (confidence 0.7)

### A-064 · defect · random

**Claim.** The signed paired difference between the interpretable benchmark (a properly specified regression with splines on the same in-fold preprocessing) and the best flexible family, with the flexible pick re-made in every bootstrap resample of the out-of-fold predictions (BBC-CV over the flexible set), estimates what interpretability costs or gains on the strictly proper primary and on calibration (distance of the calibration slope from 1) without the optimism of picking the best flexible family; when its 95% percentile interval spans 0 there is 'no measurable cost on these rows'.

**Where.** `turbotab/core/models/selection.py:1013`

**Fix.** In turbotab/core/models/selection.py:1000 and the verdict at ~1086, replace NO_COST with an absence-of-evidence wording (e.g. 'these rows cannot tell the two apart; the interval spans 0'), label the interval as BBC-CV's percentile interval of out-of-bag estimates (conservative, per Tsamardinos et al. 2018) rather than a plain '95% interval', and raise the calibration clause's floor above n >= 10 (e.g. require a minimum count of out-of-bag events per resample before reporting |1 - slope|).

**Evidence.** The procedure is BBC-CV as published (both referees and the audit agree; the pick is re-made per resample at selection.py:1048-1052), so the selection correction itself stands. But my simulation (3 equal flexible families, fixed OOF losses) gives the percentile interval ~99.5% coverage, and it spans 0 in 19% of n=100 cases with a real cost versus 5% for a calibrated paired 95% interval, confirming Referee A that 'no measurable cost' fires too often and tilts toward the interpretable model; Referee B's point that an interval spanning 0 is not evidence of no cost also holds. The shown interval keeps this a wording/regime fix rather than a wrong number, and the calibration slope on ~4 out-of-bag rows at the 10-row floor is rarely reached. (confidence 0.75)

### A-074 · defect · random

**Claim.** When a unit's rows repeat, feature-wise intervals that treat rows as independent would be too narrow, so the tests use CR2 with Bell-McCaffrey degrees of freedom, and above 200 exposures (CR2 computed per exposure) the intervals are refused rather than reported as independent.

**Where.** `turbotab/core/models/featurewise.py:209`

**Fix.** Replace the fixed CR2_MAX_FEATURES = 200 cap (featurewise.py:61, used at :209-215) with a cost budget on rows x exposures, or fall back to CR1 on t(G-1) when G is large, and make the caption at featurewise.py:261 say t(G-1) when the Bell-McCaffrey df were skipped under BM_BUDGET (featurewise.py:291).

**Evidence.** CR2 with Bell-McCaffrey df below the cap is correct and certified (test_3b against written-out matrices, plus the clubSandwich check), and the 'too narrow' direction holds for features that vary between units (referee B: 73.5% vs 95.6% coverage). I accept both referees that the refusal has no methods basis: timing _cr2_tests gave 1.3 ms per exposure at N=2,000, G=500 and 6.8 ms at N=10,000, G=2,000, so 5,000 exposures take about 7-34 s. A repeated-sample omics table loses every interval, p and q to a self-imposed limit. The text honestly says 'beyond what TurboTab computes' and offers a valid exit (one row per unit), so this is S2, not S1. (confidence 0.6)

### A-089 · defect · random

**Claim.** On the pre-fit shelf, ridge scores 2.0 from 2,000 training rows, 1.0 below 2,000 (under least squares' and the screened elastic net's 1.5), 3.0 when predictors are at least as many as rows (below the elastic net's 4.0), is fair at most 2.0 under inference, and below 50 rows carries the concern that inner CV picks the penalty from very few rows.

**Where.** `turbotab/core/models/ridge.py:208`

**Fix.** Re-derive ridge's below-2,000 score (ridge.py:72 SMALL_SCORE, ridge.py:25-28) on a methods reason, at least tying least squares' 1.5 under prediction, record it in RECIPES §2.6, and retire test_ridge_family.py::test_ridge_never_displaces_the_first_two_families_the_journeys_pick, since it pins a fixture constraint as if it were a methods rule.

**Evidence.** The stated reason for 1.0 is protecting the reference journeys' picks, which is not a methods reason, and RECIPES §2.6 does not list it. Running rank() and shelf_order() shows the referees' small-n inversion mostly does not happen: below Riley's minimum least squares drops to 'fair' 0.5, so ridge (1.0) already ranks above it, and the elastic net (2.5) leads in every case. Ridge falls below least squares only between Riley's minimum and 2,000 rows. My simulation there (n 600-1,600, p 10-30) gave a ridge/OLS test-MSE ratio of 0.995-0.999, so the ranking error is real but the order it produces is nearly immaterial. (confidence 0.7)

### A-106 · defect · random

**Claim.** For a yes/no outcome, resampling (SMOTE, over- or undersampling) or class weighting should not be applied by default, because it overestimates the rarer class's probability without raising the AUC; a decision should move the threshold instead.

**Where.** `turbotab/core/stages/modeling.py:1079 (imbalance_sentence), used at :2384`

**Fix.** At turbotab/core/stages/modeling.py:2384-2385, pass spec.levers to imbalance_sentence (modeling.py:1079) and, when levers['imbalance'] != 'none', return a sentence naming the correction and its recalibration (or None, leaving levers.decision_sentence to say it), with a test that runs a binary fit with imbalance='weights'.

**Evidence.** Both referees and the audit agree the default is sound (van den Goorbergh 2022; Referee B's weighting simulation extends it). The defect is in the code. imbalance_sentence is emitted unconditionally for every binary fit (modeling.py:2384), while pipeline.py:825-828 wraps the model in ImbalanceCorrected whenever the Explore lever (explore.py:596-612) sets weights, undersampling or oversampling. A run with the lever on therefore states 'No resampling ... or class weighting was applied' beside the lever's own sentence saying it was, and a user could copy that into a methods section. The default regime is right; only the scope of the sentence is wrong, so S2. (confidence 0.85)

### A-114 · convention shown as a finding · random

**Claim.** The regression-calibration interval is the percentile interval of the bootstrap resamples, 200 by default, and is shown only when at least 90% of the resamples could be calibrated; resamples whose day-to-day error swamped the between-person spread are dropped, which may pull the interval's end away from zero too near.

**Where.** `turbotab/core/decisions.py:293 and :622 (n_boot default 200); turbotab/core/stages/calibration.py:513 (MIN_BOOT_SHARE=0.9), :1008, :1078-1092`

**Fix.** Either raise the n_boot default for the percentile interval to 1,000 (decisions.py:293 and :622), or keep 200 and say beside MAX_BOOT (decisions.py:448) why the ms5 simulation shows it suffices here. Label MIN_BOOT_SHARE = 0.9 (stages/calibration.py:513) as a convention in the user text, and add a test of the 'fewer than 90% ... no interval is shown' branch.

**Evidence.** Both referees' simulations and the ms5 known-truth test agree that the shown interval covers near nominal and that the drop-direction caveat (calibration.py:1083-1088) is right, so referee B's 'not established' changes nothing decisive. But a percentile interval at B = 200 contradicts the app's own sourced rule: decisions.py:448 cites Carpenter & Bithell 2000, and modeling.py:43 switches to a normal band at 200 for that reason. At B = 200 the 2.5% endpoint is the 5th order statistic, with a Monte Carlo SD of about 0.19 bootstrap SDs (about 10% of the half-width). The 90% floor has no source. (confidence 0.7)

### A-117 · defect · random

**Claim.** Multiple imputation with the outcome in the imputation model is invalid under prediction, because a held-out row's own outcome would fill its predictors and a new row has none; under prediction blanks must be filled in each training fold without the outcome.

**Where.** `turbotab/core/decisions.py:4815-4843 (_missing_fits_the_purpose)`

**Fix.** Reword the refusal at decisions.py:4832-4836 to the regime Sisk et al. support: 'a new row can arrive with blanks and has no outcome, and when it can, Sisk et al. 2023 found leaving the outcome out of the imputation preferred'; also note that with complete data at deployment they recommend the opposite, and say 'unless the purpose is inference' rather than 'under prediction' since the check also fires when no purpose is set (decisions.py:4830).

**Evidence.** Both referees and the audit agree that Sisk et al. 2023 recommend putting the outcome in development-data MI when deployment data are complete, and prefer leaving it out only when missingness is allowed at deployment. So citing them for an unconditional rule overstates the source. The refusal itself is defensible in the main regime: the app's MI is whole-table and outcome-inclusive, which would leak a held-out row's outcome, and the deployed pipeline must fill blanks without an outcome. In-fold outcome-inclusive MI isn't implemented, so no valid estimate is being withheld, only overstated in its scope. (confidence 0.7)

### A-119 · defect · random

**Claim.** Under Estimate or Describe, relabeling a two-valued covariate or energy column that is not the exposure, the outcome, a declared modifier or the fixed-effects column (coding its two spellings the other way round, with no blank and no third spelling) leaves the exposure's coefficient, its interval and the fitted values exactly unchanged and only flips the sign of that column's own coefficient, so the finding's repair changes no reported number and needs no calibration or post-lock check.

**Where.** `turbotab/core/sweep.py:583`

**Fix.** In turbotab/core/sweep.py:583-585, drop 'only' and 'exactly' from the sentence, e.g. 'the estimate for X and its interval do not change (under multiple imputation, only by the imputations' random draws)'. Alternatively, have adjustment_only (sweep.py:484) return False when the missing-data strategy is multiple imputation or the primary is a stochastic tree learner. Cite reparameterization equivariance beside FWL in materiality.invariance.

**Evidence.** Both referees and the audit agree that the exposure coefficient, its SE and the fitted values do not change under the recoding for OLS, GLM and Cox fits with an intercept, so the 'no change' recommendation is right for the number that matters. The text still overstates. The intercept shifts. Under multiple imputation, the engine's _norm_draw (methods/imputation.py:280-288) draws through chol((X'X + ridge)^-1), which does not transform with the recoding, so the pooled estimate moves by Monte Carlo noise (audit probe). That contradicts 'exactly' and the not_graded verify. No bias results, so the fix is to the scope and wording. (confidence 0.7)

### A-121 · defect · random

**Claim.** Under Estimate or Describe, a Belsley condition index of 30 or more with at least two model-matrix columns having variance-decomposition proportion 0.5 or more marks a near dependency among the exposure and adjustment set; when the exposure's proportions summed over those near dependencies reach 0.5, which of the dependent columns are held fixed changes what the exposure's estimate means (more exposure at fixed partners means less of what it replaces), so it is a Decide at the adjustment question and graded 'act on it'.

**Where.** `turbotab/core/materiality.py:773`

**Fix.** At materiality.py:883-885, reword the band-2 sentence to say the exposure is nearly a fixed mix of {others}, so holding them fixed leaves little of its own variation and which stay changes the contrast the estimate describes. Keep 'less of what it replaces' for dependencies flagged as a total or compositional (nested-parts) structure, and update SURFACING_POLICY §2.5 row K5 (line 371), which still describes |r| >= 0.9 pairs.

**Evidence.** Both referees' decisive counterexample (height in cm alone forming a dependency with the intercept) does not fire. collinear_noticing counts only non-intercept columns with a proportion of at least 0.5 (materiality.py:871-872). Running the app's belsley() on single and paired large-mean covariates gave 0% band-2 firing except when two independent columns have nearly equal small coefficients of variation (26% at an equal CV of 0.05, 0% at a CV ratio of 1.2), a rare spurious case. What remains is the audit's point: the substitution reading is stated for any near dependency, though it is exact only for parts summing to a total. (confidence 0.65)

### A-125 · defect · targeted

**Claim.** For an energy-intake screen under Estimate, if the share of rows removed times the leavers' largest SMD on the adjustment set is below 0.003697, removing the screen (keeping every row) moves the exposure's estimate by less than 0.1 of its 95% interval half-width, so the app recommends 'Doesn't change your numbers here'; from 0.003697 it says 'could bias', and from 0.2 'act on it'.

**Where.** `turbotab/core/materiality_calibration.json:15`

**Fix.** Reword the band-0 reason at turbotab/core/materiality.py:1034-1036 from 'That is below what would move the estimate' to a hedged prediction ('expected to move it little; checked after the plan is fixed'), and set rows_smd 'calibrated' to false in turbotab/core/materiality_calibration.json:15 until it is calibrated per SURFACING_POLICY §2.6 (reference journeys, cry-wolf budget, held-out cases), ideally adding the pure sampling movement SE*sqrt(s/(1-s)) of dropping a share s to the predicted band.

**Evidence.** The 0.003697 cut is the m_pre of the smallest falsely reassured case among 9 in-sample cases from one fixture with a single band-0 case (audit), and the proxy cannot guarantee the stated bound: dropping a share s of rows moves the estimate by SE*sqrt(s/(1-s)) from sampling alone (Referee A; at s = 3.7% that is already ~0.1 half-width SD), and the half-width shrinks with n while a fixed share x SMD shift does not (Referee B). I treat it as S2 rather than S1 because the sensitivity stage adds Banna's every-row analysis automatically and verify() (materiality.py:1086-1102) relabels an upgraded movement openly and drafts a limitation, so the reported numbers are corrected after the lock in the common single-screen case. (confidence 0.75)

### B-041 · defect · targeted

**Claim.** The within-strata residual with total energy kept in the outcome model has the standard model's estimand: its nutrient coefficient is the standard model's exactly, because the residual and total energy span the same model as the nutrient and total energy.

**Where.** `turbotab/core/methods/energy.py:1392-1396`

**Fix.** Give StratifiedEnergyAdjuster.estimand() (turbotab/core/methods/energy.py:1392-1396) its own sentence: the nutrient's coefficient with total energy adjusted within each level of the strata, which equals the standard model with strata and strata x energy terms, not the pooled standard model. Add a test against y ~ N + E + S + S:E + C.

**Evidence.** I accept both referees and the audit. The within-stratum residual N - a_s - b_s*E lies in span{1, N, E, S, S x E}, so by FWL its coefficient equals that of the fully stratum-interacted model, not that of Y ~ N + E + C. The audit's run on the engine's own adjuster (0.2918 against 0.2985) shows the shown identity ('the standard model's exactly', inherited from METHOD_TABLE['residual']) is false. The estimate itself is a legitimate energy-adjusted substitution, and with sex already among the covariates the gap is only the sex x energy term. The sentence is wrong; the number is not. (confidence 0.85)

### B-090 · defect · random

**Claim.** A spline or categories of an energy residual estimate the nutrient's curve on the energy-adjusted scale at mean energy, not the substitution curve at fixed total energy. The residual model equals the standard (nutrient plus energy) model only for a straight line. A spline of the nutrient with energy beside it (the standard model) is the route to the substitution curve.

**Where.** `turbotab/core/methods/exposure_form.py:975`

**Fix.** Reword residual_label at turbotab/core/methods/exposure_form.py:975-981 to say the residual curve is the fixed-energy curve at mean energy under a structure in which the curve shifts with energy along N - b(E - mean E), that the standard model spline(N) + E instead assumes one curve at every energy, that the two agree only for a straight line, and that the standard model is the route to the conventional (energy-invariant) substitution curve, not the only definition of one.

**Evidence.** The straight-line identity and the 'energy-adjusted scale at mean energy' wording are right (Referee B, audit; the residual is N - b(E - mean E)). But algebra settles the referees' split in A's favor on one phrase: at E = mean E the residual is N itself, so the residual model's plotted curve is that model's own curve in N at fixed total energy (mean energy), and each model recovers the fixed-energy curve only under its own structure (Referee A's simulation; B's simulation used an additive truth). So 'not the substitution curve at fixed total energy' states the additive assumption as a definition; offering the standard model is the conventional, defensible route, which makes this a wording fix. (confidence 0.6)

### B-151 · defect · targeted

**Claim.** Complete-case analysis gives unbiased regression coefficients if and only if whether a value is missing does not depend on the outcome, given the predictors.

**Where.** `turbotab/core/methods/missing.py:75`

**Fix.** Change 'only if' to 'if' in COMPLETE_CASE_ASSUMPTION (missing.py:75-77) and in row_loss_concern (missing.py:1763-1764). Add that logistic-regression odds ratios also stay unbiased when missingness depends on the outcome alone and not on the incomplete covariate (Bartlett, Harel & Carpenter 2015). Fix the nested 'Sound if ... only if' label at missing.py:139.

**Evidence.** Both referees and the audit agree, and their simulations match theory: White & Carlin 2010 give only the sufficient direction, and complete-case logistic slopes are unbiased under outcome-only missingness, where only the intercept shifts. The stated regime includes logistic models. The behavior (complete cases offered second, MI recommended) is defensible; the necessity claim and its garbled rendering are the defect. (confidence 0.85)

### B-228 · defect · random

**Claim.** Survey weights and PSUs given without a strata column give intervals that treat the PSUs as unstratified, which are too narrow where people were sampled in clusters, so the answer is blocked until the user attests that the table has no strata column.

**Where.** `turbotab/core/survey.py:408`

**Fix.** At survey.py:408-413, give the strata-missing branch its own reason: unstratified PSUs usually make the interval wider than the design's, and if PSU codes repeat across strata (NHANES SDMVPSU 1/2) distinct PSUs merge. Before accepting the attestation, check whether the PSU column has very few distinct codes (models/survey.py:153-160 keys PSUs by code alone).

**Evidence.** Both referees and the audit agree, and their simulations differ only in size: with the PSUs named, dropping the strata adds between-stratum variation to the with-replacement variance, so the interval is conservative, not 'too narrow'. The 'too narrow where people were sampled in clusters' tail is the missing-PSU reason, reused through _names_its_units' shared message. Blocking until the user names the strata or attests is defensible, so the defect is the reason the user is given, not the gate. (confidence 0.8)

### B-264 · defect · random

**Claim.** Absolute risks cannot be predicted from individually matched case-control sets, as a limit of the design: the conditional likelihood removes every set's intercept so there is no baseline risk to recalibrate, and matched controls are not a random sample of non-cases, so no single offset turns the sample's odds into anyone's risk.

**Where.** `turbotab/core/methods/case_control.py:200`

**Fix.** At case_control.py:200-204 (and the docstring at :48-53) replace 'This is a limit of the design, not a missing feature' with 'these sets alone cannot give risks: that needs external incidence by the matching factors (an attributable-risk model, as in Gail et al. 1989) or, for sets nested in a known cohort, its risk-set sizes (Langholz & Borgan 1997), which TurboTab does not yet take', and cite conditional-likelihood theory (Breslow & Day 1980) instead of Janes & Pepe 2008 for the intercept argument.

**Evidence.** Both referees and the audit agree, and I accept, that the two mechanisms are right for these data alone, so refusing inside the app is defensible. But absolute risk is estimable from matched designs with external incidence or cohort sampling information, so 'a limit of the design, not a missing feature' is false, and the cited Janes & Pepe 2008 is about classification accuracy, not risk. Since the refusal is correct for the data the app holds and only its stated scope and reason are wrong, this is S2, not S1. (confidence 0.8)

### B-288 · convention shown as a finding · random

**Claim.** GEE intervals in a cluster trial use t on the number of clusters minus the number of cluster-level terms (intercept, arm and any covariate constant within clusters), i.e. K − 2 for two arms and no cluster-level covariate.

**Where.** `turbotab/core/methods/trials.py:1365`

**Fix.** In turbotab/core/methods/trials.py:57-58 and the interval line built from df at :1365-1366, cite Li & Redden 2015 only for K - 2 with two arms and no covariates, and label counting only the within-cluster-constant terms as the app's between-within convention. With few clusters, also count individual-level covariates whose variance is mostly between clusters, or use a Satterthwaite (Bell-McCaffrey) df.

**Evidence.** Li & Redden used K - 2 with no covariates and wrote that adding covariates 'will change ... the degrees of freedom' (the audit verified this), so the cluster-level-terms rule is the app's own extension presented under their name. My simulation (Kauermann-Carroll-corrected sandwich, null arm effect, 3,000 reps, MC SE about 0.004): t on K minus the cluster-level terms is nominal when covariates vary mainly within clusters (0.051-0.055 at K = 8-10, covariate ICC 0-0.3). It is anti-conservative when covariates are mostly between-cluster but not constant: 0.068 at K = 10 with four covariates at ICC 0.5, and 0.078 at K = 8 with ICC 0.8. So referee B's 'sensible extension' holds in the main regime, and referee A's worry holds at the edge. (confidence 0.65)

### B-373 · convention shown as a finding · random

**Claim.** Under prediction a grouping of participants decides how the rows are validated (whole groups kept together, internal-external validation across groups), not the model's intercepts, so fixed-effects or cluster-only adjustment is refused.

**Where.** `turbotab/core/estimand.py:1818`

**Fix.** Reword turbotab/core/estimand.py:1818-1822 as the app's ruling for transport to new groups, e.g. 'Under prediction this answer names the groups to validate across (internal-external, beside the headline); for groups the model will meet again, enter the column as a predictor.' Drop 'by whole groups': the headline k-fold is grouped by the unit, not by this cluster (stages/modeling.py:1878-1902).

**Evidence.** The refusal is defensible. Cluster-only SEs and confounder fixed effects are inference ideas that change no prediction, and the grouping does drive leave-one-cluster-out validation beside the headline. Both referees show, though (Falconieri 2020; Bouwmeester 2013), that group intercepts are a valid predictive choice for groups the model will see again. So 'not the model's intercepts' is the app's ruling, not a rule. The audit adds that 'validated by whole groups' overstates what the headline folds do. (confidence 0.6)

### C-046 · defect · random

**Claim.** Under the metabolomics, genomics or survey lens alone, a yes/no outcome with no numeric column that reads as a follow-up time is a status at sampling and is correctly analyzed as counted over one period for everyone (a binary model), without asking about follow-up.

**Where.** `turbotab/core/estimand.py:188 (follow_up_gate, 'skipped' branch)`

**Fix.** In turbotab/core/estimand.py:183-185, treat the stored 'other' lens as no lens (return True when lens == ['other'] as well as when it is empty), as the comment at :176-178 states. Reword the skip at :207-209 to say this is a lens-based convention ('under this lens a yes/no outcome is usually a status at sampling ... if people were followed over time, Ask me anyway'). Add 'other' to test_routing_gate_repair's per-lens cases.

**Evidence.** I ran follow_up_gate. Lens [] and ['clinical'] ask the question, while ['other'] ('Something else, or not sure', a stored Lens value, decisions.py:62) is skipped exactly like ['survey']. The code therefore contradicts its own rule that 'not sure' is always asked, and a 'not sure' cohort outcome is silently fit as one-period binary. Both referees are right that a missing follow-up-named column is absence of evidence and that assay or survey data can be incident events. The skip is soft ('Ask me anyway'), and with no time column a Cox model is unavailable anyway (Referee B's mitigation), so this is a scope and wording fix, not a wrong number in the main regime. (confidence 0.75)

### C-047 · convention shown as a finding · random

**Claim.** For a time-to-event outcome under prediction with no declared prediction horizon, scoring predicted risk (Brier score and calibration) at the median follow-up time of the training rows is an appropriate default horizon.

**Where.** `turbotab/core/decisions.py:904 (FollowUpSpec.prediction_horizon); turbotab/core/models/performance.py (HorizonCalibration, brier_at)`

**Fix.** In turbotab/core/stages/modeling.py:1014-1018 name the default correctly and as a convention ('the median observed time, event or censoring, of the training rows: a convention, as no prediction horizon was declared') or switch to the reverse-Kaplan-Meier median, and expose FollowUpSpec.prediction_horizon (decisions.py:941) in the follow-up card so a clinical horizon can be declared.

**Evidence.** Both referees find no methods basis for the median as an 'appropriate' horizon and the audit confirms the contract (validation.py:990-1010) records a decision without a reason; McLernon et al. 2023 ties the horizon to the clinical question. The behavior is defensible as a fallback (half the training rows remain at risk at that time, which keeps IPCW stable), but the text calls np.median of event-or-censoring times 'the median follow-up time', which is shorter than the reverse-KM median follow-up epidemiologists mean, and no UI control sets prediction_horizon, so this unlabeled default governs every time-to-event Brier score and calibration. (confidence 0.75)

### C-110 · defect · targeted

**Claim.** When a repeating identifier's cluster reading is unsettled, not grouping by it ('no clustering without a confirmed unit') is the conservative path: the held-out draw and the folds group only by a settled cluster reading, and an identifier that rode along unconfirmed groups nothing.

**Where.** `turbotab/core/readings.py:20`

**Fix.** Relabel 'no clustering without a confirmed unit' in readings.py:20-24 (and BLUEPRINT §14.1) as the leash-safe path, which for held-out and cross-validated scores is the optimistic one. When an identifier repeats but did not group the folds, add the seal's 'treat as exploratory' note to the fold sentence built in stages/rows.py (~1930-1944).

**Evidence.** Both referees are right on the statistics: splitting a unit's rows across train and test is the optimistic direction, so 'conservative' is backwards. The behavior is mostly defended by asking. A 'repeated' grain answer must name its column (sequence.py:178-184). 'One row per unit' against repeating data needs an attestation (sequence.py:200-205). Every ungrouped seal says 'treat held-out scores as exploratory', with a 'Keep each X's rows together' exit (seal.py:254-292, row_previews.py:556-572). The gap is the CV folds and inner tuning folds (split_inputs, rows.py:1947-1977): they ungroup in the same cases without saying so, and that can favor models that memorize units. (confidence 0.7)

### C-144 · defect · random

**Claim.** In a nutrition exposure-outcome analysis, demographic covariates (age, sex, race/ethnicity, education, income/poverty ratio, occupation, marital status) are pre-exposure common causes of the exposure and the outcome and belong in the adjustment set as confounders (the criterion's answers: causes exposure yes, causes outcome yes, after exposure no).

**Where.** `turbotab/core/covariate_guesses.py:90`

**Fix.** In covariate_guesses.py:90-96, make the demographic reason lens-neutral ('set before the exposure', not 'before the diet was measured'). Give income, poverty ratio, occupation, marital status and SES (covariate_guesses.py:151-152) a guess of after-exposure 'unknown', or add the caveat that in cross-sectional data they can follow the exposure or the outcome. Keep age, sex and race/ethnicity as pre-exposure.

**Evidence.** Both referees agree that income, occupation and marital status measured at the same visit can be consequences of the exposure or the outcome. The card asserts 'set before the diet was measured' as fact. The guess also applies under every lens, since adjustment_card has no lens gate (estimand.py:971-985), so a clinical exposure gets text about 'the diet'. The behavior is defensible in the main regime: adjusting for SES is the field's Models 1-2 convention, labeled as such and confirmed by the user. So this is a wording and scope fix, not a harmful default. (confidence 0.7)

### C-151 · convention shown as a finding · random

**Claim.** An outcome over follow-up (a time-to-event task, a binary outcome whose name says incident/onset/new/developed/conversion, or an outcome named as follow-up, change, delta, endline, final, or with a time since baseline such as 12m/5y/24wk) implies the covariates were measured at baseline, before the outcome; occasion labels such as v1, t0 or w2 do not, because they may be the baseline itself.

**Where.** `turbotab/core/covariate_guesses.py:341`

**Fix.** At covariate_guesses.py:320-321, word the guess reason as an inference: 'the outcome's name reads as follow-up, so this was probably measured at baseline, before it' (likewise 'probably at the same visit'). Fix followed()'s docstring at :342, which says 'implies'.

**Evidence.** Both referees and the audit agree that an outcome's name gives a prior about when the covariates were measured, not proof. The code states the timing as fact in the reason the user reads, and it flips the pre-filled adjustment answers. The behavior is defensible because it is a guess the user confirms and the with-and-without pair hedges it. What is wrong is presenting a naming heuristic as a finding. (confidence 0.7)

### C-241 · defect · random

**Claim.** When missingness of the predictors differs by sociodemographic group, the sound missing-data handling for inference is multiple imputation with the group in the imputation model, while complete-case analysis must state its assumption.

**Where.** `turbotab/core/stages/explore.py:745`

**Fix.** Reword the soundness text at turbotab/core/stages/explore.py:745-746 to match what the routed question delivers ('multiple imputation fills blanks given the analysis variables, so the group is in its model only when it is in the analysis; complete cases state their assumption'), or let MI take the group column as an auxiliary variable when this finding fires.

**Evidence.** The audit's point holds, and the referees' in part. The imputation model holds the analysis variables, the outcome, energy and the design (methods/missing.py:6-16), with no path for auxiliary variables. The group candidates are any non-target column named like a sociodemographic group (explore.py:391), so a group that is not a predictor never enters the imputation model the lever promises. The 'complete cases states its assumption' half is right: it does not call complete cases unsound. MI is ranked first under inference anyway, so the main regime, where the group is already a covariate, is unaffected. (confidence 0.7)

### C-L02 · defect · tier2

**Claim.** The precision of a held-out AUC is approximated by Hanley & McNeil's (1982) standard error at an assumed AUC of 0.75 for the held-out events and non-events, and the precision of a held-out macro-F1 by a proportion's worst-case 95% interval (±1.96·√(0.25/k)) on the rarest class's k held-out rows.

**Where.** `turbotab/core/seal.py:689-731`

**Fix.** In seal.measure's multiclass branch (seal.py:722-726), replace 1.96·√(0.25/k_rarest) with the class-averaged bound 1.96·√(Σ_c 0.25/k_c)/C, which came within about 1.0-1.4x of the simulated half-width, or else word the current figure as 'at most about'. Add a test of _auc_halfwidth against a hand-computed Hanley & McNeil value.

**Evidence.** Both referees agree, and my simulation confirms it: the shown macro-F1 half-width is 2.5-3.5x the true one (for example ±0.127 vs ±0.052 for 3 classes, n = 300, and ±0.098 vs ±0.031 for 5 classes, n = 1,000), because macro-F1 averages C per-class F1s. The error is conservative, the figure is labeled approximate in PRECISION_NOTE, and the AUC half follows Hanley & McNeil exactly, so this is S2: it can push users toward larger holdouts. (confidence 0.7)

### D-001 · defect · tier2

**Claim.** In a dietary analysis with total energy in the outcome model (standard model, or residual method with energy kept), a nutrient's coefficient estimates a substitution: more of that nutrient in place of the same energy from the energy sources not in the model, at fixed total energy, not the effect of simply eating more of it; with several energy sources in the model, 'everything else' is only the sources left out.

**Where.** `turbotab/core/teaching/content.py:1099-1104`

**Fix.** In methods/energy.py describe_model, when no exposure or adjusted term carries energy (terms empty with a method set, energy.py:1657-1704), state 'more of the nutrient at the same total energy; it carries no energy, so nothing is substituted' instead of METHOD_TABLE's substitution estimand (energy.py:153-155, 167-171). Scope the teaching at content.py:1099-1104 and 1397-1402 to energy-bearing nutrients.

**Evidence.** Both referees found the substitution reading wrong for a nutrient that carries no energy. I confirmed it reaches users: the energy question stays open under the dietary lens for any exposure (interview.py:278-299). describe_model for a sodium_mg exposure with the residual or standard method returns 'more of the nutrient at the same total energy, in place of the average of all other energy sources' and '...implicitly a substitution for the average of all other energy sources'. For macronutrients the reading is correct and certified by test_5b (audit accepted), so the main regime is right and only the scope is wrong. (confidence 0.75)

### D-009 · defect · random

**Claim.** Adjusting for total energy does not by itself make a substitution isocaloric: the swapped components must be measured and analyzed in calories, and with components in grams or servings the substitution reported is not the one intended.

**Where.** `turbotab/core/teaching/content.py:1243-1247`

**Fix.** Narrow turbotab/core/teaching/content.py:1219-1222, 1236-1237 and 1243-1247 to foods and servings, whose energy density varies (Tomova et al. 2022). Say that macronutrient grams are converted to kcal by Atwater factors (readings.kcal_per_unit), so a swap in grams is still isocaloric.

**Evidence.** Both referees and the audit agree. With energy held fixed, a macronutrient in grams is a constant rescaling of its kcal, and the engine itself converts grams through Atwater factors (readings.py:2312-2339; test_substitution's grams test), so the teaching text contradicts the engine. The text is right for foods and servings, where Tomova 2022 describes 'mixed-unit' estimands. A user who follows it would at worst convert units for no reason. (confidence 0.8)

### D-012 · defect · targeted

**Claim.** When a unit's rows are repeated measurements of one quantity, averaging k of them cuts within-person variance k-fold.

**Where.** `turbotab/core/coach.py:492-495`

**Fix.** At coach.py:493-494, change the note to 'Averaging `k` replicates divides the day-to-day error in each person's mean by up to `k` (fully when the replicates' errors are independent, as for non-consecutive recalls)'.

**Evidence.** The note is shown for every mean of repeats. Both referees and the audit agree it misnames the quantity (within-person variance itself does not shrink; the error variance of each person's mean does) and drops the independence condition. The app's own SETTLED teaching text (content.py:557-561) says that condition fails for consecutive-day recalls. It changes no number, so the fix is to wording. (confidence 0.6)

### D-018 · defect · random

**Claim.** For internal validation at typical clinical sample sizes, bootstrap optimism correction is the recommended default, repeated cross-validation is acceptable, and a single train/test split is the weakest option.

**Where.** `turbotab/core/teaching/content.py:91-96`

**Fix.** Reword SINGLE_SPLIT at teaching/content.py:91-96 to 'For regression-type models below about 20,000 units, bootstrap optimism correction is the recommended default and repeated cross-validation is acceptable; tree ensembles keep their cross-validated score, since the bootstrap overstates them; a single split is the weakest option', citing Steyerberg et al. 2001 and Coley et al. 2023 beside the pack.

**Evidence.** Both referees' simulations (random forest: bootstrap-corrected AUC 0.94-0.95 against a true 0.63-0.68) are right about the method, but the app already acts on it: random_forest, boosted_trees and xgboost all declare bootstrap_optimism = False and keep their cross-validated score, and the bootstrap option carries BOOTSTRAP_CAUTION (validation.py:330-332). So the referees' bite (a grossly optimistic score on the seal card) does not happen. What remains is a teaching card stating an unconditional rule that the app's own practice limits to regression-type families below RESAMPLE_BELOW: a scope fix, not a wrong number. (confidence 0.75)

### D-021 · defect · random

**Claim.** Each cross-validated score's standard error, and the interval for each pair of families' difference (the corrected repeated k-fold t over at least 10 × K folds), computed on the user's rows, tell whether two models differ, in place of a rule about sample size.

**Where.** `turbotab/core/teaching/content.py:1018-1022`

**Fix.** In turbotab/core/teaching/content.py:1018-1022, replace CLIN_A53 with the Nadeau & Bengio 2003 / Bouckaert & Frank 2004 citation. Reword the card to say the corrected repeated k-fold interval is approximate and usually conservative, and descriptive across several pairs (validation.py:581): one that excludes zero says a family is ahead, and one that includes zero says these rows cannot tell them apart, not that they are equal.

**Evidence.** The card is tagged SETTLED and cites CLINICAL_SURVEY_PACK §A5.3, which is about discrimination versus calibration and does not support it (audit, accepted). I ran the app's corrected_t under a true null with 10x10 CV. The 'ahead' rate was 0.3% for OLS at n=40, 0-1% for exchangeable deep trees at n=30-40, and 0% at n=200. That supports Referee A's 'conservative' and does not reproduce Referee B's 8.7-12% for unstable learners. The verdict wording in family_differences ('not distinguishable on these rows') is fair, so the behavior is defensible. The card overstates what the interval 'tells' and carries a wrong source. (confidence 0.7)

### D-032 · defect · targeted

**Claim.** In a randomized trial, covariate adjustment only sharpens precision and causal wording is allowed; a case-control sample cannot give prevalence or absolute risk; matched sets are analyzed within each set; an observational study adjusts for confounders and speaks of associations.

**Where.** `turbotab/core/teaching/content.py:404-407`

**Fix.** Reword DESIGN['why'] at turbotab/core/teaching/content.py:404-407: 'in a randomized trial adjustment is for precision, not confounding, and the effects reported are marginal (standardized); a case-control sample cannot give prevalence or absolute risk without its sampling fractions; individually matched sets are analyzed within each set'.

**Evidence.** Both referees and the audit are right on two points. 'Only sharpens precision' fails for conditional odds and hazard ratios, which are non-collapsible, as the app's own term at content.py:1393 says. A case-control sample with known sampling fractions does give absolute risk. The app's own trial core standardizes to marginal risks and its case-control core handles frequency matching separately, so its analyses agree with the card, but the teaching sentence states a general rule a user could carry elsewhere. The trial designs are 'Not available yet', so the card only explains for now. (confidence 0.75)

### D-037 · defect · tier2

**Claim.** On a food-frequency questionnaire a blank frequently means 'never', especially among older participants, so that missingness is not at random, and standard multiple imputation, which assumes it is, is an improvement but not a solution.

**Where.** `turbotab/core/teaching/content.py:935-939`

**Fix.** In teaching/content.py:935-939, say 'often not at random' (Fraser et al. 2009). Say that MI improves on complete cases, while coding blanks as zero is the field convention and works for rarely eaten foods. Badge the handling half DISPUTED, as NUTRITION_PACK §06 (lines 659-662) does.

**Evidence.** The mechanism half holds: Fraser 2009, verified by the audit, finds excess true zeros, more of them in older subjects, which also refutes referee B's doubt on age. But the app makes the pack's 'often' categorical and badges the whole section SETTLED, though the pack marks the handling DISPUTED. It also never says what MI improves on. Referee A's simulation, consistent with the pack's own zero-coding note, shows zero-fill beats MI for intake when blanks are mostly 'never', so the drawer can steer users away from the better analysis for rare foods. (confidence 0.7)

### D-044 · defect · random

**Claim.** A TNTC ('too numerous to count') plate count is right-censored above the countable range (15–300 colonies per plate in FDA's BAM, 2026 edition), not missing, whereas QNS or a hemolyzed specimen is a measurement failure with no value and is treated as missing.

**Where.** `turbotab/core/teaching/content.py:265-272`

**Fix.** Either add a censored lever for TNTC (lower-bound value plus a censoring indicator), or make the 'Read as numbers' consequence at repairs.py:927-933 say that blanked TNTC cells are right-censored highs whose deletion biases the column downward. Also remove 'tntc' from the failure tokens and LooksFor text (clinical.py:138, packs.py:5573), and quote the BAM 15-300 colonies-per-plate range only for plate counts (finding_words.py:303-306).

**Evidence.** The claim is true, as both referees verified against BAM ch. 3 (2026). The app's behavior contradicts its own text: the finding and teaching say TNTC is censored, not missing (finding_words.py:303-306; content.py:265-272), but no censored handling exists. The only lever, 'Read as numbers', sets TNTC cells blank, i.e. missing, and its consequence line says only that they 'become blank' (repairs.py:918-944). The pack's own LooksFor text still calls TNTC a measurement failure (audit accepted). Its sentence does disclose the blanks, so this is S2 rather than S1. The BAM plate range is also quoted for clinical columns such as urine WBC per HPF. (confidence 0.65)

### D-070 · defect · random

**Claim.** Rows that are imputed copies of one record (e.g., NHANES DXA's five imputations) are, under inference, each analyzed as a completed dataset and the estimates pooled by Rubin's rules, which carries the imputation's uncertainty.

**Where.** `turbotab/core/voice.py:2085-2088`

**Fix.** Change the imputed_copies option text at teaching/content.py:541-543 to 'under inference each copy is analyzed and the estimates pooled by Rubin's rules; combining them is recorded as a limitation', and gate the pooled wording at voice.py:2085-2088 on stages.modeling.imputed_copies_column(state) being non-None, falling back to the not-pooled sentence otherwise.

**Evidence.** Both referees are right, and the R parity test (mitools::MIcombine and mice::pool.scalar to 1e-6) confirms it: when the copy column is named and each copy is kept as a record, the engine pools by Rubin's rules as NCHS directs. Two texts misstate this, as the audit found. The teaching option says copies are 'under inference kept only as a recorded limitation', the opposite of what the fit does. The methods sentence claims pooling whenever purpose is inference and it is unacknowledged, while imputed_copies_column (modeling.py:1462-1472) pools only with implicate_column set and unit 'row'. The behavior is right; the wording needs fixing. (confidence 0.65)

### D-073 · design doc only (not built) · random

**Claim.** Values filled before the table was made by an unrecorded method are read as missing and re-imputed in each training fold at the median without indicators, because their missingness belongs to the upstream process, not the person, and does not exist at the moment of use.

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md §17 ruling 4 (lines 1486-1488)`

**Fix.** Before P22 is built, amend docs/turbotab-next/calm/BEATS_PREDICT.md:1488 (§17 ruling 4). Give the reason for re-imputing in each fold as leakage: an unrecorded upstream fill may have used the outcome or other rows. Make the missing indicator conditional on whether the fill marker will exist at deployment and whether the original missingness is person-driven. Label the median-without-indicator choice a judgment, and add a test when P22 lands.

**Evidence.** The audit's search confirms that no engine code serves this: P22 is listed as new work (BEATS_PREDICT.md:28, :1246), and the 'Read them as missing' beat was applied outside the engine. Both referees are right that the stated reason fails in realistic cases. Provider fills often stand in for person-level nonresponse, which can be informative, and if deployment data pass through the same pipeline the fill marker does exist at the moment of use. Sisk et al. 2023 found that indicators often help and do not support median fill. Re-imputing in each fold is sound for a different reason (leakage). (confidence 0.7)

### D-077 · design doc only (not built) · random

**Claim.** Internal–external cross-validation across survey cycles is summarized by random-effects meta-analysis of the log MSE with a Hartung–Knapp–Sidik–Jonkman interval and a prediction interval for a new cycle on t(k − 2), because on the identity scale cycles with larger MSE have larger standard errors and so less weight, which is optimistic.

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md §9 ruling 3 (line 1146)`

**Fix.** Wire the metafor-checked REML + HKSJ pooling from methods/site_validation.py into validation.internal_external in place of the DerSimonian-Laird z interval in turbotab/core/models/performance.py:906-917 (INBOX T3), and before building P27 drop the identity-scale-optimism reason for the log scale (and Snell 2018 as its source) or label a back-transformed log-MSE as the typical (median) cycle's MSE, not the mean.

**Evidence.** The ruling is unbuilt: performance.random_effects pools on the identity scale with DL tau², a normal-z CI and a t(k-2) prediction interval (audit, confirmed in code). My simulation (k = 8 cycles) shows the stated reason is wrong in the survey regime: identity-scale optimism vanishes by n = 1,000 per cycle (pooled 1.008 vs true mean 1.011), while the back-transformed log-scale estimate targets the median and sits below the mean under heterogeneity (1.006 vs 1.046), as both referees caveated. What does bite in the shipped engine is the interval: the DL z CI covered 0.87-0.91 under heterogeneity versus 0.94-0.96 for HKSJ, so the user's pooled interval is modestly too narrow today. (confidence 0.7)

### D-078 · design doc only (not built) · random

**Claim.** Under Predict, each continuous predictor of least squares, ridge and the elastic net enters as a restricted cubic spline (Harrell's knots, placed in each training fold) by default when Riley's sample-size criteria hold for the terms it makes (here 3,916 rows needed for 73 terms at an expected shrinkage of at least 0.9).

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md §9 ruling 6 (line 1174)`

**Fix.** Mark BEATS_PREDICT §9 ruling 6 as unbuilt (P15, line 1239) wherever the ledger presents it as behavior. When building it (models/pipeline.py, methods/levers.py), state the anticipated R^2 behind the Riley count (3,916 for 73 terms assumes sample_size.R2_SHARE = 0.15), note that it counts nominal parameters (conservative for ridge and the elastic net), and confine the curves to the linear families: explore_steps (pipeline.py:797) currently adds lever_forms to every family.

**Evidence.** The audit's code check holds. SetLevers.forms defaults to 'none' (decisions.py:1716, 1724), splines come only from an Explore lever the user records, and explore_steps applies them to every family, trees included. So no spline default conditioned on Riley's criteria exists. Referee A's reproduction (3,916 at R^2 0.15, about 2,851 at 0.20 and 6,042 at 0.10) shows the gate depends on an assumed R^2. The built sample-size module does state that assumption ('a model explaining 15% of the variance'), but the ruling does not. (confidence 0.8)

### D-082 · design doc only (not built) · tier2

**Claim.** Grouped SHAP values and accumulated local effects describe how a model turns measures into estimates, not what changing a measure would do; within a group of correlated inputs the split of credit is arbitrary, and one input's share is said to be unstable when the reseeds' rank correlation is under 0.7.

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md R3 Why? (line 842)`

**Fix.** Before BEATS_PREDICT.md §6 item 6 (line 1067) is built, label the 0.7 cut as a convention and base 'one input's share is unstable' on a per-input measure, such as the spread of its share or rank across reseeds, rather than the whole-ranking Spearman ρ that explain.py reports (Stability.rho_mean).

**Evidence.** The engine already says, correctly, that explanations describe the model, not an effect (explain.py EFFECT_SAYS, test_5a). It reports the mean Spearman correlation across bootstrap refits with no threshold, so the 0.7 rule and the within-group 'arbitrary split' line exist only in the design doc. I accept both referees that 0.7 has no source, and referee A that a whole-ranking correlation cannot flag one input. Referee B's simulation (per-row ρ = 1.00 while x1's share ranged 0.40-0.49) shows the rule would call an arbitrary split stable. (confidence 0.75)

### D-085 · defect · random

**Claim.** Under Predict, energy adjustment and substitution are not asked, because the energy model decides what a nutrient's coefficient means (an estimation question) and models that keep total energy predict alike; nutrients enter as recorded beside total calories.

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md §9 ruling 5 (line 1169)`

**Fix.** Restrict the last sentence of ENERGY_WHY (turbotab/core/methods/dietary_caveats.py:47-48) to linear models with the same column space (standard and residual), or drop it. The orchestrator should then either retire BEATS_PREDICT §9 ruling 5 or build it knowing that the forms do not predict alike for trees.

**Evidence.** The 'not asked under Predict' ruling exists only in the design doc. The engine asks the energy question under prediction (interview._energy_applicability has no purpose gate). The live sentence 'models that keep total energy predict alike' is false outside the OLS standard-versus-residual case. My simulation (n=1000, 5-fold CV) found density+E ahead of standard even for OLS (R² 0.480 vs 0.418 at p=5). For a composition-driven outcome at p=20, standard against residual gave 0.351 vs 0.445 for a random forest and 0.364 vs 0.442 for boosting. That sides with Referee B over Referee A's 'at most 0.01'. Reported scores stay honest, but the text tells the user the choice does not matter. (confidence 0.75)

### D-091 · design doc only (not built) · random

**Claim.** Views that never use the outcome may shape the representation (transforms, knots, filters, imputation rules), and under inference steps learned without the outcome may be fit on the full analysis sample; views that use the outcome may only add candidates to a declared search under prediction (never prune it) and under inference are reported, not acted on, unless a declared selection-aware method carries them.

**Where.** `docs/turbotab-next/understanding/research/WHAT_EXPLORATION_MAY_DECIDE.md §1–§2 (lines 11-33)`

**Fix.** In docs/turbotab-next/understanding/research/WHAT_EXPLORATION_MAY_DECIDE.md §1 (lines 11-21) and the prediction cell of the predictor-outcome row in §2, say that an outcome-informed candidate may enter only as a declared rule rerun in every fold, or else be scored only on the sealed hold-out. Label 'add, never prune' as the app's convention rather than a principle. Note that the registration-time enforcement (UNDERSTANDING_LAYER §0 item 5) is not yet built.

**Evidence.** Referees A and B disagree on whether add-only is safe, so I simulated it: n=60, 2,000 noise features, a declared search over 5, pure-noise outcome. Adding the top development-row correlate gave a CV MSE of 0.96 against a true MSE of 1.28. The declared search alone gave 1.14 against 1.13. Rerunning the same addition inside each fold gave 1.31, which is honest. Referee A is right that adding is as leaky as pruning. What protects the score is the doc's own §1 'rerun wherever the data are resampled' requirement or a sealed hold-out (Referee B's regime), and an eyeball addition cannot be rerun. The audit confirms no thread registry exists in turbotab/core, so the claimed refusal is unbuilt beyond specific validators. (confidence 0.7)


## S3 (16)

### A-051 · defect · random

**Claim.** For an unpenalized regression under prediction, multiplying its coefficients by the calibration slope — the bootstrap's optimism-corrected slope when the bootstrap ran, else the out-of-fold cross-validated slope — and re-estimating the intercept with the shrunk linear predictor as an offset is a sound model update that corrects for overfitting.

**Where.** `turbotab/core/models/decision_curve.py:356`

**Fix.** In stages/evaluation.py:_shrinkage (~line 448) and explore_previews.py:447-450, word the offer and caption by the slope's side of 1 (or withhold the offer when slope_of returns a slope >= 1), so 'Shrink' and 'each risk moves toward the mean' are never shown for an expansion; and name in the offer that the out-of-fold slope comes from (K-1)/K-row fold models and shrinks somewhat more than the bootstrap's.

**Evidence.** The method itself is the standard, sourced update (Collins 2024, Van Houwelingen & le Cessie 1990), offered opt-in (UpdatingSpec default 'none', decisions.py:1785), with the bootstrap slope used by default below 20,000 units; both referees agree it improves calibration on average, so referee A's small-n noise objection (Riley 2021) is a caveat for any shrinkage, not an app error. What is actually wrong is narrow, as the audit found: no guard when the slope exceeds 1 (then the coefficients expand under a 'Shrink' label and a 'toward the mean' caption), and the out-of-fold fallback over-shrinks a little, which both referees accept; it is labeled, and B's simulation shows it still beats no update. (confidence 0.65)

### A-134 · defect · random

**Claim.** Open noticings must be given their dispositions before the plan's lock under Estimate and Describe (and under an unanswered purpose), or before the held-out rows open under Predict; a disposition recorded after that point would be chosen with the numbers in view, so the triage is refused and each noticing is decided on its own as made after the estimates or held-out scores were seen.

**Where.** `turbotab/core/sweep.py:763`

**Fix.** In _confirm_the_triage (turbotab/core/sweep.py:767-773), say 'as made after the estimates were seen' only when an estimate has been shown (the after_estimates signal that ProjectService.decide and plan_lock.py:176-178 use), and otherwise 'as made after the plan was fixed'.

**Evidence.** Both referees accept the gate as sound pre-specification logic. Putting the Predict gate at the seal protects the held-out score, because the cross-validated comparison is part of the selection. But the lock is recorded when Fit is pressed, which can be before any estimate is shown, and the record marks a later decision as post hoc only once an estimate has been seen (plan_lock.py:176-178). So in that narrow window the refusal text says 'after the estimates were seen' when none were. I accept the audit's point on this. (confidence 0.65)

### B-028 · convention shown as a finding · random

**Claim.** A column named as a percent of energy whose values all lie at or below 1 holds fractions rather than percentages, and values outside 0–100 are not percentages of energy, so either is refused before a percent-of-energy swap.

**Where.** `turbotab/core/methods/percent_energy.py:52-64`

**Fix.** At percent_energy.py:62-64 say 'every value is at most 1, which reads as fractions (0.35 for 35%)' instead of 'it holds fractions', and add an exit at decisions.py:4548-4552 that confirms the values are genuine percentages.

**Evidence.** The 0-100 bound is definitional; the max <= 1 rule is a data-sanity heuristic stated to the user as fact, as both referees say. The false-positive regime is narrow in this app: the name gate admits only protein, carbohydrate, fat (including trans and SFA/PUFA), alcohol, fiber and sugars (EPA, DHA, ALA and omega-3 names are not admitted, as I checked with is_percent_of_energy), and at the default 5-point step (decisions.py:816) a genuine column with max <= 1 has no rows on support anyway. A refusal only bites for trans fat or alcohol in a small low-intake sample with a user-reduced step. (confidence 0.75)

### B-051 · convention shown as a finding · random

**Claim.** A bootstrap band whose failed refits are dropped is conditioned on the resamples a model could be fit on; with at least 90% of refits succeeding this barely moves it, but with more failures it describes another population of resamples, so no band is drawn below 90% success.

**Where.** `turbotab/core/methods/substitution.py:108-112`

**Fix.** At substitution.py:646, word the refusal as the app's convention ('by this app's convention a band needs at least 90% of them'). When a band is drawn with failures, show n_ok of n_boot beside it, noting that resamples missing a class are left out. Change the comment at :108-112 to say the band barely moves when failures are unrelated to the curve, and that a class missing from resamples shifts that class's own band.

**Evidence.** Both referees are right that informative failures shift a tail even at 90-99% success. In the app the failures that matter come from class_refit_band, where a resample missing a class fails (substitution.py:1015-1018). Reaching 1-10% failure takes a class with only about 2-5 members, where any per-class band is fragile anyway. Referee A's own run found the continuous substitution-shift band moved under 2%. The 90% is unsourced and the code says so, but the user sees it stated as a requirement. (confidence 0.65)

### B-119 · convention shown as a finding · random

**Claim.** For clustering, a column with exactly two distinct values (e.g. a 0/1 code) is a category, not an amount.

**Where.** `turbotab/core/methods/subgroups.py:263`

**Fix.** Label the two-value rule as TurboTab's convention in read_features (turbotab/core/methods/subgroups.py:216-217). In the k-means/mixture category refusal (subgroups.py:913-929), give a two-valued numeric column a 'Read it as an amount (0/1)' exit (numeric=[...]), as the ambiguous 3-10-value columns get, and stop citing Huang's arbitrary-coding reason for a single 0/1 column. This becomes S2 once the module is wired to a stage.

**Evidence.** Referee B is right that reading binaries as nominal is the usual convention for mixed-type data (Gower; Hennig & Liao). Referee A is right that the number of values observed does not fix the measurement level. A single 0/1 column has one non-arbitrary coding, so the refusal's Huang reason does not apply to it. The refusal also offers no amount exit, although read_features honors numeric= (audit). No stage calls the module yet (only contracts.py lists it), so no user reaches it today. (confidence 0.65)

### B-152 · convention shown as a finding · random

**Claim.** Under inference, losing more than 10% of the rows to complete-case analysis is enough to warrant a concern, shown with a comparison of the kept and dropped rows' outcome and predictors.

**Where.** `turbotab/core/methods/missing.py:1751`

**Fix.** At missing.py:1764-1768, replace 'more than the 10% at which CLINICAL_SURVEY_PACK flags listwise deletion' with 'more than 10%, a convention this app uses to prompt a look; the share lost does not measure bias', and add to the sentence that kept-vs-dropped differences in predictors are expected even when complete cases are unbiased.

**Evidence.** Both referees are right that the 10% figure is a rule of thumb (Bennett 2001) and that the share missing does not measure complete-case bias (Madley-Dowd 2019, White & Carlin 2010). But the app only warns, and its sentence already states the right condition: 'unbiased only if whether a value is missing does not depend on the outcome, given the predictors' (missing.py:1766-1767). What remains is labeling: the user-facing text cites an internal file name as the authority for an unsourced threshold. (confidence 0.7)

### B-506 · design doc only (not built) · random

**Claim.** Change-in-estimate selection among covariates already judged pre-exposure causes cannot detect mediators or colliders, and for an odds ratio or hazard ratio with a common outcome the estimate changes from non-collapsibility even without confounding, so it is block-and-record under inference and conflicts with OR/HR estimands for common outcomes.

**Where.** `docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 1(c)`

**Fix.** Mark MODELING_SEQUENCE §0 ruling 1(c) and §2's change-in-estimate rung as doc-only (not built) in the claims ledger, or, if change-in-estimate is ever offered, implement the block-and-record label with the VanderWeele 2019 reason; no code change is needed while no such option exists.

**Evidence.** Both referees judge the statement true and the audit verified VanderWeele 2019 ('inappropriate for non-collapsible measures such as the odds ratio or hazard ratio if the outcome is common'); test_estimand_1_effect_measure.py::test_1 certifies only the non-collapsibility premise. A grep of turbotab/ and turbotab/frontend/src finds change-in-estimate only in that test's docstring, so the claimed block-and-record label does not exist; since the app offers no change-in-estimate selection either, no user is exposed and the gap is ledger hygiene. (confidence 0.85)

### BX-063 · defect · tier2

**Claim.** sd(r_i)/√n_sim over the simulated units' final risks bounds the g-formula's Monte Carlo error from above.

**Where.** `turbotab/core/models/time_varying.py:712`

**Fix.** Change 'Monte Carlo error at most' to 'Monte Carlo standard error about' at stages/time_varying.py:1221, and rewrite the docstring at models/time_varying.py:79-81 to say it estimates the error when the base is resampled (the default, SIMULATIONS = 10,000) and overstates it only when n_sim equals the units.

**Evidence.** My simulation used the app's own tv.gformula (GTRUTH spec, 600 units, 300 seeds). With the default resampled base (n_sim = 3,000), the reported max(sd/sqrt(n)) was 0.00156 against a true run-to-run SD of 0.00168, below it in every run. With n_sim = units it was 0.00350 against 0.00261, a true upper bound. Referee B and the audit are confirmed: in the default regime the figure is an estimate, not a bound. The error is far below the sampling error, so the impact is cosmetic. (confidence 0.85)

### C-125 · defect · random

**Claim.** Under the fit and its imputation model a predictor with exactly two values needs no code-or-amount answer, because coding it as one indicator or as a numeric amount gives the same slope, interval and every prediction (only the intercept's reference changes), and both imputation models fill it as a yes/no; when a unit's rows are combined the two readings differ (mean = a share vs the most frequent value), so there it is asked.

**Where.** `turbotab/core/readings.py:1593`

**Fix.** Correct the docstring at readings.py:1601-1604 to say the fit and every prediction are the same either way, while the slope is per unit as an amount and per level as a code, which differ by the gap between the two values (and with an interaction, a partner's coefficient moves its reference).

**Evidence.** The referees are right that 'same slope, interval' holds only for a one-unit gap. Nothing a user sees relies on it, though: a two-valued number stays an amount whose coefficient is reported per unit, g-computation sets the exposure to its two values (models/effects.py:356-369, settings_of), and ridge and the elastic net standardize their columns. So skipping the question changes no number, and Referee A's 'off by the gap' bite does not occur. Only the developer-facing reason is wrong. (confidence 0.7)

### C-132 · defect · random

**Claim.** Total energy's unit is settled by its values only when its median ratio to the macronutrients' Atwater reconstruction lies within 0.90-1.10 (kcal, over the same days the macronutrients span); a ratio within 5% of 4.184 x N or of N (N >= 2) could be kJ or an N-day kcal total (a 4-day kcal total reads 4.00, within 5% of kJ's 4.184), so the unit and the day count are asked together with the fitting readings offered, and a ratio near 1 never settles how many days the values span.

**Where.** `turbotab/core/readings.py:892`

**Fix.** In readings.unit_day_candidates (readings.py:866, 884-887) use the kcal pass band's multiplicative tolerance (0.90-1.10 times factor x days) for kJ x N and kcal x N as well, so a kJ file with the reconstruction error the kcal band already tolerates keeps kJ among the offered readings; add a c2 case at a kJ reconstruction ratio of 0.94 and of 1.07.

**Evidence.** The settle rule is safe, and every referee and the audit agree: only [('kcal', 1)] settles, so nothing is settled wrongly. Referee A's asymmetry is real: probing unit_day_candidates, a 1-day kJ file at a reconstruction ratio of 0.93-0.95 is offered only 'kcal over 4 days', and one at 1.06 is offered nothing. But the ask keeps an 'Another unit or day count' exit (decisions.py:3711-3720), and kJ read as a 4-day kcal total misstates daily energy by only 4.184/4, about 4.6%, not the factor of 4.184 A states. Referee B's two-macronutrient case does not arise in my probe: protein + fat alone hits the drift gate ('the macronutrients do not reconstruct it'), with no spurious 2-day reading. (confidence 0.6)

### C-134 · defect · random

**Claim.** An energy source's amounts are settled as grams by the Atwater identity only where it excludes every other unit: the identity passes (ratio 0.90-1.10) with the column read in grams, the column is not alcohol (standard drinks hold 8-20 g by country and alcohol is a minor source), the column carries at least 1 - 1/1.10 (about 9%) of reconstructed energy on the median row (below that any unit, or none, keeps the ratio in band), and reading it in kg, kcal or kJ instead moves the ratio out of band; otherwise a substitution or partition waits for the user's recorded unit.

**Where.** `turbotab/core/readings.py:945`

**Fix.** In turbotab/core/readings.py:976-982, replace the fixed MINOR_SHARE floor with a direct test of the 'absent' reading: add ('none', 0.0) to _factor_alternatives (readings.py:937) so the loop at :984-990 rejects it from the observed ratio. Equivalently, use the floor 1 - ratio/PASS_HIGH, and reword the refusal so it no longer says any unit 'would hold as well' when that is not computed.

**Evidence.** The ratio is declared/reconstructed (readings.py:983), so dropping a source raises it. Referee A's counterexamples (0.92 to 0.846, and 1.08 at 15% to 0.918) invert the ratio and are wrong. Referee B's point (the floor is 1 - R/1.10, exact only at R=1) is right. No wrong grams settlement follows: the kcal alternative removes 75-89% of a source's energy, so whenever it leaves the band the 'absent' reading does too. The only error is a needless refusal with a false reason when the median ratio exceeds 1. For example, at R=1.05 with an 8% share the 'absent' and kcal readings give 1.14 and 1.12, both out of band, yet the app says the identity 'would hold as well'. A refusal only waits for the user's recorded unit, and such low-share non-alcohol sources are rare, so S3. The audit's 'faithful/certified' is right for the main path. (confidence 0.65)

### C-142 · convention shown as a finding · random

**Claim.** A column named like an identifier is not a unit identifier (and clustering intervals by it is invalid) when it holds fewer than 3 distinct values, or at most 10 repeating word labels without digits (e.g., a case-control `patient` 0/1, `respondent` Self/Proxy, `person` mother/father); 3-10 repeating coded values may be a small study's units or a group code and are asked; more than 10 repeating values, or one value per row, is a unit structure.

**Where.** `turbotab/core/recognizers.py:1440`

**Fix.** In identifier_values (turbotab/core/recognizers.py:1440-1467), phrase the reasons as readings ('reads as a category's labels'), label ID_MIN_UNITS = 3 and FEW_UNITS = 10 as conventions, and send up to 10 repeating digit-free word labels to 'few' (asked) rather than 'never', so a small study that codes subjects by name is asked.

**Evidence.** The recognizer is a role heuristic. A 'never' verdict keeps the column as a low-confidence predictor the user can override (rows.py:507-511), and it neither grants nor refuses clustered intervals. The referees' main objection, that 11 to 30 clusters give anti-conservative CR1 intervals, does not apply: clustered rows get CR2 with Bell-McCaffrey df and a concern below 30 clusters (models/inference.py:11, 46, 644). What remains is that the 3/10 cuts and the digit-free-word rule are shown as facts ('a category's labels, not units') although they are conventions, and they can misread a small study whose subjects are named. (confidence 0.6)

### D-025 · design doc only (not built) · random

**Claim.** Under Predict with a period held out, R² on the opened period computed against that period's own mean is conventional and conservative, and the comparison with a deployable baseline is the RMSE against the development mean's RMSE.

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md §17 ruling 7 (line 1494)`

**Fix.** Amend docs/turbotab-next/calm/BEATS_PREDICT.md:1494 (§17 ruling 7) to match the engine and its own Table 2 (line 988-989): the opened cycle's R² is measured against the development (training) mean, per models/metrics.py and voice.py:1209-1211; if an own-mean R² is ever shown beside it, label each with its reference mean.

**Evidence.** Both referees prove the statement true (own-mean SST <= development-mean SST, so own-mean R² is lower and is sklearn's convention), but the audit and code show the engine does the opposite: R² against the training rows' mean (Staerk et al. 2024, audit MA-09), and test_modeling.py:200-225 asserts it is not the own-mean r2_score. The engine's choice is the deployable out-of-sample R² and is labeled in the record, so the only defect is a design ruling that contradicts the built behavior, with no user exposure. (confidence 0.8)

### D-056 · convention shown as a finding · random

**Claim.** When an intake screen would remove more than half the rows, the column's unit (kJ, a multi-day total, a child's or athlete's intake) is the likelier error rather than the people, so the screens refuse to remove more than half the rows until the unit is stated.

**Where.** `turbotab/core/stages/finding_words.py:475-491`

**Fix.** In turbotab/core/stages/finding_words.py:486-490 and turbotab/core/decisions.py:3647-3651, call the 50% line a convention ('a screen that removes most rows usually means a unit mismatch or bounds that do not fit the population'). Stop implying that stating the unit lifts the refusal, and describe a child's or athlete's intake as a bounds mismatch, not a unit error.

**Evidence.** Both referees find the guard sound, since a kJ column read as kcal or a weekly total shifts the whole column, and audit IN-07 records the failure it prevents. The 0.5 threshold and the claim that 'the unit is the likelier error' are unsourced heuristics. The refusal also still fires after the unit is recorded (_screens_keep_most_rows has no unit check), so 'say what unit ... first' misleads in the rare legitimate case. There the only exits are rescaled bounds or dropping the rule. (confidence 0.6)

### D-084 · design doc only (not built) · random

**Claim.** A prediction model that reads a predictor known only after its intended moment of use looks better in a paper than in use, so only predictors recorded at that moment are read (e.g., no 24-hour recall for a clinic use).

**Where.** `docs/turbotab-next/calm/BEATS_PREDICT.md M1 Why? (line 415)`

**Fix.** Keep BEATS_PREDICT ruling 13 (line 1213) as unbuilt P3 work. Before any card ships its 'Why?' text, implement q:moment_of_use and add a test that a predictor recorded after the moment is excluded, and phrase the rule as 'available at the moment of use' (PROBAST 2.3) rather than 'recorded at'.

**Evidence.** No engine or frontend code reads a moment of use: grep finds no q:moment_of_use outside the design docs, and WALKABLE_PLAN.md:48 and :143 list it as missing (P3), so the app says and does nothing here yet. The principle matches PROBAST 2.3 (Referee B). Referee A's objection about the 24-hour recall does not hold for the doc's actual scenario, a clinic fasting draw where no recall is taken (BEATS_PREDICT.md:418-421), though 'available at' is the better test than 'recorded at'. (confidence 0.8)

### D-089 · design doc only (not built) · random

**Claim.** Results wording starts at the design's floor, an association for an observational design; causal wording is offered only for trials, 'an estimated effect, its assumptions named' only when no standing question points elsewhere and no stated assumption is refuted, and model explanations are never worded as effects.

**Where.** `docs/turbotab-next/calm/FOUNDATION.md §8 (lines 456-480)`

**Fix.** Mark the claim-strength drafts and the stronger-wording noticing (FOUNDATION.md §8 lines 456-480; BEATS_MODELS_RESULTS.md:683) as planned under SIZING C7a in the ledger. Until then, have the observational option at teaching/content.py:411 say 'worded as associations, or as an estimated effect with its assumptions named in the causal lane'. When C7a lands, add a test that a refuted assumption (for example, an acknowledged positivity violation, core/causal.py:791-797) withholds the 'estimated effect' tier.

**Evidence.** Both referees find the three-tier wording sound, and I agree. But no engine or frontend code drafts Results wordings by claim strength or raises a noticing on stronger user wording (searches for 'assumptions named', 'Write my own' and 'claim strength' find docs only), so the 'refuses' action is unbuilt. What is built mostly agrees: causal wording only for a trial's ITT set, explanations labeled 'not an effect estimate'. One built path departs from the doc: the causal lane still writes 'The ... effect ... was estimated' after a positivity violation is acknowledged, although it names the violation. (confidence 0.7)


## Flagged, then cleared by the adjudicator (26)

A referee or the basis audit flagged each of these, and the adjudicator found the app correct. *Referee error* means the objection did not hold. *Inventory misstatement* means the inventory's statement was not the app's claim.

- **A-003** (inventory misstatement): The repo cites random search as 'a surprisingly strong baseline' only as reason 3 for picking plain Sobol over halving or TPE (RECIPES §4.2:769-772). It never claims that 8 or 16 candidates give settings that predict at least as well as the standard on new data. What the user sees ('searched inside every training fold, with the defaults always among the candidates') is true. Both referees' objections (no fresh-data guarantee, small losses against a regularized default) target the accuracy guarantee the inventory added. The only residue is that S is an unlabeled convention, as the audit says.
- **A-008** (inventory misstatement): The app's methods text (boosted_trees.py:287-293) only states the rule; it never says early stopping 'chooses well' or that searching is 'better' below 1,500, which is the inventory's gloss. The 100-unit rationale is exact: n_plan counts the rarest class or events (tuning.effective_size and plan_size), so 1,500 x 2/3 (K = 3) x 1/10 = 100 stratified stopping units. Both referees' arithmetic objections assumed total rows, so I reject them. Referee B's small log-loss gap (about 0.003 at n >= 1,500) is about a compute/accuracy trade-off the app does not claim to optimize.
- **A-011** (inventory misstatement): Both referees judged the claim true, and the code matches it: pooled_loss is sum(w*loss)/sum(w) over every inner row (tuning.py:998-1013), Loss admits only mse/log_loss/rps (tuning.py:88), and choose() takes the rounded argmin (tuning.py:1016). The only gap is the inventory's: a forest with one row per unit, no time order, the sample answer and no outcome-reading step pools its out-of-bag predictions instead, which is still a pooled strictly proper loss, so the app's real claim holds (audit's corrected statement accepted).
- **A-030** (referee error): corrected_t (validation.py:529-561) is Bouckaert & Frank's repeated k-fold form with n2/n1 = 1/(K-1) on rK-1 df, and both referees' simulations put its null size at about 3-7%, near nominal or conservative, against roughly 55% for the naive t. Referee A's 0.90 coverage needed OLS at p/n1 of 0.5-0.7, an extreme regime, so it does not overturn the default. Referee B's 'because BBC-CV' non sequitur attacks the inventory's wording: COMPARISONS_NOTE (validation.py:581) calls each pairwise interval descriptive and uncorrected, and says only that the choice of family is corrected by BBC-CV 'instead'.
- **A-068** (referee error): Both referees and the audit confirm that q = floor(sqrt(0.2p)) gives q^2/(0.2p) <= 1, and that Shah & Samworth carry the M&B bound over to complementary pairs under its assumptions. The inventory's objections fail on all three points. For p < 5, q = 1, so each half selects one column, the selection proportions sum to 1, at most one column can reach tau = 0.6, and V <= 1 holds deterministically. LARS on a 0/1 outcome is a valid base procedure. Lifting columns to terms cannot make false terms outnumber false columns. The only weakness is a test that cannot fail.
- **A-081** (inventory misstatement): The app never says the box 'contains the well-performing settings': users see only 'Its settings are searched inside every training fold, with XGBoost's defaults always among the candidates' (xgboost_family.py:513), and RECIPES calls the ranges 'conventions, narrowed'; the per-dimension source strings feed only the citation-registry check (base.py:687-702), not user text. Referee A is right that Probst's Table 3 puts tuned depth (5.6-14) and rounds (920-4,551) partly outside the box, but rounds are searched without early stopping only below an effective size of 1,500 (tuning.py:94), where lr up to 0.3 x 1,000 rounds is ample, and Referee B's equal-budget simulation found no loss.
- **A-084** (referee error): The docstring (forest.py:4-11) credits each value to its own package. The numeric set is randomForest's (mtry p/3, nodesize 5) and the class set is ranger's probability forest (sqrt p, min.node.size 10), and the leaf-versus-node departure is stated as 'stricter'. 'Acceptable' is the inventory's word, not the app's. The referees' 'hybrid' objection repeats that attribution, which the passing provenance test checks. Referee A's remaining nuance holds: randomForest's nodesize is documented as the minimum terminal node size but works as a split threshold. That changes fits only modestly, and no user decision depends on it.
- **A-107** (inventory misstatement): Referee A's only objection (the smallest-class shortcut is not Pate et al.'s multinomial criterion) does not apply: shelf_order calls prediction_minimum, which handles only 'regression' and 'binary' (sample_size.py:175-209), so a multiclass outcome gets no Riley minimum and the shelf says so; under inference shelf_order returns before computing one and rule_spline_terms returns 0. The parameter counting itself (k-1 per category, every spline/quintile column, k-2 per bendable predictor) matches Riley et al. 2020/2019 as Referee B and the audit found, and is checked against pmsampsize and hand counts in test_explore_repair.py::test_5 and test_wp8_inference_reporting.py::test_4a/4c.
- **A-113** (inventory misstatement): The app refuses only model_based and rubin_only. Both are too narrow, and the referees and the audit agree. The refusal says 'Its interval comes from a bootstrap over the whole chain', which describes the app's route without claiming it is the only valid one. No stacked sandwich is offered or refused, so no valid estimate is withheld, and the referees' 'must' objection hits only the inventory's wording. The PSU branch is Rao & Wu's rescaling bootstrap (n_h - 1 draws, weights times n_h/(n_h - 1); calibration.py:560-583), so two-PSU strata are not under-dispersed. BOOT_COPIES = 5 is sourced to Schomaker & Heumann. What remains is the misquoted 'RC'.
- **A-123** (inventory misstatement): The energy-model question is a Router Decide (quest.py:172) whose applicability (interview.py:278-299) depends only on the dietary lens, an energy-role column and an exposure, never on r; called directly it applies with no data at all. So below 0.3 the substitution-versus-addition choice is still asked, and only the extra noticing line is absent. The app's own docstring (materiality.py:652-654) says the meaning changes at any r, so the referees' point is right about the inventory's wording, but their bite ('suppresses the energy-model decision') does not happen.
- **A-135** (referee error): I accept the basis audit. set_multiplicity reaches results only through apply_multiplicity on the feature-wise table (stages/effects.py:894-900, stages/modeling.py:2597-2601), which tests one matrix column per exposure name, so one exposure gives one test and BH's q equals p. The referees' multi-level-contrast objection does not apply: the app never passes contrasts, spline terms or modifier tests to the multiplicity method.
- **B-069** (referee error): The app recommends the spline, marks the straight line only 'available' with the conditional 'Sound when the relation is close to a line; a curve it misses biases the slope' (exposure_form.py:1058-1061), and marks quintiles 'rank_lower' for the sourced reasons of lost within-fifth variation and a step at each cut (Bennette & Vickers 2012; Greenland 1995). So referee A's worry that a U-shaped response is steered to a straight line does not hold, because the spline that handles curvature leads and the line's text names its failure. Referee B and the audit are right that the ranking is a labeled ruling with sourced reasons.
- **B-086** (referee error): The code matches the claim. The refusal is at exposure_form.py:2138, the option is offered only under inference (:1101) with a recorded estimand-change sentence (:2069), and both referees and the audit accept the definitional part and the STROBE-nut nut-14 citation. Referee B objects that a consumers-only predictor for use among consumers is blocked. It is not: the refusal applies only to the exposure-form domain. A consumers-only population can be declared through the eligibility rules (set_exclusions accepts a range rule on a numeric exposure, decisions.py:3547), and then 'the model is used on everyone' in the declared population is exactly right.
- **B-270** (referee error): The ITT definition, its protection by randomization and its effect-of-assignment interpretation are correct and sourced (White et al. 2011; CONSORT 2010 E&E item 16, verified by the audit). The referees' missing-data caveat is already handled: an MAR primary on observed outcomes plus a delta tipping-point sensitivity analysis, disclosed in the notice. Their non-inferiority caveat does not bite on this app. The engine has no non-inferiority or equivalence analysis, and the per-protocol set is offered 'reported beside intention to treat, never instead', which is what the CONSORT NI extension asks for. Trial designs are also NOT_YET, so no user reaches this.
- **B-301** (referee error): The claim rests on the identity Var(mean of k days) = σ_b² + σ_w²/k > σ_b². It is sourced (Tooze 2006, verified by the audit) and tested on a known-truth simulation, so the mean-of-days tails are too wide and ranking that view lower is right. My simulation (N = 500, true prevalence 0.20) confirms referee A that the variance-components estimate is noisier than the mean of days with 10 or fewer repeaters: RMSE 0.30 vs 0.08 with 2, and 13-23% degenerate fits. It wins clearly from about 30 repeaters (0.06 vs 0.08), so referee B's 'better even with 2' did not hold in my run. The app already warns below 50 repeaters, so this narrow regime does not make the claim wrong.
- **B-349** (referee error): Referee B is right: testing the exposure's own cox.zph term is the relevant check for the exposure's hazard ratio, and the reading already labels 0.05 as 'the customary' level (stages/effects.py:1229-1240), so the convention is labeled. Referee A's objection, that a test tracks detectability rather than size, applies to any test-based diagnostic; here the failure only offers exits (period hazard ratios, keep labeled) and refuses nothing. The fit stage's 0.01/P (survival.py:518) answers a different question, a screen over every column with multiplicity control, and its silence is never shown as a pass, so the two thresholds do not contradict each other.
- **B-391** (referee error): Selection on a continuous outcome's value is textbook truncation bias, and the wp16 test shows it on a known-truth fixture. Referee A's main exception, cleaning recording errors, is handled by another route: the refusal offers 'Impossible or coded `y` values' repair exits (_outcome_repair_exits, decisions.py:3856-3888), so error cleaning is not blocked. Referee B's odds-ratio invariance applies to a binary outcome selected on Y alone, which is not a realistic exclusion-rule case.
- **BX-013** (referee error): Referee B and the audit derive the label. With b*f(x) nonlinear, each person's effect b[f(x +/- k) - f(x)] depends on k and on intake, so each point is an average. methods/substitution.py:substitution_curve averages over the rows on support at each k and stops when the on-support share drops below min_support, which answers referee A's feasibility point ('the rows the curve reads'). Referee A read 'all-components contrast' as any reallocation. In the app it means the all-components model's coefficient difference, which has no per-kcal meaning on logged components.
- **BX-067** (referee error): The shown sentence says what Schmidt, Le & Ilies 2003 say: single-occasion alpha and omega miss transient error and under-correct. The app claims only that a repeat administration 'includes transient error', which is true. Referee A objects only to ranking test-retest above omega. In regression calibration, though, the true score is the person's stable level on this instrument, so item-specific variance is not error, and repeat administrations are the standard basis for the correction. Ranking 'recommended' above 'available' is defensible, and Referee B and the audit's source check agree.
- **C-016** (inventory misstatement): The referees object that the 100 has no derivation, but seal.py:floor_for labels both the regression floor and the multiclass/ordinal floor 'A convention' on screen, and the floor only orders the options. The inventory turned that into the factual 'needs ... to be useful', which the app does not claim.
- **C-032** (inventory misstatement): The app's text (plan_previews.py:1518-1546; the design_cv.py docstring) says only that under the population answer the cross-validation follows the design and every score is weighted, and that under the sample answer the unweighted scores are 'the procedure's performance on these n rows, not the surveyed population's'. Both are faithful to WGM 2022 and the surveyCV construction. Referee A's objection (design-based CV is sufficient, not necessary, when weights are non-informative) targets the inventory's 'only if', which the app never states; the app applies the method by the user's estimand answer, which is defensible.
- **C-085** (referee error): Both referees' objections fail on the code. Rows with a blank outcome are dropped earlier, at the 'outcome_measured' step (rows.py:1362), and the exposure is among predictor_columns (readings.py:1491), so 'only predictors' is the right scope for this step. Under inference, blanks kept as a level are blocked as the missing-indicator method, with Groenwold et al. 2012's caution (decisions.py:4850-4870). They are kept only with an attestation that a blank means not asked or that the covariates are a trial's baseline ones, which is exactly where Groenwold calls the method valid. Under Predict a Missing level is sound.
- **C-102** (referee error): I accept referee B. An infinite cell is undefined and breaks every summary (audit MA-18: it crashed DuckDB's stddev), and blanking it is the minimal repair. The finding names the zero-denominator cause, the cell is marked 'infinite' (_infinite_marks), and the user still chooses the missing-values method. Referee A's record-level point (a 0 kcal day makes the whole day suspect) belongs to the dietary lens's implausible-intake screening, not to this repair, and -inf from log(0) is a rare input outside the claim's example.
- **C-141** (inventory misstatement): Both referees judge the claim true: injection order and analytical batch are run properties that values cannot identify, and a name-plus-structure guess settled by the user (KIND_RULES, the 'one curve over the whole run' option always offered) is the right design. The only discrepancies are the audit's two detail errors in the inventory ('else' vs 'then' for block candidates; n/4 vs max(2, n//4)), which I confirmed in batch_readings; the app's own docstring describes the code correctly, and the caps are interface conventions that cannot mislead because the user chooses.
- **C-150** (referee error): Referee A's only objection is randomized trials, where a baseline measured before randomization cannot be a mediator. Every trial design is refused in this version (designs.py:55-57), so the rule runs only on observational data. There, habitual diet measured alongside HbA1c, BP or LDL makes the baseline both a possible confounder and a possible mediator (Schisterman 2009; Glymour 2005 for change outcomes). The pair reports both estimates, so neither model is hidden.
- **C-204** (referee error): Both referees confirm the reason (a single completed draw treated as observed drops the between-imputation variance; Referee A's simulation shows 84-92% coverage at 30-50% censored), and the refusal routes to the censored-normal draw within the multiple imputation, a sound estimate, so nothing valid is withheld. The audit's only objection is unevenness with half-minimum, which is accepted at <=10% (Lubin's customary share) or above it only when recorded as a limitation; that concerns the half-minimum rule's wording, not the correctness of this refusal.
