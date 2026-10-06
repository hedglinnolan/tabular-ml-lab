# Shared threads (119)

Generated from catalogs/shared.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- shared-unit-repeats: '3,000 rows are 750 people. Split by row the AUC is 0.86; split by person, 0.71. The first number was remembering people.' A score-driven pipeline picks the leak, because the leak scores higher.
- shared-predictor-after-the-index with shared-outcome-proxy (one single-column-skill detector): 'Albumin was drawn after diagnosis for 312 patients; without it the AUC falls from 0.91 to 0.78.' And: 'diabetes equals hba1c >= 6.5 in 98% of rows: you would be predicting the definition.' Resampling cannot catch either; only time and meaning can.
- shared-skip-pattern-structural-zero: 'ALQ130 is blank for exactly the 1,214 people who said No to ALQ111. They were not asked because they do not drink, so drinks per day is 0.' MI would invent drinking for abstainers, and complete cases would drop every never-drinker.
- shared-survey-population-design with shared-survey-which-weight: 'Weighted, 31% meet the guideline; unweighted, 22%.' And 'fasting glucose needs WTSAF2YR divided by the 3 cycles pooled; the exam weight describes people who were never fasted.'
- shared-units-by-magnitude: 'Energy sits at 4.18 times the macronutrients' kcal: kilojoules. Read as kcal, the usual screen would have removed 61% of your participants.'
- shared-imputed-copies: 'Each participant appears 5 times with the same age and sex but a different body fat. These are NHANES's imputations, not visits; counted as people, your intervals are 2.2 times too narrow.'
- shared-time-zero: 'Supplement users started a median 3.1 years after enrollment and had to survive those years to become users.' Immortal time makes almost any later-starting exposure look protective, and no score can see it.
- shared-missingness-mechanism with shared-informative-missingness: 'Complete cases keep 62% of rows; the 38% who leave are 7 years older and twice as likely to be poor.' And 'lactate's blank means the doctor wasn't worried (missingness predicted at AUC 0.81): that is information about the doctor.'
- shared-process-aligned-with-outcome with shared-omics-sample-total-tracks-outcome (prediction): 'Cases' libraries are 1.46x deeper. Raw counts: AUC 0.94. After TMM log-CPM in each fold: 0.52. There was no biology here; there was depth.'
- shared-grouping-above-the-person: 'Leaving each clinic out in turn, C ranges from 0.62 to 0.86. The pooled 0.81 hides clinic 4, and your readers work at new clinics.'

## Journey load

Final catalog: 91 threads (86 reviewed by the critic, plus 5 added from its gap list). Tiers: 53 asked, 24 stated, 10 surfaced, 4 silent. Most asked threads are conditional (survival, case-control, omics, trials, time-varying data) and never fire on a given file.

Typical journeys, with batching applied:
- NHANES-like cross-sectional file, inference: about 8-10 asks. These are the lens, one batched readings table (codes or amounts, missing-value codes, units, sex coding), the survey population, the skip-pattern gates, the missingness mechanism, the outcome scale, one adjustment card (unknown-timing covariates, treated values, prevalent disease, absent confounder classes), and possibly compositional or derived columns. Plus 4-6 surfaced and 15-20 stated lines.
- Clinical cohort, prediction: about 9-12 asks. These are grain, grouping and new-site use, the batched readings table, dates, predictor-after-index flags, the outcome proxy, informative missingness, follow-up and competing death, and the intended-use or case-mix question asked once. Plus 5-8 surfaced.
- Without batching the same files carry 15-25 asks.

Two places run heavy:
1. Opening: per-column readings can pass 20 prompts on an NHANES merge without a codebook. Rule: one batched table with proposed defaults; ask per column only for columns in the declared analysis; a codebook settles the rest in bulk.
2. The inference adjustment card: mediator-or-confounder, treated values, prevalent disease, missing confounder, time zero and selection on a consequence all land there. Rule: order items by rank_signal; ask selection-on-a-consequence once in the design interview instead of on the card; cap surfaced noticings at 3 per stage.

Several reads are now stated with undo instead of asked: imbalance (no correction is the default), the transposed table (when feature-ID grammar matches), identifiers, non-data rows and number formats. Repairs with only one possible reading are silent and appear in the repairs log.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| shared-unit-repeats | asked | structure | Rows repeat within a person or sample, so a row is not a unit |
| shared-grouping-above-the-person | asked | structure | Participants share a site, clinic, school, household, assessor or interviewer, and measurements and performance vary across them |
| shared-exposure-varies-between-groups | asked | predictors | The exposure varies mostly between groups, not within them |
| shared-repeats-or-time-points | asked | time | A unit's rows are replicates of one quantity, or time points of a changing one |
| shared-imputed-copies | stated | structure | Each person appears m times: the data's own imputed copies (NHANES DXA _MULT_), not visits |
| shared-one-value-per-unit | asked | structure | A column holds one value per person across that person's rows |
| shared-identifier-as-predictor | stated | structure | A column that names rows (an ID, a saved row index) would enter as a predictor or encode file order |
| shared-duplicate-records | asked | structure | The same record or person appears more than once, sometimes under different IDs |
| shared-join-changes-the-row | stated | structure | Joining a file pairs one row with many, which changes what a row is |
| shared-linkage-quality | surfaced | structure | Joining files dropped or duplicated people non-randomly |
| shared-codes-or-amounts | asked | measurement | Whole numbers with few values may be category codes or amounts |
| shared-missing-value-codes | asked | measurement | 999, -9, 7777 or 7/9 hide in a numeric column as if they were values (refused, don't know, not applicable) |
| shared-sas-transport-zeros | silent | measurement | Zeros arrived as 5.4e-79 from a SAS transport file |
| shared-units-by-magnitude | asked | measurement | A unit is known only from magnitude (kJ or kcal, lb or kg, cm or inches, months or years) |
| shared-mixed-units | asked | measurement | One column mixes two units, for example glucose in mg/dL and mmol/L |
| shared-codebook-contradicted | asked | measurement | The codebook says one thing and the values another |
| shared-sex-coding | asked | measurement | Which level of a numeric sex column is female |
| shared-children-among-adults | stated | measurement | Some rows are children, whom adult limits misjudge |
| shared-impossible-vs-extreme | stated | measurement | Some values are impossible (decimal slips, wrong unit), while others are extreme but real |
| shared-heaping-digit-preference | surfaced | measurement | Values heap at round numbers: terminal-digit preference in blood pressure, whole-kilogram self-reported weight, ages ending in 0 or 5 |
| shared-left-censoring | asked | measurement | Values below a detection limit are censored, not missing at random ('<5' in labs; low-abundance mass-spectrometry features) |
| shared-ambiguous-dates | asked | time | Dates read both month-first and day-first |
| shared-skip-pattern-structural-zero | asked | structure | A blank follows a 'No' at a gate question: it means zero or 'not asked', not unknown |
| shared-informative-missingness | asked | structure | A blank means the test was not ordered, so missingness tracks someone's suspicion |
| shared-missingness-mechanism | asked | predictors | Who is missing, what goes missing together, and how complete cases differ from everyone |
| shared-missing-in-a-derived-term | silent | predictors | Missing values fall in a column the analysis will log, spline, residualize or sum |
| shared-flags-describe-the-data | stated | structure | An imputed_* column marks where a value was filled earlier, not a fact about the person |
| shared-quality-differs-by-group | surfaced | predictors | Data quality (missingness, sentinel codes, self-report, heaping) differs across sex, race or ethnicity, age or income groups |
| shared-survey-population-design | asked | design | Survey weights, strata and PSUs are present: whose population do the numbers describe? |
| shared-survey-which-weight | stated | design | The variables come from a subsample, or from pooled cycles, and that decides the weight |
| shared-lonely-psu | silent | design | After exclusions a stratum keeps one PSU, and the design has few degrees of freedom |
| shared-case-control-sampling | asked | design | Cases were sampled separately from controls, possibly in matched sets, so probabilities reflect the sampling ratio, not risk |
| shared-selection-flow | surfaced | structure | Who leaves at each step (eligibility, no sample, failed QC, unlinked) differs from who stays |
| shared-outcome-ascertainment | asked | design | The outcome could only be detected in people who were tested, assessed or coded, and that opportunity differs across groups |
| shared-case-mix-shift | surfaced | time | Predictors or the outcome shift over calendar time, or between the development and evaluation data |
| shared-method-change | asked | measurement | An assay, instrument, food-composition database or questionnaire version changed at a date or between sites |
| shared-season | surfaced | time | Intake or a biomarker varies by season, and season is unevenly spread across groups |
| shared-preanalytical-handling | asked | measurement | Fasting state, time of draw, storage time, freeze-thaw, hemolysis or RNA integrity shift measurements, and differ by group |
| shared-predictor-after-the-index | asked | time | A predictor was recorded after the outcome, or after the moment the model would be used |
| shared-mediator-or-confounder | asked | design | A covariate measured at the same visit as the exposure may be a confounder or a mediator |
| shared-time-varying-feedback | asked | time | The exposure changes between visits, and earlier exposure changes later confounders |
| shared-follow-up-varies | asked | outcome | A yes/no outcome sits beside a follow-up time that varies |
| shared-competing-death | asked | outcome | Participants can die before the event of interest |
| shared-outcome-scale-skew | asked | outcome | A positive outcome is markedly skewed, so its scale is a choice |
| shared-ordered-or-multiclass-outcome | asked | outcome | A text outcome with 3-10 levels: ordered or not |
| shared-events-not-rows | stated | outcome | The number of events (or rows, for a continuous outcome), not the file size, sets how much the data can say |
| shared-common-outcome-measure | stated | outcome | The event is common, so odds ratios stop reading as risk ratios |
| shared-predictors-outnumber-rows | stated | predictors | More candidate predictors than rows: selection outside the folds fakes skill |
| shared-near-constant-predictors | silent | predictors | A predictor barely varies, and a default filter would drop it |
| shared-collinear-predictors | surfaced | predictors | Two or more predictors carry nearly the same information (waist and BMI, fat and energy, adducts of one metabolite) |
| shared-derived-predictors | asked | predictors | Some predictors are exact functions of others (BMI from weight and height, eGFR, HOMA-IR, energy from macronutrients, nutrient densities) |
| shared-outcome-proxy | asked | predictor-outcome | A predictor nearly is the outcome: another measure of it, its definition, a function of it, or its consequence |
| shared-sparse-levels | stated | predictors | A category has too few rows to estimate, or is missing from some folds |
| shared-tails-and-support | stated | predictors | A predictor's tails are thin, so the far ends of effect curves rest on a few people |
| shared-interaction-support | stated | predictors | A planned modifier's levels do not share the exposure's range |
| shared-exposure-family | stated | predictors | Many exposures (or outcomes) are tested together: nutrients, metabolites, genes |
| shared-replicate-reliability | asked | measurement | Replicate readings (bp_1, bp_2, bp_3; duplicate assays) reveal regression dilution |
| shared-positivity-overlap | asked | predictors | The covariates nearly decide who is exposed |
| shared-curvature-seen-in-explore | stated | predictor-outcome | The outcome bends against a continuous predictor, and choosing a form by eye moves optimism outside the score |
| shared-subgroup-seen-in-explore | stated | predictor-outcome | The exposure's relationship looks different in one subgroup |
| shared-model-no-better-than-baseline | stated | predictor-outcome | A fitted family does not beat the no-predictor baseline |
| shared-selection-unstable | stated | predictors | The selected predictors change from fold to fold |
| shared-primary-model-diagnostics | asked | predictor-outcome | The primary model's assumptions fail on these data (proportional hazards, influence, proportional odds) |
| shared-transposed-assay-table | stated | structure | The table holds one row per feature, not per sample |
| shared-lens-contradicts-table | asked | structure | The chosen lens does not describe the table |
| shared-omics-value-scale | asked | measurement | What the numbers in an assay block are: raw counts, intensities, CPM/TPM, or already logged |
| shared-omics-sample-total-tracks-outcome | surfaced | predictor-outcome | Library size (or urine dilution) differs between cases and controls |
| shared-omics-outlier-sample | asked | predictors | One sample drives correlations or sits far from the rest |
| shared-non-data-rows | stated | structure | Some rows are not participants: a repeated header, a 'Total' or 'Mean' row, a footnote |
| shared-number-format | stated | measurement | Numbers use decimal commas or thousands separators, so they parse as text or as the wrong magnitude |
| shared-top-coded | stated | measurement | A value at the top means 'this or more': public-use top codes, instrument ceilings, '>x' upper limits |
| shared-coarsened-copy | asked | predictors | A column is a binned copy of another (bmi_cat beside bmi, age_group beside age), or the outcome is a cut of a recorded measure |
| shared-compositional-parts | asked | predictors | Several columns are parts of a whole: time-use summing to 24 hours, macronutrients summing to 100% of energy, cell or microbial proportions |
| shared-zero-mass-predictor | stated | predictors | A predictor is zero for many people and continuous above zero (pack-years in never smokers, supplement dose in non-users) |
| shared-zero-mass-outcome | asked | outcome | The outcome is zero for many people and skewed above zero (costs, alcohol grams, days of activity) |
| shared-count-outcome-exposure-time | asked | outcome | The outcome is a count over an observation time that varies between rows |
| shared-bounded-outcome | asked | outcome | The outcome is bounded (a proportion, a 0-100 score) and piles up at a bound |
| shared-baseline-outcome-present | asked | time | The outcome was also measured at baseline |
| shared-treated-values | asked | measurement | Some participants' values are lowered by treatment (antihypertensives on blood pressure, statins on LDL) |
| shared-treatment-paradox | asked | design | Treatment started during follow-up because of high risk, so high-risk features look protective |
| shared-prevalent-disease-reverse-causation | asked | design | Participants already knew they had the outcome, and may have changed the exposure because of it |
| shared-time-zero | asked | time | Exposure is defined by something that happens after follow-up starts (immortal time) |
| shared-selection-on-a-consequence | asked | design | Everyone in the file was selected for something the exposure or outcome causes (hospitalized, diagnosed, survived to enrollment) |
| shared-missing-confounder | stated | design | A canonical confounder for this exposure and outcome is not in the file |
| shared-validation-subsample | asked | measurement | A better measure of the exposure exists for a subset (a biomarker, a weighed record, doubly labeled water) |
| shared-process-aligned-with-outcome | asked | structure | A processing variable (batch, plate, run, sequencing date, assay lot, interviewer) shifts the measurements and may line up with the outcome |
| shared-randomized-arm | asked | design | An arm column was assigned by randomization: the data are a trial |
| shared-informative-dropout | surfaced | time | Participants lost to follow-up differ at baseline from those who stayed |
| shared-spatial-dependence | asked | structure | Rows are located in space (county, tract, coordinates), and neighbors resemble each other |
| shared-entry-on-a-high-reading | asked | design | Eligibility required a high reading, so the follow-up regresses to the mean |
| shared-proxy-respondent | surfaced | measurement | Some answers came from a proxy (a parent, a spouse), not the participant |
| shared-score-fitted-on-these-rows † |  | predictors | A column is a score, component or cluster label fitted on these same rows, possibly using the outcome |
| shared-target-population-shift † |  | design | The participants differ from the population the results or the model are meant for |
| shared-overfit-calibration-slope † |  | predictor-outcome | The model's risks are too extreme: the calibration slope is below 1 |
| shared-decisions-after-viewing † |  | predictor-outcome | Some decisions were made after the outcome was seen, and the methods should say which |
| shared-winner-optimism † |  | predictor-outcome | Many configurations were compared, so the best score is optimistic simply for being the best |
| shared-rashomon-disagreement † |  | predictor-outcome | Several models fit equally well but rank the predictors differently |
| shared-learned-interaction † |  | predictor-outcome | The fitted model leans on an interaction that the average curve hides |
| shared-category-spellings † |  | predictors | One category is spelled several ways ('Male', 'male', 'M'), so one level becomes several |
| shared-which-level-is-the-event † |  | outcome | For a two-level text outcome, which level is the event is a choice, not a default |
| shared-holdout-too-small † |  | design | The sealed holdout would hold too few events to give an honest score |
| shared-free-text-column † |  | predictors | A text column is free text, a category written as text, or a note written after the moment of use |
| shared-outcome-assessed-knowing-the-arm † |  | design | In an unblinded trial, the outcome is self-reported or judged by someone who knows the arm |
| shared-cluster-randomized † |  | design | The arm was assigned to schools, clinics or villages, not to people |
| shared-unit-summary-spans-the-future † |  | time | In a long table, a per-person column summarizes all of the person's visits, future ones included |
| shared-id-unique-only-within-a-site † |  | structure | An identifier is unique only within a site, cycle or file, so two people share one ID |
| shared-rows-are-follow-up-intervals † |  | structure | Rows are follow-up intervals (start, stop, event) with time-varying covariates, not people or visits |
| shared-row-stands-for-many † |  | structure | A row stands for many people: frequency weights, aggregated counts, or group means |
| shared-missing-by-design-block † |  | structure | A column is missing for whole sites, cycles or waves, or outside a measured subsample: not collected there |
| shared-rows-filtered-before-upload † |  | design | Rows were removed before the file reached the app, by a rule nobody stated |
| shared-sentinel-dates † |  | time | Placeholder dates (1900-01-01, the Excel or Unix zero, 9999-12-31) stand for missing |
| shared-ordered-text-predictor † |  | predictors | A text predictor's levels are ordered, but the alphabet scrambles their order and picks the reference |
| shared-mismeasured-confounder † |  | measurement | A key confounder is measured so poorly that adjusting for it leaves most of its confounding |
| shared-exposure-assigned-or-predicted † |  | measurement | The exposure is assigned from a group or predicted by a model (Berkson-type error), not measured with noise |
| shared-many-versions-of-the-exposure † |  | predictors | The table holds the exposure several ways, inviting a choice after the results are seen |
| shared-null-is-inconclusive † |  | predictor-outcome | A null result's interval still includes the smallest effect that matters |
| shared-sensitive-attribute-role † |  | predictors | Race/ethnicity, sex/gender or SES columns: a confounder proxy, a fairness group, or a predictor? |
| shared-unequal-blocks † |  | structure | A small clinical block sits beside thousands of omics features, and one penalty treats them alike |
| shared-identifiable-values-in-the-export † |  | design | Exports would publish identifiers, an individual's values, or cells small enough to identify someone |
