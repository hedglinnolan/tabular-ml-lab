# Dietary threads (58)

Generated from catalogs/dietary.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- diet-day-to-day-variance (with diet-too-noisy-to-correct): 'About three-quarters of the spread in one day's energy is day-to-day noise. A two-day mean carries about 39% of the true slope on the log scale.' (dietary_recalls.csv). Before any model runs, the researcher learns that the headline coefficient will be attenuated by more than half, and declares regression calibration as the secondary analysis. In the same Strip, sodium and fiber sit at the bottom (day-1 vs day-2 r 0.01 and −0.11), so their nulls are labeled uninformative rather than corrected 50-fold.
- diet-zero-is-a-day with diet-mass-at-zero and diet-former-consumers-in-reference: 'Alcohol is zero on 31% of recall days, but only 9.7% of people are zero on both, about what chance alone predicts (9.5%).' (dietary_recalls.csv). The 'non-drinker' reference a score-driven pipeline would build is people who happened not to drink on two days. Alcohol goes to the two-part model. When a lifetime-use column exists, never-drinkers are split from former drinkers, who often quit because they got sick.
- diet-energy-carries-the-nutrient with diet-energy-related-outcome: 'fat_total moves with energy at r = 0.86. Unadjusted, the fat coefficient mostly measures how much people eat, and with BMI as the outcome, energy also lies on the path.' The researcher chooses between substitution at fixed energy and addition, and declares the with/without-energy pair. No score can make that choice.
- diet-misreporting-tracks-body-size with diet-implausible-reporters: 'Below the one-day Goldberg cut-off: 38% of people with BMI ≥ 35, against 14% under 25. Reported energy does not rise with BMI (r = −0.03).' (local NHANES extract, Schofield weight-only BMR). The 'eat less, weigh more' pattern comes from the measurement, not from diet. A Goldberg exclusion would remove mostly obese participants and select on a BMI outcome, so it becomes a sensitivity analysis, and recall-based RC is labeled as not fixing this bias.
- diet-energy-budget-gap: 'One day in ten has more energy than its protein, fat and carbohydrate can supply, and most of those days are men's. It is alcohol that was not exported as a column.' (local NHANES extract; the median check stays silent). Without it, a 'carbohydrate for fat at fixed energy' estimate silently moves alcohol too, and heavy drinkers are flagged as over-reporters.
- diet-food-and-its-nutrients: 'Saturated fat is largely computed from your red-meat and dairy columns.' The app's own adjustment default (covariate_guesses.py Model 4) would adjust red meat for the nutrients it supplies, which removes part of its effect. The card moves those nutrients out of the confounder lane and into a declared secondary model.
- diet-changed-because-of-disease: 'One in six follows a special or diabetic diet, and their sugar intake is much lower.' Their diet is a consequence of a diagnosis, so analyzed as a cause, sugar would look protective. The flag moves from the confounder lane to an eligibility sensitivity analysis, decided before any estimate. The flagged vs unflagged contrast reads predictors only, so it is allowed before the lock.
- diet-instrument-kind: 'These are two FFQs a year apart, not two recall days.' Their agreement measures reproducibility, not validity. A correction built on it leaves most of the attenuation in place, and an FFQ cannot give a share below the EAR. This one answer decides whether RC, NCI and the EAR share are enabled or refused.
- diet-healthy-lifestyle-cluster: 'Your top fiber quintile smokes half as often and is twice as likely to have a degree.' This Table 1 gradient reads predictors only, so it is shown before the lock. It leads the adjustment card and sets the E-value that accompanies the estimate, so a clean-looking association is read with the right distrust.
- diet-deployment-measurement-heterogeneity (prediction): 'You trained on the mean of two recalls, and the clinic will take one.' Adding one-recall noise to the test folds pushes the calibration slope below 1. The score a pipeline would report is for an instrument the deployment will not use, so the app re-scores at deployment noise and offers recalibration.

## Journey load

Typical journey: pooled NHANES-style data, two 24-hour recalls, inference, a BMI or diabetes outcome, an alcohol column, and micronutrients.

As the catalog was written, this journey had about 13 to 15 asked questions and 6 to 8 surfaced noticings, roughly 20 interactions before the model. After the critic's tiers and the card consolidation, it has 5 new asked cards, each with one question, plus 4 surfaced noticings, about 9 interactions. About 12 stated lines go into the Flow and the methods, and 4 threads run silently.

**Asked (5 new cards)**
1. **Instrument and grain:** the instrument is confirmed as two 24-hour recalls, with the evidence pre-filled. Food rows, household rows and meal rows do not fire. The substudy, the biomarker and supplement contents are asked only when present. NHANES DR1T* totals are stated as food-only.
2. **Who is in (Flow):** one question: which screens are primary and which are sensitivity analyzes. Day vs person is an option on the screen. Diet changed by disease and pregnancy are rows.
3. **Energy accounting and estimand:** substitution or addition, plus the declared with/without-energy pair for a BMI or diabetes outcome. The compositional reference is asked only for percent-energy columns. Alcohol's unit is asked only if it is recorded in drinks.
4. **Measurement error:** which intakes get regression calibration as the declared secondary. The card's Strip carries the too-noisy rows, λ intervals, the scale, season and the nuisance terms as stated lines.
5. **What a zero means: alcohol.** It covers episodic vs structural zeros, and the never/former split if ALQ is merged. At most two components per journey.

These existing cards absorb more questions at no new cost:
- **Adjustment card:** food-and-nutrients (only if foods are present) and season (only for a seasonal outcome).
- **Multiplicity card:** the nutrient-wide scan.
- **Units step:** the energy unit, which is silent here because the Atwater ratio is about 1.

**Surfaced (4)**
- The misreporting gradient, inside Who is in.
- The assessment-batch Strip for pooled cycles.
- The healthy-lifestyle gradient, on the adjustment card.
- The energy budget gap, only if no alcohol column closes it. In this journey it does not fire, so the fourth slot goes to too-noisy rows when they exist.

**Stated**
- The reliable-recall domain (DR1DRSTZ).
- The weight tier: WTDR2D, or the 4-year rule for 1999–2002.
- Nuisance terms, season coverage and the error scale.
- Nested fat parts and substitution support.
- EI:BMR refused as a covariate for a BMI outcome.
- DR1TNUMF routed out of the features.
- The meaningful increment.
- The invisible-exposure caption after the lock.

**Silent**
- Intake vs expenditure.
- The nutrient name test, including the alcohol identity.
- Median fill.
- Pattern guards.

**Prediction variant**
- The Measurement error question is replaced by the deployment-instrument question at intended use.
- The zero card offers frequency and amount features.
- The energy card states that the choice matters little to the score.
- The screen is applied in-fold only if deployment will screen too.
- About 4 asked and 3 surfaced.

**Rarer designs add one asked card at most:** an FFQ cohort (repeated-FFQ update rule, blanks), food-level files (sum to days), household surveys or CGM meal data. Each of these folds into the grain or intended-use question.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| diet-day-to-day-variance | asked | measurement | Day 1 against day 2: much of each nutrient's spread is day-to-day noise, and the share differs by nutrient |
| diet-too-noisy-to-correct | surfaced | measurement | A nutrient whose two recalls barely rank people, so no correction can rescue it |
| diet-usual-intake-distribution | asked | measurement | A share past a cut-off computed from one day is wrong, because one day's distribution is wider than usual intake |
| diet-dri-life-stage | asked | design | Participants span several DRI life-stage groups, including pregnancy, so one cut-off stands for several requirements |
| diet-mass-at-zero | asked | predictors | Many people report none: whether they form a non-consumer group depends on whether the zeros are people or days |
| diet-zero-is-a-day | asked | measurement | A zero is a day without the food, not necessarily a person who never eats it |
| diet-energy-unit-and-days | asked | measurement | Total energy is in kilojoules, or is a total over several days |
| diet-energy-budget-gap | surfaced | predictors | The energy budget does not close: a source (usually alcohol) sits inside total energy but in no column |
| diet-kcal-per-unit-of-each-source | asked | measurement | Each energy source's unit sets its calories: alcohol in drinks is not grams |
| diet-energy-is-intake-not-expenditure | silent | measurement | The 'Calories' column is what a device says was burned, not what was eaten |
| diet-name-is-not-a-nutrient | silent | predictors | A column named like a nutrient is a lab count, a body measure, or a duplicate |
| diet-energy-carries-the-nutrient | asked | predictors | Every nutrient rides on total energy, so its coefficient's meaning depends on how energy enters |
| diet-energy-related-outcome | asked | outcome | The outcome is itself energy-related (weight, BMI, diabetes), so energy may be on the path |
| diet-implausible-reporters | asked | measurement | Some energy reports are implausible for the person, and a screen that removes them can select on the outcome |
| diet-misreporting-tracks-body-size | surfaced | predictors | Under-reporting rises with body size, so reported energy falls where physiology says it should rise |
| diet-implausible-day-vs-person | asked | measurement | An implausible intake is one bad day, not necessarily one bad person |
| diet-nested-parts | stated | structure | Some nutrients are parts of others (saturated fat inside total fat), so a total cannot move while its parts stay put |
| diet-compositional-shares | asked | structure | Columns are shares of energy that sum to 100, so one must leave and every coefficient is a swap |
| diet-median-fill-breaks-the-residual | silent | predictors | A median fill of a nutrient would make its energy residual a mirror image of energy, so the app never uses one |
| diet-recall-nuisance-effects | stated | time | Day 2 can read lower than day 1 and weekends can differ: these are the instrument's effects, not diet change |
| diet-substitution-support | stated | predictors | A swap of k kcal moves some people off any diet that was observed |
| diet-ffq-recall-substudy | asked | measurement | An FFQ intake for everyone, and 24-hour recalls or a second instrument for the same nutrient in a subsample |
| diet-recovery-biomarker | asked | measurement | A recovery biomarker (doubly labeled water, urinary N, K, Na) can check self-report |
| diet-changed-because-of-disease | asked | time | Some participants changed their diet because of a diagnosis, so their diet is a consequence, not a cause |
| diet-former-consumers-in-reference | asked | measurement | The zero group mixes never-consumers with people who stopped, often because they got sick |
| diet-patterns | silent | predictors | Dietary patterns, when declared, are derived without the outcome and refit in each fold |
| diet-supplements-in-totals | asked | measurement | Nutrient totals include or omit dietary supplements, and supplement users form a separate population |
| diet-instrument-kind | asked | measurement | Which instrument measured the diet decides which errors it carries and which corrections and estimands are valid |
| diet-recall-completeness | stated | measurement | Some recalls were not done, not reliable, proxy-reported or atypical, or have no second day |
| diet-food-rows-sum-to-days | asked | structure | The rows are foods eaten within a recall day: sum them to days, and read a missing food row as zero |
| diet-ffq-blank-means-never | asked | measurement | A blank FFQ line usually means 'never', but a run of blanks is a skipped page |
| diet-ffq-frequency-not-a-scale | stated | measurement | FFQ frequency categories are consumption frequencies, not Likert responses and not amounts |
| diet-nutrient-equivalents | asked | measurement | Micronutrients come in equivalents (RAE, DFE, IU), and some conversions are not constants |
| diet-invisible-exposure | stated | measurement | A null or a low importance may be an exposure the instrument cannot see |
| diet-repeated-ffq-cumulative | asked | time | Repeated FFQs over follow-up: average up to each event, and stop updating at an intermediate diagnosis |
| diet-quality-index-is-a-density | stated | measurement | A diet-quality index is already an energy density and a formative score, and its population mean needs a ratio |
| diet-assessment-batch | surfaced | measurement | Intake shifts by survey cycle, site, interviewer or food-composition release, and only some of those shifts are artifacts |
| diet-season-of-assessment | stated | time | Recalls were collected across (or within) particular seasons, and seasonal foods and nutrients move with the calendar |
| diet-error-grows-with-intake | stated | measurement | Day-to-day error grows with intake, so correction must work on the scale where the error is additive, which is the scale the model uses |
| diet-few-replicate-persons | surfaced | design | Only a handful of people have a second recall, so λ itself is too uncertain to correct with |
| diet-food-and-its-nutrients | asked | structure | Nutrients are computed from the foods in the model, so adjusting a food for its own nutrients removes its effect |
| diet-healthy-lifestyle-cluster | surfaced | predictors | The exposure travels with a healthy lifestyle: who eats it differs in smoking, activity, education and supplements |
| diet-quantiles-sort-by-sex-and-size | stated | predictors | Quintiles of absolute intake sort people by sex and body size before they sort them by diet |
| diet-intake-as-outcome | surfaced | outcome | The outcome is a recalled intake, so its day-to-day noise caps every model and a one-day dichotomy misclassifies |
| diet-deployment-measurement-heterogeneity | asked | measurement | The model was trained on a two-day mean but will see one recall (or an FFQ) in use |
| diet-ratio-features-share-the-outcome | stated | structure | A derived diet feature divides by body weight, and the outcome is made of the same weight measurement |
| diet-recall-process-shortcuts | stated | predictors | Interview-process columns (foods named, respondent, language) predict health through frailty, not diet |
| diet-nutrient-wide-scan | asked | predictors | Sixty nutrient columns are a scan, not a hypothesis, and they behave like far fewer independent tests |
| diet-rare-events-sparse-categories | stated | outcome | Few events spread over intake categories leave some categories with almost none expected |
| diet-household-level-intake | asked | structure | The rows are households' food acquisition, not people's intake |
| diet-meal-level-rows | asked | structure | Rows are meals nested in people, each with its own response, so a random row split lets the model recognize the person |
| diet-dietary-weight-tier | stated | design | Dietary analyzes need the dietary weight that matches the days and cycles used, not the exam weight |
| diet-meaningful-increment | stated | measurement | An effect per one SD of an energy residual means nothing to a reader; per a serving or 5% of energy does |
| diet-single-baseline-long-follow-up | stated | time | One baseline diet assessment stands for decades of follow-up, and diet drifts |
| diet-delivered-already-adjusted | surfaced | predictors | A collaborator's file already holds energy-adjusted or density nutrients, which fail the nutrient test and must not be adjusted twice |
| diet-recall-day-before-the-draw † |  | time | The first 24-hour recall covers the day before the blood or urine draw, so the biomarker reads yesterday's meal |
| diet-spot-urine-estimated-excretion † |  | predictors | Sodium or potassium 'intake' is estimated from spot urine by an equation built on age, weight and creatinine |
| diet-borrowed-validity-coefficients † |  | measurement | An FFQ is the only measure, so correcting for measurement error means borrowing a validation study from another population |
