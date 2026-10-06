# Clinical threads (55)

Generated from catalogs/clinical.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- 1. A leak the score rewards (clin-predictor-after-prediction-time, measured on leaky_sepsis). 'abx_escalation_score agrees with sepsis on all 160 admissions. When was it recorded?' One tap, 'after sepsis was suspected', and the score settles from 1.00 to 0.81. Then: 'los_days doesn't separate at all (AUC 0.51), but it is also known only at discharge.' A pipeline that optimizes the score keeps the first column and never asks about the second.
- 2. Two labs, not two kinds of patients (clin-mixed-units with clin-site-differences, measured on clinical_labs). All 102 low glucose values (median 6.1) are SOUTH; NORTH's 186 have median 116.5. The predictors tell the sites apart perfectly (C = 1.00), and all of it is glucose's unit; without glucose, C = 0.53. Unconverted, the model learns 'low glucose, fewer readmissions', which is really a site difference.
- 3. Rows are visits, not people (clin-encounters-not-patients, structure measured on clinical_labs). '288 rows are 96 patients seen three times. Split by row, two of each patient's visits train the model that is scored on the third.' The score gap still needs a fixture with a patient-level signal.
- 4. The drug's reading, not the patient's (clin-treated-measurements, measured on NHANES). 'People on BP medication have the same SBP as diagnosed people who aren't treated (135.1 vs 135.3).' The transform player adds Tobin's constant and the treated hump moves right. The NHANES file is untracked, so a derived fixture must be committed.
- 5. The model would be reading the definition (clin-outcome-defined-by-predictors with clin-diagnosed-not-diseased; needs a fixture). The app shows the rule it found: 'diabetes = 1 exactly when hba1c ≥ 6.5 or glucose ≥ 126'. It then asks: 'Do you want to predict having diabetes, or being diagnosed with it?' About a third of US diabetes is undiagnosed (Menke 2015).
- 6. No follow-up means no forecast (clin-diagnostic-or-prognostic, from the existing follow-up answer). 'There is no follow-up here, so this model finds people who already have diabetes. Some of its best predictors (low sugar intake, metformin) are consequences of being diagnosed.' The paper's verb changes from 'predicts' to 'identifies'.
- 7. Your lowest values aren't missing (clin-censored-lab-results, measured on clinical_labs). 'One patient in five has a CRP below 0.3. Those aren't missing: they're your lowest values.' 55 of 288 rows; dropped, the median moves up. Above the pack's share, the handling becomes an asked slot instead of a silent substitution.
- 8. A state, not an event (clin-state-or-event, measured on clinical_longitudinal). 'Progression rises from 20% to 33% by visit, yet 67 of 200 people un-progressed. Is progression something a patient can recover from?' The answer chooses between survival, GEE or multi-state, and recurrent-event models.
- 9. Protection manufactured by the clock (clin-time-zero; needs a fixture). Each person's timeline is drawn with the 'immortal' segment before exposure lit. Users had to survive 2 years to count as users. Re-anchored at time zero after the lock, the advantage shrinks.
- 10. Sick, not erroneous (clin-impossible-vs-extreme, measured on clinical_labs). 'These 4 are impossible; these 26 are sick. A generic outlier rule would have removed all 30.'

## Journey load

Most threads are conditional, and about 8 reuse a question the engine already asks: follow-up, grain, intended use with threshold, the covariate after_exposure card, repeat_kind, levels, eligibility and the landmark. The critic's merges turn scattered questions into family cards:
- the opening 'Tell me about these columns' card, with a 'read from your data' block that settles the value-settled readings in one listed confirm;
- an outcome card;
- a follow-up card;
- a treatment card, one row per medication column;
- a design and estimand card;
- one shared known-date step detector;
- one heaping detector.

Load by journey (A = asked rows; S = stated or surfaced phrases):
- **NHANES cross-sectional, inference (diet → diabetes or BP):** about 4-5 new A rows on 3 cards (outcome definition and diagnosed-vs-disease; treatment, 1-3 rows; diagnosis-changed exposure; pregnancy on the eligibility step), plus the existing covariate card, which now names mediator paths. 7-9 S: fasting subsample and weight, impossible vs extreme, diagnostic verb, eGFR equation, pediatric z, season or BRINDA when those analytes exist, pre-filled expected directions, self-report mode. Acceptable.
- **Cohort with mortality linkage, inference:** 5-6 A rows on 4 cards (follow-up card: competing death and the existing follow-up question; outcome card: prevalent cases; time zero, only with a dated exposure; treatment; diagnosis-changed exposure), plus the existing covariate card. 4-6 S: age time scale, loss to follow-up, preclinical lag, interval censoring, index-event limitation. Acceptable.
- **Feeding trial:** 3-5 A rows on 2 cards (randomization, plus crossover if periods vary; 1-3 intercurrent-event rows). 1-3 S: regression to the mean when enrolment used a high lab, expected direction. Acceptable.
- **Admission prediction like leaky_sepsis:** 3-4 A rows (moment of use; the timing of 2 flagged columns; threshold). Light.
- **Multi-site EHR extract under prediction (clinical_labs-like): still the heaviest.** The critic counted 10-12 asked questions. With merges and caps it is about 8-9 A rows on 5 cards, all before modeling:
  - the opening card, with 4-5 rows: glucose unit at SOUTH; the vitals defaults as one listed template block (sbp 120, dbp 80, temp 98.6); hs_crp handling at 19%; the troponin verdict cut; one trajectory row listing 2 persons. Its read-from-data block lists the censoring tokens, 4 impossible SBPs and the 102 SOUTH values.
  - the existing grain question;
  - site role;
  - the follow-up card (recurrent event vs state, plus follow-up);
  - intended use with threshold and moment of use.

  Ordering stability is asked only if an indicator is kept, and the base model only if an established set is present. 4-6 S noticings come on top (informative ordering, calendar drift when dates span 3 years, derived variables, Firth).

  This is acceptable only because the card puts each row's evidence on screen and 'next slot' walks through them in order of consequence. Any further cut would have to come from the codebook settling units and defaults (§14.2.3).

Silent: digit preference and sparse levels never reach the canvas.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| clin-predictor-after-prediction-time | asked | time | A candidate predictor was recorded after the moment the model will be used, or after the outcome itself |
| clin-outcome-defined-by-predictors | asked | predictor-outcome | The outcome is defined from columns in the table, as a threshold of a measurement or a composite of components |
| clin-censored-lab-results | stated | measurement | Results written as '<0.20', '>1500', 'TNTC' or 'negative' are bounds or verdicts, not blanks and not numbers |
| clin-mixed-units | asked | measurement | One lab column holds two populations a conversion apart, and the low one is a single site or era |
| clin-default-value-entries | asked | measurement | Values pile up on a form's default (120/80 mmHg, 98.6 °F): entries, not measurements |
| clin-digit-preference | silent | measurement | Manual readings pile up on terminal digits 0 and 5, and just below diagnostic thresholds |
| clin-impossible-vs-extreme | stated | measurement | Impossible values are entry errors; extreme ones are the sickest patients |
| clin-implausible-trajectory | asked | time | A trajectory no body follows: an adult grows 9 cm, loses a third of body weight in three weeks, or has a record dated after death |
| clin-treated-measurements | asked | measurement | Treatment lowers the very measurement it was given for: antihypertensives on BP, statins on LDL, glucose-lowering drugs on glucose and HbA1c |
| clin-confounding-by-indication | asked | predictors | The exposure is a treatment given to the sicker people; in a prediction model, the treatment marks the indication |
| clin-informative-test-ordering | stated | structure | Whether a test was ordered carries information: a clinician suspected something |
| clin-coding-system-transition | stated | time | A change in coding system (ICD-9-CM to ICD-10-CM on 1 Oct 2015) shows up as a jump in recorded prevalence |
| clin-assay-change | asked | measurement | A lab's assay or calibration changed across cycles, sites or calendar time |
| clin-single-reading-dilution | stated | measurement | One reading or one blood draw is a noisy measure of a person's usual level (regression dilution) |
| clin-diagnosed-not-diseased | asked | outcome | A diagnosis-based outcome records being diagnosed, not having the disease, so undiagnosed cases sit among the controls |
| clin-fasting-status | stated | measurement | Fasting-dependent labs mix fasting and non-fasting draws, and the fasting subsample has its own weight |
| clin-unequal-follow-up | asked | outcome | A yes/no or count outcome sits beside a follow-up time that varies, so it is time to event, or a rate |
| clin-diagnostic-or-prognostic | stated | design | With no follow-up in the data, the model detects disease present now; it cannot forecast who will get it |
| clin-prevalent-cases-at-baseline | asked | time | Some rows already had the outcome when the clock started |
| clin-time-zero | asked | time | Exposure or eligibility is defined by what happened after follow-up began: immortal time and prevalent users |
| clin-competing-death | asked | outcome | Death from other causes removes people before the outcome can happen |
| clin-age-time-scale | stated | time | For a survey's mortality follow-up, age can be the clock instead of time on study |
| clin-loss-to-follow-up | stated | time | Who leaves the study early depends on who they are |
| clin-outcome-dependent-sampling | asked | design | The sample's outcome share was set by design (case-control, case-cohort, matched sets), not by the population |
| clin-randomized-arm | asked | design | Treatment was randomized, which changes what adjustment is for |
| clin-season-of-draw | stated | measurement | A biomarker swings with the season of collection, and in NHANES season is tied to latitude |
| clin-inflammation-adjustment | stated | measurement | Inflammation distorts micronutrient biomarkers: ferritin rises while retinol, zinc and iron fall |
| clin-physiological-state | asked | measurement | Pregnancy, kidney disease or acute illness changes what a biomarker means |
| clin-incremental-value | asked | design | The table already holds the established risk factors or an existing score, so the question is what the new markers add |
| clin-sparse-levels-separation | silent | predictors | A category has almost no rows or no events, so its estimate runs to infinity |
| clin-expected-directions | stated | predictor-outcome | The researcher states before the fit which way established factors push risk, and a reversed sign is explained rather than published |
| clin-state-or-event | asked | outcome | An outcome recorded at every visit can revert or recur: it is a state or a recurrent event, not a one-time event |
| clin-encounters-not-patients | asked | structure | Rows are encounters, and the same patient appears in many of them, or under two IDs |
| clin-site-differences | asked | structure | Rows come from several hospitals or clinics whose patients, practices and outcome rates differ |
| clin-calendar-drift | stated | time | Outcome rates and case mix drift across calendar years, so yesterday's calibration is not tomorrow's |
| clin-treatment-during-follow-up | asked | time | Treatment started during follow-up changes who has the outcome, so the prognosis being predicted is 'under current care' |
| clin-rare-outcome-threshold | asked | outcome | The outcome is rare, so accuracy and AUC hide what matters: how many flagged patients have it, at the threshold that would change care |
| clin-index-event-selection | stated | design | Everyone in the data already has a disease, so risk factors for getting it can look protective for what follows (index-event bias) |
| clin-biomarker-on-the-path | asked | predictors | A clinical measurement taken alongside the diet lies on the path from diet to disease (LDL, BP, HbA1c, BMI) |
| clin-diagnosis-changes-exposure | asked | predictors | People told they have a disease changed what they eat, so the diet looks protective (reverse causation through diagnosis) |
| clin-regression-to-the-mean | stated | design | People were enrolled because a lab was high, so it falls on re-measurement whatever the treatment |
| clin-crossover-design | asked | design | Each participant received every diet in sequence (a crossover), so the comparison is within person, and period and carryover matter |
| clin-intercurrent-events | asked | design | In a trial, rescue medication, stopping the diet or dropping out after randomization changes what the treatment effect means |
| clin-partial-verification | asked | structure | The reference test was done only for some people, and who got it depended on the screening result |
| clin-coded-history-absence | asked | structure | Hundreds of diagnosis and medication codes: no code does not mean no disease, and longer records hold more codes |
| clin-informative-visit-process | stated | time | In routine-care follow-up, sicker patients come back sooner and more often, so the visits themselves carry information |
| clin-derived-clinical-variables | stated | predictors | A column is a clinical formula or a cut-point of other columns (eGFR, Friedewald LDL, BMI, MAP, 'hypertension yes/no', a lab's H/L flag) |
| clin-contraindication-positivity | asked | predictors | Some patients could never have received the treatment (a contraindication), so they have no comparison |
| clin-pediatric-growth-scale | stated | measurement | Children's BMI and height mean different things at different ages and need age- and sex-specific z-scores |
| clin-event-found-at-visits | stated | outcome | An onset found only at scheduled exams is known only to lie between two visits (interval-censored) |
| clin-preclinical-disease-before-event | stated | time | The exposure changed because the disease had already begun: weight or cholesterol falls before death, cancer or dementia |
| clin-self-report-vs-measured | stated | measurement | Self-reported height, weight or BP sits beside or among measured values, and the two differ systematically |
| clin-population-specific-cutoffs | asked | measurement | A deficiency or anemia flag uses one cut-off for everyone, when the reference cut depends on sex, age, pregnancy, altitude or smoking |
| clin-specimen-quality-flags | stated | measurement | A hemolyzed, lipemic or icteric specimen still reports a number, and for some analytes the number is wrong |
| clin-lipid-carried-micronutrients † |  | predictors | Fat-soluble vitamins and carotenoids travel on lipoproteins, so their concentration tracks cholesterol and triglycerides |
