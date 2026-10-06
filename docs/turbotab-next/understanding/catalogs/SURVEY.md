# Survey threads (47)

Generated from catalogs/survey.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- 1. The outcome's condition hides in the questionnaire (survey-items-downstream-of-the-outcome, with survey-same-sitting-reverse-causation and survey-self-reported-diagnosis-outcome as one moment). The best predictors of diabetes are 'taking insulin now', whether 'taking diabetic pills' was asked, and whether 'age when first told' is blank. Each exists only because the person was already diagnosed, so a 0.97 AUC drops to an honest screening model (AUCs illustrative). In the same moment: 'told to cut salt' follows the diagnosis, and 'told by a doctor' counts the undiagnosed as healthy.
- 2. Blank because the gate said skip (survey-skip-pattern-blanks). meds_chol is blank for 17,204 people who were never told they had high cholesterol. They are 'not on treatment', and complete cases would keep 2,943 of 21,348 NHANES rows, almost all diagnosed (INBOX.md:86). One answer per questionnaire module fills every gated blank by rule.
- 3. The non-drinkers include sick quitters (survey-former-users-in-the-reference). A pipeline that codes everyone not drinking now as 0 builds the J-curve into every effect. The ever/current pair splits the zero group before any outcome is seen.
- 4. A 9 is a refusal, and a 5555 means 'more than 21' (survey-sentinel-codes, survey-top-codes). Measured on survey_sentinels.csv: item_14's 33 nines move its mean from 3.05 to 3.70, and on the reversed item_05 they bend its item–rest r from −0.58 to −0.37. Nothing is recoded until the codebook says what each code means.
- 5. The key comes from the instrument, never from the correlations (survey-reverse-keying). 8 of 40 items correlate −0.36 to −0.65 with the rest. Under the codebook's key, α goes from 0.80 to 0.94 and ω-total is 0.94 (measured with the engine). The app shows the pattern but never offers a flip, because the same pattern appears when an export has already reversed them.
- 6. The questionnaire changed, not the drinking (survey-instrument-changed-across-cycles). A step in heavy drinking at 2017–18, flat on either side, is the ALQ revision. Pooled as one variable, half the data answered a different question.
- 7. Fasting analytes need the fasting weight, and NHANES rows are not a simple random sample (survey-subsample-weight, survey-population-or-sample). Triglycerides exist only in the fasting subsample (WTSAF2YR). An interval 1.6× wider means 9,000 rows carry about 3,500 rows of information.
- 8. A strong construct looks like 40 weak features, and deployment may only collect a short form (survey-items-or-score). No item makes the top 10, yet the block costs the most AUC of any input when shuffled. A model trained on 9 items cannot be deployed where only the PHQ-2 is asked.
- 9. The total is two constructs, or one construct plus a wording factor (survey-dimensionality, survey-wording-method-factor). Parallel analysis and ECV say a 'stress' total adds worry and somatic symptoms, or that the second factor is exactly the negatively worded items. The published subscales are offered before any estimate exists.
- 10. Exclusion after the fact is a forking path (survey-careless-responding). 41 midpoint straightliners look identical whether they skimmed or are truly neutral; only their completion times tell. The rule that removes them is set before any result (illustrative).

## Journey load

Tiers in this catalog (47 threads): 8 silent, about 9 stated, about 11 surfaced-then-asked, and about 19 asked. Most of the asked ones are folded into six shared questions, so they are not separate prompts. The six:
(a) the scale-declaration form: instrument and version, unmodified?, reverse key, direction, construct type, missing-item rule;
(b) the codes question, one per code table: sentinel, don't-know and top codes; stated when a codebook labels them;
(c) one skip-gate Flow per questionnaire module, including OR-gates and age universes;
(d) the sampling question: population, sample, or nonprobability;
(e) the outcome's-condition moment: downstream items, same-sitting, self-reported diagnosis;
(f) reliability and correction under inference, or the deployment form under prediction.

Single-instrument journey (survey_instrument.csv, binary outcome, no design columns):
- Asked, about 4: the scale form (codebook key), the sampling question (opt-in or not), reliability and correction (inference) or deployment form (prediction), and the existing order question for education.
- Stated, 2: the item-MI line, because complete cases would lose 36 of 300 rows (12%); and score rather than items.
- Surfaced: 0.
- Silent, with no 'nothing fired' lines: dimensionality (eigenvalue ratio about 10), careless responding (longest longstring 11), floor and ceiling (0% of totals at a bound), small sample (7.5 respondents per item).

survey_sentinels.csv adds one codes question covering 7 = not applicable, 8 = don't know and 9 = refused, for 5 asked.

NHANES questionnaire-plus-exam journey (DPQ, DIQ outcome, diet and alcohol exposures, pooled cycles):
- Asked, about 8–10: the sampling question (exists); one skip-gate Flow for each module used (about 3–4, each a single confirmation when the codebook's skip instructions are loaded); the DPQ scale form, 1; the outcome's-condition moment, 1; the former-users reference, 1 if alcohol or smoking is the exposure; cycle harmonization, 1 if pooling across 2017–18; the wide-scan question only when there are more than 20 candidates.
- Surfaced, about 3–4: informative design (pre-lock, predictors only), the self-report/measured pair, the PHQ floor, and a DIF or mode difference only for the groups the plan names.
- Stated: the least-common-denominator weight, top codes from the codebook, effective n when deff_w > 2, and design df when q approaches d.
- Silent: domain estimation, imputation carrying the design, estimate-reliability greying, explanation weighting, and the common-method limitation.

That meets the critic's target of about 8 asked and 4 surfaced. The swing is the skip-gate Flows: without a loaded codebook, each module's gates need one answer.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| survey-reverse-keying | asked | predictors | An item runs against its scale. The instrument's key decides, and the key is checked again after it is declared |
| survey-sentinel-codes | asked | measurement | A 9 on a 1–5 item means 'refused', not strong agreement: values outside the response run are codes |
| survey-dont-know-meaning | asked | measurement | What a 'don't know' means depends on the question, not on who says it: a non-answer, a position, or a wrong answer |
| survey-top-codes | stated | measurement | A 5555 means 'more than 21', and an age of 80 means '80 and over': censoring codes, not amounts |
| survey-item-nonresponse | silent | measurement | A few skipped items should not cost the whole respondent, or be filled with their own average |
| survey-breakoff | surfaced | structure | Answers stop partway and never resume: respondents who left are not scattered skips |
| survey-skip-pattern-blanks | asked | structure | Blanks follow a gate question by design: 'No' to ever drinking skips 'drinks per day' |
| survey-check-all-that-apply | stated | structure | In a 'check all that apply' question, every unticked box is blank. Blank means no, unless the whole question is blank |
| survey-careless-responding | asked | measurement | Some respondents answered without reading. The rule to remove them comes before any result |
| survey-acquiescence | surfaced | measurement | Some respondents agree with everything, opposite statements included: a response style masquerading as the trait |
| survey-dimensionality | asked | predictors | A summed scale is one number only if its items measure one thing |
| survey-wording-method-factor | asked | structure | A second factor made only of the reverse-worded items is a wording effect, not a second construct |
| survey-reliability-attenuation | asked | measurement | A score's reliability in this sample sets how much its coefficient is diluted and how much any model can explain. The correction depends on purpose |
| survey-formative-or-reflective | stated | measurement | The score is an index defined by its components (a diet-quality score), not a reflective scale |
| survey-repeat-or-reference-measurement | asked | measurement | The construct was measured twice, or against a reference: a retest, a substudy, or self-report beside a measurement |
| survey-small-sample-psychometrics | silent | predictors | In a small sample, the scale's structure and reliability cannot be re-estimated, so the published ones are the evidence |
| survey-instrument-published-scoring | asked | structure | A recognized instrument brings its published scoring, missing-item rule, cut-points and clinically important difference, but only if it was used unmodified |
| survey-score-direction | stated | measurement | Whether a higher score means worse or better decides what every coefficient's sign says |
| survey-dif | surfaced | measurement | An item works differently by sex, language, mode or occasion at the same trait level (differential item functioning) |
| survey-floor-ceiling | surfaced | measurement | Many respondents sit at the scale's minimum or maximum: the instrument runs out of room there |
| survey-ordinal-outcome | asked | outcome | An outcome of a few ordered answers is not a number: it takes an ordinal model |
| survey-ordinal-predictor-spacing | asked | predictors | A predictor coded 1–5 (education, a single Likert item) is a set of ordered categories, not equally spaced amounts |
| survey-heaped-answers | stated | measurement | Reported amounts pile onto round numbers (10 and 20 cigarettes, 7 and 8 hours, 30 minutes): rounding, not real mass |
| survey-items-or-score | stated | predictors | Forty collinear items are one construct: the score for inference, score against items compared by resampling for prediction, and explained as a group |
| survey-item-overlap-with-outcome | stated | structure | The exposure scale and the outcome share items or content |
| survey-common-method | silent | design | Exposure and outcome come from the same self-report at the same sitting |
| survey-mode-proxy-nonequivalence | surfaced | measurement | Answers given by phone, online, in another language or by a proxy are not the same measurement |
| survey-interviewer-clustering | stated | structure | Answers cluster by interviewer, so intervals are too narrow and new interviewers are a new domain |
| survey-population-or-sample | asked | design | Is the estimate about a surveyed population, or about these participants? How were they sampled? |
| survey-nonprobability-sample | asked | design | An opt-in online sample has no design to weight by, so its prevalences describe the volunteers |
| survey-informative-design | surfaced | predictor-outcome | The variables that drove sampling (oversampled groups, exam season, cycle) also drive the data, so weighted and unweighted answers can disagree |
| survey-subsample-weight | stated | design | Variables measured on a subsample need that subsample's weight: the smallest sample your variables come from sets it |
| survey-exclusion-is-a-domain | silent | design | Under a survey design, an exclusion is a domain, not a deletion |
| survey-weight-effective-n | silent | design | Unequal weights shrink the sample: the model's complexity budget is the effective n (and the effective events), not the row count |
| survey-design-df-budget | silent | design | The variance has only as many degrees of freedom as PSUs minus strata, which caps how much the model can ask |
| survey-estimate-reliability | silent | design | Some subgroup estimates rest on too few effective participants to publish |
| survey-explanations-describe-the-sample | silent | predictor-outcome | Under the population answer, importance and effect curves averaged over unweighted rows describe the oversampled sample, not the population |
| survey-instrument-changed-across-cycles | surfaced | measurement | Pooled cycles or waves asked a question differently, under the same name or a renamed one |
| survey-question-wide-scan | asked | predictors | Hundreds of questionnaire variables screened as exposures: multiplicity, correlated exposures, and replication across cycles |
| shared-panel-attrition | surfaced | time | Respondents leave a panel between waves, and who leaves depends on what they said before |
| shared-ema-within-between | asked | structure | Many short questionnaires per person (diaries, EMA): a scale has a within-person meaning and a between-person meaning |
| survey-items-downstream-of-the-outcome | asked | structure | A question asked because of the outcome (told by a doctor, taking medicine for it, advised to change) hands the model the answer |
| survey-same-sitting-reverse-causation | asked | design | Exposure and outcome were reported at the same sitting, and people change what they eat after a diagnosis |
| survey-former-users-in-the-reference | asked | predictors | The 'non-drinkers' include people who quit because they got sick |
| survey-self-reported-diagnosis-outcome | asked | outcome | 'Ever told by a doctor' measures diagnosis, not disease, and counts the undiagnosed as healthy |
| survey-imputation-carries-the-design | silent | design | Under a survey design, the imputation model has to know the design too |
| survey-recall-period-mismatch | stated | measurement | Items in one analysis ask about different windows: past 30 days, past 12 months, 'usually', 'ever' |
