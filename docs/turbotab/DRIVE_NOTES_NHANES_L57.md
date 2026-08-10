# TurboTab happy-path drive notes (live)

File: nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv — 21,849 rows x 29 cols. Branch: TurboTab.

## STEP: Data (post-upload)
- Load receipt: "nhanes_..._imputed.csv was loaded: 21849 rows, 29 columns." + `change`. State: DATA=recorded. SURFACED-CORRECT.
- Card "noticed · 9 features need the same repair: read as binary" — imputed_bmi, imputed_bp_di, imputed_bp_sys, imputed_height, imputed_waist, imputed_weight, ... buttons [Show me what changes][Decide later]. (bulk-repair / audit-binary-text). NOTE: prereg says these 6 imputed_* are bool dtype, not text — message wording says "written as text" but they are bool. Also gender/meds_hbp/meds_chol are the text ones. Card says 9.
- Card "noticed · 9 things stand out in the shape of this file." Ranked; nothing applied. Affordance "First rows". (findings stack; on this file stack pushes 5 collapses 0 per prereg, so no "N more" affordance — need to verify count shown.)

### ORDERING FINDING (Condition 3 / wrong-order)
- OPENING_SEQUENCE mandates Q1 = the LENS ("What kind of measurements...") FIRST, BEFORE the structural diagnosis, because diagnosis is field-sensitive.
- In the running app: diagnosis (binary-repair + shape findings) runs on the DATA step immediately on upload with NO lens set. The lens question appears LOWER, under the TARGET step, AFTER the target picker.
- Both the target picker AND the lens question are labeled "01" under TARGET (duplicate step number, confusing).
- So actual order = upload -> diagnosis -> [Target step: target picker, then lens]. Intended = lens -> diagnosis -> target. WRONG-ORDER + lens mis-placed into Target step.
- Q1.5 (orientation) correctly absent (sample-major table, not assay). EXPECTED.

## STEP: Target
- Lens question "01 What kind of measurements are in this table?" multi-select. Picked Dietary intake + Clinical. Rich "Why we ask" tooltips per option (detailed, correct). Commit button counts: "Pick at least one" -> "Record this one" -> "Record these 2". SURFACED-CORRECT.
- Lens receipt (MEASUREMENTS): "The measurements were described as dietary intake and clinical measurements and labs. Domain conventions for these fields informed the defaults offered below; each is stated with its reasoning and was open to being overridden." SURFACED-CORRECT.
- Target picker "01 What are you predicting?" — all 29 columns as chips w/ dtype. Note SEQN offered as a target chip UNMARKED (prereg #1: not recognized as identifier). Picked glucose.
- Task-type receipt: "glucose is the target. The engine reads it as regression at high confidence. Target is continuous numeric (1154 unique values) - regression" + change. SURFACED-CORRECT (target-task-detection).
- Rendered skip: "NOT ASKED — SETTLED FROM THE FILE: Is glucose a regression problem? ... Stated in the transcript rather than asked — change it if it is wrong." [Ask me anyway]. SURFACED-CORRECT (task-type override as skip, DESIGN_LANGUAGE §09).
- Purpose Q2.5 "What is this model for?" numbered 2.5. Options: "Predicting an outcome for a new person" / "Estimating how strongly something is associated with the outcome". Picked prediction. Receipt "BUILT FOR: The model was built to predict the outcome for a new individual; handling was chosen to maximize predictive accuracy at deployment rather than to keep any single coefficient unbiased." SURFACED-CORRECT (model-purpose).

### MAJOR FINDING — opening sequence truncated: grain / eligibility / SEAL not surfaced
- Numbering shown: lens=01, target=01 (DUPLICATE "01"), purpose=2.5. There is NO Q3 grain question, NO Q4-7 repeats chain (expected-skip for this cross-sectional file), NO Q8 eligibility question, and NO SEAL card anywhere between Target and Explore.
- After purpose(2.5) the interview jumps straight to EXPLORE ("01 What the profile says about your data").
- Selecting the target auto-lit Explore in the nav BEFORE purpose(2.5) was answered — Explore is not gated on the opening sequence completing.
- The grain question (target-grain-question, register: "Built at L13", guided-only) and eligibility are built in the engine but NOT surfaced as answerable cards in this walking skeleton. The SEAL is a known not-yet-built gap (register cross-profile-row-scope: "the Guided interview does not ask for a seal at all yet"; target-lockbox-settings classic-only: "the act of sealing are not yet asked in the Guided interview").
- CONSEQUENCE (dead-end wiring): the impossible-value cards at Explore offer "Exclude those rows from the study" gated on TWO preconditions it prints — "Needs the grain question answered — §01 puts it before eligibility" and "Needs the held-out set not yet sealed". Since neither grain nor the seal is answerable in this UI, that eligibility exclusion path is UNREACHABLE. partially-wired / dead-end.

## STEP: Explore
- Profile card "01 What the profile says about your data." "Large sample (n=21,849). All model types are viable. Low dimensionality (p/n=0.00). Feature space is manageable." Table: rows 21849, features 28, numeric/categorical 25/3, missing overall 5.4%, p/n 0.001, data sufficiency abundant, target glucose-regression, mean(SD) 107.56 (35.58). SURFACED-CORRECT. NOTE features=28 => SEQN counted as a predictor (prereg #1 confirmed).
- Physiologic-flags card "2 physiologic flags" (caution, this table): bp_di, kcal, bp_sys, weight, height — "5 of 9 columns". Actions [Show me what this means][Decide at Preprocess][Dismiss][⚑ Mark for manuscript]. SURFACED-CORRECT.
- IMPOSSIBLE — BP_DI: 125 entries outside 15–220 mmHg, table of first 12/125 rows, [Set these entries to missing][Keep as is][Exclude those rows from the study]. Eligibility route gated (see dead-end above). Also "Mark the whole column as not trustworthy — not built (GUIDED-096)" honestly disclosed as unbuilt third option. partially-wired (GUIDED-096 = deliberately-not-built, disclosed).
- IMPOSSIBLE — KCAL: 9 entries outside 100–30,000 kcal, same structure.
- IMPROBABLE VALUES: 9 features, paginated (1/9), bp_sys 174 entries outside 90–200 band, advisory only. Carousel affordance ‹ 1/9 ›.
- No survey-weight finding, no pooled-cycle finding appeared (prereg #2 and #3 confirmed: nothing fires on total absence of design vars / 9 pooled cycles).
- Distribution histograms rendered per outlier feature (bp_sys skew=1.12 median 121.776 mean 124.019, bp_di skew=-0.79...). Nice mini-charts. SURFACED-CORRECT.

## NAV / PAGE STRUCTURE (correction)
- The app is ONE long single-page scroll: Data -> Target -> Explore -> Features -> Preprocess -> Train -> Explain -> Report/read-as-draft. Left-nav items are scroll anchors + step-status dots (filled=reached).
- Read-as-draft manuscript (TRIPOD/PROBAST scaffold) renders interleaved/near the end with rows like "Data preparation", "Model-building procedure", "Performance reported as discrimination AND calibration AND clinical utility [APP]", "Model presentation [APP]", "Fairness and subgroup evaluation", "Open-science items", "Limitations, intended use". Most rows read "not filled by the app yet ... filled when a model has been fitted / when a run has scored the held-out rows". SURFACED (cross-read-as-draft, guided-native).

## STEP: Features
- Selection Q "02 Should the models be given every column, or a chosen subset?" (choose_selection). Options include "Use every column".
- Transform catalogue (deferred/stateful): "Encode categories by how common they are [MEDIUM EXPLAINABILITY COST]", "Center and scale [LOW EXPLAINABILITY COST]", "Principal components [HIGH EXPLAINABILITY COST]" — each with a column... dropdown, "How many bins/components" inputs, and correct training-fold-leak rationale text. SURFACED-CORRECT (features catalogue, explainability tags).
- Settle control: "Nothing else — settle this step".
- Chose "Use every column" (selection). Settled. Receipt: "Feature work settled: 0 column(s) added now, 0 transform(s) recorded for fitting inside the training folds." + "This step is settled. Adding a feature now would make everything downstream stale." SURFACED-CORRECT.
- On settling Features, Explore's nav dot changed to an AMBER RING (deferred/unresolved findings indicator). Nice status affordance.

## STEP: Preprocess
- Intro: "PREPROCESSING, PER MODEL: The table below resolves once models are chosen, and models are chosen after the held-out set is sealed. Nothing is missing — the step that fills it has not happened." (references the seal as the gate).
- "01 What do the blanks in your table mean?" Asks only the two high-null CATEGORICAL columns: meds_chol (17,204/21,849 = 78.7%), meds_hbp (15,552/21,849 = 71.2%). Options Yes/No/I'm not sure. (Numeric-column missingness not asked here.)
- The same missingness question ALSO appears as a DIMMED/inactive Explore-stack finding "Is the missingness in meds_chol/meds_hbp informative?" (deferred-noticing duplicate). Clicking the dimmed Explore copy did nothing to the active step (correct — it is inactive), but the duplication is worth noting.
- Also saw dietary implausible-intake finding "kcal is below 500 on 194 record(s) and above 5000 on 307. Observed range 0 to 15,594" (dietary lens, info) = 501 implausible records (prereg confirms). SURFACED-CORRECT.
- Settled preprocessing (left both meds cols unanswered). Receipts: "This step is settled. Everything recorded here is fitted inside the training folds when the models are trained." + amber "Missingness settled: 0 recorded to be fitted inside the training folds. 2 column(s) with missing values have not been answered yet. Nothing was deferred..." + green "Missingness settled: 0 recorded...". SURFACED-CORRECT.

## STEP: Train — BLOCKED (no seal => no model shelf)
- Train region (read_page ref_78) contains ONLY: label "Train", state "open", veil "stale — an earlier answer changed", heading "Which models should be fitted?". NO model shelf, NO model checkboxes, NO fit/train button, NO seal control.
- The model shelf is gated on the seal being drawn (COPY_DECK: "Models are chosen after the seal"; Preprocess intro: "models are chosen after the held-out set is sealed ... the step that fills it has not happened"). Because the Guided interview NEVER draws the seal, the model shelf never renders and NO MODEL CAN BE FITTED via this happy path. This is the terminal functional state.
- Consequence: everything downstream of a fitted model is unreachable here: ROC/figures (FIG_DRAW roc/item_correlations), the checklist, performance numbers, the manuscript's model/performance rows.

## STEP: Explain — gated (no model) + SHAP deliberately out
- "01 Which columns is the model actually using?" (permutation importance). Gated: "No model has been fitted yet. Permutation importance is a drop in a metric when a column is shuffled, so there has to be a metric first — choose models in Train." partially-wired (blocked upstream by missing seal).
- "SHAP is not offered here" card: full deliberate rationale — shap is a prod dependency (numba/llvmlite) absent from dev env; packs carry no explainability content so no method to cite (GUIDED-101); "Classic offers SHAP and states its source". SURFACED-CORRECT as deliberate-out (GUIDED-232 / GUIDED-101). Credit: absence is explained, not silent.
- "RUN THE OTHER WAY: No missing-value decision on this project has an alternative that can be fitted the same way, so there is nothing to run both ways."

## STEP: Report == Read-as-draft manuscript (no separate report step)
- Clicking Report nav anchors to the read-as-draft methods document. Rows: Data preparation, Model-building procedure, "Performance reported as discrimination AND calibration AND clinical utility [APP]", "Model presentation ... [APP]", "Fairness and subgroup evaluation [USER+APP TEMPLATE]", "Open-science items [USER]", "Limitations, intended use... [USER+APP TEMPLATE]". Almost all read "not filled by the app yet" / "filled when a model has been fitted / when a run has scored the held-out rows". PROBAST (Wolff et al., Ann Intern Med 2019) reference. SURFACED (cross-read-as-draft) but empty because no model fitted.

## Prereg 4 gaps — confirmed as encountered
1. SEQN offered as predictor (features=28 includes SEQN) and as an unmarked TARGET chip. Confirmed.
2. Survey-design absence: no survey-weight finding fired. Confirmed (nothing surfaced about missing WTMEC2YR/SDMVSTRA/SDMVPSU).
3. Pooled 9 cycles: cycle_begin_year offered as ordinary predictor; nothing fired. Confirmed.
4. imputed_* flags read as ordinary binaries ("read as binary" bulk-repair, they are bool not text); not connected to base columns. Confirmed.

## Panels
- COACH LEDGER: 0 (nothing deferred to it during the walk).
- READ AS DRAFT: 0.


