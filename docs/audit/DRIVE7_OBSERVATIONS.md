# TurboTab Drive 7 — Observation Log

Build gate: `/dev/status` returned `rev: "0856c1d"`, `page_newer_than_engine: false`, `engine_stack_ok: true`. Proceeded.

Method note: this is an observation log — what appeared on screen, in order, with exact strings. No cause claims. Screenshots referenced by filename live in the session outputs folder.

---

## Run 1 · clinical_risk.csv · target `readmit_30d` · single-model figure check

Pre-upload screen: `screenshot-1787415244839-6c75cc3b.jpg`. Loaded: "clinical_risk.csv was loaded: 480 rows, 11 columns." (`screenshot-1787415448980-39b5db8d.jpg`)

Path walked: Data → Target (`readmit_30d`) → measurements lens ("Clinical measurements and labs") → "What is this model for?" (predicting an outcome for a new person) → "Can one person appear in more than one row?" (No) → "Is your study restricted to part of this data?" (No) → Draw the held-out set (sealed) → Train.

Seal banner: "THE HELD-OUT SET SEALED — 72 rows (15%) are held out and will not be looked at again until the models are scored." (`screenshot-1787415657394-f0b3cd90.jpg`)

### The figure list BEFORE any model was fitted (`screenshot-1787415723864-61bff80a.jpg`)

This is wrong — inconsistent explanations for the same underlying condition (nothing fitted yet). Verbatim, from the same list at the same moment:

- "Prediction instability plot — confirmatory — This figure does not apply to this project."
- "Calibration instability plot — confirmatory — This figure does not apply to this project."
- "Decision curve analysis — confirmatory — No model has been fitted yet. Choose models in Train and the curve is drawn from the held-out predictions."
- "ROC curve — confirmatory — No model has been fitted yet. Choose models in Train and the curve is drawn from the held-out predictions."
- "Model coefficients — confirmatory — This figure does not apply to this project."
- "Classification instability plot — confirmatory — This figure does not apply to this project."

ROC and decision curve state the true reason (no model yet). Model coefficients and the three instability plots claim the figure "does not apply to this project" — but after fitting a GLM, "Model coefficients" flipped to a drawn, SETTLED figure with 10 coefficients. So pre-fit, "does not apply to this project" was the wrong explanation for at least Model coefficients; the instability plots kept saying "does not apply" even after a fit, when the on-screen resampling section said the actual reason: "No resampling has been run yet. It refits the entire pipeline 200 times, so it is a job you start rather than something computed on the way past."

### Run 1a — only GLM (OLS/Logistic) ticked

Held-out: "GLM (OLS/Logistic) — Accuracy 0.75 · F1 0.677". Baseline sentence: "A model that always answers the more common level scores 0.764 on these held-out rows. Read every score against that, not against zero."

Figures present: ROC ✓ ("GLM (OLS/Logistic) C = 0.666", caption "C-statistic 0.666 (95% CI 0.492–0.808, 200 bootstrap draws)"), Calibration ✓ (intercept −0.238, slope 0.627, C 0.666, E:avg 0.080, E:max 0.671, n 72, events 17), Decision curve ✓ ("Decision curve analysis over 72 observations with 17 events (prevalence 23.6%)... Thresholds run from 5% to 35%"), Model coefficients ✓ (SETTLED, "Coefficients 10").

Not drawn, all with the identical sentence "This figure does not apply to this project.": Classification instability plot, Decision-curve instability plot, Scree plot with parallel analysis, Inter-item correlation matrix, Floor and ceiling effects, Item-response distribution panel. (`screenshot-1787416426398-d7cb9c62.jpg`)

**This is wrong — contradiction inside the ROC card** (`screenshot-1787416335600-ef536a78.jpg`, `screenshot-1787416315996-29679e0d.jpg`): the stat chip reads "C-statistic (95% CI): not estimable — The figure does not carry a value for this. A number is not shown because there is not one, rather than because it failed to render." while the plot legend directly below reads "GLM (OLS/Logistic) C = 0.666" and the caption gives "C-statistic 0.666 (95% CI 0.492–0.808, 200 bootstrap draws)". Same card, chip says the number does not exist, figure prints it. The PASS row "The C-statistic is annotated with an interval" sits between them.

### Run 1b — only Random Forest

Held-out: "Random Forest — Accuracy 0.736 · F1 0.699". Figures: ROC ✓ ("Random Forest: C-statistic 0.567 (95% CI 0.360–0.753, 200 bootstrap draws)"), Calibration ✓ (intercept −0.773, slope 0.304, C 0.567, E:avg 0.148, E:max 0.564), Decision curve ✓ (Vickers and Elkin caption present). (`screenshot-1787416690131-5bdb6aa4.jpg`)

"Model coefficients — confirmatory — This figure does not apply to this project." (`screenshot-1787416630271-10ce78ae.jpg`) — for a forest this is at least a defensible outcome, but the sentence never says why (a tree ensemble has no coefficients); it is the same generic sentence used for every withheld figure, so the reader cannot tell a principled absence from a bug.

### Run 1c — only Logistic Regression

Held-out: "Logistic Regression — Accuracy 0.75 · F1 0.677" — numerically identical to GLM (OLS/Logistic). Figures: ROC ✓ ("Logistic Regression: C-statistic 0.668 (95% CI 0.494–0.808, 200 bootstrap draws)", `screenshot-1787416877041-be2473c3.jpg`), Calibration ✓ (SETTLED), Decision curve ✓, Model coefficients ✓ (SETTLED, 10 coefficients).

**Verdict on the closed defect:** it did not reproduce. All three figures appeared in all three single-model runs.

**But the second expectation failed:** "the written report should mention calibration was assessed" — it does not, because the report never gained a Model Evaluation section at all after fitting (see Run 3).

### Other Run 1 observations (exact strings)

- Target banner: "readmit_30d is the target. The engine reads it as classification at high confidence. Target is binary (0/1) - classification" — the trailing "- classification" repeats the word; reads like two sentences glued together.
- "NOT ASKED — SETTLED FROM THE FILE" card: "Is readmit_30d a classification problem? — The engine reads readmit_30d as classification at high confidence: Target is binary (0/1) — classification Stated in the transcript rather than asked — change it there if it is wrong." — run-on; "classification Stated in the transcript" has no punctuation between the clauses.
- Measurements record: "...each is stated with its reasoning and was open to being overridden." — past tense ("was open") for something the user is still allowed to change.
- Train step, after the seal: "This step is waiting for the seal. The model shelf resolves once the held-out set is sealed — nothing is missing here and nothing has failed." — still displayed after the seal exists and even after models were fitted. Stale.
- Explain card, after a model was fitted and scored: "No model has been fitted yet. Permutation importance is a drop in a metric when a column is shuffled, so there has to be a metric first — choose models in Train." (`screenshot-1787416112712-982b5f96.jpg`) — wrong on its face at that moment.
- Three different counts of the same quantity, in three places:
  - Explore profile: "112 minority-class events for 488 candidate parameters (EPV = 0.2)"
  - Seal statement: "9 candidate predictor parameters were available to the models... excluding 1 column (encounter_id) whose every value is different... and which would otherwise have added 479"
  - Features amber card: "`encounter_id` has 408 different values across 408 rows... as a predictor it would cost 407 columns after encoding."
  - 488 vs (9+479=488) vs 407: the first two reconcile if you do the arithmetic yourself; 407 does not match 479. No screen explains the difference.
- Calibration caption phrase: "Calibration of GLM (OLS/Logistic) on 72 observations with 17 events of 1." — "17 events of 1" reads oddly.
- "SHAP is not offered here" (amber, Explain): "...`shap` is a production dependency and is deliberately absent from the development environment because it pulls numba and llvmlite, so a SHAP path could not be tested here..." followed by the bare line "Classic, pages/07_Explainability.py". This felt bad: build tooling and internal file paths speaking to a researcher mid-analysis.
- The Features question "Should the models be given every column, or a chosen subset?" was still open (card 02, unanswered) when models were fitted — training proceeded anyway with every column. Nothing said which answer was assumed at fit time; the card just stayed open.

---

## Run 3 · manuscript/checks panel, before vs after fitting (piggybacked on the Run 1 project)

### BEFORE fitting (sealed, nothing trained) — `screenshot-1787415959379-0b8cd29f.jpg`, `screenshot-1787415988920-00f46325.jpg`

Header, verbatim: "WHAT A REVIEWER WILL NOTICE   9 checks, 2 unmet · 4 declared"

(The brief expected "9 checks, 3 unmet · 4 declared" — the actual string says **2 unmet**.)

Below the header, FOUR amber-titled items are visible: "Analysis population is consistent across abstract and study design", "Final predictor count is consistent across abstract and methods", "No Model Development section", "No Model Evaluation section". Then the line "4 checks were decided before this draft was read, so they are reported rather than counted:" followed by four grey-titled declared items: "Split counts reconcile to analysis population", "Table 1 includes all finalized predictors", "Model names match between development and evaluation sections", "Abstract feature-selection language matches actual reduction".

**This felt bad:** the header says 2 unmet but four items are painted amber. I could not reconcile the count with what I was looking at without guessing that the first two amber items are neither unmet nor declared (their body text says "This check has nothing to compare... A number is not shown because there is not one, rather than because it failed to render."). Whatever the intended taxonomy, the count and the colors disagree on first read.

**Grey "declared" treatment — honest reaction:** the declared items did NOT look broken to me — they read as a third state, mostly because each carries a full explanatory paragraph. The explanations explain rather than shrug; e.g. "Split counts reconcile to analysis population — This check was decided before the draft was read: both the analysis total and the split come from the sealed_cohort partition, and nothing else in this project has counted those rows, so the sum restates the total instead of testing it, and it will agree however wrong that partition is. It is reported rather than counted, because a verdict nothing could have changed is not a unit of scrutiny." That sentence earned its place. The weakness is not the grey, it is the repetition — all four end with the identical boilerplate "It is reported rather than counted, because a verdict nothing could have changed is not a unit of scrutiny," so by the third one I was skimming.

### AFTER fitting (models fitted and scored, metrics on screen) — `screenshot-1787416980747-4266f517.jpg`

Header, verbatim, unchanged: "WHAT A REVIEWER WILL NOTICE   9 checks, 2 unmet · 4 declared"

Expected: "11 checks, 0 unmet · 2 declared". **It never appeared.** The draft above the checks also did not change: still "10 sentences · 1 gap only you can fill.", still "No exploratory decision has been recorded yet." / "No missingness or preprocessing decision has been recorded yet." / "No feature or selection decision has been recorded yet.", still amber "No Model Development section" and "No Model Evaluation section" ("Same source gap: the held-out metrics live on the run and the draft folds over decisions, so the manuscript states the analysis population and never says how the model was scored."). This was true after the GLM fit, after the RF fit, and after the LR fit, with held-out metrics visible in the Train section the whole time.

Consequence for a researcher: the draft manuscript never says the model was scored, never mentions the C-statistic, never mentions calibration — even while the app displays all three figures. The panel's own amber card describes the gap and then nothing closes it.

---

## Interaction and rendering problems hit during Runs 1/3 (this is wrong)

1. **The page shifts under the cursor between paint and click, repeatedly.** At least six of my clicks landed on the wrong control because the layout moved between when the screen was drawn and when the click landed: one click meant for a checkbox ticked "Metabolomics or proteomics" instead; one meant for "Answer it here" selected the Linear Discriminant Analysis model; one meant for "Record this one" opened a "Why we ask" disclosure; one meant for the GLM chip selected "Logistic Regression"; another selected "k-Nearest Neighbors". Each mis-click changed real state (models toggled on the shelf) that I then had to undo. A human moving slower would hit this less often, but the movement is real: identical screenshots seconds apart show content at different offsets without any scroll input.
2. **Scrolling paints half a frame.** During nearly every scroll the viewport rendered as a mostly-black screen with the app header stranded mid-viewport (e.g. header at y≈426 or y≈671 with everything above it black). It settles after a beat, but it happens on almost every scroll of any distance. You lose your place every time.
3. **Left-nav step names do not navigate reliably.** Clicking "Train" or "Explore" in the sidebar sometimes did nothing (view stayed where it was, item highlighted), sometimes moved the view a little. It took two or three presses to actually arrive. 
4. **Applying the event choice teleported me.** The second press ("Set the event for 'readmit_30d'") moved the view from the Train step all the way down to the Report section without being asked. (Documented under Run 2 rhythm as well since it is the same control.)
5. **Hover tooltips are walls of text that cover the controls.** The "Why we ask" hover on the measurements card produced a tooltip of roughly 150 words that covered the checkboxes and the confirm button (`ss_998865x7t` region; see `screenshot-1787415830054-a2d58d78.jpg` for the inline panel that later replaced it). Selected-state and hover-state on model chips are also nearly identical outlines — twice I could not tell whether GLM was selected or merely hovered until I moved the mouse away.

## Run 2 mechanism, as exercised on clinical_risk (the dedicated clinic_visits run follows separately)

The event-level question surfaced at Train as an amber card: "Which level of 'readmit_30d' is the event has not been recorded, and it decides what every score means — sensitivity and specificity are of the event, and the curves are drawn against it. There is no default: whether the event is (say) death or survival is the research question, not something the file can say. Answer “Which of these is the event you are predicting?” on the outcome, then fit."

- The card carries its OWN control: "Answer it here". Press 1 opened an inline panel directly beneath it, at Train: "THE EVENT — nothing is chosen yet" with choices "0" and "1 conventional" and the note "'1' is conventionally the event — shown as a suggestion, not applied. Nothing is selected for you here." No jump on this press. (`screenshot-1787415830054-a2d58d78.jpg`)
- Choosing "1" opened "WHAT THIS WOULD CHANGE — structure changes · nothing is applied yet" with a BEFORE/AFTER table and chips "ROWS 480 → 480 · COLUMNS 11 → 11 · CELLS CHANGED 0 → 0 · MISSING IN READMIT_30D 0 → 0 · DTYPE OF READMIT_30D int64 → Int64" and the sentence "'readmit_30d' was encoded with 1 as the event (1) and 0 as the comparison (0). Nothing has happened yet." Also inline at Train. (`screenshot-1787415844243-16f90743.jpg`)
- Press 2 ("Set the event for 'readmit_30d'") applied it — and the page jumped to the Report section. That violates "nothing should scroll or jump on either press."
- After applying, the amber refusal at Train was gone and fitting proceeded.

Judgment: the two-press rhythm itself felt like care, not a missing button — the preview genuinely told me what would change before anything changed, and for a 0/1 column the "CELLS CHANGED 0 → 0" chips made it easy to accept. The jump after the second press is what broke the feeling: the moment I committed, the app took me somewhere else and I had to scroll back up to fit.

---

## Run 2 · clinic_visits.csv · target `outcome` (text: 22 responder / 118 non-responder)

Loaded: "clinic_visits.csv was loaded: 140 rows, 14 columns." Import doctor on arrival: "4 features need the same repair: recode the missing-value codes as missing" (age, bp_2, glucose, notes), "2 features need the same repair: read as numbers" (weight, income), "11 things stand out in the shape of this file." I left the repairs at "Decide later" and drove the minimal path: target `outcome` → clinical lens → predicting-for-a-new-person → one row per person → no eligibility restriction → seal ("21 rows (15%) are held out and will not be looked at again until the models are scored.") → Train, Logistic Regression ticked.

Profile said: "22 minority-class events for 299 candidate parameters (EPV = 0.1)" and class sizes "non-responder = 118 · responder = 22".

### The refusal at Train

The wall appeared as an amber card directly under the model shelf: "Which level of 'outcome' is the event has not been recorded, and it decides what every score means — sensitivity and specificity are of the event, and the curves are drawn against it. There is no default: whether the event is (say) death or survival is the research question, not something the file can say. Answer “Which of these is the event you are predicting?” on the outcome, then fit."

- **Own control?** Yes — the card carries its own button, "Answer it here". It does not send you back to Data. (`screenshot-1787417941314-3741af72.jpg`)
- **Press 1:** an inline panel opened directly beneath the button, AT TRAIN: "THE EVENT — nothing is chosen yet", explanatory sentence, choices "non-responder" and "responder conventional", note "'responder' is conventionally the event — shown as a suggestion, not applied. Nothing is selected for you here.", plus "Close" and an amber "Decide at Explore". The page did not move on this press. (`screenshot-1787417965087-b46eafd5.jpg`)
- Choosing "responder" produced the preview, still inline at Train: "WHAT THIS WOULD CHANGE — 140 cells change, showing 8 · nothing is applied yet" with BEFORE (text labels, tinted red) → AFTER (0/1, tinted green) tables and chips "ROWS 140 → 140 · COLUMNS 14 → 14 · CELLS CHANGED 0 → 140 · MISSING IN OUTCOME 0 → 0 · DTYPE OF OUTCOME object → Int64" and the sentence "'outcome' was encoded with responder as the event (1) and non-responder as the comparison (0). Nothing has happened yet." (`screenshot-1787417979565-95cb127d.jpg`)
- **The trap check:** I scrolled all the way back to the Data step between presses — nothing had silently appeared there. The Data card region was unchanged. The defect-inside-its-own-repair did not reproduce.
- **Press 2** ("Set the event for 'outcome'"): applied; the refusal card and preview vanished from Train. Unlike the Run 1 apply (which landed me at Report), this time I stayed in the Train/Explain area — but because the tall amber card collapses on apply, everything below it shifts up several hundred pixels and the thing under your cursor changes. You keep your step but lose your place.
- After the apply, "Fit 1 model(s) on the held-out split" ran: "Logistic Regression — Accuracy 0.81 · F1 0.767" (21 held out, 119 trained; majority baseline stated as 0.857).

**Two-press judgment (asked for honestly):** on this dataset the rhythm read as care. The preview is the best screen in the app — the BEFORE/AFTER table with "CELLS CHANGED 0 → 140" showed me exactly what committing would do to a text outcome, and "Nothing has happened yet" in monospace under it is the right sentence in the right voice. I did not experience the second press as a missing button. What was annoying was everything around it: the page motion (below) meant my first "Answer it here" press physically missed the button and I spent four interactions and one full scroll round-trip confirming nothing had gone somewhere silently.

### This is wrong / felt bad, Run 2 specifics

- **Silent default imputation:** after fitting, the held-out section says "2 column(s) with missing values had no recorded handling and were filled with this app's default inside each training fold: Unnamed: 0, notes." The app that says "below high confidence the app is not allowed to choose for you" chose a missing-data strategy for two columns and disclosed it in small print after the fact. As a researcher I would want that surfaced before the fit, as a question, not after it as a footnote.
- The preprocess banner reads, in consecutive lines: "Missingness settled: 0 recorded to be fitted inside the training folds." / "2 column(s) with missing values have not been answered yet." / "Nothing was deferred, so nothing is waiting: every answer here either changed the table or deliberately left it alone." — "settled" and "not answered yet" and "nothing is waiting" cannot all be true in the plain reading; I read it three times.
- Explore finding title with broken grammar: "1 column write numbers in a format that does not parse — `income` carries a thousands separator, so every value parses as text (`110,000`)." ("1 column write numbers").
- The Train card again shows "This step is waiting for the seal..." after the seal exists (same stale sentence as Run 1).

---

## Run 4 · multiclass_stage.csv · target `bmi` (240 rows, 223 with a value)

Ground truth (allowed fixture check, run in bash): 240 rows, columns `record_id, age, sex, site, bmi, hba1c, sbp, crp, disease_stage`, **223 non-empty bmi**.

Loaded: "multiclass_stage.csv was loaded: 240 rows, 9 columns." The bmi chip's own tooltip read "float64 · 140 unique · 17 missing" — consistent with 223. Target recorded: "bmi is the target. The engine reads it as regression at high confidence. Target is continuous numeric (140 unique values) - regression" (same duplicated "- regression" suffix as Run 1's "- classification").

Path: clinical lens → predicting for a new person → one row per person → "No eligibility restriction: all 240 rows are in the study population." → seal → Ridge Regression → fit.

**Seal banner, verbatim (`screenshot-1787418372357-0a0865eb.jpg`):** "THE HELD-OUT SET SEALED — 33 rows (15% of the 223 with a value for the outcome; the other 17 of 240 rows have none) are held out and will not be looked at again until the models are scored."

**Methods/seal paragraph in the draft, verbatim:** "A test set of 33 rows was sealed before exploration and held by row label, on the basis 'cross_sectional' (user_stated). Of 223 rows with a value for the outcome, 33 were sealed as a held-out set before exploration and 190 were available for fitting. 9 candidate predictor parameters were available to the models, counted including any later dropped by feature selection, and excluding 1 column (record_id) whose every value is different, which the app does not hand the model and which would otherwise have added 239; a performance metric estimated on 33 rows carries a 95% interval up to 0.34 wide."

**The expected abstract sentence does not exist.** I searched the whole page for "observations was analyzed" — no match anywhere. There is no generated abstract; the TRIPOD row "Title and abstract identify the study as developing and/or validating a prediction model" reads "not filled by the app yet" (`screenshot-1787418657145-ab3325d6.jpg`). The only place an "abstract N" exists is inside the reviewer check, verbatim: "Analysis population is consistent across abstract and study design — Expected analysis N=223, abstract N=223, study design N not estimable..."

**Do the two agree?** Yes, where they exist: seal banner, methods paragraph, and the check all say 223 (and 190 = 223 − 33 also shows up in the model shelf: "a rough floor of 500 rows; you have 190"). No 240 leaked into any analysis-population claim. The 223-fix reached every path that renders; the specific abstract sentence from the brief simply is not in this build.

Fit result: "HELD-OUT PERFORMANCE — 33 rows held out · 190 trained on · regression"; "Ridge Regression — MAE 4.692 · RMSE 5.81 · R2 -0.117 · MedianAE 4.25". Also, again: "1 column(s) with missing values had no recorded handling and were filled with this app's default inside each training fold: crp." — same silent default-imputation disclosure pattern as Run 2.

### Other Run 4 observations

- **The app accepted a row-ID as a high-confidence target.** A shift-misclick landed my target click on `record_id` and the app recorded, without protest: "record_id is the target. The engine reads it as classification at high confidence. Target is object type (categorical/binary)" — for a column whose own tooltip says "object · 240 unique · 0 missing" on 240 rows. Every value unique, and the engine calls it categorical/binary at high confidence. Elsewhere the same app refuses to hand `record_id` to models precisely because every value is different. The two halves of the app disagree about the same column.
- **The transcript preserves the mistake with no annotation.** After I corrected the target back to bmi, the draft's methods section contains three consecutive sentences: "bmi was chosen as the target; the task was detected as regression at high confidence." / "record_id was chosen as the target; the task was detected as classification at high confidence." / "bmi was chosen as the target; the task was detected as regression at high confidence." Honest as a lab notebook; as a methods draft it now contains a false statement with nothing marking it superseded.
- **Preprocess offers to fill blanks in the outcome column.** "Does a blank in `bmi` mean something?" appears in the same list as the feature-missingness questions ("bmi numeric — 17 of 240 missing — 7.1%"), with the same three answers. Nothing distinguishes imputing a predictor from imputing the outcome you are trying to model. The "Settle preprocessing" button also renders *between* the crp question and the bmi question rather than after the last one.
- The checks header after this fit: still "9 checks, 2 unmet · 4 declared", draft "11 sentences · 1 gap only you can fill.", still "No Model Development section" / "No Model Evaluation section" — third dataset, same never-updating panel as Run 3.
- Nice sentence worth keeping (Train, no model selected): "No model is selected. The shelf orders every model this task can use and never shortens itself, so choosing is the step — nothing here is chosen for you."

---

## Run 5a · clinical_longitudinal.csv · target `progressed` (200 subjects × 3 visits = 600 rows)

Loaded: "clinical_longitudinal.csv was loaded: 600 rows, 13 columns." The arrival "noticed" card flagged only "'sex' is a binary variable written as text" — nothing about subject_id repeating. The profile card treats the table as "Adequate sample size (n=600) for most model types" with no mention of 200 subjects.

Path: target `progressed` → clinical lens → predicting for a new person → **"Can one person appear in more than one row?" answered "Yes, people repeat"** → "Are these repeats or different time points?" answered "Different time points" → "When you analyze this, what is one row?" answered "One row per record" → temporal question ("Are you predicting something that happens later from measurements taken earlier?") answered "Yes" → seal drawn.

**Does it notice? Yes — loudly, and this is the run's headline. It is not silent.** But the notices contradict each other and dead-end. In order of appearance, verbatim:

1. Ledger after the repeat answer: "ROWS PER PERSON — Asked whether one person can appear in more than one row; the answer recorded was: people repeat, and no column identifying the person was named — so the held-out rows cannot be drawn by person." **At no point did any control let me name the person column.** subject_id sat in the column list the whole time; the app never asked "which column is the person?"
2. ON THE RECORD, "WHAT ONE ROW IS": "Recorded: people repeat, and no column identifying the person has been named. Held-out rows are drawn BY ROW until one is, so the same person can sit on both sides and held-out performance would read better than the model is. Your numbers are labeled exploratory until a person column is named." (`screenshot-1787419822723-60ef42f7.jpg`)
3. Ledger after "One row per record": "ONE ROW IS — One row is one record; records stay as they are, and **held-out people never appear in training**." — this directly contradicts #2 ("same person can sit on both sides") and #4 ("drawn BY ROW"). Both statements were on screen at the same time. (`screenshot-1787419914543-5eefe6f3.jpg`, `screenshot-1787420006565-b7d4b789.jpg`)
4. Seal gate (good behavior): before the row question was settled, "Draw it now — NOT YET" with "People repeat in this table, so what one row means comes before the seal. One row per person and one row per record produce different held-out sets, and the seal cannot be drawn without knowing which." (`screenshot-1787419888011-77c1a6df.jpg`)
5. The seal itself, amber, titled "THE HELD-OUT SET   NOT A VERIFIED CLEAN SPLIT" (`screenshot-1787420073896-96f82d5f.jpg`): "90 rows (15%) are held out, drawn BY ROW because the data's shape is unknown. This is not a verified clean split: if rows repeat people, the same person is on both sides and held-out performance will read better than the model is. Treat these numbers as exploratory, and answer the grain question when you can. You stated that the task is predicting a later outcome from earlier measurements. **The held-out set was not drawn that way.** This app draws whole people at random, so the model trains on rows from after the rows it is scored on, and the held-out score is optimistic by an amount nothing here can measure. Your objective is recorded and the split is described as what it is; a temporal validation would need a split this app cannot yet draw."

### This is wrong (Run 5a)

- The ledger claims "held-out people never appear in training" while the record and the seal say the split is BY ROW and the same person can sit on both sides. One of these is false. A researcher reading only the ledger would believe the split is person-clean.
- "drawn BY ROW because the data's shape is unknown" — the shape is not unknown; I told the app people repeat, three times over. What is missing is the ID column, which the app never gave me a way to provide. "Answer the grain question when you can" points at a question with no home: Q03/Q05 were both already answered, and no card asks for the column.
- The seal text embeds literal `**` asterisks ("**The held-out set was not drawn that way.**") — markdown leaking unrendered into serif prose. Also a doubled sentence in the TASK ledger record: "The task is predicting a later outcome from earlier measurements. You stated that the task is predicting a later outcome from earlier measurements."
- "This app draws whole people at random" appears inside the same paragraph that says the split is BY ROW. Two contradictory mechanisms described in one breath.

### This felt bad (Run 5a)

- Color: the seal that declares "Treat these numbers as exploratory... the held-out score is optimistic by an amount nothing here can measure" is **amber**. Under this app's own color grammar, red is "numbers below here cannot be trusted." If any state in this whole drive earned red, it is this one; amber undersells it.
- As a researcher: the app diagnosed my clustering hazard exactly and then handed me no lever. The one action that would fix everything — "point at subject_id" — is the one action the interface does not contain. Being warned four times and empowered zero times is worse than it sounds; by the fourth warning I had stopped reading them.

---

## Run 5b · wide_assay.csv · target `responder` (60 rows, 47 columns)

Loaded: "wide_assay.csv was loaded: 60 rows, 47 columns." Arrival notice: "1 group(s) of columns look like repeated measures — probe_0 (45: probe_000, probe_001, probe_002...)."

Fixture correction to the brief: the profile reports class sizes "0 = 30 · 1 = 30" (balanced), not 22/38, and "30 minority-class events for 104 candidate parameters (EPV = 0.3)". Profile also: "Small sample (n=60). Strong regularization recommended. High dimensionality (p/n=0.77). Regularized models preferred." and "data sufficiency — scarce". Path taken: target `responder` → lens metabolomics/proteomics → BUILT FOR recorded (via a shift-misclick, see below) as "estimate the strength of association... keep the coefficients unbiased rather than to maximize predictive accuracy" → one row per person → no restriction → seal.

An Explore finding worth noting fired on arrival, already SETTLED: "These values look like they have already been transformed — Raw untargeted intensities span roughly 10² to 10⁹ and are strictly positive. Here, 1,362 values are negative (the smallest is -3.41); and 51% of features have a mean within 10% of their own standard deviation of zero, which is what centering leaves behind." (critical, metabolomics lens, confidence: high) — this is the lens system earning its keep.

### The small-holdout rule (the run's question): does it fire, and how does it speak?

It fires, at the seal, verbatim (`screenshot-1787420661494-ac230ff8.jpg`, `screenshot-1787420688847-4f85ac06.jpg`):

- Seal line: "THE HELD-OUT SET SEALED — 8 rows (13%) are held out and will not be looked at again until the models are scored."
- Bold inset: "A metric estimated on 8 held-out rows (4 of the less common outcome) carries a 95% interval up to 0.69 wide, on a scale of 0 to 1. With 2 classes a model that guesses is right 50% of the time, so the range that carries information is 0.50 wide."
- Explanation: "Shown because the widest 95% interval a metric on 8 held-out rows can have is 0.69 wide, which is more than the whole distance from chance to perfect — with 2 classes a model that guesses is right 50% of the time, so that distance is 0.50."
- Monospace: "8 held out   52 for fitting   4 of the less common outcome   45 candidate parameters" / "1 row moves a rate by 0.125"
- Closing: "This is a statement about the instrument, not about your study. The app does not know your expected effect size, which predictor is your exposure of interest, or what difference would matter — so it cannot say whether this design is adequate for your question, and it does not try. What it can do is arithmetic over what it holds, and leave the judgment where it belongs."

**Would a researcher accept it?** Yes. This is the best-written warning in the app: it says the interval (0.69) is wider than the entire informative range (0.50), makes it concrete ("1 row moves a rate by 0.125"), confines itself to arithmetic, and explicitly declines to judge my study. It does not read as being told off; it reads as a statistician who respects you.

**This felt bad, though:** the state chip is green "SEALED" and the inset is a calm green-tinted box — for a split whose own text says a single row moves the headline rate by 12.5 points and the widest interval exceeds the informative range. Under the app's color grammar that is at least amber. The strongest sentence in the app is wearing the calmest color in the app.

Also noted in this run: my BUILT FOR answer was set by a shift-misclick (the click meant for a checkbox landed on "Estimating how strongly something is associated with the outcome" as the page moved) — recorded verbatim as "The model was built to estimate the strength of association between the predictors and the outcome; handling was chosen to keep the coefficients unbiased rather than to maximize predictive accuracy." One more real state change caused by the page shifting under a click; and one more entry the transcript now carries that the user never intended.

---

## Run 6 · clinical_labs.csv · target `readmitted` · clinical lens · how many cards is too many?

Loaded: "clinical_labs.csv was loaded: 288 rows, 19 columns." Import doctor: "6 features need the same repair: read as numbers" (creatinine, ferritin, hs_crp, platelets, troponin, wbc), "2 features need the same repair: read as binary" (sex, site), "8 things stand out in the shape of this file." Profile: rows 288, features 18, class sizes "0 = 223 · 1 = 65", imbalance mild.

### The count

The Explore step now PUSHES SIX cards and FOLDS SEVEN behind a pull control. Verbatim fold row: "**7 more — 3 warnings, 4 cautions   2 from the clinical lens · 5 about this table**", which after clicking becomes "Fold those 7 back". Total accounted: 13 — the same thirteen as the known-bad measurement, but no longer all pushed. (`screenshot-1787421101638-0da765c2.jpg`)

Pushed (in order): 1. "`glucose` holds two populations a factor of 18 apart" (SETTLED, critical, high) · 2. "2 analytes carry censored values" (SETTLED, warning) · 3. "`sbp` holds impossible values and abnormal ones, and they are different categories" (SETTLED, warning) · 4. "`troponin` records both a measured value and a verdict" (SETTLED, warning) · 5. "2 columns write numbers in a format that does not parse" (SETTLED, warning; creatinine decimal-comma vs platelets thousands-separator — "'0,90' is one value in a decimal-comma locale and a hundred times that in a locale-aware reader... Nothing in the column settles which, so TurboTab has not parsed it either way.") · 6. "2 trajectories are not believable" (CONVENTION, warning; "1 adult change height by more than 5 cm between visits — the largest is 9 cm... 1 weight change exceed 30% inside 30 days — the largest is -34% over 21 days").

Unfolded seven include: "4 columns arrived as text and are mostly numbers", "`dbp` piles up on 80" (CONVENTION — "15.3% of its values are exactly 80, against 5.6% for the next most common reading. That is a documented EHR artifact... The same spike appears at 98.6 in `temp_f`, 120 in `sbp`."), "Small sample", "4 features with outliers", "2 physiologic flags", and more of the same shape.

### My call (asked for)

Six pushed is still too many, and the problem is the SAMENESS more than the count. Every card has the same serif title, the same tag row, the same four buttons ("Show me what this means / Decide at ... / Dismiss / ⚑ Mark for manuscript"). The one card marked `critical` — glucose possibly recorded in two different units, which would poison every downstream number — is visually identical in size and weight to "Small sample". I read cards 1–3 fully, skimmed 4–6, and had stopped reading tag rows entirely by the unfolded seven. If I hadn't been paid to count them I would have scrolled past the fold row without clicking it. Two pushed + the fold row would work IF the two pushed were chosen by severity and the critical one looked critical. The severity information exists — it is in the tiny monospace chips — but the layout spends it.

The fold row itself is good: "7 more — 3 warnings, 4 cautions · 2 from the clinical lens · 5 about this table" tells me exactly what I am deferring, and "Fold those 7 back" closes the loop. That is the right mechanism with the wrong threshold.

### Other Run 6 findings

- **The app dissents from a wrong answer — new good behavior.** I first answered "No, one row per person" and an amber note appeared under the question, verbatim: "`patient_id` has 96 distinct values across 288 rows, about 3 each. That is the shape of repeated measures, and you answered one row per person. One of those two readings is wrong, and which one changes how the held-out rows are chosen." This is the cross-check Run 5a's dataset never got.
- **...and then contradicts itself about the same column.** After I corrected to "Yes, people repeat", the ledger recorded: "the answer recorded was: people repeat, and no column identifying the person was named — so the held-out rows cannot be drawn by person." One card earlier the app itself named `patient_id`, counted its 96 values, and computed 3 rows each. It knows the column. It still says none was named, and still offers no way to name it. (Same missing lever as Run 5a, now with the app demonstrably holding the answer it says it lacks.)
- The seal here is gated on the grain question ("Draw it now — NOT YET ... The grain question comes before the seal") — consistent with Run 5a's gate.
- FIGURES header: "1 drawn · 16 accounted for". The drawn one, "PCA scores plot" (exploratory, SETTLED), carries the first FAIL row I saw anywhere: "FAIL — Pooled QCs overlaid in a distinct color, never dropped — Their tight central cluster IS part of the result — it is the evidence the run was stable." So the PASS/FAIL checklist can fail; it isn't decorative.
- A GOOD absence explanation exists in this build and shames the bad one: "Calibration plot — confirmatory — No model has been fitted yet, and none can be until the held-out set is sealed — a calibration curve drawn on rows the model was fitted on is a model grading its own homework." Compare "This figure does not apply to this project." — same situation class, night-and-day usefulness.
- Literal `**` markdown again, in serif card prose: "**TurboTab has not converted anything and will not.**" (glucose card), "**This is different from the 31 values outside 90–200 mmHg, which are abnormal but real and must be kept**" (sbp card), "`qns`, `tntc` are **not** censoring at a detection limit" (censored-values card). This is now confirmed across four datasets.

---

## THE DESIGN PASS (all runs, blunt)

**Motion is the app's worst enemy.** Three distinct failures, all repeated across all six runs:

1. **Recording any answer teleports the viewport.** After nearly every recorded answer the page jumped — usually to the top of the Target card, once (Run 1's event apply) all the way to Report. Between the jump and the next paint, clicks land on whatever moved under the cursor. Over the drive this caused at least ten wrong state changes: models toggled (LDA twice, k-NN, HistGB), a lens mis-ticked (Metabolomics, twice), a disclosure opened instead of a button pressed, and — worst — the TARGET changed twice (`record_id` on multiclass_stage, `probe_027` on wide_assay), and a BUILT FOR answer set to the wrong option. Each mistake also left a permanent line in the transcript. An app whose whole premise is "every answer is written down" must not let the layout answer questions for you.
2. **Scrolling paints half-frames.** On almost every multi-tick scroll the viewport rendered mostly black with the app header stranded mid-screen; it settles after a beat. You lose your place constantly.
3. **Left-nav step names navigate unreliably** — one click highlights, a second (sometimes third) actually moves. And "Start over" wipes the entire project in one click with no confirmation, while sitting in the same corner of every screen.

**Color grammar, judged by the app's own rules** (teal = act here, green = recorded, amber = advisory, red = numbers below cannot be trusted):

- Red was never used anywhere in six runs — including the two places its own definition demands it: Run 5a's "NOT A VERIFIED CLEAN SPLIT / Treat these numbers as exploratory" seal (amber) and Run 5b's "interval wider than the informative range" seal (green!). The scariest true sentences in the app wear its calmest colors.
- Two-plus teal "act here" cards are routinely on screen at once (Target picker + measurements lens + model-purpose + rows-per-person all open simultaneously on every fresh dataset). Teal borders also stay on ANSWERED cards (the target card keeps its teal border after recording), so "open question" and "settled question" look the same at a glance; only the tiny `open`/`recorded` chip differs.
- Grey "declared"/withheld-figure cards read as a third state, not as broken — the full explanatory paragraphs save them. Where the grey card carries only "This figure does not apply to this project.", it reads as a shrug and, at least once (Model coefficients pre-fit), the shrug was factually wrong.

**Three voices:** the serif/sans/mono split genuinely works — I could tell app-speech from my-actions from data without reading, and lines like "1 row moves a rate by 0.125" land precisely because they are in mono. Two leaks: raw `**markdown**` asterisks inside serif prose (four datasets), and the TRIPOD table renders whole methods paragraphs into a ~100px-wide column, one word per line, for dozens of rows.

**One question at a time:** the ideal is stated ("One card at a time") and not kept — a fresh dataset opens with three to four question cards stacked. The "NOT ASKED — SETTLED FROM THE FILE" card is a good idea wrecked by its copy: "...Target is binary (0/1) — classification Stated in the transcript rather than asked — change it there if it is wrong." — no punctuation between the clauses, and the same "- classification"/"- regression" suffix duplication appears in every target banner.

**Tooltips:** the "Why we ask" hovers are walls of 150–250 words that cover the very controls they describe (twice they swallowed my click targets). That content is good; a tooltip is the wrong container.

**Stale text is everywhere the state machine forgot to look:** "This step is waiting for the seal" after sealing (all runs); "No model has been fitted yet" in Explain after fitting (three runs); the reviewer panel frozen at "9 checks, 2 unmet · 4 declared" and the draft at "no Model Development/Evaluation section" through six fits across three datasets. The transcript records everything; the panels that summarize it do not re-read it.

**What earned trust, as the researcher this app is promised to:** the seal-before-explore discipline; the event-level two-press preview (best interaction in the app); the small-holdout arithmetic panel (best writing in the app); the import doctor's lens-specific findings (glucose units, censored assay values, decimal-comma vs thousands-separator — real chart-review-grade catches); the dissent when my grain answer contradicted `patient_id`'s shape; and the PASS/FAIL figure checklists that can actually FAIL. The skeleton of something researchers would trust is present. What breaks the promise is that the page moves underneath you, the summaries go stale the moment you act, the app repeatedly diagnoses problems (person column, calibration section, temporal split) that it gives you no control to fix, and the loudest dangers wear the quietest colors.

---

## Screenshot index (session outputs folder)

Run 1/3: screenshot-1787415244839, -1787415448980, -1787415486980, -1787415657394, -1787415670562, -1787415723864, -1787415803565, -1787415830054, -1787415844243, -1787415870525, -1787415882345, -1787415959379, -1787415988920, -1787416070502, -1787416112712, -1787416208716, -1787416315996, -1787416335600, -1787416426398, -1787416630271, -1787416980747. Run 2: -1787417675841, -1787417688656, -1787417941314, -1787417965087, -1787417979565, -1787417999386. Run 4: -1787418338184, -1787418351671, -1787418372357, -1787418657145. Run 5a: -1787419770817, -1787419822723, -1787419888011, -1787419914543, -1787419959298, -1787420006565, -1787420032087, -1787420045192, -1787420073896. Run 5b: -1787420644010, -1787420661494, -1787420688847. Run 6: -1787421101638.
