# Drive 8 — Classic UX / Surfacing Report

**App:** Tabular ML Lab — **Classic** Streamlit pages (`pages/01` … `pages/10`), branch `TurboTab` @ `8f88b6d`
**Dataset:** `_tt_tmp_nhanes.csv` — 21,849 rows × 29 columns (NHANES-style)
**Study driven:** target `meds_hbp` (binary, 71.2% missing), 27 predictors, Prediction mode, classification
**Date:** 2026-08-23 (evening; sequential, single process, one CPU-bound sweep at a time per the machine constraint)
**Interpreter:** `./venv/bin/python` (3.13.0, streamlit 1.60.0, scikit-learn 1.9.0)

## Method

Headless drive via `streamlit.testing.v1.AppTest`, the established Classic method (harness patterns from
`tests/integration/conftest.py` and `tests/integration/test_pages.py`). One page at a time, session state
carried forward verbatim between pages — which is what Streamlit itself does across pages of a multipage
app — so every page saw the state the previous page actually left.

**What was injected vs genuinely driven.** This matters and is stated plainly:

| Step | Status |
|---|---|
| **Upload of the CSV (page 01, Step 1)** | **INJECTED.** `AppTest` cannot drive `st.file_uploader`. One dataset was placed into `st.session_state["sp_projects"]` / `["datasets_registry"]` in exactly the shape `SessionProjectManager.add_dataset` leaves (`utils/session_projects.py:82-113`). **Consequence: the per-file Import Doctor card on page 01 was never rendered.** See Finding 3 — I ran `ml.import_doctor.diagnose` directly on the same file to report what that card would have said, and separately *did* drive the Import Doctor through the page-07 external-validation uploader, which is the same code path. |
| **External-validation file (page 07)** | **INJECTED file object, driven page.** A synthetic 300-row CSV was written to scratchpad and handed to the widget as a real `UploadedFile`; the load, the Import Doctor render and the "Validate on External Dataset" button were all genuinely driven. |
| Everything else on 01 (working-table confirmation, mode, target, predictor multiselect, subject declaration, holdout settings) | **DRIVEN** |
| 02 EDA (render, plausibility, VIF, Table 1) | **DRIVEN** |
| 03 FE (render only — the workflow skips it), 04 Selection (render + Run Feature Selection) | **DRIVEN** |
| 05 Preprocess (model picks, Advanced-mode inspection, Build ×2) | **DRIVEN** |
| 06 Train & Compare (Prepare Splits, Train Models, Bootstrap CIs) | **DRIVEN** |
| 07 Explainability (Run Selected Analyses, Subgroup, External Validation) | **DRIVEN** |
| 08 Sensitivity (seed sweep, all models) | **DRIVEN** |
| 09 Statistical Validation (two-sample test + override) | **DRIVEN** |
| 10 Report Export (render, validator, drafts, Evidence Map) | **DRIVEN** |

Optuna and the Neural Network were skipped as instructed. One harness artifact, not an app defect:
`st.page_link` raises `KeyError: 'url_pathname'` under `AppTest` (`pages/05_Preprocess.py:1594`) — it is the
last line of page 05 and nothing above it is affected.

Raw element dumps for every render are under the session scratchpad (`drive8/P0*.txt`); every string quoted
below is copied verbatim from those dumps.

---

## Executive summary

**Verdict: the new disclosures are, individually, the best writing in this app — and the session they
describe is one the app got substantially wrong without saying so. Two of the ten stages failed outright
on this dataset, both silently behind green banners, and the failure of one of them is asserted as a
completed method in the drafted Methods section. The paper-facing risk in this build is not that the
disclosures are missing; it is that the honest disclosures sit beside confident false statements and a
reader has no way to tell which is which.**

Five things a reviewer driving this app after reading the paper would find:

1. **Feature selection produced nothing, and the manuscript says it produced a consensus.** Both requested
   methods raised (`Unknown label type: unknown`); the page said *"Only one method completed"* when zero had,
   then *"Feature selection complete! 0 methods run."* in a **green success box**. The Evidence Map on page 10
   records the truth — `consensus: 19 → 0 predictors` — while the Methods draft one panel away reads
   *"Consensus feature selection across LASSO and RFE-CV retained all 27 candidate predictors."* The two are
   compiled from the same session. Traced to `pages/04_Feature_Selection.py:409-411` passing
   `methods_to_run` (requested) where `methods_completed` (the list `MISC-104` built ten lines earlier
   for exactly this reason) belongs.

2. **Explainability failed on all six analyses and reported success.** Every permutation and SHAP run
   died on `could not convert string to float: 'female'` — the page hands the *raw* frame to the explainers
   instead of the fitted pipelines that one-hot encoded `gender` on page 05. The six errors are folded into
   a collapsed expander; `pages/07_Explainability.py:835` then prints
   *"✅ Explainability analysis complete (0.4s)"* unconditionally. The empty run still ticks **two TRIPOD
   items** on page 10, both annotated *"Ran  on 3 models"* (note the missing analysis name — `analyses_run`
   was empty).

3. **The root cause of both is a target the app read as classification and sklearn reads as nothing.**
   `meds_hbp` arrives as an object column of Python bools with 15,552 NaNs; `type_of_target` returns
   `unknown`. Page 01 says *"Detected task type: **Classification**"* at high confidence, seals a stratified
   lockbox on it, and trains three models successfully — so nothing warns the user, and only the two steps
   that pass `y` straight to sklearn fall over. The one surface that would have caught it is the Import
   Doctor, whose finding for this column reads *"'meds_hbp' holds numbers but is stored as text — Every
   value is a plain number (e.g. 'True', 'False')"*. `True` and `False` are not plain numbers; the sentence
   contradicts itself, and applying the repair it offers is (confirmed) exactly what turns `unknown` into
   `binary` and unblocks both broken stages.

4. **The seal's denominator is stated in the Methods draft and nowhere on screen.** The chip on every page
   reads *"🔒 Test set: 15% (n=945, stratified) held out since upload"*. 15% × 21,849 = 3,277; 945 is 15% of
   the **6,297 rows that have an outcome**, a number the chip never names and no page-01 surface states
   outside a collapsed expander. The drafted Methods gets it right — *"The held-out test set (15% of
   eligible observations)"* — so the machine knows the basis and the chip declines to say it.

5. **The open counter is correct, coherent, and fires as an alarm on a clean run.** After the seed sweep the
   chip reads *"opened 2 times at Train & Compare, Sensitivity Analysis (seed sweep, re-split over the sealed
   rows)"* and a red warning says the estimate *"reads better than it will on new data"* — printed above the
   same page's own caption saying *"your reported headline metrics still come from the untouched lockbox test
   set"*. The user got there by following page 08's own advice. The counter is right; the warning attached
   to it makes a causal claim (a choice was made after seeing a held-out number) that this session does not
   support, and it now shadows pages 08, 09 and 10 for the rest of the run.

**What is genuinely good**, and should be said as plainly: the outcome-encoding disclosure on page 06
(*"'True' is class 1 — the event every metric describes … The order is alphabetical, not clinical"*) is the
best version of this the project has produced. The no-information-rate coach card, the model-selection
optimism sentence in the Methods draft, the baseline-preprocessing caption, the PROBAST objection on
univariate screening, the normality-driven test default with its Shapiro–Wilk numbers and its override
warning, and the pre-export validator (which caught two real problems and correctly disabled the downloads)
are all doing exactly what they were built to do. The second **Build Pipelines** press behaved perfectly —
identical output, no drift, no double-count.

**Noise verdict:** the app is not over-disclosing in general. It is over-disclosing in exactly two places —
the plausibility section, which prints the same two facts twice in two different vocabularies, and the
open-counter warning, which is a permanent red banner from page 08 onward.

---

## Per-stage log

### 01 · Upload & Audit

Chip and audit metrics on the first render after the (injected) upload:

> Rows **21,849** · Columns **29** · Missing Values **32,756** · Numeric Columns **20** · Duplicate Rows **0**

Cardinality flags `SEQN` as `Unique (potential ID) 🔑`. `Missing Values Detail` (collapsed) carries the only
statement anywhere on the page that the outcome is mostly absent:

> **2 column(s) have >50% missing values.** Consider removing or imputing.
> `meds_chol 17204 78.7%` · `meds_hbp 15552 71.2%`

The working-table sign-off is clean and honest:

> Everything from here on — the plots, the models, the numbers in your manuscript — describes this table and nothing else. **21,849 rows × 29 columns**
> *Nothing about this table looks surprising.*
> ☐ This is the table I want to analyze. *(help: Recorded with the table's current shape. If the table changes afterwards, this is withdrawn and you will be asked again.)*

**Predictor default.** Choosing `meds_hbp` pre-selects **all 28** remaining columns as predictors, `SEQN`
included — the column the audit two sections above flagged 🔑. No warning. Driving as a careful reader, I
deselected `SEQN` (27 predictors); the drive proceeded from there.

**The new subject/participant expander.** It renders, collapsed, labelled:

> 👤 Which column identifies a subject/participant?
> *If one person can appear in more than one row (repeated visits, several samples per participant), the held-out set must be drawn by PERSON — otherwise the same person is in training and in the test set and held-out performance reads better than it is. The app guesses from column names, and a name it does not recognize is why this control exists.*
> [Let the app work it out ▾]

Under the default answer the expander says **nothing else**. It does not report what the app worked out.
The seal record holds `seal_basis: "cross_sectional"`, `basis_source: "detected"` — a real answer the
control declines to show. The `else` branches all speak; the `_SUBJ_AUTO` branch only speaks when a prior
lockbox already has a `group_col` (`pages/01_Upload_and_Audit.py:1361-1366`).

I then drove the other two answers, because the brief asks whether the seal-basis chip is coherent with
what you declare. **Declaring `SEQN` produces four mutually inconsistent statements on one screen:**

> *(expander)* Recorded: `SEQN` identifies a participant — 21,849 of them across 21,849 rows. The held-out set is drawn by participant, and this column is held back from the predictors.
> *(info)* 🔒 Rows repeat per subject (`SEQN`), so the held-out set was drawn by **subject**, not by row — 945 rows from 945 subjects. Splitting by row would put the same subject in both training and testing.
> *(warning)* ⚠️ **The answer recorded and the data disagree.** `SEQN` was named as the participant identifier, but it has a different value on every one of its 21,849 rows. … until then the seal records that the grain is undetermined, and held-out performance may read better than it is.
> *(chip)* 🔒 Test set: 15% (n=945 rows from 945 subjects, split by 'SEQN' so no subject appears on both sides) held out since upload — not opened yet.

"Rows repeat per subject" is false (1.0 rows per value). "the seal records that the grain is undetermined"
is false — the record on that render is `seal_basis: "grouped"`. And the chip renders a clean lock whose
parenthetical ("so no subject appears on both sides") is vacuous when every subject is one row. A fifth
line adds a wrong *reason*: stratification was silently dropped (`stratified: false, strata: []`) because
grouping was requested, but the page says

> ⚠️ The held-out set could not be balanced on `meds_hbp` — too few people in some combination of those groups to put any in both halves. It IS balanced on nothing.

**Withdrawing the declaration does not undo it.** After switching to *"No — each row is a different
participant"* and then back to *"Let the app work it out"* — with `group_col` now `null` and the chip back to
*"(n=945, stratified)"* — the predictor list is still 27 of 27, and the page still says:

> Held back from the predictors: `SEQN` — the column the held-out set was split by — giving it to the model hands it the group membership the split exists to hide

**Reserved-column disclosure** is otherwise well built: it names the column and the reason. It is also one
render late in the other direction — on the render where the expander first says `SEQN` "is held back from
the predictors", the multiselect on the same screen still lists `SEQN` among 28 selected features.

Final state chip for the drive:

> 🔒 Test set: 15% (n=945, stratified) held out since upload — not opened yet — it opens at Train & Compare.
> ✅ Configuration saved: **Classification** task with **27** features

### 02 · EDA

Header tiles and the training-scope caption:

> The dataset profile and quick baselines see n=20904 training rows; held-out test rows are excluded to prevent selection leakage.
> Rows **21,849** · Features **27** · Numeric **19** · Categorical **8** · Missing **2.9%** · Sufficiency **High**

`20,904 = 21,849 − 945`. It is a true row count and a misleading statement about a *target-aware* step: a
quick baseline cannot be fit on the 15,552 rows with no outcome. Page 04 renders the same sentence template
with the right number (`n=5352`), so the two pages disagree about what "training rows" means.

The numeric/categorical counts contradict themselves twice more on the same page — the Distributions filter
offers `Numeric (25) · Categorical (2) · High Missing (1)`, and Macro Shape says *"across 25 features
(computed on 19 of 25)"*.

**Leakage:** none flagged. **Collinearity** is well surfaced:

> **2 collinearity cluster(s)** affecting 9 features total.
> ⚠️ Collinearity cluster: 3 features are intercorrelated (max r=0.90): weight, waist, bmi
> ⚠️ Collinearity cluster: 6 features are intercorrelated (max r=0.97): kcal, fat_total, carb, fat_mon, fat_sat, sugar

though the Relationships tab counts *"15 pairs above |r| ≥ 0.8"* and the VIF section warns *"9 feature pairs
are already correlated above the flagging threshold"* — two counts of the same thing, plus a third "9" one
panel up that means *features*, not pairs.

**A raw exception is printed as guidance.** Inside *Suggested Interactions*:

> Interaction detection skipped: Unknown label type: unknown. Maybe you are trying to fit a classifier, which exp

Truncated mid-word at 80 characters (`pages/02_EDA.py:1458`), and the sentence it truncates blames the
user's target for being *continuous* when it is a binary flag. This is the first appearance of the
`type_of_target == 'unknown'` defect.

**The four diagnostics.** Plausibility, Residual Normality, VIF, Influence. Two correctly decline for a
classification target (*"Regression only — residual normality is not a classification assumption."*,
*"Regression only — leverage and Cook's distance are defined for a least-squares fit."*). VIF carries the
diagnostic disclosure:

> VIF (Multicollinearity) reads the data and reports; it changes nothing. No open observation is waiting on it.

That sentence is contradicted two pages later (Finding 8).

**Improbability-band vocabulary — present, and undercut by its own duplicates.** The section caption is
exactly right:

> Reads each column against the NHANES p01–p99 improbability band and clinical guideline thresholds, after inferring its units. This is the check that catches a glucose column recorded in mmol/L being read as mg/dL — a mix-up that produces no statistical outliers at all.

and the insight beneath it says, correctly, *"That band is not a reference interval — a reference interval
is the central 95% of a healthy reference population, and a value outside this one is unusual rather than
abnormal."* Then the run prints, in this order, four warnings:

> ⚠️ kcal: 6.3% values outside the NHANES improbability band (800.0-4500.0 kcal) **after conversion from kcal**
> ⚠️ triglycerides: 9.1% values outside the NHANES improbability band (50.0-500.0 mg/dL) **after conversion from mg/dL**
> ⚠️ **kcal: 6.3% outside NHANES reference (800.0-4500.0 kcal)**
> ⚠️ **triglycerides: 9.1% outside NHANES reference (50.0-500.0 mg/dL)**

Two facts, four warnings, and the vocabulary the caption disavows in two of them.

**The pointer to the filter names a control that does not exist under that name:**

> → **Next:** Review flagged implausible values. Neither control is on Upload & Audit: target trimming is on Train & Compare, applied before the split, and plausibility filtering is on Preprocess under Advanced (full control).

On page 05 under Advanced the control is labelled **"Domain-specific range filtering"**, helped with
*"NHANES reference ranges for biomarkers"* (`pages/05_Preprocess.py:814`). Nothing on that page says
"plausibility".

**Sign-off gating:** there is none on this run — the page simply ends with
*"EDA complete. Proceed to Feature Selection or Preprocessing."* The gate exists (it is tied to leakage
blockers) and no blocker fired here, so this is silence-by-design rather than a miss.

**Table 1** builds on all 21,849 rows with no local caption, and mixes denominators inside one block:

> meds_chol, n (%) — False 1001 (21.6%) · True 3644 (78.4%) · **Missing 17204 (78.7%)**

21.6 + 78.4 = 100 (of non-missing), then 78.7% (of total), unmarked.

### 03 · Feature Engineering (skipped, as the workflow recommends)

Training-rows caption present and correct: *"Stateful transforms (PCA, binning, UMAP, TDA) are fit on
training rows, then applied to all rows."* One small internal disagreement: the polynomial tab shows
`Estimated new features ~190` beside `⚠️ This will create ~209 features.`

### 04 · Feature Selection

Scope caption, and it is the *right* number:

> Selection methods see n=5352 training rows; held-out test rows are excluded to prevent selection leakage.

**Categorical carry-through is disclosed well:**

> **8 non-numeric feature(s)** (gender, meds_chol, imputed_weight, imputed_height, imputed_bmi...) are excluded from ranking — selection methods require numeric inputs. They are carried through into the modeling feature set when you apply a selection below, and the manual selector lets you drop them by hand.

Univariate screening is off by default with the PROBAST objection stated at length — this is good, and the
literature framing on this page is the strongest prose in the app.

**Then the step fails, and three consecutive lines each say something different:**

> ⚠️ LASSO failed: Unknown label type: unknown. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
> ⚠️ RFE failed: Unknown label type: unknown. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
> ℹ️ Only one method completed, so there is no consensus to report — two methods must agree before a feature is called a consensus predictor. Run a second method, or use manual selection below.
> ✅ **Feature selection complete! 0 methods run.**

The ≥2 consensus floor itself is implemented correctly (`consensus_threshold = max(2, len(results) // 2)`,
`pages/04_Feature_Selection.py:385`, with the comment explaining why the old `max(1, …)` was a union
masquerading as agreement). Its *message* is guarded by `if len(results) < 2`
(`:426`), which cannot distinguish one from zero. The `st.success` at `:431` is unconditional.

### 05 · Preprocess

Execution-order banner and the median-imputation cost warning both present, both good:

> **Execution order:** The settings you configure here are **not applied yet.** … Preprocessing is fit on training data only, then applied to validation and test sets — this prevents data leakage. You are configuring *what* to do; the split determines *on which data* it happens.
> **What the median default costs.** Robust to skew as a point estimate — and that is all it is. Filling with one number understates that column's variance and distorts its distribution, and the outcome is not in the fill, so associations are biased toward the null. §A2 settles this as bad practice in a manuscript. Multiple imputation (MICE) preserves what a single fill cannot, and it is **not on this path** — the per-model controls, including the imputation choice, are only rendered under **🔧 Advanced (full control)**.

**Coach insights and dispositions** render as `📋 Coaching (4 open, 2 resolved)`, grouped by model family,
with a positive statement where a family is clean (*"✅ No family-specific issues for Tree-Based Models"*).
The dispositions are legible. One card should not be there:

> ⚠️ 2 feature(s) with >30% missing: **meds_hbp (71%)**, meds_chol (79%)
> → Review in Preprocessing — consider dropping or advanced imputation

`meds_hbp` is the **target**. The app is advising the investigator to drop or impute their own outcome, and
that card survives all the way into the exported report (see 10).

**Second Build press — clean.** Built pipelines for HISTGB_CLF / LOGREG / RF, then pressed Build again:
byte-identical recipes, one success banner, no duplicated pipelines, no drift in the summary cards. The
plausibility filter was left off, so no filter surfacing was exercised and no rows moved.

### 06 · Train & Compare

**CV is on by default — verified**: `Enable Cross-Validation` renders `value=True` with `CV Folds = 5`.

Splits:

> ℹ️ Outcome encoding: 'True' is class 1 — the event every metric describes — and 'False' is class 0. The order is alphabetical, not clinical; if the event you are studying is 'False', precision, recall, F1 and AUROC below describe its complement.
> ℹ️ Test set from the upload lockbox: n=945, sealed before feature engineering or selection could see it. The remaining rows were divided into train and validation.
> ✅ Splits ready — Train 4,407 (70%) · Val 945 (15%) · Test 945 (15%)

**The model badges quote the wrong n.** With 4,407 training rows on screen, the badges read:

> ✗ kernel training scales ~n² — slow at **n=20,904** *(SVC)*
> ✓ **n=20,904** supports the capacity *(Neural Network)*

`ml/model_coach.py:1654` takes `n = profile.n_rows`, the study-wide training count, not the analysis
cohort. SVC is *refused* on the basis of a size this run does not have.

Class imbalance is stated twice in two vocabularies — page 02's *"ratio=0.14"* and page 06's
*"(ratio: 7.2:1)"* — but the page-06 card is excellent:

> **Moderate class imbalance detected** (ratio: 7.2:1). Whether rebalancing is appropriate depends on what this model is for, and that has not been recorded yet. For a risk model or an association estimate it is contraindicated; for a classifier read at a fixed operating point it is defensible. (van den Goorbergh et al., JAMIA 2022;29:1525; replicated for machine-learning methods by Carriero et al., Stat Med 2025.)

**Training** (LOGREG, RF, HISTGB_CLF; 6.7 s) produced honest scoping:

> Data scope: unprefixed metrics are computed on the held-out test set (n=945); 'Train'-prefixed columns use the training rows (n=4407).
> **Dataset:** Total samples: 6,297 (4,407 train, 945 val, 945 test)

and two well-earned coach cards:

> ⚠️ Random Forest shows signs of overfitting: train F1 = 1.000 vs test F1 = 0.833 (gap: 0.167).
> ⚠️ Best accuracy (0.885, Logistic Regression) does not beat the no-information rate (0.877 — always predicting the majority class). **The models have not learned beyond prevalence.**

**Open-counter chip: one render behind.** On the render where training ran and the held-out metrics are
printed, the chip at the top still reads *"not opened yet — it opens at Train & Compare"*. It is captured
before the run (`_lb_opens_before = _lb_open_count()`, `pages/06_Train_and_Compare.py:106`) and the open is
recorded at `:1626`. On the next render:

> 🔒 Test set: 15% (n=945, stratified) held out since upload — opened once, at Train & Compare. Training again re-opens the same sealed rows — they have been scored against **1 time(s)** already.

The `time(s)` placeholder is visible in shipped copy, and "opened once … 1 time(s)" says it twice.

**CI nudge** fires as a coach card and the CI computation is properly disclosed:

> ℹ️ 3 models trained, and no bootstrap confidence intervals have been computed for the reported metrics and the ranking between them rests on point estimates alone.
> Format: estimate [95% CI lower, upper] via BCa bootstrap (1000 resamples) on the held-out test set (n=945)

— though on the render where I *did* compute them, the "no bootstrap confidence intervals have been
computed" card is still on screen above the CI table, the same one-render lag as the chip. The Baselines
caption is a model of its kind and worth preserving verbatim:

> Baselines are evaluated on the same held-out test set (n=945), on the original target scale. Baseline features: median imputation and z-score standardization for numeric columns; most-frequent imputation and one-hot encoding for categorical columns (fitted on the training rows only) — the baselines' own preprocessing, not any trained model's pipeline.

### 07 · Explainability

Chip is coherent: *"opened once, at Train & Compare. Explanations on this page are computed on the held-out
test set."* SHAP method routing is announced before the run:

> SHAP methods: **LOGREG**: ⚡ LinearExplainer · **RF**: ⚡ TreeExplainer · **HISTGB_CLF**: ⚡ TreeExplainer

Then **every analysis failed**:

> [⚠️ 6 issue(s) during analysis]  *(collapsed)*
> logreg permutation: could not convert string to float: 'female'
> logreg SHAP: requires numeric data (could not convert string to float: 'female')
> rf permutation: could not convert string to float: 'female'
> rf SHAP: requires numeric data (could not convert string to float: 'female')
> histgb_clf permutation: could not convert string to float: 'female'
> histgb_clf SHAP: requires numeric data (could not convert string to float: 'female')
> ✅ **Explainability analysis complete (0.4s)**

`gender` was one-hot encoded by all three pipelines on page 05 and all three models trained on it; the
explainers are being handed the raw frame. **No SHAP class labelling and no permutation consensus highlight
could be assessed, because neither was ever computed.** The coach card the run leaves behind is
`✅ ~~Ran  on 3 models~~ → Ran  on 3 models` — the analysis name is empty (`analyses_run == []`,
`pages/07_Explainability.py:817-824`) and the "resolution" restates the finding word for word.

**Subgroup analysis works** and reports N with CIs per stratum:

> LOGREG — Overall 945 0.8847 [0.8624, 0.9026] · female 493 0.8925 [0.8641, 0.9168] · male 452 0.8761 [0.8361, 0.8994]

The metric is **Accuracy only** — on a run where pages 02 and 06 both told the user
*"Accuracy alone may be misleading. Use F1, balanced accuracy, or AUROC"* and where accuracy does not beat
the no-information rate. There is no caption naming the resampling method or the multiplicity of comparing
three strata × three models.

**External validation — the strongest new surface in the drive.** The uploader runs the full page-01 front
door (layout disclosure, transpose, Import Doctor), then:

> ✅ Loaded external dataset: 300 rows × 28 columns
> Structural review — Found 1 note.
> ℹ️ **6 column(s) have the same value in every row** — imputed_weight, imputed_height, imputed_bmi, imputed_waist, imputed_bp_sys….
> *Why this matters: A constant carries no information for any model, though it may still be worth keeping as a study-level label.*
> *This is a question, not a recommendation — only apply it if you know these values really do mean 'missing'.*

That last sentence belongs to a missing-value-sentinel finding, not a constant-columns one. After
validating:

> ✅ Recorded. The Methods draft now reports this external validation, with these models and this cohort size.
> ✅ External validation on record: **external_validation_synthetic.csv** (300 rows), 3 model(s), 95% CIs from 500 bootstrap resamples. This is what the manuscript reports.

That is exactly right, and the draft does report it faithfully. (Accuracy collapses to ~0.45 on the
synthetic cohort because its prevalence is 23% against the development cohort's 88%; nothing on screen
mentions prevalence shift, but nothing on screen is false either.)

### 08 · Sensitivity Analysis

All-models default **verified**, with an honest cost statement:

> Models to re-seed: ● **All 3 trained models** ○ LOGREG only (faster)
> This run fits 24 models (3 model(s) × 8 seeds).
> *Note: this diagnostic deliberately re-partitions all rows (including the locked test set) to measure split sensitivity — your reported headline metrics still come from the untouched lockbox test set on Train & Compare.*

The sweep declares itself where it happens:

> ⚠️ This sweep pooled the **sealed test rows** back in and retrained over them. It is recorded as an opening of the held-out set: the spread below measures split sensitivity, not held-out performance, and the Methods section will say the sealed set was accessed here as well as at Train & Compare.

**Achieved-seed reporting: present.** The full results table carries a `seed` column with the actual values
used (0, 1, 7, 13, 42, 99, 123, 456), not just a count.

**The open counter after the sweep** — the brief's specific question. It says:

> ⚠️ The sealed test set has been **opened 2 times** — models have been scored against it, or retrained over it, on 2 separate occasions (Train & Compare, Sensitivity Analysis (seed sweep, re-split over the sealed rows)). A held-out estimate is unbiased only for a single, final evaluation; once a choice (features, preprocessing, models) is made after seeing a held-out number, that number is part of the model selection and reads better than it will on new data. Report it as such, and say in the Methods that the set was accessed 2 times.
> 🔒 Test set: 15% (n=945, stratified) held out since upload — **opened 2 times** at Train & Compare, Sensitivity Analysis (seed sweep, re-split over the sealed rows).

**Does it read coherently or alarming?** Both, and that is the problem. The *count* is right and the
*source list* is right — a real improvement over an unconditional "opened once". But the surrounding
sentence asserts a mechanism this session does not have: no choice was made after seeing a held-out number
(the sweep ran after all modelling decisions, and its own caption says the headline metrics are untouched).
So the page contradicts itself top-to-bottom, and because `render_lockbox_status` fires the warning on
every page thereafter, the last three stages of the workflow are permanently red-bannered for having
followed the app's own recommendation.

### 09 · Statistical Validation

Scope caption is precise:

> Tests on this page run on the full cohort (including locked test rows) — appropriate for Table 1 and descriptive claims, not model-performance claims.

**Normality-driven default with the assumption caption — works, and reads well:**

> **Assumption check:** male: Shapiro–Wilk p=1.09e-76; female: Shapiro–Wilk p=2.246e-77 → defaulting to **Mann-Whitney U** (normality rejected or untestable at α=0.05). You can override below; whichever test runs is recorded with the result.
> ☐ Use parametric test (t-test) *(help: The box is pre-set from the Shapiro-Wilk result above.)*

The result carries the provenance: *"Test selection: … → assumption check chose the non-parametric test."*

**Override warning — works:**

> ⚠️ Overriding the assumption check: running the **t-test** where the pre-check selected the **Mann-Whitney U**. The override is recorded with the result.

and after re-running, the caption flips to *"→ author override chose the parametric test."* This is the
cleanest disclosure loop in the app.

Two small things: on the render where the override is ticked but not yet re-run, the stale Mann–Whitney
result sits directly beneath the warning announcing a t-test; and p-values render as `0.0000` rather than
`< 0.001`, which no journal accepts. The default variable in every test panel is `SEQN`.

### 10 · Report Export

**The pre-export validator earns its keep.** 13 checks, 2 real failures, downloads correctly disabled:

> ⚠️ 2 of 13 validation checks failed. Review the report below before exporting.
> FAIL — Final predictor count is consistent across abstract and methods — *Expected predictors=27, abstract=27, predictor section=None.*
> FAIL — Abstract feature-selection language matches actual reduction — *Abstract still describes feature reduction even though original=27 and selected=27.*
> ⛔ Downloads are disabled because pre-export validation failed — open **🔍 Pre-export Manuscript Validation** above to review the failed checks or apply an override.

The readiness audit is likewise honest — `Permutation Importance ❌ Missing`, `Shap ❌ Missing`,
`Readiness summary: 4 present · 0 inferred/recomputable · 4 missing`. The **Evidence Map** preamble is a
promise the app mostly keeps:

> "NOT RECORDED" means the pipeline holds no evidence for that section — the draft omits it rather than inventing it.

**And the row directly beneath it is where the drive's headline defect becomes visible:**

> | Predictor Variables | feature-selection record | **consensus: 19 → 0 predictors** |

against the Methods draft in the panel above:

> All 27 candidate predictors were retained for final modeling. **Consensus feature selection across LASSO and RFE-CV retained all 27 candidate predictors.** The full final predictor list is provided in Supplementary Table S1.

**The Methods draft read end-to-end, as a coauthor.** What is right first: the Study Design paragraph is
excellent and states the seal basis the on-screen chip never does —

> Data were partitioned using a stratified random (test set locked at upload, before feature engineering/selection) split into training (n=4,407, 70%), validation (n=945, 15%), and test (n=945, 15%) sets (random seed=42). The held-out test set (**15% of eligible observations**) was frozen at data upload, before any feature engineering or feature selection, and was **accessed 2 times** during model development. Because model, preprocessing or feature choices could follow each access, the reported held-out performance is not a single untouched evaluation and may be optimistically biased.

and the model-selection optimism sentence is the kind of thing most tools never write:

> Because the reported model was chosen by comparing 3 models' scores on the held-out set, its reported performance is optimistic relative to a model chosen without those rows; the size of that optimism was not estimated here.

The external validation, per-model preprocessing, hyperparameters and CV are all reported accurately.
**The sentences that feel wrong, quoted:**

- *"Consensus feature selection across LASSO and RFE-CV retained all 27 candidate predictors."* — nothing ran.
- *"The raw dataset contained 27 predictor variables, feature engineering yielded 19 candidates, and feature selection retained 27 predictors for final modeling."* — feature engineering was skipped; 19 is the *numeric-selection* candidate count relabelled.
- *"Missing values were handled using median imputation."* — no mention of the missing-indicator columns that were actually added, and no mention that 15,552 rows were removed for having no outcome. Elsewhere: *"2 feature(s) with >30% missing: meds_hbp (71%), meds_chol (79%). Missing values were handled with median imputation across model pipelines."* — asserting the **outcome's** missingness was median-imputed.
- *"The sample-to-feature ratio was 809:1, supporting model estimation relative to predictor dimensionality."* and, in Strengths, *"the sample size was large relative to the number of predictors (809:1 observations per predictor)"* — 809 is 21,849 / 27, in a document whose first sentence says the analysis had 6,297 observations (233:1) and whose Sufficiency paragraph says *"654 minority-class events for 27 candidate parameters (EPV = 24.2)"*.
- *"Large sample (n=20,904). All model types are viable. Low dimensionality (p/n=0.00)."* — a fourth n for the same study, in the same document.
- *"Class imbalance detected (ratio=0.14). 3 models were trained and compared; best: Logistic Regression (best model: Logistic Regression; best metric: 88.5%)."* — training models is not a response to imbalance; the parenthetical repeats the sentence; and 88.5% is accuracy where F1 is named as the selection metric two paragraphs earlier.
- *"Collinearity cluster: 3 features are intercorrelated … **Diagnostic analysis via multicollinearity_vif.**"* — an internal action id in manuscript prose (the validator's *"No internal model keys leak"* check passed; it only looks for model keys).
- Limitations, mid-list: *"…; **Histogram Gradient Boosting (Classification): scaling robust; the recipe table does not require scaling here. Logistic Regression: scaling standard, as the recipe table requires. Random Forest: scaling robust; the recipe table does not require scaling here.**; the simpler Logistic Regression performed within 0.1%…"* — an internal preprocessing-consistency note spliced into a semicolon list of study limitations, with its own full stops inside it.
- Principal Findings: *"**A reviewer would question why the more complex model was selected.**"* — verbatim coach copy addressed to the user, printed in the Discussion. The validator's *"No coaching language detected"* check passed.
- Statistical Validation: *"Statistical validation was performed using: Mann–Whitney U, t-test (ind.). No correction for multiple comparisons was applied across the **2 tests**…"* — those two "tests" are the same comparison (glucose × gender) run once by the assumption check and once by my override. An override re-run is counted as an independent second test.
- Executive summary: *"Key Data Warnings: - Moderate imbalance - **1 features** with high missingness - 16 features with outliers"* — grammar, and it contradicts EDA's *"2 feature(s) with >30% missing"*.

**TRIPOD:** *"12/22 items addressed (auto-completed from your workflow)"*. Three of the ticks rest on
nothing:

> 15a — Present the full prediction model to allow predictions for individuals — ✅ — *Ran  on 3 models*
> 19a — Give an overall interpretation of results … — ✅ — *Ran  on 3 models*
> 9 — Describe how missing data were handled — ✅ — *Configured classification task with 27 features, target: meds_hbp*
> 8 — Explain how the study size was arrived at — ✅ — *Positive signal — no action needed*

The two *"Ran  on 3 models"* items are ticked by the explainability run in which all six analyses failed.

**Table 1 on this page is the good one** — built on the analysis cohort (N=6,297) with the basis stated:
*"Built from the rows the split RECORDED as the analysis cohort — the population the models were fitted and
evaluated on — and the finalized predictor set."* It carries the same mixed-denominator `Missing` rows as
page 02's.

**Export defaults** are sensible (model artifacts, predictions, training/explainability/calibration plots
on; sensitivity plots, raw data and LLM interpretations off) — though *Explainability plots* defaults on
with help *"Permutation importance bar charts"* in a session where the readiness audit says permutation
importance is missing. The manifest surfaces in *Advanced / State Debug*:

> • Data shape: (21849, 29) · Target: meds_hbp · Features: 27 · Trained models: 3 · Dataset profile: Available
> • Insight ledger: 26 entries (19 resolved, of which 8 are narrative-worthy — the rest are activity records)
> • Git info: {'app_version': '1.0.0', 'commit': 'n/a'}

That last line is worth noting on its own: a reproducibility manifest whose commit is `n/a`.

The `AUDIT-042` count/list mismatch appears **fixed** on this run — *"Exploratory analysis identified 14
data observations. 8 were addressed during the modeling workflow; 6 were documented and accepted"* is
followed by exactly 8 listed items.

---

## Findings, ranked

Severity: **critical** = a false statement reaches the manuscript or a number a researcher would report is
wrong; **high** = a surface asserts something the session did not do; **medium** = contradiction or stale
copy a reviewer would notice; **low** = polish.

| # | Sev | Finding | Quoted surface | Traced to |
|---|---|---|---|---|
| 1 | **critical** | **The Methods draft asserts a consensus feature selection that never ran.** Both methods raised; zero completed; the Evidence Map records `19 → 0` while the prose says 27 were retained. Cause: provenance is given the **requested** method list, not the `methods_completed` list `MISC-104` built ten lines above for exactly this purpose. | "Consensus feature selection across LASSO and RFE-CV retained all 27 candidate predictors." vs Evidence Map "consensus: 19 → 0 predictors" | `pages/04_Feature_Selection.py:409-411` (`consensus_methods=list(methods_to_run)`) → `utils/workflow_provenance.py:450`, `:722` → `ml/narrative_engine.py:896-900, 929` |
| 2 | **critical** | **Feature selection fails entirely on a boolean-with-missing target, and the failure message blames the user's target for being continuous.** `type_of_target(meds_hbp)` is `unknown`; LASSO and RFE both raise. | "⚠️ LASSO failed: Unknown label type: unknown. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values." | `pages/04_Feature_Selection.py:332`, `:346`; verified `type_of_target` = `unknown` on the raw column and `binary` after `pd.to_numeric` |
| 3 | **critical** | **Explainability fails on all six analyses and reports success**; the raw frame is handed to the explainers instead of the fitted pipelines. Two TRIPOD items are then ticked from the empty run. | "logreg SHAP: requires numeric data (could not convert string to float: 'female')" … "✅ Explainability analysis complete (0.4s)" | `pages/07_Explainability.py:811-814` (errors collapsed), `:835` (unconditional success), `:817-824` (`analyses_run` empty → provenance still recorded) |
| 4 | **critical** | **The Methods draft reports a sample-to-feature ratio computed on the wrong denominator**, in a document that states the analysis cohort three paragraphs earlier. 809 = 21,849/27; the cohort is 6,297 and the training set 4,407. | "the sample size was large relative to the number of predictors (809:1 observations per predictor), supporting stable model estimation" | `pages/02_EDA.py:708-718` (`regime.n_rows / n_features`, with `manuscript_text` carried verbatim to the Discussion) |
| 5 | **high** | **The zero/one guard on the consensus floor cannot count**, and a green success reports a null result. | "Only one method completed, so there is no consensus to report" (zero completed) followed by "✅ Feature selection complete! 0 methods run." | `pages/04_Feature_Selection.py:426-431` |
| 6 | **high** | **The Import Doctor's own description of the defect at the root of #2 and #3 contradicts itself**, and its repair is (confirmed) what would unblock both. The card was never seen in this drive because the target column reaches the doctor at upload. | "'meds_hbp' holds numbers but is stored as text — Every value is a plain number (e.g. 'True', 'False') but the column is typed as text." | `ml/import_doctor.py:685-706` |
| 7 | **high** | **Declaring a subject column produces four contradictory statements on one screen**, including a false "rows repeat" and a warning asserting an `undetermined` seal while the record says `grouped`. | "Rows repeat per subject (`SEQN`)" · "the seal records that the grain is undetermined" · chip: "split by 'SEQN' so no subject appears on both sides" | `pages/01_Upload_and_Audit.py:1414-1421`; `utils/test_lockbox.py:1295-1302` vs `:936-973` (`seal_basis`) |
| 8 | **high** | **Withdrawing a subject declaration does not release the reserved column.** `SEQN` stays out of the predictor pool with a reason the record no longer supports. The replacement-semantics helper for exactly this exists and is not used. | "Held back from the predictors: `SEQN` — the column the held-out set was split by" (while `group_col` is `null` and the chip reads "(n=945, stratified)") | `pages/01_Upload_and_Audit.py:1408-1412` uses `register_reserved_column` (additive) where `utils/combine.py:287-298` `set_reserved_columns(role=…)` is the replacement version |
| 9 | **high** | **The seal chip never states the denominator its 15% is taken from.** A reader computes 3,277 and sees 945. The Methods draft states it correctly, so the machine knows. | "🔒 Test set: 15% (n=945, stratified) held out since upload" vs draft "The held-out test set (15% of eligible observations)" | `utils/test_lockbox.py:1385-1389` (clean-path chip); `ml/narrative_engine.py` study-design section has the eligible-observations clause |
| 10 | **high** | **Model-suitability badges quote the study-wide n, not the analysis cohort**, and refuse a model on that basis: 20,904 on a screen showing 4,407 training rows. | "✗ kernel training scales ~n² — slow at n=20,904" · "✓ n=20,904 supports the capacity" | `ml/model_coach.py:1654` (`n = profile.n_rows`), rendered at `:1740` and `:1758` |
| 11 | **high** | **The coaching panel lists the target among "features with >30% missing" and advises dropping or imputing it** — and that card reaches the exported report as an addressed observation. | "⚠️ 2 feature(s) with >30% missing: meds_hbp (71%), meds_chol (79%) → Review in Preprocessing — consider dropping or advanced imputation" | page 02 insight → page 05 coach panel → report "Missing values were handled with median imputation across model pipelines" |
| 12 | **medium** | **The open-counter warning makes a causal claim this session does not support, and contradicts the same page's caption.** It then shadows pages 08–10 permanently. | "…that number is part of the model selection and reads better than it will on new data" printed above "your reported headline metrics still come from the untouched lockbox test set" | `utils/test_lockbox.py:1198-1208` (`_opens > 1` branch); page-08 caption in `pages/08_Sensitivity_Analysis.py` |
| 13 | **medium** | **"VIF … changes nothing. No open observation is waiting on it" is contradicted two pages later** by a coaching panel crediting VIF with resolving two observations. The carve-out is deliberate and tested; the sentence is the copy that did not move with it. | "VIF (Multicollinearity) reads the data and reports; it changes nothing." vs "✅ ~~Collinearity cluster…~~ → VIF (Multicollinearity): VIF computed for 19 features." | `ml/eda_actions.py:1729-1733` (`n_open <= 0` branch) vs `pages/02_EDA.py:2304-2318` (the VIF carve-out resolves `eda_corr_cluster_*` **before** the disclosure counts what is open) |
| 14 | **medium** | **Improbability-band vocabulary is duplicated in the vocabulary the caption disavows.** Two facts, four warnings, two of them saying "NHANES reference". | "kcal: 6.3% values outside the NHANES improbability band …" + "kcal: 6.3% outside NHANES reference …" under "That band is not a reference interval" | `ml/eda_actions.py:134-135` (new) vs `ml/eda_recommender.py:563` and `ml/dataset_profile.py:983` (stale); also `pages/05_Preprocess.py:814` |
| 15 | **medium** | **A raw, truncated sklearn exception is printed as user guidance**, mid-word, and blames a binary target for being continuous. | "Interaction detection skipped: Unknown label type: unknown. Maybe you are trying to fit a classifier, which exp" | `pages/02_EDA.py:1458` (`str(e)[:80]`) |
| 16 | **medium** | **Coaching language and an internal action id reach the manuscript**, past validator checks that report the opposite. | Discussion: "A reviewer would question why the more complex model was selected." · Data Observations: "Diagnostic analysis via multicollinearity_vif." · validator: "No coaching language detected." / "No internal model identifiers detected." | `pages/10_Report_Export.py` validator checks 10–11; narrative sources in `ml/narrative_engine.py` |
| 17 | **medium** | **Limitations splices an internal preprocessing note into a list of study limitations**, complete with embedded full stops. | "…; Histogram Gradient Boosting (Classification): scaling robust; the recipe table does not require scaling here. … ; the simpler Logistic Regression performed within 0.1%…" | preprocessing-consistency insight `manuscript_text` → Discussion limitations assembly |
| 18 | **medium** | **The subgroup table — the reviewer-facing one — reports Accuracy only**, on a run where two earlier pages said accuracy is misleading and where accuracy does not beat the no-information rate. No resampling or multiplicity caption. | "Overall 945 0.8847 [0.8624, 0.9026] · female 493 0.8925 · male 452 0.8761" | `pages/07_Explainability.py` subgroup block |
| 19 | **medium** | **The page counts numeric and categorical features three different ways on one screen**, and the report picks a fourth. | tiles "Numeric 19 · Categorical 8" · filter "Numeric (25) · Categorical (2)" · Macro Shape "25 features (computed on 19 of 25)" · report "Numeric Features 25 / Categorical Features 2" | `pages/02_EDA.py` header tiles vs distribution filter vs `ml/macro_shape` caption |
| 20 | **medium** | **An override re-run is counted as an independent second statistical test** in the multiplicity sentence. | "No correction for multiple comparisons was applied across the 2 tests reported here" — one comparison, run two ways | statistical-test record → `ml/narrative_engine.py` statistical-validation section |
| 21 | **medium** | **Abstract invents a feature-engineering stage that was skipped.** (The validator catches the *reduction* half of this and not the *engineering* half.) | "feature engineering yielded 19 candidates" | `ml/narrative_engine.py:902-916` (`candidate_count` from the selection record's `n_features_before`) |
| 22 | **low** | **The subject/participant expander never says what the app worked out** under its default answer, and is collapsed. The record holds `seal_basis: cross_sectional, basis_source: detected` — an answer worth one line. | expander contains only the rationale caption and the selectbox | `pages/01_Upload_and_Audit.py:1361-1366` (the `_SUBJ_AUTO` branch speaks only when a prior `group_col` exists) |
| 23 | **low** | **The 100%-unique ID the audit flags 🔑 is pre-selected as a predictor**, is the default variable in every page-09 test panel, and is offered in Table 1's continuous list. | audit "SEQN 21849 100.0% Unique (potential ID) 🔑"; multiselect default includes `SEQN` | `pages/01_Upload_and_Audit.py:1161` (`default_features = list(feature_options)`) |
| 24 | **low** | **Open-counter and CI coach cards lag one render**, each contradicting content directly below them. | "not opened yet — it opens at Train & Compare" on the render that opened it; "no bootstrap confidence intervals have been computed" above the CI table | `pages/06_Train_and_Compare.py:106-112` vs `:1626` |
| 25 | **low** | **A plural placeholder ships in the chip**, and the sentence says the count twice. | "opened once, at Train & Compare. Training again re-opens the same sealed rows — they have been scored against 1 time(s) already." | `pages/06_Train_and_Compare.py:108-111` |
| 26 | **low** | **The "held back from the predictors" claim lands one render before it is true** — the multiselect on the same screen still lists the column. | "…and this column is held back from the predictors." beside "Selected 28 of 28 features" | `pages/01_Upload_and_Audit.py:1392-1398` (caption) vs `:1101-1106` (withheld list, next render) |
| 27 | **low** | **The constant-columns finding carries a missing-value-sentinel caption.** | "ℹ️ 6 column(s) have the same value in every row … This is a question, not a recommendation — only apply it if you know these values really do mean 'missing'." | Import Doctor render path, `utils/import_ui.py` |
| 28 | **low** | **Table 1 mixes denominators inside one variable block with no note**; percentages sum to 178%. | "meds_chol, n (%) — False 1001 (21.6%) · True 3644 (78.4%) · Missing 17204 (78.7%)" | Table 1 builder (pages 02 and 10 both) |
| 29 | **low** | **Coach cards whose "resolution" is a verbatim restatement of the finding**, one of them with an empty analysis name. | "✅ ~~Ran  on 3 models~~ → Ran  on 3 models" · "✅ ~~Outcome levels encoded alphabetically; 'True' is class 1~~ → Outcome levels encoded alphabetically; 'True' is class 1" | `pages/07_Explainability.py:823`; page-06 encoding insight |
| 30 | **low** | **The plausibility "→ Next" pointer names a control that carries a different label**, and the label uses the disavowed vocabulary. | "plausibility filtering is on Preprocess under Advanced (full control)" → the control is "Domain-specific range filtering" | `ml/eda_actions.py` next-step text vs `pages/05_Preprocess.py:814` |
| 31 | **low** | **The reproducibility manifest reports no commit.** | "Git info: {'app_version': '1.0.0', 'commit': 'n/a'}" | `pages/10_Report_Export.py` state debug |
| 32 | **low** | **p-values render as `0.0000`.** | "p-value: **0.0000** (statistically significant at α=0.05)" | `pages/09_Hypothesis_Testing.py` results block |
| 33 | **low** | **"~190 estimated new features" beside "will create ~209 features"** in the same panel. | polynomial tab metric vs warning | `pages/03_Feature_Engineering.py` |
| 34 | **low** | **Class imbalance is stated in two reciprocal vocabularies across pages** (0.14 vs 7.2:1) with no bridge. | page 02 "ratio=0.14" · page 06 "(ratio: 7.2:1)" | `pages/02_EDA.py` insight vs `ml/imbalance_advice.py` |

### Not findings — verified working, recorded so the next drive does not re-litigate them

- **CV on by default** — `Enable Cross-Validation` renders checked, folds 5.
- **Second `Build Pipelines` press** — identical recipes, one banner, no drift, no duplication.
- **Outcome-encoding disclosure** (page 06) — names both levels, names the ordering rule, names the consequence.
- **All-models default and cost statement** on page 08 — "This run fits 24 models (3 model(s) × 8 seeds)."
- **Achieved seeds are reported**, not just the count.
- **Normality-driven test default, assumption caption, override warning, and the "who chose this" provenance line** on page 09 — the cleanest disclosure loop in the app.
- **External validation record chip and its faithful appearance in the draft.**
- **Pre-export validator** — caught two real defects and disabled the downloads.
- **`AUDIT-042`** — the observation count and the listed observations agreed on this run (14 identified, 8 addressed, 8 listed).
- **Study Design paragraph of the Methods draft**, including the access-count disclosure and the model-selection optimism sentence.
