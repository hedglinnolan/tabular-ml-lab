# Drive 9 — Classic re-drive and merge-gate verdict

**App:** Tabular ML Lab — **Classic** Streamlit pages (`pages/01` … `pages/10`), branch `TurboTab` @ `f84d7b0`
**Dataset:** `_tt_tmp_nhanes.csv` — 21,849 rows × 29 columns (NHANES-style), same file Drive 8 used
**Study driven:** target `meds_hbp` (binary, 71.2% missing), 27 predictors offered, Prediction mode, classification
**Date:** 2026-08-24 (early morning; sequential, single process, one CPU-bound step at a time)
**Interpreter:** `./venv/bin/python` (3.13.0, streamlit 1.60.0, scikit-learn 1.9.0)
**Baseline:** `docs/audit/DRIVE8_CLASSIC_SURFACING.md` @ `8f88b6d` — 4 criticals, 4 highs, 26 mediums/lows
**Repairs under test:** `8697717` (DRIVE-063/064/067), `2c63606` (DRIVE-065/066/069/070), `2da5528` (manuscript
surfaces), `f3ee70f` (ledger), `f84d7b0` (L68 adjudication)

## Method

Headless drive via `streamlit.testing.v1.AppTest`, one page at a time, session state carried forward verbatim
between pages — Drive 8's method, repeated. Raw element dumps for every render are under the session
scratchpad (`drive9/P0*.txt`, `P10_TEXT.txt`); every string quoted below is copied verbatim from them.

**Injected vs driven** — the same caveats Drive 8 declared, plus two probes that are labelled as such:

| Step | Status |
|---|---|
| **Upload of the CSV (page 01, Step 1)** | **INJECTED** into `sp_projects` / `datasets_registry` in the shape `SessionProjectManager.add_dataset` leaves. **Different from Drive 8 in one important way:** DRIVE-067's fix now runs the structural review on the *working table*, so the Import Doctor's finding for `meds_hbp` **did render on this drive** (Drive 8 never saw it). |
| **External-validation file (page 07)** | **INJECTED file object, driven page** — a synthetic 300-row cohort (23% prevalence) written to scratchpad and handed to `ext_val_file` as a real `UploadedFile`. |
| Everything else on 01 (sign-off, mode, target, predictors, subject declaration, holdout) | **DRIVEN** |
| 02 EDA (render, plausibility, VIF, Table 1) · 03 FE (render) · 04 Selection (render + Run + Apply consensus) | **DRIVEN** |
| 05 Preprocess (3 model picks, Build ×2) · 06 Train (Prepare Splits, Train, Bootstrap CIs) | **DRIVEN** |
| 07 Explainability (Run Selected Analyses, forced-failure probe, Subgroup, External Validation) | **DRIVEN** |
| 08 Sensitivity (all models, 8 seeds) · 09 Statistical Validation (two-sample + override) · 10 Report Export | **DRIVEN** |
| **DRIVE-063 zero-completions probe** (`P04_z3/z4`) | **INJECTED STATE, DRIVEN PAGE.** Page 01 now repairs or refuses, so an unusable target can no longer *reach* page 04 through it. The raw object booleans were restored into the carried frames to exercise the branch page 04 still ships. Declared here so it is not read as a driven path. |
| **DRIVE-065 forced-failure probe** (`P07_f1`) | **DRIVEN** — Partial Dependence enabled with permutation importance off, which is a supported in-app configuration that makes every analysis fail. |

Optuna and the Neural Network were skipped, as in Drive 8. The `st.page_link` `KeyError: 'url_pathname'`
harness artifact at the last line of page 05 reproduced and is still not an app defect.

**One methodological difference from Drive 8, and it matters.** Drive 8's feature selection produced nothing,
so it never pressed *Apply*. This drive's selection completed, so the drive applied it (27 → 14 predictors).
That is the route the repairs opened, and three of this report's findings live only on it.

---

## Executive summary

**MERGE-GATE VERDICT: HOLD.**

**All four criticals are dead.** DRIVE-063, -064, -065 and -066 are confirmed killed on the same dataset and
the same route, in the open, with quoted before/after strings. The boolean-with-missing target is now
diagnosed and repaired at page 01 with a caption that names what it did and what it left alone; feature
selection completes with two real methods and a six-feature consensus; explainability runs through the fitted
pipelines and banners its per-analysis outcomes; a forced failure produces a red error, an *expanded* issues
panel, no provenance record and no TRIPOD tick; and the sample-to-feature ratio and the manuscript N now share
one named denominator. Sixteen of Drive 8's mediums and lows are fixed as claimed. The repairs are real work
and most of them are right.

**The gate stays shut for three reasons.**

1. **A new critical false statement reaches the manuscript, on exactly the route the repairs opened.** A
   consensus selection ran, screened 19 numeric candidates, kept 6, carried 8 through, and reduced the
   predictor set from 27 to 14. The Methods draft says: *"All 14 candidate predictors were retained for final
   modeling. Consensus feature selection across LASSO and RFE-CV retained all 14 candidate predictors."* The
   Evidence Map one panel away says *"consensus: 27 → 14 predictors"*. This is Drive 8's finding 1 in mirror
   image: the draft no longer claims a selection that did not run — it now erases a selection that did. The
   provenance record is correct; the manuscript context overwrites it. (The pre-export validator does catch
   it and downloads are correctly disabled — the prose is still on screen and in the preview.)

2. **`DRIVE-068` is marked FIXED in the ledger and is not fixed.** Its note asserts three things — "no false
   rows-repeat claim", "record and sentence agree", "withdrawing a declaration releases the reserved column".
   Driving the declaration refutes all three, verbatim, at `f84d7b0`. A ledger row that says FIXED over a
   defect a five-minute drive reproduces is a worse problem than the defect.

3. **Applying a feature selection silently deletes the dataset profile, and two good surfaces go with it.**
   `dataset_profile` is in `_ANALYSIS_KEYS`, so page 04's *Apply* clears it and nothing on the quick path
   recomputes it. Page 06 then loses the class-imbalance card with its van den Goorbergh citation — the card
   Drive 8 singled out as excellent — and loses the model-suitability badges that `DRIVE-070` was fixed to
   correct, so that fix is unreachable on the route that now exists. Page 10's manifest reads
   *"Dataset profile: Not computed"*.

**What the repairs made worse:** very little, and all of it small. The improbability-band warnings now speak
one vocabulary — and consequently print the same sentence twice near-verbatim instead of twice in two
dialects, so the noise reads as a bug rather than a disagreement. The Limitations list lost its internal
preprocessing splice and gained three coach cards punctuated `finding.: rationale`. The page-01 structural
review lags one render behind its own repair.

**Noise verdict:** improved. The permanent red open-counter banner is gone — it is now a blue info note whose
causal claim is conditional and whose by-design accesses are named. It is long (five sentences on every page
from 08 on) but it is no longer an alarm on a clean run.

---

## Kill-confirmation table — the four criticals

| ID | Drive 8 (before) | Drive 9 (after) | Verdict |
|---|---|---|---|
| **DRIVE-064** target | *"⚠️ LASSO failed: Unknown label type: unknown. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values."* · *"✅ **Feature selection complete! 0 methods run.**"* | **Page 01, repaired in the open:** *"🔧 `meds_hbp` held True/False values in a text column, which scikit-learn cannot read as class labels. It was recoded to **1 (True, 5,527 rows) and 0 (False, 770 rows)**, leaving its 15,552 blank rows blank. This is the same repair the structural review in Step 3 offers; without it feature selection and explainability both fail."* — with an undo-able ledger step *"**2. Recoded 'meds_hbp' to 1/0** — 21,849 × 29"* / *"True → 1 (5,527 rows), False → 0 (770 rows), blanks unchanged."* **Page 04, selection completes:** *"✅ Feature selection complete! 2 methods run."* · tabs *"LASSO · 17/19 kept"*, *"RFE-CV · 6/19 kept"* · *"**6 features** selected by multiple methods"*. **Methods draft:** *"Consensus feature selection across LASSO and RFE-CV retained all 14 candidate predictors."* | **KILLED** — target repaired in the open, real methods completed. **But the Methods sentence does not describe the selection that ran** → new finding **D9-01**. |
| **DRIVE-065** explainability | six errors in a collapsed expander, then *"✅ **Explainability analysis complete (0.4s)**"* unconditionally; two TRIPOD items ticked *"Ran  on 3 models"* | **Success path:** *"✅ Explainability analysis complete (3.7s) — permutation importance (3/3 models), SHAP (3/3 models)."* **Forced-failure path** (PDP enabled, permutation off): expander *"⚠️ 3 issue(s) during analysis"* rendered **expanded**, then *"❌ Explainability produced no results (0.0s): 0 of 3 analyses completed — partial dependence (3/3 models) failed. Nothing on this page describes your models, and nothing was recorded for the report. Open the issues above for the reason."* · provenance `explainability = None` · page 10 on that state: **9/22 TRIPOD items**, 15a `⬜`, 19a `⬜`, readiness *"4 present · 0 inferred/recomputable · 4 missing"* (vs **11/22**, 15a `✅`, 19a `✅`, *"6 present · 2 missing"* on the successful run) | **KILLED** — banner counts outcomes; no green banner and no TRIPOD tick on failure. |
| **DRIVE-063** zero-method record | provenance recorded `consensus_methods=list(methods_to_run)`; draft asserted a consensus over a run that produced none; page said *"Only one method completed"* when zero had | Zero-completions branch driven on an unusable target: *"⚠️ LASSO could not run: 'meds_hbp' holds True/False values in a text column, with 15,552 of its 21,849 rows blank. Stored this way it is neither a number nor a label…"* (same for RFE-CV), then *"❌ **Feature selection did not run.** All 2 requested method(s) (lasso, rfe) failed — see the reason above each. No features were selected, nothing was recorded, and the Methods draft will not report a selection for this run."* Session check: `workflow_provenance.feature_selection is None`; methodology log reads *"No selection method completed; lasso, rfe were requested and all failed"*, `methods_completed=[]`. No green banner anywhere on that render. | **KILLED** — no consensus recorded, the zero is stated, the draft has nothing to report. (Route caveat: page 01 now refuses or repairs, so this state is no longer reachable through page 01 — the probe injected it.) |
| **DRIVE-066** ratio denominator | *"the sample size was large relative to the number of predictors (809:1 observations per predictor)"* — 21,849/27, in a document stating a 6,297-row cohort | **Ratio sentence:** *"Sample size of 6,297 observations — 233 observations per candidate predictor over the 27 screened. No sample-size criterion was evaluated for this analysis, so the count is stated without a verdict on whether it is adequate."* **Manuscript N:** *"A classification analysis was performed on a dataset of 6,297 observations… Of 21,849 available observations, 6,297 remained for analysis after exclusion criteria were applied prior to splitting."* **Abstract:** *"Of 21,849 observations, 6,297 remained for analysis…"* **Validator:** *"Expected analysis N=6297, abstract N=6297, study design N=6297."* PASS. | **KILLED** — one denominator, named, and it matches the manuscript N. Both numerator and denominator are stated (233 = 6,297/27, "over the 27 screened"). |

---

## Re-drive log — the surfaces that changed

### 01 · Upload & Audit

**Structural review now runs on the working table** (DRIVE-067) and its description of the defect is truthful:

> Structural review — Found 2 worth checking.
> ⚠️ **'meds_hbp' holds True/False values in a text column**
> 6,297 value(s), every one of them True or False, stored as text rather than as a yes/no column, with 15,552 blank.
> *Why this matters: Stored this way the column is neither a number nor a label. Modeling, correlation and feature selection all refuse it, and the error they raise reads 'Unknown label type: unknown … on a regression target with continuous values' — which blames the column for being continuous when the problem is how it is stored.*
> [Recode 'meds_hbp' to 1 (True) and 0 (False)]

Drive 8 finding 6 is dead: the self-contradicting *"holds numbers but is stored as text — Every value is a
plain number (e.g. 'True', 'False')"* is gone.

**Seal chip states its denominator** (DRIVE-069), on every page:

> 🔒 Test set: 15% of eligible rows (n=945 of 6,297 rows with a value for `meds_hbp`, stratified) held out since upload — not opened yet — it opens at Train & Compare.

**Not fixed, three renders deep** — the subject-declaration cluster (`DRIVE-068`, Drive 8 findings 7/8/26).
Declaring `SEQN` still produces the same four mutually inconsistent statements, and the session record
disagrees with the sentence that describes it (`test_lockbox['seal_basis'] == 'grouped'`,
`contradiction.kind == 'stated_repeats_but_column_is_unique'`, `rows_per == 1.0`):

> *(info)* 🔒 Rows repeat per subject (`SEQN`), so the held-out set was drawn by **subject**, not by row — 945 rows from 945 subjects.
> *(warning)* ⚠️ **The answer recorded and the data disagree.** `SEQN` was named as the participant identifier, but it has a different value on every one of its 21,849 rows. … until then the seal records that the grain is undetermined, and held-out performance may read better than it is.
> *(chip)* 🔒 Test set: 15% of eligible rows (n=945 rows from 945 subjects, out of 6,297 rows with a value for `meds_hbp`, split by 'SEQN' so no subject appears on both sides) held out since upload

Withdrawing the declaration still does not release the column. Two idle renders after switching back to
*"Let the app work it out"*, with `test_lockbox['group_col'] is None` and the chip back to *"stratified"*:

> Held back from the predictors: `SEQN` — the column the held-out set was split by — giving it to the model hands it the group membership the split exists to hide
> Selected 27 of 27 features

`register_reserved_column` is still what `pages/01_Upload_and_Audit.py:1535` calls; the role-scoped
replacement `utils/combine.py:287 set_reserved_columns` is still unused. See **D9-02**.

### 02 · EDA

- **The target is out of the missingness card** (DRIVE-070): *"⚠️ 1 feature(s) with >30% missing: meds_chol (79%)"*. Drive 8 finding 11 dead.
- **The truncated sklearn exception is gone** (finding 15): *Suggested Interactions* now computes and lists five pairs — `fat_poly × fat_mon (MI gain: 0.0643)` and so on — because the target is readable.
- **VIF carve-out sentence corrected** (finding 13): *"VIF (Multicollinearity) reads the data and reports — it removed, filled and transformed nothing in your dataset. **It IS the answer to 2 observations this page raised, and those are now recorded as addressed by it.** Nothing else is waiting on it."*
- **Column-type counts reconciled** (finding 19): tiles *Numeric 19 · Categorical 8* (= 27), gallery *"Showing 9 of 27 features"*, Macro Shape *"across 19 features"*, page 04 *"8 non-numeric feature(s)"*. One counting rule, held.
- **Table 1 denominators named** (finding 28): *"meds_chol, n/observed (%) — False 1001/4645 (21.6%) · True 3644/4645 (78.4%) · Missing, n/N (%) 17204/21849 (78.7%)"*.
- **Improbability vocabulary unified** (finding 14) — and the duplication is now literal. See **D9-08**.
- **Still counting the same thing two ways:** Relationships *"12 pairs above |r| ≥ 0.8"* · Insights *"2 collinearity cluster(s) affecting 9 features total"* · VIF *"9 feature pairs are already correlated above the flagging threshold"* (`pages/02_EDA.py:2528`, `len(high_corr_pairs)`). Three numbers, two of them labelled "pairs".
- **New disclosure worth keeping:** the target-scatter caption *"Showing a random 5,000 of 6,297 rows · 15,552 rows dropped for missing meds_hbp"*.

### 03 · Feature Engineering (skipped)

Finding 33 fixed: *"~190"* beside *"This will create ~190 new features (209 numeric columns in total)."*

### 04 · Feature Selection

Scope caption right (*"Selection methods see n=5352 training rows"*), categorical carry-through disclosed, the
PROBAST objection intact, and then the step **works**:

> ✅ Feature selection complete! 2 methods run.
> LASSO · 17/19 kept · RFE-CV · 6/19 kept
> **6 features** selected by multiple methods · *Applying will model on the 6 consensus predictor(s) plus the 8 non-ranked feature(s) listed above*

A guard also appeared that Drive 8 could not see: narrowing manual selection to the eight non-numeric
carry-throughs produces *"Feature selection requires at least 2 numeric features."* and no Run button — the
degenerate-config route to zero completions is closed at the door.

### 05 · Preprocess

Coaching panel is `📋 Coaching (1 open, 2 resolved)` and the card advising the investigator to drop their own
outcome is gone. Second **Build Pipelines** press: recipes byte-identical to the first (diffed; only the
sidebar progress counter moved), one banner, no drift. Finding 30 fixed at the destination — the control now
reads **"Plausibility filtering (domain-specific ranges)"**, the word page 02's pointer uses.

### 06 · Train & Compare

- CV on by default, verified. Splits *"Train 4,407 (70%) · Val 945 (15%) · Test 945 (15%)"*, `Total samples: 6,297`.
- **Plural placeholder gone** (finding 25): *"opened once, at Train & Compare. Training again re-opens the same sealed rows — they have already been scored against once."*
- **Chip lag mitigated, not removed** (finding 24): on the render that trains, the chip still reads *"not opened yet"* — now followed by *"Training on this page opens the held-out test set."*, which turns a contradiction into a forward-looking note. The CI coach card is unchanged: *"3 models trained, and no bootstrap confidence intervals have been computed…"* still sits above the CI table on the render that computed them.
- **Two disclosures Drive 8 praised did not render on this route.** The outcome-encoding note (*"'True' is class 1 — the event every metric describes"*) is moot after the recode, which is fine. The class-imbalance card with its van den Goorbergh citation is **absent**, and so are the model-suitability badges. Cause: `dataset_profile is None`. See **D9-03**.

### 07 · Explainability

SHAP routing announced as before; the run succeeds through the fitted pipelines; subgroup analysis produces
per-stratum rows with CIs (`Overall 945 0.8878 · female 493 0.8986 · male 452 0.8761`) and is still
**Accuracy-only with no resampling or multiplicity caption** (finding 18, deferred, unchanged). External
validation runs the full front door and the constant-columns finding now carries its own caption
(finding 27 fixed): *"This is a question, not a recommendation — a column that is the same for everyone may
still be a study-level label you want to keep."*

### 08 · Sensitivity Analysis

All-models default, cost statement (*"This run fits 24 models (3 model(s) × 8 seeds)."*), achieved seeds in
the table, and the sweep declares itself where it happens. **The open counter is the headline change**
(finding 12 fixed) — an `st.info`, not a red warning, with the causal claim made conditional:

> ℹ️ The sealed test set has been **accessed 2 times**, and every access recorded is one this workflow makes by design: Train & Compare; Sensitivity Analysis (seed sweep, re-split over the sealed rows). The headline metrics still come from the single scoring run; the seed sweep re-partitions the sealed rows to measure split sensitivity and reports no held-out performance. **Nothing here records a modeling choice made after seeing a held-out number — but a choice made from now on would put that number into the selection**, and the estimate would read better than it will on new data. The Methods section says the set was accessed 2 times.

Verified identical on 08, 09 and 10; the chip beneath it reads *"**opened 2 times** at Train & Compare,
Sensitivity Analysis (seed sweep, re-split over the sealed rows)"*. Coherent top to bottom.

### 09 · Statistical Validation

Assumption caption, override warning and the "who chose this" provenance line all work as before. **p-values
floor correctly** (finding 32 fixed): *"p-value: **< 0.0001** (statistically significant at α=0.05)"*. The
stale-result-under-the-override-warning render is unchanged (a Mann–Whitney result sits below a warning
announcing a t-test until you re-run). Default variable in every panel is still `SEQN`.

### 10 · Report Export

- **Manifest stamps the commit** (finding 31 fixed): *"Git info: {'app_version': '1.0.0', 'commit': 'f84d7b0'}"*.
- **Readiness matches reality**: *"Permutation Importance ✅ Present · Shap ✅ Present"*, summary *"6 present · 0 inferred/recomputable · 2 missing"*.
- **Abstract invents nothing** (finding 21 fixed): *"Of 21,849 observations, 6,297 remained for analysis after trimming/exclusion criteria were applied prior to splitting. The final modeling set contained 14 predictors."* No feature-engineering stage claimed; validator check 8 passes on *"No reduction language detected."*
- **Limitations lost the preprocessing splice** (finding 17 fixed) and gained three coach cards with broken punctuation — see **D9-06**.
- **Multiplicity fixed in the prose** (finding 20): *"No correction for multiple comparisons was applied across the **1 test** reported here…"* — while the Evidence Map on the same page still says *"Statistical Validation | statistical-test record | **2 test(s)**"*. See **D9-05**.
- **Table 1 states its denominator rule**: *"Each percentage carries its own denominator: category rows are of the values observed for that variable, and the Missing row is of all rows in the column. The two are different denominators and are not additive."*
- **Validator earns its keep again**: 1 of 13 failed — *"Final predictor count is consistent across abstract and methods — Expected predictors=14, abstract=14, predictor section=None."* — and downloads are correctly disabled. That failure is the symptom of **D9-01**.
- **The Study Design paragraph and the model-selection optimism sentence are unchanged and still the best prose in the app.**

**Retired 1/10 narration:** confirmed absent from every rendered surface. The only occurrences at HEAD are in
`tests/integration/test_routing_baseline.py` comments and `docs/turbotab/VALUE_CHECK_ADJUDICATION.md`, where
they belong.

---

## New findings, ranked

Severity uses Drive 8's scale: **critical** = a false statement reaches the manuscript or a number a
researcher would report is wrong; **high** = a surface asserts something the session did not do, or a control
that should exist does not; **medium** = contradiction or stale copy a reviewer would notice; **low** = polish.

| # | Sev | Finding | Quoted surface | Traced to |
|---|---|---|---|---|
| **D9-01** | **critical** | **The Methods draft erases the feature selection that ran.** A consensus selection screened 19 numeric candidates, kept 6, carried 8 through, and reduced 27 predictors to 14. The draft reports no reduction and uses the post-selection count as the candidate count. The Evidence Map, compiled from the same session, states the truth. This is Drive 8 finding 1 inverted. | Draft: *"All 14 candidate predictors were retained for final modeling. **Consensus feature selection across LASSO and RFE-CV retained all 14 candidate predictors.**"* vs Evidence Map: *"Predictor Variables \| feature-selection record \| **consensus: 27 → 14 predictors**"* vs page 04: *"LASSO · 17/19 kept"*, *"RFE-CV · 6/19 kept"*, *"6 features selected by multiple methods"* | `ml/publication.py:290-302` (`_resolve_workflow_feature_counts`) reads `logged_steps['Feature Selection'][-1]`, but the InsightLedger files the *Applied* entry under step `Feature Selection` too — so `logged_steps['Feature Selection Applied']` is empty and the last entry carries no `n_features_before`/`n_features_after`. Both fall back to `len(selected_features)`; `original_count` comes from `data_config.feature_cols`, which page 04 overwrote at apply. Probed live: `feature_counts = {original: 14, candidate: 14, selected: 14, engineered: 0}` while `workflow_provenance.feature_selection` holds `n_features_before=27, n_features_after=14`. `ml/narrative_engine.py:718-722` lets `feature_counts` overwrite the correct provenance ctx; `:993` and `:997` then pick the "all retained" branches. |
| **D9-02** | **high** | **`DRIVE-068` is marked FIXED and is not.** The ledger note claims "no false rows-repeat claim", "record and sentence agree" and "withdrawing a declaration releases the reserved column via the role-scoped replacement". All three are refuted by driving the control at `f84d7b0`. | *"🔒 Rows repeat per subject (`SEQN`)…"* over a column with `rows_per == 1.0` · *"the seal records that the grain is undetermined"* while `test_lockbox['seal_basis'] == 'grouped'` · after withdrawal, `group_col is None` and the chip reads *"stratified"* while the page still says *"Held back from the predictors: `SEQN` — the column the held-out set was split by"* and offers 27 of 27 | `pages/01_Upload_and_Audit.py:1541` prints the rows-repeat info unconditionally whenever `_lb['group_col']` is set; `utils/test_lockbox.py:1370` prints "undetermined" against a `grouped` record; `pages/01_Upload_and_Audit.py:1535` still calls `register_reserved_column` (additive) where `utils/combine.py:287 set_reserved_columns(role=…)` is the replacement. The cited test `tests/test_paper_risk_lockbox.py::TestImport257TheSubjectColumnCanBeDeclared` covers declaration mechanics, not any of the three claims. |
| **D9-03** | **high** | **Applying a feature selection deletes the dataset profile, and two disclosures go with it silently.** `dataset_profile` is cleared by page 04's *Apply*; nothing on the quick path (04 → 05 → 06) recomputes it. Page 06 loses the class-imbalance card **and** its rebalancing control, and loses the model-suitability badges — so `DRIVE-070`'s badge fix cannot be reached on the route the `DRIVE-064` repair opened. No surface says the profile is gone until the state debug panel on page 10. | Page 06 renders **no** *"Moderate class imbalance detected (ratio: 7.2:1)"* card and **no** viability badges on a 4,407-row training set. Page 10: *"• Dataset profile: **Not computed**"* (Drive 8: *"Dataset profile: Available"*) | `utils/session_state.py:475` (`dataset_profile` in `_ANALYSIS_KEYS`) ← `pages/04_Feature_Selection.py:637` `reset_downstream_results(...)`; guard at `pages/06_Train_and_Compare.py:1900` (`if profile and profile.target_profile and profile.target_profile.is_imbalanced`); `ml/model_coach.py:1692` falls back to `profile.n_rows` and is never called. |
| **D9-04** | **medium** | **TRIPOD item 9 is ticked by a dtype recode.** The evidence string changed with the repair; the mis-tick did not. The study drops 15,552 rows for a missing outcome and median-imputes predictors, and item 9's note describes neither. Items 15a and 19a are still both ticked by one explainability ledger line — the empty `"Ran  on"` name is fixed, the mis-mapping is not. | 9 — *Describe how missing data were handled* — ✅ — *"Recoded outcome 'meds_hbp' from True/False to 1/0 (True → 1, False → 0); blank values left blank"* · 15a — *Present the full prediction model to allow predictions for individuals* — ✅ — *"Ran permutation_importance, shap on 3 models"* · 19a — same note | TRIPOD auto-completion mapping in `pages/10_Report_Export.py` reading the methodology ledger by step name |
| **D9-05** | **medium** | **The draft and its own Evidence Map disagree on how many statistical tests were run.** `DRIVE-071`'s override-is-one-comparison fix landed in the prose and not in the Evidence Map row. | Draft: *"No correction for multiple comparisons was applied across the **1 test** reported here…"* · Evidence Map: *"Statistical Validation \| statistical-test record \| **2 test(s)**"* | statistical-test record → `ml/narrative_engine.py` statistical-validation section (deduplicated) vs the Evidence Map row builder (raw count) |
| **D9-06** | **medium** | **Limitations now splices coach cards with a full stop followed by a colon**, three times. Finding 17's preprocessing splice is gone; the same assembly acquired a punctuation defect. The validator's *"No coaching language detected"* check still passes over *"A reviewer would question why the more complex model was selected."* in Principal Findings (Drive 8 finding 16, deferred, unchanged). | *"- Logistic Regression performed within 0.7% of Histogram Gradient Boosting (F1 0.8552 vs 0.8495). A reviewer would question why the more complex model was selected.**:** When models perform comparably, parsimony favors the simpler, more interpretable model."* | Discussion limitations assembly in `ml/narrative_engine.py` / `ml/publication.py`, joining `finding` + `rationale` with `": "` |
| **D9-07** | **low** | **The page-01 structural review lags one render behind its own repair**, and its count and its list disagree on that render: the caption says 1, the expander says 1, and two cards render — the second being the just-repaired `meds_hbp`, with its repair button still offered. Clears on the next render. | *"Structural review — Found 1 worth checking."* / *"Also worth a look (1)"* above *"⚠️ **'meds_chol' holds True/False values in a text column**"* **and** *"⚠️ **'meds_hbp' holds True/False values in a text column**"* | working-table structural review render path (`utils/import_ui.py`) reading a findings list computed before the target repair rewrote the frame |
| **D9-08** | **low** | **The improbability-band duplication is now verbatim.** Unifying the vocabulary (finding 14) removed the disagreement and left two near-identical warnings for each of two facts. | *"⚠️ kcal: 6.3% values outside the NHANES improbability band (800.0-4500.0 kcal) **after conversion from kcal**"* immediately followed by *"⚠️ kcal: 6.3% values outside the NHANES improbability band (800.0-4500.0 kcal)"* (same for `triglycerides`) | two producers still firing: `ml/eda_actions.py` and `ml/eda_recommender.py` / `ml/dataset_profile.py` |
| **D9-09** | **low** | **The external-cohort structural review counts twice in two vocabularies on adjacent lines.** | *"Structural review — Found **1** worth checking, **1** note."* directly above the expander *"Also worth a look (**2**)"* | `utils/import_ui.py` review header vs expander label |
| **D9-10** | **low** | **The decision audit trail mixes a display label and a raw method key in one sentence.** | *"Action: Selected 6 features using **Lasso Regression, rfe**."* | `_feature_selection_method_label` applied to one member of the list and not the other, on the methodology-log path |

---

## Deferred items — spot-check (DRIVE-071's note: findings 16, 18, 22, 23, 24, 26, 34)

| # | Drive 8 finding | Drive 9 status | Evidence |
|---|---|---|---|
| **16** | Coaching language and an internal action id reach the manuscript past validator checks that report the opposite | **Half fixed, half as filed. Does not rise.** The action id is gone (`"Diagnostic analysis via multicollinearity_vif"` — zero occurrences in the whole page-10 text). The coaching sentence still reaches the Discussion and the validator still passes. | Discussion → *"Logistic Regression performed within 0.7% of Histogram Gradient Boosting (F1 0.8552 vs 0.8495). **A reviewer would question why the more complex model was selected.**"* · validator → *"No coaching language patterns remain in export text — PASS — No coaching language detected."* Related to **D9-06**. |
| **18** | The subgroup table reports Accuracy only, no resampling or multiplicity caption | **As filed. Rises slightly.** Unchanged, but on this run the reported accuracy (0.8878) again fails to beat the no-information rate (0.877) — a fact page 06 and the report both state — so the only reviewer-facing subgroup metric is the one the rest of the app disowns. | *"Subgroup N Accuracy 95% CI — Overall 945 0.8878 [0.8617, 0.9059] · female 493 0.8986 · male 452 0.8761"*, three models, no caption |
| **22** | The subject/participant expander never says what the app worked out under its default answer | **Exactly as filed.** Under *"Let the app work it out"* the expander contains only the rationale caption and the selectbox; the record holds a real answer nothing shows. | expander body = caption + `subject_id_declaration` selectbox, nothing else (`P01_r5_final.txt:526-528`) |
| **23** | The 100%-unique ID flagged 🔑 is pre-selected as a predictor, defaults every page-09 test panel, and is offered in Table 1 | **Exactly as filed. Now interacts with D9-02.** All three surfaces reproduce. A careful reader deselecting `SEQN` gets 27 predictors; a reader who instead *declares* it hits the DRIVE-068 cluster. | multiselect default = all 28 including `SEQN`; page 09 `two_sample_numeric` default `SEQN`; `table1_continuous` options begin `SEQN` |
| **24** | Open-counter and CI coach cards lag one render | **Half improved, half as filed.** The chip's contradiction is softened by an added forward-looking clause; the CI card is unchanged. | chip on the training render: *"not opened yet — it opens at Train & Compare. **Training on this page opens the held-out test set.**"* · CI render: *"3 models trained, and no bootstrap confidence intervals have been computed…"* above *"Format: estimate [95% CI lower, upper] via BCa bootstrap (1000 resamples)…"* |
| **26** | The "held back from the predictors" claim lands one render before it is true | **Exactly as filed**, and now also lands one render *after* it stops being true (that half is **D9-02**). | on the declaring render: *"Recorded: `SEQN` identifies a participant … and this column is held back from the predictors."* beside *"Selected 28 of 28 features"* |
| **34** | Class imbalance stated in two reciprocal vocabularies across pages (0.14 vs 7.2:1) | **Not reproducible on this route, for a bad reason.** Page 02 still says *"Class imbalance detected (ratio=0.14)"*; page 06's `7.2:1` card did not render at all, because the dataset profile was cleared. The vocabulary mismatch is unresolved and now hidden behind **D9-03**. | page 02 *"⚠️ Class imbalance detected (ratio=0.14)"* · page 06: no imbalance card, no rebalancing control |

**Verdict on the deferred set:** none has degraded on its own terms. **18** rises slightly (the metric it
reports is the one this run's own coach card disowns). **34** should be re-tested after **D9-03** is fixed —
its status here is "unobservable", not "fixed". **16** and **24** are each half-closed and should be re-filed
as the remaining half rather than left pointing at fixed text.

---

## Not findings — verified working on this drive

- The four criticals, per the kill table.
- **CV on by default**; **second `Build Pipelines` press** byte-identical (diffed).
- **Pre-export validator** caught the one real defect on this run (**D9-01**) and disabled the downloads.
- **Open-counter tone and coherence** across 06 → 08 → 10 (finding 12).
- **VIF carve-out sentence** (13), **column-type counting rule** (19), **multiplicity in the prose** (20),
  **abstract feature-engineering claim** (21), **plural placeholder** (25), **constant-columns caption** (27),
  **Table 1 denominators** (28), **plausibility control label** (30), **manifest commit** (31),
  **p-value floor** (32), **polynomial estimate** (33), and the target's removal from the missingness
  card (11) — all fixed as claimed.
- **Truncated sklearn exception** (15) — gone, because the condition that produced it is gone.
- **Retired 1/10 narration** — absent from every rendered surface.
- **Study Design paragraph** and the **model-selection optimism sentence** — unchanged and still excellent.
- New and good: *"Showing a random 5,000 of 6,297 rows · 15,552 rows dropped for missing meds_hbp"*;
  the ledger step *"Recoded 'meds_hbp' to 1/0 — True → 1 (5,527 rows), False → 0 (770 rows), blanks unchanged"*;
  the degenerate-config guard *"Feature selection requires at least 2 numeric features."*

---

## What the gate needs

1. **D9-01** — the Methods draft must describe the selection that ran. The provenance record already holds
   `27 → 14`; the fix is to stop `feature_counts` overwriting it, or to make `_resolve_workflow_feature_counts`
   read the ledger's *Applied* entry under the step name the ledger actually files it under.
2. **D9-02** — either fix `DRIVE-068` or reopen it. A FIXED row whose note is refuted by driving the control
   is the one thing an audit trail cannot carry.
3. **D9-03** — recompute or preserve `dataset_profile` after a selection is applied, or say on screen that the
   profile-dependent panels are unavailable and why.

`D9-04` … `D9-10` are below the gate and belong in the next umbrella.
