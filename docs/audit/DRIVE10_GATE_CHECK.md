# Drive 10 — targeted kill-confirmation and merge-gate verdict

**App:** Tabular ML Lab — **Classic** Streamlit pages (`pages/01` … `pages/10`), branch `TurboTab` @ `f871fa5`
**Dataset:** `_tt_tmp_nhanes.csv` — 21,849 rows × 29 columns, the file Drives 8 and 9 used
**Study driven:** target `meds_hbp` (binary, 71.2% missing), 27 predictors offered, Prediction mode, classification
**Date:** 2026-08-24 (morning; sequential, single process, one CPU-bound step at a time)
**Interpreter:** `./venv/bin/python` (3.13.0, streamlit 1.60.0, scikit-learn 1.9.0)
**Under test:** `8be8be3` (DRIVE-068, second repair) and `f871fa5` (DRIVE-072/073/074/075 + tail)
**Baseline held against:** `docs/audit/DRIVE9_RECHECK.md` @ `f84d7b0` — verdict HOLD on D9-01, D9-02, D9-03

## Scope and method

This is a **targeted kill-confirmation**, not a re-drive. Drive 9's own routes were re-driven end to end
(01 → 02 → 03 → 04 Run+Apply → 05 → 06 splits/train/CI → 07 explain/subgroup/external → 08 sweep → 09 → 10
draft), because every one of the three gate findings lives downstream of page 04's *Apply* and the ledger,
narrative and profile code all changed under it. Drive 9's drivers were reused verbatim under
`…/scratchpad/drive10/` (copied from `drive9/`, so `SCRATCH` redirects); raw element dumps for every render
are beside them (`P0*.txt`, `P10_TEXT.txt`, `P10_DFS.txt`). Every string below is copied from those dumps.

**Injection caveats are Drive 8's and Drive 9's, unchanged:** the CSV upload is INJECTED into
`sp_projects`/`datasets_registry` in the shape `SessionProjectManager.add_dataset` leaves; the
external-validation file is an INJECTED `UploadedFile` handed to a DRIVEN page. Everything else on 01–10 is
driven. Optuna and the Neural Network were skipped. The `st.page_link` `KeyError: 'url_pathname'` harness
artifact at the last line of page 05 reproduced and is still not an app defect.

The route reproduced Drive 9's exactly: selection ran 2 methods over 19 numeric candidates, kept 6, carried 8
non-ranked through, and *Apply* reduced 27 → 14 predictors. Splits `Train 4,407 · Val 945 · Test 945`,
analysis N 6,297.

---

## Kill-confirmation table

| ID | Drive 9 (before) | Drive 10 (after) | Verdict |
|---|---|---|---|
| **D9-01 / DRIVE-072** — the Methods draft erased the selection that ran | *"All 14 candidate predictors were retained for final modeling. **Consensus feature selection across LASSO and RFE-CV retained all 14 candidate predictors.**"* — with the Evidence Map one panel away reading *"consensus: 27 → 14 predictors"*, and the validator failing check 2 on *"Expected predictors=14, abstract=14, predictor section=None"* with downloads disabled | **Methods → Predictor Variables:** *"The workflow began with 27 predictor variables and retained 14 predictors for final modeling. **Consensus feature selection across LASSO and RFE-CV ranked 19 candidate predictors and retained 6; with 8 non-ranked predictors carried through, the predictor set was reduced from 27 to 14 for final modeling.**"* Both clauses present and both true against page 04 (*"LASSO · 17/19 kept"*, *"RFE-CV · 6/19 kept"*, *"**6 features** selected by multiple methods"*, *"Updated feature set to 14 features: 6 consensus predictors plus 8 non-ranked feature(s) carried through"*). **Evidence Map agrees:** *"\| Predictor Variables \| feature-selection record \| consensus: 27 → 14 predictors \|"*. **Abstract agrees:** *"The workflow began with 27 predictor variables and retained 14 predictors for final modeling."* **Validator:** *"**All 13 validation checks passed.**"*, check 2 now *"Expected predictors=14, abstract=14, **predictor section=14**."*, check 8 *"No reduction language detected."* All eleven download buttons probed `disabled=False`. | **KILLED** — screening and application each get their own clause, the Evidence Map agrees, 13/13 with downloads enabled. |
| **D9-02 / DRIVE-068** — three refuted FIXED claims, on the real file | *(a)* *"🔒 Rows repeat per subject (`SEQN`), so the held-out set was drawn by **subject**, not by row — 945 rows from 945 subjects."* over a column with `rows_per == 1.0` · *(b)* *"…until then the seal records that the grain is **undetermined**…"* under a red *"**The answer recorded and the data disagree.**"* while `seal_basis == 'grouped'`, `contradiction.kind == 'stated_repeats_but_column_is_unique'` · *(c)* two idle renders after withdrawal, `group_col is None` and the chip back to *"stratified"*, while the page still printed *"Held back from the predictors: `SEQN` — the column the held-out set was split by"* beside *"Selected 27 of 27 features"* | **(a) No rows-repeat fabrication.** Declaring `SEQN` renders the mirror case instead: *"`SEQN` **has a different value on every row, so each row is its own subject: the held-out set was drawn by subject and by row at once — 945 rows, 945 subjects.** No subject is on both sides, and `SEQN` is held back from the predictors because a value unique to each row is row identity, not a measurement. If people DO repeat here under some other column, name that one instead."* and the caption *"Recorded: `SEQN` identifies a participant — and it has a different value on every one of the 21,849 rows, so each row is its own participant. The held-out set is drawn by participant and by row at once, nobody appears on both sides, and this column is held back from the predictors."* plus *"`SEQN` has one row per value, so this is a split by subject and by row at once — no subject has a second row to land on the other side."* **(b) No "undetermined" sentence, no contradiction warning.** Probed record on that render: `seal_basis='grouped'`, `basis_source='user_stated'`, `contradiction=None`, `n_test=945`, `n_test_groups=945`. Chip: *"🔒 Test set: 15% of eligible rows (n=945 rows from 945 subjects, out of 6,297 rows with a value for `meds_hbp`, split by 'SEQN' so no subject appears on both sides) held out since upload"*. **(c) Withdrawal releases the reservation on the withdrawing render.** On that very render: *"Selected 27 of **28** features"*, the *"Held back from the predictors: `SEQN`"* caption is **gone**, the chip is back to *"🔒 Test set: 15% of eligible rows (n=945 of 6,297 rows with a value for `meds_hbp`, **stratified**)"*, and the record reads `group_col=None`, `seal_basis='cross_sectional'`, `basis_source='detected'`. Identical on the next idle render. | **KILLED** — all three claims hold on the real file, on the withdrawing render, not one render later. |
| **D9-02 spot-check** — declared-and-consistent state | (not observed in Drive 9) | Declaring `cycle_begin_year` (9 values over 21,849 rows) still reads correctly, and the rows-repeat sentence fires where it is *true*: *"Rows repeat per subject (`cycle_begin_year`), so the held-out set was drawn by **subject**, not by row — 1,298 rows from 2 subjects. Splitting by row would put the same subject in both training and testing."* Record: `seal_basis='grouped'`, `basis_source='user_stated'`, `contradiction=None`, `n_test=1298`, `n_test_groups=2`; chip *"21% of eligible rows (n=1298 rows from 2 subjects … split by 'cycle_begin_year')"* — the honest consequence of moving whole groups. | **CORRECT** |
| **D9-03 / DRIVE-073** — Apply deleted the dataset profile | Page 06 after Apply rendered **no** imbalance card, **no** rebalancing control and **no** viability badges; page 10 read *"• Dataset profile: **Not computed**"* | **Page 06 imbalance card is back:** *"⚖️ Class Imbalance Handling"* → *"**Moderate class imbalance detected** (ratio: 7.2:1). Whether rebalancing is appropriate depends on what this model is for, and that has not been recorded yet. For a risk model or an association estimate it is contraindicated; for a classifier read at a fixed operating point it is defensible. (van den Goorbergh et al., JAMIA 2022;29:1525; replicated for machine-learning methods by Carriero et al., Stat Med 2025.)"* — and it survives training (present again on the post-CI render). **Class-weight control present:** toggle `key=use_class_weight`, label *"Apply class weighting anyway"*. **Viability badges quote the realized training n:** *"✓ **n=4,407** supports the capacity"* (Neural Network card) — Drive 8 finding 10's `n=20,904` is gone from the badges. **Page 10:** *"• Dataset profile: **Available**"*. | **KILLED** |

---

## Spot-checks — the D9 mediums and lows this round claims fixed

| ID | Claim | Result | Evidence |
|---|---|---|---|
| **D9-04** | TRIPOD 9 certified by the imputation record; 15a/19a blank | **FIXED** | Checklist row 9 *"Describe how missing data were handled"* — ✅ — Notes *"**Imputation: median**"* — Page *"Preprocess"* (was: *"Recoded outcome 'meds_hbp' from True/False to 1/0…"*). Row 15a *"Present the full prediction model…"* — **⬜**, blank Notes. Row 19a *"Give an overall interpretation of results…"* — **⬜**, blank Notes. Header and table agree: *"**9/22 items addressed** (auto-completed from your workflow)"* = the nine ✅ rows in the table (4a, 6a, 7a, 9, 10a, 10b, 10d, 13b, 16). |
| **D9-05** | Both count surfaces name their universe | **FIXED** | Draft: *"No correction for multiple comparisons was applied across the **1 test reported here (2 test runs recorded: a comparison re-run under an author override is one comparison, not two)**…"* · Evidence Map: *"\| Statistical Validation \| statistical-test record \| **2 recorded test runs over 1 distinct comparison** \|"*. |
| **D9-06** | No reviewer-anticipation language in Limitations / Principal Findings | **FIXED in the manuscript; validator now states its scope** | Manuscript Limitations: *"…the simpler Logistic Regression performed within 0.5% of the more complex Histogram Gradient Boosting (F1 0.8450 vs 0.8495), so parsimony considerations favor the simpler specification; cross-validation variability (maximum fold SD = 0.0060) exceeded half the between-model performance range (0.0072), so the model ranking should be interpreted with caution; …"* — manuscript register, and the `finding.: rationale` splice is gone from every drafted section. Principal Findings: *"…This pattern suggests that the available predictive signal is largely captured by linear effects, favoring the simpler model on grounds of parsimony and interpretability."* — no reviewer sentence. Validator check 11 now discloses what it does *not* read: *"No coaching language detected in the drafted prose **(the coaching log and decision appendix are the app's own record and are not read by this check)**."* Coach register does survive in the report's own audit sections (*"A reviewer would question why the more complex model was selected.: When models perform comparably…"* under **Key Observations and Resolutions** and **Data Assessment → Open warnings**) — that is the filed `MISC-111` boundary, now stated rather than hidden. |
| **D9-08** | One improbability-band warning, not two | **FIXED** | Exactly one warning per column on page 02: *"⚠️ kcal: 6.3% values outside the NHANES improbability band (800.0-4500.0 kcal) after conversion from kcal"* and *"⚠️ triglycerides: 9.1% values outside the NHANES improbability band (50.0-500.0 mg/dL) after conversion from mg/dL"*. The near-verbatim second copy of each is gone. |
| **D9-09** | Header counts agree | **FIXED** | External cohort, page 07: *"Structural review — Found **1 worth checking, 1 note**."* directly above *"Also worth a look — **1 worth checking, 1 note**"*. One vocabulary, one count. Page 01 likewise: *"Found 1 worth checking."* / *"Also worth a look — 1 worth checking"*. |
| **D9-10** | Method lists use display labels | **FIXED** | Decision audit trail: *"Action: Selected 6 features using **LASSO, RFE-CV**. Rationale: Recorded parameters: feature count 19 -> 6."* (was *"Lasso Regression, rfe"*). The same rule now holds elsewhere: *"Preprocessing was tuned for 3 models: **Random Forest, Histogram Gradient Boosting (Classification), Logistic Regression**."* and *"models=Logistic Regression, Random Forest, Histogram Gradient Boosting (Classification)"* — no raw keys in the appendix. |

### Also re-tested

- **Drive 8 finding 34** — Drive 9 recorded it "unobservable, not fixed" because page 06's card was gone. It is
  observable again and **it reproduces**: page 02 *"⚠️ Class imbalance detected (ratio=0.14)"* against page 06
  *"**Moderate class imbalance detected** (ratio: 7.2:1)"*. Two reciprocal vocabularies for one fact, as filed.
- **D9-07** (low, below the gate) — **unchanged**. On the render that repairs `meds_hbp`, the structural review
  still reads *"Structural review — Found 1 worth checking."* / *"Also worth a look — 1 worth checking"* above
  **two** cards, the second being the just-repaired *"⚠️ **'meds_hbp' holds True/False values in a text
  column**"* with its repair button still offered. Clears on the next render. DRIVE-072..075 fixed the header
  *vocabulary* (D9-09), not the one-render lag.
- **DRIVE-066 / -064 / -065 / -063** (Drive 9's four dead criticals) — nothing on this drive disturbs them.
  Manuscript N and split sum reconcile (*"analysis_total=6297, split_sum=6297"*), selection completes with two
  real methods, explainability banners its outcomes.

---

## New findings

Severity uses Drive 8's scale. Both are below the gate.

| # | Sev | Finding | Quoted surface | Traced to |
|---|---|---|---|---|
| **D10-01** | **medium** | **The sample-size strength claim attributes its verdict to a population no other surface uses, and calls it "training rows".** DRIVE-075 preserves `dataset_profile` through *Apply*, which is right — and it makes `sample_size_claim`'s favorable branch reachable for the first time on this route (Drive 9 got the no-verdict branch precisely *because* the profile had been deleted). The verdict now arrives with a scope note built from `dataset_profile_scope`, which describes the whole profile (`n_rows=20,904` = 21,849 − 945 sealed rows, including the 15,552 rows with no outcome). The same document says the check saw 5,352, and says training is 4,407. Three numbers for one claim's population, in one export. Not a regression this round wrote — the scope machinery is `31e87ac`, pre-Drive 8 — but it is newly on screen because of this round's fix, and it lands under a **Strengths** heading. | Report (.md), Strengths: *"Sample size of 6,297 observations — 233 observations per candidate predictor over the 27 screened, which the data-sufficiency check rated abundant, **computed on training rows only, n=20,904 of 21,849**; held-out test rows are excluded to prevent selection leakage."* vs the same document's Data Assessment: *"Large sample (**n=5,352 observations with a recorded outcome**). All model types are viable."* vs Methods: *"split into **training (n=4,407, 70%)**, validation (n=945, 15%), and test (n=945, 15%)"*. `20,904` appears nowhere else in the export. | `pages/10_Report_Export.py:200-230` `_profile_scope_fields()` returns the profile's own `n_rows`; `pages/10_Report_Export.py:2012` passes it as `verdict_scope_note`; `ml/sample_size_claim.py:149-157` splices it onto the *verdict* clause (`f", computed on {verdict_scope_note}"`). The verdict itself comes from `profile.data_sufficiency`, whose narrative in `ml/dataset_profile.py:982` restricts to rows with a recorded outcome. Probed live: `dataset_profile_scope = {'rows': 'training', 'n_rows': 20904, 'n_rows_total': 21849}`, `profile.n_rows = 20904`, `data_sufficiency = 'abundant'`. Does **not** reach the manuscript LaTeX (its Strengths section carries only Limitations). |
| **D10-02** | **low** | **The report's Limitations bullets are now lowercase sentence fragments.** DRIVE-074 reshaped each limitation into a clause so the manuscript can semicolon-join them, which is right for the manuscript — and the report (.md) prints the same clauses as a bullet list, so three of four bullets open mid-sentence. Drive 9's bullets were capitalized sentences (with the `.: ` splice); the splice is gone and the capitalization went with it. | *"- **the** simpler Logistic Regression performed within 0.5% of the more complex Histogram Gradient Boosting (F1 0.8450 vs 0.8495), so parsimony considerations favor the simpler specification"* · *"- **cross-validation** variability (maximum fold SD = 0.0060) exceeded half the between-model performance range (0.0072)…"* · *"- **classification** accuracy (0.885) did not exceed the no-information rate (0.877)…"* | one clause store feeding two renderers: the semicolon-joined manuscript sentence in `ml/narrative_engine.py` and the bullet list at `pages/10_Report_Export.py` (`limitation_items`) |

### Not new, seen in passing, unfiled

- The report (.md) Limitations list omits the test-set-access limitation that the manuscript's list carries
  first (*"the held-out test set was accessed 2 times during model development rather than once at the end…"*).
  Present identically in Drive 9; two producers, one shorter than the other.
- The decision audit trail records the seed sweep as *"model=Logistic Regression; n_seeds=8"* while page 08 ran
  and displayed all three models (*"This run fits 24 models (3 model(s) × 8 seeds)."*). Byte-identical in
  Drive 9's export; page 08 was not touched by this round.
- *"It IS balanced on nothing."* still closes the grouped-seal balance warning. Pre-existing (verbatim in
  Drive 9); this round correctly replaced the *reason* clause that preceded it
  (*"too few people in some combination of those groups"* → *"a held-out set drawn by `SEQN` moves whole
  subjects at a time, and cannot also be balanced"*).

---

## Sweep lens — did this round break anything on the surfaces it touched?

Pages 04, 05, 06, 10 drafts and the page-01 declaration block were read render by render against Drive 9's
dumps.

- **Page 01 declaration block** — five states driven (undeclared, declared, declared-idle, withdrawing,
  withdrawn-idle). Every rendered sentence matched the probed record in all five; no state produced a claim the
  record contradicts. `contradiction` is `None` throughout, including on the unique-per-row declaration where
  Drive 9 found a fabricated one.
- **Page 04** — scope caption, categorical carry-through disclosure, PROBAST objection and the
  degenerate-config guard all unchanged and correct; the *Apply* banner now states both halves
  (*"Updated feature set to 14 features: 6 consensus predictors plus 8 non-ranked feature(s) carried
  through."*). Nothing new reads wrong.
- **Page 05** — coaching panel `📋 Coaching (1 open, 2 resolved)`, second **Build Pipelines** press clean, the
  plausibility control label unchanged. Nothing new.
- **Page 06** — gained back the imbalance card, the class-weight toggle and the viability badges (D9-03); the
  badges now quote 4,407 rather than 20,904. The chip's forward-looking clause and the CI coach-card lag are
  unchanged from Drive 9. Nothing new reads wrong.
- **Page 10** — the two findings above are the only things this drive found that Drive 9 did not, and neither
  is a regression the round wrote: **D10-01** is pre-existing code newly reachable through the fix, **D10-02**
  is a cosmetic side effect of the correct manuscript rewrite. Everything else on the page improved: validator
  13/13 with downloads enabled, TRIPOD ticks name their real evidence and go blank where nothing certifies
  them, both test-count surfaces name their universe, method lists use display labels, the manifest reads
  *"Dataset profile: Available"*.

---

## MERGE-GATE VERDICT: **OPEN**

All three findings Drive 9 held the gate on are dead, confirmed on Drive 9's own routes with the same dataset
and quoted after-strings:

1. **D9-01 / DRIVE-072** — the Methods draft describes the selection that ran, in two clauses that separate the
   screening from the application; the Evidence Map agrees; the validator passes 13/13 and the downloads are
   enabled.
2. **D9-02 / DRIVE-068** — all three refuted claims now hold when the control is driven on the real file, and
   the release happens on the withdrawing render rather than never. The ledger row and the app agree.
3. **D9-03 / DRIVE-073** — the dataset profile survives *Apply*; page 06's imbalance card, class-weight control
   and viability badges are back, and the badges quote the realized training n.

All six spot-checked mediums and lows (D9-04, -05, -06, -08, -09, -10) are fixed as claimed.

**Two new findings, both below the gate.** **D10-01** (medium) is the one to take first in the next umbrella:
it is a claim under a *Strengths* heading whose stated population (20,904) is neither the analysis cohort
(6,297), nor the population the check it cites reports (5,352), nor the training split the Methods names
(4,407) — and it became visible only because DRIVE-075 did the right thing. It does not reach the manuscript
LaTeX, which is why it does not hold the gate; if the gate's rule is read strictly as *"no exported claim may
misstate its population"*, this is the row that would reopen it. **D10-02** (low) is punctuation.

**Still open below the gate, carried forward:** D9-07 (structural review lags its own repair by one render),
Drive 8 finding 34 (imbalance stated as 0.14 and 7.2:1 — now observable again and reproducing), D9-16/18/22/23
/24/26 as Drive 9 filed them, and the three "not new, unfiled" items above.
