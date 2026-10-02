# TurboTab Next: validation audit

*2026-10-02 · branch `turbotab-next` at `bf3db6c` · synthesized from nine area audits (A–I) and nine independent refutation passes. The repository was read-only to every auditor; only this folder was written.*

## How to read this report

Every earlier review was **verification**: does the code do what the spec says? This audit is **validation**: is the spec itself right? Each area auditor treated every spec, pack claim, threshold and inherited rule as a hypothesis. It ran the app's own code on adversarial fixtures and read primary sources (papers, guidelines, agency documentation), quoting the sentence it relied on. A second auditor, the "skeptic", then tried to refute each critical and major finding independently: its own fixtures, its own seeds, its own reading of the sources.

**Severity.** *Critical*: a researcher could publish a wrong number, an invalid inference, a mislabeled estimand, or leaked results because of it. *Major*: methodologically weak or unjustified, a claim unsupported or overstated, a missing option a methods reviewer would expect, or a leash clearly too tight or too loose. *Minor*: everything else.

**Layers** follow your chain: sound **math** feeds sound **methods** feeds sound **intelligence** (what the app detects, recognizes and claims) feeds sound **routing** (what it asks, in what order, with what guidance). Presentation was not audited as a layer of its own. Where a caption or figure mislabels a number, the finding sits in the layer whose fix it needs.

**The areas.** A: math of scoring and the seal. B: math of transforms (energy, substitution, aggregation, repairs). C: math of data (ingest, reshaping, counts). D: nutrition methods. E: validation and inference methods. F: omics and survey methods. G: claims the app makes. H: detectors and recognizers. I: routing.

**Files in this folder**
- `AUDIT_REPORT.md`: this report.
- `findings.json`: all 170 raw findings, each with the skeptic's verdict, its final severity and the deduplicated ID it maps to here.
- `claims-ledger.md`: area G's ledger of 101 user-facing claims, each checked against a source or against the code.
- `repro.tar.gz`: the 239 reproduction scripts (and four captured outputs) the auditors and skeptics ran, preserved because their working folders under `/private/tmp/turbotab-audit/` will not survive a reboot. Fixture data, the extracted text of papers and one vendored PDF library are not included. They were written for their working folders: run them from the repository root with `PYTHONPATH=. ./venv/bin/python <script>`; some regenerate their own fixtures, while others read fixtures from `/private/tmp/turbotab-audit/` or `turbotab/sample_data/` and need those paths regenerated or repointed.

---

## 0 · At a glance

| | |
|---|---|
| Raw findings | **170** (A 21 · B 24 · C 14 · D 21 · E 16 · F 17 · G 20 · H 19 · I 18) |
| Independently re-tested | **122**, every critical and major one. **All 122 were confirmed; none was refuted or left unconfirmed.** 113 were confirmed as stated. 9 were confirmed with a different severity: 4 raised to critical (B7, B11, C8, G13) and 5 lowered to minor (C9, D8, G4, G12, H8). |
| Not re-tested | **48**, every one rated minor by its auditor. They are listed in §7.5 as not established. |
| After deduplication | **77 distinct critical or major problems: 23 critical, 54 major**, plus 4 confirmed minors. |

| Layer | Critical | Major | Total |
|---|---:|---:|---:|
| Math | 5 | 14 | 19 |
| Methods | 9 | 10 | 19 |
| Intelligence | 5 | 21 | 26 |
| Routing | 4 | 9 | 13 |
| **All** | **23** | **54** | **77** |

**The ten that matter most** (each would put a wrong number or a wrong label into a paper on an ordinary path):

1. **MA-01.** When the app knows a participant appears in several rows but the seal could not group them, the inference table treats every row as independent. A null exposure in a 6-person, 40-visit feeding study came out at p = 4 × 10⁻¹⁴ with no warning.
2. **ME-02.** "No energy adjustment" keeps total energy in the model (it is the standard model) while the estimand, the methods sentence and the coach all say "absolute intake".
3. **ME-03.** The "Willett residual" drops total energy from the outcome model but is labeled "the same substitution as the standard model". With an ordinary covariate such as sex, the coefficient differs from the standard model's and can change sign.
4. **ME-01.** "Impute" is a single median fill for every purpose. Under inference, a confounder 35–44% missing at random gave a 38–75% biased coefficient whose 95% interval never covered the truth. The methods sentence does not say "median".
5. **ME-06.** With NHANES weights, strata and PSUs present, inference runs unweighted with simple-random-sample intervals and no concern on the result. On an informative-weight fixture the sign of the estimate flipped.
6. **RO-02.** Repairing three impossible outcome values after the split silently re-drew the held-out set: 119 to 130 rows whose outcomes had trained the fits the user already inspected became "sealed" behind a clean lock.
7. **RO-04.** Under inference the app never asks for the exposure of interest, the estimand or the adjustment set. Default roles adjust for likely mediators (BMI, waist, lipids, blood pressure) and mutually adjust every nutrient.
8. **IN-01 and IN-02.** Recognizers read substrings. Body fat mass, CRP and fibrinogen become energy-bearing nutrients and default adjustment targets. A trial's treatment arm becomes "an acquisition column (batch, plate or run order)" and is dropped from the model.
9. **ME-09.** Omics counts and intensities enter the models raw. On null genes with cases sequenced 1.45× deeper, the app reported a cross-validated AUC of 0.83 to 1.0.
10. **RO-03.** Censored follow-up is never recognized. A cohort with staggered entry and no true effect produced a "protective" fiber association at p ≈ 10⁻¹³ to 10⁻²² (Cox: HR 1.01, p = 0.12–0.6).

**The fix plan in one line.** Eighteen work packages in strict layer order (§5): first make the data mean what the file says and make every interval match how the data were sampled; then make every method's label equal the model fitted and route missing data, survey design and validation by purpose; then make recognizers and detectors pass must-not-fire fixtures and make every claim match its source; last, make the seal hold at its edges and let the declared purpose change what is asked.

---

## 1 · Verdict by layer

### 1.1 Math: the arithmetic is right; the intervals and the reshaping are not

**What holds.** Where the app computes a number, it computes it correctly, and the discipline around it is sound. AUC, Brier and log loss use the event you declared and match scikit-learn exactly. Everything learned from data (imputation, energy adjustment, scaling, tuning) is refit inside every cross-validation fold, and the held-out rows are scored by a pipeline that never saw them. A grouped split keeps every participant wholly on one side. The chronological holdout keeps participants whole. Elastic-net coefficients come back in raw units to 15 decimal places. The residual algebra, the Atwater factors (4/4/9/7, fiber 2) and the kJ factor match FAO. The SAS-zero detector is exact. Row identity survives a 1.5-million-row file read in parallel. The Hanley–McNeil formula behind the seal's AUC precision is right, and the 100-event floor matches Vergouwe 2005 and Collins 2016.

**What does not.**
- **The intervals do not always describe how the data were sampled.** When an identifier repeats but the seal's grouping was abandoned or undetermined, the coefficient table treats rows as independent: 83% coverage for a nominal 95% interval, and p = 4 × 10⁻¹⁴ for a null exposure in a 6-person study (MA-01). When it does cluster, it uses normal critical values from as few as 8 clusters: 81% coverage at 8 (MA-06). There is no heteroskedasticity-robust option; when outcome variance grows with intake, classical intervals cover 59–85% (MA-07). A perfectly predictive exposure prints p = 1.0 (MA-08).
- **Reshaping the table can change the numbers and then describe itself wrongly.** Combining each person's rows by a text visit label or a non-ISO date silently uses file order, while the receipt says it ordered by that column; a change from baseline came out with the wrong sign (MA-03). Undated visits sort last. Turning a metabolomics export makes its m/z and retention-time columns into two extra "participants" (MA-04). US dates coarsened to the first of the month are all read as January (MA-05). Combining with "change" turns sex into zero for everyone; "mean" averages integer category codes into 1.67 (MA-14).
- **Ordinary file spellings break columns.** A SAS or Stata "." for missing turns age into a 61-level category and drops total energy as "free text" (MA-16). Excel and CSV copies of one table disagree about whether the answer "None" is missing (MA-17). One infinite value stops the analysis with an engine error (MA-18).
- **The within-sex energy residual manufactures an effect.** On a fixture where the nutrient has no effect, it reported p ≈ 10⁻¹⁰⁶ when sex was not also in the model (MA-02).
- **Some estimators are biased or mis-sized.** Cross-validated R² is averaged fold by fold, which cuts it by more than half at n = 100 (MA-09). The substitution curve's "95%" band is 2–2.5× too wide at 10,000 rows and covers only about 90% at 1,500 rows (MA-12). The "better than baseline" check says yes on 25–31% of pure-noise binary datasets, so the app is silent when a model has learned nothing (MA-10).

**Bottom line.** Trust the arithmetic. Do not yet trust an interval under inference when participants repeat, when clusters are few, or when variance grows with intake, and do not trust a table whose rows were combined by a text time column or that was turned from features-by-samples.

### 1.2 Methods: the labels do not always match the model, and inference borrows prediction's machinery

**What holds.** The fold discipline and nested tuning are textbook. The lockbox mechanics (withhold, open once, refuse structural changes after sealing) are sound for prediction. The app correctly offers no SMOTE (van den Goorbergh 2022). The Tomova-derived caveats for the standard, partition and density models are faithful. The partition refusal, the nested-nutrient substitution shift and the whole-pipeline bootstrap are coherent. Exclusion screens are never pre-selected and show their row counts. The repeats menu recommends the mean for replicates and refuses a default for time points. Outcome-free imputation and missing indicators are defensible for prediction, and complete cases stay on the menu.

**What does not.**
- **Several labels name a model the app did not fit.** "No energy adjustment" is the standard model (ME-02). The "residual method" drops energy but claims the standard model's estimand; with ordinary covariates the coefficient differs and can flip sign (ME-03). With several nutrients in the model, each coefficient is a swap for whatever energy source was left out, labeled a swap for "the average of all other sources" (ME-04). Logistic coefficients are log-odds captioned "change in predicted outcome" (ME-07). A total beside its own parts (fat beside SFA, MUFA, PUFA) estimates only the unclassified remainder, unlabeled (ME-15).
- **Inference runs on prediction's machinery.** Median fill is the only imputation, with no multiple imputation, and the methods sentence hides it (ME-01). Survey weights, strata and PSUs are ignored without a concern (ME-06). The split still holds out 20% and the coefficients use only the training rows, so a published p-value depends on a random seal: "significant" in 9 of 20 seeds on NHANES (ME-12). The substitution curve never checks that every energy source is in the model: 57% bias with two of four sources (ME-05).
- **Omics data are modeled raw.** Counts and intensities get no normalization or log transform, so sequencing depth or urine dilution alone produces a "signature" (ME-09). Values below detection have no detection-aware option; complete cases keep 0 of 72 samples and the median fill puts non-detects above the smallest detected value (ME-08).
- **Options a methods reviewer expects are missing.** No calibration intercept, slope or curve and no interval on any performance metric (ME-10). No repeated cross-validation or bootstrap optimism correction (ME-11). No declared final model before opening (ME-13). The all-components model is not named (ME-14). No Goldberg screen or with/without-exclusion sensitivity analysis (ME-16). No splines or quintiles (ME-17). No sound inference family for p ≥ n (ME-18). No ordinal model (ME-19). No time-to-event model (RO-03). No mixed model or GEE (MA-01).

**Bottom line.** For prediction the core is sound and the gaps are reporting (calibration, intervals, optimism correction). For inference the app currently produces numbers whose labels, intervals or populations a nutrition reviewer would reject, on its default paths.

### 1.3 Intelligence: honest about its gaps, but its recognizers read substrings and some detectors fire on clean data

**What holds.** The app says plainly what it does not do (no ordinal model, no usual-intake modeling, no Goldberg, no survey weighting, polychoric not computed). Most teaching claims check out against their sources (Tomova, SMOTE, MAQC-II, feature-selection leakage, missing indicators, NHANES weights and variance, reference intervals). The SAS-zero detector, the mixed-units factor table, the nesting design, kJ detection when macronutrients are present, and NHANES 7/9 sentinel detection all work. Seal, grain, aggregation and orientation sentences report what was drawn.

**What does not.**
- **Recognizers read names as substrings.** "fat" matches fat mass, body fat percent and fatigue; "prot" matches C-reactive protein; "fib" matches fibrinogen; "carb" matches bicarbonate and carbamazepine. These become energy-bearing nutrients and default energy-adjustment targets (IN-01). The "design" role swallows a trial's treatment arm, condition, fasting status, batch and site, and drops them from the model with the reason "an acquisition column (batch, plate or run order), not biology" (IN-02). `steps_per_day` becomes a time column; birth weight in grams becomes a "sampling weight" (IN-10). Three identifier recognizers disagree; UK Biobank's `eid` is missed while `site_id` is called a person identifier (IN-06). Common NHANES and INFOODS names are missed in the other direction (IN-08).
- **Some detectors fire on clean data and offer a one-click destructive repair.** The survey detector calls the top answer of a 6-point scale "almost certainly" a missing code and offers to blank 16% of answers (IN-03). The structural detector calls a diastolic pressure of 99 or a glucose of 66 a missing code in 2–15% of clean columns (IN-04). The run-order drift detector fires on drift-free data in every 16-injection study (IN-17). One concentrated sample makes 300 independent metabolites read as 40 (IN-18).
- **Some readings are confidently wrong and never asked.** The outcome unit is guessed from the name and written into estimand sentences, with no way to correct it: dietary choline in mg/day becomes "mg/dL" (IN-05). Log-scale feature-by-sample tables are never asked about orientation (IN-11). Repeats versus time points is read from visit spacing, so a daily feeding time course is averaged (IN-12). Lens hints send most normalized genomics and survey tables to "metabolomics" (IN-13).
- **Some claims overstate or are wrong.** "That adjustment is needed is not in dispute" (IN-20). "Attenuated, but unbiased in direction" (IN-22). Too-numerous-to-count taught as a measurement failure rather than a censored value (IN-23). The multivariable density model presented as clean "diet composition" (IN-21). "Scored on later data" for a holdout whose rows are mostly earlier (IN-24). A lineage figure that shows operations that did not happen (IN-25). Clinical plausibility bands labeled an "NHANES reference" that are unsourced demo defaults (IN-09).

**Bottom line.** The app's honesty about what it lacks is a real strength. Its recognizers and several detectors need corroboration rules and must-not-fire fixtures before their outputs can steer defaults.

### 1.4 Routing: the order is disciplined; the declared purpose barely changes anything, and the lockbox leaks at its edges

**What holds.** Questions are answered in order, structural answers are refused after the seal, the seal opens once and only on a fresh fit, the event question has no default, the purpose question has no default, grain contradictions are caught with exits, "I don't know" is honored as exploratory, and exclusions and missing-value answers never move a row across the seal. Below the 100-row or 100-event floor, cross-validation leads.

**What does not.**
- **The lockbox leaks at its edges.** An eligibility rule on the outcome itself is accepted, and its preview draws the outcome's histogram with the cut marked; the rule cut a true slope to between a half and a third of its value (RO-01). Repairing outcome values after the split re-draws the held-out rows silently (RO-02). After opening, a new seed or a new outcome serves fresh held-out scores at once, while the screen says the opened score "is then fixed in the record" (RO-05). Under inference, coefficients and p-values are live while the plan is still being changed (RO-12).
- **The declared purpose barely changes what is asked.** No gate or validator reads it (RO-06). Inference never asks for the exposure, the estimand or the adjustment set (RO-04), the survey design (ME-06), censoring (RO-03) or clusters above the person (RO-08). The energy menu puts the residual method first even for prediction, where it discards the strongest predictor (RO-07). No option carries the "customary in" and "sound for" labels that north star 5 requires.
- **Some choices are skipped as facts.** Ordered outcome levels are read as unordered classes, and skewed biomarkers get no question about scale (RO-10). A bare visit index is stated as replicates (IN-12). The lens question has no "not sure" (RO-11). Combining time points accepts predictors summarized after the outcome (RO-09).

**The leash, overall.** Almost every decision sits at "rank and state the concern" or below for both purposes. Under inference, where a wrong answer becomes a published number, the right rung is usually "block and record", and for outcome-based eligibility, censoring ignored and post-seal re-draws it is "refuse". Under prediction the current leash is closer to right, but the menu is too short (no bootstrap, no calibration). §4 grades every decision.

---

## 2 · Confirmed critical and major findings, deduplicated

Each entry gives the deduplicated ID, the final severity, the raw findings it merges (with the skeptic's verdict), the evidence in brief, and the recommendation. File references are `path:line` at `bf3db6c`. Where the auditor and skeptic measured the same quantity on different fixtures, both numbers are given. Every source quote was read in the primary source by at least one of the two, unless marked otherwise.

### 2.1 Math

#### MA-01 · critical · Repeated rows are analyzed as independent when the seal could not group them
*Merges A1 and E1, both confirmed critical by their skeptics.*

**Evidence.** The seal returns no grouping column when grouping is *abandoned* (fewer than 8 units, a named column absent, or "one row each" while an identifier repeats) or *undetermined* (grain answered "I don't know") (`core/seal.py:184-222`; the basis still names the repeating column, but the column it hands on is `None`). `fit_stage` then passes `groups=None` (`stages/modeling.py:382,399`), `statsmodels_fit` fits plain OLS or Logit (`models/linear.py:75-90`), and the cluster note (`modeling.py:470-473`) never fires. The seal's "exploratory" caution covers held-out scores only; nothing reaches the coefficient table.
- 300 people × 3 rows, no true association: 95% interval coverage **0.83** (skeptic: 0.84; 100 × 5: **0.79**) against 0.95 when clustered.
- Real server, 6 participants × 40 visits, null sodium, purpose = inference, grain = repeated: basis "abandoned", sodium p = **4.1 × 10⁻¹⁴**, concerns `[]`. A random-intercept model on the same rows: p = 0.85. Skeptic, 7 × 30, null potassium: p = 1.2 × 10⁻⁹ against 0.71 (mixed model); a participant-level null covariate at p = 3 × 10⁻¹⁴.
- Monte Carlo with the app's own fit, null effect: type-I error 0.44–0.67 (OLS) and 0.29–0.55 (logistic) for 6–20 clusters (skeptic, unbalanced clusters: 0.22–0.42).
- Cameron & Miller 2015 (*J Hum Resour* 50:317): "Failure to control for within-cluster error correlation can lead to very misleadingly small standard errors, and consequent misleadingly narrow confidence intervals, large t-statistics and low p-values."

**Recommendation.** Under inference, whenever an identifier is known to repeat, cluster the intervals by it regardless of how the seal was drawn (the CI's grouping does not depend on the seal's), and say so; or block the table and record "intervals assume independent rows, but `id` repeats". Below about 8–10 units, refuse cluster-robust intervals and offer exits: combine to one row per person, or a random-intercept mixed model or GEE (to be added; see WP12). **Leash: too loose.** The app knows the rows repeat and is silent on the result.

#### MA-02 · critical · The within-sex energy residual manufactures an exposure effect
*B1, confirmed critical.*

**Evidence.** `StratifiedEnergyAdjuster` fits one residual regression per level and adds back each level's own predicted nutrient at that level's mean energy, then drops the strata column when it is not a predictor (`models/steps.py:69, 81-82, 110-111, 134`). On a fixture where fat has **no** effect and the outcome depends on sex: per-level r(N_adj, E) ≈ 10⁻¹⁶, but pooled r(fat_adj, male) = 0.56–0.58 and r(fat_adj, E) = 0.28–0.32. Y ~ fat_adj gives b = 0.06–0.11 with p = 10⁻²⁰⁹ to 10⁻¹⁰⁶ (truth 0). Keeping sex in the model: p = 0.16; one common constant: p = 0.94. The pack contradicts itself: NUTRITION_PACK §04 line 446 says add "the predicted nutrient at the cohort mean energy"; line 480, which the code follows, says "at the sex-specific mean energy". `strata_candidates` (`stages/proposals.py:227-243`) offers sex even when it has no role, and no validator or warning checks whether the strata column is a predictor.

**Recommendation.** Add the pooled predicted nutrient at the overall mean energy (pack line 446), or force the strata column into the model, or refuse strata that are not predictors. Report the pooled r(N_adj, E) and r(N_adj, strata), not only per-level values. Correct pack line 480. **Leash: too loose** (offered silently).

#### MA-03 · critical · Combining a person's rows uses the wrong order, and the receipt says otherwise
*B11 (raised from major to critical), C2 (critical), C8 (raised from major to critical).*

**Evidence.** `rank_units` orders by `TRY_CAST(time AS TIMESTAMP)` (`stages/working.py:781-789`). Text visit labels (`baseline`, `month_6`, `V10`) and non-ISO dates (`Mar 3, 2021`, `03/14/2021`, `14-Mar-2021`) cast to NULL, and the order falls back to file order (`working.py:805`). The receipt still records `ordered_by` = that column (`working.py:927`), and the Record says "in order of `visit_date`" (`voice.py:937-939`). The structure stage itself reads such dates with pandas' lenient parser, so it proposes them as the time column (`repeats.py:129-141`).
- Visits stored month_12, baseline, month_6: "first" keeps month_12 for everyone; LDL change = **−1.0** where the true baseline-to-month-12 change is **+2.0**.
- Text dates: participant A's weight change **+5.0** where the chronological change is **−10**.
- An undated record sorts last (`ORDER BY … NULLS LAST`): "last" takes it (weight 77 against the latest dated 74), and change is −3 against a true −6. The receipt has no count of undated records. (The seal, by contrast, counts undated units explicitly.)

**Recommendation.** If the cast yields NULL for most non-missing values of the chosen time column, refuse with exits: choose another column, or declare the order of the levels. Exclude undated records from first, last and change (or ask how to place them) and count them in the receipt. Never record as `ordered_by` a column that ordered nothing. **Leash: too loose** (silent fallback, false receipt).

#### MA-04 · critical · Turning a feature-by-sample table makes annotation columns into participants
*C1, confirmed critical; F4 (transposition part).*

**Evidence.** `working.py:446` makes every column except one label column into a sample. On a metabolomics export (metabolite, mz, rt, S1…S12) the rows `mz` and `rt` become two samples (14 instead of 12), with no refusal. With a text annotation (`hmdb`) also present, every feature column becomes text. With numeric Entrez IDs the label is not found (`working.py:362-375` skips numeric columns), `entrez_id` becomes a sample whose values are the gene IDs, and the features are renamed `row_0 … row_79`. The orientation preview (`structure_previews.py:458`) shows a different table from the one the stage writes. MZmine's `row m/z` and `row retention time` columns become samples too. (The orientation reading was "undetermined" in these cases, so the user had to declare the turn; nothing then guarded it.)

**Recommendation.** Before turning, partition columns into the label, feature annotations (non-numeric, or numeric with a feature-level meaning such as m/z, RT, IDs) and sample columns; show that partition as the preview, computed by the same code as the stage. Refuse to turn, with an exit to mark annotation columns, when a non-label column is not a measurement block, or when no label is found and the first column is integer-valued and unique. **Leash: too loose.**

#### MA-05 · critical · Ambiguous dates are read day-first without a word
*C3, confirmed critical.*

**Evidence.** Ingest applies DuckDB's sniffed date format with no ambiguity check (`core/datastore.py:531-558, 578-582`). US month-first dates coarsened to the first of the month, a common de-identification: 1,840–1,843 of 2,000 dates change, every one lands in January, warnings `[]`. Downstream, quarterly visits read as 3 days apart, so the repeats reading becomes "repeats" and the time-points question is skipped as "not asked"; the ISO copy of the same data reads 91-day time points. Only columns in which every value is ambiguous are affected; any day above 12 resolves the format correctly.

**Recommendation.** After sniffing, test both month-first and day-first on every value. If both parse everything, ask once (three example rows under each reading) or keep the column as text until answered, and record the chosen format in the ingest warnings and the methods. **Leash: too loose.**

#### MA-06 · major · Cluster-robust intervals use normal critical values, from as few as 8 clusters
*A3 and E5, both confirmed major.*

**Evidence.** `linear.py:81-82` fits `cov_type='cluster'` without `use_t`, so statsmodels uses z: half-width/SE = 1.96 where t(9) would give 2.262. Grouping starts at 8 units (`utils/test_lockbox.py:70`, read by `seal.min_groups()`). Coverage of the app's nominal 95% interval for a cluster-level exposure at G = 8/12/20/50: 0.81/0.87/0.90/0.93 (with t(G−1): 0.87/0.90/0.91/0.93). Type-I error at G = 8: 0.17 for OLS and 0.18 for logistic. Even with t(G−1), the skeptic's type-I error stayed at 0.12–0.15 for G = 8–12. The concern string says the intervals are "cluster-robust", which reads as reassurance exactly where it is least warranted. Cameron & Miller 2015: "at a minimum one should use the T(G − 1) distribution rather than the standard normal"; "'few' may range from less than 20 clusters to less than 50 clusters in the balanced case"; "The best methods use the CR2VE and T(v*)".

**Recommendation.** `use_t=True` with G − 1 df at minimum; CR2 (Bell–McCaffrey) with Satterthwaite df, or a wild cluster bootstrap, below about 30–50 clusters; state "cluster-robust, G = n" in the caption; recommend a mixed model below about 20 clusters. **Leash: too loose.** The skeptic notes this is close to critical at the 8–12 clusters the app allows.

#### MA-07 · major · OLS inference offers only homoskedastic standard errors
*A4, confirmed major.*

**Evidence.** `linear.py:79-85` fits OLS with no robust covariance unless rows repeat; there is no option and no check (the legacy app's heteroskedasticity detector, COACH-003, was not carried over). Null slope on a lognormal intake: with error SD proportional to intake, classical coverage 0.67/0.61/0.59 at n = 50/200/1,000 against HC3's 0.90/0.93/0.95; with variance proportional to intake, classical 0.77–0.85 against HC3 0.91–0.95. The primary (Long & Ervin 2000) was not readable; two secondary sources report its recommendation of HC3 below n ≈ 250. The finding rests on the reproduced simulations.

**Recommendation.** Under inference default to HC3 (HC1 at large n) and say so in the caption, or at least offer it ranked first; add a residual-variance check that raises a concern. **Leash: menu too tight, guidance too loose.**

#### MA-08 · major · A perfectly predictive exposure prints a huge log-odds with p = 1.0
*A14, confirmed major.*

**Evidence.** Rare binary exposure (8%), every exposed row an event: the table shows log-odds 21–37, an interval of ±10⁴ to ±10⁷, and p ≈ 0.997–1.0 in 14 of 20 runs; in 6 of 20 statsmodels raises "Singular matrix". The only concern is "The optimizer stopped before converging…" (`modeling.py:271-291`). Separation and the column are never named.

**Recommendation.** Detect separation, name the column in a concern, suppress its Wald interval and p-value, and offer Firth-penalized logistic regression with profile-likelihood intervals.

#### MA-09 · major · Cross-validated R² is averaged fold by fold, which is strongly biased low in small studies
*A2, confirmed major.*

**Evidence.** `models/metrics.py:40` scores each fold with `r2_score` against that fold's own mean; `summarize()` averages the folds (`metrics.py:61-72`). OLS, p = 5, population R² 0.20, 5-fold, target = out-of-sample R² of a model fit on 80% of n: n = 60 per-fold mean −0.07 to −0.09 against target 0.08–0.09; n = 100 **0.05 against 0.13–0.14**; n = 300 0.16 against 0.18; agreement only by n = 1,000. Holdout R² against the holdout's own mean: 0.155–0.157 against 0.175 with the training mean (target 0.18). Hawinkel, Waegeman & Maere, *Am Stat* 2024: "The averaging R² with test MST estimator is very variable and dramatically downward biased for smaller sample sizes, and even at a sample size of 100 some of the bias persists"; "this pooling estimator should be preferred". Staerk et al. 2024: the training-mean definition "may generally be preferable based on theoretical reasons".

**Recommendation.** Report the pooled out-of-fold R² (sum of squared errors over all out-of-fold predictions against each fold's training mean; Hawinkel's estimator with its SE). Keep per-fold values for the spread. Score the holdout against the training mean. Define it in the tooltip and the methods sentence.

#### MA-10 · major · "Better than baseline" says yes on a quarter to a third of pure-noise binary datasets
*A6, confirmed major.*

**Evidence.** `models/baseline.py:42-64` calls a model "better" when the mean per-fold gain exceeds max(0.01, SD/√K), a rule its own docstring calls "lenient on purpose"; a "better" verdict raises no concern. Outcome independent of 10 predictors: "better" in 0.29–0.31 of datasets at n = 150–400 (skeptic: 0.25–0.28 across prevalence and p); a Nadeau–Bengio-corrected one-sided 5% test: 0.03–0.08. Even with independent folds the one-SE rule implies P(t₄ > 1) ≈ 0.19. (The auditor's "at least one of three families 'better' on 42% of null datasets" was not re-run.) Bengio & Grandvalet 2004: "naive estimators … grossly underestimate variance."

**Recommendation.** Use a variance-corrected comparison (Nadeau–Bengio, or repeated CV with the corrected resampled t) at one-sided 95%; otherwise say "not distinguishable from the class prior". Report the gain with an interval. **Leash: too loose.**

#### MA-11 · major · With a chronological holdout, cross-validation folds and tuning still shuffle time
*A8, confirmed major.*

**Evidence.** When the seal passes a chronological mask, `rows.py:754-755` drops stratification and `_assign_folds` shuffles (`rows.py:691-711`); every fold spans the whole period, and elastic net's inner CV is not time-aware either. Drift toys: random-fold CV R² 0.27–0.30, forward-chaining 0.23–0.24, chronological holdout −0.00 to 0.14. In the skeptic's toy the families still ranked the same, and random internal validation plus temporal external validation is the customary TRIPOD paradigm; the methods sentence's "k-fold cross-validation" is literally true. Roberts et al. 2017 (*Ecography* 40:913, abstract): "We recommend that block cross-validation be used wherever dependence structures exist in a dataset…".

**Recommendation.** When the temporal answer is yes, use blocked or forward-chaining folds by whole unit, pass the same splitter to the inner CV, and say "time-ordered folds". Decide fold stratification independently of how the holdout was drawn (A17, not re-tested: dropping it left 34% of rare-event splits with an event-free fold). **Leash: too loose.**

#### MA-12 · major · The substitution curve's "95%" band is the wrong width
*A5, confirmed major.*

**Evidence.** `modeling.py:30-31` sets BAND_ROWS = 2000 and BAND_BOOT = 50; refits use a 2,000-row subsample and the band is the 2.5th–97.5th percentile of 50 draws, recentred on the full-data curve (`methods/substitution.py:298-382`). At N = 10,000 the band is 2.0–2.5× too wide (coverage 1.000). At N = 1,500 with 50 draws, coverage is 0.89–0.91 (theory: those percentiles of 50 draws span 0.913 of the distribution); with 400–500 draws, 0.925–0.95. The width varies 11% across seeds at B = 50. The saved figure says "Shaded bands: 95% intervals from N refits per family" (`Results.tsx:198`); "errs wide" is only in a collapsible note. Carpenter & Bithell 2000: "For 90–95 per cent confidence intervals, most practitioners … suggest that B should be between 1000 and 2000."

**Recommendation.** Refit on full-size resamples, or keep the subsample as an m-out-of-n bootstrap and rescale by √(m/N). Offer B ≥ 1,000 (or label 50-refit bands "rough"). Put rows, B and the number of successful refits in the saved caption, and require a minimum share of successful refits (B16, not re-tested: a band was drawn from 2 of 50). **Leash: too loose.**

#### MA-13 · major · "Within the observed range" checks each moved column, not the diet the swap creates
*B10, confirmed major.*

**Evidence.** The support mask checks each moved column's own min–max and ≥ 0 (`substitution.py:159-167`). At k = 300 kcal, 77–84% of rows count as on-support, yet 621–1,398 of them have a fat or carbohydrate share of energy outside the observed range (fat down to 7.6–8.7% of energy). The effect sentence (`substitution.py:263-265`) overstates support. Scholbeck et al. (arXiv:2201.08837, App. A.2): "the convex hull may be comprised of many empty areas … it seems plausible to define model extrapolation differently, e.g., as predictions in areas of the feature space with a low density of training points."

**Recommendation.** Also check the shifted shares of energy against their observed range (or a joint distance in composition space) and report how many rows each check removed. Offer a fixed-population curve (B17, not re-tested: the rows each point averages change with k). **Leash: too loose.**

#### MA-14 · major · Combining rows applies one rule to every numeric column
*B12 and C7, both confirmed major.*

**Evidence.** `combine_sql` sends every numeric column through the chosen method (`working.py:830-837, 854-860`); the "varying" check covers only non-numeric columns (`working.py:905-912`). Under "change", `sex_code` becomes 0 for every person and a recorded age becomes elapsed years; the outcome can only be mean, first or last (`decisions.py:286-289`), so the result is a change-score model without baseline adjustment. Under "mean", integer category codes are averaged: smoking 1/2/3 becomes 1.67, day of week 4.33. The receipt's varying list is empty. NHANES codes categories as integers (SMQ040: 1 every day, 2 some days, 3 not at all, 7 refused, 9 don't know).

**Recommendation.** Rules per column: change for exposures, baseline for covariates, first/last/mode for codes; refuse change on columns constant within units; keep baseline values beside a change; list every numeric column that varied within units and how it was combined. (Whether ANCOVA should be the default for observational data is contested; see §7.3.) **Leash: too loose.**

#### MA-15 · major · Integer-coded categories enter models as one straight line
*B14, confirmed major.*

**Evidence.** Integer columns are typed numeric (`datastore.py:205-209`), and only non-numeric columns are treated as categorical (`models/pipeline.py:263`). RIDRETH3 (1, 2, 3, 4, 6, 7) and DMDEDUC2 (1–5) enter as single slopes. No role or decision marks a column nominal, and no repair recodes to categories.

**Recommendation.** Add a "categorical" declaration; propose it for small-cardinality integer columns and known NHANES coded variables. **Leash: too tight** (the defensible encoding is unavailable).

#### MA-16 · major · A SAS "." for missing, a decimal comma or "<LOD" turns a numeric column into text, and nothing repairs it
*C5, confirmed major.*

**Evidence.** `NULL_TOKENS` (`datastore.py:50`) has no ".". With 3% "." values, age ingests as a 61-level categorical, kcal as text with 1,212–1,484 levels (proposed "excluded" as free text), BMI as text. Decimal comma ("2000,5"), "1,234", "#DIV/0!", "<0.2" and a trailing space all ingest as text (a leading space parses; skeptic correction). The findings stage raises `numeric_as_text` and `text_missing`, but no repair family exists for them (`repairs.py` families: SAS zeros, kJ, sentinel codes, binary text, impossible values), so they read "No control for this yet." SAS documentation: "By default, SAS replaces a missing numeric value with a period…".

**Recommendation.** A row-local parse repair (trim, then parse with a shown token list or a decimal-comma reading; count and name every value that will not parse); a separate "<LOD" family with options (substitution, flag plus value, censored); "." as missing only where every other value parses. **Leash: menu too tight, guidance too loose.**

#### MA-17 · major · Excel and CSV copies of one table disagree about what is missing
*C6, confirmed major.*

**Evidence.** Excel is parsed with pandas' default missing tokens (`datastore.py:794`), while CSV uses `NULL_TOKENS`, which deliberately keeps "None" as "a real answer on a questionnaire" (`datastore.py:48-50`). A supplement column's "None" answers become blank in the xlsx copy (n_missing 2 → 5); "-nan" and "1.#IND" diverge too. Under complete cases, supplement non-users are dropped from an Excel upload only.

**Recommendation.** `keep_default_na=False, na_values=NULL_TOKENS` for Excel, and a CSV–Excel parity test over the token list.

#### MA-18 · major · One infinite value stops the analysis with an engine error
*C10, confirmed major.*

**Evidence.** DuckDB's `stddev_samp` over a column holding `inf` raises "STDDEV_SAMP is out of range" (`datastore.py:1476`), and the whole 200-column summary batch is lost. Roles depend on the profile, and the shelf and design depend on roles, so a single ratio with a zero denominator (protein per 1,000 kcal with one kcal = 0) blocks modeling with no exit. The Arrow path for wide tables silently returns None instead.

**Recommendation.** Aggregate over finite values only, count infinities, raise a finding with a set-missing repair, and add infinities to the Arrow/DuckDB parity tests. **Leash: too tight** (a refusal with no named exit).

#### MA-19 · major · Flow-diagram steps keep people whose value is unknown, under a label that says they qualified
*C11, confirmed major.*

**Evidence.** Exclusion rules keep missing values (`rows.py:410-429`: `values.isna() | inside`) while the step is labeled "`age` within `20`–`80`" (`rows.py:438-447`). With 20% of age missing, 197–202 of the rows counted at that step have no age; a "kcal within 500–5,000" step keeps 40–69 rows with no kcal. Under "impute" those rows can then carry ages outside the stated range. STROBE item 13(a): "Report the numbers of individuals at each stage of the study—e.g., numbers potentially eligible, examined for eligibility, confirmed eligible…". The repo's own test `test_a_count_is_labeled_with_the_population_it_counts.py` states the principle. D16 (not re-tested) is the same pattern for the by-sex energy screen: rows with missing or unrecognized sex are never screened, including 9,000 kcal days.

**Recommendation.** Make missing-value handling a visible option of each rule (exclude as unconfirmed, the STROBE-faithful default for eligibility; or keep and say so), and report "not recorded: n" at each step. **Leash: menu too tight, guidance too loose.** (The skeptic notes it borders on critical.)

### 2.2 Methods

#### ME-01 · critical · Missing values get one median fill whatever the purpose; inference has no sound option
*A13 (major), B4 (critical), D11 (major), E2 (critical), G3 (critical), I7 (major), and the indicator part of I6 (major); all confirmed.*

**Evidence.** The only imputer is `SimpleImputer(strategy='median')` (most frequent for categories), with the outcome deliberately excluded, running before the energy step (`models/pipeline.py:297-309`). No multiple imputation exists in `turbotab/core` (the legacy `turbotab/missingness.py` has MICE; nothing in core imports it). The missing-values question's order and coach never read purpose (`coach.py:259-290`; `ChoiceQuestions.tsx:278-321`). Reproductions through the app's own pipeline and `statsmodels_fit`:
- Confounder 35–44% missing at random given the exposure: median fill bias +38% to +75%, **95% coverage 0.00**; with indicators +0.18, coverage 0.00; complete cases unbiased, coverage 0.92–0.96; multiple imputation with the outcome (not offered) unbiased, coverage 0.92–0.95. Missingness driven by the outcome: median +60%, indicator coverage 0.24, complete cases coverage 0.80, MI 0.95.
- Intake 36–38% missing at random on age: estimate 0.45 against 0.5, coverage 0.85–0.89.
- Fat missing as a function of energy: the impute → residual path attenuates the coefficient 44–46% (also under MCAR, because the median ignores energy). Imputed rows' energy-adjusted values correlate **−1.000** with energy, against +0.78 to +0.89 for observed rows.
- Indicators are accepted under inference (an indicator coefficient is printed with an interval), although `OPENING_SEQUENCE.md:213-217` says the indicator "is blocked with both exits" under inference.
- The methods sentence says only "missing predictor values were imputed, learned from training rows only" (`voice.py:628-630`); STROBE-nut nut-13 asks to describe "any method used to handle missing values".
- The specs contradict each other. The MISSING drawer (`content.py:731-738`, SETTLED) calls median filling "indefensible in a manuscript" and says "The outcome belongs in the imputation model". ROADMAP §07 (`:441-442`) and M2_CONTRACT (`:104-105`, a Tier A test) say "never place the outcome in the imputation model, which is a blocker in any configuration". Each states one purpose's rule as universal. Sisk et al. 2023: "the outcome should be used to impute development data when using MI, yet omitted if using RI. When imputation is applied at deployment, developing a model that instead omits the outcome from imputation at development was preferred." Harrell (RMS ch. 3, citing Moons 2006, not itself read): "multiple imputation can and should use the response variable for imputing predictors."
- Groenwold et al. 2012 (*CMAJ*): in nonrandomized studies "the missing-indicator method will almost always give biased results"; "Multiple imputation provides valid estimates and standard errors in many circumstances when missing data are missing at random." The same paper notes the indicator method is valid for baseline covariates in randomized trials, which matters for feeding trials. For prediction, the current choice is defensible: Josse et al. (arXiv:1902.06931): constant imputation "is consistent when missing values are not informative. This contrasts with inferential settings".

**Recommendation.** Route by purpose. Inference: multiple imputation by chained equations including the outcome and total energy (m ≥ 20, Rubin's rules), ranked first; complete cases with its assumption stated; single median fill and indicators blocked and recorded for the inference table. Prediction: keep in-fold, outcome-free imputation and indicators. Impute nutrients conditional on energy in either case. Name the method in the methods sentence. Rewrite ROADMAP §07, the M2 Tier A test and the drawer as one purpose-conditional rule. **Leash: too loose for inference; about right for prediction.**

#### ME-02 · critical · "No energy adjustment" fits the standard model and labels it absolute intake
*B2, D1, G1, all confirmed critical; also reported by the E3 skeptic.*

**Evidence.** Energy is a predictor role (`pipeline.py:43`), the energy step returns nothing for "none" (`steps.py:128`), and the validator returns early (`decisions.py:841-842`), so under "none" the energy column stays in the model. The model matrices for "none" and "standard" are identical: protein 0.0299 against a truly unadjusted 0.0561 (D skeptic); fat 0.0212 against 0.0726 (B skeptic); protein 0.5007 against 0.6351 (G); identical CV R². Every label says otherwise: the estimand "Not energy-adjusted: a nutrient coefficient describes absolute intake" with specification Y ~ N + C (`methods/energy.py:75-84`); the methods sentence "No energy adjustment was applied: … enter the models as absolute intakes" (`voice.py:698-700`); the option "Absolute intake" (`content.py:817`); the coach and the preview caption (`coach.py:307-309`, `models/previews.py:186-187`). The app's own DISPUTED advice for BMI and adiposity outcomes, "present adjusted and unadjusted models", would therefore report the standard model twice, once labeled unadjusted.

**Recommendation.** Under "none", remove the energy-role column from the predictors so the model is Y ~ N + C. Offer "keep energy as a covariate" under its true name (standard). Add a test that the two matrices differ whenever an energy-role column exists, and build the estimand sentence from the fitted matrix, not from the method's name. **Leash: too loose.**

#### ME-03 · critical · The "residual method" drops total energy but claims the standard model's estimand
*B7 (raised from major to critical), D2 (critical), G2 (critical), B6 (major); all confirmed.*

**Evidence.** The residual step drops energy (`energy.py:527`; specification "Y ~ N_adj + C") and is labeled "The same substitution as the standard model" (`energy.py:101-104`; also `content.py:822-823`, `finding_words.py:518-521`). The methods sentence never says energy left the model (`voice.py:704-709`). Only the layer-3 drawer (`content.py:853-860`) states the condition.
- Sex drives both energy and protein share: standard **+0.0199**, app residual **−0.0067**, residual + energy +0.0199 (logistic: +0.0200 / +0.0006 / +0.0200). Skeptic: +0.0196 against −0.0059.
- With a sex covariate: −32% against standard. With truth 0.30: app residual 0.2749 (0.2638–0.2860), an interval that excludes the truth; standard 0.2934 covers it. Realistic sex, age and activity correlations over 200 replicates: −13%.
- No covariates: identical coefficient, but the interval widens (by 9% to 2.8× across four fixtures, depending on how strongly energy predicts the outcome), so the pack's "identical coefficient and p-value" (NUTRITION_PACK line 404) is false for the model the app fits.
- The log variant (B6), back-transformed with energy dropped, is N·(E/GM)^−b, which correlates 0.9999 with the density N/E, under the same residual label. The UI hard-codes `log_transform: false` (`ChoiceQuestions.tsx:415, 423`), but the API, decision replay and methods sentence accept true.
- McCullough & Byrd (*AJE* 2023;192:1801): "A variation on the simple nutrient residual model proposed by Willett and Stampfer includes the nutrient residual plus a term for total energy intake." Tomova et al. 2022 enter the residual "into a second unadjusted model", with no other covariates, which is where their identity holds.

**Recommendation.** Make the residual method keep energy in the outcome model (the Willett–Stampfer variant): it reproduces the standard coefficient exactly while keeping the nutrient in its own units. If the energy-dropped form stays, label it "energy is not in the outcome model; equals the standard model only when no covariate correlates with energy", show the gap on the training rows, and say so in the methods sentence. Give the log variant its own estimand. Correct the pack. **Leash: too loose** (the "usual" option carries an estimand its own drawer contradicts).

#### ME-04 · critical · With several energy-bearing nutrients, each coefficient is a swap for the omitted source, labeled a swap for the average of all others
*B3, confirmed critical.*

**Evidence.** By default every energy-bearing exposure is adjusted (`proposals.py:163-173`; on `dietary_recalls.csv`: protein, fat, carbohydrate, fiber), and each method carries one fixed estimand (`energy.py:90-104`). Per-kcal truths: fat 0.004, carbohydrate 0.001, protein 0.006, alcohol 0 (n = 20,000). With fat, carbohydrate and protein all in the model plus energy, standard and residual give fat **0.0039–0.0042**, which is fat versus the omitted alcohol (truth 0.0040). The label, "in place of the average of all other energy sources", names a quantity whose truth is **0.0018–0.0019**. Fat alone plus energy matches the label. NUTRITION_PACK §05(a): "Each coefficient is the effect of substituting that component for the omitted one. Name the omitted component in your results."

**Recommendation.** Make the estimand depend on the set of energy-bearing terms: with two or more, say "substitution for the energy sources not in the model" and list them; warn when fiber and carbohydrate-by-difference are both present (B19, not re-tested: fiber energy is double-counted). **Leash: too loose** (a mislabel on the default path).

#### ME-05 · critical · Substitution curves never check that every energy source is in the model
*D5, confirmed critical.*

**Evidence.** Any two energy-bearing predictors can be paired (`modeling.py:155-168`); the design warnings check nesting, closure and overlap but not completeness (`pipeline.py:423-485`); the module says it does not "remedy composite variable bias" (`substitution.py:32-34`). Truth for 100 kcal carbohydrate → protein: +0.600. With protein and carbohydrate only, the curve is **+0.940 under every energy method** (+57%; +0.918 under density + energy). With all four sources: +0.578. On the NHANES export, which has no alcohol column, the implicit "other" energy exceeds 10% of energy in 9.1% of rows. Tomova, Gilthorpe & Tennant 2022 (PMC9630885): "Wherever ≥2 components are involved in the substitution, there is scope for composite variable bias unless the individual effects are estimated and combined using an all-components approach." (The skeptic notes the mechanism here is confounding through the heterogeneous "other" composite; the result reproduces either way.)

**Recommendation.** Under inference, list the energy sources left in the implicit "other" and state that the curve carries their confounding; offer "add the remaining energy sources" in one click; block and record when the omitted share of energy exceeds a stated threshold. Offer the field's "% of energy from X replaced by Y" leave-one-out model and, later, compositional (ilr) models. **Leash: too loose for inference; right for prediction**, where the curve is a model contrast.

#### ME-06 · critical · Under inference, survey weights, strata and PSUs are recorded and then ignored, with no concern on the result
*D6 (critical), E10 (major), G13 (raised from major to critical), I4 (critical); all confirmed.*

**Evidence.** The model fit has no weights, strata or PSU, only optional clustering by participant (`linear.py:75-90`). The "design" role only removes columns from the predictors. `design_findings` returns nothing once design columns are present (`finding_words.py:658-662`); the only disclosure is the layer-3 roles drawer ("records design columns but does not yet weight its estimates", `content.py:609-615`); no methods sentence mentions it. On an informative-weight fixture, purpose = inference: app DR1TFIBE **−0.037 to −0.039** (p ≈ 10⁻³⁵ to 10⁻⁴²), concerns `[]`; weighted with PSU-within-stratum clustering **+0.0065 to +0.0066** (intervals 0.0000–0.0132 and 0.0006–0.0125), a sign flip. The holdout and the exclusions delete rows and PSUs rather than using domain flags. One nuance: the `survey_weights` warning does fire and routes to roles, and marking WTDRD1 as "design" counts as answering it, though nothing is weighted. NHANES Analytic Guidelines 2011–2016: "The complex survey design … must be considered when analyzing the data for appropriate variance estimation and to calculate statistics representative of the U.S. civilian non-institutionalized population"; "the entire set of data containing the appropriate weights … must be used to obtain the correct variance estimates." CDC variance tutorial (read by areas D and G): SRS variance estimates "are generally too low (i.e., significance levels are overstated)". Cameron & Miller 2015: "At the minimum one should cluster at the level of the primary sampling unit" (they also note unweighted estimation can be defensible when stratification is only on exogenous regressors and the model is correctly specified).

**Recommendation.** When design columns are present, ask "an estimate for the surveyed population (use the design), or for these participants (unweighted, stated)?". Under inference, until design-based estimation exists, block and record the attestation "unweighted, sample-only estimand; standard errors ignore strata and PSUs" and carry it into the methods sentence. Then build Taylor-linearized design-based estimation with the least-common-denominator weight rule (IN-19), lonely-PSU handling, the 1999–2002 four-year-weight exception, and exclusions as domains. Under prediction, state that metrics are unweighted. **Leash: too loose.**

#### ME-07 · critical · Logistic coefficients are log-odds captioned "change in predicted outcome"; no odds ratios
*E4, confirmed critical.*

**Evidence.** For binary outcomes the table holds statsmodels Logit parameters (log-odds), but the caption is task-blind: "Change in predicted `${target}` per unit of each input, holding the others; 95% confidence intervals" (`Coefficients.tsx:104-107`). The saved figure has no scale label (`resultsJournal.ts:56-90`; `Results.tsx:166-169`), and no `exp(` or "odds" exists in the results code. Fiber 0.0674 log-odds is an odds ratio of 1.070 per gram. Multinomial rows are against the first sorted class, which is never named (`linear.py:184-196`). The step text says "log-odds" (`linear.py:123`), but not on the panel or the figure. The EVENT teaching says the event "fixes the meaning of every … odds ratio", yet none is shown.

**Recommendation.** Task-aware captions and axes: linear, "difference in mean <y> per unit"; binary, "odds ratio for <event> vs <reference> per unit", plotting exp(β) on a log axis; multinomial, "relative-risk ratio vs <reference class>". Name the event and reference levels in the figure's provenance. **Leash: too loose.**

#### ME-08 · critical · Values below the detection limit have no detection-aware option, and measured zeros are taken at face value
*F1 (critical) and F11 (major), both confirmed.*

**Evidence.** The left-censoring finding routes to the missing-values question (`finding_words.py:313`), whose options are complete cases or median fill. On `metabolomics_untargeted.csv`: complete cases keep **0 of 72** rows; median fill puts `mz_0121`'s non-detects (56% of the column) at 81.3, 1.47× the smallest detected value. The app's own finding says median filling "would place non-detections in the middle of the distribution" (`packs.py:661`) and its drawer calls it indefensible. MZmine-style zeros raise a warning with no lever and enter the model as measured zeros, the pack's own anti-pattern. The methods sentence says only "imputed". Log-scale simulation (truth 0.5): median fill gives 0.53/0.56–0.59/0.66–0.67 at 20/40/60% censoring. (The raw-scale direction depends on how the outcome relates to the metabolite and is not general; skeptic correction.) F11: the pack's planned M3 default, QRILC as the "best-supported default" (`METABOLOMICS_PACK.md:243`), rests on Wei et al. 2018's benchmarks, which score value recovery (NRMSE, Procrustes; one downstream t-test p-value correlation), not bias in a coefficient. An outcome-blind truncated-normal draw attenuates the slope to 0.38 of 0.5 at 60% censoring (analytically about 0.75β). The code itself already prefers half-minimum (`packs.py:631-634`). Wei et al. 2018: values below the limit "are usually replaced with a determined small value or zero, which may lead to certain biases"; Eekhout et al. 2014: "We advise not to use any form of mean imputation."

**Recommendation.** Before M3: when `left_censored` or `zeros_or_missing` fires, block and record median fill for those columns; ask before recoding zeros as non-detects; name the method in every methods sentence. In M3: half-minimum (customary) plus a censoring-aware option (detected-only for a single censored covariate, Tobit-type models, or multiple imputation that conditions on the outcome), and the pack's two-scheme sensitivity analysis, ordered by purpose. Badge QRILC as best for value recovery, not for association estimates. **Leash: too loose and too tight at once.**

#### ME-09 · critical · Raw counts and raw intensities go straight into the models
*F2, confirmed critical.*

**Evidence.** Every family runs impute → one-hot → standardize → model (`pipeline.py:324-339`); there is no CPM, TMM, VST, log or PQN anywhere in core, and no decision can transform values. Sixty samples × 300 negative-binomial genes, **no gene differs**, cases sequenced 1.44–1.46× deeper: elastic net CV AUC **0.83** (auditor) and **1.0 on every fold** (skeptic), with no concern; library size alone has AUC 0.987; an equal-depth control scores 0.50 with the "no better than the class prior" concern. Log-CPM: 0.49–0.62. Urine 25% more dilute in cases, no metabolite differs: raw AUC 0.59–0.71, PQN + log2 0.50. Outcome linear in log concentration: CV R² 0.32 raw against 0.73 on log2. The genomics data-type finding tells users "Do not pre-normalize these" (`packs.py:3803`), DESeq2's advice for its own input, given to a pipeline with no library-size correction. Hornung et al. 2015: "Performing normalization on the entire dataset before CV did not result in a noteworthy optimistic bias in any of the investigated cases", so normalizing at all matters far more than doing it inside folds.

**Recommendation.** Under the omics lenses, refuse a linear-family fit on raw values until the user picks a transformation or records that the values are already normalized. Add log-CPM/TMM and VST (counts) and PQN or median-fold plus log (intensities). Add a label-blind check of library size or total signal against the outcome. Replace "Do not pre-normalize" with purpose-specific text. **Leash: too loose.**

#### ME-10 · major · No calibration, no interval on any performance number, and "best model" means highest AUC
*A10, E9, G11 and the interval part of E8; all confirmed major.*

**Evidence.** Binary metrics are AUC, Brier and log loss with AUC primary; multiclass primary is macro-F1, which is not a proper scoring rule (`metrics.py:16-21`). "calibrat" appears in core and the frontend only in teaching text and mocks. CV results are mean ± SD over folds and the holdout is a point (the seal's "AUC known to about ±" is a planning width, not a result). The banner picks a "best" family by cross-validated AUC (`banner/derive.ts:334-347, 369-370`, which does name the metric). The teaching promises what is not delivered: "Report the C-statistic with its interval, the calibration intercept and slope, the calibration curve and the Brier score" (`content.py:1027-1031`); "Rank models on calibration as well as discrimination" (`content.py:938-941`). (Brier and log loss do partly reflect calibration; skeptic nuance.) TRIPOD+AI 2024, item 23a: "Report model performance estimates with confidence intervals"; item 12e: measures "eg, discrimination, calibration, clinical utility". Van Calster et al. 2019: "estimated risks can be unreliable even when the algorithms have good discrimination."

**Recommendation.** Calibration-in-the-large, calibration slope and a smoothed calibration curve, out of fold and on the holdout; a DeLong or bootstrap interval for the holdout AUC; the pooled CV estimate with its SE (MA-09); log loss as the multiclass primary; label the banner "highest AUC" or rank on a proper score. **Leash: too loose.**

#### ME-11 · major · Internal validation stops at one k-fold run or a random split
*A11, E8, G10 and I14, all confirmed major.*

**Evidence.** The split offers a holdout of 0 or 10–40% and 2–10 folds, once (`decisions.py:151-162`); there is no repeated CV and no bootstrap optimism correction anywhere in core. Above the floor, "Hold out 20%" leads (`seal.py:590-593`); "Cross-validation only" is on the menu, ranked last. The teaching says "Bootstrap optimism correction or repeated cross-validation is preferred" (`content.py:789-797`). A SETTLED claim with no source says a 5-fold estimate cannot resolve 0.05 AUC only "below about 50 rows" (`content.py:794-798`): the SD of a single 5-fold AUC is 0.11–0.125 at n = 50 and 0.05–0.08 at n = 100, and the SD of a paired difference between two models is 0.035 at n = 100 and 0.014 at n = 200, so a 0.05 difference stays within noise to about n = 150–200, 2–4× the stated threshold (the auditor's "order of magnitude" was overstated). Collins et al., *BMJ* 2024: "Randomly splitting a dataset is often erroneously perceived as a methodological strength—it is not." Steyerberg 2018: "random data splitting should be abolished for validation of prediction models". PROBAST item 4.8 rates internal validation by "a single random split-sample" as no/probably no. BLUEPRINT north star 5 names this exact tension.

**Recommendation.** Offer repeated k-fold (for example 10 × 5) and Harrell's bootstrap optimism correction (whole pipeline refit per resample), ranked first for prediction below a stated n; keep the holdout with a one-line custom-versus-sound note (a lockbox against analyst overfitting; costs precision at this n). Replace the threshold with the spread computed on the user's data. **Leash: menu too tight, order too loose.**

#### ME-12 · major · Under inference the split still holds out 20%, and the coefficients use only the training rows
*A12, E6 and the holdout part of I6, all confirmed major.*

**Evidence.** The seal plan takes no purpose (`seal.py:559-596`; `stages/seal.py:23`). The coefficient table is fit on the training rows (`modeling.py:395, 446`), disclosed only in a step detail (`linear.py:124-125`). On the NHANES export under inference (2,996 complete cases): all rows give sugar −0.0576 (−0.1096, −0.0057), p = 0.030; twenty random 80% seals give −0.097 to −0.021, intervals 1.12× wider, and p < 0.05 in **9 of 20**. The spec is the origin: BLUEPRINT's "anything fit on data is fit on training rows only" is a prediction rule. Shmueli 2010: "In explanatory modeling, data partitioning is less common because of the reduction in statistical power."

**Recommendation.** Under inference, fit the coefficient table on every analyzed row and state n; lead the split with "no holdout"; ask to open the seal only when a holdout was chosen deliberately; name the tension. **Leash: too loose** (the order ignores the declared purpose).

#### ME-13 · major · Three families are compared and opened together, with no declared final model
*A7, confirmed major.*

**Evidence.** Opening the seal records no chosen model (`decisions.py:353-356`) and reveals every family's holdout at once (`seal.py:663-689`). On null binary data the CV-best family's CV AUC averages 0.536–0.537 against a true 0.50 (each family's own CV is unbiased); the best holdout picked after opening averages 0.523. With weak signal the optimism is +0.014. Varma & Simon 2006: "The CV error estimate for the classifier with the optimal parameters was found to be a substantially biased estimate of the true error". With three families the optimism is modest, and all scores are shown together; the banner's "best for" label (ME-10) is what turns it into a selected result.

**Recommendation.** Before opening, ask which family is the final model (chosen on CV) and record it; show that family's holdout as the result and the others as secondary; state or estimate the selection optimism (nested CV over the family choice). **Leash: too loose.**

#### ME-14 · major · The all-components model is not offered by name, and its relative effect is never computed
*D3 (major), the all-components parts of B8 and G6; all confirmed.*

**Evidence.** The methods are none, standard, residual, density with energy, density and partition (`energy.py:66`). A partition over every energy source is the all-components model, but it is unnamed, labeled "adding calories, not substituting" (correct for its own coefficients) and ranked after residual. No code computes the average relative causal effect. Standard and residual give 0.0237 per g protein against a true average relative effect of 0.0305; the all-components model with the energy-share-weighted difference recovers 0.0307. Tomova et al. 2022: "Accurate estimates of both the total and average relative causal effects may instead be derived by simultaneously adjusting for all dietary components … the 'all-components model'." Chiu & Wen 2026 (*AJE*): "we recommend adopting a flexible all-components model". The Willett/Stampfer/Tobias 2022 reply disputes this; nobody could read it (§7.4).

**Recommendation.** Add "all components" with two estimands (the total causal effect per component, and the average relative causal effect via the weighted difference with a delta-method or bootstrap interval). Under inference rank it first and state the dispute and Tomova's precision cost in one line.

#### ME-15 · major · A total beside its own parts silently changes what the total's coefficient means
*D7, confirmed major (the skeptic sees a case for critical under inference).*

**Evidence.** On the NHANES export nesting is detected (sugar in carbohydrate; SFA, MUFA, PUFA in total fat) and the proposals make all seven nutrients exposures. OLS glucose on age, kcal and nutrients: fat_total **0.3127** (0.1124–0.5130) with its parts in the model against **0.0734** (0.0444–0.1023) without; carbohydrate +0.0231 against −0.0023. With the parts fixed, the total estimates the unclassified remainder (a median 8.9% of total fat). The design warning speaks only of substitution (`pipeline.py:461-467`).

**Recommendation.** When a total and its parts are both predictors, relabel the total's row ("fat_total holding SFA, MUFA and PUFA fixed = remaining fat") in the table and the methods; offer totals only, parts only (plus a computed remainder), or both with the relabel.

#### ME-16 · major · No misreporting screen beyond fixed kcal cut-offs, and no with/without-exclusion sensitivity analysis
*D9, confirmed major.*

**Evidence.** The exclusion options are none, three fixed kcal screens and custom (`content.py:640-651`); Goldberg is "not offered in this version" (`content.py:659-661`). The drawer recommends presenting "the misreporter-excluded analysis as a prespecified sensitivity analysis" (`content.py:680-684`), but no mechanism fits both. On the NHANES export the screens move n by 501 to 2,094 rows. Banna et al. 2017: "Regardless of which method is used, for the time being, analyses in the total sample without exclusion of participants should also be conducted and reported." Yamamoto et al. 2023: Goldberg cut-offs reduced bias "in 14 of 24 nutrition-outcome pairs; bias was not reduced for the remaining 10".

**Recommendation.** Add Goldberg/Black with a named BMR equation (Schofield including the ≥ 60 bands, Henry, Mifflin–St Jeor), d = recall days and a stated PAL; add a "primary plus sensitivity" answer that fits both row sets and shows the estimate under each rule. Under prediction, say that excluded reporters remain in the deployment population. **Leash: menu too tight.**

#### ME-17 · major · Exposure–response is straight-line only
*D13, confirmed major.*

**Evidence.** No spline, quantile or categorization step exists in core; the linear family "Fits the straight-line effect of every column" (`linear.py:118-126`). NUTRITION_PACK §07G calls restricted cubic splines "now near-default; quintiles remain expected alongside", and BLUEPRINT north star 5 names the tension.

**Recommendation.** An exposure-form decision: linear; restricted cubic spline (3–5 knots at the pack's percentiles, a test for nonlinearity, the exposure distribution shown); quintiles with a trend test on quintile medians. Under inference, rank the spline first and tag quintiles "customary". **Leash: too tight.**

#### ME-18 · major · Omics inference has no sound family, and the shelf hides it
*F6, confirmed major.*

**Evidence.** Elastic net returns "good" before its inference caveat when p ≥ n (`models/elastic_net.py:81-85`): at n = 60, p = 497 under inference it is "good" with no word about intervals; at n = 600, p = 50 it is "fair" with "Penalized coefficients are shrunk and carry no confidence intervals." No PLS-DA, VIP, FDR, Benjamini–Hochberg or limma-style family exists in core or the frontend. METABOLOMICS_PACK §08 marks per-feature testing with multiple-testing correction SETTLED: "its absence is a fatal flaw in review". BLUEPRINT north star 5 names "VIP > 1 versus penalized or permutation-validated models".

**Recommendation.** Under inference with p ≥ n, say that no shelf family gives valid inference. Add feature-wise regression with covariates and Benjamini–Hochberg FDR. List PLS-DA with "customary in metabolomics; sound only with nested CV and permutation testing; VIP is a ranking, not a test", or explain its absence in one line. **Leash: too loose, menu too short.**

#### ME-19 · major · No ordinal outcome model, and no scale scoring, reverse-coding or reliability
*F13, confirmed major (also the ordinal part of I10).*

**Evidence.** Tasks are regression, binary and multiclass only (`decisions.py:49`). A Likert item as outcome is read as multiclass. No reverse-coding, scoring, alpha or omega exists in core. The app is honest ("This version fits no ordinal model, so say which approximation you chose"), and BLUEPRINT north star 1 lists ordinal outcomes.

**Recommendation.** A cumulative-link (proportional-odds) family; in M3, scale scoring with the codebook's reverse keys (asked, never inferred), the instrument's own missing-item rule with item-level multiple imputation as the sound alternative (Eekhout 2014), and omega with an interval. **Leash: guidance right, menu too short.**

### 2.3 Intelligence

#### IN-01 · critical · The nutrient recognizer reads substrings, so body fat, CRP, fibrinogen and fatty fish become energy-bearing nutrients
*B5 and H3, both confirmed critical.*

**Evidence.** Roles are matched on substrings after "_" becomes a space (`prot`, `carb|cho\b`, `\bfat|lipid|tfat`, `alco|etoh`, `fib`; `nutrition.py:84-90`, `methods/energy.py:165-228`). Probed names: `fatty_fish_g` → fat at 9 kcal/g, marked declared; `fat_mass_kg`, `body_fat_pct`, `fat_free_mass`, `liver_fat`, `visceral_fat_area`, `fatigue_score`, `lipid_lowering_meds` → fat; `c_reactive_protein`, `total_protein`, `serum_protein_g_dl`, `prothrombin_time` → protein; `fibrinogen`, `fib4_score` → fiber; `bicarbonate`, `carbon_monoxide`, `carbamazepine`, `carboxyhemoglobin`, `carbonated_drinks_g` → carbohydrate; `alcohol_use_disorder` → alcohol. On a diet-plus-body-composition table, `fat_mass_kg`, `c_reactive_protein`, `total_protein` and `fibrinogen` are proposed "exposure (high): A nutrient that carries energy", enter the nutrient candidates, and the usual (residual) method regresses CRP on kcal. Partition accepts `fatty_fish_g` at 9 kcal/g (fatty fish is about 2) with no unit note (`modeling.py:622-625`). The substitution menu lists body-composition columns as donors (the `set_substitution` validator does refuse non-exposures, a partial mitigation). NUTRITION_PACK §01: "match on three signals jointly, never names alone".

**Recommendation.** Match whole tokens and require name, unit suffix and plausible intake magnitude jointly; deny biomarker and body-composition stems (mass, serum, plasma, reactive, fibrin, carbon, bicarb, pct, score); give food-group grams no kcal/g unless the user declares one; require a nutrient to correlate positively with energy before it becomes a default adjustment target; show excluded columns with their reason. **Leash: too loose** (high-confidence false roles flow into defaults with methods consequences).

#### IN-02 · critical · The "design" role swallows trial arms, conditions, batch and site, and drops them from the model with a false reason
*H4 (critical), F5 (major), and the role part of I5 (major); all confirmed.*

**Evidence.** `rows._acquisition_columns` (`stages/rows.py:297-308`) returns every column `packs.design_columns` claims, including the study columns (`packs.py:1860-1871`: group, class, condition, treatment, arm, phenotype, fasting, medication, site…). `propose_roles` (`rows.py:267-268`) assigns them "design" with "An acquisition column (batch, plate or run order), not biology", and the role's option text is "Survey design … kept out of the predictors" (`content.py:591`). On an RCT table, `treatment`, `arm`, `diet_group`, `condition`, `phenotype`, `fasting` and `plate_reader_od` (an ELISA readout) become "design" under every lens and with none. Batch and run order become "design" although the pack's own default is to model the batch (`packs.py:1051`; METABOLOMICS_PACK:442): batch confounded with BMI, metabolite effect 0: slope 0.40–0.52 with batch excluded against −0.01 to 0.01 with batch as a covariate. A multisite cohort's `site` becomes "design": sodium 0.76–0.90 (p ≈ 10⁻³⁴ to 10⁻⁵¹) against −0.04 to −0.08 with site fixed effects. Nygaard et al. 2016: correct-then-test with ComBat gave 2,011 "differentially expressed" genes where blocking for batch in limma gave 11.

**Recommendation.** Give the acquisition role only to `DESIGN_COLUMNS` (run order, batch, plate, well, injection, polarity), matched as whole names. Add a separate "acquisition/batch" role whose default depends on purpose (covariate under inference, excluded under prediction, with the reason stated). Map treatment, arm and group to exposure candidates (and target candidates under omics), subject to identifier, timepoint to time. Add an RCT must-keep fixture. **Leash: too loose.**

#### IN-03 · critical · The survey code detector calls the top answers of 6- and 7-point scales "almost certainly" missing codes
*F3 and H1, both confirmed critical.*

**Evidence.** `_LIKERT_SETS` (`packs.py:3058`) holds {1–5}, {1–4}, {1–7}, {0–3}, {0–4}, tried in that order. `_breaks_the_run` (`packs.py:3101`) tests against the matched declared scale, not the observed contiguous run that the spec requires (CLINICAL_SURVEY_PACK:676-681, "Flag values that break the observed contiguous run") and that the code's own comment at `packs.py:3161` describes. A balanced 6-point block reads as 1–5 with 6 flagged: a critical finding ("almost certainly 'don't know' / 'refused' / 'not applicable' sentinel codes"), a repair that blanks 274–804 answers (16–17% of all answers), and the methods sentence "Missing-answer codes in 12 items were recoded as missing". A skewed 7-point block flags 6 and 7; a 0–5 block flags 0; a 9-point hedonic block reads as 1–7 with 8 and 9 flagged. A 1–5 block with a real 9 is still caught correctly.

**Recommendation.** Take the response support from the contiguous union across the block; flag a value only if it is separated from that run by a gap and is a known sentinel or at least two steps beyond. Add {1–6}, {0–5}, {0–6}, {1–9}, {1–10}, {0–10} and prefer the smallest scale that contains every value. Drop "almost certainly" and lower the severity when the candidate is adjacent to the run. Word the repair's sentence "recoded as missing on the user's instruction". **Leash: too loose in the dangerous direction.**

#### IN-04 · critical · The structural code detector flags real clinical extremes as missing codes
*H2, confirmed critical.*

**Evidence.** `ml/import_doctor.py:590-593` keeps a candidate code if fewer than max(3, 1% of n) real values lie on *either* side. On clean simulated columns (200 replicates per generator) it flags DBP in 2–9%, heart rate in 2.5–11.5%, glucose in 2–15.5%, prior admissions in 2–13.5%, age in 1–10% of columns. Details read "Found 99 (2x) — far outside the rest of the column (36 to 105)" and flag glucose 66 with hundreds of real values above it. In Next each carries a critical card, a set-missing repair, and the sentence "`99` in `dbp` is a code for a missing answer". The values flagged are the clinically extreme ones; recoding them removes the sickest patients.

**Recommendation.** Require a candidate to lie beyond the observed range on its side and beyond a gap of several typical spacings; for continuous columns, require a conventional code with an implausible frequency spike; keep it a warning unless the name or codebook corroborates; add clinical-tail must-not-fire fixtures. **Leash: too loose.**

#### IN-05 · critical · The outcome's unit is guessed and written into estimand sentences, with no way to correct it
*C4, confirmed critical.*

**Evidence.** `core/units.py:31-46` matches name patterns and then picks the unit whose typical value is nearest; its claim that the readings "sit far apart" fails for weight (2.2×) and height (2.54×). Mislabels: dietary choline and dietary cholesterol (mg/day) → "mg/dL"; glucosinolate intake → "mmol/L"; gestational age (weeks) → "years"; birth weight (g) → "lb"; urine creatinine (mg/dL) → "µmol/L"; `recreational_activity_min` → "µmol/L" ("creat"); telomere length (bp) → "mmHg"; `sleep_hr` → "beats/min"; toddler height (cm) → "in"; bariatric weight (kg) → "lb". The unit flows into the Record ("chosen as the outcome, in mg/dL", `voice.py:306-310`) and the substitution estimand ("predicted X (in mg/dL)", `modeling.py:557-559, 593, 642`). No decision sets or corrects it. CLINICAL_SURVEY_PACK (`:42-45`): "Please confirm units per analyte against the source data dictionary — TurboTab will not guess"; "Detect, propose, require explicit confirmation."

**Recommendation.** State a unit only when it is read from an explicit suffix; otherwise propose it and require a recorded decision (`set_outcome_unit`) before any sentence carries it. Match whole tokens, exclude intake contexts (dietary_, intake, per day) from blood-analyte patterns, and drop the magnitude rule where the conversion factor is under 3×. **Leash: too loose.**

#### IN-06 · major · Unit identifiers are recognized three different ways; common IDs are missed and measurements are offered as IDs
*H5 and I13, both confirmed major.*

**Evidence.** `rows._id_like` (`rows.py:130-138`), `finding_words.is_identifier_name` (`:527-545`) and `test_lockbox._id_kind` (`:181-243`) disagree on 11 of 46 real-world names (MRN, Participant, subjid, USUBJID, ResponseId, record, Subject, pid, person_number, Study Number, case_no); none recognizes `eid` (UK Biobank), `patid` (CPRD), `ptid`, `HHID` or `IDNO`. `eid`, `MRN`, `patid` and `Participant` are proposed "exposure (low)" while the findings stage says Participant names each row. `site_id`, `batch_id` and `household_id` are proposed "identifier (high)": with "one row each" the seal is then abandoned on a clean study, and with "repeated" it groups by site (5 units) and falls back to a row split. Grain suggestions offer measurements: `length_of_stay_days`, `age`, `sodium_mmol_l` on `clinical_risk.csv`; choosing `length_of_stay_days` produces a clean "grouped" seal over 19 "units" while the column stays an exposure.

**Recommendation.** One recognizer everywhere (the lockbox's subject/cluster/record taxonomy with the union of names plus eid, patid, ptid, hhid, idno, usubjid); propose cluster-kind columns as "cluster", not "identifier"; let the seal group only by subject-kind identifiers; never suggest float or measurement columns as unit IDs, and mark the basis exploratory if the user names one. **Leash: too loose.**

#### IN-07 · major · Energy in kilojoules is read as kilocalories, and the implausible-intake count ignores the app's own unit finding
*D10 and H9, both confirmed major.*

**Evidence.** The energy unit falls back to kcal unless a suffix or the Atwater reconstruction (which needs protein, carbohydrate and fat) says kJ (`proposals.py:297-311`). An "energy" column in kJ beside only sodium: the screens would remove 764–989 of 800–1,000 rows, and the coach says "`764` rows above `5,000` kcal: likely over-reporting." The `implausible_intake` finding and its evidence view compare the raw column with 500–5,000 regardless (`packs.py:2897-2925`; `core/evidence.py:249-262`): on `nhanes_kilojoules.csv` "118 records … above 5000" sits beside the Atwater finding "The energy column is in kilojoules"; on a kJ "energy" column, 298–299 of 300 rows. Findings read the table before repairs, so the count stays wrong after conversion. The pack's own magnitude prior (NUTRITION_PACK:41-42, "energy 1,600-2,600 kcal (7,000-11,000 → kJ)") is never used. The 500–5,000 band is the loosest in circulation; Rhee et al. 2015 give the Nurses' Health Study band as 500–3,500.

**Recommendation.** Apply the magnitude prior when the Atwater check cannot run; refuse a screen that would remove more than half the rows, with a units exit; make the finding read the unit verdict and convert; show counts under each convention, as the proposals already do. **Leash: too loose.**

#### IN-08 · major · Standard nutrient names are missed: fat subtypes, day-2 energy, older NHANES cycles, INFOODS tags
*B13 and H10, both confirmed major.*

**Evidence.** `sfa_g`, `mufa_g`, `pufa_g`, `DR1TSFAT`, `DR1TMFAT`, `DR1TPFAT`, `DR1TSUGR` get no Atwater factor, and the refusal says the column "carries no energy: no Atwater factor is known for it", which is false; `nesting.py:31-41` does know SFA, MUFA and PUFA, so the vocabularies have diverged. The pack's flagship substitution (SFA → MUFA/PUFA) cannot run on NHANES names. Energy found only by exact alias (`packs._reference_column`): `DR2TKCAL`, `DRXTKCAL`, `total_energy`, `kcal_day`, `TotalKcal`, `ENERC_KCAL` produce no dietary hint and no energy-adjustment or implausible-intake finding, although the role recognizer finds them. A 45-nutrient DR1T* table: 24 nutrients proposed "covariate (low)". INFOODS `PROCNT`, `CHOAVL`, `ALC` are unrecognized, so the Atwater check cannot run.

**Recommendation.** One energy and nutrient recognizer everywhere, shared with nesting, extended with the NHANES codebook (DR1T, DR2T, DRXT plus the suffix) and INFOODS tagnames. **Leash: too tight** (defensible choices hidden behind a false reason).

#### IN-09 · major · Clinical plausibility bands are unsourced "demo defaults", adult-only, and applied to dietary energy
*H6 and H7, both confirmed major.*

**Evidence.** `ml/physiology_reference.py:44-45` reads `'nhanes_reference_demo_v3_aliases'`, "NHANES (reference population, demo defaults)", while the docstring claims bands "derived from NHANES-like population distributions". On the real NHANES export the share outside the bundled "p01/p99" is 9.1% (triglycerides), 6.3% (kcal), 3.6–4.1% (DBP), 3.4% (glucose), where a true p01/p99 gives 2%. Total energy is in the physiology table (floor 100, ceiling 30,000, band 800–4,500), so the NHANES export gets "physiologically impossible in a living outpatient, and almost certainly entry errors … would remove the sickest patients" (SETTLED, with a set-missing repair) for nine recalls of 0–93 kcal, and "1,369 abnormal but real and must be kept", while the dietary card offers to exclude 501 rows outside 500–5,000. Pediatric weights: 279–298 children called "abnormal but real", or the column set aside as wrong units and a real 0.2 kg error missed. HbA1c in IFCC mmol/mol is called impossible. CLINICAL_SURVEY_PACK §A1.2 marks bounds as CONVENTION and says "Pediatric and growth data: never apply adult bounds".

**Recommendation.** Remove total energy from the physiology bands and route very low recalls to the dietary path; replace the bands with sourced ones (computed NHANES percentiles with cycle and version, or published EHR limits) labeled CONVENTION; gate on age (CDC modified z-scores for children) and pregnancy; say "outside the central 98% of the reference sample", not "abnormal"; treat HbA1c IFCC as a unit hypothesis. **Leash: too loose.**

#### IN-10 · major · Name tokens send rate exposures to "time" and birth weight to "sampling weight"
*H16, confirmed major.*

**Evidence.** `rows.py:100-103, 260`: `steps_per_day`, `servings_fruit_per_day`, `drinks_per_week`, `baseline_glucose`, `glucose_followup` → "time (medium)", which takes them out of the predictors. `rows.py:178-191`: `birth_weight` in grams (median about 3,300) → "design": "A sampling weight: its values are far too large for body weight."

**Recommendation.** "Time" only for columns whose values parse as dates or are small indices monotone within a unit; read `_per_day` and `_per_week` as rate units; check for grams and survey corroboration before proposing a weight. **Leash: too loose.**

#### IN-11 · major · Orientation is read from shape alone, so log-scale feature-by-sample tables are never asked about
*F4 and H13, both confirmed major.*

**Evidence.** The feature-major reading needs a spread ratio ≥ 4 *and* a row spread ≥ 0.4 (`orientation.py:75-77`; the second threshold has no stated basis), and the question is asked only when the reading is feature-major (`interview.py:196-206`). "Not applicable" is rendered as text with no "Ask me anyway" (`Record.tsx:738-763`). Log2 GEO-style matrices with ratios of 297–459 read "undetermined", with the false sentence "The rows and the columns … vary by similar amounts"; transposed VST (ratio 12), microarray (112–352) and logged metabolomics (24–25) fixtures likewise; MZmine-style exports with m/z and RT columns read 2.0–2.3. METABOLOMICS_PACK §01 ranks header tokens, feature-name grammar and m/z plausibility above shape and says "Never rely on shape alone."

**Recommendation.** Under an assay lens, ask whenever the reading is undetermined; implement the pack's cues; treat m/z and RT columns as feature metadata (see MA-04); replace the absolute spread gate with a scale-aware rule validated on real exports. **Leash: too tight** (a necessary choice hidden) **and too loose** (a destructive error passes silently).

#### IN-12 · major · Repeats versus time points is read from visit spacing, and a bare "visit" index is stated as replicates
*H14 and I9, both confirmed major.*

**Evidence.** `repeats.py:104` (14 days, "the shortest interval at which a clinical follow-up is normally booked", no source) and `:108` (CV 0.35, placed between two synthetic fixtures). A 10-day daily feeding time course with glucose falling 1.5 per day reads "repeats", stated (not asked), "too uneven to be a visit schedule" at a gap CV of 0; the menu then recommends the mean as "measurement-error reduction, not information loss". A weekly four-period crossover reads "repeats". Four 24-hour recalls 91 days apart read "time points": "averaging them destroys the signal". Visits numbered 1, 2, 3 with no dates are stated as replicates (`repeats.py:267-272`), although the module's own docstring says such an index "says there is an ORDER and says nothing about what the order means", and OPENING_SEQUENCE:238-240 says a visit label means time points and thin evidence is asked.

**Recommendation.** Ask unless the evidence is unambiguous; treat spacing as weak evidence; treat a within-unit treatment or period column and a monotone trend as time-point evidence; under the dietary lens treat recall indices as replicate evidence; fix the self-contradicting text. **Leash: too loose.**

#### IN-13 · major · Lens hints send most normalized omics, surveys and wide clinical tables to "metabolomics", and the contradiction check is never called
*H11, confirmed major.*

**Evidence.** `packs.suggest` hints genomics only for integer count matrices with ≥ 100 columns and metabolomics for any table with ≥ 30 numeric columns (`packs.py:6036-6081, 166`). Six of nine genomics fixtures, TPM and log2-TPM with Ensembl IDs, two survey fixtures, a DR1T* table and a 38-lab clinical table are all hinted "metabolomics"; two clinical tables get no hint. Under the hinted lens, surveys get "no pooled QC" and "no run order", and a VST matrix gets a critical "already transformed". `packs.contradiction` is called only from the legacy app. Hints are never pre-selected, which keeps this major.

**Recommendation.** Hint genomics from the gene-ID vocabulary and the data-type classifier; require positive metabolomics evidence (m/z or RT names, QC rows, run order); hint survey before "wide"; wire the contradiction check into the lens question as a block-and-record step. **Leash: too loose.**

#### IN-14 · major · The genomics "what your numbers are" card is silent on log2-TPM and log-CPM
*H12, confirmed major.*

**Evidence.** log2(TPM + 1) and voom log-CPM give an empty classification and no finding (`packs.py:3708-3770`). GENOMICS_PACK §02's table has no row for log2(x + offset) data, though its own coaching recommends limma-style workflows on exactly that. The thresholds `_ESTIMATED_COUNTS_CV` and `_FPKM_CV` (`packs.py:3503-3504`) were measured on fixtures the repository generates from one synthetic count table (F16, not re-tested: shallow raw counts under 10⁴ are also missed). The pack calls this card "the single most valuable artifact" and a wrong reading "the commonest real failure".

**Recommendation.** Add a log-expression signature with the limma-trend/voom capability set; validate every threshold on public matrices (GEO, recount3, GTEx); prefer integrality plus mean–variance shape to an absolute maximum. **Leash: too loose.**

#### IN-15 · major · Single-cell matrices are modeled as bulk samples
*F12, confirmed major.*

**Evidence.** GENOMICS_PACK:9 says single-cell "must be detected and refused". The legacy card detects it (`out_of_scope: single_cell`), but `_genomics_data_type` returns None for an unread card (`packs.py:4479`) and nothing in core, the server or the frontend reads the flag. `genomics_single_cell.csv` under the genomics lens yields only a p/n warning, an identifier note, constant columns and "The wide shape is expected for this kind of data; nothing needs changing". The legacy refusal test was not carried over.

**Recommendation.** Emit the out-of-scope reading as a finding by purpose: name pseudoreplication and pseudobulk aggregation for subject-level questions (block and record under inference), and allow cell-level prediction with a stated concern. **Leash: too loose** (the pack's outright refusal would be too tight for cell-type classification).

#### IN-16 · major · The instrument recognizer misses the instruments the pack names
*H17, confirmed major.*

**Evidence.** A Likert block needs at least 8 items (`packs.py:3113`) and modal share ≤ 0.60 in ≥ 70% of items (`:3066-3071`, measured on one synthetic fixture). A floor-heavy PHQ-9 (modal shares 0.44–0.95, the normal shape in community samples) gives no block, no hint and no survey finding; GAD-7 (7 items) and a 0–10 numeric rating scale are missed; a 9-point hedonic block is misread as 1–7 with 8 and 9 flagged (skeptic). CLINICAL_SURVEY_PACK §B1.1 names PHQ-9, GAD-7, EQ-5D and the 0–10 NRS.

**Recommendation.** Recognize instruments by shared name prefix with numeric suffix and inter-item correlation; allow floor-heavy items when prefix and scale agree; lower the item minimum to 3 when names corroborate; add the missing scale sets. **Leash: too loose.**

#### IN-17 · major · The run-order drift detector reports drift on drift-free data in small studies
*F8, confirmed major.*

**Evidence.** `packs.py:705-717` counts features with Pearson |r(order, log1p x)| > 0.3 and fires at a 15% share, with no significance test; the pack's spec (METABOLOMICS_PACK:167-168) is Spearman |ρ| > 0.3 *and* an FDR-significant trend. Pure noise, 200 features, 40 tables per n: fires 40/40 at n = 16, 37–39/40 at n = 20, 20–23/40 at n = 24, 2–3/40 at n = 30, 0/40 at n = 40. The pack's own rule fires 0/40 at every n.

**Recommendation.** Implement the pack's rule and report the expected null share beside the observed one. **Leash: too loose.**

#### IN-18 · major · One concentrated sample makes 300 independent metabolites look like 40
*F9, confirmed major.*

**Evidence.** `packs.py:1328` clusters features by Pearson r > 0.9 on raw intensities with single linkage. Three hundred independent log-normal features, n = 80, one sample multiplied by 20: "about 39–40 independent quantities", "overstates it by about 7.5–7.7×"; on the log scale no pair exceeds |r| = 0.65. The finding says the count feeds the manuscript's data description and the multiple-testing denominator.

**Recommendation.** Correlate on log (or rank) values; report how many clusters depend on a single high-leverage sample; use average or complete linkage. **Leash: too loose.**

#### IN-19 · major · NHANES weight advice always names the dietary day-1 weight, even when fasting-subsample analytes need a smaller one
*H15, confirmed major.*

**Evidence.** `nutrition.survey_weights_finding` (`nutrition.py:660-690`) knows only the dietary, examination and interview weights. A table with DR1TKCAL, LBXGLU, LBXTR, WTDRD1, WTMEC2YR and WTSAF2YR gets "Use the dietary weights, not the examination weight" with SETTLED evidence and `may_preselect` True, and WTSAF2YR is never mentioned. NHANES weighting tutorial: "Use 'the least common denominator' where the variable that was collected on the smallest number of respondents is the 'least common denominator'"; "You would use the fasting subsample weights (wtsaf4yr)". (No estimate is weighted yet, so today this misleads the record rather than a number; see §7.3.)

**Recommendation.** Recognize the subsample weights (WTSAF*, WTSA*, WTSB*) and fasting analytes; apply the least-common-denominator rule; badge CONVENTION or ask when components with different weights are combined. **Leash: too loose.**

#### IN-20 · major · The energy finding says adjustment "is not in dispute", and no detector notices energy-related outcomes
*D18 and G5, both confirmed major.*

**Evidence.** `finding_words.py:231-234, 516-521` and the SETTLED claim at `packs.py:2808-2812`: "every nutrient association is confounded by total intake; that adjustment is needed is not in dispute … the standard and residual methods estimate the same substitution". Tomova et al. 2022: total energy can be conceptualized as a "collider"; "Adjusting for TE opens conditional dependencies between the exposure and all competing energy sources"; adjusting for total energy and for remaining energy "evaluate very different causal estimands". The app's own DISPUTED card for BMI and adiposity outcomes (`content.py:236-241, 877-880`) and the pack (NUTRITION_PACK:440-441, 458-462: "Detect whether the outcome is itself energy-related … escalate the mediation/collider warning") are never acted on by any detector.

**Recommendation.** Restate the finding as an estimand choice ("adjusting for total energy turns the question into a substitution; leaving it out, or partitioning, asks about adding energy"); badge it by purpose and outcome; detect weight, BMI, waist and diabetes outcomes and push the dispute on the energy card; offer both the adjusted and the truly unadjusted model once ME-02 is fixed. **Leash: too loose.**

#### IN-21 · major · The multivariable density model is presented as clean "diet composition" with no caveat
*B9 and G6 (both confirmed major); D17 (minor, not re-tested) agrees.*

**Evidence.** `energy.py:109-117` gives `density_multivariate` the kind "composition", the estimand "Diet composition…" and no caveats; the option reads "diet composition" (`content.py:825-826`). Every other adjusting method carries the partial-confounding caveat. Tomova et al. 2022: Table 2 lists its estimand as "Obscure"; "the multivariable nutrient density model returns a more accurate estimate than the (unadjusted) nutrient density model, but one which is still biased"; its coefficient "conflates both the effect of the nutrient exposure and that of the reciprocal of total energy … expected to be misleading and experience composite variable bias regardless of confounding".

**Recommendation.** Give it Tomova's caveat; keep the "composition" label only with that qualification; name the all-components model as the source's recommended route (ME-14). **Leash: too loose.**

#### IN-22 · major · "The mean of recalls is attenuated but unbiased in direction" is false for the models the app fits, and results carry no measurement-error caveat
*D12 and G9, both confirmed major.*

**Evidence.** `content.py:491-492` (the visible "why", unqualified) and `:513-517` (CONVENTION, "under classical error"). The app's models hold several error-prone nutrients and, under most energy methods, energy too. STRATOS Part 1 (Keogh et al. 2020, §3.1.3): with several error-prone covariates "the estimated coefficients … may be larger or smaller than the true target values in a rather unpredictable manner". Freedman et al. 2011 (*JNCI*): with "two or more mismeasured exposures, estimated relative risks may become attenuated, inflated, or can even change direction." No attenuation or calibration code exists in core.

**Recommendation.** Qualify the claim ("toward zero only for a single error-prone exposure under classical error"); under inference attach a limitation line (number of recall days, no calibration) to the coefficient table and the methods; prioritize univariate regression calibration of energy-adjusted intakes when repeats exist (Freedman: "Univariate measurement error adjustment for energy-adjusted intake variables used in the density or residual models is recommended"). **Leash: too loose for inference; right for prediction.**

#### IN-23 · major · "Too numerous to count" is taught as a measurement failure; it is a value above the countable range
*G8, confirmed major.*

**Evidence.** `content.py:209-212` (SETTLED): "TNTC and QNS are measurement failures, not censoring … treat them as missing"; `clinical.py:135-140` puts `tntc` among failure tokens. FDA Bacteriological Analytical Manual, ch. 3: "When number of CFU per plate exceeds 250, for all dilutions, record the counts as too numerous to count (TNTC)…"; crowded plates are estimated "as greater than 100 times the highest dilution plated". Treating TNTC as missing deletes exactly the highest values.

**Recommendation.** Classify TNTC as right-censored at the laboratory's upper count limit (ask for the limit if not stated), handled like ">ULOQ"; keep QNS and hemolyzed as failures; fix the pack and the badge.

#### IN-24 · major · The temporal methods sentence says the models were "scored on later data", but most held-out rows can be earlier
*A9, confirmed major.*

**Evidence.** The chronological draw holds out whole units ranked by their last observation (`seal.py:276-321`), which is correct; the methods sentence adds "so the models are scored on later data than they learned from" (`voice.py:671-673`). With visits spread over time, 56–71% of held-out rows were observed before the latest training row (the share depends on how spread the visits are); the earliest held-out visit was 1.2 against a latest training visit of 92.5. The bracketed description and the seal's own chronology sentence are accurate.

**Recommendation.** Describe what was drawn ("units whose last observation is latest were held out whole, earlier rows included"), report the share of held-out rows that predate the training boundary, and offer a row-time cut (train before T, test after T) when the purpose is forecasting.

#### IN-25 · major · The lineage figure, the exported provenance, shows operations that did not happen
*B15, confirmed major; D15 (minor, not re-tested) is the same pattern in a methods sentence.*

**Evidence.** `models/lineage.py:132-133` applies a step's verb to every output, and `:145` merges operations across parents. The figure shows "one-hot, blank as a level" on columns the encoder passed through; "imputed" on edges from columns with no missing values; and, for the stratified log residual, "at that level's mean energy" where the number is the geometric mean (2006 against an arithmetic 2112), with "predicted fat_g" where the quantity is exp(predicted log fat_g). D15: the density method records "divided by `kcal` within levels of `gender`", a stratification that changes nothing.

**Recommendation.** Mark outputs with unchanged names and values as "kept"; attribute each operation only to the column it touched; say "geometric mean" under log; drop density from the stratifiable methods. (BLUEPRINT north star 4 makes this figure a deliverable.)

#### IN-26 · major · The survey pack's attenuation guidance mixes the reliability with its square root
*F10, confirmed major.*

**Evidence.** CLINICAL_SURVEY_PACK:1085 and :1091 say a scale "attenuates its estimated effect by approximately its reliability. With ω = 0.70, a true standardized coefficient of 0.30 is expected to show up as roughly 0.25". Simulation (n = 400,000, reliability 0.70): the unstandardized slope is 0.210 (0.30 × λ); the standardized coefficient is 0.251 (0.30 × √λ). Dividing the standardized coefficient by λ, as the text implies, gives 0.358 against a truth of 0.30. Not yet served in Next; BLUEPRINT's M3 plans "survey scale scoring with reliability and attenuation".

**Recommendation.** Correct the pack: an unstandardized slope is attenuated by λ (divide by λ); a correlation or standardized coefficient by √λ (√(λx·λy) when the outcome is also a scale). Add a replay test with known λ before M3 ships the correction.

### 2.4 Routing

#### RO-01 · critical · An eligibility rule on the outcome itself is accepted, and its preview draws the outcome's histogram
*I1, confirmed critical.*

**Evidence.** The exclusion validator checks only that the column exists, is numeric and has bounds (`decisions.py:805`); a rule `bmi 18.5–30` with `bmi` as the outcome is accepted under inference and under prediction. Its preview's second view is "Where the cut falls on `bmi`", the outcome's own histogram with the cuts marked (`row_previews.py:138`); only the coach line is suppressed (`coach.py:228`). True slope −0.25: all rows −0.236 to −0.257; after the accepted rule **−0.117 (−0.132, −0.102)** (auditor) and **−0.092 (−0.104, −0.080)** (skeptic), tight intervals around a truncated answer, with no flag. ROADMAP §04 (`:388-389`) frames the question as "does your research question restrict the outcome range?", inviting the error. Under prediction, an outcome-range rule cannot be applied at deployment.

**Recommendation.** Refuse any eligibility rule on the target, with exits: the impossible-values repair for impossible outcome values, or eligibility defined on baseline variables. Under inference, a disease-definition restriction may be admitted as block-and-record with an attestation carried as a limitation. Rewrite ROADMAP §04 as "does your research question restrict who is studied?". **Leash: too loose.**

#### RO-02 · critical · Changing outcome values after the split silently re-draws the held-out rows
*I2, confirmed critical.*

**Evidence.** The split is drawn over "every row with the outcome measured" (`rows.py:571, 848`) by permuting positions (`rows.py:669`), so any change to that set reshuffles membership: removing 3 of 1,000 rows kept only 76 of 200 held-out rows. On the real server, after the split and a fresh fit, the set-missing repair on three impossible `sbp` outcome values was accepted with the sentence "`3` physiologically impossible values in `sbp` were set to missing." Of 200 held-out rows only 41–70 stayed; **119–130 rows whose outcomes had trained the inspected fits became "sealed"**. The basis and the split step were unchanged, and no sentence said the seal was re-drawn. `set_target` and `set_task` after the split are accepted; only orientation, grain, unit and aggregation are refused (`seal.py:813`). ROADMAP §01 puts the impossibility pass before the seal, but the Router does not order repairs before the split. The stated invariant (`rows.py:729-731`, `row_previews.py:62-63`) holds for exclusions and missing-value answers only. Dwork et al. 2015: "Reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself." (The resulting optimism is from selection only and often small; the false "never seen" claim is the defect.)

**Recommendation.** Make membership a stable function of row identity (a hash of seed and row ID, by unit when grouped) over the whole oriented table, so outcome repairs and target changes cannot move a row; treat outcome-value repairs and `set_target`/`set_task` after the split as structural (refuse with the re-seal exit) or record a "re-sealed" event that the seal and the methods state; order the impossibility and sentinel repairs before the split. **Leash: too loose** (a lockbox violation with no rung at all).

#### RO-03 · critical · Censored follow-up is never recognized; a cohort is modeled as a yes/no outcome
*I3, confirmed critical.*

**Evidence.** No gate, question or finding reads follow-up or censoring; the follow-up column is proposed as "time" and leaves the model; the task is skipped as binary. Staggered entry, true fiber hazard ratio 1, later entrants eat more fiber and are followed for less time: the app reports fiber log-odds **−0.065 to −0.081**, p = 10⁻¹³ to 10⁻²², concerns `[]`; Cox: HR 1.012 (0.997–1.027), p = 0.12 (skeptic: log-HR −0.004, p = 0.6). PROBAST explanation, item 4.6: "For prognostic models to predict long-term outcomes in which censoring occurs, a time-to-event analysis, such as a Cox regression, should be used… Use of logistic regression models that simply exclude censored participants with incomplete follow-up is inappropriate." (The app does worse: it counts censored people as non-events.) BLUEPRINT north star 1 lists time-to-event.

**Recommendation.** Ask when the outcome is binary and a time or follow-up column exists: "Did everyone have the same follow-up, or could some leave, or the study end, before the event?" If follow-up varies, route to a time-to-event model (Cox, M3) or to a fixed-horizon outcome excluding those censored before the horizon, stated as a limitation. Until a survival family exists, block and record under both purposes. **Leash: too loose.**

#### RO-04 · critical · Under inference the app never asks which exposure, which estimand, or which covariates are confounders
*E3 (critical) and I8 (major), both confirmed.*

**Evidence.** No decision kind asks for an exposure of interest, an estimand, or a causal role; "confounder", "mediator" and "collider" appear in core only in teaching prose. Unrecognized columns default to exposure ("read as an exposure until you say otherwise", `rows.py:277`), everything else to covariate. NHANES export under inference with the proposed roles: the covariates include weight, height, BMI (a deterministic function of the two; its interval −2.24 to +2.04), waist, HDL, triglycerides, blood pressure and medications, which are plausible mediators of diet → glucose; seven nutrients are mutually adjusted, total fat beside its parts; total fat moves from +0.43 to +0.15–0.18 depending on these defaults; concerns `[]`. On `clinical_risk.csv`, comorbidity index, prior admissions, laboratory values and length of stay all default to "exposure" and are drawn as mutually adjusted effect estimates. The plot is titled "Coefficients of the exposures" even when it falls back to all covariates (`model.ts:115`, `Results.tsx:159`). (The forest plot does hide covariates when exposures exist; skeptic nuance.) Westreich & Greenland 2013: presenting exposure and confounder estimates from one model invites "confusion of direct-effect estimates with total-effect estimates". VanderWeele 2019: "Statistical analyses cannot in general distinguish between confounders, which ought to be controlled for in the estimation of the total effect, versus mediators, which ought not be controlled for."

**Recommendation.** Make inference a different path: require a primary exposure (or a named set stated as mutually adjusted direct effects), the estimand (total or direct effect, contrast, population), and a role per covariate (confounder, mediator, collider, precision variable, effect modifier, or an imported DAG); block and record mediators or colliders in a total-effect adjustment set; show the primary exposure's estimate by default; word the caption from the estimand. **Leash: too loose.**

#### RO-05 · major · After opening, a new seed or a new outcome serves fresh held-out scores at once, and the opened score is not kept
*I11, confirmed major.*

**Evidence.** No validator blocks `set_split`, `set_target`, `set_task`, `set_temporal` or `set_purpose` after opening; held-out scores are served whenever the seal is open (`seal.py:663-689`). Opened at AUC 0.80; seeds 1–5 then gave 0.62–0.72, each served at once; a new outcome's held-out R² was served without ever being sealed. The app does mark these (a "changed after the seal was opened" band, `post_seal` flags; skeptic correction), but nothing stores the at-opening scores, while the screen says the held-out metric "is then fixed in the record" (`OpenSeal.tsx:129`).

**Recommendation.** Freeze the at-opening scores as the reportable result in the Results and the export; show later scores as "post-seal, not independent"; require a recorded re-seal decision to re-draw; start a new seal for a new outcome, or refuse with "start a new analysis". **Leash: too loose** (mark-only where block-and-record is warranted).

#### RO-06 · major · The declared purpose barely changes anything, and no option carries the "customary" and "sound" labels
*G7 and I6, both confirmed major.*

**Evidence.** No gate in `interview.py` and no validator in `decisions.py`, `sequence.py` or `seal.py` reads the purpose; its only consumers are the shelf order, the pipeline description and the coefficient-interval switch (`modeling.py:107, 219, 446, 470`; `linear.py:124-157`). A search for "customary", "soundness" or "sound_for" across core and the frontend finds nothing. The energy menu orders by "usual"; the missing-values "recommended" tag ignores purpose (`ChoiceQuestions.tsx:254`). Indicators are accepted under inference against the spec. The PURPOSE card promises "Some advice even flips". The leash commit at HEAD changed only BLUEPRINT.md.

**Recommendation.** Add to the teaching schema, per option: *customary in <field>* (with a source) and *sound for <purpose>* (with a reason); order by soundness for the declared purpose; emit a one-line tension note when the two disagree; record departures in the methods sentence. Start with energy adjustment, missing values, the split and exclusions, and make purpose an input to gates and validators. **Leash: too loose for inference across the board.**

#### RO-07 · major · The energy menu puts the residual method first whatever the purpose
*D4 and the ordering part of B8, both confirmed major.*

**Evidence.** `USUAL_METHOD = 'residual'` is chosen with no purpose input (`proposals.py:51, 406`), and the card lists it first with "none" last (`ChoiceQuestions.tsx:399-410`). Under prediction (5-fold CV, outcome related to energy), methods that keep energy reach R² 0.575–0.789 (linear) and 0.54–0.76 (boosted trees); residual reaches 0.01–0.39 and 0.08–0.37; density alone 0.01–0.40 and −0.13 to 0.36. The tag says "usual", not "recommended", which is honest about custom; the defect is ordering by custom with no concern under prediction.

**Recommendation.** Under prediction, rank energy-keeping methods first and say that residual and density discard energy. Under inference, rank all-components (ME-14), then residual + energy or standard, and name the tension ("residual is the field's default; it estimates the same quantity with less precision").

#### RO-08 · major · Clustering above the person (site, centre, household, batch) is never asked
*I5, confirmed major.*

**Evidence.** The only clustering question is whether a person can appear in more than one row. On a 10-site cohort the grain is skipped ("every participant appears once"), `site` becomes "design" (IN-02), and the holdout is drawn by row. Within-site sodium effect 0: app 0.76–0.90 (p ≈ 10⁻³⁴ to 10⁻⁵¹), concerns `[]`; site fixed effects with site-clustered errors −0.04 (−0.11, 0.03) to −0.08 (−0.15, −0.005). Collins et al. 2024: "Internal-external validation exploits a common feature present in many datasets, namely that of clustering (eg, by centre, geographical region, or study)."

**Recommendation.** After the grain question, ask whether participants are grouped in sites, centres, households or batches. Under inference: adjust and cluster, or record why not. Under prediction: offer internal–external (leave-one-cluster-out) validation and grouped folds. **Leash: too loose.**

#### RO-09 · major · Combining time points accepts predictors summarized after the outcome
*I12, confirmed major.*

**Evidence.** The aggregation validator checks only that the outcome is named, varies, and is numeric for "mean" (`sequence.py:248-291`). With time points, predictors = last, mean or change with outcome = first are all accepted, under both purposes. The aggregation coach says nothing about temporal order.

**Recommendation.** Under prediction refuse any combination whose predictor summary uses rows later than the outcome row (exits: first predictors with last outcome; keep rows with the temporal split); under inference block and record with the reverse-causation concern. **Leash: too loose.**

#### RO-10 · major · The outcome type is skipped as a fact when it is a choice
*I10, confirmed major.*

**Evidence.** Any text column is read as classification at high confidence (`ml/triage.py:180-184`) and the task question is skipped (`interview.py:378-381`): `none/mild/moderate/severe` becomes unordered multiclass, and the ordinal caveat is never shown. Markedly skewed biomarkers (hs-CRP skew 3.2–3.9, BNP 4.4, troponin 4.2) are skipped as regression; no question anywhere offers a log scale, a ratio estimand or a count model. A text outcome with more than 20 levels is skipped as multiclass, while an explicit multiclass answer is refused above 20 (`decisions.py:712, 736-762`).

**Recommendation.** Skip at high confidence only for 0/1 outcomes and continuous outcomes that are not markedly skewed; for text with 3–10 levels ask "are these levels ordered?"; add an outcome-scale step for positive skewed outcomes (inference: log scale, stated as a ratio of geometric means; prediction: the scale and its back-transformation); reconcile the 20-class rule. **Leash: too loose.**

#### RO-11 · major · The lens question has no "something else / not sure"
*I15, confirmed major.*

**Evidence.** `decisions.py:48, 78`: a lens list of at least one of five values; the frontend offers the five. Findings, roles and the working table are blocked until a lens is chosen, and the forced lens changes role defaults (omics makes numeric columns exposures; dietary reads nutrients as exposures). OPENING_SEQUENCE:146-151: "'Something else, or not sure' is first-class: the app is fully functional with no lens."

**Recommendation.** Add it; it runs the generic checks only and is recorded and stated. **Leash: too tight.**

#### RO-12 · major · Under inference, p-values are live while the plan is still being changed
*E7, confirmed major.*

**Evidence.** Only held-out scores are withheld (`seal.py:663-689`); coefficients with p-values are served after every change of roles, exclusions, missing strategy or energy method (`modeling.py:476-486`), and with no holdout nothing is ever sealed. Post-seal marking applies only after opening (`seal.py:769-775`). Twenty to twenty-one p-values print with no multiplicity note. Gelman & Loken 2013: "Researcher degrees of freedom can lead to a multiple comparisons problem, even in settings where researchers perform only a single analysis on their data", with forks including "choices of control variables in a regression, transformations, and data coding and excluding rules".

**Recommendation.** An inference seal: estimand, exposure, adjustment set, exclusions and missing-data plan are sealed before the first coefficient is shown; later changes are marked "after the estimates were seen" in the record and the methods sentence; the primary exposure is shown by default; a pre-specified sensitivity panel replaces ad hoc toggling. **Leash: too loose.**

#### RO-13 · major · Pooled QC injections raise a critical finding with no lever, and the served text says they are already excluded
*F7, confirmed major.*

**Evidence.** On `metabolomics_untargeted.csv`, `pooled_qc` is critical with "No control for this yet", and `sample_type` is proposed as a covariate. With the usual export shape (Class = Case/Control/QC), `pooled_qc` does not fire (it needs exactly two levels, `packs.py:757`), the task is stated as three-class multiclass with QC as a class, a binary answer is refused, and exclusions accept only numeric ranges (`decisions.py:805-818`). The served text claims QC rows "stay in the table for quality assessment and out of the modeling rows" (`packs.py:782-785`). (A numeric rule keyed by a text column might work through the API; untested.)

**Recommendation.** A row-role exclusion by level of a text column, before the seal, offered from `pooled_qc` and `sample_roles`; filter the outcome's class levels accordingly; correct the text. **Leash: menu too tight, guidance too loose.**

### 2.5 Confirmed minor findings (re-tested and lowered from major)

| ID | Raw | Finding | Why minor |
|---|---|---|---|
| MI-01 | C9 | Histograms of discretized data show a sawtooth and empty bins, and the two histogram implementations disagree on the same column (`datastore.py:1575-1580`; `consequences._histogram_pair`). | The counts feed only pictures: no decision, finding or published number. |
| MI-02 | D8, G4 | "Willett, by sex: women 500–3,500 and men 800–4,200" (`proposals.py:342-349`; `content.py:643-644, 669-673`) and "Willett and the Nurses' Health Study" for men. Willett 2013, quoted in Banna 2017, gives 800–**4,000** for men; 800–4,200 is the HPFS rule (de Koning 2011); NHS enrolled only women. | The numbers are printed beside the name, a 4,000 rule can be entered as custom, and 800–4,200 is widely called "Willett's". Relabel and add a Willett 2013 preset. |
| MI-03 | G12 | The coach says a nutrient is "mostly how much people eat" whenever \|r\| ≥ 0.3, where energy explains as little as 10% of its variance (`coach.py:301-303`). | The true r is printed in the same note and no number depends on it. Say r², and "mostly" only above r² = 0.5. |
| MI-04 | H8 | Diastolic 0 (from the SAS-zero repair) called "physiologically impossible … almost certainly entry errors" with SETTLED evidence. The NHANES BPX_I documentation allows it ("Diastolic BP can be zero"; range 0 to 120). | Setting it to missing is the usual convention; only "entry error" is wrong. The floor of 15 also flags 10 and 14 mmHg, which the pack's own table treats as plausible. |

---

## 3 · Custom versus sound, method by method

North star 5 asks every option to carry two independent labels: *customary in <field>* and *sound for <your purpose>*. No option carries them today (RO-06). This table supplies them for every method the app offers or should offer. "Customary" cites where the practice is documented as common; "sound" gives the reason. "App today" says what the app does. Rows marked **absent** are options a methods reviewer would expect.

### 3.1 Energy adjustment and substitution

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| No adjustment: energy out of the model (Y ~ N + C) | Yes, as the crude model beside the adjusted one (NUTRITION_PACK §08) | As an absolute-intake estimand, only if energy truly leaves the model; confounded by energy for most questions (Tomova 2022) | Rarely: it drops the strongest predictor | As a stated crude or sensitivity model; for adiposity outcomes beside the adjusted model (pack, DISPUTED) | **Mislabeled**: keeps energy, so it is the standard model (ME-02) |
| Standard (Y ~ N + E + C) | Yes (Willett; Tomova) | Estimates a substitution for the average of the omitted sources; composite-variable bias "even in the absence of confounding" (Tomova) | Good: keeps energy | Acceptable, with the omitted sources named (ME-04) | Offered |
| Residual, energy dropped (Y ~ N_adj + C) | Yes, the field default (pack §04, Willett) | Equals standard only with no energy-correlated covariate; otherwise a different number, sign can flip; wider intervals (ME-03) | Avoid: discards energy (CV R² 0.01–0.39 vs 0.58–0.79) | Avoid in this form | Offered first, labeled "same as standard" |
| Residual plus energy (Willett–Stampfer variant) | Yes (McCullough & Byrd 2023) | Identical coefficient to standard; nutrient stays in its own units | Fine | The sound form of the residual | **Absent** |
| Log residual | Yes for skewed intakes (pack §04 default) | Sound only on the log scale with its own label; back-transformed with energy dropped it is a near-density (B6) | Fine if energy kept | Label as an energy-elasticity adjustment | Hidden in the UI; mislabeled through the API |
| Within-sex residual | Yes (pack §04) | Only with one common constant or with sex in the model; otherwise manufactures effects (MA-02) | Keep sex in the model | Pooled constant or sex as covariate | Unsound when sex is not a predictor |
| Density alone (N/E) | Yes, "weakest" (pack) | Obscure estimand, severely biased (Tomova) | Rank low (drops energy) | Avoid | Offered with caveat |
| Density plus energy | Yes | Still biased, estimand "Obscure" (Tomova Table 2) | Fine (keeps energy) | Below standard and all-components, with caveat | Offered as "diet composition", **no caveat** (IN-21) |
| Partition (kcal from each source) | Yes | Total causal effect; unbiased only without confounding or with equal effects of other sources (Tomova) | Fine | For total-effect questions | Offered, labeled correctly |
| All-components (every source as its own term; relative effect by weighted difference) | Emerging (Tomova 2022; Chiu & Wen 2026); disputed by Willett/Stampfer/Tobias 2022 | Sound for total and average relative effects in simulation, at a precision cost | Information-equivalent to keeping energy | Rank first and name the dispute | Implicit (partition over all sources), **unnamed**; relative effect never computed (ME-14) |
| Substitution with every energy source in the model (leave-one-out, kcal) | Yes (pack §05a) | Sound for single swaps | A model contrast; fine | Sound; name the omitted source | Offered, but completeness never checked (ME-05) |
| Substitution with two nutrients plus energy | Common | Confounded by the omitted composite (57% bias in a test) | Fine as a model contrast | Unsound for the stated estimand | Offered silently |
| "5% of energy from X replaced by Y" (leave-one-out in %E) | Yes, "the field's expected figure" (pack §05) | Sound with all components in the model | Fine | Sound | **Absent** (B24, D19) |
| Compositional log-ratio (ilr) models | Emerging (Dumuid) | Sound; needs a zero-handling rule | Fine | Sound | **Absent** |

### 3.2 Exclusions, repeated recalls and measurement error

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| Fixed kcal cut-offs (Willett 2013: 500–3,500 women, 800–4,000 men; HPFS: 800–4,200 men; 500–5,000; 500–3,500) | Yes (Banna 2017; Rhee 2015) | Weak for misreporting: "not individualized and does not capture all implausible reports" (Banna); designed for FFQs, applied here to single recall days | State that excluded reporters remain in the deployment population | As primary with a sensitivity analysis | Offered; attribution off (MI-02); kJ mishandled without macronutrients (IN-07) |
| Goldberg / Black EI:BMR screen | Yes in the misreporting literature | More individualized; reduced bias in 14 of 24 simulated pairs, not in 10 (Yamamoto 2023) | Optional | Offer beside fixed cut-offs | **Absent** (ME-16) |
| Analysis with and without exclusion | Recommended (Banna 2017) | Sound: shows how much the rule moves the answer | Useful | Expected | **Absent** |
| Mean of replicate recalls | Yes | Sound for ranking and prediction; for inference the bias direction is not guaranteed with several mismeasured covariates (STRATOS) | Fine | Fine with a stated limitation | Offered; the claim overstates (IN-22) |
| Usual-intake modeling / regression calibration | Yes in NCI-led work | Sound for effect magnitudes (Freedman 2011) | Not needed | Expected for magnitudes | **Absent** (honestly stated) |
| Change score (last minus first) | Yes | Sound when baseline adjustment is not needed; per-column rules required (MA-14) | Rarely | Contested against ANCOVA (§7.3) | Applies to every numeric column |

### 3.3 Missing values and values below detection

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| Complete cases | Yes in epidemiology | Unbiased when missingness does not depend on the outcome given covariates; costs rows | Wasteful but valid | Valid with the assumption and row loss stated | Offered; row loss never flagged (E14) |
| Single median fill, outcome excluded | Yes in ML pipelines | Consistent for prediction when deployment imputes the same way (Josse; Sisk 2023); biased with 0% coverage for inference (ME-01) | Fine | Block and record | The only imputer, purpose-blind, unnamed in the methods |
| Imputation conditional on energy | Implied by the pack's anti-pattern list | Keeps the nutrient–energy relation every later step depends on | Better than median | Part of MI | **Absent** |
| Missing indicator / "blank as a level" | Yes in EHR prediction | Sound for prediction (Sperrin 2020; harmful under outcome-dependent missingness, Sisk 2023); biased for observational inference, valid for baseline covariates in RCTs (Groenwold 2012) | Allow | Block and record (as the spec already says) | Allowed under inference; "recommended" tag purpose-blind |
| Multiple imputation with the outcome (MICE, Rubin's rules) | Yes in epidemiology | Sound for inference (Moons 2006 via Harrell; Groenwold) | Development only; omit the outcome when imputing at deployment (Sisk) | Recommend first | **Absent** |
| Half-minimum for values below detection | Yes (MetaboAnalyst default) | Modest bias for associations; deflates variance | Compare by CV | Customary comparison scheme | **Absent** |
| QRILC / MinProb | Partly | Best for value recovery (Wei 2018); attenuates associations as censoring grows | Compare by CV | Not the default | **Absent** (planned M3 default; ME-08) |
| Censoring-aware estimation (detected-only, Tobit, outcome-conditioned MI) | Less common | Sound for associations | Optional | Recommend | **Absent** |
| Too-numerous-to-count as right-censored | Yes (FDA BAM) | Sound | — | — | Treated as a failure, i.e. missing (IN-23) |

### 3.4 Validation and performance

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| Single random holdout | Yes in ML and many journals | Weakest internal validation (Collins 2024; Steyerberg 2001, 2018; PROBAST 4.8); defensible as an analyst lockbox at large n | Not first at moderate n | Serves no estimand | Leads above the floor |
| One k-fold run | Yes | Noisy at small n (Varoquaux 2018) | Acceptable | — | Offered |
| Repeated k-fold | Yes | Reduces partition variance | Rank first at small n | — | **Absent** |
| Bootstrap optimism correction (Harrell) | Yes in clinical prediction | Sound; uses all rows (Steyerberg 2001) | Rank first at small n | — | **Absent** |
| Nested CV over the family choice | Yes (Varma & Simon 2006) | Removes selection optimism | Recommend when picking among families | — | Only elastic net's penalty is nested (ME-13) |
| Internal–external (leave-one-cluster-out) | Increasingly (Collins 2024; TRIPOD+AI 12d) | Shows heterogeneity across sites or cycles | Recommend for multi-site data | — | **Absent** (E16) |
| Chronological holdout with time-ordered folds | Yes | Sound when folds and tuning also respect time | Recommend for forecasting | — | Holdout only; folds shuffle time (MA-11) |
| CV R² averaged per fold | Yes (scikit-learn, caret) | Biased low at small n (Hawinkel 2024) | Replace with pooled R² | — | Used |
| AUC alone to rank models | Yes | Unsound alone (Van Calster 2019) | Add calibration and a proper score | — | AUC primary; banner "best" by AUC |
| Calibration intercept, slope and curve | Expected (TRIPOD+AI) | Sound | Required | — | **Absent** (ME-10) |
| Intervals on performance (DeLong, bootstrap) | Expected (TRIPOD+AI 23a) | Sound | Required | — | **Absent** |
| Macro-F1 as the multiclass primary | Yes | Not a proper scoring rule | Log loss primary, F1 secondary | — | Macro-F1 primary |
| Baseline comparison by one fold-SD | App's own | Anti-conservative (Bengio & Grandvalet 2004) | Corrected resampled t | — | Used (MA-10) |
| EPV ≥ 10 for every purpose | Yes (Peduzzi 1996) | Not appropriate for prediction (van Smeden 2019); OLS needs about 2 per variable (Austin & Steyerberg 2015) | Riley criteria | EPV is closer to justified for logistic coefficients | Used for both (A21, E12) |

### 3.5 Inference: intervals and estimands

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| Classical OLS standard errors | Yes | Coverage 0.59–0.85 when variance grows with intake (MA-07) | — | Below HC3 | The only option |
| HC3 robust standard errors | Increasingly | Sound in small samples (Long & Ervin 2000, via secondary sources) | — | Recommend | **Absent** |
| Ignoring repeated rows | Common in applied papers | Unsound (Cameron & Miller 2015) | Held-out scores already marked exploratory | Refuse or cluster | Happens when grouping is abandoned or undetermined (MA-01) |
| Cluster-robust with z critical values | statsmodels default | Unsound with few clusters | — | Below t(G−1) | Used from 8 clusters (MA-06) |
| Cluster-robust with t(G−1); CR2; wild cluster bootstrap | Stata uses t(G−1) | CR2 + Satterthwaite or wild bootstrap sound with few clusters (Cameron & Miller) | — | Recommend | **Absent** |
| Random-intercept mixed model / GEE | Yes for repeated measures | Sound | — | Recommend for repeated measures and small G | **Absent** |
| Design-based survey variance (weights, strata, PSU) | Yes, the NHANES standard (NCHS) | Sound for population inference | State unweighted | Required for population estimands | **Absent**, silently (ME-06) |
| Logistic maximum likelihood under separation | Yes | Infinite estimate; Wald p meaningless | — | Detect and refuse the Wald row | Prints p = 1.0 (MA-08) |
| Firth logistic / profile-likelihood intervals | Yes for sparse data | Sound | — | Offer at low events per variable | **Absent** |
| Odds ratios (or risk ratios) with intervals | Yes in nutrition and clinical papers | Sound if labeled | — | Expected | Log-odds, mislabeled (ME-07) |
| Every coefficient as an effect ("Table 2") | Yes | Unsound for causal claims (Westreich & Greenland 2013) | Coefficients not interpreted (app handles this) | Primary exposure only | Every column drawn (RO-04) |
| Coefficients from training rows only | ML habit | Loses precision; estimate depends on the seal seed | — | Use all analyzed rows | Training rows only (ME-12) |

### 3.6 Model families, exposure form and outcome scale

| Method | Customary? (source) | Sound? (why) | Under prediction | Under inference | App today |
|---|---|---|---|---|---|
| Linear in each exposure | Yes | Sound only when linear | Fine for linear families | Check | The only form |
| Restricted cubic splines | Increasingly, "near-default" (pack §07G) | Sound | Optional | Rank first | **Absent** (ME-17) |
| Quintiles with a trend test | Yes, "expected alongside" | Loses information | — | Offer, tagged customary | **Absent** |
| Elastic net | Yes | Sound for prediction; no intervals | Good | Not for inference; say so at p ≥ n | Offered; caveat dropped at p ≥ n (ME-18) |
| Boosted trees | Yes | Sound for prediction; often miscalibrated | Good, with calibration | Not for coefficients | Offered |
| Feature-wise regression with BH-FDR (limma-style, MWAS) | Yes in omics | Sound for omics inference | — | Required | **Absent** |
| PLS-DA with VIP > 1 | Yes in metabolomics | Sound only with nested CV and permutation; VIP ranks, it does not test | Optional with nested permutation | Not a test | **Absent**, unexplained |
| Proportional-odds (ordinal) model | Yes for Likert outcomes | Sound | Fine | Expected | **Absent** (ME-19) |
| Cox / time-to-event | Yes with censoring | Sound (PROBAST 4.6) | Required with censoring | Required | **Absent**; censored data modeled as binary (RO-03) |
| SMOTE / resampling for imbalance | Yes in ML-nutrition papers | Damages calibration without improving AUC (van den Goorbergh 2022) | Avoid, or recalibrate | — | Absent, correctly; the tension is never shown (E15) |
| Log scale for skewed biomarker outcomes | Yes | Sound for ratio estimands | Choose with back-transformation | Ratio of geometric means | **Absent** (RO-10) |
| Integer codes as categories | Yes | Sound | Needed | Needed | **Absent** (MA-15) |
| Autoscaling vs Pareto scaling | Pareto is customary in metabolomics | Autoscaling performed better (van den Berg 2006) | Fine | Fine | Autoscaling, unstated (F17) |
| Omics normalization: log-CPM/TMM, VST; PQN + log | Yes | Sound; normalizing at all matters most (Hornung 2015) | Required | Required | **Absent** (ME-09) |
| Batch: correct-then-test (ComBat) | Yes | Unsound under batch–outcome confounding (Nygaard 2016) | — | Avoid | Absent |
| Batch as a covariate | Yes in biostatistics | Sound | Optional | Default | Batch dropped as "design" (IN-02) |

---

## 4 · The leash, decision by decision

BLUEPRINT §11.3: a leash is **too tight** when it hides or refuses a defensible choice or lectures where the researcher knows best; **too loose** when it offers an unsound choice without saying so or stays silent where errors are likely. The rungs are *refuse*, *block and record*, and *rank and state the concern*. Grades are given for prediction (P) and inference (I) where they differ.

| Decision | Grade | Why | Proposed menu | Proposed guidance |
|---|---|---|---|---|
| Lens | Too tight and too loose | No "not sure" (RO-11); hints misroute (IN-13); a contradicted lens is never checked | The five lenses, several allowed, plus "something else / not sure" | Hints from positive evidence only; a contradiction is block-and-record |
| Orientation | Too tight and too loose | Hidden whenever the reading is undetermined (IN-11); annotation columns become samples (MA-04) | Keep, or turn with annotation columns marked | Always ask under an assay lens when undetermined; refuse a turn that cannot place a column |
| Outcome (target) and its unit | Too loose | The unit is guessed and printed; nothing corrects it (IN-05) | The column, plus a unit the user confirms | A unit from a name is a proposal unless an explicit suffix states it |
| Task | Too loose | Skipped as a fact for ordered text, skewed biomarkers and censored outcomes (RO-10, RO-03) | Regression, binary, multiclass, ordinal, time-to-event, count; outcome scale | Skip only for 0/1 and non-skewed continuous outcomes; ask about order, scale and follow-up |
| Event | **Right** | Binary only, no default level | — | — |
| Purpose | Right as a question; too loose in consequence | No default, but almost nothing reads it (RO-06) | Prediction / inference | Purpose sets order, rungs and which questions are asked |
| Roles | Too loose (both); much too loose (I) | Substring nutrients (IN-01), arms and batch as "design" (IN-02), rates as time (IN-10), IDs missed (IN-06); no exposure or confounder distinction (RO-04) | Add primary exposure, confounder / mediator / collider / precision, acquisition-batch, cluster, categorical | Recognizers need corroboration; under inference refuse until the exposure and adjustment set are answered |
| Grain (does a person repeat?) | Right for contradictions; too loose for suggestions and downstream | Measurements offered as IDs (IN-06); abandoned or undetermined grouping gives independent-row intervals (MA-01) | One row each / repeated by `<id>` / I don't know | Suggest identifier-like columns only; under inference cluster or block when an ID repeats |
| Clusters above the person | Too loose (absent) | Site, centre and household never asked (RO-08) | None / grouped by `<column>` | I: fixed effect plus clustered errors, or a recorded reason. P: internal–external validation |
| Repeat kind | Too loose | Stated from spacing or a bare index (IN-12) | Replicates / time points / imputed copies (I18) | Ask unless unambiguous |
| Aggregation | Too loose | One rule for every numeric column (MA-14); text times fall back to file order (MA-03); predictors after the outcome accepted (RO-09) | Per-column rules (mean, first, last, change, mode; baseline kept beside change) | Refuse change on constant columns and unreadable time orders; P: refuse predictors after the outcome; I: block and record |
| Temporal | Slightly too loose; slightly too tight for cross-sectional cohorts | Promises forecasting but only reorders the holdout; folds ignore time (MA-11, IN-24) | Random / latest held out / internal–external by period | Reword as a validation choice; time-ordered folds |
| Exclusions | Too loose and too tight | Outcome rules accepted (RO-01); unknowns kept silently (MA-19); kJ read as kcal (IN-07); no Goldberg or sensitivity (ME-16) | None, Willett 2013, NHS/HPFS, sex-neutral bands, Goldberg, custom; per-rule handling of unknowns; primary plus sensitivity | Refuse rules on the outcome; state the instrument a screen was designed for; show counts under each convention |
| Missing values | P: about right. I: too loose. Values below detection: too tight and too loose | Median fill and indicators offered under inference without concern; no MI (ME-01); no detection-aware option (ME-08) | Complete cases; single imputation conditional on energy; indicators; MI with the outcome; half-minimum and censoring-aware options | I: MI or complete cases first, single fill and indicators block-and-record. P: current in-fold, outcome-free imputation. Left-censoring finding: median refused unless overridden |
| Energy adjustment | Too loose and too tight | Labels do not match the model (ME-02, ME-03, ME-04, IN-21); ordered by custom (RO-07); log variant hidden, residual + energy and all-components missing (ME-14) | None (energy out), standard, residual + energy, residual (energy out, so labeled), log residual, density, density + energy, partition, all-components | Order by purpose and name the tension; estimand text from the fitted matrix; detect energy-related outcomes |
| Energy strata | Too loose | Within-sex residual manufactures effects (MA-02); stratifying density does nothing (D15) | Within strata with a pooled constant, or strata kept in the model | Refuse strata that are not predictors unless the pooled constant is used |
| Split | P: menu too tight, order too loose. I: too loose | No bootstrap or repeated CV (ME-11); a holdout leads under inference and coefficients use training rows (ME-12) | Bootstrap optimism, repeated k-fold, k-fold, holdout 10–30%, internal–external, time-ordered | P: order by n with one tension line. I: all rows by default; open the seal only if a holdout was chosen |
| Models | Too loose on guidance, too tight on the menu | Baseline verdict lenient (MA-10); no declared final model (ME-13); EPV rule for prediction; no ordinal, Cox, Firth, mixed model, FDR family, PLS-DA (ME-18, ME-19) | Add those families | Corrected baseline test; final model declared before opening; Riley criteria under prediction |
| Substitution | Too loose and too tight | Completeness unchecked (ME-05); band mis-sized (MA-12); support marginal (MA-13); nonsense pairs from the recognizer (IN-01); no %E or ilr | kcal swap with all sources; %E leave-one-out; ilr; band with B ≥ 1,000 | List omitted sources; I: block and record when the omitted share is large |
| Open the seal | Too loose | Re-draws after opening serve new scores; the opened score is not kept (RO-05); no final model (ME-13) | Open once, with the final model declared | Freeze the opened scores; re-draw only with a recorded re-seal; a new outcome starts a new seal |
| Repair: missing-value codes | Too loose | Critical cards and one-click blanking on false positives (IN-03, IN-04) | Treat as missing / keep | Warning unless corroborated; must-not-fire fixtures |
| Repair: impossible values | Too loose | Dietary energy and children judged by adult physiology bands (IN-09); "entry errors" wording (MI-04) | Set missing / exclude / keep | Sourced bands, age-gated, CONVENTION badge |
| Repair: kJ → kcal | Right when detected; too loose otherwise | Works with macronutrients; misses kJ without them (IN-07) | Convert / keep | Magnitude prior; a units exit when a screen would remove most rows |
| Repair: SAS zeros | **Right** | Exact detection, conservative repair | — | — |
| Repair: numbers stored as text, "<LOD" | Too tight | "No control for this yet" (MA-16) | Parse with a shown rule; a "<LOD" family | Count and name every value that will not parse |
| Pooled QC and instrument rows | Too tight (and a false reassurance) | Critical finding with no lever (RO-13) | Exclude rows by level of a text column, before the seal | Correct the served text |
| Survey design use | Too loose under inference (absent) | Weights recorded and ignored (ME-06) | Population (design-based) / this sample (unweighted, stated) | I: block and record until design-based estimation exists |
| Follow-up and censoring | Too loose (absent) | Never asked (RO-03) | Same follow-up / varying follow-up | Varying: time-to-event or fixed horizon; until then block and record |
| Exposure of interest and adjustment set | Too loose under inference (absent) | Never asked (RO-04) | Primary exposure; role per covariate | I: refuse the table until answered |
| Exposure form | Too tight (absent) | Straight line only (ME-17) | Linear / spline / quintiles | I: spline first, quintiles tagged customary |
| Categorical declaration | Too tight (absent) | Integer codes as slopes (MA-15) | Mark nominal | Propose for small-cardinality integers and NHANES codes |
| Analysis-plan lock (inference) | Too loose (absent) | p-values live while choices change (RO-12) | Seal the plan before the first coefficient | Mark later changes "after the estimates were seen" |

---

## 5 · The fix plan

Eighteen work packages in strict layer order: math (WP1–WP5), then methods (WP6–WP12), then intelligence (WP13–WP15), then routing (WP16–WP18). Within a layer, upstream comes first: a package other packages depend on precedes them. Each package lists the findings it closes and acceptance tests that would prove it fixed. "Reference result" means a number reproduced in this audit, to be matched by an independent implementation; "source check" means a sentence a reviewer can verify in the cited primary source. The fixtures named here are in `repro.tar.gz` or regenerate from it.

Strict order has one cost worth naming. Two routing guards are one validator each and stop a published wrong number: refusing eligibility rules on the outcome (RO-01) and refusing outcome repairs after the split (RO-02). They stay in WP16 by your rule; if you want an exception, they are the candidates.

### Math

#### WP1 · Data identity: values mean what the file says, and reshaping keeps rows and order honest
*Closes MA-03, MA-04, MA-05, MA-14, MA-15, MA-16, MA-17, MA-18, MA-19. Also C12, C14, B18, B21, B22, D16 (minor). First because every later number is computed on this table.*

Acceptance tests:
1. **Ambiguous dates.** 2,000 US dates coarsened to the first of the month: ingest either asks (three example rows under each reading) or keeps the column as text; it never silently changes 1,840 dates. After the user answers "month first", quarterly visits read about 91 days apart and the repeats reading matches the ISO copy of the same file.
2. **Ordering when combining.** Visits stored month_12, baseline, month_6 with text labels: the combine is refused with "declare the order of the levels", or, once declared, the LDL change is **+2.0** (not −1.0). Text dates "Mar 3, 2021": participant A's weight change is **−10** (not +5). An undated middle visit: "last" returns the latest dated visit (74.0, outcome 132) and change is −6.0; the receipt counts undated records. `ordered_by` never names a column that ordered nothing.
3. **Turning a feature table.** A metabolomics export with `metabolite, mz, rt, hmdb, S1…S12` becomes exactly 12 sample rows with m/z, RT and HMDB kept as feature annotations; a numeric Entrez-ID matrix keeps gene IDs as feature names; MZmine's `row m/z` and `row retention time` are never samples. The preview and the stage are produced by the same function (a test compares them).
4. **Per-column combining.** Under "change", a sex code constant within persons keeps its baseline value (not 0); an integer smoking code under "mean" is combined by first, last or mode (never 1.67); the receipt lists every numeric column that varied within persons and the rule applied to each.
5. **Categorical declaration.** RIDRETH3 declared categorical enters as five indicator columns; the declaration is proposed for integer columns with few levels and for known NHANES coded variables.
6. **Spellings.** A column with 3% "." parses as numeric after the repair, which counts and names every value it could not parse; a decimal-comma file with ";" separators gives the same numbers as its dot-decimal copy; "<0.2" goes to a separate "<LOD" family. Source check: SAS documentation, "By default, SAS replaces a missing numeric value with a period".
7. **Parity.** The same table saved as CSV and as xlsx gives identical missing counts per column ("None" is kept as an answer in both), asserted over the whole `NULL_TOKENS` list.
8. **Infinite values.** One `inf` in a derived ratio: the profile completes, the summary is computed over finite values, a finding counts the infinity and offers set-missing, and the Arrow and DuckDB paths agree.
9. **Flow counts.** With 20% of age missing, the step "`age` within 20–80" counts only rows with age in range, and a separate line reads "age not recorded: n"; the by-sex energy screen reports rows it could not screen. Source check: STROBE item 13(a).

#### WP2 · Intervals that match how the data were sampled
*Closes MA-01, MA-06, MA-07, MA-08. Also A18 (minor).*

Acceptance tests:
1. **Repeated rows.** 300 people × 3 rows, null effect, purpose = inference, grain "I don't know" (undetermined) and grain "one row each" with a repeating ID (abandoned): over 400 replicates the reported 95% intervals cover the truth ≥ 0.93, or the table is blocked with a recorded attestation. Reference: plain OLS 0.83–0.84, clustered 0.947–0.948.
2. **Few units.** The 6 × 40 null-sodium server scenario no longer prints p = 4 × 10⁻¹⁴ silently: below the unit floor, cluster-robust intervals are refused with exits (combine to one row per person; mixed model, once WP12 adds it). Reference: random-intercept p = 0.85.
3. **Small-G correction.** For a cluster-robust fit with G = 10, `use_t` is true and half-width/SE equals the t(9) quantile, 2.262; the caption reads "cluster-robust, G = 10, t(9)". With CR2 or a wild cluster bootstrap, Monte Carlo type-I error at G = 12 is ≤ 0.07 (reference with z: 0.10–0.20). A concern names G whenever G < 30. Source check: Cameron & Miller 2015, "at a minimum one should use the T(G − 1) distribution".
4. **Heteroskedasticity.** Null slope on lognormal intake with error SD proportional to intake, n = 200: HC3 coverage ≥ 0.92 (reference: classical 0.61, HC3 0.93); the caption names the covariance type; a residual-variance check raises a concern on that fixture and stays silent on a homoskedastic one.
5. **Separation.** 8% exposure, every exposed row an event: a concern names separation and the column, the Wald interval and p-value are suppressed for that row, and the Firth estimate is finite.
6. **Missing identifiers.** Missing IDs are mapped to one unit per row before clustering, matching the split's mapping.

#### WP3 · Energy-transform algebra
*Closes MA-02. Also B20 (minor).*

Acceptance tests:
1. On the null fixture (n = 3,000; fat has no effect; outcome depends on sex), the within-sex residual with sex not among the predictors gives p > 0.05 in ≥ 95% of 200 replicates and pooled |r(fat_adj, sex)| < 0.05. Reference today: p ≈ 10⁻¹⁰⁶ to 10⁻²⁰⁹, r = 0.56–0.58.
2. Per-level and pooled r(N_adj, E) and r(N_adj, strata) are both reported in the lineage.
3. NUTRITION_PACK line 480 is corrected to agree with line 446 ("at the cohort mean energy").
4. A stratum with fewer than a stated minimum of rows (proposed 30) falls back to the pooled slope, and the lineage says so.

#### WP4 · Scoring estimators
*Closes MA-09, MA-10, MA-11. Also A15, A16, A17, A19/E11, A20 (minor).*

Acceptance tests:
1. **Pooled R².** OLS, p = 5, population R² 0.20, 5-fold, 500 replicates: at n = 100 the reported CV R² averages within 0.02 of the target 0.13–0.14 (today 0.05); the holdout R² uses the training mean. Source check: Hawinkel et al. 2024, "this pooling estimator should be preferred".
2. **Baseline test.** On 300 null binary datasets (n = 150 and 400) the "better than baseline" rate is ≤ 0.07 (today 0.25–0.31); the gain is reported with an interval.
3. **Time-ordered folds.** With temporal = yes, fold time ranges are ordered (forward-chaining by whole unit), elastic net's inner CV receives the same splitter, the methods sentence says "time-ordered folds", and fold stratification is decided independently (no event-free folds on the 4%-event fixture).
4. **Order independence.** The same 1,000 rows in random order and sorted by the outcome choose the same elastic-net penalty (today 0.0299 against 0.0004).
5. **Planning precision.** The seal plan's stated holdout-R² precision is within 20% of the simulated SD at n = 100 and 500 for an assumed R² of 0.2 (today it overstates 2–8×).

#### WP5 · Substitution band and support
*Closes MA-12, MA-13. Also B16, B17 (minor).*

Acceptance tests:
1. Linear fat → carbohydrate fixture at N = 10,000: band half-width within 15% of the analytic 0.042–0.044 (today 0.087–0.106).
2. At N = 1,500 the band's coverage over 300 datasets is ≥ 0.93 (today 0.89–0.91 at B = 50).
3. The saved caption states rows used, B and the number of successful refits; a band needs a stated minimum share of successful refits.
4. At k = 300 kcal, rows whose shifted share of energy falls outside the observed range are excluded (today 621–1,398 such rows are counted on-support), with the count per check reported; a fixed-population curve is available.

### Methods

#### WP6 · Energy-model estimands: the label equals the model fitted
*Closes ME-02, ME-03, ME-04, ME-05, ME-14, ME-15. Also B19, D14/G17 (minor). Depends on WP3.*

Acceptance tests:
1. **"None".** The model matrix excludes the energy-role column; on the D fixture the protein coefficient is 0.0561 (the truly unadjusted model), not 0.0299; a test asserts the "none" and "standard" matrices differ whenever an energy column exists.
2. **Residual.** Residual-plus-energy reproduces the standard coefficient to 10⁻¹⁰ with sex, age and activity covariates. If the energy-dropped form is kept, its label and methods sentence say energy left the outcome model, and on the sign-flip fixture (standard +0.0199, energy-dropped −0.0067) the card shows the gap. Source check: McCullough & Byrd 2023 ("…plus a term for total energy intake").
3. **Log residual.** It carries its own estimand, distinct from the linear residual's.
4. **Omitted sources.** With fat, carbohydrate and protein plus energy, the estimand reads "in place of the energy sources not in the model: alcohol, other" and the coefficient 0.0040 is described as fat versus alcohol. Fiber beside carbohydrate-by-difference triggers a warning.
5. **Substitution completeness.** A two-nutrient model lists the omitted sources with a concern (inference: block and record above a stated omitted share); adding the remaining sources moves carbohydrate → protein from +0.940 to about +0.58 (truth 0.600).
6. **All-components.** The named option recovers an average relative effect of 0.0307 (truth 0.0305) with an interval; under inference it ranks first with the dispute and precision cost in one line. Source check: Tomova et al. 2022, "the 'all-components model'".
7. **Nested totals.** With SFA, MUFA and PUFA beside total fat, the total's row reads "remaining fat (holding SFA, MUFA, PUFA fixed)" (NHANES reference: 0.3127 with parts against 0.0734 without).

#### WP7 · Missing data by purpose
*Closes ME-01, ME-08. Also E14, B23, F14 (minor). Depends on WP6 (energy-aware imputation).*

Acceptance tests:
1. **Inference.** Confounder 35–44% missing at random given the exposure: multiple imputation (chained equations with the outcome and energy, m ≥ 20, Rubin's rules) gives |bias| < 0.01 and coverage ≥ 0.93 (reference today: median fill bias +38–75%, coverage 0.00; MI −0.002, coverage 0.92). Complete cases stay available with their assumption stated. Median fill and indicators are blocked and recorded for the inference table.
2. **Prediction.** In-fold, outcome-free imputation is unchanged and its CV results reproduce today's to 10⁻⁹.
3. **Energy-aware fill.** With 20% of protein missing, imputed rows' energy-adjusted values correlate with energy like observed rows (reference today: −1.000 against +0.78).
4. **Methods sentence.** It names the method ("median for numbers, most frequent for categories", or "multiple imputation, m = 20"). Source check: STROBE-nut nut-13.
5. **One rule.** ROADMAP §07, the M2_CONTRACT Tier A test and the MISSING drawer state the same purpose-conditional rule, citing Sisk et al. 2023 and Moons 2006 (via Harrell).
6. **Below detection.** When left-censoring fires, median fill is refused unless overridden with a reason; half-minimum and a censoring-aware option are offered; on the log-scale censoring simulation the censoring-aware estimate stays within 5% of 0.5 at 60% censoring (reference: detected-only 0.497). Zeros are recoded as non-detects only after the user says so.
7. **Row loss.** Under inference, losing more than 10% of rows to complete cases raises a concern with a comparison of kept and dropped rows (CLINICAL_SURVEY_PACK's own threshold).

#### WP8 · Inference reporting: the right scale, every row, one declared model
*Closes ME-07, ME-12, ME-13. Also A21/E12 (minor).*

Acceptance tests:
1. **Odds ratios.** A binary inference table shows exp(β) on a log axis with the event and reference named (fiber: OR 1.070 per g); multinomial rows show relative-risk ratios against a named reference.
2. **All rows.** Under inference, the coefficient table is fit on every analyzed row with n stated; on the NHANES export sugar is −0.0576 (−0.1096, −0.0057) whatever the seed.
3. **Final model.** Opening the seal requires a declared final family (chosen on CV); its holdout is the reported result and the others are marked secondary; the selection optimism is stated or estimated (null reference: +0.036 AUC).
4. **Sample-size guidance.** Under prediction the shelf uses Riley-type criteria; for OLS under inference the warning threshold is about 2 subjects per variable (Austin & Steyerberg 2015).

#### WP9 · Validation and performance reporting
*Closes ME-10, ME-11. Also E15, E16 (minor).*

Acceptance tests:
1. **Calibration.** Out-of-fold and holdout calibration intercept and slope plus a smoothed curve; on a simulated well-calibrated logistic model the slope is 1 ± 0.1; on boosted trees fit to that data, miscalibration is flagged.
2. **Intervals.** Holdout AUC with a DeLong interval matching a reference implementation to 10⁻³; pooled CV estimates with SEs.
3. **Resampling.** Repeated k-fold and bootstrap optimism correction (whole pipeline refit, B ≈ 200) are offered and, for prediction below a stated n, ranked first; the optimism-corrected AUC matches an independent implementation on a reference dataset.
4. **Banner and primary metric.** The banner says "highest AUC" or ranks on a proper score; multiclass uses log loss as primary.
5. **Claims.** The unsourced "below about 50 rows" sentence is replaced by the spread computed on the user's data. Source check: Steyerberg 2018, "random data splitting should be abolished".
6. **Clusters.** Internal–external validation by a chosen cluster column reports per-cluster performance and its spread.

#### WP10 · Survey design
*Closes ME-06. Also D20/G20 (minor). Weight choice is in WP13 (IN-19).*

Acceptance tests:
1. With design columns present and purpose = inference, the app asks "population or this sample"; until design-based estimation exists, a recorded attestation ("unweighted, sample-only estimand; standard errors ignore strata and PSUs") is required and appears in the methods sentence.
2. Design-based estimation on the informative-weight fixture reproduces the reference: DR1TFIBE +0.0065–0.0066 (about 0.000–0.013), matched against an independent implementation (for example R `survey::svyglm`).
3. Exclusions under a design become domain flags (rows kept for variance); lonely PSUs are handled and reported; pooled 1999–2002 data use the four-year weights. Source check: NHANES Analytic Guidelines 2011–2016.

#### WP11 · Omics preprocessing and inference
*Closes ME-09, ME-18. Also F17 (minor).*

Acceptance tests:
1. On the depth-confounded null counts, a linear-family fit on raw values is refused until a transformation is chosen or the values are declared normalized; with log-CPM/TMM the CV AUC is 0.50 ± 0.05 (today 0.83–1.0). A label-blind library-size check flags the depth–outcome association.
2. On the dilution null, PQN + log2 gives AUC 0.50 ± 0.05 (today raw 0.59–0.71).
3. Under inference with p ≥ n, elastic net carries "no confidence intervals"; a feature-wise regression family with BH-FDR holds the false-discovery rate ≤ 0.05 on a null simulation.
4. "Do not pre-normalize" is replaced by purpose-specific text. Source check: Hornung et al. 2015.

#### WP12 · Methods a reviewer expects
*Closes ME-16, ME-17, ME-19, and supplies the families WP2, WP16 and WP17 point to: random-intercept mixed model and GEE (MA-01), Firth (MA-08), Cox (RO-03), ordinal (RO-10), regression calibration (IN-22), %E substitution (B24/D19).*

Acceptance tests (each against an independent reference implementation):
1. Restricted cubic splines with the pack's knot percentiles match R `rms::rcs` to 10⁻⁶, with a nonlinearity test; quintile estimates with a trend test on quintile medians.
2. Proportional-odds model matches statsmodels `OrderedModel` or R `MASS::polr`.
3. Cox on the staggered-entry fixture: HR 1.012 (0.997–1.027).
4. Random-intercept mixed model on the 6 × 40 fixture: sodium p ≈ 0.85.
5. Goldberg cut-offs with Black 2000 parameters reproduce published thresholds for a stated PAL and d; a "primary plus sensitivity" answer renders the estimate under each exclusion rule. Source check: Banna et al. 2017.
6. Univariate regression calibration of energy-adjusted intakes when two or more recalls exist. Source check: Freedman et al. 2011.

### Intelligence

#### WP13 · Recognizers: whole tokens and corroboration
*Closes IN-01, IN-02, IN-05, IN-06, IN-07, IN-08, IN-10, IN-19.*

Acceptance tests:
1. **Must not be nutrients:** `fatty_fish_g`, `fat_mass_kg`, `body_fat_pct`, `fatigue_score`, `c_reactive_protein`, `total_protein`, `prothrombin_time`, `fibrinogen`, `fib4_score`, `bicarbonate`, `carbamazepine`, `carbonated_drinks_g`, `alcohol_use_disorder`, `lipid_lowering_meds`. **Must be recognized:** `sfa_g`, `mufa_g`, `pufa_g`, `DR1TSFAT`, `DR1TMFAT`, `DR1TPFAT`, `DR1TSUGR`, `DR2TKCAL`, `DRXTKCAL`, `ENERC_KCAL`, `PROCNT`, `CHOAVL`; on the 45-nutrient DR1T* table no nutrient is proposed "covariate".
2. **Must stay predictors:** on an RCT table, `treatment`, `arm`, `diet_group`, `condition`, `phenotype`, `fasting`; `steps_per_day`, `drinks_per_week`, `baseline_glucose`; `birth_weight` in grams. Batch is proposed as "acquisition/batch", a covariate under inference.
3. **Identifiers:** one recognizer agrees with itself on the 46-name table; `eid`, `patid`, `ptid`, `HHID`, `IDNO`, `USUBJID` are identifiers; `site_id`, `household_id` are clusters; no float or measurement column is ever suggested as a unit ID.
4. **Units:** dietary choline (mg/day) is not "mg/dL"; any unit not read from an explicit suffix requires a recorded decision before a sentence carries it. Source check: CLINICAL_SURVEY_PACK, "TurboTab will not guess".
5. **Energy units:** a kJ "energy" column with only sodium beside it is read as kJ by the magnitude prior; the implausible-intake count after conversion matches the kcal screens.
6. **Weights:** with WTSAF2YR and fasting analytes present, the least-common-denominator rule names the fasting weight. Source check: NHANES weighting tutorial.

#### WP14 · Detectors: pass must-not-fire fixtures and the null
*Closes IN-03, IN-04, IN-09, IN-11, IN-12, IN-13, IN-14, IN-15, IN-16, IN-17, IN-18, MI-01, MI-04. Also H18, H19, F16 (minor).*

Acceptance tests:
1. **Survey scales:** balanced 6-point, skewed 7-point, 0–5 and 9-point blocks raise no code finding; a 1–5 block with a real 9 still does; floor-heavy PHQ-9, GAD-7 and a 0–10 rating scale are recognized as instruments.
2. **Clinical codes:** across eight clean generators at n = 100, 300 and 1,000 (200 replicates each), the code-flag rate is ≤ 1% (today up to 15.5%).
3. **Drift and redundancy:** on drift-free data at n = 16–40 the drift finding fires ≤ 5% (today 40/40 at n = 16); with one sample ×20 the redundancy count stays near 300.
4. **Plausibility:** total energy is not in the physiology bands; bands carry source, cycle and CONVENTION; children are judged by age-specific z-scores; DBP 0 in NHANES is described per the BPX documentation.
5. **Orientation:** log2 GEO-style and MZmine-style feature-major tables are asked about under an assay lens.
6. **Repeats:** a daily falling time course and a crossover are asked (not stated); quarterly recalls under the dietary lens are read as replicates or asked.
7. **Lens hints:** TPM and log2-TPM with Ensembl IDs are hinted genomics; survey blocks survey; the contradiction check runs.
8. **Genomics card:** log2-TPM and voom log-CPM are read; thresholds are validated on public matrices; single-cell raises a concern naming pseudoreplication.
9. **Histograms:** discrete data get resolution-aligned bins, and the two implementations agree on a shared test.

#### WP15 · Claims: every sentence matches its source and the computation
*Closes IN-20, IN-21, IN-22, IN-23, IN-24, IN-25, IN-26, MI-02, MI-03. Also G14, G15, G16, G18, G19, F15, D15 (minor).*

Acceptance tests:
1. Every claims-ledger row marked WRONG, OVERSTATED or SELF-CONTRADICTED is re-checked with the corrected sentence quoted beside its primary source.
2. A badge-consistency test: identical claim texts with the same source share one badge.
3. "Not in dispute" is gone; an energy-related outcome pushes the DISPUTED note on the energy card.
4. Density plus energy carries Tomova's caveat; "unbiased in direction" is qualified; TNTC is right-censored (source check: FDA BAM ch. 3).
5. The temporal methods sentence describes what was drawn and the share of held-out rows predating the training boundary (reference: 56–71% on the visit fixtures).
6. The lineage marks pass-throughs "kept", attributes operations only to the columns they touched, and says "geometric mean" under log.
7. The survey pack's attenuation text passes a replay with λ = 0.70: slope × λ = 0.210, standardized × √λ = 0.251.
8. The exclusion presets are attributed as Willett 2013 (800–4,000 for men) and NHS/HPFS (800–4,200); the coach states r² rather than "mostly".
9. The pack citations corrected: Nygaard 2016 (GSE40566, 2,011 versus 11 genes, ComBat versus limma blocking), Zindler 2020's methylation-array setting, Eekhout 2014's threshold.

### Routing

#### WP16 · The seal holds at its edges
*Closes RO-01, RO-02, RO-05, RO-12. Also I16 (minor).*

Acceptance tests:
1. **Membership by identity.** After the split, setting three impossible outcome values to missing moves **0** held-out rows (today 119–130); `set_target` and `set_task` after the split are refused with the re-seal exit or recorded as "re-sealed" and stated; the Router orders the impossibility and code repairs before the split.
2. **Outcome eligibility.** `bmi 18.5–30` with `bmi` as the outcome is refused (409) with exits, under both purposes; the exclusions preview no longer draws the outcome's histogram.
3. **After opening.** The at-opening scores are stored and reported; a new seed after opening requires a recorded re-seal; a new outcome starts its own seal with scores withheld.
4. **Inference plan lock.** Changes to exposure, adjustment set, exclusions or missing-data plan after the first coefficient is shown are recorded "after the estimates were seen" and appear in the methods sentence. Source check: Gelman & Loken 2013.

#### WP17 · The declared purpose routes the questions
*Closes RO-03, RO-04, RO-06, RO-07, RO-08. Also I17 (minor). Uses WP6, WP7, WP10 and WP12.*

Acceptance tests:
1. Every option of the energy, missing-values, split and exclusions questions carries "customary in <field>" with a source and "sound for <purpose>" with a reason; the order under inference differs from the order under prediction for each, and a one-line tension appears where custom and soundness disagree.
2. **Censoring:** on the staggered-entry fixture a follow-up question fires; until the Cox family exists the result is blocked and recorded; the p ≈ 10⁻¹³ log-odds is never served silently.
3. **Inference questions:** on the NHANES export under inference, no coefficient is shown until a primary exposure and a role for each covariate are answered; mediators in a total-effect set are blocked and recorded; the caption is worded from the estimand. Source check: VanderWeele 2019.
4. **Clusters:** on the 10-site fixture a cluster question fires; under inference site fixed effects with clustered errors reproduce −0.04 (−0.11, 0.03).
5. **Energy order:** under prediction the energy-keeping methods are ranked first and the residual's discard of energy is stated.

#### WP18 · Structural questions ask where evidence is thin
*Closes RO-09, RO-10, RO-11, RO-13. Also I18 (minor).*

Acceptance tests:
1. Time points with predictors = last and outcome = first: refused under prediction, blocked and recorded under inference.
2. `none/mild/moderate/severe` triggers "are these levels ordered?"; hs-CRP triggers an outcome-scale question; the 20-class rule agrees between the skip and the explicit answer.
3. A "something else / not sure" lens is accepted, runs the generic checks and is stated in the methods.
4. A Case/Control/QC export offers a QC exclusion before the seal and the task becomes binary; the served QC text is corrected.
5. NHANES DXA multiple-imputation copies have a route (Rubin's rules) or are blocked and recorded.

---

## 6 · Coverage: what was checked and found sound

Coverage matters as much as defects: each item here was tested by running the app's code or by reading a primary source, and held.

### 6.1 Math
- **Scoring with the declared event.** The app's AUC equals scikit-learn's with the declared event (0.788498 on a case/control fixture); AUC, Brier and log loss are invariant to flipping the event; coefficient signs and the substitution curve follow the coded event; the event is validated as a level of the outcome.
- **Fold hygiene.** Every pipeline step (imputation, energy adjustment, levels, one-hot, scaling) is cloned and refit inside each outer fold, and the holdout is scored by the pipeline refit on training rows only (`modeling.py:430-442`). The repository's comparisons with scikit-learn `cross_validate` pass.
- **Baseline.** Computed on the same folds with the same scorer; the binary baseline AUC is exactly 0.5 per fold; the comparison is paired by fold. (Its verdict threshold is the problem, MA-10.)
- **Splits.** Grouped split: 293 units, none on both sides, none spread over folds, deterministic. Stratified folds: event share 0.195–0.203 against 0.198. Chronological holdout keeps units whole; ties broken by a seeded draw and disclosed; more than 10% unreadable times is refused.
- **Elastic net.** Inner CV is grouped by unit inside every outer fold, in the final refit and in band refits; back-transformed coefficients reproduce predictions to 1.8 × 10⁻¹⁵.
- **Multinomial.** Coefficient rows align with parameters and intervals; cluster covariance works for MNLogit.
- **Seeds.** Every splitter, HistGradientBoosting, LogisticRegressionCV and the band's generator are seeded.
- **Seal precision and floor.** The Hanley–McNeil standard error matches the 1982 formula; ±0.056 at 100 events and 233 non-events with AUC 0.8 agrees with Steyerberg 2018's simulated 0.75–0.85. Vergouwe 2005: "a minimum of 100 events and 100 nonevents"; Collins 2016: "a minimum of 100 events and ideally 200 (or more)" (the floor text could add "ideally 200").
- **Energy algebra.** N_adj = N − b(E − Ē) equals the residual plus N̂(Ē) to 10⁻¹⁶; the log residual refuses zeros and negatives rather than adding a constant; partition refuses a total beside its parts and unmarked units unless the Atwater reconstruction confirms grams.
- **Constants.** Atwater 4/4/9/7 and fiber 2 kcal/g and the 5% slack match FAO Food and Nutrition Paper 77, ch. 3; 1 kcal = 4.184 kJ is applied in the right direction everywhere (the constant `KCAL_PER_KJ` actually holds kJ per kcal, but every use is correct).
- **Repairs.** SQL and pandas agree at the boundaries of the impossible-values rule; the SAS-zero band equals 16⁻⁶⁵ = 5.3976 × 10⁻⁷⁹ and fires on the NHANES export; the binary repair trims like Python.
- **Aggregation basics.** Averages ignore NULLs, change is NULL for single-record units, ties fall to file order deterministically, and every source row maps to exactly one unit.
- **Nested substitution arithmetic.** Sugar → starch leaves carbohydrate unchanged; total fat → protein moves the parts in proportion; SFA → PUFA leaves total fat unchanged.
- **Data identity.** Row identity equals file order across a 1.5-million-row parallel read; the leading-zero guard holds past the sniff sample (row 1,499,995); ZIP and FIPS codes stay text; Arrow and DuckDB materialization return identical frames for unsorted duplicate IDs across row groups; samples are seeded, proportional across row groups and never include held-out rows.
- **Summaries.** Quartiles agree between the two engines (type 7); means and SDs match exact rational arithmetic to about 10⁻¹⁵ at realistic magnitudes; Parquet NaN becomes NULL and is counted missing; integer histograms with a small range get unit-width bins; caption correlations are exact pairwise values.
- **Excel.** Text identifiers such as "001" stay text.
- **Flow.** "Outcome measured" excludes ±inf as well as NaN; complete cases are judged only on predictors that can be missing.
- **Smaller checks.** Fold-mean versus pooled macro-F1 differ little (0.575 against 0.584); bootstrap duplicates on both sides of elastic net's inner folds do not shift the chosen penalty; the fit-count arithmetic in `cost.py` is right; Belsley's scaled condition number is implemented correctly.

### 6.2 Methods
- **Nested tuning** is genuinely nested (penalty chosen inside each outer training fold).
- **Lockbox mechanics** for prediction: held-out scores withheld until opening, opened once and only on a fresh fit, structural changes refused after sealing, post-seal decisions marked.
- **No resampling for imbalance**, which is correct: van den Goorbergh et al. 2022, "random undersampling, random oversampling, or SMOTE yielded poorly calibrated models".
- **Under prediction**, coefficients are drawn without intervals and marked "not interpreted as effects"; the forest plot shows exposures and hides covariates; the comparison is ordered by shelf rank, not by score, and every family's holdout is revealed together.
- **Outcome-free imputation for prediction** is defensible (Sisk et al. 2023), and **complete cases** stay on the menu (unbiased in the simulations where missingness did not depend on the outcome).
- **Tomova-sourced caveats** for the standard, residual (as Tomova defines it), partition and density models are faithful.
- **Substitution.** The curve is estimable even under exact energy closure and close to the truth when every source is in the model (0.578 against 0.600); nested parts move coherently; the band refits the whole pipeline on grouped resamples; the 50% support floor is honestly labeled a practitioner convention.
- **Exclusions.** Screens are offered, never pre-selected, and show their counts (NHANES: 501 to 2,094 rows); the excluded-versus-kept BMI note delivers STROBE-nut's "characteristics of those excluded"; kJ bounds are converted when kJ is detected.
- **Repeated measures.** The mean is recommended for replicates and no default is given for time points; "k replicates cut within-person variance k-fold" is right; usual-intake modeling is named as not built.
- **NHANES teaching** matches the CDC tutorials (WTDRD1 versus WTDR2D, SRS variance too low, never delete records before design-based analysis, dividing two-year weights from 2001 on).
- **Omics basics.** Scaling and imputation are fitted in-fold; autoscaling is supported by van den Berg 2006; the PLS-DA rule of thumb (random labels separate once p/n ≥ 2) reproduces; the VIP > 1 heuristic is correctly described as a ranking; at p ≫ n the linear family is ranked poor and refuses a singular table rather than printing degenerate p-values.
- **Schofield equations** in NUTRITION_PACK §02 match FAO/WHO/UNU Table 5.2 (the ≥ 60 bands are omitted).

### 6.3 Intelligence
- **Detectors that work.** SAS zeros; kJ detection by Atwater ratio when macronutrients are present (NHANES spread 0.13 against the 0.25 limit; no false "mixed units" up to 8% noise); NHANES 1/2 items with 7/9 codes (8 of 8); 99 in a 0–10 rating; 9 in a 1–5 Likert block; 999 in clinic visits; a 0–4 stress scale with a rare 0 is not misflagged; reverse coding is never inferred or applied.
- **Nesting** requires a named subtype that is ≤ its parent on 99% of rows; fiber is deliberately not nested; compositions must sum to 100 within 0.5% on 99% of rows.
- **Unit suspects** (glucose and triglycerides in mmol/L, HbA1c in IFCC units as a whole column) are set aside rather than called impossible; the mixed-units factor table matches standard conversions and correctly excludes temperature.
- **Orientation** of a raw-intensity feature-by-sample matrix is read correctly (ratio 77.8) and asked.
- **Survey design findings.** The diet-only weight advice is right ("use the dietary day one sample weight (wtdrd1)"); the single-PSU stratum is found.
- **Repeats** for 24-hour recalls 3–14 days apart read as replicates; spaced clinic visits (about 90 days) read as time points.
- **Genomics card.** Raw counts read as raw counts; a TPM matrix summing to 10⁶ reads "CPM or TPM, cannot say which".
- **Identifiers.** The text rule (unique, no blanks) and SEQN recognition are right.
- **Metabolomics.** Left-censoring (ρ = −0.97) and pooled QC by relative standard deviation are detected; the Excel gene-ID detector reports and never repairs.
- **Teaching claims verified** (full list in `claims-ledger.md`): Tomova's equivalence, bias and partition statements; SMOTE; MAQC-II (badged DISPUTED); feature selection outside the fold (Ambroise & McLachlan 2002); missing indicators biased for observational inference (Groenwold 2012); NHANES weights, variance and recall spacing; the reference interval as the central 95%; the 100-event floor; "a single split is the weakest option"; the substitution-review numbers (Louie & Bhowmik 2026) and the Scholbeck citation.
- **Honest gap statements** are present and accurate: no ordinal model, no usual-intake modeling, no Goldberg, no survey weighting, polychoric not computed.
- **Methods sentences** for seal, grain, unit, aggregation, temporal and orientation report what was drawn and state exploratory status; no placeholder leaks were seen.
- **Thresholds the app made up are labeled as its own** in the payload (default-value mass, mixed-units). Threshold provenance, from area H: sourced (Atwater, 4.184, unit factors, NHANES weight names, the NHS band); measured but only on the repository's own synthetic fixtures (orientation 4.0, schedule CV 0.35, modal share 0.60, genomics CVs, Atwater drift 0.25); arbitrary or unsourced (the 500–5,000 detector band, every physiology band, the 0.4 row spread, 14 days, 25% sentinel share, 8-item minimum, 30-column "wide", and others listed in `findings.json`).

### 6.4 Routing
- Answers in order; changes to answered questions allowed; stated skips can be reopened; structural (Decision A) answers outrank and are refused after the seal with a re-seal exit.
- Orientation fires only under an assay lens with a feature-major reading; the outcome waits for the oriented table; orientation is refused once an outcome exists.
- The event question is binary-only with no default; the task skip is right for 0/1 and continuous floats; low-cardinality integers are asked, with the ordinal caveat.
- Grain contradictions ("one row each" while an ID repeats) are refused with regroup and attestation exits; "I don't know" gives an exploratory basis, and skipping never does.
- Person-level grain is stated correctly when the identifier is unique.
- Exclusions and missing-value answers never move a row across the seal (identical sealed set when 50 analyzed rows leave); the exclude-rows repair keeps the seal.
- The preview pool excludes sealed rows once a split exists.
- The purpose question has no default; under inference the shelf and fit respond (elastic net's caveat when p < n; cluster-robust intervals when grouped).
- Below the 100-row or 100-event floor, cross-validation leads and nothing is removed from the menu.

### 6.5 Test suites and sources
- Repository suites run during the audit: `test_datastore.py` + `test_working.py` (70 passed); `test_modeling.py` + `test_seal.py` (91 passed); the pack guard `test_a_pack_does_not_fire_on_the_wrong_data.py` (110 passed, 17 skipped; every fixture it covers is generated by the repository, so it cannot reach the adversarial cases above). `git status` stayed clean.
- Primary sources read and quoted (by at least one auditor or skeptic): Tomova et al. 2022 (AJCN, full text) and Tomova, Gilthorpe & Tennant 2022 (substitution, PMC page); McCullough & Byrd 2023; Chiu & Wen 2026; Banna et al. 2017; de Koning et al. 2011 and Rhee et al. 2015 (cohort cut-offs); Freedman et al. 2011; Keogh et al. 2020 (STRATOS); NCI Dietary Assessment Primer; FAO FNP 77 ch. 3; FAO/WHO/UNU Schofield table; NHANES Analytic Guidelines 2011–2016, weighting and variance tutorials, BPX_I and SMQ_J documentation; Cameron & Miller 2015; Hawinkel et al. 2024; Staerk et al. 2024; Carpenter & Bithell 2000; Varma & Simon 2006; Bengio & Grandvalet 2004 (abstract); Vergouwe 2005 and Collins 2016 (abstracts); van Smeden 2019 (abstract); Austin & Steyerberg 2015 (abstract); Steyerberg 2001 (abstract) and 2018 (accepted manuscript); TRIPOD+AI 2024; Collins et al. 2024 (BMJ); PROBAST explanation and elaboration (Moons 2019); Van Calster 2019; van den Goorbergh 2022; Groenwold 2012; Sisk et al. 2023; Sperrin 2020 (abstract); Josse et al. (arXiv); Shmueli 2010; Gelman & Loken 2013; Westreich & Greenland 2013 (abstract); VanderWeele 2019; Dwork et al. 2015 (abstract); Harrell, RMS ch. 3; Wei et al. 2018; van den Berg et al. 2006; Nygaard et al. 2016; Zindler et al. 2020 (abstract); Hornung et al. 2015; Eekhout et al. 2014 (abstract); Scholbeck et al. (arXiv v1); FDA BAM ch. 3; Ozarda 2016; Ambroise & McLachlan 2002; STROBE and STROBE-nut; Chalmers 2018 (abstract); Yamamoto et al. 2023; Louie & Bhowmik 2026; Roberts et al. 2017 (abstract); SAS Base documentation; DuckDB `read_csv` and seaborn `histplot` documentation.

### 6.6 Not covered
No area audited the export and replay of the provenance record end to end, the frontend beyond the captions and figures named above, behavior on very large files beyond ingest identity, or security. The presentation layer was not graded as its own layer.

---

## 7 · Open questions

### 7.1 Verdict tally
No finding was refuted, and none was left unconfirmed. Nine were confirmed with a different severity:

| Raw | Auditor | Skeptic | Reason |
|---|---|---|---|
| B7 | major | **critical** | With energy-correlated covariates the residual's point estimate is biased, not just imprecise, under an equivalence claim |
| B11 | major | **critical** | Wrong-signed change scores under a receipt claiming time order |
| C8 | major | **critical** | Same class as C2: a wrong "last" or change under a chronological label |
| G13 | major | **critical** | Invalid published inference on the flagship NHANES lens with no pushed warning |
| C9 | major | minor | Histogram counts feed only pictures |
| D8 | major | minor | 800–4,200 is a Willett-group cohort rule; numbers are printed; a custom rule can enter 4,000 |
| G4 | major | minor | Same as D8 |
| G12 | major | minor | The true r is printed beside "mostly" |
| H8 | major | minor | "Physiologically impossible" is defensible; only "entry error" is wrong |

### 7.2 Sub-claims the skeptics corrected
These parts of confirmed findings are **not** established as the auditor first wrote them. The report above uses the corrected versions.
- **A1/E1.** The seal's basis still names the repeating column; only the column it hands to the fit is empty.
- **A3.** The 8-unit floor lives in `utils/test_lockbox.py:70`, not in `seal.py`.
- **A5.** On the skeptic's single dataset the band was 1.96× too wide (√5 = 2.24 expected); coverage confirmed.
- **A6.** The "42% of null datasets have at least one 'better' family" figure was not re-run.
- **A7.** With three families the selection optimism is modest; Varma & Simon's setting was tuning over many values.
- **A8.** In the skeptic's toy, random folds were more optimistic but ranked the families the same; random internal plus temporal external validation is the customary TRIPOD paradigm.
- **A9.** The share of held-out rows predating training is 56% to 71% depending on how spread the visits are.
- **A12/E6.** The training-rows fact is disclosed in a pipeline step detail (`linear.py:125`), not in the coefficient caption.
- **A13.** Coverage under median fill was 0.89 in the skeptic's run against the auditor's 0.85; same direction.
- **B4.** The reported SE (0.0043) is not smaller than the empirical SD (0.0030); the defect is bias with an interval that misses the truth, not an understated SE.
- **B5.** The unit note is at `modeling.py:622-625`; the `set_substitution` validator refuses non-exposure columns, so body-composition donors appear in the menu but cannot be chosen; more false positives exist than first reported.
- **B8/D3.** "All-components is missing" overstates: a partition over every component is the all-components model, unnamed. The UI tags the residual "usual", not "recommended".
- **B9.** "Every other method carries the caveat" is slightly loose: "none" carries none either.
- **B10.** Only arXiv v1 (2022) of Scholbeck et al. exists; "2024" is the journal version (DMKD 38:2997–3042).
- **C1.** In the m/z/RT case the preview does show `mz` and `rt` as sample IDs, so an attentive user could catch it; the orientation reading was undetermined, so the user declared the turn.
- **C3.** Only columns where every value is ambiguous are affected; the downstream effect on the repeats reading is stronger than first reported.
- **C5.** A leading space parses as a number; a trailing space does not.
- **D2.** The auditor's "mathematically identical" is Tomova's abstract wording; the skeptic's "algebraically identical" is from the discussion. Both appear, so there is no misquote.
- **D5.** The mechanism here is confounding through the heterogeneous "other" composite rather than the "more than one substituting component" case the Tomova quote describes; the bias reproduces either way.
- **D10.** The kJ miss is specific to tables without macronutrients.
- **D12.** Freedman's calibration sentence was paraphrased by the auditor; the substance matches.
- **E2.** Groenwold 2012 says the indicator method is valid for baseline covariates in randomized trials.
- **E8.** TRIPOD+AI states it "does not prescribe how to develop or evaluate" a model; it supports the reporting gap more than a method choice.
- **E9.** "Calibration is never computed" is slightly strong: Brier and log loss partly reflect calibration.
- **E10.** Cameron & Miller allow unweighted estimation when stratification is only on exogenous regressors and the model is correctly specified.
- **F1.** "Silently" overstates: a warning card exists, with no lever. The raw-scale bias direction depends on the data-generating process.
- **F11.** Wei 2018's criteria include one downstream metric (t-test p-value correlation); the code already prefers half-minimum, so QRILC is only the pack's planned default.
- **G2.** `packs.py:2938-2945` does not literally say "same substitution"; `content.py:823` does.
- **G10.** The "50 rows" threshold is too low by 2–4×, not by an order of magnitude.
- **H17.** The 9-point hedonic block was not silent: it was misread as 1–7 with 8 and 9 flagged.
- **I4.** The `survey_weights` warning does fire and routes to roles; answering roles marks it answered, though nothing is weighted.
- **I8.** The roles question does separate "exposure" from "covariate", and the forest plot shows only exposures; the defect is that unrecognized columns default to exposure and inference never asks for a primary exposure or adjustment set.
- **I11.** Post-seal changes are marked (an amber band, `post_seal` flags), more than "only a sentence prefix"; the at-opening scores are still not stored.
- **I14.** "Cross-validation only" is on the menu, ranked last; the PROBAST item is 4.8, not 4.7.

### 7.3 Questions that need your ruling or more evidence
1. **The residual method's default form.** This report recommends keeping energy in the outcome model (Willett–Stampfer, McCullough & Byrd). The pack calls the energy-dropped form the field default. Which does the app offer as "residual"?
2. **All-components first under inference?** Tomova 2022 and Chiu & Wen 2026 support it; the Willett/Stampfer/Tobias 2022 reply disputes it and nobody could read it.
3. **Change scores or ANCOVA** for observational repeated measures. ANCOVA is the usual recommendation for randomized designs; for observational data it is contested (Lord's paradox). Neither side was checked against a primary source.
4. **Inference and the lockbox.** BLUEPRINT's "anything fit on data is fit on training rows only" and "the lockbox is sealed before exploration" are prediction rules. ME-12 and RO-12 propose that inference fit on all rows and seal the analysis plan instead. This changes a do-not-re-litigate principle, so it is yours to rule on.
5. **The outcome in imputation.** ROADMAP §07 and the M2 Tier A test forbid it in any configuration; the drawer requires it. ME-01 proposes one purpose-conditional rule.
6. **Severity borderlines** the skeptics flagged without changing the grade: MA-06 (close to critical at 8–12 clusters), MA-19 (borders on critical), ME-15 (a case for critical under inference), IN-19 (critical if the weight is ever pre-selected into an estimate; no estimate is weighted today, so this was not traced).
7. **Interim for censoring.** Whether a fixed-horizon binary outcome with early-censored people excluded is an acceptable interim before a Cox family exists.
8. **Pooled QC workaround.** A numeric exclusion rule keyed by the text class column might drop QC rows through the API; untested.
9. **The ledger's "G22".** `claims-ledger.md` row 88 cites a finding "G22" (the heuristic that a yes/no column blank on ≥ 50% of rows means "not asked", `proposals.py:445-458`). No finding with that ID was filed; it stands as an unsourced minor heuristic, not re-tested.

### 7.4 Sources nobody could read
No claim in this report rests on these alone, but each would settle something: Willett, *Nutritional Epidemiology* 3rd ed. (2013) (read only as quoted by Banna 2017); Willett, Howe & Kushi 1997; Willett & Stampfer 1986; Willett, Stampfer & Tobias 2022 (AJCN letter); Long & Ervin 2000 (secondary sources only); Nadeau & Bengio 2003 (used only as a simulation comparator); Moons et al. 2006 (via Harrell); Hernán, Hernández-Díaz & Robins 2004; Green & Symons 1983; Bland & Altman 1996; White & Carlin 2010; van Smeden 2020; Liddell & Kruschke 2018; Dieterle 2006 (verbatim abstract); Hutcheon 2010; Lance 2006; Smith 2018 ("Step away from stepwise"); Black 2000 (full text; abstract values seen only in a search excerpt); Roberts et al. 2017 (full text; abstract only); Collins, Ogundimu & Altman 2016 (full text; abstract only); pandas issue #30051 (search summary only). Two sources were read by one side only: the CDC growth-chart BIV page (area H) and the CDC variance tutorial (areas D and G, not the area I skeptic).

### 7.5 Minor findings no skeptic re-tested
These 48 were rated minor by their auditors and were not independently re-tested, so they are **not established**. Several duplicate or extend confirmed findings (noted in brackets).

| ID | Finding |
|---|---|
| A15 | Elastic net's ungrouped inner CV is unshuffled, so the penalty depends on file row order (0.0299 random order against 0.0004 sorted by the outcome). |
| A16 | Boosted trees' early-stopping split (above 10,000 rows) is by row, not by unit. [also E13] |
| A17 | A chronological holdout also drops fold stratification; 34% of rare-event splits get an event-free fold and the fit can crash. [MA-11] |
| A18 | Missing identifiers become one giant cluster in the interval computation, unlike the split. [WP2] |
| A19 | The seal plan's holdout-R² precision overstates the SE about 2× (E11: 2–8×). [also E11] |
| A20 | The fit-time estimate scales linearly in columns; OLS cost grows about as p², so wide tables are underestimated. |
| A21 | The EPV ≥ 10 rule ranks families under prediction too, where van Smeden 2019 says it is not an appropriate criterion. [also E12] |
| B16 | The band is drawn from however many refits succeeded (2 of 50 in a test) but labeled with the number attempted. [MA-12] |
| B17 | The rows each substitution point averages over change with k, mixing effect with selection. [MA-13] |
| B18 | The sentinel SQL crashes on text columns that the preview counts, and codes are applied after kJ conversion (a 99999 code survives as 23,900 kcal). |
| B19 | Partitioning fiber beside carbohydrate-by-difference double-counts fiber energy. [ME-04] |
| B20 | A stratum gets its own residual regression from as few as 3 rows. [WP3] |
| B21 | A covariate mistaken for the replicate index is exempt from combining and from the "varying" receipt. |
| B22 | "Mean" is allowed for a 0/1 outcome; "first" means the first non-null value for the outcome but the first record for predictors. |
| B23 | Which columns get a "Missing" level is decided from every row, held-out rows included; blanks unseen in training become the reference level. |
| B24 | Exposures in % of energy cannot be substituted ("5% of energy from X replaced by Y"). [also D19] |
| C12 | Group keys merge "007", "07" and "7", so the reported number of groups is wrong. |
| C13 | The relationship-change preview screens correlations with mean-filled values, which can pick a spurious pair. |
| C14 | Integers above 2⁵³ lose precision silently; the two summary engines differ in last digits; large integers reach the browser as rounded JSON numbers. |
| D14 | The log-scale residual cannot be chosen in the UI. [also G17; ME-03] |
| D15 | Strata are offered for density methods and the methods sentence claims a stratification that does nothing. [IN-25] |
| D16 | The by-sex energy screen keeps rows with missing or unrecognized sex, including 9,000 kcal days. [MA-19] |
| D17 | The multivariable density model has no caveat. [confirmed via B9/G6 as IN-21] |
| D19 | %E substitution and compositional models are not offered. [also B24] |
| D20 | The pooled-cycle weight rule omits the 1999–2002 four-year-weight exception. [also G20] |
| D21 | Recalls are averaged regardless of how many each person has, with no note that error variance differs. |
| E11 | Held-out R² precision formula overstates imprecision 2–8×. [also A19] |
| E12 | "n < 10·p: unstable" is unsupported for OLS (about 2 per variable suffices, Austin & Steyerberg 2015); no Firth option. [also A21; MA-08] |
| E13 | Time and grouping are not carried into tuning folds; boosted-tree early stopping splits rows at random. [MA-11, A16] |
| E14 | Complete cases under inference: an 86% row loss on NHANES (2,996 of 21,849 kept) and outcome-dependent missingness go unflagged; "blanks as a level" is recommended regardless of purpose. [ME-01] |
| E15 | Neither SMOTE nor class weighting is shown with its cost, so the blueprint's own example tension never appears. |
| E16 | No internal–external (leave-one-cluster-out) validation across cycles or sites. [ME-11, RO-08] |
| F14 | Genomics low counts get a "Treat as missing" repair whose own summary says they are counts. |
| F15 | Pack citations wrong in detail: Nygaard 2016's dataset (GSE40566), number (2,011 against 11) and method; Zindler 2020's assay (methylation arrays); Eekhout 2014's threshold (> 25%). |
| F16 | Genomics thresholds are fitted to synthetic sibling fixtures; shallow raw counts are not recognized. [IN-14] |
| F17 | Departures from metabolomics convention (autoscaling rather than Pareto; no log) are never named. |
| G14 | "The single most-checked figure in a nutrition methods review" is an unsourced SETTLED superlative. |
| G15 | The "14 of 24 pairs" evidence concerns Goldberg cut-offs, not the fixed-kcal screens it sits beside. |
| G16 | Option text contradicts the code: multiclass "scored by accuracy" (primary is macro-F1); boosted trees "handle missing values" (the median imputer runs first). |
| G17 | The "field default" log residual cannot be produced in the UI, and the log-case methods sentence misstates what is added back. [also D14] |
| G18 | The same claim carries SETTLED in one card and CONVENTION in two others. |
| G19 | "Polychoric is the appropriate choice for ordinal items" is badged SETTLED; it is contested, and an item–rest correlation calls for polyserial. |
| G20 | The pooled-cycle rule omits the 1999–2002 exception, and the 100-study substitution review is cited without author or year. [also D20] |
| H18 | The Atwater "mixed units" verdict tests the ratio's spread, not drift with energy, and fires at critical severity on 10% noise in 26 of 50 tables. |
| H19 | One coding fact produces 11 critical cards; a stratum code 999 is flagged as a missing code while another finding treats it as a real stratum. |
| I16 | Withholding the outcome distribution at the eligibility question is cosmetic: the outcome question showed it two steps earlier. [RO-01] |
| I17 | The temporal question's gate reason is false for one-row-per-person cohorts enrolled over time, and its wording promises forecasting. |
| I18 | Multiply imputed copies (NHANES DXA implicates) have no route; treating them as replicates overstates n fivefold. |
