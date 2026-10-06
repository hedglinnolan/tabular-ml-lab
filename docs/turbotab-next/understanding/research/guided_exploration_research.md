## EDA view map by lens: what each field's researchers need to see first

This is a research brief, read-only. No servers or test suites were run. Everything below rests on the repo's research packs, the calm FOUNDATION, MODELING_SEQUENCE, the engine code, and a small set of primary sources I checked (listed at the end). Where I say "engine today", I name the finding id or file. "Not found" means a grep of `turbotab/` and `turbotab/core/` turned up no reading of that kind.

---

### 0. The headline, in eight points

1. **Group views by the question they answer, four to six per lens.** Every lens's first three or four groups are about whether the data can be trusted, and they never involve the outcome. The "worth a look" ranking belongs there.
2. **Use one rule for the outcome, taken from the STRATOS initial-data-analysis literature.** Heinze et al. 2024 (*BMC Med Res Methodol* 24:178) say IDA "should – without good reason – not anticipate analysis directly related to the research question, implying that associations between outcome and predictors are not explored, neither numerically nor graphically." The outcome's own univariate distribution and missingness are allowed; pairing it with predictors is not. Their checklist domains are:
   - missing values (M1–M4);
   - univariate (U1–U2);
   - multivariate among predictors and structural variables (V1–V3).

   These domains map almost one-to-one onto the groups below.
3. **Free visual browsing produces false findings at a measured rate.** Zgraggen et al. (CHI 2018) found that over 60% of the insights participants drew from free visual exploration were false. So the app should point only at noticings ranked by a pre-declared measure with a stated chance or convention reference. It should never point at "interesting-looking" outcome relationships. Scagnostics (Wilkinson, Anand & Grossman 2005) is the classic precedent for ranking views by measures. Applied to outcome × predictor pairs, it would automate the garden of forking paths.
4. **One scaling rule covers 29 columns to 20,000 features:**
   - views about columns become one distribution over features, plus a ranked strip of the extreme ones (FOUNDATION's Strip: top 12, then "and N more");
   - views about samples stay usable at any p;
   - feature × feature matrices become sample × sample views, or cluster summaries.
5. **The Explore stage leads with outcome views, in column order.** `turbotab/core/stages/explore.py` (lines ~404–418) puts up to 12 outcome-relationship curves first, taking the first 12 numeric predictors as they appear. They are not ranked by any measure. This happens under inference too.
   - Under inference, that conflicts with STRATOS.
   - Under both purposes, it conflicts with BLUEPRINT §11 rule 2 ("chosen by measured change").
   - At genomics width it means "the first 12 genes".
   - Explore has no lens content. All lens readings live in the pre-seal `findings` stage.
6. **The canvas vocabulary lacks two view types EDA needs.** `consequences.py` has exactly five view kinds (`row_flow`, `lineage`, `table_focus`, `distribution`, `relationship`). The missing two are:
   - **`embedding`**: PCA scores, the sample map. It is the first-look figure for metabolomics and genomics, and `consequences.py` already reserves it "without a design decision".
   - **a matrix/heatmap view**: clustered correlation, the response-value audit, the PC × covariate R² grid, the sample–sample correlation.

   Both need the design decision the docstring asks for.
7. **Some views run before the seal and some after.** Outcome-free, row-descriptive views (units, plausibility, QC, missingness) may read every row before the seal. The engine already does this in `findings`. Any lever they inform is fitted in-fold under prediction; the engine already does this for the D-ratio, ComBat and variance filters. Views that read the outcome come after the seal and use training rows only under prediction.
8. **The packs' signature "sensitivity" exhibits are outcome-model estimates.** That puts them after the lock under inference (FOUNDATION §5 rule 6). The exhibits are:
   - nutrition's exclusion-sensitivity table and its five-energy-model forest;
   - metabolomics' "significant list under method A vs B" and "normalization changed the list by 30%".

   They belong to the "which of my decisions mattered?" view, not to EDA.

---

### 1. A shared frame for all five lenses

#### 1.1 How much a view involves the outcome (O-classes)

| Class | What it shows | Example | Prediction | Inference (before the lock) |
|---|---|---|---|---|
| **O0** outcome-free | data, structure, quality | QC drift, units, missingness by variable, predictor correlations, PCA colored by batch | free; may be pointed at | free; may be pointed at |
| **O1** outcome alone | the outcome's own distribution, missingness, event count | event rate, floor/ceiling of an outcome scale | allowed (STRATOS U-domain); recorded, because levers follow (imbalance, scale) — engine does this | allowed; recorded |
| **O2** outcome × design or technical variable | is the outcome tangled with how the data were made? | group × batch crosstab; library size or total signal vs outcome | compute as a gate; show the verdict; record if opened | the same; it guards the design, not the research question |
| **O3** outcome × candidate predictor or exposure | the research question, visualized | binned outcome-vs-predictor curves, Table 1 stratified by outcome, missingness-vs-outcome forest, PCA colored by the outcome group | training rows only; recorded; levers in-fold (MODELING_SEQUENCE ruling 3) | **never pointed at or ranked; not in the default gallery; reachable behind a counted "Outcome views (recorded)" door.** This squares "never block" (MODELING_SEQUENCE §4) with STRATOS |
| **O4** outcome-model estimate | Table 2, spline, volcano, PLS-DA, KM by exposure, calibration | — | model evaluation, after the fit | after the lock only (FOUNDATION §5 rule 6) |

**Ranking rule:** only O0 to O2 noticings may be ranked or pushed.

#### 1.2 How a noticing is measured, so that it can be ranked

Each noticing carries four things:
- a **measure**, deterministic and replayable;
- a **reference** shown beside it: the share expected by chance, a stated convention, or the field's typical range. It is never a PASS/FAIL stamp; the hard stop in DOMAIN_SCIENCE §01.2 forbids that.
- a **reach**: does it touch the declared outcome, exposure or predictors, and what share of rows or features does it affect?
- a **thread**: the later decision it returns at (`routes_to` / `lever_label` already exist on findings).

The engine's drift reading in `detectors/assay.py` is the template. It "states the share expected by chance beside the observed one."

Rank by tier first, then by reach, then by excess over the reference:
- **T1, blocker.** Nothing downstream is valid until it is answered. Examples: perfect batch–outcome confounding, an ambiguous energy unit, an unreadable data type, single-cell data, already-logged data facing a log, impossible values in the outcome, unresolved zero semantics.
- **T2, changes a number a reviewer would challenge.** It routes to a lever.
- **T3, shapes the interpretation.** It becomes a caveat in the manuscript.

Show "Worth a look" as the top 3 (BLUEPRINT §11 rule 7: "≤ 3 pushed findings, the rest counted and typed"). Each group header shows its own count. BLUEPRINT §14.2's "ordered by how much the field changes" is the same principle applied to readings.

#### 1.3 The scaling ladder (columns or features)

| Width | Per-column views | Feature × feature | Missingness | Engine limits already in place |
|---|---|---|---|---|
| ≤ ~30 | every column as small multiples or a full strip | clustered heatmap with values printed (k ≤ ~15, survey pack B5.4) | full pattern matrix / UpSet | — |
| ~30–500 | Strip: top 12 by the noticing's measure, then "and N more" | clustered heatmap, no values, clusters named | UpSet top 10–15 patterns | `MAX_SHOWN = 12`, `MAX_PAIRS_P = 500` (explore.py) |
| ~500–4,000 | one distribution of a per-feature statistic + top-12 strip | redundancy clusters and an effective feature count; never the matrix | per-sample bar + per-feature histogram | `MAX_CLUSTERED = 4000` (assay.py) |
| ≥ 4,000 (p ≫ n) | the same | sample × sample (n × n), PCA scores and loadings, modules | the same | Arrow column-wise summaries (`datastore.summaries`) |

Rows scale separately: above ~5,000 rows, scatters become hexbins or binned views and relationships become quantile-binned. Families (nutrients, labs, a survey block, genes) are confirmed as one block (§14.2) and drawn as one strip.

---

### 2. Dietary intake (24-hour recalls, FFQ, food records, body composition)

**Typical shape:** person-day tables with 10–65 nutrient columns (NHANES DR1T*/DR2T*), recalls in long form, FFQs with 100–200 items, and food-level long tables. Usually 600 to over 10,000 rows.

#### View groups, by question

| # | Question | Views (primary first) | Layout | Engine today |
|---|---|---|---|---|
| D1 | Are energy and nutrients in the units they claim? | Atwater scatter (declared vs 4P+4C+9F+7A, identity line, colored by suspected unit) · each nutrient's median against its plausibility band · BMI recomputed vs declared; FMI + FFMI = BMI | Focus, then a Strip of nutrients | `pack::dietary::atwater` (unit asked when the ratio admits two readings, the §14.3 amendment); nutrient corroboration r ≥ 0.3 with energy (ledger). BMI identity: not found |
| D2 | Who reported implausibly, and whom would a screen remove? | energy by sex with each competing screen drawn (Willett 500–3,500 / 800–4,000; NHS/HPFS 500–3,500 / 800–4,200) · EI:BMR with Goldberg cut-offs (PAL and d stated) · participant flow, one step per screen | Flow | `pack::dietary::implausible_intake`; `methods/misreporting.py` (Goldberg/Black, Schofield); cohort row flow |
| D3 | How much of one day is the person's usual diet? | structure card (recalls per person, day of week, sequence) · variance-components strip (within:between ratio, ICC, λ at the observed n, ranked by λ ascending) · shrinkage plot (single day vs mean vs usual, 5th/95th marked) · day-1 vs day-2 Bland–Altman · % zero days | Strip, then Focus | repeats detector (recalls stated as repeats); `usual_intake` stage (NCI MIXTRAN/DISTRIB); `calibration` stage (regression calibration). No per-nutrient λ strip yet |
| D4 | What travels with total energy, and what travels together? | R²(nutrient ~ energy) strip, then the nutrient-vs-energy scatter with residual arrows (the energy method's own storyboard) · Spearman heatmap of **energy-adjusted** nutrients, clustered, with the food-source clusters named · macronutrient closure (sum of %E; ternary) | Strip, then Focus; a heatmap (no view type for it yet) | `pack::dietary::energy_adjustment`, `pack::dietary::compositional`, `methods/energy.py`, `percent_energy`, `substitution`; Explore's \|r\| ≥ 0.9 pairs |
| D5 | Whom does the sample stand for, and who is missing? | design card (weight in use, Σw, strata, PSUs, minimum PSUs per stratum, Kish n_eff) · the exposure's distribution weighted vs unweighted · day-2 completion flow; completers vs non-completers (SMD on day-1 energy, age, sex) · missingness typed (whole recall / FFQ blank / nutrient absent from the database) | Flow + table focus | `survey_weights`, `partial_design`, `lonely_psu`; `missing.supplied_copies` (NHANES DXA 5 implicates); Explore's survey finding (prediction); energy-aware fill |

#### Noticings worth pointing at

| Noticing | Measure | Reference shown beside it | Tier | Returns at | Engine |
|---|---|---|---|---|---|
| Energy not in kcal, or mixed units across rows | Atwater ratio median; Spearman ρ of the ratio with energy | 0.90–1.10 kcal; ~4.18 kJ; drift means mixed | T1 | every energy step | yes |
| Implausible reporters | % outside each convention (both shown); % below Goldberg | DLW: AMPM under-reports ~11% overall | T2 | eligibility; a prespecified sensitivity analysis | yes |
| Misreporting that differs by body size | ρ(EI:BMR, BMI) | 0 | T2 (O0 only when BMI is not the outcome) | exclusions; covariate | partial |
| Weak reliability of the exposure | λ at the observed n; within:between ratio; days needed for r = 0.8; 1/λ² penalty | pack range 1.3–26.9 | T2 | measurement error (RC, usual intake) | computed inside stages; not surfaced |
| Episodic consumption | % zero days per item | — | T2 for the exposure | form step ("mass at zero"); NCI two-part | form handles mass at zero |
| Energy dependence | R²(log N ~ log E) | macros 0.4–0.9; micros 0.1–0.6 | T2 | energy adjustment | the finding uses r ≥ 0.3 |
| Closure | spread of Σ%E − 100; share within ±2 | — | T1 for an all-parts model (singular); otherwise T2 | substitution | yes |
| Food-source cluster of the exposure | max \|ρ\| with any other energy-adjusted nutrient; cluster size | — | T3 | confounder set ("other dietary components default to confounders"); discussion | not found |
| Design effect | Kish n_eff / n; DEFF; minimum PSUs per stratum | 1 PSU breaks Taylor variance | T2 (lonely PSU T1) | estimand (population vs sample) | lonely PSU yes; n_eff not found |
| Selective day-2 non-completion | SMD of day-1 traits, completers vs not | 0.10 shown as a convention | T2/T3 | missing data (MAR evidence; WTDR2D) | not found |
| Consecutive-day recalls | share of people with adjacent dates | — | T2 (λ overestimated) | usual intake | not found |

#### Views that involve the outcome

- **O3:**
  - Misreporting-by-BMI when adiposity is the outcome.
  - The Goldberg screen when the outcome is body size: its BMR denominator is a function of weight, so the screen is selection on the outcome. The pack's §02 badges this concern SETTLED.
  - FOUNDATION's Flow shows "how they differ from who stays". Under inference before the lock, that comparison must leave out the outcome, or record it.
  - Table 1 by exposure quintile is O0 (confounding structure) unless it carries an outcome row, which is the crude association.
  - Missingness (day 2, DXA) against the outcome.
- **O4, after the lock only:** the RCS dose–response, the quintile Table 2, the substitution forest, the exclusion-sensitivity table's "primary effect estimate" column, and the five-energy-model comparison forest.

#### From 29 columns to 20,000 features

- 10–30 nutrients: every strip shows everything; the heatmap prints its values.
- The full NHANES nutrient file plus food-pattern equivalents (~65–100 columns): strips show the top 12; the heatmap shows named clusters; blocks are nutrients, food groups, design, anthropometry.
- FFQ with 100–200 items: patterns use food groups, not nutrients (pack §07), plus a histogram of item non-response per participant.
- Food-level long tables: aggregate to person-day (the grain question) before any view.
- 20,000 arises only where the lenses intersect (diet with metabolomic biomarkers). The groups then come from both packs, de-duplicated by question.

---

### 3. Clinical measurements and labs (including EHR)

**Typical shape:** 10–300 columns; 100 to over 100,000 rows; often long and longitudinal.

#### View groups

| # | Question | Views | Layout | Engine today |
|---|---|---|---|---|
| C1 | Is each value possible, and in one unit? | per-analyte density with the reference band (source labeled), plausibility rules and a rug of extremes, ranked by impossible count · unit audit (bimodal at a conversion ratio; pre/post overlay) · censored-aware histogram (a bar at the LOD labeled `<0.3 (n=214)`) · default-value and digit mass (120/80, 98.6) | Strip, then Focus | `impossible_vs_extreme` (NHANES 2017–18 bands; CDC growth z for children), `impossible`, `mixed_units` (detect, never convert), `censored_values`, `text_numeric`, `number_format`, `mixed_result_type`, `default_value_mass`, `detectors/codes.py` |
| C2 | What is missing, and what does missingness mean here? | missing share per variable · co-missingness patterns (UpSet; panel blocks) · missingness over calendar time and by site · archetype per variable (not ordered / not resulted / not applicable / skip) | Strip + pattern view | `methods/missing.py` rules; cohort `row_loss_concern` (>10%); Explore's missing share by sociodemographic group. Co-missingness, calendar time and archetypes: not found |
| C3 | Who is in the analysis, and who left? | participant flow, one box per criterion, arithmetic verified · Table 1 with SMDs (not p-values), missing count per cell, medians for skewed labs; stratified by exposure (inference) or overall / development–validation (prediction) | Flow + table focus | cohort/eligibility row flow; a Table 1 exists in legacy `turbotab/manuscript.py`, none found in core export |
| C4 | Do repeated measurements tell a believable story? | trajectories for a sample of people, ranked by implausible jumps · visit spacing · counts of temporal implausibility | Strip, then Focus | `temporal_implausibility`; repeats detector; `time_varying` stage |
| C5 | Do site, era, device or group change the data? | distributions by site or era · missing share by sex, race/ethnicity, income (TRIPOD+AI 7) · dataquieR's "accuracy" (unexpected distributions and examiner/device associations) | Strip | Explore `_quality_finding` (gap ≥ 5 points) |
| C6 | Which predictors are redundant, sparse, or too many? | clustered Spearman correlation among predictors · near-zero variance · sparse categories · candidate parameters vs the Riley minimum n | heatmap + table | Explore `collinear` (\|r\| ≥ 0.9), `low_variance`, `wide`; shelf (Riley first) |

#### Noticings

| Noticing | Measure | Reference | Tier | Returns at | Engine |
|---|---|---|---|---|---|
| Impossible values | count beyond plausibility, per analyte | pack bounds (CONVENTION) | T1 in the outcome; otherwise T2 | repair (set to missing) | yes |
| Mixed units | ratio of two-component means vs a known factor; share in the minor component | ×18.0 glucose, ×88.4 creatinine … | T1 for that analyte | hard-stop ask | yes |
| Values below the LOD | % censored per analyte | 10% warning, >20% not defensible (CONVENTION) | T2 | censored handling by purpose | yes |
| Extreme but possible | share outside the NHANES central 98% | ~5% of healthy people fall outside an RI by construction | T3 (keep them) | coach | yes |
| Heavy missingness / complete-case loss | share per variable; rows lost | >10% concern | T2 | missing-data step | yes |
| Panel co-missingness | Jaccard of missingness indicators (≥ 0.8 blocks) | — | T3 | archetype → handling | not found |
| Calendar discontinuity | largest jump in monthly missing share | — | T2 under prediction (missing-indicator stability) | missing data | not found |
| Default or digit mass | excess at known defaults; share of terminal 0/5 | 20% expected | T3 | caveat | yes (defaults) |
| Temporal implausibility | count of implausible within-person changes | Kahn temporal rules | T2 | repair | yes |
| Data quality differs by group | gap in missing share across groups | 5 points (engine) | T2 | subgroup performance; fairness | yes |
| Redundant predictors | pairs with \|r\| ≥ 0.9 | — | T3/T2 | selection; form | yes |
| Too many candidate parameters | parameters / Riley n | — | T2 | shelf | yes |
| Race-based equation present | name/value signature | 2021 race-free CKD-EPI | T2 | caveat | not found |

#### Views that involve the outcome

- **O1:** event rate and outcome missingness. The Riley n needs prevalence.
- **O2:** outcome rate by site or era (shifting case mix).
- **O3:**
  - The pack's starred **missingness-vs-outcome forest**. Under prediction it is legitimate evidence for the missing-indicator lever, on training rows and recorded. Under inference it informs no decision, because the outcome is always in the imputation model; it goes behind the door.
  - Analyte densities **faceted by outcome**, which the pack suggests in §A1.2 "for extra value".
  - **Table 1 stratified by outcome**: the §A3 "natural stratifier" for prediction papers. Under inference, stratify by exposure.
- **O4:** KM by exposure, forest plots, calibration, ROC, decision curves, subgroup performance.
- Leakage checks need timestamps against the index date, not outcome values, so they are O0. Examples: a variable recorded because the outcome happened, or a post-baseline lab. Worth building (anti-patterns in §A5.5).

#### From 29 columns to 20,000 features

Clinical tables rarely exceed ~300 columns.
- Long EHR data: aggregate to the grain first.
- Co-missingness: show patterns, which scale with the number of patterns, not columns.
- Correlation: clustered heatmap up to ~50 predictors, a cluster list beyond.
- 100k rows: binned histograms (DuckDB already bins) and hexbins.
- Calendar views: binned by month.

---

### 4. Metabolomics and proteomics

**Typical shape:** untargeted runs have 400–30,000 features with QC and blank rows (the fixtures are 400 × 81). Targeted panels have 40–600 named analytes and can be 40 × 400, the reverse orientation.

#### View groups

| # | Question | Views | Layout | Engine today |
|---|---|---|---|---|
| M1 | Was the run stable? | sample-role timeline (injection order × role, batches shaded) · TIC per injection with LOESS per batch and QCs marked · histogram of per-feature Spearman ρ with run order, chance share beside it · QC trajectories of the most-drifting features · QC RSD CDF with 20/25/30% lines (pre-correction; pre and post when corrected) · QC detection-rate histogram | timeline + Strip, then Focus | `sample_roles`, `pooled_qc` (by variance), `no_pooled_qc`, `no_run_order`, `acquisition_design`, `run_order` (Spearman + BH, chance share); `methods/qc_drift.py` (QC detection 70%, QC-RLSC, RSD 20/30% on **corrected** QCs, refuses < 5 QCs per batch or extrapolation) |
| M2 | Is the main structure technical or biological? | PCA scores (% variance on the axes, equal or variance-proportional aspect, QCs overlaid, T² ellipse labeled as such), colored **first by batch and injection order** · PC × covariate R² grid (batch, order, TIC, plate, storage, sex, age, BMI, fasting) · variance partition · group × batch crosstab (O2) | embedding + matrix | `batch.principal_components` / `batch_figure`; `batch_findings` (Cramér's V; refusal under perfect confounding). R² grid: not found |
| M3 | What do the blanks and zeros mean? | **missing rate vs mean log intensity, one point per feature** (the figure that decides the imputation) · per-sample missing by injection · per-feature missing histogram · census of zeros vs NA · check for prior imputation | Focus | `left_censored`, `zeros_or_missing`, `already_transformed`; detection-limit options (half-min, Tobit MI, in-fold censored fill); QRILC |
| M4 | How many distinct compounds, and are the samples on a common footing? | redundancy clusters → effective feature count · total signal and RLA boxes per sample · signal dominance (top features' share; closure risk) · mean–SD plot (heteroscedasticity) · dynamic range | Strip + Focus | `redundancy` (Spearman, average linkage, leave-one-out stability), `ion_modes`, `duplicate_ids`, `empty_blocks`; PQN against the QC reference; `library_size_check` (total-signal kind, O2) |
| M5 | Which samples are unusual or not independent? | per sample: TIC ratio to the median, features detected, Hotelling T² / DModX · sample–sample correlation · census of repeated subjects | Strip | `repeated_subjects`. Sample outlier screen: not found |

#### Noticings

| Noticing | Measure | Reference | Tier | Returns at | Engine |
|---|---|---|---|---|---|
| No pooled QCs / no run order | presence | capabilities lost (RSD, D-ratio, drift); cannot be reconstructed later | T1 | QC chain | yes |
| Run-order drift | share of features with \|ρ\| > 0.3 and BH q < 0.05 | share expected by chance | T2 | QC-RLSC; batch | yes |
| QC imprecision | median QC RSD; share ≥ 20% / ≥ 30% | both shown (DISPUTED) | T2 | QC filter | post-correction only; the pre-correction CDF is missing |
| Batch dominance | R² of PC1–3 on batch vs on biological covariates | — | T2 | batch (covariate under inference; reference ComBat in-fold under prediction) | partial |
| Batch–group confounding | Cramér's V; perfect = rank-deficient | — | T1 | refusal | yes |
| Left-censoring | ρ(per-feature missing rate, mean log intensity) | strongly negative = censoring; untargeted runs typically 10–40% missing | T2 | imputation | yes |
| What zeros mean | share of exact zeros; vendor pattern | — | T1 (ask) | zeros → NA or true zero | yes |
| Already transformed | negatives; max < 40; dynamic range < 10² | raw spans 10²–10⁹ | T1 before a log | transform | yes |
| Redundancy | features / clusters | e.g. 5,000 → ~1,200 | T3 | the count you may claim; multiplicity sentence | yes |
| Signal dominance | share of total signal in the top 1% of features | — | T2 if TIC/sum normalization is chosen | normalization | not found |
| Outlying samples | TIC < 50% or > 200% of the median; T² beyond 99% | — | T2 | exclusion | not found |
| Repeated subjects | duplicated subject IDs | — | T1 for independence | grouping | yes |

#### Views that involve the outcome

- **O2:** the group × batch crosstab and total signal vs outcome. Both are gates the engine already computes. Show the verdict.
- **O3, which the pack does not flag:**
  - **PCA colored by group** when the group is the outcome. The pack's §06.1 calls PCA "honest because it never sees the labels". The computation is label-blind, but the coloring displays the research question. Default to coloring by QC, batch and order; group coloring is recorded, and under inference it sits behind the door until the lock.
  - **Missingness by group** (Fisher per feature) is a feature-wise test, which is an analysis: declare it as a detection-frequency result rather than browsing it.
  - The **modified 80% rule** is label-aware. The pack calls it the better default, and also says in §02 coaching that filters must "never [use] the group labels". These conflict. Run it in-fold under prediction, and declare it before any O3 view under inference.
  - A confounder screen against the outcome (§08) is O3; against PCs it is O0.
- **O4:** volcano, box plots per metabolite with q, PLS-DA/OPLS-DA, S-plot, VIP, a heatmap of the top-k features, panel ROC, normalization sensitivity of the significant list. All are CONFIRMATORY and need their companions.

#### From 29 columns to 20,000 features

- **Targeted panel of 29–600 named metabolites:** per-analyte strips and box plots are legible. Do not apply untargeted RSD filtering to validated analytes (pack §00).
- **Untargeted 400–30,000 features:**
  - every per-feature reading becomes one plot over features (RSD CDF, ρ histogram, missing-vs-intensity scatter, D-ratio histogram, cluster sizes);
  - plus a top-12 strip of exemplar trajectories;
  - sample views (timeline, TIC, PCA, RLA) scale with n.
- **Feature × feature:** through redundancy clusters only. Above the engine's 4,000-feature cap, cluster within retention-time windows or on a variance-filtered subset, and say which.
- **Missingness:** above a few thousand features the features × samples heatmap breaks; replace it with a per-sample bar plus a per-feature histogram (the genomics pack's replacement rule).

---

### 5. Genomics and transcriptomics (bulk; genotype briefly)

**Typical shape:** 20,000–60,000 genes × 4–1,000 samples. The fixtures are 500 × 61, with single-cell at 1,202 × 81.

#### View groups

| # | Question | Views | Layout | Engine today |
|---|---|---|---|---|
| G1 | What are these numbers, and what do they permit? | "what your numbers are" card + capability matrix · column sums with a 1e6 line · per-sample log2(x+1) densities · gene-ID audit | table focus + distribution | `pack::genomics::data_type` (nine signatures; log-CPM/TPM/voom back-transform; shallow counts), `single_cell` refusal, `gene_id_versions`, `gene_id_mixed_vocabulary`, `gene_id_excel_corruption`, `gene_id_duplicates`, `counts_p_over_n` |
| G2 | Are the libraries comparable? | library size per sample (median line; < 50% flagged) · detected genes vs library size · top-gene share (rRNA/globin/mito) · size factors outside [0.5, 2] · RLE per sample | Strip | TMM factors; `library_size_check` (depth vs outcome, O2). RLE: not found |
| G3 | What is the main structure among samples? | PCA on VST with the top 500 variable genes (ntop stated; all-genes version beside), equal aspect, shape = batch · PC × covariate R² grid (library size, % zeros, batch, RIN, sex) · sample–sample correlation heatmap with annotation bars (sequential scale, numeric legend) · MDS (leading log-FC) | embedding + matrix | `batch.principal_components`, `batch_findings` |
| G4 | Is any sample an outlier, a swap or a duplicate? | distribution of each sample's median correlation to the others · PC leverage (one or two samples driving PC1) · sex check (XIST vs chrY genes against recorded sex) | Strip | not found |
| G5 | How many genes carry information? | histogram of log mean expression, retained vs removed shaded by a label-blind filter · mean–dispersion trend · % zeros per gene · filtering waterfall | distribution | in-fold variance filter, SIS screen size, `near_zero_variance`. A filterByExpr-style rule: not found |
| (G6) | Genotype QC | call rate, MAF, HWE, heterozygosity, relatedness (KING), ancestry PCA on LD-pruned variants | — | none (the pack is brief) |

#### Noticings

| Noticing | Measure | Reference | Tier | Engine |
|---|---|---|---|---|
| Data type unreadable, or normalized values fed to a count model | classification + confidence | capability matrix | T1 | yes |
| Single-cell | zero fraction, width, median count | refusal | T1 | yes |
| Gene-ID corruption or unversioned joins | counts | ~20% of papers (Ziemann 2016) | T1/T2 (never auto-repair) | yes |
| Library imbalance | max/min ratio; # below 50% of the median; size factors outside [0.5, 2] | — | T2 | partial |
| Depth-driven structure | R²(PC1 ~ log library size), R²(PC1 ~ % zeros) | — | T2 (normalization, not biology) | not found |
| Batch-driven structure | R² per PC × batch | — | T2 | partial |
| Batch × condition confounding | rank deficiency; Cramér's V | — | T1 | yes |
| Outlier or swap | z of median inter-sample correlation; sex mismatches | ~0.8 (CONVENTION) | T2 | not found |
| Top-gene dominance | share of reads in the top gene | > 10–20% | T3/T2 | not found |
| p ≫ n | p/n | regularization mandatory | T2 | yes |
| Replicates per group | n per group | Schurch ≥ 6 | T3 | not found |

#### Views that involve the outcome

- **O2:** library size vs condition (the engine's `library_size_check` reads y), batch × condition balance, n per group.
- **O3:** PCA colored by condition, and "does the top dendrogram split follow condition?" The pack's own "no separation by condition (say so)" reads the outcome. Handle it as in metabolomics. DESeq2's `smallestGroupSize` filter uses group counts only, not a statistic, so it is acceptable. Supervised filters (DE, fold change, "top variable then DE") are refused, or run in-fold.
- **O4:** p-value histogram, volcano, MA, DE heatmap, selection stability, learning curve.
- **Genotype:** HWE in controls and case–control differential missingness (Anderson et al. 2010) are standard QC steps that touch the outcome. Pre-specify their thresholds before any association view.

#### From 29 columns to 20,000 features

- **A 29-gene targeted panel** (qPCR, NanoString): per-gene views work, but the data-type card does not cover Ct values or housekeeping normalization. The pack is honestly thin here.
- **500 genes (the fixtures):** the pair scan is allowed (`MAX_PAIRS_P`), but gene × gene is still the wrong question.
- **20,000–60,000 genes:** follow the pack's §07 "figures that break at p ≫ n" table:
  - PCA on the top 500 genes, plus an n × n sample correlation;
  - per-gene statistics drawn as distributions;
  - a top-12 strip by a label-blind measure.
- **Genotype at 1e5–1e7 variants:** distributions of per-variant statistics and sample-level views only.

---

### 6. Survey and questionnaire instruments

**Typical shape:** 20–300 items in blocks of 5–36; the fixtures are 45 × 301; NHANES modules include skip patterns.

#### View groups

| # | Question | Views | Layout | Engine today |
|---|---|---|---|---|
| S1 | Are the answers on the scale they claim? | block summary (items, response run, declared scale, instrument hypothesis, codes) · **response-value audit heatmap** (items × observed values, out-of-run cells highlighted) · NRS heaping at 0/5/10 | matrix + table | `detectors/scales.py` (blocks, run, declared scale, codes never recoded), `pack::survey::sentinel_codes`, `ordinal_declared`, `detectors/codes.py`; legacy instrument fingerprints |
| S2 | Did people answer with care? | longstring distribution · IRV distribution (count with zero variance) · Mahalanobis D² · completion time | Strip | not found |
| S3 | How are the answers distributed? | diverging stacked bars per block, sorted by net agreement, anchors verbatim, n per item (100% stacked for frequency scales) · item small multiples (bimodality) · score histogram on theoretical limits with floor/ceiling %; cut-points only for a confirmed instrument | Strip, then Focus | scale scoring (`methods/scales.py`). Floor/ceiling reading: not found |
| S4 | Do the items belong together, and which way do they point? | item–rest r dot plot (lines at 0 and 0.30; audit re-rendered after each declared reversal) · polychoric heatmap per block (fixed −1…+1) · parallel-analysis scree · ω (α labeled customary) | Strip + matrix | `turbotab/survey.py` reverse-coding audit (no auto-reverse); `methods/scales.py` ω total/hierarchical. Parallel analysis: not found |
| S5 | Who skipped what? | item missingness per item and per respondent · respondents under the half rule or the instrument rule · skip gates · don't-know share | Strip + Flow | partial (codes, scoring) |

#### Noticings

| Noticing | Measure | Reference | Tier | Engine |
|---|---|---|---|---|
| Sentinel codes | share of respondents with out-of-run values, per item | a 9 in a 1–5 block | T1 for that item (ask; never recode) | yes |
| Instrument match | block length + range + prefix | a hypothesis; cut-points only once confirmed | T2 | legacy |
| Reverse-keyed candidates | # of negative corrected item–rest r | four causes, cannot be told apart | T2 (hard stop: ask) | yes |
| Careless responding | share with longstring ≥ k, IRV = 0 | no consensus cutoff; pre-specify (Ward & Meade 2023) | T2 (with and without) | not found |
| Floor/ceiling | % at min/max per scale and item | 15% (Terwee; value shown) | T2 → ordinal vs metric | not found |
| Multidimensionality | parallel-analysis factors; ratio of first to second eigenvalue | — | T2 (the total is uninterpretable) | not found |
| Reliability | ω with CI; α > 0.95 | 0.70–0.95 (CONVENTION) | T3 | ω yes |
| Item missingness | share under the scoring rule | half rule or instrument rule | T2 | partial |
| Few categories or skew | ≤ 4 categories | B4 decision table | T2 | yes (ordinal declared) |
| NRS heaping | excess at 0/5/10 | — | T3 | not found |

#### Views that involve the outcome

- **O1** when the scale is the outcome: its categories, skew and floor/ceiling are allowed views and drive ordinal vs metric. The engine records them.
- **O3, which the pack does not flag:** B4 lists "whether compared groups have different response distribution shapes" as a diagnostic for choosing ordinal vs metric. When the groups are the exposure and the scale is the outcome, choosing after that view is a forking path.
  - Decide from O1 features, or declare ordinal with metric as the sensitivity analysis (the pack's own fallback).
  - Diverging bars faceted by exposure are also O3.
- **Not an outcome view, but a researcher degree of freedom:** the careless-responding rule. Fix it before any O3 or O4 view opens. Running EFA and then CFA on the same data is circular; label it.
- **O4:** proportional-odds results, predicted category probabilities, the metric-vs-ordinal sensitivity table.

#### From 29 columns to 20,000 features

- One block of 9 items: everything with values printed.
- 45 items in 5 blocks: views per block, plus a block strip ranked by noticing.
- 200–300 items: the block strip first, then drill into a block. Draw polychoric matrices per block only (k ≤ ~30), never 300 × 300.
- The audit heatmap shows the top 12 offending items, then "and N more".
- C/IER indices are per respondent, so they scale with n.

---

### 7. Tensions and open decisions for the orchestrator

1. **What "held back under inference" means.** MODELING_SEQUENCE §4 says "record and disclose; never block". STRATOS says IDA does not explore outcome–predictor associations. The proposal that satisfies both: O3 views are never pointed at, never ranked and not in the default gallery under inference; they sit behind a counted, recorded door. Explore's current order breaks this, because O3 comes first, unranked.
2. **Explore's ranking.** Relationship curves are the first 12 numeric predictors in column order. Rank them by a measured change under prediction. Under inference, put them behind the door. Lens groups need a home after the seal, or the EDA canvas must merge `findings` (pre-seal) with `explore` (post-seal).
3. **Two new view types** (`embedding`, plus a matrix or heatmap), and one semantic change. A `distribution` view whose units are features, not rows, needs a caption that says "each bar counts features".
4. **Pack contradictions to rule on:**
   - the modified 80% rule against "filters never use labels";
   - PCA's "honest" label against its coloring by the outcome;
   - survey B4's group-shape diagnostic;
   - the nutrition and metabolomics sensitivity exhibits, which are O4.
5. **Small engine additions with high pedagogical value, all outcome-free:**
   - the pre-correction QC RSD CDF (the engine computes RSD only after correction; pack §02 says reporting only post-correction values is circular);
   - a PC × covariate R² grid;
   - a missing-vs-intensity figure built from the existing finding;
   - per-nutrient λ and R² strips;
   - co-missingness and calendar-time missingness;
   - longstring/IRV, floor/ceiling and parallel analysis;
   - library and RLE QC, sex check, sample outlier screen;
   - Kish n_eff.
6. **Threads (Nolan's extension, in memory).** The "Returns at" column above gives each noticing's thread destination. Most map onto existing `routes_to` / `lever_label` targets.

---

### Sources checked for this brief, beyond the packs

- Heinze G, Baillie M, Lusa L, Sauerbrei W, Schmidt CO, Harrell FE, Huebner M. Regression without regrets – initial data analysis is a prerequisite for multivariable regression. *BMC Med Res Methodol* 2024;24:178 (PMC11308558). Quotes above: the outcome–predictor abstention; M/U/V domains; univariate outcome allowed; changes reported as "consequences of IDA".
- Zgraggen E, Zhao Z, Zeleznik R, Kraska T. Investigating the effect of the multiple comparisons problem in visual analysis. CHI 2018 (>60% of insights were false).
- Wilkinson L, Anand A, Grossman R. Graph-theoretic scagnostics. IEEE InfoVis 2005.
- Schmidt CO et al. Facilitating harmonized data quality assessments (dataquieR). *BMC Med Res Methodol* 2021;21:63 (integrity / completeness / consistency / accuracy).
- Kirwan JA et al. mQACC recommendations for QA/QC reporting in untargeted metabolic phenotyping. *Metabolomics* 2022;18:70.
- Love, Anders, Huber et al. Bioconductor rnaseqGene workflow, "Exploratory analysis and visualization" (pre-filtering, VST/rlog, sample distances, PCA, MDS).
- Anderson CA et al. Data quality control in genetic case-control association studies. *Nat Protoc* 2010;5:1564.
- Cited from memory, not re-fetched: Weiskopf & Weng 2013 *JAMIA*; Buja et al. 2009 *Phil Trans R Soc A* (line-up protocol).

## Sources
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/DOMAIN_PACKS.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/DOMAIN_SCIENCE.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/research/METABOLOMICS_PACK.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/research/NUTRITION_PACK.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/research/CLINICAL_SURVEY_PACK.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/research/GENOMICS_PACK.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/MODELING_SEQUENCE.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/BLUEPRINT.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/INBOX.md
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/explore.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/explore_previews.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/consequences.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/__init__.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/data.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/datastore.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/__init__.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/assay.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/genomics.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/lenses.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/plausibility.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/scales.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/codes.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/repeats.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/qc_drift.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/batch.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/omics.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/missing.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/scales.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/misreporting.py
- /Users/nhedglin/tabular-ml-lab/turbotab/survey.py
- /Users/nhedglin/tabular-ml-lab/turbotab/packs.py
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11308558/
- https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02294-3
- https://cs.brown.edu/research/ptc/assets/publications/zgraggeninvestigating.pdf
- https://dl.acm.org/doi/10.1109/INFOVIS.2005.14
- https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01252-7
- https://link.springer.com/article/10.1007/s11306-022-01926-3
- https://bioconductor.org/packages/release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html
- https://www.nature.com/articles/nprot.2010.116