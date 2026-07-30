# Metabolomics / proteomics pack — research specification

Same structure as `GENOMICS_PACK.md`: by modeling step, and within each step by **diagnostic**
(what the app detects), **coaching** (what it says, with the reason), and **presentation** (what it
draws, and what makes it publication-grade). Every recommendation carries **SETTLED** ·
**CONVENTION** · **DISPUTED**, so the app never asserts a norm as a fact.

**Research caveat, recorded because it changes how these numbers should be treated.** The session's
egress proxy blocked publisher domains (ACS, Springer, Nature, PMC, Bioconductor). The content below
comes from search-surfaced excerpts of primary sources plus domain knowledge. Items marked
**[verify-at-build]** are specific numeric thresholds that must be checked against the cited paper
before shipping — being wrong about a number is the worst failure mode a pack has.

---

## 00 · The sub-domain fork, before anything else

This pack must decide *which* sub-domain it is in, because conventions diverge sharply and the
wrong branch does real damage. Misapplied QC-RSD filtering on a targeted panel deletes validated
analytes.

| Sub-domain | Detection signal | Key divergence |
|---|---|---|
| **Untargeted LC-MS** | m/z–RT feature IDs; thousands of features; QC and blank samples | the full QC-filtering chain; PQN; Pareto |
| **Targeted** (Biocrates, MxP Quant) | named metabolites only, 100–600; concentration units (µM); LLOQ/ULOQ columns | regulatory-flavored LOD handling; internal-standard normalization; **no** RSD filtering of validated analytes |
| **NMR** | binned ppm columns, or named metabolites with no m/z | PQN near-mandatory; no blanks; glog literature is NMR-native |
| **Lipidomics** | shorthand names (`PC 34:1`, `TG 52:2`) | mol% composition reporting; class rollups; explicit compositional framing |
| **Bottom-up proteomics (DDA/LFQ)** | `proteinGroups.txt` columns, UniProt accessions, `Razor + unique peptides` | contaminant/reverse filtering; limma/msqrob; a *different* imputation literature |
| **DIA proteomics** | Spectronaut/DIA-NN report columns, `PG.Quantity` | much lower missingness; MaxLFQ; different imputation calculus |
| **TMT/iTRAQ** | `Reporter intensity corrected N`; plex structure | batch = plex; ratio compression; IRS normalization |

---

## 01 · Import and structure

### Orientation — features in rows, and the detection cascade

Untargeted tables ship both ways. Vendor exports (XCMS, MZmine, MS-DIAL, Progenesis, MaxQuant) are
overwhelmingly **features-in-rows**; MetaboAnalyst's own concentration-table format is
**samples-in-rows**. Detection, in priority order:

1. **Header-token test** on the first column: `m/z`, `mz`, `mass`, `RT`, `retention time`,
   `Compound`, `Metabolite`, `Protein.IDs`, `Accession`, `Lipid` → features-in-rows.
2. **Feature-name grammar.** Untargeted IDs follow recognizable grammars: `M123T456`
   (CAMERA/xcms), `123.4567_5.67` or `784.5876@8.21` (mz_rt), `FT0001`, `X123.4567`. Lipids follow
   shorthand nomenclature (`PC 34:1`, `TG(16:0_18:1_18:2)`, `Cer d18:1/16:0`). Proteins follow
   UniProt (`P02768`, `sp|P02768|ALBU_HUMAN`) or gene symbols. On row labels → features-in-rows; on
   column labels → samples-in-rows.
3. **m/z plausibility.** A numeric column of values ~50–1700, non-integer, 4–5 decimal places, is an
   m/z column. Paired with a second column in 0–30 (min) or 0–1800 (sec) → confirmed feature
   metadata block.
4. **Shape prior.** Features ≫ samples in untargeted work (500–30,000 vs 20–500). If one dimension
   is ≥5× the other *and* carries the feature-name grammar, that is the feature axis. **Never rely on
   shape alone** — a targeted panel can be 40 metabolites × 400 samples, i.e. the reverse.
5. **Sample-name grammar** on the other axis: `QC_01`, `Sample_003`, plate-well codes (`A01`–`H12`).

**[SETTLED]** that the tool must never guess silently. Present the inference with its evidence and
require confirmation: *"I think features are in rows because column 1 matches an m/z pattern and
4,812 rows carry `M###T###` labels."*

### Sample-role detection — the thing a generic tool cannot do

Regex library, case-insensitive, applied to sample names **and** to any metadata column named
`type`, `sample_type`, `role`, `class`, `group`:

- Pooled QC: `^QC`, `_QC`, `PQC`, `pool`, `pooled`, `QCP`, `SQC`, `QC[-_ ]?\d+`
- Dilution QC (dQC/RQC): `dQC`, `DIL`, `QC[-_ ]?(1|2|4|8|16)x`, `RQC`
- Blanks: `blank`, `BLK`, `^B\d+`, `solvent`, `extraction[-_ ]?blank`, `process[-_ ]?blank`, `water`
- System suitability: `SST`, `sys[-_ ]?suit`, `cond`, `equil`, `wash`
- Calibrants/standards: `CAL\d`, `STD`, `standard`, `IS`, `ISTD`, `NIST`, `SRM1950`, `LTR`
- Proteomics: `HeLa`, `QC_HeLa`, `iRT`, `pool`, `bridge`, `reference channel`

### Run order, batch, and design

Look for `injection.order`, `inj_order`, `run_order`, `sequence`, `acq_order`,
`AcquisitionDateTime`, `batch`, `plate`, `well`, `position`, `plex`, `TMT.channel`, `polarity`. If an
acquisition timestamp exists, derive run order from it. **If no run order exists, say so loudly** —
half the downstream diagnostics (drift, QC-RLSC, run-order PCA overlay) become impossible.

Also detect: group/class column; **subject ID**, to catch repeated measures (routinely missed);
timepoint; known confounders (age, sex, BMI, fasting status, medication, site, storage time,
freeze-thaw count).

### Value-state diagnostics

- **Zeros and NAs mean different things and vendors disagree.** XCMS `fillPeaks` writes small
  numbers, MZmine writes 0, MaxQuant writes 0 for "not quantified", Progenesis writes 0. The pack
  must ask: *do zeros here mean "not detected" or "true zero"?* Defaulting wrong corrupts every
  downstream step.
- **Already-transformed detection.** Any negative values, or a max below ~40 with a positive min and
  low dynamic range, or column means ≈ 0 → probably already log-transformed and/or scaled. Warn
  hard; a second log transform is a silent catastrophe.
- Dynamic range: raw untargeted intensities span 10^2–10^9. A ratio below 10^2 means something has
  already been done to the data.
- Duplicate feature/sample IDs, all-zero features, all-zero samples, constant features.
- Ion mode split: if `pos` and `neg` features coexist, flag that they are normally normalized
  separately then merged, and that the merge strategy affects results.

### Redundancy detection — a real differentiator

Untargeted features are not independent. Ionization produces adducts (`[M+H]+`, `[M+Na]+`,
`[M+K]+`, `[M+NH4]+`, `[M-H]-`, `[M+HCOO]-`), isotopologues (Δm/z 1.00336), dimers, and in-source
fragments. Cluster features by near-identical RT (±0.05–0.1 min) **and** high inter-feature
correlation (r > 0.9), and report an estimated *effective* feature count. If 5,000 features collapse
to ~1,200 clusters, the user's "5,000 metabolites" claim is wrong by ~4×.

### Coaching at this step

> *"I couldn't find any pooled QC samples. Pooled QCs — an aliquot of every sample, mixed, injected
> every 5–10 samples — are the field's standard evidence that your run was stable, and reviewers
> increasingly expect them (Broadhurst et al. 2018, Metabolomics 14:72; mQACC 2022). Without them I
> can't compute QC-RSD, D-ratio, or drift correction. If QCs were run but aren't in this file, add
> them now; they cannot be reconstructed later."*

> *"Subject IDs repeat — 48 samples from 24 subjects. Treating these as 48 independent observations
> would roughly double your apparent sample size and inflate significance."*

> *"Your `proteinGroups` table still contains 41 rows flagged `Potential contaminant`, 33 `Reverse`
> decoy hits, and 12 `Only identified by site`. These are search artifacts and must be removed
> before any statistics."*

### Presentation

- **Data-inventory table.** Total features, samples, biological samples by group, pooled QCs, blanks,
  standards, batches, run-order range, ion modes, % zeros, % NA, estimated redundancy-collapsed
  feature count. This becomes the seed of the manuscript's data-description sentence.
- **Sample-role timeline strip.** One horizontal strip: x = injection order, one tick per injection,
  colored by role, shaded by batch. Reveals in a glance: QCs not interspersed, QCs only at the start,
  all cases run before all controls (fatal confounding), batch boundaries.
- **Design crosstab.** Group × batch, group × plate, group × sex, with a warning on any zero cell.

---

## 02 · Quality filtering

### Diagnostic — what to compute per feature

1. **QC detection rate.** Retain features detected in **≥50% of QCs** (Dunn et al. 2011, *Nat
   Protoc* 6:1060 states the step explicitly); many labs use ≥2/3 or ≥80%. **[CONVENTION]**
2. **QC RSD (= CV) = SD/mean × 100** across pooled QCs.
   - **≤20%** — strict, FDA-flavored, standard in targeted assays. **[CONVENTION]**
   - **≤25%** — MetaboAnalyst's suggestion for LC-MS. **[CONVENTION]**
   - **≤30%** — the most common untargeted cutoff, and the QComics acceptance criterion
     (González-Domínguez et al. 2024, *Anal Chem* 96:1064), which also specifies **RT RSD < 2%** and
     **peak-width RSD < 15%**.
   - **[DISPUTED] which number.** The scoping review of pooled-QC practice (*Anal Chem* 2023) states
     plainly that *there is no widely accepted metric for delineating acceptable from unacceptable
     data quality.* Present 20/25/30 as a choice with consequences shown, never as a default with
     false confidence.
3. **D-ratio** = SD(QC) / SD(biological samples) per feature; robust variant uses MAD. Broadhurst et
   al. 2018; **acceptance criterion 50%**. Implemented as `dratio_filter` in structToolbox. Better
   than raw RSD because it asks the question that matters — *is technical noise small relative to
   biological signal* — rather than an absolute noise level. **[CONVENTION, rising toward SETTLED]**
   **[verify-at-build: 50% and the SD-vs-MAD default]**
4. **Blank ratio.** Median in biological samples ÷ median in blanks. Thresholds in use: **3×, 5×,
   10×, 20×**. **[DISPUTED — genuinely no consensus.]** Blanks must be *process* blanks (water through
   the full extraction with the same labware), not solvent-only injections; a solvent blank cannot
   detect plasticizer contamination.
5. **Missingness fraction**, overall and *per group*.
   - **80% rule**: keep a feature if non-zero in ≥80% of samples.
   - **Modified 80% rule**: keep if non-zero in ≥80% of the samples *of at least one class*. This
     preserves genuinely group-specific metabolites that the plain rule deletes. **[CONVENTION; the
     modified form is the better default and the pack should say why.]**
6. **Near-constant filter.** MetaboAnalyst scales an IQR filter to feature count: 5% when <250
   features, 10% for 250–500, 25% for 500–1000, **40% for >1000**. Aggressive and largely unreported
   in papers that use it. **[CONVENTION — and an under-reported one; flag it.]**
7. **Drift diagnostics.** Per feature, Spearman ρ of intensity vs injection order within batch; % of
   features with |ρ| > 0.3 and an FDR-significant trend; TIC vs run order.
8. **Dilution/response filter.** Where a dQC series exists, retain features whose response correlates
   with dilution (r > 0.7–0.8) — this discriminates real analytes from noise and saturated peaks.
9. **Sample-level QC.** Total signal per sample (flag <50% or >200% of median TIC), features detected
   per sample, Hotelling's T² status.

### Order of operations — a real source of error

> blank/background filter → QC-detection-rate filter → *(drift/batch correction, if any)* → RSD and
> D-ratio filter → missingness filter → sample-level normalization → imputation → transformation →
> scaling

Two subtleties the pack must get right:

- **RSD must be computed on un-imputed data.** Imputing with half-min creates a tight cluster of
  identical values that deflates SD, so post-imputation QC RSD is optimistically biased. Common
  mistake, real consequence.
- **RSD before drift correction penalizes correctable drift.** Where correction is applied, report
  RSD *both* before and after. That pair of numbers is what QC-methods reviewers want, and reporting
  only the post-correction number is circular.

### Coaching

> *"30% is the most commonly published untargeted cutoff and is the QComics acceptance criterion; 20%
> is stricter and standard in targeted assays; MetaboAnalyst suggests 25%. There is no consensus
> value — the scoping review in Analytical Chemistry says explicitly that no widely accepted metric
> exists — so pick one, state it in your methods, and I'll show you how many features each costs."*

> *"I'll keep a feature if it's detected in at least 80% of the samples in *any one* group, not 80%
> overall. The plain 80% rule silently deletes metabolites present in cases and absent in
> controls — precisely the kind of finding you're looking for."*

> *"Every filter above uses only QCs, blanks, and overall missingness — never the group labels.
> Filtering features by a group difference before testing them is circular and inflates false
> positives."*

### Presentation

- **★ Filtering waterfall table.** Rows = each filter in order; columns = features remaining, removed,
  cumulative % retained, threshold used. `4,812 → 3,940 (blank) → 3,610 (QC detection) → 2,984 (RSD
  ≤30%) → 2,701 (D-ratio ≤50%) → 2,540 (80% rule)`. Almost no published paper includes this;
  including it is a credibility marker.
- **QC RSD cumulative distribution.** x = RSD %, y = cumulative fraction of features, vertical lines
  at 15/20/25/30%. Shows exactly what a threshold costs. Overlay pre- and post-drift-correction as
  two lines — the single most persuasive QC figure in the field.
- **D-ratio histogram** with the 50% line. **TIC and QC intensity vs injection order**, batches
  shaded, LOESS overlaid. **Sample-level QC panel** ordered by injection.

---

## 03 · Missing data

### The central diagnostic is the mechanism, and it is diagnosable, not assumable

1. **Intensity-dependence plot.** Per feature, missingness rate vs mean observed intensity. A strong
   negative relationship = left-censoring / MNAR. A flat relationship = MCAR/MAR (peak-picking or
   alignment failure). **This plot decides the imputation method** and the pack should compute and
   show it. In proteomics the MNAR signature is very strong.
2. **Missingness by group.** A feature missing in 90% of controls and 5% of cases is *information*,
   not a nuisance. Fisher's exact per feature, FDR-corrected.
3. **Missingness by batch / run order / plate.** Clustered in one batch = technical artifact, not
   censoring.
4. **Missingness by sample.** Flag anomalously high (injection failure).
5. **Overall rate.** Untargeted LC-MS commonly 10–40%; DDA proteomics 20–50%; DIA often <10%;
   targeted a few %.
6. **Detect prior imputation.** If the minimum of every feature is exactly half the second-smallest,
   or a suspicious number of exact ties exists, the data have already been imputed.

### Method catalogue

| Method | Assumes | Verdict |
|---|---|---|
| **Zero replacement** | true absence | Not defensible for intensities; breaks log; creates false enormous fold changes. **Anti-pattern.** |
| **Half-minimum** | MNAR | The de-facto default (MetaboAnalyst). Deterministic, so it collapses all missing to one point, deflating variance and creating ties → can inflate significance and shrink QC RSD. Second-best after QRILC under MNAR in Wei et al. 2018, *Sci Rep* 8:663. **[CONVENTION — widely used, statistically criticized]** |
| **MinProb / MinDet / LOD-2** | MNAR | Same family. MinProb (random draw near the detection limit) beats MinDet because it preserves variance. |
| **QRILC** | MNAR | Draws from a truncated distribution estimated by quantile regression. **Best performer for MNAR** in Wei et al. 2018, with much smaller error than RF/SVD/kNN. **Best-supported default when the intensity-dependence plot shows censoring.** |
| **GSimp** | MNAR | Gibbs sampler with an embedded prediction model (Wei et al. 2018, *PLoS Comput Biol* 14:e1005973). Stronger than QRILC in principle because it uses inter-feature structure; heavier, less widely implemented. |
| **kNN** | MCAR/MAR | Good for MAR, **poor for MNAR** — no left-censoring constraint, so it *overestimates* censored values. |
| **Random forest (missForest)** | MCAR/MAR | Best for MCAR/MAR in Wei et al.; but also reported "consistently robust across all MNAR situations" and most suitable for label-free proteomics when the mechanism is unknown (*Sci Rep* 2021). **This is a genuine contradiction between the metabolomics and proteomics benchmark literature — the pack must not pretend it's resolved.** |
| **BPCA / SVD / PPCA** | MCAR/MAR | BPCA is a top performer in proteomics benchmarks; poor under MNAR. |
| **msImpute** | mixed | Purpose-built for label-free MS proteomics with an MNAR-aware mode. |
| **Multiple imputation (mi4p)** | mixed | Propagates imputation uncertainty into the test statistics instead of treating imputed values as observed. Statistically the most correct and the least used. |

### The pack's recommended default — stated as a recommendation, not a truth

1. **Filter first** — never impute a feature that fails the missingness filter.
2. If **differential missingness** is FDR-significant, do **not** impute; report presence/absence with
   a Fisher test.
3. If the intensity-dependence plot shows **censoring** (the usual case): **QRILC** or GSimp.
4. If missingness is **flat** with respect to intensity: random forest or kNN.
5. Always offer half-min as the "match what everyone else published" option, with the caveat attached.
6. **★ Always run the primary analysis under two imputation schemes and report whether conclusions
   change.** This sensitivity analysis is the single highest-value thing a tool can add here — cheap,
   almost never done, and it directly answers the reviewer's objection.

**Proteomics-specific:** impute at the **peptide/precursor level before rolling up to proteins**, not
after. Rolling up first mixes two different missingness processes.

### Coaching

> *"About 22% of your values are missing, and missingness is strongly higher for low-abundance
> features. That pattern means most of your missing values are left-censored — the metabolite was
> there, just below the detection limit — not randomly lost. kNN and random forest assume random loss
> and will systematically overestimate these values."*

> *"Feature `M287T412` is missing in 94% of controls and 8% of cases. I'm not going to impute it —
> imputing would manufacture a fake concentration for a compound that genuinely isn't there. I'll
> report it as a detection-frequency result instead."*

> *"I've run your main comparison twice, once with QRILC and once with half-minimum. 41 of the 44
> FDR-significant metabolites are significant under both. I'll note the 3 that aren't; they should
> not go in your abstract."*

### Presentation

- **Missingness-vs-intensity scatter** (the mechanism figure), LOESS trend. This is the *evidence* for
  the imputation choice and reviewers respect it.
- **Missingness heatmap** (features × samples, binary), samples ordered by injection, annotated by
  batch and group. Reveals a bad injection or a bad plate instantly.
- **★ Before/after density overlay.** Observed vs imputed values. Publication-grade version shows
  imputed values sitting in the **left tail**, not the middle — the visual proof the imputation
  respected censoring.
- **Sensitivity table:** significant features under method A vs B, with the overlap.

---

## 04 · Normalization, transformation, scaling

The field routinely conflates three orthogonal operations. **[SETTLED]** that they are conceptually
distinct; the coaching must make the user say what each is for.

> **Row (sample) normalization** removes differences in overall sample concentration/dilution.
> **Transformation** makes the data symmetric and stabilizes variance.
> **Column (feature) scaling** decides how much weight each metabolite gets.
> Applied in that order. They answer different questions.

### Sample-level normalization

| Method | What it does | Status |
|---|---|---|
| **Sum / TIC** | divide by total signal | Simplest and most common. **Fails when a few metabolites dominate or change massively.** Also imposes closure (§07). **[CONVENTION, with a known failure mode]** |
| **MSTUS** | sum over features common to all samples | Avoids xenobiotics contaminating the normalizer; reported to outperform alternatives in urinary profiling. **[CONVENTION, strong for urine]** |
| **PQN** | per-feature quotients vs a reference profile, divide by the **median quotient** | Dieterle et al. 2006, *Anal Chem* 78:4281 showed PQN "by far more robust and more accurate" than integral and vector-length normalization across simulated spectra, cyclosporin-A studies and >4,000 control animal samples. **[SETTLED as a reasonable default for biofluids; the choice of reference (QC median vs control median vs all) is [DISPUTED] and materially changes results — expose it.]** |
| **Median-fold change** | median of per-feature fold changes vs reference | Equivalent in spirit to PQN. |
| **Quantile** | force identical value distributions | Borrowed from microarrays. Very aggressive; can erase genuine global shifts. **[DISPUTED for MS metabolomics; more accepted in proteomics.]** |
| **Internal standard** | divide by spiked IS | **[SETTLED]** for targeted assays and lipidomics (class-matched IS). For untargeted, one IS normalizes only compounds with similar behavior. |
| **Creatinine (urine)** | divide by creatinine | Widely accepted, **but creatinine excretion is altered by renal impairment** — actively misleading in kidney disease. **[CONVENTION with a named clinical failure mode]** |
| **Tissue weight / cell count / protein** | divide by biomass | **[SETTLED]** for tissue/cell metabolomics; its absence is a common reviewer complaint. |

The systematic comparison in *ACS Meas Sci Au* 2024 ("Closing the Knowledge Gap of Post-Acquisition
Sample Normalization") found **"dramatic discrepancies between the outcomes of different sample
normalization methods"**, with data quality itself conditioning which method works. That is the
honest headline: **normalization choice is not cosmetic and there is no universally correct answer.**

### Transformation

- **log2 / log10** — **[SETTLED]** default for MS intensity. Converts right-skewed multiplicative data
  to approximately symmetric, converts fold changes to differences. Requires handling zeros — and a
  pseudo-count is a hidden imputation, so say so.
- **glog** — a log with an offset λ accounting for the additive technical-noise floor, so
  low-abundance features aren't variance-exploded. Parsons et al. 2007, *BMC Bioinformatics* 8:234
  reported the highest classification accuracy after glog on two of three NMR datasets. Implemented
  as `glog_transformation` in `pmp`. **[CONVENTION; underused and a genuine improvement for data with
  many near-LOD features]**
- **VSN** — variance-stabilizing normalization; produces glog-scale intensities and handles sample
  scaling simultaneously. Standard in proteomics.

### Feature-level scaling

| Scaling | Divisor | Effect | Status |
|---|---|---|---|
| **Mean-centering only** | — | abundant metabolites dominate every PC | Rarely appropriate alone |
| **Auto / unit-variance** | SD | every feature equal weight | Removes dependence of rank on average concentration and fold-change magnitude (van den Berg et al. 2006, *BMC Genomics* 7:142); amplifies noisy low-abundance features. **[SETTLED as right when you care about all metabolites equally]** |
| **Pareto** | √SD | intermediate | **The metabolomics cultural default**, especially in the SIMCA/OPLS-DA tradition. Reduces masking by abundant metabolites but is **sensitive to large fold changes**. **[CONVENTION — dominant but arbitrary; van den Berg's own analysis preferred autoscaling and range scaling]** |
| **Range** | max−min | scales to observed biological range | As good as autoscaling in van den Berg; very outlier-sensitive. |
| **Vast** | SD × (SD/mean) | favors stable, reproducible features | Good for robust discriminators; suppresses high-relative-variance features. |
| **Level** | mean | relative response, emphasizing relative change | Useful when % change is the question. |

**The near-universal published combination is `sum-or-PQN → log → Pareto`. That combination is
[CONVENTION], not [SETTLED], and van den Berg's evidence actually favors autoscaling.** A tool that
presents Pareto as *the correct choice* would be confidently wrong; a tool that presents it as *what
most of the field does, for these reasons, with this weakness* is right.

### Coaching

> *"Normalization and scaling are two different jobs. Normalization across samples fixes the fact that
> one urine sample was more concentrated than another. Scaling across features decides whether your
> most abundant metabolite gets to dominate the model. You need both, and they need to be reported
> separately — 'the data were normalized' is not sufficient for a reviewer."*

> *"Total-ion-current normalization is the simplest option and I can use it, but if any single
> metabolite changes dramatically between your groups — a drug, a contrast agent, glucose in a
> diabetes study — TIC normalization will push every other metabolite in the opposite direction and
> manufacture findings."*

> *"Changing the normalization method changed the significant-metabolite list by 30%. That is normal
> and it is why your methods section must state exactly what you did. It also means you should not go
> shopping through normalization methods for the one that gives the nicest p-values — pick on
> principle before you look at the results."*

### Presentation

- **Before/after normalization boxplot grid**, one box per sample ordered by injection. Successful
  normalization = medians aligned. The standard visual proof.
- **Relative log abundance (RLA) plots** — per-sample distribution of (value − feature median), before
  and after. Tighter, zero-centered boxes = better. Under-used and very persuasive.
- **★ Mean–SD (variance-stabilization) plot** — running SD vs rank of mean, before and after
  transformation. A flat line after glog/VSN proves heteroscedasticity was fixed. Standard in the
  `vsn`/proteomics world and almost unknown in metabolomics; adding it is a differentiator.
- **Side-by-side PCA under 2–3 scaling choices** — makes the arbitrariness visible and honest.
- **Normalization decision record** — a generated methods sentence naming all three operations.

---

## 05 · Batch correction and drift

### Before correcting anything, test whether correction is possible

1. **Confounding check — gating, not advisory.** Crosstab group × batch. If any group is wholly or
   nearly contained in one batch, batch and biology are the same variable and **no correction can
   separate them.** The pack must refuse to run a correction silently and say why.
2. **Batch-effect magnitude.** PCA colored by batch; ANOVA of PC1–PC3 scores against batch;
   **variance partitioning (PVCA-style)**. "Batch explains 31% of variance, group explains 4%" is the
   right way to decide whether correction is warranted.
3. **Within-batch drift.** Per-feature Spearman ρ vs injection order; QC intensity trajectories.
4. **QC adequacy.** QC-RLSC/QC-RSC/SERRF need QCs interspersed every 5–10 injections. With fewer than
   ~5 QCs per batch, LOESS/spline fitting is unstable and manufactures artifacts, especially by
   extrapolating past the first and last QC.
5. **Post-correction verification.** QC RSD before/after; D-ratio before/after; PCA by batch
   before/after; **and a check that biological signal was not destroyed** — technical-replicate
   correlation, or a known positive control.

### Method catalogue

**Within-batch drift** (requires QCs + run order): **QC-RLSC** (Dunn lineage, most published; noted
overfitting risk if the span is too small) · **QC-RSC** (cubic spline, in `pmp`; generally more
stable) · **QC-SVRC** (support vector regression; slightly outperforms QC-RSC) · **SERRF** (Fan et
al. 2019, *Anal Chem*: assumes systematic error depends on the behavior of *other* compounds and uses
RF to pick correlated predictors; compared against 15 methods on six lipidomics datasets, reduced
average technical error to ~5% RSD — currently the strongest empirical result for large cohorts) ·
**TIGER** · **NOMIS / RUV-random / CCMN**.

**Between-batch:** **ComBat** (empirical Bayes; effective when batches are balanced with respect to
the outcome) · **batch median centering** (crude, transparent, hard to break) · **include batch as a
covariate or random effect in the model rather than correcting the data** — statistically the
cleanest, because it propagates uncertainty instead of pretending corrected values are observed.
**[SETTLED in biostatistics; CONVENTION-contrarian in metabolomics practice, where correct-then-test
dominates.]** The pack should offer this as the default for the univariate path.

### When it does harm — the citations that let the tool be credibly cautious

- **Nygaard, Rødland & Hovig (2016), *Biostatistics*** — ComBat on **unbalanced designs** can
  "inadvertently exaggerate the differences observed." Their reanalysis: ComBat produced **>1,000
  differentially expressed probesets where an appropriate mixed-model approach recovered 11.**
- ***BMC Bioinformatics* 21 (2020)** — ComBat on **randomly generated data with no true signal**
  produced "alarming numbers" of FDR- and Bonferroni-corrected false positives, **in both unbalanced
  *and balanced* designs.** The strongest single warning available; the pack should carry it.
- **Overcorrection by QC-based smoothers** — comparative work reports certain QC-based algorithms
  **significantly decreased replicate correlation**, i.e. removed real signal along with drift.
- **Circular validation** — "QC RSD improved from 28% to 9% after correction" is *not* evidence the
  analysis is sound, because QC-based correction is fit to minimize exactly that quantity. The honest
  metric is improvement in an independent quantity.

### Coaching

> *"All 24 of your treated samples were run in batch 1 and all 24 controls in batch 2. That means
> batch and treatment are the same variable, and no correction method — ComBat, QC-RLSC, SERRF, none
> of them — can tell them apart. Anything I 'correct' here would be guessing. I will not present a
> batch-corrected group comparison, and a reviewer should not accept one."*

> *"ComBat is the most cited batch-correction method and it works, but it also has the best-documented
> failure mode: a 2020 simulation in BMC Bioinformatics showed ComBat applied to purely random data
> produced large numbers of FDR-corrected false positives even in balanced designs, and Nygaard et al.
> found ComBat inflating a result from 11 genes to over 1,000 under an unbalanced design. My default
> for your univariate tests is to include batch as a covariate rather than modify your data."*

> *"Correction reduced QC RSD from 26% to 8%. That is not evidence your biology is real — QC-based
> correction is fit to make that exact number small. The meaningful checks are that your technical
> replicates got more correlated, not less."*

### Presentation

**Paired PCA panels** before/after, colored by batch (left) and by group (right) — the field standard
four-panel figure · **Run-order trajectory** for exemplar features with the fitted correction curve
overlaid, showing the smoother is not extrapolating wildly · **Variance-partition stacked bar**
before/after · **QC RSD paired CDF** · **Replicate correlation before/after** — the honest,
non-circular check.

---

## 06 · EDA and presentation

### The organizing principle the pack must teach

The field runs on a two-tier logic that most users do not articulate:

- **Tier 1 — unsupervised.** PCA, hierarchical clustering, correlation networks. *Honest* because they
  never see the group labels. Their job is quality assurance and hypothesis generation. **They cannot
  establish a group difference**, and separation in a PCA is not a result.
- **Tier 2 — supervised.** Univariate tests with FDR, validated PLS-DA/OPLS-DA, classifiers. They see
  the labels and can therefore be fooled. They require validation machinery to mean anything.

**The signature failure of the field is presenting a Tier-2 figure with Tier-1 credibility** — most
commonly a beautiful PLS-DA scores plot with no permutation test, or a heatmap of the top 50 features
selected by the very test being illustrated. **The pack should label every figure EXPLORATORY or
CONFIRMATORY, and refuse to let a confirmatory figure into the Results bundle without its validation
companion.**

### 06.1 · PCA scores plot — the field's trust anchor

PCA here does double duty. Its scientific job is exploratory structure-finding. Its **quality-control
job is more important and is what distinguishes this domain**: if the data are reproducible, the
pooled QCs should form a **tight cluster near the center**, well inside the biological spread. QCs
scattered, or drifting along PC1 in injection order, or separating by batch, are direct evidence the
run was not stable.

**Critical distinction the pack must not get wrong.** The **Hotelling's T² ellipse** is a *single*
ellipse over all samples defining the multivariate 95% region — an outlier boundary. **Group-wise 95%
confidence ellipses** are a completely different object describing where each group's mean and spread
lie. Papers routinely mislabel one as the other. Render them differently (T² = single dashed grey;
group ellipses = filled, group-colored) and label them explicitly.

**Publication-grade checklist:**

1. Axis labels give the component **and the % variance explained**: `PC1 (28.4%)`. Omitting these is
   the single most common defect and reviewers ask for it.
2. **Pooled QCs overlaid** in a distinct consistent color, never dropped. Their tight central cluster
   *is* part of the result.
3. **Aspect ratio proportional to variance explained** (or at minimum equal aspect) — stretching PC2
   to fill the panel visually exaggerates separation. Rarely done; doing it correctly is a quality
   signal.
4. **95% Hotelling's T² ellipse** drawn and labeled as such, distinguished from group ellipses.
5. Colorblind-safe group colors, identical across every figure in the manuscript.
6. Legend states **n per group**, the **normalization/scaling used**, and whether QCs were included in
   the fit or projected.
7. **A second panel colored by batch and/or injection order** — this is what proves the structure is
   biological and not technical.
8. Sample labels only for flagged outliers. Vector output. **No 3D PCA.**

> *"Your pooled QCs cluster tightly at the center and your two groups overlap substantially. That's a
> good result for data quality and an honest null for biology. Overlapping groups in a PCA doesn't
> mean there's no difference — PCA doesn't know your groups exist, and a real but modest difference
> will usually hide behind larger sources of variation like age, sex and BMI. It does mean the
> difference isn't the dominant signal in your data, and your manuscript should say that."*

> *"I will not show you a PLS-DA plot until we've done a PCA. If groups separate in supervised space
> but not at all in unsupervised space, that's not automatically wrong, but it's the exact pattern
> overfitting produces, and you'll need the permutation test to tell the difference."*

### 06.2 · Hierarchical clustering heatmap

Choices that change the picture and must be reported: **row scaling** (z-score per feature is
near-universal for visualization; without it the heatmap shows abundance, not pattern — and z-scoring
presumes you log-transformed first); **distance** (Euclidean by default; correlation distance is more
appropriate when co-variation pattern matters more than magnitude); **linkage** — and the R gotcha
the pack must get right: **`hclust(method="ward")` is not true Ward's linkage; `ward.D2` is.**

**The field's biggest visual sin.** A heatmap of *all* features is exploratory and honest. A heatmap
of *the top 50 features selected by the FDR test you are reporting*, showing a beautiful two-block
structure, is **circular** — the block structure is guaranteed by construction. It is legitimate as a
*display* of an established result and illegitimate as *evidence* for it. **Permit it, label it
honestly in the generated caption, and never present it as validation.**

Publication-grade: caption states row scaling, distance, linkage, and what was clustered · diverging
symmetric color scale centered at zero with the bar labeled "row z-score" · **annotation bars** above
columns for group, batch, sex — this is how a reader sees that a cluster is a batch, not a phenotype ·
feature labels only if annotated to at least MSI level 2 · state whether features were pre-selected ·
bootstrap stability values (`pvclust` AU/BP) if claiming the clusters are real.

### 06.3 · Volcano plot

Publication-grade: x = **log2 fold change** with direction stated unambiguously · y = **−log10 of the
FDR-adjusted q**, or −log10(p) with the cut line drawn at the p corresponding to q = 0.05 and the
caption saying which. **Plotting raw p-values with a line at p = 0.05 on a 3,000-feature untargeted
dataset is an anti-pattern and would be flagged in review.** · threshold lines annotated numerically ·
counts of significant up/down printed on the panel · |log2FC| > 1 is common convention, > 0.58
(1.5-fold) common in metabolomics **[CONVENTION — arbitrary, justify biologically]** · **labels only
for annotated compounds with a stated MSI level.**

**Critical metabolomics-specific caveat.** The fold change must be computed where fold change is
meaningful. **After autoscaling, "fold change" is a fold change in z-units and is meaningless.** After
quantile normalization, magnitudes have been forced. **Compute FC from normalized-but-not-scaled data
and say so.** Getting this wrong is a subtle, real, embarrassing error.

> *"You have 3,000 features. At an uncorrected p < 0.05 you'd expect about 150 'significant' hits by
> chance alone, and you have 187 — which is to say, your uncorrected result is consistent with nothing
> happening. After Benjamini–Hochberg, 12 features survive at q < 0.05. Those 12 are your result."*

> *"Your features aren't independent — adducts and isotopes of the same molecule appear as several
> rows. Benjamini–Hochberg tolerates that positive dependence, so your q-values are fine, but 'we
> identified 44 significant metabolites' is not: those 44 features correspond to about 15 distinct
> compounds. I'll report both numbers."*

### 06.4 · Box plots per metabolite

**Show individual points** (jitter or beeswarm) over the box — with n < 30 per group a box plot alone
conceals the data · **never a bar chart of mean ± SEM** (the "dynamite plunger" is actively
criticized) · y-axis labeled with **what the values actually are** ("normalized peak area (PQN,
log2)"), not bare "relative abundance" · **report the q-value, not just asterisks**, or define
asterisks *and state they are FDR-adjusted* · connect paired points for repeated measures · **state
the MSI identification level** in the panel title.

> *"You're calling this metabolite Citrate. Your annotation is MSI level 2 — matched to a spectral
> library, but not confirmed against an authentic standard run in your own lab. That's a perfectly
> publishable level of confidence, but the figure and the text need to say 'putatively annotated'
> rather than 'identified'."*

### 06.5 · PLS-DA / OPLS-DA scores, S-plot, VIP plot

The field's most recognizable figures and its most dangerous.

**Scores plot — CONFIRMATORY, and inadmissible without validation.** Publication-grade requires, on or
beside the panel: **R²X, R²Y, Q², number of latent variables, the CV scheme, the permutation p-value
and the number of permutations.** A scores plot without these is not a result. Westerhuis et al.
advocate explicitly **against using PLS-DA score plots to infer class differences.**

**Permutation plot — mandatory companion.** x = correlation between permuted and original y; y = R²Y
and Q² per permuted model; the two regression lines; the original model's values at x = 1. A valid
model shows the original clearly to the right of and above the permuted cloud, with the Q² regression
line intercepting the y-axis **below zero**. **[CONVENTION, near-SETTLED as an expectation]**

**S-plot (OPLS-DA).** x = covariance p[1], y = correlation p(corr)[1]. Wing features — high covariance
*and* high |correlation| — are candidates; classic threshold |p(corr)| ≥ 0.5, 0.8 for stringency
(Wiklund et al. 2008). **[CONVENTION]** State the cutoff, label only annotated compounds, highlight
points that *also* pass univariate FDR. **An S-plot from an unvalidated model is a picture of noise.**

**VIP plot.** VIP > 1 is near-universal, justified because VIP² averages to 1 across features.
**[CONVENTION, and a heuristic ranking, not a hypothesis test.]** Publication-grade: **bootstrap or
jackknife confidence intervals** (a VIP of 1.4 whose CI spans 0.6–2.2 is not evidence), sorted
descending, annotated features only, cross-tabulated against univariate q-values. The strongest
presentation is a table: metabolite | MSI level | VIP (95% CI) | log2FC | q-value.

> *"PLS-DA will separate your groups. It separates random groups — when you have at least twice as
> many features as samples, it can find a plane that perfectly splits randomly assigned labels by
> chance alone. So the scores plot on its own carries no information about whether your groups differ.
> What carries information is the permutation test."*

### 06.6 · Composition plots, correlation networks, and the rest

**Relative-abundance / composition plots** (lipidomics, pathway rollups): state the denominator
explicitly — **mol% of total lipid vs mol% within class are different numbers and are routinely
confused.** State the internal standards per class. Coaching: *"These are proportions, so they are
necessarily linked. Unless you have an absolute normalizer, you cannot distinguish 'PC went up' from
'everything else went down'."*

**Correlation networks.** Plain correlation networks conflate direct and indirect relationships.
**Gaussian graphical models use partial correlation, conditioning each pair on all other metabolites,
which removes the indirect edges** and recovers pathway reactions substantially better (Krumsiek et
al. 2011, *BMC Syst Biol* 5:21). **WGCNA** is the module-finding alternative but *assumes a priori a
scale-free topology* — and it has not been established that metabolite association networks are
scale-free. **Anti-pattern:** presenting a full-correlation network as inferred metabolic pathway
structure.

**Also supported:** scree/cumulative variance · PCA loadings biplot (warn that loadings from autoscaled
data rank features by *pattern*, not abundance) · **ROC for a candidate panel — CONFIRMATORY**,
requiring cross-validated or externally validated AUC with a CI and the number of features stated (a
resubstitution AUC of 0.98 from a 20-feature panel on 30 samples is meaningless) · **pathway enrichment
bubble plot** with the pervasive **annotation bias** flagged: enrichment is computed over metabolites
you could *name*, biased toward well-studied, abundant, commercially-available compounds.

### 06.7 · Cross-cutting publication-grade rules

Consistent group colors and shapes across every figure; colorblind-safe (Okabe-Ito or viridis) · QCs
always in the same distinct color, always shown when they exist · n in every legend · every caption
names the normalization, transformation and scaling · every confirmatory figure names the test and the
correction · vector output, fonts ≥7 pt at final size · every named metabolite carries an MSI level ·
**no 3D PCA** · no truncated axes.

---

## 07 · Is untargeted metabolomics compositional?

**The pack's answer: partially, by construction rather than by nature, and the right response is a
sensitivity analysis rather than a doctrine. [GENUINELY DISPUTED] — confident wrongness in either
direction would embarrass the tool.**

**For.** MS detector response and the ion source have finite capacity; ion suppression means one
compound's abundance genuinely affects another's measured signal. **Any normalization that divides by
a total — TIC, sum, MSTUS, mol% — imposes closure and makes the data compositional by construction.**
On a simplex, Euclidean distance, Pearson correlation and standard PCA are not well-defined, and
spurious negative correlations are induced.

**Against.** Untargeted metabolomics observes a **small, biased, technology-dependent subset**;
compositional theory concerns a *closed whole*. Features are on **incommensurable scales** —
ionization efficiency varies by orders of magnitude, so the ratio of feature A's intensity to feature
B's is not a ratio of amounts, and CLR's geometric mean is taken across exactly those incommensurable
quantities. **Zeros and left-censoring are pervasive and CLR cannot tolerate them** — every CLR
analysis needs zero replacement, and that choice then propagates into the geometric mean of *every*
sample, contaminating all features. With absolute quantification the data are genuinely absolute.

**The empirical evidence.** *"To Impute or Not To Impute in Untargeted Metabolomics — That is the
Compositional Question"* (*JASMS* 2025) states untargeted metabolomics "can have both compositional
and noncompositional character," and reports that PCA of raw data explained **79.92%** of variance in
the first two PCs whereas **TIC-CLR-transformed data explained only 68.99%**, with the CLR PCA
**failing to resolve sample clustering at the same resolution**. A real, citable negative result for
reflexive CLR use. Counterweight: in multiomic time-series work, CLR-transformed analysis revealed
novel relationships and stronger associations (*NAR Genom Bioinform* 2020) — CLR is not universally
worse. **Where CoDA *is* settled: microbiome relative-abundance data. Do not import that consensus
into metabolomics by analogy** — the situations differ in exactly the ways above.

**Pack behavior.** Default: do not CLR-transform; use PQN/median-fold plus log. **Always warn about
closure whenever a sum-based normalizer is chosen**, and phrase results from such analyses in relative
language. Offer CLR/ILR as an explicit sensitivity analysis with a documented zero-replacement
strategy (multiplicative or Bayesian-multiplicative, not half-min). **Escalate the closure warning to
high priority when** the normalizer is sum/TIC/mol%, **or** a small number of features carry a large
fraction of total signal, **or** a treatment plausibly causes a large global shift (a drug, a
xenobiotic, glucose in diabetes, lipid infusion) — in that last case, say plainly that group
differences in every other feature may be an artifact of the normalizer.

---

## 08 · Feature selection and modeling

### Diagnostic

**p vs n ratio** (above ~2, PLS-DA can perfectly separate random labels) · **effective sample size**
accounting for repeated measures · class balance and minimum group size · **confounder screen**
(association of age, sex, BMI, batch, site, fasting, medication, storage with the outcome and with
PC1–PC3) · **independence check** (repeated subjects, technical replicates) · **leakage audit** — did
any filtering, normalization, imputation or selection step use the outcome labels or the full dataset
before cross-validation?

### The univariate path — the reviewer's workhorse

**[SETTLED]:** per-feature testing with multiple-testing correction is expected, and its absence is a
fatal flaw in review. Two-group → Welch's t on log values or Mann–Whitney; covariate-adjusted →
linear model on log values; repeated measures → linear mixed model with subject random effect.
**Benjamini–Hochberg at q < 0.05 is the field standard [SETTLED]**; Bonferroni is over-conservative
given feature correlation. **Report effect sizes, not just p** — a q of 0.03 with a 6% mean difference
is a different claim from a q of 0.03 with a 3-fold difference. **Redundancy caveat:** the count of
significant *features* overstates the count of significant *compounds*. Saccenti et al. 2014,
*Metabolomics* 10:361 is the right citation for univariate and multivariate answering different
questions.

### The multivariate path — PLS-DA and its critique

**The overfitting problem, concretely.** When features ≥ 2× samples, PLS-DA readily finds a hyperplane
that perfectly separates **randomly assigned** labels. This is not a fringe caveat; it is the central
methodological criticism of the field's favorite method.

**Required validation, in order of importance:**

1. **Permutation testing** — permute labels, refit the entire pipeline, recompute R²Y/Q², repeat.
   **≥1,000 permutations** is the modern expectation (200 is common in older SIMCA practice and is now
   considered thin). **[SETTLED that it is required; the count is CONVENTION.]**
2. **The permutation must include every supervised step.** If features were selected using the labels,
   selection must be redone inside every permutation, or the permutation test is itself invalid.
3. **Double cross-validation (2CV).** Westerhuis et al. 2008, *Metabolomics* 4:81 is canonical.
4. **Nested CV whenever anything is tuned or selected.** Ambroise & McLachlan 2002, *PNAS* 99:6562
   established selection bias from gene selection outside the resampling loop; Cawley & Talbot 2010,
   *JMLR* 11:2079 generalized it: **model selection must be treated as an integral part of model
   fitting and performed afresh inside every fold.** **[SETTLED in statistics; routinely violated
   here.]**

**Diagnostic-statistic caveats.** **Q² > 0.5 is a rule of thumb, not a test.** Triba et al.
(*Mol BioSyst* 2015) showed that in metabolomics, K-fold CV parameters depend strongly on which
individuals land in which validation subset, and **a simple permutation of dataset rows can flip the
conclusion about model significance.** **[DISPUTED — the pack must not gate on Q².]** Szymańska et al.
2012, *Metabolomics* 8:3 found NMC = 0 or Q² = 0.99 attainable purely by chance through a lucky split,
and that **NMC and AUROC are more efficient and reliable diagnostic statistics than Q².**

**OPLS-DA specifically.** It rotates the PLS solution so between-class variation concentrates in one
predictive component. **It improves interpretability; it does not improve predictive performance and
it does not reduce overfitting** — the predictive subspace is the same. **[SETTLED among
chemometricians; widely misunderstood by practitioners. This is a prime confidently-wrong hazard: the
pack must not imply OPLS-DA fixes overfitting.]** Kjeldahl & Bro 2010, *J Chemometrics* 24:558.

**Alternatives worth offering:** sparse PLS-DA (L1 selection embedded) · elastic net / LASSO / ridge
(handles p ≫ n natively; elastic net had the highest predictive power in one omics comparison) ·
random forest (Chen et al. 2013; OOB error gives a quality indication automatically, but importance is
biased toward correlated/high-cardinality features) · **and the honest headline: no single winner.**
Trainor et al. 2017 compared PLS-DA, sPLS-DA, RF, SVM, ANN, kNN and naive Bayes and found no universal
winner. **[SETTLED that there is no settled best classifier.]** Also useful ammunition: Ruiz-Perez et
al. 2020, *"So you think you can PLS-DA?"* found unsupervised PCA remarkably effective as a feature
selector, **in some cases outperforming PLS-DA** despite PLS-DA having access to the labels.

### Sample size — [DISPUTED, and honestly so]

Conventional power calculation requires an effect size that hypothesis-free untargeted work does not
have. Practical guidance, clearly labeled as guidance: controlled interventions with large effects
have identified biomarkers with **4–20 subjects** per arm; **human observational cohorts** need
substantially more — treat anything under ~20/group as hypothesis-generating only; anything claimed
as a "biomarker" requires an independent validation cohort. **The pack should compute a post-hoc
*detectable effect size* curve** — given n, α after FDR, and observed per-feature CV, what fold change
was detectable at 80% power? — rather than a post-hoc power figure, which is statistically
meaningless.

### Coaching

> *"You have 2,540 features and 38 samples. At that ratio a PLS-DA model can separate your groups
> perfectly even if you shuffle the labels at random. I'm going to run the model with 1,000 label
> permutations and show you where your real model falls in that distribution. If it's inside the
> permuted cloud, we stop and say so."*

> *"You've selected the top 200 features by t-test and now want to cross-validate a classifier on
> them. That selection used your labels, so the folds aren't independent of it and accuracy will come
> out optimistically high — this is the selection bias Ambroise and McLachlan documented in PNAS in
> 2002. I'll move the selection inside the loop. Your accuracy estimate will drop. The lower number is
> the true one."*

> *"With n = 12 per group and a typical per-metabolite CV of 25%, you could reliably detect roughly a
> 1.6-fold change after FDR correction. Anything smaller, this study cannot see — worth stating in
> your limitations rather than leaving a reviewer to compute it."*

### Presentation

**Permutation plot** (mandatory companion) · **model summary table**: components, R²X, R²Y, Q², CV
scheme, permutation n and p, NMC, AUROC with CI · **nested-CV performance distribution** (boxplot of
outer-fold AUCs, not a single number) · **★ feature evidence table**: metabolite | MSI level | m/z | RT
| log2FC | raw p | BH q | VIP (95% CI) | **selection frequency across CV folds** | direction —
selection frequency is the honest stability measure and almost nobody reports it ·
**univariate–multivariate concordance plot**: VIP vs −log10(q) with the VIP = 1 and q = 0.05 lines;
the upper-right quadrant holds the defensible candidates.

---

## 09 · Reporting standards

| Standard | Scope | Status |
|---|---|---|
| **MSI / CAWG — Sumner et al. 2007, *Metabolomics* 3:211** | sample prep, analysis, QC, identification, pre-processing; the **4-level identification scheme** | **[SETTLED]** — the reference every reviewer knows |
| **MSI CIMR** | umbrella minimum-information checklist | — |
| **mQACC 2022, *Metabolomics* 18:70** | QA/QC reporting for untargeted phenotyping, MS and NMR | **[SETTLED as the current best QC-reporting reference]** |
| **QComics 2024, *Anal Chem* 96:1064** | implementable QC workflow with acceptance criteria | **[CONVENTION, rising]** |
| **Broadhurst et al. 2018, *Metabolomics* 14:72** | system suitability, QC sample use, D-ratio | **[SETTLED as the QC-design reference]** |
| **Lipidomics Minimal Reporting Checklist**, *J Lipid Res* 2024 | lipidomics reporting + shorthand nomenclature | **[SETTLED for lipidomics]** |
| **MIAPE (HUPO-PSI)**; ProteomeXchange/PRIDE | proteomics minimum information and deposition | **[SETTLED for proteomics]** |
| **MetaboLights / Metabolomics Workbench** | raw-data deposition | **[SETTLED as an expectation; enforcement is weak]** |

**The MSI 4-level identification scheme** — the pack must attach a level to every named compound:

- **Level 1 — Identified:** matched to an **authentic chemical standard run under identical conditions
  in the same laboratory**, on ≥2 orthogonal properties.
- **Level 2 — Putatively annotated:** matched to spectral library or literature data, no in-house
  standard.
- **Level 3 — Putatively characterized compound class.**
- **Level 4 — Unknown:** unidentified but reproducibly detected and quantifiable.

Reviewers increasingly reject papers that call level-2 annotations "identified." **Policing that
language is a place the pack adds genuine value.**

**Two honest facts to tell the user.** Adherence to MSI standards in published repositories **ranged
from 0% to 97% depending on the item, and no reporting standard was complied with in every study**
(Spicer et al. 2017, *Sci Data* 4:170137). Journals **promote data sharing in metabolomics but do not
enforce it** (Spicer et al. 2018). And the 2007 standards are widely acknowledged to need revision —
present them as the operative baseline, not a finished consensus.

### The methods-section checklist the pack auto-generates

**Design:** n per group and power justification or an explicit hypothesis-generating statement ·
inclusion/exclusion, matching, measured confounders · collection, tube type, time to processing,
storage temperature and duration, freeze–thaw cycles · **randomization of preparation and injection
order, explicitly stated — and if not randomized, say so.**

**Analytical:** instrument, column, gradient, ionization mode, mass range, resolution, MS/MS strategy ·
extraction protocol and internal standards · **QC design** — what the pooled QC was made from, how
many, injection interval, conditioning injections; blank type (process vs solvent) and number ·
batches, samples per batch, batch composition with respect to study groups.

**Processing:** peak-picking software **and version** with key parameters · **the filtering chain, in
order, with every threshold and the features remaining after each step** · missing-value handling with
the *mechanism evidence* for the choice and whether a sensitivity analysis was run · **normalization,
transformation and scaling named separately** (with PQN, state the reference) · drift/batch correction
method and **QC RSD / D-ratio before and after.**

**Statistics:** tests, sidedness, pairing, covariates, random effects · **multiple-testing method and
threshold** · effect sizes and CIs · multivariate model type, components, scaling, R²X/R²Y/Q², **CV
scheme, permutation count and p, and confirmation that selection was inside the CV loop** · software
and **versions.**

**Identification:** **MSI level per reported metabolite** with the evidence (ppm error, RT match to
in-house standard, MS/MS score, library and version) · adduct/isotope handling and how redundant
features were collapsed.

**Availability:** repository and accession including **raw data**, not just the processed matrix · and
a statement of what is *not* deposited and why.

### Presentation

Auto-generated methods paragraph with every number populated from the session · **Table S1 — filtering
waterfall** · **Table S2 — QC performance** (number of QCs, median QC RSD before/after, % features
with RSD < 20/30%, median D-ratio, RT RSD, blank-filter statistics) · **Table S3 — identification
evidence** · **Table S4 — full results** for every feature, so the reader is not limited to the
significant subset · **★ reporting-compliance scorecard** — which MSI/mQACC items were satisfied,
which are missing. This is the feature most likely to make researchers recommend the tool.

---

## 10 · Anti-pattern registry

Each should be a named detector with a specific warning string.

**Structural.** Transposed matrix read silently · zeros treated as measurements, log(0) → −Inf
silently dropped · double log transformation · proteomics contaminant/reverse rows left in · repeated
measures treated as independent · adducts and isotopes counted as distinct metabolites.

**Filtering.** Filtering by a group-difference statistic before testing (circular) · computing QC RSD
after imputation · reporting only post-correction QC RSD as evidence of quality (circular) · applying
untargeted RSD filtering to a validated targeted panel · the plain 80% rule deleting group-specific
metabolites · MetaboAnalyst's 40% IQR filter applied without reporting it.

**Missing data.** Half-min without checking the mechanism · kNN/RF on clearly left-censored data ·
imputing a feature present in one group and absent in the other, then reporting a fold change ·
imputing before filtering · imputing across the whole dataset before CV (leakage) · proteomics:
imputing after protein rollup instead of at peptide level.

**Normalization.** Scaling before sample-normalizing · TIC when one compound dominates · creatinine in
a cohort with renal impairment · **computing fold changes from autoscaled or quantile-normalized
data** · normalizing QCs and biological samples with separately estimated parameters · trying multiple
normalizations and keeping the best p-values · "the data were normalized" without naming all three
operations.

**Batch.** Correcting a confounded design · QC-based correction with too few or non-interspersed QCs ·
ComBat on an unbalanced design · correcting then testing without accounting for the correction ·
correcting data that has no batch effect.

**Figures.** PCA axes without % variance · QCs omitted from the PCA · **T² ellipse labeled as a group
confidence ellipse** · 3D PCA · PLS-DA without permutation testing · PLS-DA read as if it were PCA ·
heatmap of test-selected features presented as independent validation · `hclust(method="ward")` ·
volcano with raw p on thousands of features · bar chart of mean ± SEM · asterisks without the test or
correction · full-correlation network read as pathway structure · unannotated `M###T###` labels
cluttering a figure · resubstitution AUC presented as biomarker performance.

**Modeling and reporting.** Selection outside the CV loop · Q² > 0.5 as a pass/fail gate · believing
OPLS-DA is less prone to overfitting · permutation that keeps a label-derived feature set fixed ·
post-hoc power analysis · calling MSI level-2 annotations "identified" · "biomarker" claimed from a
discovery cohort with no validation · pathway enrichment without the background set stated.

---

## 11 · Where confident automation would embarrass us

Ranked. Places the pack must hedge explicitly rather than assert.

1. **The QC RSD threshold.** No consensus; the scoping review says so outright. Assert a *default with
   a stated rationale*, never a rule.
2. **The blank-ratio fold change.** 3×/5×/10×/20× all in use. Genuinely unsettled.
3. **Imputation method.** The metabolomics benchmark says QRILC for MNAR / RF for MAR; a major
   proteomics benchmark says RF is robust *even* under MNAR. **These conflict. Present both.**
4. **Pareto vs autoscaling.** Pareto is what the field does; van den Berg's evidence favors
   autoscaling. Do not present Pareto as correct.
5. **Compositionality.** Both "always CLR" and "compositionality is irrelevant" are wrong. The JASMS
   2025 result must not be suppressed; the CoDA argument is also real.
6. **Batch correction.** ComBat is both standard and demonstrably capable of manufacturing
   FDR-corrected false positives on pure noise. Never present it as safe.
7. **OPLS-DA's supposed superiority.** It is a rotation. Saying it "reduces overfitting" is a technical
   error a chemometrician would catch instantly.
8. **Q² > 0.5.** A rule of thumb directly attacked by Triba et al. Do not gate on it.
9. **Sample size.** No valid generic power calculation exists for untargeted work. Frame any specific
   claim as detectable-effect-size guidance with assumptions shown.
10. **The 80% rule's exact form.** Plain vs modified matters; state which and why.
11. **Hotelling's T² vs group confidence ellipses.** Mixing these up in a rendered figure would be a
    visible, elementary error.
12. **Any claim about a specific software default** (MetaboAnalyst's 40% IQR filter, `pmp`'s blank
    fold change, structToolbox's D-ratio). These change between versions. **[verify-at-build]** and,
    better, read them from the user's installed version rather than hard-coding.
13. **"Metabolomics is compositional like microbiome data."** The analogy is false in specific,
    articulable ways. Do not import the microbiome consensus.

---

## 12 · The canonical pipeline order — the pack's default spine

```
1  Import → orientation, roles, run order, metadata, zero/NA semantics
2  Design audit → group×batch crosstab, randomization check, repeated-measures check
3  Blank / background filter
4  QC detection-rate filter
5  Sample-level outlier screen (TIC, feature count, Hotelling T2)
6  [optional] Within-batch drift correction (QC-RSC / QC-RLSC / SERRF)
7  [optional] Between-batch correction (ComBat) — or defer batch to the model
8  QC RSD + D-ratio filter        (report pre- and post-correction values)
9  Missingness filter (modified 80% rule)
10 Missingness mechanism diagnosis
11 Imputation (QRILC/GSimp if censored; RF/kNN if random) + sensitivity duplicate
12 Sample-level normalization (PQN default for biofluids; IS/biomass for tissue)
13 Transformation (log2, or glog/VSN)
14 Feature-level scaling (Pareto default, autoscale offered) — ONLY for multivariate;
   keep an unscaled, normalized copy for fold changes and box plots
15 EDA: PCA (+QC overlay, +batch panel), clustering heatmap, correlation structure
16 Univariate: model with covariates/random effects, BH FDR, effect sizes
17 Multivariate: nested CV + >=1000 permutations; selection inside the CV loop
18 Concordance: VIP x q-value evidence table with CV selection frequency
19 Figures: exploratory bundle vs confirmatory bundle, labeled as such
20 Reporting: methods paragraph, QC table, identification table, waterfall, scorecard
```

**Two forks the pack must expose rather than hide:** step 7 — correct the data vs model the batch; and
step 14 — scaled matrix for multivariate, unscaled for fold changes. Both are routinely botched.
