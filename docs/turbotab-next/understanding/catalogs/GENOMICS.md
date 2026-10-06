# Genomics threads (45)

Generated from catalogs/genomics.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- genomics-genes-are-the-outcomes: 'You picked diet arm as the outcome. In a feeding trial the question your design can answer is which genes the diet changed: 212 at 5% FDR. An AUC would tell you only that the arms are separable.' This is the clearest case of a question AutoML cannot ask, and it is specific to nutrition.
- genomics-batch-aligned-with-case: random folds give AUC 0.91; fitting ComBat inside each fold and holding out one batch at a time gives 0.61, and the 0.30 gap is labeled as learned from the batch. When batches are balanced, as in the shipped fixture, the app says instead: 'Batch is 97% of PC1, but every batch holds 10 cases and 10 controls, so it costs you power, not truth.'
- genomics-depth-tracks-case: 'Library size alone predicts case status with AUC 0.92; your raw-count model scored 0.94, normalized 0.61.' AutoML never prints a depth-only baseline.
- genomics-sample-sheet-alignment + genomics-sex-from-expression: 'Three samples recorded as female express RPS4Y1 at male levels and no XIST. If their sex labels were swapped, their case labels may be someone else's too.' Or: R rewrote 38 IDs, so an exact join would have silently dropped 22 samples.
- genomics-features-preselected-on-outcome + genomics-outcome-defined-from-features: 'All 200 genes differ at p < 0.001, where random data would give 0.2. Were they picked using the outcome before the file was made?' Or: 'five genes predict your subtype perfectly, and they are five of the genes that defined it.'
- genomics-sample-quality-drives-components: 'PC1 (34%) is RNA integrity, not disease: R² 0.71 with RIN, and the cases' tissue waited longer before freezing.'
- genomics-paired-or-repeated-samples: 'Your 48 samples are 24 people before and after the diet, and 70% of each gene's spread is between people. The comparison has to be within each person.'
- genomics-new-site-new-platform: 'Pooled AUC 0.89; leave one study out gives 0.71, 0.64 and 0.83. Your z-scoring step needs a batch of new samples, so a single patient's sample cannot be scored.'
- genomics-exposure-written-in-the-profile: 'Your top CpG for vegetable intake is AHRR cg05575921, the strongest smoking mark in blood, and six self-reported never-smokers have smoker-level values there.'
- genomics-what-the-numbers-are: 'Each sample's back-transformed values sum to one million within 2%: these are log2(CPM+1).' The count model and a second log are greyed out before the researcher picks anything.

## Journey load

Every thread is now gated, given a tier, and folded into four shared cards:
- **'What drives your leading components'**: one Angles view over a single PC x covariate R² matrix. It covers batch, unrecorded structure, sample quality, depth, draw conditions, mixed tissue and cell mix, and asks at most 2 questions per journey.
- **'Who are your samples'**: one Flow card for orientation, sample-sheet alignment, excluded libraries, sex check, near-identical samples, outliers and pairing.
- **'Where did these come from'**: the provenance card for preselected features, an outcome defined from the features, and sampling after diagnosis.
- **The feature waterfall**: silent and stated rules only. It covers the expression or array detection filter, probe collapse, identifier repairs and p/n.

Expected load by journey:
- **Shipped fixture (genomics_expression, 60 samples, balanced batches, gene_0001 names): 2–3 asks, 0–1 surfaced, about 5 stated.**
  - Asks: confirm the batch reading in one click, the outlier rule (GS031, z = -3.8) and the grain question the fixture requires.
  - Surfaced: depth, only under prediction and only if the model's score lands within 0.05 of the depth-only AUC of 0.66.
  - Stated: batch is balanced; cases are 1.17-fold deeper (AUC 0.66); the filter keeps 252 of 495 genes; 495 genes on 60 samples; the sex check abstains.
  - Unrecorded structure, sample quality and cell mix stay silent.
- **Typical bulk RNA-seq (blood, 60–200 samples, sheet with batch/RIN/sex): 3–5 asks, 2–4 surfaced.**
  - Asks: batch reading and imbalance, RIN meaning, cell mix, outlier rule, and sex discordance only if it occurs.
  - Surfaced: depth, p-value histogram or λ after the lock (inference), instability (prediction, when a list is exported), score noise.
- **Under prediction:** add the deployment question (new-site), and the provenance card only on its cues.
- **Feeding trial:** the estimand and pairing questions dominate. That is 2 blocking asks before anything else, then the bulk set above.
- **Genotype journey: 2–3 asks** (PRS provenance, gene x diet role, and allele harmonization only when merging). QC, relatedness and imputation are stated; ancestry is surfaced.
- **Methylation and qPCR journeys:** 1 ask each (array/tissue, or Ct/reference/cycle maximum), plus cell mix for blood.

Most asks land at the opening stage. Explanation-stage threads (modules, instability) are never asked. One compute note: the permutation null runs only on request and with sequential stopping, and should be scheduled because the machine sits beside a bedroom.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| genomics-genes-are-the-outcomes | asked-blocking | design | In a diet or intervention study the genes are the outcomes and the diet is the exposure, not the other way round |
| genomics-batch-aligned-with-case | asked-blocking | design | Samples split by processing batch on the leading components, and batch lines up with case status |
| genomics-depth-tracks-case | surfaced | design | Cases were sequenced deeper than controls, so depth alone can predict case status |
| genomics-what-the-numbers-are | silent | measurement | The values show the processing they have already had (raw counts, estimated counts, CPM/TPM, FPKM, VST, log intensity), and that closes some models |
| genomics-sample-quality-drives-components | asked | measurement | A recorded technical covariate (RIN, ischemic or post-mortem time, %mito, 3′ bias, mapping rate, array position, processing order) drives a leading component |
| genomics-unrecorded-structure | asked | structure | Samples split on a leading component by something no column names (a lane, an extraction day, a tissue) |
| genomics-sample-sheet-alignment | asked-blocking | structure | The expression matrix and the sample sheet do not line up one-to-one (renamed, reordered or missing sample IDs) |
| genomics-features-preselected-on-outcome | asked | predictor-outcome | The uploaded genes were already chosen using these samples' outcome |
| genomics-outcome-defined-from-features | asked | predictor-outcome | The outcome was derived from the same expression data (clusters, subtypes, a signature-defined 'responder'), so testing or predicting it from those features is circular |
| genomics-sex-from-expression | asked | measurement | Recorded sex disagrees with sex-chromosome expression (a sample swap or mislabel), or sex genes top a sex-imbalanced comparison |
| genomics-paired-or-repeated-samples | asked | structure | Several samples come from the same person (before and after a diet, tumour and normal, several tissues or time points), so the design is paired |
| genomics-outlier-sample | asked | predictors | One or two samples are unlike all the others (a failed library, degraded or contaminated RNA, or a mislabel) |
| genomics-near-identical-samples | asked | structure | Pairs of samples are near-identical: technical replicates, a duplicated upload, or the same person |
| genomics-cells-are-not-replicates | asked-blocking | structure | The rows are cells, many per person, so the effective n is the number of donors |
| genomics-matrix-orientation | asked-blocking | structure | Genes are in rows and samples in columns (the usual GEO or featureCounts export), so every row-wise step would treat genes as people |
| genomics-gene-identifiers-damaged | asked-blocking | measurement | Gene identifiers were turned into dates by a spreadsheet, carry version suffixes, are duplicated, mix vocabularies, or include control features |
| genomics-cell-mix-drives-signal | asked | predictors | Differences in the mix of cell types, not regulation within cells, drive the signal |
| genomics-draw-conditions | asked | time | When and how the sample was drawn (time of day, fasting or fed, season) shapes expression and may differ between groups |
| genomics-mixed-tissue-types | asked | design | Samples come from different tissues or sample types (whole blood vs PBMC, tumour vs adjacent normal, biopsy sites), and the mix differs by group |
| genomics-excluded-libraries-by-group | stated | design | Libraries that failed or were dropped before upload are not spread evenly across cases and controls |
| genomics-new-site-new-platform | asked | design | The model will meet samples from another lab, platform or study, perhaps one at a time, and the training data already hold more than one |
| genomics-normalized-across-all-samples | stated | measurement | The matrix was quantile-normalized, RMA-processed or z-scored across all samples before upload, test samples included |
| genomics-few-transcripts-take-the-reads | stated | measurement | A few transcripts (globin, rRNA, mitochondrial) take most of the reads, or most genes shift one way, so depth-only scaling misleads |
| genomics-coexpression-modules | stated | predictors | Genes move in large correlated modules, so thousands of columns carry far fewer independent signals and importance splits across a module |
| genomics-signature-instability | surfaced | predictor-outcome | The selected gene list changes from refit to refit |
| genomics-pvalue-distribution | surfaced | predictor-outcome | The feature-wise p-values are inflated or misshapen, which says the model or the structure is wrong |
| genomics-qpcr-ct-values | asked | measurement | qPCR Ct values run backwards, and 'Undetermined' or the cycle maximum is a non-detect, not a number |
| genomics-methylation-beta | asked | measurement | Methylation beta values are bounded proportions whose variance depends on their level, and blood methylation tracks the cell mix |
| genomics-ancestry-structure | surfaced | structure | Genetic ancestry structures the samples and lines up with case status or with the diet exposure |
| genomics-genotype-qc | stated | measurement | The columns are genotypes coded 0/1/2, and some samples or variants fail basic genotyping QC |
| genomics-relatedness | stated | structure | Participants are related (siblings, parent-child, duplicates), so relatives share alleles and kitchens across the split |
| genomics-genotype-shapes-the-diet | asked | predictors | A genotype predicts the dietary exposure (lactase persistence and milk, ALDH2 and alcohol), so gene-diet questions carry gene-environment correlation and sparse cells |
| genomics-polygenic-score-provenance | asked | predictors | A polygenic score column whose weights may have been learned on these participants, or in another ancestry |
| genomics-allele-harmonization | asked | measurement | External effect sizes or polygenic weights must use the same effect allele and strand as these genotypes |
| genomics-imputed-dosages | stated | measurement | Genotypes are imputed dosages (continuous 0-2) whose imputation quality varies by variant |
| genomics-many-probes-one-gene | stated | measurement | Several microarray probes map to one gene, and some probes map to several |
| genomics-array-detection-floor | silent | measurement | On arrays, many probes sit at background and a few at the scanner ceiling, so the floor and the ceiling are not values like the rest |
| genomics-low-expression-filter | silent | predictors | Most genes barely register, and an outcome-blind filter decides how many are really tested |
| genomics-mean-variance-scaling | silent | predictors | A feature's variance depends on its mean; on counts the settled scale already handles it, and a guard checks that filtering did |
| genomics-more-genes-than-samples | silent | structure | With far more features than samples, the data cannot pin down a unique model |
| genomics-enrichment-background | silent | design | A top-feature list read by pathway needs, as its background, the features that could have been selected |
| genomics-exposure-written-in-the-profile | asked | measurement | The profile records exposures the sheet does not (smoking), or contradicts the self-report |
| shared-sampled-after-diagnosis | asked | time | Cases' samples were taken after diagnosis or treatment, or stored longer, than controls', so the profile records the disease's consequences |
| shared-score-within-noise | stated | outcome | With 40 samples or 8 events, the cross-validated score's own spread is as wide as the claims made from it |
| genomics-microbiome-composition † |  | measurement | The table is a microbiome: compositional counts or relative abundances, mostly zeros, with blanks and depth |
