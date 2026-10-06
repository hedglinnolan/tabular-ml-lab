# Metabolomics threads (43)

Generated from catalogs/metabolomics.json. Tier: asked, stated, surfaced (shown when its detector fires) or silent. Threads marked † were added by the completeness critic.

## The moments that matter most

- metab-run-order-aligned-with-outcome: 'Injection order alone predicts case status at AUC 0.81; your model's 0.86 is 0.05 above the instrument.' The shortcut is in every random fold, so no score can see it. On metabolomics_untargeted it correctly stays silent (order vs responder AUC 0.545).
- metab-treatment-marker (with metab-outcome-defined-by-a-feature): 'Your top feature, m/z 130.109, is metformin: the model found the prescription. Without it, 0.71, not 0.97.' Or: 'diabetes here is defined by glucose measured at the same visit, and glucose [M+Na]+ is your top feature.' A score rewards keeping both.
- metab-wide-noise-ceiling: 'Give these 72 people random labels with the same 39/33 split, and the best of 392 features still reaches AUC ~0.69. Your strongest, mz_0094, reaches 0.70.' Analytically checked on the fixture.
- metab-sample-timing-vs-diagnosis: '31 of your 120 cases gave blood after their diagnosis: which question are you answering, risk or signature?' This is asked before any estimate, and it decides eligibility, the estimand and whether glucose counts as leakage or as the baseline to beat.
- metab-qc-drift with metab-qc-design-insufficient: the pooled QCs (one sample) slope over run order within each batch. On this export, 'each batch has four QCs and its last nine injections come after its final QC: 18 samples no QC curve can reach.' Later, which top features were the end of the run.
- metab-one-compound-many-features: '404 features. About 100 compounds.' At explanation: 'your top three features are one compound's [M+H]+, [M+Na]+ and 13C isotope, and the model swapped between them fold to fold.'
- metab-metabolome-role: 'Asked which metabolites fish raises, each metabolite is the outcome. The default direction regresses fish intake on each metabolite, an answer to a question you did not ask, and self-reported intake error shrinks every slope.'
- metab-left-censored-nondetects with metab-zeros-meaning: '4,316 cells are zero and none is blank. Read as non-detections, they gather in the least abundant features (rho -0.99): too small to see, not absent. A median fill would put them mid-distribution.'
- metab-person-fingerprint (with metab-technical-replicates): 'Each sample's nearest neighbour is the same person's other sample 88% of the time. Random folds scored 0.95 by recognizing people; grouped by person, 0.64.'
- metab-pre-corrected-batches: 'Every feature's within-batch spread is identical, and your samples' batch shift is a tenth of your QCs': someone batch-corrected this file without the QCs. If the case label was in that correction, part of any difference was put there by it.'

## Journey load

TYPICAL STUDY (untargeted LC-MS plasma, binary outcome, like metabolomics_untargeted). Tier counts across the 42 threads: 11 asked (most of them conditional), 8 surfaced, 16 stated, 6 silent, 1 parked.

Questions are grouped into shared cards, so no fact is asked twice:
1. Assay card (metab-subdomain), always asked once and pre-filled: platform, LC vs GC, matrix, any measured normalizer, and prior processing (only when the range signal fires alone).
2. Reference rows and acquisition card, pre-filled; usually one accept:
   - pool and roles from values (metab-reference-rows);
   - order and batch columns (metab-qc-drift);
   - 'was the run randomized?' only when imbalance fires;
   - 'were QCs run but not exported?' only when no_pooled_qc fires.
3. Study-question card: metabolome role, sample timing, matched sets. Asked only when an exposure column, a disease-state outcome or a set ID exists.
4. Cross-lens covariate-role card (metab-clinical-factors). Not a new metabolomics question.
5. Repeat-grain question, shared by technical-replicates, person-fingerprint and low-biological-ICC. Only when IDs or profiles repeat.
6. Deployment question, under prediction only: new batch, lab or re-assay; new vs known person; short panel vs best score.

Fixed asks: 2 to 4 cards (3 to 5 under prediction). Conditional asks, each firing in about one journey in three or fewer: treatment marker, defining analyte, pre-corrected batches, urine normalizer, pre-analytical protocol.

Surfaced noticings: 5 to 7 per journey:
- QC drift strip with run-order balance
- non-detect mechanism
- missing-by-batch, when present
- features vs compounds
- noise ceiling
- person fingerprint, when repeats exist
Under prediction, panel stability is added. The feature-flow screen is the single place where every filter threshold is fixed. It absorbs five threads' threshold decisions.

On metabolomics_untargeted itself the journey is light:
- Asked: the assay card and the covariate-role card (plus deployment under prediction).
- Stated: reference rows (one accept), QC insufficiency with the fallback pre-selected (4 QCs per batch, 18 unbracketed injections), PQN as default (a negative control at AUC 0.535), and the feature flow.
- Surfaced: the drift strip (run-order balance at 0.545 becomes one methods line), the -0.99 non-detect figure, and the noise ceiling (mz_0094 at the null).

Load controls still owed:
- Explanation: a top feature could carry about 10 badges (drift rho, D-ratio, censoring, blank ratio, MSI level, selection frequency, treatment or definitional flag). Show one trust row per feature with only the failing badges.
- Evaluation under prediction: up to about 11 alternate scores (correction off, raw vs PQN, two fills, run-position baseline, clinical baseline, LOBO, grouped vs random folds, with/without flagged rows, permutation null). Show one ladder with only the deltas above a stated margin.
- Compute: whole-pipeline label permutation multiplied by these refits is a long CPU-heavy run on the machine beside the bed. Schedule it with Nolan, or run fewer permutations with a Monte Carlo interval.

## Every thread

| id | tier | looks at | what is noticed |
|---|---|---|---|
| metab-reference-rows | stated | structure | Pooled QCs, blanks and other instrument checks sit among the samples as if they were participants |
| metab-qc-drift | surfaced | measurement | Pooled QCs drift over injection order within each batch, and batches sit at different levels |
| metab-qc-design-insufficient | stated | design | The run cannot be QC-corrected as exported: no QCs or run order, too few QCs per batch, or samples outside the QCs' span |
| metab-run-order-aligned-with-outcome | surfaced | design | Cases and controls (or exposure groups) were run in different parts of the sequence, batches or plates, so drift can impersonate the outcome |
| metab-correction-honest-check | silent | measurement | QC RSD fell after correction, but the correction was fitted to make that number small |
| metab-qc-representativeness | stated | measurement | The pooled QC does not contain what the study samples contain |
| metab-feature-technical-reliability | silent | measurement | Each feature's technical noise in the QCs, relative to its spread across people, is a reliability, not just a filter |
| metab-blank-contamination | stated | measurement | Features present in the process blanks at levels near the samples' (contamination, not biology) |
| metab-dilution-linearity | silent | measurement | A dilution series shows which features respond to concentration at all, and which saturate |
| metab-internal-standards | stated | measurement | Some columns are spiked internal standards: they are the normalizer and the extraction check, not metabolites |
| metab-failed-injection | stated | measurement | A sample whose injection or extraction failed: low total signal, features missing across the range, internal standards off |
| metab-technical-replicates | stated | structure | The same sample was injected more than once, and the duplicates are counted as people |
| metab-merge-artifacts | stated | structure | A two-polarity merge left duplicate features, a duplicated sample, empty blocks and mixed ion modes |
| metab-subdomain | asked | measurement | Which kind of assay this is (untargeted LC- or GC-MS, targeted panel, NMR, lipidomics) decides which rules apply |
| metab-already-transformed | stated | measurement | The values were logged, normalized, autoscaled or closed to a constant sum before upload |
| metab-zeros-meaning | stated | measurement | Zeros in the intensity block: non-detections, failed measurements, or real values on an already-centerd scale? |
| metab-left-censored-nondetects | surfaced | measurement | Missingness tracks abundance: blanks are below a detection limit, not lost at random |
| metab-missing-by-batch | surfaced | structure | Non-detects cluster by batch or late in the run: a technical detection failure, not low concentrations |
| metab-group-specific-detection | silent | predictor-outcome | A feature present in one group and absent in the other is information, not a missing value |
| metab-dilution | stated | measurement | Samples differ in overall concentration (dilution, total signal), and the totals may differ with the outcome |
| metab-normalizer-carries-biology | asked | measurement | The normalizer itself moves with the outcome or exposure: creatinine in kidney disease, a global shift that breaks PQN, TIC closure |
| metab-variance-structure | silent | predictors | Variance rises with abundance and explodes near the detection floor after the log |
| metab-one-compound-many-features | surfaced | predictors | Hundreds of features are a few dozen compounds (adducts, isotopes, in-source fragments, both ion modes) |
| metab-annotation-confidence | stated | measurement | Which features are identified, and how confidently (MSI level), sets what a name in a result may claim |
| metab-enrichment-background | parked | structure | Only a fraction of features have names, so any pathway reading runs over what the assay could name |
| metab-treatment-marker | asked | predictor-outcome | A top feature is a drug, or its metabolite, given for the outcome: it marks the diagnosis, not the biology |
| metab-preanalytical-aligned | asked | design | How samples were collected, handled or stored differs between the groups |
| metab-low-biological-icc | stated | measurement | One blood draw captures little of a person's usual level of a metabolite |
| metab-clinical-factors | asked | predictor-outcome | Age, sex, BMI, kidney function and fasting move much of the metabolome, and may differ by outcome |
| metab-detectable-effect | silent | design | What fold change this study could have seen, given its own technical and biological variance |
| metab-limit-flags-in-cells | stated | measurement | Cells hold '<LOD', '<0.20' or '>ULOQ': the export says where the limits are, on both sides |
| metab-pre-corrected-batches | asked | measurement | The batches were already corrected before export, possibly with the outcome in the model |
| metab-metabolome-role | asked | design | Is the metabolome the exposure, the outcome of a diet, or the path between them? |
| metab-outcome-defined-by-a-feature | asked | predictor-outcome | The outcome is defined by a metabolite that is in the panel (glucose for diabetes, creatinine for CKD, urate for gout) |
| metab-sample-timing-vs-diagnosis | asked | time | Some samples were drawn after diagnosis or treatment began: the metabolome may be the disease, not its risk |
| metab-matched-sets | asked | design | The rows are matched case-control sets, not independent people |
| metab-person-fingerprint | surfaced | structure | Repeated samples from one person are more alike than any intervention effect: the metabolome is a fingerprint |
| metab-cross-batch-transport | asked | measurement | The next batch, lab or platform will not look like these: the honest score is across batches |
| metab-wide-noise-ceiling | surfaced | structure | With 392 features and 72 people, noise alone reaches a high score: the bar any finding must clear |
| metab-panel-instability | surfaced | predictor-outcome | The 'panel' changes from fold to fold: which features are chosen is itself an estimate |
| metab-ratio-features | stated | predictors | Two features are a substrate and its product: their ratio indexes a pathway step, and it cancels dilution |
| metab-feature-flow | stated | measurement | How 2,400 features became 900: each filter's count, and which thresholds decide which features |
| metab-affinity-proteomics † |  | measurement | The assay block is affinity proteomics (Olink NPX or SomaScan RFU), with vendor normalization, plates and LOD flags |
