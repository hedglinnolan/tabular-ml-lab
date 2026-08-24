# Genomics / transcriptomics pack — research specification

Structured as the product owner asked: by modeling step, and within each step by
**diagnostic** (what the app detects), **coaching** (what it says, with the reason), and
**presentation** (what it draws, and what makes it publication-grade). Every recommendation is
tagged **SETTLED** · **CONVENTION** · **DISPUTED**, so the app never asserts a norm as a fact.

Scope: bulk transcriptomics (RNA-seq counts, microarray intensity) plus genotype matrices at a
briefing level. **Single-cell is out of scope and must be detected and refused** — zero fraction
>80–90% with >1,000 columns and median non-zero count ≤3.

---

## 00 · The non-defaultable set

The pack asserts no normalization default. This is the reason, decision by decision.

| Decision | Why it cannot be defaulted |
|---|---|
| **Normalization method** | Correct choice depends on what the values already are, whether gene length matters, whether "most genes are not DE" holds, and which test consumes it. None is fully recoverable from the matrix. |
| **CPM vs TPM** | **Formally undecidable from the matrix alone.** Both rescale each sample to exactly 1e6; TPM divides by effective length *before* the rescale, which erases the trace. No column-sum, distributional or skew test separates them. Weak hint only: correlate log(value) with log(length) within a sample — CPM retains a positive length association. Report as a hint, never a determination. |
| **Batch variable identity** | Batch is metadata, not data. PCA shows *that* structure exists, never *what* it is. |
| **Correct the matrix vs model batch in the design** | Genuinely disputed — see §05. |
| **Gene length** | Not derivable from a count matrix. |
| **Reference level of the condition factor** | R defaults to alphabetical; "KO vs WT" silently reports the inverted sign. |
| **Estimated counts vs normalized values** | Both non-integer, opposite downstream treatment. |

---

## 01 · Import and structure

### Orientation — genes in rows is the convention (SETTLED)

The expression matrix is stored **features in rows, samples in columns** — the inverse of the
tidy convention the rest of the app assumes. DESeq2: *"the value in the i-th row and j-th column
tells how many reads can be assigned to gene i in sample j."*

**Detection cascade, in priority order:**
1. **ID-pattern match** on axis labels: `ENS[A-Z]{0,4}G\d{11}(\.\d+)?` (Ensembl gene),
   `ENS[A-Z]{0,4}T\d{11}` (transcript), `^[NX][MR]_\d+` (RefSeq), `^\d+(_[a-z]+)?_at$`
   (Affymetrix), `^ILMN_\d+` (Illumina), `^A_\d+_P\d+` (Agilent), HGNC symbols.
2. **Cardinality bands**: ~20,000 protein-coding · 35,000–65,000 full GENCODE ·
   22,000–55,000 array probes · 1e5–1e7 variants. Samples typically 4–1,000.
3. **Metadata join** — the axis whose labels intersect an uploaded sample sheet. Most reliable;
   prefer it over 1 and 2 when available.
4. **Ambiguity guard** — both dimensions >2,000, or no ID match: do **not** guess. Show a 5x5
   corner preview and ask.

Emit `orientation_confidence`. Anything below "ID-pattern matched" is a confirmation step, not a
silent transpose.

**Coaching:** *"Your file looks like the genomics standard: genes in rows, samples in columns —
the transpose of how most statistics software expects data. Confirm before we go further: an
undetected transpose produces analyses that run cleanly and are entirely meaningless."* A
transposed matrix does not error. It produces a PCA of genes labeled as samples.

**Presentation:** a structure card — orientation, n, p, p/n, dtype, % zeros, min/max/median,
detected ID format — shown *before* any analysis. Plus a 6x6 labeled corner preview.

### Gene IDs

**Diagnostic:** classify vocabulary; report version suffixes present (joins against unversioned
annotation fail silently, dropping genes), duplicate IDs after symbol mapping (many-to-one),
mixed vocabularies, and **Excel corruption** — date-like strings (`1-Mar`, `2-Sep`, `44621`).

**Excel corruption is SETTLED and measured:** Ziemann, Eren & El-Osta (*Genome Biology* 17:177,
2016) found gene-name conversion errors in ~20% of papers with supplementary Excel gene lists;
Abeysooriya et al. (*PLoS Comput Biol* 2021) found the rate had **risen**. HGNC renamed
`SEPT*`→`SEPTIN*` and `MARCH*`→`MARCHF*` partly because of this. **Never auto-repair — report
and stop.**

---

## 02 · Data-type detection — the highest-leverage diagnostic in the pack

It determines what is legal downstream, and getting it wrong is the commonest real failure.

Per column: sum, min, max, median, % zeros, % integer. Per matrix: global max, negatives present.

| Signature | Classification |
|---|---|
| Integers ≥0, max ≫1e4, column sums vary widely, 20–60% zeros | **Raw counts** |
| Non-integer ≥0, sums vary, max ≫1e4 | **Estimated counts** (salmon/kallisto/RSEM) — ask |
| Every column sums to 1e6 ±1e-3 | **CPM or TPM — indistinguishable** |
| Sums roughly but not exactly equal near 1e6 | TMM- or median-of-ratios-scaled CPM |
| Non-negative, sums not constant, max 1e3–1e5, heavy skew, non-integer | **FPKM/RPKM** |
| Continuous, max ~15–25, repeated floor, roughly homoscedastic | **VST** |
| As above but small negatives permitted | **rlog** |
| Continuous 2–16, no zeros, probe-style IDs | **Microarray log2 intensity (RMA)** |
| Symmetric around 0, range ~-6..+6 | Already log-ratio / z-scored / corrected |

**Hard rule (SETTLED):** any negative value rules out raw counts, CPM, TPM and FPKM.

**Coaching, branched:**
- **Raw counts** — the only input that lets a count model estimate measurement precision. DESeq2:
  *"only the count values allow assessing the measurement precision correctly"* and it
  *"internally corrects for library size, so transformed or normalized values should not be used
  as input."* Do not pre-normalize. **SETTLED**
- **TPM/CPM/FPKM** — already per-sample-normalized, which closes off the negative-binomial route
  because count-level variance is destroyed. Either recover raw counts (strongly preferred) or
  use a limma-style Gaussian workflow on log2(x+offset). Feeding these to a count model runs
  silently and its p-values are wrong. **SETTLED**
- **FPKM specifically** — not comparable across samples *even in principle*. Wagner, Kim & Lynch
  (*Theory Biosci* 131:281, 2012) showed RPKM/FPKM violates the invariance a relative-molar-
  concentration measure must satisfy. Dillies et al. (*Brief Bioinform* 14:671, 2013): total-count
  and RPKM normalization did **not** control the false-positive rate; TMM and median-of-ratios
  did. **SETTLED**
- **VST/rlog** — for visualization, clustering and PCA. **Never** the input to a DE test. **SETTLED**
- **Microarray** — the whole count toolchain does not apply; use limma. **SETTLED**

**Presentation:** a **"what your numbers are" card** — classification, confidence, the evidence
that drove it, and a **capability matrix** showing which downstream steps are now enabled,
disabled, or require input. *This is the single most valuable artifact in the pack.* Plus a
column-sum bar chart with a 1e6 reference line, and per-sample overlaid density of log2(x+1),
whose shape is diagnostic.

---

## 03 · Filtering

**SETTLED that you filter; CONVENTION on the threshold.** It raises power by cutting the
multiple-testing burden, and standard implementations are *independent of the condition labels*,
so it does not bias the test.

- **DESeq2 recipe:** `rowSums(counts >= 10) >= smallestGroupSize`.
- **edgeR `filterByExpr()`:** CPM above a threshold derived from `min.count` (default 10) and
  library sizes, in n samples derived from the design. Recommended by its authors *because it
  avoids arbitrary user thresholds*.

**Independent filtering (SETTLED):** DESeq2's `results()` uses mean normalized count as a filter
statistic and picks the quantile maximizing discoveries at the FDR threshold. Theory: Bourgon,
Gentleman & Huber (*PNAS* 107:9546, 2010) — in their application, filtering on overall variance
before a t-test increased discoveries ~50%. **Hard requirement: the filter statistic must be
independent of the test statistic under the null**; some pairs *lose* type-I error control.

**Filter before or after normalization — DISPUTED with a conventional resolution.** edgeR/limma
filter first then compute TMM; DESeq2 pre-filters lightly and does the consequential filtering
after fitting. Not disputed: the filter must not depend on condition labels. Report the order.

**Anti-patterns:** filtering on condition labels (breaks FDR) · filtering by fold change or
p-value before testing (circular) · "top 5,000 most variable then DE on that set" (can break
type-I control).

**Presentation:** filtering waterfall (input → zero-removal → expression filter → tested) ·
mean–rejections curve (the independent-filtering optimization plot, a standard supplementary
figure) · histogram of log10 mean normalized count with retained/removed shaded.

---

## 04 · Normalization — no default asserted

| Method | Corrects | Interchangeable with | Wrong when |
|---|---|---|---|
| CPM | depth only | — | gene length matters; composition skew |
| TPM | depth + length | ~FPKM *within* a sample | cross-sample DE |
| FPKM/RPKM | depth + length (wrong order) | superseded by TPM | cross-sample — **provably inconsistent** |
| **TMM** | depth + **library composition** | ~median-of-ratios for DE | >~50% of genes shift one way |
| **Median-of-ratios** | depth + composition | ~TMM | same; also fails if no gene non-zero in all samples (use `poscounts`) |
| VST / rlog | mean–variance dependence | VST≈rlog at large n | **never** as DE input |
| Quantile | forces identical distributions | cyclic loess (more robust to one-sided DE) | distributions genuinely differ |
| Inverse normal | everything, destructively | — | effect sizes must stay interpretable |

**TMM and median-of-ratios are near-interchangeable; CPM/TPM/FPKM are not substitutes (SETTLED).**
Both correct library *composition* — if a few genes hog reads in one sample, every other gene's
count is deflated. Depth-only scaling does not.

**Both assume most genes are not DE (SETTLED assumption, DISPUTED how often violated).** Global
transcriptional shifts (MYC amplification, cross-tissue, transcription-halting drugs) break it,
and the corrected data still look fine. **The tool cannot detect this from the matrix** — needs
spike-ins or control genes via RUVg.

**What a wrong choice costs:** CPM instead of TMM on composition-skewed data → gene-wide
fold-change bias, visibly asymmetric volcano · TPM into DESeq2 → meaningless dispersions,
normal-looking wrong p-values · quantile when many genes truly move one way → real signal
flattened · VST into a test → typically anticonservative · double-normalizing → uninterpretable.

**Existence proof, not a default:** GTEx's eQTL pipeline filters >0.1 TPM in ≥20% of samples AND
≥6 reads in ≥20%, then TMM, then an inverse normal transform per gene. Tuned for QTL mapping, not
DE — the rank transform is right there and wrong when fold changes must be reported. That contrast
is why there is no universal default.

**Presentation:** before/after boxplot of log2 per sample (ordered by group, colored by condition,
y-axis named in real units) · **RLE plot** — per-sample boxplots of each gene's deviation from its
median; the most sensitive single QC figure for normalization adequacy, under-used and respected ·
size-factor table flagging anything outside ~[0.5, 2] · mean–dispersion plot (`plotDispEsts`) — if
the fitted curve does not track the cloud, the model is wrong.

---

## 05 · Batch effects

**Diagnostic:** design balance table (every technical covariate x condition) · **rank of the model
matrix** for `~batch + condition` — rank-deficient means perfectly confounded · Cramer's V for
partial confounding · regress top 5–10 PCs on each covariate, report R2 per PC x covariate · flag
PC1 correlating with a technical variable more than with condition · flag clear clustering with
**no batch metadata supplied** — that is a prompt to find the metadata, not license to invent it ·
correlate library size and % zeros with PCs (depth-driven PC1 is a normalization problem, not batch).

**Perfect confounding — stop (SETTLED).** *"All of your A samples were processed in batch 1 and all
B in batch 2. The batch effect and the biological effect are mathematically the same variable. No
software can separate them, and any tool that appears to is fabricating. This is a design problem,
not a modeling problem."*

**Partial confounding — model it, do not erase it (CONVENTION, strong support).** Include batch in
the design rather than correcting the matrix and testing corrected values.

**The two-step warning — SETTLED that the problem is real.** Nygaard, Rodland & Hovig
(*Biostatistics* 17:29, 2016): correcting with ComBat and then running a standard test inflates
significance, worst under **unbalanced** batch-by-group designs, because the correction induces a
correlation structure the downstream test ignores. In their GSE61901 example the ComBat pipeline
returned **>1,000** DE probesets where batch-as-fixed-effect returned **11**.

**When batches are unknown:** SVA (Leek & Storey 2007) estimates latent factors orthogonal to the
biological variable and you add them to the *design* — which is exactly the pattern that avoids the
Nygaard problem. RUV (Risso et al., *Nat Biotechnol* 32:896, 2014) anchors on negative controls —
RUVg (control genes/spike-ins), RUVs (replicates), RUVr (residuals). **ComBat and SVA answer
different questions and are not substitutes (SETTLED).**

**`removeBatchEffect` is for pictures only (SETTLED).** Legend must say: *"batch-corrected for
visualization; testing used uncorrected data with batch in the design."*

**Presentation:** PCA colored by condition, **shaped** by batch (or a small-multiple grid colored by
each covariate in turn) · **PC x covariate association heatmap** (R2 or -log10 p, % variance
annotated) — far more convincing than eyeballing · design balance table as a mosaic, with perfect
confounding in an alarm color and **blocking** — the user sees it before any inference runs.

---

## 06 · Multiple testing

**Diagnostic:** count tests actually performed post-filter and display expected false positives at
nominal alpha (`p_tested x 0.05`) · p-value histogram and its shape · pi0 and discoveries at BH
0.05 / 0.10 / Bonferroni · whether the user's threshold is on raw or adjusted p · count of `NA`
adjusted p-values and *why* (independent filtering vs Cook's-distance flagging — very different
meanings).

**p<0.05 across 20,000 genes is not a finding (SETTLED).** ~1,000 genes clear it by chance.

**BH controls the expected *proportion* of false discoveries among discoveries (SETTLED)** — at
FDR 5% with 400 hits, expect ~20 false. Bonferroni is conventional in GWAS (**5e-8**) and too
conservative for transcriptome-wide DE.

**padj + |log2FC| is DISPUTED — present both sides.** The common `padj<0.05 AND |LFC|>1` is a
post-hoc filter that controls nothing. If the claim is "changed more than 2-fold," test it
directly: DESeq2 `lfcThreshold=1`, edgeR `glmTreat`, limma `treat`. But the convention is
widespread and reviewers accept it — offer both, label which was used.

**Shrink LFCs before ranking or plotting (CONVENTION, well-supported).** Raw LFCs for low-count
genes produce the flared wings at the bottom of a volcano. `lfcShrink()` for the MA plot, volcano,
and top-gene tables. **Not** for computing p-values.

**Presentation — the p-value histogram is mandatory and the most under-used diagnostic in the
field.** 50 bins on [0,1]:
- Uniform with a spike near 0 → healthy, BH valid
- Perfectly flat → no signal; report it, do not hunt
- **Hill-shaped, peaking mid-range** → conservative test, often variance overestimate
- **U-shaped or rising toward 1** → **model misspecification** — wrong variance assumption,
  unmodeled confounder, or correlated samples treated as independent. *This is the shape that says
  your batch correction inflated things, or you have repeated measures you did not model.* Surface
  it loudly.
- Comb-like → discreteness at very low counts

Plus a BH staircase (sorted p vs rank with the `(i/m)*alpha` line and crossing point) and a
discoveries-vs-alpha table, which prevents threshold-shopping by making it visible.

---

## 07 · EDA and presentation — the priority

### A · PCA of samples — the single most expected figure

Compute on **variance-stabilized values**, never raw counts, where PC1 becomes a library-size
artifact. Default **top 500 most variable genes** (the `plotPCA(ntop=500)` / `plotMDS(top=500)`
convention — say so, and let the user change it). Also compute the all-genes version; if the
picture changes, that is informative.

Flag: PC1 correlating with library size or % zeros (normalization issue) · PC1 driven by 1–2
samples (outlier, not structure) · no separation by condition (say so).

**What makes it publication-grade:**
1. **Axis labels carry % variance** — `PC1 (42% variance)`. An unlabeled PCA is uninterpretable.
2. **Equal aspect ratio** — without it the visual separation is a lie proportional to the aspect
   ratio. Widely ignored; the DESeq2 workflow explicitly recommends it.
3. Color = biological condition; **shape** = batch. Never both on color.
4. Label outliers only.
5. **No confidence ellipses at n<10 per group** — they imply a sampling model that is not there.
6. Report `ntop` in the legend. 500 genes and 20,000 genes are different analyses.
7. Companion scree plot; colorblind-safe palette with shape redundancy.

**MDS alternative (edgeR house style):** distances are **leading log-fold-changes** — the RMS of
the 500 largest absolute log-FCs *per pair* — so they read directly as "typical fold change
between these two samples." Not the same computation as PCA; label which was used.

### B · Sample–sample correlation heatmap — the p>>n-safe cousin of a correlation matrix

n x n, Spearman on ranks or Pearson/Euclidean on VST. Flag samples whose median correlation to all
others falls below ~0.8 (CONVENTION, context-dependent — present as a distribution, not a rule).
Detect whether the top dendrogram split follows condition or batch.

Publication-grade: annotation bars for condition and batch (this is what makes it argue something)
· symmetric, same ordering both axes, dendrogram shown · **sequential** colormap, not diverging —
diverging falsely implies a meaningful midpoint · **always print the numeric scale**, because a
heatmap auto-scaled over 0.97–0.99 manufactures apparent contrast · state metric and linkage in
the caption.

### C · Clustered expression heatmap — contains the pack's most important honesty flag

**Row scaling (CONVENTION, near-mandatory):** z-score each gene across samples, or a few highly
expressed genes dominate the color range and clustering groups by *absolute level* rather than
*pattern*. Legend says `row z-score`, not `expression`.

**The circularity flag (SETTLED, constantly violated):** if you select genes by DE between A and B
and then show a heatmap whose clustering splits A from B, **that split is guaranteed by
construction**. Double dipping — Kriegeskorte et al. (*Nat Neurosci* 12:535, 2009). The heatmap is
a legitimate *display* of the DE result; it is not independent confirmation. For clustering to
mean something, select genes by variance (condition-blind), or select on one dataset and display
on another.

Publication-grade: diverging colormap centered exactly at 0, symmetric limits, clipping stated ·
column annotation bars · do not cluster columns if the claim is a clean group split — order by
group and say so · cap labeled genes at ~50–100 · state gene-selection criterion, n, scaling,
distance and linkage. Five items, one caption sentence.

### D · Volcano plot

**Thresholds must be on adjusted p-values** even if the axis plots raw p. Use **shrunken** LFCs on
x. Cap underflowed p=0 at a stated value.

Publication-grade: FDR line labeled · LFC lines only if an LFC criterion was applied · color by
up/down/NS with **counts in the legend** · label 10–20 genes max, repelled · **symmetric x-limits —
asymmetry in the cloud is diagnostic, not decorative**; if all significant genes go one way,
suspect normalization before believing biology · axis label names direction explicitly
(`log2 fold change (Treated / Control)`) — ambiguous direction is the commonest volcano error ·
alpha ~0.3 or rasterize the non-significant mass.

### E · MA plot

Mean expression (log x) vs log2 FC. Want: cloud centered on 0 across the whole x range, funneling
at low expression. Wrong: drift off zero at high expression (normalization failure) or a systematic
trend (composition bias). **This is the figure that catches a bad normalization when the boxplots
looked fine.** Out-of-range points as open triangles at the boundary rather than silently dropped.

### F · Library size and count distributions

Per sample: total counts, detected genes, % zeros, % of reads in top 10/100 genes, mitochondrial
or rRNA/globin fraction if IDs allow. Flag libraries <~50% of median depth; extreme top-gene
domination (one gene >10–20% usually means rRNA/globin carry-over).

Presentation: library-size bar chart with median line — check the pathological case of depth
correlating with condition · overlaid log2(CPM+1) densities before/after filtering, which visually
justifies the filter · detected-genes vs library-size scatter, which should saturate.

### G · Clustering

Sample clustering on VST with stated distance and linkage is standard QC. Gene modules via
hierarchical or k-means are descriptive, not discovered — **DISPUTED** whether any k-selection
criterion is trustworthy at this dimensionality; "we chose k=4 by inspection of the elbow" is more
honest than a false-precision statistic. **SETTLED anti-patterns:** clustering samples on genes
selected for DE between those samples · Euclidean distance on unscaled expression (clusters by
level; correlation distance on z-scored rows clusters by pattern).

### The figures that BREAK at p >> n — general high-dimensional guidance

| Standard artifact | Why it breaks at p=20,000 | Replacement |
|---|---|---|
| Feature x feature correlation matrix | 4e8 cells ~3.2 GB, unrenderable — and **rank <= n-1**, so most of its "structure" is a deterministic artifact of n | **Sample x sample** heatmap (n x n) · PCA loadings · correlation within a pre-specified set or module eigengenes |
| Scatterplot matrix | ~2e8 panels | PCA score grid on components, not features |
| Per-feature boxplots/histograms | 20,000 panels | Density of a per-feature summary statistic — one plot describing 20,000 features |
| Per-feature p-value table | unreadable | p-value histogram + volcano + top-N table |
| **VIF / multicollinearity** | X'X is singular whenever p>n; **VIF is undefined** | Condition number after PCA; report effective rank (<= n-1); collinearity is the regime, not a defect |
| Residual QQ from a saturated fit | unregularized p>n interpolates: residuals are exactly 0 | QQ of **test-fold** residuals only |
| Missingness heatmap over all features | unrenderable | Missingness per sample + histogram of per-feature missingness |
| Raw Euclidean kNN on all features | **Distance concentration** — farthest/nearest ratio → 1 (Beyer et al. 1999; conditions for avoidance in Durrant & Kaban 2009). Also **hubness** | kNN/clustering on PCA scores (first 10–50 PCs) or correlation distance over a variance-selected subset. State the reduction. |
| "Standardize everything and model" | amplifies thousands of near-zero-variance features into unit-variance noise | Filter near-zero-variance first, then scale |

---

## 08 · Modeling at p >> n

**Regularization is mandatory (SETTLED).** With p>=n an unregularized model is *degenerate*, not
merely overfit: infinitely many coefficient vectors fit exactly, the solution is not unique,
standard errors are undefined, and training error is exactly zero regardless of signal. Perfect
separation in logistic regression is the visible symptom.

- **Ridge** — keeps all features, handles correlation, no selection. **SETTLED as valid.**
- **LASSO** — **can select at most n variables**; with n=40 you can never get a 200-gene signature.
  Among a correlated group it arbitrarily picks one, which is exactly wrong when the object of
  interest is a co-regulated pathway. **SETTLED limitation** (Zou & Hastie 2005).
- **Elastic net** — the standard omics choice if one must be named. L2 restores the grouping effect
  and removes the <=n cap. alpha=0.5 is a **CONVENTION**, not a result.
- **sPLS-DA** — built for this regime; components rather than individual genes. Cost: components
  are combinations of many genes, harder to translate into an assay. **CONVENTION.**
- **Tree ensembles** — importance is biased toward correlated groups and unstable at small n.
  **DISPUTED** whether they beat penalized linear models. MAQC-II (*Nat Biotechnol* 28:827, 2010):
  across >30,000 models by 36 teams on 13 endpoints, performance depended mainly on **the endpoint
  and team proficiency**, and different algorithms produced similar performance. **Algorithm choice
  is rarely the lever people think it is** — the app should say this.

### Cross-validation — the section that prevents fraud-by-accident

**Feature selection outside the CV fold is the field's signature failure (SETTLED).** Ambroise &
McLachlan (*PNAS* 99:6562, 2002): rank genes on all samples, then cross-validate on the top 50, and
the error estimate is not merely optimistic — it can be **near zero when no signal exists at all**.
Every pre-2002 error rate that selected genes on full data is uninterpretable. **Everything
supervised — ranking, DE testing, threshold choice, hyperparameter selection — happens inside the
training fold.** Nested CV: outer for performance, inner for tuning.

**Automate the check.** Instrument the pipeline to detect any supervised transform executed on the
full dataset before the split. *This is the single most valuable automated check in the pack.*
Also detect repeated subject IDs across folds (grouped CV required) and batch-fold alignment.

**Signature instability (SETTLED).** Michiels, Koscielny & Hill (*Lancet* 365:488, 2005) reanalyzed
the seven largest published microarray prognosis studies and found the signature highly sensitive
to which samples were in the selection set, with many different signatures achieving similar
accuracy. **Report selection frequency across resamples, not a single list presented as *the*
signature.**

**Unsupervised preprocessing leakage — GENUINELY DISPUTED.** Whether normalization/VST must be
refit inside each fold is a real disagreement. Strict: any data-estimated transform should be fit
on training only. Pragmatic (what nearly all published pipelines do): label-blind transforms leak
negligibly and refitting changes the feature space between folds. **The tool must not pick a
side** — enforce fold-internal fitting for anything *supervised* where there is no dispute, offer
both for unsupervised, require the choice to be stated.

**Small-n CV is high-variance (SETTLED).** At n<50 a single 5-fold estimate has a standard error
large enough that a 0.05 AUC difference is noise. Repeated CV, report the distribution. LOOCV is
nearly unbiased with notoriously high variance and no usable spread.

**Sample size (CONVENTION / heuristic — flag as such).** Schurch et al. (*RNA* 22:839, 2016), 48
replicates per condition in yeast: with **3 replicates** most tools recovered only **20–40%** of
the genes found with the full set, rising to >85% for genes changing more than 4-fold. Their
recommendation: **>=6 biological replicates**, **>=12** when detection across all fold changes
matters, **>20** for >85% recovery. Caveat honestly: one organism, low biological variability
relative to human clinical cohorts — human studies typically need *more*. For prediction there is
no defensible closed-form rule at p>>n; report a **learning curve** instead of a power calculation,
and below ~n=50 per class advise framing the work as signature discovery requiring external
validation.

**Presentation:** learning curve with resampling bands (the honest replacement for a power
calculation) · fold-level performance boxplot, never a bare mean AUC · regularization path and
CV-error-vs-log-lambda with lambda_min and lambda_1se marked · **selection-stability plot** — the
fraction of resamples in which each gene was selected, sorted; this turns an unstable gene list
into an honest one · nested-CV schematic in the methods · **never** a training-set ROC unless
labeled "training (optimistic)" beside the CV curve.

---

## 09 · Population structure (genotype) — brief

Per-variant: call rate, MAF, HWE p in controls. Per-sample: call rate, heterozygosity outliers
(contamination or inbreeding), sex-check concordance.

**All SETTLED:** genotype PCs must be computed on an **LD-pruned** set of autosomal common
variants with **long-range LD regions excluded** (MHC on chr6, chr8 and chr17 inversions) — without
pruning the top PCs describe LD blocks, not ancestry. Include top PCs as covariates (Price et al.,
*Nat Genet* 38:904, 2006; the top-10 convention comes from that paper and is empirically justified
by its effect on lambda_GC, not derived). **PCs handle distant structure, not close relatedness** —
use an LMM (BOLT-LMM, SAIGE, fastGWA, GEMMA) or remove relatives (`--king-cutoff 0.0884`).
Genome-wide significance **p < 5e-8**.

**Caution from recent literature:** PC adjustment can induce **collider bias**, and in **admixed**
populations can create spurious associations. Do not present "10 PCs" as a solved default there.

**Presentation:** PC1 vs PC2 colored by ancestry and case/control, optionally with 1000 Genomes
projected · scree · **QQ plot with lambda_GC annotated** — a QQ without lambda is incomplete ·
Manhattan with the 5e-8 line labeled · kinship histogram with degree boundaries marked.

---

## 10 · Reporting — assembled automatically, with unfillable fields flagged

Required: design (n per group, biological vs technical replication, batch structure, balance
table) · platform, read length, depth achieved · **genome build AND annotation release** (GENCODE
vXX) · **software versions for every step** — versionless methods are the most-cited
reproducibility failure (Simoneau et al., *Brief Bioinform* 22:140, 2021) · filtering rule stated
as a rule with genes-before → genes-after · normalization named · **the full design formula
including reference levels** · test, multiple-testing method, alpha, whether independent filtering
was applied · whether LFCs are shrunken · whether batch correction touched the matrix and whether
it was used for inference or only visualization · sample exclusions with **pre-stated** criteria ·
data availability (GEO/SRA; MINSEQE or MIAME) · code availability.

For prediction, additionally: the CV scheme with an explicit statement that selection happened
inside training folds · hyperparameter search space · whether an independent validation set exists
· class distribution and the metric's baseline.

**Presentation:** a generated methods paragraph with unfilled slots **visibly bracketed** rather
than silently omitted or plausibly guessed · a reporting checklist table, exportable · a
**provenance panel** listing every parameter the app chose, every one the user chose, and **every
default it declined to set, with the reason from §00**. The pack's thinness is a feature; this
panel is where that becomes visible and trustworthy.

---

## 11 · Anti-pattern registry

| Anti-pattern | Status | Consequence |
|---|---|---|
| TPM/FPKM/CPM/VST into DESeq2 or edgeR | SETTLED wrong | Invalid dispersions → invalid p; runs silently |
| Comparing FPKM across samples | SETTLED wrong | Provably inconsistent (Wagner 2012) |
| Raw p<0.05 over ~20,000 tests | SETTLED wrong | ~1,000 expected false positives |
| Feature selection on all data, then CV | SETTLED wrong | Near-zero error from pure noise |
| Unregularized regression at p>=n | SETTLED wrong | Degenerate, non-unique; perfect separation |
| ComBat then naive test, unbalanced design | SETTLED problem | 1,000 vs 11 DE genes in Nygaard's example |
| Batch correction under perfect confounding | SETTLED wrong | Fabricated separation |
| DE-selected genes, then clustering shown as validation | SETTLED wrong | Circular (Kriegeskorte 2009) |
| PCA on raw or CPM counts | SETTLED wrong | PC1 is a library-size artifact |
| Unlabeled PCA axes, unequal aspect ratio | SETTLED presentation error | Uninterpretable / misleading |
| Heatmap without scaling, metric, linkage, selection rule | SETTLED presentation error | Irreproducible |
| Auto-ranged color scale with no numeric legend | SETTLED presentation error | Manufactures contrast |
| Excel-corrupted gene symbols | SETTLED, ~20% prevalence | Silent gene loss (Ziemann 2016) |
| Filtering using condition labels | SETTLED wrong | Breaks FDR |
| Filter statistic not independent of test statistic | SETTLED wrong | Loses type-I control (Bourgon 2010) |
| Ignoring reference-level direction | SETTLED wrong | Sign errors everywhere |
| Feature x feature correlation at p=20,000 | SETTLED broken | 3.2 GB, singular |
| Raw Euclidean kNN on all features | SETTLED risk | Distance concentration, hubness |
| n=3 presented as adequate for genome-wide DE | CONVENTION violated | 20–40% recovery (Schurch 2016) |
| A single gene list as *the* signature | SETTLED wrong framing | Unstable to resampling (Michiels 2005) |
| Unshrunken LFCs on volcano/MA and rankings | CONVENTION violated | Low-count noise dominates extremes |
| Genotype PCs without LD pruning | SETTLED wrong | PCs describe inversions and MHC |
| PC covariates where relatedness needs an LMM | SETTLED wrong | Residual inflation |
| Methods without build, annotation, versions | SETTLED reporting failure | Not reproducible |
| Bulk pipeline on a single-cell matrix | SETTLED wrong | Zero-inflation, pseudoreplication |
