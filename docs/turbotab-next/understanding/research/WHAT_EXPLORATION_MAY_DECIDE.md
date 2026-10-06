The rule table: what exploration may decide

**The short answer to the owner's question.** EDA does three jobs that go beyond a vague sense of the data. Views that never use the outcome decide how the data are represented. Under prediction, EDA writes the list of candidates that resampling will test. And EDA audits data quality and design assumptions. It never picks a winner by eye.

**Input note.** The time-series verdict reached me cut off partway through its first claim. I re-checked the Poincaré claims myself; those sources are listed under example 3 and correction 16.

---

## 1. The principle (corrected)

Exploration proposes. A proposal becomes a decision only through a rule that meets three conditions:
- it is declared before the data that will judge it are seen;
- it is rerun wherever those data are resampled (every fold, every bootstrap replicate);
- its uncertainty is carried into the reported number.

What a view may propose depends on what it saw:
- Views that never used the outcome may shape the representation.
- Views that used the outcome may only add candidates to a declared search under prediction.
- Under inference, outcome-using views are reported, not acted on, unless a declared selection-aware method carries them.

---

## 2. The rule table

Read each cell as three parts. "Informs" is what the view may decide. "How" is the mechanism: a rule refit in each fold, a declared choice, or disclosure only. "Source" is the key reference.

| What the view looks at | Inference, before the plan lock | Inference, after the plan lock | Prediction, on training rows (holdout sealed) |
|---|---|---|---|
| **Predictors only (outcome-blind):** distributions, missingness, prevalence, correlations, QC, exposure–covariate overlap | **Informs:** the representation. That covers transforms, spike-at-zero coding, knot count and placement, overall-prevalence filters with a fixed threshold, log-ratio coordinates, imputation and zero-replacement rules, measurement-error and usual-intake models, data reduction, and trimming for overlap. **How:** a declared choice, written into the plan, with every change driven by initial data analysis (IDA) logged. A transform of the exposure changes the estimand, so it is fixed here. Steps learned without the outcome may be fitted on the full analysis sample. **Source:** Harrell RMS 4.7 ("completely masked to Y"); Heinze 2024; Baillie 2022 rule 5; Rubin 2008. | **Informs:** nothing new in the primary analysis. **How:** disclosure only. A problem found now becomes a documented deviation plus a sensitivity analysis, and the primary estimate stays. **Source:** Baillie 2022: "All changes to the SAP should be well motivated, and well documented." | **Informs:** the representation and candidate preprocessing steps. **How:** refit in each fold every step that reads other rows: data-derived thresholds, imputation, zero replacement, scaling, quantile knots, PCA loadings, D-ratio. Fixed row-wise transforms need no refit (log, or CLR with a fixed pseudocount and fixed feature set). EDA views exclude the holdout. **Source:** Moscovich & Rosset 2022; Hornung 2015. |
| **Outcome alone (univariate Y)** | **Informs:** the data quality of Y (impossible values, units, detection limits, censoring) and the effective sample size that caps the model's degrees of freedom. **Does not inform:** the loss or the model family by its shape. A robust loss changes the estimand from mean to median, and zero inflation cannot be judged from a raw histogram. **How:** a declared choice; the estimand and outcome family come from the question. **Source:** Heinze 2024 ("included the outcome variable in univariate evaluations, but intentionally excluded it from any bivariate or multivariate analysis"); ESL 10.6; Warton 2005. | **Informs:** model checks conditional on X: residual tails, and excess zeros given the fitted mean. **How:** disclosure plus prespecified sensitivity analyzes (robust SEs, an alternative family). The primary estimand does not move. **Source:** Warton 2005; Popovic 2015. | **Informs:** stratified folds for rare classes; candidate losses (squared, absolute, Tweedie, hurdle) and a Y transform, all as candidates; the complexity budget. **How:** nested CV scores the candidates. Class weights, target encodings and decision thresholds are fit in each fold. The metric is declared from the use case, not from the data. **Source:** Varma & Simon 2006; Cawley & Talbot 2010. |
| **Predictor–outcome:** scatterplots, outcome-colored plots, group comparisons, univariate screens | **Informs:** nothing by eye. **How:** only a declared selection algorithm whose inference accounts for the selection. Options are post-double-selection with the exposure unpenalized, sample splitting, a bootstrap that repeats the selection, or Harrell's masked association index to allocate degrees of freedom. **Source:** Baillie 2022 rule 5; Gelman & Loken 2013; Belloni et al. 2014; Sauerbrei et al. 2020; Harrell RMS 4.1. | **Informs:** the reported results and analyzes labeled exploratory. New hypotheses go to another dataset or a held-out split. **How:** disclosure only. **Source:** Baillie 2022 (HARKing); Gelman & Loken 2013. | **Informs:** extra candidates (features, interactions, model families) added to a declared default list. It never prunes that list. **How:** supervised selection is refit in each fold, under nested CV or BBC-CV. The score is labeled as the performance of the whole procedure. Views are drawn on development rows only. **Source:** Ambroise & McLachlan 2002; ESL 7.10.2; Varma & Simon 2006; Tsamardinos et al. 2018. |
| **Repeated measures and time:** per-person series, within- vs between-person spread, Poincaré plots, autocorrelation | **Informs:** usual-intake and measurement-error models; per-person summaries used as exposures (CV, SD, time in range); data-sufficiency rules. Clustering comes from the design. Mixed model versus GEE is a choice of estimand. **How:** a declared choice. A variability exposure is a new estimand, so its window, sufficiency rule and software implementation are declared. **Source:** Tooze 2006; Hubbard 2010; Battelino 2019. | **Informs:** checks of residual autocorrelation and wear time. **How:** disclosure plus prespecified sensitivity analyzes. **Source:** Roberts 2017. | **Informs:** per-person features computed within each row, and the split design. **How:** features use only the window before the prediction time. Folds are subject-wise (grouped); forecasting uses time-ordered splits. Normalizations across the cohort are fit in each fold. Correlated variability metrics enter as candidates. **Source:** Saeb 2017; Roberts 2017; Bergmeir & Benítez 2012; Kaufman 2012. |
| **Dimension-reduction maps:** scree, PCA scores and loadings, UMAP, t-SNE | **Informs:** data reduction blind to Y, such as dietary patterns or cluster scores. The number of components comes from a declared rule (parallel analysis or MAP, not eigenvalue > 1), and the choice of covariance or correlation matrix is declared. UMAP and t-SNE only propose QC checks. **How:** a declared choice; QC proposals are confirmed in the full space. **Source:** Harrell RMS 4.7; Zwick & Velicer 1986; Jolliffe & Cadima 2016; Tran 2020; Broadhurst 2018. | **Informs:** interpretation of the loadings. An outcome-colored embedding counts as a predictor–outcome view. **How:** disclosure only. **Source:** Lause et al. 2024. | **Informs:** a reduced representation, as one candidate. **How:** the number of components k is tuned in nested CV. Loadings, scaling and the UMAP graph are refit in each fold. The reduced model is compared against an unreduced learner that does not favor high-variance directions (lasso, or the unreduced model). t-SNE is never a model input. **Source:** Jolliffe 1982; Hadi & Ling 1998; ESL 3.4.1 and 3.5; Hornung 2015; scikit-learn TSNE API. |

---

## 3. The owner's four examples

### 3.1 A relative-abundance plot informing feature selection
**Verdict:** half right. The plot is outcome-blind, and it legitimately decides the representation. It does not legitimately select features for the outcome.
- **It may decide:** the log-ratio coordinates, the zero-handling rule, and an overall-prevalence filter with a threshold fixed in advance.
- **Picking features that look different between outcome groups is supervised selection.** Under prediction it goes inside each fold. Under inference it needs a declared differential-abundance method with multiplicity control, or selection-aware inference.
- **Per-group prevalence filters only look outcome-blind.** Examples are the "modified 80% rule" and "present in X% of at least one group". Bourgon et al. flag exactly this filter.

**What the app should do:**
- Show the relative-abundance plot pooled by default. Under inference before the lock, do not split it by outcome.
- Offer the filter only as overall prevalence or abundance with a fixed threshold. Never offer a threshold tuned to maximize discoveries (Ignatiadis 2016).
- Pick the representation by learner:
  - CLR or ILR for correlations, ordination, and linear, kernel or distance learners.
  - Proportions and presence–absence as resampled candidates for tree ensembles. They match or beat CLR there (Yerke 2024; Garach Vélez 2025).
- Block CLR in an unpenalized GLM with an intercept. The design is singular, so use penalization, a zero-sum log-contrast model, or ALR/ILR.
- Fix the zero rule in advance and run a sensitivity check on it. Separate structural zeros from sampling zeros.
- Show a check of library size against outcome group (Weiss 2017).
- Offer absolute profiling when microbial load is measured.

### 3.2 EDA cueing the inductive bias
**Verdict:** holds for the cues that really are outcome-blind:
- p much larger than n -> penalization (under prediction);
- skew or outliers in predictors -> a monotone transform, or trees;
- a spike at zero in a predictor -> an indicator plus a model of the positive part;
- design clustering -> grouped CV, and a mixed model or GEE chosen by estimand;
- compositional measurement -> log-ratio models.

**Corrections:**
- **Robust losses and hurdle models are cued by Y, not X.** Zero inflation must be judged given the fitted mean ("Many zeros does not mean zero inflation", Warton 2005). Under inference, the loss is the estimand.
- **Collinearity is a weak cue for prediction.** Predictions are unaffected if new data share the collinearity (Harrell; Leeuwenberg). Under inference, it is a question for the causal diagram, not a reason to penalize the exposure (Schisterman 2017).

**What the app should do:**
- Present cues as cards that propose candidate families, each with its reason.
- Under prediction, nested CV runs the proposed candidates.
- Under inference, the cards fill in the plan before the lock and are then frozen.

### 3.3 A Poincaré plot guiding feature engineering
**Verdict:** holds with a correction. Per-person summaries of a person's own series are engineered features that need no outcome. But the Poincaré plot adds less than it seems.

**The corrections:**
- **SD1 and SD2 are linear statistics.** They are functions of the series' lag-0 and lag-1 autocovariance. Brennan, Palaniswami & Kamen (IEEE TBME 2001) showed that Poincaré geometry measures linear aspects "which existing HRV indexes already specify".
- **For continuous glucose monitoring (CGM), SD1 duplicates an existing metric.** Crenier (2014) found SD1 "was equivalent to continuous overlapping net glycemic action (CONGA)". SD2 tracks long-term variability, alongside SD, MODD and MAGE.
- **They are row-local only within a window that ends before the prediction or index time.** A later window is temporal leakage (Kaufman 2012).
- **Fix a data-sufficiency rule and a standard duration.** The consensus asks for "14 consecutive days with at least 70% wear" (Battelino 2019). Variability measures depend on recording length.
- **SD1 depends on the lag, so it depends on the sensor's sampling interval.** This follows from its lag-1 definition.
- **MAGE depends on the implementation.** Median errors ranged from 1% to 78% across calculators (Fernandes 2022), so the implementation must be pinned.
- **Under inference, a variability exposure is a new estimand.** SD scales with mean glucose. CV divides it out, and %CV 36% separates stable from unstable glycemia (Monnier 2017).
- **With many rows per person, cross-validation must be subject-wise** (Saeb 2017).

**What the app should do:**
- Compute a declared core set: mean glucose, CV and time in range.
- Add SD1, SD2, MAGE and CONGA as candidates that the resampling tests.
- Warn when a series fails the sufficiency rule.
- Where the data are repeated 24-hour recalls rather than a dense series, the analogous view is within- versus between-person spread. That view chooses the NCI usual-intake model.

### 3.4 PCA or UMAP suggesting reduced dimensionality
**Verdict:** a PCA scree plot may propose a reduced representation. It cannot show that a reduced model will predict better. UMAP and t-SNE cannot show it either.

**The reasons:**
- **Low-variance components can carry the whole signal** (Jolliffe 1982; Hadi & Ling 1998).
- **A scree plot does not show intrinsic dimension.** It shows how variance concentrates under the scaling you chose.
- **Ridge and PLS make the same high-variance assumption** (ESL 3.4.1, 3.5).
- **UMAP and t-SNE distort cluster sizes and distances.** Read by eye, they are QC proposals only (Wattenberg 2016; UMAP documentation; Lause 2024; Tran 2020).
- **t-SNE has no out-of-sample transform**, so it cannot be a model input.

**What the app should do:**
- Under prediction:
  - Treat k as a tuning parameter in nested CV.
  - Refit loadings in each fold.
  - Always include an unreduced comparator.
- Under inference:
  - PCA or factor-analysis dietary patterns are legitimate exposures if they are fixed before the outcome is seen.
  - Set the number of components by parallel analysis or MAP.
- Confirm any UMAP-flagged batch or outlier with PCA scores or a metric in the full space.

---

## 4. Extra examples worth building into the lens packs

1. **Pooled-QC precision filter (untargeted metabolomics).**
   - **View:** QC-only RSD, detection rate and D-ratio, plus a PCA of QC injections against study samples, colored by run order.
   - **Decision:** drop features with QC RSD above 20% (LC-MS) or 30% (GC-MS), D-ratio above 50%, or QC detection below 70%; correct drift.
   - **Leakage:** RSD uses only QC injections, so it cannot leak. The D-ratio uses study-sample variance, so it is fit in each fold.
   - **Source:** Broadhurst 2018.
2. **Missingness against intensity.**
   - **View:** missingness that piles up at low intensity means left-censored data, missing not at random.
   - **Decision:** censoring-aware imputation (QRILC or half-minimum) for those features, random forest for MCAR/MAR features. Imputation is fit in each fold.
   - **Filter:** the original 80% rule (across all samples) is outcome-blind. The modified per-class rule is not.
   - **Source:** Wei 2018; Yang 2015.
3. **Spike at zero in 24-hour recall intakes.**
   - **Rule:** if more than 5% of people report zero intake, the food is episodic and gets the NCI two-part usual-intake model.
   - **Not chosen by AIC.**
   - **Source:** NCHS Series 2 No. 178; Tooze 2006.
4. **Spike at zero in a dietary exposure** (alcohol, supplements). Code a binary exposed indicator plus a function of the positive part. **Source:** Lorenz 2017; Becher 2012; Heinze 2024.
5. **Knot count and placement from the exposure's marginal distribution.**
   - Fixed quantiles; k=3 for n<30, k=5 for n≥100.
   - Computed in each fold under prediction, fixed at the lock under inference.
   - **Source:** Harrell RMS 2.4.6; Heinze 2024.
6. **Correlation heatmap or variable-clustering dendrogram leading to data reduction blind to Y.**
   - Dietary patterns or cluster scores.
   - Retention by parallel analysis or MAP, not Kaiser's eigenvalue > 1. Kaiser's rule is customary in nutrition but "tended to severely overestimate" the number of components.
   - **Source:** Harrell RMS 4.7; Hu 2002; Zwick & Velicer 1986; Jannasch 2018.
7. **Library size against outcome group (sequencing).**
   - A design-balance check that decides the normalization, never the features.
   - About a 10x imbalance breaks FDR control of proportion-based tests.
   - **Source:** Weiss 2017.
8. **Exposure–covariate overlap (propensity) view under inference.**
   - Done before any outcome is seen. It decides trimming or matching.
   - Rubin: observational studies must be designed "in particular, without examining any final outcome data".
   - **Source:** Rubin 2008.
9. **Dietary composition (macronutrient shares).**
   - ALR, CLR, ILR and substitution parametrizations answer different questions.
   - Choose by estimand, not by fit.
   - **Source:** Leite 2021; Arnold 2020; Roosdorp 2026.
10. **Leak meter.**
    - Run the pipeline with preprocessing outside and inside the folds.
    - Report CVIIM and the per-fold overlap of retained features.
    - **Source:** Hornung 2015; Moscovich & Rosset 2022.
11. **CGM variability panel.**
    - A declared core (mean, CV, time in range), a sufficiency gate (14 days, ≥70% wear), and the remaining variability metrics as candidates.
    - **Source:** Battelino 2019; Monnier 2017; Crenier 2014; Fernandes 2022.

---

## 5. Every correction to the original framing

1. **A declared rule is not enough.** It must also be rerun wherever the data are resampled, and its uncertainty must be reported. Grambsch & O'Brien's fixed, declared rule (a preliminary nonlinearity test) still raised type I error "by roughly 50 per cent". The remedies are to prespecify a flexible form and keep it, or to propagate the rule by bootstrap.
2. **"Outcome-blind" was drawn too wide.** These are not outcome-blind:
   - per-group prevalence filters;
   - thresholds tuned to maximize discoveries;
   - any outcome-colored plot or embedding;
   - a Y histogram used to choose a robust loss or a hurdle model.
3. **Outcome-blind does not mean leak-free under prediction.** ESL 7.10.2's exception is contradicted by Moscovich & Rosset (bias "may be either positive or negative" and grows with p) and by Hornung (PCA before CV gives "medium to strong" optimism). Refit learned steps in each fold. Keep the holdout out of every EDA view.
4. **Bourgon et al. guarantee less than claimed.** The guarantee covers only specific filter–test pairs. A variance filter followed by limma breaks it. Covariate-adjusted models are not covered. Filtering can change the correlation among p-values.
5. **The ban on predictor–outcome selection under inference was too absolute.** Selection by eye is barred. A declared selection-aware method is valid: post-double-selection, sample splitting, or a bootstrap that repeats the selection.
6. **Outcome-alone views were missing as a category.** They may inform data quality and the degrees-of-freedom budget. They may not choose the loss or the estimand.
7. **Under inference, "fit learned parts in each fold" does not apply.** Steps learned without the outcome can be fitted on the full analysis sample. They must be declared, and exposure transforms define the estimand.
8. **Under inference, a robust loss is a different estimand** (median versus mean). Zero inflation must be judged given X.
9. **Collinearity is a weak cue for prediction and a causal-diagram question for inference.** Exposure–confounder collinearity does not bias a correctly specified model (Schisterman 2017).
10. **Repeated measures and compositionality are design facts, not exploratory findings.** Mixed model versus GEE is an estimand choice. Prediction needs grouped or subject-wise CV.
11. **Log-ratios are not universally required.**
    - For tree ensembles, proportions match or beat CLR.
    - Zero handling must be fixed in advance and checked for sensitivity.
    - CLR in an unpenalized GLM with an intercept is singular.
12. **A scree plot shows variance concentration, not intrinsic dimension.**
    - Ridge and PLS share the high-variance assumption, so the comparison needs an unreduced comparator.
    - k is tuned, not read off the plot.
13. **The UMAP and t-SNE evidence needs care.**
    - Wattenberg et al. cover only t-SNE.
    - Cite Lause 2024 alongside Chari & Pachter.
    - A QC finding by eye is a proposal.
    - t-SNE cannot be a model input.
14. **Under prediction, exploration may add candidates, never prune them.**
    - It must be drawn on development rows only.
    - Nested CV estimates the performance of the procedure, not of the family it chose.
15. **The functional form of an exposure** is prespecified and kept, or chosen by a rule whose uncertainty is propagated. Simplifying a spline after a linearity check is the customary hazard in nutrition.
16. **Poincaré features** (my own check, since the verifier's result was cut off):
    - SD1 and SD2 are linear statistics, not nonlinear ones, and largely duplicate CONGA and SD.
    - They are computed only on a window before the index time, with a sufficiency rule and a standard duration.
    - Pin the MAGE implementation.

---

## Sources added for the time-series row (verified this session)

- Brennan M, Palaniswami M, Kamen P. Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability? IEEE Trans Biomed Eng 2001;48:1342-7. https://www.semanticscholar.org/paper/06929de43a205a1f079b774cd2458c0d5ca0a181
- Karmakar CK et al. Complex Correlation Measure: a novel descriptor for Poincaré plot. Biomed Eng Online 2009;8:17. https://pmc.ncbi.nlm.nih.gov/articles/PMC2743693/ ("SD1 and SD2 are second order statistical measures")
- Crenier L. Poincaré plot quantification for assessing glucose variability from CGM. Diabetes Technol Ther 2014;16:247-54. https://pubmed.ncbi.nlm.nih.gov/24237387/
- Fernandes NJ et al. Open-source algorithm to calculate MAGE. J Diabetes Sci Technol 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC8861796 ("these programs have shown varying degrees of agreement")
- Battelino T et al. Clinical targets for CGM data interpretation. Diabetes Care 2019;42:1593-1603. https://pmc.ncbi.nlm.nih.gov/articles/PMC6973648/ (wear rule as quoted in https://pmc.ncbi.nlm.nih.gov/articles/PMC11965008/)
- Monnier L et al. Toward defining the threshold between low and high glucose variability in diabetes. Diabetes Care 2017;40:832. https://diabetesjournals.org/care/article/40/7/832/30254
- HRV measurement and influencing factors: towards the standardization of methodology (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11439429/ ("researchers should exclusively conduct experiments with the same recording duration")
- Saeb S et al. The need to approximate the use-case in clinical machine learning. GigaScience 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/ ("record-wise CV often massively overestimates the prediction accuracy")
- Bergmeir C, Benítez JM. On the use of cross-validation for time series predictor evaluation. Inf Sci 2012;191:192-213. https://www.sciencedirect.com/science/article/abs/pii/S0020025511006773
- Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining. ACM TKDD 2012;6(4):15. https://dl.acm.org/doi/10.1145/2382577.2382579
- Baillie M et al. Ten simple rules for initial data analysis. PLoS Comput Biol 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC8870512/
- Heinze G et al. Regression without regrets. BMC Med Res Methodol 2024;24:178. https://pmc.ncbi.nlm.nih.gov/articles/PMC11308558/
- Rubin DB. For objective causal inference, design trumps analysis. Ann Appl Stat 2008;2:808-40. https://arxiv.org/abs/0811.1640

All other citations are as given, with URLs, in the filtering, inductive-bias and dimension verdicts this synthesis was built from.