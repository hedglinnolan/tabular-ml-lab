# Provenance Audit: Modeling Decision Capture & Reporting

## Audit Methodology
Traced every user decision across all 10 workflow pages. For each decision:
- **Captured?** Is it stored in session state or methodology log?
- **Reported?** Does it appear in the methods/results markdown or LaTeX output?
- **Specific?** Are the exact parameter values preserved, or just a generic description?

Legend: ✅ = fully captured and reported with specifics | ⚠️ = captured but not reported (or reported generically) | ❌ = not captured at all

---

## Page 1: Upload & Audit

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Dataset file name/source | ✅ `data_source` in log | ⚠️ | No | Log has it, methods section doesn't mention data source |
| Target column selection | ✅ log + session state | ✅ | Yes | "The modeled outcome was {target}" |
| Task type (regression/classification) | ✅ log + session state | ✅ | Yes | |
| Feature column selection | ✅ log + session state | ✅ | Yes | Count reported, full list in supplementary |
| Data cleaning actions (drop rows/cols, dedup) | ✅ log with before/after counts | ❌ | N/A | Cleaning steps logged but NEVER reported in methods. A reviewer would want: "X rows were excluded due to..." |
| Multi-file merging | ⚠️ session state only | ❌ | N/A | If user merged files, the merge strategy/keys aren't in the methods |
| Sample size after cleaning | ✅ in log | ✅ | Yes | n_total reported |
| Column type overrides | ❌ | ❌ | N/A | User can force numeric→categorical but this isn't logged |

### Gap: Data cleaning provenance is logged but never reaches the report. If a user drops 5000 rows of missing data, the methods should say so.

---

## Page 2: EDA

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which EDA analyses were run | ✅ log | ❌ | N/A | EDA actions logged but not reported (appropriate — EDA is exploratory) |
| Table 1 generation | ✅ log | ✅ | Yes | Table 1 appears in both LaTeX and markdown |
| Table 1 stratification variable | ⚠️ session state | ⚠️ | No | If stratified Table 1 was generated, the stratification variable isn't in the methods |
| Outlier/distribution observations | ❌ | ❌ | N/A | EDA findings that inform preprocessing choices aren't linked to the methods |

### Gap: Table 1 stratification variable should be in the methods when stratified Table 1 is exported.

---

## Page 3: Feature Engineering

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which transforms applied | ✅ `engineering_log` in session state + methodology log | ✅ | Partially | "PCA (+10 features)" but see specifics below |
| PCA: n_components / variance threshold | ⚠️ in per-model config | ⚠️ | Partially | Fixed vs variance-threshold is now distinguished, but the specific variance explained per component isn't reported |
| Polynomial features: degree, interaction_only | ⚠️ in engineering_log | ⚠️ | No | Log has details but methods just says "polynomial features" |
| Log/sqrt transforms: which columns | ✅ engineering_log | ⚠️ | No | Methods says "log transforms" but not which specific columns |
| Binning: strategy, n_bins, which columns | ⚠️ engineering_log | ⚠️ | No | Generic mention only |
| Ratio features: which pairs | ⚠️ engineering_log | ⚠️ | No | Not listed specifically |
| TDA features: parameters | ⚠️ engineering_log | ❌ | No | TDA parameters not in methods |
| Total engineered feature count | ✅ | ✅ | Yes | |
| Which engineered features survived selection | ⚠️ | ❌ | No | Interesting provenance: did PCA features actually help? Not tracked |

### Gap: Feature engineering is reported generically. The methods should list specific transforms with their parameters — "log transform applied to {columns}, polynomial features (degree=2) created for {columns}, PCA with 10 fixed components."

---

## Page 4: Feature Selection

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Methods used (LASSO, RFE, etc.) | ✅ log | ✅ | Yes | |
| Features before/after | ✅ log | ✅ | Yes | |
| Consensus threshold | ✅ log (newly added) | ✅ | Yes | "Features retained if selected by at least N of M methods" |
| Which features were selected | ✅ log | ⚠️ | No | Full list in log but not in methods (only count). Should reference supplementary table |
| Which features were DROPPED | ⚠️ derivable | ❌ | No | Dropped features are never listed — useful for transparency |
| Per-method rankings | ✅ session state | ❌ | No | Individual method results (LASSO coefficients, RFE ranking) not in report |
| Manual override of consensus | ✅ log (separate entry) | ⚠️ | No | Logged as "manual selection" but methods doesn't distinguish manual vs consensus |

### Gap: The specific features selected (and dropped) should be listed or referenced. Manual vs consensus selection should be stated.

---

## Page 5: Preprocessing

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Imputation method | ✅ log + per-model config | ✅ | Yes | "median imputation" |
| Missing indicators | ✅ per-model config | ✅ | Yes | "Binary indicator variables were added" |
| Scaling method per model | ✅ per-model config | ✅ | Yes | "z-score standardization", "robust scaling" |
| Outlier treatment per model | ✅ per-model config | ✅ | Yes | NOW specific: "clipped at 6th and 94th percentiles" |
| Power transform per model | ✅ per-model config | ✅ | Yes | "yeo-johnson power transform" |
| Encoding method | ✅ per-model config | ✅ | Yes | "one-hot encoding" |
| PCA in preprocessing (not FE) | ✅ per-model config | ⚠️ | Partially | Mentioned but per-model PCA specifics may be lost |
| K-means features | ⚠️ per-model config | ❌ | No | If enabled, n_clusters and strategy not in methods |
| Which features got which treatment | ⚠️ in pipeline recipe | ❌ | No | Per-feature preprocessing assignments not in methods |
| Preprocessing order of operations | ⚠️ implicit in pipeline | ❌ | No | The sequence (impute → scale → encode → outlier) isn't stated |
| Numeric outlier params (per-model) | ✅ per-model config | ✅ | Yes | Newly fixed |
| Missing data rates | ✅ computable from data | ✅ | Yes | Newly added: "X of Y features had missing values" |

### Gap: Preprocessing order of operations should be explicit. K-means feature engineering in preprocessing is a hidden decision. Per-feature treatment assignments could go in supplementary.

---

## Page 6: Train & Compare

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which models trained | ✅ log | ✅ | Yes | |
| Train/val/test split ratio | ✅ split_config | ✅ | Yes | |
| Split strategy (random/stratified/chronological) | ✅ split_config | ✅ | Yes | Newly added |
| Group-based splitting | ✅ split_config | ⚠️ | Partially | `use_group_split` tracked but group column name not in methods |
| Random seed | ✅ session state | ✅ | Yes | |
| Cross-validation folds | ✅ log | ⚠️ | Partially | CV folds logged but not always in methods text |
| Hyperparameter optimization (Optuna) | ✅ log | ✅ | Yes | Newly added |
| Optuna n_trials | ⚠️ hardcoded 30 in code | ⚠️ | No | Always says "30 trials" but user might change it |
| Hyperparameters per model | ✅ trained model objects | ✅ | Yes | Newly added — extracted from model.get_params() |
| Which metrics used for comparison | ✅ session state | ✅ | Yes | |
| Best model selection criterion | ✅ session state | ✅ | Yes | "best by RMSE" |
| Baseline models included | ⚠️ implicit | ❌ | No | App auto-generates baselines but methods doesn't mention them explicitly |
| Training time per model | ⚠️ displayed in UI | ❌ | No | Not logged or reported |
| Class weights (for classification) | ⚠️ in model config | ❌ | No | If class_weight='balanced' was used, not reported |
| Early stopping (NN) | ⚠️ in model params | ❌ | No | NN early stopping criteria not in methods |

### Gap: Group splitting column, cross-validation details, baseline comparison, class weights, and early stopping should be reported. Optuna n_trials should be dynamic not hardcoded.

---

## Page 7: Explainability

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which analyses run (SHAP/permutation/PDP) | ✅ log | ✅ | Yes | "Permutation importance... SHAP values..." |
| Which models analyzed | ✅ log | ⚠️ | No | Log has model list but methods doesn't say "SHAP was computed for LASSO, HISTGB_REG, NN" |
| Top features by importance | ✅ session state | ✅ | Yes | In LaTeX Results section |
| SHAP interaction effects | ⚠️ computed in UI | ❌ | No | If user explored interactions, not captured |
| Number of SHAP evaluations | ⚠️ hardcoded | ❌ | No | Sample size for SHAP not reported |
| PDP: which features, interactions | ⚠️ session state | ❌ | No | PDP results exist but don't reach the report |
| Calibration metrics | ⚠️ session state | ⚠️ | Partially | Calibration placeholder exists but actual metrics not auto-filled |
| Decision curve analysis | ⚠️ session state | ❌ | No | DCA results not in methods/results |
| Subgroup analysis results | ⚠️ session state | ❌ | No | Subgroup findings not in report |

### Gap: Which models were analyzed for explainability should be stated. SHAP sample size matters methodologically. PDP, DCA, and subgroup results should flow into the report when available.

---

## Page 8: Sensitivity Analysis

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Seed stability: which model | ✅ log | ✅ | Yes | |
| Seed stability: n_seeds | ✅ log | ✅ | Yes | |
| Seed stability: which metric | ✅ log | ✅ | Yes | |
| Seed stability: actual results (CV%, range) | ✅ session state | ✅ | Yes | In methods + LaTeX Results |
| Feature dropout: which model | ✅ log | ✅ | Yes | |
| Feature dropout: n_features tested | ✅ log | ✅ | Yes | |
| Feature dropout: actual results | ✅ session state | ⚠️ | Partially | Summary in LaTeX but not the full ranking |
| Feature dropout: which metric | ✅ log | ✅ | Yes | |
| Multiple seed runs (different models) | ⚠️ only last run per model | ⚠️ | Partially | If user ran seed analysis on multiple models, only the last per model is in the log |

### Gap: Sensitivity is relatively well-captured. Feature dropout full results (which features hurt most when removed) could go in supplementary.

---

## Page 9: Statistical Validation

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which tests run | ✅ log | ⚠️ | Partially | Custom tests logged but only appear as Table 1 footnotes, not in methods prose |
| Test results (statistic, p-value) | ✅ log + session state | ⚠️ | Partially | In Table 1 footnotes but not in a dedicated Statistical Validation results subsection |
| Variables tested | ✅ log | ⚠️ | Partially | Variable names in footnotes |
| Correlation tests | ✅ log | ❌ | No | Correlation test results not in report |
| Multiple testing correction | ❌ | ❌ | No | If user ran 10 tests, no Bonferroni/FDR correction is mentioned or applied |

### Gap: Statistical validation results should have their own results subsection. Multiple testing correction is a methodological gap.

---

## Page 10: Report Export

This page CONSUMES provenance — it doesn't create new decisions. But there are export-level decisions:

| Decision | Captured | Reported | Specific | Notes |
|----------|----------|----------|----------|-------|
| Which models included in report | ✅ manuscript_context | ✅ | Yes | |
| Manuscript primary model | ✅ manuscript_context | ✅ | Yes | |
| Which explainability methods to include | ✅ manuscript_context | ✅ | Yes | |
| Paper title/authors/affiliation | ✅ UI inputs | ✅ | Yes | |
| Whether user edited methods text | ❌ | ❌ | No | If user manually edited, we lose track of what changed |

---

## Cross-Cutting Gaps

### 1. Data Cleaning Provenance → Methods (HIGH PRIORITY)
Data cleaning actions (row drops, column drops, deduplication) are logged on Page 1 but never reach the methods section. Any reviewer would ask: "How did you go from N_raw to N_final observations?"

**Fix:** Add a "Data Cleaning" or "Participant Flow" subsection to methods that lists exclusion criteria and counts. Reference the CONSORT flow diagram if generated.

### 2. Feature Engineering Specificity (MEDIUM PRIORITY)
Feature engineering is reported as "PCA (+10 features)" instead of "PCA dimensionality reduction (10 fixed components explaining X% of variance) applied to {list of numeric features}; log transform applied to {columns}; polynomial interaction terms (degree 2) created for {columns}."

**Fix:** Parse `engineering_log` entries more deeply and report specific transforms with their target columns and parameters.

### 3. Explainability Scope (MEDIUM PRIORITY)  
Methods says "SHAP values were computed" but doesn't say for which models or with what sample size. This matters because SHAP on 100 samples vs 10,000 is methodologically different.

**Fix:** Add model list and sample size to the interpretability subsection.

### 4. Preprocessing Order of Operations (LOW-MEDIUM PRIORITY)
The pipeline applies transforms in a specific order (impute → scale → encode → outlier clip for some models, different for others). This isn't stated. A methodological reviewer might care.

**Fix:** Add a brief statement: "Preprocessing was applied in the following order: ..." derived from the pipeline recipe.

### 5. Statistical Validation Results Missing from Report (MEDIUM PRIORITY)
Custom hypothesis tests are only visible as Table 1 footnotes. Correlation tests don't appear at all. If a user ran 10 statistical tests, there should be a results subsection and a note about multiple testing.

**Fix:** Add a "Statistical Validation" subsection to Results. Include all test results. Add a note about multiple comparisons when >3 tests are run.

### 6. Baseline Model Reporting (LOW PRIORITY)
The app auto-generates baseline models (mean predictor, simple linear) for comparison, but the methods section never mentions this. Reporting that "all models were compared against a mean baseline" is good practice.

**Fix:** Add baseline comparison to Model Development subsection.

### 7. Session-Level Provenance Metadata (LOW PRIORITY)
There's no single manifest that captures the full decision chain in order: "User uploaded file X at time T1, configured target Y, dropped 500 rows, ran EDA, engineered 10 features, selected 23 via consensus, preprocessed with {configs}, trained 3 models, ran SHAP on all 3, ran seed stability on 2, exported report." The methodology log has pieces but not a clean narrative.

**Fix:** Generate a "Decision Audit Trail" appendix from the methodology log, ordered chronologically.

---

## Priority Ranking for Improvements

1. **Data cleaning → methods** (reviewer would flag this immediately)
2. **Feature engineering specificity** (parameters and target columns)
3. **Explainability scope** (which models, sample sizes)
4. **Statistical validation results section** (with multiple testing note)
5. **Preprocessing order of operations**
6. **Baseline model reporting**
7. **Decision audit trail appendix**
