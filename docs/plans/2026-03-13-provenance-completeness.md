# Provenance Completeness — Systematic Execution Plan

## Guiding Principle
Prioritize capturing and reporting decisions that ALREADY EXIST in session state or methodology log but don't reach the report. No new UI features. No new logging infrastructure. Just threading existing data through to the output.

## Execution Order
Each fix is independent and testable. Complete one, verify, move to next.

---

## Fix 1: Data Cleaning → Methods Section
**Priority:** Highest (reviewer would flag immediately)
**What exists:** `methodology_log` entries with `step='Data Cleaning'` containing `action`, `rows_before`, `rows_after`, `cols_before`, `cols_after`, `affected_columns`
**What's missing:** No mention in methods section of exclusions or cleaning steps

### Implementation
**File:** `ml/publication.py` → `generate_methods_section()`
- After "Study Design and Participants" subsection, before "Predictor Variables"
- Read `Data Cleaning` entries from `logged_steps`
- If any exist, add prose: "Prior to analysis, the following data cleaning steps were applied: {actions}. This resulted in {rows_before - rows_after} observations being excluded, yielding a final sample of {n_total}."
- If no cleaning entries, don't add anything (clean pass-through)

### No wiring needed
`generate_methods_section()` already calls `generate_methods_from_log()` which reads all methodology log entries including Data Cleaning.

### Verify
- py_compile ml/publication.py
- pytest tests/test_publication.py

---

## Fix 2: Feature Engineering Specificity
**Priority:** High
**What exists:** `engineering_log` in session state — list of dicts with keys like `{'type': 'pca', 'n_components': 10, 'variance_explained': 1.0}`, `{'type': 'log', 'columns': ['kcal', 'sugar']}`, `{'type': 'polynomial', 'degree': 2, 'columns': [...]}` etc.
**What's missing:** Methods says "PCA (+10 features)" generically. Doesn't list target columns or specific parameters per transform.

### Implementation
**File:** `ml/publication.py` → Feature Engineering subsection (around line ~450)
- Already reads `engineering_log` from session state
- Replace generic "The following transformations were applied: {summary}" with specific per-transform descriptions
- For each entry in engineering_log:
  - PCA: "PCA dimensionality reduction ({n_components} components, {variance*100:.0f}% variance retained)"
  - Log: "Log transform applied to {', '.join(columns)}"
  - Sqrt: "Square root transform applied to {', '.join(columns)}"
  - Polynomial: "Polynomial features (degree {degree}) created from {', '.join(columns)}"
  - Ratio: "Ratio features created: {list of ratios}"
  - Binning: "Binning ({strategy}, {n_bins} bins) applied to {', '.join(columns)}"
  - TDA: "Topological features (persistence diagrams) computed with {params}"
- Keep the existing total count line at the end

### Verify
- py_compile ml/publication.py
- pytest tests/test_publication.py

---

## Fix 3: Explainability Scope in Methods
**Priority:** High
**What exists:** `methodology_log` entry with `step='Explainability'`, `details={'analyses': [...], 'models': [...]}`. Also `permutation_importance` and `shap_results` dicts in session state keyed by model name.
**What's missing:** Methods says "Permutation importance was computed..." but not for which models or with what sample size.

### Implementation
**File:** `ml/publication.py` → Model Interpretability subsection
- Already has prose about permutation importance and SHAP
- Add model list: "Permutation importance was computed for {', '.join(models)}"
- Add sample size if available: "using {n} test observations"
- For SHAP: "SHAP values were computed for {', '.join(models)} using {n} samples"
- Read from `logged_steps.get('Explainability', [])` for model list
- Read `X_test` shape for sample size (already available as `n_test` param)

### Verify
- py_compile ml/publication.py  
- pytest tests/test_publication.py

---

## Fix 4: Statistical Validation Results Section
**Priority:** Medium-High
**What exists:** `methodology_log` entries with `step='Statistical Validation'` containing test names, statistics, p-values, variables tested. Also `custom_table1_tests` in session state.
**What's missing:** No "Statistical Validation" subsection in Results. Tests only appear as Table 1 footnotes.

### Implementation
**File:** `ml/publication.py` → add to Results section generation (after sensitivity, before discussion scaffolding)
- New subsection: "### Statistical Validation" (only if stat val log entries exist)
- List each test: "{test_name} was performed on {variable}: statistic={value}, p={p_value}"
- If >3 tests: add note "Multiple statistical tests were performed; readers should consider the increased risk of Type I error when interpreting individual p-values."
- Read from `logged_steps.get('Statistical Validation', [])`

**File:** `ml/latex_report.py` → add corresponding LaTeX subsection
- After sensitivity results, before Discussion
- Same content as markdown, formatted as LaTeX

### Verify
- py_compile ml/publication.py ml/latex_report.py
- pytest tests/test_publication.py

---

## Fix 5: Baseline Model Reporting
**Priority:** Medium
**What exists:** Baseline models are auto-generated during training. Results exist in `model_results` with keys like `baseline_mean`, `baseline_linear`.
**What's missing:** Methods doesn't mention baseline comparison.

### Implementation
**File:** `ml/publication.py` → Model Development subsection
- After listing model candidates, check if any model name contains "baseline"
- If yes: "Baseline models (mean predictor and simple linear regression) were automatically generated for comparison."
- For classification: "Baseline models (majority class predictor and simple logistic regression) were automatically generated for comparison."

### Verify
- py_compile ml/publication.py
- pytest tests/test_publication.py

---

## Fix 6: Preprocessing Order of Operations
**Priority:** Medium
**What exists:** Pipeline recipe string available via `get_pipeline_recipe()`. Per-model configs in `preprocessing_config_by_model`.
**What's missing:** Methods doesn't state the order of operations.

### Implementation
**File:** `ml/publication.py` → Data Preprocessing subsection
- After listing per-model preprocessing, add: "For all models, preprocessing was applied in the following order: missing value imputation, feature scaling, categorical encoding{, outlier treatment where applicable}{, power transformation where applicable}."
- Derive order from the pipeline structure — it's consistent across models (impute → scale → encode → outlier → transform)

### Verify
- py_compile ml/publication.py
- pytest tests/test_publication.py

---

## Fix 7: Decision Audit Trail Appendix
**Priority:** Lower (but high value for reproducibility)
**What exists:** Full `methodology_log` in session state with timestamps and ordered entries.
**What's missing:** No chronological summary of all decisions.

### Implementation
**File:** `ml/publication.py` → new function `generate_decision_audit_trail()`
- Read all entries from methodology log, sorted chronologically
- Format as numbered list: "1. [Upload & Audit] Configured regression task with 25 features, target: glucose (N=21,849). 2. [EDA] Generated Table 1. 3. [Feature Engineering] Created 10 engineered features. ..."
- Return as markdown string

**File:** `pages/10_Report_Export.py` → add to markdown report as appendix
- Add "## Appendix: Decision Audit Trail" section at end of markdown report
- Call `generate_decision_audit_trail()`

**File:** `ml/latex_report.py` → add to LaTeX supplementary
- Add `\subsection{Decision Audit Trail}` in Supplementary Material
- Same content as markdown

### Verify
- py_compile on all three files
- pytest tests/test_publication.py tests/test_page_imports.py

---

## What this plan does NOT do
- No new UI features or pages
- No changes to session state shape
- No new logging calls on workflow pages (except possibly reading existing log entries more carefully)
- No changes to model training, preprocessing, or evaluation logic
- No changes to upstream workflow behavior

## Files touched
- `ml/publication.py` (Fixes 1-7)
- `ml/latex_report.py` (Fix 4, Fix 7)
- `pages/10_Report_Export.py` (Fix 7)

## Verification after all fixes
```bash
python3 -m py_compile ml/publication.py ml/latex_report.py pages/10_Report_Export.py
./venv/bin/python -m pytest -q tests/test_publication.py tests/test_page_imports.py
```
