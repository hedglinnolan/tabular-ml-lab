# Export Mechanical Completeness — 3 Passes

## Goal
Make LaTeX and markdown generation do all the "mechanical work" of writing a prediction model paper. The researcher's only edits should be domain-specific context no tool can provide.

## Guiding principle
We are NOT writing the paper. We ARE doing every piece of assembly that can be derived from workflow state: formatting, scaffolding, metric insertion, structural prompts, figure references.

---

## Pass 1: LaTeX Table Formatting (blocking issue)

### Problem
All LaTeX tables use plain `tabular` with no width management. Tables overflow margins with long model names, CI strings, wide Table 1 outputs, or custom test columns. The custom-test merge in `pages/10_Report_Export.py` adds new columns to Table 1, widening the schema before LaTeX generation.

### Fixes

#### 1a. Fix custom-test merge shape
**File:** `pages/10_Report_Export.py` (LaTeX manuscript button handler, ~line 590-610)
- Currently: custom tests concatenated as new rows with extra columns (`Test`, `Statistic`, `p-value`)
- Fix: append test info as a footnote annotation or within existing column structure
- Custom test rows should use the SAME columns as the base Table 1, with test details in a note/footnote

#### 1b. Add width containment to LaTeX preamble
**File:** `ml/latex_report.py` → `generate_latex_report()` preamble
- Add `\usepackage{tabularx}` and `\usepackage{adjustbox}`
- Add `\usepackage{longtable}` for large Table 1s

#### 1c. Width-contained performance table
**File:** `ml/latex_report.py` → `_metrics_to_latex_table()`
- Wrap in `\begin{adjustbox}{max width=\textwidth}`
- Use `\small` or `\footnotesize` font for wide tables (>4 metric columns or >4 models)
- First column (model name) should use `p{3cm}` or `X` type for wrapping
- Add `\setlength{\tabcolsep}{4pt}` for tighter spacing when needed

#### 1d. Width-contained Table 1
**File:** `ml/latex_report.py` → `_table1_to_latex()`
- Same adjustbox wrapping
- First column ("Characteristic") uses `p{4cm}` for wrapping long variable names
- Detect wide tables (>4 columns) and auto-apply `\small`

#### 1e. Standalone Table 1 export alignment
**File:** `ml/table_one.py` → `table1_to_latex()`
- Apply same width containment strategy as 1d
- Replace raw `to_latex()` with manual construction matching `latex_report.py` style

### Verification
- `python3 -m py_compile ml/latex_report.py ml/table_one.py pages/10_Report_Export.py`
- `pytest -q tests/test_publication.py`
- `pytest -q tests/test_page_imports.py`

---

## Pass 2: LaTeX Content Completeness

### Problem
The LaTeX template has good Methods but weak everything else. Abstract is placeholder. Results only has performance table. Discussion is empty. No figure references. No explainability/sensitivity results in compiled PDF.

### Fixes

#### 2a. Auto-scaffold abstract
**File:** `ml/latex_report.py` → `generate_latex_report()`
- Generate structured abstract (Objective / Methods / Results / Conclusion) from known facts
- Objective: "[PLACEHOLDER: clinical context]. This study developed and validated a prediction model for {target_name} using {task_type}."
- Methods: "A total of {n_total} observations with {n_features} predictors were split into training ({n_train}), validation ({n_val}), and test ({n_test}) sets. {len(model_results)} models were compared."
- Results: "The best model ({best_model}) achieved {primary_metric}: {value} (95% CI: {ci})." — fill from actual results
- Conclusion: "[PLACEHOLDER: Summarize clinical implications]"

#### 2b. Flow explainability results into LaTeX Results
**File:** `ml/latex_report.py` → `generate_latex_report()`
- Add optional `explainability_summary` parameter
- If SHAP results available: add a "Feature Importance" subsection listing top features
- If permutation importance available: reference it
- If calibration results available: add calibration metrics subsection
- Keep it factual/mechanical — no interpretation

#### 2c. Flow sensitivity results into LaTeX Results
**File:** `ml/latex_report.py` → `generate_latex_report()`
- Add optional `sensitivity_summary` parameter
- If seed stability was run: report CV% and range
- If feature dropout was run: reference it
- Brief, factual reporting only

#### 2d. Structural Discussion skeleton
**File:** `ml/latex_report.py` → `generate_latex_report()`
- Replace empty placeholders with result-specific prompts:
  - Principal Findings: "The {best_model} achieved {metric} on held-out data. [PLACEHOLDER: Interpret in clinical context]"
  - If SHAP available: "Key predictors identified were {top_3_features}. [PLACEHOLDER: Discuss biological plausibility]"
  - Comparison: "[PLACEHOLDER: Compare {metric} to prior work. Note: typical {task_type} models in this domain achieve ...]"
  - Strengths: auto-fill what we know (sample size, bootstrap CIs, TRIPOD compliance %)
  - Limitations: auto-fill methodological considerations from methods section

#### 2e. Figure reference placeholders
**File:** `ml/latex_report.py` → `generate_latex_report()`
- Add `\begin{figure}` blocks referencing standard export filenames:
  - `plots/train/{model}_predictions.png` (regression) or `plots/train/{model}_confusion_matrix.png` (classification)
  - `plots/explainability/{model}_permutation_importance.png`
  - SHAP summary plot reference
  - Calibration plot reference
- Comment them out by default with a note: "Uncomment after placing figure files"

#### 2f. Wire new parameters through export page
**File:** `pages/10_Report_Export.py`
- Build `explainability_summary` and `sensitivity_summary` from session state
- Pass to `generate_latex_report()`

### Verification
- Same compile + test checks as Pass 1
- Inspect generated LaTeX for structural correctness

---

## Pass 3: Markdown Report Alignment

### Problem
Markdown report dumps metrics in both prose AND table. No abstract scaffold. Discussion/interpretation section missing. Should carry same content as LaTeX.

### Fixes

#### 3a. Add abstract scaffold to markdown report
**File:** `pages/10_Report_Export.py` → `generate_report()`
- Add "Abstract (Draft)" section at top with same structured scaffold as LaTeX

#### 3b. Reduce prose/table redundancy
**File:** `pages/10_Report_Export.py` → `generate_report()`
- In Model Performance section: brief narrative ("Best model: X with RMSE Y") + table
- Remove per-model metric re-dump in prose when table exists
- Keep per-model detail section (hyperparams, coefficients) as-is

#### 3c. Add Discussion scaffold to markdown
**File:** `pages/10_Report_Export.py` → `generate_report()`
- Same structural skeleton as LaTeX Discussion
- Result-specific prompts instead of generic placeholders

### Verification
- Same compile + test checks

---

## Files touched
- `ml/latex_report.py` (Passes 1, 2)
- `ml/table_one.py` (Pass 1)
- `pages/10_Report_Export.py` (Passes 1, 2, 3)
- `tests/test_publication.py` (regression checks)

## What NOT to change
- `ml/publication.py` — methods generation logic is working well, don't touch
- Analytics/workflow behavior on any page
- Session state shape or upstream data flow
- Model training or evaluation logic

## Risk
- Low-medium. Changes are additive (new content) or formatting-only (table width)
- Main risk: breaking existing test expectations if we change LaTeX structure
- Mitigation: run full test suite after each pass
