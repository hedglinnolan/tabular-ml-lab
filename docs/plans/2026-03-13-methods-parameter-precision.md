# Methods Parameter Precision Pass

## Goal
Thread all captured modeling decisions into the methods section with full specificity. No generic descriptions when specific parameters are known.

## Fixes

### 1. Hyperparameters per model in Model Development
**File:** `ml/publication.py` → Model Development subsection
- After listing model candidates, add hyperparameter details from `manuscript_context`
- New optional param: `model_hyperparameters: Optional[Dict[str, Dict]]`
- Format: "LASSO (α=0.1), HistGradientBoosting (n_estimators=100, max_depth=5, learning_rate=0.1), Neural Network (hidden layers: [64, 32], learning rate=0.001, max epochs=200)"
- Only include the most publication-relevant params per model type (not every sklearn kwarg)

### 2. Hyperparameter optimization reporting
**File:** `ml/publication.py` → Model Development subsection
- New optional param: `hyperparameter_optimization: bool = False`
- If True: "Hyperparameter optimization was performed using Optuna (30 trials per model)."
- If False: "Default hyperparameters were used." or just omit

### 3. Split stratification
**File:** `ml/publication.py` → Study Design or Model Development
- New optional param: `split_strategy: Optional[str]` (values: "random", "stratified", "time-based", "group-based")
- Thread into split description: "Data were randomly split..." / "Data were split using stratified sampling..." / "Data were split chronologically..."

### 4. Feature selection consensus threshold
**File:** `ml/publication.py` → Feature Selection narrative
- Already reads from methodology log `Feature Selection Applied` entries
- Add: read `consensus_threshold` from the log details if available
- Also: read individual method results count from `Feature Selection` log
- Format: "Features were retained if selected by at least N of M methods"
- **Also need:** Store consensus threshold in methodology log on Feature Selection page

### 5. PCA component specification
**File:** `ml/publication.py` → Feature Engineering subsection
- Already reads engineering_log from session state
- Parse PCA entries more specifically: distinguish "10 fixed components" from "95% variance threshold"
- The preprocessing page stores `pca_mode` and `pca_n_components` in per-model configs

### 6. Missing data rates
**File:** `ml/publication.py` → Missing Data subsection
- New optional param: `missing_data_summary: Optional[Dict]` with n_features_with_missing, total_missing_rate, per-feature rates
- Format: "X of Y features had missing values (range: Z-W%)" 
- Available from `dataset_profile` in session state

## Wiring (pages/10_Report_Export.py)
- Build `model_hyperparameters` from `selected_model_params` session state
- Build `split_strategy` from `split_config` 
- Build `missing_data_summary` from `dataset_profile`
- Pass `hyperparameter_optimization` from methodology log
- Pass all to `generate_methods_section()` and `_build_methods_section_for_export()`

## What NOT to change
- ml/latex_report.py (methods flow through from publication.py via markdown)
- Model training or preprocessing logic
- Session state shape

## Verification
- py_compile on all changed files
- pytest tests/test_publication.py tests/test_page_imports.py
