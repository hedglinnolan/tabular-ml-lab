# Quick Start Guide

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **uv** (recommended): [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git (for cloning)
- 4GB RAM minimum
- **Optional:** Install and run [Ollama](https://ollama.ai) if you want LLM-powered interpretations.

## Windows (PowerShell)

### From Fresh Clone

```powershell
# 1. Install uv (one-time): irm https://astral.sh/uv/install.ps1 | iex

# 2. Clone the repository (if not already cloned)
# git clone <repo-url>
# cd glucose-mlp-interactive

# 3. First time setup (creates .venv, installs deps via uv)
.\setup.ps1

# 4. Run preflight check (optional but recommended)
uv run python preflight.py

# 5. Run the app
.\run.ps1
```

**Or minimal:**
```powershell
.\setup.ps1
.\run.ps1
```

## macOS/Linux (bash/zsh)

### From Fresh Clone

```bash
# 1. Install uv (one-time): curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository (if not already cloned)
# git clone <repo-url>
# cd glucose-mlp-interactive

# 3. Make scripts executable (first time)
chmod +x setup.sh run.sh

# 4. First time setup (creates .venv, installs deps via uv)
./setup.sh

# 5. Run preflight check (optional but recommended)
uv run python preflight.py

# 6. Run the app
./run.sh
```

**Or minimal:**
```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

## Branch Information

**Current branch:** Check with `git branch --show-current`

**To switch to feature branch (if needed):**
```bash
git checkout <branch-name>
```

## Preflight Check

Run before starting the app to verify dependencies:

```bash
# Windows
python preflight.py

# macOS/Linux
python3 preflight.py
```

This checks:
- Python version (3.8+)
- Key packages (streamlit, torch, pandas, numpy, sklearn, plotly)
- Module imports (ml/*, models/*, utils/*)

## Common Errors & Fixes

### Error: "uv not found"
**Fix:** Install uv first — see [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

### Error: "streamlit: command not found"
**Fix:** Run via uv (uses .venv automatically)
```bash
uv run streamlit run app.py
```

### Error: "No module named 'torch'"
**Fix:** Reinstall dependencies
```bash
uv pip install -r requirements.txt
```

### Error: "No module named 'sklearn'"
**Fix:** Reinstall dependencies
```bash
uv pip install -r requirements.txt
```

### Error: "numpy version incompatible"
**Fix:** Upgrade numpy
```bash
uv pip install --upgrade "numpy>=1.24.0"
```

### Error: "shap not found" (when using SHAP features)
**Fix:** Install SHAP (optional)
```bash
uv pip install shap
```

### LLM / Ollama interpretations not working
**Symptom:** "Interpret these results using an LLM" shows setup instructions or an error.  
**Fix:**
1. Install [Ollama](https://ollama.ai).
2. Run `ollama serve` in a terminal (and keep it running).
3. Pull a model, e.g. `ollama run qwen2.5:7b`.  
The app works fully without Ollama; this only affects the optional LLM feature. See [README → Troubleshooting](README.md#-troubleshooting) for more detail.

### Error: `uv pip install` fails — "llvmlite" / "only versions >=3.6,<3.10 are supported"
**Cause:** The venv uses Python 3.10+; `llvmlite` (numba/shap) supports only &lt;3.10.  
**Fix:** Use Python 3.9. Remove `.venv` and re-run setup:
```bash
rm -rf .venv && ./setup.sh   # macOS/Linux
# Windows: Remove-Item -Recurse -Force .venv; .\setup.ps1
```
Setup creates `.venv` with `uv venv --python 3.9` and installs deps.

### Error: "Port 8501 already in use"
**Fix:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Error: Import errors for ml/* or models/*
**Fix:** Ensure you're in the repo root directory
```bash
# Verify you're in the right directory
pwd  # Should show: .../glucose-mlp-interactive

# Check that directories exist
ls ml/ models/ utils/
```

## 5-Minute Smoke Test Checklist

### Test 1: Linear Regression with Outliers

1. **Upload & Audit Page**
   - Select "Linear Regression with Outliers" from dataset dropdown
   - Click "Generate Dataset"
   - **Expected:** Dataset preview shows 500 rows, 3 columns (feature_1, feature_2, target)
   - Select "target" as target variable
   - Select "feature_1" and "feature_2" as features
   - **Expected:** Task type auto-detects as "Regression"

2. **EDA Page**
   - **Expected:** Summary statistics table, target distribution histogram, correlation heatmap
   - **Expected:** Scatter plots showing target vs features

3. **Preprocess Page**
   - Click "Build Preprocessing Pipeline"
   - **Expected:** Pipeline recipe displayed, transformation preview shows numeric features

4. **Train & Compare Page**
   - Click "Prepare Splits"
   - Select: Neural Network, Random Forest, GLM (OLS), GLM (Huber)
   - Click "Train Models"
   - **Expected:** All models train successfully
   - **Expected:** Metrics table shows all models
   - **Expected:** Huber GLM should have competitive/better RMSE than OLS (due to outliers)

5. **Explainability Page**
   - Click "Calculate Permutation Importance"
   - **Expected:** Feature importance bars for each model
   - Click "Calculate Partial Dependence"
   - **Expected:** Partial dependence plots for top 3 features

6. **Report Export Page**
   - **Expected:** Markdown report displayed
   - Click "Download Complete Package (ZIP)"
   - **Expected:** ZIP downloads with report.md, metrics.csv, predictions_*.csv, plot_*.png files

### Test 2: Nonlinear Regression

1. **Upload & Audit Page**
   - Select "Nonlinear Regression" dataset
   - Generate and configure (target + 3 features)
   - **Expected:** Task type: Regression

2. **Train & Compare Page**
   - Train all models
   - **Expected:** RF and NN should outperform GLM (lower RMSE) due to nonlinearity

### Test 3: Imbalanced Classification

1. **Upload & Audit Page**
   - Select "Imbalanced Classification" dataset
   - Generate and configure
   - **Expected:** Task type auto-detects as "Classification" (10 unique values warning)
   - Override to "Classification" if needed

2. **EDA Page**
   - **Expected:** Class balance chart shows ~80% class 0, ~20% class 1

3. **Train & Compare Page**
   - **Expected:** Neural Network and Huber GLM are disabled (classification not supported)
   - Train: Random Forest, GLM (Logistic)
   - **Expected:** Metrics include Accuracy, F1, ROC-AUC, LogLoss
   - **Expected:** Confusion matrix displayed
   - **Expected:** F1 and ROC-AUC more informative than accuracy (due to imbalance)

4. **Explainability Page**
   - Calculate importance and partial dependence
   - **Expected:** Works correctly for classification models

## Expected Page Behaviors

### Upload & Audit Page
- Dataset preview table
- Data audit summary (missing values, duplicates, etc.)
- Target/feature selection dropdowns
- Task type selection (auto-detect with override)
- Built-in dataset dropdown

### EDA Page
- Summary statistics table
- Target distribution (histogram + box plot)
- Correlation heatmap
- Target vs feature plots (scatter for numeric, box for categorical)
- Class balance (if classification)

### Preprocess Page
- Numeric preprocessing options (imputation, scaling, log transform)
- Categorical preprocessing options (imputation, encoding)
- Pipeline recipe display
- Transformation preview (before/after)

### Train & Compare Page
- Split configuration sliders (train/val/test %)
- Cross-validation toggle
- Model selection checkboxes
- Hyperparameter expanders
- Training progress bars
- Metrics comparison table
- Learning curves (for NN)
- Predictions vs Actual plots
- Residual plots (regression) or Confusion matrix (classification)

### Explainability Page
- Permutation importance button
- Feature importance bar charts
- Partial dependence button
- Partial dependence plots (top 3 features)
- SHAP toggle (optional, requires shap package)

### Report Export Page
- Generated markdown report preview
- Download buttons (Markdown, ZIP)
- ZIP includes: report.md, metrics.csv, predictions_*.csv, plot_*.png

## Time Estimate

- **Setup:** 2-3 minutes (venv + dependencies)
- **Preflight:** 10 seconds
- **Smoke test:** 5 minutes (3 scenarios × ~1.5 min each)

**Total:** ~8 minutes from clone to verified working app
