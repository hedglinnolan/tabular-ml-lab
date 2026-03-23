# Competitive Gap Analysis: Tabular ML Lab vs MLJAR AutoML

**Date:** 2026-03-23
**Purpose:** Identify capability gaps to reach feature parity where it matters, while leaning into the education differentiator.

---

## Current Model/Algorithm Comparison

| Capability | Tabular ML Lab | MLJAR AutoML | Gap |
|---|---|---|---|
| **Linear models** | Ridge, Lasso, ElasticNet, LogReg, GLM, HuberGLM | Linear | ✅ We're stronger |
| **Tree models** | Random Forest, ExtraTrees, HistGradientBoosting | RF, ExtraTrees, HistGB | ✅ Parity |
| **Gradient boosting** | HistGradientBoosting only | XGBoost, LightGBM, CatBoost | 🔴 Major gap |
| **Neural networks** | Custom NNWeightedHuber | MLP-style | ✅ Parity (different impl) |
| **Distance/Margin** | KNN, SVM/SVR | KNN | ✅ We're stronger |
| **Probabilistic** | GaussianNB, LDA | Baseline | ✅ We're stronger |
| **Ensembling** | None | Greedy ensemble (Caruana) | 🔴 Gap |
| **Stacking** | None | Level-2 stacked ensemble | 🔴 Gap |
| **Hyperparameter tuning** | None (fixed defaults) | Random search + hill climbing + Optuna | 🔴 Major gap |

## Feature Engineering & Selection Comparison

| Capability | Tabular ML Lab | MLJAR AutoML | Gap |
|---|---|---|---|
| **Polynomial features** | ✅ | ✅ ("Golden Features") | Parity |
| **Interaction features** | ✅ | ✅ | Parity |
| **Log/power transforms** | ✅ | ❌ | We're stronger |
| **Binning** | ✅ | ❌ | We're stronger |
| **Missing value imputation** | KNN, median, mean, iterative | Basic | We're stronger |
| **Feature selection** | LASSO path, RFE-CV, univariate, stability, consensus | Built-in selection | We're stronger |
| **Text features** | ❌ | ✅ | Gap (low priority) |
| **Time features** | ❌ | ✅ | Gap (low priority) |
| **Target preprocessing** | ❌ | ✅ | 🟡 Gap |

## Analysis & Reporting Comparison

| Capability | Tabular ML Lab | MLJAR AutoML | Gap |
|---|---|---|---|
| **EDA** | 7 guided actions + coaching | Basic auto-EDA | ✅ Much stronger |
| **SHAP** | ✅ + AI interpretation | ✅ | We're stronger |
| **Permutation importance** | ✅ + AI interpretation | ✅ | We're stronger |
| **Learning curves** | ✅ + AI interpretation | ✅ | We're stronger |
| **Confusion matrix** | ✅ + AI interpretation | ✅ | We're stronger |
| **Hypothesis testing** | ✅ (6 test types) | ❌ | We're much stronger |
| **Sensitivity analysis** | ✅ (seed + dropout) | ❌ | We're much stronger |
| **Report export** | Publication-ready + methods section | Per-model markdown | We're stronger |
| **Coaching layer** | ✅ Full guided workflow | ❌ | Our moat |
| **Insight ledger** | ✅ Cumulative session memory | ❌ | Our moat |
| **AI interpretation** | ✅ Contextual, session-aware | ❌ | Our moat |
| **Fairness metrics** | ❌ | ✅ (demographic parity, etc.) | 🟡 Gap |
| **Cross-validation** | ❌ (train/test split only) | 5-fold, 10-fold CV | 🔴 Gap |
| **Save/resume training** | ❌ | ✅ | 🟡 Gap |

---

## Priority Gaps (what to close)

### Tier 1 — Must close (credibility gaps)

**1. XGBoost + LightGBM**
Every serious tabular ML platform has these. Researchers will ask "where's XGBoost?" and question the platform's legitimacy if it's missing. Both are pip-installable with sklearn-compatible APIs. Can slot into the existing model registry pattern.

**2. Hyperparameter tuning**
Running models with fixed defaults is a non-starter for publication. Reviewers will ask "how did you select hyperparameters?" Currently the answer is "we didn't." Optuna is the standard — sklearn-compatible, can wrap any estimator. Even a basic RandomizedSearchCV would close the gap.

**3. Cross-validation**
Train/test split only is a limitation reviewers will flag. k-fold CV is table stakes. This also unlocks proper confidence intervals on metrics (instead of single-split point estimates).

### Tier 2 — Should close (competitive gaps)

**4. Ensembling**
A simple voting/averaging ensemble of top-N models would add significant value and isn't hard to implement. Stacking is more complex but very powerful.

**5. Target preprocessing**
Log-transforming skewed targets, stratified sampling for imbalanced classification — these are common needs.

**6. Fairness metrics**
Increasingly expected in clinical/health ML. Demographic parity, equalized odds, disparate impact ratios. MLJAR has this and it's a good look.

### Tier 3 — Nice to have (not urgent)

**7. Save/resume sessions**
Session persistence across browser reloads. Useful but not a dealbreaker.

**8. Text/time features**
Niche — only matters if users have those data types.

---

## The Education Moat — Where to Double Down

The coaching layer, insight ledger, and AI interpretation are genuinely unique. No competitor has this. To make it a true moat:

### 1. Make the coaching pedagogically rigorous
- Each decision point should reference why (not just what): "We recommend standardizing features for Ridge because coefficients are penalized — without scaling, features with larger ranges dominate the penalty"
- Add links to relevant theory (your Theory Reference page exists but isn't deeply integrated)
- Every coaching message should be testable: "Can a student who reads only the coaching messages pass a methods exam?"

### 2. Make the AI interpretation interactive
- Already done in this session (follow-up questions)
- Next: let users ask "explain this to me like I'm a first-year stats student" vs "explain this for a reviewer"

### 3. Make the report export reviewer-ready
- Auto-generate a TRIPOD checklist for predictive modeling studies
- Auto-generate a STROBE checklist if it's observational
- Flag which checklist items the analysis already satisfies vs which need manual completion

### 4. Build the "learning path"
- Track which concepts the user has encountered across their session
- Surface "you've now seen SHAP for 3 models — here's what to take away about feature importance in general"
- This turns the app from a tool into a curriculum
