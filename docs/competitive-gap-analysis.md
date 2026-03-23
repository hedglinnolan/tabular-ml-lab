# Competitive Gap Analysis: Tabular ML Lab vs MLJAR AutoML

**Date:** 2026-03-23 (revised after comprehensive codebase audit)
**Purpose:** Identify real capability gaps after full code inspection.

---

## What You Actually Have (I was wrong about some gaps)

After auditing the codebase, the app is more capable than the gap analysis I wrote earlier suggested:

| Capability | Status | Details |
|---|---|---|
| **Optuna hyperparameter tuning** | ✅ Already built | `optimize_model_hyperparameters()` in Train page. Generic, works with any model's `hyperparam_schema`. Optuna 4.7.0 installed. |
| **Cross-validation** | ✅ Already built | `perform_cross_validation()` in `ml/eval.py`. Results shown in model comparison table. |
| **Manual hyperparameter controls** | ✅ Already built | Per-model expanders with sliders/selects driven by `hyperparam_schema` |
| **Partial dependence plots** | ✅ Already built | In Explainability page, checkbox-enabled |
| **Decision curve analysis** | ✅ Already built | `decision_curve_analysis()` in `ml/calibration.py` |
| **Calibration curves** | ✅ Already built | Both classification and regression calibration |
| **Bootstrap confidence intervals** | ✅ Already built | BCa CIs for all metrics in `ml/bootstrap.py` |
| **Group-aware splits** | ✅ Already built | GroupShuffleSplit, GroupKFold support |
| **Random Forest** | ✅ Already built | Custom RFWrapper, reg + clf |
| **Class imbalance detection** | ✅ Already built | `dataset_profile.is_imbalanced` flag |

## Real Remaining Gaps

### Tier 1 — Credibility (researchers will notice)

**1. No XGBoost or LightGBM**
- Neither package is installed in the venv
- These are the #1 and #2 most-used tabular ML algorithms
- Every benchmark paper includes them
- Both have sklearn-compatible APIs → can slot into existing ModelRegistry
- **Effort:** Medium — add to requirements, write 4 ModelSpec entries (xgb_reg, xgb_clf, lgbm_reg, lgbm_clf), test
- CatBoost is nice-to-have but not a credibility gap

**2. No class imbalance handling**
- Imbalance is *detected* (`dataset_profile.is_imbalanced`) but never *acted on*
- No SMOTE, no class_weight='balanced', no oversampling/undersampling
- The coaching mentions class weighting but the app doesn't offer it
- For clinical/health data this is critical — rare disease prediction, adverse event classification
- **Effort:** Medium — add `class_weight` param to relevant classifiers, optionally add imblearn SMOTE as preprocessing step

**3. No target variable preprocessing**
- Can transform *features* (log, power, polynomial) but not the *target*
- Log-transforming a right-skewed target (e.g., medical costs, hormone levels) dramatically improves linear model performance
- MLJAR handles this automatically
- **Effort:** Low-medium — add log/Box-Cox target transform option in Preprocess, inverse-transform predictions for metrics

### Tier 2 — Competitive (makes you stronger)

**4. No ensembling / model stacking**
- Train page trains models independently — no way to combine top models
- Simple voting/averaging ensemble is high-value, low-effort
- Stacking (meta-learner on base model predictions) is more complex but very powerful
- **Effort:** Medium-high for stacking, low for voting ensemble

**5. No decision tree visualization**
- You have HistGradientBoosting, RF, ExtraTrees — but no way to see individual trees
- dtreeviz or sklearn's plot_tree would make the app much more educational
- Directly supports the teaching differentiator
- **Effort:** Low — sklearn.tree.plot_tree works out of the box

**6. No feature interaction detection**
- SHAP interaction values exist but aren't surfaced
- Explicitly showing "BMI × Age interaction = 0.05" helps researchers understand what the model learned
- **Effort:** Medium — SHAP interaction values are computationally expensive, maybe tree-based only

**7. Theory Reference page is disconnected from the workflow**
- 11_Theory_Reference.py is a standalone page with extensive content
- But it's not linked contextually — when a user is looking at SHAP results, there's no "📚 Learn more about SHAP" link to the relevant theory section
- This is a missed opportunity for the education moat
- **Effort:** Low — add contextual links from analysis pages to specific theory sections

### Tier 3 — Polish

**8. No model export (pickle/ONNX/PMML)**
- Can't save a trained model for deployment
- MLJAR saves everything automatically
- Less critical for research-focused users but expected by more technical ones
- **Effort:** Low — joblib.dump for sklearn models

**9. No session persistence across browser reloads**
- SessionManager exists but full state restore on reload isn't implemented
- Losing a 30-minute analysis session to a browser refresh is painful
- **Effort:** High — Streamlit's session state is volatile by design

**10. Decision curve analysis exists but isn't exposed in the UI**
- `decision_curve_analysis()` is implemented in ml/calibration.py
- But there's no button or page that calls it for the user
- DCA is increasingly expected in clinical ML papers
- **Effort:** Low — wire into Explainability or Train page

---

## Summary: What's Actually Missing vs What I Thought Was Missing

| Previously claimed gap | Actual status |
|---|---|
| ~~No hyperparameter tuning~~ | ✅ Optuna already integrated |
| ~~No cross-validation~~ | ✅ Already implemented |
| ~~No PDP~~ | ✅ Already in Explainability |
| No XGBoost/LightGBM | 🔴 Still missing |
| No class imbalance handling | 🔴 Still missing |
| No target preprocessing | 🔴 Still missing |
| No ensembling | 🟡 Still missing |
| No decision tree viz | 🟡 Still missing |
| No contextual theory links | 🟡 Still missing |

The app is significantly more capable than I initially assessed. The three real credibility gaps are XGBoost/LightGBM, class imbalance handling, and target preprocessing. Everything else is either competitive advantage (education moat) or polish.
