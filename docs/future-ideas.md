# Future Ideas — Tabular ML Lab

Running list of deferred features, enhancements, and experiments. Prioritized loosely.

---

## Credibility Gaps (Tier 1)

### Target Variable Preprocessing
- Log/Box-Cox/Yeo-Johnson transform on the target for regression tasks
- Inverse-transform predictions for metrics
- Per-model pipeline: tree models may not need it
- Already earmarked in ARCHITECTURE.md

## Model Enhancements

### SMOTE / Synthetic Oversampling
- `imblearn` SMOTE as optional preprocessing step for imbalanced classification
- Must apply *after* preprocessing, *before* training, *only on training set*
- Adds a dependency (imbalanced-learn)
- class_weight='balanced' covers 80% of cases; SMOTE is for when that's not enough

### XGBoost/LightGBM Early Stopping
- Both support `eval_set` for monitoring validation loss during training
- Current generic RegistryModelWrapper doesn't pass eval_set
- Would prevent overfitting without needing to precisely tune n_estimators
- Low priority — Optuna handles hyperparameter selection anyway

### CatBoost
- Nice-to-have but not a credibility gap
- Native categorical handling, ordered boosting
- Would slot into existing Boosting group

### Ensembling / Model Stacking
- Train page trains models independently — no way to combine
- Simple voting/averaging ensemble is high-value, low-effort
- Stacking (meta-learner on base predictions) is more complex

## Education Moat

### Contextual Theory Links
- When user is looking at SHAP results, link to Theory Reference SHAP section
- Same for residual plots → regression diagnostics, calibration → calibration theory
- Low effort, high pedagogical value

### Decision Tree Visualization
- sklearn.tree.plot_tree or dtreeviz for individual trees
- Makes tree-based models more educational
- Works out of the box with existing models

### Feature Interaction Detection
- Surface SHAP interaction values explicitly
- "BMI × Age interaction = 0.05"
- Computationally expensive — maybe tree-based only

## Polish

### Model Export (pickle/ONNX)
- joblib.dump for sklearn models
- Less critical for research-focused users

### Session Persistence
- Full state restore on browser reload
- High effort — Streamlit session state is volatile by design

### Wire Decision Curve Analysis to UI
- `decision_curve_analysis()` exists in ml/calibration.py
- No button or page calls it for the user
- Low effort to wire into Explainability or Train page

---

*Last updated: 2026-03-23*
