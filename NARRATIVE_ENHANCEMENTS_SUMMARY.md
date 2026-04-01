# NarrativeEngine Enhancements — Implementation Summary

**Date:** 2026-04-01  
**Issues Addressed:** #81, #83, #82  
**Files Modified:**
- `ml/narrative_engine.py` (main implementation)
- `tests/test_narrative_engine.py` (test coverage)

---

## Issue #81: Hyperparameters in Model Development ✅

**Problem:** The Model Development section listed which models were trained but not their hyperparameters.

**Solution:**
- Added `_describe_hyperparameters()` helper method that generates human-readable hyperparameter descriptions
- Integrated hyperparameter reporting into `_gen_model_development()`
- Format examples:
  - Ridge: "alpha=1.0 (L2 regularization)"
  - Random Forest: "100 trees, unrestricted depth"
  - Neural Network: "architecture [64×32], dropout=0.1, learning rate=0.001, 200 epochs"

**Key Design Decisions:**
- Focus on parameters that matter to reviewers, not internal sklearn defaults
- Use domain-appropriate prose instead of raw parameter dumps
- Filter out None, random_state, n_jobs, verbose

**Test Coverage:**
- `test_hyperparameters_in_model_development`

---

## Issue #83: Confidence Intervals in Model Evaluation ✅

**Problem:** The abstract mentioned confidence intervals but the Model Evaluation section didn't include them.

**Solution:**
- Extended `_gen_model_evaluation()` to detect and format CI bounds when present in metrics_by_model
- Support for `_ci_lower` and `_ci_upper` suffixes (e.g., `RMSE_ci_lower`, `RMSE_ci_upper`)
- Format: "R² = 0.269 (95% CI: 0.255–0.283)"
- Graceful fallback to point estimates when CIs not available

**Integration Point:**
- Bootstrap CIs computed in `pages/06_Train_and_Compare.py` can be written to provenance as:
  ```python
  metrics_by_model = {
      "ridge": {
          "RMSE": 12.34,
          "RMSE_ci_lower": 11.95,
          "RMSE_ci_upper": 12.73,
          "R2": 0.72,
          "R2_ci_lower": 0.68,
          "R2_ci_upper": 0.76,
      }
  }
  ```

**Test Coverage:**
- `test_confidence_intervals_in_evaluation`

---

## Issue #82: Results + Discussion Sections ✅

**Problem:** NarrativeEngine only generated Methods section; Results and Discussion were missing to complete IMRAD structure.

**Solution:**

### Results Section
Auto-generates:
- Best model and key metrics (R²/RMSE for regression, Accuracy/F1/AUC for classification)
- Confidence intervals when available
- Comparative performance table reference ("Table 1")
- Ranking of models (best to worst)
- **Key insight:** Notes when simple models (linear, logistic) are competitive with complex ones (RF, XGB, NN)
  - "Regularized linear models achieved performance comparable to ensemble methods, suggesting that the relationship between predictors and outcome may be approximately linear."
- Placeholder for feature importance (requires explainability provenance integration)

### Discussion Section
Structured skeleton with:

1. **Principal Findings** (auto-generated)
   - Summary of best model and performance
   - R² percentage interpretation for regression
   - Note on number of models compared

2. **Comparison with Prior Work** (placeholder)
   - "[To be completed by the investigator: Compare results to published studies...]"

3. **Strengths and Limitations** (auto-populated from InsightLedger)
   - **Strengths:** Extracted from insights with severity="info" and "favorable" in finding
   - **Limitations:** Extracted from acknowledged insights
   - Fallback placeholder when ledger empty

4. **Clinical and Practical Implications** (placeholder)
   - "[To be completed by the investigator: Discuss how findings could inform practice...]"

5. **Conclusions** (auto-generated)
   - Brief restatement of findings
   - Standard call for validation: "Further validation in independent cohorts and exploration of causal mechanisms are warranted before clinical or policy implementation."

**Design Principle:**
> "Surface contrasts, not conclusions." — Report what happened; don't interpret why. The Discussion skeleton invites the human to do the interpretation.

**Manuscript Structure:**
- Results and Discussion render as **top-level sections** (not subsections of Methods)
- Markdown: `## Results`, `## Discussion`
- LaTeX: `\section{Results}`, `\section{Discussion}`
- `ManuscriptDraft.sections` property returns Methods subsections only (backward compatible)
- `ManuscriptDraft.all_sections` property returns full manuscript (Methods + Results + Discussion)

**Test Coverage:**
- `test_results_section_generated`
- `test_results_comparative_performance`
- `test_results_notes_simple_vs_complex`
- `test_discussion_section_generated`
- `test_discussion_principal_findings_auto_generated`
- `test_discussion_placeholders`
- `test_discussion_strengths_and_limitations_from_ledger`
- `test_results_and_discussion_in_markdown`
- `test_results_and_discussion_in_latex`

---

## Testing Results

**Full test suite:** ✅ 221 passed, 1 skipped, 4 warnings  
**NarrativeEngine tests:** ✅ 28 passed  
**New tests added:** 11

All tests pass. No breaking changes to existing functionality.

---

## Tone & Style

The generated manuscript sections read like peer-reviewed papers:
- No coding artifacts
- No raw Python dict dumps
- No internal session_state keys
- Reviewer-focused language throughout

Examples from test output:
- ❌ "alpha": 1.0  
- ✅ "Ridge Regression was trained with alpha=1.0 (L2 regularization)."

- ❌ metrics = {"RMSE": 10.1, "RMSE_ci_lower": 9.8, "RMSE_ci_upper": 10.4}  
- ✅ "root mean squared error (RMSE)=10.1 (95% CI: 9.8–10.4)"

- ❌ "Simple model beat complex model"  
- ✅ "Regularized linear models achieved performance comparable to ensemble methods, suggesting that the relationship between predictors and outcome may be approximately linear."

---

## Future Work / Notes

1. **Feature Importance Integration:** Results section currently has a placeholder for feature importance. When explainability provenance is integrated, this can be auto-populated with top 3-5 features from permutation importance.

2. **Bootstrap CI Integration:** The training page already computes bootstrap CIs (`st.session_state["bootstrap_results"]`). To fully integrate:
   - Update `pages/06_Train_and_Compare.py` to write CIs to provenance when bootstrap button is clicked:
     ```python
     # After bootstrap_results computed
     get_provenance().record_training(
         ...,
         metrics_by_model={
             name: {
                 **res.get('metrics', {}),
                 **{f"{k}_ci_lower": v["ci_lower"] for k, v in cis.items()},
                 **{f"{k}_ci_upper": v["ci_upper"] for k, v in cis.items()},
             }
             for name, (res, cis) in zip(model_results.items(), bootstrap_results.items())
         }
     )
     ```

3. **Cross-Validation CIs:** If CV is used, CV standard errors could also be formatted as CIs in the evaluation section.

---

## Commit

Single commit containing all three enhancements:
```
feat: #81 #83 #82 Implement NarrativeEngine enhancements
```

The changes are tightly coupled (all extend the same section generators) and tested together, so a single atomic commit was appropriate.
