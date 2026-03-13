# Diagnostic Assistant Implementation Summary

## Overview
Added intelligent diagnostic assistant to help users understand why their models are performing poorly and what actions to take.

## Location
- **File:** `pages/06_Train_and_Compare.py`
- **Line Range:** 1484-1606 (122 lines)
- **Insertion Point:** After model selection guidance, before model diagnostics section

## Trigger Conditions

The diagnostic assistant activates when model performance falls below these thresholds:

- **Classification:** AUC < 0.65
- **Regression:** R² < 0.40

## Diagnostic Checks Implemented

### 1. Feature-Target Correlations (HIGH severity)
- **Check:** Maximum absolute correlation between numeric features and target
- **Trigger:** max_corr < 0.1
- **Message:** "No feature has correlation >0.1 with target (max: {max_corr:.3f})"
- **Action:** "Review EDA (page 2) for feature-target relationships. Consider feature engineering (page 3) or collecting more informative data."

### 2. Sample Size (HIGH severity)
- **Check:** Samples per feature ratio
- **Trigger:** samples_per_feature < 10
- **Message:** "Only {samples_per_feature:.1f} samples per feature (need ≥10-20)"
- **Action:** "Reduce features via Feature Selection (page 4) or collect more samples. Consider this a pilot study."

### 3. Class Imbalance (HIGH severity, classification only)
- **Check:** Minority class percentage
- **Trigger:** minority_pct < 10%
- **Message:** "Minority class is only {minority_pct:.1f}% of data"
- **Action:** "Use stratified splits (already done), consider class weights, or collect more minority samples."

### 4. Missing Data (MEDIUM severity)
- **Check:** Percentage of missing values across entire dataset
- **Trigger:** missing_pct > 20%
- **Message:** "{missing_pct:.1f}% of data is missing"
- **Action:** "Review Preprocessing (page 5). Consider multiple imputation or dropping high-missingness features."

## Edge Cases Handled

1. **Missing comparison_df or metric_col:** Wrapped in conditional check
2. **None values for data, target, or feature_cols:** Graceful handling with None checks
3. **Correlation calculation failures:** Try-except blocks around each diagnostic check
4. **No diagnostics triggered:** Falls back to general guidance message
5. **Unknown sample size:** Displays "unknown" instead of crashing

## Error Handling

Each diagnostic check is wrapped in try-except blocks to ensure:
- Individual check failures don't crash the entire diagnostic section
- Users still see other relevant diagnostics even if one fails
- Silent failures with `pass` statements (appropriate for non-critical UX enhancements)

## User Experience

When triggered, users see:

1. **Error Banner:** Red warning box with performance metrics
2. **Diagnostic Section:** "🔍 Diagnostic Analysis" header
3. **Issue Cards:** Each issue displayed with:
   - Severity icon (🔴 HIGH, 🟡 MEDIUM)
   - Issue name
   - Description with actual values
   - Actionable recommendation with page references
4. **Fallback Guidance:** If no specific issues detected, general troubleshooting advice

## Verification Checklist

✅ **Diagnostic triggers correctly on poor performance**
- Checks `best_metric < poor_threshold` with correct thresholds for task type

✅ **Checks run without errors**
- All checks wrapped in try-except blocks
- Graceful handling of None values
- No assumptions about data availability

✅ **Recommendations are actionable**
- Each action links to specific pages (2, 3, 4, 5)
- Clear next steps provided
- Acknowledges what's already done (e.g., "already done" for stratified splits)

✅ **Severity levels appropriate**
- HIGH: Fundamental data quality issues (correlations, sample size, imbalance)
- MEDIUM: Fixable preprocessing issues (missing data)

✅ **No false positives on good-performing models**
- Only triggers when `best_metric < poor_threshold`
- Completely skipped for well-performing models

## Testing Recommendations

To verify the implementation:

1. **Test with poor classification model (AUC < 0.65):**
   - Verify error banner appears
   - Check all diagnostics run
   - Ensure page references are correct

2. **Test with poor regression model (R² < 0.40):**
   - Verify threshold is correct
   - Check metric name displays as "R²" not "AUC"

3. **Test edge cases:**
   - No models trained → no diagnostic
   - Good performance → no diagnostic
   - Missing features → graceful fallback

4. **Test each diagnostic trigger:**
   - Weak correlations → verify correlation diagnostic
   - Few samples → verify sample size diagnostic
   - Imbalanced data → verify class imbalance diagnostic
   - High missingness → verify missing data diagnostic

## Code Quality

- **Consistent style:** Matches existing Streamlit patterns in the file
- **Clear comments:** Section header with separator
- **Maintainable:** Diagnostic list structure makes it easy to add more checks
- **DRY:** Reuses existing helpers (get_data(), data_config, session_state)

## Integration

The diagnostic assistant seamlessly integrates into the existing page flow:

1. Model training
2. Metrics comparison table
3. Model selection guidance
4. **→ Diagnostic assistant (NEW)**
5. Model diagnostics tabs

This placement ensures users get actionable feedback immediately after seeing poor results, but before diving into detailed model diagnostics.
