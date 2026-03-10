# Feature Engineering Page: Comprehensive Fix Plan

**Date:** 2026-03-10  
**Status:** IN PROGRESS  
**Testing feedback from:** Nolan

---

## Issues Identified (Priority Order)

### 🔴 CRITICAL (Breaks Core Functionality)

#### ✅ #2: Missing Dependencies - FIXED
**Problem:** giotto-tda and umap-learn not installed  
**Status:** INSTALLED (with dependency warnings - see note below)  
**Actions taken:**
- Installed giotto-tda==0.6.2
- Installed umap-learn==0.5.7  
- Both already in requirements.txt

**⚠️ Dependency Conflict Note:**
- giotto-tda requires scikit-learn==1.3.2 (pins old version)
- Current env has scikit-learn 1.8.0 and numpy 2.4.3
- Conflict warnings present but imports should still work
- **Decision:** Proceed with warnings; TDA is optional feature with try/except

#### ⏳ #3: Save Functionality Broken - INVESTIGATING
**Problem:** "I run polynomial features then click save but app says feature engineering not applied"  
**Hypothesis:** Possible causes:
1. Button doesn't trigger session state update properly
2. Page doesn't rerun after save
3. `new_features` count check failing

**Action:** Will add debug mode and trace exact failure point

---

### 🟡 HIGH PRIORITY (User Confusion / Data Integrity)

#### #4: Data Pipeline Visibility - NOT STARTED
**Problem:** User can't see which dataset they're working with at each step

**Current behavior:**
- `get_data()` returns: df_engineered > filtered_data > raw_data (priority order)
- NO visual indication which one is active
- User navigates between pages blind to state changes

**Required fixes:**
1. Add banner to ALL pages showing current dataset:
   ```
   📊 Working Dataset: Engineered Data (45 original → 128 total features)
   ```

2. Feature Selection page:
   - Show which features are original vs engineered
   - Mark engineered features visually (e.g., `BMI_poly_2` vs `BMI`)

3. Preprocessing page (FYSA component):
   ```
   ℹ️ Data Source Summary
   - Original features: 45
   - Engineered features: 83
   - Total entering preprocessing: 128
   ```

4. Add `reset_feature_engineering()` function for clean slate

---

#### #5: Upload & Audit Redundancy Check - NOT STARTED
**Problem:** Upload page may have feature selection/imputation that conflicts with later steps

**Action needed:**
- Audit pages/01_Upload_and_Audit.py for any preprocessing
- Check if it modifies filtered_data
- Document interaction with Feature Engineering/Selection/Preprocess

---

#### #6: State Management Chaos - PARTIALLY ADDRESSED
**Scenarios that break:**

1. **User engineers → goes back to EDA:**
   - EDA runs on engineered features (polynomial terms, TDA)
   - Confusion: "Why do I see `BMI^2 x Age` in correlation matrix?"
   - **Fix needed:** Warning banner on EDA if feature_engineering_applied=True

2. **User engineers twice:**
   - Run polynomial: 10 → 55 features
   - Go back, add transforms: 55 → 110 features (creates features OF features)
   - **Fix needed:** Detect this and warn OR reset to original before re-engineering

3. **User wants ONLY engineered features:**
   - No way to drop originals currently
   - **Fix needed:** Add checkbox "Use only engineered features (exclude originals)"

**Proposed guardrails:**
- Detect backward navigation after engineering
- Show modal: "⚠️ You've already engineered features. Going back will show ENGINEERED data in EDA. Continue?"
- Add "Reset Feature Engineering" button to start clean
- Track engineering_applied_timestamp to detect state pollution

---

### 🟢 MEDIUM PRIORITY (UX / Branding)

#### #1: Rename to Experimental - NOT STARTED
**Actions:**
- Change page title: "🧬 Feature Engineering (⚠️ Experimental)"
- Add banner: "This feature is in active testing. Feedback welcome!"
- Update README.md: Note that app.tabularml.dev shows experimental branch
- Add note: "Not yet in main production branch"

---

#### #8: UI Redesign - NOT STARTED
**Current problems:**
- Long vertical scroll (800+ lines)
- Checkbox → Button → Save (3-step flow confusing)
- Each technique needs separate "Generate" button click
- No visual grouping

**Proposed redesign:**
1. Use tabs or accordion for techniques:
   - Tab 1: Polynomial & Interactions
   - Tab 2: Transforms
   - Tab 3: Ratios
   - Tab 4: Binning
   - Tab 5: Topological (TDA)
   - Tab 6: Dimensionality Reduction

2. Simplified flow:
   - Check techniques you want
   - Click ONE "Apply Selected Techniques" button
   - All techniques run in batch
   - Show progress bar

3. Better visual hierarchy:
   - Card-based layout for each technique
   - Estimated feature count badge
   - Warning indicators (explosion, RAM)

---

## Execution Order

**Phase 1 (Today):**
1. ✅ Install dependencies
2. ⏳ Debug save functionality (PRIORITY)
3. Add experimental branding
4. Test full workflow: Upload → Engineer → Select → Train

**Phase 2 (Next):**
5. Add dataset visibility banners
6. Add state management guardrails
7. Audit Upload & Audit redundancy

**Phase 3 (Polish):**
8. UI redesign (tabs/accordion)
9. Consolidated "Apply All" button
10. Better visual feedback

---

## Testing Protocol

After each fix:
1. Clear session state (refresh browser)
2. Upload sample data
3. Configure target in Upload & Audit
4. Navigate to Feature Engineering
5. Enable polynomial features
6. Click generate
7. Click save
8. Navigate to Feature Selection
9. Verify engineered features appear
10. Navigate back to EDA
11. Verify working with engineered data

---

## Notes for Nolan

**Dependency conflict:**
- giotto-tda pins old sklearn but we need new numpy
- Kept warnings, proceeding with latest versions
- TDA is already wrapped in try/except so failures are graceful

**Save functionality:**
- Need to trace exact failure - will add debug logging
- Suspect page doesn't rerun after button click
- May need to add `st.rerun()` after save

**Data flow complexity:**
- Current design allows flexible workflow
- Tradeoff: flexibility vs. guardrails
- Recommend adding MORE warnings rather than restricting flow

