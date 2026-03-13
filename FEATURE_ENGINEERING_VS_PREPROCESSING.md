# Feature Engineering vs. Preprocessing: Critical Distinctions

**Date:** 2026-03-10  
**Issue:** Users may confuse Feature Engineering (page 03) with Preprocessing (page 05) because both offer PCA, log transforms, etc.

---

## Key Distinction

| Aspect | Feature Engineering (Page 03) | Preprocessing (Page 05) |
|--------|------------------------------|------------------------|
| **When** | BEFORE feature selection | AFTER feature selection |
| **Action** | **ADDS** new features | **TRANSFORMS** existing features |
| **Original features** | **Kept** | May be **replaced** |
| **Goal** | Expand feature space | Meet model requirements |
| **Example** | Glucose → Glucose + log_Glucose | Glucose → log(Glucose) [replaces] |

---

## Overlapping Techniques

### 1. Log Transforms

**Feature Engineering:**
```
Input:  [Glucose: 120, BMI: 28]
Output: [Glucose: 120, BMI: 28, log_Glucose: 4.79, log1p_BMI: 3.37]
        ↑ Original kept    ↑ New features added
```
- Creates NEW columns: `log_Glucose`, `log1p_BMI`, etc.
- Original features remain untouched
- Model can learn from both original AND transformed
- **Use case:** "Maybe glucose has a non-linear relationship — let the model decide"

**Preprocessing:**
```
Input:  [Glucose: 120, BMI: 28]
Output: [Glucose: 4.79, BMI: 3.37]
        ↑ Transformed in-place (original REPLACED)
```
- Applies log transform to ALL numeric features (or selected subset)
- Original values are GONE
- **Use case:** "Linear models assume normal distributions — transform skewed features"

---

### 2. PCA

**Feature Engineering:**
```
Input:  [Age, BMI, Glucose, Cholesterol] (4 features)
Output: [Age, BMI, Glucose, Cholesterol, PCA_1, PCA_2, PCA_3] (7 features)
        ↑ Original 4 kept           ↑ 3 PCA components ADDED
```
- Computes PCA on originals, ADDS components as new features
- Now have 7 features total (4 original + 3 PCA)
- **Use case:** "Add low-dimensional representations while keeping interpretable originals"

**Preprocessing:**
```
Input:  [Age, BMI, Glucose, Cholesterol] (4 features)
Output: [PC1, PC2, PC3] (3 features)
        ↑ Original 4 REPLACED by 3 components
```
- REPLACES original features with PCA components
- Dimensionality reduction: 4 → 3
- **Use case:** "Too many features causing overfitting — reduce to essential components"

---

### 3. Scaling/Normalization

**Feature Engineering:**
- NOT offered (doesn't make sense to add scaled versions as features)

**Preprocessing:**
```
Input:  [Age: 45, Income: 75000]
Output: [Age: 0.5, Income: 0.8]  (StandardScaler)
        ↑ Same features, different scale
```
- REPLACES values with scaled versions
- Required for distance-based models (KNN, SVM)
- **Use case:** "Features on different scales — need to normalize for model convergence"

---

## When to Use Each

### Use Feature Engineering (Page 03) When:
- You want to **help models find patterns** (polynomial, interactions, ratios)
- You're using **linear models** that can't model non-linearity naturally
- You want to **keep interpretability** (original features remain)
- You'll run **Feature Selection** afterward to filter unhelpful features

### Use Preprocessing (Page 05) When:
- You need to **meet model requirements** (scaling for KNN, normalization for neural nets)
- You have **too many features** and need dimensionality reduction
- You need to **handle missing data** (imputation)
- You want **consistent transformations** applied during train/test/validation

---

## UI Clarity Needed

### Feature Engineering Page Needs:
1. **Banner at top:**
   ```
   ℹ️ Feature Engineering ADDS new features alongside your originals.
   
   This is different from Preprocessing (page 5), which TRANSFORMS features in-place.
   
   Example: Log transform here creates "log_Glucose" as a NEW column.
             Log transform in Preprocessing REPLACES Glucose with log(Glucose).
   ```

2. **Per-technique callouts:**
   - PCA: "🔔 Note: This ADDS PCA components. To REPLACE features with PCA, use Preprocessing (page 5)."
   - Log: "🔔 Note: This CREATES new log-transformed columns. To TRANSFORM existing columns, use Preprocessing (page 5)."

3. **Summary section clarification:**
   ```
   📊 Summary
   
   Original features: 45 (unchanged)
   New features created: 83
   Total features: 128
   
   → All 128 features will be available for Feature Selection (next step)
   → Preprocessing (page 5) will transform these AFTER selection
   ```

---

## Implementation Plan

### 1. Add Disambiguation Banner (Top of Page 03)
```python
st.info("""
📌 **Feature Engineering vs. Preprocessing:**

**This page (Feature Engineering):** ADDS new features alongside originals  
**Preprocessing (page 5):** TRANSFORMS existing features in-place

Example:
- Here: Glucose → Glucose + log_Glucose (both available)
- Preprocessing: Glucose → log(Glucose) (original replaced)

→ Use Feature Engineering to expand options for models  
→ Use Preprocessing to meet model requirements (scaling, normalization)
""")
```

### 2. Add Per-Technique Notes
- Polynomial: No ambiguity (only here)
- Transforms: ⚠️ Overlap with preprocessing
- Ratios: No ambiguity (only here)
- Binning: No ambiguity (only here)
- TDA: No ambiguity (only here)
- PCA/UMAP: ⚠️ Overlap with preprocessing (PCA)

### 3. Update Summary Section
Show clear distinction between:
- Original features (unchanged)
- New features (added)
- Total features (going forward)

---

## Recommended UI Redesign

### Current Problems:
1. Long vertical scroll (800+ lines)
2. No visual grouping
3. Checkbox → Button → Success (3-step flow confusing)
4. No clear "you are here" indicator in workflow
5. Disambiguation missing

### Proposed Design:

**Layout:**
```
┌─────────────────────────────────────────────┐
│ ⚠️ EXPERIMENTAL | Disambiguation Banner     │
├─────────────────────────────────────────────┤
│ Tab 1: Polynomial & Interactions            │
│ Tab 2: Mathematical Transforms ⚠️           │
│ Tab 3: Ratios & Derived Features            │
│ Tab 4: Binning                              │
│ Tab 5: Advanced (TDA, PCA, UMAP) ⚠️         │
├─────────────────────────────────────────────┤
│ Summary: X original → Y new → Z total       │
│ [Reset] [Skip] [Save & Continue]            │
└─────────────────────────────────────────────┘

⚠️ = Disambiguation note in tab
```

**Benefits:**
- Tabs reduce scroll
- Clear separation of techniques
- Disambiguation notes only where needed
- Summary always visible
- Actions consolidated at bottom

---

## Next Steps

1. Add disambiguation banner (5 min)
2. Add per-technique notes for overlaps (10 min)
3. UI redesign with tabs (60 min)
4. Test full workflow (15 min)

**Total:** ~90 minutes

