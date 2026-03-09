# Testing Feature Engineering Page - Comprehensive Guide

**Branch:** `feature/feature-engineering`  
**New Page:** `02_5_Feature_Engineering.py` — appears between EDA and Feature Selection  
**Status:** Optional step (can be skipped)

---

## What Was Built

A **comprehensive, educational feature engineering page** with:

1. **Clear intro**: Explains what feature engineering is, when to use it, when to skip it
2. **Explainability warnings**: Honest about the interpretability tradeoff  
3. **Six proven techniques**:
   - Polynomial features & interactions
   - Domain-specific mathematical transforms
   - Ratio features (BMI-style)
   - Binning/discretization
   - Topological Data Analysis (TDA)
   - Dimensionality reduction as features (PCA/UMAP)
4. **Beginner-friendly**: Guidance blocks explain each technique with real-world examples
5. **System integration**: Seamlessly connects to downstream pages (can also be skipped)

---

## Quick Start

### 1. Switch to Feature Branch

```bash
cd /home/claw/.openclaw/workspace/glucose-mlp-interactive
git checkout feature/feature-engineering
git status  # Should be on feature/feature-engineering
```

### 2. Install New Dependencies

```bash
# Activate venv if using one
source venv/bin/activate

# Install TDA and UMAP libraries
pip install giotto-tda>=0.6.0 umap-learn>=0.5.3

# Verify
python -c "import gtda, umap; print('✅ Dependencies installed')"
```

**Note:** `giotto-tda` has many dependencies. Installation may take 2-3 minutes.

### 3. Start App

```bash
streamlit run app.py
```

Navigate to: `http://localhost:8501`

---

## Core Test: Skip Feature Engineering (Verify Optional)

**This is the most important test** — the page should be truly optional.

1. **Upload data** (page 1)
2. **Run EDA** (page 2) — optional but helps verify data loaded
3. **Go to Feature Engineering** (page 2.5)
4. **At the top**, click **"⏭️ Skip Feature Engineering"**
5. **Verify:**
   - Should see: "✅ Skipped feature engineering"
   - Should see: "👉 Continue to Feature Selection"
6. **Go to Feature Selection** (page 3)
   - Should NOT see "🧬 Feature Engineering Applied" banner
   - Feature count should match original (no engineered features)
7. **Continue workflow** (Preprocess → Train)
   - Everything should work normally with original features

**✅ Expected:** Skipping the page causes no issues downstream.

---

## Feature Test #1: Polynomial Features

1. In Feature Engineering page, scroll to **"1️⃣ Polynomial Features & Interactions"**
2. **Read the guidance block** (click to expand if needed)
3. Check ☐ "Create Polynomial Features & Interactions"
4. Select degree: **2**
5. Leave "Interaction terms only" unchecked
6. Note the warning: "This will create ~X features"
7. Click **"🔬 Generate Polynomial Features"**
8. Should see: "✅ Created X polynomial features"
9. Scroll to **Summary** at bottom
   - Should show: "Original Features: M → New Features Created: +X → Total: M+X"
10. Click **"💾 Save Engineered Features & Proceed"**
11. Should see balloons 🎉
12. Go to **Feature Selection** (page 3)
    - Should see blue banner: "🧬 Feature Engineering Applied: Working with engineered dataset (X new features)"
    - Feature count should match engineered total

**✅ Expected:** Polynomial features created, saved, and visible downstream.

---

## Feature Test #2: Domain Transforms

1. In Feature Engineering page, scroll to **"2️⃣ Domain-Specific Transforms"**
2. Check ☐ "Apply Mathematical Transforms"
3. Select 2-3 numeric features from your dataset
4. Select transforms: **log(x+1)**, **sqrt(x)**, **x²**
5. Click **"🔬 Apply Transforms"**
6. Should see: "✅ Created Y transformed features"
7. Summary should update with new feature count

**Edge case to test:**
- Select a feature with **negative values**
- Try **log(x)** transform
- Should see: "⚠️ Skipped log(feature): contains non-positive values"

**✅ Expected:** Transforms apply correctly, edge cases handled gracefully.

---

## Feature Test #3: Ratio Features

1. Scroll to **"3️⃣ Ratio Features"**
2. **Read the guidance** (explains BMI example)
3. Check ☐ "Create Ratio Features"
4. Select **Numerator**: (pick a feature, e.g., "weight")
5. Select **Denominator**: (pick another, e.g., "height")
6. Click **"➕ Add Ratio"**
7. Should see: "- `weight / height`" in the list
8. Add another ratio (optional)
9. Click **"🔬 Create Ratios"**
10. Should see: "✅ Created Z ratio features"
11. List should clear after creation

**Edge case:**
- Try creating a ratio where denominator has **zeros**
- Should see: "⚠️ Skipped X/Y: denominator contains zeros"

**✅ Expected:** Ratios created, zero-division handled.

---

## Feature Test #4: Binning

1. Scroll to **"4️⃣ Binning (Discretization)"**
2. **Read the guidance** (explains age groups, clinical cutoffs)
3. Check ☐ "Apply Binning / Discretization"
4. Select 1-2 numeric features
5. Set bins: **3** (low/medium/high)
6. Strategy: **quantile**
7. Encoding: **ordinal** (simpler for first test)
8. Click **"🔬 Apply Binning"**
9. Should see: "✅ Created K binned features"
10. New columns like `feature_binned` should appear (values 0, 1, 2)

**Then test onehot:**
- Change encoding to: **onehot**
- Click apply again
- Should create 3 binary columns per feature: `feature_bin_0`, `feature_bin_1`, `feature_bin_2`

**✅ Expected:** Binning works for both ordinal and onehot encoding.

---

## Feature Test #5: TDA (Persistent Homology)

**⚠️ WARNING:** This is computationally expensive. Use **small dataset** (<500 samples) for first test.

1. Scroll to **"5️⃣ Topological Data Analysis (TDA)"**
2. **Read the guidance carefully** (explains TDA for non-experts)
3. Check ☐ "Compute TDA Features (Persistent Homology)"
4. **If dataset >500 samples:**
   - Should see warning about subsampling
   - Check "Subsample for TDA"
   - Set subsample size: **500**
5. Homology dimensions: **[0, 1]** (default)
6. Max edge length: **5.0**
7. Check "Normalize first": **Yes**
8. Click **"🔬 Compute TDA Features"**
9. **Progress bar** should appear with stages:
   - "Computing Vietoris-Rips complex..." (30%)
   - "Extracting features..." (60%)
   - "Adding TDA features..." (90%)
   - "Complete!" (100%)
10. May take **30 seconds to 2 minutes**
11. Should see: "✅ Created X TDA features"
12. Expand "View TDA features" to see columns like:
    - `TDA_H0_entropy`
    - `TDA_H1_bottleneck_amplitude`
    - etc.

**Edge cases:**
- **Large dataset (>1000 samples) without subsampling:** Will take very long (test at your own risk)
- **No homology dimensions selected:** Should see error "❌ Select at least one homology dimension"

**✅ Expected:** TDA computes successfully on subsampled data, creates 6-20 features.

---

## Feature Test #6: PCA as Features

1. Scroll to **"6️⃣ Dimensionality Reduction as Features"**
2. **Read the guidance** (explains PCA vs preprocessing)
3. Check ☐ "Add PCA or UMAP Features"
4. Select **PCA**
5. Components: **5**
6. Click **"🔬 Compute PCA"**
7. Should see: "✅ Created 5 PCA features (XX% variance)"
8. New columns: `PCA_1`, `PCA_2`, ... `PCA_5`

**✅ Expected:** PCA components added as new columns alongside originals.

---

## Feature Test #7: UMAP

**Note:** UMAP requires `n_samples > n_neighbors`. Test with dataset >100 samples.

1. Select **UMAP**
2. Components: **3**
3. Neighbors: **15**
4. Click **"🔬 Compute UMAP"**
5. May take **30-60 seconds**
6. Should see: "✅ Created 3 UMAP features"
7. New columns: `UMAP_1`, `UMAP_2`, `UMAP_3`

**Edge case (small dataset):**
- If dataset has <15 samples, reduce neighbors to match
- Or expect error about neighbors > samples

**✅ Expected:** UMAP embeddings added successfully.

---

## Integration Test: Full Workflow

This verifies the entire pipeline with engineered features.

1. **Upload data** (page 1)
2. **Run EDA** (page 2)
3. **Feature Engineering** (page 2.5):
   - Apply polynomial degree 2
   - Apply log transforms on 2 features
   - Create 1 ratio
   - **Save engineered features**
4. **Feature Selection** (page 3):
   - Should see banner
   - Run LASSO, RFE-CV
   - Should work on engineered features
   - Select top features
5. **Preprocess** (page 4):
   - Should see engineered feature count
   - Configure preprocessing
6. **Train & Compare** (page 5):
   - Train 2-3 models
   - Should see engineered features in feature importance
7. **Explainability** (page 6):
   - Run SHAP
   - Should work on engineered features

**✅ Expected:** Entire workflow works seamlessly with engineered features.

---

## Educational Content Test

**Goal:** Verify the page is beginner-friendly.

For each section (1️⃣ through 6️⃣):
1. **Read the guidance block**
2. Verify it answers:
   - **What** the technique does (with example)
   - **When** to use it
   - **When to skip** it
   - **Explainability impact** (🟢 low, 🟡 medium, 🔴 high)
   - **Scientific precedent** (real-world use cases)

**Check the top-level expander:**
- "📚 Should I use Feature Engineering?"
- Verify it has: ✅ When to Use, ❌ When to SKIP, ⚖️ Tradeoffs

**✅ Expected:** Explanations are clear, accurate, and helpful for non-experts.

---

## Edge Cases & Known Limitations

### 1. Feature Explosion with Polynomial Degree 3

- Test: 50+ features, degree 3
- Should see: **Red warning** about feature explosion
- Still allows computation (doesn't block)

### 2. TDA on Large Dataset (No Subsampling)

- Test: >1000 samples, no subsampling checked
- **Will take 5-10+ minutes** (O(n³) complexity)
- Progress bar may appear stuck (it's computing)

### 3. UMAP on Tiny Dataset

- Test: <20 samples, neighbors=15
- Should error: neighbors must be < n_samples

### 4. Ratio with Zero Denominator

- Test: Create ratio where denominator has zeros
- Should skip with warning, not crash

### 5. Log Transform on Negative Values

- Test: log(x) on feature with negatives
- Should skip with warning

### 6. Saving Without Creating Features

- Don't create any features, just click Save
- Should show: "ℹ️ No feature engineering applied yet"
- Save button should not activate (or if clicked, should warn)

---

## Troubleshooting

### "giotto-tda not installed"

```bash
pip install giotto-tda
```

If fails (dependency conflicts):
```bash
pip install --upgrade pip setuptools wheel
pip install giotto-tda --no-cache-dir
```

### "umap-learn not installed"

```bash
pip install umap-learn
```

### TDA Hangs Indefinitely

- **Cause:** Large dataset without subsampling
- **Solution:** Stop (Ctrl+C), restart app, enable subsampling

### "Session state key not found" Downstream

- **Cause:** Didn't click "Save" button
- **Solution:** Go back to Feature Engineering, click "💾 Save"

### Features Not Showing in Feature Selection

- **Cause:** Session state not saved
- **Solution:** Restart app, re-do engineering, click Save

### KBinsDiscretizer Error

- **Cause:** sklearn version mismatch
- **Solution:** `pip install --upgrade scikit-learn>=1.3.0`

---

## Testing Checklist

### Core Functionality
- [ ] Page appears in sidebar (between EDA and Feature Selection)
- [ ] Can SKIP feature engineering (button at top)
- [ ] Skipping works — downstream pages see original features
- [ ] Educational intro explains what feature engineering is
- [ ] "Should I use Feature Engineering?" expander is helpful

### Feature Creation
- [ ] Polynomial degree 2 works
- [ ] Polynomial degree 3 works
- [ ] Interaction-only option works
- [ ] Domain transforms work (log, sqrt, square, cube, inverse)
- [ ] Domain transforms skip invalid inputs (warnings shown)
- [ ] Ratio features work
- [ ] Ratio list UI works (add/delete)
- [ ] Binning works (ordinal encoding)
- [ ] Binning works (onehot encoding)
- [ ] TDA works on small dataset (<500 samples)
- [ ] TDA subsampling works on large dataset
- [ ] TDA progress bar shows stages
- [ ] PCA features work
- [ ] UMAP features work

### Integration
- [ ] Summary shows correct feature counts
- [ ] Engineering log tracks all operations
- [ ] Save button creates df_engineered in session state
- [ ] Feature Selection shows banner after engineering
- [ ] Feature Selection works on engineered features
- [ ] Preprocess page sees engineered features
- [ ] Train page works with engineered features
- [ ] Explainability works with engineered features

### UI/UX
- [ ] All guidance blocks have clear explanations
- [ ] Real-world examples given for each technique
- [ ] Explainability impact clearly marked (🟢🟡🔴)
- [ ] Warnings show for feature explosion
- [ ] Warnings show for computational cost (TDA)
- [ ] Edge case warnings work (negative log, zero division)
- [ ] Progress bars show for long operations

---

## What to Report

If you find issues, please note:

1. **Error message** (full text or screenshot)
2. **Browser console** (F12 → Console)
3. **Terminal output** (where streamlit is running)
4. **Steps to reproduce:**
   - What you clicked
   - Dataset size / feature count
   - Which technique
   - Expected vs actual behavior
5. **Python version**: `python --version`
6. **Dependency versions**: `pip list | grep -E "giotto|umap|sklearn"`

---

## After Testing: Merge to Main

Once satisfied:

```bash
cd /home/claw/.openclaw/workspace/glucose-mlp-interactive

# Make sure you're on feature branch
git branch --show-current  # Should show: feature/feature-engineering

# Switch to main and merge
git checkout main
git merge feature/feature-engineering

# Push to GitHub
git push origin main
```

To sync to deployment branches later:

```bash
# University-docker
git checkout university-docker
git cherry-pick <commit-hash>

# Enterprise-docker
git checkout enterprise-docker
git cherry-pick <commit-hash>
```

---

## Documentation Updates Needed After Merge

1. **README.md:**
   - Update workflow: "9 steps" → "10 steps"
   - Add "Feature Engineering (optional)" to workflow table

2. **QUICKSTART.md:**
   - Mention optional feature engineering step

3. **Screenshots:**
   - Capture Feature Engineering page for docs

---

**Happy testing!** 🧬🔬

Questions? Check the guidance blocks in the app — they explain each technique in detail.
