# Testing Feature Engineering Page

**Branch:** `feature/feature-engineering`  
**New Page:** `02_5_Feature_Engineering.py` (appears between EDA and Feature Selection)

---

## Quick Start

### 1. Switch to Feature Branch

```bash
cd /home/claw/.openclaw/workspace/glucose-mlp-interactive
git checkout feature/feature-engineering
```

### 2. Install New Dependencies

```bash
# Activate your venv (if using one)
source venv/bin/activate  # or: . venv/bin/activate

# Install new packages
pip install giotto-tda>=0.6.0 umap-learn>=0.5.3

# Verify installation
python -c "import gtda; import umap; print('✓ Dependencies installed')"
```

**Note:** giotto-tda has heavy dependencies (scikit-learn, scipy, joblib). May take a few minutes to install.

### 3. Start the App

```bash
streamlit run app.py
```

Navigate to: `http://localhost:8501`

---

## Test Workflow

### Basic Test (Polynomial Features)

1. **Upload data:**
   - Go to "Upload & Audit"
   - Use `example_data.csv` (if available) or upload any CSV with numeric columns
   - Select a target variable

2. **Run EDA (optional):**
   - Go to page 2 (EDA)
   - Run distribution analysis (helps validate data is loaded)

3. **Feature Engineering:**
   - Go to page 2.5 (**🧬 Feature Engineering** - NEW!)
   - Should see: "Current dataset: N samples × M features"
   
4. **Test Polynomial Features:**
   - Check ☐ "Create Polynomial Features"
   - Select degree 2
   - Note the warning about feature count
   - Click "🔬 Generate Polynomial Features"
   - Should see: "✅ Created X polynomial features"
   
5. **Save:**
   - Scroll to bottom
   - See summary: "Original Features: M → Total Features: M+X"
   - Click "💾 Save Engineered Features"
   - Should see: "✅ Saved engineered dataset!" + balloons

6. **Verify downstream:**
   - Go to page 3 (Feature Selection)
   - Should see blue banner: "🧬 Feature Engineering Applied: Working with engineered dataset (X new features)"
   - Feature count should match engineered total

---

## Advanced Tests

### Test Domain Transforms

1. In Feature Engineering page:
   - Check ☐ "Apply Domain Transforms"
   - Select 2-3 numeric features
   - Select transforms: log(x+1), sqrt(x), x²
   - Click "🔬 Apply Transforms"
   - Verify new columns created: `log1p_<feature>`, `sqrt_<feature>`, `<feature>_squared`

2. **Edge case:** Select a feature with negative values, try log(x)
   - Should see warning: "⚠️ Skipped log(X): contains non-positive values"

### Test TDA (Persistent Homology)

**Warning:** This is computationally intensive. Test with small dataset first (<500 samples).

1. In Feature Engineering page:
   - Check ☐ "Compute TDA Features (Persistent Homology)"
   - Read the explainer (click expand)
   - Select homology dimensions: [0, 1] (default)
   - Set max edge length: 5.0
   - Check "Normalize features first" (recommended)

2. **If dataset >500 samples:**
   - Should see: "📊 Dataset has N samples. Consider subsampling..."
   - Check "Subsample for TDA computation"
   - Set subsample size to 500

3. Click "🔬 Compute TDA Features"
   - Progress bar should appear
   - May take 30 seconds to 2 minutes
   - Should see: "✅ Created X TDA features from persistent homology"

4. Expand "View TDA features"
   - Should see columns like: `TDA_H0_entropy`, `TDA_H1_bottleneck_amplitude`, etc.

### Test PCA as Features

1. Check ☐ "Add Dimensionality Reduction Features"
2. Select "PCA"
3. Set components: 5
4. Click "🔬 Compute PCA Features"
5. Should see: "✅ Created 5 PCA features (explaining XX% of variance)"

### Test UMAP

1. Select "UMAP"
2. Set components: 3
3. Set neighbors: 15
4. Click "🔬 Compute UMAP Features"
5. May take 30-60 seconds
6. Should see: "✅ Created 3 UMAP features"

---

## Expected Behavior

### Session State

After saving engineered features:
- `st.session_state.df_engineered` should contain the full dataset
- `st.session_state.feature_engineering_applied` should be `True`
- `st.session_state.engineered_feature_names` should list new feature names

### Downstream Pages

All pages that use `get_data()` should automatically see engineered features:
- Feature Selection (page 3) ✓
- Preprocess (page 4) ✓
- Train & Compare (page 5) ✓
- Explainability (page 6) ✓
- etc.

**Verify:** Feature count should match engineered total throughout the workflow.

---

## Known Limitations / Edge Cases

### 1. Feature Explosion Warning

- Polynomial degree 2 on 100 features → **5,050 features**
- App warns user but doesn't prevent it
- **Recommendation:** Always run Feature Selection after polynomial features

### 2. TDA Computation Time

- O(n³) complexity for n samples
- For >1000 samples, **strongly recommend subsampling**
- Progress bar shows stages but can appear stuck (it's computing)

### 3. UMAP May Fail on Small Datasets

- UMAP requires n_samples > n_neighbors
- If error occurs, reduce n_neighbors or use PCA instead

### 4. Memory Usage

- Large datasets + polynomial degree 3 can consume significant RAM
- Monitor memory if working with >10K samples + degree 3

### 5. Categorical Features

- Most transforms only work on numeric features
- Categorical features are preserved but not transformed
- No warning if user tries to transform categorical (silently skipped)

---

## Troubleshooting

### "giotto-tda not installed" error

```bash
pip install giotto-tda
```

If that fails (dependency conflicts):
```bash
pip install giotto-tda --no-deps
pip install scikit-learn scipy joblib numpy
```

### "umap-learn not installed" error

```bash
pip install umap-learn
```

### TDA hangs or takes very long

- Check dataset size
- Enable subsampling
- Reduce max_edge_length (try 2.0-3.0)
- Reduce homology dimensions (just use [0, 1])

### "Session state key not found" errors downstream

- Make sure you clicked "💾 Save Engineered Features"
- Restart app and try again (session state may be stale)

---

## What to Look For

### ✅ Good Signs

- New page appears in sidebar between "📊 EDA" and "🎯 Feature Selection"
- Feature count increases after engineering
- Blue banner appears on Feature Selection page
- Downstream pages show increased feature count
- Can train models on engineered features

### ❌ Red Flags

- Page doesn't appear in sidebar (check file naming: `02_5_...`)
- Save button doesn't create `df_engineered` in session state
- Feature Selection doesn't see new features
- Errors about missing columns in downstream pages
- Progress bars freeze indefinitely

---

## Testing Checklist

- [ ] Page appears in sidebar (between EDA and Feature Selection)
- [ ] Can create polynomial features (degree 2)
- [ ] Can create polynomial features (degree 3)
- [ ] Can apply domain transforms (log, sqrt, square)
- [ ] Domain transform handles negative values correctly (skips log)
- [ ] Can compute TDA features (small dataset, <500 samples)
- [ ] TDA subsampling works for large datasets
- [ ] Can create PCA features
- [ ] Can create UMAP features (if dataset >100 samples)
- [ ] "Save Engineered Features" button works
- [ ] Feature Selection page shows banner
- [ ] Feature count is correct in Feature Selection
- [ ] Can run Feature Selection on engineered features
- [ ] Can train models on engineered features
- [ ] All downstream pages see engineered features

---

## Reporting Issues

If you find bugs:

1. **Note the error message** (screenshot if possible)
2. **Check browser console** (F12 → Console tab)
3. **Check terminal** (where `streamlit run` is running)
4. **Describe steps to reproduce:**
   - What you clicked
   - Dataset size / features
   - Which engineering method
   - Expected vs actual behavior

---

## Next Steps (If Testing Passes)

Once you're satisfied with functionality:

1. **Merge to main:**
   ```bash
   git checkout main
   git merge feature/feature-engineering
   git push origin main
   ```

2. **Sync to deployment branches** (when ready):
   ```bash
   # For university-docker
   git checkout university-docker
   git cherry-pick <commit-hash-from-main>
   
   # For enterprise-docker
   git checkout enterprise-docker
   git cherry-pick <commit-hash-from-main>
   ```

3. **Update documentation:**
   - Add Feature Engineering to README.md
   - Update workflow diagram (now 10 steps instead of 9)

---

**Happy testing!** 🧬🔬
