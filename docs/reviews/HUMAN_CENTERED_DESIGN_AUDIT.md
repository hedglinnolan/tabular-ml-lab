# Human-Centered Design Audit
## Complete User Journey from Data Scientist Perspective

**Auditor Persona:** Data scientist familiar with statistics/data analysis, new to machine learning  
**Date:** 2026-03-10  
**Goal:** Evaluate if this is a **unified, coherent, peak human-centered design experience**

---

## 🚨 CRITICAL ISSUES FOUND

### Issue #1: Workflow Description Outdated ❌

**Problem:** Home page says "**9-Step Workflow**" but Feature Engineering makes it **10 steps**.

**Current home page:**
```
1. Upload & Audit
2. Explore (EDA)
3. Feature Selection  ← MISSING: Feature Engineering should be here!
4. Preprocess
5. Train & Compare
...
9. Export Report
```

**Actual workflow:**
```
1. Upload & Audit
2. EDA
2a. Feature Engineering ← NEW! But not mentioned on home page
3. Feature Selection
...
10. Export Report
```

**Impact:** 🔴 **HIGH** — User confusion about workflow, broken promise ("9 steps")

**Fix Required:**
- Update `app.py` workflow description to include Feature Engineering
- Update count: "9-Step Workflow" → "10-Step Workflow"
- Add step: "(2a) Feature Engineering (Optional) — Create polynomial, ratio, TDA features"

---

### Issue #2: Missing "Why This Step?" Connective Tissue 🟡

**Problem:** Pages explain WHAT to do, but not always WHY or HOW it connects to previous steps.

**Example - Feature Selection Page:**
- ✅ Explains what LASSO/RFE-CV are
- ❌ Doesn't explain: "You need this BECAUSE your EDA showed collinearity" or "BECAUSE you just engineered 50 features"
- ❌ Doesn't reference EDA findings

**Example - Preprocessing Page:**
- ✅ Explains imputation methods
- ❌ Doesn't say: "EDA found 15% missing in glucose → Here's how to handle that"

**Impact:** 🟡 **MEDIUM** — Workflow feels like disconnected steps, not a cohesive journey

**Fix Required:**
- Add "Why This Step?" section to each page
- Reference findings from previous steps (EDA → Feature Selection → Preprocessing)
- Show decision tree: "Your EDA showed X → Therefore, you should consider Y"

---

### Issue #3: Report Export Doesn't Capture Feature Engineering ❌

**Problem:** If report export was written before Feature Engineering was added, it may not document engineered features.

**Need to verify:**
- Does report mention which features were engineered?
- Does methodology section describe polynomial/TDA/ratio creation?
- Are engineered features in the feature importance tables?

**Impact:** 🔴 **HIGH** if missing — Methodology incomplete, not reproducible

**Status:** NEEDS VERIFICATION (checking report export now)

---

### Issue #4: No "Decision Support" for Optional Steps 🟡

**Problem:** User doesn't know WHEN to use Feature Engineering vs when to skip.

**Current:**
- ✅ Feature Engineering page has "When to use / when to skip" guidance
- ❌ But user only sees this AFTER navigating to the page
- ❌ No proactive recommendation based on their data

**Better approach:**
- EDA should analyze data and suggest: "Your data has high collinearity → Consider Feature Selection"
- Feature Selection should suggest: "You selected 50 features → Consider engineering interactions"

**Impact:** 🟡 **MEDIUM** — User makes uninformed decisions about optional steps

---

## 📋 Step-by-Step Workflow Audit

### Page 1: Upload & Audit

#### ✅ What User Learns:
- How to upload CSV/Excel
- Select target variable
- See data quality (missing, types, cardinality)
- Task type detection (regression vs classification)

#### ✅ Strengths:
- Clear interface
- Automatic task type detection
- Data quality warnings

#### ❌ Gaps:
- **No "What's Next?" guidance** at the bottom
- **No preview of workflow** ("After upload, you'll explore your data...")
- **No explanation of target variable** for ML novices ("This is what you're trying to predict")

#### 🔧 Recommended Fixes:
```markdown
**After upload, add:**

### What Happens Next?

You've uploaded your data and selected a target. Here's what comes next:

1. **Explore Your Data (EDA)** — See distributions, correlations, missing patterns
2. **Optional: Engineer Features** — Create new features if needed
3. **Select Features** — Identify the most predictive variables
4. **Train Models** — Compare 18 different algorithms
5. **Validate & Export** — Generate publication-ready reports

👉 **Continue to Explore (EDA)**
```

---

### Page 2: EDA (Exploratory Data Analysis)

#### ✅ What User Learns:
- Distribution of variables
- Correlations
- Missing data patterns
- Table 1 (baseline characteristics)
- Class balance (if classification)

#### ✅ Strengths:
- Comprehensive visualizations
- Table 1 is publication-ready
- AI interpretation available

#### ❌ Gaps:
- **No actionable recommendations** based on findings
  - "You have 40% missing in glucose" → What should I do about this?
  - "BMI and Weight are correlated 0.95" → Should I remove one?
- **No "breadcrumbs" to next steps**
  - "High collinearity detected → You'll address this in Feature Selection"
  - "Skewed features detected → You might consider log transforms in Feature Engineering"

#### 🔧 Recommended Fixes:

Add **"EDA Insights → Next Steps"** section at bottom:

```markdown
### 📊 Your EDA Findings

Based on your data analysis:

✅ **Detected:** 3 highly correlated feature pairs (>0.9)  
→ **Action:** Feature Selection (page 4) will help you choose which to keep

⚠️ **Detected:** 5 features with >20% missing data  
→ **Action:** Preprocessing (page 5) will handle imputation

✅ **Detected:** Skewed distributions in income, glucose  
→ **Consider:** Log transforms in Feature Engineering (optional, page 3)

👉 **Recommended next step:** Feature Engineering (optional) or Feature Selection
```

---

### Page 2a: Feature Engineering (NEW!)

#### ✅ What User Learns:
- What feature engineering is
- 6 techniques (polynomial, transforms, ratios, binning, TDA, PCA/UMAP)
- When to use vs skip each technique

#### ✅ Strengths:
- **Excellent educational content** (best in app!)
- Clear "when to use / when to skip"
- Honest about explainability tradeoffs
- Truly optional (skip button)

#### ❌ Gaps:
- **Not integrated with EDA findings** 
  - Doesn't say: "Your EDA showed right-skewed income → Consider log transform"
  - Doesn't pre-select skewed features for transforms
- **No connection to downstream impact**
  - "Creating 50 new features → You'll need Feature Selection next"

#### 🔧 Recommended Fixes:

Add **"Based on Your EDA:"** section that references actual findings:

```python
# In Feature Engineering page, after loading data:
eda_insights = st.session_state.get('eda_insights', [])

if eda_insights:
    st.info("""
    **💡 Based on your EDA:**
    
    - Right-skewed features detected: income, glucose
      → Consider log transforms (Section 2)
    
    - High correlation between BMI and weight  
      → Polynomial features may create redundancy (use with caution)
    
    - You have domain knowledge about meaningful ratios (BMI = weight/height²)  
      → Section 3 is perfect for this!
    """)
```

---

### Page 3: Feature Selection

#### ✅ What User Learns:
- LASSO path, RFE-CV, stability selection
- Which features are most predictive
- Consensus ranking

#### ✅ Strengths:
- Multiple methods for validation
- Clear visualizations
- Consensus ranking smart

#### ❌ Gaps:
- **No reference to EDA** 
  - "Remember your EDA showed BMI/Weight correlation → LASSO chose Weight (makes sense!)"
- **No reference to Feature Engineering**
  - Currently shows banner "🧬 49 new features created"
  - ✅ Good! But doesn't explain: "Many are redundant → Feature selection will filter them"
- **No guidance on HOW MANY features to select**
  - "You selected 15 features. Is that reasonable? Too many? Too few?"

#### 🔧 Recommended Fixes:

Add **"Interpreting Your Results"** box:

```markdown
### How Many Features Should I Keep?

**Your results:** LASSO selected 8 features, RFE selected 12, Stability selected 10

**Rule of thumb:** 
- Aim for ~10-20 features (or n_samples/10, whichever is smaller)
- Your dataset has 200 samples → target ~20 features max
- Consensus features (selected by 2+ methods) are most reliable

**Next:** These selected features will be used in Preprocessing and Training.
```

---

### Page 4: Preprocessing

#### ✅ What User Learns:
- Imputation strategies
- Scaling methods
- Handling outliers
- Per-model pipelines

#### ✅ Strengths:
- Model-specific pipelines (smart!)
- Clear explanations of each method

#### ❌ Gaps:
- **Doesn't reference EDA missing data analysis**
  - "Remember EDA showed glucose has 15% missing → Here's how we'll handle it"
- **Doesn't explain WHY scaling matters**
  - "Ridge/Lasso require scaling, but Random Forest doesn't → We'll scale for linear models only"
- **No preview of impact**
  - "Your preprocessing will transform X features → Here's a preview..."

#### 🔧 Recommended Fixes:

At top, add **"Your Data Audit:"**

```markdown
### Your Data Needs

Based on Upload & EDA findings:

**Missing Data:**
- glucose: 15% missing → Will use [selected method]
- age: 3% missing → Will use [selected method]

**Feature Types:**
- 8 numeric → Will scale for linear models
- 3 categorical → Will one-hot encode

**Outliers:**
- BMI has extreme values (>50) → Will [selected handling]

These choices will be applied during training.
```

---

### Page 5: Train & Compare

#### ✅ What User Learns:
- 18 models compared
- Bootstrap confidence intervals
- Baseline comparison
- Which model performs best

#### ✅ Strengths:
- Comprehensive model zoo
- Bootstrap CIs (publication-grade)
- Baseline comparison (reviewer-proof)

#### ❌ Gaps:
- **No "What happened to my data?" explanation**
  - User uploaded 100 features, engineered 50 more, selected 15 → Where's the summary?
- **No callback to Feature Selection**
  - "Training on your 15 selected features: [list]"
- **No guidance on WHICH model to choose**
  - "All 3 models perform similarly → How do I decide?"

#### 🔧 Recommended Fixes:

Add **"Training Configuration"** summary box at top:

```markdown
### What You're Training On

**Data:** 200 samples (140 train, 30 val, 30 test)  
**Features:** 15 selected features (from original 100)  
**Target:** diabetes_status (classification)  
**Preprocessing:** Applied per-model pipelines (see Preprocessing page)

**Selected Features:**
- BMI, age, glucose, HbA1c, [+ 11 more]
- (2 engineered: log_glucose, BMI_squared)
```

Add **"Model Selection Guide"** after results:

```markdown
### How to Choose Your Model

**If models perform similarly (within confidence intervals):**
1. Choose simpler model (Logistic > Random Forest > Neural Net)
2. Prioritize interpretability if needed for publication
3. Check calibration (page 7)

**Your case:** Ridge, Random Forest, and XGBoost all achieve 0.82 AUC
→ **Recommendation:** Ridge (most interpretable) or Random Forest (robust)
```

---

### Page 6: Explainability

#### ✅ What User Learns:
- SHAP values
- Permutation importance
- Calibration
- Feature importance

#### ✅ Strengths:
- Multiple explainability methods
- SHAP is gold standard
- Calibration plots publication-ready

#### ❌ Gaps:
- **If features were engineered, SHAP shows engineered names**
  - "BMI_squared" is important → But user may forget they created this
  - **No link back to Feature Engineering:** "This is the squared BMI you created"
- **No guidance on interpreting engineered features**
  - "TDA_H1_entropy is your 2nd most important feature" → What does that mean??

#### 🔧 Recommended Fixes:

Add **"Feature Engineering Reminder"** when applicable:

```python
if st.session_state.get('feature_engineering_applied'):
    engineered_names = st.session_state.get('engineered_feature_names', [])
    
    st.info(f"""
    **💡 Remember:** {len(engineered_names)} features were engineered (page 3).
    
    Engineered features in your model:
    - log_glucose ← Log transform of glucose
    - BMI_squared ← Polynomial feature (BMI²)
    
    If interpreting these for publication, explain the transformation.
    """)
```

---

### Page 7: Sensitivity Analysis

#### ✅ What User Learns:
- Random seed robustness
- Feature dropout analysis

#### ✅ Strengths:
- Tests model stability
- Good for reviewer concerns

#### ❌ Gaps:
- **Unclear what "sensitive to seed" means**
  - "Your model's AUC varies 0.78-0.84 across seeds" → Is that good or bad?
- **No guidance on what to do if unstable**
  - "Model is sensitive" → So what should I do now?

#### 🔧 Recommended Fixes:

Add **interpretation thresholds:**

```markdown
### Interpreting Seed Sensitivity

**Your result:** AUC varies 0.78 – 0.84 (range: 0.06)

**Interpretation:**
- Range <0.03: ✅ Very stable
- Range 0.03-0.05: 🟡 Moderate stability (acceptable)
- Range >0.05: ⚠️ High sensitivity (concerning for publication)

**Your case:** 0.06 range suggests model is **moderately unstable**

**What to do:**
- Report confidence intervals (not just point estimates)
- Consider ensemble methods (averaging multiple seeds)
- Mention in limitations section
```

---

### Page 8: Hypothesis Testing

#### ✅ What User Learns:
- Statistical tests (t-test, ANOVA, chi-square)
- Non-ML approach

#### ✅ Strengths:
- Good for exploratory analysis
- Complements ML workflow

#### ❌ Gaps:
- **Unclear when to use this vs ML**
  - "You just trained ML models... why am I doing t-tests now?"
- **No connection to ML results**
  - "ML found BMI important, and t-test confirms BMI differs between groups"

#### 🔧 Recommended Fixes:

Add **"Why Hypothesis Testing?"** intro:

```markdown
### Why Use Statistical Tests After ML?

**ML tells you:** "BMI is important for predicting diabetes"  
**Statistical tests tell you:** "BMI is significantly different between diabetic and non-diabetic groups (p<0.001)"

**Use this page to:**
1. Validate ML findings with traditional statistics
2. Generate p-values for publication tables
3. Explore relationships without ML assumptions

**Note:** This is complementary to ML, not a replacement.
```

---

### Page 9: Report Export

#### ⚠️ CRITICAL: Need to Verify Feature Engineering Capture

**Must check:**
1. Does methods section mention feature engineering steps?
2. Are engineered features listed in results?
3. Is the engineering log included?

**Let me check the report export page:**


---

## 🔍 Report Export Verification

### ❌ CRITICAL: Feature Engineering NOT Captured

**Checked:** `ml/publication.py` → `generate_methods_section()`

**Result:** NO mention of:
- `feature_engineering_applied`
- `engineered_feature_names`  
- `engineering_log`

**Impact:** 🔴 **CRITICAL**
- Methods section incomplete
- Results not reproducible
- Violates publication standards

**What's Missing:**
1. **Methods section** should include:
   - "Feature engineering was performed prior to feature selection"
   - "Polynomial features (degree 2) were created from [X] numeric features"
   - "Log transforms applied to skewed features: income, glucose"
   - "Topological Data Analysis (TDA) features computed via persistent homology"

2. **Feature importance tables** should mark engineered features:
   - ✅ "BMI" (original)
   - 🧬 "log_glucose" (engineered via log transform)
   - 🧬 "BMI_squared" (engineered via polynomial degree 2)

3. **Workflow diagram** should include Feature Engineering step

---

## 📊 Summary: Critical Gaps

### 🔴 Must Fix (Breaks Core Experience):

1. **Home page workflow outdated** (says 9 steps, now 10)
2. **Report export missing feature engineering** (methods incomplete)
3. **No connective tissue between steps** (feels disconnected)

### 🟡 Should Fix (Improves Experience):

4. **No EDA → Feature Engineering recommendations** (data-driven guidance missing)
5. **No Feature Selection → Preprocessing flow** (doesn't reference selected features)
6. **No model selection guidance** (when models perform similarly)
7. **No sensitivity analysis interpretation** (what does "unstable" mean?)

### 🟢 Nice to Have (Polish):

8. **Explainability doesn't link back to Feature Engineering** (reminders for engineered features)
9. **Hypothesis Testing intro unclear** (why use this after ML?)
10. **No "Training Configuration" summary** (what data am I training on?)

---

## 🎯 Recommendations for Peak Human-Centered Design

### Principle 1: **Continuous Narrative**

Every page should answer:
1. **Where am I?** (Breadcrumb + step indicator) ✅ Already have
2. **Why am I here?** (How this step connects to previous) ❌ MISSING
3. **What should I do?** (Clear guidance) ✅ Mostly have
4. **What happens next?** (Preview of next step) ❌ MISSING

**Example pattern to add to EVERY page:**

```markdown
### Why This Step?

Based on your previous work:
- [Summary of what you did before]
- [How this step builds on that]
- [What you'll decide in this step]

### What Happens Next?

After completing this step:
- [What you'll have learned/created]
- [How the next step will use this]
```

---

### Principle 2: **Data-Driven Guidance**

Don't just explain methods — **recommend based on user's data**.

**Examples:**

**In EDA:**
```python
if has_high_missing(df):
    st.warning("""
    ⚠️ **Action Needed:** 5 features have >20% missing data
    
    → In Preprocessing (page 5), you'll need to choose an imputation strategy.
    Consider: Multiple imputation or dropping these features in Feature Selection.
    """)

if has_skewed_features(df):
    st.info("""
    💡 **Consider:** 3 features are right-skewed (income, glucose, BMI)
    
    → Feature Engineering (page 3) offers log transforms to normalize these.
    This can improve linear model performance.
    """)
```

**In Feature Engineering:**
```python
eda_insights = st.session_state.get('eda_insights', [])

# Pre-select skewed features for log transform
skewed_features = [f for f in eda_insights if f['type'] == 'skewed']
if skewed_features:
    st.info(f"""
    💡 **Your EDA detected** right-skewed features: {', '.join(skewed_features)}
    
    These are pre-selected below for log transformation (Section 2).
    """)
    # Auto-populate multiselect with skewed features
    selected_features = st.multiselect(..., default=skewed_features)
```

---

### Principle 3: **Methodology Capture as First-Class Feature**

Every action should ask: **"Will this be in the final report?"**

**Required fixes:**

1. **Capture everything in session state:**
   ```python
   st.session_state['methodology_log'] = []
   
   # Each page appends to log:
   st.session_state['methodology_log'].append({
       'step': 'Feature Engineering',
       'action': 'Applied log transform to: income, glucose',
       'timestamp': datetime.now()
   })
   ```

2. **Generate methods section from log:**
   ```python
   def generate_methods_from_log(log):
       sections = {
           'Feature Engineering': [],
           'Feature Selection': [],
           'Preprocessing': [],
           # ...
       }
       
       for entry in log:
           sections[entry['step']].append(entry['action'])
       
       return format_methods_section(sections)
   ```

3. **Preview in each step:**
   ```markdown
   ### Methods Section Preview

   **This will appear in your methods section:**

   > Feature engineering was performed using scikit-learn (v1.3.0). 
   > Log transformations were applied to right-skewed features (income, glucose) 
   > to normalize distributions. Polynomial features (degree 2) were generated 
   > for all numeric predictors, resulting in 45 additional interaction terms.
   ```

---

### Principle 4: **Progressive Disclosure with Escape Hatches**

- **Beginners:** See smart defaults + guidance
- **Experts:** Can expand advanced options
- **Everyone:** Can skip optional steps

**Current state:**
- ✅ Feature Engineering is optional (skip button)
- ✅ Advanced options in expanders
- ❌ No "Quick Mode" vs "Full Mode" toggle

**Consider adding:**
```python
mode = st.radio("Analysis depth:", ["Quick (use defaults)", "Full (customize each step)"])

if mode == "Quick":
    # Auto-select reasonable defaults
    # Show one-click "Run All" button
    st.button("🚀 Run Full Workflow with Defaults")
else:
    # Show all customization options
    # Proceed page by page
```

---

## ✅ Immediate Action Items

### Priority 1 (Critical - Do First):

1. **Update home page workflow** (app.py)
   - [ ] Change "9-Step Workflow" → "10-Step Workflow"
   - [ ] Add Feature Engineering to step list
   - [ ] Mark it as "(Optional)"

2. **Add Feature Engineering to report export** (ml/publication.py)
   - [ ] Check `st.session_state.feature_engineering_applied`
   - [ ] Include engineering_log in methods section
   - [ ] Mark engineered features in results tables

3. **Add "Why This Step?" to each page**
   - [ ] EDA: Explain why exploration before engineering
   - [ ] Feature Engineering: Reference EDA findings
   - [ ] Feature Selection: Reference engineering, explain filtering
   - [ ] Preprocessing: Reference missing data from EDA
   - [ ] Train: Reference selected features
   - [ ] Explainability: Reference trained models
   - [ ] Sensitivity: Explain stability testing
   - [ ] Hypothesis Testing: Explain validation role
   - [ ] Report: Summarize full workflow

### Priority 2 (High - Do Soon):

4. **EDA → Feature Engineering recommendations**
   - [ ] Detect skewed features → suggest log transforms
   - [ ] Detect high correlation → warn about polynomial explosion
   - [ ] Detect meaningful ratios → suggest ratio engineering

5. **Add methodology logging throughout**
   - [ ] Create global methodology_log in session state
   - [ ] Each page appends actions
   - [ ] Report generates from log

6. **Add model selection guidance** (page 6)
   - [ ] When models are tied, explain how to choose
   - [ ] Reference interpretability, deployment, robustness

### Priority 3 (Medium - Polish):

7. **Add "Training Configuration" summary** (page 6)
   - [ ] Show: n_samples, n_features (original vs selected), target
   - [ ] Reference Feature Selection choices
   - [ ] Show preprocessing configuration

8. **Add sensitivity interpretation guide** (page 8)
   - [ ] Thresholds: stable (<0.03), moderate (0.03-0.05), unstable (>0.05)
   - [ ] Recommendations for each case

9. **Explainability feature engineering reminders** (page 7)
   - [ ] When engineered features are important, remind user how they were created
   - [ ] Link back to Feature Engineering page

### Priority 4 (Low - Nice to Have):

10. **Add "Quick Mode"** option
    - [ ] Toggle between guided (page-by-page) and automated (one-click)
    - [ ] Auto-apply reasonable defaults
    - [ ] Generate full report at end

---

## 🎓 Educational Continuity Assessment

### Current Score: 7/10

**Strengths:**
- ✅ Feature Engineering page is **exceptionally educational** (10/10)
- ✅ Clear visualizations in EDA
- ✅ Good explanations of individual methods
- ✅ TRIPOD checklist helps with publication standards

**Weaknesses:**
- ❌ Pages feel isolated (6/10 on continuity)
- ❌ No "learning journey" narrative (5/10)
- ❌ Missing "why" connective tissue (5/10)
- ❌ Report doesn't capture full methodology (4/10)

### To Reach 10/10:

1. **Add narrative flow** — Each step explicitly references previous steps
2. **Data-driven recommendations** — Based on user's actual data, not generic advice
3. **Complete methodology capture** — Everything in the report, reproducible
4. **Guided decision support** — App actively helps user make informed choices

---

## 🏆 Vision: Peak Human-Centered Design

Imagine this experience:

1. **User uploads data** 
   → App says: "Great! Your data has 200 samples, 10 features, 1 target. Next, let's explore it."

2. **User does EDA**
   → App says: "I found 3 skewed features and high correlation between BMI/Weight. In the next step (Feature Engineering), I'll suggest log transforms for the skewed features."

3. **User goes to Feature Engineering**
   → App says: "Based on your EDA, I pre-selected income, glucose, BMI for log transforms. Click 'Apply' or customize."

4. **User applies transforms, sees:** "✅ Created 3 log features. These will be in your final report."

5. **User goes to Feature Selection**
   → App says: "You now have 13 features (10 original + 3 engineered). Let's find the most predictive ones."

6. **User selects 8 features**
   → App says: "You selected 8 features (6 original, 2 engineered). These will be used for training."

7. **User trains models**
   → App shows: "Training on your 8 selected features: [list with 🧬 icons for engineered ones]"

8. **User exports report**
   → Report says: "Feature engineering was performed. Log transforms applied to income, glucose, BMI based on right-skewed distributions observed in exploratory analysis."

**Every step connected. Every decision explained. Every action captured.**

---

## Final Recommendation

**Current state:** Good educational tool with excellent individual components.

**To reach peak HCD:** Add connective tissue, data-driven guidance, and complete methodology capture.

**Critical path:**
1. Fix home page workflow ✅ 30 min
2. Add feature engineering to report ✅ 2 hours  
3. Add "Why This Step?" to each page ✅ 4 hours
4. Add EDA-driven recommendations ✅ 6 hours

**Total effort:** ~2 days to transform from "good tool" to "exceptional unified experience"

**Status:** Ready to implement. All issues identified with clear fixes.

