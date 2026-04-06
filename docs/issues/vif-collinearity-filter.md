# Add VIF-based collinearity filter to feature selection pipeline

**Type:** Enhancement
**Priority:** High
**Affects:** Feature Selection (Page 04), Explainability (Page 07)

## Problem

The feature selection consensus list can retain all features from highly correlated clusters, violating VIF > 5 constraints and inflating importance metrics. This was observed with an NHANES dataset where **weight, BMI, and waist circumference** (max r=0.90) all survived consensus selection despite measuring essentially the same underlying construct.

### Specific concerns

1. **Correlated cluster retention**: The consensus list retains all three features from the weight/BMI/waist cluster (max r=0.90), which violates the VIF > 5 constraint for reliable importance metrics. The model's reliance on interaction terms (e.g., `age_x_bmi`) may be inflated by multicollinearity.

2. **Domain-relevant feature exclusion**: Critical domain-relevant features like **protein** were excluded despite being a standard nutritional metric, potentially biasing the model toward energy-dense nutrients (kcal, sugar, fat) and ignoring protein's role in glucose metabolism.

3. **Skewness + collinearity compounding**: Target skew (4.58) combined with heavily skewed selected features (triglycerides, sugar) risks violating linear regression assumptions, as the model may be fitting non-linear patterns without explicit transformation or robust loss functions.

## Current State

The app **detects** multicollinearity but does not **act** on it during feature selection:

- **VIF computation** exists in `ml/eda_actions.py` (lines 1013-1050) -- diagnostic only, displayed on the EDA page
- **Correlation pair detection** in `ml/eda_recommender.py` (lines 207-225) identifies pairs with |r| > 0.85
- **Coaching system** recommends regularization or feature selection but cannot auto-filter
- **`docs/llm-analyst-profile-design.md`** already states: *"VIF > 5 means SHAP/permutation importance rankings are unreliable"*
- **Feature selection methods** (LASSO, RFE-CV, Univariate, Stability) in `ml/feature_selection.py` handle collinearity only implicitly -- LASSO may suppress one of a correlated pair, but this is not guaranteed when all are strong predictors
- **`consensus_features()`** (line 273) uses simple vote counting with no redundancy filter

## Proposed Solution

Add an **opt-in post-consensus VIF filter** that iteratively removes the highest-VIF feature until all remaining features satisfy VIF < threshold.

### Why post-consensus?

- **Pre-filter** would discard features before methods can vote, hiding information from the user
- **5th voting method** conflates redundancy (inter-feature) with relevance (feature-target) -- a category error
- **Post-consensus** cleanly separates the two questions: "What matters?" then "Are any of those measuring the same thing?"

### Implementation sketch

#### 1. New `vif_filter()` function in `ml/feature_selection.py`

```python
def vif_filter(
    X: np.ndarray,
    feature_names: List[str],
    vif_threshold: float = 5.0,
) -> FeatureSelectionResult:
```

**Algorithm** (iterative backward elimination):
1. Compute VIF for all features in the consensus list
2. While any feature has VIF > threshold: drop the feature with the highest VIF, recompute
3. Return `FeatureSelectionResult` with surviving features and details (initial VIFs, final VIFs, drop order)

Important: VIF must be computed on the **consensus subset** of the data matrix, not the full feature set, because VIF values change depending on which other features are in the model.

#### 2. UI on Page 04 (`pages/04_Feature_Selection.py`)

Add between the consensus display and the "Apply" button:
- Checkbox: **"Apply VIF collinearity filter"** (default OFF)
- Slider: **VIF threshold** (range 2-10, default 5)
- Results table: features with VIF values, color-coded (green < 5, yellow 5-10, red > 10)
- Dropped features list with explanation (e.g., "waist_circumference dropped: VIF=23.4, correlated with BMI")

#### 3. Insight Ledger integration

- When VIF filter drops features, resolve corresponding `eda_corr_cluster_*` insights
- When VIF filter is NOT enabled but collinear features survive consensus, surface a warning:
  > "Consensus list retains N features from collinearity cluster(s). VIF > 5 means SHAP/permutation importance rankings are unreliable for linear models. Enable the VIF collinearity filter to retain one representative per cluster."

#### 4. Provenance & narrative

- `log_methodology()` records VIF threshold, dropped features, final VIF values
- Narrative engine generates: *"A VIF-based collinearity filter (threshold = 5) was applied post-consensus, removing N features to ensure no predictor shared >80% of its variance with others."*

#### 5. Tests

- `test_vif_filter_basic`: Two perfectly correlated features -> one dropped
- `test_vif_filter_threshold`: threshold=10 retains what threshold=5 drops
- `test_vif_filter_preserves_uncorrelated`: Independent features never dropped
- `test_vif_filter_iterative`: 3-feature cluster -> 2 dropped, 1 retained
- `test_consensus_with_vif_post_filter`: End-to-end consensus + VIF filter

## Acceptance Criteria

- [ ] `vif_filter()` function in `ml/feature_selection.py` with iterative backward elimination
- [ ] UI toggle and threshold slider on Feature Selection page
- [ ] VIF table with color-coded values displayed when filter is enabled
- [ ] Insight ledger resolves collinearity warnings when VIF filter is applied
- [ ] Warning surfaced when collinear features survive consensus without VIF filter
- [ ] Provenance logging and narrative engine integration
- [ ] Unit tests covering basic, threshold, iterative, and end-to-end scenarios
- [ ] Existing tests continue to pass

## Key Files

| File | Role |
|------|------|
| `ml/feature_selection.py` | Add `vif_filter()`, modify consensus pipeline |
| `pages/04_Feature_Selection.py` | Add VIF filter UI section |
| `ml/eda_actions.py` (lines 1013-1050) | Reference VIF computation (do not modify) |
| `ml/eda_recommender.py` (lines 207-225) | Reference correlation detection |
| `utils/insight_ledger.py` | Insight resolution integration |
| `ml/narrative_engine.py` | Manuscript prose for VIF filter step |
| `tests/test_feature_selection.py` | New VIF filter tests |
