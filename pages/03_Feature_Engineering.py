"""
🧬 Feature Engineering (Optional)

Create new features from existing data to improve model performance.
This step is OPTIONAL — skip if you prioritize interpretability over accuracy.

AUDIT NOTE (Data Flow):
- get_data() returns: df_engineered (if FE applied) > filtered_data > raw_data
- Operates on: data_config.feature_cols (user-selected features from Upload & Audit)
- Edge case handled: fe_work_in_progress tracks feature set hash and resets when features change
- Downstream invalidation: Saving FE clears preprocessing, models, splits, explainability
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from utils.session_state import get_data, init_session_state, log_methodology
from utils.theme import inject_custom_css, render_guidance, render_sidebar_workflow
from utils.storyline import render_breadcrumb, render_page_navigation


def _build_transform_map(engineered_features, engineering_log):
    """Map each engineered feature name to its transform type for downstream guardrails."""
    transform_map = {}
    for feat in engineered_features:
        if feat.startswith("log_") or feat.startswith("log1p_"):
            transform_map[feat] = "log"
        elif feat.startswith("sqrt_"):
            transform_map[feat] = "sqrt"
        elif feat.endswith("_squared"):
            transform_map[feat] = "power"
        elif feat.endswith("_cubed"):
            transform_map[feat] = "power"
        elif feat.startswith("inv_"):
            transform_map[feat] = "reciprocal"
        elif feat.startswith("PCA_"):
            transform_map[feat] = "pca"
        elif feat.startswith("UMAP_"):
            transform_map[feat] = "umap"
        elif feat.startswith("TDA_"):
            transform_map[feat] = "tda"
        elif "_div_" in feat:
            transform_map[feat] = "ratio"
        elif "_binned" in feat or "_bin_" in feat:
            transform_map[feat] = "binning"
        elif " " in feat:
            transform_map[feat] = "polynomial"
        else:
            transform_map[feat] = "other"
    return transform_map


# Initialize
init_session_state()
inject_custom_css()
render_sidebar_workflow(current_page="03_Feature_Engineering")

st.title("🧬 Feature Engineering")
st.caption("Advanced workflow step: expand the feature space only after the quick workflow baseline tells you richer features may be worth the complexity.")
render_breadcrumb("03_Feature_Engineering")
render_page_navigation("03_Feature_Engineering")

if st.session_state.get("workflow_mode", "quick") == "quick":
    st.info("🧭 **Advanced workflow step** — Complete the quick workflow first, then return here only if baseline performance suggests richer features are worth the complexity.")

with st.expander("📚 When to use Feature Engineering", expanded=False):
    st.markdown("""
**Feature Engineering ADDS new features** alongside originals (e.g., `Glucose` + `log_Glucose`).  
**Preprocessing (page 5) TRANSFORMS** features in-place (e.g., `Glucose` → `log(Glucose)`).

**Use this page when:**
- Baseline models are underperforming and you need another lever
- You're using linear models that can't capture non-linearity on their own
- You have domain knowledge about useful combinations (e.g., BMI = weight / height²)

**Skip if:** Tree-based models already perform well, interpretability is critical, or dataset is small (<100 samples).

⚠️ **Tradeoff:** Engineered features are harder to explain to reviewers. Start without — add only if needed.
""")

# ============================================================================
# Prerequisites & Setup
# ============================================================================

df = get_data()
if df is None:
    st.info("👈 Please upload data first in Upload & Audit")
    st.stop()

# Get data configuration
data_config = st.session_state.get('data_config')
if not data_config or not data_config.target_col:
    st.info("👈 Please configure your target variable in Upload & Audit")
    st.stop()

target = data_config.target_col

# Check if we're working with an already-engineered dataset
if st.session_state.get('feature_engineering_applied'):
    st.warning("""
    ⚠️ **Feature engineering already applied!**
    
    You're viewing an engineered dataset. To start over:
    1. Go back to Upload & Audit 
    2. Re-upload your data
    
    Or continue to modify the existing engineered features below.
    """)

# Separate features and target — ONLY use features the user selected in Upload & Audit
selected_features = data_config.feature_cols if data_config.feature_cols else [c for c in df.columns if c != target]
X = df[selected_features]
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()

if not numeric_features:
    st.error("❌ No numeric features found. Feature engineering requires at least one numeric column.")
    st.stop()

st.info(f"""
📊 **Current dataset:**  
- **{len(X):,} samples × {len(X.columns)} features**  
- {len(numeric_features)} numeric, {len(categorical_features)} categorical, {len(datetime_features)} datetime
""")

# Initialize tracking - persist across reruns using session state
# After getting selected_features, before initializing fe_work_in_progress:
import hashlib
_features_hash = hashlib.md5(",".join(sorted(selected_features)).encode()).hexdigest()[:8]

if ('fe_work_in_progress' not in st.session_state 
    or st.session_state.get('fe_reset_requested')
    or st.session_state.get('fe_features_hash') != _features_hash):
    st.session_state.fe_work_in_progress = {
        'X_engineered': X.copy(),
        'engineered_features': [],
        'engineering_log': []
    }
    st.session_state.fe_features_hash = _features_hash
    st.session_state.fe_reset_requested = False

# Get working copy from session state
X_engineered = st.session_state.fe_work_in_progress['X_engineered']
engineered_features = st.session_state.fe_work_in_progress['engineered_features']
engineering_log = st.session_state.fe_work_in_progress['engineering_log']

# ============================================================================
# Navigation: Skip or Proceed
# ============================================================================

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**Choose whether to engineer features or skip this step:**")
with col2:
    if st.button("🔄 Reset", help="Clear all engineered features and start over", type="secondary"):
        st.session_state.fe_reset_requested = True
        st.session_state.pop("df_engineered", None)
        st.session_state["feature_engineering_applied"] = False
        st.session_state.pop("engineered_feature_names", None)
        st.session_state.pop("engineering_log", None)
        st.session_state.pop("engineered_feature_transforms", None)
        # Restore original feature_cols so downstream pages don't
        # reference columns that no longer exist
        pre_fe = st.session_state.pop("pre_fe_feature_cols", None)
        if pre_fe is not None and data_config is not None:
            data_config.feature_cols = pre_fe
        # Cascade: clear downstream state that may reference engineered columns
        st.session_state.pop("feature_selection_results", None)
        st.session_state.pop("consensus_features", None)
        st.session_state["preprocessing_pipeline"] = None
        st.session_state["preprocessing_config"] = None
        st.session_state["preprocessing_pipelines_by_model"] = {}
        st.session_state["preprocessing_config_by_model"] = {}
        st.session_state["trained_models"] = {}
        st.session_state["model_results"] = {}
        st.session_state["fitted_estimators"] = {}
        st.session_state["fitted_preprocessing_pipelines"] = {}
        st.session_state["X_train"] = None
        st.session_state["X_val"] = None
        st.session_state["X_test"] = None
        st.session_state["y_train"] = None
        st.session_state["y_val"] = None
        st.session_state["y_test"] = None
        st.session_state["permutation_importance"] = {}
        st.session_state["partial_dependence"] = {}
        st.session_state["shap_results"] = {}
        st.session_state.pop("sensitivity_seed_results", None)
        st.session_state["report_data"] = None
        st.rerun()
with col3:
    if st.button("⏭️ Skip", help="Recommended for most first passes: continue with your original features"):
        st.info("✅ Skipped feature engineering. Proceeding with your original features — this is the recommended first pass for many projects.")
        # Make sure no engineered features are in session state
        st.session_state.pop("df_engineered", None)
        st.session_state["feature_engineering_applied"] = False
        st.success("👉 Continue to **Feature Selection** ")
        st.stop()

st.markdown("---")

# ============================================================================
# EDA-DRIVEN RECOMMENDATIONS (custom rendering for FE context)
# ============================================================================
from utils.insight_ledger import get_ledger as _get_ledger
_fe_ledger = _get_ledger()
_fe_unresolved = _fe_ledger.get_unresolved(page="03_Feature_Engineering")
_fe_skew_insights = [i for i in _fe_unresolved if "skew" in i.id or (i.category == "distribution" and "skew" in i.finding.lower())]
_fe_corr_insights = [i for i in _fe_unresolved if i.category == "relationship"]
_fe_topo_insights = [i for i in _fe_unresolved if i.category == "topology"]

if _fe_skew_insights or _fe_corr_insights or _fe_topo_insights:
    st.markdown("### 💡 Recommendations from EDA")

    if _fe_skew_insights:
        all_skewed_feats = []
        for i in _fe_skew_insights:
            if i.affected_features:
                skew_meta = i.metadata.get("features", {})
                for f in i.affected_features:
                    sv = skew_meta.get(f, i.metadata.get("skewness", "?"))
                    all_skewed_feats.append((f, sv))
        if all_skewed_feats:
            skewed_names = [f"**{f}** (skew: {s:.1f})" if isinstance(s, (int, float)) else f"**{f}**" for f, s in all_skewed_feats[:8]]
            st.info(f"""
**Right-skewed features detected:** {', '.join(skewed_names)}

→ **Recommended:** Apply log transforms (Section 2) to normalize these distributions.
This can improve linear model performance and reduce the influence of extreme values.
""")

    if _fe_corr_insights:
        corr_str = ", ".join([f"**{'/'.join(i.affected_features)}** ({i.metadata.get('correlation', '?'):.2f})" for i in _fe_corr_insights[:3] if i.affected_features])
        if len(_fe_corr_insights) > 3:
            corr_str += f" ... and {len(_fe_corr_insights) - 3} more"
        st.warning(f"""
**High correlation detected:** {corr_str}

→ **Caution:** Polynomial features will amplify these correlations.
Consider using Feature Selection (next step) to filter redundant interactions.
""")

    if _fe_topo_insights:
        for ins in _fe_topo_insights:
            st.info(f"💡 **{ins.finding}** → {ins.recommended_action}")

    st.markdown("---")

# ============================================================================
# Feature Engineering Techniques
# ============================================================================

st.markdown("### 🔧 Feature Engineering Techniques")
st.markdown("Enable the techniques you want to apply. Each creates new features alongside your originals.")

# Progress indicator
progress_data = []
if len(engineered_features) > 0:
    st.success(f"✅ **{len(engineered_features)} features created so far** — Continue adding more or scroll to Summary to save.")

# ============================================================================
# Section 1: Polynomial Features & Interactions
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(102,126,234,0.05) 100%); 
            border-left: 4px solid #667eea; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #667eea;">1️⃣ Polynomial Features & Interactions</h3>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** Create new features by multiplying existing features together and raising them to powers.

**Example:** From `[Age, BMI]` create `[Age, BMI, Age², BMI², Age×BMI]`

**When to use:** Linear models (Ridge, Lasso, Logistic) that can't model curves or interactions naturally.

**When to SKIP:** Tree models (Random Forest, XGBoost) already find interactions automatically.

**Explainability impact:** 🔴 **High** — "Age×BMI interaction" is harder to explain than "Age" or "BMI" alone.

**Scientific precedent:** Used in countless publications when linear models are preferred for simplicity.
""")

use_poly = st.checkbox(
    "☐ Create Polynomial Features & Interactions",
    value=False,
    help="Generate polynomial and interaction features up to specified degree"
)

if use_poly:
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        poly_degree = st.selectbox(
            "Degree",
            [2, 3],
            index=0,
            help="Degree 2 = squares + pairwise products (A², A×B). Degree 3 adds cubes + 3-way products."
        )
    
    with col2:
        interaction_only = st.checkbox(
            "Interaction terms only",
            value=False,
            help="Only create A×B, A×B×C, etc. Skip A², A³. Reduces feature count."
        )
    
    with col3:
        st.metric("Estimated new features", 
                  f"~{len(numeric_features) * (len(numeric_features) + 1) // 2 if poly_degree == 2 else len(numeric_features) * 3:,}")
    
    # Feature explosion warning
    n_numeric = len(numeric_features)
    if poly_degree == 2:
        est_features = n_numeric + (n_numeric * (n_numeric + 1)) // 2 if not interaction_only else n_numeric + (n_numeric * (n_numeric - 1)) // 2
    else:
        est_features = n_numeric * 4  # Rough estimate
    
    if est_features > 500:
        st.error(f"⚠️ **Feature explosion warning!** This will create ~{est_features:,} features. Strongly recommend running Feature Selection afterward.")
    elif est_features > 100:
        st.warning(f"⚠️ This will create ~{est_features:,} features. Feature selection recommended.")
    else:
        st.info(f"This will create ~{est_features:,} features.")
    
    if st.button("🔬 Generate Polynomial Features", key="poly_btn"):
        with st.spinner(f"Generating degree-{poly_degree} polynomial features..."):
            try:
                X_numeric = X_engineered[numeric_features]
                
                poly = PolynomialFeatures(
                    degree=poly_degree,
                    interaction_only=interaction_only,
                    include_bias=False
                )
                X_poly = poly.fit_transform(X_numeric)
                poly_feature_names = poly.get_feature_names_out(numeric_features)
                
                # Remove original features (they're duplicated)
                new_cols = [name for name in poly_feature_names if name not in numeric_features]
                new_indices = [i for i, name in enumerate(poly_feature_names) if name not in numeric_features]
                
                X_poly_new = X_poly[:, new_indices]
                poly_df = pd.DataFrame(X_poly_new, columns=new_cols, index=X_engineered.index)
                
                X_engineered = pd.concat([X_engineered, poly_df], axis=1)
                engineered_features.extend(new_cols)
                engineering_log.append(f"Polynomial degree {poly_degree} ({'interaction-only' if interaction_only else 'full'}): +{len(new_cols)} features")
                
                # Save back to session state
                st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                
                st.success(f"✅ Created **{len(new_cols):,} polynomial features**")
                st.rerun()  # Refresh to show updated summary
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---- Custom Interactions ----
st.markdown('#### Custom Interactions')
st.caption('Select specific feature pairs for domain-driven interactions. More surgical than generating all polynomial combinations.')

if len(numeric_features) >= 1:
    _ci_col1, _ci_col2, _ci_col3 = st.columns(3)
    with _ci_col1:
        _feat_a = st.selectbox('Feature A', numeric_features, key='ci_feat_a')
    with _ci_col2:
        _feat_b = st.selectbox('Feature B', numeric_features, key='ci_feat_b')
    with _ci_col3:
        _ci_op = st.selectbox('Operation', ['Multiply (A × B)', 'Divide (A / B)', 'Square (A²)'], key='ci_op')

    if st.button('Add Interaction', key='ci_add_btn'):
        if _ci_op == 'Square (A²)':
            _new_col_name = f'{_feat_a}_squared'
            _new_values = X_engineered[_feat_a] ** 2
        elif _ci_op == 'Multiply (A × B)':
            _new_col_name = f'{_feat_a}_x_{_feat_b}'
            _new_values = X_engineered[_feat_a] * X_engineered[_feat_b]
        else:  # Divide
            _new_col_name = f'{_feat_a}_div_{_feat_b}'
            _den = X_engineered[_feat_b]
            _new_values = np.where(_den != 0, X_engineered[_feat_a] / _den, np.nan)
            _new_values = pd.Series(_new_values, index=X_engineered.index)

        if _new_col_name in X_engineered.columns:
            st.warning(f'⚠️ Feature `{_new_col_name}` already exists. Skipping to avoid duplicates.')
        else:
            X_engineered[_new_col_name] = _new_values
            engineered_features.append(_new_col_name)
            engineering_log.append(f'Custom interaction ({_ci_op}): {_new_col_name}')
            st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
            st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
            st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
            st.rerun()
else:
    st.info('No numeric features available for custom interactions.')

# ============================================================================
# Section 2: Domain-Specific Mathematical Transforms
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(102,126,234,0.05) 100%); 
            border-left: 4px solid #667eea; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #667eea;">2️⃣ Domain-Specific Transforms</h3>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** Apply mathematical functions to individual features: log, sqrt, square, etc.

**Example:** `log(income)` for right-skewed financial data, `sqrt(count)` for Poisson-distributed counts.

**When to use:**
- Features are **heavily skewed** (long right tail) → log transform
- Features have **variance proportional to mean** (count data) → sqrt transform
- You have **domain knowledge** about functional relationships

**Explainability impact:** 🟡 **Medium** — "`log(glucose)`" is still interpretable, just transformed.

**Scientific precedent:** Log transforms are standard in biology, economics, epidemiology.
""")

st.caption("ℹ️ Transforms here create NEW columns alongside originals. In Preprocessing, they REPLACE features.")

use_transforms = st.checkbox(
    "☐ Apply Mathematical Transforms",
    value=False
)

if use_transforms:
    st.caption("Select features to transform (numeric only):")
    
    # Pre-populate with skewed features from EDA (via ledger)
    _skew_insights = _fe_ledger.get_unresolved(page="03_Feature_Engineering", category="distribution")
    skewed_feature_names = []
    for _si in _skew_insights:
        skewed_feature_names.extend(_si.affected_features)
    default_selection = [f for f in skewed_feature_names if f in numeric_features]
    
    selected_features = st.multiselect(
        "Features",
        numeric_features,
        default=default_selection,  # Pre-select skewed features from EDA
        help="Choose which features to transform. Original features will be kept. Pre-populated with skewed features detected in EDA."
    )
    
    if selected_features:
        transform_options = st.multiselect(
            "Transforms",
            ["log(x)", "log(x+1)", "sqrt(x)", "x²", "x³", "1/x"],
            default=["log(x+1)"],
            help="Select transforms to apply. Each creates a new column per selected feature."
        )
        
        if st.button("🔬 Apply Transforms", key="transform_btn"):
            with st.spinner("Applying transforms..."):
                try:
                    new_cols = []
                    
                    for feat in selected_features:
                        feat_data = X_engineered[feat]
                        
                        if "log(x)" in transform_options:
                            if (feat_data > 0).all():
                                X_engineered[f"log_{feat}"] = np.log(feat_data)
                                new_cols.append(f"log_{feat}")
                            else:
                                st.warning(f"⚠️ Skipped log({feat}): contains non-positive values")
                        
                        if "log(x+1)" in transform_options:
                            if (feat_data >= 0).all():
                                X_engineered[f"log1p_{feat}"] = np.log1p(feat_data)
                                new_cols.append(f"log1p_{feat}")
                            else:
                                st.warning(f"⚠️ Skipped log1p({feat}): contains negative values")
                        
                        if "sqrt(x)" in transform_options:
                            if (feat_data >= 0).all():
                                X_engineered[f"sqrt_{feat}"] = np.sqrt(feat_data)
                                new_cols.append(f"sqrt_{feat}")
                            else:
                                st.warning(f"⚠️ Skipped sqrt({feat}): contains negative values")
                        
                        if "x²" in transform_options:
                            X_engineered[f"{feat}_squared"] = feat_data ** 2
                            new_cols.append(f"{feat}_squared")
                        
                        if "x³" in transform_options:
                            X_engineered[f"{feat}_cubed"] = feat_data ** 3
                            new_cols.append(f"{feat}_cubed")
                        
                        if "1/x" in transform_options:
                            if (feat_data != 0).all():
                                X_engineered[f"inv_{feat}"] = 1.0 / feat_data
                                new_cols.append(f"inv_{feat}")
                            else:
                                st.warning(f"⚠️ Skipped 1/{feat}: contains zeros")
                    
                    engineered_features.extend(new_cols)
                    
                    # Save back to session state
                    st.session_state.fe_work_in_progress["X_engineered"] = X_engineered
                    st.session_state.fe_work_in_progress["engineered_features"] = engineered_features
                    st.session_state.fe_work_in_progress["engineering_log"] = engineering_log
                    
                    # Resolve skew insight if all skewed features have been transformed
                    # Resolve skew insight with structured details
                    _skew_ins = _fe_ledger.get("eda_skew_group")
                    if _skew_ins and _skew_ins.affected_features:
                        untransformed = [f for f in _skew_ins.affected_features if f not in selected_features]
                        if not untransformed:
                            _fe_ledger.resolve(
                                "eda_skew_group",
                                resolved_by=f"Applied transforms ({', '.join(transform_options)}) to all skewed features",
                                resolved_on_page="03_Feature_Engineering",
                                resolution_details={
                                    "action_type": "transform",
                                    "method": ", ".join(transform_options) if len(transform_options) > 1 else transform_options[0] if transform_options else "transform",
                                    "columns_affected": list(selected_features),
                                    "result": {"new_columns_created": len(new_cols), "columns": new_cols[:10]},
                                },
                            )
                    
                    st.rerun()
                    engineering_log.append(f"Mathematical transforms: +{len(new_cols)} features")
                    st.success(f"✅ Created **{len(new_cols)} transformed features**")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown("---")

# ============================================================================
# Section 3: Ratio Features (Domain-Driven)
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(102,126,234,0.05) 100%); 
            border-left: 4px solid #667eea; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #667eea;">3️⃣ Ratio Features</h3>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** Divide one feature by another to create meaningful ratios.

**Real-world examples:**
- **BMI** = weight / height²
- **Debt-to-income ratio** = total_debt / annual_income
- **Student-teacher ratio** = n_students / n_teachers

**When to use:** When domain knowledge suggests a ratio is more meaningful than individual features.

**Explainability impact:** 🟢 **Low** — Ratios are often MORE interpretable than raw features (e.g., "BMI" is clearer than "weight").

**Scientific precedent:** BMI, ratios, and normalized features are standard in clinical research.
""")

use_ratios = st.checkbox(
    "☐ Create Ratio Features",
    value=False
)

if use_ratios:
    st.caption("Define ratios to create:")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        numerator = st.selectbox("Numerator", [""] + numeric_features, key="ratio_num")
    with col2:
        denominator = st.selectbox("Denominator", [""] + numeric_features, key="ratio_den")
    with col3:
        if st.button("➕ Add Ratio"):
            if numerator and denominator and numerator != denominator:
                if "ratio_list" not in st.session_state:
                    st.session_state.ratio_list = []
                st.session_state.ratio_list.append((numerator, denominator))
    
    if "ratio_list" in st.session_state and st.session_state.ratio_list:
        st.write("**Ratios to create:**")
        for i, (num, den) in enumerate(st.session_state.ratio_list):
            col1, col2 = st.columns([4, 1])
            col1.write(f"- `{num} / {den}`")
            if col2.button("🗑️", key=f"del_ratio_{i}"):
                st.session_state.ratio_list.pop(i)
                st.rerun()
        
        if st.button("🔬 Create Ratios", key="ratio_btn"):
            with st.spinner("Creating ratio features..."):
                try:
                    new_cols = []
                    for num, den in st.session_state.ratio_list:
                        den_data = X_engineered[den]
                        if (den_data != 0).all():
                            ratio_name = f"{num}_div_{den}"
                            X_engineered[ratio_name] = X_engineered[num] / den_data
                            new_cols.append(ratio_name)
                            engineered_features.append(ratio_name)
                        else:
                            st.warning(f"⚠️ Skipped {num}/{den}: denominator contains zeros")
                    
                    if new_cols:
                        engineering_log.append(f"Ratio features: +{len(new_cols)} features")
                        
                        # Save back to session state
                        st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                        st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                        st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                        
                        st.session_state.ratio_list = []  # Clear list
                        st.success(f"✅ Created **{len(new_cols)} ratio features**")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# Section 4: Binning / Discretization
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(102,126,234,0.05) 100%); 
            border-left: 4px solid #667eea; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #667eea;">4️⃣ Binning (Discretization)</h3>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** Convert continuous numeric features into categorical bins (e.g., "low", "medium", "high").

**Example:** Age (0-100) → Age groups: [0-18, 18-65, 65+]

**When to use:**
- Non-linear relationships that are **piecewise constant** (plateaus, thresholds)
- Clinical guidelines use **cutoffs** (e.g., glucose <100 = normal, 100-125 = prediabetes, >125 = diabetes)
- You want **interpretable categories** instead of continuous values

**Explainability impact:** 🟢 **Improves explainability** — "High BMI category" is easier than "BMI=32.7"

**Caution:** Loses information (all values in a bin treated the same). Use only when justified.

**Scientific precedent:** Common in epidemiology, clinical trials (age groups, income brackets).
""")

use_binning = st.checkbox(
    "☐ Apply Binning / Discretization",
    value=False
)

if use_binning:
    selected_bin_features = st.multiselect(
        "Features to bin",
        numeric_features,
        default=[],
        help="Select continuous features to convert into categorical bins"
    )
    
    if selected_bin_features:
        col1, col2 = st.columns(2)
        with col1:
            n_bins = st.slider("Number of bins", 2, 10, 3, help="How many categories to create")
        with col2:
            strategy = st.selectbox(
                "Binning strategy",
                ["quantile", "uniform", "kmeans"],
                help="quantile=equal sample counts, uniform=equal bin widths, kmeans=cluster-based"
            )
        
        encode_as = st.radio(
            "Encoding",
            ["ordinal", "onehot"],
            help="ordinal=[0,1,2,...], onehot=separate binary column per bin"
        )
        
        if st.button("🔬 Apply Binning", key="bin_btn"):
            with st.spinner("Creating binned features..."):
                try:
                    new_cols = []
                    for feat in selected_bin_features:
                        discretizer = KBinsDiscretizer(
                            n_bins=n_bins,
                            encode='ordinal' if encode_as == 'ordinal' else 'onehot-dense',
                            strategy=strategy
                        )
                        
                        feat_data = X_engineered[[feat]]
                        binned = discretizer.fit_transform(feat_data)
                        
                        if encode_as == 'ordinal':
                            col_name = f"{feat}_binned"
                            X_engineered[col_name] = binned.flatten().astype(int)
                            new_cols.append(col_name)
                        else:  # onehot
                            for i in range(n_bins):
                                col_name = f"{feat}_bin_{i}"
                                X_engineered[col_name] = binned[:, i].astype(int)
                                new_cols.append(col_name)
                    
                    engineered_features.extend(new_cols)
                    engineering_log.append(f"Binning ({strategy}, {n_bins} bins): +{len(new_cols)} features")
                    
                    # Save back to session state
                    st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                    st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                    st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                    
                    st.success(f"✅ Created **{len(new_cols)} binned features**")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# Section 5: Topological Data Analysis (TDA) - Advanced
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255,152,0,0.1) 0%, rgba(255,152,0,0.05) 100%); 
            border-left: 4px solid #ff9800; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #ff9800;">5️⃣ Topological Data Analysis (TDA) — Advanced ⚠️</h3>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">Computationally intensive — Recommended only for advanced users</p>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** TDA captures the **shape** and **structure** of your data using algebraic topology. It computes **persistent homology** 
to find topological features (clusters, loops, voids) that persist across multiple scales.

**For non-experts:** Imagine your data points as stars in the sky. TDA asks:
- How many **clusters** (connected groups) are there?
- Are there any **loops** (circular patterns)?
- Are there any **voids** (hollow regions)?

And crucially: **Which of these structures persist as you zoom in/out?** Persistent features are real structure, not noise.

**When to use:**
- Data has **spatial/geometric structure** (patient trajectories, networks, point clouds)
- You suspect **manifold structure** (data lies on a curved surface in high dimensions)
- **Publication novelty** — TDA is cutting-edge, underutilized in ML

**When to SKIP:**
- Tabular data with **no spatial relationships** between samples
- **Small datasets** (<100 samples) — not enough structure
- **Interpretability is critical** — TDA features are abstract

**Explainability impact:** 🔴 **Very High** — "Persistent homology entropy of dimension 1" is nearly impossible to explain to non-experts.

**Scientific precedent:** Used in genomics, neuroscience, materials science. Rare in tabular ML → **good for novelty!**

**Computational cost:** O(n³) for n samples. **Strongly recommend subsampling for >500 samples.**
""")

use_tda = st.checkbox(
    "☐ Compute TDA Features (Persistent Homology)",
    value=False
)

if use_tda:
    st.warning("⚠️ **TDA is computationally intensive.** Expect 30 seconds to several minutes depending on dataset size.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        homology_dims = st.multiselect(
            "Homology Dimensions",
            [0, 1, 2],
            default=[0, 1],
            help="H₀=connected components, H₁=loops, H₂=voids"
        )
    
    with col2:
        max_edge_length = st.slider(
            "Max Edge Length",
            1.0, 10.0, 5.0, 0.5,
            help="Max distance for Vietoris-Rips complex"
        )
    
    with col3:
        normalize_tda = st.checkbox(
            "Normalize first",
            value=True,
            help="Recommended: scale features to [0,1]"
        )
    
    # Subsampling
    n_samples = len(X_engineered)
    if n_samples > 500:
        st.info(f"📊 Dataset has {n_samples:,} samples. **Subsampling recommended** to avoid long computation.")
        subsample = st.checkbox(
            "Subsample for TDA",
            value=True if n_samples > 1000 else False
        )
        
        if subsample:
            subsample_size = st.slider(
                "Subsample size",
                100,
                min(n_samples, 2000),
                min(500, n_samples),
                100
            )
    else:
        subsample = False
        subsample_size = n_samples
    
    if not homology_dims:
        st.error("❌ Select at least one homology dimension")
    elif st.button("🔬 Compute TDA Features", key="tda_btn"):
        with st.spinner("Computing persistent homology... This may take a few minutes."):
            try:
                try:
                    # Patch giotto-tda for sklearn >= 1.8 compatibility
                    # (force_all_finite renamed to ensure_all_finite)
                    from ml.compat import patch_gtda_for_sklearn
                    patch_gtda_for_sklearn()
                    from gtda.homology import VietorisRipsPersistence
                    from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
                except ImportError:
                    st.error("❌ giotto-tda not installed. Run: `pip install giotto-tda`")
                    st.stop()
                
                X_numeric = X_engineered[numeric_features].values
                
                if normalize_tda:
                    scaler = StandardScaler()
                    X_numeric = scaler.fit_transform(X_numeric)
                
                if subsample:
                    rng = np.random.RandomState(42)
                    indices = rng.choice(len(X_numeric), size=subsample_size, replace=False)
                    X_tda = X_numeric[indices]
                else:
                    X_tda = X_numeric
                
                # Treat entire dataset as one point cloud
                X_point_cloud = X_tda.reshape(1, subsample_size if subsample else n_samples, -1)
                
                vr = VietorisRipsPersistence(
                    homology_dimensions=homology_dims,
                    max_edge_length=max_edge_length,
                    n_jobs=-1
                )
                
                progress_bar = st.progress(0)
                progress_bar.progress(30, "Computing Vietoris-Rips complex...")
                
                diagrams = vr.fit_transform(X_point_cloud)
                
                progress_bar.progress(60, "Extracting features...")
                
                tda_features_list = []
                tda_feature_names = []
                
                # Persistence Entropy
                entropy = PersistenceEntropy()
                entropy_features = entropy.fit_transform(diagrams).flatten()
                for i, dim in enumerate(homology_dims):
                    tda_features_list.append(entropy_features[i])
                    tda_feature_names.append(f"TDA_H{dim}_entropy")
                
                # Amplitude
                for metric in ['bottleneck', 'wasserstein', 'landscape']:
                    try:
                        amp = Amplitude(metric=metric)
                        amp_features = amp.fit_transform(diagrams).flatten()
                        for i, dim in enumerate(homology_dims):
                            tda_features_list.append(amp_features[i])
                            tda_feature_names.append(f"TDA_H{dim}_{metric}_amplitude")
                    except:
                        pass
                
                # Number of points
                n_points = NumberOfPoints()
                n_points_features = n_points.fit_transform(diagrams).flatten()
                for i, dim in enumerate(homology_dims):
                    tda_features_list.append(n_points_features[i])
                    tda_feature_names.append(f"TDA_H{dim}_n_points")
                
                progress_bar.progress(90, "Adding TDA features...")
                
                tda_features_array = np.array(tda_features_list).reshape(1, -1)
                tda_df = pd.DataFrame(
                    np.tile(tda_features_array, (n_samples, 1)),
                    columns=tda_feature_names,
                    index=X_engineered.index
                )
                
                X_engineered = pd.concat([X_engineered, tda_df], axis=1)
                engineered_features.extend(tda_feature_names)
                engineering_log.append(f"TDA (H{homology_dims}): +{len(tda_feature_names)} features")
                
                # Save back to session state
                st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                
                progress_bar.progress(100, "Complete!")
                
                st.success(f"✅ Created **{len(tda_feature_names)} TDA features**")
                st.rerun()
                
                with st.expander("View TDA features"):
                    st.dataframe(tda_df.describe())
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# Section 6: Dimensionality Reduction as Features
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(102,126,234,0.05) 100%); 
            border-left: 4px solid #667eea; 
            padding: 1rem; 
            margin: 1.5rem 0; 
            border-radius: 8px;">
    <h3 style="margin: 0; color: #667eea;">6️⃣ Dimensionality Reduction as Features</h3>
</div>
""", unsafe_allow_html=True)

render_guidance("""
**What:** Instead of replacing features with PCA/UMAP (like in preprocessing), **add** low-dimensional embeddings 
as supplementary features alongside originals.

**When to use:**
- Tree models (Random Forest, XGBoost) — they can use both original + embeddings
- You want to **keep interpretability** of original features
- You suspect data lies on a **lower-dimensional manifold**

**Explainability impact:** 🟡 **Medium** — Original features stay interpretable, but "PC1" is abstract.

**Scientific precedent:** Common in genomics (thousands of genes → 10 PCA components + original features).
""")

st.warning("""
⚠️ **vs. Preprocessing:** Here, PCA/UMAP components are ADDED alongside originals (e.g., 10 features → 10 originals + 3 PCA = 13 total). In Preprocessing (page 5), PCA REPLACES originals (10 → 3). Use this for supplementary features, use Preprocessing for dimensionality reduction.
""")

use_dimred = st.checkbox(
    "☐ Add PCA or UMAP Features",
    value=False
)

if use_dimred:
    dimred_method = st.selectbox(
        "Method",
        ["PCA", "UMAP"],
        help="PCA=linear, UMAP=non-linear manifold"
    )
    
    if dimred_method == "PCA":
        n_components_pca = st.slider(
            "Components",
            2, min(20, len(numeric_features)),
            min(10, len(numeric_features))
        )
        
        if st.button("🔬 Compute PCA", key="pca_btn"):
            with st.spinner("Computing PCA..."):
                try:
                    X_numeric = X_engineered[numeric_features]
                    pca = PCA(n_components=n_components_pca)
                    X_pca = pca.fit_transform(X_numeric)
                    
                    pca_cols = [f"PCA_{i+1}" for i in range(n_components_pca)]
                    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X_engineered.index)
                    
                    X_engineered = pd.concat([X_engineered, pca_df], axis=1)
                    engineered_features.extend(pca_cols)
                    engineering_log.append(f"PCA: +{n_components_pca} features ({pca.explained_variance_ratio_.sum():.1%} variance)")
                    
                    # Save back to session state
                    st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                    st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                    st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                    
                    st.success(f"✅ Created **{n_components_pca} PCA features** ({pca.explained_variance_ratio_.sum():.1%} variance)")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    else:  # UMAP
        col1, col2 = st.columns(2)
        with col1:
            n_components_umap = st.slider("Components", 2, 10, 3)
        with col2:
            n_neighbors = st.slider("Neighbors", 5, 100, 15)
        
        if st.button("🔬 Compute UMAP", key="umap_btn"):
            with st.spinner("Computing UMAP... (can take a few minutes)"):
                try:
                    import umap
                    X_numeric = X_engineered[numeric_features]
                    
                    reducer = umap.UMAP(
                        n_components=n_components_umap,
                        n_neighbors=n_neighbors,
                        random_state=42,
                        n_jobs=-1
                    )
                    X_umap = reducer.fit_transform(X_numeric)
                    
                    umap_cols = [f"UMAP_{i+1}" for i in range(n_components_umap)]
                    umap_df = pd.DataFrame(X_umap, columns=umap_cols, index=X_engineered.index)
                    
                    X_engineered = pd.concat([X_engineered, umap_df], axis=1)
                    engineered_features.extend(umap_cols)
                    engineering_log.append(f"UMAP: +{n_components_umap} features")
                    
                    # Save back to session state
                    st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
                    st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
                    st.session_state.fe_work_in_progress['engineering_log'] = engineering_log
                    
                    st.success(f"✅ Created **{n_components_umap} UMAP features**")
                    st.rerun()
                    
                except ImportError:
                    st.error("❌ umap-learn not installed. Run: `pip install umap-learn`")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# Summary & Save
# ============================================================================

st.markdown("---")
st.header("📊 Summary")

original_features = len(X.columns)
current_features = len(X_engineered.columns)
new_features = current_features - original_features

col1, col2, col3 = st.columns(3)
col1.metric("Original Features", original_features)
col2.metric("New Features Created", new_features, delta=f"+{new_features}" if new_features > 0 else "0")
col3.metric("Total Features", current_features)

if new_features > 0:
    st.success(f"✅ Feature engineering created **{new_features:,} new features**")
    
    with st.expander("🔍 View Engineering Log"):
        for log_entry in engineering_log:
            st.write(f"- {log_entry}")
    
    if engineered_features:
        st.markdown('### Engineered Features')
        _features_to_remove = []
        for i, feat in enumerate(engineered_features):
            col_feat, col_btn = st.columns([4, 1])
            with col_feat:
                st.markdown(f'`{feat}`')
            with col_btn:
                if st.button('❌', key=f'remove_fe_{i}', help=f'Remove {feat}'):
                    _features_to_remove.append(feat)

        if _features_to_remove:
            for feat in _features_to_remove:
                if feat in X_engineered.columns:
                    X_engineered = X_engineered.drop(columns=[feat])
                if feat in engineered_features:
                    engineered_features.remove(feat)
            st.session_state.fe_work_in_progress['X_engineered'] = X_engineered
            st.session_state.fe_work_in_progress['engineered_features'] = engineered_features
            st.rerun()

    # Check if already saved
    already_saved = st.session_state.get('feature_engineering_applied', False)
    
    if already_saved:
        st.success("""
        ✅ **Features already saved!**
        
        Your engineered features are part of the working dataset.
        
        👉 **Next:** Navigate to **Feature Selection** to identify the most important features.
        """)
    else:
        st.warning("""
        ⚠️ **Next Steps:**
        
        1. **Save** engineered features below
        2. Go to **Feature Selection**  to identify the most important features
        3. Feature selection is **strongly recommended** after engineering to remove redundant/unhelpful features
        
        **Explainability reminder:** Be prepared to justify feature engineering choices to peer reviewers!
        """)
        
        # Preview correlations with target
        if engineered_features:
            with st.expander('📊 Preview: Top features by correlation with target', expanded=False):
                _preview_corrs = []
                for feat in engineered_features:
                    if feat in X_engineered.columns:
                        try:
                            corr = abs(X_engineered[feat].corr(y))
                            if not np.isnan(corr):
                                _preview_corrs.append((feat, corr))
                        except:
                            pass
                _preview_corrs.sort(key=lambda x: x[1], reverse=True)
                if _preview_corrs:
                    _preview_df = pd.DataFrame(_preview_corrs[:10], columns=['Feature', '|Correlation with target|'])
                    st.dataframe(_preview_df, hide_index=True)
                    st.caption('Higher correlation suggests the feature may be predictive. Use Feature Selection (next page) for rigorous evaluation.')
                else:
                    st.caption('Could not compute correlations for engineered features.')

        # Save button (only show if not already saved)
        # Warn about downstream invalidation
        _has_downstream = any([
            st.session_state.get('preprocessing_pipeline'),
            st.session_state.get('trained_models'),
            st.session_state.get('X_train') is not None,
        ])
        if _has_downstream:
            st.warning(
                "⚠️ **Saving will reset downstream work.** Your preprocessing pipelines, "
                "data splits, trained models, and explainability results will be cleared. "
                "You will need to re-run those steps."
            )
        if st.button("💾 Save Engineered Features & Proceed", type="primary", key="save_btn"):
            df_engineered = pd.concat([X_engineered, y], axis=1)
            
            # Preserve original feature_cols so reset can restore them
            if "pre_fe_feature_cols" not in st.session_state:
                st.session_state["pre_fe_feature_cols"] = list(data_config.feature_cols) if data_config.feature_cols else list(selected_features)
            
            st.session_state["df_engineered"] = df_engineered
            st.session_state["feature_engineering_applied"] = True
            st.session_state["engineered_feature_names"] = engineered_features
            st.session_state["engineering_log"] = engineering_log
            
            # Track which transforms produced each engineered feature
            st.session_state["engineered_feature_transforms"] = _build_transform_map(engineered_features, engineering_log)
            
            # CASCADE INVALIDATION: clear all downstream state
            st.session_state.pop("feature_selection_results", None)
            st.session_state.pop("consensus_features", None)
            st.session_state["preprocessing_pipeline"] = None
            st.session_state["preprocessing_config"] = None
            st.session_state["preprocessing_pipelines_by_model"] = {}
            st.session_state["preprocessing_config_by_model"] = {}
            st.session_state["trained_models"] = {}
            st.session_state["model_results"] = {}
            st.session_state["fitted_estimators"] = {}
            st.session_state["fitted_preprocessing_pipelines"] = {}
            st.session_state["X_train"] = None
            st.session_state["X_val"] = None
            st.session_state["X_test"] = None
            st.session_state["y_train"] = None
            st.session_state["y_val"] = None
            st.session_state["y_test"] = None
            st.session_state["permutation_importance"] = {}
            st.session_state["partial_dependence"] = {}
            st.session_state["shap_results"] = {}
            st.session_state.pop("sensitivity_seed_results", None)
            st.session_state["report_data"] = None
            
            # Log methodology action
            log_methodology(
                step='Feature Engineering',
                action=f"Created {len(engineered_features)} engineered features",
                details={
                    'techniques': engineering_log,
                    'feature_count': len(engineered_features)
                }
            )
            
            st.success(f"✅ Saved engineered dataset! ({len(engineered_features)} new features)")
            st.balloons()
            st.info("""
            👉 **Next step: Feature Selection**
            
            Your engineered features are now part of the working dataset.
            Navigate to **Feature Selection** to identify the most important features.
            """)
            # Force page rerun to show updated state
            st.rerun()
else:
    st.info("ℹ️ No feature engineering applied yet. Enable techniques above, or click **⏭️ Skip** at the top to proceed with original features.")

# Navigation hint
st.markdown("---")
st.markdown("""
**Navigation:**
- ← **Back to EDA** to review data distributions
- → **Next: Feature Selection** to identify predictive features (required after engineering!)
- ⏭️ **Skip this page** (button at top) if you want to proceed with original features
""")
