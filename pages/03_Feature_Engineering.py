"""
🧬 Feature Engineering (Optional)

Create new features from existing data to improve model performance.
This step is OPTIONAL — skip if you prioritize interpretability over accuracy.
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

# Initialize
init_session_state()
inject_custom_css()
render_sidebar_workflow(current_page="03_Feature_Engineering")

st.title("🧬 Feature Engineering (⚠️ Experimental)")
render_breadcrumb("03_Feature_Engineering")
render_page_navigation("03_Feature_Engineering")

# Experimental banner
st.warning("""
⚠️ **EXPERIMENTAL FEATURE** — This page is in active testing and not yet in the main production branch.

Functionality is stable but UI and workflow may change based on feedback.  
Report issues: [GitHub](https://github.com/hedglinnolan/tabular-ml-lab/issues)
""")

# ============================================================================
# Introduction & Educational Content
# ============================================================================

st.markdown("""
### What is Feature Engineering?

**Feature engineering** is the art of creating new features (columns) from your existing data to help machine learning models 
find patterns more easily. Think of it as **translating your raw data into a language models understand better**.

**Example:** You have `height` and `weight`. A model might struggle to learn obesity patterns directly. 
But if you create `BMI = weight / height²`, the pattern becomes obvious.
""")

with st.expander("📚 Should I use Feature Engineering?", expanded=True):
    st.markdown("""
    ### When to Use Feature Engineering ✅
    
    - **Linear models** (Ridge, Lasso, Logistic) that can't capture non-linearity on their own
    - You have **domain knowledge** about useful combinations (e.g., ratios, products)
    - Your data has **interactions** between features (Age × Gender affects outcomes differently)
    - You're willing to **sacrifice some interpretability** for better accuracy
    - You'll run **Feature Selection** afterward to remove redundant features
    
    ### When to SKIP This Page ❌
    
    - You're using **tree-based models** (Random Forest, XGBoost) that handle non-linearity naturally
    - **Interpretability is critical** (clinical decisions, regulatory review)
    - You have a **small dataset** (<100 samples) — feature engineering can cause overfitting
    - Your features are **already well-engineered** (domain experts prepared the data)
    
    ### The Explainability Tradeoff ⚖️
    
    **Original features:** "BMI predicts diabetes with coefficient 0.8"  
    **After polynomial engineering:** "BMI² × Age predicts diabetes..." ← **Harder to explain!**
    
    **After TDA:** "Topological persistence entropy of homology dimension 1..." ← **Very hard to explain!**
    
    ⚠️ **Peer Reviewer Concern:** "Why did you engineer these features?" Be ready to justify!
    
    💡 **Recommendation:** Start WITHOUT feature engineering. Only add it if models underperform.
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

# Separate features and target
X = df.drop(columns=[target])
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

# Initialize tracking
X_engineered = X.copy()
engineered_features = []
engineering_log = []  # Track what was done for reporting

# ============================================================================
# Navigation: Skip or Proceed
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Choose whether to engineer features or skip this step:**")
with col2:
    if st.button("⏭️ Skip Feature Engineering", help="Proceed to Feature Selection with original features"):
        st.info("✅ Skipped feature engineering. Proceeding with original features.")
        # Make sure no engineered features are in session state
        st.session_state.pop("df_engineered", None)
        st.session_state["feature_engineering_applied"] = False
        st.success("👉 Continue to **Feature Selection** ")
        st.stop()

st.markdown("---")

# ============================================================================
# EDA-DRIVEN RECOMMENDATIONS
# ============================================================================
# Show recommendations based on EDA insights
eda_insights = st.session_state.get('feature_engineering_hints', {})

if eda_insights:
    skewed = eda_insights.get('skewed_features', [])
    high_corr = eda_insights.get('high_corr_pairs', [])
    
    if skewed or high_corr:
        st.markdown("### 💡 Recommendations Based on Your EDA")
        
        if skewed:
            skewed_names = [f"**{s['name']}** (skew: {s['skewness']})" for s in skewed]
            st.info(f"""
**Right-skewed features detected:** {', '.join(skewed_names)}

→ **Recommended:** Apply log transforms (Section 2) to normalize these distributions.  
This can improve linear model performance and reduce the influence of extreme values.
""")
        
        if high_corr:
            corr_str = ", ".join([f"**{p['feature1']}/{p['feature2']}** ({p['correlation']})" for p in high_corr[:3]])
            if len(high_corr) > 3:
                corr_str += f" ... and {len(high_corr) - 3} more"
            st.warning(f"""
**High correlation detected:** {corr_str}

→ **Caution:** Polynomial features will amplify these correlations.  
Consider using Feature Selection (next step) to filter redundant interactions.
""")
        
        st.markdown("---")

# ============================================================================
# Section 1: Polynomial Features & Interactions
# ============================================================================

st.header("1️⃣ Polynomial Features & Interactions")

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
                
                st.success(f"✅ Created **{len(new_cols):,} polynomial features**")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")

# ============================================================================
# Section 2: Domain-Specific Mathematical Transforms
# ============================================================================

st.header("2️⃣ Domain-Specific Transforms")

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

use_transforms = st.checkbox(
    "☐ Apply Mathematical Transforms",
    value=False
)

if use_transforms:
    st.caption("Select features to transform (numeric only):")
    
    # Pre-populate with skewed features from EDA
    eda_insights = st.session_state.get('feature_engineering_hints', {})
    skewed_feature_names = [s['name'] for s in eda_insights.get('skewed_features', [])]
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
                    engineering_log.append(f"Mathematical transforms: +{len(new_cols)} features")
                    st.success(f"✅ Created **{len(new_cols)} transformed features**")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown("---")

# ============================================================================
# Section 3: Ratio Features (Domain-Driven)
# ============================================================================

st.header("3️⃣ Ratio Features")

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
                        st.success(f"✅ Created **{len(new_cols)} ratio features**")
                        st.session_state.ratio_list = []  # Clear list
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown("---")

# ============================================================================
# Section 4: Binning / Discretization
# ============================================================================

st.header("4️⃣ Binning (Discretization)")

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
                    st.success(f"✅ Created **{len(new_cols)} binned features**")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown("---")

# ============================================================================
# Section 5: Topological Data Analysis (TDA) - Advanced
# ============================================================================

st.header("5️⃣ Topological Data Analysis (TDA) — Advanced")

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
                
                progress_bar.progress(100, "Complete!")
                
                st.success(f"✅ Created **{len(tda_feature_names)} TDA features**")
                
                with st.expander("View TDA features"):
                    st.dataframe(tda_df.describe())
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# Section 6: Dimensionality Reduction as Features
# ============================================================================

st.header("6️⃣ Dimensionality Reduction as Features")

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
                    
                    st.success(f"✅ Created **{n_components_pca} PCA features** ({pca.explained_variance_ratio_.sum():.1%} variance)")
                    
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
                    
                    st.success(f"✅ Created **{n_components_umap} UMAP features**")
                    
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
    
    with st.expander("📋 View All New Feature Names"):
        st.write(engineered_features)
    
    st.warning("""
    ⚠️ **Next Steps:**
    
    1. **Save** engineered features below
    2. Go to **Feature Selection**  to identify the most important features
    3. Feature selection is **strongly recommended** after engineering to remove redundant/unhelpful features
    
    **Explainability reminder:** Be prepared to justify feature engineering choices to peer reviewers!
    """)
    
    # Save button
    if st.button("💾 Save Engineered Features & Proceed", type="primary", key="save_btn"):
        df_engineered = pd.concat([X_engineered, y], axis=1)
        
        st.session_state["df_engineered"] = df_engineered
        st.session_state["feature_engineering_applied"] = True
        st.session_state["engineered_feature_names"] = engineered_features
        st.session_state["engineering_log"] = engineering_log
        
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
