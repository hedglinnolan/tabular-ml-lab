"""
🧬 Feature Engineering

Create new features from existing data to improve model performance.

Techniques:
- Polynomial features (degree 2, 3)
- Interaction terms (pairwise products)
- Domain-specific transforms (log, sqrt, square, inverse)
- Topological Data Analysis (TDA) - Persistent homology
- Dimensionality reduction as features (PCA, UMAP)
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from utils.session_state import get_data, init_session_state
from utils.theme import inject_custom_css, render_breadcrumb, render_page_navigation

# Initialize
init_session_state()
inject_custom_css()

st.title("🧬 Feature Engineering")
render_breadcrumb("02_5_Feature_Engineering")
render_page_navigation("02_5_Feature_Engineering")

# Prerequisites
df = get_data()
if df is None:
    st.info("👈 Please upload data first (page 1)")
    st.stop()

target = st.session_state.get("target")
if not target:
    st.info("👈 Please select target variable in Upload & Audit (page 1)")
    st.stop()

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

st.markdown("""
**Create new features** from existing data to capture complex patterns, interactions, and hidden structure.

**⚠️ Important Notes:**
- Feature engineering can create **many new features** (e.g., 100 features → 5,050 with polynomial degree 2)
- Always run **Feature Selection** (next page) after engineering to remove redundant features
- Some transforms destroy interpretability (TDA, PCA) but can improve accuracy
""")

# Track which features are numeric (required for most transforms)
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

if not numeric_features:
    st.error("❌ No numeric features found. Feature engineering requires at least one numeric column.")
    st.stop()

st.info(f"📊 Current dataset: **{len(X)} samples × {len(X.columns)} features** ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")

# Initialize feature list (will accumulate engineered features)
engineered_features = []
X_engineered = X.copy()

# ============================================================================
# Section 1: Polynomial Features
# ============================================================================

st.header("1️⃣ Polynomial Features")

with st.expander("ℹ️ What are polynomial features?", expanded=False):
    st.markdown("""
    **Polynomial features** create new features by raising existing features to powers and computing products.
    
    **Example:** For features `[A, B]`:
    - Degree 2: `[A, B, A², B², A×B]` (5 features)
    - Degree 3: `[A, B, A², B², AB, A³, B³, A²B, AB², A²B]` (10 features)
    
    **When to use:**
    - Linear models that need to capture non-linear relationships
    - Data where interactions between features matter (e.g., Age × BMI)
    
    **Caution:** Feature count grows as O(n^d) where n=features, d=degree. Use Feature Selection afterward!
    """)

use_poly = st.checkbox(
    "☐ Create Polynomial Features",
    value=False,
    help="Generate polynomial and interaction features up to specified degree"
)

if use_poly:
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        poly_degree = st.selectbox(
            "Polynomial Degree",
            [2, 3],
            index=0,
            help="Degree 2 = squares + pairwise products. Degree 3 adds cubes + 3-way products."
        )
    
    with col2:
        include_bias = st.checkbox(
            "Include bias (constant) term",
            value=False,
            help="Add a column of 1s (rarely needed)"
        )
    
    with col3:
        interaction_only = st.checkbox(
            "Interaction terms only (no powers)",
            value=False,
            help="Only create A×B, A×B×C, etc. Skip A², A³. Useful to reduce feature count."
        )
    
    # Estimate feature count
    n_numeric = len(numeric_features)
    if poly_degree == 2 and not interaction_only:
        est_features = n_numeric + (n_numeric * (n_numeric + 1)) // 2
    elif poly_degree == 2 and interaction_only:
        est_features = n_numeric + (n_numeric * (n_numeric - 1)) // 2
    elif poly_degree == 3 and not interaction_only:
        est_features = n_numeric + (n_numeric * (n_numeric + 1) * (n_numeric + 2)) // 6
    else:
        est_features = n_numeric * 2  # Rough estimate for degree 3 interaction-only
    
    st.warning(f"⚠️ This will create approximately **{est_features:,} features** (up from {n_numeric}). Feature selection is recommended afterward.")
    
    if st.button("🔬 Generate Polynomial Features", type="primary"):
        with st.spinner(f"Generating degree-{poly_degree} polynomial features..."):
            try:
                # Apply only to numeric features
                X_numeric = X_engineered[numeric_features]
                
                poly = PolynomialFeatures(
                    degree=poly_degree,
                    interaction_only=interaction_only,
                    include_bias=include_bias
                )
                X_poly = poly.fit_transform(X_numeric)
                
                # Get feature names
                poly_feature_names = poly.get_feature_names_out(numeric_features)
                
                # Remove original features (they're duplicated)
                # Keep only the new polynomial/interaction features
                new_cols = [name for name in poly_feature_names if name not in numeric_features]
                new_indices = [i for i, name in enumerate(poly_feature_names) if name not in numeric_features]
                
                X_poly_new = X_poly[:, new_indices]
                
                # Add to engineered dataset
                poly_df = pd.DataFrame(X_poly_new, columns=new_cols, index=X_engineered.index)
                X_engineered = pd.concat([X_engineered, poly_df], axis=1)
                
                engineered_features.extend(new_cols)
                
                st.success(f"✅ Created **{len(new_cols)} polynomial features**")
                st.session_state["poly_features_created"] = True
                
            except Exception as e:
                st.error(f"❌ Error creating polynomial features: {e}")

# ============================================================================
# Section 2: Domain-Specific Transforms
# ============================================================================

st.header("2️⃣ Domain-Specific Transforms")

with st.expander("ℹ️ What are domain transforms?", expanded=False):
    st.markdown("""
    **Domain transforms** apply mathematical functions to individual features to:
    - Normalize skewed distributions (log, sqrt)
    - Create non-linear representations (square, inverse)
    - Match domain knowledge (e.g., log(glucose) for clinical data)
    
    **Common transforms:**
    - `log(x)`: Reduces right skew (e.g., income, viral load)
    - `sqrt(x)`: Moderate variance stabilization
    - `x²`: Amplifies large values
    - `1/x`: Inverse relationship
    
    **Note:** Original features are kept (these create NEW columns).
    """)

use_transforms = st.checkbox(
    "☐ Apply Domain Transforms",
    value=False,
    help="Create transformed versions of selected features"
)

if use_transforms:
    st.caption("Select features to transform (numeric only):")
    
    selected_features = st.multiselect(
        "Features to transform",
        numeric_features,
        default=[],
        help="Choose which features to apply transforms to"
    )
    
    if selected_features:
        transform_options = st.multiselect(
            "Transforms to apply",
            ["log(x)", "log(x+1)", "sqrt(x)", "x²", "1/x"],
            default=["log(x+1)"],
            help="Select one or more transforms. Each creates a new column per selected feature."
        )
        
        if st.button("🔬 Apply Transforms", type="primary"):
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
                        
                        if "1/x" in transform_options:
                            if (feat_data != 0).all():
                                X_engineered[f"inv_{feat}"] = 1.0 / feat_data
                                new_cols.append(f"inv_{feat}")
                            else:
                                st.warning(f"⚠️ Skipped 1/{feat}: contains zeros")
                    
                    engineered_features.extend(new_cols)
                    st.success(f"✅ Created **{len(new_cols)} transformed features**")
                    st.session_state["domain_transforms_created"] = True
                    
                except Exception as e:
                    st.error(f"❌ Error applying transforms: {e}")

# ============================================================================
# Section 3: Topological Data Analysis (TDA)
# ============================================================================

st.header("3️⃣ Topological Data Analysis (TDA)")

with st.expander("ℹ️ What is TDA / Persistent Homology?", expanded=False):
    st.markdown("""
    **Topological Data Analysis** captures the **shape** and **structure** of data using algebraic topology.
    
    **How it works:**
    1. Treat each data sample as a point cloud in high-dimensional space
    2. Compute **persistent homology** (H₀, H₁, H₂) to find:
       - H₀: Connected components (clusters)
       - H₁: Loops/cycles
       - H₂: Voids/cavities
    3. **Persistence diagrams** show which topological features persist across scales
    4. Convert diagrams to **feature vectors** (persistence statistics, entropy, amplitudes)
    
    **When to use:**
    - Data with spatial/relational structure (patient trajectories, networks, time series)
    - When you suspect manifold structure in your data
    - Publications: TDA is underutilized in tabular ML — good for novelty!
    
    **Example:** Health data where disease progression forms a trajectory (early → mild → severe)
    
    **Computational cost:** O(n³) for n samples. Subsample for large datasets.
    
    **References:**
    - Topological Machine Learning (giotto-tda)
    - Persistent Homology for Time Series (TDA + SVM pipelines)
    """)

use_tda = st.checkbox(
    "☐ Compute TDA Features (Persistent Homology)",
    value=False,
    help="Extract topological features from point cloud structure. Computationally intensive for >1000 samples."
)

if use_tda:
    st.warning("⚠️ **TDA Computation:** Can be slow for large datasets (O(n³) complexity). Consider subsampling.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        homology_dims = st.multiselect(
            "Homology Dimensions",
            [0, 1, 2],
            default=[0, 1],
            help="H₀=connected components, H₁=loops, H₂=voids. More dimensions = more features but slower."
        )
    
    with col2:
        max_edge_length = st.slider(
            "Max Edge Length",
            1.0, 10.0, 5.0, 0.5,
            help="Maximum distance for Vietoris-Rips complex. Larger = more connections, slower computation."
        )
    
    with col3:
        normalize_tda = st.checkbox(
            "Normalize features first",
            value=True,
            help="Recommended: Scale features to [0,1] before computing distances"
        )
    
    # Subsampling option
    n_samples = len(X_engineered)
    if n_samples > 500:
        st.info(f"📊 Dataset has {n_samples} samples. Consider subsampling for faster computation.")
        subsample = st.checkbox(
            "Subsample for TDA computation",
            value=True if n_samples > 1000 else False,
            help="Compute TDA on a random subset, then use same features for all samples"
        )
        
        if subsample:
            subsample_size = st.slider(
                "Subsample size",
                min_value=100,
                max_value=min(n_samples, 2000),
                value=min(500, n_samples),
                step=100,
                help="Number of samples to use for TDA computation"
            )
    else:
        subsample = False
        subsample_size = n_samples
    
    if not homology_dims:
        st.error("❌ Please select at least one homology dimension")
    elif st.button("🔬 Compute TDA Features", type="primary"):
        with st.spinner("Computing persistent homology... This may take a few minutes."):
            try:
                # Import TDA libraries
                try:
                    from gtda.homology import VietorisRipsPersistence
                    from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
                except ImportError:
                    st.error("❌ giotto-tda not installed. Run: `pip install giotto-tda`")
                    st.stop()
                
                # Get numeric data
                X_numeric = X_engineered[numeric_features].values
                
                # Normalize if requested
                if normalize_tda:
                    scaler = StandardScaler()
                    X_numeric = scaler.fit_transform(X_numeric)
                
                # Subsample if requested
                if subsample:
                    rng = np.random.RandomState(42)
                    indices = rng.choice(len(X_numeric), size=subsample_size, replace=False)
                    X_tda = X_numeric[indices]
                    st.info(f"Using {subsample_size} samples for TDA computation")
                else:
                    X_tda = X_numeric
                
                # Reshape for giotto-tda: (n_samples, n_features, 1)
                # For tabular data, each sample is treated as a point cloud of 1 point in n_features-dimensional space
                # BUT: We want to treat each sample as a collection of feature values
                # Actually, for point cloud TDA, we should treat the entire dataset as ONE point cloud
                
                # Correct approach: Each sample is a point in feature space
                # We compute TDA on the ENTIRE dataset as a single point cloud
                X_point_cloud = X_tda.reshape(1, subsample_size, -1)  # (1, n_samples, n_features)
                
                # Compute persistence diagrams
                vr = VietorisRipsPersistence(
                    homology_dimensions=homology_dims,
                    max_edge_length=max_edge_length,
                    n_jobs=-1
                )
                
                progress_bar = st.progress(0)
                progress_bar.progress(30, "Computing Vietoris-Rips complex...")
                
                diagrams = vr.fit_transform(X_point_cloud)
                
                progress_bar.progress(60, "Extracting features from persistence diagrams...")
                
                # Extract features from diagrams
                tda_features_list = []
                tda_feature_names = []
                
                # Persistence Entropy
                entropy = PersistenceEntropy()
                entropy_features = entropy.fit_transform(diagrams).flatten()
                for i, dim in enumerate(homology_dims):
                    tda_features_list.append(entropy_features[i])
                    tda_feature_names.append(f"TDA_H{dim}_entropy")
                
                # Amplitude (bottleneck, wasserstein, etc.)
                for metric in ['bottleneck', 'wasserstein', 'landscape']:
                    try:
                        amp = Amplitude(metric=metric)
                        amp_features = amp.fit_transform(diagrams).flatten()
                        for i, dim in enumerate(homology_dims):
                            tda_features_list.append(amp_features[i])
                            tda_feature_names.append(f"TDA_H{dim}_{metric}_amplitude")
                    except:
                        pass  # Some metrics may not work for all diagrams
                
                # Number of points
                n_points = NumberOfPoints()
                n_points_features = n_points.fit_transform(diagrams).flatten()
                for i, dim in enumerate(homology_dims):
                    tda_features_list.append(n_points_features[i])
                    tda_feature_names.append(f"TDA_H{dim}_n_points")
                
                progress_bar.progress(90, "Adding TDA features to dataset...")
                
                # Create TDA feature array
                tda_features_array = np.array(tda_features_list).reshape(1, -1)
                
                # Replicate for all samples (global topological features)
                tda_df = pd.DataFrame(
                    np.tile(tda_features_array, (n_samples, 1)),
                    columns=tda_feature_names,
                    index=X_engineered.index
                )
                
                # Add to engineered dataset
                X_engineered = pd.concat([X_engineered, tda_df], axis=1)
                engineered_features.extend(tda_feature_names)
                
                progress_bar.progress(100, "Complete!")
                
                st.success(f"✅ Created **{len(tda_feature_names)} TDA features** from persistent homology")
                st.session_state["tda_features_created"] = True
                
                # Show what was created
                with st.expander("View TDA features"):
                    st.dataframe(tda_df.describe())
                
            except Exception as e:
                st.error(f"❌ Error computing TDA features: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# Section 4: Dimensionality Reduction as Features
# ============================================================================

st.header("4️⃣ Dimensionality Reduction as Features")

with st.expander("ℹ️ What is this?", expanded=False):
    st.markdown("""
    Instead of using PCA/UMAP in the preprocessing pipeline (which replaces original features),
    create PCA/UMAP components as **additional features** alongside the originals.
    
    **Use case:**
    - Keep interpretability of original features
    - Add low-dimensional embeddings as supplementary information
    - Works well with tree models (RandomForest, XGBoost)
    
    **Example:** Original 100 features + 10 PCA components = 110 features total
    """)

use_dimred = st.checkbox(
    "☐ Add Dimensionality Reduction Features",
    value=False,
    help="Create PCA or UMAP components as new columns"
)

if use_dimred:
    dimred_method = st.selectbox(
        "Method",
        ["PCA", "UMAP"],
        help="PCA = linear, UMAP = non-linear manifold learning"
    )
    
    if dimred_method == "PCA":
        n_components_pca = st.slider(
            "Number of components",
            2, min(20, len(numeric_features)),
            min(10, len(numeric_features)),
            help="PCA components to add as features"
        )
        
        if st.button("🔬 Compute PCA Features", type="primary"):
            with st.spinner("Computing PCA..."):
                try:
                    X_numeric = X_engineered[numeric_features]
                    pca = PCA(n_components=n_components_pca)
                    X_pca = pca.fit_transform(X_numeric)
                    
                    pca_cols = [f"PCA_{i+1}" for i in range(n_components_pca)]
                    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X_engineered.index)
                    
                    X_engineered = pd.concat([X_engineered, pca_df], axis=1)
                    engineered_features.extend(pca_cols)
                    
                    variance_explained = pca.explained_variance_ratio_.sum()
                    st.success(f"✅ Created **{n_components_pca} PCA features** (explaining {variance_explained:.1%} of variance)")
                    st.session_state["pca_features_created"] = True
                    
                except Exception as e:
                    st.error(f"❌ Error computing PCA: {e}")
    
    else:  # UMAP
        n_components_umap = st.slider(
            "Number of components",
            2, 10, 3,
            help="UMAP dimensions (typically 2-3 for visualization, up to 10 for features)"
        )
        
        n_neighbors = st.slider(
            "Number of neighbors",
            5, 100, 15,
            help="Controls local vs global structure (lower = local, higher = global)"
        )
        
        if st.button("🔬 Compute UMAP Features", type="primary"):
            with st.spinner("Computing UMAP embedding... (can take a few minutes)"):
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
                    
                    st.success(f"✅ Created **{n_components_umap} UMAP features**")
                    st.session_state["umap_features_created"] = True
                    
                except ImportError:
                    st.error("❌ umap-learn not installed. Run: `pip install umap-learn`")
                except Exception as e:
                    st.error(f"❌ Error computing UMAP: {e}")

# ============================================================================
# Summary & Save
# ============================================================================

st.header("📊 Summary")

original_features = len(X.columns)
current_features = len(X_engineered.columns)
new_features = current_features - original_features

col1, col2, col3 = st.columns(3)
col1.metric("Original Features", original_features)
col2.metric("New Features Created", new_features, delta=f"+{new_features}")
col3.metric("Total Features", current_features)

if new_features > 0:
    st.success(f"✅ Created **{new_features} new features** via feature engineering")
    
    with st.expander("View new feature names"):
        st.write(engineered_features)
    
    # Save to session state
    if st.button("💾 Save Engineered Features", type="primary"):
        # Combine with target
        df_engineered = pd.concat([X_engineered, y], axis=1)
        
        # Save to session state
        st.session_state["df_engineered"] = df_engineered
        st.session_state["feature_engineering_applied"] = True
        st.session_state["engineered_feature_names"] = engineered_features
        
        st.success("✅ Saved engineered dataset! Proceed to **Feature Selection** to filter the most important features.")
        st.balloons()
else:
    st.info("No feature engineering applied yet. Enable techniques above to create new features.")

# Navigation hint
st.markdown("---")
st.markdown("**Next step:** → **Feature Selection** to identify the most predictive features from the engineered set")
