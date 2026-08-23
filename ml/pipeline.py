"""
Preprocessing pipeline builder.
Creates sklearn Pipeline with ColumnTransformer for mixed data types.
"""
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# IterativeImputer (MICE) is experimental in scikit-learn
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    _HAS_ITERATIVE = True
except ImportError:
    _HAS_ITERATIVE = False

# TargetEncoder available in scikit-learn >= 1.3
try:
    from sklearn.preprocessing import TargetEncoder
    _HAS_TARGET_ENCODER = True
except ImportError:
    _HAS_TARGET_ENCODER = False
from ml.feature_steps import create_pca_step, KMeansFeatures
from ml.preprocess_operators import UnitHarmonizer, PlausibilityGate, OutlierCapping, plausibility_row_mask
from ml.clinical_units import infer_unit, CLINICAL_VARIABLES
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_improbability_band


# ─────────────────────────────────────────────────────────────────────────────
# What the numeric imputation defaults cost  (`AUDIT-007`)
# ─────────────────────────────────────────────────────────────────────────────
#
# `AUDIT-007`, corrected in the shape `AUDIT-021` uses.
#
# BEFORE: the `median` branch below was applied on the default path with
#   nothing anywhere stating what it costs, and `pages/05_Preprocess.py`'s help
#   text for it read *"Robust to skewed distributions. Most common default."* —
#   an endorsement. `CLINICAL_SURVEY_PACK.md` §A2 anti-pattern 2 is
#   **[SETTLED as bad]**: *"Mean/median imputation. Understates variance,
#   destroys the distribution, indefensible in a manuscript."* Cross-cutting
#   item 11 is blunter for the mean: *"Mean-imputing anything, ever, without a
#   loud warning."*
#
# AFTER: the cost is stated where the default is applied, once. Nothing is
#   removed — median remains the default, and the sentence keeps the true part
#   of the old one (it IS the robust point estimate under skew) while adding
#   what that buys and what it costs.
#
# Kept here rather than in the page because this module is where the choice
# becomes an estimator; `pages/05_Preprocess.py` renders these strings, so the
# sentence and the code that makes it true cannot drift apart. §A2 anti-pattern
# 3 (**[SETTLED]**) is folded in for every strategy that fills from `X` alone:
# the outcome is not in any of these fills, so associations are biased toward
# the null. `iterative` (MICE) is the one entry that is not an anti-pattern,
# and it says what it still assumes rather than claiming to be free.
NUMERIC_IMPUTATION_COST = {
    "median": (
        "Robust to skew as a point estimate — and that is all it is. Filling "
        "with one number understates that column's variance and distorts its "
        "distribution, and the outcome is not in the fill, so associations are "
        "biased toward the null. §A2 settles this as bad practice in a "
        "manuscript."
    ),
    "mean": (
        "⚠️ Mean imputation assumes symmetry and is pulled by outliers, and "
        "like the median it understates variance, distorts the distribution "
        "and biases associations toward the null because the outcome is not in "
        "the fill. §A2 settles this as bad practice in a manuscript."
    ),
    "constant": (
        "Fills with a fixed value, so it is a claim that the blank MEANS that "
        "value. Defensible only where missing is structural or has a domain "
        "meaning; otherwise it distorts the distribution more than the median "
        "does, and the outcome is not in the fill."
    ),
    "iterative": (
        "Models each column from the others (MICE), so it preserves variance "
        "in a way single-value fills cannot. It still assumes the missingness "
        "is MAR — where a value is absent because of what it would have been, "
        "no imputation recovers it."
    ),
}

# Every scaler this builder can actually construct, read off the branches in
# `build_preprocessing_pipeline` below. It exists so a caller can ask whether a
# variant it was handed is buildable instead of naming one this module would
# silently ignore — the `else` in that branch chain is a no-op, so an unknown
# variant applies NO scaling while the interface says it applied one.
SUPPORTED_NUMERIC_SCALINGS = ("standard", "robust", "minmax", "none")


def scaling_from_recipe(
    model_key: str,
    registry: Optional[Dict[str, Any]] = None,
    fallback: str = "standard",
    origins: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """What `turbotab.recipes` says this model's scaling is, and what gets built.

    `AUDIT-013`, corrected in the shape `AUDIT-021` uses.

    BEFORE: `pages/05_Preprocess.py` read
      `spec.capabilities.requires_scaled_numeric` directly at four sites and
      hard-coded `"standard"` from it, then told the user *"scaling enabled
      (standard); appropriate for this model"* and *"Enabled standard scaling
      (model requires scaling)"*. Both sentences attribute the choice to the
      model's declared requirement, which is the `caps:requires_scaled_numeric`
      row in `turbotab/recipes.py`. A pack may override that row —
      `turbotab/packs.py:5182` registers `scale → pareto` against exactly that
      selector — and the page could not see it, so the sentence named a source
      whose answer it had not read.

    AFTER: the table is asked. `resolve()`'s precedence lattice decides, a pack
      row reaches the Classic door, and where this builder cannot construct the
      table's answer that is STATED rather than silently substituted.

    The last clause is the point. `build_preprocessing_pipeline`'s scaler branch
    has no `else`: a variant it does not know applies NO scaling at all. So
    routing the table straight through would have turned a display defect into
    a fit defect the first time a metabolomics project asked for `pareto`. The
    honest form is the fallback plus `departure` — the app refusing, out loud.

    Returns a dict. `consulted` is False and every table field is `None`/`""`
    when the table cannot answer (an unregistered key such as the synthetic
    `"default"` pipeline) — ignorance reported as ignorance rather than as a
    default (`AGENT_ONBOARD.md` §07 trap 9). Callers keep their own
    data-determined choice in that case, exactly as before.

    Invariant, asserted by
    `tests/test_the_preprocess_page_asks_the_recipe_table.py`: `departure` is
    non-empty if and only if `applied != table_variant`.
    """
    out: Dict[str, Any] = {
        "model": model_key, "consulted": False, "table_variant": None,
        "reason": "", "origin": "", "selector": "", "required": None,
        "applied": None, "buildable": None, "departure": "",
    }
    try:
        from turbotab import recipes as _recipes
        res = _recipes.resolve(model_key, "scale", registry, origins)
    except Exception:
        return out

    out.update(consulted=True, table_variant=res.variant, reason=res.reason,
               origin=res.origin, selector=res.selector)
    out["required"] = res.variant != "none"
    if not out["required"]:
        # The table's answer is "no scaling is required for this model". WHICH
        # scaling to use anyway is a judgment about the data, and the caller
        # owns it; saying nothing here is the correct amount to say.
        return out

    out["buildable"] = res.variant in SUPPORTED_NUMERIC_SCALINGS
    if out["buildable"]:
        out["applied"] = res.variant
    else:
        out["applied"] = fallback
        out["departure"] = (
            f"The recipe table asks for **{res.variant}** scaling here "
            f"(contributed by `{res.origin}`, matching `{res.selector}`), and "
            f"this pipeline builder cannot construct it — it builds "
            f"{', '.join(SUPPORTED_NUMERIC_SCALINGS)}. **{fallback}** is "
            f"applied instead, which is a departure from the table, not the "
            f"table's answer."
        )
    return out


def build_unit_harmonization_config(
    df: pd.DataFrame,
    numeric_features: List[str],
    unit_overrides: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Compute conversion factors to canonical units for numeric features."""
    unit_overrides = unit_overrides or {}
    conversion_factors = []
    inferred_units = {}
    canonical_units = {}

    for col in numeric_features:
        col_lower = col.lower()
        override_unit = unit_overrides.get(col)
        matched_var = next((var for var in CLINICAL_VARIABLES.keys() if var in col_lower), None)

        if matched_var and override_unit:
            var_config = CLINICAL_VARIABLES[matched_var]
            hypothesis = next((h for h in var_config["hypotheses"] if h[0] == override_unit), None)
            if hypothesis:
                unit_name, conv_factor, _ = hypothesis
                conversion_factors.append(conv_factor)
                inferred_units[col] = unit_name
                canonical_units[col] = var_config.get("canonical_unit")
                continue

        inferred = infer_unit(col, df[col].dropna())
        conv = inferred.get("conversion_factor", 1.0)
        conversion_factors.append(conv if conv is not None else 1.0)
        inferred_units[col] = inferred.get("inferred_unit")
        canonical_units[col] = inferred.get("canonical_unit")

    return {
        "conversion_factors": conversion_factors,
        "inferred_units": inferred_units,
        "canonical_units": canonical_units
    }


def build_plausibility_bounds(
    numeric_features: List[str],
    conversion_factors: List[float]
) -> Dict[str, Any]:
    """Build NHANES-based plausibility bounds aligned to numeric_features."""
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    lower_bounds = []
    upper_bounds = []
    bounds_by_feature = {}

    for col, factor in zip(numeric_features, conversion_factors):
        var_key = match_variable_key(col, nhanes_ref)
        improbability = get_improbability_band(nhanes_ref, var_key) if var_key else None
        if improbability:
            improbable_low, improbable_high, improbable_unit = improbability
            lower_bounds.append(improbable_low)
            upper_bounds.append(improbable_high)
            bounds_by_feature[col] = {
                "lower": improbable_low,
                "upper": improbable_high,
                "unit": improbable_unit
            }
        else:
            lower_bounds.append(None)
            upper_bounds.append(None)
            bounds_by_feature[col] = None

    return {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "bounds_by_feature": bounds_by_feature,
        "reference_version": nhanes_ref.get("version")
    }


def apply_plausibility_filter(
    df: pd.DataFrame,
    numeric_features: List[str],
    plausibility_bounds: Dict[str, Any],
    unit_conversion_factors: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Keep the rows whose plausibility-gated numeric columns are in NHANES range.

    The returned frame keeps the row labels it came in with. Those labels are
    identities here, not decoration: the test-set lockbox seals a set of labels
    at upload, and a cohort run banks the labels of the people in the group.
    Renumbering the survivors 0..n-1 would leave both of those sets pointing at
    whoever now happens to sit at those positions — a "Female" run quietly
    holding half men, with nothing on screen to say so.
    """
    lb = plausibility_bounds.get("lower_bounds", [])
    ub = plausibility_bounds.get("upper_bounds", [])
    if not lb and not ub:
        return df
    X = df[numeric_features].values.astype(float)
    if unit_conversion_factors:
        X = X * np.array(unit_conversion_factors, dtype=float)
    mask = plausibility_row_mask(X, lb, ub)
    return df.loc[mask]


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_imputation: str = 'median',  # 'mean', 'median', 'constant'
    numeric_scaling: str = 'standard',  # 'standard', 'robust', 'minmax', 'none'
    numeric_log_transform: bool = False,
    numeric_power_transform: str = 'none',  # 'none', 'log1p', 'yeo-johnson'
    numeric_missing_indicators: bool = False,
    numeric_outlier_treatment: str = 'none',  # 'none', 'percentile', 'mad'
    numeric_outlier_params: Optional[Dict[str, Any]] = None,
    unit_harmonization_factors: Optional[List[float]] = None,
    plausibility_bounds: Optional[Dict[str, Any]] = None,
    plausibility_mode: str = 'clip',  # 'clip' = set out-of-range to NaN; 'filter' = drop rows
    categorical_imputation: str = 'most_frequent',  # 'most_frequent', 'constant'
    categorical_encoding: str = 'onehot',  # 'onehot', 'target' (if enabled)
    categorical_target_type: str = 'auto',  # TargetEncoder: 'auto', 'continuous', 'binary', 'multiclass'
    handle_unknown: str = 'ignore',  # For one-hot encoding
    # Optional feature engineering steps
    use_kmeans_features: bool = False,
    kmeans_n_clusters: int = 5,
    kmeans_add_distances: bool = True,
    kmeans_add_onehot: bool = False,
    use_pca: bool = False,
    pca_n_components: Optional[Union[int, float]] = None,
    pca_whiten: bool = False,
    random_state: int = 42,
    # Passthrough features (already transformed by Feature Engineering)
    passthrough_numeric_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Build preprocessing pipeline using ColumnTransformer.
    
    Args:
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        numeric_imputation: Strategy for imputing numeric missing values
        numeric_scaling: Scaling strategy for numeric features
        numeric_log_transform: Whether to apply log transform to numeric features
        categorical_imputation: Strategy for imputing categorical missing values
        categorical_encoding: Encoding strategy for categorical features
        handle_unknown: How to handle unknown categories in one-hot encoding
        
    Returns:
        sklearn Pipeline with ColumnTransformer
    """
    transformers = []
    
    # Numeric preprocessing
    if numeric_features:
        numeric_steps = []

        # Unit harmonization (convert to canonical units)
        if unit_harmonization_factors:
            numeric_steps.append(('unit_harmonize', UnitHarmonizer(unit_harmonization_factors)))

        # Plausibility gating (clip: set out-of-range to NaN; filter: rows dropped before pipeline)
        if plausibility_bounds and plausibility_mode != 'filter':
            numeric_steps.append((
                'plausibility_gate',
                PlausibilityGate(
                    plausibility_bounds.get("lower_bounds", []),
                    plausibility_bounds.get("upper_bounds", [])
                )
            ))

        # Imputation
        if numeric_imputation == 'iterative' and _HAS_ITERATIVE:
            numeric_steps.append(('imputer', IterativeImputer(
                max_iter=10, random_state=random_state,
                add_indicator=numeric_missing_indicators,
            )))
        elif numeric_imputation == 'mean':
            numeric_steps.append(('imputer', SimpleImputer(strategy='mean', add_indicator=numeric_missing_indicators)))
        elif numeric_imputation == 'median':
            numeric_steps.append(('imputer', SimpleImputer(strategy='median', add_indicator=numeric_missing_indicators)))
        elif numeric_imputation == 'constant':
            numeric_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=numeric_missing_indicators)))
        elif numeric_imputation == 'iterative' and not _HAS_ITERATIVE:
            # Fallback if IterativeImputer not available
            numeric_steps.append(('imputer', SimpleImputer(strategy='median', add_indicator=numeric_missing_indicators)))
        
        # Power transform (optional)
        if numeric_power_transform == 'yeo-johnson':
            numeric_steps.append(('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)))
        elif numeric_log_transform or numeric_power_transform == 'log1p':
            def log_transform(X):
                return np.log1p(np.maximum(X, 0))  # log1p handles zeros
            numeric_steps.append(('log', FunctionTransformer(log_transform)))
        if numeric_power_transform in ('yeo-johnson', 'log1p') or numeric_log_transform:
            numeric_steps.append(('power_nan_guard', SimpleImputer(strategy='median')))

        # Outlier treatment
        if numeric_outlier_treatment and numeric_outlier_treatment != 'none':
            numeric_steps.append((
                'outlier',
                OutlierCapping(method=numeric_outlier_treatment, params=numeric_outlier_params or {})
            ))
        
        # Scaling
        if numeric_scaling == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif numeric_scaling == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        elif numeric_scaling == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        # 'none' means no scaling
        
        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(('numeric', numeric_pipeline, numeric_features))
    
    # Categorical preprocessing
    if categorical_features:
        categorical_steps = []
        
        # Imputation
        if categorical_imputation == 'most_frequent':
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        elif categorical_imputation == 'constant':
            categorical_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        
        # Encoding
        if categorical_encoding == 'target' and _HAS_TARGET_ENCODER:
            # target_type must be told, not inferred: type_of_target() calls an
            # INTEGER regression target (age in years, a count, a score)
            # 'multiclass', and TargetEncoder then emits one column per distinct
            # value per feature -- 121 columns where 3 were meant.
            categorical_steps.append(('encoder', TargetEncoder(
                smooth='auto',
                target_type=categorical_target_type,
            )))
        elif categorical_encoding == 'ordinal':
            categorical_steps.append(('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )))
        else:
            # Default: one-hot
            categorical_steps.append(('encoder', OneHotEncoder(
                sparse_output=True,
                handle_unknown=handle_unknown,
                drop='if_binary',
            )))
        
        categorical_pipeline = Pipeline(categorical_steps)
        transformers.append(('categorical', categorical_pipeline, categorical_features))
    
    # Passthrough numeric features: already transformed by Feature Engineering
    # They get imputation + scaling only (no log/power/outlier transforms)
    if passthrough_numeric_features:
        pt_steps = []
        # Imputation (same strategy as main numeric)
        if numeric_imputation == 'mean':
            pt_steps.append(('imputer', SimpleImputer(strategy='mean')))
        elif numeric_imputation == 'constant':
            pt_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        else:
            pt_steps.append(('imputer', SimpleImputer(strategy='median')))
        # Scaling (same as main numeric)
        if numeric_scaling == 'standard':
            pt_steps.append(('scaler', StandardScaler()))
        elif numeric_scaling == 'robust':
            pt_steps.append(('scaler', RobustScaler()))
        elif numeric_scaling == 'minmax':
            pt_steps.append(('scaler', MinMaxScaler()))
        pt_pipeline = Pipeline(pt_steps)
        transformers.append(('numeric_passthrough', pt_pipeline, passthrough_numeric_features))

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not specified
        verbose_feature_names_out=False
    )
    
    # Build pipeline steps
    steps = [('preprocessor', preprocessor)]
    
    # Optional: KMeansFeatures (must come before PCA if both enabled)
    if use_kmeans_features:
        kmeans_transformer = KMeansFeatures(
            n_clusters=kmeans_n_clusters,
            add_distances=kmeans_add_distances,
            add_onehot_label=kmeans_add_onehot,
            random_state=random_state
        )
        steps.append(('kmeans_features', kmeans_transformer))
    
    # Optional: PCA (dimensionality reduction)
    # Note: PCA n_components validation will happen at fit time
    # We can't know exact feature count until after ColumnTransformer + KMeans
    # So we'll validate in the create_pca_step or handle gracefully
    if use_pca:
        pca_transformer = create_pca_step(
            enabled=True,
            n_components=pca_n_components,
            whiten=pca_whiten,
            random_state=random_state
        )
        if pca_transformer:
            steps.append(('pca', pca_transformer))
    
    return Pipeline(steps)


def get_pipeline_recipe(pipeline: Pipeline, plausibility_mode: Optional[str] = None) -> str:
    """
    Get human-readable description of pipeline steps.
    
    Args:
        pipeline: sklearn Pipeline
        plausibility_mode: If 'filter', note that rows were filtered to NHANES range before pipeline.
        
    Returns:
        String description of pipeline
    """
    steps = []
    if plausibility_mode == "filter":
        steps.append("Plausibility: rows filtered to NHANES range (before pipeline).")
    if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
        for name, transformer, columns in pipeline.named_steps['preprocessor'].transformers_:
            if name == 'numeric':
                # Get numeric pipeline steps
                numeric_pipe = transformer
                step_desc = f"Numeric features ({len(columns)}): "
                step_parts = []
                for step_name, step_transformer in numeric_pipe.steps:
                    if step_name == 'unit_harmonize':
                        step_parts.append("Unit harmonization")
                    elif step_name == 'plausibility_gate':
                        step_parts.append("Plausibility gate (NHANES)")
                    elif step_name == 'imputer':
                        strategy = step_transformer.strategy
                        if getattr(step_transformer, 'add_indicator', False):
                            step_parts.append(f"Impute ({strategy}) + missing flags")
                        else:
                            step_parts.append(f"Impute ({strategy})")
                    elif step_name == 'log':
                        step_parts.append("Log transform")
                    elif step_name == 'outlier':
                        # Extract actual params from the fitted transformer
                        outlier_desc = "Outlier capping"
                        if hasattr(step_transformer, 'method'):
                            method = step_transformer.method
                            if method == 'percentile' and hasattr(step_transformer, 'lower_percentile'):
                                lo = getattr(step_transformer, 'lower_percentile', '?')
                                hi = getattr(step_transformer, 'upper_percentile', '?')
                                outlier_desc = f"Percentile clip ({lo}th–{hi}th)"
                            elif method == 'iqr':
                                mult = getattr(step_transformer, 'iqr_multiplier', getattr(step_transformer, 'multiplier', 1.5))
                                outlier_desc = f"IQR capping (×{mult})"
                            elif method == 'zscore':
                                thresh = getattr(step_transformer, 'threshold', 3)
                                outlier_desc = f"Z-score filter (|z| > {thresh})"
                            elif method == 'mad':
                                thresh = getattr(step_transformer, 'mad_threshold', getattr(step_transformer, 'n_mad', 3))
                                outlier_desc = f"MAD capping ({thresh}× MAD)"
                            else:
                                outlier_desc = f"Outlier treatment ({method})"
                        step_parts.append(outlier_desc)
                    elif step_name == 'scaler':
                        if isinstance(step_transformer, StandardScaler):
                            step_parts.append("Standard scaling")
                        elif isinstance(step_transformer, RobustScaler):
                            step_parts.append("Robust scaling")
                step_desc += " → ".join(step_parts) if step_parts else "No transformation"
                steps.append(step_desc)
            
            elif name == 'categorical':
                # Get categorical pipeline steps
                categorical_pipe = transformer
                step_desc = f"Categorical features ({len(columns)}): "
                step_parts = []
                for step_name, step_transformer in categorical_pipe.steps:
                    if step_name == 'imputer':
                        strategy = step_transformer.strategy
                        step_parts.append(f"Impute ({strategy})")
                    elif step_name == 'encoder':
                        if isinstance(step_transformer, OneHotEncoder):
                            step_parts.append("One-hot encoding (sparse)")
                        elif isinstance(step_transformer, OrdinalEncoder):
                            step_parts.append("Ordinal encoding")
                        elif _HAS_TARGET_ENCODER and isinstance(step_transformer, TargetEncoder):
                            step_parts.append(
                                "Target encoding (cross-fitted, smoothed means)"
                            )
                step_desc += " → ".join(step_parts) if step_parts else "No transformation"
                steps.append(step_desc)
    
    # Add optional feature engineering steps
    if "kmeans_features" in pipeline.named_steps:
        kmeans = pipeline.named_steps["kmeans_features"]
        kmeans_desc = f"KMeans features added: {kmeans.n_clusters} clusters"
        if kmeans.add_distances:
            kmeans_desc += ", distances to centroids"
        if kmeans.add_onehot_label:
            kmeans_desc += ", one-hot cluster labels"
        steps.append(kmeans_desc)

    if "pca" in pipeline.named_steps:
        pca = pipeline.named_steps["pca"]
        n_comp = pca.n_components_
        if isinstance(n_comp, (int, np.integer)):
            steps.append(f"PCA applied: {n_comp} components (output PC1, PC2, ...)")
        else:
            steps.append(f"PCA applied: {n_comp} components (variance threshold)")
        if pca.whiten:
            steps[-1] += ", whitened"
    
    return "\n".join(steps) if steps else "No preprocessing steps"


def get_feature_names_after_transform(pipeline: Pipeline, original_feature_names: List[str]) -> List[str]:
    """
    Get feature names after pipeline transformation.
    Handles preprocessor, optional KMeansFeatures, and optional PCA.
    KMeans replaces preprocessor output with kmeans_dist_cluster_* / kmeans_cluster_*.
    PCA replaces preceding features with PC1, PC2, ...
    """
    feature_names: List[str] = []

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = [str(x) for x in preprocessor.get_feature_names_out()]
        else:
            feature_names = _preprocessor_names_fallback(preprocessor, original_feature_names)
    except Exception:
        feature_names = _preprocessor_names_fallback(
            pipeline.named_steps.get("preprocessor"),
            original_feature_names,
        )

    if "kmeans_features" in pipeline.named_steps:
        kmeans = pipeline.named_steps["kmeans_features"]
        if hasattr(kmeans, "get_feature_names_out"):
            feature_names = [str(x) for x in kmeans.get_feature_names_out()]
        else:
            feature_names = []
            if getattr(kmeans, "add_distances", True):
                for i in range(kmeans.n_clusters):
                    feature_names.append(f"kmeans_dist_{i}")
            if getattr(kmeans, "add_onehot_label", False):
                for i in range(kmeans.n_clusters):
                    feature_names.append(f"kmeans_cluster_{i}")

    if "pca" in pipeline.named_steps:
        pca = pipeline.named_steps["pca"]
        n = pca.components_.shape[0]
        feature_names = [f"PC{i + 1}" for i in range(n)]

    if not feature_names:
        try:
            dummy_df = pd.DataFrame(
                np.zeros((1, len(original_feature_names))),
                columns=original_feature_names[: len(original_feature_names)],
            )
            t = pipeline.transform(dummy_df)
            t = t.toarray() if hasattr(t, "toarray") else np.asarray(t)
            n_out = t.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_out)]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(64)]

    return feature_names


def _preprocessor_names_fallback(preprocessor, original_feature_names: List[str]) -> List[str]:
    """Fallback when get_feature_names_out is not available."""
    out: List[str] = []
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return out
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            out.extend(columns)
        elif name == "categorical":
            cat_pipe = transformer
            for _sn, st in cat_pipe.steps:
                if _sn == "encoder" and isinstance(st, OneHotEncoder) and hasattr(st, "categories_"):
                    for col_idx, col_name in enumerate(columns):
                        for c in st.categories_[col_idx]:
                            out.append(f"{col_name}_{c}")
                    break
                elif _sn == "encoder":
                    out.extend([f"{c}_encoded" for c in columns])
                    break
    return out


def reconcile_pipeline_columns(pipe, available_columns):
    """Trim a preprocessing pipeline's column selectors to columns that exist.

    A stored preprocessing pipeline names its input columns explicitly (the
    ColumnTransformer keeps ``(name, transformer, [col, ...])`` tuples). If the
    feature set later changes — e.g. Feature Selection drops an engineered
    ``*_has_data`` indicator after the pipeline was built — fitting that stale
    pipeline raises ``ValueError: Some column names are not columns of the
    dataframe``. This returns an *unfitted* copy whose selectors reference only
    columns present in ``available_columns``, plus the list of dropped names, so
    a caller can rebuild-and-warn instead of crashing.

    Returns ``(new_pipe, dropped)``. ``new_pipe is pipe`` (unchanged) when
    nothing needed dropping. Transformers left with no columns are removed.
    """
    from sklearn.base import clone

    available = set(available_columns)
    dropped: List[str] = []

    def _fix_ct(ct: ColumnTransformer) -> ColumnTransformer:
        new_transformers = []
        for name, trans, cols in ct.transformers:
            if isinstance(cols, (list, tuple)):
                kept = [c for c in cols if c in available]
                dropped.extend([c for c in cols if c not in available])
                if kept:
                    new_transformers.append((name, clone(trans), kept))
                # a transformer with no surviving columns is dropped entirely
            else:
                new_transformers.append((name, clone(trans), cols))
        return ColumnTransformer(
            transformers=new_transformers,
            remainder=ct.remainder,
            sparse_threshold=ct.sparse_threshold,
            n_jobs=ct.n_jobs,
            transformer_weights=ct.transformer_weights,
            verbose_feature_names_out=getattr(ct, "verbose_feature_names_out", True),
        )

    if isinstance(pipe, ColumnTransformer):
        fixed = _fix_ct(pipe)
        return (pipe, []) if not dropped else (fixed, dropped)

    if isinstance(pipe, Pipeline):
        new_steps = []
        for sname, step in pipe.steps:
            if isinstance(step, ColumnTransformer):
                new_steps.append((sname, _fix_ct(step)))
            else:
                new_steps.append((sname, clone(step)))
        return (pipe, []) if not dropped else (Pipeline(new_steps), dropped)

    return pipe, []
