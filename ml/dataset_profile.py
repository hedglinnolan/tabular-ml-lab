"""
Dataset Profile: Computes comprehensive dataset diagnostics for intelligent model coaching.

This module provides the foundation for all coach recommendations by analyzing:
- Dataset dimensions (n rows, p features)
- Feature types (numeric vs categorical)
- Missingness patterns
- Target characteristics (type, balance)
- Cardinality for categoricals
- Outlier detection for numerics
- Data sufficiency indicators
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pandas.api import types as _pdt
from enum import Enum
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_reference_interval
from ml.outliers import detect_outliers


class DataSufficiencyLevel(Enum):
    """Indicates how sufficient the data is for various model types."""
    ABUNDANT = "abundant"      # Plenty of data for any model
    ADEQUATE = "adequate"      # Sufficient for most models
    LIMITED = "limited"        # May constrain complex models
    SCARCE = "scarce"          # Strong regularization needed
    CRITICAL = "critical"      # Only simplest models viable


class WarningLevel(Enum):
    """Warning severity levels."""
    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DataWarning:
    """A warning or caution flag about the dataset."""
    category: str  # e.g., "sample_size", "imbalance", "missingness", "dimensionality"
    level: WarningLevel
    short_message: str  # Brief message for badges/tags
    detailed_message: str  # Full explanation for expandable sections
    affected_models: List[str] = field(default_factory=list)  # Model families affected
    suggested_actions: List[str] = field(default_factory=list)  # What to do about it


@dataclass
class FeatureProfile:
    """Profile for a single feature."""
    name: str
    dtype: str
    is_numeric: bool
    is_categorical: bool
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_ratio: float  # unique_count / n
    
    # Numeric-specific
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    has_outliers: bool = False
    outlier_count: int = 0
    outlier_rate: float = 0.0
    
    # Categorical-specific
    is_high_cardinality: bool = False
    top_categories: Optional[List[Tuple[str, int]]] = None
    
    # Potential issues
    is_constant: bool = False
    is_id_like: bool = False


@dataclass
class TargetProfile:
    """Profile for the target variable."""
    name: str
    task_type: str  # "regression" or "classification"
    n_unique: int
    
    # Regression-specific
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    has_outliers: bool = False
    outlier_rate: float = 0.0
    
    # Classification-specific
    n_classes: Optional[int] = None
    class_counts: Optional[Dict[Any, int]] = None
    class_balance_ratio: Optional[float] = None  # max/min class ratio
    minority_class_size: Optional[int] = None
    is_imbalanced: bool = False
    imbalance_severity: Optional[str] = None  # "mild", "moderate", "severe"


@dataclass
class DatasetProfile:
    """
    Comprehensive dataset profile for intelligent model coaching.
    
    This is the central data structure that drives all recommendations.
    """
    # Basic dimensions
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    
    # Feature-to-sample ratios
    p_n_ratio: float  # p/n
    
    # Missingness summary
    total_missing_rate: float
    n_features_with_missing: int
    n_features_high_missing: int  # >10% missing
    n_features_critical_missing: int  # >50% missing
    
    # Optional fields with defaults
    events_per_variable: Optional[float] = None  # For classification: minority class / p
    
    # Target info
    target_profile: Optional[TargetProfile] = None
    
    # Feature details
    feature_profiles: Dict[str, FeatureProfile] = field(default_factory=dict)
    
    # Cardinality issues
    high_cardinality_features: List[str] = field(default_factory=list)
    constant_features: List[str] = field(default_factory=list)
    id_like_features: List[str] = field(default_factory=list)
    
    # Numeric issues
    features_with_outliers: List[str] = field(default_factory=list)
    highly_skewed_features: List[str] = field(default_factory=list)
    physio_plausibility_flags: List[str] = field(default_factory=list)
    physio_reference_version: Optional[str] = None
    
    # Data sufficiency assessment
    data_sufficiency: DataSufficiencyLevel = DataSufficiencyLevel.ADEQUATE
    sufficiency_narrative: str = ""
    
    # Warnings and flags
    warnings: List[DataWarning] = field(default_factory=list)
    
    # Recommended model families (preliminary)
    recommended_families: List[str] = field(default_factory=list)
    cautioned_families: List[str] = field(default_factory=list)
    discouraged_families: List[str] = field(default_factory=list)
    
    # Metadata
    profile_timestamp: Optional[str] = None


def compute_feature_profile(df: pd.DataFrame, col: str, n: int, outlier_method: str = "iqr") -> FeatureProfile:
    """Compute profile for a single feature."""
    series = df[col]
    dtype = str(series.dtype)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_categorical = not is_numeric or series.nunique() <= 10
    
    missing_count = series.isna().sum()
    missing_rate = missing_count / n if n > 0 else 0.0
    unique_count = series.nunique()
    cardinality_ratio = unique_count / n if n > 0 else 0.0
    
    profile = FeatureProfile(
        name=col,
        dtype=dtype,
        is_numeric=is_numeric,
        is_categorical=is_categorical and not is_numeric,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        is_constant=(unique_count <= 1),
        # Integer dtype asked by predicate: a nullable Int64 column is still
        # integer, and `dtype in ['int64','int32']` says it is not.
        is_id_like=(unique_count == n and is_numeric
                    and _pdt.is_integer_dtype(series) and not _pdt.is_bool_dtype(series))
    )
    
    if is_numeric:
        valid = series.dropna()
        if len(valid) > 0:
            profile.mean = float(valid.mean())
            profile.std = float(valid.std())
            profile.min_val = float(valid.min())
            profile.max_val = float(valid.max())
            profile.median = float(valid.median())
            
            # Skewness
            if len(valid) > 2:
                try:
                    profile.skewness = float(valid.skew())
                except:
                    profile.skewness = 0.0
            
            # Outlier detection (skip boolean columns - quantile fails on bool)
            if valid.dtype != bool and not (hasattr(valid.dtype, 'kind') and valid.dtype.kind == 'b'):
                outlier_mask, _ = detect_outliers(valid, method=outlier_method)
                profile.outlier_count = int(outlier_mask.sum())
            else:
                profile.outlier_count = 0
            profile.outlier_rate = profile.outlier_count / len(valid) if len(valid) > 0 else 0.0
            profile.has_outliers = profile.outlier_rate > 0.01  # >1% outliers
    
    if profile.is_categorical or (not is_numeric):
        # High cardinality check
        profile.is_high_cardinality = cardinality_ratio > 0.5 or unique_count > 50
        
        # Top categories
        top_cats = series.value_counts().head(5)
        profile.top_categories = list(zip(top_cats.index.astype(str), top_cats.values.tolist()))
    
    return profile


def compute_target_profile(df: pd.DataFrame, target_col: str, task_type: str, outlier_method: str = "iqr") -> TargetProfile:
    """Compute profile for the target variable."""
    series = df[target_col]
    n_unique = series.nunique()
    
    profile = TargetProfile(
        name=target_col,
        task_type=task_type,
        n_unique=n_unique
    )
    
    valid = series.dropna()
    
    if task_type == 'regression':
        if len(valid) > 0:
            profile.mean = float(valid.mean())
            profile.std = float(valid.std())
            profile.min_val = float(valid.min())
            profile.max_val = float(valid.max())
            profile.median = float(valid.median())
            
            if len(valid) > 2:
                try:
                    profile.skewness = float(valid.skew())
                except:
                    profile.skewness = 0.0
            
            # Outlier detection (configurable method)
            outlier_mask, _ = detect_outliers(valid, method=outlier_method)
            profile.outlier_rate = float(outlier_mask.sum() / len(valid)) if len(valid) > 0 else 0.0
            profile.has_outliers = profile.outlier_rate > 0.05
    
    else:  # classification
        profile.n_classes = n_unique
        profile.class_counts = valid.value_counts().to_dict()
        
        if profile.class_counts:
            max_count = max(profile.class_counts.values())
            min_count = min(profile.class_counts.values())
            profile.class_balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            profile.minority_class_size = min_count
            
            # Imbalance assessment
            if profile.class_balance_ratio > 10:
                profile.is_imbalanced = True
                profile.imbalance_severity = "severe"
            elif profile.class_balance_ratio > 5:
                profile.is_imbalanced = True
                profile.imbalance_severity = "moderate"
            elif profile.class_balance_ratio > 2:
                profile.is_imbalanced = True
                profile.imbalance_severity = "mild"
            else:
                profile.is_imbalanced = False
    
    return profile


def assess_data_sufficiency(
    n: int, 
    p: int, 
    task_type: str,
    minority_class_size: Optional[int] = None
) -> Tuple[DataSufficiencyLevel, str]:
    """
    Assess data sufficiency for modeling.
    
    Uses practical heuristics (not formal power calculations):
    - Events per variable (EPV) for classification
    - Observations per parameter for regression
    - Feature-to-sample ratio
    """
    p_n_ratio = p / n if n > 0 else float('inf')
    
    narratives = []
    level = DataSufficiencyLevel.ADEQUATE
    
    # Basic sample size check
    if n < 50:
        level = DataSufficiencyLevel.CRITICAL
        narratives.append(f"Very small sample (n={n:,}). Only the simplest models are viable.")
    elif n < 100:
        level = DataSufficiencyLevel.SCARCE
        narratives.append(f"Small sample (n={n:,}). Strong regularization recommended.")
    elif n < 500:
        level = DataSufficiencyLevel.LIMITED
        narratives.append(f"Modest sample size (n={n:,}). Complex models may overfit.")
    elif n < 5000:
        level = DataSufficiencyLevel.ADEQUATE
        narratives.append(f"Adequate sample size (n={n:,}) for most model types.")
    else:
        level = DataSufficiencyLevel.ABUNDANT
        narratives.append(f"Large sample (n={n:,}). All model types are viable.")
    
    # Feature-to-sample ratio check
    if p_n_ratio > 1.0:
        level = min(level, DataSufficiencyLevel.CRITICAL, key=lambda x: list(DataSufficiencyLevel).index(x))
        narratives.append(f"More features than samples (p/n={p_n_ratio:.2f}). Regularization essential; consider dimensionality reduction.")
    elif p_n_ratio > 0.5:
        level = min(level, DataSufficiencyLevel.SCARCE, key=lambda x: list(DataSufficiencyLevel).index(x))
        narratives.append(f"High dimensionality (p/n={p_n_ratio:.2f}). Regularized models preferred.")
    elif p_n_ratio > 0.1:
        narratives.append(f"Moderate dimensionality (p/n={p_n_ratio:.2f}).")
    else:
        narratives.append(f"Low dimensionality (p/n={p_n_ratio:.2f}). Feature space is manageable.")
    
    # Events per variable (classification)
    if task_type == 'classification' and minority_class_size is not None:
        epv = minority_class_size / p if p > 0 else float('inf')
        if epv < 5:
            level = min(level, DataSufficiencyLevel.CRITICAL, key=lambda x: list(DataSufficiencyLevel).index(x))
            narratives.append(f"Very low events per variable ({epv:.1f}). High overfitting risk for any model.")
        elif epv < 10:
            level = min(level, DataSufficiencyLevel.SCARCE, key=lambda x: list(DataSufficiencyLevel).index(x))
            narratives.append(f"Low events per variable ({epv:.1f}). Simple models with regularization preferred.")
        elif epv < 20:
            narratives.append(f"Moderate events per variable ({epv:.1f}). Most models viable with care.")
        else:
            narratives.append(f"Good events per variable ({epv:.1f}). Classification models have adequate signal.")
    
    # Observations per parameter heuristics for neural nets
    if n < p * 20:
        narratives.append("Neural networks may struggle: typically need 20+ samples per input feature.")
    
    narrative = " ".join(narratives)
    return level, narrative


def generate_warnings(profile: DatasetProfile) -> List[DataWarning]:
    """Generate warnings based on the dataset profile."""
    warnings = []
    
    # Sample size warning
    if profile.n_rows < 100:
        warnings.append(DataWarning(
            category="sample_size",
            level=WarningLevel.CRITICAL,
            short_message="Very small sample",
            detailed_message=f"With only {profile.n_rows:,} samples, model training is highly constrained. "
                           "Use strong regularization, cross-validation, and simple models. "
                           "Results may not generalize well.",
            affected_models=["Neural Networks", "Complex Ensembles"],
            suggested_actions=[
                "Consider collecting more data",
                "Use regularized linear models (Ridge, Lasso)",
                "Increase cross-validation folds",
                "Be cautious interpreting results"
            ]
        ))
    elif profile.n_rows < 500:
        warnings.append(DataWarning(
            category="sample_size",
            level=WarningLevel.WARNING,
            short_message="Small sample",
            detailed_message=f"With {profile.n_rows:,} samples, some models may overfit. "
                           "Regularization and validation are important.",
            affected_models=["Neural Networks", "Deep Trees"],
            suggested_actions=[
                "Use regularization",
                "Prefer simpler models",
                "Use cross-validation"
            ]
        ))
    
    # High dimensionality warning
    if profile.p_n_ratio > 0.5:
        warnings.append(DataWarning(
            category="dimensionality",
            level=WarningLevel.WARNING if profile.p_n_ratio <= 1.0 else WarningLevel.CRITICAL,
            short_message="High dimensionality",
            detailed_message=f"Feature-to-sample ratio is {profile.p_n_ratio:.2f} "
                           f"({profile.n_features} features, {profile.n_rows:,} samples). "
                           "This can cause overfitting and unstable estimates.",
            affected_models=["Unregularized Linear Models", "k-NN"],
            suggested_actions=[
                "Use regularized models (Ridge, Lasso, ElasticNet)",
                "Consider dimensionality reduction (PCA)",
                "Remove low-variance or redundant features"
            ]
        ))
    
    # Class imbalance warning
    if profile.target_profile and profile.target_profile.is_imbalanced:
        severity = profile.target_profile.imbalance_severity
        ratio = profile.target_profile.class_balance_ratio
        level = WarningLevel.CRITICAL if severity == "severe" else (
            WarningLevel.WARNING if severity == "moderate" else WarningLevel.CAUTION
        )
        warnings.append(DataWarning(
            category="imbalance",
            level=level,
            short_message=f"{severity.title()} imbalance",
            detailed_message=f"Class imbalance ratio is {ratio:.1f}:1. "
                           f"The minority class has only {profile.target_profile.minority_class_size:,} samples. "
                           "Accuracy can be misleading; use F1, PR-AUC, or balanced accuracy instead.",
            affected_models=["All classification models"],
            # `GUIDED-049`. This used to open with "Use class weights in
            # training" and "Consider SMOTE or other resampling". Van den
            # Goorbergh et al. (JAMIA 2022;29:1525) and Carriero et al. (Stat
            # Med 2025): rebalancing overestimates minority-class probability
            # without improving discrimination, and the apparent sensitivity
            # gain is reproducible by shifting the threshold. The remedy for a
            # rare outcome is penalization and sample size, because the problem
            # is small-sample overfitting rather than imbalance.
            #
            # The advice is not deleted, it is ROUTED — see
            # `ml/imbalance_advice.py`, which the Guided door reads with the
            # recorded purpose. What is removed here is the RECOMMENDATION,
            # because this list is read as one.
            suggested_actions=[
                "Focus on precision-recall metrics, not accuracy",
                "Report calibration alongside discrimination",
                "Adjust classification threshold based on costs",
                "Penalize the fit — the problem behind a rare outcome is "
                "small-sample overfitting, not imbalance",
            ]
        ))
    
    # Missing data warning
    if profile.n_features_high_missing > 0:
        level = WarningLevel.CRITICAL if profile.n_features_critical_missing > 0 else WarningLevel.WARNING
        warnings.append(DataWarning(
            category="missingness",
            level=level,
            short_message=f"{profile.n_features_high_missing} features with high missingness",
            detailed_message=f"{profile.n_features_high_missing} features have >10% missing values"
                           + (f", and {profile.n_features_critical_missing} have >50% missing" 
                              if profile.n_features_critical_missing > 0 else "") + 
                           ". Missing data can bias results if not handled properly.",
            affected_models=["Linear Models (need imputation)", "Neural Networks (need imputation)"],
            suggested_actions=[
                "Investigate if missingness is random (MAR) or informative (MNAR)",
                "Consider adding missingness indicators",
                "Use imputation (mean/median for simple, KNN/iterative for better)",
                "Tree models can handle missing values natively"
            ]
        ))
    
    # High cardinality warning
    if len(profile.high_cardinality_features) > 0:
        warnings.append(DataWarning(
            category="cardinality",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.high_cardinality_features)} high-cardinality features",
            detailed_message=f"Features {', '.join(profile.high_cardinality_features[:3])}"
                           + (f" and {len(profile.high_cardinality_features)-3} more" 
                              if len(profile.high_cardinality_features) > 3 else "") +
                           " have many unique values. One-hot encoding will create many columns.",
            affected_models=["Linear Models (with one-hot)", "Neural Networks"],
            suggested_actions=[
                "Consider target encoding",
                "Consider frequency encoding",
                "Group rare categories",
                "Tree models handle high cardinality better"
            ]
        ))
    
    # Outliers warning
    if len(profile.features_with_outliers) > 0:
        warnings.append(DataWarning(
            category="outliers",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.features_with_outliers)} features with outliers",
            detailed_message=f"{len(profile.features_with_outliers)} numeric features have significant outliers. "
                           "This can affect model performance, especially for distance-based and linear models.",
            affected_models=["Linear Regression (OLS)", "k-NN", "Neural Networks"],
            suggested_actions=[
                "Investigate if outliers are errors or genuine",
                "Consider robust models (Huber loss)",
                "Consider winsorizing or capping",
                "Tree models are robust to outliers"
            ]
        ))

    # Physiologic plausibility warning (NHANES reference)
    if profile.physio_plausibility_flags:
        warnings.append(DataWarning(
            category="physiologic_plausibility",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.physio_plausibility_flags)} physiologic flags",
            detailed_message=(
                "Empirical plausibility checks found values outside NHANES reference intervals. "
                "These checks are based on population distributions, not clinical guidance."
            ),
            affected_models=["All Models"],
            suggested_actions=[
                "Verify units and data entry",
                "Review plausible ranges for affected features",
                "Consider unit harmonization or plausibility gating"
            ]
        ))
    
    return warnings


def compute_dataset_profile(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    outlier_method: str = "iqr"
) -> DatasetProfile:
    """
    Compute a comprehensive dataset profile.
    
    Args:
        df: The dataframe to profile
        target_col: Name of target column (optional)
        feature_cols: List of feature columns (optional, defaults to all non-target)
        task_type: 'regression' or 'classification' (optional, will infer if not provided)
    
    Returns:
        DatasetProfile with all diagnostics
    """
    from datetime import datetime
    
    n = len(df)
    if n == 0 or len(df.columns) == 0:
        raise ValueError("Cannot compute profile for empty DataFrame")
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    p = len(feature_cols)
    
    # Count numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Compute feature profiles
    feature_profiles = {}
    high_cardinality = []
    constant_features = []
    id_like_features = []
    features_with_outliers = []
    highly_skewed = []
    
    for col in feature_cols:
        fp = compute_feature_profile(df, col, n, outlier_method=outlier_method)
        feature_profiles[col] = fp
        
        if fp.is_high_cardinality:
            high_cardinality.append(col)
        if fp.is_constant:
            constant_features.append(col)
        if fp.is_id_like:
            id_like_features.append(col)
        if fp.has_outliers:
            features_with_outliers.append(col)
        if fp.skewness is not None and abs(fp.skewness) > 1.0:
            highly_skewed.append(col)
    
    # Missingness summary
    missing_counts = df[feature_cols].isna().sum()
    total_missing = missing_counts.sum()
    total_cells = n * p
    total_missing_rate = total_missing / total_cells if total_cells > 0 else 0.0
    n_features_with_missing = (missing_counts > 0).sum()
    n_features_high_missing = (missing_counts / n > 0.1).sum()
    n_features_critical_missing = (missing_counts / n > 0.5).sum()
    
    # Target profile
    target_profile = None
    minority_class_size = None
    if target_col is not None and target_col in df.columns:
        # Infer task type if not provided
        if task_type is None:
            target_unique = df[target_col].nunique()
            if target_unique <= 20 and target_unique < n * 0.05:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        target_profile = compute_target_profile(df, target_col, task_type, outlier_method=outlier_method)
        if task_type == 'classification' and target_profile.minority_class_size:
            minority_class_size = target_profile.minority_class_size
    
    # Compute data sufficiency
    p_n_ratio = p / n if n > 0 else float('inf')
    data_sufficiency, sufficiency_narrative = assess_data_sufficiency(
        n, p, task_type or 'regression', minority_class_size
    )
    
    # Events per variable
    events_per_variable = None
    if task_type == 'classification' and minority_class_size is not None and p > 0:
        events_per_variable = minority_class_size / p
    
    # Physiologic plausibility flags (NHANES reference only)
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    physio_flags = []
    for col in numeric_cols:
        var_key = match_variable_key(col, nhanes_ref)
        if not var_key:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        inferred_unit_info = infer_unit(col, col_data)
        ref_interval = get_reference_interval(nhanes_ref, var_key)
        if inferred_unit_info.get('conversion_factor') and ref_interval:
            ref_low, ref_high, ref_unit = ref_interval
            converted = col_data * inferred_unit_info['conversion_factor']
            out_rate = ((converted < ref_low) | (converted > ref_high)).sum() / len(converted)
            if out_rate > 0.05:
                physio_flags.append(
                    f"{col}: {out_rate:.1%} outside NHANES reference ({ref_low}-{ref_high} {ref_unit})"
                )

    # Create profile
    profile = DatasetProfile(
        n_rows=n,
        n_features=p,
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        p_n_ratio=p_n_ratio,
        total_missing_rate=total_missing_rate,
        n_features_with_missing=n_features_with_missing,
        n_features_high_missing=n_features_high_missing,
        n_features_critical_missing=n_features_critical_missing,
        events_per_variable=events_per_variable,
        target_profile=target_profile,
        feature_profiles=feature_profiles,
        high_cardinality_features=high_cardinality,
        constant_features=constant_features,
        id_like_features=id_like_features,
        features_with_outliers=features_with_outliers,
        highly_skewed_features=highly_skewed,
        physio_plausibility_flags=physio_flags,
        physio_reference_version=nhanes_ref.get("version"),
        data_sufficiency=data_sufficiency,
        sufficiency_narrative=sufficiency_narrative,
        profile_timestamp=datetime.now().isoformat()
    )
    
    # Generate warnings
    profile.warnings = generate_warnings(profile)
    
    # Preliminary model family recommendations
    profile.recommended_families, profile.cautioned_families, profile.discouraged_families = \
        _assess_model_families(profile)
    
    return profile


def _assess_model_families(profile: DatasetProfile) -> Tuple[List[str], List[str], List[str]]:
    """
    Preliminary assessment of which model families are suitable.
    
    Returns:
        (recommended, cautioned, discouraged) lists of model family names
    """
    recommended = []
    cautioned = []
    discouraged = []
    
    n = profile.n_rows
    p = profile.n_features
    p_n_ratio = profile.p_n_ratio
    task_type = profile.target_profile.task_type if profile.target_profile else 'regression'
    
    # Always recommend regularized linear models as baseline
    recommended.append("Linear Models")
    
    # Tree-based models
    if n >= 50:
        recommended.append("Tree-Based Models")
    else:
        cautioned.append("Tree-Based Models")
    
    # Gradient Boosting
    if n >= 100:
        recommended.append("Gradient Boosting")
    elif n >= 50:
        cautioned.append("Gradient Boosting")
    else:
        discouraged.append("Gradient Boosting")
    
    # Neural Networks
    if n >= 1000 and n >= p * 20:
        recommended.append("Neural Networks")
    elif n >= 500 and n >= p * 10:
        cautioned.append("Neural Networks")
    else:
        discouraged.append("Neural Networks")
    
    # k-NN
    if p <= 20 and n >= 100 and n <= 10000:
        recommended.append("Distance-Based (k-NN)")
    elif n > 10000:
        cautioned.append("Distance-Based (k-NN)")  # Can be slow
    else:
        discouraged.append("Distance-Based (k-NN)")  # Curse of dimensionality
    
    # SVM
    if n <= 5000 and p <= 100:
        cautioned.append("Support Vector Machines")
    else:
        discouraged.append("Support Vector Machines")
    
    return recommended, cautioned, discouraged


def get_profile_summary_text(profile: DatasetProfile) -> str:
    """Generate a human-readable summary of the dataset profile."""
    lines = []
    
    # Basic stats
    lines.append(f"**Dataset Overview:** {profile.n_rows:,} samples × {profile.n_features} features")
    lines.append(f"- {profile.n_numeric} numeric, {profile.n_categorical} categorical features")
    
    # Data sufficiency
    lines.append(f"\n**Data Sufficiency:** {profile.data_sufficiency.value.title()}")
    lines.append(f"- {profile.sufficiency_narrative}")
    
    # Target info
    if profile.target_profile:
        tp = profile.target_profile
        if tp.task_type == 'regression':
            lines.append(f"\n**Target ({tp.name}):** Continuous (regression)")
            if tp.mean is not None:
                lines.append(f"- Range: {tp.min_val:.2f} to {tp.max_val:.2f}, Mean: {tp.mean:.2f}")
        else:
            lines.append(f"\n**Target ({tp.name}):** Categorical ({tp.n_classes} classes)")
            if tp.is_imbalanced:
                lines.append(f"- Imbalanced: {tp.imbalance_severity} ({tp.class_balance_ratio:.1f}:1 ratio)")
    
    # Warnings summary
    if profile.warnings:
        lines.append(f"\n**Warnings:** {len(profile.warnings)} issue(s) detected")
        for w in profile.warnings[:3]:
            lines.append(f"- {w.level.value.upper()}: {w.short_message}")
    
    return "\n".join(lines)
