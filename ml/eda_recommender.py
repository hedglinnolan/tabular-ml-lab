"""
EDA Recommendation System for Medical/Nutritional Tabular Data.
Generates contextual recommendations based on dataset signals.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_reference_interval
from ml.outliers import detect_outliers
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


@dataclass
class DatasetSignals:
    """Computed signals from the dataset for EDA recommendations."""
    n_rows: int
    n_cols: int
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    text_like_cols: List[str] = field(default_factory=list)
    missing_rate_by_col: Dict[str, float] = field(default_factory=dict)
    high_missing_cols: List[str] = field(default_factory=list)
    high_cardinality_categoricals: List[str] = field(default_factory=list)
    duplicate_row_rate: float = 0.0
    target_name: Optional[str] = None
    task_type_final: Optional[Literal["regression", "classification"]] = None
    cohort_type_final: Optional[Literal["cross_sectional", "longitudinal"]] = None
    entity_id_final: Optional[str] = None
    target_stats: Dict = field(default_factory=dict)
    leakage_flags: List[str] = field(default_factory=list)
    leakage_candidate_cols: List[str] = field(default_factory=list)
    collinearity_summary: Dict = field(default_factory=dict)
    physio_plausibility_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class EDARecommendation:
    """A single EDA recommendation card."""
    id: str
    title: str
    priority: int  # Lower = higher priority
    cost: Literal["low", "medium", "high"]
    why: List[str]  # Concrete triggered reasons
    what_you_learn: List[str]
    model_implications: List[str]
    run_action: str  # Name of analysis function to call
    description: Optional[str] = None  # Plain-language explanation
    enabled: bool = True
    disabled_reason: Optional[str] = None


def compute_dataset_signals(
    df: pd.DataFrame,
    target: Optional[str],
    task_type_final: Optional[str],
    cohort_type_final: Optional[str],
    entity_id_final: Optional[str],
    sample_size: int = 5000,
    outlier_method: str = "iqr",
    feature_cols: Optional[List[str]] = None,
) -> DatasetSignals:
    """
    Compute dataset signals for EDA recommendations.
    
    Args:
        df: DataFrame
        target: Target column name
        task_type_final: Final task type (regression/classification)
        cohort_type_final: Final cohort type (cross_sectional/longitudinal)
        entity_id_final: Final entity ID column name
        sample_size: Sample size for expensive computations
        
    Returns:
        DatasetSignals object
    """
    signals = DatasetSignals(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_name=target,
        task_type_final=task_type_final,
        cohort_type_final=cohort_type_final,
        entity_id_final=entity_id_final
    )
    
    # Sample for expensive computations
    df_sample = df.sample(min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    
    # Column type classification
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith('int') or dtype.startswith('float'):
            signals.numeric_cols.append(col)
        elif dtype == 'object' or dtype == 'category' or dtype == 'bool':
            if df[col].dtype == 'bool':
                signals.categorical_cols.append(col)
            elif df[col].dtype.name == 'category':
                signals.categorical_cols.append(col)
            else:
                # Check if it's text-like (high cardinality, mostly unique)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8 and df[col].dtype == 'object':
                    signals.text_like_cols.append(col)
                else:
                    signals.categorical_cols.append(col)
        elif 'datetime' in dtype:
            signals.datetime_cols.append(col)
    
    # Missingness
    missing_counts = df.isnull().sum()
    signals.missing_rate_by_col = (missing_counts / len(df)).to_dict()
    signals.high_missing_cols = [
        col for col, rate in signals.missing_rate_by_col.items()
        if rate > 0.05
    ]
    
    # Duplicate rows
    signals.duplicate_row_rate = df.duplicated().sum() / len(df)
    
    # High cardinality categoricals
    for col in signals.categorical_cols:
        if df[col].nunique() > len(df) * 0.5:
            signals.high_cardinality_categoricals.append(col)
    
    # Target statistics
    if target and target in df.columns:
        target_series = df[target].dropna()
        if len(target_series) > 0:
            signals.target_stats['n_unique'] = target_series.nunique()
            signals.target_stats['n_missing'] = df[target].isnull().sum()
            signals.target_stats['missing_rate'] = signals.target_stats['n_missing'] / len(df)
            
            if task_type_final == 'regression' and target_series.dtype in [np.int64, np.int32, np.float64, np.float32]:
                signals.target_stats['mean'] = target_series.mean()
                signals.target_stats['median'] = target_series.median()
                signals.target_stats['std'] = target_series.std()
                signals.target_stats['skew'] = stats.skew(target_series)
                signals.target_stats['kurtosis'] = stats.kurtosis(target_series)
                
                outlier_mask, _ = detect_outliers(target_series, method=outlier_method)
                signals.target_stats['outlier_rate'] = float(outlier_mask.sum() / len(target_series)) if len(target_series) > 0 else 0.0
            elif task_type_final == 'classification':
                value_counts = target_series.value_counts()
                signals.target_stats['class_counts'] = value_counts.to_dict()
                signals.target_stats['n_classes'] = len(value_counts)
                if len(value_counts) > 0:
                    max_class = value_counts.max()
                    min_class = value_counts.min()
                    signals.target_stats['class_imbalance_ratio'] = min_class / max_class if max_class > 0 else 0.0
                    signals.target_stats['majority_class_prop'] = max_class / len(target_series)
    
    # Leakage detection (simple heuristics)
    if target:
        # Check for perfect or near-perfect correlations
        # Only use columns that are truly numeric (can be converted to float)
        numeric_for_corr = []
        for col in signals.numeric_cols:
            if col != target:
                try:
                    # Verify column is actually numeric
                    pd.to_numeric(df[col], errors='raise')
                    numeric_for_corr.append(col)
                except (ValueError, TypeError):
                    pass  # Skip columns that can't be converted
        
        if len(numeric_for_corr) > 0 and target in df.columns:
            try:
                # Also verify target can be used in correlation
                target_numeric = pd.to_numeric(df[target], errors='coerce')
                if target_numeric.notna().sum() > 0:
                    corr_df = df[numeric_for_corr].copy()
                    corr_df['_target'] = target_numeric
                    corr_with_target = corr_df.corr()['_target'].abs()
                    high_corr = corr_with_target[corr_with_target > 0.95].drop('_target', errors='ignore')
                    if len(high_corr) > 0:
                        signals.leakage_flags.append(f"{len(high_corr)} columns with >0.95 correlation to target")
                        signals.leakage_candidate_cols = high_corr.index.tolist()
            except Exception:
                pass  # Skip leakage detection if correlation fails
    
    # Collinearity (sample if too many columns)
    # Filter to truly numeric columns within user's selected features
    _corr_candidates = signals.numeric_cols
    if feature_cols:
        _feature_set = set(feature_cols)
        _corr_candidates = [c for c in _corr_candidates if c in _feature_set]
    numeric_cols_for_corr = []
    for col in _corr_candidates:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols_for_corr.append(col)
        except (ValueError, TypeError):
            pass
    
    if len(numeric_cols_for_corr) > 50:
        # Sample by variance
        try:
            variances = df[numeric_cols_for_corr].var().sort_values(ascending=False)
            numeric_cols_for_corr = variances.head(50).index.tolist()
        except Exception:
            numeric_cols_for_corr = numeric_cols_for_corr[:50]
    
    if len(numeric_cols_for_corr) > 1:
        try:
            corr_matrix = df[numeric_cols_for_corr].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)  # Remove diagonal
            max_corr = corr_matrix.max().max()
            signals.collinearity_summary['max_corr'] = max_corr
            signals.collinearity_summary['high_corr_pairs'] = []
            if max_corr > 0.85:
                # Find high correlation pairs
                for i, col1 in enumerate(numeric_cols_for_corr):
                    for col2 in numeric_cols_for_corr[i+1:]:
                        try:
                            corr_val = abs(df[col1].corr(df[col2]))
                            if corr_val > 0.85:
                                signals.collinearity_summary['high_corr_pairs'].append((col1, col2, corr_val))
                        except Exception:
                            pass
        except Exception:
            pass  # Skip collinearity analysis if it fails
    
    # Empirical plausibility flags (NHANES percentile reference)
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    for col in signals.numeric_cols:
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
                signals.physio_plausibility_flags.append(
                    f"{col}: {out_rate:.1%} outside NHANES reference ({ref_low}-{ref_high} {ref_unit})"
                )
    
    return signals


def recommend_eda(signals: DatasetSignals) -> List[EDARecommendation]:
    """
    Generate EDA recommendations based on dataset signals.
    
    Args:
        signals: DatasetSignals object
        
    Returns:
        List of EDARecommendation objects, sorted by priority
    """
    recommendations = []
    
    # R1: Physiologic plausibility / range check (always)
    recommendations.append(EDARecommendation(
        id="r1_plausibility",
        title="Physiologic Plausibility Check",
        priority=1,
        cost="low",
        why=[
            f"Dataset has {signals.n_rows} rows with {len(signals.numeric_cols)} numeric columns",
            "Medical/nutritional data should be checked for plausible ranges"
        ],
        what_you_learn=[
            "Out-of-range values that may indicate data entry errors",
            "Potential unit mismatches (e.g., mmol/L vs mg/dL)",
            "Columns requiring transformation or winsorization"
        ],
        model_implications=[
            "Outliers may require robust loss (Huber) or winsorization",
            "Unit mismatches can cause model instability"
        ],
        run_action="plausibility_check"
    ))
    
    # R2: Missingness mechanism scan
    if len(signals.high_missing_cols) > 0:
        recommendations.append(EDARecommendation(
            id="r2_missingness",
            title="Missingness Pattern Analysis",
            priority=2,
            cost="low",
            why=[
                f"{len(signals.high_missing_cols)} columns with >5% missing values",
                f"Max missing rate: {max(signals.missing_rate_by_col.values()):.1%}"
            ],
            what_you_learn=[
                "Which columns have missing data and at what rate",
                "Whether missingness is associated with target (informative missingness)",
                "Patterns suggesting MCAR (Missing Completely At Random) vs MAR (Missing At Random) vs MNAR (Missing Not At Random)"
            ],
            model_implications=[
                "Informative missingness may require missingness indicators",
                "High missing rates may need specialized imputation strategies"
            ],
            run_action="missingness_scan",
            description=(
                "**What this is:** Analyzes patterns in missing data across columns and checks if missingness "
                "is associated with the target variable.\n\n"
                "**Why it matters:** If missingness is informative (associated with target), it contains signal "
                "that models can use. MCAR means missing is random; MAR means missing depends on observed data; "
                "MNAR means missing depends on unobserved values.\n\n"
                "**How to interpret:** If target mean differs between missing/non-missing groups, missingness is "
                "informative and you may want to add missingness indicator features."
            )
        ))
    
    # R3: Cohort structure + split warning
    if signals.cohort_type_final == "longitudinal":
        recommendations.append(EDARecommendation(
            id="r3_cohort_structure",
            title="Longitudinal Data Split Guidance",
            priority=2,
            cost="low",
            why=[
                f"Cohort type detected: {signals.cohort_type_final}",
                f"Entity ID: {signals.entity_id_final or 'Not specified'}"
            ],
            what_you_learn=[
                "Distribution of rows per entity",
                "Risk of data leakage with random splits",
                "Recommended split strategy"
            ],
            model_implications=[
                "Must use group-based splitting to prevent leakage",
                "Consider time-based splits if temporal ordering exists"
            ],
            run_action="cohort_split_guidance"
        ))
    
    # R4: Leakage risk scan
    if len(signals.leakage_flags) > 0 or len(signals.leakage_candidate_cols) > 0:
        recommendations.append(EDARecommendation(
            id="r4_leakage",
            title="Target Leakage Risk Assessment",
            priority=3,
            cost="low",
            why=signals.leakage_flags + [
                f"{len(signals.leakage_candidate_cols)} suspicious columns identified"
            ],
            what_you_learn=[
                "Columns with suspiciously high correlation to target",
                "Potential ID-like columns that should be excluded",
                "Features that may contain target information"
            ],
            model_implications=[
                "Leakage columns must be excluded from features",
                "High correlation may indicate data quality issues"
            ],
            run_action="leakage_scan",
            description=(
                "**What this is:** Identifies columns that have suspiciously high correlation (>0.95) with the target, "
                "which may indicate data leakage.\n\n"
                "**Why it matters:** Data leakage occurs when features contain information that would not be available "
                "at prediction time, leading to unrealistically high performance that won't generalize.\n\n"
                "**How to interpret:** Columns flagged should be excluded from features unless you can verify they "
                "are legitimate predictors available at prediction time."
            )
        ))
    
    # R5: Target distribution
    if signals.target_name:
        if signals.task_type_final == "regression":
            outlier_rate = signals.target_stats.get('outlier_rate', 0)
            skew = signals.target_stats.get('skew', 0)
            recommendations.append(EDARecommendation(
                id="r5_target_regression",
                title="Target Distribution & Outliers",
                priority=3,
                cost="low",
                why=[
                    f"Target: {signals.target_name}",
                    f"Skewness: {skew:.2f}",
                    f"Outlier rate: {outlier_rate:.1%}"
                ],
                what_you_learn=[
                    "Target distribution shape (normal, skewed, multimodal)",
                    "Outlier locations and impact",
                    "Need for log transformation or robust loss"
                ],
                model_implications=[
                    "High skew → consider log transform or robust loss (Huber)",
                    "High outlier rate → use Huber loss or winsorization",
                    "Multimodal → may benefit from tree-based models"
                ],
                run_action="target_profile"
            ))
        elif signals.task_type_final == "classification":
            imbalance_ratio = signals.target_stats.get('class_imbalance_ratio', 1.0)
            n_classes = signals.target_stats.get('n_classes', 0)
            recommendations.append(EDARecommendation(
                id="r5_target_classification",
                title="Class Balance & Baseline",
                priority=3,
                cost="low",
                why=[
                    f"Target: {signals.target_name}",
                    f"Classes: {n_classes}",
                    f"Imbalance ratio: {imbalance_ratio:.2f}"
                ],
                what_you_learn=[
                    "Class distribution and balance",
                    "Baseline accuracy (majority class)",
                    "Need for class weighting or resampling"
                ],
                model_implications=[
                    "Imbalanced classes → use class_weight or F1/PR-AUC metrics",
                    "Low baseline → model must significantly outperform random",
                    "Binary vs multiclass affects model choice"
                ],
                run_action="target_profile"
            ))
    
    # R6: Dose-response trends
    if signals.target_name and len(signals.numeric_cols) > 1 and signals.n_rows > 100:
        recommendations.append(EDARecommendation(
            id="r6_dose_response",
            title="Dose-Response Trends",
            priority=4,
            cost="medium",
            why=[
                f"{len(signals.numeric_cols)} numeric features available",
                f"Dataset size: {signals.n_rows} rows"
            ],
            what_you_learn=[
                "Top features by association with target",
                "Nonlinear relationships (monotonic, U-shaped, etc.)",
                "Feature ranges where target behavior changes"
            ],
            model_implications=[
                "Nonlinear trends → prefer RF or NN over GLM",
                "Monotonic trends → GLM with splines may suffice",
                "U-shaped → tree models or polynomial features"
            ],
            run_action="dose_response_trends"
        ))
    
    # R7: Interaction radar (if age/sex/BMI present)
    interaction_cols = []
    for col in signals.numeric_cols + signals.categorical_cols:
        col_lower = col.lower()
        if any(term in col_lower for term in ['age', 'sex', 'gender', 'bmi']):
            interaction_cols.append(col)
    
    if len(interaction_cols) > 0 and signals.target_name:
        recommendations.append(EDARecommendation(
            id="r7_interactions",
            title="Stratified Trends by Demographics",
            priority=5,
            cost="medium",
            why=[
                f"Found demographic columns: {', '.join(interaction_cols)}",
                "Medical data often shows age/sex/BMI interactions"
            ],
            what_you_learn=[
                "How feature-target relationships vary by demographics",
                "Potential interaction terms for GLM",
                "Subgroup-specific patterns"
            ],
            model_implications=[
                "Significant interactions → include interaction terms in GLM",
                "Tree models (RF) automatically capture interactions",
                "NN can learn complex interactions if data is sufficient"
            ],
            run_action="interaction_analysis"
        ))
    
    # R8: Collinearity map
    max_corr = signals.collinearity_summary.get('max_corr', 0)
    if max_corr > 0.85 and len(signals.numeric_cols) > 5:
        recommendations.append(EDARecommendation(
            id="r8_collinearity",
            title="Collinearity Heatmap",
            priority=4,
            cost="low",
            why=[
                f"Maximum correlation: {max_corr:.2f}",
                f"{len(signals.collinearity_summary.get('high_corr_pairs', []))} highly correlated pairs"
            ],
            what_you_learn=[
                "Feature clusters with high correlation",
                "Redundant features that can be removed",
                "Multicollinearity risks for GLM (Generalized Linear Model)"
            ],
            model_implications=[
                "High collinearity → GLM coefficients unstable, use regularization",
                "RF (Random Forest) and NN (Neural Network) are more robust to collinearity",
                "Consider PCA (Principal Component Analysis) or feature selection"
            ],
            run_action="collinearity_map",
            description=(
                "**What this is:** Shows correlation heatmap between numeric features. High correlations (>0.85) "
                "indicate collinearity (features are highly related).\n\n"
                "**Why it matters:** Collinearity makes GLM coefficients unstable and hard to interpret. "
                "Tree-based models (RF) and neural networks are more robust.\n\n"
                "**How to interpret:** Clusters of highly correlated features may be redundant. Consider removing "
                "one from each cluster or using dimensionality reduction."
            )
        ))
    
    # R9: Outlier influence (regression)
    if signals.task_type_final == "regression":
        outlier_rate = signals.target_stats.get('outlier_rate', 0)
        if outlier_rate > 0.05:
            recommendations.append(EDARecommendation(
                id="r9_outlier_influence",
                title="Outlier Influence Analysis",
                priority=5,
                cost="medium",
                why=[
                    f"Outlier rate: {outlier_rate:.1%}",
                    "Outliers can heavily influence regression models"
                ],
                what_you_learn=[
                    "Location and magnitude of outliers",
                    "Impact on model predictions",
                    "Effect of robust loss or winsorization"
                ],
                model_implications=[
                    "High outlier rate → use Huber loss or robust regression",
                    "Consider winsorization or outlier removal",
                    "NN with robust loss may outperform GLM"
                ],
                run_action="outlier_influence"
            ))
    
    # R10: Quick probe baselines (always)
    if signals.target_name:
        recommendations.append(EDARecommendation(
            id="r10_baselines",
            title="Quick Baseline Models",
            priority=6,
            cost="low",
            why=[
                "Establish performance floor before complex modeling",
                "Fast sanity check on data quality"
            ],
            what_you_learn=[
                "Baseline performance (constant predictor, simple GLM, shallow RF)",
                "Whether data has predictive signal",
                "Expected performance range"
            ],
            model_implications=[
                "If baselines perform well → data is easy, simple models may suffice",
                "If baselines fail → need complex models or feature engineering",
                "Baseline gap shows potential improvement ceiling"
            ],
            run_action="quick_probe_baselines"
        ))
    
    # Sort by priority
    recommendations.sort(key=lambda x: x.priority)
    
    return recommendations
