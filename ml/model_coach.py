"""
Model Selection Coach: Intelligent, educational assistant for model selection.

Provides data-aware recommendations that:
- Consider dataset size, dimensionality, and characteristics
- Explain recommendations in plain language
- Bucket models into Recommended/Worth Trying/Not Recommended
- Integrate throughout the ML workflow
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class RecommendationBucket(Enum):
    """Model recommendation buckets."""
    RECOMMENDED = "recommended"
    WORTH_TRYING = "worth_trying"
    NOT_RECOMMENDED = "not_recommended"


class TrainingTimeTier(Enum):
    """Expected training time tiers."""
    FAST = "fast"           # < 10 seconds
    MEDIUM = "medium"       # 10 seconds - 2 minutes
    SLOW = "slow"           # > 2 minutes


# Canonical group display names
GROUP_DISPLAY_NAMES = {
    'Linear': 'Linear Models',
    'Trees': 'Tree-Based Models',
    'Boosting': 'Gradient Boosting',
    'Distance': 'Distance-Based Models',
    'Margin': 'Support Vector Machines',
    'Probabilistic': 'Probabilistic Models',
    'Neural Net': 'Neural Networks'
}


@dataclass
class ModelRecommendation:
    """
    A single model recommendation with full context.
    """
    model_key: str  # Registry key (e.g., 'ridge', 'rf')
    model_name: str  # Display name (e.g., 'Ridge Regression')
    group: str  # Model family group
    bucket: RecommendationBucket
    
    # Rationale and context
    rationale: str  # Why this recommendation for THIS dataset
    dataset_fit_summary: str  # One-liner about fit to data
    
    # Detailed explanations (expandable)
    strengths: List[str]
    weaknesses: List[str]
    risks: List[str]  # Overfitting, data hunger, etc.
    
    # Practical info
    training_time: TrainingTimeTier
    interpretability: str  # "high", "medium", "low"
    
    # Prerequisites
    requires_scaling: bool
    requires_encoding: bool
    handles_missing: bool
    
    # Educational content
    plain_language_summary: str  # For users with rudimentary stats knowledge
    when_to_use: str
    when_to_avoid: str
    
    # Priority within bucket (lower = higher priority)
    priority: int = 50
    
    @property
    def display_name(self) -> str:
        """Get the display name for the model's group."""
        return GROUP_DISPLAY_NAMES.get(self.group, self.group)


@dataclass
class CoachRecommendation:
    """A recommendation from the model selection coach (family-level)."""
    group: str  # e.g., "Linear", "Boosting", "Trees", "Distance", "Neural Net"
    recommended_models: List[str]  # Model keys from registry
    why: List[str]  # Plain language reasons with numbers
    when_not_to_use: List[str]  # Short caveats
    suggested_preprocessing: List[str]  # e.g., "standardize numeric features", "consider PCA"
    priority: int  # Lower = higher priority
    readiness_checks: List[str] = field(default_factory=list)  # Prerequisites
    bucket: RecommendationBucket = RecommendationBucket.RECOMMENDED
    training_time: TrainingTimeTier = TrainingTimeTier.MEDIUM
    
    @property
    def display_name(self) -> str:
        """Get the display name for this group."""
        return GROUP_DISPLAY_NAMES.get(self.group, self.group)


@dataclass
class PreprocessingRecommendation:
    """A preprocessing step recommendation."""
    step_name: str
    step_key: str  # For programmatic use
    rationale: str
    priority: str  # "required", "recommended", "optional"
    affected_model_families: List[str]
    plain_language_explanation: str
    how_to_implement: str


@dataclass
class CoachOutput:
    """
    Complete output from the model selection coach.
    """
    # Dataset context
    dataset_summary: str
    data_sufficiency_narrative: str
    warnings_summary: List[str]
    
    # Model recommendations by bucket
    recommended_models: List[ModelRecommendation]
    worth_trying_models: List[ModelRecommendation]
    not_recommended_models: List[ModelRecommendation]
    
    # Family-level recommendations (legacy compatibility)
    family_recommendations: List[CoachRecommendation]
    
    # Preprocessing recommendations
    preprocessing_recommendations: List[PreprocessingRecommendation]
    
    # EDA recommendations
    baseline_eda: List[str]
    advanced_eda_by_family: Dict[str, List[str]]


def _get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get model information for recommendations.
    
    Returns dict mapping model_key to model metadata.
    """
    return {
        # Linear Models
        'glm': {
            'name': 'GLM (OLS/Logistic)',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': False,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,  # Events per variable
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        'ridge': {
            'name': 'Ridge Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'lasso': {
            'name': 'Lasso Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'elasticnet': {
            'name': 'ElasticNet',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'huber': {
            'name': 'Huber Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'logreg': {
            'name': 'Logistic Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        
        # Tree-based
        'rf': {
            'name': 'Random Forest',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'extratrees_reg': {
            'name': 'Extra Trees',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'extratrees_clf': {
            'name': 'Extra Trees',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Boosting
        'histgb_reg': {
            'name': 'Histogram Gradient Boosting',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'histgb_clf': {
            'name': 'Histogram Gradient Boosting',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Distance-based
        'knn_reg': {
            'name': 'k-Nearest Neighbors',
            'group': 'Distance',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'medium',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        'knn_clf': {
            'name': 'k-Nearest Neighbors',
            'group': 'Distance',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'medium',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        
        # SVMs
        'svr': {
            'name': 'Support Vector Regression',
            'group': 'Margin',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'svc': {
            'name': 'Support Vector Classification',
            'group': 'Margin',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Probabilistic
        'gaussian_nb': {
            'name': 'Gaussian Naive Bayes',
            'group': 'Probabilistic',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': False,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'lda': {
            'name': 'Linear Discriminant Analysis',
            'group': 'Probabilistic',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        
        # Neural Networks
        'nn': {
            'name': 'Neural Network',
            'group': 'Neural Net',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 500,
            'min_epv': 50,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },

        # XGBoost
        'xgb_reg': {
            'name': 'XGBoost',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'xgb_clf': {
            'name': 'XGBoost',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },

        # LightGBM
        'lgbm_reg': {
            'name': 'LightGBM',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'lgbm_clf': {
            'name': 'LightGBM',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
    }


def _create_model_recommendation(
    model_key: str,
    model_info: Dict[str, Any],
    profile: Any,  # DatasetProfile
    bucket: RecommendationBucket,
    priority: int
) -> ModelRecommendation:
    """Create a detailed model recommendation based on dataset profile."""
    from ml.dataset_profile import DatasetProfile
    
    n = profile.n_rows
    p = profile.n_features
    task_type = profile.target_profile.task_type if profile.target_profile else 'regression'
    
    # Build rationale based on dataset
    rationale_parts = []
    strengths = []
    weaknesses = []
    risks = []
    
    info = model_info
    
    # Sample size assessment
    if n >= info['min_samples'] * 3:
        rationale_parts.append(f"Sample size ({n:,}) is well above minimum needed")
        strengths.append(f"Adequate data for this model type")
    elif n >= info['min_samples']:
        rationale_parts.append(f"Sample size ({n:,}) meets minimum requirements")
    else:
        rationale_parts.append(f"Sample size ({n:,}) is below recommended minimum ({info['min_samples']})")
        risks.append("May overfit due to limited data")
    
    # High dimensionality
    if profile.p_n_ratio > 0.5:
        if info['good_for_high_dim']:
            strengths.append("Handles high dimensionality well with regularization")
        else:
            weaknesses.append(f"May struggle with high feature-to-sample ratio ({profile.p_n_ratio:.2f})")
            risks.append("Risk of unstable estimates in high dimensions")
    
    # Outliers
    if len(profile.features_with_outliers) > 0:
        if info['robust_to_outliers']:
            strengths.append("Robust to outliers in your data")
        else:
            weaknesses.append(f"Sensitive to the {len(profile.features_with_outliers)} features with outliers")
    
    # Missing data
    if profile.n_features_with_missing > 0:
        if info['handles_missing']:
            strengths.append("Can handle missing values natively")
        else:
            weaknesses.append("Requires imputation for missing values")
    
    # Interpretability
    if info['interpretability'] == 'high':
        strengths.append("Highly interpretable - can explain predictions")
    elif info['interpretability'] == 'low':
        weaknesses.append("Less interpretable - harder to explain individual predictions")
    
    # Build dataset fit summary
    fit_parts = []
    if bucket == RecommendationBucket.RECOMMENDED:
        fit_parts.append("Good fit")
    elif bucket == RecommendationBucket.WORTH_TRYING:
        fit_parts.append("Reasonable fit")
    else:
        fit_parts.append("Poor fit")
    
    fit_parts.append(f"for your {n:,} samples × {p} features")
    
    # Plain language summary
    group_name = GROUP_DISPLAY_NAMES.get(info['group'], info['group'])
    plain_summary = _get_plain_language_summary(model_key, info, profile)
    
    return ModelRecommendation(
        model_key=model_key,
        model_name=info['name'],
        group=info['group'],
        bucket=bucket,
        rationale=" ".join(rationale_parts),
        dataset_fit_summary=" ".join(fit_parts),
        strengths=strengths,
        weaknesses=weaknesses,
        risks=risks,
        training_time=info['training_time'],
        interpretability=info['interpretability'],
        requires_scaling=info['requires_scaling'],
        requires_encoding=info['requires_encoding'],
        handles_missing=info['handles_missing'],
        plain_language_summary=plain_summary,
        when_to_use=_get_when_to_use(model_key, info),
        when_to_avoid=_get_when_to_avoid(model_key, info),
        priority=priority
    )


def _get_plain_language_summary(model_key: str, info: Dict, profile: Any) -> str:
    """Generate a plain language summary for users with basic stats knowledge."""
    n = profile.n_rows
    p = profile.n_features
    
    summaries = {
        'glm': f"GLM (Generalized Linear Model) finds a straight-line relationship between your {p} features and the target. "
               f"It's simple, fast, and easy to interpret. With {n:,} samples, it should train in seconds.",
        
        'ridge': f"Ridge Regression is like GLM but adds a penalty to prevent overfitting. "
                 f"It's especially good when features are correlated. "
                 f"Your {p} features will each get a coefficient showing their importance.",
        
        'lasso': f"Lasso Regression also prevents overfitting but can set some feature coefficients to exactly zero, "
                 f"effectively selecting the most important features from your {p}.",
        
        'huber': f"Huber Regression is robust to outliers - it won't let extreme values dominate the fit. "
                 f"Good if you suspect some of your {n:,} samples have measurement errors.",
        
        'rf': f"Random Forest builds many decision trees and averages their predictions. "
              f"It can capture complex patterns and interactions among your {p} features without much tuning.",
        
        'histgb_reg': f"Gradient Boosting builds trees sequentially, each one correcting the previous. "
                      f"Often achieves the best accuracy but takes longer to train.",
        'histgb_clf': f"Gradient Boosting builds trees sequentially, each one correcting the previous. "
                      f"Often achieves the best accuracy but takes longer to train.",
        
        'nn': f"Neural Networks can learn very complex patterns from data. "
              f"With {n:,} samples and {p} features, it has enough data to train meaningfully, "
              f"but results may be harder to interpret than simpler models.",
        
        'knn_reg': f"k-Nearest Neighbors predicts based on similar samples in your training data. "
                   f"Simple but can be slow with large datasets. Works best with fewer features.",
        'knn_clf': f"k-Nearest Neighbors predicts based on similar samples in your training data. "
                   f"Simple but can be slow with large datasets. Works best with fewer features.",

        'xgb_reg': f"XGBoost is the most widely-used gradient boosting library. With {n:,} samples, "
                   f"it will train quickly and often achieves top accuracy on tabular data.",
        'xgb_clf': f"XGBoost is the most widely-used gradient boosting library. With {n:,} samples, "
                   f"it will train quickly and often achieves top accuracy on tabular data.",

        'lgbm_reg': f"LightGBM grows trees leaf-wise instead of level-wise, making it faster than "
                    f"XGBoost on large datasets while achieving similar accuracy.",
        'lgbm_clf': f"LightGBM grows trees leaf-wise instead of level-wise, making it faster than "
                    f"XGBoost on large datasets while achieving similar accuracy.",
    }

    return summaries.get(model_key, f"{info['name']} is a {info['group'].lower()} model.")


def _get_when_to_use(model_key: str, info: Dict) -> str:
    """Get guidance on when to use this model."""
    guidance = {
        'glm': "When you need interpretable coefficients and a simple baseline",
        'ridge': "When features are correlated or you have more features than samples",
        'lasso': "When you want automatic feature selection",
        'huber': "When your target has outliers you don't want to remove",
        'rf': "When you expect complex interactions and want a robust model",
        'histgb_reg': "When you want the best predictive accuracy",
        'histgb_clf': "When you want the best predictive accuracy",
        'nn': "When you have lots of data and expect highly nonlinear patterns",
        'knn_reg': "As a simple, non-parametric baseline",
        'knn_clf': "As a simple, non-parametric baseline",
        'xgb_reg': "When you want best-in-class accuracy with built-in L1/L2 regularization",
        'xgb_clf': "When you want best-in-class accuracy with built-in L1/L2 regularization",
        'lgbm_reg': "When training speed matters or your dataset is large (>50k rows)",
        'lgbm_clf': "When training speed matters or your dataset is large (>50k rows)",
    }
    return guidance.get(model_key, "When the data characteristics match this model's strengths")


def _get_when_to_avoid(model_key: str, info: Dict) -> str:
    """Get guidance on when to avoid this model."""
    guidance = {
        'glm': "When relationships are clearly nonlinear or data has many outliers",
        'ridge': "When you need exact feature selection (coefficients won't be zero)",
        'lasso': "When all features are truly important (may exclude some)",
        'huber': "When you need probability estimates (regression only)",
        'rf': "When you need fast predictions or very interpretable results",
        'histgb_reg': "When training time is critical or you need simple interpretability",
        'histgb_clf': "When training time is critical or you need simple interpretability",
        'nn': "When you have limited data or need to explain individual predictions",
        'knn_reg': "With high-dimensional data or when prediction speed matters",
        'knn_clf': "With high-dimensional data or when prediction speed matters",
        'xgb_reg': "When interpretability is required or dataset is very small (<100 samples)",
        'xgb_clf': "When interpretability is required or dataset is very small (<100 samples)",
        'lgbm_reg': "When you have very few samples or need interpretable coefficients",
        'lgbm_clf': "When you have very few samples or need interpretable coefficients",
    }
    return guidance.get(model_key, "When data is limited or interpretability is critical")


def compute_model_recommendations(profile: Any) -> CoachOutput:
    """
    Compute comprehensive model recommendations based on dataset profile.
    
    This is the main entry point for the upgraded coach.
    
    Args:
        profile: DatasetProfile object
        
    Returns:
        CoachOutput with all recommendations
    """
    from ml.dataset_profile import DatasetProfile, DataSufficiencyLevel
    
    model_info = _get_model_info()
    n = profile.n_rows
    p = profile.n_features
    task_type = profile.target_profile.task_type if profile.target_profile else 'regression'
    
    # Filter models by task type
    if task_type == 'regression':
        valid_models = ['glm', 'ridge', 'lasso', 'elasticnet', 'huber', 'rf',
                       'extratrees_reg', 'histgb_reg', 'xgb_reg', 'lgbm_reg',
                       'knn_reg', 'svr', 'nn']
    else:
        valid_models = ['glm', 'logreg', 'rf', 'extratrees_clf', 'histgb_clf',
                       'xgb_clf', 'lgbm_clf',
                       'knn_clf', 'svc', 'gaussian_nb', 'lda', 'nn']
    
    recommended = []
    worth_trying = []
    not_recommended = []
    
    for model_key in valid_models:
        if model_key not in model_info:
            continue
            
        info = model_info[model_key]
        bucket, priority = _assess_model_fit(model_key, info, profile)
        
        rec = _create_model_recommendation(model_key, info, profile, bucket, priority)
        
        if bucket == RecommendationBucket.RECOMMENDED:
            recommended.append(rec)
        elif bucket == RecommendationBucket.WORTH_TRYING:
            worth_trying.append(rec)
        else:
            not_recommended.append(rec)
    
    # Sort by priority
    recommended.sort(key=lambda x: x.priority)
    worth_trying.sort(key=lambda x: x.priority)
    not_recommended.sort(key=lambda x: x.priority)
    
    # Generate family-level recommendations (legacy compatibility)
    family_recs = _generate_family_recommendations(profile, recommended, worth_trying)
    
    # Generate preprocessing recommendations
    preprocessing_recs = _generate_preprocessing_recommendations(profile)
    
    # Generate EDA recommendations
    baseline_eda, advanced_eda = _generate_eda_recommendations(profile)
    
    # Build dataset summary
    dataset_summary = f"Your dataset has {n:,} samples and {p} features ({profile.n_numeric} numeric, {profile.n_categorical} categorical)."
    
    # Warnings summary
    warnings_summary = [w.short_message for w in profile.warnings]
    
    return CoachOutput(
        dataset_summary=dataset_summary,
        data_sufficiency_narrative=profile.sufficiency_narrative,
        warnings_summary=warnings_summary,
        recommended_models=recommended,
        worth_trying_models=worth_trying,
        not_recommended_models=not_recommended,
        family_recommendations=family_recs,
        preprocessing_recommendations=preprocessing_recs,
        baseline_eda=baseline_eda,
        advanced_eda_by_family=advanced_eda
    )


def _assess_model_fit(model_key: str, info: Dict, profile: Any) -> Tuple[RecommendationBucket, int]:
    """
    Assess how well a model fits the dataset.
    
    Returns:
        (bucket, priority) tuple
    """
    from ml.dataset_profile import DataSufficiencyLevel
    
    n = profile.n_rows
    p = profile.n_features
    task_type = profile.target_profile.task_type if profile.target_profile else 'regression'
    epv = profile.events_per_variable
    
    score = 100  # Start with perfect score, subtract for issues
    
    # Sample size check
    if n < info['min_samples']:
        score -= 40
    elif n < info['min_samples'] * 2:
        score -= 20
    elif n >= info['min_samples'] * 5:
        score += 10
    
    # Events per variable (classification)
    if task_type == 'classification' and epv is not None:
        if epv < info['min_epv'] / 2:
            score -= 50
        elif epv < info['min_epv']:
            score -= 30
    
    # High dimensionality
    if profile.p_n_ratio > 0.5:
        if info['good_for_high_dim']:
            score += 10
        else:
            score -= 30
    elif profile.p_n_ratio > 0.2:
        if not info['good_for_high_dim']:
            score -= 10
    
    # Outliers
    outlier_rate = len(profile.features_with_outliers) / p if p > 0 else 0
    if outlier_rate > 0.3:
        if info['robust_to_outliers']:
            score += 10
        else:
            score -= 20
    
    # Missing data
    if profile.n_features_high_missing > 0:
        if info['handles_missing']:
            score += 10
        else:
            score -= 10
    
    # Class imbalance
    if task_type == 'classification' and profile.target_profile:
        if profile.target_profile.is_imbalanced:
            if profile.target_profile.imbalance_severity == 'severe':
                score -= 15
    
    # Determine bucket
    if score >= 70:
        bucket = RecommendationBucket.RECOMMENDED
        priority = 100 - score  # Higher score = lower priority number
    elif score >= 40:
        bucket = RecommendationBucket.WORTH_TRYING
        priority = 150 - score
    else:
        bucket = RecommendationBucket.NOT_RECOMMENDED
        priority = 200 - score
    
    return bucket, priority


def _generate_family_recommendations(
    profile: Any, 
    recommended: List[ModelRecommendation],
    worth_trying: List[ModelRecommendation]
) -> List[CoachRecommendation]:
    """Generate family-level recommendations for backward compatibility."""
    
    # Group recommended models by family
    families: Dict[str, List[str]] = {}
    for rec in recommended + worth_trying:
        if rec.group not in families:
            families[rec.group] = []
        families[rec.group].append(rec.model_key)
    
    family_recs = []
    priority = 1
    
    for group, models in families.items():
        # Determine if family is recommended based on model buckets
        rec_count = sum(1 for r in recommended if r.group == group)
        
        if rec_count > 0:
            bucket = RecommendationBucket.RECOMMENDED
        else:
            bucket = RecommendationBucket.WORTH_TRYING
        
        # Build why list
        why = [f"{len(models)} model(s) suitable for your dataset"]
        
        # Add specific reasons based on group
        if group == 'Linear':
            why.append("Interpretable coefficients for each feature")
            why.append("Fast training and prediction")
        elif group == 'Trees':
            why.append("Handles nonlinear relationships automatically")
            why.append("Robust to outliers and doesn't require scaling")
        elif group == 'Boosting':
            why.append("Often achieves best predictive accuracy")
            why.append("Handles nonlinearity and interactions")
        elif group == 'Neural Net':
            why.append("Can learn complex patterns from large data")
        
        family_recs.append(CoachRecommendation(
            group=group,
            recommended_models=models,
            why=why,
            when_not_to_use=["See individual model recommendations for details"],
            suggested_preprocessing=["See preprocessing recommendations"],
            priority=priority,
            bucket=bucket
        ))
        priority += 1
    
    return family_recs


def _generate_preprocessing_recommendations(profile: Any) -> List[PreprocessingRecommendation]:
    """Generate preprocessing recommendations based on dataset profile."""
    recs = []
    
    # Missingness
    if profile.n_features_with_missing > 0:
        if profile.n_features_high_missing > 0:
            recs.append(PreprocessingRecommendation(
                step_name="Handle Missing Values",
                step_key="imputation",
                rationale=f"{profile.n_features_high_missing} features have >10% missing values",
                priority="required",
                affected_model_families=["Linear Models", "Neural Networks", "k-NN"],
                plain_language_explanation="Missing values can cause errors or bias. You need to either "
                                          "fill them in (imputation) or use a model that handles them natively (trees).",
                how_to_implement="Use mean/median for simple imputation, or KNN/iterative for better results. "
                                "Consider adding missingness indicator columns."
            ))
        else:
            recs.append(PreprocessingRecommendation(
                step_name="Handle Missing Values",
                step_key="imputation",
                rationale=f"{profile.n_features_with_missing} features have some missing values",
                priority="recommended",
                affected_model_families=["Linear Models", "Neural Networks"],
                plain_language_explanation="Even small amounts of missing data need handling for most models.",
                how_to_implement="Simple mean/median imputation is usually sufficient for low missingness."
            ))
    
    # Scaling
    if profile.n_numeric > 0:
        recs.append(PreprocessingRecommendation(
            step_name="Scale Numeric Features",
            step_key="scaling",
            rationale=f"You have {profile.n_numeric} numeric features with different scales",
            priority="required" if profile.n_numeric > 1 else "recommended",
            affected_model_families=["Linear Models", "Neural Networks", "k-NN", "SVM"],
            plain_language_explanation="Features on different scales (e.g., age 0-100 vs income 0-1M) "
                                      "can bias models. Scaling puts them on equal footing.",
            how_to_implement="StandardScaler (mean=0, std=1) is most common. "
                            "Tree-based models don't need scaling."
        ))
    
    # High cardinality
    if len(profile.high_cardinality_features) > 0:
        recs.append(PreprocessingRecommendation(
            step_name="Encode High-Cardinality Categoricals",
            step_key="high_card_encoding",
            rationale=f"{len(profile.high_cardinality_features)} features have many categories",
            priority="required",
            affected_model_families=["Linear Models", "Neural Networks"],
            plain_language_explanation="Categorical features with many values (like ZIP codes) can't be "
                                      "one-hot encoded efficiently. Special encoding is needed.",
            how_to_implement="Try target encoding (encodes with average target value) or "
                            "frequency encoding. Tree models handle high cardinality naturally."
        ))
    
    # Outliers
    if len(profile.features_with_outliers) > 0:
        recs.append(PreprocessingRecommendation(
            step_name="Address Outliers",
            step_key="outliers",
            rationale=f"{len(profile.features_with_outliers)} features have outliers",
            priority="recommended",
            affected_model_families=["Linear Models", "k-NN"],
            plain_language_explanation="Extreme values can unduly influence model training. "
                                      "You may want to cap them or use robust models.",
            how_to_implement="Options: Winsorize (cap at percentiles), remove, or use robust models (Huber, trees)."
        ))
    
    # Skewed features
    if len(profile.highly_skewed_features) > 0:
        recs.append(PreprocessingRecommendation(
            step_name="Transform Skewed Features",
            step_key="skew_transform",
            rationale=f"{len(profile.highly_skewed_features)} features are highly skewed",
            priority="optional",
            affected_model_families=["Linear Models"],
            plain_language_explanation="Heavily skewed features (like income) can be transformed "
                                      "to be more normally distributed, which helps some models.",
            how_to_implement="Try log transform for right-skewed data, or power transforms (Box-Cox, Yeo-Johnson)."
        ))
    
    # Interpretability vs performance tradeoff (always surface)
    recs.append(PreprocessingRecommendation(
        step_name="Interpretability vs Performance",
        step_key="interpretability_tradeoff",
        rationale="Preprocessing choices affect both model accuracy and explainability",
        priority="optional",
        affected_model_families=["All Models"],
        plain_language_explanation=(
            "**Interpretability-focused:** Keeps pipelines simple (no log transform, PCA, or KMeans features). "
            "Coefficients and feature importances stay meaningful. Best when you need to explain results to stakeholders. "
            "**Performance-focused:** Allows log transforms, PCA, and cluster-based features. Often improves accuracy "
            "but obscures direct feature–outcome relationships. Use when prediction quality matters most."
        ),
        how_to_implement="Set 'Interpretability preference' in Preprocessing to High (interpretability), Balanced, or Performance. "
                        "High disables log transform, PCA, and KMeans; Performance keeps them available."
    ))

    # Class imbalance
    if profile.target_profile and profile.target_profile.is_imbalanced:
        severity = profile.target_profile.imbalance_severity
        recs.append(PreprocessingRecommendation(
            step_name="Handle Class Imbalance",
            step_key="imbalance",
            rationale=f"{severity.title()} class imbalance ({profile.target_profile.class_balance_ratio:.1f}:1 ratio)",
            priority="required" if severity == "severe" else "recommended",
            affected_model_families=["All classification models"],
            plain_language_explanation="When one class is much rarer than others, models tend to ignore it. "
                                      "The minority class is often the one you care most about!",
            how_to_implement="Enable the class weighting toggle on the Train page. This sets "
                            "class_weight='balanced' for supported models (Logistic Regression, "
                            "Random Forest, ExtraTrees, HistGradientBoosting, SVM, LightGBM). "
                            "XGBoost uses computed sample weights for the same effect. "
                            "Focus on F1/PR-AUC metrics, not accuracy."
        ))
    
    return recs


def _generate_eda_recommendations(profile: Any) -> Tuple[List[str], Dict[str, List[str]]]:
    """Generate EDA recommendations."""
    
    baseline = [
        "Summary statistics for all features",
        "Target distribution visualization",
        "Missing value heatmap",
        "Correlation matrix for numeric features",
        "Feature-target correlations"
    ]
    
    advanced = {
        "Linear Models": [
            "Check linearity: scatter plots of features vs target",
            "Residual analysis: look for patterns in residuals",
            "Multicollinearity check: correlation matrix, VIF if available",
            "Influence diagnostics: identify high-leverage points",
            "Normality of residuals (for inference, not prediction)"
        ],
        "Tree-Based Models": [
            "Feature interactions: look for non-additive effects",
            "Nonlinearity indicators: binned averages by feature",
            "Monotonic trends: does target consistently increase/decrease with feature?",
            "Feature importance comparison across different tree methods",
            "Partial dependence plots (after training)"
        ],
        "Neural Networks": [
            "Data sufficiency check: at least 20× samples per feature recommended",
            "Feature scaling necessity: check feature value ranges",
            "Leakage detection: features too correlated with target",
            "Categorical encoding strategy: many categories need embedding",
            "Train/validation split quality: ensure representative distribution"
        ],
        "Boosting": [
            "Learning curve analysis: does more data help?",
            "Feature importance stability across folds",
            "Interaction detection: tree-based interaction tests",
            "Early stopping analysis: when does overfitting begin?"
        ]
    }
    
    # Add dataset-specific recommendations
    if profile.n_features_with_missing > 0:
        baseline.append("Missing data pattern analysis (MCAR/MAR/MNAR)")
    
    if profile.target_profile and profile.target_profile.task_type == 'classification':
        baseline.append("Class balance visualization")
        if profile.target_profile.is_imbalanced:
            baseline.append("Investigate minority class characteristics")
    
    if len(profile.features_with_outliers) > 0:
        baseline.append("Outlier investigation: are they errors or genuine?")
    
    return baseline, advanced


# Legacy compatibility function
def coach_recommendations(
    signals: Any,
    optional_results: Optional[Dict[str, Any]] = None,
    eda_insights: Optional[List[Dict[str, Any]]] = None
) -> List[CoachRecommendation]:
    """
    Legacy function for backward compatibility.
    
    Converts DatasetSignals to family-level recommendations.
    """
    from ml.eda_recommender import DatasetSignals
    
    # Convert signals to a mini-profile for assessment
    task_type = signals.task_type_final
    n_rows = signals.n_rows
    n_cols = signals.n_cols
    
    recommendations = []
    
    # Always recommend linear models as baseline
    if task_type == 'regression':
        outlier_rate = signals.target_stats.get('outlier_rate', 0) if signals.target_stats else 0
        recommended_models = ['glm', 'ridge']
        why_text = [
            "Start with interpretable linear models for baseline performance",
            f"Your dataset has {n_rows:,} samples × {n_cols} features"
        ]
        
        if outlier_rate > 0.1:
            recommended_models.append('huber')
            why_text.append(f"Outlier rate: {outlier_rate:.1%} - Huber regression is more robust")
        
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=recommended_models,
            why=why_text,
            when_not_to_use=["If data shows strong nonlinear patterns"],
            suggested_preprocessing=['Scale numeric features'],
            priority=1,
            bucket=RecommendationBucket.RECOMMENDED
        ))
    else:
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=['logreg', 'glm'],
            why=[
                "Start with interpretable logistic regression",
                f"Your dataset has {n_rows:,} samples × {n_cols} features"
            ],
            when_not_to_use=["If decision boundaries are highly nonlinear"],
            suggested_preprocessing=['Scale numeric features'],
            priority=1,
            bucket=RecommendationBucket.RECOMMENDED
        ))
    
    # Tree-based models
    if n_rows >= 50:
        recommendations.append(CoachRecommendation(
            group='Trees',
            recommended_models=['rf', 'extratrees_reg' if task_type == 'regression' else 'extratrees_clf'],
            why=[
                "Tree models capture nonlinearity and interactions automatically",
                "Robust to outliers and don't require scaling"
            ],
            when_not_to_use=["If you need highly interpretable coefficients"],
            suggested_preprocessing=[],
            priority=2,
            bucket=RecommendationBucket.RECOMMENDED if n_rows >= 100 else RecommendationBucket.WORTH_TRYING
        ))
    
    # Boosting
    if n_rows >= 100:
        if task_type == 'regression':
            boosting_models = ['histgb_reg', 'xgb_reg', 'lgbm_reg']
        else:
            boosting_models = ['histgb_clf', 'xgb_clf', 'lgbm_clf']
        recommendations.append(CoachRecommendation(
            group='Boosting',
            recommended_models=boosting_models,
            why=[
                "Often achieves the best predictive accuracy",
                "Handles missing values and mixed feature types"
            ],
            when_not_to_use=["When training time is critical"],
            suggested_preprocessing=[],
            priority=3,
            bucket=RecommendationBucket.RECOMMENDED if n_rows >= 500 else RecommendationBucket.WORTH_TRYING
        ))
    
    # Neural networks
    if n_rows >= 1000:
        recommendations.append(CoachRecommendation(
            group='Neural Net',
            recommended_models=['nn'],
            why=[
                f"With {n_rows:,} samples, neural networks have enough data",
                "Can learn complex nonlinear patterns"
            ],
            when_not_to_use=["If interpretability is required"],
            suggested_preprocessing=['Scale all features', 'Encode categoricals'],
            priority=4,
            bucket=RecommendationBucket.WORTH_TRYING
        ))
    
    # Merge by group to avoid duplicates
    merged = _merge_recommendations_by_group(recommendations)
    merged.sort(key=lambda x: x.priority)
    
    return merged


def _merge_recommendations_by_group(recommendations: List[CoachRecommendation]) -> List[CoachRecommendation]:
    """Merge recommendations with the same group."""
    group_map: Dict[str, CoachRecommendation] = {}
    
    for rec in recommendations:
        if rec.group not in group_map:
            group_map[rec.group] = CoachRecommendation(
                group=rec.group,
                recommended_models=list(rec.recommended_models),
                why=list(rec.why),
                when_not_to_use=list(rec.when_not_to_use),
                suggested_preprocessing=list(rec.suggested_preprocessing),
                priority=rec.priority,
                readiness_checks=list(rec.readiness_checks),
                bucket=rec.bucket
            )
        else:
            existing = group_map[rec.group]
            for model in rec.recommended_models:
                if model not in existing.recommended_models:
                    existing.recommended_models.append(model)
            for reason in rec.why:
                if reason not in existing.why:
                    existing.why.append(reason)
            for caveat in rec.when_not_to_use:
                if caveat not in existing.when_not_to_use:
                    existing.when_not_to_use.append(caveat)
            for prep in rec.suggested_preprocessing:
                if prep not in existing.suggested_preprocessing:
                    existing.suggested_preprocessing.append(prep)
            existing.priority = min(existing.priority, rec.priority)
    
    return list(group_map.values())
