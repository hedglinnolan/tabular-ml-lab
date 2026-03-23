"""
Model Registry: Centralized model specifications with capability metadata.
"""
from dataclasses import dataclass, field
from typing import Dict, Callable, Any, Literal, Optional
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from models.nn_whuber import NNWeightedHuberWrapper
from models.glm import GLMWrapper
from models.huber_glm import HuberGLMWrapper
from models.rf import RFWrapper
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


@dataclass
class ModelCapabilities:
    """Capability metadata for a model."""
    supports_regression: bool
    supports_classification: bool
    supports_predict_proba: bool
    supports_partial_dependence: bool
    supports_shap: Literal["none", "linear", "tree", "kernel"]
    requires_scaled_numeric: bool
    recommended_for_high_dim: bool
    interpretability_tier: Literal["high", "medium", "low"] = "medium"
    notes: list[str] = field(default_factory=list)
    supports_class_weight: bool = False
    supports_sample_weight_balancing: bool = False


@dataclass
class ModelSpec:
    """Specification for a model in the registry."""
    key: str
    name: str
    group: str  # Linear, Trees, Boosting, Distance, Margin, Probabilistic, Neural Net
    factory: Callable[[str, int], Any]  # (task_type, random_state) -> estimator
    default_params: Dict[str, Any]
    hyperparam_schema: Dict[str, Dict[str, Any]]  # UI control definitions
    capabilities: ModelCapabilities


def _create_ridge(task_type: str, random_state: int):
    """Factory for Ridge regression."""
    return Ridge(random_state=random_state, alpha=1.0)


def _create_lasso(task_type: str, random_state: int):
    """Factory for Lasso regression."""
    return Lasso(random_state=random_state, alpha=1.0, max_iter=1000)


def _create_elasticnet(task_type: str, random_state: int):
    """Factory for ElasticNet regression."""
    return ElasticNet(random_state=random_state, alpha=1.0, l1_ratio=0.5, max_iter=1000)


def _create_knn_reg(task_type: str, random_state: int):
    """Factory for kNN regression."""
    return KNeighborsRegressor(n_neighbors=5, weights='uniform')


def _create_knn_clf(task_type: str, random_state: int):
    """Factory for kNN classification."""
    return KNeighborsClassifier(n_neighbors=5, weights='uniform')


def _create_logreg(task_type: str, random_state: int):
    """Factory for Logistic Regression."""
    # Use saga solver — supports both l1 and l2 penalties (lbfgs only supports l2)
    return LogisticRegression(random_state=random_state, C=1.0, penalty='l2', solver='saga', max_iter=1000)


def _create_extratrees_reg(task_type: str, random_state: int):
    """Factory for ExtraTrees regression."""
    return ExtraTreesRegressor(random_state=random_state, n_estimators=100, max_depth=None)


def _create_extratrees_clf(task_type: str, random_state: int):
    """Factory for ExtraTrees classification."""
    return ExtraTreesClassifier(random_state=random_state, n_estimators=100, max_depth=None)


def _create_histgb_reg(task_type: str, random_state: int):
    """Factory for HistGradientBoosting regression."""
    return HistGradientBoostingRegressor(random_state=random_state, max_depth=3, learning_rate=0.1, max_iter=100)


def _create_histgb_clf(task_type: str, random_state: int):
    """Factory for HistGradientBoosting classification."""
    return HistGradientBoostingClassifier(random_state=random_state, max_depth=3, learning_rate=0.1, max_iter=100)


def _create_svr(task_type: str, random_state: int):
    """Factory for SVR (Support Vector Regression)."""
    return SVR(kernel='rbf', C=1.0, gamma='scale')


def _create_svc(task_type: str, random_state: int):
    """Factory for SVC (Support Vector Classification)."""
    return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state)


def _create_gaussian_nb(task_type: str, random_state: int):
    """Factory for Gaussian Naive Bayes."""
    return GaussianNB()


def _create_lda(task_type: str, random_state: int):
    """Factory for Linear Discriminant Analysis."""
    return LinearDiscriminantAnalysis()


def _create_nn(task_type: str, random_state: int):
    """Factory for Neural Network."""
    return NNWeightedHuberWrapper(dropout=0.1, task_type=task_type)


def _create_glm(task_type: str, random_state: int):
    """Factory for GLM."""
    return GLMWrapper(task_type=task_type)


def _create_huber(task_type: str, random_state: int):
    """Factory for Huber GLM."""
    return HuberGLMWrapper(epsilon=1.35, alpha=0.0)


def _create_rf(task_type: str, random_state: int):
    """Factory for Random Forest."""
    return RFWrapper(n_estimators=100, max_depth=None, min_samples_leaf=1, task_type=task_type)


def _create_xgb_reg(task_type: str, random_state: int):
    """Factory for XGBoost Regressor."""
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=random_state, verbosity=0, tree_method='hist'
    )


def _create_xgb_clf(task_type: str, random_state: int):
    """Factory for XGBoost Classifier."""
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=random_state, verbosity=0, tree_method='hist',
        eval_metric='logloss'
    )


def _create_lgbm_reg(task_type: str, random_state: int):
    """Factory for LightGBM Regressor."""
    return LGBMRegressor(
        n_estimators=100, max_depth=-1, learning_rate=0.1,
        random_state=random_state, verbosity=-1
    )


def _create_lgbm_clf(task_type: str, random_state: int):
    """Factory for LightGBM Classifier."""
    return LGBMClassifier(
        n_estimators=100, max_depth=-1, learning_rate=0.1,
        random_state=random_state, verbosity=-1
    )


def get_registry() -> Dict[str, ModelSpec]:
    """Get the complete model registry."""
    registry = {}
    
    # Linear Models - Regression
    registry['ridge'] = ModelSpec(
        key='ridge',
        name='Ridge Regression',
        group='Linear',
        factory=_create_ridge,
        default_params={'alpha': 1.0},
        hyperparam_schema={
            'alpha': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Regularization strength'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=True,
            recommended_for_high_dim=True,
            interpretability_tier="high",
            notes=['L2 regularization prevents overfitting', 'Good for multicollinearity']
        )
    )
    
    registry['lasso'] = ModelSpec(
        key='lasso',
        name='Lasso Regression',
        group='Linear',
        factory=_create_lasso,
        default_params={'alpha': 1.0},
        hyperparam_schema={
            'alpha': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Regularization strength'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=True,
            recommended_for_high_dim=True,
            interpretability_tier="high",
            notes=['L1 regularization performs feature selection', 'Can zero out coefficients']
        )
    )
    
    registry['elasticnet'] = ModelSpec(
        key='elasticnet',
        name='ElasticNet Regression',
        group='Linear',
        factory=_create_elasticnet,
        default_params={'alpha': 1.0, 'l1_ratio': 0.5},
        hyperparam_schema={
            'alpha': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Regularization strength'},
            'l1_ratio': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'help': 'L1 ratio (0=L2 only, 1=L1 only)'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=True,
            recommended_for_high_dim=True,
            interpretability_tier="high",
            notes=['Combines L1 and L2 regularization', 'Balances feature selection and stability']
        )
    )
    
    # Linear Models - Classification
    registry['logreg'] = ModelSpec(
        key='logreg',
        name='Logistic Regression',
        group='Linear',
        factory=_create_logreg,
        default_params={'C': 1.0, 'penalty': 'l2', 'max_iter': 1000},
        hyperparam_schema={
            'C': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Inverse regularization strength'},
            'penalty': {'type': 'select', 'options': ['l2', 'l1'], 'default': 'l2', 'help': 'Regularization type'},
            'max_iter': {'type': 'int', 'min': 100, 'max': 5000, 'default': 1000, 'help': 'Maximum iterations'}
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=True,
            recommended_for_high_dim=True,
            interpretability_tier="high",
            notes=['Interpretable coefficients', 'Good baseline for classification'],
            supports_class_weight=True
        )
    )
    
    # Distance-based
    registry['knn_reg'] = ModelSpec(
        key='knn_reg',
        name='k-Nearest Neighbors (Regression)',
        group='Distance',
        factory=_create_knn_reg,
        default_params={'n_neighbors': 5, 'weights': 'uniform'},
        hyperparam_schema={
            'n_neighbors': {'type': 'int', 'min': 1, 'max': 50, 'default': 5, 'help': 'Number of neighbors (must be ≤ sample size)'},
            'weights': {'type': 'select', 'options': ['uniform', 'distance'], 'default': 'uniform', 'help': 'Weight function'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='kernel',
            requires_scaled_numeric=True,
            recommended_for_high_dim=False,
            notes=['Non-parametric, instance-based', 'Sensitive to feature scaling', 'n_neighbors must be ≤ training samples', 'Slow for large datasets']
        )
    )
    
    registry['knn_clf'] = ModelSpec(
        key='knn_clf',
        name='k-Nearest Neighbors (Classification)',
        group='Distance',
        factory=_create_knn_clf,
        default_params={'n_neighbors': 5, 'weights': 'uniform'},
        hyperparam_schema={
            'n_neighbors': {'type': 'int', 'min': 1, 'max': 50, 'default': 5, 'help': 'Number of neighbors (must be ≤ sample size)'},
            'weights': {'type': 'select', 'options': ['uniform', 'distance'], 'default': 'uniform', 'help': 'Weight function'}
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='kernel',
            requires_scaled_numeric=True,
            recommended_for_high_dim=False,
            notes=['Non-parametric, instance-based', 'Sensitive to feature scaling', 'n_neighbors must be ≤ training samples', 'Slow for large datasets']
        )
    )
    
    # Trees
    registry['extratrees_reg'] = ModelSpec(
        key='extratrees_reg',
        name='Extra Trees (Regression)',
        group='Trees',
        factory=_create_extratrees_reg,
        default_params={'n_estimators': 100, 'max_depth': None},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of trees'},
            'max_depth': {'type': 'int_or_none', 'min': 1, 'max': 50, 'default': None, 'help': 'Max depth (None=unlimited)'},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20, 'default': 1, 'help': 'Min samples per leaf'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            notes=['More random splits than RF', 'Robust to outliers', 'Handles missing values']
        )
    )
    
    registry['extratrees_clf'] = ModelSpec(
        key='extratrees_clf',
        name='Extra Trees (Classification)',
        group='Trees',
        factory=_create_extratrees_clf,
        default_params={'n_estimators': 100, 'max_depth': None},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of trees'},
            'max_depth': {'type': 'int_or_none', 'min': 1, 'max': 50, 'default': None, 'help': 'Max depth (None=unlimited)'},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20, 'default': 1, 'help': 'Min samples per leaf'}
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            notes=['More random splits than RF', 'Robust to outliers', 'Handles missing values'],
            supports_class_weight=True
        )
    )
    
    # Boosting
    registry['histgb_reg'] = ModelSpec(
        key='histgb_reg',
        name='Histogram Gradient Boosting (Regression)',
        group='Boosting',
        factory=_create_histgb_reg,
        default_params={'max_depth': 3, 'learning_rate': 0.1, 'max_iter': 100},
        hyperparam_schema={
            'max_depth': {'type': 'int', 'min': 1, 'max': 20, 'default': 3, 'help': 'Max depth of trees'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'help': 'Learning rate'},
            'max_iter': {'type': 'int', 'min': 10, 'max': 500, 'default': 100, 'help': 'Number of boosting iterations'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            notes=['Fast gradient boosting', 'Handles missing values', 'Good for large datasets']
        )
    )
    
    registry['histgb_clf'] = ModelSpec(
        key='histgb_clf',
        name='Histogram Gradient Boosting (Classification)',
        group='Boosting',
        factory=_create_histgb_clf,
        default_params={'max_depth': 3, 'learning_rate': 0.1, 'max_iter': 100},
        hyperparam_schema={
            'max_depth': {'type': 'int', 'min': 1, 'max': 20, 'default': 3, 'help': 'Max depth of trees'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'help': 'Learning rate'},
            'max_iter': {'type': 'int', 'min': 10, 'max': 500, 'default': 100, 'help': 'Number of boosting iterations'}
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            notes=['Fast gradient boosting', 'Handles missing values', 'Good for large datasets'],
            supports_class_weight=True
        )
    )
    
    # Margin-based (Advanced)
    registry['svr'] = ModelSpec(
        key='svr',
        name='Support Vector Regression',
        group='Margin',
        factory=_create_svr,
        default_params={'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
        hyperparam_schema={
            'C': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Regularization parameter'},
            'gamma': {'type': 'select', 'options': ['scale', 'auto', '0.001', '0.01', '0.1', '1.0'], 'default': 'scale', 'help': 'Kernel coefficient'},
            'kernel': {'type': 'select', 'options': ['rbf', 'linear', 'poly'], 'default': 'rbf', 'help': 'Kernel type'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='kernel',
            requires_scaled_numeric=True,
            recommended_for_high_dim=False,
            notes=['Advanced model', 'Slow for large datasets', 'Requires careful tuning']
        )
    )
    
    registry['svc'] = ModelSpec(
        key='svc',
        name='Support Vector Classification',
        group='Margin',
        factory=_create_svc,
        default_params={'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
        hyperparam_schema={
            'C': {'type': 'float', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'log': True, 'help': 'Regularization parameter'},
            'gamma': {'type': 'select', 'options': ['scale', 'auto', '0.001', '0.01', '0.1', '1.0'], 'default': 'scale', 'help': 'Kernel coefficient'},
            'kernel': {'type': 'select', 'options': ['rbf', 'linear', 'poly'], 'default': 'rbf', 'help': 'Kernel type'}
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='kernel',
            requires_scaled_numeric=True,
            recommended_for_high_dim=False,
            notes=['Advanced model', 'Slow for large datasets', 'Requires careful tuning'],
            supports_class_weight=True
        )
    )
    
    # Probabilistic
    registry['gaussian_nb'] = ModelSpec(
        key='gaussian_nb',
        name='Gaussian Naive Bayes',
        group='Probabilistic',
        factory=_create_gaussian_nb,
        default_params={},
        hyperparam_schema={},
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='none',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            notes=['Fast and simple', 'Assumes feature independence', 'Good baseline']
        )
    )
    
    registry['lda'] = ModelSpec(
        key='lda',
        name='Linear Discriminant Analysis',
        group='Probabilistic',
        factory=_create_lda,
        default_params={},
        hyperparam_schema={},
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=True,
            recommended_for_high_dim=False,
            notes=['Linear dimensionality reduction', 'Assumes Gaussian distributions', 'Interpretable']
        )
    )
    
    # Existing models (wrapped in registry)
    registry['nn'] = ModelSpec(
        key='nn',
        name='Neural Network',
        group='Neural Net',
        factory=_create_nn,
        default_params={
            'dropout': 0.1, 'epochs': 200, 'batch_size': 256, 'lr': 0.0015, 
            'weight_decay': 0.0002, 'patience': 30, 'num_layers': 2, 
            'layer_width': 32, 'activation': 'relu', 'architecture_pattern': 'constant'
        },
        hyperparam_schema={
            'num_layers': {'type': 'int', 'min': 1, 'max': 5, 'default': 2, 'help': 'Number of hidden layers'},
            'layer_width': {'type': 'int', 'min': 8, 'max': 256, 'default': 32, 'help': 'Base layer width'},
            'architecture_pattern': {'type': 'select', 'options': ['constant', 'pyramid', 'funnel'], 'default': 'constant', 'help': 'Layer width pattern'},
            'activation': {'type': 'select', 'options': ['relu', 'tanh', 'leaky_relu', 'elu'], 'default': 'relu', 'help': 'Activation function'},
            'dropout': {'type': 'float', 'min': 0.0, 'max': 0.5, 'default': 0.1, 'help': 'Dropout rate'},
            'epochs': {'type': 'int', 'min': 50, 'max': 500, 'default': 200, 'help': 'Number of epochs'},
            'batch_size': {'type': 'int', 'min': 32, 'max': 512, 'default': 256, 'help': 'Batch size'},
            'lr': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'default': 0.0015, 'log': True, 'help': 'Learning rate'},
            'weight_decay': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'default': 0.0002, 'log': True, 'help': 'L2 regularization (weight decay)'},
            'patience': {'type': 'int', 'min': 5, 'max': 50, 'default': 30, 'help': 'Early stopping patience'},
            'loss_function': {'type': 'select', 'options': ['mse', 'huber', 'mae', 'weighted_huber'], 'default': 'mse', 'help': 'Loss function for regression (MSE is standard; weighted_huber emphasizes high-value targets)'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='kernel',
            requires_scaled_numeric=True,
            recommended_for_high_dim=True,
            interpretability_tier="low",
            notes=['Deep learning', 'Can capture complex patterns', 'Requires more data']
        )
    )
    
    registry['glm'] = ModelSpec(
        key='glm',
        name='GLM (OLS/Logistic)',
        group='Linear',
        factory=_create_glm,
        default_params={},
        hyperparam_schema={},
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="high",
            notes=['Simple baseline', 'Interpretable', 'Sensitive to outliers']
        )
    )
    
    registry['huber'] = ModelSpec(
        key='huber',
        name='GLM (Huber)',
        group='Linear',
        factory=_create_huber,
        default_params={'epsilon': 1.35, 'alpha': 0.0},
        hyperparam_schema={
            'epsilon': {'type': 'float', 'min': 1.0, 'max': 2.0, 'default': 1.35, 'help': 'Epsilon parameter'},
            'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.0, 'help': 'Regularization strength'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='linear',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="high",
            notes=['Robust to outliers', 'Regression only', 'Less sensitive than OLS']
        )
    )
    
    registry['rf'] = ModelSpec(
        key='rf',
        name='Random Forest',
        group='Trees',
        factory=_create_rf,
        default_params={'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of trees'},
            'max_depth': {'type': 'int_or_none', 'min': 1, 'max': 50, 'default': None, 'help': 'Max depth (None=unlimited)'},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20, 'default': 1, 'help': 'Min samples per leaf'}
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="medium",
            notes=['Robust ensemble', 'Handles missing values', 'Feature importance available'],
            supports_class_weight=True
        )
    )
    
    # XGBoost
    registry['xgb_reg'] = ModelSpec(
        key='xgb_reg',
        name='XGBoost (Regression)',
        group='Boosting',
        factory=_create_xgb_reg,
        default_params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
                        'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of boosting rounds'},
            'max_depth': {'type': 'int', 'min': 1, 'max': 20, 'default': 3, 'help': 'Max depth of trees'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'log': True, 'help': 'Learning rate'},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Row subsampling ratio'},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Column subsampling ratio'},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L1 regularization'},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 1.0, 'help': 'L2 regularization'},
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="low",
            notes=['Industry-standard gradient boosting', 'Regularization built-in', 'Handles missing values natively']
        )
    )

    registry['xgb_clf'] = ModelSpec(
        key='xgb_clf',
        name='XGBoost (Classification)',
        group='Boosting',
        factory=_create_xgb_clf,
        default_params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
                        'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of boosting rounds'},
            'max_depth': {'type': 'int', 'min': 1, 'max': 20, 'default': 3, 'help': 'Max depth of trees'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'log': True, 'help': 'Learning rate'},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Row subsampling ratio'},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Column subsampling ratio'},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L1 regularization'},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 1.0, 'help': 'L2 regularization'},
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="low",
            notes=['Industry-standard gradient boosting', 'Regularization built-in', 'Handles missing values natively'],
            supports_sample_weight_balancing=True
        )
    )

    # LightGBM
    registry['lgbm_reg'] = ModelSpec(
        key='lgbm_reg',
        name='LightGBM (Regression)',
        group='Boosting',
        factory=_create_lgbm_reg,
        default_params={'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1,
                        'num_leaves': 31, 'subsample': 1.0, 'colsample_bytree': 1.0,
                        'reg_alpha': 0.0, 'reg_lambda': 0.0},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of boosting rounds'},
            'max_depth': {'type': 'int', 'min': -1, 'max': 50, 'default': -1, 'help': 'Max depth (-1=unlimited)'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'log': True, 'help': 'Learning rate'},
            'num_leaves': {'type': 'int', 'min': 8, 'max': 256, 'default': 31, 'help': 'Max number of leaves per tree'},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Row subsampling ratio'},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Column subsampling ratio'},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L1 regularization'},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L2 regularization'},
        },
        capabilities=ModelCapabilities(
            supports_regression=True,
            supports_classification=False,
            supports_predict_proba=False,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="low",
            notes=['Leaf-wise tree growth (faster)', 'Handles categoricals natively', 'Lower memory usage than XGBoost']
        )
    )

    registry['lgbm_clf'] = ModelSpec(
        key='lgbm_clf',
        name='LightGBM (Classification)',
        group='Boosting',
        factory=_create_lgbm_clf,
        default_params={'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1,
                        'num_leaves': 31, 'subsample': 1.0, 'colsample_bytree': 1.0,
                        'reg_alpha': 0.0, 'reg_lambda': 0.0},
        hyperparam_schema={
            'n_estimators': {'type': 'int', 'min': 10, 'max': 1000, 'default': 100, 'help': 'Number of boosting rounds'},
            'max_depth': {'type': 'int', 'min': -1, 'max': 50, 'default': -1, 'help': 'Max depth (-1=unlimited)'},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'log': True, 'help': 'Learning rate'},
            'num_leaves': {'type': 'int', 'min': 8, 'max': 256, 'default': 31, 'help': 'Max number of leaves per tree'},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Row subsampling ratio'},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'help': 'Column subsampling ratio'},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L1 regularization'},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 10.0, 'default': 0.0, 'help': 'L2 regularization'},
        },
        capabilities=ModelCapabilities(
            supports_regression=False,
            supports_classification=True,
            supports_predict_proba=True,
            supports_partial_dependence=True,
            supports_shap='tree',
            requires_scaled_numeric=False,
            recommended_for_high_dim=False,
            interpretability_tier="low",
            notes=['Leaf-wise tree growth (faster)', 'Handles categoricals natively', 'Lower memory usage than XGBoost'],
            supports_class_weight=True
        )
    )

    return registry
