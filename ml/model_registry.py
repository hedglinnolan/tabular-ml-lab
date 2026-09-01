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
from models.glm import GLMWrapper
from models.huber_glm import HuberGLMWrapper
from models.rf import RFWrapper

# ─────────────────────────────────────────────────────────────────────────────
# `MODELS-026`. THE GRADIENT-BOOSTING BACKENDS ARE OPTIONAL, LIKE torch.
#
# These were `from xgboost import ...` and `from lightgbm import ...` at module
# scope, so an interpreter missing either one lost `GET /models` — and with it
# Train, Explain, the figures and the report — to an unhandled
# `ModuleNotFoundError` that reached Starlette as twenty-one characters of
# *Internal Server Error*. Four human drives were spent on that.
#
# `TEST-038` is the standard already in this codebase and it is applied here
# rather than invented: `utils/seed.py` wraps `import torch` in
# `try/except ImportError` with a comment saying it is optional, torch is
# deliberately not installed, and that absence is an expected condition that
# takes no endpoint down. The two ledger rows are the same row, one estimator
# over — and `TEST-038` said so first: *"one of the two is right about whether
# torch is optional, and they cannot both be."* `requirements.txt` declares
# xgboost and lightgbm while the environment treats them as optional; the code
# treated them as mandatory. One of those had to give.
#
# **AND `PRODUCT_VISION.md` SAYS WHICH WAY.** *"The shelf is never shortened"*
# is about not hiding a model from a user, not about refusing to start. A shelf
# that says *"gradient boosting is unavailable in this install, and here is
# why"* is the honest form; a 500 is not, and neither is quietly dropping the
# row — a user who believes a model is unavailable will not think to look for
# it.
#
# scikit-learn is NOT guarded and that is deliberate: it is not one model, it
# is the pipeline, the metrics and eleven of the entries below. Without it
# there is no registry to shorten, so `get_registry` raises
# `RegistryUnavailable` and `api.py` turns it into a refusal a person can read.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGBOOST_ERROR: Optional[str] = None
except Exception as exc:                                  # pragma: no cover
    XGBRegressor = XGBClassifier = None                    # type: ignore
    _XGBOOST_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _LIGHTGBM_ERROR: Optional[str] = None
except Exception as exc:                                  # pragma: no cover
    LGBMRegressor = LGBMClassifier = None                  # type: ignore
    _LIGHTGBM_ERROR = f"{type(exc).__name__}: {exc}"


class RegistryUnavailable(RuntimeError):
    """The registry cannot be built at all in this interpreter.

    Raised only where scikit-learn itself is missing, because then there is no
    shelf to shorten. Carries the sentence a person acts on rather than a
    traceback — `api.py` serves it as a refusal, not a 500.
    """


class ModelUnavailable(RuntimeError):
    """One model's backend is absent. Raised by its factory, never at import.

    The spec stays in the registry and on the shelf, marked, with this reason —
    so the model is *visible and unavailable* rather than silently gone.
    """


def backend_error(key: str) -> Optional[str]:
    """Why `key` cannot be fitted in this interpreter, or `None`.

    Read by `turbotab.models.shelf` so the reason travels to the page. Keyed by
    registry key rather than by package so a caller never has to know which
    model comes from which library.
    """
    if key in ("xgb_reg", "xgb_clf"):
        return _unavailable_sentence("XGBoost", "xgboost", _XGBOOST_ERROR)
    if key in ("lgbm_reg", "lgbm_clf"):
        return _unavailable_sentence("LightGBM", "lightgbm", _LIGHTGBM_ERROR)
    return None


def _unavailable_sentence(name: str, dist: str, error: Optional[str]) -> Optional[str]:
    if error is None:
        return None
    return (f"{name} is not available in this install — importing {dist} "
            f"raised {error}. The model is listed because it is part of this "
            f"shelf; it cannot be fitted here until "
            f"`pip install {dist}` succeeds in the interpreter running the "
            f"app.")


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
    # `L55-B`. Does the FITTED estimator expose `coef_` — one number per
    # predictor that an association estimate can be read off?
    #
    # THIS IS ABOUT THIS APP'S ESTIMATOR, NOT ABOUT THE METHOD. `turbotab
    # .figure_bundle._coefficients_for` draws the coefficient forest plot
    # (§A4.7) by asking `hasattr(estimator, "coef_")` on the model step of the
    # fitted pipeline, and that is the only place in this codebase a
    # per-predictor estimate comes from. So the field means what that check
    # means, and it is verified against a real fit rather than declared by
    # hand — `turbotab/test_the_shelf_reads_the_recorded_design.py`.
    #
    # `interpretability_tier` is NOT this question and must not be used for it:
    # `lda` is tier `medium` and has coefficients, `glm` is tier `high` and its
    # wrapper does not forward them. A tier is about how legible an explanation
    # is; this is about whether there is an estimate at all.
    #
    # `None` MEANS UNDECLARED AND IS NOT A NO. A model added without an answer
    # here takes no part in the purpose-ordering and the shelf says nothing
    # about its coefficients, because returning a value from ignorance is the
    # habit this project keeps rather than a default that reads as a claim.
    exposes_coefficients: Optional[bool] = None


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
    # `probability=True` is the most expensive keyword argument in this file.
    # It runs an internal 5-fold Platt calibration, so ONE "SVC fit" is six
    # libsvm solves — five calibration folds plus the final one — against
    # SVR's one at :208, which has no such parameter. Measured on a dense
    # 120-feature matrix: 0.084 s -> 0.54 s at n=1,000, 1.95 s -> 8.02 s at
    # 5,000, and 79.4 s -> 548.7 s at 20,000, i.e. 6.4x / 4.1x / 6.9x. The
    # cost is wall time and NOT memory (peak RSS 256.8 vs 258.1 MB — libsvm's
    # kernel cache is capped at 200 MB), so anywhere the app discloses it,
    # it must be denominated in minutes and never in GB.
    #
    # It stays anyway, and a caller must not "optimize" it away. ROC-AUC,
    # LogLoss and PR-AUC (ml/eval.py), the calibration curve, the ROC and PR
    # plots, and SVC's KernelExplainer path all gate on
    # `hasattr(model, "predict_proba")`, and sklearn's `available_if` makes
    # that False when the flag is off. Removing it would not raise — it would
    # SILENTLY delete those outputs from the comparison table and from the
    # exported manuscript, which is worse than a crash. The disclosure lives
    # at the moment of choice instead, in `model_viability`
    # (ml/model_coach.py), and again on the train page.
    #
    # DEADLINE, not a preference: sklearn deprecated this parameter in 1.9
    # (the installed version) and removes it in 1.11 — `fit` already emits
    # `FutureWarning: The 'probability' parameter was deprecated in 1.9 ...
    # Use CalibratedClassifierCV(SVC(), ensemble=False) instead`. requirements
    # pins only `scikit-learn>=1.3.0` with no upper bound, so a fresh install
    # breaks this line the day 1.11 ships. The replacement is a real change of
    # fitted output and of ~10 call sites, so it belongs to its own reviewed
    # PR — but that PR has a date on it.
    return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state)


def _create_gaussian_nb(task_type: str, random_state: int):
    """Factory for Gaussian Naive Bayes."""
    return GaussianNB()


def _create_lda(task_type: str, random_state: int):
    """Factory for Linear Discriminant Analysis."""
    return LinearDiscriminantAnalysis()


def _create_nn(task_type: str, random_state: int):
    # Imported here, not at module scope: torch is ~1.1 GB and is needed only
    # for this one model. A lean install runs the whole app without it and
    # simply cannot offer the neural network.
    from models.nn_whuber import NNWeightedHuberWrapper
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
    if XGBRegressor is None:                              # pragma: no cover
        raise ModelUnavailable(backend_error("xgb_reg"))
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=random_state, verbosity=0, tree_method='hist'
    )


def _create_xgb_clf(task_type: str, random_state: int):
    """Factory for XGBoost Classifier."""
    if XGBClassifier is None:                             # pragma: no cover
        raise ModelUnavailable(backend_error("xgb_clf"))
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=random_state, verbosity=0, tree_method='hist',
        eval_metric='logloss'
    )


def _create_lgbm_reg(task_type: str, random_state: int):
    """Factory for LightGBM Regressor."""
    if LGBMRegressor is None:                             # pragma: no cover
        raise ModelUnavailable(backend_error("lgbm_reg"))
    return LGBMRegressor(
        n_estimators=100, max_depth=-1, learning_rate=0.1,
        random_state=random_state, verbosity=-1
    )


def _create_lgbm_clf(task_type: str, random_state: int):
    """Factory for LightGBM Classifier."""
    if LGBMClassifier is None:                            # pragma: no cover
        raise ModelUnavailable(backend_error("lgbm_clf"))
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
            exposes_coefficients=True,
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
            exposes_coefficients=True,
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
            exposes_coefficients=True,
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
            exposes_coefficients=True,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=True,
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
            'dropout': 0.1, 'epochs': 200, 'batch_size': 256, 'lr': 0.001,
            'weight_decay': 1e-5, 'patience': 30, 'num_layers': 3,
            'layer_width': 128, 'activation': 'relu', 'architecture_pattern': 'constant',
            'use_batchnorm': False, 'lr_scheduler': 'reduce_on_plateau',
            'grad_clip_norm': None, 'loss_function': 'mse'
        },
        hyperparam_schema={
            'num_layers': {'type': 'int', 'min': 1, 'max': 5, 'default': 3, 'help': 'Number of hidden layers'},
            'layer_width': {'type': 'int', 'min': 8, 'max': 512, 'default': 128, 'help': 'Base layer width'},
            'architecture_pattern': {'type': 'select', 'options': ['constant', 'pyramid', 'funnel'], 'default': 'constant', 'help': 'Layer width pattern'},
            'activation': {'type': 'select', 'options': ['relu', 'tanh', 'leaky_relu', 'elu'], 'default': 'relu', 'help': 'Activation function'},
            'use_batchnorm': {'type': 'bool', 'default': False, 'help': 'Batch normalization (stabilizes deeper networks)'},
            'dropout': {'type': 'float', 'min': 0.0, 'max': 0.5, 'default': 0.1, 'help': 'Dropout rate'},
            'epochs': {'type': 'int', 'min': 50, 'max': 500, 'default': 200, 'help': 'Number of epochs'},
            'batch_size': {'type': 'int', 'min': 32, 'max': 512, 'default': 256, 'help': 'Batch size'},
            'lr': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'default': 0.001, 'log': True, 'help': 'Learning rate'},
            'weight_decay': {'type': 'float', 'min': 1e-7, 'max': 1e-2, 'default': 1e-5, 'log': True, 'help': 'L2 regularization (weight decay)'},
            'patience': {'type': 'int', 'min': 5, 'max': 50, 'default': 30, 'help': 'Early stopping patience'},
            'lr_scheduler': {'type': 'select', 'options': ['reduce_on_plateau', 'cosine_warm_restarts', 'one_cycle'], 'default': 'reduce_on_plateau', 'help': 'Learning rate scheduler'},
            'grad_clip_norm': {'type': 'float_or_none', 'min': 0.1, 'max': 10.0, 'default': None, 'help': 'Gradient clipping max norm (None = disabled)'},
            'loss_function': {'type': 'select', 'options': ['mse', 'huber', 'mae', 'weighted_huber'], 'default': 'mse', 'help': 'Loss function for regression (MSE is standard; weighted_huber emphasizes targets near the 90th percentile of the training target)'}
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
            exposes_coefficients=False,
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
            exposes_coefficients=True,
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
            exposes_coefficients=True,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
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
            exposes_coefficients=False,
            notes=['Leaf-wise tree growth (faster)', 'Handles categoricals natively', 'Lower memory usage than XGBoost'],
            supports_class_weight=True
        )
    )

    return registry
