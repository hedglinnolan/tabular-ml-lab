"""Random Forest wrapper."""
import numpy as np
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from models.base import BaseModelWrapper


# The constructor arguments that are ALSO constructor arguments of the forest
# this wrapper holds. `set_params` forwards these so the wrapper and its forest
# cannot disagree; `task_type` is not here because changing it means a
# different estimator class, which is a rebuild rather than a forward.
_FOREST_PARAMS = ('n_estimators', 'max_depth', 'min_samples_leaf', 'n_jobs')


class RFWrapper(BaseModelWrapper):
    """Wrapper for Random Forest.

    `n_jobs` is a constructor argument rather than a literal inside
    `__init__`, and that is the whole reason this class has a parameter
    surface worth reading. scikit-learn's `clone()` rebuilds an estimator from
    `get_params()`, and `get_params()` is the `__init__` signature — so a
    value that was hardcoded in the body was rebuilt hardcoded by every clone,
    and `set_params(n_jobs=1)` on the wrapper set an attribute nothing read.
    Both pinning routes ml/eval.py describes failed silently on this one
    class.

    The default is still `-1`, every core, on the measured trade recorded in
    `ml/eval.py::_inner_thread_overrides`: five CV folds cannot saturate eight
    cores, so the forest's own pool fills the idle ones, and pinning it cost
    ~18% of the wall clock for 3.6% of the memory. What this change buys is
    that the choice can now be MADE — by a caller, by a clone, by a future
    cost model — instead of being unreachable. And it moves no result: the
    forest is byte-identical at any `n_jobs` for a fixed seed (every tree's
    split arrays compare equal), which
    tests/test_five_folds_do_not_each_take_the_whole_box.py now asserts
    rather than cites.
    """

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 10, task_type: str = 'regression',
                 n_jobs: int = -1):
        """
        Initialize Random Forest wrapper.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
            task_type: 'regression' or 'classification'
            n_jobs: Threads the forest may use; -1 is every core
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.model = self._build_forest()

    def _build_forest(self):
        """A fresh, unfitted forest from the wrapper's current parameters."""
        forest_cls = (RandomForestRegressor if self.task_type == 'regression'
                      else RandomForestClassifier)
        return forest_cls(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=42
        )

    def set_params(self, **params):
        """`set_params` that reaches the forest, not just the wrapper.

        `BaseEstimator.set_params` sets attributes on the wrapper and stops
        there, so the forest built in `__init__` kept whatever it was built
        with. Forwarded here for the parameters the two share; a new
        `task_type` is a different estimator class and rebuilds an unfitted
        one.
        """
        super().set_params(**params)
        if 'task_type' in params:
            self.model = self._build_forest()
            self.is_fitted = False
        else:
            forwarded = {k: getattr(self, k) for k in _FOREST_PARAMS if k in params}
            if forwarded:
                self.model.set_params(**forwarded)
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate validation metrics if available
        val_rmse = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            if self.task_type == 'regression':
                val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
            else:
                # For classification, use accuracy
                val_rmse = np.mean(y_val_pred == y_val)
        
        return {
            'history': {'val_rmse': [val_rmse] if val_rmse is not None else []},
            'best_val_rmse': val_rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return self.task_type == 'classification'

    # `L64-A5`. A SECOND, BYTE-IDENTICAL COPY of `predict_proba` and
    # `supports_proba` stood here and was deleted. The duplication was inert —
    # the two copies matched character for character, so the later binding
    # shadowed the earlier with the same behavior — but editing the first copy
    # would have had no effect, which is the trap: a reader fixes the method
    # they find and the class keeps the one they did not.
