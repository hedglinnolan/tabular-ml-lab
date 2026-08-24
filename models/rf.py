"""Random Forest wrapper."""
import numpy as np
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from models.base import BaseModelWrapper


class RFWrapper(BaseModelWrapper):
    """Wrapper for Random Forest."""
    
    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 10, task_type: str = 'regression'):
        """
        Initialize Random Forest wrapper.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
            task_type: 'regression' or 'classification'
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type
        
        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=42
            )
    
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
