"""
Wrapper adapters for registry models to work with existing training infrastructure.
"""
import numpy as np
from typing import Dict, Optional, Any
from models.base import BaseModelWrapper


class RegistryModelWrapper(BaseModelWrapper):
    """
    Generic wrapper for sklearn estimators from the registry.
    Works with any sklearn-compatible estimator.
    """
    
    def __init__(self, estimator, name: str):
        """
        Initialize wrapper.
        
        Args:
            estimator: sklearn-compatible estimator
            name: Display name
        """
        super().__init__(name)
        self.model = estimator
        self.estimator = estimator  # Alias for compatibility
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model."""
        sample_weight = kwargs.get('sample_weight', None)
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate validation metrics if available
        val_metric = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            # Determine task type from y
            if len(np.unique(y_train)) < 20 and y_train.dtype in [np.int64, np.int32, 'int64', 'int32', 'int']:
                # Classification
                val_metric = np.mean(y_val_pred == y_val)  # Accuracy
            else:
                # Regression
                val_metric = np.sqrt(np.mean((y_val_pred - y_val) ** 2))  # RMSE
        
        return {
            'history': {'val_metric': [val_metric] if val_metric is not None else []},
            'best_val_metric': val_metric
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities (if available)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return hasattr(self.model, 'predict_proba')
    
    def get_model(self):
        """Get the underlying sklearn estimator."""
        return self.model
