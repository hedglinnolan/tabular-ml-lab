"""
Neural Network wrapper using weighted Huber loss (regression) or BCE/CE loss (classification).
Wraps the existing NN training implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError

from models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Copy SimpleMLP and weighted_huber_loss from existing models.py
# This wraps the existing implementation cleanly
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simplified MLP for regression and classification."""
    
    def __init__(self, input_dim: int, hidden: list = [32, 32], dropout: float = 0.1, 
                 output_dim: int = 1, activation: str = 'relu'):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden: List of hidden layer sizes
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression/binary, n_classes for multiclass)
            activation: Activation function ('relu', 'tanh', 'leaky_relu', 'elu')
        """
        super().__init__()
        
        # Select activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU())
        
        layers = []
        prev_dim = input_dim
        
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.output_dim = output_dim
        self.activation_name = activation
    
    def forward(self, x):
        return self.layers(x)


def weighted_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, 
                       t0: float = 180.0, s: float = 20.0, alpha: float = 2.5) -> torch.Tensor:
    """Weighted Huber loss focusing on high values (from existing models.py)."""
    errors = y_true - y_pred
    abs_errors = torch.abs(errors)
    
    # Huber loss component
    delta = 1.0
    huber_loss = torch.where(
        abs_errors <= delta,
        0.5 * errors ** 2,
        delta * abs_errors - 0.5 * delta ** 2
    )
    
    # Weight based on target value
    w = 1.0 + alpha * torch.exp(-((y_true - t0) / s) ** 2)
    w = torch.clamp(w, min=0.1, max=10.0)
    
    return (w * huber_loss).mean()


class SklearnCompatibleNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for PyTorch NN model (regression)."""
    
    def __init__(self, wrapper_instance=None):
        """
        Initialize with NN wrapper instance.
        
        Args:
            wrapper_instance: NNWeightedHuberWrapper instance (can be None initially)
        """
        self.wrapper_instance = wrapper_instance
        self.task_type = 'regression'
        self.is_fitted_ = False
        self.n_features_in_ = None
    
    def _check_is_fitted(self):
        """Check if estimator is fitted."""
        if not self.is_fitted_:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
    
    def fit(self, X, y):
        """
        Fit the model (sklearn interface).
        
        Note: The actual training is done by wrapper_instance.fit() before this is called.
        This method just marks the estimator as fitted and sets required attributes.
        """
        if self.wrapper_instance is None:
            raise ValueError("wrapper_instance must be set before calling fit()")
        
        # Mark as fitted
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self
    
    def predict(self, X):
        """Predict (sklearn interface)."""
        self._check_is_fitted()
        if self.wrapper_instance is None:
            raise ValueError("wrapper_instance must be set")
        return self.wrapper_instance.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters (sklearn interface)."""
        return {
            'wrapper_instance': self.wrapper_instance
        }
    
    def set_params(self, **params):
        """Set parameters (sklearn interface). Only wrapper_instance is allowed."""
        if "wrapper_instance" in params:
            self.wrapper_instance = params["wrapper_instance"]
        return self


class SklearnCompatibleNNClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for PyTorch NN model (classification)."""
    
    def __init__(self, wrapper_instance=None):
        """
        Initialize with NN wrapper instance.
        
        Args:
            wrapper_instance: NNWeightedHuberWrapper instance (can be None initially)
        """
        self.wrapper_instance = wrapper_instance
        self.task_type = 'classification'
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
    
    def _check_is_fitted(self):
        """Check if estimator is fitted."""
        if not self.is_fitted_:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
    
    def fit(self, X, y):
        """
        Fit the model (sklearn interface).
        
        Note: The actual training is done by wrapper_instance.fit() before this is called.
        This method just marks the estimator as fitted and sets required attributes.
        """
        if self.wrapper_instance is None:
            raise ValueError("wrapper_instance must be set before calling fit()")
        
        # Mark as fitted
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """Predict (sklearn interface)."""
        self._check_is_fitted()
        if self.wrapper_instance is None:
            raise ValueError("wrapper_instance must be set")
        return self.wrapper_instance.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)."""
        self._check_is_fitted()
        if self.wrapper_instance is None:
            raise ValueError("wrapper_instance must be set")
        return self.wrapper_instance.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters (sklearn interface)."""
        return {
            'wrapper_instance': self.wrapper_instance
        }
    
    def set_params(self, **params):
        """Set parameters (sklearn interface). Only wrapper_instance is allowed."""
        if "wrapper_instance" in params:
            self.wrapper_instance = params["wrapper_instance"]
        return self


# Factory function to create appropriate sklearn-compatible wrapper
def SklearnCompatibleNN(wrapper_instance=None, task_type='regression'):
    """Factory function to create appropriate sklearn-compatible wrapper."""
    if task_type == 'classification':
        return SklearnCompatibleNNClassifier(wrapper_instance)
    else:
        return SklearnCompatibleNNRegressor(wrapper_instance)


class NNWeightedHuberWrapper(BaseModelWrapper):
    """Wrapper for Neural Network with weighted Huber loss (regression) or BCE/CE loss (classification)."""
    
    # Available loss functions for regression
    REGRESSION_LOSSES = {
        'mse': 'Mean Squared Error (standard)',
        'huber': 'Huber Loss (robust to outliers)',
        'weighted_huber': 'Weighted Huber (emphasizes high-value targets, e.g., glucose)',
        'mae': 'Mean Absolute Error (robust)',
    }

    def __init__(self, hidden_layers: List[int] = None, dropout: float = 0.1, 
                 task_type: str = 'regression', activation: str = 'relu',
                 loss_function: str = 'mse'):
        """
        Initialize NN wrapper.
        
        Args:
            hidden_layers: List of hidden layer sizes (default: [32, 32])
            dropout: Dropout rate
            task_type: 'regression' or 'classification'
            activation: Activation function ('relu', 'tanh', 'leaky_relu', 'elu')
            loss_function: For regression: 'mse' (default), 'huber', 'weighted_huber', 'mae'
                          For classification: automatically set to BCE/CE
        """
        super().__init__("Neural Network")
        self.hidden_layers = hidden_layers or [32, 32]
        self.dropout = dropout
        self.task_type = task_type
        self.activation = activation
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = None
        self._sklearn_estimator = None  # Lazy initialization
        self.classes_ = None  # For classification
        self.n_classes_ = None  # For classification
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get a summary of the neural network architecture for reporting."""
        return {
            'hidden_layers': self.hidden_layers,
            'num_layers': len(self.hidden_layers),
            'dropout': self.dropout,
            'activation': self.activation,
            'total_params': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else None
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 256,
            lr: float = 0.0015,
            weight_decay: float = 0.0002,
            patience: int = 30,
            progress_callback: Optional[callable] = None,
            random_seed: Optional[int] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            patience: Early stopping patience
            progress_callback: Optional callback function(epoch, train_loss, val_loss, val_metric)
            random_seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary with training history
        """
        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
        
        # Handle classification vs regression
        if self.task_type == 'classification':
            # Get unique classes
            self.classes_ = np.unique(y_train)
            self.n_classes_ = len(self.classes_)
            
            # Map classes to 0, 1, 2, ... for training
            class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
            y_train_mapped = np.array([class_to_idx[cls] for cls in y_train])
            
            # Determine output dimension: binary = 1 logit, multiclass = n_classes logits
            output_dim = 1 if self.n_classes_ == 2 else self.n_classes_
            
            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            y_train_t = torch.LongTensor(y_train_mapped).to(self.device)
            
            if X_val is not None and y_val is not None:
                y_val_mapped = np.array([class_to_idx.get(cls, 0) for cls in y_val])
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                y_val_t = torch.LongTensor(y_val_mapped).to(self.device)
            else:
                X_val_t = None
                y_val_t = None
            
            # Loss function
            if self.n_classes_ == 2:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            # Regression
            output_dim = 1
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            y_train_t = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            
            if X_val is not None and y_val is not None:
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            else:
                X_val_t = None
                y_val_t = None
            
            # Select regression loss function
            if self.loss_function == 'mse':
                criterion = nn.MSELoss()
            elif self.loss_function == 'huber':
                criterion = nn.HuberLoss(delta=1.0)
            elif self.loss_function == 'mae':
                criterion = nn.L1Loss()
            elif self.loss_function == 'weighted_huber':
                criterion = None  # Use weighted_huber_loss function
            else:
                criterion = nn.MSELoss()  # Default to MSE
        
        # Create model
        self.model = SimpleMLP(
            input_dim=X_train.shape[1],
            hidden=self.hidden_layers,
            dropout=self.dropout,
            output_dim=output_dim,
            activation=self.activation
        )
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        if X_val_t is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
            )
        
        # Training loop
        best_val_metric = float('inf') if self.task_type == 'regression' else 0.0
        best_model_state = self.model.state_dict().copy()
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [] if self.task_type == 'regression' else [],
            'val_accuracy': [] if self.task_type == 'classification' else []
        }
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                
                if self.task_type == 'classification':
                    if self.n_classes_ == 2:
                        # Binary: squeeze to match target shape
                        loss = criterion(y_pred.squeeze(), y_batch.float())
                    else:
                        # Multiclass
                        loss = criterion(y_pred, y_batch)
                else:
                    # Regression
                    if criterion is not None:
                        loss = criterion(y_pred, y_batch)
                    else:
                        loss = weighted_huber_loss(y_pred, y_batch)
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_t)
                    
                    if self.task_type == 'classification':
                        val_loss = criterion(y_val_pred.squeeze() if self.n_classes_ == 2 else y_val_pred, 
                                            y_val_t.float() if self.n_classes_ == 2 else y_val_t)
                        # Calculate accuracy
                        if self.n_classes_ == 2:
                            val_pred_labels = (torch.sigmoid(y_val_pred.squeeze()) > 0.5).long()
                        else:
                            val_pred_labels = torch.argmax(y_val_pred, dim=1)
                        val_accuracy = (val_pred_labels == y_val_t).float().mean().item()
                        val_metric = val_accuracy
                    else:
                        if criterion is not None:
                            val_loss = criterion(y_val_pred, y_val_t)
                        else:
                            val_loss = weighted_huber_loss(y_val_pred, y_val_t)
                        val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val_t) ** 2)).item()
                        val_metric = val_rmse
                
                history['val_loss'].append(val_loss.item())
                if self.task_type == 'regression':
                    history['val_rmse'].append(val_rmse)
                else:
                    history['val_accuracy'].append(val_accuracy)
                
                # Progress callback
                if progress_callback:
                    if self.task_type == 'regression':
                        progress_callback(epoch + 1, train_loss, val_loss.item(), val_rmse)
                    else:
                        progress_callback(epoch + 1, train_loss, val_loss.item(), val_accuracy)
                
                # Learning rate scheduling
                if self.task_type == 'regression':
                    scheduler.step(val_rmse)
                else:
                    scheduler.step(1.0 - val_accuracy)  # Step on negative accuracy (higher is better)
                
                # Early stopping
                if self.task_type == 'regression':
                    is_better = val_rmse < best_val_metric - 0.0001
                else:
                    is_better = val_accuracy > best_val_metric + 0.0001
                
                if is_better:
                    best_val_metric = val_metric
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                # No validation set
                if progress_callback:
                    progress_callback(epoch + 1, train_loss, train_loss, 0.0)
        
        # Load best model if validation was used
        if X_val_t is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        self.history = history
        
        return {
            'history': history,
            'best_val_rmse': best_val_metric if self.task_type == 'regression' and X_val_t is not None else None,
            'best_val_accuracy': best_val_metric if self.task_type == 'classification' and X_val_t is not None else None
        }
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history (for plotting learning curves)."""
        return self.history
    
    def get_sklearn_estimator(self):
        """Get sklearn-compatible estimator wrapper."""
        if self._sklearn_estimator is None:
            self._sklearn_estimator = SklearnCompatibleNN(wrapper_instance=self, task_type=self.task_type)
            # Mark as fitted and set attributes since model is already trained
            # Note: fit() will be called later with actual data to set n_features_in_ and classes_ properly
            self._sklearn_estimator.is_fitted_ = True
            # Set n_features_in_ from model if available
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                if hasattr(self.model.layers[0], 'in_features'):
                    self._sklearn_estimator.n_features_in_ = self.model.layers[0].in_features
            # Set classes_ for classification
            if self.task_type == 'classification' and self.classes_ is not None:
                self._sklearn_estimator.classes_ = self.classes_
        return self._sklearn_estimator
    
    def get_model(self) -> Any:
        """Get the underlying model object (sklearn-compatible wrapper for explainability)."""
        # Return sklearn-compatible wrapper for sklearn functions
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.get_sklearn_estimator()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            
            if self.task_type == 'classification':
                if self.n_classes_ == 2:
                    # Binary: sigmoid + threshold
                    probs = torch.sigmoid(logits.squeeze())
                    pred_labels = (probs > 0.5).long().cpu().numpy()
                else:
                    # Multiclass: argmax
                    pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Map back to original class labels
                if self.classes_ is not None:
                    pred_labels = self.classes_[pred_labels]
                
                return pred_labels
            else:
                # Regression
                return logits.cpu().numpy().flatten()
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.task_type != 'classification':
            return None
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            
            if self.n_classes_ == 2:
                # Binary: sigmoid for positive class
                prob_positive = torch.sigmoid(logits.squeeze()).cpu().numpy()
                # Return shape (n_samples, 2)
                probs = np.column_stack([1 - prob_positive, prob_positive])
            else:
                # Multiclass: softmax
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            return probs
    
    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return self.task_type == 'classification'
