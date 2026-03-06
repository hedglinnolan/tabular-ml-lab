"""
Computational resource configuration for Tabular ML Lab.

Adjust these limits based on available hardware:
- GTX 1080 Ti (11GB VRAM): Use STANDARD profile
- A6000 (48GB VRAM): Use HIGH_PERFORMANCE profile
- Multi-GPU cluster: Use ENTERPRISE profile

Set via environment variable: COMPUTE_PROFILE=enterprise
"""
import os
from typing import Dict, Any


class ComputeProfile:
    """Hardware-appropriate computational limits."""
    
    # STANDARD profile: GTX 1080 Ti / consumer hardware
    STANDARD = {
        "pdp_max_samples": 2000,          # Partial dependence plot subsampling
        "shap_kernel_max_eval": 50,       # KernelExplainer evaluation samples
        "shap_kernel_max_background": 50, # KernelExplainer background samples
        "optuna_trials": 30,              # Hyperparameter optimization trials per model
        "bootstrap_resamples": 1000,      # Bootstrap CI resamples (standard)
        "seed_sensitivity_samples": 10,   # Number of random seeds to test
        "enable_nn_seed_sensitivity": False,  # PyTorch models don't support sklearn clone
    }
    
    # HIGH_PERFORMANCE profile: Single A6000 / professional GPU
    HIGH_PERFORMANCE = {
        "pdp_max_samples": 10000,         # 5x more PDP samples
        "shap_kernel_max_eval": 200,      # 4x more SHAP evaluations
        "shap_kernel_max_background": 100,
        "optuna_trials": 50,              # 67% more hyperparameter trials
        "bootstrap_resamples": 1000,      # Keep standard (1000 is publication-grade)
        "seed_sensitivity_samples": 20,   # 2x more seed tests
        "enable_nn_seed_sensitivity": False,  # Still sklearn limitation
    }
    
    # ENTERPRISE profile: Multi-GPU cluster with massive RAM
    ENTERPRISE = {
        "pdp_max_samples": 50000,         # Full dataset or 50k cap
        "shap_kernel_max_eval": 500,      # 10x more SHAP evaluations
        "shap_kernel_max_background": 200,
        "optuna_trials": 100,             # Extensive hyperparameter search
        "bootstrap_resamples": 2000,      # Double bootstrap for tighter CIs
        "seed_sensitivity_samples": 50,   # Comprehensive seed robustness
        "enable_nn_seed_sensitivity": False,  # Still sklearn limitation
    }
    
    @classmethod
    def get_profile(cls, profile_name: str = None) -> Dict[str, Any]:
        """
        Get compute profile from environment or explicit name.
        
        Args:
            profile_name: "standard", "high_performance", or "enterprise"
                         If None, reads from COMPUTE_PROFILE env var
        
        Returns:
            Dictionary of computational limits
        """
        if profile_name is None:
            profile_name = os.getenv("COMPUTE_PROFILE", "standard")
        
        profile_name = profile_name.upper()
        
        if hasattr(cls, profile_name):
            return getattr(cls, profile_name)
        else:
            # Default to standard if unknown profile
            return cls.STANDARD


def get_compute_limits() -> Dict[str, Any]:
    """
    Get current computational limits based on environment.
    
    Returns:
        Dictionary with keys: pdp_max_samples, shap_kernel_max_eval, etc.
    """
    return ComputeProfile.get_profile()


def get_limit(key: str, default: Any = None) -> Any:
    """
    Get a specific computational limit.
    
    Args:
        key: Limit name (e.g., "pdp_max_samples")
        default: Fallback value if key not found
    
    Returns:
        Limit value
    """
    limits = get_compute_limits()
    return limits.get(key, default)
