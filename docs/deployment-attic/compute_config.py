"""
Compute profile configuration for institutional deployment.

Provides hardware-aware defaults for compute-intensive operations so that
university admins can tune the app for their infrastructure without modifying
application code.

Profiles:
  standard         — Typical university VM (4-8 GB RAM, no GPU)
  high_performance — Departmental server (16-32 GB RAM, optional GPU)
  enterprise       — Research cluster (64+ GB RAM, GPU)

Configuration via environment variable:
  COMPUTE_PROFILE = standard | high_performance | enterprise

Usage in page code:
  from utils.compute_config import get_limit
  shap_bg = get_limit("shap_background")  # returns profile-appropriate value
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ComputeProfile:
    """Hardware-aware limits for compute-intensive operations."""
    name: str
    description: str
    limits: Dict[str, int] = field(default_factory=dict)


# ── Profile Definitions ──────────────────────────────────────────

PROFILES: Dict[str, ComputeProfile] = {
    "standard": ComputeProfile(
        name="standard",
        description="University VM (4-8 GB RAM, no GPU)",
        limits={
            # Explainability
            "shap_background": 50,
            "shap_eval_size": 100,
            "perm_repeats": 5,
            "pdp_grid_resolution": 20,
            # Feature selection
            "stability_bootstrap": 50,
            "rfe_cv_folds": 3,
            # Training
            "optuna_trials": 15,
            "bootstrap_resamples": 500,
            "cv_folds": 5,
            "nn_max_epochs": 100,
            # Sensitivity
            "sensitivity_seeds": 5,
            "dropout_repeats": 3,
        },
    ),
    "high_performance": ComputeProfile(
        name="high_performance",
        description="Departmental server (16-32 GB RAM, optional GPU)",
        limits={
            "shap_background": 100,
            "shap_eval_size": 200,
            "perm_repeats": 10,
            "pdp_grid_resolution": 50,
            "stability_bootstrap": 100,
            "rfe_cv_folds": 5,
            "optuna_trials": 30,
            "bootstrap_resamples": 1000,
            "cv_folds": 5,
            "nn_max_epochs": 200,
            "sensitivity_seeds": 10,
            "dropout_repeats": 5,
        },
    ),
    "enterprise": ComputeProfile(
        name="enterprise",
        description="Research cluster (64+ GB RAM, GPU)",
        limits={
            "shap_background": 200,
            "shap_eval_size": 500,
            "perm_repeats": 20,
            "pdp_grid_resolution": 100,
            "stability_bootstrap": 200,
            "rfe_cv_folds": 10,
            "optuna_trials": 100,
            "bootstrap_resamples": 2000,
            "cv_folds": 10,
            "nn_max_epochs": 500,
            "sensitivity_seeds": 20,
            "dropout_repeats": 10,
        },
    ),
}


def get_profile() -> ComputeProfile:
    """Return the active compute profile from environment."""
    name = os.environ.get("COMPUTE_PROFILE", "high_performance").lower().strip()
    profile = PROFILES.get(name)
    if profile is None:
        logger.warning(f"Unknown COMPUTE_PROFILE={name!r}, falling back to high_performance")
        profile = PROFILES["high_performance"]
    return profile


def get_limit(key: str, default: int = 0) -> int:
    """Return a specific compute limit from the active profile.

    Usage:
        from utils.compute_config import get_limit
        n_repeats = get_limit("perm_repeats", default=10)
    """
    return get_profile().limits.get(key, default)
