"""
Data-driven Neural Network configuration recommender.

Uses dataset characteristics (sample size, feature count, target distribution,
data sufficiency) to recommend NN hyperparameters with visible reasoning.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math

from ml.dataset_profile import DataSufficiencyLevel, TargetProfile


@dataclass
class NNRecommendation:
    """Recommended NN configuration with reasoning for each parameter."""
    params: Dict[str, Any]
    reasoning: Dict[str, str]
    config_source: str = "recommended"


def recommend_nn_config(
    n_samples: int,
    n_features: int,
    target_profile: Optional[TargetProfile] = None,
    data_sufficiency: Optional[DataSufficiencyLevel] = DataSufficiencyLevel.ADEQUATE,
    p_n_ratio: float = 0.1,
    task_type: str = "regression",
    has_engineered_interactions: bool = False,
) -> NNRecommendation:
    """
    Generate a data-driven NN configuration recommendation.

    Args:
        n_samples: Number of training samples.
        n_features: Number of input features.
        target_profile: Target variable profile from EDA (skewness, outliers, etc.).
        data_sufficiency: Data sufficiency level from dataset profile. `None`
            means the caller has no profile, NOT that the data are scarce — it
            is derived from `n_samples`/`n_features` here (`MISC-105`).
        p_n_ratio: Feature-to-sample ratio (p/n).
        task_type: 'regression' or 'classification'.
        has_engineered_interactions: Whether interaction features were already created.

    Returns:
        NNRecommendation with params dict and reasoning dict.
    """
    params: Dict[str, Any] = {}
    reasoning: Dict[str, str] = {}
    ratio = n_samples / max(n_features, 1)

    # `MISC-105`. A `None` sufficiency is "the caller has not profiled this
    # dataset", and every membership test below reads it as neither abundant nor
    # scarce — so a 50,000-row study fell to `reduce_on_plateau`, the schedule
    # meant for small data, against this module's own rule. Derived from the
    # sizes the caller DID supply, through the same function the profile uses,
    # so an unprofiled dataset and a profiled one of the same shape agree.
    if data_sufficiency is None:
        from ml.dataset_profile import assess_data_sufficiency

        data_sufficiency, _ = assess_data_sufficiency(
            n_samples, n_features, task_type)

    # --- Depth (num_layers) ---
    if has_engineered_interactions or n_features < 10:
        num_layers = 2
        depth_reason = "shallow — interactions already explicit" if has_engineered_interactions else "shallow — few features"
    elif n_features > 50 or data_sufficiency in (DataSufficiencyLevel.ABUNDANT,):
        num_layers = 4
        depth_reason = "deeper — many features benefit from learned representations"
    else:
        num_layers = 3
        depth_reason = "moderate depth for typical tabular data"
    # Cap depth when data is scarce
    if data_sufficiency in (DataSufficiencyLevel.SCARCE, DataSufficiencyLevel.CRITICAL):
        num_layers = min(num_layers, 2)
        depth_reason += " (capped — limited data)"
    params["num_layers"] = num_layers
    reasoning["num_layers"] = f"{num_layers} layers — {depth_reason}"

    # --- Width (layer_width) ---
    base_width = min(4 * n_features, 512)
    # Only cap width when data is actually scarce
    if data_sufficiency in (DataSufficiencyLevel.SCARCE, DataSufficiencyLevel.CRITICAL):
        capacity_cap = max(32, n_samples // (20 * num_layers))
        layer_width = min(base_width, capacity_cap)
    elif data_sufficiency == DataSufficiencyLevel.LIMITED:
        capacity_cap = max(64, n_samples // (10 * num_layers))
        layer_width = min(base_width, capacity_cap)
    else:
        layer_width = base_width
    # Round to nearest power of 2 for efficiency
    layer_width = max(32, 2 ** round(math.log2(max(1, layer_width))))
    layer_width = min(layer_width, 512)
    params["layer_width"] = layer_width
    reasoning["layer_width"] = (
        f"{layer_width} units — scaled from {n_features} features "
        f"(sample:feature ratio {ratio:.0f}:1)"
    )

    # --- Architecture pattern ---
    if n_features > n_samples / 100:
        pattern = "funnel"
        pattern_reason = "compress high-dimensional input"
    else:
        pattern = "constant"
        pattern_reason = "uniform width — sufficient data per feature"
    params["architecture_pattern"] = pattern
    reasoning["architecture_pattern"] = f"{pattern} — {pattern_reason}"

    # --- Regularization (dropout + weight_decay) ---
    if ratio > 100:
        dropout = 0.05
        weight_decay = 1e-6
        reg_reason = "minimal — high sample:feature ratio"
    elif ratio > 20:
        dropout = 0.1
        weight_decay = 1e-5
        reg_reason = "moderate — adequate data"
    elif ratio > 5:
        dropout = 0.2
        weight_decay = 1e-4
        reg_reason = "increased — limited data per feature"
    else:
        dropout = 0.3
        weight_decay = 1e-3
        reg_reason = "aggressive — data-scarce regime"
    params["dropout"] = dropout
    params["weight_decay"] = weight_decay
    reasoning["dropout"] = f"{dropout} — {reg_reason}"
    reasoning["weight_decay"] = f"{weight_decay:.0e} — {reg_reason}"

    # --- Learning rate ---
    if num_layers >= 4 or layer_width >= 256:
        lr = 0.0003
        lr_reason = "lower for deep/wide architecture"
    elif num_layers >= 3:
        lr = 0.0005
        lr_reason = "reduced for 3-layer network"
    else:
        lr = 0.001
        lr_reason = "standard Adam default for shallow network"
    params["lr"] = lr
    reasoning["lr"] = f"{lr} — {lr_reason}"

    # --- Batch size ---
    if n_samples < 500:
        batch_size = 32
    elif n_samples < 2000:
        batch_size = 64
    elif n_samples < 10000:
        batch_size = 128
    else:
        batch_size = 256
    params["batch_size"] = batch_size
    reasoning["batch_size"] = f"{batch_size} — scaled to {n_samples:,} training samples"

    # --- Loss function (regression only) ---
    if task_type == "regression" and target_profile is not None:
        skew = abs(target_profile.skewness) if target_profile.skewness is not None else 0
        outlier_rate = target_profile.outlier_rate if target_profile.outlier_rate is not None else 0
        if outlier_rate > 0.1:
            loss_fn = "weighted_huber"
            loss_reason = f"robust to high outlier rate ({outlier_rate:.0%})"
        elif skew > 1.5:
            loss_fn = "huber"
            loss_reason = f"robust to skewed target (skew={skew:.2f})"
        else:
            loss_fn = "mse"
            loss_reason = "standard — target distribution is well-behaved"
    else:
        loss_fn = "mse"
        loss_reason = "standard (classification uses BCE/CE automatically)"
    params["loss_function"] = loss_fn
    reasoning["loss_function"] = f"{loss_fn} — {loss_reason}"

    # --- BatchNorm ---
    use_batchnorm = num_layers >= 3 and n_samples >= 100
    params["use_batchnorm"] = use_batchnorm
    if use_batchnorm:
        reasoning["use_batchnorm"] = "enabled — stabilizes training for deeper networks"
    else:
        reasoning["use_batchnorm"] = "disabled — not needed for shallow architecture"

    # --- LR scheduler ---
    if n_samples >= 5000 and data_sufficiency in (DataSufficiencyLevel.ABUNDANT, DataSufficiencyLevel.ADEQUATE):
        lr_sched = "cosine_warm_restarts"
        sched_reason = "cosine annealing — sufficient data for smooth schedule"
    else:
        lr_sched = "reduce_on_plateau"
        sched_reason = "adaptive — adjusts based on validation performance"
    params["lr_scheduler"] = lr_sched
    reasoning["lr_scheduler"] = f"{lr_sched} — {sched_reason}"

    # --- Gradient clipping ---
    if loss_fn == "weighted_huber":
        grad_clip = 1.0
        clip_reason = "enabled — prevents gradient explosion with weighted loss"
    else:
        grad_clip = None
        clip_reason = "disabled — standard loss is stable"
    params["grad_clip_norm"] = grad_clip
    reasoning["grad_clip_norm"] = clip_reason

    # --- Remaining defaults ---
    params["activation"] = "relu"
    reasoning["activation"] = "relu — standard default for tabular data"
    params["epochs"] = 200
    reasoning["epochs"] = "200 with early stopping"
    params["patience"] = 30
    reasoning["patience"] = "30 epochs — balanced between convergence and overfitting"

    return NNRecommendation(params=params, reasoning=reasoning)
