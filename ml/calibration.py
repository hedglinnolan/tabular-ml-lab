"""
Calibration analysis for classification and regression models.

For classification: reliability diagrams, Brier score, ECE, Platt/isotonic recalibration.
For regression: calibration slope, calibration-in-the-large.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    # Common
    model_name: str
    task_type: str

    # Classification
    brier_score: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error
    mce: Optional[float] = None  # Maximum Calibration Error
    bin_edges: Optional[np.ndarray] = None
    bin_true_freq: Optional[np.ndarray] = None
    bin_pred_mean: Optional[np.ndarray] = None
    bin_counts: Optional[np.ndarray] = None

    # Regression
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    calibration_r2: Optional[float] = None


def calibration_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    model_name: str = "model",
) -> CalibrationResult:
    """Compute classification calibration metrics.

    Args:
        y_true: Binary true labels (0/1)
        y_proba: Predicted probabilities for the positive class
        n_bins: Number of bins for the reliability diagram
        model_name: Name of the model

    Returns:
        CalibrationResult with all classification calibration metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    # Handle multi-class probabilities (take positive class)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, -1]

    # Brier score
    brier = np.mean((y_proba - y_true) ** 2)

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_true_freq = np.zeros(n_bins)
    bin_pred_mean = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_true_freq[i] = y_true[mask].mean()
            bin_pred_mean[i] = y_proba[mask].mean()

    # ECE and MCE
    non_empty = bin_counts > 0
    weights = bin_counts[non_empty] / bin_counts.sum()
    abs_diffs = np.abs(bin_true_freq[non_empty] - bin_pred_mean[non_empty])
    ece = np.sum(weights * abs_diffs)
    mce = np.max(abs_diffs) if len(abs_diffs) > 0 else 0.0

    return CalibrationResult(
        model_name=model_name,
        task_type="classification",
        brier_score=brier,
        ece=ece,
        mce=mce,
        bin_edges=bin_edges,
        bin_true_freq=bin_true_freq,
        bin_pred_mean=bin_pred_mean,
        bin_counts=bin_counts,
    )


def calibration_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> CalibrationResult:
    """Compute regression calibration metrics.

    Calibration slope = slope of observed vs predicted regression.
    Perfect calibration: slope=1, intercept=0.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model

    Returns:
        CalibrationResult with regression calibration metrics
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # A diverged model (e.g. a neural net) can emit NaN/inf predictions;
    # LinearRegression rejects them with a page-crashing ValueError. Fit the
    # calibration line on the finite pairs only, and raise a clear error when
    # there is nothing usable — callers surface it as a per-model note.
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(finite.sum()) < 2:
        raise ValueError(
            f"{model_name}: fewer than 2 finite prediction pairs "
            f"({int((~finite).sum())} non-finite of {len(y_pred)}) — "
            "the model's predictions are unusable for calibration."
        )
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    reg = LinearRegression()
    reg.fit(y_pred.reshape(-1, 1), y_true)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(y_true, reg.predict(y_pred.reshape(-1, 1)))

    return CalibrationResult(
        model_name=model_name,
        task_type="regression",
        calibration_slope=slope,
        calibration_intercept=intercept,
        calibration_r2=r2,
    )


def recalibrate_platt(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_proba_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any]:
    """Apply Platt scaling (logistic recalibration).

    Args:
        y_true: Calibration set true labels
        y_proba: Calibration set predicted probabilities
        y_proba_test: Test set probabilities to recalibrate (optional)

    Returns:
        (recalibrated_proba, calibrator) for the test set if provided,
        otherwise for the calibration set
    """
    from sklearn.linear_model import LogisticRegression

    y_proba = np.asarray(y_proba).ravel()
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=10000)
    calibrator.fit(y_proba.reshape(-1, 1), y_true)

    target = y_proba_test if y_proba_test is not None else y_proba
    target = np.asarray(target).ravel()
    recalibrated = calibrator.predict_proba(target.reshape(-1, 1))[:, 1]

    return recalibrated, calibrator


def recalibrate_isotonic(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_proba_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any]:
    """Apply isotonic regression recalibration.

    Args:
        y_true: Calibration set true labels
        y_proba: Calibration set predicted probabilities
        y_proba_test: Test set probabilities to recalibrate (optional)

    Returns:
        (recalibrated_proba, calibrator)
    """
    from sklearn.isotonic import IsotonicRegression

    y_proba = np.asarray(y_proba).ravel()
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_proba, y_true)

    target = y_proba_test if y_proba_test is not None else y_proba
    target = np.asarray(target).ravel()
    recalibrated = calibrator.predict(target)

    return recalibrated, calibrator


def plot_calibration_curve(cal_result: CalibrationResult):
    """Generate a Plotly calibration/reliability diagram.

    Returns a Plotly figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', line=dict(dash='dash', color='gray'),
        name='Perfect calibration',
        showlegend=True,
    ))

    # Model calibration
    non_empty = cal_result.bin_counts > 0
    fig.add_trace(go.Scatter(
        x=cal_result.bin_pred_mean[non_empty],
        y=cal_result.bin_true_freq[non_empty],
        mode='lines+markers',
        name=cal_result.model_name,
        marker=dict(size=8),
    ))

    # Add bin count annotations
    fig.add_trace(go.Bar(
        x=cal_result.bin_pred_mean[non_empty],
        y=cal_result.bin_counts[non_empty] / cal_result.bin_counts.sum(),
        name='Fraction per bin',
        opacity=0.3,
        yaxis='y2',
    ))

    fig.update_layout(
        title=f"Calibration Plot — {cal_result.model_name}",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        yaxis2=dict(title="Fraction of samples", overlaying='y', side='right', range=[0, 1]),
        legend=dict(x=0.02, y=0.98),
        annotations=[
            dict(
                text=f"Brier: {cal_result.brier_score:.4f} | ECE: {cal_result.ece:.4f}",
                xref="paper", yref="paper", x=0.98, y=0.02,
                showarrow=False, font=dict(size=11),
            )
        ],
    )

    return fig


def decision_curve_analysis(
    y_true: np.ndarray,
    y_proba_dict: Dict[str, np.ndarray],
    thresholds: Optional[np.ndarray] = None,
):
    """Decision Curve Analysis: net benefit at various probability thresholds.

    Args:
        y_true: Binary true labels (0/1)
        y_proba_dict: {model_name: predicted_probabilities}
        thresholds: Probability thresholds to evaluate (default: 0.01 to 0.99)

    Returns:
        Plotly figure with net benefit curves
    """
    import plotly.graph_objects as go

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()

    fig = go.Figure()

    # Treat All strategy
    treat_all_nb = []
    for pt in thresholds:
        nb = prevalence - (1 - prevalence) * pt / (1 - pt) if pt < 1 else 0
        treat_all_nb.append(nb)

    fig.add_trace(go.Scatter(
        x=thresholds, y=treat_all_nb,
        mode='lines', line=dict(dash='dash', color='gray'),
        name='Treat All',
    ))

    # Treat None (always 0)
    fig.add_trace(go.Scatter(
        x=thresholds, y=[0] * len(thresholds),
        mode='lines', line=dict(dash='dot', color='lightgray'),
        name='Treat None',
    ))

    # Model net benefits
    for model_name, y_proba in y_proba_dict.items():
        y_proba = np.asarray(y_proba).ravel()
        net_benefits = []
        for pt in thresholds:
            pred_pos = y_proba >= pt
            tp = np.sum((pred_pos) & (y_true == 1))
            fp = np.sum((pred_pos) & (y_true == 0))
            nb = tp / n - fp / n * pt / (1 - pt) if pt < 1 else 0
            net_benefits.append(nb)

        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefits,
            mode='lines', name=model_name,
        ))

    fig.update_layout(
        title="Decision Curve Analysis",
        xaxis_title="Threshold Probability",
        yaxis_title="Net Benefit",
        yaxis=dict(range=[-0.05, max(0.5, prevalence + 0.1)]),
        legend=dict(x=0.7, y=0.98),
    )
    fig.add_annotation(
        text="Higher net benefit = more clinical utility at that threshold",
        xref="paper", yref="paper", x=0.5, y=-0.12,
        showarrow=False, font=dict(size=10, color="gray"),
    )

    return fig
