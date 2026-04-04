"""
Evaluation utilities: metrics, cross-validation, residual analysis.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, log_loss,
    average_precision_score, confusion_matrix
)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Returns:
        Dictionary with MAE, RMSE, R2, MedianAE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MedianAE': float(median_ae)
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with Accuracy, F1, ROC-AUC (if probas), LogLoss, PR-AUC
    """
    metrics = {}
    
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['F1'] = float(f1_score(y_true, y_pred, average='weighted'))
    
    if y_proba is not None:
        try:
            # ROC-AUC (binary or multiclass)
            if len(np.unique(y_true)) == 2:
                metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            
            # Log Loss
            metrics['LogLoss'] = float(log_loss(y_true, y_proba))
            
            # PR-AUC
            if len(np.unique(y_true)) == 2:
                metrics['PR-AUC'] = float(average_precision_score(y_true, y_proba[:, 1]))
        except Exception as e:
            # If metrics fail, skip them
            pass
    
    return metrics


def perform_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    task_type: str = 'regression',
    scoring: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model with fit/predict interface
        X: Features
        y: Targets
        cv_folds: Number of folds
        task_type: 'regression' or 'classification'
        scoring: Scoring metric (if None, uses default for task type)
        
    Returns:
        Dictionary with metric arrays across folds
    """
    if scoring is None:
        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
    
    # Choose CV strategy
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform CV
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Convert to positive if using negative MSE
    if 'neg_' in scoring:
        scores = -scores
    
    return {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'folds': cv_folds
    }


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Analyze residuals for regression models.
    
    Returns:
        Dictionary with residual statistics and arrays
    """
    residuals = y_true - y_pred
    
    return {
        'residuals': residuals,
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'median_residual': float(np.median(residuals))
    }


def analyze_residuals_extended(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Extended residual stats for narrative: skew, IQR, residuals-vs-predicted
    correlation, quantiles.
    """
    from scipy.stats import skew as scipy_skew

    residuals = np.asarray(y_true - y_pred, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(residuals) & np.isfinite(y_pred_arr)
    if valid.sum() < 3:
        return {}

    r = residuals[valid]
    p = y_pred_arr[valid]
    q5, q25, q75, q95 = float(np.percentile(r, 5)), float(np.percentile(r, 25)), float(np.percentile(r, 75)), float(np.percentile(r, 95))
    iqr = float(q75 - q25)
    sk = float(scipy_skew(r)) if len(r) >= 3 else 0.0
    rr = np.corrcoef(r, p)[0, 1] if np.std(r) > 0 and np.std(p) > 0 else 0.0
    resid_vs_pred_corr = float(rr) if not np.isnan(rr) else 0.0

    return {
        'residuals': residuals,
        'mean_residual': float(np.mean(r)),
        'std_residual': float(np.std(r)),
        'min_residual': float(np.min(r)),
        'max_residual': float(np.max(r)),
        'median_residual': float(np.median(r)),
        'skew': sk,
        'iqr': iqr,
        'q5': q5,
        'q25': q25,
        'q75': q75,
        'q95': q95,
        'residual_vs_predicted_corr': resid_vs_pred_corr,
    }


def analyze_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Stats for predictions-vs-actual narrative: correlation, bias by quintile,
    max over/under-prediction.
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if valid.sum() < 3:
        return {}

    yt, yp = y_true_arr[valid], y_pred_arr[valid]
    corr = np.corrcoef(yt, yp)[0, 1] if np.std(yt) > 0 and np.std(yp) > 0 else 0.0
    corr = float(corr) if not np.isnan(corr) else 0.0

    q_edges = np.percentile(yt, [0, 20, 40, 60, 80, 100])
    q_edges[-1] += 1e-9
    bias_by_quintile = []
    for i in range(5):
        mask = (yt >= q_edges[i]) & (yt < q_edges[i + 1])
        if mask.sum() > 0:
            b = float(np.mean(yp[mask] - yt[mask]))
        else:
            b = 0.0
        bias_by_quintile.append(b)

    err = yp - yt
    max_over = float(np.max(err)) if len(err) else 0.0
    max_under = float(np.min(err)) if len(err) else 0.0

    return {
        'correlation': corr,
        'bias_by_quintile': bias_by_quintile,
        'max_overprediction': max_over,
        'max_underprediction': max_under,
        'mean_error': float(np.mean(err)),
    }


def analyze_residuals_stratified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
    custom_edges: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Residual analysis stratified by target-value range.

    Returns per-bin MAE, mean bias (pred − true), RMSE, sample count,
    and a bias_direction label ('over' / 'under' / 'balanced').

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.
    n_bins : int
        Number of equal-frequency bins (ignored when *custom_edges* given).
    custom_edges : list of float, optional
        Explicit bin boundaries.  Must be monotonically increasing and span
        the target range.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(yt) & np.isfinite(yp)
    if valid.sum() < 3:
        return {"bins": [], "overall_bias_direction": "balanced"}

    yt, yp = yt[valid], yp[valid]

    if custom_edges is not None:
        edges = np.array(sorted(custom_edges), dtype=float)
    else:
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(yt, percentiles)
        # deduplicate edges that collapse on repeated values
        edges = np.unique(edges)

    # ensure last edge captures max
    edges[-1] = max(edges[-1], float(np.max(yt)) + 1e-9)

    bins: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (yt >= lo) & (yt < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({
                "range": f"{lo:.2f}–{hi:.2f}",
                "lo": float(lo), "hi": float(hi),
                "n": 0, "mae": 0.0, "rmse": 0.0,
                "mean_bias": 0.0, "bias_direction": "balanced",
            })
            continue
        err = yp[mask] - yt[mask]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean_bias = float(np.mean(err))
        if mean_bias > mae * 0.1:
            direction = "over"
        elif mean_bias < -mae * 0.1:
            direction = "under"
        else:
            direction = "balanced"
        bins.append({
            "range": f"{lo:.2f}–{hi:.2f}",
            "lo": float(lo), "hi": float(hi),
            "n": n, "mae": mae, "rmse": rmse,
            "mean_bias": mean_bias, "bias_direction": direction,
        })

    # overall bias direction from the worst-bias bin
    if bins:
        worst = max(bins, key=lambda b: abs(b["mean_bias"]))
        overall = worst["bias_direction"]
    else:
        overall = "balanced"

    return {"bins": bins, "overall_bias_direction": overall}


def analyze_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Per-class precision/recall and top confusion pairs for narrative.
    """
    from sklearn.metrics import precision_score, recall_score

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    uniq = np.unique(np.concatenate([yt, yp]))
    if len(uniq) < 2:
        return {}

    cm = confusion_matrix(yt, yp, labels=uniq)
    n = cm.shape[0]
    prec = precision_score(yt, yp, average=None, zero_division=0, labels=uniq)
    rec = recall_score(yt, yp, average=None, zero_division=0, labels=uniq)
    per_class = []
    for i in range(n):
        per_class.append({
            'label': labels[i] if labels and i < len(labels) else str(uniq[i]),
            'precision': float(prec[i]),
            'recall': float(rec[i]),
        })

    flat = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                flat.append((int(cm[i, j]), int(i), int(j)))
    flat.sort(reverse=True)
    top_confusions = [(c, str(uniq[i]), str(uniq[j])) for c, i, j in flat[:5]]

    return {
        'confusion_matrix': cm,
        'per_class': per_class,
        'top_confusions': top_confusions,
        'labels': [str(x) for x in uniq],
    }


def analyze_bland_altman(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Stats for Bland–Altman narrative: mean difference, LoA, proportion outside LoA.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return {}
    a, b = a[valid], b[valid]
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    if std_diff == 0:
        return {}
    loa_low = mean_diff - 1.96 * std_diff
    loa_high = mean_diff + 1.96 * std_diff
    n_out = np.sum((diff < loa_low) | (diff > loa_high))
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'loa_low': loa_low,
        'loa_high': loa_high,
        'width_loa': loa_high - loa_low,
        'n': int(len(diff)),
        'n_outside_loa': int(n_out),
        'pct_outside_loa': float(n_out / len(diff)),
    }


def compare_models_paired_cv(
    model_names: List[str],
    model_results: Dict[str, Dict[str, Any]],
    task_type: str = "regression",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Pairwise comparison of models using CV fold-level metrics (paired t or Wilcoxon).
    Use when use_cv is True and cv_results with 'scores' exist per model.

    Returns:
        Dict mapping (model_a, model_b) -> {mean_delta, stat, p, test_name}
        mean_delta = mean(scores_a - scores_b); positive => b better (for MSE).
    """
    from ml.stats_tests import paired_location_test, normality_check

    results = {}
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1 :]:
            ra = model_results.get(ma, {}).get("cv_results") if isinstance(model_results.get(ma), dict) else None
            rb = model_results.get(mb, {}).get("cv_results") if isinstance(model_results.get(mb), dict) else None
            if not ra or not rb or "scores" not in ra or "scores" not in rb:
                continue
            sa = np.asarray(ra["scores"])
            sb = np.asarray(rb["scores"])
            if len(sa) != len(sb) or len(sa) < 2:
                continue
            diff = sa - sb
            _, norm_p, _ = normality_check(diff)
            parametric = np.isfinite(norm_p) and norm_p >= 0.05
            stat, p, name = paired_location_test(diff, parametric)
            results[(ma, mb)] = {
                "mean_delta": float(np.mean(diff)),
                "stat": stat,
                "p": p,
                "test_name": name,
            }
    return results


def compare_importance_ranks(
    model_names: List[str],
    perm_importance_dict: Dict[str, Dict[str, Any]],
    feature_names_by_model: Dict[str, List[str]],
    top_k: int = 5
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Compare permutation importance rankings across models.
    Only compares pairs that share the same feature set (e.g. same pipeline).
    
    Returns:
        Dict mapping (model_a, model_b) -> {spearman, top_k_overlap, n_features}
    """
    from scipy.stats import spearmanr
    
    results = {}
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            if ma not in perm_importance_dict or mb not in perm_importance_dict:
                continue
            fa = feature_names_by_model.get(ma)
            fb = feature_names_by_model.get(mb)
            if fa is None or fb is None or len(fa) != len(fb) or fa != fb:
                continue
            imp_a = perm_importance_dict[ma]['importances_mean']
            imp_b = perm_importance_dict[mb]['importances_mean']
            if len(imp_a) != len(imp_b) or len(imp_a) == 0:
                continue
            r, p = spearmanr(imp_a, imp_b)
            top_a = set(np.argsort(imp_a)[-top_k:].tolist())
            top_b = set(np.argsort(imp_b)[-top_k:].tolist())
            overlap = len(top_a & top_b)
            results[(ma, mb)] = {
                'spearman': float(r) if not np.isnan(r) else None,
                'spearman_p': float(p) if not np.isnan(p) else None,
                'top_k_overlap': overlap,
                'top_k': top_k,
                'n_features': len(imp_a)
            }
    return results
