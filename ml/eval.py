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


def regression_scoring_disclosure(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Optional[Dict[str, int]]:
    """What `calculate_regression_metrics` could NOT score, or None.

    `MINE-027`: the drop is deliberate, but the NUMBER is not optional. A
    degenerate target-transform back-mapping (e.g. a power transform fit on a
    near-constant target) can make 30% of test predictions non-finite; those
    pairs are dropped, and the R² that comes back is computed on the 70% the
    model handled — biased optimistic, because the dropped rows are exactly the
    ones it blew up on.

    The first repair put `n_dropped_nonfinite` and `n_scored` INTO the metrics
    dict, and a metrics dict is iterated: they became a tile in Train &
    Compare's Test Set Metrics row, two columns of the model comparison table,
    and `n_dropped_nonfinite=30` printed as a metric in the narrative's Methods
    sentence. A disclosure is not a metric. It is returned separately here, and
    the surfaces that PUBLISH the metrics state it in prose or in a table
    footnote — see pages/10's Test R² line, `_metrics_to_latex_table`, and
    `NarrativeEngine._gen_model_evaluation`.

    Returns None when every pair was scorable, so a caller storing it beside
    the metrics stores nothing on the ordinary path.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    if not n_dropped:
        return None
    return {'n_dropped_nonfinite': n_dropped,
            'n_scored': int(finite.sum()),
            'n_pairs': int(finite.size)}


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Non-finite prediction pairs are dropped before scoring: a degenerate
    target-transform back-mapping (e.g. a power transform fit on a
    near-constant target) can produce NaN/inf predictions, and sklearn's
    metric functions raise on them. If no finite pairs remain, all metrics
    are NaN — an honest 'no valid predictions' rather than a crash.

    The count of what was dropped is NOT in this dict — everything downstream
    iterates it as metrics. Ask `regression_scoring_disclosure` for it
    (`MINE-027`); any surface that publishes these numbers must.

    Returns:
        Dictionary with MAE, RMSE, R2, MedianAE.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    if n_dropped:
        import logging
        logging.getLogger(__name__).warning(
            "calculate_regression_metrics: dropping %d non-finite prediction "
            "pair(s) — check for a degenerate target transform",
            n_dropped,
        )
        y_true, y_pred = y_true[finite], y_pred[finite]

    if y_true.size < 2:
        return {'MAE': float('nan'), 'RMSE': float('nan'),
                'R2': float('nan'), 'MedianAE': float('nan')}

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


# ── one compute thread per fold worker ───────────────────────────────────────
#
# `perform_cross_validation` below dispatches its folds with `n_jobs=-1`, so
# every fold is fitted in its own loky process. If each of those processes then
# starts a compute thread pool of its own, the product is what oversubscribes
# the box: on the 8-core / 32 GB laptop this app targets, five folds times eight
# LightGBM threads is forty compute threads over eight cores, with a browser
# already competing for the machine.
#
# **joblib solves most of that already, and this must not duplicate it.**
# `LokyBackend._prepare_worker_env` sets OMP/MKL/OPENBLAS/BLIS/NUMEXPR/VECLIB in
# every worker to `cpu_count() // n_jobs`, which for `n_jobs=-1` is exactly 1.
# Measured inside real fold workers under a five-way `Parallel(n_jobs=-1)`
# dispatch (14-core box, 20,000 x 60), extra OS threads created by `fit`:
#
#     HistGradientBoosting (no n_jobs)   +1     XGBoost      (n_jobs=None)  +1
#     ExtraTrees           (n_jobs=None)  +1    kNN          (n_jobs=None)  +1
#     LightGBM             (n_jobs=None) +12    RandomForest (n_jobs=-1)   +18
#
# So the audit line that XGBoost and HistGradientBoosting "spawn a full OpenMP
# pool inside every fold process" is measurably false — joblib pinned them
# before we got there. Two things are left, and they need different mechanisms.
#
# **1. LightGBM ignores the environment pin.** It reads neither
# `OMP_NUM_THREADS` nor `threadpoolctl` and sizes its pool from its own core
# detection, so it runs a full pool per fold regardless. Its `n_jobs`
# constructor parameter is the only lever that reaches it, and it is worth
# reaching: in-worker fit 10.6 s -> 3.9 s at 20,000 x 60, and 5-fold CV wall
# 23.4 s -> 11.7 s at 8 cores / 20,000 x 120 (medians of five runs each). Run
# end to end through this function on the app's own composite at 20,000 x 120:
# wall 8.7/8.9 s -> 5.5/6.8 s, process-tree threads 282 -> 126, peak tree RSS
# 1872 -> 1590 MB, and mean CV MSE 2.409050627730724 in every one of the four
# runs — the same number to fifteen decimals. That is `_inner_thread_overrides`.
#
# One residual risk worth naming rather than burying: LightGBM's `deterministic`
# parameter defaults to `false`, and its documentation makes bit-stability across
# `num_threads` conditional on setting it. Every shape tested here came back
# bitwise equal — 20,000 x 120 and 4,000 x 60 (clinical), and 200 x 4,000,
# 120 x 1,200 and 60 x 3,000 (p/n > 1, the omics case), `predict` and
# `predict_proba` both, `maxdiff` 0.000e+00 throughout, XGBoost alongside it —
# but five shapes is evidence, not a proof, and the vendor's own guarantee is
# weaker than the measurement. Setting `deterministic=true` would buy the
# guarantee at a documented speed cost and is a change to how the model fits,
# so it is not made here.
#
# **2. joblib's guard only fires when the variable is absent.** It is
# `os.environ.get(var, cpu_count // n_jobs)`, so a parent process that already
# has `OMP_NUM_THREADS` set hands its value to every worker instead. No app
# code sets it today — the only occurrence in the repository is
# .github/workflows/ci.yml:60, which sets it to 1 — but one line in a launcher
# or in a user's shell silently turns five folds times one thread into five
# times eight, which is a hazard the app cannot see coming. Measured on
# HistGradientBoosting as CV wall 10.9 s -> 29.5 s and 103 -> 219 process-tree
# threads. Passing `inner_max_num_threads=1` makes joblib SET the value rather
# than default it, which closes that hole; verified by presetting the parent's
# `OMP_NUM_THREADS=8` and reading the variable back inside the workers.
#
# Note the asymmetry that keeps this safe: the pin is applied to a clone, for
# the duration of one call, at the site that spawns the workers. It never
# touches the registry factories, because the single full-data fit that follows
# CV in pages/06 should still get the whole machine.

try:                                    # joblib >= 1.3
    from joblib import parallel_config as _joblib_thread_config
except ImportError:                     # pragma: no cover - joblib 1.2 and older
    from joblib import parallel_backend as _joblib_thread_config


def _worker_thread_pin():
    """Context manager setting every fold worker's thread-limit env var to 1.

    `inner_max_num_threads` is a backend constructor argument, so joblib
    requires the backend to be named explicitly; 'loky' is what
    `cross_val_score(n_jobs=-1)` would have chosen anyway, and joblib itself
    falls back to a sequential backend where loky cannot run (nested calls,
    non-main threads). Degrades to a no-op context rather than failing a CV run
    if a future joblib changes the signature — a missing thread pin is a
    performance regression, not a wrong number.
    """
    try:
        return _joblib_thread_config(backend='loky', inner_max_num_threads=1)
    except Exception:                   # pragma: no cover - signature drift
        from contextlib import nullcontext
        return nullcontext()


def _inner_thread_overrides(estimator: Any) -> Dict[str, int]:
    """`set_params` keys that pin an estimator's own thread pool to one thread.

    Every `n_jobs` leaf is collected from `get_params(deep=True)` rather than
    named literally, because the CV estimator has two shapes. The plain
    composite from `make_cv_pipeline` exposes `est__n_jobs`; the regression
    target-transform branch (pages/06 wraps the estimator in a
    `TransformedTargetRegressor` whenever a log1p / Yeo-Johnson target
    transform is active) exposes ONLY `est__regressor__n_jobs`, and a hardcoded
    `est__n_jobs` would silently no-op for every such run. Bare estimators — the
    shape tests/test_cv_strategies.py passes — expose plain `n_jobs`.

    **scikit-learn's own estimators are deliberately skipped**, whatever value
    they carry. Almost all of them ship `n_jobs=None`, which already means one
    worker — measured in a fold worker as +1 thread for both ExtraTrees and
    kNN, identical to `n_jobs=1`. So the pin would buy nothing there, and it is
    not free: scikit-learn 1.9 deprecated `LogisticRegression.n_jobs`, and its
    `fit` warns `FutureWarning: 'n_jobs' has no effect since 1.8 and will be
    removed in 1.10` for any value that is not None. The estimators that need
    pinning are the third-party OpenMP ones that read `None` as "every core" —
    LightGBM above all, XGBoost for free.

    RandomForest is the one registry estimator that breaks the `None` half of
    that argument, and it IS reachable from here — an earlier version of this
    comment claimed it was not, and was wrong about which object gets
    cross-validated. `RFWrapper` never reaches `cross_val_score`: pages/06
    cross-validates `_sklearn_clone(model.get_model())`, i.e. the bare
    `RandomForestRegressor`/`Classifier` held inside the wrapper, which carries
    the `n_jobs=-1` hardcoded at models/rf.py:34 and :42. So the composite does
    expose `est__n_jobs = -1`, `set_params` does reach it, and the value would
    survive the per-fold `clone()` — `_pin_inner_threads` sets it before
    `cross_val_score` is ever handed the object, so there is nothing left for
    the clone to undo.

    It is skipped anyway, and on measurement rather than on reachability. Five
    folds cannot saturate eight cores, so RandomForest's own thread pool is
    what fills the three that would otherwise idle, and pinning it to 1 leaves
    them idle. Real composite, 8,000 x 60, 5-fold KFold under an 8-CPU affinity
    mask, two runs each: wall 175.0 / 143.6 s at `n_jobs=-1` against
    196.5 / 180.1 s pinned. Pinning costs about 18% of the wall clock and
    returns 164 -> 109 process-tree threads and 1220 -> 1176 MB peak tree RSS,
    a 3.6% memory saving. LightGBM's pin bought 2.0x in the other direction;
    this one is a net loss on the hardware this app targets, so it is not
    taken.

    That is a resource trade and not a correctness one, which is worth
    recording so a future change here can be argued on its merits: the fitted
    forest is byte-identical either way — every tree's `feature`, `threshold`
    and `value` array compares equal at `n_jobs=-1` against `n_jobs=1` — and
    mean CV MSE was 0.4911286040039654 in all four runs above. What differs is
    the order the ensemble average is accumulated in, 1.776e-15 on `predict`,
    which is the same magnitude two runs of the UNCHANGED code differ by:
    `RandomForestRegressor(n_jobs=-1)` is not bit-reproducible run to run.
    """
    try:
        params = estimator.get_params(deep=True)
    except Exception:
        return {}                       # not an sklearn-API estimator: nothing to pin

    overrides: Dict[str, int] = {}
    for key, value in params.items():
        if key != 'n_jobs' and not key.endswith('__n_jobs'):
            continue
        owner = estimator if key == 'n_jobs' else params.get(key[:-len('__n_jobs')])
        if owner is None or value == 1:
            continue
        if type(owner).__module__.split('.')[0] == 'sklearn':
            continue
        overrides[key] = 1
    return overrides


def _pin_inner_threads(estimator: Any) -> Any:
    """The estimator to cross-validate, with its own thread pools pinned to 1.

    Returns a CLONE when there is anything to pin — the caller's object is the
    one pages/06 keeps for the full-data fit, and that fit is entitled to the
    whole machine. Returns the caller's object untouched when there is nothing
    to pin, so the common bare-estimator case costs nothing.
    """
    overrides = _inner_thread_overrides(estimator)
    if not overrides:
        return estimator
    try:
        from sklearn.base import clone
        pinned = clone(estimator)
        pinned.set_params(**overrides)
        return pinned
    except Exception:                   # pragma: no cover - non-cloneable estimator
        return estimator


def perform_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    task_type: str = 'regression',
    scoring: Optional[str] = None,
    cv_strategy: str = 'standard',
    groups: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation, matching the fold scheme to the split.

    The CV strategy must respect the same leakage semantics as the train/test
    split, or the CV score is optimistically biased:
    - 'group': the split kept each entity's rows together (longitudinal /
      repeated measures). Random KFold would put the same entity on both sides
      of a fold, so use GroupKFold (StratifiedGroupKFold for classification).
    - 'time': the split was chronological. Random KFold would train on the
      future to predict the past, so use TimeSeriesSplit (no shuffle). X is
      assumed to be in chronological order (page 06 supplies it so).
    - 'standard' (default): StratifiedKFold (classification) / KFold.

    Args:
        model: Model with fit/predict interface
        X: Features (training rows only — the lockbox test never enters CV)
        y: Targets
        cv_folds: Number of folds
        task_type: 'regression' or 'classification'
        scoring: Scoring metric (if None, uses default for task type)
        cv_strategy: 'standard' | 'group' | 'time'
        groups: entity labels aligned to X rows (required for 'group')

    Returns:
        Dictionary with metric arrays across folds, plus the strategy used.
    """
    if scoring is None:
        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'

    strat = (cv_strategy or 'standard').lower()
    cv_groups = None  # passed through to cross_val_score for group schemes

    if strat == 'group' and groups is not None and len(np.unique(groups)) >= 2:
        # Never ask for more folds than there are groups.
        n_splits = max(2, min(cv_folds, int(len(np.unique(groups)))))
        cv = None
        if task_type == 'classification':
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            except ImportError:
                cv = None
        if cv is None:
            from sklearn.model_selection import GroupKFold
            cv = GroupKFold(n_splits=n_splits)
        cv_groups = groups
    elif strat == 'time':
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=cv_folds)
    elif task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Folds run as processes; each fold's estimator gets ONE compute thread.
    # See "one compute thread per fold worker" above for what each half covers
    # and what it measurably does not — RandomForest's `n_jobs=-1` is reachable
    # but deliberately left alone, because pinning it measured ~18% slower for
    # 3.6% of the memory. `n_jobs=-1` here is unchanged too: making the fold
    # count conditional is a separate question, and SVR is 2.9x slower without
    # process-parallel folds.
    with _worker_thread_pin():
        scores = cross_val_score(_pin_inner_threads(model), X, y, cv=cv,
                                 scoring=scoring, n_jobs=-1, groups=cv_groups)

    # Convert to positive if using negative MSE
    if 'neg_' in scoring:
        scores = -scores

    return {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'folds': int(getattr(cv, 'n_splits', cv_folds)),
        'strategy': strat,
    }


def _to_dense(A):
    """Named (picklable, for n_jobs=-1) densifier for CV pipelines."""
    return A.toarray() if hasattr(A, "toarray") else A


def make_cv_pipeline(preprocessing, estimator):
    """Compose an UNFITTED clone of a preprocessing pipeline with an estimator.

    Cross-validating this composite on raw training data re-fits the
    preprocessing inside every fold, so no fold's held-out rows contribute to
    imputer/scaler/encoder/PCA statistics — scoring a pre-transformed matrix
    would leak each fold's test portion into the transformer fits.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    return Pipeline([
        ("prep", clone(preprocessing)),
        ("densify", FunctionTransformer(_to_dense)),
        ("est", estimator),
    ])


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

    # ensure edges span the full data range
    edges[0] = min(edges[0], float(np.min(yt)) - 1e-9)
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
