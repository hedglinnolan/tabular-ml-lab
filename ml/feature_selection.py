"""
Feature selection methods for publication-grade variable selection.

Provides LASSO path, RFE-CV, univariate screening with FDR correction,
and stability selection.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FeatureSelectionResult:
    """Result from a feature selection method."""
    method: str
    selected_features: List[str]
    all_features: List[str]
    scores: Dict[str, float]  # feature -> importance/score
    details: Dict  # method-specific details
    description: str


def lasso_path_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "regression",
    n_alphas: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """LASSO path analysis with CV for optimal regularization.

    Shows how feature coefficients change with increasing regularization,
    identifying the most robust predictors.
    """
    if task_type == "regression":
        from sklearn.linear_model import LassoCV, lasso_path
        model = LassoCV(cv=cv_folds, n_alphas=n_alphas, random_state=random_state, max_iter=10000)
    else:
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(
            cv=cv_folds, penalty='l1', solver='saga',
            Cs=n_alphas, random_state=random_state, max_iter=10000,
        )

    model.fit(X, y)

    if task_type == "regression":
        coefs = model.coef_
        optimal_alpha = model.alpha_
        # Get the path — alphas must be array-like or None
        alpha_max = np.abs(X.T @ y).max() / (2.0 * X.shape[0])
        eps = 1e-3
        alphas_grid = np.logspace(np.log10(alpha_max), np.log10(alpha_max * eps), n_alphas)
        alphas, path_coefs, _ = lasso_path(X, y, alphas=alphas_grid)
    else:
        coefs = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
        optimal_alpha = 1.0 / model.C_[0] if hasattr(model, 'C_') else 0.0
        alphas, path_coefs = None, None

    # Selected features (non-zero coefficients)
    selected_mask = np.abs(coefs) > 1e-10
    selected = [f for f, s in zip(feature_names, selected_mask) if s]
    scores = {f: abs(c) for f, c in zip(feature_names, coefs)}

    details = {
        "optimal_alpha": float(optimal_alpha),
        "n_selected": len(selected),
        "coefficients": {f: float(c) for f, c in zip(feature_names, coefs)},
    }
    if alphas is not None:
        details["alphas"] = alphas.tolist()
        details["path_coefs"] = path_coefs.tolist()

    return FeatureSelectionResult(
        method="LASSO",
        selected_features=selected,
        all_features=feature_names,
        scores=scores,
        details=details,
        description=f"LASSO selected {len(selected)}/{len(feature_names)} features "
                    f"at optimal α={optimal_alpha:.4f} ({cv_folds}-fold CV).",
    )


def rfe_cv_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "regression",
    cv_folds: int = 5,
    step: int = 1,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """Recursive Feature Elimination with Cross-Validation.

    Iteratively removes least important features and selects the
    subset that maximizes CV performance.
    """
    from sklearn.feature_selection import RFECV

    if task_type == "regression":
        from sklearn.linear_model import Ridge
        estimator = Ridge(alpha=1.0, random_state=random_state)
        scoring = "neg_mean_squared_error"
    else:
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(random_state=random_state, max_iter=1000)
        scoring = "accuracy"

    rfecv = RFECV(
        estimator=estimator, step=step, cv=cv_folds,
        scoring=scoring, min_features_to_select=1, n_jobs=-1,
    )
    rfecv.fit(X, y)

    selected = [f for f, s in zip(feature_names, rfecv.support_) if s]
    rankings = {f: int(r) for f, r in zip(feature_names, rfecv.ranking_)}
    scores = {f: 1.0 / r for f, r in rankings.items()}

    details = {
        "rankings": rankings,
        "n_features_optimal": int(rfecv.n_features_),
        "cv_scores": rfecv.cv_results_['mean_test_score'].tolist() if hasattr(rfecv, 'cv_results_') else [],
    }

    return FeatureSelectionResult(
        method="RFE-CV",
        selected_features=selected,
        all_features=feature_names,
        scores=scores,
        details=details,
        description=f"RFE-CV selected {len(selected)}/{len(feature_names)} features "
                    f"as the optimal subset ({cv_folds}-fold CV).",
    )


def univariate_screening(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "regression",
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> FeatureSelectionResult:
    """Univariate feature screening with multiple testing correction.

    Tests each feature individually against the target and applies
    FDR correction (Benjamini-Hochberg) or Bonferroni.
    """
    from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
    from statsmodels.stats.multitest import multipletests

    p_values = []
    test_stats = []
    test_names = []

    for i, fname in enumerate(feature_names):
        xi = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i]
        mask = ~np.isnan(xi) & ~np.isnan(y) if np.issubdtype(type(xi[0]) if len(xi) > 0 else float, np.floating) else np.ones(len(xi), dtype=bool)

        try:
            if task_type == "regression":
                stat, p = pearsonr(xi[mask], y[mask])
                test_names.append("Pearson r")
            else:
                stat, p = spearmanr(xi[mask], y[mask])
                test_names.append("Spearman ρ")
            p_values.append(p)
            test_stats.append(abs(stat))
        except Exception:
            p_values.append(1.0)
            test_stats.append(0.0)
            test_names.append("failed")

    p_values = np.array(p_values)

    # Multiple testing correction
    reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method=correction)

    selected = [f for f, r in zip(feature_names, reject) if r]
    scores = {f: float(s) for f, s in zip(feature_names, test_stats)}

    details = {
        "raw_p_values": {f: float(p) for f, p in zip(feature_names, p_values)},
        "corrected_p_values": {f: float(p) for f, p in zip(feature_names, corrected_p)},
        "test_statistics": {f: float(s) for f, s in zip(feature_names, test_stats)},
        "test_names": {f: t for f, t in zip(feature_names, test_names)},
        "correction_method": correction,
        "alpha": alpha,
        "rejected": {f: bool(r) for f, r in zip(feature_names, reject)},
    }

    correction_name = "Benjamini-Hochberg FDR" if correction == "fdr_bh" else correction
    return FeatureSelectionResult(
        method=f"Univariate ({correction_name})",
        selected_features=selected,
        all_features=feature_names,
        scores=scores,
        details=details,
        description=f"Univariate screening selected {len(selected)}/{len(feature_names)} features "
                    f"at α={alpha} with {correction_name} correction.",
    )


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "regression",
    n_bootstrap: int = 100,
    threshold: float = 0.6,
    sample_fraction: float = 0.5,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """Stability selection (Meinshausen & Bühlmann, 2010).

    Runs LASSO on random subsamples and selects features that are
    consistently chosen across subsamples.
    """
    from sklearn.linear_model import Lasso, LogisticRegression

    rng = np.random.RandomState(random_state)
    n, p = X.shape
    selection_counts = np.zeros(p)

    for i in range(n_bootstrap):
        # Random subsample
        idx = rng.choice(n, size=int(n * sample_fraction), replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # Fit LASSO
        if task_type == "regression":
            model = Lasso(alpha=0.01, max_iter=10000, random_state=rng.randint(10000))
        else:
            model = LogisticRegression(
                penalty='l1', solver='saga', C=100.0,
                max_iter=10000, random_state=rng.randint(10000),
            )

        try:
            model.fit(X_sub, y_sub)
            coefs = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
            selection_counts += (np.abs(coefs) > 1e-10).astype(float)
        except Exception:
            continue

    # Selection probability
    selection_probs = selection_counts / n_bootstrap
    selected = [f for f, p in zip(feature_names, selection_probs) if p >= threshold]
    scores = {f: float(p) for f, p in zip(feature_names, selection_probs)}

    details = {
        "selection_probabilities": scores,
        "threshold": threshold,
        "n_bootstrap": n_bootstrap,
        "sample_fraction": sample_fraction,
    }

    return FeatureSelectionResult(
        method="Stability Selection",
        selected_features=selected,
        all_features=feature_names,
        scores=scores,
        details=details,
        description=f"Stability selection found {len(selected)}/{len(feature_names)} features "
                    f"with selection probability ≥{threshold} across {n_bootstrap} subsamples.",
    )


def consensus_features(results: List[FeatureSelectionResult], min_methods: int = 2) -> List[str]:
    """Find features selected by multiple methods.

    Args:
        results: List of FeatureSelectionResult from different methods
        min_methods: Minimum number of methods that must select a feature

    Returns:
        List of consensus features
    """
    from collections import Counter

    counts = Counter()
    for r in results:
        for f in r.selected_features:
            counts[f] += 1

    return [f for f, c in counts.most_common() if c >= min_methods]
