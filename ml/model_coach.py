"""
Model Selection Coach: Intelligent, educational assistant for model selection.

Provides data-aware recommendations that:
- Consider dataset size, dimensionality, and characteristics
- Explain recommendations in plain language
- Bucket models into Recommended/Worth Trying/Not Recommended
- Integrate throughout the ML workflow
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence, Tuple
from enum import Enum

# `AUDIT-020` / `AUDIT-021` — one design, one module, `CLINICAL_SURVEY_PACK.md`
# §A5.4. The threshold, the denominator and the sentence that says the rule of
# 10 is superseded all live there so the two doors cannot state them differently.
from ml import sample_size as _ss


class TrainingTimeTier(Enum):
    """Expected training time tiers."""
    FAST = "fast"           # < 10 seconds
    MEDIUM = "medium"       # 10 seconds - 2 minutes
    SLOW = "slow"           # > 2 minutes


# Canonical group display names
GROUP_DISPLAY_NAMES = {
    'Linear': 'Linear Models',
    'Trees': 'Tree-Based Models',
    'Boosting': 'Gradient Boosting',
    'Distance': 'Distance-Based Models',
    'Margin': 'Support Vector Machines',
    'Probabilistic': 'Probabilistic Models',
    'Neural Net': 'Neural Networks'
}


def _get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get model information for recommendations.
    
    Returns dict mapping model_key to model metadata.
    """
    return {
        # Linear Models
        'glm': {
            'name': 'GLM (OLS/Logistic)',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': False,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,  # Events per variable
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        'ridge': {
            'name': 'Ridge Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'lasso': {
            'name': 'Lasso Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'elasticnet': {
            'name': 'ElasticNet',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 20,
            'min_epv': 3,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'huber': {
            'name': 'Huber Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'logreg': {
            'name': 'Logistic Regression',
            'group': 'Linear',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        
        # Tree-based
        'rf': {
            'name': 'Random Forest',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'extratrees_reg': {
            'name': 'Extra Trees',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'extratrees_clf': {
            'name': 'Extra Trees',
            'group': 'Trees',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'medium',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Boosting
        'histgb_reg': {
            'name': 'Histogram Gradient Boosting',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'histgb_clf': {
            'name': 'Histogram Gradient Boosting',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Distance-based
        'knn_reg': {
            'name': 'k-Nearest Neighbors',
            'group': 'Distance',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'medium',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        'knn_clf': {
            'name': 'k-Nearest Neighbors',
            'group': 'Distance',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'medium',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        
        # SVMs
        'svr': {
            'name': 'Support Vector Regression',
            'group': 'Margin',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'svc': {
            'name': 'Support Vector Classification',
            'group': 'Margin',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 100,
            'min_epv': 20,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        
        # Probabilistic
        'gaussian_nb': {
            'name': 'Gaussian Naive Bayes',
            'group': 'Probabilistic',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': False,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 30,
            'min_epv': 5,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },
        'lda': {
            'name': 'Linear Discriminant Analysis',
            'group': 'Probabilistic',
            'training_time': TrainingTimeTier.FAST,
            'interpretability': 'high',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 50,
            'min_epv': 10,
            'good_for_high_dim': False,
            'robust_to_outliers': False,
        },
        
        # Neural Networks
        'nn': {
            'name': 'Neural Network',
            'group': 'Neural Net',
            'training_time': TrainingTimeTier.SLOW,
            'interpretability': 'low',
            'requires_scaling': True,
            'requires_encoding': True,
            'handles_missing': False,
            'min_samples': 500,
            'min_epv': 50,
            'good_for_high_dim': True,
            'robust_to_outliers': False,
        },

        # XGBoost
        'xgb_reg': {
            'name': 'XGBoost',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'xgb_clf': {
            'name': 'XGBoost',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },

        # LightGBM
        'lgbm_reg': {
            'name': 'LightGBM',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
        'lgbm_clf': {
            'name': 'LightGBM',
            'group': 'Boosting',
            'training_time': TrainingTimeTier.MEDIUM,
            'interpretability': 'low',
            'requires_scaling': False,
            'requires_encoding': False,
            'handles_missing': True,
            'min_samples': 100,
            'min_epv': 15,
            'good_for_high_dim': False,
            'robust_to_outliers': True,
        },
    }


# Legacy compatibility function


# ============================================================================
# TOP PICKS: Role-based model selection (replaces bucket-based for UI)
# ============================================================================

@dataclass
class TopPick:
    """A single model recommendation with a role label."""
    role: str           # "Start here", "Try next", "Alternative"
    model_key: str
    model_name: str
    group: str
    why: str            # One sentence: why THIS model for THIS data
    preprocessing: str  # What preprocessing it needs
    requires_scaling: bool
    handles_missing: bool


def _skew_split(profile: Any, skewed: Sequence[str]) -> Dict[str, List[str]]:
    """Which of the skewed features a LOG could actually be applied to.

    `ml/dataset_profile.py:641` selects on `abs(skewness) > 1.0`, so `skewed`
    holds two populations a log treats oppositely and one it cannot touch:

    * ``log_ok``       strictly positive and skewed RIGHT — the log's own case
    * ``left_skewed``  skewed LEFT; a log lengthens the left tail further
    * ``not_positive`` a value at or below zero; `log` is undefined there

    Read off `FeatureProfile.min_val` and `.skewness`, which the profile already
    carries. A feature whose profile is missing either field lands in
    ``unread`` rather than being assumed positive.
    """
    fps = getattr(profile, "feature_profiles", None) or {}
    out: Dict[str, List[str]] = {"log_ok": [], "left_skewed": [],
                                 "not_positive": [], "unread": []}
    for col in skewed:
        fp = fps.get(col)
        lo = getattr(fp, "min_val", None) if fp is not None else None
        sk = getattr(fp, "skewness", None) if fp is not None else None
        if lo is None or sk is None:
            out["unread"].append(col)
        elif lo <= 0:
            out["not_positive"].append(col)
        elif sk < 0:
            out["left_skewed"].append(col)
        else:
            out["log_ok"].append(col)
    return out


def _skew_transform_clause(profile: Any, skewed: Sequence[str]) -> str:
    """`AUDIT-003` on the preprocessing card. BEFORE:

        "apply Yeo-Johnson or log transform to stabilize the feature
         distributions."

    One sentence for every skewed feature, and `log` was offered for all of
    them. `ml/dataset_profile.py:641` selects on `abs(skewness) > 1.0`, so the
    same sentence covered a LEFT-skewed feature, where a log makes the skew
    worse, and a feature holding zeros or negatives, where a log is undefined.

    AFTER: Yeo-Johnson is still recommended for all of them — it is defined on
    the whole real line and on either tail — and the LOG is named only for the
    features it is defined and useful on, with the excluded ones counted. The
    same subject, a weaker claim, true. And the clause the reading cannot
    supply is stated: nothing here can see whether a column arrived already
    transformed, and a second log is silent.
    """
    split = _skew_split(profile, skewed)
    n = len(list(skewed))
    parts = [f"apply Yeo-Johnson to all {n} — it is defined on zero, on "
             f"negatives and on either tail"]

    ok = split["log_ok"]
    if ok:
        shown = ", ".join(ok[:3]) + ("…" if len(ok) > 3 else "")
        parts.append(f"A LOG transform is an option for {len(ok)} of them "
                     f"({shown}), strictly positive and skewed right")
    else:
        parts.append("A LOG transform is an option for none of them")

    def _they(names, singular, plural):
        return f"{len(names)} of them {singular if len(names) == 1 else plural}"

    excluded = []
    if split["not_positive"]:
        excluded.append(_they(split["not_positive"], "reaches", "reach")
                        + " zero or below, where the log is undefined")
    if split["left_skewed"]:
        excluded.append(_they(split["left_skewed"], "is", "are")
                        + " skewed LEFT, where a log lengthens the tail it is "
                          "meant to shorten")
    if split["unread"]:
        excluded.append(_they(split["unread"], "carries", "carry")
                        + " no min/skew reading, so nothing here can say")
    if excluded:
        parts.append("; and ".join(excluded))

    return (". ".join(parts) + ". Whether any of these arrived already "
            "transformed is not checked here, and a second log is silent.")


def _huber_why(tp: Any) -> str:
    """Why Huber, in the tier the app can actually distinguish.

    `AUDIT-012`. BEFORE — one sentence for every regression target alike:

        "The outcome itself contains outliers (12% of values) — Huber
         downweights extreme residuals so they don't steer the fit. Feature
         outliers are handled in preprocessing instead."

    `tp.outlier_rate` is an IQR fence count (`ml/outliers.py:44`). On
    `clinical_labs.csv::sbp` it counts a systolic pressure of **0 mmHg** and one
    of **244 mmHg** as the same kind of thing. They are not: the first is an
    entry error and wants REPAIR, the second is a hypertensive crisis and wants
    MODELING — and Huber downweights both, which for the entry error is the
    wrong remedy and for the real extreme discards signal the study is about.
    `research/CLINICAL_SURVEY_PACK.md` Cross-cutting 7 is the source; the
    impossibility band that separates them was already computed one module over
    (`turbotab/engine.plausibility`) and nothing on this path read it.

    AFTER — the same recommendation, the same subject, a weaker claim: the two
    tiers are reported separately, and where the reference has nothing to say
    about this column the sentence says THAT rather than implying a check it
    did not make. Huber is still the pick in all three branches; the shelf is
    not shortened.
    """
    rate = tp.outlier_rate or 0.0
    tail = ("Feature outliers are handled in preprocessing instead.")
    read = getattr(tp, "physio_read", "unread")
    count = getattr(tp, "impossible_count", None)
    band = getattr(tp, "impossibility_band", None)

    if read == "matched" and band is not None and count:
        floor, ceiling, unit = band
        return (f"{rate:.0%} of the outcome's values fall outside the IQR "
                f"fences, and {count:,} of them are outside the published "
                f"impossible range for {tp.physio_variable} "
                f"({floor:g}–{ceiling:g} {unit}). Those are entry errors rather "
                f"than extreme measurements, and Huber DOWNWEIGHTS them instead "
                f"of removing them — repair them on the plausibility card "
                f"first, then Huber is the right model for the real extremes "
                f"that remain. {tail}")

    if read == "matched" and band is not None:
        floor, ceiling, unit = band
        return (f"The outcome itself contains outliers ({rate:.0%} of values by "
                f"the IQR fences) — Huber downweights extreme residuals so they "
                f"don't steer the fit. None of them is outside the published "
                f"impossible range for {tp.physio_variable} "
                f"({floor:g}–{ceiling:g} {unit}), so they read as real extremes "
                f"rather than entry errors, which is exactly the case Huber is "
                f"for. {tail}")

    unnamed = (f"'{tp.name}' matches no variable in the physiologic reference"
               if read == "unrecognized" else
               "no impossibility band was read for this outcome")
    return (f"The outcome itself contains outliers ({rate:.0%} of values) — "
            f"Huber downweights extreme residuals so they don't steer the fit. "
            f"That rate is an IQR fence count and nothing more: {unnamed}, so "
            f"the app cannot tell a physiologically impossible entry from an "
            f"abnormal-but-real measurement here, and Huber treats the two the "
            f"same way. {tail}")


def select_top_picks(profile: Any, probe: Any = None) -> Tuple[List[TopPick], List[Tuple[str, str]], str]:
    """Select 2-3 models from the dataset's SHAPE, dominant constraint first.

    The old logic reacted to whatever signal it checked first (feature
    outliers beat p≫n), used feature outliers to justify Huber (which is
    robust to TARGET outliers, not feature outliers), never looked at
    events-per-variable or imbalance, and claimed calibration where the
    event count could not support it. This version names the dataset's
    dominant constraint, cites the numbers, and only claims what the shape
    can back.

    Args:
        profile: DatasetProfile.
        probe: optional ml.coach_probe.ProbeResult — measured evidence from
            the training rows. When present, the advice cites measurements
            instead of priors.

    Returns:
        (picks, skip_list, headline) — headline is one sentence naming the
        dataset's dominant modeling constraint (empty when unconstrained).
    """
    n = profile.n_rows
    p = profile.n_features
    tp = profile.target_profile
    task_type = tp.task_type if tp else "regression"
    # Huber's case is outliers in the OUTCOME; feature outliers are a
    # preprocessing matter (winsorize/robust-scale), not a model choice.
    target_outliers = bool(tp and tp.has_outliers and (tp.outlier_rate or 0) > 0.01)
    has_skew = bool(getattr(profile, "highly_skewed_features", []))
    is_high_dim = profile.p_n_ratio > 0.3
    is_wide = profile.p_n_ratio > 1.0
    has_missing = profile.n_features_with_missing > 0
    epv = profile.events_per_variable
    minority = tp.minority_class_size if tp else None
    imbalanced = bool(tp and tp.is_imbalanced)
    # `AUDIT-021`. `10` is this app's caution trigger and not the field's
    # guideline — §A5.4 is [SETTLED that EPV≥10 is superseded]. It lives in
    # `ml.sample_size.CAUTION_EPV` beside the sentence that says whose number
    # it is, and the value is unchanged this loop: what moved is the
    # DENOMINATOR it is applied to (`AUDIT-020`).
    low_epv = (task_type == "classification" and epv is not None
               and epv < _ss.CAUTION_EPV)
    #: §A5.4's denominator. `None` where the profile predates the field, and
    #: then the headline names no denominator rather than naming the wrong one.
    params = getattr(profile, "n_candidate_parameters", None)
    small_n = n < 150

    model_info = _get_model_info()
    picks: List[TopPick] = []
    skip_list: List[Tuple[str, str]] = []

    # --- HEADLINE: name the dominant constraint, with the numbers ---
    if is_wide:
        headline = (f"Dominant constraint: {p:,} predictors for {n:,} rows (p≫n). "
                    f"Unpenalized fits are not identifiable here — every pick below "
                    f"is regularized, and feature attribution will be unstable.")
    elif low_epv:
        _denominator = (f"{params:,} candidate parameters" if params is not None
                        else "the candidate parameters")
        headline = (f"Dominant constraint: {minority:,} minority-class events for "
                    f"{_denominator} (EPV = {epv:.1f}). {_ss.SUPERSEDED_SHORT} "
                    f"Keep the model lineup small, prefer penalized fits, and "
                    f"report confidence intervals — estimates will be unstable.")
    elif imbalanced:
        _ratio = tp.class_balance_ratio
        headline = (f"Dominant constraint: class imbalance "
                    f"({minority:,} minority events{f', {_ratio:.0f}:1 ratio' if _ratio else ''}). "
                    f"Judge models by AUROC and calibration rather than accuracy, "
                    f"which will look deceptively good. Rebalancing is NOT the "
                    f"remedy — it overestimates minority-class probability without "
                    f"improving discrimination ({_imbalance.CITATION}); set the "
                    f"decision threshold from the costs of the two errors instead.")
    elif small_n:
        headline = (f"Dominant constraint: {n} rows. Model rankings will vary "
                    f"fold-to-fold — report CV spread, not just the mean, and "
                    f"prefer simpler models.")
    else:
        headline = ""

    # --- MEASURED EVIDENCE (outranks shape priors when available) ---
    _gain = probe.nonlinearity_gain if probe is not None else None
    _signal = probe.has_signal if probe is not None else None
    if probe is not None:
        if _signal is False and not probe.underpowered:
            headline = (
                f"⚠️ Evidence probe: {probe.summary()}. Expect null results — "
                f"verify these predictors can plausibly relate to the outcome "
                f"before investing further. " + (headline or "")).strip()
        elif _signal is False and probe.underpowered:
            headline = ((headline + " " if headline else "")
                        + f"Evidence probe: {probe.summary()}.").strip()
        elif probe.data_hungry and _signal:
            headline = ((headline + " " if headline else "")
                        + "Evidence probe: scores were still rising with more "
                          "rows — collecting more data may beat tuning more "
                          "models.").strip()

    # --- 1. CORE LINEAR MODEL ---
    if task_type == "regression":
        if is_wide:
            linear_key, linear_name = "lasso", "Lasso Regression"
            linear_why = (f"With {p:,} predictors and {n:,} rows, the L1 penalty is what "
                          f"makes the fit identifiable — and it yields a defensible "
                          f"feature shortlist for the manuscript.")
        elif target_outliers:
            linear_key, linear_name = "huber", "Huber Regression"
            linear_why = _huber_why(tp)
        elif is_high_dim:
            linear_key, linear_name = "ridge", "Ridge Regression"
            linear_why = (f"At {p} predictors for {n:,} rows, the L2 penalty stabilizes "
                          f"coefficients that would otherwise swing with the sample.")
        else:
            linear_key, linear_name = "ridge", "Ridge Regression"
            linear_why = ("Stable, interpretable baseline; the penalty costs nothing "
                          "here and protects against correlated predictors.")
    else:
        linear_key, linear_name = "logreg", "Logistic Regression"
        if low_epv:
            linear_why = (f"With {minority:,} events for {p} predictors, penalized "
                          f"logistic regression is the defensible core model — expect "
                          f"wide confidence intervals on the coefficients.")
        elif imbalanced:
            linear_why = ("Interpretable log-odds baseline. Evaluate with AUROC "
                          "and calibration rather than accuracy, and penalize the "
                          "fit rather than rebalancing the outcome.")
        elif minority is not None and minority >= 100:
            linear_why = (f"Interpretable baseline; with {minority:,} events per class, "
                          f"its probability calibration is checkable on the Train page.")
        else:
            linear_why = "Interpretable log-odds baseline for classification."

    if linear_key in model_info:
        info = model_info[linear_key]
        pp_parts = []
        if info["requires_scaling"]:
            pp_parts.append("scale")
        if has_missing and not info["handles_missing"]:
            pp_parts.append("impute")
        if has_skew:
            pp_parts.append("transform skewed features")
        # `GUIDED-049`: penalization is the registry's named remedy for a rare
        # outcome; rebalancing is the contraindicated one. This line used to
        # append "class weights".
        if task_type == "classification" and imbalanced:
            pp_parts.append("penalize")
        if (_signal and _gain is not None and _gain < 0.02):
            linear_why += (f" An evidence probe measured trees ≈ linear on this "
                           f"data (Δ{probe.metric_name} = {_gain:+.2f}) — the "
                           f"interpretable model may be all you need.")
        picks.append(TopPick(
            role="Start here", model_key=linear_key, model_name=linear_name,
            group="Linear", why=linear_why,
            preprocessing=", ".join(pp_parts) if pp_parts else "minimal",
            requires_scaling=info["requires_scaling"],
            handles_missing=info["handles_missing"],
        ))

    # --- 2. TREE/ENSEMBLE (blocked when p≫n or EPV is too low — the skip
    #        list explains; a small lineup IS the advice at low EPV) ---
    tree_key = tree_name = None
    if not is_wide and not low_epv:
        if n >= 100:
            tree_key = "histgb_reg" if task_type == "regression" else "histgb_clf"
            tree_name = "Histogram Gradient Boosting"
            tree_why = ("Strongest tabular learner here: captures non-linearity and "
                        "interactions, and handles skewness, outliers, and missing "
                        "values natively.")
        elif n >= 50:
            tree_key, tree_name = "rf", "Random Forest"
            tree_why = ("Robust non-linear benchmark with few hyperparameters — a fair "
                        "test of whether anything beats the linear model.")
        if tree_key and _signal and _gain is not None and _gain > 0.04:
            tree_why += (f" An evidence probe measured +{_gain:.2f} "
                         f"{probe.metric_name} for shallow trees over linear — "
                         f"non-linear structure is really there.")
        if tree_key and small_n:
            tree_why += (f" At n={n}, expect fold-to-fold variability — judge it by CV "
                         f"spread, not the single best score.")
        if tree_key and task_type == "classification" and imbalanced:
            tree_why += (" Trees are the worst case for rebalancing — they are "
                         "already poorly calibrated out of the box, so report the "
                         "calibration curve rather than reweighting the classes.")

    if tree_key and tree_key in model_info:
        info = model_info[tree_key]
        pp_parts = []
        if has_missing and not info["handles_missing"]:
            pp_parts.append("impute")
        picks.append(TopPick(
            role="Try next", model_key=tree_key, model_name=tree_name,
            group="Trees/Boosting", why=tree_why,
            preprocessing=", ".join(pp_parts) if pp_parts else "minimal — encode categoricals only",
            requires_scaling=False, handles_missing=info["handles_missing"],
        ))

    # --- 3. WILDCARD (adds a genuinely different bias; omitted when the
    #        data cannot support comparing more models) ---
    wildcard_key = wildcard_name = wildcard_why = None
    if is_wide and task_type == "regression":
        wildcard_key, wildcard_name = "ridge", "Ridge Regression"
        wildcard_why = ("Keeps all correlated predictors with shrunken coefficients — "
                        "compare against LASSO's sparse shortlist to see how stable "
                        "the selection is.")
    elif low_epv:
        pass  # more models = more selection noise at this event count
    elif n >= 1000 and p >= 5:
        wildcard_key, wildcard_name = "nn", "Neural Network"
        wildcard_why = (f"{n:,} rows is enough to justify the capacity; it can capture "
                        f"smooth interaction surfaces trees approximate coarsely.")
    elif n >= 500 and p <= 30 and task_type == "classification":
        wildcard_key, wildcard_name = "gaussian_nb", "Gaussian Naive Bayes"
        wildcard_why = ("Nearly instant, and a genuinely different inductive bias — a "
                        "useful calibration comparison for the other two.")
    elif n >= 100 and p <= 20 and not is_high_dim:
        if task_type == "regression":
            wildcard_key, wildcard_name = "elasticnet", "ElasticNet"
            wildcard_why = ("L1 feature selection with L2 stability — useful if you "
                            "suspect some predictors are noise.")
        else:
            wildcard_key, wildcard_name = "lda", "Linear Discriminant Analysis"
            wildcard_why = ("Models the class distributions directly — a different "
                            "lens than logistic regression's decision boundary.")

    if wildcard_key and wildcard_key in model_info and wildcard_key not in {pk.model_key for pk in picks}:
        info = model_info[wildcard_key]
        if n >= info["min_samples"]:
            pp_parts = []
            if info["requires_scaling"]:
                pp_parts.append("scale")
            if has_missing and not info["handles_missing"]:
                pp_parts.append("impute")
            picks.append(TopPick(
                role="Alternative", model_key=wildcard_key, model_name=wildcard_name,
                group=info.get("group", "Other"), why=wildcard_why,
                preprocessing=", ".join(pp_parts) if pp_parts else "minimal",
                requires_scaling=info["requires_scaling"],
                handles_missing=info["handles_missing"],
            ))

    # --- SKIP LIST: shape-specific reasons with the numbers ---
    if is_wide:
        skip_list.append(("Tree ensembles",
                          f"with {p:,} predictors and {n:,} rows they memorize rather "
                          f"than generalize — reduce features first"))
    if low_epv:
        skip_list.append(("Tree ensembles / boosting / neural nets",
                          f"{minority:,} minority-class events across the candidate "
                          f"parameters (EPV = {epv:.1f}) — a boosted ensemble spends "
                          f"far more parameters than that, and every extra model is "
                          f"another chance to overfit the selection"))
    if n < 500:
        skip_list.append(("Neural Network", f"needs roughly 500+ rows; you have {n:,}"))
    if is_high_dim and not is_wide:
        skip_list.append(("KNN", f"distances lose meaning at p = {p} — neighbors stop being 'near'"))
    elif is_wide:
        skip_list.append(("KNN", f"distances lose meaning at p = {p:,}"))
    elif p > 20:
        skip_list.append(("KNN", f"adds little over tree models at p = {p}"))
    if n > 5000:
        skip_list.append(("SVM", f"kernel training scales roughly with n² — n = {n:,} would be slow"))
    else:
        skip_list.append(("SVM", "worth it only if the linear baseline underfits — try the picks first"))

    return picks, skip_list, headline


# ── Preprocessing Coaching (Model-Scoped) ─────────────────────────────────

def _count_word_coach(n: int, noun: str) -> str:
    """'1 feature' / '3 features' — manuscript prose avoids '(s)'."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def generate_preprocessing_insights(
    selected_models: List[str],
    profile: Any,
) -> List[Dict[str, Any]]:
    """Generate model-scoped preprocessing coaching insights.

    Returns a list of dicts ready to be converted to Insight objects.
    Each insight's ``model_scope`` narrows the recommendation to the
    model families that actually need the action.

    Parameters
    ----------
    selected_models : list of str
        Model keys the user has selected (e.g. ["ridge", "rf", "nn"]).
    profile : DatasetProfile or similar
        Must expose: ``highly_skewed_features``, ``features_with_outliers``,
        ``n_features_with_missing``.

    `COACH-007`. Every insight here carries an explicit manuscript disposition,
    because the ledger's default for an unresolved insight is the manuscript's
    LIMITATIONS list. Advice that is a reassurance ("the default pipeline
    already standardizes for you") or a neutral fact is marked
    ``metadata.audit_only`` with no ``manuscript_text``: it is coaching, and a
    step the app performed must never be printed to a reviewer as an
    unaddressed limitation. The two that describe a real data condition carry a
    manuscript-register sentence that is true whether or not the user acts on
    it. Each also states whether a control resolves it, so no advice is a
    promise the app cannot keep.
    """
    from utils.insight_ledger import (
        MODEL_TO_FAMILY, ISSUE_MODEL_RELEVANCE,
        MODEL_FAMILY_LINEAR, MODEL_FAMILY_TREE, MODEL_FAMILY_NEURAL,
        MODEL_FAMILY_DISTANCE, MODEL_FAMILY_MARGIN,
    )

    if not profile or not selected_models:
        return []

    # Determine which families are in the user's selection
    user_families = set()
    for mk in selected_models:
        fam = MODEL_TO_FAMILY.get(mk)
        if fam:
            user_families.add(fam)

    insights: List[Dict[str, Any]] = []

    # Helper: family names for display
    _family_names = {
        MODEL_FAMILY_LINEAR: "linear models (Ridge, LASSO, etc.)",
        MODEL_FAMILY_TREE: "tree-based models (RF, XGBoost, etc.)",
        MODEL_FAMILY_NEURAL: "neural networks",
        MODEL_FAMILY_DISTANCE: "distance-based models (kNN)",
        MODEL_FAMILY_MARGIN: "margin-based models (SVM)",
    }

    def _family_list(families):
        return ", ".join(_family_names.get(f, f) for f in families if f in user_families)

    # 1. Skewness → power transform (affects linear, neural, distance)
    skewed = getattr(profile, "highly_skewed_features", [])
    if skewed:
        affected = [f for f in ISSUE_MODEL_RELEVANCE["skewness"] if f in user_families]
        immune = [f for f in user_families if f not in ISSUE_MODEL_RELEVANCE["skewness"]]
        if affected:
            immune_msg = ""
            if immune:
                immune_msg = f" Your {_family_list(immune)} handle skewness natively — no transform needed for them."
            insights.append({
                "id": "preprocess_skewness_transform",
                "source_page": "05_Preprocess",
                "category": "preprocessing",
                "severity": "warning",
                "finding": (
                    f"{_count_word_coach(len(skewed), 'feature')} are highly skewed "
                    f"({', '.join(skewed[:3])}{'…' if len(skewed) > 3 else ''})."
                ),
                "implication": (
                    f"Skewness can bias {_family_list(affected)}, "
                    "producing suboptimal coefficients or gradient updates."
                ),
                "recommended_action": (
                    f"For your {_family_list(affected)}, {_skew_transform_clause(profile, skewed)}"
                    f"{immune_msg} Configure the transform in this page's numeric "
                    f"transform setting (or in Feature Engineering); there is no "
                    f"one-click resolver for this observation."
                ),
                "manuscript_text": (
                    f"{_count_word_coach(len(skewed), 'predictor')} exhibited "
                    f"strong skewness, which can increase the influence of "
                    f"extreme values in scale-sensitive models"
                ),
                "model_scope": affected,
                "relevant_pages": ["05_Preprocess"],
                "theory_anchor": "skewness",
                "metadata": {"skewed_features": skewed[:10],
                             **_skew_split(profile, skewed)},
            })

    # 2. Outliers → robust scaling or clipping (affects linear, neural, distance)
    outlier_feats = getattr(profile, "features_with_outliers", [])
    if outlier_feats:
        affected = [f for f in ISSUE_MODEL_RELEVANCE["outliers"] if f in user_families]
        immune = [f for f in user_families if f not in ISSUE_MODEL_RELEVANCE["outliers"]]
        if affected:
            immune_msg = ""
            if immune:
                immune_msg = f" Your {_family_list(immune)} are naturally robust to outliers."
            # IQR flags trip easily on small samples — a single extreme point
            # marks a whole feature. Don't let detector noise read as a
            # data-quality crisis.
            n_rows = getattr(profile, "n_rows", None)
            p_total = max(getattr(profile, "n_features", 1), 1)
            detector_caveat = ""
            if n_rows is not None and n_rows < 50 and len(outlier_feats) > 0.3 * p_total:
                detector_caveat = (
                    f" Note: with only {n_rows} rows, IQR-based detection flags "
                    f"features easily — inspect a few of these before winsorizing "
                    f"wholesale."
                )
            insights.append({
                "id": "preprocess_outlier_handling",
                "source_page": "05_Preprocess",
                "category": "preprocessing",
                "severity": "info",
                "finding": (
                    f"{len(outlier_feats)} of {_count_word_coach(p_total, 'feature')} "
                    f"contain outliers "
                    f"({', '.join(outlier_feats[:3])}{'…' if len(outlier_feats) > 3 else ''})."
                ),
                "implication": (
                    f"Outliers can inflate loss and destabilize {_family_list(affected)}."
                ),
                "recommended_action": (
                    f"For your {_family_list(affected)}, consider Winsorizing or "
                    f"robust scaling.{immune_msg}{detector_caveat} Building the "
                    f"pipelines below records whichever outlier handling you "
                    f"configure against this observation."
                ),
                "manuscript_text": (
                    f"{_count_word_coach(len(outlier_feats), 'predictor')} "
                    f"contained values flagged as outlying by the interquartile "
                    f"criterion, which can influence scale-sensitive models"
                ),
                "model_scope": affected,
                "relevant_pages": ["05_Preprocess"],
                "theory_anchor": "outliers",
            })

    # 3. Feature scaling (affects linear, neural, distance, margin)
    scale_affected = [f for f in ISSUE_MODEL_RELEVANCE["feature_scale"] if f in user_families]
    scale_immune = [f for f in user_families if f not in ISSUE_MODEL_RELEVANCE["feature_scale"]]
    if scale_affected:
        immune_msg = ""
        if scale_immune:
            immune_msg = f" Your {_family_list(scale_immune)} are scale-invariant — no scaling needed."
        insights.append({
            "id": "preprocess_feature_scaling",
            "source_page": "05_Preprocess",
            "category": "preprocessing",
            "severity": "info",
            "finding": (
                f"Your {_family_list(scale_affected)} are scale-sensitive; the "
                "app's default pipeline already standardizes features for them."
            ),
            "implication": (
                f"{_family_list(scale_affected)} weight features by magnitude — "
                "unscaled features would bias distance metrics and gradient "
                "updates toward whichever variable has the largest units."
            ),
            "recommended_action": (
                f"Keep scaling ON for {_family_list(scale_affected)} (the default "
                f"does this — only verify it if you customize the pipeline).{immune_msg} "
                f"Nothing to resolve: this states what the pipeline already does."
            ),
            "model_scope": scale_affected,
            "relevant_pages": ["05_Preprocess"],
            # 'scaling' is the key that exists in THEORY_ANCHORS; the demo
            # previously rendered only via an accidental text match
            "theory_anchor": "scaling",
            # Reassurance about a step the app performs — never a limitation.
            "manuscript_text": "",
            "metadata": {"audit_only": True},
        })

    # 4. Missing data — native handling differs by family
    n_missing = getattr(profile, "n_features_with_missing", 0)
    if n_missing > 0 and MODEL_FAMILY_TREE in user_families:
        non_tree = [f for f in user_families if f != MODEL_FAMILY_TREE]
        insights.append({
            "id": "preprocess_missing_tree_native",
            "source_page": "05_Preprocess",
            "category": "preprocessing",
            "severity": "info",
            "finding": (
                f"{_count_word_coach(n_missing, 'feature')} have missing values; "
                f"model families differ in whether they need imputation."
            ),
            "implication": (
                "HistGradientBoosting, LightGBM, XGBoost, and Random Forest "
                "(scikit-learn ≥ 1.4) handle missing values natively. Other "
                "model families require imputation."
            ),
            "recommended_action": (
                "Tree-based models can skip imputation (native NaN support), "
                "though the app's default pipeline imputes for every model so "
                "downstream explainability sees complete data. "
                + (f"For your {_family_list(non_tree)}, imputation is required — "
                   f"median is the robust default; use MICE when missingness "
                   f"exceeds ~5%. "
                   if non_tree else "")
                + "Nothing to resolve here: the missingness itself is reported "
                  "by the EDA ledger, and this note only says how each family "
                  "treats it."
            ),
            "model_scope": [],  # relevant to all, but differentiates
            "relevant_pages": ["05_Preprocess"],
            "theory_anchor": "missing_data",
            # Neutral fact plus family guidance — the missingness limitation
            # itself belongs to the EDA insight that measures it, not here.
            "manuscript_text": "",
            "metadata": {"audit_only": True},
        })

    return insights


# ── Post-Training Diagnostics ──────────────────────────────────────────────

def _model_display_name_coach(key: str) -> str:
    """Return a human-readable model name from the coach's model info dict."""
    info = _get_model_info()
    return info.get(key, {}).get('name', key.upper())


def _resolve_primary_model(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    primary_model: str = "",
) -> Tuple[str, str]:
    """Return (model_key, how_it_was_chosen) for a single-model diagnostic.

    `COACH-003`. The caller's `primary_model` is honored when it names a
    trained model; otherwise the fallback used to be `next(iter(...))` — the
    first model in dict insertion order, i.e. checkbox order — and the finding
    it produced named no model at all. Checkbox order is not a defensible
    basis for a manuscript sentence, so the fallback is now best-by-metric and
    the basis travels with the key so the text can say which model it is about.
    """
    if primary_model and primary_model in model_results:
        return primary_model, "designated primary model"

    metric = "RMSE" if task_type == "regression" else "Accuracy"
    higher_better = task_type != "regression"
    scored = []
    for key, r in model_results.items():
        val = (r or {}).get("metrics", {}).get(metric)
        if val is not None:
            try:
                scored.append((key, float(val)))
            except (TypeError, ValueError):
                continue
    if scored:
        scored.sort(key=lambda kv: kv[1], reverse=higher_better)
        return scored[0][0], f"best {metric}"

    return next(iter(model_results)), "only model with residuals available"


def _detect_prefer_simpler(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
    """Detect when simple models perform within tolerance of complex ones.

    Simple = interpretability 'high' (linear, GLM, Huber, etc.)
    Complex = interpretability 'low' (boosting, neural net, etc.)
    """
    info = _get_model_info()
    simple = {}  # key -> primary metric
    complex_ = {}

    for key, results in model_results.items():
        metrics = results.get('metrics', {})
        model_info = info.get(key, info.get(key.lower(), {}))
        interp = model_info.get('interpretability', 'medium')

        if task_type == 'regression':
            val = metrics.get('RMSE')
        else:
            val = metrics.get('F1', metrics.get('Accuracy'))

        if val is None:
            continue
        if interp == 'high':
            simple[key] = val
        elif interp == 'low':
            complex_[key] = val

    if not simple or not complex_:
        return []

    if task_type == 'regression':
        best_simple_key = min(simple, key=lambda k: simple[k])
        best_complex_key = min(complex_, key=lambda k: complex_[k])
        best_simple_val = simple[best_simple_key]
        best_complex_val = complex_[best_complex_key]
        within_tolerance = best_simple_val <= best_complex_val * (1 + tolerance)
        margin_pct = ((best_simple_val - best_complex_val) / best_complex_val * 100) if best_complex_val else 0
        metric_name = 'RMSE'
    else:
        best_simple_key = max(simple, key=lambda k: simple[k])
        best_complex_key = max(complex_, key=lambda k: complex_[k])
        best_simple_val = simple[best_simple_key]
        best_complex_val = complex_[best_complex_key]
        within_tolerance = best_simple_val >= best_complex_val * (1 - tolerance)
        margin_pct = ((best_complex_val - best_simple_val) / best_complex_val * 100) if best_complex_val else 0
        metric_name = 'F1'

    if not within_tolerance:
        return []

    simple_name = _model_display_name_coach(best_simple_key)
    complex_name = _model_display_name_coach(best_complex_key)

    return [{
        'id': 'train_prefer_simpler',
        'severity': 'warning',
        'finding': (
            f"{simple_name} performed within {abs(margin_pct):.1f}% of {complex_name} "
            f"({metric_name} {best_simple_val:.4f} vs {best_complex_val:.4f}). "
            "A reviewer would question why the more complex model was selected."
        ),
        'implication': (
            "When models perform comparably, parsimony favors the simpler, more "
            "interpretable model. Complex models carry higher risk of overfitting "
            "and are harder to explain in publication."
        ),
        'recommended_action': (
            f"Consider selecting {simple_name} as the primary model, or justify "
            "the complex model's selection based on domain-specific requirements."
        ),
        'manuscript_text': (
            f"the simpler {simple_name} performed within {abs(margin_pct):.1f}% of "
            f"the more complex {complex_name} ({metric_name} {best_simple_val:.4f} "
            f"vs {best_complex_val:.4f}), so parsimony considerations favor the "
            "simpler specification"
        ),
        'model_scope': [],
        'metadata': {
            'simple_best_model': best_simple_key,
            'simple_best_name': simple_name,
            'simple_best_score': float(best_simple_val),
            'complex_best_model': best_complex_key,
            'complex_best_name': complex_name,
            'complex_best_score': float(best_complex_val),
            'margin_pct': float(margin_pct),
            'tolerance_pct': float(tolerance * 100),
            'metric_name': metric_name,
        },
    }]


def _detect_low_overall_performance(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
) -> List[Dict[str, Any]]:
    """Detect when the best model has very low performance, suggesting feature engineering."""
    if not model_results:
        return []

    if task_type == 'regression':
        best_r2 = max(
            (r.get('metrics', {}).get('R2', -1) for r in model_results.values()),
            default=-1,
        )
        if best_r2 >= 0.15 or best_r2 < 0:
            return []
        return [{
            'id': 'train_low_performance',
            'severity': 'opportunity',
            'finding': (
                f"Best model explains only {best_r2 * 100:.1f}% of outcome variance "
                f"(R\u00b2 = {best_r2:.3f}). This suggests the current features capture "
                "limited predictive signal."
            ),
            'implication': (
                "Low R\u00b2 may indicate that important predictors are missing, "
                "that the relationship is non-linear and not captured by current features, "
                "or that the outcome is inherently difficult to predict."
            ),
            'recommended_action': (
                "Return to Feature Engineering to explore interaction terms, non-linear "
                "transforms, or domain-driven composite features. Also consider whether "
                "additional data sources are available."
            ),
            'manuscript_text': (
                f"absolute predictive performance was modest (best R\u00b2 = "
                f"{best_r2:.3f}), indicating that the available predictors capture "
                "a limited share of outcome variance"
            ),
            'model_scope': [],
            'metadata': {'best_r2': float(best_r2)},
        }]
    else:
        best_auc = max(
            (r.get('metrics', {}).get('AUC', 0) for r in model_results.values()),
            default=0,
        )
        if best_auc >= 0.60 or best_auc <= 0:
            return []
        return [{
            'id': 'train_low_performance',
            'severity': 'opportunity',
            'finding': (
                f"Best model achieved AUC of {best_auc:.3f}, indicating weak "
                "discrimination between classes."
            ),
            'implication': (
                "An AUC below 0.60 suggests the model barely outperforms random "
                "guessing. The current features may not capture the decision boundary."
            ),
            'recommended_action': (
                "Return to Feature Engineering to explore interaction terms, "
                "non-linear transforms, or domain-driven composite features."
            ),
            'manuscript_text': (
                f"discriminative performance was weak (best AUC = {best_auc:.3f}), "
                "indicating limited separation between outcome classes with the "
                "available predictors"
            ),
            'model_scope': [],
            'metadata': {'best_auc': float(best_auc)},
        }]


def _detect_high_cv_variance(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
) -> List[Dict[str, Any]]:
    """Detect when CV variance is large relative to inter-model performance gaps."""
    cv_stds = []
    scores = []
    for key, results in model_results.items():
        cv = results.get('cv_results')
        if cv and cv.get('std') is not None:
            cv_stds.append(cv['std'])
        metrics = results.get('metrics', {})
        if task_type == 'regression':
            val = metrics.get('RMSE')
        else:
            val = metrics.get('F1', metrics.get('Accuracy'))
        if val is not None:
            scores.append(val)

    if len(cv_stds) < 1 or len(scores) < 2:
        return []

    max_cv_std = max(cv_stds)
    score_range = max(scores) - min(scores)

    if score_range <= 0 or max_cv_std < score_range * 0.5:
        return []

    return [{
        'id': 'train_cv_variance',
        'severity': 'info',
        'finding': (
            f"Cross-validation variability (max std = {max_cv_std:.4f}) exceeds "
            f"half the inter-model performance range ({score_range:.4f}). "
            "Model ranking may not be stable."
        ),
        'implication': (
            "When evaluation noise is large relative to model differences, "
            "the apparent best model may change with different random splits. "
            "A reviewer would question the robustness of model selection."
        ),
        'recommended_action': (
            "Run Sensitivity Analysis (seed robustness) to verify that model "
            "rankings are stable across random seeds."
        ),
        'manuscript_text': (
            f"cross-validation variability (maximum fold SD = {max_cv_std:.4f}) "
            f"exceeded half the between-model performance range "
            f"({score_range:.4f}), so the model ranking should be interpreted "
            "with caution"
        ),
        'model_scope': [],
        'metadata': {
            'max_cv_std': float(max_cv_std),
            'score_range': float(score_range),
        },
    }]


def _detect_overfit(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    gap_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """Detect when train performance significantly exceeds test performance.

    For regression, compares Train R² vs Test R².
    For classification, compares Train F1 vs Test F1.
    Flags models where the gap exceeds ``gap_threshold``.
    """
    info = _get_model_info()
    findings = []

    for key, results in model_results.items():
        train_m = results.get('train_metrics', {})
        test_m = results.get('metrics', {})
        if not train_m:
            continue

        if task_type == 'regression':
            train_val = train_m.get('R2')
            test_val = test_m.get('R2')
            metric_name = 'R²'
        else:
            train_val = train_m.get('F1', train_m.get('Accuracy'))
            test_val = test_m.get('F1', test_m.get('Accuracy'))
            metric_name = 'F1' if 'F1' in train_m else 'Accuracy'

        if train_val is None or test_val is None:
            continue

        gap = train_val - test_val
        if gap <= gap_threshold:
            continue

        display_name = _model_display_name_coach(key)
        # Use the ledger's canonical family vocabulary — the coach's display
        # groups ('Trees', 'Boosting', 'Neural Net') don't match it, and a
        # non-matching model_scope makes grouped coaching silently hide the
        # warning behind '✅ no issues'.
        from utils.insight_ledger import MODEL_TO_FAMILY as _mtf
        _fam = _mtf.get(key, _mtf.get(str(key).lower()))
        family_scope = [_fam] if _fam else []

        findings.append({
            'id': f'train_overfit_{key}',
            'severity': 'warning',
            'finding': (
                f"{display_name} shows signs of overfitting: train {metric_name} = "
                f"{train_val:.3f} vs test {metric_name} = {test_val:.3f} "
                f"(gap: {gap:.3f}). The model memorizes training data patterns "
                "that don't generalize."
            ),
            'implication': (
                "Overfitting inflates apparent performance. A reviewer would note "
                "the train/test discrepancy and question whether the model is "
                "learning signal or noise."
            ),
            'recommended_action': (
                f"Consider regularizing {display_name} (increase regularization "
                "strength, reduce model complexity, or add dropout). Alternatively, "
                "use a simpler model or collect more training data."
            ),
            'manuscript_text': (
                f"{display_name} showed a marked train–test performance gap "
                f"({metric_name} {train_val:.3f} vs {test_val:.3f}), indicating "
                "overfitting risk for this model"
            ),
            'model_scope': family_scope,
            'metadata': {
                'model_key': key,
                'model_name': display_name,
                'train_score': float(train_val),
                'test_score': float(test_val),
                'gap': float(gap),
                'metric_name': metric_name,
            },
        })

    return findings




def _detect_accuracy_vs_nir(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
) -> List[Dict[str, Any]]:
    """Accuracy must be judged against the no-information rate (NIR): the
    accuracy of always predicting the majority class. A model at or below
    the NIR has learned nothing that plain prevalence doesn't already give."""
    import numpy as np

    if task_type != "classification" or not model_results:
        return []
    y_test = None
    for r in model_results.values():
        yt = r.get("y_test")
        if yt is not None and len(yt) > 0:
            y_test = np.asarray(yt)
            break
    if y_test is None:
        return []
    _, counts = np.unique(y_test, return_counts=True)
    nir = float(counts.max() / counts.sum())

    best_key, best_acc = None, -1.0
    for key, r in model_results.items():
        acc = r.get("metrics", {}).get("Accuracy")
        if acc is not None and acc > best_acc:
            best_key, best_acc = key, float(acc)
    if best_key is None or best_acc > nir + 0.02:
        return []

    name = _model_display_name_coach(best_key)
    return [{
        'id': 'train_accuracy_below_nir',
        'severity': 'warning',
        'finding': (
            f"Best accuracy ({best_acc:.3f}, {name}) does not beat the "
            f"no-information rate ({nir:.3f} — always predicting the majority "
            f"class). The models have not learned beyond prevalence."
        ),
        'implication': (
            "Accuracy near the NIR usually means weak signal or an "
            "uninformative feature set. AUROC and F1 tell the real story."
        ),
        'recommended_action': (
            "Judge models by AUROC/F1, revisit predictors, and consider the "
            "evidence probe on the Preprocess page."
        ),
        'manuscript_text': (
            f"classification accuracy ({best_acc:.3f}) did not exceed the "
            f"no-information rate ({nir:.3f}), indicating limited discriminative "
            f"value beyond class prevalence"
        ),
        'model_scope': [],
        'metadata': {'nir': nir, 'best_accuracy': best_acc, 'best_model': best_key},
    }]


def _detect_ci_overlap(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    bootstrap_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """When bootstrap CIs exist, use them: if the top two models' intervals
    on the primary metric overlap substantially, the ranking is not
    established and the simpler/preferred model needs no apology."""
    if not bootstrap_results or len(model_results) < 2:
        return []
    metric = "RMSE" if task_type == "regression" else "F1"
    higher_better = task_type != "regression"

    scored = []
    for key, r in model_results.items():
        val = r.get("metrics", {}).get(metric)
        if val is not None:
            scored.append((key, float(val)))
    if len(scored) < 2:
        return []
    scored.sort(key=lambda kv: kv[1], reverse=higher_better)
    (k1, v1), (k2, v2) = scored[0], scored[1]

    def _ci(key):
        cis = bootstrap_results.get(key) or {}
        ci = cis.get(metric)
        lo = getattr(ci, "ci_lower", None) if ci is not None else None
        hi = getattr(ci, "ci_upper", None) if ci is not None else None
        if lo is None and isinstance(ci, dict):
            lo, hi = ci.get("ci_lower"), ci.get("ci_upper")
        return (lo, hi)

    lo1, hi1 = _ci(k1)
    lo2, hi2 = _ci(k2)
    if None in (lo1, hi1, lo2, hi2):
        return []
    import math
    if any(isinstance(x, float) and math.isnan(x) for x in (lo1, hi1, lo2, hi2)):
        return []
    overlap = min(hi1, hi2) - max(lo1, lo2)
    if overlap <= 0:
        return []

    n1, n2 = _model_display_name_coach(k1), _model_display_name_coach(k2)
    return [{
        'id': 'train_ci_overlap_top_models',
        'severity': 'info',
        'finding': (
            f"The 95% bootstrap CIs of the top two models overlap on {metric}: "
            f"{n1} [{lo1:.3f}, {hi1:.3f}] vs {n2} [{lo2:.3f}, {hi2:.3f}]. "
            f"The ranking between them is not established."
        ),
        'implication': (
            "With overlapping intervals, choosing on the point estimate alone "
            "over-interprets noise; parsimony and interpretability are valid "
            "tie-breakers."
        ),
        'recommended_action': (
            f"Feel free to prefer the simpler of {n1}/{n2}; report both CIs."
        ),
        'manuscript_text': (
            f"the bootstrap confidence intervals of the two best-performing "
            f"models overlapped on {metric} ({n1}: {lo1:.3f}\u2013{hi1:.3f}; "
            f"{n2}: {lo2:.3f}\u2013{hi2:.3f}), so their ranking should not be "
            f"over-interpreted"
        ),
        'model_scope': [],
        'metadata': {'metric': metric, 'top': k1, 'second': k2},
    }]


def _detect_no_bootstrap_ci(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    bootstrap_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Before any metric is reported, say that it has no interval yet.

    The coach's only CI check (`_detect_ci_overlap`) runs on intervals that
    already exist, so the case that matters most — a point estimate about to
    be written into a manuscript with no uncertainty attached at all — had no
    detector. This is the one that fires when there are no intervals yet.
    """
    if bootstrap_results:
        return []
    scored = [k for k, r in model_results.items()
              if (r or {}).get("metrics")]
    if not scored:
        return []

    metric = "RMSE" if task_type == "regression" else "F1"
    n_txt = "1 model" if len(scored) == 1 else f"{len(scored)} models"
    rank_clause = (
        " and the ranking between them rests on point estimates alone"
        if len(scored) > 1 else ""
    )
    return [{
        'id': 'train_no_bootstrap_ci',
        'severity': 'info',
        'finding': (
            f"{n_txt} trained, and no bootstrap confidence intervals have been "
            f"computed for the reported metrics{rank_clause}."
        ),
        'implication': (
            f"A single test split gives one draw of {metric}; without an "
            f"interval the number carries no visible sampling error, and a "
            f"difference small enough to be noise reads as a real one."
        ),
        'recommended_action': (
            "Run the bootstrap confidence intervals section on this page "
            "before reporting any metric or model ranking."
        ),
        # No manuscript_text, and audit-only: this is a prompt for an action
        # still available in the app, not a finding about the study. Nothing
        # re-runs the post-training detectors after the intervals are computed,
        # so a Discussion sentence saying "no confidence intervals were
        # computed" could be false by the time the report is exported.
        'manuscript_text': '',
        'model_scope': [],
        'metadata': {'audit_only': True, 'models_without_ci': scored},
    }]


def _detect_heteroscedastic_residuals(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    primary_model: str = "",
) -> List[Dict[str, Any]]:
    """Residual spread growing with the predicted value means single-width
    prediction intervals are wrong and a target transform likely helps."""
    import numpy as np

    if task_type != "regression" or not model_results:
        return []
    key, chosen_by = _resolve_primary_model(model_results, task_type, primary_model)
    r = model_results.get(key) or {}
    y_test, y_pred = r.get("y_test"), r.get("y_test_pred")
    if y_test is None or y_pred is None:
        return []
    y_test, y_pred = np.asarray(y_test, dtype=float), np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_test) & np.isfinite(y_pred)
    if ok.sum() < 20:
        return []
    resid = np.abs(y_test[ok] - y_pred[ok])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(resid, y_pred[ok])
    if not np.isfinite(rho) or abs(rho) < 0.3:
        return []

    name = _model_display_name_coach(key)
    direction = "grows" if rho > 0 else "shrinks"
    return [{
        'id': 'train_heteroscedastic_residuals',
        'severity': 'info',
        'finding': (
            f"{name}'s residual spread {direction} with the predicted value "
            f"(Spearman \u03c1 = {rho:.2f} between |residual| and prediction). "
            f"Checked on {name} ({chosen_by}); other trained models are not "
            f"covered by this check."
        ),
        'implication': (
            "Errors are not uniform across the outcome range: constant-width "
            "prediction intervals will be miscalibrated, and mean-based "
            "metrics understate errors at one end."
        ),
        'recommended_action': (
            "Consider a target transform (log / Yeo-Johnson) on the Train page "
            "and inspect the Bland\u2013Altman plot on Explainability."
        ),
        # The model is named here, not only in the `finding` above: the
        # manuscript sentence used to assert non-constant residual variance
        # with no attribution at all, about whichever model came first in
        # checkbox order (`COACH-003`).
        'manuscript_text': (
            f"for the {name} model, residual variance was not constant across "
            f"the predicted range (Spearman \u03c1 = {rho:.2f} between absolute "
            f"residuals and predictions), so uniform-width prediction intervals "
            f"would be miscalibrated"
        ),
        'model_scope': [],
        'metadata': {'rho': float(rho), 'model': key, 'model_chosen_by': chosen_by},
    }]


def run_post_training_diagnostics(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str,
    tolerance: float = 0.05,
    bootstrap_results: Optional[Dict[str, Any]] = None,
    primary_model: str = "",
) -> List[Dict[str, Any]]:
    """Run all post-training diagnostic checks and return a list of findings.

    Each finding is a dict with keys: id, severity, finding, implication,
    recommended_action, manuscript_text, model_scope, metadata.
    """
    findings = []
    findings.extend(_detect_prefer_simpler(model_results, task_type, tolerance))
    findings.extend(_detect_low_overall_performance(model_results, task_type))
    findings.extend(_detect_high_cv_variance(model_results, task_type))
    findings.extend(_detect_overfit(model_results, task_type))
    findings.extend(_detect_accuracy_vs_nir(model_results, task_type))
    findings.extend(_detect_ci_overlap(model_results, task_type, bootstrap_results))
    findings.extend(_detect_no_bootstrap_ci(model_results, task_type, bootstrap_results))
    findings.extend(_detect_heteroscedastic_residuals(model_results, task_type, primary_model))
    return findings


# ── Full-registry viability verdicts ──────────────────────────────────────

def _nn_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def realized_training_n() -> Optional[int]:
    """How many rows a model fitted in this session will actually see.

    `profile.n_rows` counts every row the dataset profile was computed on —
    which includes the rows with no outcome value, and those never reach a
    fit. On a 21,849-row upload with a 71%-missing outcome the profile said
    20,904 while the training set was 4,407, and the badges below refused SVC
    ("kernel training scales ~n² — slow at n=20,904") on a size the run did
    not have (`DRIVE-070`). The split's own training matrix is the only place
    that number is realized, so it is read from there when it exists.

    Returns None outside a Streamlit session or before splits are prepared;
    the caller then falls back to the profile.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        X_train = st.session_state.get("X_train")
    except Exception:
        return None
    if X_train is None:
        return None
    try:
        n = int(len(X_train))
    except TypeError:
        return None
    return n if n > 0 else None


def model_viability(profile: Any, probe: Any = None,
                    n_train: Optional[int] = None) -> Dict[str, Tuple[str, str]]:
    """One evidence-bearing verdict per registry model key.

    Returns {model_key: (verdict, clause)} with verdict in
    {"good", "ok", "poor"}. Rendered under each model card on the Train
    page so the shape reasoning is visible at the exact moment of choice.
    The clause cites the dataset's numbers; probe evidence sharpens the
    tree/boosting clauses when available.

    `n_train` is the realized training-set size. Every verdict below is a
    claim about how much data the model will be FITTED on, so when the split
    is drawn that is the number, not the profile's row count. Callers that
    know it pass it; the rest get it from the session (`realized_training_n`).
    """
    _n_realized = n_train if n_train is not None else realized_training_n()
    n = int(_n_realized) if _n_realized else profile.n_rows
    p = profile.n_features
    tp = profile.target_profile
    task_type = tp.task_type if tp else "regression"
    # p/n has to follow the same n, or a clause reads "p=200 > n=4,407".
    _p_n = (p / n) if (_n_realized and n > 0) else profile.p_n_ratio
    is_wide = _p_n > 1.0
    is_high_dim = _p_n > 0.3
    epv = profile.events_per_variable
    minority = tp.minority_class_size if tp else None
    # `AUDIT-021` — the app's caution trigger, named as that. See `select_top_picks`.
    low_epv = (task_type == "classification" and epv is not None
               and epv < _ss.CAUTION_EPV)
    target_outliers = bool(tp and tp.has_outliers and (tp.outlier_rate or 0) > 0.01)

    _gain = probe.nonlinearity_gain if probe is not None else None
    _signal = probe.has_signal if probe is not None else None
    probe_tree_note = ""
    if _signal and _gain is not None:
        if _gain > 0.04:
            probe_tree_note = f" (probe: +{_gain:.2f} {probe.metric_name} over linear)"
        elif _gain < 0.02:
            probe_tree_note = f" (probe: no gain over linear measured)"

    v: Dict[str, Tuple[str, str]] = {}

    def penalized_linear():
        if is_wide:
            return ("good", f"penalization keeps the fit identifiable at p={p:,} > n={n:,}")
        return ("good", "stable, interpretable core model for this shape")

    v["ridge"] = penalized_linear()
    v["lasso"] = penalized_linear()
    v["elasticnet"] = penalized_linear()

    if is_wide:
        v["glm"] = ("poor", f"unpenalized fit is not identifiable at p={p:,} ≥ n={n:,}")
    elif _p_n > 0.5:
        v["glm"] = ("poor", f"unpenalized estimates are unstable at p/n = {_p_n:.2f}")
    elif n < 30:
        v["glm"] = ("poor", f"n={n} is below a defensible minimum for unpenalized fits")
    else:
        v["glm"] = ("ok", "fine as a classical reference; penalized variants are safer")

    if task_type == "regression" and target_outliers:
        v["huber"] = ("good", f"outcome has outliers ({tp.outlier_rate:.0%} of values) — robust loss pays")
    elif is_wide:
        v["huber"] = ("poor", "no penalty — not identifiable at p≫n")
    else:
        v["huber"] = ("ok", "only pays when the outcome itself has outliers — yours looks clean")

    if low_epv:
        v["logreg"] = ("good", f"the defensible core at EPV={epv:.1f} — penalized, expect wide CIs")
        v["lda"] = ("poor", f"covariance estimates unreliable at EPV={epv:.1f}")
        v["gaussian_nb"] = ("ok", "low capacity is tolerable at this EPV; strong independence assumption")
    else:
        v["logreg"] = ("good", "interpretable probability baseline")
        v["lda"] = ("ok", "different lens than logistic regression; assumes Gaussian classes")
        v["gaussian_nb"] = ("ok", "very fast; assumes feature independence")

    for k in ("knn_reg", "knn_clf"):
        if is_high_dim or is_wide:
            v[k] = ("poor", f"distances lose meaning at p={p:,}")
        elif p > 20:
            v[k] = ("ok", f"adds little over trees at p={p}")
        else:
            v[k] = ("ok", "simple non-parametric reference")

    def tree_family(min_n, name_hint):
        if is_wide:
            return ("poor", f"memorizes rather than generalizes at p={p:,} > n={n:,}")
        if low_epv:
            return ("poor", f"EPV={epv:.1f} — high capacity overfits hardest "
                            f"where events per candidate parameter are lowest")
        if n < min_n:
            return ("poor", f"needs roughly {min_n}+ rows; you have {n:,}")
        if n < 150:
            return ("ok", f"viable at n={n}, but expect fold-to-fold ranking noise")
        return ("good", f"strong non-linear learner for this shape{name_hint}")

    v["rf"] = tree_family(50, probe_tree_note)
    v["extratrees_reg"] = tree_family(50, probe_tree_note)
    v["extratrees_clf"] = tree_family(50, probe_tree_note)
    for k in ("histgb_reg", "histgb_clf", "xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf"):
        v[k] = tree_family(100, probe_tree_note)

    for k in ("svr", "svc"):
        if n > 5000:
            v[k] = ("poor", f"kernel training scales ~n² — slow at n={n:,}")
        elif is_wide:
            v[k] = ("ok", "kernels tolerate p>n, but results are hard to interpret")
        else:
            v[k] = ("ok", "try only if the linear baseline underfits")

    if n < 500 or low_epv:
        # `AUDIT-021`: this used to read "needs roughly 500+ rows and EPV≥10",
        # which states the superseded rule as a requirement. The row count is
        # the app's own rule of thumb and says so; the EPV is reported as the
        # measured quantity it is, with no threshold attached.
        v["nn"] = ("poor",
                   f"more capacity than this shape supports — a rough floor of "
                   f"500 rows; you have {n:,}"
                   + (f", at EPV={epv:.1f}" if low_epv else ""))
    elif n < 1000:
        v["nn"] = ("ok", f"borderline at n={n:,} — regularize heavily")
    else:
        v["nn"] = ("good", f"n={n:,} supports the capacity")

    return v
