"""
Outcome-specific narrative for result plots.
Maps computed stats to human-readable, granular narrative blocks.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional


def narrative_residuals(stats: Dict[str, Any], model_name: Optional[str] = None) -> str:
    """Narrative for residual plot from analyze_residuals_extended stats (scientific-inquiry tone)."""
    if not stats:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    sk = stats.get("skew", 0)
    mean_r = stats.get("mean_residual", 0)
    iqr = stats.get("iqr", 0)
    rvp = stats.get("residual_vs_predicted_corr", 0)
    parts: List[str] = []
    if abs(sk) > 0.5:
        direction = "right" if sk > 0 else "left"
        parts.append(
            f"These residuals suggest {direction}-skew (skew={sk:.2f}); the pattern implies "
            f"{'under' if sk > 0 else 'over'}-prediction for {'low' if sk > 0 else 'high'} values."
        )
    else:
        parts.append("The residual distribution appears roughly symmetric.")
    if abs(mean_r) > 1e-6:
        parts.append(f"Consider whether systematic bias is meaningful: mean residual {mean_r:.4f}.")
    if iqr > 0:
        parts.append(f"Spread (IQR) is {iqr:.4f}.")
    if abs(rvp) > 0.2:
        parts.append(f"Residuals correlate with predicted values (r={rvp:.2f}); investigate heteroscedasticity.")
    return pref + " ".join(parts) if parts else ""


def narrative_pred_vs_actual(stats: Dict[str, Any], model_name: Optional[str] = None) -> str:
    """Narrative for predictions-vs-actual plot (scientific-inquiry tone)."""
    if not stats:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    corr = stats.get("correlation", 0)
    bias_q = stats.get("bias_by_quintile", [])
    max_over = stats.get("max_overprediction", 0)
    max_under = stats.get("max_underprediction", 0)
    parts: List[str] = []
    if abs(corr) < 0.3:
        parts.append("The pattern suggests weak agreement; consider whether model or features can be improved.")
    elif corr >= 0.9:
        parts.append("Predictions track actuals closely—the model captures the main signal.")
    else:
        parts.append(f"Correlation between predictions and actuals is {corr:.2f}; interpret in light of your application.")
    if bias_q:
        worst_q = max(range(5), key=lambda i: abs(bias_q[i]) if i < len(bias_q) else 0)
        b = bias_q[worst_q] if worst_q < len(bias_q) else 0
        if abs(b) > 1e-6:
            q_label = ["lowest", "low", "mid", "high", "highest"][worst_q]
            parts.append(f"Consider whether bias in the {q_label} quintile (mean error {b:.4f}) matters for deployment.")
    if abs(max_over) > 1e-6 or abs(max_under) > 1e-6:
        parts.append(f"Max over-prediction {max_over:.4f}; max under-prediction {max_under:.4f}.")
    return pref + " ".join(parts) if parts else ""


def narrative_residuals_stratified(stats: Dict[str, Any], model_name: Optional[str] = None) -> str:
    """Narrative for stratified residual analysis (scientific-inquiry tone)."""
    bins = stats.get("bins", [])
    if not bins:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    parts: List[str] = []

    over_bins = [b for b in bins if b["bias_direction"] == "over" and b["n"] > 0]
    under_bins = [b for b in bins if b["bias_direction"] == "under" and b["n"] > 0]
    worst = max(bins, key=lambda b: abs(b["mean_bias"])) if bins else None

    if over_bins and under_bins:
        parts.append(
            "The model exhibits mixed bias across the target range — "
            "over-predicting in some regions and under-predicting in others."
        )
    elif over_bins:
        parts.append("The model tends to over-predict across the target range.")
    elif under_bins:
        parts.append("The model tends to under-predict across the target range.")
    else:
        parts.append("Predictions appear unbiased across target-value bins.")

    if worst and worst["n"] > 0 and abs(worst["mean_bias"]) > 1e-6:
        parts.append(
            f"The largest bias occurs in the {worst['range']} range "
            f"(mean error {worst['mean_bias']:+.4f}, n={worst['n']})."
        )

    mae_vals = [b["mae"] for b in bins if b["n"] > 0]
    if mae_vals and max(mae_vals) > 2 * min(mae_vals):
        best_bin = min(bins, key=lambda b: b["mae"] if b["n"] > 0 else float("inf"))
        worst_mae = max(bins, key=lambda b: b["mae"] if b["n"] > 0 else 0)
        parts.append(
            f"Error varies substantially: MAE ranges from {best_bin['mae']:.4f} "
            f"({best_bin['range']}) to {worst_mae['mae']:.4f} ({worst_mae['range']})."
        )

    return pref + " ".join(parts) if parts else ""


def narrative_confusion_matrix(stats: Dict[str, Any], model_name: Optional[str] = None) -> str:
    """Narrative for confusion matrix."""
    if not stats:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    per = stats.get("per_class", [])
    top = stats.get("top_confusions", [])
    parts: List[str] = []
    for p in per[:5]:
        lab = p.get("label", "?")
        prec = p.get("precision", 0)
        rec = p.get("recall", 0)
        if prec < 0.5 or rec < 0.5:
            parts.append(f"Class {lab} has precision {prec:.2f} and recall {rec:.2f}; consider whether this aligns with deployment needs.")
    if top:
        c, t, p = top[0]
        parts.append(f"The most frequent confusion: true {t} predicted as {p} ({c} samples).")
    return pref + " ".join(parts) if parts else ""


def narrative_bland_altman(
    stats: Dict[str, Any], label_a: str = "A", label_b: str = "B"
) -> str:
    """Narrative for Bland–Altman plot."""
    if not stats:
        return ""
    mean_d = stats.get("mean_diff", 0)
    w = stats.get("width_loa", 0)
    pct = stats.get("pct_outside_loa", 0)
    parts: List[str] = []
    if abs(mean_d) > 1e-6:
        parts.append(f"The pattern suggests systematic bias: {label_a} − {label_b} = {mean_d:.4f} on average.")
    parts.append(f"Limits of agreement span {w:.4f}.")
    if pct > 0.05:
        parts.append(f"Consider whether {pct:.1%} of points outside LoA matters for your use case.")
    else:
        parts.append("Most points lie within the limits of agreement.")
    return " ".join(parts)


def interpretation_permutation_importance() -> str:
    """Short guidance: what the numbers mean (literature-based)."""
    return (
        "Importance = drop in metric (e.g. RMSE, accuracy) when the feature is shuffled; same units as the metric. "
        "Values near zero mean little predictive power; larger values indicate the feature matters for predictions. "
        "Sklearn/Altmann et al."
    )


def interpretation_shap() -> str:
    """Short guidance: what mean |SHAP| means."""
    return (
        "Mean |SHAP| is in prediction units; larger values = larger impact on the model output. "
        "Lundberg & Lee (2017); use relative magnitudes across features, not absolute thresholds."
    )


def interpretation_partial_dependence() -> str:
    """Short guidance: what PD axes mean."""
    return (
        "Y-axis = average prediction; x-axis = feature value. Scale matches model output. "
        "Flat lines suggest weak marginal effect; nonlinear shapes suggest transformations or interactions."
    )


def interpretation_bland_altman() -> str:
    """Short guidance: what LoA and limits mean."""
    return (
        "Mean difference = systematic bias; limits of agreement (LoA) = mean ± 1.96 SD of differences. "
        "Most points within LoA suggests good agreement; points outside indicate larger discrepancies."
    )


def interpretation_robustness() -> str:
    """Short guidance: what Spearman ρ and top-k overlap mean."""
    return (
        "Spearman ρ compares feature importance **ranks** across models; higher ρ = more agreement. "
        "Top-k overlap = how many top features are shared; high overlap suggests robust interpretations."
    )


def narrative_permutation_importance(
    perm_data: Dict[str, Any], model_name: Optional[str] = None
) -> str:
    """Narrative for permutation importance."""
    if not perm_data:
        return ""
    imp = perm_data.get("importances_mean", [])
    fnames = perm_data.get("feature_names", [])
    if len(imp) == 0 or not fnames:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    order = sorted(range(len(imp)), key=lambda i: -imp[i])
    top_idx = order[0]
    top_name = fnames[top_idx] if top_idx < len(fnames) else f"feature {top_idx}"
    top_val = imp[top_idx]
    second = imp[order[1]] if len(order) > 1 else 0.0
    dominance = (top_val / second) if second and second > 0 else 10.0
    parts: List[str] = []
    parts.append(f"Top driver: **{top_name}** (importance {top_val:.4f}).")
    if dominance < 2 and len(imp) > 1:
        parts.append("The pattern implies importance is spread across several features; consider multicollinearity or diffuse signal.")
    elif dominance >= 3:
        parts.append("This feature dominates; others contribute less—interpret in light of causality and leakage.")
    return pref + " ".join(parts)


def narrative_shap(
    shap_values: Any,
    feature_names: List[str],
    model_name: Optional[str] = None,
) -> str:
    """Narrative for SHAP summary: top positive/negative drivers."""
    import numpy as np

    if shap_values is None or not feature_names:
        return ""
    arr = np.asarray(shap_values)
    if arr.size == 0:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    # mean |SHAP| per feature
    mean_abs = np.mean(np.abs(arr), axis=0)
    n = min(len(feature_names), len(mean_abs))
    order = np.argsort(mean_abs)[::-1][:n]
    top_idx = order[0]
    top_name = feature_names[top_idx] if top_idx < len(feature_names) else f"feature {top_idx}"
    top_val = float(mean_abs[top_idx])
    parts: List[str] = []
    parts.append(f"SHAP suggests **{top_name}** as the top driver (mean |SHAP| {top_val:.4f}).")
    if len(order) > 1:
        second_idx = order[1]
        second_name = feature_names[second_idx] if second_idx < len(feature_names) else f"feature {second_idx}"
        parts.append(f"Next: **{second_name}**; consider whether direction and magnitude align with domain knowledge.")
    return pref + " ".join(parts)


def narrative_partial_dependence(
    pd_data: Dict[str, Any],
    model_name: Optional[str] = None,
) -> str:
    """Narrative for partial dependence plots: how features affect predictions on average."""
    if not pd_data:
        return ""
    pref = f"**{model_name or 'Model'}:** " if model_name else ""
    feats = list(pd_data.keys())[:5]
    parts: List[str] = []
    parts.append(
        f"Partial dependence for {', '.join(feats)}{'…' if len(pd_data) > 5 else ''} "
        "shows how each feature affects predictions on average. "
        "Flat lines suggest little marginal effect; nonlinear shapes suggest transformations or interactions."
    )
    return pref + " ".join(parts)


def narrative_learning_curves(history: Dict[str, List[float]]) -> str:
    """Narrative for learning curves (train/val loss)."""
    if not history:
        return ""
    tl = history.get("train_loss", [])
    vl = history.get("val_loss", [])
    if not tl or not vl:
        return ""
    last_t = tl[-1]
    last_v = vl[-1]
    gap = last_v - last_t
    parts: List[str] = []
    if gap > 0.1 * last_t:
        parts.append("The pattern suggests overfitting: validation loss notably exceeds training loss; consider regularization or early stopping.")
    elif gap < -0.1 * last_t:
        parts.append("Training loss exceeds validation loss; consider whether this reflects underfitting or validation set characteristics.")
    else:
        parts.append("Train and validation losses are close—the fit appears balanced.")
    if len(tl) >= 2 and tl[-1] < tl[0] * 0.5:
        parts.append("Loss decreased substantially over training.")
    return " ".join(parts)


def narrative_robustness(
    robustness: Dict[str, Any],
    model_pairs: Optional[List[tuple]] = None,
) -> str:
    """1–2 sentence interpretation of Spearman / top-k overlap from compare_importance_ranks."""
    if not robustness or not isinstance(robustness, dict):
        return ""
    pairs = model_pairs or list(robustness.keys())
    if not pairs:
        return ""
    parts: List[str] = []
    rhos: List[float] = []
    overlaps: List[float] = []
    for k in pairs:
        v = robustness.get(k)
        if not v or not isinstance(v, dict):
            continue
        r = v.get("spearman")
        o = v.get("top_k_overlap")
        if r is not None:
            rhos.append(float(r))
        if o is not None:
            overlaps.append(float(o))
    if rhos:
        mean_r = sum(rhos) / len(rhos)
        if mean_r >= 0.8:
            parts.append(
                "Feature importance ranks agree well across models (high Spearman ρ); "
                "interpretations are relatively robust."
            )
        elif mean_r >= 0.5:
            parts.append(
                "Moderate agreement in importance ranks across models; "
                "consider which model’s interpretation best fits your domain."
            )
        else:
            parts.append(
                "Importance ranks differ across models; "
                "consider whether preprocessing, regularization, or model family drives the discrepancy."
            )
    if overlaps:
        mean_o = sum(overlaps) / len(overlaps)
        parts.append(f"Top-k overlap across pairs averages {mean_o:.1f}.")
    return " ".join(parts)


def narrative_eda_linearity(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for linearity scatter EDA."""
    if not stats and not findings:
        return ""
    corrs = stats.get("feature_correlations") if isinstance(stats, dict) else []
    tests = stats.get("correlation_tests") if isinstance(stats, dict) else []
    parts: List[str] = []
    if corrs:
        top = max(corrs, key=lambda x: x[1]) if corrs else None
        if top:
            fname, r = top[0], top[1]
            p_str = ""
            if tests:
                for t in tests:
                    if t[0] == fname and len(t) >= 3 and t[2] is not None:
                        try:
                            pv = float(t[2])
                            if pv == pv:
                                p_str = f" (p={pv:.4f})"
                        except (TypeError, ValueError):
                            pass
                        break
            if r >= 0.7:
                parts.append(f"Strong linear association for {fname} (|r|≈{r:.2f}){p_str}; linear models may capture this well.")
            elif r >= 0.3:
                parts.append(f"Moderate association for {fname}{p_str}; consider transformations or flexible terms if fits are poor.")
            else:
                parts.append("Weak linear associations across top features; consider nonlinear or tree-based models.")
    if not parts and findings:
        parts.append(" ".join(findings[:2]))
    return " ".join(parts) if parts else ""


def narrative_eda_residuals(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Reuse residual narrative for OLS-proxy residual analysis."""
    if stats:
        return narrative_residuals(stats, model_name=None)
    if findings:
        return " ".join(findings[:2])
    return ""


def narrative_eda_influence(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for influence diagnostics (leverage, Cook's D)."""
    if not stats and not findings:
        return ""
    parts: List[str] = []
    n_high = (stats or {}).get("n_high_cooks", 0)
    max_cooks = (stats or {}).get("max_cooks", 0)
    if n_high and n_high > 0:
        parts.append(f"{n_high} point(s) have high influence (Cook's D > 1); consider reviewing them or robust regression.")
    elif max_cooks > 0.5:
        parts.append(f"Max Cook's D = {max_cooks:.2f}; a few points may drive coefficient estimates.")
    else:
        parts.append("No strongly influential points detected; OLS estimates are likely stable.")
    return " ".join(parts) if parts else (" ".join((findings or [])[:2]) or "")


def narrative_eda_normality(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for normality-of-residuals check."""
    if not stats and not findings:
        return ""
    p = (stats or {}).get("shapiro_p")
    parts: List[str] = []
    if p is not None:
        if p < 0.05:
            parts.append("Residuals deviate from normality (Shapiro–Wilk p < 0.05); inference (CIs, p-values) may be approximate.")
        else:
            parts.append("Residuals are approximately normal; standard inference for linear models is reasonable.")
    if not parts and findings:
        parts.append(" ".join(findings[:2]))
    return " ".join(parts) if parts else ""


def narrative_eda_sufficiency(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for data sufficiency check."""
    if not stats and not findings:
        return ""
    ratio = (stats or {}).get("ratio", 0)
    n, p = (stats or {}).get("n_rows", 0), (stats or {}).get("n_features", 0)
    parts: List[str] = []
    if ratio >= 20:
        parts.append(f"Sample size (n/p≈{ratio:.0f}) is adequate for many models; regularization still recommended for high p.")
    elif ratio >= 10:
        parts.append(f"Moderate n/p ({ratio:.1f}); prefer regularized or simple models and avoid overfitting.")
    else:
        parts.append(f"Limited n/p ({ratio:.1f}); use strong regularization, few features, or collect more data.")
    return " ".join(parts) if parts else (" ".join((findings or [])[:2]) or "")


def narrative_eda_scaling(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for feature-scaling check."""
    if not stats and not findings:
        return ""
    if findings:
        return " ".join(findings[:2])
    return "Features vary in scale; scaling (e.g. StandardScaler) is recommended for GLM, NN, and distance-based models."


def narrative_eda_multicollinearity(stats: Dict[str, Any], findings: Optional[List[str]] = None) -> str:
    """Interpretation for VIF / multicollinearity check."""
    if not stats and not findings:
        return ""
    vifs = (stats or {}).get("vif", [])
    high = [(c, v) for c, v in vifs if v > 10]
    parts: List[str] = []
    if high:
        names = [c for c, _ in high[:3]]
        parts.append(f"VIF > 10 for {', '.join(names)}{'…' if len(high) > 3 else ''}; consider regularization or dropping correlated features.")
    else:
        parts.append("No severe multicollinearity (VIF ≤ 10); coefficient stability is reasonable.")
    return " ".join(parts) if parts else (" ".join((findings or [])[:2]) or "")
