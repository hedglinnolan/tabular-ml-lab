"""
Portable interactive theory demos — can be rendered on any page.

Each function renders a self-contained interactive Plotly demo inside a
Streamlit expander. They use unique widget keys prefixed with a page_context
parameter to avoid key collisions when the same demo appears on multiple pages.

Usage:
    from utils.theory_demos import demo_skewness, demo_calibration
    demo_skewness(page_context="eda")       # on EDA page
    demo_calibration(page_context="train")  # on Train & Compare page
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def demo_skewness(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive skewness distribution demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: See how skewness changes a distribution", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        skew_alpha = st.slider(
            "Skewness intensity",
            min_value=1.0, max_value=20.0, value=2.0, step=0.5,
            key=f"{page_context}_demo_skew_alpha",
            help="Slide right to increase right-skew.",
        )
        rng = np.random.default_rng(42)
        skew_data = rng.gamma(shape=max(0.1, 5.0 / skew_alpha), scale=1.0, size=800)

        from scipy.stats import skew as calc_skew
        computed_skew = calc_skew(skew_data)
        mean_val = np.mean(skew_data)
        median_val = np.median(skew_data)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=skew_data, nbinsx=40,
            marker_color="rgba(99, 102, 241, 0.7)",
            marker_line=dict(color="rgba(99, 102, 241, 1)", width=1),
        ))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#dc2626",
                      annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dot", line_color="#16a34a",
                      annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
        label = "approximately symmetric" if abs(computed_skew) < 0.5 else "moderately skewed" if abs(computed_skew) < 1 else "heavily skewed"
        fig.update_layout(
            title=f"γ₁ = {computed_skew:.2f} — {label}",
            xaxis_title="Value", yaxis_title="Count",
            height=300, margin=dict(t=50, b=40, l=50, r=30),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_skew_chart")
        st.markdown(
            "**Train your eye:** Watch the **gap between mean (red) and median (green)** — "
            "as skew grows, the mean chases the tail. Linear and distance-based models are "
            "affected; tree-based models are not."
        )


def demo_collinearity(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive collinearity coefficient instability demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: Watch coefficients destabilize as correlation increases", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        corr_val = st.slider(
            "Correlation between x₁ and x₂",
            min_value=0.0, max_value=0.99, value=0.0, step=0.05,
            key=f"{page_context}_demo_collin_corr",
        )
        rng_c = np.random.default_rng(12)
        n_c = 80
        X_c = rng_c.multivariate_normal([0, 0], [[1, corr_val], [corr_val, 1]], n_c)
        y_c = 2.0 * X_c[:, 0] + 2.0 * X_c[:, 1] + rng_c.normal(0, 1, n_c)

        coefs_x1, coefs_x2 = [], []
        for seed in range(30):
            rng_boot = np.random.default_rng(seed + 100)
            idx = rng_boot.choice(n_c, size=n_c, replace=True)
            Xb_aug = np.column_stack([np.ones(n_c), X_c[idx]])
            try:
                beta = np.linalg.lstsq(Xb_aug, y_c[idx], rcond=None)[0]
                coefs_x1.append(beta[1])
                coefs_x2.append(beta[2])
            except Exception:
                pass

        fig = go.Figure()
        fig.add_trace(go.Box(y=coefs_x1, name="β₁", marker_color="rgba(99, 102, 241, 0.7)",
                             boxpoints="all", jitter=0.3, pointpos=-1.5))
        fig.add_trace(go.Box(y=coefs_x2, name="β₂", marker_color="rgba(234, 88, 12, 0.7)",
                             boxpoints="all", jitter=0.3, pointpos=-1.5))
        fig.add_hline(y=2.0, line_dash="dash", line_color="#16a34a",
                      annotation_text="True value (2.0)")
        fig.update_layout(
            title=f"Coefficient estimates (r = {corr_val:.2f}, VIF = {1/(1-corr_val**2):.1f})",
            yaxis_title="Estimated coefficient", height=320,
            margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_collin_chart")
        st.markdown(
            "**Train your eye:** At r = 0, estimates cluster around truth. "
            "Slide to r > 0.9: they scatter wildly. The model can't tell the variables apart. "
            "VIF > 10 is the conventional red line."
        )


def demo_class_imbalance(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive class imbalance metric trap demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: Why accuracy lies under class imbalance", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        imb_pct = st.slider(
            "Positive class prevalence (%)",
            min_value=1, max_value=50, value=5, step=1,
            key=f"{page_context}_demo_imbalance_pct",
        )
        n_total = 200
        n_pos = max(1, int(n_total * imb_pct / 100))
        n_neg = n_total - n_pos

        acc_majority = n_neg / n_total * 100
        tp = int(n_pos * 0.6)
        fp = int(n_neg * 0.08)
        fn = n_pos - tp
        tn = n_neg - fp
        acc_model = (tp + tn) / n_total * 100
        prec_model = tp / max(tp + fp, 1) * 100
        recall_model = tp / max(tp + fn, 1) * 100
        f1_model = 2 * (prec_model * recall_model) / max(prec_model + recall_model, 0.01)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Always predict negative",
            x=["Accuracy", "Recall", "Precision", "F1"],
            y=[acc_majority, 0, 0, 0], marker_color="rgba(220, 38, 38, 0.7)"))
        fig.add_trace(go.Bar(name="Mediocre model",
            x=["Accuracy", "Recall", "Precision", "F1"],
            y=[acc_model, recall_model, prec_model, f1_model], marker_color="rgba(22, 163, 74, 0.7)"))
        fig.update_layout(
            barmode="group", height=280, yaxis_title="%", yaxis_range=[0, 105],
            title=f"Prevalence = {imb_pct}% — accuracy is {'meaningful' if imb_pct > 30 else 'misleading'}",
            margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_imbal_chart")
        st.markdown(
            "**Train your eye:** At low prevalence, 'always predict negative' scores near-perfect accuracy "
            "with zero recall. Slide toward 50% to see accuracy become meaningful again."
        )


def demo_calibration(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive calibration reliability diagram demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: See what miscalibration looks like", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        miscal = st.slider(
            "Miscalibration strength",
            min_value=-0.3, max_value=0.3, value=0.15, step=0.05,
            key=f"{page_context}_demo_cal_miscal",
            help="Positive = overconfident. Negative = underconfident. Zero = perfect.",
        )
        bin_centers = np.linspace(0.05, 0.95, 10)
        actual_rates = np.clip(bin_centers - miscal * np.sin(np.pi * bin_centers), 0.01, 0.99)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8"), name="Perfect calibration"))
        fig.add_trace(go.Scatter(x=bin_centers, y=actual_rates, mode="lines+markers",
            line=dict(color="#dc2626", width=2.5), marker=dict(size=8), name="Model"))
        fig.add_trace(go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]),
            y=np.concatenate([actual_rates, bin_centers[::-1]]),
            fill="toself", fillcolor="rgba(220, 38, 38, 0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        ece = np.mean(np.abs(bin_centers - actual_rates))
        fig.update_layout(
            title=f"Reliability Diagram — ECE = {ece:.3f}",
            xaxis_title="Predicted probability", yaxis_title="Observed rate",
            xaxis_range=[0, 1], yaxis_range=[0, 1], height=320,
            margin=dict(t=50, b=40, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_calib_chart")
        st.markdown(
            "**Train your eye:** Below the diagonal = overconfident. Above = underconfident. "
            "Slide to 0 for perfect calibration. ECE > 0.05 warrants recalibration."
        )


def demo_threshold(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive classification threshold tradeoff demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: How the threshold trades precision for recall", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        threshold = st.slider(
            "Classification threshold",
            min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            key=f"{page_context}_demo_threshold",
            help="Predictions above this threshold are classified as positive.",
        )
        rng = np.random.default_rng(88)
        scores_pos = rng.beta(5, 2, 30)
        scores_neg = rng.beta(2, 5, 170)
        scores = np.concatenate([scores_pos, scores_neg])
        labels = np.concatenate([np.ones(30), np.zeros(170)])

        preds = (scores >= threshold).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        prec = tp / max(tp + fp, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        f1 = 2 * (prec * rec) / max(prec + rec, 0.01)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Precision", "Recall", "F1"],
            y=[prec, rec, f1],
            marker_color=["#2563eb", "#dc2626", "#7c3aed"],
            text=[f"{v:.0f}%" for v in [prec, rec, f1]], textposition="outside",
        ))
        fig.update_layout(
            title=f"Threshold = {threshold:.2f} — TP={tp}, FP={fp}, FN={fn}",
            yaxis_title="%", yaxis_range=[0, 110], height=280,
            margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_thresh_chart")
        st.markdown(
            "**Train your eye:** Low threshold → high recall, low precision (catch everyone, many false alarms). "
            "High threshold → high precision, low recall (only flag sure cases, miss many). "
            "The right threshold is a domain decision."
        )


def demo_shap(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive SHAP decomposition demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: How SHAP decomposes a prediction", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        shap_glucose = st.slider("Glucose", 70, 200, 140, key=f"{page_context}_demo_shap_glucose")
        shap_bmi = st.slider("BMI", 18.0, 45.0, 28.0, step=0.5, key=f"{page_context}_demo_shap_bmi")
        shap_age = st.slider("Age", 20, 80, 50, key=f"{page_context}_demo_shap_age")

        baseline = 0.30
        contribs = [
            ("Glucose", (shap_glucose - 120) * 0.002),
            ("BMI", (shap_bmi - 25) * 0.008),
            ("Age", (shap_age - 45) * 0.003),
        ]
        prediction = np.clip(baseline + sum(c for _, c in contribs), 0.01, 0.99)
        colors = ["#dc2626" if c > 0 else "#2563eb" for _, c in contribs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[n for n, _ in contribs], x=[c for _, c in contribs], orientation="h",
            marker_color=colors,
            text=[f"{c:+.3f}" for _, c in contribs], textposition="outside",
        ))
        fig.add_vline(x=0, line_color="#94a3b8")
        fig.update_layout(
            title=f"Baseline: {baseline:.2f} → Prediction: {prediction:.3f}",
            xaxis_title="SHAP contribution", height=240,
            margin=dict(t=50, b=30, l=80, r=60), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_shap_chart")
        st.markdown(
            "**Train your eye:** Red bars push risk up; blue push it down. "
            "Contributions always sum to the gap between baseline and prediction. "
            "SHAP explains the *model's* reasoning, not causation."
        )


def demo_seed_sensitivity(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive seed sensitivity demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: How seed choice affects reported performance", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        snr = st.slider(
            "Signal-to-noise ratio",
            min_value=0.5, max_value=5.0, value=1.5, step=0.5,
            key=f"{page_context}_demo_seed_snr",
            help="Lower = noisier data, more seed-sensitive results.",
        )
        n_seeds = 10
        scores = []
        for s in range(n_seeds):
            rng = np.random.default_rng(s * 7 + 3)
            X = rng.standard_normal((60, 2))
            y = snr * X[:, 0] + rng.normal(0, 1, 60)
            idx = rng.permutation(60)
            split = 42
            Xtr = np.column_stack([np.ones(split), X[idx[:split]]])
            Xte = np.column_stack([np.ones(60 - split), X[idx[split:]]])
            beta = np.linalg.lstsq(Xtr, y[idx[:split]], rcond=None)[0]
            pred = Xte @ beta
            yte = y[idx[split:]]
            ss_res = np.sum((yte - pred) ** 2)
            ss_tot = np.sum((yte - np.mean(yte)) ** 2)
            scores.append(1 - ss_res / max(ss_tot, 1e-10))

        cv_pct = np.std(scores) / max(abs(np.mean(scores)), 1e-10) * 100
        stability = "Robust" if cv_pct < 2 else "Acceptable" if cv_pct < 5 else "Concerning" if cv_pct < 10 else "Unstable"
        colors = ["rgba(220,38,38,0.7)" if s == min(scores) or s == max(scores) else "rgba(99,102,241,0.7)" for s in scores]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Seed {i}" for i in range(n_seeds)], y=scores, marker_color=colors))
        fig.add_hline(y=np.mean(scores), line_dash="dash", line_color="#16a34a",
                      annotation_text=f"Mean = {np.mean(scores):.3f}")
        fig.update_layout(
            title=f"R² across seeds — CV = {cv_pct:.1f}% ({stability})",
            yaxis_title="R²", height=280,
            margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_seed_chart")
        st.markdown(
            "**Train your eye:** Red bars = best and worst seeds. If they tell different stories, "
            "report mean ± SD, not a single run."
        )


def demo_outliers(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive demo: how one outlier pulls a regression line."""
    if wrapped:
        _expander = st.expander("📖 Interactive: See how one outlier pulls a regression line", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        col1, col2 = st.columns(2)
        with col1:
            ox = st.slider("Outlier X position", 0.0, 10.0, 8.0, 0.5, key=f"{page_context}_demo_outlier_x")
        with col2:
            oy = st.slider("Outlier Y position", -5.0, 20.0, 15.0, 0.5, key=f"{page_context}_demo_outlier_y")

        rng = np.random.default_rng(42)
        x_clean = rng.uniform(0, 6, 25)
        y_clean = 1.0 * x_clean + rng.normal(0, 0.8, 25)
        x_all = np.append(x_clean, ox)
        y_all = np.append(y_clean, oy)

        # Fit with and without outlier
        c_clean = np.polyfit(x_clean, y_clean, 1)
        c_all = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(0, 10, 50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_clean, y=y_clean, mode="markers",
            marker=dict(color="rgba(99,102,241,0.7)", size=7), name="Clean data"))
        fig.add_trace(go.Scatter(x=[ox], y=[oy], mode="markers",
            marker=dict(color="#dc2626", size=14, symbol="x"), name="Outlier"))
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(c_clean, xs), mode="lines",
            line=dict(color="#16a34a", dash="dash"), name=f"Without outlier (slope={c_clean[0]:.2f})"))
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(c_all, xs), mode="lines",
            line=dict(color="#dc2626"), name=f"With outlier (slope={c_all[0]:.2f})"))
        fig.update_layout(height=300, margin=dict(t=30, b=30, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_outlier_chart")
        st.markdown(
            "**Train your eye:** Drag the outlier far from the cluster. The red line tilts toward it. "
            "One point can dominate a linear model's coefficients. Robust methods (Huber, RANSAC) resist this."
        )


def demo_bias_variance(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive bias-variance tradeoff via KNN k."""
    if wrapped:
        _expander = st.expander("📖 Interactive: The bias-variance tradeoff", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        k_val = st.slider("k (number of neighbors)", 1, 30, 1, key=f"{page_context}_demo_bv_k")

        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(0, 10, 60))
        y_true = np.sin(x)
        y = y_true + rng.normal(0, 0.3, 60)

        # KNN prediction
        x_grid = np.linspace(0, 10, 200)
        y_pred = np.array([np.mean(y[np.argsort(np.abs(x - xg))[:k_val]]) for xg in x_grid])

        # Compute train MSE
        y_train_pred = np.array([np.mean(y[np.argsort(np.abs(x - xi))[:k_val]]) for xi in x])
        mse = np.mean((y - y_train_pred) ** 2)
        label = "Overfitting" if k_val <= 3 else "Underfitting" if k_val >= 20 else "Balanced"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
            marker=dict(color="rgba(99,102,241,0.5)", size=5), name="Data"))
        fig.add_trace(go.Scatter(x=x_grid, y=np.sin(x_grid), mode="lines",
            line=dict(color="#94a3b8", dash="dot"), name="True function"))
        fig.add_trace(go.Scatter(x=x_grid, y=y_pred, mode="lines",
            line=dict(color="#dc2626", width=2.5), name=f"KNN (k={k_val})"))
        fig.update_layout(
            title=f"k = {k_val} — MSE = {mse:.3f} ({label})",
            height=300, margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_bv_chart")
        st.markdown(
            "**Train your eye:** k=1 traces every point (high variance, low bias). "
            "k=30 is nearly flat (low variance, high bias). The sweet spot minimizes total error."
        )


def demo_bootstrap(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive bootstrap confidence interval demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: Watch the bootstrap distribution build up", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        n_boot = st.slider("Number of bootstrap samples", 10, 500, 100, 10, key=f"{page_context}_demo_boot_B")

        rng = np.random.default_rng(55)
        data = rng.exponential(scale=2.0, size=30)
        true_mean = np.mean(data)

        boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=boot_means, nbinsx=30,
            marker_color="rgba(99,102,241,0.7)", marker_line=dict(color="rgba(99,102,241,1)", width=1)))
        fig.add_vline(x=true_mean, line_dash="dash", line_color="#dc2626",
            annotation_text=f"Sample mean: {true_mean:.2f}")
        fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor="rgba(22,163,74,0.15)", line_width=0,
            annotation_text=f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]", annotation_position="top left")
        fig.update_layout(
            title=f"Bootstrap distribution (B={n_boot})",
            xaxis_title="Bootstrap mean", yaxis_title="Count", height=280,
            margin=dict(t=50, b=40, l=50, r=20), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_boot_chart")
        st.markdown(
            "**Train your eye:** More samples → smoother distribution → narrower CI. "
            "B=100 is usually enough for means; B=500+ for percentiles or complex statistics."
        )


# Registry for programmatic access
def demo_regularization(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive Ridge vs LASSO demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: Ridge shrinks, LASSO selects", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        c1, c2 = st.columns(2)
        with c1:
            corr = st.slider("Feature correlation", 0.0, 0.99, 0.8, 0.05, key=f"{page_context}_demo_reg_corr")
        with c2:
            alpha = st.slider("Regularization α", 0.01, 5.0, 0.5, 0.1, key=f"{page_context}_demo_reg_alpha")

        rng = np.random.default_rng(33)
        n = 60
        X = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], n)
        y = 3.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0, 1, n)
        Xa = np.column_stack([np.ones(n), X])
        # OLS
        b_ols = np.linalg.lstsq(Xa, y, rcond=None)[0][1:]
        # Ridge
        I = np.eye(3); I[0, 0] = 0
        b_ridge = np.linalg.solve(Xa.T @ Xa + alpha * n * I, Xa.T @ y)[1:]
        # LASSO (coordinate descent, simple)
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=alpha, max_iter=2000, fit_intercept=True)
        lasso.fit(X, y)
        b_lasso = lasso.coef_

        fig = go.Figure()
        for i, (name, coefs, color) in enumerate([
            ("OLS", b_ols, "#94a3b8"), ("Ridge", b_ridge, "#2563eb"), ("LASSO", b_lasso, "#dc2626")
        ]):
            fig.add_trace(go.Bar(name=name, x=["β₁", "β₂"], y=coefs, marker_color=color,
                text=[f"{c:.2f}" for c in coefs], textposition="outside"))
        fig.add_hline(y=3.0, line_dash="dash", line_color="#16a34a", annotation_text="True (3.0)")
        fig.update_layout(barmode="group", height=280, yaxis_title="Coefficient",
            margin=dict(t=30, b=30, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_reg_chart")
        st.markdown(
            "**Train your eye:** Ridge shrinks both toward zero but keeps both. "
            "LASSO drives one to exactly zero — it selects. At high correlation, OLS estimates are unstable; regularization stabilizes them."
        )


def demo_cross_validation(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive k-fold CV variation demo."""
    if wrapped:
        _expander = st.expander("📖 Interactive: See fold-to-fold variation in cross-validation", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider("Number of folds", 2, 15, 5, key=f"{page_context}_demo_cv_k")
        with c2:
            noise = st.slider("Noise level", 0.5, 5.0, 1.5, 0.5, key=f"{page_context}_demo_cv_noise")

        rng = np.random.default_rng(22)
        n = 80
        X = rng.standard_normal((n, 2))
        y = 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, noise, n)

        fold_scores = []
        indices = np.arange(n)
        rng.shuffle(indices)
        fold_size = n // k
        for f in range(k):
            val_idx = indices[f * fold_size:(f + 1) * fold_size]
            tr_idx = np.setdiff1d(indices, val_idx)
            Xtr = np.column_stack([np.ones(len(tr_idx)), X[tr_idx]])
            Xval = np.column_stack([np.ones(len(val_idx)), X[val_idx]])
            beta = np.linalg.lstsq(Xtr, y[tr_idx], rcond=None)[0]
            pred = Xval @ beta
            ss_res = np.sum((y[val_idx] - pred) ** 2)
            ss_tot = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
            fold_scores.append(max(-1, 1 - ss_res / max(ss_tot, 1e-10)))

        spread = max(fold_scores) - min(fold_scores)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(k)], y=fold_scores,
            marker_color=["rgba(220,38,38,0.7)" if s == min(fold_scores) or s == max(fold_scores) else "rgba(99,102,241,0.7)" for s in fold_scores]))
        fig.add_hline(y=np.mean(fold_scores), line_dash="dash", line_color="#16a34a",
            annotation_text=f"Mean = {np.mean(fold_scores):.3f}")
        fig.update_layout(
            title=f"{k}-fold CV — spread = {spread:.3f}",
            yaxis_title="R²", height=280, margin=dict(t=50, b=30, l=50, r=20), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_cv_chart")
        st.markdown(
            "**Train your eye:** More folds → less variation per fold but more computation. "
            "High noise → wide fold spread. Report mean ± SD, not just the mean."
        )


def demo_transforms(page_context: str = "ref", expanded: bool = False, wrapped: bool = True) -> None:
    """Interactive power transform demo (log, sqrt, Box-Cox)."""
    if wrapped:
        _expander = st.expander("📖 Interactive: See how transforms normalize a distribution", expanded=expanded)
    else:
        from contextlib import nullcontext
        _expander = nullcontext()
    with _expander:
        lam = st.slider("Box-Cox λ (0 = log, 0.5 = sqrt, 1 = identity)",
            0.0, 2.0, 0.0, 0.1, key=f"{page_context}_demo_transform_lambda")

        rng = np.random.default_rng(42)
        raw = rng.exponential(scale=2.0, size=500)
        if lam == 0:
            transformed = np.log(raw)
            label = "log(x)"
        else:
            transformed = (raw ** lam - 1) / lam
            label = f"(x^{lam:.1f} - 1) / {lam:.1f}"

        from scipy.stats import skew as calc_skew
        skew_before = calc_skew(raw)
        skew_after = calc_skew(transformed)

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Original (γ₁={skew_before:.2f})", f"Transformed: {label} (γ₁={skew_after:.2f})"])
        fig.add_trace(go.Histogram(x=raw, nbinsx=30, marker_color="rgba(220,38,38,0.6)"), row=1, col=1)
        fig.add_trace(go.Histogram(x=transformed, nbinsx=30, marker_color="rgba(22,163,74,0.6)"), row=1, col=2)
        fig.update_layout(height=260, margin=dict(t=40, b=30, l=40, r=20), template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"{page_context}_transform_chart")
        st.markdown(
            "**Train your eye:** λ=0 (log) aggressively compresses the right tail. "
            "λ=1 is identity (no change). Slide to find the λ that makes skewness ≈ 0."
        )


DEMO_REGISTRY = {
    "skewness": demo_skewness,
    "collinearity": demo_collinearity,
    "class_imbalance": demo_class_imbalance,
    "calibration": demo_calibration,
    "threshold_choice": demo_threshold,
    "shap": demo_shap,
    "seed_sensitivity": demo_seed_sensitivity,
    "outliers": demo_outliers,
    "bias_variance": demo_bias_variance,
    "bootstrap": demo_bootstrap,
    "regularization": demo_regularization,
    "cross_validation": demo_cross_validation,
    "transforms": demo_transforms,
}


def render_inline_demo(anchor_key: str, page_context: str = "page", expanded: bool = False, wrapped: bool = True) -> bool:
    """Render an inline interactive demo by anchor key.

    Args:
        wrapped: If True, demo renders inside its own expander (for Theory Reference).
                 If False, demo renders directly (for inline coaching cards).

    Returns True if a demo was rendered, False if no demo exists for this key.
    """
    demo_fn = DEMO_REGISTRY.get(anchor_key)
    if demo_fn:
        demo_fn(page_context=page_context, expanded=expanded, wrapped=wrapped)
        return True
    return False
