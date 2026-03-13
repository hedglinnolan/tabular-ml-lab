"""
07 — Sensitivity Analysis

Assess robustness of modeling results by testing how sensitive they are to:
- Random seed choice
- Individual feature removal (dropout)

This page helps answer: "Would a reviewer trust that these results aren't fragile?"

AUDIT NOTE (Data Flow):
- Operates on: trained_models, X_train/X_test/y_train/y_test from session state
- Requires: Completed Train & Compare (page 6)
- Methodology logging: Added for seed sensitivity and feature dropout analyses
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

from utils.theme import inject_custom_css, render_sidebar_workflow
from utils.table_export import table
from utils.session_state import init_session_state, log_methodology
from utils.storyline import render_breadcrumb, render_page_navigation

init_session_state()

st.set_page_config(page_title="Sensitivity Analysis | Tabular ML Lab", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="08_Sensitivity_Analysis")

st.title("🔬 Sensitivity Analysis")
st.caption("Use this after the quick workflow when you need to show that your result is robust, not just strong once.")
render_breadcrumb("08_Sensitivity_Analysis")
render_page_navigation("08_Sensitivity_Analysis")

if st.session_state.get("workflow_mode", "quick") == "quick":
    st.info("""
    🧭 **Advanced workflow step** — Return here after the quick workflow when you need to demonstrate that your result is robust across seeds or feature perturbations.
    """)

st.markdown("""
### Why Sensitivity Analysis?

Use it when a reviewer is likely to ask whether your result is robust or whether it depends on one lucky configuration.

**This page tests:**
1. **Random seed sensitivity** — Does performance vary across different train/test splits?
2. **Feature dropout** — Which features are critical vs redundant?

**For publication:** Report confidence intervals from multiple seeds, not just a single run.
""")

# ── Check prerequisites ──────────────────────────────────────────────
data_config = st.session_state.get("data_config")
trained_models = st.session_state.get("trained_models", {})
model_results = st.session_state.get("model_results", {})

if not trained_models:
    st.warning("⚠️ No trained models found. Please run **Train & Compare** first.")
    st.stop()

X_train = st.session_state.get("X_train")
X_test = st.session_state.get("X_test")
y_train = st.session_state.get("y_train")
y_test = st.session_state.get("y_test")

if X_train is None or X_test is None:
    st.warning("⚠️ Train/test split not found. Please run **Preprocess** and **Train & Compare** first.")
    st.stop()

task_type = getattr(data_config, "task_type", "regression") or "regression"
feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(X_train.shape[1])]

# ── Model selector ───────────────────────────────────────────────────
model_keys = list(trained_models.keys())
selected_model = st.selectbox(
    "Select model to analyze",
    model_keys,
    format_func=lambda k: k.upper(),
    help="Choose the model whose robustness you want to test.",
)

primary_metric = "rmse" if task_type == "regression" else "accuracy"
metric_options = ["rmse", "r2", "mae"] if task_type == "regression" else ["accuracy", "f1", "roc_auc"]
primary_metric = st.selectbox("Primary metric", metric_options, index=0)

st.markdown("---")

# ── 1. Random Seed Sensitivity ───────────────────────────────────────
st.header("🎲 Random Seed Sensitivity")
st.markdown(
    "If your results change dramatically with a different random seed, "
    "they may be driven by a lucky/unlucky train-test split rather than real signal. "
    "**Robust results show low variance across seeds.**"
)

n_seeds = st.slider("Number of seeds to test", 3, 20, 8, help="More seeds = more confident assessment, but takes longer.")
seed_list = [0, 1, 7, 13, 42, 99, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6174, 7777, 8888, 9001, 9999, 31337][:n_seeds]
baseline_seed = st.session_state.get("random_seed", 42)

if st.button("▶️ Run Seed Sensitivity", type="primary", key="run_seed"):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
    from sklearn.base import clone

    model_wrapper = trained_models[selected_model]
    # Get the underlying sklearn estimator, not the wrapper
    model_obj = model_wrapper.get_model() if hasattr(model_wrapper, 'get_model') else model_wrapper
    pipelines = st.session_state.get("fitted_preprocessing_pipelines", {})
    pipeline = pipelines.get(selected_model)

    # Skip NN models — PyTorch models don't support sklearn clone
    if selected_model == 'nn':
        st.warning("⚠️ Seed sensitivity is not supported for Neural Network models (PyTorch doesn't support sklearn clone). Select a different model.")
        st.stop()

    progress = st.progress(0, text=f"Initializing seed sensitivity for {selected_model.upper()}... (retraining model {n_seeds} times)")
    status_text = st.empty()
    results = []

    for i, seed in enumerate(seed_list):
        status_text.text(f"Training {selected_model.upper()} with seed {seed} ({i+1}/{len(seed_list)})...")
        try:
            # Clone and retrain with different seed
            cloned = clone(model_obj)
            if hasattr(cloned, "random_state"):
                cloned.set_params(random_state=seed)

            # Get preprocessed data
            if pipeline is not None:
                X_tr = pipeline.transform(X_train)
                X_te = pipeline.transform(X_test)
            else:
                X_tr = X_train.values if hasattr(X_train, "values") else X_train
                X_te = X_test.values if hasattr(X_test, "values") else X_test

            if hasattr(X_tr, "toarray"):
                X_tr = X_tr.toarray()
                X_te = X_te.toarray()

            cloned.fit(X_tr, y_train)
            preds = cloned.predict(X_te)

            metrics = {}
            if task_type == "regression":
                metrics["rmse"] = np.sqrt(mean_squared_error(y_test, preds))
                metrics["mae"] = mean_absolute_error(y_test, preds)
                metrics["r2"] = r2_score(y_test, preds)
            else:
                metrics["accuracy"] = accuracy_score(y_test, preds)
                try:
                    metrics["f1"] = f1_score(y_test, preds, average="weighted")
                except:
                    metrics["f1"] = float("nan")
                try:
                    if hasattr(cloned, "predict_proba"):
                        proba = cloned.predict_proba(X_te)
                        if proba.shape[1] == 2:
                            metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                        else:
                            metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                except:
                    metrics["roc_auc"] = float("nan")

            results.append({"seed": seed, **metrics})
        except Exception as e:
            results.append({"seed": seed, primary_metric: float("nan"), "_error": str(e)})

        progress.progress((i + 1) / len(seed_list), text=f"Seed {seed} ({i+1}/{len(seed_list)})")

    progress.empty()
    status_text.empty()

    if results:
        df_seeds = pd.DataFrame(results)
        st.session_state["sensitivity_seed_results"] = df_seeds
        log_methodology(step='Sensitivity Analysis', action='Ran seed stability analysis', details={
            'model': selected_model,
            'n_seeds': n_seeds,
            'metric': primary_metric
        })

        # Display
        valid = df_seeds[primary_metric].dropna()
        if len(valid) > 1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{valid.mean():.4f}")
            col2.metric("Std Dev", f"{valid.std():.4f}")
            col3.metric("Range", f"{valid.max() - valid.min():.4f}")
            cv = valid.std() / abs(valid.mean()) * 100 if valid.mean() != 0 else 0
            col4.metric("CV (%)", f"{cv:.1f}%")

            if cv < 2:
                st.success("✅ **Highly robust.** Less than 2% coefficient of variation across seeds.")
            elif cv < 5:
                st.info("ℹ️ **Moderately robust.** 2-5% variation — acceptable for most applications.")
            elif cv < 10:
                st.warning("⚠️ **Some instability.** 5-10% variation — consider ensemble methods or larger training set.")
            else:
                st.error("🔴 **Unstable.** >10% variation — results may not be reproducible. Investigate data or model choice.")

            st.bar_chart(df_seeds.set_index("seed")[[primary_metric]])
            with st.expander("Full results table"):
                table(df_seeds, key="seed_sensitivity", use_container_width=True)

# Show cached results if they exist
elif "sensitivity_seed_results" in st.session_state:
    df_seeds = st.session_state["sensitivity_seed_results"]
    valid = df_seeds[primary_metric].dropna()
    if len(valid) > 1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{valid.mean():.4f}")
        col2.metric("Std Dev", f"{valid.std():.4f}")
        col3.metric("Range", f"{valid.max() - valid.min():.4f}")
        cv = valid.std() / abs(valid.mean()) * 100 if valid.mean() != 0 else 0
        col4.metric("CV (%)", f"{cv:.1f}%")
        st.bar_chart(df_seeds.set_index("seed")[[primary_metric]])

# ── Interpretation Guide ─────────────────────────────────────────────
if "sensitivity_seed_results" in st.session_state:
    df_seeds = st.session_state["sensitivity_seed_results"]
    seed_results = df_seeds.to_dict('records')
    
    if len(seed_results) > 1:
        st.markdown("---")
        st.markdown("### 📊 Interpreting Seed Sensitivity")
        
        # Get metric range
        metric_col = 'roc_auc' if task_type == 'classification' and primary_metric == 'roc_auc' else primary_metric
        metric_values = [r[metric_col] for r in seed_results if metric_col in r and not np.isnan(r[metric_col])]
        
        if metric_values:
            metric_range = max(metric_values) - min(metric_values)
            metric_mean = np.mean(metric_values)
            
            st.markdown(f"""
            **Your Results:**
            - {metric_col.upper()} range: {min(metric_values):.3f} to {max(metric_values):.3f}
            - Range width: {metric_range:.3f}
            - Mean: {metric_mean:.3f}
            """)
            
            # Interpretation thresholds
            if metric_range < 0.03:
                stability = "✅ Very stable"
                interpretation = "Excellent. Your model is highly robust to different train/test splits."
                recommendation = "Report the mean with standard error. No concerns for publication."
            elif metric_range < 0.05:
                stability = "🟡 Moderate stability"
                interpretation = "Acceptable. Performance varies slightly across seeds, but within normal range."
                recommendation = "Report confidence intervals (not just point estimates). Mention in limitations if needed."
            else:
                stability = "⚠️ High sensitivity"
                interpretation = "Concerning. Large performance variation suggests model instability or small dataset."
                recommendation = """
**Action needed:**
1. Report full distribution (not just best result)
2. Consider ensemble methods (average multiple seeds)
3. Mention as limitation in discussion
4. Check if dataset is too small (n < 200 often unstable)
"""
            
            st.info(f"""
**Stability Assessment:** {stability}

**Interpretation:** {interpretation}

**Recommendation:** {recommendation}
""")
            
            # Reference standards
            with st.expander("📚 What Do These Thresholds Mean?"):
                st.markdown("""
**Range < 0.03:** Publication-ready without caveats. Model predictions are consistent.

**Range 0.03-0.05:** Common in clinical ML. Mention seed variation in methods, report CIs.

**Range > 0.05:** Red flag for reviewers. Suggests:
- Dataset too small (underpowered)
- Features unstable (high noise)
- Model overfitting

**Best practice:** Always report results across multiple seeds (5-10 runs), never cherry-pick best seed.
""")

st.markdown("---")

# ── 2. Feature Dropout ───────────────────────────────────────────────
st.header("🔀 Feature Dropout")
st.markdown(
    "Remove one feature at a time and retrain. Features whose removal causes a large "
    "performance drop are genuinely important. Features whose removal *improves* performance "
    "may be adding noise. **This complements SHAP/permutation importance with a causal flavor.**"
)

max_features = st.slider(
    "Max features to test",
    1, min(len(feature_names), 30), min(len(feature_names), 15),
    help="Testing all features can be slow. Start with the top N by importance.",
)

if st.button("▶️ Run Feature Dropout", type="primary", key="run_dropout"):
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
    from sklearn.base import clone

    model_obj = trained_models[selected_model]
    pipelines = st.session_state.get("fitted_preprocessing_pipelines", {})
    pipeline = pipelines.get(selected_model)

    # Get baseline performance
    if pipeline is not None:
        X_te_base = pipeline.transform(X_test)
    else:
        X_te_base = X_test.values if hasattr(X_test, "values") else X_test
    if hasattr(X_te_base, "toarray"):
        X_te_base = X_te_base.toarray()

    preds_base = model_obj.predict(X_te_base)
    if task_type == "regression":
        baseline_score = np.sqrt(mean_squared_error(y_test, preds_base))
    else:
        baseline_score = accuracy_score(y_test, preds_base)

    # Test dropping each feature
    features_to_test = feature_names[:max_features]
    progress = st.progress(0, text="Running feature dropout...")
    dropout_results = []

    for i, feat in enumerate(features_to_test):
        try:
            remaining = [f for f in feature_names if f != feat]
            X_tr_drop = X_train[remaining] if hasattr(X_train, "columns") else np.delete(X_train, feature_names.index(feat), axis=1)
            X_te_drop = X_test[remaining] if hasattr(X_test, "columns") else np.delete(X_test, feature_names.index(feat), axis=1)

            cloned = clone(model_obj)
            # Retrain without pipeline (raw features) for simplicity
            X_tr_vals = X_tr_drop.values if hasattr(X_tr_drop, "values") else X_tr_drop
            X_te_vals = X_te_drop.values if hasattr(X_te_drop, "values") else X_te_drop

            # Handle NaN with simple median fill for dropout test
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(strategy="median")
            X_tr_vals = imp.fit_transform(X_tr_vals)
            X_te_vals = imp.transform(X_te_vals)

            cloned.fit(X_tr_vals, y_train)
            preds_drop = cloned.predict(X_te_vals)

            if task_type == "regression":
                drop_score = np.sqrt(mean_squared_error(y_test, preds_drop))
                impact = drop_score - baseline_score  # positive = worse without feature
            else:
                drop_score = accuracy_score(y_test, preds_drop)
                impact = baseline_score - drop_score  # positive = worse without feature

            dropout_results.append({
                "feature": feat,
                "score_without": drop_score,
                "impact": impact,
            })
        except Exception as e:
            dropout_results.append({"feature": feat, "score_without": float("nan"), "impact": 0, "_error": str(e)})

        progress.progress((i + 1) / len(features_to_test), text=f"Testing without '{feat}' ({i+1}/{len(features_to_test)})")

    progress.empty()

    if dropout_results:
        df_dropout = pd.DataFrame(dropout_results).sort_values("impact", ascending=False)
        st.session_state["sensitivity_dropout_results"] = df_dropout
        st.session_state["sensitivity_dropout_baseline"] = baseline_score
        log_methodology(step='Sensitivity Analysis', action='Ran feature dropout analysis', details={
            'model': selected_model,
            'n_features_tested': len(features_to_test),
            'metric': primary_metric
        })

        st.metric(f"Baseline {primary_metric}", f"{baseline_score:.4f}")

        # Color code: features whose removal hurts (important) vs helps (noisy)
        important = df_dropout[df_dropout["impact"] > 0.001].head(10)
        noisy = df_dropout[df_dropout["impact"] < -0.001]

        if not important.empty:
            st.markdown("**Most impactful features** (removing them hurts performance):")
            chart_data = important.set_index("feature")[["impact"]]
            st.bar_chart(chart_data)

        if not noisy.empty:
            st.markdown("**Potentially noisy features** (removing them *improves* performance):")
            for _, row in noisy.iterrows():
                st.markdown(f"- `{row['feature']}`: removing it improved {primary_metric} by {abs(row['impact']):.4f}")

        with st.expander("Full dropout results"):
            table(df_dropout[["feature", "score_without", "impact"]], key="feature_dropout", use_container_width=True)

elif "sensitivity_dropout_results" in st.session_state:
    df_dropout = st.session_state["sensitivity_dropout_results"]
    baseline_score = st.session_state.get("sensitivity_dropout_baseline", 0)
    st.metric(f"Baseline {primary_metric}", f"{baseline_score:.4f}")
    important = df_dropout[df_dropout["impact"] > 0.001].head(10)
    if not important.empty:
        st.bar_chart(important.set_index("feature")[["impact"]])

st.markdown("---")
st.caption(
    "💡 **For your methods section:** Report seed sensitivity as evidence of reproducibility, "
    "and feature dropout as a complement to permutation importance. "
    "If results are robust, say so explicitly — reviewers notice."
)
