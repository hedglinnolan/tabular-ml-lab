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

from utils.coaching_ui import render_page_coaching
render_page_coaching("08_Sensitivity_Analysis")

if st.session_state.get("workflow_mode", "quick") == "quick":
    st.info("🧭 **Advanced workflow step** — Return here after the quick workflow to demonstrate result robustness.")

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

if X_train is None or X_test is None or y_train is None or y_test is None:
    st.warning("⚠️ Train/test split not found. Please run **Preprocess** and **Train & Compare** first.")
    st.stop()

task_type = getattr(data_config, "task_type", "regression") or "regression"
feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(X_train.shape[1])]

# ── Model selector ───────────────────────────────────────────────────
model_keys = list(trained_models.keys())
# Filter NN from seed sensitivity (PyTorch can't be sklearn-cloned)
_seed_compatible = [k for k in model_keys if k != 'nn']
if not _seed_compatible:
    st.warning("No models compatible with seed sensitivity analysis. Neural networks require sklearn-compatible cloning.")
    st.stop()
if 'nn' in model_keys and 'nn' not in _seed_compatible:
    st.caption("ℹ️ Neural Network excluded from sensitivity analysis (PyTorch models cannot be cloned for re-seeding).")
selected_model = st.selectbox(
    "Select model to analyze",
    _seed_compatible,
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
    "Each seed below draws a **fresh train/test split**, re-fits the "
    "preprocessing pipeline on the new training rows, and retrains the model "
    "(the model's own random seed is varied too). If results change "
    "dramatically across seeds, they were driven by a lucky/unlucky split "
    "rather than real signal. **Robust results show low variance across seeds.**"
)
st.caption(
    "Note: this diagnostic deliberately re-partitions all rows (including the "
    "locked test set) to measure split sensitivity — your reported headline "
    "metrics still come from the untouched lockbox test set on Train & Compare."
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

    progress = st.progress(0, text=f"Initializing seed sensitivity for {selected_model.upper()}... (re-splitting and retraining {n_seeds} times)")
    status_text = st.empty()
    results = []

    # Pool the stored splits back together so every seed draws a genuinely
    # fresh partition. Re-seeding a model on a FIXED split (the old behavior)
    # measures only model-internal randomness — identically zero for Ridge,
    # GLM, kNN, LDA, NB — and says nothing about split luck.
    from sklearn.model_selection import train_test_split as _seed_tts
    _as_df = lambda p: p if isinstance(p, pd.DataFrame) else pd.DataFrame(p)
    _pool_parts = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    _pool_parts = [(px_, py_) for px_, py_ in _pool_parts if px_ is not None and len(px_) > 0]
    X_pool = pd.concat([_as_df(px_) for px_, _ in _pool_parts], axis=0, ignore_index=True)
    y_pool = np.concatenate([np.asarray(py_) for _, py_ in _pool_parts])
    _test_frac = max(0.05, min(0.5, len(X_test) / max(1, len(X_pool))))
    _seed_transformer = st.session_state.get('target_transformer')

    for i, seed in enumerate(seed_list):
        status_text.text(f"Split + train {selected_model.upper()} with seed {seed} ({i+1}/{len(seed_list)})...")
        try:
            # Fresh split for this seed
            _strat = y_pool if task_type != "regression" else None
            try:
                X_tr_raw, X_te_raw, y_tr, y_te = _seed_tts(
                    X_pool, y_pool, test_size=_test_frac, random_state=seed, stratify=_strat)
            except ValueError:
                X_tr_raw, X_te_raw, y_tr, y_te = _seed_tts(
                    X_pool, y_pool, test_size=_test_frac, random_state=seed)

            # Clone model and vary its internal seed too
            cloned = clone(model_obj)
            if hasattr(cloned, "random_state"):
                cloned.set_params(random_state=seed)

            # Re-fit the preprocessing on THIS seed's training rows
            if pipeline is not None:
                _pipe_seed = clone(pipeline)
                _pipe_seed.fit(X_tr_raw)
                X_tr = _pipe_seed.transform(X_tr_raw)
                X_te = _pipe_seed.transform(X_te_raw)
            else:
                X_tr = X_tr_raw.values
                X_te = X_te_raw.values

            if hasattr(X_tr, "toarray"):
                X_tr = X_tr.toarray()
                X_te = X_te.toarray()

            cloned.fit(X_tr, y_tr)
            preds = cloned.predict(X_te)

            # Evaluate on the original target scale when a transform is active
            y_eval = y_te
            if task_type == "regression" and _seed_transformer is not None:
                if _seed_transformer == 'log1p':
                    preds = np.expm1(preds)
                    y_eval = np.expm1(np.asarray(y_te, dtype=float))
                else:
                    preds = _seed_transformer.inverse_transform(preds.reshape(-1, 1)).ravel()
                    y_eval = _seed_transformer.inverse_transform(
                        np.asarray(y_te, dtype=float).reshape(-1, 1)).ravel()

            metrics = {}
            if task_type == "regression":
                metrics["rmse"] = np.sqrt(mean_squared_error(y_eval, preds))
                metrics["mae"] = mean_absolute_error(y_eval, preds)
                metrics["r2"] = r2_score(y_eval, preds)
            else:
                metrics["accuracy"] = accuracy_score(y_te, preds)
                try:
                    metrics["f1"] = f1_score(y_te, preds, average="weighted")
                except:
                    metrics["f1"] = float("nan")
                try:
                    if hasattr(cloned, "predict_proba"):
                        proba = cloned.predict_proba(X_te)
                        if proba.shape[1] == 2:
                            metrics["roc_auc"] = roc_auc_score(y_te, proba[:, 1])
                        else:
                            metrics["roc_auc"] = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
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
        try:
            from utils.workflow_provenance import get_provenance
            _cv_pct = None
            _df_seeds = st.session_state.get("sensitivity_seed_results")
            if _df_seeds is not None and primary_metric in _df_seeds.columns:
                import numpy as np
                _vals = _df_seeds[primary_metric].dropna()
                if len(_vals) > 0 and _vals.mean() != 0:
                    _cv_pct = float(_vals.std() / abs(_vals.mean()) * 100)
            get_provenance().record_sensitivity(
                seed_stability=True,
                seed_stability_cv=_cv_pct,
            )
        except Exception:
            pass  # Provenance recording should never break the workflow

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
                table(df_seeds, key="seed_sensitivity")

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

            # LLM interpretation for seed sensitivity
            from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
            _bg_seed = gather_session_context()
            _seed_summary = (f"metric={metric_col}; mean={metric_mean:.4f}; std={np.std(metric_values):.4f}; "
                             f"range={metric_range:.4f}; cv={np.std(metric_values)/abs(metric_mean)*100:.1f}%; "
                             f"n_seeds={len(metric_values)}; stability={stability}; "
                             f"model={selected_model}")
            ctx_seed = build_llm_context(
                "seed_sensitivity", _seed_summary,
                model_name=selected_model,
                where="Seed sensitivity analysis",
                sample_size=_bg_seed.pop("sample_size", None),
                task_type=_bg_seed.pop("task_type", task_type),
                feature_names=_bg_seed.pop("feature_names", feature_names),
                **_bg_seed,
            )
            render_interpretation_with_llm_button(ctx_seed, key="llm_seed_sens", result_session_key="llm_result_seed_sens", plot_type="seed_sensitivity")

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
    _dropout_transformer = st.session_state.get('target_transformer')
    _y_test_dropout = y_test
    if task_type == "regression" and _dropout_transformer is not None:
        if _dropout_transformer == 'log1p':
            preds_base = np.expm1(preds_base)
        else:
            preds_base = _dropout_transformer.inverse_transform(preds_base.reshape(-1, 1)).ravel()
        _y_test_dropout = st.session_state.get('y_test_original', y_test)
    if task_type == "regression":
        baseline_score = np.sqrt(mean_squared_error(_y_test_dropout, preds_base))
    else:
        baseline_score = accuracy_score(y_test, preds_base)

    # Ablate each feature: neutralize it to its training median/mode, re-fit
    # the model's own preprocessing pipeline, and retrain. Two bugs in the old
    # design: (1) clone() was called on the app's model WRAPPER, which is not
    # a sklearn estimator, so every feature errored and reported impact=0;
    # (2) the baseline went through the fitted pipeline while the retrain used
    # raw median-imputed features, so 'impact' conflated removing a feature
    # with removing all preprocessing. Neutralization (rather than column
    # removal) keeps the pipeline's column contract intact.
    _est_for_dropout = model_obj.get_model() if hasattr(model_obj, "get_model") else model_obj
    features_to_test = feature_names[:max_features]
    progress = st.progress(0, text="Running feature dropout (ablation)...")
    dropout_results = []
    st.caption(
        "Method: each feature is neutralized to its training median (numeric) "
        "or mode (categorical); the preprocessing pipeline is re-fit and the "
        "model retrained. Impact = how much held-out performance degrades "
        "without that feature's variation."
    )

    _X_train_df = X_train if hasattr(X_train, "columns") else pd.DataFrame(X_train, columns=feature_names)
    _X_test_df = X_test if hasattr(X_test, "columns") else pd.DataFrame(X_test, columns=feature_names)

    for i, feat in enumerate(features_to_test):
        try:
            X_tr_abl = _X_train_df.copy()
            X_te_abl = _X_test_df.copy()
            _col = X_tr_abl[feat]
            if pd.api.types.is_numeric_dtype(_col):
                _fill = _col.median()
            else:
                _mode = _col.mode(dropna=True)
                _fill = _mode.iloc[0] if len(_mode) else None
            X_tr_abl[feat] = _fill
            X_te_abl[feat] = _fill

            cloned = clone(_est_for_dropout)

            if pipeline is not None:
                _pipe_abl = clone(pipeline)
                _pipe_abl.fit(X_tr_abl)
                X_tr_vals = _pipe_abl.transform(X_tr_abl)
                X_te_vals = _pipe_abl.transform(X_te_abl)
            else:
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(strategy="median")
                X_tr_vals = imp.fit_transform(X_tr_abl.select_dtypes(include=[np.number]))
                X_te_vals = imp.transform(X_te_abl.select_dtypes(include=[np.number]))

            if hasattr(X_tr_vals, "toarray"):
                X_tr_vals = X_tr_vals.toarray()
                X_te_vals = X_te_vals.toarray()

            cloned.fit(X_tr_vals, y_train)
            preds_drop = cloned.predict(X_te_vals)

            if task_type == "regression" and _dropout_transformer is not None:
                if _dropout_transformer == 'log1p':
                    preds_drop = np.expm1(preds_drop)
                else:
                    preds_drop = _dropout_transformer.inverse_transform(preds_drop.reshape(-1, 1)).ravel()

            if task_type == "regression":
                drop_score = np.sqrt(mean_squared_error(_y_test_dropout, preds_drop))
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
            # NaN, not 0 — a failed ablation must not masquerade as "no impact"
            dropout_results.append({"feature": feat, "score_without": float("nan"), "impact": float("nan"), "_error": str(e)})

        progress.progress((i + 1) / len(features_to_test), text=f"Testing without '{feat}' ({i+1}/{len(features_to_test)})")

    progress.empty()

    _n_drop_errors = sum(1 for r in dropout_results if r.get("_error"))
    if _n_drop_errors:
        st.warning(f"{_n_drop_errors}/{len(dropout_results)} feature ablations failed — "
                   f"see the full results table for error details.")

    if dropout_results:
        df_dropout = pd.DataFrame(dropout_results).sort_values("impact", ascending=False)
        st.session_state["sensitivity_dropout_results"] = df_dropout
        st.session_state["sensitivity_dropout_baseline"] = baseline_score
        log_methodology(step='Sensitivity Analysis', action='Ran feature dropout analysis', details={
            'model': selected_model,
            'n_features_tested': len(features_to_test),
            'metric': primary_metric
        })
        try:
            from utils.workflow_provenance import get_provenance
            _prov = get_provenance()
            get_provenance().record_sensitivity(
                seed_stability=_prov.sensitivity.seed_stability if _prov.sensitivity else False,
                seed_stability_cv=_prov.sensitivity.seed_stability_cv if _prov.sensitivity else None,
                feature_dropout=True,
            )
        except Exception:
            pass  # Provenance recording should never break the workflow

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
            table(df_dropout[["feature", "score_without", "impact"]], key="feature_dropout")

        # LLM interpretation for feature dropout
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_drop = gather_session_context()
        _top_impact = "; ".join(f"{r['feature']}={r['impact']:.4f}" for _, r in df_dropout.head(5).iterrows())
        _noisy_str = "; ".join(f"{r['feature']}={r['impact']:.4f}" for _, r in df_dropout[df_dropout['impact'] < -0.001].iterrows()) if not df_dropout[df_dropout['impact'] < -0.001].empty else "none"
        _drop_summary = (f"baseline_{primary_metric}={baseline_score:.4f}; "
                         f"top_impacts: {_top_impact}; noisy_features: {_noisy_str}; "
                         f"n_features_tested={len(df_dropout)}; model={selected_model}")
        ctx_drop = build_llm_context(
            "feature_dropout", _drop_summary,
            model_name=selected_model,
            where="Feature dropout analysis",
            sample_size=_bg_drop.pop("sample_size", None),
            task_type=_bg_drop.pop("task_type", task_type),
            feature_names=_bg_drop.pop("feature_names", feature_names),
            **_bg_drop,
        )
        render_interpretation_with_llm_button(ctx_drop, key="llm_feat_drop", result_session_key="llm_result_feat_drop", plot_type="feature_dropout")

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
