"""
Theory Anchor System — bridges coaching insights to interactive Theory Reference demos.

Each anchor maps a concept key to:
- A plain-language "why this matters" summary (shown inline on coaching cards)
- A misconception to surface (the most common mistake)
- A "what to look for" prompt (trains the user's eye on their own results)
- The chapter and section in the Theory Reference for the full interactive demo

Usage in coaching_ui.py:
    from utils.theory_anchors import render_theory_link
    render_theory_link("skewness")  # renders expandable theory card inline

Usage in insight_ledger.py:
    Insight(..., theory_anchor="skewness")
"""
from typing import Optional, Dict, Any
import streamlit as st


# ---------------------------------------------------------------------------
# Anchor definitions
# ---------------------------------------------------------------------------
# Each anchor contains the educational content that surfaces inline on
# coaching cards. The full interactive demo lives on the Theory Reference page.

THEORY_ANCHORS: Dict[str, Dict[str, str]] = {
    # ── Chapter 1: Data Quality & Assumptions ──────────────────────────────
    "skewness": {
        "chapter": "Data Quality & Assumptions",
        "section": "Skewness",
        "why_it_matters": (
            "Skewed features stretch one tail of the distribution, pulling the mean "
            "away from the center. For linear, neural, and distance-based models this "
            "distorts optimization — a few extreme values dominate squared-error loss "
            "and inflate distances. Tree-based models are unaffected because they "
            "split on rank order, not magnitude."
        ),
        "misconception": (
            "A skewed feature is not automatically a problem for every model. "
            "Check which models you're training before deciding to transform."
        ),
        "what_to_look_for": (
            "In your EDA histograms, look for whether one tail stretches much "
            "farther than the other. If the mean and median are far apart, skew "
            "is affecting the feature's center of gravity."
        ),
    },
    "outliers": {
        "chapter": "Data Quality & Assumptions",
        "section": "Outliers",
        "why_it_matters": (
            "Outliers with high leverage can reshape a regression line by themselves. "
            "The danger is not just that a value is extreme — it's that an extreme "
            "value far from the data center pulls the model toward itself. "
            "Linear and neural models are most vulnerable."
        ),
        "misconception": (
            "High leverage alone is not dangerous. An outlier needs both unusual "
            "feature values AND a large residual to actually distort the model."
        ),
        "what_to_look_for": (
            "In your EDA outlier flags, check whether flagged observations are "
            "extreme in the feature space (high leverage) AND poorly predicted "
            "(large residual). If only one is true, the outlier may be harmless."
        ),
    },
    "collinearity": {
        "chapter": "Data Quality & Assumptions",
        "section": "Collinearity",
        "why_it_matters": (
            "When features are highly correlated, the model can't tell which one "
            "deserves credit. Coefficient estimates become unstable — they can flip "
            "sign between samples — even though predictions remain fine. "
            "This is purely an interpretation problem, not a prediction problem."
        ),
        "misconception": (
            "Collinearity does not hurt prediction accuracy. It hurts your ability "
            "to interpret individual coefficients. If you only care about ŷ, it's fine."
        ),
        "what_to_look_for": (
            "Look at the VIF values in your EDA output. VIF > 10 means the feature "
            "shares over 90% of its variance with other features. The coefficient "
            "on that feature is essentially noise for interpretation purposes."
        ),
    },
    "class_imbalance": {
        "chapter": "Data Quality & Assumptions",
        "section": "Class Imbalance",
        "why_it_matters": (
            "When one class dominates, a model can achieve high accuracy by ignoring "
            "the minority class entirely. Accuracy becomes a trap — it rewards doing "
            "nothing useful. Class-aware metrics (recall, F1, AUPRC) and weighted "
            "training correct for this."
        ),
        "misconception": (
            "Class imbalance does not mean the dataset is unusable. It means your "
            "training objective, threshold, and evaluation metrics need to reflect "
            "the asymmetry."
        ),
        "what_to_look_for": (
            "If accuracy is high but recall is low, the model is probably ignoring "
            "the minority class. Compare accuracy to the majority-class baseline — "
            "if they're close, accuracy is lying to you."
        ),
    },
    "sample_size": {
        "chapter": "Data Quality & Assumptions",
        "section": "Events Per Variable (EPV)",
        "why_it_matters": (
            "With too few observations per feature, the model fits noise instead of "
            "signal. The classic rule: at least 10–20 events per variable for linear "
            "models. Neural networks need much more. Trees are moderately robust."
        ),
        "misconception": (
            "A small dataset doesn't mean you can't do ML — it means you need simpler "
            "models and stronger regularization. Choosing a neural network on n=80 "
            "is almost always a mistake."
        ),
        "what_to_look_for": (
            "Check the EPV ratio on the Upload page. If it's below 10, prefer "
            "linear or tree-based models. If it's below 5, consider whether the "
            "prediction task is feasible at all with this data."
        ),
    },
    "high_dimensionality": {
        "chapter": "Data Quality & Assumptions",
        "section": "The Curse of Dimensionality",
        "why_it_matters": (
            "As features grow relative to samples, 'local' neighborhoods cover "
            "nearly the entire dataset. Distance-based methods like KNN degrade "
            "because all points become approximately equidistant. Feature selection "
            "or dimensionality reduction becomes necessary."
        ),
        "misconception": (
            "More features is not always better. Beyond a point, each additional "
            "feature adds more noise than signal, and the model needs exponentially "
            "more data to fill the space."
        ),
        "what_to_look_for": (
            "If p/n > 0.5, be cautious with distance-based and unregularized linear "
            "models. If p > n, you must use regularization or dimensionality reduction."
        ),
    },
    "missing_data": {
        "chapter": "Preprocessing Theory",
        "section": "Missing Data Mechanisms",
        "why_it_matters": (
            "How you handle missing data depends on WHY it's missing. If missingness "
            "is related to the missing value itself (MNAR), simple imputation introduces "
            "bias. The mechanism determines the method."
        ),
        "misconception": (
            "Mean imputation is not 'safe.' It shrinks variance, distorts correlations, "
            "and can introduce bias. KNN or iterative imputation preserve more structure."
        ),
        "what_to_look_for": (
            "Check the missingness pattern: is it random across features, or concentrated "
            "in certain variables? If one feature is 40%+ missing, imputed values dominate "
            "that column — the model is partly predicting from guesses."
        ),
    },

    # ── Chapter 2: Feature Engineering & Selection ─────────────────────────
    "leakage": {
        "chapter": "Feature Engineering & Selection",
        "section": "Information Leakage",
        "why_it_matters": (
            "If any preprocessing step — feature selection, scaling, imputation — uses "
            "test data, the model has seen information it shouldn't have. Performance "
            "estimates become optimistically biased. This is the most common silent "
            "methodological error in published ML work."
        ),
        "misconception": (
            "Leakage doesn't just mean 'the target leaked into features.' It includes "
            "any step where test-set information influences training decisions, including "
            "feature selection and hyperparameter tuning."
        ),
        "what_to_look_for": (
            "If train and test performance are suspiciously close (< 1% gap on a complex "
            "model), leakage is the first hypothesis. A 0.1% gap is a red flag."
        ),
    },

    # ── Chapter 4: Preprocessing Theory ────────────────────────────────────
    "scaling": {
        "chapter": "Preprocessing Theory",
        "section": "Scaling",
        "why_it_matters": (
            "Features on different scales cause distance-based and gradient-based "
            "models to weight them unevenly — a feature ranging 0–1000 dominates "
            "one ranging 0–1, regardless of actual importance."
        ),
        "misconception": (
            "Tree-based models do not need scaling. Scaling them is unnecessary work "
            "that can hurt interpretability without improving performance."
        ),
        "what_to_look_for": (
            "Check whether your selected models include any linear, neural, KNN, or "
            "SVM models. If so, scaling is required. If only trees, skip it."
        ),
    },
    "transforms": {
        "chapter": "Data Quality & Assumptions",
        "section": "Corrective Transforms",
        "why_it_matters": (
            "Log and Box-Cox transforms compress long tails, making skewed distributions "
            "more symmetric. This helps linear models by reducing the influence of "
            "extreme values on squared-error loss."
        ),
        "misconception": (
            "Not every skewed feature needs a transform. If you're only training trees, "
            "the skew doesn't matter. Transforms change coefficient interpretation in "
            "linear models — be prepared to back-transform."
        ),
        "what_to_look_for": (
            "Compare the before/after distributions on the Preprocess page. The transform "
            "should compress the long tail without over-correcting into opposite skew."
        ),
    },

    # ── Chapter 5: Evaluation & Validation ─────────────────────────────────
    "calibration": {
        "chapter": "Evaluation & Validation",
        "section": "Calibration",
        "why_it_matters": (
            "A model can rank cases correctly (high AUROC) while still reporting "
            "probabilities that are systematically wrong. If predicted 80% means "
            "actual 60%, the probabilities can't be used for risk communication "
            "or threshold-based decisions."
        ),
        "misconception": (
            "High AUROC does not guarantee calibrated probabilities. Discrimination "
            "and calibration are independent properties."
        ),
        "what_to_look_for": (
            "On the calibration plot in Train & Compare, check whether the curve "
            "follows the diagonal. If it bows below, the model is overconfident. "
            "If above, underconfident. ECE > 0.05 warrants attention."
        ),
    },
    "cross_validation": {
        "chapter": "Evaluation & Validation",
        "section": "k-Fold Cross-Validation",
        "why_it_matters": (
            "A single train/test split gives one estimate that depends on which "
            "observations happened to land where. Cross-validation gives k estimates, "
            "revealing both expected performance and its uncertainty."
        ),
        "misconception": (
            "Cross-validation does not give k independent proofs. It gives repeated "
            "estimates of how the same procedure behaves under different partitions."
        ),
        "what_to_look_for": (
            "Look at the spread across folds, not just the mean. A model with mean "
            "AUROC 0.84 ± 0.09 is much less trustworthy than one with 0.82 ± 0.02."
        ),
    },
    "threshold_choice": {
        "chapter": "Evaluation & Validation",
        "section": "Classification Metrics",
        "why_it_matters": (
            "The default threshold of 0.5 is arbitrary. Lowering it catches more "
            "true positives (higher recall) but also more false positives (lower "
            "precision). The right threshold is a domain decision, not a statistical one."
        ),
        "misconception": (
            "There is no 'correct' threshold. A cancer screening model and a spam "
            "filter need completely different thresholds even if they use the same algorithm."
        ),
        "what_to_look_for": (
            "If recall is low but precision is high, consider lowering the threshold. "
            "If false positives are costly, raise it. Look at the precision-recall tradeoff."
        ),
    },

    # ── Chapter 5: Explainability ──────────────────────────────────────────
    "shap": {
        "chapter": "Evaluation & Validation",
        "section": "SHAP Values",
        "why_it_matters": (
            "SHAP values decompose each prediction into per-feature contributions "
            "that sum to the difference from the baseline. They explain how the "
            "model uses features, not whether those features cause the outcome."
        ),
        "misconception": (
            "SHAP importance is not causality. A feature can have high SHAP values "
            "because it's a proxy or correlate, not because it drives the outcome."
        ),
        "what_to_look_for": (
            "On the SHAP beeswarm, look for features with wide horizontal spread — "
            "they matter most. Then check the color pattern: consistent color on one "
            "side means a clear directional relationship."
        ),
    },
    "pdp_ice": {
        "chapter": "Evaluation & Validation",
        "section": "Partial Dependence",
        "why_it_matters": (
            "PDP shows the average effect of a feature; ICE shows individual effects. "
            "When ICE lines diverge, the feature interacts with something else — the "
            "PDP average can hide this by canceling opposing effects."
        ),
        "misconception": (
            "A flat PDP does not prove a feature is unimportant. It may mean the "
            "feature's effect depends on other variables — check ICE lines."
        ),
        "what_to_look_for": (
            "If the PDP is flat but SHAP says the feature matters, overlay ICE lines. "
            "Diverging ICE lines reveal interactions that the average hides."
        ),
    },

    # ── Chapter 7: Sensitivity & Robustness ────────────────────────────────
    "seed_sensitivity": {
        "chapter": "Sensitivity & Robustness",
        "section": "Seed Sensitivity",
        "why_it_matters": (
            "If your results change substantially with a different random seed, "
            "the finding is fragile — it depends on the particular data split, "
            "not a real pattern."
        ),
        "misconception": (
            "A single good seed is not evidence of a reliable model. If other seeds "
            "tell a different story, the uncertainty belongs in the paper."
        ),
        "what_to_look_for": (
            "Look at the CV% across seeds. Below 2% is robust. Above 10% means "
            "the single-seed result you'd report is not representative."
        ),
    },
    "bootstrap": {
        "chapter": "Sensitivity & Robustness",
        "section": "Bootstrap Stability",
        "why_it_matters": (
            "Bootstrap confidence intervals quantify how much a metric would move "
            "if you had drawn a slightly different sample. Overlapping CIs between "
            "models means the difference may not be real."
        ),
        "misconception": (
            "A 95% CI does not mean 95% probability the true value is inside. "
            "It means this procedure captures the true value about 95% of the time "
            "across repeated samples."
        ),
        "what_to_look_for": (
            "If two models' bootstrap CIs overlap substantially, the evidence that "
            "one is truly better is weak — regardless of which point estimate is higher."
        ),
    },

    # ── Model-family-specific ──────────────────────────────────────────────
    "regularization": {
        "chapter": "Data Quality & Assumptions",
        "section": "Collinearity",
        "why_it_matters": (
            "Regularization (Ridge/LASSO) stabilizes coefficient estimates when "
            "features are correlated. Ridge shrinks correlated coefficients toward "
            "each other; LASSO sets some to zero. Neither discovers which variable "
            "is 'truly important.'"
        ),
        "misconception": (
            "Regularization is not feature selection in the causal sense. LASSO "
            "picks variables for pragmatic prediction, not because they're "
            "biologically fundamental."
        ),
        "what_to_look_for": (
            "If using Ridge and coefficients look stable, regularization is working. "
            "If using LASSO and it zeros out a feature you expected to matter, it may "
            "be correlated with another feature that LASSO kept instead."
        ),
    },
    "bias_variance": {
        "chapter": "Model Families",
        "section": "Distance-Based",
        "why_it_matters": (
            "Every model has a flexibility knob: too flexible and it memorizes noise "
            "(overfitting); too rigid and it misses the real pattern (underfitting). "
            "The sweet spot depends on your data's complexity and sample size."
        ),
        "misconception": (
            "A model with perfect training performance is not a good model — it's "
            "almost certainly overfit. The goal is good test performance, which requires "
            "controlled complexity."
        ),
        "what_to_look_for": (
            "Compare train vs test metrics. A large gap (e.g., train R² = 0.95, "
            "test R² = 0.60) signals overfitting. Consider simpler models or stronger "
            "regularization."
        ),
    },
}

# ---------------------------------------------------------------------------
# Mapping: insight category × model_scope → theory anchor key
# ---------------------------------------------------------------------------
# Used to auto-assign theory_anchor when insights are created.
# Key: (category, issue_subtype) → anchor key
# issue_subtype comes from the insight ID pattern or metadata.

INSIGHT_CATEGORY_TO_ANCHOR = {
    "skewness": "skewness",
    "outliers": "outliers",
    "collinearity": "collinearity",
    "class_imbalance": "class_imbalance",
    "missing_data": "missing_data",
    "high_dimensionality": "high_dimensionality",
    "low_sample_size": "sample_size",
    "feature_scale": "scaling",
    "leakage": "leakage",
    "calibration": "calibration",
    "seed_sensitivity": "seed_sensitivity",
    "non_normality": "skewness",  # close enough conceptually
}


def get_theory_anchor(anchor_key: str) -> Optional[Dict[str, str]]:
    """Look up a theory anchor by key. Returns None if not found."""
    return THEORY_ANCHORS.get(anchor_key)


def infer_theory_anchor(insight: "Insight") -> Optional[str]:
    """Try to infer the best theory anchor key for an insight.

    Checks:
    1. Explicit theory_anchor field on the insight
    2. Keyword matching from insight.id and insight.category
    """
    # 1. Explicit
    anchor = getattr(insight, "theory_anchor", None)
    if anchor and anchor in THEORY_ANCHORS:
        return anchor

    # 2. Match from category-based mapping
    for keyword, anchor_key in INSIGHT_CATEGORY_TO_ANCHOR.items():
        if keyword in insight.id.lower() or keyword in insight.category.lower():
            return anchor_key

    # 3. Check finding text for known concept keywords
    finding_lower = (insight.finding or "").lower()
    for keyword, anchor_key in [
        ("skew", "skewness"),
        ("outlier", "outliers"),
        ("collinear", "collinearity"),
        ("vif", "collinearity"),
        ("imbalance", "class_imbalance"),
        ("imbalanced", "class_imbalance"),
        ("missing", "missing_data"),
        ("imput", "missing_data"),
        ("calibrat", "calibration"),
        ("ece", "calibration"),
        ("leakage", "leakage"),
        ("seed", "seed_sensitivity"),
        ("bootstrap", "bootstrap"),
        ("dimension", "high_dimensionality"),
        ("epv", "sample_size"),
        ("sample size", "sample_size"),
        ("scaling", "scaling"),
        ("shap", "shap"),
        ("regulariz", "regularization"),
    ]:
        if keyword in finding_lower:
            return anchor_key

    return None


def render_theory_link(
    anchor_key: str,
    context_hint: str = "",
    compact: bool = False,
    show_demo: bool = True,
    page_context: str = "coaching",
) -> bool:
    """Render an inline theory expansion for a coaching card.

    Args:
        anchor_key: Key into THEORY_ANCHORS
        context_hint: Optional text to prepend (e.g., "BMI has skewness = 3.2")
        compact: If True, render minimal (just the link, no expansion)
        show_demo: If True, embed the interactive demo inline when available
        page_context: Unique prefix for widget keys (avoids collisions across pages)

    Returns:
        True if rendered, False if anchor not found.
    """
    anchor = THEORY_ANCHORS.get(anchor_key)
    if not anchor:
        return False

    label = f"📖 Understand why: {anchor['section']}"

    if compact:
        st.caption(f"{label} → Theory Reference: {anchor['chapter']}")
        return True

    with st.expander(label, expanded=False):
        # Why it matters (the core educational content)
        st.markdown(f"**Why this matters for your analysis:**")
        st.markdown(anchor["why_it_matters"])

        # What to look for in YOUR results
        if anchor.get("what_to_look_for"):
            st.markdown(f"**What to look for in your results:**")
            st.markdown(anchor["what_to_look_for"])

        # Common mistake
        if anchor.get("misconception"):
            st.markdown(
                f'<div style="background:#fef2f2; border-left:3px solid #dc2626; '
                f'border-radius:0 6px 6px 0; padding:0.7rem 1rem; margin:0.5rem 0; '
                f'font-size:0.85rem; color:#7f1d1d; line-height:1.5;">'
                f'⚠️ <strong>Common mistake:</strong> {anchor["misconception"]}</div>',
                unsafe_allow_html=True,
            )

        # Inline interactive demo if available — rendered directly, no sub-expander
        if show_demo:
            from utils.theory_demos import render_inline_demo
            render_inline_demo(anchor_key, page_context=page_context, expanded=True, wrapped=False)

        # Clean reference to the full Theory Reference section (no navigation link — avoids wiping session state)
        st.caption(f"📚 Dive deeper → Theory Reference · {anchor['chapter']} · {anchor['section']}")

    return True
