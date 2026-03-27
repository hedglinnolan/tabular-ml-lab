"""
📖 Theory Reference — Statistical foundations behind every page of Tabular ML Lab.

Structured as a browsable reference: selectbox picks the chapter, tabs organize
parallel content (e.g., model families), and expanders offer optional deep dives.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.theme import inject_custom_css, render_sidebar_workflow

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Theory Reference | Tabular ML Lab",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()
render_sidebar_workflow(current_page="11_Theory")

# ── Local styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chapter intro card — gradient card, white text on purple/blue */
.theory-intro {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
    font-size: 0.95rem;
    line-height: 1.65;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
}
.theory-intro strong { color: #fff; font-weight: 700; }
.theory-intro em { color: rgba(255,255,255,0.9); }

/* Section heading inside chapters */
.theory-section-head {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.6rem 0 0.6rem 0;
    letter-spacing: -0.01em;
}

/* Citation badge */
.cite {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    font-size: 0.78rem;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    margin-left: 0.3rem;
    vertical-align: middle;
    font-weight: 600;
    border: 1px solid #ddd6fe;
}

/* App connection callout — green accent on light bg */
.app-callout {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #14532d;
    line-height: 1.55;
}
.app-callout strong { color: #052e16; }

/* Key takeaway box — amber accent on light bg */
.key-takeaway {
    background: #fffbeb;
    border-left: 3px solid #d97706;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #713f12;
    line-height: 1.55;
}
.key-takeaway strong { color: #451a03; }

/* Worked example card */
.worked-example {
    background: #eff6ff;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #1e3a8a;
    line-height: 1.6;
}
.worked-example strong { color: #1e40af; }

/* Visual guidance card */
.visual-guide {
    background: #ecfeff;
    border-left: 3px solid #0891b2;
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #164e63;
    line-height: 1.6;
}
.visual-guide strong { color: #155e75; }

/* Misconception / reviewer trap */
.misconception-box {
    background: #fef2f2;
    border-left: 3px solid #dc2626;
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #7f1d1d;
    line-height: 1.6;
}
.misconception-box strong { color: #991b1b; }

/* Self-check prompt */
.self-check {
    background: #f5f3ff;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #4c1d95;
    line-height: 1.6;
}
.self-check strong { color: #5b21b6; }

/* Guidance card */
.guidance-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.95rem 1.2rem;
    margin: 0.5rem 0 1.2rem 0;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.6;
}
.guidance-card strong { color: #1e293b; }

/* Page hero */
.theory-hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.theory-hero h1 {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.theory-hero .subtitle {
    color: #475569;
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="theory-hero">
    <h1>📖 Theory Reference</h1>
    <p class="subtitle">
        The statistical reasoning behind every recommendation, coaching prompt, and
        model assumption in Tabular ML Lab — written for researchers who want to
        understand <em>why</em>, not just <em>what</em>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    """
<div class="guidance-card">
<strong>How to use this page as a learner:</strong> move in this order — first read <em>why the concept matters</em>, then study the equation, then use the worked example and visual prompt to build intuition, and finally check yourself with the misconception and self-check boxes. The goal is not to memorize formulas, but to understand what the app is doing and why a reviewer would care.
</div>
""",
    unsafe_allow_html=True,
)

# ── Chapter selector ─────────────────────────────────────────────────────────
CHAPTERS = [
    "Data Quality & Assumptions",
    "Feature Engineering & Selection",
    "Model Families",
    "Preprocessing Theory",
    "Evaluation & Validation",
    "Statistical Testing",
    "Sensitivity & Robustness",
    "Reporting Standards (TRIPOD)",
]

chapter = st.selectbox(
    "Choose a chapter",
    CHAPTERS,
    index=0,
    help="Each chapter covers the theory behind one or more pages of the app.",
)

st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions for consistent rendering
# ══════════════════════════════════════════════════════════════════════════════

def intro(text: str):
    """Render a chapter introduction card."""
    st.markdown(f'<div class="theory-intro">{text}</div>', unsafe_allow_html=True)


def section(title: str):
    """Render a section heading within a chapter."""
    st.markdown(f'<div class="theory-section-head">{title}</div>', unsafe_allow_html=True)


def app_connection(text: str):
    """Render a callout linking theory to the app."""
    st.markdown(f'<div class="app-callout">🔬 <strong>In the app:</strong> {text}</div>', unsafe_allow_html=True)


def cite(text: str):
    """Return a citation badge HTML string for inline use."""
    return f'<span class="cite">{text}</span>'


def takeaway(text: str):
    """Render a key takeaway box."""
    st.markdown(f'<div class="key-takeaway">💡 <strong>Key takeaway:</strong> {text}</div>', unsafe_allow_html=True)


def worked_example(text: str):
    """Render a worked example box."""
    st.markdown(f'<div class="worked-example">🧮 <strong>Worked example:</strong> {text}</div>', unsafe_allow_html=True)


def visual_guide(text: str):
    """Render a visual attention prompt."""
    st.markdown(f'<div class="visual-guide">👀 <strong>What to look for:</strong> {text}</div>', unsafe_allow_html=True)


def misconception(text: str):
    """Render a misconception / reviewer trap box."""
    st.markdown(f'<div class="misconception-box">⚠️ <strong>Common misconception:</strong> {text}</div>', unsafe_allow_html=True)


def self_check(text: str):
    """Render a short self-explanation prompt."""
    st.markdown(f'<div class="self-check">🧠 <strong>Self-check:</strong> {text}</div>', unsafe_allow_html=True)


def references(refs: list[str]):
    """Render a references section at the bottom of a chapter or tab."""
    st.markdown("---")
    st.markdown(f'<div class="theory-section-head">References</div>', unsafe_allow_html=True)
    for ref in refs:
        st.markdown(f"- {ref}")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 1: Data Quality & Assumptions
# ══════════════════════════════════════════════════════════════════════════════

def render_data_quality():
    intro(
        "Before any model is trained, the quality and structure of your data "
        "determines what methods are appropriate and what conclusions are defensible. "
        "This chapter covers the diagnostic checks the app performs during "
        "<strong>Upload & Audit</strong> and <strong>EDA</strong>, and the theory behind them."
    )

    tabs = st.tabs([
        "Missing Data",
        "Distributional Shape",
        "Outliers",
        "Collinearity",
        "Sample Size",
    ])

    # ── Missing Data ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
Missing data is rarely random. The *mechanism* that produced the missingness determines
which statistical remedies are valid and which will silently bias your results. Rubin (1976)
formalized three categories that remain the foundation of modern missing data analysis.
""")
        section("The Three Missing Data Mechanisms")
        st.markdown("""
**Missing Completely at Random (MCAR)** means the probability of a value being missing
is unrelated to both the observed and unobserved data. For example, a lab instrument
fails at random intervals, losing samples regardless of their values. Under MCAR,
the observed data is a simple random subsample of the complete data, so any analysis
on the complete cases is unbiased — though less powerful due to the reduced sample size.
""")
        st.markdown("""
**Missing at Random (MAR)** means the probability of missingness depends on *observed*
variables but not on the missing value itself, after conditioning on those observed
variables. For instance, younger patients may be less likely to have a cholesterol
measurement recorded — the missingness depends on age (observed) but not on the
cholesterol value itself. Under MAR, methods that properly condition on the observed
data — such as multiple imputation or likelihood-based approaches — can produce
unbiased estimates.
""")
        st.markdown("""
**Missing Not at Random (MNAR)** means the probability of missingness depends on
the missing value itself, even after conditioning on all observed data. A patient
with dangerously high blood pressure may be too ill to attend a follow-up visit,
so the missing blood pressure values are systematically higher than the observed
ones. MNAR is the most challenging scenario: no purely statistical remedy exists
without modeling the missingness mechanism directly.
""")
        st.markdown("""
These three mechanisms can be expressed formally. Let **R** denote a missingness
indicator (0 = missing, 1 = observed), **Y_obs** denote the values we *can* see,
and **Y_mis** denote the values we *cannot*. The question is: what does the
probability of being missing depend on?
""")
        st.latex(r"""
        \begin{aligned}
        \text{MCAR:} \quad & P(R = 0 \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(R = 0) \\
        \text{MAR:} \quad & P(R = 0 \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(R = 0 \mid Y_{\text{obs}}) \\
        \text{MNAR:} \quad & P(R = 0 \mid Y_{\text{obs}}, Y_{\text{mis}}) \text{ depends on } Y_{\text{mis}}
        \end{aligned}
        """)
        st.markdown("""
Reading these from top to bottom: under MCAR, the left-hand side simplifies to a
constant — nothing predicts missingness. Under MAR, the probability of being missing
can depend on the data you *did* observe (Y_obs), but not on the value that's
actually missing (Y_mis). Under MNAR, the missing value itself influences whether
it's missing — and that's the scenario no imputation method can fully correct without
explicitly modeling *why* data is missing.
""")

        section("Why the Mechanism Matters for Imputation")
        st.markdown(f"""
The choice of imputation strategy is not a free parameter — it carries assumptions
about the missingness mechanism. {cite("Rubin, 1976")} {cite("van Buuren, 2018")}

**Mean/median imputation** replaces missing values with the column's central tendency.
This is simple and fast, but it *always* underestimates variance and can attenuate
correlations between variables. Under MCAR, the imputed mean is unbiased but the
standard errors are too small. Under MAR or MNAR, the imputed mean is biased.

**K-nearest neighbors (KNN) imputation** finds the *k* most similar complete observations
(using observed features) and averages their values. This is implicitly a MAR method:
it conditions on observed features to predict the missing value. It preserves local
structure better than mean imputation, but it is sensitive to the distance metric and
struggles when missingness is widespread across many features simultaneously.

**Iterative imputation (MICE)** — Multiple Imputation by Chained Equations — models
each feature with missing values as a function of all other features, cycling through
the features iteratively until convergence. {cite("van Buuren & Groothuis-Oudshoorn, 2011")}
This is the gold standard under MAR: it preserves multivariate relationships
and, when run as *multiple* imputation (generating several completed datasets), it
correctly propagates uncertainty from the missing data into downstream inference.
""", unsafe_allow_html=True)

        app_connection(
            "The <strong>EDA</strong> page reports missing data percentages per feature and flags "
            "features with >5% missingness. The <strong>Preprocess</strong> page offers mean, median, "
            "KNN, and iterative imputation — each configured per-model pipeline. The coaching layer "
            "recommends iterative imputation when missingness is widespread, and warns about mean "
            "imputation when features are correlated."
        )

        with st.expander("Deep Dive: Testing for MCAR — Little's Test"):
            st.markdown(f"""
Little's MCAR test (1988) is a chi-squared test of whether the means of observed
values differ across missing data patterns. Under the null hypothesis of MCAR,
the observed means should be approximately equal across all patterns of missingness.

The test statistic follows a χ² distribution with degrees of freedom equal to
the number of constraints. A significant result (p < 0.05) rejects MCAR, suggesting
the data is MAR or MNAR — but it cannot distinguish between these two.
{cite("Little, 1988")}

**Limitation:** Little's test has low power with small samples or many missing data
patterns. A non-significant result does not confirm MCAR — it may simply reflect
insufficient power to detect departures.
""", unsafe_allow_html=True)

        references([
            "Rubin, D.B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.",
            "van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press.",
            "van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.",
            "Little, R.J.A. (1988). A test of missing completely at random for multivariate data with missing values. *Journal of the American Statistical Association*, 83(404), 1198–1202.",
        ])

    # ── Distributional Shape ─────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
The distributional shape of your features — their symmetry, tail behavior, and
concentration — is not just a descriptive curiosity. It directly determines which
models will perform well and which preprocessing steps are necessary.
""")
        section("Skewness")
        st.markdown(f"""
Skewness measures the asymmetry of a distribution around its mean. A distribution
with a long right tail (e.g., income, medical costs) has positive skew; a long left
tail gives negative skew. The sample skewness is computed as:
""")
        st.latex(r"""
        \gamma_1 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
        """)
        st.markdown(f"""
Each observation is first standardized — centered by the mean x̄ and scaled by the
standard deviation s — so the result is unitless and comparable across features.
The cubing is what makes this measure sensitive to *asymmetry*: positive deviations
cubed remain positive, negative deviations cubed remain negative. If the distribution
is symmetric, these cancel out and γ₁ ≈ 0. If the right tail is longer, the large
positive cubed terms dominate, giving γ₁ > 0. As a rough guide: |γ₁| < 0.5 is
approximately symmetric, |γ₁| between 0.5 and 1 is moderately skewed, and
|γ₁| > 1 is heavily skewed.
""")
        with st.expander("🧮 Interactive: See how skewness changes a distribution"):
            skew_alpha = st.slider(
                "Skewness intensity",
                min_value=1.0, max_value=20.0, value=2.0, step=0.5,
                key="theory_skew_alpha",
                help="Slide right to increase right-skew. At low values the distribution is nearly symmetric.",
            )
            rng = np.random.default_rng(42)
            # Higher slider value → smaller gamma shape → more skewed
            skew_data = rng.gamma(shape=max(0.1, 5.0 / skew_alpha), scale=1.0, size=800)

            from scipy.stats import skew as calc_skew
            computed_skew = calc_skew(skew_data)

            mean_val = np.mean(skew_data)
            median_val = np.median(skew_data)

            fig_skew = go.Figure()
            fig_skew.add_trace(go.Histogram(
                x=skew_data, nbinsx=40,
                marker_color="rgba(99, 102, 241, 0.7)",
                marker_line=dict(color="rgba(99, 102, 241, 1)", width=1),
            ))
            fig_skew.add_vline(x=mean_val, line_dash="dash", line_color="#dc2626",
                               annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
            fig_skew.add_vline(x=median_val, line_dash="dot", line_color="#16a34a",
                               annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
            label = "approximately symmetric" if abs(computed_skew) < 0.5 else "moderately skewed" if abs(computed_skew) < 1 else "heavily skewed"
            fig_skew.update_layout(
                title=f"γ₁ = {computed_skew:.2f} — {label}",
                xaxis_title="Value", yaxis_title="Count",
                height=320, margin=dict(t=50, b=40, l=50, r=30),
                template="plotly_white",
            )
            st.plotly_chart(fig_skew, use_container_width=True)
            st.markdown(
                "**Train your eye:** Watch two things as you slide. "
                "First, the **gap between the red mean and green median** — in a symmetric distribution they overlap; "
                "as skew grows, the mean chases the tail while the median stays put. "
                "Second, notice how a handful of extreme observations in the right tail **stretch the x-axis** — "
                "these are the values that dominate squared-error loss in linear models and inflate distances in KNN."
            )

        misconception(
            "A skewed feature is not automatically a problem for every model. For tree-based models, skewness often matters very little; for linear, neural, and distance-based models, it can materially affect optimization, coefficients, and distance calculations."
        )

        self_check(
            "If you log-transform a heavily right-skewed feature, what phenomenon are you trying to reduce: the number of observations, the asymmetry of the tail, or the correlation with the target?"
        )

        st.markdown(f"""
**Why skewness matters — and for which models.** Skewness affects models
differently depending on their mathematical assumptions: {cite("ISLR, §3.3")}

- **Linear models** (Ridge, LASSO, Elastic Net) minimize squared error, which
  gives outsized influence to observations in the long tail. A single extreme value
  in a skewed feature can shift the fitted coefficient substantially. Skewness also
  violates the normality assumption needed for valid confidence intervals on
  coefficients, though the estimates themselves remain consistent.

- **Neural networks** use gradient-based optimization. Skewed features create an
  asymmetric loss landscape: gradients from extreme values can dominate updates,
  causing unstable training. This is compounded by activation functions like sigmoid
  or tanh that saturate in one direction.

- **Distance-based methods** (KNN, SVM) compute distances in feature space. A skewed
  feature with a long tail has an inflated range, causing it to dominate the distance
  calculation over other features — even after standard scaling, which centers on the
  mean but does not correct the asymmetry.

- **Tree-based methods** (Random Forest, Gradient Boosting) are invariant to monotone
  transformations of features. Since trees make decisions based on *rank order* (split
  points), it does not matter whether a feature is skewed — the splits are the same.
  This is one of the key practical advantages of tree methods. {cite("Hastie et al., ESL §9.2")}
""", unsafe_allow_html=True)

        section("Kurtosis")
        st.markdown(f"""
Kurtosis measures the heaviness of the tails relative to a normal distribution. The
"excess kurtosis" (subtracting 3, the value for a normal distribution) tells you
whether extreme values are more or less likely than a Gaussian would predict.
""")
        st.latex(r"""
        \kappa = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 - 3
        """)
        st.markdown("""
The structure is similar to skewness, but the standardized deviations are raised to
the *fourth* power instead of the third. This amplifies extreme values even more
aggressively — an observation 3 standard deviations from the mean contributes
81 times as much as one at 1 standard deviation (3⁴ = 81). The subtraction of 3
re-centers the measure so that a normal distribution has excess kurtosis of zero;
values above zero indicate **heavier tails than normal** (more extreme outliers than
a bell curve would produce), and values below zero indicate **lighter tails**
(observations are more tightly clustered, with fewer extremes).

High kurtosis is a warning sign for linear models and any method that uses
squared error: a few extreme observations contribute disproportionately to the
loss. It is also a signal that standard confidence intervals (which assume normality)
may have poor coverage — the true sampling distribution of the mean has heavier
tails than the normal approximation suggests.
""")

        app_connection(
            "The <strong>EDA</strong> page computes skewness and kurtosis for every numeric feature "
            "and flags features exceeding thresholds (|skewness| > 1, |kurtosis| > 3). "
            "The coaching layer specifically notes which model families are affected: "
            "linear and neural models get a warning; tree-based models do not."
        )

        with st.expander("Deep Dive: Transforms for Skewness"):
            st.markdown(f"""
When skewness is a problem for your chosen models, the standard remedy is a
variance-stabilizing transform. The goal is to compress the long tail, making
the distribution more symmetric.

**Log transform** (log(x) or log(1+x) for data with zeros): The most common
choice for right-skewed positive data. It compresses large values aggressively
and has a natural interpretation — a unit change in log(x) corresponds to a
percentage change in x. However, it requires strictly positive values and can
over-correct moderate skew.

**Box-Cox transform:** A parametric family that includes the log as a special case. {cite("Box & Cox, 1964")}
""", unsafe_allow_html=True)
            st.latex(r"""
            y^{(\lambda)} = \begin{cases}
            \dfrac{x^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\[8pt]
            \log(x) & \text{if } \lambda = 0
            \end{cases}
            """)
            st.markdown(f"""
The parameter λ controls *how aggressively* the transform compresses the upper tail.
The key insight is that different values of λ recover familiar transforms: λ = 1 is
no transform at all (just a shift), λ = 0.5 gives the square root, λ = 0 gives the
natural log, and λ = −1 gives the reciprocal. The optimal λ is chosen by maximum
likelihood — the value that makes the transformed data most closely resemble a normal
distribution.
""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**🧮 Interactive: Before / after transform**")
            transform_lambda = st.slider(
                "Box-Cox λ",
                min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
                key="theory_transform_lambda",
                help="λ = 1: no change. λ = 0.5: square root. λ = 0: log. λ < 0: reciprocal family.",
            )
            rng_t = np.random.default_rng(7)
            raw_data = rng_t.exponential(scale=10.0, size=500) + 1  # right-skewed, strictly positive
            if abs(transform_lambda) < 0.05:
                transformed = np.log(raw_data)
            else:
                transformed = (np.power(raw_data, transform_lambda) - 1) / transform_lambda

            from scipy.stats import skew as calc_skew_t
            raw_skew = calc_skew_t(raw_data)
            trans_skew = calc_skew_t(transformed)

            fig_tf = make_subplots(rows=1, cols=2, subplot_titles=[
                f"Raw (γ₁ = {raw_skew:.2f})",
                f"Transformed, λ = {transform_lambda:.1f} (γ₁ = {trans_skew:.2f})",
            ])
            fig_tf.add_trace(go.Histogram(x=raw_data, nbinsx=35,
                marker_color="rgba(220, 38, 38, 0.6)", marker_line=dict(color="rgba(220, 38, 38, 1)", width=1),
                showlegend=False), row=1, col=1)
            fig_tf.add_trace(go.Histogram(x=transformed, nbinsx=35,
                marker_color="rgba(22, 163, 74, 0.6)", marker_line=dict(color="rgba(22, 163, 74, 1)", width=1),
                showlegend=False), row=1, col=2)
            fig_tf.update_layout(height=280, margin=dict(t=40, b=30, l=40, r=20), template="plotly_white")
            st.plotly_chart(fig_tf, use_container_width=True)
            st.markdown(
                "**Train your eye:** Compare the two histograms as you move λ. "
                "At λ = 1 they're identical (no transform). As λ drops toward 0, watch the right tail in the green plot **compress** "
                "while the bulk of the data **fans out** — you're trading tail dominance for better resolution in the middle. "
                "Go past 0 into negative λ: the distribution over-corrects into left skew. "
                "The γ₁ values in each title track the change numerically."
            )

            st.markdown(f"""
**Yeo-Johnson transform:** Extends Box-Cox to handle zero and negative values by
using a modified formula for non-positive inputs. {cite("Yeo & Johnson, 2000")}
This is the most general-purpose option in the app, since it works regardless
of the feature's range.

**When NOT to transform:** If your only models are tree-based, skewness correction
is unnecessary and may even hurt interpretability. Transformations change the scale
of coefficients in linear models — be prepared to back-transform for interpretation.
""", unsafe_allow_html=True)

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §3.3. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §9.2. Springer.",
            "Box, G.E.P. & Cox, D.R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society, Series B*, 26(2), 211–252.",
            "Yeo, I.-K. & Johnson, R.A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954–959.",
        ])

    # ── Outliers ─────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
An outlier is an observation that lies far from the bulk of the data. But "far"
is a loaded word — it depends on what model of the data you're assuming, and
whether the extreme value reflects genuine variation, measurement error, or a
fundamentally different process.
""")
        section("Statistical Outliers vs. Domain Outliers")
        st.markdown(f"""
**Statistical outliers** are identified purely by their distance from the center
of the distribution — typically defined as falling beyond a threshold number of
standard deviations, or outside the interquartile range fences.

**Domain outliers** are identified by subject-matter knowledge: a heart rate of
300 bpm is biologically impossible regardless of its statistical rarity; a blood
glucose of 500 mg/dL is extreme but clinically real (diabetic ketoacidosis).

The distinction matters because the correct action differs:
- **Measurement error** → remove or impute (the value is wrong)
- **Rare but real** → keep, but consider robust methods that down-weight extreme values
- **Different population** → investigate whether the observation belongs to your study population

There is no statistical test that can distinguish these cases. This is where
**domain expertise** — yours — is irreplaceable. {cite("Barnett & Lewis, 1994")}
""", unsafe_allow_html=True)

        section("How Outliers Affect Different Models")
        st.markdown("""
**Linear models** are the most sensitive. OLS minimizes the sum of *squared*
residuals, which means a single outlier with a large residual contributes
quadratically to the loss. This can shift the entire fitted line toward the
outlier — a phenomenon called *leverage* when the outlier is extreme in the
feature space, and *influence* when it substantially changes the fitted values.
""")
        st.markdown("""
Two complementary diagnostics formalize this:

- **Leverage** asks: *how unusual is this observation's combination of feature values?*
  An observation sitting far from the center of the data has high leverage — it
  has a long arm to pull the regression line toward itself.
- **Cook's distance** asks: *if I remove this one observation and refit the model,
  how much do the predictions change?* It combines leverage with residual size.
""")
        st.latex(r"""
        h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i
        """)
        st.markdown("""
**In plain language:** the leverage formula measures how far observation *i* sits
from the center of all the data, accounting for correlations between features.
Think of it as: "if I drew a target at the center of the data cloud, how far
from the bullseye is this observation?" The further out, the more influence it
has on the fitted line. Leverage values range from 1/n to 1; a common
rule of thumb flags observations where hᵢᵢ > 2p/n (where p is the number of
features) as high-leverage points.

High leverage alone does not mean an observation is problematic — it may lie far
from the center but still fall along the regression surface. The concern is when
high leverage *combines* with a large residual:
""")
        st.latex(r"""
        D_i = \frac{(\hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)})^\top (\hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)})}{p \cdot \text{MSE}}
        """)
        st.markdown("""
**In plain language:** Cook's distance answers "if I delete this one observation
and refit the model, how much do *all* the predictions move?" It compares two
sets of predictions — the ones from the full model and the ones from a model fit
without observation *i*. Large Cook's distance means removing that single observation
materially changes the model's behavior for everyone else. A large Dᵢ means removing observation *i* substantially shifts the
regression surface. Common thresholds: Dᵢ > 1 for clear concern, or
Dᵢ > 4/n as a more sensitive screening criterion.
""")

        with st.expander("🧮 Interactive: See how one outlier pulls a regression line"):
            out_x = st.slider(
                "Outlier x-position",
                min_value=-1.0, max_value=8.0, value=5.0, step=0.5,
                key="theory_outlier_x",
                help="Move the outlier further from the data center to increase its leverage.",
            )
            out_y = st.slider(
                "Outlier y-position",
                min_value=-10.0, max_value=20.0, value=15.0, step=1.0,
                key="theory_outlier_y",
                help="Move the outlier away from the trend to increase its residual.",
            )
            rng_out = np.random.default_rng(77)
            x_clean = rng_out.uniform(0, 3, 40)
            y_clean = 1.5 * x_clean + rng_out.normal(0, 0.8, 40)

            # Fit without outlier
            X_no = np.column_stack([np.ones(len(x_clean)), x_clean])
            beta_no = np.linalg.lstsq(X_no, y_clean, rcond=None)[0]

            # Fit with outlier
            x_with = np.append(x_clean, out_x)
            y_with = np.append(y_clean, out_y)
            X_wi = np.column_stack([np.ones(len(x_with)), x_with])
            beta_wi = np.linalg.lstsq(X_wi, y_with, rcond=None)[0]

            xline = np.linspace(-1, 8, 100)

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(x=x_clean, y=y_clean, mode="markers",
                marker=dict(size=6, color="rgba(99, 102, 241, 0.6)"),
                name="Clean data", showlegend=True))
            fig_out.add_trace(go.Scatter(x=[out_x], y=[out_y], mode="markers",
                marker=dict(size=12, color="#dc2626", symbol="x"),
                name="Outlier", showlegend=True))
            fig_out.add_trace(go.Scatter(x=xline, y=beta_no[0] + beta_no[1] * xline,
                mode="lines", line=dict(color="#16a34a", width=2, dash="dash"),
                name=f"Without outlier (slope={beta_no[1]:.2f})"))
            fig_out.add_trace(go.Scatter(x=xline, y=beta_wi[0] + beta_wi[1] * xline,
                mode="lines", line=dict(color="#dc2626", width=2),
                name=f"With outlier (slope={beta_wi[1]:.2f})"))
            fig_out.update_layout(
                title="One observation can reshape the entire regression line",
                xaxis_title="x", yaxis_title="y", height=340,
                margin=dict(t=50, b=40, l=50, r=20), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_out, use_container_width=True)
            st.markdown(
                "**Train your eye:** The green dashed line is the fit without the outlier — it follows the data. "
                "The red solid line includes the outlier. Move the outlier far right (high leverage) **and** far up (high residual): "
                "watch the red line tilt dramatically. Now move the outlier far right but **on the trend** (y ≈ 7): "
                "the red line barely moves, because high leverage alone is not dangerous — it takes leverage **plus** a large residual. "
                "**In your own results:** if the EDA page flags outliers, check whether they have both high leverage and large residuals before deciding to clip."
            )

        st.markdown(f"""
**Neural networks** are also affected, though through a different mechanism: extreme
values create large gradients that can destabilize training, especially early in
optimization when the network is most sensitive to initialization.

**Distance-based methods** (KNN) suffer because outliers distort the distance
metric — a few extreme values in one feature inflate distances in that dimension.

**Tree-based methods** are naturally robust to outliers. Since splits are based on
rank order, an observation at x = 1000 is treated identically to x = 100 if
no split falls between them. This does not mean trees are immune — an outlier
can still appear in a leaf and affect the leaf's mean prediction — but the
effect is contained. {cite("Hastie et al., ESL §9.2")}
""", unsafe_allow_html=True)

        app_connection(
            "The <strong>EDA</strong> page flags statistical outliers using IQR fences "
            "(Q1 − 1.5·IQR, Q3 + 1.5·IQR) and shows their count per feature. "
            "The <strong>Preprocess</strong> page offers percentile clipping and IQR-based "
            "clipping, both configurable per-model. The coaching layer warns about "
            "outliers only for affected model families (linear, neural, distance-based) "
            "and suggests robust alternatives like Huber regression when outliers are prevalent."
        )

        with st.expander("Deep Dive: Robust Regression"):
            st.markdown(f"""
When outliers are present but represent genuine data (not errors), robust regression
methods provide a middle ground between discarding observations and letting them
dominate the fit.

**Huber regression** uses a loss function that is quadratic for small residuals and
linear for large ones, controlled by a threshold parameter ε:
""")
            st.latex(r"""
            L_\varepsilon(r) = \begin{cases}
            \dfrac{r^2}{2} & \text{if } |r| \leq \varepsilon \\[8pt]
            \varepsilon \, |r| - \dfrac{\varepsilon^2}{2} & \text{if } |r| > \varepsilon
            \end{cases}
            """)
            st.markdown(f"""
The residual **r** is the difference between the observed and predicted value. When
|r| is small (the observation is well-predicted), the loss is the familiar squared
error r²/2 — the same as OLS. But when |r| exceeds the threshold ε, the loss
switches to *linear* growth (ε·|r|) instead of quadratic (r²). This means an
outlier with a residual of 10 contributes roughly 10× to the loss rather than 100×
— dramatically reducing its pull on the fit.

The threshold ε controls where the transition happens: smaller ε means more
observations are treated as "outliers" and down-weighted, making the fit more robust
but less statistically efficient on clean data. The standard default of ε = 1.345
is calibrated to retain 95% of the efficiency of OLS when the data actually is
normal — a carefully chosen compromise.
{cite("Huber, 1964")} {cite("Huber & Ronchetti, 2009")}
""", unsafe_allow_html=True)

        with st.expander("Deep Dive: Clipping vs Trimming — Two Different Responses to Outliers"):
            section("The Core Distinction")
            st.markdown(f"""
**Clipping** (Winsorization) and **trimming** look similar — both cap the influence of
extreme values — but they operate on different things, affect different quantities, and
belong at different stages of the pipeline. Conflating them leads to data leakage or
population-definition errors. {cite("Wilcox, 2012")}

| | **Clipping (Winsorization)** | **Trimming** |
|---|---|---|
| **Operation** | Caps values at a quantile threshold | Removes entire rows |
| **Effect on N** | Preserved | Reduced |
| **Changes** | Distribution shape (compresses tails) | Population definition |
| **Applied to** | Features (X) | Target (y) |
| **When** | After train/test split | Before train/test split |
""", unsafe_allow_html=True)

            section("Clipping in Detail")
            st.markdown("""
**Clipping** replaces every value outside a percentile band with the boundary value:
""")
            st.latex(r"x_{\text{clip}} = \min\!\bigl(\max(x,\; q_p),\; q_{1-p}\bigr)")
            st.markdown(f"""
where $q_p$ is the *p*-th sample quantile. No rows are removed — the sample size is
preserved — but the tails are *compressed*. A value of 1,200 clipped at the 99th
percentile (say, 800) becomes 800; the row stays in the dataset.

Because the clipping thresholds must be estimated from data, they must come from the
**training set only** and then applied identically to validation and test sets. Computing
thresholds from the whole dataset — or recomputing them on the test set — is a form
of data leakage that makes test-set evaluation optimistic. {cite("Kaufman et al., 2012")}

**When to clip:** measurement error or entry-error in a feature column where the
row itself is still valid; or when a feature's extreme range destabilizes a
linear/neural model but the observation is genuine.
""", unsafe_allow_html=True)

            section("Trimming in Detail")
            st.markdown(f"""
**Trimming** removes entire rows where the target falls outside a quantile range:
""")
            st.latex(r"\text{keep row } i \iff q_p \;\leq\; y_i \;\leq\; q_{1-p}")
            st.markdown(f"""
Sample size shrinks. More importantly, trimming *redefines the population*: the
model you train and evaluate no longer represents all observations, only those
within the chosen range. This is correct when the extreme rows represent a genuinely
different process — equipment failure readings, out-of-range physiological values —
that your model is not intended to predict.

Because trimming changes which rows enter the split, it must happen **before** the
train/test split. Trimming after splitting can give train and test sets different
effective populations, making test-set metrics misleading. {cite("Hastie et al., ESL §2.9")}

**When to trim:** target outliers from a different generating process; or when you
have a clearly bounded prediction task and want to make that boundary explicit.
""", unsafe_allow_html=True)

            section("When to Use Which")
            st.markdown(f"""
The right tool depends on *why* the extreme values exist:

- **Measurement error on a feature** — the value in that column is wrong, but the
  row may otherwise be valid. → **Clip** the feature, keep the row.
- **Target outlier from a different population** — the observation itself may not
  belong to your prediction task. → **Trim** the row before splitting.
- **Real but extreme feature value** (e.g., a billionaire in an income study) —
  the observation is legitimate but distorts linear/neural models.
  → **Clip** the feature, or switch to a robust model (Huber, tree-based).
- **Moderate target skew, same population** — consider a target *transformation*
  (log, Box-Cox) before reaching for trimming; transformations preserve all rows
  and improve residual normality. {cite("Box & Cox, 1964")}
""", unsafe_allow_html=True)

            app_connection(
                "The <strong>Preprocess</strong> page applies percentile or IQR-based "
                "<strong>clipping to features (X)</strong> after the split — quantiles are "
                "estimated on the training set only, preventing leakage. "
                "The <strong>Train &amp; Compare</strong> page offers optional "
                "<strong>target trimming before the split</strong> for regression tasks, "
                "removing rows where the target falls outside a chosen quantile range."
            )

            section("Interactive Demo: Effect on a Skewed Distribution")
            trim_q = st.slider(
                "Quantile threshold (symmetric, e.g. 0.05 trims bottom 5% and top 5%)",
                min_value=0.01, max_value=0.25, value=0.05, step=0.01,
                key="theory_clip_trim_q",
                help="Both clipping and trimming use this as the lower quantile; upper = 1 − lower.",
            )
            rng_ct = np.random.default_rng(42)
            # Skewed distribution: lognormal + a few extreme values
            base = rng_ct.lognormal(mean=1.5, sigma=0.8, size=400)
            extra = rng_ct.uniform(20, 40, size=20)
            raw = np.concatenate([base, extra])

            q_lo = np.quantile(raw, trim_q)
            q_hi = np.quantile(raw, 1 - trim_q)

            # Clipped: cap at thresholds (same N)
            clipped = np.clip(raw, q_lo, q_hi)

            # Trimmed: remove rows outside thresholds (fewer N)
            trimmed = raw[(raw >= q_lo) & (raw <= q_hi)]

            n_raw = len(raw)
            n_clipped = len(clipped)
            n_trimmed = len(trimmed)
            n_removed = n_raw - n_trimmed

            fig_ct = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Clipping — N stays {n_raw}",
                    f"Trimming — N drops {n_raw} → {n_trimmed} (−{n_removed} rows)",
                ],
            )
            bins = np.linspace(raw.min(), raw.max(), 50)

            # Left: original vs clipped
            raw_hist, edges = np.histogram(raw, bins=bins)
            clipped_hist, _ = np.histogram(clipped, bins=bins)
            bin_centers = (edges[:-1] + edges[1:]) / 2

            fig_ct.add_trace(go.Bar(
                x=bin_centers, y=raw_hist,
                name="Original", marker_color="rgba(99,102,241,0.4)",
                showlegend=True,
            ), row=1, col=1)
            fig_ct.add_trace(go.Bar(
                x=bin_centers, y=clipped_hist,
                name="Clipped", marker_color="rgba(234,88,12,0.7)",
                showlegend=True,
            ), row=1, col=1)

            # Vertical lines at thresholds
            for col_idx in (1, 2):
                fig_ct.add_vline(x=q_lo, line_dash="dash", line_color="#16a34a",
                                 annotation_text=f"q={trim_q:.2f}", row=1, col=col_idx)
                fig_ct.add_vline(x=q_hi, line_dash="dash", line_color="#16a34a",
                                 annotation_text=f"q={1-trim_q:.2f}", row=1, col=col_idx)

            # Right: original vs trimmed
            trimmed_hist, _ = np.histogram(trimmed, bins=bins)
            fig_ct.add_trace(go.Bar(
                x=bin_centers, y=raw_hist,
                name="Original", marker_color="rgba(99,102,241,0.4)",
                showlegend=False,
            ), row=1, col=2)
            fig_ct.add_trace(go.Bar(
                x=bin_centers, y=trimmed_hist,
                name="Trimmed", marker_color="rgba(220,38,38,0.7)",
                showlegend=True,
            ), row=1, col=2)

            fig_ct.update_layout(
                height=360,
                barmode="overlay",
                template="plotly_white",
                margin=dict(t=60, b=40, l=50, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            )
            fig_ct.update_xaxes(title_text="Value")
            fig_ct.update_yaxes(title_text="Count")
            st.plotly_chart(fig_ct, use_container_width=True)
            st.markdown(
                f"**What to notice:** With clipping (left), the bars at the tails don't disappear — "
                f"their mass *piles up* at the threshold values (orange bars spike at the green lines). "
                f"The distribution shape changes but N stays {n_raw}. "
                f"With trimming (right), the bars outside the green lines simply vanish and N drops to {n_trimmed}. "
                f"The interior of the distribution is unchanged, but you have {n_removed} fewer observations. "
                f"**Move the slider** to a very small value (0.01) to see minimal effect, or toward 0.25 to see aggressive truncation."
            )

        references([
            "Barnett, V. & Lewis, T. (1994). *Outliers in Statistical Data* (3rd ed.). Wiley.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §9.2. Springer.",
            "Huber, P.J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.",
            "Huber, P.J. & Ronchetti, E.M. (2009). *Robust Statistics* (2nd ed.). Wiley.",
        ])

    # ── Collinearity ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(f"""
Collinearity (or multicollinearity) occurs when two or more predictor variables are
highly correlated, meaning one can be approximately predicted from the others. It is
one of the most common issues in applied regression and one of the most
misunderstood. {cite("ISLR, §3.3.3")}
""", unsafe_allow_html=True)

        section("Why Collinearity Is a Problem — and for Whom")
        st.markdown("""
Consider a simple case: predicting house price from both square footage and number
of rooms, which are highly correlated. The model cannot determine how much of the
effect belongs to each variable — a dollar of price increase attributed to an extra
100 sq ft could equally be attributed to an extra room.

The mathematical consequence is that the matrix **XᵀX** becomes nearly singular.
The OLS estimator **β̂ = (XᵀX)⁻¹Xᵀy** involves inverting this matrix, and
inverting a nearly singular matrix amplifies small perturbations in the data into
large changes in the estimated coefficients.
""")
        st.latex(r"""
        \text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{\sum_{i=1}^{n}(x_{ij} - \bar{x}_j)^2 \;\cdot\; (1 - R_j^2)}
        """)
        st.markdown("""
Let's unpack this. The variance of the *j*-th coefficient estimate depends on three things:

- **σ²** (numerator): the noise variance in the response. More noise → less precise estimates. You cannot control this.
- **Σᵢ(xᵢⱼ − x̄ⱼ)²** (denominator): the total variation in feature *j*. More spread in the feature → more information → tighter estimates.
- **(1 − Rⱼ²)** (denominator): this is the critical term. Rⱼ² is the R² you get from regressing feature *j* on *all the other features*. If feature *j* can be perfectly predicted from other features, Rⱼ² = 1, the denominator goes to zero, and the variance explodes to infinity. Even Rⱼ² = 0.9 inflates the variance by a factor of 10.

This inflated variance means:""")
        st.markdown("""
- **Coefficients become unstable:** Small changes in the data produce wildly different estimates.
- **Confidence intervals widen:** You lose the ability to determine which variables matter.
- **Sign flips:** A coefficient may be positive in one sample and negative in another.

**Critically, collinearity does not affect prediction accuracy.** If you only care
about ŷ, a collinear model predicts just as well as a non-collinear one (assuming
future data has the same collinearity structure). The problem is entirely about
*interpretation* and *inference* on individual coefficients.
""")

        with st.expander("🧮 Interactive: Watch coefficients destabilize as correlation increases"):
            corr_val = st.slider(
                "Correlation between x₁ and x₂",
                min_value=0.0, max_value=0.99, value=0.0, step=0.05,
                key="theory_collinearity_corr",
            )
            rng_c = np.random.default_rng(12)
            n_c = 80
            # Generate correlated predictors
            mean_c = [0, 0]
            cov_c = [[1, corr_val], [corr_val, 1]]
            X_c = rng_c.multivariate_normal(mean_c, cov_c, n_c)
            # True coefficients: both contribute equally
            y_c = 2.0 * X_c[:, 0] + 2.0 * X_c[:, 1] + rng_c.normal(0, 1, n_c)

            # Fit OLS multiple times with slight data perturbations to show instability
            coefs_x1, coefs_x2 = [], []
            for seed in range(30):
                rng_boot = np.random.default_rng(seed + 100)
                idx = rng_boot.choice(n_c, size=n_c, replace=True)
                Xb = X_c[idx]
                yb = y_c[idx]
                # OLS: beta = (X'X)^-1 X'y
                Xb_aug = np.column_stack([np.ones(n_c), Xb])
                try:
                    beta = np.linalg.lstsq(Xb_aug, yb, rcond=None)[0]
                    coefs_x1.append(beta[1])
                    coefs_x2.append(beta[2])
                except Exception:
                    pass

            fig_coll = go.Figure()
            fig_coll.add_trace(go.Box(y=coefs_x1, name="β₁ (x₁)", marker_color="rgba(99, 102, 241, 0.7)",
                                       boxpoints="all", jitter=0.3, pointpos=-1.5))
            fig_coll.add_trace(go.Box(y=coefs_x2, name="β₂ (x₂)", marker_color="rgba(234, 88, 12, 0.7)",
                                       boxpoints="all", jitter=0.3, pointpos=-1.5))
            fig_coll.add_hline(y=2.0, line_dash="dash", line_color="#16a34a",
                               annotation_text="True value (2.0)")
            fig_coll.update_layout(
                title=f"Coefficient estimates across 30 bootstrap samples (r = {corr_val:.2f}, VIF = {1/(1-corr_val**2):.1f})",
                yaxis_title="Estimated coefficient", height=340,
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_coll, use_container_width=True)
            st.markdown(
                "**Train your eye:** Start at r = 0 and look at the box plots — tight, centered on the green dashed truth line. "
                "Now slide slowly toward r = 0.9. Watch the boxes **stretch vertically** and individual dots scatter far from 2.0. "
                "Look for the telltale pattern: when β₁ is unusually high, β₂ tends to be unusually low (and vice versa). "
                "The model is robbing Peter to pay Paul because it can't tell the two variables apart. "
                "The VIF in the title tracks the variance inflation factor — at r = 0.95 it exceeds 10, the conventional red line."
            )

        misconception(
            "Collinearity is not mainly a prediction problem. A model can predict well and still have coefficient estimates that are too unstable to interpret scientifically."
        )

        self_check(
            "If removing one of two highly correlated predictors barely changes predictions but radically changes the coefficient table, what does that tell you about the real problem collinearity causes?"
        )

        section("Detecting Collinearity: VIF")
        st.markdown("""
The Variance Inflation Factor (VIF) answers a simple question: *can the other
features in the model predict this feature?* For each feature *j*, VIF runs a
regression of feature *j* against all other features and measures how well they
explain it:
""")
        st.latex(r"""
        \text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R_j^2}
        """)
        st.markdown("""
**In plain language:** Rⱼ² is just the R² from predicting feature *j* using all
the other features. If the other features can almost perfectly predict feature *j*
(Rⱼ² close to 1), then feature *j* is redundant — and any coefficient the model
assigns to it is unreliable because it can't tell *j*'s contribution apart from
the others.

**Reading the number:** VIF = 1 means feature *j* carries unique information. VIF = 5
means 80% of its variation is shared with other features — the coefficient's
uncertainty is 5× larger than necessary. VIF = 10 (Rⱼ² = 0.90) is the conventional
red line. VIF = 100 means the coefficient estimate is essentially noise.

A pairwise correlation matrix catches the simplest cases (two features with r > 0.9),
but it misses **multicollinearity** — where a feature is predictable from a *combination*
of other features without being highly correlated with any single one. VIF catches
both cases because it regresses each feature on all others simultaneously.
""")

        section("Which Models Are Affected?")
        st.markdown(f"""
**Linear models** are the primary victim. Collinearity inflates coefficient variance
and makes interpretation unreliable. Regularization (Ridge, LASSO) is the standard
remedy: Ridge shrinks correlated coefficients toward each other; LASSO selects one
and zeros the rest. {cite("ISLR, §6.2")}

**Tree-based models** are essentially immune. At each split, the algorithm picks whichever
feature produces the best partition — if two features are collinear, it picks one,
and the other simply isn't used at that split. The predictions are unaffected. Feature
importance scores may be split across collinear features, but the model itself is
not destabilized.

**Neural networks** can absorb collinearity through the hidden layers, but training
may be slower or less stable because the loss surface has flat directions corresponding
to the collinear subspace.
""", unsafe_allow_html=True)

        with st.expander("🧮 Interactive: Ridge shrinks, LASSO selects — see the difference"):
            reg_corr = st.slider(
                "Correlation between x₁ and x₂",
                min_value=0.0, max_value=0.99, value=0.90, step=0.05,
                key="theory_reg_corr",
            )
            reg_alpha = st.slider(
                "Regularization strength (λ)",
                min_value=0.01, max_value=10.0, value=1.0, step=0.2,
                key="theory_reg_alpha",
            )
            rng_r = np.random.default_rng(42)
            n_r = 100
            cov_r = [[1, reg_corr], [reg_corr, 1]]
            X_r = rng_r.multivariate_normal([0, 0], cov_r, n_r)
            y_r = 2.0 * X_r[:, 0] + 2.0 * X_r[:, 1] + rng_r.normal(0, 1, n_r)

            # Ridge: closed-form β = (X'X + λI)^-1 X'y
            XtX = X_r.T @ X_r
            Xty = X_r.T @ y_r
            beta_ridge = np.linalg.solve(XtX + reg_alpha * np.eye(2), Xty)

            # LASSO: coordinate descent (simple implementation for 2 features)
            beta_lasso = np.array([0.0, 0.0])
            for _ in range(200):
                for j in range(2):
                    r_j = y_r - X_r @ beta_lasso + X_r[:, j] * beta_lasso[j]
                    rho = X_r[:, j] @ r_j
                    z = np.sum(X_r[:, j] ** 2)
                    beta_lasso[j] = np.sign(rho) * max(abs(rho) - reg_alpha * n_r, 0) / z

            fig_reg = go.Figure()
            methods = ["Ridge", "LASSO"]
            for idx, (betas, color, name) in enumerate([
                (beta_ridge, "rgba(99, 102, 241, 0.8)", "Ridge"),
                (beta_lasso, "rgba(234, 88, 12, 0.8)", "LASSO"),
            ]):
                fig_reg.add_trace(go.Bar(
                    x=["β₁", "β₂"], y=betas, name=name,
                    marker_color=color,
                    text=[f"{b:.3f}" for b in betas], textposition="outside",
                ))
            fig_reg.add_hline(y=2.0, line_dash="dash", line_color="#16a34a",
                              annotation_text="True value (2.0)")
            fig_reg.update_layout(
                barmode="group",
                title=f"Estimated coefficients (r = {reg_corr:.2f}, λ = {reg_alpha:.2f})",
                yaxis_title="Coefficient estimate", height=300,
                margin=dict(t=50, b=30, l=50, r=30), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.markdown(
                "**Train your eye:** Start at high correlation (r = 0.90). Ridge (purple) pulls both coefficients toward each other — "
                "neither gets the full credit, but neither is zeroed out. LASSO (orange) makes a harsher choice: it tends to keep one and drop the other entirely. "
                "Now increase λ: both methods shrink harder, but Ridge shrinks smoothly while LASSO snaps coefficients to zero. "
                "**The lesson:** neither method discovers which variable is 'truly important' — that's your scientific judgment, not a statistical output."
            )

        misconception(
            "Regularization does not 'discover the truth' about which correlated variable is biologically fundamental. It stabilizes estimation or performs pragmatic selection; domain interpretation still requires caution."
        )

        self_check(
            "If two features are nearly interchangeable, why might Ridge keep both while LASSO drops one? What educational lesson does that teach about shrinkage versus selection?"
        )

        app_connection(
            "The <strong>EDA</strong> page computes pairwise correlations and flags highly "
            "correlated pairs (|r| > 0.8). The coaching layer only raises collinearity "
            "as an issue for linear model families, since tree-based and other models "
            "are unaffected. The <strong>Feature Selection</strong> page offers VIF-based "
            "filtering as one of its selection methods."
        )

        with st.expander("Deep Dive: Condition Number"):
            st.markdown(f"""
While VIF examines one feature at a time, the **condition number** of the design
matrix gives a single global measure of how collinear the entire system is.
""")
            st.latex(r"""
            \kappa(\mathbf{X}) = \frac{\sigma_{\max}(\mathbf{X})}{\sigma_{\min}(\mathbf{X})}
            """)
            st.markdown(f"""
The singular values σ of the design matrix X describe how "stretched" the data is
in each direction of the feature space. The largest singular value σ_max captures
the direction of maximum spread; the smallest σ_min captures the direction of
minimum spread. Their ratio — the condition number κ — tells you how elongated the
data cloud is: a sphere has κ = 1; a pancake has κ >> 1.

Why this matters: when κ is large, the matrix inversion in β̂ = (XᵀX)⁻¹Xᵀy
amplifies numerical errors. A condition number of 1000 means that a 0.1% change in
the data can produce up to a 100% change in the coefficients. Common thresholds:
κ < 30 is well-conditioned, κ > 30 indicates moderate collinearity, and κ > 1000
means coefficients are numerically unreliable. {cite("Belsley et al., 1980")}
""", unsafe_allow_html=True)

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §3.3.3, §6.2. Springer.",
            "Belsley, D.A., Kuh, E., & Welsch, R.E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. Wiley.",
        ])

    # ── Sample Size ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
Sample size determines the ceiling on what your analysis can reliably detect.
No amount of methodological sophistication can compensate for fundamentally
insufficient data — and different methods have very different data appetites.
""")

        section("Events Per Variable (EPV)")
        st.markdown(f"""
For regression models, the most commonly cited rule of thumb is the **events per
variable** (EPV) guideline. In classification, "events" refers to the count of
the minority class; in regression, it refers to the total sample size.

The classical guideline: **at least 10 events per predictor variable** for logistic
regression to produce stable coefficient estimates. {cite("Peduzzi et al., 1996")}
More recent simulation studies suggest this may be conservative for some settings
and insufficient for others, but it remains a useful starting point.
""", unsafe_allow_html=True)
        st.latex(r"""
        \text{EPV} = \frac{n_{\text{events}}}{p}
        """)
        st.markdown("""
The numerator **n_events** is the effective sample size: for classification, this is
the count of the *minority class* (since the model's ability to learn the rare class
is the binding constraint); for regression, it's the total sample size. The denominator
**p** is the number of predictor variables, including any dummy variables created from
categoricals. So a dataset with 200 positive cases and 15 features has EPV = 200/15 ≈ 13.

The thresholds are approximate but well-tested through simulation: EPV < 5 carries
serious risk of overfitting and numerically unstable coefficient estimates; EPV between
5 and 10 is marginal territory where regularization becomes essential; EPV > 20 is
generally comfortable for most linear methods.
""")

        section("The Curse of Dimensionality")
        st.markdown(f"""
As the number of features *p* grows relative to the sample size *n*, the data
becomes increasingly sparse in the feature space. This is the **curse of
dimensionality**, and it affects distance-based methods most
severely. {cite("ISLR, §2.2.3")}

Consider KNN with k = 1: in one dimension, the nearest neighbor to a test point
is typically close. In 100 dimensions, even with a million training points, the
"nearest" neighbor may be very far away — because the volume of the feature space
grows exponentially while the data points remain fixed.
""", unsafe_allow_html=True)
        st.markdown("""
A concrete way to see this: suppose your data is uniformly distributed in a
unit hypercube, and you want to find the "local neighborhood" that captures
10% of the data. How wide does that neighborhood need to be along each dimension?
""")
        st.latex(r"""
        \ell = r^{1/p}
        """)
        st.markdown("""
Here *ℓ* is the edge length of the hypercube neighborhood, *r* is the fraction
of data you want to capture, and *p* is the number of dimensions.

In 1 dimension, to capture 10% of the data you need an interval of width 0.10 —
genuinely local. In 10 dimensions, you need each edge to span 0.10^(1/10) ≈ **0.79**
of the full range. In 100 dimensions, each edge must span 0.10^(1/100) ≈ **0.977**
of the full range. Your "local" neighborhood now covers almost the entire dataset
in every direction — "local" has lost its meaning.

The practical implications:""")
        st.markdown("""
- **KNN** degrades rapidly as p grows, because all points become approximately equidistant.
- **SVM** with RBF kernels faces the same problem in the implicit feature space.
- **Linear models** can handle high p if regularized (LASSO, Ridge), but need strong regularization.
- **Tree-based models** handle high p relatively well because each split only considers one feature at a time, but random forests may waste splits on noise features.
- **Neural networks** are highly susceptible in the small-n regime: they have the capacity to memorize the training set, and without sufficient data, they will.
""")

        with st.expander("🧮 Interactive: Watch 'local' neighborhoods become global"):
            dim_p = st.slider(
                "Number of dimensions (p)",
                min_value=1, max_value=100, value=2, step=1,
                key="theory_curse_dim",
                help="The number of features in the space. Watch how quickly locality breaks down.",
            )
            fractions = [0.01, 0.05, 0.10, 0.25]
            edge_lengths = [f ** (1.0 / dim_p) for f in fractions]

            fig_curse = go.Figure()
            colors_curse = ["#dc2626", "#f59e0b", "#16a34a", "#2563eb"]
            for frac, edge, color in zip(fractions, edge_lengths, colors_curse):
                fig_curse.add_trace(go.Scatter(
                    x=[f"{frac*100:.0f}%"], y=[edge * 100],
                    mode="markers+text",
                    marker=dict(size=16, color=color),
                    text=[f"{edge*100:.1f}%"], textposition="top center",
                    name=f"Capture {frac*100:.0f}% of data",
                    showlegend=True,
                ))

            # Also show the curve across dimensions for 10% capture
            dims_range = np.arange(1, 101)
            edge_10pct = 0.10 ** (1.0 / dims_range) * 100

            fig_curse2 = go.Figure()
            fig_curse2.add_trace(go.Scatter(
                x=dims_range, y=edge_10pct, mode="lines",
                line=dict(color="#dc2626", width=2.5),
                name="Edge length to capture 10% of data",
            ))
            fig_curse2.add_hline(y=90, line_dash="dot", line_color="#94a3b8",
                                annotation_text="90% of range — 'local' is meaningless")
            fig_curse2.add_vline(x=dim_p, line_dash="dash", line_color="#2563eb",
                                annotation_text=f"p = {dim_p}")
            fig_curse2.update_layout(
                title=f"At p = {dim_p}: you need {0.10 ** (1.0/dim_p) * 100:.1f}% of each axis to capture 10% of the data",
                xaxis_title="Number of dimensions (p)",
                yaxis_title="Neighborhood edge length (% of range)",
                yaxis_range=[0, 105], height=320,
                margin=dict(t=50, b=40, l=60, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_curse2, use_container_width=True)
            st.markdown(
                "**Train your eye:** At p = 1 or 2, the neighborhood edge is small — you're genuinely looking at local data. "
                "Slide p to 20: the edge length jumps above 89%, meaning your 'local' neighborhood covers almost the entire dataset along every axis. "
                "By p = 50 it's over 95%. The blue dashed line tracks your current dimension. "
                "**The lesson:** in high dimensions, KNN's 'nearest neighbors' are not meaningfully close. "
                "This is why the app recommends feature selection or PCA before distance-based methods when p is large."
            )

        section("Class Imbalance")
        st.markdown("""
Class imbalance occurs when one class is much more prevalent than the other. A
dataset with 95% negative cases and 5% positive cases is severely imbalanced —
and this creates problems beyond just needing more data.

**Why imbalance is a problem:** Most algorithms optimize overall accuracy, which
means they can achieve excellent performance by simply predicting the majority
class for every observation. A model that predicts "healthy" for everyone in a
95/5 dataset achieves 95% accuracy while being completely useless for its
intended purpose (detecting disease).

**How imbalance affects different models:**
- **Linear models** (logistic regression) are relatively robust to imbalance because
  they optimize the likelihood, not accuracy. But the default threshold of 0.5
  may need adjustment.
- **Tree-based models** can be biased toward the majority class when using Gini
  impurity, since splits that keep the majority class pure are "rewarded."
- **KNN** is heavily affected: the neighborhood of a minority-class point is
  often dominated by majority-class neighbors.
- **Neural networks** will converge to predicting the majority class unless the
  loss function is weighted or the data is resampled.

**Metrics under imbalance:** As discussed in Chapter 5, accuracy is misleading.
AUROC, AUPRC, precision, recall, and F1 are more informative. The app reports
all of these, and the coaching layer flags datasets with class imbalance ratios
below 0.35 as requiring careful metric selection.
""")

        with st.expander("🧮 Interactive: Why accuracy lies under class imbalance"):
            imb_pct = st.slider(
                "Positive class prevalence (%)",
                min_value=1, max_value=50, value=5, step=1,
                key="theory_imbalance_pct",
            )
            n_total = 200
            n_pos = max(1, int(n_total * imb_pct / 100))
            n_neg = n_total - n_pos

            # "Always predict majority" baseline
            acc_majority = n_neg / n_total * 100
            recall_majority = 0.0
            f1_majority = 0.0

            # A mediocre model that catches ~60% of positives
            tp = int(n_pos * 0.6)
            fp = int(n_neg * 0.08)
            fn = n_pos - tp
            tn = n_neg - fp
            acc_model = (tp + tn) / n_total * 100
            prec_model = tp / max(tp + fp, 1) * 100
            recall_model = tp / max(tp + fn, 1) * 100
            f1_model = 2 * (prec_model * recall_model) / max(prec_model + recall_model, 0.01)

            fig_imb = go.Figure()
            metrics = ["Accuracy", "Recall", "Precision", "F1"]
            fig_imb.add_trace(go.Bar(
                name="Always predict negative",
                x=metrics, y=[acc_majority, recall_majority, 0, f1_majority],
                marker_color="rgba(220, 38, 38, 0.7)",
            ))
            fig_imb.add_trace(go.Bar(
                name="Mediocre model (60% recall)",
                x=metrics, y=[acc_model, recall_model, prec_model, f1_model],
                marker_color="rgba(22, 163, 74, 0.7)",
            ))
            fig_imb.update_layout(
                barmode="group", height=300,
                yaxis_title="Metric value (%)", yaxis_range=[0, 105],
                title=f"n = {n_total}, positive class = {imb_pct}% ({n_pos} cases)",
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_imb, use_container_width=True)
            st.markdown(
                "**Train your eye:** Start at 5% prevalence and compare the two bars on Accuracy — they're almost the same height. "
                "That's the trap: a model that does *nothing* useful scores nearly as well as one that actually finds cases. "
                "Now look at Recall: the red bar is zero (it catches no one), while the green bar shows real detection. "
                "Slide toward 50% and watch accuracy become a fair metric again as the classes balance. "
                "**The lesson:** in your own results, if accuracy is high but prevalence is low, don't trust accuracy alone."
            )

        misconception(
            "Class imbalance does not mean the dataset is unusable. It means your training objective, threshold choice, and evaluation metrics need to reflect the asymmetry of the problem."
        )

        self_check(
            "If your model has 96% accuracy on a 95/5 dataset, what question should you ask next before trusting it?"
        )

        section("Data Quality Basics: Duplicates, Constants, and Cardinality")
        st.markdown("""
Before any statistical analysis, several basic data quality checks are essential:

**Duplicate rows** are exact copies of other observations. They inflate sample
size artificially, reduce the effective diversity of the dataset, and can bias
any model that sees the same observation twice. Common causes: data entry errors,
merging datasets without deduplication, or row-level replication in relational joins.

**Constant features** (zero variance) have the same value for every observation.
They carry no information and should be removed — they cannot predict anything,
and some algorithms (e.g., standardization) will produce NaN or division-by-zero
errors on them.

**Near-constant features** have almost zero variance (e.g., 99.5% of values are
the same). They carry minimal information and can destabilize some algorithms.

**Cardinality analysis** examines how many unique values each feature has. This
helps identify:
- **ID columns** (cardinality = n) that should never be used as features.
- **Binary features** (cardinality = 2) that may need special encoding.
- **High-cardinality categoricals** (many unique values) that will explode the
  feature space under one-hot encoding.
""")

        app_connection(
            "The <strong>Upload & Audit</strong> page automatically checks for duplicate "
            "rows, constant features, and cardinality. The <strong>EDA</strong> page "
            "detects class imbalance and adjusts the coaching recommendations based on "
            "the imbalance ratio. Duplicate counts and cardinality tables are shown "
            "in the audit summary."
        )

        section("When You Don't Have Enough Data")
        st.markdown("""
If your sample size is genuinely small (n < 100, or EPV < 10), the honest options are:

1. **Use simpler models.** Ridge or LASSO regression with aggressive regularization.
   Random Forest with few, shallow trees. Avoid neural networks entirely.
2. **Reduce dimensionality first.** Feature selection or PCA before modeling —
   but be careful to do this properly (within cross-validation, not before splitting).
3. **Report uncertainty honestly.** Wide confidence intervals are not a failure —
   they are the truth about what your data can tell you.
4. **Consider whether prediction is the right goal.** With very small samples,
   descriptive analysis or hypothesis testing may be more appropriate than
   building a predictive model.
""")

        app_connection(
            "The <strong>Upload & Audit</strong> page reports sample size and the number of features. "
            "The <strong>EDA</strong> page uses a regime detection system that adapts the recommended "
            "workflow to your dataset's shape — small datasets trigger recommendations for simpler "
            "models and stronger regularization. The coaching layer warns when EPV is low for the "
            "selected model complexity."
        )

        with st.expander("Deep Dive: Power Analysis"):
            st.markdown(f"""
Statistical power is the probability of detecting a true effect when it exists.
The standard target is 80% power (β = 0.20), meaning a 1-in-5 chance of missing
a real effect.

Power depends on four quantities — fix any three and the fourth is determined.
For a two-sample t-test at 80% power and α = 0.05:
""")
            st.latex(r"""
            n \approx \frac{16 \, \sigma^2}{\delta^2}
            """)
            st.markdown(f"""
Here **σ** is the common standard deviation of both groups and **δ** is the true
difference in means you want to detect. The ratio δ/σ is the *effect size* — the
signal relative to the noise. Notice that n scales with the *square* of the noise-to-signal
ratio: to detect an effect half as large, you need four times as many observations.
This is why underpowered studies are so common — small effects require surprisingly
large samples.

For logistic regression, a widely used approximation is:
""")
            st.latex(r"""
            n \approx \frac{10 \, p}{\pi_{\min}}
            """)
            st.markdown(f"""
where **p** is the number of predictors and **π_min** is the proportion of the
minority class. This is essentially the EPV ≥ 10 rule rewritten as a sample
size formula.

The key insight: **power analysis should be done before data collection, not after.**
A post-hoc power analysis on a non-significant result is circular — it will always
show low power, because the observed effect size is small by definition.
{cite("Hoenig & Heisey, 2001")}

If you're analyzing an existing dataset and suspect low power, the appropriate
response is to report confidence intervals (which convey the uncertainty directly)
rather than relying on a binary significant/non-significant decision.
""", unsafe_allow_html=True)

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §2.2.3. Springer.",
            "Peduzzi, P., Concato, J., Kemper, E., Holford, T.R., & Feinstein, A.R. (1996). A simulation study of the number of events per variable in logistic regression analysis. *Journal of Clinical Epidemiology*, 49(12), 1373–1379.",
            "Hoenig, J.M. & Heisey, D.M. (2001). The abuse of power: The pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19–24.",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 2: Feature Engineering & Selection
# ══════════════════════════════════════════════════════════════════════════════

def render_feature_engineering():
    intro(
        "Feature engineering transforms raw variables into representations that "
        "better expose the underlying signal to your models. Feature selection then "
        "prunes the input space to reduce noise, improve interpretability, and avoid "
        "overfitting. This chapter covers the theory behind the "
        "<strong>Feature Engineering</strong> and <strong>Feature Selection</strong> pages."
    )

    tabs = st.tabs([
        "Transformations",
        "Encoding Categoricals",
        "Selection Methods",
        "Information Leakage",
    ])

    # ── Transformations ──────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
Feature transformations reshape individual variables to better match the
assumptions of downstream models. The goal is not to change the data's meaning
but to present the same information in a form that specific algorithms can
exploit more effectively.
""")
        section("Polynomial and Interaction Features")
        st.markdown(f"""
Linear models assume that the relationship between each feature and the response
is a straight line. When the true relationship is curved — for example, the effect
of age on medical costs accelerates after 50 — a linear model will systematically
under-predict at the extremes and over-predict in the middle.

Polynomial features address this by explicitly creating new columns that represent
powers and products of the original features. For a feature *x*, a degree-2 expansion
creates *x²*; for two features *x₁* and *x₂*, it also creates the interaction
*x₁ · x₂*. The model can then learn curved and interactive effects while remaining
linear *in the expanded feature space*. {cite("ISLR, §7.1")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \text{Degree-2 expansion of } (x_1, x_2): \quad x_1, \; x_2, \; x_1^2, \; x_2^2, \; x_1 x_2
        """)
        st.markdown("""
The tradeoff is dimensionality: a degree-*d* expansion of *p* features creates
on the order of *p^d* new features. With 20 original features, a degree-3 expansion
produces over 1,500 columns — many of which are noise. This is why polynomial
features should be used judiciously and paired with feature selection or regularization.

**Tree-based models do not benefit from polynomial features.** Trees can already
capture nonlinear effects and interactions through recursive partitioning. Adding
polynomial terms to a Random Forest just adds noise without new information.
""")
        section("Mathematical Transforms")
        st.markdown("""
When the EDA page reveals that a feature is heavily skewed, a mathematical transform
can compress the long tail and make the distribution more symmetric. The most common
options and when each is appropriate:

**Log transform** — log(x) or log(1 + x) for data with zeros. Best for right-skewed
positive data where the spread increases with the level (e.g., income, medical costs,
gene expression counts). A unit increase in log(x) corresponds to a *multiplicative*
change in x, which is often more natural for this type of data.

**Square root** — √x. A gentler compression than log, appropriate for count data
(Poisson-distributed variables) where the variance is proportional to the mean.

**Reciprocal** — 1/x. An aggressive compression that reverses the ordering. Rarely
useful as a standalone transform but arises naturally in some domains (e.g., speed = 1/time).

As discussed in the Data Quality chapter, tree-based models are invariant to monotone
transforms, so these only benefit linear, neural, and distance-based models.
""")
        section("Ratio Features")
        st.markdown("""
Ratio features divide one variable by another to create a normalized quantity that
often has more predictive signal than either variable alone. Classic examples include
BMI (weight/height²), price-per-square-foot, and debt-to-income ratio.

The intuition: ratios control for a confounding variable. Income alone and spending
alone may both correlate with default risk, but spending-to-income ratio isolates the
*overspending behavior* that actually predicts default.

**Caution:** Ratios can create extreme values when the denominator is near zero. The
app clips or warns when this occurs.
""")
        section("Binning")
        st.markdown("""
Binning (discretization) converts a continuous feature into categorical bins. This
is useful when the relationship between the feature and response is step-like rather
than smooth — for example, clinical thresholds (normal/pre-diabetic/diabetic based
on HbA1c) that represent genuine regime changes.

Three strategies are available:

- **Equal-width binning** divides the range into bins of equal size. Simple but sensitive
  to outliers (one extreme value can make most bins empty).
- **Quantile binning** creates bins with approximately equal numbers of observations.
  More robust, but boundaries may split natural clusters.
- **K-means binning** uses 1-D clustering to find natural break points in the data.
  Often the best choice when the distribution has multiple modes.

**When NOT to bin:** Binning always destroys information — within each bin, all
variation is lost. If the true relationship is smooth and monotone, binning will hurt
model performance compared to using the raw (or transformed) continuous feature.
""")
        section("Dimensionality Reduction")
        st.markdown(f"""
When the feature space is very high-dimensional — either from raw data or after
polynomial expansion — dimensionality reduction creates a smaller set of new
features that capture most of the variance in the original space.

**PCA (Principal Component Analysis)** finds orthogonal directions of maximum variance
and projects the data onto the top *k* of them. {cite("ISLR, §12.2")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \mathbf{Z} = \mathbf{X} \mathbf{W}_k \quad \text{where } \mathbf{W}_k = [w_1, \ldots, w_k]
        \text{ are the top } k \text{ eigenvectors of } \mathbf{X}^\top \mathbf{X}
        """)
        st.markdown("""
Each principal component *z_j* is a linear combination of the original features,
weighted by the corresponding eigenvector *w_j*. The first component captures the
most variance, the second captures the most remaining variance orthogonal to the
first, and so on. The key decision is how many components *k* to retain — typically
chosen to explain 90–95% of total variance.

**PCA limitations:** The components are linear combinations and may not be interpretable
in domain terms. PCA is also purely unsupervised — it finds directions of maximum
*variance*, not maximum *predictive power*. A direction with high variance might be
noise, and a direction with low variance might contain the signal.

**UMAP** is a nonlinear alternative that preserves local neighborhood structure rather
than global variance. It excels at visualization and can capture manifold structure
that PCA misses, but its components are even less interpretable than PCA's.
""")
        section("Topological Data Analysis (TDA)")
        st.markdown(f"""
TDA is an advanced feature engineering technique that captures the *shape* and
*structure* of your data using concepts from algebraic topology — specifically,
**persistent homology**. {cite("Carlsson, 2009")}

The intuition: imagine your data points as stars in the sky. Now imagine drawing
a growing circle around each point. As the circles expand:
- First, individual points are isolated (0-dimensional features).
- At some radius, circles begin to overlap, forming *connected components* — clusters.
- At larger radii, circles enclose empty regions, forming *loops* (1-dimensional holes).
- Eventually, everything merges into a single blob.

Persistent homology tracks when these topological features (components, loops, voids)
*appear* and *disappear* as the radius grows. Features that persist over a wide
range of radii represent genuine structure; features that appear and vanish quickly
are noise. This birth-death information is summarized in a **persistence diagram**,
from which numerical features are extracted: persistence entropy, amplitude statistics,
and Betti numbers at various scales.

**When TDA adds value:** Datasets where the spatial *arrangement* of points carries
signal beyond what individual feature values capture. Examples include protein
structure data, sensor networks, and datasets with complex nonlinear manifold
structure. {cite("Otter et al., 2017")}

**When to avoid:** TDA features are computationally expensive (the Vietoris-Rips
complex scales poorly beyond a few thousand points) and nearly impossible to
interpret in domain terms. For publication, using TDA features requires substantial
justification and typically a supplementary methods section. If simpler features
achieve comparable performance, prefer them.
""", unsafe_allow_html=True)
        app_connection(
            "The <strong>Feature Engineering</strong> page offers all of these transforms: "
            "polynomial features (degree 2–3), mathematical transforms (log, sqrt, square, "
            "reciprocal), ratio features, binning (equal-width, quantile, K-means), "
            "PCA, UMAP, and even TDA (topological data analysis). Each has an explainability "
            "impact rating to help you weigh the predictive benefit against interpretability cost."
        )

        with st.expander("Deep Dive: The Explainability-Performance Tradeoff"):
            st.markdown(f"""
Every feature engineering step sits on a spectrum from highly interpretable to
highly opaque. The Feature Engineering page rates each technique:

- 🟢 **Low impact** — Ratio features, mathematical transforms. "log(glucose)" is
  still one column with a clear meaning.
- 🟡 **Medium impact** — Polynomial features. "BMI² × Age" is harder to explain
  in a methods section but still traceable.
- 🔴 **High impact** — PCA, UMAP, TDA. The features are abstract constructions
  with no direct domain interpretation.

For publication, this tradeoff matters. A reviewer can evaluate "we log-transformed
skewed features" easily. "We used the first 5 principal components" requires more
justification. "We computed persistent homology on the Vietoris-Rips complex" may
require an entire supplementary section. {cite("ISLR, §2.1.3")}

The right choice depends on your audience and goals. If interpretability is paramount
(clinical prediction models), stay in the green zone. If raw prediction accuracy
matters most (competition, screening), the orange and red techniques may be worthwhile.
""", unsafe_allow_html=True)

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §7.1, §12.2. Springer.",
            "Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255–308.",
            "Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., & Harrington, H.A. (2017). A roadmap for the computation of persistent homology. *EPJ Data Science*, 6(1), 17.",
        ])

    # ── Encoding Categoricals ────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
Most machine learning algorithms require numeric inputs. When your data contains
categorical variables (sex, treatment group, hospital site), you must encode them
as numbers — and how you do this is not a neutral choice. Different encoding
strategies carry different assumptions and work better with different model families.
""")
        section("One-Hot Encoding")
        st.markdown("""
One-hot encoding creates a new binary column for each category level. A "color"
feature with values {red, green, blue} becomes three columns: *is_red*, *is_green*,
*is_blue*, each containing 0 or 1.

**Advantages:** Makes no assumption about ordering or magnitude. Each category gets
its own coefficient in a linear model, allowing the model to assign completely
different effects to each level.

**Disadvantages:** Creates *k* new columns for a feature with *k* levels. High-cardinality
features (hospital ID with 200 levels) can explode the feature space, causing
curse-of-dimensionality problems. Also introduces perfect multicollinearity — the
dummy columns sum to 1 — which is why one level is typically dropped as the
reference category.
""")
        section("Ordinal Encoding")
        st.markdown("""
Ordinal encoding assigns integers (0, 1, 2, ...) to categories. This is appropriate
*only* when the categories have a natural order: education level (high school < bachelor's
< master's < PhD), pain severity (none < mild < moderate < severe).

The encoding implies that the distance between adjacent levels is equal (the
difference between mild and moderate is the same as between moderate and severe),
which may or may not be true. For linear models, this matters because the coefficient
represents the effect of a one-unit increase in the encoded variable. For tree-based
models, the exact distances don't matter — only the rank order is used for splits.

**Never use ordinal encoding for nominal categories** (color, country, blood type).
Assigning red = 0, green = 1, blue = 2 implies that green is "between" red and blue,
which is meaningless.
""")
        section("Target Encoding")
        st.markdown("""
Target encoding replaces each category level with the mean of the target variable
for that level. For a classification task, "Hospital A" might be encoded as 0.34
(the proportion of positive outcomes at Hospital A).

This is powerful for high-cardinality features where one-hot encoding would create
too many columns. But it carries a serious risk: **data leakage**. The encoding uses
the target variable, which means information about *y* is baked into the feature *x*.
Without careful regularization, the model can overfit to the noise in the target
means — especially for rare categories where the mean is estimated from very few
observations.

The standard mitigation is **leave-one-out target encoding with additive smoothing**:
each observation's encoding uses the target mean computed *without* that observation,
and rare categories are smoothed toward the global mean. Even with these safeguards,
target encoding should be done inside cross-validation to prevent information leakage
from the test set.
""")
        app_connection(
            "The <strong>Feature Engineering</strong> page offers one-hot, ordinal, and target "
            "encoding. It warns when one-hot encoding will create more than 20 columns "
            "(suggesting ordinal or target encoding instead) and automatically handles the "
            "reference category for dummy variables."
        )

        with st.expander("Deep Dive: The Dummy Variable Trap"):
            st.markdown("""
If a categorical feature with *k* levels is one-hot encoded into *k* dummy columns,
those columns are perfectly collinear — they always sum to 1. Including all *k* in a
linear model makes **XᵀX** singular (non-invertible), because any one column can be
exactly predicted from the others.

The standard solution: drop one level as the **reference category**. The remaining
*k − 1* coefficients then represent the *difference* from the reference level. For
example, if "treatment group" has levels {placebo, drug A, drug B} and placebo is the
reference, the coefficients for drug A and drug B represent the *effect relative to
placebo*.

The choice of reference category doesn't affect predictions — only the
interpretation of individual coefficients changes. Tree-based models are not affected
by this issue, since they don't invert XᵀX.
""")

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §3.3.1. Springer.",
            "Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations*, 3(1), 27–32.",
        ])

    # ── Selection Methods ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
Feature selection removes irrelevant or redundant features before model training.
The goal is to reduce noise, improve interpretability, and avoid overfitting —
especially important when the number of features is large relative to sample
size. {cite("ISLR, §6.1")}

Three families of selection methods exist, each with different assumptions and
computational costs.
""", unsafe_allow_html=True)

        section("LASSO Path (Embedded Method)")
        st.markdown(f"""
LASSO (Least Absolute Shrinkage and Selection Operator) performs feature selection
as a byproduct of model fitting. It adds an L1 penalty to the regression loss:
{cite("Tibshirani, 1996")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \hat{\beta}^{\text{lasso}} = \underset{\beta}{\arg\min} \left\{
        \frac{1}{2n} \| \mathbf{y} - \mathbf{X}\beta \|_2^2 + \lambda \| \beta \|_1
        \right\}
        """)
        st.markdown("""
The key to LASSO's selection behavior is the geometry of the L1 penalty. The
constraint region ‖β‖₁ ≤ t forms a *diamond* in coefficient space (a square in 2D,
a cross-polytope in higher dimensions). The corners of this diamond lie on the
coordinate axes — meaning they correspond to solutions where some coefficients are
exactly zero. As the penalty strength λ increases, the constraint region shrinks,
and the optimal solution is pushed toward these corners, zeroing out features one
by one.

The **LASSO path** traces this process: starting from λ = 0 (the full OLS solution)
and increasing λ until all coefficients are zero. Features that survive to higher
values of λ are the most important. The app uses cross-validation to select the
optimal λ — the value that minimizes prediction error on held-out folds.

**Limitation:** When features are correlated, LASSO tends to select one arbitrarily
and zero the others. Elastic Net (L1 + L2 penalty) is more stable in this case,
as the L2 component encourages correlated features to share the coefficient.
""")
        section("RFE-CV (Wrapper Method)")
        st.markdown("""
Recursive Feature Elimination with Cross-Validation (RFE-CV) is a greedy backward
selection algorithm:

1. Train a model on all features.
2. Rank features by importance (e.g., absolute coefficient size for linear models,
   impurity importance for trees).
3. Remove the least important feature.
4. Repeat steps 1–3, recording cross-validated performance at each step.
5. Select the feature set that produced the best CV score.

RFE-CV is **model-aware** — it uses the actual model's importance rankings, so the
selected features are tailored to the model you'll use downstream. But it is
computationally expensive: each elimination step requires retraining the model,
and with cross-validation at each step, the total cost is O(p · k) model fits
(p features × k CV folds).

**Limitation:** Greedy elimination can miss feature interactions — removing feature A
might look fine at step 3, but feature A might be critical in combination with feature
B that's removed at step 7. Once A is removed, it never comes back.
""")
        section("Univariate Screening (Filter Method)")
        st.markdown(f"""
Univariate screening is the simplest and fastest approach: test each feature
*individually* against the target variable and keep those with statistically
significant associations.

For regression tasks, the app uses the **Pearson correlation coefficient** between
each feature and the target. For classification, it uses **Spearman's rank correlation**,
which captures monotone (not just linear) relationships. Each test produces a
p-value measuring the strength of evidence against the null hypothesis that
there is no association.

The critical challenge with univariate screening is the **multiple testing problem**.
When you test 100 features at α = 0.05, you expect about 5 false positives even
when no features are truly associated. The app applies the **Benjamini-Hochberg
(BH) procedure** to control the False Discovery Rate (FDR): {cite("Benjamini & Hochberg, 1995")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \text{FDR} = \mathbb{E}\left[\frac{\text{false positives}}{\text{total positives}}\right] \leq \alpha
        """)
        st.markdown("""
The BH procedure works by ranking the p-values from smallest to largest, then
finding the largest rank *k* where *p_(k) ≤ (k/m) · α* (with *m* being the total
number of tests). All features with rank ≤ *k* are selected. This is less
conservative than the Bonferroni correction (which divides α by the number of
tests) but still controls the expected proportion of false discoveries.

**Limitation:** Univariate screening tests each feature in isolation. It will miss
features that are only predictive *in combination* (e.g., an interaction between
age and sex where neither alone predicts the outcome). It can also retain redundant
features that carry the same signal.
""")
        section("Stability Selection (Bootstrap Method)")
        st.markdown(f"""
Stability selection addresses a fundamental question: *how confident should we be
that a feature's selection isn't an accident of this particular sample?*
{cite("Meinshausen & Bühlmann, 2010")}

The algorithm is elegantly simple:

1. Repeatedly draw random subsamples (typically 50% of the data, 100 times).
2. Run LASSO on each subsample.
3. For each feature, compute the **selection probability** — the fraction of
   subsamples in which it was selected (had a non-zero coefficient).
4. Keep features whose selection probability exceeds a threshold (default: 60%).

A feature selected in 90 out of 100 subsamples is almost certainly carrying real
signal. A feature selected in 15 out of 100 may have been an artifact of
a particular data split.
""", unsafe_allow_html=True)
        st.latex(r"""
        \hat{\Pi}_j = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\left[|\hat{\beta}_j^{(b)}| > 0\right]
        """)
        st.markdown("""
Here *Π̂_j* is the selection probability for feature *j*, *B* is the number of
bootstrap subsamples, and *β̂_j^(b)* is the LASSO coefficient for feature *j*
in subsample *b*. The indicator function counts how many times LASSO selected
this feature across all subsamples.

Stability selection provides *error control guarantees*: the expected number of
falsely selected features is bounded, regardless of the distribution of the data
or the number of features. This makes it particularly valuable in high-dimensional
settings where other methods may be unreliable.

**Tradeoff:** Stability selection is computationally expensive (B × LASSO fits) and
can be conservative — it tends to select fewer features than other methods, favoring
precision over recall.
""")
        section("Consensus Approach")
        st.markdown("""
Because each selection method has different blind spots, the app runs multiple
methods and reports the **consensus** — features selected across methods. A feature
that survives LASSO's geometric sparsity, RFE's greedy importance ranking,
univariate statistical testing, *and* stability selection's bootstrap resampling
has passed four fundamentally different filters. This provides much stronger
evidence of genuine signal than any single method alone.

The app shows which features were selected by all methods, most methods, or only
one method. Features in the consensus core are high-confidence selections. Features
selected by only one method may warrant manual review — they might represent real
signal that other methods missed, or noise that one method incorrectly retained.
""")
        app_connection(
            "The <strong>Feature Selection</strong> page runs up to four methods: LASSO path, "
            "RFE-CV, univariate screening (FDR-corrected), and stability selection. It shows "
            "individual and consensus results with manual override. The coaching layer warns "
            "when the consensus set is very small (potential signal loss) or very large "
            "(potential noise retention)."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §6.1, §6.2. Springer.",
            "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288.",
            "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289–300.",
            "Meinshausen, N. & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society, Series B*, 72(4), 417–473.",
            "Guyon, I. & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.",
        ])

    # ── Information Leakage ──────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
Information leakage occurs when information from the test set (or future data)
contaminates the training process, producing performance estimates that are
optimistically biased — sometimes drastically so. Feature selection is one of
the most common sources of leakage in applied ML.
""")
        section("The Feature Selection Leakage Problem")
        st.markdown("""
Consider the following workflow, which is **wrong but extremely common**:

1. Run feature selection on the entire dataset.
2. Split into train and test sets.
3. Train a model on the selected features.
4. Evaluate on the test set.

The problem: step 1 uses the test set. The feature selection algorithm "saw" the
test observations when deciding which features to keep. Features were selected partly
because they happened to correlate with the test set's target values, so the test
performance is optimistically biased.

The correct workflow wraps feature selection *inside* the cross-validation loop:

1. Split into train and test sets.
2. Run feature selection on the training set *only*.
3. Train a model on the selected features.
4. Evaluate on the test set using the same features selected in step 2.

This means the feature set may differ across folds — and that's correct. It
reflects the reality that different samples from the same population might lead
to different feature selections, which is itself useful information about the
stability of your features.
""")
        section("Other Sources of Leakage")
        st.markdown("""
Feature selection is just one entry point. Leakage can creep in through:

- **Target encoding before splitting.** If category means are computed on the full
  dataset, the encoding contains information about test-set targets.
- **Imputation before splitting.** Computing the median for imputation on the full
  dataset means the test set's values influence the imputed training values.
- **Scaling before splitting.** Computing mean and standard deviation on the full
  dataset means the scaler "knows" about the test distribution.

The general principle: **any operation that is estimated from data must be fit on
the training set and then applied (without re-fitting) to the test set.** This is
why the Preprocess page builds a *pipeline* — a sequence of fit-then-transform
steps that respects the train/test boundary.
""")
        with st.expander("🧮 Interactive: See what leakage does to your performance estimate"):
            leak_n = 200
            leak_p = st.slider(
                "Number of noise features (no real signal)",
                min_value=5, max_value=100, value=20, step=5,
                key="theory_leak_features",
                help="More features = more chances for spurious correlations to leak through.",
            )
            rng_leak = np.random.default_rng(44)
            X_leak = rng_leak.standard_normal((leak_n, leak_p))
            y_leak = rng_leak.choice([0, 1], size=leak_n)  # pure noise target

            # WRONG: select features on full data, then split
            from scipy.stats import pearsonr
            corrs = [abs(pearsonr(X_leak[:, j], y_leak)[0]) for j in range(leak_p)]
            top_k = min(5, leak_p)
            top_features = np.argsort(corrs)[-top_k:]

            split_idx = int(leak_n * 0.7)
            X_tr_leak = X_leak[:split_idx, :][:, top_features]
            y_tr_leak = y_leak[:split_idx]
            X_te_leak = X_leak[split_idx:, :][:, top_features]
            y_te_leak = y_leak[split_idx:]

            # Simple logistic-ish: use sign of weighted sum
            X_tr_aug_l = np.column_stack([np.ones(X_tr_leak.shape[0]), X_tr_leak])
            X_te_aug_l = np.column_stack([np.ones(X_te_leak.shape[0]), X_te_leak])
            beta_leak = np.linalg.lstsq(X_tr_aug_l, y_tr_leak, rcond=None)[0]
            pred_leak = (X_te_aug_l @ beta_leak > 0.5).astype(int)
            leaked_acc = np.mean(pred_leak == y_te_leak) * 100

            # RIGHT: select features on training data only
            corrs_honest = [abs(pearsonr(X_leak[:split_idx, j], y_leak[:split_idx])[0]) for j in range(leak_p)]
            top_features_honest = np.argsort(corrs_honest)[-top_k:]
            X_tr_h = X_leak[:split_idx, :][:, top_features_honest]
            X_te_h = X_leak[split_idx:, :][:, top_features_honest]
            X_tr_aug_h = np.column_stack([np.ones(X_tr_h.shape[0]), X_tr_h])
            X_te_aug_h = np.column_stack([np.ones(X_te_h.shape[0]), X_te_h])
            beta_h = np.linalg.lstsq(X_tr_aug_h, y_tr_leak, rcond=None)[0]
            pred_h = (X_te_aug_h @ beta_h > 0.5).astype(int)
            honest_acc = np.mean(pred_h == y_te_h) * 100

            fig_leak = go.Figure()
            fig_leak.add_trace(go.Bar(
                x=["Leaked (select on all data)", "Honest (select on train only)", "Chance (50%)"],
                y=[leaked_acc, honest_acc, 50],
                marker_color=["#dc2626", "#16a34a", "#94a3b8"],
                text=[f"{leaked_acc:.1f}%", f"{honest_acc:.1f}%", "50.0%"],
                textposition="outside",
            ))
            fig_leak.update_layout(
                title=f"Test accuracy on PURE NOISE data ({leak_p} features, top {top_k} selected)",
                yaxis_title="Accuracy (%)", yaxis_range=[0, 85], height=300,
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_leak, use_container_width=True)
            st.markdown(
                "**Train your eye:** The target is pure random noise — no real signal exists. Yet the red bar (leaked selection) "
                f"shows {leaked_acc:.0f}% accuracy, well above the 50% chance level. Why? Because feature selection on the full dataset "
                "found features that correlate with the test set by chance, then the model exploited those spurious correlations. "
                "The green bar (honest selection) should hover near 50% — which is the truth. "
                "Increase the number of noise features: more features = more chances for spurious correlations = worse leakage. "
                "**In your own results:** if test accuracy seems too good to be true, check whether any step used test data."
            )

        takeaway(
            "If your test performance is suspiciously close to your training performance, "
            "leakage is the first hypothesis to investigate. A 2% gap between train and "
            "test accuracy is normal. A 0.1% gap on a complex model is a red flag."
        )

        app_connection(
            "The app's pipeline architecture is designed to prevent leakage: all preprocessing "
            "steps (scaling, imputation, encoding) are fit on the training fold and applied to "
            "the test fold within cross-validation. Feature selection is performed before the "
            "train-test split as a practical compromise, but the coaching layer warns about this "
            "and recommends the user verify stability across CV folds."
        )

        references([
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §7.10.2. Springer.",
            "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 3: Model Families
# ══════════════════════════════════════════════════════════════════════════════

def render_model_families():
    intro(
        "Tabular ML Lab organizes models into six families based on their "
        "mathematical assumptions. Understanding these families is the key to "
        "understanding <em>why</em> the app recommends different preprocessing "
        "for different models — and why a choice that helps one model can hurt another."
    )

    with st.expander("🧮 Interactive: See how different model families fit the same data"):
        model_noise = st.slider(
            "Noise level",
            min_value=0.3, max_value=2.5, value=1.0, step=0.2,
            key="theory_family_noise",
            help="Higher noise makes the classification harder — watch how each model responds.",
        )
        model_choice = st.selectbox(
            "Model family",
            ["Linear (Logistic Regression)", "Tree (Decision Stump)", "KNN (k=3)", "KNN (k=15)"],
            key="theory_family_model",
        )

        rng_fam = np.random.default_rng(19)
        n_fam = 120
        # Two-class data with a slightly nonlinear boundary
        x1 = rng_fam.uniform(-3, 3, n_fam)
        x2 = rng_fam.uniform(-3, 3, n_fam)
        # True boundary: x2 > 0.5*sin(x1)
        true_label = (x2 > 0.5 * np.sin(x1 * 1.5)).astype(int)
        # Add noise by flipping some labels
        flip_mask = rng_fam.random(n_fam) < (model_noise * 0.15)
        labels_fam = np.where(flip_mask, 1 - true_label, true_label)

        # Create prediction grid
        gx = np.linspace(-3, 3, 60)
        gy = np.linspace(-3, 3, 60)
        gxx, gyy = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([gxx.ravel(), gyy.ravel()])

        X_fam = np.column_stack([x1, x2])

        if model_choice.startswith("Linear"):
            # Logistic regression via simple OLS on labels (good enough for demo)
            X_aug_fam = np.column_stack([np.ones(n_fam), X_fam])
            beta_fam = np.linalg.lstsq(X_aug_fam, labels_fam, rcond=None)[0]
            grid_pred = (np.column_stack([np.ones(len(grid_pts)), grid_pts]) @ beta_fam > 0.5).astype(int)
        elif model_choice.startswith("Tree"):
            # Simple decision stump: best single-axis split
            best_score, best_feat, best_thr = -1, 0, 0
            for feat in [0, 1]:
                for thr in np.linspace(-3, 3, 50):
                    pred_tmp = (X_fam[:, feat] > thr).astype(int)
                    score_tmp = np.mean(pred_tmp == labels_fam)
                    if score_tmp > best_score:
                        best_score, best_feat, best_thr = score_tmp, feat, thr
                    score_flip = np.mean((1 - pred_tmp) == labels_fam)
                    if score_flip > best_score:
                        best_score, best_feat, best_thr = score_flip, feat, thr
            pred_train = (X_fam[:, best_feat] > best_thr).astype(int)
            if np.mean(pred_train == labels_fam) < np.mean((1-pred_train) == labels_fam):
                grid_pred = (grid_pts[:, best_feat] <= best_thr).astype(int)
            else:
                grid_pred = (grid_pts[:, best_feat] > best_thr).astype(int)
        else:
            # KNN
            knn_k = 3 if "k=3" in model_choice else 15
            from scipy.spatial.distance import cdist
            dists_grid = cdist(grid_pts, X_fam)
            nn_indices = np.argsort(dists_grid, axis=1)[:, :knn_k]
            grid_pred = np.array([np.round(np.mean(labels_fam[nn_indices[i]])) for i in range(len(grid_pts))]).astype(int)

        grid_pred_2d = grid_pred.reshape(gxx.shape)

        fig_fam = go.Figure()
        fig_fam.add_trace(go.Heatmap(
            x=gx, y=gy, z=grid_pred_2d,
            colorscale=[[0, "rgba(37, 99, 235, 0.15)"], [1, "rgba(220, 38, 38, 0.15)"]],
            showscale=False,
        ))
        # Data points
        for cls, color, name in [(0, "#2563eb", "Class 0"), (1, "#dc2626", "Class 1")]:
            mask = labels_fam == cls
            fig_fam.add_trace(go.Scatter(
                x=x1[mask], y=x2[mask], mode="markers",
                marker=dict(size=5, color=color, line=dict(width=0.5, color="white")),
                name=name,
            ))
        fig_fam.update_layout(
            title=f"{model_choice} — decision regions",
            xaxis_title="Feature 1", yaxis_title="Feature 2",
            height=380, margin=dict(t=50, b=40, l=50, r=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fam, use_container_width=True)
        st.markdown(
            "**Train your eye:** The true boundary is a gentle curve. "
            "**Linear** draws a straight line — it can't capture the curve, but it's stable and interpretable. "
            "**Tree (stump)** cuts the space with one axis-aligned split — simple but rigid. "
            "**KNN (k=3)** follows the data closely, including the noise — watch how the boundary becomes jagged. "
            "**KNN (k=15)** smooths out, approaching a more stable (but less flexible) boundary. "
            "Increase the noise: linear and KNN(k=15) degrade gracefully; KNN(k=3) degrades fastest because it fits noise. "
            "**The lesson:** model choice is about matching the family's flexibility to the complexity of the real pattern in your data."
        )

    tabs = st.tabs([
        "Linear",
        "Tree-Based",
        "Neural Network",
        "Distance-Based",
        "Margin-Based",
        "Probabilistic",
    ])

    # ── Linear ───────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"""
The linear family assumes that the response is a weighted sum of the features
plus noise. This seemingly simple assumption is the foundation of the most
interpretable and theoretically well-understood models in
statistics. {cite("ISLR, §3")}
""", unsafe_allow_html=True)

        section("The Linear Model")
        st.latex(r"""
        \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p = \mathbf{x}^\top \boldsymbol{\beta}
        """)
        st.markdown("""
Each coefficient β_j represents the expected change in the response for a one-unit
increase in feature x_j, *holding all other features constant*. The intercept β₀
is the predicted response when all features are zero. This additive structure is
both the model's greatest strength (every coefficient has a direct interpretation)
and its greatest limitation (it cannot capture interactions or nonlinear effects
without explicit feature engineering).

The coefficients are estimated by minimizing the sum of squared residuals — the
Ordinary Least Squares (OLS) criterion:
""")
        st.latex(r"""
        \hat{\boldsymbol{\beta}}^{\text{OLS}} = \underset{\boldsymbol{\beta}}{\arg\min}
        \sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2
        = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
        """)
        st.markdown("""
The closed-form solution on the right exists when **XᵀX** is invertible — which
requires that no feature is a perfect linear combination of others (no perfect
collinearity) and that there are at least as many observations as features (n ≥ p).
""")

        section("Regularization: Ridge, LASSO, Elastic Net")
        st.markdown(f"""
When features are correlated, the sample is small, or there are many features,
OLS coefficients become unstable — small changes in the data produce large changes
in β̂. Regularization adds a penalty to the loss function that shrinks coefficients
toward zero, trading a small amount of bias for a large reduction in
variance. {cite("ISLR, §6.2")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \hat{\boldsymbol{\beta}}^{\text{ridge}} = \underset{\boldsymbol{\beta}}{\arg\min}
        \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
        """)
        st.markdown("""
**Ridge regression** adds the squared L2 norm of the coefficients. Geometrically,
the constraint region is a sphere: it shrinks all coefficients toward zero but
never exactly to zero. This stabilizes correlated coefficients by pulling them
toward each other. The penalty parameter λ controls the strength: λ = 0 recovers
OLS; as λ → ∞, all coefficients approach zero.
""")
        st.latex(r"""
        \hat{\boldsymbol{\beta}}^{\text{lasso}} = \underset{\boldsymbol{\beta}}{\arg\min}
        \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}
        """)
        st.markdown("""
**LASSO** uses the L1 norm (the sum of absolute values ‖β‖₁ = Σ|β_j|) instead of
the L2 norm. The constraint region is now a diamond in coefficient space, and the
corners of a diamond lie on the coordinate axes — where some coefficients are
exactly zero. As the penalty λ increases, the solution moves toward these corners,
zeroing out coefficients one by one. This makes LASSO a simultaneous estimator and
feature selector: it automatically identifies which features to keep and which to
discard.

**Elastic Net** combines both penalties: α·‖β‖₁ + (1−α)·‖β‖₂², where α controls
the mix between L1 (sparsity) and L2 (stability). When α = 1, it reduces to LASSO;
when α = 0, to Ridge. Elastic Net is often the best default for real-world data
because it handles correlated features better than pure LASSO — instead of
arbitrarily selecting one from a group of correlated features, it tends to keep
the group together and shrink their coefficients jointly.
""")

        section("Logistic Regression")
        st.markdown("""
For binary classification, the linear model is adapted by passing the linear
predictor through the **logistic (sigmoid) function**, which maps any real number
to the interval (0, 1):
""")
        st.latex(r"""
        P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \boldsymbol{\beta})
        = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}}
        """)
        st.markdown("""
The sigmoid function σ(z) = 1/(1 + e^(−z)) takes the linear combination
**x**ᵀ**β** (which can range from −∞ to +∞) and squashes it into the interval (0, 1),
producing a valid probability. When **x**ᵀ**β** is large and positive, σ is close to 1;
when large and negative, σ is close to 0; when zero, σ = 0.5.

Coefficients are estimated by maximum likelihood rather than least squares. Each
coefficient β_j represents the change in the *log-odds* of the positive class for
a one-unit increase in x_j. The log-odds (or logit) is:
""")
        st.latex(r"""
        \log \frac{P(Y=1 \mid \mathbf{x})}{1 - P(Y=1 \mid \mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta}
        """)
        st.markdown("""
This shows why logistic regression is "linear" — the log-odds is a linear function
of the features, even though the probability itself is nonlinear. The exponentiated
coefficient exp(β_j) gives the **odds ratio**: the factor by which the odds multiply
for a one-unit increase in x_j. An odds ratio of 1.5 means the odds of the positive
class are 50% higher for each additional unit of x_j. Clinicians and epidemiologists
work with odds ratios directly because they are interpretable and comparable across
studies.
""")

        section("Generalized Linear Models (GLM) and Huber Regression")
        st.markdown("""
**GLMs** extend linear regression to non-normal response distributions (Poisson for
counts, Gamma for positive continuous data, Binomial for proportions) through a
link function that connects the linear predictor to the expected response. Logistic
regression is a special case (binomial GLM with logit link).

**Huber regression** replaces the squared loss with a hybrid loss that is quadratic
for small residuals and linear for large ones, providing robustness to outliers
while retaining the interpretability of linear coefficients (see the Outliers
section in Chapter 1 for the loss function).
""")

        section("Assumptions and When Linear Models Fail")
        st.markdown("""
Linear models assume:
1. **Linearity** — the true relationship between features and response is additive.
2. **Independence** — observations are independent of each other.
3. **Homoscedasticity** — the variance of residuals is constant across the range of predictions.
4. **No perfect collinearity** — no feature is a perfect linear combination of others.

For inference (confidence intervals, p-values), an additional assumption is needed:
5. **Normality** — residuals are normally distributed (required for exact finite-sample inference; relaxed asymptotically by the Central Limit Theorem).

When these assumptions are violated, the model may still predict well but
coefficient interpretations and p-values become unreliable.
""")
        app_connection(
            "The app offers Ridge, LASSO, Elastic Net, Logistic Regression, GLM, and "
            "Huber regression in the linear family. The coaching layer recommends "
            "regularization when collinearity or high dimensionality is detected, and "
            "suggests Huber when outliers are prevalent."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §3, §6.2. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §3, §4.4. Springer.",
        ])

    # ── Tree-Based ───────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown(f"""
Tree-based methods partition the feature space into rectangular regions using a
sequence of binary splits, then make predictions within each region. They are
among the most practically useful methods for tabular data: they handle nonlinearity,
interactions, and mixed feature types naturally, with minimal
preprocessing. {cite("ISLR, §8")}
""", unsafe_allow_html=True)

        section("Decision Trees: Recursive Partitioning")
        st.markdown("""
A decision tree grows by repeatedly finding the feature and split point that best
separates the data. At each internal node, the algorithm asks "is feature x_j ≤ t?"
and sends observations left or right based on the answer. The process continues
until a stopping criterion is met (maximum depth, minimum leaf size, etc.).

For regression, "best separates" means minimizing the sum of squared residuals within
each resulting region. For classification, it means maximizing the purity of each
node, typically measured by the **Gini impurity**:
""")
        st.latex(r"""
        G = \sum_{k=1}^{K} \hat{p}_k (1 - \hat{p}_k)
        """)
        st.markdown("""
Here *p̂_k* is the proportion of observations in the node that belong to class *k*.
Gini impurity is zero when a node is perfectly pure (all one class) and maximized
when classes are evenly split. The split that produces the largest decrease in
impurity is chosen. An alternative is **entropy** (also called information gain),
which uses −Σ p̂_k log(p̂_k) instead — the two rarely disagree in practice.
""")

        section("Why Trees Don't Need Preprocessing")
        st.markdown("""
Trees have a remarkable property: they are **invariant to monotone transformations**
of the features. Whether a feature is measured in inches or centimeters, raw or
log-transformed, the splits will occur at equivalent points. This is because splits
are determined by rank order, not magnitude.

This means:
- **Scaling is irrelevant** — trees don't use distances between feature values.
- **Skewness doesn't matter** — the split is the same whether the feature is normal or heavily skewed.
- **Outliers have minimal effect** — an extreme value just ends up in its own leaf or a small group.
- **Mixed feature types are handled naturally** — categorical features are split by subset membership, numerical by threshold.
""")

        section("Random Forest: Bagging + Feature Randomization")
        st.markdown(f"""
A single decision tree is a high-variance estimator: small changes in the data
can produce very different trees. **Random Forest** reduces this variance through
two mechanisms: {cite("Breiman, 2001")}

1. **Bagging (bootstrap aggregating):** Train many trees, each on a different
   bootstrap sample (random sample with replacement) of the data. Average their
   predictions. By the law of large numbers, the average of many noisy-but-unbiased
   estimators has lower variance than any single one.

2. **Feature randomization:** At each split, only a random subset of features is
   considered (typically √p for classification, p/3 for regression). This *decorrelates*
   the trees — without it, every tree would make the same first split on the
   strongest feature, and the trees would be too similar for averaging to help.
""", unsafe_allow_html=True)

        section("Gradient Boosting: Sequential Error Correction")
        st.markdown(f"""
While Random Forest builds trees *in parallel* and averages them, gradient boosting
builds trees *sequentially*, each correcting the errors of the ensemble so far.
{cite("Friedman, 2001")}

At each step, the algorithm:
1. Computes the **residuals** (errors) of the current ensemble.
2. Fits a new (typically shallow) tree to these residuals.
3. Adds the new tree to the ensemble, scaled by a learning rate η.
""", unsafe_allow_html=True)
        st.latex(r"""
        F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
        """)
        st.markdown("""
Reading this equation: *F_{m-1}*(**x**) is the current prediction from the ensemble
of the first *m − 1* trees. The new tree *h_m*(**x**) is fit to the *residuals*
(errors) of *F_{m-1}* — it specifically targets what the ensemble is getting wrong.
Adding η · *h_m*(**x**) corrects a fraction of the remaining error. The learning
rate *η* (typically 0.01–0.3) controls the fraction: smaller η means the correction
is more cautious, which requires more trees but generally produces better results
because the ensemble avoids over-correcting to any single tree's biases.

**Histogram Gradient Boosting** discretizes continuous features into 256 bins
before training, dramatically speeding up split finding while producing virtually
identical results. Scikit-learn's HistGradientBoosting, XGBoost (with
`tree_method='hist'`), and LightGBM all use this approach.

**XGBoost** extends the standard gradient boosting objective with explicit L1
(*reg_alpha*) and L2 (*reg_lambda*) penalties on leaf weights, column subsampling
(*colsample_bytree*), and row subsampling (*subsample*). These regularization
knobs make it less prone to overfitting on noisy tabular data and are a large
part of why it dominated Kaggle competitions through the late 2010s.

**LightGBM** replaces the conventional *level-wise* (breadth-first) tree growth
strategy with *leaf-wise* (best-first) growth: at each step it splits the leaf
with the largest loss reduction, regardless of depth. This produces deeper,
more asymmetric trees that often converge faster than XGBoost on the same number
of iterations. LightGBM also introduces:

- **GOSS (Gradient-based One-Side Sampling):** retains instances with large
  gradients (where the model is most wrong) and randomly samples instances with
  small gradients, reducing data volume without discarding the most informative
  examples.
- **EFB (Exclusive Feature Bundling):** bundles mutually exclusive sparse
  features (features that rarely take non-zero values simultaneously) into a
  single feature, cutting the effective feature count and speeding up split
  finding.

Both XGBoost and LightGBM handle missing values natively (learning which
branch to send NaN observations down) and support early stopping when a
validation set is provided.
""")

        section("Feature Importance in Trees")
        st.markdown("""
Trees provide two kinds of feature importance:

**Impurity-based importance** (also called Gini importance) sums the total reduction
in impurity across all splits that use a given feature. It is fast to compute but
biased toward high-cardinality features and features with many possible split points.

**Permutation importance** randomly shuffles a feature's values across observations
and measures how much the model's performance degrades. It is more reliable but
slower, and it correctly attributes importance to features that matter for
prediction rather than just for node purity.
""")

        app_connection(
            "The app offers Random Forest, Extra Trees, Histogram Gradient Boosting, "
            "XGBoost, and LightGBM in both regression and classification variants. "
            "None of these models require feature scaling or one-hot encoding — the "
            "coaching layer skips those preprocessing steps automatically. TreeSHAP "
            "is used for explainability (fast exact Shapley values for tree models), "
            "and permutation importance is used rather than Gini importance."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §8. Springer.",
            "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.",
            "Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.",
        ])

    # ── Neural Network ───────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
The app includes a multi-layer perceptron (MLP) — a feedforward neural network with
one or more hidden layers. Neural networks are the most flexible model family
in the app, capable of learning arbitrary nonlinear functions — but this flexibility
comes at the cost of interpretability and data requirements. {cite("ISLR, §10")}
""", unsafe_allow_html=True)

        section("Architecture: Layers, Neurons, Activations")
        st.markdown("""
An MLP consists of an input layer (one neuron per feature), one or more hidden
layers, and an output layer. Each neuron in a hidden layer computes a weighted
sum of its inputs, adds a bias term, and passes the result through a nonlinear
**activation function**:
""")
        st.latex(r"""
        h_j = \sigma\!\left(\sum_{i=1}^{p} w_{ij} \, x_i + b_j\right)
        """)
        st.markdown("""
Here *w_ij* are the weights connecting input *i* to hidden neuron *j*, *b_j* is the
bias, and *σ* is the activation function. The activation introduces nonlinearity —
without it, stacking multiple layers would still produce a linear model (a
composition of linear functions is linear). Common activations include:

- **ReLU** (Rectified Linear Unit): max(0, z). Simple, fast, and avoids the vanishing
  gradient problem. The default in most modern networks.
- **Sigmoid**: 1/(1 + e^(−z)). Maps to (0, 1); used in the output layer for
  binary classification. Can cause vanishing gradients in deep networks.
- **Tanh**: (e^z − e^(−z))/(e^z + e^(−z)). Maps to (−1, 1); centered at zero,
  which can help training dynamics.
""")

        section("The Universal Approximation Theorem")
        st.markdown(f"""
A neural network with a single hidden layer containing enough neurons can approximate
any continuous function on a compact set to arbitrary accuracy. This is the
**universal approximation theorem**. {cite("Hornik et al., 1989")}

This sounds powerful, but the theorem says nothing about *how many* neurons are
needed, *how much data* is required to learn the approximation, or whether gradient
descent will find the right weights. In practice, the binding constraint for
tabular data is almost always sample size: with n = 500, the network has enough
capacity to memorize the training set, but not enough data to learn the true
underlying function.
""", unsafe_allow_html=True)

        section("Training: Backpropagation and Gradient Descent")
        st.markdown("""
The weights are learned by minimizing a loss function (squared error for regression,
cross-entropy for classification) using **gradient descent**: compute the gradient
of the loss with respect to each weight, then update the weights in the direction
that decreases the loss.

**Backpropagation** is the algorithm that efficiently computes these gradients by
applying the chain rule layer by layer, from the output back to the input. Each
training iteration:

1. **Forward pass:** compute predictions by passing inputs through all layers.
2. **Loss computation:** compare predictions to true labels.
3. **Backward pass:** compute gradients of the loss with respect to every weight.
4. **Weight update:** adjust weights by −η · gradient (where η is the learning rate).

**Why scaling matters for neural networks:** If features have very different scales,
the loss landscape becomes elongated — gradient descent oscillates in the steep
(large-scale) direction while making slow progress in the flat (small-scale)
direction. Standardizing features to zero mean and unit variance makes the landscape
more spherical and training much faster.
""")

        section("Overfitting and Regularization")
        st.markdown("""
Neural networks have many more parameters than linear models, making overfitting a
serious concern — especially on the small-to-medium datasets typical in research.
The app applies several regularization techniques:

- **Dropout:** During training, randomly zero out a fraction of neurons in each layer.
  This prevents co-adaptation (neurons that rely on each other rather than
  individually contributing signal).
- **Weight decay (L2 regularization):** Add ‖w‖² to the loss, penalizing large weights.
- **Early stopping:** Monitor validation loss during training and stop when it begins
  to increase, indicating the model is starting to memorize training noise.
""")
        takeaway(
            "For tabular data with n < 1,000, neural networks rarely outperform "
            "well-tuned gradient boosting. Consider them when you have abundant data "
            "(n > 5,000) or when the relationship between features and target is highly "
            "nonlinear and interactions are complex."
        )

        app_connection(
            "The app trains a PyTorch MLP with configurable hidden layers, dropout, "
            "learning rate, and epochs. It shows training curves (loss over epochs) "
            "and applies early stopping automatically. SHAP values are computed using "
            "the kernel method, which is model-agnostic but slower than the linear "
            "or tree-specific SHAP methods."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §10. Springer.",
            "Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.",
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 6–8. MIT Press.",
        ])

    # ── Distance-Based ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(f"""
Distance-based methods make predictions based on the *similarity* between a new
observation and the training data. The core assumption is that similar inputs
produce similar outputs — a form of the smoothness
assumption. {cite("ISLR, §2.2.3")}
""", unsafe_allow_html=True)

        section("k-Nearest Neighbors (KNN)")
        st.markdown("""
KNN is the simplest non-parametric method. To predict for a new observation **x**:

1. Compute the distance from **x** to every training observation.
2. Find the *k* nearest neighbors.
3. For regression: average their responses. For classification: take a majority vote.

There is no training phase — the algorithm stores the entire training set and does
all its work at prediction time (a "lazy learner"). The only hyperparameter is *k*,
which controls the bias-variance tradeoff:
""")
        st.latex(r"""
        \hat{y}(\mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i
        """)
        st.markdown("""
Here 𝒩_k(**x**) is the set of *k* indices of training observations closest to **x**.

- **Small k (e.g., k = 1):** The decision boundary is highly flexible, conforming
  to every local variation in the data. Low bias, high variance. Prone to overfitting.
- **Large k (e.g., k = n):** The prediction is the global average — a horizontal line.
  High bias, zero variance. Underfits completely.
""")

        with st.expander("🧮 Interactive: The bias-variance tradeoff — watch underfitting become overfitting"):
            bv_k = st.slider(
                "Number of neighbors (k)",
                min_value=1, max_value=50, value=5, step=1,
                key="theory_bv_k",
                help="k = 1 memorizes noise; k = n predicts the global mean. Find the sweet spot.",
            )
            rng_bv = np.random.default_rng(31)
            n_bv = 80
            x_bv = rng_bv.uniform(0, 6, n_bv)
            # True function: a smooth curve
            y_true_bv = np.sin(x_bv) * 2 + 0.3 * x_bv
            y_bv = y_true_bv + rng_bv.normal(0, 0.8, n_bv)

            # KNN predictions on a grid
            x_grid = np.linspace(0, 6, 200)
            y_knn = np.zeros_like(x_grid)
            for gi, xg in enumerate(x_grid):
                dists = np.abs(x_bv - xg)
                nn_idx = np.argsort(dists)[:bv_k]
                y_knn[gi] = np.mean(y_bv[nn_idx])

            # True function on grid
            y_true_grid = np.sin(x_grid) * 2 + 0.3 * x_grid

            # Compute train error and a rough proxy for test error
            y_train_pred = np.zeros(n_bv)
            for ti in range(n_bv):
                dists_t = np.abs(x_bv - x_bv[ti])
                nn_idx_t = np.argsort(dists_t)[:bv_k]
                y_train_pred[ti] = np.mean(y_bv[nn_idx_t])
            train_mse = np.mean((y_bv - y_train_pred) ** 2)
            truth_mse = np.mean((y_knn - y_true_grid) ** 2)

            fig_bv = go.Figure()
            fig_bv.add_trace(go.Scatter(x=x_bv, y=y_bv, mode="markers",
                marker=dict(size=5, color="rgba(99, 102, 241, 0.5)"),
                name="Training data"))
            fig_bv.add_trace(go.Scatter(x=x_grid, y=y_true_grid, mode="lines",
                line=dict(color="#16a34a", width=2, dash="dash"),
                name="True function"))
            fig_bv.add_trace(go.Scatter(x=x_grid, y=y_knn, mode="lines",
                line=dict(color="#dc2626", width=2),
                name=f"KNN fit (k={bv_k})"))
            fig_bv.update_layout(
                title=f"k = {bv_k} — Train MSE: {train_mse:.3f}, Deviation from truth: {truth_mse:.3f}",
                xaxis_title="x", yaxis_title="y", height=340,
                margin=dict(t=50, b=40, l=50, r=20), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bv, use_container_width=True)
            st.markdown(
                "**Train your eye:** At k = 1, the red line jumps through every data point — it memorizes the noise (overfitting). "
                "The train MSE is nearly zero, but the line deviates wildly from the green truth. "
                "At k = 50, the red line flattens — it misses the curve entirely (underfitting). "
                "Somewhere around k = 5–15, the red line tracks the green curve without chasing noise. "
                "**This is the bias-variance tradeoff:** too flexible and you fit noise; too rigid and you miss signal. "
                "Every model in this app has an equivalent knob — tree depth, regularization strength, number of hidden neurons."
            )

        section("Why Distance Metrics Matter")
        st.markdown("""
KNN uses Euclidean distance by default:
""")
        st.latex(r"""
        d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p} (x_j - x_j')^2}
        """)
        st.markdown("""
This formula treats all features equally: a difference of 1 unit in feature A
contributes the same as a difference of 1 unit in feature B. If feature A ranges
from 0 to 1 and feature B ranges from 0 to 10,000, feature B will completely
dominate the distance calculation and feature A will be effectively ignored.

This is why **feature scaling is critical** for KNN. Standardizing all features to
zero mean and unit variance puts them on an equal footing in the distance
calculation. Unlike linear models (where scaling only affects coefficient
magnitude, not predictions), scaling fundamentally changes which neighbors are
selected — and therefore changes the predictions themselves.
""")

        section("The Curse of Dimensionality (Revisited)")
        st.markdown("""
KNN suffers uniquely from the curse of dimensionality (discussed in Chapter 1).
In high dimensions, all points become approximately equidistant, so the concept
of "nearest neighbor" loses meaning. The practical threshold depends on the data,
but KNN typically struggles when p > 20 unless the data lies on a low-dimensional
manifold.
""")

        app_connection(
            "The app offers KNN for both regression and classification. The coaching "
            "layer recommends standard scaling for KNN pipelines and warns when the "
            "feature space is high-dimensional. SHAP values use the kernel method."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §2.2.3. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §13.3. Springer.",
        ])

    # ── Margin-Based ─────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"""
Support Vector Machines (SVMs) find the decision boundary that maximizes the
*margin* — the distance between the boundary and the closest data points from
each class. This geometric approach produces models that generalize well,
especially in high-dimensional spaces. {cite("ISLR, §9")}
""", unsafe_allow_html=True)

        section("The Maximum Margin Classifier")
        st.markdown("""
For a binary classification problem where the classes are linearly separable,
infinitely many hyperplanes could separate them. The SVM chooses the one with the
largest margin:
""")
        st.latex(r"""
        \underset{\mathbf{w}, b}{\max} \; \frac{2}{\|\mathbf{w}\|} \quad
        \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \;\; \forall \, i
        """)
        st.markdown("""
The vector **w** defines the direction perpendicular to the separating hyperplane,
and *b* is the offset. The constraint y_i(**w**ᵀ**x**_i + b) ≥ 1 ensures that
every observation is on the correct side of the margin — *y_i* is +1 or −1
(the class label), so the product is positive when the prediction and label agree.
The margin width is 2/‖**w**‖ — maximizing it means finding the **w** with the
smallest norm that still correctly classifies all points.

The observations that lie exactly on the margin boundaries (**w**ᵀ**x** + b = ±1)
are the **support vectors** — they alone determine the decision boundary. All other
training points could be moved or removed without changing the model, which is a
form of built-in robustness.

For data that is *not* linearly separable, the **soft margin** formulation allows
some violations by introducing slack variables ξ_i and a penalty parameter *C*
that controls the tradeoff: large *C* penalizes violations harshly (narrow margin,
few misclassifications); small *C* tolerates more violations (wider margin,
more misclassifications).
""")

        section("The Kernel Trick")
        st.markdown("""
The real power of SVMs comes from the **kernel trick**: implicitly mapping the
data into a higher-dimensional space where a linear boundary may succeed, *without
ever computing the coordinates in that space*. This works because the SVM
optimization depends only on inner products ⟨**x**_i, **x**_j⟩ between data
points, and a kernel function K(**x**_i, **x**_j) computes the inner product
in the higher-dimensional space directly.
""")
        st.latex(r"""
        K(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)
        \quad \text{(RBF kernel)}
        """)
        st.markdown("""
The RBF (Radial Basis Function) kernel computes a similarity score between two
observations **x** and **x'** based on the squared Euclidean distance between them
(‖**x** − **x'**‖²). When two points are close (small distance), the kernel value
is near 1 (high similarity); when far apart (large distance), it decays toward 0.
The parameter γ controls how quickly this decay happens: large γ means the kernel
is narrow and each training point has only local influence (low bias, high variance);
small γ means the kernel is wide and each point influences a broad region
(high bias, low variance).

**Why scaling matters for SVMs:** The kernel computes distances between data points.
Just like KNN, unscaled features with large ranges will dominate the distance
calculation, making the kernel effectively ignore small-range features.
""")

        section("SVMs for Regression (SVR)")
        st.markdown("""
Support Vector Regression adapts the margin idea to regression by defining an
ε-insensitive tube around the regression line. Points inside the tube contribute
zero loss; points outside contribute linearly proportional to their distance from
the tube boundary. This produces a model that ignores small residuals and focuses
on fitting the overall trend — a form of built-in robustness to noise.
""")

        app_connection(
            "The app offers SVM (SVC) for classification and SVR for regression. The "
            "coaching layer recommends standard scaling for SVM pipelines, since the "
            "RBF kernel is distance-based. SHAP uses the kernel method for SVMs."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §9. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §12. Springer.",
        ])

    # ── Probabilistic ────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(f"""
Probabilistic models take a generative approach: they model the joint distribution
of features and class labels, then use Bayes' theorem to compute the posterior
probability of each class given the observed features. {cite("ISLR, §4.4")}
""", unsafe_allow_html=True)

        section("Naive Bayes")
        st.markdown("""
Naive Bayes applies Bayes' theorem with a strong simplifying assumption: all
features are **conditionally independent** given the class label.
""")
        st.latex(r"""
        P(Y = k \mid \mathbf{x}) \propto P(Y = k) \prod_{j=1}^{p} P(x_j \mid Y = k)
        """)
        st.markdown("""
The right-hand side has two components: the **prior** P(Y = k) (how common class *k*
is in the training data) and the **likelihood** — the product of individual feature
probabilities given the class. The "naive" assumption is that the features contribute
independently — knowing the value of x₁ tells you nothing about x₂, given the
class label.

This assumption is almost always violated in practice (features *are* correlated),
yet Naive Bayes often performs surprisingly well for classification. The reason:
classification only requires getting the *ranking* of posterior probabilities right
(which class has the highest probability), not the probabilities themselves. Even
when the probability estimates are poorly calibrated, the ranking can be correct.

**Gaussian Naive Bayes** (the variant in the app) assumes each feature follows a
normal distribution within each class, parameterized by the class-conditional
mean and variance. This is appropriate for continuous features but can struggle
when the true feature distributions are highly non-normal.
""")

        section("Linear Discriminant Analysis (LDA)")
        st.markdown("""
LDA also uses Bayes' theorem but makes a different assumption: the feature
distribution within each class is **multivariate normal** with a **shared
covariance matrix** across all classes.
""")
        st.latex(r"""
        P(\mathbf{x} \mid Y = k) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}
        \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}^{-1}
        (\mathbf{x} - \boldsymbol{\mu}_k)\right)
        """)
        st.markdown("""
This is the multivariate normal (Gaussian) density function. The term
(**x** − **μ_k**) measures how far observation **x** is from the center of class *k*.
The covariance matrix **Σ** describes the shape and spread of the class distribution —
it accounts for both the variance of each feature and the correlations between
features. The expression (**x** − **μ_k**)ᵀ**Σ**⁻¹(**x** − **μ_k**) is the
**Mahalanobis distance**: a distance metric that accounts for correlations and
scales, measuring how "unusual" **x** is relative to class *k*'s distribution.

The key consequence of assuming all classes share the *same* **Σ** is that the
decision boundaries between classes are *linear* (hyperplanes) — hence the name
"Linear" Discriminant Analysis. Each class has its own center (mean vector), but
the spread and orientation of the distribution around each center is the same.

LDA is closely related to logistic regression: both produce linear decision
boundaries for classification. The difference is philosophical — LDA is generative
(models the class distributions) while logistic regression is discriminative
(models the boundary directly). In practice, logistic regression is more robust
because it makes fewer distributional assumptions.
""")

        section("When to Use Probabilistic Models")
        st.markdown("""
Probabilistic models excel in specific situations:

- **Very small samples** — Naive Bayes has very few parameters to estimate
  (just means and variances per class), so it can work with remarkably little data.
- **Many classes** — Naive Bayes scales gracefully to multi-class problems.
- **Baseline comparison** — They provide a well-calibrated, fast baseline against
  which to measure more complex models.
- **Text-like data** — Naive Bayes is the classical baseline for document classification.

They struggle when:
- Features are highly correlated (violates the Naive Bayes independence assumption).
- The Gaussian assumption is badly wrong (highly skewed or multimodal features).
- The shared covariance assumption fails (LDA produces poor boundaries when
  class distributions have different shapes).
""")

        app_connection(
            "The app offers Gaussian Naive Bayes and LDA. The coaching layer warns "
            "when features show strong non-normality (which violates both models' "
            "assumptions) and recommends these models primarily as baselines or for "
            "very small samples where more complex models would overfit."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §4.4, §4.5. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §4.3. Springer.",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 4: Preprocessing Theory
# ══════════════════════════════════════════════════════════════════════════════

def render_preprocessing():
    intro(
        "Preprocessing is where theory meets practice. The choices you make here — "
        "how to scale, impute, and encode — are not neutral. Each decision carries "
        "assumptions that align with some models and conflict with others. This is why "
        "Tabular ML Lab builds <strong>per-model pipelines</strong>: the preprocessing "
        "that serves a Ridge regression well can actively harm a Random Forest."
    )

    tabs = st.tabs([
        "Per-Model Pipelines",
        "Scaling Methods",
        "Imputation Strategies",
        "Train-Test Splitting",
    ])

    # ── Per-Model Pipelines ──────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
Most ML tools apply a single preprocessing pipeline to all models. This is
fundamentally wrong, because different model families have different mathematical
requirements. Consider a simple example: standardizing features to zero mean and
unit variance.
""")
        st.markdown("""
- For **Ridge regression**, scaling is essential. Without it, the penalty treats
  coefficients unequally — features measured in large units have small coefficients
  that receive less regularization, and vice versa. The model's performance depends
  on an arbitrary choice of units.
- For **Random Forest**, scaling is pointless. Trees split on thresholds, and
  the splits are the same whether a feature ranges from 0 to 1 or 0 to 10,000.
  Scaling adds computational cost without any benefit.
- For **KNN**, scaling is critical. Distances are computed in feature space, and
  features with large ranges dominate the distance calculation. Without scaling,
  KNN effectively ignores small-range features.

A single pipeline that scales everything penalizes trees (unnecessary computation)
or doesn't scale, penalizing linear and distance models (degraded performance). The
correct approach is per-model pipelines.
""")

        section("How the App Builds Pipelines")
        st.markdown("""
For each selected model, the app constructs an independent scikit-learn `Pipeline`
object containing the following steps in order:

1. **Imputation** — Fill missing values (method may vary by model).
2. **Encoding** — Convert categorical features to numeric (one-hot, ordinal, etc.).
3. **Outlier treatment** — Clip or transform extreme values (optional, per-model).
4. **Scaling** — Standardize, normalize, or robust-scale features (per-model).
5. **Feature transformation** — Log, power, or PCA transforms (per-model).
6. **The model itself** — The estimator is the final step.

Every step is **fit** on the training data only, then **applied** (without re-fitting)
to the test data. This prevents data leakage: the test set never influences
the imputation values, scaling parameters, or encoding mappings.
""")

        app_connection(
            "The <strong>Preprocess</strong> page offers three configuration modes: "
            "<strong>Auto</strong> (smart defaults per model family based on EDA findings), "
            "<strong>Manual</strong> (full control per model), and an interpretability "
            "slider that restricts which transforms are allowed. Each model gets its "
            "own expandable configuration panel."
        )

    # ── Scaling Methods ──────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
Scaling transforms features to a common range or distribution. The choice of method
depends on the data's characteristics and the model's requirements.
""")

        section("Standard Scaling (Z-score)")
        st.latex(r"""
        z = \frac{x - \bar{x}}{s}
        """)
        st.markdown("""
Subtracts the mean and divides by the standard deviation, producing features with
zero mean and unit variance. This is the default for linear models and neural
networks. It assumes that the feature is approximately normal — if the feature has
a heavy tail or extreme outliers, the mean and standard deviation will be distorted,
and the "standardized" values will still have an uneven spread.
""")

        section("Robust Scaling")
        st.latex(r"""
        z = \frac{x - \text{median}}{Q_3 - Q_1}
        """)
        st.markdown("""
Uses the median and interquartile range (IQR) instead of the mean and standard
deviation. Because the median and IQR are based on percentiles rather than moments,
they are not influenced by outliers. If your data has outliers that you want to
keep (not clip), robust scaling prevents them from distorting the scale for all
other observations.
""")

        section("Min-Max Scaling")
        st.latex(r"""
        z = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
        """)
        st.markdown("""
Maps features to the range [0, 1]. This is useful when you need bounded inputs
(e.g., for certain neural network architectures) or when the original range is
meaningful and you want to preserve it. The downside: a single outlier can compress
all other values into a tiny portion of the [0, 1] range.
""")

        section("When Scaling Helps vs. Hurts")
        st.markdown("""
| Model Family | Needs Scaling? | Why |
|:---|:---|:---|
| Linear | **Yes** | Regularization penalty must treat coefficients equally |
| Neural Network | **Yes** | Gradient descent converges much faster with scaled inputs |
| Distance-based (KNN) | **Yes** | Distances dominated by large-range features |
| Margin-based (SVM) | **Yes** | Kernel computation is distance-based |
| Tree-based | **No** | Splits based on rank order, not magnitude |
| Probabilistic | **Depends** | LDA estimates covariance; NB estimates per-feature variance |
""")

        section("Outlier Treatment in Preprocessing")
        st.markdown("""
Beyond scaling, the preprocessing pipeline can explicitly treat outliers through
clipping — capping extreme values at a threshold rather than removing observations.

**Percentile clipping** caps values at a specified percentile (e.g., the 1st and
99th percentiles). Values below the 1st percentile are set to the 1st percentile
value; values above the 99th are capped similarly. This preserves the observation
while limiting the outlier's magnitude.

**IQR-based clipping** uses the interquartile range fences (Q1 − 1.5·IQR and
Q3 + 1.5·IQR) as thresholds, consistent with the outlier detection method used
in the EDA page.

The choice between clipping and robust scaling depends on the downstream model:
for linear models, robust scaling often suffices (it reduces the outlier's
influence without altering the value). For neural networks, clipping may be
preferable because it limits the input range explicitly, preventing extreme
activations.
""")

        section("Interpretability Mode")
        st.markdown("""
The preprocessing page offers three interpretability levels that control which
transforms are available:

- **High** — Only simple, explainable transforms. No PCA, no K-means binning, no
  log/power transforms. Pipelines are transparent and easy to describe in a
  methods section.
- **Balanced** — The recommended default. Allows most transforms but limits
  complexity.
- **Performance** — All transforms available, including PCA dimensionality
  reduction and aggressive feature transformations. Maximizes potential accuracy
  but may produce pipelines that are difficult to explain.

This is a design choice, not a mathematical one. The interpretability mode reflects
the tradeoff discussed in Chapter 2: every preprocessing step sits on a spectrum
from transparent to opaque, and the right position depends on your audience and
publication venue.
""")

        app_connection(
            "The app auto-selects the appropriate scaling method per model: standard "
            "scaling for linear/neural/distance/margin models, no scaling for trees, "
            "and robust scaling when the EDA detected outliers. The interpretability "
            "mode slider controls which transforms are available across all models. "
            "You can override individual settings in the per-model configuration panels."
        )

        references([
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §3.4. Springer.",
        ])

    # ── Imputation Strategies ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
This tab complements the Missing Data section in Chapter 1 (which covers the *theory*
of missing data mechanisms) with the practical details of each imputation method
available in the preprocessing pipeline.
""")

        section("Mean / Median Imputation")
        st.markdown("""
Replace each missing value with the mean (or median) of the observed values for that
feature. This is the fastest option and is appropriate as a baseline or when
missingness is rare (< 5%).

**Mean** imputation preserves the sample mean but underestimates the variance.
**Median** imputation is more robust to skewed features and outliers, since the
median is not pulled by extreme values. For heavily skewed features, median is
almost always preferable.
""")

        section("KNN Imputation")
        st.markdown("""
For each observation with a missing value, find the *k* nearest complete observations
(using the features that are present) and average their values for the missing
feature. This preserves local structure: an observation surrounded by high-value
neighbors gets a high imputed value, even if the overall mean is low.

**Configuration:** The number of neighbors *k* (default: 5) and the distance metric
(default: Euclidean with nan-aware handling). Larger *k* produces smoother imputations;
smaller *k* preserves local variation.

**Limitation:** Computationally expensive for large datasets (must compute distances
to all training observations for each imputation). Also sensitive to the distance
metric — which means the *scaling* of features affects the imputation.
""")

        section("Iterative Imputation (MICE)")
        st.markdown("""
As described in Chapter 1, MICE models each feature with missing values as a
function of all other features, cycling iteratively until convergence. In the
pipeline context, it is fit on the training set: the regression models learned
during fitting are then applied to impute missing values in the test set.

This is the most principled approach under the MAR assumption, but also the slowest
and most computationally intensive.
""")

        section("Missing Indicators")
        st.markdown("""
In addition to imputing the missing values, the pipeline can create binary
**missing indicator** columns — one per feature with missingness. These take value
1 when the original value was missing and 0 otherwise.

Why? If the *fact that a value is missing* carries information (e.g., a patient
didn't get a test because the doctor judged it unnecessary), the missing indicator
captures that signal. The model can then learn that "missingness in feature X is
predictive of outcome Y" — information that imputation alone would destroy.
""")

        app_connection(
            "The <strong>Preprocess</strong> page offers mean, median, KNN, and iterative "
            "imputation, configurable per model. Missing indicators can be added alongside "
            "any imputation method. The auto-configuration uses median imputation with "
            "missing indicators as its default."
        )

    # ── Train-Test Splitting ─────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(f"""
Splitting the data into training and test sets is the most fundamental step in
model evaluation. It simulates the real-world scenario: the model learns from
one set of data and is evaluated on data it has never seen. {cite("ISLR, §5")}
""", unsafe_allow_html=True)

        section("Simple Holdout Split")
        st.markdown("""
The simplest approach: randomly assign a fraction (typically 70–80%) of observations
to the training set and the remainder to the test set. The training set is used
for all model fitting and tuning; the test set is used exactly once, at the end,
for final evaluation.

**Advantages:** Simple, fast, easy to understand.
**Disadvantages:** The performance estimate depends on which observations happen
to land in the test set. With a small dataset, a different random split might give
substantially different results.
""")

        section("Stratified Splitting")
        st.markdown("""
For classification tasks — especially with class imbalance — a random split may
produce training and test sets with different class proportions. Stratified splitting
ensures that each set mirrors the overall class distribution.

This matters most when the minority class is small. If only 5% of observations
are positive and you do an 80/20 split, a random split might put 3% positive in the
test set and 7% in training, distorting both training dynamics and evaluation.
Stratification guarantees 5% in both sets.
""")

        section("The Test Set Is Sacred")
        st.markdown("""
The test set must **never** be used for any decision during model development:
not for feature selection, not for hyperparameter tuning, not for choosing between
models. Any time the test set influences a decision, the final evaluation is
optimistically biased — you've adapted your model to the test set's idiosyncrasies.

This is why the app uses **cross-validation** on the training set for all
model comparison and tuning, reserving the test set for final evaluation only.
""")

        app_connection(
            "The <strong>Train & Compare</strong> page configures the train/test split: "
            "split ratio, random seed, and stratification (automatic for classification). "
            "The random seed is saved for reproducibility and can be varied in the "
            "<strong>Sensitivity Analysis</strong> page to test stability."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §5. Springer.",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 5: Evaluation & Validation
# ══════════════════════════════════════════════════════════════════════════════

def render_evaluation():
    intro(
        "A model is only as good as the evidence you can marshal for its performance. "
        "Choosing the right metrics, validating properly, and explaining predictions "
        "are what separate a publishable result from a notebook experiment. This chapter "
        "covers the theory behind <strong>Train & Compare</strong> and "
        "<strong>Explainability</strong>."
    )

    tabs = st.tabs([
        "Classification Metrics",
        "Regression Metrics",
        "Cross-Validation",
        "Calibration",
        "SHAP & Feature Importance",
        "Partial Dependence",
        "Subgroup Analysis",
    ])

    # ── Classification Metrics ───────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
For classification tasks, a single metric rarely tells the full story. Different
metrics emphasize different aspects of performance, and the right choice depends on
the costs of different types of errors in your specific application.
""")

        section("The Confusion Matrix")
        st.markdown("""
All classification metrics derive from four counts:

- **True Positives (TP):** Correctly predicted positive cases.
- **True Negatives (TN):** Correctly predicted negative cases.
- **False Positives (FP):** Negative cases incorrectly predicted as positive (Type I error).
- **False Negatives (FN):** Positive cases incorrectly predicted as negative (Type II error).
""")

        section("Accuracy — and Why It Misleads")
        st.latex(r"""
        \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
        """)
        st.markdown("""
Accuracy is the proportion of correct predictions. It is intuitive but **dangerous
under class imbalance**. If 95% of patients are healthy and 5% have the disease,
a model that predicts "healthy" for everyone achieves 95% accuracy while being
completely useless for its intended purpose. This is why the app always includes
class-aware metrics alongside accuracy.
""")

        section("Precision and Recall")
        st.latex(r"""
        \text{Precision} = \frac{TP}{TP + FP} \qquad
        \text{Recall} = \frac{TP}{TP + FN}
        """)
        st.markdown("""
**Precision** answers: "Of the observations I predicted as positive, how many
actually are?" High precision means few false alarms.

**Recall** (sensitivity) answers: "Of the observations that actually are positive,
how many did I catch?" High recall means few missed cases.

These two metrics are in tension. Lowering the classification threshold catches more
true positives (higher recall) but also lets in more false positives (lower precision).
The right balance depends on the cost of each error type: a cancer screening test
prioritizes recall (don't miss a case); a spam filter prioritizes precision
(don't delete legitimate email).
""")

        section("F1 Score")
        st.latex(r"""
        F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
        = \frac{2 \, TP}{2 \, TP + FP + FN}
        """)
        st.markdown("""
The F1 score is the harmonic mean of precision and recall. It is highest when both
are high and penalizes extreme imbalance between them. The harmonic mean (rather
than arithmetic mean) ensures that a model with 100% precision but 1% recall gets
an F1 near 2%, not 50%.
""")

        section("AUROC — Area Under the ROC Curve")
        st.markdown(f"""
The ROC curve plots **True Positive Rate** (recall) against **False Positive Rate**
(FP / (FP + TN)) at every possible classification threshold. The area under this
curve (AUROC, or AUC) summarizes discrimination ability across all
thresholds: {cite("Hanley & McNeil, 1982")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \text{AUROC} = P(\hat{y}_{\text{positive}} > \hat{y}_{\text{negative}})
        """)
        st.markdown("""
The probabilistic interpretation is elegant: AUROC equals the probability that a
randomly chosen positive observation receives a higher predicted score than a
randomly chosen negative observation. AUROC = 0.5 is random guessing; AUROC = 1.0
is perfect discrimination.

**Limitation:** AUROC can be misleadingly high under severe class imbalance, because
it treats all thresholds equally — including thresholds that no one would use in
practice. AUPRC (see below) is more informative in imbalanced settings.
""")

        section("AUPRC — Area Under the Precision-Recall Curve")
        st.markdown("""
The precision-recall curve plots precision against recall at each threshold. Unlike
the ROC curve, it does not reward correctly classifying the abundant negative class,
making it more sensitive to performance on the minority class.

AUPRC is the preferred metric when:
- The positive class is rare (< 10% prevalence).
- The cost of missing a positive case is high.
- You want to evaluate how well the model *ranks* positive cases above negative ones,
  focusing on the positive-end of the prediction spectrum.
""")

        with st.expander("🧮 Interactive: See how the decision threshold trades precision for recall"):
            threshold = st.slider(
                "Classification threshold",
                min_value=0.05, max_value=0.95, value=0.50, step=0.05,
                key="theory_threshold",
                help="Predictions above this threshold are classified as positive.",
            )
            # Generate toy predictions: 200 obs, 30 positive
            rng_thr = np.random.default_rng(88)
            n_pos_t, n_neg_t = 30, 170
            scores_pos = rng_thr.beta(5, 2, n_pos_t)  # higher scores
            scores_neg = rng_thr.beta(2, 5, n_neg_t)  # lower scores
            scores_all = np.concatenate([scores_pos, scores_neg])
            labels_all = np.concatenate([np.ones(n_pos_t), np.zeros(n_neg_t)])

            preds = (scores_all >= threshold).astype(int)
            tp_t = int(np.sum((preds == 1) & (labels_all == 1)))
            fp_t = int(np.sum((preds == 1) & (labels_all == 0)))
            fn_t = int(np.sum((preds == 0) & (labels_all == 1)))
            tn_t = int(np.sum((preds == 0) & (labels_all == 0)))

            prec_t = tp_t / max(tp_t + fp_t, 1) * 100
            rec_t = tp_t / max(tp_t + fn_t, 1) * 100
            f1_t = 2 * (prec_t * rec_t) / max(prec_t + rec_t, 0.01)
            acc_t = (tp_t + tn_t) / (n_pos_t + n_neg_t) * 100

            fig_thr = go.Figure()
            fig_thr.add_trace(go.Bar(
                x=["Precision", "Recall", "F1", "Accuracy"],
                y=[prec_t, rec_t, f1_t, acc_t],
                marker_color=["#2563eb", "#dc2626", "#7c3aed", "#64748b"],
                text=[f"{v:.0f}%" for v in [prec_t, rec_t, f1_t, acc_t]],
                textposition="outside",
            ))
            fig_thr.update_layout(
                title=f"Threshold = {threshold:.2f} — TP={tp_t}, FP={fp_t}, FN={fn_t}, TN={tn_t}",
                yaxis_title="Metric (%)", yaxis_range=[0, 110], height=300,
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_thr, use_container_width=True)
            st.markdown(
                "**Train your eye:** Start at threshold = 0.50 and note the balance between precision and recall. "
                "Now slide left toward 0.10: recall (red) climbs toward 100% because you're catching almost everyone — "
                "but precision (blue) drops because you're also flagging many healthy people. "
                "Slide right toward 0.90: precision climbs (you only flag cases you're very sure about) "
                "but recall drops (you miss many true positives). "
                "**The lesson:** the threshold is a policy decision, not a statistical one. "
                "A cancer screening model wants low thresholds (catch everyone); a surgical decision model wants high thresholds (be sure)."
            )

        app_connection(
            "The <strong>Train & Compare</strong> page reports accuracy, precision, "
            "recall, F1, AUROC, and AUPRC for every trained classification model, with "
            "bootstrap 95% confidence intervals. The coaching layer warns when accuracy "
            "is high but AUROC is low (sign of class imbalance masking poor discrimination)."
        )

        references([
            "Hanley, J.A. & McNeil, B.J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36.",
            "Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.",
        ])

    # ── Regression Metrics ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
Regression metrics quantify how far the model's predictions deviate from the
true values. Different metrics penalize errors differently, and the right choice
depends on whether you care more about large errors, average errors, or explained
variance.
""")

        section("Mean Squared Error (MSE) and Root MSE")
        st.latex(r"""
        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \qquad
        \text{RMSE} = \sqrt{\text{MSE}}
        """)
        st.markdown("""
MSE is the average squared difference between true and predicted values. The
squaring means large errors are penalized quadratically — an error of 10 contributes
100 times as much as an error of 1. RMSE (the square root of MSE) returns the
metric to the original units of *y*, making it interpretable: an RMSE of 3.2
means predictions are off by about 3.2 units on average, with large errors
weighted more heavily.
""")

        section("Mean Absolute Error (MAE)")
        st.latex(r"""
        \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
        """)
        st.markdown("""
MAE is the average *absolute* difference. Unlike MSE, it penalizes all errors
linearly — an error of 10 contributes exactly 10 times as much as an error of 1.
MAE is more robust to outliers: a single observation with a huge residual won't
dominate the metric.

**When to prefer MAE over RMSE:** When large errors are not disproportionately
more costly than small ones, or when the data has outliers in the response variable.
""")

        section("R² — Coefficient of Determination")
        st.latex(r"""
        R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
        = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
        """)
        st.markdown("""
R² compares the model's residual sum of squares (SS_res) to the total sum of
squares (SS_tot, the variance in *y*). It represents the fraction of variance
in the response that the model explains.

- R² = 1: perfect predictions (SS_res = 0).
- R² = 0: the model is no better than predicting the mean of *y*.
- R² < 0: the model is *worse* than predicting the mean — possible on test data
  when the model overfits or when applied to a different distribution.

**On test data, R² can be negative.** This is not a bug — it means the model's
predictions are less accurate than simply predicting the training mean for every
observation. It typically signals overfitting or distribution shift.
""")

        app_connection(
            "The <strong>Train & Compare</strong> page reports RMSE, MAE, and R² for "
            "every regression model, with bootstrap 95% confidence intervals. The "
            "coaching layer flags models where R² is negative on the test set as "
            "severely overfitting."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §3.1.3. Springer.",
        ])

    # ── Cross-Validation ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
A single train/test split gives one estimate of model performance. Cross-validation
gives *k* estimates from the same data, producing both a more reliable point estimate
and a measure of variability. {cite("ISLR, §5.1")}
""", unsafe_allow_html=True)

        section("k-Fold Cross-Validation")
        st.markdown("""
The data is randomly divided into *k* equal-sized folds (typically k = 5 or 10).
The model is trained *k* times, each time using *k − 1* folds for training and the
held-out fold for evaluation. The final metric is the average across all *k* folds:
""")
        st.latex(r"""
        \text{CV}_{(k)} = \frac{1}{k} \sum_{i=1}^{k} \text{Metric}_i
        """)
        st.markdown("""
**Why k = 5 or 10?** This is an empirical compromise:
- **k = n** (leave-one-out CV) has low bias (each training set is nearly the full
  data) but high variance (the training sets are almost identical, so the estimates
  are highly correlated).
- **k = 2** has high bias (each training set is only half the data) but low variance.
- **k = 5 or 10** strikes a balance, and empirical studies show it gives the best
  tradeoff between bias and variance of the performance estimate.
""")

        with st.expander("🧮 Interactive: See fold-to-fold variation in cross-validation"):
            cv_k = st.slider("Number of folds (k)", min_value=2, max_value=20, value=5, step=1, key="theory_cv_k",
                             help="k = 2 uses half the data for training; k = n is leave-one-out.")
            cv_noise = st.slider("Dataset noise level", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="theory_cv_noise",
                                 help="Higher noise → weaker signal → more fold-to-fold variability.")
            rng_cv = np.random.default_rng(21)
            n_cv = 120
            X_cv = rng_cv.standard_normal((n_cv, 3))
            y_cv = 1.5 * X_cv[:, 0] - 0.8 * X_cv[:, 1] + rng_cv.normal(0, cv_noise, n_cv)

            # Simulate k-fold CV with simple linear regression R²
            fold_ids = np.arange(n_cv) % cv_k
            rng_cv.shuffle(fold_ids)
            fold_scores = []
            for f in range(cv_k):
                mask = fold_ids == f
                X_tr, y_tr = X_cv[~mask], y_cv[~mask]
                X_te, y_te = X_cv[mask], y_cv[mask]
                # OLS fit
                X_tr_aug = np.column_stack([np.ones(X_tr.shape[0]), X_tr])
                X_te_aug = np.column_stack([np.ones(X_te.shape[0]), X_te])
                beta_cv = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)[0]
                y_pred = X_te_aug @ beta_cv
                ss_res = np.sum((y_te - y_pred) ** 2)
                ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                fold_scores.append(r2)

            fig_cv = go.Figure()
            colors = ["rgba(99, 102, 241, 0.7)"] * cv_k
            fig_cv.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(cv_k)], y=fold_scores, marker_color=colors))
            fig_cv.add_hline(y=np.mean(fold_scores), line_dash="dash", line_color="#dc2626",
                             annotation_text=f"Mean R² = {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
            fig_cv.update_layout(
                title=f"{cv_k}-fold CV: R² per fold",
                yaxis_title="R²", height=300,
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            st.markdown(
                "**Train your eye:** At low noise, the bars are nearly uniform — the model works consistently regardless of which fold is held out. "
                "Now crank the noise to 4 or 5: some folds score well, others score poorly or even go negative. "
                "That spread *is* the uncertainty in your performance estimate. "
                "A mean R² of 0.40 ± 0.25 tells a very different scientific story than 0.40 ± 0.03. "
                "**In your own results:** if the red dashed line (mean) looks good but individual bars scatter wildly, the result is fragile."
            )

        misconception(
            "Cross-validation does not give you five independent proofs that the model works. It gives repeated estimates of how the same modeling procedure behaves under different held-out partitions of the same dataset."
        )

        self_check(
            "If one model has mean AUROC 0.84 ± 0.09 and another has 0.82 ± 0.02, which one would you feel more comfortable writing up as stable — and why?"
        )

        section("Stratified k-Fold")
        st.markdown("""
For classification, standard k-fold may produce folds with different class
proportions, especially when the minority class is small. **Stratified k-fold**
ensures each fold preserves the overall class distribution. This reduces the
variance of the CV estimate and prevents folds where the minority class is
absent or overrepresented.
""")

        section("What Cross-Validation Estimates")
        st.markdown("""
An important subtlety: cross-validation estimates the performance of the
**modeling procedure** (the combination of preprocessing + model + hyperparameters),
not of the specific model fit on the full training set. The model that will
ultimately be deployed is trained on *all* the data, and its performance may be
slightly better than the CV estimate (because it sees more data).

Cross-validation is used for:
- **Model comparison:** Which algorithm performs best on this data?
- **Hyperparameter tuning:** Which settings of λ, k, or depth give the best performance?
- **Estimating generalization error:** How well will this approach work on new data?
""")

        section("Hyperparameter Tuning")
        st.markdown("""
Most models have **hyperparameters** — settings that are not learned from the data
but must be chosen before training. Examples include the regularization strength λ
in Ridge regression, the number of neighbors *k* in KNN, the learning rate and
number of hidden layers in a neural network, and the maximum tree depth in a
random forest.

**Why hyperparameters can't be tuned on training performance:** A model with more
flexibility (deeper trees, more neurons, weaker regularization) will always fit the
training data better. Tuning on training performance selects the most overfit model.
Cross-validation provides an honest estimate of out-of-sample performance for each
setting.

**Bayesian optimization (Optuna):** Rather than exhaustively trying all combinations
(grid search) or randomly sampling (random search), Bayesian optimization builds a
probabilistic model of the relationship between hyperparameter settings and
performance, then intelligently chooses the next settings to try based on where
improvement is most likely. This finds good hyperparameters with far fewer trials
than grid or random search.
""")

        app_connection(
            "The <strong>Train & Compare</strong> page runs stratified k-fold CV (default "
            "k = 5) for every trained model and reports mean ± standard deviation of the "
            "primary metric across folds. Hyperparameter tuning via Optuna (Bayesian "
            "optimization) is available for models with tunable parameters, using CV "
            "performance as the objective."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §5.1. Springer.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §7.10. Springer.",
        ])

    # ── Calibration ──────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
A model can *discriminate* well (rank positive cases above negative ones) while
being poorly *calibrated* (the predicted probabilities don't match the actual
frequencies). Calibration asks: when the model says "70% chance of positive,"
is it actually positive about 70% of the time?
""")

        section("Reliability Diagrams")
        st.markdown("""
A reliability diagram (calibration plot) bins predicted probabilities into groups
(e.g., 0–10%, 10–20%, ..., 90–100%) and plots the average predicted probability
against the actual positive rate in each bin. A perfectly calibrated model falls
on the diagonal (predicted 70% = observed 70%).

- **Above the diagonal:** The model is *underconfident* — it predicts lower probabilities
  than the actual positive rate.
- **Below the diagonal:** The model is *overconfident* — it predicts higher probabilities
  than the actual positive rate.
""")

        section("Brier Score")
        st.latex(r"""
        \text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2
        """)
        st.markdown("""
The Brier score is the mean squared error of probability predictions, where *p̂_i*
is the predicted probability and *y_i* is the binary outcome (0 or 1). Lower is
better. A Brier score of 0 means perfect probabilistic predictions; the maximum
useful reference is the Brier score of a model that always predicts the base rate.

The Brier score captures both discrimination *and* calibration: a well-discriminating
but poorly calibrated model will still have a mediocre Brier score.
""")

        section("Expected Calibration Error (ECE)")
        st.latex(r"""
        \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \bar{p}_m - \bar{y}_m \right|
        """)
        st.markdown("""
ECE divides predictions into *M* bins (e.g., 10 bins: 0–10%, 10–20%, ..., 90–100%).
Within each bin, it compares the average predicted probability (*p̄_m*) to the actual
positive rate (*ȳ_m*). For example, if observations in the "70–80%" bin have an
average prediction of 0.75 but only 0.60 are actually positive, the bin's calibration
error is |0.75 − 0.60| = 0.15. The weight |B_m|/n is the proportion of observations
in bin *m*, so bins with more observations contribute more. ECE = 0 means perfect
calibration; values above 0.05 typically warrant investigation.
""")

        with st.expander("🧮 Interactive: See what miscalibration looks like"):
            miscal = st.slider(
                "Miscalibration direction and strength",
                min_value=-0.3, max_value=0.3, value=0.15, step=0.05,
                key="theory_calibration_miscal",
                help="Positive = overconfident (predicts higher than reality). Negative = underconfident. Zero = perfect.",
            )
            # Generate 10 bins of predicted probabilities
            bin_centers = np.linspace(0.05, 0.95, 10)
            # Actual rates: if miscal > 0, model is overconfident (predicted > actual)
            actual_rates = np.clip(bin_centers - miscal * np.sin(np.pi * bin_centers), 0.01, 0.99)

            fig_cal = go.Figure()
            # Perfect calibration diagonal
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"),
                name="Perfect calibration", showlegend=True,
            ))
            # Model calibration curve
            fig_cal.add_trace(go.Scatter(
                x=bin_centers, y=actual_rates, mode="lines+markers",
                line=dict(color="#dc2626", width=2.5), marker=dict(size=8),
                name="Model calibration",
            ))
            # Shade the gap
            fig_cal.add_trace(go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([actual_rates, bin_centers[::-1]]),
                fill="toself", fillcolor="rgba(220, 38, 38, 0.1)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            ece_val = np.mean(np.abs(bin_centers - actual_rates))
            fig_cal.update_layout(
                title=f"Reliability Diagram — ECE = {ece_val:.3f}",
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed positive rate",
                xaxis_range=[0, 1], yaxis_range=[0, 1],
                height=340, margin=dict(t=50, b=40, l=50, r=20),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.markdown(
                "**Train your eye:** The dashed diagonal is the honesty line — where every probability statement is exactly right. "
                "At positive miscalibration, the red curve bows **below** it: the model says '80% risk' but reality is closer to 65%. "
                "Notice the shaded gap is widest in the middle bins — miscalibration often concentrates where the model is most 'confident.' "
                "Slide to 0 and watch the curve snap onto the diagonal. Slide negative to see the opposite failure: underconfidence. "
                "**In your own results:** if the calibration curve in Train & Compare bows away from the diagonal, the predicted probabilities need recalibration before clinical use."
            )

        misconception(
            "A high AUROC does not guarantee well-calibrated probabilities. A model can rank cases correctly while still reporting probabilities that are systematically too high or too low."
        )

        self_check(
            "If two models have similar AUROC but one has a much better calibration curve, which one would you trust more for risk communication or decision thresholds — and why?"
        )

        section("Calibration for Regression")
        st.markdown("""
For regression models, calibration takes a different form: a calibration slope and
intercept are computed by regressing the true values on the predicted values. A
perfectly calibrated model has slope = 1 and intercept = 0. A slope < 1 means the
model's predictions are too spread out (overconfident in their variation); a
slope > 1 means they are too compressed.
""")

        app_connection(
            "The <strong>Train & Compare</strong> page generates reliability diagrams, "
            "Brier scores, and ECE for classification models, and calibration slopes "
            "for regression models. These appear alongside discrimination metrics to "
            "give a complete picture of model performance."
        )

        references([
            "Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.",
            "Steyerberg, E.W. (2019). *Clinical Prediction Models* (2nd ed.), Ch. 15. Springer.",
        ])

    # ── SHAP & Feature Importance ────────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"""
Understanding *which features drive predictions* is essential for scientific
credibility and clinical trust. Two complementary approaches are available:
permutation importance (model-agnostic) and SHAP values (based on cooperative
game theory). {cite("Lundberg & Lee, 2017")}
""", unsafe_allow_html=True)

        section("Permutation Importance")
        st.markdown(f"""
Permutation importance measures how much the model's performance degrades when
a single feature's values are randomly shuffled, breaking its relationship with
the target. {cite("Breiman, 2001")}
""", unsafe_allow_html=True)
        st.latex(r"""
        \text{PI}_j = \text{Metric}_{\text{original}} - \frac{1}{R}\sum_{r=1}^{R} \text{Metric}_{\text{permuted}}^{(r)}
        """)
        st.markdown("""
Here the metric is computed on the *test set* (not training set — computing on the
training set can inflate importance of overfit features). The feature is shuffled
*R* times and the degradation is averaged. If shuffling feature *j* barely changes
performance, it's unimportant. If performance drops substantially, the model relied
heavily on that feature.

**Advantage:** Completely model-agnostic — works for any model. Correctly measures
importance for *prediction*, not just association.

**Limitation:** Correlated features share importance. If features A and B are
highly correlated, shuffling A leaves B intact, so the model can partially
compensate. Both features may appear less important than they truly are.
""")

        section("SHAP Values")
        st.markdown(f"""
SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each
individual prediction, based on Shapley values from cooperative game
theory. {cite("Lundberg & Lee, 2017")}

The Shapley value for feature *j* in prediction *i* is the average marginal
contribution of that feature across all possible orderings of features:
""", unsafe_allow_html=True)
        st.latex(r"""
        \phi_j = \sum_{S \subseteq \{1,\ldots,p\} \setminus \{j\}}
        \frac{|S|! \; (p - |S| - 1)!}{p!}
        \left[ f(S \cup \{j\}) - f(S) \right]
        """)
        st.markdown("""
This formula considers every possible subset *S* of features that *doesn't* include
feature *j*. For each subset, it computes the **marginal contribution** — how much
the prediction changes when feature *j* is added: f(S ∪ {j}) − f(S). If the model
already has features A, B, C (set *S*) and predicts 0.7, and adding feature *j*
changes the prediction to 0.8, the marginal contribution is 0.1 for that subset.

The weighting factor |S|!(p − |S| − 1)!/p! ensures that each *ordering* of features
counts equally. There are p! total orderings; |S|!(p − |S| − 1)! of them place
exactly the features in *S* before feature *j*. The Shapley value is the average
marginal contribution across all orderings — ensuring that the result is fair
regardless of the order in which features are "added" to the model.

The key properties that make SHAP values uniquely principled:
- **Local accuracy:** The SHAP values for a single prediction sum to the difference
  between the prediction and the average prediction.
- **Consistency:** If a feature's contribution increases in a modified model, its
  SHAP value cannot decrease.
- **Missingness:** A feature that has no effect receives a SHAP value of zero.

Different SHAP algorithms exploit model structure for speed:
- **TreeSHAP** computes exact Shapley values for tree models in polynomial time.
- **LinearSHAP** uses the closed-form solution for linear models.
- **KernelSHAP** is model-agnostic but slower (used for neural networks, SVMs, KNN).
""")

        with st.expander("🧮 Interactive: How SHAP values decompose a single prediction"):
            st.markdown("Adjust feature values to see how each one pushes the prediction up or down from the baseline.")
            shap_glucose = st.slider("Glucose level", 70, 200, 140, key="theory_shap_glucose")
            shap_bmi = st.slider("BMI", 18.0, 45.0, 28.0, step=0.5, key="theory_shap_bmi")
            shap_age = st.slider("Age", 20, 80, 50, key="theory_shap_age")

            # Toy linear attribution (simulates SHAP for educational purposes)
            baseline = 0.30
            contrib_glucose = (shap_glucose - 120) * 0.002
            contrib_bmi = (shap_bmi - 25) * 0.008
            contrib_age = (shap_age - 45) * 0.003
            prediction = baseline + contrib_glucose + contrib_bmi + contrib_age
            prediction = np.clip(prediction, 0.01, 0.99)

            features = ["Glucose", "BMI", "Age"]
            contributions = [contrib_glucose, contrib_bmi, contrib_age]
            colors_shap = ["#dc2626" if c > 0 else "#2563eb" for c in contributions]

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=features, x=contributions, orientation="h",
                marker_color=colors_shap,
                text=[f"{c:+.3f}" for c in contributions], textposition="outside",
            ))
            fig_shap.add_vline(x=0, line_color="#94a3b8")
            fig_shap.update_layout(
                title=f"Baseline: {baseline:.2f} → Prediction: {prediction:.3f}  (sum of contributions: {sum(contributions):+.3f})",
                xaxis_title="SHAP contribution to prediction",
                height=260, margin=dict(t=50, b=30, l=80, r=60), template="plotly_white",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown(
                "**Train your eye:** Move the glucose slider to 180 and watch its red bar extend right — it's pushing the prediction up. "
                "Now drop BMI to 20 and watch its bar turn blue and extend left — it's pulling the prediction down. "
                "Check the title: the contributions always sum exactly to the gap between baseline and prediction. "
                "That's the SHAP guarantee. **In your own results:** if a feature's SHAP bar is consistently large and red across many patients, "
                "the model is relying heavily on it — but that tells you about the model's behavior, not about causation."
            )

        misconception(
            "SHAP explains how the model used a feature, not whether the feature causes the outcome. Strong SHAP importance can reflect correlation, proxy effects, or dataset artifacts."
        )

        self_check(
            "If glucose has the largest SHAP values in a diabetes model, what can you conclude confidently — and what causal claim must you still avoid making?"
        )

        app_connection(
            "The <strong>Explainability</strong> page computes both permutation importance "
            "and SHAP values for every trained model. It generates SHAP summary plots "
            "(beeswarm), SHAP dependence plots, and global feature importance rankings. "
            "The method used (TreeSHAP, LinearSHAP, or KernelSHAP) is automatically "
            "selected based on the model type."
        )

        references([
            "Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.",
            "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.",
            "Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.), Ch. 9. christophm.github.io/interpretable-ml-book.",
        ])

    # ── Partial Dependence ───────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(f"""
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots
show *how* a feature affects predictions — not just *whether* it
matters. {cite("Friedman, 2001")}
""", unsafe_allow_html=True)

        section("Partial Dependence Plots (PDP)")
        st.markdown("""
A PDP shows the average prediction as a function of one (or two) features,
marginalizing over all other features:
""")
        st.latex(r"""
        \hat{f}_j(x_j) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_j, \mathbf{x}_{i,-j})
        """)
        st.markdown("""
For each value of feature *x_j*, the PDP computes predictions for every training
observation (substituting *x_j* while keeping all other features at their observed
values) and averages them. The resulting curve shows the *marginal effect* of
feature *j* on the prediction.

**Reading a PDP:** A flat line means the feature has no effect. A monotone increasing
line means higher values of the feature produce higher predictions. A U-shaped
curve means extreme values (high or low) both increase predictions. The *steepness*
of the curve indicates the *strength* of the effect.
""")

        section("Individual Conditional Expectation (ICE)")
        st.markdown("""
ICE plots show the same information as PDP, but for *each individual observation*
rather than the average. Each line traces how one observation's prediction changes
as the feature varies.

ICE plots reveal **interaction effects** that PDP hides. If the PDP is flat but
the ICE lines go in opposite directions, the feature has a strong effect — but
it works differently for different subgroups of the data. The average (PDP) washes
out the opposing effects, creating a false impression of unimportance.
""")

        with st.expander("🧮 Interactive: How PDP averaging hides interaction effects"):
            interaction_strength = st.slider(
                "Interaction strength (0 = no interaction, 1 = strong interaction)",
                min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                key="theory_pdp_interaction",
            )
            rng_pdp = np.random.default_rng(55)
            n_pdp = 100
            # x1 is the feature of interest; x2 is a binary grouping variable
            x1 = np.linspace(-3, 3, 50)
            group = rng_pdp.choice([0, 1], size=n_pdp)

            # For group 0: x1 has positive effect; group 1: negative effect (interaction)
            ice_lines = []
            for i in range(n_pdp):
                if group[i] == 0:
                    effect = (1 - interaction_strength * 0.5) + interaction_strength * 1.0
                    ice = effect * x1 + rng_pdp.normal(0, 0.3)
                else:
                    effect = (1 - interaction_strength * 0.5) - interaction_strength * 1.0
                    ice = effect * x1 + rng_pdp.normal(0, 0.3)
                ice_lines.append(ice)

            ice_array = np.array(ice_lines)
            pdp_mean = np.mean(ice_array, axis=0)

            fig_pdp = go.Figure()
            # ICE lines (thin, semi-transparent)
            for i in range(min(n_pdp, 40)):
                color = "rgba(220, 38, 38, 0.15)" if group[i] == 0 else "rgba(37, 99, 235, 0.15)"
                fig_pdp.add_trace(go.Scatter(
                    x=x1, y=ice_lines[i], mode="lines", line=dict(width=1, color=color),
                    showlegend=False,
                ))
            # PDP (thick black)
            fig_pdp.add_trace(go.Scatter(
                x=x1, y=pdp_mean, mode="lines", line=dict(width=3, color="#111827"),
                name="PDP (average)",
            ))
            fig_pdp.update_layout(
                title="PDP (black) vs ICE lines (red = group A, blue = group B)",
                xaxis_title="Feature value (x₁)", yaxis_title="Predicted outcome",
                height=340, margin=dict(t=50, b=40, l=50, r=20), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_pdp, use_container_width=True)
            st.markdown(
                "**Train your eye:** At high interaction strength, the red and blue ICE lines slope in **opposite directions** — "
                "the feature matters a lot, but differently for the two groups. Now look at the thick black PDP: it's nearly flat. "
                "That flat line is a lie — the average of opposing effects cancels to zero. "
                "Slide interaction to 0: now all ICE lines agree and the PDP faithfully represents the real effect. "
                "**In your own results:** whenever a PDP looks flat for a feature you expected to matter, overlay ICE lines before concluding it's unimportant."
            )

        misconception(
            "A flat PDP does not prove a feature is irrelevant. It may indicate that the feature's effect depends on other variables — the average hides the interaction."
        )

        self_check(
            "If the PDP for a feature is flat but ICE lines diverge sharply, what does that tell you about the feature's relationship with the outcome — and what should you investigate next?"
        )

        app_connection(
            "The <strong>Explainability</strong> page generates PDP for the top features "
            "identified by permutation importance. ICE plots are available as an "
            "overlay on the PDP. These plots complement SHAP values: SHAP tells you "
            "*how much* a feature matters; PDP tells you *how* it matters."
        )

        references([
            "Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.",
            "Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44–65.",
        ])

    # ── Subgroup Analysis ────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("""
A model that performs well *on average* may perform poorly for specific subgroups
of the population. Subgroup analysis stratifies performance by a categorical
variable (e.g., sex, age group, site) to detect disparities that aggregate metrics
hide.
""")

        section("Why Subgroup Analysis Matters")
        st.markdown("""
Consider a model with overall AUROC of 0.88. This looks strong — but what if
AUROC = 0.92 for males and AUROC = 0.71 for females? The aggregate metric conceals
a serious performance disparity that has implications for both clinical utility
and fairness.

Subgroup analysis is increasingly expected by reviewers, especially for:
- **Clinical prediction models** — must demonstrate equitable performance across
  demographics (sex, race, age groups).
- **Multi-site studies** — performance may vary by site due to different patient
  populations, measurement protocols, or data quality.
- **Rare subgroups** — the model may have learned patterns for common groups
  while effectively ignoring rare ones.
""")

        section("Forest Plots for Subgroup Comparison")
        st.markdown("""
A **forest plot** displays the primary metric (with confidence intervals) for each
subgroup as horizontal bars, with a vertical reference line at the overall metric
value. This makes it immediately visible which subgroups fall below the overall
performance and whether the confidence intervals are wide (few observations in
that subgroup) or narrow.

Reading a forest plot:
- Subgroups whose CI includes the overall reference line are consistent with
  overall performance.
- Subgroups whose CI lies entirely below the reference line have significantly
  worse performance.
- Wide CIs indicate small subgroup size — the estimate is unreliable.
""")

        section("Limitations")
        st.markdown("""
Subgroup analysis is exploratory, not confirmatory. With many subgroups, some will
show poor performance by chance alone (the multiple testing problem applies here
too). Findings from subgroup analysis should be treated as hypotheses to
investigate further, not as definitive conclusions.

Additionally, subgroup analysis requires sufficient observations per subgroup.
A subgroup with n = 10 will have such wide confidence intervals that the
performance estimate is nearly meaningless.
""")

        with st.expander("🧮 Interactive: See how aggregate metrics hide subgroup disparities"):
            disparity = st.slider(
                "Performance disparity between subgroups",
                min_value=0.0, max_value=0.25, value=0.12, step=0.02,
                key="theory_subgroup_disparity",
            )
            subgroups = ["Males (n=120)", "Females (n=110)", "Age < 50 (n=80)", "Age ≥ 50 (n=150)", "Site A (n=90)", "Site B (n=70)"]
            # Base performance with disparity applied to some subgroups
            base_perf = 0.85
            rng_sg = np.random.default_rng(33)
            offsets = np.array([disparity * 0.5, -disparity, disparity * 0.3, -disparity * 0.4, 0.0, -disparity * 0.7])
            perfs = base_perf + offsets
            ci_widths = np.array([0.04, 0.05, 0.06, 0.035, 0.05, 0.07])  # wider for smaller groups

            fig_sg = go.Figure()
            colors_sg = ["#16a34a" if p >= base_perf - 0.01 else "#dc2626" for p in perfs]
            for i, (sg, p, ci_w) in enumerate(zip(subgroups, perfs, ci_widths)):
                fig_sg.add_trace(go.Scatter(
                    x=[p], y=[sg], mode="markers",
                    marker=dict(size=10, color=colors_sg[i]),
                    error_x=dict(type="constant", value=ci_w, color=colors_sg[i]),
                    showlegend=False,
                ))
            fig_sg.add_vline(x=base_perf, line_dash="dash", line_color="#6366f1",
                             annotation_text=f"Overall: {base_perf:.2f}")
            fig_sg.update_layout(
                title=f"Subgroup performance (forest plot) — overall AUROC = {base_perf:.2f}",
                xaxis_title="AUROC", xaxis_range=[0.5, 1.0],
                height=320, margin=dict(t=50, b=40, l=140, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_sg, use_container_width=True)
            st.markdown(
                "**Train your eye:** Look at the purple dashed line — that's the aggregate AUROC the paper would report. "
                "Now look at the red dots: some subgroups fall well below that line. Increase the disparity slider and watch 'Females' and 'Site B' "
                "drop further from the reference. Notice that Site B also has a **wide CI** — that's not reassurance, it's ignorance (small n). "
                "**In your own results:** if any subgroup's CI lies entirely below the overall line, you have a disparity worth disclosing. "
                "If the CI is wide, you have a subgroup you can't yet evaluate — which is also worth disclosing."
            )

        misconception(
            "Finding that one subgroup performs worse does not necessarily mean the model is biased. It may reflect genuinely different data quality, different signal structure, or insufficient representation in the training set."
        )

        self_check(
            "If a subgroup has only 15 observations and its CI is very wide, should you report that the model 'fails' for that group — or is the honest conclusion that you simply don't have enough evidence?"
        )

        app_connection(
            "The <strong>Explainability</strong> page runs subgroup analysis by stratifying "
            "test set performance across any categorical variable with ≤ 10 unique values. "
            "It generates a forest plot showing metric point estimates and 95% CIs per "
            "subgroup, making performance disparities visually obvious."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 6: Statistical Testing
# ══════════════════════════════════════════════════════════════════════════════

def render_statistical_testing():
    intro(
        "Statistical tests give your results formal rigor. But a test is only meaningful "
        "when its assumptions are met and its results are correctly interpreted. "
        "This chapter covers the tests available in the <strong>Hypothesis Testing</strong> "
        "page and the theory that governs when each is appropriate."
    )

    tabs = st.tabs([
        "Hypothesis Testing Fundamentals",
        "Correlation Tests",
        "Group Comparisons",
        "Categorical Association",
        "Normality Testing",
        "Paired Comparisons",
        "Table 1",
    ])

    # ── Fundamentals ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
A hypothesis test is a formal framework for making decisions about a population
based on sample data. The logic is indirect: rather than proving what *is* true,
you assume the opposite and check whether the data is consistent with that
assumption.
""")

        section("The Null and Alternative Hypotheses")
        st.markdown("""
- **Null hypothesis (H₀):** The default position — typically "no effect," "no
  difference," or "no association." For example: "the mean blood pressure is the
  same in the treatment and control groups."
- **Alternative hypothesis (H₁):** What you're trying to find evidence for —
  "there *is* a difference."

The test asks: *if H₀ were true, how surprising would this data be?*
""")

        section("P-Values: What They Mean (and Don't)")
        st.latex(r"""
        p = P(\text{data as extreme as observed} \mid H_0 \text{ is true})
        """)
        st.markdown("""
The p-value is the probability of obtaining results at least as extreme as the
observed data, *assuming the null hypothesis is true*. It is **not**:

- The probability that H₀ is true. (That would require Bayesian reasoning and a prior.)
- The probability that the result is due to chance. (The result is always what it is.)
- The probability of making a mistake. (That depends on the base rate of true effects.)

A small p-value (typically < 0.05) means the data would be very unlikely *if* H₀
were true, which is taken as evidence against H₀. But the threshold of 0.05 is a
convention, not a physical constant — and a p-value of 0.049 is not meaningfully
different from 0.051.
""")

        section("Type I and Type II Errors")
        st.markdown("""
- **Type I error (false positive):** Rejecting H₀ when it's actually true. The
  probability of this error is α (the significance level, typically 0.05).
- **Type II error (false negative):** Failing to reject H₀ when it's actually false.
  The probability of this is β. **Power** = 1 − β is the probability of correctly
  detecting a true effect.

These are in tension: reducing α (being more conservative) increases β (you miss
more real effects). The only way to reduce both simultaneously is to increase
sample size.
""")

        section("Multiple Testing Correction")
        st.markdown(f"""
When you run many tests, the chance of at least one false positive increases rapidly.
With 20 independent tests at α = 0.05, the probability of at least one false positive
is 1 − (1 − 0.05)²⁰ ≈ 64%.

Two common corrections: {cite("Benjamini & Hochberg, 1995")}

**Bonferroni correction:** Divide α by the number of tests (α/m). Simple and
conservative — it controls the Family-Wise Error Rate (FWER, the probability of
*any* false positive). Often too conservative when m is large.

**Benjamini-Hochberg (FDR):** Controls the expected *proportion* of false positives
among rejected tests. Less conservative than Bonferroni and more appropriate when
running many tests (as in feature selection).
""", unsafe_allow_html=True)

        takeaway(
            "A statistically significant result is not necessarily clinically meaningful. "
            "Always report effect sizes (how big is the difference?) alongside p-values "
            "(how confident are we that it's not zero?)."
        )

        app_connection(
            "The <strong>Hypothesis Testing</strong> page applies these fundamentals "
            "throughout: every test reports a p-value alongside effect sizes, and the "
            "EDA page uses Benjamini-Hochberg FDR correction when running multiple "
            "univariate tests during feature selection."
        )

        references([
            "Wasserstein, R.L. & Lazar, N.A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129–133.",
            "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society, Series B*, 57(1), 289–300.",
        ])

    # ── Correlation Tests ────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
Correlation tests quantify the strength and direction of the association between
two continuous variables. Three methods are available, each with different
assumptions and appropriate use cases.
""")

        section("Pearson Correlation")
        st.latex(r"""
        r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
        {\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
        """)
        st.markdown("""
The numerator is the **covariance** — the sum of products of deviations from each
variable's mean. When both variables tend to be above their means together (or
below together), the products are positive and *r* is positive. When one tends to
be above while the other is below, the products are negative and *r* is negative.
The denominator normalizes by the standard deviations of both variables, ensuring
*r* always falls between −1 and +1 regardless of the units of measurement.

Pearson's *r* captures only *linear* relationships. It ranges from −1 (perfect
negative linear relationship) through 0 (no linear relationship) to +1 (perfect
positive linear relationship).

**Assumptions:** Both variables are approximately normally distributed, the
relationship is linear, and there are no extreme outliers. If the relationship
is curved (e.g., U-shaped), Pearson's *r* may be near zero even though there is a
strong association.
""")

        section("Spearman Rank Correlation")
        st.markdown("""
Spearman's ρ is the Pearson correlation computed on the *ranks* of the data rather
than the raw values. It measures the strength of any *monotone* relationship (not
just linear). It is robust to outliers and does not require normality.

Use Spearman when:
- The relationship might be monotone but not linear.
- The data is ordinal (ranked) rather than interval.
- There are outliers or skewed distributions.
""")

        section("Kendall's Tau")
        st.markdown("""
Kendall's τ counts the number of *concordant* and *discordant* pairs of observations.
A pair (xᵢ, yᵢ) and (xⱼ, yⱼ) is concordant if xᵢ < xⱼ and yᵢ < yⱼ (or both
greater), and discordant if they disagree. Kendall's τ is more robust than Spearman
for small samples and has better statistical properties, but its values tend to be
smaller in magnitude.
""")

        app_connection(
            "The <strong>Hypothesis Testing</strong> page runs the selected correlation "
            "test and reports the coefficient, p-value, and effect size (r²). Results "
            "can be exported directly to Table 1."
        )

    # ── Group Comparisons ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
Group comparison tests determine whether the distribution of a numeric variable
differs between groups. The choice between parametric and non-parametric tests
depends on whether the normality assumption holds.
""")

        section("Two-Sample Comparisons")
        st.markdown("""
**Student's t-test** compares the means of two independent groups:
""")
        st.latex(r"""
        t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
        """)
        st.markdown("""
The numerator (x̄₁ − x̄₂) is the observed difference in group means — the "signal."
The denominator is the **standard error** of that difference, combining the variance
within each group (s₁² and s₂²) and the sample sizes (n₁ and n₂). The ratio
tells you how many standard errors the observed difference is from zero. A large
|t| means the difference is unlikely to have arisen by chance if the true means
were equal. Under H₀ (equal means), *t* follows a t-distribution with degrees of
freedom estimated by the Welch-Satterthwaite approximation.

**Assumptions:** Independent observations, approximately normal distributions within
each group (relaxed for large n by the CLT), and continuous outcome.

**Mann-Whitney U test** is the non-parametric alternative. Instead of comparing
means, it tests whether one group tends to have *larger values* than the other.
It works on the ranks rather than raw values and makes no distributional assumptions.
Use it when the data is skewed, ordinal, or has outliers.
""")

        section("Multi-Group Comparisons")
        st.markdown("""
**One-way ANOVA** extends the t-test to three or more groups by comparing the
between-group variance to the within-group variance:
""")
        st.latex(r"""
        F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}
        = \frac{\sum_{k} n_k (\bar{x}_k - \bar{x})^2 / (K - 1)}
        {\sum_{k}\sum_{i} (x_{ik} - \bar{x}_k)^2 / (n - K)}
        """)
        st.markdown("""
The numerator (MS_between) measures how much the group means (x̄_k) deviate from
the overall mean (x̄), weighted by group size n_k and divided by K − 1 (the number
of groups minus one). The denominator (MS_within) measures how much individual
observations deviate from their own group's mean, divided by n − K. The F-statistic
is their ratio: if the groups truly have the same mean, both numerator and
denominator estimate the same variance, and F ≈ 1. A large F means the group
means are more spread out than you'd expect from the within-group noise alone.

A significant result tells you that *at least one* group differs — but not *which*
groups differ (that requires post-hoc tests like Tukey's HSD).

**Kruskal-Wallis test** is the non-parametric alternative to ANOVA, based on ranks.
Like Mann-Whitney, it makes no distributional assumptions.
""")

        app_connection(
            "The <strong>Hypothesis Testing</strong> page offers t-test / Mann-Whitney "
            "for two groups and ANOVA / Kruskal-Wallis for multiple groups. The user "
            "selects the grouping variable and outcome, and can toggle between "
            "parametric and non-parametric tests. Results include the test statistic, "
            "p-value, and effect size."
        )

    # ── Categorical Association ─────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
When both variables are categorical (e.g., treatment group vs. outcome category),
correlation and t-tests are inappropriate. Instead, we test whether the distribution
of one variable depends on the level of the other.
""")

        section("Chi-Squared Test of Independence")
        st.markdown("""
The chi-squared test compares the *observed* counts in a contingency table to the
counts you would *expect* if the two variables were independent:
""")
        st.latex(r"""
        \chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
        \quad \text{where} \quad E_{ij} = \frac{(\text{row } i \text{ total}) \times (\text{col } j \text{ total})}{n}
        """)
        st.markdown("""
Each cell in the contingency table contributes to the statistic: the squared
difference between observed count *O_ij* and expected count *E_ij*, divided by the
expected count. If the variables are truly independent, observed and expected counts
should be close, and χ² will be small. Large χ² values (with a correspondingly
small p-value) indicate that the variables are associated.

**Assumptions:**
- Observations are independent.
- Expected counts should be ≥ 5 in each cell (a rough guideline). When this is
  violated, the χ² approximation becomes unreliable.
""")

        section("Fisher's Exact Test")
        st.markdown("""
For 2×2 contingency tables with small expected counts (any cell < 5), **Fisher's
exact test** computes the *exact* probability of observing the table (or one more
extreme) under the null hypothesis of independence, using the hypergeometric
distribution. Unlike the chi-squared test, it makes no large-sample approximation
and is valid for any sample size.

Fisher's exact test is computationally intensive for larger tables, so it is
typically reserved for 2×2 tables. For larger tables with small expected counts,
alternatives include combining sparse categories or using simulation-based
methods.
""")

        section("Effect Size: Cramér's V")
        st.latex(r"""
        V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, c) - 1)}}
        """)
        st.markdown("""
Cramér's V normalizes the chi-squared statistic to a value between 0 and 1,
where 0 indicates no association and 1 indicates perfect association. The
normalization by min(r, c) − 1 (where r and c are the number of rows and columns)
accounts for the fact that larger tables can produce larger χ² values simply due
to having more cells.

Guidelines for interpreting V: 0.1 = small, 0.3 = medium, 0.5 = large (though
these depend on the table dimensions and the specific field).
""")

        app_connection(
            "The <strong>Hypothesis Testing</strong> page offers chi-squared and Fisher's "
            "exact tests for categorical variable pairs. It displays the contingency "
            "table, test statistic, p-value, and Cramér's V. Results can be exported "
            "to Table 1."
        )

    # ── Normality Testing ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
Normality tests evaluate whether a variable's distribution is consistent with
a Gaussian distribution. They are used to determine whether parametric tests
(which assume normality) are appropriate.
""")

        section("Shapiro-Wilk Test")
        st.markdown("""
The Shapiro-Wilk test is the most powerful normality test for small to moderate
sample sizes (n < 5,000). The null hypothesis is that the data *is* normally
distributed. A significant p-value (< 0.05) rejects normality.

**Important caveat:** With large samples (n > 500), the Shapiro-Wilk test becomes
very sensitive and will reject normality for trivial deviations that have no
practical impact on your analysis. At n = 5,000, even a tiny skew will produce
a significant result. In these cases, visual inspection (Q-Q plots, histograms)
is more informative than the formal test.
""")

        section("When Normality Matters — and When It Doesn't")
        st.markdown("""
Normality is required for:
- **Exact** inference in small-sample linear regression (confidence intervals, p-values).
- Parametric tests (t-test, ANOVA) with small samples.

Normality is *not* required for:
- **Large-sample** inference (the CLT ensures approximate normality of test statistics).
- Tree-based or neural network predictions.
- Descriptive statistics.

As a practical rule: if n > 30 per group, the Central Limit Theorem makes
parametric tests robust to moderate non-normality. If n < 30 or the data is
heavily skewed, use non-parametric alternatives.
""")

        app_connection(
            "The <strong>Hypothesis Testing</strong> page offers the Shapiro-Wilk test "
            "for any numeric variable. The result includes the test statistic, p-value, "
            "and guidance on whether to use parametric or non-parametric tests downstream."
        )

    # ── Paired Comparisons ───────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("""
Paired comparisons are used when measurements come in natural pairs — before/after
treatment, left eye/right eye, or matched case-control designs.
""")

        section("Paired t-Test")
        st.markdown("""
The paired t-test works by computing the *difference* within each pair and then
testing whether the mean difference is zero:
""")
        st.latex(r"""
        t = \frac{\bar{d}}{s_d / \sqrt{n}} \quad \text{where} \quad d_i = x_{i,\text{after}} - x_{i,\text{before}}
        """)
        st.markdown("""
Each pair yields a difference d_i (after minus before). The test statistic is the
mean of these differences (d̄) divided by their standard error (s_d/√n, where s_d
is the standard deviation of the differences). Essentially, it reduces a two-sample
problem to a one-sample test on the differences.

By working with differences, the test controls for all between-subject variability.
This makes it much more powerful than an unpaired test when the pairing structure
is real — a treatment effect of 2 points might be undetectable with an unpaired
test (because between-subject variability is large) but easily significant with
a paired test (because within-subject variability is small).

**Assumption:** The differences *d_i* are approximately normally distributed.
""")

        section("Wilcoxon Signed-Rank Test")
        st.markdown("""
The non-parametric alternative to the paired t-test. Instead of assuming normally
distributed differences, it ranks the absolute differences and compares the sum
of positive ranks to the sum of negative ranks. Use it when differences are
skewed or have outliers.
""")

        app_connection(
            "The <strong>Hypothesis Testing</strong> page offers paired t-test and "
            "Wilcoxon signed-rank test. Select two variables representing paired "
            "measurements (e.g., pre/post), and the app computes the test, reports "
            "results, and allows export to Table 1."
        )

    # ── Table 1 ──────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("""
Table 1 is the standard summary table in biomedical publications. It describes
the study population — typically stratified by group (treatment vs. control,
responders vs. non-responders) — with descriptive statistics and between-group
p-values for each variable.
""")

        section("What Goes in Table 1")
        st.markdown("""
For each variable:
- **Continuous variables:** Mean ± SD (if approximately normal) or Median [IQR]
  (if skewed). The appropriate summary depends on the distribution.
- **Categorical variables:** Count (percentage) per level.

For each variable, a between-group test is applied:
- **Continuous, normal:** t-test (or ANOVA for >2 groups).
- **Continuous, non-normal:** Mann-Whitney U (or Kruskal-Wallis).
- **Categorical:** Chi-squared test (or Fisher's exact test for small expected counts).
""")

        section("Choosing the Right Test Per Variable")
        st.markdown("""
The test choice is driven by:
1. **Variable type:** Continuous vs. categorical.
2. **Distribution:** Normal vs. non-normal (for continuous variables).
3. **Number of groups:** Two vs. three or more.
4. **Sample size:** Small samples may require exact tests (Fisher's) or
   non-parametric alternatives.

A common mistake is applying the same test (usually the t-test) to every variable
regardless of type and distribution. This produces misleading p-values — for example,
a t-test on a heavily skewed variable overestimates differences at the tails.
""")

        section("Table 1 Is Descriptive, Not Inferential")
        st.markdown("""
Despite the p-values, Table 1 is fundamentally *descriptive* — it characterizes
your study population. The p-values describe baseline balance, not treatment effects.
In a randomized trial, any baseline differences are due to chance, so small p-values
in Table 1 don't indicate confounding (they indicate bad luck).

In observational studies, Table 1 p-values help identify potential confounders that
need adjustment. But they should not be over-interpreted — clinical significance of
a baseline difference matters more than statistical significance.
""")

        app_connection(
            "Table 1 is generated during <strong>EDA</strong> and can be customized in the "
            "<strong>Hypothesis Testing</strong> page. Custom tests from that page are merged "
            "into Table 1 for the final export. The <strong>Report Export</strong> page "
            "includes Table 1 in both the manuscript draft and as a standalone CSV/LaTeX export."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 7: Sensitivity & Robustness
# ══════════════════════════════════════════════════════════════════════════════

def render_sensitivity():
    intro(
        "A result that changes dramatically with a different random seed or a slightly "
        "different sample is not a result — it's an accident. Sensitivity analysis "
        "quantifies how stable your findings are and where fragility lurks. "
        "This covers the theory behind the <strong>Sensitivity Analysis</strong> page."
    )

    tabs = st.tabs([
        "Seed Sensitivity",
        "Feature Dropout",
        "Bootstrap Stability",
        "Interpreting Stability",
    ])

    # ── Seed Sensitivity ─────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
Many steps in the ML pipeline involve randomness: the train/test split, the
initialization of neural network weights, the bootstrap samples in bagging, the
feature subsets in random forests. The **random seed** controls all of these. A
different seed produces a different split, different initialization, and potentially
different results.
""")

        section("Why Seed Matters")
        st.markdown("""
If your model's performance changes substantially with a different seed, it means
the result is sensitive to the particular random partition of the data. This
instability can arise from:

- **Small sample size:** With few observations, the train/test split has a large
  effect on which patterns the model sees.
- **High variance models:** Neural networks and unpruned trees are more affected
  by random initialization and different training sets.
- **Noisy data:** When the signal-to-noise ratio is low, different samples of the
  noise produce different apparent patterns.
- **Class imbalance:** A different split may produce a test set with a different
  (and unrepresentative) class distribution.
""")

        section("How the App Tests Seed Sensitivity")
        st.markdown("""
The app retrains the selected model across multiple seeds (default: 8, configurable
up to 20). For each seed:

1. A new train/test split is generated.
2. The model is retrained from scratch on the new training set.
3. Performance metrics are computed on the new test set.

The result is a distribution of metric values across seeds, from which the app
computes the mean, standard deviation, and **coefficient of variation (CV)**:
""")
        st.latex(r"""
        \text{CV} = \frac{\sigma_{\text{metric}}}{\mu_{\text{metric}}} \times 100\%
        """)
        st.markdown("""
The coefficient of variation expresses the standard deviation as a percentage of
the mean. The app uses these thresholds:

- **CV < 2%:** Highly robust — results are stable across random splits.
- **CV 2–5%:** Acceptable — some variation, but conclusions are likely reliable.
- **CV 5–10%:** Concerning — consider ensemble methods, larger training sets, or
  simpler models.
- **CV > 10%:** Unstable — the result is not reliable and should not be reported
  as a single number without the variation range.
""")

        with st.expander("🧮 Interactive: See how seed choice affects reported performance"):
            seed_noise = st.slider(
                "Signal-to-noise ratio (lower = noisier, more seed-sensitive)",
                min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="theory_seed_snr",
            )
            seed_n = 60
            n_seeds = 10
            seed_scores = []
            for s in range(n_seeds):
                rng_s = np.random.default_rng(s * 7 + 3)
                X_s = rng_s.standard_normal((seed_n, 2))
                y_s = seed_noise * X_s[:, 0] + rng_s.normal(0, 1, seed_n)
                # Random 70/30 split
                idx_s = rng_s.permutation(seed_n)
                split = int(seed_n * 0.7)
                X_tr_s, y_tr_s = X_s[idx_s[:split]], y_s[idx_s[:split]]
                X_te_s, y_te_s = X_s[idx_s[split:]], y_s[idx_s[split:]]
                X_tr_aug_s = np.column_stack([np.ones(X_tr_s.shape[0]), X_tr_s])
                X_te_aug_s = np.column_stack([np.ones(X_te_s.shape[0]), X_te_s])
                beta_s = np.linalg.lstsq(X_tr_aug_s, y_tr_s, rcond=None)[0]
                y_pred_s = X_te_aug_s @ beta_s
                ss_res_s = np.sum((y_te_s - y_pred_s) ** 2)
                ss_tot_s = np.sum((y_te_s - np.mean(y_te_s)) ** 2)
                seed_scores.append(1 - ss_res_s / max(ss_tot_s, 1e-10))

            cv_pct = np.std(seed_scores) / max(abs(np.mean(seed_scores)), 1e-10) * 100
            stability = "Highly robust" if cv_pct < 2 else "Acceptable" if cv_pct < 5 else "Concerning" if cv_pct < 10 else "Unstable"
            bar_colors = ["rgba(220, 38, 38, 0.7)" if s == min(seed_scores) or s == max(seed_scores) else "rgba(99, 102, 241, 0.7)" for s in seed_scores]

            fig_seed = go.Figure()
            fig_seed.add_trace(go.Bar(x=[f"Seed {i}" for i in range(n_seeds)], y=seed_scores, marker_color=bar_colors))
            fig_seed.add_hline(y=np.mean(seed_scores), line_dash="dash", line_color="#16a34a",
                               annotation_text=f"Mean = {np.mean(seed_scores):.3f}")
            fig_seed.update_layout(
                title=f"R² across {n_seeds} seeds — CV = {cv_pct:.1f}% ({stability})",
                yaxis_title="R²", height=300,
                margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_seed, use_container_width=True)
            st.markdown(
                "**Train your eye:** The red bars are the best and worst seeds — they define the range of results you might have reported. "
                "Slide the signal-to-noise ratio down: the bars spread apart, the CV% climbs, and the stability label shifts from 'Robust' toward 'Unstable.' "
                "**The key question:** would you feel comfortable putting the best red bar in a paper? "
                "If the worst red bar would tell a different story, you should report the mean ± SD across seeds, not a single run."
            )

        misconception(
            "A single lucky seed is not evidence of a reliable model. If results move substantially when the split changes, the uncertainty belongs in the scientific story."
        )

        self_check(
            "If a model looks excellent under one seed but mediocre under several others, what should you report: the best run, the average behavior, or both — and why?"
        )

        app_connection(
            "The <strong>Sensitivity Analysis</strong> page displays a bar chart of "
            "performance across seeds, the CV percentage with a color-coded stability "
            "rating, and a full results table. For publication, report the mean ± SD "
            "across seeds rather than a single-seed result."
        )

    # ── Feature Dropout ────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
Feature dropout analysis removes one feature at a time, retrains the model, and
measures the change in performance. This provides a *causal* complement to
SHAP and permutation importance, which are *observational*.
""")

        section("How Feature Dropout Differs from Permutation Importance")
        st.markdown("""
Permutation importance shuffles a feature's values and measures the effect on
predictions *without retraining*. This is fast but has a limitation: the model
was trained with that feature present, so its internal structure still "expects"
it. Shuffling creates unrealistic data points (e.g., a 25-year-old with the
income of a 60-year-old) that the model was never trained to handle.

Feature dropout takes the stronger approach: it *removes* the feature entirely and
*retrains from scratch*. This measures what the model would have learned *if the
feature had never been available*. It is slower (one full retraining per feature)
but answers a more informative question: "Does the model genuinely need this feature,
or can it reconstruct the same information from other features?"
""")

        section("Interpreting Dropout Results")
        st.markdown("""
Three outcomes are possible when feature *j* is removed:

- **Performance drops significantly:** Feature *j* carries unique signal that no
  other feature can substitute for. It is genuinely important.
- **Performance is unchanged:** Feature *j* is either redundant (its information is
  captured by other features) or irrelevant (it was never useful). Either way, it
  can safely be removed.
- **Performance improves:** Feature *j* was actively hurting the model — likely adding
  noise, introducing overfitting, or causing the model to learn a spurious
  association. Removing it is beneficial.

The last case is particularly valuable: it identifies features that look
important by simpler metrics (they may have high permutation importance) but are
actually harmful when the model can adapt without them.
""")

        app_connection(
            "The <strong>Sensitivity Analysis</strong> page runs feature dropout for the "
            "selected model across up to 30 features. It shows a bar chart of performance "
            "change per feature, highlighting features whose removal improves performance. "
            "This complements the SHAP analysis from the Explainability page."
        )

    # ── Bootstrap Stability ──────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
The bootstrap is a resampling method that estimates the sampling distribution of
a statistic by repeatedly sampling *with replacement* from the
data. {cite("Efron & Tibshirani, 1993")}
""", unsafe_allow_html=True)

        section("How the Bootstrap Works")
        st.markdown("""
Given a dataset of *n* observations:

1. Draw a sample of *n* observations *with replacement* (some observations appear
   multiple times, others not at all — on average, about 63.2% of unique observations
   appear in each bootstrap sample).
2. Compute the statistic of interest (e.g., AUROC, RMSE) on this bootstrap sample.
3. Repeat *B* times (typically B = 1,000).
4. The distribution of the *B* statistics approximates the sampling distribution.
""")

        section("Bootstrap Confidence Intervals")
        st.markdown("""
From the bootstrap distribution, confidence intervals can be constructed:

**Percentile method:** Use the 2.5th and 97.5th percentiles of the bootstrap
distribution as the 95% CI. Simple and intuitive, but can have poor coverage
when the bootstrap distribution is skewed.

**BCa (Bias-Corrected and accelerated):** Adjusts for both bias and skewness in
the bootstrap distribution. More reliable than the percentile method, especially
for small samples. This is the method used by the app.
""")
        st.latex(r"""
        \text{CI}_{95\%}^{\text{BCa}} = \left[\hat{\theta}^*_{(\alpha_1)}, \;\hat{\theta}^*_{(\alpha_2)}\right]
        """)
        st.markdown("""
Here *θ̂*_(α₁)* and *θ̂*_(α₂)* are adjusted percentiles of the bootstrap distribution.
The adjustment accounts for the fact that the bootstrap distribution may not be
centered on the true parameter (bias correction) and may be wider in one direction
than the other (acceleration).
""")

        with st.expander("🧮 Interactive: Watch the bootstrap distribution build up"):
            boot_B = st.slider(
                "Number of bootstrap resamples (B)",
                min_value=20, max_value=1000, value=200, step=20, key="theory_boot_B",
            )
            rng_b = np.random.default_rng(99)
            # Toy dataset: skewed metric to show asymmetric CI
            sample_data = rng_b.exponential(scale=5.0, size=40)

            boot_means = []
            for b in range(boot_B):
                resample = rng_b.choice(sample_data, size=len(sample_data), replace=True)
                boot_means.append(np.mean(resample))

            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            obs_mean = np.mean(sample_data)

            fig_boot = go.Figure()
            fig_boot.add_trace(go.Histogram(
                x=boot_means, nbinsx=min(40, boot_B // 5),
                marker_color="rgba(99, 102, 241, 0.6)",
                marker_line=dict(color="rgba(99, 102, 241, 1)", width=1),
            ))
            fig_boot.add_vline(x=obs_mean, line_dash="dash", line_color="#dc2626",
                               annotation_text=f"Observed mean: {obs_mean:.2f}")
            fig_boot.add_vline(x=ci_lo, line_dash="dot", line_color="#16a34a",
                               annotation_text=f"2.5%: {ci_lo:.2f}")
            fig_boot.add_vline(x=ci_hi, line_dash="dot", line_color="#16a34a",
                               annotation_text=f"97.5%: {ci_hi:.2f}")
            fig_boot.update_layout(
                title=f"Bootstrap distribution of the mean (B = {boot_B})",
                xaxis_title="Bootstrap mean", yaxis_title="Count",
                height=300, margin=dict(t=50, b=30, l=50, r=20), template="plotly_white",
            )
            st.plotly_chart(fig_boot, use_container_width=True)
            st.markdown(
                f"**Train your eye:** The green dotted lines mark the 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]. "
                "Start at B = 20: the histogram is choppy and the CI boundaries jump around. "
                "Slide B up to 500+: the shape stabilizes, the CI tightens, and you can see the slight **right skew** in the distribution. "
                "That asymmetry means the upper and lower CI bounds are not equidistant from the mean — "
                "this is exactly why BCa correction exists. "
                "**In your own results:** if bootstrap CIs for two models overlap substantially, the difference between them is not reliable."
            )

        misconception(
            "A 95% bootstrap confidence interval does not mean there is a 95% probability that the true value lies inside this one realized interval. It means the interval-generating procedure is designed to capture the true value about 95% of the time over repeated samples."
        )

        self_check(
            "If two models differ by 0.01 in AUROC but their bootstrap confidence intervals overlap heavily, how strong is the evidence that one model is truly better?"
        )

        app_connection(
            "The <strong>Train & Compare</strong> page computes BCa bootstrap 95% CIs "
            "(1,000 resamples) for every primary metric on every model. These CIs "
            "appear alongside point estimates and provide the uncertainty quantification "
            "needed for publication."
        )

        references([
            "Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.",
            "DiCiccio, T.J. & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189–228.",
        ])

    # ── Interpreting Stability ───────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
Stability results require judgment to interpret. Not all instability is bad, and
not all stability is good.
""")

        section("Stable Results: When to Trust Them")
        st.markdown("""
A model that produces consistent metrics across seeds and tight bootstrap CIs is
reliable in the sense that *this particular dataset and method consistently produce
this result*. But consistency is necessary, not sufficient: a model can be stably
bad (consistently mediocre performance) or stably overfit (consistently memorizing
noise).

Check stability alongside *absolute* performance: a model with AUROC 0.92 ± 0.01
is both good and stable. A model with AUROC 0.55 ± 0.01 is stable but useless.
""")

        section("Unstable Results: What They Tell You")
        st.markdown("""
Instability is a signal that your analysis is data-limited in some way:

- **All models unstable:** The dataset is probably too small or too noisy for reliable
  prediction. Consider whether the prediction task is feasible.
- **Only complex models unstable:** Simpler models (Ridge, logistic) are stable but
  complex ones (neural net, deep trees) are not. The complex models don't have
  enough data to learn their additional parameters reliably.
- **Feature importance unstable:** Different seeds produce different "top features."
  This means no individual feature has a strong, robust signal — the model is
  finding different weak patterns in each split.
""")

        section("Reporting Sensitivity Results")
        st.markdown("""
For publication, sensitivity analysis transforms a single number into a
credible range:

- Report metrics as **mean ± SD across seeds** (e.g., "AUROC: 0.87 ± 0.03 across
  8 random seeds").
- Report **bootstrap 95% CIs** alongside point estimates (e.g., "RMSE: 4.2 [3.8, 4.7]").
- If feature importance is unstable, report the *consensus* features (those
  consistently in the top ranks) rather than a single ranking.
""")

        app_connection(
            "The <strong>Sensitivity Analysis</strong> page provides all the raw material "
            "for these reporting patterns: seed sensitivity results with CV percentages, "
            "feature dropout impact charts, and the <strong>Train & Compare</strong> page "
            "provides bootstrap CIs for every metric."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 8: Reporting Standards (TRIPOD)
# ══════════════════════════════════════════════════════════════════════════════

def render_reporting():
    intro(
        "The ultimate output of this app is a defensible, reproducible manuscript. "
        "Reporting standards like TRIPOD exist because reviewers and readers need "
        "to assess <em>how</em> you built your model, not just <em>how well</em> it "
        "performed. This chapter explains the standards the "
        "<strong>Report Export</strong> page helps you meet."
    )

    tabs = st.tabs([
        "TRIPOD Guidelines",
        "Methods Section Writing",
        "Reproducibility",
    ])

    # ── TRIPOD ───────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"""
**TRIPOD** (Transparent Reporting of a multivariable prediction model for Individual
Prognosis or Diagnosis) is a 22-item checklist for reporting prediction model
studies. It was developed by a large international group of methodologists and
journal editors to address the consistently poor reporting quality in
prediction modeling research. {cite("Collins et al., 2015")}
""", unsafe_allow_html=True)

        section("Why TRIPOD Exists")
        st.markdown("""
Systematic reviews have found that most published prediction model studies fail
to report basic information needed to evaluate or reproduce the work:

- Many studies don't report how missing data was handled.
- Most don't describe the sample size relative to the number of predictors.
- Few describe the full modeling process (feature selection, hyperparameter tuning).
- Performance is often reported without confidence intervals or on training data only.

TRIPOD provides a standardized checklist so that readers, reviewers, and future
researchers can evaluate the quality and applicability of a prediction model.
""")

        section("Key TRIPOD Items")
        st.markdown("""
The 22 items span the entire study, organized by manuscript section:

**Title & Abstract:**
- Identify the study as developing and/or validating a prediction model.
- Specify the target population and outcome.

**Methods:**
- Describe the data source, eligibility criteria, and study dates.
- Report the outcome definition and how it was measured.
- Describe all candidate predictors and how they were handled.
- Report the sample size and number of events (EPV).
- Describe the modeling approach: selection method, model type, any shrinkage/regularization.
- Describe how missing data was handled.

**Results:**
- Report the flow of participants (how many excluded and why — CONSORT-style diagram).
- Describe participant characteristics (**Table 1**).
- Report model performance with confidence intervals.
- Report discrimination (AUROC) and calibration (reliability plot, Brier score).

**Discussion:**
- Discuss limitations, including overfitting risk and generalizability.
- Discuss implications for clinical use.
""")

        app_connection(
            "The <strong>Report Export</strong> page includes a TRIPOD checklist tracker "
            "that automatically checks items based on your workflow (e.g., Table 1 generated, "
            "calibration computed, CI reported). Items that need manual attention are flagged. "
            "The generated methods section is structured to address TRIPOD reporting items."
        )

        references([
            "Collins, G.S., Reitsma, J.B., Altman, D.G., & Moons, K.G.M. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *Annals of Internal Medicine*, 162(1), 55–63.",
            "Moons, K.G.M., Altman, D.G., Reitsma, J.B., et al. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): Explanation and Elaboration. *Annals of Internal Medicine*, 162(1), W1–W73.",
        ])

    # ── Methods Section ──────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
The methods section is the most scrutinized part of a prediction modeling paper.
A reviewer's first task is to determine whether the methodology is defensible —
and missing or vague methods are the most common reason for rejection.
""")

        section("What a Complete Methods Section Contains")
        st.markdown("""
A defensible methods section addresses (at minimum):

**Study Population:**
- Data source and collection period.
- Inclusion/exclusion criteria.
- Sample size and number of events.

**Predictors:**
- How candidate predictors were selected.
- How continuous predictors were handled (transformations, binning).
- How categorical predictors were encoded.
- How missing data was handled (mechanism assumed, method used).

**Modeling:**
- Model type(s) and why chosen.
- Feature selection method and rationale.
- Hyperparameter tuning approach (cross-validation, grid search).
- Software and package versions.

**Evaluation:**
- How data was split (holdout ratio, stratification, random seed).
- Performance metrics and why each was chosen.
- How confidence intervals were computed (bootstrap parameters).
- How calibration was assessed.

**Sensitivity:**
- How stability was assessed (seed sensitivity, cross-validation variability).
""")

        section("Common Pitfalls")
        st.markdown("""
- **"We used machine learning to predict X."** Which algorithm? Which settings?
- **"Features were selected based on clinical relevance."** How? By whom? Any statistical validation?
- **"The model was validated."** On what data? Train or test? How was the split done?
- **"We achieved 95% accuracy."** With what confidence interval? On balanced or imbalanced classes?

The methods section generated by the app is designed to avoid these pitfalls by
automatically documenting every decision you made throughout the workflow.
""")

        app_connection(
            "The <strong>Report Export</strong> page generates a draft methods section "
            "that reflects your actual workflow: which features were selected and how, "
            "which preprocessing was applied to each model, which metrics were computed, "
            "and which sensitivity analyses were run. It is a starting point that you "
            "should review and adapt for your specific study context."
        )

    # ── Reproducibility ──────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
Reproducibility means that another researcher, given your data and description,
can arrive at the same results. This is the minimum standard for scientific
credibility — yet most ML studies fail it.
""")

        section("What Makes a Study Reproducible")
        st.markdown("""
At minimum:
- **The exact data** (or a synthetic equivalent) is available.
- **The code** (or a complete algorithmic description) is available.
- **The random seeds** are reported for all stochastic steps.
- **Package versions** are documented (different versions can produce different results).
- **The full pipeline** (including preprocessing) is described, not just the model.
""")

        section("Session Files (.pkl)")
        st.markdown("""
The app saves the entire analysis state as a `.pkl` (pickle) file that can be
reloaded to reproduce the exact analysis. This includes:

- The uploaded data (or a reference to it).
- All configuration choices (target variable, feature selections, preprocessing settings).
- Trained models and their parameters.
- All computed metrics, plots, and intermediate results.
- The random seed used for all stochastic operations.

This is stronger than code reproducibility because it captures the *state* of
the analysis, not just the *procedure*. Even if a library version changes, the
saved models and results remain intact.
""")

        section("CONSORT-Style Flow Diagrams")
        st.markdown("""
The app generates a flow diagram showing how data moved through the pipeline:
how many observations were in the original dataset, how many were excluded (and why),
how many ended up in training vs. test sets, and how many were used for each analysis.
This is modeled on the CONSORT flow diagram used in randomized controlled trials
and is a TRIPOD reporting requirement.
""")

        app_connection(
            "The <strong>Report Export</strong> page bundles the methods section, Table 1, "
            "TRIPOD checklist, flow diagram, trained models, and session state into a "
            "single downloadable ZIP file. This package is designed to support both "
            "manuscript writing and reproducibility review."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

CHAPTER_RENDERERS = {
    "Data Quality & Assumptions": render_data_quality,
    "Feature Engineering & Selection": render_feature_engineering,
    "Model Families": render_model_families,
    "Preprocessing Theory": render_preprocessing,
    "Evaluation & Validation": render_evaluation,
    "Statistical Testing": render_statistical_testing,
    "Sensitivity & Robustness": render_sensitivity,
    "Reporting Standards (TRIPOD)": render_reporting,
}

CHAPTER_RENDERERS[chapter]()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 0.5rem 0 1rem 0;">
    This reference is part of <strong style="color: #334155;">Tabular ML Lab</strong>.
    Content is written for researchers and reviewers — if something is unclear or
    incomplete, that's a bug worth reporting.
</div>
""", unsafe_allow_html=True)
