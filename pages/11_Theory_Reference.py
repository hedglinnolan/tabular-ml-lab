"""
📖 Theory Reference — Statistical foundations behind every page of Tabular ML Lab.

Structured as a browsable reference: selectbox picks the chapter, tabs organize
parallel content (e.g., model families), and expanders offer optional deep dives.
"""

import streamlit as st
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

/* Formula block */
.formula-block {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.9rem;
    color: #e2e8f0;
    line-height: 1.7;
    overflow-x: auto;
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


def formula(text: str):
    """Render a formula/code block."""
    st.markdown(f'<div class="formula-block">{text}</div>', unsafe_allow_html=True)


def app_connection(text: str):
    """Render a callout linking theory to the app."""
    st.markdown(f'<div class="app-callout">🔬 <strong>In the app:</strong> {text}</div>', unsafe_allow_html=True)


def cite(text: str):
    """Return a citation badge HTML string for inline use."""
    return f'<span class="cite">{text}</span>'


def takeaway(text: str):
    """Render a key takeaway box."""
    st.markdown(f'<div class="key-takeaway">💡 <strong>Key takeaway:</strong> {text}</div>', unsafe_allow_html=True)


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

        formula(
            "P(R = 0 | Y_obs, Y_mis) = P(R = 0)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;← MCAR<br>"
            "P(R = 0 | Y_obs, Y_mis) = P(R = 0 | Y_obs)&emsp;&emsp;← MAR<br>"
            "P(R = 0 | Y_obs, Y_mis) depends on Y_mis&emsp;&emsp;&nbsp;← MNAR<br><br>"
            "where R is the missingness indicator (0 = missing, 1 = observed)"
        )

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
tail gives negative skew. The sample skewness is defined as:
""")
        formula(
            "γ₁ = (1/n) Σᵢ [(xᵢ - x̄) / s]³<br><br>"
            "where x̄ is the sample mean, s is the sample standard deviation, and n is the sample size."
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
        formula(
            "κ = (1/n) Σᵢ [(xᵢ - x̄) / s]⁴ − 3<br><br>"
            "κ > 0: heavy tails (leptokurtic) — more extreme values than normal<br>"
            "κ = 0: normal tails (mesokurtic)<br>"
            "κ < 0: light tails (platykurtic) — fewer extreme values than normal"
        )
        st.markdown("""
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
""")
            formula(
                "y(λ) = (xᵝ − 1) / λ&emsp;&emsp;if λ ≠ 0<br>"
                "y(λ) = log(x)&emsp;&emsp;&emsp;&emsp;if λ = 0<br><br>"
                "The optimal λ is chosen by maximum likelihood. Common values:<br>"
                "λ = 1: no transform &emsp; λ = 0.5: square root &emsp; λ = 0: log &emsp; λ = −1: reciprocal"
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
        formula(
            "<strong>Leverage:</strong>&emsp;hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ<br><br>"
            "The leverage hᵢᵢ measures how far observation i is from the center of<br>"
            "the feature space. High leverage (hᵢᵢ > 2p/n, where p = number of features)<br>"
            "means the observation has disproportionate pull on the regression surface.<br><br>"
            "<strong>Cook's Distance:</strong>&emsp;Dᵢ = (ŷ - ŷ₍ᵢ₎)ᵀ(ŷ - ŷ₍ᵢ₎) / (p · MSE)<br><br>"
            "Combines leverage and residual size into a single influence measure.<br>"
            "Dᵢ > 1 is a common threshold for concerning influence. "
            "Dᵢ > 4/n is a more sensitive threshold."
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
            formula(
                "L_ε(r) = r² / 2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;if |r| ≤ ε<br>"
                "L_ε(r) = ε|r| − ε² / 2&emsp;&emsp;&emsp;&emsp;if |r| > ε"
            )
            st.markdown(f"""
This gives the efficiency of OLS for well-behaved observations while limiting the
influence of outliers to a linear (rather than quadratic) contribution. The choice
of ε controls the tradeoff: smaller ε → more robustness, less efficiency; ε = 1.345
is the standard default, giving 95% asymptotic efficiency relative to OLS under
normality. {cite("Huber, 1964")} {cite("Huber & Ronchetti, 2009")}
""", unsafe_allow_html=True)

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
        formula(
            "Var(β̂ⱼ) = σ² · (XᵀX)⁻¹ⱼⱼ = σ² / [Σᵢ(xᵢⱼ − x̄ⱼ)² · (1 − Rⱼ²)]<br><br>"
            "where Rⱼ² is the R² from regressing feature j on all other features.<br>"
            "As Rⱼ² → 1 (perfect collinearity), the variance → ∞."
        )
        st.markdown("""
This inflated variance means:
- **Coefficients become unstable:** Small changes in the data produce wildly different estimates.
- **Confidence intervals widen:** You lose the ability to determine which variables matter.
- **Sign flips:** A coefficient may be positive in one sample and negative in another.

**Critically, collinearity does not affect prediction accuracy.** If you only care
about ŷ, a collinear model predicts just as well as a non-collinear one (assuming
future data has the same collinearity structure). The problem is entirely about
*interpretation* and *inference* on individual coefficients.
""")

        section("Detecting Collinearity: VIF")
        st.markdown("""
The Variance Inflation Factor (VIF) quantifies how much the variance of a
coefficient is inflated due to collinearity:
""")
        formula(
            "VIF(β̂ⱼ) = 1 / (1 − Rⱼ²)<br><br>"
            "where Rⱼ² = R² from regressing feature j on all other features.<br><br>"
            "VIF = 1 &nbsp;→ no collinearity<br>"
            "VIF = 5 &nbsp;→ variance inflated 5x (Rⱼ² = 0.80)<br>"
            "VIF = 10 → variance inflated 10x (Rⱼ² = 0.90) — common threshold for concern"
        )
        st.markdown("""
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

        app_connection(
            "The <strong>EDA</strong> page computes pairwise correlations and flags highly "
            "correlated pairs (|r| > 0.8). The coaching layer only raises collinearity "
            "as an issue for linear model families, since tree-based and other models "
            "are unaffected. The <strong>Feature Selection</strong> page offers VIF-based "
            "filtering as one of its selection methods."
        )

        with st.expander("Deep Dive: Condition Number"):
            st.markdown(f"""
For a more global measure of collinearity, the **condition number** of the design
matrix X examines the ratio of the largest to smallest singular values:
""")
            formula(
                "κ(X) = σ_max(X) / σ_min(X)<br><br>"
                "κ < 30: &emsp;well-conditioned<br>"
                "κ > 30: &emsp;moderate collinearity<br>"
                "κ > 1000: ill-conditioned — coefficients are numerically unstable"
            )
            st.markdown(f"""
The condition number captures the overall "health" of the design matrix in a single
number. A high condition number means that the regression problem is ill-posed:
small perturbations in y produce large changes in β̂. This is the matrix-algebra
equivalent of the VIF story, expressed at the level of the entire system rather
than individual variables. {cite("Belsley et al., 1980")}
""", unsafe_allow_html=True)

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
        formula(
            "EPV = n_events / p<br><br>"
            "where n_events = number of minority-class observations (classification)<br>"
            "&emsp;&emsp;&emsp;&emsp;or total sample size (regression)<br>"
            "and p = number of predictor variables (including dummies for categoricals)<br><br>"
            "EPV < 5: &nbsp;&nbsp;serious risk of overfitting and unstable estimates<br>"
            "EPV 5-10: marginal — simplify the model or regularize heavily<br>"
            "EPV > 20: generally comfortable for most linear methods"
        )

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
        formula(
            "In p dimensions, to capture a fraction r of the data in a hypercube,<br>"
            "the edge length must be r^(1/p).<br><br>"
            "To capture 10% of data in 1D: edge = 0.10<br>"
            "To capture 10% of data in 10D: edge = 0.10^(1/10) ≈ 0.79<br>"
            "To capture 10% of data in 100D: edge = 0.10^(1/100) ≈ 0.977<br><br>"
            "In high dimensions, the 'local' neighborhood covers nearly<br>"
            "the entire range of every feature."
        )
        st.markdown("""
The practical implications:
- **KNN** degrades rapidly as p grows, because all points become approximately equidistant.
- **SVM** with RBF kernels faces the same problem in the implicit feature space.
- **Linear models** can handle high p if regularized (LASSO, Ridge), but need strong regularization.
- **Tree-based models** handle high p relatively well because each split only considers one feature at a time, but random forests may waste splits on noise features.
- **Neural networks** are highly susceptible in the small-n regime: they have the capacity to memorize the training set, and without sufficient data, they will.
""")

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

Power depends on four quantities — fix any three and the fourth is determined:
""")
            formula(
                "Power = f(effect size, sample size, significance level, test type)<br><br>"
                "For a two-sample t-test: n ≈ 16σ² / δ²&emsp;(for 80% power, α = 0.05)<br>"
                "where δ = true difference in means, σ = common standard deviation<br><br>"
                "For logistic regression: n ≈ 10p / π_min<br>"
                "where p = number of predictors, π_min = proportion of minority class"
            )
            st.markdown(f"""
The key insight: **power analysis should be done before data collection, not after.**
A post-hoc power analysis on a non-significant result is circular — it will always
show low power, because the observed effect size is small by definition.
{cite("Hoenig & Heisey, 2001")}

If you're analyzing an existing dataset and suspect low power, the appropriate
response is to report confidence intervals (which convey the uncertainty directly)
rather than relying on a binary significant/non-significant decision.
""", unsafe_allow_html=True)


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

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")


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

    tabs = st.tabs([
        "Linear",
        "Tree-Based",
        "Neural Network",
        "Distance-Based",
        "Margin-Based",
        "Probabilistic",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[5]:
        st.markdown("*Content will be populated in the next pass.*")


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

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")


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
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")


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
        "Table 1 & Descriptive Tests",
        "Model Comparison Tests",
        "Goodness of Fit",
        "Bootstrap Methods",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")


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
        "Bootstrap Stability",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")


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
        "Reproducibility Checklist",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")


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
