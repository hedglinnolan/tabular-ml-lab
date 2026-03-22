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


def app_connection(text: str):
    """Render a callout linking theory to the app."""
    st.markdown(f'<div class="app-callout">🔬 <strong>In the app:</strong> {text}</div>', unsafe_allow_html=True)


def cite(text: str):
    """Return a citation badge HTML string for inline use."""
    return f'<span class="cite">{text}</span>'


def takeaway(text: str):
    """Render a key takeaway box."""
    st.markdown(f'<div class="key-takeaway">💡 <strong>Key takeaway:</strong> {text}</div>', unsafe_allow_html=True)


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
values above zero indicate heavier tails than normal (leptokurtic), and values below
zero indicate lighter tails (platykurtic).

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
Two complementary diagnostics formalize this. **Leverage** measures how unusual an
observation is in the feature space (how far it sits from the center of the data).
**Cook's distance** combines leverage with residual size to measure the observation's
overall *influence* on the fitted model.
""")
        st.latex(r"""
        h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i
        """)
        st.markdown("""
Here, **xᵢ** is the feature vector for observation *i*, and **(XᵀX)⁻¹** is the
inverse of the cross-product matrix of all features. The product xᵢᵀ(XᵀX)⁻¹xᵢ
measures the Mahalanobis distance of observation *i* from the centroid of the
feature space — essentially, how "unusual" this observation's feature values are
relative to the rest of the data. Leverage values range from 1/n to 1; a common
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
Cook's distance compares two sets of predictions: **ŷ**, the predictions from the
full model, and **ŷ₍ᵢ₎**, the predictions from a model fit *without* observation *i*.
The numerator measures how much all the predictions change when one observation is
removed; the denominator normalizes by the number of features *p* and the mean
squared error. A large Dᵢ means removing observation *i* substantially shifts the
regression surface. Common thresholds: Dᵢ > 1 for clear concern, or
Dᵢ > 4/n as a more sensitive screening criterion.
""")
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

        section("Detecting Collinearity: VIF")
        st.markdown("""
The Variance Inflation Factor (VIF) isolates the collinearity term from the
variance formula above, giving a clean multiplier that says "the variance of
this coefficient is VIF times larger than it would be if this feature were
completely uncorrelated with all others":
""")
        st.latex(r"""
        \text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R_j^2}
        """)
        st.markdown("""
The interpretation is direct. If Rⱼ² = 0 (feature *j* is uncorrelated with all
others), VIF = 1 — no inflation. If Rⱼ² = 0.80 (feature *j* shares 80% of its
variance with other features), VIF = 5 — the coefficient's variance is 5× larger
than it needs to be. At Rⱼ² = 0.90, VIF = 10, the conventional threshold at which
most textbooks recommend action. At Rⱼ² = 0.99, VIF = 100 — the coefficient
estimate is essentially noise.

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
        section("Consensus Approach")
        st.markdown("""
Because each selection method has different blind spots, the app runs both LASSO path
and RFE-CV by default and reports the **consensus** — features selected by both
methods. Features in the consensus set have survived two fundamentally different
selection criteria: the geometric sparsity of L1 regularization and the greedy
importance ranking of RFE. This provides stronger evidence of genuine signal
than either method alone.

The app also shows features selected by only one method, which may warrant manual
review — they might represent real signal that one method missed, or noise that one
method incorrectly retained.
""")
        app_connection(
            "The <strong>Feature Selection</strong> page runs LASSO path and RFE-CV, shows "
            "individual and consensus results, and allows manual override. The coaching "
            "layer warns when the consensus set is very small (potential signal loss) or "
            "very large (potential noise retention)."
        )

        references([
            "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), §6.1, §6.2. Springer.",
            "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288.",
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
