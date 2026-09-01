"""
Dataset regime detection — two orthogonal axes: layout, and compute.

Computed once on page load. All UI components read the regime
to decide what to render (full heatmap vs top-N list, etc).

`feature_regime` and everything derived from it answer *what to show*. That is
the original axis and it is frozen. The second half of this module answers *what
to compute* — may a p×p matrix be built at this width, is VIF defined at this
p/n, what does KernelExplainer cost here — and it is deliberately not a fifth
rung on the display ladder. See the "Compute caps" section for why the two must
not be the same ladder.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import numpy as np


# ===========================================================================
# Compute caps — the axis that governs WORK, orthogonal to the display ladder
# ===========================================================================
#
# The display ladder below (`feature_regime`: narrow/medium/wide/ultra_wide)
# decides how many plots to draw and how long a list to print. Every one of its
# ultra_wide guardrails governs what to DISPLAY; not one governs what to
# COMPUTE. So the p×p correlation matrix, the per-feature VIF loop and the
# kernel SHAP allocator ran at any width the upload happened to have — a
# 60,000-column matrix took the same code path as a 201-column panel, and the
# only difference was how much of the answer got printed.
#
# These caps are the missing axis, and they are NOT a new rung on that ladder,
# on purpose. Three of the six feature_regime-derived properties invert if
# anything falls out of `ultra_wide`, because the top rung is written as an
# equality test rather than a floor:
#
#   * `distribution_mode` tests `== "ultra_wide"`, so a wider rung would send a
#     60,000-gene matrix back to the 3×3 plot gallery;
#   * `corr_top_n` falls through to `return 0`, and its consumer does
#     `np.argsort(np.abs(vals))[-n:][::-1]` — with n=0, `x[-0:]` is `x[0:]`,
#     i.e. EVERY pair, reversed. A rung added to restrict work would hand back
#     an uncapped pair table at the one width where that is fatal;
#   * `target_relationship_top_n` returns 0, whose documented meaning is
#     "show all (paginated)".
#
# An orthogonal axis makes that class of regression structurally impossible: no
# existing branch condition changes, so a 201-column panel and a 60,000-column
# matrix render exactly as they did before this section existed.
#
# THE DISCLOSURE RULE. This app's output is a manuscript. A cap that silently
# changes which features were analyzed writes a Methods section describing an
# analysis that did not happen. So every gate here returns its user-facing
# SENTENCE alongside its number, in the shape `ml/card_evidence.py` established
# (`gallery_availability`) and for the reason stated there: two implementations
# of a threshold is drift, and two implementations of the *sentence* is worse,
# because a sentence a user reads that no engine composed cannot be reviewed.
# Pages render `reason` and file it in the insight ledger; they do not compose
# their own.
#
# Every number below is measured. The measurement is on the line above it.
#
# Plain ints in, plain data out: no pandas, no Streamlit, no session state.
# `ml/eda_actions.py` and `ml/eda_recommender.py` are headless by contract
# (tests/test_engine_is_headless.py) and import these directly; they never build
# a DatasetRegime and must not start.

# Any p×p construction — the correlation matrix behind the top-pairs table.
# Measured full path: 0.38 s / 27 MB at p=1,000; 4.64 s / 241 MB at p=3,000;
# 14.32 s / 669 MB at p=5,000; 70.76 s / 2,671 MB at p=10,000 (exponent p^2.18,
# R²=0.9990). Memory is 28p² bytes — a model that matched measurement to 0.0%
# at p=10,000, so 10.4 GB at p=20,000 and 94 GB at p=60,000, in a block that
# carries no try/except. Two independent budgets converge on 1,000: the 2 s
# interactivity crossing for both p×p sites combined is p=1,503, and the eager
# per-column page load crosses 2 s at p=984. 1,000 is the largest round number
# inside both, and it leaves the entire 201–1,000 band — where p=250 costs
# 0.07 s — running exactly as it does today.
DENSE_PAIRWISE_MAX_FEATURES = 1_000

# The same construction when it is a RANK correlation over a frame with missing
# cells, which is a different cost curve entirely. Measured 5%-missing Spearman:
# 1.52 s at p=250, 6.29 s at p=500, 24.95 s at p=1,000, 110.7 s at p=2,000 —
# 104× to 164× the same call on complete data, because pandas drops to a
# pairwise-complete O(p²) loop. The trigger is a cliff, not a gradient: 58
# missing cells in 100,000 (0.06%) already cost 27×. The 2 s crossing is p=283;
# 250 is the smallest width actually measured on that path, so the constant is
# a measurement rather than an interpolation.
#
# NOTE what this is NOT. On COMPLETE data Spearman is 3–5× FASTER than Pearson
# (1.22 s vs 5.95 s at p=3,000). Capping rank correlation on width alone would
# slow the app down for no reason. Missingness is the discriminator.
#
# AND note what it is: the width at which the rank SUBSTITUTION triggers, not a
# cap any caller enforces as a column budget. Above it the site switches to
# Pearson-of-ranks, which is no longer on the pairwise-complete path, so the
# budget that then applies is DENSE_PAIRWISE_MAX_FEATURES.
# `pairwise_correlation_plan()` is the one place that sequences those two
# decisions; ask it, not `dense_pairwise_budget()`, for a rank method.
RANK_CORR_PAIRWISE_MAX_FEATURES = 250

# Eager work that is linear in p — the per-column profile and summary loops that
# run on first paint. Fitted t = 2.122e-3 · p^0.994 (R²=1.0000 over p=500–10,000)
# once the quadratic correlation term is removed by the caps above: 10.01 s
# measured at p=5,000, 20.20 s at p=10,000. 5,000 is the 10 s crossing (fit says
# p=4,969). Peak RSS never exceeded 490 MB at any width measured on this path,
# so there is no memory case for capping it — above this the right answer is to
# make the per-column loops cheaper, not to analyze fewer columns.
PER_COLUMN_SCAN_MAX_FEATURES = 5_000

# VIF fits one OLS per feature against all the others. Measured at n=500 on 14
# cores: 4.32 s at p=200, 77.93 s at p=600, 202.19 s at p=800; at p=1,000 it
# blew through a 900 s cap while saturating ~9 cores and was killed, not
# finished. The curve is reproducibly NON-monotonic (p=300 slower than p=400,
# isolated to a LAPACK gelsd path), so any policy that predicts runtime from p
# mispredicts by 3× in that band — which is exactly why this gate is a hard
# COUNT and never a runtime estimate. Memory is irrelevant here: peak RSS was
# 209–237 MB from p=20 to p=1,000.
VIF_MAX_FEATURES = 200

# shap.KernelExplainer, measured end-to-end on the page's own hard-coded
# parameters (50 background rows, 50 explained rows): 159.1 s for the batch at
# p=20 and 182.4 s at p=50 — three minutes per model at FIFTY features. Per
# explained row: 4.82 s at p=200, 8.83 s at p=400, 20.54 s at p=800 (2.60 GB
# peak). Above 200 features the run stops being something to start on the user's
# behalf and becomes something to quote a price for.
KERNEL_SHAP_CONFIRM_FEATURES = 200

# Where the same estimator stops being affordable at all. Memory is deterministic
# and needs no timing: shap sets nsamples = 2·M + 2**11 for "auto" and tiles the
# background to (2p+2048)·n_bg·p·8 bytes, RE-ALLOCATED per explained row.
# Measured real peak was 2.03× that analytic tile at p=400, 600 and 800. p=1,000
# extrapolates to ~3.2 GB and ~24 min per model; p=2,000 to ~9.7 GB and ~1.2 h.
# EXTRAPOLATED, not measured — the largest width actually run was p=800. A
# fully-measured threshold would be 800; 800 and 1,000 differ by one
# confirmation click, not by a failure mode.
KERNEL_SHAP_MAX_FEATURES = 1_000

# Above this the refusal carries no override button. The analytic 8 GB crossing
# is p=1,756 and this is the next round number past it. JUDGMENT, not
# measurement: nothing was run at p=2,000 on this path, because it would need
# ~9.7 GB and over an hour.
KERNEL_SHAP_NO_OVERRIDE_FEATURES = 2_000

# sklearn permutation_importance, with the page's exact defaults (n_repeats=10,
# n_jobs=-1, 200 test rows), RandomForest: 26.8 s at p=200, 67.4 s at p=500,
# 101.2 s at p=1,000, 200.3 s at p=2,000 — PER MODEL, and the page runs every
# trained model by default. Above this the checkbox stops defaulting ON. It is
# deliberately never a refusal: this is the only model-agnostic importance the
# page offers, it does complete, and Cancel works. 33 minutes at p=20,000 is
# long but finite, and refusing would delete information a scientist may
# legitimately choose to wait for.
PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES = 1_000

# TreeExplainer needs no feature cap — measured FLAT in p (RandomForest at 200
# explained rows: 1.297 / 1.238 / 1.311 / 1.301 s at p = 500 / 2,000 / 5,000 /
# 20,000, fitted exponent 0.00 across a 40× range) because its cost is
# O(trees · leaves · depth²) per row and depth is set by n_train, not by p. The
# only thing that grows is the RESULT array, which stays resident in session
# state for the report: n_eval · p · n_classes float64 cells. 62.5M cells is
# 0.5 GB. JUDGMENT: the largest case measured (500 rows × 20,000 features × 5
# classes) left 400 MB resident and completed fine, so this is a round number
# below the largest thing observed to work, not a measured cliff. When it fires
# it reduces ROWS, never features.
SHAP_RESULT_CELL_BUDGET = 62_500_000

# How a capped subset is chosen, everywhere. Named because it travels into
# ledger metadata and into the manuscript sentence, and because "the first 200
# columns of the upload" — which is what `ml/macro_shape.py` silently does — is
# not a scientific criterion. Highest-variance is the idiom already used in
# `ml/eda_recommender.py` and `ml/coach_probe.py`.
#
# It is not a guarantee. NOTHING measured says the 1,000 highest-variance
# columns contain the dataset's strongest correlated pair. That is precisely why
# the sentences below say "a stronger pair may exist among them" rather than
# implying the printed list is the dataset's top N.
SELECTION_RULE_VARIANCE = "variance"


def variance_subset_phrase(n_used: int, n_total: int, noun: str = "numeric features") -> str:
    """"the 1,000 highest-variance of 12,431 numeric features" — one wording.

    Every disclosure that reduces a feature set embeds this fragment, so the
    selection rule is stated the same way on the EDA page, in the ledger
    metadata and in the manuscript sentence.
    """
    return f"the {int(n_used):,} highest-variance of {int(n_total):,} {noun}"


def dense_pairwise_budget(
    n_features: int,
    has_missing_cells: bool = False,
    method: str = "pearson",
) -> int:
    """How many columns may go into a p×p construction. Never more than exist.

    Returns `min(n_features, cap)`, so `budget == n_features` means "no cap
    fired" and `budget < n_features` means the caller must subset — by variance,
    and must say so.

    `has_missing_cells` + a rank `method` selects the much lower rank-correlation
    cap, because pandas computes rank correlation pairwise-complete when cells
    are missing and that is a different cost curve (see
    RANK_CORR_PAIRWISE_MAX_FEATURES). A caller that instead applies the rank
    SUBSTITUTION — `sub.rank().corr(method="pearson")` — is no longer on that
    path and should ask again with `method="pearson"` to get the full budget.
    `pairwise_correlation_plan()` does both in one call and is the better entry
    point for the correlation site.
    """
    n = max(int(n_features), 0)
    cap = DENSE_PAIRWISE_MAX_FEATURES
    if has_missing_cells and _is_rank_method(method):
        cap = RANK_CORR_PAIRWISE_MAX_FEATURES
    return min(n, cap)


def _is_rank_method(method: Any) -> bool:
    """Spearman by any spelling the page might pass ("Spearman", "spearman")."""
    return str(method or "").strip().lower().startswith("spearman")


def rank_correlation_substitution_applies(
    n_features: int,
    has_missing_cells: bool,
    method: str = "pearson",
) -> bool:
    """Should Spearman be computed as Pearson-of-ranks instead of pairwise?

    True only for a rank method, on a frame with missing cells, above
    RANK_CORR_PAIRWISE_MAX_FEATURES. Measured 55.4× faster at p=2,000
    (93.37 s -> 1.69 s) with max|difference| 0.0126 and median 0.00096.

    NOT numerically identical, so unlike an exact refactor it carries a
    disclosure obligation — `pairwise_correlation_plan()["rank_substitution_reason"]`
    is the sentence.
    """
    return (
        _is_rank_method(method)
        and bool(has_missing_cells)
        and int(n_features) > RANK_CORR_PAIRWISE_MAX_FEATURES
    )


def pairwise_correlation_plan(
    n_features: int,
    has_missing_cells: bool = False,
    method: str = "pearson",
    missing_cell_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """The whole decision for a p×p correlation site, sentences included.

    Returns the method actually to execute, how many columns to keep, and the
    plain-language reason for each departure from what was asked for — or None
    where nothing departed, which is the common case and must stay silent. A
    caption written for an analysis that was not reduced is a false caveat in
    the Methods section, which is the same class of error as a silent cap.
    """
    n = max(int(n_features), 0)
    substitute = rank_correlation_substitution_applies(n, has_missing_cells, method)

    # The substitution moves the work off the pairwise-complete path, so the
    # budget that then applies is the ordinary dense one, not the rank cap.
    effective_method = "pearson" if substitute else method
    budget = dense_pairwise_budget(
        n, has_missing_cells=has_missing_cells and not substitute, method=effective_method
    )
    capped = budget < n

    reason = None
    if capped:
        pairs_kept = budget * (budget - 1) // 2
        pairs_total = n * (n - 1) // 2
        reason = (
            f"Correlations were screened among "
            f"{variance_subset_phrase(budget, n)} — {pairs_kept:,} of the "
            f"{pairs_total:,} possible pairs. Correlations involving the other "
            f"{n - budget:,} features were not computed, and a stronger pair "
            f"may exist among them."
        )

    substitution_reason = None
    if substitute:
        # The tradeoff sentence is already written for the target-correlation
        # helper at utils/perf_cache.py; this is that same statement, said once.
        pct = ""
        if missing_cell_fraction is not None:
            pct = f", because {float(missing_cell_fraction) * 100:.1f}% of cells are missing"
        substitution_reason = (
            "Spearman was computed as the Pearson correlation of column ranks "
            "rather than by pairwise-complete ranking" + pct + ". With missing "
            "values the two can differ in the third decimal; in benchmarking "
            "the largest difference was 0.013."
        )

    return {
        "n_features": n,
        "method_requested": str(method or "pearson").strip().lower(),
        "method_executed": "spearman_on_ranks" if substitute else str(method or "pearson").strip().lower(),
        "rank_substitution": substitute,
        "rank_substitution_reason": substitution_reason,
        "max_features": budget,
        "capped": capped,
        "selection_rule": SELECTION_RULE_VARIANCE if capped else None,
        "reason": reason,
    }


def vif_is_defined(n_features: int, n_rows: int) -> bool:
    """May variance inflation factors be computed on p predictors and n rows?

    `p <= min(VIF_MAX_FEATURES, n/2)`, and at least 2 predictors — VIF regresses
    each feature on the others, so one feature has nothing to regress on.

    The 200 is the WALL-TIME gate (see VIF_MAX_FEATURES). The n/2 is a VALIDITY
    gate and it is the more important of the two: on features with true VIF = 1
    BY CONSTRUCTION (i.i.d. normals), the measured median in-sample VIF is 2.018
    at p/n = 0.5, 9.822 at 0.9 — where 203 of 450 independent features cross the
    app's "> 10" alarm — and 53.2 at 0.98 with every feature flagged. The law is
    E[VIF] = (n-1)/(n-p), confirmed at two sample sizes, so the limit is the
    RATIO and no column count can express it. At p >= n the estimator is
    undefined and must be refused outright rather than returning a sentinel.
    """
    p, n = int(n_features), int(n_rows)
    return 2 <= p <= VIF_MAX_FEATURES and 2 * p <= n


def vif_null_baseline(n_features: int, n_rows: int) -> Optional[float]:
    """E[VIF] for features with NO collinearity at all: (n-1)/(n-p).

    None when p >= n, where the quantity does not exist. This is what a reported
    VIF must be read against — a bare "> 10" is not defensible above p/n = 0.5,
    because at that ratio sample size alone produces it.
    """
    p, n = int(n_features), int(n_rows)
    if n - p <= 0:
        return None
    return (n - 1) / (n - p)


def vif_availability(n_features: int, n_rows: int) -> Dict[str, Any]:
    """The VIF gate and the sentence that discloses it, in one place.

    `flag_threshold` is the "severely multicollinear" line, scaled by the null
    baseline instead of the fixed 10 the app used, so the alarm means the same
    thing at p/n = 0.1 and p/n = 0.4.
    """
    p, n = int(n_features), int(n_rows)
    baseline = vif_null_baseline(p, n)
    out: Dict[str, Any] = {
        "available": False,
        "n_features": p,
        "n_rows": n,
        "limit_features": VIF_MAX_FEATURES,
        "limit_ratio": 0.5,
        "null_baseline_vif": baseline,
        "flag_threshold": 10.0 * baseline if baseline is not None else None,
        "reason": None,
    }
    if p < 2:
        out["reason"] = (
            f"VIF was not computed: it needs at least two numeric predictors to "
            f"regress each feature on the others, and {p} were selected."
        )
        return out
    if p > VIF_MAX_FEATURES:
        out["reason"] = (
            f"VIF was not computed: {p:,} numeric features were selected and the "
            f"estimator is capped at {VIF_MAX_FEATURES}. At 500 observations, VIF "
            f"takes 4 s at 200 features and over 15 minutes at 1,000. Reduce "
            f"features on the Feature Selection page, or use the collinearity "
            f"screen instead."
        )
        return out
    if 2 * p > n:
        out["reason"] = (
            f"VIF was not computed: {p:,} predictors against {n:,} observations. "
            f"Above p = n/2 the in-sample VIF is inflated by sample size alone — "
            f"at this ratio every feature would be reported as severely "
            f"multicollinear even if all of them were independent — and at "
            f"p >= n it is undefined."
        )
        return out
    out["available"] = True
    return out


def ols_diagnostic_is_defined(n_features: int, n_rows: int) -> bool:
    """May an OLS residual/influence diagnostic be computed at this shape?

    `p <= n - 2`: a fit needs at least two residual degrees of freedom before
    leverage, Cook's distance or a normality test on residuals means anything.

    The failure above this line is SILENT, which is why it needs a gate rather
    than a try/except. `np.linalg.solve` does not raise on the singular Gram
    matrix at any p from 99 to 3,000. At n=100/p=99 the influence path returns
    max leverage 1.0000000000009526 — leverage is bounded by 1, so that is a
    mathematically impossible statistic reported to four decimals — and
    Cook's D 1.0087e12. Residual normality is worse: at n=500/p=3,000 it
    reports p=2.8e-05 ("residuals deviate from normality") on residuals of
    sd 2.4e-14 from a fit with in-sample R² = 1.0, while the identical
    degeneracy at n=100/p=200 gives 0.7163 ("residuals look normal"). The
    verdict is a function of the rounding pattern, not of the data.
    """
    p, n = int(n_features), int(n_rows)
    return p >= 1 and p <= n - 2


def ols_diagnostic_availability(
    n_features: int, n_rows: int, analysis: str = "influence"
) -> Dict[str, Any]:
    """The OLS-diagnostic gate plus its sentence. `analysis` names the noun."""
    p, n = int(n_features), int(n_rows)
    label = {
        "influence": "Influence diagnostics were not computed",
        "normality": "Residual normality was not tested",
    }.get(str(analysis).strip().lower(), "The regression diagnostic was not computed")
    out: Dict[str, Any] = {
        "available": ols_diagnostic_is_defined(p, n),
        "n_features": p,
        "n_rows": n,
        "limit_features": max(n - 2, 0),
        "reason": None,
    }
    if not out["available"]:
        out["reason"] = (
            f"{label}: {p:,} predictors against {n:,} observations. A model with "
            f"at least as many predictors as observations fits the data exactly, "
            f"so every observation has leverage 1, Cook's distance is undefined, "
            f"and the residuals are rounding error rather than model error."
        )
    return out


def kernel_shap_policy(n_features: int) -> Literal["run", "confirm", "refuse"]:
    """Run KernelExplainer, quote a price first, or decline.

    "confirm" is not a soft refusal — it is the honest state for a job that
    takes minutes and gigabytes but does finish. See
    KERNEL_SHAP_CONFIRM_FEATURES / KERNEL_SHAP_MAX_FEATURES for the numbers.
    """
    p = int(n_features)
    if p <= KERNEL_SHAP_CONFIRM_FEATURES:
        return "run"
    if p <= KERNEL_SHAP_MAX_FEATURES:
        return "confirm"
    return "refuse"


def kernel_shap_cost_estimate(
    n_features: int, n_background: int = 50, n_eval: int = 50
) -> Dict[str, Any]:
    """Seconds and bytes for one model, from shap's own allocator arithmetic.

    shap sets `nsamples = 2·M + 2**11` for "auto" and tiles the background to
    `(2p + 2048) · n_bg · p` float64 cells, re-allocated for every explained
    row. Time is that cell count at the measured 141 ns/cell; peak memory is the
    analytic tile times the 2.03× measured at p=400, 600 and 800.

    The 141 ns/cell rate is calibrated at p=800 (predicts 20.57 s/row against
    20.54 s measured) and is the most conservative of the measured rates, so it
    UNDER-predicts at small p — 3.45 s/row at p=200 against 4.82 s measured.
    Treat it as an order-of-magnitude price tag, which is all a confirmation
    dialog needs it to be.
    """
    p = max(int(n_features), 0)
    n_bg = max(int(n_background), 1)
    rows = max(int(n_eval), 1)
    cells_per_row = (2 * p + 2048) * n_bg * p
    return {
        "n_features": p,
        "n_background": n_bg,
        "n_eval": rows,
        "seconds_per_row": cells_per_row * 141e-9,
        "seconds_per_model": cells_per_row * 141e-9 * rows,
        "peak_bytes_per_model": int(cells_per_row * 8 * 2.03),
    }


def kernel_shap_availability(
    n_features: int,
    n_models: int = 1,
    model_label: Optional[str] = None,
    n_background: int = 50,
    n_eval: int = 50,
) -> Dict[str, Any]:
    """Policy, price and sentence for the model-agnostic SHAP path."""
    p = int(n_features)
    policy = kernel_shap_policy(p)
    est = kernel_shap_cost_estimate(p, n_background, n_eval)
    models = max(int(n_models), 1)
    minutes = est["seconds_per_model"] / 60.0
    gb = est["peak_bytes_per_model"] / 1e9
    who = model_label or "this model"

    out: Dict[str, Any] = {
        "policy": policy,
        "n_features": p,
        "confirm_above": KERNEL_SHAP_CONFIRM_FEATURES,
        "refuse_above": KERNEL_SHAP_MAX_FEATURES,
        "override_allowed": p < KERNEL_SHAP_NO_OVERRIDE_FEATURES,
        "estimated_minutes_per_model": minutes,
        "estimated_gb_per_model": gb,
        "n_models": models,
        "reason": None,
    }
    if policy == "confirm":
        total = f", {minutes * models:,.0f} min for {models} models" if models > 1 else ""
        out["reason"] = (
            f"KernelExplainer on {p:,} features: about {minutes:,.0f} minutes and "
            f"{gb:,.1f} GB per model{total}. Nothing has been computed yet."
        )
    elif policy == "refuse":
        out["reason"] = (
            f"SHAP was not computed for {who}: the model-agnostic kernel "
            f"estimator needs about {gb:,.0f} GB and {minutes / 60:,.1f} hours at "
            f"{p:,} features. TreeExplainer models were explained normally, and "
            f"permutation importance remains available for this one."
        )
    return out


def permutation_importance_default_on(n_features: int) -> bool:
    """Whether the permutation-importance checkbox starts ticked.

    Never a refusal — above the threshold it merely stops being something the
    app starts on the user's behalf.
    """
    return int(n_features) <= PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES


def permutation_importance_cost_estimate(
    n_features: int, n_models: int = 1, n_repeats: int = 10
) -> Dict[str, Any]:
    """Seconds per model from the measured RandomForest fit, t = 2.0 + 0.0992·p.

    Fitted at n_repeats=10, 200 test rows, n_jobs=-1: 101.2 s measured at
    p=1,000 and 200.3 s at p=2,000. RandomForest was the SLOWEST estimator
    measured and Ridge at p=5,000 took 43 s, so on linear models this
    over-quotes by roughly 4×. Over-quoting is the safe direction for a
    default-off notice; a per-model timed estimate would be better and is noted
    as future work rather than guessed at here.
    """
    p = max(int(n_features), 0)
    models = max(int(n_models), 1)
    per_model = (2.0 + 0.0992 * p) * (max(int(n_repeats), 1) / 10.0)
    return {
        "n_features": p,
        "n_models": models,
        "n_repeats": int(n_repeats),
        "seconds_per_model": per_model,
        "seconds_total": per_model * models,
    }


def permutation_importance_availability(
    n_features: int, n_models: int = 1, n_repeats: int = 10
) -> Dict[str, Any]:
    """Default state plus the sentence beside the checkbox."""
    p = int(n_features)
    on = permutation_importance_default_on(p)
    est = permutation_importance_cost_estimate(p, n_models, n_repeats)
    out: Dict[str, Any] = {
        "default_on": on,
        "n_features": p,
        "limit": PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES,
        "estimated_minutes_per_model": est["seconds_per_model"] / 60.0,
        "estimated_minutes_total": est["seconds_total"] / 60.0,
        "reason": None,
    }
    if not on:
        models = max(int(n_models), 1)
        total = (
            f", {est['seconds_total'] / 60.0:,.0f} minutes for {models} models"
            if models > 1
            else ""
        )
        out["reason"] = (
            f"Permutation importance is off by default at {p:,} features — about "
            f"{est['seconds_per_model'] / 60.0:,.0f} minutes per model at "
            f"{int(n_repeats)} repeats{total}. Tick to run it."
        )
    return out


def shap_result_guard(
    n_eval_rows: int, n_features: int, n_classes: int = 1
) -> Dict[str, Any]:
    """How many rows of SHAP values may be kept for `n_features` × `n_classes`.

    The ONLY guard on the TreeExplainer path, and it reduces evaluation ROWS,
    never features: TreeExplainer is flat in p (measured exponent 0.00 from
    p=500 to p=20,000), so capping it on feature count would delete information
    for free and force a Methods caveat onto an analysis that needed none.
    What does grow is the result array held in session state for the report.

    Because it reduces rows, the consumer that appends "using N test
    observations" to the SHAP sentence must be handed `n_rows` from here rather
    than the requested count.
    """
    rows = max(int(n_eval_rows), 0)
    p = max(int(n_features), 1)
    classes = max(int(n_classes), 1)
    per_row = p * classes
    allowed = max(SHAP_RESULT_CELL_BUDGET // per_row, 1)
    kept = min(rows, allowed)
    out: Dict[str, Any] = {
        "n_rows": kept,
        "n_rows_requested": rows,
        "reduced": kept < rows,
        "n_features": p,
        "n_classes": classes,
        "cell_budget": SHAP_RESULT_CELL_BUDGET,
        "reason": None,
    }
    if out["reduced"]:
        out["reason"] = (
            f"SHAP values were computed on {kept:,} of the {rows:,} requested "
            f"evaluation rows to keep the stored result under "
            f"{SHAP_RESULT_CELL_BUDGET * 8 / 1e9:,.1f} GB; all {p:,} features "
            f"were explained."
        )
    return out


@dataclass
class DatasetRegime:
    """Describes the shape of the dataset for adaptive UI decisions.
    
    Computed from the active DataFrame + feature/target configuration.
    Immutable for the duration of a page render.
    """
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_missing_cols: int          # columns with any missing values
    n_high_missing_cols: int     # columns with >5% missing
    has_target: bool
    target_type: Optional[str]   # "numeric", "categorical", None

    # -- Feature regime ----------------------------------------------------

    @property
    def feature_regime(self) -> Literal["narrow", "medium", "wide", "ultra_wide"]:
        """How many features the dataset has — drives gallery/matrix decisions."""
        if self.n_features <= 15:
            return "narrow"
        elif self.n_features <= 50:
            return "medium"
        elif self.n_features <= 200:
            return "wide"
        else:
            return "ultra_wide"

    # -- Row regime --------------------------------------------------------

    @property
    def row_regime(self) -> Literal["tiny", "standard", "large", "massive"]:
        """How many rows — drives sampling/plotting decisions."""
        if self.n_rows < 100:
            return "tiny"
        elif self.n_rows < 10_000:
            return "standard"
        elif self.n_rows < 100_000:
            return "large"
        else:
            return "massive"

    # -- Derived properties ------------------------------------------------

    @property
    def needs_sampling(self) -> bool:
        """Whether scatter plots should sample data."""
        return self.row_regime in ("large", "massive")

    @property
    def sample_size(self) -> int:
        """Recommended sample size for scatter plots."""
        if self.row_regime == "massive":
            return 5_000
        elif self.row_regime == "large":
            return 5_000
        return self.n_rows  # no sampling needed

    @property
    def show_full_corr_matrix(self) -> bool:
        """Whether to show full NxN correlation heatmap."""
        return self.feature_regime in ("narrow", "medium")

    @property
    def show_macro_shape(self) -> bool:
        """Whether to show PCA/UMAP/TDA section."""
        return self.feature_regime != "narrow"

    @property
    def gallery_page_size(self) -> int:
        """Number of feature charts per page in distribution gallery."""
        return 9  # 3×3 grid

    @property
    def show_sample_size_warning(self) -> bool:
        """Whether to warn about small sample size."""
        return self.row_regime == "tiny"

    @property
    def use_hexbin(self) -> bool:
        """Whether scatter plots should use hexbin instead of points."""
        return self.row_regime == "massive"

    @property
    def distribution_mode(self) -> Literal["gallery", "summary"]:
        """How to show feature distributions."""
        if self.feature_regime == "ultra_wide":
            return "summary"  # summary-of-summaries view
        return "gallery"  # small multiples grid

    @property
    def corr_top_n(self) -> int:
        """How many correlation pairs to show in list view."""
        if self.feature_regime == "wide":
            return 30
        elif self.feature_regime == "ultra_wide":
            return 50
        return 0  # not used when showing full matrix

    @property
    def target_relationship_top_n(self) -> int:
        """How many features to auto-show in target relationship gallery."""
        if self.feature_regime == "ultra_wide":
            return 10
        return 0  # show all (paginated)

    @property
    def macro_shape_tiers(self) -> List[str]:
        """Which macro-shape views to offer."""
        if self.feature_regime == "narrow":
            return []
        elif self.feature_regime == "medium":
            return ["pca"]
        elif self.feature_regime == "wide":
            return ["pca", "umap"]
        else:
            return ["pca", "umap", "persistence", "mapper"]

    # -- Compute regime ----------------------------------------------------
    #
    # Orthogonal to `feature_regime` above, exactly as `row_regime` is. These
    # are PROPERTIES, not fields, so nothing that constructs or serializes a
    # DatasetRegime changes shape. The module-level functions at the top of this
    # file are the same policy for callers that never build one.

    @property
    def compute_regime(self) -> Literal["direct", "guarded", "capped"]:
        """How this width is worked, as opposed to how it is drawn.

        * `direct`  — everything runs as written. Measured first paint under
          2 s and every p×p construction inside its budget.
        * `guarded` — the quadratic paths are capped and disclosed; the linear
          per-column work still runs whole (10.01 s measured at p=5,000).
        * `capped`  — above PER_COLUMN_SCAN_MAX_FEATURES the eager per-column
          scan itself is over its 10 s budget. Note what this tier does NOT
          authorize: peak RSS never exceeded 490 MB on that path at any width
          measured, so there is no memory case for dropping columns from it.
          The right fix in this tier is a cheaper loop, not a smaller analysis.
        """
        if self.n_features <= DENSE_PAIRWISE_MAX_FEATURES:
            return "direct"
        if self.n_features <= PER_COLUMN_SCAN_MAX_FEATURES:
            return "guarded"
        return "capped"

    @property
    def dense_pairwise_max_features(self) -> int:
        """Columns allowed into a p×p construction, for the DEFAULT method.

        Measured against `n_numeric`, because every p×p site in the app is
        numeric-only. Equal to `n_numeric` when no cap fires.

        A correlation site that offers a rank method cannot be served by a bare
        property — the property cannot see which pill is selected — so it calls
        `dense_pairwise_budget_for(method)` or, better,
        `pairwise_correlation_plan()`, which also returns the sentence.
        """
        return self.dense_pairwise_budget_for("pearson")

    def dense_pairwise_budget_for(self, method: str = "pearson") -> int:
        """The budget the correlation site will actually use for `method`.

        Delegates to `pairwise_correlation_plan()` rather than calling
        `dense_pairwise_budget()` directly, because the plan applies the rank
        SUBSTITUTION first and the substitution moves the work off the
        pairwise-complete path — after which the ordinary dense budget is the
        one that applies. Asking the bare predicate instead returned
        RANK_CORR_PAIRWISE_MAX_FEATURES (250) where the site would use
        DENSE_PAIRWISE_MAX_FEATURES (1,000): two answers to one question, inside
        the module that exists so there is exactly one.

        Note what this makes RANK_CORR_PAIRWISE_MAX_FEATURES: the threshold at
        which the substitution TRIGGERS, not a cap any caller enforces.
        """
        return pairwise_correlation_plan(
            self.n_numeric,
            has_missing_cells=self.n_missing_cols > 0,
            method=method,
        )["max_features"]

    @property
    def compute_caps_engaged(self) -> bool:
        """Will any cap fire on the EAGER path — first paint, nothing clicked?

        The page reads this to decide whether a caps disclosure block renders at
        all. Scoped deliberately to work that happens without being asked:

        * a p×p correlation construction wider than its budget;
        * a width past `direct`, where the per-column scan is over budget.

        Two things are deliberately NOT folded in, both because they depend on
        a choice this object cannot see, and a first-paint block announcing a
        cap on work nobody asked for is a caveat about an analysis that did not
        happen — the mirror image of the silent truncation this axis exists to
        prevent:

        * the rank-correlation substitution, which depends on the method pill.
          `pairwise_correlation_plan()` returns its sentence at that site.
        * the on-demand regression diagnostics (VIF, influence, residual
          normality), which disclose through the `warnings` list their engine
          functions already return, when and only when they are run.
        """
        return (
            self.compute_regime != "direct"
            or self.dense_pairwise_max_features < self.n_numeric
        )

    # -- Description -------------------------------------------------------

    def describe(self) -> str:
        """Human-readable description of the regime."""
        parts = []
        parts.append(f"{self.n_rows:,} rows × {self.n_features} features")
        parts.append(f"({self.n_numeric} numeric, {self.n_categorical} categorical)")
        parts.append(f"Feature regime: {self.feature_regime}")
        parts.append(f"Row regime: {self.row_regime}")
        if self.n_high_missing_cols > 0:
            parts.append(f"{self.n_high_missing_cols} columns with >5% missing")
        return " · ".join(parts)


def detect_regime(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
) -> DatasetRegime:
    """Detect the dataset regime from the active DataFrame.
    
    Args:
        df: Active DataFrame (may include target + feature columns)
        feature_cols: List of feature column names (excludes target)
        target_col: Target column name, or None
        
    Returns:
        DatasetRegime instance
    """
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = df[feature_cols].select_dtypes(
        include=["datetime64", "datetimetz"]
    ).columns.tolist()

    missing_counts = df[feature_cols].isnull().sum()
    n_missing_cols = int((missing_counts > 0).sum())
    n_high_missing = int((missing_counts / max(len(df), 1) > 0.05).sum())

    has_target = target_col is not None and target_col in df.columns
    target_type = None
    if has_target:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            target_type = "numeric"
        else:
            target_type = "categorical"

    return DatasetRegime(
        n_rows=len(df),
        n_features=len(feature_cols),
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        n_datetime=len(datetime_cols),
        n_missing_cols=n_missing_cols,
        n_high_missing_cols=n_high_missing,
        has_target=has_target,
        target_type=target_type,
    )
