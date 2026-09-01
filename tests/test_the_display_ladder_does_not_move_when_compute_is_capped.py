"""`ml/regime.py` grew a second axis. This is the proof the first one did not move.

The file used to answer exactly one question — *how much of this dataset do we
DRAW* — through `feature_regime` (narrow/medium/wide/ultra_wide) and six
properties derived from it. It now also answers *how much of this dataset do we
COMPUTE*. The two axes are orthogonal by construction, and this file is what
makes that a fact rather than an intention.

Why an orthogonal axis and not a fifth rung: three of the six display properties
INVERT if anything falls out of `ultra_wide`, because the top rung is an
equality test rather than a floor. The worst is `corr_top_n`, which falls
through to `return 0` — and `_top_corr_pairs` in `pages/02_EDA.py` does
`np.argsort(np.abs(vals))[-n:][::-1]`, where `x[-0:]` is `x[0:]`, i.e. EVERY
pair, reversed. A rung added to restrict work would have handed back an uncapped
pair table at the one width where that is fatal. `test_no_width_can_reach_the_
uncapped_pair_table` pins the invariant that forbids it, rather than the literal
rung values, because the invariant is the thing a future rung would break.

Headlessness is not re-checked here: `tests/test_engine_is_headless.py` already
walks every non-underscore module in `ml/` and imports it with streamlit
blocked, and `ml.regime` is in that census. One cheap source assertion remains
below, because the compute axis is the reason a future edit might reach for the
host.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from ml.regime import (
    DENSE_PAIRWISE_MAX_FEATURES,
    KERNEL_SHAP_CONFIRM_FEATURES,
    KERNEL_SHAP_MAX_FEATURES,
    KERNEL_SHAP_NO_OVERRIDE_FEATURES,
    PER_COLUMN_SCAN_MAX_FEATURES,
    PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES,
    RANK_CORR_PAIRWISE_MAX_FEATURES,
    SHAP_RESULT_CELL_BUDGET,
    VIF_MAX_FEATURES,
    DatasetRegime,
    dense_pairwise_budget,
    detect_regime,
    kernel_shap_availability,
    kernel_shap_cost_estimate,
    kernel_shap_policy,
    ols_diagnostic_availability,
    ols_diagnostic_is_defined,
    pairwise_correlation_plan,
    permutation_importance_availability,
    permutation_importance_cost_estimate,
    permutation_importance_default_on,
    rank_correlation_substitution_applies,
    shap_result_guard,
    variance_subset_phrase,
    vif_availability,
    vif_is_defined,
    vif_null_baseline,
)


def _regime(n_features, n_rows=500, n_numeric=None, n_missing_cols=0):
    """A regime at a chosen shape, built by keyword so field order cannot bite."""
    return DatasetRegime(
        n_rows=n_rows,
        n_features=n_features,
        n_numeric=n_features if n_numeric is None else n_numeric,
        n_categorical=0,
        n_datetime=0,
        n_missing_cols=n_missing_cols,
        n_high_missing_cols=0,
        has_target=True,
        target_type="numeric",
    )


# ── 1 · the display ladder, pinned at every boundary and both sides ──────────

# feature_regime, show_full_corr_matrix, show_macro_shape, distribution_mode,
# corr_top_n, target_relationship_top_n, macro_shape_tiers — the six properties
# derived from the ladder, at the exact widths where it steps.
LADDER = [
    (1,      "narrow",     True,  False, "gallery", 0,  0,  []),
    (15,     "narrow",     True,  False, "gallery", 0,  0,  []),
    (16,     "medium",     True,  True,  "gallery", 0,  0,  ["pca"]),
    (50,     "medium",     True,  True,  "gallery", 0,  0,  ["pca"]),
    (51,     "wide",       False, True,  "gallery", 30, 0,  ["pca", "umap"]),
    (200,    "wide",       False, True,  "gallery", 30, 0,  ["pca", "umap"]),
    (201,    "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
    (1_000,  "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
    (1_001,  "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
    (5_000,  "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
    (5_001,  "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
    (60_000, "ultra_wide", False, True,  "summary", 50, 10, ["pca", "umap", "persistence", "mapper"]),
]


@pytest.mark.parametrize(
    "p,regime_name,full_matrix,macro,dist_mode,corr_n,target_n,tiers", LADDER
)
def test_the_display_ladder_is_byte_identical_at_every_boundary(
    p, regime_name, full_matrix, macro, dist_mode, corr_n, target_n, tiers
):
    """What the app DRAWS did not change at any width, including the new ones.

    Below 200 this is the no-new-friction guarantee: a panel dataset renders
    exactly as it did before the compute axis existed. At 1,000 and 5,000 — the
    widths where the compute caps now step — it is the proof that those steps
    did not leak into the ladder.
    """
    r = _regime(p)
    assert r.feature_regime == regime_name
    assert r.show_full_corr_matrix is full_matrix
    assert r.show_macro_shape is macro
    assert r.distribution_mode == dist_mode
    assert r.corr_top_n == corr_n
    assert r.target_relationship_top_n == target_n
    assert r.macro_shape_tiers == tiers


def test_no_width_can_reach_the_uncapped_pair_table():
    """`corr_top_n == 0` is only ever reachable where nobody slices with it.

    `pages/02_EDA.py` reads `corr_top_n` only in the `else` of
    `if regime.show_full_corr_matrix:`, and slices `[-n:]`. At n=0 that slice
    returns the whole array — every pair of a p x p matrix — so a zero on that
    branch is not a small table, it is no cap at all. The same shape holds for
    `target_relationship_top_n`, whose consumer reads 0 as "show all".
    """
    for p in [1, 15, 16, 50, 51, 200, 201, 1_000, 5_000, 60_000, 250_000]:
        r = _regime(p)
        assert r.show_full_corr_matrix or r.corr_top_n > 0, (
            f"at {p} features the top-pairs branch is reachable with n=0, which "
            f"slices to the FULL pair table")
        # The other two equality-tested properties, same shape: a width above
        # ultra_wide must keep the summary view and the truncated gallery.
        if p > 200:
            assert r.distribution_mode == "summary"
            assert r.target_relationship_top_n == 10


def test_the_dataclass_gained_no_fields():
    """The compute axis is properties only.

    Nothing in the repo constructs a DatasetRegime positionally or serializes
    one today, and a new field would be the way that stops being true quietly.
    """
    names = [f.name for f in dataclasses.fields(DatasetRegime)]
    assert names == [
        "n_rows", "n_features", "n_numeric", "n_categorical", "n_datetime",
        "n_missing_cols", "n_high_missing_cols", "has_target", "target_type",
    ]


def test_the_row_regime_axis_is_untouched():
    """The third axis, pinned, to prove the second one did not reach it."""
    cases = [
        (99, "tiny"), (100, "standard"), (9_999, "standard"),
        (10_000, "large"), (99_999, "large"), (100_000, "massive"),
    ]
    for n, expected in cases:
        r = _regime(20, n_rows=n)
        assert r.row_regime == expected
        assert r.needs_sampling is (expected in ("large", "massive"))
        assert r.use_hexbin is (expected == "massive")
        assert r.show_sample_size_warning is (expected == "tiny")
        assert r.sample_size == (5_000 if expected in ("large", "massive") else n)
        assert r.gallery_page_size == 9


def test_ml_regime_still_imports_no_host():
    """The engine census covers this; the compute axis is why it is worth saying.

    A cap that wants to render its own warning is one `import streamlit` away
    from taking `ml/eda_actions.py` — headless by contract — down with it.
    """
    import ml.regime as regime_mod

    src = open(regime_mod.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*import streamlit", src, re.M)
    assert not re.search(r"^\s*from streamlit", src, re.M)


# ── 2 · the compute axis, at the exact values either side of each boundary ───


@pytest.mark.parametrize("p,expected", [
    (1, "direct"),
    (200, "direct"),
    (1_000, "direct"),
    (1_001, "guarded"),
    (5_000, "guarded"),
    (5_001, "capped"),
    (60_000, "capped"),
])
def test_the_compute_regime_steps_where_the_measurements_do(p, expected):
    assert _regime(p).compute_regime == expected


def test_the_compute_tier_boundaries_are_the_named_constants():
    """No second copy of the number: the tier reads the constants it documents."""
    assert DENSE_PAIRWISE_MAX_FEATURES == 1_000
    assert PER_COLUMN_SCAN_MAX_FEATURES == 5_000
    assert _regime(DENSE_PAIRWISE_MAX_FEATURES).compute_regime == "direct"
    assert _regime(DENSE_PAIRWISE_MAX_FEATURES + 1).compute_regime == "guarded"
    assert _regime(PER_COLUMN_SCAN_MAX_FEATURES).compute_regime == "guarded"
    assert _regime(PER_COLUMN_SCAN_MAX_FEATURES + 1).compute_regime == "capped"


@pytest.mark.parametrize("p,has_missing,method,expected", [
    (500,   False, "pearson",  500),    # under the cap: the budget IS the width
    (1_000, False, "pearson",  1_000),
    (1_001, False, "pearson",  1_000),
    (12_431, False, "pearson", 1_000),
    (1_001, True,  "pearson",  1_000),  # missingness does not touch Pearson
    (250,   True,  "spearman", 250),
    (251,   True,  "spearman", 250),    # the rank cap, and only here
    (251,   True,  "Spearman", 250),    # the pill's own capitalization
    (5_000, False, "spearman", 1_000),  # complete data: Spearman is the FASTER one
])
def test_the_dense_pairwise_budget_never_exceeds_the_width_it_is_given(
    p, has_missing, method, expected
):
    assert dense_pairwise_budget(p, has_missing_cells=has_missing, method=method) == expected


@pytest.mark.parametrize("p,has_missing,method,expected", [
    (250, True,  "spearman", False),
    (251, True,  "spearman", True),
    (251, True,  "Spearman", True),
    (251, False, "spearman", False),   # missingness is the discriminator
    (251, True,  "pearson",  False),
    (60_000, False, "spearman", False),
])
def test_the_rank_substitution_triggers_on_missingness_not_on_width(
    p, has_missing, method, expected
):
    assert rank_correlation_substitution_applies(p, has_missing, method) is expected


def test_a_correlation_plan_that_reduces_nothing_says_nothing():
    """A caveat about an analysis that was not reduced is itself a falsehood."""
    plan = pairwise_correlation_plan(500, has_missing_cells=False, method="pearson")
    assert plan["capped"] is False
    assert plan["max_features"] == 500
    assert plan["reason"] is None
    assert plan["rank_substitution"] is False
    assert plan["rank_substitution_reason"] is None
    assert plan["selection_rule"] is None


def test_a_correlation_plan_that_reduces_says_what_it_dropped_and_how_it_chose():
    plan = pairwise_correlation_plan(12_431, has_missing_cells=False, method="pearson")
    assert plan["capped"] is True
    assert plan["max_features"] == 1_000
    assert plan["selection_rule"] == "variance"
    reason = plan["reason"]
    # The three things a reader needs: how many were screened, out of how many,
    # by what rule — and that the answer may be elsewhere.
    assert "1,000 highest-variance of 12,431" in reason
    assert "11,431" in reason
    assert "a stronger pair may exist among them" in reason


def test_the_rank_substitution_is_disclosed_because_it_is_not_exact():
    plan = pairwise_correlation_plan(
        2_000, has_missing_cells=True, method="Spearman", missing_cell_fraction=0.02
    )
    assert plan["rank_substitution"] is True
    assert plan["method_executed"] == "spearman_on_ranks"
    # Substituting moves the work off the pairwise-complete path, so the
    # ordinary dense budget applies from there — not the 250 rank cap.
    assert plan["max_features"] == DENSE_PAIRWISE_MAX_FEATURES
    said = plan["rank_substitution_reason"]
    assert "Pearson correlation of column ranks" in said
    assert "2.0% of cells are missing" in said
    assert "0.013" in said


def test_a_narrow_rank_correlation_with_gaps_is_left_alone():
    """Below the rank cap nothing is substituted and nothing is claimed."""
    plan = pairwise_correlation_plan(200, has_missing_cells=True, method="spearman")
    assert plan["rank_substitution"] is False
    assert plan["rank_substitution_reason"] is None
    assert plan["capped"] is False
    assert plan["reason"] is None


@pytest.mark.parametrize("p,n,expected", [
    (200, 500, True),    # the wall-time cap, exactly at it
    (201, 500, False),
    (200, 400, True),    # p == n/2 exactly
    (200, 399, False),   # the ratio binds before the count does
    (100, 200, True),
    (100, 199, False),
    (2, 4, True),        # the smallest defined shape
    (1, 500, False),     # one predictor has nothing to be regressed on
    (0, 500, False),
    (500, 500, False),   # p == n: undefined, never a sentinel
])
def test_vif_is_defined_only_below_both_of_its_limits(p, n, expected):
    assert vif_is_defined(p, n) is expected


def test_the_vif_null_baseline_is_the_law_the_measurements_confirmed():
    """E[VIF] = (n-1)/(n-p) on features with no collinearity by construction.

    Measured median 2.018 at p/n = 0.5 against a theoretical 2.00, which is why
    a bare "VIF > 10" is not defensible as a fixed line.
    """
    assert vif_null_baseline(250, 500) == pytest.approx(499 / 250)   # 1.996 ~ 2
    assert vif_null_baseline(450, 500) == pytest.approx(499 / 50)     # 9.98
    assert vif_null_baseline(490, 500) == pytest.approx(499 / 10)     # 49.9
    assert vif_null_baseline(500, 500) is None  # undefined, not infinite
    assert vif_null_baseline(600, 500) is None


def test_the_vif_refusal_says_which_limit_it_hit():
    too_wide = vif_availability(3_412, 500)
    assert too_wide["available"] is False
    assert "capped at 200" in too_wide["reason"]
    assert "3,412" in too_wide["reason"]

    too_few_rows = vif_availability(150, 200)
    assert too_few_rows["available"] is False
    assert "p = n/2" in too_few_rows["reason"]
    assert "150 predictors against 200 observations" in too_few_rows["reason"]

    ok = vif_availability(150, 500)
    assert ok["available"] is True
    assert ok["reason"] is None
    # The alarm line moves with the baseline instead of sitting at a flat 10.
    assert ok["null_baseline_vif"] == pytest.approx(499 / 350)
    assert ok["flag_threshold"] == pytest.approx(10 * 499 / 350)


@pytest.mark.parametrize("p,n,expected", [
    (98, 100, True),
    (99, 100, False),   # p == n - 1: no residual degrees of freedom left
    (100, 100, False),
    (3_000, 500, False),
    (20, 500, True),
    (0, 100, False),
])
def test_the_ols_diagnostics_refuse_where_the_fit_is_exact(p, n, expected):
    assert ols_diagnostic_is_defined(p, n) is expected


def test_the_ols_refusal_names_the_analysis_it_declined():
    """Two analyses share the gate; a warning that names neither helps nobody."""
    infl = ols_diagnostic_availability(3_000, 500, analysis="influence")
    assert infl["available"] is False
    assert infl["reason"].startswith("Influence diagnostics were not computed")
    assert "3,000 predictors against 500 observations" in infl["reason"]

    norm = ols_diagnostic_availability(3_000, 500, analysis="normality")
    assert norm["reason"].startswith("Residual normality was not tested")

    fine = ols_diagnostic_availability(20, 500, analysis="influence")
    assert fine["available"] is True and fine["reason"] is None


@pytest.mark.parametrize("p,expected", [
    (1, "run"),
    (199, "run"),
    (200, "run"),
    (201, "confirm"),
    (999, "confirm"),
    (1_000, "confirm"),
    (1_001, "refuse"),
    (2_000, "refuse"),
    (60_000, "refuse"),
])
def test_the_kernel_shap_policy_steps_at_its_two_named_constants(p, expected):
    assert kernel_shap_policy(p) == expected
    assert KERNEL_SHAP_CONFIRM_FEATURES == 200
    assert KERNEL_SHAP_MAX_FEATURES == 1_000


def test_the_kernel_shap_estimate_reproduces_the_measured_point():
    """141 ns/cell is calibrated at p=800: 20.54 s per explained row, measured."""
    est = kernel_shap_cost_estimate(800, n_background=50, n_eval=50)
    assert est["seconds_per_row"] == pytest.approx(20.54, rel=0.05)
    assert est["seconds_per_model"] == pytest.approx(20.54 * 50, rel=0.05)
    # 2.60 GB peak measured at that width.
    assert est["peak_bytes_per_model"] / 1e9 == pytest.approx(2.60, rel=0.10)


def test_the_kernel_shap_notice_quotes_a_price_before_it_spends_it():
    quiet = kernel_shap_availability(100)
    assert quiet["policy"] == "run" and quiet["reason"] is None

    priced = kernel_shap_availability(640, n_models=2)
    assert priced["policy"] == "confirm"
    assert "minutes" in priced["reason"] and "GB per model" in priced["reason"]
    assert priced["override_allowed"] is True

    declined = kernel_shap_availability(1_500, model_label="SVR")
    assert declined["policy"] == "refuse"
    assert "SVR" in declined["reason"]
    assert declined["override_allowed"] is True

    absolute = kernel_shap_availability(KERNEL_SHAP_NO_OVERRIDE_FEATURES)
    assert absolute["override_allowed"] is False
    assert kernel_shap_availability(
        KERNEL_SHAP_NO_OVERRIDE_FEATURES - 1)["override_allowed"] is True


@pytest.mark.parametrize("p,expected", [
    (200, True), (1_000, True), (1_001, False), (20_000, False),
])
def test_permutation_importance_stops_defaulting_on_but_never_refuses(p, expected):
    assert permutation_importance_default_on(p) is expected
    assert PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES == 1_000


def test_the_permutation_estimate_matches_the_measured_fit():
    """101.2 s at p=1,000 and 200.3 s at p=2,000, RandomForest, 10 repeats."""
    assert permutation_importance_cost_estimate(1_000)["seconds_per_model"] == pytest.approx(101.2, rel=0.01)
    assert permutation_importance_cost_estimate(2_000)["seconds_per_model"] == pytest.approx(200.3, rel=0.01)
    assert permutation_importance_cost_estimate(1_000, n_models=3)["seconds_total"] == pytest.approx(303.6, rel=0.01)

    on = permutation_importance_availability(500)
    assert on["default_on"] is True and on["reason"] is None
    off = permutation_importance_availability(4_210, n_models=3)
    assert off["default_on"] is False
    assert "4,210 features" in off["reason"] and "Tick to run" in off["reason"]


def test_the_tree_explainer_guard_reduces_rows_and_never_features():
    """TreeExplainer is flat in p; only the stored result grows."""
    untouched = shap_result_guard(200, 24_000, n_classes=1)
    assert untouched["n_rows"] == 200
    assert untouched["reduced"] is False
    assert untouched["reason"] is None

    guarded = shap_result_guard(500, 20_000, n_classes=10)
    assert guarded["reduced"] is True
    assert guarded["n_rows"] == SHAP_RESULT_CELL_BUDGET // (20_000 * 10)
    assert guarded["n_rows"] < 500
    assert guarded["n_features"] == 20_000
    assert "all 20,000 features were explained" in guarded["reason"]
    assert f"{guarded['n_rows']:,} of the 500" in guarded["reason"]


def test_one_wording_for_the_selection_rule():
    assert variance_subset_phrase(1_000, 12_431) == (
        "the 1,000 highest-variance of 12,431 numeric features")
    assert variance_subset_phrase(200, 12_431, noun="numeric features") == (
        "the 200 highest-variance of 12,431 numeric features")


# ── 3 · the shape the app is actually used at, end to end ────────────────────


def test_a_five_hundred_by_twenty_upload_acquires_no_new_friction():
    """The ordinary case, through the real detector: nothing is capped at all.

    The compute axis exists for omics widths. A 500 x 20 clinical table must
    come out the far side of it indistinguishable from before — no cap engaged,
    no disclosure owed, every display decision unchanged.
    """
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        rng.normal(size=(500, 20)), columns=[f"f{i:02d}" for i in range(20)]
    )
    df.loc[rng.choice(500, 25, replace=False), "f03"] = np.nan  # a real upload has gaps
    df["outcome"] = rng.normal(size=500)
    features = [c for c in df.columns if c != "outcome"]

    r = detect_regime(df, features, "outcome")

    assert (r.n_rows, r.n_features, r.n_numeric, r.n_missing_cols) == (500, 20, 20, 1)

    # No cap fires, and the page therefore renders no disclosure block.
    assert r.compute_regime == "direct"
    assert r.compute_caps_engaged is False
    assert r.dense_pairwise_max_features == 20 == r.n_numeric
    assert r.dense_pairwise_budget_for("spearman") == 20
    assert pairwise_correlation_plan(
        20, has_missing_cells=True, method="spearman")["reason"] is None

    # The on-demand diagnostics all remain available at this shape.
    assert vif_is_defined(20, 500) is True
    assert vif_availability(20, 500)["reason"] is None
    assert ols_diagnostic_is_defined(20, 500) is True
    assert kernel_shap_policy(20) == "run"
    assert permutation_importance_default_on(20) is True

    # And every display decision is what it was before the axis existed.
    assert r.feature_regime == "medium"
    assert r.row_regime == "standard"
    assert r.show_full_corr_matrix is True
    assert r.distribution_mode == "gallery"
    assert r.corr_top_n == 0
    assert r.macro_shape_tiers == ["pca"]


def test_caps_engage_only_on_the_eager_path():
    """`compute_caps_engaged` answers for first paint, and says so.

    It covers work that happens without being asked. It deliberately does not
    cover the rank substitution (which depends on the method pill) or the
    on-demand regression diagnostics (which disclose in their own warnings),
    because a first-paint banner about an analysis nobody ran is the mirror
    image of the silent truncation this axis exists to prevent.
    """
    assert _regime(1_000).compute_caps_engaged is False
    assert _regime(1_001).compute_caps_engaged is True
    assert _regime(60_000).compute_caps_engaged is True

    # 300 numeric columns with gaps: a Spearman session would be substituted,
    # but a Pearson session — the default — reduces nothing, so no banner.
    assert _regime(300, n_missing_cols=12).compute_caps_engaged is False
    assert rank_correlation_substitution_applies(300, True, "spearman") is True

    # Too few rows for VIF is a real refusal, but it belongs to the VIF site.
    narrow_and_short = _regime(20, n_rows=30)
    assert vif_is_defined(20, 30) is False
    assert narrow_and_short.compute_caps_engaged is False


def test_the_rank_cap_is_only_reachable_through_a_rank_method():
    """RANK_CORR_PAIRWISE_MAX_FEATURES must never touch a Pearson session.

    On complete data Spearman is 3-5x FASTER than Pearson, so a cap applied on
    width alone would slow the app down to fix a problem it does not have.
    """
    assert RANK_CORR_PAIRWISE_MAX_FEATURES == 250
    assert dense_pairwise_budget(900, has_missing_cells=True, method="pearson") == 900
    assert dense_pairwise_budget(900, has_missing_cells=False, method="spearman") == 900
    assert dense_pairwise_budget(900, has_missing_cells=True, method="spearman") == 250
    assert VIF_MAX_FEATURES == 200


def test_the_budget_property_gives_the_same_answer_as_the_plan():
    """One question, one answer — the whole reason these live in one module.

    `dense_pairwise_budget_for` used to call `dense_pairwise_budget` directly
    and so answered 250 for a rank method on a frame with gaps, while
    `pairwise_correlation_plan` — the entry point the page actually uses —
    answered 1,000 for the same dataset. The plan applies the rank SUBSTITUTION
    first, which moves the work off the pairwise-complete path and restores the
    ordinary dense budget; the bare predicate cannot see that. Two answers to
    one question inside the module that exists so there is exactly one.
    """
    for p in (100, 250, 251, 900, 3_000, 60_000):
        for missing in (False, True):
            for method in ("pearson", "spearman", "Spearman"):
                r = _regime(p, n_missing_cols=1 if missing else 0)
                assert r.dense_pairwise_budget_for(method) == pairwise_correlation_plan(
                    p, has_missing_cells=missing, method=method
                )["max_features"], (p, missing, method)

    # The case that was wrong, spelled out.
    wide_with_gaps = _regime(3_000, n_missing_cols=1)
    assert wide_with_gaps.dense_pairwise_budget_for("spearman") == 1_000
    # And the rank cap still governs the width at which the substitution starts.
    assert rank_correlation_substitution_applies(3_000, True, "spearman") is True
    assert rank_correlation_substitution_applies(250, True, "spearman") is False
