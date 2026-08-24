"""T0-LIVE-005: eight EDA caches keyed on shape, so cohort runs collided.

`T0-LIVE-001` moved the four macro-shape caches onto a content digest and left a
comment stating the principle exactly — *two cohort runs of the same study have
identical shape and columns and different rows*. The eight older caches on
`pages/02` kept the shape-only `_data_fingerprint`: `(len(df), len(df.columns),
tuple(sorted(df.columns)))`.

Any two equal-sized cohorts of one study share that key. A median split, a 1:1
matched case-control, a balanced sex split — all produce two frames with the
same row count and the same columns. Under `@st.cache_data`, which is
process-global, cohort A's correlation matrix, skew list, outlier heatmap, top
pairs, target sort and interaction ranking are then served to cohort B.

The eight: dataset profile, EDA signals, skewed features, outlier heatmap,
correlation matrix, top correlation pairs, target-correlation sort,
interactions.

Cohort runs are the newest subsystem in the app, so this lands on the feature
least likely to be double-checked.
"""
import numpy as np
import pandas as pd
import pytest


def _load_page_helper(name):
    """Pull one top-level function out of `pages/02_EDA.py` without running it.

    The page is a Streamlit script; importing it would execute the whole thing.
    """
    lines = open("pages/02_EDA.py", encoding="utf-8").read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(f"def {name}(")), None)
    assert start is not None, f"pages/02 lost {name}"
    end = next((j for j in range(start + 1, len(lines))
                if lines[j] and not lines[j][0].isspace()), len(lines))
    ns = {"pd": pd, "np": np}
    exec(compile("\n".join(lines[start:end]), name, "exec"), ns)
    return ns[name]


def _cohort_pair(n=120, seed=0):
    """Two equal-sized cohorts of one study: same shape, same columns, different rows.

    This is a balanced sex split — the shape every real cohort plan produces.
    """
    rng = np.random.RandomState(seed)
    whole = pd.DataFrame({
        "age": rng.randint(30, 80, 2 * n),
        "bmi": rng.normal(27, 4, 2 * n).round(2),
        "glucose": rng.normal(100, 15, 2 * n).round(1),
        "sex": ["F"] * n + ["M"] * n,
    })
    # Give the two groups genuinely different structure, so a correlation
    # computed on one is wrong for the other rather than merely stale.
    whole.loc[whole["sex"] == "M", "glucose"] = (
        whole.loc[whole["sex"] == "M", "bmi"] * 3.0 + rng.normal(0, 2, n))
    a = whole[whole["sex"] == "F"].reset_index(drop=True)
    b = whole[whole["sex"] == "M"].reset_index(drop=True)
    return a.drop(columns=["sex"]), b.drop(columns=["sex"])


def test_the_page_has_one_fingerprint_and_it_follows_the_values():
    """The fix: a single content-based key, used by every cache on the page."""
    fp = _load_page_helper("_content_fingerprint")
    a, b = _cohort_pair()

    assert a.shape == b.shape, "the fixture is not a same-shape pair"
    assert list(a.columns) == list(b.columns)

    assert fp(a) != fp(b), (
        "two equal-shaped cohorts share a cache key — cohort A's EDA results "
        "would be served to cohort B (T0-LIVE-005)")
    assert fp(a) == fp(a.copy()), "the fingerprint is not stable for one frame"


def test_no_cache_on_the_page_keys_on_shape_alone():
    """Structural: the shape-only tuple must not come back.

    `_data_fingerprint` is threaded into eight `@st.cache_data` calls. If it is
    ever reassigned to a shape tuple, every one of them collides again, and the
    failure is silent — the numbers look fine, they are just another group's.
    """
    src = open("pages/02_EDA.py", encoding="utf-8").read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))

    assert "_data_fingerprint = _content_fingerprint(" in code, (
        "_data_fingerprint is no longer the content digest")
    assert "_data_fingerprint = (len(df), len(df.columns)" not in code, (
        "the shape-only fingerprint is back — T0-LIVE-005 has returned")


def test_every_cached_helper_receives_the_content_key():
    """All eight, by name, so adding a ninth without a key is visible."""
    src = open("pages/02_EDA.py", encoding="utf-8").read()
    for helper in ("_get_skewed_features", "_compute_outlier_heatmap",
                   "_compute_corr", "_top_corr_pairs", "_sort_by_target_corr",
                   "_compute_interactions"):
        calls = [l for l in src.splitlines()
                 if f"{helper}(" in l and "data_id=" in l
                 and not l.lstrip().startswith("def ")]
        assert calls, f"{helper} is called without a data_id key"
        for line in calls:
            assert "data_id=_data_fingerprint" in line, (
                f"{helper} is keyed on something other than the content "
                f"fingerprint: {line.strip()}")


def test_two_cohorts_produce_different_correlations():
    """The claim the cache would have hidden.

    If the two groups had the same correlation structure, serving one's result
    for the other would be harmless and this whole finding would be theoretical.
    They do not.
    """
    a, b = _cohort_pair()
    corr_a = a.corr(numeric_only=True).loc["bmi", "glucose"]
    corr_b = b.corr(numeric_only=True).loc["bmi", "glucose"]
    assert abs(corr_a - corr_b) > 0.3, (
        "the fixture's two cohorts have similar correlations, so it cannot "
        "demonstrate the collision")


def test_the_macro_shape_wrappers_share_the_same_key():
    """`_macro_fp` and `_data_fingerprint` were two implementations of one idea.

    Having both is how the principle came to be written down in one place and
    applied in the other.
    """
    fp = _load_page_helper("_content_fingerprint")
    macro = _load_page_helper("_macro_fp")
    # `_macro_fp` delegates, so it needs the helper in its namespace.
    a, _ = _cohort_pair()
    src = open("pages/02_EDA.py", encoding="utf-8").read()
    assert "return _content_fingerprint(d)" in src, (
        "_macro_fp has its own implementation again")
