"""The top-pairs table built the whole p x p matrix, then printed 30 rows.

`_top_corr_pairs` on `pages/02_EDA.py` materialized a p x p correlation matrix
plus the triu index arrays to take the 30-50 strongest pairs: 28p^2 bytes, so
27 MB at 1,000 columns, 2.7 GB at 10,000, 94 GB at 60,000 — in a block that
carried no try/except, which makes the wide case an unhandled MemoryError
mid-render rather than a degraded table.

It is now capped at `ml.regime.DENSE_PAIRWISE_MAX_FEATURES` with a variance
pre-screen. A cap that changes which features were analyzed and does not say so
would put a Methods sentence in the manuscript describing an analysis that did
not happen, so this file tests the disclosure as hard as the arithmetic:

* uncapped widths must be untouched — same pairs, same order, no caption, no
  ledger entry, nothing owed;
* the capped path must keep the OUTCOME column whatever its variance, or every
  feature-outcome pair silently leaves the table;
* the sentence the user reads must be the one `ml/regime.py` composes, not a
  second copy written on the page.
"""
import re
import textwrap

import numpy as np
import pandas as pd
import pytest

from ml.regime import (
    DENSE_PAIRWISE_MAX_FEATURES,
    RANK_CORR_PAIRWISE_MAX_FEATURES,
    pairwise_correlation_plan,
)

PAGE = "pages/02_EDA.py"


def _load_nested(name, path=PAGE):
    """Pull one INDENTED, decorated helper out of a Streamlit page.

    `_load_page_helper` in `test_eda_caches_follow_the_data.py` only reaches
    top-level `def`s. This one lives inside a `with` block under a decorator,
    so it is sliced by indentation and dedented; the decorator line is left
    behind because `st.cache_data` needs a script run.
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.strip().startswith(f"def {name}(")), None)
    assert start is not None, f"{path} lost {name}"
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].strip()
                and (len(lines[j]) - len(lines[j].lstrip())) <= indent), len(lines))
    ns = {"pd": pd, "np": np}
    exec(compile(textwrap.dedent("\n".join(lines[start:end])), name, "exec"), ns)
    return ns[name]


def _reference_pairs(df, features, method, n):
    """The algorithm as it stood before the cap — the thing not to change."""
    corr = df[features].corr(method=method).values
    idx_upper = np.triu_indices_from(corr, k=1)
    vals = corr[idx_upper]
    top_idx = np.argsort(np.abs(vals))[-n:][::-1]
    return pd.DataFrame([
        {
            "Feature A": features[idx_upper[0][i]],
            "Feature B": features[idx_upper[1][i]],
            "Correlation": round(float(vals[i]), 3),
        }
        for i in top_idx
    ])


def _graded_frame(p=40, n=200, seed=0):
    """Columns whose variance rises with their index, and a flat outcome.

    The outcome is the lowest-variance column in the frame by two orders of
    magnitude, which is the omics case: a clinical outcome sitting beside
    expression columns that swamp it on spread. It is also strongly correlated
    with the WIDEST column, so if the screen ranks it out, the single most
    interesting pair in the table disappears.
    """
    rng = np.random.RandomState(seed)
    cols = {f"f{i:02d}": rng.normal(0, i + 1, n) for i in range(p)}
    df = pd.DataFrame(cols)
    df["outcome"] = df[f"f{p - 1:02d}"] * 0.001 + rng.normal(0, 0.0005, n)
    return df


# -- the uncapped width, which must be indistinguishable from before ---------


def test_a_five_hundred_by_twenty_upload_gets_the_same_table_as_before():
    """The ordinary clinical shape: no screen, no caption, no ledger entry."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.normal(size=(500, 20)),
                      columns=[f"f{i:02d}" for i in range(20)])
    df.loc[rng.choice(500, 25, replace=False), "f03"] = np.nan
    df["outcome"] = df["f01"] * 2 + rng.normal(size=500)
    cols = list(df.columns)

    plan = pairwise_correlation_plan(len(cols), has_missing_cells=True, method="pearson")
    assert plan["capped"] is False
    assert plan["reason"] is None, "a caveat about an analysis nobody reduced"
    assert plan["max_features"] == len(cols)

    top = _load_nested("_top_corr_pairs")
    got = top(df, cols, "pearson", 30, max_features=plan["max_features"],
              target="outcome", data_id="x")
    expected = _reference_pairs(df, cols, "pearson", 30)
    pd.testing.assert_frame_equal(got, expected)


def test_the_budget_argument_does_nothing_until_the_budget_is_exceeded():
    df = _graded_frame(p=40)
    cols = list(df.columns)
    top = _load_nested("_top_corr_pairs")
    pd.testing.assert_frame_equal(
        top(df, cols, "pearson", 10, max_features=len(cols), target="outcome", data_id="x"),
        top(df, cols, "pearson", 10, max_features=None, target=None, data_id="x"),
    )


# -- the capped width --------------------------------------------------------


def test_the_screen_keeps_the_widest_columns_and_reports_only_those():
    df = _graded_frame(p=40)
    cols = list(df.columns)
    top = _load_nested("_top_corr_pairs")

    got = top(df, cols, "pearson", 20, max_features=10, target="outcome", data_id="x")

    named = set(got["Feature A"]) | set(got["Feature B"])
    # The rule, read off the same sample variances the screen ranks on — the
    # construction order is not the ranking, since at n=200 neighbouring columns
    # trade places.
    ranked = df[[c for c in cols if c != "outcome"]].var().sort_values(ascending=False)
    kept = set(ranked.index[:9]) | {"outcome"}
    assert named <= kept, f"the table names columns the screen dropped: {named - kept}"
    # 10 kept columns is 45 pairs, so a 20-row request is really answered.
    assert len(got) == 20


def test_the_outcome_survives_the_screen_that_its_variance_would_have_lost():
    """Force-retention, and the pair that proves why it is not optional."""
    df = _graded_frame(p=40)
    cols = list(df.columns)

    ranked = df[cols].var().sort_values(ascending=False).index.tolist()
    assert "outcome" not in ranked[:10], (
        "the fixture's outcome would survive a plain variance ranking, so it "
        "cannot demonstrate the eviction this guard exists to prevent")

    top = _load_nested("_top_corr_pairs")
    got = top(df, cols, "pearson", 5, max_features=10, target="outcome", data_id="x")

    assert "outcome" in set(got["Feature A"]) | set(got["Feature B"]), (
        "every feature-outcome pair was dropped from the table")
    strongest = got.iloc[0]
    assert {strongest["Feature A"], strongest["Feature B"]} == {"f39", "outcome"}
    assert abs(strongest["Correlation"]) > 0.8


def test_the_screen_keeps_exactly_the_budget_it_was_given():
    """The construction is what costs 28p^2 bytes, so the count is the point."""
    df = _graded_frame(p=60)
    cols = list(df.columns)
    top = _load_nested("_top_corr_pairs")

    for budget in (10, 25, 41):
        got = top(df, cols, "pearson", 10_000, max_features=budget,
                  target="outcome", data_id="x")
        assert len(got) == budget * (budget - 1) // 2, (
            f"budget {budget} did not produce a {budget}-column matrix")


def test_asking_for_zero_pairs_does_not_return_every_pair():
    """`np.argsort(...)[-0:]` is the whole array, not an empty slice.

    Unreachable while `corr_top_n` is 30 or 50, and pinned anyway: this is the
    line `ml/regime.py` names as the reason the compute axis was kept off the
    display ladder.
    """
    df = _graded_frame(p=12)
    top = _load_nested("_top_corr_pairs")
    got = top(df, list(df.columns), "pearson", 0, max_features=None,
              target=None, data_id="x")
    assert len(got) == 1


# -- the rank substitution ---------------------------------------------------


def test_the_rank_substitution_is_the_statistic_it_claims_to_be():
    """Pearson-of-ranks against pandas' own Spearman, on complete data."""
    df = _graded_frame(p=12)
    cols = list(df.columns)
    top = _load_nested("_top_corr_pairs")

    substituted = top(df, cols, "spearman_on_ranks", 20, max_features=None,
                      target=None, data_id="x")
    direct = top(df, cols, "spearman", 20, max_features=None, target=None, data_id="x")
    pd.testing.assert_frame_equal(substituted, direct)


def test_the_substitution_stays_close_when_cells_are_missing():
    """Where it is NOT exact — which is why the page discloses it."""
    rng = np.random.RandomState(3)
    df = _graded_frame(p=12)
    block = df.to_numpy()
    block[rng.random_sample(block.shape) < 0.05] = np.nan
    df = pd.DataFrame(block, columns=df.columns)
    cols = list(df.columns)
    top = _load_nested("_top_corr_pairs")

    a = top(df, cols, "spearman_on_ranks", 66, max_features=None, target=None, data_id="x")
    b = top(df, cols, "spearman", 66, max_features=None, target=None, data_id="x")
    key = lambda t: t.assign(k=t["Feature A"] + "|" + t["Feature B"]).set_index("k")["Correlation"]
    diff = (key(a) - key(b)).abs().max()
    assert diff < 0.05, f"the rank identity drifted by {diff:g}"


# -- the call site: disclosure, ledger, and the guard ------------------------


def _relationships_block():
    """The Relationships tab's CODE — comment lines dropped.

    The block carries long comments that quote the very wording these tests
    assert is absent from the page (the cap sentence belongs to `ml/regime.py`),
    so a check run over the comments would report a violation for explaining the
    rule. Same idiom as `test_eda_caches_follow_the_data.py`.
    """
    src = open(PAGE, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("with _eda_tabs[2]:"))
    end = next(i for i, l in enumerate(src)
               if i > start and l.strip().startswith("# -- Target Relationship Gallery"))
    return "\n".join(l for l in src[start:end] if not l.lstrip().startswith("#"))


def test_the_page_asks_the_engine_for_the_budget_and_the_sentence():
    block = _relationships_block()
    assert "pairwise_correlation_plan(" in block
    assert "_corr_plan['reason']" in block or '_corr_plan["reason"]' in block, (
        "the page is not rendering the engine's sentence")
    assert "highest-variance of" not in block, (
        "the page composes its own cap sentence — a second copy of the wording "
        "ml/regime.py owns (see its module comment)")


def test_the_plan_is_asked_about_the_columns_that_are_actually_correlated():
    """`n_numeric` misses the target the block appends one line earlier.

    At exactly the budget that one column decides whether a cap fires, so the
    width handed to the engine has to be the width of the construction.
    """
    block = _relationships_block()
    assert re.search(r"pairwise_correlation_plan\(\s*len\(corr_cols\)", block), (
        "the plan is keyed on something other than the real construction width")
    assert "regime.dense_pairwise_max_features" not in block, (
        "the site reads the n_numeric property, which is one column short")


def test_the_pair_table_is_no_longer_an_unhandled_memory_error():
    block = _relationships_block()
    assert "try:" in block and "except MemoryError:" in block, (
        "the p x p construction is still unguarded")
    call = next(l for l in block.splitlines() if "pairs_df = _top_corr_pairs(" in l)
    assert "max_features=" in call and "target=" in call, (
        "the budget is not passed as a hashed cache argument, so a cached table "
        "can be served beside a caption written for a different budget")
    assert "data_id=_data_fingerprint" in call


def test_a_cap_that_engages_reaches_the_ledger_unresolved():
    block = _relationships_block()
    assert 'id="eda_cap_corr_pairs"' in block
    assert 'id="eda_method_spearman_rank_approx"' in block
    assert "manuscript_text=" in block
    assert "resolved=True" not in block, (
        "a resolved insight never reaches discussion_points_for_manuscript()")
    # Both writes are conditional: an uncapped narrow dataset files nothing.
    assert 'if _corr_plan["capped"]:' in block
    assert 'if _corr_plan["rank_substitution"]:' in block


@pytest.mark.parametrize("p,method,missing,expect_capped", [
    (20, "pearson", True, False),
    (900, "pearson", True, False),
    (900, "spearman", False, False),
    # Substituted onto the Pearson path, so the 250-column rank cap never
    # applies and nothing is dropped — only the method is disclosed.
    (900, "spearman", True, False),
    (DENSE_PAIRWISE_MAX_FEATURES, "pearson", False, False),
    (DENSE_PAIRWISE_MAX_FEATURES + 1, "pearson", False, True),
    (RANK_CORR_PAIRWISE_MAX_FEATURES, "spearman", True, False),
])
def test_the_site_caps_where_the_engine_says_and_nowhere_else(
        p, method, missing, expect_capped):
    """The widths the page will actually hand over, end to end.

    A Spearman session with gaps is substituted onto the Pearson path first, so
    at p=900 it is the substitution that fires and no columns are dropped.
    """
    plan = pairwise_correlation_plan(p, has_missing_cells=missing, method=method)
    assert plan["capped"] is expect_capped
    assert (plan["reason"] is not None) is expect_capped
