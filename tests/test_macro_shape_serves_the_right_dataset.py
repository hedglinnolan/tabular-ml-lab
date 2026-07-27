"""T0-LIVE-001: the macro-shape plots served the first dataset's results forever.

`compute_pca`, `compute_umap`, `compute_persistence` and `compute_mapper` were
decorated `@st.cache_data`, and their only non-default argument is
`_df_numeric` — the leading underscore is Streamlit's marker for *do not hash
this*. Every call site passed nothing else, so **the cache key was constant**.
`st.cache_data` is process-global, so the first dataset's PCA, UMAP, persistence
diagram and Mapper graph came back for every dataset afterwards, and in the
multi-user Docker deployment, across users.

The fix is not a better key on the engine's cache — it is that the engine has no
cache. Caching belongs to the host, which knows what a dataset is. `pages/02`
already threads a content fingerprint through its own caches for exactly this
reason and now does the same for these four.

Two tests, deliberately at different levels:

* the engine has no cache at all and returns per-dataset results — a unit fact;
* the page's cached wrappers key on content, so two datasets in one session get
  their own answers — the gate as stated.
"""
import numpy as np
import pandas as pd
import pytest


def _frame(seed, n=60, cols=5):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.normal(size=(n, cols)),
                        columns=[f"f{i}" for i in range(cols)])


def test_the_engine_carries_no_cache():
    """A cache in the engine cannot be keyed correctly, because the engine does
    not know what a dataset is. This asserts it has none."""
    import ml.macro_shape as macro

    import re

    src = open(macro.__file__, encoding="utf-8").read()
    # Decorator LINES, not the substring: the file explains in a comment why the
    # decorators are gone, and naming a thing is not doing it.
    assert not re.search(r"^@st\.cache_data", src, re.M), (
        "macro_shape has a Streamlit cache again — T0-LIVE-001 is back")
    assert not re.search(r"^import streamlit", src, re.M), (
        "macro_shape imports the host again")

    for fn in ("compute_pca", "compute_umap", "compute_persistence", "compute_mapper"):
        f = getattr(macro, fn)
        assert not hasattr(f, "clear"), (
            f"{fn} is still wrapped in a Streamlit cache")


def test_two_datasets_get_their_own_pca():
    """The gate, at the engine level: same process, two frames, two answers."""
    from ml.macro_shape import compute_pca

    a = compute_pca(_frame(0))
    b = compute_pca(_frame(1))
    assert a is not b

    va = np.asarray(a["explained_variance_ratio"], dtype=float)
    vb = np.asarray(b["explained_variance_ratio"], dtype=float)
    assert not np.allclose(va, vb), (
        "two different datasets produced identical PCA — the constant-key cache "
        "is back")

    # And the same frame twice is deterministic, so the difference above is the
    # data and not noise.
    again = compute_pca(_frame(0))
    np.testing.assert_allclose(
        np.asarray(again["explained_variance_ratio"], dtype=float), va)


def test_the_page_cache_keys_on_content_not_shape():
    """The host's cache must miss when the rows change.

    Shape and column names are not enough: two cohort runs of one study have
    identical shape and columns and different rows. This is the fingerprint
    `pages/02` computes, checked directly — an AppTest would exercise the whole
    page for one property.
    """
    lines = open("pages/02_EDA.py", encoding="utf-8").read().splitlines()
    # The macro-shape wrappers now delegate to the page's single
    # `_content_fingerprint` (`T0-LIVE-005`), so that is the function to check —
    # extracting `_macro_fp` alone would leave its callee undefined.
    start = next((i for i, l in enumerate(lines)
                  if l.startswith("def _content_fingerprint(")), None)
    assert start is not None, "pages/02 lost the content fingerprint helper"
    # Scan to the next top-level statement rather than to a blank line — the
    # docstring contains blank lines, and stopping at the first one truncates
    # the function mid-string.
    end = next((j for j in range(start + 1, len(lines))
                if lines[j] and not lines[j][0].isspace()), len(lines))

    ns = {"pd": pd}
    exec(compile("\n".join(lines[start:end]), "_content_fingerprint", "exec"), ns)
    fp = ns["_content_fingerprint"]

    a, b = _frame(0), _frame(1)
    assert a.shape == b.shape and list(a.columns) == list(b.columns)
    assert fp(a) != fp(b), (
        "the fingerprint ignores the rows, so two same-shaped datasets share a "
        "cache entry — which is T0-LIVE-001 moved rather than fixed")
    assert fp(a) == fp(_frame(0)), "the fingerprint is not stable for one dataset"
