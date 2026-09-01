"""The Relationships tab used to compute while a different tab was on screen.

Streamlit has no lazy tabs: `st.tabs` runs every body on every rerun, open or
not, and there is no `st.fragment` on `pages/02_EDA.py`. So the correlation
screen in tab 3 was computed on first paint, while tab 1 was what the user was
looking at, and again on every widget touch anywhere on the page that missed
its cache.

Above `ml.regime.PER_COLUMN_SCAN_MAX_FEATURES` — the width where
`DatasetRegime.compute_regime` turns "capped" — that section now waits behind a
button. What this file pins:

* BELOW the threshold nothing changed at all. That is the load-bearing
  constraint: a 500 x 20 upload must not acquire a click it never had, and the
  frozen widget counts in `test_routing_baseline.py` depend on it.
* Above it, the deferral says WHY and names the dataset's width, because a
  section that is simply missing reads as "there was nothing to show".
* A deferral writes NOTHING to the ledger. No cap engaged and no feature was
  dropped; filing a "correlations were not computed" limitation for work the
  user can still ask for would put a false caveat in the Discussion, which is
  the mirror image of the silent truncation the caps exist to prevent.
* The button stores a CONFIG, so changing the method pill re-defers instead of
  serving the old table under a caption written for the new setting.

The threshold is far too wide to build a frame for, so it is lowered the way
the sibling cap tests lower theirs: the page re-executes its imports on every
AppTest run, so a monkeypatched module constant reaches both the gate and the
sentence.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import inject_data_state


def _assert_clean(at, where):
    if at.exception:
        pytest.fail(f"{where} raised: {'; '.join(str(e.value)[:400] for e in at.exception)}")


def _button(at, key):
    for el in at.button:
        if getattr(el, "key", None) == key:
            return el
    return None


def _all_text(at):
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return " ".join(parts)


def _frame(n=200, p=80, seed=3):
    """A correlated outcome so the pair table has something to find."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f"g{i:03d}": rng.normal(0, i + 1, n) for i in range(p)})
    df["glucose"] = df[f"g{p - 1:03d}"] * 0.5 + rng.normal(0, 1, n)
    return df


def _run(df, timeout=90, **state):
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=timeout)
    inject_data_state(at, df)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


# -- below the threshold: no new friction, none at all ----------------------


class TestNarrowUploadsAreUntouched:

    def test_a_500x20_upload_gets_no_button_and_the_table_on_first_paint(self):
        """The constraint that matters most: real thresholds, nothing deferred."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.normal(size=(500, 19)),
                          columns=[f"f{i:02d}" for i in range(19)])
        df["glucose"] = df["f01"] * 2 + rng.normal(size=500)

        at = _run(df)
        _assert_clean(at, "EDA on a 500 x 20 upload")

        assert _button(at, "eda_corr_run") is None, (
            "a narrow dataset acquired a 'Compute feature correlations' click")
        text = _all_text(at)
        assert "is not run unless you ask for it" not in text, (
            "a narrow dataset was told its correlation screen was deferred")
        assert "eda_corr_config" not in at.session_state, (
            "the deferral wrote session state on a dataset it does not govern")

    def test_a_width_that_caps_the_screen_still_does_not_defer_it(self, monkeypatch):
        """The two thresholds are different questions and must stay different.

        The p x p budget fires at 1,000 columns, where the capped construction
        costs 0.38 s measured — not enough to make anyone click. If the gate
        read that constant instead, the capped section would defer itself out
        from under the disclosure it exists to render.
        """
        import ml.regime as regime
        monkeypatch.setattr(regime, "DENSE_PAIRWISE_MAX_FEATURES", 30)

        at = _run(_frame())
        _assert_clean(at, "EDA with the pair screen capped but not deferred")

        assert _button(at, "eda_corr_run") is None, (
            "the deferral gate is reading the pairwise budget, not the "
            "per-column tier")
        assert "highest-variance of" in _all_text(at), (
            "the cap disclosure vanished, which is what deferring it too early "
            "would look like")


# -- above the threshold: deferred, explained, and owing nothing ------------


class TestWideUploadsWaitToBeAsked:

    @pytest.fixture
    def capped(self, monkeypatch):
        """Put an 81-column frame in the 'capped' compute tier.

        Both constants move because `compute_regime` short-circuits to
        "direct" below the pairwise budget; only the per-column one decides
        "capped" from "guarded", which is the boundary under test.
        """
        import ml.regime as regime
        monkeypatch.setattr(regime, "DENSE_PAIRWISE_MAX_FEATURES", 30)
        monkeypatch.setattr(regime, "PER_COLUMN_SCAN_MAX_FEATURES", 50)

    def test_the_screen_does_not_run_and_says_why_with_the_width(self, capped):
        at = _run(_frame())
        _assert_clean(at, "EDA above the per-column scan tier")

        btn = _button(at, "eda_corr_run")
        assert btn is not None, "the heavy section still runs unasked"

        text = _all_text(at)
        assert "80 features" in text, (
            "the deferral does not name the dataset's width")
        assert "Above 50 the correlation screen is not run unless you ask" in text, (
            "the sentence quotes a threshold other than the one the gate uses "
            "— the number on screen has to be the boundary `compute_regime` "
            "turns 'capped' at, or the two drift apart silently")
        assert "no pair has been examined yet" in text, (
            "an absent table can be read as 'no strong pairs'")
        assert "Top 30 correlated pairs" not in text and "Top 50 correlated pairs" not in text

    def test_a_deferral_records_nothing_anywhere(self, capped):
        """Nothing was reduced, so nothing is owed to the manuscript."""
        at = _run(_frame())
        _assert_clean(at, "EDA above the per-column scan tier")

        ledger = at.session_state["insight_ledger"]
        assert ledger.get("eda_cap_corr_pairs") is None, (
            "a section nobody has run yet filed a limitation about itself")
        assert ledger.get("eda_method_spearman_rank_approx") is None
        log = at.session_state["methodology_log"]
        assert not [e for e in log if "pairwise correlations" in e["action"].lower()]

    def test_the_button_computes_the_screen(self, capped):
        at = _run(_frame())
        _button(at, "eda_corr_run").click().run()
        _assert_clean(at, "EDA after asking for the correlation screen")

        text = _all_text(at)
        assert "correlated pairs" in text, "the click produced no table"

    def test_changing_the_method_redefers_instead_of_serving_a_stale_table(
            self, capped):
        """The config tuple, not the bare button, is what makes this safe.

        A plain `if st.button(...)` latch would leave a Pearson table on screen
        under a Spearman pill.
        """
        at = _run(_frame())
        _button(at, "eda_corr_run").click().run()
        assert "correlated pairs" in _all_text(at)

        pills = [p for p in at.pills if getattr(p, "key", None) == "corr_method"]
        assert pills, "the Method pills are gone"
        pills[0].set_value("Spearman").run()
        _assert_clean(at, "EDA after switching the correlation method")

        assert _button(at, "eda_corr_run") is not None, (
            "the section did not re-defer when its inputs changed")
        assert "correlated pairs" not in _all_text(at), (
            "a table computed under Pearson is being shown under a Spearman pill")
