"""
Tier 2: the two Relationships-tab tools that a plain page render never exercises.

`test_eda_page` only proves the page imports and draws. The k-means block is
behind a button, so nothing in the existing suite touches the fit, the
permutation null, the profile heatmap, or the target association — the code
paths where a bad column reference or a shape mismatch actually lives.

The Feature Explorer tests pin two things that were previously wrong: the
scatter defaulted to no coloring at all, and it inferred continuous-vs-discrete
color from dtype, so a classification target coded 0/1 drew a continuous
colorbar instead of one color per class.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)


def _assert_clean(at, where):
    if at.exception:
        pytest.fail(f"{where} raised: {'; '.join(str(e.value)[:400] for e in at.exception)}")


def _widget(at, kind, key):
    for el in getattr(at, kind):
        if getattr(el, "key", None) == key:
            return el
    return None


def _run_eda(df, timeout=90, **inject):
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=timeout)
    inject_data_state(at, df, **inject)
    at.run()
    return at


def _run_eda_clf(df, timeout=90):
    return _run_eda(df, timeout, target_col="condition", task_type="classification")


# ── Feature Explorer ─────────────────────────────────────────────────

class TestFeatureExplorer:

    def test_color_defaults_to_the_target(self):
        """The point of the control is seeing where the outcome sits."""
        at = _run_eda(build_test_dataframe())
        _assert_clean(at, "EDA")
        color = _widget(at, "selectbox", "fe_color")
        assert color is not None, "Feature Explorer 'Color by' selectbox is missing"
        assert color.value != "None", "'Color by' should default to the target, not None"
        assert color.value == at.session_state["data_config"].target_col

    def test_same_column_on_both_axes_does_not_crash(self):
        """df[[x, y]] duplicates the column when x == y and plotly rejects it."""
        at = _run_eda(build_test_dataframe())
        x = _widget(at, "selectbox", "fe_x")
        y = _widget(at, "selectbox", "fe_y")
        assert x is not None and y is not None
        y.set_value(x.value).run()
        _assert_clean(at, "Feature Explorer with x == y")

    def test_classification_target_is_the_default_color(self):
        """The 0/1 case renders cleanly and still defaults to the target.

        Whether it draws discrete colors is pinned by the unit tests for
        utils.column_utils.color_by_category — AppTest cannot read back a
        plotly figure in this Streamlit version.
        """
        at = _run_eda_clf(build_classification_dataframe())
        _assert_clean(at, "EDA (classification)")
        color = _widget(at, "selectbox", "fe_color")
        assert color is not None and color.value == "condition"


# ── k-means cluster structure ────────────────────────────────────────

class TestClusterStructure:

    def test_block_renders_with_controls(self):
        at = _run_eda(build_test_dataframe())
        _assert_clean(at, "EDA")
        assert _widget(at, "multiselect", "eda_km_feats") is not None
        assert any(b.key == "eda_km_run" for b in at.button), "no 'Explore cluster structure' button"

    def test_full_run_regression_target(self):
        """Drive the button and walk every downstream surface."""
        at = _run_eda(build_test_dataframe(), timeout=240)
        run_btn = next(b for b in at.button if b.key == "eda_km_run")
        run_btn.click().run()
        _assert_clean(at, "k-means run (regression)")

        assert "eda_km_config" in at.session_state
        k_pick = _widget(at, "selectbox", "eda_km_k")
        assert k_pick is not None, "k selector did not appear after the run"

        text = " ".join(
            str(getattr(el, "value", ""))
            for attr in ("markdown", "caption", "success", "warning", "info")
            for el in getattr(at, attr, [])
        ).lower()
        assert "shuffled" in text, "the no-structure baseline was never mentioned"
        assert "seed stability" in text or "adjusted rand" in text

        # Methodology has to record the run for the report to be honest.
        log = at.session_state["methodology_log"] if "methodology_log" in at.session_state else []
        actions = [e.get("action", "") for e in log]
        assert any("k-means" in a for a in actions), f"run not logged: {actions}"

    def test_full_run_classification_target(self):
        at = _run_eda_clf(build_classification_dataframe(), timeout=240)
        next(b for b in at.button if b.key == "eda_km_run").click().run()
        _assert_clean(at, "k-means run (classification)")
        assert _widget(at, "selectbox", "eda_km_k") is not None

    def test_changing_k_reruns_cleanly(self):
        at = _run_eda(build_test_dataframe(), timeout=240)
        next(b for b in at.button if b.key == "eda_km_run").click().run()
        k_pick = _widget(at, "selectbox", "eda_km_k")
        # AppTest reports options as strings; the widget hands back the real
        # option object, so compare on str.
        options = [str(o) for o in k_pick.options]
        if len(options) > 1:
            other = next(o for o in options if o != str(k_pick.value))
            k_pick.set_value(int(other)).run()
            _assert_clean(at, "k-means after changing k")
            assert str(_widget(at, "selectbox", "eda_km_k").value) == other

    def test_deselecting_features_below_two_is_refused(self):
        at = _run_eda(build_test_dataframe(), timeout=120)
        feats = _widget(at, "multiselect", "eda_km_feats")
        feats.set_value([feats.value[0]]).run()
        next(b for b in at.button if b.key == "eda_km_run").click().run()
        _assert_clean(at, "k-means with one feature")
        warnings = " ".join(str(w.value).lower() for w in at.warning)
        assert "at least 2 features" in warnings


class TestClusterStructureOnNoise:
    """The load-bearing behavior: refusing to name clusters that are not there."""

    def test_uniform_noise_reports_no_structure(self):
        rng = np.random.default_rng(11)
        n = 400
        df = pd.DataFrame(rng.uniform(0, 1, (n, 5)), columns=[f"v{i}" for i in range(5)])
        # The fixture helper expects a 'glucose' target; without it the page
        # hits its stale-config guard and reruns before rendering anything.
        df["glucose"] = rng.normal(0, 1, n)

        at = _run_eda(df, timeout=240)
        next(b for b in at.button if b.key == "eda_km_run").click().run()
        _assert_clean(at, "k-means on uniform noise")

        warnings = " ".join(str(w.value).lower() for w in at.warning)
        assert "no evidence of cluster structure" in warnings, (
            "k-means on uniform noise must not be presented as discovered subgroups"
        )
        # And it must not have written a structure insight into the ledger.
        from utils.insight_ledger import get_ledger
        ids = {i.id for i in get_ledger().insights}
        assert "eda_kmeans_structure" not in ids


class TestVifResolvesTheCollinearityInsight:
    """Running VIF must close the collinearity clusters the page detected.

    The Deep Dive teardown removed _resolve_insights_from_eda_result and
    replaced it with a plain upsert of any insights an action returns. Upsert
    is not resolve, and multicollinearity_vif returns no insights, so the
    eda_corr_cluster_* warning stayed open after the user ran the very
    diagnostic that answers it — and reached the manuscript as a limitation.
    """

    @staticmethod
    def _collinear(n=300, seed=7):
        rng = np.random.default_rng(seed)
        bmi = rng.normal(27, 5, n)
        return pd.DataFrame({
            "bmi": bmi,
            "weight": bmi * 2.9 + rng.normal(0, 0.01, n),   # r ~ 1.00
            "waist": bmi * 2.4 + rng.normal(0, 0.01, n),
            "age": rng.normal(50, 12, n),
            "glucose": rng.normal(100, 15, n),
        })

    def _cluster_insights(self, at):
        return [
            (i.id, i.resolved)
            for i in at.session_state["insight_ledger"].insights
            if i.id.startswith("eda_corr_cluster_")
        ]

    def test_vif_run_resolves_it(self):
        at = _run_eda(self._collinear(), timeout=180)
        _assert_clean(at, "EDA with collinear features")

        before = self._cluster_insights(at)
        assert before, "no collinearity cluster insight was detected to begin with"
        assert all(not resolved for _, resolved in before)

        vif = [b for b in at.button if b.key == "run_multicollinearity_vif"]
        assert vif, "VIF button not found"
        vif[0].click().run()
        _assert_clean(at, "after running VIF")

        after = self._cluster_insights(at)
        assert after and all(resolved for _, resolved in after), (
            f"VIF did not resolve the collinearity insight: {after}"
        )

    def test_resolution_records_what_answered_it(self):
        at = _run_eda(self._collinear(), timeout=180)
        next(b for b in at.button if b.key == "run_multicollinearity_vif").click().run()
        resolved = [
            i for i in at.session_state["insight_ledger"].insights
            if i.id.startswith("eda_corr_cluster_") and i.resolved
        ]
        assert resolved
        assert "VIF" in (resolved[0].resolved_by or ""), resolved[0].resolved_by
        assert resolved[0].resolved_on_page == "02_EDA"


class TestClusterPlotsAlwaysRender:
    """The verdict is a sentence, not a gate.

    "No evidence of cluster structure" must never suppress the plots. A user
    who asked to see the clusters gets to see them and judge for themselves —
    the honest reading is delivered alongside the picture, not instead of it.
    """

    # AppTest reports .key as None for charts; the key is the proto.id suffix.
    @staticmethod
    def _chart_keys(at):
        return {
            str(getattr(getattr(c, "proto", None), "id", "")).rsplit("-", 1)[-1]
            for c in at.get("plotly_chart")
        }

    @staticmethod
    def _structured(n=400, seed=1):
        rng = np.random.default_rng(seed)
        centers = np.array([[0, 0, 0], [7, 7, 0], [0, 7, 7]])
        grp = rng.integers(0, 3, n)
        X = centers[grp] + rng.normal(0, 1.0, (n, 3))
        df = pd.DataFrame(X, columns=["ldl", "hdl", "crp"])
        df["glucose"] = 90 + 12 * grp + rng.normal(0, 4, n)
        return df

    @staticmethod
    def _noise(n=400, seed=2):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.uniform(0, 1, (n, 4)), columns=["v1", "v2", "v3", "v4"])
        df["glucose"] = rng.normal(100, 15, n)
        return df

    def _run(self, df):
        at = _run_eda(df, timeout=300)
        _assert_clean(at, "EDA before clustering")
        next(b for b in at.button if b.key == "eda_km_run").click().run()
        _assert_clean(at, "EDA after clustering")
        return at

    def test_structured_data_reports_structure_and_draws_everything(self):
        at = self._run(self._structured())
        keys = self._chart_keys(at)
        for expected in ("fig_eda_kmeans_sweep", "fig_eda_kmeans_scatter",
                         "fig_eda_kmeans_knife", "fig_eda_kmeans_profile"):
            assert expected in keys, f"{expected} did not render"
        blurb = " ".join(str(s.value) for s in at.success)
        assert "Strongest structure at k" in blurb

    def test_pure_noise_still_draws_the_projection(self):
        """The load-bearing case: refused verdict, plots still there."""
        at = self._run(self._noise())
        warnings = " ".join(str(w.value) for w in at.warning)
        assert "No evidence of cluster structure" in warnings, (
            "uniform noise should not be reported as structure"
        )
        keys = self._chart_keys(at)
        assert "fig_eda_kmeans_scatter" in keys, (
            "the PCA projection was suppressed when no structure was found — "
            "the verdict must not gate the plot"
        )
        for expected in ("fig_eda_kmeans_sweep", "fig_eda_kmeans_knife",
                         "fig_eda_kmeans_profile"):
            assert expected in keys, f"{expected} did not render on noise"

    def test_k_selector_is_usable_even_with_no_recommendation(self):
        """With no recommended k the selector still offers the full sweep."""
        at = self._run(self._noise())
        k_sel = _widget(at, "selectbox", "eda_km_k")
        assert k_sel is not None
        assert len(list(k_sel.options)) >= 2
        assert k_sel.value is not None
