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


# ── The p x p correlation screen, and what it says when it caps ──────

class TestCorrelationScreenDisclosure:
    """The pair table is capped now; the manuscript has to hear about it.

    `_top_corr_pairs` built a full p x p matrix — 28p^2 bytes, so 2.7 GB at
    10,000 columns — to print 30 rows. It is capped at
    `ml.regime.DENSE_PAIRWISE_MAX_FEATURES` with a variance pre-screen, and a
    reduction that changes which features were analyzed without saying so would
    write a Methods section describing an analysis that did not happen.

    The caps engage at widths too large to render inside a test suite, so the
    threshold is lowered rather than the dataset grown: the constant is read at
    call time by `ml.regime.dense_pairwise_budget`, and the code path exercised
    is the one a 12,000-column upload takes.
    """

    @staticmethod
    def _all_text(at):
        parts = []
        for attr in ("markdown", "caption", "info", "warning", "error", "success"):
            for el in getattr(at, attr, []):
                parts.append(str(getattr(el, "value", "")))
        return " ".join(parts)

    @staticmethod
    def _ledger(at):
        try:
            return at.session_state["insight_ledger"]
        except Exception:
            return None

    @staticmethod
    def _graded_frame(n=200, p=80, seed=1, missing=0.0):
        """Variance rising with column index, and a flat outcome beside it.

        The outcome is the lowest-variance column by three orders of magnitude
        — the omics shape — and is strongly correlated with the WIDEST column,
        so a screen that ranks it out loses the most interesting pair in the
        table.
        """
        rng = np.random.RandomState(seed)
        df = pd.DataFrame({f"g{i:03d}": rng.normal(0, i + 1, n) for i in range(p)})
        df["glucose"] = df[f"g{p - 1:03d}"] * 0.001 + rng.normal(0, 0.0005, n)
        if missing:
            block = df.to_numpy()
            block[rng.random_sample(block.shape) < missing] = np.nan
            df = pd.DataFrame(block, columns=df.columns)
        return df

    def test_a_narrow_upload_acquires_no_new_friction(self):
        """500 x 20, real thresholds: no screen, no caption, no ledger entry."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.normal(size=(500, 19)),
                          columns=[f"f{i:02d}" for i in range(19)])
        df.loc[rng.choice(500, 25, replace=False), "f03"] = np.nan
        df["glucose"] = df["f01"] * 2 + rng.normal(size=500)

        at = _run_eda(df)
        _assert_clean(at, "EDA on a 500 x 20 upload")

        text = self._all_text(at)
        assert "highest-variance of" not in text, (
            "a narrow dataset was told about a cap that did not engage")
        ledger = self._ledger(at)
        assert ledger is not None
        assert ledger.get("eda_cap_corr_pairs") is None
        assert ledger.get("eda_method_spearman_rank_approx") is None
        log = at.session_state["methodology_log"]
        assert not [e for e in log if "Screened pairwise correlations" in e["action"]]

    def test_the_screen_that_engages_says_so_on_screen_and_in_the_record(self, monkeypatch):
        import ml.regime as regime
        monkeypatch.setattr(regime, "DENSE_PAIRWISE_MAX_FEATURES", 30)

        at = _run_eda(self._graded_frame())
        _assert_clean(at, "EDA with the pair screen capped")

        text = self._all_text(at)
        assert "the 30 highest-variance of 81 numeric features" in text
        assert "a stronger pair may exist among them" in text, (
            "the notice implies the printed pairs are the dataset's strongest")
        assert "glucose" in text and "kept in the screen" in text, (
            "nothing says the outcome survived a variance ranking it would lose")

        insight = self._ledger(at).get("eda_cap_corr_pairs")
        assert insight is not None, "the cap never reached the ledger"
        assert insight.resolved is False, (
            "a resolved insight is skipped by discussion_points_for_manuscript()")
        assert insight.metadata["n_screened"] == 30
        assert insight.metadata["n_total"] == 81
        assert insight.metadata["selection_rule"] == "variance"
        assert insight.metadata["target_retained"] is True

        limitations = self._ledger(at).discussion_points_for_manuscript()["limitations"]
        assert any("screened among the 30 highest-variance" in s for s in limitations), (
            "the reduction does not reach the manuscript")

        log = at.session_state["methodology_log"]
        assert [e for e in log if "Screened pairwise correlations on 30 of 81" in e["action"]]

    def test_the_rank_substitution_is_disclosed_where_it_is_chosen(self, monkeypatch):
        import ml.regime as regime
        monkeypatch.setattr(regime, "RANK_CORR_PAIRWISE_MAX_FEATURES", 20)

        at = AppTest.from_file("pages/02_EDA.py", default_timeout=90)
        inject_data_state(at, self._graded_frame(missing=0.03))
        at.session_state["corr_method"] = "Spearman"
        at.run()
        _assert_clean(at, "EDA with Spearman on a frame with gaps")

        text = self._all_text(at)
        assert "Pearson correlation of column ranks" in text
        assert "% of cells are missing" in text

        insight = self._ledger(at).get("eda_method_spearman_rank_approx")
        assert insight is not None and insight.resolved is False
        assert insight.manuscript_text.startswith("Spearman correlations were computed")
        limitations = self._ledger(at).discussion_points_for_manuscript()["limitations"]
        assert any("column ranks" in s for s in limitations)

    def test_a_correlation_that_runs_out_of_memory_does_not_take_the_page_down(
            self, monkeypatch):
        """The block had no try/except at all, so this was a dead page.

        At 10,000 columns the p x p construction wants 2.7 GB and at 60,000 it
        wants 94 GB, and the failure has to leave a sentence behind: an empty
        Relationships tab reads as "no strong pairs" to anyone who did not watch
        it fail.
        """
        real_corr = pd.DataFrame.corr

        def out_of_memory(self, *args, **kwargs):
            # Only the wide construction fails; the narrower correlation work
            # elsewhere on the page (the collinearity screen caps at 50 columns)
            # keeps working, so this isolates the site under test.
            if self.shape[1] > 70:
                raise MemoryError("Unable to allocate 94.0 GiB for an array")
            return real_corr(self, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "corr", out_of_memory)

        at = _run_eda(self._graded_frame())
        _assert_clean(at, "EDA when the pair screen runs out of memory")

        said = [str(w.value) for w in at.warning if "correlation screen" in str(w.value)]
        assert said, "the screen failed silently"
        assert "not a finding of 'no strong pairs'" in said[0], (
            "the notice lets an empty tab read as a clean result")

    def test_narrowing_the_feature_set_rebuilds_the_screen_it_describes(
            self, monkeypatch):
        """A stale table under a live caption is a false Methods claim.

        `_top_corr_pairs` takes its columns as `_features`, which is
        underscore-prefixed and so takes no part in the cache key, and `data_id`
        digests the FRAME. The screened columns come from
        `data_config.feature_cols`, which the Feature Selection page changes
        without touching a cell of `df`, so two feature sets landing on the same
        budget collided. The caption and the `eda_cap_corr_pairs` insight are
        both composed live from the CURRENT feature set, so the served table
        named columns that were not in the analysis while the manuscript
        sentence beside it described a screen that had never run.
        """
        import ml.regime as regime
        monkeypatch.setattr(regime, "DENSE_PAIRWISE_MAX_FEATURES", 30)

        df = self._graded_frame()
        at = _run_eda(df)
        _assert_clean(at, "EDA with the pair screen capped")
        assert "of 81 numeric features" in self._all_text(at)

        # Narrow the feature set the way the Feature Selection page does. Same
        # frame, same budget of 30 — only the pool changes.
        kept = [f"g{i:03d}" for i in range(60)]
        at.session_state["data_config"].feature_cols = list(kept)
        at.session_state["selected_features"] = list(kept)
        at.run()
        _assert_clean(at, "EDA after narrowing the feature set")

        table_df = None
        for el in at.dataframe:
            value = el.value
            if (isinstance(value, pd.DataFrame)
                    and {"Feature A", "Feature B"} <= set(value.columns)):
                table_df = value
                break
        assert table_df is not None, "the correlation table did not render"

        named = set(table_df["Feature A"]) | set(table_df["Feature B"])
        allowed = set(kept) | {"glucose"}
        assert not (named - allowed), (
            "the table was served from the pre-narrowing cache and names columns "
            f"that are not in the analysis: {sorted(named - allowed)}")

        insight = self._ledger(at).get("eda_cap_corr_pairs")
        assert insight is not None and insight.metadata["n_total"] == 61

    def test_gaps_in_a_column_that_never_enters_the_matrix_change_nothing(
            self, monkeypatch):
        """A categorical's missingness must not caveat a numeric correlation.

        The substitution used to trigger on `regime.n_missing_cols`, which
        counts gaps across every feature column — categoricals included, and not
        one of those appears in a correlation. On complete numeric data
        `sub.rank().corr("pearson")` IS Spearman exactly, so the unresolved
        "computed as the Pearson correlation of column ranks" limitation
        described an approximation that was not one: a false caveat in the
        Discussion, the same class of error as a silent cap.
        """
        import ml.regime as regime
        monkeypatch.setattr(regime, "RANK_CORR_PAIRWISE_MAX_FEATURES", 20)

        rng = np.random.RandomState(5)
        df = self._graded_frame(n=120, p=60)
        site = np.array(["A", "B", "C"] * 40, dtype=object)[:120]
        site[rng.choice(120, 12, replace=False)] = None
        df["site"] = site  # the ONLY column with gaps, and it is categorical

        at = AppTest.from_file("pages/02_EDA.py", default_timeout=120)
        inject_data_state(at, df)
        at.session_state["corr_method"] = "Spearman"
        at.run()
        _assert_clean(at, "EDA with Spearman beside a gappy categorical")

        assert "Pearson correlation of column ranks" not in self._all_text(at), (
            "a categorical column's gaps caveated a complete numeric correlation")
        assert self._ledger(at).get("eda_method_spearman_rank_approx") is None
        limitations = self._ledger(at).discussion_points_for_manuscript()["limitations"]
        assert not any("column ranks" in s for s in limitations), limitations

        # And the disclosure still fires when a column that IS in the matrix
        # has gaps — the substitution is scoped, not disabled.
        df.loc[rng.choice(120, 8, replace=False), "g005"] = np.nan
        at2 = AppTest.from_file("pages/02_EDA.py", default_timeout=120)
        inject_data_state(at2, df)
        at2.session_state["corr_method"] = "Spearman"
        at2.run()
        _assert_clean(at2, "EDA with Spearman on a gappy numeric column")
        assert "Pearson correlation of column ranks" in self._all_text(at2)
        assert self._ledger(at2).get("eda_method_spearman_rank_approx") is not None
