"""VIF returned 999.0 for every feature and called it multicollinearity.

`multicollinearity_vif` fits one OLS per feature against all the others. Once
p >= n every one of those fits is exact, so `r2 == 1`, so `vif` was `inf`, so
the loop appended the literal `999.0` — a sentinel that sorts, formats and
reads exactly like a measurement, and that then tripped the fixed "VIF > 10"
alarm on every row. Measured on i.i.d. normals whose true VIF is 1 BY
CONSTRUCTION: n=100/p=100 produced 100 sentinels out of 100 and flagged all
100; n=500/p=500 produced 497 sentinels beside three finite values as large as
1.56e4, and flagged all 500. The app reported severe multicollinearity in data
containing no correlation at all.

The two sibling OLS diagnostics on the same page fail the same way and more
quietly: `np.linalg.solve` does not raise on the singular Gram matrix at any p
from 99 to 3,000, so `influence_diagnostics` returns leverage 1.0000000000000566
— leverage is bounded by 1 — and `normality_residuals` runs Shapiro-Wilk on
residuals of sd 1e-15 from a fit with in-sample R^2 = 1.0.

All three now refuse, through the predicates in `ml/regime.py`. Because this
app's output is a manuscript, the refusal is tested as hard as the arithmetic:

* it must be visible on screen (`result['warnings']`, which the page renders
  with `st.warning`) and in the record (an unresolved `Insight` carrying
  `manuscript_text`, which reaches the Discussion as a limitation);
* it must NOT read as a result. `ml/plot_narrative.py` composed its summary
  from an else-branch, so an empty stats dict used to produce "No severe
  multicollinearity (VIF <= 10)" and "No strongly influential points detected"
  — a confident negative about an analysis that never ran, which is a worse
  thing to publish than the sentinel it replaced;
* it must close nothing. Running VIF resolves the page's `eda_corr_cluster_*`
  insights, and a refusal that resolved them would delete a collinearity
  limitation from the manuscript on the strength of work that did not happen;
* and none of it may touch a normal dataset. A 500x20 frame must behave exactly
  as it did before: real numbers, no refusal, no ledger entry, nothing owed.
"""
import numpy as np
import pandas as pd
import pytest

from ml.eda_actions import (
    influence_diagnostics,
    multicollinearity_vif,
    normality_residuals,
    record_diagnostic_on_insights,
)
from ml.eda_recommender import DatasetSignals
from ml.plot_narrative import (
    narrative_eda_influence,
    narrative_eda_multicollinearity,
    narrative_eda_normality,
)
from ml.regime import VIF_MAX_FEATURES, vif_null_baseline
from utils.insight_ledger import Insight, InsightLedger


def _frame(n, p, seed=0):
    """Independent normals: every true VIF is 1, every true influence is nil.

    Whatever these diagnostics report about this frame is an artifact of the
    shape, never a finding about the data. That is what makes it the right
    fixture for a refusal test.
    """
    rng = np.random.default_rng(seed)
    cols = {f"f{i:05d}": rng.normal(0, 1, n) for i in range(p)}
    cols["y"] = rng.normal(0, 1, n)
    return pd.DataFrame(cols)


def _signals(df, target="y"):
    numeric = [c for c in df.columns]
    return DatasetSignals(
        n_rows=len(df), n_cols=df.shape[1], numeric_cols=numeric,
        task_type_final="regression", target_name=target,
    )


def _run(fn, df, target="y"):
    feats = [c for c in df.columns if c != target]
    return fn(df, target, feats, _signals(df), {})


def _flatten(obj):
    """Every scalar anywhere in a result, so a sentinel cannot hide in a tuple."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _flatten(v)
    elif isinstance(obj, Insight):
        yield from _flatten(vars(obj))
    elif isinstance(obj, pd.DataFrame):
        yield from _flatten(obj.to_dict("list"))
    else:
        yield obj


# == The sentinel ==========================================================

class TestVifRefusesInsteadOfReturningASentinel:

    def test_no_999_survives_at_p_equals_n(self):
        res = _run(multicollinearity_vif, _frame(100, 100))
        assert res.get("refused") is True
        assert res["stats"] == {}
        assert not [v for v in _flatten(res)
                    if isinstance(v, (int, float)) and float(v) == 999.0], (
            "the sentinel is still being produced somewhere in the result"
        )

    def test_the_refusal_is_on_screen_and_says_why(self):
        res = _run(multicollinearity_vif, _frame(100, 100))
        text = " ".join(res["warnings"])
        assert "VIF was not computed" in text
        # It must name both halves of the shape; "too many features" without n
        # is not a reason a reader can check.
        assert "100" in text
        assert res["findings"] == [], (
            "a refusal must leave findings empty — plot_narrative falls back to "
            "findings, and a sentence there becomes the page's Summary line"
        )

    def test_the_refusal_reaches_the_record(self):
        res = _run(multicollinearity_vif, _frame(100, 100))
        ins = {i.id: i for i in res["insights"]}
        assert "eda_cap_vif_refused" in ins
        entry = ins["eda_cap_vif_refused"]
        assert entry.resolved is False, "an unresolved entry is what reaches the Discussion"
        assert entry.manuscript_text, "without manuscript_text the Methods section stays silent"
        assert "not computed" in entry.manuscript_text
        # discussion_points_for_manuscript() is the actual route to the paper.
        ledger = InsightLedger()
        ledger.upsert(entry)
        limitations = ledger.discussion_points_for_manuscript()["limitations"]
        assert any("variance inflation" in t for t in limitations), limitations

    def test_the_wall_time_cap_refuses_above_200_features(self):
        # p is inside the validity band (p <= n/2) but past the time cap.
        res = _run(multicollinearity_vif, _frame(5_000, VIF_MAX_FEATURES + 50))
        assert res.get("refused") is True
        assert str(VIF_MAX_FEATURES) in " ".join(res["warnings"])

    def test_the_last_allowed_shape_still_computes(self):
        # p = 200 = VIF_MAX_FEATURES and p/n = 0.4, so both gates are satisfied
        # at their limit. The cap must bite one column later, not one earlier.
        res = _run(multicollinearity_vif, _frame(500, VIF_MAX_FEATURES))
        assert not res.get("refused")
        assert len(res["stats"]["vif"]) == VIF_MAX_FEATURES


class TestTheFlagLineTravelsWithTheShape:
    """A bare "VIF > 10" is not defensible above p/n = 0.5, and drifts below it.

    On features with no collinearity at all E[VIF] = (n-1)/(n-p). At n=500,
    p=450 that is 9.98, and 203 of 450 independent features measured above 10 —
    the alarm was firing on sample size. The threshold now scales with it.
    """

    def test_the_threshold_is_the_null_baseline_times_ten(self):
        n, p = 500, 200
        res = _run(multicollinearity_vif, _frame(n, p))
        baseline = vif_null_baseline(p, n)
        assert res["stats"]["vif_null_baseline"] == pytest.approx(baseline)
        assert res["stats"]["vif_flag_threshold"] == pytest.approx(10.0 * baseline)
        assert res["stats"]["vif_flag_threshold"] > 10.0

    def test_independent_features_are_not_flagged(self):
        res = _run(multicollinearity_vif, _frame(500, 200))
        assert not any("VIF >" in w for w in res["warnings"]), res["warnings"]

    def test_real_collinearity_is_still_caught(self):
        df = _frame(300, 5)
        df["f00001"] = df["f00000"] * 2.9 + np.random.default_rng(1).normal(0, 0.01, 300)
        res = _run(multicollinearity_vif, df)
        assert any("VIF >" in w for w in res["warnings"]), res["warnings"]
        flagged = " ".join(res["warnings"])
        assert "f00000" in flagged and "f00001" in flagged

    def test_the_narrative_quotes_the_same_line(self):
        res = _run(multicollinearity_vif, _frame(500, 200))
        text = narrative_eda_multicollinearity(res["stats"], res["findings"])
        assert "16.6" in text, text  # 10 * 499/300


# == The narratives must not invent a verdict ==============================

class TestARefusalIsNotAnAllClear:

    def test_vif_narrative_gives_no_clean_bill_of_health(self):
        res = _run(multicollinearity_vif, _frame(100, 100))
        text = narrative_eda_multicollinearity(res["stats"], res["findings"])
        assert "No severe multicollinearity" not in text, (
            "the refusal was rendered as a negative result"
        )

    def test_influence_narrative_gives_no_clean_bill_of_health(self):
        res = _run(influence_diagnostics, _frame(100, 200))
        text = narrative_eda_influence(res["stats"], res["findings"])
        assert "No strongly influential points" not in text, text

    def test_normality_narrative_stays_silent(self):
        res = _run(normality_residuals, _frame(100, 200))
        text = narrative_eda_normality(res["stats"], res["findings"])
        assert "approximately normal" not in text
        assert "deviate from normality" not in text


# == The OLS siblings ======================================================

class TestInfluenceAndNormalityRefuseWhereTheyAreUndefined:

    @pytest.mark.parametrize("n,p", [(100, 99), (100, 200), (200, 199)])
    def test_influence_refuses(self, n, p):
        res = _run(influence_diagnostics, _frame(n, p))
        assert res.get("refused") is True
        assert "max_leverage" not in res["stats"], (
            "leverage is bounded by 1 at this shape and every observation has it; "
            "reporting the number is reporting an artifact"
        )
        assert res["insights"][0].id == "eda_cap_influence_undefined"
        assert res["insights"][0].manuscript_text

    @pytest.mark.parametrize("n,p", [(100, 99), (100, 200), (200, 199)])
    def test_normality_refuses(self, n, p):
        res = _run(normality_residuals, _frame(n, p))
        assert res.get("refused") is True
        assert "shapiro_p" not in res["stats"], (
            "the fit is exact at this shape, so Shapiro-Wilk would be testing "
            "floating-point rounding error"
        )
        assert res["insights"][0].id == "eda_cap_normality_undefined"
        assert res["insights"][0].manuscript_text

    def test_the_gate_counts_the_rows_that_survive_the_dropna(self):
        """p vs n is the question, and n is the post-dropna count, not len(df)."""
        df = _frame(300, 120)
        df.loc[df.index[:200], "y"] = np.nan     # 100 usable rows against 120 predictors
        res = _run(influence_diagnostics, df)
        assert res.get("refused") is True
        assert "100" in " ".join(res["warnings"]), res["warnings"]


class TestTheLeverageCountIsNotAStructuralZero:
    """`h > 2k/n` with k = p+1 exceeds 1.0 for every p >= n/2 - 1.

    Leverage cannot exceed 1, so above that width the count was zero by
    arithmetic, not by observation — and that band is INSIDE the refusal gate,
    which only declines at p > n - 2. Measured at n=100/p=50: threshold 1.02,
    count 0, max leverage 0.68. At n=100/p=99: threshold 2.00, count 0, max
    leverage 1.0. "No high-leverage points" was a sentence the arithmetic could
    not have produced any other answer to.
    """

    def test_the_count_is_withheld_where_the_rule_does_not_exist(self):
        res = _run(influence_diagnostics, _frame(100, 50))
        assert not res.get("refused"), "p = n/2 is inside the band this still runs on"
        assert res["stats"]["leverage_threshold"] > 1.0
        assert res["stats"]["n_high_leverage"] is None, (
            "a 0 here is indistinguishable from an observed zero"
        )
        assert any("not counted" in w for w in res["warnings"]), res["warnings"]

    def test_a_normal_shape_still_gets_a_count(self):
        res = _run(influence_diagnostics, _frame(500, 20))
        assert res["stats"]["leverage_threshold"] < 1.0
        assert isinstance(res["stats"]["n_high_leverage"], int)


# == A refusal closes nothing ==============================================

class TestARefusedDiagnosticAnswersNoOpenQuestion:

    def test_it_is_not_recorded_against_the_collinearity_clusters(self):
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_corr_cluster_a_b", source_page="02_EDA",
            category="relationship", severity="warning",
            finding="a and b are correlated at r=0.99",
            implication="coefficients will be unstable",
        ))
        refused = _run(multicollinearity_vif, _frame(100, 100))
        touched = record_diagnostic_on_insights(
            ledger, "multicollinearity_vif", refused, "VIF (Multicollinearity)")
        assert touched == [], (
            "a diagnostic that declined to run is not evidence about anything"
        )
        entry = ledger.get("eda_corr_cluster_a_b")
        assert entry.resolved is False
        assert "diagnostics_run" not in entry.metadata

    def test_a_real_run_is_still_recorded(self):
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_corr_cluster_a_b", source_page="02_EDA",
            category="relationship", severity="warning",
            finding="a and b are correlated at r=0.99",
            implication="coefficients will be unstable",
        ))
        ok = _run(multicollinearity_vif, _frame(500, 20))
        touched = record_diagnostic_on_insights(
            ledger, "multicollinearity_vif", ok, "VIF (Multicollinearity)")
        assert touched == ["eda_corr_cluster_a_b"]


# == The dataset almost everyone actually has ==============================

class TestFiveHundredByTwentyIsUntouched:
    """The whole point of gating on p/n rather than on p: a normal frame must
    acquire no friction at all. 500 rows, 20 predictors — p/n = 0.04."""

    @pytest.fixture
    def df(self):
        return _frame(500, 20)

    def test_vif_computes_twenty_finite_values(self, df):
        res = _run(multicollinearity_vif, df)
        assert not res.get("refused")
        assert not res.get("insights"), "nothing was capped, so nothing is owed the record"
        vifs = res["stats"]["vif"]
        assert len(vifs) == 20
        assert all(v is not None and np.isfinite(v) for _, v in vifs)
        # Independent normals at p/n = 0.04: E[VIF] = 499/480 = 1.04.
        assert max(v for _, v in vifs) < 2.0
        assert res["warnings"] == []
        assert res["figures"] and res["figures"][0][0] == "table"

    def test_influence_reports_every_statistic_it_always_did(self, df):
        res = _run(influence_diagnostics, df)
        assert not res.get("refused")
        for key in ("max_leverage", "max_cooks", "n_high_leverage", "n_high_cooks"):
            assert key in res["stats"], f"influence diagnostics lost {key}"
        assert 0.0 < res["stats"]["max_leverage"] <= 1.0
        assert isinstance(res["stats"]["n_high_leverage"], int)
        assert res["findings"], "a run that succeeded must still say what it found"

    def test_normality_reports_shapiro(self, df):
        res = _run(normality_residuals, df)
        assert not res.get("refused")
        assert "shapiro_p" in res["stats"] and "shapiro_stat" in res["stats"]
        assert res["findings"]

    def test_the_narratives_read_exactly_as_before(self, df):
        vif = _run(multicollinearity_vif, df)
        assert "No severe multicollinearity" in narrative_eda_multicollinearity(
            vif["stats"], vif["findings"])
        infl = _run(influence_diagnostics, df)
        assert "influential" in narrative_eda_influence(infl["stats"], infl["findings"])
        norm = _run(normality_residuals, df)
        assert "normal" in narrative_eda_normality(norm["stats"], norm["findings"])
