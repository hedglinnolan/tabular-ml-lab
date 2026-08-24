"""Feature selection's predictor set, and the coach's claims about a model.

One test (or class) per finding in the selection/coach cluster of the
paper-risk sprint. Each fails if its fix is reverted:

- `CONTRACT-014` — page 04 printed "These features are retained in your
  dataset and can still be used for modeling" about the categoricals, then
  both Apply buttons wrote a numeric-only list into `data_config.feature_cols`
  — the exact list page 05 splits its categorical branch out of. `gender` and
  `smoking` left the model by the same click that promised to keep them.
- `STATE-010` — the penalized selectors ran at absolute alphas on UNSCALED
  columns, so which predictors reached the published model depended on whether
  a variable was recorded in mg/dL or mmol/L; and "consensus" meant a single
  method's picks whenever one, two or three methods ran.
- `MINE-011` — stability selection divided selection counts by the ATTEMPTED
  bootstraps while a bare `except: continue` skipped the failures, so every
  probability was deflated by the failure rate and the description asserted a
  subsample count the run never achieved.
- `MINE-004` — the >0.95 feature-target correlation scan ended in
  `except Exception: pass`, so a scan that died and a clean dataset were the
  same downstream state — and the EDA page emits "no blocking data-quality
  issues (no severe missingness, leakage candidates, or distributional
  anomalies)" on exactly that state.
- `COACH-003` — the heteroscedasticity finding took `next(iter(model_results))`
  (checkbox order) whenever the never-written `primary_model` session key was
  empty, and its manuscript sentence named no model at all.
- `COACH-007` — three of the four coach preprocessing insights had no
  manuscript disposition, so the ledger's default sent a reassurance ("the
  app's default pipeline already standardizes features for them") into the
  manuscript's LIMITATIONS list.
- Paper item: the coach recommends computing bootstrap CIs BEFORE results are
  reported when none exist; the only CI check ran on intervals that already
  existed.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_selection import (FeatureSelectionResult, consensus_features,
                                  lasso_path_selection, stability_selection)
from ml.model_coach import (_resolve_primary_model,
                            generate_preprocessing_insights,
                            run_post_training_diagnostics)

PAGE_02 = "pages/02_EDA.py"
PAGE_04 = "pages/04_Feature_Selection.py"


def source(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ── CONTRACT-014 ──────────────────────────────────────────────────────────


class TestContract014CategoricalsSurviveApply:
    """What the caption promises and what Apply does must be one thing."""

    def _run_selection(self):
        from streamlit.testing.v1 import AppTest
        from tests.integration.conftest import (build_test_dataframe,
                                                inject_data_state)

        at = AppTest.from_file(PAGE_04, default_timeout=300)
        inject_data_state(at, build_test_dataframe(), target_col="glucose",
                          task_type="regression")
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        run_buttons = [b for b in at.button if "Run Feature Selection" in b.label]
        assert run_buttons, "Feature Selection rendered no Run button — nothing was swept"
        run_buttons[0].click().run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        return at

    def test_applying_the_consensus_keeps_the_categorical_predictors(self):
        at = self._run_selection()
        apply_buttons = [b for b in at.button if "consensus features" in b.label]
        assert apply_buttons, (
            "no consensus Apply button rendered — the consensus was empty, so "
            "this test swept nothing")
        apply_buttons[0].click().run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]

        feature_cols = at.session_state["data_config"].feature_cols
        assert "gender" in feature_cols and "smoking" in feature_cols, (
            "applying the consensus dropped the categorical predictors the page "
            "had just promised were 'retained in your dataset and can still be "
            "used for modeling'. pages/05_Preprocess.py splits "
            "categorical_features out of exactly this list, so every "
            "categorical adjustment variable left the model silently. "
            f"feature_cols={feature_cols}")
        assert "gender" in at.session_state["selected_features"], (
            "selected_features and data_config.feature_cols disagree about the "
            "categoricals; downstream pages read both")

    def test_the_manual_selector_offers_the_categoricals(self):
        at = self._run_selection()
        multis = [m for m in at.multiselect if m.key == "manual_feature_selection"]
        assert multis, "the manual selection multiselect did not render"
        options = list(multis[0].options)
        assert "gender" in options and "smoking" in options, (
            "the manual multiselect offers numeric features only, so a "
            "categorical predictor cannot be re-added by hand once a selection "
            f"has dropped it. options={options}")


# ── STATE-010 ─────────────────────────────────────────────────────────────


class TestState010SelectionDoesNotDependOnUnits:
    """The published predictor set must not turn on mg/dL versus mmol/L."""

    @staticmethod
    def _data(seed=0, n=200):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 4)
        y = 3 * X[:, 0] + 0.6 * X[:, 1] + rng.randn(n) * 0.5
        return X, y, ["a", "b", "c", "d"]

    def test_stability_selection_is_invariant_to_a_unit_change(self):
        X, y, names = self._data()
        X_other_unit = X.copy()
        X_other_unit[:, 1] = X_other_unit[:, 1] * 0.01  # same variable, other unit

        base = stability_selection(X, y, names, "regression", n_bootstrap=20)
        rescaled = stability_selection(X_other_unit, y, names, "regression",
                                       n_bootstrap=20)
        assert base.scores == pytest.approx(rescaled.scores), (
            "stability selection's probabilities changed when one column was "
            "re-expressed in another unit: a fixed absolute alpha on unscaled "
            "columns shrinks the same variable ~18x differently in mg/dL and "
            f"mmol/L. base={base.scores} rescaled={rescaled.scores}")

    def test_the_lasso_path_is_invariant_to_a_unit_change(self):
        X, y, names = self._data(seed=1)
        X_other_unit = X.copy()
        X_other_unit[:, 1] = X_other_unit[:, 1] * 0.01

        base = lasso_path_selection(X, y, names, "regression", cv_folds=3)
        rescaled = lasso_path_selection(X_other_unit, y, names, "regression",
                                        cv_folds=3)
        assert set(base.selected_features) == set(rescaled.selected_features), (
            "the LASSO path selected a different predictor set purely because a "
            "column was recorded in a different unit")

    def test_consensus_needs_two_methods_to_agree(self):
        results = [
            FeatureSelectionResult(method=f"m{i}", selected_features=["a"],
                                   all_features=["a", "b"], scores={},
                                   details={}, description="")
            for i in range(3)
        ]
        results.append(FeatureSelectionResult(
            method="m3", selected_features=["b"], all_features=["a", "b"],
            scores={}, details={}, description=""))
        # A caller asking for 1 is asking for the union under the name
        # "consensus"; the engine floors the threshold at agreement.
        picked = consensus_features(results, min_methods=1)
        assert "b" not in picked, (
            "consensus_features(min_methods=1) returned a feature only one "
            "method selected — 'consensus' then names the union of the methods")
        assert "a" in picked, "the genuinely agreed feature was dropped"

    def test_the_page_requires_agreement_between_two_methods(self):
        text = source(PAGE_04)
        assert "consensus_threshold = max(2, len(results) // 2)" in text, (
            "pages/04 computes the consensus threshold as max(1, ...), so with "
            "one, two or three methods run a single method's picks are applied "
            "to data_config.feature_cols under the name 'consensus' — while the "
            "success message claims agreement between methods")


# ── MINE-011 ──────────────────────────────────────────────────────────────


class _FlakyModel:
    """A Lasso stand-in that fails on a fixed fraction of fits."""

    fail_every = 2  # fail when the call index is odd
    calls = 0

    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        cls = type(self)
        cls.calls += 1
        if cls.calls % cls.fail_every != 0:
            raise ValueError("this subsample contains a single class")
        self.coef_ = np.array([1.0] + [0.0] * (X.shape[1] - 1))
        return self


class TestMine011StabilityDenominator:
    """A fit that never ran is not a vote against a feature."""

    @staticmethod
    def _patch(monkeypatch, fail_every):
        import sklearn.linear_model as lm

        cls = type("Flaky", (_FlakyModel,), {"fail_every": fail_every, "calls": 0})
        monkeypatch.setattr(lm, "Lasso", cls)
        return cls

    def test_probabilities_divide_by_the_fits_that_succeeded(self, monkeypatch):
        self._patch(monkeypatch, fail_every=2)  # half the fits fail
        X = np.random.RandomState(0).randn(40, 3)
        y = np.random.RandomState(1).randn(40)

        result = stability_selection(X, y, ["a", "b", "c"], "regression",
                                     n_bootstrap=10, threshold=0.6)

        assert result.details["n_fits_succeeded"] == 5
        assert result.details["n_fits_failed"] == 5
        assert result.scores["a"] == pytest.approx(1.0), (
            "the selection probability was divided by the ATTEMPTED bootstraps, "
            "so a feature chosen in every fit that ran is reported at the "
            f"failure rate instead: {result.scores}")
        assert "a" in result.selected_features, (
            "the deflated probability dropped a feature below the threshold "
            "that every completed fit selected")

    def test_the_description_states_the_count_it_achieved(self, monkeypatch):
        self._patch(monkeypatch, fail_every=2)
        X = np.random.RandomState(0).randn(40, 3)
        y = np.random.RandomState(1).randn(40)

        result = stability_selection(X, y, ["a", "b", "c"], "regression",
                                     n_bootstrap=10, threshold=0.6)

        assert "across 10 subsamples" not in result.description, (
            "the result string asserts a subsample count the run did not achieve")
        assert "5 completed subsamples" in result.description
        assert "5 of 10 subsample fits failed" in result.description, (
            "the failures are not surfaced anywhere the reader can see them")

    def test_an_excessive_failure_rate_is_a_refusal(self, monkeypatch):
        self._patch(monkeypatch, fail_every=5)  # only 1 fit in 5 succeeds
        X = np.random.RandomState(0).randn(40, 3)
        y = np.random.RandomState(1).randn(40)

        with pytest.raises(RuntimeError) as exc:
            stability_selection(X, y, ["a", "b", "c"], "regression",
                                n_bootstrap=10, threshold=0.6)
        assert "subsample fits succeeded" in str(exc.value), (
            "a run where most fits failed returned quietly deflated "
            "probabilities instead of refusing")


# ── MINE-004 ──────────────────────────────────────────────────────────────


class TestMine004LeakageScanCannotFailSilently:
    """A leakage screen that did not run must not read as a clean bill."""

    @staticmethod
    def _frame(n=60):
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(27, 4, n),
            "glucose": rng.normal(100, 15, n),
        })

    def test_a_failed_scan_is_recorded_on_the_signals(self, monkeypatch):
        from ml import eda_recommender

        df = self._frame()
        real_corr = pd.DataFrame.corr

        def exploding_corr(self, *a, **k):
            if "_target" in self.columns:
                raise MemoryError("correlation matrix did not fit")
            return real_corr(self, *a, **k)

        monkeypatch.setattr(pd.DataFrame, "corr", exploding_corr)
        signals = eda_recommender.compute_dataset_signals(
            df, "glucose", "regression", "cross_sectional", None)

        assert signals.leakage_candidate_cols == [], "positive control"
        assert signals.leakage_scan_error, (
            "the >0.95 feature-target correlation scan failed and left no trace: "
            "an empty leakage_candidate_cols is then indistinguishable from a "
            "dataset that was checked and found clean")
        assert any("leakage" in n.lower() for n in signals.notes)

    def test_a_scan_that_ran_records_no_failure(self):
        from ml import eda_recommender

        signals = eda_recommender.compute_dataset_signals(
            self._frame(), "glucose", "regression", "cross_sectional", None)
        assert signals.leakage_scan_error == "", (
            "a healthy scan reported itself as failed")

    def test_the_eda_page_turns_a_failed_scan_into_a_warning(self):
        text = source(PAGE_02)
        assert "leakage_scan_error" in text, (
            "pages/02 never reads the failure flag, so a dead scan still "
            "produces the clean-data opportunity insight")
        assert 'id="eda_leakage_scan_failed"' in text
        # The clean-data opportunity — whose manuscript_text asserts "no
        # blocking data-quality issues (no severe missingness, leakage
        # candidates, or distributional anomalies)" — counts unresolved
        # blockers and warnings. The disclosure has to be in that count, and
        # written before it is taken.
        disclosure = text.index('id="eda_leakage_scan_failed"')
        gate = text.index("n_issues = len([i for i in ledger.get_unresolved()")
        assert disclosure < gate, (
            "the failed-scan insight is written after the clean-data gate reads "
            "the unresolved count, so the gate cannot see it")
        block = text[disclosure:disclosure + 1200]
        assert 'severity="warning"' in block, (
            "the failed-scan insight is not a warning, so it does not raise "
            "n_issues and the app still asserts no leakage candidates")


# ── COACH-003 ─────────────────────────────────────────────────────────────


def _regression_results(rho_scale=1.0, n=120, seed=0):
    """model_results-shaped dict whose residuals grow with the prediction."""
    rng = np.random.RandomState(seed)
    y_pred = np.linspace(1.0, 10.0, n)
    resid = rng.randn(n) * y_pred * rho_scale
    y_test = y_pred + resid
    return {"y_test": y_test, "y_test_pred": y_pred}


class TestCoach003TheModelIsNamed:
    """A residual claim must say which model it is about."""

    @staticmethod
    def _two_models():
        # Insertion (checkbox) order puts the WORSE model first.
        return {
            "rf": {**_regression_results(seed=1), "metrics": {"RMSE": 9.0}},
            "ridge": {**_regression_results(seed=2), "metrics": {"RMSE": 2.0}},
        }

    def test_the_fallback_is_best_by_metric_not_checkbox_order(self):
        results = self._two_models()
        key, chosen_by = _resolve_primary_model(results, "regression", "")
        assert key == "ridge", (
            "with no primary model designated the coach took the first model in "
            "dict insertion order — checkbox order — as the subject of a "
            "manuscript sentence")
        assert "RMSE" in chosen_by

    def test_an_explicit_primary_model_still_wins(self):
        results = self._two_models()
        key, chosen_by = _resolve_primary_model(results, "regression", "rf")
        assert key == "rf" and "primary" in chosen_by

    def test_the_manuscript_sentence_names_the_model(self):
        findings = run_post_training_diagnostics(
            model_results=self._two_models(), task_type="regression")
        het = [f for f in findings if f["id"] == "train_heteroscedastic_residuals"]
        assert het, "no heteroscedasticity finding — this test swept nothing"
        text = het[0]["manuscript_text"]
        assert "Ridge" in text or "RIDGE" in text or "ridge" in text, (
            "the Discussion sentence asserts non-constant residual variance with "
            "a specific Spearman rho and names no model, so a reader cannot tell "
            f"it may not be about the model the paper reports: {text!r}")
        assert het[0]["metadata"]["model"] == "ridge"


# ── COACH-007 ─────────────────────────────────────────────────────────────


def _profile():
    return SimpleNamespace(
        highly_skewed_features=["bmi", "glucose"],
        features_with_outliers=["bmi"],
        n_features_with_missing=2,
        n_rows=400,
        n_features=8,
        feature_profiles={},
    )


class TestCoach007NoAdviceWithoutADisposition:
    """The ledger's default for an unresolved insight is LIMITATIONS."""

    @staticmethod
    def _insights():
        return generate_preprocessing_insights(["ridge", "rf"], _profile())

    def test_every_insight_carries_an_explicit_disposition(self):
        insights = self._insights()
        assert insights, "the coach produced no preprocessing insights"
        for ins in insights:
            has_text = bool(ins.get("manuscript_text", "").strip())
            audit_only = bool(ins.get("metadata", {}).get("audit_only"))
            assert has_text or audit_only, (
                f"{ins['id']} has neither a manuscript sentence nor an "
                f"audit-only marking, so the ledger falls back to its coaching "
                f"voice `finding` and prints it under Strengths and Limitations")

    def test_a_reassurance_never_reaches_the_limitations(self):
        from utils.insight_ledger import Insight, InsightLedger

        ledger = InsightLedger()
        for ins in self._insights():
            # Exactly what pages/05_Preprocess.py upserts today.
            ledger.upsert(Insight(
                id=ins["id"], source_page=ins["source_page"],
                category=ins["category"], severity=ins["severity"],
                finding=ins["finding"], implication=ins["implication"],
                recommended_action=ins["recommended_action"],
                model_scope=ins.get("model_scope", []),
                relevant_pages=ins.get("relevant_pages", []),
                theory_anchor=ins.get("theory_anchor", ""),
                metadata=ins.get("metadata", {}),
            ))
        limitations = ledger.discussion_points_for_manuscript()["limitations"]
        joined = " ".join(limitations).lower()
        assert "already standardizes" not in joined, (
            "a step the app PERFORMED is printed to a reviewer as an "
            f"unaddressed study limitation: {limitations}")
        assert "the app's" not in joined, (
            f"coaching voice naming 'the app' reached the manuscript: {limitations}")

    def test_the_coach_does_not_write_the_s_pluralization(self):
        for ins in self._insights():
            for field in ("finding", "implication", "recommended_action",
                          "manuscript_text"):
                assert "(s)" not in ins.get(field, ""), (
                    f"{ins['id']}.{field} uses the '(s)' pluralization the rest "
                    f"of the codebase avoids, and this text can reach the "
                    f"manuscript")

    def test_advice_says_whether_anything_resolves_it(self):
        by_id = {i["id"]: i for i in self._insights()}
        assert "no one-click resolver" in by_id["preprocess_skewness_transform"]["recommended_action"], (
            "the skewness advice promises an action nothing in the app records "
            "as taken, so the observation can never be resolved and lands in "
            "the manuscript's limitations")
        assert "Nothing to resolve" in by_id["preprocess_feature_scaling"]["recommended_action"]


# ── Paper item: the coach asks for CIs BEFORE results are reported ────────


class TestTheCoachAsksForIntervalsBeforeReporting:
    """The only CI check ran on intervals that already existed."""

    @staticmethod
    def _results():
        return {
            "ridge": {**_regression_results(seed=3), "metrics": {"RMSE": 2.0}},
            "rf": {**_regression_results(seed=4), "metrics": {"RMSE": 2.4}},
        }

    def test_it_fires_when_no_intervals_exist(self):
        findings = run_post_training_diagnostics(
            model_results=self._results(), task_type="regression",
            bootstrap_results=None)
        ids = [f["id"] for f in findings]
        assert "train_no_bootstrap_ci" in ids, (
            "models were trained with no bootstrap confidence intervals and the "
            "coach said nothing before those point estimates go into a report")

    def test_it_is_silent_once_intervals_exist(self):
        findings = run_post_training_diagnostics(
            model_results=self._results(), task_type="regression",
            bootstrap_results={"ridge": {"RMSE": {"ci_lower": 1.8, "ci_upper": 2.2}}})
        ids = [f["id"] for f in findings]
        assert "train_no_bootstrap_ci" not in ids

    def test_it_makes_no_manuscript_claim(self):
        findings = run_post_training_diagnostics(
            model_results=self._results(), task_type="regression")
        ci = [f for f in findings if f["id"] == "train_no_bootstrap_ci"][0]
        assert not ci["manuscript_text"].strip()
        assert ci["metadata"]["audit_only"] is True, (
            "nothing re-runs the post-training detectors after the intervals "
            "are computed, so this prompt must not become a Discussion sentence "
            "claiming none were computed")
