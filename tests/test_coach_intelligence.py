"""The coach's expanded intelligence: evidence probe, viability verdicts,
new post-training detectors, and provenance-to-manuscript integration.

The core contract: every capability MEASURES rather than assumes, cites the
numbers it measured, respects the lockbox (training rows only), and records
its reasoning where the manuscript can cite it.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.coach_probe import run_probe
from ml.model_coach import (model_viability, run_post_training_diagnostics,
                            select_top_picks)

rng = np.random.default_rng(11)


def _X(n, p):
    return pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])


class TestEvidenceProbe:
    def test_detects_linear_signal_and_no_tree_gain(self):
        X = _X(400, 10)
        y = X["f0"] * 3 + X["f1"] - 2 * X["f2"] + rng.normal(0, 0.8, 400)
        r = run_probe(X, y, "regression")
        assert r.has_signal is True
        assert r.nonlinearity_gain < 0.02

    def test_detects_nonlinear_structure(self):
        X = _X(400, 10)
        y = np.sin(X["f0"] * 2.5) * 3 + (X["f1"] > 0) * 2.5 + rng.normal(0, 0.4, 400)
        r = run_probe(X, y, "regression")
        assert r.has_signal is True
        assert r.nonlinearity_gain > 0.2

    def test_pure_noise_flagged_no_signal(self):
        X = _X(400, 10)
        r = run_probe(X, pd.Series(rng.normal(size=400)), "regression")
        assert r.has_signal is False
        assert "little signal" in r.summary()

    def test_negative_r2_never_counts_as_signal(self):
        X = _X(30, 200)
        r = run_probe(X, pd.Series(rng.normal(size=30)), "regression")
        assert r.has_signal is not True

    def test_wide_data_signal_recovered_by_l1(self):
        X = _X(34, 300).rename(columns=lambda c: c.replace("f", "g"))
        y = X["g0"] * 5 + rng.normal(0, 0.2, 34)
        r = run_probe(X, y, "regression")
        assert r.has_signal is True, "L1 prober must see p>n signal"

    def test_underpowered_small_n_says_unconfirmed_not_absent(self):
        X = _X(34, 300).rename(columns=lambda c: c.replace("f", "g"))
        y = (X["g0"] + rng.normal(0, 0.6, 34) > 0).astype(int)
        r = run_probe(X, y, "classification")
        if r.has_signal is False:
            assert "underpowered" in r.summary() or "unconfirmed" in r.summary()

    def test_noise_never_advised_to_collect_more_data(self):
        X = _X(400, 10)
        r = run_probe(X, pd.Series(rng.normal(size=400)), "regression")
        assert "rising" not in r.summary()

    def test_seeded_and_deterministic(self):
        X = _X(200, 8)
        y = X["f0"] + rng.normal(0, 1, 200)
        r1, r2 = run_probe(X, y, "regression"), run_probe(X, y, "regression")
        assert r1.linear_score == r2.linear_score


class TestProbeInformedPicks:
    def _profile(self, df, target, task):
        from ml.dataset_profile import compute_dataset_profile
        feats = [c for c in df.columns if c != target]
        return compute_dataset_profile(df, target, feats, task)

    def test_no_signal_probe_reaches_headline(self):
        df = _X(400, 10)
        df["y"] = rng.normal(size=400)
        probe = run_probe(df[[c for c in df if c != "y"]], df["y"], "regression")
        assert probe.has_signal is False
        _, _, headline = select_top_picks(self._profile(df, "y", "regression"), probe=probe)
        assert "Expect null results" in headline

    def test_nonlinear_probe_cited_in_tree_pick(self):
        df = _X(400, 10)
        df["y"] = np.sin(df["f0"] * 2.5) * 3 + (df["f1"] > 0) * 2.5 + rng.normal(0, 0.4, 400)
        probe = run_probe(df[[c for c in df if c != "y"]], df["y"], "regression")
        picks, _, _ = select_top_picks(self._profile(df, "y", "regression"), probe=probe)
        tree = [p for p in picks if p.group == "Trees/Boosting"]
        assert tree and "probe measured" in tree[0].why

    def test_linear_probe_cited_in_linear_pick(self):
        df = _X(500, 10)
        df["y"] = df["f0"] * 3 + rng.normal(0, 0.5, 500)
        probe = run_probe(df[[c for c in df if c != "y"]], df["y"], "regression")
        picks, _, _ = select_top_picks(self._profile(df, "y", "regression"), probe=probe)
        assert "trees ≈ linear" in picks[0].why


class TestModelViability:
    def _profile(self, n, p, task="regression", imbalance=None):
        from ml.dataset_profile import compute_dataset_profile
        df = _X(n, p)
        if task == "regression":
            df["y"] = df["f0"] + rng.normal(0, 1, n)
        else:
            df["y"] = (rng.random(n) < (imbalance or 0.5)).astype(int)
        feats = [c for c in df.columns if c != "y"]
        return compute_dataset_profile(df, "y", feats, task)

    def test_covers_full_registry(self):
        from ml.model_registry import get_registry

        v = model_viability(self._profile(500, 10))
        registry_keys = set(get_registry().keys())
        missing = registry_keys - set(v.keys())
        assert not missing, f"registry models without verdicts: {missing}"

    def test_wide_data_verdicts(self):
        v = model_viability(self._profile(40, 200))
        assert v["glm"][0] == "poor" and "identifiable" in v["glm"][1]
        assert v["ridge"][0] == "good"
        assert v["rf"][0] == "poor" and "memorize" in v["rf"][1]
        assert v["knn_reg"][0] == "poor"

    def test_low_epv_verdicts(self):
        v = model_viability(self._profile(400, 12, task="classification", imbalance=0.06))
        assert v["logreg"][0] == "good" and "EPV" in v["logreg"][1]
        assert v["histgb_clf"][0] == "poor" and "EPV" in v["histgb_clf"][1]
        assert v["lda"][0] == "poor"

    def test_verdicts_carry_numbers(self):
        v = model_viability(self._profile(20000, 25, task="classification"))
        assert "20,000" in v["svc"][1]
        assert v["nn"][0] == "good"


class TestNewDetectors:
    def test_accuracy_below_nir_flagged(self):
        y_test = np.array([0] * 80 + [1] * 20)
        results = {"logreg": {"metrics": {"Accuracy": 0.80}, "y_test": y_test}}
        ids = [f["id"] for f in run_post_training_diagnostics(results, "classification")]
        assert "train_accuracy_below_nir" in ids

    def test_accuracy_above_nir_not_flagged(self):
        y_test = np.array([0] * 80 + [1] * 20)
        results = {"logreg": {"metrics": {"Accuracy": 0.93}, "y_test": y_test}}
        ids = [f["id"] for f in run_post_training_diagnostics(results, "classification")]
        assert "train_accuracy_below_nir" not in ids

    def test_ci_overlap_flagged_with_intervals_cited(self):
        class CI:
            def __init__(s, lo, hi):
                s.ci_lower, s.ci_upper = lo, hi

        boot = {"ridge": {"RMSE": CI(1.8, 2.4)}, "rf": {"RMSE": CI(1.9, 2.6)}}
        results = {"ridge": {"metrics": {"RMSE": 2.0}}, "rf": {"metrics": {"RMSE": 2.1}}}
        f = run_post_training_diagnostics(results, "regression", bootstrap_results=boot)
        hits = [x for x in f if x["id"] == "train_ci_overlap_top_models"]
        assert hits and "[1.800, 2.400]" in hits[0]["finding"]
        assert hits[0].get("manuscript_text")

    def test_heteroscedasticity_detected_with_rho(self):
        y_pred = np.linspace(1, 10, 200)
        y_true = y_pred + np.random.default_rng(1).normal(0, 1, 200) * (y_pred / 3)
        results = {"ridge": {"metrics": {"RMSE": 2.0},
                             "y_test": y_true, "y_test_pred": y_pred}}
        f = run_post_training_diagnostics(results, "regression", primary_model="ridge")
        hits = [x for x in f if x["id"] == "train_heteroscedastic_residuals"]
        assert hits and "ρ" in hits[0]["finding"]

    def test_homoscedastic_not_flagged(self):
        y_pred = np.linspace(1, 10, 200)
        y_true = y_pred + np.random.default_rng(1).normal(0, 1, 200)
        results = {"ridge": {"metrics": {"RMSE": 1.0},
                             "y_test": y_true, "y_test_pred": y_pred}}
        f = run_post_training_diagnostics(results, "regression", primary_model="ridge")
        assert not [x for x in f if x["id"] == "train_heteroscedastic_residuals"]


class TestCoachProvenance:
    def test_rationale_reaches_methods_draft(self):
        from ml.narrative_engine import NarrativeEngine
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload(target_col="y", task_type="regression",
                           feature_cols=["a", "b"], n_samples=200)
        prov.record_training(models_trained=["ridge"], primary_model="ridge",
                             metrics_by_model={"ridge": {"R2": 0.6}})
        prov.record_coach(
            headline="Dominant constraint: 200 rows.",
            picks=[{"role": "Start here", "model_key": "ridge",
                    "model_name": "Ridge Regression", "why": "stable"}],
            probe_summary="learnable signal (probe R² ≈ 0.60)",
        )
        draft = NarrativeEngine(prov).generate()
        assert "shortlisted from the dataset" in draft.model_development
        assert "advisory and are not reported as results" in draft.model_development

    def test_coach_row_in_evidence_map(self):
        from ml.narrative_engine import NarrativeEngine
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_coach(headline="h", picks=[{"model_key": "ridge"}],
                          probe_summary="s")
        emap = NarrativeEngine(prov).generate_evidence_map()
        assert "shortlist rationale" in emap

    def test_serialization_roundtrip(self):
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_coach(headline="h", picks=[{"model_key": "lasso"}], probe_summary="p")
        again = WorkflowProvenance.from_dict(prov.to_dict())
        assert again.coach.picks[0]["model_key"] == "lasso"
