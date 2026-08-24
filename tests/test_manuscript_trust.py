"""Trust guarantees of the manuscript layer.

The manuscript is a compiler from provenance to prose. These tests pin its
contract:
- no claim without evidence (conclusions state facts; adequacy is the
  author's call),
- coaching voice never reaches the manuscript register verbatim,
- author-owned passages are explicit, evidence-citing scaffolds,
- LaTeX output compiles with hostile column names,
- the evidence map traces every compiled section to recorded events and
  admits what was never recorded.
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.narrative_engine import ManuscriptDraft, NarrativeEngine
from utils.insight_ledger import Insight, InsightLedger
from utils.workflow_provenance import WorkflowProvenance


def _prov(metrics=None, models=None, primary="ridge", n=200):
    p = WorkflowProvenance()
    p.record_upload(target_col="glucose", task_type="regression",
                    feature_cols=[f"f{i}" for i in range(8)], n_samples=n)
    p.record_split(strategy="random", train_n=int(n * 0.7),
                   val_n=int(n * 0.15), test_n=n - int(n * 0.7) - int(n * 0.15),
                   random_seed=42)
    p.record_training(
        models_trained=models or ["ridge"],
        primary_model=primary,
        metrics_by_model=metrics or {"ridge": {"R2": 0.62, "RMSE": 8.1}},
    )
    return p


class TestManuscriptRegister:
    def test_manuscript_text_preferred_over_finding(self):
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_missing_severe", source_page="02_EDA",
            category="data_quality", severity="warning",
            finding="3 feature(s) with >30% missing: a (40%), b (35%), c (31%)",
            implication="x", acknowledged=True,
            manuscript_text=("3 predictors exhibited substantial missingness "
                             "(>30% of values), which may bias estimates"),
        ))
        points = ledger.discussion_points_for_manuscript()
        assert points["limitations"] == [
            "3 predictors exhibited substantial missingness (>30% of values), "
            "which may bias estimates"
        ]

    def test_finding_fallback_when_no_manuscript_text(self):
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_missing_severe", source_page="02_EDA",
            category="data_quality", severity="warning",
            finding="2 features have high missingness", implication="x",
            acknowledged=True,
        ))
        points = ledger.discussion_points_for_manuscript()
        assert points["limitations"] == ["2 features have high missingness"]

    def test_coach_findings_carry_manuscript_text(self):
        from ml.model_coach import run_post_training_diagnostics

        results = {
            'ridge': {'metrics': {'RMSE': 0.42, 'R2': 0.55},
                      'train_metrics': {'RMSE': 0.40, 'R2': 0.57}},
            'xgb_reg': {'metrics': {'RMSE': 0.41, 'R2': 0.57},
                        'train_metrics': {'RMSE': 0.10, 'R2': 0.99}},
        }
        findings = run_post_training_diagnostics(results, 'regression')
        assert findings, "diagnostics should fire on this fixture"
        import re
        for f in findings:
            # `COACH-007`: the contract is an explicit DISPOSITION, not a
            # sentence in every case. A finding marked audit_only is coaching
            # the app owes the analyst, not a claim about the study, and it
            # must carry no manuscript sentence at all.
            if f.get('metadata', {}).get('audit_only'):
                assert not f.get('manuscript_text'), (
                    f"{f['id']} is audit_only yet carries manuscript text")
                continue
            assert f.get('manuscript_text'), f"{f['id']} lacks manuscript_text"
            # coach voice must not leak into the manuscript register:
            # no addressing a hypothetical reviewer, no imperative advice
            # ("consider X" — but "parsimony considerations" is fine).
            text = f['manuscript_text'].lower()
            assert "reviewer" not in text
            assert not re.search(r"\bconsider\b", text)


class TestDiscussionHonesty:
    def test_no_unconditional_effectiveness_claim(self):
        draft = NarrativeEngine(_prov()).generate()
        assert "effectively predict" not in draft.discussion
        assert "This study demonstrates" not in draft.discussion

    def test_negative_r2_stated_plainly(self):
        prov = _prov(metrics={"ridge": {"R2": -0.12, "RMSE": 20.0}})
        draft = NarrativeEngine(prov).generate()
        assert "below a mean-only baseline" in draft.discussion
        assert "accounting for" not in draft.discussion

    def test_single_model_never_called_strongest(self):
        draft = NarrativeEngine(_prov(models=["ridge"])).generate()
        assert "strongest" not in draft.discussion.split("### Comparison")[0]

    def test_multi_model_may_claim_strongest(self):
        prov = _prov(models=["ridge", "rf"],
                     metrics={"ridge": {"R2": 0.62}, "rf": {"R2": 0.55}})
        draft = NarrativeEngine(prov).generate()
        assert "strongest held-out performance" in draft.discussion

    def test_conclusions_hand_adequacy_to_author(self):
        draft = NarrativeEngine(_prov()).generate()
        conclusions = draft.discussion.split("### Conclusions")[-1]
        assert "[AUTHOR REQUIRED" in conclusions
        assert "R² = 0.62" in conclusions or "0.62" in conclusions


class TestAuthorScaffolds:
    def test_marker_standardized(self):
        draft = NarrativeEngine(_prov()).generate()
        assert "[Investigator required" not in draft.to_markdown()
        assert draft.count_author_inputs() >= 3

    def test_prior_work_scaffold_cites_headline(self):
        draft = NarrativeEngine(_prov()).generate()
        prior = draft.discussion.split("### Comparison with Prior Work")[-1]
        prior = prior.split("###")[0]
        assert "0.62" in prior and "glucose" in prior

    def test_implications_scaffold_cites_top_features(self):
        engine = NarrativeEngine(
            _prov(), manuscript_context={"top_features": ["age", "bmi", "hdl"]})
        draft = engine.generate()
        impl = draft.discussion.split("### Clinical and Practical Implications")[-1]
        impl = impl.split("###")[0]
        assert "age" in impl and "bmi" in impl
        assert "not causal" in impl or "not present them as" in impl

    def test_preamble_not_counted_as_author_input(self):
        draft = ManuscriptDraft(study_design="We analyzed things.")
        assert "[AUTHOR REQUIRED" in draft.to_markdown()  # preamble mentions it
        assert draft.count_author_inputs() == 0


class TestLatexEscaping:
    def test_hostile_names_escaped(self):
        draft = ManuscriptDraft(
            study_design="Predictor feat_0042 was missing in 15% of rows & flagged #1."
        )
        tex = draft.to_latex()
        assert r"feat\_0042" in tex
        assert r"15\%" in tex
        assert r"\&" in tex
        assert r"\#1" in tex

    def test_markdown_bold_survives_escaping(self):
        draft = ManuscriptDraft(results="**Limitations (auto-generated)** apply to x_1.")
        tex = draft.to_latex()
        assert r"\textbf{Limitations (auto-generated)}" in tex
        assert r"x\_1" in tex


class TestEvidenceMap:
    def test_full_provenance_traced(self):
        engine = NarrativeEngine(_prov())
        emap = engine.generate_evidence_map()
        assert "| Study Design |" in emap
        assert "seed 42" in emap
        assert "glucose" in emap
        assert "AUTHOR" in emap  # discussion ownership row

    def test_empty_provenance_admits_gaps(self):
        engine = NarrativeEngine(WorkflowProvenance())
        emap = engine.generate_evidence_map()
        assert "NOT RECORDED" in emap

    def test_ledger_contributions_listed(self):
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_missing_severe", source_page="02_EDA",
            category="data_quality", severity="warning",
            finding="f", implication="i", acknowledged=True,
        ))
        engine = NarrativeEngine(_prov(), ledger)
        emap = engine.generate_evidence_map()
        assert "eda_missing_severe" in emap
