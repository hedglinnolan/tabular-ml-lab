"""The external cohort, the multiclass SHAP figure, and the consensus claim.

Four things the Explainability page asserted that were not true:

  IMPORT-213   the external-validation uploader — the file the page itself calls
               "the gold standard for publication" — was the one uploader in the
               app with no front door: a bare `load_tabular_data` and a
               missing-column check. A JSON payload holding two record lists
               validated on whichever key came first, silently.
  (paper)      external metrics were computed, displayed, and dropped: no
               session write, no provenance event, so `ml/publication.py`'s
               `external_validation` flag could never be set and the
               manuscript's external-validation section could never populate,
               while the paper claimed sections auto-expand for it.
  STATE-033    for a multiclass outcome the explainer returns
               (n_samples, n_features, n_classes); the page took `[:, :, -1]` —
               the LAST class — recorded `class_label = None`, and then drew a
               figure titled "Mean Absolute SHAP Value (Global Importance)".
               Every intermediate value was individually plausible, and the
               figure was stored for export.
  (paper)      the cross-model chart told the reader to "look for features that
               appear in the top 5 for all models" — a computation handed back
               to the person reading it — while the paper said the app
               highlights the consensus.
"""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
PAGE07 = REPO / "pages" / "07_Explainability.py"


@pytest.fixture
def session():
    """The real streamlit session_state, emptied around each test."""
    import streamlit as st
    st.session_state.clear()
    yield st.session_state
    st.session_state.clear()


def _page_helpers(names: list) -> dict:
    """Exec the named module-level defs of page 07 without a Streamlit runtime.

    Importing the page runs the whole script; these helpers are module-level and
    self-contained, so the regression test exercises the page's real source.
    """
    import streamlit as st
    from typing import Dict, List, Optional

    src = PAGE07.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in wanted} == set(names), (
        f"page 07 no longer declares {set(names) - {n.name for n in wanted}}")
    ns = {"st": st, "pd": pd, "np": np,
          "Dict": Dict, "List": List, "Optional": Optional}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(PAGE07), "exec"), ns)
    return ns


# ══════════════════════════════════════════════════════════════════════════
# IMPORT-213 · the external cohort goes through the same front door
# ══════════════════════════════════════════════════════════════════════════

class TestImport213ExternalUploadUsesTheFrontDoor:
    def test_a_wrapped_payload_has_a_key_to_choose_and_the_page_asks(self):
        """Two sibling record lists: first-wins is a wrong-cohort silently.

        `inspect_json` reports both candidates. Page 01 renders the choice;
        page 07 never called it, so the bare loader took `data` — 20 of the
        file's 220 rows — and the page announced it as a success.
        """
        from data_processor import inspect_json, load_tabular_data

        payload = {
            "results": [{"a": i, "b": i * 2, "y": i % 2} for i in range(200)],
            "data": [{"a": i, "b": i * 2, "y": i % 2} for i in range(20)],
        }
        raw = json.dumps(payload).encode()

        layout = inspect_json(io.BytesIO(raw))
        assert set(layout.candidates) == {"data", "results"}, (
            "the disclosure the external uploader must show is not available")

        bare = load_tabular_data(io.BytesIO(raw), filename="ext.json")
        chosen = load_tabular_data(io.BytesIO(raw), filename="ext.json",
                                   records_key="results")
        assert len(bare) == 20 and len(chosen) == 200, (
            "the wrapper key changes WHICH cohort is validated on")

        page = PAGE07.read_text(encoding="utf-8")
        assert "inspect_json" in page, "page 07 does not disclose the JSON layout"
        assert "records_key=ext_records_key or None" in page, (
            "the records-key choice is not passed to the loader")
        assert 'st.selectbox(\n                            "Which part of this file holds your rows?"' in page or \
               "Which part of this file holds your rows?" in page

    def test_the_import_doctor_runs_and_its_findings_are_shown(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert "render_import_doctor(ext_df, ext_key)" in page, (
            "the external file is not reviewed by the Import Doctor")
        assert "from ml.import_doctor import diagnose" in page
        # And the loader is never called bare any more.
        assert "load_tabular_data(ext_file, filename=ext_file.name)" not in page, (
            "the bare load is still there")

    def test_a_blocking_diagnosis_stops_the_validation(self):
        """A file diagnosed with a critical defect must not validate silently."""
        from ml.import_doctor import diagnose

        # A title row above the header: the classic critical finding.
        broken = pd.DataFrame({
            "Study export v2": ["age", "30", "41", "55", "62", "38"],
            "Unnamed: 1": ["glucose", "88", "94", "120", "101", "77"],
        })
        critical = [f for f in diagnose(broken) if f.severity == "critical"]
        assert critical, "fixture no longer produces a blocking-grade finding"

        page = PAGE07.read_text(encoding="utf-8")
        assert 'ext_blocking = [f for f in diagnose(ext_df) if f.severity == "critical"]' in page
        assert "disabled=bool(ext_blocking) and not ext_override" in page, (
            "the Validate button is not gated on the blocking findings")

    def test_the_transpose_option_exists_for_the_external_file(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert 'key=f"transpose_{ext_key}"' in page
        assert "transpose=ext_transpose" in page


# ══════════════════════════════════════════════════════════════════════════
# External validation is persisted, recorded, and reaches the manuscript
# ══════════════════════════════════════════════════════════════════════════

class TestExternalValidationSurvivesThePage:
    def test_the_page_writes_both_the_result_and_the_record(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert "st.session_state['external_validation_results']" in page, (
            "the external metrics are still dropped on the floor")
        assert "get_provenance().record_external_validation(" in page, (
            "no provenance event is recorded for external validation")

    def test_the_record_reaches_the_methods_context(self):
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_external_validation(
            dataset_name="cohort_b.csv", n_rows=412, n_features=6,
            models_validated=["rf", "ridge"],
            metrics={"rf": {"AUC": {"estimate": 0.81}}}, n_bootstrap=500)
        ctx = prov.get_methods_context()
        assert ctx["external_validation"] is True
        assert ctx["external_validation_n"] == 412
        assert ctx["external_validation_models"] == ["rf", "ridge"]

    def test_the_methods_section_reports_it_without_being_told(self, session):
        """`ml/publication.py:374`'s flag had no caller — page 10 never set it."""
        from ml.publication import generate_methods_section
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_external_validation(
            dataset_name="cohort_b.csv", n_rows=412, n_features=6,
            models_validated=["rf"],
            metrics={"rf": {"AUC": {"estimate": 0.812, "ci_lower": 0.75,
                                    "ci_upper": 0.87}}},
            n_bootstrap=500)
        session["workflow_provenance"] = prov

        text = generate_methods_section(
            data_config={"feature_cols": ["age", "bmi"], "target_col": "y"},
            preprocessing_config={}, model_configs={"rf": {}}, split_config={},
            n_total=500, n_train=350, n_val=75, n_test=75,
            feature_names=["age", "bmi"], target_name="y",
            task_type="classification", metrics_used=["AUC"],
        )
        assert "External validation was performed" in text
        assert "412" in text, "the external cohort's size is not reported"
        assert "cohort_b.csv" in text
        assert "0.812" in text, "the external metrics never reach the draft"

    def test_the_narrative_engine_draft_expands_for_it(self):
        """The primary path page 10 actually takes."""
        from ml.narrative_engine import NarrativeEngine
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload(target_col="y", task_type="classification",
                           feature_cols=["age", "bmi"], n_samples=500)
        prov.record_training(models_trained=["rf"], primary_model="rf")
        prov.training.metrics_by_model = {"rf": {"AUC": 0.78}}
        prov.record_external_validation(
            dataset_name="cohort_b.csv", n_rows=412, n_features=2,
            models_validated=["rf"], metrics={}, n_bootstrap=500)

        draft = NarrativeEngine(prov).generate()
        assert "External validation was performed" in draft.model_evaluation
        assert "412" in draft.model_evaluation

    def test_a_data_change_takes_the_external_claim_with_it(self, session):
        """Both halves: the result key is registered, the section is derived."""
        from utils.session_state import reset_downstream_results
        from utils.workflow_provenance import (WorkflowProvenance,
                                               downstream_sections,
                                               section_names)

        assert "external_validation" in section_names(), (
            "the new section is invisible to the schema-derived registry")
        assert "external_validation" in downstream_sections(), (
            "a section describing work computed from the data is not cleared")

        prov = WorkflowProvenance()
        prov.record_external_validation(
            dataset_name="cohort_b.csv", n_rows=412, n_features=6,
            models_validated=["rf"], metrics={}, n_bootstrap=500)
        session["workflow_provenance"] = prov
        session["external_validation_results"] = {
            "dataset_name": "cohort_b.csv", "per_model": {"rf": {}}}

        reset_downstream_results()

        assert session.get("external_validation_results") is None, (
            "external metrics computed on replaced data survived the reset")
        assert prov.external_validation is None
        assert "external_validation" not in prov.get_methods_context()

    def test_the_sentence_says_only_what_the_record_holds(self):
        from ml.publication import external_validation_sentence

        bare = external_validation_sentence(None)
        assert "independent dataset" in bare
        assert "rows" not in bare, "a count with nothing behind it"

        full = external_validation_sentence(
            {"dataset_name": "cohort_b.csv", "n_rows": 412, "models": ["rf"],
             "n_bootstrap": 500, "repairs": ["Recoded -999 to missing in age"],
             "metrics": {"rf": {"AUC": {"estimate": 0.812, "ci_lower": 0.75,
                                        "ci_upper": 0.87}}}})
        assert "412" in full and "cohort_b.csv" in full
        assert "500" in full
        assert "0.812" in full and "0.750" in full, (
            "the external metrics are asserted but not reported")
        assert "Recoded -999" in full, (
            "a repaired external file is reported as repaired")


# ══════════════════════════════════════════════════════════════════════════
# STATE-033 · a SHAP figure names the class it is of, or is not drawn
# ══════════════════════════════════════════════════════════════════════════

class TestState033MulticlassShapIsLabeled:
    def test_a_three_class_array_is_not_flattened_to_an_unnamed_class(self):
        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label"])
        arr = np.arange(6 * 4 * 3, dtype=float).reshape(6, 4, 3)

        rec = ns["_normalize_shap_values"](arr, class_names=["mild", "moderate", "severe"])

        assert rec["error"] is None
        assert rec["n_classes"] == 3
        assert rec["class_label"], "the matrix is still stored with no class attribution"
        # The values shown are the values the label names — the old code showed
        # the LAST class and labeled it nothing (or 'Class 0').
        assert rec["class_label"] == "mild"
        assert np.array_equal(rec["shap_values"], arr[:, :, 0])
        # And no class is lost, so the reader can see the others.
        assert len(rec["per_class"]) == 3
        assert np.array_equal(rec["per_class"][2], arr[:, :, 2])

    def test_the_labeled_class_is_the_class_that_was_ranked(self):
        """The bite: class 0 and the last class rank features differently."""
        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label"])
        arr = np.zeros((5, 3, 3))
        arr[:, 0, 0] = 1.0      # feature 0 dominates class 0
        arr[:, 2, 2] = 1.0      # feature 2 dominates class 2 (the LAST class)

        rec = ns["_normalize_shap_values"](arr, class_names=["a", "b", "c"])
        ranked = int(np.argmax(np.abs(rec["shap_values"]).mean(axis=0)))
        old_behavior = int(np.argmax(np.abs(arr[:, :, -1]).mean(axis=0)))

        assert ranked == 0, "the ranking is not the class the label names"
        assert old_behavior == 2, "fixture no longer distinguishes the classes"
        assert rec["class_label"] == "a"

    def test_binary_keeps_the_positive_class_and_says_so(self):
        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label"])
        arr = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)

        rec = ns["_normalize_shap_values"](arr, class_names=["no", "yes"])
        assert np.array_equal(rec["shap_values"], arr[:, :, 1])
        assert "positive" in rec["class_label"] and "yes" in rec["class_label"]

    def test_a_shape_it_cannot_account_for_is_refused_not_reduced(self):
        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label"])
        rec = ns["_normalize_shap_values"](np.zeros((2, 2, 2, 2)))
        assert rec.get("error"), "an unreadable shape is still silently reduced"

    def test_a_regression_matrix_is_untouched_and_unlabeled(self):
        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label"])
        arr = np.random.RandomState(0).rand(7, 3)
        rec = ns["_normalize_shap_values"](arr)
        assert rec["class_label"] is None and rec["per_class"] is None
        assert np.array_equal(rec["shap_values"], arr)

    def test_the_live_explainer_shape_is_the_one_that_was_mishandled(self):
        """Measured, not assumed: shap returns (n, f, k) for a 3-class forest."""
        shap = pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(0)
        X = rng.rand(60, 4)
        y = rng.randint(0, 3, 60)
        model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y)
        sv = shap.TreeExplainer(model).shap_values(X[:10])
        assert np.asarray(sv).ndim == 3 and np.asarray(sv).shape[2] == 3

        ns = _page_helpers(["_normalize_shap_values", "_shap_class_label",
                            "_shap_class_names_for"])
        names = ns["_shap_class_names_for"](model, 3, None)
        rec = ns["_normalize_shap_values"](sv, class_names=names)
        assert rec["error"] is None
        assert rec["class_label"] == names[0] == "0"
        assert np.array_equal(rec["shap_values"], np.asarray(sv)[:, :, 0])

    def test_the_render_path_no_longer_reslices_or_hides_the_class(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert "sv_plot[:, :, -1]" not in page, (
            "the compute path still takes the last class")
        assert "sv = sv[:, :, -1]" not in page, (
            "the render path still normalizes a stored array a second time")
        assert 'if cl:\n                            ax.set_title' not in page, (
            "the figure title is still conditional on a label that could be None")
        assert '"Mean Absolute SHAP Value (Global Importance)"' not in page, (
            "a one-class ranking is still titled as global importance")


# ══════════════════════════════════════════════════════════════════════════
# The cross-model consensus is computed and highlighted, not delegated
# ══════════════════════════════════════════════════════════════════════════

class TestCrossModelConsensusIsComputed:
    def test_only_features_in_every_models_top_n_are_consensus(self):
        ns = _page_helpers(["_consensus_top_features"])
        perm = {
            "rf":    {"feature_names": ["age", "bmi", "sex", "crp"],
                      "importances_mean": np.array([0.9, 0.8, 0.1, 0.05])},
            "ridge": {"feature_names": ["age", "bmi", "sex", "crp"],
                      "importances_mean": np.array([0.7, 0.6, 0.2, 0.02])},
            # Agrees about age, disagrees about bmi — so bmi is not consensus.
            "xgb":   {"feature_names": ["age", "bmi", "sex", "crp"],
                      "importances_mean": np.array([0.9, 0.05, 0.5, 0.8])},
        }
        consensus = set(ns["_consensus_top_features"](perm, top_n=2))
        assert consensus == {"age"}, (
            "consensus must be unanimity across models, not a majority")

    def test_disagreement_produces_no_consensus_rather_than_a_guess(self):
        ns = _page_helpers(["_consensus_top_features"])
        perm = {
            "rf":    {"feature_names": ["a", "b"], "importances_mean": np.array([1.0, 0.0])},
            "ridge": {"feature_names": ["a", "b"], "importances_mean": np.array([0.0, 1.0])},
        }
        assert ns["_consensus_top_features"](perm, top_n=1) == []

    def test_it_is_the_apps_one_notion_of_consensus(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert "from ml.feature_selection import FeatureSelectionResult, consensus_features" in page, (
            "page 07 invented its own consensus instead of reusing the app's")

    def test_the_chart_highlights_it_instead_of_asking_the_reader_to_look(self):
        page = PAGE07.read_text(encoding="utf-8")
        assert "Look for features that appear in the top 5 for all models" not in page, (
            "the page still hands the computation back to the reader")
        assert "_consensus_top_features(perm_data" in page
        assert "★" in page, "nothing marks the consensus features on the chart"
