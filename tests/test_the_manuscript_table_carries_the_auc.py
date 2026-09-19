"""The classification manuscript reports the AUC it computed.

`ml/eval.py` stores the ROC AUC under `'ROC-AUC'`; `ml/bootstrap.py` stores its
confidence interval under `'AUC'`, and so do the baselines. `ml/latex_report.py`
looked for `'AUC'` in the metrics dict at three sites — the performance table's
column list, the interval lookup, and the abstract's discrimination sentence —
and `ml/narrative_engine.py` did the same in the results prose. A key nobody
wrote is a filter that always fails, so every classification manuscript shipped
with accuracy and F1 and no AUC anywhere, while the app and the markdown report
showed it. Found by the Sep 2026 README audit, which ran the table on a
realistic metrics dict and read `Model & Accuracy & F1` back.
"""
from __future__ import annotations

from types import SimpleNamespace

CLASSIFICATION_METRICS = {
    "Accuracy": 0.81, "F1": 0.79, "ROC-AUC": 0.88, "LogLoss": 0.42, "PR-AUC": 0.70,
}


def _interval(lo, hi):
    return SimpleNamespace(ci_lower=lo, ci_upper=hi)


def test_the_performance_table_has_an_auc_column_with_its_interval():
    from ml.latex_report import _metrics_to_latex_table

    tex = _metrics_to_latex_table(
        {"rf": {"metrics": dict(CLASSIFICATION_METRICS)}},
        task_type="classification",
        bootstrap_results={"rf": {"Accuracy": _interval(0.75, 0.86),
                                  "F1": _interval(0.72, 0.85),
                                  "AUC": _interval(0.80, 0.95)}},
    )
    assert "Model & Accuracy & F1 & ROC-AUC" in tex, tex
    assert "0.8800 [0.8000, 0.9500]" in tex, (
        "the interval is stored under 'AUC' and the metric under 'ROC-AUC'; "
        "the table must look it up under both")


def test_the_table_still_prints_a_bare_auc_when_no_interval_was_computed():
    from ml.latex_report import _metrics_to_latex_table

    tex = _metrics_to_latex_table(
        {"rf": {"metrics": dict(CLASSIFICATION_METRICS)}}, task_type="classification")
    assert "ROC-AUC" in tex
    assert "0.8800" in tex


def test_the_abstract_states_the_discrimination():
    from ml.latex_report import _build_structured_abstract_sections

    sections = _build_structured_abstract_sections(
        task_type="classification", target_name="outcome",
        n_total=200, n_train=140, n_val=30, n_test=30,
        model_results={"rf": {"metrics": dict(CLASSIFICATION_METRICS)}},
        bootstrap_results=None,
    )
    joined = " ".join(sections.values())
    assert "AUC of 0.8800" in joined, joined


def test_the_results_prose_names_the_auc():
    from ml.narrative_engine import NarrativeEngine
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(models_trained=["rf"], primary_model="rf",
                         metrics_by_model={"rf": dict(CLASSIFICATION_METRICS)})
    engine = NarrativeEngine(
        prov,
        manuscript_context={"task_type": "classification",
                            "selected_model_results": {
                                "rf": {"metrics": dict(CLASSIFICATION_METRICS)}}})
    text = engine._gen_model_evaluation()
    assert "0.88" in text, text


def test_the_best_model_picker_accepts_either_spelling():
    """`_determine_best_model` falls back from F1 to the AUC; it must find the
    key eval actually writes as well as the one the baselines write."""
    from ml.publication import _determine_best_model

    results = {"a": {"metrics": {"ROC-AUC": 0.9}}, "b": {"metrics": {"AUC": 0.7}}}
    assert _determine_best_model(results, "classification") == "a"
