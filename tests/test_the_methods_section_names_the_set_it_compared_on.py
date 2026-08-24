"""`AUDIT-030` — the Methods section said `validation` and meant the test set.

Classic's Train page ranks the trained models by `results['metrics']`
(`pages/06_Train_and_Compare.py:1546`), which is the **test** dict written at
`:1496` from `test_metrics`. It recorded that decision as
`selection_criteria='validation <metric>'`, and `ml/narrative_engine.py` rendered
*"<Model> was selected as the primary model, based on validation <metric>."*

A real validation split exists and is used — hyperparameter optimization at
`pages/06:1260` and `:1318` draws one — but **nothing stores a per-model
validation score**, and the ranking that names the primary model never sees one.
So the word named a split this comparison did not use. `GUIDED-104`'s precedent
does not cover it: that note was accepted because the weaker claim was still
true, and there is no weaker true reading of `validation` here.

## What is asserted, and why it is two claims rather than one

The ruling is two sentences. **What was compared** — the models were ranked on
the held-out set — and **what that costs** — choosing among N by held-out score
makes the reported performance optimistic. `research/CLINICAL_SURVEY_PACK.md`
§A5.5 lists *"reporting apparent performance without optimism correction"* as an
anti-pattern, and the number is **declined** rather than invented, because this
door computes no correction.

Both are asserted here, and so is the third thing that keeps the fix honest:
**the caveat does not fire on a single model.** A caution attached to every
manuscript is the uncalibrated second layer this project forbids, which makes a
real concern and a routine one read identically.

## The false word had five producers and the row named two

`AUDIT-030`'s evidence cites `pages/06:1580` and `ml/narrative_engine.py:587`
and `:1091`. `utils/workflow_provenance.py:657` and `:659` composed the same
words as the **default** for a record that never had a criterion written to it —
so a sparse provenance acquired the claim on the way out. That is `LOOP.md` §08's
fifth adjudication question answering itself: *a sweep terminates where the
sweeper's attention ended.* The default path is driven below, by name.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.holdout_selection import criterion_phrase, optimism_sentence   # noqa: E402
from ml.narrative_engine import NarrativeEngine                        # noqa: E402
from utils.workflow_provenance import WorkflowProvenance               # noqa: E402


def _trained(n_models: int = 2, primary: str = "rf") -> WorkflowProvenance:
    """A provenance record shaped exactly as `pages/06` writes one.

    The page ranks `results['metrics']` — the test dict — picks the argmax, and
    records it. So `selected_on_holdout` is True and the criterion names the
    held-out set, because that is what happened.
    """
    prov = WorkflowProvenance()
    names = ["ridge", "rf", "hgb"][:n_models]
    prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
    prov.record_training(
        models_trained=names,
        primary_model=primary if primary in names else names[0],
        selection_criteria=criterion_phrase("RMSE"),
        selected_on_holdout=n_models >= 2,
        metrics_by_model={n: {"RMSE": 12.0 + i} for i, n in enumerate(names)},
    )
    return prov


def test_the_methods_section_does_not_call_the_test_set_a_validation_set():
    """The word itself, on the surface a reader actually gets."""
    draft = NarrativeEngine(_trained()).generate()
    said = draft.model_development
    assert "validation" not in said.lower(), (
        "the Methods section still calls the comparison a validation one, and "
        f"no per-model validation score is stored anywhere: {said!r}")
    assert "held-out" in said, (
        f"the Methods section names no set at all for the comparison: {said!r}")


def test_the_methods_section_says_what_selecting_on_the_held_out_set_costs():
    """§A5.5's anti-pattern is reporting apparent performance without stating
    the optimism. The direction is stated; the magnitude is declined."""
    draft = NarrativeEngine(_trained(n_models=3)).generate()
    said = draft.model_development
    assert "optimistic" in said, (
        f"the model was chosen by comparing three held-out scores and the "
        f"Methods section does not say that costs anything: {said!r}")
    assert "not estimated here" in said, (
        f"the optimism is asserted without saying it was not quantified, which "
        f"invites a reader to assume a correction was applied: {said!r}")
    # AND NO NUMBER IS INVENTED. `[verify-at-build]` discipline on prose: this
    # door computes no optimism correction, so any magnitude here would be made
    # up.
    for forbidden in ("%", "percent", "0.0"):
        assert forbidden not in said.split("optimistic")[-1][:220], (
            f"a magnitude appears beside the optimism claim: {said!r}")


def test_one_model_is_not_a_selection_and_carries_no_caveat():
    """The calibration that keeps the fix from becoming wallpaper."""
    said = NarrativeEngine(_trained(n_models=1)).generate().model_development
    assert "optimistic" not in said, (
        f"one trained model is not a choice among N, and attaching the caveat "
        f"anyway makes a real concern and a routine one read alike: {said!r}")
    assert optimism_sentence(1) == "", "one model is not a selection"
    assert optimism_sentence(0) == "", "no models is not a selection"


def test_a_record_that_never_named_a_criterion_does_not_acquire_the_false_word():
    """`utils/workflow_provenance.py`'s DEFAULT — the pair `AUDIT-030` missed.

    A record with `selection_criteria=''` falls through to a task-appropriate
    default on the way into the manuscript context. That default said
    `validation RMSE` / `validation F1`, so the claim arrived on a record that
    had never made it.
    """
    for task, metric in (("regression", "RMSE"), ("classification", "F1")):
        prov = WorkflowProvenance()
        prov.record_upload("y", task, ["a", "b"], 200)
        prov.record_training(models_trained=["ridge", "rf"], primary_model="rf")
        ctx = prov.get_methods_context()
        assert "validation" not in ctx["selection_criteria"], (
            f"the {task} default still names a validation split: "
            f"{ctx['selection_criteria']!r}")
        assert metric in ctx["selection_criteria"], (
            f"the {task} default dropped the metric: "
            f"{ctx['selection_criteria']!r}")


def test_the_export_path_composes_the_criterion_too_and_it_is_the_same_phrase():
    """`ml/narrative_engine.py:587`, the site `AUDIT-030` names second.

    `pages/10_Report_Export.py:281-305` ranks the models itself — the same argmax
    over `results['metrics']` — and hands the winner in as
    `manuscript_context['best_metric_name']`. `_build_context` turned that into
    the criterion, so the false word had a **second** producer that never touched
    the provenance record.

    It needs its own case because the branch that renders it is the one where an
    author HAS named a primary model: with no `manuscript_primary_model` the
    Methods section takes the other branch and never reads `selection_criteria`
    at all, so a probe pointed at that branch reports this line as not
    load-bearing when it is.
    """
    prov = WorkflowProvenance()
    prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
    prov.record_training(models_trained=["ridge", "rf"])
    draft = NarrativeEngine(prov, manuscript_context={
        "manuscript_primary_model": "rf",
        "best_model_by_metric": "rf",
        "best_metric_name": "RMSE",
    }).generate()
    said = draft.model_development
    assert "validation" not in said.lower(), (
        f"the export path still calls the held-out comparison a validation "
        f"one: {said!r}")
    assert criterion_phrase("RMSE") in said, (
        f"the export path names no set for the comparison: {said!r}")
    assert "optimistic" in said, (
        f"the export path ranked two models on the held-out set and says "
        f"nothing about what that costs: {said!r}")


def test_the_train_page_records_the_phrase_rather_than_a_literal():
    """The producer this suite cannot drive, checked as a claim about the file.

    `pages/06_Train_and_Compare.py:1546` is the ranking that started this, and
    nothing here can execute it: it lives inside a Streamlit callback over
    `st.session_state`, and this repository has no harness that runs a page. The
    manuscript half is driven — `_trained()` above builds exactly the record the
    page writes — but the **write** is verified by reading, and that limit is
    stated here rather than left for someone to assume otherwise.

    `LOOP.md` trap #5 permits it in this one direction: the question is not *does
    this run* but *what does this call site pass*, which is genuinely about the
    file. Asserted on the AST rather than on a substring, because a literal in a
    comment and a literal in an argument are the same text and different facts.
    """
    import ast
    from pathlib import Path

    page = Path(PROJECT_ROOT) / "pages" / "06_Train_and_Compare.py"
    tree = ast.parse(page.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "record_training"]
    assert calls, "pages/06 no longer records training at all"
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        criterion = kwargs.get("selection_criteria")
        assert criterion is not None, (
            "pages/06 records training without naming what it compared")
        assert not isinstance(criterion, (ast.Constant, ast.JoinedStr)) or (
            isinstance(criterion, ast.JoinedStr)
            and "validation" not in ast.unparse(criterion)), (
            f"pages/06 composes the criterion inline again: "
            f"{ast.unparse(criterion)}")
        assert "criterion_phrase" in ast.unparse(criterion), (
            f"pages/06 does not use the one composer, so the phrase has two "
            f"places to drift from: {ast.unparse(criterion)}")
        assert "selected_on_holdout" in kwargs, (
            "pages/06 ranks the models on the held-out set and records nothing "
            "that lets the manuscript say so")


def test_the_caveat_is_recorded_rather_than_parsed_back_out_of_prose():
    """A run that did not select on held-out scores never gets the sentence.

    `selected_on_holdout` defaults to False and is set by the producer that
    knows. The alternative — sniffing the criterion string — is
    `FEATURE_PARITY.md`'s *a substring of a message is a wildcard wearing an
    assertion's clothes*, and it would attach a methodological claim to a
    manuscript on the strength of a word.
    """
    prov = WorkflowProvenance()
    prov.record_upload("glucose", "regression", ["age"], 500)
    prov.record_training(
        models_trained=["ridge", "rf"],
        primary_model="rf",
        selection_criteria="the clinical team's judgment",
        metrics_by_model={"ridge": {"RMSE": 12.0}, "rf": {"RMSE": 11.0}},
    )
    said = NarrativeEngine(prov).generate().model_development
    assert "optimistic" not in said, (
        f"a selection the app did not make is carrying the app's caveat: "
        f"{said!r}")
    assert "the clinical team's judgment" in said, (
        f"the recorded criterion was replaced rather than reported: {said!r}")
