"""`AUDIT-026` and `AUDIT-029` — the Methods section describes the CV that ran.

## Two rows, one composer's worth of truth

Both are about the same sentence-level mistake at two removes, and **one signal
closes both**: *which models a fold loop actually scored.*

**`AUDIT-026`.** Both Methods composers read the CHECKBOX. `ml/narrative_engine`
emitted *"{n}-fold cross-validation was used for model evaluation."* on
`use_cv and cv_folds`, and `ml/publication` emitted *"{n}-fold cross-validation
was used for internal validation."* from the logged flag. Neither read
`model_results[*]['cv_results']`, which is the only object recording that a fold
loop ran. The two come apart routinely — `pages/06:1455` excludes the neural
network from CV outright, and `:1489` catches a CV failure, warns, and continues
with `cv_results=None` — so training only the neural network with the box ticked
produced a Methods section asserting the internal validation design §A5.5 calls
acceptable over the single split §A5.5 calls *"the weakest option ...
discouraged"*.

**`AUDIT-029`.** The *Methodological Considerations* paragraph told the reader
CV had been *"performed on data that had already been preprocessed using the
full training set"* — false since `STATE-059`. L44 rewrote that paragraph
correctly and gated it on a name, `cv_models_run`, **that was never bound
anywhere**: not a parameter, not a local, not a module global. So the corrected
paragraph never printed, and `generate_methods_section` raised
`NameError: name 'cv_models_run' is not defined` on every call where
`cv_folds or cv_to_use` was truthy — which is precisely when CV was enabled.
`test_the_fallback_composer_survives_the_cv_path_at_all` is that regression, and
it is the reason this file's first assertion is that the function returns.

## Why this drives both composers

Trap #5. `cv_models_run` appeared in the source and read as a working gate; only
calling the function shows it was a crash. Every claim here comes from invoking
a composer and reading its output.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.narrative_engine import NarrativeEngine  # noqa: E402
from ml.publication import generate_methods_section  # noqa: E402
from utils.workflow_provenance import WorkflowProvenance  # noqa: E402

#: `GUIDED-097`. Every claim below runs against a binary 0/1 outcome and a
#: continuous one, because the composers branch on `task_type` for the
#: surrounding prose and a CV sentence that only survives one branch is a
#: sentence checked once.
SHAPES = {
    "binary_0_1": ("classification", "outcome"),
    "continuous": ("regression", "hba1c"),
}

#: Named rather than left silent.
SHAPES_NOT_COVERED = {
    "multiclass": (
        "Neither composer branches on level count — `task_type` is the only "
        "discriminator either reads — so a three-level outcome exercises the "
        "same code path as `binary_0_1` for every claim here."),
    "string_binary_outcome": (
        "The Classic composers receive `task_type` and a metrics dict, never "
        "the label vector, so a string-valued outcome cannot reach these "
        "sentences differently. The shape matters upstream at `pages/06`, "
        "which this file does not drive."),
    "survival": (
        "The app has no time-to-event task type, so there is no shape."),
}

#: The neural network is the only trained model and `pages/06:1455` skipped CV
#: for it — the reachable case the row names first.
NN_ONLY_NOTHING_RAN = {"nn": {"metrics": {"AUC": 0.81}, "cv_results": None}}
#: CV ran for one of two models.
RF_RAN_NN_DID_NOT = {
    "rf": {"metrics": {"AUC": 0.83}, "cv_results": {"mean": 0.80, "std": 0.02}},
    "nn": {"metrics": {"AUC": 0.81}, "cv_results": None},
}


def _methods(task_type, target, **kw):
    """`ml.publication.generate_methods_section`, the documented fallback."""
    args = dict(
        data_config={"feature_cols": ["a", "b"], "target_col": target},
        preprocessing_config={},
        model_configs={"nn": {}},
        split_config={"stratify": True},
        n_total=300, n_train=200, n_val=50, n_test=50,
        feature_names=["a", "b"], target_name=target,
        task_type=task_type, metrics_used=["AUC"],
    )
    args.update(kw)
    return generate_methods_section(**args)


def _narrative(task_type, target, models, use_cv, cv_folds, cv_models_run):
    """`ml.narrative_engine`, the primary export path."""
    prov = WorkflowProvenance()
    prov.record_upload(target_col=target, task_type=task_type,
                       feature_cols=["a", "b"], n_samples=300)
    prov.record_split(strategy="stratified", train_n=200, val_n=50, test_n=50)
    prov.record_training(models_trained=models, primary_model=models[0],
                         use_cv=use_cv, cv_folds=cv_folds,
                         cv_models_run=cv_models_run)
    return NarrativeEngine(prov).generate().to_markdown()


# ── 1 · the crash the corrected paragraph was hiding behind ──────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_fallback_composer_survives_the_cv_path_at_all(shape):
    """`AUDIT-029`. The gate name was never bound, so the function raised.

    This is deliberately the weakest possible assertion — that a call returns a
    string — because that is exactly what was broken, and a stronger assertion
    about the paragraph's wording would have passed straight through the
    `NameError` into an error nobody attributed to CV.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=RF_RAN_NN_DID_NOT)
    assert isinstance(text, str) and text


# ── 2 · the hardest case: ticked, and nothing cross-validated ────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_fallback_does_not_claim_cross_validation_that_produced_nothing(
        shape):
    """`AUDIT-026` on `ml/publication`, in the shape the row names.

    Box ticked, neural network the only model, `cv_results` None. The false
    sentence is not merely absent — the honest replacement names the internal
    validation the run actually has, which is the split.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=NN_ONLY_NOTHING_RAN)

    assert "cross-validation was used for internal validation." not in text, (
        "the Methods section asserts an internal validation design that "
        "produced results for no model")
    assert "produced results for no model" in text
    assert "train/validation/test split" in text, (
        "the false claim went away and nothing true replaced it")


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_narrative_does_not_claim_cross_validation_that_produced_nothing(
        shape):
    """`AUDIT-026` on `ml/narrative_engine` — the PRIMARY export path.

    `pages/10:667-676` takes this composer whenever any provenance section is
    complete, so this is the sentence most exports actually carry. The
    `publication` fallback above is the documented second path, not the common
    one.
    """
    task_type, target = SHAPES[shape]
    text = _narrative(task_type, target, ["nn"], True, 5, [])

    assert "cross-validation was used for model evaluation." not in text
    assert "produced results for no model" in text
    assert "held-out split alone" in text


# ── 3 · and it still says so when CV did run ─────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_both_composers_report_cross_validation_when_it_ran_and_name_the_models(
        shape):
    """The shelf is not shortened: the sentence survives, with its scope.

    A gate that suppressed the claim in both directions would pass the two
    tests above and lose the Methods section a TRIPOD reporting item. Both
    composers must still report CV — and, because it ran for a subset, say
    which models it did not cover rather than implying all of them.
    """
    task_type, target = SHAPES[shape]

    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=RF_RAN_NN_DID_NOT)
    assert "5-fold cross-validation was used for internal validation of" in text
    assert "Random Forest" in text
    assert "It was not run for" in text
    assert "held-out split alone" in text

    narrative = _narrative(task_type, target, ["rf", "nn"], True, 5, ["rf"])
    assert "5-fold cross-validation was used for evaluation of" in narrative
    assert "It was not run for" in narrative


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_methodological_paragraph_describes_the_loop_the_code_runs(shape):
    """`AUDIT-029`'s own sentence, now that it can print.

    `ml/eval.py:182-197` clones the preprocessing into the CV composite, so
    every fold re-fits it — the paragraph asserted the opposite. It must also
    still disclose what the loop does NOT enclose, which is the half that keeps
    the correction from being a whitewash: selection and tuning happen outside
    it and no optimism correction is applied.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=RF_RAN_NN_DID_NOT)

    assert "already been preprocessed using the full training set" not in text, (
        "the Methods section describes a leak `ml/eval.make_cv_pipeline` "
        "structurally prevents")
    assert "What the cross-validation loop enclosed" in text
    assert "re-fit inside" in text and "each fold" in text
    assert "no optimism correction" in text, (
        "the paragraph was corrected into a claim of cleanliness and dropped "
        "the caveat it owes")


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_fold_paragraph_stays_out_of_a_run_that_cross_validated_nothing(
        shape):
    """The gate `cv_models_run` was written for, working for the first time.

    A paragraph explaining what the fold loop enclosed, printed for a run with
    no fold loop, is `AUDIT-026` restated at paragraph length.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=NN_ONLY_NOTHING_RAN)
    assert "What the cross-validation loop enclosed" not in text


# ── 4 · ignorance is not an answer either ────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_a_caller_who_did_not_say_gets_neither_claim(shape):
    """Trap 9. `None` is not `[]`, and reporting it as one asserts a fact.

    A draft assembled without per-model results cannot say whether CV ran. It
    must not claim it did, and it must not claim it did not — it says what it
    knows, which is that the box was ticked, and states the silence.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=None)

    assert "cross-validation was used for internal validation." not in text
    assert "produced results for no model" not in text
    assert "enabled in the training configuration" in text
    assert "does not report whether cross-validation" in text
    assert "What the cross-validation loop enclosed" not in text

    narrative = _narrative(task_type, target, ["nn"], True, 5, None)
    assert "cross-validation was used for model evaluation." not in narrative
    assert "produced results for no model" not in narrative
    assert "enabled in the training configuration" in narrative


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_an_explicit_empty_list_is_a_positive_record_and_reads_as_one(shape):
    """The other half of the same distinction.

    A caller who looked and found nothing cross-validated has recorded a fact,
    and the Methods section is entitled to state it. `pages/06` and `pages/10`
    both pass a list for exactly this reason.
    """
    task_type, target = SHAPES[shape]
    text = _methods(task_type, target, cv_folds=5,
                    selected_model_results=None, cv_models_run=[])
    assert "produced results for no model" in text
    assert "enabled in the training configuration" not in text


# ── 5 · the record that carries the signal ───────────────────────────────────

def test_the_provenance_records_which_models_were_cross_validated():
    """The plumbing, asserted end to end rather than at either end.

    `record_training` -> `TrainingProvenance` -> `get_methods_context` is what
    carries the fact from `pages/06` to the narrative composer. A field that
    never reaches the context is trap #1: a capability beside a path that never
    consumes it.
    """
    prov = WorkflowProvenance()
    prov.record_upload(target_col="y", task_type="classification",
                       feature_cols=["a"], n_samples=100)
    prov.record_training(models_trained=["rf", "nn"], primary_model="rf",
                         use_cv=True, cv_folds=5, cv_models_run=["rf"])
    ctx = prov.get_methods_context()
    assert ctx["cv_models_run"] == ["rf"]
    assert ctx["use_cv"] is True and ctx["cv_folds"] == 5

    # And the unknown state survives the trip rather than collapsing to [].
    other = WorkflowProvenance()
    other.record_upload(target_col="y", task_type="classification",
                        feature_cols=["a"], n_samples=100)
    other.record_training(models_trained=["rf"], primary_model="rf",
                          use_cv=True, cv_folds=5)
    assert other.get_methods_context()["cv_models_run"] is None, (
        "a record that never said which models were cross-validated now "
        "asserts that none were")


def test_the_page_that_trains_derives_the_record_from_the_results_it_has():
    """Trap #1's flip: change the recorded thing and see the sentence move.

    `pages/06` is a Streamlit script and cannot be imported here, so this
    asserts the derivation it performs — `cv_results` truthiness per model — is
    the one the composer's own resolution agrees with, on the same objects.
    """
    derived = [name for name, res in RF_RAN_NN_DID_NOT.items()
               if isinstance(res, dict) and res.get("cv_results")]
    assert derived == ["rf"]

    text = _methods("classification", "outcome", cv_folds=5,
                    selected_model_results=RF_RAN_NN_DID_NOT)
    explicit = _methods("classification", "outcome", cv_folds=5,
                        selected_model_results=RF_RAN_NN_DID_NOT,
                        cv_models_run=derived)
    assert _cv_lines(text) == _cv_lines(explicit), (
        "deriving the flag from the results and passing it explicitly "
        "disagree, so one of the two callers is composing a different claim")


def _cv_lines(text):
    return [" ".join(s.split())
            for s in re.findall(r"[^.]*cross-validation[^.]*\.", text)]
