"""`AUDIT-026` / `AUDIT-029` — the Methods section names the CV that RAN.

## Why this file exists beside the one already named on both rows

`tests/test_the_methods_section_reports_the_cross_validation_that_ran.py` is the
regression the L51-C merge left behind, and the adjudicator declined it as
evidence for a reason worth restating: **every one of its reverted failures was
a `TypeError` on an unexpected keyword argument or a `NameError`.** Reverting
the fix removed the `cv_models_run` parameter, so the tests that pass it stopped
being able to CALL the composer — which proves a signature changed and never
that the manuscript said something false. That is trap #2's family: a guard
testing its own description.

**Every claim in this file is composed from arguments that existed before the
fix.** The fallback composer is driven through `selected_model_results` (an
argument since the function was written) and the primary composer through
`manuscript_context` (likewise), so reverting the correction leaves both calls
valid and the failure is the SENTENCE, quoted.

## The two sentences these tests hold down

The pre-fix composers, verbatim from `40161c1`:

    ml/publication.py:1184     f" {cv_to_use}-fold cross-validation was used for
                                 internal validation."          # if cv_to_use
    ml/narrative_engine.py:1040 f"{cv_folds}-fold cross-validation was used for
                                 model evaluation."         # if use_cv and cv_folds

Both read the CHECKBOX. `pages/06:1455` excludes the neural network from CV
outright and `:1489` catches a CV failure, warns, and continues with
`cv_results=None`, so *configured* and *ran* come apart on the ordinary path —
which is what "the configured CV and the run CV differ" means below.

And `AUDIT-029`'s paragraph, same file, pre-fix:

    "**Cross-validation and preprocessing:** {n}-fold cross-validation was
     performed on data that had already been preprocessed using the full
     training set."

False since `STATE-059`: `pages/06:1478-1487` hands `make_cv_pipeline` the RAW
training partition and `ml/eval.py:182-197` clones the preprocessing into the
composite, so every fold re-fits it.

## Positive controls

`GUIDED-045`. Three of the assertions here are absences. Each one is preceded by
an assertion that the thing being swept is non-empty — that the composed text
contains cross-validation sentences at all — because "the false sentence is
gone" and "the section is gone" read identically to a passing test.
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

#: `GUIDED-097`. Both composers branch on `task_type` for the prose around the
#: CV sentence, so every claim runs against a binary 0/1 outcome and a
#: continuous one.
SHAPES = {
    "binary_0_1": ("classification", "died_30d"),
    "continuous": ("regression", "hba1c"),
}

#: The shapes these claims are NOT checked against, named rather than left to
#: be assumed covered.
SHAPES_NOT_COVERED = {
    "multiclass": (
        "`task_type` is the only shape either composer reads; a three-level "
        "outcome takes the identical path to `binary_0_1` for every sentence "
        "here."),
    "string_valued_binary_outcome": (
        "Neither composer receives the label vector — only `task_type` and "
        "metric dicts — so a string outcome cannot reach these sentences "
        "differently. Where it matters is `pages/06`, which is a Streamlit "
        "script and is not driven by this file."),
    "time_to_event": (
        "The app has no survival task type, so there is no such shape to "
        "cover."),
}

#: Configured: cross-validation, five folds. Ran: the random forest only. This
#: is the `pages/06:1455` case — the neural network is excluded from CV by the
#: page itself while the checkbox stays ticked.
CONFIGURED_5_RAN_RF_ONLY = {
    "rf": {"metrics": {"AUC": 0.83, "R2": 0.41},
           "cv_results": {"mean": 0.80, "std": 0.02}},
    "nn": {"metrics": {"AUC": 0.81, "R2": 0.38}, "cv_results": None},
}

#: Configured: cross-validation. Ran: nothing — `pages/06:1489` swallowed the
#: exception, or the neural network was the only model trained.
CONFIGURED_5_RAN_NOTHING = {
    "nn": {"metrics": {"AUC": 0.81, "R2": 0.38}, "cv_results": None},
}

#: A frozen export that never mentions cross-validation at all. Not the same
#: object as "ran for nothing", and must not be read as one.
SAYS_NOTHING_ABOUT_CV = {
    "rf": {"metrics": {"AUC": 0.83, "R2": 0.41}},
    "nn": {"metrics": {"AUC": 0.81, "R2": 0.38}},
}


def _cv_sentences(text):
    """Every sentence in the composed text that mentions cross-validation."""
    return [" ".join(s.split())
            for s in re.findall(r"[^.]*cross-validation[^.]*\.", text or "")]


def _fallback_methods(task_type, target, **kw):
    """`ml.publication.generate_methods_section` — `pages/10:684`'s fallback.

    Only arguments that predate the `AUDIT-026` fix are passed.
    """
    args = dict(
        data_config={"feature_cols": ["a", "b"], "target_col": target},
        preprocessing_config={},
        model_configs={"rf": {}, "nn": {}},
        split_config={"stratify": True},
        n_total=300, n_train=200, n_val=50, n_test=50,
        feature_names=["a", "b"], target_name=target,
        task_type=task_type, metrics_used=["AUC"],
    )
    args.update(kw)
    return generate_methods_section(**args)


def _exported_methods(task_type, target, frozen_results, cv_folds=5):
    """`ml.narrative_engine` — the PRIMARY export path, `pages/10:674`.

    `record_training` is called with the CHECKBOX and nothing else, which is
    what every run recorded before `cv_models_run` existed and what
    `pages/06` logged for years. The frozen export results are handed over as
    `manuscript_context`, exactly as `pages/10:674` hands them over.
    """
    prov = WorkflowProvenance()
    prov.record_upload(target_col=target, task_type=task_type,
                       feature_cols=["a", "b"], n_samples=300)
    prov.record_split(strategy="stratified", train_n=200, val_n=50, test_n=50)
    prov.record_training(models_trained=list(frozen_results), primary_model="rf",
                         use_cv=True, cv_folds=cv_folds)
    engine = NarrativeEngine(
        prov, manuscript_context={"selected_model_results": frozen_results})
    return engine.generate().to_markdown()


# ── 1 · configured and ran DIFFER, and the sentence names the second ─────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_fallback_methods_section_names_the_models_cross_validation_ran_for(
        shape):
    """`AUDIT-026` on `ml/publication`, configured 5-fold, ran for one of two.

    The checkbox says cross-validation; the run cross-validated the random
    forest and not the neural network. The pre-fix sentence — *"5-fold
    cross-validation was used for internal validation."* — is true of neither
    model set on its own and reads as covering both.
    """
    task_type, target = SHAPES[shape]
    text = _fallback_methods(task_type, target, cv_folds=5,
                             selected_model_results=CONFIGURED_5_RAN_RF_ONLY)

    # Positive control before the absence (`GUIDED-045`).
    assert _cv_sentences(text), (
        "no sentence in the composed Methods section mentions "
        "cross-validation at all, so the absence below proves nothing")

    assert "cross-validation was used for internal validation." not in text, (
        "the Methods section asserts cross-validation as the internal "
        "validation without naming which models it ran for, while "
        "cv_results is None for the neural network")
    assert "5-fold cross-validation was used for internal validation of" in text
    assert "Random Forest" in text
    assert "It was not run for Neural Network" in text
    assert "held-out split alone" in text


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_exported_methods_section_names_the_models_cross_validation_ran_for(
        shape):
    """`AUDIT-026` on `ml/narrative_engine` — the PRIMARY export path.

    The record holds the checkbox only. The frozen export results hold the
    answer, and `pages/10:674` already hands them to this composer, so the
    Methods section can name the models a fold loop scored instead of
    asserting a design over all of them.
    """
    task_type, target = SHAPES[shape]
    text = _exported_methods(task_type, target, CONFIGURED_5_RAN_RF_ONLY)

    assert _cv_sentences(text), (
        "the exported draft mentions cross-validation nowhere, so the "
        "absence below proves nothing")

    assert "cross-validation was used for model evaluation." not in text, (
        "the exported Methods section asserts cross-validation from the "
        "checkbox, over a run whose neural network was never cross-validated")
    assert "5-fold cross-validation was used for evaluation of" in text
    assert "Random Forest" in text
    assert "It was not run for Neural Network" in text


# ── 2 · configured, and it ran for nothing ───────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_both_composers_decline_the_claim_when_no_fold_loop_scored_anything(
        shape):
    """The row's first-named case: box ticked, neural network only, no CV.

    The claim is corrected rather than dropped — the section still states what
    the internal validation IS, which is the single split.
    """
    task_type, target = SHAPES[shape]

    fallback = _fallback_methods(task_type, target, cv_folds=5,
                                 selected_model_results=CONFIGURED_5_RAN_NOTHING)
    assert _cv_sentences(fallback)
    assert "cross-validation was used for internal validation." not in fallback
    assert "produced results for no model" in fallback
    assert "train/validation/test split" in fallback

    exported = _exported_methods(task_type, target, CONFIGURED_5_RAN_NOTHING)
    assert _cv_sentences(exported)
    assert "cross-validation was used for model evaluation." not in exported
    assert "produced results for no model" in exported
    assert "held-out split alone" in exported


# ── 3 · a record that never spoke is not a denial ────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_a_frozen_export_that_never_mentions_cv_is_not_read_as_a_denial(shape):
    """Trap 9, pointed the other way.

    Deriving *no model was cross-validated* from payloads that carry no
    `cv_results` key would make the correction commit the row's own mistake in
    the opposite direction. The composer must state the silence instead.
    """
    task_type, target = SHAPES[shape]
    text = _exported_methods(task_type, target, SAYS_NOTHING_ABOUT_CV)

    assert _cv_sentences(text)
    assert "cross-validation was used for model evaluation." not in text
    assert "produced results for no model" not in text, (
        "a frozen export that never mentions cross-validation was reported as "
        "a run that cross-validated nothing")
    assert "enabled in the training configuration" in text


# ── 4 · `AUDIT-029`: the paragraph describes the loop the code runs ──────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_methods_section_does_not_describe_a_leak_the_pipeline_prevents(
        shape):
    """`AUDIT-029`, driven with pre-fix arguments only.

    `ml/eval.make_cv_pipeline` clones the preprocessing into the CV composite,
    so the fold loop re-fits it; the paragraph asserted the opposite. It must
    also keep disclosing what the loop does NOT enclose — selection and tuning
    sit outside it and no optimism correction is applied — otherwise the
    correction is a whitewash.
    """
    task_type, target = SHAPES[shape]
    text = _fallback_methods(task_type, target, cv_folds=5,
                             selected_model_results=CONFIGURED_5_RAN_RF_ONLY)

    assert "### Methodological Considerations" in text, (
        "the section that carries this paragraph is absent, so the absence "
        "below proves nothing")

    assert "already been preprocessed using the full training set" not in text, (
        "the Methods section describes cross-validation scoring "
        "pre-preprocessed data — a leak ml/eval.py:182-197 structurally "
        "prevents")
    assert "re-fit inside" in text and "each fold" in text
    assert "no optimism correction" in text
    assert "feature selection and hyperparameter tuning" in text


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_fold_paragraph_is_absent_when_no_fold_loop_ran(shape):
    """A paragraph explaining what the fold loop enclosed, printed for a run
    with no fold loop, is `AUDIT-026` restated at paragraph length."""
    task_type, target = SHAPES[shape]
    text = _fallback_methods(task_type, target, cv_folds=5,
                             selected_model_results=CONFIGURED_5_RAN_NOTHING)
    assert "### Methodological Considerations" in text
    assert "What the cross-validation loop enclosed" not in text
