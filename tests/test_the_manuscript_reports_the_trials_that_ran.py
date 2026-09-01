"""The Optuna budget is a control, and the manuscript reports the one used.

The trial count was the literal `30` at the call site in
`pages/06_Train_and_Compare.py`, which also made the `n_trials=30` default in
`optimize_model_hyperparameters`'s own signature unreachable — nothing short of
editing the file could change it. It is a slider now, and **30 remains the
default**, because this is a resource control and not a methods change: an
untouched session resolves to the same 30 and therefore draws the same trial
sequence and selects the same hyperparameters as before.

Making it settable creates exactly one new way for the app to say something
false, and that is what most of this file guards. Two methods generators
described the search:

- `ml/publication.py` appended "(30 trials per model)" as a **literal**. It was
  accidentally true only while the call site was hardcoded to the same number;
  the first user to pick 50 would have got a manuscript asserting 30.
- `ml/narrative_engine.py` — the PRIMARY path, the one page 10 uses whenever
  the provenance record is non-empty — said the search was a **grid search**.
  The app has never run one. It builds an Optuna study whose default sampler is
  a tree-structured Parzen estimator, and scores each trial with a single fit
  against the held-out validation split.

Both are the `AUDIT-026` fault in a different field: Methods asserting a design
the run did not perform. So the count travels from the run to both generators,
and where it was not recorded the sentence names the method without inventing a
number for it.

The last test is the one that matters for the PR's scope rule: the disclosure
this PR adds is arithmetic on a caption, and the Optuna default it reads is
still 30.
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE_06 = os.path.join(ROOT, "pages", "06_Train_and_Compare.py")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code_only(src):
    """`src` as its code tokens run together, with comments and strings dropped.

    This file asserts on both what the page DOES and what it deliberately does
    not do, and the comments that explain the second necessarily quote the
    first. Dropping them keeps an explanation from failing the assertion it
    explains. Tokens are concatenated with no separator, so `trial . report`
    still matches "trial.report" however the line was spaced.
    """
    import io
    import tokenize

    return "".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING)
    )


# ── the Methods sentence in the fallback generator ────────────────────────

def _methods_kwargs(**extra):
    """Minimum viable call to `generate_methods_section`."""
    base = dict(
        data_config={},
        preprocessing_config={},
        model_configs={"ridge": {}},
        split_config={},
        n_total=100, n_train=70, n_val=15, n_test=15,
        feature_names=["a", "b"],
        target_name="y",
        task_type="regression",
        metrics_used=["RMSE"],
    )
    base.update(extra)
    return base


def test_methods_section_reports_the_recorded_trial_count():
    from ml.publication import generate_methods_section

    text = generate_methods_section(**_methods_kwargs(
        hyperparameter_optimization=True,
        hyperparameter_optimization_trials=50,
    ))
    assert "Optuna (50 trials per model)" in text, text
    # The old literal must not survive anywhere in the sentence.
    assert "30 trials per model" not in text


def test_methods_section_names_no_number_when_the_count_was_not_recorded():
    """A record written before the field existed must not be backfilled with 30.

    `None` is a caller who did not say, and reporting it as the old default
    would assert a budget nobody recorded — the distinction `cv_models_run`
    draws for cross-validation, drawn again here.
    """
    from ml.publication import generate_methods_section

    text = generate_methods_section(**_methods_kwargs(
        hyperparameter_optimization=True,
    ))
    assert "Hyperparameter optimization was performed using Optuna." in text
    assert "trials per model" not in text
    assert "30" not in text.split("Hyperparameter optimization")[1][:80]


def test_methods_section_is_silent_when_no_optimization_ran():
    from ml.publication import generate_methods_section

    text = generate_methods_section(**_methods_kwargs(
        hyperparameter_optimization=False,
        hyperparameter_optimization_trials=50,
    ))
    assert "Hyperparameter optimization" not in text


def test_no_generator_still_hardcodes_thirty_trials():
    """The literal is gone from the emitted text, not merely unreachable.

    Checked over string CONSTANTS rather than the raw file, so the comments
    that explain why the literal was wrong do not trip their own assertion.
    """
    for rel in ("ml/publication.py", "ml/narrative_engine.py"):
        tree = ast.parse(_read(os.path.join(ROOT, rel)))
        offenders = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and "30 trials" in node.value
        ]
        assert not offenders, f"{rel} still emits a hardcoded count: {offenders}"


# ── the primary path: provenance → NarrativeEngine ────────────────────────

def _provenance_with(trials):
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(
        models_trained=["ridge"],
        primary_model="ridge",
        use_cv=True,
        cv_folds=5,
        cv_models_run=["ridge"],
        use_hyperopt=True,
        hyperopt_trials=trials,
    )
    return prov


def test_the_trial_count_reaches_the_methods_context():
    prov = _provenance_with(50)
    assert prov.training.hyperopt_trials == 50
    assert prov.get_methods_context()["hyperopt_trials"] == 50


def test_narrative_reports_optuna_and_the_recorded_count():
    from ml.narrative_engine import NarrativeEngine

    text = NarrativeEngine(_provenance_with(50)).generate().model_development
    assert "Optuna" in text, text
    assert "50 trials per tunable model" in text, text
    # The method it never used.
    assert "grid search" not in text.lower(), text


def test_narrative_names_the_method_without_a_number_when_unrecorded():
    from ml.narrative_engine import NarrativeEngine

    text = NarrativeEngine(_provenance_with(None)).generate().model_development
    assert "Optuna" in text, text
    assert "trials per tunable model" not in text, text
    assert "grid search" not in text.lower(), text


def test_the_count_survives_a_provenance_round_trip():
    from utils.workflow_provenance import WorkflowProvenance

    restored = WorkflowProvenance.from_dict(_provenance_with(50).to_dict())
    assert restored.training.hyperopt_trials == 50


def test_a_record_written_before_the_field_existed_still_loads():
    """`from_dict` filters on dataclass fields, so an older payload is fine."""
    from utils.workflow_provenance import WorkflowProvenance

    payload = _provenance_with(50).to_dict()
    payload["training"].pop("hyperopt_trials")
    restored = WorkflowProvenance.from_dict(payload)
    assert restored.training.use_hyperopt is True
    assert restored.training.hyperopt_trials is None


# ── the control itself, read off page 06's source ─────────────────────────
#
# Page 06 is a Streamlit script: importing it runs the page. These read the
# source, which is enough for the two properties that matter — the literal is
# gone from the call site, and the default is still 30.

def test_the_optuna_call_site_no_longer_hardcodes_the_trial_count():
    """The signature default of 30 stays; the CALL SITE stops overriding it.

    Passing the literal there is what made the signature default unreachable,
    so no session value, config or launcher could influence the budget.
    """
    tree = ast.parse(_read(PAGE_06))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "optimize_model_hyperparameters"
    ]
    assert len(calls) == 1, f"expected one call site, found {len(calls)}"
    passed = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "n_trials" in passed
    assert isinstance(passed["n_trials"], ast.Name), (
        "the call site still passes a literal trial count"
    )
    assert passed["n_trials"].id == "n_optuna_trials"

    # …and the signature default it now defers to is still 30.
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "optimize_model_hyperparameters")
    names = [a.arg for a in fn.args.args]
    defaults = dict(zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults))
    assert defaults["n_trials"].value == 30


def test_thirty_is_still_the_default_so_no_existing_result_moves():
    """The whole scope rule for this PR rests on this line.

    Both the slider's initial value and `_train_models`'s own read fall back to
    30, so a session that never touches the control runs the identical search.
    """
    src = _read(PAGE_06)
    assert src.count("st.session_state.get('optuna_trials', 30)") == 2, (
        "expected the slider default and _train_models to both fall back to 30"
    )


# ── the disclosure, as the page actually renders it ───────────────────────
#
# Rendered rather than grepped, because the two things worth guarding are a
# number and a crash, and neither is visible in the source.


def _render_page_06(models, use_cv, trials=None):
    """Render page 06 far enough to reach the buttons, with `models` selected."""
    from streamlit.testing.v1 import AppTest
    from tests.integration.conftest import (
        build_test_dataframe, inject_data_state, inject_trained_state,
    )
    from ml.pipeline import build_preprocessing_pipeline

    df = build_test_dataframe()
    at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=120)
    inject_data_state(at, df)
    inject_trained_state(at, df)

    feats = list(at.session_state["X_train"].columns)
    pipe = build_preprocessing_pipeline(
        numeric_features=feats, categorical_features=[],
        numeric_imputation="median", numeric_scaling="standard",
    )
    pipe.fit(at.session_state["X_train"])
    at.session_state["fitted_preprocessing_pipelines"] = {"ridge": pipe}
    at.session_state["preprocessing_pipelines_by_model"] = {"ridge": pipe}
    at.session_state["feature_names_by_model"] = {"ridge": feats}

    for model in models:
        at.session_state[f"train_model_{model}"] = True
    at.session_state["use_cv"] = use_cv
    at.session_state["cv_folds"] = 5
    if trials is not None:
        at.session_state["optuna_trials"] = trials

    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    return at


def _fits_caption(at):
    for cap in at.caption:
        text = str(cap.value)
        if "Train Models" in text and "runs" in text:
            return text
    raise AssertionError("the fits disclosure did not render")


def test_the_disclosure_states_the_work_additively():
    """Folds are NOT inside the Optuna search.

    The objective fits once on the pre-transformed training matrix and scores
    the held-out validation split, so the per-model cost is
    `trials + 1 final fit + cv_folds` — 30 + 1 + 5 = 36 at the defaults, and a
    three-model run is 78, not the ~450 a `trials x models x folds` product
    would claim.
    """
    at = _render_page_06(["ridge"], use_cv=True)
    text = _fits_caption(at)
    assert "runs 6 fits (1 final fit + 1 model × 5 CV folds)" in text, text
    assert "runs 36 (1 tunable model × 30 Optuna trials, then the same 6)" in text, text


def test_the_disclosure_survives_cross_validation_being_off():
    """`cv_folds` is bound only inside `if use_cv:`, far above the buttons.

    A disclosure that read the bare local would raise NameError and take the
    whole page down whenever the box is unchecked. Gating on the checkbox is
    also what makes the number right rather than merely non-crashing.
    """
    at = _render_page_06(["ridge"], use_cv=False)
    text = _fits_caption(at)
    assert "runs 1 fit (1 final fit)." in text, text
    assert "CV folds" not in text, text
    assert "runs 31 (" in text, text


def test_a_model_with_no_schema_costs_no_trials_and_is_named():
    """Optimization is gated on a non-empty `hyperparam_schema`.

    GLM, Gaussian NB and LDA ship an empty one, so counting every selected
    model as tunable would overstate the search. They are named rather than
    silently dropped from the arithmetic.
    """
    at = _render_page_06(["ridge", "rf", "glm"], use_cv=True)
    text = _fits_caption(at)
    assert "runs 18 fits (3 final fits + 3 models × 5 CV folds)" in text, text
    assert "runs 78 (2 tunable models × 30 Optuna trials" in text, text
    assert "GLM has no tunable hyperparameters and costs no trials." in text, text


def test_the_slider_value_reaches_the_disclosure():
    at = _render_page_06(["ridge"], use_cv=True, trials=50)
    labels = {s.label: s.value for s in at.slider}
    assert labels.get("Optuna Trials") == 50, labels
    assert "× 50 Optuna trials" in _fits_caption(at)


def test_the_trials_slider_defaults_to_thirty_on_an_untouched_session():
    at = _render_page_06(["ridge"], use_cv=True)
    labels = {s.label: s.value for s in at.slider}
    assert labels.get("Optuna Trials") == 30, labels


def test_the_two_to_five_minute_promise_is_gone():
    """An existing WRONG disclosure outranks a missing one.

    Page 06 promised "typically takes 2-5 minutes with optimization" for every
    model in `slow_models`. SVC's own factory sets `probability=True`, so one
    fit is six libsvm solves, and a measured single fit at 20,000 x 120 took
    9.1 minutes — a 30-trial search there is hours. The replacement states the
    trial count in force and SVC's inner multiplier instead of a constant that
    was never measured.
    """
    src = _read(PAGE_06)
    # Over string CONSTANTS, so the comment that quotes the retired promise in
    # order to explain it does not trip its own assertion.
    emitted = [node.value for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not [s for s in emitted if "2-5 minutes with optimization" in s]
    assert any("6 internal calibration fits" in s for s in emitted), (
        "SVC's calibration multiplier should be disclosed where its cost lands"
    )


def test_no_pruner_or_timeout_was_added():
    """Deliberately out of scope: both change which trials complete.

    Optuna already attaches a MedianPruner by default, but the objective never
    calls `trial.report()`/`should_prune()`, so it cannot fire. Instrumenting
    it, or passing `timeout=`, changes which hyperparameters win — a result
    change, and it belongs in its own PR.
    """
    src = _read(PAGE_06)
    assert "optuna.create_study(direction=direction)" in src, (
        "the study is still built with no pruner and no sampler override"
    )
    assert ("study.optimize(_objective, n_trials=n_trials, "
            "show_progress_bar=False, callbacks=_callbacks)") in src, (
        "the search still runs to its trial count with no wall-clock timeout"
    )
    # Over CODE only — the comment at the study site explains why the pruner is
    # inert, and naming the calls it would need must not trip this.
    code = _code_only(src)
    for marker in ("trial.report", "should_prune", "TrialPruned"):
        assert marker not in code, f"{marker} would make the default pruner live"


def test_page_06_still_parses():
    ast.parse(_read(PAGE_06))
