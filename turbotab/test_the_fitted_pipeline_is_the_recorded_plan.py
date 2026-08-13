"""`GUIDED-095` / `GUIDED-089` — a recorded decision reaches the thing it was
recorded for.

## The defect, in one sentence

Clause §06 splits preprocessing in two: row-local operations, which may run
now, and stateful ones, *recorded now and fitted inside each training fold*.
The first class reached the model because executing rewrites `working_table`
and the trainer read `working_table`. **The second class reached nothing** —
the record was written, the receipt counted it, the sentence was composed, and
the executor did not exist.

## What this file asserts, and in what order

**Hardest first**, because that is where the abstraction bends: `leave` and
`indicator` on a model that cannot read a blank. It is the one case where
per-model divergence is forced, and if the plan survives it the rest is
fill-out.

1. A declaration that keeps the blank keeps it **in the fitted transformer** —
   read off the fitted `ColumnTransformer`, not off the payload.
2. The same declaration on a linear model **diverges, and says so**, per model,
   naming both the recorded sentence and the one now true of the fit.
3. The blank survives into a fit where it actually reaches the estimator, which
   the case in (1) does not on its own fixture — see the correction below.
4. **The sentence and the pipeline are one object**, asserted as identity: the
   plan's step carries the record's own string, not a copy that agrees today.
5. A recipe variant changes the fitted transformer AND the methods sentence.
6. A deferred transform is fitted, at last, and its sentence is the record's.
7. Where the plan cannot honor something it **refuses** rather than
   substituting — a decision the record accepts and the pipeline silently drops
   is the defect this module was written to remove, arriving from inside.

## A correction to `GUIDED-089`'s own measurement, made while building the gate

The row records: *"`metabolomics_untargeted.csv`, column `bmi`, 8 blanks of 80
… `_pipeline` puts `bmi` in the numeric block and `SimpleImputer` fills those 8
blanks with the median, 27.15."* The first half is exactly right and is the
defect. **The second half is not true on that fixture**, and it matters that
the report says so: all 8 blank-`bmi` rows are `pooled_qc` samples with no
`responder`, so `train`'s outcome mask drops them from both partitions before
the pipeline sees a row. Measured: 8 blanks in the frame, 0 in `X_train`, 0 in
`X_test`.

What was real was the **fidelity** defect, and it was real for every column
whose blanks do sit in modeled rows — `mz_0003` has 42 — and for `bmi` in the
sense that mattered: the recorded methods sentence said the value is left blank
while the pipeline placed the column in the median-impute block, so the two
disagreed about what the analysis was. That is why (3) exists beside (1).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, eventfixture, pipeline_plan, training       # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client, name, target, *, fraction=0.25):
    with open(DATA / name, "rb") as fh:
        pid = client.post("/project", files={
            "file": (name, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])
        return r.json()

    decide("set_target", column=target)
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=fraction)
    # `DRIVE-041`. Over the route, and only where the engine asks.
    eventfixture.choose_event_over_http(client, pid, target)
    return pid, api.STORE.get(pid), decide


def _partitions(project):
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    features = training._feature_frame(table, target,
                                       (project.grain or {}).get("group_col"))
    X_train = features[has_y & ~is_test]
    return features, X_train, table.loc[X_train.index, target]


def _fit(project, model_key, task):
    from ml.model_registry import get_registry

    features, X_train, y_train = _partitions(project)
    plan = pipeline_plan.compose(project, model_key, features, seed=42)
    pipe = plan.build(get_registry()[model_key].factory(task, 42))
    pipe.fit(X_train, y_train)
    return plan, pipe, features


def _block_of(prep, column):
    """Which fitted block a column was routed into."""
    return [entry[0] for entry in prep.transformers_
            if column in list(entry[2])]


# ── 1 · the hardest case: a blank a model cannot read ────────────────────────

def test_an_indicator_declaration_leaves_the_blank_in_the_fitted_transformer(
        client):
    """**The gate**, asserted on the fitted transformer rather than the payload.

    A user answers the mechanism `informative` — a blank here means something —
    and picks `indicator`, whose whole purpose is to keep that signal without
    inventing a value. Preprocess does its half: it creates `bmi_was_missing`,
    leaves the 8 blanks, and composes *"the underlying value is left blank."*

    Gradient boosting reads a blank natively, so nothing may fill it. Read off
    the fitted `ColumnTransformer`: `bmi` must be in the passthrough block, and
    transforming a frame that still contains the blanks must give them back.
    """
    pid, project, decide = _sealed(client, "metabolomics_untargeted.csv",
                                   "responder")
    blanks = int(project.df["bmi"].isna().sum())
    assert blanks == 8, (blanks, "the fixture changed under this claim")
    decide("route_missingness", column="bmi", mechanism="informative",
           strategy="indicator")

    declared = [d for d in project.missingness if d["column"] == "bmi"][0]
    assert declared["strategy"] == "indicator"
    assert "left blank" in declared["sentence"]
    assert "bmi_was_missing" in project.df.columns, (
        "the row-local half did not run, so this claim is about the wrong "
        "defect")

    plan, pipe, features = _fit(project, "histgb_clf", "classification")
    prep = pipe.named_steps["prep"]
    assert _block_of(prep, "bmi") == ["keep_blank"], (
        "`bmi` was routed into an imputer on a model that reads a blank "
        f"natively, against a recorded `indicator`: {_block_of(prep, 'bmi')}")

    # THE FITTED TRANSFORMER, applied to rows that still carry the blanks.
    out = prep.transform(features)
    assert int(pd.DataFrame(out)["bmi"].isna().sum()) == blanks, (
        "the fitted pipeline filled a blank the record said to keep")
    assert not plan.divergences, (
        "a model that reads a blank natively should not need to diverge")


def test_the_same_declaration_diverges_on_a_model_that_cannot_read_a_blank(
        client):
    """**Rule 2, and the case that bends the abstraction.**

    Linear and neural models cannot be fitted around a blank; the card's own
    copy says gradient boosting handles one natively and they do not. So the
    declaration cannot be honored — and that is a **stated divergence**, per
    model, never a silent substitution and never a shortened plan.

    The recorded sentence has become false for this model, so the step no
    longer carries it: the plan reports both, and a reader can see they differ.
    """
    pid, project, decide = _sealed(client, "metabolomics_untargeted.csv",
                                   "responder")
    decide("route_missingness", column="bmi", mechanism="informative",
           strategy="indicator")
    recorded = [d for d in project.missingness if d["column"] == "bmi"][0]

    plan, pipe, features = _fit(project, "logreg", "classification")
    assert _block_of(pipe.named_steps["prep"], "bmi") == ["fill_impute_median"]

    diverged = [d for d in plan.divergences if d.subject == "bmi"]
    assert diverged, (
        "a linear model silently filled a blank the record said to keep, and "
        "said nothing — which is exactly GUIDED-089 with a new pipeline")
    d = diverged[0]
    assert d.requested == "indicator" and d.applied == "impute_median"
    assert d.recorded_sentence == recorded["sentence"]
    assert d.fitted_sentence != d.recorded_sentence, (
        "the divergence reprints the recorded sentence, which is now false "
        "for this model")
    assert "Logistic Regression" in d.fitted_sentence, (
        "the divergence does not name the model it applies to, so a reader "
        "cannot tell which fit it is about")

    # AND THE STEP CARRIES THE TRUE SENTENCE, not the recorded one.
    step = plan.step_for("bmi")
    assert step.sentence == d.fitted_sentence

    # THE PLAN IS NEVER SHORTENED. The model still fits and still scores.
    run = training.train(project, ["logreg"])
    assert run.results[0].metrics, run.results[0].error
    assert any(d.fitted_sentence == note for note in run.notes), (
        "the divergence is in the plan and not in the run's notes, so a "
        "reader of the results never learns of it")


def test_the_blank_survives_into_a_fit_where_it_reaches_the_estimator(client):
    """(1) proves the routing; this proves it on a column whose blanks are
    actually modeled.

    `bmi`'s 8 blanks are all `pooled_qc` rows with no `responder`, so the
    outcome mask drops them before any pipeline sees them — see this module's
    docstring. `mz_0003` has 42 blanks in modeled rows, so here the kept blank
    reaches the estimator and the fitted coefficients are a fit over data
    containing one.
    """
    pid, project, decide = _sealed(client, "metabolomics_untargeted.csv",
                                   "responder")
    features, X_train, _ = _partitions(project)
    in_fit = int(X_train["mz_0003"].isna().sum())
    assert in_fit > 0, (
        "no blank in this column reaches the fit, so this claim would pass "
        "against a pipeline that filled every one of them")

    decide("route_missingness", column="mz_0003", mechanism="informative",
           strategy="leave")
    plan, pipe, features = _fit(project, "histgb_clf", "classification")
    _, X_train, _ = _partitions(project)
    prepared = pd.DataFrame(pipe.named_steps["prep"].transform(X_train))
    assert int(prepared["mz_0003"].isna().sum()) == in_fit, (
        "the blanks the model was asked to read natively were filled before "
        "it saw them")


# ── 2 · the sentence and the pipeline are one object ─────────────────────────

def test_the_sentence_and_the_pipeline_are_one_object(client):
    """**Rule 1, asserted as identity rather than as equality.**

    L34 applied this discipline to the transcript / receipt / methods triple.
    Same here: the methods sentence is derived from the same spec that builds
    the pipeline. Two strings that happen to agree today are two strings, and
    drift between them is precisely this defect — the recorded line said the
    blank was left and the fit filled it.
    """
    pid, project, decide = _sealed(client, "metabolomics_untargeted.csv",
                                   "responder")
    decide("route_missingness", column="mz_0005", mechanism="not_sure",
           strategy="impute_mean")
    recorded = [d for d in project.missingness if d["column"] == "mz_0005"][0]

    features, _, _ = _partitions(project)
    plan = pipeline_plan.compose(project, "histgb_clf", features)
    step = plan.step_for("mz_0005")
    assert step is not None and step.source == "missingness"
    assert step.sentence is recorded["sentence"], (
        "the plan composed its own sentence rather than carrying the record's, "
        "so the two can drift and nothing would notice")
    assert step.sentence in plan.sentences()


def test_a_column_nobody_answered_does_not_read_as_a_choice(client):
    """The inverse of the same rule, and it is easy to get wrong.

    A column with blanks and no recorded decision still needs an imputer. What
    it must not get is the declaration's own prose — *"Missing values in `x`
    will be filled using the median within each training fold"* reads as a
    decision somebody made. The run counts these separately for the same
    reason.
    """
    pid, project, decide = _sealed(client, "metabolomics_untargeted.csv",
                                   "responder")
    features, _, _ = _partitions(project)
    plan = pipeline_plan.compose(project, "histgb_clf", features)
    assert plan.undeclared, "this fixture has no undeclared blanks to check"

    step = plan.step_for(plan.undeclared[0])
    assert step.source == "default"
    assert "No handling was recorded" in step.sentence, step.sentence

    run = training.train(project, ["histgb_clf"])
    assert any("had no recorded handling" in n for n in run.notes), (
        "the run reports only what it honored, so a column nobody answered "
        "reads as a column somebody chose a default for")


# ── 3 · the other two inputs to the plan ─────────────────────────────────────

def test_a_recipe_variant_reaches_the_fitted_pipeline(client):
    """The recipe lattice resolved a variant per model per operation, and the
    trainer read none of it. Changing one must change the fitted transformer
    AND the sentence, or the lattice is a picture of a decision."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    pid, project, decide = _sealed(client, "leaky_sepsis.csv", "sepsis")
    decide("select_models", models=["logreg"])

    _, standard, _ = _fit(project, "logreg", "classification")
    kinds = [type(t) for _, t, _ in standard.named_steps["shape"].transformers_
             for t in ([t] if not hasattr(t, "steps") else
                       [s for _, s in t.steps])]
    assert StandardScaler in kinds, (
        "the table resolves `standard` for a linear model and no scaler was "
        "fitted")

    decide("set_model_recipe", model="logreg", operation="scale",
           variant="robust")
    plan, robust, _ = _fit(project, "logreg", "classification")
    kinds = [type(t) for _, t, _ in robust.named_steps["shape"].transformers_
             for t in ([t] if not hasattr(t, "steps") else
                       [s for _, s in t.steps])]
    assert RobustScaler in kinds and StandardScaler not in kinds, (
        "the user set `robust` and a standard scaler was fitted anyway")
    assert any("median and interquartile range" in s for s in plan.sentences()), (
        "the fitted pipeline changed and the methods sentence did not")


def test_a_deferred_transform_is_at_last_fitted(client):
    """`features.declare` produced a spec, `project.deferred_transforms` stored
    it, and nothing ever consumed it. This is the consumer."""
    from sklearn.preprocessing import KBinsDiscretizer

    pid, project, decide = _sealed(client, "leaky_sepsis.csv", "sepsis")
    numeric = [c for c in project.df.columns
               if pd.api.types.is_numeric_dtype(project.df[c])
               and c != "sepsis"]
    decide("defer_feature", transform="bin_quantile", columns=[numeric[0]],
           params={"n_bins": 4})
    assert project.deferred_transforms, "the deferral was not recorded"
    spec = project.deferred_transforms[0]

    plan, pipe, features = _fit(project, "histgb_clf", "classification")
    fitted = [t for name, t, _ in pipe.named_steps["shape"].transformers_
              if name.startswith("deferred_")]
    assert fitted, "the deferred transform reached no fitted pipeline"
    assert isinstance(fitted[0].named_steps["t"], KBinsDiscretizer)
    assert hasattr(fitted[0].named_steps["t"], "bin_edges_"), (
        "the discretizer was built and never fitted")

    step = [s for s in plan.steps if s.source == "deferred_transform"][0]
    assert step.sentence is spec["sentence"], (
        "the deferred step composed a second sentence beside the recorded one")

    # AND IT REACHES THE MODEL: the shape stage emits more columns than the
    # feature frame has, which is the binned column arriving.
    shaped = pipe.named_steps["shape"].transform(
        pipe.named_steps["prep"].transform(features))
    assert np.asarray(shaped).shape[1] > len(features.columns), (
        "the binned column was fitted and not handed to the estimator")


def test_a_categorical_declaration_reaches_the_fit_too(client):
    """The other branch, on a fixture with real text columns. `clinic_visits`
    rather than a synthetic frame, because a categorical plan that only works
    on invented data is a plan nobody has driven."""
    pid, project, decide = _sealed(client, "clinic_visits.csv", "hba1c")
    blanks = int(project.df["notes"].isna().sum())
    assert blanks == 35, blanks
    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")

    assert int(project.df["notes"].isna().sum()) == 0, (
        "the row-local half did not execute")
    plan, pipe, features = _fit(project, "histgb_reg", "regression")
    step = plan.step_for("notes")
    assert step.key == "explicit_category"
    assert step.sentence is [d for d in project.missingness
                             if d["column"] == "notes"][0]["sentence"]
    encoded = [name for name, _, cols in pipe.named_steps["shape"].transformers_
               if "notes" in list(cols)]
    assert encoded == ["cat"], encoded


# ── 4 · what it refuses ──────────────────────────────────────────────────────

def test_a_recorded_decision_the_pipeline_cannot_execute_is_refused(client):
    """**The property that keeps this module from becoming the defect it
    closed.** A decision the record accepts and the pipeline quietly drops is
    the same silence in the other direction, so an unknown deferred key raises
    rather than being skipped."""
    pid, project, decide = _sealed(client, "leaky_sepsis.csv", "sepsis")
    project.deferred_transforms.append({
        "key": "a_transform_nobody_built", "scope": "stateful",
        "columns": ["age"], "params": {}, "sentence": "…", "because": "…"})
    features, _, _ = _partitions(project)
    with pytest.raises(pipeline_plan.PlanRefusal) as caught:
        pipeline_plan.compose(project, "histgb_clf", features)
    assert "a_transform_nobody_built" in str(caught.value)

    # And the refusal reaches the caller as a refusal rather than as a crash.
    with pytest.raises(training.TrainingRefusal):
        training._plan(project, "histgb_clf", features)


def test_every_recipe_variant_the_table_offers_has_a_fitted_form(client):
    """The completeness half, and it is cheap. Every variant a user can select
    through `set_model_recipe` must build and must have a methods sentence — a
    variant the table offers and the pipeline cannot execute would be a
    silently-dropped decision waiting for the first user to pick it.

    **With every pack loaded**, which is the version that found something.
    `DOMAIN_PACKS.md` §02 lets a pack add operations and variants, not merely
    override them, and the metabolomics pack adds `pareto` to `scale`. Checked
    against the core table alone this passed while the first metabolomics fit
    would have raised — `GUIDED-095`'s shape arriving through the extension
    point rather than through the trainer.
    """
    from turbotab import packs as _packs, recipes as _rec

    state = _rec.snapshot()
    _packs.load(list(_packs.PACKS))
    try:
        _assert_every_variant_builds(_rec)
    finally:
        # BOTH halves, or the next test inherits a lie: `restore` puts the
        # table back and `_LOADED` still says the packs are in it, so the very
        # next `packs.load` is a no-op and the pack's rows never return.
        _rec.restore(state)
        _packs.unload_for_test()


def _assert_every_variant_builds(_rec):
    checked = 0
    for operation in _rec.operations():
        for variant in operation.variants:
            if operation.key == "scale":
                pipeline_plan._scaler(variant)
            elif operation.key == "encode":
                pipeline_plan._encoder(variant, 0)
            elif operation.key == "power":
                pipeline_plan._power(variant)
            elif operation.key == "outliers":
                pipeline_plan._outliers(variant)
            else:                                    # a pack added an operation
                pytest.fail(
                    f"{operation.key} is in the recipe table (from "
                    f"{operation.origin}) and this module has no builder for "
                    f"it, so choosing it would be a decision the fit silently "
                    f"drops")
            assert pipeline_plan._recipe_sentence(
                operation.key, variant, "a model", 3) is not None
            checked += 1
    assert checked >= 14, checked


def test_every_stateful_transform_in_the_catalogue_has_a_fitted_form():
    """The same completeness question for the Features step. `features.py`
    marks six transforms `STATEFUL` — *recorded now, fitted in-fold* — and
    until this loop none of them had an in-fold."""
    from turbotab import features as _feat

    deferred = _feat.deferred_keys()
    assert len(deferred) >= 6, deferred
    for key in deferred:
        made = pipeline_plan._deferred_transformer(
            {"key": key, "params": {"n_bins": 3, "n_components": 2}}, 0)
        assert hasattr(made, "fit"), key
