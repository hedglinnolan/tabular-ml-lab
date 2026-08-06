"""`AUDIT-028` — the imputation record says which rows fitted it, and means it.

## The defect

Every deferring missingness strategy's recorded sentence ended *"within each
training fold"*, and the machine-readable half beside it read `fit_on:
"training folds only"`. **This door has no folds.** `turbotab/training.py` calls
`pipe.fit(X_train, y_train)` once per model over the training partition, and
nothing under `turbotab/` imports `KFold`, `cross_val_score` or
`cross_validate` at all — a fact this file re-derives rather than quotes, in
`test_no_fold_machinery_exists_in_this_door_at_all`.

So a decision the user made in Preprocess was recorded — and exported into the
manuscript's Missing Data section — asserting the resampled design §A5.5 calls
acceptable, over the single split §A5.5 calls *"the weakest option ...
discouraged at typical clinical sample sizes"*. The run note underneath it said
the same thing one level up: *"Every statistic in it is fitted inside the
training folds."*

**The sentence was corrected, not deleted.** The guarantee it exists to give —
no held-out row informs a fitted statistic — is true, and is what it now says.
`GUIDED-104`'s adjudication set the standard: *"selection.declare exists
precisely so a door that fits once can SAY train_rows instead of implying the
stronger claim."* `missingness` now has the same two-value scope, from the same
vocabulary, checked against `selection`'s in
`test_the_two_scope_vocabularies_are_one`.

## Why this drives the fit instead of reading the sentence

Trap #3b. The name says *fits*, so an assertion here observes the fit: the
`SimpleImputer` that lands in the fitted `ColumnTransformer` must carry the
median of the **training rows**, which on both covered shapes differs from the
median of the whole column. Reading the prose back would leave the sentence
unchecked against the thing it describes, which is the defect one layer up.

## Fixture shapes

`GUIDED-097`. `SHAPES` covers a 0/1 numeric target and a three-level string
target. `SHAPES_NOT_COVERED` names the rest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, missingness as M, pipeline_plan, training  # noqa: E402
from turbotab import selection as _sel                               # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(fixture, target, task, model, column-to-impute)`.
SHAPES = {
    "binary_numeric": ("metabolomics_untargeted.csv", "responder",
                       "classification", "histgb_clf", "bmi"),
    "multiclass_string": ("multiclass_stage.csv", "disease_stage",
                          "classification", "histgb_clf", "bmi"),
    "continuous": ("multiclass_stage.csv", "crp",
                   "regression", "histgb_reg", "bmi"),
}

#: Named rather than left silent.
SHAPES_NOT_COVERED = {
    "binary_string": (
        "`multiclass_stage.csv` with `sex` as the target drives this path "
        "cleanly and was run by hand while writing this file. It is not "
        "parametrized here because it exercises the same composer as "
        "`multiclass_string` with a strictly smaller level count — the scope "
        "clause does not read the target at all."),
    "survival_or_multi_output": (
        "No fixture carries a time-to-event or multi-output target, and the "
        "Guided door does not offer one, so there is no shape to drive."),
    "categorical_branch_deferred_fill": (
        "`impute_mode` on a categorical column defers and takes the same "
        "scope clause; it is unit-tested in "
        "`test_every_deferring_strategy_takes_its_scope_from_the_one_table` "
        "but is not driven end-to-end through a fit here."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client, shape):
    fixture, target, task, model, column = SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])
        return r

    decide("set_target", column=target)
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    return pid, api.STORE.get(pid), decide


def _train_partition(project):
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    features = training._feature_frame(
        table, target, (project.grain or {}).get("group_col"))
    X_train = features[has_y & ~is_test]
    return features, X_train, table.loc[X_train.index, target]


def _fitted_median_for(project, model_key, task, column):
    """The statistic the fitted imputer actually carries for `column`."""
    from ml.model_registry import get_registry

    features, X_train, y_train = _train_partition(project)
    plan = pipeline_plan.compose(project, model_key, features, seed=42)
    pipe = plan.build(get_registry()[model_key].factory(task, 42))
    pipe.fit(X_train, y_train)

    prep = pipe.named_steps["prep"]
    for _name, trans, cols in prep.transformers_:
        cols = list(cols)
        if column not in cols:
            continue
        steps = (list(trans.named_steps.values())
                 if hasattr(trans, "named_steps") else [trans])
        for step in steps:
            if isinstance(step, SimpleImputer):
                return float(step.statistics_[cols.index(column)]), X_train
    raise AssertionError(
        f"no fitted SimpleImputer covers {column!r}; this test is about the "
        f"wrong object")


# ── 1 · the hardest case: the fit itself, not the sentence about it ──────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_declared_median_is_fitted_over_the_training_rows_the_record_names(
        client, shape):
    """**The consequence the corrected sentence asserts, observed.**

    The record now says the median is computed *"once over the training rows
    (held-out rows excluded)"*. That is two claims — ONCE, and TRAINING ROWS —
    and the second is the checkable one here: the fitted `SimpleImputer` must
    carry the median of the training partition and not the median of the whole
    column. The two differ on every covered shape, and the test refuses to run
    vacuously if a fixture ever makes them coincide.
    """
    fixture, target, task, model, column = SHAPES[shape]
    pid, project, decide = _sealed(client, shape)
    blanks = int(project.df[column].isna().sum())
    assert blanks > 0, f"{column} has no blanks, so this drive proves nothing"

    decide("route_missingness", column=column,
           mechanism=M.NOT_SURE, strategy=M.IMPUTE_MEDIAN)

    fitted, X_train = _fitted_median_for(project, model, task, column)
    features, _, _ = _train_partition(project)
    train_median = float(X_train[column].median())
    whole_median = float(features[column].median())

    assert train_median != whole_median, (
        "the training-rows median equals the whole-column median on this "
        "fixture, so this probe cannot tell the two scopes apart and is "
        "vacuous — change the fixture rather than the assertion")
    assert fitted == pytest.approx(train_median), (
        f"the imputer was fitted over something other than the training rows "
        f"the record names: fitted={fitted}, train={train_median}, "
        f"whole-column={whole_median}")


# ── 2 · the record, prose and machine-readable, saying the same true thing ───

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_recorded_sentence_claims_the_scope_this_door_actually_fits(
        client, shape):
    """The sentence a user answered into, and the manuscript quotes verbatim.

    `pipeline_plan` asserts `Step.sentence is declaration['sentence']` by
    identity, so this string IS the manuscript's Missing Data line. It must not
    claim a fold structure, and — the half that keeps this from being a
    deletion — it must still state where the statistic came from.
    """
    fixture, target, task, model, column = SHAPES[shape]
    pid, project, decide = _sealed(client, shape)
    decide("route_missingness", column=column,
           mechanism=M.NOT_SURE, strategy=M.IMPUTE_MEDIAN)

    record = [d for d in project.missingness if d["column"] == column][0]
    text = record["sentence"]

    assert "training fold" not in text, (
        f"the record claims a fold structure this door does not have: {text!r}")
    assert "once over the training rows (held-out rows excluded)" in text, (
        f"the fold claim was removed without the true one replacing it — the "
        f"shelf is never shortened: {text!r}")

    # Trap 7: the structured payload is what everything downstream reads, so it
    # must not be lossier — or falser — than the prose beside it.
    assert record["fit_scope"] == M.TRAIN_ROWS
    assert record["fit_on"] == "training rows only"
    assert record["defers"] is True, (
        "a median fill is stateful; if this flips, this test is about the "
        "wrong strategy")


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_run_note_over_the_whole_plan_makes_the_same_claim_as_the_record(
        client, shape):
    """One level up, and it was wrong in the same way.

    `training.train` appends a run-level note summarizing the composed plan. It
    read *"Every statistic in it is fitted inside the training folds."* over a
    single `pipe.fit`. The note and the per-column record are two objects with
    different lifetimes describing one event; when they disagree the archive
    keeps whichever the reader happened to open.
    """
    fixture, target, task, model, column = SHAPES[shape]
    pid, project, decide = _sealed(client, shape)
    decide("route_missingness", column=column,
           mechanism=M.NOT_SURE, strategy=M.IMPUTE_MEDIAN)

    run = training.train(project, [model])
    assert run.results and run.results[0].metrics, run.results[0].error

    note = run.notes[0]
    assert "training folds" not in note, (
        f"the run note claims folds this door does not run: {note!r}")
    assert "fitted once over the" in note and "training rows" in note, note
    assert "held-out rows inform none of them" in note, (
        "the guarantee the note exists to give was dropped rather than "
        "restated")
    # The number in it is the partition it describes, not a constant.
    _, X_train, _ = _train_partition(project)
    assert f"{len(X_train):,} training rows" in note, note


# ── 3 · the premise, re-derived rather than quoted ───────────────────────────

def test_no_fold_machinery_exists_in_this_door_at_all():
    """The premise every assertion above rests on, checked rather than assumed.

    Trap #8: a prose claim about the code decays. If a future loop gives the
    Guided door a real resampling loop, the corrected sentences become the
    understatement and this test is where that gets noticed — it fails, and the
    scope parameter is already there to carry the stronger claim honestly.
    """
    here = Path(__file__).resolve().parent
    offenders = []
    # THE POSITIVE CONTROL, and `GUIDED-045` is why it is here rather than
    # assumed: an all-absence assertion passes hardest on an empty glob. The
    # sweep below would report a clean door for a moved package, a renamed
    # suffix, or a `here` that resolved somewhere with no Python in it.
    scanned = [p for p in sorted(here.glob("*.py"))
               if not p.name.startswith("test_")]
    assert len(scanned) >= 20, (
        f"only {len(scanned)} non-test modules found under {here} — the sweep "
        f"below is an absence claim and it would pass on nothing at all")
    assert any(p.name == "training.py" for p in scanned), (
        "`training.py` is not in the swept set, and it is the module whose "
        "docstring this test is checking the code against")
    def _hits(src: str) -> list:
        found = []
        for token in ("KFold", "cross_val_score", "cross_validate",
                      "StratifiedKFold"):
            # `actions.py` carries a Classic-door LABEL for a CV control; it
            # runs nothing. Match on an import or a call, not on prose.
            if f"import {token}" in src or f"{token}(" in src:
                found.append(token)
        return found

    # AND THE DETECTOR ITSELF, on planted source. Without this, an empty
    # `offenders` list is equally consistent with a clean door and a matcher
    # that stopped matching.
    assert _hits("from sklearn.model_selection import KFold\n"), (
        "the matcher does not recognize an import of the thing it looks for")
    assert _hits("splits = StratifiedKFold(n_splits=5)\n"), (
        "the matcher does not recognize a call to the thing it looks for")
    assert not _hits('LABEL = "cross-validate this model"\n'), (
        "the matcher fires on prose, so its silence would mean nothing either")

    for path in scanned:
        src = path.read_text(encoding="utf-8", errors="ignore")
        offenders += [f"{path.name}: {token}" for token in _hits(src)]
    assert not offenders, (
        f"this door now has fold machinery, so the scope the records claim "
        f"needs re-deciding rather than re-asserting: {offenders}")


def test_the_two_scope_vocabularies_are_one():
    """`selection` and `missingness` record the same fact in the same words.

    Two spellings of `train_rows` is two vocabularies that will drift, and a
    parity check across the doors would then have to parse prose to compare
    them.
    """
    assert M.TRAIN_ROWS == _sel.TRAIN_ROWS
    assert M.TRAIN_FOLDS == _sel.TRAIN_FOLDS
    assert set(M.FIT_SCOPES) == {_sel.TRAIN_ROWS, _sel.TRAIN_FOLDS}


# ── 4 · the shelf, and the row-local half ────────────────────────────────────

def test_every_deferring_strategy_takes_its_scope_from_the_one_table():
    """No strategy composes its own timing clause, in either scope.

    The row-local three have no scope because they compute no statistic —
    `leave`, `explicit_category` and `indicator` say what they do and stop, and
    passing a scope must not add a clause to them. That is clause §06's
    distinction, and `ROW_LOCAL_STRATEGIES` is where it lives.
    """
    for branch, keys in M.STRATEGIES_BY_BRANCH.items():
        for key in keys:
            defers = M.strategy(key)["defers"]
            rows = M.sentence_for("x", branch, key, scope=M.TRAIN_ROWS)
            folds = M.sentence_for("x", branch, key, scope=M.TRAIN_FOLDS)
            if not defers:
                assert key in M.ROW_LOCAL_STRATEGIES
                assert rows == folds, (
                    f"{key!r} is row-local and its sentence still moved with "
                    f"the scope: {rows!r} vs {folds!r}")
                assert "training fold" not in rows and "training rows" not in rows
            else:
                assert rows != folds, (
                    f"{key!r} defers and its sentence ignored the scope, so "
                    f"the scope is decorative for it: {rows!r}")
                assert "once over the training rows" in rows, rows
                assert "within each training fold" in folds, folds


def test_the_shelf_still_offers_every_strategy_it_did():
    """The correction changed what the record CLAIMS, never what is offered.

    `PRODUCT_VISION.md`'s rule, asserted rather than promised: a fix to a false
    sentence that quietly drops an option is the failure mode this row's
    adjudication checks for by name.
    """
    for branch, keys in M.STRATEGIES_BY_BRANCH.items():
        for key in keys:
            spec = M.strategy(key)
            assert spec["label"] and spec["because"]
            assert M.sentence_for("x", branch, key).strip()


def test_a_scope_the_door_cannot_fit_is_refused_rather_than_defaulted():
    """A typo must not become a silent claim about the analysis.

    The same refusal `selection.declare` makes, and for the same reason: the
    third option a caller reaches for is *the whole table*, which is exactly
    the leak these records exist to rule out.
    """
    with pytest.raises(M.MissingnessRefusal) as err:
        M.sentence_for("x", "numeric", M.IMPUTE_MEDIAN, scope="whole_table")
    assert "train_rows" in str(err.value) and "train_folds" in str(err.value)
    assert "whole table" in str(err.value)

    with pytest.raises(M.MissingnessRefusal):
        M.declare("x", "numeric", M.NOT_SURE, M.IMPUTE_MEDIAN,
                  scope="everything")
