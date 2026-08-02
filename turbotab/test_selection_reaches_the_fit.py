"""`GUIDED-095`'s last member, and `DRIVE-011` with it.

## The class, and why selection was inside it rather than beside it

*Every decision recorded to be fitted inside the training fold is read by
nothing that fits.* L35 closed it for the missingness declarations, the recipe
variants and the deferred transforms. **Feature selection was still open**, and
it is the sharpest member: `selection.declare` takes `scope=TRAIN_FOLDS`,
refuses any third option, and says in its own refusal that *there is no option
that fits on the whole table* — a fold-local claim that nothing delivered.

`selection.py`'s docstring calls this the sharpest case in the project for a
reason that survives the fix: no held-out value is copied anywhere, and the
selected SET still encodes test signal, because the identity of the chosen
columns was decided partly by rows the analysis promised not to look at.

## What this asserts, hardest first

1. **The selector is inside the estimator**, so the set is chosen from the rows
   the estimator was fitted on and from no others — probed by moving the
   held-out rows and watching the selected set not move.
2. **The recorded candidate POOL is the pool.** Found by driving the first
   working version: a user who nominated six numeric columns got a top-3 chosen
   from 244 one-hot columns, because the selector ranked the shaped matrix
   rather than the record. That is `GUIDED-095` arriving inside its own fix.
3. **The stronger scope is never claimed.** This door fits each model once, so
   there is a single fold and the selector saw the training rows one time. That
   is `train_rows`, and the difference from the recorded `train_folds` is a
   stated divergence rather than a sentence nobody checked.
4. **The run says how many survived and names them**, with a denominator.
5. **The page can reach it** — `DRIVE-011`.

## Fixture shapes

`GUIDED-097`: `SHAPES` below runs the load-bearing claims against a continuous
and a binary-string target. Shapes not covered are named in
`SHAPES_NOT_COVERED`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, pageharness as H, pipeline_plan, training  # noqa: E402
from turbotab import selection as _sel                              # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Two target shapes for every load-bearing claim here.
SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c", "regression",
                   ("ridge", "histgb_reg")),
    "binary_string": ("clinic_visits.csv", "outcome", "classification",
                      ("logreg", "histgb_clf")),
}

#: And the ones this file does NOT cover, named rather than left silent.
SHAPES_NOT_COVERED = {
    "binary_numeric": (
        "`leaky_sepsis.csv` has a 0/1 target and no missing values, so its "
        "pipeline has no imputer and its selection claims would exercise a "
        "strictly simpler shape state than the two covered here."),
    "multiclass": (
        "No fixture has a three-or-more-level outcome. `mutual_info_classif` "
        "and `f_classif` both accept one, and neither branch is driven here."),
    "wide": (
        "`metabolomics_untargeted.csv` has 396 numeric columns and would be "
        "the interesting case for `stability` and `rfe` timing. The methods "
        "are unit-tested against it below; the end-to-end drive is not, "
        "because a 396-column RFE is minutes rather than seconds."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client, shape, *, fraction=0.25):
    fixture, target, task, models = SHAPES[shape]
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
    decide("seal", fraction=fraction)
    return pid, api.STORE.get(pid), decide, models


def _candidates(client, pid):
    return client.get(f"/project/{pid}/features").json()["numeric_columns"]


def _fit(project, model_key, task, seed=42):
    from ml.model_registry import get_registry

    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    features = training._feature_frame(table, target,
                                       (project.grain or {}).get("group_col"))
    X_train = features[has_y & ~is_test]
    y_train = table.loc[X_train.index, target]
    plan = pipeline_plan.compose(project, model_key, features, seed=seed)
    pipe = plan.build(get_registry()[model_key].factory(task, seed))
    pipe.fit(X_train, y_train)
    return plan, pipe


# ── 1 · the hardest case: the set must not know the held-out rows ────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_selected_set_does_not_move_when_the_held_out_rows_do(client, shape):
    """**The property the whole module is about.**

    The selected SET encodes test signal even though no held-out value is
    copied anywhere — that is `selection.py`'s own framing and it is why the
    selector goes inside the estimator. Probed the way the seal is probed:
    replace every held-out value with something no training row could produce
    and the chosen columns must be identical.
    """
    fixture, target, task, models = SHAPES[shape]
    pid, project, decide, _ = _sealed(client, shape)
    candidates = _candidates(client, pid)
    assert len(candidates) >= 4, candidates
    decide("set_selection", method="mutual_info", n_features=3,
           candidates=candidates)

    before = _fit(project, models[0], task)[1].named_steps["select"].kept_

    sealed = set(project.lockbox["labels"])
    poisoned = project.df.copy()
    mask = poisoned.index.isin(list(sealed))
    assert mask.sum() > 0                                            # control
    # VALUES ONLY, NEVER THE MISSINGNESS PATTERN. `pipeline_plan`'s docstring
    # states — and defends — that whether a column has a blank is read from the
    # whole frame, because a held-out row with a blank still has to be scored.
    # A poison that filled the sealed rows' blanks would change the pipeline's
    # STRUCTURE and this probe would report a leak that is really a different
    # plan. Found the first time this ran.
    touched = 0
    for column in candidates:
        cells = mask & poisoned[column].notna().to_numpy()
        poisoned.loc[cells, column] = 1_000_000.0
        touched += int(cells.sum())
    assert touched > 0, "the poison changed nothing, so the probe is vacuous"
    project.df = poisoned

    after = _fit(project, models[0], task)[1].named_steps["select"].kept_
    assert list(before) == list(after), (
        f"the selected set moved when only the held-out rows changed, so the "
        f"columns this analysis kept were chosen partly by rows it promised "
        f"not to look at: {list(before)} -> {list(after)}")


def test_the_probe_can_fail(client):
    """The positive control, and this file is worthless without it.

    The same comparison against a selector fitted on the WHOLE table must see
    the difference. A probe that cannot detect the leak it is aimed at proves
    nothing about the fit that passes it.
    """
    from sklearn.feature_selection import SelectKBest, f_regression

    pid, project, decide, _ = _sealed(client, "continuous")
    # An all-blank column has no median to fill with, so it is excluded here —
    # this control is about whether the COMPARISON can see a leak, and a column
    # sklearn refuses would make it fail for an unrelated reason.
    candidates = [c for c in _candidates(client, pid)
                  if project.df[c].notna().any()]
    frame = project.df[candidates].fillna(project.df[candidates].median())
    y = project.df["hba1c"]
    keep = frame.notna().all(axis=1) & y.notna()

    def chosen(table):
        fitted = SelectKBest(f_regression, k=3).fit(table[keep], y[keep])
        return [c for c, k in zip(candidates, fitted.get_support()) if k]

    before = chosen(frame)
    # A column the sealed rows make LOOK predictive. Noise would not do: the
    # question is whether the comparison can see a leak that changes the
    # answer, and only a held-out pattern that reorders the ranking tests that.
    sealed = list(project.lockbox["labels"])
    outsider = [c for c in candidates if c not in before][-1]
    poisoned = frame.copy()
    rows = poisoned.index.isin(sealed)
    poisoned.loc[rows, outsider] = y[rows].to_numpy(dtype=float) * 1000.0
    assert chosen(poisoned) != before, (
        "a selector fitted on the whole table chose the same columns after the "
        "held-out rows were poisoned, so this fixture cannot tell a leaking "
        "selector from a sealed one and the claim above would pass either way")


# ── 2 · the recorded pool is the pool ────────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_selector_ranks_the_recorded_candidates_and_nothing_else(
        client, shape):
    """**Found by driving the first working version of this part.**

    A user nominated six numeric columns and the selector's top three came from
    244 one-hot columns, because it ranked the shaped matrix rather than the
    record. Columns outside the pool pass THROUGH — `candidates` names what was
    offered to the method, and dropping everything nobody nominated would be
    shortening the feature set in the user's name.
    """
    fixture, target, task, models = SHAPES[shape]
    pid, project, decide, _ = _sealed(client, shape)
    numeric = _candidates(client, pid)
    # A STRICT SUBSET, deliberately. Nominating every numeric column would make
    # this claim unable to tell "the pool is the recorded candidates" from "the
    # pool is every numeric column" — which is exactly what the revert probe
    # found it could not tell, on the first version of this test.
    candidates = numeric[:3]
    assert len(candidates) < len(numeric)                            # control
    decide("set_selection", method="mutual_info", n_features=2,
           candidates=candidates)

    plan, pipe = _fit(project, models[0], task)
    selector = pipe.named_steps["select"]
    assert set(selector.pool_) == set(candidates), (
        f"the pool is not the recorded candidates — the selector ranked "
        f"{sorted(selector.pool_)} and the record nominated {candidates}")
    assert len(selector.kept_) == 2, selector.kept_
    assert set(selector.kept_) <= set(candidates)
    unranked = set(numeric) - set(candidates)
    assert unranked <= set(selector.passthrough_), (
        "a numeric column nobody nominated was ranked anyway")
    assert not (set(selector.passthrough_) & set(candidates))


def test_a_candidate_the_shape_stage_renames_is_a_stated_divergence(client):
    """**Return nothing rather than a wrong value**, applied to a name.

    A categorical candidate does not reach the selector under its own name —
    the encoder expanded it — so there is no single output column to rank. The
    plan says which candidates were passed through unranked rather than
    guessing which of `weight_96 kg` … `weight_53 kg` was meant.
    """
    pid, project, decide, models = _sealed(client, "continuous")
    numeric = _candidates(client, pid)
    project.selection_spec = _sel.declare(
        "mutual_info", "hba1c", list(numeric) + ["site"], n_features=2)

    plan, pipe = _fit(project, models[0], "regression")
    diverged = [d for d in plan.divergences if d.source == "selection"
                and "site" in d.subject]
    assert diverged, (
        "a candidate the shape stage renames was silently dropped from the "
        "pool with nothing said")
    assert "site" in diverged[0].fitted_sentence
    assert "site" not in pipe.named_steps["select"].pool_


# ── 3 · the stronger scope is never claimed ──────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_record_states_the_scope_this_door_actually_fits(client, shape):
    """**`GUIDED-104`. The record says what happens, and nothing retracts it.**

    `selection.declare`'s docstring says scope *"is explicit and has no default
    that hides the weaker option"*, and that `TRAIN_ROWS` exists so a door that
    inherits Classic's behavior can SAY so rather than imply the stronger
    claim. The app took the `TRAIN_FOLDS` default, fitted train-rows-once, and
    repaired the difference in a run note — so the record asserted the stronger
    claim and a note in a different object, with a different lifetime, retracted
    it. The archive, the methods sentence and any parity check read the record.

    This door fits each model once, so `train_rows` is what happens and is what
    is recorded. No divergence is needed, because there is nothing to diverge
    from.
    """
    fixture, target, task, models = SHAPES[shape]
    pid, project, decide, _ = _sealed(client, shape)
    decide("set_selection", method="univariate", n_features=2,
           candidates=_candidates(client, pid))
    assert project.selection_spec["scope"] == _sel.TRAIN_ROWS, (
        "the record claims a scope this door does not fit")
    assert "once over the training rows" in project.selection_spec["sentence"]
    assert project.selection_spec["fit_on"] == "training rows only"

    run = training.train(project, [models[0]])
    result = run.results[0]
    assert result.metrics, result.error
    assert not [d for d in result.plan["divergences"]
                if d["source"] == "selection"
                and d["requested"] == _sel.TRAIN_FOLDS], (
        "the record and the fit agree and the run still retracts something")

    step = [s for s in result.plan["steps"] if s["source"] == "selection"][0]
    assert step["params"]["scope_recorded"] == _sel.TRAIN_ROWS
    assert step["params"]["scope_fitted"] == _sel.TRAIN_ROWS


def test_a_caller_that_asks_for_fold_local_is_still_told_it_did_not_get_it(
        client):
    """**The divergence machinery stays**, and this is why.

    `GUIDED-104` changes the DEFAULT, not the capability. A client that asks
    for `train_folds` explicitly — and the day `GUIDED-103`'s resampling policy
    lands, the app itself — must still be told the door fitted once. Deleting
    the divergence with the default would leave the stronger claim
    unchallenged the moment anyone asks for it.
    """
    pid, project, decide, models = _sealed(client, "continuous")
    decide("set_selection", method="univariate", n_features=2,
           candidates=_candidates(client, pid), scope=_sel.TRAIN_FOLDS)
    assert project.selection_spec["scope"] == _sel.TRAIN_FOLDS

    run = training.train(project, [models[0]])
    scoped = [d for d in run.results[0].plan["divergences"]
              if d["source"] == "selection"]
    assert scoped, (
        "a caller asked for fold-local selection, got a single fit, and was "
        "told nothing")
    assert scoped[0]["applied"] == _sel.TRAIN_ROWS
    assert any(scoped[0]["fitted_sentence"] == note for note in run.notes)


# ── 4 · the run says what survived ───────────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_run_names_the_surviving_features_with_a_denominator(
        client, shape):
    """A count with no denominator is a number a reader cannot judge, and a
    denominator that counts the encoder's output would describe the encoding as
    if it were the user's choice."""
    fixture, target, task, models = SHAPES[shape]
    pid, project, decide, _ = _sealed(client, shape)
    candidates = _candidates(client, pid)
    decide("set_selection", method="mutual_info", n_features=3,
           candidates=candidates)

    run = training.train(project, list(models))
    for result in run.results:
        assert result.selected_features is not None, result.key
        assert len(result.selected_features) == 3
        assert result.n_candidates == len(candidates), (
            f"{result.key} reports {result.n_candidates} candidates and the "
            f"user nominated {len(candidates)}")
        assert result.n_passthrough and result.n_passthrough > 0
    note = [n for n in run.notes if "feature selection kept" in n]
    assert note and "of {} candidate".format(len(candidates)) in note[0], note


def test_no_selection_recorded_reports_nothing_rather_than_an_empty_set(client):
    """`None` and `[]` are different states and must not look alike: one is *no
    selection was recorded*, the other is *the selector kept nothing*."""
    pid, project, decide, models = _sealed(client, "continuous")
    assert project.selection_spec is None                            # control
    run = training.train(project, [models[0]])
    assert run.results[0].selected_features is None
    assert run.results[0].n_candidates is None
    assert not any("feature selection kept" in n for n in run.notes)


# ── 5 · every method the record offers has a fitted form ─────────────────────

@pytest.mark.parametrize("method", sorted(_sel.METHODS), ids=sorted(_sel.METHODS))
def test_every_recorded_method_fits_and_selects(client, method):
    """The completeness half. A method the record accepts and the pipeline
    cannot perform is a decision the fit silently drops — which is the whole
    class, arriving through the method list."""
    pid, project, decide, models = _sealed(client, "continuous")
    candidates = _candidates(client, pid)
    project.selection_spec = _sel.declare(
        method, "hba1c", candidates,
        n_features=3 if method in ("mutual_info", "univariate", "rfe") else None)

    plan, pipe = _fit(project, models[0], "regression")
    selector = pipe.named_steps["select"]
    assert selector.kept_, f"{method} kept nothing at all"
    assert set(selector.kept_) <= set(candidates)


def test_a_method_with_no_fitted_form_is_refused_rather_than_ignored(client):
    """The property that keeps this from becoming the defect it closed."""
    pid, project, decide, models = _sealed(client, "continuous")
    project.selection_spec = dict(
        _sel.declare("mutual_info", "hba1c", _candidates(client, pid),
                     n_features=2),
        method="a_method_nobody_built")
    features = training._feature_frame(project.working_table, "hba1c", None)
    with pytest.raises(pipeline_plan.PlanRefusal) as caught:
        pipeline_plan.compose(project, models[0], features)
    assert "a_method_nobody_built" in str(caught.value)


def test_the_stability_selector_returns_everything_rather_than_nothing(client):
    """**Return nothing rather than a wrong value**, in its other form: where
    no resample could be scored the selector has no evidence, and inventing a
    set from zero draws would be the app asserting a selection it did not
    make."""
    selector = pipeline_plan.StabilitySelector(n_features=2, classify=True,
                                               n_resamples=5, random_state=0)
    frame = pd.DataFrame({"a": range(20), "b": range(20), "c": range(20)})
    # One class only, so every classification resample declines.
    selector.fit(frame, np.zeros(20))
    assert selector.n_resamples_scored_ == 0
    assert selector.get_support().all(), (
        "a selector with no evidence narrowed the feature set anyway")


# ── 6 · DRIVE-011 · the page can reach it ────────────────────────────────────

@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_page_records_a_selection_the_record_accepts(client):
    """`DRIVE-011`. The picker exists since `L33`; what had never been checked
    is whether the body it composes is one the record takes.

    **It was not.** Driven at L36: the page offered *keep 10* on a six-column
    table and `selection.declare` answered `400 — Asked for 10 of 6
    candidates`, so a user's first press on this control was a refusal. The
    counts come from the served candidate pool now, which is the same
    correction `GUIDED-084` made one surface over.
    """
    pid, project, decide, models = _sealed(client, "continuous")
    served = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": served,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/features":
            client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/preprocess":
            client.get(f"/project/{pid}/preprocess").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/models": client.get(f"/project/{pid}/models").json(),
        f"POST /project/{pid}/decision": served,
    }
    out = H.run(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(24).then(function(){
          __harness.drainRaf();
          var pick = __harness.target({'data-sel': 'method'}, []);
          pick.value = 'mutual_info';
          __harness.dispatch('change', pick);
          return settle(6);
        }).then(function(){
          __harness.dispatch('click',
            __harness.target({'data-sel-set': '1'}, ['answer', 'primary']));
          return settle(8);
        }).then(function(){
          __emit({posts: __harness.posts(),
                  build: __harness.html('selBuild')});
        });
        """, routes=routes, search=f"?project={pid}")

    posted = [p["body"] if isinstance(p["body"], dict) else json.loads(p["body"])
              for p in out["posts"]]
    assert posted, "the press produced no request at all"
    body = posted[-1]
    assert body["kind"] == "set_selection"
    assert body["payload"]["method"] == "mutual_info"
    candidates = _candidates(client, pid)
    assert body["payload"]["n_features"] <= len(candidates), (
        f"the page proposed keeping {body['payload']['n_features']} of "
        f"{len(candidates)} candidates, which the record refuses")

    accepted = client.post(f"/project/{pid}/decision", json=body)
    assert accepted.status_code == 200, (
        "the body the page composes is refused by the record", accepted.text[:250])
    assert accepted.json()["selection_spec"]["method"] == "mutual_info"
